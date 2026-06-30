import logging
import math
import os

import torch
from torch.amp import autocast
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from src.config import settings
from src.data import prepare_sft_dataset
from src.distributed import (
    clip_grad_norm,
    dist_context,
    distributed_dataloader,
    is_distributed,
    push_to_hub,
    save_model,
    wrap_model,
)
from src.model_utils import load_causal_lm, load_tokenizer

logger = logging.getLogger(__name__)


class SFTCollator:
    def __init__(self, pad_token_id) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        input_ids = [row["input_ids"] for row in batch]
        labels = [row["labels"] for row in batch]

        max_len = max(len(x) for x in input_ids)

        batch_input_ids, batch_attention_mask, batch_labels = [], [], []

        for row in zip(input_ids, labels):
            true_len, pad_len = len(row[0]), (max_len - len(row[0]))
            input_id = row[0] + ([self.pad_token_id] * pad_len)
            label = row[1] + [-100] * (max_len - len(row[1]))
            attention_mask = [1] * true_len + [0] * pad_len

            batch_input_ids.append(input_id)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(label)

        return {
            "input_ids": torch.tensor(batch_input_ids),  # (batch, max_len) long
            "attention_mask": torch.tensor(batch_attention_mask),  # (batch, max_len) long
            "labels": torch.tensor(batch_labels),  # (batch, max_len) long
        }


def train_sft() -> str:
    os.makedirs(settings.sft_output_dir, exist_ok=True)
    ctx = dist_context()
    max_length = settings.max_prompt_len + settings.max_target_len
    use_bf16 = settings.sft_bf16 and torch.cuda.is_available()

    if ctx.is_main:
        logger.info(
            "Loading dataset %s (max %d samples)", settings.dataset_id, settings.max_train_samples
        )
    tokenizer = load_tokenizer()
    dataset = prepare_sft_dataset(tokenizer)
    pad_token_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    dataloader = distributed_dataloader(
        dataset,
        settings.sft_per_device_batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=SFTCollator(pad_token_id),
    )

    model = load_causal_lm()
    model = wrap_model(
        model,
        use_bf16=use_bf16,
        sharding=settings.fsdp_sharding,
        cpu_offload=settings.fsdp_cpu_offload,
        use_orig_params=settings.use_lora,
    )

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=settings.sft_lr,
    )

    steps_per_epoch = max(1, math.ceil(len(dataloader) / settings.sft_grad_accum))
    total_steps = max(1, int(steps_per_epoch * settings.sft_epochs))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * settings.sft_warmup_ratio)),
        num_training_steps=total_steps,
    )

    global_step = 0
    model.train()

    if ctx.is_main:
        logger.info(
            "Starting SFT (%s) -> %s | steps=%d",
            ctx.strategy if ctx.enabled else "single-gpu",
            settings.sft_output_dir,
            total_steps,
        )

    for epoch in range(int(settings.sft_epochs)):
        if is_distributed():
            dataloader.sampler.set_epoch(epoch)  # type: ignore[union-attr]

        progress = tqdm(dataloader, desc=f"SFT epoch {epoch + 1}", disable=not ctx.is_main)
        optimizer.zero_grad()

        for step, raw_batch in enumerate(progress):
            batch = {k: v.to(ctx.device) for k, v in raw_batch.items()}
            with autocast(device_type=ctx.device.type, dtype=torch.bfloat16, enabled=use_bf16):
                loss = model(**batch).loss / settings.sft_grad_accum

            loss.backward()

            if (step + 1) % settings.sft_grad_accum == 0:
                clip_grad_norm(model, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if ctx.is_main and global_step % settings.sft_logging_steps == 0:
                    progress.set_postfix(
                        loss=float(loss.detach()) * settings.sft_grad_accum, step=global_step
                    )

    save_model(model, settings.sft_output_dir, tokenizer)
    if ctx.is_main:
        logger.info("SFT complete.")

    if settings.hf_push_repo_id and ctx.is_main:
        logger.info("Pushing SFT model to Hub: %s", settings.hf_push_repo_id)
        push_to_hub(model, settings.hf_push_repo_id, tokenizer, settings.hf_token)

    return settings.sft_output_dir
