import logging
import math
import os

import torch
from torch.cuda.amp import autocast
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


def _collate(batch: list[dict]) -> dict:
    return {"text": [r["text"] for r in batch], "prompt": [r["prompt"] for r in batch]}


def _prepare_batch(batch: dict, tokenizer, device: torch.device, max_length: int):
    enc = tokenizer(
        batch["text"],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    labels = enc["input_ids"].clone()
    for i, prompt in enumerate(batch["prompt"]):
        text_ids = tokenizer(batch["text"][i], add_special_tokens=True)["input_ids"]
        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        prompt_len = len(prompt_ids)
        if text_ids[:prompt_len] == prompt_ids:
            labels[i, :prompt_len] = -100
        else:
            prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            labels[i, :prompt_len] = -100
    labels[enc["attention_mask"] == 0] = -100
    return {k: v.to(device) for k, v in enc.items()}, labels.to(device)


def train_sft() -> str:
    os.makedirs(settings.sft_output_dir, exist_ok=True)
    ctx = dist_context()
    max_length = settings.max_prompt_len + settings.max_target_len
    use_bf16 = settings.sft_bf16 and torch.cuda.is_available()

    if ctx.is_main:
        logger.info(
            "Loading dataset %s (max %d samples)", settings.dataset_id, settings.max_train_samples
        )
    dataset = prepare_sft_dataset()
    dataloader = distributed_dataloader(
        dataset,
        settings.sft_per_device_batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=_collate,
    )

    tokenizer = load_tokenizer()
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
            batch = _collate(raw_batch)
            inputs, labels = _prepare_batch(batch, tokenizer, ctx.device, max_length)

            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                loss = model(**inputs, labels=labels).loss / settings.sft_grad_accum

            loss.backward()

            if (step + 1) % settings.sft_grad_accum == 0:
                clip_grad_norm(model, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if ctx.is_main and global_step % settings.sft_logging_steps == 0:
                    progress.set_postfix(loss=float(loss.detach()) * settings.sft_grad_accum, step=global_step)

    save_model(model, settings.sft_output_dir, tokenizer)
    if ctx.is_main:
        logger.info("SFT complete.")

    if settings.hf_push_repo_id and ctx.is_main:
        logger.info("Pushing SFT model to Hub: %s", settings.hf_push_repo_id)
        push_to_hub(model, settings.hf_push_repo_id, tokenizer, settings.hf_token)

    return settings.sft_output_dir
