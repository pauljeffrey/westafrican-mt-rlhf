import logging
import os

import torch
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from src.config import settings
from src.data import prepare_rl_dataset
from src.distributed import (
    apply_tensor_parallel,
    broadcast_tensor,
    dist_context,
    distributed_dataloader,
    is_distributed,
    push_to_hub,
    save_fsdp_model,
    summon_full_params,
    wrap_fsdp,
)
from src.model_utils import load_causal_lm, load_tokenizer
from src.prompts import extract_translation
from src.reward import AfricometReward

logger = logging.getLogger(__name__)


def _sequence_logprobs(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits[:, :-1], dim=-1)
    target = labels[:, 1:]
    mask = mask[:, 1:]
    gathered = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return (gathered * mask).sum(dim=-1)


def _forward_logprobs(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, gen_mask: torch.Tensor):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return _sequence_logprobs(outputs.logits, input_ids, gen_mask)


def train_rlhf() -> str:
    os.makedirs(settings.rl_output_dir, exist_ok=True)
    ctx = dist_context()

    policy_path = settings.sft_output_dir
    if not os.path.isdir(policy_path):
        raise FileNotFoundError(
            f"SFT checkpoint not found at {policy_path}. Run SFT first: python scripts/run_sft.py"
        )

    if ctx.is_main:
        logger.info("Loading RL dataset (max %d samples)", settings.max_rl_samples)
    dataset = prepare_rl_dataset()
    dataloader = distributed_dataloader(
        dataset,
        settings.rl_per_device_batch_size,
        shuffle=True,
        drop_last=True,
    )

    tokenizer = load_tokenizer(policy_path)
    policy = load_causal_lm(policy_path)
    reference = load_causal_lm(policy_path, trainable=False)

    if settings.fsdp_enabled and is_distributed():
        if ctx.tp_size > 1:
            policy = apply_tensor_parallel(policy)
            reference = apply_tensor_parallel(reference)
        policy = wrap_fsdp(
            policy,
            sharding=settings.fsdp_sharding,
            use_bf16=settings.rl_bf16,
            cpu_offload=settings.fsdp_cpu_offload,
            use_orig_params=settings.use_lora,
        )
        reference = wrap_fsdp(
            reference,
            sharding=settings.fsdp_sharding,
            use_bf16=settings.rl_bf16,
            cpu_offload=settings.fsdp_cpu_offload,
            use_orig_params=False,
        )

    reward_model = AfricometReward() if (ctx.is_main or not ctx.enabled) else None
    device = ctx.device

    optimizer = torch.optim.AdamW(
        (p for p in policy.parameters() if p.requires_grad),
        lr=settings.rl_learning_rate,
    )

    steps_per_epoch = len(dataloader)
    total_steps = min(settings.rl_max_steps, steps_per_epoch * settings.rl_num_epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    global_step = 0
    policy.train()

    for epoch in range(settings.rl_num_epochs):
        if is_distributed():
            dataloader.sampler.set_epoch(epoch)  # type: ignore[union-attr]

        progress = tqdm(
            dataloader,
            desc=f"RL epoch {epoch + 1}/{settings.rl_num_epochs}",
            disable=not ctx.is_main,
        )
        for batch in progress:
            if global_step >= settings.rl_max_steps:
                break

            queries = batch["query"]
            srcs = batch["src"]
            refs = batch["ref"]

            prompts = tokenizer(
                queries,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=settings.max_prompt_len,
            ).to(device)

            with torch.no_grad(), summon_full_params(policy):
                generated = policy.generate(
                    **prompts,
                    max_new_tokens=settings.max_new_tokens,
                    do_sample=True,
                    temperature=settings.rl_temperature,
                    top_p=settings.rl_top_p,
                    pad_token_id=tokenizer.pad_token_id,
                )

            full_texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
            translations = [
                extract_translation(text, query) for text, query in zip(full_texts, queries, strict=True)
            ]

            if reward_model is not None:
                rewards = reward_model.score_batch(srcs, translations, refs)
            else:
                rewards = [0.0] * len(translations)

            if settings.rl_reward_normalize and reward_model is not None:
                rewards = AfricometReward.normalize(rewards)

            reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
            broadcast_tensor(reward_tensor)

            attn = (generated != tokenizer.pad_token_id).long()
            prompt_lens = prompts["attention_mask"].sum(dim=1)
            gen_mask = torch.zeros_like(attn, dtype=torch.float32)
            for i, plen in enumerate(prompt_lens):
                gen_mask[i, plen:] = attn[i, plen:].float()

            policy_lp = _forward_logprobs(policy, generated, attn, gen_mask)
            with torch.no_grad():
                ref_lp = _forward_logprobs(reference, generated, attn, gen_mask)

            kl = (policy_lp - ref_lp).mean()
            pg_loss = -(reward_tensor * policy_lp).mean()
            loss = (pg_loss + settings.rl_kl_coef * kl) / settings.rl_grad_accum
            loss.backward()

            if (global_step + 1) % settings.rl_grad_accum == 0:
                if isinstance(policy, FSDP):
                    policy.clip_grad_norm_(1.0)
                else:
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            if ctx.is_main:
                progress.set_postfix(
                    reward=float(reward_tensor.mean()),
                    kl=float(kl.detach()),
                    loss=float(loss.detach()) * settings.rl_grad_accum,
                )

        if global_step >= settings.rl_max_steps:
            break

    save_fsdp_model(policy, settings.rl_output_dir, tokenizer)
    if ctx.is_main:
        logger.info("RLHF training complete -> %s", settings.rl_output_dir)

    if settings.hf_push_repo_id and ctx.is_main:
        logger.info("Pushing RLHF model to Hub: %s", settings.hf_push_repo_id)
        push_to_hub(policy, settings.hf_push_repo_id, tokenizer, settings.hf_token)

    return settings.rl_output_dir
