import logging
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from src.config import settings
from src.data import prepare_rl_dataset
from src.model_utils import load_causal_lm, load_tokenizer
from src.prompts import extract_translation
from src.reward import AfricometReward

logger = logging.getLogger(__name__)


def _sequence_logprobs(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Sum token log-probs for generated tokens (labels already shifted)."""
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

    policy_path = settings.sft_output_dir
    if not os.path.isdir(policy_path):
        raise FileNotFoundError(
            f"SFT checkpoint not found at {policy_path}. Run SFT first: python scripts/run_sft.py"
        )

    logger.info("Loading RL dataset (max %d samples)", settings.max_rl_samples)
    dataset = prepare_rl_dataset()
    dataloader = DataLoader(
        dataset,
        batch_size=settings.rl_per_device_batch_size,
        shuffle=True,
        drop_last=True,
    )

    tokenizer = load_tokenizer(policy_path)
    policy = load_causal_lm(policy_path)
    reference = load_causal_lm(policy_path, trainable=False)

    reward_model = AfricometReward()
    device = next(policy.parameters()).device

    optimizer = torch.optim.AdamW(
        (p for p in policy.parameters() if p.requires_grad),
        lr=settings.rl_learning_rate,
    )

    total_steps = min(settings.rl_max_steps, len(dataloader) * settings.rl_num_epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    global_step = 0
    policy.train()

    for epoch in range(settings.rl_num_epochs):
        progress = tqdm(dataloader, desc=f"RL epoch {epoch + 1}/{settings.rl_num_epochs}")
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

            with torch.no_grad():
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

            rewards = reward_model.score_batch(srcs, translations, refs)
            if settings.rl_reward_normalize:
                rewards = AfricometReward.normalize(rewards)
            reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)

            attn = (generated != tokenizer.pad_token_id).long()
            prompt_lens = prompts["attention_mask"].sum(dim=1)
            gen_mask = torch.zeros_like(attn, dtype=torch.float32)
            for i, plen in enumerate(prompt_lens):
                gen_mask[i, plen:] = attn[i, plen:].float()

            policy_lp = _forward_logprobs(policy, generated, attn, gen_mask)
            ref_lp = _forward_logprobs(reference, generated, attn, gen_mask)
            kl = (policy_lp - ref_lp).mean()

            # Policy gradient: maximize reward * log pi(a|s)
            pg_loss = -(reward_tensor * policy_lp).mean()
            loss = pg_loss + settings.rl_kl_coef * kl

            loss = loss / settings.rl_grad_accum
            loss.backward()

            if (global_step + 1) % settings.rl_grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            progress.set_postfix(
                reward=float(reward_tensor.mean()),
                kl=float(kl.detach()),
                loss=float(loss.detach()) * settings.rl_grad_accum,
            )

        if global_step >= settings.rl_max_steps:
            break

    policy.save_pretrained(settings.rl_output_dir)
    tokenizer.save_pretrained(settings.rl_output_dir)
    logger.info("RLHF training complete -> %s", settings.rl_output_dir)
    return settings.rl_output_dir
