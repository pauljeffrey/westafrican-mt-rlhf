import logging
import os

import torch
from trl import SFTConfig, SFTTrainer
from trl.trainer.utils import DataCollatorForCompletionOnlyLM

from src.config import settings
from src.data import completion_only_template, prepare_sft_dataset
from src.model_utils import load_causal_lm, load_tokenizer

logger = logging.getLogger(__name__)


def train_sft() -> str:
    os.makedirs(settings.sft_output_dir, exist_ok=True)

    logger.info("Loading dataset %s (max %d samples)", settings.dataset_id, settings.max_train_samples)
    dataset = prepare_sft_dataset()

    tokenizer = load_tokenizer()
    model = load_causal_lm()

    collator = DataCollatorForCompletionOnlyLM(
        completion_only_template(),
        tokenizer=tokenizer,
    )

    training_args = SFTConfig(
        output_dir=settings.sft_output_dir,
        per_device_train_batch_size=settings.sft_per_device_batch_size,
        gradient_accumulation_steps=settings.sft_grad_accum,
        num_train_epochs=settings.sft_epochs,
        learning_rate=settings.sft_lr,
        warmup_ratio=settings.sft_warmup_ratio,
        bf16=settings.sft_bf16 and torch.cuda.is_available(),
        logging_steps=settings.sft_logging_steps,
        save_steps=settings.sft_save_steps,
        save_total_limit=2,
        max_length=settings.max_prompt_len + settings.max_target_len,
        report_to="wandb" if settings.wandb_enabled else "none",
        run_name="sft",
        seed=settings.seed,
        dataset_text_field="text",
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    logger.info("Starting SFT training -> %s", settings.sft_output_dir)
    trainer.train()
    trainer.save_model(settings.sft_output_dir)
    tokenizer.save_pretrained(settings.sft_output_dir)
    logger.info("SFT complete.")

    if settings.hf_push_repo_id:
        logger.info("Pushing SFT model to Hub: %s", settings.hf_push_repo_id)
        trainer.model.push_to_hub(settings.hf_push_repo_id, token=settings.hf_token)
        tokenizer.push_to_hub(settings.hf_push_repo_id, token=settings.hf_token)
        logger.info("SFT model pushed to Hub.")

    return settings.sft_output_dir
