import logging
import os

import torch
from trl import SFTConfig, SFTTrainer
from trl.trainer.utils import DataCollatorForCompletionOnlyLM

from src.config import settings
from src.data import completion_only_template, prepare_sft_dataset
from src.distributed import (
    apply_tensor_parallel,
    dist_context,
    fsdp_trainer_config,
    is_distributed,
    push_to_hub,
    save_fsdp_model,
)
from src.model_utils import load_causal_lm, load_tokenizer

logger = logging.getLogger(__name__)


def train_sft() -> str:
    os.makedirs(settings.sft_output_dir, exist_ok=True)
    ctx = dist_context()

    if ctx.is_main:
        logger.info(
            "Loading dataset %s (max %d samples)", settings.dataset_id, settings.max_train_samples
        )
    dataset = prepare_sft_dataset()

    tokenizer = load_tokenizer()
    model = load_causal_lm()
    if settings.fsdp_enabled and ctx.tp_size > 1:
        model = apply_tensor_parallel(model)

    collator = DataCollatorForCompletionOnlyLM(
        completion_only_template(),
        processing_class=tokenizer,
    )

    sft_kwargs: dict = {
        "output_dir": settings.sft_output_dir,
        "per_device_train_batch_size": settings.sft_per_device_batch_size,
        "gradient_accumulation_steps": settings.sft_grad_accum,
        "num_train_epochs": settings.sft_epochs,
        "learning_rate": settings.sft_lr,
        "warmup_ratio": settings.sft_warmup_ratio,
        "bf16": settings.sft_bf16 and torch.cuda.is_available(),
        "logging_steps": settings.sft_logging_steps,
        "save_steps": settings.sft_save_steps,
        "save_total_limit": 2,
        "max_length": settings.max_prompt_len + settings.max_target_len,
        "report_to": "wandb" if settings.wandb_enabled and ctx.is_main else "none",
        "run_name": "sft",
        "seed": settings.seed,
        "dataset_text_field": "text",
        "remove_unused_columns": False,
        "ddp_find_unused_parameters": False,
    }

    if settings.fsdp_enabled and is_distributed():
        fsdp_flag, fsdp_config = fsdp_trainer_config(
            sharding=settings.fsdp_sharding,
            use_bf16=settings.sft_bf16,
            use_lora=settings.use_lora,
            layer_cls_name=settings.fsdp_transformer_layer_cls or None,
            cpu_offload=settings.fsdp_cpu_offload,
        )
        sft_kwargs["fsdp"] = fsdp_flag
        sft_kwargs["fsdp_config"] = fsdp_config
        if ctx.is_main:
            logger.info("SFT FSDP enabled: %s (tp_size=%d)", fsdp_flag, ctx.tp_size)

    training_args = SFTConfig(**sft_kwargs)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    if ctx.is_main:
        logger.info("Starting SFT training -> %s", settings.sft_output_dir)
    trainer.train()

    if settings.fsdp_enabled and is_distributed():
        save_fsdp_model(trainer.model, settings.sft_output_dir, tokenizer)
    else:
        trainer.save_model(settings.sft_output_dir)
        tokenizer.save_pretrained(settings.sft_output_dir)

    if ctx.is_main:
        logger.info("SFT complete.")

    if settings.hf_push_repo_id and ctx.is_main:
        logger.info("Pushing SFT model to Hub: %s", settings.hf_push_repo_id)
        push_to_hub(trainer.model, settings.hf_push_repo_id, tokenizer, settings.hf_token)

    return settings.sft_output_dir
