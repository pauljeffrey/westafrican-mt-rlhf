import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _bool(key: str, default: str = "false") -> bool:
    return os.getenv(key, default).lower() in ("1", "true", "yes")


def _int(key: str, default: int) -> int:
    return int(os.getenv(key, str(default)))


def _float(key: str, default: float) -> float:
    return float(os.getenv(key, str(default)))


@dataclass(frozen=True)
class Settings:
    hf_token: str | None = os.getenv("HF_TOKEN") or None
    model_name: str = os.getenv("MODEL_NAME_OR_PATH", "google/gemma-2-2b-it")
    trust_remote_code: bool = _bool("TRUST_REMOTE_CODE", "false")

    reward_model_id: str = os.getenv("REWARD_MODEL_ID", "masakhane/africomet-stl")
    reward_batch_size: int = _int("REWARD_BATCH_SIZE", 8)
    reward_gpus: int = _int("REWARD_GPUS", 0)

    dataset_id: str = os.getenv("DATASET_ID", "Aletheia-ng/tds-sft")
    dataset_split: str = os.getenv("DATASET_SPLIT", "train")
    dataset_streaming: bool = _bool("DATASET_STREAMING", "true")
    max_train_samples: int = _int("MAX_TRAIN_SAMPLES", 50_000)
    max_rl_samples: int = _int("MAX_RL_SAMPLES", 10_000)
    seed: int = _int("SEED", 42)

    output_dir: str = os.getenv("OUTPUT_DIR", "./outputs")
    sft_output_dir: str = os.getenv("SFT_OUTPUT_DIR", "./outputs/sft")
    rl_output_dir: str = os.getenv("RL_OUTPUT_DIR", "./outputs/rlhf")

    max_source_len: int = _int("MAX_SOURCE_LEN", 256)
    max_target_len: int = _int("MAX_TARGET_LEN", 256)
    max_prompt_len: int = _int("MAX_PROMPT_LEN", 512)
    max_new_tokens: int = _int("MAX_NEW_TOKENS", 128)

    use_lora: bool = _bool("USE_LORA", "false")
    lora_r: int = _int("LORA_R", 16)
    lora_alpha: int = _int("LORA_ALPHA", 32)
    lora_dropout: float = _float("LORA_DROPOUT", 0.05)

    wandb_enabled: bool = _bool("WANDB_ENABLED", "false")
    wandb_project: str = os.getenv("WANDB_PROJECT", "west-african-mt-rlhf")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    # SFT
    sft_per_device_batch_size: int = _int("SFT_PER_DEVICE_BATCH_SIZE", 4)
    sft_grad_accum: int = _int("SFT_GRAD_ACCUM", 4)
    sft_epochs: float = _float("SFT_EPOCHS", 1.0)
    sft_lr: float = _float("SFT_LR", 2e-5)
    sft_warmup_ratio: float = _float("SFT_WARMUP_RATIO", 0.03)
    sft_bf16: bool = _bool("SFT_BF16", "true")
    sft_save_steps: int = _int("SFT_SAVE_STEPS", 500)
    sft_logging_steps: int = _int("SFT_LOGGING_STEPS", 10)

    # RLHF
    rl_per_device_batch_size: int = _int("RL_PER_DEVICE_BATCH_SIZE", 2)
    rl_grad_accum: int = _int("RL_GRAD_ACCUM", 4)
    rl_learning_rate: float = _float("RL_LEARNING_RATE", 1e-5)
    rl_num_epochs: int = _int("RL_NUM_EPOCHS", 1)
    rl_max_steps: int = _int("RL_MAX_STEPS", 1000)
    rl_kl_coef: float = _float("RL_KL_COEF", 0.1)
    rl_bf16: bool = _bool("RL_BF16", "true")
    rl_temperature: float = _float("RL_TEMPERATURE", 0.7)
    rl_top_p: float = _float("RL_TOP_P", 0.9)
    rl_reward_normalize: bool = _bool("RL_REWARD_NORMALIZE", "true")


settings = Settings()
