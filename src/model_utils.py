import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import settings


def load_tokenizer(model_path: str | None = None):
    path = model_path or settings.model_name
    tokenizer = AutoTokenizer.from_pretrained(
        path,
        trust_remote_code=settings.trust_remote_code,
        token=settings.hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _dtype():
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


def load_causal_lm(model_path: str | None = None, *, trainable: bool = True):
    path = model_path or settings.model_name
    kwargs = {
        "trust_remote_code": settings.trust_remote_code,
        "token": settings.hf_token,
        "torch_dtype": _dtype(),
    }
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    if settings.use_lora and trainable:
        model = get_peft_model(
            model,
            LoraConfig(
                r=settings.lora_r,
                lora_alpha=settings.lora_alpha,
                lora_dropout=settings.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                task_type="CAUSAL_LM",
            ),
        )
    if trainable:
        model.train()
    else:
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
    return model

