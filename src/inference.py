import torch

from src.config import settings
from src.model_utils import load_causal_lm, load_tokenizer
from src.prompts import build_prompt, extract_translation


def translate(instruction: str, source: str, model_path: str | None = None) -> str:
    path = model_path or settings.rl_output_dir
    tokenizer = load_tokenizer(path)
    model = load_causal_lm(path, trainable=False)

    prompt = build_prompt(instruction, source)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=settings.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return extract_translation(text, prompt)
