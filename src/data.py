from datasets import Dataset, load_dataset

from src.config import settings
from src.prompts import build_prompt


def _hf_token_kwargs() -> dict:
    return {"token": settings.hf_token} if settings.hf_token else {}


def load_tds(
    *,
    split: str | None = None,
    max_samples: int | None = None,
    streaming: bool | None = None,
) -> Dataset:
    split = split or settings.dataset_split
    streaming = settings.dataset_streaming if streaming is None else streaming
    max_samples = max_samples if max_samples is not None else settings.max_train_samples

    ds = load_dataset(
        settings.dataset_id,
        split=split,
        streaming=streaming,
        **_hf_token_kwargs(),
    )

    if streaming:
        rows = []
        for i, row in enumerate(ds):
            if max_samples and i >= max_samples:
                break
            rows.append(row)
        return Dataset.from_list(rows)

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


def format_sft_row(row: dict, tokenizer) -> dict:
    prompt = build_prompt(row["instruction"], row["input"])
    completion = row["response"]

    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    completion_ids = (
        completion_ids + [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    )

    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids
    return {
        "input_ids": input_ids,
        "labels": labels,
        "src": row["input"],
        "ref": row["response"],
    }


def format_rl_row(row: dict) -> dict:
    return {
        "query": build_prompt(row["instruction"], row["input"]),
        "src": row["input"],
        "ref": row["response"],
    }


def prepare_sft_dataset(tokenizer, max_samples: int | None = None) -> Dataset:
    ds = load_tds(max_samples=max_samples or settings.max_train_samples, streaming=True)
    return ds.map(
        format_sft_row, fn_kwargs={"tokenizer": tokenizer}, remove_columns=ds.column_names
    )


def prepare_rl_dataset(max_samples: int | None = None) -> Dataset:
    ds = load_tds(max_samples=max_samples or settings.max_rl_samples, streaming=True)
    return ds.map(format_rl_row, remove_columns=ds.column_names)
