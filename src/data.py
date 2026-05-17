from datasets import Dataset, load_dataset

from src.config import settings
from src.prompts import RESPONSE_TEMPLATE, build_prompt


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


def format_sft_row(row: dict) -> dict:
    prompt = build_prompt(row["instruction"], row["input"])
    return {
        "text": prompt + row["response"],
        "prompt": prompt,
        "src": row["input"],
        "ref": row["response"],
    }


def format_rl_row(row: dict) -> dict:
    return {
        "query": build_prompt(row["instruction"], row["input"]),
        "src": row["input"],
        "ref": row["response"],
    }


def prepare_sft_dataset(max_samples: int | None = None) -> Dataset:
    ds = load_tds(max_samples=max_samples or settings.max_train_samples, streaming=True)
    return ds.map(format_sft_row, remove_columns=ds.column_names)


def prepare_rl_dataset(max_samples: int | None = None) -> Dataset:
    ds = load_tds(max_samples=max_samples or settings.max_rl_samples, streaming=True)
    return ds.map(format_rl_row, remove_columns=ds.column_names)


def completion_only_template() -> str:
    return RESPONSE_TEMPLATE
