#!/usr/bin/env python
"""Evaluate a fine-tuned checkpoint on the masakhane/mafand benchmark.

Metrics: BLEU, chrF (always), AfriCOMET (opt-in via --use-africomet).
Dataset: masakhane/mafand, validation split.

Default language pairs:
  en-ibo  English → Igbo
  en-hau  English → Hausa
  en-yor  English → Yoruba
  en-wol  English → Wolof
  en-ewe  English → Ewe
  en-fon  English → Fon
  en-twi  English → Twi

Usage examples:
  python scripts/eval.py --model-path ./outputs/rlhf
  python scripts/eval.py --model-path ./outputs/rlhf --langs en-hau,en-ibo
  python scripts/eval.py --model-path ./outputs/rlhf --use-africomet --max-samples 500
  python scripts/eval.py --model-path ./outputs/rlhf --output-file results.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import settings
from src.model_utils import load_causal_lm, load_tokenizer
from src.prompts import build_prompt, extract_translation

logger = logging.getLogger(__name__)

EVAL_DATASET = "masakhane/mafand"
EVAL_SPLIT = "validation"

DEFAULT_LANGS = ["en-ibo", "en-hau", "en-yor", "en-wol", "en-ewe", "en-fon", "en-twi"]

LANG_NAMES: dict[str, str] = {
    "ibo": "Igbo",
    "hau": "Hausa",
    "yor": "Yoruba",
    "wol": "Wolof",
    "ewe": "Ewe",
    "fon": "Fon",
    "twi": "Twi",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _target_code(config_name: str) -> str:
    """'en-ibo'  →  'ibo'"""
    return config_name.split("-", 1)[1]


def _instruction(lang_code: str) -> str:
    name = LANG_NAMES.get(lang_code, lang_code.capitalize())
    return f"Translate the sentence below to {name}:"


def _generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
) -> list[str]:
    """Tokenize a batch of prompts and return decoded generation strings."""
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=settings.max_prompt_len,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    return [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]


# ---------------------------------------------------------------------------
# Per-language evaluation
# ---------------------------------------------------------------------------


def evaluate_lang(
    config_name: str,
    model,
    tokenizer,
    *,
    max_samples: int | None,
    batch_size: int,
    max_new_tokens: int,
    use_africomet: bool,
    africomet_reward,
) -> dict[str, Any]:
    """Evaluate the model on one mafand language pair and return metric scores."""
    try:
        from sacrebleu.metrics import BLEU, CHRF
    except ImportError as e:
        raise ImportError(
            "sacrebleu is required for evaluation. Install it with: pip install sacrebleu"
        ) from e

    tgt_code = _target_code(config_name)
    instruction = _instruction(tgt_code)
    lang_name = LANG_NAMES.get(tgt_code, tgt_code)

    logger.info("Loading %s / %s [%s] ...", EVAL_DATASET, config_name, EVAL_SPLIT)
    dataset = load_dataset(
        EVAL_DATASET,
        config_name,
        split=EVAL_SPLIT,
        token=settings.hf_token or None,
    )

    if max_samples is not None and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))

    sources: list[str] = []
    references: list[str] = []
    for row in dataset:
        trans = row["translation"]
        sources.append(trans["en"])
        references.append(trans[tgt_code])

    logger.info(
        "[%s] %d examples — generating translations ...", config_name, len(sources)
    )

    hypotheses: list[str] = []
    prompts_all = [build_prompt(instruction, src) for src in sources]

    for i in tqdm(range(0, len(prompts_all), batch_size), desc=config_name, unit="batch"):
        batch_prompts = prompts_all[i : i + batch_size]
        batch_sources = sources[i : i + batch_size]
        decoded = _generate_batch(model, tokenizer, batch_prompts, max_new_tokens)
        for text, prompt in zip(decoded, batch_prompts):
            hypotheses.append(extract_translation(text, prompt))

    # BLEU and chrF
    bleu = BLEU(effective_order=True)
    chrf = CHRF()
    bleu_score = bleu.corpus_score(hypotheses, [references]).score
    chrf_score = chrf.corpus_score(hypotheses, [references]).score

    result: dict[str, Any] = {
        "lang_pair": config_name,
        "language": lang_name,
        "n_examples": len(sources),
        "bleu": round(bleu_score, 2),
        "chrf": round(chrf_score, 2),
    }

    if use_africomet and africomet_reward is not None:
        logger.info("[%s] Scoring with AfriCOMET ...", config_name)
        comet_scores = africomet_reward.score_batch(sources, hypotheses, references)
        result["africomet"] = round(sum(comet_scores) / len(comet_scores), 4)

    return result


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def _print_table(results: list[dict[str, Any]], use_africomet: bool) -> None:
    header_cols = ["Lang Pair", "Language", "N", "BLEU", "chrF"]
    if use_africomet:
        header_cols.append("AfriCOMET")

    col_widths = [max(len(h), 10) for h in header_cols]

    def row_str(values: list[str]) -> str:
        return "  ".join(v.ljust(w) for v, w in zip(values, col_widths))

    sep = "  ".join("-" * w for w in col_widths)
    print()
    print(row_str(header_cols))
    print(sep)

    bleu_sum = chrf_sum = comet_sum = 0.0
    for r in results:
        vals = [
            r["lang_pair"],
            r["language"],
            str(r["n_examples"]),
            f"{r['bleu']:.2f}",
            f"{r['chrf']:.2f}",
        ]
        if use_africomet:
            vals.append(f"{r.get('africomet', float('nan')):.4f}")
        print(row_str(vals))
        bleu_sum += r["bleu"]
        chrf_sum += r["chrf"]
        if use_africomet:
            comet_sum += r.get("africomet", 0.0)

    n = len(results)
    if n > 1:
        print(sep)
        avg_vals = [
            "Average",
            "",
            "",
            f"{bleu_sum / n:.2f}",
            f"{chrf_sum / n:.2f}",
        ]
        if use_africomet:
            avg_vals.append(f"{comet_sum / n:.4f}")
        print(row_str(avg_vals))

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint on masakhane/mafand",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Checkpoint directory (default: RL_OUTPUT_DIR from .env)",
    )
    parser.add_argument(
        "--langs",
        default=",".join(DEFAULT_LANGS),
        help=(
            "Comma-separated language pairs to evaluate, or 'all'. "
            f"Defaults to all {len(DEFAULT_LANGS)} pairs. "
            f"Available: {', '.join(DEFAULT_LANGS)}"
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum examples per language (default: use full validation split)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Generation batch size (default: 8)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Max tokens to generate (default: MAX_NEW_TOKENS from .env)",
    )
    parser.add_argument(
        "--use-africomet",
        action="store_true",
        help="Also score translations with AfriCOMET (slower; requires GPU)",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Save results to this JSON file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # Resolve language list
    if args.langs.strip().lower() == "all":
        lang_list = DEFAULT_LANGS
    else:
        lang_list = [l.strip() for l in args.langs.split(",") if l.strip()]

    unknown = [l for l in lang_list if l not in DEFAULT_LANGS]
    if unknown:
        logger.warning(
            "Unknown language pair(s) %s — will attempt anyway. "
            "Supported: %s",
            unknown,
            DEFAULT_LANGS,
        )

    model_path = args.model_path or settings.rl_output_dir
    max_new_tokens = args.max_new_tokens or settings.max_new_tokens

    logger.info("Model checkpoint : %s", model_path)
    logger.info("Language pairs   : %s", lang_list)
    logger.info("Max samples/lang : %s", args.max_samples or "all")

    logger.info("Loading tokenizer and model ...")
    tokenizer = load_tokenizer(model_path)
    model = load_causal_lm(model_path, trainable=False)

    africomet_reward = None
    if args.use_africomet:
        from src.reward import AfricometReward

        logger.info("Loading AfriCOMET reward model ...")
        africomet_reward = AfricometReward()

    all_results: list[dict[str, Any]] = []
    for config_name in lang_list:
        try:
            result = evaluate_lang(
                config_name,
                model,
                tokenizer,
                max_samples=args.max_samples,
                batch_size=args.batch_size,
                max_new_tokens=max_new_tokens,
                use_africomet=args.use_africomet,
                africomet_reward=africomet_reward,
            )
            all_results.append(result)
        except Exception:
            logger.exception("Failed to evaluate %s — skipping.", config_name)

    if not all_results:
        logger.error("No results to report.")
        sys.exit(1)

    _print_table(all_results, use_africomet=args.use_africomet)

    if args.output_file:
        out_path = Path(args.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
