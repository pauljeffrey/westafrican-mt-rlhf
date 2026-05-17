#!/usr/bin/env python
"""Quick CLI to test a trained checkpoint."""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import settings
from src.inference import translate


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate with fine-tuned model")
    parser.add_argument("--instruction", required=True, help="e.g. Translate to Hausa:")
    parser.add_argument("--input", required=True, help="Source sentence")
    parser.add_argument("--model-path", default=None, help="Checkpoint path (default: RL output)")
    args = parser.parse_args()

    logging.basicConfig(level=settings.log_level)
    result = translate(args.instruction, args.input, model_path=args.model_path)
    print(result)


if __name__ == "__main__":
    main()
