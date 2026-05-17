#!/usr/bin/env python
"""Run supervised fine-tuning on Aletheia-ng/tds-sft."""

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import settings
from src.sft_train import train_sft


def main() -> None:
    logging.basicConfig(level=settings.log_level)
    train_sft()


if __name__ == "__main__":
    main()
