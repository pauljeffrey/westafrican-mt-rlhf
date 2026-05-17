#!/usr/bin/env python
"""Run AfriCOMET-guided RLHF after SFT."""

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import settings
from src.rlhf_train import train_rlhf


def main() -> None:
    logging.basicConfig(level=settings.log_level)
    train_rlhf()


if __name__ == "__main__":
    main()
