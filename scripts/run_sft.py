#!/usr/bin/env python
"""Run supervised fine-tuning on Aletheia-ng/tds-sft."""

import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import settings
from src.distributed import setup_distributed, teardown_distributed
from src.sft_train import train_sft


def main() -> None:
    logging.basicConfig(level=settings.log_level)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1 and settings.dist_strategy not in ("ddp", "fsdp"):
        raise SystemExit("Multi-GPU launch requires DIST_STRATEGY=ddp or fsdp in .env")
    setup_distributed(strategy=settings.dist_strategy)
    try:
        train_sft()
    finally:
        teardown_distributed()


if __name__ == "__main__":
    main()
