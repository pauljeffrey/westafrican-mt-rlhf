#!/usr/bin/env python
"""Run AfriCOMET-guided RLHF after SFT."""

import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import settings
from src.distributed import setup_distributed, teardown_distributed
from src.rlhf_train import train_rlhf


def main() -> None:
    logging.basicConfig(level=settings.log_level)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1 and settings.dist_strategy not in ("ddp", "fsdp"):
        raise SystemExit("Multi-GPU launch requires DIST_STRATEGY=ddp or fsdp in .env")
    setup_distributed(strategy=settings.dist_strategy)
    try:
        train_rlhf()
    finally:
        teardown_distributed()


if __name__ == "__main__":
    main()
