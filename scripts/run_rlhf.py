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
    if world > 1 and not settings.fsdp_enabled:
        raise SystemExit("Multi-GPU launch requires FSDP_ENABLED=true in .env")
    setup_distributed(tp_size=settings.fsdp_tp_size if settings.fsdp_enabled else 1)
    try:
        train_rlhf()
    finally:
        teardown_distributed()


if __name__ == "__main__":
    main()
