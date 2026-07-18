#!/usr/bin/env python3
"""Thin wrapper: run SFT training (see src.train.sft_train)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train.sft_train import main


if __name__ == "__main__":
    sys.exit(main())
