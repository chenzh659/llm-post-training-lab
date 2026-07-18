#!/usr/bin/env python3
"""Stage: compare base vs SFT vs DPO (see evaluation.compare_models)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.compare_models import main


if __name__ == "__main__":
    sys.exit(main())
