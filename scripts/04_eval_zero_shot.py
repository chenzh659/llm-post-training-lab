#!/usr/bin/env python3
"""Stage: zero-shot evaluation of base model (see evaluation.zero_shot_eval)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.zero_shot_eval import main


if __name__ == "__main__":
    sys.exit(main())
