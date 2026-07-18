#!/usr/bin/env python3
"""Stage: error taxonomy analysis (see evaluation.error_analysis)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.error_analysis import main


if __name__ == "__main__":
    sys.exit(main())
