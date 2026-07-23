#!/usr/bin/env python3
"""CLI entry for LLM / rule / hybrid judge.

Examples
--------
# Offline mock judge on zero-shot results (no API key)
python scripts/10_llm_judge.py --demo --from-zero-shot reports/zero_shot_results.json

# Rule-only (default if judge.llm.enabled is false)
python scripts/10_llm_judge.py --mode rule --from-zero-shot reports/zero_shot_results.json

# Hybrid / real LLM (needs OPENAI_API_KEY or LLM_JUDGE_API_KEY)
python scripts/10_llm_judge.py --mode hybrid --from-comparison reports/comparison.json --pair sft,dpo
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.llm_judge import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
