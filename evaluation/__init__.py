"""Offline evaluation suite for Chinese e-commerce CS post-training.

Modules
-------
metrics
    Format / keyword / hallucination / safety / ROUGE-L char F1.
zero_shot_eval
    Base-model generation eval on a test suite.
compare_models
    Rule-based win-rate across base / SFT / DPO.
error_analysis
    Error taxonomy: 幻觉 / 政策错误 / 格式违规 / 态度不当 / 答非所问 / 信息缺失.
build_test_suite
    Enrich SFT test split with gold keyword checks; load fixtures.
"""

from __future__ import annotations

__all__ = [
    "metrics",
    "zero_shot_eval",
    "compare_models",
    "error_analysis",
    "build_test_suite",
]

__version__ = "0.1.0"
