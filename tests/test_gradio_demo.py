"""Offline unit tests for Gradio demo chat engine (no Gradio server, no GPU)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.demo.chat_engine import (
    EXAMPLE_PROMPTS,
    format_score_markdown,
    generate_reply,
    history_to_messages,
    infer_category,
    load_model_bundle,
    mock_reply,
    resolve_model_path,
    score_reply,
)


def test_infer_category_returns():
    assert infer_category("七天无理由怎么退货") == "退换货"
    assert infer_category("物流不更新") == "物流查询"
    assert infer_category("优惠券过期了") == "优惠活动"


def test_mock_reply_nonempty_and_polite():
    text = mock_reply("客服您好，护眼台灯怎么退？单号：ORD-2026-88421")
    assert len(text) >= 20
    assert "您好" in text or "感谢" in text


def test_mock_reply_hallucination_path():
    # No order id in user text → mock invents SF tracking for demo scoring
    text = mock_reply("能告诉我运单号吗？")
    assert "SF" in text or "单号" in text


def test_score_reply_good_composite():
    user = "客服您好，护眼台灯签收3天了，七天无理由怎么退？单号：ORD-2026-88421"
    ans = mock_reply(user, category="退换货")
    s = score_reply(ans, user, category="退换货")
    assert s["composite"] is not None
    assert 0.0 <= float(s["composite"]) <= 1.0
    assert s["category"] == "退换货"
    assert s["safety_passed"] is True


def test_score_reply_flags_hallucination():
    user = "物流怎么还没更新？"  # no tracking in context
    ans = "您好。您的快递单号是 SF1234567890123，明天到。"
    s = score_reply(ans, user)
    assert s["hallucination"] is True
    assert s["invented_ids"]


def test_format_score_markdown():
    md = format_score_markdown(
        {
            "composite": 0.8,
            "passed": True,
            "category": "退换货",
            "expected_keywords": ["7天"],
            "format_score": 0.9,
            "keyword_hit_rate": 1.0,
            "hallucination": False,
            "invented_ids": [],
            "safety_passed": True,
        },
        latency_ms=12.3,
    )
    assert "Composite" in md
    assert "12" in md


def test_load_mock_bundle():
    b = load_model_bundle("mock", force_mock=True)
    assert b.mode == "mock"
    reply, ms = generate_reply(b, "你好，想退货")
    assert reply
    assert ms >= 0


def test_history_to_messages_tuples():
    hist = [["hi", "hello"], ["退货", "好的"]]
    msgs = history_to_messages(hist)
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"
    assert len(msgs) == 4


def test_history_to_messages_dicts():
    hist = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
    ]
    msgs = history_to_messages(hist)
    assert len(msgs) == 2
    assert msgs[1]["content"] == "b"


def test_resolve_model_path_aliases():
    assert resolve_model_path("mock") == "mock"
    # base maps to HF id string
    r = resolve_model_path("base")
    assert "Qwen" in str(r)


def test_examples_nonempty():
    assert len(EXAMPLE_PROMPTS) >= 4
    for cat, prompt in EXAMPLE_PROMPTS:
        assert cat and prompt
