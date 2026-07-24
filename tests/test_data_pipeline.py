"""Unit tests for synthetic data generation, clean, and split."""

from __future__ import annotations

from src.data.clean import clean_dataset, contains_toxic, load_config, normalize_text
from src.data.generate_sft import CATEGORIES, generate_samples
from src.data.split import category_of, stratified_split


def test_generate_sft_count_and_schema() -> None:
    rows = generate_samples(n=14, seed=42, min_per_category=2)
    assert len(rows) == 14
    for r in rows:
        assert "messages" in r and isinstance(r["messages"], list)
        roles = {m["role"] for m in r["messages"]}
        assert "user" in roles
        assert "assistant" in roles
        for m in r["messages"]:
            assert str(m.get("content", "")).strip()


def test_generate_sft_covers_categories() -> None:
    rows = generate_samples(n=28, seed=0, min_per_category=2)
    cats = {r.get("category") for r in rows if r.get("category")}
    # Most categories should appear when n is large enough
    assert len(cats) >= min(5, len(CATEGORIES))


def test_clean_drops_toxic_and_empty(sample_sft_rows: list[dict]) -> None:
    toxic = {
        "id": "toxic",
        "category": "投诉建议",
        "messages": [
            {"role": "user", "content": "你们太差了"},
            {"role": "assistant", "content": "你去死吧傻逼"},
        ],
    }
    empty = {
        "id": "empty",
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": ""},
        ],
    }
    rows = sample_sft_rows + [toxic, empty]
    cleaning = load_config(None).get("cleaning") or {}
    cleaned, stats = clean_dataset(rows, "sft", cleaning)
    assert stats["output"] <= stats["input"]
    assert stats["output"] > 0
    # toxic assistant should be dropped
    for r in cleaned:
        texts = " ".join(m.get("content", "") for m in r.get("messages") or [])
        assert not contains_toxic(texts)


def test_clean_dedup() -> None:
    row = {
        "id": "dup",
        "category": "物流查询",
        "messages": [
            {"role": "user", "content": "物流到哪了？"},
            {
                "role": "assistant",
                "content": "您好。请提供订单号后我再帮您查物流轨迹。祝您购物愉快！",
            },
        ],
    }
    cleaning = load_config(None).get("cleaning") or {}
    cleaned, stats = clean_dataset([row, dict(row)], "sft", cleaning)
    assert stats["output"] == 1
    assert len(cleaned) == 1


def test_normalize_text() -> None:
    assert normalize_text("  Hello   WORLD  ") == "hello world"


def test_stratified_split_covers_all(sample_sft_rows: list[dict]) -> None:
    train, val, test = stratified_split(sample_sft_rows, (0.8, 0.1, 0.1), seed=42)
    total = len(train) + len(val) + len(test)
    assert total == len(sample_sft_rows)
    assert len(train) >= 1
    # Same seed → stable
    t2, v2, te2 = stratified_split(sample_sft_rows, (0.8, 0.1, 0.1), seed=42)
    assert [r["id"] for r in train] == [r["id"] for r in t2]
    assert [r["id"] for r in val] == [r["id"] for r in v2]
    assert [r["id"] for r in test] == [r["id"] for r in te2]


def test_category_of() -> None:
    assert category_of({"category": "退换货"}) == "退换货"
    assert category_of({"scenario": "物流"}) == "物流"
    assert category_of({}) is None
