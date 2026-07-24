"""Tests that the committed evaluation fixture suite loads cleanly."""

from __future__ import annotations

from evaluation.build_test_suite import load_suite


def test_fixture_load_min_size() -> None:
    items = load_suite(None, max_samples=None, prefer_fixture=True)
    assert len(items) >= 10
    for it in items:
        assert "id" in it
        assert it.get("user") or it.get("messages")
        assert "category" in it or "scenario" in it


def test_fixture_max_samples() -> None:
    items = load_suite(None, max_samples=5, prefer_fixture=True)
    assert len(items) == 5


def test_fixture_has_hallucination_trap() -> None:
    items = load_suite(None, max_samples=None, prefer_fixture=True)
    # At least one item is designed as a hallucination trap (no order id in user)
    trap = [
        it
        for it in items
        if (it.get("meta") or {}).get("trap") == "hallucination"
        or "幻觉" in str(it.get("scenario") or "")
        or "运单号" in str(it.get("user") or "")
    ]
    assert len(trap) >= 1
