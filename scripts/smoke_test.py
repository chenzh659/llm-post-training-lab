#!/usr/bin/env python3
"""Lightweight smoke test — no model download, no GPU required.

Checks
------
1. Tiny SFT + preference generation
2. Schema validation (messages / chosen-rejected)
3. Clean + split
4. Metric unit checks on fixtures / synthetic strings
5. Optional offline eval mock path (if evaluation package importable)

Exit 0 on success; non-zero on first hard failure.
"""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PASS = 0
FAIL = 0
ERRORS: list[str] = []


def _ok(msg: str) -> None:
    global PASS
    PASS += 1
    print(f"  [PASS] {msg}")


def _fail(msg: str) -> None:
    global FAIL
    FAIL += 1
    ERRORS.append(msg)
    print(f"  [FAIL] {msg}")


def _assert(cond: bool, msg: str) -> None:
    if cond:
        _ok(msg)
    else:
        _fail(msg)


def validate_sft_schema(row: dict[str, Any]) -> list[str]:
    errs: list[str] = []
    if "messages" not in row or not isinstance(row["messages"], list):
        errs.append("missing messages list")
        return errs
    if len(row["messages"]) < 2:
        errs.append("messages too short")
    roles = [m.get("role") for m in row["messages"] if isinstance(m, dict)]
    if "user" not in roles:
        errs.append("no user turn")
    if "assistant" not in roles:
        errs.append("no assistant turn")
    for m in row["messages"]:
        if not isinstance(m, dict):
            errs.append("message not dict")
            continue
        if "role" not in m or "content" not in m:
            errs.append("message missing role/content")
        elif not str(m.get("content", "")).strip():
            errs.append("empty content")
    return errs


def validate_pref_schema(row: dict[str, Any]) -> list[str]:
    errs: list[str] = []
    # Accept either prompt/chosen/rejected or messages + chosen/rejected
    has_triple = all(k in row for k in ("prompt", "chosen", "rejected"))
    has_msgs = "messages" in row or "prompt" in row
    if not has_triple and not (
        has_msgs and ("chosen" in row or "chosen_messages" in row)
    ):
        # also accept nested format from generator
        if not (("chosen" in row and "rejected" in row) or ("chosen" in row)):
            errs.append("missing preference fields")
    chosen = row.get("chosen") or row.get("chosen_response")
    rejected = row.get("rejected") or row.get("rejected_response")
    if chosen is None and isinstance(row.get("chosen"), (str, list, dict)):
        chosen = row["chosen"]
    if isinstance(chosen, list):
        # messages list
        pass
    elif chosen is not None and not str(chosen).strip() and not isinstance(chosen, (list, dict)):
        errs.append("empty chosen")
    if rejected is not None and isinstance(rejected, str) and not rejected.strip():
        errs.append("empty rejected")
    return errs


def test_generate_tiny() -> list[dict[str, Any]]:
    print("\n== 1. Tiny data generation ==")
    from src.data.generate_sft import generate_samples
    from src.data import generate_preference as gen_pref

    sft = generate_samples(n=14, seed=42, min_per_category=2)
    _assert(len(sft) == 14, f"SFT count == 14 (got {len(sft)})")
    schema_errs = 0
    for r in sft:
        e = validate_sft_schema(r)
        if e:
            schema_errs += 1
    _assert(schema_errs == 0, "All SFT rows pass schema")

    # Preference: call generator main API or function
    pref_fn = getattr(gen_pref, "generate_pairs", None) or getattr(
        gen_pref, "generate_samples", None
    )
    pref_rows: list[dict[str, Any]] = []
    if pref_fn is not None:
        try:
            pref_rows = pref_fn(n=8, seed=42)  # type: ignore[misc]
        except TypeError:
            try:
                pref_rows = pref_fn(8, 42)  # type: ignore[misc]
            except Exception:
                pref_rows = []
    if not pref_rows:
        # Fall back: write via CLI main into temp
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "pref.jsonl"
            rc = gen_pref.main(
                ["--num-samples", "8", "--seed", "42", "--output", str(out)]
            )
            _assert(rc == 0, "preference main() rc==0")
            if out.is_file():
                with out.open("r", encoding="utf-8") as f:
                    pref_rows = [json.loads(line) for line in f if line.strip()]
    _assert(len(pref_rows) >= 4, f"Preference rows >= 4 (got {len(pref_rows)})")
    p_err = 0
    for r in pref_rows:
        if validate_pref_schema(r):
            p_err += 1
    _assert(p_err == 0, "All preference rows pass schema")
    return sft


def test_clean_split(sft_rows: list[dict[str, Any]]) -> None:
    print("\n== 2. Clean + split ==")
    from src.data.clean import clean_dataset, load_config
    from src.data.split import stratified_split

    cleaning = load_config(None).get("cleaning") or {}
    cleaned, stats = clean_dataset(sft_rows, "sft", cleaning)
    _assert(stats["output"] > 0, f"clean kept >0 (kept={stats['output']})")
    _assert(stats["output"] <= stats["input"], "clean output <= input")

    train, val, test = stratified_split(cleaned, (0.8, 0.1, 0.1), seed=42)
    total = len(train) + len(val) + len(test)
    _assert(total == len(cleaned), f"split covers all ({total}=={len(cleaned)})")
    _assert(len(train) >= 1, "train non-empty")


def test_metrics() -> None:
    print("\n== 3. Metrics unit tests ==")
    from evaluation.metrics import (
        format_compliance,
        keyword_hit,
        hallucination_heuristic,
        safety_score,
        rouge_l_char,
        score_sample,
        aggregate_scores,
        char_f1,
    )

    good = (
        "您好，很高兴为您服务。关于退货，一般支持签收后7天无理由退货，"
        "请在订单页申请售后。祝您购物愉快！"
    )
    fmt = format_compliance(good)
    _assert(fmt["score"] >= 0.6, f"format_compliance score>={0.6} ({fmt['score']})")

    kw = keyword_hit(good, ["7天", "售后", "退货"])
    _assert(kw["hit_rate"] >= 0.5, f"keyword hit_rate>={0.5} ({kw['hit_rate']})")

    hall = hallucination_heuristic(
        "您的运单号是 SF1234567890123，明天到。",
        context="用户问物流，未提供单号",
    )
    _assert(hall["is_hallucination"] is True, "hallucination detects invented tracking")

    safe = safety_score("您好，我来帮您处理退货。")
    _assert(safe["passed"] is True, "safety_score passes clean reply")
    unsafe = safety_score("你去死吧滚")
    _assert(unsafe["passed"] is False, "safety_score fails banned phrases")

    rouge = rouge_l_char("七天无理由退货申请售后", "七天无理由退货")
    _assert(rouge["f1"] > 0.3, f"rouge_l_char f1>0.3 ({rouge['f1']})")
    _assert(char_f1("abc", "abc") == 1.0, "char_f1 identical == 1.0")

    scored = score_sample(
        good,
        context="订单退货咨询 ORD-1",
        reference=good,
        expected_keywords=["7天", "售后"],
        must_not_contain=["绝对能退"],
    )
    _assert(0.0 <= scored["composite"] <= 1.0, "composite in [0,1]")
    _assert(scored["passed"] is True, "score_sample passed on good reply")

    agg = aggregate_scores([scored, scored])
    _assert(agg["n"] == 2, "aggregate n==2")
    _assert(agg["mean_composite"] > 0.5, "aggregate mean_composite>0.5")


def test_fixture_load() -> None:
    print("\n== 4. Fixture suite load ==")
    from evaluation.build_test_suite import load_suite

    items = load_suite(None, max_samples=5, prefer_fixture=True)
    _assert(len(items) >= 1, f"fixture load n>=1 (got {len(items)})")
    _assert("id" in items[0], "fixture item has id")
    _assert("user" in items[0] or "messages" in items[0], "fixture has user/messages")


def test_utils() -> None:
    print("\n== 5. Utils ==")
    from src.utils import (
        project_root,
        set_seed,
        save_json,
        load_json,
        write_jsonl,
        read_jsonl,
        ensure_dir,
    )

    root = project_root()
    _assert(root == ROOT.resolve() or root.exists(), f"project_root exists: {root}")
    set_seed(42)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "sub" / "t.json"
        ensure_dir(p.parent)
        save_json(p, {"a": 1})
        _assert(load_json(p)["a"] == 1, "save/load_json roundtrip")
        jl = Path(td) / "x.jsonl"
        write_jsonl(jl, [{"x": 1}, {"x": 2}])
        rows = read_jsonl(jl)
        _assert(len(rows) == 2, "jsonl roundtrip")


def test_imports() -> None:
    print("\n== 0. Imports ==")
    modules = [
        "src.utils",
        "src.data.generate_sft",
        "src.data.generate_preference",
        "src.data.clean",
        "src.data.split",
        "src.data.analyze",
        "src.demo.chat_engine",
        "evaluation.metrics",
        "evaluation.build_test_suite",
        "evaluation.zero_shot_eval",
        "evaluation.compare_models",
        "evaluation.error_analysis",
    ]
    for m in modules:
        try:
            __import__(m)
            _ok(f"import {m}")
        except Exception as e:
            _fail(f"import {m}: {e}")


def main() -> int:
    print("=" * 60)
    print("llm-post-training-lab :: smoke_test")
    print("=" * 60)
    print(f"ROOT={ROOT}")
    print(f"python={sys.version.split()[0]}")

    try:
        test_imports()
        sft = test_generate_tiny()
        test_clean_split(sft)
        test_metrics()
        test_fixture_load()
        test_utils()
    except Exception as e:
        _fail(f"Unhandled exception: {e}")
        traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"RESULTS: {PASS} passed, {FAIL} failed")
    if ERRORS:
        print("Failures:")
        for e in ERRORS:
            print(f"  - {e}")
    print("=" * 60)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
