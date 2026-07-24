"""Unit tests for evaluation.metrics (no model / GPU)."""

from __future__ import annotations

from evaluation.metrics import (
    aggregate_scores,
    char_f1,
    format_compliance,
    hallucination_heuristic,
    keyword_hit,
    mc_accuracy,
    rouge_l_char,
    rule_judge_score,
    safety_score,
    score_sample,
)


def test_format_compliance_good(good_cs_reply: str) -> None:
    fmt = format_compliance(good_cs_reply)
    assert fmt["score"] >= 0.6
    assert fmt["passed"] is True
    assert fmt["checks"]["non_empty"] is True


def test_format_compliance_empty() -> None:
    fmt = format_compliance("   ")
    assert fmt["score"] == 0.0
    assert fmt["passed"] is False


def test_format_compliance_too_short() -> None:
    fmt = format_compliance("好的")
    assert fmt["checks"]["non_empty"] is False or fmt["score"] < 0.6


def test_keyword_hit_partial() -> None:
    kw = keyword_hit("支持七天无理由退货，请走售后", ["7天", "售后", "退货"])
    # "7天" may miss if answer says "七天" — accept either partial or full
    assert 0.0 <= kw["hit_rate"] <= 1.0
    assert "售后" in kw["found"]
    assert "退货" in kw["found"]


def test_keyword_hit_forbidden_zeros_score() -> None:
    kw = keyword_hit(
        "可以退，绝对能退，走售后",
        ["售后"],
        must_not_contain=["绝对能退"],
    )
    assert kw["score"] == 0.0
    assert "绝对能退" in kw["forbidden_hits"]
    assert kw["passed"] is False


def test_keyword_hit_no_expected_is_neutral() -> None:
    kw = keyword_hit("任意回复内容足够长即可", None)
    assert kw["hit_rate"] == 1.0
    assert kw["score"] == 1.0


def test_hallucination_detects_tracking() -> None:
    hall = hallucination_heuristic(
        "您的运单号是 SF1234567890123，明天到。",
        context="用户问物流，未提供单号",
    )
    assert hall["is_hallucination"] is True
    assert hall["n_flags"] >= 1
    assert hall["passed"] is False


def test_hallucination_allows_id_in_context() -> None:
    ctx = "订单号 ORD-2026-88421 查询物流"
    ans = "您好。订单 ORD-2026-88421 请以订单页轨迹为准。"
    hall = hallucination_heuristic(ans, context=ctx)
    assert hall["is_hallucination"] is False
    assert hall["passed"] is True


def test_hallucination_detects_price() -> None:
    hall = hallucination_heuristic(
        "这款只要 ¥199.00 包邮。",
        context="用户询问参数，未谈价格",
        also_check_prices=True,
    )
    assert hall["is_hallucination"] is True
    assert "199.00" in hall["invented_prices"] or hall["n_flags"] > 0


def test_safety_pass_and_fail() -> None:
    assert safety_score("您好，我来帮您处理退货。")["passed"] is True
    unsafe = safety_score("你去死吧滚")
    assert unsafe["passed"] is False
    assert unsafe["score"] < 1.0


def test_rouge_l_and_char_f1() -> None:
    rouge = rouge_l_char("七天无理由退货申请售后", "七天无理由退货")
    assert rouge["f1"] > 0.3
    assert char_f1("abc", "abc") == 1.0
    assert char_f1("", "x") == 0.0
    assert char_f1("", "") == 1.0


def test_mc_accuracy_letter() -> None:
    r = mc_accuracy("B. 引导以详情页为准", "B")
    assert r["correct"] is True
    assert r["score"] == 1.0
    assert r["skipped"] is False


def test_mc_accuracy_skipped_without_gold() -> None:
    r = mc_accuracy("任意", None)
    assert r["skipped"] is True
    assert r["score"] is None


def test_score_sample_and_aggregate(good_cs_reply: str) -> None:
    scored = score_sample(
        good_cs_reply,
        context="订单退货咨询 ORD-1",
        reference=good_cs_reply,
        expected_keywords=["7天", "售后"],
        must_not_contain=["绝对能退"],
    )
    assert 0.0 <= scored["composite"] <= 1.0
    assert scored["passed"] is True
    assert "components" in scored

    agg = aggregate_scores([scored, scored])
    assert agg["n"] == 2
    assert agg["mean_composite"] > 0.5
    assert agg["pass_rate"] == 1.0


def test_aggregate_empty() -> None:
    agg = aggregate_scores([])
    assert agg["n"] == 0
    assert agg["mean_composite"] == 0.0


def test_rule_judge_score_range(good_cs_reply: str) -> None:
    s = rule_judge_score(good_cs_reply, context="退货", expected_keywords=["7天", "售后"])
    assert 0.0 <= s <= 1.0
    assert rule_judge_score("") == 0.0
