"""Unit tests for evaluation.llm_judge (offline / mock only)."""

from __future__ import annotations

import json

from evaluation.llm_judge import (
    JudgeConfig,
    MockJudgeClient,
    hybrid_pairwise,
    judge_pairwise,
    judge_pointwise,
    normalize_pairwise,
    normalize_pointwise,
    parse_judge_json,
    rule_pairwise,
    rule_pointwise,
    run_judge_on_predictions,
)


def test_parse_judge_json_raw() -> None:
    obj = parse_judge_json(
        '{"helpfulness":8,"politeness":9,"faithfulness":10,"policy":8,"safety":10,"overall":9,"rationale":"ok"}'
    )
    assert obj["overall"] == 9


def test_parse_judge_json_fenced() -> None:
    text = """```json
{"winner":"A","score_a":8,"score_b":5,"rationale":"A cleaner"}
```"""
    obj = parse_judge_json(text)
    assert obj["winner"] == "A"


def test_normalize_pointwise_clamps() -> None:
    out = normalize_pointwise(
        {
            "helpfulness": 99,
            "politeness": -3,
            "faithfulness": 7,
            "policy": 7,
            "safety": 7,
            "overall": 7,
            "rationale": "x",
        }
    )
    assert out["helpfulness"] == 10
    assert out["politeness"] == 0
    assert 0.0 <= out["score_01"] <= 1.0


def test_normalize_pairwise_aliases() -> None:
    assert normalize_pairwise({"winner": "B", "score_a": 3, "score_b": 8})["winner"] == "b"
    assert normalize_pairwise({"winner": "tie"})["winner"] == "tie"
    assert normalize_pairwise({"prefer": "answer_a"})["winner"] == "a"


def test_rule_pointwise_prefers_good_reply() -> None:
    good = "您好。关于退货，一般支持签收后7天无理由，请在订单页申请售后。祝您购物愉快！"
    bad = "滚。运单号SF1234567890123，绝对能退。"
    g = rule_pointwise(good, user="怎么退货", expected_keywords=["7天", "售后", "退货"])
    b = rule_pointwise(bad, user="怎么退货", expected_keywords=["7天", "售后", "退货"], must_not_contain=["绝对能退"])
    assert g["score_01"] > b["score_01"]
    assert g["source"] == "rule"


def test_rule_pairwise_winner() -> None:
    good = "您好。请提供订单号后我帮您查物流轨迹。如还有其他问题，随时告诉我。"
    bad = "单号SF1234567890123明天一定到。"
    r = rule_pairwise(good, bad, user="物流到哪了", expected_keywords=["订单号", "物流"])
    assert r["winner"] == "a"
    r2 = rule_pairwise(bad, good, user="物流到哪了", expected_keywords=["订单号", "物流"])
    assert r2["winner"] == "b"


def test_mock_client_pointwise_and_pairwise() -> None:
    client = MockJudgeClient()
    good = "您好。请提供订单号后查询物流。祝您购物愉快！"
    bad = "运单号SF1234567890123，明天一定到。"
    # pointwise via judge_pointwise mock mode
    cfg = JudgeConfig(mode="mock", enabled=True, model="mock-judge")
    pw = judge_pointwise(cfg, user="查物流", answer=good, context="无单号", client=client)
    assert pw["source"] == "llm" or pw.get("overall") is not None
    assert 0 <= pw["overall"] <= 10

    pair = judge_pairwise(
        cfg,
        user="查物流",
        answer_a=good,
        answer_b=bad,
        context="无单号",
        client=client,
    )
    assert pair["winner"] in ("a", "b", "tie")
    # good should beat hallucinated bad under mock
    assert pair["winner"] == "a"


def test_hybrid_escalates_on_close_or_sample() -> None:
    client = MockJudgeClient()
    a = "您好。请提供订单号后查询物流轨迹。"
    b = "您好。请提供订单号，我帮您看物流。"
    # close scores → escalate
    out = hybrid_pairwise(
        client,
        user="物流呢",
        answer_a=a,
        answer_b=b,
        context="物流",
        margin=0.5,  # force close
        sample_ratio=0.0,
        sample_key="x",
    )
    assert out.get("escalated") is True
    assert out["source"] in ("hybrid", "llm")


def test_judge_config_from_yaml() -> None:
    cfg = JudgeConfig.from_eval_yaml(
        {
            "judge": {
                "mode": "hybrid",
                "llm": {
                    "enabled": True,
                    "model": "gpt-4o-mini",
                    "sample_ratio": 0.3,
                },
            }
        }
    )
    assert cfg.mode == "hybrid"
    assert cfg.enabled is True
    assert cfg.sample_ratio == 0.3


def test_run_judge_on_predictions_pointwise() -> None:
    samples = [
        {
            "id": "1",
            "user": "怎么退货",
            "prediction": "您好。签收后7天无理由，请走售后。祝您购物愉快！",
            "expected_keywords": ["7天", "售后"],
        },
        {
            "id": "2",
            "user": "查物流",
            "prediction": "运单号SF1234567890123明天到",
            "expected_keywords": ["订单号"],
        },
    ]
    cfg = JudgeConfig(mode="mock", enabled=True)
    report = run_judge_on_predictions(samples, cfg=cfg)
    assert report["n_pointwise"] == 2
    assert report["mean_score_01"] is not None
    assert report["pointwise"][0]["id"] == "1"


def test_run_judge_pairwise_from_nested() -> None:
    samples = [
        {
            "id": "p1",
            "user": "怎么退货",
            "predictions": {
                "base": "绝对能退，运单SF1234567890123",
                "dpo": "您好。请在订单页申请售后，7天无理由以页面为准。",
            },
            "expected_keywords": ["售后", "7天"],
            "must_not_contain": ["绝对能退"],
        }
    ]
    cfg = JudgeConfig(mode="mock", enabled=True)
    report = run_judge_on_predictions(samples, cfg=cfg, pairwise_labels=("base", "dpo"))
    assert report["n_pairwise"] == 1
    assert report["pairwise"][0]["winner"] in ("a", "b", "tie")
    # dpo (b) should win
    assert report["pairwise"][0]["winner"] == "b"


def test_parse_rejects_empty() -> None:
    try:
        parse_judge_json("")
        assert False, "expected ValueError"
    except ValueError:
        pass
