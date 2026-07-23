"""Unit tests for evaluation.error_analysis taxonomy."""

from __future__ import annotations

from evaluation.error_analysis import ERROR_LABELS, analyze_predictions, classify_errors


def test_classify_hallucination() -> None:
    clf = classify_errors(
        "您的运单号是 SF1234567890123，明天一定到。",
        user="帮我查物流",
        context="用户未提供单号",
        expected_keywords=["订单号", "物流"],
    )
    assert "幻觉" in clf["labels"]
    assert clf["is_error"] is True
    assert clf["primary"] in ERROR_LABELS


def test_classify_attitude() -> None:
    clf = classify_errors("滚，自己查去。", user="物流呢", context="物流")
    assert "态度不当" in clf["labels"]
    assert clf["severity"] == "high"


def test_classify_format() -> None:
    clf = classify_errors("好的。", user="怎么退货", expected_keywords=["退货", "售后"])
    assert "格式违规" in clf["labels"] or clf["is_error"] is True


def test_classify_policy() -> None:
    clf = classify_errors(
        "您好。可以退，绝对能退，我保证一定给您退款。祝您购物愉快！",
        user="能退吗",
        expected_keywords=["售后"],
        must_not_contain=["绝对能退"],
    )
    assert "政策错误" in clf["labels"]


def test_classify_off_topic() -> None:
    # Zero keyword hits + low overlap with user → 答非所问
    # Regression: hit_rate=0.0 must not be coerced to 1.0 via ``x or 1.0``.
    clf = classify_errors(
        "今天天气不错，推荐您去旅游看海晒太阳。",
        user="怎么申请退货？七天无理由退货流程是什么",
        expected_keywords=["退货", "7天", "售后", "订单"],
    )
    assert clf["is_error"] is True
    assert "答非所问" in clf["labels"]
    assert clf["signals"]["keyword_hit_rate"] == 0.0


def test_classify_clean_ok() -> None:
    clf = classify_errors(
        "您好。关于退货，一般支持签收后7天无理由，请在订单页申请售后。祝您购物愉快！",
        user="怎么退货",
        context="退货咨询",
        expected_keywords=["7天", "售后", "退货"],
    )
    assert clf["is_error"] is False
    assert clf["primary"] is None
    assert clf["labels"] == []


def test_analyze_predictions_distribution() -> None:
    samples = [
        {
            "id": "a",
            "category": "物流查询",
            "user": "查物流",
            "prediction": "运单号SF1234567890123明天到",
            "expected_keywords": ["订单号"],
        },
        {
            "id": "b",
            "category": "退换货",
            "user": "退货",
            "prediction": "您好。请走售后申请7天无理由退货。祝您购物愉快！",
            "expected_keywords": ["7天", "售后", "退货"],
        },
    ]
    report = analyze_predictions(samples)
    assert report["n"] == 2
    assert report["n_error"] >= 1
    assert "label_distribution" in report
    for lb in ERROR_LABELS:
        assert lb in report["label_distribution"]
