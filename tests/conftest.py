"""Shared pytest fixtures. Keep tests free of model downloads / GPU."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


@pytest.fixture
def good_cs_reply() -> str:
    return (
        "您好，很高兴为您服务。关于退货，一般支持签收后7天无理由退货，"
        "请在订单页申请售后。祝您购物愉快！"
    )


@pytest.fixture
def sample_sft_rows() -> list[dict]:
    return [
        {
            "id": f"sft_{i}",
            "category": cat,
            "messages": [
                {"role": "system", "content": "你是中文电商智能客服。"},
                {"role": "user", "content": f"请问关于{cat}怎么处理？订单号ORD-{1000 + i}"},
                {
                    "role": "assistant",
                    "content": (
                        f"您好。关于您咨询的{cat}问题，请以订单页信息为准，"
                        "需要时请提供订单号。祝您购物愉快！"
                    ),
                },
            ],
        }
        for i, cat in enumerate(
            ["退换货", "物流查询", "优惠活动", "支付问题", "商品咨询"] * 3
        )
    ]
