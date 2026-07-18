"""Generate synthetic DPO preference pairs for Chinese e-commerce CS.

Each pair: prompt (messages or text), chosen (good CS answer), rejected (bad answer).
Rejected failure modes: rude, hallucinated tracking, wrong policy, incomplete, off-topic.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore

from src.data.generate_sft import (
    CATEGORIES,
    DEFAULT_SYSTEM,
    USER_PARAPHRASE,
    _pick,
    _product_phrase,
    _project_root,
    _with_order,
    load_config,
)

REJECT_TYPES = [
    "rude",
    "hallucinated_tracking",
    "wrong_policy",
    "incomplete",
    "off_topic",
]


_USER_PREFIX = ["", "你好，", "在吗？", "客服您好，", "麻烦问下，", "急！"]


def _user_prompt(rng: random.Random, category: str) -> tuple[str, str | None, str]:
    """Return (user_text, order_id_or_none, product_hint)."""
    product = _product_phrase(rng)
    prefix = _pick(rng, _USER_PREFIX)
    oid: str | None = None
    extra = _pick(rng, ["", " 谢谢。", " 尽快回复。", " 在线等。", f" 商品是{product}。"])
    if category == "商品咨询":
        intent = _pick(rng, ["stock", "spec", "compat"])
        user = f"{prefix}想了解{product}，{_pick(rng, USER_PARAPHRASE[intent])}{extra}".strip()
    elif category == "物流查询":
        base = f"{prefix}买的{product}，{_pick(rng, USER_PARAPHRASE['logistics'])}{extra}".strip()
        user, oid = _with_order(rng, base)
    elif category == "退换货":
        base = f"{prefix}{product}，{_pick(rng, USER_PARAPHRASE['return'])}{extra}".strip()
        user, oid = _with_order(rng, base)
    elif category == "优惠活动":
        user = f"{prefix}关于{product}，{_pick(rng, USER_PARAPHRASE['coupon'])}{extra}".strip()
    elif category == "投诉建议":
        base = f"{prefix}{product}，{_pick(rng, USER_PARAPHRASE['complaint'])}{extra}".strip()
        user, oid = _with_order(rng, base)
    elif category == "账户订单":
        base = f"{prefix}{_pick(rng, USER_PARAPHRASE['order'])}{extra}".strip()
        user, oid = _with_order(rng, base)
    else:  # 支付问题
        base = f"{prefix}{_pick(rng, USER_PARAPHRASE['pay'])}{extra}".strip()
        user, oid = _with_order(rng, base)
    return user, oid, product


_CHOSEN_CLOSINGS = [
    "如还有疑问请继续补充。",
    "需要我一步步带您操作也可以说。",
    "感谢您的耐心，祝您购物愉快。",
    "我会在权限范围内尽力协助您。",
]


def chosen_reply(rng: random.Random, category: str, user: str, product: str, oid: str | None) -> str:
    """Polite, policy-aware, non-hallucinating CS reply with light variation."""
    oid_mention = f"订单号 {oid} " if oid else ""
    close = _pick(rng, _CHOSEN_CLOSINGS)
    tip = _pick(
        rng,
        [
            "建议您以订单页/详情页实时信息为准。",
            "关键规则以平台与店铺公示为准。",
            "我无法编造未查询到的单号或价格。",
        ],
    )
    templates = {
        "商品咨询": (
            f"您好，关于【{product}】：规格与库存请以商品详情页和下单时页面为准。"
            f"若您补充关注点（尺寸、材质、适配机型），我可以按详情要点帮您梳理。{tip}{close}"
        ),
        "物流查询": (
            f"您好，很抱歉让您久等了。"
            f"{('已记录您提供的 ' + oid_mention + '。') if oid else '请您提供订单号以便核对。'}"
            f"请先在“我的订单-查看物流”确认最新节点；我不会编造运单轨迹。"
            f"若长时间无更新，可凭单号申请催派。{tip}{close}"
        ),
        "退换货": (
            f"您好，关于{oid_mention or '该商品的'}退换货：签收后7天内商品完好通常可申请无理由退货；"
            f"质量问题请上传凭证走售后质检。路径：我的订单→申请售后。"
            f"运费与是否支持以售后页规则为准。请保留包装便于核对。{tip}{close}"
        ),
        "优惠活动": (
            f"您好，针对【{product}】的优惠券/满减是否可叠加以结算页为准；活动商品可能排除用券。"
            f"请在有效期内使用并注意门槛与适用类目。过期券一般无法补发。{tip}{close}"
        ),
        "投诉建议": (
            f"您好，非常理解您的心情，给您带来不便深表歉意。"
            f"{oid_mention}请补充问题类型与图片凭证，我们协助登记售后或升级处理"
            f"（退换/补发/按规则协商）。请保留包裹与记录。{tip}{close}"
        ),
        "账户订单": (
            f"您好，订单取消/改址/查单请以订单页状态为准；"
            f"{oid_mention}未发货通常可取消或改地址，已发货需按物流规则处理。"
            f"账户安全相关操作需本人验证，客服不会绕过验证直接改绑。{tip}{close}"
        ),
        "支付问题": (
            f"您好，支付失败可换方式重试并检查限额；疑似重复扣款请提供流水号协助核查退款。"
            f"{oid_mention}发票请在订单详情申请。请勿高频重复支付。到账时间视渠道而定。{tip}{close}"
        ),
    }
    return templates.get(category, templates["商品咨询"])


def rejected_reply(
    rng: random.Random,
    reject_type: str,
    category: str,
    product: str,
    oid: str | None,
) -> str:
    """Bad CS reply for the given failure mode."""
    fake_track = _pick(
        rng,
        [
            "YT8888777766665",
            "SF1234567890123",
            "ZTO998877665544",
            "JD009988776655",
        ],
    )
    fake_oid = oid or "SYN-FAKE-00000"
    if reject_type == "rude":
        return _pick(
            rng,
            [
                "你自己不会看订单页吗？别来烦我。",
                "这种问题也问，智商欠费吧。不想买就滚。",
                "爱退不退，爱买不买，没空陪你扯。",
                "投诉有用吗？爱找平台找平台去。",
            ],
        )
    if reject_type == "hallucinated_tracking":
        return (
            f"您的包裹已经到了，运单号是{fake_track}，今天下午三点前必达，"
            f"快递员电话13800000000。不用查了，我说的就是准的。"
            f"（未查询真实系统，纯属编造轨迹）"
        )
    if reject_type == "wrong_policy":
        return _pick(
            rng,
            [
                f"所有商品签收后30天都无条件全额退款，运费我们永远包，不用质检直接退。",
                f"优惠券过期了我可以后台给您无限补发，满减和所有券随便叠，保证全网最低价再减100。",
                f"未发货也可以要求我们伪造发货记录；已使用商品也能无理由退且不用承担任何费用。",
                f"您可以直接把银行卡号发我，我帮您改价并私下转账差价。",
            ],
        )
    if reject_type == "incomplete":
        return _pick(
            rng,
            [
                "好的。",
                "收到。",
                "嗯。",
                "你再等等。",
                "看详情页。",
            ],
        )
    # off_topic
    return _pick(
        rng,
        [
            f"今天天气不错，推荐您去旅游，顺便聊聊足球比分。",
            f"我是写代码的，您这个问题我改成用Python排序吧。",
            f"给您讲个冷笑话：为什么程序员分不清万圣节和圣诞节？因为 Oct 31 == Dec 25。",
            f"关于宇宙起源有很多理论，霍金曾说过……（与{product}售后无关）",
        ],
    )


def build_pair(rng: random.Random, idx: int) -> dict[str, Any]:
    category = _pick(rng, CATEGORIES)
    user, oid, product = _user_prompt(rng, category)
    reject_type = _pick(rng, REJECT_TYPES)
    chosen = chosen_reply(rng, category, user, product, oid)
    rejected = rejected_reply(rng, reject_type, category, product, oid)
    prompt_messages = [
        {"role": "system", "content": DEFAULT_SYSTEM},
        {"role": "user", "content": user},
    ]
    return {
        "id": f"pref_{idx:06d}",
        "category": category,
        "scenario": category,
        "prompt": user,
        "prompt_messages": prompt_messages,
        "chosen": chosen,
        "rejected": rejected,
        "meta": {
            "reject_type": reject_type,
            "order_id": oid,
            "product": product,
        },
    }


def generate_pairs(n: int, seed: int = 42) -> list[dict[str, Any]]:
    """Generate n unique (prompt+chosen) pairs; oversample then trim."""
    rng = random.Random(seed + 17)
    pairs: list[dict[str, Any]] = []
    seen: set[str] = set()
    idx = 0
    # oversample budget to absorb template collisions after cleaning
    max_attempts = max(n * 8, n + 100)
    attempts = 0
    while len(pairs) < n and attempts < max_attempts:
        attempts += 1
        rec = build_pair(rng, idx)
        key = (rec.get("prompt") or "") + "\n" + (rec.get("chosen") or "")
        if key in seen:
            continue
        seen.add(key)
        rec["id"] = f"pref_{len(pairs):06d}"
        pairs.append(rec)
        idx += 1
    return pairs


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate DPO preference pairs")
    p.add_argument("--config", type=str, default="configs/data.yaml")
    p.add_argument("--output", type=str, default="data/raw/preference_raw.jsonl")
    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = _project_root()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (root / cfg_path).resolve()
    cfg = load_config(cfg_path)

    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    gen_cfg = cfg.get("generation") or {}
    n = args.num_samples
    if n is None:
        n = int(
            gen_cfg.get("preference_pairs")
            or (cfg.get("sizes") or {}).get("preference_raw")
            or 800
        )

    out = Path(args.output)
    if not out.is_absolute():
        out = (root / out).resolve()

    pairs = generate_pairs(n, seed=seed)
    write_jsonl(out, pairs)
    from collections import Counter

    rc = Counter((r.get("meta") or {}).get("reject_type") for r in pairs)
    cc = Counter(r["category"] for r in pairs)
    print(f"[generate_preference] wrote {len(pairs)} pairs -> {out}")
    print(f"[generate_preference] reject_types={dict(rc)}")
    print(f"[generate_preference] categories={dict(cc)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
