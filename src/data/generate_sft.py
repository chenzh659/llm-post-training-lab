"""Generate synthetic Chinese e-commerce CS SFT samples (chat messages).

No network required. Template + combinatorial expansion + light paraphrase slots.
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

# Default system prompt for Chinese e-commerce CS assistant
DEFAULT_SYSTEM = (
    "你是一名专业、礼貌的中文电商智能客服助手。"
    "请用清晰、可执行的建议回复用户，语气友好、不夸大承诺。"
    "不要编造用户未提供的订单号、物流单号或价格；缺少信息时请礼貌追问。"
    "涉及退换货、优惠规则时，说明一般政策并提示以订单页/平台规则为准。"
)

CATEGORIES = [
    "商品咨询",
    "物流查询",
    "退换货",
    "优惠活动",
    "投诉建议",
    "账户订单",
    "支付问题",
]

# Slot pools for combinatorial expansion
PRODUCTS = [
    "无线蓝牙耳机",
    "纯棉T恤",
    "智能手表",
    "便携充电宝",
    "家用破壁机",
    "儿童绘本套装",
    "机械键盘",
    "防晒霜SPF50",
    "跑步鞋",
    "保温杯",
    "笔记本电脑支架",
    "电动牙刷",
    "空气炸锅",
    "降噪耳塞",
    "运动外套",
    "手机壳",
    "路由器",
    "床上四件套",
    "咖啡机",
    "护眼台灯",
    "体脂秤",
    "无人机航拍器",
    "猫粮试吃装",
    "露营帐篷",
]

COLORS = ["黑色", "白色", "蓝色", "粉色", "灰色", "红色"]
SIZES = ["S", "M", "L", "XL", "均码", "42码"]
BRANDS = ["本店自营", "品牌联名", "官方旗舰"]

GREETINGS_USER = ["", "你好，", "在吗？", "客服您好，", "请问一下，"]
GREETINGS_ASST = [
    "您好，很高兴为您服务。",
    "您好，我是智能客服小助，很乐意为您解答。",
    "您好，感谢您的咨询。",
]
CLOSINGS = [
    "如还有其他问题，随时告诉我。",
    "祝您购物愉快！",
    "需要我继续帮您处理也可以直接说哦。",
    "感谢您的耐心，祝您生活愉快。",
]

# Paraphrase variants for common intents
USER_PARAPHRASE = {
    "stock": [
        "还有货吗？",
        "现在能下单吗？库存怎么样？",
        "这个会不会缺货？",
        "想买，不确定有没有现货。",
    ],
    "spec": [
        "参数是什么？",
        "能详细介绍下规格吗？",
        "材质/尺寸方面有说明吗？",
        "和详情页写的一致吗？关键参数帮我说说。",
    ],
    "compat": [
        "和我的手机兼容吗？",
        "能不能配我现在的设备用？",
        "兼容性如何？",
    ],
    "logistics": [
        "我的快递到哪了？",
        "物流怎么还没更新？",
        "什么时候能送到？",
        "包裹显示运输中，大概还要多久？",
    ],
    "return": [
        "怎么申请退货？",
        "不想要了，可以退吗？",
        "退换货流程是怎样的？",
        "七天无理由还能用吗？",
    ],
    "coupon": [
        "优惠券怎么用？",
        "满减活动叠不了券吗？",
        "现在有什么活动？",
        "券过期了还能补吗？",
    ],
    "complaint": [
        "收到货有问题，我要投诉。",
        "服务态度太差了。",
        "商品和描述不符，怎么处理？",
        "物流暴力分拣，包装都破了。",
    ],
    "order": [
        "怎么修改收货地址？",
        "订单可以取消吗？",
        "我想查一下历史订单。",
        "账号绑定的手机号怎么换？",
    ],
    "pay": [
        "支付失败怎么办？",
        "重复扣款了怎么退？",
        "可以用花呗吗？",
        "发票怎么开？",
    ],
}

# Synthetic order IDs only used when user message includes them
ORDER_IDS = [
    "SYN2026031500123",
    "ORD-2026-88421",
    "EC2026070100991",
    "A202605288877",
    "PO-77881234",
    "SYN2026042200456",
    "ORD-2026-11002",
    "EC2026081500333",
    "B202603019988",
    "PO-99001122",
    "JD-CS-20260718-01",
    "TB2026050507788",
]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: Path | None) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "seed": 42,
        "sizes": {"sft_raw": 2000},
        "generation": {
            "sft_samples": 2000,
            "min_per_category": 50,
        },
        "paths": {
            "raw_dir": "data/raw",
        },
    }
    if path is None or not path.is_file():
        return defaults
    if yaml is None:
        raise RuntimeError("PyYAML is required to load config; pip install pyyaml")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    # shallow merge
    for k, v in defaults.items():
        if k not in cfg:
            cfg[k] = v
        elif isinstance(v, dict) and isinstance(cfg.get(k), dict):
            merged = dict(v)
            merged.update(cfg[k])
            cfg[k] = merged
    return cfg


def _pick(rng: random.Random, items: list[str]) -> str:
    return items[rng.randrange(len(items))]


def _product_phrase(rng: random.Random) -> str:
    p = _pick(rng, PRODUCTS)
    if rng.random() < 0.4:
        p = f"{_pick(rng, COLORS)}{_pick(rng, SIZES)}的{p}" if "T恤" in p or "鞋" in p else f"{_pick(rng, COLORS)}{p}"
    if rng.random() < 0.25:
        p = f"{_pick(rng, BRANDS)}{p}"
    return p


def _with_order(rng: random.Random, text: str) -> tuple[str, str | None]:
    """Optionally inject a synthetic order id into the user text."""
    if rng.random() < 0.45:
        oid = _pick(rng, ORDER_IDS)
        variants = [
            f"{text} 订单号是{oid}。",
            f"{text} 单号：{oid}",
            f"订单{oid}，{text}",
        ]
        return _pick(rng, variants), oid
    return text, None


def _asst(rng: random.Random, body: str) -> str:
    g = _pick(rng, GREETINGS_ASST) if rng.random() < 0.7 else "您好。"
    c = _pick(rng, CLOSINGS) if rng.random() < 0.65 else ""
    parts = [g, body.strip()]
    if c:
        parts.append(c)
    return "".join(parts) if not body.startswith("您好") else (body.strip() + (c and f"{c}" or ""))


# --- Category-specific templates ---

def gen_product(rng: random.Random) -> dict[str, Any]:
    product = _product_phrase(rng)
    intent = _pick(rng, ["stock", "spec", "compat"])
    u_q = _pick(rng, USER_PARAPHRASE[intent])
    user = f"{_pick(rng, GREETINGS_USER)}想了解{product}，{u_q}".strip()
    if intent == "stock":
        body = (
            f"关于【{product}】：当前详情页显示为可售状态时一般可正常下单；"
            f"具体库存以您下单时页面为准。若提示无货，可点“到货通知”或看看相近规格。"
            f"需要我帮您对照颜色/尺码建议也可以说明您的偏好。"
        )
    elif intent == "spec":
        body = (
            f"【{product}】的规格请以商品详情页参数表为准（材质、尺寸、适用人群等）。"
            f"若您关心某一点（例如是否防水、是否含充电器），请告诉我，我按详情要点为您梳理。"
            f"我无法编造详情页未写明的参数。"
        )
    else:
        body = (
            f"兼容性方面，建议您核对【{product}】详情页“适配说明/包装清单”。"
            f"若您补充设备型号，我可以帮您对照常见适配要点；最终以官方说明与到货实测为准。"
        )
    return _pack("商品咨询", user, _asst(rng, body), product=product, intent=intent)


def gen_logistics(rng: random.Random) -> dict[str, Any]:
    product = _product_phrase(rng)
    u_q = _pick(rng, USER_PARAPHRASE["logistics"])
    user_base = f"{_pick(rng, GREETINGS_USER)}买的{product}，{u_q}".strip()
    user, oid = _with_order(rng, user_base)
    if oid:
        body = (
            f"已收到您提供的订单号 {oid}。"
            f"我这边无法实时连接物流系统时，请您打开“我的订单 → 查看物流”核对最新节点；"
            f"若长时间无更新，可备注异常后申请催派或联系承运商。"
            f"请您放心，我不会编造不存在的运单轨迹。"
            f"如需改地址，需在发货前或按物流可改址规则操作。"
        )
    else:
        body = (
            f"很抱歉让您久等了。请您提供订单号，我才能按单协助核对物流进度；"
            f"您也可以在订单详情页直接查看轨迹。"
            f"若显示“运输中”但多日未动，可提供单号后我们帮您登记催查。"
        )
    return _pack("物流查询", user, _asst(rng, body), product=product, order_id=oid)


def gen_return(rng: random.Random) -> dict[str, Any]:
    product = _product_phrase(rng)
    u_q = _pick(rng, USER_PARAPHRASE["return"])
    reason = _pick(
        rng,
        ["尺码不合适", "不喜欢颜色", "质量问题", "发错货", "七天无理由"],
    )
    user_base = f"{_pick(rng, GREETINGS_USER)}{product}，{u_q} 原因是{reason}。".strip()
    user, oid = _with_order(rng, user_base)
    oid_part = f"订单 {oid} " if oid else "您的订单"
    body = (
        f"关于{oid_part}的退换货：一般支持签收后7天无理由（商品完好、配件齐全、不影响二次销售），"
        f"质量问题可走售后质检。"
        f"建议路径：我的订单 → 申请售后 → 选择退货/换货并上传凭证。"
        f"运费承担规则以售后页说明为准；质量问题通常由商家侧承担。"
        f"请勿在未创建售后单前自行丢弃包装，以便核对。"
    )
    return _pack("退换货", user, _asst(rng, body), product=product, order_id=oid, reason=reason)


def gen_promo(rng: random.Random) -> dict[str, Any]:
    product = _product_phrase(rng)
    u_q = _pick(rng, USER_PARAPHRASE["coupon"])
    user = f"{_pick(rng, GREETINGS_USER)}关于{product}，{u_q}".strip()
    body = (
        f"优惠说明：店铺券、平台券与满减是否可叠加，以结算页最终优惠明细为准；"
        f"活动商品、特殊价商品有时会排除用券。"
        f"券需在有效期内领取并在适用店铺/类目使用；过期券一般无法补发。"
        f"若结算时券不可用，请检查门槛金额、适用商品与互斥规则。"
        f"我不会承诺结算页未展示的额外折扣。"
    )
    return _pack("优惠活动", user, _asst(rng, body), product=product)


def gen_complaint(rng: random.Random) -> dict[str, Any]:
    product = _product_phrase(rng)
    u_q = _pick(rng, USER_PARAPHRASE["complaint"])
    user_base = f"{_pick(rng, GREETINGS_USER)}{product}，{u_q}".strip()
    user, oid = _with_order(rng, user_base)
    oid_part = f"（订单 {oid}）" if oid else ""
    body = (
        f"非常理解您的不满，给您带来不好的体验我们深表歉意{oid_part}。"
        f"请您补充问题类型（破损/错发/态度/延误等）与照片凭证，"
        f"我将协助登记售后或升级专员跟进：可退换、补发配件或按规则协商补偿。"
        f"处理进度以售后工单为准；请您保留包裹与聊天记录。"
        f"我们会认真对待您的建议，持续改进服务。"
    )
    return _pack("投诉建议", user, _asst(rng, body), product=product, order_id=oid)


def gen_account_order(rng: random.Random) -> dict[str, Any]:
    u_q = _pick(rng, USER_PARAPHRASE["order"])
    user_base = f"{_pick(rng, GREETINGS_USER)}{u_q}".strip()
    user, oid = _with_order(rng, user_base)
    if "地址" in user:
        body = (
            "修改收货地址：若订单尚未发货，一般可在订单详情中申请修改，或联系客服协助备注；"
            "已发货后需承运商支持改址，可能产生费用且不保证成功。"
            "请提供订单号与正确省市区+详细地址+联系人电话以便核对。"
            "为保护隐私，请勿在公开场合泄露完整证件号。"
        )
    elif "取消" in user:
        body = (
            f"取消订单：未发货订单通常可在订单页直接取消；"
            f"{'您提供的 ' + oid + ' 请以订单页状态为准。' if oid else '请打开对应订单查看状态。'}"
            f"已发货需拒收或签收后走退货。支付成功但未发货的退款原路返回，到账时间视支付渠道而定。"
        )
    elif "手机" in user or "账号" in user:
        body = (
            "更换绑定手机：请在“账户设置 → 账号与安全”按页面验证流程操作；"
            "需原手机验证码或申诉材料。客服无法在未验证身份时直接改绑，以保障账户安全。"
            "若手机已丢失，请走账号申诉并准备订单/实名相关凭证。"
        )
    else:
        body = (
            "历史订单可在“我的订单”按时间/状态筛选查看；"
            "如需开票或售后，请进入对应订单操作。"
            "我无法调取您未登录会话外的隐私订单列表；请在App内确认。"
        )
    return _pack("账户订单", user, _asst(rng, body), order_id=oid)


def gen_payment(rng: random.Random) -> dict[str, Any]:
    u_q = _pick(rng, USER_PARAPHRASE["pay"])
    user_base = f"{_pick(rng, GREETINGS_USER)}{u_q}".strip()
    user, oid = _with_order(rng, user_base)
    if "失败" in user:
        body = (
            "支付失败常见原因：余额不足、银行风控、网络中断、超时未确认。"
            "建议：更换支付方式重试、检查银行卡限额、稍后重试；"
            "若提示已扣款但订单未生成，请保留支付凭证与流水号，一般会在1-3个工作日自动退回，"
            "超时未到账可提供流水我们协助核查。"
            "请勿重复高频点击支付以免多次预授权。"
        )
    elif "重复" in user or "扣款" in user:
        body = (
            f"关于疑似重复扣款{'（订单 ' + oid + '）' if oid else ''}："
            f"请提供支付流水号/到账截图。若确认为重复扣款，我们协助提交原路退款申请；"
            f"到账时间因微信/支付宝/银行卡而异。"
            f"请暂勿继续支付同一订单，以免再次重复。"
        )
    elif "花呗" in user or "分期" in user:
        body = (
            "是否支持花呗/信用卡分期以结算页展示的支付方式为准；"
            "部分活动价或虚拟商品可能不支持。费率与期数由支付机构展示，请在支付前确认。"
        )
    else:
        body = (
            "发票：请在订单详情申请电子普通发票或按页面选择抬头类型（个人/单位）；"
            "单位抬头需填写正确税号。开具后可在发票中心下载。"
            "若超过可开票时效，请提供订单号由专员确认。"
        )
    return _pack("支付问题", user, _asst(rng, body), order_id=oid)


def _pack(
    category: str,
    user: str,
    assistant: str,
    **extra: Any,
) -> dict[str, Any]:
    meta = {k: v for k, v in extra.items() if v is not None}
    return {
        "category": category,
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "meta": meta,
    }


GENERATORS = {
    "商品咨询": gen_product,
    "物流查询": gen_logistics,
    "退换货": gen_return,
    "优惠活动": gen_promo,
    "投诉建议": gen_complaint,
    "账户订单": gen_account_order,
    "支付问题": gen_payment,
}


def generate_samples(n: int, seed: int = 42, min_per_category: int = 50) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    cats = list(CATEGORIES)
    # ensure coverage
    target_floor = min(min_per_category, max(1, n // len(cats)))
    samples: list[dict[str, Any]] = []
    for cat in cats:
        gen = GENERATORS[cat]
        for _ in range(target_floor):
            rec = gen(rng)
            samples.append(rec)
    # fill remaining with weighted random
    while len(samples) < n:
        cat = _pick(rng, cats)
        samples.append(GENERATORS[cat](rng))
    rng.shuffle(samples)
    samples = samples[:n]
    # assign ids
    for i, rec in enumerate(samples):
        rec["id"] = f"sft_{i:06d}"
        rec["scenario"] = rec["category"]
    return samples


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Chinese e-commerce CS SFT data")
    p.add_argument(
        "--config",
        type=str,
        default="configs/data.yaml",
        help="Path to data config YAML",
    )
    p.add_argument(
        "--output",
        type=str,
        default="data/raw/sft_raw.jsonl",
        help="Output JSONL path",
    )
    p.add_argument("--num-samples", type=int, default=None, help="Override sample count")
    p.add_argument("--seed", type=int, default=None, help="Override RNG seed")
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
        n = int(gen_cfg.get("sft_samples") or (cfg.get("sizes") or {}).get("sft_raw") or 2000)
    min_per = int(gen_cfg.get("min_per_category", 50))

    out = Path(args.output)
    if not out.is_absolute():
        out = (root / out).resolve()

    samples = generate_samples(n, seed=seed, min_per_category=min_per)
    write_jsonl(out, samples)
    # category counts
    from collections import Counter

    counts = Counter(r["category"] for r in samples)
    print(f"[generate_sft] wrote {len(samples)} samples -> {out}")
    print(f"[generate_sft] seed={seed} category_counts={dict(counts)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
