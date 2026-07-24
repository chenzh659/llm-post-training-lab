"""Chat engine for Gradio demo: mock or HF/PEFT generation + rule scoring.

Designed so unit tests can exercise mock path without torch/gradio.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evaluation.metrics import score_sample
from src.utils import format_chat, get_device, project_root, resolve_path

DEFAULT_SYSTEM = (
    "你是一名专业、礼貌的中文电商智能客服助手。"
    "请用清晰、可执行的建议回复用户，语气友好、不夸大承诺。"
    "不要编造用户未提供的订单号、物流单号或价格；缺少信息时请礼貌追问。"
    "涉及退换货、优惠规则时，说明一般政策并提示以订单页/平台规则为准。"
)

DEFAULT_BASE = "Qwen/Qwen2.5-0.5B-Instruct"

# Sidebar example prompts (Chinese e-commerce CS)
EXAMPLE_PROMPTS: list[tuple[str, str]] = [
    ("退换货", "客服您好，护眼台灯签收3天了，七天无理由怎么退？单号：ORD-2026-88421"),
    ("物流查询", "买的无线蓝牙耳机怎么还没发货？物流也不更新。"),
    ("优惠活动", "满200减30的券过期了，能补发吗？想买家用破壁机。"),
    ("支付问题", "支付失败怎么办？订单号是A202605288877。"),
    ("投诉建议", "订单PO-99001122，咖啡机和描述不符，包装还破了，怎么处理？"),
    ("缺少信息", "物流怎么还没更新？"),
]

# Keyword hints per rough category for mock + scoring
_CATEGORY_HINTS: dict[str, list[str]] = {
    "退换货": ["7天", "售后", "退货"],
    "物流查询": ["订单号", "物流", "轨迹"],
    "优惠活动": ["结算页", "有效期", "无法补发"],
    "支付问题": ["支付", "重试", "退回"],
    "投诉建议": ["歉意", "售后", "凭证"],
    "商品咨询": ["商品", "详情页", "规格"],
    "账户订单": ["订单", "账户", "登录"],
}


def infer_category(user_text: str) -> str:
    """Lightweight category guess from user message."""
    t = user_text or ""
    rules: list[tuple[str, tuple[str, ...]]] = [
        ("退换货", ("退货", "退换", "七天", "7天", "换货", "售后")),
        ("物流查询", ("物流", "发货", "快递", "运单", "轨迹", "未更新")),
        ("优惠活动", ("券", "优惠", "满减", "折扣", "活动")),
        ("支付问题", ("支付", "付款", "扣款", "银行卡")),
        ("投诉建议", ("投诉", "不满", "破损", "不符", "差评")),
        ("账户订单", ("账号", "登录", "密码", "订单取消")),
        ("商品咨询", ("多少钱", "规格", "参数", "有货", "颜色", "尺寸")),
    ]
    for cat, kws in rules:
        if any(k in t for k in kws):
            return cat
    return "商品咨询"


def extract_context_ids(text: str) -> str:
    """Keep order/tracking IDs visible in user text for hallucination checks."""
    return text or ""


def mock_reply(user_text: str, *, category: str | None = None) -> str:
    """Deterministic offline CS reply (no model weights).

    Intentionally imperfect on some prompts so the score panel stays useful:
    invents a tracking number when the user asks for a waybill without providing one.
    """
    user = (user_text or "").strip()
    cat = category or infer_category(user)

    # Hallucination demo path: user wants tracking but never gave order/waybill
    if re.search(r"运单号|快递单号", user) and not re.search(
        r"ORD|PO-|订单号|单号[：:]", user, re.I
    ):
        return (
            "您好。您的快递单号是 SF1234567890123，预计明天送达。"
            "祝您购物愉快！"
        )

    kws = _CATEGORY_HINTS.get(cat, ["订单页", "详情"])
    openers = {
        "退换货": (
            "您好，很高兴为您服务。关于退换货：一般支持签收后7天无理由退货"
            "（商品完好、配件齐全）。建议路径：我的订单 → 申请售后 → 选择退货。"
        ),
        "物流查询": (
            "您好，感谢您的咨询。请您提供订单号，我才能按单协助核对物流进度；"
            "您也可以在订单详情页直接查看轨迹。"
        ),
        "优惠活动": (
            "您好。优惠说明：券需在有效期内领取并在适用店铺/类目使用；"
            "过期券一般无法补发。是否可叠加以结算页最终明细为准。"
        ),
        "支付问题": (
            "您好，很高兴为您服务。支付失败常见原因：余额不足、银行风控、网络中断。"
            "建议更换支付方式重试；若已扣款未成单，一般会在1-3个工作日自动退回。"
        ),
        "投诉建议": (
            "您好。非常理解您的不满，给您带来不好的体验我们深表歉意。"
            "请补充问题类型与照片凭证，我将协助登记售后。"
        ),
        "商品咨询": (
            "您好。商品规格、库存与价格请以商品详情页实时展示为准；"
            "如需对比型号，可说明您关注的功能点。"
        ),
        "账户订单": (
            "您好。账户与订单相关操作请在「我的订单 / 账户安全」中完成；"
            "如遇登录异常可尝试找回密码或绑定手机号。"
        ),
    }
    body = openers.get(cat, openers["商品咨询"])
    # Surface order id if user provided one
    m = re.search(
        r"(?:订单号|单号)[是为：:\s]*([A-Za-z0-9\-]{6,})"
        r"|((?:ORD|PO|A)[-\w]{6,})",
        user,
        re.I,
    )
    if m:
        oid = m.group(1) or m.group(2)
        body += f" 已记录您提供的单号 {oid}。"
    body += " 请关注：" + "、".join(kws[:3]) + "。"
    body += " 如还有其他问题，随时告诉我。"
    return body


def score_reply(
    answer: str,
    user_text: str,
    *,
    expected_keywords: list[str] | None = None,
    reference: str | None = None,
    category: str | None = None,
) -> dict[str, Any]:
    """Run offline rule metrics; return flat summary for UI."""
    cat = category or infer_category(user_text)
    kws = expected_keywords or list(_CATEGORY_HINTS.get(cat, []))
    scored = score_sample(
        answer,
        context=extract_context_ids(user_text),
        reference=reference,
        expected_keywords=kws,
        must_not_contain=["百分百到货", "绝对能退", "我保证一定"],
    )
    fmt = scored.get("format") or {}
    kw = scored.get("keyword") or {}
    hall = scored.get("hallucination") or {}
    safe = scored.get("safety") or {}
    return {
        "composite": scored.get("composite"),
        "passed": scored.get("passed"),
        "format_score": fmt.get("score"),
        "keyword_hit_rate": kw.get("hit_rate"),
        "hallucination": hall.get("is_hallucination"),
        "invented_ids": hall.get("invented_ids") or [],
        "safety_passed": safe.get("passed"),
        "category": cat,
        "expected_keywords": kws,
        "raw": scored,
    }


def format_score_markdown(summary: dict[str, Any], *, latency_ms: float | None = None) -> str:
    """Human-readable score panel for Gradio Markdown."""
    comp = summary.get("composite")
    passed = summary.get("passed")
    lines = [
        "### 规则评测（离线）",
        "",
        f"- **场景推断**: `{summary.get('category')}`",
        f"- **Composite**: **{comp}**  ·  Pass: `{'✓' if passed else '✗'}`",
        f"- Format: `{summary.get('format_score')}`",
        f"- Keyword hit: `{summary.get('keyword_hit_rate')}`"
        f"  (期望: {', '.join(summary.get('expected_keywords') or []) or '—'})",
        f"- Hallucination: `{'⚠ 检出' if summary.get('hallucination') else 'clean'}`",
    ]
    inv = summary.get("invented_ids") or []
    if inv:
        lines.append(f"  - invented IDs: `{', '.join(map(str, inv))}`")
    lines.append(f"- Safety: `{'pass' if summary.get('safety_passed') else 'FAIL'}`")
    if latency_ms is not None:
        lines.append(f"- Latency: `{latency_ms:.0f} ms`")
    lines.append("")
    lines.append(
        "> 规则裁判强调礼貌结构、关键词、不编造单号/价格与安全禁词。"
        " 完整流水线见 `evaluation/metrics.py`。"
    )
    return "\n".join(lines)


@dataclass
class ModelBundle:
    """Loaded HF model + tokenizer (or mock)."""

    name: str
    mode: str  # mock | hf
    model: Any = None
    tokenizer: Any = None
    device: str = "cpu"
    base_model: str | None = None
    path: str | None = None
    load_error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def resolve_model_path(name: str, root: Path | None = None) -> Path | str:
    """Map short labels to paths / HF ids."""
    root = root or project_root()
    aliases = {
        "base": DEFAULT_BASE,
        "sft": "outputs/sft",
        "dpo": "outputs/dpo",
        "mock": "mock",
    }
    key = (name or "mock").strip().lower()
    target = aliases.get(key, name)
    if target == "mock":
        return "mock"
    p = Path(target)
    if not p.is_absolute():
        cand = root / target
        if cand.exists():
            return cand
    if p.exists():
        return p
    return target  # HF id or missing path (caller decides)


def load_model_bundle(
    model_choice: str,
    *,
    base_model: str | None = None,
    trust_remote_code: bool = True,
    torch_dtype: str = "auto",
    force_mock: bool = False,
) -> ModelBundle:
    """Load base / PEFT adapter, or return mock bundle."""
    root = project_root()
    choice = (model_choice or "mock").strip()
    if force_mock or choice.lower() in ("mock", "demo", ""):
        return ModelBundle(name="mock", mode="mock", meta={"note": "offline mock replies"})

    resolved = resolve_model_path(choice, root)
    if resolved == "mock":
        return ModelBundle(name="mock", mode="mock")

    path_str = str(resolved)
    # Missing local adapter → soft mock with warning
    if choice.lower() in ("sft", "dpo") or path_str.startswith(str(root)):
        p = Path(path_str)
        if p.is_dir() and not (p / "adapter_config.json").is_file() and not any(p.glob("*.safetensors")):
            return ModelBundle(
                name=choice,
                mode="mock",
                path=path_str,
                load_error=f"checkpoint not found at {path_str}; using mock replies",
            )

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:  # pragma: no cover
        return ModelBundle(
            name=choice,
            mode="mock",
            load_error=f"transformers/torch unavailable ({e}); mock mode",
        )

    dev = get_device()
    dtype = torch.float32
    if torch_dtype in ("bfloat16", "bf16") and dev == "cuda":
        dtype = torch.bfloat16
    elif torch_dtype in ("float16", "fp16", "half") and dev != "cpu":
        dtype = torch.float16
    elif torch_dtype == "auto" and dev == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    path = Path(path_str) if not isinstance(resolved, str) or Path(path_str).exists() else None
    try:
        if path is not None and path.is_dir() and (path / "adapter_config.json").is_file():
            from peft import PeftModel

            base = base_model
            if not base:
                try:
                    with (path / "adapter_config.json").open("r", encoding="utf-8") as f:
                        base = (json.load(f) or {}).get("base_model_name_or_path")
                except Exception:
                    base = None
            base = base or DEFAULT_BASE
            tok = AutoTokenizer.from_pretrained(base, trust_remote_code=trust_remote_code)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            base_m = AutoModelForCausalLM.from_pretrained(
                base, trust_remote_code=trust_remote_code, torch_dtype=dtype
            )
            model = PeftModel.from_pretrained(base_m, str(path))
            model.to(dev)
            model.eval()
            return ModelBundle(
                name=choice,
                mode="hf",
                model=model,
                tokenizer=tok,
                device=dev,
                base_model=base,
                path=str(path),
            )

        load_name = path_str if path is None or not path.exists() else str(path)
        # HF hub id when path missing
        if path is not None and not path.exists():
            load_name = choice if "/" in choice else DEFAULT_BASE
            if choice.lower() == "base":
                load_name = base_model or DEFAULT_BASE

        if choice.lower() == "base":
            load_name = base_model or DEFAULT_BASE

        tok = AutoTokenizer.from_pretrained(load_name, trust_remote_code=trust_remote_code)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            load_name, trust_remote_code=trust_remote_code, torch_dtype=dtype
        )
        model.to(dev)
        model.eval()
        return ModelBundle(
            name=choice,
            mode="hf",
            model=model,
            tokenizer=tok,
            device=dev,
            base_model=load_name,
            path=load_name,
        )
    except Exception as e:
        return ModelBundle(
            name=choice,
            mode="mock",
            path=path_str,
            load_error=f"load failed: {e}; using mock replies",
        )


def generate_reply(
    bundle: ModelBundle,
    user_text: str,
    *,
    history: list[dict[str, str]] | None = None,
    system: str = DEFAULT_SYSTEM,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> tuple[str, float]:
    """Return (reply_text, latency_ms)."""
    t0 = time.perf_counter()
    if bundle.mode != "hf" or bundle.model is None or bundle.tokenizer is None:
        text = mock_reply(user_text)
        return text, (time.perf_counter() - t0) * 1000.0

    import torch

    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    for turn in history or []:
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": str(content)})
    messages.append({"role": "user", "content": user_text})

    prompt = format_chat(messages, tokenizer=bundle.tokenizer, add_generation_prompt=True)
    inputs = bundle.tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(bundle.device) for k, v in inputs.items()}
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": bundle.tokenizer.pad_token_id,
        "eos_token_id": bundle.tokenizer.eos_token_id,
    }
    if temperature and temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=float(temperature), top_p=float(top_p))
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        out = bundle.model.generate(**inputs, **gen_kwargs)
    gen_ids = out[0][inputs["input_ids"].shape[-1] :]
    text = bundle.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return text, (time.perf_counter() - t0) * 1000.0


def history_to_messages(history: list[list[str]] | list[dict[str, str]]) -> list[dict[str, str]]:
    """Normalize Gradio chatbot history (tuples or messages) to role/content list."""
    out: list[dict[str, str]] = []
    if not history:
        return out
    first = history[0]
    if isinstance(first, dict):
        for m in history:  # type: ignore[assignment]
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            content = m.get("content") or m.get("text")
            if role and content is not None:
                out.append({"role": str(role), "content": str(content)})
        return out
    # [[user, assistant], ...]
    for pair in history:
        if not isinstance(pair, (list, tuple)) or len(pair) < 1:
            continue
        u, a = pair[0], pair[1] if len(pair) > 1 else None
        if u:
            out.append({"role": "user", "content": str(u)})
        if a:
            out.append({"role": "assistant", "content": str(a)})
    return out
