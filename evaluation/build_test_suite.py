"""Build / load evaluation test suites with gold keyword fields.

Sources (in order of preference when resolving):
1. ``evaluation/fixtures/sample_test.jsonl`` — handcrafted offline demo
2. ``data/splits/test.sft.jsonl`` enriched with category keyword maps
3. paths from ``configs/eval.yaml``

Each item is normalized to::

    {
      "id": str,
      "category": str,
      "messages": [...],          # chat history ending before assistant (or full)
      "user": str,
      "gold": str | null,         # reference assistant reply if available
      "context": str,             # user (+ system) for hallucination checks
      "expected_keywords": [str],
      "must_not_contain": [str],
      "gold_mc": str | null,
      "choices": [str] | null,
      "meta": dict,
    }
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils import (  # noqa: E402
    ensure_dir,
    project_root,
    read_jsonl,
    resolve_path,
    write_jsonl,
)

# Category → expected policy keywords for Chinese e-commerce CS
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "商品咨询": ["详情页", "规格"],
    "物流查询": ["订单号", "物流"],
    "退换货": ["7天", "售后"],
    "优惠活动": ["结算页", "有效期"],
    "投诉建议": ["歉意", "售后"],
    "账户订单": ["订单"],
    "支付问题": ["支付", "退回"],
}

# Soft synonyms: if any synonym present, treat keyword as hit via expansion at score time
CATEGORY_KEYWORD_ALTS: dict[str, list[str]] = {
    "退换货": ["七天无理由", "申请售后", "退货", "换货"],
    "优惠活动": ["优惠", "优惠券", "满减", "不可补发"],
    "投诉建议": ["抱歉", "理解", "登记", "凭证"],
    "物流查询": ["轨迹", "催查", "运输中", "订单详情"],
    "支付问题": ["银行", "流水", "重试", "发票", "花呗"],
    "商品咨询": ["库存", "参数", "下单", "材质"],
    "账户订单": ["取消", "地址", "绑定", "我的订单"],
}

DEFAULT_MUST_NOT: list[str] = [
    "百分百到货",
    "我保证一定",
    "绝对能退",
]


def extract_user_and_gold(messages: list[dict[str, Any]]) -> tuple[str, str | None, str]:
    """Return (user_text, gold_assistant, context_for_hallucination)."""
    user_parts: list[str] = []
    gold: str | None = None
    ctx_parts: list[str] = []
    for m in messages:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if role == "system":
            ctx_parts.append(content)
        elif role == "user":
            user_parts.append(content)
            ctx_parts.append(content)
        elif role == "assistant":
            gold = content
        elif role == "tool":
            ctx_parts.append(content)
    user = "\n".join(user_parts)
    context = "\n".join(ctx_parts)
    return user, gold, context


def keywords_for_category(category: str, gold: str | None = None) -> list[str]:
    """Pick expected keywords from category map, filtered to those that appear in gold if gold exists."""
    cat = category or ""
    base = list(CATEGORY_KEYWORDS.get(cat, []))
    alts = list(CATEGORY_KEYWORD_ALTS.get(cat, []))
    # Prefer keywords that gold actually uses when available
    if gold:
        present = [k for k in base + alts if k in gold]
        if present:
            # keep unique, prefer base order then alts
            seen: set[str] = set()
            out: list[str] = []
            for k in base + alts:
                if k in present and k not in seen:
                    seen.add(k)
                    out.append(k)
            return out[:4]
    # Fallback: primary category keywords + first alt
    out = base[:]
    for a in alts[:2]:
        if a not in out:
            out.append(a)
    return out[:4]


def normalize_item(raw: dict[str, Any], idx: int = 0) -> dict[str, Any]:
    """Normalize a raw JSONL row into a standard eval item."""
    messages = raw.get("messages")
    if isinstance(messages, list) and messages:
        user, gold, context = extract_user_and_gold(messages)
        # Prompt messages: drop final assistant for generation
        prompt_messages = [m for m in messages if m.get("role") != "assistant"]
        # If last was assistant we already dropped; if multi-turn keep all but last asst
        if messages and messages[-1].get("role") == "assistant":
            prompt_messages = messages[:-1]
    else:
        user = (raw.get("user") or raw.get("prompt") or raw.get("instruction") or "").strip()
        gold = raw.get("gold") or raw.get("output") or raw.get("response") or raw.get("assistant")
        if gold is not None:
            gold = str(gold).strip()
        system = (raw.get("system") or "").strip()
        context = "\n".join(x for x in (system, user) if x)
        prompt_messages = []
        if system:
            prompt_messages.append({"role": "system", "content": system})
        if user:
            prompt_messages.append({"role": "user", "content": user})

    category = raw.get("category") or raw.get("scenario") or "通用"
    item_id = str(raw.get("id") or f"eval_{idx:04d}")

    expected = raw.get("expected_keywords")
    if not expected:
        expected = keywords_for_category(str(category), gold if isinstance(gold, str) else None)
    must_not = raw.get("must_not_contain") or list(DEFAULT_MUST_NOT)

    # Order id from meta should be in context for hallucination fairness
    meta = dict(raw.get("meta") or {})
    if meta.get("order_id") and meta["order_id"] not in context:
        context = f"{context}\n订单号 {meta['order_id']}".strip()

    return {
        "id": item_id,
        "category": category,
        "messages": prompt_messages,
        "user": user,
        "gold": gold,
        "context": context,
        "expected_keywords": list(expected),
        "must_not_contain": list(must_not),
        "gold_mc": raw.get("gold_mc") or raw.get("answer"),
        "choices": raw.get("choices"),
        "meta": meta,
        "scenario": raw.get("scenario") or category,
    }


def load_suite(
    path: str | Path | None = None,
    *,
    max_samples: int | None = None,
    prefer_fixture: bool = False,
) -> list[dict[str, Any]]:
    """Load and normalize a test suite."""
    root = project_root()
    candidates: list[Path] = []
    if path:
        candidates.append(resolve_path(path, root))
    if prefer_fixture or not path:
        candidates.append(root / "evaluation" / "fixtures" / "sample_test.jsonl")
    candidates.extend(
        [
            root / "data" / "eval" / "test_suite.jsonl",
            root / "data" / "splits" / "test.sft.jsonl",
            root / "data" / "processed" / "sft_eval.jsonl",
        ]
    )

    chosen: Path | None = None
    for c in candidates:
        if c.is_file():
            chosen = c
            break
    if chosen is None:
        raise FileNotFoundError(
            "No test suite found. Tried:\n  "
            + "\n  ".join(str(c) for c in candidates)
            + "\nCreate evaluation/fixtures/sample_test.jsonl or run data pipeline."
        )

    rows = read_jsonl(chosen)
    items = [normalize_item(r, i) for i, r in enumerate(rows)]
    if max_samples is not None and max_samples > 0:
        items = items[: int(max_samples)]
    # attach source path for reports
    for it in items:
        it["_source"] = str(chosen.relative_to(root)) if chosen.is_relative_to(root) else str(chosen)
    return items


def build_from_sft_split(
    sft_path: str | Path | None = None,
    out_path: str | Path | None = None,
    *,
    max_samples: int | None = None,
) -> Path:
    """Enrich SFT test split and write ``data/eval/test_suite.jsonl``."""
    root = project_root()
    src = resolve_path(sft_path or "data/splits/test.sft.jsonl", root)
    if not src.is_file():
        raise FileNotFoundError(f"SFT test split not found: {src}")
    items = load_suite(src, max_samples=max_samples)
    # Serialize without internal keys that are huge duplicates
    out_rows: list[dict[str, Any]] = []
    for it in items:
        out_rows.append(
            {
                "id": it["id"],
                "category": it["category"],
                "scenario": it.get("scenario"),
                "messages": it["messages"]
                + (
                    [{"role": "assistant", "content": it["gold"]}]
                    if it.get("gold")
                    else []
                ),
                "expected_keywords": it["expected_keywords"],
                "must_not_contain": it["must_not_contain"],
                "gold_mc": it.get("gold_mc"),
                "choices": it.get("choices"),
                "meta": it.get("meta") or {},
            }
        )
    dest = resolve_path(out_path or "data/eval/test_suite.jsonl", root)
    write_jsonl(dest, out_rows)
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build eval test suite with gold keyword fields")
    parser.add_argument("--sft-path", type=str, default="data/splits/test.sft.jsonl")
    parser.add_argument("--out", type=str, default="data/eval/test_suite.jsonl")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--from-fixture",
        action="store_true",
        help="Copy/normalize evaluation/fixtures/sample_test.jsonl to --out",
    )
    args = parser.parse_args(argv)

    root = project_root()
    if args.from_fixture:
        items = load_suite(
            root / "evaluation" / "fixtures" / "sample_test.jsonl",
            max_samples=args.max_samples,
            prefer_fixture=True,
        )
        out_rows = []
        for it in items:
            out_rows.append(
                {
                    "id": it["id"],
                    "category": it["category"],
                    "scenario": it.get("scenario"),
                    "messages": it["messages"]
                    + (
                        [{"role": "assistant", "content": it["gold"]}]
                        if it.get("gold")
                        else []
                    ),
                    "user": it["user"],
                    "gold": it.get("gold"),
                    "expected_keywords": it["expected_keywords"],
                    "must_not_contain": it["must_not_contain"],
                    "gold_mc": it.get("gold_mc"),
                    "choices": it.get("choices"),
                    "meta": it.get("meta") or {},
                }
            )
        dest = resolve_path(args.out, root)
        write_jsonl(dest, out_rows)
        print(f"Wrote {len(out_rows)} items -> {dest}")
        return 0

    dest = build_from_sft_split(args.sft_path, args.out, max_samples=args.max_samples)
    n = len(read_jsonl(dest))
    print(f"Wrote {n} items -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
