"""Error analysis taxonomy for Chinese e-commerce CS model outputs.

Classes
-------
幻觉
    Invented order IDs / tracking numbers / prices not in context.
政策错误
    Contradicts policy keywords or asserts absolute refunds/discounts.
格式违规
    Empty, too short, missing basic CS structure.
态度不当
    Banned / rude / unprofessional phrases.
答非所问
    Low lexical overlap with user intent / category keywords entirely missing.
信息缺失
    Missing expected keywords while not fully off-topic.

Writes ``reports/error_analysis.json`` and ``reports/error_analysis.md``.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluation.build_test_suite import load_suite  # noqa: E402
from evaluation.metrics import (  # noqa: E402
    char_f1,
    format_compliance,
    hallucination_heuristic,
    keyword_hit,
    safety_score,
)
from src.utils import (  # noqa: E402
    load_json,
    project_root,
    read_jsonl,
    resolve_path,
    save_json,
)

ERROR_LABELS = (
    "幻觉",
    "政策错误",
    "格式违规",
    "态度不当",
    "答非所问",
    "信息缺失",
)

POLICY_BAD_ASSERTIONS = (
    "绝对能退",
    "一定能退",
    "百分百",
    "我保证一定",
    "必须给您退",
    "肯定补发",
    "一定补货",
    "明天一定到",
    "立刻到账保证",
)


def classify_errors(
    answer: str,
    *,
    user: str = "",
    context: str = "",
    expected_keywords: list[str] | None = None,
    must_not_contain: list[str] | None = None,
    gold: str | None = None,
) -> dict[str, Any]:
    """Return multi-label error classification for one answer."""
    text = (answer or "").strip()
    labels: list[str] = []
    reasons: list[str] = []

    fmt = format_compliance(text)
    hall = hallucination_heuristic(text, context or user)
    safe = safety_score(text)
    kw = keyword_hit(text, expected_keywords, must_not_contain=must_not_contain)

    if not fmt["checks"]["non_empty"] or not fmt["passed"]:
        labels.append("格式违规")
        reasons.append("回复过短/空白或结构不合格")

    if hall.get("is_hallucination"):
        labels.append("幻觉")
        inv = hall.get("invented_ids") or []
        prices = hall.get("invented_prices") or []
        reasons.append(f"编造标识: ids={inv} prices={prices}")

    if not safe["passed"]:
        labels.append("态度不当")
        reasons.append(f"命中不当/禁用表达: {safe.get('banned_hits')}")

    # Policy errors: forbidden must_not or absolute policy claims
    policy_hits = [p for p in POLICY_BAD_ASSERTIONS if p in text]
    forbid_hits = kw.get("forbidden_hits") or []
    if policy_hits or forbid_hits:
        if "政策错误" not in labels:
            labels.append("政策错误")
        reasons.append(f"不当政策承诺/禁语: {policy_hits or forbid_hits}")

    # Off-topic vs missing info
    # NOTE: do not use ``x or default`` on hit_rate — 0.0 is a valid value.
    missing = kw.get("missing") or []
    n_exp = int(kw.get("n_expected") or 0)
    hit_rate_raw = kw.get("hit_rate")
    hit_rate = float(hit_rate_raw) if hit_rate_raw is not None else 1.0
    user_overlap = char_f1(text, user) if user else 0.0

    if n_exp > 0 and hit_rate == 0.0 and user_overlap < 0.08:
        labels.append("答非所问")
        reasons.append("未命中任何期望关键词且与用户表述重叠极低")
    elif n_exp > 0 and missing and hit_rate < 0.5:
        labels.append("信息缺失")
        reasons.append(f"缺少关键要点: {missing}")

    # Gold-based off-topic if provided
    if gold and "答非所问" not in labels and "格式违规" not in labels:
        g_ov = char_f1(text, gold)
        if g_ov < 0.05 and hit_rate < 0.34 and len(text) > 10:
            # weak signal only if not already labeled
            if not labels:
                labels.append("答非所问")
                reasons.append(f"与参考回复字符重叠过低 ({g_ov:.3f})")

    if not labels:
        primary = None
        severity = "ok"
    else:
        # priority order for primary
        priority = ["态度不当", "幻觉", "政策错误", "答非所问", "格式违规", "信息缺失"]
        primary = next((p for p in priority if p in labels), labels[0])
        severity = "high" if primary in ("态度不当", "幻觉", "政策错误") else "medium"

    return {
        "labels": labels,
        "primary": primary,
        "severity": severity,
        "reasons": reasons,
        "signals": {
            "format_score": fmt["score"],
            "hallucination_score": hall["score"],
            "safety_score": safe["score"],
            "keyword_hit_rate": hit_rate,
            "user_char_f1": round(user_overlap, 4),
        },
        "is_error": bool(labels),
    }


def analyze_predictions(
    samples: list[dict[str, Any]],
    *,
    suite_items: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Analyze a list of {id, prediction/answer, ...} optionally joined with suite gold fields."""
    by_id = {it["id"]: it for it in (suite_items or [])}
    rows: list[dict[str, Any]] = []
    label_counter: Counter[str] = Counter()
    primary_counter: Counter[str] = Counter()
    cat_errors: dict[str, Counter[str]] = {}

    for s in samples:
        iid = str(s.get("id") or "")
        gold_item = by_id.get(iid, {})
        pred = (
            s.get("prediction")
            or s.get("answer")
            or s.get("output")
            or s.get("response")
            or ""
        )
        user = s.get("user") or gold_item.get("user") or ""
        context = s.get("context") or gold_item.get("context") or user
        exp = s.get("expected_keywords") or gold_item.get("expected_keywords")
        forbid = s.get("must_not_contain") or gold_item.get("must_not_contain")
        gold = s.get("gold") or gold_item.get("gold")
        category = s.get("category") or gold_item.get("category") or "未知"

        # If sample already has hallucination flags from score_sample, still re-classify
        clf = classify_errors(
            str(pred),
            user=str(user),
            context=str(context),
            expected_keywords=list(exp) if exp else None,
            must_not_contain=list(forbid) if forbid else None,
            gold=str(gold) if gold else None,
        )
        for lb in clf["labels"]:
            label_counter[lb] += 1
        if clf["primary"]:
            primary_counter[clf["primary"]] += 1
            cat_errors.setdefault(str(category), Counter())[clf["primary"]] += 1

        rows.append(
            {
                "id": iid,
                "category": category,
                "user": user,
                "prediction": pred,
                "gold": gold,
                **clf,
            }
        )

    n = len(rows)
    n_err = sum(1 for r in rows if r["is_error"])
    distribution = {
        lb: {"count": label_counter.get(lb, 0), "rate": round(label_counter.get(lb, 0) / n, 4) if n else 0.0}
        for lb in ERROR_LABELS
    }
    primary_dist = {
        lb: {"count": primary_counter.get(lb, 0), "rate": round(primary_counter.get(lb, 0) / n, 4) if n else 0.0}
        for lb in ERROR_LABELS
    }

    # representative examples (up to 2 per primary label)
    examples: dict[str, list[dict[str, Any]]] = {lb: [] for lb in ERROR_LABELS}
    for r in rows:
        p = r.get("primary")
        if p and len(examples[p]) < 2:
            examples[p].append(
                {
                    "id": r["id"],
                    "category": r["category"],
                    "user": (r.get("user") or "")[:200],
                    "prediction": (r.get("prediction") or "")[:300],
                    "reasons": r.get("reasons"),
                }
            )

    return {
        "n": n,
        "n_error": n_err,
        "error_rate": round(n_err / n, 4) if n else 0.0,
        "label_distribution": distribution,
        "primary_distribution": primary_dist,
        "by_category_primary": {
            cat: dict(cnt) for cat, cnt in sorted(cat_errors.items())
        },
        "examples": examples,
        "items": rows,
    }


def to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Error Analysis Report",
        "",
        f"- Samples: **{report.get('n', 0)}**",
        f"- Items with ≥1 error label: **{report.get('n_error', 0)}** "
        f"(rate {report.get('error_rate', 0):.2%})",
        f"- Model / source: `{report.get('source', 'n/a')}`",
        "",
        "## Primary error distribution",
        "",
        "| Error | Count | Rate |",
        "|-------|------:|-----:|",
    ]
    for lb in ERROR_LABELS:
        d = (report.get("primary_distribution") or {}).get(lb) or {}
        lines.append(f"| {lb} | {d.get('count', 0)} | {d.get('rate', 0):.2%} |")

    lines += ["", "## Multi-label distribution", "", "| Error | Count | Rate |", "|-------|------:|-----:|"]
    for lb in ERROR_LABELS:
        d = (report.get("label_distribution") or {}).get(lb) or {}
        lines.append(f"| {lb} | {d.get('count', 0)} | {d.get('rate', 0):.2%} |")

    by_cat = report.get("by_category_primary") or {}
    if by_cat:
        lines += ["", "## Primary errors by category", ""]
        for cat, cnt in by_cat.items():
            parts = ", ".join(f"{k}:{v}" for k, v in sorted(cnt.items(), key=lambda x: -x[1]))
            lines.append(f"- **{cat}**: {parts}")

    examples = report.get("examples") or {}
    lines += ["", "## Representative examples", ""]
    for lb in ERROR_LABELS:
        exs = examples.get(lb) or []
        if not exs:
            continue
        lines.append(f"### {lb}")
        for ex in exs:
            lines.append(f"- `{ex.get('id')}` ({ex.get('category')})")
            lines.append(f"  - user: {ex.get('user')}")
            lines.append(f"  - pred: {ex.get('prediction')}")
            lines.append(f"  - reasons: {ex.get('reasons')}")
        lines.append("")

    lines += [
        "## Taxonomy notes",
        "",
        "| Label | Detection heuristic |",
        "|-------|---------------------|",
        "| 幻觉 | Order/tracking IDs or prices in answer not present in context |",
        "| 政策错误 | Absolute refund/discount promises or must_not_contain hits |",
        "| 格式违规 | Empty/too short or failed format_compliance |",
        "| 态度不当 | Banned / abusive / unprofessional phrases |",
        "| 答非所问 | Zero keyword hits and very low overlap with user text |",
        "| 信息缺失 | Partial keyword coverage (<50%) without full off-topic |",
        "",
    ]
    return "\n".join(lines) + "\n"


def run_error_analysis(
    *,
    predictions_path: str | Path | None = None,
    zero_shot_path: str | Path | None = None,
    test_path: str | Path | None = None,
    out_json: str | Path | None = None,
    out_md: str | Path | None = None,
    max_samples: int | None = None,
    prefer_fixture: bool = True,
    model_label: str = "unknown",
) -> dict[str, Any]:
    root = project_root()
    suite = load_suite(test_path, max_samples=max_samples, prefer_fixture=prefer_fixture)

    samples: list[dict[str, Any]] = []
    source = model_label

    # Prefer explicit predictions jsonl
    if predictions_path:
        p = resolve_path(predictions_path, root)
        samples = read_jsonl(p)
        source = str(p)
    else:
        zs = resolve_path(zero_shot_path or "reports/zero_shot_results.json", root)
        if zs.is_file():
            data = load_json(zs)
            samples = data.get("samples") or []
            source = data.get("model_name") or str(zs)
        else:
            # Score gold answers as a clean baseline (mostly zero errors)
            samples = [
                {
                    "id": it["id"],
                    "category": it.get("category"),
                    "user": it.get("user"),
                    "prediction": it.get("gold") or "",
                    "gold": it.get("gold"),
                    "context": it.get("context"),
                    "expected_keywords": it.get("expected_keywords"),
                    "must_not_contain": it.get("must_not_contain"),
                }
                for it in suite
            ]
            source = "gold_reference_fallback"

    # Inject mock errorful samples if analyzing gold-only and empty? keep as-is.

    analysis = analyze_predictions(samples, suite_items=suite)
    report = {
        "task": "error_analysis",
        "source": source,
        "test_source": suite[0].get("_source") if suite else None,
        **analysis,
    }

    jpath = resolve_path(out_json or "reports/error_analysis.json", root)
    mpath = resolve_path(out_md or "reports/error_analysis.md", root)
    save_json(jpath, report)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(to_markdown(report), encoding="utf-8")
    report["_out_json"] = str(jpath)
    report["_out_md"] = str(mpath)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Classify CS reply errors into taxonomy")
    parser.add_argument("--predictions", type=str, default=None, help="JSONL id+prediction")
    parser.add_argument(
        "--from-zero-shot",
        type=str,
        default="reports/zero_shot_results.json",
        help="zero_shot_results.json path",
    )
    parser.add_argument("--test-path", type=str, default=None)
    parser.add_argument("--out-json", type=str, default="reports/error_analysis.json")
    parser.add_argument("--out-md", type=str, default="reports/error_analysis.md")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-fixture", action="store_true")
    parser.add_argument(
        "--demo-errors",
        action="store_true",
        help="Analyze built-in bad replies for taxonomy demo without model outputs",
    )
    args = parser.parse_args(argv)

    root = project_root()
    print("=" * 60)
    print("llm-post-training-lab :: error analysis")
    print("=" * 60)

    if args.demo_errors:
        suite = load_suite(args.test_path, max_samples=args.max_samples, prefer_fixture=not args.no_fixture)
        bad = []
        for i, it in enumerate(suite):
            if i % 4 == 0:
                pred = "滚，自己查去。运单号SF9988776655443，明天一定到。"
            elif i % 4 == 1:
                pred = "好的。"  # format
            elif i % 4 == 2:
                pred = "今天天气不错，推荐您去旅游。"  # off topic
            else:
                pred = "您好。可以退，绝对能退，我保证一定给您退款。祝您购物愉快！"
            bad.append(
                {
                    "id": it["id"],
                    "category": it.get("category"),
                    "user": it.get("user"),
                    "prediction": pred,
                    "gold": it.get("gold"),
                    "context": it.get("context"),
                    "expected_keywords": it.get("expected_keywords"),
                    "must_not_contain": it.get("must_not_contain"),
                }
            )
        analysis = analyze_predictions(bad, suite_items=suite)
        report = {
            "task": "error_analysis",
            "source": "demo_errors",
            "test_source": suite[0].get("_source") if suite else None,
            **analysis,
        }
        jpath = resolve_path(args.out_json, root)
        mpath = resolve_path(args.out_md, root)
        save_json(jpath, report)
        mpath.parent.mkdir(parents=True, exist_ok=True)
        mpath.write_text(to_markdown(report), encoding="utf-8")
        report["_out_json"] = str(jpath)
        report["_out_md"] = str(mpath)
    else:
        report = run_error_analysis(
            predictions_path=args.predictions,
            zero_shot_path=args.from_zero_shot,
            test_path=args.test_path,
            out_json=args.out_json,
            out_md=args.out_md,
            max_samples=args.max_samples,
            prefer_fixture=not args.no_fixture,
        )

    print(f"n={report['n']} error_rate={report['error_rate']}")
    for lb in ERROR_LABELS:
        d = report["primary_distribution"][lb]
        if d["count"]:
            print(f"  primary {lb}: {d['count']} ({d['rate']:.1%})")
    print(f"wrote: {report.get('_out_json')}")
    print(f"wrote: {report.get('_out_md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
