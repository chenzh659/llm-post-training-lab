"""Compare base vs SFT vs DPO predictions with a rule-based judge.

Win-rate uses composite score (format + keyword + safety + hallucination +
length preference). Optional pairwise tournament across model pairs.

Writes ``reports/comparison.json``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluation.build_test_suite import load_suite  # noqa: E402
from evaluation.metrics import (  # noqa: E402
    aggregate_scores,
    rule_judge_score,
    score_sample,
)
from evaluation.zero_shot_eval import generate_one, mock_generate, _load_model_and_tokenizer  # noqa: E402
from src.utils import (  # noqa: E402
    get_device,
    load_json,
    load_yaml,
    peak_gpu_memory_mb,
    project_root,
    read_jsonl,
    reset_peak_gpu_memory,
    resolve_path,
    save_json,
    set_seed,
)


def _load_predictions_jsonl(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in read_jsonl(path):
        iid = str(row.get("id") or "")
        if not iid:
            continue
        pred = row.get("prediction") or row.get("output") or row.get("answer") or row.get("response")
        if pred is None and isinstance(row.get("messages"), list):
            for m in reversed(row["messages"]):
                if m.get("role") == "assistant":
                    pred = m.get("content")
                    break
        if pred is not None:
            out[iid] = str(pred).strip()
    return out


def _weak_mock_generate(item: dict[str, Any], noise: str = "base") -> str:
    """Intentionally weaker replies for offline base/dpo stubs (demo ranking)."""
    user = item.get("user") or ""
    cat = item.get("category") or ""
    # Inject occasional hallucinations / policy errors for dpo-as-unaligned stub
    if noise == "base":
        if "运单" in user or "物流" in user:
            return "您好。单号SF1234567890123，预计明天一定到。祝您购物愉快！"
        if "退" in user:
            return "可以退，绝对能退，我保证一定给您退款。"
        return f"好的，关于{cat}我知道了。"
    # dpo mock: mixed quality
    if "券" in user or "优惠" in user:
        return "您好。过期券一般无法补发，请以结算页为准。祝您购物愉快！"
    return mock_generate(item)


def _try_load_peft(model_name: str, base_model: str | None, device: str):
    """Load full model or PEFT adapter on top of base."""
    from pathlib import Path as P

    p = P(model_name)
    # Adapter dir heuristic
    if p.is_dir() and (p / "adapter_config.json").is_file():
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        base = base_model or "Qwen/Qwen2.5-0.5B-Instruct"
        tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        dtype = torch.float32
        if device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        base_m = AutoModelForCausalLM.from_pretrained(base, trust_remote_code=True, torch_dtype=dtype)
        model = PeftModel.from_pretrained(base_m, str(p))
        model.to(device)
        model.eval()
        return model, tok, device
    return _load_model_and_tokenizer(model_name, device=device)


def generate_for_model(
    items: list[dict[str, Any]],
    *,
    model_name: str,
    base_model: str | None = None,
    mock: bool = False,
    predictions_path: str | Path | None = None,
    max_new_tokens: int = 256,
    label: str = "model",
) -> dict[str, str]:
    root = project_root()
    if predictions_path:
        p = resolve_path(predictions_path, root)
        if p.is_file():
            return _load_predictions_jsonl(p)
        print(f"[warn] predictions not found for {label}: {p}")

    # Gold-as-SFT style demo: always honor reference replies (even under --mock)
    if model_name.lower() in ("gold", "reference"):
        out = {}
        for it in items:
            out[it["id"]] = (it.get("gold") or mock_generate(it)).strip()
        return out

    if mock or model_name.lower() == "mock":
        # Weaker base-like stub so ranking vs gold is visible offline
        if label in ("base", "dpo") or model_name.lower() == "mock":
            return {it["id"]: _weak_mock_generate(it, noise=label) for it in items}
        return {it["id"]: mock_generate(it) for it in items}

    device = get_device()
    try:
        reset_peak_gpu_memory()
        model, tok, device = _try_load_peft(model_name, base_model, device)
    except Exception as e:
        print(f"[warn] failed to load {label}={model_name} ({e}); using mock")
        return {it["id"]: mock_generate(it) for it in items}

    preds: dict[str, str] = {}
    for it in items:
        msgs = it.get("messages") or [{"role": "user", "content": it.get("user", "")}]
        try:
            preds[it["id"]] = generate_one(
                model, tok, msgs, device=device, max_new_tokens=max_new_tokens, do_sample=False
            )
        except Exception as e:
            print(f"[warn] gen fail {it['id']}: {e}")
            preds[it["id"]] = mock_generate(it)
    # free
    try:
        del model
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return preds


def pairwise_winner(
    a: str,
    b: str,
    *,
    context: str,
    expected_keywords: list[str] | None,
    must_not_contain: list[str] | None,
) -> str:
    """Return 'a' | 'b' | 'tie'."""
    sa = rule_judge_score(
        a, context=context, expected_keywords=expected_keywords, must_not_contain=must_not_contain
    )
    sb = rule_judge_score(
        b, context=context, expected_keywords=expected_keywords, must_not_contain=must_not_contain
    )
    if abs(sa - sb) < 0.02:
        return "tie"
    return "a" if sa > sb else "b"


def compare_models(
    *,
    models: dict[str, str],
    base_model: str | None = None,
    test_path: str | Path | None = None,
    max_samples: int | None = 50,
    max_new_tokens: int = 256,
    mock: bool = False,
    prediction_files: dict[str, str] | None = None,
    out_path: str | Path | None = None,
    seed: int = 42,
    prefer_fixture: bool = True,
) -> dict[str, Any]:
    """``models`` maps label -> HF path/id (or 'gold' / 'mock')."""
    root = project_root()
    set_seed(seed)
    items = load_suite(test_path, max_samples=max_samples, prefer_fixture=prefer_fixture)
    prediction_files = prediction_files or {}

    all_preds: dict[str, dict[str, str]] = {}
    t0 = time.time()
    for label, mpath in models.items():
        print(f"[compare] generating for {label} <- {mpath}")
        all_preds[label] = generate_for_model(
            items,
            model_name=mpath,
            base_model=base_model,
            mock=mock or mpath.lower() == "mock",
            predictions_path=prediction_files.get(label),
            max_new_tokens=max_new_tokens,
            label=label,
        )

    labels = list(models.keys())
    per_model_rows: dict[str, list[dict[str, Any]]] = {lb: [] for lb in labels}
    pairwise: dict[str, dict[str, Any]] = {}

    # init pairwise stats
    for i, a in enumerate(labels):
        for b in labels[i + 1 :]:
            key = f"{a}_vs_{b}"
            pairwise[key] = {"a": a, "b": b, "wins_a": 0, "wins_b": 0, "ties": 0, "n": 0}

    detailed: list[dict[str, Any]] = []
    for it in items:
        iid = it["id"]
        ctx = it.get("context") or it.get("user") or ""
        exp = it.get("expected_keywords")
        forbid = it.get("must_not_contain")
        entry: dict[str, Any] = {
            "id": iid,
            "category": it.get("category"),
            "user": it.get("user"),
            "predictions": {},
            "scores": {},
        }
        scored_map: dict[str, dict[str, Any]] = {}
        for lb in labels:
            pred = all_preds[lb].get(iid, "")
            sc = score_sample(
                pred,
                context=ctx,
                reference=it.get("gold"),
                expected_keywords=exp,
                must_not_contain=forbid,
                gold_mc=it.get("gold_mc"),
                mc_choices=it.get("choices"),
            )
            sc["answer"] = pred
            scored_map[lb] = sc
            per_model_rows[lb].append(sc)
            entry["predictions"][lb] = pred
            entry["scores"][lb] = {
                "composite": sc["composite"],
                "passed": sc["passed"],
                "components": sc["components"],
            }

        # pairwise
        for i, a in enumerate(labels):
            for b in labels[i + 1 :]:
                key = f"{a}_vs_{b}"
                w = pairwise_winner(
                    all_preds[a].get(iid, ""),
                    all_preds[b].get(iid, ""),
                    context=ctx,
                    expected_keywords=exp,
                    must_not_contain=forbid,
                )
                pairwise[key]["n"] += 1
                if w == "a":
                    pairwise[key]["wins_a"] += 1
                elif w == "b":
                    pairwise[key]["wins_b"] += 1
                else:
                    pairwise[key]["ties"] += 1
                entry.setdefault("pairwise", {})[key] = w

        detailed.append(entry)

    model_summaries = {lb: aggregate_scores(rows) for lb, rows in per_model_rows.items()}
    for key, st in pairwise.items():
        n = max(1, st["n"])
        st["win_rate_a"] = round(st["wins_a"] / n, 4)
        st["win_rate_b"] = round(st["wins_b"] / n, 4)
        st["tie_rate"] = round(st["ties"] / n, 4)

    report = {
        "task": "compare_models",
        "models": models,
        "base_model": base_model,
        "mock": mock,
        "n": len(items),
        "test_source": items[0].get("_source") if items else None,
        "seed": seed,
        "model_summaries": model_summaries,
        "pairwise": pairwise,
        "ranking": _rank_models(model_summaries),
        "timing": {
            "seconds": round(time.time() - t0, 3),
            "peak_gpu_memory_mb": peak_gpu_memory_mb(),
        },
        "samples": detailed,
    }
    dest = resolve_path(out_path or "reports/comparison.json", root)
    save_json(dest, report)
    report["_out_path"] = str(dest)
    return report


def _rank_models(summaries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        (
            {
                "model": k,
                "mean_composite": v.get("mean_composite", 0.0),
                "pass_rate": v.get("pass_rate", 0.0),
                "hallucination_rate": v.get("hallucination_rate", 0.0),
            }
            for k, v in summaries.items()
        ),
        key=lambda x: (-x["mean_composite"], -x["pass_rate"], x["hallucination_rate"]),
    )
    for i, r in enumerate(ranked, 1):
        r["rank"] = i
    return ranked


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare base / SFT / DPO with rule judge")
    parser.add_argument("--config", type=str, default="configs/eval.yaml")
    parser.add_argument("--base", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--sft", type=str, default="gold", help="SFT checkpoint, or 'gold' for reference replies")
    parser.add_argument("--dpo", type=str, default="mock", help="DPO checkpoint or 'mock'")
    parser.add_argument("--base-model", type=str, default=None, help="Base for PEFT adapters")
    parser.add_argument("--test-path", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--out", type=str, default="reports/comparison.json")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--pred-base", type=str, default=None)
    parser.add_argument("--pred-sft", type=str, default=None)
    parser.add_argument("--pred-dpo", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-fixture", action="store_true")
    args = parser.parse_args(argv)

    root = project_root()
    cfg_path = resolve_path(args.config, root)
    max_samples = args.max_samples
    if cfg_path.is_file():
        try:
            cfg = load_yaml(cfg_path)
            if max_samples is None:
                max_samples = (cfg.get("data") or {}).get("max_samples") or 50
        except Exception:
            pass
    if max_samples is None:
        max_samples = 50

    models = {"base": args.base, "sft": args.sft, "dpo": args.dpo}
    pred_files = {}
    if args.pred_base:
        pred_files["base"] = args.pred_base
    if args.pred_sft:
        pred_files["sft"] = args.pred_sft
    if args.pred_dpo:
        pred_files["dpo"] = args.pred_dpo

    print("=" * 60)
    print("llm-post-training-lab :: compare models")
    print("=" * 60)
    for k, v in models.items():
        print(f"  {k}: {v}")

    report = compare_models(
        models=models,
        base_model=args.base_model or args.base,
        test_path=args.test_path,
        max_samples=int(max_samples),
        max_new_tokens=args.max_new_tokens,
        mock=args.mock,
        prediction_files=pred_files or None,
        out_path=args.out,
        seed=args.seed,
        prefer_fixture=not args.no_fixture,
    )

    print("\nRanking:")
    for r in report["ranking"]:
        print(
            f"  #{r['rank']} {r['model']}: composite={r['mean_composite']} "
            f"pass={r['pass_rate']} hallu={r['hallucination_rate']}"
        )
    print("\nPairwise win-rates:")
    for key, st in report["pairwise"].items():
        print(
            f"  {st['a']} vs {st['b']}: "
            f"{st['win_rate_a']:.2%} / {st['win_rate_b']:.2%} (tie {st['tie_rate']:.2%})"
        )
    print(f"\nwrote: {report.get('_out_path')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
