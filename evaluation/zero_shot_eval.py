"""Zero-shot / base-model evaluation on the CS test suite.

Generates with HuggingFace transformers (or uses ``--predictions`` / mock mode)
and writes ``reports/zero_shot_results.json``.
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
from evaluation.metrics import aggregate_scores, score_sample  # noqa: E402
from src.utils import (  # noqa: E402
    ensure_dir,
    format_chat,
    get_device,
    load_yaml,
    peak_gpu_memory_mb,
    project_root,
    read_jsonl,
    reset_peak_gpu_memory,
    resolve_path,
    save_json,
    set_seed,
)


def _load_model_and_tokenizer(
    model_name: str,
    *,
    trust_remote_code: bool = True,
    torch_dtype: str = "auto",
    device: str | None = None,
    base_model: str | None = None,
):
    """Load a full HF causal LM, or a PEFT adapter directory on top of base_model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = device or get_device()
    dtype = torch.float32
    if torch_dtype in ("bfloat16", "bf16") and dev == "cuda":
        dtype = torch.bfloat16
    elif torch_dtype in ("float16", "fp16", "half") and dev != "cpu":
        dtype = torch.float16
    elif torch_dtype == "auto" and dev == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    root = project_root()
    path = Path(model_name)
    if not path.is_absolute():
        cand = root / model_name
        if cand.exists():
            path = cand

    # PEFT adapter directory
    if path.is_dir() and (path / "adapter_config.json").is_file():
        import json

        from peft import PeftModel

        base = base_model
        if not base:
            try:
                with (path / "adapter_config.json").open("r", encoding="utf-8") as f:
                    base = (json.load(f) or {}).get("base_model_name_or_path")
            except Exception:
                base = None
        base = base or "Qwen/Qwen2.5-0.5B-Instruct"
        tok = AutoTokenizer.from_pretrained(base, trust_remote_code=trust_remote_code)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        base_m = AutoModelForCausalLM.from_pretrained(
            base, trust_remote_code=trust_remote_code, torch_dtype=dtype
        )
        model = PeftModel.from_pretrained(base_m, str(path))
        model.to(dev)
        model.eval()
        return model, tok, dev

    load_name = str(path) if path.exists() else model_name
    tok = AutoTokenizer.from_pretrained(load_name, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        load_name,
        trust_remote_code=trust_remote_code,
        torch_dtype=dtype,
    )
    model.to(dev)
    model.eval()
    return model, tok, dev


def generate_one(
    model,
    tokenizer,
    messages: list[dict[str, str]],
    *,
    device: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    do_sample: bool = False,
) -> str:
    import torch

    prompt = format_chat(messages, tokenizer=tokenizer, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample and temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    gen_ids = out[0][inputs["input_ids"].shape[-1] :]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return text


def mock_generate(item: dict[str, Any]) -> str:
    """Deterministic stub for offline CI without model weights."""
    user = item.get("user") or ""
    cat = item.get("category") or ""
    # Intentionally imperfect: sometimes omit keywords / invent IDs for error analysis demos
    if "运单号" in user and "订单号" not in user:
        return "您好。您的快递单号是 SF1234567890123，预计明天送达。祝您购物愉快！"
    if item.get("gold_mc"):
        return str(item["gold_mc"])
    kws = item.get("expected_keywords") or []
    body = f"您好。关于您咨询的{cat}问题，建议您以订单页/详情页信息为准。"
    if kws:
        body += "请关注：" + "、".join(kws[:3]) + "。"
    body += "如还有其他问题，随时告诉我。"
    return body


def run_zero_shot(
    *,
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    base_model: str | None = None,
    test_path: str | Path | None = None,
    max_samples: int | None = 50,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    out_path: str | Path | None = None,
    mock: bool = False,
    predictions_path: str | Path | None = None,
    seed: int = 42,
    prefer_fixture: bool = True,
) -> dict[str, Any]:
    root = project_root()
    set_seed(seed)
    items = load_suite(test_path, max_samples=max_samples, prefer_fixture=prefer_fixture)

    pred_map: dict[str, str] = {}
    if predictions_path:
        for row in read_jsonl(resolve_path(predictions_path, root)):
            pid = str(row.get("id") or "")
            pred_map[pid] = (row.get("prediction") or row.get("output") or row.get("answer") or "").strip()

    model = tok = None
    device = "cpu"
    t_load0 = time.time()
    if not mock and not predictions_path:
        try:
            reset_peak_gpu_memory()
            model, tok, device = _load_model_and_tokenizer(
                model_name, base_model=base_model
            )
        except Exception as e:  # pragma: no cover
            print(f"[warn] model load failed ({e}); falling back to --mock generation")
            mock = True
    load_s = time.time() - t_load0

    results: list[dict[str, Any]] = []
    t_gen0 = time.time()
    for item in items:
        iid = item["id"]
        if iid in pred_map:
            pred = pred_map[iid]
        elif mock or model is None:
            pred = mock_generate(item)
        else:
            msgs = item.get("messages") or [{"role": "user", "content": item.get("user", "")}]
            pred = generate_one(
                model,
                tok,
                msgs,
                device=device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
            )

        scored = score_sample(
            pred,
            context=item.get("context") or item.get("user") or "",
            reference=item.get("gold"),
            expected_keywords=item.get("expected_keywords"),
            must_not_contain=item.get("must_not_contain"),
            gold_mc=item.get("gold_mc"),
            mc_choices=item.get("choices"),
        )
        row = {
            "id": iid,
            "category": item.get("category"),
            "user": item.get("user"),
            "gold": item.get("gold"),
            "prediction": pred,
            "expected_keywords": item.get("expected_keywords"),
            "must_not_contain": item.get("must_not_contain"),
            **scored,
        }
        results.append(row)

    gen_s = time.time() - t_gen0
    summary = aggregate_scores(results)
    # attach answer lengths properly
    for r in results:
        r["answer"] = r.get("prediction", "")
    summary = aggregate_scores(results)

    report = {
        "task": "zero_shot_eval",
        "model_name": model_name if not mock else f"mock::{model_name}",
        "mock": mock or bool(predictions_path and model is None),
        "device": device,
        "n": len(results),
        "test_source": items[0].get("_source") if items else None,
        "seed": seed,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "timing": {
            "load_seconds": round(load_s, 3),
            "generate_seconds": round(gen_s, 3),
            "peak_gpu_memory_mb": peak_gpu_memory_mb(),
        },
        "summary": summary,
        "per_category": _by_category(results),
        "samples": results,
    }

    dest = resolve_path(out_path or "reports/zero_shot_results.json", root)
    save_json(dest, report)
    report["_out_path"] = str(dest)
    return report


def _by_category(results: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        cat = str(r.get("category") or "未知")
        buckets.setdefault(cat, []).append(r)
    return {cat: aggregate_scores(rows) for cat, rows in sorted(buckets.items())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Zero-shot eval of base model on CS test suite")
    parser.add_argument("--config", type=str, default="configs/eval.yaml")
    parser.add_argument("--model", type=str, default=None, help="HF id, full checkpoint, or PEFT adapter dir")
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base HF id when --model is a PEFT adapter directory",
    )
    parser.add_argument("--test-path", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--out", type=str, default="reports/zero_shot_results.json")
    parser.add_argument("--mock", action="store_true", help="No model load; deterministic stub replies")
    parser.add_argument("--predictions", type=str, default=None, help="JSONL with id+prediction")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-fixture", action="store_true", help="Do not prefer fixture suite")
    args = parser.parse_args(argv)

    root = project_root()
    cfg: dict[str, Any] = {}
    cfg_path = resolve_path(args.config, root)
    if cfg_path.is_file():
        try:
            cfg = load_yaml(cfg_path)
        except Exception:
            cfg = {}

    model_cfg = cfg.get("model") or {}
    data_cfg = cfg.get("data") or {}

    model_name = args.model or model_cfg.get("name") or "Qwen/Qwen2.5-0.5B-Instruct"
    base_model = args.base_model or model_cfg.get("base_model")
    max_samples = args.max_samples
    if max_samples is None:
        max_samples = data_cfg.get("max_samples") or data_cfg.get("sample_size") or 50
    max_new = args.max_new_tokens or model_cfg.get("max_new_tokens") or 256
    seed = args.seed if args.seed is not None else int(cfg.get("seed") or 42)
    temperature = float(model_cfg.get("temperature") or 0.0)

    print("=" * 60)
    print("llm-post-training-lab :: zero-shot eval")
    print("=" * 60)
    print(f"model: {model_name}  base={base_model}  mock={args.mock}")

    report = run_zero_shot(
        model_name=model_name,
        base_model=base_model,
        test_path=args.test_path,
        max_samples=int(max_samples) if max_samples else None,
        max_new_tokens=int(max_new),
        temperature=temperature,
        out_path=args.out,
        mock=args.mock,
        predictions_path=args.predictions,
        seed=seed,
        prefer_fixture=not args.no_fixture,
    )
    s = report["summary"]
    print(f"n={s['n']}  mean_composite={s['mean_composite']}  pass_rate={s['pass_rate']}")
    print(f"hallucination_rate={s['hallucination_rate']}  safety_fail_rate={s['safety_fail_rate']}")
    print(f"wrote: {report.get('_out_path')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
