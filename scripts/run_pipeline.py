#!/usr/bin/env python3
"""End-to-end pipeline orchestrator for llm-post-training-lab.

Stages
------
data   Build / clean / split domain data (电商智能客服)
sft    Supervised fine-tuning (or mock metrics in --demo without GPU)
dpo    Preference alignment (or mock metrics in --demo without GPU)
eval   Zero-shot, compare, error analysis (mock generation when --demo)
deploy vLLM helper docs + offline/demo serving bench (TTFT / p95 / tok/s)
all    data -> sft -> dpo -> eval -> deploy

Examples
--------
python scripts/run_pipeline.py --stage all --demo
python scripts/run_pipeline.py --stage data
python scripts/run_pipeline.py --stage eval --demo
python scripts/run_pipeline.py --stage deploy --demo
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from src.utils import configure_stdio_utf8

    configure_stdio_utf8()
except Exception:
    pass


def _has_rich() -> bool:
    try:
        import rich  # noqa: F401

        return True
    except ImportError:
        return False


def _print(msg: str) -> None:
    if _has_rich():
        from rich.console import Console

        Console().print(msg)
    else:
        print(msg)


def _banner(title: str) -> None:
    line = "=" * 64
    _print(f"\n{line}")
    _print(f"  {title}")
    _print(line)


def _step(msg: str) -> None:
    _print(f"[*] {msg}")


def _ok(msg: str) -> None:
    _print(f"[ok] {msg}")


def _warn(msg: str) -> None:
    _print(f"[warn] {msg}")


def _fail(msg: str) -> None:
    _print(f"[fail] {msg}")


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _load_script(name: str, rel_path: str):
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_module_main(main_fn: Callable[..., int], argv: list[str], name: str) -> int:
    _step(f"Running {name}: {' '.join(argv) if argv else '(defaults)'}")
    try:
        rc = int(main_fn(argv))
    except SystemExit as e:
        rc = int(e.code) if isinstance(e.code, int) else (0 if e.code is None else 1)
    except Exception as e:
        _fail(f"{name} raised: {e}")
        return 1
    if rc != 0:
        _fail(f"{name} exited with code {rc}")
    else:
        _ok(f"{name} finished")
    return rc


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _write_mock_train_metrics(stage: str, output_dir: Path, demo: bool) -> Path:
    """Write mock train metrics when skipping heavy training (no GPU / offline demo)."""
    metrics_name = "sft_train_metrics.json" if stage == "sft" else "dpo_train_metrics.json"
    metrics_path = ROOT / "reports" / metrics_name
    out_dir = ROOT / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / "MOCK_CHECKPOINT.txt"
    marker.write_text(
        f"Mock {stage} checkpoint written by run_pipeline.py --demo (no GPU training).\n",
        encoding="utf-8",
    )
    adapter_cfg = out_dir / "adapter_config.json"
    if not adapter_cfg.is_file():
        _write_json(
            adapter_cfg,
            {
                "peft_type": "LORA",
                "base_model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
                "r": 8,
                "lora_alpha": 16,
                "mock": True,
            },
        )
    train_loss = 1.234 if stage == "sft" else 0.456
    metrics: dict[str, Any] = {
        "stage": stage,
        "demo": demo,
        "mock": True,
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "device": "cpu",
        "output_dir": str(out_dir),
        "num_train_samples": 32 if demo else 0,
        "train_wall_time_sec": 0.01,
        "peak_gpu_memory_mb": None,
        "seed": 42,
        "train_loss": train_loss,
        "metrics": {
            "train_loss": train_loss,
            "train_runtime": 0.01,
            "epoch": 1.0,
        },
        "note": "Mock metrics — heavy training skipped (no CUDA / offline demo path).",
    }
    if stage == "dpo":
        metrics["beta"] = 0.1
        metrics["base_model"] = "Qwen/Qwen2.5-0.5B-Instruct"
    _write_json(metrics_path, metrics)
    _ok(f"Wrote mock {stage} metrics -> {metrics_path}")
    return metrics_path


def stage_data(*, demo: bool, seed: int) -> int:
    _banner("Stage: data")
    mod = _load_script("build_data_mod", "scripts/01_build_data.py")
    argv = ["--config", "configs/data.yaml", "--seed", str(seed), "--skip-plots"]
    if demo:
        argv += ["--sft-samples", "64", "--pref-pairs", "32"]
    return _run_module_main(mod.main, argv, "01_build_data")


def stage_sft(*, demo: bool, force_mock: bool) -> int:
    _banner("Stage: sft")
    if force_mock:
        _warn("Skipping real SFT (demo/no GPU) — writing mock metrics")
        _write_mock_train_metrics("sft", Path("outputs/sft"), demo=demo)
        return 0

    from src.train.sft_train import main as sft_main

    argv = ["--config", "configs/sft.yaml"]
    if demo:
        argv.append("--demo")
    return _run_module_main(sft_main, argv, "sft_train")


def stage_dpo(*, demo: bool, force_mock: bool) -> int:
    _banner("Stage: dpo")
    if force_mock:
        _warn("Skipping real DPO (demo/no GPU) — writing mock metrics")
        _write_mock_train_metrics("dpo", Path("outputs/dpo"), demo=demo)
        return 0

    from src.train.dpo_train import main as dpo_main

    argv = ["--config", "configs/dpo.yaml"]
    if demo:
        argv.append("--demo")
    return _run_module_main(dpo_main, argv, "dpo_train")


def stage_eval(*, demo: bool) -> int:
    _banner("Stage: eval")
    from evaluation.zero_shot_eval import main as zs_main
    from evaluation.compare_models import main as cmp_main
    from evaluation.error_analysis import main as err_main

    max_samples = "20" if demo else "50"

    zs_argv = [
        "--config",
        "configs/eval.yaml",
        "--max-samples",
        max_samples,
        "--out",
        "reports/zero_shot_results.json",
    ]
    if demo:
        zs_argv.append("--mock")
    rc = _run_module_main(zs_main, zs_argv, "zero_shot_eval")
    if rc != 0:
        return rc

    cmp_argv = [
        "--config",
        "configs/eval.yaml",
        "--base",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "--sft",
        "gold",
        "--dpo",
        "mock",
        "--max-samples",
        max_samples,
        "--out",
        "reports/comparison.json",
    ]
    if demo:
        cmp_argv.append("--mock")
    rc = _run_module_main(cmp_main, cmp_argv, "compare_models")
    if rc != 0:
        return rc

    err_argv = [
        "--from-zero-shot",
        "reports/zero_shot_results.json",
        "--out-json",
        "reports/error_analysis.json",
        "--out-md",
        "reports/error_analysis.md",
        "--max-samples",
        max_samples,
    ]
    if demo:
        err_argv.append("--demo-errors")
    return _run_module_main(err_main, err_argv, "error_analysis")


def stage_deploy(*, demo: bool) -> int:
    """Print vLLM deploy guidance and run serving bench (demo-safe offline mock)."""
    _banner("Stage: deploy / serving bench")
    deploy_mod = _load_script("deploy_vllm_mod", "scripts/07_deploy_vllm.py")
    # Always print install + recommended command (no --run unless GPU host)
    rc = _run_module_main(
        deploy_mod.main,
        ["--config", "configs/deploy.yaml"],
        "07_deploy_vllm",
    )
    if rc != 0:
        _warn("deploy helper returned non-zero; continuing to bench")

    bench_mod = _load_script("bench_serving_mod", "scripts/08_bench_serving.py")
    bench_argv = [
        "--config",
        "configs/deploy.yaml",
        "--out",
        "reports/bench_serving.json",
        "--num-prompts",
        "16" if demo else "32",
        "--concurrency",
        "4" if demo else "8",
    ]
    if demo:
        bench_argv.append("--demo")
    return _run_module_main(bench_mod.main, bench_argv, "08_bench_serving")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="llm-post-training-lab pipeline runner")
    p.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["data", "sft", "dpo", "eval", "deploy", "all"],
        help="Pipeline stage to run",
    )
    p.add_argument(
        "--demo",
        action="store_true",
        help="Small data; mock train if no GPU; mock/fixture eval",
    )
    p.add_argument(
        "--force-mock-train",
        action="store_true",
        help="Always write mock train metrics (never download/train models)",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t0 = time.time()
    _banner("llm-post-training-lab :: run_pipeline")
    _print(f"stage={args.stage} demo={args.demo} seed={args.seed} root={ROOT}")
    cuda = _cuda_available()
    _print(f"cuda_available={cuda}")

    # Mock train when: forced, or demo without CUDA
    mock_train = bool(args.force_mock_train) or (args.demo and not cuda)

    if args.stage == "all":
        stages = ["data", "sft", "dpo", "eval", "deploy"]
    else:
        stages = [args.stage]

    for st in stages:
        if st == "data":
            rc = stage_data(demo=args.demo, seed=args.seed)
        elif st == "sft":
            rc = stage_sft(demo=args.demo, force_mock=mock_train)
        elif st == "dpo":
            rc = stage_dpo(demo=args.demo, force_mock=mock_train)
        elif st == "eval":
            rc = stage_eval(demo=args.demo)
        elif st == "deploy":
            rc = stage_deploy(demo=args.demo)
        else:
            _fail(f"Unknown stage {st}")
            return 2
        if rc != 0:
            _fail(f"Pipeline stopped at stage={st} rc={rc}")
            return rc

    elapsed = time.time() - t0
    _banner("Pipeline complete")
    _ok(f"Finished stages={stages} in {elapsed:.1f}s")
    _print(f"Reports: {ROOT / 'reports'}")
    _print(f"Data:    {ROOT / 'data'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
