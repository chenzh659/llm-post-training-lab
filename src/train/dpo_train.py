"""Direct Preference Optimization (DPO) on top of SFT adapter via TRL."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

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

DEMO_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEMO_SAMPLES = 32
DEMO_EPOCHS = 1


def _first_existing(candidates: list[Path]) -> Path | None:
    for p in candidates:
        if p.is_file() or p.is_dir():
            return p
    return None


def resolve_pref_paths(cfg: dict[str, Any], root: Path) -> tuple[Path, Path | None]:
    data = cfg.get("data") or {}
    train_cfg = data.get("train_path") or "data/processed/dpo_train.jsonl"
    eval_cfg = data.get("eval_path") or "data/processed/dpo_eval.jsonl"
    train_cands = [
        resolve_path(train_cfg, root),
        root / "data" / "splits" / "train.pref.jsonl",
        root / "data" / "processed" / "preference_clean.jsonl",
        root / "data" / "raw" / "preference_raw.jsonl",
    ]
    eval_cands = [
        resolve_path(eval_cfg, root),
        root / "data" / "splits" / "val.pref.jsonl",
        root / "data" / "splits" / "test.pref.jsonl",
    ]
    train_path = _first_existing(train_cands)
    if train_path is None:
        raise FileNotFoundError(
            "Preference train JSONL not found. Tried:\n  "
            + "\n  ".join(str(p) for p in train_cands)
            + "\nRun: python scripts/01_build_data.py"
        )
    eval_path = _first_existing(eval_cands)
    return train_path, eval_path  # type: ignore[return-value]


def _want_qlora(model_cfg: dict[str, Any], demo: bool, device: str) -> bool:
    if demo or device == "cpu":
        return False
    if model_cfg.get("use_qlora") is False:
        return False
    if model_cfg.get("use_qlora") is True:
        return True
    bits = model_cfg.get("bits")
    return bits is not None and int(bits) == 4


def _dtype_from_name(name: str | None):
    import torch

    if not name:
        return torch.float32
    n = str(name).lower()
    if n in ("bfloat16", "bf16"):
        return torch.bfloat16
    if n in ("float16", "fp16", "half"):
        return torch.float16
    return torch.float32


def build_bnb_config(model_cfg: dict[str, Any]):
    from transformers import BitsAndBytesConfig
    import torch

    compute = _dtype_from_name(model_cfg.get("bnb_4bit_compute_dtype", "bfloat16"))
    if compute == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        compute = torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute,
        bnb_4bit_quant_type=str(model_cfg.get("bnb_4bit_quant_type", "nf4")),
        bnb_4bit_use_double_quant=bool(model_cfg.get("bnb_4bit_use_double_quant", True)),
    )


def resolve_base_and_adapter(
    model_cfg: dict[str, Any],
    root: Path,
    *,
    demo: bool,
    model_override: str | None,
) -> tuple[str, Path | None]:
    """
    Return (base_model_name_or_path, sft_adapter_dir_or_None).

    Config ``model.name`` may be ``outputs/sft`` (adapter) or a HF id.
    """
    name = model_override or model_cfg.get("name") or DEMO_MODEL
    if demo and model_override is None:
        # Prefer tiny base in demo; still try SFT adapter if present
        sft_dir = root / "outputs" / "sft"
        adapter_cfg = sft_dir / "adapter_config.json"
        if adapter_cfg.is_file():
            # Read base from adapter config if possible
            try:
                import json

                with adapter_cfg.open("r", encoding="utf-8") as f:
                    ac = json.load(f)
                base = ac.get("base_model_name_or_path") or DEMO_MODEL
                return str(base), sft_dir
            except Exception:
                return DEMO_MODEL, sft_dir
        return DEMO_MODEL, None

    path = resolve_path(name, root)
    if path.is_dir():
        adapter_cfg = path / "adapter_config.json"
        if adapter_cfg.is_file():
            import json

            with adapter_cfg.open("r", encoding="utf-8") as f:
                ac = json.load(f)
            base = ac.get("base_model_name_or_path") or DEMO_MODEL
            return str(base), path
        # Full model checkpoint dir
        return str(path), None

    # HF hub id
    sft_fallback = root / "outputs" / "sft"
    if (sft_fallback / "adapter_config.json").is_file() and name == (model_cfg.get("name") or ""):
        # name is hub id but SFT adapter exists — load both
        return str(name), sft_fallback
    # When config says outputs/sft as string that isn't created yet
    if str(name).replace("\\", "/").endswith("outputs/sft") or str(name).endswith("sft"):
        sft_dir = root / "outputs" / "sft"
        if (sft_dir / "adapter_config.json").is_file():
            import json

            with (sft_dir / "adapter_config.json").open("r", encoding="utf-8") as f:
                ac = json.load(f)
            return str(ac.get("base_model_name_or_path") or DEMO_MODEL), sft_dir
        # Fall back to default base; warn
        print("[dpo] WARNING: SFT adapter not found at outputs/sft; using base model only.")
        # Try sft.yaml for base name
        sft_yaml = root / "configs" / "sft.yaml"
        if sft_yaml.is_file():
            sft_cfg = load_yaml(sft_yaml)
            base = (sft_cfg.get("model") or {}).get("name") or DEMO_MODEL
            return str(base), None
        return DEMO_MODEL, None
    return str(name), None


def load_policy_model(
    base_name: str,
    adapter_path: Path | None,
    model_cfg: dict[str, Any],
    *,
    use_qlora: bool,
    device: str,
    demo: bool,
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training

    trust = bool(model_cfg.get("trust_remote_code", True))
    tokenizer = AutoTokenizer.from_pretrained(
        str(adapter_path) if adapter_path and (adapter_path / "tokenizer_config.json").is_file() else base_name,
        trust_remote_code=trust,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # common for DPO generation-style batching

    dtype = _dtype_from_name(model_cfg.get("torch_dtype", "bfloat16"))
    if device == "cpu":
        dtype = torch.float32

    kwargs: dict[str, Any] = {
        "trust_remote_code": trust,
        "torch_dtype": dtype,
    }
    if use_qlora:
        kwargs["quantization_config"] = build_bnb_config(model_cfg)
        kwargs["device_map"] = "auto"
    else:
        if device == "cuda":
            kwargs["device_map"] = {"": 0}
        elif device == "cpu":
            kwargs["device_map"] = {"": "cpu"}

    try:
        model = AutoModelForCausalLM.from_pretrained(base_name, **kwargs)
    except OSError as e:
        raise RuntimeError(
            f"Failed to load base model '{base_name}'. "
            f"Optional: python scripts/download_model.py --model {base_name}\n{e}"
        ) from e

    if hasattr(model, "config"):
        model.config.use_cache = bool(model_cfg.get("use_cache", False))
        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = tokenizer.pad_token_id

    lora_cfg = model_cfg.get("_lora") or {}  # injected by caller optionally

    if adapter_path is not None and (adapter_path / "adapter_config.json").is_file():
        print(f"[dpo] loading PEFT adapter from {adapter_path}")
        if use_qlora:
            model = prepare_model_for_kbit_training(model)
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=True)
        # Ensure trainable
        for n, p in model.named_parameters():
            if "lora_" in n:
                p.requires_grad = True
    else:
        # Fresh LoRA on base
        from peft import LoraConfig as LC

        if use_qlora:
            model = prepare_model_for_kbit_training(model)
        peft_cfg = LC(
            r=int(lora_cfg.get("r", 16)),
            lora_alpha=int(lora_cfg.get("lora_alpha", 32)),
            lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
            bias=str(lora_cfg.get("bias", "none")),
            task_type=TaskType.CAUSAL_LM,
            target_modules=list(
                lora_cfg.get(
                    "target_modules",
                    ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                )
            ),
        )
        model = get_peft_model(model, peft_cfg)

    try:
        model.print_trainable_parameters()
    except Exception:
        pass
    return model, tokenizer


def normalize_pref_row(
    row: dict[str, Any],
    tokenizer,
    *,
    prompt_key: str,
    chosen_key: str,
    rejected_key: str,
) -> dict[str, str]:
    """Return TRL DPO-style prompt / chosen / rejected strings."""
    chosen = row.get(chosen_key) or row.get("chosen") or ""
    rejected = row.get(rejected_key) or row.get("rejected") or ""

    if "prompt_messages" in row and isinstance(row["prompt_messages"], list):
        prompt = format_chat(row["prompt_messages"], tokenizer=tokenizer, add_generation_prompt=True)
    elif prompt_key in row and isinstance(row[prompt_key], str):
        # Wrap plain prompt with chat template if possible
        msgs = [{"role": "user", "content": row[prompt_key]}]
        # Prefer system if present in messages field
        if row.get("messages"):
            # Use all non-assistant messages as prompt
            prompt_msgs = [m for m in row["messages"] if m.get("role") != "assistant"]
            if prompt_msgs:
                prompt = format_chat(prompt_msgs, tokenizer=tokenizer, add_generation_prompt=True)
            else:
                prompt = format_chat(msgs, tokenizer=tokenizer, add_generation_prompt=True)
        else:
            prompt = format_chat(msgs, tokenizer=tokenizer, add_generation_prompt=True)
    elif "messages" in row:
        msgs = row["messages"]
        # last assistant is chosen? better require explicit keys
        prompt_msgs = [m for m in msgs if m.get("role") != "assistant"]
        prompt = format_chat(prompt_msgs, tokenizer=tokenizer, add_generation_prompt=True)
    else:
        prompt = str(row.get("prompt", ""))

    return {
        "prompt": prompt if isinstance(prompt, str) else str(prompt),
        "chosen": str(chosen),
        "rejected": str(rejected),
    }


def build_pref_dataset(
    path: Path,
    tokenizer,
    data_cfg: dict[str, Any],
    max_samples: int | None = None,
):
    from datasets import Dataset

    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"No preference samples in {path}")
    if max_samples is not None and max_samples > 0:
        rows = rows[:max_samples]

    prompt_key = data_cfg.get("prompt_key", "prompt")
    chosen_key = data_cfg.get("chosen_key", "chosen")
    rejected_key = data_cfg.get("rejected_key", "rejected")

    normed = [
        normalize_pref_row(
            r,
            tokenizer,
            prompt_key=prompt_key,
            chosen_key=chosen_key,
            rejected_key=rejected_key,
        )
        for r in rows
    ]
    # Drop empty
    normed = [r for r in normed if r["prompt"] and r["chosen"] and r["rejected"]]
    if not normed:
        raise ValueError(f"No valid prompt/chosen/rejected triples in {path}")
    return Dataset.from_list(normed)


def build_dpo_config(
    train_cfg: dict[str, Any],
    dpo_cfg: dict[str, Any],
    *,
    output_dir: str,
    demo: bool,
    device: str,
    has_eval: bool,
    seed: int,
):
    import torch

    try:
        from trl import DPOConfig

        ConfigCls = DPOConfig
    except ImportError:
        from transformers import TrainingArguments

        ConfigCls = TrainingArguments

    epochs = DEMO_EPOCHS if demo else float(train_cfg.get("num_train_epochs", 1))
    batch = 1 if demo else int(train_cfg.get("per_device_train_batch_size", 1))
    eval_batch = 1 if demo else int(train_cfg.get("per_device_eval_batch_size", 1))
    grad_accum = 1 if demo else int(train_cfg.get("gradient_accumulation_steps", 16))
    lr = float(train_cfg.get("learning_rate", 5e-6))

    use_bf16 = bool(train_cfg.get("bf16", True)) and device == "cuda" and torch.cuda.is_bf16_supported()
    use_fp16 = bool(train_cfg.get("fp16", False)) and device == "cuda" and not use_bf16
    if demo and device == "cpu":
        use_bf16 = use_fp16 = False

    optim = train_cfg.get("optim", "paged_adamw_8bit")
    if demo or device == "cpu":
        optim = "adamw_torch"

    max_length = int(dpo_cfg.get("max_length", 512 if demo else 2048))
    max_prompt_length = int(dpo_cfg.get("max_prompt_length", 256 if demo else 1024))
    if demo:
        max_length = min(max_length, 512)
        max_prompt_length = min(max_prompt_length, 256)

    kwargs: dict[str, Any] = dict(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=eval_batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type=str(train_cfg.get("lr_scheduler_type", "cosine")),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.1)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
        logging_steps=1 if demo else int(train_cfg.get("logging_steps", 10)),
        save_strategy="no" if demo else str(train_cfg.get("save_strategy", "steps")),
        save_steps=int(train_cfg.get("save_steps", 100)),
        save_total_limit=int(train_cfg.get("save_total_limit", 2)),
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)) and not demo,
        optim=optim,
        report_to=train_cfg.get("report_to", "none"),
        remove_unused_columns=bool(train_cfg.get("remove_unused_columns", False)),
        seed=seed,
        beta=float(dpo_cfg.get("beta", 0.1)),
        loss_type=str(dpo_cfg.get("loss_type", "sigmoid")),
        max_length=max_length,
        max_prompt_length=max_prompt_length,
    )
    if has_eval and not demo:
        kwargs["eval_strategy"] = str(train_cfg.get("eval_strategy", "steps"))
        kwargs["eval_steps"] = int(train_cfg.get("eval_steps", 50))
    else:
        kwargs["eval_strategy"] = "no"

    try:
        return ConfigCls(**kwargs)
    except TypeError:
        import inspect

        if "eval_strategy" in kwargs:
            kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
        sig = inspect.signature(ConfigCls.__init__)
        allowed = set(sig.parameters.keys()) - {"self"}
        filtered = {k: v for k, v in kwargs.items() if k in allowed}
        return ConfigCls(**filtered)


def run_dpo(
    config_path: str | Path = "configs/dpo.yaml",
    *,
    demo: bool = False,
    model_name_override: str | None = None,
    max_samples: int | None = None,
    output_dir_override: str | None = None,
    merge_adapter: bool = False,
) -> dict[str, Any]:
    root = project_root()
    cfg = load_yaml(resolve_path(config_path, root))
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    device = get_device()
    model_cfg = dict(cfg.get("model") or {})
    lora_cfg = dict(cfg.get("lora") or {})
    model_cfg["_lora"] = lora_cfg
    data_cfg = dict(cfg.get("data") or {})
    train_cfg = dict(cfg.get("training") or {})
    dpo_cfg = dict(cfg.get("dpo") or {})

    if not demo and device == "cpu":
        raise RuntimeError(
            "Full DPO requires a CUDA GPU (or set --demo for CPU smoke with a tiny subset). "
            f"Detected device={device}."
        )

    use_qlora = _want_qlora(model_cfg, demo=demo, device=device)
    base_name, adapter_path = resolve_base_and_adapter(
        model_cfg, root, demo=demo, model_override=model_name_override
    )
    if demo:
        max_samples = max_samples if max_samples is not None else DEMO_SAMPLES

    train_path, eval_path = resolve_pref_paths(cfg, root)
    out_dir = resolve_path(output_dir_override or train_cfg.get("output_dir") or "outputs/dpo", root)
    ensure_dir(out_dir)
    metrics_path = root / "reports" / "dpo_train_metrics.json"
    ensure_dir(metrics_path.parent)

    print(f"[dpo] device={device} base={base_name} adapter={adapter_path} qlora={use_qlora} demo={demo}")
    print(f"[dpo] train_data={train_path}")
    print(f"[dpo] output_dir={out_dir}")

    reset_peak_gpu_memory()
    t0 = time.perf_counter()

    model, tokenizer = load_policy_model(
        base_name,
        adapter_path,
        model_cfg,
        use_qlora=use_qlora,
        device=device,
        demo=demo,
    )

    if merge_adapter and adapter_path is not None and hasattr(model, "merge_and_unload"):
        print("[dpo] merging SFT adapter into base before DPO LoRA (optional path)")
        # After merge, re-apply fresh LoRA for DPO
        from peft import LoraConfig, TaskType, get_peft_model

        model = model.merge_and_unload()
        peft_cfg = LoraConfig(
            r=int(lora_cfg.get("r", 16)),
            lora_alpha=int(lora_cfg.get("lora_alpha", 32)),
            lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
            bias=str(lora_cfg.get("bias", "none")),
            task_type=TaskType.CAUSAL_LM,
            target_modules=list(
                lora_cfg.get(
                    "target_modules",
                    ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                )
            ),
        )
        model = get_peft_model(model, peft_cfg)

    train_ds = build_pref_dataset(train_path, tokenizer, data_cfg, max_samples=max_samples)
    eval_ds = None
    if eval_path and not demo:
        eval_ds = build_pref_dataset(eval_path, tokenizer, data_cfg, max_samples=None)

    dpo_args = build_dpo_config(
        train_cfg,
        dpo_cfg,
        output_dir=str(out_dir),
        demo=demo,
        device=device,
        has_eval=eval_ds is not None,
        seed=seed,
    )

    from trl import DPOTrainer

    # TRL 0.12+: ref_model=None uses implicit reference (policy copy / disable adapter)
    ref_model = model_cfg.get("ref_model")
    if ref_model in (None, "null", ""):
        ref = None
    else:
        ref = str(ref_model)

    trainer_kwargs: dict[str, Any] = dict(
        model=model,
        ref_model=ref,
        args=dpo_args,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )
    if eval_ds is not None:
        trainer_kwargs["eval_dataset"] = eval_ds

    try:
        trainer = DPOTrainer(**trainer_kwargs)
    except TypeError:
        trainer_kwargs.pop("processing_class", None)
        trainer_kwargs["tokenizer"] = tokenizer
        # Older TRL may need beta on trainer
        trainer_kwargs.setdefault("beta", float(dpo_cfg.get("beta", 0.1)))
        try:
            trainer = DPOTrainer(**trainer_kwargs)
        except TypeError as e:
            # Last resort: drop unknown
            import inspect

            sig = inspect.signature(DPOTrainer.__init__)
            allowed = set(sig.parameters.keys()) - {"self"}
            filtered = {k: v for k, v in trainer_kwargs.items() if k in allowed}
            trainer = DPOTrainer(**filtered)

    train_result = trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    wall = time.perf_counter() - t0
    peak_mb = peak_gpu_memory_mb()
    metrics: dict[str, Any] = {
        "stage": "dpo",
        "demo": demo,
        "base_model": base_name,
        "sft_adapter": str(adapter_path) if adapter_path else None,
        "use_qlora": use_qlora,
        "device": device,
        "train_path": str(train_path),
        "eval_path": str(eval_path) if eval_path else None,
        "output_dir": str(out_dir),
        "num_train_samples": len(train_ds),
        "train_wall_time_sec": round(wall, 3),
        "peak_gpu_memory_mb": round(peak_mb, 2) if peak_mb is not None else None,
        "seed": seed,
        "beta": float(dpo_cfg.get("beta", 0.1)),
        "train_loss": None,
        "metrics": {},
    }
    if train_result is not None and getattr(train_result, "metrics", None):
        metrics["metrics"] = dict(train_result.metrics)
        metrics["train_loss"] = train_result.metrics.get("train_loss")

    save_json(metrics_path, metrics)
    print(f"[dpo] done in {wall:.1f}s | peak_gpu_mb={metrics['peak_gpu_memory_mb']}")
    print(f"[dpo] adapter saved to {out_dir}")
    print(f"[dpo] metrics -> {metrics_path}")
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DPO training for llm-post-training-lab")
    p.add_argument("--config", type=str, default="configs/dpo.yaml")
    p.add_argument("--demo", action="store_true", help="Tiny subset, 1 epoch; CPU OK; skip 4-bit")
    p.add_argument("--model", type=str, default=None, help="Override base model or adapter path")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument(
        "--merge-adapter",
        action="store_true",
        help="Merge SFT adapter into base then attach new LoRA for DPO",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_dpo(
            config_path=args.config,
            demo=args.demo,
            model_name_override=args.model,
            max_samples=args.max_samples,
            output_dir_override=args.output_dir,
            merge_adapter=args.merge_adapter,
        )
    except Exception as e:
        print(f"[dpo] ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
