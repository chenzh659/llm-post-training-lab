"""Supervised Fine-Tuning (SFT) with PEFT LoRA / optional QLoRA via TRL."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

# Project root on path when run as script
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

# Demo defaults
DEMO_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEMO_SAMPLES = 32
DEMO_EPOCHS = 1


def _first_existing(candidates: list[Path]) -> Path | None:
    for p in candidates:
        if p.is_file():
            return p
    return None


def resolve_sft_paths(cfg: dict[str, Any], root: Path) -> tuple[Path, Path | None]:
    """Resolve train/eval JSONL paths with fallbacks to data/splits/."""
    data = cfg.get("data") or {}
    train_cfg = data.get("train_path") or "data/processed/sft_train.jsonl"
    eval_cfg = data.get("eval_path") or "data/processed/sft_eval.jsonl"

    train_cands = [
        resolve_path(train_cfg, root),
        root / "data" / "splits" / "train.sft.jsonl",
        root / "data" / "processed" / "sft_clean.jsonl",
        root / "data" / "raw" / "sft_raw.jsonl",
    ]
    eval_cands = [
        resolve_path(eval_cfg, root),
        root / "data" / "splits" / "val.sft.jsonl",
        root / "data" / "splits" / "test.sft.jsonl",
    ]
    train_path = _first_existing(train_cands)
    if train_path is None:
        raise FileNotFoundError(
            "SFT train JSONL not found. Tried:\n  "
            + "\n  ".join(str(p) for p in train_cands)
            + "\nRun: python scripts/01_build_data.py"
        )
    eval_path = _first_existing(eval_cands)
    return train_path, eval_path


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
    """Build BitsAndBytesConfig for 4-bit QLoRA when available."""
    from transformers import BitsAndBytesConfig
    import torch

    compute = _dtype_from_name(model_cfg.get("bnb_4bit_compute_dtype", "bfloat16"))
    # On older GPUs without bf16, fall back to fp16
    if compute == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        compute = torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute,
        bnb_4bit_quant_type=str(model_cfg.get("bnb_4bit_quant_type", "nf4")),
        bnb_4bit_use_double_quant=bool(model_cfg.get("bnb_4bit_use_double_quant", True)),
    )


def load_model_and_tokenizer(
    model_name: str,
    model_cfg: dict[str, Any],
    *,
    use_qlora: bool,
    device: str,
    demo: bool,
):
    """Load causal LM + tokenizer; optional 4-bit quantization."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    trust = bool(model_cfg.get("trust_remote_code", True))
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype = _dtype_from_name(model_cfg.get("torch_dtype", "bfloat16"))
    if device == "cpu":
        dtype = torch.float32

    kwargs: dict[str, Any] = {
        "trust_remote_code": trust,
        "torch_dtype": dtype,
    }
    attn = model_cfg.get("attn_implementation")
    if attn and device == "cuda" and not demo:
        kwargs["attn_implementation"] = attn

    if use_qlora:
        try:
            kwargs["quantization_config"] = build_bnb_config(model_cfg)
            kwargs["device_map"] = "auto"
        except Exception as e:
            raise RuntimeError(
                "QLoRA (bitsandbytes 4-bit) requested but failed to configure. "
                "Install bitsandbytes on a CUDA host, or set model.bits: null / use_qlora: false. "
                f"Underlying error: {e}"
            ) from e
    else:
        # Place on single device when not quantized
        if device == "cuda":
            kwargs["device_map"] = {"": 0}
        elif device == "cpu":
            kwargs["device_map"] = {"": "cpu"}

    # Disable cache for training
    if model_cfg.get("use_cache") is False:
        pass  # set after load on config

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    except OSError as e:
        raise RuntimeError(
            f"Failed to load model '{model_name}'. Ensure network access or local cache. "
            f"Optional: python scripts/download_model.py --model {model_name}\n{e}"
        ) from e

    if hasattr(model, "config"):
        model.config.use_cache = bool(model_cfg.get("use_cache", False))
        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def apply_lora(model, lora_cfg: dict[str, Any], use_qlora: bool):
    """Attach PEFT LoRA adapters."""
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    if use_qlora:
        model = prepare_model_for_kbit_training(model)

    task = lora_cfg.get("task_type", "CAUSAL_LM")
    try:
        task_type = TaskType[task] if isinstance(task, str) else task
    except KeyError:
        task_type = TaskType.CAUSAL_LM

    peft_config = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("lora_alpha", 32)),
        lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
        bias=str(lora_cfg.get("bias", "none")),
        task_type=task_type,
        target_modules=list(
            lora_cfg.get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )
        ),
    )
    model = get_peft_model(model, peft_config)
    try:
        model.print_trainable_parameters()
    except Exception:
        pass
    return model


def messages_to_text(example: dict[str, Any], tokenizer) -> str:
    """Convert a row with ``messages`` (or instruction fields) to a single training string."""
    messages = example.get("messages")
    if messages:
        return format_chat(messages, tokenizer=tokenizer, add_generation_prompt=False)
    # Fallback instruction-style
    instruction = example.get("instruction") or example.get("prompt") or ""
    inp = example.get("input") or ""
    output = example.get("output") or example.get("response") or example.get("chosen") or ""
    user_content = instruction if not inp else f"{instruction}\n{inp}"
    msgs = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output},
    ]
    return format_chat(msgs, tokenizer=tokenizer, add_generation_prompt=False)


def build_sft_dataset(path: Path, tokenizer, max_samples: int | None = None):
    """Load JSONL and map to ``text`` field for SFTTrainer."""
    from datasets import Dataset

    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"No samples in {path}")
    if max_samples is not None and max_samples > 0:
        rows = rows[:max_samples]

    texts = [messages_to_text(r, tokenizer) for r in rows]
    return Dataset.from_dict({"text": texts})


def build_sft_config(
    train_cfg: dict[str, Any],
    *,
    output_dir: str,
    demo: bool,
    device: str,
    has_eval: bool,
):
    """Construct TRL SFTConfig / TrainingArguments-compatible config."""
    import torch

    # Prefer SFTConfig (TRL 0.12+); fall back to TrainingArguments
    try:
        from trl import SFTConfig

        ConfigCls = SFTConfig
    except ImportError:
        from transformers import TrainingArguments

        ConfigCls = TrainingArguments

    epochs = DEMO_EPOCHS if demo else float(train_cfg.get("num_train_epochs", 2))
    batch = 1 if demo else int(train_cfg.get("per_device_train_batch_size", 2))
    eval_batch = 1 if demo else int(train_cfg.get("per_device_eval_batch_size", 2))
    grad_accum = 1 if demo else int(train_cfg.get("gradient_accumulation_steps", 8))
    lr = float(train_cfg.get("learning_rate", 2e-4))

    use_bf16 = bool(train_cfg.get("bf16", True)) and device == "cuda" and torch.cuda.is_bf16_supported()
    use_fp16 = bool(train_cfg.get("fp16", False)) and device == "cuda" and not use_bf16
    if demo and device == "cpu":
        use_bf16 = use_fp16 = False

    optim = train_cfg.get("optim", "paged_adamw_8bit")
    if demo or device == "cpu" or optim == "paged_adamw_8bit" and device != "cuda":
        optim = "adamw_torch"

    kwargs: dict[str, Any] = dict(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=eval_batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type=str(train_cfg.get("lr_scheduler_type", "cosine")),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.03)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
        logging_steps=int(train_cfg.get("logging_steps", 10)) if not demo else 1,
        save_strategy="no" if demo else str(train_cfg.get("save_strategy", "steps")),
        save_steps=int(train_cfg.get("save_steps", 200)),
        save_total_limit=int(train_cfg.get("save_total_limit", 2)),
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)) and not demo,
        optim=optim,
        report_to=train_cfg.get("report_to", "none"),
        remove_unused_columns=bool(train_cfg.get("remove_unused_columns", False)),
        seed=int(train_cfg.get("seed", 42)) if "seed" in train_cfg else 42,
        dataloader_num_workers=0 if demo or device == "cpu" else int(train_cfg.get("dataloader_num_workers", 0)),
    )

    if has_eval and not demo:
        kwargs["eval_strategy"] = str(train_cfg.get("eval_strategy", "steps"))
        kwargs["eval_steps"] = int(train_cfg.get("eval_steps", 100))
    else:
        # transformers/trl use evaluation_strategy or eval_strategy depending on version
        kwargs["eval_strategy"] = "no"

    # SFT-specific fields when using SFTConfig
    if ConfigCls.__name__ == "SFTConfig":
        kwargs["max_seq_length"] = int((train_cfg.get("max_seq_len") or 512) if demo else 2048)
        # packing can break tiny demos
        kwargs["packing"] = False if demo else bool(train_cfg.get("packing", False))
        kwargs["dataset_text_field"] = "text"

    # Drop unknown keys that older TrainingArguments reject
    try:
        return ConfigCls(**kwargs)
    except TypeError:
        # Retry without SFT-only keys / rename eval_strategy
        kwargs.pop("max_seq_length", None)
        kwargs.pop("packing", None)
        kwargs.pop("dataset_text_field", None)
        if "eval_strategy" in kwargs:
            kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
        # Filter to known fields loosely
        import inspect

        sig = inspect.signature(ConfigCls.__init__)
        allowed = set(sig.parameters.keys()) - {"self"}
        filtered = {k: v for k, v in kwargs.items() if k in allowed or not allowed}
        if allowed:
            filtered = {k: v for k, v in kwargs.items() if k in allowed}
        return ConfigCls(**filtered)


def run_sft(
    config_path: str | Path = "configs/sft.yaml",
    *,
    demo: bool = False,
    model_name_override: str | None = None,
    max_samples: int | None = None,
    output_dir_override: str | None = None,
) -> dict[str, Any]:
    """Execute SFT training and write metrics + adapter."""
    root = project_root()
    cfg = load_yaml(resolve_path(config_path, root))
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    device = get_device()
    model_cfg = dict(cfg.get("model") or {})
    lora_cfg = dict(cfg.get("lora") or {})
    data_cfg = dict(cfg.get("data") or {})
    train_cfg = dict(cfg.get("training") or {})
    train_cfg["seed"] = seed
    train_cfg["max_seq_len"] = data_cfg.get("max_seq_len", 2048)
    train_cfg["packing"] = data_cfg.get("packing", False)

    model_name = model_name_override or model_cfg.get("name") or DEMO_MODEL
    if demo:
        model_name = model_name_override or DEMO_MODEL
        max_samples = max_samples if max_samples is not None else DEMO_SAMPLES

    if not demo and device == "cpu":
        raise RuntimeError(
            "Full SFT requires a CUDA GPU (or set --demo for CPU smoke with a tiny subset). "
            f"Detected device={device}."
        )

    use_qlora = _want_qlora(model_cfg, demo=demo, device=device)

    train_path, eval_path = resolve_sft_paths(cfg, root)
    out_dir = resolve_path(output_dir_override or train_cfg.get("output_dir") or "outputs/sft", root)
    ensure_dir(out_dir)
    metrics_path = root / "reports" / "sft_train_metrics.json"
    ensure_dir(metrics_path.parent)

    print(f"[sft] device={device} model={model_name} qlora={use_qlora} demo={demo}")
    print(f"[sft] train_data={train_path}")
    if eval_path:
        print(f"[sft] eval_data={eval_path}")
    print(f"[sft] output_dir={out_dir}")

    reset_peak_gpu_memory()
    t0 = time.perf_counter()

    model, tokenizer = load_model_and_tokenizer(
        model_name,
        model_cfg,
        use_qlora=use_qlora,
        device=device,
        demo=demo,
    )
    model = apply_lora(model, lora_cfg, use_qlora=use_qlora)

    train_ds = build_sft_dataset(train_path, tokenizer, max_samples=max_samples)
    eval_ds = None
    if eval_path and not demo:
        eval_ds = build_sft_dataset(eval_path, tokenizer, max_samples=None)

    # Cap max seq for demo
    if demo:
        train_cfg["max_seq_len"] = min(int(data_cfg.get("max_seq_len", 512)), 512)

    sft_args = build_sft_config(
        train_cfg,
        output_dir=str(out_dir),
        demo=demo,
        device=device,
        has_eval=eval_ds is not None,
    )

    from trl import SFTTrainer

    trainer_kwargs: dict[str, Any] = dict(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        processing_class=tokenizer,  # TRL >= 0.12
    )
    # Compatibility: older TRL uses tokenizer=
    try:
        if eval_ds is not None:
            trainer_kwargs["eval_dataset"] = eval_ds
        # dataset_text_field may be on trainer or config
        if not hasattr(sft_args, "dataset_text_field"):
            trainer_kwargs["dataset_text_field"] = "text"
        if not hasattr(sft_args, "max_seq_length") and not hasattr(sft_args, "max_length"):
            trainer_kwargs["max_seq_length"] = int(train_cfg.get("max_seq_len", 2048))
        trainer = SFTTrainer(**trainer_kwargs)
    except TypeError:
        trainer_kwargs.pop("processing_class", None)
        trainer_kwargs["tokenizer"] = tokenizer
        trainer_kwargs.setdefault("dataset_text_field", "text")
        trainer_kwargs.setdefault("max_seq_length", int(train_cfg.get("max_seq_len", 512 if demo else 2048)))
        if eval_ds is not None:
            trainer_kwargs["eval_dataset"] = eval_ds
        trainer = SFTTrainer(**trainer_kwargs)

    train_result = trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    wall = time.perf_counter() - t0
    peak_mb = peak_gpu_memory_mb()
    metrics: dict[str, Any] = {
        "stage": "sft",
        "demo": demo,
        "model_name": model_name,
        "use_qlora": use_qlora,
        "device": device,
        "train_path": str(train_path),
        "eval_path": str(eval_path) if eval_path else None,
        "output_dir": str(out_dir),
        "num_train_samples": len(train_ds),
        "train_wall_time_sec": round(wall, 3),
        "peak_gpu_memory_mb": round(peak_mb, 2) if peak_mb is not None else None,
        "seed": seed,
        "train_loss": None,
        "metrics": {},
    }
    if train_result is not None and getattr(train_result, "metrics", None):
        metrics["metrics"] = dict(train_result.metrics)
        metrics["train_loss"] = train_result.metrics.get("train_loss")

    save_json(metrics_path, metrics)
    print(f"[sft] done in {wall:.1f}s | peak_gpu_mb={metrics['peak_gpu_memory_mb']}")
    print(f"[sft] adapter saved to {out_dir}")
    print(f"[sft] metrics -> {metrics_path}")
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SFT / QLoRA training for llm-post-training-lab")
    p.add_argument("--config", type=str, default="configs/sft.yaml")
    p.add_argument("--demo", action="store_true", help="Tiny subset, 1 epoch; CPU OK; skip 4-bit")
    p.add_argument("--model", type=str, default=None, help="Override model name")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_sft(
            config_path=args.config,
            demo=args.demo,
            model_name_override=args.model,
            max_samples=args.max_samples,
            output_dir_override=args.output_dir,
        )
    except Exception as e:
        print(f"[sft] ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
