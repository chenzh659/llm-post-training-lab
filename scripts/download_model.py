#!/usr/bin/env python3
"""Optional helper: prefetch a Hugging Face model into the local cache.

Does not run during package install. Requires network + ``transformers`` / ``huggingface_hub``.

Examples::

    python scripts/download_model.py
    python scripts/download_model.py --model Qwen/Qwen2.5-0.5B-Instruct
    python scripts/download_model.py --model Qwen/Qwen2.5-1.5B-Instruct --revision main
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download / cache a HF causal LM + tokenizer")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="HF model id or local path")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--no-trust-remote-code", action="store_true")
    args = parser.parse_args(argv)

    trust = not args.no_trust_remote_code
    model_id = args.model
    print(f"[download] model={model_id} trust_remote_code={trust}")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("[download] ERROR: transformers not installed. pip install transformers", file=sys.stderr)
        return 1

    try:
        tok_kwargs = {"trust_remote_code": trust}
        model_kwargs = {"trust_remote_code": trust}
        if args.revision:
            tok_kwargs["revision"] = args.revision
            model_kwargs["revision"] = args.revision

        print("[download] tokenizer ...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, **tok_kwargs)
        print("[download] model weights (this may take a while) ...")
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[download] OK | params={n_params:,} | vocab={len(tokenizer)}")
        print("[download] Cached under HF_HOME / TRANSFORMERS_CACHE as applicable.")
        return 0
    except Exception as e:
        print(f"[download] ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
