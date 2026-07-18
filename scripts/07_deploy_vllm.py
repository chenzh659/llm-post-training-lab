#!/usr/bin/env python3
"""Launch helper + documentation for a vLLM OpenAI-compatible server.

vLLM primarily targets Linux + NVIDIA CUDA. On Windows/macOS this script prints
install guidance and can optionally fall back to a tiny transformers+FastAPI
demo server for local API-shape testing.

Examples
--------
# Print recommended install + launch command (default):
python scripts/07_deploy_vllm.py --config configs/deploy.yaml

# Actually spawn vLLM (Linux+CUDA, package installed):
python scripts/07_deploy_vllm.py --config configs/deploy.yaml --run

# Transformers fallback OpenAI-ish server (CPU OK, slow):
python scripts/07_deploy_vllm.py --fallback-transformers --run --port 8000
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import load_yaml, project_root, resolve_path  # noqa: E402


INSTALL_HELP = """
vLLM install (Linux + NVIDIA CUDA recommended)
==============================================
# Create env with matching CUDA torch first, then:
pip install vllm

# If build fails, use official wheels for your CUDA version, e.g.:
#   https://docs.vllm.ai/en/latest/getting_started/installation.html

Windows note
------------
vLLM does not officially support Windows. Options:
  1) WSL2 + Ubuntu + CUDA drivers
  2) Linux GPU machine / cloud GPU
  3) This script's --fallback-transformers server (demo only)

macOS note
----------
Use --fallback-transformers or remote GPU host; vLLM needs NVIDIA GPUs.
""".strip()


def build_vllm_command(cfg: dict[str, Any]) -> list[str]:
    server = cfg.get("server") or {}
    model = server.get("model") or "Qwen/Qwen2.5-0.5B-Instruct"
    host = server.get("host") or "0.0.0.0"
    port = int(server.get("port") or 8000)
    dtype = server.get("dtype") or "auto"
    max_len = int(server.get("max_model_len") or 4096)
    gpu_util = float(server.get("gpu_memory_utilization") or 0.85)
    tp = int(server.get("tensor_parallel_size") or 1)
    served = server.get("served_model_name") or "lab-assistant"
    trust = server.get("trust_remote_code", True)

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(model),
        "--host",
        str(host),
        "--port",
        str(port),
        "--dtype",
        str(dtype),
        "--max-model-len",
        str(max_len),
        "--gpu-memory-utilization",
        str(gpu_util),
        "--tensor-parallel-size",
        str(tp),
        "--served-model-name",
        str(served),
    ]
    if trust:
        cmd.append("--trust-remote-code")
    if server.get("enforce_eager"):
        cmd.append("--enforce-eager")
    if server.get("enable_prefix_caching"):
        cmd.append("--enable-prefix-caching")
    # LoRA base hint (document only — merging adapters is separate)
    if server.get("base_model"):
        # vllm can load adapters via --enable-lora in newer versions; keep note in print
        pass
    return cmd


def print_docs(cfg: dict[str, Any]) -> None:
    cmd = build_vllm_command(cfg)
    server = cfg.get("server") or {}
    port = server.get("port") or 8000
    served = server.get("served_model_name") or "lab-assistant"
    print(INSTALL_HELP)
    print()
    print("Launch command")
    print("--------------")
    print(" ".join(cmd))
    print()
    print("Smoke test (OpenAI-compatible)")
    print("------------------------------")
    print(
        textwrap.dedent(
            f"""
            curl http://127.0.0.1:{port}/v1/models
            curl http://127.0.0.1:{port}/v1/chat/completions \\
              -H "Content-Type: application/json" \\
              -d '{{"model":"{served}","messages":[{{"role":"user","content":"你好，如何退货？"}}],"max_tokens":128}}'
            """
        ).strip()
    )
    print()
    print("Benchmark")
    print("---------")
    print(f"python scripts/08_bench_serving.py --base-url http://127.0.0.1:{port}/v1 --model {served}")
    if server.get("base_model") and server.get("model") != server.get("base_model"):
        print()
        print(
            f"Note: config.server.model={server.get('model')} may be a LoRA adapter. "
            f"Merge adapters into a full model or serve base={server.get('base_model')} with --enable-lora."
        )


def run_vllm(cfg: dict[str, Any]) -> int:
    try:
        import vllm  # noqa: F401
    except ImportError:
        print("vLLM is not installed in this environment.\n")
        print(INSTALL_HELP)
        print("\nRe-run with --fallback-transformers --run for a CPU demo server.")
        return 2
    cmd = build_vllm_command(cfg)
    print("Executing:", " ".join(cmd))
    return subprocess.call(cmd)


def run_transformers_server(cfg: dict[str, Any], host: str, port: int) -> int:
    """Minimal OpenAI-compatible chat completions server using transformers."""
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel, Field
        import uvicorn
    except ImportError:
        print("Need fastapi + uvicorn: pip install fastapi uvicorn")
        return 2

    server = cfg.get("server") or {}
    model_name = server.get("base_model") or server.get("model") or "Qwen/Qwen2.5-0.5B-Instruct"
    served = server.get("served_model_name") or "lab-assistant"

    print(f"[fallback] Loading transformers model: {model_name}")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.utils import format_chat, get_device

    device = get_device()
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=dtype
    )
    model.to(device)
    model.eval()

    app = FastAPI(title="llm-post-training-lab transformers fallback")

    class Msg(BaseModel):
        role: str
        content: str

    class ChatReq(BaseModel):
        model: str = served
        messages: list[Msg]
        max_tokens: int = 128
        temperature: float = 0.7
        top_p: float = 0.9

    @app.get("/health")
    def health():
        return {"status": "ok", "backend": "transformers", "device": device}

    @app.get("/v1/models")
    def models():
        return {
            "object": "list",
            "data": [{"id": served, "object": "model", "owned_by": "lab"}],
        }

    @app.post("/v1/chat/completions")
    def chat(req: ChatReq):
        import time as _time

        t0 = _time.perf_counter()
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        prompt = format_chat(messages, tokenizer=tok, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        do_sample = req.temperature is not None and req.temperature > 0
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=int(req.max_tokens or 128),
                do_sample=do_sample,
                temperature=max(float(req.temperature or 0.7), 1e-5) if do_sample else None,
                top_p=float(req.top_p or 0.9) if do_sample else None,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[-1] :]
        text = tok.decode(gen, skip_special_tokens=True)
        latency = _time.perf_counter() - t0
        return {
            "id": "chatcmpl-lab",
            "object": "chat.completion",
            "model": req.model or served,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": int(inputs["input_ids"].shape[-1]),
                "completion_tokens": int(gen.shape[-1]),
                "total_tokens": int(inputs["input_ids"].shape[-1] + gen.shape[-1]),
            },
            "lab_latency_s": latency,
        }

    print(f"[fallback] Serving on http://{host}:{port}  model={served}")
    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="vLLM deploy helper for llm-post-training-lab")
    parser.add_argument("--config", type=str, default="configs/deploy.yaml")
    parser.add_argument("--run", action="store_true", help="Actually start the server")
    parser.add_argument(
        "--fallback-transformers",
        action="store_true",
        help="Use transformers+FastAPI instead of vLLM",
    )
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args(argv)

    root = project_root()
    cfg_path = resolve_path(args.config, root)
    cfg: dict[str, Any] = {}
    if cfg_path.is_file():
        cfg = load_yaml(cfg_path)
    cfg.setdefault("server", {})
    if args.model:
        cfg["server"]["model"] = args.model
    if args.host:
        cfg["server"]["host"] = args.host
    if args.port:
        cfg["server"]["port"] = args.port

    print("=" * 60)
    print("llm-post-training-lab :: deploy / vLLM")
    print("=" * 60)

    if not args.run:
        print_docs(cfg)
        print("\n(Tip: pass --run to launch, or --fallback-transformers --run)")
        return 0

    host = str(cfg["server"].get("host") or "0.0.0.0")
    port = int(cfg["server"].get("port") or 8000)

    if args.fallback_transformers:
        return run_transformers_server(cfg, host, port)

    # Auto-fallback message if no vllm
    try:
        import vllm  # noqa: F401
    except ImportError:
        print("vLLM not installed — printing instructions. Use --fallback-transformers --run for demo.\n")
        print_docs(cfg)
        return 2

    return run_vllm(cfg)


if __name__ == "__main__":
    sys.exit(main())
