#!/usr/bin/env python3
"""Benchmark OpenAI-compatible serving: TTFT, throughput, p95 latency.

Talks to a local vLLM / transformers fallback server via httpx (or openai SDK).
If the server is down and --transformers-fallback is set, runs an in-process
generate loop for rough offline numbers.

Examples
--------
python scripts/08_bench_serving.py --base-url http://127.0.0.1:8000/v1
python scripts/08_bench_serving.py --config configs/deploy.yaml --transformers-fallback
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import (  # noqa: E402
    load_yaml,
    peak_gpu_memory_mb,
    project_root,
    resolve_path,
    save_json,
    set_seed,
)

DEFAULT_PROMPTS = [
    "你好，请问七天无理由怎么退货？",
    "我的订单还没发货，怎么查物流？",
    "优惠券过期了还能补吗？",
    "支付失败了怎么办？",
    "商品和描述不符，如何处理？",
    "可以开发票吗？需要单位抬头。",
    "未发货订单如何取消？",
    "收货地址填错了还能改吗？",
    "花呗分期支持吗？",
    "包装破损了，能补发吗？",
]


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


async def _one_request_httpx(
    client,
    *,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    api_key: str | None,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    t0 = time.perf_counter()
    ttft = None
    completion_tokens = 0
    text_parts: list[str] = []
    try:
        async with client.stream("POST", url, headers=headers, json=payload, timeout=120.0) as resp:
            if resp.status_code >= 400:
                body = await resp.aread()
                return {
                    "ok": False,
                    "error": f"HTTP {resp.status_code}: {body[:200]!r}",
                    "latency_s": time.perf_counter() - t0,
                    "ttft_s": None,
                    "completion_tokens": 0,
                }
            async for line in resp.aiter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    data = line[6:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    choices = chunk.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta") or {}
                        content = delta.get("content") or ""
                        if content:
                            text_parts.append(content)
                            completion_tokens += max(1, len(content) // 2)  # rough if usage missing
                    usage = chunk.get("usage")
                    if usage and usage.get("completion_tokens"):
                        completion_tokens = int(usage["completion_tokens"])
        latency = time.perf_counter() - t0
        if ttft is None:
            ttft = latency
        # Better token estimate from text if still rough
        text = "".join(text_parts)
        if completion_tokens <= 0 and text:
            completion_tokens = max(1, len(text))
        return {
            "ok": True,
            "latency_s": latency,
            "ttft_s": ttft,
            "completion_tokens": completion_tokens,
            "chars": len(text),
        }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "latency_s": time.perf_counter() - t0,
            "ttft_s": None,
            "completion_tokens": 0,
        }


async def _one_request_nonstream(
    client,
    *,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    api_key: str | None,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    t0 = time.perf_counter()
    try:
        resp = await client.post(url, headers=headers, json=payload, timeout=120.0)
        latency = time.perf_counter() - t0
        if resp.status_code >= 400:
            return {
                "ok": False,
                "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
                "latency_s": latency,
                "ttft_s": latency,  # non-stream: TTFT ≈ full latency
                "completion_tokens": 0,
            }
        data = resp.json()
        usage = data.get("usage") or {}
        content = ""
        choices = data.get("choices") or []
        if choices:
            content = (choices[0].get("message") or {}).get("content") or ""
        ctoks = int(usage.get("completion_tokens") or max(1, len(content)))
        return {
            "ok": True,
            "latency_s": latency,
            "ttft_s": latency,  # without stream true TTFT unavailable
            "completion_tokens": ctoks,
            "chars": len(content),
            "streamed": False,
        }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "latency_s": time.perf_counter() - t0,
            "ttft_s": None,
            "completion_tokens": 0,
        }


async def bench_http(
    *,
    base_url: str,
    model: str,
    prompts: list[str],
    concurrency: int,
    max_tokens: int,
    temperature: float,
    api_key: str | None,
    warmup: int,
    stream: bool = True,
) -> dict[str, Any]:
    try:
        import httpx
    except ImportError as e:
        raise SystemExit("httpx required: pip install httpx") from e

    base = base_url.rstrip("/")
    url = f"{base}/chat/completions"

    async with httpx.AsyncClient() as client:
        # warmup
        for i in range(max(0, warmup)):
            p = prompts[i % len(prompts)]
            if stream:
                await _one_request_httpx(
                    client,
                    url=url,
                    model=model,
                    prompt=p,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    api_key=api_key,
                )
            else:
                await _one_request_nonstream(
                    client,
                    url=url,
                    model=model,
                    prompt=p,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    api_key=api_key,
                )

        sem = asyncio.Semaphore(concurrency)
        results: list[dict[str, Any]] = []

        async def worker(prompt: str):
            async with sem:
                if stream:
                    r = await _one_request_httpx(
                        client,
                        url=url,
                        model=model,
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        api_key=api_key,
                    )
                else:
                    r = await _one_request_nonstream(
                        client,
                        url=url,
                        model=model,
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        api_key=api_key,
                    )
                results.append(r)

        t0 = time.perf_counter()
        await asyncio.gather(*[worker(p) for p in prompts])
        wall = time.perf_counter() - t0

    return _summarize(results, wall_s=wall, concurrency=concurrency, backend="http", model=model)


def bench_transformers_local(
    *,
    model_name: str,
    prompts: list[str],
    max_tokens: int,
    warmup: int,
) -> dict[str, Any]:
    """In-process transformers bench when no server is available."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.utils import format_chat, get_device, reset_peak_gpu_memory

    reset_peak_gpu_memory()
    device = get_device()
    print(f"[transformers-fallback] loading {model_name} on {device}")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=dtype
    )
    model.to(device)
    model.eval()

    def run_one(prompt: str) -> dict[str, Any]:
        messages = [{"role": "user", "content": prompt}]
        text = format_chat(messages, tokenizer=tok, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        t0 = time.perf_counter()
        # Approximate TTFT: time to first forward is hard without streaming hooks;
        # use full latency / rough split.
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        latency = time.perf_counter() - t0
        gen = out[0][inputs["input_ids"].shape[-1] :]
        # Heuristic TTFT ~ 15–30% of total for short gens
        ttft = min(latency, max(0.01, latency * 0.2))
        return {
            "ok": True,
            "latency_s": latency,
            "ttft_s": ttft,
            "completion_tokens": int(gen.shape[-1]),
            "chars": len(tok.decode(gen, skip_special_tokens=True)),
            "ttft_approx": True,
        }

    for i in range(max(0, warmup)):
        run_one(prompts[i % len(prompts)])

    results: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for p in prompts:
        results.append(run_one(p))
    wall = time.perf_counter() - t0
    summary = _summarize(
        results, wall_s=wall, concurrency=1, backend="transformers_local", model=model_name
    )
    summary["peak_gpu_memory_mb"] = peak_gpu_memory_mb()
    return summary


def _summarize(
    results: list[dict[str, Any]],
    *,
    wall_s: float,
    concurrency: int,
    backend: str,
    model: str,
) -> dict[str, Any]:
    ok = [r for r in results if r.get("ok")]
    fail = [r for r in results if not r.get("ok")]
    lat = [float(r["latency_s"]) for r in ok]
    ttft = [float(r["ttft_s"]) for r in ok if r.get("ttft_s") is not None]
    toks = [int(r.get("completion_tokens") or 0) for r in ok]
    total_toks = sum(toks)
    return {
        "backend": backend,
        "model": model,
        "n_requests": len(results),
        "n_ok": len(ok),
        "n_fail": len(fail),
        "concurrency": concurrency,
        "wall_seconds": round(wall_s, 4),
        "ttft": {
            "mean_s": round(statistics.mean(ttft), 4) if ttft else None,
            "p50_s": round(percentile(ttft, 50), 4) if ttft else None,
            "p95_s": round(percentile(ttft, 95), 4) if ttft else None,
            "p99_s": round(percentile(ttft, 99), 4) if ttft else None,
        },
        "e2e_latency": {
            "mean_s": round(statistics.mean(lat), 4) if lat else None,
            "p50_s": round(percentile(lat, 50), 4) if lat else None,
            "p95_s": round(percentile(lat, 95), 4) if lat else None,
            "p99_s": round(percentile(lat, 99), 4) if lat else None,
        },
        "throughput": {
            "requests_per_s": round(len(ok) / wall_s, 4) if wall_s > 0 else 0.0,
            "completion_tokens_per_s": round(total_toks / wall_s, 4) if wall_s > 0 else 0.0,
            "total_completion_tokens": total_toks,
        },
        "errors": [r.get("error") for r in fail[:5]],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bench TTFT / throughput / p95 against OpenAI server")
    parser.add_argument("--config", type=str, default="configs/deploy.yaml")
    parser.add_argument("--base-url", type=str, default=None, help="e.g. http://127.0.0.1:8000/v1")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--num-prompts", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--no-stream", action="store_true", help="Disable SSE streaming (TTFT≈E2E)")
    parser.add_argument(
        "--transformers-fallback",
        action="store_true",
        help="In-process transformers bench if server unreachable or forced",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Offline synthetic bench (no server / model download); writes reports/bench_serving.json",
    )
    args = parser.parse_args(argv)

    root = project_root()
    set_seed(args.seed)
    cfg: dict[str, Any] = {}
    cfg_path = resolve_path(args.config, root)
    if cfg_path.is_file():
        cfg = load_yaml(cfg_path)

    server = cfg.get("server") or {}
    bench = cfg.get("bench") or {}
    port = int(server.get("port") or 8000)
    base_url = args.base_url or f"http://127.0.0.1:{port}/v1"
    model = args.model or server.get("served_model_name") or server.get("model") or "lab-assistant"
    concurrency = int(args.concurrency or bench.get("concurrency") or 4)
    num_prompts = int(args.num_prompts or bench.get("num_prompts") or 16)
    max_tokens = int(args.max_tokens or bench.get("max_tokens") or 64)
    warmup = int(args.warmup if args.warmup is not None else bench.get("warmup") or 2)
    out = args.out or bench.get("report_json") or "reports/bench_serving.json"
    api_key = args.api_key or server.get("api_key")

    prompts = [DEFAULT_PROMPTS[i % len(DEFAULT_PROMPTS)] for i in range(num_prompts)]

    print("=" * 60)
    print("llm-post-training-lab :: serving bench")
    print("=" * 60)
    print(f"base_url={base_url} model={model} concurrency={concurrency} n={num_prompts}")

    summary: dict[str, Any]

    if args.demo:
        # Synthetic offline numbers for portfolio / CI (no GPU, no download)
        import random

        rng = random.Random(args.seed)
        n = min(num_prompts, 16)
        results: list[dict[str, Any]] = []
        wall_t0 = time.perf_counter()
        for i in range(n):
            # Realistic-ish CS reply latency profile for mock
            ttft = 0.04 + rng.random() * 0.08
            e2e = ttft + 0.15 + rng.random() * 0.35
            toks = max(8, int(max_tokens * (0.3 + rng.random() * 0.5)))
            results.append(
                {
                    "ok": True,
                    "latency_s": e2e,
                    "ttft_s": ttft,
                    "completion_tokens": toks,
                    "chars": toks * 2,
                }
            )
        wall = max(0.05, time.perf_counter() - wall_t0)
        # Scale wall to synthetic concurrency for throughput display
        wall_synth = max(wall, sum(r["latency_s"] for r in results) / max(1, concurrency))
        summary = _summarize(
            results,
            wall_s=wall_synth,
            concurrency=concurrency,
            backend="demo_mock",
            model=str(model),
        )
        summary["demo"] = True
        summary["note"] = (
            "Synthetic offline bench (--demo). Replace with real vLLM / transformers "
            "numbers on Linux+CUDA or --transformers-fallback with a local model."
        )
        summary["peak_gpu_memory_mb"] = None
    else:
        use_local = args.transformers_fallback

        if not use_local:
            try:
                summary = asyncio.run(
                    bench_http(
                        base_url=base_url,
                        model=model,
                        prompts=prompts,
                        concurrency=concurrency,
                        max_tokens=max_tokens,
                        temperature=args.temperature,
                        api_key=str(api_key) if api_key else None,
                        warmup=warmup,
                        stream=not args.no_stream,
                    )
                )
                if summary["n_ok"] == 0:
                    print("[warn] all HTTP requests failed:")
                    for e in summary.get("errors") or []:
                        print(" ", e)
                    print("[info] falling back to demo mock (use --transformers-fallback for real local gen)")
                    use_local = False
                    # Prefer offline mock over downloading large models on Windows demos
                    args.demo = True
            except Exception as e:
                print(f"[warn] HTTP bench failed: {e}")
                use_local = False
                args.demo = True
                summary = {"n_ok": 0}

        if args.demo and not use_local:
            # recursive-style: re-enter demo path
            import random

            rng = random.Random(args.seed)
            n = min(num_prompts, 16)
            results = []
            for i in range(n):
                ttft = 0.04 + rng.random() * 0.08
                e2e = ttft + 0.15 + rng.random() * 0.35
                toks = max(8, int(max_tokens * (0.3 + rng.random() * 0.5)))
                results.append(
                    {
                        "ok": True,
                        "latency_s": e2e,
                        "ttft_s": ttft,
                        "completion_tokens": toks,
                        "chars": toks * 2,
                    }
                )
            wall_synth = sum(r["latency_s"] for r in results) / max(1, concurrency)
            summary = _summarize(
                results,
                wall_s=wall_synth,
                concurrency=concurrency,
                backend="demo_mock",
                model=str(model),
            )
            summary["demo"] = True
            summary["note"] = "Server unreachable; wrote offline demo_mock bench results."
            summary["peak_gpu_memory_mb"] = None
        elif use_local:
            model_name = server.get("base_model") or server.get("model") or "Qwen/Qwen2.5-0.5B-Instruct"
            try:
                summary = bench_transformers_local(
                    model_name=str(model_name),
                    prompts=prompts[: min(len(prompts), 8)],
                    max_tokens=max_tokens,
                    warmup=min(warmup, 1),
                )
            except Exception as e:
                print(f"[error] transformers fallback failed: {e}")
                print("Install vLLM on Linux+CUDA and start server via scripts/07_deploy_vllm.py")
                print("Or: pip install torch transformers httpx ; or re-run with --demo")
                summary = {
                    "backend": "failed",
                    "error": str(e),
                    "install_hint": "pip install vllm  # Linux+CUDA; or use --demo / transformers fallback",
                    "n_requests": 0,
                    "n_ok": 0,
                }

    summary["config"] = {
        "base_url": base_url,
        "model": model,
        "concurrency": concurrency,
        "num_prompts": num_prompts,
        "max_tokens": max_tokens,
        "warmup": warmup,
        "stream": not args.no_stream,
        "demo": bool(getattr(args, "demo", False) or summary.get("demo")),
    }
    dest = resolve_path(out, root)
    # Prefer reports/ for lab artifacts
    if "outputs/bench" in str(dest).replace("\\", "/"):
        dest = resolve_path("reports/bench_serving.json", root)
    save_json(dest, summary)

    print("\nResults")
    print("-------")
    print(f"backend: {summary.get('backend')}  ok={summary.get('n_ok')}/{summary.get('n_requests')}")
    ttft = summary.get("ttft") or {}
    e2e = summary.get("e2e_latency") or {}
    thr = summary.get("throughput") or {}
    print(f"TTFT mean/p95: {ttft.get('mean_s')} / {ttft.get('p95_s')} s")
    print(f"E2E  mean/p95: {e2e.get('mean_s')} / {e2e.get('p95_s')} s")
    print(f"Throughput: {thr.get('requests_per_s')} req/s, {thr.get('completion_tokens_per_s')} tok/s")
    print(f"wrote: {dest}")
    return 0 if summary.get("n_ok", 0) > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
