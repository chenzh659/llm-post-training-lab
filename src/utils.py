"""Shared utilities for llm-post-training-lab."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore


def configure_stdio_utf8() -> None:
    """Best-effort UTF-8 for stdout/stderr (fixes Windows console mojibake)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass


def project_root() -> Path:
    """Return repository root (parent of ``src/``)."""
    return Path(__file__).resolve().parents[1]


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a dict."""
    if yaml is None:
        raise ImportError("PyYAML is required: pip install pyyaml")
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at root of YAML: {p}")
    return data


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and torch (if available) for reproducibility.

    Import / DLL failures (common on broken Windows torch installs) are ignored
    so data-only scripts and unit tests still run without a working GPU stack.
    """
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        # ImportError, or broken native deps (e.g. WinError on torch/numpy DLL)
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        # ImportError, OSError (DLL), or runtime CUDA init failures
        pass


def get_device(prefer_cuda: bool = True) -> str:
    """Return best available device string: ``cuda``, ``mps``, or ``cpu``.

    Falls back to ``cpu`` if torch is missing or fails to load (e.g. broken DLLs).
    """
    try:
        import torch

        if prefer_cuda and torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        # ImportError or OSError when torch DLLs fail to load on Windows
        pass
    return "cpu"


def format_chat(
    messages: list[dict[str, str]],
    tokenizer: Any | None = None,
    add_generation_prompt: bool = False,
) -> str:
    """Format chat messages with the tokenizer chat template, or a simple fallback."""
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception:
            pass
    # Fallback plain formatting
    parts: list[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"<|{role}|>\n{content}")
    if add_generation_prompt:
        parts.append("<|assistant|>\n")
    return "\n".join(parts)


def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if missing; return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | Path, obj: Any, indent: int = 2) -> Path:
    """Write JSON with UTF-8; create parent dirs. Returns path written."""
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
        f.write("\n")
    return p


def load_json(path: str | Path) -> Any:
    """Load a JSON file."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts."""
    rows: list[dict[str, Any]] = []
    p = Path(path)
    if not p.is_file():
        return rows
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    """Write rows as JSONL; create parent dirs."""
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return p


def resolve_path(path: str | Path, root: Path | None = None) -> Path:
    """Resolve path relative to project root if not absolute."""
    p = Path(path)
    if p.is_absolute():
        return p
    base = root or project_root()
    return (base / p).resolve()


def peak_gpu_memory_mb() -> float | None:
    """Peak allocated GPU memory in MiB, or None if CUDA unavailable."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return float(torch.cuda.max_memory_allocated() / (1024**2))
    except Exception:
        return None


def reset_peak_gpu_memory() -> None:
    """Reset CUDA peak memory stats when available."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
    except Exception:
        pass


def add_src_to_path() -> Path:
    """Ensure project root is on ``sys.path``; return root."""
    root = project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root
