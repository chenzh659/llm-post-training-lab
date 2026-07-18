"""Clean SFT and preference JSONL: dedup, length filters, toxic keywords."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore

# Approx tokens for Chinese: chars / 1.5
CHARS_PER_TOKEN = 1.5

TOXIC_KEYWORDS = [
    "去死",
    "傻逼",
    "傻b",
    "煞笔",
    "操你妈",
    "他妈的",
    "白痴",
    "智障",
    "滚粗",
    "操你",
    "妈的",
    "废物客服",
    "人渣",
    "自杀教程",
    "制毒",
    "色情服务",
]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: Path | None) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "cleaning": {
            "min_chars": 8,
            "max_chars": 12000,
            "min_prompt_chars": 4,
            "min_response_chars": 8,
            "max_response_chars": 8000,
            "min_tokens_approx": 4,
            "max_tokens_approx": 8000,
            "drop_empty": True,
            "drop_duplicates": True,
            "normalize_whitespace": True,
        }
    }
    if path is None or not path.is_file():
        return defaults
    if yaml is None:
        return defaults
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cleaning = dict(defaults["cleaning"])
    cleaning.update(cfg.get("cleaning") or {})
    cfg["cleaning"] = cleaning
    return cfg


def normalize_text(text: str) -> str:
    text = text or ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text, flags=re.UNICODE).strip()
    return text


def text_hash(text: str) -> str:
    return hashlib.md5(normalize_text(text).encode("utf-8")).hexdigest()


def approx_tokens(text: str) -> float:
    return len(text) / CHARS_PER_TOKEN


def contains_toxic(text: str) -> bool:
    t = text or ""
    low = t.lower()
    for kw in TOXIC_KEYWORDS:
        if kw.lower() in low or kw in t:
            return True
    return False


def extract_sft_user_assistant(row: dict[str, Any]) -> tuple[str, str]:
    messages = row.get("messages") or []
    user_parts: list[str] = []
    asst_parts: list[str] = []
    for m in messages:
        role = (m.get("role") or "").lower()
        content = m.get("content") or ""
        if role == "user":
            user_parts.append(content)
        elif role == "assistant":
            asst_parts.append(content)
    return "\n".join(user_parts), "\n".join(asst_parts)


def sft_dedup_key(row: dict[str, Any]) -> str:
    user, asst = extract_sft_user_assistant(row)
    return text_hash(user + "\n" + asst)


def pref_dedup_key(row: dict[str, Any]) -> str:
    prompt = row.get("prompt") or ""
    chosen = row.get("chosen") or ""
    return text_hash(prompt + "\n" + chosen)


def filter_sft_row(row: dict[str, Any], cleaning: dict[str, Any]) -> tuple[bool, str]:
    user, asst = extract_sft_user_assistant(row)
    if cleaning.get("drop_empty", True):
        if not user.strip() or not asst.strip():
            return False, "empty"
    full = user + asst
    min_c = int(cleaning.get("min_chars", 8))
    max_c = int(cleaning.get("max_chars", 12000))
    min_p = int(cleaning.get("min_prompt_chars", 4))
    min_r = int(cleaning.get("min_response_chars", 8))
    max_r = int(cleaning.get("max_response_chars", 8000))
    if len(user) < min_p:
        return False, "prompt_too_short"
    if len(asst) < min_r:
        return False, "response_too_short"
    if len(asst) > max_r or len(full) > max_c:
        return False, "too_long"
    if len(full) < min_c:
        return False, "too_short"
    min_tok = float(cleaning.get("min_tokens_approx", 4))
    max_tok = float(cleaning.get("max_tokens_approx", 8000))
    tok = approx_tokens(full)
    if tok < min_tok:
        return False, "tokens_too_short"
    if tok > max_tok:
        return False, "tokens_too_long"
    if contains_toxic(user) or contains_toxic(asst):
        return False, "toxic"
    return True, "ok"


def filter_pref_row(row: dict[str, Any], cleaning: dict[str, Any]) -> tuple[bool, str]:
    prompt = (row.get("prompt") or "").strip()
    chosen = (row.get("chosen") or "").strip()
    rejected = (row.get("rejected") or "").strip()
    if cleaning.get("drop_empty", True):
        if not prompt or not chosen or not rejected:
            return False, "empty"
    min_p = int(cleaning.get("min_prompt_chars", 4))
    min_r = int(cleaning.get("min_response_chars", 8))
    max_c = int(cleaning.get("max_chars", 12000))
    if len(prompt) < min_p:
        return False, "prompt_too_short"
    if len(chosen) < min_r:
        return False, "chosen_too_short"
    if len(rejected) < 1:
        return False, "rejected_empty"
    if len(prompt) + len(chosen) > max_c:
        return False, "too_long"
    # toxic on chosen (good answer must be clean); rejected may be rude by design
    if contains_toxic(chosen):
        return False, "toxic_chosen"
    if contains_toxic(prompt):
        return False, "toxic_prompt"
    return True, "ok"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_dataset(
    rows: list[dict[str, Any]],
    kind: str,
    cleaning: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stats: dict[str, Any] = {
        "kind": kind,
        "input": len(rows),
        "dropped": Counter(),
        "output": 0,
        "duplicates_removed": 0,
    }
    filter_fn = filter_sft_row if kind == "sft" else filter_pref_row
    key_fn = sft_dedup_key if kind == "sft" else pref_dedup_key

    kept: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        ok, reason = filter_fn(row, cleaning)
        if not ok:
            stats["dropped"][reason] += 1
            continue
        if cleaning.get("drop_duplicates", True):
            h = key_fn(row)
            if h in seen:
                stats["duplicates_removed"] += 1
                stats["dropped"]["duplicate"] += 1
                continue
            seen.add(h)
        kept.append(row)

    stats["dropped"] = dict(stats["dropped"])
    stats["output"] = len(kept)
    stats["retention_rate"] = (len(kept) / len(rows)) if rows else 0.0
    return kept, stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean SFT and preference JSONL")
    p.add_argument("--config", type=str, default="configs/data.yaml")
    p.add_argument("--sft-input", type=str, default="data/raw/sft_raw.jsonl")
    p.add_argument("--pref-input", type=str, default="data/raw/preference_raw.jsonl")
    p.add_argument("--sft-output", type=str, default="data/processed/sft_clean.jsonl")
    p.add_argument("--pref-output", type=str, default="data/processed/preference_clean.jsonl")
    p.add_argument(
        "--stats-output",
        type=str,
        default="reports/data_cleaning_stats.json",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = _project_root()

    def resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else (root / path).resolve()

    cfg_path = resolve(args.config)
    cfg = load_config(cfg_path)
    cleaning = cfg.get("cleaning") or {}

    sft_in = resolve(args.sft_input)
    pref_in = resolve(args.pref_input)
    sft_out = resolve(args.sft_output)
    pref_out = resolve(args.pref_output)
    stats_out = resolve(args.stats_output)

    sft_rows = read_jsonl(sft_in)
    pref_rows = read_jsonl(pref_in)

    sft_clean, sft_stats = clean_dataset(sft_rows, "sft", cleaning)
    pref_clean, pref_stats = clean_dataset(pref_rows, "preference", cleaning)

    write_jsonl(sft_out, sft_clean)
    write_jsonl(pref_out, pref_clean)

    report = {
        "sft": sft_stats,
        "preference": pref_stats,
        "paths": {
            "sft_input": str(sft_in),
            "sft_output": str(sft_out),
            "pref_input": str(pref_in),
            "pref_output": str(pref_out),
        },
    }
    stats_out.parent.mkdir(parents=True, exist_ok=True)
    with stats_out.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[clean] sft: {sft_stats['input']} -> {sft_stats['output']} "
          f"(retention={sft_stats['retention_rate']:.2%}) -> {sft_out}")
    print(f"[clean] pref: {pref_stats['input']} -> {pref_stats['output']} "
          f"(retention={pref_stats['retention_rate']:.2%}) -> {pref_out}")
    print(f"[clean] stats -> {stats_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
