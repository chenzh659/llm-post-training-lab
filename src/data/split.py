"""Train/val/test split for SFT and preference JSONL (stratified by category when present)."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


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


def category_of(row: dict[str, Any]) -> str | None:
    for key in ("category", "scenario"):
        v = row.get(key)
        if v:
            return str(v)
    return None


def stratified_split(
    rows: list[dict[str, Any]],
    ratios: tuple[float, float, float],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """8:1:1 style split; stratified if category exists else pure random."""
    train_r, val_r, test_r = ratios
    total_r = train_r + val_r + test_r
    train_r, val_r, test_r = train_r / total_r, val_r / total_r, test_r / total_r

    rng = random.Random(seed)
    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    has_cat = False
    for row in rows:
        cat = category_of(row)
        if cat is not None:
            has_cat = True
            by_cat[cat].append(row)
        else:
            by_cat["_all_"].append(row)

    if not has_cat:
        by_cat = {"_all_": list(rows)}

    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []

    for cat, items in by_cat.items():
        items = list(items)
        rng.shuffle(items)
        n = len(items)
        if n == 0:
            continue
        n_train = int(n * train_r)
        n_val = int(n * val_r)
        # ensure remainder goes to test; fix tiny sets
        if n >= 3:
            n_train = max(1, n_train)
            n_val = max(1, n_val) if n - n_train >= 2 else 0
            n_test = n - n_train - n_val
            if n_test < 1 and n - n_train >= 2:
                n_val = max(1, n_val - 1) if n_val > 0 else 0
                n_test = n - n_train - n_val
            if n_test < 1:
                n_test = 1
                if n_val > 1:
                    n_val -= 1
                else:
                    n_train = max(1, n_train - 1)
                n_test = n - n_train - n_val
        elif n == 2:
            n_train, n_val, n_test = 1, 0, 1
        else:
            n_train, n_val, n_test = 1, 0, 0

        train.extend(items[:n_train])
        val.extend(items[n_train : n_train + n_val])
        test.extend(items[n_train + n_val :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split cleaned datasets 8:1:1")
    p.add_argument("--sft-input", type=str, default="data/processed/sft_clean.jsonl")
    p.add_argument("--pref-input", type=str, default="data/processed/preference_clean.jsonl")
    p.add_argument("--output-dir", type=str, default="data/splits")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.1)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = _project_root()

    def resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else (root / path).resolve()

    out_dir = resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    summary: dict[str, Any] = {}

    for kind, in_path, suffix in [
        ("sft", resolve(args.sft_input), "sft"),
        ("pref", resolve(args.pref_input), "pref"),
    ]:
        rows = read_jsonl(in_path)
        train, val, test = stratified_split(rows, ratios, seed=args.seed + (0 if kind == "sft" else 1))
        paths = {
            "train": out_dir / f"train.{suffix}.jsonl",
            "val": out_dir / f"val.{suffix}.jsonl",
            "test": out_dir / f"test.{suffix}.jsonl",
        }
        write_jsonl(paths["train"], train)
        write_jsonl(paths["val"], val)
        write_jsonl(paths["test"], test)
        summary[kind] = {
            "input": len(rows),
            "train": len(train),
            "val": len(val),
            "test": len(test),
            "paths": {k: str(v) for k, v in paths.items()},
        }
        print(
            f"[split] {kind}: {len(rows)} -> train={len(train)} val={len(val)} test={len(test)}"
        )

    summary_path = out_dir / "split_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[split] summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
