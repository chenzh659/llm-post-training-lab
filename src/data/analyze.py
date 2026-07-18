"""Analyze cleaned datasets: length histograms and category distribution."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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


def sft_lengths(row: dict[str, Any]) -> dict[str, int]:
    user_len = 0
    asst_len = 0
    total = 0
    for m in row.get("messages") or []:
        c = m.get("content") or ""
        total += len(c)
        role = (m.get("role") or "").lower()
        if role == "user":
            user_len += len(c)
        elif role == "assistant":
            asst_len += len(c)
    return {"user_chars": user_len, "assistant_chars": asst_len, "total_chars": total}


def pref_lengths(row: dict[str, Any]) -> dict[str, int]:
    prompt = row.get("prompt") or ""
    chosen = row.get("chosen") or ""
    rejected = row.get("rejected") or ""
    return {
        "prompt_chars": len(prompt),
        "chosen_chars": len(chosen),
        "rejected_chars": len(rejected),
        "total_chars": len(prompt) + len(chosen) + len(rejected),
    }


def histogram(values: list[int], bin_size: int = 50) -> dict[str, int]:
    """Bucket counts keyed by 'lo-hi' char ranges."""
    if not values:
        return {}
    hist: Counter[str] = Counter()
    for v in values:
        lo = (v // bin_size) * bin_size
        hi = lo + bin_size - 1
        hist[f"{lo}-{hi}"] += 1
    # sort keys by numeric lo
    ordered = dict(sorted(hist.items(), key=lambda kv: int(kv[0].split("-")[0])))
    return ordered


def summarize_numeric(values: list[int]) -> dict[str, float]:
    if not values:
        return {"count": 0, "min": 0, "max": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0}
    s = sorted(values)
    n = len(s)

    def pct(p: float) -> float:
        if n == 1:
            return float(s[0])
        idx = min(n - 1, max(0, int(round((p / 100.0) * (n - 1)))))
        return float(s[idx])

    return {
        "count": n,
        "min": float(s[0]),
        "max": float(s[-1]),
        "mean": float(sum(s) / n),
        "p50": pct(50),
        "p90": pct(90),
    }


def analyze_sft(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = [sft_lengths(r) for r in rows]
    user_c = [x["user_chars"] for x in lengths]
    asst_c = [x["assistant_chars"] for x in lengths]
    total_c = [x["total_chars"] for x in lengths]
    cats = Counter(r.get("category") or r.get("scenario") or "unknown" for r in rows)
    return {
        "n": len(rows),
        "category_distribution": dict(cats),
        "user_chars": summarize_numeric(user_c),
        "assistant_chars": summarize_numeric(asst_c),
        "total_chars": summarize_numeric(total_c),
        "histograms": {
            "user_chars": histogram(user_c, 20),
            "assistant_chars": histogram(asst_c, 50),
            "total_chars": histogram(total_c, 50),
        },
    }


def analyze_pref(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = [pref_lengths(r) for r in rows]
    prompt_c = [x["prompt_chars"] for x in lengths]
    chosen_c = [x["chosen_chars"] for x in lengths]
    rejected_c = [x["rejected_chars"] for x in lengths]
    cats = Counter(r.get("category") or r.get("scenario") or "unknown" for r in rows)
    reject_types = Counter((r.get("meta") or {}).get("reject_type") or "unknown" for r in rows)
    return {
        "n": len(rows),
        "category_distribution": dict(cats),
        "reject_type_distribution": dict(reject_types),
        "prompt_chars": summarize_numeric(prompt_c),
        "chosen_chars": summarize_numeric(chosen_c),
        "rejected_chars": summarize_numeric(rejected_c),
        "histograms": {
            "prompt_chars": histogram(prompt_c, 20),
            "chosen_chars": histogram(chosen_c, 50),
            "rejected_chars": histogram(rejected_c, 50),
        },
    }


def try_plots(report: dict[str, Any], figures_dir: Path) -> list[str]:
    """Save simple matplotlib bar charts; skip silently on failure (headless)."""
    saved: list[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[analyze] skip plots (matplotlib unavailable): {exc}")
        return saved

    figures_dir.mkdir(parents=True, exist_ok=True)

    def bar_from_hist(hist: dict[str, int], title: str, fname: str) -> None:
        if not hist:
            return
        labels = list(hist.keys())
        vals = list(hist.values())
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(range(len(labels)), vals, color="#4C78A8")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(title)
        ax.set_ylabel("count")
        fig.tight_layout()
        out = figures_dir / fname
        fig.savefig(out, dpi=120)
        plt.close(fig)
        saved.append(str(out))

    def bar_from_dist(dist: dict[str, int], title: str, fname: str) -> None:
        if not dist:
            return
        labels = list(dist.keys())
        vals = list(dist.values())
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(len(labels)), vals, color="#F58518")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_title(title)
        ax.set_ylabel("count")
        fig.tight_layout()
        out = figures_dir / fname
        fig.savefig(out, dpi=120)
        plt.close(fig)
        saved.append(str(out))

    sft = report.get("sft") or {}
    pref = report.get("preference") or {}
    try:
        bar_from_hist(
            (sft.get("histograms") or {}).get("assistant_chars") or {},
            "SFT assistant length (chars)",
            "sft_assistant_chars_hist.png",
        )
        bar_from_dist(
            sft.get("category_distribution") or {},
            "SFT category distribution",
            "sft_category_dist.png",
        )
        bar_from_hist(
            (pref.get("histograms") or {}).get("chosen_chars") or {},
            "Preference chosen length (chars)",
            "pref_chosen_chars_hist.png",
        )
        bar_from_dist(
            pref.get("category_distribution") or {},
            "Preference category distribution",
            "pref_category_dist.png",
        )
    except Exception as exc:  # pragma: no cover
        print(f"[analyze] plot failed (headless?): {exc}")
    return saved


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze cleaned CS datasets")
    p.add_argument("--sft-input", type=str, default="data/processed/sft_clean.jsonl")
    p.add_argument("--pref-input", type=str, default="data/processed/preference_clean.jsonl")
    p.add_argument("--output", type=str, default="reports/data_length_analysis.json")
    p.add_argument("--figures-dir", type=str, default="reports/figures")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = _project_root()

    def resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else (root / path).resolve()

    sft_rows = read_jsonl(resolve(args.sft_input))
    pref_rows = read_jsonl(resolve(args.pref_input))
    report = {
        "sft": analyze_sft(sft_rows),
        "preference": analyze_pref(pref_rows),
    }
    out = resolve(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    figures: list[str] = []
    if not args.no_plots:
        figures = try_plots(report, resolve(args.figures_dir))
    report["figures"] = figures

    with out.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[analyze] sft_n={report['sft']['n']} pref_n={report['preference']['n']}")
    print(f"[analyze] wrote {out}")
    if figures:
        print(f"[analyze] figures: {len(figures)} files under {resolve(args.figures_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
