#!/usr/bin/env python3
"""Orchestrate domain data pipeline: generate -> clean -> analyze -> split."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path for `src.*` imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _print_table(rows: list[tuple[str, ...]], headers: tuple[str, ...]) -> None:
    cols = list(zip(*([headers] + rows))) if rows else [headers]
    widths = [max(len(str(c)) for c in col) for col in zip(*([headers] + rows))]
    fmt = " | ".join(f"{{:<{w}}}" for w in widths)

    def line(cells: tuple[str, ...]) -> str:
        return fmt.format(*[str(c) for c in cells])

    sep = "-+-".join("-" * w for w in widths)
    print(line(headers))
    print(sep)
    for r in rows:
        print(line(r))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Chinese e-commerce CS datasets")
    parser.add_argument("--config", type=str, default="configs/data.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sft-samples", type=int, default=None)
    parser.add_argument("--pref-pairs", type=int, default=None)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--skip-generate", action="store_true")
    args = parser.parse_args(argv)

    from src.utils import configure_stdio_utf8

    configure_stdio_utf8()
    t0 = time.time()
    print("=" * 60)
    print("llm-post-training-lab :: Stage data build")
    print("=" * 60)

    from src.data import generate_sft, generate_preference, clean, analyze, split

    gen_sft_argv = ["--config", args.config]
    gen_pref_argv = ["--config", args.config]
    if args.seed is not None:
        gen_sft_argv += ["--seed", str(args.seed)]
        gen_pref_argv += ["--seed", str(args.seed)]
    if args.sft_samples is not None:
        gen_sft_argv += ["--num-samples", str(args.sft_samples)]
    if args.pref_pairs is not None:
        gen_pref_argv += ["--num-samples", str(args.pref_pairs)]

    if not args.skip_generate:
        print("\n[1/4] Generating SFT ...")
        rc = generate_sft.main(gen_sft_argv)
        if rc != 0:
            return rc
        print("\n[2/4] Generating preference pairs ...")
        rc = generate_preference.main(gen_pref_argv)
        if rc != 0:
            return rc
    else:
        print("\n[1-2/4] skip generate")

    print("\n[3/4] Cleaning ...")
    rc = clean.main(["--config", args.config])
    if rc != 0:
        return rc

    print("\n[3b/4] Analyzing ...")
    analyze_argv = []
    if args.skip_plots:
        analyze_argv.append("--no-plots")
    rc = analyze.main(analyze_argv)
    if rc != 0:
        return rc

    print("\n[4/4] Splitting 8:1:1 ...")
    split_argv = []
    if args.seed is not None:
        split_argv += ["--seed", str(args.seed)]
    rc = split.main(split_argv)
    if rc != 0:
        return rc

    # Summary table from artifacts
    root = ROOT
    stats_path = root / "reports" / "data_cleaning_stats.json"
    split_path = root / "data" / "splits" / "split_summary.json"
    analysis_path = root / "reports" / "data_length_analysis.json"

    sft_in = sft_out = pref_in = pref_out = 0
    train_s = val_s = test_s = 0
    train_p = val_p = test_p = 0
    if stats_path.is_file():
        with stats_path.open("r", encoding="utf-8") as f:
            st = json.load(f)
        sft_in = st.get("sft", {}).get("input", 0)
        sft_out = st.get("sft", {}).get("output", 0)
        pref_in = st.get("preference", {}).get("input", 0)
        pref_out = st.get("preference", {}).get("output", 0)
    if split_path.is_file():
        with split_path.open("r", encoding="utf-8") as f:
            sp = json.load(f)
        train_s = sp.get("sft", {}).get("train", 0)
        val_s = sp.get("sft", {}).get("val", 0)
        test_s = sp.get("sft", {}).get("test", 0)
        train_p = sp.get("pref", {}).get("train", 0)
        val_p = sp.get("pref", {}).get("val", 0)
        test_p = sp.get("pref", {}).get("test", 0)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    _print_table(
        [
            ("SFT raw", str(sft_in), str(sft_out), f"{train_s}/{val_s}/{test_s}"),
            ("Preference raw", str(pref_in), str(pref_out), f"{train_p}/{val_p}/{test_p}"),
        ],
        ("dataset", "generated", "after_clean", "train/val/test"),
    )

    # category peek
    if analysis_path.is_file():
        with analysis_path.open("r", encoding="utf-8") as f:
            an = json.load(f)
        cats = (an.get("sft") or {}).get("category_distribution") or {}
        if cats:
            print("\nSFT categories:")
            for k, v in sorted(cats.items(), key=lambda x: -x[1]):
                print(f"  {k}: {v}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Artifacts under: {root / 'data'} and {root / 'reports'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
