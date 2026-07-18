#!/usr/bin/env python3
"""Generate README / report figures from reports/*.json artifacts.

Usage:
  python scripts/09_plot_reports.py
  python scripts/09_plot_reports.py --out-dir docs/assets

Outputs PNG (light theme) under docs/assets/ by default.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# dataviz reference categorical palette (light)
SERIES = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SEC = "#52514e"
GRID = "#e6e5e0"
STATUS_GOOD = "#008300"
STATUS_WARN = "#eda100"
STATUS_BAD = "#e34948"


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _setup_mpl():
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm

    # Prefer Chinese-capable fonts on Windows; fall back gracefully.
    preferred = [
        "Microsoft YaHei",
        "Noto Sans SC",
        "SimHei",
        "PingFang SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    family = next((n for n in preferred if n in available), "DejaVu Sans")
    mpl.rcParams.update(
        {
            "font.family": family,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_SEC,
            "axes.titlecolor": TEXT,
            "axes.facecolor": SURFACE,
            "figure.facecolor": SURFACE,
            "text.color": TEXT,
            "xtick.color": TEXT_SEC,
            "ytick.color": TEXT_SEC,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "legend.frameon": False,
            "figure.dpi": 160,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.25,
        }
    )
    return plt


def _style_ax(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", linestyle="-", alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)


def plot_pipeline(out: Path, plt) -> None:
    """Horizontal pipeline flowchart (static architecture)."""
    stages = [
        "数据构建",
        "清洗切分",
        "SFT\nLoRA/QLoRA",
        "DPO\n偏好对齐",
        "规则评测",
        "错误分析",
        "vLLM\n压测",
    ]
    fig, ax = plt.subplots(figsize=(11.5, 2.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("端到端后训练流水线 · 中文电商智能客服", pad=10, fontweight="medium")

    n = len(stages)
    box_w = 0.105
    box_h = 0.42
    y = 0.38
    gap = (1.0 - n * box_w) / (n + 1)
    xs = [gap + i * (box_w + gap) for i in range(n)]

    for i, (x, label) in enumerate(zip(xs, stages)):
        color = SERIES[i % len(SERIES)]
        rect = plt.Rectangle(
            (x, y),
            box_w,
            box_h,
            facecolor=color,
            edgecolor="none",
            linewidth=0,
            zorder=2,
            joinstyle="round",
        )
        # slight round via FancyBbox
        from matplotlib.patches import FancyBboxPatch

        rect = FancyBboxPatch(
            (x, y),
            box_w,
            box_h,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor=color,
            edgecolor="none",
            zorder=2,
        )
        ax.add_patch(rect)
        ax.text(
            x + box_w / 2,
            y + box_h / 2,
            label,
            ha="center",
            va="center",
            color="white",
            fontsize=9.5,
            fontweight="medium",
            zorder=3,
        )
        if i < n - 1:
            x0 = x + box_w + 0.005
            x1 = xs[i + 1] - 0.005
            ax.annotate(
                "",
                xy=(x1, y + box_h / 2),
                xytext=(x0, y + box_h / 2),
                arrowprops=dict(arrowstyle="->", color=TEXT_SEC, lw=1.4),
                zorder=1,
            )

    ax.text(
        0.5,
        0.12,
        "配置驱动 · 可 --demo 无 GPU 复现 · 决策看业务指标而非仅 train loss",
        ha="center",
        va="center",
        color=TEXT_SEC,
        fontsize=9,
    )
    fig.savefig(out / "pipeline.png")
    plt.close(fig)


def plot_category_dist(out: Path, plt, length: dict) -> None:
    sft = (length.get("sft") or {}).get("category_distribution") or {}
    if not sft:
        return
    # fixed category order for readability
    order = ["商品咨询", "物流查询", "退换货", "优惠活动", "投诉建议", "账户订单", "支付问题"]
    labels = [c for c in order if c in sft] + [c for c in sft if c not in order]
    values = [sft[c] for c in labels]

    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    bars = ax.bar(labels, values, color=SERIES[0], width=0.68, zorder=3, edgecolor=SURFACE, linewidth=1.5)
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.02,
            str(v),
            ha="center",
            va="bottom",
            color=TEXT,
            fontsize=9,
        )
    _style_ax(ax)
    ax.set_ylabel("样本数")
    ax.set_title(f"SFT 场景分布（n={sum(values)}，清洗后）")
    ax.tick_params(axis="x", rotation=18)
    fig.savefig(out / "category_distribution.png")
    plt.close(fig)


def plot_model_compare(out: Path, plt, comparison: dict) -> None:
    summaries = comparison.get("model_summaries") or {}
    if not summaries:
        return
    # Prefer base / sft / dpo order
    order = [k for k in ("base", "sft", "dpo") if k in summaries] + [
        k for k in summaries if k not in ("base", "sft", "dpo")
    ]
    metrics = [
        ("mean_composite", "综合分 composite"),
        ("pass_rate", "通过率 pass"),
        ("hallucination_rate", "幻觉率 ↓ 更好"),
    ]
    x = list(range(len(order)))
    width = 0.24

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for i, (key, label) in enumerate(metrics):
        vals = [float(summaries[m].get(key, 0) or 0) for m in order]
        color = SERIES[i]
        offset = (i - 1) * width
        bars = ax.bar(
            [xi + offset for xi in x],
            vals,
            width=width,
            label=label,
            color=color,
            zorder=3,
            edgecolor=SURFACE,
            linewidth=1.2,
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=TEXT,
            )

    display = []
    for m in order:
        if m == "sft":
            display.append("SFT / gold")
        elif m == "dpo":
            display.append("DPO")
        elif m == "base":
            display.append("Base")
        else:
            display.append(m)
    ax.set_xticks(x)
    ax.set_xticklabels(display)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("分数 / 比率")
    title = "Base vs SFT vs DPO 离线对比"
    if comparison.get("mock"):
        title += "（demo mock 评测）"
    ax.set_title(title)
    ax.legend(loc="upper left", ncol=3, fontsize=9)
    _style_ax(ax)
    fig.savefig(out / "model_comparison.png")
    plt.close(fig)


def plot_winrate(out: Path, plt, comparison: dict) -> None:
    pairwise = comparison.get("pairwise") or {}
    if not pairwise:
        return
    pairs = []
    win_a = []
    win_b = []
    ties = []
    for key, row in pairwise.items():
        a = row.get("a", "A")
        b = row.get("b", "B")
        pairs.append(f"{a} vs {b}")
        win_a.append(float(row.get("win_rate_a", 0)))
        win_b.append(float(row.get("win_rate_b", 0)))
        ties.append(float(row.get("tie_rate", 0)))

    y = list(range(len(pairs)))
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    ax.barh(y, win_a, color=SERIES[0], height=0.55, label="A 胜率", zorder=3, edgecolor=SURFACE, linewidth=1)
    left = win_a[:]
    ax.barh(y, win_b, left=left, color=SERIES[1], height=0.55, label="B 胜率", zorder=3, edgecolor=SURFACE, linewidth=1)
    left2 = [a + b for a, b in zip(win_a, win_b)]
    ax.barh(y, ties, left=left2, color=SERIES[2], height=0.55, label="平局", zorder=3, edgecolor=SURFACE, linewidth=1)

    for i, (a, b, t) in enumerate(zip(win_a, win_b, ties)):
        # label dominant side
        ax.text(0.02, i, f"A {a:.0%}", va="center", ha="left", color="white", fontsize=8, fontweight="medium")
        ax.text(a + b / 2 if b > 0.05 else a + 0.02, i, f"B {b:.0%}" if b > 0.05 else "", va="center", ha="center", color="white", fontsize=8)
        if t > 0.08:
            ax.text(a + b + t / 2, i, f"平 {t:.0%}", va="center", ha="center", color=TEXT, fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(pairs)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("比例")
    ax.set_title("成对规则裁判胜率（越高越好）")
    ax.legend(loc="lower right", ncol=3, fontsize=9)
    ax.invert_yaxis()
    _style_ax(ax)
    ax.grid(axis="x", linestyle="-", alpha=0.9)
    ax.grid(axis="y", visible=False)
    fig.savefig(out / "win_rates.png")
    plt.close(fig)


def plot_error_taxonomy(out: Path, plt, errors: dict) -> None:
    dist = errors.get("primary_distribution") or errors.get("label_distribution") or {}
    if not dist:
        return
    # sort by count desc
    items = []
    for k, v in dist.items():
        if isinstance(v, dict):
            items.append((k, int(v.get("count", 0))))
        else:
            items.append((k, int(v)))
    items = sorted(items, key=lambda x: x[1], reverse=True)
    labels = [i[0] for i in items]
    values = [i[1] for i in items]
    colors = [SERIES[i % len(SERIES)] for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    y = list(range(len(labels)))
    bars = ax.barh(y, values, color=colors, height=0.58, zorder=3, edgecolor=SURFACE, linewidth=1.2)
    for bar, v in zip(bars, values):
        ax.text(v + max(values + [1]) * 0.02, bar.get_y() + bar.get_height() / 2, str(v), va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("主错误类型计数")
    n = errors.get("n", "?")
    rate = errors.get("error_rate")
    title = f"错误类型分析（n={n}"
    if rate is not None:
        title += f"，error_rate={rate:.0%}"
    title += "）"
    ax.set_title(title)
    _style_ax(ax)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    fig.savefig(out / "error_taxonomy.png")
    plt.close(fig)


def plot_length_hist(out: Path, plt, length: dict) -> None:
    hist = ((length.get("sft") or {}).get("histograms") or {}).get("assistant_chars") or {}
    if not hist:
        return
    labels = list(hist.keys())
    values = [hist[k] for k in labels]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.bar(labels, values, color=SERIES[4], width=0.7, zorder=3, edgecolor=SURFACE, linewidth=1.2)
    for i, v in enumerate(values):
        ax.text(i, v + max(values) * 0.02, str(v), ha="center", fontsize=9)
    _style_ax(ax)
    ax.set_xlabel("助手回复字符数区间")
    ax.set_ylabel("样本数")
    ax.set_title("SFT 助手回复长度分布")
    fig.savefig(out / "length_histogram.png")
    plt.close(fig)


def plot_serving(out: Path, plt, bench: dict) -> None:
    if not bench:
        return
    ttft = bench.get("ttft") or {}
    e2e = bench.get("e2e_latency") or {}
    thr = bench.get("throughput") or {}

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))

    # Left: latency bars
    ax = axes[0]
    names = ["TTFT mean", "TTFT p95", "E2E mean", "E2E p95"]
    vals = [
        float(ttft.get("mean_s") or 0) * 1000,
        float(ttft.get("p95_s") or 0) * 1000,
        float(e2e.get("mean_s") or 0) * 1000,
        float(e2e.get("p95_s") or 0) * 1000,
    ]
    colors = [SERIES[0], SERIES[0], SERIES[1], SERIES[1]]
    bars = ax.bar(names, vals, color=colors, width=0.65, zorder=3, edgecolor=SURFACE, linewidth=1.2)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(vals + [1]) * 0.03, f"{v:.0f}", ha="center", fontsize=9)
    ax.set_ylabel("延迟 (ms)")
    title = "服务延迟"
    if bench.get("demo"):
        title += "（demo mock）"
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=12)
    _style_ax(ax)

    # Right: throughput KPIs as bars
    ax = axes[1]
    names2 = ["req/s", "tok/s"]
    vals2 = [
        float(thr.get("requests_per_s") or 0),
        float(thr.get("completion_tokens_per_s") or 0),
    ]
    bars = ax.bar(names2, vals2, color=[SERIES[5], SERIES[3]], width=0.5, zorder=3, edgecolor=SURFACE, linewidth=1.2)
    for bar, v in zip(bars, vals2):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(vals2 + [1]) * 0.03, f"{v:.1f}", ha="center", fontsize=9)
    ax.set_ylabel("吞吐")
    ax.set_title(f"吞吐（并发={bench.get('concurrency', '?')}）")
    _style_ax(ax)

    fig.suptitle("vLLM / Serving Bench", y=1.02, fontsize=13, color=TEXT)
    fig.tight_layout()
    fig.savefig(out / "serving_bench.png")
    plt.close(fig)


def plot_quality_vs_loss_story(out: Path, plt) -> None:
    """Illustrative curves: loss can fall while quality plateaus — teaching chart.

    Schematic only (not real GPU logs). Two panels — never dual-axis.
    """
    import numpy as np

    steps = np.arange(0, 101)
    train_loss = 2.2 * np.exp(-steps / 28) + 0.35 + 0.04 * np.sin(steps / 6)
    composite = 0.55 + 0.38 * (1 - np.exp(-steps / 22)) + 0.01 * np.sin(steps / 9)
    hallu = 0.28 * np.exp(-steps / 35) + 0.06 + 0.01 * np.cos(steps / 11)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 5.2), sharex=True, gridspec_kw={"hspace": 0.18})

    ax1.plot(steps, train_loss, color=SERIES[5], lw=2.0)
    ax1.set_ylabel("train loss")
    ax1.set_title("Loss ≠ Quality（示意）：过程量与决策量要分开看")
    ax1.axvline(55, color=GRID, lw=1.2, linestyle=":")
    ax1.text(56, float(train_loss.max()) * 0.82, "过拟合风险区", color=TEXT_SEC, fontsize=8)
    _style_ax(ax1)

    ax2.plot(steps, composite, color=SERIES[0], lw=2.0, label="composite 质量分")
    ax2.plot(steps, hallu, color=SERIES[2], lw=2.0, linestyle="--", label="幻觉率 ↓更好")
    ax2.set_xlabel("训练 step（示意，非真实日志）")
    ax2.set_ylabel("质量指标")
    ax2.set_ylim(0, 1.05)
    ax2.axvline(55, color=GRID, lw=1.2, linestyle=":")
    ax2.legend(loc="center right", fontsize=9)
    _style_ax(ax2)

    fig.savefig(out / "loss_vs_quality.png")
    plt.close(fig)


def plot_data_funnel(out: Path, plt, clean: dict, splits: dict) -> None:
    sft_in = (clean.get("sft") or {}).get("input", 2000)
    sft_out = (clean.get("sft") or {}).get("output", 1932)
    pref_in = (clean.get("preference") or {}).get("input", 800)
    pref_out = (clean.get("preference") or {}).get("output", 800)
    train = (splits.get("sft") or {}).get("train", 0)
    val = (splits.get("sft") or {}).get("val", 0)
    test = (splits.get("sft") or {}).get("test", 0)

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.8))

    # retention
    ax = axes[0]
    labels = ["SFT raw", "SFT clean", "Pref raw", "Pref clean"]
    vals = [sft_in, sft_out, pref_in, pref_out]
    colors = [SERIES[0], SERIES[1], SERIES[0], SERIES[1]]
    bars = ax.bar(labels, vals, color=colors, width=0.65, zorder=3, edgecolor=SURFACE, linewidth=1.2)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(vals) * 0.02, str(v), ha="center", fontsize=9)
    ret = (clean.get("sft") or {}).get("retention_rate")
    title = "清洗保留"
    if ret is not None:
        title += f"（SFT 保留率 {ret:.1%}）"
    ax.set_title(title)
    ax.set_ylabel("条数")
    ax.tick_params(axis="x", rotation=10)
    _style_ax(ax)

    # split
    ax = axes[1]
    if train:
        labels2 = ["train", "val", "test"]
        vals2 = [train, val, test]
        colors2 = [SERIES[0], SERIES[4], SERIES[2]]
        bars = ax.bar(labels2, vals2, color=colors2, width=0.55, zorder=3, edgecolor=SURFACE, linewidth=1.2)
        for bar, v in zip(bars, vals2):
            ax.text(bar.get_x() + bar.get_width() / 2, v + max(vals2) * 0.02, str(v), ha="center", fontsize=9)
        ax.set_title("SFT 划分 8:1:1")
        ax.set_ylabel("条数")
        _style_ax(ax)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "missing split_summary.json", ha="center")

    fig.suptitle("数据工程：生成 → 清洗 → 划分", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "data_pipeline_stats.png")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot report figures for README")
    parser.add_argument("--out-dir", type=str, default="docs/assets")
    parser.add_argument("--reports-dir", type=str, default="reports")
    args = parser.parse_args(argv)

    out = (ROOT / args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rep = (ROOT / args.reports_dir).resolve()

    try:
        plt = _setup_mpl()
    except ImportError:
        print("matplotlib is required: pip install matplotlib", file=sys.stderr)
        return 1

    length = _load(rep / "data_length_analysis.json")
    comparison = _load(rep / "comparison.json")
    errors = _load(rep / "error_analysis.json")
    bench = _load(rep / "bench_serving.json")
    clean = _load(rep / "data_cleaning_stats.json")
    splits = _load(ROOT / "data" / "splits" / "split_summary.json")

    plot_pipeline(out, plt)
    plot_data_funnel(out, plt, clean, splits)
    plot_category_dist(out, plt, length)
    plot_length_hist(out, plt, length)
    plot_model_compare(out, plt, comparison)
    plot_winrate(out, plt, comparison)
    plot_error_taxonomy(out, plt, errors)
    plot_serving(out, plt, bench)
    plot_quality_vs_loss_story(out, plt)

    written = sorted(p.name for p in out.glob("*.png"))
    print(f"Wrote {len(written)} figures -> {out}")
    for name in written:
        print(f"  - {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
