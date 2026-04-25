#!/usr/bin/env python3
"""Create sample-style 3-panel plots for mask three-modes metrics.

Layout (like the example image):
1) Overall metric by section
2) Per-category metric comparison (Baseline vs Steered vs Dynamic)
3) Delta vs Baseline per category
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

SECTIONS = ["baseline", "steered", "gated"]
SECTION_LABELS = {
    "baseline": "Baseline",
    "steered": "Steered",
    "gated": "Dynamic",
}
SECTION_COLORS = {
    "baseline": "#4c78a8",  # blue
    "steered": "#f58518",   # orange
    "gated": "#54a24b",   # green
}

METRICS: Dict[str, Tuple[str, str]] = {
    "honesty": ("Honesty", "is_honest"),
    "lie": ("Lie", "is_lie"),
    "evade": ("Evade", "is_evade"),
}


def safe_rate(records: List[Dict[str, Any]], field: str) -> float:
    if not records:
        return 0.0
    return sum(1 for r in records if r.get(field) == 1) / len(records)


def all_categories(data: Dict[str, Any]) -> List[str]:
    cats = set()
    for section in SECTIONS:
        block = data.get(section, {})
        if isinstance(block, dict):
            cats.update(block.keys())
    return sorted(cats)


def section_records(data: Dict[str, Any], section: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for cat_rows in data.get(section, {}).values():
        if isinstance(cat_rows, list):
            rows.extend(cat_rows)
    return rows


def compute_metric_views(
    data: Dict[str, Any],
    metric_field: str,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]], List[str]]:
    categories = all_categories(data)

    overall = {
        section: safe_rate(section_records(data, section), metric_field)
        for section in SECTIONS
    }

    per_category: Dict[str, Dict[str, float]] = {section: {} for section in SECTIONS}
    for section in SECTIONS:
        section_block = data.get(section, {})
        for cat in categories:
            rows = section_block.get(cat, []) if isinstance(section_block, dict) else []
            if not isinstance(rows, list):
                rows = []
            per_category[section][cat] = safe_rate(rows, metric_field)

    return overall, per_category, categories


def plot_metric_like_example(
    data: Dict[str, Any],
    metric_key: str,
    output_path: Path,
) -> None:
    metric_label, metric_field = METRICS[metric_key]
    overall, per_category, categories = compute_metric_views(data, metric_field)

    fig, (ax0, ax1, ax2) = plt.subplots(
        1,
        3,
        figsize=(20, 9),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.0, 1.5, 1.1]},
    )

    # --- Panel 1: Overall ---
    x = np.arange(len(SECTIONS))
    vals = [overall[s] for s in SECTIONS]
    bars = ax0.bar(x, vals, color=[SECTION_COLORS[s] for s in SECTIONS], alpha=0.9)
    ax0.set_xticks(x)
    ax0.set_xticklabels([SECTION_LABELS[s] for s in SECTIONS])
    ax0.set_ylim(0, 1.0)
    ax0.set_ylabel(f"{metric_label} Rate")
    ax0.set_title(f"Overall {metric_label} Rate")
    ax0.grid(axis="y", linestyle="--", alpha=0.25)

    for b, v in zip(bars, vals):
        ax0.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    # --- Panel 2: Per-category grouped horizontal bars ---
    y = np.arange(len(categories))
    h = 0.24

    for i, section in enumerate(SECTIONS):
        section_vals = [per_category[section][cat] for cat in categories]
        ax1.barh(
            y + (i - 1) * h,
            section_vals,
            height=h,
            label=SECTION_LABELS[section],
            color=SECTION_COLORS[section],
            alpha=0.9,
        )

    ax1.set_yticks(y)
    ax1.set_yticklabels(categories, fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 1.0)
    ax1.set_xlabel(f"{metric_label} Rate")
    ax1.set_title(f"{metric_label} Comparison by Category")
    ax1.grid(axis="x", linestyle="--", alpha=0.25)
    ax1.legend(loc="lower right", fontsize=9)

    # --- Panel 3: Delta vs Baseline ---
    delta_steered = [per_category["steered"][cat] - per_category["baseline"][cat] for cat in categories]
    delta_dynamic = [per_category["gated"][cat] - per_category["baseline"][cat] for cat in categories]

    ax2.barh(y - h / 2, delta_steered, height=h, color=SECTION_COLORS["steered"], alpha=0.9, label="Steered - Baseline")
    ax2.barh(y + h / 2, delta_dynamic, height=h, color=SECTION_COLORS["gated"], alpha=0.9, label="Gated - Baseline")

    max_abs = max(0.05, np.max(np.abs(np.array(delta_steered + delta_dynamic))))
    lim = min(1.0, max_abs * 1.25)
    ax2.set_xlim(-lim, lim)
    ax2.axvline(0, color="gray", linewidth=1)
    ax2.set_yticks(y)
    ax2.set_yticklabels(categories, fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel(f"Δ {metric_label} Rate")
    ax2.set_title(f"Delta vs Baseline ({metric_label}, Per Category)")
    ax2.grid(axis="x", linestyle="--", alpha=0.25)
    ax2.legend(loc="lower right", fontsize=9)

    fig.suptitle(f"Mask Three-Modes: {metric_label} (Baseline vs Steered vs Dynamic)", fontsize=14)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate sample-style mask metric plots.")
    default_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--input",
        type=Path,
        default=default_dir / "mask_results_updated.json",
        help="Merged JSON input file",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=default_dir / "mask_three_modes_like_example",
        help="Output prefix for generated PNG files",
    )
    parser.add_argument(
        "--metric",
        choices=["honesty", "lie", "evade", "all"],
        default="all",
        help="Metric to plot",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with args.input.open("r", encoding="utf-8") as f:
        data = json.load(f)

    metrics_to_plot = [args.metric] if args.metric != "all" else ["honesty", "lie", "evade"]

    for metric in metrics_to_plot:
        out_path = Path(f"{args.output_prefix}_{metric}.png")
        plot_metric_like_example(data, metric, out_path)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
