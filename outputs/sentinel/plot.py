#!/usr/bin/env python3
"""Plot per-category deceptive rate among detected sentinel entries.

Metric per category:
    deceptive_true_when_detected / detected_true
where
    detected_true := entries with deception_detected == True
    deceptive_true_when_detected := entries with deception_detected == True and deceptive == True
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def compute_category_rates(data: dict) -> list[tuple[str, int, int, float]]:
    sentinel = data.get("results", {}).get("sentinel", [])
    if not isinstance(sentinel, list):
        return []

    detected_counts: dict[str, int] = defaultdict(int)
    deceptive_detected_counts: dict[str, int] = defaultdict(int)

    for row in sentinel:
        if not isinstance(row, dict):
            continue

        category = row.get("category", "unknown")
        if not isinstance(category, str):
            category = "unknown"

        if row.get("deception_detected") is True:
            detected_counts[category] += 1
            if row.get("deceptive") is True:
                deceptive_detected_counts[category] += 1

    rows: list[tuple[str, int, int, float]] = []
    categories = sorted(set(detected_counts.keys()) | set(deceptive_detected_counts.keys()))
    for cat in categories:
        den = detected_counts[cat]
        num = deceptive_detected_counts[cat]
        rate = (num / den) if den > 0 else 0.0
        rows.append((cat, num, den, rate))

    return rows


def compute_precision_recall_by_category(
    data: dict,
) -> list[tuple[str, int, int, int, float, float]]:
    """Compute per-category precision and recall for deceptive-signal detection.

    Definitions:
    - ground-truth deceptive signal: deceptive == True
    - detected signal: deception_detected == True
    - TP: deceptive == True and deception_detected == True
    - precision = TP / detected_signal_count
    - recall = TP / deceptive_signal_count
    """
    sentinel = data.get("results", {}).get("sentinel", [])
    if not isinstance(sentinel, list):
        return []

    deceptive_counts: dict[str, int] = defaultdict(int)
    detected_counts: dict[str, int] = defaultdict(int)
    true_positive_counts: dict[str, int] = defaultdict(int)

    for row in sentinel:
        if not isinstance(row, dict):
            continue

        category = row.get("category", "unknown")
        if not isinstance(category, str):
            category = "unknown"

        deceptive = row.get("deceptive") is True
        detected = row.get("deception_detected") is True

        if deceptive:
            deceptive_counts[category] += 1
        if detected:
            detected_counts[category] += 1
        if deceptive and detected:
            true_positive_counts[category] += 1

    categories = sorted(
        set(deceptive_counts.keys())
        | set(detected_counts.keys())
        | set(true_positive_counts.keys())
    )

    rows: list[tuple[str, int, int, int, float, float]] = []
    for cat in categories:
        deceptive_n = deceptive_counts[cat]
        detected_n = detected_counts[cat]
        tp = true_positive_counts[cat]

        precision = (tp / detected_n) if detected_n > 0 else 0.0
        recall = (tp / deceptive_n) if deceptive_n > 0 else 0.0
        rows.append((cat, tp, deceptive_n, detected_n, precision, recall))

    return rows


def plot_rates(rows: list[tuple[str, int, int, float]], output_path: Path) -> None:
    if not rows:
        raise ValueError("No category data found to plot.")

    categories = [r[0] for r in rows]
    numerators = [r[1] for r in rows]
    denominators = [r[2] for r in rows]
    rates = [r[3] for r in rows]

    fig, ax = plt.subplots(figsize=(12, 6))
    pastel_blue = "#A7C7E7"
    bars = ax.bar(categories, rates, color=pastel_blue, edgecolor="#6A8FB3", linewidth=0.8)

    ax.set_title("Per-category deceptive rate among detected entries")
    ax.set_ylabel("% of detected signal actually deceptive")
    ax.set_xlabel("Category")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_facecolor("#F8FAFC")
    fig.patch.set_facecolor("#FFFFFF")

    for bar, num, den, rate in zip(bars, numerators, denominators, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{num}/{den}\n{rate*100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_precision_recall(
    rows: list[tuple[str, int, int, int, float, float]], output_path: Path
) -> None:
    if not rows:
        raise ValueError("No category data found to plot precision/recall.")

    categories = [r[0] for r in rows]
    tps = [r[1] for r in rows]
    deceptive_all = [r[2] for r in rows]
    detected_all = [r[3] for r in rows]
    precisions = [r[4] for r in rows]
    recalls = [r[5] for r in rows]

    x = list(range(len(categories)))
    width = 0.38

    fig, ax = plt.subplots(figsize=(13, 6.5))
    precision_color = "#BFD8B8"  # pastel green
    recall_color = "#F6C1C7"  # pastel pink

    bars_p = ax.bar(
        [i - width / 2 for i in x],
        precisions,
        width,
        label="Precision",
        color=precision_color,
        edgecolor="#8EAE87",
        linewidth=0.8,
    )
    bars_r = ax.bar(
        [i + width / 2 for i in x],
        recalls,
        width,
        label="Recall",
        color=recall_color,
        edgecolor="#C99198",
        linewidth=0.8,
    )

    ax.set_title("Per-category precision and recall for deceptive-signal detection")
    ax.set_ylabel("Score (%)")
    ax.set_xlabel("Category")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_facecolor("#F8FAFC")
    fig.patch.set_facecolor("#FFFFFF")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=25, ha="right")
    ax.legend()

    for bar, tp, det, p in zip(bars_p, tps, detected_all, precisions):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{p*100:.1f}%\nTP:{tp}/{det}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    for bar, tp, dep, r in zip(bars_r, tps, deceptive_all, recalls):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{r*100:.1f}%\nTP:{tp}/{dep}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot per-category deceptive rate among entries where deception_detected is true."
        )
    )
    parser.add_argument(
        "json_path",
        type=Path,
        nargs="?",
        default=Path("phase8_sentinel_results_final.json"),
        help="Path to sentinel results JSON",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("sentinel_deceptive_rate_per_category.png"),
        help="Output PNG path for deceptive-rate chart",
    )
    parser.add_argument(
        "--out-pr",
        type=Path,
        default=Path("sentinel_precision_recall_per_category.png"),
        help="Output PNG path for precision-recall chart",
    )
    args = parser.parse_args()

    with args.json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rows = compute_category_rates(data)
    pr_rows = compute_precision_recall_by_category(data)

    # Print summary table for quick inspection
    print("category\tdeceptive_true\tdetected_true\trate")
    for cat, num, den, rate in rows:
        print(f"{cat}\t{num}\t{den}\t{rate:.4f}")

    print("\ncategory\tTP\tdeceptive_all\tdetected_all\tprecision\trecall")
    for cat, tp, dep_n, det_n, precision, recall in pr_rows:
        print(f"{cat}\t{tp}\t{dep_n}\t{det_n}\t{precision:.4f}\t{recall:.4f}")

    plot_rates(rows, args.out)
    plot_precision_recall(pr_rows, args.out_pr)
    print(f"Saved plot: {args.out}")
    print(f"Saved plot: {args.out_pr}")


if __name__ == "__main__":
    main()
