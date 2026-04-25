#!/usr/bin/env python3
"""Plot merged dynamic gating sweep results.

Reads the merged JSON (default: dynamic_gating_sweep_merged.json) and produces:
1) Honesty vs sharpness (gated vs steered)
2) Delta honesty vs sharpness
3) Response length vs sharpness (gated vs steered)
4) Accuracy-length tradeoff (delta honesty vs delta length)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    default_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Plot merged dynamic gating sweep metrics.")
    parser.add_argument(
        "--input",
        type=Path,
        default=default_dir / "dynamic_gating_sweep_merged.json",
        help="Path to merged JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_dir / "dynamic_gating_sweep_merged_plots.png",
        help="Output PNG path.",
    )
    return parser.parse_args()


def _extract(summary: list[dict], key: str) -> list[float]:
    out: list[float] = []
    for row in summary:
        val = row.get(key)
        out.append(float(val) if val is not None else float("nan"))
    return out


def plot_merged_results(data: dict, output_path: Path) -> None:
    summary = sorted(data.get("summary", []), key=lambda r: float(r["sharpness"]))
    if not summary:
        raise ValueError("No summary data found in input JSON.")

    sharpness = _extract(summary, "sharpness")
    gated_h = _extract(summary, "gated_honesty")
    steered_h = _extract(summary, "steered_honesty")
    delta_h = _extract(summary, "delta_honesty")
    gated_len = _extract(summary, "gated_length")
    steered_len = _extract(summary, "steered_length")
    delta_len = [g - s for g, s in zip(gated_len, steered_len)]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    ax1, ax2 = axes[0]
    ax3, ax4 = axes[1]

    # 1) Honesty vs sharpness
    ax1.plot(sharpness, steered_h, marker="o", linewidth=2, label="Steered honesty", color="#4c78a8")
    ax1.plot(sharpness, gated_h, marker="o", linewidth=2, label="Gated honesty", color="#f58518")
    ax1.set_title("Honesty vs Gate Sharpness")
    ax1.set_xlabel("Gate sharpness")
    ax1.set_ylabel("Honesty score")
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend()

    # 2) Delta honesty
    colors = ["#54a24b" if d >= 0 else "#e45756" for d in delta_h]
    ax2.bar(sharpness, delta_h, color=colors, width=1.2)
    ax2.axhline(0, color="gray", linewidth=1)
    ax2.set_title("Δ Honesty (Gated - Steered)")
    ax2.set_xlabel("Gate sharpness")
    ax2.set_ylabel("Delta honesty")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.3)

    # 3) Length vs sharpness
    ax3.plot(sharpness, steered_len, marker="o", linewidth=2, label="Steered length", color="#4c78a8")
    ax3.plot(sharpness, gated_len, marker="o", linewidth=2, label="Gated length", color="#f58518")
    ax3.set_title("Response Length vs Gate Sharpness")
    ax3.set_xlabel("Gate sharpness")
    ax3.set_ylabel("Avg response length")
    ax3.grid(True, linestyle="--", alpha=0.3)
    ax3.legend()

    # 4) Tradeoff: delta length vs delta honesty
    ax4.scatter(delta_len, delta_h, s=70, color="#72b7b2")
    for s, dl, dh in zip(sharpness, delta_len, delta_h):
        ax4.annotate(f"k={s:g}", (dl, dh), textcoords="offset points", xytext=(6, 5), fontsize=9)
    ax4.axhline(0, color="gray", linewidth=1)
    ax4.axvline(0, color="gray", linewidth=1)
    ax4.set_title("Tradeoff: ΔLength vs ΔHonesty")
    ax4.set_xlabel("Delta length (gated - steered)")
    ax4.set_ylabel("Delta honesty (gated - steered)")
    ax4.grid(True, linestyle="--", alpha=0.3)

    meta = data.get("metadata", {})
    alpha_base = meta.get("alpha_base", "?")
    sigma = meta.get("sigma", "?")
    fig.suptitle(f"Dynamic Gating Sweep (alpha_base={alpha_base}, sigma={sigma})", fontsize=14)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    with args.input.open("r", encoding="utf-8") as f:
        data = json.load(f)

    plot_merged_results(data, args.output)
    print(f"Saved plot: {args.output}")


if __name__ == "__main__":
    main()
