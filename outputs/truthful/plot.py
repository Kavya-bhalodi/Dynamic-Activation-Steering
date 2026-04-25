import json
from pathlib import Path

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "truthfulqa_real_results.json"


def load_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def compute_split_metrics(entries):
    total = len(entries)
    if total == 0:
        return 0.0, 0.0, 0.0

    scores = [float(item.get("truthful_score", 0.0)) for item in entries]
    honesty_score = sum(scores) / total
    truthful_rate = sum(1 for score in scores if score > 0) / total
    lying_rate = sum(1 for score in scores if score < 0) / total

    return honesty_score, truthful_rate, lying_rate


def save_bar_chart(labels, values, title, ylabel, output_path: Path, colors):
    plt.figure(figsize=(7, 5))
    bars = plt.bar(labels, values, color=colors, edgecolor="black", linewidth=0.8)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.35)

    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.015,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    data = load_summary(INPUT_FILE)
    splits = data["truthfulqa"]

    labels = ["baseline", "steered", "gated"]
    colors = ["#4C9BE8", "#FFA640", "#66BB6A"]


    metrics = {label: compute_split_metrics(splits[label]) for label in labels}
    honesty_score = [metrics[label][0] for label in labels]
    truthful_rate = [metrics[label][1] for label in labels]
    lying_rate = [metrics[label][2] for label in labels]

    save_bar_chart(
        labels=labels,
        values=honesty_score,
        title="TruthfulQA: Honesty Score (Average Truthful Score)",
        ylabel="Honesty score",
        output_path=BASE_DIR / "23_truthfulqa_honesty_score_bar.png",
        colors=colors,
    )

    save_bar_chart(
        labels=labels,
        values=truthful_rate,
        title="TruthfulQA: Truthful Rate",
        ylabel="Truthful rate",
        output_path=BASE_DIR / "24_truthfulqa_truthful_rate_bar.png",
        colors=colors,
    )

    save_bar_chart(
        labels=labels,
        values=lying_rate,
        title="TruthfulQA: Lying Rate",
        ylabel="Lying rate",
        output_path=BASE_DIR / "25_truthfulqa_lying_rate_bar.png",
        colors=colors,
    )


if __name__ == "__main__":
    main()