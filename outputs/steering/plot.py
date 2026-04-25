import json
import re
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt


def parse_alpha_from_filename(filename: str):
    """
    Extract alpha from filenames like:
    results_alpha_0_chaitu.json
    results_alpha_0-1_name.json  (supports '-' as decimal separator)
    """
    m = re.search(r"results_alpha_([^_]+)_", filename)
    if not m:
        return None
    raw = m.group(1).replace("-", ".")
    try:
        return float(raw)
    except ValueError:
        return None


def extract_name_from_filename(filename: str):
    """
    Extract name from filename:
    results_alpha_0_chaitu.json -> chaitu
    """
    stem = Path(filename).stem  # removes .json
    return stem.split("_")[-1]  # last part


def collect_run_scores(folder="activations"):
    folder_path = Path(folder)
    files = sorted(folder_path.glob("results_alpha_*_*.json"))

    all_points = []  # list of tuples: (alpha, score, file, name)

    for f in files:
        with f.open("r", encoding="utf-8") as fp:
            data = json.load(fp)

        # Prefer alpha inside file, else parse from filename
        alpha = data.get("alpha")
        if alpha is None:
            alpha = parse_alpha_from_filename(f.name)
        if alpha is None:
            print(f"Skipping {f.name}: could not determine alpha")
            continue
        alpha = float(alpha)

        name = extract_name_from_filename(f.name)

        results = data.get("results", {})
        runs = results.get("runs", {})

        if isinstance(runs, dict) and runs:
            for _, run_info in runs.items():
                score = run_info.get("avg_honesty_score")
                if score is not None:
                    all_points.append((alpha, float(score), f.name, name))
        else:
            # fallback
            score = results.get("avg_honesty_score")
            if score is not None:
                all_points.append((alpha, float(score), f.name, name))

    return all_points


def plot_alpha_vs_honesty(all_points, out_file="alpha_vs_avg_honesty_score.png"):
    if not all_points:
        raise ValueError("No data points found.")

    grouped = defaultdict(list)
    runs_grouped = defaultdict(list)

    # Group data
    for alpha, score, file, name in all_points:
        grouped[alpha].append(score)
        runs_grouped[name].append((alpha, score))

    # Scatter points
    x_all = [p[0] for p in all_points]
    y_all = [p[1] for p in all_points]

    # Stats per alpha
    alphas_sorted = sorted(grouped.keys())
    means = [sum(grouped[a]) / len(grouped[a]) for a in alphas_sorted]
    mins = [min(grouped[a]) for a in alphas_sorted]
    maxs = [max(grouped[a]) for a in alphas_sorted]

    plt.figure(figsize=(10, 6))

    # Scatter
    plt.scatter(x_all, y_all, alpha=0.4, s=30, label="Run-level points")

    # 🔥 Plot each name as a line
    plt.plot(
        alphas_sorted,
        means,
        marker="o",
        linewidth=2.5,
        color="black",
        label="Mean"
    )
    for name in sorted(runs_grouped.keys()):
        points = runs_grouped[name]
        points_sorted = sorted(points, key=lambda x: x[0])

        x_vals = [p[0] for p in points_sorted]
        y_vals = [p[1] for p in points_sorted]

        if len(x_vals) > 1:
            plt.plot(x_vals, y_vals, linewidth=1.5, alpha=0.7, label=name)

    # Min-max band
    plt.fill_between(
        alphas_sorted,
        mins,
        maxs,
        color="tab:orange",
        alpha=0.15,
        label="Min-Max band"
    )

    # Mean line

    plt.title("Alpha vs Avg Honesty Score (All Runs)")
    plt.xlabel("Alpha")
    plt.ylabel("Avg Honesty Score")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.xticks(alphas_sorted)

    # Avoid legend clutter
    # if len(runs_grouped) <= 15:
    #     plt.legend()
    # else:
    plt.legend(["Run-level points", "Min-Max band", "Mean"])

    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    plt.show()

    print(f"Saved plot to: {out_file}")


if __name__ == "__main__":
    points = collect_run_scores("activations")
    plot_alpha_vs_honesty(points)