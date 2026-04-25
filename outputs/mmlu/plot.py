import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# Config
# -----------------------------
json_path = Path("/home/shailshah/sem-6/RSAI/phase4/mmlu/mmlu_results.json")
output_path = Path("/home/shailshah/sem-6/RSAI/phase4/mmlu/mmlu_plots.png")

top_n_classes = None  # set e.g. 25 if needed

sections = ["baseline", "steered", "gated"]
colors = {
    "baseline": "#4C78A8",
    "steered":  "#F58518",
    "gated":    "#54A24B",
}

# -----------------------------
# Load data
# -----------------------------
with open(json_path, "r") as f:
    data = json.load(f)

results = data["results"]

# Get common classes
all_classes = set(results[sections[0]].keys())
for s in sections[1:]:
    all_classes &= set(results[s].keys())

classes = sorted(all_classes)

# -----------------------------
# Helpers
# -----------------------------
def get_accuracy(section_data, cls):
    cls_obj = section_data.get(cls, {})
    metrics = cls_obj.get("metrics", {})

    acc = metrics.get("accuracy", None)
    if acc is not None:
        return float(acc)

    correct = metrics.get("correct", None)
    total = metrics.get("total", None)
    if correct is not None and total:
        return float(correct) / float(total)

    return np.nan


def get_correct_total(section_data, cls):
    metrics = section_data.get(cls, {}).get("metrics", {})
    c = metrics.get("correct", 0)
    t = metrics.get("total", 0)
    return float(c), float(t)


# -----------------------------
# Build arrays
# -----------------------------
acc = {
    s: np.array([get_accuracy(results[s], cls) for cls in classes], dtype=float)
    for s in sections
}

# Optional filtering
if top_n_classes is not None and top_n_classes < len(classes):
    idx_sorted = np.argsort(-np.nan_to_num(acc["baseline"], nan=-1))
    idx_keep = np.sort(idx_sorted[:top_n_classes])

    classes = [classes[i] for i in idx_keep]
    for s in sections:
        acc[s] = acc[s][idx_keep]

# -----------------------------
# Overall accuracy
# -----------------------------
overall = {}
for s in sections:
    total_correct = 0.0
    total_count = 0.0

    for cls in classes:
        c, t = get_correct_total(results[s], cls)
        total_correct += c
        total_count += t

    overall[s] = (total_correct / total_count) if total_count > 0 else np.nan

# -----------------------------
# Delta
# -----------------------------
delta_steered = acc["steered"] - acc["baseline"]
delta_gated = acc["gated"] - acc["baseline"]

# -----------------------------
# Plot
# -----------------------------
fig, axes = plt.subplots(
    1, 3,
    figsize=(28, max(10, 0.3 * len(classes))),
    constrained_layout=True  # ⭐ FIXED layout
)

fig.suptitle("MMLU Metrics Comparison (Baseline vs Steered vs Gated)", fontsize=16)

# --- Plot 1: Overall accuracy
ax0 = axes[0]
x0 = np.arange(len(sections))
y0 = [overall[s] for s in sections]

bars = ax0.bar(x0, y0, color=[colors[s] for s in sections], edgecolor="black", alpha=0.9)

ax0.set_xticks(x0)
ax0.set_xticklabels([s.capitalize() for s in sections])
ax0.set_ylabel("Accuracy")
ax0.set_title("Overall Accuracy")
ax0.set_ylim(0, max(1.0, np.nanmax(y0) + 0.05))
ax0.grid(axis="y", alpha=0.25)

for b, v in zip(bars, y0):
    ax0.text(
        b.get_x() + b.get_width() / 2,
        b.get_height() + 0.01,
        f"{v:.3f}",
        ha="center",
        va="bottom",
        fontsize=10
    )

# --- Plot 2: Per-class accuracy
ax1 = axes[1]
y = np.arange(len(classes))
bar_h = 0.25

ax1.barh(y + bar_h, acc["baseline"], height=bar_h, color=colors["baseline"], label="Baseline", alpha=0.9)
ax1.barh(y,         acc["steered"],  height=bar_h, color=colors["steered"],  label="Steered",  alpha=0.9)
ax1.barh(y - bar_h, acc["gated"],    height=bar_h, color=colors["gated"],    label="Gated",    alpha=0.9)

ax1.set_yticks(y)
ax1.set_yticklabels(classes, fontsize=7)  # ⭐ smaller text helps
ax1.set_xlabel("Accuracy")
ax1.set_title("Per-Class Accuracy")
ax1.set_xlim(0, 1.0)
ax1.grid(axis="x", alpha=0.25)
ax1.legend(loc="lower right")

# --- Plot 3: Delta
ax2 = axes[2]
d_h = 0.35

ax2.barh(y + d_h/2, delta_steered, height=d_h, color=colors["steered"], alpha=0.9, label="Steered - Baseline")
ax2.barh(y - d_h/2, delta_gated,   height=d_h, color=colors["gated"],   alpha=0.9, label="Gated - Baseline")

ax2.axvline(0, color="black", linewidth=1)

ax2.set_yticks(y)
ax2.set_yticklabels(classes, fontsize=7)
ax2.set_xlabel("Delta Accuracy")
ax2.set_title("Delta vs Baseline (Per Class)")
ax2.grid(axis="x", alpha=0.25)
ax2.legend(loc="lower right")

# -----------------------------
# Save
# -----------------------------
plt.savefig(output_path, dpi=220, bbox_inches="tight")
plt.show()

print(f"Saved plot to: {output_path}")