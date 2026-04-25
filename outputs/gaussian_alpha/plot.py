import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# ---- config ----
json_path = "gaussian_alpha/gaussian_steering_results_disentangled.json"
ymin = 0.20
ymax = 0.50
save_path = "gaussian_alpha/honesty_vs_sigma_only.png"
cmap_name = "viridis"
# ----------------

# ---- load data ----
with open(json_path, "r") as f:
    data = json.load(f)

sweep_results = data["sweep_results"]

# ---- collect scores ----
scores = defaultdict(dict)
all_alphas = set()
all_sigmas = set()

for run_key, run_data in sweep_results.items():
    alpha = float(run_data["alpha_base"])
    sigma = int(run_data["sigma"])
    honesty = float(run_data["avg_honesty_score"])

    scores[alpha][sigma] = honesty
    all_alphas.add(alpha)
    all_sigmas.add(sigma)

alphas = sorted(all_alphas)
sigmas = sorted(all_sigmas)

# ---- print table ----
print("\nHonesty scores (avg_honesty_score):")
header = "sigma".ljust(8) + "".join([f"alpha={a:<10}".ljust(16) for a in alphas])
print(header)
print("-" * len(header))

for s in sigmas:
    row = str(s).ljust(8)
    for a in alphas:
        val = scores[a].get(s, None)
        row += (f"{val:.6f}" if val is not None else "N/A").ljust(16)
    print(row)

# ---- plotting ----
fig, ax = plt.subplots(figsize=(10, 6))

cmap = plt.get_cmap(cmap_name)
norm = mcolors.Normalize(vmin=min(alphas), vmax=max(alphas))

for a in alphas:
    y = [scores[a].get(s, None) for s in sigmas]
    color = cmap(norm(a))

    ax.plot(
        sigmas,
        y,
        marker="o",
        linewidth=2,
        markersize=7,
        color=color,
        label=f"α_base={a}"
    )

# ---- colorbar (FIXED) ----
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

fig.colorbar(sm, ax=ax, label="α_base")  # <-- FIX HERE

# ---- labels ----
ax.set_title("Honesty Score vs Gaussian Spread by Base Steering Strength")
ax.set_xlabel("Gaussian Spread (σ)")
ax.set_ylabel("Avg Honesty Score")
ax.set_xticks(sigmas)
ax.set_ylim(ymin, ymax)
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig(save_path, dpi=200)
plt.show()

print(f"\nSaved plot: {save_path}")