#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# =========================
# Config
# =========================
JSON_PATH = Path("/home/shailshah/sem-6/RSAI/phase4/behonest/behonest_results.json")
OUT_PATH = Path("/home/shailshah/sem-6/RSAI/phase4/behonest/behonest_multi_plots.png")

SECTIONS = ["baseline", "steered", "gated"]
SECTION_LABELS = {
    "baseline": "Baseline",
    "steered": "Steered",
    "gated": "Gated",
}
SECTION_COLORS = {
    "baseline": "#4C9BE8",
    "steered": "#FFA640",
    "gated": "#66BB6A",
}

# Optional pretty labels
DIMENSION_LABELS = {
    "self_knowledge": "Self-Knowledge",
    "non_deceptiveness": "Non-Deceptiveness",
    "consistency": "Consistency",
}
SCENARIO_LABELS = {
    "expressing_unknowns": "Expressing\nUnknowns",
    "admitting_knowns": "Admitting\nKnowns",
    "persona_sycophancy": "Persona\nSycophancy",
    "preference_sycophancy": "Preference\nSycophancy",
    "burglar_deception": "Burglar\nDeception",
    "game_deception": "Game\nDeception",
    "prompt_format_consistency": "Prompt\nFormat\nConsistency",
    "mc_consistency": "MC\nConsistency",
    "open_form_consistency": "Open\nForm\nConsistency",
}

# =========================
# Helpers
# =========================
def safe_get(d, *keys, default=0.0):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def pretty_label(key, mapping):
    return mapping.get(key, key.replace("_", " ").title())

# =========================
# Load data
# =========================
with JSON_PATH.open("r", encoding="utf-8") as f:
    data = json.load(f)

metrics = data.get("metrics", {})

# Keep only available sections
sections = [s for s in SECTIONS if s in metrics]
if not sections:
    raise ValueError("No expected sections found in metrics. Expected one of: baseline, steered, gated")

# Collect union of dimensions and scenarios across sections
all_dims = []
all_scenarios = []

for s in sections:
    dims = safe_get(metrics, s, "dimensions", default={})
    scs = safe_get(metrics, s, "scenarios", default={})
    if isinstance(dims, dict):
        for k in dims.keys():
            if k not in all_dims:
                all_dims.append(k)
    if isinstance(scs, dict):
        for k in scs.keys():
            if k not in all_scenarios:
                all_scenarios.append(k)

# Put common keys first if present
preferred_dim_order = ["self_knowledge", "non_deceptiveness", "consistency"]
all_dims = [d for d in preferred_dim_order if d in all_dims] + [d for d in all_dims if d not in preferred_dim_order]

preferred_scenario_order = [
    "admitting_knowns",
    "expressing_unknowns",
    "game_deception",
    "burglar_deception",
    "mc_consistency",
    "open_form_consistency",
    "persona_sycophancy",
    "preference_sycophancy",
    "prompt_format_consistency",
]
all_scenarios = [s for s in preferred_scenario_order if s in all_scenarios] + [s for s in all_scenarios if s not in preferred_scenario_order]

# =========================
# Figure layout
# =========================
# plt.style.use("ggplot")
fig, axes = plt.subplots(1, 3, figsize=(22, 9), constrained_layout=False)

# ---------------------------------------------------
# Plot 1: Overall + Dimensions (grouped bars, 3 models)
# ---------------------------------------------------
ax = axes[0]
x_labels = ["overall"] + ["overall_without_prompt"] + all_dims
x_pretty = ["Overall"] + ["Overall without prompt"] + [pretty_label(d, DIMENSION_LABELS) for d in all_dims]

x = np.arange(len(x_labels))
n = len(sections)
bar_w = 0.75 / max(n, 1)

for i, sec in enumerate(sections):
    vals = [safe_get(metrics, sec, "overall", default=0.0)]
    vals += [safe_get(metrics, sec, "overall_without_prompt", default=0.0)]
    vals += [safe_get(metrics, sec, "dimensions", d, default=0.0) for d in all_dims]
    offset = (i - (n - 1) / 2) * bar_w
    ax.bar(
        x + offset,
        vals,
        width=bar_w,
        label=SECTION_LABELS.get(sec, sec.title()),
        color=SECTION_COLORS.get(sec, None),
        edgecolor="black",
        linewidth=0.5,
    )

ax.set_title("BeHonest: Overall and Dimension Scores")
ax.set_ylabel("BeHonest Score")
ax.set_xticks(x)
ax.set_xticklabels(x_pretty, rotation=15)
ax.set_ylim(0, 1.15)
ax.legend(loc="upper left")

# ---------------------------------------------------
# Plot 2: Per-Scenario Scores (grouped horizontal bars)
# ---------------------------------------------------
ax = axes[1]
y = np.arange(len(all_scenarios))
h = 0.75 / max(n, 1)

for i, sec in enumerate(sections):
    vals = [safe_get(metrics, sec, "scenarios", sc, default=0.0) for sc in all_scenarios]
    offset = (i - (n - 1) / 2) * h
    ax.barh(
        y + offset,
        vals,
        height=h,
        label=SECTION_LABELS.get(sec, sec.title()),
        color=SECTION_COLORS.get(sec, None),
        edgecolor="black",
        linewidth=0.4,
    )

ax.set_title("Per-Scenario Scores")
ax.set_xlabel("Score")
ax.set_yticks(y)
ax.set_yticklabels([pretty_label(sc, SCENARIO_LABELS) for sc in all_scenarios])
ax.set_xlim(0, 1.1)
ax.legend(loc="lower right")

# ---------------------------------------------------
# Plot 3: Delta vs Baseline (Steered and Gated vs Baseline)
# ---------------------------------------------------
ax = axes[2]
baseline_vals = [safe_get(metrics, "baseline", "scenarios", sc, default=0.0) for sc in all_scenarios]

comparators = [s for s in sections if s != "baseline"]
if comparators:
    m = len(comparators)
    hh = 0.7 / m
    delta_palette = {
        "steered": {"pos": "#1E88E5", "neg": "#64B5F6"},
        "gated": {"pos": "#43A047", "neg": "#A5D6A7"},
    }
    for j, comp in enumerate(comparators):
        comp_vals = [safe_get(metrics, comp, "scenarios", sc, default=0.0) for sc in all_scenarios]
        deltas = np.array(comp_vals) - np.array(baseline_vals)
        offs = (j - (m - 1) / 2) * hh

        pal = delta_palette.get(comp, {"pos": "#7E57C2", "neg": "#B39DDB"})
        colors = [pal["pos"] if d >= 0 else pal["neg"] for d in deltas]
        ax.barh(
            y + offs,
            deltas,
            height=hh,
            color=colors,
            edgecolor="black",
            linewidth=0.35,
            label=f"{SECTION_LABELS.get(comp, comp.title())} - Baseline",
            alpha=0.9,
        )

ax.axvline(0, color="black", linewidth=1)
ax.set_title("Delta vs Baseline (Per Scenario)")
ax.set_xlabel("Delta score")
ax.set_yticks(y)
ax.set_yticklabels([pretty_label(sc, SCENARIO_LABELS) for sc in all_scenarios])
ax.legend(loc="lower right")

# Save + show
fig.suptitle("BeHonest Metrics Comparison (Baseline vs Steered vs Gated)", fontsize=15, y=0.995)
fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
fig.savefig(OUT_PATH, dpi=180, bbox_inches="tight", pad_inches=0.2)
plt.show()

print(f"Saved plot to: {OUT_PATH}")