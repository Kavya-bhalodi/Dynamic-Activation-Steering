import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import os, json, gc, time, re, math
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import scipy.stats as stats
from collections import defaultdict
from utils.steering_utils import (
    BaseConfig,
    init_environment,
    load_model,
    load_steering_vectors,
    compute_gaussian_weights,
    compute_per_layer_alphas,
    generate_responses_batched,
)

init_environment()
import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

DATA_DIR = "activations_output"
OUTPUT_DIR = "activations_output"
PLOTS_DIR = "plots"


def compute_ttpd_direction(
    honest_acts: np.ndarray,
    scheming_acts: np.ndarray,
    aff_acts: np.ndarray,
    neg_acts: np.ndarray,
) -> Dict:
    N_honest = honest_acts.shape[0]
    N_scheming = scheming_acts.shape[0]
    N_aff = aff_acts.shape[0]
    N_neg = neg_acts.shape[0]
    hidden_dim = honest_acts.shape[1]
    X_data = np.vstack([honest_acts, scheming_acts, aff_acts, neg_acts])
    global_mean = X_data.mean(axis=0, keepdims=True)
    X_centered = X_data - global_mean
    n_total = X_data.shape[0]
    Z = np.zeros((n_total, 2))
    idx = 0
    Z[idx : idx + N_honest, 0] = +1.0
    idx += N_honest
    Z[idx : idx + N_scheming, 0] = -1.0
    idx += N_scheming
    Z[idx : idx + N_aff, 1] = +1.0
    idx += N_aff
    Z[idx : idx + N_neg, 1] = -1.0
    idx += N_neg
    ZtZ = Z.T @ Z
    ZtX = Z.T @ X_centered
    W = np.linalg.solve(ZtZ, ZtX)
    t_G = W[0]
    t_P = W[1]
    t_G_norm = np.linalg.norm(t_G)
    t_P_norm = np.linalg.norm(t_P)
    t_G_unit = t_G / (t_G_norm + 1e-08)
    t_P_unit = t_P / (t_P_norm + 1e-08)
    cos_GP = np.dot(t_G_unit, t_P_unit)
    proj_honest = honest_acts @ t_G_unit
    proj_scheming = scheming_acts @ t_G_unit
    threshold = (proj_honest.mean() + proj_scheming.mean()) / 2
    correct_h = (proj_honest > threshold).sum()
    correct_s = (proj_scheming <= threshold).sum()
    acc_truth = (correct_h + correct_s) / (N_honest + N_scheming)
    proj_aff = aff_acts @ t_P_unit
    proj_neg = neg_acts @ t_P_unit
    threshold_p = (proj_aff.mean() + proj_neg.mean()) / 2
    correct_a = (proj_aff > threshold_p).sum()
    correct_n = (proj_neg <= threshold_p).sum()
    acc_polarity = (correct_a + correct_n) / (N_aff + N_neg)
    proj_aff_truth = aff_acts @ t_G_unit
    proj_neg_truth = neg_acts @ t_G_unit
    neg_generalization = 1.0 - abs(proj_aff_truth.mean() - proj_neg_truth.mean()) / (
        abs(proj_honest.mean() - proj_scheming.mean()) + 1e-08
    )
    pooled_std = (
        np.sqrt((proj_honest.std() ** 2 + proj_scheming.std() ** 2) / 2) + 1e-08
    )
    cohens_d = (proj_honest.mean() - proj_scheming.mean()) / pooled_std
    X_pred = Z @ W
    ss_res = np.sum((X_centered - X_pred) ** 2)
    ss_tot = np.sum(X_centered**2)
    r_squared = 1 - ss_res / (ss_tot + 1e-08)
    return {
        "t_G": t_G,
        "t_G_unit": t_G_unit,
        "t_G_norm": float(t_G_norm),
        "t_P": t_P,
        "t_P_unit": t_P_unit,
        "t_P_norm": float(t_P_norm),
        "cos_truth_polarity": float(cos_GP),
        "cohens_d": float(cohens_d),
        "accuracy_truth": float(acc_truth),
        "accuracy_polarity": float(acc_polarity),
        "negation_generalization": float(neg_generalization),
        "r_squared": float(r_squared),
        "threshold_truth": float(threshold),
        "threshold_polarity": float(threshold_p),
        "global_mean": global_mean.flatten(),
    }


def create_visualizations(
    data_dir: str,
    ttpd_results: Dict[str, Dict],
    plots_dir: str,
    layers: List[str],
    best_layer: str,
):
    os.makedirs(plots_dir, exist_ok=True)
    with h5py.File(f"{data_dir}/activations_consolidated.h5", "r") as f:
        honest = np.array(f["honest"][best_layer])
        scheming = np.array(f["scheming"][best_layer])
    metadata = json.load(open(f"{data_dir}/metadata.json"))
    categories = [m["category"] for m in metadata]
    unique_cats = sorted(set(categories))
    cat_colors = plt.cm.Set2(np.linspace(0, 1, len(unique_cats)))
    cat_cmap = {c: cat_colors[i] for i, c in enumerate(unique_cats)}
    ttpd = ttpd_results[best_layer]
    print("  Creating PCA scatter plot...")
    all_data = np.vstack([honest, scheming])
    labels = np.array([0] * len(honest) + [1] * len(scheming))
    pca = PCA(n_components=2, random_state=42)
    pca_2d = pca.fit_transform(all_data)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    ax = axes[0]
    n = len(honest)
    ax.scatter(
        pca_2d[:n, 0],
        pca_2d[:n, 1],
        c="#2196F3",
        alpha=0.3,
        s=8,
        label="Honest",
        rasterized=True,
    )
    ax.scatter(
        pca_2d[n:, 0],
        pca_2d[n:, 1],
        c="#F44336",
        alpha=0.3,
        s=8,
        label="Scheming",
        rasterized=True,
    )
    t_G_pca = (
        pca.transform(ttpd["t_G_unit"].reshape(1, -1) * 5)[0]
        - pca.transform(np.zeros((1, all_data.shape[1])))[0]
    )
    origin = pca_2d.mean(axis=0)
    ax.annotate(
        "",
        xy=origin + t_G_pca * 2,
        xytext=origin,
        arrowprops=dict(arrowstyle="->", color="black", lw=2.5),
    )
    ax.text(
        origin[0] + t_G_pca[0] * 2.2,
        origin[1] + t_G_pca[1] * 2.2,
        "$t_G$ (TTPD)",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)", fontsize=12)
    ax.set_title(
        f"Honest vs Scheming Activations — {best_layer}\n(PCA projection, n=1496 each)",
        fontsize=13,
    )
    ax.legend(fontsize=11, markerscale=3)
    ax.grid(True, alpha=0.3)
    ax = axes[1]
    for cat in unique_cats:
        mask = np.array([c == cat for c in categories])
        ax.scatter(
            pca_2d[:n][mask, 0],
            pca_2d[:n][mask, 1],
            c=[cat_cmap[cat]],
            alpha=0.4,
            s=10,
            marker="o",
        )
        ax.scatter(
            pca_2d[n:][mask, 0],
            pca_2d[n:][mask, 1],
            c=[cat_cmap[cat]],
            alpha=0.4,
            s=10,
            marker="x",
        )
    legend_elements = [
        Patch(facecolor=cat_cmap[c], label=c.replace("_", " ").title())
        for c in unique_cats
    ]
    legend_elements.extend(
        [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="gray",
                linestyle="None",
                label="Honest",
                markersize=8,
            ),
            plt.Line2D(
                [0],
                [0],
                marker="x",
                color="gray",
                linestyle="None",
                label="Scheming",
                markersize=8,
            ),
        ]
    )
    ax.legend(handles=legend_elements, fontsize=8, loc="upper right", ncol=2)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)", fontsize=12)
    ax.set_title(f"By Category — {best_layer}\n(o=Honest, x=Scheming)", fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(
        f"{plots_dir}/01_pca_honest_vs_scheming.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
    print(f"    Saved: {plots_dir}/01_pca_honest_vs_scheming.png")
    print("  Creating projection histograms...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes[0, 0]
    proj_h = honest @ ttpd["t_G_unit"]
    proj_s = scheming @ ttpd["t_G_unit"]
    ax.hist(proj_h, bins=60, alpha=0.6, color="#2196F3", label="Honest", density=True)
    ax.hist(proj_s, bins=60, alpha=0.6, color="#F44336", label="Scheming", density=True)
    ax.axvline(
        ttpd["threshold_truth"],
        color="black",
        linestyle="--",
        lw=1.5,
        label="Threshold",
    )
    ax.set_title(
        f"Projection onto $t_G$ (TTPD)\nCohen's d = {ttpd['cohens_d']:.3f}", fontsize=12
    )
    ax.set_xlabel("Projection value", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax = axes[0, 1]
    sv = np.load(f"{data_dir}/steering_vectors.npz")
    theta_L = sv[f"{best_layer}_theta"]
    theta_L_unit = theta_L / (np.linalg.norm(theta_L) + 1e-08)
    proj_h_dim = honest @ theta_L_unit
    proj_s_dim = scheming @ theta_L_unit
    d_dim = (proj_h_dim.mean() - proj_s_dim.mean()) / (
        np.sqrt((proj_h_dim.std() ** 2 + proj_s_dim.std() ** 2) / 2) + 1e-08
    )
    ax.hist(
        proj_h_dim, bins=60, alpha=0.6, color="#2196F3", label="Honest", density=True
    )
    ax.hist(
        proj_s_dim, bins=60, alpha=0.6, color="#F44336", label="Scheming", density=True
    )
    thresh_dim = (proj_h_dim.mean() + proj_s_dim.mean()) / 2
    ax.axvline(thresh_dim, color="black", linestyle="--", lw=1.5, label="Threshold")
    ax.set_title(f"Projection onto θ_L (DIM)\nCohen's d = {d_dim:.3f}", fontsize=12)
    ax.set_xlabel("Projection value", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax = axes[1, 0]
    proj_h_p = honest @ ttpd["t_P_unit"]
    proj_s_p = scheming @ ttpd["t_P_unit"]
    d_polarity = (proj_h_p.mean() - proj_s_p.mean()) / (
        np.sqrt((proj_h_p.std() ** 2 + proj_s_p.std() ** 2) / 2) + 1e-08
    )
    ax.hist(proj_h_p, bins=60, alpha=0.6, color="#2196F3", label="Honest", density=True)
    ax.hist(
        proj_s_p, bins=60, alpha=0.6, color="#F44336", label="Scheming", density=True
    )
    ax.set_title(
        f"Projection onto $t_P$ (Polarity)\nCohen's d = {d_polarity:.3f}", fontsize=12
    )
    ax.set_xlabel("Projection value", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax = axes[1, 1]
    sv_dis = np.load(f"{data_dir}/steering_vectors_disentangled.npz")
    theta_true = sv_dis[f"{best_layer}_theta_true"]
    theta_true_unit = theta_true / (np.linalg.norm(theta_true) + 1e-08)
    proj_h_3b = honest @ theta_true_unit
    proj_s_3b = scheming @ theta_true_unit
    d_3b = (proj_h_3b.mean() - proj_s_3b.mean()) / (
        np.sqrt((proj_h_3b.std() ** 2 + proj_s_3b.std() ** 2) / 2) + 1e-08
    )
    ax.hist(
        proj_h_3b, bins=60, alpha=0.6, color="#2196F3", label="Honest", density=True
    )
    ax.hist(
        proj_s_3b, bins=60, alpha=0.6, color="#F44336", label="Scheming", density=True
    )
    thresh_3b = (proj_h_3b.mean() + proj_s_3b.mean()) / 2
    ax.axvline(thresh_3b, color="black", linestyle="--", lw=1.5, label="Threshold")
    ax.set_title(
        f"Projection onto θ_true (Phase 3b)\nCohen's d = {d_3b:.3f}", fontsize=12
    )
    ax.set_xlabel("Projection value", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.suptitle(
        f"Honest vs Scheming Separation Along Different Directions — {best_layer}",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(
        f"{plots_dir}/02_projection_histograms.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
    print(f"    Saved: {plots_dir}/02_projection_histograms.png")
    print("  Creating layer comparison chart...")
    fig, ax = plt.subplots(figsize=(12, 6))
    layer_ids = [int(ln.split("_")[1]) for ln in layers]
    d_dim_all = []
    d_3b_all = []
    d_ttpd_all = []
    for ln in layers:
        theta = sv[f"{ln}_theta"]
        tu = theta / (np.linalg.norm(theta) + 1e-08)
        with h5py.File(f"{data_dir}/activations_consolidated.h5", "r") as f:
            h = np.array(f["honest"][ln])
            s = np.array(f["scheming"][ln])
        ph, ps = (h @ tu, s @ tu)
        d_dim_all.append(
            (ph.mean() - ps.mean())
            / (np.sqrt((ph.std() ** 2 + ps.std() ** 2) / 2) + 1e-08)
        )
        tt = sv_dis[f"{ln}_theta_true"]
        ttu = tt / (np.linalg.norm(tt) + 1e-08)
        ph3, ps3 = (h @ ttu, s @ ttu)
        d_3b_all.append(
            (ph3.mean() - ps3.mean())
            / (np.sqrt((ph3.std() ** 2 + ps3.std() ** 2) / 2) + 1e-08)
        )
        d_ttpd_all.append(ttpd_results[ln]["cohens_d"])
    x = np.arange(len(layers))
    width = 0.25
    bars1 = ax.bar(
        x - width,
        d_dim_all,
        width,
        label="DIM (Phase 3)",
        color="#90CAF9",
        edgecolor="#1565C0",
    )
    bars2 = ax.bar(
        x,
        d_3b_all,
        width,
        label="Disentangled (Phase 3b)",
        color="#A5D6A7",
        edgecolor="#2E7D32",
    )
    bars3 = ax.bar(
        x + width,
        d_ttpd_all,
        width,
        label="TTPD (Phase 4)",
        color="#FFB74D",
        edgecolor="#E65100",
    )
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Cohen's d", fontsize=12)
    ax.set_title(
        "Honest vs Scheming Discrimination by Method Across Layers\n(Higher = better separation)",
        fontsize=13,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{lid}" for lid in layer_ids], fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=2.0, color="gray", linestyle=":", alpha=0.5)
    ax.text(
        len(layers) - 0.5, 2.05, "EXCELLENT threshold (d=2.0)", fontsize=9, color="gray"
    )
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    plt.tight_layout()
    fig.savefig(f"{plots_dir}/03_cohens_d_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {plots_dir}/03_cohens_d_comparison.png")
    print("  Creating t_G vs t_P scatter...")
    fig, ax = plt.subplots(figsize=(10, 8))
    proj_h_tG = honest @ ttpd["t_G_unit"]
    proj_s_tG = scheming @ ttpd["t_G_unit"]
    proj_h_tP = honest @ ttpd["t_P_unit"]
    proj_s_tP = scheming @ ttpd["t_P_unit"]
    ax.scatter(
        proj_h_tG,
        proj_h_tP,
        c="#2196F3",
        alpha=0.3,
        s=10,
        label="Honest",
        rasterized=True,
    )
    ax.scatter(
        proj_s_tG,
        proj_s_tP,
        c="#F44336",
        alpha=0.3,
        s=10,
        label="Scheming",
        rasterized=True,
    )
    ax.axvline(
        ttpd["threshold_truth"],
        color="black",
        linestyle="--",
        lw=1.5,
        alpha=0.7,
        label=f"Truth threshold",
    )
    ax.scatter(
        proj_h_tG.mean(),
        proj_h_tP.mean(),
        c="blue",
        s=200,
        marker="*",
        edgecolor="white",
        linewidth=1.5,
        zorder=5,
        label="Honest centroid",
    )
    ax.scatter(
        proj_s_tG.mean(),
        proj_s_tP.mean(),
        c="red",
        s=200,
        marker="*",
        edgecolor="white",
        linewidth=1.5,
        zorder=5,
        label="Scheming centroid",
    )
    ax.set_xlabel("Projection onto $t_G$ (Truth Direction) →", fontsize=12)
    ax.set_ylabel("Projection onto $t_P$ (Polarity Direction) →", fontsize=12)
    ax.set_title(
        f"TTPD Decomposition — {best_layer}\nTruth separates horizontally, polarity separates vertically",
        fontsize=13,
    )
    ax.legend(fontsize=10, markerscale=1.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{plots_dir}/04_ttpd_2d_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {plots_dir}/04_ttpd_2d_scatter.png")
    print("  Creating t-SNE visualization (this takes ~20 seconds)...")
    n_sub = min(500, len(honest))
    rng = np.random.RandomState(42)
    idx_h = rng.choice(len(honest), n_sub, replace=False)
    idx_s = rng.choice(len(scheming), n_sub, replace=False)
    sub_data = np.vstack([honest[idx_h], scheming[idx_s]])
    sub_labels = np.array([0] * n_sub + [1] * n_sub)
    sub_cats = [categories[i] for i in idx_h]
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    tsne_2d = tsne.fit_transform(sub_data)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    ax = axes[0]
    ax.scatter(
        tsne_2d[:n_sub, 0],
        tsne_2d[:n_sub, 1],
        c="#2196F3",
        alpha=0.5,
        s=15,
        label="Honest",
        rasterized=True,
    )
    ax.scatter(
        tsne_2d[n_sub:, 0],
        tsne_2d[n_sub:, 1],
        c="#F44336",
        alpha=0.5,
        s=15,
        label="Scheming",
        rasterized=True,
    )
    ax.set_title(
        f"t-SNE: Honest vs Scheming — {best_layer}\n(n={n_sub} each)", fontsize=13
    )
    ax.legend(fontsize=11, markerscale=2)
    ax.set_xlabel("t-SNE 1", fontsize=11)
    ax.set_ylabel("t-SNE 2", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax = axes[1]
    for cat in unique_cats:
        mask = np.array([c == cat for c in sub_cats])
        if mask.sum() > 0:
            ax.scatter(
                tsne_2d[:n_sub][mask, 0],
                tsne_2d[:n_sub][mask, 1],
                c=[cat_cmap[cat]],
                alpha=0.5,
                s=15,
                marker="o",
            )
            ax.scatter(
                tsne_2d[n_sub:][mask, 0],
                tsne_2d[n_sub:][mask, 1],
                c=[cat_cmap[cat]],
                alpha=0.5,
                s=15,
                marker="x",
            )
    legend_elements = [
        Patch(facecolor=cat_cmap[c], label=c.replace("_", " ").title())
        for c in unique_cats
    ]
    legend_elements.extend(
        [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="gray",
                linestyle="None",
                label="Honest",
                markersize=8,
            ),
            plt.Line2D(
                [0],
                [0],
                marker="x",
                color="gray",
                linestyle="None",
                label="Scheming",
                markersize=8,
            ),
        ]
    )
    ax.legend(handles=legend_elements, fontsize=8, loc="upper right", ncol=2)
    ax.set_title(
        f"t-SNE by Category — {best_layer}\n(o=Honest, x=Scheming)", fontsize=13
    )
    ax.set_xlabel("t-SNE 1", fontsize=11)
    ax.set_ylabel("t-SNE 2", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(
        f"{plots_dir}/05_tsne_honest_vs_scheming.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
    print(f"    Saved: {plots_dir}/05_tsne_honest_vs_scheming.png")
    print("  Creating per-category separation chart...")
    fig, ax = plt.subplots(figsize=(12, 6))
    cat_d_values = {}
    for cat in unique_cats:
        mask = np.array([c == cat for c in categories])
        h_cat = honest[mask]
        s_cat = scheming[mask]
        ph = h_cat @ ttpd["t_G_unit"]
        ps = s_cat @ ttpd["t_G_unit"]
        pooled = np.sqrt((ph.std() ** 2 + ps.std() ** 2) / 2) + 1e-08
        cat_d_values[cat] = (ph.mean() - ps.mean()) / pooled
    cats_sorted = sorted(
        cat_d_values.keys(), key=lambda c: cat_d_values[c], reverse=True
    )
    d_vals = [cat_d_values[c] for c in cats_sorted]
    colors = [cat_cmap[c] for c in cats_sorted]
    bars = ax.bar(
        range(len(cats_sorted)),
        d_vals,
        color=colors,
        edgecolor="#333333",
        linewidth=0.5,
    )
    ax.set_xticks(range(len(cats_sorted)))
    ax.set_xticklabels(
        [c.replace("_", " ").title() for c in cats_sorted],
        rotation=30,
        ha="right",
        fontsize=10,
    )
    ax.set_ylabel("Cohen's d (TTPD $t_G$)", fontsize=12)
    ax.set_title(
        f"Per-Category Honest/Scheming Separation via TTPD — {best_layer}", fontsize=13
    )
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=2.0, color="gray", linestyle=":", alpha=0.5)
    ax.text(len(cats_sorted) - 0.5, 2.05, "d=2.0 (EXCELLENT)", fontsize=9, color="gray")
    for bar, d in zip(bars, d_vals):
        ax.annotate(
            f"{d:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, d),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=10,
        )
    plt.tight_layout()
    fig.savefig(
        f"{plots_dir}/06_per_category_separation.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
    print(f"    Saved: {plots_dir}/06_per_category_separation.png")


def main():
    parser = argparse.ArgumentParser(description="Phase 4: TTPD Geometry Discovery")
    parser.add_argument("--test", action="store_true", help="Quick test with mock data")
    parser.add_argument(
        "--data-dir", type=str, default=DATA_DIR, help="Path to activations"
    )
    parser.add_argument(
        "--plots-dir", type=str, default=PLOTS_DIR, help="Where to save plots"
    )
    args = parser.parse_args()
    data_dir = args.data_dir
    plots_dir = args.plots_dir
    print("=" * 65)
    print("PHASE 4: GEOMETRY DISCOVERY VIA TTPD")
    print("  Training of Truth and Polarity Direction")
    print("=" * 65)
    if args.test:
        run_test_mode(plots_dir)
        return
    print("\n[1/3] Loading activations and vectors...")
    with h5py.File(f"{data_dir}/activations_consolidated.h5", "r") as f:
        layers = sorted(f["honest"].keys(), key=lambda x: int(x.split("_")[1]))
        honest_all = {ln: np.array(f["honest"][ln]) for ln in layers}
        scheming_all = {ln: np.array(f["scheming"][ln]) for ln in layers}
    with h5py.File(f"{data_dir}/polarity_activations.h5", "r") as f:
        aff_all = {ln: np.array(f["affirmative"][ln]) for ln in layers}
        neg_all = {ln: np.array(f["negated"][ln]) for ln in layers}
    n_honest = honest_all[layers[0]].shape[0]
    n_polarity = aff_all[layers[0]].shape[0]
    hidden_dim = honest_all[layers[0]].shape[1]
    print(f"  Honest/Scheming pairs: {n_honest}")
    print(f"  Polarity pairs: {n_polarity}")
    print(f"  Layers: {len(layers)}")
    print(f"  Hidden dim: {hidden_dim}")
    print("\n[2/3] Computing TTPD directions per layer...")
    header = f"  {'Layer':<10} {'‖t_G‖':>7} {'‖t_P‖':>7} {'cos(G,P)':>9} {'d_TTPD':>7} {'Acc_truth':>10} {'Acc_pol':>8} {'Neg_gen':>8} {'R²':>6}"
    print(f"\n{header}")
    print("  " + "-" * 88)
    ttpd_results = {}
    for ln in layers:
        result = compute_ttpd_direction(
            honest_acts=honest_all[ln],
            scheming_acts=scheming_all[ln],
            aff_acts=aff_all[ln],
            neg_acts=neg_all[ln],
        )
        ttpd_results[ln] = result
        lid = ln.split("_")[1]
        print(
            f"  Layer {lid:<4} {result['t_G_norm']:>7.4f} {result['t_P_norm']:>7.4f} {result['cos_truth_polarity']:>9.4f} {result['cohens_d']:>7.3f} {result['accuracy_truth'] * 100:>9.1f}% {result['accuracy_polarity'] * 100:>7.1f}% {result['negation_generalization'] * 100:>7.1f}% {result['r_squared']:>6.4f}"
        )
    best_layer = max(ttpd_results, key=lambda x: ttpd_results[x]["cohens_d"])
    best = ttpd_results[best_layer]
    print(f"\n  Best layer: {best_layer}")
    print(f"    Cohen's d:              {best['cohens_d']:.3f}")
    print(f"    Truth accuracy:         {best['accuracy_truth'] * 100:.1f}%")
    print(f"    Polarity accuracy:      {best['accuracy_polarity'] * 100:.1f}%")
    print(f"    Negation generalization: {best['negation_generalization'] * 100:.1f}%")
    print(f"    cos(t_G, t_P):          {best['cos_truth_polarity']:.4f}")
    neg_gen_pass = best["negation_generalization"] > 0.9
    print(
        f"\n  Success metric (neg generalization > 90%): {('PASS ✓' if neg_gen_pass else 'FAIL ✗')}"
    )
    print("\n  Saving TTPD vectors...")
    os.makedirs(data_dir, exist_ok=True)
    save_dict = {}
    for ln, res in ttpd_results.items():
        save_dict[f"{ln}_t_G"] = res["t_G"]
        save_dict[f"{ln}_t_G_unit"] = res["t_G_unit"]
        save_dict[f"{ln}_t_P"] = res["t_P"]
        save_dict[f"{ln}_t_P_unit"] = res["t_P_unit"]
        save_dict[f"{ln}_global_mean"] = res["global_mean"]
    np.savez_compressed(f"{data_dir}/steering_vectors_ttpd.npz", **save_dict)
    print(f"    {data_dir}/steering_vectors_ttpd.npz")
    stats_out = {
        "method": "TTPD",
        "n_honest_scheming": n_honest,
        "n_polarity": n_polarity,
        "hidden_dim": hidden_dim,
        "best_layer": best_layer,
        "best_cohens_d": best["cohens_d"],
        "best_accuracy_truth": best["accuracy_truth"],
        "best_negation_generalization": best["negation_generalization"],
        "layers": {
            ln: {
                k: v
                for k, v in res.items()
                if k not in ("t_G", "t_G_unit", "t_P", "t_P_unit", "global_mean")
            }
            for ln, res in ttpd_results.items()
        },
    }
    json.dump(stats_out, open(f"{data_dir}/ttpd_stats.json", "w"), indent=2)
    print(f"    {data_dir}/ttpd_stats.json")
    print("\n[3/3] Creating visualizations...")
    create_visualizations(data_dir, ttpd_results, plots_dir, layers, best_layer)
    print(f"\n{'=' * 65}")
    print("PHASE 4 COMPLETE")
    print(f"{'=' * 65}")
    print(f"\n  TTPD learned t_G (truth) and t_P (polarity) jointly via OLS regression")
    print(f"  Best layer: {best_layer} (d = {best['cohens_d']:.3f})")
    print(f"  Negation generalization: {best['negation_generalization'] * 100:.1f}%")
    print(f"\n  New files:")
    print(f"    {data_dir}/steering_vectors_ttpd.npz   — t_G and t_P for all layers")
    print(f"    {data_dir}/ttpd_stats.json             — accuracy, Cohen's d, R²")
    print(f"\n  Plots ({plots_dir}/):")
    print(f"    01_pca_honest_vs_scheming.png    — PCA scatter with t_G arrow")
    print(f"    02_projection_histograms.png     — separation along 4 directions")
    print(f"    03_cohens_d_comparison.png        — DIM vs Phase3b vs TTPD per layer")
    print(f"    04_ttpd_2d_scatter.png            — t_G vs t_P decomposition")
    print(f"    05_tsne_honest_vs_scheming.png    — t-SNE clustering")
    print(f"    06_per_category_separation.png    — category-wise Cohen's d")


def run_test_mode(plots_dir):
    print("\n  TEST MODE — using synthetic data\n")
    hidden_dim = 64
    n = 100
    rng = np.random.RandomState(42)
    true_truth = rng.randn(hidden_dim)
    true_truth /= np.linalg.norm(true_truth)
    true_polarity = rng.randn(hidden_dim)
    true_polarity -= np.dot(true_polarity, true_truth) * true_truth
    true_polarity /= np.linalg.norm(true_polarity)
    noise = 0.3
    honest = 2 * true_truth[None, :] + noise * rng.randn(n, hidden_dim)
    scheming = -2 * true_truth[None, :] + noise * rng.randn(n, hidden_dim)
    aff = 1.5 * true_polarity[None, :] + noise * rng.randn(n, hidden_dim)
    neg = -1.5 * true_polarity[None, :] + noise * rng.randn(n, hidden_dim)
    result = compute_ttpd_direction(honest, scheming, aff, neg)
    print(f"  ‖t_G‖ = {result['t_G_norm']:.4f}")
    print(f"  ‖t_P‖ = {result['t_P_norm']:.4f}")
    print(f"  cos(t_G, t_P) = {result['cos_truth_polarity']:.4f}")
    print(f"  Cohen's d = {result['cohens_d']:.3f}")
    print(f"  Accuracy (truth) = {result['accuracy_truth'] * 100:.1f}%")
    print(f"  Accuracy (polarity) = {result['accuracy_polarity'] * 100:.1f}%")
    print(f"  Neg generalization = {result['negation_generalization'] * 100:.1f}%")
    print(f"  R² = {result['r_squared']:.4f}")
    cos_gt = abs(np.dot(result["t_G_unit"], true_truth))
    cos_gp = abs(np.dot(result["t_P_unit"], true_polarity))
    print(f"\n  Alignment with ground truth:")
    print(f"    |cos(t_G, true_truth)| = {cos_gt:.4f}")
    print(f"    |cos(t_P, true_polarity)| = {cos_gp:.4f}")
    all_pass = (
        result["accuracy_truth"] > 0.95
        and result["negation_generalization"] > 0.9
        and (cos_gt > 0.95)
        and (cos_gp > 0.95)
    )
    print(f"\n  {('ALL TESTS PASSED ✓' if all_pass else 'SOME TESTS FAILED ✗')}")


if __name__ == "__main__":
    main()
