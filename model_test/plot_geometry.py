# ==============================================================================
# Plot Geometry Results — Aggregate CNN 8 seeds vs SAE 4 seeds
#
# 저장된 geometry_eval 결과를 읽어서:
#   1) Ricci curvature (mean, frac_positive, frac_negative) vs K — line plot
#   2) Gromov δ-hyperbolicity — bar chart
#
# Expected directory structure (from run_geometry_eval.sh compare):
#   {base_dir}/
#     cnn_seed_42/
#       k_5/geometry_results_k5.json
#       k_10/geometry_results_k10.json
#       ...
#     cnn_seed_87/...
#     sae_seed_48/...
#     sae_seed_856/...
#
# Usage:
# !python -m model_test.plot_geometry \
#     --base_dir /content/drive/MyDrive/Final_paper/lambda_labs_moco_only/geometry_eval/compare
# ==============================================================================

import os
import sys
import json
import glob
import argparse
import numpy as np
from collections import defaultdict

import matplotlib
_IN_COLAB = "google.colab" in sys.modules
if not _IN_COLAB:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, kendalltau, pearsonr

try:
    from pygam import LinearGAM, s as gam_s
    _HAS_PYGAM = True
except ImportError:
    from scipy.interpolate import UnivariateSpline
    _HAS_PYGAM = False

from sae_project.step02_logging_utils import get_logger

logger = get_logger("plot_geometry")

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'
sns.set_style("ticks")


# ==============================================================================
# Argument Parser
# ==============================================================================
def get_args():
    p = argparse.ArgumentParser(
        description="Plot aggregated geometry results (CNN vs SAE)")
    p.add_argument("--base_dir", type=str, required=True,
                   help="Directory with cnn_seed_*/sae_seed_* subdirs")
    p.add_argument("--output_dir", type=str, default="",
                   help="Output directory (default: base_dir)")
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args()


# ==============================================================================
# Scan and load results
# ==============================================================================
def scan_results(base_dir):
    data = {"CNN": defaultdict(list), "SAE": defaultdict(list)}
    delta_data = {"CNN": [], "SAE": []}
    per_class_data = {"CNN": defaultdict(lambda: defaultdict(list)),
                      "SAE": defaultdict(lambda: defaultdict(list))}

    for entry in sorted(os.listdir(base_dir)):
        entry_path = os.path.join(base_dir, entry)
        if not os.path.isdir(entry_path):
            continue

        if entry.startswith("cnn_seed_"):
            source = "CNN"
            seed_str = entry.replace("cnn_seed_", "")
        elif entry.startswith("sae_seed_"):
            source = "SAE"
            seed_str = entry.replace("sae_seed_", "")
        else:
            continue

        k_dirs = sorted(glob.glob(os.path.join(entry_path, "k_*")))

        if k_dirs:
            for k_dir in k_dirs:
                k_name = os.path.basename(k_dir)
                k_val = int(k_name.replace("k_", ""))

                jsons = glob.glob(os.path.join(k_dir, "geometry_results*.json"))
                if not jsons:
                    continue
                with open(jsons[0], "r") as f:
                    results = json.load(f)

                for src_label, r in results.items():
                    ricci = r.get("ricci", {})
                    delta = r.get("delta", {})

                    if "error" in ricci:
                        continue

                    data[source][k_val].append({
                        "seed": seed_str,
                        "k": k_val,
                        "ricci_mean": ricci.get("mean", np.nan),
                        "ricci_median": ricci.get("median", np.nan),
                        "ricci_std": ricci.get("std", np.nan),
                        "frac_positive": ricci.get("frac_positive", np.nan),
                        "frac_negative": ricci.get("frac_negative", np.nan),
                        "q25": ricci.get("q25", np.nan),
                        "q75": ricci.get("q75", np.nan),
                    })

                    ricci_pc = r.get("ricci_per_class", {})
                    for cls, cls_stats in ricci_pc.items():
                        per_class_data[source][k_val][cls].append({
                            "seed": seed_str,
                            "mean": cls_stats.get("mean", np.nan),
                            "median": cls_stats.get("median", np.nan),
                            "n_edges": cls_stats.get("n_edges", 0),
                        })

                    if delta and "error" not in delta and "delta" in delta:
                        delta_data[source].append({
                            "seed": seed_str,
                            "delta": delta["delta"],
                            "delta_rel": delta["delta_rel"],
                            "diameter": delta["diameter"],
                            "delta_mean": delta.get("delta_mean", np.nan),
                            "delta_median": delta.get("delta_median", np.nan),
                        })
        else:
            jsons = glob.glob(os.path.join(entry_path, "geometry_results*.json"))
            if not jsons:
                continue
            with open(jsons[0], "r") as f:
                results = json.load(f)

            for src_label, r in results.items():
                ricci = r.get("ricci", {})
                delta = r.get("delta", {})
                k_val = ricci.get("k", r.get("k", 15))

                if "error" in ricci:
                    continue

                data[source][k_val].append({
                    "seed": seed_str,
                    "k": k_val,
                    "ricci_mean": ricci.get("mean", np.nan),
                    "ricci_median": ricci.get("median", np.nan),
                    "ricci_std": ricci.get("std", np.nan),
                    "frac_positive": ricci.get("frac_positive", np.nan),
                    "frac_negative": ricci.get("frac_negative", np.nan),
                    "q25": ricci.get("q25", np.nan),
                    "q75": ricci.get("q75", np.nan),
                })

                if delta and "error" not in delta and "delta" in delta:
                    delta_data[source].append({
                        "seed": seed_str,
                        "delta": delta["delta"],
                        "delta_rel": delta["delta_rel"],
                        "diameter": delta["diameter"],
                        "delta_mean": delta.get("delta_mean", np.nan),
                        "delta_median": delta.get("delta_median", np.nan),
                    })

    for source in ["CNN", "SAE"]:
        k_vals = sorted(data[source].keys())
        n_seeds = {k: len(data[source][k]) for k in k_vals}
        n_delta = len(delta_data[source])
        logger.info(f"  {source}: k_vals={k_vals}, "
                    f"seeds_per_k={n_seeds}, δ_seeds={n_delta}")

    return dict(data), delta_data, {s: dict(v) for s, v in per_class_data.items()}


# ==============================================================================
# Aggregate: mean ± std across seeds
# ==============================================================================
def aggregate(data):
    agg = {}
    for source, k_dict in data.items():
        agg[source] = {}
        for k_val, entries in sorted(k_dict.items()):
            fields = ["ricci_mean", "ricci_median", "frac_positive",
                       "frac_negative", "q25", "q75"]
            row = {"k": k_val, "n_seeds": len(entries)}
            for f in fields:
                vals = np.array([e[f] for e in entries if not np.isnan(e[f])])
                row[f"{f}_mean"] = float(np.mean(vals)) if len(vals) > 0 else np.nan
                row[f"{f}_std"] = float(np.std(vals)) if len(vals) > 0 else np.nan
            agg[source][k_val] = row
    return agg


# ==============================================================================
# Plot 1: Ricci curvature statistics vs K
# ==============================================================================
def plot_ricci_vs_k(agg, out_dir, dpi=200):
    colors = {"CNN": "#3A7EBF", "SAE": "#E8833A"}
    markers = {"CNN": "o", "SAE": "s"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = [
        ("ricci_mean", "Mean Curvature (κ)", "κ"),
        ("frac_positive", "Fraction Positive Curvature", "%"),
        ("frac_negative", "Fraction Negative Curvature", "%"),
    ]

    for ax, (field, title, unit) in zip(axes, metrics):
        for source in ["CNN", "SAE"]:
            if source not in agg:
                continue
            k_vals = sorted(agg[source].keys())
            means = np.array([agg[source][k][f"{field}_mean"] for k in k_vals])
            stds = np.array([agg[source][k][f"{field}_std"] for k in k_vals])

            # Percentage scaling
            if "frac" in field:
                means *= 100
                stds *= 100

            ax.plot(k_vals, means, f"{markers[source]}-",
                    color=colors[source], linewidth=2.5, markersize=8,
                    label=source, zorder=3)
            ax.fill_between(k_vals, means - stds, means + stds,
                            color=colors[source], alpha=0.15)

        ax.axhline(0 if "frac" not in field else 50,
                   color="gray", linestyle=":", linewidth=1, alpha=0.5)

        ax.set_xlabel("Neighborhood Size (K)", fontsize=11)
        ax.set_ylabel(unit, fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.15)
        sns.despine(ax=ax)

    fig.suptitle("Ollivier-Ricci Curvature Statistics: CNN vs SAE",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()

    fname = "ricci_statistics_vs_k"
    for ext in [".png", ".svg"]:
        fig.savefig(os.path.join(out_dir, fname + ext),
                    dpi=dpi, bbox_inches="tight")
    logger.info(f"  Saved: {fname}.png/.svg")
    if _IN_COLAB:
        plt.show()
    plt.close(fig)


# ==============================================================================
# Plot: Per-class intra-class curvature vs K
# ==============================================================================
def plot_ricci_per_class_vs_k(per_class_data, out_dir, dpi=200):
    all_classes = set()
    for source_dict in per_class_data.values():
        for k_val, cls_dict in source_dict.items():
            all_classes.update(cls_dict.keys())
    if not all_classes:
        logger.warning("  No per-class data found. Skipping per-class plot.")
        return

    class_order = ["Control", "SNCA", "GBA", "LRRK2"]
    classes = [c for c in class_order if c in all_classes]
    for c in sorted(all_classes):
        if c not in classes:
            classes.append(c)

    n_cls = len(classes)
    colors = {"CNN": "#3A7EBF", "SAE": "#E8833A"}
    markers = {"CNN": "o", "SAE": "s"}
    class_colors = {
        "Control": "#55A868", "SNCA": "#C44E52",
        "GBA": "#8172B2", "LRRK2": "#CCB974",
    }

    ncols = min(n_cls, 4)
    nrows = (n_cls + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows),
                             squeeze=False)

    for idx, cls in enumerate(classes):
        ax = axes[idx // ncols][idx % ncols]
        for source in ["CNN", "SAE"]:
            source_dict = per_class_data.get(source, {})
            k_vals = sorted(source_dict.keys())
            means_list = []
            stds_list = []
            valid_ks = []
            for k in k_vals:
                entries = source_dict[k].get(cls, [])
                if not entries:
                    continue
                vals = np.array([e["mean"] for e in entries
                                 if not np.isnan(e["mean"])])
                if len(vals) == 0:
                    continue
                valid_ks.append(k)
                means_list.append(np.mean(vals))
                stds_list.append(np.std(vals))

            if not valid_ks:
                continue
            means_arr = np.array(means_list)
            stds_arr = np.array(stds_list)

            ax.plot(valid_ks, means_arr, f"{markers[source]}-",
                    color=colors[source], linewidth=2, markersize=7,
                    label=source, zorder=3)
            ax.fill_between(valid_ks, means_arr - stds_arr,
                            means_arr + stds_arr,
                            color=colors[source], alpha=0.15)

        ax.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.5)
        face_color = class_colors.get(cls, "#EEEEEE")
        ax.set_facecolor(f"{face_color}11")
        ax.set_xlabel("K", fontsize=10)
        ax.set_ylabel("Intra-class mean κ", fontsize=10)
        ax.set_title(cls, fontsize=12, fontweight="bold",
                     color=class_colors.get(cls, "black"))
        ax.legend(fontsize=9, framealpha=0.85)
        ax.grid(True, alpha=0.15)
        sns.despine(ax=ax)

    for idx in range(n_cls, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("Intra-class Ollivier-Ricci Curvature vs K",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()

    fname = "ricci_per_class_vs_k"
    for ext in [".png", ".svg"]:
        fig.savefig(os.path.join(out_dir, fname + ext),
                    dpi=dpi, bbox_inches="tight")
    logger.info(f"  Saved: {fname}.png/.svg")
    if _IN_COLAB:
        plt.show()
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, source in zip(axes, ["CNN", "SAE"]):
        source_dict = per_class_data.get(source, {})
        k_vals = sorted(source_dict.keys())
        for cls in classes:
            valid_ks = []
            means_list = []
            for k in k_vals:
                entries = source_dict[k].get(cls, [])
                if not entries:
                    continue
                vals = np.array([e["mean"] for e in entries
                                 if not np.isnan(e["mean"])])
                if len(vals) == 0:
                    continue
                valid_ks.append(k)
                means_list.append(np.mean(vals))
            if valid_ks:
                c = class_colors.get(cls, "#999999")
                ax.plot(valid_ks, means_list, "o-", color=c,
                        linewidth=2, markersize=6, label=cls)

        ax.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.5)
        ax.set_xlabel("K", fontsize=11)
        ax.set_ylabel("Intra-class mean κ", fontsize=11)
        ax.set_title(f"{source} — Intra-class Curvature",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, framealpha=0.85)
        ax.grid(True, alpha=0.15)
        sns.despine(ax=ax)

    fig.suptitle("Per-class Intra-class Curvature: CNN vs SAE",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    fname = "ricci_per_class_by_source"
    for ext in [".png", ".svg"]:
        fig.savefig(os.path.join(out_dir, fname + ext),
                    dpi=dpi, bbox_inches="tight")
    logger.info(f"  Saved: {fname}.png/.svg")
    if _IN_COLAB:
        plt.show()
    plt.close(fig)


# ==============================================================================
# Plot 2: δ-Hyperbolicity comparison
# ==============================================================================
def plot_delta_comparison(delta_data, out_dir, dpi=200):
    colors = {"CNN": "#3A7EBF", "SAE": "#E8833A"}

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    fields = [
        ("delta", "δ-Hyperbolicity (δ)", "δ"),
        ("delta_rel", "Normalized δ (δ / diameter)", "δ / D"),
    ]

    for ax, (field, title, ylabel) in zip(axes, fields):
        positions = []
        labels = []
        pos_idx = 0

        for source in ["CNN", "SAE"]:
            entries = delta_data.get(source, [])
            if not entries:
                continue

            seen_seeds = set()
            unique_entries = []
            for e in entries:
                if e["seed"] not in seen_seeds:
                    seen_seeds.add(e["seed"])
                    unique_entries.append(e)

            vals = np.array([e[field] for e in unique_entries])
            mean_val = np.mean(vals)
            std_val = np.std(vals)

            ax.bar(pos_idx, mean_val, width=0.6,
                   color=colors[source], alpha=0.6,
                   edgecolor=colors[source], linewidth=1.5,
                   zorder=2)

            ax.errorbar(pos_idx, mean_val, yerr=std_val,
                        color="black", capsize=6, capthick=1.5,
                        linewidth=1.5, zorder=4)

            jitter = np.random.RandomState(42).uniform(
                -0.15, 0.15, size=len(vals))
            ax.scatter(pos_idx + jitter, vals, color=colors[source],
                       edgecolor="white", s=40, linewidth=0.8,
                       zorder=5, alpha=0.8)

            ax.text(pos_idx, mean_val + std_val + 0.005,
                    f"{mean_val:.4f}\n±{std_val:.4f}",
                    ha="center", va="bottom", fontsize=9,
                    fontweight="bold")

            positions.append(pos_idx)
            labels.append(f"{source}\n(n={len(vals)})")
            pos_idx += 1

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.15, axis="y")
        sns.despine(ax=ax)

    fig.suptitle("Gromov δ-Hyperbolicity: CNN vs SAE",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    fname = "delta_hyperbolicity_aggregated"
    for ext in [".png", ".svg"]:
        fig.savefig(os.path.join(out_dir, fname + ext),
                    dpi=dpi, bbox_inches="tight")
    logger.info(f"  Saved: {fname}.png/.svg")
    if _IN_COLAB:
        plt.show()
    plt.close(fig)


# ==============================================================================
# Plot 3: Combined summary — 2 × 2 figure
# ==============================================================================
def plot_combined_summary(agg, delta_data, out_dir, dpi=200):
    colors = {"CNN": "#3A7EBF", "SAE": "#E8833A"}
    markers = {"CNN": "o", "SAE": "s"}

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    ax = axes[0, 0]
    for source in ["CNN", "SAE"]:
        if source not in agg:
            continue
        k_vals = sorted(agg[source].keys())
        means = np.array([agg[source][k]["ricci_mean_mean"] for k in k_vals])
        stds = np.array([agg[source][k]["ricci_mean_std"] for k in k_vals])
        ax.plot(k_vals, means, f"{markers[source]}-",
                color=colors[source], linewidth=2, markersize=7,
                label=source, zorder=3)
        ax.fill_between(k_vals, means - stds, means + stds,
                        color=colors[source], alpha=0.15)
    ax.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.set_xlabel("K", fontsize=11)
    ax.set_ylabel("Mean κ", fontsize=11)
    ax.set_title("(A) Mean Ollivier-Ricci Curvature", fontsize=12,
                 fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.15)

    ax = axes[0, 1]
    for source in ["CNN", "SAE"]:
        if source not in agg:
            continue
        k_vals = sorted(agg[source].keys())
        means = np.array([agg[source][k]["frac_positive_mean"] for k in k_vals])
        stds = np.array([agg[source][k]["frac_positive_std"] for k in k_vals])
        ax.plot(k_vals, means * 100, f"{markers[source]}-",
                color=colors[source], linewidth=2, markersize=7,
                label=source, zorder=3)
        ax.fill_between(k_vals, (means - stds) * 100, (means + stds) * 100,
                        color=colors[source], alpha=0.15)
    ax.axhline(50, color="gray", linestyle=":", linewidth=1, alpha=0.5,
               label="50% (balanced)")
    ax.set_xlabel("K", fontsize=11)
    ax.set_ylabel("% edges with κ > 0", fontsize=11)
    ax.set_title("(B) Fraction of Positively Curved Edges",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15)

    ax = axes[1, 0]
    for i, source in enumerate(["CNN", "SAE"]):
        entries = delta_data.get(source, [])
        seen = set()
        unique = []
        for e in entries:
            if e["seed"] not in seen:
                seen.add(e["seed"])
                unique.append(e)
        if not unique:
            continue
        vals = np.array([e["delta"] for e in unique])
        mean_v = np.mean(vals)
        std_v = np.std(vals)
        ax.bar(i, mean_v, width=0.6, color=colors[source], alpha=0.6,
               edgecolor=colors[source], linewidth=1.5)
        ax.errorbar(i, mean_v, yerr=std_v, color="black",
                    capsize=6, capthick=1.5, linewidth=1.5, zorder=4)
        jitter = np.random.RandomState(42).uniform(-0.12, 0.12, len(vals))
        ax.scatter(i + jitter, vals, color=colors[source],
                   edgecolor="white", s=40, linewidth=0.8, zorder=5)
        ax.text(i, mean_v + std_v + 0.003,
                f"{mean_v:.4f}±{std_v:.4f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["CNN", "SAE"], fontsize=11, fontweight="bold")
    ax.set_ylabel("δ", fontsize=12)
    ax.set_title("(C) Gromov δ-Hyperbolicity", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.15, axis="y")

    ax = axes[1, 1]
    for i, source in enumerate(["CNN", "SAE"]):
        entries = delta_data.get(source, [])
        seen = set()
        unique = []
        for e in entries:
            if e["seed"] not in seen:
                seen.add(e["seed"])
                unique.append(e)
        if not unique:
            continue
        vals = np.array([e["delta_rel"] for e in unique])
        mean_v = np.mean(vals)
        std_v = np.std(vals)
        ax.bar(i, mean_v, width=0.6, color=colors[source], alpha=0.6,
               edgecolor=colors[source], linewidth=1.5)
        ax.errorbar(i, mean_v, yerr=std_v, color="black",
                    capsize=6, capthick=1.5, linewidth=1.5, zorder=4)
        jitter = np.random.RandomState(42).uniform(-0.12, 0.12, len(vals))
        ax.scatter(i + jitter, vals, color=colors[source],
                   edgecolor="white", s=40, linewidth=0.8, zorder=5)
        ax.text(i, mean_v + std_v + 0.003,
                f"{mean_v:.4f}±{std_v:.4f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["CNN", "SAE"], fontsize=11, fontweight="bold")
    ax.set_ylabel("δ / diameter", fontsize=12)
    ax.set_title("(D) Normalized δ-Hyperbolicity", fontsize=12,
                 fontweight="bold")
    ax.grid(True, alpha=0.15, axis="y")

    for ax in axes.flat:
        sns.despine(ax=ax)

    fig.suptitle("Intrinsic Geometry: CNN vs SAE Representations",
                 fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()

    fname = "geometry_summary_2x2"
    for ext in [".png", ".svg"]:
        fig.savefig(os.path.join(out_dir, fname + ext),
                    dpi=dpi, bbox_inches="tight")
    logger.info(f"  Saved: {fname}.png/.svg")
    if _IN_COLAB:
        plt.show()
    plt.close(fig)


# ==============================================================================
# Global Geometry (All Classes) Scatter + GAM and Correlations
# ==============================================================================
def fit_gam(cos, dpt, n_splines=20, lam=0.6, degree=3, seed=42,
            clip_pct=1.0, ci_width=0.95):
    """Fit GAM regression to (cosine, dpt) points."""
    rng   = np.random.RandomState(seed)
    n_fit = min(200_000, len(cos))
    idx   = rng.choice(len(cos), n_fit, replace=False)
    xf    = cos[idx].astype(np.float64)
    yf    = dpt[idx].astype(np.float64)

    if clip_pct > 0:
        lo_x = np.percentile(xf, clip_pct)
        hi_x = np.percentile(xf, 100.0 - clip_pct)
        lo_y = np.percentile(yf, clip_pct)
        hi_y = np.percentile(yf, 100.0 - clip_pct)
        keep = (xf >= lo_x) & (xf <= hi_x) & (yf >= lo_y) & (yf <= hi_y)
        xf, yf = xf[keep], yf[keep]

    x_pred = np.linspace(float(xf.min()), float(xf.max()), 500)
    y_lo, y_hi = np.full(500, np.nan), np.full(500, np.nan)

    if _HAS_PYGAM:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gam = LinearGAM(gam_s(0, n_splines=n_splines), lam=lam)
            gam.fit(xf[:, None], yf)
        yfit  = gam.predict(xf[:, None])
        yline = gam.predict(x_pred[:, None])
        if ci_width > 0:
            try:
                ci_ = gam.confidence_intervals(x_pred[:, None], width=ci_width)
                y_lo, y_hi = ci_[:, 0], ci_[:, 1]
            except Exception:
                pass
    else:
        sort_idx = np.argsort(xf)
        xs, ys   = xf[sort_idx], yf[sort_idx]
        x_uniq, inv = np.unique(xs, return_inverse=True)
        y_uniq = np.bincount(inv, weights=ys) / np.bincount(inv)
        if len(x_uniq) < degree + 1:
            coeffs = np.polyfit(xf, yf, 1)
            yfit   = np.polyval(coeffs, xf)
            yline  = np.polyval(coeffs, x_pred)
        else:
            s_val = max(len(x_uniq) * y_uniq.var() * 0.05, 1e-6)
            try:
                spl   = UnivariateSpline(x_uniq, y_uniq, k=min(degree, 5), s=s_val)
                yfit  = spl(xf)
                yline = spl(x_pred)
            except Exception:
                coeffs = np.polyfit(xf, yf, min(8, n_splines // 2))
                yfit   = np.polyval(coeffs, xf)
                yline  = np.polyval(coeffs, x_pred)

    ss_res = float(np.sum((yf - yfit) ** 2))
    ss_tot = float(np.sum((yf - yf.mean()) ** 2))
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    n, p_  = len(yf), n_splines
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / max(n - p_ - 1, 1)

    return x_pred, yline, y_lo, y_hi, float(adj_r2)

def compute_global_correlations_and_plot(base_dir, out_dir, dpi=200):
    """Computes global Mantel & row-wise correlations across seeds and plots GAM scatter."""
    logger.info(f"\n{'='*60}")
    logger.info(f"  Global Geometry: Scatter + GAM (All Classes Combined)")
    logger.info(f"{'='*60}")
    
    rng = np.random.RandomState(42)
    n_sample_plot = 80_000
    n_corr_subsample = 2_000_000
    
    stats = {"CNN": defaultdict(list), "SAE": defaultdict(list)}
    plot_data = {"CNN": {"cos": [], "dpt": []}, "SAE": {"cos": [], "dpt": []}}

    for source, pattern in [("CNN", "cnn_seed_*"), ("SAE", "sae_seed_*")]:
        seed_dirs = sorted(glob.glob(os.path.join(base_dir, pattern)))
        if not seed_dirs:
            continue
            
        for seed_dir in seed_dirs:
            # 1. Read Row-wise from rank_correlation.json
            rc_jsons = glob.glob(os.path.join(seed_dir, "rank_correlation.json"))
            if not rc_jsons:
                rc_jsons = glob.glob(os.path.join(seed_dir, "k_*", "rank_correlation.json"))
            
            if rc_jsons:
                try:
                    with open(rc_jsons[0], "r") as f:
                        rc_data = json.load(f)
                    for src_key, src_val in rc_data.items():
                        if isinstance(src_val, dict) and "error" not in src_val:
                            if "spearman_mean" in src_val: stats[source]["rw_spearman"].append(src_val["spearman_mean"])
                            if "kendall_mean" in src_val:  stats[source]["rw_kendall"].append(src_val["kendall_mean"])
                            if "pearson_mean" in src_val:  stats[source]["rw_pearson"].append(src_val["pearson_mean"])
                except Exception as e:
                    logger.warning(f"  Error reading {rc_jsons[0]}: {e}")
                    
            # 2. Read Global Pairs from combined.npz
            npz_cands = glob.glob(os.path.join(seed_dir, f"pairwise_{source}_combined.npz"))
            if not npz_cands:
                npz_cands = glob.glob(os.path.join(seed_dir, "pairwise_*combined.npz"))
            
            if npz_cands:
                try:
                    d = np.load(npz_cands[0], allow_pickle=True)
                    cos = d["cosine_dist"].astype(np.float32)
                    dpt = d["dpt_dist"].astype(np.float32)
                    
                    mask = np.isfinite(cos) & np.isfinite(dpt)
                    cos, dpt = cos[mask], dpt[mask]
                    N_total = len(cos)
                    
                    if N_total > 0:
                        idx_corr = rng.choice(N_total, min(N_total, n_corr_subsample), replace=False)
                        cos_sub, dpt_sub = cos[idx_corr], dpt[idx_corr]
                        
                        rho = float(spearmanr(cos_sub, dpt_sub).statistic)
                        tau = float(kendalltau(cos_sub, dpt_sub).statistic)
                        pr  = float(pearsonr(cos_sub, dpt_sub).statistic)
                        
                        stats[source]["gl_spearman"].append(rho)
                        stats[source]["gl_kendall"].append(tau)
                        stats[source]["gl_pearson"].append(pr)
                        
                        idx_plot = rng.choice(N_total, min(N_total, n_sample_plot // len(seed_dirs)), replace=False)
                        plot_data[source]["cos"].append(cos[idx_plot])
                        plot_data[source]["dpt"].append(dpt[idx_plot])
                except Exception as e:
                    logger.warning(f"  Error reading {npz_cands[0]}: {e}")

    logger.info(f"\n  ── Full Dataset Correlation (Inter + Intra Classes) ──")
    logger.info(f"  {'Metric':<20s}  {'CNN mean±std':>16s}  {'SAE mean±std':>16s}")
    logger.info(f"  {'─'*58}")
    
    avg_stats = {}
    for metric_name, display_name in [
        ("gl_spearman", "Global Spearman ρ"),
        ("gl_kendall",  "Global Kendall τ"),
        ("gl_pearson",  "Global Pearson r"),
        ("rw_spearman", "Row-wise Spearman ρ"),
        ("rw_kendall",  "Row-wise Kendall τ"),
        ("rw_pearson",  "Row-wise Pearson r"),
    ]:
        c_vals = stats["CNN"].get(metric_name, [])
        s_vals = stats["SAE"].get(metric_name, [])
        
        c_str = f"{np.mean(c_vals):+.4f}±{np.std(c_vals):.4f}" if c_vals else "—"
        s_str = f"{np.mean(s_vals):+.4f}±{np.std(s_vals):.4f}" if s_vals else "—"
        logger.info(f"  {display_name:<20s}  {c_str:>16s}  {s_str:>16s}")
        
        avg_stats[f"CNN_{metric_name}"] = np.mean(c_vals) if c_vals else np.nan
        avg_stats[f"SAE_{metric_name}"] = np.mean(s_vals) if s_vals else np.nan

    # 3. Plotting 1x2 GAM Scatter
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), squeeze=False)
    colors = {"CNN": "#3A7EBF", "SAE": "#E8833A"}
    
    for col_i, source in enumerate(["CNN", "SAE"]):
        ax = axes[0, col_i]
        
        if not plot_data[source]["cos"]:
            ax.set_visible(False)
            continue
            
        x_cls = np.concatenate(plot_data[source]["cos"])
        y_cls = np.concatenate(plot_data[source]["dpt"])
        color = colors[source]
        
        ax.scatter(x_cls, y_cls, c=color, alpha=0.12, s=1.0, linewidths=0, rasterized=True)
        
        has_gam = False
        adj_r2 = np.nan
        try:
            x_pred, y_line, y_lo, y_hi, adj_r2 = fit_gam(x_cls, y_cls, n_splines=12, lam=0.6, degree=3)
            ax.plot(x_pred, y_line, color="black", linewidth=2.2, zorder=5, label=f"GAM  adj-R²={adj_r2:.3f}")
            if np.isfinite(y_lo).any():
                ax.fill_between(x_pred, y_lo, y_hi, alpha=0.18, color="black", zorder=4)
            has_gam = True
        except Exception as e:
            logger.warning(f"  [{source}] GAM failed: {e}")
            
        xy_max = max(float(x_cls.max()), float(y_cls.max())) * 1.05
        ax.plot([0, xy_max], [0, xy_max], color="#aaaaaa", linewidth=1.0, linestyle="--", alpha=0.65)
        
        x_lim = float(x_cls.max()) * 1.06
        y_lim = float(y_cls.max()) * 1.06
        ax.set_xlim(-x_lim * 0.01, x_lim)
        ax.set_ylim(-y_lim * 0.01, y_lim)
        
        g_rho = avg_stats.get(f"{source}_gl_spearman", np.nan)
        g_tau = avg_stats.get(f"{source}_gl_kendall", np.nan)
        r_rho = avg_stats.get(f"{source}_rw_spearman", np.nan)
        
        ann = f"Global ρ={g_rho:.3f}  τ={g_tau:.3f}"
        ann += f"\nRow-wise ρ={r_rho:.3f}"
        if has_gam:
            ann += f"\nadj-R²={adj_r2:.3f}"
        ann += f"\nn_disp={len(x_cls):,}"
        
        ax.text(0.04, 0.96, ann, transform=ax.transAxes, fontsize=8.0, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="#cccccc"))
                
        ax.set_xlabel("Cosine distance", fontsize=11)
        ax.set_ylabel("DPT geodesic" if col_i == 0 else "", fontsize=11)
        ax.set_title(f"{source}  │  All Classes (Inter + Intra)", fontsize=12, fontweight="bold")
        if has_gam:
            ax.legend(fontsize=9, loc="lower right")
        sns.despine(ax=ax)

    fig.suptitle("Global Geometry: Cosine vs DPT (All Pairs)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    
    fname = "global_scatter_gam"
    for ext in [".png", ".svg"]:
        fig.savefig(os.path.join(out_dir, fname + ext), dpi=dpi, bbox_inches="tight")
    logger.info(f"  Saved: {fname}.png/.svg")
    if _IN_COLAB:
        plt.show()
    plt.close(fig)


# ==============================================================================
# Main
# ==============================================================================
def main():
    args = get_args()
    out_dir = args.output_dir or args.base_dir
    os.makedirs(out_dir, exist_ok=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"  Aggregating Geometry Results")
    logger.info(f"  Base: {args.base_dir}")
    logger.info(f"{'='*60}")

    # ── Scan results ──
    data, delta_data, per_class_data = scan_results(args.base_dir)

    # ── Aggregate ──
    agg = aggregate(data)

    # ── Print table ──
    logger.info(f"\n  ── Aggregated Ricci Curvature ──")
    logger.info(f"  {'Source':6s}  {'K':>4s}  {'n':>3s}  "
                f"{'κ_mean':>12s}  {'%pos':>12s}  {'%neg':>12s}")
    logger.info(f"  {'─'*55}")
    for source in ["CNN", "SAE"]:
        for k_val in sorted(agg.get(source, {}).keys()):
            r = agg[source][k_val]
            logger.info(
                f"  {source:6s}  {k_val:4d}  {r['n_seeds']:3d}  "
                f"{r['ricci_mean_mean']:6.4f}±{r['ricci_mean_std']:.4f}  "
                f"{r['frac_positive_mean']*100:5.1f}±{r['frac_positive_std']*100:.1f}%  "
                f"{r['frac_negative_mean']*100:5.1f}±{r['frac_negative_std']*100:.1f}%")

    logger.info(f"\n  ── Aggregated δ-Hyperbolicity ──")
    for source in ["CNN", "SAE"]:
        entries = delta_data.get(source, [])
        seen = set()
        unique = [e for e in entries if e["seed"] not in seen
                  and not seen.add(e["seed"])]
        if unique:
            deltas = np.array([e["delta"] for e in unique])
            d_rels = np.array([e["delta_rel"] for e in unique])
            logger.info(f"  {source}: δ = {np.mean(deltas):.4f}±{np.std(deltas):.4f}, "
                        f"δ/D = {np.mean(d_rels):.4f}±{np.std(d_rels):.4f} "
                        f"(n={len(unique)})")

    # ── Plots ──
    logger.info(f"\n  Generating plots...")

    plot_ricci_vs_k(agg, out_dir, args.dpi)
    plot_ricci_per_class_vs_k(per_class_data, out_dir, args.dpi)
    plot_delta_comparison(delta_data, out_dir, args.dpi)
    plot_combined_summary(agg, delta_data, out_dir, args.dpi)

    # ── Global Correlation & Scatter GAM ──
    compute_global_correlations_and_plot(args.base_dir, out_dir, args.dpi)

    logger.info(f"\n{'='*60}")
    logger.info(f"  Output: {out_dir}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
