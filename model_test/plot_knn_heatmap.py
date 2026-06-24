# ==============================================================================
# KNN Accuracy Heatmap — CNN 3 layers + SAE
#
# Reads eval_results_*.json from knn_fewshot_eval output directories and
# produces heatmaps of KNN accuracy across seeds (rows) × k values (columns).
#
# Modes:
#   --mode accuracy : Two side-by-side heatmaps (CNN, SAE) with raw accuracy
#   --mode delta    : Single heatmap showing Δ = SAE − CNN_mean per k
#   --mode both     : All three panels in one figure
#
# Usage:
# !python -m model_test.plot_knn_heatmap \
#       --results_dir /content/drive/MyDrive/Final_paper/lambda_labs_moco_only/knn_fewshot_eval \
#       --mode both \
#       --metric accuracy
# ==============================================================================

import argparse
import glob
import json
import os
import re
import sys

import matplotlib
import numpy as np

_IN_COLAB = ("google.colab" in sys.modules) or os.path.isdir("/content")
if not _IN_COLAB:
    matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from sae_project.step02_logging_utils import get_logger

logger = get_logger("knn_heatmap")

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42


# ==============================================================================
# Helper: truncate colormap to use only a sub-range
# ==============================================================================
def truncate_cmap(cmap_name, lo=0.0, hi=1.0, end_color=None, n=256):
    """Return a modified colormap.

    - If end_color is given: blend from cmap(lo) to end_color (hex, e.g. '#AF7BA1').
    - Otherwise: truncate cmap to [lo, hi] range.

    Example:
        truncate_cmap('YlOrRd', 0.0, 0.5)                  → yellow-to-light-orange
        truncate_cmap('YlOrRd', end_color='#AF7BA1')        → yellow-to-muted-pink
    """
    base = plt.get_cmap(cmap_name)

    if end_color:
        # Sample the base cmap from lo up to hi for the starting portion,
        # then blend into the target end_color
        start_colors = base(np.linspace(lo, hi, n // 2))
        end_rgba = np.array(mcolors.to_rgba(end_color))
        # Last color from the sampled range
        mid_rgba = start_colors[-1]
        # Blend from mid to end_color
        blend = np.linspace(0, 1, n - n // 2)[:, None]
        end_colors = mid_rgba * (1 - blend) + end_rgba * blend
        all_colors = np.vstack([start_colors, end_colors])
        return mcolors.LinearSegmentedColormap.from_list(
            f"{cmap_name}_to_{end_color.strip('#')}", all_colors, N=n
        )

    if lo == 0.0 and hi == 1.0:
        return base
    colors = base(np.linspace(lo, hi, n))
    return mcolors.LinearSegmentedColormap.from_list(
        f"{cmap_name}_{lo:.2f}_{hi:.2f}", colors, N=n
    )


# ==============================================================================
# Argument Parser
# ==============================================================================
def get_args():
    p = argparse.ArgumentParser(
        description="KNN Accuracy Heatmap — CNN vs SAE comparison"
    )
    p.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Root directory containing CNN_seed*/SAE_sp*_seed*/ subdirs",
    )
    p.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["accuracy", "delta", "both", "compact"],
        help="Heatmap mode: 'accuracy' (side-by-side CNN/SAE), "
        "'delta' (SAE − CNN_mean), 'both', "
        "'compact' (3-row paper-ready: CNN mean, SAE mean, Δ). "
        "Default: both",
    )
    p.add_argument(
        "--metric",
        type=str,
        default="accuracy",
        choices=["accuracy", "macro_f1"],
        help="Metric to plot. Default: accuracy",
    )
    p.add_argument(
        "--cmap_acc",
        type=str,
        default="YlGnBu",
        help="Colormap for accuracy heatmaps. Default: YlGnBu",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Heatmap alpha (transparency). Default: 1.0",
    )
    p.add_argument(
        "--cmap_lo",
        type=float,
        default=0.0,
        help="Colormap lower bound (0.0–1.0). Default: 0.0",
    )
    p.add_argument(
        "--cmap_hi",
        type=float,
        default=1.0,
        help="Colormap upper bound (0.0–1.0). Default: 1.0",
    )
    p.add_argument(
        "--cmap_end_color",
        type=str,
        default="",
        help="Override cmap end color with hex code (e.g. '#AF7BA1'). "
        "Start follows cmap_acc from cmap_lo, end blends to this color. "
        "Default: '' (use cmap as-is)",
    )
    p.add_argument(
        "--cmap_delta",
        type=str,
        default="RdBu_r",
        help="Colormap for delta heatmap (diverging). Default: RdBu_r",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory (default: results_dir)",
    )
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument(
        "--vmin",
        type=float,
        default=0.0,
        help="Min value for accuracy colorbar (default: 0.0 = auto)",
    )
    p.add_argument(
        "--vmax",
        type=float,
        default=0.0,
        help="Max value for accuracy colorbar (default: 0.0 = auto)",
    )
    p.add_argument(
        "--annot_fontsize",
        type=float,
        default=11,
        help="Font size for cell annotations. Default: 11",
    )
    p.add_argument(
        "--figscale",
        type=float,
        default=1.0,
        help="Figure size multiplier. Default: 1.0",
    )
    return p.parse_args()


# ==============================================================================
# Scan results directory and collect data
# ==============================================================================
CNN_LAYER_ORDER = ["stage5_mid", "stage5_out", "refine_out"]


def collect_results(results_dir, metric="accuracy"):
    """Scan result directories and build data tables.

    Supports directory names:
      CNN_stage5_mid_seed42, CNN_stage5_out_seed42, CNN_refine_out_seed42
      CNN_seed42 (legacy, treated as 'stage5_out')
      SAE_sp800_seed48

    Returns
    -------
    layer_data : dict  {layer_name: {seed: {k: value}}}
                 e.g. {'stage5_mid': {42: {1: 0.85, ...}}, ...}
    sae_data   : dict  {seed: {k: value}}
    k_values   : sorted list of k values
    """
    layer_data = {}  # layer -> {seed: {k: val}}
    sae_data = {}
    k_values_set = set()

    for subdir in sorted(os.listdir(results_dir)):
        subdir_path = os.path.join(results_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue

        # Find JSON files
        json_files = glob.glob(os.path.join(subdir_path, "eval_results_*.json"))
        if not json_files:
            continue

        for jf in json_files:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)

            knn_results = data.get("knn", [])
            if not knn_results:
                continue

            # Parse seed from directory name
            seed_match = re.search(r"seed(\d+)$", subdir)
            if not seed_match:
                continue
            seed = int(seed_match.group(1))

            # Determine model type and layer from directory name
            if subdir.startswith("CNN"):
                # Try to extract layer: CNN_stage5_out_seed42
                layer_match = re.match(
                    r"CNN_(stage5_mid|stage5_out|refine_out)_seed\d+$", subdir
                )
                if layer_match:
                    layer = layer_match.group(1)
                else:
                    # Legacy: CNN_seed42 → treat as stage5_out
                    layer = "stage5_out"

                if layer not in layer_data:
                    layer_data[layer] = {}
                layer_data[layer][seed] = {}

                for r in knn_results:
                    k = r["k"]
                    k_values_set.add(k)
                    layer_data[layer][seed][k] = (
                        r["accuracy"] if metric == "accuracy" else r["macro_f1"]
                    )
                logger.info(
                    f"  Loaded CNN {layer} seed={seed}: " f"{len(knn_results)} k values"
                )

            elif subdir.startswith("SAE"):
                sae_data[seed] = {}
                for r in knn_results:
                    k = r["k"]
                    k_values_set.add(k)
                    sae_data[seed][k] = (
                        r["accuracy"] if metric == "accuracy" else r["macro_f1"]
                    )
                logger.info(
                    f"  Loaded SAE seed={seed}: " f"{len(knn_results)} k values"
                )

    k_values = sorted(k_values_set)

    # Sort layers in canonical order
    sorted_layers = [l for l in CNN_LAYER_ORDER if l in layer_data]
    # Add any extra layers not in the canonical list
    for l in sorted(layer_data.keys()):
        if l not in sorted_layers:
            sorted_layers.append(l)
    layer_data = {l: layer_data[l] for l in sorted_layers}

    for layer, ld in layer_data.items():
        logger.info(f"  CNN {layer} seeds: {sorted(ld.keys())}")
    logger.info(f"  SAE seeds: {sorted(sae_data.keys())}")
    logger.info(f"  K values : {k_values}")

    return layer_data, sae_data, k_values


# ==============================================================================
# Build matrix from data dict
# ==============================================================================
def build_matrix(data, seeds, k_values):
    """Build (n_seeds × n_k) numpy matrix from data dict."""
    mat = np.full((len(seeds), len(k_values)), np.nan)
    for i, seed in enumerate(seeds):
        for j, k in enumerate(k_values):
            if seed in data and k in data[seed]:
                mat[i, j] = data[seed][k]
    return mat


# ==============================================================================
# Draw single heatmap on an axis
# ==============================================================================
def draw_heatmap(
    ax,
    mat,
    row_labels,
    col_labels,
    title,
    alpha,
    cmap="YlGnBu",
    vmin=None,
    vmax=None,
    fmt=".1%",
    annot_fontsize=11,
    is_delta=False,
):
    """Draw an annotated heatmap on the given axis."""

    if vmin is None:
        vmin = np.nanmin(mat)
    if vmax is None:
        vmax = np.nanmax(mat)

    # For delta mode, center colormap at 0
    if is_delta:
        abs_max = max(abs(np.nanmin(mat)), abs(np.nanmax(mat)), 0.01)
        vmin, vmax = -abs_max, abs_max

    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", alpha=alpha)

    # Annotate each cell
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isnan(val):
                ax.text(
                    j,
                    i,
                    "—",
                    ha="center",
                    va="center",
                    fontsize=annot_fontsize,
                    color="gray",
                )
                continue

            # Choose text color based on background brightness
            norm_val = (val - vmin) / max(vmax - vmin, 1e-12)
            # Get the RGB color at this position in the colormap
            cmap_obj = plt.get_cmap(cmap)
            bg_color = cmap_obj(norm_val)
            # Calculate luminance
            luminance = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
            text_color = "white" if luminance < 0.55 else "black"

            if is_delta:
                # Show delta as percentage points with sign
                text = f"{val*100:+.1f}"
            else:
                text = f"{val:.1%}"

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=annot_fontsize,
                color=text_color,
                fontweight="bold" if is_delta and abs(val) > 0.02 else "normal",
            )

    # Axis labels
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels([str(k) for k in col_labels], fontsize=11)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels([str(s) for s in row_labels], fontsize=11)
    ax.set_xlabel("k (neighbors)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)

    # Grid lines between cells
    for i in range(mat.shape[0] + 1):
        ax.axhline(i - 0.5, color="white", linewidth=1.5)
    for j in range(mat.shape[1] + 1):
        ax.axvline(j - 0.5, color="white", linewidth=1.5)

    return im


# ==============================================================================
# Plot: Accuracy heatmaps (side-by-side CNN / SAE)
# ==============================================================================
def plot_accuracy_heatmaps(
    cnn_mat,
    sae_mat,
    cnn_seeds,
    sae_seeds,
    k_values,
    metric_name,
    cmap,
    output_path,
    dpi=200,
    alpha=1.0,
    vmin=None,
    vmax=None,
    annot_fontsize=11,
    figscale=1.0,
):
    """Two side-by-side heatmaps: CNN and SAE."""

    # Determine shared color range
    all_vals = np.concatenate([cnn_mat.ravel(), sae_mat.ravel()])
    all_vals = all_vals[~np.isnan(all_vals)]
    if vmin is None:
        vmin = max(all_vals.min() - 0.02, 0)
    if vmax is None:
        vmax = min(all_vals.max() + 0.02, 1)

    n_rows_max = max(len(cnn_seeds), len(sae_seeds))
    fig_w = (len(k_values) * 1.2 + 2) * 2 * figscale
    fig_h = (n_rows_max * 0.7 + 2.5) * figscale
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h))

    # CNN
    im1 = draw_heatmap(
        axes[0],
        cnn_mat,
        [f"seed {s}" for s in cnn_seeds],
        k_values,
        f"CNN — KNN {metric_name.title()}",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        annot_fontsize=annot_fontsize,
        alpha=alpha,
    )
    axes[0].set_ylabel("CNN Encoder Seed", fontsize=12)

    # SAE
    im2 = draw_heatmap(
        axes[1],
        sae_mat,
        [f"seed {s}" for s in sae_seeds],
        k_values,
        f"SAE — KNN {metric_name.title()}",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        annot_fontsize=annot_fontsize,
        alpha=alpha,
    )
    axes[1].set_ylabel("SAE Seed", fontsize=12)

    # Shared colorbar
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cb = fig.colorbar(im2, cax=cbar_ax)
    cb.set_label(metric_name.title(), fontsize=12)

    # Mean ± std annotation at bottom
    cnn_mean = np.nanmean(cnn_mat, axis=0)
    sae_mean = np.nanmean(sae_mat, axis=0)
    cnn_overall = np.nanmean(cnn_mat)
    sae_overall = np.nanmean(sae_mat)
    fig.text(
        0.5,
        0.01,
        f"CNN mean: {cnn_overall:.1%}  |  SAE mean: {sae_overall:.1%}  |  "
        f"Δ(SAE−CNN): {(sae_overall - cnn_overall)*100:+.1f}pp",
        ha="center",
        fontsize=11,
        style="italic",
        color="#333333",
    )

    fig.tight_layout(rect=[0, 0.04, 0.88, 1.0])
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(output_path.replace(".png", ".svg"), bbox_inches="tight")
    if _IN_COLAB:
        plt.show()
    plt.close(fig)
    logger.info(f"  Saved accuracy heatmap: {output_path}")


# ==============================================================================
# Plot: Delta heatmap (SAE − CNN_mean)
# ==============================================================================
def plot_delta_heatmap(
    cnn_mat,
    sae_mat,
    sae_seeds,
    k_values,
    metric_name,
    cmap,
    output_path,
    dpi=200,
    alpha=1.0,
    annot_fontsize=11,
    figscale=1.0,
):
    """Single heatmap: Δ = SAE_seed_k − CNN_mean_k for each cell."""

    # CNN mean per k (across all CNN seeds)
    cnn_mean_per_k = np.nanmean(cnn_mat, axis=0)  # (n_k,)

    # Delta matrix: each SAE seed vs CNN mean
    delta_mat = sae_mat - cnn_mean_per_k[np.newaxis, :]

    fig_w = (len(k_values) * 1.3 + 3.5) * figscale
    fig_h = (len(sae_seeds) * 0.8 + 3) * figscale
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = draw_heatmap(
        ax,
        delta_mat,
        [f"SAE seed {s}" for s in sae_seeds],
        k_values,
        f"Δ{metric_name.title()} (SAE − CNN mean)",
        alpha=alpha,
        cmap=cmap,
        is_delta=True,
        annot_fontsize=annot_fontsize,
    )
    ax.set_ylabel("SAE Seed", fontsize=12)

    # Colorbar
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.08)
    cb.set_label(f"Δ {metric_name.title()} (pp)", fontsize=11)

    # Annotation: CNN baseline
    cnn_str = "  ".join([f"k={k}: {v:.1%}" for k, v in zip(k_values, cnn_mean_per_k)])
    fig.text(
        0.5,
        -0.01,
        f"CNN mean baseline — {cnn_str}",
        ha="center",
        fontsize=9,
        style="italic",
        color="#555555",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(output_path.replace(".png", ".svg"), bbox_inches="tight")
    if _IN_COLAB:
        plt.show()
    plt.close(fig)
    logger.info(f"  Saved delta heatmap: {output_path}")


# ==============================================================================
# Plot: Combined (accuracy + delta in one figure)
# ==============================================================================
def plot_combined(
    cnn_mat,
    sae_mat,
    cnn_seeds,
    sae_seeds,
    k_values,
    metric_name,
    cmap_acc,
    cmap_delta,
    output_path,
    dpi=200,
    alpha=1.0,
    vmin=None,
    vmax=None,
    annot_fontsize=11,
    figscale=1.0,
):
    """Three panels: CNN accuracy, SAE accuracy, Delta."""

    # Shared color range for accuracy panels
    all_vals = np.concatenate([cnn_mat.ravel(), sae_mat.ravel()])
    all_vals = all_vals[~np.isnan(all_vals)]
    if vmin is None:
        vmin = max(all_vals.min() - 0.02, 0)
    if vmax is None:
        vmax = min(all_vals.max() + 0.02, 1)

    # Delta
    cnn_mean_per_k = np.nanmean(cnn_mat, axis=0)
    delta_mat = sae_mat - cnn_mean_per_k[np.newaxis, :]

    n_rows_max = max(len(cnn_seeds), len(sae_seeds))
    fig_w = (len(k_values) * 1.1 + 1.8) * 3 * figscale
    fig_h = (n_rows_max * 0.7 + 3) * figscale
    fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h))

    # CNN accuracy
    im1 = draw_heatmap(
        axes[0],
        cnn_mat,
        [f"seed {s}" for s in cnn_seeds],
        k_values,
        f"CNN — {metric_name.title()}",
        alpha=alpha,
        cmap=cmap_acc,
        vmin=vmin,
        vmax=vmax,
        annot_fontsize=annot_fontsize,
    )
    axes[0].set_ylabel("CNN Encoder Seed", fontsize=12)

    # SAE accuracy
    im2 = draw_heatmap(
        axes[1],
        sae_mat,
        [f"seed {s}" for s in sae_seeds],
        k_values,
        f"SAE — {metric_name.title()}",
        alpha=alpha,
        cmap=cmap_acc,
        vmin=vmin,
        vmax=vmax,
        annot_fontsize=annot_fontsize,
    )
    axes[1].set_ylabel("SAE Seed", fontsize=12)

    # Delta
    im3 = draw_heatmap(
        axes[2],
        delta_mat,
        [f"seed {s}" for s in sae_seeds],
        k_values,
        f"Δ (SAE − CNN mean)",
        alpha=alpha,
        cmap=cmap_delta,
        is_delta=True,
        annot_fontsize=annot_fontsize,
    )
    axes[2].set_ylabel("SAE Seed", fontsize=12)

    # Colorbars
    # Accuracy colorbar (shared for panels 0, 1)
    cb1 = fig.colorbar(
        im2, ax=axes[:2], shrink=0.7, pad=0.03, location="bottom", aspect=30
    )
    cb1.set_label(metric_name.title(), fontsize=11)

    # Delta colorbar
    cb2 = fig.colorbar(
        im3, ax=axes[2], shrink=0.7, pad=0.03, location="bottom", aspect=15
    )
    cb2.set_label(f"Δ (pp)", fontsize=11)

    # Overall summary
    cnn_overall = np.nanmean(cnn_mat)
    sae_overall = np.nanmean(sae_mat)
    delta_overall = sae_overall - cnn_overall
    fig.suptitle(
        f"KNN Evaluation — CNN vs SAE\n"
        f"CNN: {cnn_overall:.1%}  |  SAE: {sae_overall:.1%}  |  "
        f"Δ: {delta_overall*100:+.1f}pp",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(output_path.replace(".png", ".svg"), bbox_inches="tight")
    if _IN_COLAB:
        plt.show()
    plt.close(fig)
    logger.info(f"  Saved combined heatmap: {output_path}")


# ==============================================================================
# Plot: Compact paper-ready heatmap — 3 CNN layers + SAE + Δ per layer
# ==============================================================================
def plot_compact(
    layer_mats,
    sae_mat,
    k_values,
    metric_name,
    cmap_acc,
    cmap_delta,
    output_path,
    dpi=200,
    alpha=1.0,
    vmin=None,
    vmax=None,
    annot_fontsize=12,
    figscale=1.0,
):
    """Compact heatmap for paper.

    Top rows: mean(±std) for each CNN layer + SAE
    Bottom rows: Δ = SAE_mean − CNN_layer_mean (one per layer)
    """
    n_k = len(k_values)
    layers = list(layer_mats.keys())  # ordered

    # Compute means/stds
    layer_means = {}
    layer_stds = {}
    layer_n = {}
    for layer, mat in layer_mats.items():
        layer_means[layer] = np.nanmean(mat, axis=0)
        layer_stds[layer] = np.nanstd(mat, axis=0)
        layer_n[layer] = int(np.sum(~np.isnan(mat[:, 0])))

    sae_mean = np.nanmean(sae_mat, axis=0)
    sae_std = np.nanstd(sae_mat, axis=0)
    n_sae = int(np.sum(~np.isnan(sae_mat[:, 0])))

    # Accuracy matrix: CNN layers + SAE
    n_acc_rows = len(layers) + 1  # CNN layers + SAE
    acc_mat = np.vstack([layer_means[l] for l in layers] + [sae_mean])
    stds = [layer_stds[l] for l in layers] + [sae_std]

    # Delta matrix: SAE - each CNN layer
    n_delta_rows = len(layers)
    delta_mat = np.vstack([sae_mean - layer_means[l] for l in layers])

    # Row labels
    LAYER_DISPLAY = {
        "stage5_mid": "CNN stage5_mid",
        "stage5_out": "CNN stage5_out",
        "refine_out": "CNN refine_out",
    }
    acc_labels = [f"{LAYER_DISPLAY.get(l, l)} (n={layer_n[l]})" for l in layers] + [
        f"SAE (n={n_sae})"
    ]
    delta_labels = [f"Δ vs {LAYER_DISPLAY.get(l, l)[:12]}" for l in layers]

    # ── Figure sizing ──
    cell_w = 1.4 * figscale
    cell_h = 0.9 * figscale
    fig_w = n_k * cell_w + 4.5 * figscale
    fig_h = (n_acc_rows + n_delta_rows) * cell_h + 3 * figscale

    fig, (ax_acc, ax_delta) = plt.subplots(
        2,
        1,
        figsize=(fig_w, fig_h),
        gridspec_kw={"height_ratios": [n_acc_rows, n_delta_rows], "hspace": 0.08},
    )

    # ── Color range ──
    all_acc = acc_mat.ravel()
    all_acc = all_acc[~np.isnan(all_acc)]
    if vmin is None:
        vmin_acc = max(all_acc.min() - 0.02, 0)
    else:
        vmin_acc = vmin
    if vmax is None:
        vmax_acc = min(all_acc.max() + 0.02, 1)
    else:
        vmax_acc = vmax

    # ── Accuracy heatmap ──
    im_acc = ax_acc.imshow(
        acc_mat, cmap=cmap_acc, vmin=vmin_acc, vmax=vmax_acc, aspect="auto", alpha=alpha
    )

    cmap_obj_acc = cmap_acc if not isinstance(cmap_acc, str) else plt.get_cmap(cmap_acc)
    for i in range(n_acc_rows):
        for j in range(n_k):
            val = acc_mat[i, j]
            std_val = stds[i][j]
            if np.isnan(val):
                continue
            norm_v = (val - vmin_acc) / max(vmax_acc - vmin_acc, 1e-12)
            bg = cmap_obj_acc(norm_v)
            lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
            tc = "white" if lum < 0.55 else "black"
            tc_sub = "#dddddd" if lum < 0.55 else "#555555"

            ax_acc.text(
                j,
                i - 0.08,
                f"{val:.1%}",
                ha="center",
                va="center",
                fontsize=annot_fontsize,
                fontweight="bold",
                color=tc,
            )
            ax_acc.text(
                j,
                i + 0.28,
                f"±{std_val:.1%}",
                ha="center",
                va="center",
                fontsize=annot_fontsize - 3,
                color=tc_sub,
            )

    ax_acc.set_xticks(range(n_k))
    ax_acc.set_xticklabels([])
    ax_acc.set_yticks(range(n_acc_rows))
    ax_acc.set_yticklabels(acc_labels, fontsize=11, fontweight="bold")
    ax_acc.tick_params(bottom=False)

    for i in range(n_acc_rows + 1):
        lw = 3.0 if i == len(layers) else 1.5  # thicker line before SAE
        ax_acc.axhline(i - 0.5, color="white", linewidth=lw)
    for j in range(n_k + 1):
        ax_acc.axvline(j - 0.5, color="white", linewidth=1.5)

    # ── Delta heatmap ──
    abs_max = max(np.abs(delta_mat).max(), 0.005)
    im_delta = ax_delta.imshow(
        delta_mat,
        cmap=cmap_delta,
        vmin=-abs_max,
        vmax=abs_max,
        aspect="auto",
        alpha=alpha,
    )

    cmap_obj_delta = plt.get_cmap(cmap_delta)
    for i in range(n_delta_rows):
        for j in range(n_k):
            val = delta_mat[i, j]
            norm_v = (val + abs_max) / max(2 * abs_max, 1e-12)
            bg = cmap_obj_delta(norm_v)
            lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
            tc = "white" if lum < 0.55 else "black"
            sign = "+" if val > 0 else ""
            ax_delta.text(
                j,
                i,
                f"{sign}{val*100:.1f}pp",
                ha="center",
                va="center",
                fontsize=annot_fontsize,
                fontweight="bold",
                color=tc,
            )

    ax_delta.set_xticks(range(n_k))
    ax_delta.set_xticklabels([f"k={k}" for k in k_values], fontsize=11)
    ax_delta.set_yticks(range(n_delta_rows))
    ax_delta.set_yticklabels(delta_labels, fontsize=11, fontweight="bold")
    ax_delta.set_xlabel("Number of Neighbors (k)", fontsize=12)

    for i in range(n_delta_rows + 1):
        ax_delta.axhline(i - 0.5, color="white", linewidth=1.5)
    for j in range(n_k + 1):
        ax_delta.axvline(j - 0.5, color="white", linewidth=1.5)

    # ── Colorbars ──
    cb_acc = fig.colorbar(im_acc, ax=ax_acc, shrink=0.85, pad=0.03)
    cb_acc.set_label(metric_name.title(), fontsize=10)
    cb_delta = fig.colorbar(im_delta, ax=ax_delta, shrink=0.85, pad=0.03)
    cb_delta.set_label("Δ (pp)", fontsize=10)

    fig.suptitle(
        f"KNN Classification — {metric_name.title()}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(output_path.replace(".png", ".svg"), bbox_inches="tight")
    if _IN_COLAB:
        plt.show()
    plt.close(fig)
    logger.info(f"  Saved compact heatmap: {output_path}")


# ==============================================================================
# Plot: Summary table — rows = k, columns = layers + SAE (mean ± std)
# ==============================================================================
def plot_summary_table(
    layer_mats,
    sae_mat,
    k_values,
    metric_name,
    cmap_acc,
    output_path,
    dpi=200,
    alpha=1.0,
    vmin=None,
    vmax=None,
    annot_fontsize=13,
    figscale=1.0,
):
    """Summary table: rows = k values, columns = CNN layers + SAE.
    Each cell shows mean +/- std across seeds.
    """
    layers = list(layer_mats.keys())
    n_k = len(k_values)
    n_cols = len(layers) + 1  # +1 for SAE

    LAYER_DISPLAY = {
        "stage5_mid": "CNN\nstage5_mid",
        "stage5_out": "CNN\nstage5_out",
        "refine_out": "CNN\nrefine_out",
    }

    # Build mean / std arrays
    means = []
    stds = []
    col_labels = []
    for layer in layers:
        mat = layer_mats[layer]
        means.append(np.nanmean(mat, axis=0))
        stds.append(np.nanstd(mat, axis=0))
        n = int(np.sum(~np.isnan(mat[:, 0])))
        col_labels.append(f"{LAYER_DISPLAY.get(layer, layer)}\n(n={n})")

    means.append(np.nanmean(sae_mat, axis=0))
    stds.append(np.nanstd(sae_mat, axis=0))
    n_sae = int(np.sum(~np.isnan(sae_mat[:, 0])))
    col_labels.append(f"SAE\n(n={n_sae})")

    # (n_cols, n_k) → transpose to (n_k, n_cols)
    mean_mat = np.vstack(means).T
    std_mat = np.vstack(stds).T

    # ── Figure ──
    cell_w = 2.5 * figscale
    cell_h = 1.0 * figscale
    fig_w = n_cols * cell_w + 3 * figscale
    fig_h = n_k * cell_h + 3 * figscale
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Color range
    all_vals = mean_mat[~np.isnan(mean_mat)]
    vmin_use = vmin if vmin is not None else max(all_vals.min() - 0.02, 0)
    vmax_use = vmax if vmax is not None else min(all_vals.max() + 0.02, 1)

    im = ax.imshow(
        mean_mat,
        cmap=cmap_acc,
        vmin=vmin_use,
        vmax=vmax_use,
        aspect="auto",
        alpha=alpha,
    )

    cmap_obj = cmap_acc if not isinstance(cmap_acc, str) else plt.get_cmap(cmap_acc)
    for i in range(n_k):
        for j in range(n_cols):
            val = mean_mat[i, j]
            std_val = std_mat[i, j]
            if np.isnan(val):
                continue
            norm_v = (val - vmin_use) / max(vmax_use - vmin_use, 1e-12)
            bg = cmap_obj(norm_v)
            lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
            tc = "white" if lum < 0.55 else "black"
            tc_sub = "#dddddd" if lum < 0.55 else "#555555"

            ax.text(
                j,
                i - 0.12,
                f"{val:.1%}",
                ha="center",
                va="center",
                fontsize=annot_fontsize,
                fontweight="bold",
                color=tc,
            )
            ax.text(
                j,
                i + 0.25,
                f"±{std_val:.1%}",
                ha="center",
                va="center",
                fontsize=annot_fontsize - 3,
                color=tc_sub,
            )

    # Axis
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=11, fontweight="bold")
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.set_yticks(range(n_k))
    ax.set_yticklabels([f"k = {k}" for k in k_values], fontsize=12, fontweight="bold")

    # Grid
    for i in range(n_k + 1):
        ax.axhline(i - 0.5, color="white", linewidth=2)
    for j in range(n_cols + 1):
        lw = 3.0 if j == len(layers) else 2.0
        ax.axvline(j - 0.5, color="white", linewidth=lw)

    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.05)
    cb.set_label(metric_name.title(), fontsize=11)

    ax.set_title(
        f"KNN {metric_name.title()} — Summary (mean ± std)",
        fontsize=14,
        fontweight="bold",
        pad=50,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(output_path.replace(".png", ".svg"), bbox_inches="tight")
    if _IN_COLAB:
        plt.show()
    plt.close(fig)
    logger.info(f"  Saved summary table: {output_path}")


# ==============================================================================
# Plot: All seeds detail — one big heatmap, grouped by layer
# ==============================================================================
def plot_all_seeds_detail(
    layer_data,
    layer_mats,
    sae_mat,
    sae_seeds,
    k_values,
    metric_name,
    cmap_acc,
    output_path,
    dpi=200,
    alpha=1.0,
    vmin=None,
    vmax=None,
    annot_fontsize=10,
    figscale=1.0,
):
    """All seeds × all layers in one heatmap.
    Rows: seeds grouped by layer.  Cols: k values.
    """
    layers = list(layer_mats.keys())

    LAYER_DISPLAY = {
        "stage5_mid": "CNN stage5_mid",
        "stage5_out": "CNN stage5_out",
        "refine_out": "CNN refine_out",
    }

    all_mats = []
    row_labels = []
    group_boundaries = []  # row indices where each new group starts

    for layer in layers:
        seeds = sorted(layer_data[layer].keys())
        mat = layer_mats[layer]
        group_boundaries.append(len(row_labels))
        for seed in seeds:
            row_labels.append(f"{LAYER_DISPLAY.get(layer, layer)}  s{seed}")
        all_mats.append(mat)

    # SAE
    group_boundaries.append(len(row_labels))
    for seed in sae_seeds:
        row_labels.append(f"SAE  s{seed}")
    all_mats.append(sae_mat)

    full_mat = np.vstack(all_mats)  # (total_rows, n_k)
    n_rows = full_mat.shape[0]
    n_k = len(k_values)

    # ── Figure ──
    cell_w = 1.3 * figscale
    cell_h = 0.6 * figscale
    fig_w = n_k * cell_w + 5.5 * figscale
    fig_h = n_rows * cell_h + 3 * figscale
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Color range
    all_vals = full_mat[~np.isnan(full_mat)]
    vmin_use = vmin if vmin is not None else max(all_vals.min() - 0.02, 0)
    vmax_use = vmax if vmax is not None else min(all_vals.max() + 0.02, 1)

    im = ax.imshow(
        full_mat,
        cmap=cmap_acc,
        vmin=vmin_use,
        vmax=vmax_use,
        aspect="auto",
        alpha=alpha,
    )

    cmap_obj = cmap_acc if not isinstance(cmap_acc, str) else plt.get_cmap(cmap_acc)
    for i in range(n_rows):
        for j in range(n_k):
            val = full_mat[i, j]
            if np.isnan(val):
                ax.text(
                    j,
                    i,
                    "—",
                    ha="center",
                    va="center",
                    fontsize=annot_fontsize,
                    color="gray",
                )
                continue
            norm_v = (val - vmin_use) / max(vmax_use - vmin_use, 1e-12)
            bg = cmap_obj(norm_v)
            lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
            tc = "white" if lum < 0.55 else "black"
            ax.text(
                j,
                i,
                f"{val:.1%}",
                ha="center",
                va="center",
                fontsize=annot_fontsize,
                fontweight="bold",
                color=tc,
            )

    # Axis
    ax.set_xticks(range(n_k))
    ax.set_xticklabels([f"k={k}" for k in k_values], fontsize=11)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xlabel("Number of Neighbors (k)", fontsize=12)

    # Grid — thick lines at group boundaries
    for i in range(n_rows + 1):
        lw = 3.5 if i in group_boundaries else 1.5
        ax.axhline(i - 0.5, color="white", linewidth=lw)
    for j in range(n_k + 1):
        ax.axvline(j - 0.5, color="white", linewidth=1.5)

    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.04)
    cb.set_label(metric_name.title(), fontsize=11)

    ax.set_title(
        f"KNN {metric_name.title()} — All Seeds", fontsize=14, fontweight="bold", pad=10
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(output_path.replace(".png", ".svg"), bbox_inches="tight")
    if _IN_COLAB:
        plt.show()
    plt.close(fig)
    logger.info(f"  Saved all-seeds detail heatmap: {output_path}")


# ==============================================================================
# Main
# ==============================================================================
def main():
    args = get_args()

    out_dir = args.output_dir if args.output_dir else args.results_dir
    os.makedirs(out_dir, exist_ok=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"  KNN Heatmap — Scanning: {args.results_dir}")
    logger.info(f"{'='*60}")

    # Collect results (new: layer_data instead of flat cnn_data)
    layer_data, sae_data, k_values = collect_results(args.results_dir, args.metric)

    if not layer_data:
        logger.error("No CNN results found!")
        return
    if not sae_data:
        logger.error("No SAE results found!")
        return

    sae_seeds = sorted(sae_data.keys())
    sae_mat = build_matrix(sae_data, sae_seeds, k_values)

    # Build per-layer matrices
    layer_mats = {}  # layer -> (mat, seeds)
    for layer, ld in layer_data.items():
        seeds = sorted(ld.keys())
        mat = build_matrix(ld, seeds, k_values)
        layer_mats[layer] = mat
        logger.info(f"\n  ── CNN {layer} Accuracy ──")
        for i, seed in enumerate(seeds):
            vals = "  ".join(
                [
                    f"k={k}: {mat[i,j]:.1%}" if not np.isnan(mat[i, j]) else f"k={k}: —"
                    for j, k in enumerate(k_values)
                ]
            )
            logger.info(f"    seed {seed:>4d}  {vals}")

    logger.info(f"\n  ── SAE Accuracy ──")
    for i, seed in enumerate(sae_seeds):
        vals = "  ".join(
            [
                (
                    f"k={k}: {sae_mat[i,j]:.1%}"
                    if not np.isnan(sae_mat[i, j])
                    else f"k={k}: —"
                )
                for j, k in enumerate(k_values)
            ]
        )
        logger.info(f"    seed {seed:>4d}  {vals}")

    # Determine vmin/vmax
    vmin = args.vmin if args.vmin > 0 else None
    vmax = args.vmax if args.vmax > 0 else None

    # Truncate colormaps if requested
    cmap_acc = truncate_cmap(
        args.cmap_acc, args.cmap_lo, args.cmap_hi, end_color=args.cmap_end_color or None
    )
    cmap_delta_name = args.cmap_delta

    metric_name = args.metric

    # For backward compat: flatten all CNN layers into one matrix for
    # accuracy/delta/both modes (using stage5_out if available, else first)
    primary_layer = (
        "stage5_out" if "stage5_out" in layer_mats else list(layer_mats.keys())[0]
    )
    cnn_mat = layer_mats[primary_layer]
    cnn_seeds = sorted(layer_data[primary_layer].keys())

    if args.mode in ("accuracy", "both"):
        acc_path = os.path.join(out_dir, f"heatmap_knn_{metric_name}.png")
        plot_accuracy_heatmaps(
            cnn_mat,
            sae_mat,
            cnn_seeds,
            sae_seeds,
            k_values,
            metric_name,
            cmap_acc,
            acc_path,
            args.dpi,
            alpha=args.alpha,
            vmin=vmin,
            vmax=vmax,
            annot_fontsize=args.annot_fontsize,
            figscale=args.figscale,
        )

    if args.mode in ("delta", "both"):
        delta_path = os.path.join(out_dir, f"heatmap_knn_delta_{metric_name}.png")
        plot_delta_heatmap(
            cnn_mat,
            sae_mat,
            sae_seeds,
            k_values,
            metric_name,
            args.cmap_delta,
            delta_path,
            args.dpi,
            alpha=args.alpha,
            annot_fontsize=args.annot_fontsize,
            figscale=args.figscale,
        )

    if args.mode == "both":
        combo_path = os.path.join(out_dir, f"heatmap_knn_combined_{metric_name}.png")
        plot_combined(
            cnn_mat,
            sae_mat,
            cnn_seeds,
            sae_seeds,
            k_values,
            metric_name,
            cmap_acc,
            args.cmap_delta,
            combo_path,
            args.dpi,
            alpha=args.alpha,
            vmin=vmin,
            vmax=vmax,
            annot_fontsize=args.annot_fontsize,
            figscale=args.figscale,
        )

    if args.mode == "compact":
        compact_path = os.path.join(out_dir, f"heatmap_knn_compact_{metric_name}.png")
        plot_compact(
            layer_mats,
            sae_mat,
            k_values,
            metric_name,
            cmap_acc,
            args.cmap_delta,
            compact_path,
            args.dpi,
            alpha=args.alpha,
            vmin=vmin,
            vmax=vmax,
            annot_fontsize=args.annot_fontsize,
            figscale=args.figscale,
        )

        # --- Summary table: k-rows × layer-columns (mean ± std) ---
        summary_path = os.path.join(out_dir, f"heatmap_knn_summary_{metric_name}.png")
        plot_summary_table(
            layer_mats,
            sae_mat,
            k_values,
            metric_name,
            cmap_acc,
            summary_path,
            args.dpi,
            alpha=args.alpha,
            vmin=vmin,
            vmax=vmax,
            annot_fontsize=args.annot_fontsize,
            figscale=args.figscale,
        )

        # --- All-seeds detail: every seed for every layer ---
        detail_path = os.path.join(out_dir, f"heatmap_knn_all_seeds_{metric_name}.png")
        plot_all_seeds_detail(
            layer_data,
            layer_mats,
            sae_mat,
            sae_seeds,
            k_values,
            metric_name,
            cmap_acc,
            detail_path,
            args.dpi,
            alpha=args.alpha,
            vmin=vmin,
            vmax=vmax,
            annot_fontsize=args.annot_fontsize,
            figscale=args.figscale,
        )

    # Save raw data as CSV
    csv_path = os.path.join(out_dir, f"knn_heatmap_data_{metric_name}.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        header = ["model", "layer", "seed"] + [f"k={k}" for k in k_values]
        f.write(",".join(header) + "\n")
        for layer, ld in layer_data.items():
            seeds = sorted(ld.keys())
            mat = layer_mats[layer]
            for i, seed in enumerate(seeds):
                vals = [
                    f"{mat[i,j]:.4f}" if not np.isnan(mat[i, j]) else ""
                    for j in range(len(k_values))
                ]
                f.write(f"CNN,{layer},{seed}," + ",".join(vals) + "\n")
        for i, seed in enumerate(sae_seeds):
            vals = [
                f"{sae_mat[i,j]:.4f}" if not np.isnan(sae_mat[i, j]) else ""
                for j in range(len(k_values))
            ]
            f.write(f"SAE,stage5_out,{seed}," + ",".join(vals) + "\n")
    logger.info(f"\n  Saved CSV: {csv_path}")

    logger.info(f"\n{'='*60}")
    logger.info(f"  Heatmap generation complete!")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
