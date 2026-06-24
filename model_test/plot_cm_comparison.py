# ==============================================================================
# Confusion Matrix Comparison — CNN vs SAE (side-by-side)
#
# Reads CM CSV files from knn_fewshot_eval output directories.
# Shows one specific CNN seed and one specific SAE seed.
# Row-normalized to percentages, one shared colorbar on the right.
#
# Usage:
#   python -m model_test.plot_cm_comparison \
#       --results_dir /path/to/knn_fewshot_eval \
#       --cnn_seed 87 --sae_seed 856 \
#       --k 5
# ==============================================================================

import argparse
import csv
import glob
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

logger = get_logger("cm_comparison")

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42

CLASS_NAMES = ["Control", "SNCA", "GBA", "LRRK2"]
NUM_CLASSES = 4


# ==============================================================================
# Argument Parser
# ==============================================================================
def get_args():
    p = argparse.ArgumentParser(description="Confusion Matrix Comparison — CNN vs SAE")
    p.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Root directory containing CNN_seed*/SAE_sp*_seed*/ subdirs",
    )
    p.add_argument(
        "--cnn_seed",
        type=int,
        required=True,
        help="CNN encoder seed to display (e.g. 87)",
    )
    p.add_argument(
        "--sae_seed", type=int, required=True, help="SAE seed to display (e.g. 856)"
    )
    p.add_argument(
        "--k",
        type=int,
        default=5,
        help="K value to plot confusion matrix for. Default: 5",
    )
    p.add_argument(
        "--cmap",
        type=str,
        default="Blues",
        help="Colormap for confusion matrix. Default: Blues",
    )
    p.add_argument(
        "--alpha", type=float, default=1.0, help="Heatmap alpha. Default: 1.0"
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory (default: results_dir)",
    )
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument(
        "--figscale",
        type=float,
        default=1.0,
        help="Figure size multiplier. Default: 1.0",
    )
    p.add_argument(
        "--annot_fontsize",
        type=float,
        default=14,
        help="Font size for cell annotations. Default: 14",
    )
    return p.parse_args()


# ==============================================================================
# Read confusion matrix from CSV
# ==============================================================================
def read_cm_csv(csv_path):
    """Read confusion matrix CSV and return (4×4) integer count matrix."""
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            true_class = row["true_class"]
            if true_class == "TOTAL":
                continue
            i = class_to_idx.get(true_class, -1)
            if i < 0:
                continue
            for j, name in enumerate(CLASS_NAMES):
                cm[i, j] = int(float(row[f"pred_{name}"]))
    return cm


# ==============================================================================
# Load single seed confusion matrix
# ==============================================================================
def load_single_cm(results_dir, model_type, seed, k):
    """Load CM CSV for a specific model type and seed.

    Parameters
    ----------
    model_type : 'CNN' or 'SAE'
    seed : int
    k : int

    Returns
    -------
    cm : (4×4) int array
    cm_pct : (4×4) float array, row-normalized percentages (0–100)
    acc : float, overall accuracy
    """
    # Find the matching directory
    pattern = f"{model_type}*seed{seed}"
    candidates = [
        d
        for d in os.listdir(results_dir)
        if d.startswith(model_type) and d.endswith(f"seed{seed}")
    ]

    if not candidates:
        # Try broader match
        candidates = [
            d for d in os.listdir(results_dir) if model_type in d and f"seed{seed}" in d
        ]

    if not candidates:
        raise FileNotFoundError(
            f"No directory found for {model_type} seed={seed} in {results_dir}. "
            f"Available: {[d for d in os.listdir(results_dir) if d.startswith(model_type)]}"
        )

    subdir = candidates[0]
    source_label = "CNN" if model_type == "CNN" else "SAE"
    csv_path = os.path.join(results_dir, subdir, f"cm_knn_k{k}_{source_label}.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CM CSV not found: {csv_path}")

    cm = read_cm_csv(csv_path)
    acc = np.trace(cm) / cm.sum() if cm.sum() > 0 else 0.0

    # Row-normalize to percentage
    row_sums = cm.sum(axis=1, keepdims=True).astype(float)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    cm_pct = cm.astype(float) / row_sums * 100

    logger.info(
        f"  Loaded {model_type} seed={seed} k={k}: " f"total={cm.sum()}, acc={acc:.1%}"
    )
    return cm, cm_pct, acc


# ==============================================================================
# Plot side-by-side confusion matrices
# ==============================================================================
def plot_cm_comparison(
    cnn_pct,
    sae_pct,
    cnn_acc,
    sae_acc,
    cnn_seed,
    sae_seed,
    k,
    cmap,
    alpha,
    output_path,
    dpi=200,
    annot_fontsize=14,
    figscale=1.0,
):
    """Two confusion matrices side-by-side with one shared colorbar."""

    fig_w = 12 * figscale
    fig_h = 5.5 * figscale

    # Use gridspec: 2 heatmap axes + 1 thin colorbar axis
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(fig_w, fig_h),
        gridspec_kw={"width_ratios": [1, 1, 0.05], "wspace": 0.25},
    )
    ax_cnn, ax_sae, ax_cb = axes

    # Shared color range (0–100%)
    vmin, vmax = 0, 100

    datasets = [
        (ax_cnn, cnn_pct, f"CNN (seed {cnn_seed})\nAccuracy: {cnn_acc:.1%}"),
        (ax_sae, sae_pct, f"SAE (seed {sae_seed})\nAccuracy: {sae_acc:.1%}"),
    ]

    im = None
    for ax, cm_pct, title in datasets:
        im = ax.imshow(
            cm_pct,
            interpolation="nearest",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
            alpha=alpha,
        )

        # Cell annotations — percentage only
        cmap_obj = plt.get_cmap(cmap)
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                val = cm_pct[i, j]
                norm_v = val / 100.0
                bg = cmap_obj(norm_v * alpha)
                lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
                tc = "white" if lum < 0.55 else "black"

                # Bold for diagonal (correct predictions)
                weight = "bold" if i == j else "normal"
                ax.text(
                    j,
                    i,
                    f"{val:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=annot_fontsize,
                    fontweight=weight,
                    color=tc,
                )

        ax.set_xticks(range(NUM_CLASSES))
        ax.set_xticklabels(CLASS_NAMES, fontsize=11, rotation=45, ha="right")
        ax.set_yticks(range(NUM_CLASSES))
        ax.set_yticklabels(CLASS_NAMES, fontsize=11)
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

        # Grid
        for edge in range(NUM_CLASSES + 1):
            ax.axhline(edge - 0.5, color="white", linewidth=1.5)
            ax.axvline(edge - 0.5, color="white", linewidth=1.5)

    # Only left panel gets y-axis label
    ax_cnn.set_ylabel("True Label", fontsize=12)

    # Shared colorbar on the right
    cb = fig.colorbar(im, cax=ax_cb)
    cb.set_label("Classification Rate (%)", fontsize=11)

    fig.suptitle(
        f"KNN Confusion Matrix (k={k})", fontsize=15, fontweight="bold", y=1.01
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(output_path.replace(".png", ".svg"), bbox_inches="tight")
    plt.show()
    plt.close(fig)
    logger.info(f"  Saved: {output_path}")


# ==============================================================================
# Main
# ==============================================================================
def main():
    args = get_args()

    out_dir = args.output_dir if args.output_dir else args.results_dir
    os.makedirs(out_dir, exist_ok=True)

    logger.info(f"\n{'='*60}")
    logger.info(
        f"  CM Comparison — CNN seed={args.cnn_seed}, "
        f"SAE seed={args.sae_seed}, k={args.k}"
    )
    logger.info(f"{'='*60}")

    # Load specific seeds
    cnn_cm, cnn_pct, cnn_acc = load_single_cm(
        args.results_dir, "CNN", args.cnn_seed, args.k
    )
    sae_cm, sae_pct, sae_acc = load_single_cm(
        args.results_dir, "SAE", args.sae_seed, args.k
    )

    logger.info(f"\n  CNN seed={args.cnn_seed} accuracy: {cnn_acc:.1%}")
    logger.info(f"  SAE seed={args.sae_seed} accuracy: {sae_acc:.1%}")
    logger.info(f"  Δ(SAE−CNN): {(sae_acc - cnn_acc)*100:+.1f}pp")

    # Log matrices
    logger.info(f"\n  CNN confusion (%):")
    for i, name in enumerate(CLASS_NAMES):
        row = "  ".join([f"{cnn_pct[i,j]:5.1f}" for j in range(NUM_CLASSES)])
        logger.info(f"    {name:>8s}:  {row}")

    logger.info(f"\n  SAE confusion (%):")
    for i, name in enumerate(CLASS_NAMES):
        row = "  ".join([f"{sae_pct[i,j]:5.1f}" for j in range(NUM_CLASSES)])
        logger.info(f"    {name:>8s}:  {row}")

    # Plot
    output_path = os.path.join(
        out_dir, f"cm_comparison_CNN{args.cnn_seed}_SAE{args.sae_seed}_k{args.k}.png"
    )
    plot_cm_comparison(
        cnn_pct,
        sae_pct,
        cnn_acc,
        sae_acc,
        args.cnn_seed,
        args.sae_seed,
        args.k,
        cmap=args.cmap,
        alpha=args.alpha,
        output_path=output_path,
        dpi=args.dpi,
        annot_fontsize=args.annot_fontsize,
        figscale=args.figscale,
    )

    # Save CSV
    csv_path = os.path.join(
        out_dir, f"cm_comparison_CNN{args.cnn_seed}_SAE{args.sae_seed}_k{args.k}.csv"
    )
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["model", "seed", "true_class"]
            + [f"pred_{c}" for c in CLASS_NAMES]
            + ["accuracy"]
        )
        for model, seed, pct in [
            ("CNN", args.cnn_seed, cnn_pct),
            ("SAE", args.sae_seed, sae_pct),
        ]:
            for i, name in enumerate(CLASS_NAMES):
                writer.writerow(
                    [model, seed, name]
                    + [f"{pct[i,j]:.1f}" for j in range(NUM_CLASSES)]
                    + [f"{pct[i,i]:.1f}"]
                )
    logger.info(f"  Saved CSV: {csv_path}")

    logger.info(f"\n{'='*60}")
    logger.info(f"  Done!")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
