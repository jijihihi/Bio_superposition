#!/usr/bin/env python
# ==============================================================================
# CNN GAP UMAP Visualization
#
# Usage:
#   python -m visualization.umap_plot_gap \
#       --cache_path outputs/MoCo_seed42/CNN_GAP/cnn_gap_stage5_out.npz \
#       --cell_death_csv outputs/CellDeath_QC_patches/per_image_celldeath_rate.csv \
#       --output_dir outputs/UMAP_Plots \
#       --n_samples 3000 \
#       --apply_l2_norm
# ==============================================================================

import argparse
import os
import sys
import numpy as np

# Ensure project root is on sys.path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import matplotlib
_IN_COLAB = "google.colab" in sys.modules
if not _IN_COLAB:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F

try:
    import umap
except ImportError:
    print("Installing umap-learn...")
    os.system("pip -q install umap-learn")
    import umap

from trajectory_inference_pipeline.trajectory_utils import load_and_match_cell_death
from run_CNN.logging_utils import get_logger

logger = get_logger("umap_plot_gap")

CLASS_NAMES = {0: "Control", 1: "SNCA", 2: "GBA", 3: "LRRK2"}
CLASS_COLORS = {
    0: "#2176AE",  # vivid blue (Control)
    1: "#E8553A",  # vivid red-orange (SNCA)
    2: "#1DB954",  # vivid green (GBA)
    3: "#9B59B6",  # vivid purple (LRRK2)
}

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
sns.set_style("ticks")

def get_args():
    p = argparse.ArgumentParser("CNN GAP UMAP Visualization")
    p.add_argument("--cache_path", type=str, required=True, help="Path to cnn_gap.npz")
    p.add_argument("--cell_death_csv", type=str, required=True, help="Path to per-image cell_death rate CSV")
    p.add_argument("--output_dir", type=str, required=True)
    
    p.add_argument("--n_samples", type=int, default=5000, help="Number of samples per class")
    p.add_argument("--apply_l2_norm", action="store_true", help="Apply L2 norm before UMAP")
    
    p.add_argument("--n_neighbors", type=int, default=15)
    p.add_argument("--min_dist", type=float, default=0.1)
    p.add_argument("--metric", type=str, default="cosine")
    
    p.add_argument("--dot_size", type=float, default=5.0)
    p.add_argument("--alpha", type=float, default=0.6)
    p.add_argument("--cmap", type=str, default="magma_r")
    p.add_argument("--cmap_start", type=float, default=0.0, help="Start ratio for colormap (e.g. 0.1 to skip first 10%)")
    p.add_argument("--vmax_pctl", type=float, default=95.0)
    
    return p.parse_args()

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info(f"Loading cache from {args.cache_path}")
    data = np.load(args.cache_path, allow_pickle=True)
    X = data["X_gap"]
    y = data["y"]
    uids = data["uids"]
    
    if args.apply_l2_norm:
        logger.info("Applying L2 Normalization...")
        X = F.normalize(torch.tensor(X), dim=1).numpy()
        
    logger.info(f"Loaded {X.shape[0]} samples. Feature dim: {X.shape[1]}")

    # -------------------------------------------------------------------------
    # 0. Filter by Cell Death First
    # -------------------------------------------------------------------------
    from trajectory_inference_pipeline.trajectory_utils import load_and_match_cell_death
    logger.info("--- Loading Cell Death Rates and Filtering ---")
    cell_death_rates = load_and_match_cell_death(args.cell_death_csv, uids)
    
    valid_mask = ~np.isnan(cell_death_rates)
    X = X[valid_mask]
    y = y[valid_mask]
    cell_death_rates = cell_death_rates[valid_mask]
    uids = [uids[i] for i in range(len(uids)) if valid_mask[i]]
    
    logger.info(f"Filtered down to {len(X)} samples that have cell death information.")
    
    # -------------------------------------------------------------------------
    # 1. 4-Class UMAP
    # -------------------------------------------------------------------------
    logger.info(f"--- Preparing 4-Class Data (n={args.n_samples} per class) ---")
    idx_4class = []
    for c in [0, 1, 2, 3]:
        c_idx = np.where(y == c)[0]
        if len(c_idx) > args.n_samples:
            np.random.seed(42 + c)
            c_idx = np.random.choice(c_idx, args.n_samples, replace=False)
        idx_4class.extend(c_idx)
        
    idx_4class = np.array(idx_4class)
    X_4 = X[idx_4class]
    y_4 = y[idx_4class]
    
    logger.info(f"Fitting UMAP on {len(X_4)} samples for 4 classes...")
    reducer_4 = umap.UMAP(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=42
    )
    emb_4 = reducer_4.fit_transform(X_4)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    for c in [0, 1, 2, 3]:
        mask = (y_4 == c)
        ax.scatter(
            emb_4[mask, 0], emb_4[mask, 1],
            c=CLASS_COLORS[c], label=CLASS_NAMES[c],
            s=args.dot_size, alpha=args.alpha, edgecolors="none"
        )
    ax.set_title("UMAP: All 4 Classes")
    ax.legend(markerscale=3)
    ax.axis("off")
    fig.tight_layout()
    out1 = os.path.join(args.output_dir, "UMAP_4classes_colored_by_class.png")
    fig.savefig(out1, dpi=300)
    plt.close(fig)
    logger.info(f"Saved 4-class UMAP to {out1}")
    
    # -------------------------------------------------------------------------
    # 2. 3-Mutation UMAP (SNCA, GBA, LRRK2)
    # -------------------------------------------------------------------------
    logger.info(f"--- Preparing 3-Mutation Data (n={args.n_samples} per class) ---")
    idx_3class = []
    for c in [1, 2, 3]:  # Exclude Control (0)
        c_idx = np.where(y == c)[0]
        if len(c_idx) > args.n_samples:
            np.random.seed(42 + c)
            c_idx = np.random.choice(c_idx, args.n_samples, replace=False)
        idx_3class.extend(c_idx)
        
    idx_3class = np.array(idx_3class)
    X_3 = X[idx_3class]
    y_3 = y[idx_3class]
    uids_3 = [uids[i] for i in idx_3class]
    
    logger.info(f"Fitting UMAP on {len(X_3)} samples for 3 mutations...")
    reducer_3 = umap.UMAP(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=42
    )
    emb_3 = reducer_3.fit_transform(X_3)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    for c in [1, 2, 3]:
        mask = (y_3 == c)
        ax.scatter(
            emb_3[mask, 0], emb_3[mask, 1],
            c=CLASS_COLORS[c], label=CLASS_NAMES[c],
            s=args.dot_size, alpha=args.alpha, edgecolors="none"
        )
    ax.set_title("UMAP: 3 Mutations Only")
    ax.legend(markerscale=3)
    ax.axis("off")
    fig.tight_layout()
    out2 = os.path.join(args.output_dir, "UMAP_3mutations_colored_by_class.png")
    fig.savefig(out2, dpi=300)
    plt.close(fig)
    logger.info(f"Saved 3-mutation class UMAP to {out2}")
    
    # -------------------------------------------------------------------------
    # 3. 3-Mutation UMAP colored by Cell Death
    # -------------------------------------------------------------------------
    # Since we filtered first, ALL samples in emb_3 have valid cell death rates
    emb_3_valid = emb_3
    cd_valid = cell_death_rates[idx_3class]
    
    vmax = np.percentile(cd_valid, args.vmax_pctl)
    
    # Sort points so highest cell death is plotted last (on top)
    sort_idx = np.argsort(cd_valid)
    emb_sorted = emb_3_valid[sort_idx]
    cd_sorted = cd_valid[sort_idx]
    
    fig, ax = plt.subplots(figsize=(9, 8))
    
    # Set a faint gray background so that 'white' dots (0 cell death in afmhot_r) are visible
    ax.set_facecolor("#f5f5f5")
    
    # Create truncated colormap if cmap_start > 0
    from matplotlib.colors import LinearSegmentedColormap
    orig_cmap = plt.get_cmap(args.cmap)
    if args.cmap_start > 0.0:
        custom_cmap = LinearSegmentedColormap.from_list(
            f"trunc_{args.cmap}", orig_cmap(np.linspace(args.cmap_start, 1.0, 256))
        )
    else:
        custom_cmap = orig_cmap

    sc = ax.scatter(
        emb_sorted[:, 0], emb_sorted[:, 1],
        c=cd_sorted, cmap=custom_cmap,
        s=args.dot_size, alpha=args.alpha, edgecolors="none",
        vmax=vmax, vmin=0
    )
    ax.set_title("UMAP: 3 Mutations (Cell Death Rate)")
    ax.axis("off")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cell Death Rate")
    fig.tight_layout()
    out3 = os.path.join(args.output_dir, "UMAP_3mutations_colored_by_celldeath.png")
    fig.savefig(out3, dpi=300)
    plt.close(fig)
    logger.info(f"Saved 3-mutation cell death UMAP to {out3}")
    
    logger.info("🎉 All UMAP plots generated successfully.")

if __name__ == "__main__":
    main()
