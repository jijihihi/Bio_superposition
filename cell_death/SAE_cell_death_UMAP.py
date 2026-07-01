#!/usr/bin/env python
# ==============================================================================
# Simplified UMAP Cell Death Visualization
#
# SAE cache를 입력받아 지정된 각 mutation(e.g., SNCA, GBA, LRRK2)별로
# 독립적으로 UMAP을 수행한 뒤, 세포사멸율(cell_death rate)을 색상으로 표시합니다.
#
# Usage:
# !python cell_death/3D_cell_death_plot.py \
#     --sae_cache "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/caches_per_image_centering/SAE_vector_per_image_centering/CNN_seed123_SAE/sae_gap_d8192_lam800_normrestored_withnewclass.npz" \
#     --cell_death_csv "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/세포이미지별 사멸율/이미지별_세포사멸율_7200.csv" \
#     --classes SNCA LRRK2 GBA \
#     --umap_n_neighbors 15 \
#     --umap_min_dist 0.2 \
#     --umap_metric cosine \
#     --dot_size 7 \
#     --alpha 0.5 \
#     --gamma 1.0 \
#     --vmax_pctl 90 \
#     --cmap magma_r
# ==============================================================================

import argparse
import logging
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
from matplotlib.colors import Normalize, PowerNorm

from cell_death.local_knn_std import load_cache
from trajectory_inference_pipeline.trajectory_utils import \
    load_and_match_cell_death
from run_CNN.logging_utils import SUPERCLASS_MAP, get_logger

try:
    import umap as umap_lib

    _HAS_UMAP = True
except ImportError:
    umap_lib = None
    _HAS_UMAP = False

logger = get_logger("umap_cell_death")

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
sns.set_style("ticks")


def truncate_cmap(cmap_name, start=0.0, end=1.0, n=256):
    """Truncate a colormap to [start, end] range."""
    if start == 0.0 and end == 1.0:
        return plt.get_cmap(cmap_name)
    from matplotlib.colors import LinearSegmentedColormap

    base = plt.get_cmap(cmap_name)
    colors = base(np.linspace(start, end, n))
    return LinearSegmentedColormap.from_list(f"{cmap_name}_trunc", colors, N=n)


def get_args():
    p = argparse.ArgumentParser(description="SAE UMAP Cell Death Visualization")
    p.add_argument(
        "--sae_cache", type=str, required=True, help="Path to SAE .npz cache"
    )
    p.add_argument(
        "--cell_death_csv",
        type=str,
        required=True,
        help="Path to per-image cell_death rate CSV",
    )
    p.add_argument(
        "--rate_col",
        type=str,
        default=None,
        help="CSV column for coloring. None=auto (intensity_rate)",
    )
    p.add_argument("--dead_threshold", type=float, default=1e-5)

    # UMAP parameters
    p.add_argument("--umap_n_neighbors", type=int, default=15)
    p.add_argument("--umap_min_dist", type=float, default=0.2)
    p.add_argument("--umap_metric", type=str, default="cosine")

    # Visual styling
    p.add_argument(
        "--gamma",
        type=float,
        default=2.0,
        help="Gamma for power-law color mapping. Default: 2.0 (enhanced contrast)",
    )
    p.add_argument(
        "--dot_size", type=float, default=7.0, help="Scatter dot size. Default: 7"
    )
    p.add_argument(
        "--alpha", type=float, default=0.5, help="Dot transparency (0-1). Default: 0.5"
    )
    p.add_argument(
        "--cmap",
        type=str,
        default="magma_r",
        help="Colormap for cell_death rate. Default: magma_r",
    )
    p.add_argument(
        "--cmap_start",
        type=float,
        default=0.0,
        help="Truncate colormap start (0-1). Default: 0.0",
    )
    p.add_argument(
        "--cmap_end",
        type=float,
        default=1.0,
        help="Truncate colormap end (0-1). Default: 1.0",
    )
    p.add_argument(
        "--bg_color", type=str, default="white", help="Background color. Default: white"
    )
    p.add_argument(
        "--vmax_pctl",
        type=float,
        default=90.0,
        help="Percentile for color scale max. Default: 90",
    )
    p.add_argument(
        "--gap_l2_norm",
        action="store_true",
        help="L2 normalize feature vectors before UMAP",
    )

    # Output
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument(
        "--classes",
        type=str,
        nargs="+",
        default=["SNCA", "GBA", "LRRK2"],
        help="Mutations to plot separately.",
    )

    # Colab에서 그냥 실행 시 args 파싱 에러 방지
    if "ipykernel" in sys.modules and len(sys.argv) == 1:
        return p.parse_known_args(args=[])[0]
    return p.parse_args()


def plot_mutation_umap(embedding, cell_death, mutation, args, out_dir):
    is_dark = args.bg_color.lower() not in ("white", "#ffffff", "w")
    txt_color = "white" if is_dark else "black"

    # vmin/vmax with percentile robust scaling (mutation-specific)
    vmin = float(np.percentile(cell_death, 100 - args.vmax_pctl))
    vmax = float(np.percentile(cell_death, args.vmax_pctl))
    logger.info(
        f"    [{mutation}] Drawing UMAP scatter: vmin={vmin:.4f}, vmax={vmax:.4f}, gamma={args.gamma}"
    )

    # Gamma correction (power norm)
    if args.gamma != 1.0:
        norm = PowerNorm(gamma=args.gamma, vmin=vmin, vmax=vmax, clip=True)
        gamma_label = f" (γ={args.gamma})"
    else:
        norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
        gamma_label = ""

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(args.bg_color)
    ax.set_facecolor(args.bg_color)

    if args.cmap_start > 0.0 or args.cmap_end < 1.0:
        actual_cmap = truncate_cmap(args.cmap, args.cmap_start, args.cmap_end)
    else:
        actual_cmap = plt.get_cmap(args.cmap)

    sc = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=cell_death,
        cmap=actual_cmap,
        s=args.dot_size,
        alpha=args.alpha,
        norm=norm,
        edgecolors="none",
        rasterized=True,
    )

    ax.set_title(
        f"{mutation} UMAP — cell_death Rate",
        fontsize=14,
        fontweight="bold",
        color=txt_color,
    )
    ax.set_xlabel("UMAP 1", fontsize=11, color=txt_color)
    ax.set_ylabel("UMAP 2", fontsize=11, color=txt_color)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=9, colors=txt_color)

    for spine in ax.spines.values():
        spine.set_edgecolor(txt_color if is_dark else "black")

    cbar = fig.colorbar(
        sc, ax=ax, shrink=0.8, pad=0.03, label=f"cell_death Rate{gamma_label}"
    )
    cbar.ax.yaxis.set_tick_params(color=txt_color)
    cbar.ax.yaxis.label.set_color(txt_color)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=txt_color)

    fig.tight_layout()

    # Save files
    param_suffix = (
        f"nn{args.umap_n_neighbors}_md{args.umap_min_dist}_{args.umap_metric}"
    )
    base_path = os.path.join(out_dir, f"umap_cell_death_{mutation}_{param_suffix}")

    fig.savefig(
        f"{base_path}.png", dpi=args.dpi, bbox_inches="tight", facecolor=args.bg_color
    )
    fig.savefig(
        f"{base_path}.svg", format="svg", bbox_inches="tight", facecolor=args.bg_color
    )
    fig.savefig(
        f"{base_path}.pdf", format="pdf", bbox_inches="tight", facecolor=args.bg_color
    )

    if _IN_COLAB:
        plt.show()
    plt.close(fig)
    logger.info(f"  Saved {mutation} plot to {base_path}.png")


def main():
    args = get_args()
    np.random.seed(args.seed)

    if not _HAS_UMAP:
        logger.error("umap-learn is not installed. Please run: pip install umap-learn")
        return

    if not args.output_dir:
        # Default save directory alongside the cache
        args.output_dir = os.path.join(
            os.path.dirname(args.sae_cache), "umap_mutations"
        )
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Output directory: {args.output_dir}")

    # Load SAE cache
    logger.info(f"Loading SAE cache: {args.sae_cache}")
    X_sae, lines, uids, label = load_cache(args.sae_cache, args.dead_threshold)

    if args.gap_l2_norm:
        norms = np.linalg.norm(X_sae, axis=1, keepdims=True)
        X_sae = X_sae / np.where(norms == 0, 1e-12, norms)
        logger.info("  Applied L2 normalization")

    superclasses = np.array([SUPERCLASS_MAP.get(ln, ln) for ln in lines])

    # Load cell_death rates
    logger.info(f"Loading cell_death rates from: {args.cell_death_csv}")
    cell_death = load_and_match_cell_death(
        args.cell_death_csv, uids, rate_col=args.rate_col
    )

    # Compute global color scale percentiles (same for all mutations)
    valid_apop_all = cell_death[np.isfinite(cell_death)]
    if len(valid_apop_all) > 0:
        args.global_vmin = float(np.percentile(valid_apop_all, 100 - args.vmax_pctl))
        args.global_vmax = float(np.percentile(valid_apop_all, args.vmax_pctl))
        logger.info(
            f"Global cell_death color scale: vmin={args.global_vmin:.4f}, vmax={args.global_vmax:.4f}"
        )
    else:
        args.global_vmin = 0.0
        args.global_vmax = 1.0
        logger.warning("No valid cell_death values found for global scaling.")

    valid_apop = cell_death[np.isfinite(cell_death)]
    if len(valid_apop) > 0:
        logger.info(
            f"  cell_death rates: min={valid_apop.min():.4f}, max={valid_apop.max():.4f}, "
            f"mean={valid_apop.mean():.4f}, median={np.median(valid_apop):.4f}"
        )

    # Process each requested mutation individually
    for mutation in args.classes:
        logger.info(f"\n{'-'*50}\nProcessing mutation: {mutation}\n{'-'*50}")

        # Filter for this mutation and finite cell_death values
        mask = (superclasses == mutation) & np.isfinite(cell_death)
        X_mut = X_sae[mask]
        apop_mut = cell_death[mask]

        if len(X_mut) < 10:
            logger.warning(
                f"Not enough valid samples for {mutation} (N={len(X_mut)}). Skipping."
            )
            continue

        logger.info(
            f"  cell_death rates for {mutation}: min={apop_mut.min():.4f}, max={apop_mut.max():.4f}, "
            f"mean={apop_mut.mean():.4f}, median={np.median(apop_mut):.4f}"
        )

        logger.info(f"  Running UMAP for {mutation} (N={len(X_mut)})...")
        reducer = umap_lib.UMAP(
            n_components=2,
            n_neighbors=args.umap_n_neighbors,
            min_dist=args.umap_min_dist,
            metric=args.umap_metric,
            random_state=args.seed,
        )

        embedding = reducer.fit_transform(X_mut)
        logger.info(f"  UMAP completed.")

        logger.info(f"  Plotting...")
        plot_mutation_umap(embedding, apop_mut, mutation, args, args.output_dir)

    logger.info("All UMAP tasks completed.")


if __name__ == "__main__":
    main()
