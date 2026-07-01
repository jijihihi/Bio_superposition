import argparse
import glob
import os
import sys

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


plt.rcParams["svg.fonttype"] = "none"  
plt.rcParams["pdf.fonttype"] = 42  
# ───────────────────────────────────────────────────────────────────────

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
from trajectory_utils import get_logger

logger = get_logger("plot_dpt_heatmap")

base_cmap = plt.get_cmap("cool")  
colors = base_cmap(np.linspace(0, 1, 256))
white = np.array([1, 1, 1, 1])

pastel_colors = (1 - 0.6) * colors + 0.6 * white
smooth_pastel_cool = mcolors.ListedColormap(pastel_colors)


def get_args():
    p = argparse.ArgumentParser(description="Plot heatmaps for DPT parameter sweep")
    p.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing dpt_summary_...csv files",
    )
    return p.parse_args()


def run_plot_heatmap(args):
    csv_files = glob.glob(os.path.join(args.input_dir, "dpt_summary_*.csv"))
    if not csv_files:
        logger.error(f"No CSV files found in {args.input_dir}")
        return

    logger.info(f"Found {len(csv_files)} CSV files. Aggregating...")
    df_list = []
    for f in csv_files:
        df_list.append(pd.read_csv(f))

    df = pd.concat(df_list, ignore_index=True)

    if "PCA" not in df.columns:
        df["PCA"] = df["Features"]

    df_agg = (
        df.groupby(["Mutation", "PCA", "kNN"])
        .agg({"rho": "mean", "r": "mean"})
        .reset_index()
    )

    rho_min, rho_max = df_agg["rho"].min(), df_agg["rho"].max()
    r_min, r_max = df_agg["r"].min(), df_agg["r"].max()

    mutations_order = [
        m for m in ["SNCA", "GBA", "LRRK2"] if m in df_agg["Mutation"].unique()
    ]

    # 1. Spearman 1x3 Heatmap
    fig_rho, axes_rho = plt.subplots(
        1, len(mutations_order), figsize=(6 * len(mutations_order), 5)
    )
    if len(mutations_order) == 1:
        axes_rho = [axes_rho]

    # 2. Pearson 1x3 Heatmap
    fig_r, axes_r = plt.subplots(
        1, len(mutations_order), figsize=(6 * len(mutations_order), 5)
    )
    if len(mutations_order) == 1:
        axes_r = [axes_r]

    for i, mut in enumerate(mutations_order):
        df_mut = df_agg[df_agg["Mutation"] == mut]

        pivot_rho = df_mut.pivot(index="kNN", columns="PCA", values="rho")
        pivot_r = df_mut.pivot(index="kNN", columns="PCA", values="r")

        
        sns.heatmap(
            pivot_rho,
            annot=True,
            fmt=".3f",
            cmap=smooth_pastel_cool,
            vmin=-0.5,
            vmax=-0.15,
            ax=axes_rho[i],
            linewidths=0.5,  
            linecolor="lightgray",  
            cbar=(i == len(mutations_order) - 1),
            cbar_kws={"label": "Spearman ρ"} if i == len(mutations_order) - 1 else None,
        )
        axes_rho[i].set_title(f"{mut} - Spearman ρ")
        axes_rho[i].invert_yaxis()

        
        sns.heatmap(
            pivot_r,
            annot=True,
            fmt=".3f",
            cmap="spring",
            vmin=-0.5,
            vmax=-0.15,
            ax=axes_r[i],
            linewidths=0.5,  
            linecolor="lightgray",  
            cbar=(i == len(mutations_order) - 1),
            cbar_kws={"label": "Pearson r"} if i == len(mutations_order) - 1 else None,
        )
        axes_r[i].set_title(f"{mut} - Pearson r")
        axes_r[i].invert_yaxis()

    
    fig_rho.tight_layout()
    rho_path = os.path.join(args.input_dir, "heatmap_dpt_spearman_all_mutations.png")
    fig_rho.savefig(rho_path, dpi=600, bbox_inches="tight")
    fig_rho.savefig(rho_path.replace(".png", ".pdf"), bbox_inches="tight")
    fig_rho.savefig(rho_path.replace(".png", ".svg"), format="svg", bbox_inches="tight")
    plt.close(fig_rho)

    
    fig_r.tight_layout()
    r_path = os.path.join(args.input_dir, "heatmap_dpt_pearson_all_mutations.png")
    fig_r.savefig(r_path, dpi=600, bbox_inches="tight")
    fig_r.savefig(r_path.replace(".png", ".pdf"), bbox_inches="tight")
    fig_r.savefig(r_path.replace(".png", ".svg"), format="svg", bbox_inches="tight")
    plt.close(fig_r)

    logger.info(
        "Saved consolidated 1x3 heatmaps for Spearman and Pearson with editable text."
    )


if __name__ == "__main__":
    args = get_args()
    run_plot_heatmap(args)
