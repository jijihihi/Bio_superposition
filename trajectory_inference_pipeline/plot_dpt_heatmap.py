import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trajectory_utils import get_logger

logger = get_logger("plot_dpt_heatmap")

def get_args():
    p = argparse.ArgumentParser(description="Plot heatmaps for DPT parameter sweep")
    p.add_argument("--input_dir", type=str, required=True, help="Directory containing dpt_summary_...csv files")
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
    
    # Backward compatibility if someone runs this on older files
    if "PCA" not in df.columns:
        df["PCA"] = df["Features"]
        
    # Group by Mutation, PCA, kNN and compute mean
    df_agg = df.groupby(["Mutation", "PCA", "kNN"]).agg({"rho": "mean", "r": "mean"}).reset_index()
    
    mutations = df_agg["Mutation"].unique()
    
    for mut in mutations:
        df_mut = df_agg[df_agg["Mutation"] == mut]
        
        # Pivot tables for heatmaps
        pivot_rho = df_mut.pivot(index="PCA", columns="kNN", values="rho")
        pivot_r = df_mut.pivot(index="PCA", columns="kNN", values="r")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Spearman
        sns.heatmap(pivot_rho, annot=True, fmt=".3f", cmap="YlGnBu", ax=axes[0], cbar_kws={'label': 'Spearman ρ'})
        axes[0].set_title(f"{mut} - Spearman ρ\n(Mean over seeds)")
        axes[0].invert_yaxis()
        
        # Pearson
        sns.heatmap(pivot_r, annot=True, fmt=".3f", cmap="YlOrRd", ax=axes[1], cbar_kws={'label': 'Pearson r'})
        axes[1].set_title(f"{mut} - Pearson r\n(Mean over seeds)")
        axes[1].invert_yaxis()
        
        fig.tight_layout()
        out_path = os.path.join(args.input_dir, f"heatmap_dpt_{mut}.png")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved heatmaps for {mut} to {out_path}")

if __name__ == "__main__":
    args = get_args()
    run_plot_heatmap(args)
