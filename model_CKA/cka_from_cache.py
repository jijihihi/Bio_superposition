import argparse
import os
import sys

import matplotlib
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F

_IN_COLAB = "google.colab" in sys.modules
if not _IN_COLAB:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import the CKA math functions from cka_analysis.py
from model_CKA.cka_analysis import linear_CKA

def load_cache(npz_path: str, apply_l2_norm: bool) -> np.ndarray:
    """Load X_gap from npz and optionally apply L2 norm"""
    data = np.load(npz_path, allow_pickle=True)
    X = data["X_gap"]
    if apply_l2_norm:
        X = F.normalize(torch.tensor(X), dim=1).numpy()
    return X

def get_seed_from_path(path: str) -> str:
    """Extract seed name from path for plotting"""
    # e.g. .../outputs/MoCo_seed42_L2norm/CNN_GAP/cnn_gap_stage5_mid.npz -> "42_L2"
    basename = os.path.basename(os.path.dirname(os.path.dirname(path)))
    if "MoCo_seed" in basename:
        return basename.replace("MoCo_seed", "")
    return os.path.basename(os.path.dirname(path))

def plot_heatmap(matrix: np.ndarray, labels_x: list, labels_y: list, title: str, out_path: str):
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.family"] = "DejaVu Sans"

    fig, ax = plt.subplots(figsize=(max(4, len(labels_x) * 0.8), max(4, len(labels_y) * 0.8)))
    sns.heatmap(matrix, annot=True, fmt=".3f", cmap="Blues", 
                xticklabels=labels_x, yticklabels=labels_y, ax=ax,
                vmin=max(0.5, matrix.min() - 0.1), vmax=1.0)
    
    plt.title(title, pad=20)
    plt.tight_layout()
    plt.savefig(out_path, format="svg")
    print(f"Saved CKA heatmap to {out_path}")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser("Compute CKA from extracted CNN caches")
    parser.add_argument("--group_a", nargs="+", required=True, help="List of Group A .npz files")
    parser.add_argument("--group_b", nargs="+", default=[], help="List of Group B .npz files")
    parser.add_argument("--norm_a", action="store_true", help="Apply L2 norm to Group A caches")
    parser.add_argument("--norm_b", action="store_true", help="Apply L2 norm to Group B caches")
    parser.add_argument("--output_svg", type=str, required=True, help="Path to save output heatmap")
    parser.add_argument("--title", type=str, default="Linear CKA Similarity", help="Plot title")
    args = parser.parse_args()

    print(f"Loading Group A ({len(args.group_a)} caches) - L2 Norm: {args.norm_a}")
    feats_a = [load_cache(p, args.norm_a) for p in args.group_a]
    labels_a = [get_seed_from_path(p) for p in args.group_a]

    if args.group_b:
        print(f"Loading Group B ({len(args.group_b)} caches) - L2 Norm: {args.norm_b}")
        feats_b = [load_cache(p, args.norm_b) for p in args.group_b]
        labels_b = [get_seed_from_path(p) for p in args.group_b]

        # Compute A vs B matrix
        matrix = np.zeros((len(feats_a), len(feats_b)))
        for i, fa in enumerate(feats_a):
            for j, fb in enumerate(feats_b):
                matrix[i, j] = linear_CKA(fa, fb)
        
        plot_heatmap(matrix, labels_x=labels_b, labels_y=labels_a, title=args.title, out_path=args.output_svg)
    else:
        # Compute A vs A matrix
        matrix = np.zeros((len(feats_a), len(feats_a)))
        for i, fa in enumerate(feats_a):
            for j in range(i, len(feats_a)):
                cka_val = linear_CKA(fa, feats_a[j])
                matrix[i, j] = cka_val
                matrix[j, i] = cka_val  # Symmetric
        
        plot_heatmap(matrix, labels_x=labels_a, labels_y=labels_a, title=args.title, out_path=args.output_svg)

if __name__ == "__main__":
    main()
