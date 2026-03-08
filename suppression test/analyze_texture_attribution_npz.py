# ==============================================================================
# Analyze texture_attribution .npz results (Colab)
#
# Loads one or more npz files from step08c_texture_attribution.py and produces:
#   1. Per-neuron DataFrame with 9-component GAP values (raw + fractions)
#   2. Linearity check summary
#   3. Stacked bar chart of component fractions per neuron
#   4. Comparison between conditions (e.g., patch_size=8 vs 4)
#
# Usage (Colab):
#   Mount Drive, set NPZ_PATHS below, run all cells.
# ==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

# ── Configuration ──
# Adjust these paths to your Drive-mounted npz files
NPZ_PATHS = {
    "ps8_blur4.0": "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/MoCo_seed87/SAE_sparsity3200_loss_L2norm곱해줌/texture_attribution/texture_attribution_ps8_blur4.0.npz",
    "ps4_blur2.0": "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/MoCo_seed87/SAE_sparsity3200_loss_L2norm곱해줌/texture_attribution/texture_attribution_ps4_blur2.0.npz",
}

METRIC = "gap"  # default; overridden by --metric arg

COMPONENT_NAMES = [
    "RG_inter", "RB_inter", "GB_inter",
    "R_local",  "G_local",  "B_local",
    "R_tex",    "G_tex",    "B_tex",
]

CATEGORY_MAP = {
    "RG_inter": "Interaction", "RB_inter": "Interaction", "GB_inter": "Interaction",
    "R_local": "Local Shape",  "G_local": "Local Shape",  "B_local": "Local Shape",
    "R_tex": "Texture",        "G_tex": "Texture",        "B_tex": "Texture",
}

# Colors for stacked bar
COLORS = {
    "RG_inter": "#e74c3c", "RB_inter": "#e67e22", "GB_inter": "#f1c40f",
    "R_local":  "#2ecc71", "G_local":  "#27ae60", "B_local":  "#1abc9c",
    "R_tex":    "#3498db", "G_tex":    "#2980b9", "B_tex":    "#8e44ad",
}


def load_npz(path, metric=None):
    """Load npz and return a per-neuron DataFrame for the chosen metric."""
    data = np.load(path, allow_pickle=True)
    alive_mask = data["alive_mask"].astype(bool)
    d_sae = len(alive_mask)
    m = metric or METRIC

    rows = []
    for nid in range(d_sae):
        if not alive_mask[nid]:
            continue
        row = {"neuron_id": nid}
        # Original and baseline
        row["orig"] = float(data[f"{m}_tk_orig"][nid])
        row["baseline"] = float(data[f"{m}_tk_baseline"][nid])
        row["all_broken"] = float(data[f"{m}_tk_all_broken"][nid])
        # 9 components (raw, can be negative)
        for comp in COMPONENT_NAMES:
            row[comp] = float(data[f"{m}_{comp}"][nid])
        # Linearity
        row["linearity"] = float(data[f"{m}_linearity"][nid])
        rows.append(row)

    df = pd.DataFrame(rows)

    # Total spatial = orig - baseline
    df["total_spatial"] = df["orig"] - df["baseline"]

    # Sum of 9 components
    df["sum_9"] = df[COMPONENT_NAMES].sum(axis=1)

    # Reconstructed = baseline + sum_9
    df["reconstructed"] = df["baseline"] + df["sum_9"]

    # Reconstruction ratio = reconstructed / orig
    df["recon_ratio"] = df["reconstructed"] / (df["orig"] + 1e-12)

    # Fractions (raw, can be negative)
    for comp in COMPONENT_NAMES:
        df[f"{comp}_frac"] = df[comp] / (df["total_spatial"] + 1e-12)

    # Clipped fractions (for visualization)
    clipped = df[COMPONENT_NAMES].clip(lower=0)
    total_clipped = clipped.sum(axis=1) + 1e-12
    for comp in COMPONENT_NAMES:
        df[f"{comp}_frac_clip"] = clipped[comp] / total_clipped

    # Category fractions (clipped)
    df["interaction_frac"] = df[["RG_inter_frac_clip", "RB_inter_frac_clip", "GB_inter_frac_clip"]].sum(axis=1)
    df["local_shape_frac"] = df[["R_local_frac_clip", "G_local_frac_clip", "B_local_frac_clip"]].sum(axis=1)
    df["texture_frac"] = df[["R_tex_frac_clip", "G_tex_frac_clip", "B_tex_frac_clip"]].sum(axis=1)

    # Metadata
    df.attrs["patch_size"] = int(data["patch_size"])
    df.attrs["blur_sigma"] = float(data["blur_sigma"])
    df.attrs["top_k"] = int(data["top_k"])
    df.attrs["n_alive"] = int(alive_mask.sum())

    return df


def print_summary(df, label=""):
    """Print aggregate statistics."""
    ps = df.attrs.get("patch_size", "?")
    bs = df.attrs.get("blur_sigma", "?")
    print(f"\n{'='*60}")
    print(f"  {label}  (patch_size={ps}, blur_sigma={bs}, metric={METRIC})")
    print(f"  {len(df)} alive neurons")
    print(f"{'='*60}")

    print(f"\n  orig mean:      {df['orig'].mean():.6f}")
    print(f"  baseline mean:  {df['baseline'].mean():.6f}")
    print(f"  all_broken mean:{df['all_broken'].mean():.6f}")

    print(f"\n  ── 9 Components (raw GAP, mean ± std) ──")
    for comp in COMPONENT_NAMES:
        v = df[comp]
        n_neg = (v < 0).sum()
        print(f"  {comp:12s}: {v.mean():+.6f} ± {v.std():.6f}  "
              f"(neg: {n_neg}/{len(v)}, {n_neg/len(v)*100:.1f}%)")

    print(f"\n  ── Category Fractions (clipped, mean ± std) ──")
    for cat, col in [("Interaction", "interaction_frac"),
                     ("Local Shape", "local_shape_frac"),
                     ("Texture", "texture_frac")]:
        v = df[col]
        print(f"  {cat:14s}: {v.mean():.4f} ± {v.std():.4f}")

    print(f"\n  ── Linearity ──")
    lin = df["linearity"].dropna()
    print(f"  N neurons: {len(lin)}")
    print(f"  mean={lin.mean():.4f}, median={lin.median():.4f}")
    print(f"  5th/95th: [{lin.quantile(0.05):.4f}, {lin.quantile(0.95):.4f}]")
    in_band = ((lin > 0.5) & (lin < 1.5)).sum()
    print(f"  [0.5-1.5]: {in_band}/{len(lin)} ({in_band/len(lin)*100:.1f}%)")


def plot_stacked_bar(df, label="", top_n=50, sort_by="orig"):
    """Stacked bar chart of 9-component fractions per neuron."""
    df_sorted = df.sort_values(sort_by, ascending=False).head(top_n).copy()

    fig, ax = plt.subplots(figsize=(max(16, top_n * 0.3), 5))
    x = np.arange(len(df_sorted))
    bottom = np.zeros(len(df_sorted))

    for comp in COMPONENT_NAMES:
        vals = df_sorted[f"{comp}_frac_clip"].values
        ax.bar(x, vals, bottom=bottom, label=comp, color=COLORS[comp], width=0.85)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(nid)}" for nid in df_sorted["neuron_id"]],
                       rotation=90, fontsize=6)
    ax.set_ylabel("Fraction (clipped ≥0)")
    ax.set_title(f"9-Component Decomposition per Neuron ({label}, top {top_n} by {sort_by})")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.set_xlim(-0.5, len(x) - 0.5)
    plt.tight_layout()
    plt.show()


def plot_linearity_hist(df, label=""):
    """Histogram of per-neuron linearity values."""
    lin = df["linearity"].dropna()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(lin, bins=50, range=(0.5, 1.5), edgecolor="black", alpha=0.7)
    ax.axvline(1.0, color="red", linestyle="--", label="Perfect (1.0)")
    ax.set_xlabel("Linearity (reconstructed / orig)")
    ax.set_ylabel("Count")
    ax.set_title(f"Per-Neuron Linearity Distribution ({label})")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_category_scatter(df, label=""):
    """Scatter: interaction fraction vs local shape fraction, colored by texture."""
    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(df["interaction_frac"], df["local_shape_frac"],
                    c=df["texture_frac"], cmap="viridis", s=15, alpha=0.7)
    plt.colorbar(sc, ax=ax, label="Texture fraction")
    ax.set_xlabel("Interaction fraction")
    ax.set_ylabel("Local Shape fraction")
    ax.set_title(f"Category Fractions ({label})")
    ax.plot([0, 1], [1, 0], "k--", alpha=0.3)  # diagonal
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.show()


def compare_conditions(dfs, comp_list=None):
    """Compare same neurons across different conditions (patch_size/blur_sigma)."""
    if comp_list is None:
        comp_list = COMPONENT_NAMES

    labels = list(dfs.keys())
    if len(labels) < 2:
        print("Need at least 2 conditions to compare")
        return

    # Merge on neuron_id
    df1 = dfs[labels[0]].set_index("neuron_id")
    df2 = dfs[labels[1]].set_index("neuron_id")
    common = df1.index.intersection(df2.index)
    print(f"\nComparing {labels[0]} vs {labels[1]}: {len(common)} common neurons\n")

    # Per-component comparison
    print(f"  {'Component':14s} | {labels[0]:>12s} | {labels[1]:>12s} | {'Δ mean':>10s}")
    print(f"  {'-'*14}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}")
    for comp in comp_list:
        m1 = df1.loc[common, comp].mean()
        m2 = df2.loc[common, comp].mean()
        delta = m2 - m1
        print(f"  {comp:14s} | {m1:12.6f} | {m2:12.6f} | {delta:+10.6f}")

    # Category comparison
    print()
    for cat, cols in [("Interaction", ["RG_inter", "RB_inter", "GB_inter"]),
                      ("Local Shape", ["R_local", "G_local", "B_local"]),
                      ("Texture",     ["R_tex", "G_tex", "B_tex"])]:
        m1 = df1.loc[common, cols].sum(axis=1).mean()
        m2 = df2.loc[common, cols].sum(axis=1).mean()
        delta = m2 - m1
        print(f"  [{cat:12s}] | {m1:12.6f} | {m2:12.6f} | {delta:+10.6f}")

    # Plot per-component comparison
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    for i, comp in enumerate(COMPONENT_NAMES):
        ax = axes[i // 3, i % 3]
        v1 = df1.loc[common, comp].values
        v2 = df2.loc[common, comp].values
        ax.scatter(v1, v2, s=8, alpha=0.4)
        lim = max(abs(v1).max(), abs(v2).max()) * 1.1
        ax.plot([-lim, lim], [-lim, lim], "r--", alpha=0.5)
        ax.set_xlabel(labels[0], fontsize=8)
        ax.set_ylabel(labels[1], fontsize=8)
        ax.set_title(comp, fontsize=10)
        ax.set_aspect("equal")
    plt.suptitle(f"Per-Neuron Component Comparison", fontsize=13)
    plt.tight_layout()
    plt.show()


def export_per_neuron_csv(df, output_path, label=""):
    """Export per-neuron data to CSV for detailed inspection."""
    cols = (["neuron_id", "orig", "baseline", "all_broken", "total_spatial"]
            + COMPONENT_NAMES
            + [f"{c}_frac" for c in COMPONENT_NAMES]
            + ["interaction_frac", "local_shape_frac", "texture_frac"]
            + ["linearity", "recon_ratio"])
    df[cols].to_csv(output_path, index=False, float_format="%.6f")
    print(f"Saved: {output_path}  ({len(df)} neurons)")


# ==============================================================================
# Main
# ==============================================================================
def get_args():
    p = argparse.ArgumentParser("Analyze texture attribution npz")
    p.add_argument("--metric", type=str, default="gap", choices=["gap", "l2sq"],
                   help="Which metric to analyze (default: gap)")
    return p.parse_args()


def main():
    global METRIC
    args = get_args()
    METRIC = args.metric
    print(f"Metric: {METRIC}")

    dfs = {}
    for label, path in NPZ_PATHS.items():
        if not os.path.exists(path):
            print(f"⚠ Not found: {path}")
            continue
        print(f"Loading: {path}")
        df = load_npz(path, metric=METRIC)
        dfs[label] = df
        print_summary(df, label=label)

    # Per-condition visualizations
    for label, df in dfs.items():
        plot_stacked_bar(df, label=label, top_n=50, sort_by="orig")
        plot_linearity_hist(df, label=label)
        plot_category_scatter(df, label=label)

        # Export CSV (includes metric in filename)
        csv_path = NPZ_PATHS[label].replace(".npz", f"_{METRIC}_per_neuron.csv")
        export_per_neuron_csv(df, csv_path, label=label)

    # Cross-condition comparison
    if len(dfs) >= 2:
        compare_conditions(dfs)


if __name__ == "__main__":
    main()
