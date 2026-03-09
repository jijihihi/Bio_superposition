# ==============================================================================
# Analyze texture_attribution .npz results (Colab)
#
# Loads one or more npz files from step08c_texture_attribution.py and produces:
#   1. Per-neuron DataFrame with 9/12-component GAP values (raw + signed fractions)
#   2. Linearity check summary
#   3. Stacked bar chart of component fractions per neuron (signed, via total_spatial)
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

## fraction = component / (orig - baseline). 음수 포함. 전체 spatial info 중 이 component가 차지하는 signed 비율.

# ── Configuration ──
# Adjust these paths to your Drive-mounted npz files
NPZ_PATHS = {
    "ps8_blur4.0": "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/MoCo_seed87/SAE_sparsity3200_loss_L2norm곱해줌/texture_attribution/texture_attribution_ps8_blur4.0.npz",
    "ps4_blur2.0": "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/MoCo_seed87/SAE_sparsity3200_loss_L2norm곱해줌/texture_attribution/texture_attribution_ps4_blur2.0.npz",
}

METRIC = "l2sq"  # default; overridden by --metric arg

COMPONENT_NAMES_12 = [
    "RG_inter", "RB_inter", "GB_inter",
    "R_local",  "G_local",  "B_local",
    "R_ref",    "G_ref",    "B_ref",
    "R_tex",    "G_tex",    "B_tex",
]

# 9-component fallback (old npz without rotation)
COMPONENT_NAMES_9 = [
    "RG_inter", "RB_inter", "GB_inter",
    "R_local",  "G_local",  "B_local",
    "R_tex",    "G_tex",    "B_tex",
]

# Colors for stacked bar
COLORS = {
    "RG_inter": "#e74c3c", "RB_inter": "#e67e22", "GB_inter": "#f1c40f",
    "R_local":  "#2ecc71", "G_local":  "#27ae60", "B_local":  "#1abc9c",
    "R_ref":    "#9b59b6", "G_ref":    "#8e44ad", "B_ref":    "#6c3483",
    "R_tex":    "#3498db", "G_tex":    "#2980b9", "B_tex":    "#1f618d",
    # hybrid (for summary only)
    "R_hybrid": "#95a5a6", "G_hybrid": "#7f8c8d", "B_hybrid": "#566573",
}


def load_npz(path):
    """Load npz and return a per-neuron DataFrame with BOTH gap and l2sq metrics."""
    data = np.load(path, allow_pickle=True)
    alive_mask = data["alive_mask"].astype(bool)
    d_sae = len(alive_mask)

    # Detect if npz has rotation data (check with gap metric)
    has_rotation = "gap_R_ref" in data
    comp_names = COMPONENT_NAMES_12 if has_rotation else COMPONENT_NAMES_9

    rows = []
    for nid in range(d_sae):
        if not alive_mask[nid]:
            continue
        row = {"neuron_id": nid}
        for m in ["gap", "l2sq"]:
            row[f"{m}_orig"] = float(data[f"{m}_tk_orig"][nid])
            row[f"{m}_baseline"] = float(data[f"{m}_tk_baseline"][nid])
            row[f"{m}_all_broken"] = float(data[f"{m}_tk_all_broken"][nid])
            for comp in comp_names:
                row[f"{m}_{comp}"] = float(data[f"{m}_{comp}"][nid])
            if has_rotation:
                for ch in ["R", "G", "B"]:
                    row[f"{m}_{ch}_hybrid"] = float(data[f"{m}_{ch}_hybrid"][nid])
            row[f"{m}_linearity"] = float(data[f"{m}_linearity"][nid])
        rows.append(row)

    df = pd.DataFrame(rows)
    df.attrs["has_rotation"] = has_rotation
    df.attrs["comp_names"] = comp_names

    # Compute fractions for each metric
    for m in ["gap", "l2sq"]:
        df[f"{m}_total_spatial"] = df[f"{m}_orig"] - df[f"{m}_baseline"]
        # Fraction is unreliable when total_spatial ≤ 0 or very small → set to NaN
        min_denom = max(df[f"{m}_orig"].median() * 0.01, 1e-8)
        valid = df[f"{m}_total_spatial"] > min_denom
        for comp in comp_names:
            df[f"{m}_{comp}_frac"] = np.where(
                valid, df[f"{m}_{comp}"] / df[f"{m}_total_spatial"], np.nan)
        # Category fractions
        df[f"{m}_interaction_frac"] = df[[f"{m}_RG_inter_frac", f"{m}_RB_inter_frac", f"{m}_GB_inter_frac"]].sum(axis=1)
        df[f"{m}_local_shape_frac"] = df[[f"{m}_R_local_frac", f"{m}_G_local_frac", f"{m}_B_local_frac"]].sum(axis=1)
        df[f"{m}_texture_frac"] = df[[f"{m}_R_tex_frac", f"{m}_G_tex_frac", f"{m}_B_tex_frac"]].sum(axis=1)
        if has_rotation:
            df[f"{m}_reference_frac"] = df[[f"{m}_R_ref_frac", f"{m}_G_ref_frac", f"{m}_B_ref_frac"]].sum(axis=1)

    # Convenience aliases for the primary metric (for plotting)
    # These will be set by set_primary_metric()

    # Metadata
    df.attrs["patch_size"] = int(data["patch_size"])
    df.attrs["blur_sigma"] = float(data["blur_sigma"])
    df.attrs["top_k"] = int(data["top_k"])
    df.attrs["n_alive"] = int(alive_mask.sum())

    return df


def set_primary_metric(df, metric):
    """Create un-prefixed aliases so plotting functions work with either metric."""
    comp_names = df.attrs.get("comp_names", COMPONENT_NAMES_9)
    has_rot = df.attrs.get("has_rotation", False)
    m = metric
    for col in ["orig", "baseline", "all_broken", "total_spatial", "linearity"]:
        df[col] = df[f"{m}_{col}"]
    for comp in comp_names:
        df[comp] = df[f"{m}_{comp}"]
        df[f"{comp}_frac"] = df[f"{m}_{comp}_frac"]
    if has_rot:
        for ch in ["R", "G", "B"]:
            df[f"{ch}_hybrid"] = df[f"{m}_{ch}_hybrid"]
    for cat in ["interaction_frac", "local_shape_frac", "texture_frac"]:
        df[cat] = df[f"{m}_{cat}"]
    if has_rot:
        df["reference_frac"] = df[f"{m}_reference_frac"]
    return df


def print_summary(df, label=""):
    """Print aggregate statistics."""
    ps = df.attrs.get("patch_size", "?")
    bs = df.attrs.get("blur_sigma", "?")
    has_rot = df.attrs.get("has_rotation", False)
    comp_names = df.attrs.get("comp_names", COMPONENT_NAMES_9)
    mode = "12-comp (with rotation)" if has_rot else "9-comp (no rotation)"
    print(f"\n{'='*60}")
    print(f"  {label}  (ps={ps}, blur={bs}, metric={METRIC}, {mode})")
    print(f"  {len(df)} alive neurons")
    print(f"{'='*60}")

    print(f"\n  orig mean:      {df['orig'].mean():.6f}")
    print(f"  baseline mean:  {df['baseline'].mean():.6f}")
    print(f"  all_broken mean:{df['all_broken'].mean():.6f}")

    print(f"\n  ── Components (raw, mean ± std) ──")
    for comp in comp_names:
        v = df[comp]
        n_neg = (v < 0).sum()
        print(f"  {comp:12s}: {v.mean():+.6f} ± {v.std():.6f}  "
              f"(neg: {n_neg}/{len(v)}, {n_neg/len(v)*100:.1f}%)")

    if has_rot:
        print(f"\n  ── Hybrid (local + ref, for verification) ──")
        for ch in ["R", "G", "B"]:
            h = df[f"{ch}_hybrid"].mean()
            l = df[f"{ch}_local"].mean()
            r = df[f"{ch}_ref"].mean()
            print(f"  {ch}: hybrid={h:+.6f}, local+ref={l+r:+.6f}, diff={h-(l+r):.2e}")

    print(f"\n  ── Category Fractions (signed, mean ± std) ──")
    cats = [("Interaction", "interaction_frac"),
            ("Local Shape", "local_shape_frac")]
    if has_rot:
        cats.append(("Reference", "reference_frac"))
    cats.append(("Texture", "texture_frac"))
    for cat, col in cats:
        v = df[col]
        print(f"  {cat:14s}: {v.mean():+.4f} ± {v.std():.4f}")

    print(f"\n  ── Linearity ──")
    lin = df["linearity"].dropna()
    print(f"  N neurons: {len(lin)}")
    print(f"  mean={lin.mean():.4f}, median={lin.median():.4f}")
    print(f"  5th/95th: [{lin.quantile(0.05):.4f}, {lin.quantile(0.95):.4f}]")
    in_band = ((lin > 0.5) & (lin < 1.5)).sum()
    print(f"  [0.5-1.5]: {in_band}/{len(lin)} ({in_band/len(lin)*100:.1f}%)")


def plot_stacked_bar(df, label="", top_n=50, sort_by="orig"):
    """Stacked bar chart of component fractions per neuron."""
    comp_names = df.attrs.get("comp_names", COMPONENT_NAMES_9)
    df_sorted = df.sort_values(sort_by, ascending=False).head(top_n).copy()

    fig, ax = plt.subplots(figsize=(max(16, top_n * 0.3), 5))
    x = np.arange(len(df_sorted))
    bottom = np.zeros(len(df_sorted))

    # Separate positive and negative parts for stacked bar
    for comp in comp_names:
        vals = df_sorted[f"{comp}_frac"].values
        pos = np.maximum(vals, 0)
        neg = np.minimum(vals, 0)
        ax.bar(x, pos, bottom=bottom, label=comp, color=COLORS[comp], width=0.85)
        ax.bar(x, neg, bottom=0, color=COLORS[comp], width=0.85, alpha=0.4)
        bottom += pos

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(nid)}" for nid in df_sorted["neuron_id"]],
                       rotation=90, fontsize=6)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Fraction of total spatial info (signed)")
    ax.set_title(f"Component Decomposition per Neuron ({label}, top {top_n} by {sort_by})")
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
                    c=df["texture_frac"], cmap="RdBu_r", s=15, alpha=0.7)
    plt.colorbar(sc, ax=ax, label="Texture fraction")
    ax.set_xlabel("Interaction fraction")
    ax.set_ylabel("Local Shape fraction")
    ax.set_title(f"Category Fractions - signed ({label})")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    plt.tight_layout()
    plt.show()


def compare_conditions(dfs, comp_list=None):
    """Compare same neurons across different conditions (patch_size/blur_sigma)."""
    if comp_list is None:
        comp_list = COMPONENT_NAMES_12

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
    n_comp = len(comp_list)
    n_cols = 3
    n_rows = (n_comp + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.5 * n_rows))
    for i, comp in enumerate(comp_list):
        ax = axes[i // n_cols, i % n_cols]
        v1 = df1.loc[common, comp].values
        v2 = df2.loc[common, comp].values
        ax.scatter(v1, v2, s=8, alpha=0.4)
        lim = max(abs(v1).max(), abs(v2).max()) * 1.1
        ax.plot([-lim, lim], [-lim, lim], "r--", alpha=0.5)
        ax.set_xlabel(labels[0], fontsize=8)
        ax.set_ylabel(labels[1], fontsize=8)
        ax.set_title(comp, fontsize=10)
        ax.set_aspect("equal")
    # Hide unused axes
    for i in range(n_comp, n_rows * n_cols):
        axes[i // n_cols, i % n_cols].set_visible(False)
    plt.suptitle(f"Per-Neuron Component Comparison", fontsize=13)
    plt.tight_layout()
    plt.show()


def export_per_neuron_csv(df, output_path, metric, label=""):
    """Export per-neuron data to CSV for one metric."""
    comp_names = df.attrs.get("comp_names", COMPONENT_NAMES_9)
    has_rot = df.attrs.get("has_rotation", False)
    m = metric
    # Build source columns (prefixed) and clean names (un-prefixed)
    src_cols = ["neuron_id"]
    out_names = ["neuron_id"]
    for col in ["orig", "baseline", "all_broken", "total_spatial"]:
        src_cols.append(f"{m}_{col}")
        out_names.append(col)
    for c in comp_names:
        src_cols.append(f"{m}_{c}")
        out_names.append(c)
    if has_rot:
        for ch in ["R", "G", "B"]:
            src_cols.append(f"{m}_{ch}_hybrid")
            out_names.append(f"{ch}_hybrid")
    for c in comp_names:
        src_cols.append(f"{m}_{c}_frac")
        out_names.append(f"{c}_frac")
    cat_cols = ["interaction_frac", "local_shape_frac"]
    if has_rot:
        cat_cols.append("reference_frac")
    cat_cols.append("texture_frac")
    for cat in cat_cols:
        src_cols.append(f"{m}_{cat}")
        out_names.append(cat)
    src_cols.append(f"{m}_linearity")
    out_names.append("linearity")
    out_df = df[src_cols].copy()
    out_df.columns = out_names
    out_df.to_csv(output_path, index=False, float_format="%.6f")
    print(f"Saved: {output_path}  ({len(df)} neurons, metric={metric})")


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
    print(f"Primary metric for visualization: {METRIC}")

    dfs = {}
    for label, path in NPZ_PATHS.items():
        if not os.path.exists(path):
            print(f"⚠ Not found: {path}")
            continue
        print(f"Loading: {path}")
        df = load_npz(path)
        set_primary_metric(df, METRIC)  # aliases for plotting
        dfs[label] = df
        print_summary(df, label=label)

    # Per-condition visualizations
    for label, df in dfs.items():
        plot_stacked_bar(df, label=label, top_n=50, sort_by="orig")
        plot_linearity_hist(df, label=label)
        plot_category_scatter(df, label=label)

        # Export CSV — one per metric
        for m in ["gap", "l2sq"]:
            csv_path = NPZ_PATHS[label].replace(".npz", f"_{m}_per_neuron.csv")
            export_per_neuron_csv(df, csv_path, metric=m, label=label)

    # Cross-condition comparison
    if len(dfs) >= 2:
        compare_conditions(dfs)


if __name__ == "__main__":
    main()
