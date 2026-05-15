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
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
import argparse

# ── Illustrator-friendly SVG defaults ──
matplotlib.rcParams['svg.fonttype'] = 'none'       # text as text, not paths
matplotlib.rcParams['pdf.fonttype'] = 42            # TrueType in PDF too
matplotlib.rcParams['font.family'] = 'DejaVu Sans'   # bundled with matplotlib; swap to Arial in Illustrator if needed
matplotlib.rcParams['axes.unicode_minus'] = False

## fraction = component / (orig - baseline). 음수 포함. 전체 spatial info 중 이 component가 차지하는 signed 비율.

# ── Configuration ──
# Adjust these paths to your Drive-mounted npz files
NPZ_PATHS = {
    "ps16_blur8.0": "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/MoCo_seed87/SAE_sparsity3200_loss_L2norm곱해줌/texture_attribution/texture_attribution_ps16_blur8.0.npz",
    "ps8_blur4.0": "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/MoCo_seed87/SAE_sparsity3200_loss_L2norm곱해줌/texture_attribution/texture_attribution_ps8_blur4.0.npz",
    "ps4_blur2.0": "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/MoCo_seed87/SAE_sparsity3200_loss_L2norm곱해줌/texture_attribution/texture_attribution_ps4_blur2.0.npz",
    
}

METRIC = "l2sq"  # default; overridden by --metric arg

# Path to concept visualization directories (concept_XXXX_CLASSNAME)
CONCEPT_DIR = "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/MoCo_seed87/SAE_sparsity3200_loss_L2norm곱해줌/concept_by_gap_csv_d4096_sp3200_max"

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

    # === Merge REF into LOCAL (User requested: 12 -> 9 components) ===
    if has_rotation:
        print("  Merging _ref into _local to reduce 12 components to 9...")
        for m in ["gap", "l2sq"]:
            for ch in ["R", "G", "B"]:
                df[f"{m}_{ch}_local"] += df[f"{m}_{ch}_ref"]
        has_rotation = False
        comp_names = COMPONENT_NAMES_9

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
    in_band = ((lin > 0.7) & (lin < 1.3)).sum()
    print(f"  [0.7-1.3]: {in_band}/{len(lin)} ({in_band/len(lin)*100:.1f}%)")


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
        comp_list = list(dfs.values())[0].attrs.get("comp_names", COMPONENT_NAMES_9)

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


def export_per_neuron_csv(df, output_path, metric, label="", neuron_to_class=None):
    """Export per-neuron data to CSV for one metric.
    If neuron_to_class is provided, adds class_label column."""
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
    # Add class label if available
    if neuron_to_class:
        out_df["class_label"] = out_df["neuron_id"].map(
            lambda nid: neuron_to_class.get(int(nid), ""))
    out_df.to_csv(output_path, index=False, float_format="%.6f")
    print(f"Saved: {output_path}  ({len(df)} neurons, metric={metric})")


def export_class_summary_csv(df, neuron_to_class, output_path, metric="gap"):
    """Export per-class summary statistics (mean/median/std) for each component
    and category fraction to CSV. Each row = one class × one component."""
    comp_names = df.attrs.get("comp_names", COMPONENT_NAMES_9)
    has_rot = df.attrs.get("has_rotation", False)
    m = metric
    frac_cols = [f"{m}_{c}_frac" for c in comp_names]
    cat_defs = [
        ("interaction_frac", [f"{m}_RG_inter_frac", f"{m}_RB_inter_frac", f"{m}_GB_inter_frac"]),
        ("local_shape_frac", [f"{m}_R_local_frac", f"{m}_G_local_frac", f"{m}_B_local_frac"]),
    ]
    if has_rot:
        cat_defs.append(
            ("reference_frac", [f"{m}_R_ref_frac", f"{m}_G_ref_frac", f"{m}_B_ref_frac"])
        )
    cat_defs.append(
        ("texture_frac", [f"{m}_R_tex_frac", f"{m}_G_tex_frac", f"{m}_B_tex_frac"])
    )

    MUTATIONS = {"SNCA", "GBA", "LRRK2"}
    from collections import defaultdict
    group_nids = defaultdict(list)
    df_nid = df.set_index("neuron_id")
    for nid, raw_cls in neuron_to_class.items():
        if nid not in df_nid.index:
            continue
        parts = set(raw_cls.split("_"))
        muts = parts & MUTATIONS
        if len(muts) == 0:
            continue
        elif len(muts) == 1:
            group_nids[muts.pop()].append(nid)
        else:
            label = "_".join(sorted(muts))
            group_nids[label].append(nid)

    rows = []
    for cls in sorted(group_nids.keys()):
        nids = group_nids[cls]
        sub = df_nid.loc[nids, frac_cols].dropna(subset=frac_cols)
        if len(sub) == 0:
            continue
        n_neurons = len(sub)
        # Per-component fractions
        for i, col in enumerate(frac_cols):
            clean = comp_names[i]
            vals = sub[col].values
            rows.append({
                "class": cls, "component": clean, "type": "component",
                "n_neurons": n_neurons,
                "mean": vals.mean(), "median": np.median(vals), "std": vals.std(),
            })
        # Category-level fractions
        for cat_name, cat_cols_list in cat_defs:
            valid_cols = [c for c in cat_cols_list if c in sub.columns]
            if valid_cols:
                cat_sum = sub[valid_cols].sum(axis=1).values
                rows.append({
                    "class": cls, "component": cat_name, "type": "category",
                    "n_neurons": n_neurons,
                    "mean": cat_sum.mean(), "median": np.median(cat_sum),
                    "std": cat_sum.std(),
                })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_path, index=False, float_format="%.6f")
    print(f"Saved class summary: {output_path}  ({len(out_df)} rows)")


def export_per_neuron_with_class_csv(df, neuron_to_class, output_path, metric="gap"):
    """Export per-neuron CSV with class label, filtered to mutation-relevant neurons.
    Columns: neuron_id, class_label, mutation_group, each component frac, category fracs,
    linearity — ready for figure generation."""
    comp_names = df.attrs.get("comp_names", COMPONENT_NAMES_9)
    has_rot = df.attrs.get("has_rotation", False)
    m = metric

    MUTATIONS = {"SNCA", "GBA", "LRRK2"}
    rows = []
    for _, r in df.iterrows():
        nid = int(r["neuron_id"])
        raw_cls = neuron_to_class.get(nid, None)
        if raw_cls is None:
            continue
        parts = set(raw_cls.split("_"))
        muts = parts & MUTATIONS
        if len(muts) == 0:
            continue
        mut_group = "_".join(sorted(muts)) if len(muts) > 1 else muts.pop()

        row = {
            "neuron_id": nid,
            "class_label": raw_cls,
            "mutation_group": mut_group,
            "orig": float(r[f"{m}_orig"]),
            "baseline": float(r[f"{m}_baseline"]),
            "total_spatial": float(r[f"{m}_total_spatial"]),
            "linearity": float(r[f"{m}_linearity"]),
        }
        for c in comp_names:
            row[c] = float(r[f"{m}_{c}"])
            row[f"{c}_frac"] = float(r[f"{m}_{c}_frac"]) if pd.notna(r[f"{m}_{c}_frac"]) else np.nan
        # Category fracs
        for cat in ["interaction_frac", "local_shape_frac", "texture_frac"]:
            row[cat] = float(r[f"{m}_{cat}"]) if f"{m}_{cat}" in r.index else np.nan
        if has_rot and f"{m}_reference_frac" in r.index:
            row["reference_frac"] = float(r[f"{m}_reference_frac"])
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_path, index=False, float_format="%.6f")
    print(f"Saved per-neuron with class: {output_path}  ({len(out_df)} neurons)")


# ==============================================================================
# Mutation-Specific Decomposition Analysis
# ==============================================================================

def parse_concept_dirs(concept_dir):
    """Parse concept_XXXX_CLASS directories → {neuron_id: class_label}.
    Handles multi-class labels like 'concept_0037_GBA_SNCA' → 'GBA_SNCA'.
    """
    import re
    neuron_to_class = {}
    if not os.path.isdir(concept_dir):
        print(f"⚠ Concept dir not found: {concept_dir}")
        return neuron_to_class
    for name in os.listdir(concept_dir):
        m = re.match(r"concept_(\d+)_(.+)", name)
        if m and os.path.isdir(os.path.join(concept_dir, name)):
            nid = int(m.group(1))
            cls = m.group(2)
            neuron_to_class[nid] = cls
    print(f"  Parsed {len(neuron_to_class)} concepts from {concept_dir}")
    return neuron_to_class


def analyze_by_mutation(df, neuron_to_class, metric="gap"):
    """Per-mutation summary of 12-component fractions (signed + absolute)."""
    comp_names = df.attrs.get("comp_names", COMPONENT_NAMES_9)
    has_rot = df.attrs.get("has_rotation", False)
    m = metric
    frac_cols = [f"{m}_{c}_frac" for c in comp_names]

    # Map neuron_id → mutation group(s)
    # Rules:
    #   - pure "Control" → excluded
    #   - "Control_GBA" or "GBA_Control" → assign to "GBA"
    #   - "Control_SNCA" → assign to "SNCA"
    #   - "LRRK2_SNCA" → keep as "LRRK2_SNCA" (multi-mutation)
    #   - single mutation "SNCA" → assign to "SNCA"
    MUTATIONS = {"SNCA", "GBA", "LRRK2"}
    df_nid = df.set_index("neuron_id")

    # Build mutation → list of neuron_ids
    from collections import defaultdict
    group_nids = defaultdict(list)
    for nid, raw_cls in neuron_to_class.items():
        if nid not in df_nid.index:
            continue
        parts = set(raw_cls.split("_"))
        muts = parts & MUTATIONS
        if len(muts) == 0:
            continue  # pure Control → skip
        elif len(muts) == 1:
            group_nids[muts.pop()].append(nid)
        else:
            # Multi-mutation: keep original sorted label
            label = "_".join(sorted(muts))
            group_nids[label].append(nid)

    # Build DataFrames per group
    mutation_groups = {}
    for cls in sorted(group_nids.keys()):
        nids = group_nids[cls]
        sub = df_nid.loc[nids, frac_cols].copy()
        sub = sub.dropna(subset=frac_cols)
        if len(sub) == 0:
            continue
        mutation_groups[cls] = sub

    if not mutation_groups:
        print("  No matching neurons found for mutation analysis.")
        return

    # Clean column names for display (remove metric prefix)
    clean_names = [c.replace(f"{m}_", "").replace("_frac", "") for c in frac_cols]

    print(f"\n{'='*80}")
    print(f"  Mutation-Specific Decomposition (metric={metric})")
    print(f"{'='*80}")

    # Compute dynamic column width based on longest class label
    cls_labels = {}  # cls → "cls(n=XX)"
    for cls, sub in mutation_groups.items():
        cls_labels[cls] = f"{cls}(n={len(sub)})"
    cw = max(max(len(v) for v in cls_labels.values()), 20) + 2  # column width

    def _header():
        h = f"  {'Component':16s}"
        for cls in mutation_groups:
            h += f" | {cls_labels[cls]:>{cw}s}"
        return h

    def _sep():
        return f"  {'-'*16}" + " | " + " | ".join([f"{'─'*cw}"] * len(mutation_groups))

    # ── Signed fractions ──
    print(f"\n  ── Signed Fractions (mean / median) ──")
    print(_header())
    print(_sep())

    for i, col in enumerate(frac_cols):
        row = f"  {clean_names[i]:16s}"
        for cls, sub in mutation_groups.items():
            vals = sub[col].values
            cell = f"{vals.mean():+.4f} / {np.median(vals):+.4f}"
            row += f" | {cell:>{cw}s}"
        print(row)

    # Category-level signed
    print(f"\n  {'--- Category ---':16s}")
    cat_defs = [
        ("Interaction", [f"{m}_RG_inter_frac", f"{m}_RB_inter_frac", f"{m}_GB_inter_frac"]),
        ("Local Shape", [f"{m}_R_local_frac", f"{m}_G_local_frac", f"{m}_B_local_frac"]),
    ]
    if has_rot:
        cat_defs.append(
            ("Reference", [f"{m}_R_ref_frac", f"{m}_G_ref_frac", f"{m}_B_ref_frac"])
        )
    cat_defs.append(
        ("Texture", [f"{m}_R_tex_frac", f"{m}_G_tex_frac", f"{m}_B_tex_frac"])
    )
    for cat_name, cat_cols_list in cat_defs:
        row = f"  {cat_name:16s}"
        for cls, sub in mutation_groups.items():
            valid_cols = [c for c in cat_cols_list if c in sub.columns]
            if valid_cols:
                cat_sum = sub[valid_cols].sum(axis=1).values
                cell = f"{cat_sum.mean():+.4f} / {np.median(cat_sum):+.4f}"
                row += f" | {cell:>{cw}s}"
            else:
                row += f" | {'N/A':>{cw}s}"
        print(row)

    # ── Absolute fractions ──
    print(f"\n  ── Absolute |Fractions| (mean / median) ──")
    print(_header())
    print(_sep())

    for i, col in enumerate(frac_cols):
        row = f"  {clean_names[i]:16s}"
        for cls, sub in mutation_groups.items():
            vals = np.abs(sub[col].values)
            cell = f"{vals.mean():.4f} / {np.median(vals):.4f}"
            row += f" | {cell:>{cw}s}"
        print(row)

    # Category-level absolute
    print(f"\n  {'--- Category ---':16s}")
    for cat_name, cat_cols_list in cat_defs:
        row = f"  {cat_name:16s}"
        for cls, sub in mutation_groups.items():
            valid_cols = [c for c in cat_cols_list if c in sub.columns]
            if valid_cols:
                cat_abs = sub[valid_cols].abs().sum(axis=1).values
                cell = f"{cat_abs.mean():.4f} / {np.median(cat_abs):.4f}"
                row += f" | {cell:>{cw}s}"
            else:
                row += f" | {'N/A':>{cw}s}"
        print(row)

    # ── Signed proportions ──
    print(f"\n  ── Signed Proportions (component mean / Σmeans) ──")
    print(_header())
    print(_sep())

    signed_means = {}
    for cls, sub in mutation_groups.items():
        signed_means[cls] = np.array([sub[col].values.mean() for col in frac_cols])

    for i, col in enumerate(frac_cols):
        row = f"  {clean_names[i]:16s}"
        for cls in mutation_groups:
            denom = signed_means[cls].sum()
            if abs(denom) < 1e-12:
                row += f" | {'N/A':>{cw}s}"
            else:
                cell = f"{signed_means[cls][i] / denom:+.4f}"
                row += f" | {cell:>{cw}s}"
        print(row)

    print(f"\n  {'--- Category ---':16s}")
    for cat_name, cat_cols_list in cat_defs:
        row = f"  {cat_name:16s}"
        for cls in mutation_groups:
            valid_idx = [j for j, c in enumerate(frac_cols) if c in cat_cols_list]
            denom = signed_means[cls].sum()
            if abs(denom) < 1e-12 or not valid_idx:
                row += f" | {'N/A':>{cw}s}"
            else:
                cat_val = sum(signed_means[cls][j] for j in valid_idx)
                cell = f"{cat_val / denom:+.4f}"
                row += f" | {cell:>{cw}s}"
        print(row)

    # ── Absolute proportions ──
    print(f"\n  ── Absolute Proportions (|mean| / Σ|means|, sums to 1.0) ──")
    print(_header())
    print(_sep())

    abs_means = {}
    for cls, sub in mutation_groups.items():
        abs_means[cls] = np.array([np.abs(sub[col].values).mean() for col in frac_cols])

    for i, col in enumerate(frac_cols):
        row = f"  {clean_names[i]:16s}"
        for cls in mutation_groups:
            denom = abs_means[cls].sum()
            if denom < 1e-12:
                row += f" | {'N/A':>{cw}s}"
            else:
                cell = f"{abs_means[cls][i] / denom:.4f}"
                row += f" | {cell:>{cw}s}"
        print(row)

    print(f"\n  {'--- Category ---':16s}")
    for cat_name, cat_cols_list in cat_defs:
        row = f"  {cat_name:16s}"
        for cls in mutation_groups:
            valid_idx = [j for j, c in enumerate(frac_cols) if c in cat_cols_list]
            denom = abs_means[cls].sum()
            if denom < 1e-12 or not valid_idx:
                row += f" | {'N/A':>{cw}s}"
            else:
                cat_val = sum(abs_means[cls][j] for j in valid_idx)
                cell = f"{cat_val / denom:.4f}"
                row += f" | {cell:>{cw}s}"
        print(row)


# ==============================================================================
# Publication-Quality Figures  (SVG → Illustrator)
# ==============================================================================

def _save_svg(fig, path, dpi=300):
    """Save figure as Illustrator-optimised SVG.
    • text remains editable (svg.fonttype='none' in rcParams)
    • no rasterisation of vector elements
    • tight bounding box
    """
    fig.savefig(path, format='svg', dpi=dpi,
                bbox_inches='tight', pad_inches=0.05,
                transparent=True)
    plt.close(fig)
    print(f"  ✓ Saved SVG: {path}")


def _build_mutation_groups(df, neuron_to_class, metric):
    """Return {mutation_group: sub_df} with frac columns AND per-neuron
    absolute proportion columns (_absprop).  absprop_i = |frac_i| / Σ|frac_j|
    so each neuron's profile sums to 1.0."""
    comp_names = df.attrs.get('comp_names', COMPONENT_NAMES_9)
    m = metric
    frac_cols = [f'{m}_{c}_frac' for c in comp_names]
    absprop_cols = [f'{m}_{c}_absprop' for c in comp_names]
    MUTATIONS = {'SNCA', 'GBA', 'LRRK2'}
    from collections import defaultdict
    group_nids = defaultdict(list)
    df_nid = df.set_index('neuron_id').copy()

    # ── Compute per-neuron absolute proportions ──
    abs_mat = df_nid[frac_cols].abs()
    row_sum = abs_mat.sum(axis=1).replace(0, np.nan)
    for fc, ac in zip(frac_cols, absprop_cols):
        df_nid[ac] = abs_mat[fc] / row_sum

    for nid, raw_cls in neuron_to_class.items():
        if nid not in df_nid.index:
            continue
        parts = set(raw_cls.split('_'))
        muts = parts & MUTATIONS
        if not muts:
            continue
        label = '_'.join(sorted(muts)) if len(muts) > 1 else muts.pop()
        group_nids[label].append(nid)
    groups = {}
    all_cols = frac_cols + absprop_cols
    for cls in sorted(group_nids.keys()):
        nids = group_nids[cls]
        sub = df_nid.loc[nids, all_cols].dropna(subset=frac_cols)
        if len(sub) > 1:
            groups[cls] = sub
    return groups, frac_cols, absprop_cols, comp_names


# ==========  Figure 1: 12-component × 4-class  bar + strip  ==========

def plot_attribution_bar_with_dots(df, neuron_to_class, metric, save_dir, label=''):
    """Bar plot using Absolute Proportions: |mean| / Σ|means|.
    Bar height = mean(|frac_i|) / Σ_j mean(|frac_j|)  (group-level, sums to 1).
    Dots  = per-neuron |frac_i| / Σ|frac_j|  (individual neuron proportions).
    Total bars = 12 components × N_classes (typically 48).
    """
    groups, frac_cols, absprop_cols, comp_names = _build_mutation_groups(
        df, neuron_to_class, metric)
    if not groups:
        print('  ⚠ No mutation groups for attribution bar plot.')
        return

    class_list = list(groups.keys())
    n_comp = len(comp_names)
    n_cls  = len(class_list)

    CLASS_PALETTE = {
        'SNCA':  '#4C72B0',
        'GBA':   '#DD8452',
        'LRRK2': '#55A868',
    }
    _extra = ['#C44E52', '#8172B3', '#937860', '#DA8BC3', '#8C8C8C']
    cls_colors = [CLASS_PALETTE.get(c, _extra[i % len(_extra)])
                  for i, c in enumerate(class_list)]

    bar_width = 0.8 / n_cls
    fig, ax = plt.subplots(figsize=(max(10, n_comp * 1.1), 4.5))

    for ci, cls in enumerate(class_list):
        sub = groups[cls]

        # ── Group-level: |mean| / Σ|means| ──
        abs_means = np.array([np.abs(sub[col].values).mean() for col in frac_cols])
        denom = abs_means.sum()
        bar_vals = abs_means / denom if denom > 1e-12 else abs_means * 0

        # ── Per-neuron dots: |frac_i| / Σ|frac_j| (absprop) ──
        dot_vals = [sub[col].values for col in absprop_cols]

        x_base = np.arange(n_comp)
        offset = (ci - (n_cls - 1) / 2) * bar_width
        x_pos = x_base + offset

        ax.bar(x_pos, bar_vals, bar_width * 0.92,
               color=cls_colors[ci],
               edgecolor='white', linewidth=0.4,
               label=f'{cls} (n={len(sub)})', alpha=0.85, zorder=2)

        rng = np.random.default_rng(42 + ci)
        for j, v in enumerate(dot_vals):
            jitter = rng.uniform(-bar_width * 0.3, bar_width * 0.3, size=len(v))
            ax.scatter(x_pos[j] + jitter, v,
                       s=8, color=cls_colors[ci], edgecolors='k',
                       linewidths=0.25, alpha=0.55, zorder=3)

    ax.set_xticks(np.arange(n_comp))
    ax.set_xticklabels(comp_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Absolute Proportion  (|mean| / Σ|means|)', fontsize=9)
    ax.legend(fontsize=7, frameon=True, framealpha=0.9,
              edgecolor='#cccccc', loc='upper right')
    ax.set_title('Spatial Attribution by Mutation Class (Absolute Proportion)',
                 fontsize=10, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    svg_path = os.path.join(save_dir, f'fig_attribution_bar_dots_{label}_{metric}.svg')
    _save_svg(fig, svg_path)


# ==========  Figure 2: Linearity distribution (ALL neurons)  ==========

def plot_linearity_distribution_all(df, metric, save_dir, label=''):
    """KDE + histogram of linearity for ALL alive neurons (no linearity filter).
    X-axis is kept compact to show a smooth continuous distribution.
    """
    lin_col = f'{metric}_linearity'
    if lin_col not in df.columns:
        print('  ⚠ linearity column not found')
        return
    lin = df[lin_col].dropna().values
    if len(lin) == 0:
        return

    # Auto-range: clip extreme outliers for compact axis
    q01, q99 = np.percentile(lin, [0.5, 99.5])
    x_lo = max(0, q01 - 0.1)
    x_hi = q99 + 0.1

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    # Histogram (normalised density)
    bins = np.linspace(x_lo, x_hi, 80)
    ax.hist(lin, bins=bins, density=False,
            color='#5B9BD5', edgecolor='white', linewidth=0.3,
            alpha=0.7, zorder=1)
    # Reference line at 1.0
    ax.axvline(1.0, color='#E74C3C', linestyle='--', linewidth=1.2,
               label='Perfect (1.0)', zorder=3)

    # Annotate median
    med = np.median(lin)
    ax.axvline(med, color='#F39C12', linestyle=':', linewidth=1.0,
               label=f'Median ({med:.3f})', zorder=3)

    ax.set_xlabel('Linearity (sum of components / original)', fontsize=9)
    ax.set_ylabel('Number of neurons', fontsize=9)
    ax.set_title(f'Linearity Distribution – All {len(lin)} Neurons ({metric})',
                 fontsize=10, fontweight='bold')
    ax.set_xlim(x_lo, x_hi)
    ax.legend(fontsize=7, frameon=True, framealpha=0.9, edgecolor='#cccccc')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_major_locator(MaxNLocator(5))

    svg_path = os.path.join(save_dir, f'fig_linearity_dist_all_{label}_{metric}.svg')
    _save_svg(fig, svg_path)


# ==========  Figure 3: Category-level grouped bar + dots  ==========

def plot_category_bar_with_dots(df, neuron_to_class, metric, save_dir, label=''):
    """Category-level grouped bar using |mean| / Σ|means| formula.
    Bar  = sum of abs_means within category / Σ all abs_means.
    Dots = per-neuron absprop summed within category."""
    groups, frac_cols, absprop_cols, comp_names = _build_mutation_groups(
        df, neuron_to_class, metric)
    if not groups:
        return
    has_rot = df.attrs.get('has_rotation', False)
    m = metric

    # Category definitions using BOTH frac (for bar) and absprop (for dots)
    _fc = {c: fc for c, fc in zip(comp_names, frac_cols)}
    _ap = {c: ac for c, ac in zip(comp_names, absprop_cols)}
    cat_comp_names = [
        ('Interaction', ['RG_inter', 'RB_inter', 'GB_inter']),
        ('Local Shape', ['R_local', 'G_local', 'B_local']),
    ]
    if has_rot:
        cat_comp_names.append(('Reference', ['R_ref', 'G_ref', 'B_ref']))
    cat_comp_names.append(('Texture', ['R_tex', 'G_tex', 'B_tex']))

    cat_names = [c[0] for c in cat_comp_names]
    class_list = list(groups.keys())
    n_cat = len(cat_names)
    n_cls = len(class_list)

    CLASS_PALETTE = {'SNCA': '#4C72B0', 'GBA': '#DD8452', 'LRRK2': '#55A868'}
    _extra = ['#C44E52', '#8172B3', '#937860']
    cls_colors = [CLASS_PALETTE.get(c, _extra[i % len(_extra)])
                  for i, c in enumerate(class_list)]

    bar_w = 0.8 / n_cls
    fig, ax = plt.subplots(figsize=(max(5, n_cat * 1.8), 4.2))

    for ci, cls in enumerate(class_list):
        sub = groups[cls]

        # Group-level: abs_means per component, then category sums
        all_abs_means = np.array([np.abs(sub[fc].values).mean()
                                  for fc in frac_cols])
        total_denom = all_abs_means.sum()

        bar_vals, dot_vals = [], []
        for cat_name, cnames in cat_comp_names:
            idx = [comp_names.index(cn) for cn in cnames]
            # Bar: sum of abs_means for this category / total
            cat_bar = sum(all_abs_means[j] for j in idx) / total_denom if total_denom > 1e-12 else 0
            bar_vals.append(cat_bar)
            # Dots: per-neuron absprop summed within category
            ap_cols = [_ap[cn] for cn in cnames]
            valid = [c for c in ap_cols if c in sub.columns]
            v = sub[valid].sum(axis=1).values if valid else np.array([])
            dot_vals.append(v)

        x_base = np.arange(n_cat)
        offset = (ci - (n_cls - 1) / 2) * bar_w
        x_pos = x_base + offset

        ax.bar(x_pos, bar_vals, bar_w * 0.92,
               color=cls_colors[ci],
               edgecolor='white', linewidth=0.4,
               label=f'{cls} (n={len(sub)})', alpha=0.85, zorder=2)

        rng = np.random.default_rng(99 + ci)
        for j, v in enumerate(dot_vals):
            if len(v) == 0:
                continue
            jitter = rng.uniform(-bar_w * 0.3, bar_w * 0.3, size=len(v))
            ax.scatter(x_pos[j] + jitter, v,
                       s=10, color=cls_colors[ci], edgecolors='k',
                       linewidths=0.25, alpha=0.5, zorder=3)

    ax.set_xticks(np.arange(n_cat))
    ax.set_xticklabels(cat_names, fontsize=9)
    ax.set_ylabel('Absolute Proportion  (|mean| / Σ|means|)', fontsize=9)
    ax.legend(fontsize=7, frameon=True, framealpha=0.9, edgecolor='#cccccc')
    ax.set_title('Category-Level Attribution (Absolute Proportion)',
                 fontsize=10, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    svg_path = os.path.join(save_dir, f'fig_category_bar_dots_{label}_{metric}.svg')
    _save_svg(fig, svg_path)


# ==========  Figure 4: Heatmap of component profiles  ==========

def plot_attribution_heatmap(df, neuron_to_class, metric, save_dir, label=''):
    """Heatmap where rows = mutation classes, columns = 12 components.
    Value = |mean| / Σ|means| (group-level absolute proportion)."""
    groups, frac_cols, absprop_cols, comp_names = _build_mutation_groups(
        df, neuron_to_class, metric)
    if not groups:
        return
    class_list = list(groups.keys())
    mat = np.zeros((len(class_list), len(comp_names)))
    for i, cls in enumerate(class_list):
        sub = groups[cls]
        abs_means = np.array([np.abs(sub[col].values).mean() for col in frac_cols])
        denom = abs_means.sum()
        mat[i, :] = abs_means / denom if denom > 1e-12 else 0

    fig, ax = plt.subplots(figsize=(max(7, len(comp_names) * 0.65), max(2.5, len(class_list) * 0.7)))
    vmax = np.max(mat) * 1.05
    im = ax.imshow(mat, cmap='YlOrRd', aspect='auto',
                   vmin=0, vmax=vmax, interpolation='nearest')
    ax.set_xticks(np.arange(len(comp_names)))
    ax.set_xticklabels(comp_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(np.arange(len(class_list)))
    n_neurons = [len(groups[c]) for c in class_list]
    ax.set_yticklabels([f'{c} (n={n})' for c, n in zip(class_list, n_neurons)],
                       fontsize=9)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            txt_col = 'white' if mat[i, j] > vmax * 0.6 else 'black'
            ax.text(j, i, f'{mat[i,j]:.3f}', ha='center', va='center',
                    fontsize=6.5, color=txt_col)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('|mean| / Σ|means|', fontsize=8)
    ax.set_title('Component Attribution Profile (|mean| / Σ|means|)',
                 fontsize=10, fontweight='bold')

    svg_path = os.path.join(save_dir, f'fig_attribution_heatmap_{label}_{metric}.svg')
    _save_svg(fig, svg_path)


# ==========  Figure 5: Radar / Polar chart per class  ==========

def plot_attribution_radar(df, neuron_to_class, metric, save_dir, label=''):
    """Radar chart: |mean| / Σ|means| per mutation class."""
    groups, frac_cols, absprop_cols, comp_names = _build_mutation_groups(
        df, neuron_to_class, metric)
    if not groups:
        return
    class_list = list(groups.keys())
    n_comp = len(comp_names)
    angles = np.linspace(0, 2 * np.pi, n_comp, endpoint=False).tolist()
    angles += angles[:1]

    CLASS_PALETTE = {'SNCA': '#4C72B0', 'GBA': '#DD8452', 'LRRK2': '#55A868'}
    _extra = ['#C44E52', '#8172B3', '#937860']

    fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw=dict(polar=True))
    for ci, cls in enumerate(class_list):
        sub = groups[cls]
        abs_means = np.array([np.abs(sub[col].values).mean() for col in frac_cols])
        denom = abs_means.sum()
        props = (abs_means / denom).tolist() if denom > 1e-12 else [0]*n_comp
        props += props[:1]
        color = CLASS_PALETTE.get(cls, _extra[ci % len(_extra)])
        ax.plot(angles, props, linewidth=1.4, label=f'{cls} (n={len(sub)})',
                color=color, zorder=2)
        ax.fill(angles, props, alpha=0.12, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(comp_names, fontsize=7)
    ax.set_title('Spatial Attribution Radar (|mean| / Σ|means|)',
                 fontsize=10, fontweight='bold', pad=20)
    ax.legend(fontsize=7, loc='upper right', bbox_to_anchor=(1.3, 1.15),
              frameon=True, framealpha=0.9, edgecolor='#cccccc')

    svg_path = os.path.join(save_dir, f'fig_attribution_radar_{label}_{metric}.svg')
    _save_svg(fig, svg_path)


# ==============================================================================
# Main
# ==============================================================================
def get_args():
    p = argparse.ArgumentParser("Analyze texture attribution npz")
    p.add_argument("--metric", type=str, default="gap", choices=["gap", "l2sq"],
                   help="Primary metric for visualization (default: gap)")
    p.add_argument("--npz", type=str, nargs="*", default=None,
                   help="One or more npz file paths. If not given, uses NPZ_PATHS dict.")
    p.add_argument("--concept_dir", type=str,
                   default=CONCEPT_DIR,
                   help="Path to concept directories (concept_XXXX_CLASS)")
    p.add_argument("--linearity_min", type=float, default=0.3,
                   help="Minimum linearity to include a neuron (default: 0.0 = no filter)")
    p.add_argument("--linearity_max", type=float, default=1.3,
                   help="Maximum linearity to include a neuron (default: 999 = no filter)")
    p.add_argument("--min_spatial_frac", type=float, default=0.1,
                   help="Minimum (orig-baseline)/orig to include a neuron. "
                        "Filters out neurons with negligible spatial info (default: 0.05)")
    p.add_argument("--plot_label", type=str, default="",
                   help="Only generate publication SVGs for this label "
                        "(e.g. 'ps8_blur4.0'). Empty = all.")
    return p.parse_args()


def _apply_linearity_filter(df, metric, lo, hi):
    """Filter neurons by linearity range. Returns filtered copy."""
    lin_col = f"{metric}_linearity"
    if lin_col not in df.columns:
        return df
    before = len(df)
    mask = (df[lin_col] >= lo) & (df[lin_col] <= hi)
    df_out = df[mask].copy()
    # Preserve attrs
    for k, v in df.attrs.items():
        df_out.attrs[k] = v
    after = len(df_out)
    if before != after:
        print(f"  Linearity filter [{lo}, {hi}]: {before} → {after} neurons")
    return df_out


def _apply_spatial_frac_filter(df, metric, min_frac):
    """Filter neurons where |orig - baseline| / orig < min_frac.
    Keeps neurons with sufficient spatial effect in EITHER direction:
      - orig > baseline: spatial info contributes to activation
      - baseline > orig: spatial info suppresses activation (also valid)
    Only removes neurons where the spatial effect is negligible."""
    orig_col = f"{metric}_orig"
    baseline_col = f"{metric}_baseline"
    if orig_col not in df.columns or baseline_col not in df.columns:
        return df
    before = len(df)
    orig = df[orig_col]
    spatial = (df[orig_col] - df[baseline_col]).abs()
    # |orig - baseline| / orig >= min_frac
    safe_orig = orig.where(orig.abs() > 1e-12, 1e-12)
    ratio = spatial / safe_orig
    mask = ratio >= min_frac
    df_out = df[mask].copy()
    for k, v in df.attrs.items():
        df_out.attrs[k] = v
    after = len(df_out)
    if before != after:
        print(f"  Spatial frac filter [|orig-baseline|/orig >= {min_frac}]: "
              f"{before} → {after} neurons (removed {before - after})")
    return df_out


def main():
    global METRIC
    args = get_args()
    METRIC = args.metric
    print(f"Primary metric for visualization: {METRIC}")
    lin_lo, lin_hi = args.linearity_min, args.linearity_max
    has_lin_filter = lin_lo > 0 or lin_hi < 100
    if has_lin_filter:
        print(f"Linearity filter: [{lin_lo}, {lin_hi}]")
    min_spatial_frac = args.min_spatial_frac
    if min_spatial_frac > 0:
        print(f"Spatial frac filter: (orig-baseline)/orig >= {min_spatial_frac}")

    # Build NPZ path dict
    if args.npz:
        npz_paths = {}
        for p in args.npz:
            label = os.path.splitext(os.path.basename(p))[0]
            label = label.replace("texture_attribution_", "")
            npz_paths[label] = p
    else:
        npz_paths = NPZ_PATHS

    # ── Load ALL npz (unfiltered) first for linearity distribution ──
    dfs_raw = {}
    for label, path in npz_paths.items():
        if not os.path.exists(path):
            print(f"⚠ Not found: {path}")
            continue
        print(f"Loading: {path}")
        df_raw = load_npz(path)
        set_primary_metric(df_raw, METRIC)
        dfs_raw[label] = df_raw

    # ── Filtered copies for analysis ──
    dfs = {}
    for label, df_raw in dfs_raw.items():
        df = df_raw.copy()
        for k, v in df_raw.attrs.items():
            df.attrs[k] = v
        if has_lin_filter:
            df = _apply_linearity_filter(df, METRIC, lin_lo, lin_hi)
        if min_spatial_frac > 0:
            df = _apply_spatial_frac_filter(df, METRIC, min_spatial_frac)
        dfs[label] = df
        print_summary(df, label=label)

    # Parse concept directories for mutation-specific analysis
    concept_dir = args.concept_dir
    neuron_to_class = parse_concept_dirs(concept_dir) if os.path.isdir(concept_dir) else {}

    # Per-condition visualizations
    for label, df in dfs.items():
        path = npz_paths[label]
        save_dir = os.path.dirname(path)

        plot_stacked_bar(df, label=label, top_n=50, sort_by="orig")
        plot_linearity_hist(df, label=label)
        plot_category_scatter(df, label=label)

        # ═══  Publication SVG figures  ═══
        plot_label = args.plot_label
        if plot_label and label != plot_label:
            print(f"  Skipping SVG for '{label}' (--plot_label={plot_label})")
        else:
            # Fig 2: Linearity distribution — ALL neurons (unfiltered)
            df_raw = dfs_raw[label]
            plot_linearity_distribution_all(df_raw, metric=METRIC,
                                            save_dir=save_dir, label=label)

            if neuron_to_class:
                # Fig 1: 12-comp × class  bar + dots
                plot_attribution_bar_with_dots(
                    df, neuron_to_class, metric=METRIC,
                    save_dir=save_dir, label=label)
                # Fig 3: Category-level bar + dots
                plot_category_bar_with_dots(
                    df, neuron_to_class, metric=METRIC,
                    save_dir=save_dir, label=label)
                # Fig 4: Heatmap
                plot_attribution_heatmap(
                    df, neuron_to_class, metric=METRIC,
                    save_dir=save_dir, label=label)
                # Fig 5: Radar chart
                plot_attribution_radar(
                    df, neuron_to_class, metric=METRIC,
                    save_dir=save_dir, label=label)

        # Export CSV — one per metric
        for m in ["gap", "l2sq"]:
            csv_path = path.replace(".npz", f"_{m}_per_neuron.csv")
            export_per_neuron_csv(df, csv_path, metric=m, label=label,
                                 neuron_to_class=neuron_to_class)

        # Mutation-specific analysis + CSV export
        if neuron_to_class:
            for m in ["gap", "l2sq"]:
                print(f"\n  ─── Mutation analysis for {label} (metric={m}) ───")
                analyze_by_mutation(df, neuron_to_class, metric=m)
                summary_csv = path.replace(
                    ".npz", f"_{m}_class_summary.csv")
                export_class_summary_csv(
                    df, neuron_to_class, summary_csv, metric=m)
                neuron_cls_csv = path.replace(
                    ".npz", f"_{m}_per_neuron_with_class.csv")
                export_per_neuron_with_class_csv(
                    df, neuron_to_class, neuron_cls_csv, metric=m)

    # Cross-condition comparison
    if len(dfs) >= 2:
        compare_conditions(dfs)


if __name__ == "__main__":
    main()

