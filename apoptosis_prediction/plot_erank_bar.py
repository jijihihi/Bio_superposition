# ==============================================================================
# Erank Bar Plot — 3 CNN layers + SAE, 3 conditions (Raw / PCA250 / PCA250+std)
#
# Handles both old (cnn_seed_X/sae_seed_Y nested) and new directory structures.
# Deduplicates SAE results that were duplicated across CNN seeds.
#
# Usage:
#   python -m apoptosis_prediction.plot_erank_bar \
#       --base_dir "/content/drive/.../effective_rank" \
#       --output_dir "/content/drive/.../effective_rank/plots"
# ==============================================================================

import os
import sys
import json
import argparse
import glob
import re
import numpy as np
from collections import defaultdict

import matplotlib
_IN_COLAB = "google.colab" in sys.modules
if not _IN_COLAB:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'
sns.set_style("ticks")

# ── Visual config ──
COLORS = {
    "stage5_mid": "#88BEDC",     # soft sky blue
    "stage5_out": "#3A7EBF",     # medium blue
    "refine_out": "#1B4876",     # deep navy
    "SAE":        "#E8833A",     # warm orange
}

MARKERS = {
    "stage5_mid": "D",
    "stage5_out": "s",
    "refine_out": "^",
    "SAE":        "o",
}

DISPLAY_NAMES = {
    "stage5_mid": "CNN stage5_mid",
    "stage5_out": "CNN stage5_out",
    "refine_out": "CNN refine_out",
    "SAE":        "SAE (stage5_out)",
}

# 3 conditions only
CONDITIONS = [
    ("Raw (no PCA)",   "raw",       "raw",      "unfiltered"),
    ("PCA 250",        "pca50",     "pca",      "unfiltered"),
    ("PCA 250 (std)",  "pca50_std", "norm_pca", "unfiltered"),
]

CNN_LAYERS = ["stage5_mid", "stage5_out", "refine_out"]
GROUP_KEYS = CNN_LAYERS + ["SAE"]


def get_args():
    p = argparse.ArgumentParser(
        description="Erank bar plot — 3 CNN layers + SAE, 3 conditions")
    p.add_argument("--base_dir", type=str, required=True,
                   help="Root effective_rank directory")
    p.add_argument("--output_dir", type=str, default="")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--mutations", type=str, nargs="+",
                   default=["SNCA", "GBA", "LRRK2"])
    return p.parse_args()


# ==============================================================================
# JSON scanning — handles both old and new directory structures
# ==============================================================================
def scan_jsons_for_layer(base_dir, layer, subdir):
    """Scan {base_dir}/{layer}/{subdir}/**/effective_rank_results.json"""
    search_dir = os.path.join(base_dir, layer, subdir)
    pattern = os.path.join(search_dir, "**", "effective_rank_results.json")
    results = []
    for jpath in glob.glob(pattern, recursive=True):
        try:
            with open(jpath, "r") as f:
                results.append((jpath, json.load(f)))
        except (json.JSONDecodeError, FileNotFoundError):
            continue
    return results


def extract_erank_values(json_list, cond_key, filter_key, source_filter=None):
    """Extract erank values from JSON results.

    Returns: {mutation: [erank_values]}
    """
    result = defaultdict(list)
    for data in json_list:
        for entry in data.get("results", []):
            if entry.get("condition") != cond_key:
                continue
            if entry.get("filter") != filter_key:
                continue
            if source_filter and entry.get("source") != source_filter:
                continue
            result[entry["mutation"]].append(entry["erank"])
    return result


def extract_sae_erank_deduped(json_with_paths, cond_key, filter_key):
    """Extract SAE erank values, deduplicating across CNN seeds.

    Old structure: .../cnn_seed_42/sae_seed_48/sae/effective_rank_results.json
                   .../cnn_seed_87/sae_seed_48/sae/effective_rank_results.json  ← duplicate
    We group by sae_seed and take only one value per (sae_seed, mutation).
    """
    # Group by sae_seed
    sae_seed_data = defaultdict(list)  # sae_seed -> [json_data]
    for jpath, jdata in json_with_paths:
        # Extract sae_seed from path
        parts = jpath.replace("\\", "/").split("/")
        sae_seed = None
        for part in parts:
            m = re.match(r"sae_seed_(\d+)", part)
            if m:
                sae_seed = m.group(1)
                break
        if sae_seed is None:
            # New structure without cnn_seed nesting
            sae_seed_data["_all"].append(jdata)
        else:
            sae_seed_data[sae_seed].append(jdata)

    # Take first occurrence per sae_seed (all duplicates are identical)
    result = defaultdict(list)
    for sae_seed, jlist in sae_seed_data.items():
        # Use only the first JSON per sae_seed (deduplicate)
        jdata = jlist[0]
        for entry in jdata.get("results", []):
            if entry.get("condition") != cond_key:
                continue
            if entry.get("filter") != filter_key:
                continue
            if entry.get("source") != "SAE":
                continue
            result[entry["mutation"]].append(entry["erank"])

    return result


def collect_all_erank(base_dir, subdir, cond_key, filter_key):
    """Collect erank for all 3 CNN layers + SAE.

    Returns: {group_key: {mutation: [values]}}
    """
    all_erank = {}

    # CNN layers
    for layer in CNN_LAYERS:
        json_with_paths = scan_jsons_for_layer(base_dir, layer, subdir)
        json_data = [jd for _, jd in json_with_paths]
        cnn_vals = extract_erank_values(json_data, cond_key, filter_key,
                                        source_filter="CNN")
        if cnn_vals:
            all_erank[layer] = dict(cnn_vals)

    # SAE (only under stage5_out, deduplicated)
    sae_jsons = scan_jsons_for_layer(base_dir, "stage5_out", subdir)
    # Filter to SAE paths only (contain "sae" in path)
    sae_jsons_only = [(p, d) for p, d in sae_jsons
                      if "/sae/" in p.replace("\\", "/")]
    if sae_jsons_only:
        sae_vals = extract_sae_erank_deduped(sae_jsons_only, cond_key,
                                             filter_key)
        if sae_vals:
            all_erank["SAE"] = dict(sae_vals)

    return all_erank


# ==============================================================================
# Plotting
# ==============================================================================
def plot_single_condition(all_erank, cond_label, mutations, out_dir, dpi):
    """One figure per condition: grouped bar for 3 CNN layers + SAE."""
    n_groups = len(GROUP_KEYS)
    n_muts = len(mutations)

    fig, ax = plt.subplots(figsize=(6.5, 5.0))

    bar_width = 0.18
    x = np.arange(n_muts)
    rng = np.random.RandomState(42)

    for gi, gkey in enumerate(GROUP_KEYS):
        offset = (gi - (n_groups - 1) / 2) * bar_width
        color = COLORS[gkey]
        marker = MARKERS[gkey]
        display = DISPLAY_NAMES[gkey]

        means, sems = [], []
        scatter_x, scatter_y = [], []

        for j, mut in enumerate(mutations):
            vals = np.array(all_erank.get(gkey, {}).get(mut, []))
            if len(vals) > 0:
                means.append(np.mean(vals))
                sems.append(np.std(vals) / max(np.sqrt(len(vals)), 1))
                jitter = rng.uniform(-bar_width * 0.3, bar_width * 0.3,
                                     len(vals))
                for v, jt in zip(vals, jitter):
                    scatter_x.append(j + offset + jt)
                    scatter_y.append(v)
            else:
                means.append(0)
                sems.append(0)

        means = np.array(means)
        sems = np.array(sems)

        ax.bar(x + offset, means, bar_width,
               color=color, alpha=0.78,
               edgecolor="white", linewidth=0.8,
               label=display, zorder=2)
        ax.errorbar(x + offset, means, yerr=sems,
                    fmt="none", ecolor="#333333", elinewidth=1.2,
                    capsize=3.5, capthick=1.0, zorder=3)
        ax.scatter(scatter_x, scatter_y,
                   color=color, alpha=0.55, s=24,
                   edgecolors="black", linewidths=0.4,
                   marker=marker, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(mutations, fontsize=13, fontweight="bold")
    ax.set_ylabel("Effective Rank", fontsize=13)
    ax.set_title(cond_label, fontsize=14, fontweight="bold", pad=10)
    ax.legend(fontsize=9, framealpha=0.85, loc="best")
    ax.grid(True, alpha=0.15, axis="y")
    ax.set_ylim(bottom=0)
    sns.despine()
    fig.tight_layout()

    safe_name = (cond_label.lower()
                 .replace(" ", "_").replace("(", "").replace(")", "")
                 .replace("+", ""))
    fname = f"erank_bar_layers_{safe_name}"
    for ext in [".png", ".svg"]:
        fig.savefig(os.path.join(out_dir, fname + ext),
                    dpi=dpi, bbox_inches="tight")
    print(f"  Saved: {fname}.png/.svg")

    if _IN_COLAB:
        plt.show()
    plt.close(fig)


def plot_combined_3panel(all_cond_data, mutations, out_dir, dpi):
    """3-panel figure: one subplot per condition."""
    n_panels = len(all_cond_data)
    n_groups = len(GROUP_KEYS)
    n_muts = len(mutations)
    bar_width = 0.18
    x = np.arange(n_muts)

    fig, axes = plt.subplots(1, n_panels, figsize=(6.0 * n_panels, 5.2),
                             sharey=False)
    if n_panels == 1:
        axes = [axes]

    for ax_idx, (cond_label, all_erank) in enumerate(all_cond_data):
        ax = axes[ax_idx]
        rng = np.random.RandomState(42)

        for gi, gkey in enumerate(GROUP_KEYS):
            offset = (gi - (n_groups - 1) / 2) * bar_width
            color = COLORS[gkey]
            marker = MARKERS[gkey]
            display = DISPLAY_NAMES[gkey]

            means, sems = [], []
            scatter_x, scatter_y = [], []

            for j, mut in enumerate(mutations):
                vals = np.array(all_erank.get(gkey, {}).get(mut, []))
                if len(vals) > 0:
                    means.append(np.mean(vals))
                    sems.append(np.std(vals) / max(np.sqrt(len(vals)), 1))
                    jitter = rng.uniform(-bar_width * 0.3, bar_width * 0.3,
                                         len(vals))
                    for v, jt in zip(vals, jitter):
                        scatter_x.append(j + offset + jt)
                        scatter_y.append(v)
                else:
                    means.append(0)
                    sems.append(0)

            means = np.array(means)
            sems = np.array(sems)

            ax.bar(x + offset, means, bar_width,
                   color=color, alpha=0.78,
                   edgecolor="white", linewidth=0.8,
                   label=display if ax_idx == 0 else "",
                   zorder=2)
            ax.errorbar(x + offset, means, yerr=sems,
                        fmt="none", ecolor="#333333", elinewidth=1.0,
                        capsize=2.5, capthick=0.8, zorder=3)
            ax.scatter(scatter_x, scatter_y,
                       color=color, alpha=0.5, s=18,
                       edgecolors="black", linewidths=0.3,
                       marker=marker, zorder=4)

        ax.set_xticks(x)
        ax.set_xticklabels(mutations, fontsize=12, fontweight="bold")
        ax.set_title(cond_label, fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.12, axis="y")
        ax.set_ylim(bottom=0)
        if ax_idx == 0:
            ax.set_ylabel("Effective Rank", fontsize=13)
        sns.despine(ax=ax)

    # Shared legend at top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               ncol=n_groups, fontsize=10, framealpha=0.9,
               bbox_to_anchor=(0.5, 1.02))

    fig.tight_layout(rect=[0, 0, 1, 0.93])

    fname = "erank_bar_3panel_layers"
    for ext in [".png", ".svg"]:
        fig.savefig(os.path.join(out_dir, fname + ext),
                    dpi=dpi, bbox_inches="tight")
    print(f"  Saved: {fname}.png/.svg")

    if _IN_COLAB:
        plt.show()
    plt.close(fig)


# ==============================================================================
# Main
# ==============================================================================
def main():
    args = get_args()
    out_dir = args.output_dir or os.path.join(args.base_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)
    mutations = args.mutations

    print(f"Base: {args.base_dir}")
    print(f"Layers: {CNN_LAYERS}")
    print(f"Groups: {GROUP_KEYS}")
    print(f"Conditions: {[c[0] for c in CONDITIONS]}")
    print(f"Mutations: {mutations}\n")

    all_cond_data = []

    for cond_label, subdir, cond_key, filter_key in CONDITIONS:
        print(f"{'─'*60}")
        print(f"  [{cond_label}]")
        print(f"{'─'*60}")

        all_erank = collect_all_erank(args.base_dir, subdir, cond_key,
                                      filter_key)

        # Summary
        for gk in GROUP_KEYS:
            for mut in mutations:
                vals = all_erank.get(gk, {}).get(mut, [])
                if vals:
                    print(f"  {DISPLAY_NAMES[gk]:22s} {mut}: "
                          f"{np.mean(vals):7.2f} ± "
                          f"{np.std(vals)/max(np.sqrt(len(vals)),1):.2f}  "
                          f"(n={len(vals)})")

        has_data = any(
            len(all_erank.get(gk, {}).get(m, [])) > 0
            for gk in GROUP_KEYS for m in mutations
        )

        if has_data:
            plot_single_condition(all_erank, cond_label, mutations,
                                  out_dir, args.dpi)
            all_cond_data.append((cond_label, all_erank))
        else:
            print(f"  ⚠ No data found for this condition")
        print()

    # Combined 3-panel
    if len(all_cond_data) > 1:
        print("Generating combined 3-panel plot...")
        plot_combined_3panel(all_cond_data, mutations, out_dir, args.dpi)

    print(f"\nAll plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
