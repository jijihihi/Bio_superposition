# ==============================================================================
# Erank Bar Plot — 4 separate panels, CNN vs SAE × 3 mutations
#
# Each condition gets its own figure with independent y-axis.
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

COLORS = {"CNN": "#4C72B0", "SAE": "#DD8452"}

# 4 conditions: (label, subdir, json_condition_key, json_filter_key)
CONDITIONS = [
    ("Raw (no PCA)",       "raw",       "raw",      "unfiltered"),
    ("PCA 50",             "pca50",     "pca",      "unfiltered"),
    ("std + PCA 50",       "pca50_std", "norm_pca", "unfiltered"),
    ("DE (log₂FC≥1) + PCA 50", "de_sweep", "pca",  "filtered"),
]


def get_args():
    p = argparse.ArgumentParser(
        description="Erank bar plot per condition")
    p.add_argument("--base_dir", type=str, required=True)
    p.add_argument("--de_log2fc", type=float, default=1.0)
    p.add_argument("--output_dir", type=str, default="")
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args()


def scan_jsons(base_dir, subdir, de_log2fc=None):
    search_dir = os.path.join(base_dir, subdir)
    pattern = os.path.join(search_dir, "**", "effective_rank_results.json")
    all_data = []
    for jpath in glob.glob(pattern, recursive=True):
        if subdir == "de_sweep" and de_log2fc is not None:
            parts = jpath.replace("\\", "/").split("/")
            log2fc_val = None
            for part in parts:
                m = re.match(r"log2fc_([\d.]+)", part)
                if m:
                    log2fc_val = float(m.group(1))
                    break
            if log2fc_val is None or abs(log2fc_val - de_log2fc) > 0.01:
                continue
        try:
            with open(jpath, "r") as f:
                all_data.append(json.load(f))
        except (json.JSONDecodeError, FileNotFoundError):
            continue
    return all_data


def extract_erank(json_list, cond_key, filter_key):
    """Extract erank values, filtered by condition + filter tag."""
    result = defaultdict(lambda: defaultdict(list))
    for data in json_list:
        for entry in data.get("results", []):
            if entry.get("condition") != cond_key:
                continue
            if entry.get("filter") != filter_key:
                continue
            result[entry["source"]][entry["mutation"]].append(entry["erank"])
    return result


def plot_single_condition(erank_data, cond_label, mutations, out_dir, dpi):
    """One figure per condition: grouped bar CNN vs SAE for 3 mutations."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    bar_width = 0.35
    x = np.arange(len(mutations))
    rng = np.random.RandomState(42)

    for i, source in enumerate(["CNN", "SAE"]):
        offset = -bar_width / 2 if source == "CNN" else bar_width / 2
        means, sems = [], []
        scatter_x, scatter_y = [], []

        for j, mut in enumerate(mutations):
            vals = np.array(erank_data.get(source, {}).get(mut, []))
            if len(vals) > 0:
                means.append(np.mean(vals))
                sems.append(np.std(vals) / max(np.sqrt(len(vals)), 1))
                jitter = rng.uniform(-bar_width * 0.2, bar_width * 0.2,
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
               color=COLORS[source], alpha=0.7,
               edgecolor="white", linewidth=0.8,
               label=source, zorder=2)
        ax.errorbar(x + offset, means, yerr=sems,
                    fmt="none", ecolor="black", elinewidth=1.2,
                    capsize=4, capthick=1.2, zorder=3)
        ax.scatter(scatter_x, scatter_y,
                   color=COLORS[source], alpha=0.55, s=28,
                   edgecolors="black", linewidths=0.4,
                   marker="o" if source == "SAE" else "s",
                   zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(mutations, fontsize=12)
    ax.set_ylabel("Effective Rank", fontsize=12)
    ax.set_title(cond_label, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.8)
    ax.grid(True, alpha=0.12, axis="y")
    ax.set_ylim(bottom=0)
    sns.despine()
    fig.tight_layout()

    safe_name = (cond_label.lower()
                 .replace(" ", "_").replace("(", "").replace(")", "")
                 .replace("₂", "2").replace("≥", "ge").replace("+", ""))
    fname = f"erank_bar_{safe_name}"
    for ext in [".png", ".svg"]:
        fig.savefig(os.path.join(out_dir, fname + ext),
                    dpi=dpi, bbox_inches="tight")
    print(f"  Saved: {fname}.png/.svg")

    if _IN_COLAB:
        plt.show()
    plt.close(fig)


def main():
    args = get_args()
    out_dir = args.output_dir or os.path.join(args.base_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)
    mutations = ["SNCA", "GBA", "LRRK2"]

    print(f"Base: {args.base_dir}\n")

    for cond_label, subdir, cond_key, filter_key in CONDITIONS:
        de_lfc = args.de_log2fc if subdir == "de_sweep" else None
        jsons = scan_jsons(args.base_dir, subdir, de_log2fc=de_lfc)
        erank_data = extract_erank(jsons, cond_key, filter_key)

        # Count
        n_cnn = sum(len(v) for v in erank_data.get("CNN", {}).values())
        n_sae = sum(len(v) for v in erank_data.get("SAE", {}).values())
        print(f"[{cond_label}]  JSONs={len(jsons)}  "
              f"CNN points={n_cnn}  SAE points={n_sae}")

        # Summary
        for source in ["CNN", "SAE"]:
            for mut in mutations:
                vals = erank_data.get(source, {}).get(mut, [])
                if vals:
                    print(f"  {source} {mut}: {np.mean(vals):.2f} ± "
                          f"{np.std(vals)/max(np.sqrt(len(vals)),1):.2f}  "
                          f"(n={len(vals)})")

        if n_cnn > 0 or n_sae > 0:
            plot_single_condition(erank_data, cond_label, mutations,
                                 out_dir, args.dpi)
        else:
            print(f"  ⚠ No data, skipping plot")
        print()

    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
