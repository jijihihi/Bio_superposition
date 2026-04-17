# ==============================================================================
# Cumulative Variance Plot — CNN vs SAE PCA 차원별 누적 분산
#
# effective_rank.py가 저장한 JSON의 cumulative_variance를 읽어서
# PCA component 개수에 따른 누적 분산 비율을 시각화
#
# Usage (Colab):
#   import sys
#   sys.argv = [
#       "plot_cumvar",
#       "--base_dir", "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/effective_rank",
#   ]
#   from apoptosis_prediction.plot_cumvar import main
#   main()
#
# Usage (terminal):
#   python -m apoptosis_prediction.plot_cumvar \
#       --base_dir "/content/drive/.../effective_rank" \
#       --condition raw \
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
import seaborn as sns

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'
sns.set_style("ticks")

COLORS = {"CNN": "#4C72B0", "SAE": "#DD8452"}
MUTATIONS = ["SNCA", "GBA", "LRRK2"]
MUT_COLORS = {"SNCA": "#E24A33", "GBA": "#348ABD", "LRRK2": "#988ED5"}

# Condition → (subdir_in_base, json_condition_key, json_filter_key)
CONDITION_MAP = {
    "raw":      ("raw",       "raw",      "unfiltered"),
    "pca50":    ("pca50",     "pca",      "unfiltered"),
    "pca_std":  ("pca50_std", "norm_pca", "unfiltered"),
}


# ==============================================================================
# Args
# ==============================================================================
def get_args():
    p = argparse.ArgumentParser(
        description="Plot cumulative variance explained by PCA components (CNN vs SAE)")
    p.add_argument("--base_dir", type=str, required=True,
                   help="Base directory containing effective_rank subdirectories")
    p.add_argument("--condition", type=str, default="raw",
                   choices=["raw", "pca50", "pca_std", "all"],
                   help="Which condition to plot. 'all' = overlay all.")
    p.add_argument("--max_components", type=int, default=100,
                   help="Max PCA components to show on x-axis (0 = all)")
    p.add_argument("--output_dir", type=str, default="")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--separate_mutations", action="store_true",
                   help="Plot each mutation separately instead of averaging")
    return p.parse_args()


# ==============================================================================
# Scan & collect cumvar from JSONs
# ==============================================================================
def scan_cumvar(base_dir, subdir, cond_key, filter_key):
    """
    Scan JSON files and extract cumulative_variance arrays.
    Returns: dict[source][mutation] = list of np.array (one per seed)
    """
    search_dir = os.path.join(base_dir, subdir)
    pattern = os.path.join(search_dir, "**", "effective_rank_results.json")
    result = defaultdict(lambda: defaultdict(list))

    for jpath in glob.glob(pattern, recursive=True):
        try:
            with open(jpath, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            continue

        for entry in data.get("results", []):
            if entry.get("condition") != cond_key:
                continue
            if entry.get("filter") != filter_key:
                continue
            cumvar = entry.get("cumulative_variance", [])
            if len(cumvar) == 0:
                continue

            result[entry["source"]][entry["mutation"]].append(np.array(cumvar))

    return result


# ==============================================================================
# Compute mean ± SEM across seeds (aligning to shortest length)
# ==============================================================================
def aggregate_cumvar(cumvar_list, max_components=0):
    """
    Given a list of cumvar arrays (possibly different lengths),
    truncate to the shortest (or max_components), compute mean ± sem.
    """
    if not cumvar_list:
        return None, None, None, 0

    min_len = min(len(c) for c in cumvar_list)
    if max_components > 0:
        min_len = min(min_len, max_components)

    aligned = np.array([c[:min_len] for c in cumvar_list])
    mean = aligned.mean(axis=0)
    sem = aligned.std(axis=0, ddof=1) / np.sqrt(len(aligned)) if len(aligned) > 1 else np.zeros(min_len)
    return mean, sem, min_len, len(aligned)


# ==============================================================================
# Plot: CNN vs SAE cumulative variance (averaged across mutations)
# ==============================================================================
def plot_cumvar_averaged(cumvar_data, cond_label, out_dir, max_comp, dpi):
    """
    Single panel: CNN vs SAE, averaged across all mutations.
    Shaded region = ± SEM across seeds × mutations.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    for source in ["CNN", "SAE"]:
        # Pool all cumvar arrays across mutations
        all_curves = []
        for mut in MUTATIONS:
            all_curves.extend(cumvar_data.get(source, {}).get(mut, []))

        if not all_curves:
            continue

        mean, sem, n_comp, n_seeds = aggregate_cumvar(all_curves, max_comp)
        if mean is None:
            continue

        x = np.arange(1, n_comp + 1)
        ax.plot(x, mean, "-", color=COLORS[source], linewidth=2.0,
                label=f"{source} (n={n_seeds})", zorder=3)
        ax.fill_between(x, mean - sem, np.minimum(mean + sem, 1.0),
                         color=COLORS[source], alpha=0.15, zorder=1)

    # Reference lines
    for threshold, ls in [(0.90, ":"), (0.95, "--"), (0.99, "-.")]:
        ax.axhline(threshold, color="gray", linewidth=0.8, linestyle=ls, alpha=0.5)
        ax.text(ax.get_xlim()[1] * 0.02, threshold + 0.005,
                f"{threshold:.0%}", fontsize=8, color="gray", va="bottom")

    # Mark where 95% is reached
    for source in ["CNN", "SAE"]:
        all_curves = []
        for mut in MUTATIONS:
            all_curves.extend(cumvar_data.get(source, {}).get(mut, []))
        mean, _, n_comp, _ = aggregate_cumvar(all_curves, max_comp)
        if mean is not None:
            idx_95 = np.searchsorted(mean, 0.95)
            if idx_95 < len(mean):
                ax.axvline(idx_95 + 1, color=COLORS[source], linewidth=1,
                           linestyle=":", alpha=0.6)
                ax.annotate(f"{idx_95 + 1}",
                            xy=(idx_95 + 1, 0.95),
                            xytext=(idx_95 + 1 + max(n_comp * 0.03, 2), 0.92),
                            fontsize=9, fontweight="bold",
                            color=COLORS[source],
                            arrowprops=dict(arrowstyle="->",
                                            color=COLORS[source], lw=1.2))

    ax.set_xlabel("Number of PCA Components", fontsize=12)
    ax.set_ylabel("Cumulative Variance Explained", fontsize=12)
    ax.set_title(f"Cumulative Variance — CNN vs SAE\n({cond_label})",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(1, None)
    ax.legend(fontsize=10, loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.15)
    sns.despine()
    fig.tight_layout()

    safe = cond_label.lower().replace(" ", "_").replace("(", "").replace(")", "")
    fname = f"cumvar_{safe}"
    for ext in [".png", ".svg", ".pdf"]:
        fig.savefig(os.path.join(out_dir, fname + ext),
                    dpi=dpi, bbox_inches="tight")
    print(f"  Saved: {fname}.png/.svg/.pdf")

    if _IN_COLAB:
        plt.show()
    plt.close(fig)


# ==============================================================================
# Plot: Per-mutation panels
# ==============================================================================
def plot_cumvar_per_mutation(cumvar_data, cond_label, out_dir, max_comp, dpi):
    """
    3-panel figure (SNCA, GBA, LRRK2), each showing CNN vs SAE.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for ax, mut in zip(axes, MUTATIONS):
        for source in ["CNN", "SAE"]:
            curves = cumvar_data.get(source, {}).get(mut, [])
            if not curves:
                continue
            mean, sem, n_comp, n_seeds = aggregate_cumvar(curves, max_comp)
            if mean is None:
                continue

            x = np.arange(1, n_comp + 1)
            ax.plot(x, mean, "-", color=COLORS[source], linewidth=2.0,
                    label=f"{source} (n={n_seeds})", zorder=3)
            ax.fill_between(x, mean - sem, np.minimum(mean + sem, 1.0),
                             color=COLORS[source], alpha=0.15, zorder=1)

            # 95% marker
            idx_95 = np.searchsorted(mean, 0.95)
            if idx_95 < len(mean):
                ax.axvline(idx_95 + 1, color=COLORS[source], linewidth=1,
                           linestyle=":", alpha=0.6)
                y_offset = 0.88 if source == "CNN" else 0.84
                ax.text(idx_95 + 1 + 1, y_offset,
                        f"95%@{idx_95 + 1}",
                        fontsize=8, fontweight="bold", color=COLORS[source])

        for threshold in [0.90, 0.95, 0.99]:
            ax.axhline(threshold, color="gray", linewidth=0.6,
                       linestyle="--", alpha=0.4)

        ax.set_title(mut, fontsize=13, fontweight="bold",
                     color=MUT_COLORS.get(mut, "black"))
        ax.set_xlabel("PCA Components", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(1, None)
        ax.grid(True, alpha=0.12)
        ax.legend(fontsize=9, loc="lower right", framealpha=0.9)

    axes[0].set_ylabel("Cumulative Variance Explained", fontsize=12)

    fig.suptitle(f"Cumulative Variance by Mutation — CNN vs SAE ({cond_label})",
                 fontsize=14, fontweight="bold", y=1.02)
    sns.despine()
    fig.tight_layout()

    safe = cond_label.lower().replace(" ", "_").replace("(", "").replace(")", "")
    fname = f"cumvar_per_mut_{safe}"
    for ext in [".png", ".svg", ".pdf"]:
        fig.savefig(os.path.join(out_dir, fname + ext),
                    dpi=dpi, bbox_inches="tight")
    print(f"  Saved: {fname}.png/.svg/.pdf")

    if _IN_COLAB:
        plt.show()
    plt.close(fig)


# ==============================================================================
# Print summary table
# ==============================================================================
def print_summary(cumvar_data, cond_label):
    """Print where 90/95/99% variance is reached."""
    print(f"\n  === {cond_label} — Components needed for %%var ===")
    print(f"  {'Source':6s} {'Mut':6s} {'90%':>6s} {'95%':>6s} {'99%':>6s} {'N seeds':>8s}")
    print("  " + "-" * 45)

    for source in ["CNN", "SAE"]:
        for mut in MUTATIONS:
            curves = cumvar_data.get(source, {}).get(mut, [])
            if not curves:
                continue
            mean, _, _, n_seeds = aggregate_cumvar(curves, max_components=0)
            if mean is None:
                continue
            n90 = int(np.searchsorted(mean, 0.90)) + 1
            n95 = int(np.searchsorted(mean, 0.95)) + 1
            n99 = int(np.searchsorted(mean, 0.99)) + 1
            # Clamp to max available
            n90 = min(n90, len(mean))
            n95 = min(n95, len(mean))
            n99 = min(n99, len(mean))
            print(f"  {source:6s} {mut:6s} {n90:>6d} {n95:>6d} {n99:>6d} {n_seeds:>8d}")


# ==============================================================================
# Main
# ==============================================================================
def main():
    args = get_args()
    out_dir = args.output_dir or os.path.join(args.base_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Base: {args.base_dir}")
    print(f"Output: {out_dir}\n")

    if args.condition == "all":
        conds_to_plot = list(CONDITION_MAP.keys())
    else:
        conds_to_plot = [args.condition]

    for cond_name in conds_to_plot:
        if cond_name not in CONDITION_MAP:
            print(f"  ⚠ Unknown condition: {cond_name}")
            continue

        subdir, cond_key, filter_key = CONDITION_MAP[cond_name]
        cond_label = {
            "raw": "Raw (no PCA, no norm)",
            "pca50": "PCA 50 (no norm)",
            "pca_std": "std + PCA 50",
        }.get(cond_name, cond_name)

        print(f"\n{'='*60}")
        print(f"  Condition: {cond_label}")
        print(f"{'='*60}")

        cumvar_data = scan_cumvar(args.base_dir, subdir, cond_key, filter_key)

        # Count
        n_cnn = sum(len(v) for v in cumvar_data.get("CNN", {}).values())
        n_sae = sum(len(v) for v in cumvar_data.get("SAE", {}).values())
        print(f"  CNN curves: {n_cnn}, SAE curves: {n_sae}")

        if n_cnn == 0 and n_sae == 0:
            print(f"  ⚠ No cumulative_variance data found. "
                  f"Re-run effective_rank.py to generate it.")
            continue

        # Summary table
        print_summary(cumvar_data, cond_label)

        # Plots
        plot_cumvar_averaged(cumvar_data, cond_label, out_dir,
                             args.max_components, args.dpi)

        if args.separate_mutations:
            plot_cumvar_per_mutation(cumvar_data, cond_label, out_dir,
                                     args.max_components, args.dpi)


if __name__ == "__main__":
    main()
