# ==============================================================================
# Layer Comparison R² Bar Plot
#
# Reads aggregated_r2_all_folds.csv and plots:
#   - Grouped bar chart: x = Gene (SNCA, GBA, LRRK2)
#   - Bars grouped by Layer (stage5_mid, stage5_out, refine_out)
#   - 95% CI error bars
#   - Pairwise Wilcoxon signed-rank test between layers (paired by seed × fold)
#   - Significance brackets with * notation
#
# Usage (Colab):
#   import sys
#   sys.argv = [
#       "plot_layer_comparison",
#       "--results_dir", "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/apoptosis_r2_results",
#   ]
#   from apoptosis_prediction.plot_layer_comparison import main
#   main()
# ==============================================================================

import os
import sys
import csv
import argparse
from collections import defaultdict
from itertools import combinations

import numpy as np

import matplotlib
if "google.colab" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from scipy import stats
from scipy.stats import wilcoxon


# ==============================================================================
# Constants
# ==============================================================================
GROUPS_OF_INTEREST = ["SNCA only", "GBA only", "LRRK2 only"]
GENE_LABELS = {"SNCA only": "SNCA", "GBA only": "GBA", "LRRK2 only": "LRRK2"}
LAYERS = ["stage5_mid", "stage5_out", "refine_out"]
LAYER_LABELS = {
    "stage5_mid": "Stage5 Mid",
    "stage5_out": "Stage5 Out",
    "refine_out": "Refine Out",
}
LAYER_COLORS = {
    "stage5_mid": "#4C72B0",   # muted blue
    "stage5_out": "#DD8452",   # muted orange
    "refine_out": "#55A868",   # muted green
}


# ==============================================================================
# Read CSV
# ==============================================================================
def read_all_folds_csv(csv_path):
    """
    Read aggregated_r2_all_folds.csv.
    Returns list of dicts with keys: Config, Seed, Model, Layer, Group, Fold_idx, R2
    """
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "config": row["Config"],
                "seed": int(row["Seed"]),
                "model": row["Model"],
                "layer": row["Layer"],
                "group": row["Group"],
                "fold_idx": int(row["Fold_idx"]),
                "r2": float(row["R2"]),
            })
    return rows


# ==============================================================================
# Compute stats per (layer, group)
# ==============================================================================
def compute_stats(folds, config, model):
    """
    For each (layer, group), compute mean, 95% CI, and collect fold R²s.
    Returns dict: (layer, group) -> {mean, ci_lo, ci_hi, values}
    """
    grouped = defaultdict(list)
    for row in folds:
        if row["config"] != config or row["model"] != model:
            continue
        if row["group"] not in GROUPS_OF_INTEREST:
            continue
        if row["layer"] not in LAYERS:
            continue
        grouped[(row["layer"], row["group"])].append(row["r2"])

    result = {}
    for key, vals in grouped.items():
        arr = np.array(vals)
        n = len(arr)
        mean = arr.mean()
        se = arr.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0

        if n > 1:
            ci_lo, ci_hi = stats.t.interval(0.95, df=n - 1, loc=mean, scale=se)
        else:
            ci_lo, ci_hi = mean, mean

        result[key] = {
            "mean": mean,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "values": arr,
            "n": n,
        }

    return result


# ==============================================================================
# Significance string from p-value
# ==============================================================================
def pval_to_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"


# ==============================================================================
# Pairwise Wilcoxon signed-rank test between layers (paired by seed × fold)
# ==============================================================================
def pairwise_wilcoxon(folds, config, model, group):
    """
    For a given group, compute Wilcoxon signed-rank test (paired)
    for all pairs of layers, paired by (seed, fold_idx).
    Returns list of (layer_a, layer_b, n_pairs, p_value, stars_str)
    """
    # Build lookup: layer -> {(seed, fold_idx): r2}
    lookup = defaultdict(dict)
    for row in folds:
        if row["config"] != config or row["model"] != model:
            continue
        if row["group"] != group or row["layer"] not in LAYERS:
            continue
        lookup[row["layer"]][(row["seed"], row["fold_idx"])] = row["r2"]

    results = []
    for layer_a, layer_b in combinations(LAYERS, 2):
        dict_a = lookup.get(layer_a, {})
        dict_b = lookup.get(layer_b, {})
        if not dict_a or not dict_b:
            continue

        common_keys = sorted(set(dict_a.keys()) & set(dict_b.keys()))
        if len(common_keys) < 5:
            print(f"  ⚠ Wilcoxon skip {group}: {layer_a} vs {layer_b} — only {len(common_keys)} pairs")
            continue

        r2_a = np.array([dict_a[k] for k in common_keys])
        r2_b = np.array([dict_b[k] for k in common_keys])
        diff = r2_a - r2_b

        try:
            _stat, pval = wilcoxon(diff, alternative="two-sided")
        except ValueError:
            pval = 1.0

        results.append((layer_a, layer_b, len(common_keys), pval, pval_to_stars(pval)))

    return results


# ==============================================================================
# Draw significance bracket
# ==============================================================================
def draw_bracket(ax, x1, x2, y, h, text, fontsize=9):
    """
    Draw a bracket between x1 and x2 at height y with text annotation.
    h is the bracket height.
    """
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.2, color="black")
    ax.text((x1 + x2) / 2, y + h, text,
            ha="center", va="bottom", fontsize=fontsize, fontweight="bold")


# ==============================================================================
# Main plot function
# ==============================================================================
def plot_layer_comparison(stat_dict, folds_data, results_dir, config, model):
    """
    Create grouped bar plot comparing layers across genes.
    """
    n_groups = len(GROUPS_OF_INTEREST)
    n_layers = len(LAYERS)
    bar_width = 0.22
    gap = 0.04

    fig, ax = plt.subplots(figsize=(8, 5.5))

    # X positions for groups
    x_centers = np.arange(n_groups)

    # Bar positions within each group
    offsets = np.array([
        -(bar_width + gap),
        0,
        (bar_width + gap),
    ])

    # Track max bar height for bracket placement
    max_heights = {}  # group_idx -> max bar top (including error bar)

    for layer_idx, layer in enumerate(LAYERS):
        means = []
        ci_errors_lo = []
        ci_errors_hi = []
        positions = []

        for grp_idx, grp in enumerate(GROUPS_OF_INTEREST):
            key = (layer, grp)
            if key in stat_dict:
                d = stat_dict[key]
                means.append(d["mean"])
                ci_errors_lo.append(d["mean"] - d["ci_lo"])
                ci_errors_hi.append(d["ci_hi"] - d["mean"])
            else:
                means.append(0)
                ci_errors_lo.append(0)
                ci_errors_hi.append(0)

            pos = x_centers[grp_idx] + offsets[layer_idx]
            positions.append(pos)

            # Track max height
            top = (d["ci_hi"] if key in stat_dict else 0)
            if grp_idx not in max_heights or top > max_heights[grp_idx]:
                max_heights[grp_idx] = top

        positions = np.array(positions)
        errors = np.array([ci_errors_lo, ci_errors_hi])

        ax.bar(
            positions, means,
            width=bar_width,
            color=LAYER_COLORS[layer],
            edgecolor="white",
            linewidth=0.5,
            label=LAYER_LABELS[layer],
            yerr=errors,
            capsize=3,
            error_kw={"linewidth": 1.0, "capthick": 1.0},
            zorder=3,
        )

    # ── Significance brackets ──
    for grp_idx, grp in enumerate(GROUPS_OF_INTEREST):
        pairwise = pairwise_wilcoxon(folds_data, config, model, grp)
        if not pairwise:
            continue

        # Sort by bracket width (narrowest first)
        layer_index = {l: i for i, l in enumerate(LAYERS)}
        pairwise_sorted = sorted(
            pairwise, key=lambda x: abs(layer_index[x[1]] - layer_index[x[0]])
        )

        # Starting height for brackets
        base_y = max_heights.get(grp_idx, 0)
        bracket_h = 0.008  # bracket arm height
        bracket_gap = 0.025  # gap between stacked brackets

        for pair_idx, (layer_a, layer_b, _n, pval, stars) in enumerate(pairwise_sorted):
            idx_a = layer_index[layer_a]
            idx_b = layer_index[layer_b]

            x1 = x_centers[grp_idx] + offsets[idx_a]
            x2 = x_centers[grp_idx] + offsets[idx_b]

            y = base_y + bracket_gap * (pair_idx + 1)

            # Add p-value text
            if pval < 0.001:
                p_text = f"{stars}\np<0.001"
            else:
                p_text = f"{stars}\np={pval:.3f}"

            draw_bracket(ax, x1, x2, y, bracket_h, p_text, fontsize=7)

    # ── Axes formatting ──
    ax.set_xticks(x_centers)
    ax.set_xticklabels(
        [GENE_LABELS[g] for g in GROUPS_OF_INTEREST],
        fontsize=12, fontweight="bold"
    )
    ax.set_ylabel("R² (Cell death Rate Prediction)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Layer-wise Information Content for Cell Death Prediction\n"
        f"({config} | {model})",
        fontsize=13, fontweight="bold", pad=15
    )

    # Legend
    ax.legend(
        title="CNN Layer", title_fontsize=10,
        fontsize=9, loc="upper left",
        framealpha=0.9, edgecolor="gray"
    )

    # Grid
    ax.grid(axis="y", alpha=0.2, zorder=0)
    ax.set_axisbelow(True)

    # Y-axis start from 0 if min is positive
    y_min, y_max = ax.get_ylim()
    if y_min > 0:
        ax.set_ylim(bottom=0)

    # Adjust top margin for brackets
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(top=y_max * 1.15)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    # Save
    base_name = f"layer_comparison_{config}_{model}"
    save_pdf = os.path.join(results_dir, f"{base_name}.pdf")
    save_png = os.path.join(results_dir, f"{base_name}.png")
    save_svg = os.path.join(results_dir, f"{base_name}.svg")

    fig.savefig(save_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(save_png, dpi=200, bbox_inches="tight")
    fig.savefig(save_svg, bbox_inches="tight")

    plt.close(fig)
    print(f"\n  Saved layer comparison plot:")
    print(f"    PDF: {save_pdf}")
    print(f"    PNG: {save_png}")
    print(f"    SVG: {save_svg}")

    return save_pdf


# ==============================================================================
# Print pairwise Wilcoxon results table
# ==============================================================================
def print_pairwise_summary(folds, config, model, stat_dict):
    """Print pairwise layer comparison table."""
    print("\n" + "=" * 90)
    print("  PAIRWISE WILCOXON SIGNED-RANK TEST: Layer Comparison")
    print("  (paired by seed × fold_idx)")
    print("=" * 90)

    for grp in GROUPS_OF_INTEREST:
        results = pairwise_wilcoxon(folds, config, model, grp)
        if not results:
            continue

        print(f"\n  ── {GENE_LABELS[grp]} ──")
        print(f"  {'Layer A':15s} {'Layer B':15s} {'N pairs':>8s} "
              f"{'Mean A':>8s} {'Mean B':>8s} {'p-value':>10s} {'Sig':>5s}")
        print("  " + "-" * 75)

        for layer_a, layer_b, n_pairs, pval, stars in results:
            key_a = (layer_a, grp)
            key_b = (layer_b, grp)
            mean_a = stat_dict[key_a]["mean"] if key_a in stat_dict else 0
            mean_b = stat_dict[key_b]["mean"] if key_b in stat_dict else 0

            print(f"  {LAYER_LABELS[layer_a]:15s} {LAYER_LABELS[layer_b]:15s} "
                  f"{n_pairs:>8d} "
                  f"{mean_a:>8.4f} {mean_b:>8.4f} "
                  f"{pval:>10.6f} {stars:>5s}")


# ==============================================================================
# Save pairwise results CSV
# ==============================================================================
def save_pairwise_csv(folds, config, model, stat_dict, results_dir):
    """Save pairwise Wilcoxon results to CSV."""
    csv_path = os.path.join(results_dir, f"layer_pairwise_wilcoxon_{config}_{model}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Gene", "Layer_A", "Layer_B", "N_pairs",
                     "Mean_R2_A", "Mean_R2_B", "P_value", "Significance"])

        for grp in GROUPS_OF_INTEREST:
            results = pairwise_wilcoxon(folds, config, model, grp)
            for layer_a, layer_b, n_pairs, pval, stars in results:
                key_a = (layer_a, grp)
                key_b = (layer_b, grp)
                w.writerow([
                    GENE_LABELS[grp],
                    layer_a, layer_b,
                    n_pairs,
                    f"{stat_dict[key_a]['mean']:.6f}",
                    f"{stat_dict[key_b]['mean']:.6f}",
                    f"{pval:.6f}",
                    stars,
                ])

    print(f"  Saved pairwise CSV: {csv_path}")


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Plot layer comparison R² bar chart with Wilcoxon tests"
    )
    parser.add_argument(
        "--results_dir", type=str, required=True,
        help="Path to results directory containing aggregated_r2_all_folds.csv"
    )
    parser.add_argument(
        "--config", type=str, default="MoCo_l2norm",
        help="Config to plot (default: MoCo_l2norm)"
    )
    parser.add_argument(
        "--model", type=str, default="Ridge",
        help="Model to plot (default: Ridge)"
    )
    args = parser.parse_args()

    csv_path = os.path.join(args.results_dir, "aggregated_r2_all_folds.csv")
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV not found: {csv_path}")
        sys.exit(1)

    # ── Read data ──
    folds = read_all_folds_csv(csv_path)
    print(f"\n  Read {len(folds)} fold entries from {csv_path}")

    # ── Filter & compute stats ──
    stat_dict = compute_stats(folds, args.config, args.model)
    if not stat_dict:
        print(f"ERROR: No data for config={args.config}, model={args.model}")
        sys.exit(1)

    # Print counts
    for layer in LAYERS:
        for grp in GROUPS_OF_INTEREST:
            key = (layer, grp)
            if key in stat_dict:
                d = stat_dict[key]
                print(f"  {LAYER_LABELS[layer]:15s} | {GENE_LABELS[grp]:5s}: "
                      f"R²={d['mean']:.4f} [{d['ci_lo']:.4f}, {d['ci_hi']:.4f}]  "
                      f"(n={d['n']})")

    # ── Pairwise tests ──
    print_pairwise_summary(folds, args.config, args.model, stat_dict)

    # ── Plot ──
    plot_layer_comparison(stat_dict, folds, args.results_dir, args.config, args.model)

    # ── Save CSV ──
    save_pairwise_csv(folds, args.config, args.model, stat_dict, args.results_dir)

    print("\n  DONE")


if __name__ == "__main__":
    main()
