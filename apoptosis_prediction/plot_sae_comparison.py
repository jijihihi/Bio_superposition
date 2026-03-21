# ==============================================================================
# SAE Comparison Plots
#
# Two bar plots from sae_r2_all_folds.csv:
#   1) Ridge vs XGBoost comparison (no p-value — difference is obvious)
#   2) GAP L2 norm effect on XGBoost with Wilcoxon signed-rank test
#
# Usage (Colab):
#   import sys
#   sys.argv = [
#       "plot_sae_comparison",
#       "--results_dir", "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/apoptosis_r2_results",
#   ]
#   from apoptosis_prediction.plot_sae_comparison import main
#   main()
# ==============================================================================

import os
import sys
import csv
import argparse
from collections import defaultdict

import numpy as np

import matplotlib
if "google.colab" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import wilcoxon


# ==============================================================================
# Constants
# ==============================================================================
GROUPS_OF_INTEREST = ["SNCA only", "GBA only", "LRRK2 only"]
GENE_LABELS = {"SNCA only": "SNCA", "GBA only": "GBA", "LRRK2 only": "LRRK2"}


# ==============================================================================
# Read CSV
# ==============================================================================
def read_sae_all_folds(csv_path):
    """Read sae_r2_all_folds.csv."""
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "sae_seed": int(row["SAE_Seed"]),
                "l2_norm": row["GAP_L2_Norm"],
                "filter": row["Filter"],
                "model": row["Model"],
                "group": row["Group"],
                "fold_idx": int(row["Fold_idx"]),
                "r2": float(row["R2"]),
            })
    return rows


def read_sae_pooled_perm(csv_path):
    """Read sae_r2_pooled_perm.csv."""
    rows = []
    if not os.path.exists(csv_path):
        return rows
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "l2_norm": row["GAP_L2_Norm"],
                "filter": row["Filter"],
                "model": row["Model"],
                "group": row["Group"],
                "null_mean_r2": float(row["Null_mean_R2"]),
                "pooled_p_value": float(row["Pooled_p_value"]),
            })
    return rows


# ==============================================================================
# Compute stats for a subset
# ==============================================================================
def compute_group_stats(folds, l2_norm, model, filter_label=None):
    """
    Compute per-group stats for a specific (l2_norm, model) condition.
    Returns dict: group -> {mean, ci_lo, ci_hi, values, n}
    """
    grouped = defaultdict(list)
    for row in folds:
        if row["l2_norm"] != l2_norm or row["model"] != model:
            continue
        if row["group"] not in GROUPS_OF_INTEREST:
            continue
        if filter_label is not None and row["filter"] != filter_label:
            continue
        grouped[row["group"]].append(row["r2"])

    result = {}
    for grp, vals in grouped.items():
        arr = np.array(vals)
        n = len(arr)
        mean = arr.mean()
        se = arr.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0

        if n > 1:
            ci_lo, ci_hi = stats.t.interval(0.95, df=n - 1, loc=mean, scale=se)
        else:
            ci_lo, ci_hi = mean, mean

        result[grp] = {
            "mean": mean,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "values": arr,
            "n": n,
        }
    return result


def pval_to_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"


def draw_bracket(ax, x1, x2, y, h, text, fontsize=9):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.2, color="black")
    ax.text((x1 + x2) / 2, y + h, text,
            ha="center", va="bottom", fontsize=fontsize, fontweight="bold")


# ==============================================================================
# Plot 1: Ridge vs XGBoost
# ==============================================================================
def plot_model_comparison(folds, results_dir, l2_norm, null_perm, filter_label=None):
    """Bar plot: Ridge vs XGBoost (no p-value), with null permutation lines."""
    stats_ridge = compute_group_stats(folds, l2_norm, "Ridge", filter_label)
    stats_xgb = compute_group_stats(folds, l2_norm, "XGBoost", filter_label)

    if not stats_ridge and not stats_xgb:
        print("  ⚠ No data for model comparison")
        return

    n_groups = len(GROUPS_OF_INTEREST)
    bar_width = 0.28
    gap = 0.05
    fig, ax = plt.subplots(figsize=(7, 5))
    x_centers = np.arange(n_groups)
    offsets = [-(bar_width / 2 + gap / 2), (bar_width / 2 + gap / 2)]

    colors = {"Ridge": "#4C72B0", "XGBoost": "#DD8452"}

    for cond_idx, (label, stats_dict) in enumerate([
        ("Ridge", stats_ridge),
        ("XGBoost", stats_xgb),
    ]):
        means, ci_lo_errs, ci_hi_errs, positions = [], [], [], []

        for grp_idx, grp in enumerate(GROUPS_OF_INTEREST):
            if grp in stats_dict:
                d = stats_dict[grp]
                means.append(d["mean"])
                ci_lo_errs.append(d["mean"] - d["ci_lo"])
                ci_hi_errs.append(d["ci_hi"] - d["mean"])
            else:
                means.append(0)
                ci_lo_errs.append(0)
                ci_hi_errs.append(0)
            positions.append(x_centers[grp_idx] + offsets[cond_idx])

        ax.bar(
            positions, means, width=bar_width,
            color=colors[label], edgecolor="white", linewidth=0.5,
            label=label,
            yerr=[ci_lo_errs, ci_hi_errs],
            capsize=4,
            error_kw={"linewidth": 1.0, "capthick": 1.0},
            zorder=3,
        )

    # ── Null permutation lines (per model per gene) ──
    for grp_idx, grp in enumerate(GROUPS_OF_INTEREST):
        for cond_idx, (model_name, color) in enumerate([
            ("Ridge", "#4C72B0"), ("XGBoost", "#DD8452")
        ]):
            matches = [r for r in null_perm
                       if r["l2_norm"] == l2_norm
                       and r["model"] == model_name
                       and r["group"] == grp
                       and (filter_label is None or r["filter"] == filter_label)]
            if not matches:
                continue

            null_mean = np.mean([m["null_mean_r2"] for m in matches])
            null_clipped = max(0.0, null_mean)

            bar_x = x_centers[grp_idx] + offsets[cond_idx]
            x_left = bar_x - bar_width / 2
            x_right = bar_x + bar_width / 2

            ax.hlines(
                null_clipped, x_left, x_right,
                colors="#888888", linewidth=1.5, linestyles="--", zorder=4,
                label="Null Permutation" if grp_idx == 0 and cond_idx == 0 else None,
            )
            ax.text(
                x_right + 0.01, null_clipped,
                f"{null_mean:.3f}",
                fontsize=6, color="#666666", va="center", ha="left",
            )

    ax.set_xticks(x_centers)
    ax.set_xticklabels([GENE_LABELS[g] for g in GROUPS_OF_INTEREST],
                       fontsize=12, fontweight="bold")
    ax.set_ylabel("R² (Cell Death Rate Prediction)", fontsize=11, fontweight="bold")

    filt_str = f" | filter={filter_label}" if filter_label and filter_label != "no_filter" else ""
    ax.set_title(
        f"SAE: Ridge vs XGBoost — Cell Death Prediction\n"
        f"(L2={l2_norm}{filt_str})",
        fontsize=13, fontweight="bold", pad=15,
    )

    ax.legend(fontsize=10, loc="upper left", framealpha=0.9, edgecolor="gray")
    ax.grid(axis="y", alpha=0.2, zorder=0)
    ax.set_axisbelow(True)

    y_min, _ = ax.get_ylim()
    if y_min > 0:
        ax.set_ylim(bottom=0)
    _, y_max = ax.get_ylim()
    ax.set_ylim(top=y_max * 1.08)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    filt_tag = f"_{filter_label}" if filter_label and filter_label != "no_filter" else ""
    base = f"sae_model_comparison_{l2_norm}{filt_tag}"
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(os.path.join(results_dir, f"{base}.{ext}"),
                    dpi=300 if ext == "pdf" else 200, bbox_inches="tight")

    plt.close(fig)
    print(f"\n  Saved model comparison: {base}.[pdf|png|svg]")

    # Print summary
    print(f"\n  {'Gene':6s}  {'Ridge':>10s}  {'XGBoost':>10s}  {'Δ(XGB−Ridge)':>14s}")
    print("  " + "-" * 45)
    for grp in GROUPS_OF_INTEREST:
        r = stats_ridge.get(grp)
        x = stats_xgb.get(grp)
        if r and x:
            print(f"  {GENE_LABELS[grp]:6s}  {r['mean']:>10.4f}  {x['mean']:>10.4f}  "
                  f"{x['mean'] - r['mean']:>+14.4f}")


# ==============================================================================
# Plot 2: GAP L2 norm effect (XGBoost, paired Wilcoxon signed-rank)
# ==============================================================================
def plot_l2norm_effect(folds, results_dir, model="XGBoost", filter_label=None):
    """Bar plot: L2 OFF vs L2 ON for selected model, with Wilcoxon signed-rank."""
    stats_off = compute_group_stats(folds, "no_l2norm", model, filter_label)
    stats_on = compute_group_stats(folds, "l2norm", model, filter_label)

    if not stats_off and not stats_on:
        print("  ⚠ No data for L2 norm effect")
        return

    # ── Paired Wilcoxon signed-rank ──
    # Build lookup: (l2_norm, group) → {(sae_seed, fold_idx): r2}
    lookup = defaultdict(dict)
    for row in folds:
        if row["model"] != model:
            continue
        if row["group"] not in GROUPS_OF_INTEREST:
            continue
        if filter_label is not None and row["filter"] != filter_label:
            continue
        key = (row["l2_norm"], row["group"])
        pair_key = (row["sae_seed"], row["fold_idx"])
        lookup[key][pair_key] = row["r2"]

    wilcoxon_results = {}
    for grp in GROUPS_OF_INTEREST:
        on_dict = lookup.get(("l2norm", grp), {})
        off_dict = lookup.get(("no_l2norm", grp), {})
        if not on_dict or not off_dict:
            continue

        common_keys = sorted(set(on_dict.keys()) & set(off_dict.keys()))
        if len(common_keys) < 5:
            print(f"  ⚠ Wilcoxon skip {grp}: only {len(common_keys)} pairs")
            continue

        r2_on = np.array([on_dict[k] for k in common_keys])
        r2_off = np.array([off_dict[k] for k in common_keys])
        diff = r2_on - r2_off

        try:
            stat, pval = wilcoxon(diff, alternative="two-sided")
        except ValueError:
            stat, pval = 0.0, 1.0

        wilcoxon_results[grp] = {
            "n_pairs": len(common_keys),
            "W_stat": float(stat),
            "p_value": float(pval),
        }

    # ── Plot ──
    n_groups = len(GROUPS_OF_INTEREST)
    bar_width = 0.28
    gap = 0.05
    fig, ax = plt.subplots(figsize=(7, 5.5))
    x_centers = np.arange(n_groups)
    offsets = [-(bar_width / 2 + gap / 2), (bar_width / 2 + gap / 2)]

    colors = {"off": "#7A9DC7", "on": "#E07B54"}
    max_heights = {}

    for cond_idx, (label, stats_dict, color_key) in enumerate([
        ("L2 OFF", stats_off, "off"),
        ("L2 ON",  stats_on,  "on"),
    ]):
        means, ci_lo_errs, ci_hi_errs, positions = [], [], [], []

        for grp_idx, grp in enumerate(GROUPS_OF_INTEREST):
            if grp in stats_dict:
                d = stats_dict[grp]
                means.append(d["mean"])
                ci_lo_errs.append(d["mean"] - d["ci_lo"])
                ci_hi_errs.append(d["ci_hi"] - d["mean"])
                top = d["ci_hi"]
            else:
                means.append(0)
                ci_lo_errs.append(0)
                ci_hi_errs.append(0)
                top = 0

            positions.append(x_centers[grp_idx] + offsets[cond_idx])
            if grp_idx not in max_heights or top > max_heights[grp_idx]:
                max_heights[grp_idx] = top

        ax.bar(
            positions, means, width=bar_width,
            color=colors[color_key], edgecolor="white", linewidth=0.5,
            label=label,
            yerr=[ci_lo_errs, ci_hi_errs],
            capsize=4,
            error_kw={"linewidth": 1.0, "capthick": 1.0},
            zorder=3,
        )

    # ── Wilcoxon brackets ──
    for grp_idx, grp in enumerate(GROUPS_OF_INTEREST):
        if grp not in wilcoxon_results:
            continue
        w = wilcoxon_results[grp]
        pval = w["p_value"]
        stars = pval_to_stars(pval)

        x1 = x_centers[grp_idx] + offsets[0]
        x2 = x_centers[grp_idx] + offsets[1]

        base_y = max_heights.get(grp_idx, 0)
        bracket_y = base_y + 0.02
        bracket_h = 0.008

        if pval < 0.001:
            p_text = f"{stars}\np<0.001"
        else:
            p_text = f"{stars}\np={pval:.3f}"

        draw_bracket(ax, x1, x2, bracket_y, bracket_h, p_text, fontsize=8)

    ax.set_xticks(x_centers)
    ax.set_xticklabels([GENE_LABELS[g] for g in GROUPS_OF_INTEREST],
                       fontsize=12, fontweight="bold")
    ax.set_ylabel("R² (Cell Death Rate Prediction)", fontsize=11, fontweight="bold")

    filt_str = f" | filter={filter_label}" if filter_label and filter_label != "no_filter" else ""
    ax.set_title(
        f"SAE: GAP L2 Norm Effect — {model}\n"
        f"(Cell Death Prediction{filt_str})",
        fontsize=13, fontweight="bold", pad=15,
    )

    ax.legend(fontsize=10, loc="upper left", framealpha=0.9, edgecolor="gray")
    ax.grid(axis="y", alpha=0.2, zorder=0)
    ax.set_axisbelow(True)

    y_min, _ = ax.get_ylim()
    if y_min > 0:
        ax.set_ylim(bottom=0)
    _, y_max = ax.get_ylim()
    ax.set_ylim(top=y_max * 1.15)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    filt_tag = f"_{filter_label}" if filter_label and filter_label != "no_filter" else ""
    base = f"sae_l2norm_effect_{model}{filt_tag}"
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(os.path.join(results_dir, f"{base}.{ext}"),
                    dpi=300 if ext == "pdf" else 200, bbox_inches="tight")

    plt.close(fig)
    print(f"\n  Saved L2 norm effect: {base}.[pdf|png|svg]")

    # Print summary
    print(f"\n  Wilcoxon Signed-Rank Test (paired by SAE_seed × fold_idx):")
    print(f"  {'Gene':6s}  {'N pairs':>8s}  {'L2 OFF':>8s}  {'L2 ON':>8s}  "
          f"{'p-value':>10s}  {'Sig':>5s}")
    print("  " + "-" * 55)
    for grp in GROUPS_OF_INTEREST:
        off = stats_off.get(grp)
        on = stats_on.get(grp)
        w = wilcoxon_results.get(grp)
        if off and on and w:
            print(f"  {GENE_LABELS[grp]:6s}  {w['n_pairs']:>8d}  {off['mean']:>8.4f}  "
                  f"{on['mean']:>8.4f}  {w['p_value']:>10.6f}  {pval_to_stars(w['p_value']):>5s}")


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="SAE comparison plots: Ridge vs XGBoost, GAP L2 norm effect"
    )
    parser.add_argument(
        "--results_dir", type=str, required=True,
        help="Path to SAE results directory containing sae_r2_all_folds.csv"
    )
    parser.add_argument(
        "--l2_norm", type=str, default="l2norm",
        choices=["l2norm", "no_l2norm"],
        help="L2 norm condition for Ridge vs XGBoost comparison (default: l2norm)"
    )
    parser.add_argument(
        "--filter", type=str, default=None,
        help="Filter label to use (default: None = use all filters)"
    )
    parser.add_argument(
        "--l2_model", type=str, default="XGBoost",
        help="Model for L2 norm effect plot (default: XGBoost)"
    )
    args = parser.parse_args()

    csv_path = os.path.join(args.results_dir, "sae_r2_all_folds.csv")
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV not found: {csv_path}")
        sys.exit(1)

    folds = read_sae_all_folds(csv_path)
    print(f"\n  Read {len(folds)} fold entries from {csv_path}")

    # Print available conditions
    l2_norms = sorted(set(r["l2_norm"] for r in folds))
    models = sorted(set(r["model"] for r in folds))
    filters = sorted(set(r["filter"] for r in folds))
    print(f"  L2 norms: {l2_norms}")
    print(f"  Models:   {models}")
    print(f"  Filters:  {filters}")

    perm_csv = os.path.join(args.results_dir, "sae_r2_pooled_perm.csv")
    null_perm = read_sae_pooled_perm(perm_csv)
    if null_perm:
        print(f"  Read {len(null_perm)} pooled permutation entries")
    else:
        print(f"  ⚠ No pooled permutation CSV found")

    # Plot 1: Ridge vs XGBoost
    print("\n" + "=" * 70)
    print("  PLOT 1: Ridge vs XGBoost")
    print("=" * 70)
    plot_model_comparison(folds, args.results_dir, args.l2_norm, null_perm, args.filter)

    # Plot 2: GAP L2 norm effect (XGBoost)
    print("\n" + "=" * 70)
    print("  PLOT 2: GAP L2 Norm Effect (XGBoost)")
    print("=" * 70)
    plot_l2norm_effect(folds, args.results_dir, args.l2_model, args.filter)

    print("\n  DONE")


if __name__ == "__main__":
    main()
