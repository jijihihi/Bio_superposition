# ==============================================================================
# GAP L2 Norm Effect Plot
#
# Bar plot comparing L2 norm ON vs OFF for a selected layer and model,
# with null permutation R² line, 95% CI error bars, and
# paired Wilcoxon signed-rank test (paired by seed × fold).
#
# Usage (Colab):
#   import sys
#   sys.argv = [
#       "plot_l2norm_effect",
#       "--results_dir", "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/apoptosis_r2_results",
#       "--training_config", "MoCo",
#       "--layer", "stage5_out",
#       "--model", "Ridge",
#   ]
#   from apoptosis_prediction.plot_l2norm_effect import main
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

# Training config → (l2norm config name, raw config name)
CONFIG_PAIRS = {
    "MoCo":   ("MoCo_l2norm",   "MoCo_raw"),
    "noNorm": ("noNorm_l2norm", "noNorm_raw"),
}

L2_COLORS = {
    "raw":    "#7A9DC7",   # soft blue — L2 OFF
    "l2norm": "#E07B54",   # warm orange — L2 ON
}


# ==============================================================================
# Read CSVs
# ==============================================================================
def read_all_folds_csv(csv_path):
    """Read aggregated_r2_all_folds.csv."""
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


def read_pooled_perm_csv(csv_path):
    """Read aggregated_r2_pooled_perm.csv."""
    rows = []
    if not os.path.exists(csv_path):
        return rows
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "config": row["Config"],
                "model": row["Model"],
                "layer": row["Layer"],
                "group": row["Group"],
                "null_mean_r2": float(row["Null_mean_R2"]),
                "pooled_p_value": float(row["Pooled_p_value"]),
            })
    return rows


# ==============================================================================
# Compute stats
# ==============================================================================
def compute_group_stats(folds, config, layer, model):
    """
    For a single config, compute per-group stats.
    Returns dict: group -> {mean, ci_lo, ci_hi, values, n}
    """
    grouped = defaultdict(list)
    for row in folds:
        if row["config"] != config or row["layer"] != layer or row["model"] != model:
            continue
        if row["group"] not in GROUPS_OF_INTEREST:
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


# ==============================================================================
# Paired Wilcoxon signed-rank test: L2 ON vs OFF
# ==============================================================================
def paired_wilcoxon_test(folds, cfg_on, cfg_off, layer, model):
    """
    Pair fold R²s by (seed, fold_idx) between l2norm and raw configs.
    Returns dict: group -> {n_pairs, W_stat, p_value, median_diff, mean_on, mean_off}
    """
    # Build lookup: (config, group) -> {(seed, fold_idx): r2}
    lookup = defaultdict(dict)
    for row in folds:
        if row["layer"] != layer or row["model"] != model:
            continue
        if row["config"] not in (cfg_on, cfg_off):
            continue
        if row["group"] not in GROUPS_OF_INTEREST:
            continue
        key = (row["config"], row["group"])
        pair_key = (row["seed"], row["fold_idx"])
        lookup[key][pair_key] = row["r2"]

    results = {}
    for grp in GROUPS_OF_INTEREST:
        on_dict = lookup.get((cfg_on, grp), {})
        off_dict = lookup.get((cfg_off, grp), {})

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

        results[grp] = {
            "n_pairs": len(common_keys),
            "W_stat": float(stat),
            "p_value": float(pval),
            "median_diff": float(np.median(diff)),
            "mean_on": float(np.mean(r2_on)),
            "mean_off": float(np.mean(r2_off)),
        }

    return results


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
# Draw significance bracket
# ==============================================================================
def draw_bracket(ax, x1, x2, y, h, text, fontsize=9):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.2, color="black")
    ax.text((x1 + x2) / 2, y + h, text,
            ha="center", va="bottom", fontsize=fontsize, fontweight="bold")


# ==============================================================================
# Main plot
# ==============================================================================
def plot_l2norm_effect(stats_off, stats_on, wilcoxon_results, null_perm,
                       results_dir, training_config, layer, model):
    """
    Bar plot: L2 OFF vs L2 ON for each gene, with null permutation line.
    """
    n_groups = len(GROUPS_OF_INTEREST)
    bar_width = 0.28
    gap = 0.05

    fig, ax = plt.subplots(figsize=(8, 5.5))

    x_centers = np.arange(n_groups)
    offsets = [-(bar_width / 2 + gap / 2), (bar_width / 2 + gap / 2)]

    max_heights = {}

    # --- Bars ---
    for cond_idx, (label, stats_dict, color_key) in enumerate([
        ("L2 OFF", stats_off, "raw"),
        ("L2 ON",  stats_on,  "l2norm"),
    ]):
        means = []
        ci_lo_errs = []
        ci_hi_errs = []
        positions = []

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

            pos = x_centers[grp_idx] + offsets[cond_idx]
            positions.append(pos)

            if grp_idx not in max_heights or top > max_heights[grp_idx]:
                max_heights[grp_idx] = top

        positions = np.array(positions)
        errors = np.array([ci_lo_errs, ci_hi_errs])

        ax.bar(
            positions, means,
            width=bar_width,
            color=L2_COLORS[color_key],
            edgecolor="white",
            linewidth=0.5,
            label=label,
            yerr=errors,
            capsize=4,
            error_kw={"linewidth": 1.0, "capthick": 1.0},
            zorder=3,
        )

    # --- Null permutation line ---
    for grp_idx, grp in enumerate(GROUPS_OF_INTEREST):
        null_vals = []
        for cond_cfg in [CONFIG_PAIRS[training_config][1],
                         CONFIG_PAIRS[training_config][0]]:
            matches = [r for r in null_perm
                       if r["config"] == cond_cfg
                       and r["layer"] == layer
                       and r["model"] == model
                       and r["group"] == grp]
            if matches:
                null_vals.append(matches[0]["null_mean_r2"])

        if null_vals:
            null_mean = np.mean(null_vals)
            null_clipped = max(0.0, null_mean)

            x_left = x_centers[grp_idx] - bar_width - gap
            x_right = x_centers[grp_idx] + bar_width + gap

            ax.hlines(
                null_clipped, x_left, x_right,
                colors="#888888", linewidth=1.5, linestyles="--", zorder=4,
                label="Null Permutation" if grp_idx == 0 else None,
            )

            # Annotate null value (show raw value even if clipped)
            ax.text(
                x_right + 0.02, null_clipped,
                f"null={null_mean:.4f}",
                fontsize=7, color="#666666", va="center", ha="left",
            )

    # --- Wilcoxon brackets ---
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

    # --- Formatting ---
    ax.set_xticks(x_centers)
    ax.set_xticklabels(
        [GENE_LABELS[g] for g in GROUPS_OF_INTEREST],
        fontsize=12, fontweight="bold"
    )
    ax.set_ylabel("R² (Cell Death Rate Prediction)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"GAP L2 Norm Effect on Cell Death Prediction\n"
        f"({training_config} | {layer} | {model})",
        fontsize=13, fontweight="bold", pad=15,
    )

    ax.legend(
        fontsize=9, loc="upper left",
        framealpha=0.9, edgecolor="gray",
    )

    ax.grid(axis="y", alpha=0.2, zorder=0)
    ax.set_axisbelow(True)

    y_min, y_max = ax.get_ylim()
    if y_min > 0:
        ax.set_ylim(bottom=0)
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(top=y_max * 1.15)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    # --- Save ---
    base_name = f"l2norm_effect_{training_config}_{layer}_{model}"
    save_pdf = os.path.join(results_dir, f"{base_name}.pdf")
    save_png = os.path.join(results_dir, f"{base_name}.png")
    save_svg = os.path.join(results_dir, f"{base_name}.svg")

    fig.savefig(save_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(save_png, dpi=200, bbox_inches="tight")
    fig.savefig(save_svg, bbox_inches="tight")

    plt.close(fig)
    print(f"\n  Saved L2 norm effect plot:")
    print(f"    PDF: {save_pdf}")
    print(f"    PNG: {save_png}")
    print(f"    SVG: {save_svg}")


# ==============================================================================
# Print summary
# ==============================================================================
def print_summary(stats_off, stats_on, wilcoxon_results, training_config, layer, model):
    cfg_on, cfg_off = CONFIG_PAIRS[training_config]

    print(f"\n{'=' * 90}")
    print(f"  GAP L2 NORM EFFECT — {training_config} | {layer} | {model}")
    print(f"  Config ON:  {cfg_on}")
    print(f"  Config OFF: {cfg_off}")
    print(f"{'=' * 90}")

    print(f"\n  {'Gene':6s}  {'L2 OFF mean':>12s}  {'L2 ON mean':>12s}  "
          f"{'Δ(ON−OFF)':>10s}  {'95% CI OFF':>22s}  {'95% CI ON':>22s}")
    print("  " + "-" * 95)

    for grp in GROUPS_OF_INTEREST:
        off = stats_off.get(grp)
        on = stats_on.get(grp)
        if not off or not on:
            continue

        delta = on["mean"] - off["mean"]
        print(f"  {GENE_LABELS[grp]:6s}  {off['mean']:>12.4f}  {on['mean']:>12.4f}  "
              f"{delta:>+10.4f}  [{off['ci_lo']:.4f}, {off['ci_hi']:.4f}]  "
              f"[{on['ci_lo']:.4f}, {on['ci_hi']:.4f}]")

    if wilcoxon_results:
        print(f"\n  Wilcoxon Signed-Rank Test (paired by seed × fold):")
        print(f"  {'Gene':6s}  {'N pairs':>8s}  {'W stat':>8s}  {'p-value':>10s}  {'Sig':>5s}")
        print("  " + "-" * 45)

        for grp in GROUPS_OF_INTEREST:
            if grp not in wilcoxon_results:
                continue
            w = wilcoxon_results[grp]
            stars = pval_to_stars(w["p_value"])
            print(f"  {GENE_LABELS[grp]:6s}  {w['n_pairs']:>8d}  "
                  f"{w['W_stat']:>8.1f}  {w['p_value']:>10.6f}  {stars:>5s}")


# ==============================================================================
# Save CSV
# ==============================================================================
def save_results_csv(stats_off, stats_on, wilcoxon_results, null_perm,
                     results_dir, training_config, layer, model):
    csv_path = os.path.join(
        results_dir,
        f"l2norm_effect_{training_config}_{layer}_{model}.csv"
    )
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "Gene", "Condition", "Mean_R2", "CI95_lower", "CI95_upper",
            "N_folds", "Null_mean_R2", "Wilcoxon_W", "Wilcoxon_p", "Sig",
        ])

        cfg_on, cfg_off = CONFIG_PAIRS[training_config]

        for grp in GROUPS_OF_INTEREST:
            off = stats_off.get(grp)
            on = stats_on.get(grp)
            wres = wilcoxon_results.get(grp)

            # Null perm (average of both conditions)
            null_vals = []
            for cfg in [cfg_on, cfg_off]:
                matches = [r for r in null_perm
                           if r["config"] == cfg and r["layer"] == layer
                           and r["model"] == model and r["group"] == grp]
                if matches:
                    null_vals.append(matches[0]["null_mean_r2"])
            null_mean = np.mean(null_vals) if null_vals else ""

            for label, st in [("L2_OFF", off), ("L2_ON", on)]:
                if not st:
                    continue
                w.writerow([
                    GENE_LABELS[grp], label,
                    f"{st['mean']:.6f}",
                    f"{st['ci_lo']:.6f}",
                    f"{st['ci_hi']:.6f}",
                    st["n"],
                    f"{null_mean:.6f}" if isinstance(null_mean, float) else "",
                    f"{wres['W_stat']:.1f}" if wres else "",
                    f"{wres['p_value']:.6f}" if wres else "",
                    pval_to_stars(wres["p_value"]) if wres else "",
                ])

    print(f"  Saved results CSV: {csv_path}")


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Plot GAP L2 Norm effect on R² with paired Wilcoxon test"
    )
    parser.add_argument(
        "--results_dir", type=str, required=True,
        help="Path to results directory"
    )
    parser.add_argument(
        "--training_config", type=str, default="MoCo",
        choices=["MoCo", "noNorm"],
        help="Training config pair (default: MoCo)"
    )
    parser.add_argument(
        "--layer", type=str, default="stage5_out",
        choices=["stage5_mid", "stage5_out", "refine_out"],
        help="Layer to plot (default: stage5_out)"
    )
    parser.add_argument(
        "--model", type=str, default="Ridge",
        help="Model to plot (default: Ridge)"
    )
    args = parser.parse_args()

    # Resolve config pair
    cfg_on, cfg_off = CONFIG_PAIRS[args.training_config]

    # Read data
    folds_csv = os.path.join(args.results_dir, "aggregated_r2_all_folds.csv")
    if not os.path.exists(folds_csv):
        print(f"ERROR: CSV not found: {folds_csv}")
        sys.exit(1)

    folds = read_all_folds_csv(folds_csv)
    print(f"\n  Read {len(folds)} fold entries")

    perm_csv = os.path.join(args.results_dir, "aggregated_r2_pooled_perm.csv")
    null_perm = read_pooled_perm_csv(perm_csv)
    if null_perm:
        print(f"  Read {len(null_perm)} pooled permutation entries")
    else:
        print(f"  ⚠ No pooled permutation CSV found (will skip null line)")

    # Compute stats
    stats_off = compute_group_stats(folds, cfg_off, args.layer, args.model)
    stats_on = compute_group_stats(folds, cfg_on, args.layer, args.model)

    if not stats_off and not stats_on:
        print(f"ERROR: No data for training_config={args.training_config}, "
              f"layer={args.layer}, model={args.model}")
        sys.exit(1)

    # Wilcoxon
    wilcoxon_results = paired_wilcoxon_test(
        folds, cfg_on, cfg_off, args.layer, args.model
    )

    # Print
    print_summary(stats_off, stats_on, wilcoxon_results,
                  args.training_config, args.layer, args.model)

    # Plot
    plot_l2norm_effect(
        stats_off, stats_on, wilcoxon_results, null_perm,
        args.results_dir, args.training_config, args.layer, args.model,
    )

    # Save CSV
    save_results_csv(
        stats_off, stats_on, wilcoxon_results, null_perm,
        args.results_dir, args.training_config, args.layer, args.model,
    )

    print("\n  DONE")


if __name__ == "__main__":
    main()
