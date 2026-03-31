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
#       "--model", "XGBoost",
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
import seaborn as sns

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42      
sns.set_style("ticks")


#  정규 근사로 한다.


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
                       results_dir, training_config, layer, model,
                       folds=None, cfg_on=None, cfg_off=None):
    """
    Paired dot-line plot: L2 OFF vs L2 ON, one figure per mutation.
    Dots = per-seed mean R² (averaged over CV folds).
    P-value & effect size r are computed from individual CV folds.
    Each mutation saved as a separate file.
    """
    if folds is None or cfg_on is None or cfg_off is None:
        print("  ⚠ No raw folds data — cannot create paired plot")
        return

    # ── Build per-seed-mean paired data & per-fold paired data ──
    fold_lookup = defaultdict(dict)  # (config, grp) -> {(seed, fold_idx): r2}
    for row in folds:
        if row["layer"] != layer or row["model"] != model:
            continue
        if row["config"] not in (cfg_on, cfg_off):
            continue
        if row["group"] not in GROUPS_OF_INTEREST:
            continue
        key = (row["config"], row["group"])
        pair_key = (row["seed"], row["fold_idx"])
        fold_lookup[key][pair_key] = row["r2"]

    for grp in GROUPS_OF_INTEREST:
        gene_label = GENE_LABELS[grp]

        on_dict = fold_lookup.get((cfg_on, grp), {})
        off_dict = fold_lookup.get((cfg_off, grp), {})
        common_fold_keys = sorted(set(on_dict.keys()) & set(off_dict.keys()))

        if len(common_fold_keys) < 5:
            print(f"  ⚠ {gene_label}: only {len(common_fold_keys)} pairs — skipping")
            continue

        # Per-fold arrays (for stats)
        fold_r2_off = np.array([off_dict[k] for k in common_fold_keys])
        fold_r2_on = np.array([on_dict[k] for k in common_fold_keys])

        # Per-seed mean (for plotting)
        seed_off = defaultdict(list)
        seed_on = defaultdict(list)
        for (seed, fold_idx) in common_fold_keys:
            seed_off[seed].append(off_dict[(seed, fold_idx)])
            seed_on[seed].append(on_dict[(seed, fold_idx)])

        seeds = sorted(seed_off.keys())
        seed_mean_off = np.array([np.mean(seed_off[s]) for s in seeds])
        seed_mean_on = np.array([np.mean(seed_on[s]) for s in seeds])
        n_seeds = len(seeds)

        # ── Figure ──
        fig, ax = plt.subplots(figsize=(4.0, 5.2))
        x_off, x_on = 0, 1

        # Compute jitter once — shared by lines and dots
        jitter_rng = np.random.default_rng(42)
        jitter = jitter_rng.uniform(-0.06, 0.06, size=n_seeds)

        # Individual paired lines (semi-transparent, jittered to match dots)
        for i in range(n_seeds):
            ax.plot(
                [x_off + jitter[i], x_on + jitter[i]],
                [seed_mean_off[i], seed_mean_on[i]],
                color="#AAAAAA", alpha=0.35, linewidth=1.0, zorder=2,
            )

        # Individual dots (same jitter as lines)
        ax.scatter(
            x_off + jitter, seed_mean_off,
            s=40, color=L2_COLORS["raw"], alpha=0.7,
            edgecolors="white", linewidths=0.5, zorder=4,
        )
        ax.scatter(
            x_on + jitter, seed_mean_on,
            s=40, color=L2_COLORS["l2norm"], alpha=0.7,
            edgecolors="white", linewidths=0.5, zorder=4,
        )

        # Grand mean (RED, prominent)
        grand_mean_off = seed_mean_off.mean()
        grand_mean_on = seed_mean_on.mean()

        ax.plot(
            [x_off, x_on], [grand_mean_off, grand_mean_on],
            color="#D32F2F", linewidth=2.8, zorder=6, solid_capstyle="round",
        )
        ax.scatter(
            [x_off, x_on], [grand_mean_off, grand_mean_on],
            s=90, color="#D32F2F", edgecolors="white", linewidths=1.8,
            zorder=7,
        )

        # Annotate grand means
        ax.text(x_off - 0.15, grand_mean_off, f"{grand_mean_off:.3f}",
                fontsize=9, color="#D32F2F", fontweight="bold",
                ha="right", va="center")
        ax.text(x_on + 0.15, grand_mean_on, f"{grand_mean_on:.3f}",
                fontsize=9, color="#D32F2F", fontweight="bold",
                ha="left", va="center")

        # ── Wilcoxon p-value + effect size r (independently computed) ──
        if grp in wilcoxon_results:
            w = wilcoxon_results[grp]
            pval = w["p_value"]
            n_folds = w["n_pairs"]
            W_stat = w["W_stat"]

            # Effect size r: computed directly from W statistic (NOT from p-value)
            mean_W = n_folds * (n_folds + 1) / 4
            std_W = np.sqrt(n_folds * (n_folds + 1) * (2 * n_folds + 1) / 24)
            z_from_W = abs((W_stat - mean_W) / std_W) if std_W > 0 else 0.0
            effect_r = z_from_W / np.sqrt(n_folds)

            # Cross-validation: also compute Z from p-value for sanity check
            from scipy.stats import norm
            z_from_p = abs(norm.ppf(pval / 2)) if pval > 0 else 5.0
            r_from_p = z_from_p / np.sqrt(n_folds)

            p_str = f"p<0.001" if pval < 0.001 else f"p={pval:.3f}"
            stat_text = f"{p_str},  r={effect_r:.2f}"
            ax.text(0.98, 0.97, stat_text, transform=ax.transAxes,
                    fontsize=8, fontweight="bold", color="#333333",
                    ha="right", va="top")

            print(f"  {gene_label}: Wilcoxon W={W_stat:.1f}, p={pval:.6f}, "
                  f"r(from W)={effect_r:.3f}, r(from p)={r_from_p:.3f}, "
                  f"n_pairs={n_folds}")

        # ── Y-axis: tight zoom to data range ──
        all_vals = np.concatenate([seed_mean_off, seed_mean_on])
        data_min = all_vals.min()
        data_max = all_vals.max()
        margin = (data_max - data_min) * 0.15
        ax.set_ylim(data_min - margin, data_max + margin)

        # ── Formatting ──
        ax.set_xticks([x_off, x_on])
        ax.set_xticklabels(["L2 OFF", "L2 ON"], fontsize=12, fontweight="bold")
        ax.set_ylabel("R² (Cell Death Prediction)", fontsize=10, fontweight="bold")
        ax.set_title(
            f"{gene_label} — GAP L2 Norm Effect\n"
            f"({training_config} | {layer} | {model})",
            fontsize=12, fontweight="bold", pad=10,
        )
        ax.set_xlim(-0.4, 1.4)
        ax.grid(axis="y", alpha=0.15, zorder=0)
        ax.set_axisbelow(True)
        sns.despine(ax=ax)

        fig.tight_layout()

        # ── Save per-mutation ──
        base = f"l2norm_effect_{training_config}_{layer}_{model}_{gene_label}"
        for ext in ["pdf", "png", "svg"]:
            path = os.path.join(results_dir, f"{base}.{ext}")
            fig.savefig(path, dpi=300 if ext != "png" else 200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {gene_label}: {base}.svg / .png / .pdf")


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
        help="Training config pair (default: MoCo) or noNorm"
    )
    parser.add_argument(
        "--layer", type=str, default="stage5_out",
        choices=["stage5_mid", "stage5_out", "refine_out"],
        help="Layer to plot (default: stage5_out)"
    )
    parser.add_argument(
        "--model", type=str, default="Ridge",
        help="Model to plot (default: Ridge) or XGBoost"
    )
    args = parser.parse_args()

    sns.despine()

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
        folds=folds, cfg_on=cfg_on, cfg_off=cfg_off,
    )

    # Save CSV
    save_results_csv(
        stats_off, stats_on, wilcoxon_results, null_perm,
        args.results_dir, args.training_config, args.layer, args.model,
    )

    print("\n  DONE")


if __name__ == "__main__":
    main()
