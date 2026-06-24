# ==============================================================================
# Aggregate SAE R² results — SAE seed × GAP L2 × Filter mode summary
#
# Scans JSON result files from apoptosis_r2_test.py for SAE vectors.
# Directory pattern: SAE_seed{N}_{l2norm|no_l2norm}_{filter_label}/
#
# NEW: Collects per-fold R² scores (from RepeatedKFold), computes:
#   - 95% confidence intervals
#   - Wilcoxon signed-rank test (L2 norm ON vs OFF, paired by seed×fold) 네, 맞습니다. 코드를 확인하면 GAP L2 norm을 제외하고 모든 것이 동일합니다:
#   - Permutation p-value distribution summary
#   - Forest plot & paired comparison plot
#
# Usage:
#   python -m kendall_correlation_coefficient.aggregate_sae_r2_results \
#       --results_dir /path/to/apoptosis_r2_results/SAE_vector
# ==============================================================================


import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict

import matplotlib
import numpy as np

if "google.colab" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

GROUPS_OF_INTEREST = ["SNCA only", "GBA only", "LRRK2 only"]
MODEL_TAGS = ["Ridge", "XGBoost"]


# ==============================================================================
# Collect results
# ==============================================================================
def collect_results(results_dir: str):
    """
    Scan results_dir for CNN_seed*/ directories.
    Parses: CNN_seed{N}_{l2label}_{filter_label}
    Returns agg, per_seed, all_folds.
    """
    agg = defaultdict(
        list
    )  # key = (l2_label, filter_label, model, group) → list of r2_mean
    per_seed = []
    all_folds = []  # every individual fold R²

    dir_pattern = re.compile(r"CNN_seed(\d+)_(l2norm|no_l2norm)(?:_(.+))?$")

    for entry in sorted(os.listdir(results_dir)):
        entry_path = os.path.join(results_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        m = dir_pattern.match(entry)
        if not m:
            continue

        sae_seed = int(m.group(1))
        l2_label = m.group(2)
        filter_label = m.group(3) or "no_filter"

        for fname in os.listdir(entry_path):
            if not fname.endswith(".json") or not fname.startswith("r2_results_"):
                continue
            basename = fname.replace("r2_results_", "").replace(".json", "")
            model_tag = None
            for mt in MODEL_TAGS:
                if basename.endswith(f"_{mt}"):
                    model_tag = mt
                    break
            if model_tag is None:
                continue

            json_path = os.path.join(entry_path, fname)
            with open(json_path) as f:
                data = json.load(f)

            perm_pval_global = None  # top-level perm p-value if present

            for res in data.get("results", []):
                grp = res["group"]
                r2_mean = res["r2_mean"]
                r2_std = res.get("r2_std", 0.0)
                r2_scores = res.get("r2_scores", [])
                perm_pval = res.get("perm_pval", None)

                key = (l2_label, filter_label, model_tag, grp)
                agg[key].append(r2_mean)

                per_seed.append(
                    {
                        "sae_seed": sae_seed,
                        "l2_norm": l2_label,
                        "filter": filter_label,
                        "model": model_tag,
                        "group": grp,
                        "r2_mean": r2_mean,
                        "r2_std": r2_std,
                        "r2_scores": r2_scores,
                        "perm_pval": perm_pval,
                        "perm_fold_r2s": res.get("perm_fold_r2s", []),
                    }
                )

                # Store individual fold R²s
                for fold_i, fold_r2 in enumerate(r2_scores):
                    all_folds.append(
                        {
                            "sae_seed": sae_seed,
                            "l2_norm": l2_label,
                            "filter": filter_label,
                            "model": model_tag,
                            "group": grp,
                            "fold_idx": fold_i,
                            "r2": fold_r2,
                        }
                    )

    return dict(agg), per_seed, all_folds


# ==============================================================================
# Print summary
# ==============================================================================
def print_summary(agg, per_seed):
    """Print summary tables."""
    print("\n" + "=" * 100)
    print("  SAE VECTOR APOPTOSIS R² SUMMARY (mean ± std across CNN seeds)")
    print("=" * 100)

    l2_labels = sorted(set(k[0] for k in agg.keys()))
    filter_labels = sorted(set(k[1] for k in agg.keys()))

    for l2_label in l2_labels:
        for filter_label in filter_labels:
            for model_tag in MODEL_TAGS:
                vals_exist = any(
                    agg.get((l2_label, filter_label, model_tag, g))
                    for g in GROUPS_OF_INTEREST
                )
                if not vals_exist:
                    continue

                n_seeds = max(
                    len(agg.get((l2_label, filter_label, model_tag, g), []))
                    for g in GROUPS_OF_INTEREST
                )
                print(
                    f"\n  ── L2={l2_label} | filter={filter_label} | {model_tag} ({n_seeds} seeds) ──"
                )
                print(
                    f"  {'Group':20s} {'Mean R²':>10s} {'±Std':>10s} {'Min':>10s} {'Max':>10s}"
                )
                print("  " + "-" * 60)

                for grp in GROUPS_OF_INTEREST:
                    key = (l2_label, filter_label, model_tag, grp)
                    vals = agg.get(key, [])
                    if vals:
                        m = np.mean(vals)
                        s = np.std(vals)
                        print(
                            f"  {grp:20s} {m:>10.4f} {s:>10.4f} "
                            f"{np.min(vals):>10.4f} {np.max(vals):>10.4f}"
                        )


# ==============================================================================
# 95% Confidence Intervals
# ==============================================================================
def compute_ci95(all_folds):
    """
    Compute 95% CI for each (l2_norm, filter, model, group) using all fold R²s.
    Uses t-distribution CI.
    """
    from scipy import stats

    grouped = defaultdict(list)
    for row in all_folds:
        key = (row["l2_norm"], row["filter"], row["model"], row["group"])
        grouped[key].append(row["r2"])

    ci_results = []
    for key, vals in sorted(grouped.items()):
        l2, filt, model, grp = key
        arr = np.array(vals)
        n = len(arr)
        mean = arr.mean()
        se = arr.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0

        if n > 1:
            ci_low, ci_high = stats.t.interval(0.95, df=n - 1, loc=mean, scale=se)
        else:
            ci_low, ci_high = mean, mean

        ci_results.append(
            {
                "l2_norm": l2,
                "filter": filt,
                "model": model,
                "group": grp,
                "mean_r2": mean,
                "ci95_lower": ci_low,
                "ci95_upper": ci_high,
                "n_folds": n,
                "std": arr.std(ddof=1) if n > 1 else 0.0,
            }
        )

    return ci_results


# ==============================================================================
# Wilcoxon signed-rank test: L2 ON vs OFF (paired by seed × fold_idx)
# ==============================================================================
def compute_wilcoxon(all_folds):
    """
    For each (filter, model, group), pair fold R²s by (sae_seed, fold_idx)
    between l2norm and no_l2norm conditions.
    Returns Wilcoxon test results.
    """
    from scipy.stats import wilcoxon

    # Group fold R²s by (filter, model, group, l2_norm) → {(sae_seed, fold_idx): r2}
    lookup = defaultdict(dict)
    for row in all_folds:
        condition_key = (row["filter"], row["model"], row["group"], row["l2_norm"])
        pair_key = (row["sae_seed"], row["fold_idx"])
        lookup[condition_key][pair_key] = row["r2"]

    # Identify unique (filter, model, group) combos
    combos = set()
    for filt, model, grp, l2 in lookup:
        combos.add((filt, model, grp))

    results = []
    for filt, model, grp in sorted(combos):
        l2on_key = (filt, model, grp, "l2norm")
        l2off_key = (filt, model, grp, "no_l2norm")

        l2on_dict = lookup.get(l2on_key, {})
        l2off_dict = lookup.get(l2off_key, {})

        if not l2on_dict or not l2off_dict:
            continue

        # Find common paired keys
        common_keys = sorted(set(l2on_dict.keys()) & set(l2off_dict.keys()))
        if len(common_keys) < 5:
            print(
                f"  ⚠ Wilcoxon skip: {filt}|{model}|{grp} — only {len(common_keys)} pairs"
            )
            continue

        r2_on = np.array([l2on_dict[k] for k in common_keys])
        r2_off = np.array([l2off_dict[k] for k in common_keys])
        diff = r2_on - r2_off

        # Wilcoxon signed-rank test (two-sided)
        try:
            stat, pval = wilcoxon(diff, alternative="two-sided")
        except ValueError:
            # All differences are zero
            stat, pval = 0.0, 1.0

        results.append(
            {
                "filter": filt,
                "model": model,
                "group": grp,
                "n_pairs": len(common_keys),
                "median_l2on": float(np.median(r2_on)),
                "median_l2off": float(np.median(r2_off)),
                "mean_l2on": float(np.mean(r2_on)),
                "mean_l2off": float(np.mean(r2_off)),
                "median_diff": float(np.median(diff)),
                "W_stat": float(stat),
                "p_value": float(pval),
            }
        )

    return results


# ==============================================================================
# Permutation p-value summary
# ==============================================================================
def print_perm_summary(per_seed):
    """Print permutation p-value summary across seeds."""
    print("\n" + "=" * 100)
    print("  PERMUTATION P-VALUE SUMMARY")
    print("=" * 100)

    # Group by (l2_norm, filter, model, group)
    grouped = defaultdict(list)
    for r in per_seed:
        if r["perm_pval"] is not None:
            key = (r["l2_norm"], r["filter"], r["model"], r["group"])
            grouped[key].append(r["perm_pval"])

    if not grouped:
        print("  No permutation p-values found.")
        return

    for key in sorted(grouped.keys()):
        l2, filt, model, grp = key
        pvals = np.array(grouped[key])
        print(
            f"\n  ── L2={l2} | filter={filt} | {model} | {grp} ({len(pvals)} seeds) ──"
        )
        print(
            f"    Mean p = {pvals.mean():.4f}, Median p = {np.median(pvals):.4f}, "
            f"Min = {pvals.min():.4f}, Max = {pvals.max():.4f}"
        )
        n_sig = np.sum(pvals < 0.05)
        print(f"    Significant (p<0.05): {n_sig}/{len(pvals)}")


# ==============================================================================
# Global pooled permutation p-value
# ==============================================================================
def compute_pooled_perm_pval(per_seed, all_folds):
    """
    Pool all null fold R²s across seeds → compute global permutation p-value.
    For each (l2_norm, filter, model, group):
      real_r2 = mean of all real fold R²s
      null_pool = all perm_fold_r2s from every seed
      p = (# null >= real_r2 + 1) / (len(null_pool) + 1)
    """
    # Real fold R²s grouped by condition
    real_grouped = defaultdict(list)
    for row in all_folds:
        key = (row["l2_norm"], row["filter"], row["model"], row["group"])
        real_grouped[key].append(row["r2"])

    # Null fold R²s grouped by condition
    null_grouped = defaultdict(list)
    for r in per_seed:
        if not r.get("perm_fold_r2s"):
            continue
        key = (r["l2_norm"], r["filter"], r["model"], r["group"])
        null_grouped[key].extend(r["perm_fold_r2s"])

    results = []
    for key in sorted(real_grouped.keys()):
        real_vals = np.array(real_grouped[key])
        null_vals = np.array(null_grouped.get(key, []))
        if len(null_vals) == 0:
            continue

        real_mean = real_vals.mean()
        p_value = (np.sum(null_vals >= real_mean) + 1) / (len(null_vals) + 1)

        results.append(
            {
                "l2_norm": key[0],
                "filter": key[1],
                "model": key[2],
                "group": key[3],
                "real_mean_r2": float(real_mean),
                "n_real_folds": len(real_vals),
                "null_mean_r2": float(null_vals.mean()),
                "n_null_folds": len(null_vals),
                "pooled_p_value": float(p_value),
            }
        )

    return results


def print_pooled_perm(pooled_perm):
    """Print global pooled permutation p-value."""
    if not pooled_perm:
        print("\n  No pooled permutation results (no perm_fold_r2s in JSONs).")
        return

    print("\n" + "=" * 100)
    print("  GLOBAL POOLED PERMUTATION P-VALUE (all seeds pooled)")
    print("=" * 100)

    for r in pooled_perm:
        if r["group"] not in GROUPS_OF_INTEREST:
            continue
        sig = "✅" if r["pooled_p_value"] < 0.05 else "  "
        print(
            f"  {sig} L2={r['l2_norm']:12s} | filter={r['filter']:15s} | "
            f"{r['model']:8s} | {r['group']:12s}: "
            f"p = {r['pooled_p_value']:.4f}  "
            f"(real={r['real_mean_r2']:.4f}, null={r['null_mean_r2']:.4f}, "
            f"n_real={r['n_real_folds']}, n_null={r['n_null_folds']})"
        )


# ==============================================================================
# Save CSVs
# ==============================================================================
def save_csvs(
    results_dir,
    agg,
    per_seed,
    all_folds,
    ci_results,
    wilcoxon_results,
    pooled_perm=None,
):
    """Save all CSV files."""

    # 1. Per-seed CSV (backward compatible)
    csv_path = os.path.join(results_dir, "sae_r2_per_seed.csv")
    rows_sorted = sorted(
        per_seed, key=lambda x: (x["l2_norm"], x["filter"], x["model"], x["sae_seed"])
    )
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "CNN_Seed",
                "GAP_L2_Norm",
                "Filter",
                "Model",
                "Group",
                "R2_mean",
                "R2_std",
                "Perm_pval",
                "R2_fold_scores",
            ]
        )
        for r in rows_sorted:
            w.writerow(
                [
                    r["sae_seed"],
                    r["l2_norm"],
                    r["filter"],
                    r["model"],
                    r["group"],
                    f"{r['r2_mean']:.6f}",
                    f"{r['r2_std']:.6f}",
                    f"{r['perm_pval']:.4f}" if r["perm_pval"] is not None else "",
                    (
                        ";".join(f"{s:.6f}" for s in r["r2_scores"])
                        if r["r2_scores"]
                        else ""
                    ),
                ]
            )
    print(f"\n  Saved per-seed CSV: {csv_path}")

    # 2. All folds CSV
    folds_csv = os.path.join(results_dir, "sae_r2_all_folds.csv")
    folds_sorted = sorted(
        all_folds,
        key=lambda x: (
            x["l2_norm"],
            x["filter"],
            x["model"],
            x["sae_seed"],
            x["fold_idx"],
        ),
    )
    with open(folds_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["CNN_Seed", "GAP_L2_Norm", "Filter", "Model", "Group", "Fold_idx", "R2"]
        )
        for r in folds_sorted:
            w.writerow(
                [
                    r["sae_seed"],
                    r["l2_norm"],
                    r["filter"],
                    r["model"],
                    r["group"],
                    r["fold_idx"],
                    f"{r['r2']:.6f}",
                ]
            )
    print(f"  Saved all-folds CSV: {folds_csv}")

    # 3. Summary CSV
    summary_path = os.path.join(results_dir, "sae_r2_summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["GAP_L2_Norm", "Filter", "Model", "Group", "Mean_R2", "Std_R2", "N_seeds"]
        )
        for key in sorted(agg.keys()):
            l2_label, filter_label, model_tag, grp = key
            vals = agg[key]
            w.writerow(
                [
                    l2_label,
                    filter_label,
                    model_tag,
                    grp,
                    f"{np.mean(vals):.6f}",
                    f"{np.std(vals):.6f}",
                    len(vals),
                ]
            )
    print(f"  Saved summary CSV:  {summary_path}")

    # 4. 95% CI CSV
    ci_csv = os.path.join(results_dir, "sae_r2_ci95.csv")
    with open(ci_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "GAP_L2_Norm",
                "Filter",
                "Model",
                "Group",
                "Mean_R2",
                "CI95_lower",
                "CI95_upper",
                "Std",
                "N_folds",
            ]
        )
        for r in ci_results:
            w.writerow(
                [
                    r["l2_norm"],
                    r["filter"],
                    r["model"],
                    r["group"],
                    f"{r['mean_r2']:.6f}",
                    f"{r['ci95_lower']:.6f}",
                    f"{r['ci95_upper']:.6f}",
                    f"{r['std']:.6f}",
                    r["n_folds"],
                ]
            )
    print(f"  Saved 95% CI CSV:   {ci_csv}")

    # 5. Wilcoxon CSV
    if wilcoxon_results:
        wilcoxon_csv = os.path.join(results_dir, "sae_r2_wilcoxon.csv")
        with open(wilcoxon_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "Filter",
                    "Model",
                    "Group",
                    "N_pairs",
                    "Median_L2on",
                    "Median_L2off",
                    "Mean_L2on",
                    "Mean_L2off",
                    "Median_diff",
                    "W_stat",
                    "P_value",
                ]
            )
            for r in wilcoxon_results:
                w.writerow(
                    [
                        r["filter"],
                        r["model"],
                        r["group"],
                        r["n_pairs"],
                        f"{r['median_l2on']:.6f}",
                        f"{r['median_l2off']:.6f}",
                        f"{r['mean_l2on']:.6f}",
                        f"{r['mean_l2off']:.6f}",
                        f"{r['median_diff']:.6f}",
                        f"{r['W_stat']:.1f}",
                        f"{r['p_value']:.6f}",
                    ]
                )
        print(f"  Saved Wilcoxon CSV: {wilcoxon_csv}")

    # 6. Pooled permutation p-value CSV
    if pooled_perm:
        perm_csv = os.path.join(results_dir, "sae_r2_pooled_perm.csv")
        with open(perm_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "GAP_L2_Norm",
                    "Filter",
                    "Model",
                    "Group",
                    "Real_mean_R2",
                    "N_real_folds",
                    "Null_mean_R2",
                    "N_null_folds",
                    "Pooled_p_value",
                ]
            )
            for r in pooled_perm:
                w.writerow(
                    [
                        r["l2_norm"],
                        r["filter"],
                        r["model"],
                        r["group"],
                        f"{r['real_mean_r2']:.6f}",
                        r["n_real_folds"],
                        f"{r['null_mean_r2']:.6f}",
                        r["n_null_folds"],
                        f"{r['pooled_p_value']:.6f}",
                    ]
                )
        print(f"  Saved pooled perm CSV: {perm_csv}")


# ==============================================================================
# Plot: Paired comparison (L2 ON vs OFF)
# ==============================================================================
def plot_paired_comparison(per_seed, results_dir):
    """
    Paired dot plot: each SAE seed's mean R² connected by a line.
    One subplot per (model, group).
    """
    # Collect: (filter, model, group, l2_norm) → {sae_seed: r2_mean}
    lookup = defaultdict(dict)
    for r in per_seed:
        key = (r["filter"], r["model"], r["group"], r["l2_norm"])
        lookup[key][r["sae_seed"]] = r["r2_mean"]

    filter_labels = sorted(set(r["filter"] for r in per_seed))

    for filt in filter_labels:
        for model in MODEL_TAGS:
            groups_present = [
                g
                for g in GROUPS_OF_INTEREST
                if lookup.get((filt, model, g, "l2norm"))
                and lookup.get((filt, model, g, "no_l2norm"))
            ]
            if not groups_present:
                continue

            fig, axes = plt.subplots(
                1, len(groups_present), figsize=(5 * len(groups_present), 5)
            )
            if len(groups_present) == 1:
                axes = [axes]

            for ax, grp in zip(axes, groups_present):
                l2on_dict = lookup[(filt, model, grp, "l2norm")]
                l2off_dict = lookup[(filt, model, grp, "no_l2norm")]
                common_seeds = sorted(set(l2on_dict.keys()) & set(l2off_dict.keys()))

                for seed in common_seeds:
                    ax.plot(
                        [0, 1],
                        [l2off_dict[seed], l2on_dict[seed]],
                        "o-",
                        color="#555555",
                        alpha=0.5,
                        markersize=6,
                    )

                # Means
                if common_seeds:
                    mean_off = np.mean([l2off_dict[s] for s in common_seeds])
                    mean_on = np.mean([l2on_dict[s] for s in common_seeds])
                    ax.plot(
                        [0, 1],
                        [mean_off, mean_on],
                        "s-",
                        color="#E24A33",
                        markersize=10,
                        linewidth=2.5,
                        label=f"Mean",
                        zorder=5,
                    )

                ax.set_xticks([0, 1])
                ax.set_xticklabels(["L2 OFF", "L2 ON"], fontsize=11)
                ax.set_ylabel("R² (mean across folds)", fontsize=11)
                ax.set_title(
                    f"{grp}\n{model} | filter={filt}", fontsize=12, fontweight="bold"
                )
                ax.grid(True, alpha=0.2, axis="y")
                ax.legend(fontsize=9)

            fig.suptitle(
                "SAE Vector: GAP L2 Norm Effect (Paired by CNN Seed)",
                fontsize=14,
                fontweight="bold",
                y=1.02,
            )
            fig.tight_layout()
            safe = f"paired_{model}_{filt}".replace(" ", "_")
            save_path = os.path.join(results_dir, f"{safe}.png")
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved paired plot: {save_path}")


# ==============================================================================
# Plot: Forest plot (mean ± 95% CI)
# ==============================================================================
def plot_forest_ci(ci_results, results_dir):
    """
    Forest plot: mean R² with 95% CI error bars.
    L2 ON vs OFF side by side.
    """
    for model in MODEL_TAGS:
        # Get relevant results
        model_results = [
            r
            for r in ci_results
            if r["model"] == model and r["group"] in GROUPS_OF_INTEREST
        ]
        if not model_results:
            continue

        filter_labels = sorted(set(r["filter"] for r in model_results))

        for filt in filter_labels:
            filt_results = [r for r in model_results if r["filter"] == filt]
            if not filt_results:
                continue

            groups_present = [
                g
                for g in GROUPS_OF_INTEREST
                if any(r["group"] == g for r in filt_results)
            ]
            if not groups_present:
                continue

            fig, ax = plt.subplots(figsize=(8, max(3, len(groups_present) * 1.2)))

            y_positions = []
            y_labels = []
            y_offset = 0

            for grp in groups_present:
                for l2 in ["l2norm", "no_l2norm"]:
                    matches = [
                        r
                        for r in filt_results
                        if r["group"] == grp and r["l2_norm"] == l2
                    ]
                    if not matches:
                        continue
                    r = matches[0]
                    color = "#E24A33" if l2 == "l2norm" else "#348ABD"
                    label_str = f"L2 {'ON' if l2 == 'l2norm' else 'OFF'}"

                    ci_err = [
                        [r["mean_r2"] - r["ci95_lower"]],
                        [r["ci95_upper"] - r["mean_r2"]],
                    ]

                    ax.errorbar(
                        r["mean_r2"],
                        y_offset,
                        xerr=ci_err,
                        fmt="o",
                        color=color,
                        markersize=8,
                        capsize=5,
                        capthick=1.5,
                        linewidth=1.5,
                        label=label_str if y_offset < 2 else None,
                    )

                    ax.text(
                        r["ci95_upper"] + 0.002,
                        y_offset,
                        f"  {r['mean_r2']:.4f} [{r['ci95_lower']:.4f}, {r['ci95_upper']:.4f}]",
                        va="center",
                        fontsize=8,
                        color=color,
                    )

                    y_positions.append(y_offset)
                    y_labels.append(f"{grp} ({label_str})")
                    y_offset += 1

                y_offset += 0.5  # gap between groups

            ax.set_yticks(y_positions)
            ax.set_yticklabels(y_labels, fontsize=9)
            ax.set_xlabel("R²", fontsize=11)
            ax.set_title(
                f"SAE {model} | filter={filt}\n95% CI (all folds pooled)",
                fontsize=12,
                fontweight="bold",
            )
            ax.axvline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
            ax.grid(True, alpha=0.15, axis="x")
            ax.invert_yaxis()

            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles[:2], labels[:2], loc="lower right", fontsize=9)

            fig.tight_layout()
            safe = f"forest_{model}_{filt}".replace(" ", "_")
            save_path = os.path.join(results_dir, f"{safe}.png")
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved forest plot: {save_path}")


# ==============================================================================
# Print Wilcoxon results
# ==============================================================================
def print_wilcoxon_results(wilcoxon_results):
    """Print Wilcoxon signed-rank test results."""
    if not wilcoxon_results:
        print("\n  No Wilcoxon results (need both l2norm and no_l2norm data).")
        return

    print("\n" + "=" * 100)
    print("  WILCOXON SIGNED-RANK TEST: GAP L2 norm ON vs OFF")
    print("  (paired by CNN seed × fold index)")
    print("=" * 100)

    for r in wilcoxon_results:
        sig = "✅ *" if r["p_value"] < 0.05 else "  "
        print(f"\n  {sig} filter={r['filter']} | {r['model']} | {r['group']}")
        print(f"    N pairs = {r['n_pairs']}")
        print(
            f"    Median R²:  L2 ON = {r['median_l2on']:.4f},  "
            f"L2 OFF = {r['median_l2off']:.4f},  Δ = {r['median_diff']:+.4f}"
        )
        print(
            f"    Mean R²:    L2 ON = {r['mean_l2on']:.4f},  "
            f"L2 OFF = {r['mean_l2off']:.4f}"
        )
        print(f"    W = {r['W_stat']:.1f},  p = {r['p_value']:.6f}")

    # Summary
    n_sig = sum(1 for r in wilcoxon_results if r["p_value"] < 0.05)
    print(f"\n  ── {n_sig}/{len(wilcoxon_results)} tests significant at α=0.05 ──")


# ==============================================================================
# Print CI results
# ==============================================================================
def print_ci_results(ci_results):
    """Print 95% CI summary."""
    print("\n" + "=" * 100)
    print("  95% CONFIDENCE INTERVALS (all fold R²s pooled across seeds)")
    print("=" * 100)

    for r in ci_results:
        if r["group"] not in GROUPS_OF_INTEREST:
            continue
        print(
            f"  L2={r['l2_norm']:12s} | filter={r['filter']:15s} | {r['model']:8s} | "
            f"{r['group']:12s}: "
            f"{r['mean_r2']:.4f} [{r['ci95_lower']:.4f}, {r['ci95_upper']:.4f}]  "
            f"(n={r['n_folds']})"
        )


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Aggregate SAE vector apoptosis R² results"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Path to SAE_vector results directory",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        print(f"ERROR: not found: {args.results_dir}")
        sys.exit(1)

    # ── Collect ──
    agg, per_seed, all_folds = collect_results(args.results_dir)
    if not agg:
        print("ERROR: No results found.")
        sys.exit(1)

    print(
        f"\n  Found {len(per_seed)} seed-level entries, {len(all_folds)} fold-level entries."
    )

    # ── Print basic summary ──
    print_summary(agg, per_seed)

    # ── Permutation p-value summary ──
    print_perm_summary(per_seed)

    # ── Global pooled permutation p-value ──
    pooled_perm = compute_pooled_perm_pval(per_seed, all_folds)
    print_pooled_perm(pooled_perm)

    # ── 95% CI ──
    ci_results = compute_ci95(all_folds)
    print_ci_results(ci_results)

    # ── Wilcoxon ──
    wilcoxon_results = compute_wilcoxon(all_folds)
    print_wilcoxon_results(wilcoxon_results)

    # ── Save CSVs ──
    print("\n" + "=" * 100)
    print("  SAVING OUTPUT FILES")
    print("=" * 100)
    save_csvs(
        args.results_dir,
        agg,
        per_seed,
        all_folds,
        ci_results,
        wilcoxon_results,
        pooled_perm,
    )

    # ── Plots ──
    print("\n" + "=" * 100)
    print("  GENERATING PLOTS")
    print("=" * 100)
    plot_paired_comparison(per_seed, args.results_dir)
    plot_forest_ci(ci_results, args.results_dir)

    print("\n" + "=" * 100)
    print("  DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()
