# ==============================================================================
# Aggregate R² results across all CNN seeds — Paper-ready summary
#
# Scans all JSON result files produced by apoptosis_r2_test.py,
# computes per-(config, model, layer, group) statistics:
#   - Mean R² ± Std across seeds
#   - 95% confidence intervals (from per-fold R²s)
#   - Wilcoxon signed-rank test: L2 norm ON vs OFF (paired by seed × layer × fold)
#   - Individual seed R²s
#   - LaTeX-ready table
#
# Outputs:
#   1. Console: aggregated tables, CI, Wilcoxon results
#   2. CSV:     aggregated_r2_summary.csv   (mean/std per condition)
#   3. CSV:     aggregated_r2_per_seed.csv  (every seed's result)
#   4. CSV:     aggregated_r2_all_folds.csv (every individual fold R²)
#   5. CSV:     aggregated_r2_ci95.csv      (95% CI per condition)
#   6. CSV:     aggregated_r2_wilcoxon.csv  (Wilcoxon test results)
#   7. CSV:     paper_table_r2.csv          (논문용 한 줄 요약)
#   8. LaTeX:   paper_table_r2.tex          (LaTeX table source)
#   9. Plots:   paired & forest plots
#
# Usage (Colab — notebook cell):
#   import sys
#   sys.argv = [
#       "aggregate_r2_results",
#       "--results_dir", "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/apoptosis_r2_results",
#   ]
#   from kendall_correlation_coefficient.aggregate_r2_results import main
#   main()
#
# Usage (terminal):
#   python -m kendall_correlation_coefficient.aggregate_r2_results \
#       --results_dir /content/drive/MyDrive/Final_paper/lambda_labs_moco_only/apoptosis_r2_results
# ==============================================================================

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import matplotlib
import numpy as np

if "google.colab" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==============================================================================
# Config: which directories / seeds to scan
# ==============================================================================
CONFIGS = [
    # (label, directory_pattern, seeds, description)
    (
        "MoCo_l2norm",
        "MoCo_seed{}_l2norm",
        [42, 87, 95, 123, 124, 256, 445, 457],
        "MoCo (GAP L2 norm training), L2 norm ON",
    ),
    (
        "MoCo_raw",
        "MoCo_seed{}_raw",
        [42, 87, 95, 123, 124, 256, 445, 457],
        "MoCo (GAP L2 norm training), L2 norm OFF",
    ),
    (
        "noNorm_l2norm",
        "noNorm_seed{}_l2norm",
        [42, 87, 124],
        "MoCo (no GAP L2 norm training), L2 norm ON",
    ),
    (
        "noNorm_raw",
        "noNorm_seed{}_raw",
        [42, 87, 124],
        "MoCo (no GAP L2 norm training), L2 norm OFF",
    ),
]

LAYERS = ["stage5_mid", "stage5_out", "refine_out"]
MODEL_TAGS = ["Ridge", "XGBoost"]
GROUPS_OF_INTEREST = ["SNCA only", "GBA only", "LRRK2 only"]


# ==============================================================================
# Scan & collect
# ==============================================================================
def collect_all_results(results_dir: str):
    """
    Scan results_dir for all JSON files and collect R² values.

    Returns
    -------
    agg : dict
        key = (config_label, model, layer, group)
        value = list of r2_mean (one per seed)
    per_seed : list[dict]
        Each dict has config, seed, model, layer, group, r2_mean, r2_std, r2_scores, perm_pval
    all_folds : list[dict]
        Every individual fold R² with config, seed, model, layer, group, fold_idx, r2
    missing : list[str]
        List of expected but missing JSON paths
    """
    agg = defaultdict(list)
    per_seed = []
    all_folds = []
    missing = []

    for cfg_label, dir_pat, seeds, _desc in CONFIGS:
        for seed in seeds:
            seed_dir = os.path.join(results_dir, dir_pat.format(seed))
            for layer in LAYERS:
                for model_tag in MODEL_TAGS:
                    json_path = os.path.join(
                        seed_dir, f"r2_results_{layer}_{model_tag}.json"
                    )
                    if not os.path.exists(json_path):
                        missing.append(json_path)
                        continue

                    with open(json_path) as f:
                        data = json.load(f)

                    for res in data.get("results", []):
                        grp = res["group"]
                        r2_mean = res["r2_mean"]
                        r2_std = res.get("r2_std", 0.0)
                        r2_scores = res.get("r2_scores", [])
                        perm_pval = res.get("perm_pval", None)

                        key = (cfg_label, model_tag, layer, grp)
                        agg[key].append(r2_mean)

                        per_seed.append(
                            {
                                "config": cfg_label,
                                "seed": seed,
                                "model": model_tag,
                                "layer": layer,
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
                                    "config": cfg_label,
                                    "seed": seed,
                                    "model": model_tag,
                                    "layer": layer,
                                    "group": grp,
                                    "fold_idx": fold_i,
                                    "r2": fold_r2,
                                }
                            )

    return dict(agg), per_seed, all_folds, missing


# ==============================================================================
# Print aggregated table
# ==============================================================================
def print_aggregated_table(agg):
    """Print mean ± std across seeds for each condition."""
    print("\n" + "=" * 110)
    print("  AGGREGATED R² (mean ± std across seeds)")
    print("=" * 110)

    for cfg_label, _dir_pat, seeds, desc in CONFIGS:
        for model_tag in MODEL_TAGS:
            for layer in LAYERS:
                vals_exist = any(
                    agg.get((cfg_label, model_tag, layer, g))
                    for g in GROUPS_OF_INTEREST
                )
                if not vals_exist:
                    continue

                n_seeds = len(
                    [
                        v
                        for g in GROUPS_OF_INTEREST
                        for v in agg.get((cfg_label, model_tag, layer, g), [])
                    ]
                ) // max(
                    sum(
                        1
                        for g in GROUPS_OF_INTEREST
                        if agg.get((cfg_label, model_tag, layer, g))
                    ),
                    1,
                )

                print(f"\n  ── {desc} | {model_tag} | {layer} ({n_seeds} seeds) ──")
                print(
                    f"  {'Group':20s} {'Mean R²':>10s} {'±Std':>10s} {'Min':>10s} {'Max':>10s} {'N':>5s}"
                )
                print("  " + "-" * 60)

                for grp in GROUPS_OF_INTEREST:
                    key = (cfg_label, model_tag, layer, grp)
                    vals = agg.get(key, [])
                    if vals:
                        m = np.mean(vals)
                        s = np.std(vals)
                        mn = np.min(vals)
                        mx = np.max(vals)
                        print(
                            f"  {grp:20s} {m:>10.4f} {s:>10.4f} {mn:>10.4f} {mx:>10.4f} {len(vals):>5d}"
                        )


# ==============================================================================
# Print per-seed table
# ==============================================================================
def print_per_seed_table(agg, per_seed):
    """Print R² for each seed individually."""
    print("\n\n" + "=" * 110)
    print("  PER-SEED R²")
    print("=" * 110)

    for cfg_label, _dir_pat, seeds, desc in CONFIGS:
        for model_tag in MODEL_TAGS:
            for layer in LAYERS:
                vals_exist = any(
                    agg.get((cfg_label, model_tag, layer, g))
                    for g in GROUPS_OF_INTEREST
                )
                if not vals_exist:
                    continue

                print(f"\n  ── {desc} | {model_tag} | {layer} ──")
                header = f"  {'Seed':>6s}"
                for grp in GROUPS_OF_INTEREST:
                    header += f"  {grp:>14s}"
                print(header)
                print("  " + "-" * (6 + 16 * len(GROUPS_OF_INTEREST)))

                for seed in seeds:
                    row = f"  {seed:>6d}"
                    for grp in GROUPS_OF_INTEREST:
                        matches = [
                            r
                            for r in per_seed
                            if r["config"] == cfg_label
                            and r["seed"] == seed
                            and r["model"] == model_tag
                            and r["layer"] == layer
                            and r["group"] == grp
                        ]
                        if matches:
                            row += f"  {matches[0]['r2_mean']:>14.4f}"
                        else:
                            row += f"  {'N/A':>14s}"
                    print(row)

                # Mean ± Std row
                row_mean = f"  {'Mean':>6s}"
                row_std = f"  {'±Std':>6s}"
                for grp in GROUPS_OF_INTEREST:
                    key = (cfg_label, model_tag, layer, grp)
                    vals = agg.get(key, [])
                    if vals:
                        row_mean += f"  {np.mean(vals):>14.4f}"
                        row_std += f"  {np.std(vals):>14.4f}"
                    else:
                        row_mean += f"  {'N/A':>14s}"
                        row_std += f"  {'N/A':>14s}"
                print("  " + "-" * (6 + 16 * len(GROUPS_OF_INTEREST)))
                print(row_mean)
                print(row_std)


# ==============================================================================
# 95% Confidence Intervals
# ==============================================================================
def compute_ci95(all_folds):
    """Compute 95% CI per (config, model, layer, group) using all fold R²s."""
    from scipy import stats

    grouped = defaultdict(list)
    for row in all_folds:
        key = (row["config"], row["model"], row["layer"], row["group"])
        grouped[key].append(row["r2"])

    ci_results = []
    for key, vals in sorted(grouped.items()):
        cfg, model, layer, grp = key
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
                "config": cfg,
                "model": model,
                "layer": layer,
                "group": grp,
                "mean_r2": mean,
                "ci95_lower": ci_low,
                "ci95_upper": ci_high,
                "n_folds": n,
                "std": arr.std(ddof=1) if n > 1 else 0.0,
            }
        )

    return ci_results


def print_ci_results(ci_results):
    """Print 95% CI summary."""
    print("\n" + "=" * 110)
    print("  95% CONFIDENCE INTERVALS (all fold R²s pooled across seeds)")
    print("=" * 110)

    for r in ci_results:
        if r["group"] not in GROUPS_OF_INTEREST:
            continue
        print(
            f"  {r['config']:16s} | {r['model']:8s} | {r['layer']:12s} | "
            f"{r['group']:12s}: "
            f"{r['mean_r2']:.4f} [{r['ci95_lower']:.4f}, {r['ci95_upper']:.4f}]  "
            f"(n={r['n_folds']})"
        )


# ==============================================================================
# Wilcoxon signed-rank test: L2 ON vs OFF
# ==============================================================================
def compute_wilcoxon(all_folds):
    """
    Compare l2norm ON vs OFF configs.
    Pairs: (MoCo_l2norm vs MoCo_raw) and (noNorm_l2norm vs noNorm_raw)
    Paired by (seed, layer, fold_idx).
    """
    from scipy.stats import wilcoxon

    # Define L2 ON/OFF config pairs
    config_pairs = [
        ("MoCo_l2norm", "MoCo_raw", "MoCo (GAP L2 norm trained)"),
        ("noNorm_l2norm", "noNorm_raw", "MoCo (no GAP L2 norm trained)"),
    ]

    # Build lookup: (config, model, layer, group) → {(seed, fold_idx): r2}
    lookup = defaultdict(dict)
    for row in all_folds:
        condition_key = (row["config"], row["model"], row["layer"], row["group"])
        pair_key = (row["seed"], row["fold_idx"])
        lookup[condition_key][pair_key] = row["r2"]

    results = []
    for cfg_on, cfg_off, pair_desc in config_pairs:
        for model in MODEL_TAGS:
            for layer in LAYERS:
                for grp in GROUPS_OF_INTEREST:
                    on_key = (cfg_on, model, layer, grp)
                    off_key = (cfg_off, model, layer, grp)
                    on_dict = lookup.get(on_key, {})
                    off_dict = lookup.get(off_key, {})

                    if not on_dict or not off_dict:
                        continue

                    common_keys = sorted(set(on_dict.keys()) & set(off_dict.keys()))
                    if len(common_keys) < 5:
                        continue

                    r2_on = np.array([on_dict[k] for k in common_keys])
                    r2_off = np.array([off_dict[k] for k in common_keys])
                    diff = r2_on - r2_off

                    try:
                        stat, pval = wilcoxon(diff, alternative="two-sided")
                    except ValueError:
                        stat, pval = 0.0, 1.0

                    results.append(
                        {
                            "pair_desc": pair_desc,
                            "config_on": cfg_on,
                            "config_off": cfg_off,
                            "model": model,
                            "layer": layer,
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


def print_wilcoxon_results(wilcoxon_results):
    """Print Wilcoxon signed-rank test results."""
    if not wilcoxon_results:
        print("\n  No Wilcoxon results (need both L2 ON and OFF data).")
        return

    print("\n" + "=" * 110)
    print("  WILCOXON SIGNED-RANK TEST: GAP L2 norm ON vs OFF")
    print("  (paired by CNN seed × fold index)")
    print("=" * 110)

    for r in wilcoxon_results:
        sig = "✅ *" if r["p_value"] < 0.05 else "  "
        print(
            f"\n  {sig} {r['pair_desc']} | {r['model']} | {r['layer']} | {r['group']}"
        )
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

    n_sig = sum(1 for r in wilcoxon_results if r["p_value"] < 0.05)
    print(f"\n  ── {n_sig}/{len(wilcoxon_results)} tests significant at α=0.05 ──")


# ==============================================================================
# Permutation p-value summary
# ==============================================================================
def print_perm_summary(per_seed):
    """Print permutation p-value summary."""
    print("\n" + "=" * 110)
    print("  PERMUTATION P-VALUE SUMMARY")
    print("=" * 110)

    grouped = defaultdict(list)
    for r in per_seed:
        if r.get("perm_pval") is not None:
            key = (r["config"], r["model"], r["layer"], r["group"])
            grouped[key].append(r["perm_pval"])

    if not grouped:
        print("  No permutation p-values found.")
        return

    for key in sorted(grouped.keys()):
        cfg, model, layer, grp = key
        if grp not in GROUPS_OF_INTEREST:
            continue
        pvals = np.array(grouped[key])
        n_sig = np.sum(pvals < 0.05)
        print(
            f"  {cfg:16s} | {model:8s} | {layer:12s} | {grp:12s}: "
            f"mean p={pvals.mean():.4f}, median p={np.median(pvals):.4f} "
            f"[{pvals.min():.4f}-{pvals.max():.4f}], "
            f"sig={n_sig}/{len(pvals)}"
        )


# ==============================================================================
# Global pooled permutation p-value
# ==============================================================================
def compute_pooled_perm_pval(per_seed, all_folds):
    """
    Pool all null fold R²s across seeds → compute global permutation p-value.
    For each (config, model, layer, group):
      real_r2 = mean of all real fold R²s
      null_pool = all perm_fold_r2s from every seed
      p = (# null >= real_r2 + 1) / (len(null_pool) + 1)
    """
    real_grouped = defaultdict(list)
    for row in all_folds:
        key = (row["config"], row["model"], row["layer"], row["group"])
        real_grouped[key].append(row["r2"])

    null_grouped = defaultdict(list)
    for r in per_seed:
        if not r.get("perm_fold_r2s"):
            continue
        key = (r["config"], r["model"], r["layer"], r["group"])
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
                "config": key[0],
                "model": key[1],
                "layer": key[2],
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

    print("\n" + "=" * 110)
    print("  GLOBAL POOLED PERMUTATION P-VALUE (all seeds pooled)")
    print("=" * 110)

    for r in pooled_perm:
        if r["group"] not in GROUPS_OF_INTEREST:
            continue
        sig = "✅" if r["pooled_p_value"] < 0.05 else "  "
        print(
            f"  {sig} {r['config']:16s} | {r['model']:8s} | {r['layer']:12s} | "
            f"{r['group']:12s}: "
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
    """Save per-seed CSV, all-folds CSV, summary CSV, CI CSV, Wilcoxon CSV, pooled perm CSV."""

    # 1. Per-seed CSV
    per_seed_csv = os.path.join(results_dir, "aggregated_r2_per_seed.csv")
    rows_sorted = sorted(
        per_seed, key=lambda x: (x["config"], x["model"], x["layer"], x["seed"])
    )
    with open(per_seed_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "Config",
                "Seed",
                "Model",
                "Layer",
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
                    r["config"],
                    r["seed"],
                    r["model"],
                    r["layer"],
                    r["group"],
                    f"{r['r2_mean']:.6f}",
                    f"{r['r2_std']:.6f}",
                    f"{r['perm_pval']:.4f}" if r.get("perm_pval") is not None else "",
                    (
                        ";".join(f"{s:.6f}" for s in r["r2_scores"])
                        if r["r2_scores"]
                        else ""
                    ),
                ]
            )
    print(f"\n  Saved per-seed CSV:  {per_seed_csv}")

    # 2. All-folds CSV
    folds_csv = os.path.join(results_dir, "aggregated_r2_all_folds.csv")
    folds_sorted = sorted(
        all_folds,
        key=lambda x: (x["config"], x["model"], x["layer"], x["seed"], x["fold_idx"]),
    )
    with open(folds_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Config", "Seed", "Model", "Layer", "Group", "Fold_idx", "R2"])
        for r in folds_sorted:
            w.writerow(
                [
                    r["config"],
                    r["seed"],
                    r["model"],
                    r["layer"],
                    r["group"],
                    r["fold_idx"],
                    f"{r['r2']:.6f}",
                ]
            )
    print(f"  Saved all-folds CSV: {folds_csv}")

    # 3. Aggregated summary CSV
    agg_csv = os.path.join(results_dir, "aggregated_r2_summary.csv")
    with open(agg_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "Config",
                "Model",
                "Layer",
                "Group",
                "Mean_R2",
                "Std_R2",
                "Min_R2",
                "Max_R2",
                "N_seeds",
            ]
        )
        for cfg_label, _, seeds, _ in CONFIGS:
            for model_tag in MODEL_TAGS:
                for layer in LAYERS:
                    for grp in GROUPS_OF_INTEREST:
                        key = (cfg_label, model_tag, layer, grp)
                        vals = agg.get(key, [])
                        if vals:
                            w.writerow(
                                [
                                    cfg_label,
                                    model_tag,
                                    layer,
                                    grp,
                                    f"{np.mean(vals):.6f}",
                                    f"{np.std(vals):.6f}",
                                    f"{np.min(vals):.6f}",
                                    f"{np.max(vals):.6f}",
                                    len(vals),
                                ]
                            )
    print(f"  Saved summary CSV:  {agg_csv}")

    # 4. 95% CI CSV
    ci_csv = os.path.join(results_dir, "aggregated_r2_ci95.csv")
    with open(ci_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "Config",
                "Model",
                "Layer",
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
                    r["config"],
                    r["model"],
                    r["layer"],
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
        wilcoxon_csv = os.path.join(results_dir, "aggregated_r2_wilcoxon.csv")
        with open(wilcoxon_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "Pair_Desc",
                    "Config_ON",
                    "Config_OFF",
                    "Model",
                    "Layer",
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
                        r["pair_desc"],
                        r["config_on"],
                        r["config_off"],
                        r["model"],
                        r["layer"],
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
        perm_csv = os.path.join(results_dir, "aggregated_r2_pooled_perm.csv")
        with open(perm_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "Config",
                    "Model",
                    "Layer",
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
                        r["config"],
                        r["model"],
                        r["layer"],
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
# Paper table: one-line-per-condition
# ==============================================================================
def save_paper_table(results_dir, agg):
    """논문용 테이블 (CSV + LaTeX)."""
    paper_csv = os.path.join(results_dir, "paper_table_r2.csv")
    paper_tex = os.path.join(results_dir, "paper_table_r2.tex")

    csv_rows = []
    header = ["Config", "Model", "Layer"] + GROUPS_OF_INTEREST
    csv_rows.append(header)

    for cfg_label, _, seeds, desc in CONFIGS:
        for model_tag in MODEL_TAGS:
            for layer in LAYERS:
                vals_exist = any(
                    agg.get((cfg_label, model_tag, layer, g))
                    for g in GROUPS_OF_INTEREST
                )
                if not vals_exist:
                    continue

                row = [cfg_label, model_tag, layer]
                for grp in GROUPS_OF_INTEREST:
                    key = (cfg_label, model_tag, layer, grp)
                    vals = agg.get(key, [])
                    if vals:
                        m, s = np.mean(vals), np.std(vals)
                        row.append(f"{m:.4f} ± {s:.4f}")
                    else:
                        row.append("N/A")
                csv_rows.append(row)

    with open(paper_csv, "w", newline="") as f:
        w = csv.writer(f)
        for row in csv_rows:
            w.writerow(row)
    print(f"  Saved paper CSV:    {paper_csv}")

    # LaTeX
    col_spec = "l" * 3 + "c" * len(GROUPS_OF_INTEREST)
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Cross-validated $R^2$ for apoptosis rate prediction (mean $\pm$ std across CNN training seeds).}"
    )
    lines.append(r"\label{tab:r2_apoptosis}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    h = " & ".join(
        ["Config", "Model", "Layer"]
        + [g.replace("_", r"\_") for g in GROUPS_OF_INTEREST]
    )
    lines.append(h + r" \\")
    lines.append(r"\midrule")

    prev_config = None
    for row in csv_rows[1:]:
        cfg = row[0]
        if prev_config is not None and cfg != prev_config:
            lines.append(r"\midrule")
        prev_config = cfg

        escaped = []
        for cell in row:
            cell_str = str(cell)
            cell_str = cell_str.replace("_", r"\_")
            cell_str = cell_str.replace("±", r"$\pm$")
            escaped.append(cell_str)
        lines.append(" & ".join(escaped) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table}")

    with open(paper_tex, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved LaTeX table:  {paper_tex}")


# ==============================================================================
# Best condition finder
# ==============================================================================
def print_best_conditions(agg):
    """Find and print the best (config, model, layer) per group."""
    print("\n" + "=" * 110)
    print("  BEST CONDITIONS (highest mean R² per group)")
    print("=" * 110)

    for grp in GROUPS_OF_INTEREST:
        best_key = None
        best_mean = -999.0
        for key, vals in agg.items():
            cfg, model, layer, g = key
            if g != grp:
                continue
            m = np.mean(vals)
            if m > best_mean:
                best_mean = m
                best_key = key

        if best_key:
            cfg, model, layer, _ = best_key
            vals = agg[best_key]
            s = np.std(vals)
            print(
                f"  {grp:20s}: R² = {best_mean:.4f} ± {s:.4f}  "
                f"({cfg} | {model} | {layer}, N={len(vals)})"
            )


# ==============================================================================
# Ridge vs XGBoost comparison
# ==============================================================================
def print_model_comparison(agg):
    """Compare Ridge vs XGBoost for same (config, layer, group)."""
    print("\n" + "=" * 110)
    print("  RIDGE vs XGBOOST COMPARISON (ΔR² = XGBoost − Ridge)")
    print("=" * 110)

    for cfg_label, _, seeds, desc in CONFIGS:
        any_printed = False
        for layer in LAYERS:
            header_printed = False
            for grp in GROUPS_OF_INTEREST:
                ridge_key = (cfg_label, "Ridge", layer, grp)
                xgb_key = (cfg_label, "XGBoost", layer, grp)
                ridge_vals = agg.get(ridge_key, [])
                xgb_vals = agg.get(xgb_key, [])

                if not ridge_vals or not xgb_vals:
                    continue

                if not header_printed:
                    print(f"\n  ── {desc} | {layer} ──")
                    print(
                        f"  {'Group':20s} {'Ridge':>12s} {'XGBoost':>12s} {'ΔR²':>10s}"
                    )
                    print("  " + "-" * 58)
                    header_printed = True
                    any_printed = True

                r_m = np.mean(ridge_vals)
                x_m = np.mean(xgb_vals)
                delta = x_m - r_m
                print(f"  {grp:20s} {r_m:>12.4f} {x_m:>12.4f} {delta:>+10.4f}")

        if not any_printed:
            continue


# ==============================================================================
# Plots: Paired comparison
# ==============================================================================
def plot_paired_comparison(per_seed, results_dir):
    """Paired dot plot: L2 ON vs OFF for each CNN seed."""
    config_pairs = [
        ("MoCo_l2norm", "MoCo_raw", "MoCo"),
        ("noNorm_l2norm", "noNorm_raw", "noNorm"),
    ]

    for cfg_on, cfg_off, pair_label in config_pairs:
        for model in MODEL_TAGS:
            for layer in LAYERS:
                # Build {(seed): r2_mean} for ON and OFF
                on_dict = {}
                off_dict = {}
                for r in per_seed:
                    if r["model"] != model or r["layer"] != layer:
                        continue
                    if r["config"] == cfg_on:
                        on_dict.setdefault(r["group"], {})[r["seed"]] = r["r2_mean"]
                    elif r["config"] == cfg_off:
                        off_dict.setdefault(r["group"], {})[r["seed"]] = r["r2_mean"]

                groups_present = [
                    g for g in GROUPS_OF_INTEREST if g in on_dict and g in off_dict
                ]
                if not groups_present:
                    continue

                fig, axes = plt.subplots(
                    1, len(groups_present), figsize=(5 * len(groups_present), 5)
                )
                if len(groups_present) == 1:
                    axes = [axes]

                for ax, grp in zip(axes, groups_present):
                    on_g = on_dict[grp]
                    off_g = off_dict[grp]
                    common_seeds = sorted(set(on_g.keys()) & set(off_g.keys()))

                    for seed in common_seeds:
                        ax.plot(
                            [0, 1],
                            [off_g[seed], on_g[seed]],
                            "o-",
                            color="#555555",
                            alpha=0.5,
                            markersize=6,
                        )

                    if common_seeds:
                        mean_off = np.mean([off_g[s] for s in common_seeds])
                        mean_on = np.mean([on_g[s] for s in common_seeds])
                        ax.plot(
                            [0, 1],
                            [mean_off, mean_on],
                            "s-",
                            color="#E24A33",
                            markersize=10,
                            linewidth=2.5,
                            label="Mean",
                            zorder=5,
                        )

                    ax.set_xticks([0, 1])
                    ax.set_xticklabels(["L2 OFF", "L2 ON"], fontsize=11)
                    ax.set_ylabel("R² (mean across folds)", fontsize=11)
                    ax.set_title(
                        f"{grp}\n{pair_label} | {model} | {layer}",
                        fontsize=12,
                        fontweight="bold",
                    )
                    ax.grid(True, alpha=0.2, axis="y")
                    ax.legend(fontsize=9)

                fig.suptitle(
                    f"CNN GAP: {pair_label} L2 Norm Effect (Paired by Seed)",
                    fontsize=14,
                    fontweight="bold",
                    y=1.02,
                )
                fig.tight_layout()
                safe = f"paired_{pair_label}_{model}_{layer}".replace(" ", "_")
                save_path = os.path.join(results_dir, f"{safe}.png")
                fig.savefig(save_path, dpi=200, bbox_inches="tight")
                plt.close(fig)
                print(f"  Saved paired plot: {save_path}")


# ==============================================================================
# Plots: Forest plot (mean ± 95% CI)
# ==============================================================================
def plot_forest_ci(ci_results, results_dir):
    """Forest plot: L2 ON vs OFF with 95% CI."""
    config_pairs = [
        ("MoCo_l2norm", "MoCo_raw", "MoCo"),
        ("noNorm_l2norm", "noNorm_raw", "noNorm"),
    ]

    for cfg_on, cfg_off, pair_label in config_pairs:
        for model in MODEL_TAGS:
            for layer in LAYERS:
                on_results = [
                    r
                    for r in ci_results
                    if r["config"] == cfg_on
                    and r["model"] == model
                    and r["layer"] == layer
                    and r["group"] in GROUPS_OF_INTEREST
                ]
                off_results = [
                    r
                    for r in ci_results
                    if r["config"] == cfg_off
                    and r["model"] == model
                    and r["layer"] == layer
                    and r["group"] in GROUPS_OF_INTEREST
                ]
                if not on_results or not off_results:
                    continue

                fig, ax = plt.subplots(
                    figsize=(8, max(3, len(GROUPS_OF_INTEREST) * 1.2))
                )
                y_positions = []
                y_labels = []
                y_offset = 0

                for grp in GROUPS_OF_INTEREST:
                    for r_list, l2_label, color in [
                        (on_results, "L2 ON", "#E24A33"),
                        (off_results, "L2 OFF", "#348ABD"),
                    ]:
                        matches = [r for r in r_list if r["group"] == grp]
                        if not matches:
                            continue
                        r = matches[0]
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
                            label=l2_label if y_offset < 2 else None,
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
                        y_labels.append(f"{grp} ({l2_label})")
                        y_offset += 1
                    y_offset += 0.5

                ax.set_yticks(y_positions)
                ax.set_yticklabels(y_labels, fontsize=9)
                ax.set_xlabel("R²", fontsize=11)
                ax.set_title(
                    f"CNN GAP {pair_label} | {model} | {layer}\n95% CI",
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
                safe = f"forest_{pair_label}_{model}_{layer}".replace(" ", "_")
                save_path = os.path.join(results_dir, f"{safe}.png")
                fig.savefig(save_path, dpi=200, bbox_inches="tight")
                plt.close(fig)
                print(f"  Saved forest plot: {save_path}")


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Aggregate R² results across seeds — paper-ready summary"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Path to apoptosis_r2_results directory",
    )
    parser.add_argument(
        "--groups",
        type=str,
        nargs="*",
        default=None,
        help="Override groups of interest",
    )
    args = parser.parse_args()

    if args.groups:
        global GROUPS_OF_INTEREST
        GROUPS_OF_INTEREST = args.groups

    if not os.path.isdir(args.results_dir):
        print(f"ERROR: results_dir not found: {args.results_dir}")
        sys.exit(1)

    # ── Collect ──
    agg, per_seed, all_folds, missing = collect_all_results(args.results_dir)

    if not agg:
        print("ERROR: No results found. Check --results_dir path.")
        sys.exit(1)

    total_jsons = len(
        set((r["config"], r["seed"], r["model"], r["layer"]) for r in per_seed)
    )
    print(
        f"\n  Loaded {total_jsons} JSON result files, {len(all_folds)} fold-level entries."
    )
    if missing:
        print(f"  ⚠ {len(missing)} expected JSON files not found:")
        for m in missing[:10]:
            print(f"    ❌ {m}")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more")

    # ── Print ──
    print_aggregated_table(agg)
    print_per_seed_table(agg, per_seed)
    print_best_conditions(agg)
    print_model_comparison(agg)

    # ── Permutation summary ──
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

    # ── Save ──
    print("\n" + "=" * 110)
    print("  SAVING OUTPUT FILES")
    print("=" * 110)
    save_csvs(
        args.results_dir,
        agg,
        per_seed,
        all_folds,
        ci_results,
        wilcoxon_results,
        pooled_perm,
    )
    save_paper_table(args.results_dir, agg)

    # ── Plots ──
    print("\n" + "=" * 110)
    print("  GENERATING PLOTS")
    print("=" * 110)
    plot_paired_comparison(per_seed, args.results_dir)
    plot_forest_ci(ci_results, args.results_dir)

    print("\n" + "=" * 110)
    print("  DONE — All aggregated results saved.")
    print("=" * 110)


if __name__ == "__main__":
    main()
