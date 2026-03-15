# ==============================================================================
# Aggregate R² results across all seeds — Paper-ready summary
#
# Scans all JSON result files produced by apoptosis_r2_test.py,
# computes per-(config, model, layer, group) statistics:
#   - Mean R² ± Std across seeds
#   - Individual seed R²s
#   - LaTeX-ready table
#
# Outputs:
#   1. Console: aggregated tables
#   2. CSV:     aggregated_r2_summary.csv  (mean/std per condition)
#   3. CSV:     aggregated_r2_per_seed.csv (every seed's result)
#   4. CSV:     paper_table_r2.csv         (논문용 한 줄 요약)
#   5. LaTeX:   paper_table_r2.tex         (LaTeX table source)
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

import os
import sys
import json
import argparse
import csv
from collections import defaultdict

import numpy as np


# ==============================================================================
# Config: which directories / seeds to scan
# ==============================================================================
CONFIGS = [
    # (label, directory_pattern, seeds, description)
    ("MoCo_l2norm",  "MoCo_seed{}_l2norm",  [42, 87, 95, 123, 124, 256, 445, 457],
     "MoCo (GAP L2 norm training), L2 norm ON"),
    ("MoCo_raw",     "MoCo_seed{}_raw",     [42, 87, 95, 123, 124, 256, 445, 457],
     "MoCo (GAP L2 norm training), L2 norm OFF"),
    ("noNorm_l2norm", "noNorm_seed{}_l2norm", [42, 87, 124],
     "MoCo (no GAP L2 norm training), L2 norm ON"),
    ("noNorm_raw",    "noNorm_seed{}_raw",    [42, 87, 124],
     "MoCo (no GAP L2 norm training), L2 norm OFF"),
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
        Each dict has config, seed, model, layer, group, r2_mean, r2_std, r2_scores
    missing : list[str]
        List of expected but missing JSON paths
    """
    agg = defaultdict(list)
    per_seed = []
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
                        r2_std  = res.get("r2_std", 0.0)
                        r2_scores = res.get("r2_scores", [])

                        key = (cfg_label, model_tag, layer, grp)
                        agg[key].append(r2_mean)

                        per_seed.append({
                            "config": cfg_label,
                            "seed": seed,
                            "model": model_tag,
                            "layer": layer,
                            "group": grp,
                            "r2_mean": r2_mean,
                            "r2_std": r2_std,
                            "r2_scores": r2_scores,
                        })

    return dict(agg), per_seed, missing


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

                n_seeds = len([
                    v for g in GROUPS_OF_INTEREST
                    for v in agg.get((cfg_label, model_tag, layer, g), [])
                ]) // max(
                    sum(1 for g in GROUPS_OF_INTEREST
                        if agg.get((cfg_label, model_tag, layer, g))),
                    1
                )

                print(f"\n  ── {desc} | {model_tag} | {layer} ({n_seeds} seeds) ──")
                print(f"  {'Group':20s} {'Mean R²':>10s} {'±Std':>10s} {'Min':>10s} {'Max':>10s} {'N':>5s}")
                print("  " + "-" * 60)

                for grp in GROUPS_OF_INTEREST:
                    key = (cfg_label, model_tag, layer, grp)
                    vals = agg.get(key, [])
                    if vals:
                        m = np.mean(vals)
                        s = np.std(vals)
                        mn = np.min(vals)
                        mx = np.max(vals)
                        print(f"  {grp:20s} {m:>10.4f} {s:>10.4f} {mn:>10.4f} {mx:>10.4f} {len(vals):>5d}")


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
                            r for r in per_seed
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
                row_std  = f"  {'±Std':>6s}"
                for grp in GROUPS_OF_INTEREST:
                    key = (cfg_label, model_tag, layer, grp)
                    vals = agg.get(key, [])
                    if vals:
                        row_mean += f"  {np.mean(vals):>14.4f}"
                        row_std  += f"  {np.std(vals):>14.4f}"
                    else:
                        row_mean += f"  {'N/A':>14s}"
                        row_std  += f"  {'N/A':>14s}"
                print("  " + "-" * (6 + 16 * len(GROUPS_OF_INTEREST)))
                print(row_mean)
                print(row_std)


# ==============================================================================
# Save CSVs
# ==============================================================================
def save_csvs(results_dir, agg, per_seed):
    """Save per-seed CSV and aggregated summary CSV."""

    # 1. Per-seed CSV
    per_seed_csv = os.path.join(results_dir, "aggregated_r2_per_seed.csv")
    rows_sorted = sorted(
        per_seed,
        key=lambda x: (x["config"], x["model"], x["layer"], x["seed"])
    )
    with open(per_seed_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Config", "Seed", "Model", "Layer", "Group",
                     "R2_mean", "R2_std", "R2_fold_scores"])
        for r in rows_sorted:
            w.writerow([
                r["config"], r["seed"], r["model"], r["layer"], r["group"],
                f"{r['r2_mean']:.6f}",
                f"{r['r2_std']:.6f}",
                ";".join(f"{s:.6f}" for s in r["r2_scores"]) if r["r2_scores"] else ""
            ])
    print(f"\n  Saved per-seed CSV:  {per_seed_csv}")

    # 2. Aggregated summary CSV
    agg_csv = os.path.join(results_dir, "aggregated_r2_summary.csv")
    with open(agg_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Config", "Model", "Layer", "Group",
                     "Mean_R2", "Std_R2", "Min_R2", "Max_R2", "N_seeds"])
        for cfg_label, _, seeds, _ in CONFIGS:
            for model_tag in MODEL_TAGS:
                for layer in LAYERS:
                    for grp in GROUPS_OF_INTEREST:
                        key = (cfg_label, model_tag, layer, grp)
                        vals = agg.get(key, [])
                        if vals:
                            w.writerow([
                                cfg_label, model_tag, layer, grp,
                                f"{np.mean(vals):.6f}",
                                f"{np.std(vals):.6f}",
                                f"{np.min(vals):.6f}",
                                f"{np.max(vals):.6f}",
                                len(vals),
                            ])
    print(f"  Saved summary CSV:  {agg_csv}")


# ==============================================================================
# Paper table: one-line-per-condition, formatted as mean±std
# ==============================================================================
def save_paper_table(results_dir, agg):
    """
    논문용 테이블.
    Rows: (Config, Model, Layer)
    Columns: Groups (All, All Mutations, SNCA, GBA, LRRK2)
    Cell: mean ± std
    """
    paper_csv = os.path.join(results_dir, "paper_table_r2.csv")
    paper_tex = os.path.join(results_dir, "paper_table_r2.tex")

    # ── CSV ──
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

    # ── LaTeX ──
    n_cols = len(header)
    col_spec = "l" * 3 + "c" * len(GROUPS_OF_INTEREST)

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Cross-validated $R^2$ for apoptosis rate prediction (mean $\pm$ std across CNN training seeds).}")
    lines.append(r"\label{tab:r2_apoptosis}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Header
    h = " & ".join(["Config", "Model", "Layer"] +
                    [g.replace("_", r"\_") for g in GROUPS_OF_INTEREST])
    lines.append(h + r" \\")
    lines.append(r"\midrule")

    prev_config = None
    for row in csv_rows[1:]:
        cfg = row[0]
        if prev_config is not None and cfg != prev_config:
            lines.append(r"\midrule")
        prev_config = cfg

        # Escape underscores for LaTeX
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
            print(f"  {grp:20s}: R² = {best_mean:.4f} ± {s:.4f}  "
                  f"({cfg} | {model} | {layer}, N={len(vals)})")


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
                xgb_key   = (cfg_label, "XGBoost", layer, grp)
                ridge_vals = agg.get(ridge_key, [])
                xgb_vals   = agg.get(xgb_key, [])

                if not ridge_vals or not xgb_vals:
                    continue

                if not header_printed:
                    print(f"\n  ── {desc} | {layer} ──")
                    print(f"  {'Group':20s} {'Ridge':>12s} {'XGBoost':>12s} {'ΔR²':>10s}")
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
# Duplicate detection: find configs with identical R² for same seed
# ==============================================================================
def detect_duplicates(per_seed):
    """
    For each (seed, model, layer), compare R² across configs.
    Flag exact matches — likely means identical feature files.
    """
    print("\n" + "=" * 110)
    print("  DUPLICATE DETECTION (identical R² across configs for same seed)")
    print("=" * 110)

    # Build lookup: (seed, model, layer, group) → {config: r2_mean}
    lookup = defaultdict(dict)
    for r in per_seed:
        key = (r["seed"], r["model"], r["layer"], r["group"])
        lookup[key][r["config"]] = r["r2_mean"]

    # Compare all config pairs
    config_labels = [c[0] for c in CONFIGS]
    duplicates = []  # (seed, model, layer, config_a, config_b, n_matching_groups)

    checked_pairs = set()
    for key, cfg_vals in lookup.items():
        seed, model, layer, group = key
        configs_present = list(cfg_vals.keys())

        for i, ca in enumerate(configs_present):
            for cb in configs_present[i+1:]:
                pair_key = (seed, model, layer, ca, cb)
                if pair_key in checked_pairs:
                    continue
                checked_pairs.add(pair_key)

                # Count how many groups have identical R² for this pair
                matching_groups = []
                for g in GROUPS_OF_INTEREST:
                    gkey = (seed, model, layer, g)
                    gvals = lookup.get(gkey, {})
                    if ca in gvals and cb in gvals:
                        if gvals[ca] == gvals[cb]:
                            matching_groups.append(g)

                if len(matching_groups) >= 3:  # At least 3 groups match → suspicious
                    duplicates.append((seed, model, layer, ca, cb, matching_groups))

    if not duplicates:
        print("  ✅ No suspicious duplicates found.")
    else:
        print(f"  ⚠ Found {len(duplicates)} suspicious duplicate(s):\n")
        for seed, model, layer, ca, cb, groups in sorted(duplicates):
            print(f"    🔴 seed={seed} | {model} | {layer}")
            print(f"       {ca} == {cb}")
            print(f"       Matching groups ({len(groups)}): {', '.join(groups)}")

            # Show the actual R² values
            for g in groups:
                gkey = (seed, model, layer, g)
                val = lookup[gkey][ca]
                print(f"         {g}: R² = {val:.6f}")
            print()

        print("  💡 Likely cause: identical .npz feature files.")
        print("     Check if the CNN GAP extraction used the wrong checkpoint.")


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Aggregate R² results across seeds — paper-ready summary"
    )
    parser.add_argument(
        "--results_dir", type=str, required=True,
        help="Path to apoptosis_r2_results directory"
    )
    parser.add_argument(
        "--groups", type=str, nargs="*", default=None,
        help="Override groups of interest (default: All, All Mutations, SNCA, GBA, LRRK2)"
    )
    args = parser.parse_args()

    if args.groups:
        global GROUPS_OF_INTEREST
        GROUPS_OF_INTEREST = args.groups

    if not os.path.isdir(args.results_dir):
        print(f"ERROR: results_dir not found: {args.results_dir}")
        sys.exit(1)

    # ── Collect ──
    agg, per_seed, missing = collect_all_results(args.results_dir)

    if not agg:
        print("ERROR: No results found. Check --results_dir path.")
        sys.exit(1)

    # Count stats
    total_jsons = len(set(
        (r["config"], r["seed"], r["model"], r["layer"]) for r in per_seed
    ))
    print(f"\n  Loaded {total_jsons} JSON result files.")
    if missing:
        print(f"  ⚠ {len(missing)} expected JSON files not found (possibly not yet computed):")
        for m in missing:
            print(f"    ❌ {m}")

    # ── Print ──
    print_aggregated_table(agg)
    print_per_seed_table(agg, per_seed)
    print_best_conditions(agg)
    print_model_comparison(agg)
    detect_duplicates(per_seed)

    # ── Save ──
    print("\n" + "=" * 110)
    print("  SAVING OUTPUT FILES")
    print("=" * 110)
    save_csvs(args.results_dir, agg, per_seed)
    save_paper_table(args.results_dir, agg)

    print("\n" + "=" * 110)
    print("  DONE — All aggregated results saved.")
    print("=" * 110)


if __name__ == "__main__":
    main()
