# ==============================================================================
# Aggregate SAE R² results — SAE seed × GAP L2 × Filter mode summary
#
# Scans JSON result files from apoptosis_r2_test.py for SAE vectors.
# Directory pattern: SAE_seed{N}_{l2norm|no_l2norm}_{filter_label}/
#
# Usage:
#   python -m kendall_correlation_coefficient.aggregate_sae_r2_results \
#       --results_dir /path/to/apoptosis_r2_results/SAE_vector
# ==============================================================================

import os
import sys
import json
import re
import argparse
import csv
from collections import defaultdict

import numpy as np

GROUPS_OF_INTEREST = ["SNCA only", "GBA only", "LRRK2 only"]
MODEL_TAGS = ["Ridge", "XGBoost"]


def collect_results(results_dir: str):
    """
    Scan results_dir for SAE_seed*/ directories.
    Parses: SAE_seed{N}_{l2label}_{filter_label}
    Returns agg, per_seed.
    """
    agg = defaultdict(list)   # key = (l2_label, filter_label, model, group)
    per_seed = []

    dir_pattern = re.compile(
        r"SAE_seed(\d+)_(l2norm|no_l2norm)(?:_(.+))?$"
    )

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

            for res in data.get("results", []):
                grp = res["group"]
                r2_mean = res["r2_mean"]
                r2_std = res.get("r2_std", 0.0)

                key = (l2_label, filter_label, model_tag, grp)
                agg[key].append(r2_mean)

                per_seed.append({
                    "sae_seed": sae_seed,
                    "l2_norm": l2_label,
                    "filter": filter_label,
                    "model": model_tag,
                    "group": grp,
                    "r2_mean": r2_mean,
                    "r2_std": r2_std,
                })

    return dict(agg), per_seed


def print_summary(agg, per_seed):
    """Print summary tables."""
    print("\n" + "=" * 100)
    print("  SAE VECTOR APOPTOSIS R² SUMMARY (mean ± std across SAE seeds)")
    print("=" * 100)

    # Get unique conditions
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
                print(f"\n  ── L2={l2_label} | filter={filter_label} | {model_tag} ({n_seeds} seeds) ──")
                print(f"  {'Group':20s} {'Mean R²':>10s} {'±Std':>10s} {'Min':>10s} {'Max':>10s}")
                print("  " + "-" * 60)

                for grp in GROUPS_OF_INTEREST:
                    key = (l2_label, filter_label, model_tag, grp)
                    vals = agg.get(key, [])
                    if vals:
                        m = np.mean(vals)
                        s = np.std(vals)
                        print(f"  {grp:20s} {m:>10.4f} {s:>10.4f} "
                              f"{np.min(vals):>10.4f} {np.max(vals):>10.4f}")

    # ── Per-seed detail ──
    print("\n\n" + "=" * 100)
    print("  PER SAE-SEED R²")
    print("=" * 100)

    for l2_label in l2_labels:
        for filter_label in filter_labels:
            for model_tag in MODEL_TAGS:
                vals_exist = any(
                    agg.get((l2_label, filter_label, model_tag, g))
                    for g in GROUPS_OF_INTEREST
                )
                if not vals_exist:
                    continue

                sae_seeds = sorted(set(
                    r["sae_seed"] for r in per_seed
                    if r["l2_norm"] == l2_label
                    and r["filter"] == filter_label
                    and r["model"] == model_tag
                ))

                print(f"\n  ── L2={l2_label} | filter={filter_label} | {model_tag} ──")
                header = f"  {'SAE Seed':>10s}"
                for grp in GROUPS_OF_INTEREST:
                    header += f"  {grp:>14s}"
                print(header)
                print("  " + "-" * (10 + 16 * len(GROUPS_OF_INTEREST)))

                for seed in sae_seeds:
                    row = f"  {seed:>10d}"
                    for grp in GROUPS_OF_INTEREST:
                        matches = [
                            r for r in per_seed
                            if r["sae_seed"] == seed
                            and r["l2_norm"] == l2_label
                            and r["filter"] == filter_label
                            and r["model"] == model_tag
                            and r["group"] == grp
                        ]
                        if matches:
                            row += f"  {matches[0]['r2_mean']:>14.4f}"
                        else:
                            row += f"  {'N/A':>14s}"
                    print(row)

                # Mean row
                row_mean = f"  {'Mean':>10s}"
                for grp in GROUPS_OF_INTEREST:
                    key = (l2_label, filter_label, model_tag, grp)
                    vals = agg.get(key, [])
                    if vals:
                        row_mean += f"  {np.mean(vals):>14.4f}"
                    else:
                        row_mean += f"  {'N/A':>14s}"
                print("  " + "-" * (10 + 16 * len(GROUPS_OF_INTEREST)))
                print(row_mean)


def save_csvs(results_dir, agg, per_seed):
    """Save CSV files."""
    # Per-seed CSV
    csv_path = os.path.join(results_dir, "sae_r2_per_seed.csv")
    rows_sorted = sorted(
        per_seed,
        key=lambda x: (x["l2_norm"], x["filter"], x["model"], x["sae_seed"])
    )
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["SAE_Seed", "GAP_L2_Norm", "Filter", "Model",
                     "Group", "R2_mean", "R2_std"])
        for r in rows_sorted:
            w.writerow([
                r["sae_seed"], r["l2_norm"], r["filter"], r["model"],
                r["group"], f"{r['r2_mean']:.6f}", f"{r['r2_std']:.6f}",
            ])
    print(f"\n  Saved per-seed CSV: {csv_path}")

    # Summary CSV
    summary_path = os.path.join(results_dir, "sae_r2_summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["GAP_L2_Norm", "Filter", "Model", "Group",
                     "Mean_R2", "Std_R2", "N_seeds"])
        for key in sorted(agg.keys()):
            l2_label, filter_label, model_tag, grp = key
            vals = agg[key]
            w.writerow([
                l2_label, filter_label, model_tag, grp,
                f"{np.mean(vals):.6f}", f"{np.std(vals):.6f}", len(vals),
            ])
    print(f"  Saved summary CSV:  {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate SAE vector apoptosis R² results"
    )
    parser.add_argument(
        "--results_dir", type=str, required=True,
        help="Path to SAE_vector results directory"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        print(f"ERROR: not found: {args.results_dir}")
        sys.exit(1)

    agg, per_seed = collect_results(args.results_dir)
    if not agg:
        print("ERROR: No results found.")
        sys.exit(1)

    print(f"\n  Found {len(per_seed)} result entries.")
    print_summary(agg, per_seed)
    save_csvs(args.results_dir, agg, per_seed)

    print("\n" + "=" * 100)
    print("  DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()
