#!/bin/bash
# ==============================================================================
# Run cell_death R² test across all CNN seeds — Colab version
#
# MoCo (GAP L2 norm):       9 seeds × 3 layers × 2 models = 54 runs
# MoCo (no GAP L2 norm):    5 seeds × 3 layers × 2 models = 30 runs
# + both --gap_l2_norm ON/OFF for each
# + aggregation at the end
#
# Usage:
#   bash /content/cell_death_prediction/run_cell_death_r2_seeds.sh
# ==============================================================================

# 여기서는 뉴런 필터링 없다.

BASE_DIR="/content/drive/MyDrive/Final_paper/lambda_labs_moco_only"
cell_death_CSV="/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/세포이미지별 사멸율/이미지별_세포사멸율_7200.csv"

# ── MoCo (GAP L2 norm) seeds ──
MOCO_SEEDS=(42 87 95 123 124 256 445 457)


# ── MoCo (no GAP L2 norm) seeds ──
NO_NORM_SEEDS=(42 87 124)


LAYERS=("stage5_mid" "stage5_out" "refine_out")
MODELS=("ridge" "xgboost") #"ridge" "xgboost"

OUT_ROOT="${BASE_DIR}/cell_death_r2_results"
mkdir -p "${OUT_ROOT}"

echo "═══════════════════════════════════════════════════════"
echo "  cell_death R² Test"
echo "  MoCo seeds:    ${MOCO_SEEDS[*]}"
echo "  No-norm seeds: ${NO_NORM_SEEDS[*]}"
echo "  Layers: ${LAYERS[*]}"
echo "  Models: ${MODELS[*]}"
echo "═══════════════════════════════════════════════════════"

pip -q install xgboost 2>/dev/null

# ==============================================================
# 1) MoCo (GAP L2 norm) — with and without --gap_l2_norm
# ==============================================================
for MODEL in "${MODELS[@]}"; do
    for LAYER in "${LAYERS[@]}"; do
        for SEED in "${MOCO_SEEDS[@]}"; do
            CACHE="${BASE_DIR}/MoCo_seed${SEED}/CNN_GAP/cnn_gap_${LAYER}_all.npz"

            if [ ! -f "$CACHE" ]; then
                echo "⚠️  Not found: $CACHE"
                continue
            fi

            # ---- with L2 norm ----
            SEED_OUT="${OUT_ROOT}/MoCo_seed${SEED}_l2norm"
            mkdir -p "${SEED_OUT}"

            echo ""
            echo "▶ MoCo seed=${SEED} | ${LAYER} | ${MODEL} | L2norm=ON"

            python -m cell_death_prediction.cell_death_r2_test \
                --features_cache "${CACHE}" \
                --cell_death_csv "${cell_death_CSV}" \
                --model "${MODEL}" \
                --gap_l2_norm \
                --output_dir "${SEED_OUT}" \
                --seed 42 \
                --cv_folds 5 \
                --n_repeats 2 \
                --n_permutations 2 \
                --quiet

            # ---- without L2 norm ----
            SEED_OUT="${OUT_ROOT}/MoCo_seed${SEED}_raw"
            mkdir -p "${SEED_OUT}"

            echo ""
            echo "▶ MoCo seed=${SEED} | ${LAYER} | ${MODEL} | L2norm=OFF"

            python -m cell_death_prediction.cell_death_r2_test \
                --features_cache "${CACHE}" \
                --cell_death_csv "${cell_death_CSV}" \
                --model "${MODEL}" \
                --output_dir "${SEED_OUT}" \
                --seed 42 \
                --cv_folds 5 \
                --n_repeats 2 \
                --n_permutations 2 \
                --quiet
        done
    done
done

# ==============================================================
# 2) MoCo no GAP L2 norm models — with and without --gap_l2_norm
# ==============================================================
for MODEL in "${MODELS[@]}"; do
    for LAYER in "${LAYERS[@]}"; do
        for SEED in "${NO_NORM_SEEDS[@]}"; do
            CACHE="${BASE_DIR}/MoCo_seed${SEED}_no_GAPL2norm/CNN_GAP/cnn_gap_${LAYER}_all.npz"

            if [ ! -f "$CACHE" ]; then
                echo "⚠️  Not found: $CACHE"
                continue
            fi

            # ---- with L2 norm ----
            SEED_OUT="${OUT_ROOT}/noNorm_seed${SEED}_l2norm"
            mkdir -p "${SEED_OUT}"

            echo ""
            echo "▶ noNorm seed=${SEED} | ${LAYER} | ${MODEL} | L2norm=ON"

            python -m cell_death_prediction.cell_death_r2_test \
                --features_cache "${CACHE}" \
                --cell_death_csv "${cell_death_CSV}" \
                --model "${MODEL}" \
                --gap_l2_norm \
                --output_dir "${SEED_OUT}" \
                --seed 42 \
                --cv_folds 5 \
                --n_repeats 2 \
                --n_permutations 2 \
                --quiet

            # ---- without L2 norm ----
            SEED_OUT="${OUT_ROOT}/noNorm_seed${SEED}_raw"
            mkdir -p "${SEED_OUT}"

            echo ""
            echo "▶ noNorm seed=${SEED} | ${LAYER} | ${MODEL} | L2norm=OFF"

            python -m cell_death_prediction.cell_death_r2_test \
                --features_cache "${CACHE}" \
                --cell_death_csv "${cell_death_CSV}" \
                --model "${MODEL}" \
                --output_dir "${SEED_OUT}" \
                --seed 42 \
                --cv_folds 5 \
                --n_repeats 2 \
                --n_permutations 2 \
                --quiet
        done
    done
done


# ==============================================================
# 3) Aggregate all results
# ==============================================================
echo ""
echo "═══════════════════════════════════════════"
echo "  Aggregating results..."
echo "═══════════════════════════════════════════"

python3 << 'PYEOF'
import os, json
import numpy as np
import csv

base = "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/cell_death_r2_results"

configs = [
    # (label, dir_pattern, seeds)
    ("MoCo_l2norm",   "MoCo_seed{}_l2norm",   [42,87,95,123,124,256,445,457]),
    ("MoCo_raw",      "MoCo_seed{}_raw",      [42,87,95,123,124,256,445,457]),
    ("noNorm_l2norm",  "noNorm_seed{}_l2norm", [42,87,124]),
    ("noNorm_raw",     "noNorm_seed{}_raw",    [42,87,124]),
]

layers = ["stage5_mid", "stage5_out", "refine_out"]
models = ["Ridge", "XGBoost"]
groups_of_interest = ["SNCA only", "GBA only", "LRRK2 only"]

# Collect all results
all_data = {}    # key: (config_label, model, layer, group) → list of r2
per_seed_rows = []

for cfg_label, dir_pat, seeds in configs:
    for seed in seeds:
        seed_dir = os.path.join(base, dir_pat.format(seed))
        for layer in layers:
            for model in models:
                json_path = os.path.join(seed_dir, f"r2_results_{layer}_{model}.json")
                if not os.path.exists(json_path):
                    continue
                with open(json_path) as f:
                    data = json.load(f)
                for res in data["results"]:
                    grp = res["group"]
                    key = (cfg_label, model, layer, grp)
                    if key not in all_data:
                        all_data[key] = []
                    all_data[key].append(res["r2_mean"])

                    per_seed_rows.append({
                        "config": cfg_label, "seed": seed,
                        "model": model, "layer": layer,
                        "group": grp, "r2_mean": res["r2_mean"],
                        "r2_std": res["r2_std"],
                    })

# ── Print aggregated table ──
print("\n" + "="*100)
print("AGGREGATED R² (mean ± std across seeds)")
print("="*100)

for cfg_label, _, seeds in configs:
    for model in models:
        for layer in layers:
            vals_exist = any(all_data.get((cfg_label, model, layer, g)) for g in groups_of_interest)
            if not vals_exist:
                continue
            print(f"\n── {cfg_label} | {model} | {layer} ({len(seeds)} seeds) ──")
            header = f"  {'Group':20s} {'Mean R²':>10s} {'±Std':>10s} {'N':>5s}"
            print(header)
            print("  " + "-"*48)
            for grp in groups_of_interest:
                key = (cfg_label, model, layer, grp)
                vals = all_data.get(key, [])
                if vals:
                    m, s = np.mean(vals), np.std(vals)
                    print(f"  {grp:20s} {m:>10.4f} {s:>10.4f} {len(vals):>5d}")

# ── Per-seed table for best seed selection ──
print("\n\n" + "="*100)
print("PER-SEED R² (for best seed selection)")
print("="*100)

for cfg_label, _, seeds in configs:
    for model in models:
        for layer in layers:
            vals_exist = any(all_data.get((cfg_label, model, layer, g)) for g in groups_of_interest)
            if not vals_exist:
                continue
            print(f"\n── {cfg_label} | {model} | {layer} ──")
            header = f"  {'Seed':>6s}"
            for grp in groups_of_interest:
                header += f"  {grp:>12s}"
            print(header)
            print("  " + "-"*80)
            for seed in seeds:
                row = f"  {seed:>6d}"
                for grp in groups_of_interest:
                    matches = [r for r in per_seed_rows
                              if r["config"]==cfg_label and r["seed"]==seed
                              and r["model"]==model and r["layer"]==layer
                              and r["group"]==grp]
                    if matches:
                        row += f"  {matches[0]['r2_mean']:>12.4f}"
                    else:
                        row += f"  {'N/A':>12s}"
                print(row)

# ── Save per-seed CSV ──
csv_path = os.path.join(base, "aggregated_r2.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Config", "Seed", "Model", "Layer", "Group", "R2_mean", "R2_std"])
    for r in sorted(per_seed_rows, key=lambda x: (x["config"], x["seed"], x["model"], x["layer"])):
        w.writerow([r["config"], r["seed"], r["model"], r["layer"], r["group"],
                     f"{r['r2_mean']:.4f}", f"{r['r2_std']:.4f}"])
print(f"\nSaved: {csv_path}")

# ── Save aggregated summary CSV ──
agg_csv = os.path.join(base, "aggregated_r2_summary.csv")
with open(agg_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Config", "Model", "Layer", "Group", "Mean_R2", "Std_R2", "N_seeds"])
    for cfg_label, _, _ in configs:
        for model in models:
            for layer in layers:
                for grp in groups_of_interest:
                    key = (cfg_label, model, layer, grp)
                    vals = all_data.get(key, [])
                    if vals:
                        w.writerow([cfg_label, model, layer, grp,
                                   f"{np.mean(vals):.4f}", f"{np.std(vals):.4f}", len(vals)])
print(f"Saved: {agg_csv}")

print("\n" + "="*100)
print("DONE")
print("="*100)
PYEOF

echo ""
echo "═══════════════════════════════════════════"
echo "  ALL DONE"
echo "═══════════════════════════════════════════"
