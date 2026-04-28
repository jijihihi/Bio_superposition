#!/bin/bash
# ==============================================================================
# Run Layer-wise A&U + Linear Classification across all CNN seeds & layers
#
# Seeds × Layers × L2norm ON/OFF
#   8 seeds × 3 layers × 1 (L2 norm ON) = 24 runs
#
# Output:
#   {OUT_ROOT}/layer_au_results.csv  ← all results in one CSV
#   {OUT_ROOT}/au_result_{layer}_seed{seed}.json  ← per-run JSON
#
# Usage:
#   bash /content/model_test/run_layer_au_eval.sh
# ==============================================================================

BASE_DIR="/content/drive/MyDrive/Final_paper/lambda_labs_moco_only"

# ── Seeds ──
CNN_SEEDS=(42 87 95 123 124 256 445 457)

# ── Layers to evaluate ──
LAYERS=("stage5_mid" "stage5_out" "refine_out")

# ── Output ──
OUT_ROOT="${BASE_DIR}/layer_au_results"
mkdir -p "${OUT_ROOT}"

# Remove old CSV to start fresh
rm -f "${OUT_ROOT}/layer_au_results.csv"

echo "═══════════════════════════════════════════════════════"
echo "  Layer A&U + Linear Classification Evaluation"
echo "  Seeds:  ${CNN_SEEDS[*]}"
echo "  Layers: ${LAYERS[*]}"
echo "  Output: ${OUT_ROOT}"
echo "═══════════════════════════════════════════════════════"

# ==============================================================
# Main loop: seed × layer
# ==============================================================
for SEED in "${CNN_SEEDS[@]}"; do
    SPLIT_DIR="${BASE_DIR}/MoCo_seed${SEED}"

    for LAYER in "${LAYERS[@]}"; do
        CACHE="${BASE_DIR}/MoCo_seed${SEED}/CNN_GAP/cnn_gap_${LAYER}_all.npz"

        if [ ! -f "$CACHE" ]; then
            echo "⚠️  Not found: $CACHE"
            continue
        fi

        echo ""
        echo "▶ seed=${SEED} | ${LAYER} | L2norm=ON"

        python -m model_test.layer_au_eval \
            --features_cache "${CACHE}" \
            --split_dir "${SPLIT_DIR}" \
            --layer_name "${LAYER}" \
            --seed_name "${SEED}" \
            --gap_l2_norm \
            --output_dir "${OUT_ROOT}" \
            --lp_lr 0.1 \
            --lp_epochs 50 \
            --seed 42 \
            --quiet
    done
done


# ==============================================================
# Aggregation: print summary table from CSV
# ==============================================================
echo ""
echo "═══════════════════════════════════════════"
echo "  Aggregating results..."
echo "═══════════════════════════════════════════"

python3 << 'PYEOF'
import os
import csv
import numpy as np
from collections import defaultdict

csv_path = "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/layer_au_results/layer_au_results.csv"

if not os.path.exists(csv_path):
    print("❌ No results CSV found!")
    exit(1)

# Read
rows = []
with open(csv_path, "r") as f:
    for row in csv.DictReader(f):
        rows.append(row)

if not rows:
    print("❌ No data rows!")
    exit(1)

# Group by layer
layer_data = defaultdict(list)
for r in rows:
    layer_data[r["layer"]].append(r)

# Print summary
print("\n" + "="*90)
print(f"{'Layer':>15s}  {'N':>3s}  {'LinAcc':<20s}  {'Alignment':<20s}  {'Uniformity':<20s}")
print("="*90)

for layer in ["stage5_mid", "stage5_out", "refine_out"]:
    if layer not in layer_data:
        continue
    d = layer_data[layer]
    n = len(d)
    accs = [float(r["linear_acc"]) for r in d]
    aligns = [float(r["alignment"]) for r in d]
    uniforms = [float(r["uniformity"]) for r in d]

    acc_str = f"{np.mean(accs):.4f} ± {np.std(accs):.4f}"
    align_str = f"{np.mean(aligns):.6f} ± {np.std(aligns):.6f}"
    unif_str = f"{np.mean(uniforms):.4f} ± {np.std(uniforms):.4f}"

    print(f"{layer:>15s}  {n:>3d}  {acc_str:<20s}  {align_str:<20s}  {unif_str:<20s}")

# Best layer selection
print("\n── Layer Ranking (lower align+unif = better representation) ──")
for layer in ["stage5_mid", "stage5_out", "refine_out"]:
    if layer not in layer_data:
        continue
    d = layer_data[layer]
    acc = np.mean([float(r["linear_acc"]) for r in d])
    align = np.mean([float(r["alignment"]) for r in d])
    unif = np.mean([float(r["uniformity"]) for r in d])

    # Higher acc + lower align + lower unif = better
    quality = acc - 0.5 * align  # heuristic composite score
    print(f"  {layer:>15s}: acc={acc:.4f}, align={align:.4f}↓, "
          f"unif={unif:.4f}↓, quality={quality:.4f}")

# Save aggregated summary
agg_path = csv_path.replace("layer_au_results.csv", "layer_au_summary.csv")
with open(agg_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["layer", "n_seeds", "mean_acc", "std_acc",
                "mean_alignment", "std_alignment",
                "mean_uniformity", "std_uniformity"])
    for layer in ["stage5_mid", "stage5_out", "refine_out"]:
        if layer not in layer_data:
            continue
        d = layer_data[layer]
        accs = [float(r["linear_acc"]) for r in d]
        aligns = [float(r["alignment"]) for r in d]
        uniforms = [float(r["uniformity"]) for r in d]
        w.writerow([layer, len(d),
                    f"{np.mean(accs):.4f}", f"{np.std(accs):.4f}",
                    f"{np.mean(aligns):.6f}", f"{np.std(aligns):.6f}",
                    f"{np.mean(uniforms):.4f}", f"{np.std(uniforms):.4f}"])
print(f"\nSaved summary: {agg_path}")

print("\n" + "="*90)
print("DONE")
print("="*90)
PYEOF

echo ""
echo "═══════════════════════════════════════════"
echo "  ALL DONE"
echo "═══════════════════════════════════════════"
