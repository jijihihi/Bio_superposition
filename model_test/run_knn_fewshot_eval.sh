#!/bin/bash
# ==============================================================================
# KNN & Few-Shot Evaluation — CNN 3 layers × 8 seeds + SAE 4 seeds
#
# Evaluates:
#   CNN: 8 seeds × 3 layers (stage5_mid, stage5_out, refine_out) = 24 runs
#   SAE: 4 seeds (CNN seed87 기반, sp800) = 4 runs
#   Total: 28 runs
#
# Metrics:
#   1. Weighted KNN (1/d², k = 1,3,5,10,15)
#   2. Few-shot prototypical (1,3,5,10,15-shot, 1000 episodes)
#
# Usage:
#   bash model_test/run_knn_fewshot_eval.sh
# ==============================================================================
set -e

BASE="/content/drive/MyDrive/Final_paper/lambda_labs_moco_only"
OUT_BASE="${BASE}/knn_fewshot_eval"
mkdir -p "$OUT_BASE"

# ── Common settings ──
KNN_K="1 3 5 10 15"
N_SHOTS="1 3 5 10 15"
N_EPISODES=1000
SEED=42

CNN_SEEDS=(42 87 95 123 124 256 445 457)
CNN_LAYERS=(stage5_mid stage5_out refine_out)
SAE_SEEDS=(48 123 777 856)

ENCODER_SEED=87
ENCODER_SAVE_DIR="${BASE}/MoCo_seed${ENCODER_SEED}"

SAE_CACHE_TEMPLATE="${BASE}/MoCo_seed${ENCODER_SEED}/SAE_seed{SAE_SEED}_no_L2norm_loss/features_cache_stage5_out_normrestored_all_no_SAE_GAP_L2_norm_again_d8192_sp800.npz"

TOTAL=0
FAIL=0

# ==============================================================================
# 1. CNN GAP — 8 seeds × 3 layers = 24 runs
# ==============================================================================
echo "══════════════════════════════════════════════════════════"
echo "  CNN GAP Evaluation (3 layers × 8 seeds, L2 norm)"
echo "══════════════════════════════════════════════════════════"

for LAYER in "${CNN_LAYERS[@]}"; do
    echo ""
    echo "──── Layer: ${LAYER} ────"

    for S in "${CNN_SEEDS[@]}"; do
        CACHE="${BASE}/MoCo_seed${S}/CNN_GAP/cnn_gap_${LAYER}_all.npz"
        SAVE_DIR="${BASE}/MoCo_seed${S}"
        OUT_DIR="${OUT_BASE}/CNN_${LAYER}_seed${S}"

        if [ ! -f "$CACHE" ]; then
            echo "⚠️  CNN cache not found: $CACHE"
            continue
        fi

        if [ ! -f "${SAVE_DIR}/test_split.csv" ]; then
            echo "⚠️  Split CSV not found: ${SAVE_DIR}/test_split.csv"
            continue
        fi

        echo ""
        echo "── CNN ${LAYER} seed=${S} ──"
        TOTAL=$((TOTAL + 1))

        python -m model_test.knn_fewshot_eval \
            --cache_path "$CACHE" \
            --save_dir "$SAVE_DIR" \
            --gap_l2_norm \
            --dead_threshold 1e-5 \
            --eval_modes knn fewshot \
            --knn_k $KNN_K \
            --knn_weights inv_sq \
            --n_shots $N_SHOTS \
            --n_episodes $N_EPISODES \
            --output_dir "$OUT_DIR" \
            --seed $SEED \
        || { echo "❌ Failed: CNN ${LAYER} seed=${S}"; FAIL=$((FAIL + 1)); }

        echo "✅ CNN ${LAYER} seed=${S} done"
    done
done


# ==============================================================================
# 2. SAE — 4 SAE seeds (CNN seed87 기반, sp800)
# ==============================================================================
echo ""
echo "══════════════════════════════════════════════════════════"
echo "  SAE Evaluation (4 SAE seeds, CNN seed87)"
echo "══════════════════════════════════════════════════════════"

for S in "${SAE_SEEDS[@]}"; do
    CACHE="${SAE_CACHE_TEMPLATE//\{SAE_SEED\}/$S}"
    OUT_DIR="${OUT_BASE}/SAE_sp800_seed${S}"

    if [ ! -f "$CACHE" ]; then
        echo "⚠️  SAE cache not found: $CACHE"
        continue
    fi

    echo ""
    echo "── SAE sp800 seed=${S} ──"
    TOTAL=$((TOTAL + 1))

    python -m model_test.knn_fewshot_eval \
        --cache_path "$CACHE" \
        --save_dir "$ENCODER_SAVE_DIR" \
        --gap_l2_norm \
        --dead_threshold 5e-5 \
        --eval_modes knn fewshot \
        --knn_k $KNN_K \
        --knn_weights inv_sq \
        --n_shots $N_SHOTS \
        --n_episodes $N_EPISODES \
        --output_dir "$OUT_DIR" \
        --seed $SEED \
    || { echo "❌ Failed: SAE sp800 seed=${S}"; FAIL=$((FAIL + 1)); }

    echo "✅ SAE sp800 seed=${S} done"
done


# ==============================================================================
# Summary
# ==============================================================================
echo ""
echo "══════════════════════════════════════════════════════════"
echo "  ALL DONE"
echo "  Total: ${TOTAL}, Failed: ${FAIL}"
echo "  Results: ${OUT_BASE}/"
echo "══════════════════════════════════════════════════════════"
