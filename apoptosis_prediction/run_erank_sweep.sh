#!/bin/bash
# ===============================================================================
# Effective Rank Sweep — CNN (3 layers) vs SAE
#   Conditions:
#     1) Raw (no PCA, no norm)
#     2) PCA 250 (no norm)
#     3) PCA 250 + std norm
# ===============================================================================
set -e

DRIVE_BASE="/content/drive/MyDrive/Final_paper/lambda_labs_moco_only"
BASE_OUT="${DRIVE_BASE}/caches_per_image_centering/erank"

# Seeds to process
CNN_SEEDS="42 87 95 123 124 256 445 457"
# CNN layers to evaluate
CNN_LAYERS=("stage5_mid" "stage5_out" "refine_out")

DEAD_THRESHOLD="1e-5"
RANDOM_SEED=856

# Helper to get CNN cache path for a given seed and layer
get_cnn_cache() {
    local cnn_seed=$1
    local layer=$2
    echo "${DRIVE_BASE}/MoCo_seed${cnn_seed}/CNN_GAP/cnn_gap_${layer}_all.npz"
}

# SAE cache (same for all layers, only once per seed)
get_sae_cache() {
    local cnn_seed=$1
    echo "${DRIVE_BASE}/caches_per_image_centering/CNN_seed${cnn_seed}_SAE/sae_gap_d8192_lam800_normrestored_withnewclass.npz"
}

# ---------------------------------------------------------------------------
# Core function to run effective_rank for a specific condition
# ---------------------------------------------------------------------------
run_condition() {
    local cond_name=$1      # raw | pca250 | pca250_std
    local pca_dim=$2       # 0 or 250
    local norm_opt=$3      # "none" or "std"

    echo "=================================================="
    echo "  CONDITION: ${cond_name^^} (pca_dim=${pca_dim}, norm=${norm_opt})"
    echo "=================================================="

    for SEED in $CNN_SEEDS; do
        # SAE (run once per seed, stored under condition folder)
        SAE_CACHE=$(get_sae_cache $SEED)
        SAE_OUT="${BASE_OUT}/${cond_name}/SAE/${SEED}"
        if [ -f "$SAE_CACHE" ]; then
            echo "── SAE ${cond_name^^}: seed=$SEED ──"
            python -m apoptosis_prediction.effective_rank \
                --sae_cache "$SAE_CACHE" \
                --gap_l2_norm \
                --dead_threshold $DEAD_THRESHOLD \
                --filter_mode none \
                --pca_dim $pca_dim \
                --norm $norm_opt \
                --samples_per_class 5000 \
                --seed $RANDOM_SEED \
                --output_dir "$SAE_OUT"
        else
            echo "⚠️  Not found: $SAE_CACHE"
        fi

        # CNN layers
        for LAYER in "${CNN_LAYERS[@]}"; do
            CNN_CACHE=$(get_cnn_cache $SEED $LAYER)
            CNN_OUT="${BASE_OUT}/${cond_name}/${LAYER}/${SEED}"
            if [ -f "$CNN_CACHE" ]; then
                echo "── CNN ${cond_name^^} (${LAYER}): seed=$SEED ──"
                python -m apoptosis_prediction.effective_rank \
                    --cnn_cache "$CNN_CACHE" \
                    --gap_l2_norm \
                    --dead_threshold $DEAD_THRESHOLD \
                    --filter_mode none \
                    --pca_dim $pca_dim \
                    --norm $norm_opt \
                    --samples_per_class 5000 \
                    --seed $RANDOM_SEED \
                    --output_dir "$CNN_OUT"
            else
                echo "⚠️  Not found: $CNN_CACHE"
            fi
        done
    done
}

# ---------------------------------------------------------------------------
# Run all three conditions
# ---------------------------------------------------------------------------
run_condition raw 0 none
run_condition pca250 250 none
run_condition pca250_std 250 std

# ---------------------------------------------------------------------------
# Completion message
# ---------------------------------------------------------------------------
echo ""
echo "=================================================="
echo "  All effective rank experiments complete!"
echo "  Results stored under: $BASE_OUT"
echo "=================================================="
