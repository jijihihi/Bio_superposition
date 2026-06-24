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
        # SAE (loop over all configs for this seed)
        SAE_DIR="${DRIVE_BASE}/caches_per_image_centering/CNN_seed${SEED}_SAE"
        if [ -d "$SAE_DIR" ]; then
            for SAE_CACHE in "${SAE_DIR}"/sae_gap_d*_lam*_normrestored_withnewclass.npz; do
                if [ -f "$SAE_CACHE" ]; then
                    FILENAME=$(basename "$SAE_CACHE")
                    CONFIG=$(echo "$FILENAME" | grep -o 'd[0-9]\+_lam[0-9]\+')
                    SAE_OUT="${BASE_OUT}/${cond_name}/SAE_${CONFIG}/${SEED}"

                    echo "── SAE ${cond_name^^} ($CONFIG): seed=$SEED ──"
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
                fi
            done
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
