#!/bin/bash
# ==============================================================================
# Feature Extraction (Colab)
# StrictPlateBalancedBatchSampler + batch_size=64 로 SAE 학습/평가와
# 동일한 centering 일관성 보장
# ==============================================================================

set -e

SHARD="/content/wds_shards"
BASE="/content/drive/MyDrive/Final_paper/lambda_labs_moco_only"
#87
CNN_SEED=45
SEED_DIR="${BASE}/MoCo_seed${CNN_SEED}"
MODEL="${SEED_DIR}/best_model.pt"

LAYER="stage5_out"
SAE_FILENAME="${LAYER}_d4096_gated_sp3200.0_aux0.03125_tied_ep008.pt" 
# SAE sub-directories to process
SAE_DIRS=(
    "SAE_sparsity3200_loss_L2norm곱해줌"
)

# Output directory — batch64로 학습과 동일하게 centering
OUT_BASE="${SEED_DIR}/SAE_sparsity3200_loss_L2norm곱해줌/batch64_consistent"

TOTAL=$(( ${#SAE_DIRS[@]} * 2 ))  # 2 norm options per SAE dir
COUNT=0

for SAE_SUBDIR in "${SAE_DIRS[@]}"; do
    SAE_CKPT="${SEED_DIR}/${SAE_SUBDIR}/${SAE_FILENAME}"

    # Check SAE checkpoint exists
    if [ ! -f "$SAE_CKPT" ]; then
        echo "⚠️  SAE not found, skipping: $SAE_CKPT"
        continue
    fi

    for NORM_FLAG in "" "--restore_token_norm"; do
        COUNT=$((COUNT + 1))

        if [ -z "$NORM_FLAG" ]; then
            NORM_LABEL="raw"
        else
            NORM_LABEL="normrestored"
        fi

        OUTPUT="${OUT_BASE}/features_cache_${LAYER}_${NORM_LABEL}_batch64_StrictPlateBalancedBatchSamplerOnBank.npz"

        echo ""
        echo "=================================================================="
        echo "[$COUNT/$TOTAL] ${SAE_SUBDIR} layer=${LAYER} norm=${NORM_LABEL}"
        echo "=================================================================="
        echo "  SAE:    $SAE_CKPT"
        echo "  Output: $OUTPUT"
        echo "  batch_size=64 (same as SAE training)"

        python -m kendall_correlation_coefficient.extract_features \
            --sae_ckpt "$SAE_CKPT" \
            --save_dir "${SEED_DIR}" \
            --model_state_path "$MODEL" \
            --shard_root "$SHARD" \
            --which_layer "$LAYER" \
            --use_all_data \
            --batch_size 64 \
            --output_path "$OUTPUT" \
            --num_workers 0 \
            $NORM_FLAG

        echo "✅ Done: ${SAE_SUBDIR} layer=${LAYER} norm=${NORM_LABEL}"
    done
done

echo ""
echo "=================================================================="
echo "All $COUNT extractions complete!"
echo "=================================================================="
