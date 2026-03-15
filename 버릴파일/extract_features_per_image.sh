#!/bin/bash
# ==============================================================================
# Feature Extraction (Colab)
# StrictPlateBalancedBatchSampler + batch_size=64 로 SAE 학습/평가와
# 동일한 centering 일관성 보장
# ==============================================================================

set -e

BASE="/content/drive/MyDrive/Final_paper/lambda_labs_moco_only"
SHARD="/content/wds_shards"

CNN_SEED=87
SEED_DIR="${BASE}/MoCo_seed${CNN_SEED}"
MODEL="${SEED_DIR}/best_model.pt"

LAYER="stage5_out"
SAE_FILENAME="${LAYER}_d8192_gated_sp800.0_aux0.03125_tied_ep008.pt"

# SAE sub-directories to process
SAE_DIRS=(
    "SAE_seed48_no_L2norm_loss"
    "SAE_seed123_no_L2norm_loss"
    "SAE_seed777_no_L2norm_loss"
    "SAE_seed856_no_L2norm_loss"
)

# Output directory — batch64로 학습과 동일하게 centering
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

        OUTPUT="${SEED_DIR}/${SAE_SUBDIR}/features_cache_${LAYER}_${NORM_LABEL}_all_no_SAE_GAP_L2_norm_batch64_StrictPlateBalancedBatchSamplerOnBank_mean_per_image_centering.npz"

        echo ""
        echo "=================================================================="
        echo "[$COUNT/$TOTAL] ${SAE_SUBDIR} layer=${LAYER} norm=${NORM_LABEL}"
        echo "=================================================================="
        echo "  SAE:    $SAE_CKPT"
        echo "  Output: $OUTPUT"

        python -m kendall_correlation_coefficient.extract_features_per_image \
            --sae_ckpt "$SAE_CKPT" \
            --save_dir "${SEED_DIR}" \
            --model_state_path "$MODEL" \
            --shard_root "$SHARD" \
            --which_layer "$LAYER" \
            --use_all_data \
            --output_path "$OUTPUT" \
            $NORM_FLAG

        echo "✅ Done: ${SAE_SUBDIR} layer=${LAYER} norm=${NORM_LABEL}"
    done
done

echo ""
echo "=================================================================="
echo "All $COUNT extractions complete!"
echo "=================================================================="
