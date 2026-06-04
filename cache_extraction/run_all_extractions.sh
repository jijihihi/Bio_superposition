#!/bin/bash
# ==============================================================================
# Batch Feature Extraction for SAE sub-directories
# MoCo_seed87 안의 SAE_seed*_no_L2norm_loss 디렉토리들에서
# stage5_out_d8192_gated_sp800.0_aux0.03125_tied_ep008.pt 기준으로
# normrestored + raw 2개씩 추출
# ==============================================================================



set -e

BASE="/home/ubuntu/model-east3/outputs"
SHARD="/home/ubuntu/model-east3/wds_shards_tar"

# BASE="/content/drive/MyDrive/Final_paper/lambda_labs_moco_only"
# SHARD="/content/wds_shards"

CNN_SEED=87
SEED_DIR="${BASE}/MoCo_seed${CNN_SEED}"
MODEL="${SEED_DIR}/best_model.pt"

LAYER="stage5_out"
SAE_FILENAME="${LAYER}_d8192_gated_sp800.0_aux0.03125_tied_ep008.pt"

# SAE sub-directories to process   #SAE seed 48 123 777 856



SAE_DIRS=(
    "SAE_seed48_no_L2norm_loss"
    "SAE_seed123_no_L2norm_loss"
    "SAE_seed777_no_L2norm_loss"
    "SAE_seed856_no_L2norm_loss"

)

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

        OUTPUT="${SEED_DIR}/${SAE_SUBDIR}/features_cache_${LAYER}_${NORM_LABEL}_all_no_SAE_GAP_L2_norm_again_d8192_sp800.npz"  ### 이렇게 했을때만 DPT가 양수로 유효하게 잘 나타남. 또한 control과 mutation 별 DPT가 후자가 더 크게 나타남. strict... 이거 하면 DPT 음수 값 나오고, control DPT mutation DPT 이게 거의 차이가 없다.

        echo ""
        echo "=================================================================="
        echo "[$COUNT/$TOTAL] ${SAE_SUBDIR} layer=${LAYER} norm=${NORM_LABEL}"
        echo "=================================================================="
        echo "  SAE:    $SAE_CKPT"
        echo "  Output: $OUTPUT"
        ## 람다랩스에서 kendall_correlation_coefficient.extract_features 이거는 로컬에서 extract_features_lambda_labs 이것과 동일하다.
        python -m cache_extraction.extract_features \ 
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




































 =====================================================
set -e

BASE="/home/ubuntu/model-east3/outputs"
SHARD="/home/ubuntu/model-east3/wds_shards_tar"

# BASE="/content/drive/MyDrive/Final_paper/lambda_labs_moco_only"
# SHARD="/content/wds_shards"

CNN_SEED=87
SEED_DIR="${BASE}/MoCo_seed${CNN_SEED}"
MODEL="${SEED_DIR}/best_model.pt"

LAYER="stage5_out"
SAE_FILENAME="${LAYER}_d8192_gated_sp800.0_aux0.03125_tied_ep008.pt"

# SAE sub-directories to process   #SAE seed 48 123 777 856



SAE_DIRS=(
    "SAE_seed48_no_L2norm_loss"
    "SAE_seed123_no_L2norm_loss"
    "SAE_seed777_no_L2norm_loss"
    "SAE_seed856_no_L2norm_loss"

)

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

        OUTPUT="${SEED_DIR}/${SAE_SUBDIR}/features_cache_${LAYER}_${NORM_LABEL}_all_no_SAE_GAP_L2_norm_StrictPlateBalancedBatchSamplerOnBank_d8192_sp800.npz"

        echo ""
        echo "=================================================================="
        echo "[$COUNT/$TOTAL] ${SAE_SUBDIR} layer=${LAYER} norm=${NORM_LABEL}"
        echo "=================================================================="
        echo "  SAE:    $SAE_CKPT"
        echo "  Output: $OUTPUT"

        python -m kendall_correlation_coefficient.extract_features_StrictPlateBalancedBatchSamplerOnBank \
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



