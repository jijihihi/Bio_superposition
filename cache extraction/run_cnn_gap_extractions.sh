#!/bin/bash
# ==============================================================================
# Batch CNN GAP Extraction: 5 seeds × N layers
#
# SAE 없이 CNN 자체의 GAP 벡터를 추출하여 .npz로 저장
# 저장 위치: {SEED_DIR}/CNN_GAP/cnn_gap_{layer}_all.npz
# ==============================================================================
set -e

BASE="/home/ubuntu/model-east3/outputs"
SHARD="/home/ubuntu/model-east3/wds_shards_tar"

SEEDS=(42 45 123 124 256)
LAYERS=("stage5_mid" "stage5_out" "refine_out")

TOTAL=$(( ${#SEEDS[@]} * ${#LAYERS[@]} ))
COUNT=0

for SEED in "${SEEDS[@]}"; do
    SEED_DIR="${BASE}/MoCo_seed${SEED}"
    MODEL="${SEED_DIR}/best_model.pt"

    if [ ! -f "$MODEL" ]; then
        echo "⚠️  Model not found, skipping seed ${SEED}: $MODEL"
        continue
    fi

    for LAYER in "${LAYERS[@]}"; do
        COUNT=$((COUNT + 1))

        echo ""
        echo "=================================================================="
        echo "[$COUNT/$TOTAL] seed=${SEED} layer=${LAYER}"
        echo "=================================================================="

        python -m kendall_correlation_coefficient.extract_cnn_gap \
            --save_dir "$SEED_DIR" \
            --model_state_path "$MODEL" \
            --shard_root "$SHARD" \
            --which_layer "$LAYER" \
            --use_all_data \
            --batch_size 64

        echo "✅ Done: seed=${SEED} layer=${LAYER}"
    done
done

echo ""
echo "=================================================================="
echo "All $COUNT CNN GAP extractions complete!"
echo "=================================================================="
