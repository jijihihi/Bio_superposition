#!/bin/bash
# ==============================================================================
# Run Pairwise CKA Analysis on Pre-extracted SAE Caches
# ==============================================================================
set -e

# Base directory for caches and model outputs
BASE_DIR="/content/drive/MyDrive/Final_paper/lambda_labs_moco_only"
CACHE_ROOT="${BASE_DIR}/caches_CNN_SAE_class_27000_withnewclass"
OUTPUT_ROOT="${BASE_DIR}/cka_results"

# 비교할 모델들의 Seed (8개 모델)
CNN_SEEDS=(42 87 95 123 124 256 445 457)

# 평가할 SAE 설정 (d_sae lambda)
D_SAE=8192
LAMBDA=800

mkdir -p "$OUTPUT_ROOT"

echo "=================================================================="
echo "🚀 Gathering SAE Caches for CKA (dim=${D_SAE}, lam=${LAMBDA})"
echo "=================================================================="

CACHE_LIST=()
for SEED in "${CNN_SEEDS[@]}"; do
    SAE_CACHE_DIR="${CACHE_ROOT}/CNN_seed${SEED}_SAE"
    CACHE_PATH="${SAE_CACHE_DIR}/sae_gap_d${D_SAE}_lam${LAMBDA}_normrestored_withnewclass.npz"
    
    if [ ! -f "$CACHE_PATH" ]; then
        # fallback to non-withnewclass if it doesn't exist
        CACHE_PATH="${SAE_CACHE_DIR}/sae_gap_d${D_SAE}_lam${LAMBDA}_normrestored.npz"
    fi
    
    if [ -f "$CACHE_PATH" ]; then
        CACHE_LIST+=("$CACHE_PATH")
    else
        echo "⚠️  [SAE] Cache not found for seed ${SEED}. Skipping..."
    fi
done

if [ ${#CACHE_LIST[@]} -lt 2 ]; then
    echo "❌ Error: Need at least 2 caches to compute pairwise CKA."
    exit 1
fi

OUT_DIR="${OUTPUT_ROOT}/SAE_d${D_SAE}_lam${LAMBDA}"
mkdir -p "$OUT_DIR"

echo "▶️  Running CKA Analysis on ${#CACHE_LIST[@]} caches..."
python -m model_similarity_lambda_labs.cka_analysis\ copy \
    --caches "${CACHE_LIST[@]}" \
    --output_dir "$OUT_DIR" \
    --dead_threshold 1e-5 \
    --gap_l2_norm

echo "=================================================================="
echo "✅ CKA Analysis complete! Results saved to $OUT_DIR"
