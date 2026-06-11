#!/bin/bash
# ==============================================================================
# Run KNN Evaluation for all SAE Caches
# ==============================================================================
set -e

# Base directory for caches and model outputs
BASE_DIR="/content/drive/MyDrive/Final_paper/lambda_labs_moco_only"
CACHE_ROOT="${BASE_DIR}/caches"
OUTPUT_ROOT="${BASE_DIR}/knn_eval_results"

CNN_SEEDS=(42 87 95 123 124 256 445 457)

# SAE Configs: "d_sae lambda" 형식
SAE_CONFIGS=(
    "600 50"
    "1024 800"
    "2048 800"
    "4096 800"
    "8192 800"
    "4096 200"
    "4096 3200"
)

mkdir -p "$OUTPUT_ROOT"

for SEED in "${CNN_SEEDS[@]}"; do
    echo "=================================================================="
    echo "🚀 Processing CNN SEED: $SEED"
    echo "=================================================================="
    
    MODEL_DIR="${BASE_DIR}/MoCo_seed${SEED}"
    SAE_CACHE_DIR="${CACHE_ROOT}/CNN_seed${SEED}_SAE"
    
    for CONFIG in "${SAE_CONFIGS[@]}"; do
        read -r D_SAE LAMBDA <<< "$CONFIG"
        
        # 캐시 파일 경로 확인 (이전 버전 이름과 withnewclass 버전 이름 모두 체크)
        CACHE_PATH="${SAE_CACHE_DIR}/sae_gap_d${D_SAE}_lam${LAMBDA}_normrestored.npz"
        if [ ! -f "$CACHE_PATH" ]; then
            CACHE_PATH="${SAE_CACHE_DIR}/sae_gap_d${D_SAE}_lam${LAMBDA}_normrestored_withnewclass.npz"
        fi
        
        if [ ! -f "$CACHE_PATH" ]; then
            echo "⚠️  [SAE] Cache not found for dim=${D_SAE}, lam=${LAMBDA} at seed ${SEED}. Skipping..."
            continue
        fi
        
        OUT_DIR="${OUTPUT_ROOT}/seed${SEED}/d${D_SAE}_lam${LAMBDA}"
        mkdir -p "$OUT_DIR"
        
        echo "▶️  [SAE] Evaluating KNN: dim=${D_SAE} lam=${LAMBDA}"
        python -m model_test.knn_fewshot_eval \
            --cache_path "$CACHE_PATH" \
            --save_dir "$MODEL_DIR" \
            --eval_modes knn \
            --knn_k 1 5 10 20 \
            --output_dir "$OUT_DIR" \
            --dead_threshold 1e-5
            
    done
done

echo "=================================================================="
echo "✅ All SAE KNN evaluations complete! Results saved to $OUTPUT_ROOT"
