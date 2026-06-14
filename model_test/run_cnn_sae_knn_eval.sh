#!/bin/bash
# ==============================================================================
# Run KNN Evaluation for all CNN (3 layers) and SAE Caches
# ==============================================================================
set -e

# Base directory for caches and model outputs
BASE_DIR="/content/drive/MyDrive/Final_paper/lambda_labs_moco_only"
CACHE_ROOT="${BASE_DIR}/caches_dynamic_batch_centering"
OUTPUT_ROOT="${BASE_DIR}/knn_eval_results"

CNN_SEEDS=(42 87 95 123 124 256 445 457)
CNN_LAYERS=(stage5_mid stage5_out refine_out)

# SAE Configs: "d_sae lambda" 형식
SAE_CONFIGS=(
    # "600 50"
    # "1024 800"
    # "2048 800"
    # "4096 800"
    "8192 800"
    # "4096 200"
    # "4096 3200"
)

mkdir -p "$OUTPUT_ROOT"

for SEED in "${CNN_SEEDS[@]}"; do
    echo "=================================================================="
    echo "🚀 Processing CNN SEED: $SEED"
    echo "=================================================================="
    
    MODEL_DIR="${BASE_DIR}/MoCo_seed${SEED}"
    CNN_CACHE_DIR="${CACHE_ROOT}/CNN_seed${SEED}"
    SAE_CACHE_DIR="${CACHE_ROOT}/CNN_seed${SEED}_SAE"
    
    # -----------------------------------------------------
    # 1. CNN Evaluation
    # -----------------------------------------------------
    # for LAYER in "${CNN_LAYERS[@]}"; do
    #     CACHE_PATH="${CNN_CACHE_DIR}/cnn_gap_${LAYER}_withnewclass.npz"
        
    #     # 이전 버전 1: 캐시 루트 내부
    #     if [ ! -f "$CACHE_PATH" ]; then
    #         CACHE_PATH="${CNN_CACHE_DIR}/cnn_gap_${LAYER}_all.npz"
    #     fi
        
    #     # 이전 버전 2: 모델 폴더 내부 (구버전)
    #     if [ ! -f "$CACHE_PATH" ]; then
    #         CACHE_PATH="${MODEL_DIR}/CNN_GAP/cnn_gap_${LAYER}_all.npz"
    #     fi
        
    #     if [ ! -f "$CACHE_PATH" ]; then
    #         echo "⚠️  [CNN] Cache not found for layer=${LAYER} at seed ${SEED}. Skipping..."
    #         continue
    #     fi
        
    #     OUT_DIR="${OUTPUT_ROOT}/seed${SEED}/CNN_${LAYER}"
    #     mkdir -p "$OUT_DIR"
        
    #     echo "▶️  [CNN] Evaluating KNN: layer=${LAYER}"
    #     python -m model_test.knn_fewshot_eval \
    #         --cache_path "$CACHE_PATH" \
    #         --save_dir "$MODEL_DIR" \
    #         --gap_l2_norm \
    #         --dead_threshold 1e-10 \
    #         --eval_modes knn \
    #         --knn_k 1 3 5 10 15 \
    #         --knn_weights inv_sq \
    #         --output_dir "$OUT_DIR" \
    #         --seed 42
    # done

    # -----------------------------------------------------
    # 2. SAE Evaluation
    # -----------------------------------------------------
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
        
        OUT_DIR="${OUTPUT_ROOT}/seed${SEED}/SAE_d${D_SAE}_lam${LAMBDA}"
        mkdir -p "$OUT_DIR"
        
        echo "▶️  [SAE] Evaluating KNN: dim=${D_SAE} lam=${LAMBDA}"
        python -m model_test.knn_fewshot_eval \
            --cache_path "$CACHE_PATH" \
            --save_dir "$MODEL_DIR" \
            --dead_threshold 1e-5 \
            --gap_l2_norm \
            --eval_modes knn \
            --knn_k 1 3 5 10 15 \
            --knn_weights uniform \
            --output_dir "$OUT_DIR" \
            --seed 42
            
    done
done

echo "=================================================================="
echo "✅ All CNN & SAE KNN evaluations complete! Results saved to $OUTPUT_ROOT"
