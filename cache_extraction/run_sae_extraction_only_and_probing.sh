#!/bin/bash
# ==============================================================================
# SAE Only Feature Extraction & Linear Probing
# 
# - CNN 추출 생략
# - SAE 추출 시 학습에 사용한 데이터(CSV splits) 기반으로 클래스당 10000장 추출 (10000개 미만인 경우 전체 추출)
# - 배치 센터링 문제(Signal Collapse)가 해결된 추출 코드를 사용하여 학습 환경 완벽 재현
# - 추출 완료 후 한 번에 step18_eval_probe_from_cache.py 실행하여 linear classification 진행
# ==============================================================================

#결론부터 말씀드리면, L0와 FVU는 Dead Neuron Threshold(5e-4나 1e-5)의 영향을 아예 받지 않습니다.
#set -e


# ============================
# 설정 (Configuration)
# ============================
BASE="/home/ubuntu/model-east3/outputs"
SHARD="/home/ubuntu/model-east3/wds_shards_tar"
CACHE_ROOT="/home/ubuntu/model-east3/caches_sae_10k"

# CNN Seeds (총 8개)
CNN_SEEDS=(42 87 95 123 124 256 445 457)

# SAE 설정
SAE_SEED=48
SAE_LAYER="stage5_out"

# SAE Configs: "d_sae lambda" 형식 (람다 0은 제외됨)
SAE_CONFIGS=(
    "600 50"
    "1024 800"
    "2048 800"
    "4096 800"
    "8192 800"
    "4096 200"
    "4096 3200"
)

# ============================
# 1. 메인 루프 (SAE Extraction)
# ============================
mkdir -p "$CACHE_ROOT"

for SEED in "${CNN_SEEDS[@]}"; do
    echo ""
    echo "=================================================================="
    echo "🚀 Processing CNN SEED: $SEED"
    echo "=================================================================="
    
    MODEL_DIR="${BASE}/MoCo_seed${SEED}"
    MODEL="${MODEL_DIR}/best_model.pt"
    
    if [ ! -f "$MODEL" ]; then
        echo "⚠️  Model not found, skipping seed ${SEED}: $MODEL"
        continue
    fi
    
    # -----------------------------------------------------
    # SAE GAP Extraction (7개 설정)
    # -----------------------------------------------------
    SAE_CACHE_DIR="${CACHE_ROOT}/CNN_seed${SEED}_SAE"
    mkdir -p "$SAE_CACHE_DIR"
    
    for CONFIG in "${SAE_CONFIGS[@]}"; do
        read -r D_SAE LAMBDA <<< "$CONFIG"
        SAE_SAVE_DIR="${MODEL_DIR}/SAE_dim${D_SAE}_lambda${LAMBDA}_seed${SAE_SEED}_no_L2norm_loss"
        
        # 무조건 ep008.pt 체크포인트 파일 사용
        SAE_CKPT=$(find "${SAE_SAVE_DIR}" -maxdepth 1 -name "*_ep008.pt" 2>/dev/null | head -n 1)
        
        if [ -z "$SAE_CKPT" ] || [ ! -f "$SAE_CKPT" ]; then
            echo "⚠️  [SAE] Checkpoint not found for dim=${D_SAE}, lam=${LAMBDA}. Skipping..."
            continue
        fi
        
        OUTPUT_PATH="${SAE_CACHE_DIR}/sae_gap_d${D_SAE}_lam${LAMBDA}_normrestored.npz"
        
        if [ -f "$OUTPUT_PATH" ]; then
            echo "✅  [SAE] Cache already exists for dim=${D_SAE} lam=${LAMBDA}. Skipping..."
            continue
        fi
        
        echo "▶️  [SAE] Extracting dim=${D_SAE} lam=${LAMBDA} into ${OUTPUT_PATH}"
        
        # --ignore_splits 를 제거하고 --use_all_data 와 --samples_per_class 10000 을 추가하여 
        # 학습때 사용한 (Train/Val/Test CSV에 명시된) 데이터에서만 클래스당 최대 1만장 균등 추출
        python -m cache_extraction.extract_features_lambda_labs \
            --sae_ckpt "$SAE_CKPT" \
            --save_dir "$MODEL_DIR" \
            --model_state_path "$MODEL" \
            --shard_root "$SHARD" \
            --which_layer "$SAE_LAYER" \
            --use_all_data \
            --samples_per_class 10000 \
            --restore_token_norm \
            --output_path "$OUTPUT_PATH"
            
    done
done

echo ""
echo "=================================================================="
echo "✅ All SAE extractions complete! Caches saved to $CACHE_ROOT"
echo "=================================================================="

# ============================
# 2. Linear Classification
# ============================
echo "▶️  Running Linear Probe Evaluation on all extracted SAE caches..."

# step18 스크립트는 기본적으로 cache_dir 의 한 단계 위 부모 디렉토리에 save_csv 경로를 씁니다.
# 결과물 경로: /home/ubuntu/model-east3/L0_FVU_linear_classification/sae_linear_probe_10k_results.csv
OUTPUT_CSV_NAME="L0_FVU_linear_classification/sae_linear_probe_10k_results.csv"

# 폴더 미리 생성
mkdir -p "/home/ubuntu/model-east3/L0_FVU_linear_classification"

python -m sae_project.step18_eval_probe_from_cache \
    --cache_dir "$CACHE_ROOT" \
    --save_csv "$OUTPUT_CSV_NAME" \
    --model_base_dir "$BASE"

echo "=================================================================="
echo "✅ Linear Classification complete!"
echo "=================================================================="
