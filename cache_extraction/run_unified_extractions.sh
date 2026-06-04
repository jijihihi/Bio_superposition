#!/bin/bash
# ==============================================================================
# Unified Batch Feature Extraction for CNN and SAE
# 
# 이 스크립트는 8개의 CNN SEED와 각각의 SAE 설정들에 대해 
# CNN 3개 레이어, SAE 7개 설정의 cache를 모두 추출하여 새로운 'caches' 폴더 구조에 저장합니다.
# ==============================================================================

set -e

# ============================
# 설정 (Configuration)
# ============================
BASE="/home/ubuntu/model-east3/outputs"
SHARD="/home/ubuntu/model-east3/wds_shards_tar"
CACHE_ROOT="/home/ubuntu/model-east3/caches"

# CNN Seeds (총 8개)
CNN_SEEDS=(42 87 95 123 124 256 445 457)
# CNN Layers (총 3개)
CNN_LAYERS=("stage5_mid" "stage5_out" "refine_out")

# SAE 설정
SAE_SEED=48
SAE_LAYER="stage5_out"

# SAE Configs: "d_sae lambda" 형식 (총 8개 조합)
SAE_CONFIGS=(
    "600 50"
    "1024 800"
    "2048 800"
    "4096 800"
    "8192 800"
    "4096 0"
    "4096 200"
    "4096 3200"
)


# ============================
# 메인 루프 (Extraction Loop)
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
    # 1. CNN GAP Extraction (3개 레이어)
    # -----------------------------------------------------
    CNN_CACHE_DIR="${CACHE_ROOT}/CNN_seed${SEED}"
    mkdir -p "$CNN_CACHE_DIR"
    
    for LAYER in "${CNN_LAYERS[@]}"; do
        echo "▶️  [CNN] Extracting layer=${LAYER} into ${CNN_CACHE_DIR}"
        python -m cache_extraction.extract_cnn_gap \
            --save_dir "$MODEL_DIR" \
            --model_state_path "$MODEL" \
            --shard_root "$SHARD" \
            --which_layer "$LAYER" \
            --ignore_splits \
            --output_dir "$CNN_CACHE_DIR" \
            --batch_size 64
    done
    
    # -----------------------------------------------------
    # 2. SAE GAP Extraction (7개 설정)
    # -----------------------------------------------------
    SAE_CACHE_DIR="${CACHE_ROOT}/CNN_seed${SEED}_SAE"
    mkdir -p "$SAE_CACHE_DIR"
    
    for CONFIG in "${SAE_CONFIGS[@]}"; do
        read -r D_SAE LAMBDA <<< "$CONFIG"
        SAE_SAVE_DIR="${MODEL_DIR}/SAE_dim${D_SAE}_lambda${LAMBDA}_seed${SAE_SEED}_no_L2norm_loss"
        
        # 무조건 ep008.pt 체크포인트 파일 사용
        SAE_CKPT=$(ls ${SAE_SAVE_DIR}/*_ep008.pt 2>/dev/null | head -n 1) || true
        
        if [ -z "$SAE_CKPT" ] || [ ! -f "$SAE_CKPT" ]; then
            echo "⚠️  [SAE] Checkpoint not found for dim=${D_SAE}, lam=${LAMBDA}. Skipping..."
            continue
        fi
        
        OUTPUT_PATH="${SAE_CACHE_DIR}/sae_gap_d${D_SAE}_lam${LAMBDA}_normrestored_withnewclass.npz"
        
        echo "▶️  [SAE] Extracting dim=${D_SAE} lam=${LAMBDA} into ${OUTPUT_PATH}"
        python -m cache_extraction.extract_features_lambda_labs \
            --sae_ckpt "$SAE_CKPT" \
            --save_dir "$MODEL_DIR" \
            --model_state_path "$MODEL" \
            --shard_root "$SHARD" \
            --which_layer "$SAE_LAYER" \
            --ignore_splits \
            --restore_token_norm \
            --output_path "$OUTPUT_PATH"
            
    done
done

echo ""
echo "=================================================================="
echo "✅ All extractions complete! Caches saved to $CACHE_ROOT"
echo "=================================================================="
