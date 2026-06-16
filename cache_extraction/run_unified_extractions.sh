# pip install numpy==1.26.4
# pip install tifffile tqdm scanpy


#!/bin/bash
# ==============================================================================
# Unified Batch Feature Extraction for CNN and SAE
# 
# 이 스크립트는 8개의 CNN SEED와 각각의 SAE 설정들에 대해 
# CNN 3개 레이어, SAE 7개 설정의 cache를 모두 추출하여 새로운 'caches' 폴더 구조에 저장합니다.
# ==============================================================================
#set -e  # <-- 중간에 에러가 나도 스크립트가 튕기지 않고 다음 모델로 넘어가도록 비활성화





# 내 생각에 configuration 조금 줄이고. "학습한 데이터"에 대해서 각 class 별로 10000개 뽑아서 linear probe 학습시켜야한다. 그래야지 과적합이 안일어나.
# shuffle = Ture, False는 얼마 정도의 영향인지 모르겠어. 왜 이런결과가 나왔을가. 다만 knn accuracy도 같이 봐야해.
# 지금 이상한게 잘나온건 너무 잘 나왔고 못나온건 너무 못나왔어. 왜 이런 결과가 나왔을까.
# sae_gap_d4096_lam0_normrestored_withnewclass.npz 같은 방식 람다=0인건 할 필요 없다고 생각. cache 뽑는데 하루종일 걸리는데 큰일이네.
# CNN cache는 안뽑아도 된다. SAE만 뽑으면 돼.

# ============================
# 설정 (Configuration)
# ============================
BASE="/home/ubuntu/model-east3/outputs"
SHARD="/home/ubuntu/model-east3/wds_shards_tar"
CACHE_ROOT="/home/ubuntu/model-east3/caches_per_image_centering"

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
    "4096 200"
    "4096 3200"
)


# ============================
# 메인 루프 (Extraction Loop)
# ============================
# mkdir -p "$CACHE_ROOT"

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
    
    #-----------------------------------------------------
    #1. CNN GAP Extraction (3개 레이어)
    #-----------------------------------------------------
    # CNN_CACHE_DIR="${CACHE_ROOT}/CNN_seed${SEED}"
    # mkdir -p "$CNN_CACHE_DIR"
    
    # for LAYER in "${CNN_LAYERS[@]}"; do
    #     CNN_OUT_FILE="${CNN_CACHE_DIR}/cnn_gap_${LAYER}_withnewclass.npz"
    #     if [ -f "$CNN_OUT_FILE" ]; then
    #         echo "✅  [CNN] Cache already exists for layer=${LAYER}. Skipping..."
    #         continue
    #     fi
        
    #     echo "▶️  [CNN] Extracting layer=${LAYER} into ${CNN_CACHE_DIR}"
    #     python -m cache_extraction.extract_cnn_gap \
    #         --save_dir "$MODEL_DIR" \
    #         --model_state_path "$MODEL" \
    #         --shard_root "$SHARD" \
    #         --which_layer "$LAYER" \
    #         --use_all_data \
    #         --output_dir "$CNN_CACHE_DIR" \
    #         --batch_size 64
    # done
    
    # -----------------------------------------------------
    # 2. SAE GAP Extraction (7개 설정)
    # -----------------------------------------------------
    SAE_CACHE_DIR="${CACHE_ROOT}/CNN_seed${SEED}_SAE"
    mkdir -p "$SAE_CACHE_DIR"
    
    # CNN 모델에 종속되는 고정 Reference Mean 저장 경로
    # REFERENCE_MEAN_PATH="${MODEL_DIR}/reference_mean_stage5_out.pt"
    
    for CONFIG in "${SAE_CONFIGS[@]}"; do
        read -r D_SAE LAMBDA <<< "$CONFIG"
        SAE_SAVE_DIR="${MODEL_DIR}/SAE_dim${D_SAE}_lambda${LAMBDA}_seed${SAE_SEED}_no_L2norm_loss"
        
        # 무조건 ep008.pt 체크포인트 파일 사용 (find 명령어 사용하여 Bash 에러 방지)
        SAE_CKPT=$(find "${SAE_SAVE_DIR}" -maxdepth 1 -name "*_ep008.pt" 2>/dev/null | head -n 1)
        
        if [ -z "$SAE_CKPT" ] || [ ! -f "$SAE_CKPT" ]; then
            echo "⚠️  [SAE] Checkpoint not found for dim=${D_SAE}, lam=${LAMBDA}. Skipping..."
            continue
        fi
        
        # 만약 Reference Mean이 아직 없다면 지금 바로 계산! (SAE 학습시 배치단위 중심화를 썼으므로 글로벌 평균은 사용하지 않음!)
        # if [ ! -f "$REFERENCE_MEAN_PATH" ]; then
        #     echo "▶️  [SAE Reference Mean] Computing global token mean for CNN seed ${SEED}..."
        #     python -m cache_extraction.extract_features_lambda_labs \
        #         --sae_ckpt "$SAE_CKPT" \
        #         --save_dir "$MODEL_DIR" \
        #         --model_state_path "$MODEL" \
        #         --shard_root "$SHARD" \
        #         --which_layer "$SAE_LAYER" \
        #         --use_all_data \
        #         --compute_reference_mean \
        #         --reference_mean_path "$REFERENCE_MEAN_PATH"
        # fi
        
        OUTPUT_PATH="${SAE_CACHE_DIR}/sae_gap_d${D_SAE}_lam${LAMBDA}_normrestored_withnewclass.npz"
        
        if [ -f "$OUTPUT_PATH" ]; then
            echo "✅  [SAE] Cache already exists for dim=${D_SAE} lam=${LAMBDA}. Skipping..."
            continue
        fi
        
        echo "▶️  [SAE] Extracting dim=${D_SAE} lam=${LAMBDA} into ${OUTPUT_PATH}"
        python -m cache_extraction.extract_features_lambda_labs \
            --sae_ckpt "$SAE_CKPT" \
            --save_dir "$MODEL_DIR" \
            --model_state_path "$MODEL" \
            --shard_root "$SHARD" \
            --which_layer "$SAE_LAYER" \
            --use_all_data \
            --restore_token_norm \
            --output_path "$OUTPUT_PATH"
            
    done
done

echo ""
echo "=================================================================="
echo "✅ All extractions complete! Caches saved to $CACHE_ROOT"
echo "=================================================================="

============================
3. Linear Classification
============================
echo "▶️  Running Linear Probe Evaluation on all extracted SAE caches..."

# 결과물 CSV 경로 (훈련에 쓴 기존 4개 클래스만 자동 필터링되어 평가됨)
OUTPUT_CSV_NAME="L0_FVU_linear_classification/unified_sae_linear_probe_results.csv"

# 폴더 미리 생성
mkdir -p "/home/ubuntu/model-east3/L0_FVU_linear_classification"

python -m sae_project.step18_eval_probe_from_cache \
    --cache_dir "$CACHE_ROOT" \
    --save_csv "$OUTPUT_CSV_NAME" \
    --model_base_dir "$BASE"

echo "=================================================================="
echo "✅ Linear Classification complete!"
echo "=================================================================="
