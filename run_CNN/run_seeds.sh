SCRIPT_DIR="/home/ubuntu/model-east3"                  # 상위 폴더를 기준으로 설정
BASE_DIR="/home/ubuntu/model-east3/outputs"
SEEDS=(42 45)

# 1. 파이썬이 run_CNN을 패키지로 인식할 수 있도록 상위 디렉토리로 먼저 이동합니다.
cd "$SCRIPT_DIR"

for SEED in "${SEEDS[@]}"; do
    SAVE_DIR="${BASE_DIR}/MoCo_seed${SEED}_refactoring"
    echo ""
    echo "============================================"
    echo "  Starting seed=${SEED}  ->  ${SAVE_DIR}"
    echo "  $(date)"
    echo "============================================"

    # 2. 파일 경로 대신 'run_CNN.train' 이라는 모듈 형태로 호출합니다.
    python -m run_CNN.train \
        --seed "$SEED" \
        --save_dir "$SAVE_DIR" \
        --epochs 100 \
        --batch_size 512 \
        --queue_size 65536 \
        --moco_m 0.995 \
        --temp 0.07 \
        --use_bf16 \
        --auto_resume \
        --shard_root "/home/ubuntu/model-east3/wds_shards_tar"

    echo "Seed ${SEED} finished at $(date)"
    echo ""
done
