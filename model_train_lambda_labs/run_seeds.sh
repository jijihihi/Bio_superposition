#!/bin/bash
# Lambda Labs에서 seed별로 순차 실행
# 사용법:  bash /home/ubuntu/run_seeds.sh
# 또는:   nohup bash /home/ubuntu/run_seeds.sh > /home/ubuntu/run_seeds.log 2>&1 &
#         (SSH 끊겨도 계속 실행됨)

SCRIPT="/home/ubuntu/model-east3/train.py"
BASE_DIR="/home/ubuntu/model-east3/outputs"

SEEDS=(42 45 123)

for SEED in "${SEEDS[@]}"; do
    SAVE_DIR="${BASE_DIR}/MoCo_seed${SEED}_no_GAPL2norm"
    echo ""
    echo "============================================"
    echo "  Starting seed=${SEED}  ->  ${SAVE_DIR}"
    echo "  $(date)"
    echo "============================================"

    python "$SCRIPT" \
        --seed "$SEED" \
        --save_dir "$SAVE_DIR" \
        --epochs 100 \
        --batch_size 512 \
        --queue_size 65536 \
        --moco_m 0.995 \
        --temp 0.07 \
        --use_bf16 \
        --auto_resume

    echo "Seed ${SEED} finished at $(date)"
    echo ""
done

echo "=== All seeds done! ==="
