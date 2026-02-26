#!/bin/bash
# ==============================================================================
# Run pairwise CKA between all CNN encoder seeds
# Outputs: cka_results/cka_matrix.csv  (표 형태)
# ==============================================================================

BASE_DIR="/home/ubuntu/model-east3/outputs"
SHARD_ROOT="/home/ubuntu/model-east3/wds_shards_tar"
OUTPUT_DIR="${BASE_DIR}/cka_results_pooling1"
mkdir -p "${OUTPUT_DIR}"

# ── MoCo seeds (GAP L2 norm 적용) ──
MOCO_SEEDS=(42 45 87 95 123 124 125 256 445 457)

# ── MoCo seeds (no GAP L2 norm) ──
NO_NORM_SEEDS=(42 45 123 87)

# ── 비교할 모델 체크포인트 목록 생성 ──
declare -a CKPT_PATHS
declare -a CKPT_NAMES

for SEED in "${MOCO_SEEDS[@]}"; do
    CKPT="${BASE_DIR}/MoCo_seed${SEED}/best_model.pt"
    if [ -f "$CKPT" ]; then
        CKPT_PATHS+=("$CKPT")
        CKPT_NAMES+=("MoCo_s${SEED}")
    else
        echo "⚠️  Not found: $CKPT"
    fi
done

for SEED in "${NO_NORM_SEEDS[@]}"; do
    CKPT="${BASE_DIR}/MoCo_seed${SEED}_no_GAPL2norm/best_model.pt"
    if [ -f "$CKPT" ]; then
        CKPT_PATHS+=("$CKPT")
        CKPT_NAMES+=("noNorm_s${SEED}")
    else
        echo "⚠️  Not found: $CKPT"
    fi
done

N=${#CKPT_PATHS[@]}
TOTAL_PAIRS=$(( N * (N - 1) / 2 ))
echo "Found ${N} models → ${TOTAL_PAIRS} pairwise comparisons"
echo ""

# ── val_split.csv가 있는 save_dir (첫 번째 모델 기준) ──
SAVE_DIR_1="${BASE_DIR}/MoCo_seed${MOCO_SEEDS[0]}"

# ── Pairwise CKA 실행 (skip if already done) ──
DONE=0
SKIPPED=0
COUNT=0
for ((i=0; i<N; i++)); do
    for ((j=i+1; j<N; j++)); do
        NAME_A="${CKPT_NAMES[$i]}"
        NAME_B="${CKPT_NAMES[$j]}"
        CKPT_A="${CKPT_PATHS[$i]}"
        CKPT_B="${CKPT_PATHS[$j]}"
        COUNT=$((COUNT + 1))

        PAIR_DIR="${OUTPUT_DIR}/${NAME_A}_vs_${NAME_B}"
        mkdir -p "${PAIR_DIR}"
        JSON_FILE="${PAIR_DIR}/cka_results.json"

        # ── Skip if already computed ──
        if [ -f "$JSON_FILE" ]; then
            SKIPPED=$((SKIPPED + 1))
            echo "  ⏭ [${COUNT}/${TOTAL_PAIRS}] ${NAME_A} vs ${NAME_B} — already done, skipping"
            continue
        fi

        echo "═══════════════════════════════════════════"
        echo "  [${COUNT}/${TOTAL_PAIRS}] CKA: ${NAME_A} vs ${NAME_B}"
        echo "═══════════════════════════════════════════"

        python -m model_similarity_lambda_labs.cka_analysis \
            --ckpt_path_1 "${CKPT_A}" \
            --ckpt_path_2 "${CKPT_B}" \
            --shard_root "${SHARD_ROOT}" \
            --save_dir_1 "${SAVE_DIR_1}" \
            --output_dir "${PAIR_DIR}" \
            --pooling_size 1 \
            --num_samples 10000 \
            --which_layer "refine_out" \
            --seed 42 2>&1 | tee "${PAIR_DIR}/log.txt"

        DONE=$((DONE + 1))
        echo ""
    done
done

echo ""
echo "═══════════════════════════════════════════"
echo "  Computed: ${DONE}, Skipped: ${SKIPPED}, Total: ${TOTAL_PAIRS}"
echo "═══════════════════════════════════════════"

# ── Rebuild CSV from ALL existing JSON files ──
CSV_FILE="${OUTPUT_DIR}/cka_matrix.csv"
echo "Model_A,Model_B,Linear_CKA,Ctrl_CKA,SNCA_CKA,GBA_CKA,LRRK2_CKA" > "${CSV_FILE}"

for ((i=0; i<N; i++)); do
    for ((j=i+1; j<N; j++)); do
        NAME_A="${CKPT_NAMES[$i]}"
        NAME_B="${CKPT_NAMES[$j]}"
        PAIR_DIR="${OUTPUT_DIR}/${NAME_A}_vs_${NAME_B}"
        JSON_FILE="${PAIR_DIR}/cka_results.json"

        if [ -f "$JSON_FILE" ]; then
            LINEAR=$(python3 -c "import json; d=json.load(open('${JSON_FILE}')); print(f\"{d['linear_cka']:.4f}\")")
            CTRL=$(python3 -c "import json; d=json.load(open('${JSON_FILE}')); print(f\"{d['per_class_linear_cka'].get('Control',0):.4f}\")")
            SNCA=$(python3 -c "import json; d=json.load(open('${JSON_FILE}')); print(f\"{d['per_class_linear_cka'].get('SNCA',0):.4f}\")")
            GBA=$(python3 -c "import json; d=json.load(open('${JSON_FILE}')); print(f\"{d['per_class_linear_cka'].get('GBA',0):.4f}\")")
            LRRK2=$(python3 -c "import json; d=json.load(open('${JSON_FILE}')); print(f\"{d['per_class_linear_cka'].get('LRRK2',0):.4f}\")")
            echo "${NAME_A},${NAME_B},${LINEAR},${CTRL},${SNCA},${GBA},${LRRK2}" >> "${CSV_FILE}"
        fi
    done
done

echo ""
echo "═══════════════════════════════════════════"
echo "  ALL DONE – Results saved to:"
echo "  ${CSV_FILE}"
echo "═══════════════════════════════════════════"
echo ""
echo "=== CKA Matrix ==="
column -t -s',' "${CSV_FILE}"
