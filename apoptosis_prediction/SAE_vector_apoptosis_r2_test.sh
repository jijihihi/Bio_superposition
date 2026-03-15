#!/bin/bash
# ==============================================================================
# SAE vector → Apoptosis R² prediction sweep
#
# Sweeps: 4 SAE seeds × 2 GAP L2 × 2 models × 5 filter combos = 80 runs
# ==============================================================================

## 뽑아낸 cache 쓴다. 따라서 extract_features_lambda_labs 로 뽑아낸 cache 쓴다. 이거는 batch centering. shuffle = false해서.

# flat_tokens = fmap.view(-1, C)                                    # 배치 전체 토큰을 1차원으로
# flat_tokens = flat_tokens - flat_tokens.mean(dim=0, keepdim=True)  # 배치 차원에서 평균 빼기

set -e

BASE="/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/MoCo_seed87"
APOPTOSIS_CSV="/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/세포이미지별 사멸율/이미지별_세포사멸율_7200.csv"
OUTPUT_BASE="/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/apoptosis_r2_results/SAE_vector"

SAE_SEEDS=(48 123 777 856)
GAP_L2_LIST=("" "--gap_l2_norm")
MODELS=("ridge" "xgboost")

# Filter combos: "label|filter_args"
FILTER_COMBOS=(
    "no_filter|--filter_mode none"
    "cv0.1|--filter_mode cv --min_cv 0.1"
    "cv0.2|--filter_mode cv --min_cv 0.2"
    "cv0.1_de0.58|--filter_mode cv de --min_cv 0.1 --de_min_log2fc 0.58 --de_adj_p 0.05"
    "cv0.1_de1.0|--filter_mode cv de --min_cv 0.1 --de_min_log2fc 1.0 --de_adj_p 0.05"
    "cv0.2_de0.58|--filter_mode cv de --min_cv 0.2 --de_min_log2fc 0.58 --de_adj_p 0.05"
    "cv0.2_de1.0|--filter_mode cv de --min_cv 0.2 --de_min_log2fc 1.0 --de_adj_p 0.05"
)

TOTAL=0
SKIP=0
for SAE_SEED in "${SAE_SEEDS[@]}"; do
    CACHE="${BASE}/SAE_seed${SAE_SEED}_no_L2norm_loss/features_cache_stage5_out_normrestored_all_no_SAE_GAP_L2_norm_again_d8192_sp800.npz"

    if [ ! -f "$CACHE" ]; then
        echo "⚠️  Cache not found: $CACHE"
        continue
    fi

    for GAP_L2 in "${GAP_L2_LIST[@]}"; do
        if [ -n "$GAP_L2" ]; then
            L2_LABEL="l2norm"
        else
            L2_LABEL="no_l2norm"
        fi

        for MODEL in "${MODELS[@]}"; do
            for COMBO in "${FILTER_COMBOS[@]}"; do
                FILTER_LABEL="${COMBO%%|*}"
                FILTER_ARGS="${COMBO#*|}"

                OUT_DIR="${OUTPUT_BASE}/SAE_seed${SAE_SEED}_${L2_LABEL}_${FILTER_LABEL}"
                TOTAL=$((TOTAL + 1))

                # Resume: skip if result JSON already exists
                RESULT_JSON=$(find "$OUT_DIR" -name "r2_results_*_${MODEL^}.json" 2>/dev/null | head -1)
                if [ -n "$RESULT_JSON" ]; then
                    SKIP=$((SKIP + 1))
                    continue
                fi

                echo -ne "\r[$TOTAL] seed=$SAE_SEED l2=$L2_LABEL model=$MODEL filter=$FILTER_LABEL   "

                python -m kendall_correlation_coefficient.apoptosis_r2_test \
                    --features_cache "$CACHE" \
                    --apoptosis_csv "$APOPTOSIS_CSV" \
                    --model "$MODEL" \
                    --dead_threshold 5e-5 \
                    --seed 42 \
                    --cv_folds 5 \
                    --output_dir "$OUT_DIR" \
                    --quiet \
                    $GAP_L2 \
                    $FILTER_ARGS

            done
        done
    done
done

echo ""
echo "════════════════════════════════════════════════════════"
echo "All SAE R² runs complete! Total: $TOTAL, Skipped: $SKIP"
echo "Results: ${OUTPUT_BASE}"
echo "════════════════════════════════════════════════════════"
