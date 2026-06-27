#!/bin/bash
# ==============================================================================
# SAE vector → cell_death R² prediction sweep (Iterating over CNN Seeds)
# ==============================================================================

set -e

BASE="/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/caches_per_image_centering"
cell_death_CSV="/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/세포이미지별 사멸율/이미지별_세포사멸율_7200.csv"
OUTPUT_BASE="/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/caches_per_image_centering/SAE_vector_per_image_centering"
CNN_SEEDS=(42 87 95 123 124 256 445 457)
GAP_L2_LIST=("--gap_l2_norm")
MODELS=("ridge")

# Filter combos: "label|filter_args"
FILTER_COMBOS=(
    "no_filter|--filter_mode none"
)

TOTAL=0
SKIP=0
for CNN_SEED in "${CNN_SEEDS[@]}"; do
    CACHE="${BASE}/CNN_seed${CNN_SEED}_SAE/sae_gap_d8192_lam800_normrestored_withnewclass.npz"

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

                OUT_DIR="${OUTPUT_BASE}/CNN_seed${CNN_SEED}_${L2_LABEL}_${FILTER_LABEL}"
                TOTAL=$((TOTAL + 1))

                # Resume: skip if result JSON already exists
                RESULT_JSON=$(find "$OUT_DIR" -name "r2_results_*_${MODEL^}.json" 2>/dev/null | head -1)
                if [ -n "$RESULT_JSON" ]; then
                    SKIP=$((SKIP + 1))
                    continue
                fi

                echo -ne "\r[$TOTAL] cnn_seed=$CNN_SEED l2=$L2_LABEL model=$MODEL filter=$FILTER_LABEL   \n"

                python -m cell_death_prediction.cell_death_r2_test \
                    --features_cache "$CACHE" \
                    --cell_death_csv "$cell_death_CSV" \
                    --model "$MODEL" \
                    --dead_threshold 1e-5 \
                    --seed 42 \
                    --cv_folds 5 \
                    --n_repeats 2 \
                    --n_permutations 0 \
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
