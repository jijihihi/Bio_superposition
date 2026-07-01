#!/bin/bash


SCRIPT_DIR="/home/ubuntu/model-east3"                  
BASE_DIR="/home/ubuntu/model-east3/outputs"


cd "$SCRIPT_DIR"

echo "============================================"
echo " 1) CNN Training with L2 Normalization (Default)"
echo "============================================"
SEEDS_L2=(42 87 95 123 124 256 445 457)

for SEED in "${SEEDS_L2[@]}"; do
    SAVE_DIR="${BASE_DIR}/MoCo_seed${SEED}_refactoring"
    echo ""
    echo "--------------------------------------------"
    echo "  Starting seed=${SEED} (L2 Norm) ->  ${SAVE_DIR}"
    echo "  $(date)"
    echo "--------------------------------------------"

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
        --shard_root "${SCRIPT_DIR}/wds_shards_tar" \
        --lr 0.1 \
        --wd 1e-4 \
        --embed_dim 512 \
        --blocks "2,2,2,3" \
        --dilations "1,1,1,1" \
        --refine_blocks 1 \
        --proj_layers 2 \
        --proj_hidden 2048 \
        --proj_dropout 0.0 \
        --sgd_momentum 0.9 \
        --sgd_nesterov \
        --grad_clip 1.0 \
        --symmetric_loss \
        --renorm_every 1 \
        --renorm_k_every 0 \
        --queue_dtype_fp16 \
        --num_classes 4 \
        --patience 100 \
        --lp_epochs 3 \
        --lp_batch_size 16384 \
        --lp_lr 0.1 \
        --lp_wd 0.0 \
        --lp_momentum 0.9 \
        --lp_enc_bs 128 \
        --warmup_epochs 4

    for LAYER in "stage5_mid" "stage5_out" "refine_out"; do
        python -m representation_eval.extract_cnn_gap \
            --save_dir "$SAVE_DIR" \
            --model_state_path "$SAVE_DIR/best_model.pt" \
            --shard_root "${SCRIPT_DIR}/wds_shards_tar" \
            --which_layer "$LAYER" \
            --use_all_data

    done
        
    python -m run_CNN.linear_eval_from_cache \
        --save_dir "$SAVE_DIR" \
        --cache_path "$SAVE_DIR/CNN_GAP/cnn_gap_refine_out.npz" \
        --batch_size 512 \
        --wd 1e-4 \
        --apply_l2_norm \
        --save_cm_plot

    echo "Seed ${SEED} (L2 Norm) finished at $(date)"
done

echo ""
echo "============================================"
echo " 2) CNN Evaluation without L2 Normalization"
echo "============================================"
SEEDS_NO_L2=(42 87 124)

for SEED in "${SEEDS_NO_L2[@]}"; do
    SAVE_DIR="${BASE_DIR}/MoCo_seed${SEED}_no_L2norm"
    echo ""
    echo "--------------------------------------------"
    echo "  Evaluating seed=${SEED} (NO L2 Norm) ->  ${SAVE_DIR}"
    echo "--------------------------------------------"

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
        --shard_root "${SCRIPT_DIR}/wds_shards_tar" \
        --no_l2_norm_pool \
        --lr 0.1 \
        --wd 1e-4 \
        --embed_dim 512 \
        --blocks "2,2,2,3" \
        --dilations "1,1,1,1" \
        --refine_blocks 1 \
        --proj_layers 2 \
        --proj_hidden 2048 \
        --proj_dropout 0.0 \
        --sgd_momentum 0.9 \
        --sgd_nesterov \
        --grad_clip 1.0 \
        --symmetric_loss \
        --renorm_every 1 \
        --renorm_k_every 0 \
        --queue_dtype_fp16 \
        --num_classes 4 \
        --patience 100 \
        --lp_epochs 3 \
        --lp_batch_size 16384 \
        --lp_lr 0.1 \
        --lp_wd 0.0 \
        --lp_momentum 0.9 \
        --lp_enc_bs 128 \
        --warmup_epochs 4

    for LAYER in "stage5_mid" "stage5_out" "refine_out"; do
        python -m representation_eval.extract_cnn_gap \
            --save_dir "$SAVE_DIR" \
            --model_state_path "$SAVE_DIR/best_model.pt" \
            --shard_root "${SCRIPT_DIR}/wds_shards_tar" \
            --which_layer "$LAYER" \
            --use_all_data
    done
        
    python -m run_CNN.linear_eval_from_cache \
        --save_dir "$SAVE_DIR" \
        --cache_path "$SAVE_DIR/CNN_GAP/cnn_gap_refine_out.npz"

    echo "Seed ${SEED} (NO L2 Norm) eval finished at $(date)"
done

echo ""
echo "============================================"
echo " 3) Plotting Accuracy Results (L2 Norm Models Only)"
echo "============================================"
L2_DIRS=()
L2_CACHES=()
for SEED in "${SEEDS_L2[@]}"; do
    L2_DIRS+=("${BASE_DIR}/MoCo_seed${SEED}_refactoring")
    L2_CACHES+=("${BASE_DIR}/MoCo_seed${SEED}_refactoring/CNN_GAP/cnn_gap_refine_out_all.npz")
done

python -m run_CNN.plot_eval_results \
    --result_dirs "${L2_DIRS[@]}" \
    --output_file "${BASE_DIR}/L2_Norm_Accuracy_Plot.svg"

echo ""
echo "============================================"
echo " 4) CKA Similarity Analysis (From Caches)"
echo "============================================"
NO_L2_CACHES=()
for SEED in "${SEEDS_NO_L2[@]}"; do
    NO_L2_CACHES+=("${BASE_DIR}/MoCo_seed${SEED}_no_L2norm/CNN_GAP/cnn_gap_refine_out_all.npz")
done

# A vs A (L2 Norm vs L2 Norm)
echo "  [CKA] Group A vs Group A..."
python -m model_CKA.cka_from_cache \
    --group_a "${L2_CACHES[@]}" \
    --norm_a \
    --title "Linear CKA: Group A (L2 Norm) vs Group A" \
    --output_svg "${BASE_DIR}/CKA_GroupA_vs_GroupA.svg"

# A vs B (L2 Norm vs No L2 Norm)
echo "  [CKA] Group A vs Group B..."
python -m model_CKA.cka_from_cache \
    --group_a "${L2_CACHES[@]}" \
    --group_b "${NO_L2_CACHES[@]}" \
    --norm_a \
    --title "Linear CKA: Group A (L2 Norm) vs Group B (No L2 Norm)" \
    --output_svg "${BASE_DIR}/CKA_GroupA_vs_GroupB.svg"

echo ""
echo "=========================================================================="
echo " 5) Cell Death R² Prediction Evaluation"
echo "=========================================================================="

CELL_DEATH_CSV="${BASE_DIR}/CellDeath_QC_patches/per_image_celldeath_rate.csv"
OUT_ROOT="${BASE_DIR}/refactoring/cell_death_evaluation"
mkdir -p "${OUT_ROOT}"

# Remove existing aggregated CSVs to prevent duplicate appending on re-runs
rm -f "${OUT_ROOT}/aggregated_r2_all_folds.csv"
rm -f "${OUT_ROOT}/aggregated_r2_l2effect_folds.csv"

MODELS=("ridge" "xgboost")
LAYERS=("stage5_mid" "stage5_out" "refine_out")

echo "  5.1) Layer Information Comparison (L2-Trained Models with L2 Norm)"
for SEED in "${SEEDS_L2[@]}"; do
    MODEL_DIR="${BASE_DIR}/MoCo_seed${SEED}_refactoring"
    for LAYER in "${LAYERS[@]}"; do
        CACHE="${MODEL_DIR}/CNN_GAP/cnn_gap_${LAYER}_all.npz"
        if [ ! -f "$CACHE" ]; then continue; fi

        for MODEL in "${MODELS[@]}"; do
            OUT_DIR="${OUT_ROOT}/Layer_Comparison/seed${SEED}_${LAYER}_l2norm"
            mkdir -p "${OUT_DIR}"

            echo "    ▶ [Layer Comp] Seed=${SEED} | Layer=${LAYER} | Model=${MODEL}"
            python -m cell_death.apoptosis_r2_test_clean \
                --features_cache "$CACHE" \
                --cell_death_csv "$CELL_DEATH_CSV" \
                --model "$MODEL" \
                --pca_dim 250 \
                --gap_l2_norm \
                --seed 42 \
                --cv_folds 5 \
                --n_permutations 0 \
                --output_dir "$OUT_DIR" \
                --config_name "L2Norm_Trained_Inf_L2_ON" \
                --seed_name "$SEED" \
                --layer_name "$LAYER" \
                --csv_out "${OUT_ROOT}/aggregated_r2_all_folds.csv"
        done
    done
done

echo "  5.1.1) Generating Layer Comparison Bar Plot"
for MODEL in "${MODELS[@]}"; do
    python -m cell_death.plot_layer_comparison \
        --results_dir "${OUT_ROOT}" \
        --config "L2Norm_Trained_Inf_L2_ON" \
        --model "$MODEL"
done

echo ""
echo "  5.2) L2 Normalization Effectiveness (stage5_out layer)"
# 5.2.1 L2-Trained Models
for SEED in "${SEEDS_L2[@]}"; do
    MODEL_DIR="${BASE_DIR}/MoCo_seed${SEED}_refactoring"
    CACHE="${MODEL_DIR}/CNN_GAP/cnn_gap_stage5_out_all.npz"
    if [ ! -f "$CACHE" ]; then continue; fi

    for MODEL in "${MODELS[@]}"; do
        for L2_MODE in "ON" "OFF"; do
            L2_ARG=""
            if [ "$L2_MODE" == "ON" ]; then
                CONFIG_NAME="MoCo_l2norm"
                L2_ARG="--gap_l2_norm"
            else
                CONFIG_NAME="MoCo_raw"
            fi
            
            OUT_DIR="${OUT_ROOT}/L2_Effect/Train_L2_seed${SEED}_Inf_${L2_MODE}"
            mkdir -p "${OUT_DIR}"

            echo "    ▶ [L2 Effect - Train:L2] Seed=${SEED} | Inf L2=${L2_MODE} | Model=${MODEL}"
            python -m cell_death.apoptosis_r2_test_clean \
                --features_cache "$CACHE" \
                --cell_death_csv "$CELL_DEATH_CSV" \
                --model "$MODEL" \
                --pca_dim 250 \
                $L2_ARG \
                --seed 42 \
                --cv_folds 5 \
                --n_permutations 0 \
                --output_dir "$OUT_DIR" \
                --config_name "$CONFIG_NAME" \
                --seed_name "$SEED" \
                --layer_name "stage5_out" \
                --csv_out "${OUT_ROOT}/aggregated_r2_l2effect_folds.csv"
        done
    done
done

# 5.2.2 No-L2-Trained Models
for SEED in "${SEEDS_NO_L2[@]}"; do
    MODEL_DIR="${BASE_DIR}/MoCo_seed${SEED}_no_L2norm"
    CACHE="${MODEL_DIR}/CNN_GAP/cnn_gap_stage5_out_all.npz"
    if [ ! -f "$CACHE" ]; then continue; fi

    for MODEL in "${MODELS[@]}"; do
        for L2_MODE in "ON" "OFF"; do
            L2_ARG=""
            if [ "$L2_MODE" == "ON" ]; then
                CONFIG_NAME="noNorm_l2norm"
                L2_ARG="--gap_l2_norm"
            else
                CONFIG_NAME="noNorm_raw"
            fi
            
            OUT_DIR="${OUT_ROOT}/L2_Effect/Train_NoL2_seed${SEED}_Inf_${L2_MODE}"
            mkdir -p "${OUT_DIR}"

            echo "    ▶ [L2 Effect - Train:NoL2] Seed=${SEED} | Inf L2=${L2_MODE} | Model=${MODEL}"
            python -m cell_death.apoptosis_r2_test_clean \
                --features_cache "$CACHE" \
                --cell_death_csv "$CELL_DEATH_CSV" \
                --model "$MODEL" \
                --pca_dim 250 \
                $L2_ARG \
                --seed 42 \
                --cv_folds 5 \
                --n_permutations 0 \
                --output_dir "$OUT_DIR" \
                --config_name "$CONFIG_NAME" \
                --seed_name "$SEED" \
                --layer_name "stage5_out" \
                --csv_out "${OUT_ROOT}/aggregated_r2_l2effect_folds.csv"
        done
    done
done

echo "  5.2.3) Generating L2 Normalization Effect Plots"
for MODEL in "${MODELS[@]}"; do

    python -m cell_death.plot_l2norm_effect_slope \
        --results_dir "${OUT_ROOT}" \
        --training_config "MoCo" \
        --layer "stage5_out" \
        --model "$MODEL"

    python -m cell_death.plot_l2norm_effect_slope \
        --results_dir "${OUT_ROOT}" \
        --training_config "noNorm" \
        --layer "stage5_out" \
        --model "$MODEL"
done

echo ""
echo "=========================================================================="
echo " 6) eRank Evaluation (CNN 3 Layers)"
echo "=========================================================================="
for SEED in "${SEEDS_L2[@]}"; do
    MODEL_DIR="${BASE_DIR}/MoCo_seed${SEED}_refactoring"
    
    for LAYER in "stage5_mid" "stage5_out" "refine_out"; do
        CACHE="${MODEL_DIR}/CNN_GAP/cnn_gap_${LAYER}_all.npz"
        if [ ! -f "$CACHE" ]; then continue; fi
        
        ERANK_OUT="${MODEL_DIR}/CNN_GAP/erank/${LAYER}"
        mkdir -p "${ERANK_OUT}"
        
        echo "  ▶ Measuring eRank for Seed=${SEED}, Layer=${LAYER}..."
        python -m eRank.effective_rank \
            --cnn_cache "$CACHE" \
            --gap_l2_norm \
            --pca_dim 250 \
            --norm std \
            --output_dir "$ERANK_OUT"
    done
done

echo "  ▶ Generating eRank Bar Plot..."
ERANK_PLOT_OUT="${BASE_DIR}/eRank_Plots"
mkdir -p "${ERANK_PLOT_OUT}"
python -m eRank.plot_erank_bar \
    --base_dir "${BASE_DIR}" \
    --output_dir "${ERANK_PLOT_OUT}"


echo ""
echo "=========================================================================="
echo " 7) UMAP Visualizations (stage5_out)"
echo "=========================================================================="
for SEED in "${SEEDS_L2[@]}"; do
    MODEL_DIR="${BASE_DIR}/MoCo_seed87_refactoring"
    CACHE="${MODEL_DIR}/CNN_GAP/cnn_gap_stage5_out_all.npz"
    if [ ! -f "$CACHE" ]; then continue; fi

    UMAP_OUT="${BASE_DIR}/UMAP_Plots/seed${SEED}_stage5_out"
    mkdir -p "${UMAP_OUT}"
    
    echo "  ▶ Generating UMAP for Seed=${SEED}..."
    python -m visualization.umap_plot_gap \
        --cache_path "$CACHE" \
        --cell_death_csv "$CELL_DEATH_CSV" \
        --output_dir "$UMAP_OUT" \
        --n_samples 6000 \
        --apply_l2_norm \
        --cmap magma_r \
        --cmap_start 0.2 \
        --alpha 0.6 \
        --n_neighbors 10 \
        --min_dist 0.25 \
        --vmax_pctl 90.0 \
        --dot_size 6.0 \

        
done

echo "=== All CNN trainings, evaluations, CKA, Cell Death R2, eRank, and UMAP done! ==="
