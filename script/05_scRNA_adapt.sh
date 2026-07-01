#!/bin/bash
set -e

BASE="/home/ubuntu/model-east3"
CELL_DEATH_CSV="${BASE_DIR}/CellDeath_QC_patches/per_image_celldeath_rate.csv"
OUT_BASE="${BASE}/outputs/refactoring/scRNA_adapt"

PCAS=(30 40 50)
KS=(15 20 25 30 35)

echo "=========================================================================="
echo " 0. Setup & Finding Cache Files for SAE_dim8192_lambda800"
echo "=========================================================================="
CACHE_FILES=$(find "${BASE}/outputs" -type f -name "sae_refactoring_gap_*_all.npz" | sort)

for CACHE_OUT in $CACHE_FILES; do
    if [[ "$CACHE_OUT" =~ SAE_refactoring_dim8192_lambda800 ]]; then
        if [[ "$CACHE_OUT" =~ MoCo_seed([0-9]+)_refactoring ]]; then
            CNN_SEED="${BASH_REMATCH[1]}"
            
            SEED_OUT_DIR="${OUT_BASE}/seed_${CNN_SEED}"
            mkdir -p "${SEED_OUT_DIR}"

            echo ""
            echo "=========================================================================="
            echo " [SEED: ${CNN_SEED}] 1. Pairwise Visualization (PHATE, PAGA)"
            echo "=========================================================================="
            python -m trajectory_inference_pipeline.pairwise_vis \
                --features_cache "$CACHE_OUT" \
                --cell_death_csv "$CELL_DEATH_CSV" \
                --output_dir "${SEED_OUT_DIR}/pairwise_phate" \
                --n_neighbors 5 \
                --pca_dim 100 \
                --filter_mode "none" \
                --dead_threshold 1e-5 \
                --norm "log_std" \
                --gap_l2_norm \
                --seed $CNN_SEED \
                --phate_decay 120 \
                --phate_t 20 \
                --plot_ctrl_size 6 \
                --plot_ctrl_alpha 0.6 \
                --plot_mut_size 6 \
                --plot_ctrl_color "#CCCCCC" \
                --plot_mut_alpha 0.6 \
                --plot_invalid_color "#333333" \
                --paga_figsize 2.0 2.0 \
                --paga_threshold 0.1 \
                --paga_edge_width_scale 0.65 \
                --paga_min_edge_width 0.15 \
                --leiden_resolution 0.45 \
                --de_mode "per_mut" \
                --samples_per_class 5000

            echo ""
            echo "=========================================================================="
            echo " [SEED: ${CNN_SEED}] 2. DPT & Downstream Stats Parameter Sweep"
            echo "=========================================================================="
            SWEEP_OUT_DIR="${SEED_OUT_DIR}/DPT_Sweep"
            mkdir -p "$SWEEP_OUT_DIR"

            for PCA in "${PCAS[@]}"; do
                for K in "${KS[@]}"; do
                    echo "--------------------------------------------------------"
                    echo "  Running DPT & Stats for PCA=$PCA, K=$K"
                    
                    # Pairwise DPT
                    python -m trajectory_inference_pipeline.pairwise_dpt \
                        --features_cache "$CACHE_OUT" \
                        --cell_death_csv "$CELL_DEATH_CSV" \
                        --output_dir "$SWEEP_OUT_DIR" \
                        --n_neighbors $K \
                        --pca_dim $PCA \
                        --seed $CNN_SEED \
                        --filter_mode "none" \
                        --min_cv 0.2 \
                        --de_adj_p 1.0 \
                        --de_min_log2fc 0.0 \
                        --dead_threshold 1e-5 \
                        --norm "log_std" \
                        --gap_l2_norm \
                        --root_mode "diffmap" \
                        --gam_splines 5 \
                        --gam_trim_pctl 5 95 \
                        --de_eval_split 0.5 \
                        --de_mode "per_mut" \
                        --samples_per_class 5000 \
                        --no_plot

                    # Downstream Stats
                    python -m trajectory_inference_pipeline.downstream_stats \
                        --features_cache "$CACHE_OUT" \
                        --cell_death_csv "$CELL_DEATH_CSV" \
                        --output_dir "$SWEEP_OUT_DIR" \
                        --n_neighbors $K \
                        --pca_dim $PCA \
                        --seed $CNN_SEED \
                        --filter_mode "none" \
                        --min_cv 0.1 \
                        --de_adj_p 0.05 \
                        --de_min_log2fc 1.0 \
                        --dead_threshold 1e-5 \
                        --norm "log_std" \
                        --gap_l2_norm \
                        --permutation_n 10
                done
            done

            echo "  [SEED: ${CNN_SEED}] Running heatmap generation for DPT Sweep..."
            python -m trajectory_inference_pipeline.plot_dpt_heatmap \
                --input_dir "$SWEEP_OUT_DIR"
        fi
    fi
done

echo ""
echo "🏁 All scRNA adaptation pipeline steps completed!"
