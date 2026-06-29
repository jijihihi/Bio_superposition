#!/bin/bash
set -e

BASE="/home/ubuntu/model-east3"

CELL_DEATH_CSV="/home/ubuntu/model-east3/이미지별_세포사멸율_7200.csv"

echo "=========================================================================="
echo " 0. SAE Local Linearity (KNN Std) Evaluation"
echo "=========================================================================="
KNN_OUT_BASE="${BASE}/outputs/refactoring/local_linearity_dim_and_k_sweep"
mkdir -p "${KNN_OUT_BASE}"
K_NEIGHBORS="5 10 15 20 25"

echo "▶️ Finding SAE cache files..."
# Find all extracted SAE cache files
CACHE_FILES=$(find "${BASE}/outputs" -type f -name "sae_refactoring_gap_*_all.npz" | sort)

for CACHE_OUT in $CACHE_FILES; do
    # Regex to extract CNN_SEED, D_SAE, and LAMBDA from the folder structure
    if [[ "$CACHE_OUT" =~ MoCo_seed([0-9]+)_refactoring/SAE_refactoring_dim([0-9]+)_lambda([0-9]+) ]]; then
        CNN_SEED="${BASH_REMATCH[1]}"
        D_SAE="${BASH_REMATCH[2]}"
        LAMBDA="${BASH_REMATCH[3]}"
        
        CNN_CACHE="${BASE}/outputs/MoCo_seed${CNN_SEED}_refactoring/CNN_GAP/cnn_gap_stage5_out_all.npz"
        
        if [ ! -f "$CNN_CACHE" ]; then
            echo "⚠️ CNN Cache not found for Seed=${CNN_SEED}. Skipping KNN..."
            continue
        fi

        OUT_DIR="${KNN_OUT_BASE}/seed_${CNN_SEED}/d${D_SAE}_lam${LAMBDA}"
        mkdir -p "$OUT_DIR"

        echo "▶️ Running Local KNN Std for CNN_SEED=${CNN_SEED}, d_sae=${D_SAE}, lambda=${LAMBDA}..."
        python -m cell_death.local_knn_std \
            --cnn_cache "$CNN_CACHE" \
            --sae_cache "$CACHE_OUT" \
            --cell_death_csv "$CELL_DEATH_CSV" \
            --k_neighbors $K_NEIGHBORS \
            --dead_threshold 1e-5 \
            --pca_dim 0 \
            --output_dir "$OUT_DIR" \
            --n_permutations 0
    fi
done

echo "🏁 Finished SAE Local KNN Std Evaluation!"
echo ""

echo "▶️ Generating KNN Std & Moran's I Trend Plots..."
python -m cell_death.plot_trend_sweep_multiseed --base_dir "${KNN_OUT_BASE}"
echo "🏁 Plots generated at ${KNN_OUT_BASE}/plots"
echo ""

echo "=========================================================================="
echo " 1. SAE Cell Death Prediction"
echo "=========================================================================="
SAE_RESULTS_DIR="${BASE}/outputs/refactoring/cell_death_evaluation_SAE"
mkdir -p "${SAE_RESULTS_DIR}"

for CACHE_OUT in $CACHE_FILES; do
    # Regex to extract CNN_SEED, D_SAE, and LAMBDA from the folder structure
    if [[ "$CACHE_OUT" =~ MoCo_seed([0-9]+)_refactoring/SAE_refactoring_dim([0-9]+)_lambda([0-9]+) ]]; then
        CNN_SEED="${BASH_REMATCH[1]}"
        D_SAE="${BASH_REMATCH[2]}"
        LAMBDA="${BASH_REMATCH[3]}"
        
        echo "▶️ Running Apoptosis Prediction for CNN_SEED=${CNN_SEED}, d_sae=${D_SAE}, lambda=${LAMBDA}..."
        python -m cell_death.apoptosis_r2_test_clean \
            --features_cache "$CACHE_OUT" \
            --cell_death_csv "$CELL_DEATH_CSV" \
            --model "ridge" \
            --pca_dim 250 \
            --gap_l2_norm \
            --seed 42 \
            --cv_folds 5 \
            --n_permutations 1 \
            --output_dir "${SAE_RESULTS_DIR}" \
            --config_name "SAE_dim${D_SAE}_lam${LAMBDA}" \
            --seed_name "$CNN_SEED" \
            --layer_name "stage5_out" \
            --csv_out "${SAE_RESULTS_DIR}/aggregated_r2_l2effect_folds.csv"
    fi
done

echo "🏁 Finished SAE Apoptosis Prediction!"

echo ""
echo "=========================================================================="
echo " 2. CNN vs SAE Paired Slope Plot (Global R²)"
echo "=========================================================================="
# NOTE: This plot requires CNN and SAE evaluation results to exist.
CNN_RESULTS_DIR="${BASE}/outputs/refactoring/cell_death_evaluation"
SLOPE_PLOT_OUT="${BASE}/outputs/refactoring/CNN_vs_SAE_Plots"
mkdir -p "${SLOPE_PLOT_OUT}"

echo "▶️ Generating Slope Plot..."
python -m cell_death.plot_cnn_vs_sae_slope_plot \
    --cnn_results_dir "${CNN_RESULTS_DIR}" \
    --sae_results_dir "${SAE_RESULTS_DIR}" \
    --cnn_config "MoCo_l2norm" \
    --cnn_layer "stage5_out" \
    --sae_l2norm "l2norm" \
    --sae_config "SAE_dim8192_lam800" \
    --output_dir "${SLOPE_PLOT_OUT}" || echo "⚠️ Warning: Could not generate CNN vs SAE plot (SAE results might be missing or arguments mismatch)"

echo "=== Slope Plot execution complete ==="

echo "=========================================================================="
echo " 3. SAE UMAP Visualization (d=8192, lambda=800)"
echo "=========================================================================="
UMAP_OUT_DIR="${BASE}/outputs/refactoring/UMAP_Plots"
mkdir -p "${UMAP_OUT_DIR}"

echo "▶️ Generating UMAPs for dim8192_lambda800..."
for CACHE_OUT in $CACHE_FILES; do
    if [[ "$CACHE_OUT" =~ SAE_refactoring_dim8192_lambda800 ]]; then
        if [[ "$CACHE_OUT" =~ MoCo_seed([0-9]+)_refactoring ]]; then
            CNN_SEED="${BASH_REMATCH[1]}"
            
            echo "  - Running UMAP for Seed ${CNN_SEED}..."
            python -m cell_death.SAE_cell_death_UMAP \
                --sae_cache "$CACHE_OUT" \
                --cell_death_csv "$CELL_DEATH_CSV" \
                --classes SNCA LRRK2 GBA \
                --umap_n_neighbors 15 \
                --umap_min_dist 0.2 \
                --umap_metric cosine \
                --dot_size 7 \
                --alpha 0.7 \
                --gamma 1.4 \
                --vmax_pctl 90 \
                --cmap "YlGnBu" \
                --cmap_start 0.15 \
                --gap_l2_norm \
                --output_dir "${UMAP_OUT_DIR}/seed${CNN_SEED}"
        fi
    fi
done
echo "🏁 UMAP Generation Complete!"
