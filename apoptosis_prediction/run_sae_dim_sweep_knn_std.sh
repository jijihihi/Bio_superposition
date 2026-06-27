#!/bin/bash
# ==============================================================================
# Local Linearity Sweep: K-Neighbors & SAE Dimension (Multi-Seed)
#
# 여러 시드(8개)에 대해 K=(5,10,15,20,25), SAE 차원별로
# local_knn_std.py를 실행하고, 시드 평균/표준편차를 포함한 최종 추세선을 그립니다.
# ==============================================================================
set -e

BASE="/content/drive/MyDrive/Final_paper/lambda_labs_moco_only"
cell_death_CSV="${BASE}/세포이미지별 사멸율/이미지별_세포사멸율_7200.csv"
OUT_BASE="${BASE}/local_linearity_dim_and_k_sweep"

SEEDS=(42 87 95 123 124 256 445 457)
DIMS=(1024 2048 4096 8192)
K_NEIGHBORS="5 10 15 20 25"

echo "=================================================================="
echo "🚀 Running local_knn_std for multiple seeds & dimensions & k-sweep"
echo "📂 Outputs will be saved to: $OUT_BASE"
echo "=================================================================="

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "=================================================================="
    echo "🌱 Processing Seed: $SEED"
    echo "=================================================================="
    
    # CNN Cache
    CNN_CACHE="${BASE}/caches_per_image_centering/CNN_seed${SEED}/cnn_gap_stage5_out_withnewclass.npz"
    if [ ! -f "$CNN_CACHE" ]; then
        # 혹시 몰라 예전 경로도 체크
        CNN_CACHE="${BASE}/MoCo_seed${SEED}/CNN_GAP/cnn_gap_stage5_out_all.npz"
    fi

    for D in "${DIMS[@]}"; do
        echo "▶️ Processing SAE Dimension: $D"
        
        SAE_CACHE="${BASE}/caches_per_image_centering/CNN_seed${SEED}_SAE/sae_gap_d${D}_lam800_normrestored_withnewclass.npz"

        if [ ! -f "$SAE_CACHE" ] || [ ! -f "$CNN_CACHE" ]; then
            echo "⚠️ Cache not found for Seed=$SEED, dim=$D. Skipping..."
            continue
        fi

        OUT_DIR="${OUT_BASE}/seed_${SEED}/d${D}"
        mkdir -p "$OUT_DIR"

        python -m cell_death_prediction.local_knn_std \
            --cnn_cache "$CNN_CACHE" \
            --sae_cache "$SAE_CACHE" \
            --cell_death_csv "$cell_death_CSV" \
            --k_neighbors $K_NEIGHBORS \
            --dead_threshold 1e-5 \
            --filter_mode none \
            --pca_dim 0 \
            --output_dir "$OUT_DIR" \
            --n_permutations 0
            
        echo "✅ Completed Seed=$SEED, dim=$D"
    done
done

# ==============================================================================
# Plotting Script Execution
# ==============================================================================

echo "=================================================================="
echo "📈 Generating Multi-Seed K-Sweep Trend Plots (.svg)..."
python -m cell_death_prediction.plot_trend_sweep_multiseed
echo "✅ All complete! Check the plots in ${OUT_BASE}/plots"
