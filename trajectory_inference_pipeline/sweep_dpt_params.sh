#!/bin/bash
# DPT Parameter Sweep for multiple seeds, PCA dims, and k-neighbors

SEEDS=(42 87 95 123 124 256 445 457)
PCAS=(30 40 50)
KS=(15 20 25 30 35)

BASE_CACHE_DIR="/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/caches_per_image_centering"
APOP_CSV="/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/세포이미지별 사멸율/이미지별_세포사멸율_7200.csv"
OUT_DIR="${BASE_CACHE_DIR}/DPT_Sweep"

mkdir -p "$OUT_DIR"

echo "Starting DPT parameter sweep..."

for SEED in "${SEEDS[@]}"; do
  CACHE_FILE="${BASE_CACHE_DIR}/CNN_seed${SEED}_SAE/sae_gap_d8192_lam800_normrestored_withnewclass.npz"
  
  if [ ! -f "$CACHE_FILE" ]; then
    echo "Warning: Cache file not found for seed $SEED: $CACHE_FILE"
    continue
  fi

  for PCA in "${PCAS[@]}"; do
    for K in "${KS[@]}"; do
      echo "--------------------------------------------------------"
      echo "Running DPT for SEED=$SEED, PCA=$PCA, K=$K"
      
      python -m trajectory_inference_pipeline.pairwise_dpt \
        --features_cache "$CACHE_FILE" \
        --cell_death_csv "$APOP_CSV" \
        --output_dir "$OUT_DIR" \
        --n_neighbors $K \
        --pca_dim $PCA \
        --seed $SEED \
        --filter_mode "none" \
        --min_cv 0.0 \
        --de_adj_p 1.0 \
        --de_min_log2fc 0.0 \
        --dead_threshold 1e-5 \
        --norm "log_std" \
        --gap_l2_norm \
        --root_mode "diffmap" \
        --gam_splines 8 \
        --gam_trim_pctl 5 95 \
        --de_eval_split 0.5 \
        --gpu \
        --no_plot
        
    done
  done
done

echo "Sweep completed."
echo "Running heatmap generation..."

python -m trajectory_inference_pipeline.plot_dpt_heatmap \
  --input_dir "$OUT_DIR"

echo "All done!"
