#!/bin/bash
# ==============================================================================
# PHATE Visualization for CNN and SAE Caches
# 
# Usage:
#   bash phate_visualize/run_phate_cnn_sae.sh
# ==============================================================================
set -e

# ============================
# 1. User Configurations
# ============================
BASE_DIR="/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/caches_CNN_SAE_class_27000_withnewclass"
CNN_CACHE="${BASE_DIR}/CNN_seed95/cnn_gap_stage5_out_all.npz"
SAE_CACHE="${BASE_DIR}/CNN_seed95_SAE/sae_gap_d8192_lam800_normrestored_withnewclass.npz"

OUTPUT_ROOT="/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/phate_results"
mkdir -p "$OUTPUT_ROOT"

# 클래스 지정 (예: 0 1 2 8 또는 Control GBA 등, 캐시에 저장된 y 값의 형식에 맞게 입력)
# 빈 문자열("")로 두면 전체 클래스를 사용합니다.
CLASSES="0 1 2 3"

# 클래스당 샘플 개수
SAMPLES_PER_CLASS=5000

# SAE Dead Neuron Threshold
DEAD_THRESHOLD="1e-5"

# ============================
# 2. Run CNN PHATE
# ============================
# echo "========================================================"
# echo "▶️  Running PHATE for CNN Cache"
# echo "========================================================"
# CNN_OUT_DIR="${OUTPUT_ROOT}/CNN"
# mkdir -p "$CNN_OUT_DIR"

# python -m phate_visualize.phate \
#     --features_cache "$CNN_CACHE" \
#     --output_dir "$CNN_OUT_DIR" \
#     --classes $CLASSES \
#     --samples_per_class $SAMPLES_PER_CLASS \
#     --gap_l2_norm \
#     --knn 5 \
#     --t "auto"

# ============================
# 3. Run SAE PHATE
# ============================
echo ""
echo "========================================================"
echo "▶️  Running PHATE for SAE Cache"
echo "========================================================"
SAE_OUT_DIR="${OUTPUT_ROOT}/SAE"
mkdir -p "$SAE_OUT_DIR"

python -m phate_visualize.phate \
    --features_cache "$SAE_CACHE" \
    --output_dir "$SAE_OUT_DIR" \
    --dead_threshold $DEAD_THRESHOLD \
    --classes $CLASSES \
    --samples_per_class $SAMPLES_PER_CLASS \
    --gap_l2_norm \
    --knn 5 \
    --t "auto" \
    --knn_dist "euclidean" \
    --batch_size 64 \
    --min_cv 0.1 \
    --norm log_std \
    --filter_mode cv de
    --de_adj_p 0.05 \
    --de_min_log2fc 1.0 \
    --decay 100 \
    --paga \
    --paga_n_neighbors 30 \
    --paga_n_pcs 50

echo ""
echo "========================================================"
echo "✅ PHATE visualizations complete!"
echo "CNN results: $CNN_OUT_DIR"
echo "SAE results: $SAE_OUT_DIR"
echo "========================================================"



