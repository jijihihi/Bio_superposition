#!/bin/bash
# ==============================================================================
# Effective Rank Sweep — CNN vs SAE 정보 풍부도 비교
#
# 4가지 조건:
#   1) Raw: no PCA, no filter, no norm → 원본 공간 내재적 차원
#   2) PCA 50: no filter, no norm → 같은 차원에서 분산 균등성
#   3) PCA 50 + std norm → 정규화 후에도 유지되는지
#   4) PCA 50 + DE sweep → 필터링에 대한 robustness
#
# CNN seed와 SAE seed는 독립적이므로 각각 별도 루프로 sweep.
# ==============================================================================
set -e

# ──────────────────────────────────────────────────────────────
# PATHS — 여기만 수정
# ──────────────────────────────────────────────────────────────
CNN_CACHE_TEMPLATE="/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/MoCo_seed{CNN_SEED}/CNN_GAP/cnn_gap_stage5_out_all.npz"
SAE_CACHE_TEMPLATE="/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/MoCo_seed87/SAE_seed{SAE_SEED}_no_L2norm_loss/features_cache_stage5_out_normrestored_all_no_SAE_GAP_L2_norm_again_d8192_sp800.npz"
BASE_OUT="/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/effective_rank"

DEAD_THRESHOLD="5e-5"

# Model Seeds — CNN과 SAE 독립
CNN_SEEDS="42 87 95 124"
SAE_SEEDS="48 123 777 856"

RANDOM_SEED=856

# Helper functions
get_cnn_cache() {
    local cnn_seed=$1
    echo "${CNN_CACHE_TEMPLATE//\{CNN_SEED\}/$cnn_seed}"
}
get_sae_cache() {
    local cnn_seed=$1
    local sae_seed=$2
    local path="${SAE_CACHE_TEMPLATE//\{CNN_SEED\}/$cnn_seed}"
    echo "${path//\{SAE_SEED\}/$sae_seed}"
}



# !python -m apoptosis_prediction.local_vs_global_ridge \
#     --cnn_cache "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/MoCo_seed87/CNN_GAP/cnn_gap_stage5_out_all.npz" \
#     --sae_cache "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/MoCo_seed87/SAE_seed856_no_L2norm_loss/features_cache_stage5_out_normrestored_all_no_SAE_GAP_L2_norm_again_d8192_sp800.npz" \
#     --apoptosis_csv "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/세포이미지별 사멸율/이미지별_세포사멸율_7200.csv" \
#     --k_neighbors 20 \
#     --gap_l2_norm \
#     --dead_threshold 5e-5 \
#     --filter_mode cv de \
#     --min_cv 0.1 \
#     --de_min_log2fc 1.0 \
#     --de_mode union \
#     --de_eval_split 0.5 \
#     --norm log_std \
#     --samples_per_class 5000 \
#     --seed 856 \
#     --n_permutations 0 \
#     --output_dir "/content/local_vs_global_ridge" \
#     --pca_dim 15 \
#     --min_local_n 14


# ──────────────────────────────────────────────────────────────
# CONDITION 1: Raw — no PCA, no filter, no norm
#   → 원본 공간의 내재적 차원 (CNN 512D vs SAE ~alive D)
# ──────────────────────────────────────────────────────────────
run_raw() {
    echo "══════════════════════════════════════════════"
    echo "  CONDITION 1: Raw (no PCA, no filter, no norm)"
    echo "══════════════════════════════════════════════"

    for CNN_SEED in $CNN_SEEDS; do
        CNN_CACHE=$(get_cnn_cache $CNN_SEED)
        OUT="$BASE_OUT/raw/cnn_seed_${CNN_SEED}"

        echo "── CNN: cnn_seed=$CNN_SEED ──"
        python -m apoptosis_prediction.effective_rank \
            --cnn_cache "$CNN_CACHE" \
            --gap_l2_norm \
            --dead_threshold $DEAD_THRESHOLD \
            --filter_mode none \
            --pca_dim 0 \
            --norm none \
            --samples_per_class 5000 \
            --seed $RANDOM_SEED \
            --output_dir "${OUT}/cnn"
    done

    for CNN_SEED in $CNN_SEEDS; do
        for SAE_SEED in $SAE_SEEDS; do
            SAE_CACHE=$(get_sae_cache $CNN_SEED $SAE_SEED)
            OUT="$BASE_OUT/raw/cnn_seed_${CNN_SEED}/sae_seed_${SAE_SEED}"

            echo "── SAE: cnn_seed=$CNN_SEED, sae_seed=$SAE_SEED ──"
            python -m apoptosis_prediction.effective_rank \
                --sae_cache "$SAE_CACHE" \
                --dead_threshold $DEAD_THRESHOLD \
                --pca_dim 0 \
                --filter_mode none \
                --samples_per_class 5000 \
                --norm none \
                --seed $RANDOM_SEED \
                --output_dir "${OUT}/sae"
        done
    done
}


# ──────────────────────────────────────────────────────────────
# CONDITION 2: PCA 50 only — no filter, no norm
#   → 같은 차원(50D)에서 분산 균등성 비교 (공정 비교 핵심)
# ──────────────────────────────────────────────────────────────
run_pca_only() {
    echo "══════════════════════════════════════════════"
    echo "  CONDITION 2: PCA 250 (no filter, no norm)"
    echo "══════════════════════════════════════════════"

    for CNN_SEED in $CNN_SEEDS; do
        CNN_CACHE=$(get_cnn_cache $CNN_SEED)
        OUT="$BASE_OUT/pca50/cnn_seed_${CNN_SEED}"

        echo "── CNN: cnn_seed=$CNN_SEED ──"
        python -m apoptosis_prediction.effective_rank \
            --cnn_cache "$CNN_CACHE" \
            --gap_l2_norm \
            --dead_threshold $DEAD_THRESHOLD \
            --pca_dim 250 \
            --filter_mode none \
            --norm none \
            --samples_per_class 5000 \
            --seed $RANDOM_SEED \
            --output_dir "${OUT}/cnn"
    done

    for CNN_SEED in $CNN_SEEDS; do
        for SAE_SEED in $SAE_SEEDS; do
            SAE_CACHE=$(get_sae_cache $CNN_SEED $SAE_SEED)
            OUT="$BASE_OUT/pca50/cnn_seed_${CNN_SEED}/sae_seed_${SAE_SEED}"

            echo "── SAE: cnn_seed=$CNN_SEED, sae_seed=$SAE_SEED ──"
            python -m apoptosis_prediction.effective_rank \
                --sae_cache "$SAE_CACHE" \
                --dead_threshold $DEAD_THRESHOLD \
                --pca_dim 250 \
                --filter_mode none \
                --norm none \
                --samples_per_class 5000 \
                --seed $RANDOM_SEED \
                --output_dir "${OUT}/sae"
        done
    done
}


# ──────────────────────────────────────────────────────────────
# CONDITION 3: PCA 50 + std norm — 정규화 후에도 유지되는지
# ──────────────────────────────────────────────────────────────
run_pca_std() {
    echo "══════════════════════════════════════════════"
    echo "  CONDITION 3: PCA 250 + std norm"
    echo "══════════════════════════════════════════════"

    for CNN_SEED in $CNN_SEEDS; do
        CNN_CACHE=$(get_cnn_cache $CNN_SEED)
        OUT="$BASE_OUT/pca50_std/cnn_seed_${CNN_SEED}"

        echo "── CNN: cnn_seed=$CNN_SEED ──"
        python -m apoptosis_prediction.effective_rank \
            --cnn_cache "$CNN_CACHE" \
            --gap_l2_norm \
            --dead_threshold $DEAD_THRESHOLD \
            --pca_dim 250 \
            --norm std \
            --filter_mode none \
            --samples_per_class 5000 \
            --seed $RANDOM_SEED \
            --output_dir "${OUT}/cnn"
    done

    for CNN_SEED in $CNN_SEEDS; do
        for SAE_SEED in $SAE_SEEDS; do
            SAE_CACHE=$(get_sae_cache $CNN_SEED $SAE_SEED)
            OUT="$BASE_OUT/pca50_std/cnn_seed_${CNN_SEED}/sae_seed_${SAE_SEED}"

            echo "── SAE: cnn_seed=$CNN_SEED, sae_seed=$SAE_SEED ──"
            python -m apoptosis_prediction.effective_rank \
                --sae_cache "$SAE_CACHE" \
                --dead_threshold $DEAD_THRESHOLD \
                --pca_dim 250 \
                --norm std \
                --filter_mode none \
                --samples_per_class 5000 \
                --seed $RANDOM_SEED \
                --output_dir "${OUT}/sae"
        done
    done
}


# ──────────────────────────────────────────────────────────────
# CONDITION 4: PCA 50 + DE log2FC sweep
#   → 필터링 강도에 대한 robustness (CNN erank ↓ vs SAE 안정)
# ──────────────────────────────────────────────────────────────
run_de_sweep() {
    echo "══════════════════════════════════════════════"
    echo "  CONDITION 4: PCA 250 + DE log2FC sweep"
    echo "══════════════════════════════════════════════"

    DE_LOG2FC_VALUES="0.0 0.5 1.0 1.5 2.0"

    # --- CNN seed sweep ---
    for CNN_SEED in $CNN_SEEDS; do
        CNN_CACHE=$(get_cnn_cache $CNN_SEED)

        for LOG2FC in $DE_LOG2FC_VALUES; do
            OUT="$BASE_OUT/de_sweep/cnn_seed_${CNN_SEED}/log2fc_${LOG2FC}"

            echo "── CNN: cnn_seed=$CNN_SEED, DE log2fc=$LOG2FC ──"
            python -m apoptosis_prediction.effective_rank \
                --cnn_cache "$CNN_CACHE" \
                --gap_l2_norm \
                --dead_threshold $DEAD_THRESHOLD \
                --pca_dim 250 \
                --filter_mode cv de \
                --min_cv 0.1 \
                --de_min_log2fc $LOG2FC \
                --de_mode union \
                --norm none \
                --de_eval_split 0.5 \
                --samples_per_class 5000 \
                --seed $RANDOM_SEED \
                --output_dir "${OUT}/cnn"
        done
    done

    # --- SAE seed sweep ---
    for CNN_SEED in $CNN_SEEDS; do
        for SAE_SEED in $SAE_SEEDS; do
            SAE_CACHE=$(get_sae_cache $CNN_SEED $SAE_SEED)

            for LOG2FC in $DE_LOG2FC_VALUES; do
                OUT="$BASE_OUT/de_sweep/cnn_seed_${CNN_SEED}/sae_seed_${SAE_SEED}/log2fc_${LOG2FC}"

                echo "── SAE: cnn_seed=$CNN_SEED, sae_seed=$SAE_SEED, DE log2fc=$LOG2FC ──"
                python -m apoptosis_prediction.effective_rank \
                    --sae_cache "$SAE_CACHE" \
                    --dead_threshold $DEAD_THRESHOLD \
                    --pca_dim 250 \
                    --filter_mode cv de \
                    --min_cv 0.1 \
                    --de_min_log2fc $LOG2FC \
                    --de_mode union \
                    --norm none \
                    --de_eval_split 0.5 \
                    --samples_per_class 5000 \
                    --seed $RANDOM_SEED \
                    --output_dir "${OUT}/sae"
            done
        done
    done
}


# ──────────────────────────────────────────────────────────────
# Run selected experiment(s)
# Usage:
#   bash run_erank_sweep.sh raw         # Condition 1
#   bash run_erank_sweep.sh pca         # Condition 2
#   bash run_erank_sweep.sh pca_std     # Condition 3
#   bash run_erank_sweep.sh de          # Condition 4
#   bash run_erank_sweep.sh all         # All conditions
# ──────────────────────────────────────────────────────────────
case "${1:-all}" in
    raw)        run_raw ;;
    pca)        run_pca_only ;;
    pca_std)    run_pca_std ;;
    de)         run_de_sweep ;;
    all)
        run_raw
        run_pca_only
        run_pca_std
        run_de_sweep
        ;;
    *)
        echo "Usage: bash $0 {raw|pca|pca_std|de|all}"
        exit 1
        ;;
esac

echo ""
echo "════════════════════════════════════════════════"
echo "  All erank experiments complete!"
echo "  Results in: $BASE_OUT"
echo "════════════════════════════════════════════════"
