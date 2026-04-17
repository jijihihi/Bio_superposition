#!/bin/bash
# ==============================================================================
# Local Linearity Sweep — KNN Std + Local Ridge
#
# 1) Raw CNN vs SAE (no filter, no norm) — local linearity 기본 비교
# 2) DE strength sweep on CNN — 중첩 제거가 linearity 회복하는지 검증
# 3) DPT 동일 조건 (filter + log_std + PCA) — 최종 비교
#
# CNN seed와 SAE seed는 독립적이므로 각각 별도 루프로 sweep.
# ==============================================================================
set -e

# ──────────────────────────────────────────────────────────────
# PATHS — 여기만 수정
# ──────────────────────────────────────────────────────────────
CNN_CACHE_TEMPLATE="/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/MoCo_seed{CNN_SEED}/CNN_GAP/cnn_gap_stage5_out_all.npz"
SAE_CACHE_TEMPLATE="/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/MoCo_seed87/SAE_seed{SAE_SEED}_no_L2norm_loss/features_cache_stage5_out_normrestored_all_no_SAE_GAP_L2_norm_again_d8192_sp800.npz"
APOPTOSIS_CSV="/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/세포이미지별 사멸율/이미지별_세포사멸율_7200.csv"
BASE_OUT="/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/local_linearity"

# K neighbors — KNN std vs Ridge 분리
K_NEIGHBORS_KNN="5 10 15 20 25"
K_NEIGHBORS_RIDGE="25 30 35 50 80 120 150 200"

DEAD_THRESHOLD="1e-5"

# Model Seeds — CNN과 SAE 독립
CNN_SEEDS="42 87 95 124"
SAE_SEEDS="48 123 777 856"

# Fixed random seed for reproducibility (subsampling, splits 등)
RANDOM_SEED=42

# Helper: CNN cache path 생성
get_cnn_cache() {
    local cnn_seed=$1
    echo "${CNN_CACHE_TEMPLATE//\{CNN_SEED\}/$cnn_seed}"
}

# Helper: SAE cache path 생성 (SAE는 항상 MoCo_seed87 위에 고정 — CNN seed와 무관)
get_sae_cache() {
    local sae_seed=$1
    echo "${SAE_CACHE_TEMPLATE//\{SAE_SEED\}/$sae_seed}"
}



# !python -m apoptosis_prediction.local_linearity_knn \
#     --cnn_cache "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/MoCo_seed87/CNN_GAP/cnn_gap_stage5_out_all.npz" \
#     --sae_cache "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/MoCo_seed87/SAE_seed856_no_L2norm_loss/features_cache_stage5_out_normrestored_all_no_SAE_GAP_L2_norm_again_d8192_sp800.npz" \
#     --apoptosis_csv "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/세포이미지별 사멸율/이미지별_세포사멸율_7200.csv" \
#     --k_neighbors 5 10 15 \
#     --n_permutations 0 \
#     --filter none \
#     --gap_l2_norm \
#     --pca_dim 0 \
#     --dead_threshold 1e-5 \
#     --output_dir "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/local_linearity/original_no_filter_no_norm_no_PCA/cnn_87_sae_856"


# ──────────────────────────────────────────────────────────────
# EXPERIMENT 1: Raw — no filter, no norm, no PCA
#   CNN seed sweep → KNN std + local ridge
#   SAE seed sweep (각 CNN seed 위에) → KNN std + local ridge
#   → local linearity 기본 비교
# ──────────────────────────────────────────────────────────────
run_raw() {
    echo "══════════════════════════════════════════════"
    echo "  EXPERIMENT 1: Raw (no filter, no norm)"
    echo "══════════════════════════════════════════════"

    # --- CNN seed sweep ---
    for CNN_SEED in $CNN_SEEDS; do
        CNN_CACHE=$(get_cnn_cache $CNN_SEED)
        OUT="$BASE_OUT/raw/cnn_seed_${CNN_SEED}"

        echo "── KNN Std: CNN (cnn_seed=$CNN_SEED) ──"
        python -m apoptosis_prediction.local_linearity_knn \
            --cnn_cache "$CNN_CACHE" \
            --apoptosis_csv "$APOPTOSIS_CSV" \
            --k_neighbors $K_NEIGHBORS_KNN \
            --gap_l2_norm \
            --dead_threshold $DEAD_THRESHOLD \
            --n_permutations 0 \
            --filter_mode none \
            --pca_dim 0 \
            --seed $RANDOM_SEED \
            --output_dir "${OUT}/knn_std_cnn"

    #     echo "── Local Ridge: CNN (cnn_seed=$CNN_SEED) ──"
    #     python -m apoptosis_prediction.local_vs_global_ridge \
    #         --cnn_cache "$CNN_CACHE" \
    #         --apoptosis_csv "$APOPTOSIS_CSV" \
    #         --k_neighbors $K_NEIGHBORS_RIDGE \
    #         --gap_l2_norm \
    #         --dead_threshold $DEAD_THRESHOLD \
    #         --n_permutations 0 \
    #         --filter "none" \
    #         --pca_dim 5 \
    #         --seed $RANDOM_SEED \
    #         --samples_per_class 20000 \
    #         --output_dir "${OUT}/ridge_cnn"
    done


# !python -m apoptosis_prediction.local_linearity_knn \
#     --cnn_cache "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/MoCo_seed87/CNN_GAP/cnn_gap_stage5_out_all.npz" \
#     --sae_cache "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/MoCo_seed87/SAE_seed856_no_L2norm_loss/features_cache_stage5_out_normrestored_all_no_SAE_GAP_L2_norm_again_d8192_sp800.npz" \
#     --apoptosis_csv "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/세포이미지별 사멸율/이미지별_세포사멸율_7200.csv" \
#     --k_neighbors 5 10 15 \
#     --n_permutations 0 \
#     --filter none \
#     --gap_l2_norm \
#     --pca_dim 0 \
#     --dead_threshold 1e-5 \
#     --output_dir "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/local_linearity/original_no_filter_no_n


    # --- SAE seed sweep (SAE는 MoCo_seed87 고정, SAE seed만 sweep) ---
    for SAE_SEED in $SAE_SEEDS; do
        SAE_CACHE=$(get_sae_cache $SAE_SEED)
        OUT="$BASE_OUT/raw/sae_seed_${SAE_SEED}"

        echo "── KNN Std: SAE (sae_seed=$SAE_SEED) ──"
        python -m apoptosis_prediction.local_linearity_knn \
            --sae_cache "$SAE_CACHE" \
            --apoptosis_csv "$APOPTOSIS_CSV" \
            --k_neighbors $K_NEIGHBORS_KNN \
            --dead_threshold $DEAD_THRESHOLD \
            --gap_l2_norm \
            --filter_mode none \
            --pca_dim 0 \
            --n_permutations 0 \
            --seed $RANDOM_SEED \
            --output_dir "${OUT}/knn_std_sae"

    #     echo "── Local Ridge: SAE (sae_seed=$SAE_SEED) ──"
    #     python -m apoptosis_prediction.local_vs_global_ridge \
    #         --sae_cache "$SAE_CACHE" \
    #         --apoptosis_csv "$APOPTOSIS_CSV" \
    #         --k_neighbors $K_NEIGHBORS_RIDGE \
    #         --dead_threshold $DEAD_THRESHOLD \
    #         --gap_l2_norm \
    #         --filter "none" \
    #         --pca_dim 5 \
    #         --n_permutations 0 \
    #         --seed $RANDOM_SEED \
    #         --samples_per_class 20000 \
    #         --output_dir "${OUT}/ridge_sae"
    done
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
# EXPERIMENT 2: DE strength sweep on CNN
#   log2fc 기준을 강화할수록 CNN local linearity가 회복되는지
#   → "중첩 제거 = local linearity 회복" 증거
# ──────────────────────────────────────────────────────────────
run_de_sweep() {
    echo "══════════════════════════════════════════════"
    echo "  EXPERIMENT 2: DE strength sweep (CNN only)"
    echo "══════════════════════════════════════════════"

    DE_LOG2FC_VALUES="0.0 0.2 0.5 0.8 1.0"

    for CNN_SEED in $CNN_SEEDS; do
        CNN_CACHE=$(get_cnn_cache $CNN_SEED)

        for LOG2FC in $DE_LOG2FC_VALUES; do
            OUT="$BASE_OUT/de_sweep/cnn_seed_${CNN_SEED}/log2fc_${LOG2FC}"

            echo "── CNN: cnn_seed=$CNN_SEED, DE log2fc=$LOG2FC ──"

            # KNN Std
            python -m apoptosis_prediction.local_linearity_knn \
                --cnn_cache "$CNN_CACHE" \
                --apoptosis_csv "$APOPTOSIS_CSV" \
                --k_neighbors $K_NEIGHBORS_KNN \
                --gap_l2_norm \
                --dead_threshold $DEAD_THRESHOLD \
                --filter_mode cv de \
                --min_cv 0.1 \
                --de_min_log2fc $LOG2FC \
                --de_mode union \
                --de_eval_split 0.5 \
                --n_permutations 0 \
                --norm "log_std" \
                --seed $RANDOM_SEED \
                --output_dir "${OUT}/knn_log_std_cnn_test"

            # Local Ridge
    #         python -m apoptosis_prediction.local_vs_global_ridge \
    #             --cnn_cache "$CNN_CACHE" \
    #             --apoptosis_csv "$APOPTOSIS_CSV" \
    #             --k_neighbors $K_NEIGHBORS_RIDGE \
    #             --gap_l2_norm \
    #             --dead_threshold $DEAD_THRESHOLD \
    #             --filter_mode cv de \
    #             --min_cv 0.1 \
    #             --de_min_log2fc $LOG2FC \
    #             --de_mode union \
    #             --de_eval_split 0.5 \
    #             --pca_dim 5 \
    #             --samples_per_class 20000 \
    #             --seed $RANDOM_SEED \
    #             --output_dir "${OUT}/ridge_cnn"
        done
    done
}



#  "--n_neighbors", "35",
#  "--pca_dim", "15",
#  "--filter_mode", "cv", "de",
#    "--min_cv", "0.1",
#     "--de_adj_p", "0.05",
#      "--de_min_log2fc", "1.0",
#      "--dead_threshold", "5e-5",
#      "--root_mode", "diffmap",
#     "--root_perturbation_n", "10",
#     "--dpt_scope", "ctrl_mut_pair",
#     "--norm", "log_std",
#     "--n_diffmap_comps", "10",
#     "--de_eval_split", "0.5",
#     "--gam_splines", "8",
#     "--gam_trim_pctl", "5", "95",
#      "--gap_l2_norm",
#      "--seed", "856",


# ──────────────────────────────────────────────────────────────
# EXPERIMENT 3: DPT-identical conditions
#   filter + log_std + PCA — 최종 fair comparison
# ──────────────────────────────────────────────────────────────
run_dpt_matched() {
    echo "══════════════════════════════════════════════"
    echo "  EXPERIMENT 3: DPT-identical (filter+log_std+PCA)"
    echo "══════════════════════════════════════════════"

    DPT_LOG2FC_VALUES="0.0 0.2 0.3 0.5 1.0"

    # Common filter (WITHOUT de_min_log2fc — that goes in the loop)
    COMMON_BASE="--filter_mode cv de \
        --min_cv 0.1 \
        --de_mode union \
        --de_eval_split 0.5 \
        --norm log_std \
        --pca_dim 15 \
        --samples_per_class 5000"

    # --- CNN seed sweep ---
    for CNN_SEED in $CNN_SEEDS; do
        CNN_CACHE=$(get_cnn_cache $CNN_SEED)

        for LOG2FC in $DPT_LOG2FC_VALUES; do
            OUT="$BASE_OUT/dpt_matched/cnn_seed_${CNN_SEED}/log2fc_${LOG2FC}"

            echo "── KNN Std: CNN (cnn_seed=$CNN_SEED, log2fc=$LOG2FC) ──"
            python -m apoptosis_prediction.local_linearity_knn \
                --cnn_cache "$CNN_CACHE" \
                --apoptosis_csv "$APOPTOSIS_CSV" \
                --k_neighbors $K_NEIGHBORS_KNN \
                --gap_l2_norm \
                --dead_threshold 5e-5 \
                --de_adj_p 0.05 \
                $COMMON_BASE \
                --de_min_log2fc $LOG2FC \
                --n_permutations 0 \
                --seed $RANDOM_SEED \
                --output_dir "${OUT}/knn_std_cnn"

            # echo "── Local Ridge: CNN (cnn_seed=$CNN_SEED, log2fc=$LOG2FC) ──"
            # python -m apoptosis_prediction.local_vs_global_ridge \
            #     --cnn_cache "$CNN_CACHE" \
            #     --apoptosis_csv "$APOPTOSIS_CSV" \
            #     --k_neighbors $K_NEIGHBORS_RIDGE \
            #     --gap_l2_norm \
            #     --dead_threshold 5e-5 \
            #     $COMMON_BASE \
            #     --de_min_log2fc $LOG2FC \
            #     --n_permutations 0 \
            #     --seed $RANDOM_SEED \
            #     --output_dir "${OUT}/ridge_cnn"
        done
    done

    # --- SAE seed sweep (SAE는 MoCo_seed87 고정, SAE seed만 sweep) ---
    for SAE_SEED in $SAE_SEEDS; do
        SAE_CACHE=$(get_sae_cache $SAE_SEED)

        for LOG2FC in $DPT_LOG2FC_VALUES; do
            OUT="$BASE_OUT/dpt_matched/sae_seed_${SAE_SEED}/log2fc_${LOG2FC}"

            echo "── KNN Std: SAE (sae_seed=$SAE_SEED, log2fc=$LOG2FC) ──"
            python -m apoptosis_prediction.local_linearity_knn \
                --sae_cache "$SAE_CACHE" \
                --apoptosis_csv "$APOPTOSIS_CSV" \
                --k_neighbors $K_NEIGHBORS_KNN \
                --dead_threshold 5e-5 \
                --de_adj_p 0.05 \
                $COMMON_BASE \
                --de_min_log2fc $LOG2FC \
                --n_permutations 0 \
                --seed $RANDOM_SEED \
                --output_dir "${OUT}/knn_std_sae"

            # echo "── Local Ridge: SAE (sae_seed=$SAE_SEED, log2fc=$LOG2FC) ──"
            # python -m apoptosis_prediction.local_vs_global_ridge \
            #     --sae_cache "$SAE_CACHE" \
            #     --apoptosis_csv "$APOPTOSIS_CSV" \
            #     --k_neighbors $K_NEIGHBORS_RIDGE \
            #     --dead_threshold 5e-5 \
            #     $COMMON_BASE \
            #     --de_min_log2fc $LOG2FC \
            #     --n_permutations 0 \
            #     --seed $RANDOM_SEED \
            #     --output_dir "${OUT}/ridge_sae"
        done
    done
}

# ──────────────────────────────────────────────────────────────
# Run selected experiment(s)
# Usage:
#   bash run_local_linearity_sweep.sh raw        # Experiment 1
#   bash run_local_linearity_sweep.sh de_sweep   # Experiment 2
#   bash run_local_linearity_sweep.sh dpt        # Experiment 3
#   bash run_local_linearity_sweep.sh all        # All experiments
# ──────────────────────────────────────────────────────────────
case "${1:-all}" in
    raw)        run_raw ;;
    de_sweep)   run_de_sweep ;;
    dpt)        run_dpt_matched ;;
    all)
        run_raw
        run_de_sweep
        run_dpt_matched
        ;;
    *)
        echo "Usage: bash $0 {raw|de_sweep|dpt|all}"
        exit 1
        ;;
esac

echo ""
echo "════════════════════════════════════════════════"
echo "  All experiments complete!"
echo "  Results in: $BASE_OUT"
echo "════════════════════════════════════════════════"
