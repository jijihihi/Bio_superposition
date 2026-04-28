#!/bin/bash
# ==============================================================================
# Effective Rank Sweep — CNN Layer-wise + SAE 비교
#
# 3 CNN layers (stage5_mid, stage5_out, refine_out) × 8 seeds 한번에 실행.
# SAE는 stage5_out에서만 학습됐으므로, SAE 비교는 stage5_out만.
#
# 4가지 조건:
#   1) Raw: no PCA, no filter, no norm → 원본 공간 내재적 차원
#   2) PCA 250: no filter, no norm → 같은 차원에서 분산 균등성
#   3) PCA 250 + std norm → 정규화 후에도 유지되는지
#   4) PCA 250 + DE sweep → 필터링에 대한 robustness
#
# CNN seed와 SAE seed는 독립적이므로 각각 별도 루프로 sweep.
# ==============================================================================
set -e

# ──────────────────────────────────────────────────────────────
# PATHS — 여기만 수정
# ──────────────────────────────────────────────────────────────
DRIVE_BASE="/content/drive/MyDrive/Final_paper/lambda_labs_moco_only"
BASE_OUT="${DRIVE_BASE}/effective_rank"

# 3 CNN layers to evaluate
CNN_LAYERS="stage5_mid stage5_out refine_out"

# SAE cache template (only for stage5_out — SAE가 stage5_out에서 학습됨)
SAE_CACHE_TEMPLATE="${DRIVE_BASE}/MoCo_seed87/SAE_seed{SAE_SEED}_no_L2norm_loss/features_cache_stage5_out_normrestored_all_no_SAE_GAP_L2_norm_again_d8192_sp800.npz"

DEAD_THRESHOLD="1e-5"

# Model Seeds — CNN과 SAE 독립
CNN_SEEDS="42 87 95 123 124 256 445 457"
SAE_SEEDS="48 123 777 856"

RANDOM_SEED=856

# Helper functions
get_cnn_cache() {
    local cnn_seed=$1
    local layer=$2
    echo "${DRIVE_BASE}/MoCo_seed${cnn_seed}/CNN_GAP/cnn_gap_${layer}_all.npz"
}
get_sae_cache() {
    local sae_seed=$1
    echo "${SAE_CACHE_TEMPLATE//\{SAE_SEED\}/$sae_seed}"
}


# ──────────────────────────────────────────────────────────────
# CONDITION 1: Raw — no PCA, no filter, no norm
#   → 원본 공간의 내재적 차원 (CNN 512D vs SAE ~alive D)
# ──────────────────────────────────────────────────────────────
run_raw() {
    echo "══════════════════════════════════════════════"
    echo "  CONDITION 1: Raw (no PCA, no filter, no norm)"
    echo "══════════════════════════════════════════════"

    # --- CNN: all 3 layers × 8 seeds ---
    for LAYER in $CNN_LAYERS; do
        echo ""
        echo "──── CNN Layer: ${LAYER} ────"
        for CNN_SEED in $CNN_SEEDS; do
            CNN_CACHE=$(get_cnn_cache $CNN_SEED $LAYER)

            if [ ! -f "$CNN_CACHE" ]; then
                echo "⚠️  Not found: $CNN_CACHE"
                continue
            fi

            OUT="$BASE_OUT/${LAYER}/raw/cnn_seed_${CNN_SEED}"

            echo "── CNN: layer=$LAYER, cnn_seed=$CNN_SEED ──"
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
    done

    # --- SAE: stage5_out only (SAE는 CNN seed87에서만 학습됨) ---
    echo ""
    echo "──── SAE (stage5_out, CNN seed87 only) ────"
    for SAE_SEED in $SAE_SEEDS; do
        SAE_CACHE=$(get_sae_cache $SAE_SEED)

        if [ ! -f "$SAE_CACHE" ]; then
            echo "⚠️  Not found: $SAE_CACHE"
            continue
        fi

        OUT="$BASE_OUT/stage5_out/raw/sae_seed_${SAE_SEED}"

        echo "── SAE: sae_seed=$SAE_SEED ──"
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
}


# ──────────────────────────────────────────────────────────────
# CONDITION 2: PCA 250 only — no filter, no norm
#   → 같은 차원(250D)에서 분산 균등성 비교 (공정 비교 핵심)
# ──────────────────────────────────────────────────────────────
run_pca_only() {
    echo "══════════════════════════════════════════════"
    echo "  CONDITION 2: PCA 250 (no filter, no norm)"
    echo "══════════════════════════════════════════════"

    for LAYER in $CNN_LAYERS; do
        echo ""
        echo "──── CNN Layer: ${LAYER} ────"
        for CNN_SEED in $CNN_SEEDS; do
            CNN_CACHE=$(get_cnn_cache $CNN_SEED $LAYER)

            if [ ! -f "$CNN_CACHE" ]; then
                echo "⚠️  Not found: $CNN_CACHE"
                continue
            fi

            OUT="$BASE_OUT/${LAYER}/pca50/cnn_seed_${CNN_SEED}"

            echo "── CNN: layer=$LAYER, cnn_seed=$CNN_SEED ──"
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
    done

    # SAE: stage5_out only (CNN seed87에서만 학습됨)
    echo ""
    echo "──── SAE (stage5_out, CNN seed87 only) ────"
    for SAE_SEED in $SAE_SEEDS; do
        SAE_CACHE=$(get_sae_cache $SAE_SEED)

        if [ ! -f "$SAE_CACHE" ]; then
            echo "⚠️  Not found: $SAE_CACHE"
            continue
        fi

        OUT="$BASE_OUT/stage5_out/pca50/sae_seed_${SAE_SEED}"

        echo "── SAE: sae_seed=$SAE_SEED ──"
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
}


# ──────────────────────────────────────────────────────────────
# CONDITION 3: PCA 250 + std norm — 정규화 후에도 유지되는지
# ──────────────────────────────────────────────────────────────
run_pca_std() {
    echo "══════════════════════════════════════════════"
    echo "  CONDITION 3: PCA 250 + std norm"
    echo "══════════════════════════════════════════════"

    for LAYER in $CNN_LAYERS; do
        echo ""
        echo "──── CNN Layer: ${LAYER} ────"
        for CNN_SEED in $CNN_SEEDS; do
            CNN_CACHE=$(get_cnn_cache $CNN_SEED $LAYER)

            if [ ! -f "$CNN_CACHE" ]; then
                echo "⚠️  Not found: $CNN_CACHE"
                continue
            fi

            OUT="$BASE_OUT/${LAYER}/pca50_std/cnn_seed_${CNN_SEED}"

            echo "── CNN: layer=$LAYER, cnn_seed=$CNN_SEED ──"
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
    done

    # SAE: stage5_out only (CNN seed87에서만 학습됨)
    echo ""
    echo "──── SAE (stage5_out, CNN seed87 only) ────"
    for SAE_SEED in $SAE_SEEDS; do
        SAE_CACHE=$(get_sae_cache $SAE_SEED)

        if [ ! -f "$SAE_CACHE" ]; then
            echo "⚠️  Not found: $SAE_CACHE"
            continue
        fi

        OUT="$BASE_OUT/stage5_out/pca50_std/sae_seed_${SAE_SEED}"

        echo "── SAE: sae_seed=$SAE_SEED ──"
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
}


# ──────────────────────────────────────────────────────────────
# CONDITION 4: PCA 250 + DE log2FC sweep
#   → 필터링 강도에 대한 robustness (CNN erank ↓ vs SAE 안정)
# ──────────────────────────────────────────────────────────────
run_de_sweep() {
    echo "══════════════════════════════════════════════"
    echo "  CONDITION 4: PCA 250 + DE log2FC sweep"
    echo "══════════════════════════════════════════════"

    DE_LOG2FC_VALUES="0.0 0.5 1.0 1.5 2.0"

    # --- CNN: all 3 layers ---
    for LAYER in $CNN_LAYERS; do
        echo ""
        echo "──── CNN Layer: ${LAYER} ────"
        for CNN_SEED in $CNN_SEEDS; do
            CNN_CACHE=$(get_cnn_cache $CNN_SEED $LAYER)

            if [ ! -f "$CNN_CACHE" ]; then
                echo "⚠️  Not found: $CNN_CACHE"
                continue
            fi

            for LOG2FC in $DE_LOG2FC_VALUES; do
                OUT="$BASE_OUT/${LAYER}/de_sweep/cnn_seed_${CNN_SEED}/log2fc_${LOG2FC}"

                echo "── CNN: layer=$LAYER, cnn_seed=$CNN_SEED, DE log2fc=$LOG2FC ──"
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
    done

    # --- SAE: stage5_out only (CNN seed87에서만 학습됨) ---
    echo ""
    echo "──── SAE (stage5_out, CNN seed87 only) ────"
    for SAE_SEED in $SAE_SEEDS; do
        SAE_CACHE=$(get_sae_cache $SAE_SEED)

        if [ ! -f "$SAE_CACHE" ]; then
            echo "⚠️  Not found: $SAE_CACHE"
            continue
        fi

        for LOG2FC in $DE_LOG2FC_VALUES; do
            OUT="$BASE_OUT/stage5_out/de_sweep/sae_seed_${SAE_SEED}/log2fc_${LOG2FC}"

            echo "── SAE: sae_seed=$SAE_SEED, DE log2fc=$LOG2FC ──"
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
echo "  Structure: {layer}/{condition}/cnn_seed_{S}/..."
echo "════════════════════════════════════════════════"
