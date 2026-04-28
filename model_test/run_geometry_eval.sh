#!/bin/bash
# ==============================================================================
# Geometry Evaluation Sweep — CNN 8 seeds + SAE 4 seeds
#
# Modes:
#   compare   — CNN stage5_out × 8 seeds + SAE × 4 seeds (Euclidean)
#   diffusion — Same as compare but with diffusion distance (recommended)
#   cnn       — CNN 3 layers × 8 seeds
#   all       — cnn + sae + compare + diffusion
#
# Usage:
#   bash model_test/run_geometry_eval.sh diffusion  # 추천: diffusion distance
#   bash model_test/run_geometry_eval.sh compare    # Euclidean 비교
#   bash model_test/run_geometry_eval.sh all
# ==============================================================================
set -e

DRIVE_BASE="/content/drive/MyDrive/Final_paper/lambda_labs_moco_only"
BASE_OUT="${DRIVE_BASE}/geometry_eval"

CNN_LAYERS="stage5_mid stage5_out refine_out"
CNN_SEEDS="42 87 95 123 124 256 445 457"
SAE_SEEDS="48 123 777 856"

SAE_CACHE_TEMPLATE="${DRIVE_BASE}/MoCo_seed87/SAE_seed{SAE_SEED}_no_L2norm_loss/features_cache_stage5_out_normrestored_all_no_SAE_GAP_L2_norm_again_d8192_sp800.npz"

SAMPLES_PER_CLASS=5000
K_NEIGHBORS="3 5 7 10"
RICCI_ALPHA=0.5
DELTA_N_SAMPLES=5000
SEED=42

# Apoptosis CSV (required for local std cosine vs DPT analysis)
APOPTOSIS_CSV="${DRIVE_BASE}/세포이미지별 사멸율/이미지별_세포사멸율_7200.csv"
APOP_K="5 10 15 20 25"

# PCA_DIM=0: NO dimensionality reduction — evaluate geometry in the
# original representation space. Same rationale as KNN accuracy eval.
# If we PCA first, we can't distinguish "PCA artifact" from "intrinsic geometry".
# DIFFMAP_COMPS=0: use ALL eigenvectors (N-1) for faithful DPT geodesic.
# DPT sums Σ_l (1/(1-λ_l))² (ψ_l(x)-ψ_l(y))² — truncation = approximation error.
DIFFMAP_COMPS=0              # 0 = ALL eigenvectors
PCA_DIM=0                    # 0 = skip PCA entirely
N_NEIGHBORS_DIFFMAP=35

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
# Common: run 12 models with given extra flags
# ──────────────────────────────────────────────────────────────
run_12_models() {
    local OUT_SUBDIR=$1
    shift
    local EXTRA_FLAGS="$@"

    echo "══════════════════════════════════════════════"
    echo "  Geometry: CNN 8 seeds + SAE 4 seeds"
    echo "  Output: ${BASE_OUT}/${OUT_SUBDIR}"
    echo "  Extra: ${EXTRA_FLAGS}"
    echo "══════════════════════════════════════════════"

    # ── CNN stage5_out: 8 seeds ──
    for CNN_SEED in $CNN_SEEDS; do
        CNN_CACHE=$(get_cnn_cache $CNN_SEED stage5_out)

        if [ ! -f "$CNN_CACHE" ]; then
            echo "⚠️  Not found: $CNN_CACHE"
            continue
        fi

        OUT="$BASE_OUT/${OUT_SUBDIR}/cnn_seed_${CNN_SEED}"

        echo "── CNN stage5_out seed=$CNN_SEED ──"
        python -m model_test.geometry_eval \
            --cnn_cache "$CNN_CACHE" \
            --gap_l2_norm \
            --label "stage5_out" \
            --samples_per_class $SAMPLES_PER_CLASS \
            --k_neighbors $K_NEIGHBORS \
            --ricci_alpha $RICCI_ALPHA \
            --delta_n_samples $DELTA_N_SAMPLES \
            --per_class \
            --seed $SEED \
            --output_dir "$OUT" \
            ${APOPTOSIS_CSV:+--apoptosis_csv "$APOPTOSIS_CSV" --apoptosis_k_neighbors $APOP_K} \
            $EXTRA_FLAGS
    done

    # ── SAE: 4 SAE seeds (CNN seed87 기반) ──
    for SAE_SEED in $SAE_SEEDS; do
        SAE_CACHE=$(get_sae_cache $SAE_SEED)

        if [ ! -f "$SAE_CACHE" ]; then
            echo "⚠️  Not found: $SAE_CACHE"
            continue
        fi

        OUT="$BASE_OUT/${OUT_SUBDIR}/sae_seed_${SAE_SEED}"

        echo "── SAE sae_seed=$SAE_SEED (CNN seed87) ──"
        python -m model_test.geometry_eval \
            --sae_cache "$SAE_CACHE" \
            --gap_l2_norm \
            --label "stage5_out" \
            --samples_per_class $SAMPLES_PER_CLASS \
            --k_neighbors $K_NEIGHBORS \
            --ricci_alpha $RICCI_ALPHA \
            --delta_n_samples $DELTA_N_SAMPLES \
            --per_class \
            --seed $SEED \
            --output_dir "$OUT" \
            ${APOPTOSIS_CSV:+--apoptosis_csv "$APOPTOSIS_CSV" --apoptosis_k_neighbors $APOP_K} \
            $EXTRA_FLAGS
    done
}


# ──────────────────────────────────────────────────────────────
# compare: Euclidean distance (기존) + Intrinsic Geometry
# ──────────────────────────────────────────────────────────────
run_compare() {
    run_12_models "compare_euclidean" "--compute_geometry"
}


# ──────────────────────────────────────────────────────────────
# diffusion: Diffusion distance + Intrinsic Geometry
# ──────────────────────────────────────────────────────────────
run_diffusion() {
    run_12_models "compare_diffusion" \
        "--compute_geometry --use_diffusion --n_diffmap_comps $DIFFMAP_COMPS --pca_dim $PCA_DIM --n_neighbors_diffmap $N_NEIGHBORS_DIFFMAP"
}


# ──────────────────────────────────────────────────────────────
# apop_std: Apoptosis local std only (NO geometry — fast)
#   cosine KNN vs DPT KNN, k=5 10 15 20 25
# ──────────────────────────────────────────────────────────────
run_apop_std() {
    if [ -z "$APOPTOSIS_CSV" ] || [ ! -f "$APOPTOSIS_CSV" ]; then
        echo "⚠️  APOPTOSIS_CSV not found: $APOPTOSIS_CSV"
        echo "   Set APOPTOSIS_CSV in the script and re-run."
        exit 1
    fi
    run_12_models "apop_std" \
        "--rank_correlation --rank_n_neighbors $N_NEIGHBORS_DIFFMAP --rank_pca_dim $PCA_DIM \
         --save_pairwise"


    # ── Aggregate across all seeds ──
    APOP_STD_DIR="${BASE_OUT}/apop_std"
    echo ""
    echo "══════════════════════════════════════════════"
    echo "  Aggregating apop_std results across seeds"
    echo "  Dir: ${APOP_STD_DIR}"
    echo "══════════════════════════════════════════════"
    python -m model_test.geometry_eval \
        --summarize_dir "$APOP_STD_DIR" \
        --output_dir    "$APOP_STD_DIR" \
        --dpi 200
}


# ──────────────────────────────────────────────────────────────
# CNN: 3 layers × 8 seeds (Euclidean, layer comparison)
# ──────────────────────────────────────────────────────────────
run_cnn() {
    echo "══════════════════════════════════════════════"
    echo "  Geometry: CNN 3 layers × 8 seeds"
    echo "══════════════════════════════════════════════"

    for LAYER in $CNN_LAYERS; do
        echo ""
        echo "──── Layer: ${LAYER} ────"
        for CNN_SEED in $CNN_SEEDS; do
            CNN_CACHE=$(get_cnn_cache $CNN_SEED $LAYER)

            if [ ! -f "$CNN_CACHE" ]; then
                echo "⚠️  Not found: $CNN_CACHE"
                continue
            fi

            OUT="$BASE_OUT/cnn/${LAYER}/seed_${CNN_SEED}"

            echo "── CNN: layer=$LAYER, seed=$CNN_SEED ──"
            python -m model_test.geometry_eval \
                --cnn_cache "$CNN_CACHE" \
                --gap_l2_norm \
                --label "${LAYER}" \
                --samples_per_class $SAMPLES_PER_CLASS \
                --k_neighbors $K_NEIGHBORS \
                --ricci_alpha $RICCI_ALPHA \
                --delta_n_samples $DELTA_N_SAMPLES \
                --per_class \
                --seed $SEED \
                --output_dir "$OUT"
        done
    done
}


# ──────────────────────────────────────────────────────────────
# SAE only: 4 SAE seeds
# ──────────────────────────────────────────────────────────────
run_sae() {
    echo "══════════════════════════════════════════════"
    echo "  Geometry: SAE (CNN seed87) × 4 SAE seeds"
    echo "══════════════════════════════════════════════"

    for SAE_SEED in $SAE_SEEDS; do
        SAE_CACHE=$(get_sae_cache $SAE_SEED)

        if [ ! -f "$SAE_CACHE" ]; then
            echo "⚠️  Not found: $SAE_CACHE"
            continue
        fi

        OUT="$BASE_OUT/sae/sae_seed_${SAE_SEED}"

        echo "── SAE: sae_seed=$SAE_SEED ──"
        python -m model_test.geometry_eval \
            --sae_cache "$SAE_CACHE" \
            --gap_l2_norm \
            --label "stage5_out" \
            --samples_per_class $SAMPLES_PER_CLASS \
            --k_neighbors $K_NEIGHBORS \
            --ricci_alpha $RICCI_ALPHA \
            --delta_n_samples $DELTA_N_SAMPLES \
            --per_class \
            --seed $SEED \
            --output_dir "$OUT"
    done
}


# ──────────────────────────────────────────────────────────────
# clean_rank: LOO KNN filter + DPT-based geometry + isometry test
#   --compute_geometry: Ricci/δ on diffusion coords
#   --rank_correlation: DPT vs cosine isometry (uses own dense DPT)
# ──────────────────────────────────────────────────────────────
run_clean_rank() {
    run_12_models "clean_rank" \
        "--compute_geometry --use_diffusion --n_diffmap_comps $DIFFMAP_COMPS --pca_dim $PCA_DIM --n_neighbors_diffmap $N_NEIGHBORS_DIFFMAP --knn_filter --knn_filter_k 10 --knn_filter_weights inv_sq --rank_correlation --n_rank_anchors 900 --rank_pca_dim $PCA_DIM --rank_n_neighbors $N_NEIGHBORS_DIFFMAP"
}


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────
case "${1:-apop_std}" in
    compare)    run_compare ;;
    diffusion)  run_diffusion ;;
    clean_rank) run_clean_rank ;;
    apop_std)   run_apop_std ;;
    cnn)        run_cnn ;;
    sae)        run_sae ;;
    all)
        run_cnn
        run_sae
        run_compare
        run_diffusion
        run_clean_rank
        run_apop_std
        ;;
    *)
        echo "Usage: bash $0 {compare|diffusion|clean_rank|apop_std|cnn|sae|all}"
        exit 1
        ;;
esac

echo ""
echo "════════════════════════════════════════════════"
echo "  Geometry evaluation complete!"
echo "  Results in: $BASE_OUT"
echo "════════════════════════════════════════════════"
