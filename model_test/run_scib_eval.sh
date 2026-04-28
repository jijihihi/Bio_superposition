#!/bin/bash
# ==============================================================================
# scib Evaluation Sweep — CNN layers + SAE
#
# 1) CNN 3 layers × 8 seeds
# 2) SAE (stage5_out, CNN seed87) × 4 SAE seeds
#
# Usage:
#   bash model_test/run_scib_eval.sh cnn         # CNN only (3 layers)
#   bash model_test/run_scib_eval.sh sae         # SAE only
#   bash model_test/run_scib_eval.sh compare     # CNN stage5_out + SAE (one figure)
#   bash model_test/run_scib_eval.sh all         # Everything
# ==============================================================================
set -e

# ── Paths ──
DRIVE_BASE="/content/drive/MyDrive/Final_paper/lambda_labs_moco_only"
BASE_OUT="${DRIVE_BASE}/scib_eval"

CNN_LAYERS="stage5_mid stage5_out refine_out"
CNN_SEEDS="42 87 95 123 124 256 445 457"
SAE_SEEDS="48 123 777 856"

SAE_CACHE_TEMPLATE="${DRIVE_BASE}/MoCo_seed87/SAE_seed{SAE_SEED}_no_L2norm_loss/features_cache_stage5_out_normrestored_all_no_SAE_GAP_L2_norm_again_d8192_sp800.npz"

DEAD_THRESHOLD="1e-5"
SAMPLES_PER_CLASS=10000
N_PCS=0
N_NEIGHBORS=15
SEED=42

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
# CNN evaluation — 3 layers × 8 seeds
# ──────────────────────────────────────────────────────────────
run_cnn() {
    echo "══════════════════════════════════════════════"
    echo "  scib CNN Layer Evaluation"
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
            python -m model_test.scib_eval \
                --cnn_cache "$CNN_CACHE" \
                --gap_l2_norm \
                --label "${LAYER}" \
                --n_pcs $N_PCS \
                --n_neighbors $N_NEIGHBORS \
                --samples_per_class $SAMPLES_PER_CLASS \
                --dead_threshold $DEAD_THRESHOLD \
                --seed $SEED \
                --output_dir "$OUT"
        done
    done
}


# ──────────────────────────────────────────────────────────────
# SAE evaluation — 4 SAE seeds (CNN seed 87 only)
# ──────────────────────────────────────────────────────────────
run_sae() {
    echo "══════════════════════════════════════════════"
    echo "  scib SAE Evaluation (CNN seed87)"
    echo "══════════════════════════════════════════════"

    for SAE_SEED in $SAE_SEEDS; do
        SAE_CACHE=$(get_sae_cache $SAE_SEED)

        if [ ! -f "$SAE_CACHE" ]; then
            echo "⚠️  Not found: $SAE_CACHE"
            continue
        fi

        OUT="$BASE_OUT/sae/sae_seed_${SAE_SEED}"

        echo "── SAE: sae_seed=$SAE_SEED ──"
        python -m model_test.scib_eval \
            --sae_cache "$SAE_CACHE" \
            --label "stage5_out" \
            --n_pcs $N_PCS \
            --gap_l2_norm \
            --n_neighbors $N_NEIGHBORS \
            --samples_per_class $SAMPLES_PER_CLASS \
            --dead_threshold $DEAD_THRESHOLD \
            --seed $SEED \
            --output_dir "$OUT"
    done
}


# ──────────────────────────────────────────────────────────────
# Direct comparison — CNN stage5_out (seed87) vs SAE in one run
# ──────────────────────────────────────────────────────────────
run_compare() {
    echo "══════════════════════════════════════════════"
    echo "  scib Direct Comparison: CNN vs SAE"
    echo "══════════════════════════════════════════════"

    CNN_CACHE=$(get_cnn_cache 87 stage5_out)
    # Use first available SAE seed
    for SAE_SEED in $SAE_SEEDS; do
        SAE_CACHE=$(get_sae_cache $SAE_SEED)

        if [ ! -f "$CNN_CACHE" ] || [ ! -f "$SAE_CACHE" ]; then
            echo "⚠️  Cache not found"
            continue
        fi

        OUT="$BASE_OUT/comparison/sae_seed_${SAE_SEED}"

        echo "── Compare: CNN seed87 vs SAE seed${SAE_SEED} ──"
        python -m model_test.scib_eval \
            --cnn_cache "$CNN_CACHE" \
            --sae_cache "$SAE_CACHE" \
            --gap_l2_norm \
            --label "stage5_out" \
            --n_pcs $N_PCS \
            --n_neighbors $N_NEIGHBORS \
            --samples_per_class $SAMPLES_PER_CLASS \
            --dead_threshold $DEAD_THRESHOLD \
            --seed $SEED \
            --output_dir "$OUT"
    done
}


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────
case "${1:-all}" in
    cnn)     run_cnn ;;
    sae)     run_sae ;;
    compare) run_compare ;;
    all)
        run_cnn
        run_sae
        run_compare
        ;;
    *)
        echo "Usage: bash $0 {cnn|sae|compare|all}"
        exit 1
        ;;
esac

echo ""
echo "════════════════════════════════════════════════"
echo "  scib evaluation complete!"
echo "  Results in: $BASE_OUT"
echo "════════════════════════════════════════════════"
