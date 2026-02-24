#!/bin/bash
# ==============================================================================
# Run linear evaluation for all CNN encoder seeds
# ==============================================================================

BASE_DIR="/home/ubuntu/model-east3/outputs"
SHARD_ROOT="/home/ubuntu/model-east3/wds_shards_tar"

# ── MoCo (GAP L2 norm) seeds ──
MOCO_SEEDS=(42 45 123 87 95)

# ── MoCo (no GAP L2 norm) seeds ──  
NO_NORM_SEEDS=(42 45 123 87 95)

# ── Build checkpoint + save_dir lists ──
MOCO_CKPTS=()
MOCO_DIRS=()
for S in "${MOCO_SEEDS[@]}"; do
    CKPT="${BASE_DIR}/MoCo_seed${S}/best_model.pt"
    if [ -f "$CKPT" ]; then
        MOCO_CKPTS+=("$CKPT")
        MOCO_DIRS+=("${BASE_DIR}/MoCo_seed${S}")
    else
        echo "⚠️  Not found: $CKPT"
    fi
done

NO_NORM_CKPTS=()
NO_NORM_DIRS=()
for S in "${NO_NORM_SEEDS[@]}"; do
    CKPT="${BASE_DIR}/MoCo_seed${S}_no_GAPL2norm/best_model.pt"
    if [ -f "$CKPT" ]; then
        NO_NORM_CKPTS+=("$CKPT")
        NO_NORM_DIRS+=("${BASE_DIR}/MoCo_seed${S}_no_GAPL2norm")
    else
        echo "⚠️  Not found: $CKPT"
    fi
done

echo "═══════════════════════════════════════════"
echo "  MoCo models: ${#MOCO_CKPTS[@]}"
echo "  No-norm models: ${#NO_NORM_CKPTS[@]}"
echo "═══════════════════════════════════════════"

# ── Run MoCo evaluation ──
if [ ${#MOCO_CKPTS[@]} -gt 0 ]; then
    echo ""
    echo "▶ Evaluating MoCo (GAP L2 norm) models..."
    python -m model_test.linear_eval \
        --ckpt_paths "${MOCO_CKPTS[@]}" \
        --save_dirs "${MOCO_DIRS[@]}" \
        --shard_root "${SHARD_ROOT}" \
        --output_dir "${BASE_DIR}/linear_eval_moco" \
        --lp_epochs 50 \
        --lp_lr 0.1 \
        --batch_size 128 \
        --seed 42
fi

# ── Run no-norm evaluation ──
if [ ${#NO_NORM_CKPTS[@]} -gt 0 ]; then
    echo ""
    echo "▶ Evaluating MoCo (no GAP L2 norm) models..."
    python -m model_test.linear_eval \
        --ckpt_paths "${NO_NORM_CKPTS[@]}" \
        --save_dirs "${NO_NORM_DIRS[@]}" \
        --shard_root "${SHARD_ROOT}" \
        --output_dir "${BASE_DIR}/linear_eval_no_norm" \
        --lp_epochs 50 \
        --lp_lr 0.1 \
        --batch_size 128 \
        --seed 42
fi

echo ""
echo "═══════════════════════════════════════════"
echo "  ALL DONE"
echo "═══════════════════════════════════════════"
