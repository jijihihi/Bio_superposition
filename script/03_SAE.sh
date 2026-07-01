#!/bin/bash
# ==============================================================================
# SAE Training and Feature Extraction Pipeline
#
# This script trains Gated Sparse Autoencoders (SAEs) on CNN features
# and subsequently extracts the SAE feature representations (cache).
#
# Notes on spatial positioning and L2 norm:
#   - To apply weight based on spatial positioning via L2 norm in the loss function,
#     multiply by `token_l2_norm`.
# ==============================================================================

export PYTHONPATH="$(cd .. && pwd):$PYTHONPATH"

SCRIPT="python -m sae_project.step09_train_gated_sae"
EXTRACT_SCRIPT="python -m cache_extraction.extract_features_lambda_labs"

# ==============================================================================
# Settings
# ==============================================================================
# We only trained CNN models for seed 42 in 02_run_cnn_models.sh (for SAE)
CNN_SEEDS=(42)
SAE_SEED=48

BASE="/home/ubuntu/model-east3/outputs"
SHARD_ROOT="/home/ubuntu/model-east3/wds_shards_tar"

# Configurations: "d_sae final_sparsity_coeff (lambda)"
CONFIGS=(
    "600 50"
    "1024 800"
    "2048 800"
    # "4096 800"
    # "8192 800"
    # "4096 0"
    # "4096 200"
    # "4096 3200"
)

# ==============================================================================
# Main Pipeline
# ==============================================================================

# for CNN_SEED in "${CNN_SEEDS[@]}"; do
#     MODEL_DIR="${BASE}/MoCo_seed${CNN_SEED}_refactoring"
    
#     for CONFIG in "${CONFIGS[@]}"; do
#         # Parse d_sae and lambda
#         read -r D_SAE LAMBDA <<< "$CONFIG"
        
#         # Exception condition: specific config is already trained
#         if [ "$CNN_SEED" -eq 87 ] && [ "$SAE_SEED" -eq 48 ] && [ "$D_SAE" -eq 8192 ] && [ "$LAMBDA" -eq 800 ]; then
#             echo "⏩ Skipping manually excluded config: CNN_SEED=${CNN_SEED}, d_sae=${D_SAE}, lambda=${LAMBDA}"
#             continue
#         fi

#         # ----------------------------------------------------------------------
#         # 1. Train SAE (Commented out for evaluation only)
#         # ----------------------------------------------------------------------
#         SAE_SAVE_DIR="${MODEL_DIR}/SAE_refactoring_dim${D_SAE}_lambda${LAMBDA}_seed${SAE_SEED}_no_L2norm_loss"
#         mkdir -p "${SAE_SAVE_DIR}"
        
#         # # Check if training is already finished (ep008.pt exists)
#         # if ls ${SAE_SAVE_DIR}/*_ep008.pt 1> /dev/null 2>&1; then
#         #     echo "✅ Training already finished (ep008.pt exists): CNN_SEED=${CNN_SEED}, d_sae=${D_SAE}, lambda=${LAMBDA}"
#         # else
#         #     echo "▶️ Starting SAE Training: CNN_SEED=${CNN_SEED}, SAE_SEED=${SAE_SEED}, d_sae=${D_SAE}, lambda=${LAMBDA} ====="
            
#         #     $SCRIPT \
#         #     --save_dir "$MODEL_DIR" \
#         #     --model_state_path "${MODEL_DIR}/best_model.pt" \
#         #     --sae_save_dir "${SAE_SAVE_DIR}" \
#         #     --shard_root "$SHARD_ROOT" \
#         #     --which_layer stage5_out \
#         #     --use_bf16 \
#         #     --batch_size 64 \
#         #     --epochs 8 \
#         #     --d_sae "$D_SAE" \
#         #     --final_sparsity_coeff "$LAMBDA" \
#         #     --aux_coeff 0.03125 \
#         #     --tie_gate_weights \
#         #     --sparsity_warmup_steps 100 \
#         #     --seed "$SAE_SEED" \
#         #     2>&1 | tee "${SAE_SAVE_DIR}/sae_train_log.txt"
            
#         #     echo "🏁 Finished training config: CNN_SEED=${CNN_SEED}, d_sae=${D_SAE}, lambda=${LAMBDA} at $(date)"
#         # fi

        # ----------------------------------------------------------------------
        # 2. Extract Features (Cache)
        # ----------------------------------------------------------------------
        # Find the final epoch checkpoint
#         SAE_CKPT=$(ls ${SAE_SAVE_DIR}/*_ep008.pt 2>/dev/null | head -n 1)
        
#         if [ -z "$SAE_CKPT" ]; then
#             echo "⚠️ Warning: SAE checkpoint not found in ${SAE_SAVE_DIR}. Skipping extraction."
#         else
#             CKPT_BASENAME=$(basename "$SAE_CKPT" .pt)
#             CACHE_OUT="${SAE_SAVE_DIR}/sae_refactoring_gap_${CKPT_BASENAME}_all.npz"
            
#             if [ -f "$CACHE_OUT" ]; then
#                 echo "✅ Cache already extracted: $CACHE_OUT"
#             else
#                 echo "▶️ Extracting cache for ${CKPT_BASENAME}..."
#                 $EXTRACT_SCRIPT \
#                     --sae_ckpt "$SAE_CKPT" \
#                     --save_dir "$MODEL_DIR" \
#                     --model_state_path "${MODEL_DIR}/best_model.pt" \
#                     --shard_root "$SHARD_ROOT" \
#                     --which_layer "stage5_out" \
#                     --use_all_data \
#                     --batch_size 128 \
#                     --num_workers 4 \
#                     --output_path "$CACHE_OUT"
                
#                 echo "🏁 Finished cache extraction for config: d_sae=${D_SAE}, lambda=${LAMBDA} at $(date)"
#             fi
#         fi
        
#         echo "--------------------------------------------------------------------------"
#     done
# done

# ==============================================================================
# 3. Evaluate SAE Linear Probes
# ==============================================================================
echo ""
echo "=========================================================================="
echo " Evaluating SAE Linear Probes from Cache"
echo "=========================================================================="
PROBE_RESULTS_CSV="${BASE}/refactoring_sae_linear_probe_results.csv"

# Construct TARGET_CONFIGS from the array
TARGET_CONFIGS=""
for CONFIG in "${CONFIGS[@]}"; do
    read -r D_SAE LAMBDA <<< "$CONFIG"
    TARGET_CONFIGS="${TARGET_CONFIGS} ${D_SAE}_${LAMBDA}"
done

echo "  ▶ Running Linear Probes on extracted SAE caches (Targets: $TARGET_CONFIGS)..."
python -m sae_project.step18_eval_probe_from_cache \
    --cache_dir "${BASE}" \
    --save_csv "${PROBE_RESULTS_CSV}" \
    --model_base_dir "${BASE}" \
    --dead_threshold 1e-5 \
    --num_classes 4 \
    --target_configs $TARGET_CONFIGS

echo "🏁 Linear Probe evaluation complete! Results saved to ${PROBE_RESULTS_CSV}"


# ==============================================================================
# 5. Superposition Disentanglement 
# ==============================================================================
DISENTANGLEMENT_OUT="${BASE}/refactoring/disentanglement"
mkdir -p "${DISENTANGLEMENT_OUT}"

echo ""
echo "=========================================================================="
echo " 5. Calculating Superposition Disentanglement & Plotting Heatmaps"
echo "=========================================================================="
python -m sae_project.step17_superposition_disentanglement \
    --base_dir "${BASE}" \
    --save_dir "${DISENTANGLEMENT_OUT}" \
    --heatmap_gamma 2.5 \
    --target_configs $TARGET_CONFIGS

echo "🏁 Disentanglement evaluation complete! Results saved to ${DISENTANGLEMENT_OUT}"

# ==============================================================================
# 6. SAE Pareto Metrics Plotting
# ==============================================================================
PARETO_OUT="${BASE}/refactoring/sae_pareto_metrics"
mkdir -p "${PARETO_OUT}"

echo ""
echo "=========================================================================="
echo " 6. Generating SAE Pareto Metrics Plots"
echo "=========================================================================="
python -m sae_project.plot_sae_metrics \
    --base_dir "${BASE}" \
    --linear_probe_csv "${PROBE_RESULTS_CSV}" \
    --cnn_baseline_csv "${BASE}/MoCo_seed42_refactoring/linear_eval_results.csv" \
    --disentanglement_csv "${DISENTANGLEMENT_OUT}/disentanglement_results.csv" \
    --erank_dir "${BASE}/refactoring/erank" \
    --save_dir "${PARETO_OUT}"

echo "🏁 SAE Pareto Metrics plotting complete! Plots saved to ${PARETO_OUT}"
