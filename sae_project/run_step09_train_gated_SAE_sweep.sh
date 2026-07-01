#!/bin/bash




export PYTHONPATH="$(cd .. && pwd):$PYTHONPATH"

SCRIPT="python -m sae_project.step09_train_gated_sae"
CNN_SEEDS=(42 87 95 123 124 256 445 457)
SAE_SEED=48
BASE="/home/ubuntu/model-east3/outputs"
SHARD_ROOT="/home/ubuntu/model-east3/wds_shards_tar"

# Configurations: "d_sae final_sparsity_coeff"
# (4096, 800) is mentioned twice in the prompt (once in lambda=800 list, once in dim=4096 list), 
# so there are 8 unique combinations instead of 9.
CONFIGS=(
    "600 50"
    "1024 800"
    "2048 800"
    "4096 800"
    "8192 800"
    "4096 0"
    "4096 200"
    "4096 3200"
)

for CNN_SEED in "${CNN_SEEDS[@]}"; do
    MODEL_DIR="${BASE}/MoCo_seed${CNN_SEED}"
    
    for CONFIG in "${CONFIGS[@]}"; do
        # Parse d_sae and lambda
        read -r D_SAE LAMBDA <<< "$CONFIG"
        
        # Exception condition: CNN seed 87, SAE seed 48, dim 8192, lambda 800 is already trained.
        if [ "$CNN_SEED" -eq 87 ] && [ "$SAE_SEED" -eq 48 ] && [ "$D_SAE" -eq 8192 ] && [ "$LAMBDA" -eq 800 ]; then
            echo "Skipping already trained config: CNN_SEED=${CNN_SEED}, SAE_SEED=${SAE_SEED}, d_sae=${D_SAE}, lambda=${LAMBDA}"
            continue
        fi

        # SAE Directory naming
        SAE_SAVE_DIR="${MODEL_DIR}/SAE_dim${D_SAE}_lambda${LAMBDA}_seed${SAE_SEED}_no_L2norm_loss"
        
        
        if ls ${SAE_SAVE_DIR}/*_ep008.pt 1> /dev/null 2>&1; then
            echo "⏩ Skipping already finished config (ep008.pt exists): CNN_SEED=${CNN_SEED}, d_sae=${D_SAE}, lambda=${LAMBDA}"
            continue
        fi
        
        echo "===== CNN_SEED=${CNN_SEED}, SAE_SEED=${SAE_SEED}, d_sae=${D_SAE}, lambda=${LAMBDA} ====="
        
        # Make sure directory exists for tee log file
        mkdir -p "${SAE_SAVE_DIR}"
        
        $SCRIPT \
            --save_dir "$MODEL_DIR" \
            --model_state_path "${MODEL_DIR}/best_model.pt" \
            --sae_save_dir "${SAE_SAVE_DIR}" \
            --shard_root "$SHARD_ROOT" \
            --which_layer stage5_out \
            --use_bf16 \
            --batch_size 64 \
            --epochs 8 \
            --d_sae "$D_SAE" \
            --final_sparsity_coeff "$LAMBDA" \
            --aux_coeff 0.03125 \
            --tie_gate_weights \
            --sparsity_warmup_steps 100 \
            --seed "$SAE_SEED" \
            2>&1 | tee "${SAE_SAVE_DIR}/sae_train_log.txt"
            
        echo "Finished config: CNN_SEED=${CNN_SEED}, SAE_SEED=${SAE_SEED}, d_sae=${D_SAE}, lambda=${LAMBDA} at $(date)"
        echo ""
    done
done
