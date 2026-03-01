pip install numpy==1.26.4
pip install tifffile tqdm scanpy

SCRIPT="python -m sae_project.step09_train_gated_sae"
SEEDS=(256 87 95 457) # 다른 시드도 이걸로 학습
BASE="/home/ubuntu/model-east3/outputs"
for SEED in "${SEEDS[@]}"; do
    MODEL_DIR="${BASE}/MoCo_seed${SEED}"
    echo "===== SAE for seed=${SEED} ====="
    
    $SCRIPT \
        --save_dir "$MODEL_DIR" \
        --model_state_path "${MODEL_DIR}/best_model.pt" \
        --sae_save_dir "${MODEL_DIR}/SAE" \
        --shard_root /home/ubuntu/model-east3/wds_shards_tar \
        --train_all_layers \
        --use_bf16 \
        --batch_size 64 \
        --epochs 8 \
        --d_sae 4096 \
        --final_sparsity_coeff 3200 \
        --aux_coeff 0.03125 \
        --tie_gate_weights \
        --sparsity_warmup_steps 100 \
        --seed "$SEED" \
        2>&1 | tee "${MODEL_DIR}/SAE/sae_train_log.txt"
    
    echo "Seed ${SEED} SAE done at $(date)"
done

# layer 2개, seed 개수 만큼 또 곱하고.

############## 2026.02.23 MoCo_seed457 stage5_out 해야할 차례!!!! #########################

SCRIPT="python -m sae_project.step09_train_gated_sae"
SEEDS=(457) # 다른 시드도 이걸로 학습
BASE="/home/ubuntu/model-east3/outputs"
for SEED in "${SEEDS[@]}"; do
    MODEL_DIR="${BASE}/MoCo_seed${SEED}"
    echo "===== SAE for seed=${SEED} ====="
    
    $SCRIPT \
        --save_dir "$MODEL_DIR" \
        --model_state_path "${MODEL_DIR}/best_model.pt" \
        --sae_save_dir "${MODEL_DIR}/SAE" \
        --shard_root /home/ubuntu/model-east3/wds_shards_tar \
        --train_all_layers \
        --use_bf16 \
        --batch_size 64 \
        --epochs 8 \
        --d_sae 4096 \
        --final_sparsity_coeff 3200 \
        --aux_coeff 0.03125 \
        --tie_gate_weights \
        --sparsity_warmup_steps 100 \
        --seed "$SEED" \
        2>&1 | tee "${MODEL_DIR}/SAE/sae_train_log.txt"
    
    echo "Seed ${SEED} SAE done at $(date)"
done

stage5_mid 도 학습하자


SCRIPT="python -m sae_project.step09_train_gated_sae"
SEEDS=(42 45 123 124 256) # 다른 시드도 이걸로 학습
BASE="/home/ubuntu/model-east3/outputs"
for SEED in "${SEEDS[@]}"; do
    MODEL_DIR="${BASE}/MoCo_seed${SEED}"
    echo "===== SAE for seed=${SEED} ====="
    
    $SCRIPT \
        --save_dir "$MODEL_DIR" \
        --model_state_path "${MODEL_DIR}/best_model.pt" \
        --sae_save_dir "${MODEL_DIR}/SAE" \
        --shard_root /home/ubuntu/model-east3/wds_shards_tar \
        ----which_layer stage5_mid \
        --use_bf16 \
        --batch_size 64 \
        --epochs 8 \
        --d_sae 4096 \
        --final_sparsity_coeff 3200 \
        --aux_coeff 0.03125 \
        --tie_gate_weights \
        --sparsity_warmup_steps 100 \
        --seed "$SEED" \
        2>&1 | tee "${MODEL_DIR}/SAE/sae_train_log.txt"
    
    echo "Seed ${SEED} SAE done at $(date)"
done


################

SCRIPT="python -m sae_project.step09_train_gated_sae"
SEEDS=(87) # 다른 시드도 이걸로 학습
BASE="/home/ubuntu/model-east3/outputs"
for SEED in "${SEEDS[@]}"; do
    MODEL_DIR="${BASE}/MoCo_seed${SEED}"
    echo "===== SAE for seed=${SEED} ====="
    
    $SCRIPT \
        --save_dir "$MODEL_DIR" \
        --model_state_path "${MODEL_DIR}/best_model.pt" \
        --sae_save_dir "${MODEL_DIR}/SAE_no_L2norm_loss" \
        --shard_root /home/ubuntu/model-east3/wds_shards_tar \
        --which_layer stage5_out \
        --use_bf16 \
        --batch_size 64 \
        --epochs 8 \
        --d_sae 8192 \
        --final_sparsity_coeff 800 \
        --aux_coeff 0.03125 \
        --tie_gate_weights \
        --sparsity_warmup_steps 100 \
        --seed "$SEED" \
        2>&1 | tee "${MODEL_DIR}/SAE/sae_train_log.txt"
    
    echo "Seed ${SEED} SAE done at $(date)"
done




### CNN 87에 대해서 SAE 다른 seed로도 학습해보자.



# SAE seeds (3개) # 넣어줄 때 loss function에 L2 norm 안 곱해준다. 세세한것도 보게. L2 norm이 큰 분류에만 도움되는거 보지 않게. 즉 CNN은 분류를 넘어선 내부적인 것들도 학습한 것이다. 이게 DPT에서 좋은 결과가 나왔었음.
SAE_SEEDS=(48 123 777)

for SAE_SEED in "${SAE_SEEDS[@]}"; do
    echo "=== SAE seed ${SAE_SEED} on CNN seed 87 ==="
    
    python -m sae_project.step09_train_gated_sae \
        --seed ${SAE_SEED} \
        --save_dir "${SAVE_DIR}" \
        --model_state_path "${MODEL}" \
        --shard_root "${SHARD}" \
        --sae_save_dir "${SAVE_DIR}/SAE_seed${SAE_SEED}_no_L2norm_loss" \
        --which_layer refine_out \
        --use_bf16 \
        --batch_size 64 \
        --d_sae 8192 \
        --final_sparsity_coeff 800.0 \
        --aux_coeff 0.03125 \
        --epochs 8 \
        --which_layer stage5_out \
        --tie_gate_weights \
        --sparsity_warmup_steps 100
    
    echo "SAE seed ${SAE_SEED} done at $(date)"
done


# SAE seeds (3개) # 넣어줄 때 loss function에 L2 norm 안 곱해준다. 세세한것도 보게. L2 norm이 큰 분류에만 도움되는거 보지 않게. 즉 CNN은 분류를 넘어선 내부적인 것들도 학습한 것이다. 이게 DPT에서 좋은 결과가 나왔었음.
SAE_SEEDS=(777 856)

for SAE_SEED in "${SAE_SEEDS[@]}"; do
    echo "=== SAE seed ${SAE_SEED} on CNN seed 87 ==="
    
    python -m sae_project.step09_train_gated_sae \
        --seed ${SAE_SEED} \
        --save_dir "${SAVE_DIR}" \
        --model_state_path "${MODEL}" \
        --shard_root "${SHARD}" \
        --sae_save_dir "${SAVE_DIR}/SAE_seed${SAE_SEED}_no_L2norm_loss" \
        --which_layer refine_out \
        --use_bf16 \
        --batch_size 64 \
        --d_sae 8192 \
        --final_sparsity_coeff 800.0 \
        --aux_coeff 0.03125 \
        --epochs 8 \
        --which_layer stage5_out \
        --tie_gate_weights \
        --sparsity_warmup_steps 100
    
    echo "SAE seed ${SAE_SEED} done at $(date)"
done




stage5_mid 도 학습하자


SCRIPT="python -m sae_project.step09_train_gated_sae"
SEEDS=(42 45 123 124 256) # 다른 시드도 이걸로 학습
BASE="/home/ubuntu/model-east3/outputs"
for SEED in "${SEEDS[@]}"; do
    MODEL_DIR="${BASE}/MoCo_seed${SEED}"
    echo "===== SAE for seed=${SEED} ====="
    
    $SCRIPT \
        --save_dir "$MODEL_DIR" \
        --model_state_path "${MODEL_DIR}/best_model.pt" \
        --sae_save_dir "${MODEL_DIR}/SAE" \
        --shard_root /home/ubuntu/model-east3/wds_shards_tar \
        ----which_layer stage5_mid \
        --use_bf16 \
        --batch_size 64 \
        --epochs 8 \
        --d_sae 4096 \
        --final_sparsity_coeff 3200 \
        --aux_coeff 0.03125 \
        --tie_gate_weights \
        --sparsity_warmup_steps 100 \
        --seed "$SEED" \
        2>&1 | tee "${MODEL_DIR}/SAE/sae_train_log.txt"
    
    echo "Seed ${SEED} SAE done at $(date)"
done