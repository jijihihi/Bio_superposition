#SEEDS=(42 45 123)
#SEEDS=(42 45 123 124 256) # 전체 다 같이 한 phate
SEEDS=(87)
BASE="/home/ubuntu/model-east3/outputs"
for SEED in "${SEEDS[@]}"; do
    MODEL_DIR="${BASE}/MoCo_seed${SEED}"
    SAE_DIR="${MODEL_DIR}/SAE_seed856_no_L2norm_loss" # _no_L2norm_loss
    # for LAYER in stage5_out refine_out # d4096_gated_sp3200.0_aux0.03125_tied_ep008.pt"
    for LAYER in stage5_out; do  
        CKPT="${SAE_DIR}/${LAYER}_d8192_gated_sp800.0_aux0.03125_tied_ep008.pt"
        echo "===== PHATE seed=${SEED} layer=${LAYER} ====="
        python -m kendall_correlation_coefficient.phate \
            --features_cache "${SAE_DIR}/features_cache_stage5_out_normrestored_all_no_SAE_GAP_L2_norm_again_d8192_sp800.npz" \
            --sae_ckpt "$CKPT" \
            --save_dir "$MODEL_DIR" \
            --model_state_path "${MODEL_DIR}/best_model.pt" \
            --shard_root /home/ubuntu/model-east3/wds_shards_tar \
            --output_dir "${SAE_DIR}/phate" \
            --dead_threshold 5e-5 \
            --samples_per_class 8000 \
	        --t "auto" \
            --seed "$SEED" \
	        --knn_dist "euclidean" \
            --batch_size 64 \
            --restore_token_norm \
            --min_cv 0.1 \
            --norm log_std \
            --filter_mode cv de \
            --de_adj_p 0.05 \
            --de_min_log2fc 1.0 \
            --n_pca 100 \
            --knn 10 \
            --dead_threshold 5e-5 \
            --decay 100 \
            --gap_l2_norm \
            --paga \
            --paga_n_neighbors 30 \
            --paga_n_pcs 50
            
    done
done

# --restore_token_norm

#--filter_mode cv de \
# --de_adj_p 0.05 \
# --de_min_log2fc 1.0 \

        



SEEDS=(87)
BASE="/home/ubuntu/model-east3/outputs"
for SEED in "${SEEDS[@]}"; do
    MODEL_DIR="${BASE}/MoCo_seed${SEED}"
    SAE_DIR="${MODEL_DIR}/SAE_seed856_no_L2norm_loss" # _no_L2norm_loss
    # for LAYER in stage5_out refine_out # d4096_gated_sp3200.0_aux0.03125_tied_ep008.pt"
    for LAYER in stage5_out; do  
        CKPT="${SAE_DIR}/${LAYER}_d8192_gated_sp800.0_aux0.03125_tied_ep008.pt"
        echo "===== PHATE seed=${SEED} layer=${LAYER} ====="
        python -m kendall_correlation_coefficient.phate \
            --features_cache "${SAE_DIR}/features_cache_stage5_out_normrestored_all_no_SAE_GAP_L2_norm_again_d8192_sp800.npz" \
            --sae_ckpt "$CKPT" \
            --save_dir "$MODEL_DIR" \
            --model_state_path "${MODEL_DIR}/best_model.pt" \
            --shard_root /home/ubuntu/model-east3/wds_shards_tar \
            --output_dir "${SAE_DIR}/phate" \
            --dead_threshold 5e-5 \
            --samples_per_class 8000 \
	        --t "auto" \
            --seed "$SEED" \
	        --knn_dist "euclidean" \
            --batch_size 64 \
            --restore_token_norm \
            --norm log_std \
            --filter_mode cv de \
            --min_cv 0.1 \
            --de_adj_p 0.05 \
            --de_min_log2fc 0.3 \
            --n_pca 100 \
            --knn 10 \
            --decay 100 \
            --gap_l2_norm \
            --paga \
            --paga_n_neighbors 30 \
            --paga_n_pcs 50 \
            --de_mode "per_mut"
            
    done
done

# --filter_mode cv de 가 없으면 per_mut가 안된다.
# filter mode하고 uinon할수도 있고 filter mode하고 per mut할수도 있다.