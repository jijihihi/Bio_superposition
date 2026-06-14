#!/bin/bash
# ==============================================================================
# Local Linearity Sweep: K-Neighbors & SAE Dimension (Multi-Seed)
#
# 여러 시드(8개)에 대해 K=(5,10,15,20,25), SAE 차원별로
# local_knn_std.py를 실행하고, 시드 평균/표준편차를 포함한 최종 추세선을 그립니다.
# ==============================================================================
set -e

BASE="/content/drive/MyDrive/Final_paper/lambda_labs_moco_only"
APOPTOSIS_CSV="${BASE}/세포이미지별 사멸율/이미지별_세포사멸율_7200.csv"
OUT_BASE="${BASE}/local_linearity_dim_and_k_sweep"

SEEDS=(42 87 95 123 124 256 445 457)
DIMS=(1024 2048 4096 8192)
K_NEIGHBORS="5 10 15 20 25"

echo "=================================================================="
echo "🚀 Running local_knn_std for multiple seeds & dimensions & k-sweep"
echo "📂 Outputs will be saved to: $OUT_BASE"
echo "=================================================================="

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "=================================================================="
    echo "🌱 Processing Seed: $SEED"
    echo "=================================================================="
    
    # CNN Cache
    CNN_CACHE="${BASE}/caches_CNN_SAE_class_27000_withnewclass/CNN_seed${SEED}/cnn_gap_stage5_out_withnewclass.npz"
    if [ ! -f "$CNN_CACHE" ]; then
        # 혹시 몰라 예전 경로도 체크
        CNN_CACHE="${BASE}/MoCo_seed${SEED}/CNN_GAP/cnn_gap_stage5_out_all.npz"
    fi

    for D in "${DIMS[@]}"; do
        echo "▶️ Processing SAE Dimension: $D"
        
        SAE_CACHE="${BASE}/caches_CNN_SAE_class_27000_withnewclass/CNN_seed${SEED}_SAE/sae_gap_d${D}_lam800_normrestored_withnewclass.npz"

        if [ ! -f "$SAE_CACHE" ] || [ ! -f "$CNN_CACHE" ]; then
            echo "⚠️ Cache not found for Seed=$SEED, dim=$D. Skipping..."
            continue
        fi

        OUT_DIR="${OUT_BASE}/seed_${SEED}/d${D}"
        mkdir -p "$OUT_DIR"

        python -m apoptosis_prediction.local_knn_std \
            --cnn_cache "$CNN_CACHE" \
            --sae_cache "$SAE_CACHE" \
            --apoptosis_csv "$APOPTOSIS_CSV" \
            --k_neighbors $K_NEIGHBORS \
            --dead_threshold 1e-5 \
            --filter_mode none \
            --pca_dim 0 \
            --output_dir "$OUT_DIR" \
            --n_permutations 0
            
        echo "✅ Completed Seed=$SEED, dim=$D"
    done
done

# ==============================================================================
# Plotting Script Generation & Execution
# ==============================================================================
PLOT_SCRIPT="${OUT_BASE}/plot_trend_sweep_multiseed.py"

cat << 'EOF' > "$PLOT_SCRIPT"
import os
import json
import numpy as np
import matplotlib.pyplot as plt

base_dir = "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/local_linearity_dim_and_k_sweep"
seeds = [42, 87, 95, 123, 124, 256, 445, 457]
dims = [1024, 2048, 4096, 8192]
mutations = ["SNCA", "GBA", "LRRK2"]
ks = [5, 10, 15, 20, 25]

# Colors for different SAE dimensions
colors = {1024: "#E8553A", 2048: "#1DB954", 4096: "#DD8452", 8192: "#9B59B6"}

for mut in mutations:
    plt.figure(figsize=(9, 6))
    
    # 딕셔너리 구조: dict[k] = list of (seed_means)
    cnn_data = {k: [] for k in ks}
    sae_data = {d: {k: [] for k in ks} for d in dims}
    
    for seed in seeds:
        # CNN 결과는 아무 차원의 결과 파일에서나 동일하므로 첫번째 존재하는 차원에서 가져옵니다.
        cnn_extracted = False
        
        for d in dims:
            json_path = os.path.join(base_dir, f"seed_{seed}", f"d{d}", "local_linearity_results.json")
            if not os.path.exists(json_path):
                continue
                
            with open(json_path, 'r') as f:
                data = json.load(f)
                
            for k in ks:
                for res in data["results"]:
                    if res["mutation"] == mut and res["k"] == k:
                        if res["source"] == "SAE":
                            sae_data[d][k].append(res["mean_ratio"])
                        elif res["source"] == "CNN" and not cnn_extracted:
                            cnn_data[k].append(res["mean_ratio"])
                            
            cnn_extracted = True  # 한 시드에 대해 CNN은 한 번만 뽑으면 됨

    # Plot SAE lines per dimension (Mean + Std Dev Shade)
    for d in dims:
        means = []
        stds = []
        valid_ks = []
        for k in ks:
            if sae_data[d][k]:
                valid_ks.append(k)
                means.append(np.mean(sae_data[d][k]))
                stds.append(np.std(sae_data[d][k]))
                
        if valid_ks:
            means = np.array(means)
            stds = np.array(stds)
            line, = plt.plot(valid_ks, means, 'o-', color=colors.get(d, "black"), 
                             linewidth=2.5, markersize=8, label=f"SAE (d={d})")
            plt.fill_between(valid_ks, means - stds, means + stds, color=line.get_color(), alpha=0.15)
            
    # Plot CNN Baseline
    cnn_means = []
    cnn_stds = []
    valid_cnn_ks = []
    for k in ks:
        if cnn_data[k]:
            valid_cnn_ks.append(k)
            cnn_means.append(np.mean(cnn_data[k]))
            cnn_stds.append(np.std(cnn_data[k]))
            
    if valid_cnn_ks:
        cnn_means = np.array(cnn_means)
        cnn_stds = np.array(cnn_stds)
        line, = plt.plot(valid_cnn_ks, cnn_means, 's--', color="#4C72B0", 
                         linewidth=2.5, markersize=8, label="CNN (Baseline)")
        plt.fill_between(valid_cnn_ks, cnn_means - cnn_stds, cnn_means + cnn_stds, color=line.get_color(), alpha=0.15)

    # Styling
    plt.title(f"{mut} - Local Std Ratio Trend across 8 Seeds (Sparsity=800)", fontsize=14, fontweight='bold')
    plt.xlabel("k (Number of Neighbors)", fontsize=12)
    plt.ylabel("Mean(Local Std / Global Std)", fontsize=12)
    plt.xticks(ks)
    plt.ylim(0, 1.05)
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    
    # Reference line
    plt.axhline(1.0, color="gray", linestyle=":", linewidth=2, alpha=0.5, label="Global Std (Ratio=1.0)")
    
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Save
    out_path = os.path.join(base_dir, f"trend_plot_multiseed_{mut}.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"📊 Saved multi-seed trend plot: {out_path}")
    plt.close()

EOF

echo "=================================================================="
echo "📈 Generating Multi-Seed K-Sweep Trend Plots..."
python "$PLOT_SCRIPT"
echo "✅ All complete! Check the plots in $OUT_BASE"
