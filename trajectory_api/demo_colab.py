import os
from trajectory_api import (
    load_and_preprocess,
    plot_global_phate,
    plot_global_paga,
    run_pairwise_trajectory,
    plot_feature_trends,
    plot_trajectory_statistics
)

# ==============================================================================
# Colab Trajectory Inference Workflow
# ==============================================================================

# 1. 설정
CACHE_PATH = "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/caches_dynamic_batch_centering/features_cache_gap_refine_out.npz"
APOPTOSIS_CSV = "/content/drive/MyDrive/Final_paper/Data/apoptosis_rate_table.csv"
OUT_DIR = "./trajectory_results"

mutations = ["SNCA", "GBA", "LRRK2"]

# 2. 데이터 로드 및 전처리
print("Loading data...")
adata = load_and_preprocess(
    cache_path=CACHE_PATH,
    apoptosis_csv=APOPTOSIS_CSV,
    norm_method="log_std",
    gap_l2_norm=True,
    filter_modes=["de"],         # DE filter 적용
    de_mutation="SNCA",          # SNCA 기준 DE filter
    de_adj_p=0.05,
    n_subsample=0                # 서브샘플링 없이 전부 사용
)
print(adata)

# 3. Global Analysis
print("Running Global PHATE & PAGA...")
plot_global_phate(adata, out_dir=OUT_DIR, prefix="log_std")
plot_global_paga(adata, out_dir=OUT_DIR, prefix="log_std")

# 4. Pairwise Analysis
for mut in mutations:
    print(f"\n--- Analyzing Pair: Control vs {mut} ---")
    
    # 4-1. Trajectory 계산 및 시각화 (Diffmap, DPT 등)
    adata_pair = run_pairwise_trajectory(
        adata=adata,
        mutation=mut,
        out_dir=OUT_DIR,
        prefix="log_std"
    )
    
    if adata_pair is not None:
        # 4-2. 피처 트렌드 및 히트맵
        plot_feature_trends(adata_pair, mutation=mut, out_dir=OUT_DIR, prefix="log_std")
        
        # 4-3. 통계 및 확산 정도(Spread) 진단
        plot_trajectory_statistics(adata_pair, mutation=mut, out_dir=OUT_DIR, prefix="log_std")

print("\nPipeline complete! Check the output directory.")
