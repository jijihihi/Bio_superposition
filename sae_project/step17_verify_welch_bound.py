import os
import glob
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import re
import argparse
from tqdm import tqdm

def calculate_welch_bound(N, d):
    """
    Welch bound: Lower bound on maximum absolute inner product between N unit vectors in d dimensions.
    """
    if N <= d:
        return 0.0
    return np.sqrt((N - d) / (d * (N - 1)))

def plot_heatmaps(data_dict, title_prefix, save_path):
    """
    Plot a grid of heatmaps for the provided data dict.
    data_dict: {config_label: sim_matrix_numpy}
    """
    num_plots = len(data_dict)
    cols = min(4, num_plots)
    rows = (num_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if num_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
        
    for idx, (label, sim_mat) in enumerate(data_dict.items()):
        ax = axes[idx]
        # vmin, vmax를 -1.0 ~ 1.0으로 설정하되, SymLogNorm을 사용하여 0 근처에서 빠르게 Saturation 되도록 만듭니다.
        # linthresh=0.01 이내에서는 선형, 그 밖에서는 로그 스케일로 압축하여 시각적 효과를 극대화합니다.
        norm = mcolors.SymLogNorm(linthresh=0.01, linscale=0.1, vmin=-1.0, vmax=1.0)
        im = ax.imshow(sim_mat, cmap="coolwarm", norm=norm, aspect='auto')
        ax.set_title(label, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_ticks([-1.0, -0.5, -0.1, 0, 0.1, 0.5, 1.0])
        cbar.set_ticklabels(['-1.0', '-0.5', '-0.1', '0', '0.1', '0.5', '1.0'])
        
    for idx in range(len(data_dict), len(axes)):
        axes[idx].axis('off')
        
    plt.suptitle(title_prefix, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()  # 코랩에서 그래프 바로 출력
    plt.close()

def plot_distributions(data_dict, title_prefix, save_path):
    """
    Plot a grid of histograms for the raw and absolute inner products.
    """
    num_plots = len(data_dict)
    cols = min(4, num_plots)
    rows = (num_plots + cols - 1) // cols
    
    # 2배 높이로 생성 (위: Raw, 아래: Absolute)
    fig, axes = plt.subplots(rows * 2, cols, figsize=(5 * cols, 4 * rows * 2))
    
    # axes가 1차원이 되도록 평탄화하되, 구조를 명확히 잡기 위해 2D(rows*2, cols)로 다룹니다.
    axes_2d = axes.reshape(rows * 2, cols) if num_plots > 1 else np.array([[axes[0]], [axes[1]]])
        
    for idx, (label, sim_mat) in enumerate(data_dict.items()):
        row = idx // cols
        col = idx % cols
        
        ax_raw = axes_2d[row * 2, col]
        ax_abs = axes_2d[row * 2 + 1, col]
        
        N = sim_mat.shape[0]
        
        # Extract upper triangular off-diagonal elements
        i, j = np.triu_indices(N, k=1)
        raw_vals = sim_mat[i, j]
        abs_vals = np.abs(raw_vals)
        
        # Parse Welch bound
        match = re.search(r"Welch=([0-9\.]+)", label)
        welch_b = float(match.group(1)) if match else 0.0
        
        # 1. Raw Distribution (-1 to 1)
        sns.histplot(raw_vals, bins=50, kde=True, ax=ax_raw, color='green')
        ax_raw.axvline(welch_b, color='blue', linestyle='--', linewidth=1.5, label='+Welch')
        ax_raw.axvline(-welch_b, color='blue', linestyle='--', linewidth=1.5, label='-Welch')
        ax_raw.set_title(f"[Raw] {label}", fontsize=10)
        ax_raw.set_xlim(-1.0, 1.0)
        ax_raw.set_xlabel("Raw Inner Product")
        ax_raw.set_ylabel("Frequency")
        ax_raw.legend(fontsize=8)
        
        # 2. Absolute Distribution (0 to 1)
        sns.histplot(abs_vals, bins=50, kde=True, ax=ax_abs, color='purple')
        ax_abs.axvline(welch_b, color='blue', linestyle='--', linewidth=2, label=f'Welch ({welch_b:.3f})')
        ax_abs.set_title(f"[Absolute] {label}", fontsize=10)
        ax_abs.set_xlim(0.0, 1.0)
        ax_abs.set_xlabel("Absolute Inner Product")
        ax_abs.set_ylabel("Frequency")
        ax_abs.legend(fontsize=8)
        
    # Hide unused subplots
    for idx in range(num_plots, rows * cols):
        row = idx // cols
        col = idx % cols
        axes_2d[row * 2, col].axis('off')
        axes_2d[row * 2 + 1, col].axis('off')
        
    plt.suptitle(title_prefix, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
    plt.close()
    
def main():
    parser = argparse.ArgumentParser(description="Verify Welch Bound from SAE decoder weights.")
    parser.add_argument("--base_dir", type=str, default="/content/drive/MyDrive/Final_paper/lambda_labs_moco_only",
                        help="Base directory containing MoCo_seed* folders")
    parser.add_argument("--save_dir", type=str, default="/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/welch_bound",
                        help="Directory to save the plots and aggregated csv")
    # Colab(Jupyter) 환경에서 셀 실행 시 발생하는 argument 에러 방지
    import sys
    if 'ipykernel' in sys.modules:
        args, _ = parser.parse_known_args(args=[])
    else:
        args, _ = parser.parse_known_args()

    print(f"🔍 Scanning for checkpoint files in {args.base_dir}...")
    
    # 폴더 이름 대소문자나 띄어쓰기 등이 다를 수 있으므로, 이름 제한을 없애고 깊이(Depth 2)만 맞춥니다.
    pattern = os.path.join(args.base_dir, "*", "*", "*_ep008.pt")
    pt_files = glob.glob(pattern)
    
    if not pt_files:
        print(f"No checkpoint files found in {args.base_dir}!")
        return

    data = []
    
    # Dictionaries to store similarity matrices for heatmaps (only for CNN seed 42)
    heatmaps_dim = {}  # Fixed lambda=800, vary dim
    heatmaps_lam = {}  # Fixed dim=4096, vary lambda
    
    print(f"Found {len(pt_files)} checkpoints. Processing...")
    
    for pt_file in tqdm(pt_files):
        # 정규식을 유연하게 변경하여 폴더 이름이 살짝 달라져도(MoCo_seed -> seed 등) 추출 가능하게 함
        match = re.search(r'(?:MoCo_seed|seed)(\d+).*?SAE_dim(\d+)_lambda(\d+)', pt_file)
        if not match:
            continue
        cnn_seed = int(match.group(1))
        dim = int(match.group(2))
        lam = int(match.group(3))
        
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # GPU가 켜져 있으면 GPU 메모리로, 아니면 CPU로 불러옵니다.
            ckpt = torch.load(pt_file, map_location=device, weights_only=False)
            if "sae" in ckpt:
                state_dict = ckpt["sae"]
            else:
                state_dict = ckpt
                
            if 'W_dec' not in state_dict or 'usage_ema' not in state_dict:
                continue
                
            W_dec = state_dict['W_dec'] # shape (d_sae, d_in)
            usage_ema = state_dict['usage_ema']
            
            # 1. Filter alive neurons (usage >= 1e-5)
            alive_mask = usage_ema >= 1e-5
            W_alive = W_dec[alive_mask]
            
            N_alive = W_alive.shape[0]
            d_in = W_alive.shape[1]
            
            if N_alive <= 1:
                continue # Cannot calculate inner product of 1 vector
                
            # 2. Normalize rows
            W_alive = W_alive / W_alive.norm(dim=1, keepdim=True).clamp_min(1e-12)
            
            # 3. Calculate inner products (GPU 연산 시 초고속 처리)
            sim_raw = torch.mm(W_alive, W_alive.t())
            
            # Heatmap 저장을 위해 대각선(1.0)이 보존된 원본 복사본 생성
            sim_heatmap = sim_raw.clone()
            
            # Mask diagonal (set to 0 so it doesn't affect Max IP computation)
            sim_raw.fill_diagonal_(0.0)
            sim_abs = sim_raw.abs()
            
            # 4. Extract metrics
            max_ip = sim_abs.max().item()
            
            # Upper triangular indices for mean and std (excluding diagonal)
            triu_indices = torch.triu_indices(N_alive, N_alive, offset=1)
            off_diag_vals = sim_abs[triu_indices[0], triu_indices[1]]
            
            mean_ip = off_diag_vals.mean().item()
            std_ip = off_diag_vals.std().item()
            
            welch_b = calculate_welch_bound(N_alive, d_in)
            
            # =========================================================================
            # 5. Concept Vector Weight Distribution Fitting (Power-law vs Alternatives)
            # =========================================================================
            try:
                import powerlaw
                # Get all weights of alive concept vectors, take absolute value
                # numpy 연산을 위해 .cpu().numpy() 사용
                weights = np.abs(W_alive.cpu().numpy().flatten())
                weights = weights[weights > 1e-7] # Exclude zeros
                
                # Subsample to 5k to ensure ultra-fast fitting (50k is still too slow for exhaustive xmin search in Python)
                if len(weights) > 5000:
                    np.random.seed(42)
                    weights = np.random.choice(weights, size=5000, replace=False)
                    
                # Fit the distribution
                # We suppress output and warnings
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit = powerlaw.Fit(weights, verbose=False)
                    
                alpha = fit.power_law.alpha
                xmin = fit.power_law.xmin
                
                # Compare Power Law vs Exponential (R > 0 means Power Law is a better fit)
                R_exp, p_exp = fit.distribution_compare('power_law', 'exponential', normalized_ratio=True)
                # Compare Power Law vs Lognormal
                R_log, p_log = fit.distribution_compare('power_law', 'lognormal', normalized_ratio=True)
            except Exception as e:
                # If powerlaw is not installed or fitting fails, fill with NaNs
                alpha, xmin, R_exp, p_exp, R_log, p_log = [np.nan] * 6
                # print(f"⚠️  Statistical fitting skipped/failed: {e}")
            
            data.append({
                "CNN_Seed": cnn_seed,
                "Dimension": dim,
                "Lambda": lam,
                "N_Alive": N_alive,
                "Max_IP": max_ip,
                "Mean_IP": mean_ip,
                "Std_IP": std_ip,
                "Welch_Bound": welch_b,
                "PL_Alpha": alpha,
                "PL_xmin": xmin,
                "R_vs_Exp": R_exp,
                "p_vs_Exp": p_exp,
                "R_vs_Lognormal": R_log,
                "p_vs_Lognormal": p_log
            })
            
            # Save heatmap for CNN seed 42
            if cnn_seed == 42:
                sim_np = sim_heatmap.cpu().numpy()
                label = f"d={dim}, λ={lam}\nAlive={N_alive}, Max IP={max_ip:.3f}, Welch={welch_b:.3f}\nMean IP={mean_ip:.4f}±{std_ip:.4f}"
                
                if lam == 800 or (dim == 600 and lam == 50):
                    heatmaps_dim[dim] = (label, sim_np)
                if dim == 4096:
                    heatmaps_lam[lam] = (label, sim_np)
                    
            # Memory management
            del sim_raw, sim_heatmap, sim_abs, off_diag_vals, W_alive, W_dec, state_dict
            
        except Exception as e:
            print(f"Error processing {pt_file}: {e}")

    if not data:
        print("No valid data found to plot!")
        return

    df_all = pd.DataFrame(data)
    
    # Save CSV
    os.makedirs(args.save_dir, exist_ok=True)
    csv_path = os.path.join(args.save_dir, "welch_bound_results.csv")
    df_all.to_csv(csv_path, index=False)
    print(f"\nAggregated metrics saved to {csv_path}")
    
    # 코랩 셀 바로 아래에 CSV 데이터프레임 시각화
    try:
        from IPython.display import display
        print("\n📊 [결과 데이터프레임 미리보기]")
        display(df_all.head(10)) # 데이터가 많을 수 있으니 상위 10개만 출력, 혹은 전체 출력
    except ImportError:
        print("\n📊 [결과 데이터프레임 미리보기]")
        print(df_all.head(10))

    sns.set_theme(style="whitegrid")

    # =========================================================================
    # Trend Plot 1: Varying Dimension (Lambda = 800)
    # =========================================================================
    df_dim = df_all[(df_all["Lambda"] == 800) | ((df_all["Dimension"] == 600) & (df_all["Lambda"] == 50))].copy()
    if not df_dim.empty:
        df_dim = df_dim.sort_values(by="Dimension")
        df_dim["Config"] = df_dim["Dimension"].astype(str)
        
        plt.figure(figsize=(9, 6))
        
        # Plot Max IP (mean and std across seeds)
        sns.pointplot(data=df_dim, x="Config", y="Max_IP", color="red", label="Max Inner Product (Empirical)", capsize=.1, errorbar="sd")
        
        # Plot Welch Bound (Since N_alive might vary slightly per seed, we plot the mean Welch Bound)
        sns.pointplot(data=df_dim, x="Config", y="Welch_Bound", color="blue", linestyles="--", markers="X", label="Theoretical Welch Bound", errorbar=None)
        
        plt.title("Disentanglement vs Dimension (Fixed λ=800)", fontsize=14, fontweight='bold')
        plt.xlabel("Dimension (d_sae)", fontsize=12)
        plt.ylabel("Absolute Inner Product", fontsize=12)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(args.save_dir, "Trend_MaxIP_vs_Dimension.png"), dpi=300)
        plt.show() # 코랩에서 그래프 바로 출력
        plt.close()
        
    # =========================================================================
    # Trend Plot 2: Varying Lambda (Dimension = 4096)
    # =========================================================================
    df_lam = df_all[df_all["Dimension"] == 4096].copy()
    if not df_lam.empty:
        df_lam = df_lam.sort_values(by="Lambda")
        df_lam["Lambda_str"] = df_lam["Lambda"].astype(str)
        
        plt.figure(figsize=(9, 6))
        
        # Plot Max IP
        sns.pointplot(data=df_lam, x="Lambda_str", y="Max_IP", color="red", label="Max Inner Product (Empirical)", capsize=.1, errorbar="sd")
        
        # Plot Welch Bound
        sns.pointplot(data=df_lam, x="Lambda_str", y="Welch_Bound", color="blue", linestyles="--", markers="X", label="Theoretical Welch Bound", errorbar=None)
        
        plt.title("Disentanglement vs Sparsity (Fixed d=4096)", fontsize=14, fontweight='bold')
        plt.xlabel("Lambda (Sparsity Coeff)", fontsize=12)
        plt.ylabel("Absolute Inner Product", fontsize=12)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(args.save_dir, "Trend_MaxIP_vs_Lambda.png"), dpi=300)
        plt.show() # 코랩에서 그래프 바로 출력
        plt.close()

    # =========================================================================
    # Heatmaps (CNN Seed 42)
    # =========================================================================
    if heatmaps_dim:
        sorted_dims = sorted(heatmaps_dim.keys())
        plot_dict = {heatmaps_dim[dim][0]: heatmaps_dim[dim][1] for dim in sorted_dims}
        plot_heatmaps(plot_dict, "Concept Vectors Inner Product Heatmaps vs Dimension (Seed 42)", 
                      os.path.join(args.save_dir, "Heatmaps_vs_Dimension.png"))
        plot_distributions(plot_dict, "Concept Vectors Absolute IP Distribution vs Dimension (Seed 42)", 
                      os.path.join(args.save_dir, "Distributions_vs_Dimension.png"))
                      
    if heatmaps_lam:
        sorted_lams = sorted(heatmaps_lam.keys())
        plot_dict = {heatmaps_lam[lam][0]: heatmaps_lam[lam][1] for lam in sorted_lams}
        plot_heatmaps(plot_dict, "Concept Vectors Inner Product Heatmaps vs Sparsity (Seed 42)", 
                      os.path.join(args.save_dir, "Heatmaps_vs_Lambda.png"))
        plot_distributions(plot_dict, "Concept Vectors Absolute IP Distribution vs Sparsity (Seed 42)", 
                      os.path.join(args.save_dir, "Distributions_vs_Lambda.png"))

    # =========================================================================
    # 8개 Seed 평균 요약 테이블 출력
    # =========================================================================
    print("\n📊 [Configuration별 내적 평균 및 분산 요약 (8개 Seed 평균)]")
    agg_df = df_all.groupby(["Dimension", "Lambda"]).agg({
        "N_Alive": ["mean", "std"],
        "Mean_IP": ["mean", "std"],
        "Std_IP": ["mean", "std"],
        "Max_IP": ["mean", "std"]
    }).round(4)
    
    try:
        from IPython.display import display
        display(agg_df)
    except:
        print(agg_df)

if __name__ == "__main__":
    main()
