import os
import glob
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
        # To avoid massive file sizes and rendering issues, we use imshow directly.
        # vmax=1.0, vmin=-1.0 for cosine similarity
        im = ax.imshow(sim_mat, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect='auto')
        ax.set_title(label, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
    for idx in range(len(data_dict), len(axes)):
        axes[idx].axis('off')
        
    plt.suptitle(title_prefix, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
def main():
    parser = argparse.ArgumentParser(description="Verify Welch Bound from SAE decoder weights.")
    parser.add_argument("--base_dir", type=str, default="/home/ubuntu/model-east3/outputs",
                        help="Base directory containing MoCo_seed* folders")
    parser.add_argument("--save_dir", type=str, default="./",
                        help="Directory to save the plots and aggregated csv")
    args = parser.parse_args()

    pattern = os.path.join(args.base_dir, "MoCo_seed*", "SAE_dim*_lambda*_seed48_no_L2norm_loss", "*_ep008.pt")
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
        match = re.search(r'MoCo_seed(\d+)[/\\]SAE_dim(\d+)_lambda(\d+)_seed48', pt_file)
        if not match:
            continue
        cnn_seed = int(match.group(1))
        dim = int(match.group(2))
        lam = int(match.group(3))
        
        try:
            # map_location="cpu" prevents VRAM leaks if script runs on GPU server
            state_dict = torch.load(pt_file, map_location="cpu", weights_only=True)
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
                
            # 2. Normalize rows (they should already be normalized, but we double-check)
            W_alive = W_alive / W_alive.norm(dim=1, keepdim=True).clamp_min(1e-12)
            
            # 3. Calculate inner products
            sim_raw = torch.mm(W_alive, W_alive.t())
            
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
            
            data.append({
                "CNN_Seed": cnn_seed,
                "Dimension": dim,
                "Lambda": lam,
                "N_Alive": N_alive,
                "Max_IP": max_ip,
                "Mean_IP": mean_ip,
                "Std_IP": std_ip,
                "Welch_Bound": welch_b
            })
            
            # Save heatmap for CNN seed 42
            if cnn_seed == 42:
                sim_np = sim_raw.numpy()
                label = f"d={dim}, λ={lam}\nAlive={N_alive}, Max IP={max_ip:.3f}, Welch={welch_b:.3f}\nMean IP={mean_ip:.4f}±{std_ip:.4f}"
                
                if lam == 800 or (dim == 600 and lam == 50):
                    heatmaps_dim[dim] = (label, sim_np)
                if dim == 4096:
                    heatmaps_lam[lam] = (label, sim_np)
                    
            # Memory management
            del sim_raw, sim_abs, off_diag_vals, W_alive, W_dec, state_dict
            
        except Exception as e:
            print(f"Error processing {pt_file}: {e}")

    if not data:
        print("No valid data found to plot!")
        return

    df_all = pd.DataFrame(data)
    
    # Save CSV
    csv_path = os.path.join(args.save_dir, "welch_bound_results.csv")
    df_all.to_csv(csv_path, index=False)
    print(f"\nAggregated metrics saved to {csv_path}")

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
        plt.close()

    # =========================================================================
    # Heatmaps (CNN Seed 42)
    # =========================================================================
    if heatmaps_dim:
        sorted_dims = sorted(heatmaps_dim.keys())
        plot_dict = {heatmaps_dim[dim][0]: heatmaps_dim[dim][1] for dim in sorted_dims}
        plot_heatmaps(plot_dict, "Concept Vectors Inner Product Heatmaps vs Dimension (Seed 42)", 
                      os.path.join(args.save_dir, "Heatmaps_vs_Dimension.png"))
                      
    if heatmaps_lam:
        sorted_lams = sorted(heatmaps_lam.keys())
        plot_dict = {heatmaps_lam[lam][0]: heatmaps_lam[lam][1] for lam in sorted_lams}
        plot_heatmaps(plot_dict, "Concept Vectors Inner Product Heatmaps vs Sparsity (Seed 42)", 
                      os.path.join(args.save_dir, "Heatmaps_vs_Lambda.png"))

if __name__ == "__main__":
    main()
