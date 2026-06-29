import argparse
import glob
import os
import re

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

# --- SVG Save settings ---
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42


class SymmetricPowerNorm(mcolors.Normalize):
    def __init__(self, gamma=1.0, vmin=-1.0, vmax=1.0, clip=False):
        self.gamma = gamma
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        limit = max(abs(self.vmin), abs(self.vmax))
        if limit == 0:
            limit = 1e-12
        # Clip to symmetric limits
        val = np.clip(value, -limit, limit)
        x = val / limit
        exponent = 1.0 / self.gamma if self.gamma > 0 else 1.0
        x_trans = np.sign(x) * (np.abs(x) ** exponent)
        return (x_trans + 1.0) / 2.0

    def inverse(self, value):
        x_trans = value * 2.0 - 1.0
        limit = max(abs(self.vmin), abs(self.vmax))
        if limit == 0:
            limit = 1e-12
        x = np.sign(x_trans) * (np.abs(x_trans) ** self.gamma)
        return x * limit


def plot_heatmaps(data_dict, title_prefix, save_path_base, gamma=1.0):
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
        ax.grid(False) # Add grid(False) before imshow to suppress warning
        # Use symmetric PowerNorm centered at zero to boost contrast of small values
        norm = SymmetricPowerNorm(gamma=gamma, vmin=-0.4, vmax=0.4)
        im = ax.imshow(sim_mat, cmap="coolwarm", norm=norm, aspect="auto")
        ax.set_title(label, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, extend="both")

        # Colorbar 눈금도 -0.4 ~ 0.4에 맞춰서 수정
        cbar.set_ticks([-0.4, -0.2, 0.0, 0.2, 0.4])
        cbar.set_ticklabels(["-0.4", "-0.2", "0.0", "0.2", "0.4"])

    for idx in range(len(data_dict), len(axes)):
        axes[idx].axis("off")

    plt.suptitle(title_prefix, fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path_base + ".png", dpi=300, bbox_inches="tight")
    plt.savefig(save_path_base + ".svg", format="svg", bbox_inches="tight")
    plt.savefig(save_path_base + ".pdf", format="pdf", bbox_inches="tight")
    plt.close()


def plot_distributions(data_dict, title_prefix, save_path_base):
    """
    Plot a grid of histograms for the raw and absolute inner products.
    """
    num_plots = len(data_dict)
    cols = min(4, num_plots)
    rows = (num_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows * 2, cols, figsize=(5 * cols, 4 * rows * 2))

    axes_2d = (
        axes.reshape(rows * 2, cols)
        if num_plots > 1
        else np.array([[axes[0]], [axes[1]]])
    )

    for idx, (label, sim_mat) in enumerate(data_dict.items()):
        row = idx // cols
        col = idx % cols

        ax_raw = axes_2d[row * 2, col]
        ax_abs = axes_2d[row * 2 + 1, col]

        N = sim_mat.shape[0]
        i, j = np.triu_indices(N, k=1)
        raw_vals = np.array(sim_mat[i, j], dtype=np.float64).flatten()
        abs_vals = np.abs(raw_vals)

        # 1. Raw Distribution (-1 to 1)
        sns.histplot(raw_vals, bins=50, ax=ax_raw, color="green")
        ax_raw.set_title(f"[Raw] {label}", fontsize=10)
        ax_raw.set_xlim(-1.0, 1.0)
        ax_raw.set_xlabel("Raw Inner Product")
        ax_raw.set_ylabel("Frequency")

        # 2. Absolute Distribution (0 to 1)
        sns.histplot(abs_vals, bins=50, ax=ax_abs, color="purple")
        ax_abs.set_title(f"[Absolute] {label}", fontsize=10)
        ax_abs.set_xlim(0.0, 1.0)
        ax_abs.set_xlabel("Absolute Inner Product")
        ax_abs.set_ylabel("Frequency")

    for idx in range(num_plots, rows * cols):
        row = idx // cols
        col = idx % cols
        axes_2d[row * 2, col].axis("off")
        axes_2d[row * 2 + 1, col].axis("off")

    plt.suptitle(title_prefix, fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path_base + ".png", dpi=300, bbox_inches="tight")
    plt.savefig(save_path_base + ".svg", format="svg", bbox_inches="tight")
    plt.savefig(save_path_base + ".pdf", format="pdf", bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Superposition Disentanglement from SAE decoder weights."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/content/drive/MyDrive/Final_paper/lambda_labs_moco_only",
    )
    parser.add_argument(
        "--heatmap_gamma",
        type=float,
        default=2.5,
        help="Gamma exponent for heatmap color scaling (power law). Default 2.5 (enhanced contrast)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/caches_per_image_centering/disentanglement",
    )
    parser.add_argument(
        "--target_configs",
        type=str,
        nargs="+",
        default=None,
        help="List of target configs (e.g. 600_50 1024_800) to evaluate. If none, evaluates all.",
    )

    import sys

    if "ipykernel" in sys.modules:
        args, _ = parser.parse_known_args(args=[])
    else:
        args, _ = parser.parse_known_args()

    print(f"🔍 Scanning for checkpoint files in {args.base_dir}...")

    pattern = os.path.join(args.base_dir, "*", "*", "*_ep008.pt")
    pt_files = glob.glob(pattern)

    if not pt_files:
        print(f"No checkpoint files found in {args.base_dir}!")
        return

    data = []

    heatmaps_dim = {}  # Fixed lambda=800, vary dim
    heatmaps_lam = {}  # Fixed dim=4096, vary lambda

    print(f"Found {len(pt_files)} checkpoints. Processing...")

    for pt_file in tqdm(pt_files):
        match = re.search(
            r"(?:MoCo_seed|seed)(\d+).*?SAE_dim(\d+)_lambda(\d+)", pt_file
        )
        if not match:
            continue
        cnn_seed = int(match.group(1))
        dim = int(match.group(2))
        lam = int(match.group(3))

        if args.target_configs:
            config_str = f"{dim}_{lam}"
            if config_str not in args.target_configs:
                continue

        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            ckpt = torch.load(pt_file, map_location=device, weights_only=False)
            if "sae" in ckpt:
                state_dict = ckpt["sae"]
            else:
                state_dict = ckpt

            if "W_dec" not in state_dict or "usage_ema" not in state_dict:
                continue

            W_dec = state_dict["W_dec"]  # shape (d_sae, d_in)
            usage_ema = state_dict["usage_ema"]

            alive_mask = usage_ema >= 1e-5
            W_alive = W_dec[alive_mask]

            N_alive = W_alive.shape[0]
            d_in = W_alive.shape[1]

            if N_alive <= 1:
                continue

            W_alive = W_alive / W_alive.norm(dim=1, keepdim=True).clamp_min(1e-12)

            sim_raw = torch.mm(W_alive, W_alive.t())

            sim_heatmap = sim_raw.clone()

            sim_raw.fill_diagonal_(0.0)
            sim_abs = sim_raw.abs()


            triu_indices = torch.triu_indices(N_alive, N_alive, offset=1)
            off_diag_vals = sim_abs[triu_indices[0], triu_indices[1]]

            mean_ip = off_diag_vals.mean().item()
            std_ip = off_diag_vals.std().item()

            data.append(
                {
                    "CNN_Seed": cnn_seed,
                    "Dimension": dim,
                    "Lambda": lam,
                    "N_Alive": N_alive,
                    "Mean_IP": mean_ip,
                    "Std_IP": std_ip,
                }
            )

            if cnn_seed == 42:
                sim_np = sim_heatmap.cpu().numpy()
                label = f"d={dim}, λ={lam}\nAlive={N_alive}\nMean IP={mean_ip:.4f}±{std_ip:.4f}"

                if lam == 800 or (dim == 600 and lam == 50):
                    heatmaps_dim[dim] = (label, sim_np)
                if dim == 4096:
                    heatmaps_lam[lam] = (label, sim_np)

            del sim_raw, sim_heatmap, sim_abs, off_diag_vals, W_alive, W_dec, state_dict

        except Exception as e:
            print(f"Error processing {pt_file}: {e}")

    if not data:
        print("No valid data found to plot!")
        return

    df_all = pd.DataFrame(data)

    os.makedirs(args.save_dir, exist_ok=True)
    csv_path = os.path.join(args.save_dir, "disentanglement_results.csv")
    df_all.to_csv(csv_path, index=False)
    print(f"\nAggregated metrics saved to {csv_path}")

    sns.set_theme(style="whitegrid")



    # =========================================================================
    # Heatmaps (CNN Seed 42)
    # =========================================================================
    if heatmaps_dim:
        sorted_dims = sorted(heatmaps_dim.keys())
        plot_dict = {heatmaps_dim[dim][0]: heatmaps_dim[dim][1] for dim in sorted_dims}
        plot_heatmaps(
            plot_dict,
            "Concept Vectors Inner Product Heatmaps vs Dimension (Seed 42)",
            os.path.join(args.save_dir, "Heatmaps_vs_Dimension"),
            gamma=args.heatmap_gamma,
        )
        plot_distributions(
            plot_dict,
            "Concept Vectors Absolute IP Distribution vs Dimension (Seed 42)",
            os.path.join(args.save_dir, "Distributions_vs_Dimension"),
        )

    if heatmaps_lam:
        sorted_lams = sorted(heatmaps_lam.keys())
        plot_dict = {heatmaps_lam[lam][0]: heatmaps_lam[lam][1] for lam in sorted_lams}
        plot_heatmaps(
            plot_dict,
            "Concept Vectors Inner Product Heatmaps vs Sparsity (Seed 42)",
            os.path.join(args.save_dir, "Heatmaps_vs_Lambda"),
            gamma=args.heatmap_gamma,
        )
        plot_distributions(
            plot_dict,
            "Concept Vectors Absolute IP Distribution vs Sparsity (Seed 42)",
            os.path.join(args.save_dir, "Distributions_vs_Lambda"),
        )




if __name__ == "__main__":
    main()
