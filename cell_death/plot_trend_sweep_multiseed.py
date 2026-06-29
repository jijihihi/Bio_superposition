import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory containing the sweep results")
    return parser.parse_args()

def plot_metric(mut, ks, cnn_data, sae_data, dims, ylabel, title, save_path):
    plt.figure(figsize=(8, 6))
    
    # Plot CNN
    cnn_means = [np.nanmean(cnn_data[k]) if cnn_data[k] else np.nan for k in ks]
    cnn_stds = [np.nanstd(cnn_data[k]) if cnn_data[k] else 0 for k in ks]
    plt.errorbar(ks, cnn_means, yerr=cnn_stds, fmt='-o', color='black', label='CNN (Baseline)', linewidth=2, capsize=5)
    
    # Plot SAEs
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(dims)))
    for d, color in zip(dims, colors):
        sae_means = [np.nanmean(sae_data[d][k]) if sae_data[d][k] else np.nan for k in ks]
        if np.isnan(sae_means).all():
            continue
        sae_stds = [np.nanstd(sae_data[d][k]) if sae_data[d][k] else 0 for k in ks]
        plt.errorbar(ks, sae_means, yerr=sae_stds, fmt='--s', color=color, label=f'SAE (d={d})', alpha=0.8, capsize=4)
        
    plt.xlabel("k (Neighbors)", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks(ks)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, format="svg")
    plt.close()

def main():
    args = parse_args()
    base_dir = args.base_dir
    
    tables_dir = os.path.join(base_dir, "tables")
    plots_dir = os.path.join(base_dir, "plots")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    seeds = [42, 87, 95, 123, 124, 256, 445, 457]
    dims = [1024, 2048, 4096, 8192]
    lam = 800
    mutations = ["SNCA", "GBA", "LRRK2"]
    ks = [5, 10, 15, 20, 25]

    for mut in mutations:
        cnn_std_data = {k: [] for k in ks}
        sae_std_data = {d: {k: [] for k in ks} for d in dims}

        cnn_morans_data = {k: [] for k in ks}
        sae_morans_data = {d: {k: [] for k in ks} for d in dims}

        for seed in seeds:
            cnn_extracted = False
            for d in dims:
                json_path = os.path.join(base_dir, f"seed_{seed}", f"d{d}_lam{lam}", "local_linearity_results.json")
                if not os.path.exists(json_path):
                    continue

                with open(json_path, "r") as f:
                    data = json.load(f)

                for k in ks:
                    for res in data["results"]:
                        if res["mutation"] == mut and res["k"] == k:
                            if res["source"] == "SAE":
                                sae_std_data[d][k].append(res["mean_ratio"])
                                if "morans_I" in res and res["morans_I"] is not None:
                                    sae_morans_data[d][k].append(res["morans_I"])
                            elif res["source"] == "CNN" and not cnn_extracted:
                                cnn_std_data[k].append(res["mean_ratio"])
                                if "morans_I" in res and res["morans_I"] is not None:
                                    cnn_morans_data[k].append(res["morans_I"])

                cnn_extracted = True

        row_names = ["CNN (Baseline)"] + [f"SAE (d={d})" for d in dims]
        col_names = [f"k={k}" for k in ks]

        def build_dataframe(sae_source, cnn_source, stat_func):
            matrix = []
            cnn_row = []
            for k in ks:
                vals = cnn_source[k]
                cnn_row.append(stat_func(vals) if vals else np.nan)
            matrix.append(cnn_row)

            for d in dims:
                sae_row = []
                for k in ks:
                    vals = sae_source[d][k]
                    sae_row.append(stat_func(vals) if vals else np.nan)
                matrix.append(sae_row)

            return pd.DataFrame(matrix, index=row_names, columns=col_names)

        # 1. Local Std Ratio
        df_std_mean = build_dataframe(sae_std_data, cnn_std_data, np.mean)
        df_std_median = build_dataframe(sae_std_data, cnn_std_data, np.median)

        df_std_mean.to_csv(os.path.join(tables_dir, f"table_knn_std_mean_{mut}.csv"))
        df_std_median.to_csv(os.path.join(tables_dir, f"table_knn_std_median_{mut}.csv"))

        plot_metric(
            mut, ks, cnn_std_data, sae_std_data, dims,
            ylabel="Local Std Ratio (Lower is better)",
            title=f"KNN Local Linearity: {mut}",
            save_path=os.path.join(plots_dir, f"plot_knn_std_{mut}.svg")
        )

        # 2. Moran's I
        has_morans = any(sae_morans_data[d][k] for d in dims for k in ks) or any(cnn_morans_data[k] for k in ks)
        if has_morans:
            df_moran_mean = build_dataframe(sae_morans_data, cnn_morans_data, np.mean)
            df_moran_median = build_dataframe(sae_morans_data, cnn_morans_data, np.median)

            df_moran_mean.to_csv(os.path.join(tables_dir, f"table_morans_I_mean_{mut}.csv"))
            df_moran_median.to_csv(os.path.join(tables_dir, f"table_morans_I_median_{mut}.csv"))

            plot_metric(
                mut, ks, cnn_morans_data, sae_morans_data, dims,
                ylabel="Moran's I (Higher is better)",
                title=f"Moran's I Autocorrelation: {mut}",
                save_path=os.path.join(plots_dir, f"plot_morans_I_{mut}.svg")
            )

if __name__ == "__main__":
    main()
