import glob
import json
import os
import re

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    base_dir = (
        "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/knn_eval_results"
    )
    plots_dir = os.path.join(base_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    plt.rcParams["svg.fonttype"] = "none"

    # Dictionary to store results: dict[config][weight][k] = [seed_vals...]
    # config like "SAE_d8192_lam800", weight like "uniform" or "inv_sq"
    results_acc = {}
    results_f1 = {}

    # Traverse all seed directories
    seed_dirs = glob.glob(os.path.join(base_dir, "seed*"))
    for sdir in seed_dirs:
        # e.g., sdir = .../knn_eval_results/seed42
        seed_match = re.search(r"seed(\d+)", os.path.basename(sdir))
        if not seed_match:
            continue
        seed = int(seed_match.group(1))

        # Look for evaluation folders
        eval_dirs = glob.glob(os.path.join(sdir, "*"))
        for edir in eval_dirs:
            dirname = os.path.basename(edir)

            # Parse dirname to get config and weight
            # e.g., SAE_d8192_lam800_uniform
            # e.g., CNN_stage5_out_all (if we evaluated CNN, weight might be missing)

            if "SAE" in dirname:
                match = re.search(
                    r"(SAE_d\d+_lam\d+(?:_norm_\w+|))_(uniform|inv_sq)", dirname
                )
                if match:
                    config = match.group(1)
                    weight = match.group(2)
                else:
                    continue
            else:
                # For CNN, we might just have CNN_stage5_out
                config = dirname
                weight = "uniform"  # default assumption if not explicitly named

            # Read JSON
            json_files = glob.glob(os.path.join(edir, "eval_results_*.json"))
            if not json_files:
                continue

            with open(json_files[0], "r") as f:
                data = json.load(f)

            knn_data = data.get("knn", [])
            for res in knn_data:
                k = res["k"]
                acc = res["accuracy"]
                f1 = res["macro_f1"]

                if config not in results_acc:
                    results_acc[config] = {}
                    results_f1[config] = {}
                if weight not in results_acc[config]:
                    results_acc[config][weight] = {}
                    results_f1[config][weight] = {}
                if k not in results_acc[config][weight]:
                    results_acc[config][weight][k] = []
                    results_f1[config][weight][k] = []

                results_acc[config][weight][k].append(acc)
                results_f1[config][weight][k].append(f1)

    if not results_acc:
        print("No evaluation results found.")
        return

    # 1. Generate CSV summary
    records = []
    configs = sorted(list(results_acc.keys()))

    # Collect all unique weights and ks
    weights = set()
    ks = set()
    for c in configs:
        for w in results_acc[c].keys():
            weights.add(w)
            for k in results_acc[c][w].keys():
                ks.add(k)

    weights = sorted(list(weights))
    ks = sorted(list(ks))

    for config in configs:
        for w in weights:
            if w not in results_acc[config]:
                continue
            for k in ks:
                if k not in results_acc[config][w]:
                    continue

                accs = results_acc[config][w][k]
                f1s = results_f1[config][w][k]

                records.append(
                    {
                        "Config": config,
                        "Weight": w,
                        "K": k,
                        "Num_Seeds": len(accs),
                        "Accuracy_Mean": np.mean(accs),
                        "Accuracy_Std": np.std(accs),
                        "Macro_F1_Mean": np.mean(f1s),
                        "Macro_F1_Std": np.std(f1s),
                    }
                )

    df = pd.DataFrame(records)
    csv_path = os.path.join(plots_dir, "knn_evaluation_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV Summary: {csv_path}")

    # 2. Generate Heatmaps for all K values
    for target_k in ks:
        # Prepare heatmap data
        heatmap_data = pd.DataFrame(index=configs, columns=weights, dtype=float)

        for config in configs:
            for w in weights:
                if w in results_acc[config] and target_k in results_acc[config][w]:
                    val = np.mean(results_acc[config][w][target_k])
                    heatmap_data.loc[config, w] = val

        heatmap_data = heatmap_data.dropna(how="all").dropna(axis=1, how="all")

        if heatmap_data.empty:
            continue

        fig, ax = plt.subplots(figsize=(6, 4 + 0.5 * len(configs)))

        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".4f",
            cmap="YlGnBu",
            cbar_kws={"label": "Mean Accuracy"},
            ax=ax,
        )

        ax.set_title(
            f"KNN Accuracy Heatmap (K={target_k})",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )
        ax.set_xlabel("KNN Weight Strategy", fontsize=12)
        ax.set_ylabel("Configuration", fontsize=12)
        plt.tight_layout()

        out_path = os.path.join(plots_dir, f"knn_accuracy_heatmap_k{target_k}.svg")
        fig.savefig(out_path, dpi=300, format="svg")
        print(f"Saved Heatmap: {out_path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
