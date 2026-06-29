import argparse
import json
import os
import sys

import matplotlib
import numpy as np
import seaborn as sns

_IN_COLAB = "google.colab" in sys.modules
if not _IN_COLAB:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser("Plot Accuracy from Linear Eval Results")
    parser.add_argument("--result_dirs", nargs="+", required=True, help="List of directories containing linear_eval_results.json")
    parser.add_argument("--output_file", type=str, default="accuracy_plot.svg")
    args = parser.parse_args()

    accuracy_list = []
    
    for d in args.result_dirs:
        json_path = os.path.join(d, "linear_eval_results.json")
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                res = json.load(f)
                accuracy_list.append(res["accuracy"])
        else:
            print(f"Warning: {json_path} not found.")
            
    if not accuracy_list:
        print("No valid results found. Exiting.")
        return

    print(f"Accuracies: {accuracy_list}")
    
    mean_val = np.mean(accuracy_list)
    std_val = np.std(accuracy_list, ddof=1) if len(accuracy_list) > 1 else 0.0

    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.family"] = "DejaVu Sans"
    sns.set_style("ticks")

    fig, ax = plt.subplots(figsize=(3, 5))
    plt.grid(True, axis="y", zorder=1, linestyle=":", alpha=0.5)

    sns.boxplot(
        data=accuracy_list, width=0.5, color="#3498db", linewidth=1.5, fliersize=0, zorder=2
    )
    sns.stripplot(data=accuracy_list, color=".25", size=6, jitter=0.1, alpha=0.7, zorder=3)

    text_content = f"Mean: {mean_val:.4f}  ±  Std: {std_val:.4f}"
    plt.title(text_content, fontsize=10, pad=20)
    plt.ylabel("Accuracy", fontsize=12)
    
    # Auto-adjust ylim slightly
    min_acc, max_acc = min(accuracy_list), max(accuracy_list)
    padding = (max_acc - min_acc) * 0.5 if max_acc > min_acc else 0.05
    plt.ylim(min_acc - padding, max_acc + padding)
    
    ax.set_xticks([]) 
    sns.despine()
    plt.tight_layout()
    
    plt.savefig(args.output_file, format="svg", transparent=True)
    print(f"Accuracy plot saved to {args.output_file}")
    plt.close(fig)

if __name__ == "__main__":
    main()
