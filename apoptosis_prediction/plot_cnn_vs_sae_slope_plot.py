# ==============================================================================
# CNN GAP vs SAE Slope Chart
#
# Dot plot comparing CNN GAP and SAE feature vectors for cell death
# prediction. Per-seed mean R² dots, grand mean in red.
# Mann-Whitney U test + rank-biserial r on individual CV folds (independent samples).
# One figure per mutation, saved as SVG/PNG/PDF.
#
# Usage (Colab):
#   import sys
#   sys.argv = [
#       "plot_cnn_vs_sae",
#       "--cnn_results_dir", "/content/drive/MyDrive/.../apoptosis_r2_results",
#       "--sae_results_dir", "/content/drive/MyDrive/.../apoptosis_r2_results/SAE_vector",
#       "--cnn_config", "MoCo_l2norm",
#       "--cnn_layer", "stage5_out",
#       "--sae_l2norm", "l2norm",
#       "--model", "XGBoost",
#   ]
#   from apoptosis_prediction.plot_cnn_vs_sae import main
#   main()
# ==============================================================================



# 변경 전	                                변경 후
# Wilcoxon signed-rank (짝짓기 부적절)	Mann-Whitney U (독립 표본)
# rank-based pairing → 연결선	       연결선 제거 (independent)
# r = Z/√N	                          rank-biserial r = 1 - 2U/(n₁·n₂)


import os
import sys
import csv
import argparse
from collections import defaultdict

import numpy as np

import matplotlib
if "google.colab" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import mannwhitneyu

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
sns.set_style("ticks")


# ==============================================================================
# Constants
# ==============================================================================
GROUPS_OF_INTEREST = ["SNCA only", "GBA only", "LRRK2 only"]
GENE_LABELS = {"SNCA only": "SNCA", "GBA only": "GBA", "LRRK2 only": "LRRK2"}

COLORS = {
    "CNN":  "#5B8DB8",   # steel blue
    "SAE":  "#E07B54",   # warm coral
    "mean": "#D32F2F",   # red for mean
}


# ==============================================================================
# Read CSV helpers
# ==============================================================================
def read_cnn_folds(csv_path):
    """Read aggregated_r2_all_folds.csv (CNN GAP)."""
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "config": row["Config"],
                "seed": int(row["Seed"]),
                "model": row["Model"],
                "layer": row["Layer"],
                "group": row["Group"],
                "fold_idx": int(row["Fold_idx"]),
                "r2": float(row["R2"]),
            })
    return rows


def read_sae_folds(csv_path):
    """Read sae_r2_all_folds.csv (SAE vector)."""
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "sae_seed": int(row["SAE_Seed"]),
                "l2_norm": row["GAP_L2_Norm"],
                "filter": row["Filter"],
                "model": row["Model"],
                "group": row["Group"],
                "fold_idx": int(row["Fold_idx"]),
                "r2": float(row["R2"]),
            })
    return rows


# ==============================================================================
# Build per-seed mean R² arrays
# ==============================================================================
def build_seed_means(folds, group_key_fn, seed_key, fold_key="fold_idx"):
    """
    Group folds by seed, compute mean R² per seed.
    Returns dict: group -> {seed: mean_r2}
    """
    seed_folds = defaultdict(lambda: defaultdict(list))
    for row in folds:
        grp = row["group"]
        s = row[seed_key]
        seed_folds[grp][s].append(row["r2"])

    result = {}
    for grp, seed_dict in seed_folds.items():
        result[grp] = {s: np.mean(vals) for s, vals in seed_dict.items()}
    return result


def pval_to_stars(p):
    if p < 0.001:   return "***"
    elif p < 0.01:  return "**"
    elif p < 0.05:  return "*"
    else:           return "ns"


# ==============================================================================
# Main plot
# ==============================================================================
MODELS = ["Ridge", "XGBoost"]
MODEL_COLORS = {
    "Ridge":   "#5A9BD5",   # blue
    "XGBoost": "#E86830",   # orange
}


def plot_cnn_vs_sae(cnn_folds, sae_folds, output_dir,
                    cnn_config, cnn_layer, sae_l2norm, sae_filter):
    """
    Dot plot: CNN GAP vs SAE, both Ridge and XGBoost on shared x-axis.
    CNN = circles (○), SAE = X markers (×).
    Each model has its own color for dots and connecting mean line.
    MWU test on individual CV folds (independent samples).
    """

    os.makedirs(output_dir, exist_ok=True)

    for grp in GROUPS_OF_INTEREST:
        gene_label = GENE_LABELS[grp]

        x_cnn, x_sae = 0, 1
        fig, ax = plt.subplots(figsize=(4.5, 5.2))
        all_y_vals = []
        has_data = False
        legend_handles = []

        for m_idx, model in enumerate(MODELS):
            color = MODEL_COLORS[model]

            # Filter folds
            cnn_f = [r for r in cnn_folds
                     if r["config"] == cnn_config
                     and r["layer"] == cnn_layer
                     and r["model"] == model
                     and r["group"] == grp]

            sae_f = [r for r in sae_folds
                     if r["l2_norm"] == sae_l2norm
                     and r["model"] == model
                     and r["group"] == grp
                     and (sae_filter is None or r["filter"] == sae_filter)]

            if not cnn_f or not sae_f:
                continue
            has_data = True

            # Per-seed means
            cnn_seed_dict = defaultdict(list)
            for r in cnn_f:
                cnn_seed_dict[r["seed"]].append(r["r2"])
            cnn_means = np.array([np.mean(v) for v in cnn_seed_dict.values()])

            sae_seed_dict = defaultdict(list)
            for r in sae_f:
                sae_seed_dict[r["sae_seed"]].append(r["r2"])
            sae_means = np.array([np.mean(v) for v in sae_seed_dict.values()])

            # Jitter (slightly offset per model to avoid overlap)
            offset = -0.05 + m_idx * 0.10  # Ridge left, XGBoost right
            j_cnn = np.random.default_rng(42 + m_idx).uniform(-0.05, 0.05, size=len(cnn_means))
            j_sae = np.random.default_rng(99 + m_idx).uniform(-0.05, 0.05, size=len(sae_means))

            # CNN = circles (○)
            ax.scatter(x_cnn + offset + j_cnn, cnn_means, s=35, color=color,
                       alpha=0.6, edgecolors="white", linewidths=0.4,
                       marker="o", zorder=4)
            # SAE = X markers (×)
            ax.scatter(x_sae + offset + j_sae, sae_means, s=40, color=color,
                       alpha=0.6, edgecolors=color, linewidths=1.0,
                       marker="X", zorder=4)

            # Grand means + connecting line
            gc = cnn_means.mean()
            gs = sae_means.mean()

            line_h, = ax.plot([x_cnn + offset, x_sae + offset], [gc, gs],
                              color=color, linewidth=2.5, zorder=6,
                              solid_capstyle="round", label=model)
            ax.scatter([x_cnn + offset], [gc], s=80, color=color,
                       edgecolors="white", linewidths=1.5, marker="o", zorder=7)
            ax.scatter([x_sae + offset], [gs], s=85, color=color,
                       edgecolors="white", linewidths=1.5, marker="X", zorder=7)

            # Annotate means (offset text position per model)
            ha_cnn = "right" if m_idx == 0 else "right"
            y_off = 0.008 * (1 if m_idx == 0 else -1)
            ax.text(x_cnn - 0.18, gc + y_off, f"{gc:.3f}", fontsize=8,
                    color=color, fontweight="bold", ha="right", va="center")
            ax.text(x_sae + 0.18, gs + y_off, f"{gs:.3f}", fontsize=8,
                    color=color, fontweight="bold", ha="left", va="center")

            all_y_vals.extend(cnn_means.tolist())
            all_y_vals.extend(sae_means.tolist())
            legend_handles.append(line_h)

            # ── MWU on individual folds ──
            cnn_all = np.array([r["r2"] for r in cnn_f])
            sae_all = np.array([r["r2"] for r in sae_f])
            n1, n2 = len(cnn_all), len(sae_all)

            if n1 >= 5 and n2 >= 5:
                U_stat, pval = mannwhitneyu(sae_all, cnn_all, alternative="two-sided")
                # Rank-biserial correlation: positive = SAE higher (first arg)
                r_rb = (2 * U_stat) / (n1 * n2) - 1

                p_str = f"p<0.001" if pval < 0.001 else f"p={pval:.3f}"
                # Place stats at top, stacked per model
                y_pos = 0.97 - m_idx * 0.06
                stat_text = f"{model}: {p_str}, r\u1D63\u1D47={r_rb:.2f}"
                ax.text(0.98, y_pos, stat_text, transform=ax.transAxes,
                        fontsize=7.5, fontweight="bold", color=color,
                        ha="right", va="top")

                print(f"  {gene_label} {model}: MWU U={U_stat:.1f}, p={pval:.6f}, "
                      f"r_rb={r_rb:.3f}, n_cnn={n1}, n_sae={n2}")

        if not has_data:
            plt.close(fig)
            print(f"  ⚠ {gene_label}: no data — skipping")
            continue

        # ── Y-axis: include 0 if negatives exist ──
        all_y = np.array(all_y_vals)
        data_min = min(all_y.min(), 0)
        data_max = all_y.max()
        margin = (data_max - data_min) * 0.12
        ax.set_ylim(data_min - margin * 0.5, data_max + margin * 2)

        # Zero line
        ax.axhline(0, color="#CCCCCC", linewidth=0.8, linestyle="-", zorder=1)

        # ── Formatting ──
        ax.set_xticks([x_cnn, x_sae])
        ax.set_xticklabels(["CNN GAP", "SAE"], fontsize=13, fontweight="bold")
        ax.set_ylabel("R² (Cell Death Prediction)", fontsize=10, fontweight="bold")
        ax.set_title(f"{gene_label} — CNN GAP vs SAE",
                     fontsize=13, fontweight="bold", pad=10)
        ax.set_xlim(-0.5, 1.5)
        ax.grid(axis="y", alpha=0.15, zorder=0)
        ax.set_axisbelow(True)
        ax.legend(handles=legend_handles, loc="upper left", fontsize=9,
                  framealpha=0.8, edgecolor="#DDDDDD")
        sns.despine(ax=ax)

        fig.tight_layout()

        # ── Save per-mutation ──
        filt_tag = f"_{sae_filter}" if sae_filter else ""
        base = f"cnn_vs_sae_{gene_label}{filt_tag}"
        for ext in ["pdf", "png", "svg"]:
            path = os.path.join(output_dir, f"{base}.{ext}")
            fig.savefig(path, dpi=300 if ext != "png" else 200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {gene_label}: {base}.svg / .png / .pdf")


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="CNN GAP vs SAE dot plot for cell death prediction"
    )
    parser.add_argument("--cnn_results_dir", type=str, required=True,
                        help="Dir with aggregated_r2_all_folds.csv")
    parser.add_argument("--sae_results_dir", type=str, required=True,
                        help="Dir with sae_r2_all_folds.csv")
    parser.add_argument("--output_dir", type=str, default="",
                        help="Output dir (default: sae_results_dir)")
    parser.add_argument("--cnn_config", type=str, default="MoCo_l2norm",
                        help="CNN config (default: MoCo_l2norm)")
    parser.add_argument("--cnn_layer", type=str, default="stage5_out",
                        help="CNN layer (default: stage5_out)")
    parser.add_argument("--sae_l2norm", type=str, default="l2norm", # --sae_l2norm l2norm = SAE feature vector를 예측 전에 L2 normalization sclens 효과 보정. 
                        help="SAE L2 norm condition (default: l2norm)")
    parser.add_argument("--sae_filter", type=str, default="no_filter",
                        help="SAE filter label (default: None = all)")
    args = parser.parse_args()

    # Read CSVs
    cnn_csv = os.path.join(args.cnn_results_dir, "aggregated_r2_all_folds.csv")
    sae_csv = os.path.join(args.sae_results_dir, "sae_r2_all_folds.csv")

    if not os.path.exists(cnn_csv):
        print(f"ERROR: CNN CSV not found: {cnn_csv}")
        sys.exit(1)
    if not os.path.exists(sae_csv):
        print(f"ERROR: SAE CSV not found: {sae_csv}")
        sys.exit(1)

    cnn_folds = read_cnn_folds(cnn_csv)
    sae_folds = read_sae_folds(sae_csv)
    print(f"\n  CNN folds: {len(cnn_folds)}")
    print(f"  SAE folds: {len(sae_folds)}")

    output_dir = args.output_dir or args.sae_results_dir

    plot_cnn_vs_sae(
        cnn_folds, sae_folds, output_dir,
        args.cnn_config, args.cnn_layer, args.sae_l2norm, args.sae_filter,
    )

    print("\n  DONE")


if __name__ == "__main__":
    main()

