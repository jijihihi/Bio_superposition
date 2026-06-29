# ==============================================================================
# CNN GAP vs SAE Paired Slope Chart (Global R²)
#
# Dot plot comparing CNN GAP and SAE feature vectors for cell death prediction.
# Plots Global R² per seed (n=8). Draws connecting lines for each paired seed.
# Performs Wilcoxon signed-rank test on the paired Global R² values.
#
# Usage (Colab):
#   import sys
#   sys.argv = [
#       "plot_cnn_vs_sae",
#       "--cnn_results_dir", "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/cell_death_r2_results",
#       "--sae_results_dir", "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/caches_per_image_centering/SAE_vector_per_image_centering",
#       "--cnn_config", "MoCo_l2norm",
#       "--cnn_layer", "stage5_out",
#       "--sae_l2norm", "l2norm",
#   ]
#   from cell_death.plot_cnn_vs_sae_slope_plot import main
#   main()
# ==============================================================================
import argparse
import csv
import os
import sys

import matplotlib
import numpy as np

if "google.colab" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
sns.set_style("ticks")

GROUPS_OF_INTEREST = ["SNCA only", "GBA only", "LRRK2 only"]
GENE_LABELS = {"SNCA only": "SNCA", "GBA only": "GBA", "LRRK2 only": "LRRK2"}

COLORS = {
    "CNN": "#f0a74f",  # CNN 각 seed 색상
    "SAE": "#a9789c",  # SAE 각 seed 색상
    "mean_line": "#000000",  # 가운데 평균 연결 선 색상 (검은색)
}
MODELS = ["Ridge", "XGBoost"]


def read_cnn_seeds(csv_path):
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "config": row["Config"],
                    "seed": int(row["Seed"]),
                    "model": row["Model"],
                    "layer": row["Layer"],
                    "group": row["Group"],
                    "global_r2": float(row["R2_mean"]),
                }
            )
    return rows


def read_sae_seeds(csv_path):
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "cnn_seed": int(row["CNN_Seed"]),
                    "l2_norm": row["GAP_L2_Norm"],
                    "filter": row["Filter"],
                    "model": row["Model"],
                    "group": row["Group"],
                    "config": row.get("Config", ""),
                    "global_r2": float(row["R2_mean"]),
                }
            )
    return rows


def plot_cnn_vs_sae(
    cnn_data, sae_data, output_dir, cnn_config, cnn_layer, sae_l2norm, sae_filter, sae_config
):
    os.makedirs(output_dir, exist_ok=True)

    # ── mm -> inch 변환 및 레이아웃 정밀 계산 ──
    mm_to_inch = 1.0 / 25.4

    # [수정] 순수 x축 데이터 영역 너비를 기존 19.087mm에서 15% 늘린 21.950mm로 확장
    ax_w_mm = 21.950
    ax_h_mm = 34.373

    # 긴 이름이 들어올 것을 대비해 주변 여백 및 서브플롯 간격을 넉넉히 확보 (mm 단위)
    margin_left_mm = 22.0
    margin_right_mm = 20.0
    margin_bottom_mm = 14.0
    margin_top_mm = 18.0
    w_space_mm = 42.0  # 축 이름이 길어지므로 플롯 사이 간격을 더 확장

    total_w_mm = margin_left_mm + (3 * ax_w_mm) + (2 * w_space_mm) + margin_right_mm
    total_h_mm = margin_bottom_mm + ax_h_mm + margin_top_mm

    fig_w_in = total_w_mm * mm_to_inch
    fig_h_in = total_h_mm * mm_to_inch

    # 개별 seed 점 크기 (지름 1.969mm로 정밀 고정)
    mm_to_pt = 72.0 / 25.4
    seed_diameter_mm = 1.969
    seed_size_pt2 = (seed_diameter_mm * mm_to_pt) ** 2

    for model in MODELS:
        fig = plt.figure(figsize=(fig_w_in, fig_h_in))
        has_data = False

        for idx, grp in enumerate(GROUPS_OF_INTEREST):
            gene_label = GENE_LABELS[grp]

            current_left_mm = margin_left_mm + idx * (ax_w_mm + w_space_mm)
            current_bottom_mm = margin_bottom_mm

            ax_left = current_left_mm / total_w_mm
            ax_bottom = current_bottom_mm / total_h_mm
            ax_width = ax_w_mm / total_w_mm
            ax_height = ax_h_mm / total_h_mm

            ax = fig.add_axes([ax_left, ax_bottom, ax_width, ax_height])

            color_cnn = COLORS["CNN"]
            color_sae = COLORS["SAE"]

            # [수정] 그래프 자체 내부에서 두 점의 거리를 더 벌리기 위해 좌표 간격을 0~1에서 0~1.5로 확장
            x_cnn, x_sae = 0.0, 1.5

            # 데이터 필터링
            cnn_f = [
                r
                for r in cnn_data
                if r["config"] == cnn_config
                and r["layer"] == cnn_layer
                and r["model"] == model
                and r["group"] == grp
            ]

            sae_f = [
                r
                for r in sae_data
                if r["l2_norm"] == sae_l2norm
                and r["model"] == model
                and r["group"] == grp
                and (sae_filter is None or r["filter"] == sae_filter)
                and (sae_config is None or sae_config == "" or r.get("config", "") == sae_config)
            ]

            if not cnn_f or not sae_f:
                ax.text(
                    0.5,
                    0.5,
                    f"No Data\n{gene_label}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=8,
                )
                sns.despine(ax=ax)
                continue

            cnn_dict = {r["seed"]: r["global_r2"] for r in cnn_f}
            sae_dict = {r["cnn_seed"]: r["global_r2"] for r in sae_f}
            common_seeds = sorted(set(cnn_dict.keys()) & set(sae_dict.keys()))

            if len(common_seeds) < 3:
                ax.text(
                    0.5,
                    0.5,
                    f"Low Seeds\n({len(common_seeds)})",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=8,
                )
                sns.despine(ax=ax)
                continue

            has_data = True
            y_cnn_vals = []
            y_sae_vals = []

            # ── Paired lines and dots ──
            # 넓어진 축 너비 비율에 맞춰 jitter 범위 최적화
            rng = np.random.default_rng(42)
            for seed in common_seeds:
                y_c = cnn_dict[seed]
                y_s = sae_dict[seed]
                y_cnn_vals.append(y_c)
                y_sae_vals.append(y_s)

                j_c = rng.uniform(-0.06, 0.06)
                j_s = rng.uniform(-0.06, 0.06)

                ax.plot(
                    [x_cnn + j_c, x_sae + j_s],
                    [y_c, y_s],
                    color="#AAAAAA",
                    alpha=0.4,
                    linewidth=0.9,
                    zorder=2,
                )

                ax.scatter(
                    x_cnn + j_c,
                    y_c,
                    s=seed_size_pt2,
                    color=color_cnn,
                    alpha=0.85,
                    edgecolors="white",
                    linewidths=0.4,
                    marker="o",
                    zorder=3,
                )
                ax.scatter(
                    x_sae + j_s,
                    y_s,
                    s=seed_size_pt2,
                    color=color_sae,
                    alpha=0.85,
                    edgecolors="white",
                    linewidths=0.4,
                    marker="o",
                    zorder=3,
                )

            y_cnn_vals = np.array(y_cnn_vals)
            y_sae_vals = np.array(y_sae_vals)

            # ── Grand means + thick connecting line ──
            gc = y_cnn_vals.mean()
            gs = y_sae_vals.mean()

            ax.plot(
                [x_cnn, x_sae],
                [gc, gs],
                color=COLORS["mean_line"],
                linewidth=2.3,
                zorder=6,
                solid_capstyle="round",
            )

            # 평균값 마커 흰색 테두리 두께 2배 강화 (linewidths=1.5)
            ax.scatter(
                [x_cnn],
                [gc],
                s=65,
                color=color_cnn,
                edgecolors="white",
                linewidths=1.5,
                marker="o",
                zorder=7,
            )
            ax.scatter(
                [x_sae],
                [gs],
                s=65,
                color=color_sae,
                edgecolors="white",
                linewidths=1.5,
                marker="o",
                zorder=7,
            )

            # 축 너비가 늘어났으므로 숫자 텍스트가 점과 겹치지 않게 가로 오프셋 간격을 0.35로 조정
            ax.text(
                x_cnn - 0.35,
                gc,
                f"{gc:.3f}",
                fontsize=7.5,
                color=color_cnn,
                fontweight="bold",
                ha="right",
                va="center",
            )
            ax.text(
                x_sae + 0.35,
                gs,
                f"{gs:.3f}",
                fontsize=7.5,
                color=color_sae,
                fontweight="bold",
                ha="left",
                va="center",
            )

            # ── Wilcoxon Signed-Rank Test ──
            diff = y_sae_vals - y_cnn_vals
            diff_nz = diff[diff != 0]
            n_nz = len(diff_nz)

            try:
                stat, pval = wilcoxon(diff, alternative="two-sided")
                if n_nz > 0:
                    S = (n_nz * (n_nz + 1)) / 2
                    r_rb = 1.0 - (2.0 * stat) / S
                    if np.mean(diff) < 0:
                        r_rb = -r_rb
                else:
                    r_rb = 0.0
            except ValueError:
                stat, pval, r_rb = 0.0, 1.0, 0.0

            p_str = f"p<0.001" if pval < 0.001 else f"p={pval:.3f}"
            stat_text = f"p={p_str.replace('p=', '')}\nr_rb={r_rb:.2f}"
            ax.text(
                0.5 * (x_cnn + x_sae),
                1.06,
                stat_text,
                transform=ax.transAxes,
                fontsize=7.5,
                fontweight="bold",
                color="#333333",
                ha="center",
                va="bottom",
            )

            # [수정] 잘림 현상 해결을 위한 고정 Y축 범위 및 눈금 상한선 상향 조정 (6개 눈금)
            if gene_label == "SNCA":
                ax.set_ylim(0.4, 0.65)
                ax.set_yticks([0.4, 0.45, 0.5, 0.55, 0.6, 0.65])
            else:  # GBA 및 LRRK2
                ax.set_ylim(0.2, 0.45)
                ax.set_yticks([0.2, 0.25, 0.3, 0.35, 0.4, 0.45])

            ax.set_xticks([x_cnn, x_sae])
            # 실제 사용하실 훨씬 긴 이름을 축 라벨 자리에 그대로 넣어주시면 됩니다.
            ax.set_xticklabels(["CNN GAP", "SAE"], fontsize=9.5, fontweight="bold")

            ax.tick_params(axis="y", labelsize=8.5)

            if idx == 0:
                ax.set_ylabel("Global R²", fontsize=10, fontweight="bold")
            else:
                ax.set_ylabel("")

            ax.set_title(f"{gene_label}", fontsize=11, fontweight="bold", pad=28)

            # [수정] 데이터와 텍스트가 양옆으로 균형 있게 벌어지도록 축 제한값(xlim) 확장 (-0.8, 2.3)
            ax.set_xlim(-0.8, 2.3)
            ax.grid(axis="y", alpha=0.15, zorder=0)
            ax.set_axisbelow(True)
            sns.despine(ax=ax)

        # ── Save ──
        if has_data:
            filt_tag = (
                f"_{sae_filter}" if sae_filter and sae_filter != "no_filter" else ""
            )
            base = f"cnn_vs_sae_paired_mm_extended_{model}{filt_tag}"
            for ext in ["pdf", "png", "svg"]:
                path = os.path.join(output_dir, f"{base}.{ext}")
                fig.savefig(path, dpi=300 if ext != "png" else 200, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved Extended Layout {model}: {base}.svg / .png / .pdf")
        else:
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="CNN GAP vs SAE Paired Slope Chart for Cell Death Prediction"
    )
    parser.add_argument("--cnn_results_dir", type=str, required=True)
    parser.add_argument("--sae_results_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--cnn_config", type=str, default="MoCo_l2norm")
    parser.add_argument("--cnn_layer", type=str, default="stage5_out")
    parser.add_argument("--sae_l2norm", type=str, default="l2norm")
    parser.add_argument("--sae_filter", type=str, default="no_filter")
    parser.add_argument("--sae_config", type=str, default=None)
    args = parser.parse_args()

    cnn_csv = os.path.join(args.cnn_results_dir, "aggregated_r2_per_seed.csv")
    sae_csv = os.path.join(args.sae_results_dir, "sae_r2_per_seed.csv")

    if not os.path.exists(cnn_csv):
        print(f"ERROR: CNN CSV not found: {cnn_csv}")
        sys.exit(1)
    if not os.path.exists(sae_csv):
        print(f"ERROR: SAE CSV not found: {sae_csv}")
        sys.exit(1)

    cnn_data = read_cnn_seeds(cnn_csv)
    sae_data = read_sae_seeds(sae_csv)
    print(f"\n  CNN seed entries: {len(cnn_data)}")
    print(f"  SAE seed entries: {len(sae_data)}")

    output_dir = args.output_dir or args.sae_results_dir

    plot_cnn_vs_sae(
        cnn_data,
        sae_data,
        output_dir,
        args.cnn_config,
        args.cnn_layer,
        sae_l2norm=args.sae_l2norm,
        sae_filter=args.sae_filter,
        sae_config=args.sae_config,
    )

    print("\n  DONE")


if __name__ == "__main__":
    main()
