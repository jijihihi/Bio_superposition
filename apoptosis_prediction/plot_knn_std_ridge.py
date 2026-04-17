# ==============================================================================
# Ridgeline KDE Plot — Local Std Distribution: CNN vs SAE
#
# Per mutation subplot, K values as stacked rows.
# Each row: CNN (blue) and SAE (orange) KDE overlaid.
# X-axis : local_std of KNN apoptosis neighbors
# Y-rows  : k values (small k at bottom, large k at top)
# Annotation: MWU p-value (pooled per-sample across seeds) + median dashed line
#
# Data source: ratios_*.npz files (local_stds arrays)
#
# Usage (Colab):
# !python -m apoptosis_prediction.plot_knn_std_ridge \
#         --base_dir "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/local_linearity/raw" \
#         --output_dir "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/local_linearity/plots" \
#         --experiment raw
# ==============================================================================

import os
import re
import sys
import argparse
import glob
import numpy as np
from collections import defaultdict

import matplotlib
_IN_COLAB = "google.colab" in sys.modules
if not _IN_COLAB:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import gaussian_kde
import scipy.stats as stats_mod

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'
sns.set_style("white")

CNN_COLOR = "#4C72B0"   # blue
SAE_COLOR = "#DD8452"   # orange
MUTATIONS_ORDER = ["SNCA", "GBA", "LRRK2"]

# ─────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser(
        description="Ridgeline KDE: local std distribution per (mutation, k), CNN vs SAE"
    )
    p.add_argument("--base_dir", type=str, required=True,
                   help="Directory with sweep results (e.g. .../local_linearity/raw)")
    p.add_argument("--output_dir", type=str, default="")
    p.add_argument("--experiment", type=str, default="raw",
                   help="Label for output file names")
    p.add_argument("--fixed_log2fc", type=float, default=-1,
                   help="dpt_matched: filter by log2fc value (e.g. 1.0)")
    p.add_argument("--k_neighbors", type=int, nargs="*", default=[5, 15, 25],
                   help="Subset of k values to plot (default: 5 15 25)")
    p.add_argument("--bw_method", default=0.15,
                   help="KDE bandwidth: 'scott', 'silverman', or float scalar")
    p.add_argument("--x_max_pct", type=float, default=99.0,
                   help="X-axis upper percentile cutoff (default 99.0)")
    p.add_argument("--row_height", type=float, default=1.0,
                   help="Vertical spacing between rows (default 1.0)")
    p.add_argument("--kde_scale", type=float, default=0.75,
                   help="KDE peak height as fraction of row_height (default 0.75)")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────
def load_npz_pooled(base_dir, fixed_log2fc=-1):
    """
    ratios_*.npz를 재귀 탐색 → (source, mut, k)별 local_stds를
    모든 seed에서 concat한 pooled 배열 반환.
    Returns: {source: {mut: {k: np.ndarray(N_total,)}}}
    """
    pooled = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    pattern = os.path.join(base_dir, "**", "ratios_*.npz")
    n_loaded = 0

    for npz_path in glob.glob(pattern, recursive=True):
        # log2fc 필터
        if fixed_log2fc >= 0:
            parts = npz_path.replace("\\", "/").split("/")
            matched = False
            for part in parts:
                m = re.match(r"log2fc_([\d.]+)", part)
                if m:
                    if abs(float(m.group(1)) - fixed_log2fc) < 0.01:
                        matched = True
                    break
            if not matched:
                continue

        fname = os.path.basename(npz_path)
        m = re.match(r"ratios_(CNN|SAE)_(\w+)_k(\d+)\.npz", fname)
        if not m:
            continue
        source, mut, k = m.group(1), m.group(2), int(m.group(3))

        try:
            data = np.load(npz_path, allow_pickle=False)
            # local_stds 우선 (새 포맷); 구버전은 ratios fallback
            if "local_stds" in data:
                arr = data["local_stds"]
            elif "ratios" in data:
                arr = data["ratios"]
            else:
                continue
            valid = arr[np.isfinite(arr) & (arr >= 0)]
            if len(valid) > 0:
                pooled[source][mut][k].append(valid)
                n_loaded += 1
        except Exception as e:
            print(f"  [SKIP] {npz_path}: {e}")

    result = defaultdict(lambda: defaultdict(dict))
    for source in pooled:
        for mut in pooled[source]:
            for k, arrays in pooled[source][mut].items():
                result[source][mut][k] = np.concatenate(arrays)

    print(f"  Loaded {n_loaded} NPZ files across "
          f"{sum(len(v) for v in result.values())} (source×mut×k) combinations")
    return result


# ─────────────────────────────────────────────
# MWU (pooled per-sample)
# ─────────────────────────────────────────────
def mwu_pooled(cnn_arr, sae_arr):
    """
    One-sided MWU: H₁ SAE < CNN (lower local std = smoother).
    Returns (p, rank_biserial_r).
    """
    if len(cnn_arr) < 5 or len(sae_arr) < 5:
        return np.nan, np.nan
    stat, p = stats_mod.mannwhitneyu(sae_arr, cnn_arr, alternative="less")
    r_rb = 1 - (2 * stat) / (len(sae_arr) * len(cnn_arr))
    return p, r_rb


# ─────────────────────────────────────────────
# Core plot
# ─────────────────────────────────────────────
def plot_ridgeline(pooled_stds, mutations, k_list, output_dir, dpi,
                   experiment, bw_method="scott",
                   x_max_pct=99.0, row_height=1.0, kde_scale=0.75):
    """
    Ridgeline KDE per mutation.
    pooled_stds : {source: {mut: {k: np.ndarray}}}
    mutations   : list of mutation names
    k_list      : sorted list of k values (small → large, plotted bottom → top)
    """
    n_mut  = len(mutations)
    n_rows = len(k_list)
    RH     = row_height    # vertical spacing
    KS     = kde_scale     # KDE peak height fraction

    # ── Global x-axis range (log10(10x+1) Transform) ──
    all_vals = []
    for src_d in pooled_stds.values():
        for mut_d in src_d.values():
            for arr in mut_d.values():
                all_vals.extend(arr)
                
    def trans(x):
        return np.log10(100.0 * x + 1.0)
    
    if len(all_vals) > 0: 
        log_vals = trans(np.array(all_vals))
        log_min = 0.0  # 음수 꼬리(0.00의 뒷부분) 차단
        log_max = np.percentile(log_vals, x_max_pct)
    else:
        log_min, log_max = 0.0, 1.0

    # 최대값 방향으로만 패딩
    log_max += 0.1
    x_range = np.linspace(log_min, log_max, 500)
    x_pad = (log_max - log_min) * 0.20

    # ── Figure layout ────────────────────────
    fig_w = 2.5 * n_mut  # 원래 4.5 였던 가로 폭을 줄여서 그래프를 양옆으로 압축(더 뾰족해 보임)
    fig_h = max(3.5, n_rows * RH * 1.3 + 1.0)
    fig, axes = plt.subplots(1, n_mut, figsize=(fig_w, fig_h))
    if n_mut == 1:
        axes = [axes]

    bw = bw_method
    try:
        bw = float(bw)
    except (ValueError, TypeError):
        pass

    # ── Global peak for exact equal-area scaling ──
    # 면적을 1:1로 완전히 동일하게 맞추기 위해 전체 분포 중 가장 큰 peak를 구합니다.
    global_peak = 0.0
    for src_d in pooled_stds.values():
        for mut_d in src_d.values():
            for arr in mut_d.values():
                if len(arr) >= 5:
                    try:
                        k_temp = gaussian_kde(trans(arr), bw_method=bw)
                        global_peak = max(global_peak, k_temp(x_range).max())
                    except:
                        pass
    if global_peak <= 0:
        global_peak = 1.0

    for col_idx, (ax, mut) in enumerate(zip(axes, mutations)):
        for row_idx, k in enumerate(k_list):
            y_base = row_idx * RH

            cnn_arr = pooled_stds.get("CNN", {}).get(mut, {}).get(k, np.array([]))
            sae_arr = pooled_stds.get("SAE", {}).get(mut, {}).get(k, np.array([]))

            # ── MWU ──────────────────────────
            p_val, r_rb = mwu_pooled(cnn_arr, sae_arr)

            # ── Draw KDE for each source ──────
            for arr, color, label, zorder_base in [
                (cnn_arr, CNN_COLOR, "CNN", 3),
                (sae_arr, SAE_COLOR, "SAE", 4),
            ]:
                if len(arr) < 5:
                    continue
                try:
                    log_arr = trans(arr)
                    kde = gaussian_kde(log_arr, bw_method=bw)
                    y_kde = kde(x_range)
                    if y_kde.max() <= 0:
                        continue
                    
                    # normalize: 면적 동일 유지를 위해 개별 peak 대신 모든 그래프에 동일한 global_peak 상수 사용
                    y_kde = y_kde / global_peak * KS * RH

                    # Fill under curve
                    ax.fill_between(x_range, y_base, y_base + y_kde,
                                    color=color, alpha=0.50,
                                    zorder=zorder_base, linewidth=0)
                    # Outline
                    ax.plot(x_range, y_base + y_kde,
                            color=color, linewidth=1.2, alpha=0.95,
                            zorder=zorder_base + 0.5)

                    # Median dashed vertical line
                    median_val = np.median(arr)
                    log_median = trans(median_val)
                    if log_min <= log_median <= log_max:
                        med_idx = np.argmin(np.abs(x_range - log_median))
                        kde_at_med = y_kde[med_idx]
                        ax.vlines(log_median, y_base, y_base + kde_at_med,
                                  color=color, linewidth=1.5,
                                  linestyle="--", alpha=0.90,
                                  zorder=zorder_base + 1)

                except Exception as e:
                    print(f"  [KDE warn] {mut} k={k} {label}: {e}")

            # ── Baseline ─────────────────────
            ax.axhline(y_base, color="gray", linewidth=0.5, alpha=0.35, zorder=2)

            # ── P-value annotation ────────────
            if not np.isnan(p_val):
                star = ("***" if p_val < 0.001 else "**" if p_val < 0.01
                        else "*"   if p_val < 0.05 else "n.s.")
                color_star = "black" if star != "n.s." else "#888888"
                ax.text(log_max + x_pad * 0.05,
                        y_base + KS * RH * 0.40,
                        f"{star}\np={p_val:.2e}",
                        ha="left", va="center",
                        fontsize=7, color=color_star,
                        fontweight="bold" if star != "n.s." else "normal")

        # ── Y-axis: k labels ─────────────────
        yticks     = [i * RH + KS * RH * 0.20 for i in range(n_rows)]
        yticklabels = [f"k = {k}" for k in k_list]
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels, fontsize=10)
        ax.tick_params(axis="y", length=0, pad=4)

        ax.set_xlim(log_min, log_max + x_pad)
        
        # User-friendly real-value ticks mapped to log10(10x+1)
        target_ticks = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        tick_pos = [trans(t) for t in target_ticks if trans(t) <= log_max]
        tick_labels = [f"{t:.2f}" for t in target_ticks if trans(t) <= log_max]
        
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels)
        
        ax.set_ylim(-0.08 * RH, n_rows * RH)
        ax.set_xlabel("Local Apoptosis Std (Transformed: log10(10x + 1))", fontsize=11)
        ax.set_title(mut, fontsize=14, fontweight="bold", pad=8)

        # ── Legend (first subplot only) ───────
        if col_idx == 0:
            patches = [
                mpatches.Patch(color=CNN_COLOR, alpha=0.6, label="CNN"),
                mpatches.Patch(color=SAE_COLOR, alpha=0.6, label="SAE"),
            ]
            ax.legend(handles=patches, fontsize=10,
                      loc="upper right", framealpha=0.85,
                      edgecolor="white")

        sns.despine(ax=ax, left=True)

    fig.suptitle(
        f"Local Apoptosis Std Distribution — CNN vs SAE  [{experiment}]\n"
        f"(lower std → tighter neighbourhood → better feature organisation)",
        fontsize=12, fontweight="bold", y=1.03,
    )
    fig.tight_layout()

    fname = f"knn_std_ridge_{experiment}"
    for ext in [".png", ".svg"]:
        fpath = os.path.join(output_dir, fname + ext)
        fig.savefig(fpath, dpi=dpi, bbox_inches="tight")
    print(f"  Saved: {fname}.png / .svg")

    if _IN_COLAB:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    args = get_args()
    out_dir = args.output_dir or os.path.join(args.base_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading NPZ pooled local_stds from: {args.base_dir}")
    pooled_stds = load_npz_pooled(args.base_dir,
                                  fixed_log2fc=args.fixed_log2fc)

    # Collect mutations and k values
    all_muts = set()
    all_ks   = set()
    for src_d in pooled_stds.values():
        for mut, k_dict in src_d.items():
            all_muts.add(mut)
            all_ks.update(k_dict.keys())

    mutations = [m for m in MUTATIONS_ORDER if m in all_muts]
    mutations += sorted(all_muts - set(MUTATIONS_ORDER))

    # 기본값으로 지정된 [5, 15, 25] 중 실제로 로드된 k들만 남깁니다.
    if args.k_neighbors:
        k_list = sorted([k for k in args.k_neighbors if k in all_ks])
        if not k_list:
            print("Warning: None of the requested k_neighbors were found in NPZ files. Falling back to all found k values.")
            k_list = sorted(all_ks)
    else:
        k_list = sorted(all_ks)

    print(f"  Mutations : {mutations}")
    print(f"  K values  : {k_list}")

    # Summary table
    print(f"\n{'Src':4s} {'Mut':8s} {'k':>4s} {'N':>8s} "
          f"{'Median':>9s} {'Q25':>9s} {'Q75':>9s} {'MWU p':>12s} {'r_rb':>7s}")
    print("-" * 75)
    for mut in mutations:
        for k in k_list:
            cnn_arr = pooled_stds.get("CNN", {}).get(mut, {}).get(k, np.array([]))
            sae_arr = pooled_stds.get("SAE", {}).get(mut, {}).get(k, np.array([]))
            p, r_rb = mwu_pooled(cnn_arr, sae_arr)
            for src, arr in [("CNN", cnn_arr), ("SAE", sae_arr)]:
                if len(arr) > 0:
                    print(f"{src:4s} {mut:8s} {k:4d} {len(arr):8d} "
                          f"{np.median(arr):9.4f} "
                          f"{np.percentile(arr, 25):9.4f} "
                          f"{np.percentile(arr, 75):9.4f} "
                          f"{'':12s} {'':7s}")
            # MWU per (mut, k)
            p_str  = f"{p:.2e}" if not np.isnan(p) else "N/A"
            rb_str = f"{r_rb:.4f}" if not np.isnan(r_rb) else "N/A"
            star   = ("***" if (not np.isnan(p) and p < 0.001) else
                      "**"  if (not np.isnan(p) and p < 0.01)  else
                      "*"   if (not np.isnan(p) and p < 0.05)  else "n.s.")
            print(f"{'MWU':4s} {mut:8s} {k:4d} {'':8s} "
                  f"{'':9s} {'':9s} {'':9s} "
                  f"{p_str + ' ' + star:>12s} {rb_str:>7s}")
            print()

    # ── Ridgeline plot ──
    exp_label = args.experiment
    if args.fixed_log2fc >= 0:
        exp_label = f"{args.experiment}_log2fc{args.fixed_log2fc}"

    plot_ridgeline(
        pooled_stds, mutations, k_list,
        output_dir  = out_dir,
        dpi         = args.dpi,
        experiment  = exp_label,
        bw_method   = args.bw_method,
        x_max_pct   = args.x_max_pct,
        row_height  = args.row_height,
        kde_scale   = args.kde_scale,
    )

    print(f"\n  Output: {out_dir}")


if __name__ == "__main__":
    main()
