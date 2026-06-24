# ==============================================================================
# Plot KNN Std Ratio Trend: K vs mean/median ratio for CNN & SAE
#
# Reads local_linearity_results.json from sweep output directories.
# Aggregates across seeds → plots mean ± SEM per (source, mutation, k).
#
# Modes:
#   raw / dpt_matched: x-axis = K, y-axis = ratio (CNN vs SAE)
#   de_sweep:          x-axis = DE log2FC threshold, y-axis = ratio at fixed K
#
# Usage (Colab):
#   python -m apoptosis_prediction.plot_knn_std_trend \
#       --base_dir "/content/drive/MyDrive/Final_paper/.../local_linearity/raw" \
#       --experiment raw \
#       --output_dir "/content/drive/MyDrive/Final_paper/.../local_linearity/plots"
#
#   # DE sweep mode:
#   python -m apoptosis_prediction.plot_knn_std_trend \
#       --base_dir "/content/drive/.../local_linearity/de_sweep" \
#       --experiment de_sweep \
#       --fixed_k 15 \
#       --output_dir "/content/drive/.../local_linearity/plots"
# ==============================================================================

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

import matplotlib
import numpy as np

_IN_COLAB = "google.colab" in sys.modules
if not _IN_COLAB:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.family"] = "sans-serif"
sns.set_style("ticks")

import scipy.stats as stats

COLORS = {"CNN": "#4C72B0", "SAE": "#DD8452"}
MARKERS = {"CNN": "s", "SAE": "o"}


def get_args():
    p = argparse.ArgumentParser(
        description="Plot KNN std ratio trend from local_linearity sweep results"
    )
    p.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Base directory containing sweep results (e.g., .../local_linearity/raw)",
    )
    p.add_argument(
        "--experiment",
        type=str,
        default="raw",
        choices=["raw", "dpt_matched", "de_sweep"],
        help="Experiment type: 'raw', 'dpt_matched', or 'de_sweep'",
    )
    p.add_argument(
        "--fixed_k",
        type=int,
        default=0,
        help="For de_sweep: which K to plot (required for de_sweep)",
    )
    p.add_argument(
        "--fixed_log2fc",
        type=float,
        default=-1,
        help="For dpt_matched: filter by log2fc dir (e.g., 1.0)",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory for plots (default: base_dir/plots)",
    )
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args()


def find_json_files(base_dir, experiment, fixed_log2fc=-1):
    """
    Recursively find all local_linearity_results.json files.
    If fixed_log2fc >= 0, filter to only paths containing log2fc_X.X.
    Returns list of (source_label, json_path) tuples.
    """
    results = []
    pattern = os.path.join(base_dir, "**", "local_linearity_results.json")
    for jpath in glob.glob(pattern, recursive=True):
        # Filter by log2fc if specified
        if fixed_log2fc >= 0:
            parts = jpath.replace("\\", "/").split("/")
            log2fc_val = None
            for part in parts:
                m = re.match(r"log2fc_([\d.]+)", part)
                if m:
                    log2fc_val = float(m.group(1))
                    break
            if log2fc_val is None or abs(log2fc_val - fixed_log2fc) > 0.01:
                continue

        # Determine source from directory name
        parent = os.path.basename(os.path.dirname(jpath))
        if "knn_std_cnn" in parent or "knn_log_std_cnn" in parent:
            source = "CNN"
        elif "knn_std_sae" in parent or "knn_log_std_sae" in parent:
            source = "SAE"
        else:
            source = None
        results.append((source, jpath))
    return results


def load_results(json_files):
    """
    Load all JSON files and aggregate results.
    Returns: {source: {mutation: {k: [ratio_values_across_seeds]}}}
    for both mean_ratio and median_ratio.
    """
    mean_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    median_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for source_hint, jpath in json_files:
        try:
            with open(jpath, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"  [SKIP] {jpath}: {e}")
            continue

        for entry in data.get("results", []):
            source = entry.get("source", source_hint)
            if source is None:
                continue
            mut = entry["mutation"]
            k = entry["k"]
            mean_data[source][mut][k].append(entry["mean_ratio"])
            median_data[source][mut][k].append(entry["median_ratio"])

    return mean_data, median_data


def load_npz_pooled_stds(base_dir, fixed_log2fc=-1):
    """
    ratios_*.npz 파일을 재귀 탐색하여 (source, mut, k)별로 per-sample
    local_stds 배열을 모든 seed에서 concat한 pooled 배열을 반환.

    Filename: ratios_{source}_{mut}_k{k}.npz
    Keys: local_stds (N,)  ← 핵심 / ratios (N,) / global_std (1,)

    Returns: {source: {mut: {k: np.ndarray(N_total,)}}}
    """
    pooled = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    pattern = os.path.join(base_dir, "**", "ratios_*.npz")
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
            # local_stds 우선; 구버전 NPZ는 ratios fallback
            if "local_stds" in data:
                arr = data["local_stds"]
            elif "ratios" in data:
                arr = data["ratios"]
            else:
                continue
            valid = arr[np.isfinite(arr)]
            if len(valid) > 0:
                pooled[source][mut][k].append(valid)
        except Exception as e:
            print(f"  [SKIP NPZ] {npz_path}: {e}")

    # seed별 배열을 concat → 하나의 큰 배열
    result = defaultdict(lambda: defaultdict(dict))
    for source in pooled:
        for mut in pooled[source]:
            for k, arrays in pooled[source][mut].items():
                result[source][mut][k] = np.concatenate(arrays)
                print(
                    f"  NPZ pool  {source:3s} / {mut:6s} / k={k:3d}: "
                    f"{len(arrays)} seeds, N={result[source][mut][k].shape[0]}"
                )
    return result


def compute_mwu_from_pooled(pooled_stds, mutations):
    """
    각 (mutation, k)에 대해 one-sided MWU: H₁: SAE < CNN (local std 작을수록 좋음).
    모든 seed를 pooling한 per-sample 배열을 사용.

    Returns: {mut: {k: {"p", "stat", "r_rb", "n_cnn", "n_sae"}}}
    rank-biserial r_rb > 0 이면 SAE가 CNN보다 작은 방향.
    """
    mwu_results = defaultdict(dict)
    for mut in mutations:
        cnn_dict = pooled_stds.get("CNN", {}).get(mut, {})
        sae_dict = pooled_stds.get("SAE", {}).get(mut, {})
        all_ks = sorted(set(cnn_dict.keys()) | set(sae_dict.keys()))
        for k in all_ks:
            cnn_arr = cnn_dict.get(k, np.array([]))
            sae_arr = sae_dict.get(k, np.array([]))
            if len(cnn_arr) < 5 or len(sae_arr) < 5:
                continue
            stat, p = stats.mannwhitneyu(sae_arr, cnn_arr, alternative="less")
            n1, n2 = len(sae_arr), len(cnn_arr)
            r_rb = 1 - (2 * stat) / (n1 * n2)  # > 0 이면 SAE < CNN
            mwu_results[mut][k] = {
                "stat": float(stat),
                "p": float(p),
                "r_rb": float(r_rb),
                "n_cnn": n2,
                "n_sae": n1,
            }
    return mwu_results


def load_morans_i_from_jsons(json_files):
    """
    JSON 파일에서 morans_I 스칼라를 (source, mut, k)별로 수집.
    각 JSON은 하나의 seed 실행 결과 → per (mut, k)당 Moran's I 스칼라 1개.

    Returns: {source: {mut: {k: [I_seed1, I_seed2, ...]}}}
    """
    morans_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for source_hint, jpath in json_files:
        try:
            with open(jpath, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            continue
        for entry in data.get("results", []):
            source = entry.get("source", source_hint)
            if source is None or entry.get("morans_I") is None:
                continue
            mut = entry["mutation"]
            k = entry["k"]
            morans_data[source][mut][k].append(entry["morans_I"])
    return morans_data


def permutation_test_morans_i(cnn_vals, sae_vals, n_perm=0, seed=42):
    """
    Moran's I 비교: exact (or random) permutation test.
    H₁: I(SAE) > I(CNN) 단측 — SAE가 feature space에서 spatial autocorrelation이 더 강함.

    귀무가설: CNN/SAE 라벨이 Moran's I 분포에 영향 없다.
    n1+n2개 I값을 도리 여서 n2개를 SAE로 지정하는 모든 조합 열거.

    Parameters
    ----------
    cnn_vals, sae_vals : list/array of Moran's I scalars (one per seed)
    n_perm : 0 = exhaustive C(n1+n2, n2), >0 = random permutation
    seed   : RNG seed

    Returns
    -------
    observed_delta : float          mean(SAE_I) − mean(CNN_I)
    p_value        : float          P(null_delta ≥ observed_delta)
    n_perm_actual  : int            실제 사용된 permutation 수
    null_deltas    : np.ndarray     null 분포
    """
    from itertools import combinations

    cnn = np.array(cnn_vals, dtype=float)
    sae = np.array(sae_vals, dtype=float)
    n1, n2 = len(cnn), len(sae)
    n_total = n1 + n2
    pooled = np.concatenate([cnn, sae])
    observed_delta = np.mean(sae) - np.mean(cnn)

    if n_perm == 0:
        # Exhaustive: C(n_total, n2) 조합 전수
        deltas = []
        for sae_idx in combinations(range(n_total), n2):
            sae_set = set(sae_idx)
            cnn_idx = [i for i in range(n_total) if i not in sae_set]
            d = np.mean(pooled[list(sae_idx)]) - np.mean(pooled[cnn_idx])
            deltas.append(d)
        deltas = np.array(deltas)
    else:
        rng = np.random.RandomState(seed)
        deltas = np.zeros(n_perm)
        for i in range(n_perm):
            perm = rng.permutation(n_total)
            d = np.mean(pooled[perm[:n2]]) - np.mean(pooled[perm[n2:]])
            deltas[i] = d

    # 단측 p: null 중 observed_delta 이상인 비율
    p_value = np.mean(deltas >= observed_delta)
    return observed_delta, p_value, len(deltas), deltas


def plot_trend(
    data_dict, metric_name, mutations, output_dir, dpi, experiment, mwu_pvals=None
):
    """
    Plot K vs ratio trend for CNN & SAE, one subplot per mutation.
    data_dict: {source: {mutation: {k: [values]}}}
    mwu_pvals: {mutation: {k: p_value}}
    """
    n_mut = len(mutations)
    fig, axes = plt.subplots(1, n_mut, figsize=(5 * n_mut, 4.5), sharey=True)
    if n_mut == 1:
        axes = [axes]

    for ax, mut in zip(axes, mutations):
        for source in ["CNN", "SAE"]:
            if source not in data_dict or mut not in data_dict[source]:
                continue

            k_vals = sorted(data_dict[source][mut].keys())
            means = []
            sems = []
            all_k_scatter = []
            all_v_scatter = []
            for k in k_vals:
                vals = np.array(data_dict[source][mut][k])
                means.append(np.mean(vals))
                sems.append(np.std(vals) / max(np.sqrt(len(vals)), 1))
                # Collect raw points for scatter
                for v in vals:
                    all_k_scatter.append(k)
                    all_v_scatter.append(v)

            means = np.array(means)
            sems = np.array(sems)

            # Individual seed points (jittered x + random y-nudge)
            rng = np.random.RandomState(42)
            k_span = (max(k_vals) - min(k_vals)) if len(k_vals) > 1 else 1
            jitter_x = k_span * 0.025
            jitter_sign = -1 if source == "CNN" else 1
            k_jittered = (
                np.array(all_k_scatter, dtype=float)
                + jitter_sign * jitter_x
                + rng.uniform(-jitter_x * 0.4, jitter_x * 0.4, len(all_k_scatter))
            )
            ax.scatter(
                k_jittered,
                all_v_scatter,
                color=COLORS[source],
                alpha=0.45,
                s=30,
                marker=MARKERS[source],
                edgecolors=COLORS[source],
                linewidths=0.5,
                zorder=2,
            )

            # Mean trend line on top
            ax.plot(
                k_vals,
                means,
                "-",
                color=COLORS[source],
                marker=MARKERS[source],
                markersize=7,
                linewidth=2.5,
                label=source,
                markeredgecolor="white",
                markeredgewidth=0.8,
                zorder=3,
            )
            ax.fill_between(
                k_vals,
                means - sems,
                means + sems,
                color=COLORS[source],
                alpha=0.15,
                zorder=1,
            )

        # Annotate statistical significance if available
        if mwu_pvals and mut in mwu_pvals:
            for k_val, p_val in mwu_pvals[mut].items():
                if p_val < 0.05:
                    if "SAE" in data_dict and "CNN" in data_dict:
                        try:
                            # place star above the higher mean
                            h_sae = np.mean(data_dict["SAE"][mut][k_val])
                            h_cnn = np.mean(data_dict["CNN"][mut][k_val])
                            h_max = max(h_sae, h_cnn)
                            star = (
                                "***"
                                if p_val < 0.001
                                else "**" if p_val < 0.01 else "*"
                            )
                            ax.annotate(
                                star,
                                (k_val, h_max),
                                textcoords="offset points",
                                xytext=(0, 5),
                                ha="center",
                                fontsize=12,
                                fontweight="bold",
                                color="black",
                            )
                        except Exception:
                            pass

        # Reference line at ratio = 1 (global std = local std)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_xlabel("K (neighbors)", fontsize=12)
        if ax == axes[0]:
            ax.set_ylabel(f"{metric_name} Ratio (local / global)", fontsize=12)
        ax.set_title(mut, fontsize=14, fontweight="bold")
        ax.legend(fontsize=10, framealpha=0.8)
        ax.grid(True, alpha=0.15)
        ax.set_ylim(bottom=0)

    sns.despine()
    fig.suptitle(
        f"KNN Apoptosis Std Ratio — {metric_name} ({experiment})",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()

    fname = f"knn_std_{metric_name.lower()}_{experiment}"
    for ext in [".png", ".svg"]:
        fig.savefig(os.path.join(output_dir, fname + ext), dpi=dpi, bbox_inches="tight")
    print(f"  Saved: {fname}.png/.svg")

    if _IN_COLAB:
        plt.show()
    plt.close(fig)


def plot_morans_i_trend(
    morans_data, mutations, perm_pvals, output_dir, dpi, experiment
):
    """
    K vs Global Moran's I trend plot.
    morans_data : {source: {mut: {k: [I_seed1, I_seed2, ...]}}}
    perm_pvals  : {mut: {k: p}}  (exact permutation, None = skip annotation)

    트리줄 요약:
      - CNN / SAE 라인 각각 평균 ± SEM
      - 개별 seed 도트 (jitter)
      - 하단 y=0 기준선 (무상관 기대값)—클수록 좋으므로 SAE가 높아야 함
      - 통계적 유의성: permutation p-value 각 k점에 star 표시
    """
    n_mut = len(mutations)
    fig, axes = plt.subplots(1, n_mut, figsize=(5 * n_mut, 4.5), sharey=True)
    if n_mut == 1:
        axes = [axes]

    for ax, mut in zip(axes, mutations):
        y_all = []  # y range 추적
        for source in ["CNN", "SAE"]:
            if source not in morans_data or mut not in morans_data[source]:
                continue

            k_vals = sorted(morans_data[source][mut].keys())
            means, sems = [], []
            all_k_sc, all_v_sc = [], []

            for k in k_vals:
                vals = np.array(morans_data[source][mut][k], dtype=float)
                means.append(np.mean(vals))
                sems.append(np.std(vals) / max(np.sqrt(len(vals)), 1))
                for v in vals:
                    all_k_sc.append(k)
                    all_v_sc.append(v)
                    y_all.append(v)

            means = np.array(means)
            sems = np.array(sems)

            # Jittered seed dots
            rng = np.random.RandomState(42)
            k_span = (max(k_vals) - min(k_vals)) if len(k_vals) > 1 else 1
            jitter_x = k_span * 0.025
            jitter_sign = -1 if source == "CNN" else 1
            k_jit = (
                np.array(all_k_sc, dtype=float)
                + jitter_sign * jitter_x
                + rng.uniform(-jitter_x * 0.4, jitter_x * 0.4, len(all_k_sc))
            )
            ax.scatter(
                k_jit,
                all_v_sc,
                color=COLORS[source],
                alpha=0.45,
                s=30,
                marker=MARKERS[source],
                edgecolors=COLORS[source],
                linewidths=0.5,
                zorder=2,
            )

            # Mean trend
            ax.plot(
                k_vals,
                means,
                "-",
                color=COLORS[source],
                marker=MARKERS[source],
                markersize=7,
                linewidth=2.5,
                label=source,
                markeredgecolor="white",
                markeredgewidth=0.8,
                zorder=3,
            )
            ax.fill_between(
                k_vals,
                means - sems,
                means + sems,
                color=COLORS[source],
                alpha=0.15,
                zorder=1,
            )

        # Star annotations (exact permutation p-values)
        if perm_pvals and mut in perm_pvals:
            y_top = max(y_all) if y_all else 0.5
            ann_y = y_top * 1.04
            for k_val, p_val in sorted(perm_pvals[mut].items()):
                star = (
                    "***"
                    if p_val < 0.001
                    else "**" if p_val < 0.01 else "*" if p_val < 0.05 else None
                )
                if star:
                    ax.annotate(
                        star,
                        (k_val, ann_y),
                        textcoords="offset points",
                        xytext=(0, 2),
                        ha="center",
                        fontsize=13,
                        fontweight="bold",
                        color="black",
                    )

        # y=0 reference: Moran's I ~ 0 는 무작위 수체와 동일
        ax.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.5)

        ax.set_xlabel("K (neighbors)", fontsize=12)
        if ax == axes[0]:
            ax.set_ylabel("Global Moran's I", fontsize=12)
        ax.set_title(mut, fontsize=14, fontweight="bold")
        ax.legend(fontsize=10, framealpha=0.8)
        ax.grid(True, alpha=0.15)

    sns.despine()
    fig.suptitle(
        f"Moran's I vs K — Spatial Autocorrelation ({experiment})",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()

    fname = f"morans_I_trend_{experiment}"
    for ext in [".png", ".svg"]:
        fig.savefig(os.path.join(output_dir, fname + ext), dpi=dpi, bbox_inches="tight")
    print(f"  Saved: {fname}.png/.svg")

    if _IN_COLAB:
        plt.show()
    plt.close(fig)


# ==============================================================================
# DE Sweep: x-axis = log2FC threshold, y-axis = ratio at fixed K
# ==============================================================================
def find_de_sweep_jsons(base_dir):
    """
    Scan de_sweep directories.
    Structure: base_dir/cnn_seed_*/log2fc_*/knn_std_cnn/local_linearity_results.json
    Returns list of (log2fc_val, json_path).
    """
    results = []
    pattern = os.path.join(base_dir, "**", "local_linearity_results.json")
    for jpath in glob.glob(pattern, recursive=True):
        # Extract log2fc from path: look for "log2fc_X.X" in any parent
        parts = jpath.replace("\\", "/").split("/")
        log2fc_val = None
        for part in parts:
            m = re.match(r"log2fc_([\d.]+)", part)
            if m:
                log2fc_val = float(m.group(1))
                break
        if log2fc_val is not None:
            results.append((log2fc_val, jpath))
    return sorted(results, key=lambda x: x[0])


def load_de_sweep_results(de_json_files, fixed_k):
    """
    Load DE sweep JSONs and extract ratio at fixed_k.
    Returns: {mutation: {log2fc: [mean_ratio_values], ...}}
    for both mean and median.
    """
    mean_data = defaultdict(lambda: defaultdict(list))
    median_data = defaultdict(lambda: defaultdict(list))

    for log2fc_val, jpath in de_json_files:
        try:
            with open(jpath, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"  [SKIP] {jpath}: {e}")
            continue

        for entry in data.get("results", []):
            if entry["k"] != fixed_k:
                continue
            mut = entry["mutation"]
            mean_data[mut][log2fc_val].append(entry["mean_ratio"])
            median_data[mut][log2fc_val].append(entry["median_ratio"])

    return mean_data, median_data


def plot_de_sweep_trend(data_dict, metric_name, mutations, fixed_k, output_dir, dpi):
    """
    Plot x=log2FC threshold, y=ratio for each mutation.
    """
    n_mut = len(mutations)
    fig, axes = plt.subplots(1, n_mut, figsize=(5 * n_mut, 4.5), sharey=True)
    if n_mut == 1:
        axes = [axes]

    color = "#4C72B0"  # CNN only for DE sweep

    for ax, mut in zip(axes, mutations):
        if mut not in data_dict:
            continue

        log2fc_vals = sorted(data_dict[mut].keys())
        means = []
        sems = []
        all_x_scatter = []
        all_y_scatter = []
        for lfc in log2fc_vals:
            vals = np.array(data_dict[mut][lfc])
            means.append(np.mean(vals))
            sems.append(np.std(vals) / max(np.sqrt(len(vals)), 1))
            for v in vals:
                all_x_scatter.append(lfc)
                all_y_scatter.append(v)

        means = np.array(means)
        sems = np.array(sems)

        # Individual seed points (jittered)
        rng = np.random.RandomState(42)
        x_span = (max(log2fc_vals) - min(log2fc_vals)) if len(log2fc_vals) > 1 else 0.1
        jitter_x = x_span * 0.02
        x_jittered = np.array(all_x_scatter, dtype=float) + rng.uniform(
            -jitter_x, jitter_x, len(all_x_scatter)
        )
        ax.scatter(
            x_jittered,
            all_y_scatter,
            color=color,
            alpha=0.4,
            s=30,
            marker="s",
            edgecolors=color,
            linewidths=0.5,
            zorder=2,
        )

        # Mean trend
        ax.plot(
            log2fc_vals,
            means,
            "-",
            color=color,
            marker="s",
            markersize=7,
            linewidth=2.5,
            markeredgecolor="white",
            markeredgewidth=0.8,
            zorder=3,
            label="CNN",
        )
        ax.fill_between(
            log2fc_vals, means - sems, means + sems, color=color, alpha=0.15, zorder=1
        )

        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_xlabel("DE log₂FC threshold", fontsize=12)
        if ax == axes[0]:
            ax.set_ylabel(f"{metric_name} Ratio (local / global)", fontsize=12)
        ax.set_title(mut, fontsize=14, fontweight="bold")
        ax.legend(fontsize=10, framealpha=0.8)
        ax.grid(True, alpha=0.15)
        ax.set_ylim(bottom=0)

    sns.despine()
    fig.suptitle(
        f"CNN DE Sweep — {metric_name} Ratio at K={fixed_k}",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()

    fname = f"de_sweep_{metric_name.lower()}_k{fixed_k}"
    for ext in [".png", ".svg"]:
        fig.savefig(os.path.join(output_dir, fname + ext), dpi=dpi, bbox_inches="tight")
    print(f"  Saved: {fname}.png/.svg")

    if _IN_COLAB:
        plt.show()
    plt.close(fig)


# ==============================================================================
# Main
# ==============================================================================
def main():
    args = get_args()
    out_dir = args.output_dir or os.path.join(args.base_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    if args.experiment == "de_sweep":
        # ── DE Sweep mode ──
        if args.fixed_k <= 0:
            print("ERROR: --fixed_k is required for de_sweep mode.")
            return

        print(f"Scanning DE sweep: {args.base_dir}")
        de_jsons = find_de_sweep_jsons(args.base_dir)
        print(f"  Found {len(de_jsons)} JSON files")

        if not de_jsons:
            print("  No results found. Check --base_dir path.")
            return

        log2fc_set = sorted(set(lfc for lfc, _ in de_jsons))
        print(f"  log2FC values: {log2fc_set}")

        mean_data, median_data = load_de_sweep_results(de_jsons, args.fixed_k)
        mutations = sorted(
            mean_data.keys(),
            key=lambda x: (
                ["SNCA", "GBA", "LRRK2"].index(x)
                if x in ["SNCA", "GBA", "LRRK2"]
                else 99
            ),
        )

        print(f"  Mutations: {mutations}")

        # Summary table
        print(
            f"\n{'Mutation':8s}  {'log2FC':>7s}  {'Mean±SEM':>14s}  "
            f"{'Median±SEM':>14s}  {'N seeds':>7s}"
        )
        print("-" * 60)
        for mut in mutations:
            for lfc in sorted(mean_data[mut].keys()):
                mv = np.array(mean_data[mut][lfc])
                md = np.array(median_data[mut][lfc])
                print(
                    f"{mut:8s}  {lfc:7.2f}  "
                    f"{mv.mean():6.4f}±{mv.std()/max(np.sqrt(len(mv)),1):.4f}  "
                    f"{md.mean():6.4f}±{md.std()/max(np.sqrt(len(md)),1):.4f}  "
                    f"{len(mv):7d}"
                )

        plot_de_sweep_trend(
            mean_data, "Mean", mutations, args.fixed_k, out_dir, args.dpi
        )
        plot_de_sweep_trend(
            median_data, "Median", mutations, args.fixed_k, out_dir, args.dpi
        )

    else:
        # ── K-sweep mode (raw / dpt_matched) ──
        log2fc_filter = args.fixed_log2fc if args.experiment == "dpt_matched" else -1
        print(f"Scanning: {args.base_dir}")
        if log2fc_filter >= 0:
            print(f"  Filtering by log2fc = {log2fc_filter}")
        json_files = find_json_files(
            args.base_dir, args.experiment, fixed_log2fc=log2fc_filter
        )
        print(f"  Found {len(json_files)} JSON files")

        if not json_files:
            print("  No results found. Check --base_dir path.")
            return

        mean_data, median_data = load_results(json_files)

        all_muts = set()
        for source in mean_data:
            all_muts.update(mean_data[source].keys())
        mutations = sorted(
            all_muts,
            key=lambda x: (
                ["SNCA", "GBA", "LRRK2"].index(x)
                if x in ["SNCA", "GBA", "LRRK2"]
                else 99
            ),
        )

        print(f"  Sources: {list(mean_data.keys())}")
        print(f"  Mutations: {mutations}")

        # ── Per-sample pooled MWU (각 seed 의 local_stds 배열을 concat하여 MWU) ──
        print(f"\nLoading per-sample NPZ arrays for pooled MWU...")
        pooled_stds = load_npz_pooled_stds(args.base_dir, fixed_log2fc=log2fc_filter)
        mwu_pooled = compute_mwu_from_pooled(pooled_stds, mutations)

        # Summary table
        hdr = (
            f"{'Mutation':8s}  {'K':>4s} | {'CNN seeds Mean±SEM':>22s} | "
            f"{'SAE seeds Mean±SEM':>22s} | {'MWU p (pooled)':>16s} | "
            f"{'r_rb':>6s} | {'n_CNN':>8s} | {'n_SAE':>8s}"
        )
        print(f"\n{hdr}")
        print("-" * len(hdr))

        mwu_pvals_for_plot = defaultdict(dict)  # star 어노테이션용

        for mut in mutations:
            ks = set()
            if "CNN" in mean_data and mut in mean_data["CNN"]:
                ks.update(mean_data["CNN"][mut].keys())
            if "SAE" in mean_data and mut in mean_data["SAE"]:
                ks.update(mean_data["SAE"][mut].keys())

            for k in sorted(ks):
                cnn_m = np.array(mean_data.get("CNN", {}).get(mut, {}).get(k, []))
                sae_m = np.array(mean_data.get("SAE", {}).get(mut, {}).get(k, []))

                cnn_str = (
                    f"{cnn_m.mean():.4f}±{cnn_m.std()/max(np.sqrt(len(cnn_m)),1):.4f}"
                    if len(cnn_m) > 0
                    else "N/A"
                )
                sae_str = (
                    f"{sae_m.mean():.4f}±{sae_m.std()/max(np.sqrt(len(sae_m)),1):.4f}"
                    if len(sae_m) > 0
                    else "N/A"
                )

                mwu_entry = mwu_pooled.get(mut, {}).get(k)
                if mwu_entry:
                    p = mwu_entry["p"]
                    r_rb = mwu_entry["r_rb"]
                    n_cnn = mwu_entry["n_cnn"]
                    n_sae = mwu_entry["n_sae"]
                    star = (
                        "***"
                        if p < 0.001
                        else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
                    )
                    p_str = f"{p:.2e} {star}"
                    mwu_pvals_for_plot[mut][k] = p
                else:
                    p_str = "N/A (no NPZ)"
                    r_rb = float("nan")
                    n_cnn = n_sae = 0

                print(
                    f"{mut:8s}  {k:4d} | {cnn_str:>22s} | {sae_str:>22s} | "
                    f"{p_str:>16s} | {r_rb:>6.3f} | {n_cnn:>8d} | {n_sae:>8d}"
                )

        # ── Moran's I: Exact permutation test (seed-level scalar) ──
        # 각 seed마다 Moran's I scalar 1개 → CNN 4값 vs SAE 4값
        # Seed가 적어서 MWU 대신 라벨 permutation (C(8,4)=70 exhaustive)
        morans_data = load_morans_i_from_jsons(json_files)

        # n_total 확인: 반복 가능 요소 수 여부 판단
        sample_n_cnn = max(
            (
                len(v)
                for mut in morans_data.get("CNN", {}).values()
                for v in mut.values()
            ),
            default=0,
        )
        sample_n_sae = max(
            (
                len(v)
                for mut in morans_data.get("SAE", {}).values()
                for v in mut.values()
            ),
            default=0,
        )
        n_total_perm = sample_n_cnn + sample_n_sae

        # C(12,6)=924 이하면 exhaustive, 초과시 9999 random
        use_exhaustive = n_total_perm <= 12
        n_perm_moran = 0 if use_exhaustive else 9999
        perm_label = "exhaustive" if use_exhaustive else f"n_perm={n_perm_moran}"

        moran_hdr = (
            f"\n\u2500\u2500 Moran's I Permutation Test ({perm_label}: "
            f"min p={'1/70≈0.014' if use_exhaustive else '~0.000'}) \u2500\u2500\n"
            f"{'Mutation':8s}  {'K':>4s} | {'mean(SAE I)':>12s} | "
            f"{'mean(CNN I)':>12s} | {'\u0394I(SAE-CNN)':>12s} | "
            f"{'perm p':>10s} | {'n_perm':>8s}"
        )
        print(moran_hdr)
        print("-" * 80)

        morans_pvals_for_plot = defaultdict(dict)

        for mut in mutations:
            cnn_morans = morans_data.get("CNN", {}).get(mut, {})
            sae_morans = morans_data.get("SAE", {}).get(mut, {})
            all_ks = sorted(set(cnn_morans.keys()) | set(sae_morans.keys()))
            for k in all_ks:
                cnn_I = cnn_morans.get(k, [])
                sae_I = sae_morans.get(k, [])
                if len(cnn_I) < 2 or len(sae_I) < 2:
                    print(
                        f"{mut:8s}  {k:4d} | not enough data "
                        f"(n_CNN={len(cnn_I)}, n_SAE={len(sae_I)})"
                    )
                    continue
                delta, p, n_act, _ = permutation_test_morans_i(
                    cnn_I, sae_I, n_perm=n_perm_moran
                )
                star = (
                    "***"
                    if p < 0.001
                    else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
                )
                morans_pvals_for_plot[mut][k] = p
                print(
                    f"{mut:8s}  {k:4d} | {np.mean(sae_I):12.4f} | "
                    f"{np.mean(cnn_I):12.4f} | {delta:12.4f} | "
                    f"{p:.4f} {star:5s} | {n_act:8d}"
                )

        # ── label (모든 plot 공통으로 사용) ──
        exp_label = args.experiment
        if log2fc_filter >= 0:
            exp_label = f"{args.experiment}_log2fc{log2fc_filter}"

        # Moran's I trend plot (K vs I per mutation, CNN vs SAE)
        plot_morans_i_trend(
            morans_data,
            mutations,
            perm_pvals=morans_pvals_for_plot,
            output_dir=out_dir,
            dpi=args.dpi,
            experiment=exp_label,
        )

        plot_trend(
            mean_data,
            "Mean",
            mutations,
            out_dir,
            args.dpi,
            exp_label,
            mwu_pvals=mwu_pvals_for_plot,
        )
        plot_trend(
            median_data,
            "Median",
            mutations,
            out_dir,
            args.dpi,
            exp_label,
            mwu_pvals=mwu_pvals_for_plot,
        )

    print(f"\n  Output: {out_dir}")


if __name__ == "__main__":
    main()
