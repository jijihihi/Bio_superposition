# ==============================================================================
# Local Linearity Verification — KNN cell_death Rate Standard Deviation
#




#

#
# Usage (Colab):
# !python -m cell_death.local_knn_std \
#     --cnn_cache "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/MoCo_seed87/CNN_GAP/cnn_gap_stage5_out_all.npz" \
#     --sae_cache "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/MoCo_seed87/SAE_seed856_no_L2norm_loss/features_cache_stage5_out_normrestored_all_no_SAE_GAP_L2_norm_again_d8192_sp800.npz" \

#     --k_neighbors 5 10 15 \
#     --n_permutations 1000 \
#     --gap_l2_norm \
#     --dead_threshold 1e-5 \
#     --output_dir "/content/local_linearity"
# ==============================================================================

# dpt. L2 norm → DE filter → log_std → PCA → KNN (Euclidean)



import argparse
import json
import os
import sys

import matplotlib
import numpy as np

_IN_COLAB = "google.colab" in sys.modules
if not _IN_COLAB:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from sklearn.neighbors import NearestNeighbors

from run_CNN.logging_utils import SUPERCLASS_MAP, get_logger

logger = get_logger("local_knn_std")

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
sns.set_style("ticks")







# ==============================================================================
# Argument Parser
# ==============================================================================
def get_args():
    p = argparse.ArgumentParser(
        description="Local linearity verification: KNN cell_death rate std vs class global std"
    )
    p.add_argument(
        "--cnn_cache", type=str, default="", help="Path to CNN GAP .npz cache (X_gap)"
    )
    p.add_argument(
        "--sae_cache",
        type=str,
        default="",
        help="Path to SAE .npz cache (X_all + usage_ema)",
    )
    p.add_argument("--cell_death_csv", type=str, required=True)
    p.add_argument("--dead_threshold", type=float, default=1e-5)
    p.add_argument(
        "--gap_l2_norm",
        action="store_true",
        help="Apply L2 normalization to feature vectors (useful for GAP)",
    )
    p.add_argument(
        "--pre_l2_norm",
        action="store_true",
        help="Apply per-image L2 normalization BEFORE any other processing "
        "(divide_hw, gap_l2_norm, feature norm). "
        "Matches old F.normalize(pooled) in extract_features.py.",
    )
    p.add_argument(
        "--divide_hw",
        type=int,
        default=0,
        help="Divide features by H*W to convert sum→mean (e.g. 256 for 16x16). "
        "Applied before any normalization.",
    )

    # Normalization (applied AFTER DE filtering)
    p.add_argument(
        "--norm",
        type=str,
        default="",
        help="Feature normalization: '', 'log', 'std', 'log_std'. "
        "Applied AFTER DE filtering. Default: '' (none).",
    )



    # PCA (applied AFTER normalization, BEFORE KNN — matches dpt.py)
    p.add_argument(
        "--pca_dim",
        type=int,
        default=0,
        help="PCA dimensions after norm, before KNN. "
        "0 = no PCA (default). Set to 50 to match DPT pipeline.",
    )

    p.add_argument(
        "--k_neighbors",
        type=int,
        nargs="+",
        default=[10, 20, 50],
        help="K values for KNN (multiple for sweep). Default: 10 20 50",
    )
    p.add_argument(
        "--n_permutations",
        type=int,
        default=0,
        help="Number of permutations for null distribution (default: 1000)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument(
        "--n_bootstrap",
        type=int,
        default=0,
        help="Bootstrap iterations for Moran's I CNN vs SAE comparison. "
        "0 = skip. 999 = standard. Default: 0",
    )
    p.add_argument(
        "--samples_per_class",
        type=int,
        default=0,
        help="Max samples per class (0 = use ALL). "
        "Prioritizes samples with valid cell_death. "
        "Set to 5000 to match dpt.py default.",
    )

    return p.parse_args()


# ==============================================================================
# Load features (reuse pattern from dpt.py main)
# ==============================================================================
def load_cache(cache_path, dead_threshold):
    """Load feature cache. Returns X, lines, uids, label.
    No L2 norm here — handled uniformly in main() to match dpt.py."""
    from trajectory_inference_pipeline.trajectory_utils import load_features_cache

    data = np.load(cache_path, allow_pickle=True)
    cache_keys = list(data.keys())

    if "X_gap" in data:
        X = data["X_gap"]
        lines = (
            data["lines"].astype(str)
            if data["lines"].dtype.kind != "U"
            else data["lines"]
        )
        uids = (
            data["uids"].astype(str) if data["uids"].dtype.kind != "U" else data["uids"]
        )
        label = "CNN"
        logger.info(f"  Detected CNN GAP cache: {X.shape}")
    elif "X_all" in data:
        X, _, lines, uids, _, _ = load_features_cache(cache_path, dead_threshold)
        label = "SAE"
    else:
        raise ValueError(f"Unknown cache format. Keys: {cache_keys}")

    return X, lines, uids, label


# ==============================================================================
# Core: compute KNN local std ratio for one feature set
# ==============================================================================
def compute_local_std_ratios(X, cell_death, k):
    """
    For each sample, find K nearest neighbors in feature space,
    compute std of their cell_death rates.
    Return ratio = local_std / global_std for each sample.

    Parameters
    ----------
    X : np.ndarray (N, d) — feature matrix (only valid cell_death samples)
    cell_death : np.ndarray (N,) — cell_death rates (no NaN)
    k : int — number of neighbors

    Returns
    -------
    local_stds : np.ndarray (N,) — std of KNN neighbors' cell_death rates
    global_std : float — std of all cell_death rates
    ratios : np.ndarray (N,) — local_std / global_std
    """
    n = len(X)
    k_actual = min(k, n - 1)
    if k_actual < 2:
        return np.full(n, np.nan), np.nan, np.full(n, np.nan)

    # k+1 because the sample itself is included as its own neighbor
    nn = NearestNeighbors(n_neighbors=k_actual + 1, metric="euclidean", n_jobs=-1)
    nn.fit(X)
    _, indices = nn.kneighbors(X)

    # indices[:, 0] is the sample itself → use indices[:, 1:]
    neighbor_indices = indices[:, 1:]  # (N, k_actual)

    # Compute local std for each sample's neighbors
    neighbor_cell_death = cell_death[neighbor_indices]  # (N, k_actual)
    local_stds = np.std(neighbor_cell_death, axis=1)  # (N,)

    global_std = np.std(cell_death)
    ratios = local_stds / max(global_std, 1e-12)

    return local_stds, global_std, ratios


# ==============================================================================
# Global Moran's I from KNN adjacency
# ==============================================================================
def compute_global_morans_i(X, cell_death, k):
    """Compute Global Moran's I on KNN adjacency graph.

    Moran's I measures spatial autocorrelation: do neighbors in feature space
    have similar cell_death values?

    I = (N / W) * (sum_i sum_j w_ij (x_i - xbar)(x_j - xbar)) / (sum_i (x_i - xbar)^2)

    Returns
    -------
    I : float — Moran's I statistic (>0 = positive autocorrelation)
    z : float — z-score under normality assumption
    p : float — two-sided p-value
    expected : float — E[I] under null = -1/(N-1)
    """
    n = len(X)
    k_actual = min(k, n - 1)
    if k_actual < 2 or n < 10:
        return np.nan, np.nan, np.nan, np.nan

    nn = NearestNeighbors(n_neighbors=k_actual + 1, metric="euclidean", n_jobs=-1)
    nn.fit(X)
    _, indices = nn.kneighbors(X)
    neighbor_indices = indices[:, 1:]  # exclude self

    xbar = np.mean(cell_death)
    z_vals = cell_death - xbar  # deviations
    denom = np.sum(z_vals**2)

    if denom < 1e-15:
        return np.nan, np.nan, np.nan, np.nan

    # Binary adjacency: w_ij = 1 if j in KNN(i)
    W = n * k_actual  # total weight (each row has k_actual neighbors)

    # Numerator: sum over all (i, j) pairs where j is neighbor of i
    numer = 0.0
    for i in range(n):
        numer += np.sum(z_vals[i] * z_vals[neighbor_indices[i]])

    I = (n / W) * (numer / denom)

    # Expected value under null
    expected = -1.0 / (n - 1)

    # Variance under normality assumption (for z-test)
    S1 = 2 * W  # sum of (w_ij + w_ji)^2 — for binary symmetric approx
    S2 = n * (2 * k_actual) ** 2  # (sum_j w_ij + sum_i w_ij)^2 per node
    # Simplified variance formula
    var_I = (
        n * ((n**2 - 3 * n + 3) * S1 - n * S2 + 3 * W**2)
        - (n**2 - n) * S1
        + 2 * n * S2
        - 6 * W**2
    ) / ((n - 1) * (n - 2) * (n - 3) * W**2 + 1e-15)
    # Use simpler approximation for large n
    var_I_simple = (1.0 / (n - 1)) * (1 - expected**2) if n > 30 else max(var_I, 1e-12)
    var_I_simple = max(var_I_simple, 1e-12)

    z_score = (I - expected) / np.sqrt(var_I_simple)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # two-sided

    return I, z_score, p_value, expected


# ==============================================================================
# Bootstrap paired test: Moran's I difference (CNN vs SAE)
# ==============================================================================
def bootstrap_morans_i_diff(X_a, X_b, cell_death, k, n_boot=999, seed=42, ci_alpha=0.05):
    """Bootstrap test for ΔI = I(B) - I(A) using paired resampling.

    Same images, different feature spaces → paired bootstrap.
    Resamples the SAME indices for both A and B to preserve pairing.

    Returns
    -------
    delta_real : float — I_B - I_A (observed)
    ci_lo, ci_hi : float — confidence interval
    p_value : float — P(ΔI ≤ 0) from bootstrap (one-sided: is B > A?)
    """
    n = len(cell_death)
    I_a = compute_global_morans_i(X_a, cell_death, k)[0]
    I_b = compute_global_morans_i(X_b, cell_death, k)[0]
    delta_real = I_b - I_a

    rng = np.random.RandomState(seed)
    delta_boots = np.zeros(n_boot)

    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        I_a_b = compute_global_morans_i(X_a[idx], cell_death[idx], k)[0]
        I_b_b = compute_global_morans_i(X_b[idx], cell_death[idx], k)[0]
        delta_boots[b] = I_b_b - I_a_b

    lo = np.percentile(delta_boots, 100 * ci_alpha / 2)
    hi = np.percentile(delta_boots, 100 * (1 - ci_alpha / 2))
    # One-sided p: fraction of bootstrap where SAE ≤ CNN
    p_value = np.mean(delta_boots <= 0)

    return delta_real, lo, hi, p_value, delta_boots


# ==============================================================================
# Permutation test
# ==============================================================================
def permutation_test_ratio(X, cell_death, k, n_permutations, seed):
    """
    Permutation test: shuffle cell_death labels, compute mean local_std/global_std.
    Compare real mean ratio against null distribution.

    Returns
    -------
    real_mean_ratio : float
    null_ratios : np.ndarray (n_permutations,) — mean ratio under null
    p_value : float
    """
    _, _, real_ratios = compute_local_std_ratios(X, cell_death, k)
    real_mean_ratio = np.nanmean(real_ratios)

    rng = np.random.RandomState(seed)
    null_mean_ratios = np.zeros(n_permutations)

    for i in range(n_permutations):
        perm_cell_death = rng.permutation(cell_death)
        _, _, perm_ratios = compute_local_std_ratios(X, perm_cell_death, k)
        null_mean_ratios[i] = np.nanmean(perm_ratios)

    # p-value: fraction of null ≤ real (one-sided, lower is better)
    p_value = (np.sum(null_mean_ratios <= real_mean_ratio) + 1) / (n_permutations + 1)

    return real_mean_ratio, null_mean_ratios, p_value


# ==============================================================================
# Plot 1: Violin/Box — CNN vs SAE ratio distributions
# ==============================================================================
def plot_ratio_comparison(results_by_source, mutation, k, output_path, dpi=200):
    """
    Violin + strip plot comparing local_std/global_std distributions
    between CNN and SAE for a given mutation and k.
    """
    fig, ax = plt.subplots(figsize=(5, 5))

    plot_data = []
    labels = []
    for source_label, res in results_by_source.items():
        ratios = res["ratios"]
        valid = ratios[~np.isnan(ratios)]
        plot_data.append(valid)
        labels.append(source_label)

    colors = {"CNN": "#4C72B0", "SAE": "#DD8452"}
    box_colors = [colors.get(lbl, "gray") for lbl in labels]

    parts = ax.violinplot(
        plot_data,
        positions=range(len(labels)),
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for pc, c in zip(parts["bodies"], box_colors):
        pc.set_facecolor(c)
        pc.set_alpha(0.3)

    bp = ax.boxplot(
        plot_data,
        positions=range(len(labels)),
        widths=0.3,
        patch_artist=True,
        showfliers=False,
        zorder=3,
    )
    for patch, c in zip(bp["boxes"], box_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    for element in ["whiskers", "caps", "medians"]:
        for line in bp[element]:
            line.set_color("black")
            line.set_linewidth(1.0)

    # Reference line at 1.0 (no improvement over global)
    ax.axhline(
        1.0, color="red", linewidth=1.5, linestyle="--", alpha=0.7, label="Global std"
    )

    # Annotations
    for i, (lbl, res) in enumerate(results_by_source.items()):
        mean_r = res["mean_ratio"]
        pval = res.get("perm_pval", None)
        pval_str = f"p={pval:.4f}" if pval is not None else ""
        ax.text(
            i,
            ax.get_ylim()[1] * 0.95,
            f"mean={mean_r:.3f}\n{pval_str}",
            ha="center",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
        )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Local Std / Global Std", fontsize=11)
    ax.set_title(
        f"{mutation} — Local Linearity (k={k})", fontsize=13, fontweight="bold"
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.2, axis="y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    # SVG version
    svg_path = output_path.replace(".png", ".svg")
    fig.savefig(svg_path, bbox_inches="tight")
    if _IN_COLAB:
        plt.show()
    plt.close(fig)


# ==============================================================================
# Plot 2: Permutation null distribution
# ==============================================================================
def plot_permutation_null(
    real_ratio, null_ratios, source_label, mutation, k, pval, output_path, dpi=200
):
    """Histogram of permutation null mean-ratio with real marked."""
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.hist(
        null_ratios,
        bins=50,
        color="#888888",
        alpha=0.7,
        edgecolor="white",
        label=f"Null (n={len(null_ratios)})",
    )
    ax.axvline(
        real_ratio,
        color="red",
        linewidth=2,
        linestyle="--",
        label=f"Real mean ratio={real_ratio:.4f}",
    )

    ax.set_xlabel("Mean(Local Std / Global Std)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(
        f"{source_label} — {mutation} — Permutation (k={k}, p={pval:.4f})",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if _IN_COLAB:
        plt.show()
    plt.close(fig)


# ==============================================================================
# Plot 3: K sweep — ratio vs k for both sources
# ==============================================================================
def plot_k_sweep(sweep_results, mutation, output_path, dpi=200):
    """
    Line plot: mean ratio vs k, for CNN and SAE.
    sweep_results: dict[source_label] → list of (k, mean_ratio, pval)
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    colors = {"CNN": "#4C72B0", "SAE": "#DD8452"}

    for source_label, kv_list in sweep_results.items():
        ks = [r[0] for r in kv_list]
        means = [r[1] for r in kv_list]
        pvals = [r[2] for r in kv_list]
        c = colors.get(source_label, "gray")
        ax.plot(
            ks,
            means,
            "o-",
            color=c,
            linewidth=2,
            markersize=6,
            label=source_label,
            alpha=0.9,
        )

        # Annotate p-values
        for k_val, m, p in zip(ks, means, pvals):
            if p is not None:
                star = (
                    "***"
                    if p < 0.001
                    else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
                )
                ax.annotate(
                    star,
                    (k_val, m),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=8,
                    color=c,
                )

    ax.axhline(
        1.0,
        color="red",
        linewidth=1.2,
        linestyle="--",
        alpha=0.6,
        label="Random (ratio=1)",
    )
    ax.set_xlabel("k (number of neighbors)", fontsize=11)
    ax.set_ylabel("Mean(Local Std / Global Std)", fontsize=11)
    ax.set_title(f"{mutation} — Local Linearity vs k", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    svg_path = output_path.replace(".png", ".svg")
    fig.savefig(svg_path, bbox_inches="tight")
    if _IN_COLAB:
        plt.show()
    plt.close(fig)


# ==============================================================================
# Plot 4: Moran's I — K sweep for CNN vs SAE
# ==============================================================================
def plot_morans_k_sweep(morans_data, mutation, output_path, dpi=200):
    """
    Line plot: Global Moran's I vs k, for CNN and SAE.
    morans_data: dict[source_label] → list of (k, I, z, p)
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    colors = {"CNN": "#4C72B0", "SAE": "#DD8452"}

    for source_label, kv_list in morans_data.items():
        ks = [r[0] for r in kv_list]
        morans_vals = [r[1] for r in kv_list]
        pvals = [r[3] for r in kv_list]
        c = colors.get(source_label, "gray")
        ax.plot(
            ks,
            morans_vals,
            "o-",
            color=c,
            linewidth=2,
            markersize=7,
            label=source_label,
            alpha=0.9,
        )

        # Annotate significance
        for k_val, m_i, p in zip(ks, morans_vals, pvals):
            if p is not None and not np.isnan(p):
                star = (
                    "***"
                    if p < 0.001
                    else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
                )
                ax.annotate(
                    star,
                    (k_val, m_i),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=9,
                    color=c,
                    fontweight="bold",
                )

    ax.axhline(0, color="gray", linewidth=1, linestyle=":", alpha=0.5)
    ax.set_xlabel("k (number of neighbors)", fontsize=11)
    ax.set_ylabel("Global Moran's I", fontsize=11)
    ax.set_title(
        f"{mutation} — Spatial Autocorrelation (Moran's I)",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    sns.despine()

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(output_path.replace(".png", ".svg"), bbox_inches="tight")
    if _IN_COLAB:
        plt.show()
    plt.close(fig)


# ==============================================================================
# Main
# ==============================================================================
def main():
    args = get_args()
    np.random.seed(args.seed)

    from sklearn.decomposition import PCA

    from trajectory_inference_pipeline.trajectory_utils import (
        apply_normalization, compute_cv_per_neuron, compute_de_neurons,
        load_and_match_cell_death)

    if not args.cnn_cache and not args.sae_cache:
        raise ValueError("At least one of --cnn_cache or --sae_cache must be provided")

    # ── Output directory ──
    if args.output_dir:
        out_dir = args.output_dir
    else:
        ref_cache = args.cnn_cache or args.sae_cache
        out_dir = os.path.join(os.path.dirname(ref_cache), "local_knn_std")
    os.makedirs(out_dir, exist_ok=True)

    # ── Load feature caches ──────────────────────────────────────────
    # Matches dpt.py main(): load raw → pre_l2_norm → divide_hw → gap_l2_norm
    # Both CNN and SAE go through the same pipeline.
    sources = {}  # label → (X, superclasses, uids, cell_death)

    def _load_and_preprocess(cache_path):
        """Load cache and apply pre-processing exactly like dpt.py main()."""
        X, lines, uids, label = load_cache(cache_path, args.dead_threshold)

        # Optional: per-image L2 normalize BEFORE everything else
        if args.pre_l2_norm:
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1e-12, norms)
            X = X / norms
            logger.info(f"  Applied pre-L2 normalization (before filters/norm)")

        # Optional: divide by H*W (sum → mean conversion)
        if args.divide_hw > 0:
            X = X / args.divide_hw
            logger.info(f"  Divided by H*W={args.divide_hw} (sum→mean)")

        # Optional: L2 normalize (after divide_hw)
        if args.gap_l2_norm:
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1e-12, norms)
            X = X / norms
            logger.info(f"  Applied L2 normalization")

        superclasses = [SUPERCLASS_MAP.get(ln, ln) for ln in lines]
        cell_death = load_and_match_cell_death(args.cell_death_csv, uids)
        return X, superclasses, uids, cell_death, label

    if args.cnn_cache:
        logger.info("Loading CNN cache...")
        X_cnn, sc_cnn, uids_cnn, apop_cnn, _ = _load_and_preprocess(args.cnn_cache)
        sources["CNN"] = (X_cnn, sc_cnn, uids_cnn, apop_cnn)
        logger.info(f"  CNN features: {X_cnn.shape}")

    if args.sae_cache:
        logger.info("Loading SAE cache...")
        X_sae, sc_sae, uids_sae, apop_sae, _ = _load_and_preprocess(args.sae_cache)
        sources["SAE"] = (X_sae, sc_sae, uids_sae, apop_sae)
        logger.info(f"  SAE features: {X_sae.shape}")

    # ── Mutations to analyze ──
    mutations = ["SNCA", "GBA", "LRRK2"]

    # ── Main analysis ──
    all_results = {}  # (source, mutation, k) → result dict
    k_sweep_data = {}  # mutation → {source → [(k, mean_ratio, pval), ...]}
    morans_sweep_data = {}  # mutation → {source → [(k, I, z, p), ...]}
    feature_cache = {}  # (source, mutation) → (X_mut, apop_mut) for paired bootstrap

    for source_label, (X_raw, superclasses, uids, cell_death) in sources.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"  Source: {source_label}")
        logger.info(f"{'='*60}")

        superclasses_arr = np.array(superclasses)

        # ── Subsample per class (prioritize valid cell_death) ──
        spc = args.samples_per_class
        if spc > 0:
            rng_sub = np.random.RandomState(args.seed)
            keep_indices = []
            for cls in sorted(np.unique(superclasses_arr)):
                cls_idx = np.where(superclasses_arr == cls)[0]
                valid_mask_cls = ~np.isnan(cell_death[cls_idx])
                valid_idx = cls_idx[valid_mask_cls]
                invalid_idx = cls_idx[~valid_mask_cls]
                ordered = np.concatenate([valid_idx, invalid_idx])
                n_take = min(spc, len(ordered))
                chosen = rng_sub.choice(
                    ordered[: max(n_take, len(valid_idx))],
                    size=min(n_take, len(ordered)),
                    replace=False,
                )
                keep_indices.extend(chosen.tolist())
                logger.info(
                    f"    Subsample {cls}: {len(cls_idx)} → {len(chosen)} "
                    f"(valid apop: {valid_mask_cls.sum()})"
                )
            keep_indices = sorted(keep_indices)
            X_raw = X_raw[keep_indices]
            superclasses = [superclasses[i] for i in keep_indices]
            superclasses_arr = np.array(superclasses)
            cell_death = cell_death[keep_indices]
            uids = (
                [uids[i] for i in keep_indices]
                if isinstance(uids, list)
                else uids[keep_indices]
            )
            logger.info(f"    After subsampling: {X_raw.shape[0]} samples")

        X = X_raw.copy()  # Don't modify original
        X_use = X

        # ── Per-mutation loop ─────────────────────────────────────────
        for mut in mutations:
            logger.info(f"\n  ── {source_label} / {mut} ──")

            if mut not in k_sweep_data:
                k_sweep_data[mut] = {}

            mut_mask = superclasses_arr == mut
            valid_mask = mut_mask & np.isfinite(cell_death)
            n_valid = int(valid_mask.sum())

            if n_valid < 10:
                logger.warning(
                    f"    Too few valid samples for {mut} ({n_valid}), skipping"
                )
                continue

            # ── Normalization (after DE filtering) ──
            if args.norm:
                X_use = apply_normalization(X_use, args.norm)
                logger.info(f"    Applied normalization: '{args.norm}'")

            # ── PCA (after norm, before KNN — matches dpt.py) ──
            if args.pca_dim > 0 and X_use.shape[1] > args.pca_dim:
                # Fit more PCs for diagnostics, keep pca_dim for downstream
                n_plot = min(30, X_use.shape[1], X_use.shape[0] - 1)
                n_pca = min(args.pca_dim, X_use.shape[1], X_use.shape[0] - 1)
                n_fit = max(n_pca, n_plot)
                pca = PCA(n_components=n_fit, random_state=args.seed)
                X_all_pcs = pca.fit_transform(X_use)
                X_use = X_all_pcs[:, :n_pca]  # keep only pca_dim for downstream

                var_ratios = pca.explained_variance_ratio_
                cum_var = np.cumsum(var_ratios)
                logger.info(
                    f"    PCA: {pca.n_features_in_}D → {n_pca}D "
                    f"(total explained var: {cum_var[n_pca-1]:.1%})"
                )
                for i in range(n_fit):
                    marker = " ◀ cutoff" if i == n_pca - 1 else ""
                    logger.info(
                        f"      PC{i+1:3d}: {var_ratios[i]:.4f}  "
                        f"(cumulative: {cum_var[i]:.4f}){marker}"
                    )

                # Scree plot — save + show in Colab
                fig_sc, ax_sc = plt.subplots(figsize=(8, 4))
                x_pos = np.arange(1, n_fit + 1)
                ax_sc.bar(
                    x_pos,
                    var_ratios,
                    color="#5B9BD5",
                    alpha=0.7,
                    edgecolor="white",
                    label="Individual",
                )
                ax_sc.plot(
                    x_pos,
                    cum_var,
                    "o-",
                    color="#ED7D31",
                    linewidth=2,
                    markersize=4,
                    label="Cumulative",
                )
                ax_sc.axvline(
                    n_pca + 0.5,
                    color="red",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.7,
                    label=f"Cutoff (n={n_pca})",
                )
                ax_sc.set_xlabel("Principal Component", fontsize=11)
                ax_sc.set_ylabel("Explained Variance Ratio", fontsize=11)
                ax_sc.set_title(
                    f"PCA Scree — {source_label} / {mut} "
                    f"(using {n_pca}/{n_fit} PCs, "
                    f"kept var={cum_var[n_pca-1]:.1%})",
                    fontsize=12,
                    fontweight="bold",
                )
                ax_sc.legend(fontsize=9)
                ax_sc.set_ylim(0, max(var_ratios[0] * 1.15, cum_var[-1] * 1.05))
                ax_sc.grid(True, alpha=0.2, axis="y")
                sns.despine()
                fig_sc.tight_layout()
                scree_path = os.path.join(
                    out_dir, f"pca_scree_{source_label}_{mut}.png"
                )
                fig_sc.savefig(scree_path, dpi=args.dpi, bbox_inches="tight")
                svg_path = scree_path.replace(".png", ".svg")
                fig_sc.savefig(svg_path, format="svg", bbox_inches="tight")
                logger.info(f"    Scree plot: {scree_path}")
                if _IN_COLAB:
                    plt.show()
                plt.close(fig_sc)

            X_mut = X_use[valid_mask]
            apop_mut = cell_death[valid_mask]
            global_std = np.std(apop_mut)
            logger.info(
                f"    n={n_valid}, features={X_mut.shape[1]}, global_std={global_std:.6f}"
            )

            # Store for paired bootstrap comparison
            feature_cache[(source_label, mut)] = (X_mut, apop_mut)

            sweep_for_source = []
            morans_for_source = []

            for k in args.k_neighbors:
                logger.info(f"\n    k={k}")

                # Compute local std ratios
                local_stds, g_std, ratios = compute_local_std_ratios(X_mut, apop_mut, k)
                mean_ratio = np.nanmean(ratios)
                median_ratio = np.nanmedian(ratios)

                logger.info(f"      mean(local/global) = {mean_ratio:.4f}")
                logger.info(f"      median(local/global) = {median_ratio:.4f}")

                # ── Global Moran's I ──
                morans_I, morans_z, morans_p, morans_exp = compute_global_morans_i(
                    X_mut, apop_mut, k
                )
                logger.info(
                    f"      Moran's I = {morans_I:.4f} "
                    f"(z={morans_z:.2f}, p={morans_p:.2e}, "
                    f"E[I]={morans_exp:.4f})"
                )
                morans_for_source.append((k, morans_I, morans_z, morans_p))

                # One-sample Wilcoxon signed-rank test: ratio < 1.0?
                valid_ratios = ratios[~np.isnan(ratios)]
                if len(valid_ratios) > 10:
                    try:
                        wilcox_stat, wilcox_p = stats.wilcoxon(
                            valid_ratios - 1.0, alternative="less"
                        )
                        logger.info(
                            f"      Wilcoxon (ratio < 1): stat={wilcox_stat:.2f}, "
                            f"p={wilcox_p:.2e}"
                        )
                    except Exception:
                        wilcox_p = np.nan
                else:
                    wilcox_p = np.nan

                # Permutation test
                perm_pval = None
                null_ratios = np.array([])
                if args.n_permutations > 0:
                    logger.info(
                        f"      Permutation test ({args.n_permutations} perms)..."
                    )
                    real_mr, null_ratios, perm_pval = permutation_test_ratio(
                        X_mut, apop_mut, k, args.n_permutations, args.seed
                    )
                    logger.info(
                        f"      Perm p={perm_pval:.4f} "
                        f"(null mean={np.mean(null_ratios):.4f}, "
                        f"real={real_mr:.4f})"
                    )

                result = {
                    "ratios": ratios,
                    "mean_ratio": mean_ratio,
                    "median_ratio": median_ratio,
                    "global_std": g_std,
                    "local_stds": local_stds,
                    "wilcoxon_p": wilcox_p,
                    "perm_pval": perm_pval,
                    "null_ratios": null_ratios,
                    "n": n_valid,
                    "k": k,
                    "morans_I": morans_I,
                    "morans_z": morans_z,
                    "morans_p": morans_p,
                }
                all_results[(source_label, mut, k)] = result
                sweep_for_source.append((k, mean_ratio, perm_pval))

                # Plot permutation null for each (source, mutation, k)
                if args.n_permutations > 0 and len(null_ratios) > 0:
                    plot_permutation_null(
                        mean_ratio,
                        null_ratios,
                        source_label,
                        mut,
                        k,
                        perm_pval,
                        os.path.join(
                            out_dir, f"perm_null_{source_label}_{mut}_k{k}.png"
                        ),
                        dpi=args.dpi,
                    )

            k_sweep_data[mut][source_label] = sweep_for_source
            if mut not in morans_sweep_data:
                morans_sweep_data[mut] = {}
            morans_sweep_data[mut][source_label] = morans_for_source

    # ── Per-mutation comparison plots (CNN vs SAE) ──
    for mut in mutations:
        for k in args.k_neighbors:
            results_by_source = {}
            for source_label in sources.keys():
                key = (source_label, mut, k)
                if key in all_results:
                    results_by_source[source_label] = all_results[key]

            if len(results_by_source) > 0:
                plot_ratio_comparison(
                    results_by_source,
                    mut,
                    k,
                    os.path.join(out_dir, f"ratio_comparison_{mut}_k{k}.png"),
                    dpi=args.dpi,
                )

        # ── K sweep plot per mutation ──
        if k_sweep_data.get(mut):
            plot_k_sweep(
                k_sweep_data[mut],
                mut,
                os.path.join(out_dir, f"k_sweep_{mut}.png"),
                dpi=args.dpi,
            )

        # ── Moran's I K sweep plot per mutation ──
        if morans_sweep_data.get(mut):
            plot_morans_k_sweep(
                morans_sweep_data[mut],
                mut,
                os.path.join(out_dir, f"morans_I_k_sweep_{mut}.png"),
                dpi=args.dpi,
            )

    # ── Bootstrap Moran's I paired comparison (CNN vs SAE) ──
    bootstrap_results = {}  # (mutation, k) → {delta, ci_lo, ci_hi, p}
    if args.n_bootstrap > 0 and "CNN" in sources and "SAE" in sources:
        logger.info(f"\n{'='*60}")
        logger.info(
            f"  Bootstrap Moran's I: CNN vs SAE ({args.n_bootstrap} iterations)"
        )
        logger.info(f"{'='*60}")

        for mut in mutations:
            key_cnn = ("CNN", mut)
            key_sae = ("SAE", mut)
            if key_cnn not in feature_cache or key_sae not in feature_cache:
                continue

            X_cnn_mut, apop_cnn_mut = feature_cache[key_cnn]
            X_sae_mut, apop_sae_mut = feature_cache[key_sae]

            # Align samples: use intersection of valid indices
            # (Both should use same valid_mask since cell_death is same)
            n_min = min(len(apop_cnn_mut), len(apop_sae_mut))
            if n_min < 20:
                logger.warning(f"    {mut}: too few paired samples ({n_min}), skip")
                continue

            # If sample counts match, they're already aligned
            # If not, truncate to min (edge case)
            X_c = X_cnn_mut[:n_min]
            X_s = X_sae_mut[:n_min]
            apop = apop_cnn_mut[:n_min]

            for k in args.k_neighbors:
                logger.info(f"\n    {mut} k={k}: bootstrapping...")
                delta, ci_lo, ci_hi, p_boot, _ = bootstrap_morans_i_diff(
                    X_c, X_s, apop, k, n_boot=args.n_bootstrap, seed=args.seed
                )

                star = (
                    "***"
                    if p_boot < 0.001
                    else "**" if p_boot < 0.01 else "*" if p_boot < 0.05 else "n.s."
                )
                logger.info(
                    f"      ΔI(SAE-CNN) = {delta:.4f}  "
                    f"95%% CI [{ci_lo:.4f}, {ci_hi:.4f}]  "
                    f"p = {p_boot:.4f} {star}"
                )
                bootstrap_results[(mut, k)] = {
                    "delta": delta,
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                    "p_boot": p_boot,
                }

    # ── Summary table ──
    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY — Local Linearity Verification")
    logger.info(f"{'='*80}")
    logger.info(
        f"  {'Source':6s} {'Mutation':8s} {'k':>4s} {'n':>6s} "
        f"{'MeanRatio':>10s} {'MedRatio':>10s} {'WilcoxP':>10s} {'PermP':>10s} "
        f"{'MoranI':>8s} {'MoranZ':>8s} {'MoranP':>10s}"
    )
    logger.info("  " + "-" * 100)

    for (source, mut, k), res in sorted(all_results.items()):
        wilcox_str = (
            f"{res['wilcoxon_p']:.2e}"
            if not np.isnan(res.get("wilcoxon_p", np.nan))
            else "N/A"
        )
        perm_str = (
            f"{res['perm_pval']:.4f}" if res.get("perm_pval") is not None else "N/A"
        )
        mi = res.get("morans_I", np.nan)
        mz = res.get("morans_z", np.nan)
        mp = res.get("morans_p", np.nan)
        mi_str = f"{mi:.4f}" if not np.isnan(mi) else "N/A"
        mz_str = f"{mz:.2f}" if not np.isnan(mz) else "N/A"
        mp_str = f"{mp:.2e}" if not np.isnan(mp) else "N/A"
        logger.info(
            f"  {source:6s} {mut:8s} {k:4d} {res['n']:6d} "
            f"{res['mean_ratio']:10.4f} {res['median_ratio']:10.4f} "
            f"{wilcox_str:>10s} {perm_str:>10s} "
            f"{mi_str:>8s} {mz_str:>8s} {mp_str:>10s}"
        )

    # ── Save JSON & Arrays ──
    json_results = []
    for (source, mut, k), res in sorted(all_results.items()):

        # Save per-sample local_stds and ratios for later MWU
        
        npz_filename = os.path.join(out_dir, f"ratios_{source}_{mut}_k{k}.npz")
        np.savez_compressed(
            npz_filename,
            local_stds=res["local_stds"],  
            ratios=res["ratios"],  # local_std / global_std (N,)
            global_std=np.array([res["global_std"]]),  
        )

        json_results.append(
            {
                "source": source,
                "mutation": mut,
                "k": k,
                "n": res["n"],
                "mean_ratio": float(res["mean_ratio"]),
                "median_ratio": float(res["median_ratio"]),
                "global_std": float(res["global_std"]),
                "wilcoxon_p": (
                    float(res["wilcoxon_p"])
                    if not np.isnan(res.get("wilcoxon_p", np.nan))
                    else None
                ),
                "perm_pval": (
                    float(res["perm_pval"])
                    if res.get("perm_pval") is not None
                    else None
                ),
                "morans_I": (
                    float(res.get("morans_I", np.nan))
                    if not np.isnan(res.get("morans_I", np.nan))
                    else None
                ),
                "morans_z": (
                    float(res.get("morans_z", np.nan))
                    if not np.isnan(res.get("morans_z", np.nan))
                    else None
                ),
                "morans_p": (
                    float(res.get("morans_p", np.nan))
                    if not np.isnan(res.get("morans_p", np.nan))
                    else None
                ),
            }
        )

    # Bootstrap comparison results
    boot_json = []
    for (mut, k), bres in sorted(bootstrap_results.items()):
        boot_json.append(
            {
                "mutation": mut,
                "k": k,
                "delta_I_SAE_minus_CNN": float(bres["delta"]),
                "ci_lo_95": float(bres["ci_lo"]),
                "ci_hi_95": float(bres["ci_hi"]),
                "p_bootstrap": float(bres["p_boot"]),
            }
        )

    json_path = os.path.join(out_dir, "local_linearity_results.json")
    with open(json_path, "w") as f:
        json.dump(
            {
                "args": vars(args),
                "results": json_results,
                "bootstrap_morans_comparison": boot_json,
            },
            f,
            indent=2,
        )
    logger.info(f"\n  Saved JSON: {json_path}")
    logger.info(f"  Output: {out_dir}")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()
