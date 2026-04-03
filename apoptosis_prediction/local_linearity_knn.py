# ==============================================================================
# Local Linearity Verification — KNN Apoptosis Rate Standard Deviation
#
# Feature space에서 가까운 이미지들이 비슷한 세포사멸율을 가지는지 검증.
# 이거 잘 보면 ridge regression에서 예측이 된다고 해서 local로 비슷한게 아닐 수 있다.
# 이 스크립트는 KNN 이웃들의 apoptosis rate 표준편차가 해당 클래스 전체
# 표준편차보다 유의미하게 작은지를 검증한다.
#
# CNN feature vector와 SAE feature vector 각각에 대해 수행.
#
# Usage (Colab):
# !python -m apoptosis_prediction.local_linearity_knn \
#     --cnn_cache "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/MoCo_seed87/CNN_GAP/cnn_gap_stage5_out_all.npz" \
#     --sae_cache "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/MoCo_seed87/SAE_seed856_no_L2norm_loss/features_cache_stage5_out_normrestored_all_no_SAE_GAP_L2_norm_again_d8192_sp800.npz" \
#     --apoptosis_csv "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/세포이미지별 사멸율/이미지별_세포사멸율_7200.csv" \
#     --k_neighbors 5 10 15 \
#     --n_permutations 1000 \
#     --gap_l2_norm \
#     --dead_threshold 1e-5 \
#     --output_dir "/content/local_linearity"
# ==============================================================================

import os
import sys
import json
import argparse
import numpy as np

import matplotlib
_IN_COLAB = "google.colab" in sys.modules
if not _IN_COLAB:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.neighbors import NearestNeighbors
from scipy import stats

from sae_project.step02_logging_utils import get_logger, SUPERCLASS_MAP

logger = get_logger("local_linearity_knn")

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
sns.set_style("ticks")


# ==============================================================================
# Argument Parser
# ==============================================================================
def get_args():
    p = argparse.ArgumentParser(
        description="Local linearity verification: KNN apoptosis rate std vs class global std"
    )
    p.add_argument("--cnn_cache", type=str, default="",
                   help="Path to CNN GAP .npz cache (X_gap)")
    p.add_argument("--sae_cache", type=str, default="",
                   help="Path to SAE .npz cache (X_all + usage_ema)")
    p.add_argument("--apoptosis_csv", type=str, required=True)
    p.add_argument("--dead_threshold", type=float, default=1e-5)
    p.add_argument("--gap_l2_norm", action="store_true",
                   help="Apply L2 normalization to CNN feature vectors")

    # Normalization (applied AFTER DE filtering)
    p.add_argument("--norm", type=str, default="",
                   help="Feature normalization: '', 'log', 'std', 'log_std'. "
                        "Applied AFTER DE filtering. Default: '' (none).")

    # DE / CV neuron filtering
    p.add_argument("--filter_mode", type=str, nargs="+", default=["none"],
                   help="Sequential filter: 'cv', 'de', 'none'. "
                        "e.g. '--filter_mode cv de'")
    p.add_argument("--min_cv", type=float, default=0.0,
                   help="Min CV threshold for cv filter (default: 0.0)")
    p.add_argument("--de_adj_p", type=float, default=0.05,
                   help="Adjusted p-value threshold for DE filter (default: 0.05)")
    p.add_argument("--de_min_log2fc", type=float, default=1.0,
                   help="Min |log2FC| for DE filter (default: 1.0)")
    p.add_argument("--de_top_k", type=int, default=0,
                   help="Max DE neurons per mutation (by |log2FC| rank). "
                        "0 = keep all significant.")

    p.add_argument("--k_neighbors", type=int, nargs="+", default=[10, 20, 50],
                   help="K values for KNN (multiple for sweep). Default: 10 20 50")
    p.add_argument("--n_permutations", type=int, default=1000,
                   help="Number of permutations for null distribution (default: 1000)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="")
    p.add_argument("--dpi", type=int, default=200)

    return p.parse_args()


# ==============================================================================
# Load features (reuse pattern from apoptosis_r2_test.py)
# ==============================================================================
def load_cache(cache_path, dead_threshold, apply_l2_norm=False):
    """Load feature cache. Returns X, lines, uids, label."""
    from kendall_correlation_coefficient.dpt_kendall import load_features_cache

    data = np.load(cache_path, allow_pickle=True)
    cache_keys = list(data.keys())

    if "X_gap" in data:
        X = data["X_gap"]
        lines = data["lines"].astype(str) if data["lines"].dtype.kind != 'U' else data["lines"]
        uids = data["uids"].astype(str) if data["uids"].dtype.kind != 'U' else data["uids"]
        label = "CNN"
        logger.info(f"  Detected CNN GAP cache: {X.shape}")
    elif "X_all" in data:
        X, _, lines, uids, _, _ = load_features_cache(cache_path, dead_threshold)
        label = "SAE"
    else:
        raise ValueError(f"Unknown cache format. Keys: {cache_keys}")

    if apply_l2_norm:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-12, norms)
        X = X / norms
        logger.info(f"  Applied L2 normalization: {X.shape}")

    return X, lines, uids, label


# ==============================================================================
# Core: compute KNN local std ratio for one feature set
# ==============================================================================
def compute_local_std_ratios(X, apoptosis, k):
    """
    For each sample, find K nearest neighbors in feature space,
    compute std of their apoptosis rates.
    Return ratio = local_std / global_std for each sample.

    Parameters
    ----------
    X : np.ndarray (N, d) — feature matrix (only valid apoptosis samples)
    apoptosis : np.ndarray (N,) — apoptosis rates (no NaN)
    k : int — number of neighbors

    Returns
    -------
    local_stds : np.ndarray (N,) — std of KNN neighbors' apoptosis rates
    global_std : float — std of all apoptosis rates
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
    neighbor_apoptosis = apoptosis[neighbor_indices]  # (N, k_actual)
    local_stds = np.std(neighbor_apoptosis, axis=1)   # (N,)

    global_std = np.std(apoptosis)
    ratios = local_stds / max(global_std, 1e-12)

    return local_stds, global_std, ratios


# ==============================================================================
# Permutation test
# ==============================================================================
def permutation_test_ratio(X, apoptosis, k, n_permutations, seed):
    """
    Permutation test: shuffle apoptosis labels, compute mean local_std/global_std.
    Compare real mean ratio against null distribution.

    Returns
    -------
    real_mean_ratio : float
    null_ratios : np.ndarray (n_permutations,) — mean ratio under null
    p_value : float
    """
    _, _, real_ratios = compute_local_std_ratios(X, apoptosis, k)
    real_mean_ratio = np.nanmean(real_ratios)

    rng = np.random.RandomState(seed)
    null_mean_ratios = np.zeros(n_permutations)

    for i in range(n_permutations):
        perm_apoptosis = rng.permutation(apoptosis)
        _, _, perm_ratios = compute_local_std_ratios(X, perm_apoptosis, k)
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

    parts = ax.violinplot(plot_data, positions=range(len(labels)),
                          showmeans=False, showmedians=False, showextrema=False)
    for pc, c in zip(parts['bodies'], box_colors):
        pc.set_facecolor(c)
        pc.set_alpha(0.3)

    bp = ax.boxplot(plot_data, positions=range(len(labels)),
                    widths=0.3, patch_artist=True,
                    showfliers=False, zorder=3)
    for patch, c in zip(bp['boxes'], box_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    for element in ['whiskers', 'caps', 'medians']:
        for line in bp[element]:
            line.set_color('black')
            line.set_linewidth(1.0)

    # Reference line at 1.0 (no improvement over global)
    ax.axhline(1.0, color="red", linewidth=1.5, linestyle="--", alpha=0.7,
               label="Global std")

    # Annotations
    for i, (lbl, res) in enumerate(results_by_source.items()):
        mean_r = res["mean_ratio"]
        pval = res.get("perm_pval", None)
        pval_str = f"p={pval:.4f}" if pval is not None else ""
        ax.text(i, ax.get_ylim()[1] * 0.95,
                f"mean={mean_r:.3f}\n{pval_str}",
                ha="center", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Local Std / Global Std", fontsize=11)
    ax.set_title(f"{mutation} — Local Linearity (k={k})", fontsize=13, fontweight="bold")
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
def plot_permutation_null(real_ratio, null_ratios, source_label, mutation, k,
                          pval, output_path, dpi=200):
    """Histogram of permutation null mean-ratio with real marked."""
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.hist(null_ratios, bins=50, color="#888888", alpha=0.7, edgecolor="white",
            label=f"Null (n={len(null_ratios)})")
    ax.axvline(real_ratio, color="red", linewidth=2, linestyle="--",
               label=f"Real mean ratio={real_ratio:.4f}")

    ax.set_xlabel("Mean(Local Std / Global Std)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"{source_label} — {mutation} — Permutation (k={k}, p={pval:.4f})",
                 fontsize=12, fontweight="bold")
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
        ax.plot(ks, means, "o-", color=c, linewidth=2, markersize=6,
                label=source_label, alpha=0.9)

        # Annotate p-values
        for k_val, m, p in zip(ks, means, pvals):
            if p is not None:
                star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
                ax.annotate(star, (k_val, m), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=8, color=c)

    ax.axhline(1.0, color="red", linewidth=1.2, linestyle="--", alpha=0.6,
               label="Random (ratio=1)")
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
# Main
# ==============================================================================
def main():
    args = get_args()
    np.random.seed(args.seed)

    from kendall_correlation_coefficient.dpt_kendall import load_and_match_apoptosis

    if not args.cnn_cache and not args.sae_cache:
        raise ValueError("At least one of --cnn_cache or --sae_cache must be provided")

    # ── Output directory ──
    if args.output_dir:
        out_dir = args.output_dir
    else:
        ref_cache = args.cnn_cache or args.sae_cache
        out_dir = os.path.join(os.path.dirname(ref_cache), "local_linearity_knn")
    os.makedirs(out_dir, exist_ok=True)

    # ── Load feature caches ──
    sources = {}  # label → (X, lines, uids)
    if args.cnn_cache:
        logger.info("Loading CNN cache...")
        X_cnn, lines_cnn, uids_cnn, _ = load_cache(
            args.cnn_cache, args.dead_threshold, apply_l2_norm=args.gap_l2_norm)
        sources["CNN"] = (X_cnn, lines_cnn, uids_cnn)
        logger.info(f"  CNN features: {X_cnn.shape}")

    if args.sae_cache:
        logger.info("Loading SAE cache...")
        # SAE는 L2 norm 적용하지 않음 (이미 파이프라인에서 처리됨)
        X_sae, lines_sae, uids_sae, _ = load_cache(
            args.sae_cache, args.dead_threshold, apply_l2_norm=False)
        sources["SAE"] = (X_sae, lines_sae, uids_sae)
        logger.info(f"  SAE features: {X_sae.shape}")

    # ── Mutations to analyze ──
    mutations = ["SNCA", "GBA", "LRRK2"]

    # ── Main analysis ──
    all_results = {}  # (source, mutation, k) → result dict
    k_sweep_data = {}  # mutation → {source → [(k, mean_ratio, pval), ...]}

    for mut in mutations:
        logger.info(f"\n{'='*60}")
        logger.info(f"  Mutation: {mut}")
        logger.info(f"{'='*60}")

        k_sweep_data[mut] = {}

        for source_label, (X, lines, uids) in sources.items():
            logger.info(f"\n  ── {source_label} ──")

            # Map to superclasses and filter mutation
            superclasses = np.array([SUPERCLASS_MAP.get(ln, ln) for ln in lines])
            mut_mask = superclasses == mut

            # Load apoptosis and filter
            apoptosis = load_and_match_apoptosis(args.apoptosis_csv, uids)
            valid_mask = mut_mask & np.isfinite(apoptosis)
            n_valid = int(valid_mask.sum())

            if n_valid < 10:
                logger.warning(f"    Too few valid samples for {mut} ({n_valid}), skipping")
                continue

            # ── DE / CV neuron filtering (Control vs Mutation) ──
            X_use = X
            has_filter = any(fm != "none" for fm in args.filter_mode)
            if has_filter:
                from apoptosis_prediction.apoptosis_r2_test import select_features_global
                ctrl_mask = superclasses == "Control"
                keep_for_de = ctrl_mask | mut_mask
                sc_sub = list(superclasses[keep_for_de])
                feat_mask = select_features_global(
                    X[keep_for_de], sc_sub, mut,
                    filter_mode=args.filter_mode,
                    min_cv=args.min_cv,
                    de_adj_p=args.de_adj_p,
                    de_min_log2fc=args.de_min_log2fc,
                    de_top_k=args.de_top_k,
                )
                n_features = int(feat_mask.sum())
                if n_features < 2:
                    logger.warning(f"    Filter returned {n_features} features, using all")
                    feat_mask = np.ones(X.shape[1], dtype=bool)
                logger.info(f"    Filtered: {int(feat_mask.sum())}/{X.shape[1]} features")
                X_use = X[:, feat_mask]

            # ── Normalization (after DE filtering) ──
            if args.norm:
                from kendall_correlation_coefficient.dpt_kendall import apply_normalization
                X_use = apply_normalization(X_use, args.norm)
                logger.info(f"    Applied normalization: '{args.norm}'")

            X_mut = X_use[valid_mask]
            apop_mut = apoptosis[valid_mask]
            global_std = np.std(apop_mut)
            logger.info(f"    n={n_valid}, features={X_mut.shape[1]}, global_std={global_std:.6f}")

            sweep_for_source = []

            for k in args.k_neighbors:
                logger.info(f"\n    k={k}")

                # Compute local std ratios
                local_stds, g_std, ratios = compute_local_std_ratios(X_mut, apop_mut, k)
                mean_ratio = np.nanmean(ratios)
                median_ratio = np.nanmedian(ratios)

                logger.info(f"      mean(local/global) = {mean_ratio:.4f}")
                logger.info(f"      median(local/global) = {median_ratio:.4f}")

                # One-sample Wilcoxon signed-rank test: ratio < 1.0?
                valid_ratios = ratios[~np.isnan(ratios)]
                if len(valid_ratios) > 10:
                    try:
                        wilcox_stat, wilcox_p = stats.wilcoxon(
                            valid_ratios - 1.0, alternative="less")
                        logger.info(f"      Wilcoxon (ratio < 1): stat={wilcox_stat:.2f}, "
                                    f"p={wilcox_p:.2e}")
                    except Exception:
                        wilcox_p = np.nan
                else:
                    wilcox_p = np.nan

                # Permutation test
                perm_pval = None
                null_ratios = np.array([])
                if args.n_permutations > 0:
                    logger.info(f"      Permutation test ({args.n_permutations} perms)...")
                    real_mr, null_ratios, perm_pval = permutation_test_ratio(
                        X_mut, apop_mut, k, args.n_permutations, args.seed)
                    logger.info(f"      Perm p={perm_pval:.4f} "
                                f"(null mean={np.mean(null_ratios):.4f}, "
                                f"real={real_mr:.4f})")

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
                }
                all_results[(source_label, mut, k)] = result
                sweep_for_source.append((k, mean_ratio, perm_pval))

                # Plot permutation null for each (source, mutation, k)
                if args.n_permutations > 0 and len(null_ratios) > 0:
                    plot_permutation_null(
                        mean_ratio, null_ratios, source_label, mut, k, perm_pval,
                        os.path.join(out_dir,
                                     f"perm_null_{source_label}_{mut}_k{k}.png"),
                        dpi=args.dpi,
                    )

            k_sweep_data[mut][source_label] = sweep_for_source

        # ── Per-mutation comparison plots (CNN vs SAE) ──
        for k in args.k_neighbors:
            results_by_source = {}
            for source_label in sources.keys():
                key = (source_label, mut, k)
                if key in all_results:
                    results_by_source[source_label] = all_results[key]

            if len(results_by_source) > 0:
                plot_ratio_comparison(
                    results_by_source, mut, k,
                    os.path.join(out_dir, f"ratio_comparison_{mut}_k{k}.png"),
                    dpi=args.dpi,
                )

        # ── K sweep plot per mutation ──
        if k_sweep_data.get(mut):
            plot_k_sweep(
                k_sweep_data[mut], mut,
                os.path.join(out_dir, f"k_sweep_{mut}.png"),
                dpi=args.dpi,
            )

    # ── Summary table ──
    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY — Local Linearity Verification")
    logger.info(f"{'='*80}")
    logger.info(f"  {'Source':6s} {'Mutation':8s} {'k':>4s} {'n':>6s} "
                f"{'MeanRatio':>10s} {'MedRatio':>10s} {'WilcoxP':>10s} {'PermP':>10s}")
    logger.info("  " + "-" * 72)

    for (source, mut, k), res in sorted(all_results.items()):
        wilcox_str = f"{res['wilcoxon_p']:.2e}" if not np.isnan(res.get('wilcoxon_p', np.nan)) else "N/A"
        perm_str = f"{res['perm_pval']:.4f}" if res.get('perm_pval') is not None else "N/A"
        logger.info(f"  {source:6s} {mut:8s} {k:4d} {res['n']:6d} "
                    f"{res['mean_ratio']:10.4f} {res['median_ratio']:10.4f} "
                    f"{wilcox_str:>10s} {perm_str:>10s}")

    # ── Save JSON ──
    json_results = []
    for (source, mut, k), res in sorted(all_results.items()):
        json_results.append({
            "source": source,
            "mutation": mut,
            "k": k,
            "n": res["n"],
            "mean_ratio": float(res["mean_ratio"]),
            "median_ratio": float(res["median_ratio"]),
            "global_std": float(res["global_std"]),
            "wilcoxon_p": float(res["wilcoxon_p"]) if not np.isnan(res.get("wilcoxon_p", np.nan)) else None,
            "perm_pval": float(res["perm_pval"]) if res.get("perm_pval") is not None else None,
        })

    json_path = os.path.join(out_dir, "local_linearity_results.json")
    with open(json_path, "w") as f:
        json.dump({
            "k_neighbors": args.k_neighbors,
            "n_permutations": args.n_permutations,
            "gap_l2_norm": args.gap_l2_norm,
            "norm": args.norm,
            "filter_mode": args.filter_mode,
            "de_adj_p": args.de_adj_p,
            "de_min_log2fc": args.de_min_log2fc,
            "de_top_k": args.de_top_k,
            "min_cv": args.min_cv,
            "results": json_results,
        }, f, indent=2)
    logger.info(f"\n  Saved JSON: {json_path}")
    logger.info(f"  Output: {out_dir}")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()
