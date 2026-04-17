# ==============================================================================
# KNN Distance Uniformity — DPT 동일 파이프라인에서 밀도 균일성 평가
#
# Feature space에서 KNN 이웃들까지의 유클리드 거리가 얼마나 균일한지 평가.
# 밀도가 들쭉날쭉하면 graph-based manifold learning (DPT, diffusion map)에서
# 지름길(shortcut)이 생기거나 transition probability가 왜곡될 수 있다.
#
# Per-sample metric: CV(distances) = std(d_1..d_k) / mean(d_1..d_k)
#   - 낮을수록 이웃 거리가 균일 → 밀도 일정 → manifold learning에 유리
#   - 높을수록 일부 이웃이 매우 멀거나 가까움 → 밀도 불균일
#
# DPT와 정확히 동일한 전처리 파이프라인 적용:
#   L2 norm → CV filter → DE/eval split → DE union+CtrlHigh → norm → PCA → KNN
#
# Usage (Colab):
# !python -m apoptosis_prediction.knn_std_ratio \
#     --cnn_cache "..." --sae_cache "..." \
#     --k_neighbors 5 10 15 20 35 \
#     --gap_l2_norm --dead_threshold 5e-5 \
#     --filter_mode cv de --min_cv 0.1 --de_min_log2fc 1.0 \
#     --norm log_std --de_mode union --de_eval_split 0.5 \
#     --pca_dim 15 --samples_per_class 5000 --seed 856
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
from sklearn.decomposition import PCA

from sae_project.step02_logging_utils import get_logger, SUPERCLASS_MAP
from kendall_correlation_coefficient.dpt_kendall import (
    load_features_cache,
    apply_normalization,
    compute_cv_per_neuron,
    compute_de_neurons,
)

logger = get_logger("knn_std_ratio")

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
sns.set_style("ticks")


# ==============================================================================
# Argument Parser
# ==============================================================================
def get_args():
    p = argparse.ArgumentParser(
        description="KNN distance uniformity — DPT 동일 파이프라인에서 밀도 균일성 평가"
    )
    p.add_argument("--cnn_cache", type=str, default="")
    p.add_argument("--sae_cache", type=str, default="")
    p.add_argument("--dead_threshold", type=float, default=1e-5)
    p.add_argument("--gap_l2_norm", action="store_true")
    p.add_argument("--pre_l2_norm", action="store_true")
    p.add_argument("--divide_hw", type=int, default=0)

    # Neuron filtering (identical to dpt_kendall.py)
    p.add_argument("--filter_mode", type=str, nargs="+", default=["none"])
    p.add_argument("--min_cv", type=float, default=0.0)
    p.add_argument("--de_adj_p", type=float, default=0.05)
    p.add_argument("--de_min_log2fc", type=float, default=1.0)
    p.add_argument("--de_top_k", type=int, default=0)
    p.add_argument("--de_mode", type=str, default="union",
                   choices=["union", "per_mut"])
    p.add_argument("--de_eval_split", type=float, default=0.5)

    p.add_argument("--norm", type=str, default="")
    p.add_argument("--pca_dim", type=int, default=0)

    p.add_argument("--k_neighbors", type=int, nargs="+", default=[10, 20, 50])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--samples_per_class", type=int, default=0,
                   help="Max samples per class (0 = ALL). 5000 = match dpt_kendall.py.")

    return p.parse_args()


# ==============================================================================
# Core: compute KNN distance statistics
# ==============================================================================
def compute_knn_distance_stats(X, k):
    """
    For each sample, find K nearest neighbors and compute distance statistics.

    Returns:
        distances: (N, k) — Euclidean distances to each neighbor
        per_sample_mean: (N,) — mean distance to KNN neighbors
        per_sample_std: (N,) — std of distances to KNN neighbors
        per_sample_cv: (N,) — CV = std/mean of distances (density uniformity)
    """
    n = len(X)
    k_actual = min(k, n - 1)
    if k_actual < 2:
        return (np.full((n, 1), np.nan), np.full(n, np.nan),
                np.full(n, np.nan), np.full(n, np.nan))

    nn = NearestNeighbors(n_neighbors=k_actual + 1, metric="euclidean", n_jobs=-1)
    nn.fit(X)
    dists, _ = nn.kneighbors(X)
    # dists[:, 0] ≈ 0 (self) → use dists[:, 1:]
    neighbor_dists = dists[:, 1:]  # (N, k_actual)

    per_sample_mean = np.mean(neighbor_dists, axis=1)   # (N,)
    per_sample_std = np.std(neighbor_dists, axis=1)      # (N,)
    # CV = std / mean — 낮을수록 거리 균일
    per_sample_cv = per_sample_std / np.where(
        per_sample_mean < 1e-12, 1e-12, per_sample_mean)

    return neighbor_dists, per_sample_mean, per_sample_std, per_sample_cv


# ==============================================================================
# Effective rank via SVD (Shannon entropy of normalized singular values)
# ==============================================================================
def compute_effective_rank(X):
    """
    Effective rank = exp(H(p)) where p_i = σ_i / Σσ_j
    and H(p) = -Σ p_i log(p_i) is the Shannon entropy.

    Measures intrinsic dimensionality: higher = information spread across
    more dimensions; lower = concentrated in few dimensions.
    """
    # Center the data
    X_centered = X - X.mean(axis=0)
    # SVD (economy)
    s = np.linalg.svd(X_centered, compute_uv=False)
    # Remove near-zero singular values
    s = s[s > 1e-12]
    if len(s) == 0:
        return 0.0
    # Normalize to probability distribution
    p = s / s.sum()
    # Shannon entropy
    entropy = -np.sum(p * np.log(p))
    # Effective rank
    return float(np.exp(entropy))


# ==============================================================================
# Plot: Violin/Box — CNN vs SAE distance CV distributions
# ==============================================================================
def plot_cv_comparison(results_by_source, mutation, k, output_path, dpi=200):
    """Violin + box comparing distance CV for CNN vs SAE."""
    fig, ax = plt.subplots(figsize=(5, 5))

    plot_data = []
    labels = []
    for source_label, res in results_by_source.items():
        cv_vals = res["per_sample_cv"]
        valid = cv_vals[~np.isnan(cv_vals)]
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

    for i, (lbl, res) in enumerate(results_by_source.items()):
        mean_cv = res["agg_mean_cv"]
        med_cv = res["agg_median_cv"]
        ax.text(i, ax.get_ylim()[1] * 0.95,
                f"mean={mean_cv:.4f}\nmed={med_cv:.4f}",
                ha="center", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Distance CV (std/mean)", fontsize=11)
    ax.set_title(f"{mutation} — KNN Distance Uniformity (k={k})",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(output_path.replace(".png", ".svg"), bbox_inches="tight")
    if _IN_COLAB:
        plt.show()
    plt.close(fig)


# ==============================================================================
# Plot: Violin/Box — CNN vs SAE mean distance distributions
# ==============================================================================
def plot_mean_dist_comparison(results_by_source, mutation, k, output_path, dpi=200):
    """Violin + box comparing mean KNN distance for CNN vs SAE."""
    fig, ax = plt.subplots(figsize=(5, 5))

    plot_data = []
    labels = []
    for source_label, res in results_by_source.items():
        m = res["per_sample_mean"]
        plot_data.append(m[~np.isnan(m)])
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

    for i, (lbl, res) in enumerate(results_by_source.items()):
        gm = np.nanmean(res["per_sample_mean"])
        gmed = np.nanmedian(res["per_sample_mean"])
        ax.text(i, ax.get_ylim()[1] * 0.95,
                f"mean={gm:.4f}\nmed={gmed:.4f}",
                ha="center", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Mean KNN Distance", fontsize=11)
    ax.set_title(f"{mutation} — Mean KNN Distance (k={k})",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(output_path.replace(".png", ".svg"), bbox_inches="tight")
    if _IN_COLAB:
        plt.show()
    plt.close(fig)


# ==============================================================================
# Plot: K sweep — distance CV vs k
# ==============================================================================
def plot_k_sweep(sweep_results, mutation, output_path, dpi=200):
    """Line plot: mean/median distance CV vs k."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"CNN": "#4C72B0", "SAE": "#DD8452"}

    for source_label, kv_list in sweep_results.items():
        ks = [r[0] for r in kv_list]
        mean_cvs = [r[1] for r in kv_list]
        med_cvs = [r[2] for r in kv_list]
        c = colors.get(source_label, "gray")
        ax.plot(ks, mean_cvs, "o-", color=c, linewidth=2, markersize=6,
                label=f"{source_label} (mean CV)", alpha=0.9)
        ax.plot(ks, med_cvs, "s--", color=c, linewidth=1.5, markersize=5,
                label=f"{source_label} (median CV)", alpha=0.6)

    ax.set_xlabel("k (number of neighbors)", fontsize=11)
    ax.set_ylabel("Distance CV (std/mean)", fontsize=11)
    ax.set_title(f"{mutation} — KNN Distance CV vs k", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(output_path.replace(".png", ".svg"), bbox_inches="tight")
    if _IN_COLAB:
        plt.show()
    plt.close(fig)


# ==============================================================================
# Plot: K sweep — mean distance std vs k
# ==============================================================================
def plot_k_sweep_std(sweep_results, mutation, output_path, dpi=200):
    """Line plot: mean/median of per-sample distance std vs k."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"CNN": "#4C72B0", "SAE": "#DD8452"}

    for source_label, kv_list in sweep_results.items():
        ks = [r[0] for r in kv_list]
        mean_stds = [r[3] for r in kv_list]
        med_stds = [r[4] for r in kv_list]
        c = colors.get(source_label, "gray")
        ax.plot(ks, mean_stds, "o-", color=c, linewidth=2, markersize=6,
                label=f"{source_label} (mean std)", alpha=0.9)
        ax.plot(ks, med_stds, "s--", color=c, linewidth=1.5, markersize=5,
                label=f"{source_label} (median std)", alpha=0.6)

    ax.set_xlabel("k (number of neighbors)", fontsize=11)
    ax.set_ylabel("KNN Distance Std", fontsize=11)
    ax.set_title(f"{mutation} — KNN Distance Std vs k",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(output_path.replace(".png", ".svg"), bbox_inches="tight")
    if _IN_COLAB:
        plt.show()
    plt.close(fig)


# ==============================================================================
# Main — dpt_kendall.py 동일 파이프라인
# ==============================================================================
def main():
    args = get_args()
    np.random.seed(args.seed)

    if not args.cnn_cache and not args.sae_cache:
        raise ValueError("At least one of --cnn_cache or --sae_cache required")

    # Output
    if args.output_dir:
        out_dir = args.output_dir
    else:
        ref = args.cnn_cache or args.sae_cache
        out_dir = os.path.join(os.path.dirname(ref), "knn_dist_uniformity")
    os.makedirs(out_dir, exist_ok=True)

    # ── Load feature caches ──
    sources = {}

    def _load_and_preprocess(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        if "X_gap" in data:
            X = data["X_gap"]
            lines = data["lines"].astype(str) if data["lines"].dtype.kind != 'U' else data["lines"]
            uids = data["uids"].astype(str) if data["uids"].dtype.kind != 'U' else data["uids"]
            label = "CNN"
            logger.info(f"  Detected CNN GAP cache: {X.shape}")
        elif "X_all" in data:
            X, _, lines, uids, _, _ = load_features_cache(cache_path, args.dead_threshold)
            label = "SAE"
        else:
            raise ValueError(f"Unknown cache format. Keys: {list(data.keys())}")

        if args.pre_l2_norm:
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1e-12, norms)
            X = X / norms
            logger.info(f"  Applied pre-L2 normalization")

        if args.divide_hw > 0:
            X = X / args.divide_hw
            logger.info(f"  Divided by H*W={args.divide_hw}")

        if args.gap_l2_norm:
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1e-12, norms)
            X = X / norms
            logger.info(f"  Applied L2 normalization")

        superclasses = [SUPERCLASS_MAP.get(ln, ln) for ln in lines]
        return X, superclasses, uids, label

    if args.cnn_cache:
        logger.info("Loading CNN cache...")
        X_cnn, sc_cnn, uids_cnn, _ = _load_and_preprocess(args.cnn_cache)
        sources["CNN"] = (X_cnn, sc_cnn, uids_cnn)
        logger.info(f"  CNN features: {X_cnn.shape}")

    if args.sae_cache:
        logger.info("Loading SAE cache...")
        X_sae, sc_sae, uids_sae, _ = _load_and_preprocess(args.sae_cache)
        sources["SAE"] = (X_sae, sc_sae, uids_sae)
        logger.info(f"  SAE features: {X_sae.shape}")

    mutations = ["SNCA", "GBA", "LRRK2"]
    all_results = {}   # (source, mutation, k) → result dict
    k_sweep_data = {}  # mutation → {source → [(k, mean_cv, med_cv, mean_std, med_std), ...]}

    for source_label, (X_raw, superclasses, uids) in sources.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"  Source: {source_label}")
        logger.info(f"{'='*60}")

        superclasses_arr = np.array(superclasses)

        # ── Subsample per class (simple stratified random) ──
        spc = args.samples_per_class
        if spc > 0:
            rng_sub = np.random.RandomState(args.seed)
            keep_indices = []
            for cls in sorted(np.unique(superclasses_arr)):
                cls_idx = np.where(superclasses_arr == cls)[0]
                n_take = min(spc, len(cls_idx))
                chosen = rng_sub.choice(cls_idx, size=n_take, replace=False)
                keep_indices.extend(chosen.tolist())
                logger.info(f"    Subsample {cls}: {len(cls_idx)} → {len(chosen)}")
            keep_indices = sorted(keep_indices)
            X_raw = X_raw[keep_indices]
            superclasses = [superclasses[i] for i in keep_indices]
            superclasses_arr = np.array(superclasses)
            logger.info(f"    After subsampling: {X_raw.shape[0]} samples")

        X = X_raw.copy()

        # ══════════════════════════════════════════════════════════════
        # Global filters (cv) — identical to dpt_kendall.py run_analysis()
        # ══════════════════════════════════════════════════════════════
        has_de = "de" in args.filter_mode
        de_mode = getattr(args, "de_mode", "union")
        filter_steps = []

        for fm in args.filter_mode:
            if fm in ("none", "de"):
                continue
            n_before = X.shape[1]
            if fm == "cv":
                cv = compute_cv_per_neuron(X, superclasses)
                X = X[:, cv >= args.min_cv]
                step = f"cv≥{args.min_cv}: {n_before}→{X.shape[1]}"
            else:
                continue
            filter_steps.append(step)
            logger.info(f"  Filter [{fm}]: {step}")

        # ── DE/Eval split ──
        de_eval_split = getattr(args, 'de_eval_split', 0.0)
        if de_eval_split > 0 and has_de:
            rng_split = np.random.RandomState(args.seed)
            n_total = len(superclasses_arr)
            eval_mask = np.zeros(n_total, dtype=bool)
            for cls in sorted(set(superclasses_arr)):
                cls_idx = np.where(superclasses_arr == cls)[0]
                n_eval = max(1, int(len(cls_idx) * de_eval_split))
                chosen = rng_split.choice(cls_idx, size=n_eval, replace=False)
                eval_mask[chosen] = True
            de_mask_global = ~eval_mask
            logger.info(f"  DE/Eval split: DE={int(de_mask_global.sum())}, "
                        f"Eval={int(eval_mask.sum())}")
        else:
            eval_mask = np.ones(len(superclasses_arr), dtype=bool)
            de_mask_global = np.ones(len(superclasses_arr), dtype=bool)

        # ── DE union mode ──
        if has_de and de_mode == "union":
            de_masks = []
            X_de = X[de_mask_global]
            sc_de = list(superclasses_arr[de_mask_global])
            for m in ["SNCA", "GBA", "LRRK2"]:
                de_result = compute_de_neurons(
                    X_de, sc_de, m,
                    adj_p_threshold=args.de_adj_p,
                    min_log2fc=args.de_min_log2fc,
                )
                mask = de_result["mask"]
                if args.de_top_k > 0 and mask.sum() > args.de_top_k:
                    sig_indices = np.where(mask)[0]
                    abs_fc = np.abs(de_result["log2fc"][sig_indices])
                    top_k_idx = sig_indices[np.argsort(abs_fc)[::-1][:args.de_top_k]]
                    mask = np.zeros_like(mask)
                    mask[top_k_idx] = True
                de_masks.append(mask)

            superclasses_allm = [("AllMut" if s != "Control" else "Control")
                                 for s in sc_de]
            de_ctrl = compute_de_neurons(
                X_de, superclasses_allm, "AllMut",
                adj_p_threshold=args.de_adj_p,
                min_log2fc=args.de_min_log2fc,
            )
            ctrl_high_mask = de_ctrl["mask"] & (de_ctrl["log2fc"] < 0)
            logger.info(f"    DE(Ctrl vs AllMut): Control-high={int(ctrl_high_mask.sum())}")
            de_masks.append(ctrl_high_mask)

            union_mask = de_masks[0] | de_masks[1] | de_masks[2] | de_masks[3]
            n_before_de = X.shape[1]
            X = X[:, union_mask]
            de_step = f"DE_union+CtrlHigh: {n_before_de}→{X.shape[1]}"
            filter_steps.append(de_step)
            logger.info(f"  {de_step}")
        elif has_de and de_mode == "per_mut":
            filter_steps.append("DE_per_mut (deferred)")

        filter_label = " → ".join(filter_steps) if filter_steps else "none"
        logger.info(f"  Filter summary: {filter_label}")
        logger.info(f"  Features after filter: {X.shape}")

        # ── Per-mutation loop ──
        for mut in mutations:
            logger.info(f"\n  ── {source_label} / {mut} ──")

            if mut not in k_sweep_data:
                k_sweep_data[mut] = {}

            # Use all cells of this mutation (no apoptosis needed for distance)
            mut_mask = superclasses_arr == mut
            n_mut = int(mut_mask.sum())

            if n_mut < 10:
                logger.warning(f"    Too few samples ({n_mut}), skipping")
                continue

            # ── Per-mutation DE (if de_mode=per_mut) ──
            X_use = X
            if has_de and de_mode == "per_mut":
                de_result = compute_de_neurons(
                    X[de_mask_global], list(superclasses_arr[de_mask_global]), mut,
                    adj_p_threshold=args.de_adj_p,
                    min_log2fc=args.de_min_log2fc,
                )
                de_mask = de_result["mask"]
                if args.de_top_k > 0 and de_mask.sum() > args.de_top_k:
                    sig_idx = np.where(de_mask)[0]
                    abs_fc = np.abs(de_result["log2fc"][sig_idx])
                    top_k_idx = sig_idx[np.argsort(abs_fc)[::-1][:args.de_top_k]]
                    de_mask = np.zeros_like(de_mask)
                    de_mask[top_k_idx] = True
                if de_mask.sum() < 2:
                    de_mask = np.ones(X.shape[1], dtype=bool)
                X_use = X[:, de_mask]

            # ── Normalization ──
            if args.norm:
                X_use = apply_normalization(X_use, args.norm)
                logger.info(f"    Applied normalization: '{args.norm}'")

            # ── PCA ──
            if args.pca_dim > 0 and X_use.shape[1] > args.pca_dim:
                n_pca = min(args.pca_dim, X_use.shape[1], X_use.shape[0] - 1)
                pca = PCA(n_components=n_pca, random_state=args.seed)
                X_use = pca.fit_transform(X_use)
                var_exp = pca.explained_variance_ratio_.sum()
                logger.info(f"    PCA: {n_pca}D (explained var: {var_exp:.1%})")

            X_mut = X_use[mut_mask]
            logger.info(f"    n={n_mut}, features={X_mut.shape[1]}")

            # ── Effective rank (SVD) ──
            eff_rank = compute_effective_rank(X_mut)
            logger.info(f"    Effective rank: {eff_rank:.2f} / {X_mut.shape[1]}")

            sweep_for_source = []

            for k in args.k_neighbors:
                logger.info(f"\n    k={k}")

                _, per_mean, per_std, per_cv = compute_knn_distance_stats(X_mut, k)

                agg_mean_cv = float(np.nanmean(per_cv))
                agg_median_cv = float(np.nanmedian(per_cv))
                agg_mean_std = float(np.nanmean(per_std))
                agg_median_std = float(np.nanmedian(per_std))
                agg_mean_dist = float(np.nanmean(per_mean))
                agg_median_dist = float(np.nanmedian(per_mean))

                logger.info(f"      Distance: mean={agg_mean_dist:.6f}, "
                            f"median={agg_median_dist:.6f}")
                logger.info(f"      Dist Std: mean={agg_mean_std:.6f}, "
                            f"median={agg_median_std:.6f}")
                logger.info(f"      Dist CV:  mean={agg_mean_cv:.4f}, "
                            f"median={agg_median_cv:.4f}")

                result = {
                    "per_sample_mean": per_mean,
                    "per_sample_std": per_std,
                    "per_sample_cv": per_cv,
                    "agg_mean_cv": agg_mean_cv,
                    "agg_median_cv": agg_median_cv,
                    "agg_mean_std": agg_mean_std,
                    "agg_median_std": agg_median_std,
                    "agg_mean_dist": agg_mean_dist,
                    "agg_median_dist": agg_median_dist,
                    "effective_rank": eff_rank,
                    "n": n_mut,
                    "k": k,
                }
                all_results[(source_label, mut, k)] = result
                sweep_for_source.append((
                    k, agg_mean_cv, agg_median_cv, agg_mean_std, agg_median_std))

            k_sweep_data[mut][source_label] = sweep_for_source

    # ── Per-mutation comparison plots ──
    for mut in mutations:
        for k in args.k_neighbors:
            results_by_source = {}
            for source_label in sources.keys():
                key = (source_label, mut, k)
                if key in all_results:
                    results_by_source[source_label] = all_results[key]

            if len(results_by_source) > 0:
                plot_cv_comparison(
                    results_by_source, mut, k,
                    os.path.join(out_dir, f"dist_cv_{mut}_k{k}.png"),
                    dpi=args.dpi)
                plot_mean_dist_comparison(
                    results_by_source, mut, k,
                    os.path.join(out_dir, f"mean_dist_{mut}_k{k}.png"),
                    dpi=args.dpi)

        if k_sweep_data.get(mut):
            plot_k_sweep(
                k_sweep_data[mut], mut,
                os.path.join(out_dir, f"k_sweep_cv_{mut}.png"),
                dpi=args.dpi)
            plot_k_sweep_std(
                k_sweep_data[mut], mut,
                os.path.join(out_dir, f"k_sweep_std_{mut}.png"),
                dpi=args.dpi)

    # ── Summary ──
    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY — KNN Distance Uniformity")
    logger.info(f"{'='*80}")
    logger.info(f"  Filter: {filter_label}")
    logger.info(f"  Norm: {args.norm or 'none'}, PCA: {args.pca_dim}")
    logger.info(f"  {'Source':6s} {'Mutation':8s} {'k':>4s} {'n':>6s} "
                f"{'MeanDist':>10s} {'MeanStd':>10s} {'MeanCV':>10s} {'MedCV':>10s} "
                f"{'EffRank':>8s}")
    logger.info("  " + "-" * 80)

    for (source, mut, k), res in sorted(all_results.items()):
        logger.info(f"  {source:6s} {mut:8s} {k:4d} {res['n']:6d} "
                    f"{res['agg_mean_dist']:10.6f} "
                    f"{res['agg_mean_std']:10.6f} "
                    f"{res['agg_mean_cv']:10.4f} "
                    f"{res['agg_median_cv']:10.4f} "
                    f"{res['effective_rank']:8.2f}")

    # ── Save JSON ──
    json_results = []
    for (source, mut, k), res in sorted(all_results.items()):
        json_results.append({
            "source": source, "mutation": mut, "k": k, "n": res["n"],
            "agg_mean_dist": res["agg_mean_dist"],
            "agg_median_dist": res["agg_median_dist"],
            "agg_mean_std": res["agg_mean_std"],
            "agg_median_std": res["agg_median_std"],
            "agg_mean_cv": res["agg_mean_cv"],
            "agg_median_cv": res["agg_median_cv"],
            "effective_rank": res["effective_rank"],
        })

    json_path = os.path.join(out_dir, "knn_dist_uniformity_results.json")
    with open(json_path, "w") as f:
        json.dump({
            "k_neighbors": args.k_neighbors,
            "gap_l2_norm": args.gap_l2_norm,
            "norm": args.norm,
            "filter_mode": args.filter_mode,
            "de_min_log2fc": args.de_min_log2fc,
            "de_mode": args.de_mode,
            "de_eval_split": args.de_eval_split,
            "min_cv": args.min_cv,
            "pca_dim": args.pca_dim,
            "samples_per_class": args.samples_per_class,
            "seed": args.seed,
            "results": json_results,
        }, f, indent=2)
    logger.info(f"\n  Saved JSON: {json_path}")
    logger.info(f"  Output: {out_dir}")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()
