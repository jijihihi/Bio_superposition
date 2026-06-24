# ==============================================================================
# Local Ridge vs Global Ridge — Residual Analysis
#
# "Locally linear, globally curved" 가설을 직접 검증:
#   - Global Ridge: 전체 mutation 데이터에 Ridge fit → R²
#   - Local Ridge: 각 샘플의 KNN 이웃에서 Ridge fit → per-sample local R²
#   - local R² >> global R² 이면 국소적으로는 선형이지만 전역적으로 비선형
#
# CNN feature vector와 SAE feature vector 각각에 대해 수행.
# dpt_kendall.py 와 정확히 동일한 필터링 파이프라인 적용:
#   L2 norm → CV filter → DE/eval split → DE union+CtrlHigh → norm → PCA → KNN Ridge
#
# Usage (Colab):
# !python -m apoptosis_prediction.local_vs_global_ridge \
#     --cnn_cache "..." \
#     --sae_cache "..." \
#     --apoptosis_csv "..." \
#     --k_neighbors 50 \
#     --gap_l2_norm \
#     --dead_threshold 5e-5 \
#     --filter_mode cv de \
#     --min_cv 0.1 \
#     --de_min_log2fc 1.0 \
#     --norm log_std \
#     --de_mode union \
#     --de_eval_split 0.5 \
#     --pca_dim 15 \
#     --samples_per_class 5000 \
#     --seed 856 \
#     --output_dir "/content/local_vs_global_ridge"
# ==============================================================================

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
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from kendall_correlation_coefficient.dpt_kendall import (
    apply_normalization, compute_cv_per_neuron, compute_de_neurons,
    load_and_match_apoptosis, load_features_cache)
from sae_project.step02_logging_utils import SUPERCLASS_MAP, get_logger

logger = get_logger("local_vs_global_ridge")

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
sns.set_style("ticks")

ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]


# ==============================================================================
# Argument Parser — mirrors dpt_kendall.py + local_knn_std.py
# ==============================================================================
def get_args():
    p = argparse.ArgumentParser(
        description="Local Ridge vs Global Ridge — locally linear, globally curved 검증"
    )
    p.add_argument("--cnn_cache", type=str, default="")
    p.add_argument("--sae_cache", type=str, default="")
    p.add_argument("--apoptosis_csv", type=str, required=True)
    p.add_argument("--dead_threshold", type=float, default=1e-5)
    p.add_argument("--gap_l2_norm", action="store_true",
                   help="Apply L2 normalization to feature vectors (useful for GAP)")
    p.add_argument("--pre_l2_norm", action="store_true",
                   help="Apply per-image L2 normalization BEFORE any other processing.")
    p.add_argument("--divide_hw", type=int, default=0,
                   help="Divide features by H*W to convert sum→mean (e.g. 256 for 16x16).")

    # Neuron filtering (identical to dpt_kendall.py)
    p.add_argument("--filter_mode", type=str, nargs="+", default=["none"],
                   help="Sequential: 'cv', 'de', 'none'. e.g. '--filter_mode cv de'")
    p.add_argument("--min_cv", type=float, default=0.0)
    p.add_argument("--de_adj_p", type=float, default=0.05)
    p.add_argument("--de_min_log2fc", type=float, default=1.0)
    p.add_argument("--de_top_k", type=int, default=0,
                   help="Max DE neurons per mutation (by |log2FC| rank). 0 = keep all.")
    p.add_argument("--de_mode", type=str, default="union",
                   choices=["union", "per_mut"])
    p.add_argument("--de_eval_split", type=float, default=0.5,
                   help="Fraction of data held out for evaluation (0 = no split).")

    # Normalization (after DE filter, before PCA — identical to dpt_kendall.py)
    p.add_argument("--norm", type=str, default="",
                   help="Feature normalization: '', 'log', 'std', 'log_std'. "
                        "Applied AFTER DE filtering.")

    # PCA (after norm, before KNN — identical to dpt_kendall.py)
    p.add_argument("--pca_dim", type=int, default=10,
                   help="PCA dimensions after norm, before KNN Ridge. "
                        "0 = no PCA. Set to 15 to match DPT pipeline.")

    p.add_argument("--k_neighbors", type=int, nargs="+", default=[35],
                   help="K values for local Ridge neighborhood (supports multiple for sweep).")
    p.add_argument("--min_local_n", type=int, default=20,
                   help="Minimum samples in neighborhood for local Ridge (default: 20)")
    p.add_argument("--n_permutations", type=int, default=0,
                   help="Permutation null for global R² (default: 0 = skip)")
    p.add_argument("--cv_folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--samples_per_class", type=int, default=0,
                   help="Max samples per class (0 = use ALL). "
                        "Prioritizes samples with valid apoptosis. "
                        "Set to 5000 to match dpt_kendall.py default.")

    return p.parse_args()


# ==============================================================================
# Global Ridge R² (LOO-style per-sample prediction from CV)
# ==============================================================================
def compute_global_ridge_r2(X, y, cv_folds, seed):
    """
    Global Ridge R² via KFold CV. Returns overall R² and per-fold R²s.
    """
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    y_pred = np.full_like(y, np.nan)
    fold_r2s = []

    for train_idx, test_idx in kf.split(X):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=ALPHAS)),
        ])
        pipe.fit(X[train_idx], y[train_idx])
        pred = pipe.predict(X[test_idx])
        y_pred[test_idx] = pred

        ss_res = np.sum((y[test_idx] - pred) ** 2)
        ss_tot = np.sum((y[test_idx] - y[test_idx].mean()) ** 2)
        fold_r2s.append(1.0 - ss_res / max(ss_tot, 1e-12))

    # Overall R²
    ss_res_all = np.sum((y - y_pred) ** 2)
    ss_tot_all = np.sum((y - y.mean()) ** 2)
    overall_r2 = 1.0 - ss_res_all / max(ss_tot_all, 1e-12)

    return overall_r2, np.array(fold_r2s), y_pred


# ==============================================================================
# Local Ridge R² — LOO: train on K neighbors, predict held-out sample
# ==============================================================================
def compute_local_ridge_r2(X, y, k, min_local_n=20, pca_dim=0):
    """
    For each sample i:
      1. Find K nearest neighbors (excluding self)
      2. Fit Ridge on those K neighbors
      3. Predict sample i (held-out) → local_preds[i]

    Aggregate test R² = R²(y, local_preds) across all samples.
    This is a proper held-out metric: sample i is NEVER in its own training set.

    NOTE: When used with the full DPT pipeline, PCA is applied BEFORE this function
    is called. The internal pca_dim parameter is for standalone use without the
    pipeline. Set pca_dim=0 when pipeline PCA is already applied.

    Returns:
        local_preds: (N,) predicted value from each sample's local Ridge (held-out)
        aggregate_r2: float — R²(y, local_preds) over all valid samples
    """
    n, d = X.shape
    k_actual = min(k, n - 1)

    if k_actual < min_local_n:
        logger.warning(f"k={k_actual} < min_local_n={min_local_n}, "
                       f"clamping min_local_n to {k_actual}")
        min_local_n = k_actual

    # Optional internal PCA (only if pca_dim > 0 and not already done externally)
    if pca_dim > 0 and d > pca_dim:
        logger.info(f"      Internal PCA: {d} → {pca_dim} dims")
        pca = PCA(n_components=pca_dim, random_state=42)
        X_use = pca.fit_transform(X)
        explained = pca.explained_variance_ratio_.sum()
        logger.info(f"      PCA explained variance: {explained:.4f}")
    else:
        X_use = X

    # KNN in the (possibly PCA-reduced) space
    nn = NearestNeighbors(n_neighbors=k_actual + 1, metric="euclidean", n_jobs=-1)
    nn.fit(X_use)
    _, indices = nn.kneighbors(X_use)
    neighbor_indices = indices[:, 1:]  # (N, k_actual)

    local_preds = np.full(n, np.nan)

    for i in range(n):
        nbr_idx = neighbor_indices[i]
        X_nbr = X_use[nbr_idx]
        y_nbr = y[nbr_idx]

        if len(nbr_idx) < min_local_n
            continue

        try:
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", RidgeCV(alphas=ALPHAS)),
            ])
            pipe.fit(X_nbr, y_nbr)
            pred_i = pipe.predict(X_use[i:i+1])[0]
            local_preds[i] = pred_i
        except Exception:
            continue

    # Aggregate test R²
    valid = ~np.isnan(local_preds)
    n_valid = int(valid.sum())
    if n_valid < 10:
        return local_preds, 0.0

    ss_res = np.sum((y[valid] - local_preds[valid]) ** 2)
    ss_tot = np.sum((y[valid] - y[valid].mean()) ** 2)
    aggregate_r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

    return local_preds, aggregate_r2


# ==============================================================================
# Plot: Local R² distribution — CNN vs SAE
# ==============================================================================
def plot_local_r2_comparison(results_by_source, mutation, k, global_r2s,
                             output_path, dpi=200):
    """Bar chart: Global Ridge R² vs Local LOO Ridge R² for CNN and SAE."""
    fig, ax = plt.subplots(figsize=(6, 5))

    colors = {"CNN": "#4C72B0", "SAE": "#DD8452"}
    labels = list(results_by_source.keys())
    x = np.arange(len(labels))
    width = 0.35

    global_vals = [global_r2s.get(lbl, 0.0) for lbl in labels]
    local_vals = [results_by_source[lbl]["aggregate_r2"] for lbl in labels]

    bars1 = ax.bar(x - width/2, global_vals, width,
                   color=[colors.get(l, "gray") for l in labels],
                   alpha=0.5, edgecolor="white", label="Global Ridge R²")
    bars2 = ax.bar(x + width/2, local_vals, width,
                   color=[colors.get(l, "gray") for l in labels],
                   alpha=0.9, edgecolor="white", label="Local LOO Ridge R²")

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                f"{h:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.axhline(0, color="gray", linewidth=0.5, linestyle="-")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("R²", fontsize=11)
    ax.set_title(f"{mutation} — Global vs Local LOO Ridge R² (k={k})",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(output_path.replace(".png", ".svg"), bbox_inches="tight")
    if _IN_COLAB:
        plt.show()
    plt.close(fig)


# ==============================================================================
# Plot: Gap (local R² - global R²) comparison
# ==============================================================================
def plot_r2_gap(results_by_source, mutation, k, global_r2s,
                output_path, dpi=200):
    """Bar chart: Local LOO R² - Global R² for each source."""
    fig, ax = plt.subplots(figsize=(5, 4))
    colors = {"CNN": "#4C72B0", "SAE": "#DD8452"}

    labels = list(results_by_source.keys())
    gaps = []

    for lbl in labels:
        local_r2 = results_by_source[lbl]["aggregate_r2"]
        g_r2 = global_r2s.get(lbl, 0.0)
        gaps.append(local_r2 - g_r2)

    bars = ax.bar(labels, gaps, color=[colors.get(l, "gray") for l in labels],
                  alpha=0.8, edgecolor="white")

    ax.axhline(0, color="gray", linewidth=1, linestyle="-", alpha=0.5)
    ax.set_ylabel("Local LOO R² − Global R²", fontsize=11)
    ax.set_title(f"{mutation} — R² Gap: Local LOO − Global (k={k})",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="y")

    for bar, g in zip(bars, gaps):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{g:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(output_path.replace(".png", ".svg"), bbox_inches="tight")
    if _IN_COLAB:
        plt.show()
    plt.close(fig)


# ==============================================================================
# Plot: Scatter — local predicted vs actual
# ==============================================================================
def plot_local_pred_scatter(y_true, local_preds, aggregate_r2, source_label,
                            mutation, k, output_path, dpi=200):
    """Scatter of local LOO Ridge predicted vs actual."""
    valid = ~np.isnan(local_preds)
    if valid.sum() == 0:
        logger.warning(f"    No valid local predictions for {source_label}/{mutation}, skip scatter")
        return
    y_v = y_true[valid]
    pred_v = local_preds[valid]

    colors = {"CNN": "#4C72B0", "SAE": "#DD8452"}
    c = colors.get(source_label, "gray")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_v, pred_v, c=c, s=8, alpha=0.3,
               edgecolors="none", rasterized=True)

    lims = [min(y_v.min(), pred_v.min()), max(y_v.max(), pred_v.max())]
    ax.plot(lims, lims, "--", color="gray", alpha=0.5, linewidth=1)

    if len(y_v) > 2:
        z = np.polyfit(y_v, pred_v, 1)
        x_line = np.linspace(y_v.min(), y_v.max(), 100)
        ax.plot(x_line, np.polyval(z, x_line), "-", color=c, linewidth=2, alpha=0.8)

    from scipy.stats import pearsonr
    r, _ = pearsonr(y_v, pred_v)
    ax.text(0.05, 0.95,
            f"n={len(y_v)}\nLOO R²={aggregate_r2:.4f}\nr={r:.4f}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_xlabel("Actual Apoptosis Rate", fontsize=11)
    ax.set_ylabel("Local LOO Ridge Predicted", fontsize=11)
    ax.set_title(f"{source_label} — {mutation} (k={k})", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
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
        out_dir = os.path.join(os.path.dirname(ref), "local_vs_global_ridge")
    os.makedirs(out_dir, exist_ok=True)

    # ── Load feature caches ──────────────────────────────────────────
    # Matches dpt_kendall.py main(): load raw → pre_l2_norm → divide_hw → gap_l2_norm
    sources = {}  # label → (X, superclasses, uids, apoptosis)

    def _load_and_preprocess(cache_path):
        """Load cache and apply pre-processing exactly like dpt_kendall.py main()."""
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
        apoptosis = load_and_match_apoptosis(args.apoptosis_csv, uids)
        return X, superclasses, uids, apoptosis, label

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

    mutations = ["SNCA", "GBA", "LRRK2"]
    all_results = {}

    # ── Per-source pipeline (identical to dpt_kendall.py run_analysis) ──
    for source_label, (X_raw, superclasses, uids, apoptosis) in sources.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"  Source: {source_label}")
        logger.info(f"{'='*60}")

        superclasses_arr = np.array(superclasses)

        # ── Subsample per class (prioritize valid apoptosis) ──
        spc = args.samples_per_class
        if spc > 0:
            rng_sub = np.random.RandomState(args.seed)
            keep_indices = []
            for cls in sorted(np.unique(superclasses_arr)):
                cls_idx = np.where(superclasses_arr == cls)[0]
                valid_mask_cls = ~np.isnan(apoptosis[cls_idx])
                valid_idx = cls_idx[valid_mask_cls]
                invalid_idx = cls_idx[~valid_mask_cls]
                ordered = np.concatenate([valid_idx, invalid_idx])
                n_take = min(spc, len(ordered))
                chosen = rng_sub.choice(
                    ordered[:max(n_take, len(valid_idx))],
                    size=min(n_take, len(ordered)), replace=False)
                keep_indices.extend(chosen.tolist())
                logger.info(f"    Subsample {cls}: {len(cls_idx)} → {len(chosen)} "
                            f"(valid apop: {valid_mask_cls.sum()})")
            keep_indices = sorted(keep_indices)
            X_raw = X_raw[keep_indices]
            superclasses = [superclasses[i] for i in keep_indices]
            superclasses_arr = np.array(superclasses)
            apoptosis = apoptosis[keep_indices]
            uids = [uids[i] for i in keep_indices] if isinstance(uids, list) else uids[keep_indices]
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

        # ── DE/Eval split (avoid circular analysis) ────────────────
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
            n_de = int(de_mask_global.sum())
            n_eval_count = int(eval_mask.sum())
            logger.info(f"  DE/Eval split: DE={n_de}, Eval={n_eval_count} "
                        f"(split={de_eval_split:.0%})")
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

            # Control vs AllMut — Control-high direction only (log2fc < 0)
            superclasses_allm = [("AllMut" if s != "Control" else "Control")
                                 for s in sc_de]
            de_ctrl = compute_de_neurons(
                X_de, superclasses_allm, "AllMut",
                adj_p_threshold=args.de_adj_p,
                min_log2fc=args.de_min_log2fc,
            )
            ctrl_high_mask = de_ctrl["mask"] & (de_ctrl["log2fc"] < 0)
            n_ctrl_high = int(ctrl_high_mask.sum())
            n_mut_high = int((de_ctrl["mask"] & (de_ctrl["log2fc"] > 0)).sum())
            logger.info(f"    DE(Ctrl vs AllMut): {int(de_ctrl['mask'].sum())} total → "
                        f"Control-high: {n_ctrl_high}, Mut-high: {n_mut_high} (kept: Ctrl-high only)")
            de_masks.append(ctrl_high_mask)

            union_mask = de_masks[0] | de_masks[1] | de_masks[2] | de_masks[3]
            n_before_de = X.shape[1]
            X = X[:, union_mask]
            top_k_str = f"(top{args.de_top_k}/mut)" if args.de_top_k > 0 else ""
            de_step = f"DE_union+CtrlHigh{top_k_str}: {n_before_de}→{X.shape[1]}"
            filter_steps.append(de_step)
            logger.info(f"  DE union+CtrlHigh: {n_before_de} → {X.shape[1]} neurons")
        elif has_de and de_mode == "per_mut":
            filter_steps.append("DE_per_mut (deferred)")
            logger.info(f"  DE mode: per_mut (applied per mutation inside loop)")

        filter_label = " → ".join(filter_steps) if filter_steps else "none"
        logger.info(f"  Filter summary: {filter_label}")
        logger.info(f"  Features after filter: {X.shape}")

        # ── Per-mutation loop ─────────────────────────────────────────
        for mut in mutations:
            logger.info(f"\n  ── {source_label} / {mut} ──")

            mut_mask = superclasses_arr == mut
            valid_mask = mut_mask & np.isfinite(apoptosis)

            # Apply eval_mask: only evaluate on eval set
            if de_eval_split > 0 and has_de:
                valid_mask = valid_mask & eval_mask

            n_valid = int(valid_mask.sum())

            if n_valid < args.min_local_n * 2:
                logger.warning(f"    Too few valid samples for {mut} ({n_valid}), skipping")
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
                n_de_feat = int(de_mask.sum())
                logger.info(f"    DE({mut}): {n_de_feat}/{X.shape[1]} features")
                if n_de_feat < 2:
                    logger.warning(f"    DE({mut}): too few features, using all")
                    de_mask = np.ones(X.shape[1], dtype=bool)
                X_use = X[:, de_mask]

            # ── Normalization (after DE filtering) ──
            if args.norm:
                X_use = apply_normalization(X_use, args.norm)
                logger.info(f"    Applied normalization: '{args.norm}'")

            # ── PCA (after norm, before KNN — matches dpt_kendall.py) ──
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
                logger.info(f"    PCA: {pca.n_features_in_}D → {n_pca}D "
                            f"(total explained var: {cum_var[n_pca-1]:.1%})")
                for i in range(n_fit):
                    marker = " ◀ cutoff" if i == n_pca - 1 else ""
                    logger.info(f"      PC{i+1:3d}: {var_ratios[i]:.4f}  "
                                f"(cumulative: {cum_var[i]:.4f}){marker}")

                # Scree plot — save + show in Colab
                fig_sc, ax_sc = plt.subplots(figsize=(8, 4))
                x_pos = np.arange(1, n_fit + 1)
                ax_sc.bar(x_pos, var_ratios, color="#5B9BD5", alpha=0.7,
                          edgecolor="white", label="Individual")
                ax_sc.plot(x_pos, cum_var, "o-", color="#ED7D31", linewidth=2,
                           markersize=4, label="Cumulative")
                ax_sc.axvline(n_pca + 0.5, color="red", linestyle="--",
                              linewidth=1.5, alpha=0.7,
                              label=f"Cutoff (n={n_pca})")
                ax_sc.set_xlabel("Principal Component", fontsize=11)
                ax_sc.set_ylabel("Explained Variance Ratio", fontsize=11)
                ax_sc.set_title(f"PCA Scree — {source_label} / {mut} "
                                f"(using {n_pca}/{n_fit} PCs, "
                                f"kept var={cum_var[n_pca-1]:.1%})",
                                fontsize=12, fontweight="bold")
                ax_sc.legend(fontsize=9)
                ax_sc.set_ylim(0, max(var_ratios[0] * 1.15, cum_var[-1] * 1.05))
                ax_sc.grid(True, alpha=0.2, axis="y")
                sns.despine()
                fig_sc.tight_layout()
                scree_path = os.path.join(
                    out_dir,
                    f"pca_scree_{source_label}_{mut}.png")
                fig_sc.savefig(scree_path, dpi=args.dpi, bbox_inches="tight")
                svg_path = scree_path.replace(".png", ".svg")
                fig_sc.savefig(svg_path, format="svg", bbox_inches="tight")
                logger.info(f"    Scree plot: {scree_path}")
                if _IN_COLAB:
                    plt.show()
                plt.close(fig_sc)

            X_mut = X_use[valid_mask]
            y_mut = apoptosis[valid_mask]
            logger.info(f"    n={n_valid}, features={X_mut.shape[1]}")

            # ── Global Ridge R² (computed once, same for all k) ──
            global_r2, fold_r2s, global_preds = compute_global_ridge_r2(
                X_mut, y_mut, args.cv_folds, args.seed)
            logger.info(f"    Global Ridge R² = {global_r2:.4f} "
                        f"(folds: {fold_r2s.mean():.4f} ± {fold_r2s.std():.4f})")

            # ── K sweep: Local LOO Ridge R² ──
            for k in args.k_neighbors:
                logger.info(f"\n    ── k={k} ──")
                local_preds, aggregate_r2 = compute_local_ridge_r2(
                    X_mut, y_mut, k, args.min_local_n,
                    pca_dim=0)  # PCA already applied

                valid_local = ~np.isnan(local_preds)
                n_computed = int(valid_local.sum())

                logger.info(f"      Local LOO Ridge R² = {aggregate_r2:.4f} "
                            f"({n_computed}/{n_valid} samples predicted)")
                logger.info(f"      Gap (local LOO − global): {aggregate_r2 - global_r2:.4f}")

                # Store results
                key = (source_label, mut, k)
                all_results[key] = {
                    "local_preds": local_preds,
                    "aggregate_r2": aggregate_r2,
                    "global_r2": global_r2,
                    "global_fold_r2s": fold_r2s.tolist(),
                    "n": n_valid,
                    "n_local_computed": n_computed,
                    "y_true": y_mut,
                    "k": k,
                }

                # Local pred scatter
                plot_local_pred_scatter(
                    y_mut, local_preds, aggregate_r2, source_label, mut, k,
                    os.path.join(out_dir, f"local_pred_{source_label}_{mut}_k{k}.png"),
                    dpi=args.dpi)

        # end per-mutation loop

    # ── Per-mutation comparison plots (CNN vs SAE, per k) ──
    for mut in mutations:
        for k in args.k_neighbors:
            results_by_source = {}
            global_r2s = {}
            for source_label in sources.keys():
                key = (source_label, mut, k)
                if key in all_results:
                    results_by_source[source_label] = all_results[key]
                    global_r2s[source_label] = all_results[key]["global_r2"]

            if len(results_by_source) > 0:
                plot_local_r2_comparison(
                    results_by_source, mut, k, global_r2s,
                    os.path.join(out_dir, f"local_r2_comparison_{mut}_k{k}.png"),
                    dpi=args.dpi)

                plot_r2_gap(
                    results_by_source, mut, k, global_r2s,
                    os.path.join(out_dir, f"r2_gap_{mut}_k{k}.png"),
                    dpi=args.dpi)

    # ── Summary ──
    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY — Local vs Global Ridge R²")
    logger.info(f"{'='*80}")
    logger.info(f"  {'Source':6s} {'Mutation':8s} {'k':>4s} {'GlobalR²':>10s} "
                f"{'LocalLOOR²':>12s} {'Gap':>10s} {'n':>6s}")
    logger.info("  " + "-" * 62)

    for (source, mut, k), res in sorted(all_results.items()):
        gap = res["aggregate_r2"] - res["global_r2"]
        logger.info(f"  {source:6s} {mut:8s} {k:4d} {res['global_r2']:10.4f} "
                    f"{res['aggregate_r2']:12.4f} "
                    f"{gap:10.4f} {res['n']:6d}")

    # ── Save JSON ──
    json_path = os.path.join(out_dir, "local_vs_global_ridge_results.json")

    def _to_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    json_results = []
    for (src, mut, k), res in sorted(all_results.items()):
        json_results.append({
            "source": src,
            "mutation": mut,
            "k": k,
            **{kk: _to_serializable(vv)
               for kk, vv in res.items()
               if kk not in ("local_preds", "y_true")}
        })

    json_data = {
        "args": vars(args),
        "results": json_results,
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    logger.info(f"\n  Saved JSON: {json_path}")
    logger.info(f"  Output: {out_dir}")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()
