# ==============================================================================
# Local Ridge vs Global Ridge — Residual Analysis
#
# "Locally linear, globally curved" 가설을 직접 검증:
#   - Global Ridge: 전체 mutation 데이터에 Ridge fit → R²
#   - Local Ridge: 각 샘플의 KNN 이웃에서 Ridge fit → per-sample local R²
#   - local R² >> global R² 이면 국소적으로는 선형이지만 전역적으로 비선형
#
# CNN feature vector와 SAE feature vector 각각에 대해 수행.
#
# Usage (Colab):
# !python -m apoptosis_prediction.local_vs_global_ridge \
#     --cnn_cache "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/MoCo_seed87/CNN_GAP/cnn_gap_stage5_out_all.npz" \
#     --sae_cache "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/MoCo_seed87/SAE_seed856_no_L2norm_loss/features_cache_stage5_out_normrestored_all_no_SAE_GAP_L2_norm_again_d8192_sp800.npz" \
#     --apoptosis_csv "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/세포이미지별 사멸율/이미지별_세포사멸율_7200.csv" \
#     --k_neighbors 50 \
#     --n_permutations 500 \
#     --gap_l2_norm \
#     --dead_threshold 1e-5 \
#     --output_dir "/content/local_vs_global_ridge"
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
import seaborn as sns

from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline

from sae_project.step02_logging_utils import get_logger, SUPERCLASS_MAP
from apoptosis_prediction.local_linearity_knn import load_cache

logger = get_logger("local_vs_global_ridge")

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
sns.set_style("ticks")

ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]


# ==============================================================================
# Argument Parser
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
                   help="Apply L2 normalization to CNN feature vectors")

    p.add_argument("--k_neighbors", type=int, default=50,
                   help="K for local Ridge neighborhood (default: 50).")
    p.add_argument("--min_local_n", type=int, default=40,
                   help="Minimum samples in neighborhood for local Ridge (default: 20)")
    p.add_argument("--pca_dim", type=int, default=10,
                   help="PCA dimensions before local Ridge to avoid d>>k overfitting (default: 30). "
                        "0 = no PCA.")
    p.add_argument("--n_permutations", type=int, default=500,
                   help="Permutation null for global R² (default: 500)")
    p.add_argument("--cv_folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="")
    p.add_argument("--dpi", type=int, default=200)

    return p.parse_args()


# ==============================================================================
# Global Ridge R² (LOO-style per-sample prediction from CV)
# ==============================================================================
def compute_global_ridge_r2(X, y, cv_folds, seed):
    """
    Global Ridge R² via KFold CV. Returns overall R² and per-fold R²s.
    Reuses pattern from apoptosis_r2_test.py.
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
def compute_local_ridge_r2(X, y, k, min_local_n=20, pca_dim=30):
    """
    For each sample i:
      1. Find K nearest neighbors (excluding self)
      2. Fit Ridge on those K neighbors
      3. Predict sample i (held-out) → local_preds[i]

    Aggregate test R² = R²(y, local_preds) across all samples.
    This is a proper held-out metric: sample i is NEVER in its own training set.

    Parameters
    ----------
    pca_dim : int — PCA reduction before local Ridge (0 = no PCA).
        Critical when d >> k to prevent Ridge overfitting.

    Returns:
        local_preds: (N,) predicted value from each sample's local Ridge (held-out)
        aggregate_r2: float — R²(y, local_preds) over all valid samples
    """
    from sklearn.decomposition import PCA

    n, d = X.shape
    k_actual = min(k, n - 1)

    if k_actual < min_local_n:
        logger.warning(f"k={k_actual} < min_local_n={min_local_n}, "
                       f"local Ridge may be unreliable")

    # Optional PCA to avoid d >> k overfitting
    if pca_dim > 0 and d > pca_dim:
        logger.info(f"      PCA: {d} → {pca_dim} dims (before local Ridge)")
        pca = PCA(n_components=pca_dim, random_state=42)
        X_use = pca.fit_transform(X)
        explained = pca.explained_variance_ratio_.sum()
        logger.info(f"      PCA explained variance: {explained:.4f}")
    else:
        X_use = X

    # Find neighbors in ORIGINAL space (not PCA) for fair KNN
    nn = NearestNeighbors(n_neighbors=k_actual + 1, metric="euclidean", n_jobs=-1)
    nn.fit(X)  # KNN in original feature space
    _, indices = nn.kneighbors(X)
    # indices[:, 0] is self → neighbors are indices[:, 1:]
    neighbor_indices = indices[:, 1:]  # (N, k_actual)

    local_preds = np.full(n, np.nan)

    for i in range(n):
        nbr_idx = neighbor_indices[i]  # K neighbors of sample i
        X_nbr = X_use[nbr_idx]        # PCA-reduced features of neighbors
        y_nbr = y[nbr_idx]

        # Skip if too few unique values or too few samples
        if len(nbr_idx) < min_local_n or np.std(y_nbr) < 1e-12:
            continue

        # Fit Ridge on neighbors, predict sample i (TRUE held-out)
        try:
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", RidgeCV(alphas=ALPHAS)),
            ])
            pipe.fit(X_nbr, y_nbr)
            pred_i = pipe.predict(X_use[i:i+1])[0]  # held-out prediction
            local_preds[i] = pred_i
        except Exception:
            continue

    # Aggregate test R²: across all valid held-out predictions
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
    """
    Bar chart: Global Ridge R² vs Local LOO Ridge R² for CNN and SAE.
    """
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

    # Value annotations
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
# Plot: Scatter — local predicted vs actual (color by local R²)
# ==============================================================================
def plot_local_pred_scatter(y_true, local_preds, aggregate_r2, source_label,
                            mutation, k, output_path, dpi=200):
    """Scatter of local LOO Ridge predicted vs actual."""
    valid = ~np.isnan(local_preds)
    y_v = y_true[valid]
    pred_v = local_preds[valid]

    colors = {"CNN": "#4C72B0", "SAE": "#DD8452"}
    c = colors.get(source_label, "gray")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_v, pred_v, c=c, s=8, alpha=0.3,
               edgecolors="none", rasterized=True)

    lims = [min(y_v.min(), pred_v.min()), max(y_v.max(), pred_v.max())]
    ax.plot(lims, lims, "--", color="gray", alpha=0.5, linewidth=1)

    # Regression line
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
# Main
# ==============================================================================
def main():
    args = get_args()
    np.random.seed(args.seed)

    from kendall_correlation_coefficient.dpt_kendall import load_and_match_apoptosis

    if not args.cnn_cache and not args.sae_cache:
        raise ValueError("At least one of --cnn_cache or --sae_cache required")

    # Output
    if args.output_dir:
        out_dir = args.output_dir
    else:
        ref = args.cnn_cache or args.sae_cache
        out_dir = os.path.join(os.path.dirname(ref), "local_vs_global_ridge")
    os.makedirs(out_dir, exist_ok=True)

    # Load caches (reuse from local_linearity_knn.py)
    sources = {}
    if args.cnn_cache:
        logger.info("Loading CNN cache...")
        X, lines, uids, _ = load_cache(
            args.cnn_cache, args.dead_threshold, apply_l2_norm=args.gap_l2_norm)
        sources["CNN"] = (X, lines, uids)

    if args.sae_cache:
        logger.info("Loading SAE cache...")
        X, lines, uids, _ = load_cache(
            args.sae_cache, args.dead_threshold, apply_l2_norm=False)
        sources["SAE"] = (X, lines, uids)

    mutations = ["SNCA", "GBA", "LRRK2"]
    all_results = {}

    for mut in mutations:
        logger.info(f"\n{'='*60}")
        logger.info(f"  Mutation: {mut}")
        logger.info(f"{'='*60}")

        results_by_source = {}
        global_r2s = {}

        for source_label, (X, lines, uids) in sources.items():
            logger.info(f"\n  ── {source_label} ──")

            superclasses = np.array([SUPERCLASS_MAP.get(ln, ln) for ln in lines])
            apoptosis = load_and_match_apoptosis(args.apoptosis_csv, uids)
            valid_mask = (superclasses == mut) & np.isfinite(apoptosis)
            n_valid = int(valid_mask.sum())

            if n_valid < args.min_local_n * 2:
                logger.warning(f"    Too few samples ({n_valid}), skipping")
                continue

            X_mut = X[valid_mask]
            y_mut = apoptosis[valid_mask]
            logger.info(f"    n={n_valid}, features={X_mut.shape[1]}")

            # ── Global Ridge R² ──
            global_r2, fold_r2s, global_preds = compute_global_ridge_r2(
                X_mut, y_mut, args.cv_folds, args.seed)
            global_r2s[source_label] = global_r2
            logger.info(f"    Global Ridge R² = {global_r2:.4f} "
                        f"(folds: {fold_r2s.mean():.4f} ± {fold_r2s.std():.4f})")

            # ── Local LOO Ridge R² ──
            logger.info(f"    Computing local LOO Ridge R² (k={args.k_neighbors}, "
                        f"pca_dim={args.pca_dim})...")
            local_preds, aggregate_r2 = compute_local_ridge_r2(
                X_mut, y_mut, args.k_neighbors, args.min_local_n,
                pca_dim=args.pca_dim)

            valid_local = ~np.isnan(local_preds)
            n_computed = int(valid_local.sum())

            logger.info(f"    Local LOO Ridge R² = {aggregate_r2:.4f} "
                        f"({n_computed}/{n_valid} samples predicted)")
            logger.info(f"    Gap (local LOO − global): {aggregate_r2 - global_r2:.4f}")

            results_by_source[source_label] = {
                "local_preds": local_preds,
                "aggregate_r2": aggregate_r2,
                "global_r2": global_r2,
                "global_fold_r2s": fold_r2s.tolist(),
                "n": n_valid,
                "n_local_computed": n_computed,
                "y_true": y_mut,
            }

            # Local pred scatter
            plot_local_pred_scatter(
                y_mut, local_preds, aggregate_r2, source_label, mut,
                args.k_neighbors,
                os.path.join(out_dir, f"local_pred_{source_label}_{mut}_k{args.k_neighbors}.png"),
                dpi=args.dpi)

        # ── Comparison plots (if both sources available) ──
        if len(results_by_source) > 0:
            plot_local_r2_comparison(
                results_by_source, mut, args.k_neighbors, global_r2s,
                os.path.join(out_dir, f"local_r2_comparison_{mut}_k{args.k_neighbors}.png"),
                dpi=args.dpi)

            plot_r2_gap(
                results_by_source, mut, args.k_neighbors, global_r2s,
                os.path.join(out_dir, f"r2_gap_{mut}_k{args.k_neighbors}.png"),
                dpi=args.dpi)

        all_results[mut] = {
            src: {k: v for k, v in res.items()
                  if k not in ("local_preds", "y_true")}
            for src, res in results_by_source.items()
        }

    # ── Summary ──
    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY — Local vs Global Ridge R²")
    logger.info(f"{'='*80}")
    logger.info(f"  {'Source':6s} {'Mutation':8s} {'GlobalR²':>10s} "
                f"{'LocalLOOR²':>12s} {'Gap':>10s} {'n':>6s}")
    logger.info("  " + "-" * 56)

    for mut in mutations:
        for src, res in all_results.get(mut, {}).items():
            gap = res["aggregate_r2"] - res["global_r2"]
            logger.info(f"  {src:6s} {mut:8s} {res['global_r2']:10.4f} "
                        f"{res['aggregate_r2']:12.4f} "
                        f"{gap:10.4f} {res['n']:6d}")

    # ── Save JSON ──
    json_path = os.path.join(out_dir, "local_vs_global_ridge_results.json")
    # Convert numpy types for JSON serialization
    def _to_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    json_data = {
        "k_neighbors": args.k_neighbors,
        "cv_folds": args.cv_folds,
        "gap_l2_norm": args.gap_l2_norm,
        "results": {
            mut: {
                src: {k: _to_serializable(v) for k, v in res.items()}
                for src, res in mut_res.items()
            }
            for mut, mut_res in all_results.items()
        },
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    logger.info(f"\n  Saved JSON: {json_path}")
    logger.info(f"  Output: {out_dir}")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()
