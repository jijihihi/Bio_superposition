# ==============================================================================
# Apoptosis R² Test — SAE features에 apoptosis 정보가 있는가?
#
# Ridge regression (5-fold CV)으로 R² 측정.
# Feature selection: Ctrl+Mut 전체로 DE/CV mask 확정 → 같은 mask로 Mut-only도 평가.
# Permutation test로 통계적 유의성 검증 (p-value).
#
# Usage (Colab):
#   %matplotlib inline
#   import logging; logging.basicConfig(level=logging.INFO, force=True)
#   import sys
#   sys.argv = [
#       "apoptosis_r2_test",
#       "--features_cache", "/path/to/features_cache.npz",
#       "--apoptosis_csv", "/path/to/apoptosis.csv",
#       "--filter_mode", "cv", "de",
#       "--min_cv", "0.5",
#       "--de_top_k", "100",
#   ]
#   from kendall_correlation_coefficient.apoptosis_r2_test import main
#   main()
# ==============================================================================

import os
import argparse
import numpy as np
import sys

import matplotlib
_IN_COLAB = "google.colab" in sys.modules
if not _IN_COLAB:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import kendalltau, pearsonr

from sae_project.step02_logging_utils import get_logger, SUPERCLASS_MAP

logger = get_logger("apoptosis_r2_test")

def _fmt_p(pval):
    """Format perm p-value, handling None."""
    return f"{pval:.4f}" if pval is not None else "N/A"


# ==============================================================================
# Argument Parser
# ==============================================================================
def get_args():
    p = argparse.ArgumentParser(
        description="Test if SAE features predict apoptosis rate (Ridge R² or XGBoost R²)"
    )
    p.add_argument("--features_cache", type=str, required=True,
                   help="Path to .npz cache (SAE: X_all+usage_ema, or CNN GAP: X_gap)")
    p.add_argument("--apoptosis_csv", type=str, required=True)
    p.add_argument("--dead_threshold", type=float, default=1e-5)
    p.add_argument("--gap_l2_norm", action="store_true",  # CNN 아니고 SAE 인 경우에도 적용된다. SAE 인 경우에는 전체 GAP L2 정규화를 할 이유가 없다. CNN인 경우에 하는 것은 세포양 보정 때문이고 SAE에 넣어줄떄 이미 L2 정규화를 했기 때문에.
                   help="Apply L2 normalization to feature vectors (useful for GAP)")
    p.add_argument("--gap_l1_norm", action="store_true",
                   help="Apply L1 normalization (library-size norm, like scRNA-seq)")
    p.add_argument("--model", type=str, default="ridge",
                   choices=["ridge", "xgboost"],
                   help="ridge: OLS-like (for L2 norm effect), xgboost: nonlinear (for best seed selection)")

    # Filtering
    p.add_argument("--filter_mode", type=str, nargs="+", default=["none"])
    p.add_argument("--min_cv", type=float, default=0.0)
    p.add_argument("--de_adj_p", type=float, default=0.05)
    p.add_argument("--de_min_log2fc", type=float, default=1.0)
    p.add_argument("--de_top_k", type=int, default=0)

    # XGBoost hyperparameters
    p.add_argument("--xgb_n_estimators", type=int, default=300)
    p.add_argument("--xgb_max_depth", type=int, default=5)
    p.add_argument("--xgb_learning_rate", type=float, default=0.04)
    p.add_argument("--xgb_subsample", type=float, default=0.8)
    p.add_argument("--xgb_colsample_bytree", type=float, default=0.8)
    p.add_argument("--xgb_early_stopping", type=int, default=30)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cv_folds", type=int, default=5)
    p.add_argument("--n_permutations", type=int, default=0,
                   help="Number of permutations for null distribution")
    p.add_argument("--output_dir", type=str, default="")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--quiet", action="store_true",
                   help="Minimal output: skip plots, fold logs, interpretation. For batch runs.")

    return p.parse_args()


# ==============================================================================
# Global feature selection: Ctrl+Mut → feature mask
# ==============================================================================
def select_features_global(
    X: np.ndarray,
    superclasses: list,
    mutation: str,
    filter_mode: list,
    min_cv: float = 0.0,
    de_adj_p: float = 0.05,
    de_min_log2fc: float = 0.0,
    de_top_k: int = 0,
):
    """
    Feature selection using Ctrl+Mut data (globally, not per-fold).
    Returns boolean mask over columns (features).

    This mask is then applied to BOTH Ctrl+Mut and Mut-only groups.
    """
    from kendall_correlation_coefficient.dpt_kendall import (
        compute_cv_per_neuron, compute_de_neurons,
    )

    d = X.shape[1]
    mask = np.ones(d, dtype=bool)

    for fm in filter_mode:
        if fm == "none":
            continue

        elif fm == "cv" and min_cv > 0:
            n_classes = len(set(superclasses))
            if n_classes < 2:
                logger.info(f"    [filter] CV skip: only {n_classes} class")
                continue
            cv = compute_cv_per_neuron(X[:, mask], superclasses)
            sub_mask = cv >= min_cv
            n_before = int(mask.sum())
            active_idx = np.where(mask)[0]
            mask[active_idx[~sub_mask]] = False
            logger.info(f"    [filter] CV(>={min_cv}): {n_before} → {int(mask.sum())}")

        elif fm == "de" and mutation is not None:
            sc_arr = np.array(superclasses)
            has_ctrl = np.any(sc_arr == "Control")
            has_mut = np.any(sc_arr == mutation)
            if not has_ctrl or not has_mut:
                logger.info(f"    [filter] DE skip: ctrl={has_ctrl}, mut={has_mut}")
                continue

            n_active = int(mask.sum())
            de_result = compute_de_neurons(
                X[:, mask], superclasses, mutation,
                adj_p_threshold=de_adj_p,
                min_log2fc=de_min_log2fc,
            )
            de_mask = de_result["mask"]
            n_sig = int(de_mask.sum())
            logger.info(f"    [filter] DE({mutation}): {n_sig}/{n_active} significant")

            if de_top_k > 0 and de_mask.sum() > de_top_k:
                sig_idx = np.where(de_mask)[0]
                abs_fc = np.abs(de_result["log2fc"][sig_idx])
                top_k_idx = sig_idx[np.argsort(abs_fc)[::-1][:de_top_k]]
                de_mask = np.zeros_like(de_mask)
                de_mask[top_k_idx] = True
                logger.info(f"    [filter] DE top_k={de_top_k}: kept {int(de_mask.sum())}")

            active_idx = np.where(mask)[0]
            mask[active_idx[~de_mask]] = False
            logger.info(f"    [filter] mask after DE: {int(mask.sum())}")

    logger.info(f"    [filter] FINAL: {int(mask.sum())}/{d} features selected")
    return mask


# ==============================================================================
# Plot: R² bar chart
# ==============================================================================
def plot_r2_summary(results, output_path, dpi=200):
    """Bar chart of R² per group."""
    fig, ax = plt.subplots(figsize=(14, 5))

    groups = [r["group"] for r in results]
    r2_means = [r["r2_mean"] for r in results]
    r2_stds = [r["r2_std"] for r in results]

    colors = {
        "All": "#2CA02C", "All Mutations": "#FF7F0E",
        "SNCA": "#E24A33", "GBA": "#348ABD", "LRRK2": "#988ED5",
        "Control": "#8C8C8C",
    }
    bar_colors = [colors.get(g, "gray") for g in groups]

    bars = ax.bar(groups, r2_means, yerr=r2_stds, color=bar_colors,
                  alpha=0.85, edgecolor="white", capsize=5)

    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_ylabel("R² (5-fold CV)", fontsize=12)
    ax.set_title("SAE Features → Apoptosis Rate (Ridge Regression)",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="y")

    for bar, mean, std in zip(bars, r2_means, r2_stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.005,
                f"{mean:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# ==============================================================================
# Plot: Predicted vs Actual scatter
# ==============================================================================
def plot_pred_vs_actual(y_true, y_pred, group, r2, pval, output_path, dpi=200):
    """Scatter of predicted vs actual apoptosis."""
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = {"SNCA": "#E24A33", "GBA": "#348ABD", "LRRK2": "#988ED5"}
    c = colors.get(group, "#2CA02C")

    ax.scatter(y_true, y_pred, s=6, alpha=0.3, c=c, edgecolors="none")

    # 1:1 line
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "--", color="gray", alpha=0.5, linewidth=1)

    # Regression line
    if len(y_true) > 2:
        z = np.polyfit(y_true, y_pred, 1)
        x_line = np.linspace(y_true.min(), y_true.max(), 100)
        ax.plot(x_line, np.polyval(z, x_line), "-", color=c, linewidth=2, alpha=0.8)

    r, _ = pearsonr(y_true, y_pred)
    tau, _ = kendalltau(y_true, y_pred)

    ax.set_xlabel("Actual Apoptosis Rate", fontsize=11)
    ax.set_ylabel("Predicted Apoptosis Rate", fontsize=11)
    ax.set_title(f"{group} (R²={r2:.4f})", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.2)

    pval_str = _fmt_p(pval)
    ax.text(0.05, 0.95,
            f"n={len(y_true)}\nR²={r2:.4f}\nr={r:.4f}\nτ={tau:.4f}\nperm p={pval_str}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# ==============================================================================
# Plot: Permutation null distribution
# ==============================================================================
def plot_permutation_null(real_r2, perm_r2s, group, pval, output_path, dpi=200):
    """Histogram of permutation null R² with real R² marked."""
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.hist(perm_r2s, bins=50, color="#888888", alpha=0.7, edgecolor="white",
            label=f"Null (n={len(perm_r2s)})")
    ax.axvline(real_r2, color="red", linewidth=2, linestyle="--",
               label=f"Real R²={real_r2:.4f}")

    ax.set_xlabel("R² (permuted)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"{group} — Permutation Test (p={_fmt_p(pval)})", fontsize=13,
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# ==============================================================================
# Core: evaluate R² with fixed feature mask
# ==============================================================================
def evaluate_r2(X, y, seed, cv_folds, n_permutations=0):
    """
    Ridge CV R² with pre-selected features (feature mask already applied).

    Parameters
    ----------
    X : np.ndarray — features (already filtered)
    y : np.ndarray — apoptosis rates
    seed : int
    cv_folds : int
    n_permutations : int — 0 to skip permutation test

    Returns dict with r2_mean, r2_std, perm_pval, etc.
    """
    valid = np.isfinite(y)
    X_v, y_v = X[valid], y[valid]

    if len(X_v) < cv_folds * 2:
        return {"r2_mean": 0.0, "r2_std": 0.0, "r2_scores": [],
                "y_true": y_v, "y_pred": np.zeros_like(y_v),
                "perm_pval": None, "perm_r2s": np.array([])}

    def _cv_pass(X_in, y_in, debug=False):
        """Run one full CV pass."""
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        fold_r2s = []
        y_pred = np.full_like(y_in, np.nan)

        for fold_i, (train_idx, test_idx) in enumerate(kf.split(X_in)):
            X_train, X_test = X_in[train_idx], X_in[test_idx]
            y_train, y_test = y_in[train_idx], y_in[test_idx]

            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])),
            ])
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            y_pred[test_idx] = pred

            ss_res = np.sum((y_test - pred) ** 2)
            ss_tot = np.sum((y_test - y_test.mean()) ** 2)
            fold_r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
            fold_r2s.append(fold_r2)

            if debug:
                alpha = pipe.named_steps["ridge"].alpha_
                logger.info(f"      Fold {fold_i}: n_train={len(train_idx)}, "
                            f"n_test={len(test_idx)}, features={X_in.shape[1]}, "
                            f"R²={fold_r2:.4f}, alpha={alpha}")

        return np.array(fold_r2s), y_pred

    # Real run
    logger.info(f"    X shape: {X_v.shape}")
    _debug = not getattr(get_args, '_quiet', False)
    real_scores, y_pred = _cv_pass(X_v, y_v, debug=_debug)
    r2_mean = real_scores.mean()
    r2_std = real_scores.std()

    # Permutation test
    if n_permutations > 0:
        rng = np.random.RandomState(seed)
        perm_r2s = np.zeros(n_permutations)

        for i in range(n_permutations):
            y_perm = rng.permutation(y_v)
            perm_scores, _ = _cv_pass(X_v, y_perm)
            perm_r2s[i] = perm_scores.mean()

            if (i + 1) % 100 == 0:
                logger.info(f"    Permutation {i+1}/{n_permutations}...")

        perm_pval = (np.sum(perm_r2s >= r2_mean) + 1) / (n_permutations + 1)
        logger.info(f"    Permutation p={perm_pval:.4f} "
                    f"(null mean={np.nanmean(perm_r2s):.4f}, real={r2_mean:.4f})")
    else:
        perm_pval = None
        perm_r2s = np.array([])
        logger.info("    Permutation test skipped")

    return {
        "r2_mean": r2_mean,
        "r2_std": r2_std,
        "r2_scores": real_scores.tolist(),
        "y_true": y_v,
        "y_pred": y_pred,
        "perm_pval": perm_pval,
        "perm_r2s": perm_r2s,
    }


# ==============================================================================
# Core: evaluate R² with XGBoost
# ==============================================================================
def evaluate_r2_xgboost(X, y, seed, cv_folds, args, n_permutations=0):
    """
    XGBoost CV R² with early stopping.
    Measures maximum nonlinear predictive capacity.
    """
    try:
        from xgboost import XGBRegressor
    except ImportError:
        raise ImportError("pip install xgboost")

    valid = np.isfinite(y)
    X_v, y_v = X[valid], y[valid]

    if len(X_v) < cv_folds * 2:
        return {"r2_mean": 0.0, "r2_std": 0.0, "r2_scores": [],
                "y_true": y_v, "y_pred": np.zeros_like(y_v),
                "perm_pval": None, "perm_r2s": np.array([])}

    def _cv_pass(X_in, y_in, debug=False):
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        fold_r2s = []
        y_pred = np.full_like(y_in, np.nan)

        for fold_i, (train_idx, test_idx) in enumerate(kf.split(X_in)):
            X_train, X_test = X_in[train_idx], X_in[test_idx]
            y_train, y_test = y_in[train_idx], y_in[test_idx]

            # Split train into train/val for early stopping
            n_train = len(X_train)
            n_val = max(1, int(n_train * 0.15))
            val_idx = np.random.RandomState(seed + fold_i).permutation(n_train)[:n_val]
            tr_idx = np.setdiff1d(np.arange(n_train), val_idx)

            # GPU if available
            try:
                import torch as _th
                _use_gpu = _th.cuda.is_available()
            except ImportError:
                _use_gpu = False

            xgb = XGBRegressor(
                n_estimators=args.xgb_n_estimators,
                max_depth=args.xgb_max_depth,
                learning_rate=args.xgb_learning_rate,
                subsample=args.xgb_subsample,
                colsample_bytree=args.xgb_colsample_bytree,
                random_state=seed,
                n_jobs=-1,
                verbosity=0,
                early_stopping_rounds=args.xgb_early_stopping,
                tree_method="hist",
                device="cuda" if _use_gpu else "cpu",
            )
            xgb.fit(
                X_train[tr_idx], y_train[tr_idx],
                eval_set=[(X_train[val_idx], y_train[val_idx])],
                verbose=False,
            )
            pred = xgb.predict(X_test)
            y_pred[test_idx] = pred

            ss_res = np.sum((y_test - pred) ** 2)
            ss_tot = np.sum((y_test - y_test.mean()) ** 2)
            fold_r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
            fold_r2s.append(fold_r2)

            if debug:
                best_iter = xgb.best_iteration if hasattr(xgb, 'best_iteration') else args.xgb_n_estimators
                logger.info(f"      Fold {fold_i}: n_train={len(tr_idx)}, "
                            f"n_val={n_val}, n_test={len(test_idx)}, "
                            f"R²={fold_r2:.4f}, best_iter={best_iter}")

        return np.array(fold_r2s), y_pred

    # Real run
    logger.info(f"    X shape: {X_v.shape} (XGBoost)")
    _debug = not getattr(get_args, '_quiet', False)
    real_scores, y_pred = _cv_pass(X_v, y_v, debug=_debug)
    r2_mean = real_scores.mean()
    r2_std = real_scores.std()

    # Permutation test
    if n_permutations > 0:
        rng = np.random.RandomState(seed)
        perm_r2s = np.zeros(n_permutations)
        for i in range(n_permutations):
            y_perm = rng.permutation(y_v)
            perm_scores, _ = _cv_pass(X_v, y_perm)
            perm_r2s[i] = perm_scores.mean()
            if (i + 1) % 50 == 0:
                logger.info(f"    Permutation {i+1}/{n_permutations}...")
        perm_pval = (np.sum(perm_r2s >= r2_mean) + 1) / (n_permutations + 1)
        logger.info(f"    Permutation p={perm_pval:.4f} "
                    f"(null mean={np.nanmean(perm_r2s):.4f}, real={r2_mean:.4f})")
    else:
        perm_pval = None
        perm_r2s = np.array([])
        logger.info("    Permutation test skipped")

    return {
        "r2_mean": r2_mean,
        "r2_std": r2_std,
        "r2_scores": real_scores.tolist(),
        "y_true": y_v,
        "y_pred": y_pred,
        "perm_pval": perm_pval,
        "perm_r2s": perm_r2s,
    }


# ==============================================================================
# Main
# ==============================================================================
def main():
    args = get_args()
    np.random.seed(args.seed)
    get_args._quiet = args.quiet  # propagate to evaluate_r2

    from kendall_correlation_coefficient.dpt_kendall import (
        load_features_cache, load_and_match_apoptosis,
    )

    # ── Load features (auto-detect SAE vs GAP cache) ──
    logger.info(f"\n{'='*60}")
    logger.info("Loading features")
    data = np.load(args.features_cache, allow_pickle=True)
    cache_keys = list(data.keys())
    logger.info(f"  Cache keys: {cache_keys}")

    if "X_gap" in data:
        # CNN GAP cache: X_gap, y, lines, uids, which_layer
        X = data["X_gap"]
        lines = data["lines"].astype(str) if data["lines"].dtype.kind != 'U' else data["lines"]
        uids = data["uids"].astype(str) if data["uids"].dtype.kind != 'U' else data["uids"]
        which_layer = str(data["which_layer"])
        alive_info = f"GAP raw, shape={X.shape}"
        logger.info(f"  Detected CNN GAP cache: {X.shape}")
    elif "X_all" in data:
        # SAE cache: X_all, y, lines, uids, usage_ema, which_layer
        X, _, lines, uids, which_layer, alive_info = load_features_cache(
            args.features_cache, args.dead_threshold
        )
    else:
        raise ValueError(f"Unknown cache format. Keys: {cache_keys}. "
                         f"Expected 'X_gap' (GAP) or 'X_all' (SAE).")

    superclasses = [SUPERCLASS_MAP.get(ln, ln) for ln in lines]
    sc_arr = np.array(superclasses)
    logger.info(f"  Features: {X.shape} ({alive_info})")

    logger.info("Loading apoptosis")
    apoptosis = load_and_match_apoptosis(args.apoptosis_csv, uids)
    n_valid = np.sum(~np.isnan(apoptosis))
    logger.info(f"  Matched: {n_valid}/{len(apoptosis)}")

    # Optional: L1 normalize feature vectors (library-size normalization)
    if args.gap_l1_norm:
        l1_norms = np.sum(np.abs(X), axis=1, keepdims=True)
        l1_norms = np.where(l1_norms == 0, 1e-12, l1_norms)
        X = X / l1_norms
        alive_info += " + L1norm"
        logger.info(f"  Applied L1 normalization (library-size, per-sample)")

    # Optional: L2 normalize feature vectors
    if args.gap_l2_norm:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-12, norms)
        X = X / norms
        alive_info += " + L2norm"
        logger.info(f"  Applied L2 normalization (per-sample)")

    # ── Output ──
    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = os.path.join(os.path.dirname(args.features_cache),
                               "apoptosis_r2_test")
    os.makedirs(out_dir, exist_ok=True)

    has_filter = any(fm != "none" for fm in args.filter_mode)

    # Select evaluation function based on --model
    if args.model == "xgboost":
        eval_fn = lambda X, y: evaluate_r2_xgboost(
            X, y, args.seed, args.cv_folds, args,
            n_permutations=args.n_permutations)
        model_tag = "XGBoost"
    else:
        eval_fn = lambda X, y: evaluate_r2(
            X, y, args.seed, args.cv_folds,
            n_permutations=args.n_permutations)
        model_tag = "Ridge"

    # ── Evaluate: mutation-only ──
    logger.info(f"\n{'='*60}")
    logger.info(f"{model_tag} R² Test ({args.cv_folds}-fold CV) — Mutation-only")
    if has_filter:
        logger.info(f"Filter: {args.filter_mode}, min_cv={args.min_cv}, "
                    f"de_adj_p={args.de_adj_p}, de_min_log2fc={args.de_min_log2fc}, "
                    f"de_top_k={args.de_top_k}")
    logger.info("=" * 60)

    mutations = ["SNCA", "GBA", "LRRK2"]
    ctrl_mask = sc_arr == "Control"
    results = []

    for mut in mutations:
        mut_mask = sc_arr == mut
        keep = ctrl_mask | mut_mask

        # --- Feature selection from Ctrl+Mut ---
        if has_filter:
            logger.info(f"\n  ── Feature selection for {mut} (Ctrl + {mut}) ──")
            sc_sub = [sc_arr[i] for i in range(len(sc_arr)) if keep[i]]
            feat_mask = select_features_global(
                X[keep], sc_sub, mut,
                filter_mode=args.filter_mode,
                min_cv=args.min_cv,
                de_adj_p=args.de_adj_p,
                de_min_log2fc=args.de_min_log2fc,
                de_top_k=args.de_top_k,
            )
            n_features = int(feat_mask.sum())
            if n_features < 2:
                logger.warning(f"    Feature selection returned {n_features} features, "
                               f"using all {X.shape[1]}")
                feat_mask = np.ones(X.shape[1], dtype=bool)
                n_features = X.shape[1]
        else:
            feat_mask = np.ones(X.shape[1], dtype=bool)
            n_features = X.shape[1]

        # Mut only (features from Ctrl+Mut selection)
        logger.info(f"\n  ── {mut} only, {n_features} features ──")
        res = eval_fn(X[mut_mask][:, feat_mask], apoptosis[mut_mask])
        res["group"] = f"{mut} only"
        results.append(res)
        logger.info(f"  {mut+' only':12s}:  R² = {res['r2_mean']:.4f} ± {res['r2_std']:.4f} "
                    f"(p = {_fmt_p(res['perm_pval'])})")

    # ── Plots ──
    if not args.quiet:
        plot_r2_summary(results, os.path.join(out_dir, f"r2_summary_{which_layer}.png"),
                        dpi=args.dpi)

        for res in results:
            if len(res.get("y_true", [])) > 10:
                safe_name = res["group"].replace(" ", "_")
                plot_pred_vs_actual(
                    res["y_true"], res["y_pred"], res["group"],
                    res["r2_mean"], res["perm_pval"],
                    os.path.join(out_dir, f"pred_vs_actual_{safe_name}_{which_layer}.png"),
                    dpi=args.dpi,
                )
                if len(res.get("perm_r2s", [])) > 0:
                    plot_permutation_null(
                        res["r2_mean"], res["perm_r2s"], res["group"],
                        res["perm_pval"],
                        os.path.join(out_dir, f"perm_null_{safe_name}_{which_layer}.png"),
                        dpi=args.dpi,
                    )

    # ── Summary ──
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info("=" * 60)
    for res in results:
        logger.info(f"  {res['group']:15s}: R² = {res['r2_mean']:.4f} ± {res['r2_std']:.4f}  "
                    f"p = {_fmt_p(res['perm_pval'])}")


    # ── Save JSON results ──
    import json
    json_results = []
    for res in results:
        jr = {
            "group": res["group"],
            "r2_mean": float(res["r2_mean"]),
            "r2_std": float(res["r2_std"]),
            "r2_scores": res.get("r2_scores", []),
            "perm_pval": float(res["perm_pval"]) if res.get("perm_pval") is not None else None,
        }
        json_results.append(jr)

    json_path = os.path.join(out_dir, f"r2_results_{which_layer}_{model_tag}.json")
    with open(json_path, "w") as f:
        json.dump({
            "model": model_tag,
            "layer": which_layer,
            "gap_l2_norm": args.gap_l2_norm,
            "features_cache": args.features_cache,
            "results": json_results,
        }, f, indent=2)
    logger.info(f"  Saved JSON: {json_path}")

    logger.info(f"\n  Output: {out_dir}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
