# ==============================================================================
# Intrinsic Dimension Estimation via Weyl's Law + RMT (Marchenko-Pastur)
#
# Cache-based: reads features_cache.npz from extract_features.py
# → No GPU, encoder, or shard access required
#
# Pipeline:
#   1. Load cache → apply alive_mask (configurable dead_threshold)
#   2. For each normalization config:
#       a. Normalize → PCA
#       b. Weyl's Law: sparse kNN Laplacian → eigsh → log-log slope → d = 2/slope
#       c. RMT: covariance eigenvalues → Marchenko-Pastur → count signal eigenvalues
#   3. Output: summary CSV + per-config plots + comparison bar chart
# pip install powerlaw
#

#
# Usage (Colab / local — CPU only):
#   python -m kendall_correlation_coefficient.intrinsic_dim \
#       --features_cache /path/to/features_cache_refine_out_normrestored.npz
# ==============================================================================

import argparse
import csv
import os
import random

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

try:
    import powerlaw

    HAS_POWERLAW = True
except ImportError:
    HAS_POWERLAW = False

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse from dpt_kendall (cache-based version)
from kendall_correlation_coefficient.dpt_kendall import (
    NORM_CONFIGS, apply_normalization, build_knn_graph_and_decompose,
    compute_cv_per_neuron, compute_de_neurons, compute_gini_impurity,
    load_features_cache)
from sae_project.step02_logging_utils import SUPERCLASS_MAP, get_logger

logger = get_logger("intrinsic_dim")


# ==============================================================================
# Argument Parser
# ==============================================================================
def get_args():
    p = argparse.ArgumentParser(
        description="Intrinsic dimension estimation via Weyl's Law and RMT (cache-based)"
    )

    # Cache input (required)
    p.add_argument(
        "--features_cache",
        type=str,
        required=True,
        help="Path to .npz cache from extract_features.py",
    )

    # Output
    p.add_argument("--output_dir", type=str, default="")

    # Dead neuron threshold
    p.add_argument("--dead_threshold", type=float, default=1e-5)

    # Gini impurity filter
    p.add_argument(
        "--max_gini",
        type=float,
        default=1.0,
        help="Max Gini impurity per neuron (default 1.0 = no filter)",
    )

    # CV filter
    p.add_argument(
        "--filter_mode",
        nargs="*",
        default=[],
        choices=["gini", "cv", "de"],
        help="Global pre-filters to apply. 'de' triggers per-mutation subsets.",
    )
    p.add_argument(
        "--min_cv",
        type=float,
        default=0.3,
        help="Minimum CV (coefficient of variation) for neuron selection",
    )

    # DE filter
    p.add_argument(
        "--de_adj_p",
        type=float,
        default=0.05,
        help="Adjusted p-value threshold for DE neurons",
    )
    p.add_argument(
        "--de_min_log2fc",
        type=float,
        default=0.0,
        help="Minimum |log2FC| for DE neurons",
    )

    # Apoptosis CSV (needed for DE context but optional)
    p.add_argument(
        "--apoptosis_csv",
        type=str,
        default="",
        help="Path to apoptosis CSV (optional, for context only)",
    )

    # Seed
    p.add_argument("--seed", type=int, default=42)

    # PCA
    p.add_argument(
        "--pca_dim",
        type=int,
        default=100,
        help="PCA target dimensions before Laplacian",
    )

    # kNN graph
    p.add_argument("--n_neighbors", type=int, default=50)

    # Weyl's law fitting range
    p.add_argument(
        "--weyl_start",
        type=int,
        default=2,
        help="Start eigenvalue index for Weyl log-log fit",
    )
    p.add_argument(
        "--weyl_end",
        type=int,
        default=40,
        help="End eigenvalue index for Weyl log-log fit",
    )

    # Number of eigenvalues
    p.add_argument(
        "--n_eigen",
        type=int,
        default=50,
        help="Number of Laplacian eigenvalues to compute",
    )

    # Plot
    p.add_argument("--dpi", type=int, default=150)

    return p.parse_args()


# ==============================================================================
# Weyl's Law: log(λ_k) ~ (2/d) * log(k)  →  d = 2 / slope
# ==============================================================================
def estimate_weyl_dimension(
    evals: np.ndarray,
    fit_start: int = 2,
    fit_end: int = 20,
) -> dict:
    """
    Estimate intrinsic dimension from Laplacian eigenvalues using Weyl's law.

    Weyl's asymptotic law for a d-dimensional compact Riemannian manifold:
        λ_k ~ C · k^{2/d}  ⟹  log(λ_k) = (2/d) · log(k) + const

    d_est = 2 / slope.
    """
    k_indices = np.arange(fit_start, fit_end + 1)

    if fit_end >= len(evals):
        fit_end = len(evals) - 1
        k_indices = np.arange(fit_start, fit_end + 1)

    lambda_vals = evals[fit_start : fit_end + 1]
    valid_mask = lambda_vals > 1e-9

    result = {
        "estimated_d": 0.0,
        "slope": 0.0,
        "intercept": 0.0,
        "r2": 0.0,
        "log_k": None,
        "log_lambda": None,
        "fit_line": None,
    }

    if np.sum(valid_mask) > 3:
        log_k = np.log(k_indices[valid_mask]).reshape(-1, 1)
        log_lambda = np.log(lambda_vals[valid_mask]).reshape(-1, 1)
        reg = LinearRegression().fit(log_k, log_lambda)

        slope = reg.coef_[0][0]
        intercept = reg.intercept_[0]
        r2 = reg.score(log_k, log_lambda)
        estimated_d = 2.0 / slope if slope > 0.01 else 999.9

        result.update(
            {
                "estimated_d": estimated_d,
                "slope": slope,
                "intercept": intercept,
                "r2": r2,
                "log_k": log_k,
                "log_lambda": log_lambda,
                "fit_line": slope * log_k + intercept,
            }
        )

    return result


# ==============================================================================
# RMT (Marchenko-Pastur) intrinsic dimension — 3 methods
# ==============================================================================
def _compute_cov_eigenvalues(X: np.ndarray) -> np.ndarray:
    """Compute sorted (desc) covariance eigenvalues."""
    N = X.shape[0]
    X_centered = X - X.mean(axis=0)
    cov_eigenvalues = np.linalg.svd(X_centered, compute_uv=False) ** 2 / (N - 1)
    return np.sort(cov_eigenvalues)[::-1]


def _mp_lambda_plus(sigma2: float, gamma: float) -> float:
    """Upper edge of the Marchenko-Pastur distribution."""
    return sigma2 * (1 + np.sqrt(gamma)) ** 2


def _mp_lambda_minus(sigma2: float, gamma: float) -> float:
    """Lower edge of the Marchenko-Pastur distribution."""
    return sigma2 * (1 - np.sqrt(gamma)) ** 2


def estimate_rmt_standard(X: np.ndarray, noise_percentile: float = 25.0) -> dict:
    """
    Method 1: Standard MP with low-percentile σ² estimation.

    σ²를 하위 퍼센타일(기본 25%)에서 추정하여
    순수 노이즈 크기를 먼저 재고, λ₊ 위의 고유값 수를 셈.
    """
    N, p = X.shape
    gamma = p / N
    eigs = _compute_cov_eigenvalues(X)

    sigma2 = np.percentile(eigs, noise_percentile)
    lp = _mp_lambda_plus(sigma2, gamma)
    lm = _mp_lambda_minus(sigma2, gamma)
    n_signal = int(np.sum(eigs > lp))

    return {
        "method": "Standard MP",
        "estimated_d": n_signal,
        "lambda_plus": lp,
        "lambda_minus": lm,
        "sigma2": sigma2,
        "gamma": gamma,
        "eigenvalues": eigs,
    }


def _gavish_donoho_omega(beta: float) -> float:
    """
    Gavish-Donoho (2014) optimal hard threshold coefficient ω(β)
    for singular values.  β = min(m,n)/max(m,n) ∈ (0, 1].

    ω(β) = √(2(β+1) + 8β / ((β+1) + √(β²+14β+1)))

    For eigenvalues (singular values²), the threshold is ω(β)²·σ².
    """
    numerator = 8 * beta
    denominator = (beta + 1) + np.sqrt(beta**2 + 14 * beta + 1)
    return np.sqrt(2 * (beta + 1) + numerator / denominator)


def estimate_rmt_gavish_donoho(X: np.ndarray) -> dict:
    """
    Method 2: Gavish-Donoho (2014) optimal hard thresholding.

    σ를 모를 때 median singular value로 추정:
        σ̂ = median(s) / μ_β
    여기서 μ_β ≈ √median of MP distribution.

    최적 threshold for eigenvalues: τ_eig = ω(β)² · σ̂²
    """
    N, p = X.shape
    beta = min(p, N) / max(p, N)
    gamma = p / N

    X_centered = X - X.mean(axis=0)
    singular_values = np.linalg.svd(X_centered, compute_uv=False)
    eigs = singular_values**2 / (N - 1)
    eigs = np.sort(eigs)[::-1]

    # Estimate σ using median singular value
    # For MP: median of the squared singular value distribution ≈ σ² · μ_MP(β)
    # Approximation: μ_MP(β) ≈ (1 + √β)² — upper-edge proxy scaled down
    # More precise: use the median of the MP distribution
    median_sv = np.median(singular_values)
    # MP median approximation (Gavish & Donoho use μ_β from the MP CDF)
    # A good approximation: μ_β ≈ √((1 + √β)² · correction)
    # For simplicity, use: σ̂² = median(eigs) as bulk median
    sigma2_hat = np.median(eigs)

    omega = _gavish_donoho_omega(beta)
    # Threshold on eigenvalues: ω² · σ̂²
    tau_eig = omega**2 * sigma2_hat

    lp = _mp_lambda_plus(sigma2_hat, gamma)
    n_signal = int(np.sum(eigs > tau_eig))

    return {
        "method": "Gavish-Donoho",
        "estimated_d": n_signal,
        "tau_eig": tau_eig,
        "omega": omega,
        "beta": beta,
        "sigma2": sigma2_hat,
        "lambda_plus": lp,
        "gamma": gamma,
        "eigenvalues": eigs,
    }


def estimate_rmt_incremental(X: np.ndarray, n_sigma: float = 3.0) -> dict:
    """
    Method 3: Incremental MP fitting.

    하위 고유값부터 하나씩 늘려가며 MP 분포를 피팅.
    실제 고유값이 이론적 λ₊ + n_sigma·σ 범위를 벗어나면
    노이즈 벌크가 끝나고 신호가 시작되는 지점으로 판단.
    """
    N, p = X.shape
    eigs = _compute_cov_eigenvalues(X)

    # Start from the smallest eigenvalue, expand the noise bulk
    eigs_asc = eigs[::-1]  # ascending order
    best_cutoff = len(eigs)

    for k in range(10, len(eigs_asc)):
        bulk = eigs_asc[:k]  # candidate noise eigenvalues
        sigma2_k = np.mean(bulk)
        gamma_k = k / N  # effective γ using k noise dims

        lp_k = _mp_lambda_plus(sigma2_k, gamma_k)

        # Check: does the next eigenvalue exceed the MP upper edge?
        if k < len(eigs_asc):
            next_eig = eigs_asc[k]
            # Allow some tolerance
            tolerance = n_sigma * np.sqrt(sigma2_k) * (gamma_k ** (1 / 6))
            if next_eig > lp_k + tolerance:
                best_cutoff = k
                break

    n_signal = len(eigs) - best_cutoff
    sigma2_final = np.mean(eigs_asc[:best_cutoff]) if best_cutoff > 0 else eigs[-1]
    gamma_final = best_cutoff / N
    lp_final = _mp_lambda_plus(sigma2_final, gamma_final)

    return {
        "method": "Incremental",
        "estimated_d": n_signal,
        "noise_dims": best_cutoff,
        "sigma2": sigma2_final,
        "lambda_plus": lp_final,
        "gamma": gamma_final,
        "eigenvalues": eigs,
    }


def fit_powerlaw(eigenvalues: np.ndarray) -> dict:
    """
    Power-law fit using the `powerlaw` library (Clauset, Shalizi, Newman 2009).

    Uses MLE (Maximum Likelihood Estimation) for α instead of naive
    log-log linear regression, and KS test for goodness-of-fit.
    """
    _default = {
        "alpha": 0.0,
        "xmin": 0.0,
        "D": 1.0,
        "R": 0.0,
        "p_value": 1.0,
        "sigma": 0.0,
        "fit_object": None,
    }

    eigs = np.sort(eigenvalues)[::-1]
    eigs = eigs[eigs > 1e-12]

    if not HAS_POWERLAW:
        logger.warning("  powerlaw library not installed (pip install powerlaw)")
        return _default

    if len(eigs) < 10:
        logger.warning(f"  Too few eigenvalues for power-law fit: {len(eigs)}")
        return _default

    try:
        # powerlaw.Fit: MLE fitting with automatic xmin selection
        fit = powerlaw.Fit(eigs, verbose=False)

        # Compare power_law vs lognormal (WeightWatcher style)
        R, p_val = fit.distribution_compare(
            "power_law", "lognormal", normalized_ratio=True
        )

        return {
            "alpha": fit.alpha,
            "xmin": fit.xmin,
            "D": fit.D,  # KS distance
            "R": R,  # >0 favors power-law, <0 favors lognormal
            "p_value": p_val,
            "sigma": fit.sigma,  # standard error
            "fit_object": fit,  # for plotting CCDF
        }
    except Exception as e:
        logger.warning(
            f"  powerlaw.Fit failed: {e} "
            f"(n_eigs={len(eigs)}, range=[{eigs[-1]:.6f}, {eigs[0]:.6f}])"
        )
        return _default


def estimate_rmt_dimension(X: np.ndarray) -> dict:
    """
    Run all 3 RMT methods + power-law fit and return combined results.
    """
    std = estimate_rmt_standard(X, noise_percentile=70.0)
    gd = estimate_rmt_gavish_donoho(X)
    inc = estimate_rmt_incremental(X, n_sigma=3.0)
    pl = fit_powerlaw(gd["eigenvalues"])

    return {
        "standard": std,
        "gavish_donoho": gd,
        "incremental": inc,
        "powerlaw": pl,
        # For backward compat: use Gavish-Donoho as primary
        "estimated_d": gd["estimated_d"],
        "lambda_plus": gd["lambda_plus"],
        "sigma2": gd["sigma2"],
        "gamma": gd["gamma"],
        "eigenvalues": gd["eigenvalues"],
    }


# ==============================================================================
# Eigengap
# ==============================================================================
def compute_eigengap(evals: np.ndarray, k_limit: int = 20) -> np.ndarray:
    return np.diff(evals[: k_limit + 1])


# ==============================================================================
# Plotting
# ==============================================================================
def plot_intrinsic_dim(
    weyl_result: dict,
    rmt_result: dict,
    eigengaps: np.ndarray,
    config_name: str,
    output_path: str,
    dpi: int = 150,
):
    """5-panel: Weyl log-log | Eigengap | RMT overview | Log-scale | Power-law fit"""
    fig, axes = plt.subplots(1, 5, figsize=(30, 5))

    weyl_d = weyl_result["estimated_d"]
    std_d = rmt_result["standard"]["estimated_d"]
    gd_d = rmt_result["gavish_donoho"]["estimated_d"]
    inc_d = rmt_result["incremental"]["estimated_d"]
    fig.suptitle(
        f"{config_name}  |  Weyl d={weyl_d:.1f}  |  "
        f"RMT: Std={std_d}  GD={gd_d}  Inc={inc_d}",
        fontsize=13,
        fontweight="bold",
    )

    # Panel 1: Weyl's Law
    ax = axes[0]
    if weyl_result["log_k"] is not None:
        ax.scatter(
            weyl_result["log_k"],
            weyl_result["log_lambda"],
            color="black",
            s=20,
            zorder=3,
            label="Data",
        )
        ax.plot(
            weyl_result["log_k"],
            weyl_result["fit_line"],
            "r--",
            linewidth=2,
            label=f"slope={weyl_result['slope']:.2f}, R²={weyl_result['r2']:.3f}",
        )
        ax.set_xlabel("log(k)")
        ax.set_ylabel("log(λ_k)")
        ax.set_title(f"Weyl's Law → d ≈ {weyl_d:.1f}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(
            0.5,
            0.5,
            "Invalid eigenvalues",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    # Panel 2: Eigengap
    ax = axes[1]
    k_vals = np.arange(1, len(eigengaps) + 1)
    ax.bar(k_vals, eigengaps, color="skyblue", edgecolor="steelblue")
    max_gap_idx = np.argmax(eigengaps) + 1
    ax.bar(max_gap_idx, eigengaps[max_gap_idx - 1], color="salmon", edgecolor="red")
    ax.set_xlabel("Index k")
    ax.set_ylabel("Eigengap")
    ax.set_title(f"Eigengap (max at k={max_gap_idx})")
    ax.grid(True, alpha=0.3)

    # Panel 3: RMT overview — all 3 thresholds on eigenvalue spectrum
    ax = axes[2]
    eigs = rmt_result["eigenvalues"]
    n_show = min(50, len(eigs))
    k_vals = np.arange(1, n_show + 1)

    # Color by Gavish-Donoho threshold (primary)
    gd_tau = rmt_result["gavish_donoho"]["tau_eig"]
    bar_colors = ["#C44E52" if eigs[i] > gd_tau else "#888888" for i in range(n_show)]
    ax.bar(k_vals, eigs[:n_show], color=bar_colors, edgecolor="none", width=0.8)

    # Standard MP λ₊
    std_lp = rmt_result["standard"]["lambda_plus"]
    ax.axhline(
        std_lp,
        color="blue",
        linestyle="--",
        linewidth=1.5,
        label=f"Std MP λ₊ (d={std_d})",
    )
    # Gavish-Donoho τ
    ax.axhline(
        gd_tau, color="red", linestyle="-", linewidth=2, label=f"GD τ (d={gd_d})"
    )
    # Incremental λ₊
    inc_lp = rmt_result["incremental"]["lambda_plus"]
    ax.axhline(
        inc_lp, color="green", linestyle=":", linewidth=1.5, label=f"Inc λ₊ (d={inc_d})"
    )

    ax.set_xlabel("Eigenvalue Index")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("RMT: 3-Method Comparison")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Panel 4: Zoomed view — log scale for better visibility
    ax = axes[3]
    n_show_zoom = min(100, len(eigs))
    k_vals_z = np.arange(1, n_show_zoom + 1)
    ax.semilogy(
        k_vals_z,
        eigs[:n_show_zoom],
        "o-",
        color="#333333",
        markersize=3,
        linewidth=1,
        label="Eigenvalues",
    )
    ax.axhline(
        std_lp, color="blue", linestyle="--", linewidth=1.5, label=f"Std MP (d={std_d})"
    )
    ax.axhline(gd_tau, color="red", linestyle="-", linewidth=2, label=f"GD (d={gd_d})")
    ax.axhline(
        inc_lp, color="green", linestyle=":", linewidth=1.5, label=f"Inc (d={inc_d})"
    )
    ax.axhline(
        rmt_result["standard"]["sigma2"],
        color="gray",
        linestyle="-.",
        alpha=0.5,
        label=f"σ² = {rmt_result['standard']['sigma2']:.4f}",
    )

    ax.set_xlabel("Eigenvalue Index")
    ax.set_ylabel("Eigenvalue (log)")
    ax.set_title("Eigenvalue Spectrum (log scale)")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Panel 5: Power-law fit (powerlaw library — Clauset et al. 2009)
    ax = axes[4]
    pl = rmt_result.get("powerlaw", {})
    fit_obj = pl.get("fit_object", None)
    if fit_obj is not None:
        # Plot empirical CCDF + fitted power-law
        fit_obj.plot_ccdf(ax=ax, color="black", linewidth=1.5, label="Empirical CCDF")
        fit_obj.power_law.plot_ccdf(
            ax=ax,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f'Power-law (α={pl["alpha"]:.2f})',
        )
        ax.set_xlabel("Eigenvalue")
        ax.set_ylabel("P(X ≥ x)")
        # Annotate with key stats
        stats_text = (
            f"α = {pl['alpha']:.2f} ± {pl['sigma']:.2f}\n"
            f"xmin = {pl['xmin']:.4f}\n"
            f"KS D = {pl['D']:.3f}\n"
            f"R = {pl['R']:.3f} (p={pl['p_value']:.3f})"
        )
        ax.text(
            0.97,
            0.55,
            stats_text,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )
        ax.set_title(f"Power-Law Fit  α={pl['alpha']:.2f}")
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(
            0.5,
            0.5,
            "powerlaw not installed\npip install powerlaw",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
        )

    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {output_path}")


# ==============================================================================
# Main
# ==============================================================================
def _run_intrinsic_dim_analysis(
    X_subset,
    superclasses_subset,
    subset_name,
    norm_configs,
    args,
    out_dir,
    which_layer,
):
    """
    Run Weyl + RMT + eigengap + power-law for one subset across all normalization configs.
    Returns list of result dicts.
    """
    results = []

    for cfg_idx, (cfg_name, metric, norm_method) in enumerate(norm_configs):
        label = f"{subset_name}__{cfg_name}"
        logger.info(
            f"\n  [{cfg_idx+1}/{len(norm_configs)}] {label} "
            f"(metric={metric}, norm={norm_method})"
        )

        # Normalize
        X_norm = apply_normalization(X_subset, norm_method)

        # PCA
        current_dim = X_norm.shape[1]
        pca_dim = min(args.pca_dim, current_dim)
        if current_dim > pca_dim:
            pca = PCA(n_components=pca_dim, random_state=args.seed)
            X_pca = pca.fit_transform(X_norm)
            var_explained = np.sum(pca.explained_variance_ratio_)
            logger.info(f"    PCA: {current_dim} → {pca_dim}, var: {var_explained:.1%}")
        else:
            X_pca = X_norm
            var_explained = 1.0

        # Weyl's Law
        try:
            n_eigen_safe = min(args.n_eigen, X_pca.shape[0] - 2)
            evals, evecs = build_knn_graph_and_decompose(
                X_pca, args.n_neighbors, metric, n_eigen_safe
            )
            weyl = estimate_weyl_dimension(evals, args.weyl_start, args.weyl_end)
            eigengaps = compute_eigengap(evals, k_limit=min(n_eigen_safe - 1, 25))
            logger.info(
                f"    Weyl: d={weyl['estimated_d']:.2f}, "
                f"slope={weyl['slope']:.3f}, R²={weyl['r2']:.3f}"
            )
        except Exception as e:
            logger.error(f"    Weyl failed: {e}")
            weyl = {
                "estimated_d": 0,
                "slope": 0,
                "r2": 0,
                "log_k": None,
                "log_lambda": None,
                "fit_line": None,
                "intercept": 0,
            }
            eigengaps = np.zeros(5)

        # RMT (3 methods)
        try:
            rmt = estimate_rmt_dimension(X_pca)
            std = rmt["standard"]
            gd = rmt["gavish_donoho"]
            inc = rmt["incremental"]
            pl = rmt["powerlaw"]
            logger.info(
                f"    RMT: Std={std['estimated_d']}  GD={gd['estimated_d']}  "
                f"Inc={inc['estimated_d']}  PL α={pl['alpha']:.2f}"
            )
        except Exception as e:
            logger.error(f"    RMT failed: {e}")
            _empty = {
                "estimated_d": 0,
                "lambda_plus": 0,
                "sigma2": 0,
                "gamma": 0,
                "eigenvalues": np.zeros(5),
                "method": "",
            }
            rmt = {
                "standard": {**_empty, "lambda_minus": 0},
                "gavish_donoho": {**_empty, "tau_eig": 0, "omega": 0, "beta": 0},
                "incremental": {**_empty, "noise_dims": 0},
                "powerlaw": {
                    "alpha": 0.0,
                    "xmin": 0.0,
                    "D": 1.0,
                    "R": 0.0,
                    "p_value": 1.0,
                    "sigma": 0.0,
                    "fit_object": None,
                },
                "estimated_d": 0,
                "lambda_plus": 0,
                "sigma2": 0,
                "gamma": 0,
                "eigenvalues": np.zeros(5),
            }
            std = rmt["standard"]
            gd = rmt["gavish_donoho"]
            inc = rmt["incremental"]
            pl = rmt["powerlaw"]

        eigengap_d = int(np.argmax(eigengaps)) + 1 if len(eigengaps) > 0 else 0

        result = {
            "subset": subset_name,
            "config_name": cfg_name,
            "metric": metric,
            "norm": norm_method,
            "n_samples": X_subset.shape[0],
            "n_features": X_subset.shape[1],
            "pca_var": var_explained,
            "weyl_d": weyl["estimated_d"],
            "weyl_slope": weyl["slope"],
            "weyl_r2": weyl["r2"],
            "rmt_std_d": std["estimated_d"],
            "rmt_gd_d": gd["estimated_d"],
            "rmt_inc_d": inc["estimated_d"],
            "pl_alpha": pl["alpha"],
            "pl_D": pl["D"],
            "pl_R": pl["R"],
            "eigengap_d": eigengap_d,
        }
        results.append(result)

        # Per-config plot
        plot_path = os.path.join(out_dir, f"idim_{subset_name}__{cfg_name}.png")
        plot_intrinsic_dim(weyl, rmt, eigengaps, label, plot_path, args.dpi)

    return results


# ==============================================================================
# Main
# ==============================================================================
def main():
    args = get_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # ── 1. Load features from cache ──────────────────────────────────────
    logger.info(f"\n{'='*60}")
    X_raw, y, lines, uids, which_layer, alive_info = load_features_cache(
        args.features_cache, args.dead_threshold
    )

    # Superclass mapping
    superclasses = [SUPERCLASS_MAP.get(ln, ln) for ln in lines]
    unique_classes, class_counts = np.unique(superclasses, return_counts=True)
    logger.info(f"Classes: {dict(zip(unique_classes, class_counts))}")

    mutations = ["SNCA", "GBA", "LRRK2"]
    filter_modes = args.filter_mode

    # ── 1b. Global pre-filters (Gini) ─────────────────────────────────────
    X = X_raw.copy()
    if "gini" in filter_modes or args.max_gini < 1.0:
        logger.info(f"\nComputing per-neuron Gini impurity...")
        gini_values = compute_gini_impurity(X, superclasses)
        gini_mask = gini_values <= args.max_gini
        n_before = X.shape[1]
        X = X[:, gini_mask]
        logger.info(
            f"  Gini filter: {n_before} → {X.shape[1]} neurons "
            f"(max_gini={args.max_gini:.3f})"
        )
    else:
        gini_mask = np.ones(X.shape[1], dtype=bool)
        logger.info(f"Gini filter: disabled (max_gini=1.0)")

    # ── 1c. CV computation ────────────────────────────────────────────────
    cv_values = compute_cv_per_neuron(X, superclasses)
    cv_mask = cv_values >= args.min_cv
    logger.info(
        f"CV stats: min={cv_values.min():.4f}, median={np.median(cv_values):.4f}, "
        f"max={cv_values.max():.4f}"
    )
    logger.info(f"CV >= {args.min_cv}: {cv_mask.sum()} / {len(cv_mask)} neurons")

    # ── 2. Output directory ──────────────────────────────────────────────
    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = os.path.join(os.path.dirname(args.features_cache), "intrinsic_dim")
    os.makedirs(out_dir, exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════
    # ── 3. Build subset configs ──────────────────────────────────────────
    # ══════════════════════════════════════════════════════════════════════
    subsets = []  # List of (name, X_sub, superclasses_sub)

    # Helper: sample mask for Control + mutation
    def ctrl_mut_mask(mut):
        return np.array([(s == "Control" or s == mut) for s in superclasses])

    # ── 3a. Global subsets (all samples) ─────────────────────────────────
    # (1) All neurons (after gini)
    subsets.append(("all", X, superclasses))

    # (2) Per-mutation pair (all neurons)
    for mut in mutations:
        m = ctrl_mut_mask(mut)
        subs = [s for s, keep in zip(superclasses, m) if keep]
        subsets.append((f"ctrl_{mut}", X[m], subs))

    # ── 3b. CV-filtered subsets ──────────────────────────────────────────
    if "cv" in filter_modes:
        X_cv = X[:, cv_mask]
        logger.info(
            f"\nCV filter: {X.shape[1]} → {X_cv.shape[1]} neurons "
            f"(min_cv={args.min_cv})"
        )

        subsets.append(("cv_all", X_cv, superclasses))
        for mut in mutations:
            m = ctrl_mut_mask(mut)
            subs = [s for s, keep in zip(superclasses, m) if keep]
            subsets.append((f"cv_ctrl_{mut}", X_cv[m], subs))

    # ── 3c. DE-filtered subsets (per-mutation) ───────────────────────────
    if "de" in filter_modes:
        logger.info(
            f"\nDE filter: Wilcoxon + BH (adj_p<{args.de_adj_p}, "
            f"|log2FC|>={args.de_min_log2fc:.2f})"
        )
        for mut in mutations:
            de_result = compute_de_neurons(
                X,
                superclasses,
                mut,
                adj_p_threshold=args.de_adj_p,
                min_log2fc=args.de_min_log2fc,
            )
            de_mask = de_result["mask"]
            n_de = de_result["n_selected"]
            if n_de < 5:
                logger.warning(f"  {mut}: only {n_de} DE neurons — skipping")
                continue

            # DE only
            m = ctrl_mut_mask(mut)
            subs = [s for s, keep in zip(superclasses, m) if keep]
            X_de = X[np.ix_(m, de_mask)]
            subsets.append((f"de_{mut}", X_de, subs))

            # CV ∩ DE
            if "cv" in filter_modes:
                cv_de_mask = cv_mask & de_mask
                n_cv_de = cv_de_mask.sum()
                if n_cv_de < 5:
                    logger.warning(f"  {mut}: only {n_cv_de} CV∩DE neurons — skipping")
                    continue
                X_cv_de = X[np.ix_(m, cv_de_mask)]
                subsets.append((f"cv_de_{mut}", X_cv_de, subs))

    logger.info(f"\n{'='*60}")
    logger.info(f"Total subsets to analyze: {len(subsets)}")
    for name, Xs, _ in subsets:
        logger.info(
            f"  {name:25s} → {Xs.shape[0]:6d} samples × {Xs.shape[1]:5d} features"
        )
    logger.info("=" * 60)

    # ══════════════════════════════════════════════════════════════════════
    # ── 4. Run analysis for each subset × normalization config ───────────
    # ══════════════════════════════════════════════════════════════════════
    all_results = []

    for subset_idx, (subset_name, X_sub, sc_sub) in enumerate(subsets):
        logger.info(f"\n{'='*60}")
        logger.info(
            f"[Subset {subset_idx+1}/{len(subsets)}] {subset_name}  "
            f"({X_sub.shape[0]} samples × {X_sub.shape[1]} features)"
        )
        logger.info("=" * 60)

        results = _run_intrinsic_dim_analysis(
            X_sub,
            sc_sub,
            subset_name,
            NORM_CONFIGS,
            args,
            out_dir,
            which_layer,
        )
        all_results.extend(results)

    # ══════════════════════════════════════════════════════════════════════
    # ── 5. Summary CSV ───────────────────────────────────────────────────
    # ══════════════════════════════════════════════════════════════════════
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY: Intrinsic Dimension Estimates (all subsets)")
    logger.info("=" * 60)

    csv_path = os.path.join(out_dir, f"intrinsic_dim_summary_{which_layer}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Subset",
                "Config",
                "Metric",
                "Norm",
                "N_Samples",
                "N_Features",
                "PCA_Var",
                "Weyl_d",
                "Weyl_slope",
                "Weyl_R2",
                "RMT_Std_d",
                "RMT_GD_d",
                "RMT_Inc_d",
                "PL_alpha",
                "PL_D",
                "PL_R",
                "Eigengap_d",
            ]
        )
        for r in all_results:
            writer.writerow(
                [
                    r["subset"],
                    r["config_name"],
                    r["metric"],
                    r["norm"],
                    r["n_samples"],
                    r["n_features"],
                    f"{r['pca_var']:.4f}",
                    f"{r['weyl_d']:.2f}",
                    f"{r['weyl_slope']:.4f}",
                    f"{r['weyl_r2']:.4f}",
                    r["rmt_std_d"],
                    r["rmt_gd_d"],
                    r["rmt_inc_d"],
                    f"{r['pl_alpha']:.3f}",
                    f"{r['pl_D']:.3f}",
                    f"{r['pl_R']:.3f}",
                    r["eigengap_d"],
                ]
            )
    logger.info(f"Saved CSV: {csv_path}")

    # Print summary table
    logger.info(
        f"\n{'Subset':25s} {'Config':<16s} {'Weyl_d':>8s} {'W_R²':>8s} "
        f"{'Std':>6s} {'GD':>6s} {'Inc':>6s} {'EGap':>6s} "
        f"{'N_samp':>7s} {'N_feat':>7s}"
    )
    logger.info("-" * 105)
    for r in all_results:
        logger.info(
            f"{r['subset']:25s} {r['config_name']:<16s} "
            f"{r['weyl_d']:>8.1f} {r['weyl_r2']:>8.3f} "
            f"{r['rmt_std_d']:>6d} {r['rmt_gd_d']:>6d} "
            f"{r['rmt_inc_d']:>6d} {r['eigengap_d']:>6d} "
            f"{r['n_samples']:>7d} {r['n_features']:>7d}"
        )

    # ══════════════════════════════════════════════════════════════════════
    # ── 6. Summary comparison plot ───────────────────────────────────────
    # ══════════════════════════════════════════════════════════════════════
    # Aggregate: per-subset best (averaged across norm configs)
    subset_names_all = []
    seen = set()
    for r in all_results:
        if r["subset"] not in seen:
            seen.add(r["subset"])
            subset_names_all.append(r["subset"])

    # Average each method across norm configs for each subset
    agg = {}
    for sn in subset_names_all:
        rows = [r for r in all_results if r["subset"] == sn]
        agg[sn] = {
            "weyl_d": np.mean([r["weyl_d"] for r in rows]),
            "rmt_std_d": np.mean([r["rmt_std_d"] for r in rows]),
            "rmt_gd_d": np.mean([r["rmt_gd_d"] for r in rows]),
            "rmt_inc_d": np.mean([r["rmt_inc_d"] for r in rows]),
            "eigengap_d": np.mean([r["eigengap_d"] for r in rows]),
            "n_samples": rows[0]["n_samples"],
            "n_features": rows[0]["n_features"],
        }

    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(subset_names_all) * 0.45)))

    y_pos = np.arange(len(subset_names_all))
    bar_width = 0.15

    ax = axes[0]
    ax.barh(
        y_pos - 2 * bar_width,
        [agg[s]["weyl_d"] for s in subset_names_all],
        bar_width,
        color="#4C72B0",
        label="Weyl",
    )
    ax.barh(
        y_pos - bar_width,
        [agg[s]["rmt_std_d"] for s in subset_names_all],
        bar_width,
        color="#C44E52",
        label="RMT Std",
    )
    ax.barh(
        y_pos,
        [agg[s]["rmt_gd_d"] for s in subset_names_all],
        bar_width,
        color="#E8770E",
        label="RMT GD",
    )
    ax.barh(
        y_pos + bar_width,
        [agg[s]["rmt_inc_d"] for s in subset_names_all],
        bar_width,
        color="#8172B2",
        label="RMT Inc",
    )
    ax.barh(
        y_pos + 2 * bar_width,
        [agg[s]["eigengap_d"] for s in subset_names_all],
        bar_width,
        color="#55A868",
        label="Eigengap",
    )
    # Labels: subset name + (n_samples × n_features)
    labels = [
        f"{s}  ({agg[s]['n_samples']}×{agg[s]['n_features']})" for s in subset_names_all
    ]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Estimated Intrinsic Dimension")
    ax.set_title("Intrinsic Dimension by Subset (avg over norms)", fontweight="bold")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()

    # Panel 2: Weyl R² per subset (averaged)
    ax = axes[1]
    avg_r2 = []
    for sn in subset_names_all:
        rows = [r for r in all_results if r["subset"] == sn]
        avg_r2.append(np.mean([r["weyl_r2"] for r in rows]))
    clrs = [
        "#55A868" if r > 0.9 else "#DD8452" if r > 0.7 else "#C44E52" for r in avg_r2
    ]
    ax.barh(y_pos, avg_r2, color=clrs)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("R² (Weyl Fit Quality)")
    ax.set_title("Weyl's Law Fit Quality by Subset", fontweight="bold")
    ax.axvline(0.9, color="green", linestyle="--", alpha=0.5, label="R²=0.9")
    ax.axvline(0.7, color="orange", linestyle="--", alpha=0.5, label="R²=0.7")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="x")
    ax.set_xlim(0, 1.05)
    ax.invert_yaxis()

    fig.tight_layout()
    summary_path = os.path.join(out_dir, f"intrinsic_dim_comparison_{which_layer}.png")
    fig.savefig(summary_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved summary: {summary_path}")

    logger.info(f"\n{'='*60}")
    logger.info("Intrinsic dimension analysis complete!")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
