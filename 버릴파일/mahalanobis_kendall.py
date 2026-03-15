# ==============================================================================
# Mahalanobis Distance from Control → Kendall Tau with Apoptosis
#
# For each mutation:
#   1. (Optional) DE neuron selection: mutation vs Control → keep significant
#   2. (Optional) Gini / CV filtering
#   3. (Optional) log transform
#   4. PCA
#   5. Fit Mahalanobis on Control distribution (μ_ctrl, Σ_ctrl)
#   6. Compute d_M(x) for each mutation image
#   7. Kendall tau(d_M, apoptosis_rate)
#
# Cache mode: reads features_cache.npz from extract_features.py
#
# Usage (Colab):
#   %matplotlib inline
#   import logging; logging.basicConfig(level=logging.INFO, force=True)
#   import sys
#   sys.argv = [
#       "mahalanobis_kendall",
#       "--features_cache", "/path/to/features_cache.npz",
#       "--apoptosis_csv", "/path/to/apoptosis.csv",
#       "--filter_mode", "cv", "de",
#       "--pca_dim", "50",
#   ]
#   from kendall_correlation_coefficient.mahalanobis_kendall import main
#   main()
# ==============================================================================

import os
import argparse
import numpy as np
from scipy.stats import kendalltau

import matplotlib
import sys
_IN_COLAB = "google.colab" in sys.modules
if not _IN_COLAB:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from sae_project.step02_logging_utils import get_logger, SUPERCLASS_MAP

logger = get_logger("mahalanobis")


# ==============================================================================
# Argument Parser
# ==============================================================================
def get_args():
    p = argparse.ArgumentParser(
        description="Mahalanobis distance from Control → Kendall tau with apoptosis"
    )

    # Data
    p.add_argument("--features_cache", type=str, required=True,
                   help="Path to features_cache.npz from extract_features.py")
    p.add_argument("--apoptosis_csv", type=str, required=True,
                   help="Path to apoptosis CSV (columns: uid, intensity_rate)")

    # Dead neuron
    p.add_argument("--dead_threshold", type=float, default=1e-5)

    # Neuron filtering
    p.add_argument("--filter_mode", type=str, nargs="+", default=["none"],
                   help="Sequential filtering. e.g. '--filter_mode cv de' "
                        "applies CV globally first, then DE per-mutation")
    p.add_argument("--max_gini", type=float, default=0.75)
    p.add_argument("--min_cv", type=float, default=0.0)
    p.add_argument("--de_adj_p", type=float, default=0.05)
    p.add_argument("--de_min_log2fc", type=float, default=0.0)

    # Normalization
    p.add_argument("--norm", type=str, default="none",
                   choices=["none", "log", "median", "std",
                            "log_median", "log_std", "log_IQR", "IQR"],
                   help="Feature normalization before PCA")

    # PCA
    p.add_argument("--pca_dim", type=int, default=50,
                   help="PCA dimensions before Mahalanobis")

    # Mahalanobis
    p.add_argument("--regularize", type=float, default=1e-6,
                   help="Regularization added to covariance diagonal")

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="")
    p.add_argument("--dpi", type=int, default=200)

    return p.parse_args()


# ==============================================================================
# Mahalanobis distance from Control distribution
# ==============================================================================
def mahalanobis_from_control(
    X_ctrl: np.ndarray,
    X_query: np.ndarray,
    regularize: float = 1e-6,
) -> np.ndarray:
    """
    Compute Mahalanobis distance of each query point from Control distribution.

    Args:
        X_ctrl:  (N_ctrl, D) — Control samples in PCA space
        X_query: (N_query, D) — query samples (mutation images)
        regularize: added to covariance diagonal for numerical stability

    Returns:
        (N_query,) array of Mahalanobis distances
    """
    mu = np.mean(X_ctrl, axis=0)  # (D,)
    cov = np.cov(X_ctrl, rowvar=False)  # (D, D)

    # Regularize
    cov += np.eye(cov.shape[0]) * regularize

    # Inverse covariance
    cov_inv = np.linalg.inv(cov)

    # Mahalanobis: d_M(x) = sqrt((x-μ)ᵀ Σ⁻¹ (x-μ))
    diff = X_query - mu  # (N_query, D)
    left = diff @ cov_inv  # (N_query, D)
    d_sq = np.sum(left * diff, axis=1)  # (N_query,)
    d_sq = np.maximum(d_sq, 0)  # numerical safety

    return np.sqrt(d_sq)


# ==============================================================================
# Plot: Mahalanobis distance vs Apoptosis rate (scatter + regression line)
# ==============================================================================
MUTATION_COLORS = {"SNCA": "#E24A33", "GBA": "#348ABD", "LRRK2": "#988ED5"}


def plot_mahal_vs_apoptosis(
    distances: dict,
    apoptosis_vals: dict,
    taus: dict,
    title: str,
    output_path: str,
    dpi: int = 200,
):
    """Scatter plot of Mahalanobis distance vs apoptosis rate per mutation."""
    mutations = [m for m in ["SNCA", "GBA", "LRRK2"] if m in distances]
    n_muts = len(mutations)
    if n_muts == 0:
        return

    fig, axes = plt.subplots(1, n_muts, figsize=(6 * n_muts, 5), squeeze=False)

    for idx, mut in enumerate(mutations):
        ax = axes[0, idx]
        d = distances[mut]
        a = apoptosis_vals[mut]
        tau = taus[mut]

        color = MUTATION_COLORS.get(mut, "gray")
        ax.scatter(d, a, s=8, alpha=0.4, c=color, edgecolors="none")

        # Trend line (linear fit for visual guide)
        if len(d) > 2:
            z = np.polyfit(d, a, 1)
            x_line = np.linspace(d.min(), d.max(), 100)
            ax.plot(x_line, np.polyval(z, x_line), "--", color=color,
                    linewidth=2, alpha=0.8)

        ax.set_xlabel("Mahalanobis distance from Control", fontsize=11)
        ax.set_ylabel("Apoptosis rate", fontsize=11)
        ax.set_title(f"{mut}  (τ = {tau:.4f})", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.2)

        # Info box
        ax.text(0.95, 0.05, f"n = {len(d)}\nτ = {tau:.4f}",
                transform=ax.transAxes, fontsize=9, ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    logger.info(f"  Saved: {output_path}")


# ==============================================================================
# Plot: Summary bar chart
# ==============================================================================
def plot_summary_bar(
    results: list,
    output_path: str,
    dpi: int = 200,
):
    """Bar chart of Kendall tau per mutation across filter configs."""
    mutations = ["SNCA", "GBA", "LRRK2"]
    colors = [MUTATION_COLORS[m] for m in mutations]

    fig, ax = plt.subplots(figsize=(max(6, len(results) * 1.5), 5))

    x = np.arange(len(results))
    width = 0.25

    for i, mut in enumerate(mutations):
        taus = [r.get(f"tau_{mut}", 0) for r in results]
        ax.bar(x + i * width, taus, width, label=mut, color=colors[i], alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels([r["label"] for r in results], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Kendall τ (Mahalanobis ↔ Apoptosis)", fontsize=11)
    ax.set_title("Mahalanobis–Apoptosis Correlation", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, axis="y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    logger.info(f"  Saved summary: {output_path}")


# ==============================================================================
# Main
# ==============================================================================
def main():
    args = get_args()
    np.random.seed(args.seed)

    # ── 1. Load features ─────────────────────────────────────────────────
    from kendall_correlation_coefficient.dpt_kendall import (
        load_features_cache, load_and_match_apoptosis,
        apply_normalization,
        compute_gini_impurity, compute_cv_per_neuron, compute_de_neurons,
    )

    logger.info(f"\n{'='*60}")
    logger.info("Loading features cache")
    X, y, lines, uids, which_layer, alive_info = load_features_cache(
        args.features_cache, args.dead_threshold
    )
    superclasses = [SUPERCLASS_MAP.get(ln, ln) for ln in lines]
    superclasses_arr = np.array(superclasses)

    unique_sc, sc_counts = np.unique(superclasses, return_counts=True)
    logger.info(f"  Features: {X.shape} ({alive_info})")
    logger.info(f"  Classes: {dict(zip(unique_sc, sc_counts))}")

    # ── 2. Load apoptosis ────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("Loading apoptosis data")
    apoptosis = load_and_match_apoptosis(args.apoptosis_csv, uids)

    mutations = ["SNCA", "GBA", "LRRK2"]
    for mut in mutations:
        is_mut = superclasses_arr == mut
        has_apop = ~np.isnan(apoptosis)
        n_matched = (is_mut & has_apop).sum()
        logger.info(f"  {mut}: {n_matched} matched (of {is_mut.sum()} total)")

    # ── 3. Global filter (gini / cv applied first, DE deferred) ────────
    logger.info(f"\n{'='*60}")
    filter_steps = []
    has_de = "de" in args.filter_mode

    for fm in args.filter_mode:
        if fm == "none" or fm == "de":
            continue
        n_before = X.shape[1]

        if fm == "gini":
            gini = compute_gini_impurity(X, superclasses)
            mask = gini <= args.max_gini
            X = X[:, mask]
            step = f"gini≤{args.max_gini:.2f}: {n_before}→{X.shape[1]}"
        elif fm == "cv":
            cv = compute_cv_per_neuron(X, superclasses)
            mask = cv >= args.min_cv
            X = X[:, mask]
            step = f"cv≥{args.min_cv:.2f}: {n_before}→{X.shape[1]}"
        else:
            logger.warning(f"  Unknown filter: {fm}")
            continue

        filter_steps.append(step)
        logger.info(f"  Filter [{fm}]: {step}")

    if has_de:
        filter_steps.append("DE (per-mutation)")
        logger.info("  DE mode: per-mutation neuron selection (applied below)")

    filter_label = " → ".join(filter_steps) if filter_steps else "none"
    if not filter_steps:
        logger.info("  Filter: none")

    # ── 4. Output directory ──────────────────────────────────────────────
    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = os.path.join(os.path.dirname(args.features_cache),
                               "mahalanobis_kendall")
    os.makedirs(out_dir, exist_ok=True)

    # ── 5. Per-mutation Mahalanobis analysis ──────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"Mahalanobis analysis (norm={args.norm}, pca={args.pca_dim}, "
                 f"filter={filter_label})")
    logger.info("=" * 60)

    ctrl_mask_global = superclasses_arr == "Control"
    all_distances = {}
    all_apoptosis = {}
    all_taus = {}

    summary_results = []

    for mut in mutations:
        logger.info(f"\n  ── {mut} vs Control ──")

        mut_mask_global = superclasses_arr == mut

        # --- DE filter: per-mutation neuron selection ---
        if has_de:
            de_result = compute_de_neurons(
                X, superclasses, mut,
                adj_p_threshold=args.de_adj_p,
                min_log2fc=args.de_min_log2fc,
            )

            if de_result["n_selected"] < 3:
                logger.warning(f"    Only {de_result['n_selected']} DE neurons — skipping")
                all_taus[mut] = 0.0
                continue

            # Subset: Control + this mutation, DE neurons only
            keep_samples = ctrl_mask_global | mut_mask_global
            X_sub = X[keep_samples][:, de_result["mask"]]
            superclasses_sub = superclasses_arr[keep_samples]
            apoptosis_sub = apoptosis[keep_samples]

            logger.info(f"    DE subset: {X_sub.shape[0]} samples × "
                         f"{X_sub.shape[1]} neurons")
        else:
            # Global filter already applied, use all
            keep_samples = ctrl_mask_global | mut_mask_global
            X_sub = X[keep_samples]
            superclasses_sub = superclasses_arr[keep_samples]
            apoptosis_sub = apoptosis[keep_samples]

        # --- Normalize ---
        if args.norm != "none":
            X_sub_norm = apply_normalization(X_sub, args.norm)
        else:
            X_sub_norm = X_sub.copy()

        # --- PCA ---
        current_dim = X_sub_norm.shape[1]
        pca_dim = min(args.pca_dim, current_dim, X_sub_norm.shape[0] - 1)
        if current_dim > pca_dim:
            pca = PCA(n_components=pca_dim, random_state=args.seed)
            X_pca = pca.fit_transform(X_sub_norm)
            var_exp = np.sum(pca.explained_variance_ratio_)
            logger.info(f"    PCA: {current_dim} → {pca_dim}, "
                         f"var explained: {var_exp:.1%}")
        else:
            X_pca = X_sub_norm
            var_exp = 1.0

        # --- Mahalanobis from Control ---
        ctrl_in_sub = superclasses_sub == "Control"
        mut_in_sub = superclasses_sub == mut

        X_ctrl_pca = X_pca[ctrl_in_sub]
        X_mut_pca = X_pca[mut_in_sub]

        d_mahal = mahalanobis_from_control(
            X_ctrl_pca, X_mut_pca, regularize=args.regularize
        )
        logger.info(f"    Mahalanobis: min={d_mahal.min():.2f}, "
                     f"median={np.median(d_mahal):.2f}, "
                     f"max={d_mahal.max():.2f}")

        # --- Kendall tau ---
        apop_mut = apoptosis_sub[mut_in_sub]
        valid = ~np.isnan(apop_mut)

        if valid.sum() < 10:
            logger.warning(f"    Too few valid apoptosis values: {valid.sum()}")
            all_taus[mut] = 0.0
            continue

        tau, pval = kendalltau(d_mahal[valid], apop_mut[valid])
        tau = tau if not np.isnan(tau) else 0.0
        all_taus[mut] = tau

        logger.info(f"    Kendall τ = {tau:.4f} (p = {pval:.2e}, "
                     f"n = {valid.sum()})")

        all_distances[mut] = d_mahal[valid]
        all_apoptosis[mut] = apop_mut[valid]

        # Also compute tau for Control (should be ~0)
        d_ctrl = mahalanobis_from_control(
            X_ctrl_pca, X_ctrl_pca, regularize=args.regularize
        )
        apop_ctrl = apoptosis_sub[ctrl_in_sub]
        valid_ctrl = ~np.isnan(apop_ctrl)
        if valid_ctrl.sum() >= 10:
            tau_ctrl, p_ctrl = kendalltau(d_ctrl[valid_ctrl], apop_ctrl[valid_ctrl])
            logger.info(f"    Control self τ = {tau_ctrl:.4f} (sanity check, "
                         f"should be ~0)")

    # Summary result
    result_entry = {
        "label": f"{filter_label}_norm={args.norm}_pca{args.pca_dim}",
    }
    for mut in mutations:
        result_entry[f"tau_{mut}"] = all_taus.get(mut, 0.0)
    summary_results.append(result_entry)

    # ── 6. Plots ─────────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("Plotting")

    plot_title = (
        f"Mahalanobis → Apoptosis\n"
        f"filter={filter_label}, norm={args.norm}, PCA={args.pca_dim}"
    )
    fm_str = "_".join(args.filter_mode)
    plot_path = os.path.join(
        out_dir,
        f"mahal_{fm_str}_norm{args.norm}_pca{args.pca_dim}.png"
    )
    plot_mahal_vs_apoptosis(
        all_distances, all_apoptosis, all_taus,
        title=plot_title, output_path=plot_path, dpi=args.dpi,
    )

    # ── 7. Summary ───────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Filter: {filter_label}")
    logger.info(f"  Norm: {args.norm}")
    logger.info(f"  PCA: {args.pca_dim}")
    for mut in mutations:
        logger.info(f"  {mut}: τ = {all_taus.get(mut, 0.0):.4f}")

    # CSV
    csv_path = os.path.join(out_dir, f"mahalanobis_summary_{which_layer}.csv")
    import csv
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            header = ["Filter", "Norm", "PCA_dim", "Regularize"]
            for mut in mutations:
                header.extend([f"τ_{mut}", f"n_{mut}"])
            writer.writerow(header)
        row = [filter_label, args.norm, args.pca_dim, args.regularize]
        for mut in mutations:
            row.append(f"{all_taus.get(mut, 0.0):.4f}")
            row.append(len(all_distances.get(mut, [])))
        writer.writerow(row)
    logger.info(f"  Appended to CSV: {csv_path}")

    logger.info(f"\n{'='*60}")
    logger.info("Mahalanobis analysis complete!")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
