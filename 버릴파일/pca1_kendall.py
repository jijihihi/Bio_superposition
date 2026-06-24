# ==============================================================================
# PC1 Score from Control Medoid → Kendall Tau with Apoptosis
#
# Simple, assumption-free approach:
#   1. (Optional) CV / DE neuron filtering
#   2. PCA on Control + Mutation subset
#   3. PC1 score = signed distance from Control medoid along PC1
#   4. Kendall tau(PC1_distance, apoptosis_rate)
#
# Also sweeps PC1..PCk to find best dimension.
#
# Usage (Colab):
#   %matplotlib inline
#   import logging; logging.basicConfig(level=logging.INFO, force=True)
#   import sys
#   sys.argv = [
#       "pca1_kendall",
#       "--features_cache", "/path/to/features_cache.npz",
#       "--apoptosis_csv", "/path/to/apoptosis.csv",
#       "--filter_mode", "cv", "de",
#       "--min_cv", "0.5",
#       "--max_pc", "20",
#   ]
#   from kendall_correlation_coefficient.pca1_kendall import main
#   main()
# ==============================================================================

import argparse
import os
import sys

import matplotlib
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import kendalltau

_IN_COLAB = "google.colab" in sys.modules
if not _IN_COLAB:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from sae_project.step02_logging_utils import SUPERCLASS_MAP, get_logger

logger = get_logger("pca1_kendall")

MUTATION_COLORS = {"SNCA": "#E24A33", "GBA": "#348ABD", "LRRK2": "#988ED5"}


# ==============================================================================
# Argument Parser
# ==============================================================================
def get_args():
    p = argparse.ArgumentParser(
        description="PC score distance from Control medoid → Kendall tau"
    )

    p.add_argument("--features_cache", type=str, required=True)
    p.add_argument("--apoptosis_csv", type=str, required=True)
    p.add_argument("--dead_threshold", type=float, default=1e-5)

    # Neuron filtering (sequential)
    p.add_argument(
        "--filter_mode",
        type=str,
        nargs="+",
        default=["none"],
        help="e.g. '--filter_mode cv de'",
    )
    p.add_argument("--max_gini", type=float, default=0.75)
    p.add_argument("--min_cv", type=float, default=0.0)
    p.add_argument("--de_adj_p", type=float, default=0.05)
    p.add_argument("--de_min_log2fc", type=float, default=0.0)
    p.add_argument(
        "--de_top_k",
        type=int,
        default=0,
        help="Max DE neurons per mutation (by |log2FC| rank). "
        "0 = keep all significant. e.g. 500",
    )

    # Normalization
    p.add_argument(
        "--norm",
        type=str,
        default="none",
        choices=[
            "none",
            "log",
            "median",
            "std",
            "log_median",
            "log_std",
            "log_IQR",
            "IQR",
        ],
    )

    # PCA sweep
    p.add_argument(
        "--max_pc", type=int, default=20, help="Sweep PC1..PCk, report tau for each"
    )
    p.add_argument(
        "--n_pca", type=int, default=50, help="Total PCA components to compute"
    )

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="")
    p.add_argument("--dpi", type=int, default=200)

    return p.parse_args()


# ==============================================================================
# Plots
# ==============================================================================
def plot_pc_tau_sweep(
    mutation: str,
    pc_taus: list,
    output_path: str,
    title: str = "",
    dpi: int = 200,
):
    """Bar chart of Kendall tau per PC dimension."""
    fig, ax = plt.subplots(figsize=(max(8, len(pc_taus) * 0.5), 4))

    pcs = list(range(1, len(pc_taus) + 1))
    color = MUTATION_COLORS.get(mutation, "gray")
    bars = ax.bar(pcs, pc_taus, color=color, alpha=0.8, edgecolor="white")

    # Highlight best
    best_idx = int(np.argmax(np.abs(pc_taus)))
    bars[best_idx].set_edgecolor("black")
    bars[best_idx].set_linewidth(2)

    ax.set_xlabel("PC dimension", fontsize=11)
    ax.set_ylabel("Kendall τ", fontsize=11)
    ax.set_title(title or f"{mutation}: τ per PC", fontsize=13, fontweight="bold")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.grid(True, alpha=0.2, axis="y")

    ax.text(
        0.98,
        0.95,
        f"Best: PC{best_idx+1} (τ={pc_taus[best_idx]:.4f})",
        transform=ax.transAxes,
        fontsize=9,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_pc1_scatter(
    pc1_mut: np.ndarray,
    apop_mut: np.ndarray,
    mutation: str,
    tau: float,
    output_path: str,
    dpi: int = 200,
):
    """Scatter plot of PC1 distance vs apoptosis rate."""
    fig, ax = plt.subplots(figsize=(7, 5))
    color = MUTATION_COLORS.get(mutation, "gray")

    ax.scatter(pc1_mut, apop_mut, s=8, alpha=0.4, c=color, edgecolors="none")

    if len(pc1_mut) > 2:
        z = np.polyfit(pc1_mut, apop_mut, 1)
        x_line = np.linspace(pc1_mut.min(), pc1_mut.max(), 100)
        ax.plot(
            x_line, np.polyval(z, x_line), "--", color=color, linewidth=2, alpha=0.8
        )

    ax.set_xlabel("PC1 score (distance from Control medoid)", fontsize=11)
    ax.set_ylabel("Apoptosis rate", fontsize=11)
    ax.set_title(f"{mutation}  (τ = {tau:.4f})", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.2)

    ax.text(
        0.95,
        0.05,
        f"n = {len(pc1_mut)}\nτ = {tau:.4f}",
        transform=ax.transAxes,
        fontsize=9,
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# ==============================================================================
# Main
# ==============================================================================
def main():
    args = get_args()
    np.random.seed(args.seed)

    from kendall_correlation_coefficient.dpt_kendall import (
        apply_normalization, compute_cv_per_neuron, compute_de_neurons,
        compute_gini_impurity, load_and_match_apoptosis, load_features_cache)

    # ── 1. Load ──────────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("Loading features")
    X, y, lines, uids, which_layer, alive_info = load_features_cache(
        args.features_cache, args.dead_threshold
    )
    superclasses = [SUPERCLASS_MAP.get(ln, ln) for ln in lines]
    superclasses_arr = np.array(superclasses)
    logger.info(f"  Features: {X.shape} ({alive_info})")

    # ── 2. Load apoptosis ────────────────────────────────────────────────
    logger.info("Loading apoptosis")
    apoptosis = load_and_match_apoptosis(args.apoptosis_csv, uids)

    # ── 3. Global filters (gini / cv) ────────────────────────────────────
    logger.info(f"\n{'='*60}")
    has_de = "de" in args.filter_mode
    filter_steps = []

    for fm in args.filter_mode:
        if fm in ("none", "de"):
            continue
        n_before = X.shape[1]
        if fm == "gini":
            gini = compute_gini_impurity(X, superclasses)
            X = X[:, gini <= args.max_gini]
            step = f"gini≤{args.max_gini}: {n_before}→{X.shape[1]}"
        elif fm == "cv":
            cv = compute_cv_per_neuron(X, superclasses)
            X = X[:, cv >= args.min_cv]
            step = f"cv≥{args.min_cv}: {n_before}→{X.shape[1]}"
        else:
            continue
        filter_steps.append(step)
        logger.info(f"  Filter [{fm}]: {step}")

    if has_de:
        filter_steps.append("DE (per-mutation)")
    filter_label = " → ".join(filter_steps) if filter_steps else "none"
    if not filter_steps:
        logger.info("  Filter: none")

    # ── 4. Output directory ──────────────────────────────────────────────
    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = os.path.join(os.path.dirname(args.features_cache), "pca1_kendall")
    os.makedirs(out_dir, exist_ok=True)

    # ── 5. Per-mutation analysis ─────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(
        f"PC-score analysis (norm={args.norm}, max_pc={args.max_pc}, "
        f"filter={filter_label})"
    )
    logger.info("=" * 60)

    mutations = ["SNCA", "GBA", "LRRK2"]
    ctrl_mask = superclasses_arr == "Control"

    all_best_taus = {}

    for mut in mutations:
        logger.info(f"\n  ── {mut} vs Control ──")
        mut_mask = superclasses_arr == mut

        # --- DE filter ---
        if has_de:
            de_result = compute_de_neurons(
                X,
                superclasses,
                mut,
                adj_p_threshold=args.de_adj_p,
                min_log2fc=args.de_min_log2fc,
            )
            mask = de_result["mask"]

            # Top-k: keep only top de_top_k by |log2FC|
            if args.de_top_k > 0 and mask.sum() > args.de_top_k:
                sig_indices = np.where(mask)[0]
                abs_fc = np.abs(de_result["log2fc"][sig_indices])
                top_k_idx = sig_indices[np.argsort(abs_fc)[::-1][: args.de_top_k]]
                mask = np.zeros_like(mask)
                mask[top_k_idx] = True
                logger.info(
                    f"    DE top-{args.de_top_k}: "
                    f"{de_result['n_selected']} → {mask.sum()}"
                )

            if mask.sum() < 3:
                logger.warning(f"    Only {mask.sum()} DE neurons — skip")
                all_best_taus[mut] = 0.0
                continue

            keep = ctrl_mask | mut_mask
            X_sub = X[keep][:, mask]
            sc_sub = superclasses_arr[keep]
            apop_sub = apoptosis[keep]
            logger.info(f"    DE subset: {X_sub.shape}")
        else:
            keep = ctrl_mask | mut_mask
            X_sub = X[keep]
            sc_sub = superclasses_arr[keep]
            apop_sub = apoptosis[keep]

        # --- Normalize ---
        if args.norm != "none":
            X_sub = apply_normalization(X_sub, args.norm)

        # --- PCA ---
        n_pca = min(args.n_pca, X_sub.shape[1], X_sub.shape[0] - 1)
        pca = PCA(n_components=n_pca, random_state=args.seed)
        X_pca = pca.fit_transform(X_sub)  # (N, n_pca)
        var_exp = np.cumsum(pca.explained_variance_ratio_)
        logger.info(
            f"    PCA: {X_sub.shape[1]}D → {n_pca}D, "
            f"PC1 var: {pca.explained_variance_ratio_[0]:.1%}, "
            f"total: {var_exp[-1]:.1%}"
        )

        # --- Control medoid (most central Control point) ---
        ctrl_in_sub = sc_sub == "Control"
        mut_in_sub = sc_sub == mut
        X_ctrl_pca = X_pca[ctrl_in_sub]

        # Medoid = point closest to centroid
        centroid = X_ctrl_pca.mean(axis=0)
        dists_to_centroid = np.linalg.norm(X_ctrl_pca - centroid, axis=1)
        medoid_idx = np.argmin(dists_to_centroid)
        medoid = X_ctrl_pca[medoid_idx]
        logger.info(
            f"    Control medoid: idx={medoid_idx}, "
            f"dist_to_centroid={dists_to_centroid[medoid_idx]:.4f}"
        )

        # --- PC sweep: tau per individual PC ---
        max_pc = min(args.max_pc, n_pca)
        pc_taus = []
        pc_pvals = []

        apop_mut = apop_sub[mut_in_sub]
        valid = ~np.isnan(apop_mut)

        if valid.sum() < 10:
            logger.warning(f"    Too few valid apoptosis: {valid.sum()}")
            all_best_taus[mut] = 0.0
            continue

        X_mut_pca = X_pca[mut_in_sub]

        for k in range(max_pc):
            # Signed distance from medoid along PC(k+1)
            pc_score = X_mut_pca[:, k] - medoid[k]

            tau, pval = kendalltau(pc_score[valid], apop_mut[valid])
            tau = tau if not np.isnan(tau) else 0.0
            pc_taus.append(tau)
            pc_pvals.append(pval if not np.isnan(pval) else 1.0)

        # --- Report ---
        best_k = int(np.argmax(np.abs(pc_taus)))
        best_tau = pc_taus[best_k]
        all_best_taus[mut] = best_tau

        logger.info(f"    PC-by-PC Kendall τ (n={valid.sum()}):")
        for k in range(min(10, max_pc)):
            marker = " <<<" if k == best_k else ""
            logger.info(
                f"      PC{k+1}: τ={pc_taus[k]:+.4f} "
                f"(p={pc_pvals[k]:.2e}, "
                f"var={pca.explained_variance_ratio_[k]:.1%}){marker}"
            )
        if max_pc > 10:
            logger.info(f"      ... (showing top 10 of {max_pc})")

        logger.info(f"    Best: PC{best_k+1}, τ = {best_tau:.4f}")

        # --- Also report PC1 specifically ---
        pc1_tau = pc_taus[0]
        logger.info(f"    PC1 τ = {pc1_tau:.4f}")

        # --- Plots ---
        # 1. PC sweep bar chart
        sweep_path = os.path.join(out_dir, f"pc_sweep_{mut}_{which_layer}.png")
        plot_pc_tau_sweep(
            mut,
            pc_taus,
            sweep_path,
            title=f"{mut}: Kendall τ per PC\nfilter={filter_label}, norm={args.norm}",
            dpi=args.dpi,
        )

        # 2. PC1 scatter
        pc1_score = X_mut_pca[:, 0] - medoid[0]
        scatter_path = os.path.join(out_dir, f"pc1_scatter_{mut}_{which_layer}.png")
        plot_pc1_scatter(
            pc1_score[valid],
            apop_mut[valid],
            mut,
            pc1_tau,
            scatter_path,
            dpi=args.dpi,
        )

        # 3. Best PC scatter (if different from PC1)
        if best_k != 0:
            best_score = X_mut_pca[:, best_k] - medoid[best_k]
            best_path = os.path.join(
                out_dir, f"pc{best_k+1}_scatter_{mut}_{which_layer}.png"
            )
            plot_pc1_scatter(
                best_score[valid],
                apop_mut[valid],
                mut,
                best_tau,
                best_path,
                dpi=args.dpi,
            )

    # ── 6. Summary ───────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Filter: {filter_label}")
    logger.info(f"  Norm: {args.norm}")
    for mut in mutations:
        logger.info(f"  {mut}: best τ = {all_best_taus.get(mut, 0.0):.4f}")

    # CSV append
    csv_path = os.path.join(out_dir, f"pca1_summary_{which_layer}.csv")
    import csv

    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            header = ["Cache", "Filter", "Norm"]
            for mut in mutations:
                header.append(f"best_τ_{mut}")
            writer.writerow(header)
        row = [
            os.path.basename(args.features_cache),
            filter_label,
            args.norm,
        ]
        for mut in mutations:
            row.append(f"{all_best_taus.get(mut, 0.0):.4f}")
        writer.writerow(row)
    logger.info(f"  CSV: {csv_path}")

    logger.info(f"\n{'='*60}")
    logger.info("PC1 Kendall analysis complete!")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
