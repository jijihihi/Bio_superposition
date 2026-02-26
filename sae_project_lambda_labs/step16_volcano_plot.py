# ==============================================================================
# Volcano Plot: CNN GAP vs SAE — Class-specific feature discovery
#
# Analogy: CNN channels ≈ genes, SAE neurons ≈ refined transcripts
#   → Volcano plot shows how many "features" are differentially
#     expressed (DE) between Control and each Mutation.
#   → SAE should yield more class-specific features than raw CNN GAP.
#
# Usage (Colab):
#   import sys
#   sys.argv = [
#       "step16_volcano_plot",
#       "--cnn_gap_cache", "/path/to/cnn_gap_stage5_out_all.npz",
#       "--sae_cache", "/path/to/features_stage5_out_all.npz",
#       "--output_dir", "/path/to/output",
#   ]
#   from sae_project_lambda_labs.step16_volcano_plot import main
#   main()
# ==============================================================================

import os
import sys
import argparse
import numpy as np

import matplotlib
_IN_COLAB = "google.colab" in sys.modules
if not _IN_COLAB:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

from sae_project.step02_logging_utils import get_logger, SUPERCLASS_MAP

logger = get_logger("volcano_plot")


# ==============================================================================
# Differential Expression Analysis
# ==============================================================================
def compute_de(X, superclasses, mutation, adj_p_threshold=0.05, min_log2fc=0.0):
    """
    DEG-like feature selection: Wilcoxon rank-sum test + BH correction.
    Control vs Mutation for each feature (channel or neuron).

    Returns dict: adj_pvalues, log2fc, mask, n_sig, n_up, n_down
    """
    sc_arr = np.array(superclasses)
    ctrl_mask = sc_arr == "Control"
    mut_mask = sc_arr == mutation

    X_ctrl = X[ctrl_mask]
    X_mut = X[mut_mask]
    d = X.shape[1]

    pvals = np.ones(d)
    eps = 1e-10

    # Log2 fold change
    ctrl_means = X_ctrl.mean(axis=0)
    mut_means = X_mut.mean(axis=0)
    log2fc = np.log2((mut_means + eps) / (ctrl_means + eps))

    # Wilcoxon rank-sum test per feature
    for j in range(d):
        c_vals = X_ctrl[:, j]
        m_vals = X_mut[:, j]
        if c_vals.std() == 0 and m_vals.std() == 0:
            continue
        try:
            _, p = mannwhitneyu(c_vals, m_vals, alternative="two-sided")
            pvals[j] = p
        except ValueError:
            pass

    # BH correction
    _, adj_p, _, _ = multipletests(pvals, method="fdr_bh")

    # Significance mask
    sig_mask = adj_p < adj_p_threshold
    if min_log2fc > 0:
        sig_mask &= np.abs(log2fc) >= min_log2fc

    n_up = int(np.sum(sig_mask & (log2fc > 0)))
    n_down = int(np.sum(sig_mask & (log2fc < 0)))

    return {
        "adj_pvalues": adj_p,
        "log2fc": log2fc,
        "mask": sig_mask,
        "n_sig": int(sig_mask.sum()),
        "n_up": n_up,
        "n_down": n_down,
        "n_total": d,
    }


# ==============================================================================
# Volcano Plot
# ==============================================================================
def plot_volcano(
    de_result,
    title,
    feature_type,   # "CNN GAP Channel" or "SAE Neuron"
    output_path,
    adj_p_threshold=0.05,
    min_log2fc=0.0,
    max_neg_log10p=50,
    dpi=300,
):
    """Single volcano plot."""
    log2fc = de_result["log2fc"]
    neg_log10p = -np.log10(de_result["adj_pvalues"] + 1e-300)
    if max_neg_log10p > 0:
        # Add jitter to capped points so they form a band, not a line
        capped = neg_log10p >= max_neg_log10p
        if capped.any():
            jitter = np.random.RandomState(42).uniform(-max_neg_log10p*0.05, 0, size=capped.sum())
            neg_log10p[capped] = max_neg_log10p + jitter
        neg_log10p = np.clip(neg_log10p, 0, max_neg_log10p)
    sig_mask = de_result["mask"]

    fig, ax = plt.subplots(figsize=(7, 6))

    # Non-significant
    ns_mask = ~sig_mask
    ax.scatter(log2fc[ns_mask], neg_log10p[ns_mask],
               s=8, alpha=0.3, c="#AAAAAA", edgecolors="none", label="NS")

    # Significant UP (mutation > control)
    up_mask = sig_mask & (log2fc > 0)
    ax.scatter(log2fc[up_mask], neg_log10p[up_mask],
               s=12, alpha=0.6, c="#E24A33", edgecolors="none",
               label=f"Up ({int(up_mask.sum())})")

    # Significant DOWN (mutation < control)
    down_mask = sig_mask & (log2fc < 0)
    ax.scatter(log2fc[down_mask], neg_log10p[down_mask],
               s=12, alpha=0.6, c="#348ABD", edgecolors="none",
               label=f"Down ({int(down_mask.sum())})")

    # Threshold lines
    ax.axhline(-np.log10(adj_p_threshold), color="gray", linestyle="--",
               linewidth=0.8, alpha=0.5)
    if min_log2fc > 0:
        ax.axvline(-min_log2fc, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.axvline(min_log2fc, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("log₂ Fold Change (Mutation / Control)", fontsize=12)
    ax.set_ylabel("−log₁₀ (adjusted p-value)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.15)

    # Stats annotation
    n_sig = de_result["n_sig"]
    n_total = de_result["n_total"]
    pct = 100 * n_sig / max(n_total, 1)
    ax.text(0.02, 0.98,
            f"Total {feature_type}s: {n_total}\n"
            f"DE {feature_type}s: {n_sig} ({pct:.1f}%)\n"
            f"Up: {de_result['n_up']} | Down: {de_result['n_down']}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    logger.info(f"  Saved: {output_path}")


# ==============================================================================
# Side-by-side comparison plot
# ==============================================================================
def plot_volcano_comparison(
    de_cnn, de_sae, mutation,
    output_path, adj_p_threshold=0.05, min_log2fc=0.0, max_neg_log10p=50, dpi=300,
):
    """CNN GAP vs SAE side-by-side volcano plots."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, de_result, feat_type, color_up, color_dn in [
        (axes[0], de_cnn, "Channel", "#E24A33", "#348ABD"),
        (axes[1], de_sae, "Neuron", "#E24A33", "#348ABD"),
    ]:
        log2fc = de_result["log2fc"]
        neg_log10p = -np.log10(de_result["adj_pvalues"] + 1e-300)
        if max_neg_log10p > 0:
            capped = neg_log10p >= max_neg_log10p
            if capped.any():
                jitter = np.random.RandomState(42).uniform(-max_neg_log10p*0.05, 0, size=capped.sum())
                neg_log10p[capped] = max_neg_log10p + jitter
            neg_log10p = np.clip(neg_log10p, 0, max_neg_log10p)
        sig_mask = de_result["mask"]

        # Non-significant
        ns = ~sig_mask
        ax.scatter(log2fc[ns], neg_log10p[ns], s=6, alpha=0.2,
                   c="#CCCCCC", edgecolors="none")

        # Up & Down
        up = sig_mask & (log2fc > 0)
        dn = sig_mask & (log2fc < 0)
        ax.scatter(log2fc[up], neg_log10p[up], s=10, alpha=0.5,
                   c=color_up, edgecolors="none")
        ax.scatter(log2fc[dn], neg_log10p[dn], s=10, alpha=0.5,
                   c=color_dn, edgecolors="none")

        # Threshold
        ax.axhline(-np.log10(adj_p_threshold), color="gray",
                   linestyle="--", linewidth=0.8, alpha=0.5)
        if min_log2fc > 0:
            ax.axvline(-min_log2fc, color="gray", linestyle="--",
                       linewidth=0.8, alpha=0.5)
            ax.axvline(min_log2fc, color="gray", linestyle="--",
                       linewidth=0.8, alpha=0.5)

        n_sig = de_result["n_sig"]
        n_total = de_result["n_total"]
        pct = 100 * n_sig / max(n_total, 1)

        source = "CNN GAP" if feat_type == "Channel" else "SAE"
        ax.set_title(f"{source} — {mutation} vs Control\n"
                     f"DE {feat_type}s: {n_sig}/{n_total} ({pct:.1f}%)",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("log₂ FC", fontsize=11)
        ax.set_ylabel("−log₁₀(adj p)", fontsize=11)
        ax.grid(True, alpha=0.15)

        # Legend
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color_up,
                   markersize=7, label=f'Up ({int(up.sum())})'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color_dn,
                   markersize=7, label=f'Down ({int(dn.sum())})'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#CCCCCC',
                   markersize=7, label=f'NS ({int(ns.sum())})'),
        ]
        ax.legend(handles=legend_elements, fontsize=8, loc="upper right")

    fig.suptitle(f"Differential Feature Analysis: CNN GAP Channels vs SAE Neurons",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    logger.info(f"  Saved: {output_path}")


# ==============================================================================
# Summary bar chart: % DE features
# ==============================================================================
def plot_de_summary_bar(de_results_cnn, de_results_sae, mutations,
                        output_path, dpi=300):
    """Bar chart comparing % DE features for CNN vs SAE across mutations."""
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(mutations))
    width = 0.35

    pcts_cnn = []
    pcts_sae = []
    for mut in mutations:
        cnn = de_results_cnn[mut]
        sae = de_results_sae[mut]
        pcts_cnn.append(100 * cnn["n_sig"] / max(cnn["n_total"], 1))
        pcts_sae.append(100 * sae["n_sig"] / max(sae["n_total"], 1))

    bars_cnn = ax.bar(x - width/2, pcts_cnn, width, label="CNN GAP Channels",
                      color="#5B9BD5", alpha=0.85, edgecolor="white")
    bars_sae = ax.bar(x + width/2, pcts_sae, width, label="SAE Neurons",
                      color="#ED7D31", alpha=0.85, edgecolor="white")

    # Value labels
    for bars, pcts in [(bars_cnn, pcts_cnn), (bars_sae, pcts_sae)]:
        for bar, pct in zip(bars, pcts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{pct:.1f}%", ha="center", va="bottom", fontsize=10,
                    fontweight="bold")

    # Count labels
    for i, mut in enumerate(mutations):
        cnn = de_results_cnn[mut]
        sae = de_results_sae[mut]
        ax.text(x[i] - width/2, -2.5,
                f"{cnn['n_sig']}/{cnn['n_total']}", ha="center",
                fontsize=8, color="#5B9BD5")
        ax.text(x[i] + width/2, -2.5,
                f"{sae['n_sig']}/{sae['n_total']}", ha="center",
                fontsize=8, color="#ED7D31")

    ax.set_ylabel("% Differentially Expressed Features", fontsize=12)
    ax.set_title("Class-Specific Feature Discovery:\nCNN GAP Channels vs SAE Neurons",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m} vs Control" for m in mutations], fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.15, axis="y")
    ax.set_ylim(bottom=-5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    logger.info(f"  Saved: {output_path}")


# ==============================================================================
# Load features
# ==============================================================================
def load_features(cache_path, apply_l2_norm=False, dead_threshold=1e-5):
    """Load CNN GAP or SAE cache, return (X, superclasses, feature_type)."""
    data = np.load(cache_path, allow_pickle=True)
    keys = list(data.keys())

    if "X_gap" in data:
        X = data["X_gap"]
        lines = data["lines"].astype(str) if data["lines"].dtype.kind != 'U' else data["lines"]
        feature_type = "Channel"
        logger.info(f"  CNN GAP cache: {X.shape}")
    elif "X_all" in data:
        X = data["X_all"]
        lines = data["lines"] if "lines" in data else data["y"]
        if lines.dtype.kind != 'U':
            lines = lines.astype(str)
        # Remove dead neurons
        if "usage_ema" in data:
            usage = data["usage_ema"]
            alive = usage > dead_threshold
            X = X[:, alive]
            logger.info(f"  SAE cache: {data['X_all'].shape} → alive: {X.shape}")
        feature_type = "Neuron"
    else:
        raise ValueError(f"Unknown cache format. Keys: {keys}")

    superclasses = [SUPERCLASS_MAP.get(ln, ln) for ln in lines]

    if apply_l2_norm:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-12, norms)
        X = X / norms
        logger.info(f"  Applied L2 normalization")

    return X, superclasses, feature_type


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Volcano plot: CNN GAP channels vs SAE neurons (DE analysis)"
    )
    parser.add_argument("--cnn_gap_cache", type=str, required=True,
                        help="Path to CNN GAP .npz cache")
    parser.add_argument("--sae_cache", type=str, required=True,
                        help="Path to SAE features .npz cache")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gap_l2_norm", action="store_true",
                        help="L2 normalize CNN GAP features before DE")
    parser.add_argument("--adj_p", type=float, default=0.05,
                        help="Adjusted p-value threshold for DE")
    parser.add_argument("--min_log2fc", type=float, default=1.5,
                        help="Minimum absolute log2 fold change for DE")
    parser.add_argument("--max_neg_log10p", type=float, default=50,
                        help="Cap -log10(p) at this value for display (0=no cap, default=50)")
    parser.add_argument("--dead_threshold", type=float, default=5e-5,
                        help="Usage EMA threshold for dead SAE neurons")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    mutations = ["SNCA", "GBA", "LRRK2"]

    # ── Load ──
    logger.info("=" * 60)
    logger.info("Loading CNN GAP features")
    X_cnn, sc_cnn, _ = load_features(
        args.cnn_gap_cache, apply_l2_norm=args.gap_l2_norm)

    logger.info("Loading SAE features")
    X_sae, sc_sae, _ = load_features(
        args.sae_cache, dead_threshold=args.dead_threshold)

    logger.info(f"  CNN GAP: {X_cnn.shape[1]} channels")
    logger.info(f"  SAE:     {X_sae.shape[1]} neurons (alive)")

    # ── DE analysis per mutation ──
    de_cnn = {}
    de_sae = {}

    for mut in mutations:
        logger.info(f"\n{'='*60}")
        logger.info(f"DE Analysis: {mut} vs Control")
        logger.info("=" * 60)

        # CNN GAP
        logger.info(f"  CNN GAP ({X_cnn.shape[1]} channels):")
        de_cnn[mut] = compute_de(
            X_cnn, sc_cnn, mut,
            adj_p_threshold=args.adj_p, min_log2fc=args.min_log2fc)
        c = de_cnn[mut]
        logger.info(f"    DE channels: {c['n_sig']}/{c['n_total']} "
                     f"({100*c['n_sig']/c['n_total']:.1f}%) "
                     f"[Up:{c['n_up']}, Down:{c['n_down']}]")

        # SAE
        logger.info(f"  SAE ({X_sae.shape[1]} neurons):")
        de_sae[mut] = compute_de(
            X_sae, sc_sae, mut,
            adj_p_threshold=args.adj_p, min_log2fc=args.min_log2fc)
        s = de_sae[mut]
        logger.info(f"    DE neurons:  {s['n_sig']}/{s['n_total']} "
                     f"({100*s['n_sig']/s['n_total']:.1f}%) "
                     f"[Up:{s['n_up']}, Down:{s['n_down']}]")

        # ── Individual volcano plots ──
        plot_volcano(
            de_cnn[mut],
            f"CNN GAP — {mut} vs Control",
            "Channel",
            os.path.join(args.output_dir, f"volcano_cnn_{mut}.png"),
            adj_p_threshold=args.adj_p,
            min_log2fc=args.min_log2fc,
            max_neg_log10p=args.max_neg_log10p,
            dpi=args.dpi,
        )
        plot_volcano(
            de_sae[mut],
            f"SAE — {mut} vs Control",
            "Neuron",
            os.path.join(args.output_dir, f"volcano_sae_{mut}.png"),
            adj_p_threshold=args.adj_p,
            min_log2fc=args.min_log2fc,
            max_neg_log10p=args.max_neg_log10p,
            dpi=args.dpi,
        )

        # ── Side-by-side comparison ──
        plot_volcano_comparison(
            de_cnn[mut], de_sae[mut], mut,
            os.path.join(args.output_dir, f"volcano_comparison_{mut}.png"),
            adj_p_threshold=args.adj_p,
            min_log2fc=args.min_log2fc,
            max_neg_log10p=args.max_neg_log10p,
            dpi=args.dpi,
        )

    # ── Summary bar chart ──
    plot_de_summary_bar(
        de_cnn, de_sae, mutations,
        os.path.join(args.output_dir, "de_summary_bar.png"),
        dpi=args.dpi,
    )

    # ── Print summary ──
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY: CNN GAP Channels vs SAE Neurons")
    logger.info("=" * 60)
    logger.info(f"  {'':15s} {'CNN Channels':>20s} {'SAE Neurons':>20s}")
    logger.info(f"  {'':15s} {'('+str(X_cnn.shape[1])+' total)':>20s} "
                f"{'('+str(X_sae.shape[1])+' total)':>20s}")
    logger.info("  " + "-" * 56)
    for mut in mutations:
        c = de_cnn[mut]
        s = de_sae[mut]
        c_pct = 100 * c["n_sig"] / c["n_total"]
        s_pct = 100 * s["n_sig"] / s["n_total"]
        logger.info(f"  {mut+' vs Ctrl':15s} "
                     f"{c['n_sig']:>4d} ({c_pct:5.1f}%)       "
                     f"{s['n_sig']:>4d} ({s_pct:5.1f}%)")

    logger.info(f"\n  Output: {args.output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
