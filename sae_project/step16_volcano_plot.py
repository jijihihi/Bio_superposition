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

## SAE는 step14의 select_concepts_by_gap_csv_de와 동일한 방식으로 DE 필터
## CNN GAP은 per-image class mean log2FC 기반
## 둘 다 mut_only, dedup, gini 필터 적용 (step14 일관성)

# import sys
# sys.argv = [
#     "step16_volcano_plot",
#     "--cnn_gap_cache", "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/MoCo_seed87/CNN_GAP/cnn_gap_stage5_out_all.npz",
#     "--sae_cache", "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/MoCo_seed87/SAE_sparsity3200_loss_L2norm곱해줌/gated_sae_stage5_out_d4096_sp3200.0_aux0.03125_tied_class_gap_means_per_image.npz",
#     "--output_dir", "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/volcano_plots",
#     "--min_log2fc", "0.58",
#     "--adj_p", "1e-40",
#     "--min_log2fc", "0.58",
#     "--max_gini", "1.0",
#     "--mut_only",
#     "--dead_threshold", "5e-5"
# ]
# from sae_project.step16_volcano_plot import main
# main()

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

# ── step14에서 import (일관성 유지) ──
from concept_visulaize.step14_visualize_concept_activations import (
    load_gap_csv,
    compute_gini_impurity,
    select_concepts_by_gap_csv_de,
)

logger = get_logger("volcano_plot")


# ==============================================================================
# CNN GAP DE: per-image class-mean log2FC
# ==============================================================================
def compute_de_cnn(X, superclasses, mutation, min_log2fc=0.0):
    """
    CNN GAP: per-image 값으로 class mean log2FC 계산.
    Wilcoxon은 volcano y축 시각화용.
    """
    sc_arr = np.array(superclasses)
    ctrl_mask = sc_arr == "Control"
    mut_mask = sc_arr == mutation

    X_ctrl = X[ctrl_mask]
    X_mut = X[mut_mask]
    d = X.shape[1]
    eps = 1e-10

    ctrl_means = X_ctrl.mean(axis=0)
    mut_means = X_mut.mean(axis=0)
    log2fc = np.log2((mut_means + eps) / (ctrl_means + eps))

    # Wilcoxon (volcano y축 시각화용)
    pvals = np.ones(d)
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

    _, adj_p, _, _ = multipletests(pvals, method="fdr_bh")

    return {
        "adj_pvalues": adj_p,
        "log2fc": log2fc,
        "n_total": d,
    }


def select_cnn_de_like_step14(X, superclasses, min_log2fc, max_gini, mut_only):
    """
    CNN GAP에 step14와 동일한 로직 적용:
    - class mean log2FC
    - Gini filter
    - mut_only
    - dedup

    Returns (deduped_list, per_mutation_de_results)
    """
    sc_arr = np.array(superclasses)
    d = X.shape[1]
    eps = 1e-10
    class_names = ["Control", "SNCA", "GBA", "LRRK2"]
    mutations = ["SNCA", "GBA", "LRRK2"]

    # Class means per channel
    class_means = {}
    for cn in class_names:
        mask = sc_arr == cn
        if mask.sum() > 0:
            class_means[cn] = X[mask].mean(axis=0)
        else:
            class_means[cn] = np.zeros(d)

    # Build gap_info-like dict for CNN channels
    cnn_gap_info = {}
    for ch in range(d):
        vals = {cn: float(class_means[cn][ch]) for cn in class_names}
        cnn_gap_info[ch] = {
            "is_alive": True,  # CNN channels are always alive
            **vals,
        }

    # Use step14's function directly
    selected = select_concepts_by_gap_csv_de(
        cnn_gap_info, max_gini=max_gini, de_min_log2fc=min_log2fc
    )

    # mut_only filter
    if mut_only:
        _MUTS = {"SNCA", "GBA", "LRRK2"}
        filtered = []
        for cid, label, fc, direction in selected:
            parts = set(label.split("_"))
            muts = parts & _MUTS
            if len(muts) == 0:
                continue
            new_label = "_".join(sorted(muts))
            filtered.append((cid, new_label, fc, direction))
        n_dropped = len(selected) - len(filtered)
        logger.info(f"    mut_only: {n_dropped} Control-only dropped, "
                    f"{len(filtered)} remaining")
        selected = filtered

    # Per-mutation DE results for volcano plots
    per_mut_de = {}
    for mut in mutations:
        de = compute_de_cnn(X, superclasses, mut, min_log2fc=min_log2fc)
        # Mark mut_only significant channels
        sig = np.abs(de["log2fc"]) >= min_log2fc
        if mut_only:
            sig = sig & (de["log2fc"] > 0)  # mut_high only
        de["mask"] = sig
        de["n_sig"] = int(sig.sum())
        de["n_up"] = int(np.sum(sig & (de["log2fc"] > 0)))
        de["n_down"] = int(np.sum(sig & (de["log2fc"] < 0)))
        per_mut_de[mut] = de

    return selected, per_mut_de


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
# Summary bar chart: absolute count of class-specific features
# ==============================================================================
def plot_de_summary_bar(de_results_cnn, de_results_sae, mutations,
                        output_path, dpi=300):
    """Bar chart comparing COUNT of DE features for CNN vs SAE across mutations.
    Absolute count matters more than % — more class-specific features = more interpretable."""
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(mutations))
    width = 0.35

    counts_cnn = []
    counts_sae = []
    for mut in mutations:
        counts_cnn.append(de_results_cnn[mut]["n_sig"])
        counts_sae.append(de_results_sae[mut]["n_sig"])

    bars_cnn = ax.bar(x - width/2, counts_cnn, width,
                      label=f"CNN GAP ({de_results_cnn[mutations[0]]['n_total']} channels)",
                      color="#5B9BD5", alpha=0.85, edgecolor="white")
    bars_sae = ax.bar(x + width/2, counts_sae, width,
                      label=f"SAE ({de_results_sae[mutations[0]]['n_total']} alive neurons)",
                      color="#ED7D31", alpha=0.85, edgecolor="white")

    ax.set_ylabel("Number of Class-Specific Features (DE)", fontsize=12)
    ax.set_title("Class-Specific Feature Discovery:\nCNN GAP Channels vs SAE Neurons",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m} vs Control" for m in mutations], fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.15, axis="y")
    ymax = max(counts_cnn + counts_sae) * 1.15
    ax.set_ylim(bottom=0, top=ymax)

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
        description="Volcano plot: CNN GAP channels vs SAE neurons (step14-consistent DE)"
    )
    parser.add_argument("--cnn_gap_cache", type=str, required=True,
                        help="Path to CNN GAP .npz cache (per-image values)")
    parser.add_argument("--sae_cache", type=str, required=True,
                        help="Path to SAE per-image .npz cache (with usage_ema)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gap_l2_norm", action="store_true",
                        help="L2 normalize CNN GAP features before DE")
    parser.add_argument("--min_log2fc", type=float, default=0.58,
                        help="Min |log2FC| for DE (step14 default=0.58)")
    parser.add_argument("--max_gini", type=float, default=0.75,
                        help="Max Gini impurity for class-specificity")
    parser.add_argument("--mut_only", action="store_true",
                        help="Only keep mutation-high features (drop Control-only)")
    parser.add_argument("--dead_threshold", type=float, default=5e-4,
                        help="usage_ema threshold for dead SAE neurons (default=5e-4, same as step09)")
    parser.add_argument("--adj_p", type=float, default=1e-10,
                        help="Adjusted p-value threshold line for CNN volcano plot")
    parser.add_argument("--max_neg_log10p", type=float, default=50,
                        help="Cap -log10(p) for CNN volcano display")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    mutations = ["SNCA", "GBA", "LRRK2"]
    class_names = ["Control", "SNCA", "GBA", "LRRK2"]

    # ==========================================================================
    # 1) SAE: per-image npz → usage_ema 필터 → class mean log2FC (step14 동일)
    # ==========================================================================
    logger.info("=" * 60)
    logger.info("SAE: Loading per-image cache with usage_ema")
    sae_data = np.load(args.sae_cache, allow_pickle=True)
    X_sae = sae_data["X_all"]
    sae_lines = sae_data["lines"] if "lines" in sae_data else sae_data["y"]
    if sae_lines.dtype.kind != 'U':
        sae_lines = sae_lines.astype(str)
    sc_sae = [SUPERCLASS_MAP.get(ln, ln) for ln in sae_lines]

    # usage_ema 기반 dead neuron 필터
    n_total_sae = X_sae.shape[1]
    if "usage_ema" in sae_data:
        usage_ema = sae_data["usage_ema"]
        alive_mask = usage_ema > args.dead_threshold
        alive_indices = np.where(alive_mask)[0]
        X_sae_alive = X_sae[:, alive_mask]
        n_alive_sae = int(alive_mask.sum())
        logger.info(f"  SAE: {n_total_sae} total, {n_alive_sae} alive "
                    f"(usage_ema > {args.dead_threshold})")
    else:
        alive_indices = np.arange(n_total_sae)
        X_sae_alive = X_sae
        n_alive_sae = n_total_sae
        logger.info(f"  SAE: {n_total_sae} neurons (no usage_ema found)")

    # Per-class mean 계산 → gap_info dict 구성 (select_concepts_by_gap_csv_de에 전달)
    sc_arr = np.array(sc_sae)
    sae_gap_info = {}
    for i, orig_idx in enumerate(alive_indices):
        vals = {}
        for cn in class_names:
            mask = sc_arr == cn
            if mask.sum() > 0:
                vals[cn] = float(X_sae_alive[mask, i].mean())
            else:
                vals[cn] = 0.0
        sae_gap_info[int(orig_idx)] = {
            "is_alive": True,
            **vals,
        }

    # step14 select_concepts_by_gap_csv_de 동일 로직
    sae_selected = select_concepts_by_gap_csv_de(
        sae_gap_info, max_gini=args.max_gini, de_min_log2fc=args.min_log2fc
    )
    logger.info(f"  Before mut_only: {len(sae_selected)} concepts")

    # mut_only filter
    if args.mut_only:
        _MUTS = {"SNCA", "GBA", "LRRK2"}
        filtered = []
        for cid, label, fc, direction in sae_selected:
            parts = set(label.split("_"))
            muts = parts & _MUTS
            if len(muts) == 0:
                continue
            new_label = "_".join(sorted(muts))
            filtered.append((cid, new_label, fc, direction))
        n_dropped = len(sae_selected) - len(filtered)
        sae_selected = filtered
        logger.info(f"  mut_only: {n_dropped} Control-only dropped, "
                    f"{len(sae_selected)} remaining")

    logger.info(f"  SAE: {n_alive_sae} alive → {len(sae_selected)} DE concepts (deduped)")

    # Count per-mutation for SAE
    sae_per_mut = {mut: 0 for mut in mutations}
    for cid, label, fc, direction in sae_selected:
        for mut in mutations:
            if mut in label:
                sae_per_mut[mut] += 1

    # ==========================================================================
    # 2) CNN GAP: step14-style DE filter
    # ==========================================================================
    logger.info(f"\n{'='*60}")
    logger.info("CNN GAP: Loading cache and running step14-style DE filter")
    X_cnn, sc_cnn, _ = load_features(
        args.cnn_gap_cache, apply_l2_norm=args.gap_l2_norm)
    logger.info(f"  CNN GAP: {X_cnn.shape[1]} channels, {X_cnn.shape[0]} images")

    cnn_selected, cnn_per_mut_de = select_cnn_de_like_step14(
        X_cnn, sc_cnn, args.min_log2fc, args.max_gini, args.mut_only
    )
    logger.info(f"  CNN: {X_cnn.shape[1]} channels → {len(cnn_selected)} DE channels (deduped)")

    # Count per-mutation for CNN
    cnn_per_mut = {mut: 0 for mut in mutations}
    for cid, label, fc, direction in cnn_selected:
        for mut in mutations:
            if mut in label:
                cnn_per_mut[mut] += 1

    # ==========================================================================
    # 3) Volcano plots (CNN + SAE — both have per-image Wilcoxon)
    # ==========================================================================
    # CNN volcano
    for mut in mutations:
        if mut in cnn_per_mut_de:
            plot_volcano(
                cnn_per_mut_de[mut],
                f"CNN GAP — {mut} vs Control",
                "Channel",
                os.path.join(args.output_dir, f"volcano_cnn_{mut}.svg"),
                adj_p_threshold=args.adj_p,
                min_log2fc=args.min_log2fc,
                max_neg_log10p=args.max_neg_log10p,
                dpi=args.dpi,
            )

    # SAE volcano (per-image Wilcoxon on alive neurons)
    for mut in mutations:
        de_sae = compute_de_cnn(X_sae_alive, sc_sae, mut, min_log2fc=args.min_log2fc)
        sig = np.abs(de_sae["log2fc"]) >= args.min_log2fc
        if args.mut_only:
            sig = sig & (de_sae["log2fc"] > 0)
        de_sae["mask"] = sig
        de_sae["n_sig"] = int(sig.sum())
        de_sae["n_up"] = int(np.sum(sig & (de_sae["log2fc"] > 0)))
        de_sae["n_down"] = int(np.sum(sig & (de_sae["log2fc"] < 0)))
        plot_volcano(
            de_sae,
            f"SAE — {mut} vs Control",
            "Neuron",
            os.path.join(args.output_dir, f"volcano_sae_{mut}.svg"),
            adj_p_threshold=args.adj_p,
            min_log2fc=args.min_log2fc,
            max_neg_log10p=args.max_neg_log10p,
            dpi=args.dpi,
        )

    # ==========================================================================
    # 4) Summary bar chart: deduped counts
    # ==========================================================================
    de_cnn_summary = {}
    de_sae_summary = {}
    for mut in mutations:
        de_cnn_summary[mut] = {
            "n_sig": cnn_per_mut[mut],
            "n_total": X_cnn.shape[1],
            "n_up": cnn_per_mut[mut],
            "n_down": 0,
        }
        de_sae_summary[mut] = {
            "n_sig": sae_per_mut[mut],
            "n_total": n_alive_sae,
            "n_up": sae_per_mut[mut],
            "n_down": 0,
        }

    plot_de_summary_bar(
        de_cnn_summary, de_sae_summary, mutations,
        os.path.join(args.output_dir, "de_summary_bar.svg"),
        dpi=args.dpi,
    )

    # ==========================================================================
    # 5) Print summary
    # ==========================================================================
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY: CNN GAP Channels vs SAE Neurons")
    logger.info(f"  Method: per-image class-mean log2FC (step14 consistent)")
    logger.info(f"  Method: per-image class-mean log2FC (step14 consistent)")
    logger.info(f"  min_log2fc={args.min_log2fc}, max_gini={args.max_gini}, "
                f"mut_only={args.mut_only}, dead_threshold={args.dead_threshold}")
    logger.info("=" * 60)
    logger.info(f"  {'':15s} {'CNN (dedup)':>15s} {'SAE (dedup)':>15s}")
    logger.info(f"  {'':15s} {'('+str(X_cnn.shape[1])+' ch)':>15s} "
                f"{'('+str(n_alive_sae)+' alive)':>15s}")
    logger.info("  " + "-" * 46)

    for mut in mutations:
        logger.info(f"  {mut+' vs Ctrl':15s} "
                     f"{cnn_per_mut[mut]:>6d}         "
                     f"{sae_per_mut[mut]:>6d}")

    logger.info(f"  {'TOTAL (dedup)':15s} "
                 f"{len(cnn_selected):>6d}         "
                 f"{len(sae_selected):>6d}")

    logger.info(f"\n  Output: {args.output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
