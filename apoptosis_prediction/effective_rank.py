# ==============================================================================
# Effective Rank Analysis — CNN vs SAE 정보 풍부도 비교
#
# SVD 기반 effective rank = exp(H(σ/Σσ)) 로 feature matrix의
# 실질적 차원 수(정보 풍부도)를 측정.
#
# 6가지 조건에서 erank 계산:
#   1) raw (no filter, no norm, no PCA)
#   2) raw → PCA
#   3) raw → std → PCA
#   4) CV+DE filter
#   5) CV+DE filter → PCA
#   6) CV+DE filter → std → PCA
#
# Usage (CNN only):
# !python -m cell_death_prediction.effective_rank \
#     --cnn_cache "..." \
#     --dead_threshold 5e-5 --gap_l2_norm \
#     --pca_dim 50 \
#     --filter_mode cv de --min_cv 0.1 --de_min_log2fc 1.0 \
#     --de_mode union --de_eval_split 0.5 \
#     --seed 856 --output_dir "/content/erank"
#
# Usage (SAE only):
# !python -m cell_death_prediction.effective_rank \
#     --sae_cache "..." \
#     --dead_threshold 5e-5 \
#     --pca_dim 50 \
#     --filter_mode cv de --min_cv 0.1 --de_min_log2fc 1.0 \
#     --de_mode union --de_eval_split 0.5 \
#     --seed 856 --output_dir "/content/erank"
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

from kendall_correlation_coefficient.dpt import (apply_normalization,
                                                         compute_cv_per_neuron,
                                                         compute_de_neurons,
                                                         load_features_cache)
from model_train.logging_utils import SUPERCLASS_MAP, get_logger

logger = get_logger("effective_rank")

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
sns.set_style("ticks")


# ==============================================================================
# Argument Parser
# ==============================================================================
def get_args():
    p = argparse.ArgumentParser(
        description="Effective rank analysis — SVD-based information richness"
    )
    p.add_argument("--cnn_cache", type=str, default="")
    p.add_argument("--sae_cache", type=str, default="")
    p.add_argument("--dead_threshold", type=float, default=1e-5)
    p.add_argument("--gap_l2_norm", action="store_true")
    p.add_argument("--pre_l2_norm", action="store_true")
    p.add_argument("--divide_hw", type=int, default=0)

    # Neuron filtering
    p.add_argument("--filter_mode", type=str, nargs="+", default=["none"])
    p.add_argument("--min_cv", type=float, default=0.0)
    p.add_argument("--de_adj_p", type=float, default=0.05)
    p.add_argument("--de_min_log2fc", type=float, default=1.0)
    p.add_argument("--de_top_k", type=int, default=0)
    p.add_argument("--de_mode", type=str, default="union", choices=["union", "per_mut"])
    p.add_argument("--de_eval_split", type=float, default=0.5)

    p.add_argument(
        "--pca_dim",
        type=int,
        default=50,
        help="PCA dimensions for PCA-based erank. 0 = skip PCA conditions.",
    )
    p.add_argument(
        "--norm",
        type=str,
        default="",
        choices=["", "none", "std", "log_std"],
        help="Normalization before PCA: '' or 'none' (skip), 'std', 'log_std'",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--samples_per_class", type=int, default=0)

    return p.parse_args()


# ==============================================================================
# Effective rank
# ==============================================================================
def compute_effective_rank(X):
    """
    Effective rank = exp(H(p)) where p_i = σ_i / Σσ_j.
    H(p) = -Σ p_i log(p_i) is Shannon entropy of normalized singular values.

    Returns:
        erank: float — effective rank
        svs: np.ndarray — singular values (descending)
        cumvar: np.ndarray — cumulative variance ratio (len = len(svs))
    """
    X_centered = X - X.mean(axis=0)
    s = np.linalg.svd(X_centered, compute_uv=False)
    s = s[s > 1e-12]
    if len(s) == 0:
        return 0.0, np.array([]), np.array([])
    p = s / s.sum()
    entropy = -np.sum(p * np.log(p))
    # Cumulative variance ratio: σ_i² / Σσ_j²
    var_explained = s**2
    cumvar = np.cumsum(var_explained) / var_explained.sum()
    return float(np.exp(entropy)), s, cumvar


# ==============================================================================
# Plot: singular value spectrum
# ==============================================================================
def plot_sv_spectrum(sv_dict, mutation, condition_label, output_path, dpi=200):
    """Plot normalized singular value spectrum for CNN and/or SAE."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"CNN": "#4C72B0", "SAE": "#DD8452"}

    for source_label, (erank, svs) in sv_dict.items():
        if len(svs) == 0:
            continue
        p = svs / svs.sum()
        ax.plot(
            range(1, len(p) + 1),
            p,
            "-",
            color=colors.get(source_label, "gray"),
            linewidth=1.5,
            alpha=0.8,
            label=f"{source_label} (erank={erank:.1f}/{len(svs)})",
        )

    ax.set_xlabel("Singular value index", fontsize=11)
    ax.set_ylabel("Normalized σ_i / Σσ", fontsize=11)
    ax.set_title(
        f"{mutation} — SV Spectrum ({condition_label})", fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=9)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(output_path.replace(".png", ".svg"), bbox_inches="tight")
    if _IN_COLAB:
        plt.show()
    plt.close(fig)


# ==============================================================================
# Apply filters to X (shared logic)
# ==============================================================================
def apply_filters(X, superclasses_arr, args):
    """Apply CV + DE filters exactly like dpt_kendall.py. Returns filtered X."""
    has_de = "de" in args.filter_mode
    de_mode = getattr(args, "de_mode", "union")
    filter_steps = []

    for fm in args.filter_mode:
        if fm in ("none", "de"):
            continue
        n_before = X.shape[1]
        if fm == "cv":
            cv = compute_cv_per_neuron(X, list(superclasses_arr))
            X = X[:, cv >= args.min_cv]
            step = f"cv≥{args.min_cv}: {n_before}→{X.shape[1]}"
        else:
            continue
        filter_steps.append(step)
        logger.info(f"    Filter [{fm}]: {step}")

    # DE/Eval split
    de_eval_split = getattr(args, "de_eval_split", 0.0)
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
        logger.info(
            f"    DE/Eval split: DE={int(de_mask_global.sum())}, "
            f"Eval={int(eval_mask.sum())}"
        )
    else:
        de_mask_global = np.ones(len(superclasses_arr), dtype=bool)

    # DE union
    if has_de and de_mode == "union":
        de_masks = []
        X_de = X[de_mask_global]
        sc_de = list(superclasses_arr[de_mask_global])
        for m in ["SNCA", "GBA", "LRRK2"]:
            de_result = compute_de_neurons(
                X_de,
                sc_de,
                m,
                adj_p_threshold=args.de_adj_p,
                min_log2fc=args.de_min_log2fc,
            )
            mask = de_result["mask"]
            if args.de_top_k > 0 and mask.sum() > args.de_top_k:
                sig_indices = np.where(mask)[0]
                abs_fc = np.abs(de_result["log2fc"][sig_indices])
                top_k_idx = sig_indices[np.argsort(abs_fc)[::-1][: args.de_top_k]]
                mask = np.zeros_like(mask)
                mask[top_k_idx] = True
            de_masks.append(mask)

        superclasses_allm = [("AllMut" if s != "Control" else "Control") for s in sc_de]
        de_ctrl = compute_de_neurons(
            X_de,
            superclasses_allm,
            "AllMut",
            adj_p_threshold=args.de_adj_p,
            min_log2fc=args.de_min_log2fc,
        )
        ctrl_high_mask = de_ctrl["mask"] & (de_ctrl["log2fc"] < 0)
        de_masks.append(ctrl_high_mask)

        union_mask = de_masks[0] | de_masks[1] | de_masks[2] | de_masks[3]
        n_before_de = X.shape[1]
        X = X[:, union_mask]
        de_step = f"DE_union+CtrlHigh: {n_before_de}→{X.shape[1]}"
        filter_steps.append(de_step)
        logger.info(f"    {de_step}")

    filter_label = " → ".join(filter_steps) if filter_steps else "none"
    return X, filter_label


# ==============================================================================
# Compute erank under a specific condition
# ==============================================================================
def compute_erank_condition(X_mut, condition_name, pca_dim, seed, norm_type=""):
    """
    Compute erank for a specific processing condition.
    condition_name: 'raw', 'pca', 'norm_pca'
    norm_type: '' (none), 'std', 'log_std'

    Returns: (erank, svs, n_dims, cumvar)
    """
    if condition_name == "raw":
        erank, svs, cumvar = compute_effective_rank(X_mut)
        return erank, svs, X_mut.shape[1], cumvar

    elif condition_name == "pca":
        if pca_dim <= 0 or X_mut.shape[1] <= pca_dim:
            erank, svs, cumvar = compute_effective_rank(X_mut)
            return erank, svs, X_mut.shape[1], cumvar
        n_pca = min(pca_dim, X_mut.shape[1], X_mut.shape[0] - 1)
        pca = PCA(n_components=n_pca, random_state=seed)
        X_pca = pca.fit_transform(X_mut)
        erank, svs, cumvar = compute_effective_rank(X_pca)
        return erank, svs, n_pca, cumvar

    elif condition_name == "norm_pca":
        nt = norm_type if norm_type and norm_type != "none" else "std"
        X_normed = apply_normalization(X_mut, nt)
        if pca_dim <= 0 or X_normed.shape[1] <= pca_dim:
            erank, svs, cumvar = compute_effective_rank(X_normed)
            return erank, svs, X_normed.shape[1], cumvar
        n_pca = min(pca_dim, X_normed.shape[1], X_normed.shape[0] - 1)
        pca = PCA(n_components=n_pca, random_state=seed)
        X_pca = pca.fit_transform(X_normed)
        erank, svs, cumvar = compute_effective_rank(X_pca)
        return erank, svs, n_pca, cumvar

    else:
        raise ValueError(f"Unknown condition: {condition_name}")


# ==============================================================================
# Main
# ==============================================================================
def main():
    args = get_args()
    np.random.seed(args.seed)

    if not args.cnn_cache and not args.sae_cache:
        raise ValueError("At least one of --cnn_cache or --sae_cache required")

    if args.output_dir:
        out_dir = args.output_dir
    else:
        ref = args.cnn_cache or args.sae_cache
        out_dir = os.path.join(os.path.dirname(ref), "effective_rank")
    os.makedirs(out_dir, exist_ok=True)

    # ── Load feature caches ──
    sources = {}

    def _load_and_preprocess(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        if "X_gap" in data:
            X = data["X_gap"]
            lines = (
                data["lines"].astype(str)
                if data["lines"].dtype.kind != "U"
                else data["lines"]
            )
            label = "CNN"
            logger.info(f"  Detected CNN GAP cache: {X.shape}")
        elif "X_all" in data:
            X, _, lines, _, _, _ = load_features_cache(cache_path, args.dead_threshold)
            label = "SAE"
        else:
            raise ValueError(f"Unknown cache format. Keys: {list(data.keys())}")

        if args.pre_l2_norm:
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1e-12, norms)
            X = X / norms

        if args.divide_hw > 0:
            X = X / args.divide_hw

        if args.gap_l2_norm:
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1e-12, norms)
            X = X / norms

        superclasses = np.array([SUPERCLASS_MAP.get(ln, ln) for ln in lines])
        return X, superclasses, label

    if args.cnn_cache:
        logger.info("Loading CNN cache...")
        X_cnn, sc_cnn, _ = _load_and_preprocess(args.cnn_cache)
        sources["CNN"] = (X_cnn, sc_cnn)
        logger.info(f"  CNN features: {X_cnn.shape}")

    if args.sae_cache:
        logger.info("Loading SAE cache...")
        X_sae, sc_sae, _ = _load_and_preprocess(args.sae_cache)
        sources["SAE"] = (X_sae, sc_sae)
        logger.info(f"  SAE features: {X_sae.shape}")

    mutations = ["SNCA", "GBA", "LRRK2"]

    # Determine which conditions to compute based on args
    norm_set = args.norm and args.norm != "none"
    if args.pca_dim <= 0 and not norm_set:
        conditions = ["raw"]
    elif args.pca_dim > 0 and not norm_set:
        conditions = ["raw", "pca"]
    elif args.pca_dim > 0 and norm_set:
        conditions = ["raw", "pca", "norm_pca"]
    else:
        # pca_dim=0 + norm set → norm only (no PCA)
        conditions = ["raw", "norm_pca"]
    logger.info(f"  Conditions to compute: {conditions}")

    has_filter = not (args.filter_mode == ["none"])

    # Result storage: list of dicts for easy DataFrame conversion
    all_results = []

    for source_label, (X_raw, superclasses_arr) in sources.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"  Source: {source_label}")
        logger.info(f"{'='*60}")

        # Subsample
        spc = args.samples_per_class
        if spc > 0:
            rng_sub = np.random.RandomState(args.seed)
            keep_indices = []
            for cls in sorted(np.unique(superclasses_arr)):
                cls_idx = np.where(superclasses_arr == cls)[0]
                n_take = min(spc, len(cls_idx))
                chosen = rng_sub.choice(cls_idx, size=n_take, replace=False)
                keep_indices.extend(chosen.tolist())
            keep_indices = sorted(keep_indices)
            X_raw = X_raw[keep_indices]
            superclasses_arr = superclasses_arr[keep_indices]
            logger.info(f"  After subsampling: {X_raw.shape[0]} samples")

        X_nofilter = X_raw.copy()

        # Apply filters
        if has_filter:
            X_filtered, filter_label = apply_filters(
                X_raw.copy(), superclasses_arr, args
            )
            logger.info(f"  Filter: {filter_label}")
        else:
            X_filtered = None
            filter_label = "none"

        for mut in mutations:
            logger.info(f"\n  ── {source_label} / {mut} ──")
            mut_mask = superclasses_arr == mut
            n_mut = int(mut_mask.sum())
            if n_mut < 10:
                logger.warning(f"    Too few samples ({n_mut}), skipping")
                continue

            # Process both unfiltered and filtered
            datasets = [("unfiltered", X_nofilter)]
            if X_filtered is not None:
                datasets.append(("filtered", X_filtered))

            for filter_tag, X_base in datasets:
                X_mut = X_base[mut_mask]

                for cond in conditions:
                    cond_label = f"{filter_tag}_{cond}"
                    erank, svs, n_dims, cumvar = compute_erank_condition(
                        X_mut, cond, args.pca_dim, args.seed, norm_type=args.norm
                    )

                    logger.info(f"    {cond_label:25s}: erank={erank:8.2f} / {n_dims}")

                    all_results.append(
                        {
                            "source": source_label,
                            "mutation": mut,
                            "filter": filter_tag,
                            "condition": cond,
                            "erank": erank,
                            "n_dims": n_dims,
                            "erank_ratio": erank / max(n_dims, 1),
                            "n_samples": n_mut,
                            "cumulative_variance": (
                                cumvar.tolist() if len(cumvar) > 0 else []
                            ),
                        }
                    )

                    # Store SVs for spectrum plot
                    if cond == "raw":
                        sv_key = (source_label, mut, filter_tag)
                        # Will plot later
                        all_results[-1]["_svs"] = svs

    # ── Spectrum plots ──
    for mut in mutations:
        for filter_tag in ["unfiltered", "filtered"]:
            sv_dict = {}
            for res in all_results:
                if (
                    res["mutation"] == mut
                    and res["filter"] == filter_tag
                    and res["condition"] == "raw"
                    and "_svs" in res
                ):
                    sv_dict[res["source"]] = (res["erank"], res["_svs"])
            if sv_dict:
                plot_sv_spectrum(
                    sv_dict,
                    mut,
                    filter_tag,
                    os.path.join(out_dir, f"sv_spectrum_{mut}_{filter_tag}.png"),
                    dpi=args.dpi,
                )

    # ── Summary ──
    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY — Effective Rank")
    logger.info(f"{'='*80}")
    logger.info(f"  PCA dim: {args.pca_dim}")
    logger.info(f"  Filter: {filter_label if has_filter else 'none'}")
    logger.info(
        f"  {'Source':6s} {'Mut':6s} {'Filter':12s} {'Condition':10s} "
        f"{'erank':>8s} {'dims':>6s} {'ratio':>7s}"
    )
    logger.info("  " + "-" * 65)

    for res in all_results:
        logger.info(
            f"  {res['source']:6s} {res['mutation']:6s} "
            f"{res['filter']:12s} {res['condition']:10s} "
            f"{res['erank']:8.2f} {res['n_dims']:6d} "
            f"{res['erank_ratio']:7.3f}"
        )

    # ── Save JSON (trend-plot friendly) ──
    json_results = [
        {k: v for k, v in res.items() if k != "_svs"} for res in all_results
    ]
    json_path = os.path.join(out_dir, "effective_rank_results.json")
    with open(json_path, "w") as f:
        json.dump(
            {
                "pca_dim": args.pca_dim,
                "gap_l2_norm": args.gap_l2_norm,
                "filter_mode": args.filter_mode,
                "min_cv": args.min_cv,
                "de_min_log2fc": args.de_min_log2fc,
                "de_mode": args.de_mode,
                "de_eval_split": args.de_eval_split,
                "norm": args.norm,
                "samples_per_class": args.samples_per_class,
                "seed": args.seed,
                "results": json_results,
            },
            f,
            indent=2,
        )
    logger.info(f"\n  Saved JSON: {json_path}")
    logger.info(f"  Output: {out_dir}")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()
