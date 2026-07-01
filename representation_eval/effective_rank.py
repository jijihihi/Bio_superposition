# ==============================================================================

#


#

#   1) raw (no filter, no norm, no PCA)
#   2) raw → PCA
#   3) raw → std → PCA
#   4) CV+DE filter
#   5) CV+DE filter → PCA
#   6) CV+DE filter → std → PCA
#
# Usage (CNN only):
# !python -m cell_death.effective_rank \
#     --cnn_cache "..." \
#     --dead_threshold 5e-5 --gap_l2_norm \
#     --pca_dim 50 \
#     --filter_mode cv de --min_cv 0.1 --de_min_log2fc 1.0 \
#     --de_mode union --de_eval_split 0.5 \
#     --seed 856 --output_dir "/content/erank"
#
# Usage (SAE only):
# !python -m cell_death.effective_rank \
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

from trajectory_inference_pipeline.trajectory_utils import (apply_normalization,
                                                         compute_cv_per_neuron,
                                                         compute_de_neurons,
                                                         load_features_cache)
from run_CNN.logging_utils import SUPERCLASS_MAP, get_logger

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

    all_results = []

    for source_label, (X_raw, superclasses_arr) in sources.items():
        # Only process raw data (unfiltered)
        filter_tag = "unfiltered"
        X_base = X_raw

        for mut in mutations:
            logger.info(f"\n  ── {source_label} / {mut} ──")
            mut_mask = superclasses_arr == mut
            n_mut = int(mut_mask.sum())
            if n_mut < 10:
                logger.warning(f"    Too few samples ({n_mut}), skipping")
                continue

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
    logger.info(
        f"  {'Source':6s} {'Mut':6s} {'Condition':10s} "

        f"{'erank':>8s} {'dims':>6s} {'ratio':>7s}"
    )
    logger.info("  " + "-" * 65)

    for res in all_results:
        logger.info(
            f"  {res['source']:6s} {res['mutation']:6s} "
            f"{res['condition']:10s} "
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
