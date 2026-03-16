# ==============================================================================
# DPT vs Concept Activation Analysis
#
# 각 SAE concept(피처맵)의 GAP 활성화(L2 norm)를 DPT 축에 대해 시각화.
# x축: DPT (Diffusion Pseudotime) — ctrl_mut_pair scope
# y축: concept GAP activation (per-image L2 normalized)
# 각 concept마다 Spearman ρ + GAM fit + adj.R² 표시, 개별 PNG 저장.
#
# 기존 dpt_kendall.py 함수를 최대한 import하여 일관성 유지.
# ==============================================================================

import os
import sys
import argparse
import numpy as np

from sklearn.decomposition import PCA
from scipy.stats import spearmanr

import scanpy as sc

import matplotlib
_IN_COLAB = "google.colab" in sys.modules
if not _IN_COLAB:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sae_project.step02_logging_utils import get_logger, SUPERCLASS_MAP

# ── dpt_kendall 에서 import ──
from kendall_correlation_coefficient.dpt_kendall import (
    load_features_cache,
    load_and_match_apoptosis,
    compute_cv_per_neuron,
    compute_de_neurons,
    apply_normalization,
)

logger = get_logger("dpt_concept_act")

MUTATION_COLORS = {"SNCA": "#E24A33", "GBA": "#348ABD", "LRRK2": "#988ED5"}


# ==============================================================================
# Argument Parser
# ==============================================================================
def get_args():
    p = argparse.ArgumentParser(
        description="DPT vs per-concept GAP activation (Spearman ρ + GAM adj.R²)"
    )

    p.add_argument("--features_cache", type=str, required=True,
                   help="Path to .npz cache (SAE: X_all+usage_ema)")
    p.add_argument("--apoptosis_csv", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="",
                   help="Output directory for plots (default: next to cache)")
    p.add_argument("--dead_threshold", type=float, default=5e-5)
    p.add_argument("--gap_l2_norm", action="store_true",
                   help="Apply per-image L2 normalization (양 보정)")

    # Neuron filtering (same as dpt_kendall)
    p.add_argument("--filter_mode", type=str, nargs="+", default=["none"],
                   help="Sequential filters: 'cv', 'de', 'none'. e.g. '--filter_mode cv de'")
    p.add_argument("--min_cv", type=float, default=0.0)
    p.add_argument("--de_adj_p", type=float, default=0.05)
    p.add_argument("--de_min_log2fc", type=float, default=1.0)

    # Normalization for DPT manifold
    p.add_argument("--norm", type=str, default="log_std",
                   help="Feature normalization for DPT manifold (e.g. log_std)")

    # PCA / kNN / diffmap
    p.add_argument("--pca_dim", type=int, default=15)
    p.add_argument("--n_neighbors", type=int, default=35)
    p.add_argument("--n_diffmap_comps", type=int, default=10)
    p.add_argument("--n_dcs", type=int, default=10)

    # DPT scope
    p.add_argument("--dpt_scope", type=str, default="ctrl_mut_pair",
                   choices=["ctrl_mut_pair", "global"],
                   help="'ctrl_mut_pair': Control+Mut pair별 DPT. 'global': 전체.")

    # Root selection
    p.add_argument("--root_mode", type=str, default="diffmap",
                   choices=["pca", "diffmap"])
    p.add_argument("--root_perturbation_n", type=int, default=10)

    # GAM
    p.add_argument("--gam_splines", type=int, default=8)
    p.add_argument("--gam_trim_pctl", type=float, nargs=2, default=[5, 95])

    # Misc
    p.add_argument("--seed", type=int, default=856)
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--samples_per_class", type=int, default=5000)
    p.add_argument("--concepts", type=int, nargs="*", default=None,
                   help="Specific concept indices to plot (default: all alive)")
    p.add_argument("--de_eval_split", type=float, default=0.5)

    return p.parse_args()


# ==============================================================================
# DPT computation — ctrl_mut_pair scope (from dpt_kendall logic)
# ==============================================================================
def compute_dpt_ctrl_mut_pair(X_pca, superclasses_arr, n_neighbors, n_pca,
                               n_diffmap, n_dcs, mutations=None):
    """
    Compute DPT for each Ctrl+Mut pair (ctrl_mut_pair scope).

    Returns
    -------
    dpt_dict : dict
        {mutation: dpt_array} — DPT values for ALL cells in the pair (Ctrl+Mut)
    pair_mask_dict : dict
        {mutation: boolean mask (len = N_total)} — which cells belong to this pair
    """
    if mutations is None:
        mutations = ["SNCA", "GBA", "LRRK2"]

    dpt_dict = {}
    pair_mask_dict = {}

    for mut in mutations:
        ctrl_mask = superclasses_arr == "Control"
        mut_mask = superclasses_arr == mut
        pair_mask = ctrl_mask | mut_mask

        if mut_mask.sum() < 10:
            logger.warning(f"  {mut}: too few cells ({mut_mask.sum()}), skip")
            continue

        X_pca_pair = X_pca[pair_mask]
        pair_sc = superclasses_arr[pair_mask]
        n_pair = X_pca_pair.shape[0]
        logger.info(f"  {mut} pair: {n_pair} cells "
                    f"(Ctrl={ctrl_mask.sum()}, {mut}={mut_mask.sum()})")

        # Build diffmap on pair
        adata_pair = sc.AnnData(X_pca_pair.astype(np.float32))
        adata_pair.obsm["X_pca"] = X_pca_pair.astype(np.float32)
        adata_pair.obs["superclass"] = list(pair_sc)

        n_diffmap_pair = min(n_diffmap, n_pair - 2)
        n_diffmap_pair = max(n_diffmap_pair, 2)
        n_dcs_pair = min(n_dcs, n_diffmap_pair)
        n_dcs_pair = max(n_dcs_pair, 2)

        sc.pp.neighbors(adata_pair, n_neighbors=n_neighbors,
                        n_pcs=n_pca, use_rep="X_pca")
        sc.tl.diffmap(adata_pair, n_comps=n_diffmap_pair)

        evals = adata_pair.uns["diffmap_evals"]
        logger.info(f"    Eigenvalues (top 3): {evals[:3]}")

        # Root: Control centroid in diffmap space
        diffmap_coords = adata_pair.obsm["X_diffmap"]
        pair_ctrl_mask = np.array(pair_sc) == "Control"
        ctrl_centroid = diffmap_coords[pair_ctrl_mask].mean(axis=0)
        ctrl_dists = np.linalg.norm(
            diffmap_coords[pair_ctrl_mask] - ctrl_centroid, axis=1)
        root_in_pair = np.where(pair_ctrl_mask)[0][np.argmin(ctrl_dists)]
        logger.info(f"    Root: Ctrl cell (pair idx {root_in_pair})")

        adata_pair.uns["iroot"] = int(root_in_pair)
        sc.tl.dpt(adata_pair, n_dcs=n_dcs_pair)
        dpt_pair = adata_pair.obs["dpt_pseudotime"].values

        logger.info(f"    Root DPT = {dpt_pair[root_in_pair]:.6f}")
        logger.info(f"    Ctrl mean DPT = "
                    f"{np.nanmean(dpt_pair[np.array(pair_sc) == 'Control']):.4f}, "
                    f"{mut} mean DPT = "
                    f"{np.nanmean(dpt_pair[np.array(pair_sc) == mut]):.4f}")

        dpt_dict[mut] = dpt_pair
        pair_mask_dict[mut] = pair_mask

        del adata_pair

    return dpt_dict, pair_mask_dict


# ==============================================================================
# Plot: single concept × mutation — scatter + GAM + Spearman ρ + adj.R²
# ==============================================================================
def plot_concept_vs_dpt(dpt_vals, act_vals, concept_idx, mutation,
                        output_path, dpi=200,
                        gam_splines=8, gam_trim_pctl=(5, 95)):
    """
    DPT (x) vs concept GAP activation (y) scatter + GAM fit.

    Returns
    -------
    rho : float — Spearman ρ
    adj_r2 : float — GAM adjusted R²
    """
    # Filter valid
    valid = np.isfinite(dpt_vals) & np.isfinite(act_vals)
    if valid.sum() < 20:
        return 0.0, 0.0

    dpt_v = dpt_vals[valid]
    act_v = act_vals[valid]

    rho, pval = spearmanr(dpt_v, act_v)
    rho = rho if not np.isnan(rho) else 0.0

    color = MUTATION_COLORS.get(mutation, "gray")

    fig, ax = plt.subplots(figsize=(8, 5))

    # Scatter
    ax.scatter(dpt_v, act_v, s=6, alpha=0.25, c=color,
               edgecolors="none", rasterized=True, zorder=1)

    # GAM fit
    adj_r2 = 0.0
    pct_lo, pct_hi = np.percentile(dpt_v, list(gam_trim_pctl))
    dense_mask = (dpt_v >= pct_lo) & (dpt_v <= pct_hi)
    dpt_dense = dpt_v[dense_mask]
    act_dense = act_v[dense_mask]

    x_line = np.linspace(pct_lo, pct_hi, 200)
    try:
        from pygam import LinearGAM, s as s_term
        n_sp = min(gam_splines, max(5, len(dpt_dense) // 50))
        gam = LinearGAM(s_term(0, n_splines=n_sp, spline_order=3)).fit(
            dpt_dense.reshape(-1, 1), act_dense)
        y_gam = gam.predict(x_line.reshape(-1, 1))
        ci = gam.confidence_intervals(x_line.reshape(-1, 1), width=0.95)

        ax.plot(x_line, y_gam, "-", color="black", lw=2.5,
                alpha=0.9, zorder=5, label="GAM fit")
        ax.fill_between(x_line, ci[:, 0], ci[:, 1],
                        color="black", alpha=0.12, zorder=2, label="95% CI")

        # Adjusted R²
        ss_res = np.sum((act_dense - gam.predict(dpt_dense.reshape(-1, 1))) ** 2)
        ss_tot = np.sum((act_dense - act_dense.mean()) ** 2)
        n = len(act_dense)
        p = gam.statistics_['edof']
        if ss_tot > 0 and n > p + 1:
            adj_r2 = 1 - (ss_res / (n - p - 1)) / (ss_tot / (n - 1))
        else:
            adj_r2 = 0.0
    except ImportError:
        # Fallback: linear fit
        if len(dpt_v) > 2:
            z = np.polyfit(dpt_v, act_v, 1)
            ax.plot(x_line, np.polyval(z, x_line), "--", color="black",
                    lw=2, alpha=0.7, zorder=3, label="Linear fit")

    ax.set_xlabel("DPT (Diffusion Pseudotime)", fontsize=12)
    ax.set_ylabel(f"Concept {concept_idx} activation (L2-normed GAP)", fontsize=12)
    ax.set_title(f"Concept {concept_idx} — {mutation}", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper left", fontsize=9)

    # Info box
    info_lines = [
        f"n = {valid.sum()}",
        f"Spearman ρ = {rho:.4f} (p = {pval:.2e})",
        f"GAM adj.R² = {adj_r2:.4f}",
    ]
    ax.text(0.95, 0.95, "\n".join(info_lines),
            transform=ax.transAxes, fontsize=10, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if _IN_COLAB:
        plt.show()
    plt.close(fig)

    return rho, adj_r2


# ==============================================================================
# Main
# ==============================================================================
def main():
    args = get_args()
    np.random.seed(args.seed)

    logger.info(f"\n{'='*60}")
    logger.info("Loading features cache")

    # ── Load cache ──
    X_raw, y, lines, uids, which_layer, alive_info = load_features_cache(
        args.features_cache, args.dead_threshold
    )
    logger.info(f"  Raw shape: {X_raw.shape} ({alive_info})")

    # ── GAP L2 norm (양 보정) ──
    # 각 이미지의 벡터를 L2 norm으로 나눠서 세포 양 효과를 보정한다.
    # ex) GAP=[1,2] → L2=√5 → normalized=[1/√5, 2/√5]
    if args.gap_l2_norm:
        norms = np.linalg.norm(X_raw, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-12, norms)
        X_raw = X_raw / norms
        alive_info += " + L2norm"
        logger.info(f"  Applied L2 normalization (양 보정)")

    # concept activation 저장 (L2-normed 상태): 나중에 y축에 사용
    X_concept = X_raw.copy()  # (N, d_alive)

    superclasses = [SUPERCLASS_MAP.get(ln, ln) for ln in lines]
    superclasses_arr = np.array(superclasses)

    unique_sc, sc_counts = np.unique(superclasses_arr, return_counts=True)
    logger.info(f"  Classes: {dict(zip(unique_sc, sc_counts))}")

    # ── Apoptosis ──
    logger.info(f"\n{'='*60}")
    logger.info("Loading apoptosis data")
    apoptosis = load_and_match_apoptosis(args.apoptosis_csv, uids)

    # ── Subsample per class ──
    spc = args.samples_per_class
    if spc > 0:
        rng = np.random.RandomState(args.seed)
        keep_indices = []
        for cls in np.unique(superclasses_arr):
            cls_idx = np.where(superclasses_arr == cls)[0]
            valid_mask = ~np.isnan(apoptosis[cls_idx])
            valid_idx = cls_idx[valid_mask]
            invalid_idx = cls_idx[~valid_mask]
            ordered = np.concatenate([valid_idx, invalid_idx])
            n_take = min(spc, len(ordered))
            chosen = rng.choice(ordered[:max(n_take, len(valid_idx))],
                                size=min(n_take, len(ordered)), replace=False)
            keep_indices.extend(chosen.tolist())
            logger.info(f"  Subsample {cls}: {len(cls_idx)} → {len(chosen)}")
        keep_indices = sorted(keep_indices)
        X_raw = X_raw[keep_indices]
        X_concept = X_concept[keep_indices]
        superclasses = [superclasses[i] for i in keep_indices]
        superclasses_arr = np.array(superclasses)
        apoptosis = apoptosis[keep_indices]
        logger.info(f"  After subsampling: {X_raw.shape[0]} samples")

    # ── Feature filtering (CV / DE) — for DPT manifold ──
    X = X_raw.copy()
    has_de = "de" in args.filter_mode
    filter_steps = []

    for fm in args.filter_mode:
        if fm in ("none", "de"):
            continue
        n_before = X.shape[1]
        if fm == "cv":
            cv = compute_cv_per_neuron(X, superclasses)
            keep_mask = cv >= args.min_cv
            X = X[:, keep_mask]
            step = f"cv≥{args.min_cv}: {n_before}→{X.shape[1]}"
        else:
            continue
        filter_steps.append(step)
        logger.info(f"  Filter [{fm}]: {step}")

    # DE union (for DPT manifold)
    if has_de:
        de_eval_split = args.de_eval_split
        if de_eval_split > 0:
            rng_split = np.random.RandomState(args.seed)
            n_total = len(superclasses_arr)
            eval_mask = np.zeros(n_total, dtype=bool)
            for cls in sorted(set(superclasses_arr)):
                cls_idx = np.where(superclasses_arr == cls)[0]
                n_eval = max(1, int(len(cls_idx) * de_eval_split))
                chosen = rng_split.choice(cls_idx, size=n_eval, replace=False)
                eval_mask[chosen] = True
            de_mask_global = ~eval_mask
        else:
            de_mask_global = np.ones(len(superclasses_arr), dtype=bool)

        de_masks = []
        X_de = X[de_mask_global]
        sc_de = list(superclasses_arr[de_mask_global])
        for mut in ["SNCA", "GBA", "LRRK2"]:
            de_result = compute_de_neurons(
                X_de, sc_de, mut,
                adj_p_threshold=args.de_adj_p,
                min_log2fc=args.de_min_log2fc,
            )
            de_masks.append(de_result["mask"])

        # Control vs AllMut — Control-high
        superclasses_allm = [("AllMut" if s != "Control" else "Control")
                             for s in sc_de]
        de_ctrl = compute_de_neurons(
            X_de, superclasses_allm, "AllMut",
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
        logger.info(f"  DE union: {n_before_de} → {X.shape[1]} neurons")

    filter_label = " → ".join(filter_steps) if filter_steps else "none"
    logger.info(f"  Filter: {filter_label}")

    # ── Normalization → PCA (for DPT manifold) ──
    norm_method = args.norm if args.norm else "none"
    if norm_method != "none":
        X_norm = apply_normalization(X, norm_method)
    else:
        X_norm = X.copy()

    n_pca = min(args.pca_dim, X_norm.shape[1], X_norm.shape[0] - 1)
    pca = PCA(n_components=n_pca, random_state=args.seed)
    X_pca = pca.fit_transform(X_norm)
    var_exp = np.sum(pca.explained_variance_ratio_)
    logger.info(f"  PCA: {X_norm.shape[1]}D → {n_pca}D (var: {var_exp:.1%})")

    n_diffmap = min(args.n_diffmap_comps, n_pca - 1)
    n_diffmap = max(n_diffmap, 2)
    n_dcs = min(args.n_dcs, n_diffmap)
    n_dcs = max(n_dcs, 2)

    # ── Compute DPT (ctrl_mut_pair) ──
    logger.info(f"\n{'='*60}")
    logger.info("Computing DPT (ctrl_mut_pair scope)")
    mutations = ["SNCA", "GBA", "LRRK2"]

    dpt_dict, pair_mask_dict = compute_dpt_ctrl_mut_pair(
        X_pca, superclasses_arr,
        n_neighbors=args.n_neighbors,
        n_pca=n_pca,
        n_diffmap=n_diffmap,
        n_dcs=n_dcs,
        mutations=mutations,
    )

    # ── Output directory ──
    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = os.path.join(os.path.dirname(args.features_cache),
                               "dpt_concept_activation")
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"  Output dir: {out_dir}")

    # ── Concept selection ──
    d_alive = X_concept.shape[1]
    if args.concepts is not None:
        concept_indices = [c for c in args.concepts if c < d_alive]
    else:
        concept_indices = list(range(d_alive))
    logger.info(f"  Total concepts to analyze: {len(concept_indices)}")

    # ── Per-concept × per-mutation analysis ──
    logger.info(f"\n{'='*60}")
    logger.info("Per-concept × per-mutation analysis")

    summary_rows = []

    for ci, concept_idx in enumerate(concept_indices):
        if ci % 100 == 0 and ci > 0:
            logger.info(f"  ... processed {ci}/{len(concept_indices)} concepts")

        for mut in mutations:
            if mut not in dpt_dict:
                continue

            dpt_pair = dpt_dict[mut]
            pair_mask = pair_mask_dict[mut]
            pair_sc = superclasses_arr[pair_mask]

            # Mutation cells only (within the pair)
            mut_in_pair = np.array(pair_sc) == mut
            dpt_mut = dpt_pair[mut_in_pair]

            # Concept activation for mutation cells
            act_mut = X_concept[pair_mask][mut_in_pair, concept_idx]

            # Plot
            fname = f"concept_{concept_idx}_{mut}.png"
            out_path = os.path.join(out_dir, fname)

            rho, adj_r2 = plot_concept_vs_dpt(
                dpt_mut, act_mut, concept_idx, mut,
                out_path, dpi=args.dpi,
                gam_splines=args.gam_splines,
                gam_trim_pctl=tuple(args.gam_trim_pctl),
            )

            summary_rows.append({
                "concept": concept_idx,
                "mutation": mut,
                "spearman_rho": rho,
                "gam_adj_r2": adj_r2,
                "n_cells": int(mut_in_pair.sum()),
            })

    # ── Summary CSV ──
    import pandas as pd
    df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(out_dir, "concept_dpt_summary.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"\n{'='*60}")
    logger.info(f"Summary saved to {csv_path}")
    logger.info(f"Total plots: {len(summary_rows)}")

    # Top concepts by |rho|
    if len(df) > 0:
        df["abs_rho"] = df["spearman_rho"].abs()
        top = df.sort_values("abs_rho", ascending=False).head(20)
        logger.info("\nTop 20 concepts by |Spearman ρ|:")
        for _, row in top.iterrows():
            logger.info(f"  Concept {int(row['concept']):5d}  {row['mutation']:6s}  "
                        f"ρ={row['spearman_rho']:+.4f}  adj.R²={row['gam_adj_r2']:.4f}")

    logger.info(f"\n{'='*60}")
    logger.info("DPT concept activation analysis complete!")


if __name__ == "__main__":
    main()
