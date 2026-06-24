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


# import sys
# sys.argv = [
#     "dpt_concept_activation",
#     "--features_cache", "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/MoCo_seed87/SAE_seed856_no_L2norm_loss/features_cache_stage5_out_normrestored_all_no_SAE_GAP_L2_norm_again_d8192_sp800.npz",
#     "--concept_cache", "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/MoCo_seed87/SAE_seed856_no_L2norm_loss/features_cache_stage5_out_normrestored_all_no_SAE_GAP_L2_norm_again_d8192_sp3200.npz",
#     "--concept_vis_dir", "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/MoCo_seed87/SAE_seed856_no_L2norm_loss/concept_vis_output",
#     "--apoptosis_csv", "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/세포이미지별 사멸율/이미지별_세포사멸율_7200.csv",
#     "--filter_mode", "cv", "de",
#     "--min_cv", "0.1",
#     "--de_adj_p", "0.05",
#     "--de_min_log2fc", "1.0",
#     "--dead_threshold", "5e-5",
#     "--concept_dead_threshold", "1e-5",
#     "--root_mode", "diffmap",
#     "--dpt_scope", "ctrl_mut_pair",
#     "--norm", "log_std",
#     "--n_neighbors", "35",
#     "--pca_dim", "15",
#     "--n_diffmap_comps", "10",
#     "--de_eval_split", "0.5",
#     "--gam_splines", "8",
#     "--gam_trim_pctl", "5", "95",
#     "--gap_l2_norm",
#     "--seed", "856",
# ]

# from concept_visulaize.dpt_concept_activation import main
# main()

# bilinear interpolation concpet 번호가 일치.
# features_cache (sp800) → DPT 매니폴드
# concept_cache  (sp3200) → y축 concept GAP activation
# concept_vis_dir → step14 출력 폴더에서 concept ID + class label 파싱


# 값 이상하다. GAP_CSV 뽑을때와 동일한 방법으로 cache를 뽑아야하는거 같아. 그래야 잘 될듯하다.


# concept_0037_LRRK2_SNCA. 이렇게 되면 control LRRK2로만 DPT만들고 이 궤적 위에서 LRRK2 이미지들 GAP 분석. SNCA도 마찬가지로

# → mutation별로 Control+해당Mutation 이미지만 사용해서 DPT 구축


import argparse
import os
import sys

import matplotlib
import numpy as np
import scanpy as sc
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

_IN_COLAB = "google.colab" in sys.modules
if not _IN_COLAB:
    matplotlib.use("Agg")
import re

import matplotlib.pyplot as plt

# ── dpt_kendall 에서 import ──
from kendall_correlation_coefficient.dpt_kendall import (
    apply_normalization, compute_cv_per_neuron, compute_de_neurons,
    load_and_match_apoptosis, load_features_cache)
from sae_project.step02_logging_utils import SUPERCLASS_MAP, get_logger

logger = get_logger("dpt_concept_act")

MUTATION_COLORS = {"SNCA": "#E24A33", "GBA": "#348ABD", "LRRK2": "#988ED5"}


# ==============================================================================
# Argument Parser
# ==============================================================================
def get_args():
    p = argparse.ArgumentParser(
        description="DPT vs per-concept GAP activation (Spearman ρ + GAM adj.R²)"
    )

    p.add_argument(
        "--features_cache",
        type=str,
        required=True,
        help="DPT 매니폴드용 cache (sparsity 800, 정보 손실 작음)",
    )
    p.add_argument(
        "--concept_cache",
        type=str,
        required=True,
        help="concept GAP activation용 cache (sparsity 3200, monosemantic)",
    )
    p.add_argument("--apoptosis_csv", type=str, required=True)
    p.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory for plots (default: next to cache)",
    )
    p.add_argument(
        "--dead_threshold",
        type=float,
        default=5e-5,
        help="DPT cache (sp800) dead neuron threshold",
    )
    p.add_argument(
        "--gap_l2_norm", action="store_true", help="Apply per-image L2 normalization"
    )

    # Concept selection: step14 출력 폴더에서 파싱
    p.add_argument(
        "--concept_vis_dir",
        type=str,
        default="",
        help="step14 출력 폴더 경로 (concept_XXXX_CLASS 폴더명에서 ID/class 파싱)",
    )
    p.add_argument(
        "--concept_dead_threshold",
        type=float,
        default=1e-5,
        help="Concept cache (sp3200) dead neuron threshold",
    )

    # ===== DPT manifold filtering (sp800) =====
    p.add_argument(
        "--filter_mode",
        type=str,
        nargs="+",
        default=["none"],
        help="DPT 매니폴드용 필터: 'cv', 'de', 'none'. e.g. '--filter_mode cv de'",
    )
    p.add_argument("--min_cv", type=float, default=0.1, help="DPT 매니폴드용 CV 임계값")
    p.add_argument(
        "--de_adj_p",
        type=float,
        default=0.05,
        help="DPT 매니폴드용 DE adjusted p-value",
    )
    p.add_argument(
        "--de_min_log2fc",
        type=float,
        default=1.0,
        help="DPT 매니폴드용 DE min |log2FC|",
    )

    # Normalization for DPT manifold
    p.add_argument(
        "--norm",
        type=str,
        default="log_std",
        help="Feature normalization for DPT manifold (e.g. log_std)",
    )

    # PCA / kNN / diffmap
    p.add_argument("--pca_dim", type=int, default=15)
    p.add_argument("--n_neighbors", type=int, default=35)
    p.add_argument("--n_diffmap_comps", type=int, default=10)
    p.add_argument("--n_dcs", type=int, default=10)

    # DPT scope
    p.add_argument(
        "--dpt_scope",
        type=str,
        default="ctrl_mut_pair",
        choices=["ctrl_mut_pair", "global"],
        help="'ctrl_mut_pair': Control+Mut pair별 DPT. 'global': 전체.",
    )

    # Root selection
    p.add_argument(
        "--root_mode", type=str, default="diffmap", choices=["pca", "diffmap"]
    )
    p.add_argument("--root_perturbation_n", type=int, default=10)

    # GAM
    p.add_argument("--gam_splines", type=int, default=8)
    p.add_argument("--gam_trim_pctl", type=float, nargs=2, default=[5, 95])

    # Misc
    p.add_argument("--seed", type=int, default=856)
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--samples_per_class", type=int, default=5000)
    p.add_argument(
        "--concepts",
        type=int,
        nargs="*",
        default=None,
        help="Specific original concept indices to plot (overrides --concept_vis_dir)",
    )
    p.add_argument("--de_eval_split", type=float, default=0.5)

    return p.parse_args()


# ==============================================================================
# DPT computation — ctrl_mut_pair scope (from dpt_kendall logic)
# ==============================================================================
def compute_dpt_ctrl_mut_pair(
    X_pca, superclasses_arr, n_neighbors, n_pca, n_diffmap, n_dcs, mutations=None
):
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
        logger.info(
            f"  {mut} pair: {n_pair} cells "
            f"(Ctrl={ctrl_mask.sum()}, {mut}={mut_mask.sum()})"
        )

        # Build diffmap on pair
        adata_pair = sc.AnnData(X_pca_pair.astype(np.float32))
        adata_pair.obsm["X_pca"] = X_pca_pair.astype(np.float32)
        adata_pair.obs["superclass"] = list(pair_sc)

        n_diffmap_pair = min(n_diffmap, n_pair - 2)
        n_diffmap_pair = max(n_diffmap_pair, 2)
        n_dcs_pair = min(n_dcs, n_diffmap_pair)
        n_dcs_pair = max(n_dcs_pair, 2)

        sc.pp.neighbors(
            adata_pair, n_neighbors=n_neighbors, n_pcs=n_pca, use_rep="X_pca"
        )
        sc.tl.diffmap(adata_pair, n_comps=n_diffmap_pair)

        evals = adata_pair.uns["diffmap_evals"]
        logger.info(f"    Eigenvalues (top 3): {evals[:3]}")

        # Root: Control centroid in diffmap space
        diffmap_coords = adata_pair.obsm["X_diffmap"]
        pair_ctrl_mask = np.array(pair_sc) == "Control"
        ctrl_centroid = diffmap_coords[pair_ctrl_mask].mean(axis=0)
        ctrl_dists = np.linalg.norm(
            diffmap_coords[pair_ctrl_mask] - ctrl_centroid, axis=1
        )
        root_in_pair = np.where(pair_ctrl_mask)[0][np.argmin(ctrl_dists)]
        logger.info(f"    Root: Ctrl cell (pair idx {root_in_pair})")

        adata_pair.uns["iroot"] = int(root_in_pair)
        sc.tl.dpt(adata_pair, n_dcs=n_dcs_pair)
        dpt_pair = adata_pair.obs["dpt_pseudotime"].values.copy()

        # ── Direction verification & auto-flip ──
        # DPT 방향 보장: Control mean DPT < Mutation mean DPT
        # scanpy은 root=0에서 방사형으로 DPT 할당하지만,
        # diffmap 고유벡터 방향에 따라 Mutation이 Control보다
        # 작은 DPT를 받을 수 있음 → 이 경우 flip
        pair_ctrl_dpt = np.nanmean(dpt_pair[pair_ctrl_mask])
        pair_mut_dpt = np.nanmean(dpt_pair[np.array(pair_sc) == mut])

        if pair_ctrl_dpt > pair_mut_dpt:
            max_dpt = np.nanmax(dpt_pair)
            dpt_pair = max_dpt - dpt_pair
            logger.warning(
                f"    ⚠ DPT FLIPPED: Ctrl({pair_ctrl_dpt:.4f}) > "
                f"{mut}({pair_mut_dpt:.4f}) → reversed"
            )
            # 재확인
            pair_ctrl_dpt_new = np.nanmean(dpt_pair[pair_ctrl_mask])
            pair_mut_dpt_new = np.nanmean(dpt_pair[np.array(pair_sc) == mut])
            logger.info(
                f"    After flip: Ctrl={pair_ctrl_dpt_new:.4f}, "
                f"{mut}={pair_mut_dpt_new:.4f}"
            )
        else:
            logger.info(
                f"    Direction OK: Ctrl({pair_ctrl_dpt:.4f}) < "
                f"{mut}({pair_mut_dpt:.4f})"
            )

        logger.info(
            f"    Control medoid DPT = {dpt_pair[root_in_pair]:.6f} "
            f"(pair idx {root_in_pair})"
        )
        logger.info(
            f"    Ctrl: mean={np.nanmean(dpt_pair[np.array(pair_sc) == 'Control']):.4f}, "
            f"median={np.nanmedian(dpt_pair[np.array(pair_sc) == 'Control']):.4f}"
        )
        logger.info(
            f"    {mut}: mean={np.nanmean(dpt_pair[np.array(pair_sc) == mut]):.4f}, "
            f"median={np.nanmedian(dpt_pair[np.array(pair_sc) == mut]):.4f}"
        )

        dpt_dict[mut] = dpt_pair
        pair_mask_dict[mut] = pair_mask

        del adata_pair

    return dpt_dict, pair_mask_dict


# ==============================================================================
# Plot: single concept × mutation — scatter + GAM + Spearman ρ + adj.R²
# ==============================================================================
def plot_concept_vs_dpt(
    dpt_vals,
    act_vals,
    concept_idx,
    mutation,
    output_path,
    dpi=200,
    gam_splines=8,
    gam_trim_pctl=(5, 95),
    class_label="",
):
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
    ax.scatter(
        dpt_v,
        act_v,
        s=6,
        alpha=0.25,
        c=color,
        edgecolors="none",
        rasterized=True,
        zorder=1,
    )

    # GAM fit
    adj_r2 = 0.0
    pct_lo, pct_hi = np.percentile(dpt_v, list(gam_trim_pctl))
    dense_mask = (dpt_v >= pct_lo) & (dpt_v <= pct_hi)
    dpt_dense = dpt_v[dense_mask]
    act_dense = act_v[dense_mask]

    x_line = np.linspace(pct_lo, pct_hi, 200)
    try:
        from pygam import LinearGAM
        from pygam import s as s_term

        n_sp = min(gam_splines, max(5, len(dpt_dense) // 50))
        gam = LinearGAM(s_term(0, n_splines=n_sp, spline_order=3)).fit(
            dpt_dense.reshape(-1, 1), act_dense
        )
        y_gam = gam.predict(x_line.reshape(-1, 1))
        ci = gam.confidence_intervals(x_line.reshape(-1, 1), width=0.95)

        ax.plot(
            x_line,
            y_gam,
            "-",
            color="black",
            lw=2.5,
            alpha=0.9,
            zorder=5,
            label="GAM fit",
        )
        ax.fill_between(
            x_line,
            ci[:, 0],
            ci[:, 1],
            color="black",
            alpha=0.12,
            zorder=2,
            label="95% CI",
        )

        # Adjusted R²
        ss_res = np.sum((act_dense - gam.predict(dpt_dense.reshape(-1, 1))) ** 2)
        ss_tot = np.sum((act_dense - act_dense.mean()) ** 2)
        n = len(act_dense)
        p = gam.statistics_["edof"]
        if ss_tot > 0 and n > p + 1:
            adj_r2 = 1 - (ss_res / (n - p - 1)) / (ss_tot / (n - 1))
        else:
            adj_r2 = 0.0
    except ImportError:
        # Fallback: linear fit
        if len(dpt_v) > 2:
            z = np.polyfit(dpt_v, act_v, 1)
            ax.plot(
                x_line,
                np.polyval(z, x_line),
                "--",
                color="black",
                lw=2,
                alpha=0.7,
                zorder=3,
                label="Linear fit",
            )

    ax.set_xlabel("DPT (Diffusion Pseudotime)", fontsize=12)
    ax.set_ylabel(f"Concept {concept_idx} activation (L2-normed GAP)", fontsize=12)
    title = f"Concept {concept_idx}"
    if class_label:
        title += f" ({class_label})"
    title += f" — {mutation}"
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper left", fontsize=9)

    # Info box
    info_lines = [
        f"n = {valid.sum()}",
        f"Spearman ρ = {rho:.4f} (p = {pval:.2e})",
        f"GAM adj.R² = {adj_r2:.4f}",
    ]
    ax.text(
        0.95,
        0.95,
        "\n".join(info_lines),
        transform=ax.transAxes,
        fontsize=10,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if _IN_COLAB:
        plt.show()
    plt.close(fig)

    return rho, adj_r2


def parse_relevant_mutations(cls_label: str, all_mutations: list) -> list:
    """
    Concept 폴더명의 class label에서 관련 mutation을 추출.

    Examples:
        'GBA' → ['GBA']
        'LRRK2_SNCA' → ['LRRK2', 'SNCA']
        'SNCA_GBA' → ['SNCA', 'GBA']
        'Control' → all_mutations  (모든 mutation에 대해 분석)
        '' → all_mutations
    """
    if not cls_label or cls_label.lower() == "control":
        return list(all_mutations)

    relevant = [m for m in all_mutations if m in cls_label]
    if not relevant:
        # label에 알려진 mutation이 없으면 전체
        return list(all_mutations)
    return relevant


# ==============================================================================
# Main
# ==============================================================================
def main():
    args = get_args()
    np.random.seed(args.seed)

    # ===================================================================
    # 1) DPT 매니폴드용 cache 로드 (sparsity 800)
    # ===================================================================
    logger.info(f"\n{'='*60}")
    logger.info("Loading DPT cache (sp800, for manifold)")
    logger.info(f"  {args.features_cache}")

    X_dpt, y, lines, uids_dpt, which_layer, alive_info_dpt = load_features_cache(
        args.features_cache, args.dead_threshold
    )
    logger.info(f"  DPT shape: {X_dpt.shape} ({alive_info_dpt})")

    # DPT cache에도 GAP L2 norm 적용 (DPT 매니폴드 구축 시 양 보정)
    if args.gap_l2_norm:
        norms = np.linalg.norm(X_dpt, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-12, norms)
        X_dpt = X_dpt / norms
        logger.info(f"  Applied L2 norm to DPT features")

    # ===================================================================
    # 2) Concept activation용 cache 로드 (sparsity 3200)
    # ===================================================================
    logger.info(f"\n{'='*60}")
    logger.info("Loading concept cache (sp3200, for y-axis activation)")
    logger.info(f"  {args.concept_cache}")

    # concept cache의 alive index 매핑 (concept_dead_threshold 사용)
    data_concept = np.load(args.concept_cache, allow_pickle=True)
    uids_concept = list(data_concept["uids"])
    if "usage_ema" in data_concept:
        usage_concept = data_concept["usage_ema"]
        alive_mask_concept = usage_concept >= args.concept_dead_threshold
        alive_indices_concept = np.where(alive_mask_concept)[0]
    else:
        alive_indices_concept = np.arange(data_concept["X_all"].shape[1])

    X_concept_raw, _, _, _, _, alive_info_concept = load_features_cache(
        args.concept_cache, args.concept_dead_threshold
    )
    logger.info(f"  Concept shape: {X_concept_raw.shape} ({alive_info_concept})")

    # original concept ID → alive column index 매핑 (concept cache 기준)
    orig_to_alive_col = {
        int(orig_id): col for col, orig_id in enumerate(alive_indices_concept)
    }

    # ===================================================================
    # 3) UID 정렬 — 두 cache의 교집합 UID로 정렬
    #    sp800 (extract_features_lambda_labs): 순차적 순서
    #    sp3200 (step09 eval_ckpt): StrictPlateBalanced 순서 → 순서 다를 수 있음
    # ===================================================================
    set_dpt = set(uids_dpt)
    set_concept = set(uids_concept)
    common_uids = set_dpt & set_concept

    if len(common_uids) == 0:
        raise ValueError("features_cache와 concept_cache에 공통 UID가 없습니다!")

    n_dpt_only = len(set_dpt - common_uids)
    n_concept_only = len(set_concept - common_uids)
    if n_dpt_only > 0 or n_concept_only > 0:
        logger.warning(
            f"  UID mismatch: DPT-only={n_dpt_only}, Concept-only={n_concept_only}"
        )

    # DPT cache 기준 순서로 정렬 (DPT가 주축)
    dpt_uid_to_idx = {uid: i for i, uid in enumerate(uids_dpt)}
    concept_uid_to_idx = {uid: i for i, uid in enumerate(uids_concept)}

    # common UID를 DPT cache 순서로 정렬
    common_ordered = sorted(common_uids, key=lambda u: dpt_uid_to_idx[u])
    dpt_indices = [dpt_uid_to_idx[u] for u in common_ordered]
    concept_indices = [concept_uid_to_idx[u] for u in common_ordered]

    # 재정렬
    X_dpt = X_dpt[dpt_indices]
    X_concept_raw = X_concept_raw[concept_indices]
    lines = [lines[i] for i in dpt_indices]
    y = y[dpt_indices]
    uids = common_ordered

    logger.info(
        f"  UID alignment: {len(common_ordered)} common images "
        f"(DPT={len(uids_dpt)}, Concept={len(uids_concept)})"
    )

    # Concept cache에 GAP L2 norm 적용 (y축 양 보정)
    if args.gap_l2_norm:
        norms = np.linalg.norm(X_concept_raw, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-12, norms)
        X_concept_raw = X_concept_raw / norms
        logger.info(f"  Applied L2 norm to concept features")

    # concept activation 저장 (L2-normed 상태): 나중에 y축에 사용
    X_concept = X_concept_raw.copy()  # (N, d_alive_sp3200)

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
            chosen = rng.choice(
                ordered[: max(n_take, len(valid_idx))],
                size=min(n_take, len(ordered)),
                replace=False,
            )
            keep_indices.extend(chosen.tolist())
            logger.info(f"  Subsample {cls}: {len(cls_idx)} → {len(chosen)}")
        keep_indices = sorted(keep_indices)
        X_dpt = X_dpt[keep_indices]
        X_concept = X_concept[keep_indices]
        superclasses = [superclasses[i] for i in keep_indices]
        superclasses_arr = np.array(superclasses)
        apoptosis = apoptosis[keep_indices]
        logger.info(f"  After subsampling: {X_dpt.shape[0]} samples")

    # ── Feature filtering (CV / DE) — for DPT manifold (sp800) ──
    X = X_dpt.copy()
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
                X_de,
                sc_de,
                mut,
                adj_p_threshold=args.de_adj_p,
                min_log2fc=args.de_min_log2fc,
            )
            de_masks.append(de_result["mask"])

        # Control vs AllMut — Control-high
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
        X_pca,
        superclasses_arr,
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
        out_dir = os.path.join(
            os.path.dirname(args.features_cache), "dpt_concept_activation"
        )
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"  Output dir: {out_dir}")

    # ── Concept selection ──
    d_alive = X_concept.shape[1]

    if args.concepts is not None:
        # Manual: 직접 concept ID 지정
        concept_entries = []
        for orig_id in args.concepts:
            if orig_id in orig_to_alive_col:
                concept_entries.append((orig_to_alive_col[orig_id], orig_id, ""))
            else:
                logger.warning(f"  Concept {orig_id} not in alive set, skip")
        logger.info(f"  Manual: {len(concept_entries)} concepts")

    elif args.concept_vis_dir:
        # step14 출력 폴더에서 concept ID + class label 파싱
        # 폴더명 예시: concept_0037_GBA, concept_0152_SNCA_GBA, concept_0018_Control
        logger.info(f"\n{'='*60}")
        logger.info(f"Parsing concepts from step14 output: {args.concept_vis_dir}")
        concept_entries = []
        pattern = re.compile(r"^concept_(\d+)(?:_(.+))?$")

        for folder_name in sorted(os.listdir(args.concept_vis_dir)):
            folder_path = os.path.join(args.concept_vis_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
            m = pattern.match(folder_name)
            if m:
                orig_id = int(m.group(1))
                cls_label = m.group(2) or ""  # e.g. "GBA", "SNCA_GBA", "Control"
                if orig_id in orig_to_alive_col:
                    concept_entries.append(
                        (orig_to_alive_col[orig_id], orig_id, cls_label)
                    )
                else:
                    logger.warning(
                        f"  Concept {orig_id} ({cls_label}) not in alive set, skip"
                    )

        logger.info(f"  Found {len(concept_entries)} concepts from step14 output")
        for alive_col, orig_id, cls in concept_entries[:20]:
            logger.info(f"    Concept {orig_id} (col={alive_col}): {cls}")
        if len(concept_entries) > 20:
            logger.info(f"    ... and {len(concept_entries) - 20} more")

    else:
        # all alive
        concept_entries = [
            (col, int(alive_indices_concept[col]), "") for col in range(d_alive)
        ]
        logger.info(f"  All alive: {len(concept_entries)} concepts")

    if len(concept_entries) == 0:
        logger.warning("No concepts to analyze!")
        return

    # ── Per-concept × per-mutation analysis ──
    logger.info(f"\n{'='*60}")
    logger.info(
        f"Per-concept × per-mutation analysis ({len(concept_entries)} concepts)"
    )

    summary_rows = []

    for ci, (alive_col, orig_id, cls_label) in enumerate(concept_entries):
        if ci % 50 == 0 and ci > 0:
            logger.info(f"  ... processed {ci}/{len(concept_entries)} concepts")

        # cls_label에서 해당 concept에 관련된 mutation만 추출
        relevant_muts = parse_relevant_mutations(cls_label, mutations)

        for mut in relevant_muts:
            if mut not in dpt_dict:
                continue

            dpt_pair = dpt_dict[mut]
            pair_mask = pair_mask_dict[mut]
            pair_sc = superclasses_arr[pair_mask]

            # Mutation cells only (within the pair)
            mut_in_pair = np.array(pair_sc) == mut
            dpt_mut = dpt_pair[mut_in_pair]

            # Concept activation for mutation cells (using alive column index)
            act_mut = X_concept[pair_mask][mut_in_pair, alive_col]

            # Plot — per-concept subfolder, like activation_maximization
            label_str = f"_{cls_label}" if cls_label else ""
            concept_folder = os.path.join(out_dir, f"concept_{orig_id:04d}{label_str}")
            os.makedirs(concept_folder, exist_ok=True)
            fname = f"concept_{orig_id:04d}{label_str}_{mut}.png"
            out_path = os.path.join(concept_folder, fname)

            rho, adj_r2 = plot_concept_vs_dpt(
                dpt_mut,
                act_mut,
                orig_id,
                mut,
                out_path,
                dpi=args.dpi,
                gam_splines=args.gam_splines,
                gam_trim_pctl=tuple(args.gam_trim_pctl),
                class_label=cls_label,
            )

            summary_rows.append(
                {
                    "concept_original_id": orig_id,
                    "alive_col": alive_col,
                    "class_label": cls_label,
                    "mutation": mut,
                    "spearman_rho": rho,
                    "gam_adj_r2": adj_r2,
                    "n_cells": int(mut_in_pair.sum()),
                }
            )

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
            cls_str = f" ({row['class_label']})" if row.get("class_label") else ""
            logger.info(
                f"  Concept {int(row['concept_original_id']):5d}{cls_str}  "
                f"{row['mutation']:6s}  "
                f"ρ={row['spearman_rho']:+.4f}  adj.R²={row['gam_adj_r2']:.4f}"
            )

    logger.info(f"\n{'='*60}")
    logger.info("DPT concept activation analysis complete!")


if __name__ == "__main__":
    main()
