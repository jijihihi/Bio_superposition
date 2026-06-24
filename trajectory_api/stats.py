import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

from sae_project.step02_logging_utils import get_logger

logger = get_logger("trajectory_api_stats")

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
sns.set_style("ticks")

MUTATION_COLORS = {
    "SNCA": "#DD8452",
    "GBA": "#55A868",
    "LRRK2": "#C44E52",
    "Control": "#4C72B0",
}


# ==============================================================================
# DPT Spread & Statistics
# ==============================================================================
def plot_trajectory_statistics(
    adata_pair: sc.AnnData,
    mutation: str,
    out_dir: str,
    prefix: str = "",
    dpi: int = 300,
    n_root_perturb: int = 10,
    n_dcs: int = 10,
):
    """
    Run statistics on the computed pairwise trajectory.
    Includes Early/Mid/Late binning, Root perturbation robustness, and Spread calculation.
    """
    os.makedirs(out_dir, exist_ok=True)

    if "dpt_pseudotime" not in adata_pair.obs:
        logger.warning("No dpt_pseudotime found in adata.")
        return

    dpt_vals = adata_pair.obs["dpt_pseudotime"].values
    superclasses = adata_pair.obs["mutation"].values

    # 1. DPT Distribution & Diagnostics
    _diagnose_dpt_distribution(dpt_vals, superclasses, mutation, out_dir, prefix, dpi)

    # 2. Root Perturbation
    if n_root_perturb > 0:
        apop = (
            adata_pair.obs["apoptosis"].values
            if "apoptosis" in adata_pair.obs
            else None
        )
        if apop is not None:
            _run_root_perturbation(
                adata_pair,
                superclasses,
                apop,
                mutation,
                out_dir,
                prefix,
                dpi,
                n_root_perturb,
                n_dcs,
            )


def _diagnose_dpt_distribution(dpt_vals, superclasses, mutation, out_dir, prefix, dpi):
    logger.info(f"\n--- DPT Distribution Diagnostics ({mutation}) ---")

    mut_mask = superclasses == mutation
    ctrl_mask = superclasses == "Control"

    for grp, mask in [("Control", ctrl_mask), (mutation, mut_mask)]:
        dpt_grp = dpt_vals[mask]
        valid = np.isfinite(dpt_grp)
        if valid.sum() < 2:
            continue

        dpt_v = dpt_grp[valid]
        p5, p95 = np.percentile(dpt_v, [5, 95])
        spread = p95 - p5

        logger.info(
            f"[{grp}] n={valid.sum()} | spread={spread:.4f} | mean={dpt_v.mean():.4f}"
        )

        # Plot Histogram
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(
            dpt_v,
            bins=60,
            alpha=0.7,
            color=MUTATION_COLORS.get(grp, "#888"),
            edgecolor="black",
            linewidth=0.3,
        )
        ax.set_xlabel("Diffusion Pseudotime", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(f"{grp} DPT Distribution (spread={spread:.4f})", fontsize=13)
        sns.despine(ax=ax)
        fig.tight_layout()

        out_name = f"dpt_dist_{grp}_{prefix}.png" if prefix else f"dpt_dist_{grp}.png"
        fig.savefig(os.path.join(out_dir, out_name), dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def _run_root_perturbation(
    adata, superclasses, apoptosis, mutation, out_dir, prefix, dpi, n_roots=10, n_dcs=10
):
    logger.info(f"\n--- Root Perturbation Robustness Test ({mutation}) ---")

    diffmap_coords = adata.obsm["X_diffmap"]
    ctrl_mask = superclasses == "Control"
    ctrl_indices = np.where(ctrl_mask)[0]

    if len(ctrl_indices) < n_roots:
        n_roots = len(ctrl_indices)

    ctrl_centroid = diffmap_coords[ctrl_mask].mean(axis=0)
    ctrl_dists = np.linalg.norm(diffmap_coords[ctrl_mask] - ctrl_centroid, axis=1)
    root_candidates = ctrl_indices[np.argsort(ctrl_dists)[:n_roots]]

    rhos = []
    for root_idx in root_candidates:
        adata.uns["iroot"] = int(root_idx)
        sc.tl.dpt(adata, n_dcs=n_dcs)
        dpt_vals = adata.obs["dpt_pseudotime"].values

        mut_mask = superclasses == mutation
        dpt_m = dpt_vals[mut_mask]
        apop_m = apoptosis[mut_mask]

        valid = np.isfinite(dpt_m) & ~np.isnan(apop_m)
        if valid.sum() > 10:
            rho, _ = spearmanr(dpt_m[valid], apop_m[valid])
            rhos.append(rho if not np.isnan(rho) else 0.0)

    if rhos:
        rhos = np.array(rhos)
        m, s = rhos.mean(), rhos.std()
        logger.info(
            f"Root Perturbation (n={len(rhos)}): Spearman mean={m:.4f}, std={s:.4f}"
        )
        if s > 0.05:
            logger.warning(f"High sensitivity to root selection (std > 0.05).")
