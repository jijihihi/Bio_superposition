import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

from model_train.logging_utils import get_logger

logger = get_logger("trajectory_api_pairwise")

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
sns.set_style("ticks")


# ==============================================================================
# Helper: GAM R2
# ==============================================================================
def compute_gam_r2(dpt_vals, apop_vals, gam_splines=20, gam_trim_pctl=(1, 99)):
    try:
        from pygam import LinearGAM
        from pygam import s as s_term

        pct_lo, pct_hi = np.percentile(dpt_vals, list(gam_trim_pctl))
        dense_mask = (dpt_vals >= pct_lo) & (dpt_vals <= pct_hi)
        dpt_dense = dpt_vals[dense_mask]
        apop_dense = apop_vals[dense_mask]
        if len(dpt_dense) < 20:
            return 0.0
        n_sp = min(gam_splines, max(5, len(dpt_dense) // 50))
        gam = LinearGAM(s_term(0, n_splines=n_sp, spline_order=3)).fit(
            dpt_dense.reshape(-1, 1), apop_dense
        )
        ss_res = np.sum((apop_dense - gam.predict(dpt_dense.reshape(-1, 1))) ** 2)
        ss_tot = np.sum((apop_dense - apop_dense.mean()) ** 2)
        n = len(apop_dense)
        p = gam.statistics_["edof"]
        if ss_tot > 0 and n > p + 1:
            return 1 - (ss_res / (n - p - 1)) / (ss_tot / (n - 1))
        return 0.0
    except ImportError:
        return 0.0


# ==============================================================================
# Main Pairwise Trajectory
# ==============================================================================
def run_pairwise_trajectory(
    adata: sc.AnnData,
    mutation: str,
    out_dir: str,
    n_pca: int = 100,
    n_neighbors: int = 15,
    n_diffmap: int = 15,
    n_dcs: int = 10,
    diffmap_dc: list = [2, 3],
    dpi: int = 300,
    prefix: str = "",
) -> sc.AnnData:
    """
    Compute Diffmap, PAGA, PHATE, and DPT for a Control+Mutation pair.
    Returns adata_pair.
    """
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"\n--- Running Pairwise Trajectory: Control + {mutation} ---")

    superclasses = adata.obs["mutation"].values
    pair_mask = (superclasses == "Control") | (superclasses == mutation)

    if np.sum(superclasses == mutation) < 10:
        logger.warning(f"Not enough {mutation} samples.")
        return None

    # Subset
    adata_pair = adata[pair_mask].copy()
    pair_sc = adata_pair.obs["mutation"].values

    n_pair = adata_pair.n_obs
    n_pca_actual = min(n_pca, adata_pair.n_vars - 1)

    # PCA -> Neighbors -> Diffmap
    sc.pp.pca(adata_pair, n_comps=n_pca_actual)
    sc.pp.neighbors(adata_pair, n_neighbors=n_neighbors, n_pcs=n_pca_actual)

    n_diff_actual = min(n_diffmap, n_pair - 2)
    n_dcs_actual = min(n_dcs, n_diff_actual)
    sc.tl.diffmap(adata_pair, n_comps=n_diff_actual)

    # Root: Control centroid in diffmap space
    diffmap_coords = adata_pair.obsm["X_diffmap"]
    pair_ctrl_mask = pair_sc == "Control"
    ctrl_centroid = diffmap_coords[pair_ctrl_mask].mean(axis=0)
    ctrl_dists = np.linalg.norm(diffmap_coords[pair_ctrl_mask] - ctrl_centroid, axis=1)
    root_in_pair = np.where(pair_ctrl_mask)[0][np.argmin(ctrl_dists)]

    adata_pair.uns["iroot"] = int(root_in_pair)
    sc.tl.dpt(adata_pair, n_dcs=n_dcs_actual)
    dpt_pair = adata_pair.obs["dpt_pseudotime"].values

    logger.info(f"Root DPT = {dpt_pair[root_in_pair]:.6f}")

    # Evaluate cell_death Correlation
    if "cell_death" in adata_pair.obs:
        apop = adata_pair.obs["cell_death"].values
        mut_mask = pair_sc == mutation
        valid_mask = mut_mask & ~np.isnan(apop)

        if valid_mask.sum() > 10:
            dpt_mut = dpt_pair[valid_mask]
            apop_mut = apop[valid_mask]

            rho, p_rho = spearmanr(dpt_mut, apop_mut)
            r, p_r = pearsonr(dpt_mut, apop_mut)
            gam_r2 = compute_gam_r2(dpt_mut, apop_mut)

            logger.info(f"[{mutation}] Spearman Rho: {rho:.4f} (p={p_rho:.2e})")
            logger.info(f"[{mutation}] Pearson R:    {r:.4f} (p={p_r:.2e})")
            logger.info(f"[{mutation}] GAM R²:       {gam_r2:.4f}")

            adata_pair.uns["dpt_stats"] = {"rho": rho, "r": r, "gam_r2": gam_r2}

    # Plot Diffusion Map
    _plot_diffmap(adata_pair, diffmap_dc, mutation, out_dir, prefix, dpi)

    return adata_pair


def _plot_diffmap(adata_pair, diffmap_dc, mutation, out_dir, prefix, dpi):
    diffmap_coords = adata_pair.obsm["X_diffmap"]
    n_dc_avail = diffmap_coords.shape[1]

    dc_x, dc_y = diffmap_dc
    if dc_x > n_dc_avail or dc_y > n_dc_avail:
        return

    x_vals = diffmap_coords[:, dc_x - 1]
    y_vals = diffmap_coords[:, dc_y - 1]

    pair_sc = adata_pair.obs["mutation"].values
    ctrl_mask = pair_sc == "Control"
    mut_mask = pair_sc == mutation

    fig, ax = plt.subplots(figsize=(7, 6))

    # Control (gray)
    ax.scatter(
        x_vals[ctrl_mask],
        y_vals[ctrl_mask],
        c="#FFAB91",
        s=10,
        alpha=0.3,
        label="Control",
        rasterized=True,
    )

    # Mutation (cell_death gradient)
    if "cell_death" in adata_pair.obs:
        apop = adata_pair.obs["cell_death"].values[mut_mask]
        sc = ax.scatter(
            x_vals[mut_mask],
            y_vals[mut_mask],
            c=apop,
            cmap="YlOrRd",
            s=10,
            alpha=0.8,
            rasterized=True,
        )
        plt.colorbar(sc, ax=ax, label="cell_death Rate")
    else:
        ax.scatter(
            x_vals[mut_mask],
            y_vals[mut_mask],
            c="#DD8452",
            s=10,
            alpha=0.8,
            label=mutation,
            rasterized=True,
        )

    ax.set_xlabel(f"DC{dc_x}")
    ax.set_ylabel(f"DC{dc_y}")
    ax.legend()
    sns.despine()

    out_name = (
        f"pairwise_diffmap_{mutation}_{prefix}"
        if prefix
        else f"pairwise_diffmap_{mutation}"
    )
    fig.savefig(os.path.join(out_dir, f"{out_name}.png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(
        os.path.join(out_dir, f"{out_name}.svg"), format="svg", bbox_inches="tight"
    )
    plt.close(fig)
