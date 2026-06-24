import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import phate
import scanpy as sc
import seaborn as sns

from sae_project.step02_logging_utils import get_logger

logger = get_logger("trajectory_api_global")

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
sns.set_style("ticks")

SUPERCLASS_COLORS = {
    "Control": "#4C72B0",  # blue
    "SNCA": "#DD8452",  # orange
    "GBA": "#55A868",  # green
    "LRRK2": "#C44E52",  # red
}


# ==============================================================================
# 1. Global PHATE
# ==============================================================================
def plot_global_phate(
    adata: sc.AnnData,
    out_dir: str,
    prefix: str = "",
    knn: int = 5,
    t: str = "auto",
    n_pca: int = 100,
    decay: int = 40,
    knn_dist: str = "euclidean",
    point_size: float = 1.5,
    alpha: float = 0.2,
    dpi: int = 300,
    n_jobs: int = -1,
):
    """
    Compute and plot Global PHATE embedding for all classes.
    """
    os.makedirs(out_dir, exist_ok=True)

    X = adata.X
    superclasses = adata.obs["mutation"].values

    n_pca_actual = min(n_pca, X.shape[1])
    t_value = t if t == "auto" else int(t)

    logger.info(f"Running Global PHATE: knn={knn}, t={t_value}, n_pca={n_pca_actual}")

    phate_op = phate.PHATE(
        n_components=2,
        knn=knn,
        t=t_value,
        decay=decay,
        knn_dist=knn_dist,
        n_pca=n_pca_actual,
        n_jobs=n_jobs,
        verbose=False,
    )

    phate_coords = phate_op.fit_transform(X)

    # Save coordinates in AnnData
    adata.obsm["X_phate"] = phate_coords

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    unique_classes = sorted(np.unique(superclasses))
    plot_order = ["Control"] + [c for c in unique_classes if c != "Control"]

    palette = sns.color_palette("tab20", n_colors=max(20, len(unique_classes)))
    dynamic_colors = {
        cls: palette[i % len(palette)]
        for i, cls in enumerate(unique_classes)
        if cls not in SUPERCLASS_COLORS
    }

    for cls in plot_order:
        mask = superclasses == cls
        if mask.sum() == 0:
            continue

        n = int(mask.sum())
        color = SUPERCLASS_COLORS.get(cls, dynamic_colors.get(cls, "gray"))

        ax.scatter(
            phate_coords[mask, 0],
            phate_coords[mask, 1],
            s=point_size,
            alpha=alpha,
            label=f"{cls} (n={n:,})",
            c=color,
            edgecolors="none",
            rasterized=True,
        )

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("equal", adjustable="datalim")

    info_text = f"Global PHATE\nFeatures: {X.shape[1]}\nknn={knn}, t={t_value}"
    ax.text(
        0.02,
        0.02,
        info_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="bottom",
        fontstyle="italic",
        color="#555555",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc", alpha=0.85
        ),
    )

    leg = ax.legend(
        loc="upper right",
        markerscale=4,
        fontsize=9,
        frameon=True,
        framealpha=0.9,
        edgecolor="#cccccc",
        handletextpad=0.3,
        borderpad=0.4,
    )
    for lh in leg.legend_handles:
        lh.set_alpha(1.0)

    fig.tight_layout(pad=0.3)

    out_name = f"global_phate_{prefix}" if prefix else "global_phate"
    png_path = os.path.join(out_dir, f"{out_name}.png")
    fig.savefig(
        png_path, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none"
    )
    fig.savefig(
        png_path.replace(".png", ".svg"),
        format="svg",
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close(fig)

    logger.info(f"Saved Global PHATE: {png_path}")
    return phate_coords


# ==============================================================================
# 2. Global PAGA
# ==============================================================================
def plot_global_paga(
    adata: sc.AnnData,
    out_dir: str,
    prefix: str = "",
    n_neighbors: int = 30,
    n_pcs: int = 50,
    dpi: int = 300,
):
    """
    Compute and plot Global PAGA connectivity graph for all classes.
    """
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Running Global PAGA (n_neighbors={n_neighbors}, n_pcs={n_pcs})")

    # Create a clean adata for PAGA to avoid modifying the original heavily
    adata_paga = sc.AnnData(adata.X.copy())
    adata_paga.obs["mutation"] = adata.obs["mutation"].astype("category")

    n_pcs_actual = min(n_pcs, adata_paga.X.shape[1] - 1)
    if n_pcs_actual > 0:
        sc.pp.pca(adata_paga, n_comps=n_pcs_actual)
    else:
        adata_paga.obsm["X_pca"] = adata_paga.X

    sc.pp.neighbors(
        adata_paga,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs_actual if n_pcs_actual > 0 else 0,
    )
    sc.tl.paga(adata_paga, groups="mutation")

    # Log connectivity
    conn = adata_paga.uns["paga"]["connectivities"].toarray()
    groups = adata_paga.obs["mutation"].cat.categories.tolist()

    for i, gi in enumerate(groups):
        for j, gj in enumerate(groups):
            if j > i:
                logger.info(f"  PAGA {gi} <-> {gj}: {conn[i,j]:.4f}")

    # Set colors matching PHATE
    colors = [SUPERCLASS_COLORS.get(g, "#888888") for g in groups]
    adata_paga.uns["mutation_colors"] = colors

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    sc.pl.paga(
        adata_paga,
        color="mutation",
        ax=ax,
        show=False,
        title="",
        fontsize=10,
    )

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0.3)

    out_name = f"global_paga_{prefix}" if prefix else "global_paga"
    png_path = os.path.join(out_dir, f"{out_name}.png")
    fig.savefig(
        png_path, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none"
    )
    fig.savefig(
        png_path.replace(".png", ".svg"),
        format="svg",
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close(fig)

    logger.info(f"Saved Global PAGA: {png_path}")
    return adata_paga.uns["paga"]["connectivities"]
