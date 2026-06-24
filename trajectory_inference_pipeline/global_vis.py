import argparse
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

_IN_COLAB = "google.colab" in sys.modules
if not _IN_COLAB:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scanpy as sc
import seaborn as sns

try:
    import phate as phate_lib
except ImportError:
    phate_lib = None

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trajectory_utils import (SUPERCLASS_COLORS, add_trajectory_arguments,
                              get_logger, load_and_preprocess,
                              save_crosstab_as_svg)

logger = get_logger("global_vis")
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
sns.set_style("ticks")


def get_args():
    p = argparse.ArgumentParser(description="Global Vis: PHATE and PAGA on all cells")
    p = add_trajectory_arguments(p)

    # Global vis specific
    p.add_argument("--phate_knn", type=int, default=5, help="PHATE k-nearest neighbors")
    p.add_argument(
        "--phate_decay", type=int, default=40, help="PHATE alpha decay parameter"
    )
    p.add_argument(
        "--phate_t", type=str, default="auto", help="PHATE t parameter (int or 'auto')"
    )
    p.add_argument(
        "--leiden_resolution",
        type=float,
        default=1.0,
        help="Leiden clustering resolution",
    )
    p.add_argument(
        "--paga_threshold",
        type=float,
        default=0.01,
        help="Min PAGA weight to draw an edge (default: 0.01)",
    )
    p.add_argument(
        "--paga_edge_width_scale",
        type=float,
        default=1.0,
        help="Global scale for PAGA edge widths (default: 1.0)",
    )
    p.add_argument(
        "--paga_min_edge_width",
        type=float,
        default=0.3,
        help="Min edge width in PAGA plot (default: 0.3)",
    )
    p.add_argument(
        "--paga_max_edge_width",
        type=float,
        default=3.0,
        help="Max edge width in PAGA plot (default: 3.0)",
    )
    p.add_argument(
        "--paga_figsize",
        type=float,
        nargs=2,
        default=[6.0, 6.0],
        help="Figure size for PAGA plot (width height)",
    )

    # Plotting arguments
    p.add_argument(
        "--plot_size",
        type=float,
        default=1.5,
        help="Point size for global PHATE scatter",
    )
    p.add_argument(
        "--plot_alpha",
        type=float,
        default=0.3,
        help="Point alpha for global PHATE scatter",
    )

    return p.parse_args()


def plot_phate(phate_coords, labels, output_path, dpi=200, point_size=1.5, alpha=0.3):
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    sns.despine(ax=ax)

    unique_labels = sorted(set(labels))
    plot_order = ["Control"] + [c for c in unique_labels if c != "Control"]

    for cls in plot_order:
        mask = labels == cls
        if mask.sum() == 0:
            continue
        color = SUPERCLASS_COLORS.get(cls, "gray")
        ax.scatter(
            phate_coords[mask, 0],
            phate_coords[mask, 1],
            s=point_size,
            alpha=alpha,
            label=f"{cls} (n={mask.sum():,})",
            c=color,
            edgecolors="none",
            rasterized=True,
        )

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("equal", adjustable="datalim")

    leg = ax.legend(loc="upper right", markerscale=4, fontsize=9, frameon=True)
    for lh in leg.legend_handles:
        lh.set_alpha(1.0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(output_path.replace(".png", ".svg"), format="svg", bbox_inches="tight")
    if _IN_COLAB:
        plt.show()
    plt.close(fig)
    logger.info(f"    Saved global PHATE: {output_path}")


def run_global_vis(args):
    np.random.seed(args.seed)
    X, superclasses, apoptosis, which_layer = load_and_preprocess(args)

    out_dir = args.output_dir or os.path.join(
        os.path.dirname(args.features_cache), "global_vis"
    )
    os.makedirs(out_dir, exist_ok=True)

    prefix = f"global_{args.norm}_{which_layer}"

    # 1. Global PCA
    from sklearn.decomposition import PCA

    n_pca = min(args.pca_dim, X.shape[1], X.shape[0] - 1)
    X_pca = PCA(n_components=n_pca, random_state=args.seed).fit_transform(X)

    # 2. Global PHATE
    if phate_lib is not None:
        logger.info("\n  ── Global PHATE ──")
        phate_t_val = args.phate_t if args.phate_t == "auto" else int(args.phate_t)
        phate_op = phate_lib.PHATE(
            n_components=2,
            knn=args.phate_knn,
            t=phate_t_val,
            decay=args.phate_decay,
            n_jobs=-1,
            random_state=args.seed,
            verbose=0,
        )
        X_phate = phate_op.fit_transform(X_pca)
        phate_out = os.path.join(out_dir, f"phate_{prefix}.png")
        plot_phate(
            X_phate,
            superclasses,
            phate_out,
            dpi=args.dpi,
            point_size=args.plot_size,
            alpha=args.plot_alpha,
        )
    else:
        logger.warning("PHATE not installed. Skipping.")

    # 3. Global PAGA
    logger.info("\n  ── Global PAGA ──")
    adata = sc.AnnData(X_pca.astype(np.float32))
    adata.obsm["X_pca"] = X_pca.astype(np.float32)
    adata.obs["superclass"] = pd.Categorical(superclasses)

    sc.pp.neighbors(adata, n_neighbors=args.n_neighbors, use_rep="X_pca")
    sc.tl.leiden(
        adata,
        resolution=args.leiden_resolution,
        random_state=args.seed,
        flavor="igraph",
        directed=False,
    )

    composition = pd.crosstab(adata.obs["leiden"], adata.obs["superclass"])
    logger.info(f"    Global PAGA Cluster Composition:\n{composition}")

    ct_path = os.path.join(out_dir, f"paga_composition_{prefix}.svg")
    save_crosstab_as_svg(
        composition, ct_path, dpi=args.dpi, title="Global PAGA Cluster Composition"
    )
    logger.info(f"    Saved composition table: {ct_path}")

    sc.tl.paga(adata, groups="leiden")

    # Log PAGA Connectivities
    conn = adata.uns["paga"]["connectivities"].toarray()
    edges = []
    for i in range(conn.shape[0]):
        for j in range(i + 1, conn.shape[1]):
            if conn[i, j] >= args.paga_threshold:
                edges.append((i, j, conn[i, j]))
    if edges:
        edges.sort(key=lambda x: x[2], reverse=True)
        edge_strs = [f"Cluster {u} <-> Cluster {v}: {w:.4f}" for u, v, w in edges]
        logger.info(
            f"    Global PAGA Connectivities (>={args.paga_threshold}):\n      "
            + "\n      ".join(edge_strs)
        )
    else:
        logger.info(
            f"    Global PAGA Connectivities: No edges above threshold {args.paga_threshold}"
        )

    fig, ax = plt.subplots(figsize=tuple(args.paga_figsize))
    sc.pl.paga(
        adata,
        ax=ax,
        show=False,
        title="Global PAGA (Leiden)",
        threshold=args.paga_threshold,
        edge_width_scale=args.paga_edge_width_scale,
        min_edge_width=args.paga_min_edge_width,
        max_edge_width=args.paga_max_edge_width,
    )
    fig.tight_layout()
    paga_out = os.path.join(out_dir, f"paga_leiden_{prefix}.svg")
    fig.savefig(paga_out, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(paga_out.replace(".svg", ".png"), dpi=args.dpi, bbox_inches="tight")
    if _IN_COLAB:
        plt.show()
    plt.close(fig)
    logger.info(f"    Saved global PAGA: {paga_out}")


if __name__ == "__main__":
    args = get_args()
    if not args.norm:
        args.norm = "log_std"  # Default for explicit run
    run_global_vis(args)
