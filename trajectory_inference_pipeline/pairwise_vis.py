# !python -m trajectory_inference_pipeline.pairwise_vis \
# --features_cache "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/caches_per_image_centering/CNN_seed95_SAE/sae_gap_d8192_lam800_normrestored_withnewclass.npz" \
# --apoptosis_csv "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/세포이미지별 사멸율/이미지별_세포사멸율_7200.csv" \
# --output_dir "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/caches_per_image_centering/pairwise_phate" \
# --phate_decay 100 \
# --n_neighbors 5 \
# --pca_dim 50 \
# --filter_mode "de" \
# --min_cv 0.1 \
#  --de_adj_p 0.05 \
# --de_min_log2fc 1.0 \
# --dead_threshold 1e-5 \
# --norm "log_std" \
# --gap_l2_norm \
# --seed 856 \
#   --phate_decay 120 \
#   --phate_t 35 \
#   --plot_ctrl_size 6 \
#   --plot_ctrl_alpha 0.6 \
#   --plot_mut_size 6 \
#   --plot_ctrl_color "#CCCCCC" \
#   --plot_mut_alpha 0.6 \
#   --plot_invalid_color "#333333" \
# --paga_figsize 2.0 2.0 \
#     --paga_threshold 0.15 \
#     --paga_edge_width_scale 0.6 \
#     --paga_min_edge_width 0.2 \
#     --leiden_resolution 0.4



import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
import sys
_IN_COLAB = "google.colab" in sys.modules
if not _IN_COLAB:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import matplotlib.colors as mcolors
import colorsys
try:
    import phate as phate_lib
except ImportError:
    phate_lib = None

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trajectory_utils import add_trajectory_arguments, load_and_preprocess, SUPERCLASS_COLORS, get_logger, save_crosstab_as_svg

logger = get_logger("pairwise_vis")
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
sns.set_style("ticks")

def get_args():
    p = argparse.ArgumentParser(description="Pairwise Vis: PHATE, PAGA, DiffMap for Ctrl+Mut pairs")
    p = add_trajectory_arguments(p)
    p.add_argument("--phate_knn", type=int, default=5)
    p.add_argument("--phate_decay", type=int, default=40)
    p.add_argument("--phate_t", type=str, default="auto")
    p.add_argument("--leiden_resolution", type=float, default=1.0, help="Leiden clustering resolution (higher = more clusters)")
    
    # PAGA arguments
    p.add_argument("--paga_threshold", type=float, default=0.01, help="Min PAGA weight to draw an edge (default: 0.01)")
    p.add_argument("--paga_edge_width_scale", type=float, default=1.0, help="Global scale for PAGA edge widths (default: 1.0)")
    p.add_argument("--paga_min_edge_width", type=float, default=0.3, help="Min edge width in PAGA plot (default: 0.3)")
    p.add_argument("--paga_max_edge_width", type=float, default=3.0, help="Max edge width in PAGA plot (default: 3.0)")
    p.add_argument("--paga_figsize", type=float, nargs=2, default=[4.0, 4.0], help="Figure size for PAGA plot (width height)")
    
    # Plotting arguments
    p.add_argument("--plot_ctrl_size", type=float, default=7.5)
    p.add_argument("--plot_ctrl_alpha", type=float, default=0.6)
    p.add_argument("--plot_ctrl_color", type=str, default="#FFAB91")
    p.add_argument("--plot_mut_size", type=float, default=7.5)
    p.add_argument("--plot_mut_alpha", type=float, default=0.5)
    p.add_argument("--plot_invalid_size", type=float, default=3.0)
    p.add_argument("--plot_invalid_alpha", type=float, default=0.35)
    p.add_argument("--plot_invalid_color", type=str, default="#cccccc")
    
    return p.parse_args()


def plot_phate_apoptosis_gradient(
    phate_coords: np.ndarray,
    pair_sc: np.ndarray,
    apoptosis_pair: np.ndarray,
    mutation: str,
    out_dir: str,
    prefix: str,
    dpi: int = 200,
    ctrl_size: float = 7.5,
    ctrl_alpha: float = 0.6,
    ctrl_color: str = "#FFAB91",
    mut_size: float = 7.5,
    mut_alpha: float = 0.5,
    invalid_size: float = 3.0,
    invalid_alpha: float = 0.35,
    invalid_color: str = "#cccccc"
):
    ctrl_mask = pair_sc == "Control"
    mut_mask = pair_sc == mutation
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    sns.despine(ax=ax)

    # Control
    ax.scatter(phate_coords[ctrl_mask, 0], phate_coords[ctrl_mask, 1],
               s=ctrl_size, alpha=ctrl_alpha, c=ctrl_color, edgecolors="none", rasterized=True, zorder=1)

    # Mutation
    apop_mut = apoptosis_pair[mut_mask]
    valid_apop = ~np.isnan(apop_mut)
    phate_mut = phate_coords[mut_mask]

    if valid_apop.sum() > 0:
        apop_valid_vals = apop_mut[valid_apop]
        vmin_p = np.percentile(apop_valid_vals, 5)
        vmax_p = np.percentile(apop_valid_vals, 90)
        norm = mcolors.PowerNorm(gamma=3.0, vmin=vmin_p, vmax=vmax_p)
        sort_idx = np.argsort(apop_valid_vals)
        new_orrd = mcolors.LinearSegmentedColormap.from_list('truncated_orrd', plt.cm.YlGnBu(np.linspace(0.1, 1.0, 100)))

        sc_plot = ax.scatter(phate_mut[valid_apop][sort_idx, 0], phate_mut[valid_apop][sort_idx, 1],
                             s=mut_size, alpha=mut_alpha, c=apop_valid_vals[sort_idx], cmap=new_orrd, norm=norm,
                             edgecolors="none", rasterized=True, zorder=3)
        cbar = fig.colorbar(sc_plot, ax=ax, shrink=0.55, aspect=30, pad=0.02)
        cbar.set_label("Apoptosis rate", fontsize=10)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.set_aspect("equal", adjustable="datalim")
    
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"{prefix}_{mutation}.png")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".svg"), format="svg", bbox_inches="tight")
    if _IN_COLAB: plt.show()
    plt.close(fig)
    logger.info(f"    Saved: {out_path}")


def run_pairwise_vis(args):
    np.random.seed(args.seed)
    X, superclasses, apoptosis, which_layer = load_and_preprocess(args)
    out_dir = args.output_dir or os.path.join(os.path.dirname(args.features_cache), "pairwise_vis")
    os.makedirs(out_dir, exist_ok=True)
    
    from sklearn.decomposition import PCA
    n_pca = min(args.pca_dim, X.shape[1], X.shape[0] - 1)
    X_pca = PCA(n_components=n_pca, random_state=args.seed).fit_transform(X)
    
    for mut in ["SNCA", "GBA", "LRRK2"]:
        mut_mask = superclasses == mut
        ctrl_mask = superclasses == "Control"
        pair_mask = ctrl_mask | mut_mask
        if mut_mask.sum() < 10: continue
        
        logger.info(f"\n  ── Pairwise Vis: Control + {mut} ──")
        X_pair_pca = X_pca[pair_mask]
        pair_sc = superclasses[pair_mask]
        pair_apop = apoptosis[pair_mask]
        
        # 1. DiffMap
        adata_pair = sc.AnnData(X_pair_pca.astype(np.float32))
        adata_pair.obsm["X_pca"] = X_pair_pca.astype(np.float32)
        adata_pair.obs["superclass"] = pd.Categorical(pair_sc)
        
        sc.pp.neighbors(adata_pair, n_neighbors=args.n_neighbors, use_rep="X_pca")
        sc.tl.diffmap(adata_pair, n_comps=args.n_diffmap_comps)
        
        diffmap_coords = adata_pair.obsm["X_diffmap"]
        dc_2d = np.column_stack([diffmap_coords[:, 1], diffmap_coords[:, 2]]) # DC1 vs DC2 (0-indexed internally but Scanpy returns from DC1=idx1)
        plot_phate_apoptosis_gradient(
            dc_2d, pair_sc, pair_apop, mutation=mut, out_dir=out_dir, prefix=f"diffmap_{args.norm}_{which_layer}",
            ctrl_size=args.plot_ctrl_size, ctrl_alpha=args.plot_ctrl_alpha, ctrl_color=args.plot_ctrl_color,
            mut_size=args.plot_mut_size, mut_alpha=args.plot_mut_alpha,
            invalid_size=args.plot_invalid_size, invalid_alpha=args.plot_invalid_alpha, invalid_color=args.plot_invalid_color
        )
        
        # 2. PAGA
        sc.tl.leiden(adata_pair, resolution=args.leiden_resolution, random_state=args.seed, flavor='igraph', directed=False)
        
        composition = pd.crosstab(adata_pair.obs["leiden"], adata_pair.obs["superclass"])
        logger.info(f"    PAGA Cluster Composition (Control+{mut}):\n{composition}")
        
        ct_path = os.path.join(out_dir, f"paga_composition_{args.norm}_{which_layer}_{mut}.svg")
        save_crosstab_as_svg(composition, ct_path, dpi=args.dpi, title=f"PAGA Cluster Composition (Control+{mut})")
        logger.info(f"    Saved composition table: {ct_path}")
        
        # Color nodes by dominant class
        dom_class = composition.idxmax(axis=1)
        ctrl_clusters = dom_class[dom_class == "Control"].index.tolist()
        mut_clusters = dom_class[dom_class == mut].index.tolist()
        
        def get_shades(base_color, n_shades):
            if n_shades == 0: return []
            if n_shades == 1: return [base_color]
            rgb = mcolors.to_rgb(base_color)
            h, l, s = colorsys.rgb_to_hls(*rgb)
            l_vals = np.linspace(0.35, 0.85, n_shades)
            return [mcolors.to_hex(colorsys.hls_to_rgb(h, lv, s)) for lv in l_vals]
            
        ctrl_shades = get_shades(SUPERCLASS_COLORS.get("Control", "#808080"), len(ctrl_clusters))
        mut_shades = get_shades(SUPERCLASS_COLORS.get(mut, "#DD8452"), len(mut_clusters))
        
        color_map = {}
        for c, color in zip(ctrl_clusters, ctrl_shades): color_map[c] = color
        for c, color in zip(mut_clusters, mut_shades): color_map[c] = color
        
        leiden_cats = adata_pair.obs["leiden"].cat.categories
        adata_pair.uns["leiden_colors"] = [color_map.get(cat, "#cccccc") for cat in leiden_cats]
        
        sc.tl.paga(adata_pair, groups="leiden")
        
        # Log PAGA Connectivities
        conn = adata_pair.uns['paga']['connectivities'].toarray()
        edges = []
        for i in range(conn.shape[0]):
            for j in range(i + 1, conn.shape[1]):
                if conn[i, j] >= args.paga_threshold:
                    edges.append((i, j, conn[i, j]))
        if edges:
            edges.sort(key=lambda x: x[2], reverse=True)
            edge_strs = [f"Cluster {u} <-> Cluster {v}: {w:.4f}" for u, v, w in edges]
            logger.info(f"    PAGA Connectivities (>={args.paga_threshold}):\n      " + "\n      ".join(edge_strs))
        else:
            logger.info(f"    PAGA Connectivities: No edges above threshold {args.paga_threshold}")
        
        fig, ax = plt.subplots(figsize=tuple(args.paga_figsize))
        sc.pl.paga(
            adata_pair, color="leiden", ax=ax, show=False, title=f"PAGA (Control+{mut})",
            threshold=args.paga_threshold,
            edge_width_scale=args.paga_edge_width_scale,
            min_edge_width=args.paga_min_edge_width,
            max_edge_width=args.paga_max_edge_width
        )
        fig.tight_layout()
        paga_out = os.path.join(out_dir, f"paga_{args.norm}_{which_layer}_{mut}.svg")
        fig.savefig(paga_out, dpi=args.dpi, bbox_inches="tight")
        fig.savefig(paga_out.replace(".svg", ".png"), dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"    Saved PAGA: {paga_out}")

        # 3. PHATE
        if phate_lib is not None:
            phate_t_val = args.phate_t if args.phate_t == "auto" else int(args.phate_t)
            phate_op = phate_lib.PHATE(n_components=2, knn=args.phate_knn, t=phate_t_val, decay=args.phate_decay, n_jobs=-1, random_state=args.seed, verbose=0)
            X_phate_pair = phate_op.fit_transform(X_pair_pca)
            plot_phate_apoptosis_gradient(
                X_phate_pair, pair_sc, pair_apop, mutation=mut, out_dir=out_dir, prefix=f"phate_{args.norm}_{which_layer}",
                ctrl_size=args.plot_ctrl_size, ctrl_alpha=args.plot_ctrl_alpha, ctrl_color=args.plot_ctrl_color,
                mut_size=args.plot_mut_size, mut_alpha=args.plot_mut_alpha,
                invalid_size=args.plot_invalid_size, invalid_alpha=args.plot_invalid_alpha, invalid_color=args.plot_invalid_color
            )
        
        del adata_pair

if __name__ == "__main__":
    args = get_args()
    if not args.norm: args.norm = "log_std"
    run_pairwise_vis(args)
