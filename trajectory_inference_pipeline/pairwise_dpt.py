import os
import argparse
import csv
import numpy as np
import pandas as pd
import matplotlib
import sys
_IN_COLAB = "google.colab" in sys.modules
if not _IN_COLAB:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
import scanpy as sc

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trajectory_utils import (
    add_trajectory_arguments, load_and_preprocess, find_root_mnn, find_root_diffmap, 
    find_root_pca, MUTATION_COLORS, get_logger
)

logger = get_logger("pairwise_dpt")
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
sns.set_style("ticks")

def get_args():
    p = argparse.ArgumentParser(description="Pairwise DPT, Correlation, GAM fitting")
    p = add_trajectory_arguments(p)
    p.add_argument("--root_mode", type=str, default="mnn", choices=["pca", "diffmap", "mnn"])
    p.add_argument("--mnn_k", type=int, default=30)
    p.add_argument("--gam_splines", type=int, default=8)
    p.add_argument("--gam_trim_pctl", type=float, nargs=2, default=[5, 95])
    p.add_argument("--no_plot", action="store_true")
    p.add_argument("--de_eval_split", type=float, default=0.5)
    return p.parse_args()


def plot_dpt_scatter(dpt_mut, apop_mut, rho, r_val, mutation, output_path, dpi=200, gam_splines=20, gam_trim_pctl=(1, 99)):
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    
    color = MUTATION_COLORS.get(mutation, "gray")
    if len(dpt_mut) > 0:
        ax.scatter(dpt_mut, apop_mut, s=8, alpha=0.6, c=color, edgecolors="none", zorder=2, rasterized=True, label=mutation)

    gam_dev_expl = 0.0
    pct_lo, pct_hi = np.percentile(dpt_mut, list(gam_trim_pctl))
    dense_mask = (dpt_mut >= pct_lo) & (dpt_mut <= pct_hi)
    dpt_dense = dpt_mut[dense_mask]
    apop_dense = apop_mut[dense_mask]

    x_line = np.linspace(pct_lo, pct_hi, 200)
    try:
        from pygam import LinearGAM, s as s_term
        n_splines = min(gam_splines, max(5, len(dpt_dense) // 50))
        gam = LinearGAM(s_term(0, n_splines=n_splines, spline_order=3)).fit(dpt_dense.reshape(-1, 1), apop_dense)
        y_gam = gam.predict(x_line.reshape(-1, 1))
        ci = gam.confidence_intervals(x_line.reshape(-1, 1), width=0.95)

        ax.plot(x_line, y_gam, "-", color="black", lw=2.5, alpha=0.9, zorder=5)
        ax.fill_between(x_line, ci[:, 0], ci[:, 1], color="black", alpha=0.12, zorder=2, linewidth=0)

        ss_res = np.sum((apop_dense - gam.predict(dpt_dense.reshape(-1, 1))) ** 2)
        ss_tot = np.sum((apop_dense - apop_dense.mean()) ** 2)
        n = len(apop_dense)
        p = gam.statistics_['edof']
        if ss_tot > 0 and n > p + 1:
            gam_dev_expl = 1 - (ss_res / (n - p - 1)) / (ss_tot / (n - 1))
            
    except ImportError:
        if len(dpt_mut) > 2:
            z = np.polyfit(dpt_mut, apop_mut, 1)
            x_line_full = np.linspace(dpt_mut.min(), dpt_mut.max(), 200)
            ax.plot(x_line_full, np.polyval(z, x_line_full), "--", color="black", lw=2, alpha=0.7, zorder=3)

    ax.set_xlabel("Diffusion Pseudotime →", fontsize=12)
    ax.set_ylabel("Apoptosis rate", fontsize=12)
    ax.set_xlim(pct_lo, pct_hi)
    ax.set_xticks([])
    ax.grid(True, alpha=0.2, axis="y")
    
    # Annotate stats
    stats_text = f"Spearman ρ: {rho:.3f}\nPearson r: {r_val:.3f}\nGAM Adj. R²: {gam_dev_expl:.3f}"
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
            ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(output_path.replace(".png", ".svg"), format="svg", bbox_inches="tight")
    if _IN_COLAB: plt.show()
    plt.close(fig)
    
    logger.info(f"    Saved DPT scatter: {output_path}")
    return gam_dev_expl


def compute_gam_r2_only(dpt_vals, apop_vals, gam_splines=20, gam_trim_pctl=(1, 99)):
    try:
        from pygam import LinearGAM, s as s_term
        pct_lo, pct_hi = np.percentile(dpt_vals, list(gam_trim_pctl))
        dense_mask = (dpt_vals >= pct_lo) & (dpt_vals <= pct_hi)
        dpt_dense = dpt_vals[dense_mask]
        apop_dense = apop_vals[dense_mask]
        if len(dpt_dense) < 20: return 0.0
        n_sp = min(gam_splines, max(5, len(dpt_dense) // 50))
        gam = LinearGAM(s_term(0, n_splines=n_sp, spline_order=3)).fit(dpt_dense.reshape(-1, 1), apop_dense)
        ss_res = np.sum((apop_dense - gam.predict(dpt_dense.reshape(-1, 1))) ** 2)
        ss_tot = np.sum((apop_dense - apop_dense.mean()) ** 2)
        n, p = len(apop_dense), gam.statistics_['edof']
        return 1 - (ss_res / (n - p - 1)) / (ss_tot / (n - 1)) if ss_tot > 0 and n > p + 1 else 0.0
    except ImportError: return 0.0


def run_pairwise_dpt(args):
    np.random.seed(args.seed)
    X, superclasses, apoptosis, which_layer = load_and_preprocess(args)
    out_dir = args.output_dir or os.path.join(os.path.dirname(args.features_cache), "pairwise_dpt")
    os.makedirs(out_dir, exist_ok=True)
    
    from sklearn.decomposition import PCA
    n_pca = min(args.pca_dim, X.shape[1], X.shape[0] - 1)
    X_pca = PCA(n_components=n_pca, random_state=args.seed).fit_transform(X)
    
    # Eval Split (if used)
    eval_mask = np.ones(len(superclasses), dtype=bool)
    if args.de_eval_split > 0 and "de" in args.filter_mode:
        rng_split = np.random.RandomState(args.seed)
        for cls in sorted(set(superclasses)):
            cls_idx = np.where(superclasses == cls)[0]
            chosen = rng_split.choice(cls_idx, size=max(1, int(len(cls_idx) * args.de_eval_split)), replace=False)
            eval_mask[chosen] = True # All are eval here since DE was applied before in load_and_preprocess
            # (In original script, DE uses ~eval_mask. Since load_and_preprocess applies DE on all, we just split eval)
            # Actually, to match original, evaluate only on eval_mask (True by default for all if split=0)

    results = []
    dpt_cache = {}

    for mut in ["SNCA", "GBA", "LRRK2"]:
        logger.info(f"\n  ── Pairwise DPT: Control + {mut} ──")
        mut_mask = superclasses == mut
        ctrl_mask = superclasses == "Control"
        pair_mask = ctrl_mask | mut_mask
        if mut_mask.sum() < 10: continue
        
        X_pair_pca = X_pca[pair_mask]
        pair_sc = superclasses[pair_mask]
        pair_apop = apoptosis[pair_mask]
        pair_eval_mask = eval_mask[pair_mask]
        
        adata_pair = sc.AnnData(X_pair_pca.astype(np.float32))
        adata_pair.obsm["X_pca"] = X_pair_pca.astype(np.float32)
        sc.pp.neighbors(adata_pair, n_neighbors=args.n_neighbors, use_rep="X_pca")
        
        n_diffmap_pair = max(min(args.n_diffmap_comps, X_pair_pca.shape[0] - 2), 2)
        n_dcs_pair = max(min(args.n_dcs, n_diffmap_pair), 2)
        sc.tl.diffmap(adata_pair, n_comps=n_diffmap_pair)
        
        diffmap_coords = adata_pair.obsm["X_diffmap"]
        pair_ctrl_mask = pair_sc == "Control"
        ctrl_centroid = diffmap_coords[pair_ctrl_mask].mean(axis=0)
        root_in_pair = np.where(pair_ctrl_mask)[0][np.argmin(np.linalg.norm(diffmap_coords[pair_ctrl_mask] - ctrl_centroid, axis=1))]
        
        adata_pair.uns["iroot"] = int(root_in_pair)
        sc.tl.dpt(adata_pair, n_dcs=n_dcs_pair)
        dpt_pair = adata_pair.obs["dpt_pseudotime"].values
        
        # Save for downstream tools
        dpt_cache[mut] = {
            "dpt": dpt_pair,
            "apop": pair_apop,
            "sc": pair_sc,
            "mask": pair_mask
        }
        
        # Evaluate
        pair_mut_mask = pair_sc == mut
        pair_mut_eval = pair_mut_mask & pair_eval_mask
        
        dpt_mut = dpt_pair[pair_mut_eval]
        apop_mut = pair_apop[pair_mut_eval]
        
        valid = np.isfinite(dpt_mut) & ~np.isnan(apop_mut)
        if valid.sum() < 10: continue
        dpt_v, apop_v = dpt_mut[valid], apop_mut[valid]
        
        rho, pval = spearmanr(dpt_v, apop_v)
        rho = rho if not np.isnan(rho) else 0.0
        r, r_pval = pearsonr(dpt_v, apop_v)
        r = r if not np.isnan(r) else 0.0
        
        logger.info(f"    {mut}: ρ = {rho:.4f} (p={pval:.2e}), r = {r:.4f} (p={r_pval:.2e}), n={valid.sum()}")

        gam_r2 = 0.0
        if not args.no_plot:
            out_path = os.path.join(out_dir, f"dpt_scatter_{args.norm}_{which_layer}_{mut}.png")
            gam_r2 = plot_dpt_scatter(dpt_v, apop_v, rho, r, mut, out_path, dpi=args.dpi, gam_splines=args.gam_splines, gam_trim_pctl=args.gam_trim_pctl)
        else:
            gam_r2 = compute_gam_r2_only(dpt_v, apop_v, gam_splines=args.gam_splines, gam_trim_pctl=args.gam_trim_pctl)
            
        results.append({
            "Mutation": mut,
            "Seed": args.seed,
            "Norm": args.norm,
            "kNN": args.n_neighbors,
            "Features": X.shape[1],
            "PCA": args.pca_dim,
            "rho": rho,
            "r": r,
            "gam_r2": gam_r2,
            "n_valid": valid.sum()
        })
        del adata_pair

    # Save results
    if results:
        csv_path = os.path.join(out_dir, f"dpt_summary_{args.norm}_{which_layer}_seed{args.seed}_pca{args.pca_dim}_k{args.n_neighbors}.csv")
        pd.DataFrame(results).to_csv(csv_path, index=False)
        logger.info(f"  Saved summary: {csv_path}")
        
    if not args.no_plot:
        npz_path = os.path.join(out_dir, f"dpt_results_{args.norm}_{which_layer}.npz")
        np.savez_compressed(npz_path, **dpt_cache)
        logger.info(f"  Saved DPT cache for downstream tools: {npz_path}")

if __name__ == "__main__":
    args = get_args()
    if not args.norm: args.norm = "log_std"
    run_pairwise_dpt(args)
