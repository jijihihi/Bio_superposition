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

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trajectory_utils import add_trajectory_arguments, load_and_preprocess, MUTATION_COLORS, get_logger

logger = get_logger("downstream_stats")
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
sns.set_style("ticks")

def get_args():
    p = argparse.ArgumentParser(description="Downstream DPT Stats: terciles, spread, robustness")
    p = add_trajectory_arguments(p)
    p.add_argument("--root_perturbation_n", type=int, default=10)
    p.add_argument("--n_terciles", type=int, default=3)
    p.add_argument("--permutation_n", type=int, default=10)
    return p.parse_args()


def _compute_cliffs_delta(x, y):
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0: return 0.0, "n/a"
    y_sorted = np.sort(y)
    more, less = 0.0, 0.0
    for xi in x:
        less += np.searchsorted(y_sorted, xi, side='left')
        more += n_y - np.searchsorted(y_sorted, xi, side='right')
    delta = (less - more) / (n_x * n_y)
    ad = abs(delta)
    if ad < 0.147: mag = "negligible"
    elif ad < 0.33: mag = "small"
    elif ad < 0.474: mag = "medium"
    else: mag = "large"
    return delta, mag

def diagnose_dpt_distribution(dpt_all, superclasses_arr, out_dir, prefix, dpi=200):
    mutations = ["SNCA", "GBA", "LRRK2"]
    diagnostics = {}
    logger.info(f"\n    ── DPT Distribution Diagnostics ──")
    for grp in ["Control"] + mutations:
        mask = superclasses_arr == grp
        if mask.sum() < 5: continue
        dpt_grp = dpt_all[mask]
        valid = np.isfinite(dpt_grp)
        n_valid, n_inf = valid.sum(), np.isinf(dpt_grp).sum()
        if n_valid < 2: continue
        dpt_v = dpt_grp[valid]
        spread = np.percentile(dpt_v, 95) - np.percentile(dpt_v, 5)
        
        logger.info(f"    {grp:>10s}: n={n_valid}, inf={n_inf}, spread={spread:.4f}")
        
        fig, ax = plt.subplots(figsize=(8, 4))
        c = "#CCCCCC" if grp == "Control" else MUTATION_COLORS.get(grp, "#888")
        ax.hist(dpt_v, bins=60, alpha=0.7, color=c, edgecolor="black", linewidth=0.3)
        ax.set_title(f"{grp} DPT Distribution (spread={spread:.4f})")
        fig.tight_layout()
        sns.despine()
        fig.savefig(os.path.join(out_dir, f"dpt_dist_{grp}_{prefix}.svg"), dpi=dpi, bbox_inches="tight")
        if _IN_COLAB: plt.show()
        plt.close(fig)
    return diagnostics


def plot_jt_terciles(dpt_vals, apop_vals, mutation, output_path, dpi=200, n_terciles=3):
    n_terciles_jt = n_terciles
    p5, p95 = np.percentile(dpt_vals, [5, 95])
    tercile_edges = np.linspace(p5, p95, n_terciles_jt + 1)
    tercile_labels = ["Proximal", "Intermediate", "Distal"] if n_terciles_jt == 3 else [f"Q{i+1}" for i in range(n_terciles_jt)]
    
    group_data = []
    for i in range(n_terciles_jt):
        if i == n_terciles_jt - 1: mask_t = (dpt_vals >= tercile_edges[i]) & (dpt_vals <= tercile_edges[i + 1])
        else: mask_t = (dpt_vals >= tercile_edges[i]) & (dpt_vals < tercile_edges[i + 1])
        group_data.append(apop_vals[mask_t])

    # JT Test
    jt_p = np.nan
    try:
        jt_stat = 0.0
        for i in range(len(group_data)):
            for j in range(i + 1, len(group_data)):
                gi, gj = group_data[i], np.sort(group_data[j])
                idx_right = np.searchsorted(gj, gi, side='right')
                idx_left = np.searchsorted(gj, gi, side='left')
                n_less = idx_left  # count how many elements in gj are strictly LESS than gi
                n_equal = idx_right - idx_left
                jt_stat += float(np.sum(n_less) + 0.5 * np.sum(n_equal))
        ns = [len(g) for g in group_data if len(g) > 0]
        N = sum(ns)
        mu_jt = (N**2 - sum(n**2 for n in ns)) / 4
        all_vals_jt = np.concatenate([g for g in group_data if len(g) > 0])
        _, tie_counts = np.unique(all_vals_jt, return_counts=True)
        tie_term = sum(t * (t - 1) * (2 * t + 5) for t in tie_counts if t > 1)
        var_jt = ((N**2 * (2*N + 3) - sum(n**2 * (2*n + 3) for n in ns)) / 72 - tie_term / 72)
        z_jt = (jt_stat - mu_jt - 0.5) / np.sqrt(max(var_jt, 1e-12)) if jt_stat > mu_jt else (jt_stat - mu_jt + 0.5) / np.sqrt(max(var_jt, 1e-12))
        from scipy.stats import norm
        jt_p = norm.sf(z_jt)
        logger.info(f"    JT test: p = {jt_p:.6e}")
    except Exception as e:
        logger.warning(f"    JT test failed: {e}")

    # Dunn's post hoc
    from scipy.stats import mannwhitneyu
    dunn_results = {}
    comparisons = [(0, 1, "Proximal", "Intermediate"), (1, 2, "Intermediate", "Distal"), (0, 2, "Proximal", "Distal")] if n_terciles_jt == 3 else [(i, i+1, f"Q{i+1}", f"Q{i+2}") for i in range(n_terciles_jt - 1)]
    n_comp = len(comparisons)
    
    for i_grp, j_grp, li, lj in comparisons:
        gi, gj = group_data[i_grp], group_data[j_grp]
        if len(gi) < 2 or len(gj) < 2: continue
        try: _, p_raw = mannwhitneyu(gi, gj, alternative="two-sided")
        except ValueError: p_raw = 1.0
        p_adj = min(p_raw * n_comp, 1.0)
        delta, mag = _compute_cliffs_delta(gi, gj)
        
        if p_adj < 0.001: sig = "***"
        elif p_adj < 0.01: sig = "**"
        elif p_adj < 0.05: sig = "*"
        else: sig = "ns"
        
        dunn_results[f"{li}_vs_{lj}"] = {"sig": sig, "p_adj": p_adj, "delta": delta}
        logger.info(f"    {li} vs {lj}: p_adj={p_adj:.2e} ({sig}), δ={delta:.3f}")

    # Plot
    fig_jt, ax_jt = plt.subplots(figsize=(5, 4))
    
    # Generate shades of the base mutation color
    base_color = MUTATION_COLORS.get(mutation, "#DD8452")
    import matplotlib.colors as mcolors
    import colorsys
    rgb = mcolors.to_rgb(base_color)
    h, l, s = colorsys.rgb_to_hls(*rgb)
    l_vals = np.linspace(0.85, 0.4, n_terciles_jt) # light to dark
    colors_tercile = [mcolors.to_hex(colorsys.hls_to_rgb(h, lv, s)) for lv in l_vals]
    
    means = [g.mean() if len(g)>0 else 0 for g in group_data]
    sems = [g.std() / np.sqrt(len(g)) if len(g)>0 else 0 for g in group_data]
    x_pos = np.arange(n_terciles_jt)
    
    ax_jt.bar(x_pos, means, yerr=sems, capsize=6, color=colors_tercile[:n_terciles_jt], alpha=0.7, edgecolor="black", linewidth=0.8)
    ax_jt.set_xticks(x_pos)
    ax_jt.set_xticklabels(tercile_labels, fontsize=11)
    ax_jt.set_xlabel("DPT stage", fontsize=12)
    ax_jt.set_ylabel("Mean apoptosis rate", fontsize=12)
    ax_jt.set_title(f"Apoptosis by DPT Tercile ({mutation})")
    
    bar_tops = [means[k] + sems[k] for k in range(n_terciles_jt)]
    y_max = max(bar_tops) if bar_tops else 0
    bracket_y = y_max + y_max * 0.08
    
    for ci, (i_grp, j_grp, li, lj) in enumerate(comparisons):
        sig = dunn_results.get(f"{li}_vs_{lj}", {}).get("sig", "ns")
        if sig == "ns": continue
        x1, x2 = x_pos[i_grp], x_pos[j_grp]
        y_b = bracket_y + ci * (y_max * 0.08) * 1.5
        h = (y_max * 0.08) * 0.3
        ax_jt.plot([x1, x1, x2, x2], [y_b - h, y_b, y_b, y_b - h], lw=1.2, color="black")
        ax_jt.text((x1 + x2) / 2, y_b + h * 0.3, sig, ha="center", va="bottom", fontsize=10, fontweight="bold")
        
    if not np.isnan(jt_p):
        ax_jt.text(0.5, 0.97, f"JT p = {jt_p:.2e}", transform=ax_jt.transAxes, ha="center", va="top", fontsize=10, fontstyle="italic")

    fig_jt.tight_layout()
    sns.despine()
    fig_jt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    fig_jt.savefig(output_path.replace(".png", ".svg"), format="svg", bbox_inches="tight")
    if _IN_COLAB: plt.show()
    plt.close(fig_jt)
    logger.info(f"    Saved JT Terciles: {output_path}")

def run_downstream_stats(args):
    np.random.seed(args.seed)
    X, superclasses, apoptosis, which_layer = load_and_preprocess(args)
    out_dir = args.output_dir or os.path.join(os.path.dirname(args.features_cache), "downstream_stats")
    os.makedirs(out_dir, exist_ok=True)
    
    from sklearn.decomposition import PCA
    n_pca = min(args.pca_dim, X.shape[1], X.shape[0] - 1)
    X_pca = PCA(n_components=n_pca, random_state=args.seed).fit_transform(X)

    for mut in ["SNCA", "GBA", "LRRK2"]:
        logger.info(f"\n  ── Downstream Stats: Control + {mut} ──")
        mut_mask = superclasses == mut
        ctrl_mask = superclasses == "Control"
        pair_mask = ctrl_mask | mut_mask
        if mut_mask.sum() < 10: continue
        
        X_pair_pca = X_pca[pair_mask]
        pair_sc = superclasses[pair_mask]
        pair_apop = apoptosis[pair_mask]
        
        adata_pair = sc.AnnData(X_pair_pca.astype(np.float32))
        adata_pair.obsm["X_pca"] = X_pair_pca.astype(np.float32)
        sc.pp.neighbors(adata_pair, n_neighbors=args.n_neighbors, use_rep="X_pca")
        
        n_diffmap_pair = max(min(args.n_diffmap_comps, X_pair_pca.shape[0] - 2), 2)
        sc.tl.diffmap(adata_pair, n_comps=n_diffmap_pair)
        
        diffmap_coords = adata_pair.obsm["X_diffmap"]
        pair_ctrl_mask = pair_sc == "Control"
        ctrl_centroid = diffmap_coords[pair_ctrl_mask].mean(axis=0)
        root_in_pair = np.where(pair_ctrl_mask)[0][np.argmin(np.linalg.norm(diffmap_coords[pair_ctrl_mask] - ctrl_centroid, axis=1))]
        
        adata_pair.uns["iroot"] = int(root_in_pair)
        sc.tl.dpt(adata_pair, n_dcs=max(min(args.n_dcs, n_diffmap_pair), 2))
        dpt_pair = adata_pair.obs["dpt_pseudotime"].values
        
        # Diagnostics
        diagnose_dpt_distribution(dpt_pair, pair_sc, out_dir, prefix=f"{args.norm}_{which_layer}_{mut}", dpi=args.dpi)
        
        # Terciles & JT
        pair_mut_mask = pair_sc == mut
        dpt_mut = dpt_pair[pair_mut_mask]
        apop_mut = pair_apop[pair_mut_mask]
        valid = np.isfinite(dpt_mut) & ~np.isnan(apop_mut)
        if valid.sum() > 10:
            jt_path = os.path.join(out_dir, f"terciles_jt_{args.norm}_{which_layer}_{mut}.png")
            plot_jt_terciles(dpt_mut[valid], apop_mut[valid], mut, jt_path, dpi=args.dpi, n_terciles=args.n_terciles)
            
        del adata_pair

if __name__ == "__main__":
    args = get_args()
    if not args.norm: args.norm = "log_std"
    run_downstream_stats(args)
