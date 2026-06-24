import argparse
import csv
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

_IN_COLAB = "google.colab" in sys.modules
if not _IN_COLAB:
    matplotlib.use("Agg")
import sys

import matplotlib.pyplot as plt
import scanpy as sc
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trajectory_utils import (MUTATION_COLORS, add_trajectory_arguments,
                              find_root_diffmap, find_root_mnn, find_root_pca,
                              get_logger, load_and_preprocess,
                              save_args_to_json)

logger = get_logger("pairwise_dpt")
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
sns.set_style("ticks")


def get_args():
    p = argparse.ArgumentParser(description="Pairwise DPT, Correlation, GAM fitting")
    p = add_trajectory_arguments(p)
    p.add_argument(
        "--root_mode", type=str, default="mnn", choices=["pca", "diffmap", "mnn"]
    )
    p.add_argument("--mnn_k", type=int, default=30)
    p.add_argument(
        "--gpu", action="store_true", help="Use GPU for DPT via rapids-singlecell"
    )
    p.add_argument("--gam_splines", type=int, default=8)
    p.add_argument("--gam_trim_pctl", type=float, nargs=2, default=[5, 95])
    p.add_argument("--no_plot", action="store_true")
    p.add_argument("--de_eval_split", type=float, default=0.5)
    p.add_argument(
        "--dpt_window_size",
        type=float,
        default=0.1,
        help="Width of sliding window in DPT space",
    )
    p.add_argument(
        "--dpt_step", type=float, default=0.02, help="Step size for DPT sliding window"
    )
    p.add_argument(
        "--density_samples",
        type=int,
        default=1000,
        help="Fixed number of cells to sample per DPT window",
    )
    p.add_argument(
        "--erank_pca_dim",
        type=int,
        default=0,
        help="If > 0, apply PCA before ERank sliding window to remove high-dim noise",
    )
    return p.parse_args()


def plot_dpt_scatter(
    dpt_mut,
    apop_mut,
    rho,
    r_val,
    mutation,
    output_path,
    dpi=200,
    gam_splines=20,
    gam_trim_pctl=(1, 99),
):
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    color = MUTATION_COLORS.get(mutation, "gray")
    if len(dpt_mut) > 0:
        ax.scatter(
            dpt_mut,
            apop_mut,
            s=8,
            alpha=0.8,
            c=color,
            edgecolors="none",
            zorder=2,
            rasterized=True,
            label=mutation,
        )

    gam_dev_expl = 0.0
    pct_lo, pct_hi = np.percentile(dpt_mut, list(gam_trim_pctl))
    dense_mask = (dpt_mut >= pct_lo) & (dpt_mut <= pct_hi)
    dpt_dense = dpt_mut[dense_mask]
    apop_dense = apop_mut[dense_mask]

    x_line = np.linspace(pct_lo, pct_hi, 200)
    try:
        from pygam import LinearGAM
        from pygam import s as s_term

        n_splines = min(gam_splines, max(5, len(dpt_dense) // 50))
        gam = LinearGAM(s_term(0, n_splines=n_splines, spline_order=3)).fit(
            dpt_dense.reshape(-1, 1), apop_dense
        )
        y_gam = gam.predict(x_line.reshape(-1, 1))
        ci = gam.confidence_intervals(x_line.reshape(-1, 1), width=0.95)

        ax.plot(x_line, y_gam, "-", color="black", lw=2.5, alpha=0.9, zorder=5)
        ax.fill_between(
            x_line, ci[:, 0], ci[:, 1], color="black", alpha=0.12, zorder=2, linewidth=0
        )

        ss_res = np.sum((apop_dense - gam.predict(dpt_dense.reshape(-1, 1))) ** 2)
        ss_tot = np.sum((apop_dense - apop_dense.mean()) ** 2)
        n = len(apop_dense)
        p = gam.statistics_["edof"]
        if ss_tot > 0 and n > p + 1:
            gam_dev_expl = 1 - (ss_res / (n - p - 1)) / (ss_tot / (n - 1))

    except ImportError:
        if len(dpt_mut) > 2:
            z = np.polyfit(dpt_mut, apop_mut, 1)
            x_line_full = np.linspace(dpt_mut.min(), dpt_mut.max(), 200)
            ax.plot(
                x_line_full,
                np.polyval(z, x_line_full),
                "--",
                color="black",
                lw=2,
                alpha=0.7,
                zorder=3,
            )

    ax.set_xlabel("Diffusion Pseudotime →", fontsize=12)
    ax.set_ylabel("Apoptosis rate", fontsize=12)
    ax.set_xlim(pct_lo, pct_hi)
    ax.set_xticks([])
    ax.grid(True, alpha=0.2, axis="y")

    # Annotate stats
    stats_text = f"Spearman ρ: {rho:.3f}\nPearson r: {r_val:.3f}\nGAM Adj. R²: {gam_dev_expl:.3f}"
    ax.text(
        0.95,
        0.95,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.5", facecolor="white", alpha=0.8, edgecolor="gray"
        ),
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(output_path.replace(".png", ".svg"), format="svg", bbox_inches="tight")
    if _IN_COLAB:
        plt.show()
    plt.close(fig)

    logger.info(f"    Saved DPT scatter: {output_path}")
    return gam_dev_expl


def compute_gam_r2_only(dpt_vals, apop_vals, gam_splines=20, gam_trim_pctl=(1, 99)):
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
        n, p = len(apop_dense), gam.statistics_["edof"]
        return (
            1 - (ss_res / (n - p - 1)) / (ss_tot / (n - 1))
            if ss_tot > 0 and n > p + 1
            else 0.0
        )
    except ImportError:
        return 0.0


def compute_twonn_id(X):
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    if len(X) < 3:
        return 0.0
    nn = NearestNeighbors(n_neighbors=3, algorithm="auto").fit(X)
    distances, _ = nn.kneighbors(X)
    r1 = distances[:, 1]
    r2 = distances[:, 2]
    valid = r1 > 1e-12
    if valid.sum() < 3:
        return 0.0
    r1, r2 = r1[valid], r2[valid]
    mu = r2 / r1
    mu_sorted = np.sort(mu)
    N = len(mu_sorted)
    F_mu = np.arange(1, N + 1) / N
    mu_sorted = mu_sorted[:-1]
    F_mu = F_mu[:-1]
    x = np.log(mu_sorted)
    y = -np.log(1 - F_mu)
    d = np.dot(x, y) / np.dot(x, x)
    return float(d)


def compute_sliding_metrics(
    X_mut,
    dpt_mut,
    apop_mut,
    dpt_window=0.1,
    dpt_step=0.02,
    density_samples=1000,
    top_k_sv=0,
    seed=42,
):
    rng = np.random.RandomState(seed)
    eranks, twonns, apops, dpts = [], [], [], []
    eranks_perm, twonns_perm = [], []

    min_dpt = np.nanmin(dpt_mut)
    max_dpt = np.nanmax(dpt_mut)

    starts = np.arange(min_dpt, max_dpt - dpt_window + 1e-5, dpt_step)

    for start in starts:
        end = start + dpt_window

        # Find cells within this DPT range
        mask = (dpt_mut >= start) & (dpt_mut < end)
        idx_in_window = np.where(mask)[0]

        if len(idx_in_window) < density_samples:
            # Skip this window because it lacks the required density
            continue

        # Downsample to exactly `density_samples`
        chosen_idx = rng.choice(idx_in_window, size=density_samples, replace=False)

        X_win = X_mut[chosen_idx]
        apop_win = apop_mut[chosen_idx]
        dpt_win = dpt_mut[chosen_idx]

        # Center the window
        X_win_centered = X_win - X_win.mean(axis=0)
        s = np.linalg.svd(X_win_centered, compute_uv=False)
        s = s[s > 1e-12]

        if top_k_sv > 0:
            s = s[:top_k_sv]

        if len(s) > 0:
            p = s / s.sum()
            erank = np.exp(-np.sum(p * np.log(p)))
            eranks.append(erank)

            twonn = compute_twonn_id(X_win)
            twonns.append(twonn)

            apops.append(
                np.nanmean(apop_win) if not np.isnan(apop_win).all() else np.nan
            )
            dpts.append(dpt_win.mean())

            # Permutation Control (random cells from the entire manifold)
            idx_perm = rng.choice(len(X_mut), size=density_samples, replace=False)
            X_win_perm = X_mut[idx_perm]

            X_win_perm_centered = X_win_perm - X_win_perm.mean(axis=0)
            s_perm = np.linalg.svd(X_win_perm_centered, compute_uv=False)
            s_perm = s_perm[s_perm > 1e-12]
            if top_k_sv > 0:
                s_perm = s_perm[:top_k_sv]
            if len(s_perm) > 0:
                p_perm = s_perm / s_perm.sum()
                eranks_perm.append(np.exp(-np.sum(p_perm * np.log(p_perm))))
            else:
                eranks_perm.append(np.nan)

            twonns_perm.append(compute_twonn_id(X_win_perm))

    return (
        np.array(dpts),
        np.array(eranks),
        np.array(twonns),
        np.array(apops),
        np.array(eranks_perm),
        np.array(twonns_perm),
    )


def plot_sliding_metrics(
    dpts,
    eranks,
    twonns,
    apops,
    eranks_perm,
    twonns_perm,
    mutation,
    output_path,
    dpi=200,
):
    if len(dpts) < 3:
        logger.warning(f"Not enough sliding windows for {mutation} to plot metrics.")
        return 0.0, 0.0, 0.0, 0.0

    fig, axes = plt.subplots(1, 5, figsize=(25, 4.5))
    color = MUTATION_COLORS.get(mutation, "gray")

    # 1. DPT vs ERank
    ax = axes[0]
    ax.scatter(dpts, eranks_perm, color="gray", alpha=0.3, s=15, label="Permuted Null")
    ax.scatter(dpts, eranks, color=color, alpha=0.85, s=30, label="Real")
    rho_de, p_de = spearmanr(dpts, eranks)
    rho_de = rho_de if not np.isnan(rho_de) else 0.0
    ax.set_title(f"DPT vs ERank (ρ={rho_de:.2f}, p={p_de:.2e})", fontsize=11)
    ax.set_xlabel("Mean Pseudotime (Window)", fontsize=10)
    ax.set_ylabel("Effective Rank", fontsize=10)
    ax.legend(fontsize=8)

    try:
        from pygam import LinearGAM
        from pygam import s as s_term

        gam_e = LinearGAM(s_term(0, n_splines=min(8, max(4, len(dpts) // 5)))).fit(
            dpts.reshape(-1, 1), eranks
        )
        x_line = np.linspace(dpts.min(), dpts.max(), 100)
        gam_e_pred = gam_e.predict(x_line)
        ax.plot(x_line, gam_e_pred, color="black", lw=2)

        # GAM for Permuted
        gam_ep = LinearGAM(s_term(0, n_splines=min(8, max(4, len(dpts) // 5)))).fit(
            dpts.reshape(-1, 1), eranks_perm
        )
        ax.plot(x_line, gam_ep.predict(x_line), "--", color="gray", lw=1.5)
    except:
        gam_e = None
        gam_e_pred = None
        x_line = np.linspace(dpts.min(), dpts.max(), 100)

    # 2. DPT vs Two-NN
    ax = axes[1]
    ax.scatter(dpts, twonns_perm, color="gray", alpha=0.3, s=15, label="Permuted Null")
    ax.scatter(dpts, twonns, color="#2CA02C", alpha=0.85, s=30, label="Real")
    rho_dt, p_dt = spearmanr(dpts, twonns)
    rho_dt = rho_dt if not np.isnan(rho_dt) else 0.0
    ax.set_title(f"DPT vs Two-NN (ρ={rho_dt:.2f}, p={p_dt:.2e})", fontsize=11)
    ax.set_xlabel("Mean Pseudotime (Window)", fontsize=10)
    ax.set_ylabel("Two-NN Intrinsic Dim", fontsize=10)
    ax.legend(fontsize=8)

    try:
        gam_t = LinearGAM(s_term(0, n_splines=min(8, max(4, len(dpts) // 5)))).fit(
            dpts.reshape(-1, 1), twonns
        )
        gam_t_pred = gam_t.predict(x_line)
        ax.plot(x_line, gam_t_pred, color="black", lw=2)

        # GAM for Permuted
        gam_tp = LinearGAM(s_term(0, n_splines=min(8, max(4, len(dpts) // 5)))).fit(
            dpts.reshape(-1, 1), twonns_perm
        )
        ax.plot(x_line, gam_tp.predict(x_line), "--", color="gray", lw=1.5)
    except:
        gam_t = None
        gam_t_pred = None

    # 3. DPT vs Apoptosis
    ax = axes[2]
    ax.scatter(dpts, apops, color=color, alpha=0.85, s=30)
    rho_da, p_da = spearmanr(dpts, apops)
    rho_da = rho_da if not np.isnan(rho_da) else 0.0
    ax.set_title(f"DPT vs Apoptosis (ρ={rho_da:.2f}, p={p_da:.2e})", fontsize=11)
    ax.set_xlabel("Mean Pseudotime (Window)", fontsize=10)
    ax.set_ylabel("Mean Apoptosis", fontsize=10)

    try:
        gam_a = LinearGAM(s_term(0, n_splines=min(8, max(4, len(dpts) // 5)))).fit(
            dpts.reshape(-1, 1), apops
        )
        gam_a_pred = gam_a.predict(x_line)
        ax.plot(x_line, gam_a_pred, color="black", lw=2)
    except:
        gam_a = None
        gam_a_pred = None

    # 4. ERank vs Apoptosis
    ax = axes[3]
    ax.scatter(eranks, apops, color=color, alpha=0.85, s=30)
    rho_ea, p_ea = spearmanr(eranks, apops)
    rho_ea = rho_ea if not np.isnan(rho_ea) else 0.0
    ax.set_title(f"ERank vs Apoptosis (ρ={rho_ea:.2f}, p={p_ea:.2e})", fontsize=11)
    ax.set_xlabel("Effective Rank", fontsize=10)
    ax.set_ylabel("Mean Apoptosis", fontsize=10)

    if len(eranks) > 1 and np.std(eranks) > 0:
        z = np.polyfit(eranks, apops, 1)
        x_line_e = np.linspace(eranks.min(), eranks.max(), 100)
        ax.plot(x_line_e, np.polyval(z, x_line_e), "--", color="black", lw=2)

    # 5. Combined Plot (DPT vs ERank, Two-NN & Apoptosis)
    ax_c = axes[4]
    ax_c.set_title(f"DPT Trajectory Overlay", fontsize=11)
    ax_c.set_xlabel("Mean Pseudotime (Window)", fontsize=10)

    ax_erank = ax_c.twinx()

    # Plot Apoptosis on left Y
    ax_c.scatter(dpts, apops, color="#DD8452", alpha=0.5, s=15)
    ax_c.set_ylabel("Mean Apoptosis", color="#DD8452", fontsize=10, fontweight="bold")
    ax_c.tick_params(axis="y", labelcolor="#DD8452")
    if gam_a is not None:
        ax_c.plot(x_line, gam_a_pred, color="#A55628", lw=2.5, label="Apoptosis (GAM)")

    # Plot ERank and Two-NN on right Y
    ax_erank.scatter(dpts, eranks_perm, color="gray", alpha=0.3, s=10)
    ax_erank.scatter(dpts, twonns_perm, color="gray", alpha=0.3, s=10)

    ax_erank.scatter(dpts, eranks, color="#4C72B0", alpha=0.5, s=15)
    ax_erank.scatter(dpts, twonns, color="#2CA02C", alpha=0.5, s=15)
    ax_erank.set_ylabel(
        "Intrinsic Dim (ERank / Two-NN)",
        color="#4C72B0",
        fontsize=10,
        fontweight="bold",
    )
    ax_erank.tick_params(axis="y", labelcolor="#4C72B0")
    if gam_e is not None:
        ax_erank.plot(x_line, gam_e_pred, color="#2B4B80", lw=2.5, label="ERank (GAM)")
    if gam_t is not None:
        ax_erank.plot(x_line, gam_t_pred, color="#1A601A", lw=2.5, label="Two-NN (GAM)")

    lines_1, labels_1 = ax_c.get_legend_handles_labels()
    lines_2, labels_2 = ax_erank.get_legend_handles_labels()
    if lines_1 or lines_2:
        ax_c.legend(
            lines_1 + lines_2,
            labels_1 + labels_2,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=2,
            fontsize=9,
        )

    for idx, ax in enumerate(axes):
        ax.grid(True, alpha=0.2)
        if idx < 4:
            sns.despine(ax=ax)
        else:
            sns.despine(ax=ax, right=False)  # Keep right spine for twinx

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(output_path.replace(".png", ".svg"), format="svg", bbox_inches="tight")
    if _IN_COLAB:
        plt.show()
    plt.close(fig)
    return rho_de, rho_dt, rho_da, rho_ea


def run_pairwise_dpt(args):
    np.random.seed(args.seed)
    X, superclasses, apoptosis, which_layer, X_log = load_and_preprocess(args)
    out_dir = args.output_dir or os.path.join(
        os.path.dirname(args.features_cache), "pairwise_dpt"
    )
    os.makedirs(out_dir, exist_ok=True)

    if args.de_mode != "per_mut":
        from sklearn.decomposition import PCA

        n_pca = min(args.pca_dim, X.shape[1], X.shape[0] - 1)
        X_pca_global = PCA(n_components=n_pca, random_state=args.seed).fit_transform(X)
    else:
        X_pca_global = None

    # Eval Split (if used)
    eval_mask = np.ones(len(superclasses), dtype=bool)
    if args.de_eval_split > 0 and "de" in args.filter_mode:
        rng_split = np.random.RandomState(args.seed)
        for cls in sorted(set(superclasses)):
            cls_idx = np.where(superclasses == cls)[0]
            chosen = rng_split.choice(
                cls_idx,
                size=max(1, int(len(cls_idx) * args.de_eval_split)),
                replace=False,
            )
            eval_mask[chosen] = True

    results = []
    dpt_cache = {}

    for mut in ["SNCA", "GBA", "LRRK2"]:
        logger.info(f"\n  ── Pairwise DPT: Control + {mut} ──")
        mut_mask = superclasses == mut
        ctrl_mask = superclasses == "Control"
        pair_mask = ctrl_mask | mut_mask
        if mut_mask.sum() < 10:
            continue

        pair_sc = superclasses[pair_mask]
        pair_apop = apoptosis[pair_mask]
        pair_eval_mask = eval_mask[pair_mask]

        if args.de_mode == "per_mut":
            X_pair = X[pair_mask]
            if "de" in args.filter_mode:
                from trajectory_utils import compute_de_neurons

                res = compute_de_neurons(
                    X_pair, pair_sc, mut, args.de_adj_p, args.de_min_log2fc
                )
                m = res["mask"]
                if args.de_top_k > 0 and m.sum() > args.de_top_k:
                    sig_idx = np.where(m)[0]
                    top_k = sig_idx[
                        np.argsort(np.abs(res["log2fc"][sig_idx]))[::-1][
                            : args.de_top_k
                        ]
                    ]
                    m = np.zeros_like(m)
                    m[top_k] = True
                X_pair = X_pair[:, m]
                logger.info(
                    f"    [per_mut DE] Used {m.sum()} neurons for {mut} vs Control"
                )

            from sklearn.decomposition import PCA

            n_pca_pair = min(args.pca_dim, X_pair.shape[1], X_pair.shape[0] - 1)
            X_pair_pca = PCA(
                n_components=n_pca_pair, random_state=args.seed
            ).fit_transform(X_pair)
        else:
            X_pair_pca = X_pca_global[pair_mask]

        adata_pair = sc.AnnData(X_pair_pca.astype(np.float32))
        adata_pair.obsm["X_pca"] = X_pair_pca.astype(np.float32)

        n_diffmap_pair = max(min(args.n_diffmap_comps, X_pair_pca.shape[0] - 2), 2)
        n_dcs_pair = max(min(args.n_dcs, n_diffmap_pair), 2)

        if getattr(args, "gpu", False):
            try:
                import rapids_singlecell as rsc

                rsc.get.anndata_to_GPU(adata_pair)
                rsc.pp.neighbors(
                    adata_pair, n_neighbors=args.n_neighbors, use_rep="X_pca"
                )
                rsc.tl.diffmap(adata_pair, n_comps=n_diffmap_pair)
                rsc.get.anndata_to_CPU(adata_pair)
            except ImportError:
                logger.warning(
                    "rapids_singlecell not found. Falling back to CPU scanpy."
                )
                sc.pp.neighbors(
                    adata_pair, n_neighbors=args.n_neighbors, use_rep="X_pca"
                )
                sc.tl.diffmap(adata_pair, n_comps=n_diffmap_pair)
        else:
            sc.pp.neighbors(adata_pair, n_neighbors=args.n_neighbors, use_rep="X_pca")
            sc.tl.diffmap(adata_pair, n_comps=n_diffmap_pair)

        diffmap_coords = adata_pair.obsm["X_diffmap"]
        pair_ctrl_mask = pair_sc == "Control"
        ctrl_centroid = diffmap_coords[pair_ctrl_mask].mean(axis=0)
        root_in_pair = np.where(pair_ctrl_mask)[0][
            np.argmin(
                np.linalg.norm(diffmap_coords[pair_ctrl_mask] - ctrl_centroid, axis=1)
            )
        ]

        adata_pair.uns["iroot"] = int(root_in_pair)

        sc.tl.dpt(adata_pair, n_dcs=n_dcs_pair)

        dpt_pair = adata_pair.obs["dpt_pseudotime"].values

        # Save for downstream tools
        dpt_cache[mut] = {
            "dpt": dpt_pair,
            "apop": pair_apop,
            "sc": pair_sc,
            "mask": pair_mask,
        }

        # Evaluate
        pair_mut_mask = pair_sc == mut
        pair_mut_eval = pair_mut_mask & pair_eval_mask

        dpt_mut = dpt_pair[pair_mut_eval]
        apop_mut = pair_apop[pair_mut_eval]

        valid = np.isfinite(dpt_mut) & ~np.isnan(apop_mut)
        if valid.sum() < 10:
            continue
        dpt_v, apop_v = dpt_mut[valid], apop_mut[valid]

        rho, pval = spearmanr(dpt_v, apop_v)
        rho = rho if not np.isnan(rho) else 0.0
        r, r_pval = pearsonr(dpt_v, apop_v)
        r = r if not np.isnan(r) else 0.0

        logger.info(
            f"    {mut}: ρ = {rho:.4f} (p={pval:.2e}), r = {r:.4f} (p={r_pval:.2e}), n={valid.sum()}"
        )

        gam_r2 = 0.0
        rho_de, rho_da, rho_ea = 0.0, 0.0, 0.0

        if not args.no_plot:
            out_path = os.path.join(
                out_dir, f"dpt_scatter_{args.norm}_{which_layer}_{mut}.png"
            )
            gam_r2 = plot_dpt_scatter(
                dpt_v,
                apop_v,
                rho,
                r,
                mut,
                out_path,
                dpi=args.dpi,
                gam_splines=args.gam_splines,
                gam_trim_pctl=args.gam_trim_pctl,
            )

            # Compute and Plot Sliding ERank
            # ERank only needs features and DPT order. We don't need to filter out cells missing apoptosis labels!
            valid_erank = np.isfinite(dpt_mut)
            X_mut_raw = X_log[pair_mask][pair_mut_eval]

            X_mut_e = X_mut_raw[valid_erank]
            dpt_e = dpt_mut[valid_erank]
            apop_e = apop_mut[valid_erank]

            dpts, eranks, twonns, apops, eranks_perm, twonns_perm = (
                compute_sliding_metrics(
                    X_mut_e,
                    dpt_e,
                    apop_e,
                    dpt_window=args.dpt_window_size,
                    dpt_step=args.dpt_step,
                    density_samples=args.density_samples,
                    top_k_sv=args.erank_pca_dim,
                    seed=args.seed,
                )
            )
            erank_out = os.path.join(
                out_dir, f"sliding_metrics_dpt_{args.norm}_{which_layer}_{mut}.png"
            )
            rho_de, rho_dt, rho_da, rho_ea = plot_sliding_metrics(
                dpts,
                eranks,
                twonns,
                apops,
                eranks_perm,
                twonns_perm,
                mut,
                erank_out,
                dpi=args.dpi,
            )
        else:
            gam_r2 = compute_gam_r2_only(
                dpt_v,
                apop_v,
                gam_splines=args.gam_splines,
                gam_trim_pctl=args.gam_trim_pctl,
            )

        results.append(
            {
                "Mutation": mut,
                "Seed": args.seed,
                "Norm": args.norm,
                "kNN": args.n_neighbors,
                "Features": X.shape[1],
                "PCA": args.pca_dim,
                "rho": rho,
                "r": r,
                "gam_r2": gam_r2,
                "rho_dpt_erank": rho_de,
                "rho_dpt_twonn": rho_dt,
                "rho_dpt_apop": rho_da,
                "rho_erank_apop": rho_ea,
                "n_valid": valid.sum(),
            }
        )
        del adata_pair

    # Save results
    if results:
        csv_path = os.path.join(
            out_dir,
            f"dpt_summary_{args.norm}_{which_layer}_seed{args.seed}_pca{args.pca_dim}_k{args.n_neighbors}.csv",
        )
        pd.DataFrame(results).to_csv(csv_path, index=False)
        logger.info(f"  Saved summary: {csv_path}")

    if not args.no_plot:
        npz_path = os.path.join(out_dir, f"dpt_results_{args.norm}_{which_layer}.npz")
        np.savez_compressed(npz_path, **dpt_cache)
        logger.info(f"  Saved DPT cache for downstream tools: {npz_path}")


if __name__ == "__main__":
    args = get_args()
    save_args_to_json(args)
    if not args.norm:
        args.norm = "log_std"
    run_pairwise_dpt(args)
