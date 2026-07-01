# !python -m trajectory_inference_pipeline.pairwise_dpt \
# --features_cache "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/caches_per_image_centering/CNN_seed445_SAE/sae_gap_d8192_lam800_normrestored_withnewclass.npz" \
# --cell_death_csv "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/세포이미지별 사멸율/이미지별_세포사멸율_7200.csv" \
# --output_dir "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/caches_per_image_centering/DPT_445" \
#   --n_neighbors 15 \
#   --pca_dim 50 \
#   --filter_mode "none" \
#   --min_cv 0.2 \
#   --de_adj_p 1.0 \
#   --de_min_log2fc 0.0 \
#   --dead_threshold 1e-5 \
#   --norm "log_std" \
#   --gap_l2_norm \
#   --root_mode "diffmap" \
#   --gam_splines 5 \
#   --gam_trim_pctl 5 95 \
#   --de_eval_split 0.5 \
#   --de_mode "per_mut" \
#   --erank_pca_dim 100 \
#   --samples_per_class 15000 \
#   --dpt_window_size 0.2 \
#   --dpt_step 0.02 \
#   --density_samples 800 \
#   --de_eval_split 0.5


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
from pygam import LinearGAM, s
from scipy.stats import kendalltau, pearsonr, spearmanr

try:
    import rapids_singlecell as rsc
    HAS_RSC = True
except Exception:
    HAS_RSC = False

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
        "--root_mode", type=str, default="diffmap", choices=["diffmap"]
    )
    p.add_argument(
        "--gpu", action="store_true", help="Use GPU for DPT via rapids-singlecell"
    )
    p.add_argument("--gam_splines", type=int, default=8)
    p.add_argument("--gam_trim_pctl", type=float, nargs=2, default=[5, 95])
    p.add_argument("--no_plot", action="store_true")
    p.add_argument("--de_eval_split", type=float, default=0.5)

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
    ax.set_ylabel("cell_death rate", fontsize=12)
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


def run_pairwise_dpt(args):
    np.random.seed(args.seed)
    X, superclasses, cell_death, which_layer, X_log = load_and_preprocess(args)
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
        pair_apop = cell_death[pair_mask]
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

        if getattr(args, "gpu", False) and HAS_RSC:
            rsc.get.anndata_to_GPU(adata_pair)
            rsc.pp.neighbors(
                adata_pair, n_neighbors=args.n_neighbors, use_rep="X_pca"
            )
            rsc.tl.diffmap(adata_pair, n_comps=n_diffmap_pair)
            rsc.get.anndata_to_CPU(adata_pair)
        else:
            if getattr(args, "gpu", False) and not HAS_RSC:
                logger.warning(
                    "GPU was requested but rapids_singlecell is not available (cuml missing). Falling back to CPU scanpy."
                )
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
