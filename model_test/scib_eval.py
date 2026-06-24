# ==============================================================================
# scib Representation Quality Evaluation — CNN vs SAE
#
# Computational biology 표준 지표(scib)로 representation 품질 평가:
#   1) ASW (cell type silhouette) — 클래스 분리도
#   2) NMI — Leiden clustering vs true label 일치도
#   3) ARI — 보정된 clustering 일치도
#   4) cLISI — 이웃 중 같은 class 순도
#   5) Graph connectivity — 같은 class의 graph 연결성
#
# 사용 가능한 데이터:
#   - 4 genotype classes (Control, SNCA, GBA, LRRK2)
#   - Precomputed .npz caches (CNN GAP / SAE features)
#
# Usage:
# %matplotlib inline
# import logging
# logging.basicConfig(level=logging.INFO, force=True)
# !python -m model_test.plot_scib_dot \
#     --base_dir /content/drive/MyDrive/Final_paper/lambda_labs_moco_only/scib_eval \
#     --output_dir /content/drive/MyDrive/Final_paper/lambda_labs_moco_only/scib_eval/plots
#
#   # CNN only (layer comparison):
#   python -m model_test.scib_eval \
#       --cnn_cache /path/to/cnn_gap_stage5_out_all.npz \
#       --split_dir /path/to/MoCo_seed87 \
#       --gap_l2_norm \
#       --label stage5_out \
#       --output_dir /path/to/output
# ==============================================================================

# All representation quality metrics were computed in the original feature space without dimensionality reduction to avoid distortion of the intrinsic structure

import argparse
import csv
import json
import logging
import os
import sys

import matplotlib
import numpy as np

_IN_COLAB = "google.colab" in sys.modules
if not _IN_COLAB:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from apoptosis_prediction.local_knn_std import load_cache
from sae_project.step02_logging_utils import SUPERCLASS_MAP, get_logger

logger = get_logger("scib_eval")

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.family"] = "sans-serif"
sns.set_style("ticks")

CLASS_NAMES = {0: "Control", 1: "SNCA", 2: "GBA", 3: "LRRK2"}
SUPERCLASS_TO_INT = {"Control": 0, "SNCA": 1, "GBA": 2, "LRRK2": 3}


# ==============================================================================
# Argument Parser
# ==============================================================================
def get_args():
    p = argparse.ArgumentParser(
        description="scib-based representation quality evaluation (CNN vs SAE)"
    )

    # Data
    p.add_argument(
        "--cnn_cache", type=str, default="", help="Path to CNN GAP .npz cache"
    )
    p.add_argument("--sae_cache", type=str, default="", help="Path to SAE .npz cache")
    p.add_argument(
        "--split_dir",
        type=str,
        default="",
        help="Directory with train/val/test_split.csv "
        "(for test-only evaluation). If empty, use all data.",
    )
    p.add_argument("--dead_threshold", type=float, default=1e-5)
    p.add_argument(
        "--gap_l2_norm", action="store_true", help="L2 normalize feature vectors"
    )
    p.add_argument(
        "--label",
        type=str,
        default="",
        help="Custom label for this run (e.g. 'stage5_out', 'seed87')",
    )

    # scib parameters
    p.add_argument(
        "--n_neighbors",
        type=int,
        default=15,
        help="Number of neighbors for KNN graph (scanpy). Default: 15",
    )
    p.add_argument(
        "--leiden_resolutions",
        type=float,
        nargs="+",
        default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        help="Leiden resolutions to sweep for optimal NMI/ARI",
    )
    p.add_argument(
        "--n_pcs",
        type=int,
        default=50,
        help="Number of PCs for neighbor graph. 0 = use raw features.",
    )
    p.add_argument(
        "--samples_per_class",
        type=int,
        default=0,
        help="Max samples per class (0 = use ALL). Default: 0",
    )

    # Output
    p.add_argument("--output_dir", type=str, default="")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


# ==============================================================================
# Build AnnData object from feature matrix + labels
# ==============================================================================
def build_adata(X, superclasses):
    """Create AnnData from features and class labels.

    Parameters
    ----------
    X : np.ndarray (N, d) — feature matrix
    superclasses : list[str] — class labels (Control/SNCA/GBA/LRRK2)

    Returns
    -------
    adata : AnnData
    """
    import anndata as ad
    import pandas as pd

    adata = ad.AnnData(X=X.astype(np.float32))
    adata.obs["celltype"] = pd.Categorical(superclasses)
    adata.obs["celltype_str"] = superclasses  # string copy for scib

    return adata


# ==============================================================================
# Pure-Python cLISI fallback (avoids scib C++ dependency)
# ==============================================================================
def _compute_clisi_python(adata, label_key, n_neighbors):
    """Compute cell-type LISI from pre-computed KNN graph (pure Python).

    LISI = Inverse Simpson's Index computed on the label distribution
    of each cell's K nearest neighbors.

    For cell-type LISI (cLISI):
      - Perfect separation → LISI = 1.0 (all neighbors same type)
      - Random mixing → LISI = n_labels

    scib convention: cLISI score = 1 - (median_LISI - 1) / (n_labels - 1)
      → 1.0 = perfect separation, 0.0 = fully mixed.

    Parameters
    ----------
    adata : AnnData — must have neighbors computed (adata.obsp['connectivities'])
    label_key : str — column in adata.obs with cell type labels
    n_neighbors : int — K for KNN

    Returns
    -------
    clisi_score : float — scib-scaled cLISI score [0, 1]
    """
    from sklearn.neighbors import NearestNeighbors

    # Get embedding
    if "X_emb" in adata.obsm:
        X = adata.obsm["X_emb"]
    elif "X_pca" in adata.obsm:
        X = adata.obsm["X_pca"]
    else:
        X = adata.X

    labels = np.array(adata.obs[label_key])
    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)
    label_to_int = {l: i for i, l in enumerate(unique_labels)}
    labels_int = np.array([label_to_int[l] for l in labels])

    # KNN
    k = min(n_neighbors, len(X) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=-1)
    nn.fit(X)
    _, indices = nn.kneighbors(X)
    neighbor_indices = indices[:, 1:]  # exclude self

    # Compute per-cell LISI
    n = len(X)
    lisi_values = np.zeros(n)
    for i in range(n):
        neighbor_labels = labels_int[neighbor_indices[i]]
        # Proportion of each label in neighborhood
        counts = np.bincount(neighbor_labels, minlength=n_labels)
        p = counts / counts.sum()
        # Inverse Simpson's Index: 1 / Σ(p²)
        simpson = np.sum(p**2)
        lisi_values[i] = 1.0 / max(simpson, 1e-12)

    median_lisi = np.median(lisi_values)

    # scib scaling: 1.0 = perfect separation, 0.0 = fully mixed
    clisi_score = 1.0 - (median_lisi - 1.0) / max(n_labels - 1.0, 1e-12)
    clisi_score = np.clip(clisi_score, 0.0, 1.0)

    logger.info(
        f"    LISI median={median_lisi:.4f}, "
        f"n_labels={n_labels}, scaled={clisi_score:.4f}"
    )

    return clisi_score


# ==============================================================================
# Compute all scib metrics
# ==============================================================================
def compute_scib_metrics(
    adata, n_neighbors=15, n_pcs=50, leiden_resolutions=None, seed=42
):
    """Compute bio-conservation scib metrics.

    Returns
    -------
    metrics : dict with keys: asw, nmi, ari, clisi, graph_conn,
              best_resolution, n_clusters
    """
    import scanpy as sc
    import scib

    label_key = "celltype"
    results = {}

    # ── 1. PCA (if n_pcs > 0) ──
    if n_pcs > 0 and adata.X.shape[1] > n_pcs:
        sc.tl.pca(adata, n_comps=n_pcs, random_state=seed)
        use_rep = "X_pca"
        embed_key = "X_pca"
        logger.info(f"    PCA: {adata.X.shape[1]}D → {n_pcs}D")
    else:
        # Store raw features as embedding
        adata.obsm["X_feat"] = adata.X.copy()
        use_rep = "X_feat"
        embed_key = "X_feat"
        logger.info(f"    No PCA (d={adata.X.shape[1]})")

    # cLISI expects 'X_emb' in obsm — copy the embedding under that key
    adata.obsm["X_emb"] = adata.obsm[embed_key].copy()

    # ── 2. Neighbor graph ──
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=use_rep, random_state=seed)
    logger.info(f"    KNN graph: n_neighbors={n_neighbors}")

    # ── 3. ASW (Cell type Silhouette) ──
    try:
        asw = scib.me.silhouette(adata, label_key=label_key, embed=embed_key)
        results["asw"] = float(asw)
        logger.info(f"    ASW: {asw:.4f}")
    except Exception as e:
        logger.warning(f"    ASW failed: {e}")
        results["asw"] = float("nan")

    # ── 4. Optimal Leiden clustering → NMI / ARI ──
    # Constrained: n_clusters must be near n_true_labels to prevent NMI
    # inflation from over-clustering (e.g., 100 clusters with 4 labels).
    # Allowed range: [n_labels - 1, n_labels + 2]
    if leiden_resolutions is None:
        leiden_resolutions = [
            0.05,
            0.1,
            0.15,
            0.2,
            0.25,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
        ]

    n_true_labels = len(adata.obs[label_key].unique())
    cluster_min = max(2, n_true_labels - 1)  # e.g., 3 for 4 classes
    cluster_max = n_true_labels * 3.5  # e.g., 14 for 4 classes
    logger.info(
        f"    True labels: {n_true_labels}, "
        f"allowed n_clusters: [{cluster_min}, {cluster_max}]"
    )

    # Track both constrained and unconstrained best
    best_nmi_c, best_ari_c, best_res_c, best_nc_c = -1, -1, None, 0
    best_nmi_u, best_ari_u, best_res_u, best_nc_u = -1, -1, None, 0
    all_leiden = []  # for logging

    for res in leiden_resolutions:
        cluster_key = f"leiden_{res}"
        try:
            sc.tl.leiden(
                adata,
                resolution=res,
                key_added=cluster_key,
                flavor="igraph",
                n_iterations=2,
                random_state=seed,
            )
            n_clusters = len(adata.obs[cluster_key].unique())

            nmi = scib.me.nmi(adata, cluster_key=cluster_key, label_key=label_key)
            ari = scib.me.ari(adata, cluster_key=cluster_key, label_key=label_key)

            in_range = cluster_min <= n_clusters <= cluster_max
            tag = " ✓" if in_range else ""
            logger.info(
                f"    Leiden res={res:.2f}: "
                f"n_clusters={n_clusters}, "
                f"NMI={nmi:.4f}, ARI={ari:.4f}{tag}"
            )
            all_leiden.append(
                {
                    "res": res,
                    "n_clusters": n_clusters,
                    "nmi": nmi,
                    "ari": ari,
                    "in_range": in_range,
                }
            )

            # Unconstrained best (for reference)
            if nmi > best_nmi_u:
                best_nmi_u, best_ari_u = nmi, ari
                best_res_u, best_nc_u = res, n_clusters

            # Constrained best (for reporting)
            if in_range and nmi > best_nmi_c:
                best_nmi_c, best_ari_c = nmi, ari
                best_res_c, best_nc_c = res, n_clusters
                adata.obs["leiden_best"] = adata.obs[cluster_key]
        except Exception as e:
            logger.warning(f"    Leiden res={res} failed: {e}")

    # Use constrained results for NMI/ARI
    if best_nmi_c >= 0:
        results["nmi"] = float(best_nmi_c)
        results["ari"] = float(best_ari_c)
        results["best_resolution"] = best_res_c
        results["n_clusters"] = best_nc_c
        logger.info(
            f"    Constrained best: res={best_res_c:.2f}, "
            f"n_clusters={best_nc_c}, "
            f"NMI={best_nmi_c:.4f}, ARI={best_ari_c:.4f}"
        )
    else:
        # Fallback: no resolution gave clusters in range
        logger.warning(
            f"    No resolution produced {cluster_min}-{cluster_max} "
            f"clusters. Using unconstrained best."
        )
        results["nmi"] = float(best_nmi_u) if best_nmi_u >= 0 else float("nan")
        results["ari"] = float(best_ari_u) if best_ari_u >= 0 else float("nan")
        results["best_resolution"] = best_res_u
        results["n_clusters"] = best_nc_u

    # Also store unconstrained for transparency
    results["nmi_unconstrained"] = (
        float(best_nmi_u) if best_nmi_u >= 0 else float("nan")
    )
    results["ari_unconstrained"] = (
        float(best_ari_u) if best_ari_u >= 0 else float("nan")
    )
    results["n_clusters_unconstrained"] = best_nc_u
    results["cluster_constraint"] = f"[{cluster_min}, {cluster_max}]"

    if best_nmi_u >= 0 and best_nmi_c >= 0:
        nmi_diff = best_nmi_u - best_nmi_c
        if nmi_diff > 0.05:
            logger.info(
                f"    ⚠ Unconstrained NMI={best_nmi_u:.4f} (nc={best_nc_u}) "
                f"vs constrained NMI={best_nmi_c:.4f} (nc={best_nc_c}) — "
                f"Δ={nmi_diff:.4f} confirms over-clustering inflation"
            )
        else:
            logger.info(
                f"    Unconstrained NMI={best_nmi_u:.4f} vs "
                f"constrained NMI={best_nmi_c:.4f} — Δ={nmi_diff:.4f} (minimal)"
            )
    results["leiden_sweep"] = all_leiden

    # ── 5. cLISI (cell-type LISI) ──
    # scib's C++ LISI binary has GLIBCXX compatibility issues on some systems.
    # Fallback: compute cLISI directly from pre-computed KNN graph.
    try:
        clisi = scib.me.clisi_graph(adata, label_key=label_key, type_="embed")
        results["clisi"] = float(clisi)
        logger.info(f"    cLISI (scib): {clisi:.4f}")
    except Exception as e:
        logger.warning(f"    scib cLISI failed ({e}), using pure-Python fallback...")
        try:
            clisi = _compute_clisi_python(adata, label_key, n_neighbors)
            results["clisi"] = float(clisi)
            logger.info(f"    cLISI (python): {clisi:.4f}")
        except Exception as e2:
            logger.warning(f"    cLISI fallback also failed: {e2}")
            results["clisi"] = float("nan")

    # ── 6. Graph connectivity ──
    try:
        gc = scib.me.graph_connectivity(adata, label_key=label_key)
        results["graph_conn"] = float(gc)
        logger.info(f"    Graph connectivity: {gc:.4f}")
    except Exception as e:
        logger.warning(f"    Graph connectivity failed: {e}")
        results["graph_conn"] = float("nan")

    return results


# ==============================================================================
# Plotting: scib metrics comparison bar chart
# ==============================================================================
def plot_scib_comparison(all_results, out_dir, dpi=200):
    """Bar chart comparing scib metrics across sources (CNN/SAE/layers)."""
    metrics_to_plot = ["asw", "nmi", "ari", "clisi", "graph_conn"]
    metric_labels = {
        "asw": "ASW\n(Silhouette)",
        "nmi": "NMI",
        "ari": "ARI",
        "clisi": "cLISI",
        "graph_conn": "Graph\nConnectivity",
    }

    sources = list(all_results.keys())
    n_sources = len(sources)
    n_metrics = len(metrics_to_plot)

    # Color palette
    palette = [
        "#3A7EBF",
        "#E8833A",
        "#88BEDC",
        "#1B4876",
        "#55A868",
        "#C44E52",
        "#8172B2",
    ]

    fig, ax = plt.subplots(figsize=(3.0 + n_metrics * 1.5, 5.5))

    bar_width = 0.8 / n_sources
    x = np.arange(n_metrics)

    for si, source in enumerate(sources):
        offset = (si - (n_sources - 1) / 2) * bar_width
        vals = [all_results[source].get(m, 0) for m in metrics_to_plot]
        color = palette[si % len(palette)]

        bars = ax.bar(
            x + offset,
            vals,
            bar_width,
            color=color,
            alpha=0.8,
            edgecolor="white",
            linewidth=0.8,
            label=source,
            zorder=2,
        )

        # Value labels
        for bar, v in zip(bars, vals):
            if not np.isnan(v) and v > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{v:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    fontweight="bold",
                )

    ax.set_xticks(x)
    ax.set_xticklabels([metric_labels[m] for m in metrics_to_plot], fontsize=10)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(
        "scib Representation Quality Metrics", fontsize=14, fontweight="bold", pad=10
    )
    ax.legend(fontsize=9, framealpha=0.85, loc="best")
    ax.grid(True, alpha=0.15, axis="y")
    ax.set_ylim(0, 1.12)
    sns.despine()
    fig.tight_layout()

    fname = "scib_comparison"
    for ext in [".png", ".svg"]:
        fig.savefig(os.path.join(out_dir, fname + ext), dpi=dpi, bbox_inches="tight")
    logger.info(f"  Saved: {fname}.png/.svg")

    if _IN_COLAB:
        plt.show()
    plt.close(fig)


# ==============================================================================
# Load and preprocess cache
# ==============================================================================
def load_and_preprocess(
    cache_path, dead_threshold, gap_l2_norm, samples_per_class=0, seed=42
):
    """Load cache, apply L2 norm, subsample.

    Returns: X, superclasses (list[str]), source_label
    """
    X, lines, uids, source_label = load_cache(cache_path, dead_threshold)

    if gap_l2_norm:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-12, norms)
        X = X / norms
        logger.info(f"  Applied L2 normalization")

    superclasses = [SUPERCLASS_MAP.get(str(ln), str(ln)) for ln in lines]

    # Subsample
    if samples_per_class > 0:
        rng = np.random.RandomState(seed)
        sc_arr = np.array(superclasses)
        keep = []
        for cls in sorted(np.unique(sc_arr)):
            idx = np.where(sc_arr == cls)[0]
            n_take = min(samples_per_class, len(idx))
            chosen = rng.choice(idx, size=n_take, replace=False)
            keep.extend(chosen.tolist())
        keep = sorted(keep)
        X = X[keep]
        superclasses = [superclasses[i] for i in keep]
        logger.info(f"  Subsampled: {len(keep)} total " f"({samples_per_class}/class)")

    logger.info(f"  Features: {X.shape}, classes: {np.unique(superclasses)}")
    return X, superclasses, source_label


# ==============================================================================
# Main
# ==============================================================================
def main():
    args = get_args()
    np.random.seed(args.seed)

    if not args.cnn_cache and not args.sae_cache:
        raise ValueError("At least one of --cnn_cache or --sae_cache required")

    out_dir = args.output_dir or "./scib_eval_results"
    os.makedirs(out_dir, exist_ok=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"  scib Representation Quality Evaluation")
    logger.info(f"{'='*60}")

    all_results = {}

    # ── Load features ──
    sources = {}
    if args.cnn_cache:
        logger.info(f"\nLoading CNN cache: {args.cnn_cache}")
        X_cnn, sc_cnn, _ = load_and_preprocess(
            args.cnn_cache,
            args.dead_threshold,
            args.gap_l2_norm,
            args.samples_per_class,
            args.seed,
        )
        cnn_label = f"CNN ({args.label})" if args.label else "CNN"
        sources[cnn_label] = (X_cnn, sc_cnn)

    if args.sae_cache:
        logger.info(f"\nLoading SAE cache: {args.sae_cache}")
        X_sae, sc_sae, _ = load_and_preprocess(
            args.sae_cache,
            args.dead_threshold,
            False,  # SAE는 GAP L2 norm 안 함
            args.samples_per_class,
            args.seed,
        )
        sae_label = f"SAE ({args.label})" if args.label else "SAE"
        sources[sae_label] = (X_sae, sc_sae)

    # ── Compute scib metrics for each source ──
    for source_label, (X, superclasses) in sources.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"  Computing scib metrics: {source_label}")
        logger.info(f"  Shape: {X.shape}")
        logger.info(f"{'='*60}")

        adata = build_adata(X, superclasses)

        metrics = compute_scib_metrics(
            adata,
            n_neighbors=args.n_neighbors,
            n_pcs=args.n_pcs,
            leiden_resolutions=args.leiden_resolutions,
            seed=args.seed,
        )

        metrics["source"] = source_label
        metrics["n_samples"] = X.shape[0]
        metrics["n_features"] = X.shape[1]
        metrics["n_pcs"] = args.n_pcs
        metrics["n_neighbors"] = args.n_neighbors

        all_results[source_label] = metrics

    # ── Comparison plot (if multiple sources) ──
    if len(all_results) >= 1:
        plot_scib_comparison(all_results, out_dir, args.dpi)

    # ── Save results ──
    # JSON
    json_path = os.path.join(out_dir, "scib_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nSaved JSON: {json_path}")

    # CSV
    csv_path = os.path.join(out_dir, "scib_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "source",
                "asw",
                "nmi",
                "ari",
                "clisi",
                "graph_conn",
                "best_resolution",
                "n_clusters",
                "n_samples",
                "n_features",
            ]
        )
        for label, m in all_results.items():
            writer.writerow(
                [
                    label,
                    f"{m.get('asw', 'nan'):.4f}",
                    f"{m.get('nmi', 'nan'):.4f}",
                    f"{m.get('ari', 'nan'):.4f}",
                    f"{m.get('clisi', 'nan'):.4f}",
                    f"{m.get('graph_conn', 'nan'):.4f}",
                    m.get("best_resolution", ""),
                    m.get("n_clusters", ""),
                    m.get("n_samples", ""),
                    m.get("n_features", ""),
                ]
            )
    logger.info(f"Saved CSV: {csv_path}")

    # ── Summary ──
    logger.info(f"\n{'='*70}")
    logger.info(
        f"  {'Source':25s}  {'ASW':>7s}  {'NMI':>7s}  {'ARI':>7s}  "
        f"{'cLISI':>7s}  {'GConn':>7s}"
    )
    logger.info(f"  {'─'*65}")
    for label, m in all_results.items():
        logger.info(
            f"  {label:25s}  {m.get('asw',0):7.4f}  {m.get('nmi',0):7.4f}  "
            f"{m.get('ari',0):7.4f}  {m.get('clisi',0):7.4f}  "
            f"{m.get('graph_conn',0):7.4f}"
        )
    logger.info(f"{'='*70}")
    logger.info(f"  Output: {out_dir}")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    main()
