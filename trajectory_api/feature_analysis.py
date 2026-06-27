import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import seaborn as sns
from scipy.ndimage import gaussian_filter1d

from model_train.logging_utils import get_logger

logger = get_logger("trajectory_api_feature")

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
sns.set_style("ticks")


# ==============================================================================
# DPT Feature Heatmap
# ==============================================================================
def plot_dpt_feature_heatmap(
    adata_pair: sc.AnnData,
    mutation: str,
    out_dir: str,
    prefix: str = "",
    n_bins: int = 100,
    seed: int = 42,
    dpi: int = 300,
):
    """
    Plot heatmap of all DE features along DPT.
    Cells are binned by DPT. Features are clustered by Leiden based on their bin-profiles.
    """
    os.makedirs(out_dir, exist_ok=True)

    pair_sc = adata_pair.obs["mutation"].values
    mut_mask = pair_sc == mutation

    if "dpt_pseudotime" not in adata_pair.obs:
        logger.warning("No dpt_pseudotime found in adata.")
        return

    dpt_vals = adata_pair.obs["dpt_pseudotime"].values[mut_mask]
    X_features = adata_pair.X[mut_mask]

    valid = np.isfinite(dpt_vals)
    if valid.sum() < n_bins:
        logger.warning(
            f"Not enough valid cells ({valid.sum()}) for {n_bins} bins. Skipping heatmap."
        )
        return

    dpt_v = dpt_vals[valid]
    X_v = X_features[valid]

    sort_idx = np.argsort(dpt_v)
    X_sorted = X_v[sort_idx]

    n_cells = len(X_sorted)
    bin_size = n_cells // n_bins

    X_binned = []
    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx = (i + 1) * bin_size if i < n_bins - 1 else n_cells
        bin_mean = X_sorted[start_idx:end_idx].mean(axis=0)
        X_binned.append(bin_mean)

    X_binned = np.array(X_binned).T

    X_mean = X_binned.mean(axis=1, keepdims=True)
    X_std = X_binned.std(axis=1, keepdims=True) + 1e-8
    X_binned_z = (X_binned - X_mean) / X_std
    X_binned_z = gaussian_filter1d(X_binned_z, sigma=1.5, axis=1)

    adata_feat = sc.AnnData(X_binned_z.astype(np.float32))
    sc.pp.neighbors(adata_feat, n_neighbors=min(15, len(adata_feat) - 1), use_rep="X")
    sc.tl.leiden(
        adata_feat, resolution=1.0, random_state=seed, flavor="igraph", directed=False
    )

    cluster_labels = adata_feat.obs["leiden"].values
    unique_clusters = np.unique(cluster_labels)

    cluster_peaks = []
    for cl in unique_clusters:
        cl_mask = cluster_labels == cl
        cl_mean_profile = X_binned_z[cl_mask].mean(axis=0)
        peak_bin = np.argmax(cl_mean_profile)
        cluster_peaks.append((cl, peak_bin))

    cluster_peaks.sort(key=lambda x: x[1])
    sorted_clusters = [x[0] for x in cluster_peaks]

    sorted_feat_idx = []
    for cl in sorted_clusters:
        idx_in_cl = np.where(cluster_labels == cl)[0]
        peaks_in_cl = np.argmax(X_binned_z[idx_in_cl], axis=1)
        sorted_in_cl = idx_in_cl[np.argsort(peaks_in_cl)]
        sorted_feat_idx.extend(sorted_in_cl)

    X_heatmap_final = X_binned_z[sorted_feat_idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(
        X_heatmap_final,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-2,
        vmax=2,
        interpolation="nearest",
    )

    ax.set_xlabel("Diffusion Pseudotime (Binned)", fontsize=12)
    ax.set_ylabel(f"All {X_features.shape[1]} Features (Leiden Clustered)", fontsize=12)
    ax.set_title(f"Feature Module Activations along DPT ({mutation})", fontsize=14)

    ax.set_yticks([])
    ax.set_xticks(np.linspace(0, n_bins - 1, 5))
    ax.set_xticklabels(["Early", "", "Mid", "", "Late"])

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Z-scored Activation", fontsize=10)

    fig.tight_layout()
    out_name = (
        f"dpt_heatmap_{mutation}_{prefix}" if prefix else f"dpt_heatmap_{mutation}"
    )
    hm_path = os.path.join(out_dir, f"{out_name}.png")
    fig.savefig(hm_path, dpi=dpi, bbox_inches="tight", transparent=True)
    sns.despine(ax=ax, left=True, bottom=True)
    fig.savefig(
        hm_path.replace(".png", ".svg"),
        format="svg",
        bbox_inches="tight",
        transparent=True,
    )
    plt.close(fig)

    logger.info(f"Heatmap saved: {hm_path}")


# ==============================================================================
# Single Feature Trend Line
# ==============================================================================
def plot_feature_trends(
    adata_pair: sc.AnnData,
    mutation: str,
    out_dir: str,
    feature_indices: list = [],
    prefix: str = "",
    gam_trim_pctl: list = [1, 99],
    alpha: float = 0.3,
    point_size: float = 10.0,
    dpi: int = 300,
):
    """
    Plot the smoothed trend (GAM or rolling mean) of specified features along DPT.
    If feature_indices is empty, plots the top 5 most variable features.
    """
    os.makedirs(out_dir, exist_ok=True)

    pair_sc = adata_pair.obs["mutation"].values
    mut_mask = pair_sc == mutation

    if "dpt_pseudotime" not in adata_pair.obs:
        return

    dpt_vals = adata_pair.obs["dpt_pseudotime"].values[mut_mask]
    X_features = adata_pair.X[mut_mask]

    if not feature_indices:
        # Default to top 5 features with highest variance in this mutation
        variances = np.var(X_features, axis=0)
        feature_indices = np.argsort(variances)[-5:][::-1].tolist()

    for f_idx in feature_indices:
        if f_idx >= X_features.shape[1]:
            continue

        feat_vals = X_features[:, f_idx]
        _plot_single_trend(
            dpt_vals,
            feat_vals,
            mutation,
            out_dir,
            f"{prefix}_feat{f_idx}",
            dpi,
            gam_trim_pctl,
            alpha,
            point_size,
        )


def _plot_single_trend(
    dpt_vals,
    feat_vals,
    mutation,
    out_dir,
    prefix,
    dpi,
    gam_trim_pctl,
    alpha,
    point_size,
):
    valid = np.isfinite(dpt_vals)
    if valid.sum() < 10:
        return

    dpt_v = dpt_vals[valid]
    feat_v = feat_vals[valid]

    sort_idx = np.argsort(dpt_v)
    dpt_v = dpt_v[sort_idx]
    feat_v = feat_v[sort_idx]

    n_points = len(dpt_v)
    low_idx = int(n_points * (gam_trim_pctl[0] / 100.0))
    high_idx = int(n_points * (gam_trim_pctl[1] / 100.0))

    if high_idx <= low_idx + 10:
        low_idx, high_idx = 0, n_points

    dpt_fit = dpt_v[low_idx:high_idx]
    feat_fit = feat_v[low_idx:high_idx]

    fig, ax = plt.subplots(figsize=(6, 4))

    TREND_COLORS = {"SNCA": "#f2c3c3", "GBA": "#f9d2ab", "LRRK2": "#c2d7f2"}
    scatter_color = TREND_COLORS.get(mutation, "#DD8452")

    ax.scatter(
        dpt_fit,
        feat_fit,
        color=scatter_color,
        alpha=alpha,
        s=point_size,
        label=mutation,
        edgecolors="none",
    )

    try:
        from pygam import LinearGAM
        from pygam import s as s_term

        gam = LinearGAM(s_term(0, n_splines=10, spline_order=3)).fit(
            dpt_fit.reshape(-1, 1), feat_fit
        )
        x_line = np.linspace(dpt_fit.min(), dpt_fit.max(), 100)
        y_line = gam.predict(x_line.reshape(-1, 1))
        ci = gam.confidence_intervals(x_line.reshape(-1, 1), width=0.95)

        ax.plot(x_line, y_line, color="black", lw=2.5, zorder=5)
        ax.fill_between(
            x_line, ci[:, 0], ci[:, 1], color="black", alpha=0.15, zorder=4, linewidth=0
        )
    except ImportError:
        window = max(5, len(feat_v) // 20)
        y_smooth = np.convolve(feat_v, np.ones(window) / window, mode="valid")
        x_smooth = dpt_v[window // 2 : -window // 2 + 1]
        ax.plot(x_smooth, y_smooth, color="black", lw=2.5, zorder=5)

    ax.set_xlabel("Diffusion Pseudotime", fontsize=12)
    ax.set_ylabel("Normalized Feature Activation", fontsize=12)
    ax.set_title(f"Feature Trend along DPT ({mutation})", fontsize=13)
    ax.legend(fontsize=10)

    fig.tight_layout()
    sns.despine(ax=ax, left=True, bottom=True)

    out_name = f"trend_{mutation}_{prefix}.png" if prefix else f"trend_{mutation}.png"
    out_path = os.path.join(out_dir, out_name)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", transparent=True)
    fig.savefig(
        out_path.replace(".png", ".svg"),
        format="svg",
        bbox_inches="tight",
        transparent=True,
    )
    plt.close(fig)
