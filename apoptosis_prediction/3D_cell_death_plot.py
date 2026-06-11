#!/usr/bin/env python
# ==============================================================================
# Semantic Manifold Surface — 3D Local Gradient Magnitude Visualization
#
# UMAP 2D 차원축소 후 grid 기반 세포사멸율 분석:
#   z축 = grid cell 내 세포사멸율 평균
#   색상 = grid cell 내 세포사멸율 표준편차
#
# CNN → jagged surface + patchy std (의미론적 유사성 미반영)
# SAE → smooth surface + gradient std (의미론적 유사성 반영)
#
# 기존 함수 재사용:
#   load_cache()              ← local_knn_std.py
#   load_and_match_apoptosis() ← dpt_kendall.py
#   SUPERCLASS_MAP, get_logger ← sae_project.step02_logging_utils
#
# Usage:
#   python apoptosis_prediction/3D_cell_death_plot.py \
#       --cnn_cache "..." --sae_cache "..." \
#       --apoptosis_csv "..." \
#       --gap_l2_norm --dead_threshold 1e-5 \
#       --output_dir "..."
# ==============================================================================

import os
import sys
import json
import argparse
import numpy as np
import logging

# Ensure project root is on sys.path (for direct script execution)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import matplotlib
_IN_COLAB = "google.colab" in sys.modules
if not _IN_COLAB:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize, PowerNorm
import matplotlib.gridspec as gridspec
import seaborn as sns

from apoptosis_prediction.local_knn_std import load_cache
from kendall_correlation_coefficient.dpt_kendall import load_and_match_apoptosis
from sae_project.step02_logging_utils import get_logger, SUPERCLASS_MAP

try:
    import umap as umap_lib
    _HAS_UMAP = True
except ImportError:
    umap_lib = None
    _HAS_UMAP = False

logger = get_logger("manifold_surface_3d")

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
sns.set_style("ticks")

# Suppress noisy fonttools logs (font subsetting during SVG/PDF save)
logging.getLogger("fontTools").setLevel(logging.WARNING)
logging.getLogger("fontTools.subset").setLevel(logging.WARNING)

# Consistent source colors (matches local_knn_std.py, plot_cnn_vs_sae)
SOURCE_COLORS = {"CNN": "#4C72B0", "SAE": "#DD8452"}
MUTATION_COLORS = {"SNCA": "#E8553A", "GBA": "#1DB954", "LRRK2": "#9B59B6",
                   "Control": "#2176AE"}






def truncate_cmap(cmap_name, start=0.0, end=1.0, n=256):
    """Truncate a colormap to [start, end] range.
    e.g. truncate_cmap('mako_r', 0.15) removes the lightest 15%."""
    if start == 0.0 and end == 1.0:
        return plt.get_cmap(cmap_name)
    from matplotlib.colors import LinearSegmentedColormap
    base = plt.get_cmap(cmap_name)
    colors = base(np.linspace(start, end, n))
    return LinearSegmentedColormap.from_list(
        f"{cmap_name}_trunc", colors, N=n)


# ==============================================================================
# Argument Parser
# ==============================================================================
def get_args():
    p = argparse.ArgumentParser(
        description="Semantic Manifold Surface: 3D cell death landscape visualization"
    )
    p.add_argument("--cnn_cache", type=str, default="",
                   help="Path to CNN GAP .npz cache")
    p.add_argument("--sae_cache", type=str, default="",
                   help="Path to SAE .npz cache")
    p.add_argument("--apoptosis_csv", type=str, required=True,
                   help="Path to per-image apoptosis rate CSV")
    p.add_argument("--rate_col", type=str, default=None,
                   help="CSV column for coloring. None=auto (intensity_rate), "
                        "'MFI'=auto-compute total_intensity/total_nucleus_pixels. "
                        "Default: None (intensity_rate)")
    p.add_argument("--dead_threshold", type=float, default=1e-5)

    # Feature preprocessing (minimal — raw by default)
    p.add_argument("--gap_l2_norm", action="store_true",
                   help="L2 normalize feature vectors")
    p.add_argument("--pre_l2_norm", action="store_true")
    p.add_argument("--divide_hw", type=int, default=0)
    p.add_argument("--pca_dim", type=int, default=0,
                   help="PCA components before KNN/UMAP (0 = no PCA). Default: 0")

    # Dimensionality reduction for 2D embedding
    p.add_argument("--dr_method", type=str, default="umap",
                   choices=["umap", "tsne", "le", "lle"],
                   help="2D embedding method: umap, tsne (t-SNE), "
                        "le (Laplacian Eigenmap), lle (Locally Linear Embedding). "
                        "Default: umap")

    # UMAP-specific
    p.add_argument("--umap_n_neighbors", type=int, default=15)
    p.add_argument("--umap_min_dist", type=float, default=0.05)
    p.add_argument("--umap_metric", type=str, default="euclidean")

    # t-SNE-specific
    p.add_argument("--tsne_perplexity", type=float, default=30.0,
                   help="t-SNE perplexity. Default: 30")
    p.add_argument("--tsne_lr", type=float, default=200.0,
                   help="t-SNE learning rate. Default: 200 (auto)")

    # Laplacian Eigenmap-specific
    p.add_argument("--le_n_neighbors", type=int, default=15,
                   help="Laplacian Eigenmap n_neighbors. Default: 15")
    p.add_argument("--le_affinity", type=str, default="nearest_neighbors",
                   choices=["nearest_neighbors", "rbf"],
                   help="LE affinity kernel. Default: nearest_neighbors")

    # LLE-specific
    p.add_argument("--lle_n_neighbors", type=int, default=15,
                   help="LLE n_neighbors. Default: 15")
    p.add_argument("--lle_method", type=str, default="standard",
                   choices=["standard", "modified", "hessian", "ltsa"],
                   help="LLE method variant. Default: standard")

    # Grid
    p.add_argument("--grid_resolution", type=int, default=25,
                   help="Number of grid cells per axis (NxN). Default: 25")
    p.add_argument("--min_samples_per_cell", type=int, default=3,
                   help="Minimum samples per grid cell. Default: 3")

    # KNN metacell
    p.add_argument("--knn_k", type=int, default=15,
                   help="K neighbors for KNN metacell approach. Default: 15")

    # Visual styling
    p.add_argument("--gamma", type=float, default=1.0,
                   help="Gamma for power-law color mapping on apoptosis rate. "
                        "<1 expands low values (more contrast at bottom), "
                        ">1 expands high values. Default: 1.0 (linear)")
    p.add_argument("--dot_size", type=float, default=3.0,
                   help="Scatter dot size (matplotlib 's' parameter). Default: 3")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="Dot transparency (0=invisible, 1=opaque). Default: 0.5")
    p.add_argument("--cmap_std", type=str, default="inferno",
                   help="Colormap for local std plots. Default: inferno")
    p.add_argument("--cmap_mean", type=str, default="viridis",
                   help="Colormap for mean apoptosis plots. Default: viridis")
    p.add_argument("--cmap_start", type=float, default=0.0,
                   help="Truncate colormap start (0-1). e.g. 0.15 removes "
                        "lightest 15 pct. Default: 0.0")
    p.add_argument("--cmap_end", type=float, default=1.0,
                   help="Truncate colormap end (0-1). Default: 1.0")
    p.add_argument("--bg_color", type=str, default="white",
                   help="Background color for scatter plots. Default: white")
    p.add_argument("--std_mode", type=str, default="std",
                   choices=["std", "cv"],
                   help="Top-row metric: 'std' (raw local σ) or 'cv' (σ/μ, coefficient "
                        "of variation — normalized for mean-variance coupling). Default: std")
    p.add_argument("--density_bg", action="store_true",
                   help="Add a KDE density heatmap behind scatter dots "
                        "(fills sparse regions, useful when N is small)")
    p.add_argument("--vmax_pctl", type=float, default=97.5,
                   help="Percentile for color scale max (lower = more saturated, "
                        "more contrast). e.g. 85 for dramatic. Default: 97.5")

    # 3D view
    p.add_argument("--elev", type=float, default=25)
    p.add_argument("--azim", type=float, default=-60)

    # Output
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--samples_per_class", type=int, default=0,
                   help="Max samples per class (0 = ALL)")
    p.add_argument("--classes", type=str, nargs="+",
                   default=["Control", "SNCA", "GBA", "LRRK2"],
                   help="Which classes to include. "
                        "e.g. '--classes Control SNCA GBA' for 3 classes. "
                        "Default: Control SNCA GBA LRRK2")

    # ── Subsample UMAP (apoptosis 정보 무관) ──
    p.add_argument("--subsample_n", type=int, default=0,
                   help="Total N for apoptosis-free UMAP (subsampled from ALL vectors, "
                        "balanced across classes). 0 = skip. e.g. 20000")
    p.add_argument("--subsample_per_class", type=int, default=0,
                   help="N per class for apoptosis-free UMAP. "
                        "If set, overrides --subsample_n balance logic. "
                        "0 = use --subsample_n evenly split. e.g. 5000")

    return p.parse_args()


# ==============================================================================
# Grid Statistics
# ==============================================================================
def compute_grid_stats(umap_xy, apoptosis, grid_res, min_samples):
    """
    Partition UMAP 2D space into grid, compute per-cell stats.

    Returns
    -------
    grid_mean : (grid_res, grid_res) — mean apoptosis per cell (NaN if < min_samples)
    grid_std  : (grid_res, grid_res) — std of apoptosis per cell
    grid_count: (grid_res, grid_res) — sample count per cell
    x_edges, y_edges : bin edges
    """
    x_min, x_max = umap_xy[:, 0].min(), umap_xy[:, 0].max()
    y_min, y_max = umap_xy[:, 1].min(), umap_xy[:, 1].max()

    margin_x = (x_max - x_min) * 0.01
    margin_y = (y_max - y_min) * 0.01

    x_edges = np.linspace(x_min - margin_x, x_max + margin_x, grid_res + 1)
    y_edges = np.linspace(y_min - margin_y, y_max + margin_y, grid_res + 1)

    grid_mean = np.full((grid_res, grid_res), np.nan)
    grid_std = np.full((grid_res, grid_res), np.nan)
    grid_count = np.zeros((grid_res, grid_res), dtype=int)

    x_idx = np.clip(np.digitize(umap_xy[:, 0], x_edges) - 1, 0, grid_res - 1)
    y_idx = np.clip(np.digitize(umap_xy[:, 1], y_edges) - 1, 0, grid_res - 1)

    for i in range(grid_res):
        for j in range(grid_res):
            mask = (x_idx == i) & (y_idx == j)
            count = int(mask.sum())
            grid_count[i, j] = count
            if count >= min_samples:
                vals = apoptosis[mask]
                grid_mean[i, j] = np.mean(vals)
                grid_std[i, j] = np.std(vals)

    return grid_mean, grid_std, grid_count, x_edges, y_edges


# ==============================================================================
# KNN Local Stats — per-point KNN in UMAP 2D space
# ==============================================================================
def compute_knn_local_stats(X, apoptosis, k):
    """
    Per-point local stats via KNN.

    Parameters
    ----------
    X : (N, D) — coordinates (UMAP 2D or any feature space)
    apoptosis : (N,) — apoptosis rates
    k : int — number of neighbors

    Returns
    -------
    local_means : (N,) — mean of K neighbors' apoptosis
    local_stds  : (N,) — std of K neighbors' apoptosis
    """
    from sklearn.neighbors import NearestNeighbors

    n = X.shape[0]
    k_actual = min(k, n - 1)
    if k_actual < 2:
        return np.full(n, np.nan), np.full(n, np.nan)

    nn = NearestNeighbors(n_neighbors=k_actual + 1, metric="euclidean", n_jobs=-1)
    nn.fit(X)
    _, indices = nn.kneighbors(X)
    neighbor_indices = indices[:, 1:]  # exclude self

    neighbor_apoptosis = apoptosis[neighbor_indices]  # (N, k)
    local_means = np.mean(neighbor_apoptosis, axis=1)
    local_stds = np.std(neighbor_apoptosis, axis=1)

    return local_means, local_stds


def interpolate_to_surface_grid(umap_xy, values, grid_res):
    """
    Interpolate per-point values onto a regular grid for surface plotting.
    Uses UMAP 2D coordinates as the spatial layout — visualization only.

    Returns
    -------
    grid_values : (grid_res, grid_res) — interpolated values (NaN outside hull)
    x_edges, y_edges : bin edges
    """
    from scipy.interpolate import griddata

    x_min, x_max = umap_xy[:, 0].min(), umap_xy[:, 0].max()
    y_min, y_max = umap_xy[:, 1].min(), umap_xy[:, 1].max()
    margin_x = (x_max - x_min) * 0.02
    margin_y = (y_max - y_min) * 0.02

    xi = np.linspace(x_min + margin_x, x_max - margin_x, grid_res)
    yi = np.linspace(y_min + margin_y, y_max - margin_y, grid_res)
    Xi, Yi = np.meshgrid(xi, yi, indexing='ij')

    grid_values = griddata(umap_xy, values, (Xi, Yi), method='linear')

    x_edges = np.linspace(x_min, x_max, grid_res + 1)
    y_edges = np.linspace(y_min, y_max, grid_res + 1)

    return grid_values, x_edges, y_edges


# ==============================================================================
# Moran's I — Spatial Autocorrelation (queen contiguity on grid)
# ==============================================================================
def compute_morans_i(grid_values):
    """
    Moran's I on 2D grid using queen contiguity (8-neighbor).
    I > expected → spatial clustering of apoptosis rates.

    Returns: (I, expected_I, N_cells)
    """
    valid_mask = ~np.isnan(grid_values)
    rows, cols = np.where(valid_mask)
    n = len(rows)
    if n < 4:
        return np.nan, np.nan, n

    values = grid_values[valid_mask]
    x_bar = values.mean()
    dev = values - x_bar

    coord_to_idx = {(rows[k], cols[k]): k for k in range(n)}

    W_total = 0.0
    cross_sum = 0.0
    for k in range(n):
        r, c = rows[k], cols[k]
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nb = (r + dr, c + dc)
                if nb in coord_to_idx:
                    j = coord_to_idx[nb]
                    cross_sum += dev[k] * dev[j]
                    W_total += 1.0

    var_sum = np.sum(dev ** 2)
    if W_total == 0 or var_sum < 1e-15:
        return np.nan, np.nan, n

    I = (n / W_total) * (cross_sum / var_sum)
    expected = -1.0 / (n - 1)
    return float(I), float(expected), n


# ==============================================================================
# Surface Roughness — Mean |∇z|
# ==============================================================================
def compute_surface_roughness(grid_values):
    """
    Surface roughness = mean |∇z| over interior valid grid cells.
    Higher = more jagged (CNN), lower = smoother (SAE).
    """
    from scipy.ndimage import binary_erosion

    valid = ~np.isnan(grid_values)
    if valid.sum() < 4:
        return np.nan

    filled = np.where(valid, grid_values, 0.0)
    gy, gx = np.gradient(filled)
    grad_mag = np.sqrt(gx**2 + gy**2)
    grad_mag[~valid] = np.nan

    # Only keep cells where all 8-connected neighbors are also valid
    interior = binary_erosion(valid, structure=np.ones((3, 3)))
    grad_mag[~interior] = np.nan

    if np.all(np.isnan(grad_mag)):
        return np.nan
    return float(np.nanmean(grad_mag))


# ==============================================================================
# Plot 1: 3D Surface — CNN vs SAE side-by-side
# ==============================================================================
def plot_manifold_surface_3d(results, scope_name, output_path,
                              elev=25, azim=-60, dpi=200,
                              axis_labels=("UMAP 1", "UMAP 2")):
    """
    Side-by-side 3D surface.
    z = mean apoptosis rate,  facecolor = local apoptosis std.
    """
    sources = list(results.keys())
    n_sources = len(sources)

    # ── Shared ranges ──
    valid_stds = []
    valid_means = []
    for s in sources:
        gs = results[s]["grid_std"]
        gm = results[s]["grid_mean"]
        if np.any(~np.isnan(gs)):
            valid_stds.append(gs[~np.isnan(gs)])
        if np.any(~np.isnan(gm)):
            valid_means.append(gm[~np.isnan(gm)])

    all_stds = np.concatenate(valid_stds)
    all_means = np.concatenate(valid_means)

    # Use 97.5th percentile for vmax to avoid outlier compression
    vmin_std = 0.0
    vmax_std = float(np.percentile(all_stds, 97.5))
    z_min = float(np.min(all_means))
    z_max = float(np.max(all_means))
    z_margin = (z_max - z_min) * 0.05

    norm_std = Normalize(vmin=vmin_std, vmax=vmax_std, clip=True)
    cmap = cm.get_cmap("inferno")

    fig = plt.figure(figsize=(7.5 * n_sources + 1.5, 6.5))
    gs = gridspec.GridSpec(1, n_sources + 1,
                           width_ratios=[1] * n_sources + [0.05])

    for idx, src in enumerate(sources):
        ax = fig.add_subplot(gs[0, idx], projection='3d')

        res = results[src]
        grid_mean = res["grid_mean"]
        grid_std = res["grid_std"]
        x_edges = res["x_edges"]
        y_edges = res["y_edges"]

        grid_res = grid_mean.shape[0]
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
        X, Y = np.meshgrid(x_centers, y_centers, indexing='ij')

        # Handle NaN: masked array + transparent facecolors
        nan_mask = np.isnan(grid_mean)
        Z = grid_mean.copy()
        Z[nan_mask] = z_min - z_margin * 3  # push invisible cells below

        std_clean = grid_std.copy()
        std_clean[nan_mask] = 0.0
        fc = cmap(norm_std(std_clean))
        fc[nan_mask, 3] = 0.0  # fully transparent for empty cells

        ax.plot_surface(
            X, Y, Z, facecolors=fc,
            rstride=1, cstride=1,
            shade=False, antialiased=True,
            edgecolor=(0.3, 0.3, 0.3, 0.08),
        )

        # Axis labels and limits
        ax.set_xlabel(axis_labels[0], fontsize=9, labelpad=5)
        ax.set_ylabel(axis_labels[1], fontsize=9, labelpad=5)
        ax.set_zlabel("Apoptosis Rate", fontsize=9, labelpad=5)
        ax.set_zlim(z_min - z_margin, z_max + z_margin)
        ax.view_init(elev=elev, azim=azim)

        # Clean panes
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor((0.8, 0.8, 0.8, 0.5))
        ax.yaxis.pane.set_edgecolor((0.8, 0.8, 0.8, 0.5))
        ax.zaxis.pane.set_edgecolor((0.8, 0.8, 0.8, 0.5))
        ax.tick_params(labelsize=7)

        # Metrics text box
        roughness = res.get("roughness", np.nan)
        morans_i = res.get("morans_i", np.nan)
        n_samp = res.get("n_samples", 0)
        n_cells = res.get("n_valid_cells", 0)

        text_lines = [f"{src}"]
        text_lines.append(f"n = {n_samp:,}")
        if not np.isnan(roughness):
            text_lines.append(f"Roughness = {roughness:.4f}")
        if not np.isnan(morans_i):
            text_lines.append(f"Moran's I = {morans_i:.3f}")
        text_lines.append(f"Valid cells = {n_cells}")

        ax.text2D(0.02, 0.97, "\n".join(text_lines),
                  transform=ax.transAxes, fontsize=8,
                  verticalalignment='top',
                  bbox=dict(boxstyle='round,pad=0.4',
                            facecolor='white', alpha=0.85,
                            edgecolor=(0.7, 0.7, 0.7)))

    # Shared colorbar
    cbar_ax = fig.add_subplot(gs[0, -1])
    sm = cm.ScalarMappable(cmap=cmap, norm=norm_std)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Local Apoptosis Std (σ)", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle(f"Semantic Manifold Surface — {scope_name}",
                 fontsize=14, fontweight="bold", y=0.98)

    fig.subplots_adjust(left=0.05, right=0.88, bottom=0.05, top=0.92,
                         wspace=0.15)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    for ext in [".svg", ".pdf"]:
        fig.savefig(output_path.replace(".png", ext), bbox_inches="tight")
    if _IN_COLAB:
        plt.show()
    plt.close(fig)
    logger.info(f"  Saved 3D surface: {output_path}")


# ==============================================================================
# Plot 2: 2D Heatmap — top row: mean apoptosis, bottom row: local std
# ==============================================================================
def plot_manifold_heatmap_2d(results, scope_name, output_path, dpi=200,
                              axis_labels=("UMAP 1", "UMAP 2")):
    """2×n_sources heatmap: Mean apoptosis (top) + Local Std (bottom)."""
    sources = list(results.keys())
    n = len(sources)

    # Shared ranges
    all_means = np.concatenate([
        results[s]["grid_mean"][~np.isnan(results[s]["grid_mean"])]
        for s in sources])
    all_stds = np.concatenate([
        results[s]["grid_std"][~np.isnan(results[s]["grid_std"])]
        for s in sources])

    vmin_mean = float(np.percentile(all_means, 2.5))
    vmax_mean = float(np.percentile(all_means, 97.5))
    vmin_std = 0.0
    vmax_std = float(np.percentile(all_stds, 97.5))

    fig, axes = plt.subplots(2, n, figsize=(5.5 * n, 9))
    if n == 1:
        axes = axes.reshape(2, 1)

    im_mean_last = None
    im_std_last = None

    for idx, src in enumerate(sources):
        gm = results[src]["grid_mean"].T  # transpose for imshow (row=y)
        gs = results[src]["grid_std"].T

        # Mean heatmap
        ax = axes[0, idx]
        im_mean = ax.imshow(gm, origin='lower', cmap='viridis',
                            vmin=vmin_mean, vmax=vmax_mean, aspect='equal',
                            interpolation='nearest')
        ax.set_title(f"{src} — Mean Apoptosis Rate", fontsize=11,
                     fontweight='bold')
        ax.set_xlabel(f"{axis_labels[0]} bin", fontsize=9)
        ax.set_ylabel(f"{axis_labels[1]} bin", fontsize=9)
        ax.tick_params(labelsize=7)
        im_mean_last = im_mean

        # Std heatmap
        ax = axes[1, idx]
        im_std = ax.imshow(gs, origin='lower', cmap='inferno',
                           vmin=vmin_std, vmax=vmax_std, aspect='equal',
                           interpolation='nearest')
        ax.set_title(f"{src} — Local Std (σ)", fontsize=11, fontweight='bold')
        ax.set_xlabel(f"{axis_labels[0]} bin", fontsize=9)
        ax.set_ylabel(f"{axis_labels[1]} bin", fontsize=9)
        ax.tick_params(labelsize=7)
        im_std_last = im_std

    # Colorbars
    fig.colorbar(im_mean_last, ax=axes[0, :].ravel().tolist(),
                 shrink=0.8, label="Mean Apoptosis Rate", pad=0.02)
    fig.colorbar(im_std_last, ax=axes[1, :].ravel().tolist(),
                 shrink=0.8, label="Local Std (σ)", pad=0.02)

    fig.suptitle(f"Manifold Grid Heatmap — {scope_name}",
                 fontsize=13, fontweight="bold")
    fig.subplots_adjust(left=0.06, right=0.88, top=0.92, bottom=0.06,
                         hspace=0.3, wspace=0.25)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(output_path.replace(".png", ".svg"), bbox_inches="tight")
    if _IN_COLAB:
        plt.show()
    plt.close(fig)
    logger.info(f"  Saved 2D heatmap: {output_path}")


# ==============================================================================
# Plot 3: UMAP Scatter — colored by apoptosis rate (context visualization)
# ==============================================================================
def plot_umap_scatter(results, scope_name, output_path, dpi=200,
                      gamma=1.0, dot_size=3, alpha=0.5,
                      cmap_mean='viridis', bg_color='white',
                      axis_labels=("UMAP 1", "UMAP 2"),
                      density_bg=False, vmax_pctl=97.5):
    """UMAP scatter colored by apoptosis rate — CNN vs SAE side by side."""
    sources = list(results.keys())
    n = len(sources)
    is_dark = bg_color.lower() not in ('white', '#ffffff', 'w')

    # Shared value range
    all_apop = np.concatenate([results[s]["apoptosis"] for s in sources])
    vmin = float(np.percentile(all_apop, 100 - vmax_pctl))
    vmax = float(np.percentile(all_apop, vmax_pctl))

    # Power-law normalization (gamma=1 → linear)
    if gamma != 1.0:
        norm = PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax, clip=True)
        gamma_label = f" (γ={gamma})"
    else:
        norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
        gamma_label = ""

    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 5.5))
    fig.patch.set_facecolor(bg_color)
    if n == 1:
        axes = [axes]

    txt_color = 'white' if is_dark else 'black'

    sc_last = None
    for idx, src in enumerate(sources):
        ax = axes[idx]
        ax.set_facecolor(bg_color)
        emb = results[src]["embedding"]
        apop = results[src]["apoptosis"]

        if density_bg:
            _add_density_bg(ax, emb, alpha=0.25, bg_color=bg_color)
        sc_obj = ax.scatter(emb[:, 0], emb[:, 1], c=apop, cmap=cmap_mean,
                            s=dot_size, alpha=alpha, norm=norm,
                            edgecolors='none', rasterized=True)
        ax.set_title(f"{src}", fontsize=13, fontweight='bold', color=txt_color)
        ax.set_xlabel(axis_labels[0], fontsize=10, color=txt_color)
        ax.set_ylabel(axis_labels[1], fontsize=10, color=txt_color)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=8, colors=txt_color)
        for spine in ax.spines.values():
            spine.set_edgecolor(txt_color if is_dark else 'black')
        sc_last = sc_obj

    cbar = fig.colorbar(sc_last, ax=axes, shrink=0.7,
                        label=f"Apoptosis Rate{gamma_label}", pad=0.02)
    cbar.ax.yaxis.set_tick_params(color=txt_color)
    cbar.ax.yaxis.label.set_color(txt_color)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=txt_color)
    fig.suptitle(f"UMAP Scatter — {scope_name}", fontsize=13,
                 fontweight="bold", color=txt_color)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=bg_color)
    fig.savefig(output_path.replace(".png", ".svg"), bbox_inches="tight",
                facecolor=bg_color)
    if _IN_COLAB:
        plt.show()
    plt.close(fig)
    logger.info(f"  Saved UMAP scatter: {output_path}")


# ==============================================================================
# Plot 3a: Class Scatter — colored by genotype/superclass
# ==============================================================================
def plot_class_scatter(results, scope_name, output_path, dpi=200,
                       dot_size=3, alpha=0.5, bg_color='white',
                       axis_labels=("UMAP 1", "UMAP 2"),
                       density_bg=False,
                       class_colors=None):
    """Scatter colored by genotype class — CNN vs SAE side by side."""
    if class_colors is None:
        class_colors = MUTATION_COLORS

    sources = list(results.keys())
    n = len(sources)
    is_dark = bg_color.lower() not in ('white', '#ffffff', 'w')
    txt_color = 'white' if is_dark else 'black'

    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 5.5))
    fig.patch.set_facecolor(bg_color)
    if n == 1:
        axes = [axes]

    for idx, src in enumerate(sources):
        ax = axes[idx]
        ax.set_facecolor(bg_color)
        emb = results[src]["embedding"]
        sc_arr = results[src]["superclasses"]

        if density_bg:
            _add_density_bg(ax, emb, alpha=0.25, bg_color=bg_color)

        # Plot each class separately for proper legend
        classes = sorted(np.unique(sc_arr),
                         key=lambda c: (c != "Control", c))  # Control first
        for cls in classes:
            cls_mask = sc_arr == cls
            color = class_colors.get(cls, "#999999")
            ax.scatter(emb[cls_mask, 0], emb[cls_mask, 1],
                       c=color, s=dot_size, alpha=alpha,
                       edgecolors='none', rasterized=True,
                       label=f"{cls} ({cls_mask.sum():,})")

        ax.set_title(f"{src}", fontsize=13, fontweight='bold', color=txt_color)
        ax.set_xlabel(axis_labels[0], fontsize=10, color=txt_color)
        ax.set_ylabel(axis_labels[1], fontsize=10, color=txt_color)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=8, colors=txt_color)
        for spine in ax.spines.values():
            spine.set_edgecolor(txt_color if is_dark else 'black')
        leg = ax.legend(fontsize=8, loc='best', framealpha=0.7,
                        markerscale=2.0, scatterpoints=1)
        if is_dark:
            leg.get_frame().set_facecolor('#333333')
            for text in leg.get_texts():
                text.set_color('white')

    fig.suptitle(f"Class Scatter — {scope_name}", fontsize=13,
                 fontweight="bold", color=txt_color)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=bg_color)
    fig.savefig(output_path.replace(".png", ".svg"), bbox_inches="tight",
                facecolor=bg_color)
    if _IN_COLAB:
        plt.show()
    plt.close(fig)
    logger.info(f"  Saved class scatter: {output_path}")


# ==============================================================================
# Plot 3e: Subsample UMAP — apoptosis 무관, 모든 벡터에서 subsample (umap_gap 스타일)
# ==============================================================================
def plot_subsample_umap(X_all, superclasses_all, src_label, args, out_dir, psuffix,
                        ax_labels=("UMAP 1", "UMAP 2")):
    """
    세포사멸율 정보와 무관하게 전체 feature 벡터에서 subsample 후 UMAP.
    umap_gap.py 스타일 — genotype 색상, clean publication figure.

    Parameters
    ----------
    X_all          : (N_total, D) — 전체 feature 벡터 (apoptosis 매칭 불필요)
    superclasses_all : (N_total,) — 각 벡터의 genotype 레이블
    src_label      : str — 'CNN' or 'SAE'
    args           : argparse.Namespace
    out_dir        : str — output directory
    psuffix        : str — parameter suffix for filename
    ax_labels      : (str, str) — axis label tuple
    """
    rng = np.random.RandomState(args.seed)
    classes_avail = args.classes

    # ── 클래스별 subsample 계산 ──
    if args.subsample_per_class > 0:
        n_per_cls = args.subsample_per_class
    elif args.subsample_n > 0:
        n_cls = max(1, len(classes_avail))
        n_per_cls = args.subsample_n // n_cls
    else:
        return  # subsample_n=0 이면 skip

    keep_idx = []
    counts = {}
    for cls in classes_avail:
        cls_mask = np.where(superclasses_all == cls)[0]
        if len(cls_mask) == 0:
            continue
        n_take = min(n_per_cls, len(cls_mask))
        chosen = rng.choice(cls_mask, size=n_take, replace=False)
        keep_idx.extend(chosen.tolist())
        counts[cls] = n_take

    if len(keep_idx) < 10:
        logger.warning(f"  {src_label} subsample UMAP: 너무 적은 샘플 ({len(keep_idx)}), skip")
        return

    keep_idx = np.array(keep_idx)
    X_sub = X_all[keep_idx]
    sc_sub = superclasses_all[keep_idx]

    counts_str = ", ".join(f"{c}:{counts.get(c,0):,}" for c in classes_avail if c in counts)
    n_total = len(keep_idx)
    logger.info(f"  {src_label} subsample UMAP: n={n_total} [{counts_str}]")

    # ── Optional PCA before UMAP ──
    X_umap_in = X_sub
    if args.pca_dim > 0:
        from sklearn.decomposition import PCA
        n_comp = min(args.pca_dim, X_sub.shape[1], n_total - 1)
        pca = PCA(n_components=n_comp, random_state=args.seed)
        X_umap_in = pca.fit_transform(X_sub)
        logger.info(f"    PCA: {X_sub.shape[1]}D → {n_comp}D")

    # ── UMAP ──
    if not _HAS_UMAP:
        logger.warning("  umap-learn not installed. pip install umap-learn")
        return
    import time as _time
    logger.info(f"    UMAP (n={n_total}, n_neighbors={args.umap_n_neighbors}, "
                f"min_dist={args.umap_min_dist}, metric={args.umap_metric})...")
    t0 = _time.time()
    reducer = umap_lib.UMAP(
        n_components=2,
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        random_state=args.seed,
    )
    coords = reducer.fit_transform(X_umap_in)
    logger.info(f"    UMAP done in {_time.time()-t0:.1f}s: {coords.shape}")

    # ── Plot (umap_gap.py 스타일) ──
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    sns.despine()

    # Control 먼저 그리고 나머지 위에 (minority on top)
    plot_order = ["Control"] + [c for c in classes_avail if c != "Control"]
    for cls in plot_order:
        cls_mask = sc_sub == cls
        if cls_mask.sum() == 0:
            continue
        color = MUTATION_COLORS.get(cls, "#999999")
        ax.scatter(
            coords[cls_mask, 0], coords[cls_mask, 1],
            s=args.dot_size, alpha=args.alpha, rasterized=True,
            label=f"{cls} (n={cls_mask.sum():,})",
            c=color, edgecolors="none",
        )

    # Publication style — axes 없애기
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("equal", adjustable="datalim")

    # Info text
    info_text = (f"n={n_total:,}  dim={X_sub.shape[1]}\n"
                 f"nn={args.umap_n_neighbors}  min_dist={args.umap_min_dist}  "
                 f"{args.umap_metric}")
    ax.text(0.02, 0.02, info_text, transform=ax.transAxes,
            fontsize=8, verticalalignment="bottom", fontstyle="italic",
            color="#555555",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#cccccc", alpha=0.85))

    leg = ax.legend(loc="upper right", markerscale=4, fontsize=9,
                    frameon=True, framealpha=0.9, edgecolor="#cccccc",
                    handletextpad=0.3, borderpad=0.4)
    for lh in leg.legend_handles:
        lh.set_alpha(1.0)

    fig.suptitle(f"{src_label} — Subsample UMAP (no apoptosis filter)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(pad=0.3)

    png_path = os.path.join(out_dir, f"subsample_umap_{src_label}_{psuffix}_n{n_total}.png")
    svg_path = png_path.replace(".png", ".svg")
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    fig.savefig(svg_path, format="svg", bbox_inches="tight",
                facecolor="white", edgecolor="none")
    logger.info(f"  Saved subsample UMAP: {png_path}")

    if _IN_COLAB:
        plt.show()
    plt.close(fig)

    # NPZ 저장 (재사용 가능)
    npz_path = png_path.replace(".png", "_coords.npz")
    np.savez_compressed(npz_path, coords=coords, superclasses=sc_sub,
                        n_neighbors=args.umap_n_neighbors,
                        min_dist=args.umap_min_dist, metric=args.umap_metric,
                        n_total=n_total)
    logger.info(f"  Saved coords NPZ: {npz_path}")


# ==============================================================================
# Plot 3b: KNN Scatter — color = per-point KNN local_std (no grid needed)
# ==============================================================================
def _add_density_bg(ax, emb, cmap_name='Greys', alpha=0.3, levels=15, bg_color='white'):
    """Add a KDE density contour behind scatter dots to fill sparse regions."""
    from scipy.stats import gaussian_kde
    try:
        kde = gaussian_kde(emb.T, bw_method='scott')
        x_pad = (emb[:, 0].max() - emb[:, 0].min()) * 0.05
        y_pad = (emb[:, 1].max() - emb[:, 1].min()) * 0.05
        xg = np.linspace(emb[:, 0].min() - x_pad, emb[:, 0].max() + x_pad, 120)
        yg = np.linspace(emb[:, 1].min() - y_pad, emb[:, 1].max() + y_pad, 120)
        Xg, Yg = np.meshgrid(xg, yg)
        Z = kde(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)
        is_dark = bg_color.lower() not in ('white', '#ffffff', 'w')
        density_cmap = 'Greys' if is_dark else 'Greys_r'
        ax.contourf(Xg, Yg, Z, levels=levels, cmap=density_cmap,
                    alpha=alpha, zorder=0, antialiased=True)
    except (np.linalg.LinAlgError, ValueError):
        pass  # Skip if KDE fails


def plot_knn_scatter(results, scope_name, output_path, dpi=200,
                     gamma=1.0, dot_size=3, alpha=0.5,
                     cmap_std='inferno', cmap_mean='viridis', bg_color='white',
                     axis_labels=("UMAP 1", "UMAP 2"),
                     std_mode='std', density_bg=False, vmax_pctl=97.5):
    """
    Scatter colored by per-point KNN statistics.
    Top row: color = local_std or CV (std_mode).
    Bottom row: color = local_mean.
    """
    sources = list(results.keys())
    n = len(sources)
    is_dark = bg_color.lower() not in ('white', '#ffffff', 'w')
    txt_color = 'white' if is_dark else 'black'
    use_cv = (std_mode == 'cv')

    # Compute display values: raw std or CV (σ/μ)
    display_vals = {}
    for s in sources:
        stds = results[s]["pt_local_stds"]
        means = results[s]["pt_local_means"]
        if use_cv:
            safe_means = np.where(np.abs(means) < 1e-10, 1e-10, np.abs(means))
            display_vals[s] = stds / safe_means
        else:
            display_vals[s] = stds

    # Shared ranges
    all_disp = np.concatenate([display_vals[s] for s in sources])
    all_means = np.concatenate([results[s]["pt_local_means"] for s in sources])

    vmin_disp, vmax_disp = 0.0, float(np.percentile(all_disp, vmax_pctl))
    vmin_mean = float(np.percentile(all_means, 100 - vmax_pctl))
    vmax_mean = float(np.percentile(all_means, vmax_pctl))

    if use_cv:
        top_label = "KNN Local CV (σ/μ)"
        top_short = "CV"
    else:
        top_label = "KNN Local Std (σ)"
        top_short = "σ"

    if gamma != 1.0:
        norm_mean = PowerNorm(gamma=gamma, vmin=vmin_mean, vmax=vmax_mean, clip=True)
        gamma_label = f" (γ={gamma})"
    else:
        norm_mean = Normalize(vmin=vmin_mean, vmax=vmax_mean, clip=True)
        gamma_label = ""

    fig, axes = plt.subplots(2, n, figsize=(6.5 * n, 10))
    fig.patch.set_facecolor(bg_color)
    if n == 1:
        axes = axes.reshape(2, 1)

    sc_std_last = None
    sc_mean_last = None

    for idx, src in enumerate(sources):
        emb = results[src]["embedding"]
        pt_disp = display_vals[src]
        pt_means = results[src]["pt_local_means"]
        n_samp = len(emb)
        disp_avg = float(np.mean(pt_disp))

        # Top: local_std or CV
        ax = axes[0, idx]
        ax.set_facecolor(bg_color)
        if density_bg:
            _add_density_bg(ax, emb, alpha=0.25, bg_color=bg_color)
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=pt_disp, cmap=cmap_std,
                        s=dot_size, alpha=alpha, vmin=vmin_disp, vmax=vmax_disp,
                        edgecolors='none', rasterized=True)
        ax.set_title(f"{src} — {top_label}\n"
                     f"n={n_samp:,}, mean {top_short}={disp_avg:.4f}",
                     fontsize=11, fontweight='bold', color=txt_color)
        ax.set_xlabel(axis_labels[0], fontsize=9, color=txt_color)
        ax.set_ylabel(axis_labels[1], fontsize=9, color=txt_color)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=7, colors=txt_color)
        for spine in ax.spines.values():
            spine.set_edgecolor(txt_color if is_dark else 'black')
        sc_std_last = sc

        # Bottom: local_mean
        ax = axes[1, idx]
        ax.set_facecolor(bg_color)
        if density_bg:
            _add_density_bg(ax, emb, alpha=0.25, bg_color=bg_color)
        sc2 = ax.scatter(emb[:, 0], emb[:, 1], c=pt_means, cmap=cmap_mean,
                         s=dot_size, alpha=alpha, norm=norm_mean,
                         edgecolors='none', rasterized=True)
        ax.set_title(f"{src} — KNN Local Mean{gamma_label}",
                     fontsize=11, fontweight='bold', color=txt_color)
        ax.set_xlabel(axis_labels[0], fontsize=9, color=txt_color)
        ax.set_ylabel(axis_labels[1], fontsize=9, color=txt_color)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=7, colors=txt_color)
        for spine in ax.spines.values():
            spine.set_edgecolor(txt_color if is_dark else 'black')
        sc_mean_last = sc2

    for cb_sc, cb_axes, cb_label in [
        (sc_std_last, axes[0, :], top_label),
        (sc_mean_last, axes[1, :], f"KNN Local Mean{gamma_label}"),
    ]:
        cbar = fig.colorbar(cb_sc, ax=cb_axes.ravel().tolist(),
                            shrink=0.7, label=cb_label, pad=0.02)
        cbar.ax.yaxis.set_tick_params(color=txt_color)
        cbar.ax.yaxis.label.set_color(txt_color)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=txt_color)

    fig.suptitle(f"KNN Scatter — {scope_name}",
                 fontsize=13, fontweight="bold", color=txt_color)
    fig.subplots_adjust(left=0.06, right=0.88, top=0.92, bottom=0.05,
                         hspace=0.25, wspace=0.2)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=bg_color)
    fig.savefig(output_path.replace(".png", ".svg"), bbox_inches="tight",
                facecolor=bg_color)
    if _IN_COLAB:
        plt.show()
    plt.close(fig)
    logger.info(f"  Saved KNN scatter: {output_path}")


# ==============================================================================
# Plot 3c: KNN 3D Scatter — z=local_mean, color=local_std (no grid)
# ==============================================================================
def plot_knn_scatter_3d(results, scope_name, output_path, dot_size=3,
                        alpha=0.5, cmap_std='inferno',
                        elev=25, azim=-60, dpi=200,
                        axis_labels=("UMAP 1", "UMAP 2")):
    """
    3D scatter: x=UMAP1, y=UMAP2, z=KNN local_mean, color=KNN local_std.
    High-dim KNN → per-point values → direct 3D scatter (no grid).
    """
    sources = list(results.keys())
    n_sources = len(sources)

    # Shared ranges
    all_stds = np.concatenate([results[s]["pt_local_stds"] for s in sources])
    all_means = np.concatenate([results[s]["pt_local_means"] for s in sources])
    vmin_std, vmax_std = 0.0, float(np.percentile(all_stds, 97.5))
    z_min = float(np.percentile(all_means, 1))
    z_max = float(np.percentile(all_means, 99))
    z_margin = (z_max - z_min) * 0.05

    norm_std = Normalize(vmin=vmin_std, vmax=vmax_std, clip=True)
    cmap = cm.get_cmap(cmap_std)

    fig = plt.figure(figsize=(7.5 * n_sources + 1.5, 6.5))
    gs = gridspec.GridSpec(1, n_sources + 1,
                           width_ratios=[1] * n_sources + [0.05])

    for idx, src in enumerate(sources):
        ax = fig.add_subplot(gs[0, idx], projection='3d')

        emb = results[src]["embedding"]
        pt_stds = results[src]["pt_local_stds"]
        pt_means = results[src]["pt_local_means"]
        n_samp = len(emb)
        pt_avg_std = float(np.mean(pt_stds))

        colors = cmap(norm_std(pt_stds))

        ax.scatter(
            emb[:, 0], emb[:, 1], pt_means,
            c=colors, s=dot_size, alpha=alpha,
            edgecolors='none', depthshade=True,
            rasterized=True,
        )

        ax.set_xlabel(axis_labels[0], fontsize=9, labelpad=5)
        ax.set_ylabel(axis_labels[1], fontsize=9, labelpad=5)
        ax.set_zlabel("KNN Local Mean\n(Apoptosis)", fontsize=9, labelpad=5)
        ax.set_zlim(z_min - z_margin, z_max + z_margin)
        ax.view_init(elev=elev, azim=azim)

        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor((0.8, 0.8, 0.8, 0.5))
        ax.yaxis.pane.set_edgecolor((0.8, 0.8, 0.8, 0.5))
        ax.zaxis.pane.set_edgecolor((0.8, 0.8, 0.8, 0.5))
        ax.tick_params(labelsize=7)

        ax.text2D(0.02, 0.97,
                  f"{src}\nn = {n_samp:,}\nmean σ = {pt_avg_std:.4f}",
                  transform=ax.transAxes, fontsize=8,
                  verticalalignment='top',
                  bbox=dict(boxstyle='round,pad=0.4',
                            facecolor='white', alpha=0.85,
                            edgecolor=(0.7, 0.7, 0.7)))

    # Shared colorbar
    cbar_ax = fig.add_subplot(gs[0, -1])
    sm = cm.ScalarMappable(cmap=cmap, norm=norm_std)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("KNN Local Std (σ)", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle(f"3D KNN Manifold — {scope_name}",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.05, right=0.88, bottom=0.05, top=0.92,
                         wspace=0.15)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    for ext in [".svg", ".pdf"]:
        fig.savefig(output_path.replace(".png", ext), bbox_inches="tight")
    if _IN_COLAB:
        plt.show()
    plt.close(fig)
    logger.info(f"  Saved 3D KNN scatter: {output_path}")


# ==============================================================================
# Plot 3d: Interactive 3D Scatter (Plotly) — HTML output
# ==============================================================================
def plot_knn_scatter_3d_interactive(results, scope_name, output_path,
                                     dot_size=3, alpha=0.5, cmap_std='inferno',
                                     axis_labels=("UMAP 1", "UMAP 2")):
    """
    Interactive 3D scatter using Plotly.
    x=UMAP1, y=UMAP2, z=KNN local_mean, color=KNN local_std.
    Outputs .html file that can be opened in a browser.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        logger.warning("  plotly not installed — skipping interactive 3D. "
                       "Install with: pip install plotly")
        return

    sources = list(results.keys())
    n_sources = len(sources)

    # Shared range
    all_stds = np.concatenate([results[s]["pt_local_stds"] for s in sources])
    vmax_std = float(np.percentile(all_stds, 97.5))

    fig = make_subplots(
        rows=1, cols=n_sources,
        specs=[[{"type": "scatter3d"}] * n_sources],
        subplot_titles=[f"{s}" for s in sources],
        horizontal_spacing=0.02,
    )

    for idx, src in enumerate(sources):
        emb = results[src]["embedding"]
        pt_stds = results[src]["pt_local_stds"]
        pt_means = results[src]["pt_local_means"]
        apop = results[src]["apoptosis"]
        n_samp = len(emb)
        pt_avg_std = float(np.mean(pt_stds))

        hover_text = [
            f"{axis_labels[0]}: {emb[i,0]:.2f}<br>"
            f"{axis_labels[1]}: {emb[i,1]:.2f}<br>"
            f"KNN mean: {pt_means[i]:.4f}<br>"
            f"KNN std: {pt_stds[i]:.4f}<br>"
            f"Raw apoptosis: {apop[i]:.4f}"
            for i in range(n_samp)
        ]

        fig.add_trace(
            go.Scatter3d(
                x=emb[:, 0], y=emb[:, 1], z=pt_means,
                mode='markers',
                marker=dict(
                    size=max(1.0, dot_size * 0.5),
                    color=pt_stds,
                    colorscale=(cmap_std.capitalize() if isinstance(cmap_std, str)
                                else cmap_std.name.split('_trunc')[0].capitalize()),
                    cmin=0, cmax=vmax_std,
                    opacity=alpha,
                    colorbar=dict(
                        title="KNN σ",
                        thickness=15,
                        len=0.6,
                        x=1.02,
                    ) if idx == n_sources - 1 else None,
                    showscale=(idx == n_sources - 1),
                ),
                text=hover_text,
                hoverinfo='text',
                name=f"{src} (n={n_samp:,}, μσ={pt_avg_std:.4f})",
            ),
            row=1, col=idx + 1,
        )

    fig.update_layout(
        title=dict(
            text=f"Interactive 3D KNN Manifold — {scope_name}",
            font=dict(size=16),
        ),
        width=600 * n_sources + 100,
        height=600,
        margin=dict(l=10, r=10, t=60, b=10),
        showlegend=True,
        legend=dict(x=0.01, y=0.01, font=dict(size=10)),
    )

    # Apply same axis labels to all scenes
    for idx in range(n_sources):
        scene_key = f"scene{idx + 1}" if idx > 0 else "scene"
        fig.update_layout(**{
            scene_key: dict(
                xaxis_title="UMAP 1",
                yaxis_title="UMAP 2",
                zaxis_title="KNN Local Mean",
                aspectmode='auto',
            )
        })

    html_path = output_path.replace(".png", ".html")
    fig.write_html(html_path, include_plotlyjs='cdn')
    logger.info(f"  Saved interactive 3D: {html_path}")

    # Show inline in Colab
    if _IN_COLAB:
        fig.show()


# ==============================================================================
# Plot 4a: Interactive 3D Surface (Plotly) — grid-based surface HTML
# ==============================================================================
def plot_surface_3d_interactive(results, scope_name, output_path,
                                 axis_labels=("UMAP 1", "UMAP 2")):
    """
    Interactive 3D surface using Plotly.
    z = grid mean apoptosis rate,  surfacecolor = grid std.
    Outputs .html file for browser viewing.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        logger.warning("  plotly not installed — skipping interactive surface. "
                       "Install with: pip install plotly")
        return

    sources = list(results.keys())
    n_sources = len(sources)

    # Shared ranges
    valid_stds = []
    valid_means = []
    for s in sources:
        gs = results[s]["grid_std"]
        gm = results[s]["grid_mean"]
        if np.any(~np.isnan(gs)):
            valid_stds.append(gs[~np.isnan(gs)])
        if np.any(~np.isnan(gm)):
            valid_means.append(gm[~np.isnan(gm)])

    all_stds = np.concatenate(valid_stds)
    all_means = np.concatenate(valid_means)

    vmax_std = float(np.percentile(all_stds, 97.5))
    z_min = float(np.min(all_means))
    z_max = float(np.max(all_means))

    fig = make_subplots(
        rows=1, cols=n_sources,
        specs=[[{"type": "surface"}] * n_sources],
        subplot_titles=[f"{s}" for s in sources],
        horizontal_spacing=0.02,
    )

    for idx, src in enumerate(sources):
        res = results[src]
        grid_mean = res["grid_mean"]
        grid_std = res["grid_std"]
        x_edges = res["x_edges"]
        y_edges = res["y_edges"]

        grid_res = grid_mean.shape[0]
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

        roughness = res.get("roughness", np.nan)
        morans_i = res.get("morans_i", np.nan)
        n_samp = res.get("n_samples", 0)
        n_cells = res.get("n_valid_cells", 0)

        # Build hover text grid
        hover_text = [[
            f"{axis_labels[0]}: {x_centers[i]:.2f}<br>"
            f"{axis_labels[1]}: {y_centers[j]:.2f}<br>"
            f"Mean apoptosis: {grid_mean[i, j]:.4f}<br>"
            f"Std: {grid_std[i, j]:.4f}"
            if not np.isnan(grid_mean[i, j]) else ""
            for j in range(grid_res)
        ] for i in range(grid_res)]

        fig.add_trace(
            go.Surface(
                x=x_centers,
                y=y_centers,
                z=grid_mean,
                surfacecolor=grid_std,
                colorscale='Inferno',
                cmin=0, cmax=vmax_std,
                text=hover_text,
                hoverinfo='text',
                colorbar=dict(
                    title="Local Std (σ)",
                    thickness=15,
                    len=0.6,
                    x=1.02,
                ) if idx == n_sources - 1 else None,
                showscale=(idx == n_sources - 1),
                name=f"{src} (n={n_samp:,})",
                connectgaps=False,
            ),
            row=1, col=idx + 1,
        )

    fig.update_layout(
        title=dict(
            text=f"Interactive 3D Surface — {scope_name}",
            font=dict(size=16),
        ),
        width=600 * n_sources + 100,
        height=600,
        margin=dict(l=10, r=10, t=60, b=10),
        showlegend=True,
        legend=dict(x=0.01, y=0.01, font=dict(size=10)),
    )

    for idx in range(n_sources):
        scene_key = f"scene{idx + 1}" if idx > 0 else "scene"
        fig.update_layout(**{
            scene_key: dict(
                xaxis_title=axis_labels[0],
                yaxis_title=axis_labels[1],
                zaxis_title="Apoptosis Rate",
                zaxis=dict(range=[z_min, z_max]),
                aspectmode='auto',
            )
        })

    html_path = output_path.replace(".png", ".html")
    fig.write_html(html_path, include_plotlyjs='cdn')
    logger.info(f"  Saved interactive 3D surface: {html_path}")

    if _IN_COLAB:
        fig.show()


# ==============================================================================
# Plot 4: Metrics Bar — Roughness + Moran's I across scopes
# ==============================================================================
def plot_metrics_bar(all_metrics, output_path, dpi=200):
    """Grouped bar chart: Roughness + Moran's I across scopes for CNN vs SAE."""
    # Collect unique scopes and sources in order
    scopes = []
    for m in all_metrics:
        if m["scope"] not in scopes:
            scopes.append(m["scope"])
    sources_found = []
    for m in all_metrics:
        if m["source"] not in sources_found:
            sources_found.append(m["source"])

    n_scopes = len(scopes)
    n_sources = len(sources_found)
    bar_width = 0.35
    x = np.arange(n_scopes)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6 + 3 * n_scopes, 5))

    for si, src in enumerate(sources_found):
        roughness_vals = []
        morans_vals = []
        for scope in scopes:
            entry = next((m for m in all_metrics
                          if m["scope"] == scope and m["source"] == src), None)
            roughness_vals.append(entry["roughness"] if entry else 0)
            morans_vals.append(entry["morans_i"] if entry else 0)

        offset = (si - (n_sources - 1) / 2) * bar_width
        color = SOURCE_COLORS.get(src, "gray")

        ax1.bar(x + offset, roughness_vals, bar_width,
                label=src, color=color, alpha=0.85, edgecolor='white',
                linewidth=0.8)
        ax2.bar(x + offset, morans_vals, bar_width,
                label=src, color=color, alpha=0.85, edgecolor='white',
                linewidth=0.8)

        # Value annotations
        for xi, val in zip(x + offset, roughness_vals):
            if not np.isnan(val) and val > 0:
                ax1.text(xi, val + max(roughness_vals) * 0.02,
                         f"{val:.4f}", ha='center', va='bottom', fontsize=7)
        for xi, val in zip(x + offset, morans_vals):
            if not np.isnan(val):
                ax2.text(xi, val + abs(max(morans_vals)) * 0.02,
                         f"{val:.3f}", ha='center', va='bottom', fontsize=7)

    for ax, title, ylabel in [
        (ax1, "Surface Roughness (mean |∇z|)", "Roughness"),
        (ax2, "Moran's I (Spatial Autocorrelation)", "Moran's I"),
    ]:
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(scopes, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.15, axis='y')
        sns.despine(ax=ax)

    fig.suptitle("Manifold Surface Metrics — CNN vs SAE",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(output_path.replace(".png", ".svg"), bbox_inches="tight")
    if _IN_COLAB:
        plt.show()
    plt.close(fig)
    logger.info(f"  Saved metrics bar: {output_path}")


# ==============================================================================
# 2D Embedding Dispatcher — UMAP / t-SNE / Laplacian Eigenmap
# ==============================================================================
DR_AXIS_LABELS = {
    "umap": ("UMAP 1", "UMAP 2"),
    "tsne": ("t-SNE 1", "t-SNE 2"),
    "le":   ("LE 1", "LE 2"),
    "lle":  ("LLE 1", "LLE 2"),
}


def fit_2d_embedding(X, args):
    """
    Fit a 2D embedding using the method specified by args.dr_method.

    Parameters
    ----------
    X : (N, D) ndarray — features (PCA-reduced or raw)
    args : argparse.Namespace

    Returns
    -------
    embedding : (N, 2) ndarray
    """
    import time as _time
    method = args.dr_method.lower()

    if method == "umap":
        if not _HAS_UMAP:
            raise ImportError("umap-learn required. Install: pip install umap-learn")
        logger.info(f"    UMAP (n_neighbors={args.umap_n_neighbors}, "
                    f"min_dist={args.umap_min_dist}, "
                    f"metric={args.umap_metric})...")
        reducer = umap_lib.UMAP(
            n_components=2,
            n_neighbors=args.umap_n_neighbors,
            min_dist=args.umap_min_dist,
            metric=args.umap_metric,
            random_state=args.seed,
        )
        t0 = _time.time()
        emb = reducer.fit_transform(X)
        logger.info(f"    UMAP completed in {_time.time() - t0:.1f}s")
        return emb

    elif method == "tsne":
        logger.info(f"    t-SNE (perplexity={args.tsne_perplexity}, "
                    f"lr={args.tsne_lr}, N={X.shape[0]})...")
        t0 = _time.time()

        # Priority 1: openTSNE (FFT-accelerated, ~5-10x faster than sklearn)
        try:
            from openTSNE import TSNE as OpenTSNE
            tsne = OpenTSNE(
                n_components=2,
                perplexity=args.tsne_perplexity,
                learning_rate=args.tsne_lr if args.tsne_lr > 0 else "auto",
                initialization="pca",
                random_state=args.seed,
                n_jobs=-1,
            )
            emb = np.asarray(tsne.fit(X))
            logger.info(f"    t-SNE (openTSNE/FFT) completed in {_time.time() - t0:.1f}s")
            return emb
        except ImportError:
            pass

        # Priority 2: cuML GPU t-SNE (RAPIDS)
        try:
            from cuml.manifold import TSNE as cuTSNE
            tsne = cuTSNE(
                n_components=2,
                perplexity=args.tsne_perplexity,
                learning_rate=args.tsne_lr if args.tsne_lr > 0 else 200.0,
                random_state=args.seed,
            )
            emb = tsne.fit_transform(X)
            logger.info(f"    t-SNE (cuML/GPU) completed in {_time.time() - t0:.1f}s")
            return np.asarray(emb)
        except ImportError:
            pass

        # Fallback: sklearn (slowest)
        logger.info("    Using sklearn t-SNE (slow). "
                     "Install openTSNE for ~5-10x speedup: pip install openTSNE")
        from sklearn.manifold import TSNE
        tsne = TSNE(
            n_components=2,
            perplexity=args.tsne_perplexity,
            learning_rate=args.tsne_lr if args.tsne_lr > 0 else "auto",
            init="pca",
            random_state=args.seed,
            n_jobs=-1,
        )
        emb = tsne.fit_transform(X)
        logger.info(f"    t-SNE (sklearn) completed in {_time.time() - t0:.1f}s")
        return emb

    elif method == "le":
        logger.info(f"    Laplacian Eigenmap (n_neighbors={args.le_n_neighbors}, "
                    f"affinity={args.le_affinity}, N={X.shape[0]})...")
        t0 = _time.time()

        # Priority 1: cuML GPU SpectralEmbedding
        try:
            from cuml.manifold import SpectralEmbedding as cuSpectral
            se = cuSpectral(
                n_components=2,
                n_neighbors=args.le_n_neighbors,
                random_state=args.seed,
            )
            emb = se.fit_transform(X)
            logger.info(f"    LE (cuML/GPU) completed in {_time.time() - t0:.1f}s")
            return np.asarray(emb)
        except ImportError:
            pass

        # Fallback: sklearn
        logger.info("    Using sklearn SpectralEmbedding. "
                     "Install cuml (RAPIDS) for GPU acceleration.")
        from sklearn.manifold import SpectralEmbedding
        se = SpectralEmbedding(
            n_components=2,
            n_neighbors=args.le_n_neighbors,
            affinity=args.le_affinity,
            random_state=args.seed,
            n_jobs=-1,
        )
        emb = se.fit_transform(X)
        logger.info(f"    LE (sklearn) completed in {_time.time() - t0:.1f}s")
        return emb

    elif method == "lle":
        from sklearn.manifold import LocallyLinearEmbedding
        logger.info(f"    LLE (n_neighbors={args.lle_n_neighbors}, "
                    f"method={args.lle_method}, N={X.shape[0]})...")
        t0 = _time.time()
        lle = LocallyLinearEmbedding(
            n_components=2,
            n_neighbors=args.lle_n_neighbors,
            method=args.lle_method,
            random_state=args.seed,
            n_jobs=-1,
        )
        emb = lle.fit_transform(X)
        logger.info(f"    LLE (sklearn) completed in {_time.time() - t0:.1f}s, "
                    f"reconstruction_error={lle.reconstruction_error_:.6f}")
        return emb


    else:
        raise ValueError(f"Unknown DR method: {method}. "
                         f"Choose from: umap, tsne, le, lle")


# ==============================================================================
# Main
# ==============================================================================
def _param_suffix(args):
    """Build a short parameter tag for filenames so different runs are distinguishable."""
    parts = []
    parts.append(args.dr_method)  # Always include DR method
    if args.pca_dim > 0:
        parts.append(f"pca{args.pca_dim}")
    parts.append(f"knn{args.knn_k}")
    # DR-method-specific params
    if args.dr_method == "umap":
        parts.append(f"nn{args.umap_n_neighbors}")
        parts.append(f"md{args.umap_min_dist}")
    elif args.dr_method == "tsne":
        parts.append(f"pp{args.tsne_perplexity}")
    elif args.dr_method == "le":
        parts.append(f"len{args.le_n_neighbors}")
    elif args.dr_method == "lle":
        parts.append(f"llen{args.lle_n_neighbors}")
        if args.lle_method != "standard":
            parts.append(args.lle_method)
    if args.gap_l2_norm:
        parts.append("l2")
    if args.gamma != 1.0:
        parts.append(f"g{args.gamma}")
    return "_".join(parts)


def main():
    args = get_args()
    np.random.seed(args.seed)
    psuffix = _param_suffix(args)
    ax_labels = DR_AXIS_LABELS.get(args.dr_method, ("Dim 1", "Dim 2"))

    if not args.cnn_cache and not args.sae_cache:
        raise ValueError("At least one of --cnn_cache or --sae_cache required")

    # ── Output directory ──
    if args.output_dir:
        out_dir = args.output_dir
    else:
        ref = args.cnn_cache or args.sae_cache
        out_dir = os.path.join(os.path.dirname(ref), "manifold_surface_3d")
    os.makedirs(out_dir, exist_ok=True)

    # ── Truncate colormaps if requested ──
    # Keep original string names for plotly (which needs strings, not cmap objects)
    args.cmap_mean_str = args.cmap_mean
    args.cmap_std_str = args.cmap_std
    if args.cmap_start > 0.0 or args.cmap_end < 1.0:
        args.cmap_mean = truncate_cmap(args.cmap_mean, args.cmap_start, args.cmap_end)
        args.cmap_std = truncate_cmap(args.cmap_std, args.cmap_start, args.cmap_end)
        logger.info(f"  Colormap truncated: [{args.cmap_start:.2f}, {args.cmap_end:.2f}]")

    # ── Load features — minimal preprocessing (raw + optional L2 norm) ──
    sources = {}  # label → (X, superclasses, apoptosis)

    def _load_and_preprocess(cache_path):
        """Load cache + optional L2 norm. NO filter/norm/PCA — raw features."""
        X, lines, uids, label = load_cache(cache_path, args.dead_threshold)

        if args.pre_l2_norm:
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            X = X / np.where(norms == 0, 1e-12, norms)
            logger.info(f"  Applied pre-L2 normalization")

        if args.divide_hw > 0:
            X = X / args.divide_hw
            logger.info(f"  Divided by H*W={args.divide_hw}")

        if args.gap_l2_norm:
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            X = X / np.where(norms == 0, 1e-12, norms)
            logger.info(f"  Applied L2 normalization")

        superclasses = np.array([SUPERCLASS_MAP.get(ln, ln) for ln in lines])
        logger.info(f"  rate_col arg = {repr(args.rate_col)}")
        apoptosis = load_and_match_apoptosis(args.apoptosis_csv, uids,
                                              rate_col=args.rate_col)
        valid_apop = apoptosis[np.isfinite(apoptosis)]
        if len(valid_apop) > 0:
            logger.info(f"  ★ Rate values: min={valid_apop.min():.4f}, "
                        f"max={valid_apop.max():.4f}, "
                        f"mean={valid_apop.mean():.4f}, "
                        f"median={np.median(valid_apop):.4f}, "
                        f"zeros={np.sum(valid_apop == 0)}/{len(valid_apop)}")
        return X, superclasses, apoptosis, label

    if args.cnn_cache:
        logger.info("Loading CNN cache...")
        X_cnn, sc_cnn, apop_cnn, _ = _load_and_preprocess(args.cnn_cache)
        sources["CNN"] = (X_cnn, sc_cnn, apop_cnn)
        logger.info(f"  CNN features: {X_cnn.shape}")

    if args.sae_cache:
        logger.info("Loading SAE cache...")
        X_sae, sc_sae, apop_sae, _ = _load_and_preprocess(args.sae_cache)
        sources["SAE"] = (X_sae, sc_sae, apop_sae)
        logger.info(f"  SAE features: {X_sae.shape}")

    # ── Subsample UMAP (apoptosis 무관, --subsample_n > 0 일 때만 실행) ──
    if args.subsample_n > 0 or args.subsample_per_class > 0:
        logger.info(f"\n{'='*60}")
        logger.info("  [Subsample UMAP] apoptosis 무관 전체벡터 UMAP")
        logger.info(f"{'='*60}")
        # sources에 들어 있는 원본 전체 벡터 사용 (apoptosis 필터 없음)
        _raw_sources = {}
        if args.cnn_cache:
            X_cnn_raw, lines_cnn_raw, _, _ = load_cache(args.cnn_cache, args.dead_threshold)
            if args.gap_l2_norm:
                norms = np.linalg.norm(X_cnn_raw, axis=1, keepdims=True)
                X_cnn_raw = X_cnn_raw / np.where(norms == 0, 1e-12, norms)
            sc_cnn_raw = np.array([SUPERCLASS_MAP.get(ln, ln) for ln in lines_cnn_raw])
            _raw_sources["CNN"] = (X_cnn_raw, sc_cnn_raw)
        if args.sae_cache:
            X_sae_raw, lines_sae_raw, _, _ = load_cache(args.sae_cache, args.dead_threshold)
            if args.gap_l2_norm:
                norms = np.linalg.norm(X_sae_raw, axis=1, keepdims=True)
                X_sae_raw = X_sae_raw / np.where(norms == 0, 1e-12, norms)
            sc_sae_raw = np.array([SUPERCLASS_MAP.get(ln, ln) for ln in lines_sae_raw])
            _raw_sources["SAE"] = (X_sae_raw, sc_sae_raw)

        for src_lbl, (X_raw, sc_raw) in _raw_sources.items():
            # classes 필터 적용
            cls_mask = np.isin(sc_raw, args.classes)
            X_raw_f = X_raw[cls_mask]
            sc_raw_f = sc_raw[cls_mask]
            logger.info(f"  {src_lbl}: total={len(X_raw_f):,} (classes={args.classes})")
            plot_subsample_umap(
                X_raw_f, sc_raw_f, src_lbl, args, out_dir, psuffix,
                ax_labels=ax_labels,
            )

    # ── Analysis scopes ──
    classes = args.classes
    mutations = [c for c in classes if c != "Control"]
    # "All" scope uses all selected classes; per-mutation scopes filter individually
    scopes = [("_".join(classes), None)]
    if len(mutations) > 1:
        scopes += [(m, m) for m in mutations]
    logger.info(f"  Classes: {classes}")
    logger.info(f"  Scopes:  {[s[0] for s in scopes]}")

    all_scope_metrics = []

    for scope_name, mut_filter in scopes:
        logger.info(f"\n{'='*60}")
        logger.info(f"  Scope: {scope_name}")
        logger.info(f"{'='*60}")

        scope_results = {}
        scope_results_knn = {}

        for src_label, (X, superclasses, apoptosis) in sources.items():
            # Filter: valid apoptosis only (+ optional mutation filter)
            if mut_filter:
                mask = (superclasses == mut_filter) & np.isfinite(apoptosis)
            else:
                mask = np.isin(superclasses, classes) & np.isfinite(apoptosis)

            X_valid = X[mask]
            apop_valid = apoptosis[mask]
            sc_valid = superclasses[mask]
            n_valid = len(X_valid)

            if n_valid < 50:
                logger.warning(f"  {src_label}: too few samples ({n_valid}), skipping")
                continue

            logger.info(f"  {src_label}: n={n_valid}, dim={X_valid.shape[1]}")

            # ── Subsample per class if requested ──
            if args.samples_per_class > 0:
                rng = np.random.RandomState(args.seed)
                keep = []
                for cls in sorted(np.unique(sc_valid)):
                    cls_idx = np.where(sc_valid == cls)[0]
                    n_take = min(args.samples_per_class, len(cls_idx))
                    chosen = rng.choice(cls_idx, size=n_take, replace=False)
                    keep.extend(chosen.tolist())
                keep = sorted(keep)
                X_valid = X_valid[keep]
                apop_valid = apop_valid[keep]
                sc_valid = sc_valid[keep]
                n_valid = len(X_valid)
                logger.info(f"    Subsampled to {n_valid}")

            # ── Step 1: Optional PCA ──
            orig_dim = X_valid.shape[1]
            if args.pca_dim > 0:
                from sklearn.decomposition import PCA
                n_comp = min(args.pca_dim, X_valid.shape[1], n_valid)
                pca = PCA(n_components=n_comp, random_state=args.seed)
                X_valid = pca.fit_transform(X_valid)
                var_explained = float(np.sum(pca.explained_variance_ratio_) * 100)
                logger.info(f"    PCA: {orig_dim}D → {n_comp}D "
                           f"({var_explained:.1f}% variance)")

            # ── Step 2: KNN in PCA space (or raw if no PCA) ──
            logger.info(f"    KNN (k={args.knn_k}, D={X_valid.shape[1]})...")
            pt_means, pt_stds = compute_knn_local_stats(
                X_valid, apop_valid, args.knn_k)
            pt_mean_std = float(np.mean(pt_stds))
            logger.info(f"      All {n_valid} points → mean local_std: {pt_mean_std:.6f}")

            # ── Step 3: 2D embedding ──
            embedding = fit_2d_embedding(X_valid, args)
            logger.info(f"    {args.dr_method.upper()} done: {embedding.shape}")

            # ── Step 4a: Grid statistics (on UMAP embedding) ──
            grid_mean, grid_std, grid_count, x_edges, y_edges = \
                compute_grid_stats(embedding, apop_valid,
                                   args.grid_resolution,
                                   args.min_samples_per_cell)

            n_valid_cells = int(np.sum(~np.isnan(grid_mean)))
            n_empty = args.grid_resolution**2 - n_valid_cells
            logger.info(f"    Grid: {args.grid_resolution}×{args.grid_resolution}, "
                       f"valid={n_valid_cells}, empty={n_empty}")

            roughness = compute_surface_roughness(grid_mean)
            morans_i, morans_expected, morans_n = compute_morans_i(grid_mean)
            mean_grid_std = float(np.nanmean(grid_std))

            logger.info(f"    Grid roughness={roughness:.6f}, "
                       f"Moran's I={morans_i:.4f}, mean_std={mean_grid_std:.6f}")

            scope_results[src_label] = {
                "grid_mean": grid_mean,
                "grid_std": grid_std,
                "grid_count": grid_count,
                "x_edges": x_edges,
                "y_edges": y_edges,
                "roughness": roughness,
                "morans_i": morans_i,
                "morans_expected": morans_expected,
                "n_valid_cells": n_valid_cells,
                "n_samples": n_valid,
                "embedding": embedding,
                "apoptosis": apop_valid,
                "superclasses": sc_valid,
                "mean_grid_std": mean_grid_std,
            }

            all_scope_metrics.append({
                "scope": scope_name,
                "source": src_label,
                "method": "grid",
                "n_samples": n_valid,
                "n_valid_cells": n_valid_cells,
                "roughness": roughness,
                "morans_i": morans_i,
                "morans_expected": morans_expected,
                "mean_grid_std": mean_grid_std,
            })

            # ── Step 4b: KNN per-point → interpolate to UMAP grid ──
            knn_grid_mean, knn_xe, knn_ye = interpolate_to_surface_grid(
                embedding, pt_means, args.grid_resolution)
            knn_grid_std, _, _ = interpolate_to_surface_grid(
                embedding, pt_stds, args.grid_resolution)
            if knn_grid_std is not None:
                knn_grid_std = np.where(np.isnan(knn_grid_std), np.nan,
                                        np.clip(knn_grid_std, 0, None))

            knn_valid_cells = int(np.sum(~np.isnan(knn_grid_mean)))
            knn_roughness = compute_surface_roughness(knn_grid_mean)
            knn_morans_i, knn_morans_exp, _ = compute_morans_i(knn_grid_mean)
            knn_grid_mean_std = float(np.nanmean(knn_grid_std)) if knn_grid_std is not None else np.nan

            logger.info(f"      UMAP grid: {knn_valid_cells}/{args.grid_resolution**2} "
                        f"cells (convex hull coverage)")
            logger.info(f"      Grid roughness={knn_roughness:.6f}, "
                        f"Moran's I={knn_morans_i:.4f}")

            scope_results_knn[src_label] = {
                "grid_mean": knn_grid_mean,
                "grid_std": knn_grid_std,
                "x_edges": knn_xe,
                "y_edges": knn_ye,
                "roughness": knn_roughness,
                "morans_i": knn_morans_i,
                "morans_expected": knn_morans_exp,
                "n_valid_cells": knn_valid_cells,
                "n_samples": n_valid,
                "embedding": embedding,
                "apoptosis": apop_valid,
                "mean_grid_std": knn_grid_mean_std,
                "pt_mean_std": pt_mean_std,
                "pt_local_stds": pt_stds,
                "pt_local_means": pt_means,
            }

            all_scope_metrics.append({
                "scope": scope_name,
                "source": src_label,
                "method": "knn_highd",
                "n_samples": n_valid,
                "n_valid_cells": knn_valid_cells,
                "roughness": knn_roughness,
                "morans_i": knn_morans_i,
                "morans_expected": knn_morans_exp,
                "mean_grid_std": knn_grid_mean_std,
                "pt_mean_std": pt_mean_std,
            })

        # ── Generate plots for this scope ──

        # === PRIMARY: KNN scatter 2D + 3D (all points, no grid) ===
        if len(scope_results_knn) > 0:
            plot_knn_scatter(
                scope_results_knn, f"{scope_name} (KNN k={args.knn_k})",
                os.path.join(out_dir, f"knn_local_std_{scope_name}_{psuffix}.png"),
                dpi=args.dpi, gamma=args.gamma, dot_size=args.dot_size,
                alpha=args.alpha, cmap_std=args.cmap_std,
                cmap_mean=args.cmap_mean, bg_color=args.bg_color,
                axis_labels=ax_labels,
                std_mode=args.std_mode, density_bg=args.density_bg,
                vmax_pctl=args.vmax_pctl,
            )
            plot_knn_scatter_3d(
                scope_results_knn, f"{scope_name} (KNN k={args.knn_k})",
                os.path.join(out_dir, f"knn_3d_scatter_{scope_name}_{psuffix}.png"),
                elev=args.elev, azim=args.azim, dpi=args.dpi,
                dot_size=args.dot_size, alpha=args.alpha,
                cmap_std=args.cmap_std, axis_labels=ax_labels,
            )
            plot_knn_scatter_3d_interactive(
                scope_results_knn, f"{scope_name} (KNN k={args.knn_k})",
                os.path.join(out_dir, f"knn_3d_scatter_{scope_name}_{psuffix}.png"),
                dot_size=args.dot_size, alpha=args.alpha,
                cmap_std=args.cmap_std, axis_labels=ax_labels,
            )

        # === CONTEXT: Scatter colored by apoptosis rate ===
        if len(scope_results) > 0:
            plot_umap_scatter(
                scope_results, scope_name,
                os.path.join(out_dir, f"umap_scatter_{scope_name}_{psuffix}.png"),
                dpi=args.dpi, gamma=args.gamma, dot_size=args.dot_size,
                alpha=args.alpha, cmap_mean=args.cmap_mean,
                bg_color=args.bg_color, axis_labels=ax_labels,
                density_bg=args.density_bg, vmax_pctl=args.vmax_pctl,
            )
            plot_class_scatter(
                scope_results, scope_name,
                os.path.join(out_dir, f"class_scatter_{scope_name}_{psuffix}.png"),
                dpi=args.dpi, dot_size=args.dot_size,
                alpha=args.alpha, bg_color=args.bg_color,
                axis_labels=ax_labels, density_bg=args.density_bg,
            )

        # === SUPPLEMENTARY: Grid-based 3D surface + heatmap ===
        if len(scope_results) > 0:
            plot_manifold_surface_3d(
                scope_results, f"{scope_name} (Grid)",
                os.path.join(out_dir, f"suppl_surface_3d_{scope_name}_grid_{psuffix}.png"),
                elev=args.elev, azim=args.azim, dpi=args.dpi,
                axis_labels=ax_labels,
            )
            plot_surface_3d_interactive(
                scope_results, f"{scope_name} (Grid)",
                os.path.join(out_dir, f"suppl_surface_3d_{scope_name}_grid_{psuffix}.png"),
                axis_labels=ax_labels,
            )
            plot_manifold_heatmap_2d(
                scope_results, f"{scope_name} (Grid)",
                os.path.join(out_dir, f"suppl_heatmap_2d_{scope_name}_grid_{psuffix}.png"),
                dpi=args.dpi, axis_labels=ax_labels,
            )
        if len(scope_results_knn) > 0:
            plot_manifold_surface_3d(
                scope_results_knn, f"{scope_name} (KNN k={args.knn_k})",
                os.path.join(out_dir, f"suppl_surface_3d_{scope_name}_knn_{psuffix}.png"),
                elev=args.elev, azim=args.azim, dpi=args.dpi,
                axis_labels=ax_labels,
            )
            plot_surface_3d_interactive(
                scope_results_knn, f"{scope_name} (KNN k={args.knn_k})",
                os.path.join(out_dir, f"suppl_surface_3d_{scope_name}_knn_{psuffix}.png"),
                axis_labels=ax_labels,
            )
            plot_manifold_heatmap_2d(
                scope_results_knn, f"{scope_name} (KNN k={args.knn_k})",
                os.path.join(out_dir, f"suppl_heatmap_2d_{scope_name}_knn_{psuffix}.png"),
                dpi=args.dpi, axis_labels=ax_labels,
            )

    # ── Metrics bar chart (all scopes combined) ──
    if all_scope_metrics:
        plot_metrics_bar(
            all_scope_metrics,
            os.path.join(out_dir, "manifold_metrics_bar.png"),
            dpi=args.dpi,
        )

    # ── Summary table ──
    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY — Semantic Manifold Surface")
    logger.info(f"{'='*80}")
    logger.info(f"  {'Source':6s} {'Method':12s} {'Scope':15s} {'n':<7s} {'Cells':<7s} "
                f"{'Roughness':<12s} {'Moran I':<10s} {'MeanStd':<10s}")
    logger.info("  " + "-" * 90)
    for m in all_scope_metrics:
        logger.info(
            f"  {m['source']:6s} {m.get('method','grid'):12s} "
            f"{m['scope']:15s} {m['n_samples']:<7d} "
            f"{m['n_valid_cells']:<7d} "
            f"{m['roughness']:<12.6f} "
            f"{m['morans_i']:<10.4f} "
            f"{m['mean_grid_std']:<10.6f}"
        )

    # ── Save JSON ──
    json_path = os.path.join(out_dir, "manifold_surface_results.json")
    with open(json_path, "w") as f:
        json.dump({
            "args": {k: str(v) if not isinstance(v, (int, float, bool, type(None)))
                     else v for k, v in vars(args).items()},
            "metrics": [
                {k: (float(v) if isinstance(v, (float, np.floating)) else v)
                 for k, v in m.items()}
                for m in all_scope_metrics
            ],
        }, f, indent=2)
    logger.info(f"\n  Saved JSON: {json_path}")
    logger.info(f"  Output dir: {out_dir}")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()
