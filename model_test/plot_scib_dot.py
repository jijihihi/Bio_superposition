# ==============================================================================
# scib Dot Heatmap (Nature Methods style)
#
# Reads scib_results.json across all CNN layers + SAE seeds,
# averages per condition (CNN layer / SAE), and produces a
# dot-heatmap in the style of scIB benchmarking (Luecken et al. 2022).
#
# Rows: CNN stage5_mid, CNN stage5_out, CNN refine_out, SAE
# Columns: ASW, NMI, ARI, cLISI, Graph Connectivity, Overall
#
# Dots: size ∝ value, color ∝ value, text annotation inside dot
#
# Usage:
#   python -m model_test.plot_scib_dot \
#       --base_dir /content/drive/.../scib_eval \
#       --output_dir /content/drive/.../scib_eval/plots
#
# %matplotlib inline
# import logging
# logging.basicConfig(level=logging.INFO, force=True)
#
# !python -m model_test.plot_scib_dot \
#     --base_dir /content/drive/MyDrive/Final_paper/lambda_labs_moco_only/scib_eval \
#     --output_dir /content/drive/MyDrive/Final_paper/lambda_labs_moco_only/scib_eval/plots
# ==============================================================================

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

import matplotlib
import numpy as np

_IN_COLAB = ("google.colab" in sys.modules) or os.path.isdir("/content")
if not _IN_COLAB:
    matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from sae_project.step02_logging_utils import get_logger

logger = get_logger("scib_dot")

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.family"] = "sans-serif"


# ── Metric display config ──
METRIC_KEYS = ["asw", "nmi", "ari", "clisi", "graph_conn"]
METRIC_DISPLAY = {
    "asw": "ASW",
    "nmi": "NMI",
    "ari": "ARI",
    "clisi": "cLISI",
    "graph_conn": "Graph\nConn.",
    "overall": "Overall",
}

# Row config
ROW_ORDER = ["stage5_mid", "stage5_out", "refine_out", "SAE"]
ROW_DISPLAY = {
    "stage5_mid": "CNN stage5_mid",
    "stage5_out": "CNN stage5_out",
    "refine_out": "CNN refine_out",
    "SAE": "SAE (stage5_out)",
}

# Colors per row
ROW_COLORS = {
    "stage5_mid": "#88BEDC",
    "stage5_out": "#3A7EBF",
    "refine_out": "#1B4876",
    "SAE": "#E8833A",
}


# ==============================================================================
# Scan and aggregate results
# ==============================================================================
def scan_scib_results(base_dir):
    """Scan scib_results.json files and aggregate per condition.

    Expected directory structure:
      cnn/{layer}/seed_{S}/scib_results.json
      sae/sae_seed_{S}/scib_results.json
      comparison/sae_seed_{S}/scib_results.json  (optional)

    Returns: {condition_key: {metric: [values across seeds]}}
    """
    data = defaultdict(lambda: defaultdict(list))

    # ── CNN layers ──
    for layer in ["stage5_mid", "stage5_out", "refine_out"]:
        layer_dir = os.path.join(base_dir, "cnn", layer)
        if not os.path.isdir(layer_dir):
            continue

        for seed_dir in sorted(glob.glob(os.path.join(layer_dir, "seed_*"))):
            json_path = os.path.join(seed_dir, "scib_results.json")
            if not os.path.isfile(json_path):
                continue

            with open(json_path, "r", encoding="utf-8") as f:
                results = json.load(f)

            # Find the CNN entry in results
            for label, metrics in results.items():
                if not isinstance(metrics, dict):
                    continue
                # Match CNN entries
                if "CNN" in label or label.startswith("CNN"):
                    for mk in METRIC_KEYS:
                        val = metrics.get(mk)
                        if val is not None and val != "nan":
                            data[layer][mk].append(float(val))
                    break  # only one CNN entry per file

    # ── SAE ──
    sae_dir = os.path.join(base_dir, "sae")
    if os.path.isdir(sae_dir):
        for seed_dir in sorted(glob.glob(os.path.join(sae_dir, "sae_seed_*"))):
            json_path = os.path.join(seed_dir, "scib_results.json")
            if not os.path.isfile(json_path):
                continue

            with open(json_path, "r", encoding="utf-8") as f:
                results = json.load(f)

            for label, metrics in results.items():
                if not isinstance(metrics, dict):
                    continue
                if "SAE" in label:
                    for mk in METRIC_KEYS:
                        val = metrics.get(mk)
                        if val is not None and val != "nan":
                            data["SAE"][mk].append(float(val))
                    break

    return data


def compute_summary(data):
    """Compute mean ± std for each condition × metric.

    Returns: {condition: {metric: {"mean": ..., "std": ..., "n": ...}}}
    Also computes "overall" = mean of all metric means.
    """
    summary = {}
    for cond, metrics in data.items():
        summary[cond] = {}
        metric_means = []
        for mk in METRIC_KEYS:
            vals = metrics.get(mk, [])
            if vals:
                m = float(np.mean(vals))
                s = float(np.std(vals))
                n = len(vals)
                summary[cond][mk] = {"mean": m, "std": s, "n": n}
                metric_means.append(m)
            else:
                summary[cond][mk] = {"mean": np.nan, "std": 0, "n": 0}

        # Overall score = unweighted mean of all metrics
        if metric_means:
            overall = float(np.mean(metric_means))
        else:
            overall = np.nan
        summary[cond]["overall"] = {"mean": overall, "std": 0, "n": 0}

    return summary


# ==============================================================================
# Dot Heatmap (Nature Methods style)
# ==============================================================================
def plot_dot_heatmap(summary, output_path, dpi=200):
    """Create a scIB-style dot heatmap.

    Rows: conditions (CNN layers + SAE)
    Columns: metrics + overall
    Dot size ∝ value, dot color = intensity gradient
    Text annotation inside each dot
    """
    # Determine rows and columns
    rows = [r for r in ROW_ORDER if r in summary]
    cols = METRIC_KEYS + ["overall"]

    if not rows:
        logger.error("No data found to plot!")
        return

    n_rows = len(rows)
    n_cols = len(cols)

    # Build value matrix (rows × cols)
    val_mat = np.full((n_rows, n_cols), np.nan)
    std_mat = np.full((n_rows, n_cols), 0.0)
    n_mat = np.zeros((n_rows, n_cols), dtype=int)

    for i, row_key in enumerate(rows):
        for j, col_key in enumerate(cols):
            info = summary[row_key].get(col_key, {})
            val_mat[i, j] = info.get("mean", np.nan)
            std_mat[i, j] = info.get("std", 0)
            n_mat[i, j] = info.get("n", 0)

    # ── Figure ──
    cell_w = 1.6
    cell_h = 1.3
    left_margin = 3.0
    right_margin = 1.0
    top_margin = 1.8
    bottom_margin = 0.5

    fig_w = left_margin + n_cols * cell_w + right_margin
    fig_h = top_margin + n_rows * cell_h + bottom_margin

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # ── Colormap for dots (per column, normalized 0-1 within column) ──
    # Nature Methods style: each column has its own color scale
    cmap = plt.get_cmap("YlOrRd")

    # Global min/max for size scaling
    valid_vals = val_mat[~np.isnan(val_mat)]
    global_min = 0.0  # metrics are 0-1
    global_max = 1.0

    # Max dot size
    max_dot_radius = min(cell_w, cell_h) * 0.38

    for i in range(n_rows):
        for j in range(n_cols):
            val = val_mat[i, j]
            std = std_mat[i, j]

            if np.isnan(val):
                ax.text(j, i, "—", ha="center", va="center", fontsize=10, color="gray")
                continue

            # Dot size: proportional to absolute value (0→1)
            size_frac = np.clip(val, 0, 1)
            radius = max_dot_radius * (0.3 + 0.7 * size_frac)  # min 30% of max

            # Color: value-based intensity
            color_norm = np.clip(val, 0, 1)

            # Use row-specific color for overall column, cmap for metrics
            if j == n_cols - 1:  # overall column
                base_color = ROW_COLORS.get(rows[i], "#888888")
                # Adjust alpha based on value
                rgba = list(mcolors.to_rgba(base_color))
                rgba[3] = 0.4 + 0.6 * color_norm
                dot_color = rgba
                edge_color = base_color
            else:
                dot_color = cmap(0.15 + 0.75 * color_norm)
                edge_color = cmap(min(0.15 + 0.75 * color_norm + 0.15, 1.0))

            # Draw dot
            circle = plt.Circle(
                (j, i),
                radius,
                facecolor=dot_color,
                edgecolor=edge_color,
                linewidth=1.5,
                zorder=3,
            )
            ax.add_patch(circle)

            # Text inside dot
            luminance = (
                0.299 * mcolors.to_rgba(dot_color)[0]
                + 0.587 * mcolors.to_rgba(dot_color)[1]
                + 0.114 * mcolors.to_rgba(dot_color)[2]
            )
            text_color = "white" if luminance < 0.55 else "#222222"

            # Main value
            ax.text(
                j,
                i - 0.08,
                f"{val:.3f}",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color=text_color,
                zorder=4,
            )

            # ±std (smaller, below)
            if std > 0 and n_mat[i, j] > 1:
                std_color = "#dddddd" if luminance < 0.55 else "#777777"
                ax.text(
                    j,
                    i + 0.22,
                    f"±{std:.3f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color=std_color,
                    zorder=4,
                )

    # ── Axis setup ──
    ax.set_xlim(-0.7, n_cols - 0.3)
    ax.set_ylim(n_rows - 0.3, -0.7)  # inverted

    # Column labels (top)
    col_labels = [METRIC_DISPLAY.get(c, c) for c in cols]
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=11, fontweight="bold", ha="center")
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    # Row labels (left)
    row_labels = [ROW_DISPLAY.get(r, r) for r in rows]
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=11, fontweight="bold")

    # Separator line before "Overall" column
    ax.axvline(
        x=n_cols - 1.5, color="#888888", linewidth=1.0, linestyle="--", alpha=0.5
    )

    # Separator line before SAE row
    sae_idx = None
    for i, r in enumerate(rows):
        if r == "SAE":
            sae_idx = i
            break
    if sae_idx is not None and sae_idx > 0:
        ax.axhline(
            y=sae_idx - 0.5, color="#888888", linewidth=1.5, linestyle="-", alpha=0.6
        )

    # Light grid
    for i in range(n_rows):
        ax.axhline(y=i, color="#f0f0f0", linewidth=0.5, zorder=0)
    for j in range(n_cols):
        ax.axvline(x=j, color="#f0f0f0", linewidth=0.5, zorder=0)

    # Seed count annotation (right margin)
    for i, row_key in enumerate(rows):
        n = n_mat[i, 0]
        if n > 0:
            ax.text(
                n_cols - 0.15,
                i,
                f"n={n}",
                ha="left",
                va="center",
                fontsize=8,
                color="#999999",
                style="italic",
            )

    ax.set_frame_on(False)
    ax.tick_params(length=0)

    # Title
    fig.suptitle(
        "Representation Quality — scIB Metrics", fontsize=14, fontweight="bold", y=0.97
    )

    fig.tight_layout(rect=[0, 0, 1, 0.93])

    # Save
    svg_path = output_path.replace(".png", ".svg")
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    logger.info(f"  Saved: {output_path}")
    logger.info(f"  Saved: {svg_path}")

    if _IN_COLAB:
        plt.show()
    plt.close(fig)


# ==============================================================================
# Print summary table (for Colab display)
# ==============================================================================
def print_summary_table(summary):
    """Print aggregated results as formatted table."""
    cols = METRIC_KEYS + ["overall"]
    rows = [r for r in ROW_ORDER if r in summary]

    # Header
    header = f"  {'Condition':25s}"
    for c in cols:
        header += f"  {METRIC_DISPLAY.get(c, c):>10s}"
    header += f"  {'n':>4s}"

    logger.info(f"\n{'='*90}")
    logger.info(f"  scIB Aggregated Results (mean ± std)")
    logger.info(f"{'='*90}")
    logger.info(header)
    logger.info(f"  {'─'*85}")

    for row_key in rows:
        line = f"  {ROW_DISPLAY.get(row_key, row_key):25s}"
        for c in cols:
            info = summary[row_key].get(c, {})
            m = info.get("mean", np.nan)
            s = info.get("std", 0)
            n = info.get("n", 0)
            if np.isnan(m):
                line += f"  {'—':>10s}"
            elif s > 0 and n > 1:
                line += f"  {m:.3f}±{s:.3f}"
            else:
                line += f"  {m:>10.3f}"
        n = summary[row_key].get(METRIC_KEYS[0], {}).get("n", 0)
        line += f"  {n:>4d}"
        logger.info(line)

    logger.info(f"{'='*90}")


# ==============================================================================
# Main
# ==============================================================================
def get_args():
    p = argparse.ArgumentParser(description="scIB Dot Heatmap — Nature Methods style")
    p.add_argument(
        "--base_dir", type=str, required=True, help="Root scib_eval directory"
    )
    p.add_argument("--output_dir", type=str, default="")
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args()


def main():
    args = get_args()
    out_dir = args.output_dir or os.path.join(args.base_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"  scIB Dot Heatmap")
    logger.info(f"  Scanning: {args.base_dir}")
    logger.info(f"{'='*60}")

    # Scan and aggregate
    data = scan_scib_results(args.base_dir)

    if not data:
        logger.error("No scIB results found!")
        return

    summary = compute_summary(data)

    # Print table
    print_summary_table(summary)

    # Save aggregated CSV
    csv_path = os.path.join(out_dir, "scib_aggregated.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        cols_all = METRIC_KEYS + ["overall"]
        header = (
            ["condition"]
            + [f"{c}_mean" for c in cols_all]
            + [f"{c}_std" for c in cols_all]
            + ["n"]
        )
        f.write(",".join(header) + "\n")
        for row_key in ROW_ORDER:
            if row_key not in summary:
                continue
            parts = [ROW_DISPLAY.get(row_key, row_key)]
            for c in cols_all:
                parts.append(f"{summary[row_key][c]['mean']:.4f}")
            for c in cols_all:
                parts.append(f"{summary[row_key][c]['std']:.4f}")
            parts.append(str(summary[row_key][METRIC_KEYS[0]].get("n", 0)))
            f.write(",".join(parts) + "\n")
    logger.info(f"\n  Saved CSV: {csv_path}")

    # Dot heatmap
    output_path = os.path.join(out_dir, "scib_dot_heatmap.png")
    plot_dot_heatmap(summary, output_path, args.dpi)

    logger.info(f"\n{'='*60}")
    logger.info(f"  Done! Output: {out_dir}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
