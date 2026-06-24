#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# plot_pairwise_gam.py   VERSION 2.0-perclass
# ─────────────────────────────────────────────────────────────────────────────
# QUICK VERSION CHECK: python plot_pairwise_gam.py --check_version
# Expected output:  plot_pairwise_gam  v2.0-perclass
#
# HOW TO VERIFY IN COLAB:
#   !python /content/model_test/plot_pairwise_gam.py --check_version
#   Should print: plot_pairwise_gam  v2.0-perclass
#
# If you see "v1.0" or "command not found", the old file is still in Colab.
# Re-upload model_test/plot_pairwise_gam.py from Google Drive / local machine.
# ─────────────────────────────────────────────────────────────────────────────
__version__ = "2.1-rowwise"

"""
plot_pairwise_gam.py   v2.0-perclass
-------------------------------------
Pairwise (cosine_dist, DPT_dist) analysis from geometry_eval --save_pairwise output.

Changes in v2.0
  - Per-class scatter + GAM panels (rows=model, cols=class)
  - Independent per-panel axis limits (CNN cosine range is no longer crushed)
  - Seed selection by directory name: --cnn_seed_idx 445 finds cnn_seed_445/
  - Robust GAM fallback: deduplicated UnivariateSpline with proper smoothing
  - MatplotlibDeprecationWarning fixed (get_cmap)

Usage
-----
  # Full pipeline
  !pip install pygam -q
  !python model_test/plot_pairwise_gam.py \\
      --results_dir /content/results/apop_std \\
      --cnn_seed_idx 445 \\
      --sae_seed_idx 856 \\
      --intra_only \\
      --n_splines 8 \\
      --alpha 0.2 \\
      --output_dir ./gam_plots

  # Quick version check
  !python model_test/plot_pairwise_gam.py --check_version
"""


# !pip install pygam -q
# !python /content/model_test/plot_pairwise_gam.py \
#     --results_dir /content/drive/MyDrive/Final_paper/lambda_labs_moco_only/geometry_eval/apop_std \
#     --cnn_seed_idx 445 \
#     --sae_seed_idx 856 \
#     --n_splines 8 \
#     --intra_only \
#     --alpha 0.2 \
#     --output_dir /content/drive/MyDrive/Final_paper/lambda_labs_moco_only/geometry_eval/apop_std/gam_plots \
#     --window_size 0.16 \
#     --window_step 0.04 \
#     --window_min_pairs 800


import argparse
import csv
import glob
import json
import os
import sys
import warnings

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kendalltau, pearsonr, spearmanr

try:
    from pygam import LinearGAM
    from pygam import s as gam_s

    _HAS_PYGAM = True
except ImportError:
    from scipy.interpolate import UnivariateSpline

    _HAS_PYGAM = False

sns.set_theme(style="whitegrid", font_scale=1.05)

# "Sliding Window Regression" 으로 에러바 표시한다. 지금 2RMSD bin나눠서 slidding하면서 구해서 직선 근처에 표시해준다.


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
def get_args():
    p = argparse.ArgumentParser(
        description="GAM + sliding-window pairwise analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Input: multi-seed auto-discovery ──
    p.add_argument(
        "--results_dir",
        type=str,
        default="",
        help="Root dir containing cnn_seed_*/ and sae_seed_*/ subdirs. "
        "Automatically finds all pairwise_*_combined.npz files.",
    )

    # ── Input: single-file override ──
    p.add_argument(
        "--cnn_npz",
        type=str,
        default="",
        help="Direct path to CNN combined NPZ (single-seed mode).",
    )
    p.add_argument(
        "--sae_npz",
        type=str,
        default="",
        help="Direct path to SAE combined NPZ (single-seed mode).",
    )

    # ── Seed selection for scatter/GAM display ──
    p.add_argument(
        "--cnn_seed_idx",
        type=int,
        default=0,
        help="Seed selector for CNN: if the value matches a seed dir name "
        "(e.g. 445 matches cnn_seed_445/), that dir is used. "
        "Otherwise interpreted as a 0-based list index.",
    )
    p.add_argument(
        "--sae_seed_idx",
        type=int,
        default=0,
        help="Same as --cnn_seed_idx but for SAE.",
    )

    # ── Filtering ──
    p.add_argument(
        "--intra_only", action="store_true", help="Use only intra-class pairs."
    )
    p.add_argument(
        "--classes", nargs="+", default=None, help="Filter to specific class names."
    )

    # ── Sampling ──
    p.add_argument(
        "--n_sample",
        type=int,
        default=80_000,
        help="Points to show in scatter (stats still use ALL data).",
    )
    p.add_argument(
        "--n_corr_subsample",
        type=int,
        default=3_000_000,
        help="Max pairs for per-seed/per-class correlation (Kendall is slow).",
    )
    p.add_argument("--seed", type=int, default=42)

    # ── GAM ──
    p.add_argument(
        "--n_splines", type=int, default=20, help="Number of splines for GAM."
    )
    p.add_argument(
        "--gam_lam", type=float, default=0.6, help="GAM regularisation lambda (pygam)."
    )
    p.add_argument(
        "--poly_degree", type=int, default=3, help="Spline degree (scipy fallback)."
    )
    p.add_argument(
        "--gam_clip_pct",
        type=float,
        default=1.0,
        help="Clip lower+upper X%% of cosine dist before GAM fitting. "
        "0 = no clip. Default 1.0 removes extreme outliers.",
    )
    p.add_argument(
        "--gam_ci_width",
        type=float,
        default=0.95,
        help="Confidence interval width for GAM band (0 = no CI).",
    )
    p.add_argument(
        "--n_anchor_sub",
        type=int,
        default=0,
        help="Max anchors for row-wise correlation "
        "(0 = use all N points — accurate but slower).",
    )

    # ── Sliding window ──
    p.add_argument("--window_size", type=float, default=0.08)
    p.add_argument("--window_step", type=float, default=0.02)
    p.add_argument("--window_min_pairs", type=int, default=200)

    # ── Aesthetics ──
    p.add_argument("--color_cnn", type=str, default="#3A7EBF")
    p.add_argument("--color_sae", type=str, default="#E8833A")
    p.add_argument("--alpha", type=float, default=0.12)
    p.add_argument("--point_size", type=float, default=1.0)
    p.add_argument(
        "--inlier_alpha",
        type=float,
        default=0.05,
        help="Alpha transparency for inlier scatter points",
    )
    p.add_argument(
        "--outlier_alpha",
        type=float,
        default=0.8,
        help="Alpha transparency for outlier scatter points",
    )
    p.add_argument(
        "--rmsd_alpha",
        type=float,
        default=0.15,
        help="Alpha transparency for 2*RMSD band",
    )
    p.add_argument(
        "--x_clip_pct",
        type=float,
        default=0.5,
        help="Percentage of top X-axis (cosine distance) values to clip from the plot.",
    )
    p.add_argument(
        "--fit_clip_pct",
        type=float,
        default=1.0,
        help="Percentage of top X-axis values to exclude from the linear fit calculation.",
    )

    # ── Output ──
    p.add_argument("--output_dir", type=str, default="./gam_plots")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--no_svg", action="store_true")
    p.add_argument(
        "--scatter_only",
        action="store_true",
        help="Skip full sweeps and only generate the scatter/regression plot for the selected seeds.",
    )
    p.add_argument(
        "--check_version",
        action="store_true",
        help="Print version and exit immediately.",
    )

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Seed discovery
# ══════════════════════════════════════════════════════════════════════════════
def discover_seed_npzs(results_dir):
    """Find all pairwise_*_combined.npz files under cnn_seed_*/ and sae_seed_*/.

    Returns:
        {"CNN": [path0, path1, ...], "SAE": [path0, path1, ...]}
    """
    out = {}
    for model, pattern in [("CNN", "cnn_seed_*"), ("SAE", "sae_seed_*")]:
        seed_dirs = sorted(glob.glob(os.path.join(results_dir, pattern)))
        paths = []
        for sd in seed_dirs:
            # Try combined first, then any pairwise NPZ
            cand = glob.glob(os.path.join(sd, f"pairwise_{model}_combined.npz"))
            if not cand:
                cand = glob.glob(os.path.join(sd, "pairwise_*combined.npz"))
            if cand:
                paths.append(cand[0])
        if paths:
            out[model] = paths
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════
def load_npz(path, args):
    """Load pairwise NPZ and return (cos, dpt, class_i, class_j)."""
    d = np.load(path, allow_pickle=True)
    cos = d["cosine_dist"].astype(np.float32)
    dpt = d["dpt_dist"].astype(np.float32)

    ci = np.asarray(
        d.get("class_i", d.get("i_class", np.full(len(cos), "?"))), dtype=str
    )
    cj = np.asarray(d.get("class_j", ci), dtype=str)
    pt = np.asarray(d.get("pair_type", np.where(ci == cj, "intra", "inter")), dtype=str)

    # Finite + filter
    mask = np.isfinite(cos) & np.isfinite(dpt)
    if args.intra_only:
        mask &= pt == "intra"
    if args.classes:
        cls_mask = np.zeros(len(cos), dtype=bool)
        for c in args.classes:
            cls_mask |= ci == c
        mask &= cls_mask

    return cos[mask], dpt[mask], ci[mask], cj[mask]


# ══════════════════════════════════════════════════════════════════════════════
# Per-class × per-seed correlation
# ══════════════════════════════════════════════════════════════════════════════
def compute_corr(c_arr, d_arr, n_sub, seed):
    """Compute Spearman/Pearson/Kendall on a pair of arrays."""
    rng = np.random.RandomState(seed)
    n = len(c_arr)
    if n < 10:
        return dict(spearman=np.nan, pearson=np.nan, kendall=np.nan, n=n)
    if n > n_sub:
        idx = rng.choice(n, n_sub, replace=False)
        c, d = c_arr[idx], d_arr[idx]
    else:
        c, d = c_arr, d_arr
    rho = float(spearmanr(c, d).statistic)
    pr = float(pearsonr(c, d).statistic)
    tau = float(kendalltau(c, d).statistic)
    return dict(spearman=rho, pearson=pr, kendall=tau, n=int(n))


def sweep_all_seeds(seed_paths, args):
    """Compute global (all points) per-seed correlations for one model."""
    rows = []
    for seed_idx, path in enumerate(seed_paths):
        print(f"    Seed {seed_idx}: {os.path.basename(os.path.dirname(path))}")
        try:
            cos, dpt, ci, cj = load_npz(path, args)
        except Exception as e:
            print(f"      ERROR: {e}")
            continue

        # Only compute across all classes combined
        stats_all = compute_corr(cos, dpt, args.n_corr_subsample, args.seed)
        row_all = {"seed_idx": seed_idx, "class": "_ALL_"}
        row_all.update(stats_all)
        rows.append(row_all)
        print(
            f"      [_ALL_] ρ={stats_all['spearman']:.4f}  "
            f"r={stats_all['pearson']:.4f}  τ={stats_all['kendall']:.4f}  "
            f"n={stats_all['n']:,}"
        )

    return rows


def summarise_rows(rows, label=""):
    """Aggregate rows: per-class mean±SEM across seeds, grand mean.

    Returns:
        cls_summary: dict {class: {metric: {mean, sem, n_seeds}}}
    """
    from collections import defaultdict

    cls_vals = defaultdict(lambda: defaultdict(list))
    for row in rows:
        cls = row["class"]
        for m in ["spearman", "pearson", "kendall"]:
            v = row.get(m, np.nan)
            if np.isfinite(v):
                cls_vals[cls][m].append(v)

    summary = {}
    for cls, mdata in cls_vals.items():
        summary[cls] = {}
        for m, vals in mdata.items():
            arr = np.array(vals)
            summary[cls][m] = {
                "mean": float(np.mean(arr)),
                "sem": float(np.std(arr) / np.sqrt(len(arr))),
                "std": float(np.std(arr)),
                "n_seeds": len(arr),
            }

    # Print
    print(f"\n  [{label}] Per-class summary (mean ± SEM across seeds):")
    all_classes = sorted(k for k in summary if k != "_ALL_")
    header = f"  {'Class':<20s}  {'Spearman ρ':>14s}  {'Pearson r':>14s}  {'Kendall τ':>14s}  seeds"
    print(header)
    print("  " + "─" * 72)
    for cls in all_classes + ["_ALL_"]:
        if cls not in summary:
            continue
        d = summary[cls]

        def fmt(m):
            if m in d:
                return f"{d[m]['mean']:+.4f}±{d[m]['sem']:.4f}"
            return "     —     "

        n = d.get("spearman", {}).get("n_seeds", "?")
        print(
            f"  {cls:<20s}  {fmt('spearman'):>14s}  "
            f"{fmt('pearson'):>14s}  {fmt('kendall'):>14s}  {n}"
        )

    return summary


def print_cnn_sae_comparison(cnn_summ, sae_summ):
    """Print side-by-side CNN vs SAE comparison table."""
    print(f"\n  {'='*80}")
    print(f"  CNN vs SAE — scatter-based correlations (seed-averaged, class-averaged)")
    print(f"  {'='*80}")
    all_cls = sorted(set(cnn_summ) | set(sae_summ))
    hdr = f"  {'Class':<20s}  {'CNN ρ':>10s}  {'SAE ρ':>10s}  {'Δ(CNN-SAE)':>12s}  dir"
    print(hdr)
    print("  " + "─" * 65)
    for cls in all_cls:
        c_rho = cnn_summ.get(cls, {}).get("spearman", {}).get("mean", np.nan)
        s_rho = sae_summ.get(cls, {}).get("spearman", {}).get("mean", np.nan)
        delta = c_rho - s_rho
        direction = "CNN >" if delta > 0 else "SAE >"
        print(
            f"  {cls:<20s}  {c_rho:>10.4f}  {s_rho:>10.4f}  "
            f"{delta:>+12.4f}  {direction}"
        )
    print(f"  {'='*80}")


# ══════════════════════════════════════════════════════════════════════════════
# GAM fitting
# ══════════════════════════════════════════════════════════════════════════════
def _resolve_seed_path(paths, spec, model_label=""):
    """Resolve which NPZ to use for scatter/GAM display.

    spec (int) is tried in two ways:
      1. Search for a seed directory whose name contains str(spec).
         e.g. spec=445 matches  cnn_seed_445/pairwise_CNN_combined.npz
      2. If not found, treat spec as a 0-based positional index.

    Returns (path, seed_label_str).
    """
    spec_str = str(spec)
    for p in paths:
        dirname = os.path.basename(os.path.dirname(p))
        if spec_str in dirname:
            print(f"  [{model_label}] Matched seed '{dirname}' for spec={spec}")
            return p, dirname
    # Positional fallback
    idx = spec if spec < len(paths) else 0
    if spec >= len(paths):
        print(
            f"  [{model_label}] WARNING: spec={spec} not found in dir names "
            f"and >= len(paths)={len(paths)}. Using index 0."
        )
    dirname = os.path.basename(os.path.dirname(paths[idx]))
    print(f"  [{model_label}] Using positional index {idx}: '{dirname}'")
    return paths[idx], dirname


def fit_gam(
    cos, dpt, n_splines=20, lam=0.6, degree=3, seed=42, clip_pct=1.0, ci_width=0.95
):
    """Fit smooth curve to (cos, dpt) pairs.

    Returns (x_pred, y_line, y_lo, y_hi, r2, adj_r2)
    where y_lo/y_hi are confidence band boundaries.
    """
    rng = np.random.RandomState(seed)
    n_fit = min(200_000, len(cos))
    idx = rng.choice(len(cos), n_fit, replace=False)
    xf = cos[idx].astype(np.float64)
    yf = dpt[idx].astype(np.float64)

    # ── Outlier clipping ──
    if clip_pct > 0:
        lo_x = np.percentile(xf, clip_pct)
        hi_x = np.percentile(xf, 100.0 - clip_pct)
        lo_y = np.percentile(yf, clip_pct)
        hi_y = np.percentile(yf, 100.0 - clip_pct)
        keep = (xf >= lo_x) & (xf <= hi_x) & (yf >= lo_y) & (yf <= hi_y)
        xf, yf = xf[keep], yf[keep]

    if len(xf) < 10:
        nan5 = np.full(5, np.nan)
        return nan5, nan5, nan5, nan5, np.nan, np.nan

    x_pred = np.linspace(float(xf.min()), float(xf.max()), 500)
    y_lo = np.full(500, np.nan)
    y_hi = np.full(500, np.nan)

    if _HAS_PYGAM:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gam = LinearGAM(gam_s(0, n_splines=n_splines), lam=lam)
            gam.fit(xf[:, None], yf)
        yfit = gam.predict(xf[:, None])
        yline = gam.predict(x_pred[:, None])
        if ci_width > 0:
            try:
                ci_ = gam.confidence_intervals(x_pred[:, None], width=ci_width)
                y_lo, y_hi = ci_[:, 0], ci_[:, 1]
            except Exception:
                pass
    else:
        from scipy.interpolate import UnivariateSpline

        sort_idx = np.argsort(xf)
        xs, ys = xf[sort_idx], yf[sort_idx]
        x_uniq, inv = np.unique(xs, return_inverse=True)
        y_uniq = np.bincount(inv, weights=ys) / np.bincount(inv)
        if len(x_uniq) < degree + 1:
            coeffs = np.polyfit(xf, yf, 1)
            yfit = np.polyval(coeffs, xf)
            yline = np.polyval(coeffs, x_pred)
        else:
            s_val = max(len(x_uniq) * y_uniq.var() * 0.05, 1e-6)
            try:
                spl = UnivariateSpline(x_uniq, y_uniq, k=min(degree, 5), s=s_val)
                yfit = spl(xf)
                yline = spl(x_pred)
            except Exception:
                coeffs = np.polyfit(xf, yf, min(8, n_splines // 2))
                yfit = np.polyval(coeffs, xf)
                yline = np.polyval(coeffs, x_pred)

        # Bootstrap CI for scipy path
        if ci_width > 0:
            n_boot = 80
            n_sub = min(len(x_uniq), 2000)
            boot_ps = []
            for _ in range(n_boot):
                bi = rng.choice(len(xf), len(xf), replace=True)
                xb, yb = xf[bi], yf[bi]
                xbs = np.sort(np.unique(xb))
                ubs, inv_b = np.unique(np.sort(xb), return_inverse=True)
                ybs = np.bincount(inv_b, weights=yb[np.argsort(xb)]) / np.bincount(
                    inv_b
                )
                try:
                    sb = UnivariateSpline(ubs, ybs, k=min(degree, 5), s=s_val)
                    boot_ps.append(sb(x_pred))
                except Exception:
                    pass
            if boot_ps:
                ba = np.array(boot_ps)
                alo = (1.0 - ci_width) / 2 * 100
                y_lo = np.percentile(ba, alo, axis=0)
                y_hi = np.percentile(ba, 100 - alo, axis=0)

    ss_res = float(np.sum((yf - yfit) ** 2))
    ss_tot = float(np.sum((yf - yf.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    n, p_ = len(yf), n_splines
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / max(n - p_ - 1, 1)

    return x_pred, yline, y_lo, y_hi, float(r2), float(adj_r2)


# ============================================================================
# Row-wise (per-anchor) correlation
# ============================================================================
def compute_rowwise_correlation(npz_path, args):
    """For each anchor i, compute Spearman(cos(i,*), DPT(i,*)) using ALL points.

    Reconstructs the full upper-triangle row slices for the whole dataset.
    """
    rng = np.random.RandomState(args.seed)
    d = np.load(npz_path, allow_pickle=True)
    cos = d["cosine_dist"].astype(np.float64)
    dpt = d["dpt_dist"].astype(np.float64)

    results = {}
    cls = "_ALL_"

    mask = np.isfinite(cos) & np.isfinite(dpt)
    x = cos[mask]
    y = dpt[mask]
    M = len(x)
    if M < 3:
        return {}

    # Infer N from M = N*(N-1)/2
    N = int(round((1.0 + (1.0 + 8.0 * M) ** 0.5) / 2.0))
    M_check = N * (N - 1) // 2

    if M != M_check:
        n_s = min(M, 300_000)
        idx_s = rng.choice(M, n_s, replace=False)
        rho_g = float(spearmanr(x[idx_s], y[idx_s]).statistic)
        pr_g = float(pearsonr(x[idx_s], y[idx_s]).statistic)
        results[cls] = dict(
            mean_sp=rho_g,
            std_sp=np.nan,
            sem_sp=np.nan,
            mean_pr=pr_g,
            std_pr=np.nan,
            sem_pr=np.nan,
            n_anchors=1,
            method="global_fallback",
        )
        print(f"    [{cls}] rowwise fallback (M={M} != expected {M_check})")
        return results

    ri, ci_idx = np.triu_indices(N, k=1)
    ci_sort_order = np.argsort(ci_idx, kind="stable")
    ci_sorted = ci_idx[ci_sort_order]

    row_starts = np.searchsorted(ri, np.arange(N), side="left")
    row_ends = np.searchsorted(ri, np.arange(N), side="right")
    col_starts = np.searchsorted(ci_sorted, np.arange(N), side="left")
    col_ends = np.searchsorted(ci_sorted, np.arange(N), side="right")

    anchors = np.arange(N)
    if args.n_anchor_sub > 0 and N > args.n_anchor_sub:
        anchors = rng.choice(N, args.n_anchor_sub, replace=False)
        print(f"    [{cls}] row-wise using {args.n_anchor_sub}/{N} anchors")
    else:
        print(f"    [{cls}] row-wise using all {N} anchors")

    sp_list, pr_list = [], []
    for k in anchors:
        r_sl = slice(row_starts[k], row_ends[k])
        c_sl_sorted = slice(col_starts[k], col_ends[k])
        c_orig_idx = ci_sort_order[c_sl_sorted]

        x_k = np.concatenate([x[r_sl], x[c_orig_idx]])
        y_k = np.concatenate([y[r_sl], y[c_orig_idx]])
        if len(x_k) < 5:
            continue
        sp_list.append(float(spearmanr(x_k, y_k).statistic))
        pr_list.append(float(pearsonr(x_k, y_k).statistic))

    sp_arr = np.array(sp_list)
    pr_arr = np.array(pr_list)
    n_a = len(sp_arr)
    results[cls] = dict(
        mean_sp=float(np.nanmean(sp_arr)),
        std_sp=float(np.nanstd(sp_arr)),
        sem_sp=float(np.nanstd(sp_arr) / max(n_a**0.5, 1)),
        mean_pr=float(np.nanmean(pr_arr)),
        std_pr=float(np.nanstd(pr_arr)),
        sem_pr=float(np.nanstd(pr_arr) / max(n_a**0.5, 1)),
        n_anchors=n_a,
        method="rowwise",
    )
    print(
        f"    [{cls}] row-wise Spearman: "
        f"{results[cls]['mean_sp']:.4f} ± {results[cls]['std_sp']:.4f}  "
        f"(n={n_a} anchors)"
    )
    return results


def sweep_rowwise_seeds(seed_paths, args):
    """Run compute_rowwise_correlation for every seed.

    Returns {model: {seed_label: {cls: stats_dict}}}
    """
    out = {}
    for model, paths in sorted(seed_paths.items()):
        out[model] = {}
        for seed_idx, path in enumerate(paths):
            seed_label = os.path.basename(os.path.dirname(path))
            print(f"    [{model} seed {seed_idx}: {seed_label}]")
            try:
                res = compute_rowwise_correlation(path, args)
                out[model][seed_label] = res
            except Exception as e:
                print(f"      ERROR: {e}")
    return out


def summarise_rowwise(rw_data):
    """Average row-wise results across seeds, per model and class.

    rw_data: {model: {seed_label: {cls: stats}}}
    Returns:  {model: {cls: {"rw_sp": mean, "rw_sp_sem": SEM, ...}}}
    """
    from collections import defaultdict

    out = {}
    for model, seed_dict in rw_data.items():
        cls_sp = defaultdict(list)
        cls_pr = defaultdict(list)
        for seed_label, cls_dict in seed_dict.items():
            for cls, st in cls_dict.items():
                if np.isfinite(st.get("mean_sp", np.nan)):
                    cls_sp[cls].append(st["mean_sp"])
                if np.isfinite(st.get("mean_pr", np.nan)):
                    cls_pr[cls].append(st["mean_pr"])
        out[model] = {}
        for cls in set(list(cls_sp) + list(cls_pr)):
            sp_arr = np.array(cls_sp.get(cls, [np.nan]))
            pr_arr = np.array(cls_pr.get(cls, [np.nan]))
            out[model][cls] = dict(
                rw_sp=float(np.nanmean(sp_arr)),
                rw_sp_sem=float(np.nanstd(sp_arr) / max(len(sp_arr) ** 0.5, 1)),
                rw_pr=float(np.nanmean(pr_arr)),
                rw_pr_sem=float(np.nanstd(pr_arr) / max(len(pr_arr) ** 0.5, 1)),
                n_seeds=len(sp_arr),
            )
        # Print summary
        print(f"\n  [Row-wise {model}] mean across seeds:")
        print(f"  {'Class':<20s}  {'RW Spearman':>14s}  {'RW Pearson':>12s}  seeds")
        print("  " + "-" * 55)
        for cls, v in sorted(out[model].items()):
            print(
                f"  {cls:<20s}  "
                f"{v['rw_sp']:+.4f}±{v['rw_sp_sem']:.4f}  "
                f"{v['rw_pr']:+.4f}±{v['rw_pr_sem']:.4f}  "
                f"{v['n_seeds']}"
            )
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1: Scatter + GAM  — per-class, per-model
# ══════════════════════════════════════════════════════════════════════════════
def plot_scatter_gam(display_data, corr_rows_by_model, args, out_dir):
    """
    Plots a 1x2 GAM scatter plot (CNN vs SAE) using the entire dataset.
    """
    models = ["CNN", "SAE"]
    models = [m for m in models if m in display_data]
    if not models:
        return

    n_cols = len(models)
    colors = {"CNN": args.color_cnn, "SAE": args.color_sae}
    rng = np.random.RandomState(args.seed)

    fig, axes = plt.subplots(1, n_cols, figsize=(5.5 * n_cols, 5.2), squeeze=False)

    global_y_max = 0.0
    for model in models:
        _, dpt, _, _, _ = display_data[model]
        if len(dpt) > 0:
            global_y_max = max(global_y_max, float(np.nanmax(dpt)))
    y_lim = global_y_max * 1.06

    for col_i, model in enumerate(models):
        cos, dpt, ci, cj, seed_label = display_data[model]
        color = colors.get(model, "#666")
        ax = axes[0, col_i]

        mask = np.ones(len(cos), dtype=bool)
        x_cls = cos[mask].astype(np.float32)
        y_cls = dpt[mask].astype(np.float32)

        print(
            f"    [Debug {model}] Raw Cosine Dist - Min: {x_cls.min():.4f}, Max: {x_cls.max():.4f}, Mean: {x_cls.mean():.4f}, 99.5th Pctl: {np.percentile(x_cls, 99.5):.4f}"
        )

        if args.x_clip_pct > 0 and len(x_cls) > 0:
            x_thresh = np.percentile(x_cls, 100.0 - args.x_clip_pct)
            clip_mask = x_cls <= x_thresh
            n_before = len(x_cls)
            x_cls = x_cls[clip_mask]
            y_cls = y_cls[clip_mask]
            print(
                f"    [Debug {model}] After clipping top {args.x_clip_pct}% (> {x_thresh:.4f}): Max becomes {x_cls.max():.4f} (removed {n_before - len(x_cls)} pairs)"
            )

        if len(x_cls) < 20:
            ax.set_visible(False)
            continue

        budget = args.n_sample
        n_disp = max(min(budget, len(x_cls)), min(80_000, len(x_cls)))
        idx_disp = rng.choice(len(x_cls), n_disp, replace=False)

        x_disp = x_cls[idx_disp]
        y_disp = y_cls[idx_disp]

        # 1. Linear Fit instead of GAM
        if args.fit_clip_pct > 0 and len(x_cls) > 0:
            fit_thresh = np.percentile(x_cls, 100.0 - args.fit_clip_pct)
            fit_mask = x_cls <= fit_thresh
            x_fit = x_cls[fit_mask]
            y_fit = y_cls[fit_mask]
        else:
            x_fit = x_cls
            y_fit = y_cls

        coeffs = np.polyfit(x_fit, y_fit, 1)

        y_pred_all = np.polyval(coeffs, x_cls)
        y_pred_fit = np.polyval(coeffs, x_fit)
        ss_res = float(np.sum((y_fit - y_pred_fit) ** 2))
        ss_tot = float(np.sum((y_fit - np.mean(y_fit)) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

        # 2. Sliding RMSD calculation
        eval_x = np.linspace(float(x_cls.min()), float(x_cls.max()), 200)
        window_size = (x_cls.max() - x_cls.min()) * 0.1
        rmsd_list = []
        residuals = y_cls - y_pred_all
        for ex in eval_x:
            w_mask = (x_cls >= ex - window_size / 2) & (x_cls <= ex + window_size / 2)
            if w_mask.sum() > 5:
                rmsd_list.append(np.sqrt(np.mean(residuals[w_mask] ** 2)))
            else:
                rmsd_list.append(np.nan)
        rmsd_array = np.array(rmsd_list)

        valid_mask = np.isfinite(rmsd_array)
        if valid_mask.sum() > 0:
            rmsd_array = np.interp(eval_x, eval_x[valid_mask], rmsd_array[valid_mask])
        else:
            rmsd_array = np.zeros_like(eval_x)

        y_line = np.polyval(coeffs, eval_x)

        # 3. Identify Outliers for display
        y_pred_disp = np.polyval(coeffs, x_disp)
        res_disp = y_disp - y_pred_disp
        rmsd_disp = np.interp(x_disp, eval_x, rmsd_array)
        outlier_mask = np.abs(res_disp) > 2 * rmsd_disp
        inlier_mask = ~outlier_mask

        # 4. Plot Scatter for Inliers, and Scatter for Outliers
        ax.scatter(
            x_disp[inlier_mask],
            y_disp[inlier_mask],
            c=color,
            alpha=args.inlier_alpha,
            s=args.point_size,
            linewidths=0,
            rasterized=True,
            label="Inliers (≤2σ)",
        )

        ax.scatter(
            x_disp[outlier_mask],
            y_disp[outlier_mask],
            c=color,
            alpha=args.outlier_alpha,
            s=args.point_size * 2.5,
            linewidths=0,
            rasterized=True,
            label="Outliers (>2σ)",
        )

        ax.plot(
            eval_x,
            y_line,
            color="black",
            linewidth=2.2,
            zorder=5,
            label=f"Linear Fit R²={r2:.3f}",
        )
        ax.fill_between(
            eval_x,
            y_line - 2 * rmsd_array,
            y_line + 2 * rmsd_array,
            alpha=args.rmsd_alpha,
            color="gray",
            zorder=4,
            label="±2 RMSD",
            linewidth=0,
        )

        xy_max = max(float(x_cls.max()), global_y_max) * 1.05
        ax.plot(
            [0, xy_max],
            [0, xy_max],
            color="#aaaaaa",
            linewidth=1.0,
            linestyle="--",
            alpha=0.65,
        )

        x_lim = float(x_cls.max()) * 1.06
        ax.set_xlim(-x_lim * 0.01, x_lim)
        ax.set_ylim(-y_lim * 0.01, y_lim)

        # 5. Compute Correlation for THIS specific seed
        st = compute_corr(x_cls, y_cls, args.n_corr_subsample, args.seed)
        pr_m = st["pearson"]
        rho_m = st["spearman"]
        tau_m = st["kendall"]

        rw_sp = np.nan
        if hasattr(args, "_rw_summ"):
            rw_sp = args._rw_summ.get(model, {}).get("_ALL_", {}).get("rw_sp", np.nan)

        ann = f"Seed: {seed_label}\n"
        ann += f"r={pr_m:.3f}  ρ={rho_m:.3f}  τ={tau_m:.3f}"
        if np.isfinite(rw_sp):
            ann += f"\nRow-wise ρ (avg)={rw_sp:.3f}"
        ann += f"\nLinear R²={r2:.3f}"
        ann += f"\nn_disp={n_disp:,}"

        ax.text(
            0.04,
            0.96,
            ann,
            transform=ax.transAxes,
            fontsize=8.0,
            va="top",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                alpha=0.85,
                edgecolor="#cccccc",
            ),
        )

        ax.set_xlabel("Cosine distance", fontsize=11)
        if col_i == 0:
            ax.set_ylabel("DPT geodesic", fontsize=11)
        ax.set_title(f"{model}  │  All Classes", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, loc="lower right")
        sns.despine(ax=ax)

    fig.suptitle(
        "Global Geometry: Cosine vs DPT (All Pairs)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    _save_fig(fig, out_dir, "fig_scatter_gam_global", args)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2: Sliding window (selected seeds)
# ══════════════════════════════════════════════════════════════════════════════
def sliding_window_corr(cos, dpt, window_size, step, min_pairs):
    starts = np.arange(cos.min(), cos.max() - window_size + step, step)
    mids, pear, spear = [], [], []
    for lo in starts:
        hi = lo + window_size
        mask = (cos >= lo) & (cos < hi)
        if mask.sum() < min_pairs:
            continue
        c_w, d_w = cos[mask], dpt[mask]
        try:
            pr = float(pearsonr(c_w, d_w).statistic)
            rho = float(spearmanr(c_w, d_w).statistic)
        except Exception:
            continue
        mids.append((lo + hi) / 2)
        pear.append(pr)
        spear.append(rho)
    return np.array(mids), np.array(pear), np.array(spear)


def plot_sliding_window(display_data, args, out_dir):
    colors = {"CNN": args.color_cnn, "SAE": args.color_sae}
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax_p, ax_s = axes

    for label, (cos, dpt, used_seed) in display_data.items():
        print(f"  [{label} seed {used_seed}] sliding window ...")
        mids, pear, spear = sliding_window_corr(
            cos, dpt, args.window_size, args.window_step, args.window_min_pairs
        )
        color = colors.get(label, "#666")
        ax_p.plot(
            mids,
            pear,
            color=color,
            linewidth=2,
            marker="o",
            markersize=3.5,
            alpha=0.9,
            label=f"{label} s{used_seed}",
        )
        ax_s.plot(
            mids,
            spear,
            color=color,
            linewidth=2,
            marker="s",
            markersize=3.5,
            alpha=0.9,
            label=f"{label} s{used_seed}",
        )
        np.savez(
            os.path.join(out_dir, f"sliding_{label}_seed{used_seed}.npz"),
            midpoints=mids,
            pearson=pear,
            spearman=spear,
        )

    for ax, ttl, ylbl in [
        (ax_p, "Pearson r", "Pearson r (local)"),
        (ax_s, "Spearman ρ", "Spearman ρ (local)"),
    ]:
        ax.axhline(0, color="#bbb", linewidth=1, linestyle="--")
        ax.set_ylabel(ylbl, fontsize=11)
        ax.set_title(f"Sliding-window {ttl}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.15)
        sns.despine(ax=ax)

    ax_s.set_xlabel("Cosine distance (window midpoint)", fontsize=11)
    fig.suptitle(
        f"Local correlation: cosine vs DPT "
        f"(window={args.window_size:.2f}, step={args.window_step:.2f})",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    _save_fig(fig, out_dir, "fig_sliding_window", args)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3: KDE cosine distribution — ALL seeds combined (supplementary)
# ══════════════════════════════════════════════════════════════════════════════
def plot_kde_cosine(seed_paths_by_model, args, out_dir):
    """KDE of cosine_dist CNN vs SAE (combining all seeds) globally."""
    rng = np.random.RandomState(args.seed)
    colors = {"CNN": args.color_cnn, "SAE": args.color_sae}

    cls_all = {}  # {model: all_cos}

    for model, paths in seed_paths_by_model.items():
        all_cos_m = []
        for path in paths:
            try:
                d = np.load(path, allow_pickle=True)
                cos = d["cosine_dist"].astype(np.float32)
                fin = np.isfinite(cos)
                cos = cos[fin]

                # Subsample per seed to keep memory manageable
                n_all = min(80_000, len(cos))
                all_cos_m.extend(
                    cos[rng.choice(len(cos), n_all, replace=False)].tolist()
                )
            except Exception as e:
                print(f"    KDE load error ({path}): {e}")

        cls_all[model] = np.array(all_cos_m, dtype=np.float32)

    # ── All-class combined KDE ──
    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    for model, cos_arr in cls_all.items():
        if len(cos_arr) == 0:
            continue
        sns.kdeplot(
            cos_arr,
            ax=ax2,
            fill=True,
            alpha=0.4,
            color=colors.get(model, "#999"),
            linewidth=2.5,
            label=model,
            bw_adjust=0.5,
            clip=(0, None),
        )
        med = float(np.median(cos_arr))
        ax2.axvline(
            med,
            color=colors.get(model, "#999"),
            linewidth=1.5,
            linestyle=":",
            alpha=0.7,
            label=f"{model} median={med:.3f}",
        )

    ax2.set_xlabel("Cosine distance (all pairs, all seeds)", fontsize=12)
    ax2.set_ylabel("Density", fontsize=12)
    ax2.set_title(
        "Cosine distance distribution — CNN vs SAE (Global)",
        fontsize=13,
        fontweight="bold",
    )
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.12)
    sns.despine(ax=ax2)
    fig2.tight_layout()
    _save_fig(fig2, out_dir, "fig_kde_all", args)


# ══════════════════════════════════════════════════════════════════════════════
# Save statistics CSV
# ══════════════════════════════════════════════════════════════════════════════
def save_stats_csv(rows_by_model, summaries, out_dir):
    # Raw per-seed per-class
    raw_path = os.path.join(out_dir, "corr_per_seed_class.csv")
    with open(raw_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "seed_idx",
                "class",
                "spearman",
                "pearson",
                "kendall",
                "n",
            ],
        )
        w.writeheader()
        for model, rows in rows_by_model.items():
            for row in rows:
                w.writerow({"model": model, **row})
    print(f"  Saved raw CSV: {raw_path}")

    # Summary
    summ_path = os.path.join(out_dir, "corr_summary.csv")
    with open(summ_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "class", "metric", "mean", "sem", "std", "n_seeds"])
        for model, summ in summaries.items():
            for cls, mdata in summ.items():
                for metric, vals in mdata.items():
                    w.writerow(
                        [
                            model,
                            cls,
                            metric,
                            f"{vals['mean']:.6f}",
                            f"{vals['sem']:.6f}",
                            f"{vals['std']:.6f}",
                            vals["n_seeds"],
                        ]
                    )
    print(f"  Saved summary CSV: {summ_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4: Seed-level cosine distribution stability  (supplementary)
# ══════════════════════════════════════════════════════════════════════════════
def compute_seed_cosine_stats(seed_paths_by_model, rng_seed=42):
    """For each seed of each model, compute cosine_dist summary stats.

    Returns
    -------
    rows : list of dicts
        {model, seed_idx, seed_label,
         median, mean, q25, q75, iqr, p95, p99, max}
    """
    rows = []
    rng = np.random.RandomState(rng_seed)

    for model, paths in sorted(seed_paths_by_model.items()):
        for seed_idx, path in enumerate(paths):
            seed_label = os.path.basename(os.path.dirname(path))
            try:
                d = np.load(path, allow_pickle=True)
                cos = d["cosine_dist"].astype(np.float32)
                pt = np.asarray(
                    d.get("pair_type", np.full(len(cos), "intra")), dtype=str
                )
                # Use intra-class pairs for cleaner comparison
                intra = cos[np.isfinite(cos) & (pt == "intra")]
                if len(intra) == 0:
                    intra = cos[np.isfinite(cos)]
                if len(intra) == 0:
                    continue
                # Subsample for speed
                n_s = min(500_000, len(intra))
                samp = intra[rng.choice(len(intra), n_s, replace=False)]

                q25, med, q75, p95, p99 = np.percentile(samp, [25, 50, 75, 95, 99])
                row = dict(
                    model=model,
                    seed_idx=seed_idx,
                    seed_label=seed_label,
                    median=float(med),
                    mean=float(samp.mean()),
                    q25=float(q25),
                    q75=float(q75),
                    iqr=float(q75 - q25),
                    p95=float(p95),
                    p99=float(p99),
                    max=float(samp.max()),
                    n=int(len(intra)),
                )
                rows.append(row)
                print(
                    f"    [{model} seed {seed_idx}] "
                    f"median={med:.4f}  IQR={q75-q25:.4f}  "
                    f"p95={p95:.4f}  max={samp.max():.4f}"
                )
            except Exception as e:
                print(f"    [{model} seed {seed_idx}] ERROR: {e}")
    return rows


# ==============================================================================
# Global Geometry Scatter + GAM (All Classes Combined)
# ==============================================================================
def plot_global_scatter_gam(seed_paths, args, out_dir):
    """Computes global Mantel & row-wise correlations across seeds and plots GAM scatter."""
    print(f"\n{'-'*60}")
    print(f"  Figure 1.5: Global Geometry Scatter + GAM (All Classes Combined)")
    print(f"{'-'*60}")

    rng = np.random.RandomState(args.seed)
    n_sample_plot = 80_000
    n_corr_subsample = args.n_corr_subsample

    stats = {"CNN": defaultdict(list), "SAE": defaultdict(list)}
    plot_data = {"CNN": {"cos": [], "dpt": []}, "SAE": {"cos": [], "dpt": []}}

    for model, paths in seed_paths.items():
        if not paths:
            continue

        for path in paths:
            seed_dir = os.path.dirname(path)

            # 1. Read Row-wise from rank_correlation.json
            rc_jsons = glob.glob(os.path.join(seed_dir, "rank_correlation.json"))
            if not rc_jsons:
                rc_jsons = glob.glob(
                    os.path.join(seed_dir, "k_*", "rank_correlation.json")
                )

            if rc_jsons:
                try:
                    with open(rc_jsons[0], "r") as f:
                        rc_data = json.load(f)
                    for src_key, src_val in rc_data.items():
                        if isinstance(src_val, dict) and "error" not in src_val:
                            if "spearman_mean" in src_val:
                                stats[model]["rw_spearman"].append(
                                    src_val["spearman_mean"]
                                )
                            if "kendall_mean" in src_val:
                                stats[model]["rw_kendall"].append(
                                    src_val["kendall_mean"]
                                )
                            if "pearson_mean" in src_val:
                                stats[model]["rw_pearson"].append(
                                    src_val["pearson_mean"]
                                )
                except Exception as e:
                    print(f"    Error reading {rc_jsons[0]}: {e}")

            # 2. Read Global Pairs from combined.npz
            npz_cands = glob.glob(
                os.path.join(seed_dir, f"pairwise_{model}_combined.npz")
            )
            if not npz_cands:
                npz_cands = glob.glob(os.path.join(seed_dir, "pairwise_*combined.npz"))

            if npz_cands:
                try:
                    d = np.load(npz_cands[0], allow_pickle=True)
                    cos = d["cosine_dist"].astype(np.float32)
                    dpt = d["dpt_dist"].astype(np.float32)

                    mask = np.isfinite(cos) & np.isfinite(dpt)
                    cos, dpt = cos[mask], dpt[mask]
                    N_total = len(cos)

                    if N_total > 0:
                        idx_corr = rng.choice(
                            N_total, min(N_total, n_corr_subsample), replace=False
                        )
                        cos_sub, dpt_sub = cos[idx_corr], dpt[idx_corr]

                        rho = float(spearmanr(cos_sub, dpt_sub).statistic)
                        tau = float(kendalltau(cos_sub, dpt_sub).statistic)
                        pr = float(pearsonr(cos_sub, dpt_sub).statistic)

                        stats[model]["gl_spearman"].append(rho)
                        stats[model]["gl_kendall"].append(tau)
                        stats[model]["gl_pearson"].append(pr)

                        idx_plot = rng.choice(
                            N_total,
                            min(N_total, n_sample_plot // len(paths)),
                            replace=False,
                        )
                        plot_data[model]["cos"].append(cos[idx_plot])
                        plot_data[model]["dpt"].append(dpt[idx_plot])
                except Exception as e:
                    print(f"    Error reading {npz_cands[0]}: {e}")

    print(f"\n  ── Full Dataset Correlation (Inter + Intra Classes) ──")
    print(f"  {'Metric':<20s}  {'CNN mean±std':>16s}  {'SAE mean±std':>16s}")
    print(f"  {'─'*58}")

    avg_stats = {}
    for metric_name, display_name in [
        ("gl_spearman", "Global Spearman ρ"),
        ("gl_kendall", "Global Kendall τ"),
        ("gl_pearson", "Global Pearson r"),
        ("rw_spearman", "Row-wise Spearman ρ"),
        ("rw_kendall", "Row-wise Kendall τ"),
        ("rw_pearson", "Row-wise Pearson r"),
    ]:
        c_vals = stats["CNN"].get(metric_name, [])
        s_vals = stats["SAE"].get(metric_name, [])

        c_str = f"{np.mean(c_vals):+.4f}±{np.std(c_vals):.4f}" if c_vals else "—"
        s_str = f"{np.mean(s_vals):+.4f}±{np.std(s_vals):.4f}" if s_vals else "—"
        print(f"  {display_name:<20s}  {c_str:>16s}  {s_str:>16s}")

        avg_stats[f"CNN_{metric_name}"] = np.mean(c_vals) if c_vals else np.nan
        avg_stats[f"SAE_{metric_name}"] = np.mean(s_vals) if s_vals else np.nan

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), squeeze=False)
    colors = {"CNN": args.color_cnn, "SAE": args.color_sae}

    for col_i, model in enumerate(["CNN", "SAE"]):
        ax = axes[0, col_i]

        if not plot_data[model]["cos"]:
            ax.set_visible(False)
            continue

        x_cls = np.concatenate(plot_data[model]["cos"])
        y_cls = np.concatenate(plot_data[model]["dpt"])
        color = colors.get(model, "#555555")

        ax.scatter(
            x_cls,
            y_cls,
            c=color,
            alpha=args.alpha,
            s=args.point_size,
            linewidths=0,
            rasterized=True,
        )

        has_gam = False
        adj_r2 = np.nan
        try:
            x_pred, y_line, y_lo, y_hi, adj_r2 = fit_gam_regression(
                x_cls,
                y_cls,
                n_splines=args.n_splines,
                lam=args.gam_lam,
                degree=args.poly_degree,
                seed=args.seed,
                clip_pct=args.gam_clip_pct,
                ci_width=args.gam_ci_width,
            )
            ax.plot(
                x_pred,
                y_line,
                color="black",
                linewidth=2.2,
                zorder=5,
                label=f"GAM  adj-R²={adj_r2:.3f}",
            )
            if np.isfinite(y_lo).any():
                ax.fill_between(
                    x_pred, y_lo, y_hi, alpha=0.18, color="black", zorder=4, linewidth=0
                )
            has_gam = True
        except Exception as e:
            print(f"    [{model}] GAM failed: {e}")

        xy_max = max(float(x_cls.max()), float(y_cls.max())) * 1.05
        ax.plot(
            [0, xy_max],
            [0, xy_max],
            color="#aaaaaa",
            linewidth=1.0,
            linestyle="--",
            alpha=0.65,
        )

        x_lim = float(x_cls.max()) * 1.06
        y_lim = float(y_cls.max()) * 1.06
        ax.set_xlim(-x_lim * 0.01, x_lim)
        ax.set_ylim(-y_lim * 0.01, y_lim)

        g_rho = avg_stats.get(f"{model}_gl_spearman", np.nan)
        g_tau = avg_stats.get(f"{model}_gl_kendall", np.nan)
        r_rho = avg_stats.get(f"{model}_rw_spearman", np.nan)

        ann = f"Global ρ={g_rho:.3f}  τ={g_tau:.3f}"
        ann += f"\nRow-wise ρ={r_rho:.3f}"
        if has_gam:
            ann += f"\nadj-R²={adj_r2:.3f}"
        ann += f"\nn_disp={len(x_cls):,}"

        ax.text(
            0.04,
            0.96,
            ann,
            transform=ax.transAxes,
            fontsize=8.0,
            va="top",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                alpha=0.85,
                edgecolor="#cccccc",
            ),
        )

        ax.set_xlabel("Cosine distance", fontsize=11)
        ax.set_ylabel("DPT geodesic" if col_i == 0 else "", fontsize=11)
        ax.set_title(
            f"{model}  │  All Classes (Inter + Intra)", fontsize=12, fontweight="bold"
        )
        if has_gam:
            ax.legend(fontsize=9, loc="lower right")
        import seaborn as sns

        sns.despine(ax=ax)

    fig.suptitle(
        "Global Geometry: Cosine vs DPT (All Pairs)", fontsize=14, fontweight="bold"
    )
    fig.tight_layout()
    _save_fig(fig, out_dir, "fig_global_scatter_gam", args)


def plot_seed_stability(seed_rows, args, out_dir):
    """Plot seed-to-seed variability in cosine distance distribution.

    Fig: 4 panels (one per stat: median, IQR, p95, max).
    Each panel: strip plot of per-seed values, box overlay, CNN vs SAE.
    Highlights that CNN varies widely across seeds, SAE is more stable.
    """
    import pandas as pd

    if not seed_rows:
        print("  [seed stability] No data — skipping.")
        return

    df = pd.DataFrame(seed_rows)
    colors = {"CNN": args.color_cnn, "SAE": args.color_sae}
    models = [m for m in ["CNN", "SAE"] if m in df["model"].values]

    metrics = [
        ("median", "Median cosine distance"),
        ("iqr", "IQR (Q75 − Q25)"),
        ("p95", "95th percentile"),
        ("max", "Maximum cosine distance"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 5.0))

    for ax, (stat, ylabel) in zip(axes, metrics):
        # Box
        bp_data = [df.loc[df["model"] == m, stat].values for m in models]
        bp = ax.boxplot(
            bp_data,
            positions=range(len(models)),
            widths=0.38,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(linewidth=1.4),
            capprops=dict(linewidth=1.4),
            flierprops=dict(marker=""),
        )
        for patch, model in zip(bp["boxes"], models):
            patch.set_facecolor(colors.get(model, "#ccc"))
            patch.set_alpha(0.35)

        # Strip (individual seeds)
        for xi, model in enumerate(models):
            vals = df.loc[df["model"] == model, stat].values
            jitter = np.random.RandomState(args.seed).uniform(
                -0.12, 0.12, size=len(vals)
            )
            ax.scatter(
                xi + jitter,
                vals,
                color=colors.get(model, "#555"),
                s=40,
                zorder=5,
                alpha=0.85,
                edgecolors="white",
                linewidths=0.5,
            )
            # Annotate each seed
            for j, (jit, v) in enumerate(zip(jitter, vals)):
                ax.text(xi + jit, v, f" {j}", fontsize=6, va="center", alpha=0.7)

        # Mean line per model
        for xi, model in enumerate(models):
            mn = df.loc[df["model"] == model, stat].mean()
            ax.hlines(
                mn,
                xi - 0.22,
                xi + 0.22,
                colors=colors.get(model, "#333"),
                linewidths=2.2,
                linestyles="-",
                zorder=6,
            )

        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"Per-seed {ylabel}", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.15, axis="y")
        sns.despine(ax=ax)

    # Compute and print CV (coefficient of variation)
    print(f"\n  [Seed stability] CV = std/mean per stat:")
    print(f"  {'Stat':<10s}", end="")
    for m in models:
        print(f"  {m:>10s}", end="")
    print()
    for stat, _ in metrics:
        print(f"  {stat:<10s}", end="")
        for m in models:
            vals = df.loc[df["model"] == m, stat].values
            cv = float(np.std(vals) / np.mean(vals)) if np.mean(vals) > 0 else np.nan
            print(f"  {cv:>10.4f}", end="")
        print()

    fig.suptitle(
        "Seed-to-seed variability of cosine distance distribution"
        "\n(CNN = high variance, SAE = stable)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    _save_fig(fig, out_dir, "fig_seed_stability", args)

    # ── Additional: ridge-style overlay of per-seed KDEs ──────────────────
    n_models = len(models)
    fig2, axes2 = plt.subplots(1, n_models, figsize=(6.5 * n_models, 4.5), sharey=False)
    if n_models == 1:
        axes2 = [axes2]

    for ax2, model in zip(axes2, models):
        paths_m = [r["seed_label"] for r in seed_rows if r["model"] == model]
        base_color = colors.get(model, "#666")
        cmap = matplotlib.colormaps.get_cmap("Blues" if model == "CNN" else "Oranges")
        n_seeds = len(df[df["model"] == model])

        # Reload each seed's raw cosine for KDE
        # (use the rows' seed data  — fast heuristic via stats values)
        for si, row in enumerate(r for r in seed_rows if r["model"] == model):
            # Reconstruct approximate distribution from quantiles via interpolation
            # (avoids re-loading large NPZ; gives a rough shape)
            alpha_i = 0.55 + 0.35 * si / max(n_seeds - 1, 1)
            color_i = cmap(0.35 + 0.55 * si / max(n_seeds - 1, 1))
            # Use actual per-seed lines from stored npz if accessible
            # Approximate: normal around median with iqr-based sigma
            approx_mu = row["median"]
            approx_sigma = row["iqr"] / 1.35  # Gaussian IQR approx
            if approx_sigma <= 0:
                continue
            x_range = np.linspace(
                max(0, approx_mu - 3 * approx_sigma), row["p99"] * 1.1, 300
            )
            from scipy.stats import norm

            y_kde = norm.pdf(x_range, approx_mu, approx_sigma)
            ax2.plot(
                x_range,
                y_kde,
                color=color_i,
                linewidth=1.6,
                alpha=0.8,
                label=f"seed {si}",
            )

        ax2.set_xlabel("Cosine distance (intra-class)", fontsize=11)
        ax2.set_ylabel("Density (approx)", fontsize=10)
        ax2.set_title(
            f"{model} — per-seed distribution", fontsize=12, fontweight="bold"
        )
        ax2.legend(fontsize=7, ncol=2)
        ax2.grid(True, alpha=0.12)
        sns.despine(ax=ax2)

    fig2.suptitle(
        "Per-seed cosine distribution shape (approximate)",
        fontsize=13,
        fontweight="bold",
    )
    fig2.tight_layout()
    _save_fig(fig2, out_dir, "fig_seed_kde_ridge", args)


# ══════════════════════════════════════════════════════════════════════════════
# Helper
# ══════════════════════════════════════════════════════════════════════════════
def _save_fig(fig, out_dir, name, args):
    os.makedirs(out_dir, exist_ok=True)
    exts = [".png"] if args.no_svg else [".png", ".svg"]
    for ext in exts:
        fig.savefig(
            os.path.join(out_dir, name + ext), dpi=args.dpi, bbox_inches="tight"
        )
    print(f"  Saved: {name}.png" + ("" if args.no_svg else "/.svg"))
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    args = get_args()

    # ──────────────────────────────────────────────────────────
    # VERSION BANNER — always print first
    print(f"\n{'='*70}")
    print(f"  plot_pairwise_gam   VERSION {__version__}")
    print(f"  If this is not v2.0-perclass, you have the WRONG file in Colab.")
    print(f"  Run:  !python plot_pairwise_gam.py --check_version")
    print(f"{'='*70}\n")

    if args.check_version:
        print(f"plot_pairwise_gam  v{__version__}")
        print(f"  Per-class GAM: YES")
        print(f"  Seed by name:  YES  (--cnn_seed_idx 445 finds cnn_seed_445/)")
        print(
            f"  pygam backend: {'YES (LinearGAM)' if _HAS_PYGAM else 'NO  (scipy fallback)'}"
        )
        sys.exit(0)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  plot_pairwise_gam — Cosine vs DPT Analysis")
    print(
        f"  backend: {'pygam' if _HAS_PYGAM else 'scipy (install pygam for full GAM)'}"
    )
    print(f"{'='*70}\n")

    # ── 1. Discover seed files ─────────────────────────────────────────────
    seed_paths = {}  # {"CNN": [path0,...], "SAE": [path0,...]}

    if args.results_dir:
        print(f"  Scanning {args.results_dir} ...")
        seed_paths = discover_seed_npzs(args.results_dir)
        for model, paths in seed_paths.items():
            print(f"  {model}: {len(paths)} seeds found")
            for i, p in enumerate(paths):
                print(f"    [{i}] {p}")

    # Single-file override
    if args.cnn_npz:
        seed_paths["CNN"] = [args.cnn_npz]
    if args.sae_npz:
        seed_paths["SAE"] = [args.sae_npz]

    if not seed_paths:
        print("ERROR: provide --results_dir or --cnn_npz / --sae_npz")
        sys.exit(1)

    # ── 2. Multi-seed per-class correlations ───────────────────────────────
    rows_by_model = {}
    summaries = {}
    if not args.scatter_only:
        print(f"\n{'─'*60}")
        print(f"  Multi-seed × per-class correlations")
        print(f"{'─'*60}")
        for model, paths in seed_paths.items():
            print(f"\n  [{model}]")
            rows = sweep_all_seeds(paths, args)
            rows_by_model[model] = rows
            summaries[model] = summarise_rows(rows, label=model)

        # CNN vs SAE comparison
        if "CNN" in summaries and "SAE" in summaries:
            print_cnn_sae_comparison(summaries["CNN"], summaries["SAE"])

        # Save CSVs
        save_stats_csv(rows_by_model, summaries, args.output_dir)

        # Save summary JSON
        with open(os.path.join(args.output_dir, "corr_summary.json"), "w") as f:
            json.dump(summaries, f, indent=2, default=str)

    # ── 3. Load selected seeds for scatter / GAM / sliding window ──────────
    print(f"\n{'─'*60}")
    print(f"  Loading selected seeds for scatter + GAM")
    print(f"  CNN seed_idx={args.cnn_seed_idx}  |  SAE seed_idx={args.sae_seed_idx}")
    print(f"{'─'*60}")
    display_data = {}
    seed_idx_map = {"CNN": args.cnn_seed_idx, "SAE": args.sae_seed_idx}

    for model, paths in seed_paths.items():
        path, seed_label = _resolve_seed_path(
            paths, seed_idx_map.get(model, 0), model_label=model
        )
        print(f"  [{model}] Loading: {path}")
        cos, dpt, ci, cj = load_npz(path, args)
        print(
            f"    {len(cos):,} pairs  cosine range [{cos.min():.4f}, {cos.max():.4f}]"
        )
        display_data[model] = (cos, dpt, ci, cj, seed_label)

    # ── 3.5. Row-wise (per-anchor) correlations ────────────────────────
    if not args.scatter_only:
        print(f"\n{'-'*60}")
        print(f"  Row-wise correlation sweep (all points as anchors)")
        rw_raw = sweep_rowwise_seeds(seed_paths, args)
        rw_summ = summarise_rowwise(rw_raw)

        # Attach summary to args so plot_scatter_gam can read it via getattr
        args._rw_summ = rw_summ

        # Save row-wise CSV
        rw_csv = os.path.join(args.output_dir, "rowwise_corr.csv")
        import csv as _csv2

        with open(rw_csv, "w", newline="", encoding="utf-8") as f:
            w = _csv2.writer(f)
            w.writerow(
                [
                    "model",
                    "class",
                    "rw_spearman_mean",
                    "rw_spearman_sem",
                    "rw_pearson_mean",
                    "rw_pearson_sem",
                    "n_seeds",
                ]
            )
            for model, cls_dict in rw_summ.items():
                for cls, v in sorted(cls_dict.items()):
                    w.writerow(
                        [
                            model,
                            cls,
                            f"{v['rw_sp']:.6f}",
                            f"{v.get('rw_sp_sem',np.nan):.6f}",
                            f"{v['rw_pr']:.6f}",
                            f"{v.get('rw_pr_sem',np.nan):.6f}",
                            v.get("n_seeds", "?"),
                        ]
                    )
        print(f"  Saved row-wise CSV: {rw_csv}")

    # ── 4. Figure 1: Global Scatter + GAM ────────────────────────────────
    print(f"\n{'-'*60}")
    print(f"  Figure 1: Global Scatter + Linear Regression (All Classes Combined)")
    plot_scatter_gam(display_data, rows_by_model, args, args.output_dir)

    if args.scatter_only:
        print(f"\n{'='*70}")
        print(f"  [--scatter_only] Skip remaining. Done → {args.output_dir}")
        print(f"{'='*70}\n")
        return

    # ── 5. Figure 2: Sliding window ────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Figure 2: Sliding window")
    # Adapt display_data to (cos, dpt, seed_label) for sliding window
    sw_data = {m: (tup[0], tup[1], tup[4]) for m, tup in display_data.items()}
    plot_sliding_window(sw_data, args, args.output_dir)

    # ── 6. Figure 3: KDE (all seeds pooled) ───────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Figure 3: KDE cosine distribution (all seeds)")
    plot_kde_cosine(seed_paths, args, args.output_dir)

    # ── 7. Figure 4: Seed stability ────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Figure 4: Seed-to-seed cosine stability")
    seed_stats_rows = compute_seed_cosine_stats(seed_paths, rng_seed=args.seed)

    # Save seed stats CSV
    if seed_stats_rows:
        stab_path = os.path.join(args.output_dir, "seed_cosine_stats.csv")
        import csv as _csv_mod

        with open(stab_path, "w", newline="", encoding="utf-8") as f:
            w = _csv_mod.DictWriter(f, fieldnames=list(seed_stats_rows[0].keys()))
            w.writeheader()
            w.writerows(seed_stats_rows)
        print(f"  Saved seed stats: {stab_path}")

    plot_seed_stability(seed_stats_rows, args, args.output_dir)

    print(f"\n{'='*70}")
    print(f"  Done → {args.output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
