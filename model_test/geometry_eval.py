# ==============================================================================
# Intrinsic Geometry Evaluation — Ollivier-Ricci Curvature & δ-Hyperbolicity
#
# CNN SupCon은 features를 unit hypersphere 위에 강제 → 양의 곡률(positive curvature).
# SAE는 이 제약을 해제 → 생물학적 본질의 비선형 매니폴드 복원 가능.
#
# 이 스크립트는 두 가지 내재적 기하학 지표를 측정:
#   1) Ollivier-Ricci Curvature — KNN graph edge별 곡률 분포
#      - Positive: sphere-like (SupCon → 이쪽으로 편향)
#      - Negative: hyperbolic/tree-like (생물학적 계층 구조)
#      - Zero: flat/Euclidean
#   2) Gromov δ-Hyperbolicity — metric space가 tree-like인 정도
#      - δ ≈ 0: strongly hyperbolic (tree-like)
#      - δ large: non-hyperbolic
#
# Usage:
#   python -m model_test.geometry_eval \
#       --cnn_cache /path/to/cnn_gap_stage5_out_all.npz \
#       --sae_cache /path/to/features_cache_...npz \
#       --gap_l2_norm \
#       --samples_per_class 500 \
#       --k_neighbors 15 \
#       --output_dir /path/to/output
#
#   # Single source:
#   python -m model_test.geometry_eval \
#       --cnn_cache /path/to/cnn_gap_stage5_out_all.npz \
#       --gap_l2_norm \
#       --label stage5_out \
#       --output_dir /path/to/output
#
# Dependencies:
#   pip install GraphRicciCurvature networkx
# ==============================================================================

import os
import sys
import csv
import json
import time
import argparse
import numpy as np

import matplotlib
_IN_COLAB = "google.colab" in sys.modules
if not _IN_COLAB:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sae_project.step02_logging_utils import get_logger, SUPERCLASS_MAP, CLASS_TO_LABEL
from apoptosis_prediction.local_knn_std import load_cache
from model_test.knn_fewshot_eval import _weighted_vote, NUM_CLASSES, CLASS_NAMES

# GPU detection
import torch
_HAS_CUDA = torch.cuda.is_available()
_DEVICE = torch.device("cuda" if _HAS_CUDA else "cpu")

logger = get_logger("geometry_eval")

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'
sns.set_style("ticks")


# ==============================================================================
# Argument Parser
# ==============================================================================
def get_args():
    p = argparse.ArgumentParser(
        description="Intrinsic geometry evaluation: "
                    "Ollivier-Ricci curvature & δ-hyperbolicity")

    # Data
    p.add_argument("--cnn_cache", type=str, default="")
    p.add_argument("--sae_cache", type=str, default="")
    p.add_argument("--dead_threshold", type=float, default=1e-5)
    p.add_argument("--gap_l2_norm", action="store_true",
                   help="L2 normalize feature vectors")
    p.add_argument("--label", type=str, default="",
                   help="Custom label for this run")

    # Geometry parameters
    p.add_argument("--k_neighbors", type=int, nargs="+", default=[15],
                   help="K for KNN graph construction (multiple for sweep). "
                        "Default: 15")
    p.add_argument("--ricci_alpha", type=float, default=0.5,
                   help="Ollivier-Ricci alpha (lazy random walk param). "
                        "0.5 = standard. Default: 0.5")
    p.add_argument("--delta_n_samples", type=int, default=10000,
                   help="Number of random quadruples for δ-hyperbolicity. "
                        "Default: 10000")

    # Diffusion distance (recommended for biological data)
    p.add_argument("--use_diffusion", action="store_true",
                   help="Use diffusion distance instead of Euclidean. "
                        "Computes sc.tl.diffmap and uses X_diffmap coordinates. "
                        "More robust to noise, respects manifold structure.")
    p.add_argument("--n_diffmap_comps", type=int, default=100,
                   help="Number of diffusion map components to compute. "
                        "DC0 (constant) is excluded automatically. Default: 15")
    p.add_argument("--pca_dim", type=int, default=300,
                   help="PCA dimensions before diffmap (0 = skip PCA). Default: 50")
    p.add_argument("--n_neighbors_diffmap", type=int, default=15,
                   help="K for scanpy KNN graph (adaptive kernel). Default: 30")

    # Subsampling (CRITICAL for Ricci — O(N·K²) per edge)
    p.add_argument("--samples_per_class", type=int, default=500,
                   help="Max samples per class. Default: 500 "
                        "(2000 total for 4 classes)")

    # Analysis scope
    p.add_argument("--per_class", action="store_true",
                   help="Also compute per-class curvature distributions")

    # ── Intrinsic Geometry gate ──────────────────────────────────────
    # Ricci curvature and δ-hyperbolicity are O(N·K²) and very slow.
    # Only computed when this flag is set.
    p.add_argument("--compute_geometry", action="store_true",
                   help="Compute Ollivier-Ricci curvature and Gromov "
                        "δ-hyperbolicity. VERY SLOW — skip by default.")

    # ── Apoptosis local std analysis ─────────────────────────────────
    p.add_argument("--apoptosis_csv", type=str, default="",
                   help="Path to per-image apoptosis CSV "
                        "(required for local apoptosis std analysis).")
    p.add_argument("--apoptosis_k_neighbors", type=int, nargs="+",
                   default=[5, 10, 15, 20, 25],
                   help="K values for apoptosis local std sweep. "
                        "Default: 5 10 15 20 25")

    # KNN confidence filter — clean subset
    p.add_argument("--knn_filter", action="store_true",
                   help="Filter to correctly classified samples (LOO KNN). "
                        "Runs geometry on both full and clean subsets.")
    p.add_argument("--knn_filter_k", type=int, default=10,
                   help="K for LOO KNN filter. Default: 10")
    p.add_argument("--knn_filter_weights", type=str, default="inv_sq",
                   choices=["inv_sq", "distance", "uniform"],
                   help="KNN weighting: inv_sq (1/d²), distance (1/d), "
                        "uniform. Default: inv_sq")

    # DPT vs cosine rank correlation
    p.add_argument("--rank_correlation", action="store_true",
                   help="Compute DPT vs cosine rank correlation "
                        "(isometry test)")
    p.add_argument("--n_rank_anchors", type=int, default=300,
                   help="Number of anchor points for rank correlation. "
                        "Default: 300")
    p.add_argument("--rank_pca_dim", type=int, default=0,
                   help="PCA dims for DPT in rank correlation. "
                        "0 = skip PCA (raw space). Default: 0")
    p.add_argument("--rank_n_neighbors", type=int, default=15,
                   help="K for scanpy neighbors in rank DPT. Default: 15")

    # ── Pairwise distance dump (for GAM fitting / downstream analysis) ──
    p.add_argument("--save_pairwise", action="store_true",
                   help="Save all sampled pairwise (cosine_dist, dpt_dist) "
                        "with point IDs/classes to NPZ + CSV. "
                        "Requires --rank_correlation to be set. "
                        "Files: pairwise_distances_<label>.npz/.csv")

    # Output
    p.add_argument("--output_dir", type=str, default="")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)

    # ── Aggregation mode ──────────────────────────────────────────────
    # When --summarize_dir is given, skip normal evaluation and instead
    # aggregate all apop_std_cosine_vs_dpt.json files found under that
    # directory, grouping by CNN/SAE, and print mean±std across seeds.
    p.add_argument("--summarize_dir", type=str, default="",
                   help="Aggregate apop_std results under this directory. "
                        "Groups cnn_seed_* / sae_seed_* subdirs, computes "
                        "mean±std across seeds per (metric, k). "
                        "Saves aggregated_apop_std_summary.csv/json.")

    return p.parse_args()






# ==============================================================================
# Comprehensive aggregation: all geometry metrics across seeds
# ==============================================================================
def aggregate_all_metrics(base_dir, out_dir=None, dpi=200):
    """Aggregate ALL geometry metrics across every cnn_seed_* / sae_seed_* dir.

    Reads (whichever exist in each seed dir):
      - rank_correlation.json          → Spearman ρ, Kendall τ, Pearson r,
                                         Kruskal stress, NPR(k=5/10/15/20/25)
      - apop_std_cosine_vs_dpt.json    → local apoptosis std (cosine vs DPT)
      - k_*/geometry_results*.json     → Ricci curvature, δ-hyperbolicity
        or geometry_results*.json

    Prints per-category summary tables:
        [Isometry] [Apoptosis std] [Ricci] [δ-Hyperbolicity]
    and a CNN vs SAE comparison block.

    Saves:
      aggregated_metrics_summary.csv
      aggregated_metrics_summary.json
      aggregated_metrics_comparison.png/.svg
    """
    import glob

    if out_dir is None:
        out_dir = base_dir
    os.makedirs(out_dir, exist_ok=True)

    # ── 1. Find seed directories ──────────────────────────────────────
    def _find_seed_dirs(base):
        dirs = {}
        for pattern, grp in [("cnn_seed_*", "CNN"), ("sae_seed_*", "SAE")]:
            for d in sorted(glob.glob(os.path.join(base, pattern))):
                if os.path.isdir(d):
                    dirs.setdefault(grp, []).append(d)
        return dirs

    seed_dirs = _find_seed_dirs(base_dir)

    # Fallback: if no cnn_seed_*/sae_seed_* found, try parent directory
    if not seed_dirs:
        seed_dirs = _find_seed_dirs(os.path.dirname(base_dir))

    if not seed_dirs:
        logger.warning(f"  No cnn_seed_*/sae_seed_* directories found under {base_dir}")
        logger.warning(f"  Trying to infer from JSON files directly...")
        # Last resort: scan json files and guess group from dirname
        seed_dirs = _infer_groups_from_jsons(base_dir)

    if not seed_dirs:
        logger.error(f"  Cannot find any seed directories. Aborting aggregation.")
        return {}

    for grp, dirs in seed_dirs.items():
        logger.info(f"  {grp}: {len(dirs)} seed dirs")

    # ── 2. Accumulate metrics per group ───────────────────────────────
    # acc[grp][metric_name] = [val_seed1, val_seed2, ...]
    acc = {grp: {} for grp in seed_dirs}

    for grp, dirs in seed_dirs.items():
        for d in dirs:
            _read_rank_correlation(d, acc[grp])
            _read_apop_std(d, acc[grp])
            _read_geometry_results(d, acc[grp])

    # ── 3. Compute statistics ─────────────────────────────────────────
    stats = {}
    for grp, grp_acc in acc.items():
        stats[grp] = {}
        for mname, vals in grp_acc.items():
            arr = np.array([v for v in vals if np.isfinite(v)], dtype=float)
            if len(arr) == 0:
                continue
            n = len(arr)
            stats[grp][mname] = {
                "mean":   float(np.mean(arr)),
                "sem":    float(np.std(arr) / np.sqrt(n)),
                "std":    float(np.std(arr)),
                "median": float(np.median(arr)),
                "min":    float(np.min(arr)),
                "max":    float(np.max(arr)),
                "n":      n,
            }

    # ── 4. Print Tables ───────────────────────────────────────────────
    _print_agg_tables(stats)

    # ── 5. CNN vs SAE comparison ──────────────────────────────────────
    _print_cnn_sae_comparison(stats)

    # ── 6. Save CSV ───────────────────────────────────────────────────
    csv_path = os.path.join(out_dir, "aggregated_metrics_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        import csv as _csv
        w = _csv.writer(f)
        w.writerow(["group", "metric", "mean", "sem", "std",
                    "median", "min", "max", "n"])
        for grp in sorted(stats):
            for mname in sorted(stats[grp]):
                v = stats[grp][mname]
                w.writerow([grp, mname,
                             f"{v['mean']:.6f}", f"{v['sem']:.6f}",
                             f"{v['std']:.6f}",  f"{v['median']:.6f}",
                             f"{v['min']:.6f}",  f"{v['max']:.6f}",
                             v['n']])
    logger.info(f"  Saved: {csv_path}")

    # ── 7. Save JSON ──────────────────────────────────────────────────
    json_path = os.path.join(out_dir, "aggregated_metrics_summary.json")
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    logger.info(f"  Saved: {json_path}")

    # ── 8. Plot ───────────────────────────────────────────────────────
    _plot_aggregated_metrics(stats, out_dir, dpi)

    return stats


# Keep old name as alias for backward compat
def aggregate_apop_std_results(base_dir, out_dir=None, dpi=200):
    return aggregate_all_metrics(base_dir, out_dir=out_dir, dpi=dpi)


# ── Readers ────────────────────────────────────────────────────────────

def _infer_groups_from_jsons(base_dir):
    """Fallback: find any apop_std/rank_correlation JSON and group by dirname."""
    import glob
    dirs = {}
    for jf in glob.glob(os.path.join(base_dir, "**", "*.json"), recursive=True):
        d = os.path.dirname(jf)
        dn = os.path.basename(d).lower()
        grp = "CNN" if "cnn" in dn else ("SAE" if "sae" in dn else None)
        if grp:
            if d not in dirs.get(grp, []):
                dirs.setdefault(grp, []).append(d)
    return dirs


def _src_to_group(src_key):
    """Map a source label like 'CNN (stage5_out)' → 'CNN'."""
    sk = src_key.upper()
    if "CNN" in sk:
        return "CNN"
    if "SAE" in sk:
        return "SAE"
    return None


def _read_rank_correlation(seed_dir, acc):
    """Read rank_correlation.json and accumulate isometry metrics."""
    import glob
    # Try both directly in seed_dir and in k_*/ subdirs
    candidates = glob.glob(os.path.join(seed_dir, "rank_correlation.json"))
    candidates += glob.glob(os.path.join(seed_dir, "k_*", "rank_correlation.json"))

    for fpath in candidates:
        try:
            with open(fpath) as f:
                data = json.load(f)
        except Exception:
            continue

        for src_key, src_data in data.items():
            if not isinstance(src_data, dict):
                continue
            if "error" in src_data:
                continue

            # Scalar isometry metrics
            for mname in [
                "spearman_mean", "spearman_std", "spearman_median",
                "kendall_mean",  "kendall_std",  "kendall_median",
                "pearson_mean",  "pearson_std",
                "kruskal_stress",
                "npr_mean", "npr_std",
            ]:
                if mname in src_data and isinstance(src_data[mname], (int, float)):
                    acc.setdefault(mname, []).append(float(src_data[mname]))

            # NPR per k
            for npr_key in ["npr_all", "npr_all_vals"]:
                if npr_key not in src_data or not isinstance(src_data[npr_key], dict):
                    continue
                for k_str, k_data in src_data[npr_key].items():
                    if not k_str.isdigit():
                        continue
                    if isinstance(k_data, dict):
                        for stat in ["mean", "std", "median"]:
                            if stat in k_data and isinstance(k_data[stat], (int, float)):
                                acc.setdefault(f"npr_k{k_str}_{stat}", []).append(
                                    float(k_data[stat]))
                    elif isinstance(k_data, list):
                        arr = [v for v in k_data if isinstance(v, (int, float))]
                        if arr:
                            acc.setdefault(f"npr_k{k_str}_mean", []).append(
                                float(np.mean(arr)))

            # Per-class spearman/kendall/pearson (average across classes)
            for prefix in ["per_class_spearman", "per_class_kendall",
                           "per_class_pearson", "per_class_stress"]:
                if prefix not in src_data or not isinstance(src_data[prefix], dict):
                    continue
                vals = [v for v in src_data[prefix].values()
                        if isinstance(v, (int, float)) and np.isfinite(v)]
                if vals:
                    base_name = prefix.replace("per_class_", "") + "_mean"
                    acc.setdefault(f"cls_avg_{base_name}", []).append(
                        float(np.mean(vals)))


def _read_apop_std(seed_dir, acc):
    """Read apop_std_cosine_vs_dpt.json and accumulate apoptosis local std."""
    import glob
    candidates = glob.glob(os.path.join(seed_dir, "apop_std_cosine_vs_dpt.json"))

    for fpath in candidates:
        try:
            with open(fpath) as f:
                data = json.load(f)
        except Exception:
            continue

        for metric in ["cosine", "DPT"]:
            if metric not in data:
                continue
            for k_str, kdata in data[metric].items():
                if not k_str.isdigit() or not isinstance(kdata, dict):
                    continue
                for stat in ["mean_std", "median_std", "std_std"]:
                    if stat in kdata and isinstance(kdata[stat], (int, float)):
                        acc.setdefault(f"apop_{metric}_k{k_str}_{stat}", []).append(
                            float(kdata[stat]))

        # Compute paired Δ (DPT - cosine) right here per seed
        for k_str in data.get("cosine", {}):
            if not k_str.isdigit():
                continue
            c = data["cosine"].get(k_str, {})
            d = data["DPT"].get(k_str, {})
            if "mean_std" in c and "mean_std" in d:
                delta = float(d["mean_std"]) - float(c["mean_std"])
                acc.setdefault(f"apop_delta_k{k_str}_mean_std", []).append(delta)
            if "median_std" in c and "median_std" in d:
                delta = float(d["median_std"]) - float(c["median_std"])
                acc.setdefault(f"apop_delta_k{k_str}_median_std", []).append(delta)


def _read_geometry_results(seed_dir, acc):
    """Read geometry_results*.json for Ricci curvature and δ-hyperbolicity."""
    import glob
    candidates = glob.glob(os.path.join(seed_dir, "geometry_results*.json"))
    candidates += glob.glob(os.path.join(seed_dir, "k_*",
                                         "geometry_results*.json"))

    for fpath in candidates:
        try:
            with open(fpath) as f:
                data = json.load(f)
        except Exception:
            continue

        for src_key, src_data in data.items():
            if not isinstance(src_data, dict):
                continue

            # Ricci
            ricci = src_data.get("ricci", {})
            if isinstance(ricci, dict) and "error" not in ricci:
                for m in ["mean", "median", "std",
                          "frac_positive", "frac_negative"]:
                    if m in ricci and isinstance(ricci[m], (int, float)):
                        acc.setdefault(f"ricci_{m}", []).append(float(ricci[m]))

            # δ-Hyperbolicity
            delta = src_data.get("delta", {})
            if isinstance(delta, dict) and "error" not in delta:
                for m in ["delta", "delta_rel", "diameter"]:
                    if m in delta and isinstance(delta[m], (int, float)):
                        acc.setdefault(f"hyp_{m}", []).append(float(delta[m]))


# ── Print helpers ──────────────────────────────────────────────────────

_METRIC_CATEGORIES = {
    "Isometry (Cosine vs DPT)": [
        ("spearman_mean",    "Spearman ρ  (mean)"),
        ("spearman_median",  "Spearman ρ  (median)"),
        ("kendall_mean",     "Kendall τ   (mean)"),
        ("kendall_median",   "Kendall τ   (median)"),
        ("pearson_mean",     "Pearson r   (mean)"),
        ("kruskal_stress",   "Kruskal stress"),
        ("npr_mean",         "NPR         (mean, default k)"),
        ("npr_k5_mean",      "NPR k=5"),
        ("npr_k10_mean",     "NPR k=10"),
        ("npr_k15_mean",     "NPR k=15"),
        ("npr_k20_mean",     "NPR k=20"),
        ("npr_k25_mean",     "NPR k=25"),
    ],
    "Apoptosis local std — cosine KNN": [
        ("apop_cosine_k5_mean_std",    "cosine k=5  mean_std"),
        ("apop_cosine_k10_mean_std",   "cosine k=10 mean_std"),
        ("apop_cosine_k15_mean_std",   "cosine k=15 mean_std"),
        ("apop_cosine_k20_mean_std",   "cosine k=20 mean_std"),
        ("apop_cosine_k25_mean_std",   "cosine k=25 mean_std"),
        ("apop_cosine_k5_median_std",  "cosine k=5  median_std"),
        ("apop_cosine_k10_median_std", "cosine k=10 median_std"),
        ("apop_cosine_k15_median_std", "cosine k=15 median_std"),
        ("apop_cosine_k20_median_std", "cosine k=20 median_std"),
        ("apop_cosine_k25_median_std", "cosine k=25 median_std"),
    ],
    "Apoptosis local std — DPT KNN": [
        ("apop_DPT_k5_mean_std",    "DPT k=5  mean_std"),
        ("apop_DPT_k10_mean_std",   "DPT k=10 mean_std"),
        ("apop_DPT_k15_mean_std",   "DPT k=15 mean_std"),
        ("apop_DPT_k20_mean_std",   "DPT k=20 mean_std"),
        ("apop_DPT_k25_mean_std",   "DPT k=25 mean_std"),
        ("apop_DPT_k5_median_std",  "DPT k=5  median_std"),
        ("apop_DPT_k10_median_std", "DPT k=10 median_std"),
        ("apop_DPT_k15_median_std", "DPT k=15 median_std"),
        ("apop_DPT_k20_median_std", "DPT k=20 median_std"),
        ("apop_DPT_k25_median_std", "DPT k=25 median_std"),
    ],
    "Apoptosis local std — Δ (DPT - cosine)": [
        ("apop_delta_k5_mean_std",    "Δ mean_std   k=5"),
        ("apop_delta_k10_mean_std",   "Δ mean_std   k=10"),
        ("apop_delta_k15_mean_std",   "Δ mean_std   k=15"),
        ("apop_delta_k20_mean_std",   "Δ mean_std   k=20"),
        ("apop_delta_k25_mean_std",   "Δ mean_std   k=25"),
        ("apop_delta_k5_median_std",  "Δ median_std k=5"),
        ("apop_delta_k10_median_std", "Δ median_std k=10"),
        ("apop_delta_k15_median_std", "Δ median_std k=15"),
        ("apop_delta_k20_median_std", "Δ median_std k=20"),
        ("apop_delta_k25_median_std", "Δ median_std k=25"),
    ],
    "Ricci Curvature": [
        ("ricci_mean",          "Ricci mean"),
        ("ricci_median",        "Ricci median"),
        ("ricci_std",           "Ricci std"),
        ("ricci_frac_positive", "Frac positive edges"),
        ("ricci_frac_negative", "Frac negative edges"),
    ],
    "δ-Hyperbolicity": [
        ("hyp_delta",     "δ"),
        ("hyp_delta_rel", "δ/diameter (relative)"),
        ("hyp_diameter",  "Graph diameter"),
    ],
}


def _print_agg_tables(stats):
    logger.info(f"\n{'='*80}")
    logger.info(f"  AGGREGATED GEOMETRY METRICS — ALL SEEDS")
    logger.info(f"{'='*80}")

    groups = sorted(stats.keys())

    for cat_name, mlist in _METRIC_CATEGORIES.items():
        # Check if any metric in this category exists in any group
        has_any = any(
            mname in stats.get(grp, {})
            for mname, _ in mlist
            for grp in groups
        )
        if not has_any:
            continue

        logger.info(f"\n  ── {cat_name} ──")
        # Header
        hdr = f"  {'Metric':<32s}"
        for grp in groups:
            n_seeds = max(
                (stats[grp].get(mname, {}).get("n", 0) for mname, _ in mlist),
                default=0)
            hdr += f"  {grp:>8s}({n_seeds:d}s) mean±SEM"
        logger.info(hdr)
        logger.info(f"  {'─'*70}")

        for mname, label in mlist:
            row = f"  {label:<32s}"
            has_val = False
            for grp in groups:
                v = stats[grp].get(mname)
                if v:
                    row += f"  {v['mean']:+9.5f} ±{v['sem']:.5f}"
                    has_val = True
                else:
                    row += f"  {'—':>20s}"
            if has_val:
                logger.info(row)


def _print_cnn_sae_comparison(stats):
    if "CNN" not in stats or "SAE" not in stats:
        return

    logger.info(f"\n{'='*80}")
    logger.info(f"  CNN vs SAE — per-metric comparison")
    logger.info(f"{'='*80}")
    logger.info(f"  {'Metric':<32s}  {'CNN mean':>12s}  {'SAE mean':>12s}"
                f"  {'Δ(CNN-SAE)':>12s}  Direction")
    logger.info(f"  {'─'*80}")

    all_metric_names = []
    for _, mlist in _METRIC_CATEGORIES.items():
        for mname, label in mlist:
            if mname in stats["CNN"] or mname in stats["SAE"]:
                all_metric_names.append((mname, label))

    current_cat = None
    for cat_name, mlist in _METRIC_CATEGORIES.items():
        mnames_in_cat = [mn for mn, _ in mlist]
        cat_printed = False
        for mname, label in mlist:
            c = stats["CNN"].get(mname)
            s = stats["SAE"].get(mname)
            if c is None and s is None:
                continue
            if not cat_printed:
                logger.info(f"  [{cat_name}]")
                cat_printed = True
            c_val = c["mean"] if c else float("nan")
            s_val = s["mean"] if s else float("nan")
            delta = c_val - s_val
            # Higher-is-better metrics
            higher_is_better = any(kw in mname for kw in [
                "spearman", "kendall", "pearson", "npr", "ricci_frac_pos"])
            # Lower-is-better metrics
            lower_is_better = any(kw in mname for kw in [
                "kruskal", "apop_", "hyp_delta"])
            if higher_is_better:
                direction = "✓ SAE better" if s_val > c_val else "✗ CNN better"
            elif lower_is_better:
                direction = "✓ SAE better" if s_val < c_val else "✗ CNN better"
            else:
                direction = "—"
            logger.info(f"  {label:<32s}  {c_val:>12.5f}  {s_val:>12.5f}"
                        f"  {delta:>+12.5f}  {direction}")

    logger.info(f"{'='*80}")


# ── Comparison plot ─────────────────────────────────────────────────────

def _plot_aggregated_metrics(stats, out_dir, dpi=200):
    """Bar charts: CNN vs SAE for key summary metrics."""
    # Select representative scalar metrics for plotting
    plot_metrics = [
        ("spearman_mean",        "Spearman ρ"),
        ("kendall_mean",         "Kendall τ"),
        ("pearson_mean",         "Pearson r"),
        ("kruskal_stress",       "Kruskal stress"),
        ("npr_k5_mean",          "NPR k=5"),
        ("npr_k15_mean",         "NPR k=15"),
        ("apop_cosine_k5_mean_std",  "Apop std\ncosine k=5"),
        ("apop_DPT_k5_mean_std",     "Apop std\nDPT k=5"),
        ("apop_delta_k5_mean_std",   "Δ apop std\nDPT-cos k=5"),
        ("ricci_mean",           "Ricci mean"),
        ("ricci_frac_positive",  "Ricci frac+"),
        ("hyp_delta",            "δ-Hyperbolicity"),
    ]

    # Filter to available metrics
    available = [(mn, lbl) for mn, lbl in plot_metrics
                 if any(mn in stats.get(g, {}) for g in stats)]
    if not available:
        return

    n_metrics = len(available)
    n_cols = min(6, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.2 * n_cols, 3.5 * n_rows))
    axes = np.array(axes).reshape(-1)

    colors_map = {"CNN": "#3A7EBF", "SAE": "#E8833A"}
    groups = [g for g in ["CNN", "SAE"] if g in stats]

    for ax_idx, (mname, label) in enumerate(available):
        ax = axes[ax_idx]
        vals = [stats[g].get(mname, {}).get("mean", np.nan) for g in groups]
        sems = [stats[g].get(mname, {}).get("sem",  np.nan) for g in groups]
        ns   = [stats[g].get(mname, {}).get("n",  0)        for g in groups]
        x = np.arange(len(groups))
        bars = ax.bar(x, vals, color=[colors_map.get(g, "#999") for g in groups],
                      alpha=0.82, width=0.5, zorder=3)
        ax.errorbar(x, vals, yerr=sems, fmt="none", color="black",
                    capsize=4, linewidth=1.5, zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{g}\n(n={n})" for g, n in zip(groups, ns)],
                           fontsize=8)
        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.grid(True, alpha=0.15, axis="y")
        sns.despine(ax=ax)

    # Hide unused axes
    for ax in axes[n_metrics:]:
        ax.set_visible(False)

    fig.suptitle("Aggregated Geometry Metrics — CNN vs SAE (mean ± SEM across seeds)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fname = "aggregated_metrics_comparison"
    for ext in [".png", ".svg"]:
        fig.savefig(os.path.join(out_dir, fname + ext),
                    dpi=dpi, bbox_inches="tight")
    logger.info(f"  Saved: {fname}.png/.svg")
    if _IN_COLAB:
        plt.show()
    plt.close(fig)




# ==============================================================================
# Build KNN graph (networkx)
# ==============================================================================
def build_knn_graph(X, k, labels=None):
    """Build a weighted KNN graph from feature matrix.

    Parameters
    ----------
    X : np.ndarray (N, d)
    k : int — number of neighbors
    labels : list[str] or None — node labels for analysis

    Returns
    -------
    G : networkx.Graph — weighted KNN graph (weight = Euclidean distance)
    """
    import networkx as nx
    from sklearn.neighbors import NearestNeighbors

    n = len(X)
    k_actual = min(k, n - 1)

    nn = NearestNeighbors(n_neighbors=k_actual + 1, metric="euclidean",
                          n_jobs=-1)
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    G = nx.Graph()
    for i in range(n):
        node_attrs = {"label": labels[i]} if labels is not None else {}
        G.add_node(i, **node_attrs)

    for i in range(n):
        for j_idx in range(1, k_actual + 1):  # skip self (index 0)
            j = indices[i, j_idx]
            d = distances[i, j_idx]
            if not G.has_edge(i, j):
                G.add_edge(i, j, weight=float(d))

    logger.info(f"    KNN graph: {G.number_of_nodes()} nodes, "
                f"{G.number_of_edges()} edges (k={k_actual})")
    return G


# ==============================================================================
# Ollivier-Ricci Curvature
# ==============================================================================
def compute_ricci_curvature(X, k, alpha=0.5, labels=None):
    """Compute Ollivier-Ricci curvature on KNN graph.

    Returns
    -------
    curvatures : np.ndarray — per-edge curvatures
    stats : dict — summary statistics
    G : networkx.Graph — graph with curvature annotations
    """
    from GraphRicciCurvature.OllivierRicci import OllivierRicci

    logger.info(f"    Building KNN graph (k={k})...")
    G = build_knn_graph(X, k, labels=labels)

    logger.info(f"    Computing Ollivier-Ricci curvature (alpha={alpha})...")
    t0 = time.time()
    # Force method="OT" (Exact Wasserstein/EMD). 
    # Sinkhorn is prone to numerical underflow (divide by zero) and is slower for small k.
    orc = OllivierRicci(G, alpha=alpha, method="OTD", verbose="ERROR",
                        proc=os.cpu_count())  # multiprocessing
    orc.compute_ricci_curvature()
    elapsed = time.time() - t0
    logger.info(f"    Ricci computation: {elapsed:.1f}s ({os.cpu_count()} procs)")

    # Extract edge curvatures
    curvatures = np.array([
        d.get("ricciCurvature", 0.0)
        for u, v, d in orc.G.edges(data=True)
    ])

    # Summary statistics
    stats = {
        "mean": float(np.mean(curvatures)),
        "median": float(np.median(curvatures)),
        "std": float(np.std(curvatures)),
        "min": float(np.min(curvatures)),
        "max": float(np.max(curvatures)),
        "q25": float(np.percentile(curvatures, 25)),
        "q75": float(np.percentile(curvatures, 75)),
        "frac_positive": float(np.mean(curvatures > 0)),
        "frac_negative": float(np.mean(curvatures < 0)),
        "frac_zero": float(np.mean(np.abs(curvatures) < 1e-6)),
        "n_edges": len(curvatures),
        "compute_time_s": elapsed,
    }

    logger.info(f"    Ricci curvature: mean={stats['mean']:.4f}, "
                f"median={stats['median']:.4f}, "
                f"positive={stats['frac_positive']:.1%}, "
                f"negative={stats['frac_negative']:.1%}")

    return curvatures, stats, orc.G


def compute_ricci_per_class(G_ricci, labels):
    """Extract curvature distributions for edges within each class.

    Returns
    -------
    class_curvatures : dict[str, np.ndarray]
    """
    class_curvatures = {}
    labels_arr = np.array(labels)

    for cls in sorted(np.unique(labels_arr)):
        cls_nodes = set(np.where(labels_arr == cls)[0])
        cls_curvs = []
        for u, v, d in G_ricci.edges(data=True):
            if u in cls_nodes and v in cls_nodes:
                cls_curvs.append(d.get("ricciCurvature", 0.0))
        if cls_curvs:
            class_curvatures[cls] = np.array(cls_curvs)
            logger.info(f"      {cls}: {len(cls_curvs)} intra-class edges, "
                        f"mean={np.mean(cls_curvs):.4f}")

    return class_curvatures


# ==============================================================================
# Gromov δ-Hyperbolicity
# ==============================================================================
def compute_delta_hyperbolicity(X, n_quadruples=10000, seed=42):
    """Estimate Gromov δ-hyperbolicity by sampling random quadruples.

    For a metric space (X, d), the 4-point condition:
      For any (x, y, z, w), sort the three sums:
        S1 = d(x,y) + d(z,w)
        S2 = d(x,z) + d(y,w)
        S3 = d(x,w) + d(y,z)
      Then δ(x,y,z,w) = (S_max - S_mid) / 2

    δ = max over all quadruples.
    Normalized: δ_rel = δ / diameter

    Returns
    -------
    delta : float — estimated δ-hyperbolicity
    delta_rel : float — δ / diameter (scale-invariant)
    stats : dict
    """
    from scipy.spatial.distance import pdist, squareform

    n = len(X)
    logger.info(f"    Computing pairwise distances ({n} samples)...")

    # Pairwise distance matrix — GPU if available
    if _HAS_CUDA and n <= 20000:
        logger.info(f"    Using GPU (torch) for pdist...")
        X_t = torch.from_numpy(X.astype(np.float32)).to(_DEVICE)
        D_t = torch.cdist(X_t, X_t)  # (N, N)
        D = D_t.cpu().numpy().astype(np.float64)
        del X_t, D_t
        torch.cuda.empty_cache()
    else:
        mem_gb = (n * n * 8) / (1024 ** 3)
        if n > 30000:
            logger.warning(f"    N={n}: distance matrix ~{mem_gb:.1f} GB. "
                           f"Ensure sufficient RAM.")
        D = squareform(pdist(X, metric="euclidean"))

    diameter = float(np.max(D))

    logger.info(f"    Diameter: {diameter:.4f}")
    logger.info(f"    Sampling {n_quadruples} quadruples...")

    # Vectorized quadruple sampling for speed
    rng = np.random.RandomState(seed)
    idx_all = rng.choice(n, size=(n_quadruples, 4), replace=True)
    # Ensure no duplicate within each quadruple
    for i in range(n_quadruples):
        while len(set(idx_all[i])) < 4:
            idx_all[i] = rng.choice(n, size=4, replace=False)

    x, y, z, w = idx_all[:, 0], idx_all[:, 1], idx_all[:, 2], idx_all[:, 3]
    s1 = D[x, y] + D[z, w]
    s2 = D[x, z] + D[y, w]
    s3 = D[x, w] + D[y, z]

    sums = np.stack([s1, s2, s3], axis=1)  # (n_quadruples, 3)
    sums.sort(axis=1)
    delta_values = (sums[:, 2] - sums[:, 1]) / 2.0
    max_delta = float(delta_values.max())

    delta_rel = max_delta / max(diameter, 1e-12)

    stats = {
        "delta": float(max_delta),
        "delta_rel": float(delta_rel),
        "diameter": diameter,
        "delta_mean": float(np.mean(delta_values)),
        "delta_median": float(np.median(delta_values)),
        "delta_q95": float(np.percentile(delta_values, 95)),
        "delta_q99": float(np.percentile(delta_values, 99)),
        "n_quadruples": n_quadruples,
        "n_samples": n,
    }

    logger.info(f"    δ-hyperbolicity: δ={max_delta:.4f}, "
                f"δ/diameter={delta_rel:.4f}, "
                f"δ_median={np.median(delta_values):.4f}")

    return max_delta, delta_rel, delta_values, stats


# ==============================================================================
# Plotting: Ricci curvature histogram comparison
# ==============================================================================
def plot_ricci_comparison(curvature_dict, out_dir, dpi=200):
    """Overlaid histogram of Ricci curvatures for CNN vs SAE."""
    colors = {"CNN": "#3A7EBF", "SAE": "#E8833A",
              "CNN (stage5_mid)": "#88BEDC", "CNN (stage5_out)": "#3A7EBF",
              "CNN (refine_out)": "#1B4876"}

    fig, ax = plt.subplots(figsize=(8, 5))

    for source_label, curvatures in curvature_dict.items():
        color = colors.get(source_label, "#999999")
        # Fallback color for labeled sources
        for key in colors:
            if key in source_label:
                color = colors[key]
                break

        ax.hist(curvatures, bins=80, alpha=0.45, color=color,
                edgecolor="none", label=source_label, density=True)
        # KDE overlay
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(curvatures, bw_method=0.15)
        x_range = np.linspace(curvatures.min() - 0.1,
                              curvatures.max() + 0.1, 300)
        ax.plot(x_range, kde(x_range), color=color, linewidth=2)

    ax.axvline(0, color="red", linestyle="--", linewidth=1.2, alpha=0.7,
               label="κ = 0 (flat)")
    ax.set_xlabel("Ollivier-Ricci Curvature (κ)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Edge Curvature Distribution", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.85)
    ax.grid(True, alpha=0.12, axis="y")
    sns.despine()
    fig.tight_layout()

    fname = "ricci_curvature_comparison"
    for ext in [".png", ".svg"]:
        fig.savefig(os.path.join(out_dir, fname + ext),
                    dpi=dpi, bbox_inches="tight")
    logger.info(f"  Saved: {fname}.png/.svg")
    if _IN_COLAB:
        plt.show()
    plt.close(fig)


def plot_ricci_per_class(class_curvatures, source_label, out_dir, dpi=200):
    """Per-class curvature violin plot."""
    classes = sorted(class_curvatures.keys())
    data = [class_curvatures[c] for c in classes]
    class_colors = {
        "Control": "#55A868", "SNCA": "#C44E52",
        "GBA": "#8172B2", "LRRK2": "#CCB974",
    }

    fig, ax = plt.subplots(figsize=(6, 5))

    parts = ax.violinplot(data, positions=range(len(classes)),
                          showmeans=True, showmedians=True,
                          showextrema=False)
    for i, pc in enumerate(parts["bodies"]):
        c = class_colors.get(classes[i], "#999999")
        pc.set_facecolor(c)
        pc.set_alpha(0.5)
    parts["cmeans"].set_color("black")
    parts["cmedians"].set_color("red")

    ax.axhline(0, color="red", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, fontsize=11, fontweight="bold")
    ax.set_ylabel("Ollivier-Ricci κ", fontsize=12)
    ax.set_title(f"Intra-class Curvature — {source_label}",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.15, axis="y")
    sns.despine()
    fig.tight_layout()

    safe = source_label.lower().replace(" ", "_").replace("(", "").replace(")", "")
    fname = f"ricci_per_class_{safe}"
    for ext in [".png", ".svg"]:
        fig.savefig(os.path.join(out_dir, fname + ext),
                    dpi=dpi, bbox_inches="tight")
    if _IN_COLAB:
        plt.show()
    plt.close(fig)


def plot_delta_comparison(delta_dict, out_dir, dpi=200):
    """Bar chart comparing δ-hyperbolicity across sources."""
    sources = list(delta_dict.keys())
    colors_list = ["#3A7EBF", "#E8833A", "#88BEDC", "#1B4876"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel A: δ (absolute)
    deltas = [delta_dict[s]["delta"] for s in sources]
    bars = axes[0].bar(range(len(sources)), deltas,
                       color=[colors_list[i % len(colors_list)]
                              for i in range(len(sources))],
                       alpha=0.8, edgecolor="white")
    for bar, v in zip(bars, deltas):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.001,
                     f"{v:.4f}", ha="center", va="bottom",
                     fontsize=9, fontweight="bold")
    axes[0].set_xticks(range(len(sources)))
    axes[0].set_xticklabels(sources, fontsize=9, rotation=15, ha="right")
    axes[0].set_ylabel("δ", fontsize=12)
    axes[0].set_title("δ-Hyperbolicity", fontsize=13, fontweight="bold")
    axes[0].grid(True, alpha=0.15, axis="y")

    # Panel B: δ/diameter (normalized)
    delta_rels = [delta_dict[s]["delta_rel"] for s in sources]
    bars = axes[1].bar(range(len(sources)), delta_rels,
                       color=[colors_list[i % len(colors_list)]
                              for i in range(len(sources))],
                       alpha=0.8, edgecolor="white")
    for bar, v in zip(bars, delta_rels):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.001,
                     f"{v:.4f}", ha="center", va="bottom",
                     fontsize=9, fontweight="bold")
    axes[1].set_xticks(range(len(sources)))
    axes[1].set_xticklabels(sources, fontsize=9, rotation=15, ha="right")
    axes[1].set_ylabel("δ / diameter", fontsize=12)
    axes[1].set_title("Normalized δ-Hyperbolicity",
                      fontsize=13, fontweight="bold")
    axes[1].grid(True, alpha=0.15, axis="y")

    for ax in axes:
        sns.despine(ax=ax)
    fig.tight_layout()

    fname = "delta_hyperbolicity_comparison"
    for ext in [".png", ".svg"]:
        fig.savefig(os.path.join(out_dir, fname + ext),
                    dpi=dpi, bbox_inches="tight")
    logger.info(f"  Saved: {fname}.png/.svg")
    if _IN_COLAB:
        plt.show()
    plt.close(fig)


# ==============================================================================
# LOO KNN confidence filter — correctly classified samples only
# ==============================================================================
def filter_correct_by_loo_knn(X, superclasses, k=10, weights="inv_sq",
                              seed=42):
    """Leave-one-out KNN: classify each sample using the rest, keep correct.

    Uses the same inverse-distance-squared weighting as knn_fewshot_eval.py.

    Parameters
    ----------
    X : np.ndarray (N, d)
    superclasses : list[str] — class labels
    k : int — number of neighbors
    weights : str — 'inv_sq', 'distance', 'uniform'

    Returns
    -------
    correct_mask : np.ndarray (N,) bool
    accuracy : float
    per_class_acc : dict — {class_name: accuracy}
    """
    from sklearn.neighbors import NearestNeighbors

    # Map labels to integers
    y = np.array([CLASS_TO_LABEL.get(s, -1) for s in superclasses])
    valid = y >= 0
    X_v, y_v = X[valid], y[valid]
    n = len(X_v)
    k_actual = min(k, n - 1)

    logger.info(f"  LOO KNN filter: n={n}, k={k_actual}, weights={weights}")

    # Find k+1 neighbors (first one is self)
    nn = NearestNeighbors(n_neighbors=k_actual + 1, metric="euclidean",
                          n_jobs=-1)
    nn.fit(X_v)
    distances, indices = nn.kneighbors(X_v)

    # Exclude self (index 0)
    distances = distances[:, 1:]  # (n, k)
    indices = indices[:, 1:]      # (n, k)

    # Weighted vote
    y_pred = _weighted_vote(distances, indices, y_v, k_actual,
                            weights, NUM_CLASSES)

    correct = y_pred == y_v
    accuracy = float(correct.mean())

    per_class_acc = {}
    for c in range(NUM_CLASSES):
        mask = y_v == c
        if mask.sum() > 0:
            per_class_acc[CLASS_NAMES[c]] = float(correct[mask].mean())
        else:
            per_class_acc[CLASS_NAMES[c]] = 0.0

    # Map back to original indices
    correct_mask = np.zeros(len(X), dtype=bool)
    valid_indices = np.where(valid)[0]
    correct_mask[valid_indices[correct]] = True

    logger.info(f"    LOO KNN accuracy: {accuracy:.4f}")
    logger.info(f"    Correct: {correct_mask.sum()}/{len(X)} "
                f"({correct_mask.mean():.1%})")
    for cn, acc in per_class_acc.items():
        logger.info(f"      {cn:>8s}: {acc:.4f}")

    return correct_mask, accuracy, per_class_acc


# ==============================================================================
# Apoptosis local std: cosine KNN vs DPT KNN
# ==============================================================================
def compute_local_apop_std_knn_vs_dpt(
        X, uids, superclasses, apoptosis_csv, k_list,
        pca_dim=0, n_neighbors_dpt=15, seed=42):
    """Compare apoptosis rate local std under cosine KNN vs DPT KNN.

    For each sample with a valid apoptosis rate:
      - cosine KNN: top-K neighbors by cosine distance (argsort of 1-X_norm@X_norm.T)
      - DPT KNN:    top-K neighbors by dense DPT distance

    For each K, report mean±std and median of per-point local apoptosis std.
    A LOWER local std = neighbors share more similar apoptosis → better structure.

    Parameters
    ----------
    X : np.ndarray (N, d)       — feature matrix (already L2-normed if desired)
    uids : list[str]            — per-sample image UIDs
    superclasses : list[str]    — per-sample class labels
    apoptosis_csv : str         — path to per-image apoptosis CSV
    k_list : list[int]          — K values to sweep
    pca_dim : int               — PCA dims before DPT (0 = skip)
    n_neighbors_dpt : int       — k for DPT graph construction
    seed : int

    Returns
    -------
    results : dict — structured results per metric_type and k
    """
    from apoptosis_prediction.local_knn_std import load_cache as _lc  # noqa: unused
    from kendall_correlation_coefficient.dpt_kendall import load_and_match_apoptosis

    logger.info(f"\n  ── Apoptosis Local Std: cosine vs DPT ──")
    logger.info(f"  Loading apoptosis CSV: {apoptosis_csv}")

    apop = load_and_match_apoptosis(apoptosis_csv, list(uids))
    valid_mask = np.isfinite(apop)
    n_valid = int(valid_mask.sum())
    logger.info(f"  Valid apoptosis samples: {n_valid}/{len(apop)}")

    if n_valid < 20:
        logger.warning("  Too few valid apoptosis samples — skipping.")
        return {}

    X_v = X[valid_mask]
    apop_v = apop[valid_mask]
    sc_v = np.array(superclasses)[valid_mask]
    n = len(X_v)

    # ── 1. Cosine distance matrix (N_valid × N_valid) ──
    logger.info(f"  Building cosine distance matrix ({n}×{n})...")
    X_norm = X_v / np.linalg.norm(X_v, axis=1, keepdims=True).clip(min=1e-12)
    if _HAS_CUDA and n <= 15000:
        import torch as _t
        Xn_t = _t.from_numpy(X_norm.astype(np.float32)).to(_DEVICE)
        cos_full = (1.0 - (_t.mm(Xn_t, Xn_t.T)).cpu().numpy()).astype(np.float64)
        del Xn_t
        torch.cuda.empty_cache()
    else:
        cos_full = 1.0 - X_norm @ X_norm.T
    np.fill_diagonal(cos_full, np.inf)          # exclude self

    # ── 2. DPT distance matrix ──
    logger.info(f"  Computing dense DPT ({n}×{n})...")
    dpt_matrix, _ = compute_dense_dpt_distances(
        X_v, n_neighbors=n_neighbors_dpt, pca_dim=pca_dim, seed=seed)
    np.fill_diagonal(dpt_matrix, np.inf)        # exclude self

    # ── 3. Sweep over K ──
    results = {}
    max_k = min(max(k_list), n - 1)

    # Pre-sort once
    cos_sorted = np.argsort(cos_full, axis=1)   # (n, n-1)
    dpt_sorted = np.argsort(dpt_matrix, axis=1) # (n, n-1)

    logger.info(f"  K sweep: {k_list}")
    logger.info(f"  {'Metric':12s}  {'K':>4s}  "
                f"{'mean_std':>10s}  {'std_std':>10s}  {'median_std':>10s}")
    logger.info(f"  {'─'*55}")

    for metric_name, sorted_idx in [("cosine", cos_sorted), ("DPT", dpt_sorted)]:
        results[metric_name] = {}
        for k in k_list:
            k_use = min(k, n - 1)
            neighbor_idx = sorted_idx[:, :k_use]           # (n, k_use)
            neighbor_apop = apop_v[neighbor_idx]           # (n, k_use)
            local_stds = np.std(neighbor_apop, axis=1)     # (n,)

            m_mean   = float(np.mean(local_stds))
            m_std    = float(np.std(local_stds))
            m_median = float(np.median(local_stds))

            # Per-class breakdown
            per_class = {}
            for cls in sorted(set(sc_v)):
                cls_m = sc_v == cls
                if cls_m.sum() >= 2:
                    per_class[cls] = {
                        "mean":   float(np.mean(local_stds[cls_m])),
                        "median": float(np.median(local_stds[cls_m])),
                        "std":    float(np.std(local_stds[cls_m])),
                        "n":      int(cls_m.sum()),
                    }

            results[metric_name][str(k)] = {
                "mean_std":   m_mean,
                "std_std":    m_std,
                "median_std": m_median,
                "n":          n,
                "per_class":  per_class,
            }
            logger.info(f"  {metric_name:12s}  {k:>4d}  "
                        f"{m_mean:10.5f}  {m_std:10.5f}  {m_median:10.5f}")

    # ── 4. Δ summary (DPT – cosine, negative = DPT has tighter neighborhoods) ──
    logger.info(f"  {'─'*55}")
    logger.info(f"  Δ (DPT - cosine):")
    for k in k_list:
        ks = str(k)
        if ks in results["cosine"] and ks in results["DPT"]:
            delta_mean   = results["DPT"][ks]["mean_std"]   - results["cosine"][ks]["mean_std"]
            delta_median = results["DPT"][ks]["median_std"] - results["cosine"][ks]["median_std"]
            results[ks + "_delta"] = {"mean": delta_mean, "median": delta_median}
            sign = "↓ DPT tighter" if delta_mean < 0 else "↑ cosine tighter"
            logger.info(f"    k={k:>2d}: Δmean={delta_mean:+.5f}  "
                        f"Δmedian={delta_median:+.5f}  ({sign})")

    return results


def plot_apop_std_comparison(results_by_source, k_list, out_dir, dpi=200):
    """Line plot: mean local apoptosis std vs K for cosine and DPT, per source."""
    if not results_by_source:
        return

    colors = {
        "CNN cosine": "#3A7EBF", "CNN DPT": "#1B4876",
        "SAE cosine": "#E8833A", "SAE DPT": "#8B3A0A",
    }
    lss = {"cosine": "--", "DPT": "-"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for stat_key, ax, stat_label in [
            ("mean_std", axes[0], "Mean local apoptosis std"),
            ("median_std", axes[1], "Median local apoptosis std")]:

        for src_label, res in results_by_source.items():
            for metric in ["cosine", "DPT"]:
                if metric not in res:
                    continue
                ys = [res[metric].get(str(k), {}).get(stat_key, np.nan)
                      for k in k_list]
                key = f"{src_label} {metric}"
                color = colors.get(key, "#999999")
                ls = lss.get(metric, "-")
                ax.plot(k_list, ys, ls, marker="o", color=color,
                        linewidth=2, markersize=6,
                        label=key, alpha=0.9)

        ax.set_xlabel("K (neighbors)", fontsize=11)
        ax.set_ylabel(stat_label, fontsize=11)
        ax.set_title(f"Local Apoptosis Std\n({stat_label})",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.15)
        sns.despine(ax=ax)

    fig.suptitle("Cosine KNN vs DPT KNN — Local Apoptosis Std",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fname = "apop_std_cosine_vs_dpt"
    for ext in [".png", ".svg"]:
        fig.savefig(os.path.join(out_dir, fname + ext),
                    dpi=dpi, bbox_inches="tight")
    logger.info(f"  Saved: {fname}.png/.svg")
    if _IN_COLAB:
        plt.show()
    plt.close(fig)


# ==============================================================================
# Custom Dense DPT — use ALL eigenvectors via numpy.linalg.eigh
# ==============================================================================
def compute_dense_dpt_distances(X, n_neighbors=15, pca_dim=0, seed=42):
    """Compute DPT distances using dense eigendecomposition (all eigvecs).

    scanpy uses scipy.sparse.linalg.eigsh (ARPACK), which is designed for
    finding a FEW eigenvalues of sparse matrices. Requesting N-2 eigenvalues
    makes it extremely slow or crashes.

    This function builds the transition matrix from the KNN graph with
    adaptive Gaussian kernel, then does a FULL dense eigendecomposition
    via numpy.linalg.eigh.

    DPT distance: d²(x,y) = Σ_l [1/(1-λ_l)]² [ψ_l(x)-ψ_l(y)]²

    Parameters
    ----------
    X : np.ndarray (N, d) — features (raw or L2-normed)
    n_neighbors : int — K for KNN graph
    pca_dim : int — PCA dims before KNN (0 = skip PCA)
    seed : int

    Returns
    -------
    dpt_matrix : np.ndarray (N, N) — pairwise DPT distances
    eigenvalues : np.ndarray — all eigenvalues of transition matrix
    """
    import scanpy as sc
    import anndata

    n = len(X)
    logger.info(f"    Dense DPT: N={n}, k={n_neighbors}, pca_dim={pca_dim}")

    # ── 1. Build KNN graph via scanpy (adaptive Gaussian kernel) ──
    adata = anndata.AnnData(X.astype(np.float32))
    if pca_dim > 0 and X.shape[1] > pca_dim:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=pca_dim, random_state=seed)
        adata.obsm["X_pca"] = pca.fit_transform(X).astype(np.float32)
        n_pcs = pca_dim
    else:
        adata.obsm["X_pca"] = X.astype(np.float32)
        n_pcs = X.shape[1]

    k_nn = min(n_neighbors, n - 1)
    sc.pp.neighbors(adata, n_neighbors=k_nn, n_pcs=n_pcs,
                    use_rep="X_pca", random_state=seed)

    # ── 2. Extract transition matrix T (row-stochastic) ──
    # scanpy stores connectivities (symmetric, weighted adjacency)
    # and distances. We symmetrize and normalize to get T.
    W = adata.obsp["connectivities"].toarray().astype(np.float64)  # (N, N)
    # Row-normalize → transition matrix
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    T = W / row_sums  # row-stochastic transition matrix

    # ── 3. Symmetrize for eigendecomposition ──
    # T_sym = D^{1/2} T D^{-1/2} where D = diag(stationary distribution)
    # For row-stochastic T with symmetric W: stationary π ∝ row_sums
    pi = row_sums.ravel()
    pi = pi / pi.sum()
    sqrt_pi = np.sqrt(pi)
    inv_sqrt_pi = 1.0 / np.where(sqrt_pi == 0, 1e-12, sqrt_pi)

    T_sym = (sqrt_pi[:, None]) * T * (inv_sqrt_pi[None, :])
    # Make exactly symmetric (numerical)
    T_sym = 0.5 * (T_sym + T_sym.T)

    # ── 4. Dense eigendecomposition (ALL eigenvectors) ──
    if _HAS_CUDA:
        logger.info(f"    Dense eigh on {n}×{n} (GPU torch)...")
        T_sym_t = torch.from_numpy(T_sym).to(_DEVICE)
        eigenvalues_t, eigenvectors_t = torch.linalg.eigh(T_sym_t)
        # Reverse to descending
        eigenvalues = eigenvalues_t.flip(0).cpu().numpy()
        eigenvectors = eigenvectors_t.flip(1).cpu().numpy()
        del T_sym_t, eigenvalues_t, eigenvectors_t
        torch.cuda.empty_cache()
    else:
        logger.info(f"    Dense eigh on {n}×{n} (CPU numpy)...")
        eigenvalues, eigenvectors = np.linalg.eigh(T_sym)
        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]

    # Convert back to right eigenvectors of T: ψ_l = D^{-1/2} v_l
    right_evecs = inv_sqrt_pi[:, None] * eigenvectors  # (N, N)

    # ── 5. Compute DPT distance matrix ──
    # Skip first eigenvector (trivial, λ₀=1)
    # d²(x,y) = Σ_{l≥1} [1/(1-λ_l)]² [ψ_l(x)-ψ_l(y)]²
    evals_nontrivial = eigenvalues[1:]
    evecs_nontrivial = right_evecs[:, 1:]

    # Filter: only use λ < 1
    valid_mask = evals_nontrivial < 1.0 - 1e-10
    evals_use = evals_nontrivial[valid_mask]
    evecs_use = evecs_nontrivial[:, valid_mask]

    # Weights: 1/(1-λ)
    weights = 1.0 / (1.0 - evals_use)

    # DPT distance matrix — GPU if available
    if _HAS_CUDA:
        wc = torch.from_numpy(
            (evecs_use * weights[None, :]).astype(np.float32)).to(_DEVICE)
        dpt_matrix_t = torch.cdist(wc, wc)  # (N, N) Euclidean in weighted space
        dpt_matrix = dpt_matrix_t.cpu().numpy().astype(np.float64)
        del wc, dpt_matrix_t
        torch.cuda.empty_cache()
    else:
        weighted_coords = evecs_use * weights[None, :]
        sq_norms = (weighted_coords ** 2).sum(axis=1)
        dpt_sq = sq_norms[:, None] + sq_norms[None, :] - 2 * weighted_coords @ weighted_coords.T
        dpt_matrix = np.sqrt(np.maximum(dpt_sq, 0.0))

    n_used = valid_mask.sum()
    logger.info(f"    Dense DPT: used {n_used}/{len(evals_nontrivial)} eigenvectors")
    logger.info(f"    Top eigenvalues: {eigenvalues[:5]}")

    del adata
    return dpt_matrix, eigenvalues


# ==============================================================================
# DPT vs Cosine Rank Correlation — Isometry test
# ==============================================================================
def compute_rank_correlation(X, superclasses, n_anchors=300,
                             pca_dim=300, n_neighbors=15, seed=42,
                             return_full_matrices=False):
    """Compare DPT geodesic distance ranking vs cosine similarity ranking.

    Parameters
    ----------
    return_full_matrices : bool
        If True, store the full N×N dpt_matrix and cos_full in the result dict
        under keys '_dpt_matrix' and '_cos_full' (used by save_pairwise_distances_for_gam).
        These are large (N² float32) — only set when --save_pairwise is requested.
    """
    import scanpy as sc
    import anndata
    from scipy.stats import spearmanr, kendalltau, pearsonr

    n = len(X)
    rng = np.random.RandomState(seed)
    n_anchors = min(n_anchors, n)
    anchor_idx = rng.choice(n, size=n_anchors, replace=False)

    logger.info(f"  Rank correlation: {n_anchors} anchors, n={n}")

    # ── 1. DPT distances (dense eigendecomposition — all eigenvectors) ──
    logger.info(f"    Computing dense DPT...")
    dpt_matrix, dpt_evals = compute_dense_dpt_distances(
        X, n_neighbors=n_neighbors, pca_dim=pca_dim, seed=seed)

    # ── 2. Cosine distance ──
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-12)

    # ── 3. Per-anchor rank correlation ──
    sc_arr = np.array(superclasses)
    spearman_all = []
    kendall_all = []
    pearson_all = []
    per_class_spearman = {cn: [] for cn in sorted(set(sc_arr))}
    per_class_kendall = {cn: [] for cn in sorted(set(sc_arr))}
    per_class_pearson = {cn: [] for cn in sorted(set(sc_arr))}

    for i, ai in enumerate(anchor_idx):
        dpt_dist = dpt_matrix[ai]  # (n,)

        # Handle NaN/inf
        valid = np.isfinite(dpt_dist)
        valid[ai] = False  # exclude self
        if valid.sum() < 10:
            continue

        # Cosine distance from anchor to all
        cos_dist = 1.0 - X_norm @ X_norm[ai]

        dpt_valid = dpt_dist[valid]
        cos_valid = cos_dist[valid]

        # Rank correlation & Pearson (linear)
        rho, _ = spearmanr(dpt_valid, cos_valid)
        tau, _ = kendalltau(dpt_valid, cos_valid)
        pr, _ = pearsonr(dpt_valid, cos_valid)

        if np.isfinite(rho):
            spearman_all.append(rho)
            anchor_class = sc_arr[ai]
            per_class_spearman[anchor_class].append(rho)
        if np.isfinite(tau):
            kendall_all.append(tau)
            anchor_class = sc_arr[ai]
            per_class_kendall[anchor_class].append(tau)
        if np.isfinite(pr):
            pearson_all.append(pr)
            anchor_class = sc_arr[ai]
            per_class_pearson[anchor_class].append(pr)

    if len(spearman_all) == 0:
        logger.warning("    No valid anchor correlations computed!")
        return {"error": "no valid anchors"}

    # ── 4. Kruskal Stress-1 (Nash embedding quality) ──
    # stress = sqrt( Σ(d_cos - d̂_cos)² / Σ d_dpt² )
    # where d̂_cos = linear fit of cos_dist from dpt_dist
    # Lower stress = better distance preservation (isometry)
    logger.info(f"    Computing Kruskal stress & NPR...")
    cos_full = 1.0 - X_norm @ X_norm.T  # (N, N) cosine distance matrix

    # Sample pairs for stress (full N² is expensive)
    rng2 = np.random.RandomState(seed + 1)
    n_stress_pairs = min(50000, n * (n - 1) // 2)
    triu_r, triu_c = np.triu_indices(n, k=1)
    if len(triu_r) > n_stress_pairs:
        pair_idx = rng2.choice(len(triu_r), size=n_stress_pairs, replace=False)
        pi_r, pi_c = triu_r[pair_idx], triu_c[pair_idx]
    else:
        pi_r, pi_c = triu_r, triu_c

    dpt_pairs = dpt_matrix[pi_r, pi_c]
    cos_pairs = cos_full[pi_r, pi_c]
    
    cls_r = sc_arr[pi_r]
    cls_c = sc_arr[pi_c]
    # Label pair with class name if they match, else "Inter-class"
    pair_cls = np.where(cls_r == cls_c, cls_r, "Inter-class")

    # Only use finite pairs
    fin_mask = np.isfinite(dpt_pairs) & np.isfinite(cos_pairs)
    dpt_fin = dpt_pairs[fin_mask]
    cos_fin = cos_pairs[fin_mask]
    pair_cls_fin = pair_cls[fin_mask]

    # Kruskal stress-1: normalized residual after monotonic regression
    # Calculated PER CLASS for intra-class pairs
    per_class_stress = {}
    unique_classes = sorted([c for c in set(sc_arr)])
    
    for cls in unique_classes:
        mask = (pair_cls_fin == cls)
        dpt_cls = dpt_fin[mask]
        cos_cls = cos_fin[mask]
        
        if len(dpt_cls) > 10:
            A = np.vstack([dpt_cls, np.ones(len(dpt_cls))]).T
            slope, intercept = np.linalg.lstsq(A, cos_cls, rcond=None)[0]
            cos_pred = slope * dpt_cls + intercept
            residuals = cos_cls - cos_pred
            stress = float(np.sqrt((residuals ** 2).sum() / (cos_cls ** 2).sum()))
            per_class_stress[cls] = stress

    if per_class_stress:
        kruskal_stress = float(np.mean(list(per_class_stress.values())))
    else:
        kruskal_stress = float('nan')

    logger.info(f"    Kruskal stress-1 (mean): {kruskal_stress:.4f}")
    for cls, stress in per_class_stress.items():
        logger.info(f"      {cls:>8s}: {stress:.4f}")

    # ── 5. KNN Neighborhood Preservation Ratio (Whitney quality) ──
    # For each point, do top-K neighbors match between cosine and DPT space?
    max_k_npr = min(15, n - 1)
    
    # Pre-compute up to max_k
    cos_knn_all = np.argsort(cos_full, axis=1)[:, 1:max_k_npr+1]  # (N, max_k)
    dpt_knn_all = np.argsort(dpt_matrix, axis=1)[:, 1:max_k_npr+1]

    npr_k_list = [k for k in [5, 10, 15] if k <= max_k_npr]
    npr_all = {}
    
    logger.info(f"    NPR multi-k results:")
    for k in npr_k_list:
        overlap = np.array([
            len(np.intersect1d(cos_knn_all[i, :k], dpt_knn_all[i, :k])) / k
            for i in range(n)
        ])
        
        pk_npr = {}
        for cn in sorted(set(sc_arr)):
            cn_mask = sc_arr == cn
            if cn_mask.sum() > 0:
                pk_npr[cn] = float(overlap[cn_mask].mean())
                
        mean_v = float(overlap.mean())
        std_v = float(overlap.std())
        
        npr_all[str(k)] = {
            "mean": mean_v,
            "std": std_v,
            "per_class": pk_npr,
            "_npr_vals": overlap
        }
        logger.info(f"      k={k:<2d}: {mean_v:.4f} ± {std_v:.4f}")

    # Set k=10 as default for summary plots
    default_k = str(min(10, max_k_npr))
    if default_k in npr_all:
        npr_mean = npr_all[default_k]["mean"]
        npr_std = npr_all[default_k]["std"]
        per_class_npr = npr_all[default_k]["per_class"]
        overlap_per_point = npr_all[default_k]["_npr_vals"]
        k_npr = int(default_k)
    else:
        last_k = str(npr_k_list[-1])
        npr_mean = npr_all[last_k]["mean"]
        npr_std = npr_all[last_k]["std"]
        per_class_npr = npr_all[last_k]["per_class"]
        overlap_per_point = npr_all[last_k]["_npr_vals"]
        k_npr = int(last_k)

    results = {
        "spearman_mean": float(np.mean(spearman_all)),
        "spearman_std": float(np.std(spearman_all)),
        "spearman_median": float(np.median(spearman_all)),
        "kendall_mean": float(np.mean(kendall_all)),
        "kendall_std": float(np.std(kendall_all)),
        "kendall_median": float(np.median(kendall_all)),
        "pearson_mean": float(np.mean(pearson_all)) if pearson_all else float('nan'),
        "pearson_std": float(np.std(pearson_all)) if pearson_all else float('nan'),
        "n_anchors_valid": len(spearman_all),
        "kruskal_stress": kruskal_stress,
        "per_class_stress": per_class_stress,
        "npr_mean": npr_mean,
        "npr_std": npr_std,
        "npr_k": k_npr,
        "per_class_npr": per_class_npr,
        "npr_all": {
            k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
            for k, v in npr_all.items()
        },
        "per_class_spearman": {
            cn: {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                 "n": len(vals)}
            for cn, vals in per_class_spearman.items() if len(vals) > 0
        },
        "per_class_kendall": {
            cn: {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                 "n": len(vals)}
            for cn, vals in per_class_kendall.items() if len(vals) > 0
        },
        "per_class_pearson": {
            cn: {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                 "n": len(vals)}
            for cn, vals in per_class_pearson.items() if len(vals) > 0
        },
        "_dpt_pairs": dpt_fin,
        "_cos_pairs": cos_fin,
        "_cls_pairs": pair_cls_fin,
        "_npr_vals": overlap_per_point,
        "_npr_all_vals": {k: v["_npr_vals"] for k, v in npr_all.items()}
    }

    # ── Full N×N matrices for per-class exhaustive pairwise dump ──
    if return_full_matrices:
        results["_dpt_matrix"]  = dpt_matrix   # (N, N) float64/32
        results["_cos_full"]    = cos_full      # (N, N) float32
        results["_sc_arr"]      = sc_arr        # (N,) str
        logger.info(f"    Storing full N×N matrices for pairwise dump "
                    f"(dpt_matrix: {dpt_matrix.nbytes/1e6:.0f} MB, "
                    f"cos_full: {cos_full.nbytes/1e6:.0f} MB)")

    logger.info(f"    Spearman ρ: {results['spearman_mean']:.4f} "
                f"± {results['spearman_std']:.4f} "
                f"(median={results['spearman_median']:.4f})")
    logger.info(f"    Kendall τ:  {results['kendall_mean']:.4f} "
                f"± {results['kendall_std']:.4f}")
    if not np.isnan(results['pearson_mean']):
        logger.info(f"    Pearson r:  {results['pearson_mean']:.4f} "
                    f"± {results['pearson_std']:.4f}")
                    
    logger.info("    Per-class correlations (ρ / τ / r):")
    for cn in results["per_class_spearman"].keys():
        s_rho = results["per_class_spearman"][cn]
        s_tau = results.get("per_class_kendall", {}).get(cn, {"mean": float("nan"), "std": float("nan")})
        s_pr = results.get("per_class_pearson", {}).get(cn, {"mean": float("nan"), "std": float("nan")})
        logger.info(f"      {cn:>8s}: ρ={s_rho['mean']:.4f}±{s_rho['std']:.2f} | "
                    f"τ={s_tau['mean']:.4f}±{s_tau['std']:.2f} | "
                    f"r={s_pr['mean']:.4f}±{s_pr['std']:.2f} (n={s_rho['n']})")

    return results


# ==============================================================================
# Pairwise distance dump for GAM fitting
# ==============================================================================
def save_pairwise_distances_for_gam(
        corr_dict, sources, out_dir,
        n_inter_sample=500_000):
    """Save ALL per-class pairwise (cosine, DPT) distances using full N×N matrices.

    Strategy
    --------
    Intra-class pairs  : EXHAUSTIVE upper triangle per class
                         (5000 × 4999 / 2 ≈ 12.5M per class → 50M total)
    Inter-class pairs  : random sample (n_inter_sample, default 500k)

    Memory estimate (4 classes × 5000 pts each):
      float32 × 2 arrays × 50M = 400MB uncompressed → ~50-100MB compressed NPZ ✅

    Files per source label
    ----------------------
    pairwise_{safe}_intra_{cls}.npz  — per-class intra NPZ (float32 arrays)
    pairwise_{safe}_inter.npz        — sampled inter-class NPZ
    pairwise_{safe}_combined.npz     — all intra + inter combined
    pairwise_{safe}_head.csv         — first 20,000 rows (quick inspection)
    pairwise_{safe}_summary.json     — overall correlations + bin analysis

    NPZ array keys
    --------------
    cosine_dist     float32 (N_pairs,)
    dpt_dist        float32 (N_pairs,)
    class_i         U32 str (N_pairs,)  — class of point i
    class_j         U32 str (N_pairs,)  — class of point j
    pair_type       U8 str  (N_pairs,)  — 'intra' or 'inter'
    """
    import csv as _csv
    from scipy.stats import spearmanr, kendalltau, pearsonr

    os.makedirs(out_dir, exist_ok=True)

    for label, rc in corr_dict.items():
        if "error" in rc:
            continue

        # ── Check full matrices available ──────────────────────────────
        dpt_matrix = rc.get("_dpt_matrix")
        cos_full   = rc.get("_cos_full")
        sc_arr     = rc.get("_sc_arr")

        if dpt_matrix is None or cos_full is None or sc_arr is None:
            # Fallback to sampled pairs from Kruskal computation
            logger.warning(f"  [{label}] Full matrices not available "
                           f"(--save_pairwise requires --rank_correlation). "
                           f"Falling back to sampled Kruskal pairs.")
            dpt_f = rc.get("_dpt_pairs")
            cos_f = rc.get("_cos_pairs")
            cls_f = rc.get("_cls_pairs")
            if dpt_f is None:
                continue
            _save_npz_and_csv(label, out_dir,
                              np.asarray(cos_f, dtype=np.float32),
                              np.asarray(dpt_f, dtype=np.float32),
                              np.asarray(cls_f, dtype=str),
                              np.asarray(cls_f, dtype=str))
            continue

        dpt_matrix = np.asarray(dpt_matrix, dtype=np.float32)
        cos_full   = np.asarray(cos_full,   dtype=np.float32)
        sc_arr     = np.asarray(sc_arr,     dtype=str)
        n          = len(sc_arr)

        safe = label.replace(" ", "_").replace("(", "").replace(")", "") \
                    .replace("/", "_").replace("\\", "_")

        unique_classes = sorted(set(sc_arr.tolist()))
        logger.info(f"\n  [{label}]  n={n}")
        logger.info(f"    Classes: {unique_classes}")

        # ── Intra-class: exhaustive upper triangle per class ──────────
        for cls in unique_classes:
            cls_idx = np.where(sc_arr == cls)[0]
            nc = len(cls_idx)
            n_pairs_cls = nc * (nc - 1) // 2
            logger.info(f"    [{cls}]  n={nc}  → {n_pairs_cls:,} intra-class pairs")

            # Extract sub-matrix
            sub_dpt = dpt_matrix[np.ix_(cls_idx, cls_idx)]
            sub_cos = cos_full  [np.ix_(cls_idx, cls_idx)]

            # Upper triangle (k=1: exclude diagonal)
            ri, ci = np.triu_indices(nc, k=1)
            cos_v = sub_cos[ri, ci].astype(np.float32)
            dpt_v = sub_dpt[ri, ci].astype(np.float32)
            ci_v  = np.full(len(ri), cls, dtype="U32")

            # Per-class NPZ
            cls_safe = cls.replace(" ", "_")
            npz_path = os.path.join(out_dir, f"pairwise_{safe}_intra_{cls_safe}.npz")
            np.savez_compressed(
                npz_path,
                cosine_dist = cos_v,
                dpt_dist    = dpt_v,
                class_i     = ci_v,
                class_j     = ci_v,
                pair_type   = np.full(len(ri), "intra", dtype="U8"),
            )
            logger.info(f"      Saved: {npz_path}  ({cos_v.nbytes/1e6:.0f} MB uncompressed)")

            del sub_dpt, sub_cos, ri, ci, cos_v, dpt_v

        # ── Combined NPZ: 100% Exhaustive Upper Triangle (Inter + Intra) ──
        logger.info("    Extracting FULL N x N upper triangle for combined NPZ (Inter + Intra)...")
        rng = np.random.RandomState(42)
        ri_all, ci_all_idx = np.triu_indices(n, k=1)
        
        cos_all = cos_full[ri_all, ci_all_idx].astype(np.float32)
        dpt_all = dpt_matrix[ri_all, ci_all_idx].astype(np.float32)
        ci_all  = sc_arr[ri_all]
        cj_all  = sc_arr[ci_all_idx]
        type_all = np.where(ci_all == cj_all, "intra", "inter").astype("U8")

        # Shuffle for unbiased head CSV
        perm = rng.permutation(len(cos_all))
        cos_all = cos_all[perm]; dpt_all = dpt_all[perm]
        ci_all  = ci_all[perm];  cj_all  = cj_all[perm]
        type_all = type_all[perm]

        total_pairs = len(cos_all)
        logger.info(f"    Total pairs saved: {total_pairs:,}")

        # Overall correlations (subsample 5M for speed)
        fin = np.isfinite(cos_all) & np.isfinite(dpt_all)
        samp_n = min(5_000_000, fin.sum())
        idx_s  = rng.choice(np.where(fin)[0], size=samp_n, replace=False)
        rho_all = float(spearmanr(cos_all[idx_s], dpt_all[idx_s]).statistic)
        tau_all = float(kendalltau(cos_all[idx_s], dpt_all[idx_s]).statistic)
        pr_all  = float(pearsonr( cos_all[idx_s], dpt_all[idx_s]).statistic)

        logger.info(f"    Overall (subsample {samp_n:,}): "
                    f"Spearman={rho_all:.4f}  "
                    f"Kendall={tau_all:.4f}  "
                    f"Pearson={pr_all:.4f}")

        # Bin analysis (12 bins)
        bins = np.percentile(cos_all[fin], np.linspace(0, 100, 13))
        logger.info(f"    cosine_bin → DPT_mean  Δ(DPT-cos):")
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = fin & (cos_all >= lo) & (cos_all < hi)
            if mask.sum() < 10:
                continue
            d_m = float(np.mean(dpt_all[mask]))
            c_m = float(np.mean(cos_all[mask]))
            logger.info(f"      [{c_m:.4f}] DPT={d_m:.4f}  Δ={d_m-c_m:+.4f}  n={mask.sum():,}")

        # Save combined NPZ
        combined_path = os.path.join(out_dir, f"pairwise_{safe}_combined.npz")
        np.savez_compressed(
            combined_path,
            cosine_dist = cos_all,
            dpt_dist    = dpt_all,
            class_i     = ci_all,
            class_j     = cj_all,
            pair_type   = type_all,
            spearman    = np.array(rho_all),
            kendall     = np.array(tau_all),
            pearson     = np.array(pr_all),
        )
        logger.info(f"    Saved combined NPZ: {combined_path}")

        # Head CSV (first 20k rows, shuffled)
        csv_path = os.path.join(out_dir, f"pairwise_{safe}_head.csv")
        n_head = min(20_000, total_pairs)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(["cosine_dist", "dpt_dist",
                        "class_i", "class_j", "pair_type", "dpt_minus_cosine"])
            for k in range(n_head):
                w.writerow([
                    f"{cos_all[k]:.6f}", f"{dpt_all[k]:.6f}",
                    ci_all[k], cj_all[k], type_all[k],
                    f"{float(dpt_all[k]) - float(cos_all[k]):+.6f}",
                ])
        logger.info(f"    Saved head CSV ({n_head} rows): {csv_path}")

        # Summary JSON
        summ_path = os.path.join(out_dir, f"pairwise_{safe}_summary.json")
        with open(summ_path, "w") as f:
            json.dump({
                "label":        label,
                "n_total":      int(total_pairs),
                "n_intra":      int((type_all == "intra").sum()),
                "n_inter":      int((type_all == "inter").sum()),
                "spearman":     rho_all,
                "kendall":      tau_all,
                "pearson":      pr_all,
                "cosine_range": [float(cos_all.min()), float(cos_all.max())],
                "dpt_range":    [float(dpt_all.min()), float(dpt_all.max())],
                "classes":      unique_classes,
            }, f, indent=2, default=str)
        logger.info(f"    Saved summary JSON: {summ_path}")

        # Free memory
        del dpt_matrix, cos_full, cos_all, dpt_all, ci_all, cj_all


def _save_npz_and_csv(label, out_dir, cos_arr, dpt_arr, ci_arr, cj_arr):
    """Minimal fallback save when full matrices not available."""
    safe = label.replace(" ", "_").replace("(", "").replace(")", "")
    type_arr = np.where(ci_arr == cj_arr, "intra", "inter").astype("U8")
    npz_path = os.path.join(out_dir, f"pairwise_{safe}_sampled.npz")
    np.savez_compressed(npz_path,
                        cosine_dist=cos_arr, dpt_dist=dpt_arr,
                        class_i=ci_arr, class_j=cj_arr, pair_type=type_arr)
    logger.info(f"  Saved (sampled fallback): {npz_path}")



def plot_rank_correlation(corr_dict, out_dir, dpi=200):
    """Bar chart comparing rank correlation across sources."""
    sources = list(corr_dict.keys())
    colors = {"CNN": "#3A7EBF", "SAE": "#E8833A"}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel A: Spearman
    vals = [corr_dict[s].get("spearman_mean", 0) for s in sources]
    errs = [corr_dict[s].get("spearman_std", 0) for s in sources]
    cols = [next((v for k, v in colors.items() if k in s), "#999")
            for s in sources]

    bars = axes[0].bar(range(len(sources)), vals, yerr=errs,
                       color=cols, alpha=0.8, edgecolor="white",
                       capsize=5)
    for bar, v in zip(bars, vals):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.01,
                     f"{v:.3f}", ha="center", va="bottom",
                     fontsize=9, fontweight="bold")
    axes[0].set_xticks(range(len(sources)))
    axes[0].set_xticklabels(sources, fontsize=9, rotation=15, ha="right")
    axes[0].set_ylabel("Spearman ρ", fontsize=12)
    axes[0].set_title("DPT vs Cosine Distance\n(Spearman)",
                      fontsize=13, fontweight="bold")
    axes[0].grid(True, alpha=0.15, axis="y")

    # Panel B: Kendall
    vals = [corr_dict[s].get("kendall_mean", 0) for s in sources]
    errs = [corr_dict[s].get("kendall_std", 0) for s in sources]
    bars = axes[1].bar(range(len(sources)), vals, yerr=errs,
                       color=cols, alpha=0.8, edgecolor="white",
                       capsize=5)
    for bar, v in zip(bars, vals):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.01,
                     f"{v:.3f}", ha="center", va="bottom",
                     fontsize=9, fontweight="bold")
    axes[1].set_xticks(range(len(sources)))
    axes[1].set_xticklabels(sources, fontsize=9, rotation=15, ha="right")
    axes[1].set_ylabel("Kendall τ", fontsize=12)
    axes[1].set_title("DPT vs Cosine Distance\n(Kendall)",
                      fontsize=13, fontweight="bold")
    axes[1].grid(True, alpha=0.15, axis="y")

    for ax in axes:
        sns.despine(ax=ax)
    fig.tight_layout()

    fname = "rank_correlation_comparison"
    for ext in [".png", ".svg"]:
        fig.savefig(os.path.join(out_dir, fname + ext),
                    dpi=dpi, bbox_inches="tight")
    logger.info(f"  Saved: {fname}.png/.svg")
    if _IN_COLAB:
        plt.show()
    plt.close(fig)


def plot_dpt_vs_cosine_scatter(corr_dict, out_dir, dpi=200):
    """Scatter plot: X=cosine distance, Y=DPT geodesic distance per source and per class."""
    sources = [s for s in corr_dict if "_dpt_pairs" in corr_dict[s]]
    if not sources:
        return
    
    # Flatten all classes across sources to find unique intra-classes
    all_classes = set()
    for s in sources:
        all_classes.update(np.unique(corr_dict[s]["_cls_pairs"]))
    classes = [c for c in sorted(all_classes) if c != "Inter-class"]
    
    n_src = len(sources)
    n_cls = len(classes)
    
    fig, axes = plt.subplots(n_src, n_cls, figsize=(4 * n_cls, 4 * n_src), squeeze=False)

    for i, src in enumerate(sources):
        d = corr_dict[src]
        dpt_p = d["_dpt_pairs"]
        cos_p = d["_cos_pairs"]
        cls_p = d["_cls_pairs"]
        
        for j, cls in enumerate(classes):
            ax = axes[i, j]
            mask = (cls_p == cls)
            if not np.any(mask):
                ax.axis('off')
                continue
                
            y_vals = dpt_p[mask]
            x_vals = cos_p[mask]
            
            # Subsample for plotting if > 5000 per class
            if len(y_vals) > 5000:
                sel = np.random.RandomState(42).choice(len(y_vals), 5000, replace=False)
                x_vals, y_vals = x_vals[sel], y_vals[sel]

            ax.scatter(x_vals, y_vals, s=2, alpha=0.3, color="#3A7EBF", rasterized=True)

            # Linear fit line for this specific class
            if len(y_vals) > 10:
                A = np.vstack([y_vals, np.ones(len(y_vals))]).T
                slope, intercept = np.linalg.lstsq(A, x_vals, rcond=None)[0]
                x_range = np.linspace(x_vals.min(), x_vals.max(), 100)
                if abs(slope) > 1e-10:
                    y_fit = (x_range - intercept) / slope
                    ax.plot(x_range, y_fit, color="red", linewidth=1.5, linestyle="--", alpha=0.8)

            ax.set_xlabel("Cosine Distance", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"DPT Distance\n{src}", fontsize=11, fontweight="bold")
            else:
                ax.set_ylabel("DPT Distance", fontsize=10)
            
            stress = d.get("per_class_stress", {}).get(cls, float("nan"))
            npr = d.get("per_class_npr", {}).get(cls, float("nan"))
            rho = d.get("per_class_spearman", {}).get(cls, {}).get("mean", float("nan"))
            
            ax.set_title(f"{cls}\nStress={stress:.3f} NPR={npr:.3f} ρ={rho:.3f}", 
                         fontsize=10, fontweight="bold")
            ax.grid(True, alpha=0.15)
            sns.despine(ax=ax)

    fig.tight_layout()
    fname = "dpt_vs_cosine_scatter_per_class"
    for ext in [".png", ".svg"]:
        fig.savefig(os.path.join(out_dir, fname + ext), dpi=dpi, bbox_inches="tight")
    logger.info(f"  Saved: {fname}.png/.svg")
    if _IN_COLAB:
        plt.show()
    plt.close(fig)


def plot_isometry_summary(corr_dict, out_dir, dpi=200):
    """4-panel bar chart: Spearman, Kendall, Kruskal Stress, NPR."""
    sources = [s for s in corr_dict if "spearman_mean" in corr_dict[s]]
    if not sources:
        return
    colors_map = {"CNN": "#3A7EBF", "SAE": "#E8833A"}
    cols = [next((v for k, v in colors_map.items() if k in s), "#999")
            for s in sources]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    metrics = [
        ("spearman_mean", "spearman_std", "Spearman ρ", True),
        ("kendall_mean", "kendall_std", "Kendall τ", True),
        ("kruskal_stress", None, "Kruskal Stress", False),  # lower=better
        ("npr_mean", "npr_std", "NPR (k=10)", True),
    ]

    for ax, (key, err_key, label, higher_better) in zip(axes, metrics):
        vals = [corr_dict[s].get(key, 0) for s in sources]
        errs = ([corr_dict[s].get(err_key, 0) for s in sources]
                if err_key else None)
        bars = ax.bar(range(len(sources)), vals,
                      yerr=errs, color=cols, alpha=0.8,
                      edgecolor="white", capsize=4)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
        ax.set_xticks(range(len(sources)))
        ax.set_xticklabels(sources, fontsize=8, rotation=20, ha="right")
        ax.set_ylabel(label, fontsize=11)
        arrow = "↑" if higher_better else "↓"
        ax.set_title(f"{label} ({arrow} better)",
                     fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.15, axis="y")
        sns.despine(ax=ax)

    fig.tight_layout()
    fname = "isometry_summary"
    for ext in [".png", ".svg"]:
        fig.savefig(os.path.join(out_dir, fname + ext),
                    dpi=dpi, bbox_inches="tight")
    logger.info(f"  Saved: {fname}.png/.svg")
    if _IN_COLAB:
        plt.show()
    plt.close(fig)


def plot_npr_distribution(corr_dict, out_dir, dpi=200):
    """Plot distribution (histogram & KDE) of NPR values for K=5, 10, 15."""
    sources = [s for s in corr_dict if "_npr_all_vals" in corr_dict[s]]
    if not sources:
        return
        
    colors_map = {"CNN": "#3A7EBF", "SAE": "#E8833A"}
    n_src = len(sources)
    
    fig, axes = plt.subplots(1, n_src, figsize=(5 * n_src, 4.5), squeeze=False)
    
    # Color palette for different Ks
    k_colors = {"5": "#2ca02c", "10": "#d62728", "15": "#9467bd"}
    
    for idx, src in enumerate(sources):
        ax = axes[0, idx]
        d = corr_dict[src]
        all_vals = d.get("_npr_all_vals", {})
        
        # Plot K=10 specifically as the background histogram to anchor the visual
        k_npr = str(d.get("npr_k", 10))
        if k_npr in all_vals:
            col = next((v for k, v in colors_map.items() if k in src), "#999")
            sns.histplot(all_vals[k_npr], bins=11, element="bars", color=col, 
                         stat="density", ax=ax, alpha=0.2, label=f"k={k_npr} (bars)")
            npr_mean = d.get("npr_mean", float("nan"))
            ax.axvline(npr_mean, color="red", linestyle="--", linewidth=2, alpha=0.7)

        # Plot KDE lines for all available K's to show how overlap changes with neighborhood size
        for kval in ["5", "10", "15"]:
            if kval in all_vals:
                vals = all_vals[kval]
                line_col = k_colors.get(kval, "#333333")
                sns.kdeplot(vals, color=line_col, ax=ax, lw=2.5, label=f"k={kval} (KDE)")
        
        ax.set_xlabel("NPR (KNN Overlap Ratio)", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(f"{src}\nKNN Overlap Distribution (K=5,10,15)", fontsize=12, fontweight="bold")
        ax.set_xlim(-0.05, 1.05)
        ax.grid(True, alpha=0.15)
        ax.legend(loc="upper left")
        sns.despine(ax=ax)
        
    fig.tight_layout()
    fname = "npr_distribution"
    for ext in [".png", ".svg"]:
        fig.savefig(os.path.join(out_dir, fname + ext), dpi=dpi, bbox_inches="tight")
    logger.info(f"  Saved: {fname}.png/.svg")
    if _IN_COLAB:
        plt.show()
    plt.close(fig)


# ==============================================================================
# Load and preprocess
# ==============================================================================
def load_and_preprocess(cache_path, dead_threshold, gap_l2_norm,
                        samples_per_class=500, seed=42):
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

    # Subsample (critical for Ricci computation)
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
        logger.info(f"  Subsampled: {len(keep)} total "
                    f"(≤{samples_per_class}/class)")

    logger.info(f"  Features: {X.shape}")
    return X, superclasses, source_label


# ==============================================================================
# Diffusion Map embedding — adaptive kernel → diffusion distance
# ==============================================================================
def compute_diffusion_coords(X, n_comps=15, pca_dim=50, n_neighbors=30,
                              seed=42):
    """Compute diffusion map coordinates.

    Pipeline: [PCA] → sc.pp.neighbors (adaptive kernel) → sc.tl.diffmap
    Returns X_diffmap[:, 1:] (DC0 = trivial constant, excluded).

    Euclidean distance in diffmap space ≈ diffusion distance.

    Parameters
    ----------
    X : np.ndarray (N, d)
    n_comps : int — number of eigenvectors (excluding DC0)
    pca_dim : int — PCA dims before neighbors (0 = skip PCA)
    n_neighbors : int — KNN for scanpy (adaptive kernel)
    seed : int

    Returns
    -------
    X_diff : np.ndarray (N, n_comps) — diffusion coordinates
    evals : np.ndarray — eigenvalues
    """
    import scanpy as sc
    import anndata

    logger.info(f"  Computing diffusion map (n_comps={n_comps}, "
                f"pca_dim={pca_dim}, n_neighbors={n_neighbors})...")

    adata = anndata.AnnData(X.astype(np.float32))

    # PCA
    if pca_dim > 0 and X.shape[1] > pca_dim:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=pca_dim, random_state=seed)
        X_pca = pca.fit_transform(X)
        adata.obsm["X_pca"] = X_pca.astype(np.float32)
        n_pcs = pca_dim
        var_explained = pca.explained_variance_ratio_.sum()
        logger.info(f"    PCA: {X.shape[1]} → {pca_dim} dims "
                    f"({var_explained:.1%} variance)")
    else:
        adata.obsm["X_pca"] = X.astype(np.float32)
        n_pcs = X.shape[1]
        logger.info(f"    PCA: skipped (d={X.shape[1]})")

    # KNN with adaptive kernel
    k_actual = min(n_neighbors, len(X) - 1)
    sc.pp.neighbors(adata, n_neighbors=k_actual, n_pcs=n_pcs,
                    use_rep="X_pca", random_state=seed)

    # Diffusion map — n_comps=0 means "use ALL eigenvectors"
    if n_comps <= 0:
        # ALL eigenvectors via dense eigendecomposition
        # ARPACK (eigsh) is designed for FEW eigenvalues — N-2 is extremely slow
        
        W = adata.obsp["connectivities"].toarray().astype(np.float64)
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        T = W / row_sums
        pi = row_sums.ravel(); pi = pi / pi.sum()
        sqrt_pi = np.sqrt(pi)
        inv_sqrt_pi = 1.0 / np.where(sqrt_pi == 0, 1e-12, sqrt_pi)
        T_sym = (sqrt_pi[:, None]) * T * (inv_sqrt_pi[None, :])
        T_sym = 0.5 * (T_sym + T_sym.T)

        if _HAS_CUDA:
            logger.info(f"    n_comps=0 → dense eigh on {len(X)}x{len(X)} (GPU torch)")
            T_sym_t = torch.from_numpy(T_sym).to(_DEVICE)
            eigenvalues_t, eigenvectors_t = torch.linalg.eigh(T_sym_t)
            eigenvalues = eigenvalues_t.flip(0).cpu().numpy()
            eigenvectors = eigenvectors_t.flip(1).cpu().numpy()
            del T_sym_t, eigenvalues_t, eigenvectors_t
            torch.cuda.empty_cache()
        else:
            logger.info(f"    n_comps=0 → dense eigh on {len(X)}x{len(X)} (CPU numpy)")
            eigenvalues, eigenvectors = np.linalg.eigh(T_sym)
            eigenvalues = eigenvalues[::-1]
            eigenvectors = eigenvectors[:, ::-1]

        right_evecs = inv_sqrt_pi[:, None] * eigenvectors

        # Skip DC0 (trivial eigenvector with λ=1)
        X_diff = right_evecs[:, 1:].astype(np.float32)
        evals = eigenvalues[1:]

        logger.info(f"    Dense diffusion map: {X_diff.shape[1]} components")
        logger.info(f"    Eigenvalues (top 5): {evals[:5]}")
        del adata
        return X_diff, evals
    else:
        n_comps_actual = min(n_comps, len(X) - 2)
        n_comps_actual = max(n_comps_actual, 2)
    sc.tl.diffmap(adata, n_comps=n_comps_actual)

    # X_diffmap: scanpy includes DC0 at index 0 (trivial eigenvector)
    # We exclude it: use columns 1: onwards
    X_diff = adata.obsm["X_diffmap"][:, 1:]  # (N, n_comps)
    evals = adata.uns["diffmap_evals"]

    logger.info(f"    Diffusion map: {X_diff.shape[1]} components")
    logger.info(f"    Eigenvalues (top 5): {evals[:5]}")

    del adata
    return X_diff, evals

# ==============================================================================
# Main
# ==============================================================================
def main():
    args = get_args()
    np.random.seed(args.seed)

    # ── Aggregation mode: skip normal eval ──────────────────────────
    if args.summarize_dir:
        logger.info(f"\n  Mode: AGGREGATE — {args.summarize_dir}")
        out_dir = args.output_dir or args.summarize_dir
        aggregate_apop_std_results(
            base_dir=args.summarize_dir,
            out_dir=out_dir,
            dpi=args.dpi)
        return

    if not args.cnn_cache and not args.sae_cache:
        raise ValueError("At least one of --cnn_cache or --sae_cache required")

    out_dir = args.output_dir or "./geometry_eval_results"
    os.makedirs(out_dir, exist_ok=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"  Intrinsic Geometry Evaluation")
    logger.info(f"  Ollivier-Ricci Curvature & δ-Hyperbolicity")
    logger.info(f"{'='*60}")

    # ── Load features ──
    sources = {}  # label → (X, superclasses)
    if args.cnn_cache:
        logger.info(f"\nLoading CNN cache: {args.cnn_cache}")
        X_cnn, sc_cnn, _ = load_and_preprocess(
            args.cnn_cache, args.dead_threshold, args.gap_l2_norm,
            args.samples_per_class, args.seed)
        cnn_label = f"CNN ({args.label})" if args.label else "CNN"
        sources[cnn_label] = (X_cnn, sc_cnn)

    if args.sae_cache:
        logger.info(f"\nLoading SAE cache: {args.sae_cache}")
        X_sae, sc_sae, _ = load_and_preprocess(
            args.sae_cache, args.dead_threshold, args.gap_l2_norm,
            args.samples_per_class, args.seed)
        sae_label = f"SAE ({args.label})" if args.label else "SAE"
        sources[sae_label] = (X_sae, sc_sae)

    # ── KNN confidence filter → clean subsets (on ORIGINAL features) ──
    sources_clean = {}
    if args.knn_filter:
        logger.info(f"\n  ── KNN Confidence Filter (LOO, k={args.knn_filter_k}) ──")
        logger.info(f"  (on original feature space, before diffusion transform)")
        for label, (X, sc_list) in sources.items():
            logger.info(f"\n  {label}:")
            correct_mask, acc, per_cls = filter_correct_by_loo_knn(
                X, sc_list, k=args.knn_filter_k,
                weights=args.knn_filter_weights, seed=args.seed)
            X_clean = X[correct_mask]
            sc_clean = [sc_list[i] for i in range(len(sc_list))
                        if correct_mask[i]]
            clean_label = f"{label} (clean)"
            sources_clean[clean_label] = (X_clean, sc_clean)
            logger.info(f"    Clean subset: {X_clean.shape}")

    # ── DPT vs Cosine Rank Correlation (on ORIGINAL features) ──
    if args.rank_correlation:
        logger.info(f"\n  ── DPT vs Cosine Rank Correlation ──")
        logger.info(f"  (on original feature space — raw cosine vs dense DPT)")
        corr_dict = {}
        all_sources = dict(sources)
        if sources_clean:
            all_sources.update(sources_clean)
        for label, (X, sc_list) in all_sources.items():
            logger.info(f"\n  {label}: {X.shape}")
            try:
                rc = compute_rank_correlation(
                    X, sc_list,
                    n_anchors=args.n_rank_anchors,
                    pca_dim=args.rank_pca_dim,
                    n_neighbors=args.rank_n_neighbors,
                    seed=args.seed,
                    return_full_matrices=args.save_pairwise)
                corr_dict[label] = rc
            except Exception as e:
                logger.error(f"    Rank correlation failed: {e}")
                corr_dict[label] = {"error": str(e)}

        if len(corr_dict) >= 1:
            plot_rank_correlation(corr_dict, out_dir, args.dpi)
            plot_dpt_vs_cosine_scatter(corr_dict, out_dir, args.dpi)
            plot_isometry_summary(corr_dict, out_dir, args.dpi)
            plot_npr_distribution(corr_dict, out_dir, args.dpi)

        # Save rank correlation JSON (strip numpy arrays for serialization)
        rc_save = {}
        for k, v in corr_dict.items():
            rc_save[k] = {kk: vv for kk, vv in v.items()
                          if not kk.startswith("_")}
        rc_path = os.path.join(out_dir, "rank_correlation.json")
        with open(rc_path, "w") as f:
            json.dump(rc_save, f, indent=2, default=str)
        logger.info(f"  Saved: {rc_path}")

        # ── Pairwise dump for GAM fitting ──────────────────────────────
        if args.save_pairwise:
            logger.info(f"\n  ── Saving pairwise (cosine, DPT) for GAM ──")
            save_pairwise_distances_for_gam(corr_dict, sources, out_dir)

    # ── Diffusion embedding (if requested) — for Ricci/δ geometry ──
    if args.use_diffusion:
        logger.info(f"\n  ── Applying Diffusion Map embedding ──")
        logger.info(f"  (transform to diffusion coords for Ricci/δ geometry)")
        logger.info(f"  PCA={args.pca_dim}, n_comps={args.n_diffmap_comps}, "
                    f"n_neighbors={args.n_neighbors_diffmap}")
        new_sources = {}
        for label, (X, sc_list) in sources.items():
            logger.info(f"\n  {label}: {X.shape}")
            X_diff, evals = compute_diffusion_coords(
                X, n_comps=args.n_diffmap_comps,
                pca_dim=args.pca_dim,
                n_neighbors=args.n_neighbors_diffmap,
                seed=args.seed)
            logger.info(f"    → diffusion coords: {X_diff.shape}")
            new_sources[label] = (X_diff, sc_list)
        sources = new_sources

        # Also transform clean subsets
        if sources_clean:
            new_clean = {}
            for label, (X, sc_list) in sources_clean.items():
                logger.info(f"\n  {label}: {X.shape}")
                X_diff, evals = compute_diffusion_coords(
                    X, n_comps=args.n_diffmap_comps,
                    pca_dim=args.pca_dim,
                    n_neighbors=args.n_neighbors_diffmap,
                    seed=args.seed)
                logger.info(f"    → diffusion coords: {X_diff.shape}")
                new_clean[label] = (X_diff, sc_list)
            sources_clean = new_clean
    else:
        logger.info(f"\n  Distance mode: EUCLIDEAN (raw feature space)")

    # Merge clean sources into main for geometry analysis
    if sources_clean:
        sources.update(sources_clean)

    # ── K sweep loop ──
    k_values = args.k_neighbors  # list

    for k_val in k_values:
        logger.info(f"\n{'#'*60}")
        logger.info(f"  K = {k_val}")
        logger.info(f"{'#'*60}")

        k_suffix = f"_k{k_val}" if len(k_values) > 1 else ""
        k_out_dir = os.path.join(out_dir, f"k_{k_val}") if len(k_values) > 1 else out_dir
        os.makedirs(k_out_dir, exist_ok=True)

        all_results = {}
        curvature_dict = {}  # for comparison plot
        delta_dict = {}      # for comparison plot

        for source_label, (X, superclasses) in sources.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"  Source: {source_label} ({X.shape}), k={k_val}")
            logger.info(f"{'='*60}")

            result = {"source": source_label, "n_samples": X.shape[0],
                      "n_features": X.shape[1], "k": k_val}

            if args.compute_geometry:
                # ══════════════════════════════════════════════════════
                # 1. Ollivier-Ricci Curvature
                # ══════════════════════════════════════════════════════
                logger.info(f"\n  ── Ollivier-Ricci Curvature (k={k_val}) ──")
                try:
                    curvatures, ricci_stats, G_ricci = compute_ricci_curvature(
                        X, k=k_val, alpha=args.ricci_alpha,
                        labels=superclasses)
                    ricci_stats["k"] = k_val
                    result["ricci"] = ricci_stats
                    curvature_dict[source_label] = curvatures

                    # Per-class analysis
                    if args.per_class:
                        logger.info(f"    Per-class intra-class curvatures:")
                        class_curvs = compute_ricci_per_class(G_ricci, superclasses)
                        result["ricci_per_class"] = {
                            cls: {"mean": float(np.mean(c)), "median": float(np.median(c)),
                                  "std": float(np.std(c)), "n_edges": len(c)}
                            for cls, c in class_curvs.items()
                        }
                        plot_ricci_per_class(class_curvs, source_label,
                                            k_out_dir, args.dpi)

                except ImportError:
                    logger.error("GraphRicciCurvature not installed. "
                                 "Install with: pip install GraphRicciCurvature")
                    result["ricci"] = {"error": "GraphRicciCurvature not installed"}
                except Exception as e:
                    logger.error(f"Ricci computation failed: {e}")
                    result["ricci"] = {"error": str(e)}

                # ══════════════════════════════════════════════════════
                # 2. δ-Hyperbolicity (k-independent, run once per source)
                # ══════════════════════════════════════════════════════
                if k_val == k_values[0]:  # only compute δ on first k
                    logger.info(f"\n  ── δ-Hyperbolicity ──")
                    try:
                        delta, delta_rel, delta_vals, delta_stats = \
                            compute_delta_hyperbolicity(
                                X, n_quadruples=args.delta_n_samples, seed=args.seed)
                        result["delta"] = delta_stats
                        delta_dict[source_label] = delta_stats
                    except Exception as e:
                        logger.error(f"δ-hyperbolicity failed: {e}")
                        result["delta"] = {"error": str(e)}
                else:
                    logger.info(
                        f"  (δ-hyperbolicity: k-independent, skipping for k={k_val})")
            else:
                logger.info(f"  [Ricci + δ skipped — pass --compute_geometry to enable]")

            all_results[source_label] = result

        # ══════════════════════════════════════════════════════════════
        # Comparison plots (per k) — only when geometry was computed
        # ══════════════════════════════════════════════════════════════
        if args.compute_geometry:
            if len(curvature_dict) >= 1:
                logger.info(f"\n  Generating Ricci comparison plot (k={k_val})...")
                plot_ricci_comparison(curvature_dict, k_out_dir, args.dpi)

            if len(delta_dict) >= 1 and k_val == k_values[0]:
                logger.info(f"\n  Generating δ-hyperbolicity comparison...")
                plot_delta_comparison(delta_dict, k_out_dir, args.dpi)

        # ── Save JSON ──
        json_results = {}
        for k, v in all_results.items():
            jr = {}
            for k2, v2 in v.items():
                if isinstance(v2, dict):
                    jr[k2] = {k3: v3 for k3, v3 in v2.items()
                              if not isinstance(v3, np.ndarray)}
                else:
                    jr[k2] = v2
            json_results[k] = jr

        json_path = os.path.join(k_out_dir, f"geometry_results{k_suffix}.json")
        with open(json_path, "w") as f:
            json.dump(json_results, f, indent=2, default=str)
        logger.info(f"\nSaved JSON: {json_path}")

        # ── CSV summary ──
        csv_path = os.path.join(k_out_dir, f"geometry_results{k_suffix}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "source", "k", "n_samples", "n_features",
                "ricci_mean", "ricci_median", "ricci_std",
                "frac_positive", "frac_negative",
                "delta", "delta_rel", "diameter",
            ])
            for label, r in all_results.items():
                ricci = r.get("ricci", {})
                delta = r.get("delta", {})
                writer.writerow([
                    label, k_val,
                    r.get("n_samples", ""), r.get("n_features", ""),
                    f"{ricci.get('mean', 'nan'):.4f}" if isinstance(ricci.get('mean'), float) else "",
                    f"{ricci.get('median', 'nan'):.4f}" if isinstance(ricci.get('median'), float) else "",
                    f"{ricci.get('std', 'nan'):.4f}" if isinstance(ricci.get('std'), float) else "",
                    f"{ricci.get('frac_positive', 'nan'):.4f}" if isinstance(ricci.get('frac_positive'), float) else "",
                    f"{ricci.get('frac_negative', 'nan'):.4f}" if isinstance(ricci.get('frac_negative'), float) else "",
                    f"{delta.get('delta', 'nan'):.4f}" if isinstance(delta.get('delta'), float) else "",
                    f"{delta.get('delta_rel', 'nan'):.4f}" if isinstance(delta.get('delta_rel'), float) else "",
                    f"{delta.get('diameter', 'nan'):.4f}" if isinstance(delta.get('diameter'), float) else "",
                ])
        logger.info(f"Saved CSV: {csv_path}")

        # ── Summary ──
        logger.info(f"\n{'='*80}")
        logger.info(f"  SUMMARY — Intrinsic Geometry (k={k_val})")
        logger.info(f"{'='*80}")
        logger.info(f"  {'Source':25s}  {'κ_mean':>8s}  {'κ_med':>8s}  "
                    f"{'%pos':>6s}  {'%neg':>6s}  {'δ':>8s}  {'δ/D':>8s}")
        logger.info(f"  {'─'*75}")

        for label, r in all_results.items():
            ricci = r.get("ricci", {})
            delta = r.get("delta", {})
            d_val = delta.get("delta", "n/a")
            d_rel = delta.get("delta_rel", "n/a")
            d_str = f"{d_val:.4f}" if isinstance(d_val, float) else str(d_val)
            dr_str = f"{d_rel:.4f}" if isinstance(d_rel, float) else str(d_rel)
            if "error" not in ricci:
                logger.info(
                    f"  {label:25s}  "
                    f"{ricci.get('mean', 0):8.4f}  "
                    f"{ricci.get('median', 0):8.4f}  "
                    f"{ricci.get('frac_positive', 0)*100:5.1f}%  "
                    f"{ricci.get('frac_negative', 0)*100:5.1f}%  "
                    f"{d_str:>8s}  "
                    f"{dr_str:>8s}")
            else:
                logger.info(f"  {label:25s}  (errors occurred)")

        logger.info(f"{'='*80}")

    # ══════════════════════════════════════════════════════════════════
    # Apoptosis local std: cosine KNN vs DPT KNN
    # (run once, independent of k_neighbors sweep)
    # ══════════════════════════════════════════════════════════════════
    if args.apoptosis_csv:
        logger.info(f"\n{'='*60}")
        logger.info(f"  Apoptosis Local Std Analysis (cosine vs DPT)")
        logger.info(f"{'='*60}")

        # Use original (pre-diffusion) sources for apoptosis analysis
        base_sources = {}
        if args.cnn_cache:
            base_sources["CNN"] = (X_cnn, sc_cnn)
        if args.sae_cache:
            base_sources["SAE"] = (X_sae, sc_sae)

        apop_results_by_source = {}
        for src_label, (X_src, sc_src) in base_sources.items():
            logger.info(f"\n  [{src_label}] {X_src.shape}")

            # Retrieve matching UIDs from the original load
            # Re-load UIDs from cache (lightweight, just need uid column)
            cache_path = args.cnn_cache if src_label == "CNN" else args.sae_cache
            _data = np.load(cache_path, allow_pickle=True)
            if "uids" in _data:
                _uids = _data["uids"].astype(str)
            else:
                logger.warning(f"  No 'uids' key in {cache_path} — skipping apoptosis analysis for {src_label}")
                continue

            # Match subsampling if samples_per_class was applied
            # (X_src already subsampled in load_and_preprocess)
            # We need the uids in the same subsampled order.
            # Re-derive subsampled uids using same seed & class logic.
            rng_uid = np.random.RandomState(args.seed)
            _sc_orig = np.array([
                __import__('sae_project.step02_logging_utils',
                           fromlist=['SUPERCLASS_MAP']).SUPERCLASS_MAP.get(str(ln), str(ln))
                for ln in _data.get("lines", np.array([]))
            ])
            if args.samples_per_class > 0 and len(_sc_orig) > 0:
                keep_uid = []
                for cls in sorted(np.unique(_sc_orig)):
                    idx = np.where(_sc_orig == cls)[0]
                    n_take = min(args.samples_per_class, len(idx))
                    chosen = rng_uid.choice(idx, size=n_take, replace=False)
                    keep_uid.extend(chosen.tolist())
                keep_uid = sorted(keep_uid)
                _uids_sub = _uids[keep_uid]
            else:
                _uids_sub = _uids

            try:
                apop_res = compute_local_apop_std_knn_vs_dpt(
                    X_src, _uids_sub, sc_src,
                    apoptosis_csv=args.apoptosis_csv,
                    k_list=args.apoptosis_k_neighbors,
                    pca_dim=args.rank_pca_dim,
                    n_neighbors_dpt=args.rank_n_neighbors,
                    seed=args.seed)
                apop_results_by_source[src_label] = apop_res
            except Exception as e:
                logger.error(f"  Apoptosis std analysis failed for {src_label}: {e}")

        # Plot
        if apop_results_by_source:
            plot_apop_std_comparison(
                apop_results_by_source,
                args.apoptosis_k_neighbors,
                out_dir, args.dpi)

            # Save JSON
            apop_json_path = os.path.join(out_dir, "apop_std_cosine_vs_dpt.json")
            with open(apop_json_path, "w") as f:
                json.dump(apop_results_by_source, f, indent=2, default=str)
            logger.info(f"  Saved: {apop_json_path}")

            # CSV summary
            apop_csv_path = os.path.join(out_dir, "apop_std_cosine_vs_dpt.csv")
            with open(apop_csv_path, "w", newline="", encoding="utf-8") as f:
                import csv as _csv
                w = _csv.writer(f)
                w.writerow(["source", "metric", "k",
                            "mean_local_std", "std_local_std", "median_local_std", "n"])
                for src_label, res in apop_results_by_source.items():
                    for metric in ["cosine", "DPT"]:
                        if metric not in res:
                            continue
                        for k in args.apoptosis_k_neighbors:
                            ks = str(k)
                            if ks not in res[metric]:
                                continue
                            row = res[metric][ks]
                            w.writerow([
                                src_label, metric, k,
                                f"{row['mean_std']:.6f}",
                                f"{row['std_std']:.6f}",
                                f"{row['median_std']:.6f}",
                                row['n'],
                            ])
            logger.info(f"  Saved: {apop_csv_path}")

    logger.info(f"\n  Output: {out_dir}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
