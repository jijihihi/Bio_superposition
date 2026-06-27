import argparse
import csv
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from model_train.logging_utils import SUPERCLASS_MAP, get_logger

logger = get_logger("trajectory_utils")

MUTATION_COLORS = {"SNCA": "#d97a7a", "GBA": "#eea363", "LRRK2": "#7ba4db"}
SUPERCLASS_COLORS = {
    "Control": "#a6a6a6",
    "SNCA": "#d97a7a",
    "GBA": "#eea363",
    "LRRK2": "#7ba4db",
}

NORM_CONFIGS = ["none", "log", "std", "log_std"]


def add_trajectory_arguments(p):
    """Add common arguments for all trajectory inference scripts."""
    p.add_argument(
        "--features_cache",
        type=str,
        required=True,
        help="Path to .npz cache (SAE: X_all+usage_ema, or CNN GAP: X_gap)",
    )
    p.add_argument("--cell_death_csv", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="")
    p.add_argument("--dead_threshold", type=float, default=1e-5)

    p.add_argument(
        "--gap_l2_norm",
        action="store_true",
        help="Apply L2 normalization to feature vectors (useful for GAP)",
    )
    p.add_argument(
        "--pre_l2_norm",
        action="store_true",
        help="Apply per-image L2 normalization BEFORE any other processing. Matches old F.normalize(pooled)",
    )

    # Neuron filtering
    p.add_argument(
        "--filter_mode",
        type=str,
        nargs="+",
        default=["none"],
        help="Sequential: 'cv', 'de', 'gini', 'none'. e.g. '--filter_mode cv de'",
    )
    p.add_argument("--min_cv", type=float, default=0.0)
    p.add_argument("--de_adj_p", type=float, default=0.05)
    p.add_argument("--de_min_log2fc", type=float, default=1.0)
    p.add_argument(
        "--de_top_k",
        type=int,
        default=0,
        help="Max DE neurons per mutation (by |log2FC| rank). 0 = keep all significant.",
    )
    p.add_argument(
        "--de_mode",
        type=str,
        default="union",
        choices=["union", "per_mut"],
        help="'union': DE union of all 3 mutations (shared features). "
        "'per_mut': DE per Ctrl+Mutation pair (each mut gets own features).",
    )

    # Normalization (override: run only this norm instead of sweeping all)
    p.add_argument(
        "--norm",
        type=str,
        default="",
        help="If set, run only this norm. Otherwise sweep all NORM_CONFIGS.",
    )

    # PCA & KNN
    p.add_argument("--pca_dim", type=int, default=50)
    p.add_argument(
        "--n_neighbors", type=int, default=15, help="kNN neighbors for sc.pp.neighbors"
    )
    p.add_argument(
        "--n_diffmap_comps",
        type=int,
        default=15,
        help="Eigenvectors to compute in sc.tl.diffmap",
    )
    p.add_argument(
        "--n_dcs",
        type=int,
        default=10,
        help="Eigenvectors to USE for sc.tl.dpt (≤ n_diffmap_comps)",
    )

    # Misc common
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument(
        "--samples_per_class",
        type=int,
        default=15000,
        help="Max samples per class (0 = use ALL). Prioritizes valid cell_death.",
    )

    return p


def save_args_to_json(args, script_name=""):
    """Save the argparse arguments to a JSON file in the output directory."""
    if not hasattr(args, "output_dir") or not args.output_dir:
        return

    os.makedirs(args.output_dir, exist_ok=True)

    import json
    import sys
    from datetime import datetime

    if not script_name:
        script_name = os.path.basename(sys.argv[0]).replace(".py", "")

    out_path = os.path.join(args.output_dir, f"run_args_{script_name}.json")

    args_dict = vars(args).copy()
    args_dict["_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    args_dict["_script"] = sys.argv[0]

    try:
        with open(out_path, "w") as f:
            json.dump(args_dict, f, indent=4)
        logger.info(f"Saved execution arguments to: {out_path}")
    except Exception as e:
        logger.warning(f"Failed to save execution arguments: {e}")


# ==============================================================================
# Load cache + apply alive_mask
# ==============================================================================
def load_features_cache(cache_path: str, dead_threshold: float):
    data = np.load(cache_path, allow_pickle=True)

    if "X_all" in data:
        X_all = data["X_all"]
    elif "X_gap" in data:
        X_all = data["X_gap"]
    else:
        raise KeyError(
            f"Cache has neither 'X_all' nor 'X_gap'. Keys: {list(data.keys())}"
        )

    y = data["y"]
    lines = (
        data["lines"].astype(str) if data["lines"].dtype.kind != "U" else data["lines"]
    )
    uids = data["uids"].astype(str) if data["uids"].dtype.kind != "U" else data["uids"]
    which_layer = str(data["which_layer"])

    if "usage_ema" in data:
        usage_ema = data["usage_ema"]
        alive_mask = usage_ema >= dead_threshold
        n_alive = int(alive_mask.sum())
        n_total = len(usage_ema)
        X = X_all[:, alive_mask]
        alive_info = f"alive={n_alive}/{n_total}, dead_thresh={dead_threshold}"
    else:
        X = X_all
        n_total = X_all.shape[1]
        alive_info = f"all={n_total} (no usage_ema, CNN GAP mode)"

    logger.info(f"  Loaded: {cache_path}")
    logger.info(f"  Shape: {X_all.shape} → {X.shape} ({alive_info})")
    return X, y, lines, uids, which_layer, alive_info


# ==============================================================================
# Filters
# ==============================================================================
def compute_cv_per_neuron(X: np.ndarray, labels: list):
    labels_arr = np.array(labels)
    classes = np.unique(labels_arr)
    class_means = np.zeros((len(classes), X.shape[1]))
    for i, c in enumerate(classes):
        class_means[i] = X[labels_arr == c].mean(axis=0)

    means = class_means.mean(axis=0)
    stds = class_means.std(axis=0)
    means_safe = np.where(np.abs(means) < 1e-12, 1e-12, np.abs(means))
    cv = stds / means_safe
    return cv


def compute_de_neurons(
    X: np.ndarray,
    superclasses: list,
    mutation: str,
    adj_p_threshold: float = 0.05,
    min_log2fc: float = 0.0,
):
    from scipy.stats import mannwhitneyu
    from statsmodels.stats.multitest import multipletests

    superclasses_arr = np.array(superclasses)
    ctrl_mask = superclasses_arr == "Control"
    mut_mask = superclasses_arr == mutation

    if ctrl_mask.sum() == 0 or mut_mask.sum() == 0:
        return {
            "mask": np.zeros(X.shape[1], dtype=bool),
            "adj_pvalues": np.ones(X.shape[1]),
            "log2fc": np.zeros(X.shape[1]),
            "n_selected": 0,
        }

    X_ctrl = X[ctrl_mask]
    X_mut = X[mut_mask]

    d = X.shape[1]
    pvals = np.ones(d)

    ctrl_means = X_ctrl.mean(axis=0)
    mut_means = X_mut.mean(axis=0)
    eps = 1e-10
    log2fc = np.log2((mut_means + eps) / (ctrl_means + eps))

    for j in range(d):
        ctrl_vals = X_ctrl[:, j]
        mut_vals = X_mut[:, j]
        if ctrl_vals.std() == 0 and mut_vals.std() == 0:
            continue
        try:
            _, p = mannwhitneyu(ctrl_vals, mut_vals, alternative="two-sided")
            pvals[j] = p
        except ValueError:
            pass

    reject, adj_p, _, _ = multipletests(pvals, method="fdr_bh")
    mask = adj_p < adj_p_threshold
    if min_log2fc > 0:
        mask &= np.abs(log2fc) >= min_log2fc

    n_selected = int(mask.sum())
    logger.info(
        f"    DE ({mutation} vs Control): {n_selected}/{d} neurons (adj_p<{adj_p_threshold}, |log2FC|>={min_log2fc})"
    )

    return {
        "mask": mask,
        "adj_pvalues": adj_p,
        "log2fc": log2fc,
        "n_selected": n_selected,
    }


# ==============================================================================
# Load cell_death
# ==============================================================================
def load_and_match_cell_death(cell_death_csv: str, uids: list, rate_col=None):
    df = pd.read_csv(cell_death_csv)
    uid_col = next(
        (c for c in ["filename", "uid", "image_uid", "UID"] if c in df.columns),
        df.columns[0],
    )

    if rate_col and rate_col.upper() == "MFI":
        df["_MFI"] = df["total_intensity"] / df["total_nucleus_pixels"]
        use_col = "_MFI"
    elif rate_col and rate_col in df.columns:
        use_col = rate_col
    else:
        use_col = next(
            (
                c
                for c in ["intensity_rate", "cell_death_rate", "rate"]
                if c in df.columns
            ),
            df.columns[1],
        )

    uid_to_rate = {}
    for _, row in df.iterrows():
        key = os.path.splitext(str(row[uid_col]).replace("_mask", ""))[0]
        uid_to_rate[key] = float(row[use_col])

    def _normalize_cache_uid(uid_str):
        if ":" in uid_str:
            uid_str = uid_str.split(":")[-1]
        return os.path.splitext(uid_str.replace("_mask", ""))[0]

    cache_uids_norm = [_normalize_cache_uid(str(u)) for u in uids]
    cell_death = np.full(len(uids), np.nan)
    n_matched = 0
    for i, norm_uid in enumerate(cache_uids_norm):
        if norm_uid in uid_to_rate:
            cell_death[i] = uid_to_rate[norm_uid]
            n_matched += 1

    logger.info(f"  cell_death matched: {n_matched}/{len(uids)}")
    return cell_death


# ==============================================================================
# Feature Normalization
# ==============================================================================
def apply_normalization(X: np.ndarray, norm_method: str):
    X_out = X.copy()
    if "log" in norm_method:
        X_out = np.log1p(X_out)
    if (
        "median" in norm_method
        and "log_median" in norm_method
        or norm_method == "median"
    ):
        medians = np.median(X_out, axis=0)
        medians = np.where(medians == 0, 1e-12, medians)
        X_out = X_out / medians
    elif "IQR" in norm_method:
        iqr = np.percentile(X_out, 75, axis=0) - np.percentile(X_out, 25, axis=0)
        iqr = np.where(iqr == 0, 1e-12, iqr)
        X_out = (X_out - np.median(X_out, axis=0)) / iqr
    elif "std" in norm_method:
        std = X_out.std(axis=0)
        std = np.where(std == 0, 1e-12, std)
        X_out = (X_out - X_out.mean(axis=0)) / std
    return X_out


# ==============================================================================
# Roots
# ==============================================================================
def find_root_pca(X_ctrl_pca, X_mut_pca):
    centroid = X_ctrl_pca.mean(axis=0)
    dists = np.linalg.norm(X_ctrl_pca - centroid, axis=1)
    medoid_coord = X_ctrl_pca[np.argmin(dists)]
    dists_to_medoid = np.linalg.norm(X_mut_pca - medoid_coord, axis=1)

    k = max(10, int(0.01 * len(X_mut_pca)))
    k = min(k, len(X_mut_pca))
    top_k = np.argsort(dists_to_medoid)[:k]

    n_nn = min(5, k - 1)
    if n_nn < 1:
        root = top_k[0]
    else:
        nn = NearestNeighbors(n_neighbors=n_nn).fit(X_mut_pca[top_k])
        avg_dist = nn.kneighbors()[0].mean(axis=1)
        root = top_k[np.argmin(avg_dist)]
    return root


def find_root_diffmap(X_pca_all, superclasses, mutation, n_neighbors=15, n_comps=10):
    import scanpy as sc

    sc_arr = np.array(superclasses)
    ctrl_mask = sc_arr == "Control"
    mut_mask = sc_arr == mutation

    adata_all = sc.AnnData(X_pca_all.astype(np.float32))
    adata_all.obsm["X_pca"] = X_pca_all.astype(np.float32)
    n_comps = max(min(n_comps, X_pca_all.shape[1] - 1), 2)

    sc.pp.neighbors(
        adata_all, n_neighbors=n_neighbors, n_pcs=X_pca_all.shape[1], use_rep="X_pca"
    )
    sc.tl.diffmap(adata_all, n_comps=n_comps)
    Z = adata_all.obsm["X_diffmap"]

    ctrl_centroid = Z[ctrl_mask].mean(axis=0)
    ctrl_dists = np.linalg.norm(Z[ctrl_mask] - ctrl_centroid, axis=1)
    medoid_coord = Z[ctrl_mask][np.argmin(ctrl_dists)]

    Z_mut = Z[mut_mask]
    dists = np.linalg.norm(Z_mut - medoid_coord, axis=1)

    k = max(10, int(0.01 * len(Z_mut)))
    k = min(k, len(Z_mut))
    top_k = np.argsort(dists)[:k]

    n_nn = min(5, k - 1)
    if n_nn < 1:
        root = top_k[0]
    else:
        nn = NearestNeighbors(n_neighbors=n_nn).fit(Z_mut[top_k])
        avg_dist = nn.kneighbors()[0].mean(axis=1)
        root = top_k[np.argmin(avg_dist)]
    return root


def find_root_mnn(X_ctrl_pca, X_mut_pca, mnn_k=30):
    k = max(min(mnn_k, len(X_ctrl_pca) - 1, len(X_mut_pca) - 1), 1)
    nn_mut = NearestNeighbors(n_neighbors=k).fit(X_mut_pca)
    _, idx_c2m = nn_mut.kneighbors(X_ctrl_pca)
    nn_ctrl = NearestNeighbors(n_neighbors=k).fit(X_ctrl_pca)
    _, idx_m2c = nn_ctrl.kneighbors(X_mut_pca)

    mnn_count = np.zeros(len(X_mut_pca), dtype=int)
    for i in range(len(X_ctrl_pca)):
        for j in idx_c2m[i]:
            if i in idx_m2c[j]:
                mnn_count[j] += 1

    if mnn_count.max() == 0:
        return find_root_pca(X_ctrl_pca, X_mut_pca)

    max_count = mnn_count.max()
    top_mask = mnn_count >= max(1, max_count // 2)
    top_indices = np.where(top_mask)[0]

    if len(top_indices) <= 1:
        root = top_indices[0] if len(top_indices) == 1 else np.argmax(mnn_count)
    else:
        n_nn = min(5, len(top_indices) - 1)
        nn = NearestNeighbors(n_neighbors=n_nn).fit(X_mut_pca[top_indices])
        avg_dist = nn.kneighbors()[0].mean(axis=1)
        root = top_indices[np.argmin(avg_dist)]
    return root


# ==============================================================================
# Helper to perform standard preprocessing steps
# ==============================================================================
def load_and_preprocess(args):
    """Loads cache, normalizes, applies filters and returns preprocessed data."""
    data = np.load(args.features_cache, allow_pickle=True)
    if "X_gap" in data:
        X = data["X_gap"]
        lines = data["lines"]
        uids = data["uids"]
        which_layer = str(data["which_layer"])
    else:
        X, _, lines, uids, which_layer, _ = load_features_cache(
            args.features_cache, args.dead_threshold
        )

    lines = lines.astype(str) if lines.dtype.kind != "U" else lines
    uids = uids.astype(str) if uids.dtype.kind != "U" else uids

    # Save purely raw log1p features before any L2 norm or std scaling
    X_log = np.log1p(X)

    if getattr(args, "pre_l2_norm", False):
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-12, norms)
        X = X / norms

    if getattr(args, "gap_l2_norm", False):
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-12, norms)
        X = X / norms

    superclasses = [SUPERCLASS_MAP.get(ln, ln) for ln in lines]
    superclasses_arr = np.array(superclasses)

    target_classes = ["Control", "SNCA", "GBA", "LRRK2"]
    mask = np.isin(superclasses_arr, target_classes)
    if not np.all(mask):
        X = X[mask]
        X_log = X_log[mask]
        superclasses_arr = superclasses_arr[mask]
        uids = (
            uids[mask]
            if isinstance(uids, np.ndarray)
            else [uids[i] for i, m in enumerate(mask) if m]
        )

    cell_death = load_and_match_cell_death(args.cell_death_csv, uids)

    # Filtering (union applied only)
    has_de = "de" in args.filter_mode
    for fm in args.filter_mode:
        if fm in ("none", "de"):
            continue
        if fm == "cv":
            cv = compute_cv_per_neuron(X, superclasses_arr)
            X = X[:, cv >= args.min_cv]
            X_log = X_log[:, cv >= args.min_cv]

    if has_de and args.de_mode == "union":
        de_masks = []
        for mut in ["SNCA", "GBA", "LRRK2"]:
            res = compute_de_neurons(
                X, superclasses_arr, mut, args.de_adj_p, args.de_min_log2fc
            )
            m = res["mask"]
            if args.de_top_k > 0 and m.sum() > args.de_top_k:
                sig_idx = np.where(m)[0]
                top_k = sig_idx[
                    np.argsort(np.abs(res["log2fc"][sig_idx]))[::-1][: args.de_top_k]
                ]
                m = np.zeros_like(m)
                m[top_k] = True
            de_masks.append(m)
        allm = [("AllMut" if s != "Control" else "Control") for s in superclasses_arr]
        de_ctrl = compute_de_neurons(
            X, allm, "AllMut", args.de_adj_p, args.de_min_log2fc
        )
        m_ctrl = de_ctrl["mask"] & (de_ctrl["log2fc"] < 0)
        de_masks.append(m_ctrl)
        union_mask = de_masks[0] | de_masks[1] | de_masks[2] | de_masks[3]
        X = X[:, union_mask]
        X_log = X_log[:, union_mask]

    # Subsampling
    spc = args.samples_per_class
    if spc > 0:
        rng = np.random.RandomState(args.seed)
        keep_indices = []
        for cls in np.unique(superclasses_arr):
            cls_idx = np.where(superclasses_arr == cls)[0]
            valid_mask = ~np.isnan(cell_death[cls_idx])
            valid_idx = cls_idx[valid_mask]
            invalid_idx = cls_idx[~valid_mask]
            ordered = np.concatenate([valid_idx, invalid_idx])
            n_take = min(spc, len(ordered))
            chosen = rng.choice(
                ordered[: max(n_take, len(valid_idx))],
                size=min(n_take, len(ordered)),
                replace=False,
            )
            keep_indices.extend(chosen.tolist())
        keep_indices = sorted(keep_indices)
        X = X[keep_indices]
        X_log = X_log[keep_indices]
        superclasses_arr = superclasses_arr[keep_indices]
        cell_death = cell_death[keep_indices]

    if args.norm and args.norm != "none":
        X = apply_normalization(X, args.norm)

    return X, superclasses_arr, cell_death, which_layer, X_log


# ==============================================================================
# Helper to save pandas crosstab as SVG
# ==============================================================================
def save_crosstab_as_svg(crosstab_df, output_path, dpi=200, title=""):
    import matplotlib
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(
        figsize=(crosstab_df.shape[1] * 1.5 + 2, crosstab_df.shape[0] * 0.5 + 1.5)
    )
    ax.axis("tight")
    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    table_data = crosstab_df.reset_index().values
    columns = [crosstab_df.index.name or "Cluster"] + list(crosstab_df.columns)

    table = ax.table(
        cellText=table_data, colLabels=columns, loc="center", cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Highlight header
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f2f2f2")

    fig.tight_layout()
    fig.savefig(output_path, format="svg", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
