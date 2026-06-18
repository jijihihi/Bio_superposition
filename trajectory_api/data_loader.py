import os
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import logging

from sae_project.step02_logging_utils import get_logger, SUPERCLASS_MAP

logger = get_logger("trajectory_api_data")

# ==============================================================================
# 1. Base Cache Loader
# ==============================================================================
def load_features_cache(cache_path: str, dead_threshold: float = 1e-5):
    """
    Load feature cache and apply alive_mask based on dead_threshold.
    """
    data = np.load(cache_path, allow_pickle=True)

    if "X_all" in data:
        X_all = data["X_all"]
    elif "X_gap" in data:
        X_all = data["X_gap"]
    else:
        raise KeyError(f"Cache has neither 'X_all' nor 'X_gap'. Keys: {list(data.keys())}")

    y = data["y"]
    lines = data["lines"].astype(str) if data["lines"].dtype.kind != 'U' else data["lines"]
    uids = data["uids"].astype(str) if data["uids"].dtype.kind != 'U' else data["uids"]
    which_layer = str(data["which_layer"])

    if "usage_ema" in data:
        usage_ema = data["usage_ema"]
        alive_mask = usage_ema >= dead_threshold
        n_alive = int(alive_mask.sum())
        n_total = len(usage_ema)
        X = X_all[:, alive_mask]
        alive_info = f"alive={n_alive}/{n_total}, dead_thresh={dead_threshold}"
        # Keep original indices for plotting later
        alive_indices = np.where(alive_mask)[0]
    else:
        X = X_all
        n_total = X_all.shape[1]
        alive_info = f"all={n_total} (no usage_ema, CNN GAP mode)"
        alive_indices = np.arange(n_total)

    logger.info(f"Loaded cache: {cache_path}")
    logger.info(f"Shape: {X_all.shape} → {X.shape} ({alive_info})")
    
    return X, y, lines, uids, which_layer, alive_info, alive_indices

# ==============================================================================
# 2. Filtering Utilities
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
    return stds / means_safe

def compute_de_neurons(X: np.ndarray, superclasses: list, mutation: str, 
                       adj_p_threshold: float = 0.05, min_log2fc: float = 0.0):
    superclasses_arr = np.array(superclasses)
    ctrl_mask = superclasses_arr == "Control"
    mut_mask = superclasses_arr == mutation

    if ctrl_mask.sum() == 0 or mut_mask.sum() == 0:
        logger.warning(f"No Control or {mutation} samples found for DE.")
        return {"mask": np.zeros(X.shape[1], dtype=bool)}

    X_ctrl = X[ctrl_mask]
    X_mut = X[mut_mask]

    d = X.shape[1]
    pvals = np.ones(d)
    log2fc = np.zeros(d)

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
    mask = (adj_p < adj_p_threshold)
    if min_log2fc > 0:
        mask &= (np.abs(log2fc) >= min_log2fc)

    logger.info(f"DE ({mutation} vs Control): {int(mask.sum())}/{d} neurons (adj_p<{adj_p_threshold})")
    return {"mask": mask, "adj_pvalues": adj_p, "log2fc": log2fc}

# ==============================================================================
# 3. Normalization and Metadata
# ==============================================================================
def apply_normalization(X: np.ndarray, norm_method: str):
    X_out = X.copy()
    if "log" in norm_method:
        X_out = np.log1p(np.maximum(X_out, 0))
    if "median" in norm_method and "log_median" in norm_method or norm_method == "median":
        medians = np.median(X_out, axis=0)
        medians = np.where(medians == 0, 1e-12, medians)
        X_out = X_out / medians
    elif "std" in norm_method:
        std = X_out.std(axis=0)
        std = np.where(std == 0, 1e-12, std)
        X_out = (X_out - X_out.mean(axis=0)) / std
    return X_out

def load_and_match_apoptosis(apoptosis_csv: str, uids: list, rate_col=None):
    df = pd.read_csv(apoptosis_csv)
    uid_col = df.columns[0]
    for c in ["filename", "uid", "image_uid", "UID"]:
        if c in df.columns:
            uid_col = c
            break

    if rate_col and rate_col.upper() == "MFI":
        df["_MFI"] = df["total_intensity"] / df["total_nucleus_pixels"]
        use_col = "_MFI"
    elif rate_col and rate_col in df.columns:
        use_col = rate_col
    else:
        use_col = df.columns[1]
        for c in ["intensity_rate", "apoptosis_rate", "rate"]:
            if c in df.columns:
                use_col = c
                break

    uid_to_rate = {}
    for _, row in df.iterrows():
        key = str(row[uid_col]).replace("_mask", "")
        key = os.path.splitext(key)[0]
        uid_to_rate[key] = float(row[use_col])

    def _normalize_cache_uid(uid_str):
        if ":" in uid_str: uid_str = uid_str.split(":")[-1]
        return os.path.splitext(uid_str.replace("_mask", ""))[0]

    cache_uids_norm = [_normalize_cache_uid(str(u)) for u in uids]
    apoptosis = np.full(len(uids), np.nan)
    for i, norm_uid in enumerate(cache_uids_norm):
        if norm_uid in uid_to_rate:
            apoptosis[i] = uid_to_rate[norm_uid]
            
    logger.info(f"Apoptosis matched: {np.sum(~np.isnan(apoptosis))}/{len(uids)}")
    return apoptosis

# ==============================================================================
# 4. Main API Wrapper
# ==============================================================================
def load_and_preprocess(
    cache_path: str,
    dead_threshold: float = 1e-5,
    gap_l2_norm: bool = False,
    norm_method: str = "log_std",
    apoptosis_csv: str = None,
    apoptosis_rate_col: str = None,
    filter_modes: list = [],
    min_cv: float = 0.0,
    de_mutation: str = "SNCA",
    de_adj_p: float = 0.05,
    de_min_log2fc: float = 0.0,
    n_subsample: int = 0,
    seed: int = 42
) -> sc.AnnData:
    """
    Load data, apply filtering/normalization, and return a scanpy AnnData object.
    """
    np.random.seed(seed)
    
    # 1. Load Data
    X, y, lines, uids, which_layer, _, original_indices = load_features_cache(cache_path, dead_threshold)
    superclasses = np.array([SUPERCLASS_MAP.get(ln, "Control") for ln in lines])
    
    # 2. Apoptosis Match
    apoptosis = None
    if apoptosis_csv and os.path.exists(apoptosis_csv):
        apoptosis = load_and_match_apoptosis(apoptosis_csv, uids, apoptosis_rate_col)
    
    # 3. L2 Norm (Optional)
    if gap_l2_norm:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-12, norms)
        X = X / norms
        logger.info("Applied L2 Normalization")

    # 4. Filtering
    mask = np.ones(X.shape[1], dtype=bool)
    if "cv" in filter_modes:
        cv_vals = compute_cv_per_neuron(X, superclasses.tolist())
        cv_mask = cv_vals >= min_cv
        mask &= cv_mask
        logger.info(f"CV filter (>= {min_cv}): kept {cv_mask.sum()} neurons")
        
    if "de" in filter_modes:
        de_res = compute_de_neurons(X, superclasses.tolist(), de_mutation, de_adj_p, de_min_log2fc)
        mask &= de_res["mask"]

    X = X[:, mask]
    final_indices = original_indices[mask]
    
    # 5. Normalization
    if norm_method and norm_method != "none":
        X = apply_normalization(X, norm_method)
        logger.info(f"Applied normalization: {norm_method}")
        
    # 6. Subsampling (Balanced across superclasses)
    if n_subsample > 0:
        unique_sc = np.unique(superclasses)
        idx_list = []
        n_per_group = max(1, n_subsample // len(unique_sc))
        for sc_name in unique_sc:
            sc_idx = np.where(superclasses == sc_name)[0]
            k = min(len(sc_idx), n_per_group)
            if k > 0:
                idx_list.append(np.random.choice(sc_idx, k, replace=False))
        if idx_list:
            subset = np.concatenate(idx_list)
            np.random.shuffle(subset)
            X = X[subset]
            superclasses = superclasses[subset]
            uids = uids[subset]
            if apoptosis is not None:
                apoptosis = apoptosis[subset]
            logger.info(f"Subsampled to {len(subset)} cells")

    # 7. Create AnnData
    obs_dict = {
        "mutation": superclasses,
        "uid": uids
    }
    if apoptosis is not None:
        obs_dict["apoptosis"] = apoptosis
        
    adata = sc.AnnData(X=X.astype(np.float32), obs=pd.DataFrame(obs_dict))
    adata.uns["which_layer"] = which_layer
    adata.uns["norm_method"] = norm_method
    adata.var["original_index"] = final_indices
    
    logger.info(f"Successfully created AnnData: {adata}")
    return adata
