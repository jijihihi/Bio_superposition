# ==============================================================================
# SAE Concept Activation Visualization
# - Load GAP means CSV and SAE checkpoint
# - Select top-K images per concept by GAP value
# - Visualize concept activations via bilinear interpolation
# ==============================================================================


# 이거할때는 GAP_csv 기준. GAP_csv는 dead threshold가 5e-4 기준으로 했기때문에 통과된 피처맴 개수가 적은것.


import argparse
import csv
import io
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

try:
    import tifffile
except ImportError:
    raise RuntimeError("tifffile not installed. pip install tifffile")

try:
    import matplotlib.cm as cm
except ImportError:
    raise RuntimeError("matplotlib not installed. pip install matplotlib")

from sae_project.step01_configs import get_args, resolve_paths
from run_CNN.logging_utils import SUPERCLASS_MAP, get_logger
from run_CNN.data_shards import (build_uid_to_refidx,
                                            load_all_sample_refs)
from run_CNN.data_bank import (InMemoryTarBank,
                                          SafeInstanceNormalize,
                                          load_split_csv)
from run_CNN.model_encoder import (SupMoCoModel, parse_int_list,
                                              renorm_unit_per_out_channel_,
                                              robust_load_state_dict)
from sae_project.step06_gated_sae import GatedSAE

logger = get_logger("visualize_concept")


"""
concept 선별(DE)과 heatmap 시각화는 분리해서 생각해야 합니다:

1. Concept 선별 (DE filter) → normrestored + L2 norm ✅
features_cache에 normrestored + L2 norm된 것을 넣으면 dpt_kendall과 동일한 기준으로 DE가 뽑힙니다. step14에 --features_cache로 해당 cache를 넣으면 됩니다. 이건 이미 된 겁니다.

2. 이미지 랭킹 (top-K 선택) → normrestored만, L2 norm은 안 하는게 좋음
step14의 

precompute_all_gap_values
에서 이미지를 정렬할 때:

Image A: concept_j GAP = 5.0, 전체 벡터 norm = 100  
Image B: concept_j GAP = 3.0, 전체 벡터 norm = 20
L2 norm 전: A가 상위 (5.0 > 3.0) → "이 concept이 강하게 나타난 이미지"
L2 norm 후: A=0.05, B=0.15 → B가 상위 → "다른 concept 대비 비율이 높은 이미지"
시각화 목적으로는 **"이 concept이 실제로 강하게 나타난 이미지"**를 보고 싶으므로 L2 norm 없이 normrestored GAP으로 정렬하는 게 맞습니다.

3. Heatmap → normrestored만 ✅ (이미 적용됨)
spatial activation map에 L2 norm을 적용하는 것은 의미가 없습니다. 어디에서 강하게 fire하는지를 보여줘야 하므로 normrestored 그대로가 맞습니다.

토큰 centering per imgae로 하고 있다.

"""

## 시각화할때 control vs mutation 각 쌍으로 해서 mutation이 더 많이 나타난면, 그 거 선택 DE filter
# 이 떄 bilinear interpolation할때 그 mutation에서 많이 나타난다고 하면, 그 mutation만 bilinear interpolation해서 필터링 함.
# 이떄 SNCA와 GBA 모두에서 control보다 많이 나타나는거라고 하면, 둘 다 나오게끔. Top-20에서 순위대로 했을때 SNAC GBA 순서대로 해서 다 보이게끔.
# control - high인 경우에는, 모든 class 에서 top-k를 뽑느다. control vs all mutation이어서 정확하지 않을 수 있다.


# %matplotlib inline
# import logging
# logging.basicConfig(level=logging.INFO, force=True)

# ##이거하자
# %matplotlib inline
# import sys
# sys.argv = [
#     "step14",
#     "--features_cache","/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/MoCo_seed87/SAE_sparsity3200_loss_L2norm곱해줌/features_cache_stage5_out_normrestored_all.npz",   # L2 norm 된 상태. gap_csv 만들때와 동일하게. strict 사용.
#     "--sae_ckpt", "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/MoCo_seed87/SAE_sparsity3200_loss_L2norm곱해줌/stage5_out_d4096_gated_sp3200.0_aux0.03125_tied_ep008.pt",
#     "--cell_death_csv", "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/세포이미지별 사멸율/이미지별_세포사멸율_7200.csv",
#     "--model_state_path", "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/MoCo_seed87/best_model.pt",
#     "--shard_root", "/content/wds_shards",
#     "--save_dir", "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/MoCo_seed87",
#     "--output_dir", "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/MoCo_seed87/SAE_sparsity3200_loss_L2norm곱해줌/concept_by_gap_csv_d4096_sp3200_max_0.58",
#     "--concept_ids", "de_filter_csv",
#    "--max_gini", "0.75",
#     "--de_min_log2fc", "0.58",
#     "--mut_only",
#     "--gap_csv", "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/MoCo_seed87/SAE_sparsity3200_loss_L2norm곱해줌/gated_sae_stage5_out_d4096_sp3200.0_aux0.03125_tied_class_gap_means.csv"


# ]
# from concept_visulaize.step14_visualize_concept_activations import main
# main()


# %matplotlib inline
# import logging
# logging.basicConfig(level=logging.INFO, force=True)

# ##이거하자
# %matplotlib inline
# import sys
# sys.argv = [
#     "step14",
#     "--features_cache","/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/MoCo_seed87/SAE_sparsity3200_loss_L2norm곱해줌/features_cache_stage5_out_normrestored_all.npz",   # L2 norm 된 상태. gap_csv 만들때와 동일하게. strict 사용.
#     "--sae_ckpt", "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/MoCo_seed87/SAE_sparsity3200_loss_L2norm곱해줌/stage5_out_d4096_gated_sp3200.0_aux0.03125_tied_ep008.pt",
#     "--cell_death_csv", "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/세포이미지별 사멸율/이미지별_세포사멸율_7200.csv",
#     "--model_state_path", "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/MoCo_seed87/best_model.pt",
#     "--shard_root", "/content/wds_shards",
#     "--save_dir", "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/MoCo_seed87",
#     "--output_dir", "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/MoCo_seed87/SAE_sparsity3200_loss_L2norm곱해줌/concept_by_gap_csv_d4096_sp3200_max_0.58_superposition_interpretable",
#     "--concept_ids", "de_filter_csv",
#    "--max_gini", "1.0",
#     "--de_min_log2fc", "0.58",
#     "--mut_only",
#     "--gap_csv", "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/MoCo_seed87/SAE_sparsity3200_loss_L2norm곱해줌/gated_sae_stage5_out_d4096_sp3200.0_aux0.03125_tied_class_gap_means.csv"
#     ]
# from concept_visulaize.step14_visualize_concept_activations import main
# main()

# bilinear interpolation을 할때 dead threshold를 설정 안했고 alive 로 그냥 햇기 때문에 "--dead_threshold", "5e-4" 된 상태로 된거. 이거는 usage ema 기준. GAP_csv 뽑아낸 방식 (strict)으로 DE filter한것.abs
# 그렇다면 sparsity 800 인 경우도, dead threshold 잘 만 설정하면 보기 좋은 애들 많이 나오는거 아닐까?


### pseudotime heatmap

# pseudotime heatmap 위홰서 sparsity 800 시각화
# %matplotlib inline
# import sys
# sys.argv = [
#     "step14",
#     "--features_cache","/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/MoCo_seed87/SAE_seed856_no_L2norm_loss/features_cache_stage5_out_normrestored_all_no_SAE_GAP_L2_norm_again_d8192_sp800.npz",   # L2 norm 된 상태. gap_csv 만들때와 동일하게. strict 사용.
#     "--sae_ckpt", "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/MoCo_seed87/SAE_seed856_no_L2norm_loss/stage5_out_d8192_gated_sp800.0_aux0.03125_tied_ep008.pt",
#     "--cell_death_csv", "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/세포이미지별 사멸율/이미지별_세포사멸율_7200.csv",
#     "--model_state_path", "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/MoCo_seed87/best_model.pt",
#     "--shard_root", "/content/wds_shards",
#     "--save_dir", "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/MoCo_seed87",
#     "--output_dir", "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/MoCo_seed87/SAE_seed856_no_L2norm_loss/concept_activation_pseudotime_heatmap",
#     "--concept_ids", "de_filter_csv",
#    "--max_gini", "1.0",
#     "--de_min_log2fc", "0.58",
#     "--mut_only",
#     "--gap_csv", "/content/drive/MyDrive/Final_paper/lambda_labs_moco_only/MoCo_seed87/SAE_seed856_no_L2norm_loss/gated_sae_stage5_out_d8192_sp800.0_aux0.03125_tied_class_gap_means.csv",
#     "--dead_threshold", "1e-5",
#     ]
# from concept_visulaize.step14_visualize_concept_activations import main
# main()


# ==============================================================================
# Visualization Utilities
# ==============================================================================


def linear_uint16_to_uint8_rgb(img_u16: np.ndarray) -> np.ndarray:
    """Linear conversion from uint16 to uint8."""
    return (img_u16.astype(np.float32) / 65535.0 * 255.0).round().astype(np.uint8)


def fiji_linear_scaling_to_uint8(
    img_u16: np.ndarray,
    min_saturation_percent: float = 10.0,
    max_saturation_percent: float = 0.5,
) -> np.ndarray:
    """
    Fiji-style linear scaling (matches 9. cropped 된 이미지 QC.py).

    Args:
        img_u16: (H, W, 3) uint16 image
        min_saturation_percent: Bottom percentile to map to 0 (default: 10%)
        max_saturation_percent: Top percentile to saturate (default: 0.5%)

    Returns:
        (H, W, 3) uint8 image
    """
    MIN_STD_THRESHOLD = 655.0
    target_max = 255.0

    img = img_u16.astype(np.float32)
    out = np.zeros_like(img, dtype=np.uint8)

    for c in range(3):
        channel = img[..., c]
        raw_std = np.std(channel)

        if raw_std < MIN_STD_THRESHOLD:
            # Low variance: no scaling
            scaled = channel / 65535.0 * 255.0
        else:
            # Min cutoff: bottom n% (background removal)
            min_cutoff = np.percentile(channel, min_saturation_percent)

            # Max cutoff: top n% (signal preservation)
            max_cutoff = np.percentile(channel, 100 - max_saturation_percent)

            if max_cutoff <= min_cutoff:
                scaled = np.zeros_like(channel)
            else:
                # Subtract background (min_cutoff)
                channel_shifted = channel - min_cutoff

                # Scale to [0, 255]
                scale_factor = target_max / (max_cutoff - min_cutoff)
                scaled = channel_shifted * scale_factor

        out[..., c] = np.clip(scaled, 0, 255).astype(np.uint8)

    return out


def apply_colormap_01(a01: np.ndarray, cmap_name: str = "jet") -> np.ndarray:
    """
    Apply colormap to [0,1] normalized array.
    Returns (H,W,3) uint8.
    """
    a01 = np.clip(a01.astype(np.float32), 0.0, 1.0)
    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(a01)
    rgb8 = (rgba[..., :3] * 255.0).round().astype(np.uint8)
    return rgb8


def minmax_normalize(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Min-max normalize to [0, 1]."""
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx <= mn + eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def create_overlay(
    base_rgb: np.ndarray,
    heatmap_rgb: np.ndarray,
    alpha: float = 0.5,
    base_alpha: float = 1.0,
) -> np.ndarray:
    """
    Blend heatmap onto base image.
    """
    base = base_rgb.astype(np.float32) * base_alpha
    heat = heatmap_rgb.astype(np.float32)
    blended = base * (1 - alpha) + heat * alpha
    return blended.clip(0, 255).astype(np.uint8)


# ==============================================================================
# Data Loading
# ==============================================================================


def load_gap_csv(csv_path: str, dead_threshold: float = 1e-5) -> Dict[int, Dict]:
    """
    Load GAP means CSV.

    If dead_threshold > 0, recompute is_alive from n_* and total_active_imgs
    columns: activation_rate = total_active_imgs / total_images.
    A neuron is dead if activation_rate < dead_threshold.

    Returns dict: concept_id -> {
        'is_alive': bool,
        'Control': float, 'SNCA': float, 'GBA': float, 'LRRK2': float,
        'n_Control': int, ..., 'total_active_imgs': int,
        'max_class': str, 'class_diff': float, 'entropy': float
    }
    """
    concepts = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        has_n_cols = "n_Control" in headers and "total_active_imgs" in headers
        for row in reader:
            cid = int(row["concept_id"])
            info = {
                "is_alive": bool(int(row["is_alive"])),
                "Control": float(row["Control"]),
                "SNCA": float(row["SNCA"]),
                "GBA": float(row["GBA"]),
                "LRRK2": float(row["LRRK2"]),
                "max_class": row["max_class"],
                "class_diff": float(row["class_diff"]),
                "entropy": float(row["entropy"]),
            }
            if has_n_cols:
                info["n_Control"] = int(row["n_Control"])
                info["n_SNCA"] = int(row["n_SNCA"])
                info["n_GBA"] = int(row["n_GBA"])
                info["n_LRRK2"] = int(row["n_LRRK2"])
                info["total_active_imgs"] = int(row["total_active_imgs"])
            concepts[cid] = info

    # ── Recompute is_alive using dead_threshold ──
    if dead_threshold > 0 and has_n_cols:
        # Total images = max(total_active_imgs) across ALL concepts gives a
        # lower bound, but a better estimate is sum of n_* for the most-active
        # concept. Use the concept with the highest total_active_imgs as proxy.
        max_active = max((c["total_active_imgs"] for c in concepts.values()), default=1)
        # total_images ≈ n_Control + n_SNCA + n_GBA + n_LRRK2 for that concept
        best_cid = max(concepts, key=lambda k: concepts[k]["total_active_imgs"])
        best = concepts[best_cid]
        total_images = (
            best["n_Control"] + best["n_SNCA"] + best["n_GBA"] + best["n_LRRK2"]
        )
        if total_images == 0:
            total_images = max_active  # fallback
        logger.info(
            f"  Estimated total images: {total_images} "
            f"(from concept {best_cid}, total_active={max_active})"
        )

        n_original_alive = sum(1 for c in concepts.values() if c["is_alive"])
        n_recomputed_dead = 0
        for cid, info in concepts.items():
            activation_rate = info["total_active_imgs"] / max(total_images, 1)
            if activation_rate < dead_threshold:
                info["is_alive"] = False
                n_recomputed_dead += 1
        n_new_alive = sum(1 for c in concepts.values() if c["is_alive"])
        logger.info(
            f"  Dead neuron filter (threshold={dead_threshold:.1e}): "
            f"original alive={n_original_alive}, "
            f"newly killed={n_recomputed_dead}, "
            f"final alive={n_new_alive}"
        )

    return concepts


def compute_gini_impurity(class_values: List[float], eps: float = 1e-8) -> float:
    """
    Compute Gini impurity from class GAP values.
    Same formula as step13_class_specific_eval.py.

    Lower Gini impurity = more class-specific (concentrated in one class).

    Args:
        class_values: List of GAP values for each class [Control, SNCA, GBA, LRRK2]
        eps: Small value to avoid division by zero

    Returns:
        Gini impurity (0 = pure/one class, 0.75 = uniform for 4 classes)
    """
    values = np.array(class_values, dtype=np.float64)
    values = np.maximum(values, 0)  # Ensure non-negative

    total = values.sum() + eps
    probs = values / total

    # Gini impurity = 1 - sum(p_i^2)
    gini_impurity = 1.0 - np.sum(probs**2)
    return float(gini_impurity)


def filter_concepts_by_gini(
    gap_info: Dict[int, Dict],
    max_gini: float,
    classes: List[str] = ["Control", "SNCA", "GBA", "LRRK2"],
) -> List[int]:
    """
    Filter concepts by Gini coefficient threshold.
    Returns concept IDs with Gini <= max_gini (class-specific concepts).
    """
    filtered = []
    for cid, info in gap_info.items():
        if not info.get("is_alive", False):
            continue

        class_values = [info[c] for c in classes]
        gini = compute_gini_impurity(class_values)

        if gini <= max_gini:
            filtered.append((cid, gini, info["max_class"]))

    # Sort by Gini (most class-specific first)
    filtered.sort(key=lambda x: x[1])
    return filtered


def compute_gap_info_from_cache(
    features_cache_path: str,
    dead_threshold: float = 1e-5,
) -> Dict[int, Dict]:
    """
    Build gap_info dict (same format as load_gap_csv) from features_cache.
    Computes per-class GAP means directly from the cache.

    Returns: dict concept_id -> {
        'is_alive': bool,
        'Control': float, 'SNCA': float, 'GBA': float, 'LRRK2': float,
        'max_class': str, 'class_diff': float, 'entropy': float
    }
    """
    from run_CNN.logging_utils import SUPERCLASS_MAP

    data = np.load(features_cache_path, allow_pickle=True)
    X_all = data["X_all"]  # (N, d_sae)
    lines = data["lines"]  # (N,) string line names
    usage_ema = data["usage_ema"]  # (d_sae,)

    d_sae = len(usage_ema)
    class_names = ["Control", "SNCA", "GBA", "LRRK2"]

    # Map lines → superclasses
    superclasses = np.array([SUPERCLASS_MAP.get(str(ln), str(ln)) for ln in lines])

    # If X_all has all d_sae columns, use as is; if alive-only, need mapping
    alive_mask = usage_ema >= dead_threshold
    n_alive = int(alive_mask.sum())

    if X_all.shape[1] == d_sae:
        # All neurons present
        pass
    elif X_all.shape[1] == n_alive:
        # Alive only — expand back to full d_sae
        X_full = np.zeros((X_all.shape[0], d_sae), dtype=X_all.dtype)
        X_full[:, alive_mask] = X_all
        X_all = X_full
    else:
        logger.warning(
            f"  Cache X_all.shape[1]={X_all.shape[1]} != d_sae={d_sae} "
            f"and != n_alive={n_alive}"
        )

    # Compute per-class means for ALL neurons
    class_means = {}
    for cn in class_names:
        mask = superclasses == cn
        if mask.sum() > 0:
            class_means[cn] = X_all[mask].mean(axis=0)  # (d_sae,)
        else:
            class_means[cn] = np.zeros(d_sae)

    # Build gap_info dict
    gap_info = {}
    for i in range(d_sae):
        is_alive = bool(alive_mask[i])
        gaps = [float(class_means[cn][i]) for cn in class_names]
        max_class = class_names[np.argmax(gaps)] if max(gaps) > 0 else "None"
        class_diff = max(gaps) - min(gaps)
        total = sum(gaps) + 1e-10
        probs = [g / total for g in gaps]
        ent = -sum(p * np.log2(max(p, 1e-10)) for p in probs)
        gap_info[i] = {
            "is_alive": is_alive,
            "Control": gaps[0],
            "SNCA": gaps[1],
            "GBA": gaps[2],
            "LRRK2": gaps[3],
            "max_class": max_class,
            "class_diff": class_diff,
            "entropy": ent,
        }

    n_alive_count = sum(1 for v in gap_info.values() if v["is_alive"])
    logger.info(
        f"  Computed GAP info from cache: {d_sae} neurons, {n_alive_count} alive"
    )
    return gap_info


# ==============================================================================
# DE-based Concept Selection (from features_cache.npz)
# ==============================================================================


def compute_cv_per_neuron(X, superclasses):
    """Coefficient of variation per neuron (across-class std / global mean)."""
    sc_arr = np.array(superclasses)
    classes = sorted(set(sc_arr))
    class_means = np.array([X[sc_arr == c].mean(axis=0) for c in classes])
    cv = class_means.std(axis=0) / (class_means.mean(axis=0) + 1e-10)
    return cv


def compute_de_neurons_local(
    X: np.ndarray,
    superclasses: list,
    mutation: str,
    adj_p_threshold: float = 0.05,
    min_log2fc: float = 0.0,
):
    """
    DEG-like neuron selection: Wilcoxon rank-sum test + BH correction.
    Compares Control vs Mutation. Returns dict with mask, adj_pvalues, log2fc.
    """
    from scipy.stats import mannwhitneyu
    from statsmodels.stats.multitest import multipletests

    sc_arr = np.array(superclasses)
    ctrl_mask = sc_arr == "Control"
    mut_mask = sc_arr == mutation

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

    eps = 1e-10
    log2fc = np.log2((X_mut.mean(0) + eps) / (X_ctrl.mean(0) + eps))

    for j in range(d):
        if X_ctrl[:, j].std() == 0 and X_mut[:, j].std() == 0:
            continue
        try:
            _, p = mannwhitneyu(X_ctrl[:, j], X_mut[:, j], alternative="two-sided")
            pvals[j] = p
        except ValueError:
            pass

    reject, adj_p, _, _ = multipletests(pvals, method="fdr_bh")
    mask = adj_p < adj_p_threshold
    if min_log2fc > 0:
        mask &= np.abs(log2fc) >= min_log2fc

    return {
        "mask": mask,
        "adj_pvalues": adj_p,
        "log2fc": log2fc,
        "n_selected": int(mask.sum()),
    }


def select_concepts_by_de(
    features_cache_path: str,
    cell_death_csv_path: str,
    min_cv: float = 0.2,
    de_adj_p: float = 0.05,
    de_min_log2fc: float = 1.5,
    dead_threshold: float = 1e-5,
):
    """
    Select class-specific concepts using DE analysis.
    Returns list of (concept_id, dominant_class, log2fc_value, direction).
    """
    from run_CNN.logging_utils import SUPERCLASS_MAP

    # Load features
    data = np.load(features_cache_path, allow_pickle=True)
    X = data["X_all"].astype(np.float32)
    uids = list(data["uids"])

    # Dead neuron filter
    if "usage_ema" in data:
        usage = data["usage_ema"]
        alive = usage >= dead_threshold
        alive_indices = np.where(alive)[0]
        X = X[:, alive]
        logger.info(f"  Alive neurons: {len(alive_indices)}/{len(usage)}")
    else:
        alive_indices = np.arange(X.shape[1])

    # Load superclasses from cell_death CSV (same matching logic as dpt_kendall.py)
    import csv as csv_mod

    # Build uid_to_superclass from cell_death CSV
    uid_to_sc = {}
    with open(cell_death_csv_path, "r", encoding="utf-8") as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            # Get filename and normalize (strip _mask, extension)
            fn = row.get("filename") or row.get("uid") or row.get("UID") or ""
            fn = fn.replace("_mask", "")
            fn = os.path.splitext(fn)[0]

            # Get class/line info for superclass mapping
            line_val = (
                row.get("folder")
                or row.get("class")
                or row.get("line")
                or row.get("Line")
                or ""
            )
            sc = SUPERCLASS_MAP.get(line_val, None)
            if fn and sc:
                uid_to_sc[fn] = sc

    logger.info(f"  cell_death CSV entries: {len(uid_to_sc)}")
    if uid_to_sc:
        sample_keys = list(uid_to_sc.keys())[:3]
        logger.info(f"  Sample CSV keys: {sample_keys}")

    # Normalize cache UIDs: 'path/to.tar:IMAGE_NAME' → 'IMAGE_NAME'
    def _norm_uid(u):
        u = str(u)
        if ":" in u:
            u = u.split(":")[-1]
        u = os.path.splitext(u)[0]
        u = u.replace("_mask", "")
        return u

    cache_uids_norm = [_norm_uid(u) for u in uids]
    logger.info(f"  Sample cache UIDs (normalized): {cache_uids_norm[:3]}")

    # Match UIDs (using normalized keys)
    superclasses = []
    valid_mask = []
    for i, norm_uid in enumerate(cache_uids_norm):
        if norm_uid in uid_to_sc:
            superclasses.append(uid_to_sc[norm_uid])
            valid_mask.append(True)
        else:
            valid_mask.append(False)
    valid_mask = np.array(valid_mask)
    X = X[valid_mask]
    logger.info(f"  Matched images: {len(superclasses)}")

    # CV filter
    cv = compute_cv_per_neuron(X, superclasses)
    cv_mask = cv >= min_cv
    logger.info(f"  CV >= {min_cv}: {cv_mask.sum()}/{len(cv)} neurons")

    # DE: per-mutation (mutation-high)
    selected = []  # (original_concept_id, class, log2fc, direction)
    mutations = ["SNCA", "GBA", "LRRK2"]

    for mut in mutations:
        de = compute_de_neurons_local(
            X, superclasses, mut, adj_p_threshold=de_adj_p, min_log2fc=de_min_log2fc
        )
        # Mutation-high: log2fc > 0 (higher in mutation)
        mut_high = de["mask"] & (de["log2fc"] > 0) & cv_mask
        n = int(mut_high.sum())
        logger.info(f"    {mut}-high: {n} neurons")
        for local_idx in np.where(mut_high)[0]:
            orig_id = int(alive_indices[local_idx])
            selected.append((orig_id, mut, float(de["log2fc"][local_idx]), "mut_high"))

    # DE: Control vs AllMut (control-high)
    sc_allm = [("AllMut" if s != "Control" else "Control") for s in superclasses]
    de_ctrl = compute_de_neurons_local(
        X, sc_allm, "AllMut", adj_p_threshold=de_adj_p, min_log2fc=de_min_log2fc
    )
    ctrl_high = de_ctrl["mask"] & (de_ctrl["log2fc"] < 0) & cv_mask
    n_ctrl = int(ctrl_high.sum())
    logger.info(f"    Control-high: {n_ctrl} neurons")
    for local_idx in np.where(ctrl_high)[0]:
        orig_id = int(alive_indices[local_idx])
        selected.append(
            (orig_id, "Control", float(de_ctrl["log2fc"][local_idx]), "ctrl_high")
        )

    # Merge classes per concept (same neuron can be DE in multiple mutations)
    from collections import defaultdict as _ddict

    concept_classes = _ddict(list)  # concept_id -> [(class, log2fc, direction), ...]
    for cid, cls, fc, direction in selected:
        concept_classes[cid].append((cls, fc, direction))

    deduped = []
    for cid, entries in concept_classes.items():
        classes = sorted(set(e[0] for e in entries))
        max_fc = max(abs(e[1]) for e in entries)
        label = "_".join(classes)  # e.g. "GBA_SNCA"
        direction = entries[0][2]
        deduped.append((cid, label, max_fc, direction))

    # Sort by |log2fc| descending
    deduped.sort(key=lambda x: abs(x[2]), reverse=True)
    logger.info(f"  Total DE-selected concepts: {len(deduped)}")

    return deduped


def select_concepts_by_gap_csv_de(
    gap_info: Dict[int, Dict],
    max_gini: float = 0.74,
    de_min_log2fc: float = 1.0,
) -> list:
    """
    Select class-specific concepts using gap_csv per-class means.
    Uses log2FC (no Wilcoxon — gap_csv only has means, not per-image values).
    Optionally pre-filters by Gini impurity.

    Returns list of (concept_id, dominant_class, log2fc_value, direction).
    """
    class_names = ["Control", "SNCA", "GBA", "LRRK2"]
    mutations = ["SNCA", "GBA", "LRRK2"]
    eps = 1e-10

    selected = []  # (concept_id, class, log2fc, direction)

    for cid, info in gap_info.items():
        if not info.get("is_alive", False):
            continue

        # Optional Gini pre-filter
        class_vals = [max(info[cn], 0) for cn in class_names]
        gini = compute_gini_impurity(class_vals)
        if gini > max_gini:
            continue

        ctrl_mean = info["Control"]

        # Mutation-high: each mutation vs Control
        for mut in mutations:
            mut_mean = info[mut]
            log2fc = np.log2((mut_mean + eps) / (ctrl_mean + eps))
            if abs(log2fc) >= de_min_log2fc:
                if log2fc > 0:
                    selected.append((cid, mut, float(log2fc), "mut_high"))
                else:
                    selected.append((cid, "Control", float(log2fc), "ctrl_high"))

    # Deduplicate: one concept can be DE in multiple mutations
    from collections import defaultdict as _ddict

    concept_classes = _ddict(list)
    for cid, cls, fc, direction in selected:
        concept_classes[cid].append((cls, fc, direction))

    deduped = []
    for cid, entries in concept_classes.items():
        classes = sorted(set(e[0] for e in entries))
        max_fc = max(abs(e[1]) for e in entries)
        label = "_".join(classes)
        direction = entries[0][2]
        deduped.append((cid, label, max_fc, direction))

    deduped.sort(key=lambda x: abs(x[2]), reverse=True)
    logger.info(
        f"  gap_csv DE filter: {len(deduped)} concepts selected "
        f"(max_gini={max_gini}, min_log2fc={de_min_log2fc})"
    )
    return deduped


# ==============================================================================
# SAE Activation Extraction
# ==============================================================================


@torch.inference_mode()
def compute_concept_activations_for_images(
    encoder: nn.Module,
    sae: GatedSAE,
    bank: InMemoryTarBank,
    indices: List[int],
    concept_id: int,
    device: torch.device,
    which_layer: str = "stage5_out",
    batch_size: int = 32,
) -> Tuple[np.ndarray, List[int]]:
    """
    Compute per-image GAP for a specific concept.

    Returns:
        gap_values: (N,) array of GAP values
        valid_indices: list of bank indices that were successfully processed
    """
    encoder.eval()
    sae.eval()

    normalize = SafeInstanceNormalize(threshold=0.01)

    autocast_kwargs = dict(device_type="cuda", enabled=torch.cuda.is_available())
    if torch.cuda.is_available():
        autocast_kwargs["dtype"] = torch.bfloat16

    gap_values = []
    valid_indices = []

    for start in range(0, len(indices), batch_size):
        end = min(start + batch_size, len(indices))
        batch_indices = indices[start:end]

        xs = []
        batch_valid = []
        for bi in batch_indices:
            img = bank.images[bi]
            if img is None:
                continue
            x = img.astype(np.float32) / 65535.0
            x = torch.from_numpy(x).permute(2, 0, 1)
            x = normalize(x)
            xs.append(x)
            batch_valid.append(bi)

        if len(xs) == 0:
            continue

        xb = (
            torch.stack(xs, 0)
            .to(device, non_blocking=True)
            .contiguous(memory_format=torch.channels_last)
        )

        with torch.amp.autocast(**autocast_kwargs):
            fmap = encoder.forward_feature_maps(xb, which=which_layer)

        # Normalize same as training: GAP-scalar norm
        # config 1에서 dfualt로 지정한 --token_norm_mode", type=str, default="gap-scalar" 이렇게 되서 상관없다. 즉 양 보정하는 효과.
        B = fmap.size(0)
        gap = fmap.mean(dim=(2, 3))
        gap_norm = gap.norm(dim=1, keepdim=True).view(B, 1, 1, 1).clamp_min(1e-12)
        fmap = fmap / gap_norm

        fmap = fmap.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        C = fmap.shape[-1]
        _, Hf, Wf, _ = fmap.shape

        # Per-image centering + normalize (must match get_concept_activation_map)
        for b_idx in range(B):
            img_tokens = fmap[b_idx].view(Hf * Wf, C)  # (H*W, C)
            img_tokens = img_tokens - img_tokens.mean(dim=0, keepdim=True)
            img_tokens = F.normalize(img_tokens, dim=1, eps=1e-12)

            # SAE forward for this image
            with torch.amp.autocast(**autocast_kwargs):
                _, acts, _, _, _ = sae(img_tokens)

            acts = acts.float()  # (H*W, d_sae)
            concept_gap = acts[:, concept_id].mean()  # scalar GAP

            gap_values.append(concept_gap.cpu().item())
            valid_indices.append(batch_valid[b_idx])

    return np.array(gap_values), valid_indices


@torch.inference_mode()
def precompute_all_gap_values(
    encoder: nn.Module,
    sae: GatedSAE,
    bank: InMemoryTarBank,
    indices: List[int],
    device: torch.device,
    which_layer: str = "stage5_out",
    batch_size: int = 32,
) -> Tuple[np.ndarray, List[int]]:
    """
    Single pass: compute GAP values for ALL SAE concepts x ALL images.
    Returns:
        gap_all: (N_valid, d_sae) array
        valid_indices: list of bank indices
    """
    encoder.eval()
    sae.eval()
    normalize = SafeInstanceNormalize(threshold=0.01)
    autocast_kwargs = dict(device_type="cuda", enabled=torch.cuda.is_available())
    if torch.cuda.is_available():
        autocast_kwargs["dtype"] = torch.bfloat16

    gap_list = []
    valid_indices = []

    for start in tqdm(
        range(0, len(indices), batch_size), desc="Precompute GAP (all concepts)"
    ):
        end = min(start + batch_size, len(indices))
        batch_indices = indices[start:end]
        xs = []
        batch_valid = []
        for bi in batch_indices:
            img = bank.images[bi]
            if img is None:
                continue
            x = img.astype(np.float32) / 65535.0
            x = torch.from_numpy(x).permute(2, 0, 1)
            x = normalize(x)
            xs.append(x)
            batch_valid.append(bi)
        if len(xs) == 0:
            continue
        xb = (
            torch.stack(xs, 0)
            .to(device, non_blocking=True)
            .contiguous(memory_format=torch.channels_last)
        )
        with torch.amp.autocast(**autocast_kwargs):
            fmap = encoder.forward_feature_maps(xb, which=which_layer)
        B = fmap.size(0)
        gap = fmap.mean(dim=(2, 3))
        gap_norm = gap.norm(dim=1, keepdim=True).view(B, 1, 1, 1).clamp_min(1e-12)
        fmap = fmap / gap_norm
        fmap = fmap.permute(0, 2, 3, 1).contiguous()
        C = fmap.shape[-1]
        _, Hf, Wf, _ = fmap.shape
        for b_idx in range(B):
            img_tokens = fmap[b_idx].view(Hf * Wf, C)
            img_tokens = img_tokens - img_tokens.mean(dim=0, keepdim=True)
            token_l2 = img_tokens.norm(dim=1, keepdim=True).clamp_min(1e-12)
            img_tokens = F.normalize(img_tokens, dim=1, eps=1e-12)
            with torch.amp.autocast(**autocast_kwargs):
                _, acts, _, _, _ = sae(img_tokens)
            acts = acts.float() * token_l2  # restore token norm
            gap_list.append(acts.mean(dim=0).cpu().numpy())  # (d_sae,)
            valid_indices.append(batch_valid[b_idx])

    return np.stack(gap_list, axis=0), valid_indices


@torch.inference_mode()
def batch_compute_activation_maps(
    encoder: nn.Module,
    sae: GatedSAE,
    bank: InMemoryTarBank,
    image_concept_map: Dict[int, List[int]],
    device: torch.device,
    which_layer: str = "stage5_out",
    batch_size: int = 32,
) -> Dict[Tuple[int, int], np.ndarray]:
    """
    Batch compute spatial activation maps for specific (image, concept) pairs.
    Args:
        image_concept_map: bank_index -> list of concept IDs to extract
    Returns:
        act_cache: (bank_index, concept_id) -> (Hf, Wf) activation map
    """
    encoder.eval()
    sae.eval()
    normalize = SafeInstanceNormalize(threshold=0.01)
    autocast_kwargs = dict(device_type="cuda", enabled=torch.cuda.is_available())
    if torch.cuda.is_available():
        autocast_kwargs["dtype"] = torch.bfloat16

    act_cache = {}
    bank_indices = list(image_concept_map.keys())

    for start in tqdm(
        range(0, len(bank_indices), batch_size), desc="Computing activation maps"
    ):
        end = min(start + batch_size, len(bank_indices))
        batch_bi = bank_indices[start:end]
        xs = []
        batch_valid = []
        for bi in batch_bi:
            img = bank.images[bi]
            if img is None:
                continue
            x = img.astype(np.float32) / 65535.0
            x = torch.from_numpy(x).permute(2, 0, 1)
            x = normalize(x)
            xs.append(x)
            batch_valid.append(bi)
        if len(xs) == 0:
            continue
        xb = (
            torch.stack(xs, 0)
            .to(device, non_blocking=True)
            .contiguous(memory_format=torch.channels_last)
        )
        with torch.amp.autocast(**autocast_kwargs):
            fmap = encoder.forward_feature_maps(xb, which=which_layer)
        B = fmap.size(0)
        gap = fmap.mean(dim=(2, 3))
        gap_norm = gap.norm(dim=1, keepdim=True).view(B, 1, 1, 1).clamp_min(1e-12)
        fmap = fmap / gap_norm
        fmap = fmap.permute(0, 2, 3, 1).contiguous()
        _, Hf, Wf, C = fmap.shape
        for b_idx in range(B):
            bi = batch_valid[b_idx]
            img_tokens = fmap[b_idx].view(Hf * Wf, C)
            img_tokens = img_tokens - img_tokens.mean(dim=0, keepdim=True)
            token_l2 = img_tokens.norm(dim=1, keepdim=True).clamp_min(1e-12)
            img_tokens = F.normalize(img_tokens, dim=1, eps=1e-12)
            with torch.amp.autocast(**autocast_kwargs):
                _, acts, _, _, _ = sae(img_tokens)
            acts = (acts.float() * token_l2).view(Hf, Wf, -1)  # restore token norm
            for cid in image_concept_map[bi]:
                act_cache[(bi, cid)] = acts[:, :, cid].cpu().numpy()

    return act_cache


@torch.inference_mode()
def get_concept_activation_map(
    encoder: nn.Module,
    sae: GatedSAE,
    img_np: np.ndarray,
    concept_id: int,
    device: torch.device,
    which_layer: str = "stage5_out",
) -> np.ndarray:
    """
    Get activation map for a specific concept for a single image.

    Returns:
        act_hw: (H, W) activation map (at feature map resolution, e.g., 64x64)
    """
    encoder.eval()
    sae.eval()

    normalize = SafeInstanceNormalize(threshold=0.01)

    autocast_kwargs = dict(device_type="cuda", enabled=torch.cuda.is_available())
    if torch.cuda.is_available():
        autocast_kwargs["dtype"] = torch.bfloat16

    x = img_np.astype(np.float32) / 65535.0
    x = torch.from_numpy(x).permute(2, 0, 1)
    x = normalize(x)
    x = x.unsqueeze(0).to(device).contiguous(memory_format=torch.channels_last)

    with torch.amp.autocast(**autocast_kwargs):
        fmap = encoder.forward_feature_maps(x, which=which_layer)

    # Normalize same as training
    gap = fmap.mean(dim=(2, 3))
    gap_norm = gap.norm(dim=1, keepdim=True).view(1, 1, 1, 1).clamp_min(1e-12)
    fmap = fmap / gap_norm

    fmap = fmap.permute(0, 2, 3, 1).contiguous()
    _, Hf, Wf, C = fmap.shape

    flat_tokens = fmap.view(-1, C)
    flat_tokens = flat_tokens - flat_tokens.mean(dim=0, keepdim=True)
    token_l2 = flat_tokens.norm(dim=1, keepdim=True).clamp_min(1e-12)
    flat_tokens = F.normalize(flat_tokens, dim=1, eps=1e-12)

    # SAE forward
    with torch.amp.autocast(**autocast_kwargs):
        _, acts, _, _, _ = sae(flat_tokens)

    acts = (acts.float() * token_l2).view(Hf, Wf, -1)  # restore token norm
    act_hw = acts[:, :, concept_id].cpu().numpy()

    return act_hw


# ==============================================================================
# Main Visualization
# ==============================================================================


def visualize_concept(
    encoder: nn.Module,
    sae: GatedSAE,
    bank: InMemoryTarBank,
    concept_id: int,
    valid_indices: List[int],
    gap_values: np.ndarray,
    top_k: int,
    output_dir: str,
    device: torch.device,
    which_layer: str = "stage5_out",
    img_size: int = 128,
    cmap_name: str = "jet",
    overlay_alpha: float = 0.5,
    base_alpha: float = 1.0,
    act_cache: Dict[Tuple[int, int], np.ndarray] = None,
    **kwargs,
) -> List[Dict]:
    """
    Visualize top-K images for a specific concept.

    Returns:
        List of dicts — one row per saved image — for building top_k_images.csv.
        Columns: concept_id, concept_class, rank, img_name, line, gap_val,
                 max_act, concept_dir, base_filename

    CSV workflow:
        After step14 runs, use top_k_images.csv to:
        1) Get the union of img_name values (unique images across all concepts).
        2) Pass that list to step15_cnn_featuremap_compare.py which, for each
           image, runs CNN forward_feature_maps() and saves 512 bilinear-
           interpolated channel heatmaps into one grid PNG named by img_name.
        3) Load the SAE overlay (step14 output) and the CNN grid (step15 output)
           side-by-side — no Ctrl+F needed, both are keyed by img_name.
    """
    csv_rows: List[Dict] = []
    # Include dominant class in directory name if provided
    class_label = kwargs.get("class_label", "")
    if class_label:
        concept_dir = os.path.join(
            output_dir, f"concept_{concept_id:04d}_{class_label}"
        )
    else:
        concept_dir = os.path.join(output_dir, f"concept_{concept_id:04d}")
    os.makedirs(concept_dir, exist_ok=True)

    # Sort by GAP and take top-K
    sorted_idx = np.argsort(gap_values)[::-1]

    # DEBUG: log GAP distribution
    top_gaps = gap_values[sorted_idx[: min(20, len(sorted_idx))]]
    nonzero = (gap_values > 1e-6).sum()
    logger.info(
        f"  Concept {concept_id}: nonzero GAP = {nonzero}/{len(gap_values)}, "
        f"top-5 GAP = {top_gaps[:5].tolist()}"
    )

    # Only keep images with nonzero GAP
    sorted_idx = sorted_idx[gap_values[sorted_idx] > 1e-6]
    top_indices = sorted_idx[:top_k]

    if len(top_indices) == 0:
        logger.warning(f"  Concept {concept_id}: no images with nonzero GAP!")
        return

    for rank, idx in enumerate(top_indices, start=1):
        bi = valid_indices[idx]
        gap_val = gap_values[idx]

        img_u16 = bank.images[bi]
        if img_u16 is None:
            continue

        line = bank.lines[bi]
        label = bank.labels[bi]

        # Extract image name from uid: "{tar_path}:{prefix}" → prefix without extension
        raw_uid = bank.uids[bi] if bi < len(bank.uids) else ""
        img_name = raw_uid.split(":")[-1] if ":" in raw_uid else raw_uid
        img_name = os.path.splitext(img_name)[0]  # strip extension if any
        img_name = img_name.replace("/", "_").replace(
            "\\", "_"
        )  # sanitize path separators

        # Get activation map (use cache if available)
        if act_cache is not None and (bi, concept_id) in act_cache:
            act_hw = act_cache[(bi, concept_id)]
        else:
            act_hw = get_concept_activation_map(
                encoder, sae, img_u16, concept_id, device, which_layer
            )

        # Upsample to image size
        act_t = torch.from_numpy(act_hw).unsqueeze(0).unsqueeze(0).float()
        act_up = (
            F.interpolate(
                act_t, size=(img_size, img_size), mode="bilinear", align_corners=False
            )
            .squeeze()
            .numpy()
        )

        # Check if there's any meaningful activation
        max_act = float(act_up.max())
        mean_act = float(act_up.mean())

        # Skip if no activation at all (avoid all-blue heatmaps)
        if max_act < 1e-6:
            logger.warning(
                f"  Concept {concept_id} rank {rank}: no activation (max={max_act:.6f}), skipping"
            )
            continue

        # Normalize for visualization using percentile-based scaling
        # This highlights the relative differences within each image
        p_low = np.percentile(
            act_up, 50
        )  # Use median as baseline (most sparse activations are 0)
        p_high = np.percentile(act_up, 99.9)  # Top 0.1% as maximum

        if p_high <= p_low + 1e-8:
            # Fallback to minmax if percentiles are too close
            act_norm = minmax_normalize(act_up)
        else:
            act_norm = np.clip((act_up - p_low) / (p_high - p_low), 0, 1)

        # Generate images (Fiji-style linear scaling)
        orig_rgb = linear_uint16_to_uint8_rgb(img_u16)
        bright_rgb = fiji_linear_scaling_to_uint8(img_u16)  # 10%~99.5% linear scaling
        heatmap_rgb = apply_colormap_01(act_norm, cmap_name=cmap_name)
        overlay_rgb = create_overlay(
            bright_rgb, heatmap_rgb, alpha=overlay_alpha, base_alpha=base_alpha
        )

        # CNN input view: SafeInstanceNormalize → global scale to uint8
        # Key: scale all 3 channels together (not per-channel) to show
        # how InstanceNorm equalizes channel intensities — THIS is what CNN sees
        # 채널별로 safenormalization 후 채널별로 percnetile해서 linear scaling을 하면, IN해도 각 채널별로 밝기 순위는 남는데 여기서 linear scaling을 하면 가장 밝은애는 255 가장 어두운 애는 0으로 만들어 버리기때문에 IN 없이 lienar scaling하는 것과 아무런 의미가 없어.
        # 대신 global percentil로 scaling을 하면. 많이 밝은 애는 최대값에 많이 포함된다. 이 percentil에 많이 포함 안되는 채널도 있겠지. 걔는 가장 밝은 픽셀은 없겠지
        # p99에 포함되지 않을 정도로만 밝은 애들만 있는 채널의 경우에는 global percentil하면 포함이 안된다. 따라서 그렇게 밝은 애는 없다.
        # 이게 IN 한 CNN에 넣어주는 이미지와 비슷하다. 왜냐하면 채널별로 IN했을때 평균 0 표준편차 1 되지나 tail은 다 다르다. 근데 이걸 단순히 linear scaling하면 tail이 손상되는 효과가 있기 때문이다.
        normalize = SafeInstanceNormalize(threshold=0.01)
        cnn_float = img_u16.astype(np.float32) / 65535.0
        cnn_tensor = torch.from_numpy(cnn_float).permute(2, 0, 1)  # (3, H, W)
        cnn_tensor = normalize(cnn_tensor)  # per-channel instance norm
        cnn_np = cnn_tensor.permute(1, 2, 0).numpy()  # (H, W, 3)
        # Global scale across ALL channels (preserves cross-channel balance)
        lo, hi = np.percentile(cnn_np, [1, 99])
        if hi - lo < 1e-6:
            hi = lo + 1e-6
        cnn_rgb = np.clip((cnn_np - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
        cnn_overlay_rgb = create_overlay(
            cnn_rgb, heatmap_rgb, alpha=overlay_alpha, base_alpha=base_alpha
        )

        # Base filename: {class}_{img_name}__rank{rank}_{line}_gap..._max...
        # class_label comes from kwargs (e.g. "LRRK2"), line is the specific line name
        # img_name allows CSV-based lookup to match SAE images with CNN feature maps
        _cls = class_label if class_label else line
        base = f"{_cls}_{img_name}__rank{rank:02d}_{line}_gap{gap_val:.4f}_max{max_act:.4f}"

        # Save images
        Image.fromarray(orig_rgb, mode="RGB").save(
            os.path.join(concept_dir, f"{base}_01_orig.png")
        )
        Image.fromarray(bright_rgb, mode="RGB").save(
            os.path.join(concept_dir, f"{base}_02_bright.png")
        )
        Image.fromarray(heatmap_rgb, mode="RGB").save(
            os.path.join(concept_dir, f"{base}_03_heatmap.png")
        )
        Image.fromarray(overlay_rgb, mode="RGB").save(
            os.path.join(concept_dir, f"{base}_04_overlay.png")
        )
        Image.fromarray(cnn_rgb, mode="RGB").save(
            os.path.join(concept_dir, f"{base}_05_cnn_input.png")
        )
        Image.fromarray(cnn_overlay_rgb, mode="RGB").save(
            os.path.join(concept_dir, f"{base}_06_cnn_overlay.png")
        )

        # Accumulate CSV row
        csv_rows.append(
            {
                "concept_id": concept_id,
                "concept_class": class_label if class_label else line,
                "rank": rank,
                "img_name": img_name,
                "line": line,
                "gap_val": round(float(gap_val), 6),
                "max_act": round(float(max_act), 6),
                "concept_dir": os.path.basename(concept_dir),
                "base_filename": base,
            }
        )

    logger.info(
        f"  Concept {concept_id}: saved {len(csv_rows)} images to {concept_dir}"
    )
    return csv_rows


# ==============================================================================
# Main Entry Point
# ==============================================================================


def get_visualization_args():
    parser = argparse.ArgumentParser(description="SAE Concept Activation Visualization")

    # Required paths
    parser.add_argument(
        "--gap_csv",
        type=str,
        default="",
        help="Path to class-wise GAP means CSV (optional if --features_cache used)",
    )
    parser.add_argument(
        "--sae_ckpt", type=str, required=True, help="Path to trained SAE checkpoint"
    )
    parser.add_argument(
        "--model_state_path",
        type=str,
        required=True,
        help="Path to CNN backbone weights",
    )
    parser.add_argument(
        "--shard_root", type=str, required=True, help="Path to sharded image data"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory containing train/val/test split CSVs",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for visualizations",
    )

    # Concept selection
    parser.add_argument(
        "--concept_ids",
        type=str,
        default="de_filter",
        help="Comma-separated concept IDs, 'all_alive', 'gini_filter', "
        "'de_filter' (features_cache), or 'de_filter_csv' (gap_csv log2FC)",
    )
    parser.add_argument(
        "--max_gini",
        type=float,
        default=0.3,
        help="Max Gini coefficient for class-specific filtering (lower = more specific)",
    )
    parser.add_argument(
        "--max_concepts",
        type=int,
        default=0,
        help="Maximum number of concepts to visualize (0 = all)",
    )

    # DE filter options
    parser.add_argument(
        "--features_cache",
        type=str,
        default="",
        help="Path to features_cache.npz (required for de_filter)",
    )
    parser.add_argument(
        "--cell_death_csv",
        type=str,
        default="",
        help="Path to cell_death CSV with line/UID info (required for de_filter)",
    )
    parser.add_argument(
        "--min_cv",
        type=float,
        default=0.2,
        help="Min CV for neuron filtering in de_filter mode",
    )
    parser.add_argument(
        "--de_min_log2fc", type=float, default=1.5, help="Min |log2FC| for DE filtering"
    )
    parser.add_argument(
        "--de_adj_p", type=float, default=0.05, help="Adjusted p-value threshold for DE"
    )
    parser.add_argument(
        "--dead_threshold",
        type=float,
        default=1e-5,
        help="Dead neuron threshold: neurons with activation_rate "
        "< this value are excluded. Uses total_active_imgs / "
        "total_images from gap_csv. Default: 1e-5",
    )

    # Visualization options
    parser.add_argument(
        "--top_k", type=int, default=20, help="Number of top images per concept"
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="jet",
        help="Colormap name (jet, hot, viridis, etc.)",
    )
    parser.add_argument(
        "--mut_only",
        action="store_true",
        help="Only visualize mutation-high concepts (skip Control-only, remap Control_X→X)",
    )
    parser.add_argument(
        "--overlay_alpha", type=float, default=0.5, help="Overlay blend alpha (0-1)"
    )
    parser.add_argument(
        "--base_alpha",
        type=float,
        default=1.0,
        help="Base image alpha (0-1) to darken the background",
    )

    # Model config
    parser.add_argument(
        "--which_layer",
        type=str,
        default="stage5_out",
        help="Layer to extract features from",
    )
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)

    # Architecture params (should match training)
    parser.add_argument("--blocks", type=str, default="2,2,2,3")
    parser.add_argument("--dilations", type=str, default="1,1,1,1")
    parser.add_argument("--refine_blocks", type=int, default=1)
    parser.add_argument("--ckpt_segments", type=int, default=0)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--proj_layers", type=int, default=2)
    parser.add_argument("--proj_hidden", type=int, default=2048)
    parser.add_argument("--proj_bn", type=int, default=0)
    parser.add_argument("--proj_dropout", type=float, default=0.0)

    return parser.parse_args()


def main():
    args = get_visualization_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ===== Load GAP CSV (optional) =====
    gap_info = {}
    alive_concepts = []
    if args.gap_csv and os.path.exists(args.gap_csv):
        logger.info(f"Loading GAP CSV: {args.gap_csv}")
        gap_info = load_gap_csv(args.gap_csv, dead_threshold=args.dead_threshold)
        logger.info(f"  Total concepts: {len(gap_info)}")
        alive_concepts = [cid for cid, info in gap_info.items() if info["is_alive"]]
        logger.info(f"  Alive concepts: {len(alive_concepts)}")
    else:
        logger.info("No GAP CSV provided")
        if args.features_cache:
            if args.concept_ids == "gini_filter":
                logger.info(
                    "  → Computing GAP info from features_cache for gini_filter"
                )
                gap_info = compute_gap_info_from_cache(
                    args.features_cache, dead_threshold=args.dead_threshold
                )
                alive_concepts = [
                    cid for cid, info in gap_info.items() if info["is_alive"]
                ]
                logger.info(f"  Alive concepts: {len(alive_concepts)}")
            else:
                logger.info("  → Using features_cache for DE-based concept selection")
                if args.concept_ids not in ("de_filter",):
                    logger.info("  → Auto-switching concept_ids to 'de_filter'")
                    args.concept_ids = "de_filter"

    # ===== Parse concept IDs =====
    concept_class_labels = {}  # concept_id -> dominant class label (for DE mode)
    if args.concept_ids == "all_alive":
        concept_ids = (
            alive_concepts
            if args.max_concepts == 0
            else alive_concepts[: args.max_concepts]
        )
        logger.info(f"  Using first {len(concept_ids)} alive concepts")
    elif args.concept_ids == "gini_filter":
        # Filter by Gini coefficient (class-specific concepts)
        filtered = filter_concepts_by_gini(gap_info, max_gini=args.max_gini)
        logger.info(
            f"  Gini filter (max_gini={args.max_gini}): found {len(filtered)} class-specific concepts"
        )

        if len(filtered) == 0:
            logger.warning("  No concepts pass Gini filter! Try increasing --max_gini")
            return

        # Log top concepts
        for cid, gini, max_cls in filtered[:10]:
            logger.info(f"    Concept {cid}: Gini={gini:.4f}, max_class={max_cls}")
        if len(filtered) > 10:
            logger.info(f"    ... and {len(filtered) - 10} more")

        concept_ids = (
            [x[0] for x in filtered]
            if args.max_concepts == 0
            else [x[0] for x in filtered[: args.max_concepts]]
        )
    elif args.concept_ids == "de_filter":
        # DE-based class-specific concept selection (requires features_cache)
        if not args.features_cache or not args.cell_death_csv:
            logger.error("de_filter requires --features_cache and --cell_death_csv")
            return
        de_concepts = select_concepts_by_de(
            args.features_cache,
            args.cell_death_csv,
            min_cv=args.min_cv,
            de_adj_p=args.de_adj_p,
            de_min_log2fc=args.de_min_log2fc,
        )
        if len(de_concepts) == 0:
            logger.warning("No concepts pass DE filter!")
            return

        # Log selected concepts
        for cid, cls, fc, direction in de_concepts[:20]:
            logger.info(f"    Concept {cid}: {cls} (log2fc={fc:.2f}, {direction})")
        if len(de_concepts) > 20:
            logger.info(f"    ... and {len(de_concepts) - 20} more")

        # Build concept_ids and class_labels mapping
        if args.max_concepts > 0:
            de_concepts = de_concepts[: args.max_concepts]
        concept_ids = [x[0] for x in de_concepts]
        concept_class_labels = {x[0]: x[1] for x in de_concepts}
        logger.info(f"  DE filter: {len(concept_ids)} concepts selected")
    elif args.concept_ids == "de_filter_csv":
        # DE-based selection using gap_csv only (log2FC, no Wilcoxon)
        if not gap_info:
            logger.error("de_filter_csv requires --gap_csv")
            return
        de_concepts = select_concepts_by_gap_csv_de(
            gap_info,
            max_gini=args.max_gini,
            de_min_log2fc=args.de_min_log2fc,
        )
        if len(de_concepts) == 0:
            logger.warning(
                "No concepts pass DE filter (CSV)! "
                "Try increasing --max_gini or decreasing --de_min_log2fc"
            )
            return

        for cid, cls, fc, direction in de_concepts[:20]:
            logger.info(f"    Concept {cid}: {cls} (log2fc={fc:.2f}, {direction})")
        if len(de_concepts) > 20:
            logger.info(f"    ... and {len(de_concepts) - 20} more")

        if args.max_concepts > 0:
            de_concepts = de_concepts[: args.max_concepts]
        concept_ids = [x[0] for x in de_concepts]
        concept_class_labels = {x[0]: x[1] for x in de_concepts}
        logger.info(f"  DE filter (CSV): {len(concept_ids)} concepts selected")
    else:
        concept_ids = [int(x.strip()) for x in args.concept_ids.split(",")]
        logger.info(f"  Using specified concepts: {concept_ids}")

    # ===== Load SAE =====
    logger.info(f"Loading SAE: {args.sae_ckpt}")
    ckpt = torch.load(args.sae_ckpt, map_location="cpu", weights_only=False)
    ckpt_args = ckpt["args"]

    sae = GatedSAE(
        d_in=ckpt_args.get("d_in", 512),
        d_sae=ckpt_args.get("d_sae", 4096),
        tie_weights=ckpt_args.get("tie_gate_weights", False),
        aux_k=ckpt_args.get("aux_k", 32),
    )
    sae.load_state_dict(ckpt["sae"])
    sae.eval().to(device)
    logger.info(f"  SAE: d_in={sae.d_in}, d_sae={sae.d_sae}")

    # ===== Load Encoder =====
    logger.info(f"Loading encoder: {args.model_state_path}")
    blocks = parse_int_list(args.blocks, 4)
    dilations = parse_int_list(args.dilations, 4)

    model = SupMoCoModel(
        embed_dim=args.embed_dim,
        blocks=blocks,
        dilations=dilations,
        refine_blocks=args.refine_blocks,
        ckpt_segments=args.ckpt_segments,
        proj_layers=args.proj_layers,
        proj_hidden=args.proj_hidden,
        proj_bn=bool(args.proj_bn),
        proj_dropout=args.proj_dropout,
    )
    sd = torch.load(args.model_state_path, map_location="cpu", weights_only=False)
    robust_load_state_dict(model, sd, strict=True)
    encoder = model.encoder
    renorm_unit_per_out_channel_(encoder)
    encoder.eval().to(device).to(memory_format=torch.channels_last)

    del model
    del sd

    # ===== Load Data (val + test) =====
    logger.info("Loading image shards...")
    refs = load_all_sample_refs(args.shard_root)
    uid_to_refidx = build_uid_to_refidx(refs)

    # Build image-name → refidx mapping (handles path mismatches between machines)
    # UID format: '/path/to.tar:IMAGE_NAME' — match by IMAGE_NAME only
    name_to_refidx = {}
    for full_uid, ridx in uid_to_refidx.items():
        if ":" in full_uid:
            img_name = full_uid.split(":")[-1]
            name_to_refidx[img_name] = ridx
        name_to_refidx[full_uid] = ridx  # also keep full path match

    # Load val + test UIDs
    eval_uids = []
    for split_name in ["val_split.csv", "test_split.csv"]:
        split_path = os.path.join(args.save_dir, split_name)
        if os.path.exists(split_path):
            eval_uids.extend(load_split_csv(split_path))

    logger.info(f"  Eval images (val+test): {len(eval_uids)}")

    # Build bank — match by image name if full path fails
    eval_refidx = []
    for u in eval_uids:
        if u in uid_to_refidx:
            eval_refidx.append(uid_to_refidx[u])
        elif ":" in u:
            img_name = u.split(":")[-1]
            if img_name in name_to_refidx:
                eval_refidx.append(name_to_refidx[img_name])

    logger.info(f"  Matched eval images: {len(eval_refidx)}/{len(eval_uids)}")
    bank = InMemoryTarBank(refs, eval_refidx, args.img_size)
    bank_indices = list(range(len(eval_refidx)))

    # DEBUG: check label mapping
    from collections import Counter

    if len(bank.labels) > 0:
        sample_labels = bank.labels[: min(100, len(bank.labels))]
        from run_CNN.logging_utils import SUPERCLASS_MAP

        label_counts = Counter(sample_labels)
        mapped = Counter(SUPERCLASS_MAP.get(l, f"UNMAPPED:{l}") for l in sample_labels)
        logger.info(f"  DEBUG bank.labels sample: {dict(label_counts)}")
        logger.info(f"  DEBUG mapped superclasses: {dict(mapped)}")
    else:
        logger.warning("  DEBUG bank.labels is EMPTY!")

    # ===== Phase 1: Precompute ALL GAP values in a single pass =====
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(
        f"\nPhase 1: Precomputing GAP for ALL concepts ({len(concept_ids)} selected)..."
    )
    gap_all, valid_indices = precompute_all_gap_values(
        encoder,
        sae,
        bank,
        bank_indices,
        device,
        which_layer=args.which_layer,
        batch_size=args.batch_size,
    )
    logger.info(f"  GAP matrix: {gap_all.shape} (images x concepts)")

    # ===== Phase 2: Determine top-K per concept & collect unique images =====
    logger.info("Phase 2: Selecting top-K images per concept...")
    LABEL_TO_CLASS = {0: "Control", 1: "SNCA", 2: "GBA", 3: "LRRK2"}
    image_concept_map = {}  # bank_index -> set of concept_ids
    per_concept_data = {}  # cid -> (gap_1d, filtered_valid)

    for cid in concept_ids:
        cls_label = concept_class_labels.get(cid, "")
        cid_gap = gap_all[:, cid].copy()
        cid_valid = list(valid_indices)

        # Class filtering
        filter_classes = []
        if cls_label and cls_label != "Control":
            filter_classes = cls_label.split("_")
        if filter_classes:
            class_mask = np.array(
                [
                    LABEL_TO_CLASS.get(bank.labels[cid_valid[i]], "") in filter_classes
                    for i in range(len(cid_valid))
                ]
            )
            if class_mask.sum() == 0:
                logger.warning(
                    f"  Concept {cid} ({cls_label}): no images from {filter_classes}"
                )
                continue
            cid_gap = cid_gap[class_mask]
            cid_valid = [cid_valid[i] for i in range(len(class_mask)) if class_mask[i]]
            logger.info(
                f"  Concept {cid} ({cls_label}): filtered to {len(cid_valid)} images"
            )

        # Determine top-K
        sorted_idx = np.argsort(cid_gap)[::-1]
        sorted_idx = sorted_idx[cid_gap[sorted_idx] > 1e-6]
        top_indices = sorted_idx[: args.top_k]
        if len(top_indices) == 0:
            logger.warning(f"  Concept {cid}: no images with nonzero GAP")
            continue

        per_concept_data[cid] = (cid_gap, cid_valid)
        for idx in top_indices:
            bi = cid_valid[idx]
            if bi not in image_concept_map:
                image_concept_map[bi] = set()
            image_concept_map[bi].add(cid)

    image_concept_map = {bi: list(cids) for bi, cids in image_concept_map.items()}
    logger.info(
        f"  Concepts with data: {len(per_concept_data)}, "
        f"unique images for heatmaps: {len(image_concept_map)}"
    )

    # ===== Phase 3: Batch compute activation maps for top-K images =====
    logger.info("Phase 3: Batch computing activation maps...")
    act_cache = batch_compute_activation_maps(
        encoder,
        sae,
        bank,
        image_concept_map,
        device,
        which_layer=args.which_layer,
        batch_size=args.batch_size,
    )
    logger.info(f"  Cached {len(act_cache)} activation maps")

    # ===== Phase 4: Visualize (local first, then copy to Drive) =====

    # If output_dir is on Google Drive, save locally first to avoid FUSE file loss
    final_output_dir = args.output_dir
    is_drive = "/drive/" in args.output_dir or "/content/drive/" in args.output_dir
    if is_drive:
        local_output_dir = os.path.join(tempfile.gettempdir(), "concept_vis_local")
        if os.path.exists(local_output_dir):
            shutil.rmtree(local_output_dir)
        os.makedirs(local_output_dir, exist_ok=True)
        logger.info(f"  Saving to local disk first: {local_output_dir}")
    else:
        local_output_dir = final_output_dir

    # ── mut_only: filter out Control-only, remap Control_X → X ──
    if args.mut_only and concept_class_labels:
        _MUTS = {"SNCA", "GBA", "LRRK2"}
        filtered_ids = []
        for cid in concept_ids:
            raw = concept_class_labels.get(cid, "")
            parts = set(raw.split("_"))
            muts = parts & _MUTS
            if len(muts) == 0:
                continue  # pure Control → skip
            # Remap: keep only mutation parts
            new_label = "_".join(sorted(muts))
            concept_class_labels[cid] = new_label
            filtered_ids.append(cid)
        n_dropped = len(concept_ids) - len(filtered_ids)
        concept_ids = filtered_ids
        logger.info(
            f"  --mut_only: {n_dropped} Control-only concepts dropped, "
            f"{len(concept_ids)} remaining"
        )

    logger.info(
        f"\nPhase 4: Generating visualizations for {len(per_concept_data)} concepts..."
    )
    all_csv_rows: List[Dict] = []
    for cid in tqdm(concept_ids, desc="Visualizing"):
        if cid not in per_concept_data:
            continue
        cid_gap, cid_valid = per_concept_data[cid]
        cls_label = concept_class_labels.get(cid, "")
        rows = visualize_concept(
            encoder,
            sae,
            bank,
            cid,
            cid_valid,
            cid_gap,
            top_k=args.top_k,
            output_dir=local_output_dir,
            device=device,
            which_layer=args.which_layer,
            img_size=args.img_size,
            cmap_name=args.cmap,
            overlay_alpha=args.overlay_alpha,
            base_alpha=args.base_alpha,
            act_cache=act_cache,
            class_label=cls_label,
        )
        all_csv_rows.extend(rows)

    # ===== Save top-K CSV =====
    # Columns: concept_id, concept_class, rank, img_name, line, gap_val, max_act,
    #          concept_dir, base_filename
    # Usage:
    #   df = pd.read_csv('top_k_images.csv')
    #   union_imgs = df['img_name'].unique()          # all unique images
    #   top_imgs   = df[df['rank'] <= 5]['img_name'].unique()  # rank 1-5 only
    #   → Feed union_imgs into step15_cnn_featuremap.py to generate
    #     one grid PNG (512 channels x bilinear interp) per image,
    #     named by img_name — so step14 SAE overlays and step15 CNN maps
    #     are automatically paired by file name without any manual search.
    if all_csv_rows:
        csv_path = os.path.join(local_output_dir, "top_k_images.csv")
        _csv_fields = [
            "concept_id",
            "concept_class",
            "rank",
            "img_name",
            "line",
            "gap_val",
            "max_act",
            "concept_dir",
            "base_filename",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_csv_fields)
            writer.writeheader()
            writer.writerows(all_csv_rows)
        # Also save union image list (unique img_names across all concepts)
        union_names = sorted(set(r["img_name"] for r in all_csv_rows))
        union_path = os.path.join(local_output_dir, "union_images.txt")
        with open(union_path, "w", encoding="utf-8") as f:
            f.write("\n".join(union_names))
        logger.info(f"  Saved CSV: {csv_path} ({len(all_csv_rows)} rows)")
        logger.info(f"  Union images: {len(union_names)} unique images → {union_path}")

    # Copy local results to Drive
    if is_drive and local_output_dir != final_output_dir:
        logger.info(f"\nCopying results to Drive: {final_output_dir}")
        if os.path.exists(final_output_dir):
            shutil.rmtree(final_output_dir)
        shutil.copytree(local_output_dir, final_output_dir)
        n_files = sum(len(files) for _, _, files in os.walk(final_output_dir))
        logger.info(f"  Copied {n_files} files to Drive")
        shutil.rmtree(local_output_dir)

    logger.info(f"\nDone! Output -> {final_output_dir}")


if __name__ == "__main__":
    main()
