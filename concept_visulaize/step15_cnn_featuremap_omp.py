"""
step15_cnn_featuremap_omp.py
============================
"CNN 채널 하나가 여러 SAE concept의 합으로 표현된다" 시각화.

목적:
  SAE concept 기준으로 해석 잘 되는 이미지들을 선택하고,
  그 이미지를 CNN에 넣었을 때 '가장 활성화된 CNN 채널들'이
  실제로는 '여러 SAE concept이 혼재된 polysemantic 채널'임을 보임.

  = CNN superposition의 직접적 시각 증거.

CNN 채널 선택 기준: IoU 기반 아님 → 이미지 내 GAP 값 기준 (객관적)
  gap[ch] = mean(cnn_maps[ch]) → top-N 선택

구조:
  SAE concept 20개 × 이미지 top-k 10개 × CNN 채널 top 10개
  = 총 2000개 CNN 채널 OMP 블록

저장 (per concept, per image 파일):
  output_dir/
    concept_{cid}_{class}/
      {class}_{img_name}__rank{r}_omp.png    ← 이미지 1장당 1파일
                                                (top-N CNN 채널 블록 수직 적층)
      summary_panel.png
      polysemanticity.csv
  omp_log_all.csv

파일 레이아웃 (1개 이미지):
  ╔═══════════ Reference ════════════╗
  ║ [orig] [SAE#cid overlay] [binary]║
  ╠═══════════ CNN ch{47} ════════════╗  ← GAP rank #1 (가장 활성화된 채널)
  ║ [CNN overlay] [CNN binary]       ║
  ║ indiv: [SAE#A] [SAE#B] [SAE#C]  ║  ← OMP 순서대로
  ║ cumul: [1]     [1+2]   [1+2+3]  ║  ← 누적 (coverage% 기록)
  ╠═══════════ CNN ch{203} ═══════════╣  ← GAP rank #2
  ║ ...                              ║
  ╚══════════════════════════════════╝

Usage (Colab):
  !python -m concept_visulaize.step15_cnn_featuremap_omp \\
      --csv ".../top_k_images.csv" \\
      --concept_ids "441,3563,4080,833,1416,1434,2567,824,1322,1636" \\
      --top_k 10 \\
      --top_cnn_by_gap 10 \\
      --n_select 8 \\
      --model_state_path ".../best_model.pt" \\
      --sae_ckpt ".../stage5_out_d4096_...pt" \\
      --shard_root "/content/wds_shards" \\
      --save_dir ".../MoCo_seed87" \\
      --output_dir ".../cnn_omp_results"
"""

# CNN activation feature map에서 median 이하 다 0으로 남겨서 binary 만들고 나머지 애들에 대해서 OMP soft IOU 기준으로 선택한다.


import os, csv, argparse, shutil, tempfile
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

try:
    import matplotlib.cm as _cm
except ImportError:
    raise RuntimeError("pip install matplotlib")

from sae_project.step02_logging_utils import get_logger
from sae_project.step03_data_shards import load_all_sample_refs, build_uid_to_refidx
from sae_project.step04_data_bank import InMemoryTarBank, SafeInstanceNormalize
from sae_project.step05_model_encoder import (
    SupMoCoModel, parse_int_list, renorm_unit_per_out_channel_, robust_load_state_dict,
)
from sae_project.step06_gated_sae import GatedSAE

logger = get_logger("step15_omp")


# ══════════════════════════════════════════════════════════════════
# Viz helpers
# ══════════════════════════════════════════════════════════════════

def _cmap(a01: np.ndarray, name: str = "jet") -> np.ndarray:
    a01 = np.clip(a01.astype(np.float32), 0, 1)
    return (_cm.get_cmap(name)(a01)[..., :3] * 255).astype(np.uint8)

def _overlay(base: np.ndarray, heat: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    return np.clip(base * (1 - alpha) + heat * alpha, 0, 255).astype(np.uint8)

def _fiji(img_u16: np.ndarray) -> np.ndarray:
    out = np.zeros_like(img_u16, dtype=np.uint8)
    img = img_u16.astype(np.float32)
    for c in range(3):
        ch = img[..., c]
        if np.std(ch) < 655.0:
            out[..., c] = np.clip(ch / 65535 * 255, 0, 255).astype(np.uint8)
        else:
            lo, hi = np.percentile(ch, 10), np.percentile(ch, 99.5)
            if hi > lo:
                out[..., c] = np.clip((ch - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    return out

def _label(arr: np.ndarray, text: str, color=(255, 255, 0)) -> np.ndarray:
    pil = Image.fromarray(arr)
    ImageDraw.Draw(pil).text((2, 2), text, fill=color)
    return np.array(pil)

def _up(arr2d: np.ndarray, S: int) -> np.ndarray:
    t = torch.from_numpy(arr2d.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    return F.interpolate(t, (S, S), mode="bilinear", align_corners=False).squeeze().numpy()

def _resize(arr: np.ndarray, S: int) -> np.ndarray:
    return np.array(Image.fromarray(arr).resize((S, S)))

def _norm01(arr: np.ndarray, plo: float = 50.0, phi: float = 99.9) -> np.ndarray:
    lo = float(np.percentile(arr, plo))
    hi = float(np.percentile(arr, phi))
    return np.clip((arr - lo) / max(hi - lo, 1e-8), 0, 1)

def _pad_w(arr: np.ndarray, w: int) -> np.ndarray:
    if arr.shape[1] < w:
        pad = np.zeros((arr.shape[0], w - arr.shape[1], 3), dtype=np.uint8)
        return np.concatenate([arr, pad], axis=1)
    return arr

def _hsep(w: int, h: int = 2, v: int = 80) -> np.ndarray:
    return np.full((h, w, 3), v, dtype=np.uint8)


# ══════════════════════════════════════════════════════════════════
# OMP-IoU: CNN channel → SAE concept list
# ══════════════════════════════════════════════════════════════════

def _norm_map(arr2d: np.ndarray, plo: float = 50.0, phi: float = 99.9) -> np.ndarray:
    """Normalize a single 2D activation map: median → 0, 99.9th pct → 1 (same as step14)."""
    lo = float(np.percentile(arr2d, plo))
    hi = float(np.percentile(arr2d, phi))
    return np.clip((arr2d - lo) / max(hi - lo, 1e-8), 0, 1).astype(np.float32)


def omp_cnn_to_sae_soft(
    cnn_norm:       np.ndarray,    # (Hf, Wf) float [0,1] — CNN normalized (median-zeroed)
    sae_norm_stack: np.ndarray,    # (K, Hf, Wf) float [0,1] — SAE normalized (median-zeroed)
    candidate_cids: List[int],
    n:              int = 8,
) -> Tuple[List[int], List[float]]:
    """
    Soft/continuous OMP — uses actual activation VALUES (not binary masks).

    Why this fixes the binary-threshold problem:
      Binary median threshold: 50% of pixels always selected, including background.
      Soft IoU: pixels near 0 (background, after median subtraction) contribute ≈0
               to both numerator and denominator → background ignored naturally.

    Metric:
      soft_IoU = Σ min(residual, sae) / Σ max(residual, sae)
      • Strong overlap in bright regions → high soft_IoU
      • Background (≈0) → contributes ≈0 regardless of how many background pixels

    Residual update:
      residual ← max(residual - sae_selected, 0)
      • Subtracts what the selected SAE explained, leaving remainder
      • Continuous: after round 1 covers nucleus (0.8→0), round 2 focuses on lysosome
    """
    K        = sae_norm_stack.shape[0]
    residual = cnn_norm.copy().astype(np.float32)
    excl     = np.zeros(K, dtype=bool)
    sel:  List[int]   = []
    ious: List[float] = []

    for _ in range(n):
        if residual.max() < 1e-6:
            break
        # Soft IoU: Σ min / Σ max  (vectorised over K)
        soft_inter = np.minimum(residual[None], sae_norm_stack).sum(axis=(1, 2))  # (K,)
        soft_union = np.maximum(residual[None], sae_norm_stack).sum(axis=(1, 2))  # (K,)
        soft_iou   = soft_inter / np.maximum(soft_union, 1e-8)
        soft_iou[excl] = -1.0

        best_k   = int(np.argmax(soft_iou))
        best_iou = float(soft_iou[best_k])
        if best_iou < 1e-6:
            break

        sel.append(candidate_cids[best_k])
        ious.append(round(best_iou, 4))
        excl[best_k] = True
        # Soft residual: subtract what SAE explained, clip to 0
        residual = np.maximum(residual - sae_norm_stack[best_k], 0)

    return sel, ious


def n_needed_for_coverage_soft(
    cnn_norm:       np.ndarray,    # (Hf, Wf) float [0,1]
    sae_norm_stack: np.ndarray,    # (K, Hf, Wf) float [0,1]
    candidate_cids: List[int],
    sel_cids:       List[int],
    threshold:      float = 0.80,
) -> int:
    """
    Soft coverage: Σ min(covered_union, cnn_norm) / Σ cnn_norm >= threshold.
    '80% soft coverage' = 80% of CNN's activation energy explained by selected SAE concepts.
    """
    total_cnn = max(float(cnn_norm.sum()), 1e-8)
    covered   = np.zeros_like(cnn_norm)
    c2i       = {c: i for i, c in enumerate(candidate_cids)}
    for k, cid in enumerate(sel_cids, 1):
        if cid not in c2i:
            continue
        covered = np.maximum(covered, sae_norm_stack[c2i[cid]])  # soft union
        cov     = float(np.minimum(covered, cnn_norm).sum()) / total_cnn
        if cov >= threshold:
            return k
    return -1


# ══════════════════════════════════════════════════════════════════
# Feature extraction — ONE forward pass per image
# ══════════════════════════════════════════════════════════════════

@torch.inference_mode()
def extract_all_maps(
    encoder: nn.Module,
    sae:     GatedSAE,
    img_u16: np.ndarray,
    device:  torch.device,
    which_layer: str = "stage5_out",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        cnn_maps : (512, Hf, Wf)    — CNN feature maps (post-GAP-norm)
        sae_all  : (Hf, Wf, d_sae) — all SAE activations (slice by concept_id)
    """
    norm = SafeInstanceNormalize(threshold=0.01)
    akw  = dict(device_type="cuda", enabled=torch.cuda.is_available())
    if torch.cuda.is_available():
        akw["dtype"] = torch.bfloat16

    x = torch.from_numpy(img_u16.astype(np.float32) / 65535.0).permute(2, 0, 1)
    x = norm(x).unsqueeze(0).to(device).contiguous(memory_format=torch.channels_last)

    with torch.amp.autocast(**akw):
        fmap = encoder.forward_feature_maps(x, which=which_layer)   # (1, 512, Hf, Wf)

    gap_norm = (fmap.mean(dim=(2, 3)).norm(dim=1, keepdim=True)
                .view(1, 1, 1, 1).clamp_min(1e-12))
    fmap /= gap_norm

    _, C, Hf, Wf = fmap.shape
    cnn_maps = fmap.squeeze(0).float().cpu().numpy()                   # (512, Hf, Wf)

    flat   = fmap.permute(0, 2, 3, 1).reshape(-1, C)
    flat   = flat - flat.mean(dim=0, keepdim=True)
    tok_l2 = flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
    flat   = F.normalize(flat, dim=1, eps=1e-12)

    with torch.amp.autocast(**akw):
        _, acts, _, _, _ = sae(flat)

    sae_all = (acts.float() * tok_l2).view(Hf, Wf, -1).cpu().numpy()  # (Hf, Wf, d_sae)
    return cnn_maps, sae_all


# ══════════════════════════════════════════════════════════════════
# Figure builders
# ══════════════════════════════════════════════════════════════════

def _overlay_median_zero(arr2d: np.ndarray, orig_s: np.ndarray,
                          S: int, cmap_name: str, alpha: float) -> np.ndarray:
    """
    step14-style overlay — ALL statistics at NATIVE (Hf, Wf) resolution.
    Bilinear upsample to S×S happens ONLY at the very end (display only).

      1. Normalize at native res: median → 0, 99.9th pct → 1  (via _norm_map)
      2. Upsample normalized map to S×S  ← display only, no stat computed here
      3. Apply colormap → overlay on original image
    """
    n_native = _norm_map(arr2d)          # (Hf, Wf) [0,1] — stats at native res
    up       = _up(n_native, S)          # upsample ONLY for display
    return _overlay(orig_s, _cmap(up, cmap_name), alpha)


def make_cnn_omp_block(
    orig_s:         np.ndarray,    # (S, S, 3)
    cnn_map_ch:     np.ndarray,    # (Hf, Wf)  — CNN channel to decompose
    cnn_b_ch:       np.ndarray,    # (Hf, Wf) bool   (used only for OMP, not displayed raw)
    gap_rank:       int,
    gap_val:        float,
    sae_all:        np.ndarray,    # (Hf, Wf, d_sae)
    sae_b_stack:    np.ndarray,    # (K, Hf, Wf) bool
    candidate_cids: List[int],
    sel_cids:       List[int],     # OMP result
    sel_ious:       List[float],
    concept_class_map: Dict[int, str],
    cnn_ch:         int,
    n_sae_80pct:    int,
    S:              int   = 128,
    cmap_name:      str   = "jet",
    alpha:          float = 0.5,
) -> np.ndarray:

    """
    3-row block for ONE CNN channel — all panels use step14-style overlay:
      (median 이하 → 0, normalize, heatmap overlay on original)

    OMP 계산 자체: binary mask (median threshold) → IoU 계산 (변경 없음)
    표시만 overlay 방식으로 통일.

    Row 0 (header):  [CNN ch overlay]  [orig]    (median-zeroed heatmap, same as step14)
    Row 1 (indiv):   [SAE#A overlay]   [SAE#B overlay]  ...   (OMP 순서, IoU 표시)
    Row 2 (cumul):   [cumul SAE 1-1]   [cumul SAE 1-2]  ...   (누적합 overlay + cov%)
    """
    N      = len(sel_cids)
    n_cols = max(N + 2, 4)
    blank  = np.zeros((S, S, 3), dtype=np.uint8)

    # ── Row 0: CNN channel header ──
    cnn_ovl = _overlay_median_zero(cnn_map_ch, orig_s, S, cmap_name, alpha)
    h1 = _label(cnn_ovl, f"CNN ch{cnn_ch}\nGAP#{gap_rank}={gap_val:.3f}\nSAE×80%={n_sae_80pct}")
    h2 = _label(orig_s.copy(), "orig")
    hdr_cells = [h1, h2] + [blank] * max(0, n_cols - 2)
    row0 = np.concatenate(hdr_cells[:n_cols], axis=1)

    # ── Row 1: individual SAE overlays (OMP 선택 순서) ──
    # Soft IoU 기준으로 선택된 SAE — step14 스타일 overlay
    # IoU: 이번 round의 soft IoU (연속값 기준이므로 배경 오염 없음)
    sae_cells = [blank, blank]
    for k, (cid, iou_v) in enumerate(zip(sel_cids, sel_ious), 1):
        sae_m   = sae_all[:, :, cid]
        sae_ovl = _overlay_median_zero(sae_m, orig_s, S, cmap_name, alpha)
        cls_lbl = concept_class_map.get(cid, "")
        sae_cells.append(_label(sae_ovl,
                                f"#{k} SAE{cid}\n({cls_lbl})\nsoftIoU={iou_v:.2f}"))
    while len(sae_cells) < n_cols:
        sae_cells.append(blank)
    row1 = np.concatenate(sae_cells[:n_cols], axis=1)

    # ── Row 2: cumulative overlay + soft coverage% ──
    # soft coverage = Σ min(covered_union, cnn_norm) / Σ cnn_norm
    # 배경(≈0) 픽셀은 기여 없음 → 실제 활성 영역의 몇 %를 누적 SAE가 설명하는지
    cnn_norm_ch = _norm_map(cnn_map_ch)                    # (Hf, Wf) [0,1]
    total_cnn   = max(float(cnn_norm_ch.sum()), 1e-8)
    covered_soft = np.zeros_like(cnn_norm_ch)
    cumul_sae    = np.zeros_like(sae_all[:, :, 0])
    cumul_cells  = [blank, blank]
    for k, cid in enumerate(sel_cids, 1):
        sae_m        = sae_all[:, :, cid]
        sae_n        = _norm_map(sae_m)                    # (Hf, Wf) [0,1]
        cumul_sae    = cumul_sae + np.maximum(sae_m, 0)
        covered_soft = np.maximum(covered_soft, sae_n)    # soft union = element-wise max
        cum_ovl      = _overlay_median_zero(cumul_sae, orig_s, S, cmap_name, alpha)
        cov_pct      = 100.0 * float(np.minimum(covered_soft, cnn_norm_ch).sum()) / total_cnn
        cumul_cells.append(_label(cum_ovl, f"cumul 1-{k}\ncov={cov_pct:.0f}%"))
    while len(cumul_cells) < n_cols:
        cumul_cells.append(blank)
    row2 = np.concatenate(cumul_cells[:n_cols], axis=1)

    target_w = S * n_cols
    thin = _hsep(target_w, h=1, v=60)
    return np.concatenate([
        _pad_w(row0, target_w), thin,
        _pad_w(row1, target_w), thin,
        _pad_w(row2, target_w),
    ], axis=0)



def make_image_figure(
    orig_s:       np.ndarray,      # (S, S, 3)
    sae_cid:      int,
    sae_map:      np.ndarray,      # (Hf, Wf) — SAE concept activation at native res
    gap_val_csv:  float,
    cls_label:    str,
    cnn_blocks:   List[np.ndarray],
    S:            int   = 128,
    cmap_name:    str   = "jet",
    alpha:        float = 0.5,
) -> np.ndarray:
    """
    Full figure for one (SAE concept, image) pair:
      [Reference header]
      [CNN block #1 (GAP rank 1)]
      [CNN block #2 (GAP rank 2)]
      ...
    """
    # Reference SAE header — ALL stats at native (Hf, Wf), upsample for display only
    sae_ovl = _overlay_median_zero(sae_map, orig_s, S, cmap_name, alpha)

    h1 = _label(orig_s,  "orig")
    h2 = _label(sae_ovl, f"SAE#{sae_cid} ({cls_label})\ngap={gap_val_csv:.3f}")

    if not cnn_blocks:
        return np.concatenate([h1, h2], axis=1)

    max_w = max(b.shape[1] for b in cnn_blocks)
    hdr   = _pad_w(np.concatenate([h1, h2], axis=1), max_w)
    thick = _hsep(max_w, h=5, v=180)  # bright thick separator
    thin  = _hsep(max_w, h=2, v=90)

    parts = [hdr, thick]
    for i, blk in enumerate(cnn_blocks):
        parts.append(_pad_w(blk, max_w))
        if i < len(cnn_blocks) - 1:
            parts.append(thin)
    return np.concatenate(parts, axis=0)


# ══════════════════════════════════════════════════════════════════
# Argument parser
# ══════════════════════════════════════════════════════════════════

def get_args():
    p = argparse.ArgumentParser(
        description="CNN polysemanticity: for SAE-interpretable images, show that "
                    "the most active CNN channels require multiple SAE concepts to explain.")
    p.add_argument("--csv",          type=str, required=True,
                   help="top_k_images.csv from step14")
    p.add_argument("--concept_ids",  type=str, required=True,
                   help="SAE concept IDs for image selection AND OMP candidate pool.\n"
                        "e.g. '441,3563,4080,833,1416'")
    p.add_argument("--top_k",        type=int, default=10,
                   help="Images per concept from CSV")
    p.add_argument("--top_cnn_by_gap", type=int, default=10,
                   help="CNN channels to show per image, selected by GAP rank "
                        "(highest channel-wise mean activation in that image)")
    p.add_argument("--n_select",     type=int, default=8,
                   help="OMP rounds: how many SAE concepts to use per CNN channel")
    p.add_argument("--model_state_path", type=str, required=True)
    p.add_argument("--sae_ckpt",     type=str, required=True)
    p.add_argument("--shard_root",   type=str, required=True)
    p.add_argument("--save_dir",     type=str, required=True)
    p.add_argument("--output_dir",   type=str, required=True)
    p.add_argument("--img_size",     type=int,   default=128)
    p.add_argument("--cmap",         type=str,   default="jet")
    p.add_argument("--overlay_alpha",type=float, default=0.5)
    p.add_argument("--which_layer",  type=str,   default="stage5_out")
    p.add_argument("--blocks",       type=str,   default="2,2,2,3")
    p.add_argument("--dilations",    type=str,   default="1,1,1,1")
    p.add_argument("--refine_blocks",type=int,   default=1)
    p.add_argument("--ckpt_segments",type=int,   default=0)
    p.add_argument("--embed_dim",    type=int,   default=512)
    p.add_argument("--proj_layers",  type=int,   default=2)
    p.add_argument("--proj_hidden",  type=int,   default=2048)
    p.add_argument("--proj_bn",      type=int,   default=0)
    p.add_argument("--proj_dropout", type=float, default=0.0)
    p.add_argument("--cnn_neg_mode", type=str, default="relu",
                   choices=["relu", "abs"],
                   help="How to handle negative CNN activation values before GAP ranking "
                        "and OMP normalization.\n"
                        "  relu: max(x, 0)  — only positive activations count\n"
                        "        (negative = 'not detected here', ignored)\n"
                        "  abs:  |x|        — magnitude regardless of sign\n"
                        "        (positive/negative equally meaningful, e.g. ETF geometry)")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    selected_cids     = [int(x.strip()) for x in args.concept_ids.split(",")]
    selected_cids_set = set(selected_cids)

    logger.info(f"SAE concepts: {selected_cids}")
    logger.info(f"Top CNN channels by GAP per image: {args.top_cnn_by_gap}")
    logger.info(f"OMP rounds: {args.n_select}")
    logger.info(f"Expected blocks: {len(selected_cids)} × {args.top_k} × "
                f"{args.top_cnn_by_gap} = "
                f"{len(selected_cids) * args.top_k * args.top_cnn_by_gap}")

    # ── 1. Load CSV  (rows indexed by [cid][img_name]) ──
    concept_class_map: Dict[int, str] = {}
    # rows_by_cid[cid] = list of {img_name, rank, gap_val, cls_label}
    rows_by_cid: Dict[int, List[dict]] = {c: [] for c in selected_cids}

    with open(args.csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid  = int(row["concept_id"])
            rank = int(row["rank"])
            if cid not in selected_cids_set:
                continue
            if args.top_k > 0 and rank > args.top_k:
                continue
            concept_class_map[cid] = row.get("concept_class", "")
            rows_by_cid[cid].append({
                "img_name": row["img_name"],
                "rank":     rank,
                "gap_val":  float(row.get("gap_val", 0.0)),
                "cls_label":row.get("concept_class", ""),
            })

    rows_by_cid = {k: v for k, v in rows_by_cid.items() if v}
    if not rows_by_cid:
        logger.error("No rows found. Check --concept_ids / --top_k.")
        return

    union_imgs = sorted(set(
        r["img_name"] for rows in rows_by_cid.values() for r in rows))
    logger.info(f"Union of images: {len(union_imgs)}")

    # ── 2. Load encoder ──
    model = SupMoCoModel(
        embed_dim   = args.embed_dim,
        blocks      = parse_int_list(args.blocks, 4),
        dilations   = parse_int_list(args.dilations, 4),
        refine_blocks=args.refine_blocks, ckpt_segments=args.ckpt_segments,
        proj_layers  =args.proj_layers, proj_hidden=args.proj_hidden,
        proj_bn=bool(args.proj_bn), proj_dropout=args.proj_dropout,
    )
    sd = torch.load(args.model_state_path, map_location="cpu", weights_only=False)
    robust_load_state_dict(model, sd, strict=True)
    encoder = model.encoder
    renorm_unit_per_out_channel_(encoder)
    encoder.eval().to(device).to(memory_format=torch.channels_last)
    del model, sd

    # ── 3. Load SAE ──
    ckpt = torch.load(args.sae_ckpt, map_location="cpu", weights_only=False)
    ca   = ckpt["args"]
    sae  = GatedSAE(d_in=ca.get("d_in", 512), d_sae=ca.get("d_sae", 4096),
                    tie_weights=ca.get("tie_gate_weights", False),
                    aux_k=ca.get("aux_k", 32))
    sae.load_state_dict(ckpt["sae"])
    sae.eval().to(device)
    logger.info(f"SAE d_sae={sae.d_sae}")

    # ── 4. TarBank ──
    refs        = load_all_sample_refs(args.shard_root)
    uid_to_ridx = build_uid_to_refidx(refs)
    name_to_ridx: Dict[str, int] = {}
    for uid, ridx in uid_to_ridx.items():
        name_to_ridx[uid.split(":")[-1] if ":" in uid else uid] = ridx

    found: List[str] = []
    for name in union_imgs:
        if name in name_to_ridx:
            found.append(name)
        else:
            m = next((r for k, r in name_to_ridx.items()
                      if k.startswith(name) or name.startswith(k.rsplit(".", 1)[0])), None)
            if m is not None:
                name_to_ridx[name] = m; found.append(name)
            else:
                logger.warning(f"  Not found in shards: {name}")

    bank      = InMemoryTarBank(refs, [name_to_ridx[n] for n in found], args.img_size)
    name_to_bi = {n: i for i, n in enumerate(found)}
    logger.info(f"Loaded {len(found)}/{len(union_imgs)} images")

    # ── 5. Output dirs ──
    os.makedirs(args.output_dir, exist_ok=True)
    is_drive = "/drive/" in args.output_dir or "/content/drive/" in args.output_dir
    if is_drive:
        local_out = os.path.join(tempfile.gettempdir(), "step15_omp_local")
        if os.path.exists(local_out): shutil.rmtree(local_out)
        os.makedirs(local_out, exist_ok=True)
    else:
        local_out = args.output_dir

    # ── 6. Feature-map cache (avoid recomputing for same image across concepts) ──
    cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    all_log: List[dict] = []

    # Pre-compute SAE binary stack (same for all images, indexed by selected_cids)
    # Actually: sae_b_stack depends on sae_all which is per-image.
    # We compute per image but selected_cids is fixed.

    # ── 7. Outer loop: SAE concept → images ──
    for cid in tqdm(selected_cids, desc="SAE concepts"):
        rows = rows_by_cid.get(cid, [])
        if not rows:
            continue

        cls_lbl = concept_class_map.get(cid, "")
        dir_name = (f"concept_{cid:04d}_{cls_lbl}" if cls_lbl
                    else f"concept_{cid:04d}")
        concept_dir = os.path.join(local_out, dir_name)
        os.makedirs(concept_dir, exist_ok=True)

        summary_figures: List[np.ndarray] = []
        poly_rows: List[dict] = []

        # Inner loop: image
        for row in tqdm(rows, desc=f"SAE#{cid}", leave=False):
            img_name = row["img_name"]
            rank     = row["rank"]
            gap_csv  = row["gap_val"]

            if img_name not in name_to_bi:
                continue
            bi      = name_to_bi[img_name]
            img_u16 = bank.images[bi]
            line    = bank.lines[bi]
            if img_u16 is None:
                continue

            # Use cached maps if available
            if img_name not in cache:
                cache[img_name] = extract_all_maps(encoder, sae, img_u16, device, args.which_layer)
            cnn_maps, sae_all = cache[img_name]
            # cnn_maps: (512, Hf, Wf),  sae_all: (Hf, Wf, d_sae)

            # ── Select top CNN channels by GAP value in this image ──
            # cnn_neg_mode 따라 음수 처리 방식 선택:
            #   relu → max(x, 0).mean()   : 양수 활성화 영역만 집계
            #   abs  → |x|.mean()          : 부호 무관하게 신호 크기로 집계 (ETF 관점)
            if args.cnn_neg_mode == "abs":
                cnn_maps_proc = np.abs(cnn_maps)
            else:  # relu
                cnn_maps_proc = np.maximum(cnn_maps, 0)
            ch_gap_vals = cnn_maps_proc.mean(axis=(1, 2))                      # (512,)
            top_cnn_chs = np.argsort(ch_gap_vals)[::-1][:args.top_cnn_by_gap]  # (N,)


            # ── Precompute SAE normalized maps (soft, [0,1]) for OMP candidates ──
            # _norm_map: median → 0, 99.9th pct → 1  (same as step14 display)
            # Background pixels (near 0) contribute ≈0 to Soft IoU → no background pollution
            sae_norm_stack = np.stack([
                _norm_map(sae_all[:, :, c])
                for c in selected_cids
            ], axis=0)   # (K, Hf, Wf) float [0,1]

            # SAE reference for this concept (native res only)
            sae_map = sae_all[:, :, cid]    # (Hf, Wf)

            orig_s  = _resize(_fiji(img_u16), args.img_size)

            cnn_blocks: List[np.ndarray] = []

            # ── Inner-inner loop: per CNN channel ──
            for gap_rank, cnn_ch in enumerate(top_cnn_chs, 1):
                cnn_ch     = int(cnn_ch)
                cnn_map_ch = cnn_maps[cnn_ch]             # (Hf, Wf) raw (may have negatives)

                # cnn_neg_mode에 따라 음수 처리:
                #   relu → max(x, 0) : 눈에 안 띄는 구역(inhibited)은 0으로 제거
                #   abs  → |x|       : 음수 활성화도 '|magnitude|'로 확인 (ETF 관점)
                if args.cnn_neg_mode == "abs":
                    cnn_map_ch_proc = np.abs(cnn_map_ch)
                else:  # relu
                    cnn_map_ch_proc = np.maximum(cnn_map_ch, 0.0)

                # cnn_maps_proc의 channel도 동일한 모드를 쓴으므로 ch_gap_vals도 일관\uc131 유지
                cnn_norm_ch = _norm_map(cnn_map_ch_proc)        # (Hf, Wf) float [0,1]
                ch_gap      = float(ch_gap_vals[cnn_ch])

                # Soft OMP: CNN channel (normalized) ← SAE concept maps (normalized)
                # soft_IoU 사용 → 배경(≈0) 픽셀은 IoU에 거의 기여 없음
                sel_cids_omp, sel_ious_omp = omp_cnn_to_sae_soft(
                    cnn_norm=cnn_norm_ch,
                    sae_norm_stack=sae_norm_stack,
                    candidate_cids=selected_cids,
                    n=args.n_select,
                )

                # Soft polysemanticity: how many SAE needed to cover 80% of CNN activation energy?
                n_sae_80 = n_needed_for_coverage_soft(
                    cnn_norm_ch, sae_norm_stack, selected_cids, sel_cids_omp, threshold=0.80)

                blk = make_cnn_omp_block(
                    orig_s=orig_s, cnn_map_ch=cnn_map_ch_proc, cnn_b_ch=None,
                    gap_rank=gap_rank, gap_val=ch_gap,
                    sae_all=sae_all, sae_b_stack=None,
                    candidate_cids=selected_cids,
                    sel_cids=sel_cids_omp, sel_ious=sel_ious_omp,
                    concept_class_map=concept_class_map,
                    cnn_ch=cnn_ch, n_sae_80pct=n_sae_80,
                    S=args.img_size, cmap_name=args.cmap, alpha=args.overlay_alpha,
                )
                cnn_blocks.append(blk)

                poly_rows.append({
                    "img_name":    img_name, "rank_in_sae": rank, "line": line,
                    "cnn_channel": cnn_ch,  "gap_rank":    gap_rank,
                    "gap_val":     round(ch_gap, 5),
                    "n_sae_80pct": n_sae_80,
                    "omp_cids":    ",".join(str(c) for c in sel_cids_omp),
                    "omp_ious":    ",".join(str(v) for v in sel_ious_omp),
                })

                # Log per OMP step
                for k, (omp_cid, omp_iou) in enumerate(zip(sel_cids_omp, sel_ious_omp), 1):
                    all_log.append({
                        "sae_concept_id": cid, "sae_class": cls_lbl,
                        "img_name": img_name,  "rank_in_sae": rank, "line": line,
                        "cnn_channel": cnn_ch, "gap_rank": gap_rank, "gap_val": round(ch_gap, 5),
                        "omp_order":  k, "omp_sae_cid": omp_cid,
                        "omp_sae_class": concept_class_map.get(omp_cid, ""),
                        "omp_iou": omp_iou, "n_sae_80pct": n_sae_80,
                    })

            # ── Assemble per-image figure ──
            fig = make_image_figure(
                orig_s=orig_s, sae_cid=cid, sae_map=sae_map,
                gap_val_csv=gap_csv, cls_label=cls_lbl,
                cnn_blocks=cnn_blocks,
                S=args.img_size, cmap_name=args.cmap, alpha=args.overlay_alpha,
            )
            summary_figures.append(fig)

            fname = (f"{cls_lbl or line}_{img_name}"
                     f"__rank{rank:02d}_omp.png")
            Image.fromarray(fig).save(os.path.join(concept_dir, fname))

        # ── Summary panel (all images vertically stacked) ──
        if summary_figures:
            max_w = max(f.shape[1] for f in summary_figures)
            panel = np.concatenate([_pad_w(f, max_w) for f in summary_figures], axis=0)
            Image.fromarray(panel).save(
                os.path.join(concept_dir, "summary_panel.png"))

        # ── Polysemanticity CSV ──
        if poly_rows:
            pf = ["img_name", "rank_in_sae", "line", "cnn_channel", "gap_rank",
                  "gap_val", "n_sae_80pct", "omp_cids", "omp_ious"]
            with open(os.path.join(concept_dir, f"concept{cid:04d}_polysemanticity.csv"),
                      "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=pf)
                w.writeheader(); w.writerows(poly_rows)

            valid = [r["n_sae_80pct"] for r in poly_rows if r["n_sae_80pct"] > 0]
            if valid:
                logger.info(f"  SAE#{cid}({cls_lbl}): "
                            f"avg SAE concepts for 80% CNN coverage = {np.mean(valid):.2f} "
                            f"(across {len(valid)} CNN channels)")

    # ── 8. Global log CSV ──
    if all_log:
        gfields = ["sae_concept_id", "sae_class", "img_name", "rank_in_sae", "line",
                   "cnn_channel", "gap_rank", "gap_val",
                   "omp_order", "omp_sae_cid", "omp_sae_class", "omp_iou", "n_sae_80pct"]
        with open(os.path.join(local_out, "omp_log_all.csv"),
                  "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=gfields)
            w.writeheader(); w.writerows(all_log)
        logger.info(f"Global log: {len(all_log)} rows")

    # ── 9. Copy to Drive ──
    if is_drive and local_out != args.output_dir:
        logger.info(f"Copying to Drive: {args.output_dir}")
        if os.path.exists(args.output_dir): shutil.rmtree(args.output_dir)
        shutil.copytree(local_out, args.output_dir)
        shutil.rmtree(local_out)

    logger.info(f"\nDone → {args.output_dir}")


if __name__ == "__main__":
    main()
