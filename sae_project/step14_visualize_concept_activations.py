# ==============================================================================
# SAE Concept Activation Visualization
# - Load GAP means CSV and SAE checkpoint
# - Select top-K images per concept by GAP value
# - Visualize concept activations via bilinear interpolation
# ==============================================================================

import os
import io
import csv
import json
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from PIL import Image

try:
    import tifffile
except ImportError:
    raise RuntimeError("tifffile not installed. pip install tifffile")

try:
    import matplotlib.cm as cm
except ImportError:
    raise RuntimeError("matplotlib not installed. pip install matplotlib")

from sae_project.step01_configs import get_args, resolve_paths
from sae_project.step02_logging_utils import get_logger, SUPERCLASS_MAP
from sae_project.step03_data_shards import load_all_sample_refs, build_uid_to_refidx
from sae_project.step04_data_bank import (
    InMemoryTarBank, load_split_csv, SafeInstanceNormalize
)
from sae_project.step05_model_encoder import (
    SupConMoCoModel, parse_int_list, renorm_unit_per_out_channel_, robust_load_state_dict
)
from sae_project.step06_gated_sae import GatedSAE

logger = get_logger("visualize_concept")


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


def create_overlay(base_rgb: np.ndarray, heatmap_rgb: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Blend heatmap onto base image.
    """
    base = base_rgb.astype(np.float32)
    heat = heatmap_rgb.astype(np.float32)
    blended = base * (1 - alpha) + heat * alpha
    return blended.clip(0, 255).astype(np.uint8)


# ==============================================================================
# Data Loading
# ==============================================================================

def load_gap_csv(csv_path: str) -> Dict[int, Dict]:
    """
    Load GAP means CSV.
    Returns dict: concept_id -> {
        'is_alive': bool,
        'Control': float, 'SNCA': float, 'GBA': float, 'LRRK2': float,
        'max_class': str, 'class_diff': float, 'entropy': float
    }
    """
    concepts = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = int(row["concept_id"])
            concepts[cid] = {
                "is_alive": bool(int(row["is_alive"])),
                "Control": float(row["Control"]),
                "SNCA": float(row["SNCA"]),
                "GBA": float(row["GBA"]),
                "LRRK2": float(row["LRRK2"]),
                "max_class": row["max_class"],
                "class_diff": float(row["class_diff"]),
                "entropy": float(row["entropy"]),
            }
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
    gini_impurity = 1.0 - np.sum(probs ** 2)
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
            x = (img.astype(np.float32) / 65535.0)
            x = torch.from_numpy(x).permute(2, 0, 1)
            x = normalize(x)
            xs.append(x)
            batch_valid.append(bi)
        
        if len(xs) == 0:
            continue
        
        xb = torch.stack(xs, 0).to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        
        with torch.amp.autocast(**autocast_kwargs):
            fmap = encoder.forward_feature_maps(xb, which=which_layer)
        
        # Normalize same as training
        B = fmap.size(0)
        gap = fmap.mean(dim=(2, 3))
        gap_norm = gap.norm(dim=1, keepdim=True).view(B, 1, 1, 1).clamp_min(1e-12)
        fmap = fmap / gap_norm
        
        fmap = fmap.permute(0, 2, 3, 1).contiguous()
        C = fmap.shape[-1]
        _, Hf, Wf, _ = fmap.shape
        
        flat_tokens = fmap.view(-1, C)
        flat_tokens = flat_tokens - flat_tokens.mean(dim=0, keepdim=True)
        flat_tokens = F.normalize(flat_tokens, dim=1, eps=1e-12)
        
        # SAE forward
        with torch.amp.autocast(**autocast_kwargs):
            _, acts, _, _, _ = sae(flat_tokens)
        
        acts = acts.float().view(B, Hf * Wf, -1)
        
        # Get GAP for specific concept
        concept_acts = acts[:, :, concept_id]  # (B, H*W)
        concept_gap = concept_acts.mean(dim=1)  # (B,)
        
        gap_values.extend(concept_gap.cpu().numpy().tolist())
        valid_indices.extend(batch_valid)
    
    return np.array(gap_values), valid_indices


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
    
    x = (img_np.astype(np.float32) / 65535.0)
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
    flat_tokens = F.normalize(flat_tokens, dim=1, eps=1e-12)
    
    # SAE forward
    with torch.amp.autocast(**autocast_kwargs):
        _, acts, _, _, _ = sae(flat_tokens)
    
    acts = acts.float().view(Hf, Wf, -1)
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
):
    """
    Visualize top-K images for a specific concept.
    """
    concept_dir = os.path.join(output_dir, f"concept_{concept_id:04d}")
    os.makedirs(concept_dir, exist_ok=True)
    
    # Sort by GAP and take top-K
    sorted_idx = np.argsort(gap_values)[::-1]
    top_indices = sorted_idx[:top_k]
    
    for rank, idx in enumerate(top_indices, start=1):
        bi = valid_indices[idx]
        gap_val = gap_values[idx]
        
        img_u16 = bank.images[bi]
        if img_u16 is None:
            continue
        
        line = bank.lines[bi]
        label = bank.labels[bi]
        
        # Get activation map
        act_hw = get_concept_activation_map(
            encoder, sae, img_u16, concept_id, device, which_layer
        )
        
        # Upsample to image size
        act_t = torch.from_numpy(act_hw).unsqueeze(0).unsqueeze(0).float()
        act_up = F.interpolate(
            act_t,
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False
        ).squeeze().numpy()
        
        # Check if there's any meaningful activation
        max_act = float(act_up.max())
        mean_act = float(act_up.mean())
        
        # Skip if no activation at all (avoid all-blue heatmaps)
        if max_act < 1e-6:
            logger.warning(f"  Concept {concept_id} rank {rank}: no activation (max={max_act:.6f}), skipping")
            continue
        
        # Normalize for visualization using percentile-based scaling
        # This highlights the relative differences within each image
        p_low = np.percentile(act_up, 50)  # Use median as baseline (most sparse activations are 0)
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
        overlay_rgb = create_overlay(bright_rgb, heatmap_rgb, alpha=overlay_alpha)
        
        # Base filename (include max activation value for debugging)
        base = f"rank{rank:02d}_{line}_gap{gap_val:.4f}_max{max_act:.4f}"
        
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
    
    logger.info(f"  Concept {concept_id}: saved {min(top_k, len(gap_values))} images to {concept_dir}")


# ==============================================================================
# Main Entry Point
# ==============================================================================

def get_visualization_args():
    parser = argparse.ArgumentParser(description="SAE Concept Activation Visualization")
    
    # Required paths
    parser.add_argument("--gap_csv", type=str, required=True,
                        help="Path to class-wise GAP means CSV")
    parser.add_argument("--sae_ckpt", type=str, required=True,
                        help="Path to trained SAE checkpoint")
    parser.add_argument("--model_state_path", type=str, required=True,
                        help="Path to CNN backbone weights")
    parser.add_argument("--shard_root", type=str, required=True,
                        help="Path to sharded image data")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Directory containing train/val/test split CSVs")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for visualizations")
    
    # Concept selection
    parser.add_argument("--concept_ids", type=str, default="gini_filter",
                        help="Comma-separated concept IDs, 'all_alive', or 'gini_filter'")
    parser.add_argument("--max_gini", type=float, default=0.3,
                        help="Max Gini coefficient for class-specific filtering (lower = more specific)")
    parser.add_argument("--max_concepts", type=int, default=100,
                        help="Maximum number of concepts to visualize")
    
    # Visualization options
    parser.add_argument("--top_k", type=int, default=20,
                        help="Number of top images per concept")
    parser.add_argument("--cmap", type=str, default="jet",
                        help="Colormap name (jet, hot, viridis, etc.)")
    parser.add_argument("--overlay_alpha", type=float, default=0.5,
                        help="Overlay blend alpha (0-1)")
    
    # Model config
    parser.add_argument("--which_layer", type=str, default="stage5_out",
                        help="Layer to extract features from")
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
    
    # ===== Load GAP CSV =====
    logger.info(f"Loading GAP CSV: {args.gap_csv}")
    gap_info = load_gap_csv(args.gap_csv)
    logger.info(f"  Total concepts: {len(gap_info)}")
    
    alive_concepts = [cid for cid, info in gap_info.items() if info["is_alive"]]
    logger.info(f"  Alive concepts: {len(alive_concepts)}")
    
    # ===== Parse concept IDs =====
    if args.concept_ids == "all_alive":
        concept_ids = alive_concepts[:args.max_concepts]
        logger.info(f"  Using first {len(concept_ids)} alive concepts")
    elif args.concept_ids == "gini_filter":
        # Filter by Gini coefficient (class-specific concepts)
        filtered = filter_concepts_by_gini(gap_info, max_gini=args.max_gini)
        logger.info(f"  Gini filter (max_gini={args.max_gini}): found {len(filtered)} class-specific concepts")
        
        if len(filtered) == 0:
            logger.warning("  No concepts pass Gini filter! Try increasing --max_gini")
            return
        
        # Log top concepts
        for cid, gini, max_cls in filtered[:10]:
            logger.info(f"    Concept {cid}: Gini={gini:.4f}, max_class={max_cls}")
        if len(filtered) > 10:
            logger.info(f"    ... and {len(filtered) - 10} more")
        
        concept_ids = [x[0] for x in filtered[:args.max_concepts]]
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
    
    model = SupConMoCoModel(
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
    
    # Load val + test UIDs
    eval_uids = []
    for split_name in ["val_split.csv", "test_split.csv"]:
        split_path = os.path.join(args.save_dir, split_name)
        if os.path.exists(split_path):
            eval_uids.extend(load_split_csv(split_path))
    
    logger.info(f"  Eval images (val+test): {len(eval_uids)}")
    
    # Build bank
    eval_refidx = [uid_to_refidx[u] for u in eval_uids if u in uid_to_refidx]
    bank = InMemoryTarBank(refs, eval_refidx, args.img_size)
    bank_indices = list(range(len(eval_refidx)))
    
    # ===== Visualize Each Concept =====
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info(f"\nVisualizing {len(concept_ids)} concepts...")
    
    for cid in tqdm(concept_ids, desc="Concepts"):
        # Compute GAP for all images for this concept
        gap_values, valid_indices = compute_concept_activations_for_images(
            encoder, sae, bank, bank_indices, cid, device,
            which_layer=args.which_layer,
            batch_size=args.batch_size,
        )
        
        if len(gap_values) == 0:
            logger.warning(f"  Concept {cid}: no valid images")
            continue
        
        # Visualize top-K
        visualize_concept(
            encoder, sae, bank, cid, valid_indices, gap_values,
            top_k=args.top_k,
            output_dir=args.output_dir,
            device=device,
            which_layer=args.which_layer,
            img_size=args.img_size,
            cmap_name=args.cmap,
            overlay_alpha=args.overlay_alpha,
        )
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    logger.info(f"\nDone! Output -> {args.output_dir}")


if __name__ == "__main__":
    main()
