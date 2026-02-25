# ==============================================================================
# Global Shape Suppression Test
#
# Tests how much each SAE neuron relies on global shape vs local features.
#
# Method:
#   1. Original image → CNN encoder → feature map → SAE activation map
#   2. Perturbed image (global shape destroyed) → same pipeline
#   3. Compare per-neuron:
#      a) Cosine similarity of 16x16 flattened activation maps (spatial pattern)
#      b) GAP change ratio: (GAP_perturbed - GAP_original) / |GAP_original|
#
# Perturbation types:
#   - patch_shuffle_2x2: cut image into 2x2 patches, randomly shuffle
#   - left_right_swap:   swap left and right halves
#   - patch_shuffle_4x4: cut into 4x4 patches, randomly shuffle (finer)
#
# These destroy global shape but preserve local texture, RGB relationships,
# and spatial interactions within each patch.
#
# Usage (Colab with L4):
#   python -m suppression_test.step06_global_shape_suppression \
#       --model_state_path /path/to/best_model.pt \
#       --sae_ckpt /path/to/sae_checkpoint.pt \
#       --save_dir /path/to/MoCo_seed42 \
#       --shard_root /path/to/wds_shards_tar \
#       --samples_per_class 500
# ==============================================================================

import os
import sys
import csv
import json
import random
import argparse
from typing import List, Tuple, Dict
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import matplotlib
_IN_COLAB = "google.colab" in sys.modules
if not _IN_COLAB:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sae_project.step02_logging_utils import get_logger, OUT_DIM
from sae_project.step03_data_shards import load_all_sample_refs, build_uid_to_refidx
from sae_project.step04_data_bank import (
    InMemoryTarBank, InMemorySixteenBitDataset,
    seed_worker, collate_skip_none,
)
from sae_project.step05_model_encoder import (
    Encoder, SupMoCoModel, parse_int_list,
    renorm_unit_per_out_channel_, robust_load_state_dict,
)
from sae_project.step06_gated_sae import GatedSAE

logger = get_logger("global_shape_suppression")


# ==============================================================================
# Spatial perturbations (applied to input image tensor)
# ==============================================================================
def patch_shuffle_2x2(x: torch.Tensor, seed: int = None) -> Tuple[torch.Tensor, Dict]:
    """
    Shuffle image into 2x2 grid patches.
    Returns (perturbed_image, perm_info) for unshuffling activation maps.
    """
    if x.dim() == 3:
        x = x.unsqueeze(0)
    B, C, H, W = x.shape
    h2, w2 = H // 2, W // 2

    patches = [
        x[:, :, :h2, :w2],
        x[:, :, :h2, w2:],
        x[:, :, h2:, :w2],
        x[:, :, h2:, w2:],
    ]

    rng = random.Random(seed)
    perm = list(range(4))
    rng.shuffle(perm)

    top = torch.cat([patches[perm[0]], patches[perm[1]]], dim=3)
    bot = torch.cat([patches[perm[2]], patches[perm[3]]], dim=3)
    out = torch.cat([top, bot], dim=2)

    return out, {"type": "grid", "grid": 2, "perm": perm}


def patch_shuffle_4x4(x: torch.Tensor, seed: int = None) -> Tuple[torch.Tensor, Dict]:
    """Shuffle image into 4x4 grid patches. Returns (perturbed, perm_info)."""
    if x.dim() == 3:
        x = x.unsqueeze(0)
    B, C, H, W = x.shape
    ph, pw = H // 4, W // 4

    patches = []
    for i in range(4):
        for j in range(4):
            patches.append(x[:, :, i*ph:(i+1)*ph, j*pw:(j+1)*pw])

    rng = random.Random(seed)
    perm = list(range(16))
    rng.shuffle(perm)

    rows = []
    for i in range(4):
        row_patches = [patches[perm[i*4 + j]] for j in range(4)]
        rows.append(torch.cat(row_patches, dim=3))
    out = torch.cat(rows, dim=2)

    return out, {"type": "grid", "grid": 4, "perm": perm}


def left_right_swap(x: torch.Tensor, seed: int = None) -> Tuple[torch.Tensor, Dict]:
    """Swap left and right halves. Returns (perturbed, perm_info)."""
    if x.dim() == 3:
        x = x.unsqueeze(0)
    B, C, H, W = x.shape
    w2 = W // 2
    left = x[:, :, :, :w2]
    right = x[:, :, :, w2:]
    out = torch.cat([right, left], dim=3)
    return out, {"type": "lr_swap"}


PERTURBATION_FNS = {
    "patch_2x2": patch_shuffle_2x2,
    "patch_4x4": patch_shuffle_4x4,
    "lr_swap": left_right_swap,
}


# ==============================================================================
# Unshuffle activation map back to original spatial layout
# ==============================================================================
def unshuffle_activation_map(
    act_map: torch.Tensor, perm_info: Dict
) -> torch.Tensor:
    """
    Reverse the spatial perturbation on an activation map.
    act_map: (B, d_sae, H, W)

    For grid shuffle: we know which patch went where, so we put them back.
    For lr_swap: swap left/right back.
    """
    B, D, H, W = act_map.shape

    if perm_info["type"] == "lr_swap":
        w2 = W // 2
        left = act_map[:, :, :, :w2]
        right = act_map[:, :, :, w2:]
        return torch.cat([right, left], dim=3)

    elif perm_info["type"] == "grid":
        g = perm_info["grid"]
        perm = perm_info["perm"]
        ph, pw = H // g, W // g

        # Build inverse permutation
        inv_perm = [0] * len(perm)
        for new_pos, old_pos in enumerate(perm):
            inv_perm[old_pos] = new_pos

        # Extract patches from perturbed activation map
        pert_patches = []
        for i in range(g):
            for j in range(g):
                pert_patches.append(act_map[:, :, i*ph:(i+1)*ph, j*pw:(j+1)*pw])

        # Reassemble in original order
        rows = []
        for i in range(g):
            row_patches = [pert_patches[inv_perm[i*g + j]] for j in range(g)]
            rows.append(torch.cat(row_patches, dim=3))
        return torch.cat(rows, dim=2)

    return act_map  # fallback


# ==============================================================================
# SAE forward: get per-neuron spatial activation maps
# ==============================================================================
@torch.no_grad()
def get_sae_activation_maps(
    encoder: Encoder, sae: GatedSAE,
    x: torch.Tensor, which_layer: str,
    pool_size: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run encoder + SAE, return per-neuron spatial activation maps.

    Returns:
        act_maps: (B, d_sae, pool_size, pool_size) — spatial activation maps
        gap_per_neuron: (B, d_sae) — GAP of each neuron's activation map
    """
    # Encoder feature map: (B, C, H, W)
    fmap = encoder.forward_feature_maps(x, which=which_layer)
    B, C, H, W = fmap.shape

    # GAP L2 normalization (same as SAE training)
    gap = fmap.mean(dim=(2, 3))
    gap_norm = gap.norm(dim=1, keepdim=True).view(B, 1, 1, 1).clamp_min(1e-12)
    fmap_normed = fmap / gap_norm

    # Prepare tokens: (B, H, W, C)
    tokens = fmap_normed.permute(0, 2, 3, 1).contiguous()
    flat_tokens = tokens.view(-1, C)

    # Center (same as SAE training)
    flat_tokens = flat_tokens - flat_tokens.mean(dim=0, keepdim=True)

    # L2 normalize tokens (same as SAE training)
    flat_tokens = F.normalize(flat_tokens, dim=1, eps=1e-12)

    # SAE forward in chunks
    chunk_size = 8192
    acts_list = []
    for start in range(0, flat_tokens.size(0), chunk_size):
        end = min(start + chunk_size, flat_tokens.size(0))
        chunk = flat_tokens[start:end]
        _, chunk_acts, _, _, _ = sae(chunk)
        acts_list.append(chunk_acts.float())
    acts = torch.cat(acts_list, dim=0)  # (B*H*W, d_sae)

    d_sae = acts.shape[1]

    # Reshape to spatial maps: (B, H, W, d_sae) → (B, d_sae, H, W)
    act_maps = acts.view(B, H, W, d_sae).permute(0, 3, 1, 2)

    # Adaptive pool to target size
    if pool_size < H:
        adapt = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        act_maps_pooled = adapt(act_maps)
    else:
        act_maps_pooled = act_maps

    # GAP per neuron: (B, d_sae)
    gap_per_neuron = act_maps.mean(dim=(2, 3))

    return act_maps_pooled, gap_per_neuron


# ==============================================================================
# Compare original vs perturbed
# ==============================================================================
def compute_suppression_metrics(
    act_maps_orig: torch.Tensor,
    gap_orig: torch.Tensor,
    act_maps_pert: torch.Tensor,
    gap_pert: torch.Tensor,
    perm_info: Dict = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compare original vs perturbed per-neuron.
    Unshuffles perturbed activation maps before cosine similarity.

    Returns:
        cos_sim: (d_sae,) mean cosine similarity across batch
        gap_change: (d_sae,) mean relative GAP change across batch
    """
    B, d_sae, H, W = act_maps_orig.shape

    # Unshuffle perturbed activation map back to original layout
    if perm_info is not None:
        act_maps_pert = unshuffle_activation_map(act_maps_pert, perm_info)

    # Flatten spatial: (B, d_sae, H*W)
    flat_orig = act_maps_orig.view(B, d_sae, -1)
    flat_pert = act_maps_pert.view(B, d_sae, -1)

    # Cosine similarity per neuron per image: (B, d_sae)
    cos = F.cosine_similarity(flat_orig, flat_pert, dim=2)

    # GAP relative change: (GAP_pert - GAP_orig) / (|GAP_orig| + 1)
    gap_change = (gap_pert - gap_orig) / (gap_orig.abs() + 1.0)

    return cos.mean(dim=0).cpu().numpy(), gap_change.mean(dim=0).cpu().numpy()


# ==============================================================================
# Load split CSV
# ==============================================================================
KNOWN_SHARD_ROOTS = [
    "/home/ubuntu/model-east3/wds_shards_tar",
    "/home/ubuntu/model-east3/wds_shards",
    "/content/wds_shards_tar",
    "/content/wds_shards",
]


def remap_uid(uid: str, new_shard_root: str) -> str:
    """Replace old shard_root prefix in UID with the current one."""
    for old_root in KNOWN_SHARD_ROOTS:
        if uid.startswith(old_root):
            return new_shard_root + uid[len(old_root):]
    return uid


def load_split_csv(csv_path: str, shard_root: str = None) -> List[str]:
    uids = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row["uid"]
            if shard_root:
                uid = remap_uid(uid, shard_root)
            uids.append(uid)
    return uids


# ==============================================================================
# Plot results
# ==============================================================================
def plot_histogram(values: np.ndarray, title: str, xlabel: str,
                   output_path: str, alive_mask: np.ndarray = None,
                   vline: float = None, dpi: int = 200):
    """Plot histogram of per-neuron values."""
    if alive_mask is not None:
        values = values[alive_mask]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(values, bins=100, alpha=0.7, color="#4C72B0", edgecolor="black",
            linewidth=0.3)
    if vline is not None:
        ax.axvline(vline, color="red", linestyle="--", linewidth=1.5,
                   label=f"threshold={vline}")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Count (neurons)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.text(0.98, 0.95,
            f"n={len(values)}\nmean={values.mean():.4f}\nstd={values.std():.4f}",
            transform=ax.transAxes, fontsize=9, va="top", ha="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    if vline is not None:
        ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if _IN_COLAB:
        plt.show()
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def plot_scatter_cos_vs_gap(cos_sim: np.ndarray, gap_change: np.ndarray,
                            perturb_name: str, output_path: str,
                            alive_mask: np.ndarray = None, dpi: int = 200):
    """Scatter: cosine similarity vs GAP change ratio."""
    if alive_mask is not None:
        cos_sim = cos_sim[alive_mask]
        gap_change = gap_change[alive_mask]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(cos_sim, gap_change, s=5, alpha=0.4, c="#4C72B0", edgecolors="none")
    ax.set_xlabel("Cosine Similarity (spatial pattern)", fontsize=12)
    ax.set_ylabel("GAP Change Ratio", fontsize=12)
    ax.set_title(f"Global Shape Sensitivity – {perturb_name}\n"
                 f"(n={len(cos_sim)} alive neurons)",
                 fontsize=13, fontweight="bold")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if _IN_COLAB:
        plt.show()
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


# ==============================================================================
# Main
# ==============================================================================
def get_args():
    p = argparse.ArgumentParser("Global Shape Suppression Test")

    # Model
    p.add_argument("--model_state_path", type=str, required=True,
                   help="CNN encoder checkpoint (best_model.pt)")
    p.add_argument("--sae_ckpt", type=str, required=True,
                   help="SAE checkpoint (.pt)")

    # Data
    p.add_argument("--save_dir", type=str, default="",
                   help="Dir with val/test_split.csv")
    p.add_argument("--shard_root", type=str,
                   default="/content/wds_shards")
    p.add_argument("--samples_per_class", type=int, default=500)
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)

    # Encoder
    p.add_argument("--blocks", type=str, default="2,2,2,3")
    p.add_argument("--dilations", type=str, default="1,1,1,1")
    p.add_argument("--refine_blocks", type=int, default=1)
    p.add_argument("--ckpt_segments", type=int, default=0)

    # Analysis
    p.add_argument("--pool_size", type=int, default=16,
                   help="Adaptive pool size for activation maps (default: 16)")
    p.add_argument("--dead_threshold", type=float, default=5e-5)
    p.add_argument("--perturbations", type=str, nargs="+",
                   default=["patch_2x2", "lr_swap"],
                   choices=["patch_2x2", "patch_4x4", "lr_swap"])

    # Output
    p.add_argument("--output_dir", type=str, default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dpi", type=int, default=200)

    return p.parse_args()


def main():
    args = get_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Load SAE ──
    logger.info("Loading SAE...")
    sae_ckpt = torch.load(args.sae_ckpt, map_location="cpu", weights_only=False)
    sae_args = sae_ckpt["args"]
    sae = GatedSAE(
        d_in=sae_args.get("d_in", 512),
        d_sae=sae_args.get("d_sae", 4096),
        tie_weights=sae_args.get("tie_gate_weights", False),
        aux_k=sae_args.get("aux_k", 32),
    )
    sae.load_state_dict(sae_ckpt["sae"])
    sae.to(device).eval()
    which_layer = sae_args.get("which_layer", "refine_out")
    d_sae = sae.d_sae

    usage_ema = sae.usage_ema.cpu().numpy()
    alive_mask = usage_ema >= args.dead_threshold
    n_alive = int(alive_mask.sum())
    logger.info(f"  SAE: d_sae={d_sae}, alive={n_alive}, layer={which_layer}")

    # ── Load encoder ──
    logger.info("Loading encoder...")
    blocks = parse_int_list(args.blocks, 4)
    dilations = parse_int_list(args.dilations, 4)
    model = SupMoCoModel(
        embed_dim=512, blocks=blocks, dilations=dilations,
        refine_blocks=args.refine_blocks,
        ckpt_segments=args.ckpt_segments,
        proj_layers=2, proj_hidden=2048,
    )
    ckpt = torch.load(args.model_state_path, map_location="cpu", weights_only=False)
    robust_load_state_dict(model, ckpt, strict=False)
    encoder = model.encoder
    encoder.eval().to(device).to(memory_format=torch.channels_last)
    renorm_unit_per_out_channel_(encoder)
    del model

    # ── Load data ──
    save_dir = args.save_dir or os.path.dirname(args.model_state_path)
    refs = load_all_sample_refs(args.shard_root)
    uid_to_refidx = build_uid_to_refidx(refs)

    eval_uids = []
    for csv_name in ["val_split.csv", "test_split.csv"]:
        csv_path = os.path.join(save_dir, csv_name)
        if os.path.exists(csv_path):
            loaded = load_split_csv(csv_path, shard_root=args.shard_root)
            eval_uids.extend(loaded)
            logger.info(f"  Loaded {len(loaded)} UIDs from {csv_name}")

    ref_indices = [uid_to_refidx[u] for u in eval_uids if u in uid_to_refidx]
    logger.info(f"  Matched {len(ref_indices)}/{len(eval_uids)} UIDs")

    if len(ref_indices) == 0:
        raise RuntimeError(
            f"No UIDs matched! CSV UID example: {eval_uids[0] if eval_uids else 'N/A'}, "
            f"ref UID example: {list(uid_to_refidx.keys())[0] if uid_to_refidx else 'N/A'}")

    # Balanced subsample
    spc = args.samples_per_class
    if spc > 0:
        rng = random.Random(args.seed)
        class_to_idx = defaultdict(list)
        for idx in ref_indices:
            class_to_idx[refs[idx].label].append(idx)
        sampled = []
        for cls in sorted(class_to_idx.keys()):
            idxs = class_to_idx[cls]
            rng.shuffle(idxs)
            n = min(spc, len(idxs))
            sampled.extend(idxs[:n])
        ref_indices = sampled

    logger.info(f"Samples: {len(ref_indices)}")

    bank = InMemoryTarBank(refs, ref_indices, args.img_size)
    ib = list(range(len(ref_indices)))
    ds = InMemorySixteenBitDataset(bank, ib, args.img_size, augment=False)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        worker_init_fn=seed_worker, collate_fn=collate_skip_none)

    # ── Output dir ──
    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = os.path.join(save_dir, "global_shape_suppression")
    os.makedirs(out_dir, exist_ok=True)

    # ── Run perturbation analysis ──
    autocast_kwargs = dict(device_type="cuda", enabled=torch.cuda.is_available())
    if torch.cuda.is_available():
        autocast_kwargs["dtype"] = torch.bfloat16

    for perturb_name in args.perturbations:
        logger.info(f"\n{'='*60}")
        logger.info(f"Perturbation: {perturb_name}")
        logger.info(f"{'='*60}")

        perturb_fn = PERTURBATION_FNS[perturb_name]

        all_cos = []
        all_gap_change = []

        for batch in tqdm(loader, desc=f"{perturb_name}", leave=True):
            if batch is None:
                continue
            x, y, *_ = batch
            if x.numel() < 1:
                continue

            x_orig = x.to(device, non_blocking=True).contiguous(
                memory_format=torch.channels_last)

            # Apply perturbation
            x_pert, perm_info = perturb_fn(x_orig, seed=args.seed)
            x_pert = x_pert.contiguous(memory_format=torch.channels_last)

            # Get SAE activation maps
            with torch.amp.autocast(**autocast_kwargs):
                act_orig, gap_orig = get_sae_activation_maps(
                    encoder, sae, x_orig, which_layer, args.pool_size)
                act_pert, gap_pert = get_sae_activation_maps(
                    encoder, sae, x_pert, which_layer, args.pool_size)

            cos, gap_ch = compute_suppression_metrics(
                act_orig, gap_orig, act_pert, gap_pert, perm_info)
            all_cos.append(cos)
            all_gap_change.append(gap_ch)

        # Average across batches
        cos_sim = np.stack(all_cos).mean(axis=0)      # (d_sae,)
        gap_change = np.stack(all_gap_change).mean(axis=0)  # (d_sae,)

        # ── Log summary ──
        cos_alive = cos_sim[alive_mask]
        gap_alive = gap_change[alive_mask]
        logger.info(f"\nResults ({perturb_name}, {n_alive} alive neurons):")
        logger.info(f"  Cosine sim: mean={cos_alive.mean():.4f}, "
                    f"std={cos_alive.std():.4f}, "
                    f"min={cos_alive.min():.4f}, max={cos_alive.max():.4f}")
        logger.info(f"  GAP change: mean={gap_alive.mean():.4f}, "
                    f"std={gap_alive.std():.4f}")

        n_shape_dep = (cos_alive < 0.8).sum()
        logger.info(f"  Shape-dependent (cos<0.8): {n_shape_dep}/{n_alive} "
                    f"({n_shape_dep/n_alive*100:.1f}%)")

        # ── Save results ──
        npz_path = os.path.join(out_dir, f"results_{perturb_name}.npz")
        np.savez_compressed(npz_path,
                            cos_sim=cos_sim, gap_change=gap_change,
                            alive_mask=alive_mask, usage_ema=usage_ema,
                            perturbation=perturb_name)
        logger.info(f"Saved: {npz_path}")

        # ── Plots ──
        plot_histogram(
            cos_sim,
            f"Cosine Similarity – {perturb_name}",
            "Cosine Similarity (1.0 = no change)",
            os.path.join(out_dir, f"hist_cos_{perturb_name}.png"),
            alive_mask=alive_mask, vline=0.8, dpi=args.dpi)

        plot_histogram(
            np.abs(gap_change),
            f"|GAP Change Ratio| – {perturb_name}",
            "|ΔG/G| (0 = no change)",
            os.path.join(out_dir, f"hist_gap_{perturb_name}.png"),
            alive_mask=alive_mask, dpi=args.dpi)

        plot_scatter_cos_vs_gap(
            cos_sim, gap_change, perturb_name,
            os.path.join(out_dir, f"scatter_{perturb_name}.png"),
            alive_mask=alive_mask, dpi=args.dpi)

    logger.info(f"\n{'='*60}")
    logger.info("Global shape suppression test complete!")
    logger.info(f"Results: {out_dir}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
