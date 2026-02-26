# ==============================================================================
# Channel Spatial Interaction Decomposition Test
#
# Tests pairwise RGB spatial interactions (RG, RB, GB) per SAE neuron.
#
# Method:
#   1. V_orig = SAE activation map from original image
#   2. V_r    = SAE activation map from R-channel-only lr_swap (G,B unchanged)
#   3. V_g    = SAE activation map from G-channel-only lr_swap
#   4. V_b    = SAE activation map from B-channel-only lr_swap
#
#   diff_r = V_orig - V_r = deltaRG + deltaRB
#   diff_g = V_orig - V_g = deltaRG + deltaGB
#   diff_b = V_orig - V_b = deltaRB + deltaGB
#
#   Algebraic solution:
#     deltaRG = (diff_r + diff_g - diff_b) / 2
#     deltaRB = (diff_r + diff_b - diff_g) / 2
#     deltaGB = (diff_g + diff_b - diff_r) / 2
#
#   Inner product verification (high-dim orthogonality):
#     ||deltaRB||² ≈ diff_r · diff_b
#     ||deltaRG||² ≈ diff_r · diff_g
#     ||deltaGB||² ≈ diff_g · diff_b
#
#   Consistency = inner_product_estimate / algebraic_estimate
#   High consistency → decomposition assumption is valid.
#
# Usage:
#   python -m suppression_test.step07_channel_interaction_test \
#       --model_state_path /path/to/best_model.pt \
#       --sae_ckpt /path/to/sae_checkpoint.pt \
#       --save_dir /path/to/MoCo_seed42 \
#       --shard_root /path/to/wds_shards_tar \
#       --samples_per_class 500
# ==============================================================================

## lr swep.

'''
  python -m suppression_test.step07_channel_interaction_test \
      --model_state_path /home/ubuntu/model-east3/outputs/MoCo_seed87/best_model.pt \
      --sae_ckpt /home/ubuntu/model-east3/outputs/MoCo_seed87/SAE/stage5_out_d4096_gated_sp3200.0_aux0.03125_tied_ep008.pt \
      --save_dir /home/ubuntu/model-east3/outputs/MoCo_seed87 \
      --shard_root /home/ubuntu/model-east3/wds_shards_tar \
      --samples_per_class 500 \
      --seam_margin 5

'''

'''
   --model_state_path /home/ubuntu/model-east3/outputs/MoCo_seed87/best_model.pt \
    --sae_ckpt /home/ubuntu/model-east3/outputs/MoCo_seed87/SAE/stage5_out_d4096_gated_sp3200.0_aux0.03125_tied_ep008.pt  \
    --save_dir /home/ubuntu/model-east3/outputs/MoCo_seed87 \
    --shard_root /home/ubuntu/model-east3/wds_shards_tar \
    --samples_per_class 500 \
    --seam_margin 4 \
    --pool_size 64

'''


import os
import sys
import csv
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

logger = get_logger("channel_interaction")


# ==============================================================================
# Single-channel lr_swap
# ==============================================================================
def lr_swap_single_channel(x: torch.Tensor, channel: int) -> torch.Tensor:
    """
    lr_swap only one RGB channel, leave others unchanged.
    x: (B, 3, H, W)
    channel: 0=R, 1=G, 2=B
    """
    out = x.clone()
    B, C, H, W = x.shape
    w2 = W // 2
    # Swap left/right for specified channel only
    out[:, channel, :, :] = torch.cat([x[:, channel, :, w2:], x[:, channel, :, :w2]], dim=2)
    return out


# ==============================================================================
# Build seam mask for lr_swap (same as step06)
# ==============================================================================
def build_lr_seam_mask(H: int, W: int, margin: int, device: torch.device) -> torch.Tensor:
    """
    Build spatial mask for lr_swap: mask center ±margin and edges ±margin.
    Returns: (1, 1, H, W) binary mask.
    """
    mask_w = torch.ones(W, device=device)
    center = W // 2
    mask_w[max(0, center - margin):min(W, center + margin)] = 0
    mask_w[:margin] = 0
    mask_w[W - margin:] = 0
    return mask_w.view(1, 1, 1, W).expand(1, 1, H, W)


# ==============================================================================
# Unshuffle lr_swap activation map
# ==============================================================================
def unshuffle_lr(act_map: torch.Tensor) -> torch.Tensor:
    """Reverse lr_swap on activation map: swap left/right halves back."""
    B, D, H, W = act_map.shape
    w2 = W // 2
    left = act_map[:, :, :, :w2]
    right = act_map[:, :, :, w2:]
    return torch.cat([right, left], dim=3)


# ==============================================================================
# SAE forward (reuse from step06)
# ==============================================================================
@torch.no_grad()
def get_sae_activation_maps(
    encoder: Encoder, sae: GatedSAE,
    x: torch.Tensor, which_layer: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        act_maps: (B, d_sae, H, W) — full spatial activation maps (no pooling)
        gap: (B, d_sae) — GAP per neuron
    """
    fmap = encoder.forward_feature_maps(x, which=which_layer)
    B, C, H, W = fmap.shape

    # GAP L2 norm
    gap = fmap.mean(dim=(2, 3))
    gap_norm = gap.norm(dim=1, keepdim=True).view(B, 1, 1, 1).clamp_min(1e-12)
    fmap_normed = fmap / gap_norm

    tokens = fmap_normed.permute(0, 2, 3, 1).contiguous()
    flat_tokens = tokens.view(-1, C)
    flat_tokens = flat_tokens - flat_tokens.mean(dim=0, keepdim=True)
    flat_tokens = F.normalize(flat_tokens, dim=1, eps=1e-12)

    chunk_size = 8192
    acts_list = []
    for start in range(0, flat_tokens.size(0), chunk_size):
        end = min(start + chunk_size, flat_tokens.size(0))
        chunk = flat_tokens[start:end]
        _, chunk_acts, _, _, _ = sae(chunk)
        acts_list.append(chunk_acts.float())
    acts = torch.cat(acts_list, dim=0)

    d_sae = acts.shape[1]
    act_maps = acts.view(B, H, W, d_sae).permute(0, 3, 1, 2)
    gap_per_neuron = act_maps.mean(dim=(2, 3))

    return act_maps, gap_per_neuron


# ==============================================================================
# Split CSV loading (same as step06)
# ==============================================================================
KNOWN_SHARD_ROOTS = [
    "/home/ubuntu/model-east3/wds_shards_tar",
    "/home/ubuntu/model-east3/wds_shards",
    "/content/wds_shards_tar",
    "/content/wds_shards",
]


def remap_uid(uid: str, new_shard_root: str) -> str:
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
# Plotting
# ==============================================================================
def plot_histogram(values: np.ndarray, title: str, xlabel: str,
                   output_path: str, alive_mask: np.ndarray = None,
                   vline: float = None, dpi: int = 200):
    if alive_mask is not None:
        values = values[alive_mask]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(values[np.isfinite(values)], bins=100, alpha=0.7,
            color="#4C72B0", edgecolor="black", linewidth=0.3)
    if vline is not None:
        ax.axvline(vline, color="red", linestyle="--", linewidth=1.5,
                   label=f"={vline}")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Count (neurons)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.text(0.98, 0.95,
            f"n={len(values)}\nmean={np.nanmean(values):.6f}\nstd={np.nanstd(values):.6f}",
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


# ==============================================================================
# Args
# ==============================================================================
def get_args():
    p = argparse.ArgumentParser("Channel Spatial Interaction Decomposition")

    p.add_argument("--model_state_path", type=str, required=True)
    p.add_argument("--sae_ckpt", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="")
    p.add_argument("--shard_root", type=str, default="/content/wds_shards")
    p.add_argument("--samples_per_class", type=int, default=500)
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--blocks", type=str, default="2,2,2,3")
    p.add_argument("--dilations", type=str, default="1,1,1,1")
    p.add_argument("--refine_blocks", type=int, default=1)
    p.add_argument("--ckpt_segments", type=int, default=0)

    p.add_argument("--seam_margin", type=int, default=3)
    p.add_argument("--dead_threshold", type=float, default=5e-5)
    p.add_argument("--top_k_per_neuron", type=int, default=100)

    p.add_argument("--output_dir", type=str, default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dpi", type=int, default=200)

    return p.parse_args()


# ==============================================================================
# Main
# ==============================================================================
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
        raise RuntimeError("No UIDs matched!")

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
            sampled.extend(idxs[:min(spc, len(idxs))])
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
        out_dir = os.path.join(save_dir, "channel_interaction")
    os.makedirs(out_dir, exist_ok=True)

    # ── Run analysis ──
    autocast_kwargs = dict(device_type="cuda", enabled=torch.cuda.is_available())
    if torch.cuda.is_available():
        autocast_kwargs["dtype"] = torch.bfloat16

    top_k = args.top_k_per_neuron
    channel_names = ["R", "G", "B"]

    # Accumulate per-image per-neuron: diff vectors (masked, flattened)
    # We need the full masked vector per image per neuron to do inner products
    # But that's too much memory: (N_images, d_sae, H*W)
    # Instead accumulate running sums of:
    #   ||diff_r||², ||diff_g||², ||diff_b||²
    #   diff_r · diff_g, diff_r · diff_b, diff_g · diff_b
    #   ||deltaRG_alg||², ||deltaRB_alg||², ||deltaGB_alg||²
    # All per-image per-neuron → (N_images, d_sae), then top-K select

    all_gap_orig = []  # for top-K selection

    # Per-image per-neuron scalars
    all_norm2_diff = {ch: [] for ch in "RGB"}           # ||diff_ch||²
    all_dot_rg = []   # diff_r · diff_g  ≈ ||deltaRG||²
    all_dot_rb = []   # diff_r · diff_b  ≈ ||deltaRB||²
    all_dot_gb = []   # diff_g · diff_b  ≈ ||deltaGB||²

    all_norm2_alg_RG = []  # ||deltaRG_alg||²
    all_norm2_alg_RB = []  # ||deltaRB_alg||²
    all_norm2_alg_GB = []  # ||deltaGB_alg||²

    logger.info(f"\n{'='*60}")
    logger.info(f"Channel Interaction Decomposition (seam_margin={args.seam_margin})")
    logger.info(f"{'='*60}")

    seam_mask = None

    for batch in tqdm(loader, desc="channel_interaction", leave=True):
        if batch is None:
            continue
        x, y, *_ = batch
        if x.numel() < 1:
            continue

        x_orig = x.to(device, non_blocking=True).contiguous(
            memory_format=torch.channels_last)
        B_cur = x_orig.shape[0]

        with torch.amp.autocast(**autocast_kwargs):
            act_orig, gap_orig = get_sae_activation_maps(
                encoder, sae, x_orig, which_layer)

        _, _, H_act, W_act = act_orig.shape

        # Build seam mask once
        if seam_mask is None:
            seam_mask = build_lr_seam_mask(H_act, W_act, args.seam_margin, device)
            n_valid = seam_mask.sum().item()
            logger.info(f"  Activation map: {H_act}x{W_act}, "
                        f"seam_margin={args.seam_margin}, "
                        f"valid pixels={int(n_valid)}/{H_act*W_act}")

        all_gap_orig.append(gap_orig.cpu().float().numpy())

        # Per-channel lr_swap → diff vectors
        diffs = {}  # "R", "G", "B" → (B, d_sae, n_valid_flat)
        for ch_idx, ch_name in enumerate(channel_names):
            x_ch_swap = lr_swap_single_channel(x_orig, ch_idx)
            x_ch_swap = x_ch_swap.contiguous(memory_format=torch.channels_last)

            with torch.amp.autocast(**autocast_kwargs):
                act_ch, _ = get_sae_activation_maps(
                    encoder, sae, x_ch_swap, which_layer)

            # Unshuffle the swapped channel's activation map
            act_ch_unshuf = unshuffle_lr(act_ch)

            # Apply seam mask and flatten
            # diff = V_orig - V_ch (masked)
            diff = (act_orig - act_ch_unshuf) * seam_mask  # (B, d_sae, H, W)
            diffs[ch_name] = diff

        # Compute per-image per-neuron scalars
        # Flatten: (B, d_sae, H*W)
        diff_r = diffs["R"].view(B_cur, -1, H_act * W_act)
        diff_g = diffs["G"].view(B_cur, -1, H_act * W_act)
        diff_b = diffs["B"].view(B_cur, -1, H_act * W_act)

        # ||diff||² per neuron: (B, d_sae)
        all_norm2_diff["R"].append((diff_r ** 2).sum(dim=2).cpu().float().numpy())
        all_norm2_diff["G"].append((diff_g ** 2).sum(dim=2).cpu().float().numpy())
        all_norm2_diff["B"].append((diff_b ** 2).sum(dim=2).cpu().float().numpy())

        # Inner products (high-dim orthogonality verification)
        # diff_r · diff_g ≈ ||deltaRG||²
        dot_rg = (diff_r * diff_g).sum(dim=2).cpu().float().numpy()
        dot_rb = (diff_r * diff_b).sum(dim=2).cpu().float().numpy()
        dot_gb = (diff_g * diff_b).sum(dim=2).cpu().float().numpy()
        all_dot_rg.append(dot_rg)
        all_dot_rb.append(dot_rb)
        all_dot_gb.append(dot_gb)

        # Algebraic solution
        # deltaRG = (diff_r + diff_g - diff_b) / 2
        delta_RG = (diff_r + diff_g - diff_b) / 2.0
        delta_RB = (diff_r + diff_b - diff_g) / 2.0
        delta_GB = (diff_g + diff_b - diff_r) / 2.0

        all_norm2_alg_RG.append((delta_RG ** 2).sum(dim=2).cpu().float().numpy())
        all_norm2_alg_RB.append((delta_RB ** 2).sum(dim=2).cpu().float().numpy())
        all_norm2_alg_GB.append((delta_GB ** 2).sum(dim=2).cpu().float().numpy())

        # Free memory
        del diffs, diff_r, diff_g, diff_b, delta_RG, delta_RB, delta_GB

    # ── Stack all batches: (N_total, d_sae) ──
    gap_orig_all = np.concatenate(all_gap_orig, axis=0)
    N_total = gap_orig_all.shape[0]

    norm2_diff = {ch: np.concatenate(all_norm2_diff[ch], axis=0) for ch in "RGB"}
    dot_rg = np.concatenate(all_dot_rg, axis=0)
    dot_rb = np.concatenate(all_dot_rb, axis=0)
    dot_gb = np.concatenate(all_dot_gb, axis=0)
    norm2_alg_RG = np.concatenate(all_norm2_alg_RG, axis=0)
    norm2_alg_RB = np.concatenate(all_norm2_alg_RB, axis=0)
    norm2_alg_GB = np.concatenate(all_norm2_alg_GB, axis=0)

    logger.info(f"\nTotal images: {N_total}")

    # ── Per-neuron top-K selection ──
    k_actual = min(top_k, N_total)
    logger.info(f"Selecting top-{k_actual} images per neuron by GAP_orig")

    # Results per neuron
    consistency_RG = np.zeros(d_sae)
    consistency_RB = np.zeros(d_sae)
    consistency_GB = np.zeros(d_sae)

    magnitude_RG = np.zeros(d_sae)
    magnitude_RB = np.zeros(d_sae)
    magnitude_GB = np.zeros(d_sae)

    for n_i in range(d_sae):
        if not alive_mask[n_i]:
            continue
        gap_col = gap_orig_all[:, n_i]
        topk_idx = np.argpartition(gap_col, -k_actual)[-k_actual:]

        # Inner product estimates (avg over top-K images)
        ip_RG = dot_rg[topk_idx, n_i].mean()
        ip_RB = dot_rb[topk_idx, n_i].mean()
        ip_GB = dot_gb[topk_idx, n_i].mean()

        # Algebraic estimates (avg over top-K images)
        alg_RG = norm2_alg_RG[topk_idx, n_i].mean()
        alg_RB = norm2_alg_RB[topk_idx, n_i].mean()
        alg_GB = norm2_alg_GB[topk_idx, n_i].mean()

        # Consistency = inner_product / algebraic (should be ~1.0 if decomposition valid)
        consistency_RG[n_i] = ip_RG / (alg_RG + 1e-12)
        consistency_RB[n_i] = ip_RB / (alg_RB + 1e-12)
        consistency_GB[n_i] = ip_GB / (alg_GB + 1e-12)

        # Relative magnitudes (what fraction of total diff is each interaction?)
        total = alg_RG + alg_RB + alg_GB + 1e-12
        magnitude_RG[n_i] = alg_RG / total
        magnitude_RB[n_i] = alg_RB / total
        magnitude_GB[n_i] = alg_GB / total

    # ── Log summary ──
    logger.info(f"\n{'='*60}")
    logger.info(f"Results ({n_alive} alive neurons, top-{k_actual} images)")
    logger.info(f"{'='*60}")

    for name, cons in [("RG", consistency_RG), ("RB", consistency_RB), ("GB", consistency_GB)]:
        vals = cons[alive_mask]
        logger.info(f"\n  Consistency delta{name}:")
        logger.info(f"    mean={vals.mean():.6f}, std={vals.std():.6f}")
        logger.info(f"    median={np.median(vals):.6f}")
        logger.info(f"    [0.8-1.2] range: {((vals > 0.8) & (vals < 1.2)).sum()}/{n_alive} "
                    f"({((vals > 0.8) & (vals < 1.2)).sum()/n_alive*100:.1f}%)")

    logger.info(f"\n  Interaction magnitudes (fraction of total):")
    for name, mag in [("RG", magnitude_RG), ("RB", magnitude_RB), ("GB", magnitude_GB)]:
        vals = mag[alive_mask]
        logger.info(f"    delta{name}: mean={vals.mean():.6f}, std={vals.std():.6f}")

    # ── Save ──
    npz_path = os.path.join(out_dir, "channel_interaction_results.npz")
    np.savez_compressed(npz_path,
                        consistency_RG=consistency_RG,
                        consistency_RB=consistency_RB,
                        consistency_GB=consistency_GB,
                        magnitude_RG=magnitude_RG,
                        magnitude_RB=magnitude_RB,
                        magnitude_GB=magnitude_GB,
                        alive_mask=alive_mask,
                        usage_ema=usage_ema,
                        top_k=k_actual,
                        seam_margin=args.seam_margin)
    logger.info(f"\nSaved: {npz_path}")

    # ── Plots ──
    for name, cons in [("RG", consistency_RG), ("RB", consistency_RB), ("GB", consistency_GB)]:
        plot_histogram(
            cons, f"Consistency ratio – delta{name}",
            f"inner_product / algebraic (1.0 = perfect)",
            os.path.join(out_dir, f"hist_consistency_{name}.png"),
            alive_mask=alive_mask, vline=1.0, dpi=args.dpi)

    # Stacked bar: interaction magnitudes
    fig, ax = plt.subplots(figsize=(10, 5))
    mag_RG_a = magnitude_RG[alive_mask]
    mag_RB_a = magnitude_RB[alive_mask]
    mag_GB_a = magnitude_GB[alive_mask]
    # Sort by RG magnitude
    sort_idx = np.argsort(mag_RG_a)
    x_pos = np.arange(n_alive)
    ax.bar(x_pos, mag_RG_a[sort_idx], label="deltaRG", color="#E74C3C", alpha=0.8)
    ax.bar(x_pos, mag_RB_a[sort_idx], bottom=mag_RG_a[sort_idx],
           label="deltaRB", color="#3498DB", alpha=0.8)
    ax.bar(x_pos, mag_GB_a[sort_idx],
           bottom=mag_RG_a[sort_idx] + mag_RB_a[sort_idx],
           label="deltaGB", color="#2ECC71", alpha=0.8)
    ax.set_xlabel("Neuron (sorted by deltaRG fraction)", fontsize=12)
    ax.set_ylabel("Fraction of total interaction", fontsize=12)
    ax.set_title("Channel Interaction Decomposition per SAE Neuron", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xlim(0, n_alive)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "interaction_magnitudes.png"),
                dpi=args.dpi, bbox_inches="tight")
    if _IN_COLAB:
        plt.show()
    plt.close(fig)
    logger.info(f"Saved: interaction_magnitudes.png")

    logger.info(f"\n{'='*60}")
    logger.info("Channel interaction analysis complete!")
    logger.info(f"Results: {out_dir}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
