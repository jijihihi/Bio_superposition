# ==============================================================================
# Channel Spatial Interaction Attribution
#
# Two-step analysis:
#   Step 1: Direction invariance validation
#     - Apply lr_swap vs ud_swap to different channel pairs
#     - If ||diff|| is similar regardless of swap direction,
#       L2 norm reliably measures spatial info destruction
#
#   Step 2: Marginal attribution of RG/RB/GB interactions
#     - Swap all 3 channels independently → effect_all (all interactions broken)
#     - Swap 2 channels together → preserves their interaction
#     - RG contribution = effect_all - effect_preserve_RG
#
# Perturbation types per channel:
#   - lr_swap: swap left/right halves
#   - ud_swap: swap top/bottom halves
#
# Usage:
#   python -m suppression_test.step08_channel_attribution \
#       --model_state_path /path/to/best_model.pt \
#       --sae_ckpt /path/to/sae_checkpoint.pt \
#       --save_dir /path/to/MoCo_seed42 \
#       --shard_root /path/to/wds_shards_tar \
#       --samples_per_class 500
# ==============================================================================

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

logger = get_logger("channel_attribution")


# ==============================================================================
# Per-channel spatial perturbations
# ==============================================================================
def lr_swap_channels(x: torch.Tensor, channels: List[int]) -> torch.Tensor:
    """lr_swap specified channels, leave others unchanged. x: (B, 3, H, W)"""
    out = x.clone()
    w2 = x.shape[3] // 2
    for ch in channels:
        out[:, ch, :, :] = torch.cat([x[:, ch, :, w2:], x[:, ch, :, :w2]], dim=2)
    return out


def ud_swap_channels(x: torch.Tensor, channels: List[int]) -> torch.Tensor:
    """up-down swap specified channels, leave others unchanged. x: (B, 3, H, W)"""
    out = x.clone()
    h2 = x.shape[2] // 2
    for ch in channels:
        out[:, ch, :, :] = torch.cat([x[:, ch, h2:, :], x[:, ch, :h2, :]], dim=1)
    return out


# ==============================================================================
# Seam masks
# ==============================================================================
def build_lr_seam_mask(H: int, W: int, margin: int, device: torch.device) -> torch.Tensor:
    """Mask center ±margin and edges ±margin (width axis). Returns (1,1,H,W)."""
    mask = torch.ones(H, W, device=device)
    center = W // 2
    mask[:, max(0, center - margin):min(W, center + margin)] = 0
    mask[:, :margin] = 0
    mask[:, W - margin:] = 0
    return mask.view(1, 1, H, W)


def build_ud_seam_mask(H: int, W: int, margin: int, device: torch.device) -> torch.Tensor:
    """Mask center ±margin and edges ±margin (height axis). Returns (1,1,H,W)."""
    mask = torch.ones(H, W, device=device)
    center = H // 2
    mask[max(0, center - margin):min(H, center + margin), :] = 0
    mask[:margin, :] = 0
    mask[H - margin:, :] = 0
    return mask.view(1, 1, H, W)


def build_combined_seam_mask(H: int, W: int, margin: int, device: torch.device) -> torch.Tensor:
    """Mask both lr and ud seams. Returns (1,1,H,W)."""
    lr = build_lr_seam_mask(H, W, margin, device).squeeze()
    ud = build_ud_seam_mask(H, W, margin, device).squeeze()
    return (lr.squeeze() * ud.squeeze()).view(1, 1, H, W)


# ==============================================================================
# Unshuffle activation maps
# ==============================================================================
def unshuffle_lr(act: torch.Tensor) -> torch.Tensor:
    """Reverse lr_swap on activation map."""
    w2 = act.shape[3] // 2
    return torch.cat([act[:, :, :, w2:], act[:, :, :, :w2]], dim=3)


def unshuffle_ud(act: torch.Tensor) -> torch.Tensor:
    """Reverse ud_swap on activation map."""
    h2 = act.shape[2] // 2
    return torch.cat([act[:, :, h2:, :], act[:, :, :h2, :]], dim=2)


# ==============================================================================
# SAE forward
# ==============================================================================
@torch.no_grad()
def get_sae_activation_maps(
    encoder: Encoder, sae: GatedSAE,
    x: torch.Tensor, which_layer: str,
) -> torch.Tensor:
    """Returns act_maps: (B, d_sae, H, W) — full spatial."""
    fmap = encoder.forward_feature_maps(x, which=which_layer)
    B, C, H, W = fmap.shape

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
        _, chunk_acts, _, _, _ = sae(flat_tokens[start:end])
        acts_list.append(chunk_acts.float())
    acts = torch.cat(acts_list, dim=0)

    d_sae = acts.shape[1]
    return acts.view(B, H, W, d_sae).permute(0, 3, 1, 2)


# ==============================================================================
# Split CSV (reuse)
# ==============================================================================
KNOWN_SHARD_ROOTS = [
    "/home/ubuntu/model-east3/wds_shards_tar",
    "/home/ubuntu/model-east3/wds_shards",
    "/content/wds_shards_tar",
    "/content/wds_shards",
]

def remap_uid(uid, new_root):
    for old in KNOWN_SHARD_ROOTS:
        if uid.startswith(old):
            return new_root + uid[len(old):]
    return uid

def load_split_csv(csv_path, shard_root=None):
    uids = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            uid = row["uid"]
            if shard_root:
                uid = remap_uid(uid, shard_root)
            uids.append(uid)
    return uids


# ==============================================================================
# Plotting
# ==============================================================================
def plot_histogram(values, title, xlabel, path, alive_mask=None, vline=None, dpi=200):
    if alive_mask is not None:
        values = values[alive_mask]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(values[np.isfinite(values)], bins=100, alpha=0.7,
            color="#4C72B0", edgecolor="black", linewidth=0.3)
    if vline is not None:
        ax.axvline(vline, color="red", linestyle="--", linewidth=1.5)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.text(0.98, 0.95,
            f"n={len(values)}\nmean={np.nanmean(values):.6f}\nstd={np.nanstd(values):.6f}",
            transform=ax.transAxes, fontsize=9, va="top", ha="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if _IN_COLAB:
        plt.show()
    plt.close(fig)
    logger.info(f"Saved: {path}")


# ==============================================================================
# Args
# ==============================================================================
def get_args():
    p = argparse.ArgumentParser("Channel Spatial Interaction Attribution")

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

    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = os.path.join(save_dir, "channel_attribution")
    os.makedirs(out_dir, exist_ok=True)

    autocast_kwargs = dict(device_type="cuda", enabled=torch.cuda.is_available())
    if torch.cuda.is_available():
        autocast_kwargs["dtype"] = torch.bfloat16

    margin = args.seam_margin
    top_k = args.top_k_per_neuron

    # ══════════════════════════════════════════════════════════════
    # Define perturbation conditions
    # ══════════════════════════════════════════════════════════════
    # Channel indices: R=0, G=1, B=2
    #
    # Step 1: Direction invariance (3 conditions, all break ALL interactions)
    #   Each swaps 2 channels in different directions, 3rd unchanged.
    #   If ||diff|| similar across all 3 → measurement is direction-invariant.
    #
    # Step 2: Marginal attribution (3 conditions, each preserves one interaction)
    #   Swap 2 channels TOGETHER (same direction) + 3rd in other direction.

    conditions = {
        # ── Step 1: Direction invariance (all interactions broken) ──
        # RG방식: R=lr, G=ud, B=unchanged
        "break_RG": {
            "desc": "R=lr, G=ud, B=unchanged (all broken)",
            "perturb": lambda x: ud_swap_channels(lr_swap_channels(x, [0]), [1]),
            "unshuffle": lambda act: unshuffle_ud(unshuffle_lr(act)),
            "mask_builder": lambda H, W, m, d: build_combined_seam_mask(H, W, m, d),
        },
        # GB방식: G=lr, B=ud, R=unchanged
        "break_GB": {
            "desc": "G=lr, B=ud, R=unchanged (all broken)",
            "perturb": lambda x: ud_swap_channels(lr_swap_channels(x, [1]), [2]),
            "unshuffle": lambda act: unshuffle_ud(unshuffle_lr(act)),
            "mask_builder": lambda H, W, m, d: build_combined_seam_mask(H, W, m, d),
        },
        # BR방식: B=lr, R=ud, G=unchanged
        "break_BR": {
            "desc": "B=lr, R=ud, G=unchanged (all broken)",
            "perturb": lambda x: ud_swap_channels(lr_swap_channels(x, [2]), [0]),
            "unshuffle": lambda act: unshuffle_ud(unshuffle_lr(act)),
            "mask_builder": lambda H, W, m, d: build_combined_seam_mask(H, W, m, d),
        },

        # ── Step 2: Preserve one interaction ──
        # Preserve RG: R+G together lr, B=ud
        "preserve_RG": {
            "desc": "R+G together lr + B ud (RG preserved)",
            "perturb": lambda x: ud_swap_channels(lr_swap_channels(x, [0, 1]), [2]),
            "unshuffle": lambda act: unshuffle_ud(unshuffle_lr(act)),
            "mask_builder": lambda H, W, m, d: build_combined_seam_mask(H, W, m, d),
        },
        # Preserve RB: R+B together lr, G=ud
        "preserve_RB": {
            "desc": "R+B together lr + G ud (RB preserved)",
            "perturb": lambda x: ud_swap_channels(lr_swap_channels(x, [0, 2]), [1]),
            "unshuffle": lambda act: unshuffle_ud(unshuffle_lr(act)),
            "mask_builder": lambda H, W, m, d: build_combined_seam_mask(H, W, m, d),
        },
        # Preserve GB: G+B together lr, R=ud
        "preserve_GB": {
            "desc": "G+B together lr + R ud (GB preserved)",
            "perturb": lambda x: ud_swap_channels(lr_swap_channels(x, [1, 2]), [0]),
            "unshuffle": lambda act: unshuffle_ud(unshuffle_lr(act)),
            "mask_builder": lambda H, W, m, d: build_combined_seam_mask(H, W, m, d),
        },
    }

    # ══════════════════════════════════════════════════════════════
    # Run all conditions
    # ══════════════════════════════════════════════════════════════

    results = {}  # name → (d_sae,) array of mean ||diff||² for top-K images

    for cond_name, cond in conditions.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Condition: {cond_name} — {cond['desc']}")
        logger.info(f"{'='*60}")

        all_gap_orig = []
        all_norm2 = []
        seam_mask = None

        for batch in tqdm(loader, desc=cond_name, leave=True):
            if batch is None:
                continue
            x, y, *_ = batch
            if x.numel() < 1:
                continue

            x_orig = x.to(device, non_blocking=True).contiguous(
                memory_format=torch.channels_last)

            x_pert = cond["perturb"](x_orig)
            x_pert = x_pert.contiguous(memory_format=torch.channels_last)

            with torch.amp.autocast(**autocast_kwargs):
                act_orig = get_sae_activation_maps(encoder, sae, x_orig, which_layer)
                act_pert = get_sae_activation_maps(encoder, sae, x_pert, which_layer)

            act_pert_unshuf = cond["unshuffle"](act_pert)
            B, d_sae_b, H, W = act_orig.shape

            if seam_mask is None:
                seam_mask = cond["mask_builder"](H, W, margin, device)
                n_valid = seam_mask.sum().item()
                logger.info(f"  Act map: {H}x{W}, valid pixels: {int(n_valid)}/{H*W}")

            diff = (act_orig - act_pert_unshuf) * seam_mask
            norm2 = (diff ** 2).sum(dim=(2, 3))
            gap_orig = act_orig.mean(dim=(2, 3))

            all_gap_orig.append(gap_orig.cpu().float().numpy())
            all_norm2.append(norm2.cpu().float().numpy())

        gap_orig_all = np.concatenate(all_gap_orig, axis=0)
        norm2_all = np.concatenate(all_norm2, axis=0)
        N_total = gap_orig_all.shape[0]

        k_actual = min(top_k, N_total)
        norm2_topk = np.zeros(d_sae)
        for n_i in range(d_sae):
            if not alive_mask[n_i]:
                continue
            gap_col = gap_orig_all[:, n_i]
            topk_idx = np.argpartition(gap_col, -k_actual)[-k_actual:]
            norm2_topk[n_i] = norm2_all[topk_idx, n_i].mean()

        results[cond_name] = norm2_topk
        vals = norm2_topk[alive_mask]
        logger.info(f"  ||diff||² (top-{k_actual}): mean={vals.mean():.6f}, "
                    f"std={vals.std():.6f}")
        seam_mask = None

    # ══════════════════════════════════════════════════════════════
    # Step 1: Direction invariance analysis
    # ══════════════════════════════════════════════════════════════
    logger.info(f"\n{'='*60}")
    logger.info("STEP 1: Direction Invariance Validation")
    logger.info(f"{'='*60}")

    break_names = ["break_RG", "break_GB", "break_BR"]
    break_vals = [results[n][alive_mask] for n in break_names]
    for i, na in enumerate(break_names):
        for j, nb in enumerate(break_names):
            if j <= i:
                continue
            ratio = break_vals[i] / (break_vals[j] + 1e-12)
            logger.info(f"\n  {na} vs {nb}:")
            logger.info(f"    mean_A={break_vals[i].mean():.6f}, mean_B={break_vals[j].mean():.6f}")
            logger.info(f"    Ratio A/B: mean={ratio.mean():.4f}, std={ratio.std():.4f}")
            logger.info(f"    Correlation: {np.corrcoef(break_vals[i], break_vals[j])[0,1]:.4f}")

            plot_histogram(
                ratio, f"Direction Invariance – {na} vs {nb}\n(ratio, 1.0=perfect)",
                "||diff_A||² / ||diff_B||²",
                os.path.join(out_dir, f"invariance_{na}_vs_{nb}.png"),
                alive_mask=None, vline=1.0, dpi=args.dpi)

    # ══════════════════════════════════════════════════════════════
    # Step 2: Marginal Attribution
    # ══════════════════════════════════════════════════════════════
    logger.info(f"\n{'='*60}")
    logger.info("STEP 2: Marginal Attribution (RG, RB, GB)")
    logger.info(f"{'='*60}")

    # effect_all = average of the 3 break conditions
    effect_all = (results["break_RG"] + results["break_GB"] + results["break_BR"]) / 3.0
    effect_pRG = results["preserve_RG"]
    effect_pRB = results["preserve_RB"]
    effect_pGB = results["preserve_GB"]

    # Marginal contribution = effect_all - effect_preserve
    contrib_RG = effect_all - effect_pRG  # Breaking RG adds this much
    contrib_RB = effect_all - effect_pRB
    contrib_GB = effect_all - effect_pGB

    # Normalize to fractions
    total_contrib = np.abs(contrib_RG) + np.abs(contrib_RB) + np.abs(contrib_GB) + 1e-12
    frac_RG = np.abs(contrib_RG) / total_contrib
    frac_RB = np.abs(contrib_RB) / total_contrib
    frac_GB = np.abs(contrib_GB) / total_contrib

    for name, contrib, frac in [("RG", contrib_RG, frac_RG),
                                 ("RB", contrib_RB, frac_RB),
                                 ("GB", contrib_GB, frac_GB)]:
        vals_c = contrib[alive_mask]
        vals_f = frac[alive_mask]
        logger.info(f"\n  {name} interaction:")
        logger.info(f"    Marginal contribution: mean={vals_c.mean():.8f}, std={vals_c.std():.8f}")
        logger.info(f"    Fraction: mean={vals_f.mean():.4f}, std={vals_f.std():.4f}")

    # Additivity check: sum of marginals vs total effect
    sum_marginals = np.abs(contrib_RG) + np.abs(contrib_RB) + np.abs(contrib_GB)
    additivity = sum_marginals[alive_mask] / (effect_all[alive_mask] + 1e-12)
    logger.info(f"\n  Additivity check (sum_marginals / effect_all):")
    logger.info(f"    mean={additivity.mean():.4f}, std={additivity.std():.4f}")

    # ── Save ──
    npz_path = os.path.join(out_dir, "channel_attribution_results.npz")
    np.savez_compressed(npz_path,
                        **{f"norm2_{k}": v for k, v in results.items()},
                        contrib_RG=contrib_RG, contrib_RB=contrib_RB, contrib_GB=contrib_GB,
                        frac_RG=frac_RG, frac_RB=frac_RB, frac_GB=frac_GB,
                        alive_mask=alive_mask, usage_ema=usage_ema,
                        top_k=top_k, seam_margin=margin)
    logger.info(f"\nSaved: {npz_path}")

    # ── Stacked bar plot ──
    fig, ax = plt.subplots(figsize=(12, 5))
    fRG = frac_RG[alive_mask]
    fRB = frac_RB[alive_mask]
    fGB = frac_GB[alive_mask]
    sort_idx = np.argsort(fRG)
    x_pos = np.arange(n_alive)
    ax.bar(x_pos, fRG[sort_idx], label="RG (Mito-Lyso)", color="#E74C3C", alpha=0.8, width=1.0)
    ax.bar(x_pos, fRB[sort_idx], bottom=fRG[sort_idx],
           label="RB (Mito-Cell)", color="#3498DB", alpha=0.8, width=1.0)
    ax.bar(x_pos, fGB[sort_idx],
           bottom=fRG[sort_idx] + fRB[sort_idx],
           label="GB (Lyso-Cell)", color="#2ECC71", alpha=0.8, width=1.0)
    ax.set_xlabel("Neuron (sorted by RG fraction)", fontsize=12)
    ax.set_ylabel("Interaction fraction", fontsize=12)
    ax.set_title("Channel Spatial Interaction Attribution per SAE Neuron",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xlim(0, n_alive)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "attribution_stacked.png"),
                dpi=args.dpi, bbox_inches="tight")
    if _IN_COLAB:
        plt.show()
    plt.close(fig)

    # ── Per-pair histograms ──
    for name, frac in [("RG", frac_RG), ("RB", frac_RB), ("GB", frac_GB)]:
        plot_histogram(
            frac, f"{name} Interaction Fraction",
            "Fraction (0–1)",
            os.path.join(out_dir, f"hist_frac_{name}.png"),
            alive_mask=alive_mask, dpi=args.dpi)

    logger.info(f"\n{'='*60}")
    logger.info("Channel spatial interaction attribution complete!")
    logger.info(f"Results: {out_dir}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
