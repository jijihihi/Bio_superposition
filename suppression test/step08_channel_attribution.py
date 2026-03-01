# ==============================================================================
# Channel Spatial Interaction Attribution — L2² Norm Approach
#
# Key insight: CNN pooling/convolution makes spatial tracking impossible after
# channel shifts. Instead, we compare the L2² norm of each SAE neuron's
# activation map, with seam regions masked out to exclude swap artifacts.
#
# Three-phase analysis:
#   Phase A: Direction invariance validation
#     - 2 representative "all-broken" conditions with different direction assignments
#     - Per-(image, neuron) ratio should be ≈1 if direction-invariant
#
#   Phase B: Marginal attribution of RG/RB/GB interactions
#     - All-broken: 2 channels shifted independently → all pairs broken
#     - R-only shifted → RG,RB broken, GB preserved
#     - G-only shifted → RG,GB broken, RB preserved
#     - B-only shifted → RB,GB broken, RG preserved
#     - GB_contrib = L2²(R-only) - L2²(all-broken)
#
#   Phase C: Additivity validation
#     - RG + RB + GB ≈ L2²(orig) - L2²(all-broken)
#
# Usage:
#   python -m suppression_test.step08_channel_attribution \
#       --model_state_path /path/to/best_model.pt \
#       --sae_ckpt /path/to/sae_checkpoint.pt \
#       --save_dir /path/to/MoCo_seed42 \
#       --shard_root /path/to/wds_shards_tar \
#       --samples_per_class 500
# ==============================================================================

# restore token norm 해야해. 그게 더 효과가 좋다. 여기서는 하고 있다. 즉 token L2 norm cnn에서 곱해주는 방식.

# python -m suppression_test.step08_channel_attribution \
#     --model_state_path /home/ubuntu/model-east3/outputs/MoCo_seed87/best_model.pt \
#     --sae_ckpt /home/ubuntu/model-east3/outputs/MoCo_seed87/SAE_seed123_no_L2norm_loss/stage5_out_d8192_gated_sp800.0_aux0.03125_tied_ep008.pt \
#     --save_dir /home/ubuntu/model-east3/outputs/MoCo_seed87 \
#     --shard_root /home/ubuntu/model-east3/wds_shards_tar \
#     --samples_per_class 500 \
#     --seam_margin 4 \
#     --dead_threshold 1e-5

import os
import sys
import csv
import random
import argparse
from typing import List, Dict
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


### PHASE A: Direction Invariance Validation (6 conditions) 에서
### mean±std 이거는 per (image, neruon) ratio의 전체 median.
### median_ratio 이거는 뉴런마다.


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
# Seam masks — exclude swap-boundary artifacts from L2² computation
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
    return (lr * ud).view(1, 1, H, W)


# ==============================================================================
# SAE forward → full activation maps
# ==============================================================================
@torch.no_grad()
def get_sae_activation_maps(
    encoder: Encoder, sae: GatedSAE,
    x: torch.Tensor, which_layer: str,
) -> torch.Tensor:
    """Returns act_maps: (B, d_sae, H, W) — full spatial, token-norm restored."""
    fmap = encoder.forward_feature_maps(x, which=which_layer)
    B, C, H, W = fmap.shape

    gap = fmap.mean(dim=(2, 3))
    gap_norm = gap.norm(dim=1, keepdim=True).view(B, 1, 1, 1).clamp_min(1e-12)
    fmap_normed = fmap / gap_norm

    tokens = fmap_normed.permute(0, 2, 3, 1).contiguous()
    flat_tokens = tokens.view(-1, C)
    flat_tokens = flat_tokens - flat_tokens.mean(dim=0, keepdim=True)

    # Save per-token L2 norms before normalization
    token_l2_norms = flat_tokens.norm(dim=1, keepdim=True).clamp_min(1e-12)
    flat_tokens = F.normalize(flat_tokens, dim=1, eps=1e-12)

    chunk_size = 8192
    acts_list = []
    for start in range(0, flat_tokens.size(0), chunk_size):
        end = min(start + chunk_size, flat_tokens.size(0))
        _, chunk_acts, _, _, _ = sae(flat_tokens[start:end])
        acts_list.append(chunk_acts.float())
    acts = torch.cat(acts_list, dim=0)              # (B*H*W, d_sae)

    # Restore per-token L2 norms
    acts = acts * token_l2_norms

    d_sae = acts.shape[1]
    return acts.view(B, H, W, d_sae).permute(0, 3, 1, 2)  # (B, d_sae, H, W)




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

    p.add_argument("--seam_margin", type=int, default=4)
    p.add_argument("--dead_threshold", type=float, default=5e-5)
    p.add_argument("--top_k_per_neuron", type=int, default=100)

    # DEG-based neuron filtering
    p.add_argument("--de_adj_p", type=float, default=0.05)
    p.add_argument("--de_min_log2fc", type=float, default=1.5)
    p.add_argument("--de_top_k", type=int, default=0,
                   help="If >0, keep only top-K DE neurons by |log2fc|")

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

    # Build superclass labels (needed for DE filtering)
    superclasses = [refs[idx].superclass for idx in ref_indices]

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

    top_k = args.top_k_per_neuron
    margin = args.seam_margin

    # ══════════════════════════════════════════════════════════════
    # Collect GAP values for top-K selection + build seam mask
    # ══════════════════════════════════════════════════════════════
    logger.info(f"\n{'='*60}")
    logger.info("Collecting GAP values for top-K selection...")
    logger.info(f"{'='*60}")

    gap_all_list = []
    seam_mask = None  # built on first batch from activation map resolution

    for batch in tqdm(loader, desc="gap_collect", leave=True):
        if batch is None:
            continue
        x, y, *_ = batch
        if x.numel() < 1:
            continue
        x_orig = x.to(device, non_blocking=True).contiguous(
            memory_format=torch.channels_last)
        with torch.amp.autocast(**autocast_kwargs):
            act_maps = get_sae_activation_maps(encoder, sae, x_orig, which_layer)
        # Build seam mask from activation map resolution (once)
        if seam_mask is None:
            _, _, H_act, W_act = act_maps.shape
            seam_mask = build_combined_seam_mask(H_act, W_act, margin, device)
            n_valid = int(seam_mask.sum().item())
            logger.info(f"  Act map: {H_act}x{W_act}, seam margin={margin}, "
                        f"valid pixels: {n_valid}/{H_act*W_act}")
        gap_vals = act_maps.mean(dim=(2, 3))  # (B, d_sae)
        gap_vals = F.normalize(gap_vals, dim=1)  # L2 normalize (match extract_features)
        gap_all_list.append(gap_vals.cpu().float().numpy())

    gap_all = np.concatenate(gap_all_list, axis=0)   # (N, d_sae)
    N_total = gap_all.shape[0]
    k_actual = min(top_k, N_total)
    logger.info(f"  Total images: {N_total}, top-K: {k_actual}")

    # Pre-compute top-K indices per neuron
    topk_indices = np.zeros((d_sae, k_actual), dtype=np.int64)
    for n_i in range(d_sae):
        if not alive_mask[n_i]:
            continue
        topk_indices[n_i] = np.argpartition(
            gap_all[:, n_i], -k_actual)[-k_actual:]

    del gap_all_list
    logger.info("  Top-K indices ready.")

    # ══════════════════════════════════════════════════════════════
    # DE-based neuron filtering (union of 4 comparisons on GAP values)
    # Uses compute_de_neurons from dpt_kendall for consistency
    # ══════════════════════════════════════════════════════════════
    logger.info(f"\n{'='*60}")
    logger.info("DE-based neuron selection (union: AllMut + per-mutation)")
    logger.info(f"{'='*60}")

    from kendall_correlation_coefficient.dpt_kendall import compute_de_neurons

    sc_arr = np.array(superclasses)
    alive_indices = np.where(alive_mask)[0]
    gap_alive = gap_all[:, alive_mask]   # (N, n_alive) — BH on alive only

    logger.info(f"  Control: {(sc_arr == 'Control').sum()}, "
                f"alive neurons for DE: {gap_alive.shape[1]}")

    # Run 4 comparisons and union (mask over alive neurons)
    de_mask_full = np.zeros(d_sae, dtype=bool)
    de_log2fc_full = np.zeros(d_sae)

    comparisons = [
        ("AllMut",  [("AllMut" if s != "Control" else "Control") for s in superclasses]),
        ("SNCA",    superclasses),
        ("GBA",     superclasses),
        ("LRRK2",   superclasses),
    ]

    for target_name, sc_list in comparisons:
        # Check if enough target samples exist
        sc_check = np.array(sc_list)
        if target_name == "AllMut":
            n_target = int((sc_check == "AllMut").sum())
        else:
            n_target = int((sc_check == target_name).sum())
        if n_target < 5:
            logger.info(f"    {target_name:10s}: skipped (n={n_target})")
            continue

        de_result = compute_de_neurons(
            gap_alive, sc_list, target_name,
            adj_p_threshold=args.de_adj_p,
            min_log2fc=args.de_min_log2fc,
        )
        # Map alive-space mask back to full d_sae
        mask_alive = de_result["mask"]   # (n_alive,)
        mask_full = np.zeros(d_sae, dtype=bool)
        mask_full[alive_indices[mask_alive]] = True
        de_mask_full |= mask_full

        if target_name == "AllMut":
            # Map log2fc back to full d_sae for reference
            de_log2fc_full[alive_indices] = de_result["log2fc"]

    de_mask = de_mask_full

    if args.de_top_k > 0 and de_mask.sum() > args.de_top_k:
        sig_idx = np.where(de_mask)[0]
        abs_fc = np.abs(de_log2fc_full[sig_idx])
        top_k_idx = sig_idx[np.argsort(abs_fc)[::-1][:args.de_top_k]]
        de_mask = np.zeros_like(de_mask)
        de_mask[top_k_idx] = True

    n_de = int(de_mask.sum())
    n_ctrl_high = int((de_mask & (de_log2fc_full < 0)).sum())
    n_mut_high = int((de_mask & (de_log2fc_full > 0)).sum())
    logger.info(f"\n  Union DE significant: {n_de} neurons")
    logger.info(f"    (AllMut log2fc ref) Control-high: {n_ctrl_high}, Mut-high: {n_mut_high}")

    # Combine: must be alive AND DE-significant
    alive_mask = alive_mask & de_mask
    n_alive = int(alive_mask.sum())
    logger.info(f"  Final analysis neurons: {n_alive} (alive & DE-significant)")

    if n_alive < 2:
        logger.warning("Too few neurons after DE filtering! Falling back to alive-only.")
        alive_mask = usage_ema >= args.dead_threshold
        n_alive = int(alive_mask.sum())

    # ══════════════════════════════════════════════════════════════
    # Define perturbation conditions
    # ══════════════════════════════════════════════════════════════
    # Channel indices: R=0, G=1, B=2

    # --- Phase A: Direction Invariance (6 "all-broken" conditions) ---
    # Each shifts 2 channels in different directions, 1 unchanged.
    # All 3 pairwise interactions are broken in every case.
    inv_conditions = {
        "Rlr_Gud":  lambda x: ud_swap_channels(lr_swap_channels(x, [0]), [1]),
        "Rud_Glr":  lambda x: lr_swap_channels(ud_swap_channels(x, [0]), [1]),
        "Rlr_Bud":  lambda x: ud_swap_channels(lr_swap_channels(x, [0]), [2]),
        "Rud_Blr":  lambda x: lr_swap_channels(ud_swap_channels(x, [0]), [2]),
        "Glr_Bud":  lambda x: ud_swap_channels(lr_swap_channels(x, [1]), [2]),
        "Gud_Blr":  lambda x: lr_swap_channels(ud_swap_channels(x, [1]), [2]),
    }

    # --- Phase B: Attribution conditions ---
    # all_broken reuses first invariance condition
    attribution_conditions = {
        "R_only_lr":   lambda x: lr_swap_channels(x, [0]),             # GB preserved
        "R_only_ud":   lambda x: ud_swap_channels(x, [0]),             # GB preserved
        "G_only_lr":   lambda x: lr_swap_channels(x, [1]),             # RB preserved
        "G_only_ud":   lambda x: ud_swap_channels(x, [1]),             # RB preserved
        "B_only_lr":   lambda x: lr_swap_channels(x, [2]),             # RG preserved
        "B_only_ud":   lambda x: ud_swap_channels(x, [2]),             # RG preserved
    }

    # ══════════════════════════════════════════════════════════════
    # Helper: collect per-image L2² and GAP for a perturbation
    # ══════════════════════════════════════════════════════════════
    n_active_pixels = float(seam_mask.sum().item())  # for GAP denominator

    def collect_spatial(perturb_fn, desc=""):
        """Returns dict with 'l2sq' and 'gap', each (N, d_sae).

        l2sq(n, img) = sum_{h,w} [act(n,h,w) * seam_mask(h,w)]²
        gap(n, img)  = mean_{masked h,w} act(n,h,w)
        """
        l2sq_list, gap_list = [], []
        for batch in tqdm(loader, desc=desc, leave=True):
            if batch is None:
                continue
            x, y, *_ = batch
            if x.numel() < 1:
                continue
            x_dev = x.to(device, non_blocking=True).contiguous(
                memory_format=torch.channels_last)
            if perturb_fn is not None:
                x_dev = perturb_fn(x_dev).contiguous(
                    memory_format=torch.channels_last)
            with torch.amp.autocast(**autocast_kwargs):
                act_maps = get_sae_activation_maps(
                    encoder, sae, x_dev, which_layer)  # (B, d_sae, H, W)
            masked = act_maps * seam_mask             # broadcast (1,1,H,W)
            l2sq = (masked ** 2).sum(dim=(2, 3))      # (B, d_sae)
            gap = masked.sum(dim=(2, 3)) / n_active_pixels  # (B, d_sae)
            l2sq_list.append(l2sq.cpu().float().numpy())
            gap_list.append(gap.cpu().float().numpy())
        return {
            "l2sq": np.concatenate(l2sq_list, axis=0),
            "gap":  np.concatenate(gap_list, axis=0),
        }

    # ══════════════════════════════════════════════════════════════
    # Collect orig — needed for Phase C
    # ══════════════════════════════════════════════════════════════
    logger.info(f"\n{'='*60}")
    logger.info("Collecting L2² + GAP for original images (seam-masked)...")
    logger.info(f"{'='*60}")
    orig_data = collect_spatial(None, desc="original")
    l2sq_orig = orig_data["l2sq"]  # (N, d_sae)
    gap_orig = orig_data["gap"]    # (N, d_sae)

    # ══════════════════════════════════════════════════════════════
    # PHASE A: Direction Invariance (all 6 conditions, pairwise)
    # ══════════════════════════════════════════════════════════════
    logger.info(f"\n{'='*60}")
    logger.info("PHASE A: Direction Invariance Validation (6 conditions)")
    logger.info(f"{'='*60}")

    # Collect L2² + GAP for all 6 conditions
    inv_l2sq = {}    # name → (N, d_sae)
    inv_gap = {}     # name → (N, d_sae)
    inv_names = list(inv_conditions.keys())
    for cond_name, perturb_fn in inv_conditions.items():
        logger.info(f"\n  Collecting: {cond_name}")
        data = collect_spatial(perturb_fn, desc=cond_name)
        inv_l2sq[cond_name] = data["l2sq"]
        inv_gap[cond_name] = data["gap"]

    # Helper: compute per-neuron median ratio between two conditions
    def compute_pairwise_ratio(l2sq_a, l2sq_b):
        """Returns per-neuron median ratio and flat array of all per-(img,neuron) ratios."""
        per_neuron = np.full(d_sae, np.nan)
        all_r = []
        for n_i in range(d_sae):
            if not alive_mask[n_i]:
                continue
            idx = topk_indices[n_i]
            a = l2sq_a[idx, n_i]
            b = l2sq_b[idx, n_i]
            valid = (a > 1e-10) & (b > 1e-10)
            if valid.sum() < 2:
                continue
            ratios = a[valid] / b[valid]
            per_neuron[n_i] = np.median(ratios)
            all_r.append(ratios)
        flat = np.concatenate(all_r) if all_r else np.array([])
        return per_neuron, flat

    # Pairwise comparison — all 15 pairs
    logger.info(f"\n  {'Pair':<20s}  {'median_ratio':>12s}  {'mean±std':>16s}  {'[0.8-1.2]':>10s}")
    logger.info(f"  {'-'*20}  {'-'*12}  {'-'*16}  {'-'*10}")

    first_pair_flat = None
    first_pair_name = None
    for i in range(len(inv_names)):
        for j in range(i + 1, len(inv_names)):
            na, nb = inv_names[i], inv_names[j]
            pn_ratio, flat_ratio = compute_pairwise_ratio(
                inv_l2sq[na], inv_l2sq[nb])
            if len(flat_ratio) == 0:
                continue
            valid_pn = pn_ratio[alive_mask & np.isfinite(pn_ratio)]
            in_band = ((flat_ratio > 0.8) & (flat_ratio < 1.2)).sum()
            pct = in_band / len(flat_ratio) * 100
            logger.info(f"  {na} vs {nb:<10s}  "
                        f"{np.median(flat_ratio):12.4f}  "
                        f"{flat_ratio.mean():.4f}±{flat_ratio.std():.4f}  "
                        f"{pct:8.1f}%")
            if first_pair_flat is None:
                first_pair_flat = flat_ratio
                first_pair_name = f"{na} vs {nb}"

    # Detailed stats for the first pair
    if first_pair_flat is not None:
        logger.info(f"\n  Detailed: {first_pair_name}")
        logger.info(f"    N ratios: {len(first_pair_flat)}")
        logger.info(f"    5th/95th pctl: [{np.percentile(first_pair_flat, 5):.4f}, "
                    f"{np.percentile(first_pair_flat, 95):.4f}]")

        plot_histogram(
            first_pair_flat,
            f"Direction Invariance – {first_pair_name}",
            "L2²(A) / L2²(B)",
            os.path.join(out_dir, "invariance_ratio_per_image.png"),
            alive_mask=None, vline=1.0, dpi=args.dpi)

    # Per-neuron median ratio (first pair for histogram)
    pn_median_first, _ = compute_pairwise_ratio(
        inv_l2sq[inv_names[0]], inv_l2sq[inv_names[1]])
    plot_histogram(
        pn_median_first,
        "Direction Invariance – per-neuron median ratio",
        "median(L2²(A) / L2²(B))",
        os.path.join(out_dir, "invariance_ratio_per_neuron.png"),
        alive_mask=alive_mask, vline=1.0, dpi=args.dpi)

    # ══════════════════════════════════════════════════════════════
    # PHASE B: Attribution (averaged over all direction variants)
    # ══════════════════════════════════════════════════════════════
    logger.info(f"\n{'='*60}")
    logger.info("PHASE B: Channel Interaction Attribution (direction-averaged)")
    logger.info(f"{'='*60}")

    # all_broken: average all 6 Phase A conditions
    l2sq_all = np.mean([inv_l2sq[n] for n in inv_names], axis=0)
    gap_all_broken = np.mean([inv_gap[n] for n in inv_names], axis=0)
    logger.info(f"  all_broken: averaged {len(inv_names)} Phase A conditions")

    # Single-channel shifts: collect both directions, then average
    l2sq_conds = {"all_broken": l2sq_all}
    gap_conds = {"all_broken": gap_all_broken}
    single_shift_pairs = [
        ("R_only", ["R_only_lr", "R_only_ud"]),   # GB preserved
        ("G_only", ["G_only_lr", "G_only_ud"]),   # RB preserved
        ("B_only", ["B_only_lr", "B_only_ud"]),   # RG preserved
    ]

    for avg_name, cond_names in single_shift_pairs:
        l2_variants, gap_variants = [], []
        for cname in cond_names:
            logger.info(f"  Collecting: {cname}")
            data = collect_spatial(attribution_conditions[cname], desc=cname)
            l2_variants.append(data["l2sq"])
            gap_variants.append(data["gap"])
        l2sq_conds[avg_name] = np.mean(l2_variants, axis=0)
        gap_conds[avg_name] = np.mean(gap_variants, axis=0)
        logger.info(f"  → {avg_name}: averaged {len(l2_variants)} directions")

    # Per-neuron top-K mean
    topk_orig = np.zeros(d_sae)
    topk_all = np.zeros(d_sae)
    topk_R = np.zeros(d_sae)
    topk_G = np.zeros(d_sae)
    topk_B = np.zeros(d_sae)
    for n_i in range(d_sae):
        if not alive_mask[n_i]:
            continue
        idx = topk_indices[n_i]
        topk_orig[n_i] = l2sq_orig[idx, n_i].mean()
        topk_all[n_i]  = l2sq_all[idx, n_i].mean()
        topk_R[n_i]    = l2sq_conds["R_only"][idx, n_i].mean()
        topk_G[n_i]    = l2sq_conds["G_only"][idx, n_i].mean()
        topk_B[n_i]    = l2sq_conds["B_only"][idx, n_i].mean()

    # ── Compute contributions ──
    # GB_contrib = L2²(R_only) - L2²(all_broken)  (GB preserved when only R shifts)
    # RB_contrib = L2²(G_only) - L2²(all_broken)  (RB preserved when only G shifts)
    # RG_contrib = L2²(B_only) - L2²(all_broken)  (RG preserved when only B shifts)
    contrib_GB = topk_R - topk_all
    contrib_RB = topk_G - topk_all
    contrib_RG = topk_B - topk_all

    total_spatial = topk_orig - topk_all

    logger.info(f"\n{'='*60}")
    logger.info(f"Attribution Results ({n_alive} alive neurons)")
    logger.info(f"{'='*60}")

    logger.info(f"\n  L2²(orig) mean: {topk_orig[alive_mask].mean():.6f}")
    logger.info(f"  L2²(all_broken) mean: {topk_all[alive_mask].mean():.6f}")
    logger.info(f"  Total spatial effect mean: {total_spatial[alive_mask].mean():.6f}")

    for pair, contrib in [("RG", contrib_RG), ("RB", contrib_RB), ("GB", contrib_GB)]:
        vals = contrib[alive_mask]
        n_neg = int((vals < 0).sum())
        logger.info(f"  {pair}_contrib: mean={vals.mean():.6f}, "
                    f"median={np.median(vals):.6f}, std={vals.std():.6f}, "
                    f"negative={n_neg}/{len(vals)} ({n_neg/len(vals)*100:.1f}%)")

    # L2² sanity check: should always be ≥ 0 (sum of squares)
    for name, arr in [("orig", topk_orig), ("all_broken", topk_all),
                      ("R_only", topk_R), ("G_only", topk_G), ("B_only", topk_B)]:
        neg_count = int((arr[alive_mask] < 0).sum())
        if neg_count > 0:
            logger.warning(f"  ⚠️ L2²({name}) has {neg_count} negative values!")
        else:
            logger.info(f"  L2²({name}): all ≥ 0 ✓")

    # ══════════════════════════════════════════════════════════════
    # PHASE C: Additivity Validation
    # ══════════════════════════════════════════════════════════════
    logger.info(f"\n{'='*60}")
    logger.info("PHASE C: Additivity Validation")
    logger.info(f"{'='*60}")

    sum_contribs = contrib_RG + contrib_RB + contrib_GB

    # --- Standard additivity ---
    logger.info(f"\n  ── Standard additivity (raw contributions) ──")
    agg_sum = sum_contribs[alive_mask].sum()
    agg_total = total_spatial[alive_mask].sum()
    logger.info(f"  Aggregate ratio: sum_contribs/total_spatial = "
                f"{agg_sum:.4f} / {agg_total:.4f} = {agg_sum / (agg_total + 1e-12):.4f}")

    # Per-neuron ratio (exclude neurons with negligible total_spatial)
    additivity_ratio = np.full(d_sae, np.nan)
    spatial_threshold = np.percentile(total_spatial[alive_mask], 10)  # exclude bottom 10%
    spatial_threshold = max(spatial_threshold, 1e-6)
    for n_i in range(d_sae):
        if not alive_mask[n_i]:
            continue
        if total_spatial[n_i] > spatial_threshold:
            additivity_ratio[n_i] = sum_contribs[n_i] / total_spatial[n_i]

    valid_add = additivity_ratio[alive_mask]
    valid_add = valid_add[np.isfinite(valid_add)]
    logger.info(f"\n  Per-neuron ratio (total_spatial > {spatial_threshold:.6f}):")
    logger.info(f"    N neurons: {len(valid_add)}")
    logger.info(f"    mean={valid_add.mean():.4f}, median={np.median(valid_add):.4f}")
    logger.info(f"    5th/95th pctl: [{np.percentile(valid_add, 5):.4f}, "
                f"{np.percentile(valid_add, 95):.4f}]")
    in_band = ((valid_add > 0.8) & (valid_add < 1.2)).sum()
    logger.info(f"    [0.8-1.2]: {in_band}/{len(valid_add)} "
                f"({in_band/len(valid_add)*100:.1f}%)")

    # --- Absolute-value additivity ---
    # If a pair interaction suppresses activation (negative contrib),
    # sum(raw) underestimates total effect. Use |contrib| instead.
    logger.info(f"\n  ── Absolute-value additivity (|contrib|) ──")
    abs_sum_contribs = np.abs(contrib_RG) + np.abs(contrib_RB) + np.abs(contrib_GB)

    abs_agg_sum = abs_sum_contribs[alive_mask].sum()
    logger.info(f"  Aggregate |ratio|: sum(|contribs|)/total_spatial = "
                f"{abs_agg_sum:.4f} / {agg_total:.4f} = {abs_agg_sum / (agg_total + 1e-12):.4f}")

    abs_additivity_ratio = np.full(d_sae, np.nan)
    for n_i in range(d_sae):
        if not alive_mask[n_i]:
            continue
        if total_spatial[n_i] > spatial_threshold:
            abs_additivity_ratio[n_i] = abs_sum_contribs[n_i] / total_spatial[n_i]

    valid_abs = abs_additivity_ratio[alive_mask]
    valid_abs = valid_abs[np.isfinite(valid_abs)]
    logger.info(f"  Per-neuron |ratio| (total_spatial > {spatial_threshold:.6f}):")
    logger.info(f"    N neurons: {len(valid_abs)}")
    logger.info(f"    mean={valid_abs.mean():.4f}, median={np.median(valid_abs):.4f}")
    logger.info(f"    5th/95th pctl: [{np.percentile(valid_abs, 5):.4f}, "
                f"{np.percentile(valid_abs, 95):.4f}]")
    abs_in_band = ((valid_abs > 0.8) & (valid_abs < 1.2)).sum()
    logger.info(f"    [0.8-1.2]: {abs_in_band}/{len(valid_abs)} "
                f"({abs_in_band/len(valid_abs)*100:.1f}%)")

    # --- Per-image linearity ratio (orig as denominator — robust) ---
    # For each (image, neuron): (all_broken + RG + RB + GB) / L2²(orig)
    # L2²(orig) is always large for top-K images → no near-zero denominator
    logger.info(f"\n  ── Per-image linearity ratio: reconstructed / L2²(orig) ──")

    l2sq_R_arr = l2sq_conds["R_only"]  # (N, d_sae) — averaged lr+ud
    l2sq_G_arr = l2sq_conds["G_only"]
    l2sq_B_arr = l2sq_conds["B_only"]
    l2sq_all_arr = l2sq_all  # all_broken, (N, d_sae)

    per_neuron_linearity = np.full(d_sae, np.nan)
    for n_i in range(d_sae):
        if not alive_mask[n_i]:
            continue
        idx = topk_indices[n_i]  # top-K image indices for this neuron
        orig_vals = l2sq_orig[idx, n_i]      # (K,)
        all_vals  = l2sq_all_arr[idx, n_i]   # (K,)
        R_vals    = l2sq_R_arr[idx, n_i]     # (K,)
        G_vals    = l2sq_G_arr[idx, n_i]     # (K,)
        B_vals    = l2sq_B_arr[idx, n_i]     # (K,)

        # Per-image contributions
        rg_i = B_vals - all_vals   # RG preserved when B shifts
        rb_i = G_vals - all_vals   # RB preserved when G shifts
        gb_i = R_vals - all_vals   # GB preserved when R shifts

        # Reconstructed = all_broken + pairwise contributions
        reconstructed = all_vals + rg_i + rb_i + gb_i
        # = R_vals + G_vals + B_vals - 2 * all_vals

        # Ratio per image (orig is denominator, always > 0 for top-K)
        valid_orig = orig_vals > 1e-10
        if valid_orig.sum() < 2:
            continue
        ratios = reconstructed[valid_orig] / orig_vals[valid_orig]
        per_neuron_linearity[n_i] = np.mean(ratios)

    valid_lin = per_neuron_linearity[alive_mask]
    valid_lin = valid_lin[np.isfinite(valid_lin)]
    logger.info(f"    N neurons: {len(valid_lin)}")
    logger.info(f"    mean={valid_lin.mean():.4f}, median={np.median(valid_lin):.4f}")
    logger.info(f"    5th/95th pctl: [{np.percentile(valid_lin, 5):.4f}, "
                f"{np.percentile(valid_lin, 95):.4f}]")
    lin_in_band = ((valid_lin > 0.8) & (valid_lin < 1.2)).sum()
    logger.info(f"    [0.8-1.2]: {lin_in_band}/{len(valid_lin)} "
                f"({lin_in_band/len(valid_lin)*100:.1f}%)")
    logger.info(f"    Interpretation: 1.0 = perfect pairwise decomposition, "
                f"<1.0 = 3-way interaction exists")

    # ══════════════════════════════════════════════════════════════
    # PHASE D: GAP-based Attribution (parallel to L2²)
    # ══════════════════════════════════════════════════════════════
    logger.info(f"\n{'='*60}")
    logger.info("PHASE D: GAP-based Attribution (direct feature metric)")
    logger.info(f"{'='*60}")

    # Per-neuron top-K mean (GAP)
    g_topk_orig = np.zeros(d_sae)
    g_topk_all = np.zeros(d_sae)
    g_topk_R = np.zeros(d_sae)
    g_topk_G = np.zeros(d_sae)
    g_topk_B = np.zeros(d_sae)
    for n_i in range(d_sae):
        if not alive_mask[n_i]:
            continue
        idx = topk_indices[n_i]  # same top-K (based on orig L2²)
        g_topk_orig[n_i] = gap_orig[idx, n_i].mean()
        g_topk_all[n_i]  = gap_conds["all_broken"][idx, n_i].mean()
        g_topk_R[n_i]    = gap_conds["R_only"][idx, n_i].mean()
        g_topk_G[n_i]    = gap_conds["G_only"][idx, n_i].mean()
        g_topk_B[n_i]    = gap_conds["B_only"][idx, n_i].mean()

    g_contrib_GB = g_topk_R - g_topk_all
    g_contrib_RB = g_topk_G - g_topk_all
    g_contrib_RG = g_topk_B - g_topk_all
    g_total_spatial = g_topk_orig - g_topk_all

    logger.info(f"\n  GAP(orig) mean: {g_topk_orig[alive_mask].mean():.6f}")
    logger.info(f"  GAP(all_broken) mean: {g_topk_all[alive_mask].mean():.6f}")
    logger.info(f"  GAP total spatial effect mean: {g_total_spatial[alive_mask].mean():.6f}")

    for pair, contrib in [("RG", g_contrib_RG), ("RB", g_contrib_RB), ("GB", g_contrib_GB)]:
        vals = contrib[alive_mask]
        n_neg = int((vals < 0).sum())
        logger.info(f"  GAP {pair}_contrib: mean={vals.mean():.6f}, "
                    f"median={np.median(vals):.6f}, std={vals.std():.6f}, "
                    f"negative={n_neg}/{len(vals)} ({n_neg/len(vals)*100:.1f}%)")

    # GAP additivity
    g_sum_contribs = g_contrib_RG + g_contrib_RB + g_contrib_GB
    g_agg_sum = g_sum_contribs[alive_mask].sum()
    g_agg_total = g_total_spatial[alive_mask].sum()
    logger.info(f"\n  GAP aggregate ratio: {g_agg_sum:.4f} / {g_agg_total:.4f} = "
                f"{g_agg_sum / (g_agg_total + 1e-12):.4f}")

    # GAP per-image linearity (orig as denominator)
    g_linearity = np.full(d_sae, np.nan)
    for n_i in range(d_sae):
        if not alive_mask[n_i]:
            continue
        idx = topk_indices[n_i]
        go = gap_orig[idx, n_i]
        ga = gap_conds["all_broken"][idx, n_i]
        gR = gap_conds["R_only"][idx, n_i]
        gG = gap_conds["G_only"][idx, n_i]
        gB = gap_conds["B_only"][idx, n_i]
        reconstructed = gR + gG + gB - 2 * ga
        valid_g = go > 1e-10
        if valid_g.sum() < 2:
            continue
        g_linearity[n_i] = np.mean(reconstructed[valid_g] / go[valid_g])

    g_valid = g_linearity[alive_mask]
    g_valid = g_valid[np.isfinite(g_valid)]
    logger.info(f"\n  GAP per-image linearity: reconstructed / GAP(orig)")
    logger.info(f"    N neurons: {len(g_valid)}")
    logger.info(f"    mean={g_valid.mean():.4f}, median={np.median(g_valid):.4f}")
    logger.info(f"    5th/95th pctl: [{np.percentile(g_valid, 5):.4f}, "
                f"{np.percentile(g_valid, 95):.4f}]")
    g_in_band = ((g_valid > 0.8) & (g_valid < 1.2)).sum()
    logger.info(f"    [0.8-1.2]: {g_in_band}/{len(g_valid)} "
                f"({g_in_band/len(g_valid)*100:.1f}%)")

    # GAP fractions
    g_RG_clip = np.maximum(g_contrib_RG, 0)
    g_RB_clip = np.maximum(g_contrib_RB, 0)
    g_GB_clip = np.maximum(g_contrib_GB, 0)
    g_total_clip = g_RG_clip + g_RB_clip + g_GB_clip + 1e-12
    g_frac_RG = g_RG_clip / g_total_clip
    g_frac_RB = g_RB_clip / g_total_clip
    g_frac_GB = g_GB_clip / g_total_clip
    for pair, frac in [("RG", g_frac_RG), ("RB", g_frac_RB), ("GB", g_frac_GB)]:
        vals = frac[alive_mask]
        logger.info(f"  GAP {pair} fraction: mean={vals.mean():.4f}, std={vals.std():.4f}")

    # ── Interaction fractions (normalized to sum=1) — L2² ──
    total_contrib = contrib_RG + contrib_RB + contrib_GB
    # Clip negatives to 0 before computing fractions
    contrib_RG_clip = np.maximum(contrib_RG, 0)
    contrib_RB_clip = np.maximum(contrib_RB, 0)
    contrib_GB_clip = np.maximum(contrib_GB, 0)
    total_clip = contrib_RG_clip + contrib_RB_clip + contrib_GB_clip + 1e-12

    frac_RG = contrib_RG_clip / total_clip
    frac_RB = contrib_RB_clip / total_clip
    frac_GB = contrib_GB_clip / total_clip

    for pair, frac in [("RG", frac_RG), ("RB", frac_RB), ("GB", frac_GB)]:
        vals = frac[alive_mask]
        logger.info(f"  {pair} fraction: mean={vals.mean():.4f}, std={vals.std():.4f}")

    # ── Save ──
    npz_path = os.path.join(out_dir, "channel_attribution_results.npz")
    np.savez_compressed(npz_path,
                        # Phase A: invariance (first pair)
                        inv_ratio_all=first_pair_flat,
                        inv_ratio_per_neuron=pn_median_first,
                        # Phase B: attribution
                        l2sq_orig=topk_orig,
                        l2sq_all_broken=topk_all,
                        l2sq_R_only=topk_R,
                        l2sq_G_only=topk_G,
                        l2sq_B_only=topk_B,
                        contrib_RG=contrib_RG,
                        contrib_RB=contrib_RB,
                        contrib_GB=contrib_GB,
                        total_spatial=total_spatial,
                        # Phase C: additivity
                        additivity_ratio=additivity_ratio,
                        # Fractions
                        frac_RG=frac_RG, frac_RB=frac_RB, frac_GB=frac_GB,
                        # Metadata
                        alive_mask=alive_mask, usage_ema=usage_ema,
                        de_mask=de_mask, de_log2fc=de_log2fc_full,
                        top_k=k_actual, seam_margin=margin)
    logger.info(f"\nSaved: {npz_path}")

    # ══════════════════════════════════════════════════════════════
    # Plots
    # ══════════════════════════════════════════════════════════════

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

    # ── Additivity ratio histogram ──
    plot_histogram(
        additivity_ratio,
        "Additivity Check: sum(RG+RB+GB) / total_spatial",
        "Ratio",
        os.path.join(out_dir, "hist_additivity_ratio.png"),
        alive_mask=alive_mask, vline=1.0, dpi=args.dpi)

    # ── Raw contribution histograms ──
    for name, contrib in [("RG", contrib_RG), ("RB", contrib_RB), ("GB", contrib_GB)]:
        plot_histogram(
            contrib, f"{name} Raw Contribution (L2² diff)",
            "L2² (single-channel) - L2² (all-broken)",
            os.path.join(out_dir, f"hist_contrib_{name}.png"),
            alive_mask=alive_mask, dpi=args.dpi)

    logger.info(f"\n{'='*60}")
    logger.info("Channel spatial interaction attribution complete!")
    logger.info(f"Results: {out_dir}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
