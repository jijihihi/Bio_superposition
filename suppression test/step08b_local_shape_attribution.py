# ==============================================================================
# Per-Channel Local Shape Attribution via Patch Shuffling
#
# Decomposes spatial information into:
#   - Per-channel local shape (R, G, B texture/arrangement info)
#   - Cross-channel spatial relationships (from step08)
#
# Method:
#   R_local = L2²(R_lr_swap) - L2²(R_patch_shuffle)
#     R_lr: RG+RB broken, R local shape preserved
#     R_patch: RG+RB broken, R local shape destroyed
#     → difference = R local shape contribution
#
#   Additivity: all_same_perm_shuffle + R_local + G_local + B_local ≈ orig
#     all_same_perm: R,G,B patches co-move → local shape broken, cross-ch preserved
#
# Usage:
#   python -m suppression_test.step08b_local_shape_attribution \
#       --model_state_path /path/to/best_model.pt \
#       --sae_ckpt /path/to/sae_checkpoint.pt \
#       --save_dir /path/to/MoCo_seed87 \
#       --shard_root /path/to/wds_shards_tar \
#       --samples_per_class 500 \
#       --seam_margin 4 \
#       --patch_size 8
# ==============================================================================

import argparse
import csv
import os
import random
import sys
from typing import List

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

_IN_COLAB = "google.colab" in sys.modules
if not _IN_COLAB:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Reuse from step08
from suppression_test.step08_channel_attribution import (
    KNOWN_SHARD_ROOTS, build_combined_seam_mask, get_sae_activation_maps,
    load_split_csv, lr_swap_channels, plot_histogram, remap_uid,
    ud_swap_channels)

from sae_project.step02_logging_utils import OUT_DIM, get_logger
from sae_project.step03_data_shards import (build_uid_to_refidx,
                                            load_all_sample_refs)
from sae_project.step04_data_bank import (InMemorySixteenBitDataset,
                                          InMemoryTarBank, collate_skip_none,
                                          seed_worker)
from sae_project.step05_model_encoder import (Encoder, SupMoCoModel,
                                              parse_int_list,
                                              renorm_unit_per_out_channel_,
                                              robust_load_state_dict)
from sae_project.step06_gated_sae import GatedSAE

logger = get_logger("local_shape_attribution")


# ==============================================================================
# Patch shuffle perturbations
# ==============================================================================
def _derangement_avoiding(n, forbidden_perms):
    """Generate a derangement of [0..n-1] that doesn't match any
    forbidden permutation at any position.
    Uses Sattolo's algorithm + rejection for cross-channel conflicts.
    For n=256, k<=3: converges in ~3-8 attempts on average.
    """
    import random as _rng

    max_attempts = 500
    for _ in range(max_attempts):
        # Sattolo's derangement (guaranteed no fixed points)
        perm = list(range(n))
        for i in range(n - 1, 0, -1):
            j = _rng.randint(0, i - 1)  # j ∈ [0, i-1] → no fixed points
            perm[i], perm[j] = perm[j], perm[i]
        # Check against all forbidden perms (no shared position→destination)
        ok = True
        for fp in forbidden_perms:
            for i in range(n):
                if perm[i] == fp[i]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return perm
    # Fallback (extremely unlikely for n=256, k<=3)
    return perm


def patch_shuffle_channels(
    x: torch.Tensor,
    channels: List[int],
    patch_size: int = 8,
) -> torch.Tensor:
    """Shuffle specified channels independently using random patch permutations.
    Each channel gets its own random derangement (no patch stays in place).
    When multiple channels are shuffled, their permutations are guaranteed
    to never map the same source position to the same destination
    (cross-channel derangement).
    x: (B, 3, H, W)
    """
    B, C, H, W = x.shape
    nH = H // patch_size
    nW = W // patch_size
    n_patches = nH * nW
    out = x.clone()

    for b in range(B):
        prev_perms = []  # track perms for cross-channel derangement
        for ch in channels:
            perm = _derangement_avoiding(n_patches, prev_perms)
            prev_perms.append(perm)
            perm_t = torch.tensor(perm, device=x.device, dtype=torch.long)

            # Extract patches: (nH*ps, nW*ps) → (n_patches, ps, ps)
            ch_data = out[b, ch, : nH * patch_size, : nW * patch_size]
            patches = ch_data.view(nH, patch_size, nW, patch_size)
            patches = patches.permute(0, 2, 1, 3).contiguous()  # (nH, nW, ps, ps)
            patches = patches.view(n_patches, patch_size, patch_size)

            # Apply permutation
            patches = patches[perm_t]

            # Reshape back: (n_patches, ps, ps) → (nH*ps, nW*ps)
            patches = patches.view(nH, nW, patch_size, patch_size)
            patches = patches.permute(0, 2, 1, 3).contiguous()
            out[b, ch, : nH * patch_size, : nW * patch_size] = patches.view(
                nH * patch_size, nW * patch_size
            )

    return out


def patch_rotate_channels(
    x: torch.Tensor,
    channels: List[int],
    patch_size: int = 8,
) -> torch.Tensor:
    """Rotate each patch IN-PLACE by random {90°, 180°, 270°} per patch.
    Preserves coarse spatial density (bright regions stay where they are)
    while destroying fine orientation/shape within each patch.
    x: (B, 3, H, W)
    """
    import random as _rng

    B, C, H, W = x.shape
    nH = H // patch_size
    nW = W // patch_size
    out = x.clone()

    for b in range(B):
        for ch in channels:
            ch_data = out[b, ch, : nH * patch_size, : nW * patch_size]
            patches = ch_data.view(nH, patch_size, nW, patch_size)
            patches = patches.permute(0, 2, 1, 3).contiguous()  # (nH, nW, ps, ps)

            for i in range(nH):
                for j in range(nW):
                    k = _rng.randint(1, 3)  # 90°, 180°, or 270° (exclude 0°)
                    patches[i, j] = torch.rot90(patches[i, j], k, dims=(0, 1))

            patches = patches.permute(0, 2, 1, 3).contiguous()
            out[b, ch, : nH * patch_size, : nW * patch_size] = patches.view(
                nH * patch_size, nW * patch_size
            )

    return out


def patch_shuffle_all_same_perm(
    x: torch.Tensor,
    patch_size: int = 8,
) -> torch.Tensor:
    """Shuffle ALL channels with the SAME random permutation per image.
    Preserves cross-channel spatial relationships; destroys per-channel local shape.
    x: (B, 3, H, W)
    """
    B, C, H, W = x.shape
    nH = H // patch_size
    nW = W // patch_size
    out = x.clone()

    # Crop to exact patch grid
    crop = x[:, :, : nH * patch_size, : nW * patch_size]  # (B, C, nH*ps, nW*ps)
    # → (B, C, nH, ps, nW, ps)
    patches = crop.view(B, C, nH, patch_size, nW, patch_size)
    patches = patches.permute(0, 1, 2, 4, 3, 5).contiguous()  # (B, C, nH, nW, ps, ps)
    patches = patches.view(B, C, nH * nW, patch_size, patch_size)

    for b in range(B):
        perm = torch.randperm(nH * nW, device=x.device)
        patches[b] = patches[b, :, perm]  # same perm for all channels

    # Reshape back
    patches = patches.view(B, C, nH, nW, patch_size, patch_size)
    patches = patches.permute(0, 1, 2, 4, 3, 5).contiguous()
    out[:, :, : nH * patch_size, : nW * patch_size] = patches.view(
        B, C, nH * patch_size, nW * patch_size
    )

    return out


# ==============================================================================
# Args
# ==============================================================================
def get_args():
    p = argparse.ArgumentParser("Per-Channel Local Shape Attribution")

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
    p.add_argument(
        "--patch_size",
        type=int,
        default=8,
        help="Patch size for shuffle (img_size must be divisible)",
    )

    # DE-based neuron filtering
    p.add_argument("--de_adj_p", type=float, default=0.05)
    p.add_argument("--de_min_log2fc", type=float, default=1.5)
    p.add_argument("--de_top_k", type=int, default=0)

    p.add_argument("--which_layer", type=str, default="stage5_out")
    p.add_argument("--dpi", type=int, default=200)

    # Number of shuffle repeats for noise reduction
    p.add_argument(
        "--n_shuffle_repeats",
        type=int,
        default=3,
        help="Average over N random shuffles for stable estimates",
    )

    return p.parse_args()


# ==============================================================================
# Main
# ==============================================================================
def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    which_layer = args.which_layer

    # ── Output dir ──
    sae_parent = os.path.dirname(args.sae_ckpt)
    out_dir = os.path.join(sae_parent, "local_shape_attribution")
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Output dir: {out_dir}")

    # ── Model + SAE ──
    logger.info("Loading model + SAE...")
    blocks = parse_int_list(args.blocks, 4)
    dilations = parse_int_list(args.dilations, 4)
    model = SupMoCoModel(
        embed_dim=512,
        blocks=blocks,
        dilations=dilations,
        refine_blocks=args.refine_blocks,
        ckpt_segments=args.ckpt_segments,
        proj_layers=2,
        proj_hidden=2048,
    )
    ckpt = torch.load(args.model_state_path, map_location="cpu", weights_only=False)
    robust_load_state_dict(model, ckpt, strict=False)
    encoder = model.encoder
    encoder.eval().to(device).to(memory_format=torch.channels_last)
    renorm_unit_per_out_channel_(encoder)
    del model

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
    which_layer = sae_args.get("which_layer", args.which_layer)
    d_sae = sae.d_sae

    # alive mask
    usage_ema = sae.usage_ema.cpu().numpy()
    alive_mask = usage_ema >= args.dead_threshold
    n_alive = int(alive_mask.sum())
    logger.info(f"  d_sae={d_sae}, alive={n_alive} (threshold={args.dead_threshold})")

    autocast_kwargs = dict(device_type="cuda", enabled=torch.cuda.is_available())
    if torch.cuda.is_available():
        autocast_kwargs["dtype"] = torch.bfloat16

    # ── Data ──
    logger.info("Loading data...")
    refs = load_all_sample_refs(args.shard_root)
    uid_to_refidx = build_uid_to_refidx(refs)

    split_dir = args.save_dir or os.path.dirname(args.model_state_path)
    eval_uids = []
    for split_name in ["val_split.csv", "test_split.csv"]:
        csv_path = os.path.join(split_dir, split_name)
        if os.path.exists(csv_path):
            eval_uids.extend(load_split_csv(csv_path, args.shard_root))

    if not eval_uids:
        eval_uids = list(uid_to_refidx.keys())
        logger.info(f"  No split CSVs, using all: {len(eval_uids)}")

    eval_refidx = [uid_to_refidx[u] for u in eval_uids if u in uid_to_refidx]
    bank = InMemoryTarBank(refs, eval_refidx, args.img_size)

    from sae_project.step02_logging_utils import SUPERCLASS_MAP

    superclasses = [SUPERCLASS_MAP.get(line, "Unknown") for line in bank.lines]

    from collections import Counter

    sc_counts = Counter(superclasses)
    max_per_class = args.samples_per_class
    class_indices = {}
    for i, sc in enumerate(superclasses):
        if sc == "Unknown":
            continue
        class_indices.setdefault(sc, []).append(i)
    selected = []
    for sc, idxs in sorted(class_indices.items()):
        random.seed(42)
        chosen = random.sample(idxs, min(len(idxs), max_per_class))
        selected.extend(chosen)
        logger.info(f"  {sc}: {len(chosen)} images")
    superclasses = [superclasses[i] for i in selected]

    ds = InMemorySixteenBitDataset(bank, selected, args.img_size, augment=False)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        collate_fn=collate_skip_none,
    )

    # ── Seam mask + feature map size ──
    sample_x = (
        next(iter(loader))[0][:1]
        .to(device)
        .contiguous(memory_format=torch.channels_last)
    )
    with torch.amp.autocast(**autocast_kwargs):
        sample_act = get_sae_activation_maps(encoder, sae, sample_x, which_layer)
    _, _, Hf, Wf = sample_act.shape
    seam_mask = build_combined_seam_mask(Hf, Wf, args.seam_margin, device)
    n_active_pixels = float(seam_mask.sum().item())
    logger.info(f"  Feature map: {Hf}x{Wf}, seam_mask active: {int(n_active_pixels)}")

    # ── GAP-based data for DE filtering ──
    logger.info("Collecting GAP for DE filtering...")
    gap_all = []
    for batch in tqdm(loader, desc="GAP", leave=True):
        if batch is None:
            continue
        x, y, *_ = batch
        if x.numel() < 1:
            continue
        x_dev = x.to(device, non_blocking=True).contiguous(
            memory_format=torch.channels_last
        )
        with torch.amp.autocast(**autocast_kwargs):
            act_maps = get_sae_activation_maps(encoder, sae, x_dev, which_layer)
        gap_batch = act_maps.mean(dim=(2, 3))
        gap_all.append(gap_batch.cpu().float().numpy())
    gap_all = np.concatenate(gap_all, axis=0)  # (N, d_sae)

    # ── Top-K indices ──
    topk_orig_l2sq = np.zeros((len(gap_all), d_sae))
    # Will compute proper orig L2² below; for now use GAP for top-K
    k = args.top_k_per_neuron
    k_actual = min(k, gap_all.shape[0])
    topk_indices = np.zeros((d_sae, k_actual), dtype=int)
    for n_i in range(d_sae):
        if not alive_mask[n_i]:
            continue
        col = gap_all[:, n_i]
        topk_indices[n_i] = np.argsort(col)[::-1][:k_actual]

    # ── DE filtering ──
    logger.info(f"\n{'='*60}")
    logger.info("DE-based neuron selection")
    logger.info(f"{'='*60}")
    from kendall_correlation_coefficient.dpt_kendall import compute_de_neurons

    sc_arr = np.array(superclasses)
    alive_indices = np.where(alive_mask)[0]
    gap_alive = gap_all[:, alive_mask]

    de_mask_full = np.zeros(d_sae, dtype=bool)
    comparisons = [
        ("AllMut", [("AllMut" if s != "Control" else "Control") for s in superclasses]),
        ("SNCA", superclasses),
        ("GBA", superclasses),
        ("LRRK2", superclasses),
    ]
    for target_name, sc_list in comparisons:
        sc_check = np.array(sc_list)
        n_target = (
            int((sc_check == target_name).sum())
            if target_name != "AllMut"
            else int((sc_check == "AllMut").sum())
        )
        if n_target < 5:
            continue
        de_result = compute_de_neurons(
            gap_alive,
            sc_list,
            target_name,
            adj_p_threshold=args.de_adj_p,
            min_log2fc=args.de_min_log2fc,
        )
        mask_alive = de_result["mask"]
        mask_full = np.zeros(d_sae, dtype=bool)
        mask_full[alive_indices[mask_alive]] = True
        de_mask_full |= mask_full

    alive_mask = alive_mask & de_mask_full
    n_alive = int(alive_mask.sum())
    logger.info(f"  Final analysis neurons: {n_alive}")

    # ── Collection helper ──
    def collect_spatial(perturb_fn, desc=""):
        """Returns dict with 'l2sq' and 'gap', each (N, d_sae)."""
        l2sq_list, gap_list = [], []
        for batch in tqdm(loader, desc=desc, leave=True):
            if batch is None:
                continue
            x, y, *_ = batch
            if x.numel() < 1:
                continue
            x_dev = x.to(device, non_blocking=True).contiguous(
                memory_format=torch.channels_last
            )
            if perturb_fn is not None:
                x_dev = perturb_fn(x_dev).contiguous(memory_format=torch.channels_last)
            with torch.amp.autocast(**autocast_kwargs):
                act_maps = get_sae_activation_maps(encoder, sae, x_dev, which_layer)
            masked = act_maps * seam_mask
            l2sq = (masked**2).sum(dim=(2, 3))
            gap = masked.sum(dim=(2, 3)) / n_active_pixels
            l2sq_list.append(l2sq.cpu().float().numpy())
            gap_list.append(gap.cpu().float().numpy())
        return {
            "l2sq": np.concatenate(l2sq_list, axis=0),
            "gap": np.concatenate(gap_list, axis=0),
        }

    def collect_spatial_averaged(perturb_fn, desc="", n_repeats=1):
        """Average multiple random shuffle runs for stable estimates."""
        if n_repeats <= 1:
            return collect_spatial(perturb_fn, desc=desc)
        all_l2sq, all_gap = [], []
        for rep in range(n_repeats):
            data = collect_spatial(perturb_fn, desc=f"{desc} rep{rep+1}/{n_repeats}")
            all_l2sq.append(data["l2sq"])
            all_gap.append(data["gap"])
        return {
            "l2sq": np.mean(all_l2sq, axis=0),
            "gap": np.mean(all_gap, axis=0),
        }

    # ══════════════════════════════════════════════════════════════
    # Collect all conditions
    # ══════════════════════════════════════════════════════════════
    ps = args.patch_size
    n_rep = args.n_shuffle_repeats

    logger.info(f"\n{'='*60}")
    logger.info(f"Collecting conditions (patch_size={ps}, repeats={n_rep})")
    logger.info(f"{'='*60}")

    # Original
    orig = collect_spatial(None, desc="original")

    # Update top-K with proper L2² orig
    for n_i in range(d_sae):
        if not alive_mask[n_i]:
            continue
        col = orig["l2sq"][:, n_i]
        topk_indices[n_i] = np.argsort(col)[::-1][:k_actual]

    # Per-channel lr/ud swap (from step08 — breaks pairwise, preserves local shape)
    logger.info("\n  ── Per-channel swap (pairwise broken, local shape preserved) ──")
    R_lr = collect_spatial(lambda x: lr_swap_channels(x, [0]), desc="R_lr")
    R_ud = collect_spatial(lambda x: ud_swap_channels(x, [0]), desc="R_ud")
    G_lr = collect_spatial(lambda x: lr_swap_channels(x, [1]), desc="G_lr")
    G_ud = collect_spatial(lambda x: ud_swap_channels(x, [1]), desc="G_ud")
    B_lr = collect_spatial(lambda x: lr_swap_channels(x, [2]), desc="B_lr")
    B_ud = collect_spatial(lambda x: ud_swap_channels(x, [2]), desc="B_ud")

    # Average lr + ud per channel
    R_swap = {k: (R_lr[k] + R_ud[k]) / 2 for k in ["l2sq", "gap"]}
    G_swap = {k: (G_lr[k] + G_ud[k]) / 2 for k in ["l2sq", "gap"]}
    B_swap = {k: (B_lr[k] + B_ud[k]) / 2 for k in ["l2sq", "gap"]}

    # Per-channel patch shuffle (pairwise broken + local shape broken)
    logger.info("\n  ── Per-channel patch shuffle (pairwise + local shape broken) ──")
    R_patch = collect_spatial_averaged(
        lambda x: patch_shuffle_channels(x, [0], ps), desc="R_patch", n_repeats=n_rep
    )
    G_patch = collect_spatial_averaged(
        lambda x: patch_shuffle_channels(x, [1], ps), desc="G_patch", n_repeats=n_rep
    )
    B_patch = collect_spatial_averaged(
        lambda x: patch_shuffle_channels(x, [2], ps), desc="B_patch", n_repeats=n_rep
    )

    # All channels same-permutation shuffle (local shape broken, cross-ch preserved)
    logger.info(
        "\n  ── All-channel same-perm shuffle (local shape broken, cross-ch preserved) ──"
    )
    all_same = collect_spatial_averaged(
        lambda x: patch_shuffle_all_same_perm(x, ps),
        desc="all_same_perm",
        n_repeats=n_rep,
    )

    # ══════════════════════════════════════════════════════════════
    # Compute per-channel local shape contributions
    # ══════════════════════════════════════════════════════════════
    logger.info(f"\n{'='*60}")
    logger.info("Local Shape Attribution Results")
    logger.info(f"{'='*60}")

    for metric in ["l2sq", "gap"]:
        metric_label = "L2²" if metric == "l2sq" else "GAP"
        logger.info(f"\n  ══ {metric_label}-based ══")

        # Per-neuron top-K mean
        def topk_mean(arr):
            vals = np.zeros(d_sae)
            for n_i in range(d_sae):
                if not alive_mask[n_i]:
                    continue
                idx = topk_indices[n_i]
                vals[n_i] = arr[idx, n_i].mean()
            return vals

        tk_orig = topk_mean(orig[metric])
        tk_R_swap = topk_mean(R_swap[metric])
        tk_G_swap = topk_mean(G_swap[metric])
        tk_B_swap = topk_mean(B_swap[metric])
        tk_R_patch = topk_mean(R_patch[metric])
        tk_G_patch = topk_mean(G_patch[metric])
        tk_B_patch = topk_mean(B_patch[metric])
        tk_all_same = topk_mean(all_same[metric])

        # Local shape contributions
        R_local = tk_R_swap - tk_R_patch
        G_local = tk_G_swap - tk_G_patch
        B_local = tk_B_swap - tk_B_patch

        logger.info(f"\n  {metric_label}(orig) mean: {tk_orig[alive_mask].mean():.6f}")
        logger.info(
            f"  {metric_label}(all_same_perm) mean: {tk_all_same[alive_mask].mean():.6f}"
        )

        for ch, local, swap, patch in [
            ("R", R_local, tk_R_swap, tk_R_patch),
            ("G", G_local, tk_G_swap, tk_G_patch),
            ("B", B_local, tk_B_swap, tk_B_patch),
        ]:
            vals = local[alive_mask]
            n_neg = int((vals < 0).sum())
            logger.info(
                f"\n  {ch}_local_shape: mean={vals.mean():.6f}, "
                f"median={np.median(vals):.6f}, std={vals.std():.6f}, "
                f"negative={n_neg}/{len(vals)} ({n_neg/len(vals)*100:.1f}%)"
            )
            logger.info(
                f"    {ch}_swap={swap[alive_mask].mean():.6f}, "
                f"{ch}_patch={patch[alive_mask].mean():.6f}"
            )

        # ── Additivity check ──
        # all_same_perm + R_local + G_local + B_local ≈ orig
        logger.info(f"\n  ── Additivity: all_same_perm + R + G + B local ≈ orig ──")
        reconstructed = tk_all_same + R_local + G_local + B_local

        agg_recon = reconstructed[alive_mask].sum()
        agg_orig = tk_orig[alive_mask].sum()
        logger.info(
            f"  Aggregate: {agg_recon:.4f} / {agg_orig:.4f} = "
            f"{agg_recon / (agg_orig + 1e-12):.4f}"
        )

        # Per-image linearity
        linearity = np.full(d_sae, np.nan)
        for n_i in range(d_sae):
            if not alive_mask[n_i]:
                continue
            idx = topk_indices[n_i]
            o = orig[metric][idx, n_i]
            a = all_same[metric][idx, n_i]
            rs = R_swap[metric][idx, n_i]
            gs = G_swap[metric][idx, n_i]
            bs = B_swap[metric][idx, n_i]
            rp = R_patch[metric][idx, n_i]
            gp = G_patch[metric][idx, n_i]
            bp = B_patch[metric][idx, n_i]
            recon = a + (rs - rp) + (gs - gp) + (bs - bp)
            valid = o > 1e-10
            if valid.sum() < 2:
                continue
            linearity[n_i] = np.mean(recon[valid] / o[valid])

        valid_lin = linearity[alive_mask]
        valid_lin = valid_lin[np.isfinite(valid_lin)]
        logger.info(f"\n  Per-image linearity: reconstructed / {metric_label}(orig)")
        logger.info(f"    N neurons: {len(valid_lin)}")
        logger.info(
            f"    mean={valid_lin.mean():.4f}, median={np.median(valid_lin):.4f}"
        )
        logger.info(
            f"    5th/95th pctl: [{np.percentile(valid_lin, 5):.4f}, "
            f"{np.percentile(valid_lin, 95):.4f}]"
        )
        in_band = ((valid_lin > 0.8) & (valid_lin < 1.2)).sum()
        logger.info(
            f"    [0.8-1.2]: {in_band}/{len(valid_lin)} "
            f"({in_band/len(valid_lin)*100:.1f}%)"
        )

        # ── Fraction of local shape per channel ──
        R_clip = np.maximum(R_local, 0)
        G_clip = np.maximum(G_local, 0)
        B_clip = np.maximum(B_local, 0)
        total_local = R_clip + G_clip + B_clip + 1e-12
        for ch, frac in [
            ("R", R_clip / total_local),
            ("G", G_clip / total_local),
            ("B", B_clip / total_local),
        ]:
            v = frac[alive_mask]
            logger.info(
                f"  {ch} local fraction: mean={v.mean():.4f}, std={v.std():.4f}"
            )

    # ── Save ──
    npz_path = os.path.join(out_dir, "local_shape_attribution_results.npz")
    np.savez_compressed(
        npz_path,
        alive_mask=alive_mask,
        usage_ema=usage_ema,
        patch_size=ps,
        n_shuffle_repeats=n_rep,
        top_k=k_actual,
        seam_margin=args.seam_margin,
    )
    logger.info(f"\nSaved: {npz_path}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
