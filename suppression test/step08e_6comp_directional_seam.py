# ==============================================================================
# 6-Component Decomposition with Direction-Specific Seam Masking
#
# Same 6 components as step08d, but seam masking is applied ONLY where
# a non-common seam artifact exists:
#
#   Interactions — seam mask on the "extra" channel's seam only:
#     GB_lr = GAP(R_lr|ud_mask) - GAP(Rlr_Gud|ud_mask)  ← G ud seam masked
#     GB_ud = GAP(R_ud|lr_mask) - GAP(Rud_Glr|lr_mask)  ← G lr seam masked
#     GB = (GB_lr + GB_ud) / 2
#     (similarly RB, RG)
#
#   Local+Texture — NO seam mask (no lr/ud seam artifacts):
#     R_local_tex = GAP(R_swap) - GAP(blur_R + R_patch_shuffle)
#     (no seam involved in either condition)
#
#   Baseline — NO seam mask
#
# Usage:
#   python -m suppression_test.step08e_6comp_directional_seam \
#       --model_state_path /path/to/best_model.pt \
#       --sae_ckpt /path/to/sae_checkpoint.pt \
#       --save_dir /path/to/MoCo_seed87 \
#       --shard_root /path/to/wds_shards_tar \
#       --samples_per_class 500 \
#       --patch_size 8 \
#       --blur_sigma 15.0 \
#       --blur_kernel_size 91 \
#       --n_shuffle_repeats 3 \
#       --seam_margin 4 \
#       --batch_size 64
# ==============================================================================

# python -m suppression_test.step08e_6comp_directional_seam \
#     --model_state_path /home/ubuntu/model-east3/outputs/MoCo_seed87/best_model.pt \
#     --sae_ckpt /home/ubuntu/model-east3/outputs/MoCo_seed87/SAE_seed123_no_L2norm_loss/stage5_out_d8192_gated_sp800.0_aux0.03125_tied_ep008.pt \
#     --save_dir /home/ubuntu/model-east3/outputs/MoCo_seed87 \
#     --shard_root /home/ubuntu/model-east3/wds_shards_tar \
#     --samples_per_class 500 \
#     --patch_size 8 \
#     --blur_sigma 15.0 \
#     --blur_kernel_size 91 \
#     --n_shuffle_repeats 3 \
#     --seam_margin 4 \
#     --batch_size 64


import argparse
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
    KNOWN_SHARD_ROOTS, build_lr_seam_mask, build_ud_seam_mask,
    get_sae_activation_maps, load_split_csv, lr_swap_channels, remap_uid,
    ud_swap_channels)
# Reuse from step08b
from suppression_test.step08b_local_shape_attribution import \
    patch_shuffle_channels
# Reuse blur helpers from step08c
from suppression_test.step08c_texture_attribution import (
    all_independent_blur_and_shuffle, blur_then_patch_shuffle_channels)

from sae_project.step02_logging_utils import SUPERCLASS_MAP, get_logger
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

logger = get_logger("6comp_directional_seam")


# ==============================================================================
# Args
# ==============================================================================
def get_args():
    p = argparse.ArgumentParser("6-Component Directional Seam")

    p.add_argument("--model_state_path", type=str, required=True)
    p.add_argument("--sae_ckpt", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="")
    p.add_argument("--shard_root", type=str, default="/content/wds_shards")
    p.add_argument("--samples_per_class", type=int, default=500)
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--blocks", type=str, default="2,2,2,3")
    p.add_argument("--dilations", type=str, default="1,1,1,1")
    p.add_argument("--refine_blocks", type=int, default=1)
    p.add_argument("--ckpt_segments", type=int, default=0)

    p.add_argument(
        "--seam_margin",
        type=int,
        default=4,
        help="Pixels to mask at non-common seam for interactions",
    )
    p.add_argument("--dead_threshold", type=float, default=5e-5)
    p.add_argument("--top_k_per_neuron", type=int, default=100)
    p.add_argument("--patch_size", type=int, default=8)

    p.add_argument("--blur_sigma", type=float, default=15.0)
    p.add_argument("--blur_kernel_size", type=int, default=91)

    p.add_argument("--de_adj_p", type=float, default=0.05)
    p.add_argument("--de_min_log2fc", type=float, default=1.5)

    p.add_argument("--which_layer", type=str, default="stage5_out")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--n_shuffle_repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


# ==============================================================================
# Main
# ==============================================================================
def main():
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    which_layer = args.which_layer

    # ── Output dir ──
    sae_parent = os.path.dirname(args.sae_ckpt)
    out_dir = os.path.join(sae_parent, "6comp_directional_seam")
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

    usage_ema = sae.usage_ema.cpu().numpy()
    alive_mask = usage_ema >= args.dead_threshold
    n_alive = int(alive_mask.sum())
    logger.info(f"  d_sae={d_sae}, alive={n_alive}")

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

    superclasses = [SUPERCLASS_MAP.get(line, "Unknown") for line in bank.lines]

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

    # ── Feature map size probe ──
    sample_x = (
        next(iter(loader))[0][:1]
        .to(device)
        .contiguous(memory_format=torch.channels_last)
    )
    with torch.amp.autocast(**autocast_kwargs):
        sample_act = get_sae_activation_maps(encoder, sae, sample_x, which_layer)
    _, _, Hf, Wf = sample_act.shape
    HW = Hf * Wf

    # ── Build direction-specific seam masks ──
    margin = args.seam_margin
    lr_mask = build_lr_seam_mask(
        Hf, Wf, margin, device
    )  # (1,1,H,W), masks vertical center
    ud_mask = build_ud_seam_mask(
        Hf, Wf, margin, device
    )  # (1,1,H,W), masks horizontal center
    no_mask = torch.ones(1, 1, Hf, Wf, device=device)  # all ones = no masking

    lr_n = float(lr_mask.sum().item())
    ud_n = float(ud_mask.sum().item())
    logger.info(f"  Feature map: {Hf}x{Wf} = {HW} pixels")
    logger.info(f"  seam_margin={margin}")
    logger.info(f"  lr_mask active: {int(lr_n)}/{HW}, ud_mask active: {int(ud_n)}/{HW}")
    logger.info(f"  no_mask active: {HW}/{HW}")

    # ── GAP for top-K selection (no mask) ──
    logger.info("Collecting GAP for top-K selection...")
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
    gap_all = np.concatenate(gap_all, axis=0)

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

    # ══════════════════════════════════════════════════════════════
    # Collection helpers
    # ══════════════════════════════════════════════════════════════
    def collect_spatial(perturb_fn, desc="", mask=None):
        """Returns dict with 'l2sq' and 'gap', each (N, d_sae).
        If mask is provided, apply it before computing metrics.
        """
        if mask is None:
            mask = no_mask
        n_pix = float(mask.sum().item())
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
            masked = act_maps * mask
            l2sq = (masked**2).sum(dim=(2, 3))
            gap = masked.sum(dim=(2, 3)) / n_pix
            l2sq_list.append(l2sq.cpu().float().numpy())
            gap_list.append(gap.cpu().float().numpy())
        return {
            "l2sq": np.concatenate(l2sq_list, axis=0),
            "gap": np.concatenate(gap_list, axis=0),
        }

    def collect_spatial_averaged(perturb_fn, desc="", n_repeats=1, mask=None):
        if n_repeats <= 1:
            return collect_spatial(perturb_fn, desc=desc, mask=mask)
        all_l2sq, all_gap = [], []
        for rep in range(n_repeats):
            data = collect_spatial(
                perturb_fn, desc=f"{desc} rep{rep+1}/{n_repeats}", mask=mask
            )
            all_l2sq.append(data["l2sq"])
            all_gap.append(data["gap"])
        return {
            "l2sq": np.mean(all_l2sq, axis=0),
            "gap": np.mean(all_gap, axis=0),
        }

    def topk_mean(arr):
        vals = np.zeros(d_sae)
        for n_i in range(d_sae):
            if not alive_mask[n_i]:
                continue
            idx = topk_indices[n_i]
            vals[n_i] = arr[idx, n_i].mean()
        return vals

    # ══════════════════════════════════════════════════════════════
    # Collect all conditions
    # ══════════════════════════════════════════════════════════════
    ps = args.patch_size
    n_rep = args.n_shuffle_repeats
    bk = args.blur_kernel_size
    bsig = args.blur_sigma

    logger.info(f"\n{'='*60}")
    logger.info(f"Collecting all conditions")
    logger.info(
        f"  patch_size={ps}, blur_sigma={bsig}, blur_kernel={bk}, repeats={n_rep}"
    )
    logger.info(f"{'='*60}")

    # 1. Original (no mask)
    logger.info("\n  ── Original (no mask) ──")
    orig = collect_spatial(None, desc="original", mask=None)

    # Update top-K
    for n_i in range(d_sae):
        if not alive_mask[n_i]:
            continue
        col = orig["l2sq"][:, n_i]
        topk_indices[n_i] = np.argsort(col)[::-1][:k_actual]

    # ──────────────────────────────────────────────────────────────
    # 2. Interaction conditions — direction-specific seam masking
    #
    # For each interaction, we pair single-channel shift with all-broken
    # and mask ONLY the non-common channel's seam:
    #
    #   GB = R_only - all_broken
    #     lr pair: R_lr vs Rlr_Gud → G has ud seam → apply ud_mask
    #     ud pair: R_ud vs Rud_Glr → G has lr seam → apply lr_mask
    #
    #   RB = G_only - all_broken
    #     lr pair: G_lr vs Rlr_Gud → R has lr seam → apply lr_mask
    #     ud pair: G_ud vs Rud_Glr → R has ud seam → apply ud_mask
    #
    #   RG = B_only - all_broken
    #     lr pair: B_lr vs Rlr_Gud → both R lr + G ud seams → combined mask
    #     ud pair: B_ud vs Rud_Glr → both R ud + G lr seams → combined mask
    # ──────────────────────────────────────────────────────────────
    logger.info("\n  ── Interaction conditions (direction-specific seam masks) ──")

    # all-broken sub-conditions (collected with each needed mask)
    # Rlr_Gud: R lr seam + G ud seam
    # Rud_Glr: R ud seam + G lr seam

    # We need each sub-condition with different masks, so collect once
    # and apply masks during metric computation.
    # Actually, since masks affect GAP/L2² computation, we need to
    # collect with the correct mask.

    # --- GB interaction: mask the "extra" G seam ---
    # lr pair: both have R lr seam (common), G has ud seam in all_broken only → ud_mask
    logger.info("  GB interaction:")
    R_lr__ud_mask = collect_spatial(
        lambda x: lr_swap_channels(x, [0]), desc="R_lr (ud_mask)", mask=ud_mask
    )
    Rlr_Gud__ud_mask = collect_spatial(
        lambda x: ud_swap_channels(lr_swap_channels(x, [0]), [1]),
        desc="Rlr_Gud (ud_mask)",
        mask=ud_mask,
    )

    # ud pair: both have R ud seam (common), G has lr seam in all_broken only → lr_mask
    R_ud__lr_mask = collect_spatial(
        lambda x: ud_swap_channels(x, [0]), desc="R_ud (lr_mask)", mask=lr_mask
    )
    Rud_Glr__lr_mask = collect_spatial(
        lambda x: lr_swap_channels(ud_swap_channels(x, [0]), [1]),
        desc="Rud_Glr (lr_mask)",
        mask=lr_mask,
    )

    # --- RB interaction: mask the "extra" R seam ---
    # lr pair: both have G lr seam (common in G_lr), R has lr seam in all_broken → lr_mask
    logger.info("  RB interaction:")
    G_lr__lr_mask = collect_spatial(
        lambda x: lr_swap_channels(x, [1]), desc="G_lr (lr_mask)", mask=lr_mask
    )
    Rlr_Gud__lr_mask = collect_spatial(
        lambda x: ud_swap_channels(lr_swap_channels(x, [0]), [1]),
        desc="Rlr_Gud (lr_mask)",
        mask=lr_mask,
    )

    # ud pair: both have G ud seam (common in G_ud), R has ud seam in all_broken → ud_mask
    G_ud__ud_mask = collect_spatial(
        lambda x: ud_swap_channels(x, [1]), desc="G_ud (ud_mask)", mask=ud_mask
    )
    Rud_Glr__ud_mask = collect_spatial(
        lambda x: lr_swap_channels(ud_swap_channels(x, [0]), [1]),
        desc="Rud_Glr (ud_mask)",
        mask=ud_mask,
    )

    # --- RG interaction: mask both R and G seams (B is not shifted, both R+G are "extra") ---
    logger.info("  RG interaction:")
    # lr pair: B_lr has no seam, Rlr_Gud has R lr + G ud → mask both (lr × ud = combined)
    combined_mask = lr_mask * ud_mask  # intersection
    combined_n = float(combined_mask.sum().item())
    logger.info(f"    combined_mask active: {int(combined_n)}/{HW}")

    B_lr__combined = collect_spatial(
        lambda x: lr_swap_channels(x, [2]),
        desc="B_lr (combined_mask)",
        mask=combined_mask,
    )
    Rlr_Gud__combined = collect_spatial(
        lambda x: ud_swap_channels(lr_swap_channels(x, [0]), [1]),
        desc="Rlr_Gud (combined_mask)",
        mask=combined_mask,
    )

    # ud pair: B_ud has no seam, Rud_Glr has R ud + G lr → mask both
    B_ud__combined = collect_spatial(
        lambda x: ud_swap_channels(x, [2]),
        desc="B_ud (combined_mask)",
        mask=combined_mask,
    )
    Rud_Glr__combined = collect_spatial(
        lambda x: lr_swap_channels(ud_swap_channels(x, [0]), [1]),
        desc="Rud_Glr (combined_mask)",
        mask=combined_mask,
    )

    # ──────────────────────────────────────────────────────────────
    # 3. Local+Texture (no mask)
    # ──────────────────────────────────────────────────────────────
    logger.info("\n  ── Single-channel shifts (no mask, for local+tex) ──")
    R_only_lr_nm = collect_spatial(
        lambda x: lr_swap_channels(x, [0]), desc="R_only_lr (no mask)", mask=None
    )
    R_only_ud_nm = collect_spatial(
        lambda x: ud_swap_channels(x, [0]), desc="R_only_ud (no mask)", mask=None
    )
    G_only_lr_nm = collect_spatial(
        lambda x: lr_swap_channels(x, [1]), desc="G_only_lr (no mask)", mask=None
    )
    G_only_ud_nm = collect_spatial(
        lambda x: ud_swap_channels(x, [1]), desc="G_only_ud (no mask)", mask=None
    )
    B_only_lr_nm = collect_spatial(
        lambda x: lr_swap_channels(x, [2]), desc="B_only_lr (no mask)", mask=None
    )
    B_only_ud_nm = collect_spatial(
        lambda x: ud_swap_channels(x, [2]), desc="B_only_ud (no mask)", mask=None
    )

    R_swap = {k: (R_only_lr_nm[k] + R_only_ud_nm[k]) / 2 for k in ["l2sq", "gap"]}
    G_swap = {k: (G_only_lr_nm[k] + G_only_ud_nm[k]) / 2 for k in ["l2sq", "gap"]}
    B_swap = {k: (B_only_lr_nm[k] + B_only_ud_nm[k]) / 2 for k in ["l2sq", "gap"]}

    logger.info(f"\n  ── Per-channel blur(σ={bsig})→shuffle (no mask) ──")
    R_blur_patch = collect_spatial_averaged(
        lambda x: blur_then_patch_shuffle_channels(x, [0], ps, bk, bsig),
        desc="R_blur_patch",
        n_repeats=n_rep,
        mask=None,
    )
    G_blur_patch = collect_spatial_averaged(
        lambda x: blur_then_patch_shuffle_channels(x, [1], ps, bk, bsig),
        desc="G_blur_patch",
        n_repeats=n_rep,
        mask=None,
    )
    B_blur_patch = collect_spatial_averaged(
        lambda x: blur_then_patch_shuffle_channels(x, [2], ps, bk, bsig),
        desc="B_blur_patch",
        n_repeats=n_rep,
        mask=None,
    )

    # ──────────────────────────────────────────────────────────────
    # 4. Baseline (no mask)
    # ──────────────────────────────────────────────────────────────
    logger.info(f"\n  ── Baseline (no mask) ──")
    baseline = collect_spatial_averaged(
        lambda x: all_independent_blur_and_shuffle(x, ps, bk, bsig),
        desc="baseline",
        n_repeats=n_rep,
        mask=None,
    )

    # ══════════════════════════════════════════════════════════════
    # Compute & report 6-component decomposition
    # ══════════════════════════════════════════════════════════════
    for metric in ["l2sq", "gap"]:
        metric_label = "L2²" if metric == "l2sq" else "GAP"

        logger.info(f"\n{'='*60}")
        logger.info(f"  ══ {metric_label}-based 6-Component (Directional Seam) ══")
        logger.info(f"{'='*60}")

        tk_orig = topk_mean(orig[metric])
        tk_baseline = topk_mean(baseline[metric])
        tk_R_swap = topk_mean(R_swap[metric])
        tk_G_swap = topk_mean(G_swap[metric])
        tk_B_swap = topk_mean(B_swap[metric])
        tk_R_bp = topk_mean(R_blur_patch[metric])
        tk_G_bp = topk_mean(G_blur_patch[metric])
        tk_B_bp = topk_mean(B_blur_patch[metric])

        # ── Interactions (direction-specific seam masked) ──
        # GB = avg(lr_pair, ud_pair)
        GB_lr = topk_mean(R_lr__ud_mask[metric]) - topk_mean(Rlr_Gud__ud_mask[metric])
        GB_ud = topk_mean(R_ud__lr_mask[metric]) - topk_mean(Rud_Glr__lr_mask[metric])
        GB_inter = (GB_lr + GB_ud) / 2

        # RB = avg(lr_pair, ud_pair)
        RB_lr = topk_mean(G_lr__lr_mask[metric]) - topk_mean(Rlr_Gud__lr_mask[metric])
        RB_ud = topk_mean(G_ud__ud_mask[metric]) - topk_mean(Rud_Glr__ud_mask[metric])
        RB_inter = (RB_lr + RB_ud) / 2

        # RG = avg(lr_pair, ud_pair) with combined mask
        RG_lr = topk_mean(B_lr__combined[metric]) - topk_mean(Rlr_Gud__combined[metric])
        RG_ud = topk_mean(B_ud__combined[metric]) - topk_mean(Rud_Glr__combined[metric])
        RG_inter = (RG_lr + RG_ud) / 2

        # ── Local+Texture (no mask) ──
        R_local_tex = tk_R_swap - tk_R_bp
        G_local_tex = tk_G_swap - tk_G_bp
        B_local_tex = tk_B_swap - tk_B_bp

        # ── Report ──
        logger.info(f"\n  {metric_label}(orig) mean: {tk_orig[alive_mask].mean():.6f}")
        logger.info(
            f"  {metric_label}(baseline) mean: {tk_baseline[alive_mask].mean():.6f}"
        )

        logger.info(
            f"\n  ── Pairwise Channel Interactions (direction-specific seam) ──"
        )
        for pair, contrib in [("RG", RG_inter), ("RB", RB_inter), ("GB", GB_inter)]:
            vals = contrib[alive_mask]
            n_neg = int((vals < 0).sum())
            logger.info(
                f"  {pair}: mean={vals.mean():.6f}, "
                f"median={np.median(vals):.6f}, std={vals.std():.6f}, "
                f"negative={n_neg}/{len(vals)} ({n_neg/len(vals)*100:.1f}%)"
            )

        logger.info(f"\n  ── Per-Channel Local+Texture (no mask) ──")
        for ch, lt, swap, bp in [
            ("R", R_local_tex, tk_R_swap, tk_R_bp),
            ("G", G_local_tex, tk_G_swap, tk_G_bp),
            ("B", B_local_tex, tk_B_swap, tk_B_bp),
        ]:
            vals = lt[alive_mask]
            n_neg = int((vals < 0).sum())
            logger.info(
                f"  {ch}_local_tex: mean={vals.mean():.6f}, "
                f"median={np.median(vals):.6f}, std={vals.std():.6f}, "
                f"negative={n_neg}/{len(vals)} ({n_neg/len(vals)*100:.1f}%)"
            )
            logger.info(
                f"    {ch}_swap={swap[alive_mask].mean():.6f}, "
                f"{ch}_blur_patch={bp[alive_mask].mean():.6f}"
            )

        # ── Fractional contributions ──
        logger.info(f"\n  ── Fractional Contributions (raw, no clip) ──")
        total_spatial = tk_orig - tk_baseline
        components = {
            "RG_inter": RG_inter,
            "RB_inter": RB_inter,
            "GB_inter": GB_inter,
            "R_local_tex": R_local_tex,
            "G_local_tex": G_local_tex,
            "B_local_tex": B_local_tex,
        }
        for name, vals in components.items():
            frac = vals / (total_spatial + 1e-12)
            v = frac[alive_mask]
            logger.info(f"  {name:14s}: mean={v.mean():.4f}, std={v.std():.4f}")

        inter_frac = (RG_inter + RB_inter + GB_inter) / (total_spatial + 1e-12)
        lt_frac = (R_local_tex + G_local_tex + B_local_tex) / (total_spatial + 1e-12)
        logger.info(
            f"  {'[Interactions]':14s}: mean={inter_frac[alive_mask].mean():.4f}"
        )
        logger.info(f"  {'[Local+Tex]':14s}: mean={lt_frac[alive_mask].mean():.4f}")

        # ── 6-Component Additivity Check ──
        logger.info(f"\n  ── 6-Component Additivity Check ──")
        logger.info(f"  baseline + Σ(6 components) ≈ orig")
        logger.info(
            f"  NOTE: interactions use seam-masked GAP, local+tex use full GAP."
        )
        logger.info(f"        Additivity is approximate due to different pixel sets.\n")

        sum_6 = RG_inter + RB_inter + GB_inter + R_local_tex + G_local_tex + B_local_tex
        reconstructed = tk_baseline + sum_6

        agg_recon = reconstructed[alive_mask].sum()
        agg_orig = tk_orig[alive_mask].sum()
        logger.info(
            f"  Aggregate: {agg_recon:.4f} / {agg_orig:.4f} = "
            f"{agg_recon / (agg_orig + 1e-12):.4f}"
        )

        # Per-neuron linearity
        # NOTE: This uses per-image data. For interactions, we need the
        # seam-masked per-image values. We approximate by using the
        # direction-specific masked collections.
        linearity = np.full(d_sae, np.nan)
        for n_i in range(d_sae):
            if not alive_mask[n_i]:
                continue
            idx = topk_indices[n_i]
            o = orig[metric][idx, n_i]

            bl = baseline[metric][idx, n_i]

            # Interactions (seam-masked, averaged over directions)
            gb_lr_i = (
                R_lr__ud_mask[metric][idx, n_i] - Rlr_Gud__ud_mask[metric][idx, n_i]
            )
            gb_ud_i = (
                R_ud__lr_mask[metric][idx, n_i] - Rud_Glr__lr_mask[metric][idx, n_i]
            )
            gb_i = (gb_lr_i + gb_ud_i) / 2

            rb_lr_i = (
                G_lr__lr_mask[metric][idx, n_i] - Rlr_Gud__lr_mask[metric][idx, n_i]
            )
            rb_ud_i = (
                G_ud__ud_mask[metric][idx, n_i] - Rud_Glr__ud_mask[metric][idx, n_i]
            )
            rb_i = (rb_lr_i + rb_ud_i) / 2

            rg_lr_i = (
                B_lr__combined[metric][idx, n_i] - Rlr_Gud__combined[metric][idx, n_i]
            )
            rg_ud_i = (
                B_ud__combined[metric][idx, n_i] - Rud_Glr__combined[metric][idx, n_i]
            )
            rg_i = (rg_lr_i + rg_ud_i) / 2

            # Local+Tex (no mask)
            r_lt = R_swap[metric][idx, n_i] - R_blur_patch[metric][idx, n_i]
            g_lt = G_swap[metric][idx, n_i] - G_blur_patch[metric][idx, n_i]
            b_lt = B_swap[metric][idx, n_i] - B_blur_patch[metric][idx, n_i]

            recon = bl + rg_i + rb_i + gb_i + r_lt + g_lt + b_lt

            valid = o > 1e-10
            if valid.sum() < 2:
                continue
            linearity[n_i] = np.median(recon[valid] / o[valid])

        valid_lin = linearity[alive_mask]
        valid_lin = valid_lin[np.isfinite(valid_lin)]
        logger.info(f"  Per-neuron linearity (median of per-image ratio):")
        logger.info(f"    N neurons: {len(valid_lin)}")
        logger.info(
            f"    mean={valid_lin.mean():.4f}, median={np.median(valid_lin):.4f}"
        )
        logger.info(
            f"    5th/95th pctl: [{np.percentile(valid_lin, 5):.4f}, "
            f"{np.percentile(valid_lin, 95):.4f}]"
        )
        for blo, bhi, label in [
            (0.8, 1.2, "0.8-1.2"),
            (0.5, 1.5, "0.5-1.5"),
            (0.9, 1.1, "0.9-1.1"),
        ]:
            in_b = ((valid_lin > blo) & (valid_lin < bhi)).sum()
            logger.info(
                f"    [{label}]: {in_b}/{len(valid_lin)} "
                f"({in_b/len(valid_lin)*100:.1f}%)"
            )

    # ── Save ──
    npz_path = os.path.join(out_dir, "6comp_directional_seam_results.npz")
    np.savez_compressed(
        npz_path,
        alive_mask=alive_mask,
        usage_ema=usage_ema,
        patch_size=ps,
        blur_sigma=bsig,
        blur_kernel_size=bk,
        n_shuffle_repeats=n_rep,
        top_k=k_actual,
        seam_margin=margin,
    )
    logger.info(f"\nSaved: {npz_path}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
