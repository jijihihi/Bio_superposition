# ==============================================================================
# Full 9-Component Spatial Information Decomposition
#
# Decomposes spatial information in SAE neuron activations into 9 components:
#
# ── Direction averaging strategy ──
#
#   lr_swap = left-right half swap,  ud_swap = up-down half swap
#   For each "shift" condition, we average lr and ud to cancel direction bias.
#   For patch shuffle/blur conditions, direction is irrelevant (random permutation).
#
# ── Pairwise Channel Interactions ──
#
#   all_broken = avg over 3 channel pairs × 2 directions = 6 combos:  # all_broken: 공간적 정보 다 날려버린 이미지의 feature map.
#     R+G: Rlr_Gud, Rud_Glr  (R G 에서 반반씩 잘라서 공간적 정보 다 없앤다.)
#     R+B: Rlr_Bud, Rud_Blr
#     G+B: Glr_Bud, Gud_Blr
#     → All 3 pairwise interactions are broken in every combo (2 channels shifted).
#       Averaging over all 3 pairs makes the baseline symmetric across channels.
#
#   GB = avg(R_lr, R_ud) - all_broken       ← R shift preserves GB  # R만 lr 하거나 ud 하면 GB 정보는 남는다. 그래서 이 피처맵에서, 위에서 구한대로 공간적 정보 다 없앤 GAP 값 빼면 GB 정보의 GAP 기여도가 남는다.
#   RB = avg(G_lr, G_ud) - all_broken       ← G shift preserves RB
#   RG = avg(B_lr, B_ud) - all_broken       ← B shift preserves RG
#     → single-channel shifts: 2 directions (lr, ud) averaged
#
# ── Per-Channel Local Shape ──
#
#   R_local = avg(R_lr, R_ud) - R_patch_shuffle  # 왼쪽은 RG RB의 공간적 정보만 파괴된다. 오른쪽은 RG RB의 공간적 정보와 R의 local shape이 파괴된다. 이거 두개 빼면, R local shape이 GAP에 기여하는 값 알 수 있다.
#   G_local = avg(G_lr, G_ud) - G_patch_shuffle
#   B_local = avg(B_lr, B_ud) - B_patch_shuffle
#     → swap term: 2 directions (lr, ud) averaged  (same arrays as interaction)
#     → patch shuffle: direction-irrelevant (random permutation, n_rep averaged)
#
# ── Per-Channel Texture ──
#
#   R_tex = R_patch_shuffle - blur_then_shuffle(R) # 왼쪽은 R의 local shape과 RB RG의 공간적 정보 파괴된다. 이때 blurring을 먼저한 후에 patch shuffling한다. 오른쪽은 여기다가 R의 texture도 파괴된다.
#   G_tex = G_patch_shuffle - blur_then_shuffle(G)
#   B_tex = B_patch_shuffle - blur_then_shuffle(B)
#     → Both terms are direction-irrelevant (random patch permutation)
#     → Only within-patch texture differs (blur destroys it)
#     → Blur BEFORE shuffle ensures identical patch boundary artifacts.
#
# ── Baseline ──  # R의 1번 패치가 9번으로 갔는데 우연히 B의 패치가 9번으로 간다면 이 둘의 공간적 정보가 파괴되지 않고 유지될 가능성이 있다.
#                 # 따라서 어떤 패치도 공간적으로 겹치지 않도록 강제했다.
#   baseline = all_independent_blur_and_shuffle(R,G,B) # R G B texture 다 나누고 local shape도 다 섞는다.
#     → Direction-irrelevant. Destroys all interactions + local shape + texture.
#
# ── Additivity ──
#
#   baseline + Σ(9 components) ≈ orig
#
# Usage:
#   python -m suppression_test.step08c_texture_attribution \
#       --model_state_path /path/to/best_model.pt \
#       --sae_ckpt /path/to/sae_checkpoint.pt \
#       --save_dir /path/to/MoCo_seed87 \
#       --shard_root /path/to/wds_shards_tar \
#       --samples_per_class 500 \
#       --patch_size 8 \
#       --blur_sigma 3.0 \
#       --blur_kernel_size 13 \
#       --n_shuffle_repeats 3
# ==============================================================================




# python -m suppression_test.step08c_texture_attribution \
#     --model_state_path /home/ubuntu/model-east3/outputs/MoCo_seed87/best_model.pt \
#     --sae_ckpt /home/ubuntu/model-east3/outputs/MoCo_seed87/SAE_seed123_no_L2norm_loss/stage5_out_d8192_gated_sp800.0_aux0.03125_tied_ep008.pt \
#     --save_dir /home/ubuntu/model-east3/outputs/MoCo_seed87 \
#     --shard_root /home/ubuntu/model-east3/wds_shards_tar \
#     --samples_per_class 500 \
#     --patch_size 8 \
#     --blur_sigma 5.0 \
#     --blur_kernel_size 31 \
#     --n_shuffle_repeats 3 \
#     --seam_margin 0

# python -m suppression_test.step08c_texture_attribution \
#     --model_state_path /home/ubuntu/model-east3/outputs/MoCo_seed87/best_model.pt \
#     --sae_ckpt /home/ubuntu/model-east3/outputs/MoCo_seed87/SAE_seed123_no_L2norm_loss/stage5_out_d8192_gated_sp800.0_aux0.03125_tied_ep008.pt \
#     --save_dir /home/ubuntu/model-east3/outputs/MoCo_seed87 \
#     --shard_root /home/ubuntu/model-east3/wds_shards_tar \
#     --samples_per_class 1000 \
#     --patch_size 8 \
#     --blur_sigma 4.0 \
#     --blur_kernel_size 25 \
#     --n_shuffle_repeats 3 \
#     --seam_margin 0

## ## 패치 하나의 사이즈. patch_size

###### MoCo_seed87 안의 "SAE"랑 "SAE_sparsity3200_loss_L2norm곱해줌" 랑 동일한 파일.

# python -m suppression_test.step08c_texture_attribution \
#     --model_state_path /home/ubuntu/model-east3/outputs/MoCo_seed87/best_model.pt \
#     --concept_ids "0018,0037,0046,0048,0049,0056,0080,0094,0101,0117,0134,0203,0237,0264,0277,0300,0301,0306,0326,0331,0337,0340,0358,0381,0405,0432,0441,0474,0484,0491,0522,0539,0548,0557,0576,0581,0621,0679,0700,0738,0776,0783,0788,0807,0812,0814,0824,0849,0857,0885,0909,0914,0916,0932,0936,0952,0968,0985,1048,1049,1070,1086,1090,1093,1105,1127,1146,1181,1196,1203,1211,1219,1226,1240,1265,1281,1282,1286,1301,1304,1322,1326,1332,1338,1339,1341,1366,1372,1380,1398,1408,1416,1438,1447,1450,1461,1464,1470,1539,1555,1558,1560,1566,1578,1581,1592,1632,1647,1648,1659,1667,1673,1685,1688,1699,1710,1737,1766,1778,1783,1803,1815,1819,1850,1879,1880,1885,1932,1945,1953,1962,1975,1978,1995,2001,2003,2019,2048,2052,2055,2057,2122,2147,2156,2170,2173,2182,2183,2198,2221,2246,2248,2290,2291,2314,2315,2338,2363,2375,2389,2421,2424,2468,2490,2502,2530,2541,2567,2587,2595,2598,2616,2683,2716,2748,2773,2785,2796,2798,2800,2805,2814,2835,2837,2851,2864,2872,2873,2903,2910,2920,2924,2926,2943,2954,2956,2959,2967,2968,3001,3012,3026,3035,3044,3079,3128,3129,3133,3135,3154,3181,3186,3221,3245,3249,3259,3272,3301,3304,3315,3323,3336,3338,3366,3370,3403,3439,3462,3466,3481,3497,3533,3538,3542,3557,3585,3592,3625,3627,3635,3637,3656,3683,3685,3701,3713,3720,3736,3742,3760,3762,3779,3788,3789,3794,3820,3845,3847,3900,3922,3933,3934,3977,3979,4014,4060,4071,4080,4092" \
#     --sae_ckpt /home/ubuntu/model-east3/outputs/MoCo_seed87/SAE/stage5_out_d4096_gated_sp3200.0_aux0.03125_tied_ep008.pt \
#     --save_dir /home/ubuntu/model-east3/outputs/MoCo_seed87/SAE \
#     --shard_root /home/ubuntu/model-east3/wds_shards_tar \
#     --samples_per_class 1000 \
#     --patch_size 8 \
#     --blur_sigma 4.0 \
#     --blur_kernel_size 25 \
#     --n_shuffle_repeats 3 \
#     --seam_margin 0

# python -m suppression_test.step08c_texture_attribution \
#     --model_state_path /home/ubuntu/model-east3/outputs/MoCo_seed87/best_model.pt \
#     --concept_ids "0018,0037,0046,0048,0049,0056,0080,0094,0101,0117,0134,0203,0237,0264,0277,0300,0301,0306,0326,0331,0337,0340,0358,0381,0405,0432,0441,0474,0484,0491,0522,0539,0548,0557,0576,0581,0621,0679,0700,0738,0776,0783,0788,0807,0812,0814,0824,0849,0857,0885,0909,0914,0916,0932,0936,0952,0968,0985,1048,1049,1070,1086,1090,1093,1105,1127,1146,1181,1196,1203,1211,1219,1226,1240,1265,1281,1282,1286,1301,1304,1322,1326,1332,1338,1339,1341,1366,1372,1380,1398,1408,1416,1438,1447,1450,1461,1464,1470,1539,1555,1558,1560,1566,1578,1581,1592,1632,1647,1648,1659,1667,1673,1685,1688,1699,1710,1737,1766,1778,1783,1803,1815,1819,1850,1879,1880,1885,1932,1945,1953,1962,1975,1978,1995,2001,2003,2019,2048,2052,2055,2057,2122,2147,2156,2170,2173,2182,2183,2198,2221,2246,2248,2290,2291,2314,2315,2338,2363,2375,2389,2421,2424,2468,2490,2502,2530,2541,2567,2587,2595,2598,2616,2683,2716,2748,2773,2785,2796,2798,2800,2805,2814,2835,2837,2851,2864,2872,2873,2903,2910,2920,2924,2926,2943,2954,2956,2959,2967,2968,3001,3012,3026,3035,3044,3079,3128,3129,3133,3135,3154,3181,3186,3221,3245,3249,3259,3272,3301,3304,3315,3323,3336,3338,3366,3370,3403,3439,3462,3466,3481,3497,3533,3538,3542,3557,3585,3592,3625,3627,3635,3637,3656,3683,3685,3701,3713,3720,3736,3742,3760,3762,3779,3788,3789,3794,3820,3845,3847,3900,3922,3933,3934,3977,3979,4014,4060,4071,4080,4092" \
#     --sae_ckpt /home/ubuntu/model-east3/outputs/MoCo_seed87/SAE/stage5_out_d4096_gated_sp3200.0_aux0.03125_tied_ep008.pt \
#     --save_dir /home/ubuntu/model-east3/outputs/MoCo_seed87/SAE \
#     --shard_root /home/ubuntu/model-east3/wds_shards_tar \
#     --samples_per_class 1000 \
#     --patch_size 4 \
#     --blur_sigma 2.0 \
#     --blur_kernel_size 13 \
#     --n_shuffle_repeats 3 \
#     --seam_margin 0



# 패치 사이즈는 --patch_size 8 이 기본이다. 16*16으로 나눠준다.


# Sattolo's algorithm
# perm = list(range(n_patches))
# for i in range(n_patches - 1, 0, -1):
#     j = torch.randint(0, i, (1,), device=x.device).item()  # j ∈ [0, i-1]
#     perm[i], perm[j] = perm[j], perm[i]
# 수정함. 자기 자신 제외하게


import os, sys, csv, random, argparse
from typing import List
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

from sae_project.step02_logging_utils import get_logger, OUT_DIM, SUPERCLASS_MAP
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

# Reuse from step08
from suppression_test.step08_channel_attribution import (
    lr_swap_channels, ud_swap_channels,
    build_combined_seam_mask,
    get_sae_activation_maps,
    load_split_csv, remap_uid, KNOWN_SHARD_ROOTS,
    plot_histogram,
)

# Reuse from step08b
from suppression_test.step08b_local_shape_attribution import (
    patch_shuffle_channels,
    patch_rotate_channels,
)

logger = get_logger("texture_attribution")


# ==============================================================================
# Gaussian blur helpers
# ==============================================================================
def make_gaussian_kernel(kernel_size: int, sigma: float,
                         device: torch.device) -> torch.Tensor:
    """Create a 2D Gaussian kernel (kernel_size x kernel_size)."""
    coords = torch.arange(kernel_size, dtype=torch.float32,
                           device=device) - (kernel_size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = g[:, None] * g[None, :]
    kernel = kernel / kernel.sum()
    return kernel


def gaussian_blur_channels(
    x: torch.Tensor, channels: List[int],
    kernel_size: int, sigma: float,
) -> torch.Tensor:
    """Apply Gaussian blur to specified channels. x: (B, C, H, W)."""
    device = x.device
    kernel = make_gaussian_kernel(kernel_size, sigma, device)
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, k, k)
    padding = kernel_size // 2
    out = x.clone()
    for ch in channels:
        ch_data = x[:, ch:ch+1]  # (B, 1, H, W)
        out[:, ch:ch+1] = F.conv2d(ch_data, kernel, padding=padding)
    return out


# ==============================================================================
# Combined perturbation helpers
# ==============================================================================
def blur_then_patch_shuffle_channels(
    x: torch.Tensor, channels: List[int],
    patch_size: int, blur_kernel: int, blur_sigma: float,
) -> torch.Tensor:
    """Gaussian-blur then patch-shuffle specified channels.
    Blur first so both shuffle-only and blur+shuffle conditions have
    identical patch boundary artifacts. Only within-patch texture differs."""
    x = gaussian_blur_channels(x, channels, blur_kernel, blur_sigma)
    x = patch_shuffle_channels(x, channels, patch_size)
    return x


def all_independent_blur_and_shuffle(
    x: torch.Tensor, patch_size: int,
    blur_kernel: int, blur_sigma: float,
) -> torch.Tensor:
    """Blur all channels, then independently shuffle each channel's patches.
    Destroys: all interactions, all local shapes, all textures.
    Preserves: only mean color / DC component."""
    x = gaussian_blur_channels(x, [0, 1, 2], blur_kernel, blur_sigma)
    x = patch_shuffle_channels(x, [0, 1, 2], patch_size)
    return x


# ==============================================================================
# Args
# ==============================================================================
def get_args():
    p = argparse.ArgumentParser("Full 9-Component Spatial Decomposition")

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
    p.add_argument("--patch_size", type=int, default=8,
                   help="Patch size for shuffle (img_size must be divisible)")

    # Gaussian blur parameters for texture removal
    p.add_argument("--blur_sigma", type=float, default=3.0,
                   help="Gaussian blur sigma (higher = more texture removed)")
    p.add_argument("--blur_kernel_size", type=int, default=13,
                   help="Gaussian blur kernel size (must be odd)")

    # DE-based neuron filtering
    p.add_argument("--de_adj_p", type=float, default=0.05)
    p.add_argument("--de_min_log2fc", type=float, default=1.5)
    p.add_argument("--de_top_k", type=int, default=0)

    # External concept ID selection (skip DE filter)
    p.add_argument("--concept_ids", type=str, default="",
                   help="Comma-separated concept IDs OR path to a .txt/.csv file "
                        "containing comma-separated IDs. When set, DE filter is skipped.")

    p.add_argument("--which_layer", type=str, default="stage5_out")
    p.add_argument("--dpi", type=int, default=200)

    # Number of shuffle repeats for noise reduction
    p.add_argument("--n_shuffle_repeats", type=int, default=3,
                   help="Average over N random shuffles for stable estimates")

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
    out_dir = os.path.join(sae_parent, "texture_attribution")
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Output dir: {out_dir}")

    # ── Model + SAE ──
    logger.info("Loading model + SAE...")
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

    superclasses = [SUPERCLASS_MAP.get(line, "Unknown") for line in bank.lines]

    from collections import Counter
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
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True,
                        worker_init_fn=seed_worker, collate_fn=collate_skip_none)

    # ── Seam mask + feature map size ──
    sample_x = next(iter(loader))[0][:1].to(device).contiguous(
        memory_format=torch.channels_last)
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
            memory_format=torch.channels_last)
        with torch.amp.autocast(**autocast_kwargs):
            act_maps = get_sae_activation_maps(encoder, sae, x_dev, which_layer)
        gap_batch = act_maps.mean(dim=(2, 3))
        gap_all.append(gap_batch.cpu().float().numpy())
    gap_all = np.concatenate(gap_all, axis=0)  # (N, d_sae)

    # ── Top-K indices ──
    k = args.top_k_per_neuron
    k_actual = min(k, gap_all.shape[0])
    topk_indices = np.zeros((d_sae, k_actual), dtype=int)
    for n_i in range(d_sae):
        if not alive_mask[n_i]:
            continue
        col = gap_all[:, n_i]
        topk_indices[n_i] = np.argsort(col)[::-1][:k_actual]

    # ── Neuron selection ──
    if args.concept_ids:
        # Parse concept IDs from argument (file path or comma-separated)
        cid_str = args.concept_ids.strip()
        if os.path.isfile(cid_str):
            logger.info(f"Loading concept IDs from file: {cid_str}")
            with open(cid_str, "r") as f:
                cid_str = f.read().strip()
        selected_ids = [int(x.strip()) for x in cid_str.split(",") if x.strip()]
        alive_mask = np.zeros(d_sae, dtype=bool)
        for cid in selected_ids:
            if 0 <= cid < d_sae:
                alive_mask[cid] = True
        n_alive = int(alive_mask.sum())
        logger.info(f"\n{'='*60}")
        logger.info(f"Using {n_alive} externally specified concept IDs (DE filter skipped)")
        logger.info(f"{'='*60}")
    else:
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
            ("SNCA",   superclasses), ("GBA", superclasses), ("LRRK2", superclasses),
        ]
        for target_name, sc_list in comparisons:
            sc_check = np.array(sc_list)
            n_target = int((sc_check == target_name).sum()) if target_name != "AllMut" else int((sc_check == "AllMut").sum())
            if n_target < 5:
                continue
            de_result = compute_de_neurons(gap_alive, sc_list, target_name,
                                            adj_p_threshold=args.de_adj_p,
                                            min_log2fc=args.de_min_log2fc)
            mask_alive = de_result["mask"]
            mask_full = np.zeros(d_sae, dtype=bool)
            mask_full[alive_indices[mask_alive]] = True
            de_mask_full |= mask_full

        alive_mask = alive_mask & de_mask_full
        n_alive = int(alive_mask.sum())
        logger.info(f"  Final analysis neurons: {n_alive}")

    # ══════════════════════════════════════════════════════════════
    # Collection helper
    # ══════════════════════════════════════════════════════════════
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
                memory_format=torch.channels_last)
            if perturb_fn is not None:
                x_dev = perturb_fn(x_dev).contiguous(
                    memory_format=torch.channels_last)
            with torch.amp.autocast(**autocast_kwargs):
                act_maps = get_sae_activation_maps(
                    encoder, sae, x_dev, which_layer)
            masked = act_maps * seam_mask
            l2sq = (masked ** 2).sum(dim=(2, 3))
            gap = masked.sum(dim=(2, 3)) / n_active_pixels
            l2sq_list.append(l2sq.cpu().float().numpy())
            gap_list.append(gap.cpu().float().numpy())
        return {
            "l2sq": np.concatenate(l2sq_list, axis=0),
            "gap":  np.concatenate(gap_list, axis=0),
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
            "gap":  np.mean(all_gap, axis=0),
        }

    # Per-neuron top-K mean helper
    def topk_mean(arr):
        vals = np.zeros(d_sae)
        for n_i in range(d_sae):
            if not alive_mask[n_i]:
                continue
            idx = topk_indices[n_i]
            vals[n_i] = arr[idx, n_i].mean()
        return vals

    # ══════════════════════════════════════════════════════════════
    # Phase 1: Collect all conditions
    # ══════════════════════════════════════════════════════════════
    ps = args.patch_size
    n_rep = args.n_shuffle_repeats
    bk = args.blur_kernel_size
    bs = args.blur_sigma

    logger.info(f"\n{'='*60}")
    logger.info(f"Collecting all conditions")
    logger.info(f"  patch_size={ps}, blur_sigma={bs}, blur_kernel={bk}, repeats={n_rep}")
    logger.info(f"{'='*60}")

    # ── 1. Original ──
    logger.info("\n  ── Original ──")
    orig = collect_spatial(None, desc="original")

    # Update top-K with proper L2² orig
    for n_i in range(d_sae):
        if not alive_mask[n_i]:
            continue
        col = orig["l2sq"][:, n_i]
        topk_indices[n_i] = np.argsort(col)[::-1][:k_actual]

    # ── 2. All-broken (for interaction baseline) ──
    # Average 3 channel pairs × 2 directions = 6 combos
    logger.info("\n  ── All-broken (interaction baseline, 6 combos) ──")
    # R+G pair
    ab_rg1 = collect_spatial(
        lambda x: ud_swap_channels(lr_swap_channels(x, [0]), [1]),
        desc="Rlr_Gud")
    ab_rg2 = collect_spatial(
        lambda x: lr_swap_channels(ud_swap_channels(x, [0]), [1]),
        desc="Rud_Glr")
    # R+B pair
    ab_rb1 = collect_spatial(
        lambda x: ud_swap_channels(lr_swap_channels(x, [0]), [2]),
        desc="Rlr_Bud")
    ab_rb2 = collect_spatial(
        lambda x: lr_swap_channels(ud_swap_channels(x, [0]), [2]),
        desc="Rud_Blr")
    # G+B pair
    ab_gb1 = collect_spatial(
        lambda x: ud_swap_channels(lr_swap_channels(x, [1]), [2]),
        desc="Glr_Bud")
    ab_gb2 = collect_spatial(
        lambda x: lr_swap_channels(ud_swap_channels(x, [1]), [2]),
        desc="Gud_Blr")
    all_broken = {k: (ab_rg1[k] + ab_rg2[k] + ab_rb1[k] + ab_rb2[k] +
                      ab_gb1[k] + ab_gb2[k]) / 6 for k in ["l2sq", "gap"]}
    logger.info(f"  all_broken averaged over 6 combos")

    # ── 3. Single-channel shifts (for interaction attribution) ──
    logger.info("\n  ── Single-channel shifts ──")
    R_only_lr = collect_spatial(lambda x: lr_swap_channels(x, [0]), desc="R_only_lr")
    R_only_ud = collect_spatial(lambda x: ud_swap_channels(x, [0]), desc="R_only_ud")
    G_only_lr = collect_spatial(lambda x: lr_swap_channels(x, [1]), desc="G_only_lr")
    G_only_ud = collect_spatial(lambda x: ud_swap_channels(x, [1]), desc="G_only_ud")
    B_only_lr = collect_spatial(lambda x: lr_swap_channels(x, [2]), desc="B_only_lr")
    B_only_ud = collect_spatial(lambda x: ud_swap_channels(x, [2]), desc="B_only_ud")

    R_only = {k: (R_only_lr[k] + R_only_ud[k]) / 2 for k in ["l2sq", "gap"]}
    G_only = {k: (G_only_lr[k] + G_only_ud[k]) / 2 for k in ["l2sq", "gap"]}
    B_only = {k: (B_only_lr[k] + B_only_ud[k]) / 2 for k in ["l2sq", "gap"]}

    # ── 4. Per-channel swap lr+ud (for local shape) ──
    logger.info("\n  ── Per-channel swap (local shape baseline) ──")
    R_swap = {k: (R_only_lr[k] + R_only_ud[k]) / 2 for k in ["l2sq", "gap"]}
    G_swap = {k: (G_only_lr[k] + G_only_ud[k]) / 2 for k in ["l2sq", "gap"]}
    B_swap = {k: (B_only_lr[k] + B_only_ud[k]) / 2 for k in ["l2sq", "gap"]}
    # Note: R_swap == R_only since shifting only R is the same as R_only.
    # Both break RG/RB interactions while preserving R local shape + texture.

    # ── 5. Per-channel patch shuffle (shared by local shape & texture) ──
    logger.info("\n  ── Per-channel patch shuffle ──")
    R_patch = collect_spatial_averaged(
        lambda x: patch_shuffle_channels(x, [0], ps),
        desc="R_patch", n_repeats=n_rep)
    G_patch = collect_spatial_averaged(
        lambda x: patch_shuffle_channels(x, [1], ps),
        desc="G_patch", n_repeats=n_rep)
    B_patch = collect_spatial_averaged(
        lambda x: patch_shuffle_channels(x, [2], ps),
        desc="B_patch", n_repeats=n_rep)

    # ── 5b. Per-channel patch ROTATION (reference frame decomposition) ──
    logger.info("\n  ── Per-channel patch rotation (density preserved, shape destroyed) ──")
    R_rotated = collect_spatial_averaged(
        lambda x: patch_rotate_channels(x, [0], ps),
        desc="R_rotated", n_repeats=n_rep)
    G_rotated = collect_spatial_averaged(
        lambda x: patch_rotate_channels(x, [1], ps),
        desc="G_rotated", n_repeats=n_rep)
    B_rotated = collect_spatial_averaged(
        lambda x: patch_rotate_channels(x, [2], ps),
        desc="B_rotated", n_repeats=n_rep)

    # ── 6. Per-channel blur then patch shuffle (NEW — texture removal) ──
    # Blur FIRST so patch boundary artifacts are identical to shuffle-only.
    logger.info("\n  ── Per-channel blur→shuffle (texture removal) ──")
    R_patch_blur = collect_spatial_averaged(
        lambda x: blur_then_patch_shuffle_channels(x, [0], ps, bk, bs),
        desc="R_blur_shuf", n_repeats=n_rep)
    G_patch_blur = collect_spatial_averaged(
        lambda x: blur_then_patch_shuffle_channels(x, [1], ps, bk, bs),
        desc="G_blur_shuf", n_repeats=n_rep)
    B_patch_blur = collect_spatial_averaged(
        lambda x: blur_then_patch_shuffle_channels(x, [2], ps, bk, bs),
        desc="B_blur_shuf", n_repeats=n_rep)

    # ── 7. All-independent blur + shuffle (full baseline) ──
    logger.info("\n  ── All-independent blur→shuffle (9-comp baseline) ──")
    baseline = collect_spatial_averaged(
        lambda x: all_independent_blur_and_shuffle(x, ps, bk, bs),
        desc="all_indep_blur_shuf", n_repeats=n_rep)

    # ══════════════════════════════════════════════════════════════
    # Phase 2: Compute 9 components & report
    # ══════════════════════════════════════════════════════════════
    save_data = {}  # collect per-metric results for saving

    for metric in ["l2sq", "gap"]:
        metric_label = "L2²" if metric == "l2sq" else "GAP"

        logger.info(f"\n{'='*60}")
        logger.info(f"  ══ {metric_label}-based 9-Component Decomposition ══")
        logger.info(f"{'='*60}")

        # Top-K means (each is (d_sae,) array)
        tk_orig     = topk_mean(orig[metric])
        tk_ab       = topk_mean(all_broken[metric])
        tk_R_only   = topk_mean(R_only[metric])
        tk_G_only   = topk_mean(G_only[metric])
        tk_B_only   = topk_mean(B_only[metric])
        tk_R_swap   = topk_mean(R_swap[metric])
        tk_G_swap   = topk_mean(G_swap[metric])
        tk_B_swap   = topk_mean(B_swap[metric])
        tk_R_patch  = topk_mean(R_patch[metric])
        tk_G_patch  = topk_mean(G_patch[metric])
        tk_B_patch  = topk_mean(B_patch[metric])
        tk_R_rot    = topk_mean(R_rotated[metric])
        tk_G_rot    = topk_mean(G_rotated[metric])
        tk_B_rot    = topk_mean(B_rotated[metric])
        tk_R_pb     = topk_mean(R_patch_blur[metric])
        tk_G_pb     = topk_mean(G_patch_blur[metric])
        tk_B_pb     = topk_mean(B_patch_blur[metric])
        tk_baseline = topk_mean(baseline[metric])

        # ── Components ──
        # Interactions (from step08 methodology)
        GB_inter = tk_R_only - tk_ab    # GB preserved when only R shifted
        RB_inter = tk_G_only - tk_ab    # RB preserved when only G shifted
        RG_inter = tk_B_only - tk_ab    # RG preserved when only B shifted

        # Hybrid (old "local") = swap - patch (confounded: local + reference)
        R_hybrid = tk_R_swap - tk_R_patch
        G_hybrid = tk_G_swap - tk_G_patch
        B_hybrid = tk_B_swap - tk_B_patch

        # Local shape (pure) = swap - rotated
        R_local = tk_R_swap - tk_R_rot
        G_local = tk_G_swap - tk_G_rot
        B_local = tk_B_swap - tk_B_rot

        # Reference frame = rotated - patch
        R_ref = tk_R_rot - tk_R_patch
        G_ref = tk_G_rot - tk_G_patch
        B_ref = tk_B_rot - tk_B_patch

        # Texture
        R_tex = tk_R_patch - tk_R_pb
        G_tex = tk_G_patch - tk_G_pb
        B_tex = tk_B_patch - tk_B_pb

        # ── Report ──
        logger.info(f"\n  {metric_label}(orig) mean: {tk_orig[alive_mask].mean():.6f}")
        logger.info(f"  {metric_label}(baseline) mean: {tk_baseline[alive_mask].mean():.6f}")
        logger.info(f"  {metric_label}(all_broken) mean: {tk_ab[alive_mask].mean():.6f}")

        def _log_comp(name, vals_arr):
            vals = vals_arr[alive_mask]
            n_neg = int((vals < 0).sum())
            logger.info(
                f"  {name:12s}: mean={vals.mean():+.6f}, "
                f"std={vals.std():.6f}, "
                f"neg={n_neg}/{len(vals)} ({n_neg/len(vals)*100:.1f}%)")

        logger.info(f"\n  ── Pairwise Channel Interactions ──")
        _log_comp("RG_inter", RG_inter)
        _log_comp("RB_inter", RB_inter)
        _log_comp("GB_inter", GB_inter)

        logger.info(f"\n  ── Per-Channel Hybrid (local + ref, confounded) ──")
        _log_comp("R_hybrid", R_hybrid)
        _log_comp("G_hybrid", G_hybrid)
        _log_comp("B_hybrid", B_hybrid)

        logger.info(f"\n  ── Per-Channel Local Shape (pure: swap - rotated) ──")
        _log_comp("R_local", R_local)
        _log_comp("G_local", G_local)
        _log_comp("B_local", B_local)

        logger.info(f"\n  ── Per-Channel Reference Frame (rotated - patch) ──")
        _log_comp("R_ref", R_ref)
        _log_comp("G_ref", G_ref)
        _log_comp("B_ref", B_ref)

        # Verify hybrid = local + ref
        logger.info(f"\n  ── Hybrid = Local + Ref verification ──")
        for ch in ["R", "G", "B"]:
            h = eval(f"{ch}_hybrid")[alive_mask].mean()
            l = eval(f"{ch}_local")[alive_mask].mean()
            r = eval(f"{ch}_ref")[alive_mask].mean()
            logger.info(f"  {ch}: hybrid={h:.6f}, local+ref={l+r:.6f}, diff={h-(l+r):.2e}")

        logger.info(f"\n  ── Per-Channel Texture ──")
        _log_comp("R_tex", R_tex)
        _log_comp("G_tex", G_tex)
        _log_comp("B_tex", B_tex)

        # ── Fractional contributions (12 components) ──
        logger.info(f"\n  ── Fractional Contributions (clipped to ≥0) ──")
        components = {
            "RG_inter": np.maximum(RG_inter, 0),
            "RB_inter": np.maximum(RB_inter, 0),
            "GB_inter": np.maximum(GB_inter, 0),
            "R_local":  np.maximum(R_local, 0),
            "G_local":  np.maximum(G_local, 0),
            "B_local":  np.maximum(B_local, 0),
            "R_ref":    np.maximum(R_ref, 0),
            "G_ref":    np.maximum(G_ref, 0),
            "B_ref":    np.maximum(B_ref, 0),
            "R_tex":    np.maximum(R_tex, 0),
            "G_tex":    np.maximum(G_tex, 0),
            "B_tex":    np.maximum(B_tex, 0),
        }
        total_comp = sum(components.values()) + 1e-12
        for name, vals in components.items():
            frac = vals / total_comp
            v = frac[alive_mask]
            logger.info(f"  {name:12s}: mean={v.mean():.4f}, std={v.std():.4f}")

        # Category-level fractions (4 categories)
        inter_total = components["RG_inter"] + components["RB_inter"] + components["GB_inter"]
        local_total = components["R_local"] + components["G_local"] + components["B_local"]
        ref_total   = components["R_ref"] + components["G_ref"] + components["B_ref"]
        tex_total   = components["R_tex"] + components["G_tex"] + components["B_tex"]

        for cat_name, cat_vals in [
            ("Interactions", inter_total),
            ("Local Shape",  local_total),
            ("Reference",    ref_total),
            ("Texture",      tex_total),
        ]:
            frac = cat_vals / total_comp
            v = frac[alive_mask]
            logger.info(f"  {'['+cat_name+']':14s}: mean={v.mean():.4f}, std={v.std():.4f}")

        # ── 9-Component Additivity Check (hybrid = local + ref) ──
        logger.info(f"\n  ── 9-Component Additivity Check ──")
        logger.info(f"  baseline + Σ(3 inter + 3 hybrid + 3 tex) ≈ orig")
        logger.info(f"  (hybrid = local + ref, so 9 = 3 inter + 3 local + 3 ref + 3 tex is 12)\n")

        sum_9 = (RG_inter + RB_inter + GB_inter +
                 R_hybrid + G_hybrid + B_hybrid +
                 R_tex + G_tex + B_tex)
        reconstructed = tk_baseline + sum_9

        # Aggregate
        agg_recon = reconstructed[alive_mask].sum()
        agg_orig = tk_orig[alive_mask].sum()
        logger.info(f"  Aggregate: {agg_recon:.4f} / {agg_orig:.4f} = "
                    f"{agg_recon / (agg_orig + 1e-12):.4f}")

        # Per-neuron linearity
        linearity = np.full(d_sae, np.nan)
        for n_i in range(d_sae):
            if not alive_mask[n_i]:
                continue
            idx = topk_indices[n_i]
            o = orig[metric][idx, n_i]
            # Reconstruct per-image (using hybrid = swap - patch)
            bl = baseline[metric][idx, n_i]
            rg_i = B_only[metric][idx, n_i] - all_broken[metric][idx, n_i]
            rb_i = G_only[metric][idx, n_i] - all_broken[metric][idx, n_i]
            gb_i = R_only[metric][idx, n_i] - all_broken[metric][idx, n_i]
            rl = R_swap[metric][idx, n_i] - R_patch[metric][idx, n_i]
            gl = G_swap[metric][idx, n_i] - G_patch[metric][idx, n_i]
            bbl = B_swap[metric][idx, n_i] - B_patch[metric][idx, n_i]
            rt = R_patch[metric][idx, n_i] - R_patch_blur[metric][idx, n_i]
            gt = G_patch[metric][idx, n_i] - G_patch_blur[metric][idx, n_i]
            bt = B_patch[metric][idx, n_i] - B_patch_blur[metric][idx, n_i]
            recon = bl + rg_i + rb_i + gb_i + rl + gl + bbl + rt + gt + bt

            valid = o > 1e-10
            if valid.sum() < 2:
                continue
            linearity[n_i] = np.median(recon[valid] / o[valid])

        valid_lin = linearity[alive_mask]
        valid_lin = valid_lin[np.isfinite(valid_lin)]
        logger.info(f"  Per-neuron linearity (median of per-image ratio):")
        logger.info(f"    N neurons: {len(valid_lin)}")
        logger.info(f"    mean={valid_lin.mean():.4f}, median={np.median(valid_lin):.4f}")
        logger.info(f"    5th/95th pctl: [{np.percentile(valid_lin, 5):.4f}, "
                    f"{np.percentile(valid_lin, 95):.4f}]")
        in_band = ((valid_lin > 0.8) & (valid_lin < 1.2)).sum()
        logger.info(f"    [0.8-1.2]: {in_band}/{len(valid_lin)} "
                    f"({in_band/len(valid_lin)*100:.1f}%)")

        # ── Partial checks ──
        logger.info(f"\n  ── Partial Additivity (consistency checks) ──")
        inter_sum = (RG_inter + RB_inter + GB_inter)[alive_mask].sum()
        total_spatial = (tk_orig - tk_ab)[alive_mask].sum()
        logger.info(f"  Interactions: sum={inter_sum:.4f} / "
                    f"total_spatial={total_spatial:.4f} = "
                    f"{inter_sum / (total_spatial + 1e-12):.4f}")

        for ch_name, swap_val, rot_val, patch_val, pb_val in [
            ("R", tk_R_swap, tk_R_rot, tk_R_patch, tk_R_pb),
            ("G", tk_G_swap, tk_G_rot, tk_G_patch, tk_G_pb),
            ("B", tk_B_swap, tk_B_rot, tk_B_patch, tk_B_pb),
        ]:
            hybrid_v = (swap_val - patch_val)[alive_mask].mean()
            local_v = (swap_val - rot_val)[alive_mask].mean()
            ref_v = (rot_val - patch_val)[alive_mask].mean()
            tex_v = (patch_val - pb_val)[alive_mask].mean()
            logger.info(f"    {ch_name}: hybrid={hybrid_v:.6f} (local={local_v:.6f} + ref={ref_v:.6f}), tex={tex_v:.6f}")

        # ── Collect per-metric data for saving ──
        m = metric  # "l2sq" or "gap"
        save_data[f"{m}_RG_inter"]    = RG_inter
        save_data[f"{m}_RB_inter"]    = RB_inter
        save_data[f"{m}_GB_inter"]    = GB_inter
        save_data[f"{m}_R_hybrid"]    = R_hybrid
        save_data[f"{m}_G_hybrid"]    = G_hybrid
        save_data[f"{m}_B_hybrid"]    = B_hybrid
        save_data[f"{m}_R_local"]     = R_local
        save_data[f"{m}_G_local"]     = G_local
        save_data[f"{m}_B_local"]     = B_local
        save_data[f"{m}_R_ref"]       = R_ref
        save_data[f"{m}_G_ref"]       = G_ref
        save_data[f"{m}_B_ref"]       = B_ref
        save_data[f"{m}_R_tex"]       = R_tex
        save_data[f"{m}_G_tex"]       = G_tex
        save_data[f"{m}_B_tex"]       = B_tex
        save_data[f"{m}_linearity"]   = linearity
        save_data[f"{m}_tk_orig"]     = tk_orig
        save_data[f"{m}_tk_baseline"] = tk_baseline
        save_data[f"{m}_tk_all_broken"] = tk_ab
        save_data[f"{m}_tk_R_only"]   = tk_R_only
        save_data[f"{m}_tk_G_only"]   = tk_G_only
        save_data[f"{m}_tk_B_only"]   = tk_B_only
        save_data[f"{m}_tk_R_swap"]   = tk_R_swap
        save_data[f"{m}_tk_G_swap"]   = tk_G_swap
        save_data[f"{m}_tk_B_swap"]   = tk_B_swap
        save_data[f"{m}_tk_R_patch"]  = tk_R_patch
        save_data[f"{m}_tk_G_patch"]  = tk_G_patch
        save_data[f"{m}_tk_B_patch"]  = tk_B_patch
        save_data[f"{m}_tk_R_rot"]    = tk_R_rot
        save_data[f"{m}_tk_G_rot"]    = tk_G_rot
        save_data[f"{m}_tk_B_rot"]    = tk_B_rot
        save_data[f"{m}_tk_R_pb"]     = tk_R_pb
        save_data[f"{m}_tk_G_pb"]     = tk_G_pb
        save_data[f"{m}_tk_B_pb"]     = tk_B_pb

    # ── Save ──
    npz_path = os.path.join(out_dir, f"texture_attribution_ps{ps}_blur{bs:.1f}.npz")
    np.savez_compressed(
        npz_path,
        # ── Metadata ──
        alive_mask=alive_mask,
        usage_ema=usage_ema,
        topk_indices=topk_indices,
        patch_size=ps,
        blur_sigma=bs,
        blur_kernel_size=bk,
        n_shuffle_repeats=n_rep,
        top_k=k_actual,
        seam_margin=args.seam_margin,
        # ── Per-metric per-neuron data ──
        **save_data,
    )
    n_keys = len(save_data) + 9  # 9 metadata keys
    logger.info(f"\nSaved: {npz_path}")
    logger.info(f"  {n_keys} arrays, file size: {os.path.getsize(npz_path)/1e6:.1f} MB")
    logger.info("Done!")


if __name__ == "__main__":
    main()
