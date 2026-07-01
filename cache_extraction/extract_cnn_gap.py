# ==============================================================================
# CNN GAP Feature Extraction (SAE 없이 CNN 자체의 GAP 벡터 추출)
#
# 저장 내용 (.npz):
#   X_gap       : (N, C)         — CNN feature map GAP vector (L2 normalized)
#   y           : (N,)           — class labels
#   lines       : (N,)           — cell line names
#   uids        : (N,)           — unique image IDs
#   which_layer : str            — 추출한 layer 이름
#
# Usage:
#   python -m cache_extraction.extract_cnn_gap \
#       --save_dir /path/to/MoCo_seedXX \
#       --model_state_path /path/to/best_model.pt \
#       --shard_root /path/to/wds_shards_tar \
#       --which_layer stage5_mid
# ==============================================================================

import argparse
import gc
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from run_CNN.logging_utils import SUPERCLASS_MAP, get_logger
from run_CNN.data_shards import (build_uid_to_refidx,
                                            load_all_sample_refs)
from run_CNN.data_bank import (InMemorySixteenBitDataset,
                                          InMemoryTarBank, collate_skip_none,
                                          load_split_csv, seed_worker)
from run_CNN.model_encoder import (SupMoCoModel, parse_int_list,
                                              renorm_unit_per_out_channel_,
                                              robust_load_state_dict)

logger = get_logger("extract_cnn_gap")


# ==============================================================================
# Argument Parser
# ==============================================================================
def get_args():
    p = argparse.ArgumentParser(
        description="Extract CNN GAP features (no SAE) and save to cache (.npz)"
    )

    # Model
    p.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Model output dir (contains train/val/test_split.csv)",
    )
    p.add_argument("--model_state_path", type=str, required=True)
    p.add_argument("--shard_root", type=str, required=True)

    # Output
    p.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Directory for .npz cache file (default: <save_dir>/CNN_GAP)",
    )

    # Feature extraction
    p.add_argument(
        "--which_layer",
        type=str,
        default="stage5_mid",
        choices=["stage5_mid", "stage5_out", "refine_out"],
        help="Encoder layer to extract GAP from",
    )

    # Sampling
    p.add_argument(
        "--samples_per_class",
        type=int,
        default=5000,
        help="Samples per class (0 = use ALL, no sampling)",
    )
    p.add_argument(
        "--use_all_data",
        action="store_true",
        help="Load train+val+test (default: val+test only). "
        "Also sets samples_per_class=0 if not explicitly set",
    )
    p.add_argument(
        "--ignore_splits",
        action="store_true",
        help="Ignore train/val/test split CSVs and use ALL images found in shard_root",
    )
    p.add_argument("--seed", type=int, default=42)

    # Encoder architecture
    p.add_argument("--blocks", type=str, default="2,2,2,3")
    p.add_argument("--dilations", type=str, default="1,1,1,1")
    p.add_argument("--refine_blocks", type=int, default=1)
    p.add_argument("--ckpt_segments", type=int, default=0)
    p.add_argument("--embed_dim", type=int, default=512)
    p.add_argument("--proj_layers", type=int, default=2)
    p.add_argument("--proj_hidden", type=int, default=2048)
    p.add_argument("--proj_bn", action="store_true")
    p.add_argument("--proj_dropout", type=float, default=0.0)

    # Data
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)

    return p.parse_args()


# ==============================================================================
# CNN GAP Feature Extraction (no SAE)
# ==============================================================================
@torch.no_grad()
def extract_cnn_gap_features(
    encoder,
    loader: DataLoader,
    device: torch.device,
    which_layer: str,
) -> tuple:
    """
    Extract CNN GAP features directly from encoder feature maps.

    Process:
        1. encoder.forward_feature_maps(x, which_layer) → (B, C, H, W)
        2. GAP: mean(dim=(2,3)) → (B, C)
        3. L2 normalize → (B, C)

    Returns:
        X_gap: (N, C) numpy float32  — L2-normalized GAP vectors
        labels: (N,) int array
        lines: list[str]
        uids: list[str]
    """
    encoder.eval()

    autocast_kwargs = dict(device_type="cuda", enabled=torch.cuda.is_available())
    if torch.cuda.is_available():
        autocast_kwargs["dtype"] = torch.bfloat16

    X_list, y_list, line_list, uid_list = [], [], [], []

    for batch in tqdm(loader, desc=f"Extracting CNN GAP ({which_layer})", leave=True):
        if batch is None:
            continue

        x_cpu = batch[0]
        y_cpu = batch[1]
        batch_lines = batch[3] if len(batch) > 3 else ["unknown"] * x_cpu.size(0)
        batch_uids = batch[4] if len(batch) > 4 else ["unknown"] * x_cpu.size(0)

        if x_cpu.numel() < 1:
            continue

        x = x_cpu.to(device, non_blocking=True).contiguous(
            memory_format=torch.channels_last
        )

        with torch.amp.autocast(**autocast_kwargs):
            fmap = encoder.forward_feature_maps(x, which=which_layer)  # (B, C, H, W)

        # GAP → (B, C)  — raw (L2 norm 안 함, 필요시 후처리)
        gap = fmap.float().mean(dim=(2, 3))

        X_list.append(gap.cpu().numpy())
        y_list.extend(y_cpu.tolist())
        line_list.extend(batch_lines)
        uid_list.extend(batch_uids)

    if len(X_list) == 0:
        raise ValueError(
            "No images were extracted! This is likely a UID path mismatch: "
            "split CSVs contain paths from a different machine. "
            "Fix: create a symlink, e.g. ln -s /content/wds_shards /home/ubuntu/model-east3/wds_shards_tar"
        )

    X_gap = np.concatenate(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)

    return X_gap, y, line_list, uid_list


# ==============================================================================
# Data Loading (reused from extract_features.py)
# ==============================================================================
def make_balanced_loader(
    args, refs, uid_to_refidx, samples_per_class, seed, include_train=False
):
    """Load data, optionally balanced per class.

    include_train: if True, also loads train_split.csv
    samples_per_class: 0 = use ALL (no sampling)
    """
    if getattr(args, "ignore_splits", False):
        refidx_list = list(range(len(refs)))
        logger.info(
            f"  [!] --ignore_splits is ON: Using all {len(refs)} images found in shard_root, bypassing CSVs."
        )
    else:
        csv_paths = []
        if include_train:
            csv_paths.append(os.path.join(args.save_dir, "train_split.csv"))
        csv_paths.append(os.path.join(args.save_dir, "val_split.csv"))
        csv_paths.append(os.path.join(args.save_dir, "test_split.csv"))

        all_uids = []
        for csv_path in csv_paths:
            if os.path.exists(csv_path):
                uids = load_split_csv(csv_path)
                all_uids.extend(uids)
                logger.info(f"  Loaded {csv_path}: {len(uids)} UIDs")
            else:
                logger.warning(f"  Not found (skipping): {csv_path}")

        if not all_uids:
            raise FileNotFoundError(f"No val/test CSVs in {args.save_dir}")

        # Normalize UIDs for cross-machine path matching
        KNOWN_ROOTS = [
            "/home/ubuntu/model-east3/wds_shards_tar/",
            "/home/ubuntu/model-east3/wds_shards_tar\\",
            args.shard_root.rstrip("/\\") + "/",
            args.shard_root.rstrip("/\\") + "\\",
        ]

        def uid_to_relative(uid: str) -> str:
            for root in KNOWN_ROOTS:
                if uid.startswith(root):
                    return uid[len(root) :]
            for cls_prefix in ["Control/", "SNCA/", "GBA/", "LRRK2/"]:
                idx = uid.find(cls_prefix)
                if idx >= 0:
                    return uid[idx:]
            return uid

        rel_to_refidx = {uid_to_relative(k): v for k, v in uid_to_refidx.items()}
        refidx_list = []
        n_missing = 0
        for uid in all_uids:
            rel_key = uid_to_relative(uid)
            if rel_key in rel_to_refidx:
                refidx_list.append(rel_to_refidx[rel_key])

        # [NEW] CSV 파일에 없는 OOD 클래스 (Label 4 이상)는 폴더에 있는 모든 이미지를 무조건 포함!
        # (이미 targz_to_wds 변환 단계에서 27,000장으로 맞춰두었으므로 그대로 다 쓰면 됩니다)
        for rel_key, ridx in rel_to_refidx.items():
            if int(refs[ridx].label) >= 4:
                refidx_list.append(ridx)
            else:
                n_missing += 1

        if n_missing > 0:
            logger.warning(
                f"  {n_missing}/{len(all_uids)} UIDs not matched (path mismatch?)"
            )
        logger.info(f"  Matched: {len(refidx_list)}/{len(all_uids)} UIDs")

    # Group by class AND line to ensure strict balancing among lines within the same class
    class_to_lines = defaultdict(lambda: defaultdict(list))
    for i, ridx in enumerate(refidx_list):
        label = int(refs[ridx].label)
        line = refs[ridx].line
        class_to_lines[label][line].append(i)

    rng = np.random.default_rng(seed)
    selected = []

    for cls in sorted(class_to_lines.keys()):
        lines_dict = class_to_lines[cls]
        num_lines = len(lines_dict)

        if samples_per_class > 0:
            target_total = samples_per_class
            base_take = target_total // num_lines
            remainder = target_total % num_lines

            sorted_lines = sorted(lines_dict.keys())
            class_selected = []

            for idx, line_name in enumerate(sorted_lines):
                pool = lines_dict[line_name]
                take_for_this_line = base_take + (1 if idx < remainder else 0)
                n_take = min(take_for_this_line, len(pool))
                chosen = rng.choice(pool, size=n_take, replace=False).tolist()
                class_selected.extend(chosen)
                logger.info(
                    f"    Line {line_name} (Class {cls}): {n_take}/{len(pool)} selected"
                )

            selected.extend(class_selected)
            logger.info(
                f"  Class {cls} Total: {len(class_selected)} selected (across {num_lines} lines)"
            )
        else:
            class_selected = []
            for line_name in sorted(lines_dict.keys()):
                class_selected.extend(lines_dict[line_name])
            selected.extend(class_selected)
            logger.info(f"  Class {cls}: {len(class_selected)} selected (all)")

    selected_refidx = [refidx_list[i] for i in selected]

    bank = InMemoryTarBank(refs, selected_refidx, args.img_size)
    ib = list(range(len(selected_refidx)))
    ds = InMemorySixteenBitDataset(bank, ib, args.img_size, augment=False)

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        collate_fn=collate_skip_none,
    )
    logger.info(f"  Total: {len(selected)} images")
    return loader


# ==============================================================================
# Main
# ==============================================================================
def main():
    args = get_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── 1. Load encoder ──────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
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
        proj_bn=args.proj_bn,
        proj_dropout=args.proj_dropout,
    )
    sd = torch.load(args.model_state_path, map_location="cpu", weights_only=False)
    robust_load_state_dict(model, sd, strict=True)
    encoder = model.encoder
    renorm_unit_per_out_channel_(encoder)
    encoder.to(device).eval()

    del model, sd
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── 2. Load data ─────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")

    use_all = args.use_all_data
    spc = args.samples_per_class
    if use_all and spc == 5000:  # default not changed
        spc = 0

    splits_label = "train+val+test" if use_all else "val+test"
    if spc == 0:
        logger.info(f"Loading ALL data ({splits_label}) — no sampling")
    else:
        logger.info(f"Loading data ({splits_label}), {spc}/class")

    refs = load_all_sample_refs(args.shard_root)
    uid_to_refidx = build_uid_to_refidx(refs)

    loader = make_balanced_loader(
        args,
        refs,
        uid_to_refidx,
        samples_per_class=spc,
        seed=args.seed,
        include_train=use_all,
    )

    # ── 3. Extract CNN GAP features ──────────────────────────────────────
    which_layer = args.which_layer
    logger.info(f"\n{'='*60}")
    logger.info(f"Extracting CNN GAP features — layer={which_layer}")

    X_gap, y, lines, uids = extract_cnn_gap_features(
        encoder,
        loader,
        device,
        which_layer,
    )
    logger.info(f"Features: {X_gap.shape}")

    # ── 4. Save cache ────────────────────────────────────────────────────
    output_dir = (
        args.output_dir if args.output_dir else os.path.join(args.save_dir, "CNN_GAP")
    )
    os.makedirs(output_dir, exist_ok=True)

    if getattr(args, "ignore_splits", False):
        all_tag = "_withnewclass"
    else:
        all_tag = "_all" if use_all else ""
    out_path = os.path.join(output_dir, f"cnn_gap_{which_layer}{all_tag}.npz")

    logger.info(f"\n{'='*60}")
    logger.info(f"Saving cache: {out_path}")
    logger.info(f"  X_gap: {X_gap.shape} ({X_gap.nbytes / 1e6:.1f} MB)")

    np.savez_compressed(
        out_path,
        X_gap=X_gap,
        y=y,
        lines=np.array(lines, dtype=object),
        uids=np.array(uids, dtype=object),
        which_layer=np.array(which_layer),
    )

    file_size_mb = os.path.getsize(out_path) / 1e6
    logger.info(f"  File size: {file_size_mb:.1f} MB")

    # ── Summary ──────────────────────────────────────────────────────────
    superclasses = [SUPERCLASS_MAP.get(ln, ln) for ln in lines]
    unique_classes, class_counts = np.unique(superclasses, return_counts=True)
    logger.info(f"\n  Classes: {dict(zip(unique_classes, class_counts))}")

    logger.info(f"\n{'='*60}")
    logger.info("CNN GAP feature extraction complete!")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
