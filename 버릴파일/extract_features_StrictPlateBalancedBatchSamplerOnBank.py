# ==============================================================================
# Feature Extraction + Cache  (Lambda Labs — GPU 필요)
#
# ALL d_sae neurons 저장 (alive_mask 적용 안 함)
# → 코랩/로컬에서 dead_threshold 자유롭게 변경 가능
#
# 저장 내용 (.npz):
#   X_all       : (N, d_sae)     — 전체 SAE neuron GAP features
#   y           : (N,)           — class labels
#   lines       : (N,)           — cell line names
#   uids        : (N,)           — unique image IDs
#   usage_ema   : (d_sae,)       — SAE neuron usage EMA (alive_mask 재구성용)
#   which_layer : str            — SAE가 학습된 layer 이름
#
# Usage:
#   python -m kendall_correlation_coefficient.extract_features \
#       --sae_ckpt /path/to/sae.pt \
#       --save_dir /path/to/MoCo_seedXX \
#       --model_state_path /path/to/best_model.pt \
#       --shard_root /path/to/wds_shards_tar \
#       --restore_token_norm
# ==============================================================================

# SAE에서 batch size = 64로 해서 벡터들 뽑고 있었다.

## 토큰 배치 센터링할때 학습시에는 StrictPlateBalancedBatchSamplerOnBank 이걸 이용해서, 골고루 플레이트와 클래스별로 다 들어가서 토큰 센터링 하고 그 토큰들로 학습한다.
# 따라서 StrictPlateBalancedBatchSamplerOnBank 이걸 쓰는게 학습시와 동일하게 한다. 다만 배치 랜덤성이 개입될 여지가 있다.

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

from sae_project.step02_logging_utils import SUPERCLASS_MAP, get_logger
from sae_project.step03_data_shards import (build_uid_to_refidx,
                                            load_all_sample_refs)
from sae_project.step04_data_bank import (
    InMemorySixteenBitDataset, InMemoryTarBank,
    StrictPlateBalancedBatchSamplerOnBank, collate_skip_none, load_split_csv,
    seed_worker)
from sae_project.step05_model_encoder import (SupMoCoModel, parse_int_list,
                                              renorm_unit_per_out_channel_,
                                              robust_load_state_dict)
from sae_project.step06_gated_sae import GatedSAE

logger = get_logger("extract_features")


# ==============================================================================
# Argument Parser
# ==============================================================================
def get_args():
    p = argparse.ArgumentParser(
        description="Extract SAE GAP features and save to cache (.npz)"
    )

    # SAE / Model
    p.add_argument("--sae_ckpt", type=str, required=True)
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
        "--output_path",
        type=str,
        default="",
        help="Path for .npz cache file (default: next to SAE ckpt)",
    )

    # Feature extraction
    p.add_argument(
        "--which_layer",
        type=str,
        default="",
        help="Encoder layer to extract (default: use SAE ckpt value, e.g. refine_out, stage5_out)",
    )
    p.add_argument(
        "--restore_token_norm",
        action="store_true",
        help="Multiply SAE activations by original per-token L2 norms before pooling",
    )

    # Sampling
    p.add_argument(
        "--use_all_data",
        action="store_true",
        help="Load train+val+test (default: val+test only)",
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
# Feature Extraction — ALL neurons (no alive_mask)
# ==============================================================================
@torch.no_grad()
def extract_all_sae_features(
    encoder,
    sae: GatedSAE,
    loaders: list,
    device: torch.device,
    which_layer: str,
    restore_token_norm: bool = False,
) -> tuple:
    """
    Extract SAE GAP features for ALL neurons (no alive_mask filtering).

    Args:
        loaders: list of (name, DataLoader) pairs — iterated sequentially
                 (same pattern as step09_sae_eval.evaluate_concepts_for_sae)

    Returns:
        X_all: (N, d_sae) numpy float32  — full neuron activations
        labels: (N,) int array
        lines: list[str]
        uids: list[str]
    """
    encoder.eval()
    sae.eval()

    autocast_kwargs = dict(device_type="cuda", enabled=torch.cuda.is_available())
    if torch.cuda.is_available():
        autocast_kwargs["dtype"] = torch.bfloat16

    X_list, y_list, line_list, uid_list = [], [], [], []

    for split_name, loader in loaders:
        for batch in tqdm(
            loader, desc=f"Extracting SAE features ({split_name})", leave=True
        ):
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
            curr_bs = x.size(0)

            with torch.amp.autocast(**autocast_kwargs):
                fmap = encoder.forward_feature_maps(x, which=which_layer)

            # GAP-scalar normalization
            gap = fmap.mean(dim=(2, 3))
            gap_norm = (
                gap.norm(dim=1, keepdim=True).view(curr_bs, 1, 1, 1).clamp_min(1e-12)
            )
            fmap = fmap / gap_norm

            fmap = fmap.permute(0, 2, 3, 1).contiguous()
            C = fmap.shape[-1]
            H_W = fmap.shape[1] * fmap.shape[2]

            flat_tokens = fmap.view(-1, C)
            flat_tokens = flat_tokens - flat_tokens.mean(dim=0, keepdim=True)

            # Save per-token L2 norms before normalization
            token_l2_norms = flat_tokens.norm(dim=1, keepdim=True).clamp_min(1e-12)

            flat_tokens = F.normalize(flat_tokens, dim=1, eps=1e-12)

            # SAE forward in chunks — accumulate per-image sums directly
            # (same pattern as step09_sae_eval.py to avoid OOM on large d_sae)
            num_tokens_per_img = H_W
            token_batch_size = 8192
            num_flat_tokens = flat_tokens.size(0)
            image_act_sums = torch.zeros(
                (curr_bs, sae.d_sae), device=device, dtype=torch.float32
            )

            for s in range(0, num_flat_tokens, token_batch_size):
                e = min(s + token_batch_size, num_flat_tokens)
                chunk = flat_tokens[s:e]
                with torch.amp.autocast(**autocast_kwargs):
                    _, chunk_acts, _, _, _ = sae(chunk)
                chunk_acts = chunk_acts.float()

                # Optionally restore per-token L2 norms
                if restore_token_norm:
                    chunk_acts = chunk_acts * token_l2_norms[s:e]

                # Accumulate sum per image
                for i in range(curr_bs):
                    img_start = i * num_tokens_per_img
                    img_end = (i + 1) * num_tokens_per_img
                    rel_start = max(0, img_start - s)
                    rel_end = min(e - s, img_end - s)
                    if rel_start < rel_end and img_start < e and img_end > s:
                        image_act_sums[i] += chunk_acts[rel_start:rel_end].sum(dim=0)

                del chunk_acts

            pooled = image_act_sums  # (B, d_sae) — sum over tokens

            # Save ALL neurons (no alive_mask filtering!)
            X_list.append(pooled.cpu().numpy())

            y_list.extend(y_cpu.tolist())
            line_list.extend(batch_lines)
            uid_list.extend(batch_uids)

    if len(X_list) == 0:
        raise ValueError(
            "No images were extracted! This is likely a UID path mismatch: "
            "split CSVs contain paths from a different machine (e.g. Lambda Labs). "
            "Fix: create a symlink, e.g. ln -s /content/wds_shards /home/ubuntu/model-east3/wds_shards_tar"
        )

    X_all = np.concatenate(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)

    return X_all, y, line_list, uid_list


# ==============================================================================
# Data Loading — StrictPlateBalancedBatchSampler
# (step09_train_gated_sae.py의 _make_eval_loaders와 동일한 방식)
# ==============================================================================
def _make_single_split_loader(args, refs, uid_to_refidx, split_csv_path, seed_offset):
    """Create a single split loader using StrictPlateBalancedBatchSamplerOnBank.

    Matches step09_train_gated_sae._make_loader_from_split (eval mode).
    Includes cross-machine path normalization for Lambda Labs ↔ Colab compatibility.
    """
    if not os.path.exists(split_csv_path):
        logger.warning(f"  Not found (skipping): {split_csv_path}")
        return None, 0

    uids = load_split_csv(split_csv_path)
    logger.info(f"  Loaded {split_csv_path}: {len(uids)} UIDs")

    # Cross-machine path normalization
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
        for cls_prefix in [
            "Control/",
            "SNCA/",
            "GBA/",
            "LRRK2/",
            "Control_C4/",
            "Control_C18/",
            "Control_C19/",
        ]:
            idx = uid.find(cls_prefix)
            if idx >= 0:
                return uid[idx:]
        return uid

    # Build relative-path lookup from shard refs
    rel_to_refidx = {}
    for uid, ridx in uid_to_refidx.items():
        rel_key = uid_to_relative(uid)
        rel_to_refidx[rel_key] = ridx

    # Match UIDs via relative paths
    refidx_list = []
    n_missing = 0
    matched_uids = []
    for uid in uids:
        rel_key = uid_to_relative(uid)
        if rel_key in rel_to_refidx:
            refidx_list.append(rel_to_refidx[rel_key])
            matched_uids.append(uid)
        else:
            n_missing += 1

    if n_missing > 0:
        logger.warning(f"  {n_missing}/{len(uids)} UIDs not matched (path mismatch?)")
        if n_missing == len(uids) and len(uids) > 0:
            # Show example for debugging
            logger.warning(f"    CSV UID example: {uids[0]}")
            logger.warning(f"    Shard UID example: {list(uid_to_refidx.keys())[0]}")
    logger.info(f"  Matched: {len(refidx_list)}/{len(uids)} UIDs")

    if not refidx_list:
        return None, 0

    bank = InMemoryTarBank(refs, refidx_list, args.img_size)
    ib = list(range(len(refidx_list)))
    ds = InMemorySixteenBitDataset(bank, ib, args.img_size, augment=False)

    # StrictPlateBalancedBatchSamplerOnBank: match SAE training centering
    sampler = StrictPlateBalancedBatchSamplerOnBank(
        bank,
        batch_size=args.batch_size,
        seed=args.seed + seed_offset,
    )
    loader = DataLoader(
        ds,
        batch_sampler=sampler,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        collate_fn=collate_skip_none,
    )
    n_covered = sum(len(b) for b in sampler)
    logger.info(
        f"  → {len(refidx_list)} images, StrictPlateBalanced ({n_covered} covered)"
    )
    return loader, n_covered


def make_balanced_loaders(args, refs, uid_to_refidx, include_train=False):
    """Create per-split loaders using StrictPlateBalancedBatchSamplerOnBank.

    Returns list of (name, loader) pairs matching step09's eval loader pattern.
    Centering is per-batch with mixed-class batches, consistent with SAE training.
    """
    loaders = []
    total = 0

    if include_train:
        train_csv = os.path.join(args.save_dir, "train_split.csv")
        loader, n = _make_single_split_loader(
            args, refs, uid_to_refidx, train_csv, seed_offset=100
        )
        if loader is not None:
            loaders.append(("train", loader))
            total += n

    val_csv = os.path.join(args.save_dir, "val_split.csv")
    loader, n = _make_single_split_loader(
        args, refs, uid_to_refidx, val_csv, seed_offset=102
    )
    if loader is not None:
        loaders.append(("val", loader))
        total += n

    test_csv = os.path.join(args.save_dir, "test_split.csv")
    loader, n = _make_single_split_loader(
        args, refs, uid_to_refidx, test_csv, seed_offset=101
    )
    if loader is not None:
        loaders.append(("test", loader))
        total += n

    if not loaders:
        raise FileNotFoundError(f"No split CSVs found in {args.save_dir}")

    logger.info(f"  Total: {total} images across {len(loaders)} splits")
    return loaders


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

    # ── 1. Load SAE ──────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
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
    sae.to(device).eval()

    which_layer_ckpt = ckpt_args.get("which_layer", "refine_out")
    which_layer = args.which_layer if args.which_layer else which_layer_ckpt
    d_sae = sae.d_sae
    usage_ema = sae.usage_ema.cpu().numpy()

    if args.which_layer and args.which_layer != which_layer_ckpt:
        logger.warning(
            f"Overriding SAE ckpt layer '{which_layer_ckpt}' → '{which_layer}'"
        )
    logger.info(f"SAE: d_sae={d_sae}, layer={which_layer}")

    # ── 2. Load encoder ──────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("Loading encoder")

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

    # ── 3. Load data ─────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")

    use_all = args.use_all_data
    splits_label = "train+val+test" if use_all else "val+test"
    logger.info(f"Loading data ({splits_label}) with StrictPlateBalancedBatchSampler")

    refs = load_all_sample_refs(args.shard_root)
    uid_to_refidx = build_uid_to_refidx(refs)

    loaders = make_balanced_loaders(
        args,
        refs,
        uid_to_refidx,
        include_train=use_all,
    )

    # ── 4. Extract features (ALL neurons) ────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(
        f"Extracting SAE GAP features — ALL {d_sae} neurons "
        f"(restore_token_norm={args.restore_token_norm})"
    )

    X_all, y, lines, uids = extract_all_sae_features(
        encoder,
        sae,
        loaders,
        device,
        which_layer,
        restore_token_norm=args.restore_token_norm,
    )
    logger.info(f"Features: {X_all.shape}")

    # ── 5. Save cache ────────────────────────────────────────────────────
    if args.output_path:
        out_path = args.output_path
    else:
        suffix = "_normrestored" if args.restore_token_norm else ""
        all_tag = "_all" if use_all else ""
        out_path = os.path.join(
            os.path.dirname(args.sae_ckpt),
            f"features_cache_{which_layer}{suffix}{all_tag}.npz",
        )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"Saving cache: {out_path}")
    logger.info(f"  X_all:     {X_all.shape} ({X_all.nbytes / 1e6:.1f} MB)")
    logger.info(f"  usage_ema: {usage_ema.shape}")

    np.savez_compressed(
        out_path,
        X_all=X_all,
        y=y,
        lines=np.array(lines, dtype=object),
        uids=np.array(uids, dtype=object),
        usage_ema=usage_ema,
        which_layer=np.array(which_layer),
    )

    file_size_mb = os.path.getsize(out_path) / 1e6
    logger.info(f"  File size: {file_size_mb:.1f} MB")

    # ── Summary ──────────────────────────────────────────────────────────
    from sae_project.step02_logging_utils import SUPERCLASS_MAP

    superclasses = [SUPERCLASS_MAP.get(ln, ln) for ln in lines]
    unique_classes, class_counts = np.unique(superclasses, return_counts=True)
    logger.info(f"\n  Classes: {dict(zip(unique_classes, class_counts))}")

    # Quick alive neuron stats at various thresholds
    for thr in [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]:
        n_alive = int(np.sum(usage_ema >= thr))
        logger.info(f"  dead_threshold={thr:.0e}: {n_alive}/{d_sae} alive")

    logger.info(f"\n{'='*60}")
    logger.info("Feature extraction complete!")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
