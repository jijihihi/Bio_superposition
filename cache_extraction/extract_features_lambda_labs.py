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

### 람다 랩스에서 돌려야지만 결과 나옴. 코랩은 안됨. 이유는 모르겠어.

## 배치 센터링 할때, 한 배치의 모든 이미지 모든 토큰 함쳐서 B H W = 64*64*64 토큰. 이 토큰 채널별 평균 구함. 모든 토큰에서 이 평균 뺌.
# 즉 같은 class 이미지들이 같은 배치에 들어갈 확률이 높음.


# 토큰 워크 플로우

# DPT. SAE sparsity 800, loss function에 L2 norm 안 곱해서 훈련. (strict plate 사용)
# 각 이미지 뽑아낼때는 strict plate 사용 안하고 shuffle = false 해서 배치해서 각 이미지 벡터로 뽑아냄(extract_features_lambda_labs.py) 배치 사이즈 64. 토큰 centering하기 위한. (DPT 벡터 뽑아낼때)
# 이걸로 min cv DE filter 함


# bilinear interpolation. sparsity 3200, loss function에 L2 norm 안 곱해서 훈련 (strict plate 사용)
# 각 이미지 bilinear interpolation 할때는 per imgae token centering함.
# 어떤 뉴런 볼지 할때는 gap_csv 기반 했는데, batch centering (StrictPlateBalanced) 해서 각 뉴런 각 이미지에 대한 활성화하고, 여기서 DE filter로 해서 뽑아냄. (❌ restore_token_norm 없음, ❌ pooled 후 F.normalize 없음, ✅ 토큰 L2 정규화 있음, ✅ batch centering (StrictPlateBalanced))


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
from sae_project.step04_data_bank import (InMemorySixteenBitDataset,
                                          InMemoryTarBank, collate_skip_none,
                                          load_split_csv, seed_worker)
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
        "--samples_per_class",
        type=int,
        default=0,
        help="Samples per class (0 = use ALL, no sampling)",
    )
    p.add_argument(
        "--use_all_data",
        action="store_true",
        help="Load train+val+test (default: val+test only). "
        "Also sets samples_per_class=0 if not explicitly set",
    )
    p.add_argument(
        "--test_only", action="store_true", help="Load strictly ONLY test_split.csv"
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
# Feature Extraction — ALL neurons (no alive_mask)
# ==============================================================================
@torch.no_grad()
def extract_all_sae_features(
    encoder,
    sae: GatedSAE,
    loader: DataLoader,
    device: torch.device,
    which_layer: str,
    restore_token_norm: bool = False,
) -> tuple:
    """
    Extract SAE GAP features for ALL neurons (no alive_mask filtering).

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

    for batch in tqdm(loader, desc="Extracting SAE features (all neurons)", leave=True):
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
        gap_norm = gap.norm(dim=1, keepdim=True).view(curr_bs, 1, 1, 1).clamp_min(1e-12)
        fmap = fmap / gap_norm

        fmap = fmap.permute(0, 2, 3, 1).contiguous()
        C = fmap.shape[-1]
        H_W = fmap.shape[1] * fmap.shape[2]

        # ==============================================================================
        # [SAFE & CORRECT] Per-Image Spatial Centering (Data Leakage 0%)
        # ==============================================================================
        # 1. fmap_reshaped: 현재 배치의 형태를 (B, H_W, C)로 변환합니다.
        fmap_reshaped = fmap.view(curr_bs, H_W, C)

        # 2. image_means: 오직 '자기 자신 1장의 이미지' 내부에 있는 공간 토큰들(dim=1)의 평균만 구합니다.
        #    배치 차원(dim=0)에 대해서 평균을 구하지 않으므로, 옆에 있는 다른 이미지의 정보나
        #    클래스 라벨 정보가 절대로 섞여 들어가지 않습니다. (Data Leakage 원천 차단)
        image_means = fmap_reshaped.mean(dim=1, keepdim=True)  # (B, 1, C)

        # 3. 평균 빼기: 각 이미지에서 '자신의 평균'만을 뺍니다.
        #    이렇게 하면 데이터셋의 라벨이나 배치 구성(shuffle 여부)에 완벽하게 독립적인,
        #    가장 정석적이고 안전한 정규화(Normalization)가 수행됩니다.
        fmap_reshaped = fmap_reshaped - image_means

        flat_tokens = fmap_reshaped.view(-1, C)

        # Save per-token L2 norms before normalization
        token_l2_norms = flat_tokens.norm(dim=1, keepdim=True).clamp_min(1e-12)

        flat_tokens = F.normalize(
            flat_tokens, dim=1, eps=1e-12
        )  ## SAE 넣어줄때 토큰 L2 정규화해서 넣어준다. 그렇게 학습했으니까.

        # SAE forward in chunks
        token_batch_size = 8192
        acts_chunks = []
        for s in range(0, flat_tokens.size(0), token_batch_size):
            chunk = flat_tokens[s : s + token_batch_size]
            with torch.amp.autocast(**autocast_kwargs):
                _, chunk_acts, _, _, _ = sae(chunk)
            acts_chunks.append(chunk_acts)
        acts = torch.cat(acts_chunks, dim=0)

        # Optionally restore per-token L2 norms
        if restore_token_norm:
            acts = acts.float() * token_l2_norms
        else:
            acts = acts.float()

        # Pool → image-level GAP
        acts = acts.view(curr_bs, H_W, sae.d_sae)
        pooled = acts.mean(dim=1)  # (B, d_sae)
        # pooled = F.normalize(pooled, dim=1)  # 이걸 왜 하지. 이미지별로 L2 norm을 할 이유가 없어. SAE 벡터인데 이걸 L2 norm을 왜 해. 물론 DPT할때는 아마 크게 영향은 안줬을꺼야. 그래도 모르긴 한다. 이거 없이 다시 뽑아야해.

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
# Data Loading (val + test, balanced) — copied from dpt_kendall.py
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
        if getattr(args, "test_only", False):
            csv_paths.append(os.path.join(args.save_dir, "test_split.csv"))
        else:
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

        rel_to_refidx = {}
        for uid, ridx in uid_to_refidx.items():
            rel_key = uid_to_relative(uid)
            rel_to_refidx[rel_key] = ridx

        refidx_list = []
        n_missing = 0
        for uid in all_uids:
            rel_key = uid_to_relative(uid)
            if rel_key in rel_to_refidx:
                refidx_list.append(rel_to_refidx[rel_key])
            else:
                n_missing += 1

        if n_missing > 0:
            logger.warning(
                f"  {n_missing}/{len(all_uids)} UIDs not matched (path mismatch?)"
            )
        logger.info(f"  Matched: {len(refidx_list)}/{len(all_uids)} UIDs from CSVs")

        # [CRITICAL RESTORE] CSV 파일에 없는 OOD 클래스 (Label 4 이상, 예: Alpha-Synuclein) 무조건 포함!
        n_ood = 0
        for rel_key, ridx in rel_to_refidx.items():
            if int(refs[ridx].label) >= 4:
                refidx_list.append(ridx)
                n_ood += 1
        logger.info(f"  Appended {n_ood} OOD images (label >= 4) not found in CSVs.")

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
        collate_fn=collate_skip_none,
    )

    logger.info(
        f"  Total: {len(selected)} images (Sequential Sequential Loading, shuffle=False)"
    )
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

    # If --use_all_data and user didn't explicitly set samples_per_class,
    # set it to 0 (= no sampling)
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

    # ── 4. Extract features (ALL neurons) ────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(
        f"Extracting SAE GAP features — ALL {d_sae} neurons "
        f"(restore_token_norm={args.restore_token_norm})"
    )

    X_all, y, lines, uids = extract_all_sae_features(
        encoder,
        sae,
        loader,
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
