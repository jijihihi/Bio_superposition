# ==============================================================================
# Token L2 Norm Distribution Analysis
# 각 레이어(stage5_out, refine_out)에서 나온 feature map token들의 L2 norm 분포를
# 히스토그램으로 시각화하는 코드
# ==============================================================================

## ===============================
# 결과. L2 norm이 감소하는 지수 형태로 나옴. 
# 유의미한 L2 norm과 아닌 L2 norm을 딱 자를수가 없어. 그래서 어렵다. 그래서 loss에 L2 norm을 곱한 형태로 loss를 줄 생각이다
# 이때 클러스터링의 경우 L2 norm에 비례해서 뽑아서 유의미한 것들에 대해서 가중치를 줄 생각
#====================================



import os
import random
from typing import List, Tuple, Dict

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from sae_project.step01_configs import get_args, resolve_paths
from sae_project.step02_logging_utils import get_logger, OUT_DIM, SUPERCLASS_MAP
from sae_project.step03_data_shards import load_all_sample_refs, build_uid_to_refidx
from sae_project.step04_data_bank import (
    InMemoryTarBank, InMemorySixteenBitDataset, load_split_csv,
    seed_worker, collate_skip_none
)
from sae_project.step05_model_encoder import SupMoCoModel, parse_int_list
from sae_project.step12_build_token_cache import pick_balanced_images

logger = get_logger("token_l2norm_hist")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def estimate_max_images_for_l4_gpu(feature_map_size: int = 64, channels: int = 512) -> int:
    """
    L4 GPU (24GB VRAM) 기준 최대 처리 가능한 이미지 수 계산
    
    메모리 구성:
    - feature map: (B, C, H, W) = (B, 512, 64, 64) in float16
    - 모델 파라미터 (약 ~50MB for encoder)
    - 추가 버퍼
    
    계산:
    - 이미지당 feature map: 512 * 64 * 64 * 2 bytes (float16) = 4MB
    - L4 GPU 24GB, 안전 여유 두고 20GB 사용 가정
    - 배치당 최대: 20GB / 4MB ≈ 5000 이미지 (이론적 최대)
    - 실제로는 모델 + 버퍼 고려해서 약 3000-4000 이미지 권장
    """
    bytes_per_image = channels * feature_map_size * feature_map_size * 2  # float16 = 2 bytes
    available_vram_gb = 20  # L4 24GB 중 안전하게 20GB
    available_vram_bytes = available_vram_gb * 1024 * 1024 * 1024
    
    # 배치 처리시 추가 메모리 오버헤드 (x2 안전계수)
    max_images = int(available_vram_bytes / (bytes_per_image * 2))
    
    # 실제 권장값
    recommended = min(max_images, 3000)  # 3000개 이상은 RAM에서 preload 병목
    
    logger.info(f"[Memory Estimate] L4 GPU")
    logger.info(f"  Feature map size: {feature_map_size}x{feature_map_size}x{channels}")
    logger.info(f"  Bytes per image: {bytes_per_image / 1024 / 1024:.2f} MB")
    logger.info(f"  Theoretical max: ~{max_images} images")
    logger.info(f"  Recommended: {recommended} images")
    
    return recommended


@torch.no_grad()
def extract_tokens_raw(
    encoder,
    x: torch.Tensor,
    which_layer: str,
) -> Tuple[torch.Tensor, int, int, int]:
    """
    L2 norm 전처리 없이 raw token 추출
    returns tokens (B*HW, C=512) and (B,H,W)
    """
    fmap = encoder.forward_feature_maps(x, which=which_layer)  # (B,C,H,W)
    B, C, H, W = fmap.shape
    fmap = fmap.permute(0, 2, 3, 1).contiguous()              # (B,H,W,C)
    tokens = fmap.view(B * H * W, C)
    return tokens, B, H, W


def collect_l2_norms(
    args,
    encoder,
    loader,
    which_layer: str,
    device: torch.device,
    max_images: int = None,
) -> np.ndarray:
    """
    데이터로더에서 이미지들을 가져와서 token L2 norm 수집
    """
    encoder.eval()
    autocast_kwargs = dict(device_type="cuda", enabled=torch.cuda.is_available(), dtype=torch.bfloat16)
    
    all_norms = []
    img_count = 0
    
    pbar = tqdm(loader, desc=f"[L2 norm] {which_layer}", leave=True)
    for batch in pbar:
        if batch is None:
            continue
        x_cpu, y_cpu, plate, line, uid = batch
        if y_cpu.numel() < 1:
            continue
        
        B = x_cpu.size(0)
        x = x_cpu.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        
        with torch.amp.autocast(**autocast_kwargs):
            tokens, B2, H, W = extract_tokens_raw(encoder, x, which_layer)
        
        # L2 norm 계산 (per token)
        l2_norms = tokens.norm(dim=1)  # (B*H*W,)
        all_norms.append(l2_norms.float().cpu().numpy())  # bfloat16 -> float32 for numpy
        
        img_count += B
        pbar.set_postfix({"images": img_count, "tokens": img_count * H * W})
        
        if max_images is not None and img_count >= max_images:
            break
    
    return np.concatenate(all_norms, axis=0)


def plot_l2_histogram(
    norms: np.ndarray,
    which_layer: str,
    save_path: str,
    bins: int = 100,
):
    """
    L2 norm 분포 히스토그램 플로팅
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 히스토그램 플롯
    ax.hist(norms, bins=bins, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # 통계 정보 추가
    mean_val = np.mean(norms)
    std_val = np.std(norms)
    median_val = np.median(norms)
    min_val = np.min(norms)
    max_val = np.max(norms)
    
    stats_text = (
        f"Layer: {which_layer}\n"
        f"Total tokens: {len(norms):,}\n"
        f"Mean: {mean_val:.4f}\n"
        f"Std: {std_val:.4f}\n"
        f"Median: {median_val:.4f}\n"
        f"Min: {min_val:.4f}\n"
        f"Max: {max_val:.4f}"
    )
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('L2 Norm', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Token L2 Norm Distribution ({which_layer})', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"[{which_layer}] Saved histogram to: {save_path}")
    logger.info(f"[{which_layer}] Stats - mean: {mean_val:.4f}, std: {std_val:.4f}, "
                f"median: {median_val:.4f}, min: {min_val:.4f}, max: {max_val:.4f}")


def plot_combined_histogram(
    norms_stage5: np.ndarray,
    norms_refine: np.ndarray,
    save_path: str,
    bins: int = 100,
):
    """
    두 레이어의 L2 norm 분포를 하나의 그래프에서 비교
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Stage5 히스토그램
    ax1.hist(norms_stage5, bins=bins, color='steelblue', alpha=0.7, 
             edgecolor='black', linewidth=0.5, label='stage5_out')
    ax1.set_xlabel('L2 Norm', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Token L2 Norm Distribution (stage5_out)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    mean_s5 = np.mean(norms_stage5)
    std_s5 = np.std(norms_stage5)
    stats_s5 = f"Mean: {mean_s5:.4f}\nStd: {std_s5:.4f}\nN: {len(norms_stage5):,}"
    ax1.text(0.95, 0.95, stats_s5, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Refine 히스토그램
    ax2.hist(norms_refine, bins=bins, color='coral', alpha=0.7,
             edgecolor='black', linewidth=0.5, label='refine_out')
    ax2.set_xlabel('L2 Norm', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Token L2 Norm Distribution (refine_out)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    mean_rf = np.mean(norms_refine)
    std_rf = np.std(norms_refine)
    stats_rf = f"Mean: {mean_rf:.4f}\nStd: {std_rf:.4f}\nN: {len(norms_refine):,}"
    ax2.text(0.95, 0.95, stats_rf, transform=ax2.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved combined histogram to: {save_path}")


def main():
    args = resolve_paths(get_args())
    set_seed(args.seed)
    
    # ========== 설정값 ==========
    # L4 GPU 기준 최대 이미지 수 계산 (64x64 feature map, 512 channels)
    # - 이미지당 feature map: 512 * 64 * 64 * 2 bytes (float16) = 4MB
    # - L4 GPU 24GB에서 안전하게 약 2500-3000 이미지 처리 가능
    # - RAM preload 고려하면 약 2000-2500 이미지가 적절
    MAX_IMAGES = estimate_max_images_for_l4_gpu(feature_map_size=64, channels=512)
    HIST_BINS = 100   # 히스토그램 빈 개수
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[L2 norm analysis] device={device}")
    
    # Load refs
    refs = load_all_sample_refs(args.shard_root)
    uid_to_refidx = build_uid_to_refidx(refs)
    
    # Load encoder
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
    sd = torch.load(args.model_state_path, map_location="cpu")
    from sae_project.step05_model_encoder import robust_load_state_dict
    robust_load_state_dict(model, sd, strict=True)
    encoder = model.encoder.to(device).eval().to(memory_format=torch.channels_last)
    logger.info(f"Loaded encoder from: {args.model_state_path}")
    
    # Load train split
    train_csv = os.path.join(args.save_dir, "train_split.csv")
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Train split not found: {train_csv}")
    
    uids_all = load_split_csv(train_csv)
    
    # ========== 클래스/라인/플레이트별 균형 샘플링 ==========
    # refs_by_uid 구성: uid -> SampleRef 객체
    # pick_balanced_images는 (superclass, line, plate) 기준으로 균등하게 이미지 선택
    valid_uids = [u for u in uids_all if u in uid_to_refidx]
    refs_by_uid_obj = {u: refs[uid_to_refidx[u]] for u in valid_uids}
    
    # 균형 잡힌 이미지 선택 (superclass, line, plate 별로 동일 개수)
    picked_uids = pick_balanced_images(
        uids=valid_uids,
        refs_by_uid=refs_by_uid_obj,
        images_target=min(MAX_IMAGES, len(valid_uids)),
        seed=args.seed
    )
    
    # 선택된 이미지들의 분포 로깅
    from collections import Counter
    class_dist = Counter()
    line_dist = Counter()
    plate_dist = Counter()
    for u in picked_uids:
        r = refs_by_uid_obj[u]
        sup = getattr(r, "superclass", SUPERCLASS_MAP.get(getattr(r, "line", "UNK"), "UNK"))
        class_dist[sup] += 1
        line_dist[getattr(r, "line", "UNK")] += 1
        plate_dist[(sup, getattr(r, "line", "UNK"), getattr(r, "plate", "UNK"))] += 1
    
    logger.info(f"\n=== Balanced Sampling Results ===")
    logger.info(f"Total images selected: {len(picked_uids)}")
    logger.info(f"Class distribution: {dict(class_dist)}")
    logger.info(f"Line distribution: {dict(line_dist)}")
    logger.info(f"Unique (class, line, plate) groups: {len(plate_dist)}")
    
    # refidx 변환
    refidx = [uid_to_refidx[u] for u in picked_uids]
    
    logger.info(f"Using {len(refidx)} balanced images for L2 norm analysis")
    
    # Build dataset/loader
    # InMemorySixteenBitDataset은 내부적으로 instance normalization 수행
    # 이는 LRRK2 모델의 validation과 동일한 전처리
    bank = InMemoryTarBank(refs, refidx, args.img_size)
    ib = list(range(len(refidx)))
    ds = InMemorySixteenBitDataset(bank, ib, args.img_size, augment=False)
    
    pin = torch.cuda.is_available()
    loader = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(getattr(args, "num_workers", 0)),
        pin_memory=pin,
        worker_init_fn=seed_worker,
        collate_fn=collate_skip_none,
        drop_last=False,
    )
    
    # Output directory
    output_dir = os.path.join(args.save_dir, "l2_norm_analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    # ========== Feature map 정보 출력 ==========
    logger.info(f"\n=== Feature Map Info ===")
    logger.info(f"Image size: {args.img_size}x{args.img_size}")
    logger.info(f"Feature map size: 64x64 (img_size/2 due to stride=2 stem)")
    logger.info(f"Tokens per image: 64 * 64 = 4096")
    logger.info(f"Total tokens to analyze: {len(refidx) * 4096:,}")
    
    # Collect L2 norms for both layers
    logger.info("\nCollecting L2 norms for stage5_out...")
    norms_stage5 = collect_l2_norms(args, encoder, loader, "stage5_out", device, MAX_IMAGES)
    
    # Need to recreate loader for second pass (iterator exhausted)
    loader = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(getattr(args, "num_workers", 0)),
        pin_memory=pin,
        worker_init_fn=seed_worker,
        collate_fn=collate_skip_none,
        drop_last=False,
    )
    
    logger.info("Collecting L2 norms for refine_out...")
    norms_refine = collect_l2_norms(args, encoder, loader, "refine_out", device, MAX_IMAGES)
    
    # Plot individual histograms
    plot_l2_histogram(norms_stage5, "stage5_out", 
                     os.path.join(output_dir, "l2_hist_stage5_out.png"), bins=HIST_BINS)
    plot_l2_histogram(norms_refine, "refine_out",
                     os.path.join(output_dir, "l2_hist_refine_out.png"), bins=HIST_BINS)
    
    # Plot combined comparison
    plot_combined_histogram(norms_stage5, norms_refine,
                           os.path.join(output_dir, "l2_hist_combined.png"), bins=HIST_BINS)
    
    # Save raw data for further analysis
    np.save(os.path.join(output_dir, "l2_norms_stage5_out.npy"), norms_stage5)
    np.save(os.path.join(output_dir, "l2_norms_refine_out.npy"), norms_refine)
    logger.info(f"Saved raw L2 norm arrays to: {output_dir}")
    
    # Print percentile info (useful for filtering)
    for layer_name, norms in [("stage5_out", norms_stage5), ("refine_out", norms_refine)]:
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        pvals = np.percentile(norms, percentiles)
        logger.info(f"\n[{layer_name}] Percentiles:")
        for p, v in zip(percentiles, pvals):
            logger.info(f"  {p}th: {v:.4f}")
    
    logger.info(f"\n✅ Done. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
