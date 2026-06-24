#!/usr/bin/env python3
"""
GAP Feature Map 분석 스크립트
- 각 GAP 채널의 상대 표준편차 (CV = std/mean) 계산
- 클래스별 GAP 평균값 분석
- Dead feature map 탐지

Usage:
    python step_gap_analysis.py --ckpt_path /path/to/best_model.pt --shard_root /path/to/wds_shards
"""

import argparse
import csv
import io
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# =============================================================================
# Encoder 정의 (학습 코드와 동일)
# =============================================================================
OUT_DIM = 512


def conv2d(in_ch, out_ch, k=3, stride=1, padding=1, dilation=1, bias=True):
    return nn.Conv2d(
        in_ch,
        out_ch,
        kernel_size=k,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
    )


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1):
        super().__init__()
        self.c1 = conv2d(
            in_ch, out_ch, 3, 1, padding=dilation, dilation=dilation, bias=True
        )
        self.c2 = conv2d(
            out_ch, out_ch, 3, 1, padding=dilation, dilation=dilation, bias=True
        )
        self.proj = None
        if in_ch != out_ch:
            self.proj = conv2d(in_ch, out_ch, 1, 1, padding=0, bias=False)

    def forward(self, x):
        identity = x
        x = F.relu(x, inplace=True)
        x = self.c1(x)
        x = F.relu(x, inplace=True)
        x = self.c2(x)
        if self.proj is not None:
            identity = self.proj(identity)
        return x + identity


class Stage(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        n_blocks,
        dilation,
        use_ckpt: bool = False,
        ckpt_segments: int = 1,
    ):
        super().__init__()
        blocks = [ResBlock(in_ch, out_ch, dilation=dilation)]
        for _ in range(n_blocks - 1):
            blocks.append(ResBlock(out_ch, out_ch, dilation=dilation))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class Encoder(nn.Module):
    def __init__(
        self,
        blocks=(2, 2, 4, 4),
        dilations=(1, 1, 1, 1),
        refine_blocks=1,
        ckpt_segments=2,
    ):
        super().__init__()
        b2, b3, b4, b5 = blocks
        d2, d3, d4, d5 = dilations

        self.stem = nn.Sequential(conv2d(3, 64, k=3, stride=2, padding=1, bias=True))
        self.stage2 = Stage(64, 128, b2, d2)
        self.stage3 = Stage(128, 256, b3, d3)
        self.stage4 = Stage(256, 512, b4, d4)
        self.stage5 = Stage(512, OUT_DIM, b5, d5)
        self.refine = Stage(OUT_DIM, OUT_DIM, int(refine_blocks), 1)

        self.trunk = nn.Sequential(
            self.stem, self.stage2, self.stage3, self.stage4, self.stage5, self.refine
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.trunk(x)
        x = self.gap(x).flatten(1)
        return x


# =============================================================================
# Safe Instance Normalize
# =============================================================================
class SafeInstanceNormalize:
    def __init__(self, threshold: float = 0.01):
        self.threshold = float(threshold)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = tensor.mean(dim=[1, 2], keepdim=True)
        std = tensor.std(dim=[1, 2], keepdim=True).clamp_min(self.threshold)
        return (tensor - mean) / std


import re
import tarfile
# =============================================================================
# 간단한 데이터셋 (tar shard에서 로드)
# =============================================================================
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class SampleRef:
    tar_path: str
    prefix: str
    tif_off: int
    tif_size: int
    label: int
    superclass: str
    line: str
    plate: str


SUPERCLASS_TO_LABEL = {"Control": 0, "SNCA": 1, "GBA": 2, "LRRK2": 3}
LABEL_TO_SUPERCLASS = {v: k for k, v in SUPERCLASS_TO_LABEL.items()}


def scan_shards(shard_root: str, max_samples: int = 999999) -> List[SampleRef]:
    """스캔 후 클래스별 균등 샘플링"""
    import random

    refs_by_class = {0: [], 1: [], 2: [], 3: []}  # Control, SNCA, GBA, LRRK2

    for line_dir in sorted(os.listdir(shard_root)):
        line_path = os.path.join(shard_root, line_dir)
        if not os.path.isdir(line_path):
            continue

        if line_dir.startswith("Control"):
            superclass = "Control"
            line_name = line_dir
        else:
            superclass = line_dir
            line_name = line_dir

        label = SUPERCLASS_TO_LABEL.get(superclass, -1)
        if label < 0:
            continue

        for plate_dir in sorted(os.listdir(line_path)):
            m = re.match(r"plate=(\d+)", plate_dir)
            if not m:
                continue
            plate = m.group(1)
            plate_path = os.path.join(line_path, plate_dir)

            for tar_name in sorted(os.listdir(plate_path)):
                if not tar_name.endswith(".tar"):
                    continue
                tar_path = os.path.join(plate_path, tar_name)

                try:
                    with tarfile.open(tar_path, "r") as tf:
                        members = tf.getmembers()
                        prefixes = set()
                        for mem in members:
                            if mem.name.endswith(".tif"):
                                prefix = mem.name.replace(".tif", "")
                                prefixes.add(prefix)

                        for prefix in prefixes:
                            tif_name = prefix + ".tif"
                            for mem in members:
                                if mem.name == tif_name:
                                    refs_by_class[label].append(
                                        SampleRef(
                                            tar_path=tar_path,
                                            prefix=prefix,
                                            tif_off=mem.offset_data,
                                            tif_size=mem.size,
                                            label=label,
                                            superclass=superclass,
                                            line=line_name,
                                            plate=plate,
                                        )
                                    )
                                    break
                except Exception as e:
                    print(f"Error scanning {tar_path}: {e}")
                    continue

    # 클래스별 균등 샘플링
    samples_per_class = max_samples // 4
    final_refs = []

    for label in range(4):
        class_refs = refs_by_class[label]
        print(f"   {LABEL_TO_SUPERCLASS[label]}: {len(class_refs)} found")
        if len(class_refs) > samples_per_class:
            random.shuffle(class_refs)
            class_refs = class_refs[:samples_per_class]
        final_refs.extend(class_refs)

    random.shuffle(final_refs)  # 최종 셔플
    return final_refs


class SimpleDataset(Dataset):
    def __init__(self, refs: List[SampleRef], img_size: int = 128):
        self.refs = refs
        self.img_size = img_size
        self.normalize = SafeInstanceNormalize(threshold=0.01)

    def __len__(self):
        return len(self.refs)

    def __getitem__(self, idx):
        r = self.refs[idx]
        try:
            with open(r.tar_path, "rb") as f:
                f.seek(r.tif_off)
                data = f.read(r.tif_size)
            img = tifffile.imread(io.BytesIO(data))

            if img.dtype != np.uint16:
                return None
            if img.ndim != 3 or img.shape[2] != 3:
                return None

            x = img.astype(np.float32) / 65535.0
            x = torch.from_numpy(x).permute(2, 0, 1)
            x = self.normalize(x)

            return x, r.label, r.superclass
        except:
            return None


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    xs, ys, supercls = zip(*batch)
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long), list(supercls)


# =============================================================================
# 메인 분석 함수
# =============================================================================
def get_args():
    p = argparse.ArgumentParser("GAP Feature Map Analysis")
    p.add_argument("--ckpt_path", type=str, required=True, help="Path to best_model.pt")
    p.add_argument("--shard_root", type=str, default="/content/wds_shards")
    p.add_argument("--output_dir", type=str, default="./gap_analysis")
    p.add_argument("--max_samples", type=int, default=20000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--blocks", type=str, default="2,2,2,3")
    p.add_argument("--dilations", type=str, default="1,1,1,1")
    p.add_argument("--refine_blocks", type=int, default=1)
    return p.parse_args()


def parse_int_list(s: str, n: int):
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return tuple(int(p) for p in parts)


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("🔬 GAP Feature Map Analysis")
    print("=" * 70)

    # 1. 모델 로드
    print("\n[1] Loading encoder...")
    encoder = Encoder(
        blocks=parse_int_list(args.blocks, 4),
        dilations=parse_int_list(args.dilations, 4),
        refine_blocks=args.refine_blocks,
    )

    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    if "encoder" in ckpt:
        encoder.load_state_dict(ckpt["encoder"], strict=False)
    else:
        # SupConMoCoModel 전체 체크포인트인 경우
        enc_sd = {
            k.replace("encoder.", ""): v
            for k, v in ckpt.items()
            if k.startswith("encoder.")
        }
        if enc_sd:
            encoder.load_state_dict(enc_sd, strict=False)
        else:
            encoder.load_state_dict(ckpt, strict=False)

    encoder.eval().to(device).to(memory_format=torch.channels_last)
    print(f"   Loaded from: {args.ckpt_path}")

    # 2. 데이터 로드
    print("\n[2] Scanning shards...")
    refs = scan_shards(args.shard_root, max_samples=args.max_samples)
    print(f"   Found {len(refs)} samples")

    dataset = SimpleDataset(refs)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # 3. GAP 추출
    print("\n[3] Extracting GAP features...")
    all_gaps = []
    all_labels = []
    all_superclasses = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting"):
            if batch is None:
                continue
            x, y, supercls = batch
            x = x.to(device).contiguous(memory_format=torch.channels_last)

            gap = encoder(x)  # (B, 512)
            gap = F.normalize(gap, dim=1)  # L2 정규화 (학습 시와 동일)

            all_gaps.append(gap.cpu())
            all_labels.append(y)
            all_superclasses.extend(supercls)

    all_gaps = torch.cat(all_gaps, dim=0)  # (N, 512)
    all_labels = torch.cat(all_labels, dim=0)  # (N,)

    print(f"   Total samples: {all_gaps.shape[0]}")

    # 4. 전체 통계 계산
    print("\n[4] Computing statistics...")

    # 채널별 통계
    mean_per_channel = all_gaps.mean(dim=0)  # (512,)
    std_per_channel = all_gaps.std(dim=0)  # (512,)

    # 상대 표준편차 (CV = std / |mean|)
    # mean이 0에 가까울 때를 위해 eps 추가
    cv_per_channel = std_per_channel / (mean_per_channel.abs() + 1e-8)  # (512,)

    # 5. 클래스별 GAP 평균
    print("\n[5] Computing per-class GAP means...")
    class_means = {}
    for label in range(4):
        mask = all_labels == label
        if mask.sum() > 0:
            class_mean = all_gaps[mask].mean(dim=0)  # (512,)
            class_means[LABEL_TO_SUPERCLASS[label]] = class_mean

    # 6. 결과 저장 - CSV
    print("\n[6] Saving results...")

    # 6-1. 채널별 통계 CSV
    stats_csv_path = os.path.join(args.output_dir, "channel_statistics.csv")
    with open(stats_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["channel", "mean", "std", "CV (std/|mean|)"] + list(
            class_means.keys()
        )
        writer.writerow(header)

        for ch in range(OUT_DIM):
            row = [
                ch,
                f"{mean_per_channel[ch].item():.6f}",
                f"{std_per_channel[ch].item():.6f}",
                f"{cv_per_channel[ch].item():.4f}",
            ]
            for cls_name in class_means.keys():
                row.append(f"{class_means[cls_name][ch].item():.6f}")
            writer.writerow(row)

    print(f"   Saved: {stats_csv_path}")

    # 6-2. 요약 통계
    dead_threshold = 0.01  # std < 0.01인 채널을 "죽은" 것으로 간주
    dead_count = (std_per_channel < dead_threshold).sum().item()

    low_cv_threshold = 0.1  # CV < 0.1인 채널도 문제일 수 있음
    low_cv_count = (cv_per_channel < low_cv_threshold).sum().item()

    summary_path = os.path.join(args.output_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("GAP Feature Map Analysis Summary\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Total samples analyzed: {all_gaps.shape[0]}\n")
        f.write(f"Total channels: {OUT_DIM}\n\n")

        f.write("--- Dead Channel Analysis ---\n")
        f.write(
            f"Dead channels (std < {dead_threshold}): {dead_count}/{OUT_DIM} ({dead_count/OUT_DIM*100:.1f}%)\n"
        )
        f.write(
            f"Low CV channels (CV < {low_cv_threshold}): {low_cv_count}/{OUT_DIM} ({low_cv_count/OUT_DIM*100:.1f}%)\n\n"
        )

        f.write("--- Overall Statistics ---\n")
        f.write(f"Mean of channel means: {mean_per_channel.mean().item():.6f}\n")
        f.write(f"Mean of channel stds: {std_per_channel.mean().item():.6f}\n")
        f.write(f"Mean CV: {cv_per_channel.mean().item():.4f}\n")
        f.write(f"Median CV: {cv_per_channel.median().item():.4f}\n\n")

        f.write("--- Per-Class Sample Counts ---\n")
        for label in range(4):
            count = (all_labels == label).sum().item()
            f.write(f"{LABEL_TO_SUPERCLASS[label]}: {count}\n")

    print(f"   Saved: {summary_path}")

    # 7. 시각화
    print("\n[7] Creating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 7-1. 채널별 표준편차
    ax = axes[0, 0]
    ax.bar(range(OUT_DIM), std_per_channel.numpy(), width=1.0)
    ax.axhline(
        dead_threshold,
        color="r",
        linestyle="--",
        label=f"Dead threshold ({dead_threshold})",
    )
    ax.set_xlabel("Channel Index")
    ax.set_ylabel("Std")
    ax.set_title(f"Channel Std (Dead: {dead_count}/{OUT_DIM})")
    ax.legend()

    # 7-2. CV 분포 히스토그램
    ax = axes[0, 1]
    ax.hist(cv_per_channel.numpy(), bins=50, edgecolor="black")
    ax.axvline(
        low_cv_threshold,
        color="r",
        linestyle="--",
        label=f"Low CV threshold ({low_cv_threshold})",
    )
    ax.set_xlabel("Coefficient of Variation (CV)")
    ax.set_ylabel("Count")
    ax.set_title("CV Distribution")
    ax.legend()

    # 7-3. 클래스별 GAP 평균 히트맵 (존재하는 클래스만)
    ax = axes[1, 0]
    available_classes = [
        cls for cls in ["Control", "SNCA", "GBA", "LRRK2"] if cls in class_means
    ]
    if len(available_classes) > 0:
        class_mean_matrix = torch.stack(
            [class_means[cls] for cls in available_classes], dim=0
        )
        im = ax.imshow(
            class_mean_matrix.numpy(), aspect="auto", cmap="RdBu_r", vmin=-0.2, vmax=0.2
        )
        ax.set_yticks(range(len(available_classes)))
        ax.set_yticklabels(available_classes)
        ax.set_xlabel("Channel Index")
        ax.set_title("Per-Class GAP Mean")
        plt.colorbar(im, ax=ax)
    else:
        ax.text(0.5, 0.5, "No class data", ha="center", va="center")
        ax.set_title("Per-Class GAP Mean")

    # 7-4. 클래스 간 차이 (각 클래스 - Control, Control이 있는 경우만)
    ax = axes[1, 1]
    if "Control" in class_means and len(available_classes) > 1:
        other_classes = [cls for cls in available_classes if cls != "Control"]
        diff_list = [class_means[cls] - class_means["Control"] for cls in other_classes]
        diff_matrix = torch.stack(diff_list, dim=0)
        im = ax.imshow(
            diff_matrix.numpy(), aspect="auto", cmap="RdBu_r", vmin=-0.1, vmax=0.1
        )
        ax.set_yticks(range(len(other_classes)))
        ax.set_yticklabels([f"{cls} - Control" for cls in other_classes])
        ax.set_xlabel("Channel Index")
        ax.set_title("Class Difference from Control")
        plt.colorbar(im, ax=ax)
    else:
        ax.text(0.5, 0.5, "No Control or only 1 class", ha="center", va="center")
        ax.set_title("Class Difference from Control")

    plt.tight_layout()
    fig_path = os.path.join(args.output_dir, "gap_analysis.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()

    print(f"   Saved: {fig_path}")

    # 8. 최종 요약 출력
    print("\n" + "=" * 70)
    print("📊 Analysis Complete!")
    print("=" * 70)
    print(
        f"\n   Dead channels (std < {dead_threshold}): {dead_count}/{OUT_DIM} ({dead_count/OUT_DIM*100:.1f}%)"
    )
    print(
        f"   Low CV channels (CV < {low_cv_threshold}): {low_cv_count}/{OUT_DIM} ({low_cv_count/OUT_DIM*100:.1f}%)"
    )
    print(f"   Mean CV: {cv_per_channel.mean().item():.4f}")
    print(f"\n   Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
