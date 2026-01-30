# ==============================================================================
# Perturbation Test: Evaluate CNN's Spatial Dependency via Grid Shuffling
# ==============================================================================
# A100 GPU 최적화 버전
# - GPU에서 batch-wise shuffle 수행
# - Mixed precision (BF16) 기본 활성화
# - 병렬 trial 처리로 GPU 활용 극대화
# - channels_last 메모리 포맷
#
# Colab에서 실행:
#   from perturbation_test import main, get_args
#   args = get_args()
#   results = main()
# ==============================================================================

import os
import io
import re
import sys
import glob
import json
import time
import random
import pickle
import logging
import argparse
import csv
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.checkpoint import checkpoint_sequential
from torchvision import transforms
from tqdm.auto import tqdm

try:
    import tifffile
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "tifffile"])
    import tifffile

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ==============================================================================
# Logging
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("PerturbationTest")

# ==============================================================================
# Constants
# ==============================================================================
DEFAULT_SHARD_ROOT = "/content/wds_shards"

LINE_FOLDERS = [
    "Control_C4", "Control_C18", "Control_C19",
    "SNCA", "GBA", "LRRK2"
]

SUPERCLASS_MAP = {
    "Control_C4":  "Control",
    "Control_C18": "Control",
    "Control_C19": "Control",
    "SNCA":        "SNCA",
    "GBA":         "GBA",
    "LRRK2":       "LRRK2",
}
CLASS_TO_LABEL = {"Control": 0, "SNCA": 1, "GBA": 2, "LRRK2": 3}
LABEL_TO_CLASS = {v: k for k, v in CLASS_TO_LABEL.items()}

PLATE_DIR_RE = re.compile(r"plate=(\d{6})")

OUT_DIM = 512


# ==============================================================================
# 1) Reproducibility & GPU Optimization
# ==============================================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True  # A100 최적화
    torch.backends.cuda.matmul.allow_tf32 = True  # A100 TF32 활성화
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)


# ==============================================================================
# 2) Tar Index & SampleRef
# ==============================================================================
@dataclass(frozen=True)
class SampleRef:
    tar_path: str
    prefix: str
    tif_off: int
    tif_size: int
    js_off: int
    js_size: int
    line: str
    superclass: str
    label: int
    plate: str


def _infer_line_and_plate_from_tarpath(tar_path: str) -> Tuple[str, str]:
    parts = tar_path.replace("\\", "/").split("/")
    line = parts[-3]
    m = PLATE_DIR_RE.search(parts[-2])
    plate = m.group(1) if m else "UNKNOWN"
    return line, plate


def build_tar_index_if_needed(tar_path: str):
    idx_path = tar_path + ".pkl"
    if os.path.exists(idx_path):
        return

    t0 = time.time()
    items = {}

    import tarfile
    with tarfile.open(tar_path, "r") as tf:
        for m in tf.getmembers():
            if not m.isreg():
                continue
            name = m.name
            if name.endswith(".tif"):
                pref = name[:-4]
                it = items.get(pref, {})
                it["tif_off"] = m.offset_data
                it["tif_size"] = m.size
                items[pref] = it
            elif name.endswith(".json"):
                pref = name[:-5]
                it = items.get(pref, {})
                it["js_off"] = m.offset_data
                it["js_size"] = m.size
                items[pref] = it

    pairs = []
    for pref, it in items.items():
        if "tif_off" in it and "js_off" in it:
            pairs.append((pref, it["tif_off"], it["tif_size"], it["js_off"], it["js_size"]))

    with open(idx_path, "wb") as f:
        pickle.dump(pairs, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info(f"[tar-index] built {len(pairs)} pairs: {os.path.basename(tar_path)} ({time.time()-t0:.1f}s)")


def load_all_sample_refs(shard_root: str) -> List[SampleRef]:
    tar_paths = sorted(glob.glob(os.path.join(shard_root, "*", "plate=*", "*.tar")))
    if len(tar_paths) == 0:
        raise FileNotFoundError(f"No tar shards found under: {shard_root}")

    for tp in tar_paths:
        build_tar_index_if_needed(tp)

    refs: List[SampleRef] = []
    for tp in tar_paths:
        line, plate = _infer_line_and_plate_from_tarpath(tp)
        superclass = SUPERCLASS_MAP.get(line, line)
        label = CLASS_TO_LABEL[superclass]

        with open(tp + ".pkl", "rb") as f:
            pairs = pickle.load(f)

        for pref, tif_off, tif_size, js_off, js_size in pairs:
            refs.append(SampleRef(
                tar_path=tp,
                prefix=pref,
                tif_off=int(tif_off),
                tif_size=int(tif_size),
                js_off=int(js_off),
                js_size=int(js_size),
                line=line,
                superclass=superclass,
                label=label,
                plate=plate
            ))

    logger.info(f"Loaded sample refs: {len(refs)}")
    return refs


# ==============================================================================
# 3) Dataset (CPU에서는 정규화만, shuffle은 GPU에서)
# ==============================================================================
def validate_uint16_rgb_128(img: np.ndarray, img_size: int):
    if img is None:
        raise ValueError("decoded None")
    if img.dtype != np.uint16:
        raise ValueError(f"dtype must be uint16, got {img.dtype}")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"shape must be HxWx3, got {img.shape}")
    h, w = img.shape[:2]
    if (h, w) != (img_size, img_size):
        raise ValueError(f"size must be {(img_size, img_size)}, got {(h, w)}")


class SafeInstanceNormalize:
    """Instance normalization with std flooring"""
    def __init__(self, threshold: float = 0.01):
        self.threshold = float(threshold)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # tensor: (C, H, W) or (B, C, H, W)
        if tensor.dim() == 3:
            mean = tensor.mean(dim=[1, 2], keepdim=True)
            std = tensor.std(dim=[1, 2], keepdim=True).clamp_min(self.threshold)
        else:  # (B, C, H, W)
            mean = tensor.mean(dim=[2, 3], keepdim=True)
            std = tensor.std(dim=[2, 3], keepdim=True).clamp_min(self.threshold)
        return (tensor - mean) / std


class InMemoryTarBank:
    """Shared RAM bank for images"""
    def __init__(self, refs: List[SampleRef], ref_indices: List[int], img_size: int):
        self.refs = refs
        self.ref_indices = ref_indices
        self.img_size = int(img_size)

        self.images: List[Optional[np.ndarray]] = [None] * len(ref_indices)
        self.labels: List[int] = [0] * len(ref_indices)
        self.plates: List[str] = [""] * len(ref_indices)
        self.lines: List[str] = [""] * len(ref_indices)
        self.uids: List[str] = [""] * len(ref_indices)

        logger.info(f"⚡ Preloading {len(ref_indices)} images into RAM from tar shards...")

        tar_to_fh = {}

        def read_bytes(tp: str, off: int, size: int) -> bytes:
            fh = tar_to_fh.get(tp, None)
            if fh is None:
                fh = open(tp, "rb", buffering=0)
                tar_to_fh[tp] = fh
            fh.seek(off)
            return fh.read(size)

        bad = 0
        t0 = time.time()
        for j, ridx in enumerate(tqdm(ref_indices, desc="preload", leave=True)):
            r = refs[ridx]
            try:
                tif_bytes = read_bytes(r.tar_path, r.tif_off, r.tif_size)
                img = tifffile.imread(io.BytesIO(tif_bytes))
                validate_uint16_rgb_128(img, self.img_size)

                self.images[j] = img
                self.labels[j] = int(r.label)
                self.plates[j] = r.plate
                self.lines[j] = r.line
                self.uids[j] = f"{r.tar_path}:{r.prefix}"
            except Exception:
                bad += 1
                self.images[j] = None

        for fh in tar_to_fh.values():
            try:
                fh.close()
            except:
                pass

        logger.info(f"Preload done. bad={bad}/{len(ref_indices)} elapsed={(time.time()-t0)/60:.1f} min")


class InMemorySixteenBitDataset(Dataset):
    """Dataset - CPU에서는 float 변환과 정규화만"""
    def __init__(
        self,
        bank: InMemoryTarBank,
        indices_in_bank: List[int],
        img_size: int
    ):
        self.bank = bank
        self.ib = indices_in_bank
        self.img_size = int(img_size)
        self.normalize = SafeInstanceNormalize(threshold=0.01)

    def __len__(self):
        return len(self.ib)

    def __getitem__(self, idx: int):
        j = self.ib[idx]
        img = self.bank.images[j]
        if img is None:
            return None

        y = torch.tensor(self.bank.labels[j], dtype=torch.long)
        uid = self.bank.uids[j]

        # 16-bit to float32 [0, 1]
        x = (img.astype(np.float32) / 65535.0)
        x = torch.from_numpy(x).permute(2, 0, 1)  # (C, H, W)

        # Instance normalize (CPU)
        x = self.normalize(x)

        return x, y, uid


# ==============================================================================
# 4) GPU-based Grid Shuffle (배치 단위로 GPU에서 처리)
# ==============================================================================
@torch.jit.script
def grid_shuffle_batch_gpu(images: torch.Tensor, grid_size: int) -> torch.Tensor:
    """
    GPU에서 배치 단위로 그리드 셔플 수행 (JIT 컴파일)
    
    Args:
        images: (B, C, H, W) GPU tensor
        grid_size: 그리드 크기 (N)
    
    Returns:
        셔플된 이미지 (B, C, H, W)
    """
    B, C, H, W = images.shape
    
    if grid_size <= 1:
        return images
    
    # 패치 크기 계산
    patch_h = H // grid_size
    patch_w = W // grid_size
    
    if patch_h < 1 or patch_w < 1:
        return images
    
    # 사용할 영역
    used_h = patch_h * grid_size
    used_w = patch_w * grid_size
    
    # 결과 이미지 (원본 복사)
    result = images.clone()
    
    # 그리드 영역 추출
    grid_region = images[:, :, :used_h, :used_w]  # (B, C, used_h, used_w)
    
    # 패치로 분할
    # (B, C, grid_size, patch_h, grid_size, patch_w)
    patches = grid_region.view(B, C, grid_size, patch_h, grid_size, patch_w)
    # (B, C, grid_size, grid_size, patch_h, patch_w)
    patches = patches.permute(0, 1, 2, 4, 3, 5)
    # (B, C, N*N, patch_h, patch_w)
    num_patches = grid_size * grid_size
    patches = patches.reshape(B, C, num_patches, patch_h, patch_w)
    
    # 각 이미지마다 다른 랜덤 셔플
    shuffled_patches = torch.empty_like(patches)
    for b in range(B):
        perm = torch.randperm(num_patches, device=images.device)
        shuffled_patches[b] = patches[b, :, perm, :, :]
    
    # 재조합
    # (B, C, grid_size, grid_size, patch_h, patch_w)
    shuffled_patches = shuffled_patches.view(B, C, grid_size, grid_size, patch_h, patch_w)
    # (B, C, grid_size, patch_h, grid_size, patch_w)
    shuffled_patches = shuffled_patches.permute(0, 1, 2, 4, 3, 5)
    # (B, C, used_h, used_w)
    shuffled_region = shuffled_patches.reshape(B, C, used_h, used_w)
    
    # 결과에 삽입
    result[:, :, :used_h, :used_w] = shuffled_region
    
    return result


def grid_shuffle_batch_gpu_non_jit(images: torch.Tensor, grid_size: int) -> torch.Tensor:
    """
    Non-JIT 버전 (fallback)
    """
    B, C, H, W = images.shape
    
    if grid_size <= 1:
        return images
    
    patch_h = H // grid_size
    patch_w = W // grid_size
    
    if patch_h < 1 or patch_w < 1:
        return images
    
    used_h = patch_h * grid_size
    used_w = patch_w * grid_size
    
    result = images.clone()
    grid_region = images[:, :, :used_h, :used_w]
    
    patches = grid_region.view(B, C, grid_size, patch_h, grid_size, patch_w)
    patches = patches.permute(0, 1, 2, 4, 3, 5)
    num_patches = grid_size * grid_size
    patches = patches.reshape(B, C, num_patches, patch_h, patch_w)
    
    shuffled_patches = torch.empty_like(patches)
    for b in range(B):
        perm = torch.randperm(num_patches, device=images.device)
        shuffled_patches[b] = patches[b, :, perm, :, :]
    
    shuffled_patches = shuffled_patches.view(B, C, grid_size, grid_size, patch_h, patch_w)
    shuffled_patches = shuffled_patches.permute(0, 1, 2, 4, 3, 5)
    shuffled_region = shuffled_patches.reshape(B, C, used_h, used_w)
    
    result[:, :, :used_h, :used_w] = shuffled_region
    
    return result


# ==============================================================================
# 5) Model Definition (Encoder)
# ==============================================================================
def parse_int_list(s: str, n: int) -> Tuple[int, ...]:
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    vals = [int(p) for p in parts]
    if len(vals) != n:
        raise ValueError(f"Expected {n} ints, got {len(vals)} from '{s}'")
    return tuple(vals)


def conv2d(in_ch, out_ch, k=3, stride=1, padding=1, dilation=1, bias=True):
    return nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=stride, padding=padding, dilation=dilation, bias=bias)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1):
        super().__init__()
        self.c1 = conv2d(in_ch, out_ch, 3, 1, padding=dilation, dilation=dilation, bias=True)
        self.c2 = conv2d(out_ch, out_ch, 3, 1, padding=dilation, dilation=dilation, bias=True)
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
    def __init__(self, in_ch, out_ch, n_blocks, dilation, use_ckpt: bool, ckpt_segments: int):
        super().__init__()
        self.use_ckpt = bool(use_ckpt)
        self.ckpt_segments = int(ckpt_segments)
        blocks = [ResBlock(in_ch, out_ch, dilation=dilation)]
        for _ in range(n_blocks - 1):
            blocks.append(ResBlock(out_ch, out_ch, dilation=dilation))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        if self.use_ckpt and self.training and self.ckpt_segments > 1 and len(self.blocks) > 1:
            seg = min(self.ckpt_segments, len(self.blocks))
            return checkpoint_sequential(self.blocks, seg, x, use_reentrant=False)
        return self.blocks(x)


class Encoder(nn.Module):
    def __init__(self, blocks=(2, 2, 4, 4), dilations=(1, 1, 1, 1), refine_blocks=1, ckpt_segments=2):
        super().__init__()
        b2, b3, b4, b5 = blocks
        d2, d3, d4, d5 = dilations

        self.stem = nn.Sequential(conv2d(3, 64, k=3, stride=2, padding=1, bias=True))
        self.stage2 = Stage(64, 128, b2, d2, use_ckpt=False, ckpt_segments=1)
        self.stage3 = Stage(128, 256, b3, d3, use_ckpt=False, ckpt_segments=1)
        self.stage4 = Stage(256, 512, b4, d4, use_ckpt=True, ckpt_segments=ckpt_segments)
        self.stage5 = Stage(512, OUT_DIM, b5, d5, use_ckpt=True, ckpt_segments=ckpt_segments)
        self.refine = Stage(OUT_DIM, OUT_DIM, int(refine_blocks), 1, use_ckpt=True, ckpt_segments=ckpt_segments)

        self.trunk = nn.Sequential(self.stem, self.stage2, self.stage3, self.stage4, self.stage5, self.refine)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.trunk(x)
        x = self.gap(x).flatten(1)
        return x


# ==============================================================================
# 6) GPU-optimized Feature Extraction & Evaluation
# ==============================================================================
@torch.no_grad()
def renorm_unit_per_out_channel_(model: nn.Module, eps: float = 1e-12):
    """각 출력 채널의 가중치를 단위 노름으로 정규화"""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            w = m.weight.data
            n = w.flatten(1).norm(dim=1, keepdim=True).clamp_min(eps)
            w.div_(n.view(-1, 1, 1, 1))
        elif isinstance(m, nn.Linear):
            w = m.weight.data
            n = w.norm(dim=1, keepdim=True).clamp_min(eps)
            w.div_(n)


def extract_features_gpu(
    encoder: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    grid_size: int = 0,  # 0 = no shuffle
    use_bf16: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GPU에서 feature 추출 (셔플 포함)
    모든 연산을 GPU에서 수행하고 마지막에만 CPU로 복사
    
    Returns:
        features: (N, OUT_DIM) GPU tensor
        labels: (N,) GPU tensor
    """
    encoder.eval()
    all_features = []
    all_labels = []
    
    autocast_kwargs = dict(device_type="cuda", enabled=True, dtype=torch.bfloat16 if use_bf16 else torch.float16)
    
    with torch.inference_mode():
        for batch in tqdm(data_loader, desc=f"Extract (grid={grid_size})", leave=False):
            if batch is None:
                continue
            x, y, uid = batch
            
            # GPU로 이동 + channels_last
            x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            
            # GPU에서 셔플 수행
            if grid_size > 1:
                x = grid_shuffle_batch_gpu_non_jit(x, grid_size)
            
            # Feature extraction with mixed precision
            with torch.amp.autocast(**autocast_kwargs):
                feat = encoder(x)
            
            # L2 normalize
            feat = F.normalize(feat.float(), dim=1)
            
            all_features.append(feat)
            all_labels.append(y)
    
    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    return features, labels


def train_linear_probe_gpu(
    features_train: torch.Tensor,
    labels_train: torch.Tensor,
    num_classes: int,
    device: torch.device,
    args
) -> nn.Module:
    """
    GPU에서 Linear classifier 학습 (feature도 GPU에 유지)
    """
    # bias=False as per standard linear classification
    probe = nn.Linear(OUT_DIM, num_classes, bias=False).to(device)
    
    # SGD optimizer
    optimizer = optim.SGD(
        probe.parameters(),
        lr=args.lp_lr,
        momentum=args.lp_momentum,
        weight_decay=args.lp_wd
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # GPU에서 직접 TensorDataset 생성
    dataset = torch.utils.data.TensorDataset(features_train, labels_train)
    loader = DataLoader(dataset, batch_size=args.lp_batch_size, shuffle=True, drop_last=False)
    
    probe.train()
    for epoch in range(1, args.lp_epochs + 1):
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(loader, desc=f"LP Epoch {epoch}/{args.lp_epochs}", leave=False)
        for x_batch, y_batch in pbar:
            optimizer.zero_grad(set_to_none=True)
            logits = probe(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * x_batch.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y_batch).sum().item()
            total += x_batch.size(0)
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct/total*100:.1f}%"})
        
        epoch_loss = total_loss / total
        epoch_acc = correct / total
        logger.info(f"LP Epoch {epoch}: Loss={epoch_loss:.4f}, Train Acc={epoch_acc*100:.2f}%")
    
    probe.eval()
    return probe


def evaluate_probe_gpu(
    probe: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    GPU에서 평가 수행
    """
    probe.eval()
    
    with torch.inference_mode():
        logits = probe(features)
        predictions = logits.argmax(dim=1)
    
    # CPU로 복사해서 sklearn 사용
    predictions_np = predictions.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    accuracy = accuracy_score(labels_np, predictions_np)
    
    return accuracy, predictions_np, labels_np


# ==============================================================================
# 7) Visualization Functions
# ==============================================================================
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: str,
    title: str = "Confusion Matrix"
):
    """Confusion matrix 시각화 및 저장"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": 14}
    )
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved confusion matrix: {save_path}")


def plot_perturbation_curve(
    grid_sizes: List[int],
    accuracies_mean: List[float],
    accuracies_std: List[float],
    save_path: str
):
    """그리드 크기별 정확도 변화 그래프"""
    plt.figure(figsize=(12, 6))
    
    # Error bar plot
    plt.errorbar(
        grid_sizes,
        [acc * 100 for acc in accuracies_mean],
        yerr=[std * 100 for std in accuracies_std],
        marker='o',
        markersize=10,
        linewidth=2.5,
        capsize=6,
        capthick=2,
        color='#2ecc71',
        ecolor='#27ae60',
        label='Test Accuracy'
    )
    
    plt.xlabel("Grid Size (N×N)", fontsize=14)
    plt.ylabel("Accuracy (%)", fontsize=14)
    plt.title("Perturbation Test: Accuracy vs Grid Shuffle Size", fontsize=16)
    plt.xticks(grid_sizes, [f"{g}×{g}" for g in grid_sizes], fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Baseline 참조선
    if 1 in grid_sizes:
        baseline_acc = accuracies_mean[grid_sizes.index(1)] * 100
        plt.axhline(y=baseline_acc, color='r', linestyle='--', alpha=0.5)
        plt.text(grid_sizes[-1], baseline_acc + 1, f'Baseline: {baseline_acc:.1f}%', 
                 color='r', fontsize=11, ha='right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved perturbation curve: {save_path}")


def save_results_table(results: List[Dict], save_path: str):
    """결과 테이블 CSV 저장"""
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["grid_size", "accuracy_mean", "accuracy_std", "num_trials"])
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"Saved results table: {save_path}")


def visualize_shuffled_examples(
    bank: InMemoryTarBank,
    indices: List[int],
    grid_sizes: List[int],
    save_dir: str,
    num_examples: int = 5
):
    """각 그리드 크기별 셔플된 이미지 예시 저장"""
    os.makedirs(save_dir, exist_ok=True)
    
    normalize = SafeInstanceNormalize(threshold=0.01)
    
    for i, idx in enumerate(indices[:num_examples]):
        img_np = bank.images[idx]
        if img_np is None:
            continue
        
        label = bank.labels[idx]
        class_name = LABEL_TO_CLASS[label]
        
        # 16-bit to float32 [0, 1]
        x = (img_np.astype(np.float32) / 65535.0)
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        
        fig, axes = plt.subplots(1, len(grid_sizes) + 1, figsize=(4 * (len(grid_sizes) + 1), 4))
        
        # Original
        orig_display = x[0].permute(1, 2, 0).numpy()
        orig_display = (orig_display - orig_display.min()) / (orig_display.max() - orig_display.min() + 1e-8)
        axes[0].imshow(orig_display)
        axes[0].set_title(f"Original ({class_name})", fontsize=11)
        axes[0].axis("off")
        
        # Shuffled versions
        for j, grid_size in enumerate(grid_sizes):
            shuffled = grid_shuffle_batch_gpu_non_jit(x.clone(), grid_size)
            shuffled_display = shuffled[0].permute(1, 2, 0).numpy()
            shuffled_display = (shuffled_display - shuffled_display.min()) / (shuffled_display.max() - shuffled_display.min() + 1e-8)
            axes[j + 1].imshow(shuffled_display)
            axes[j + 1].set_title(f"{grid_size}×{grid_size} Shuffle", fontsize=11)
            axes[j + 1].axis("off")
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"example_{i+1}_{class_name}.png"), dpi=100, bbox_inches="tight")
        plt.close()
    
    logger.info(f"Saved {min(num_examples, len(indices))} shuffle examples to {save_dir}")


# ==============================================================================
# 8) Load Split CSV
# ==============================================================================
def load_split_csv(csv_path: str) -> List[str]:
    """CSV에서 uid 목록 로드"""
    uids = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uids.append(row["uid"])
    return uids


# ==============================================================================
# 9) Main Evaluation Pipeline (GPU Optimized)
# ==============================================================================
def run_perturbation_test(args):
    """전체 perturbation test 실행 (A100 최적화)"""
    set_seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ==== Load refs ====
    logger.info("Loading sample refs...")
    refs = load_all_sample_refs(args.shard_root)
    
    # ==== Load train/test split from CSV ====
    train_csv = os.path.join(args.save_dir, "train_split.csv")
    test_csv = os.path.join(args.save_dir, "test_split.csv")
    
    if not os.path.exists(train_csv) or not os.path.exists(test_csv):
        raise FileNotFoundError(f"Split CSVs not found in {args.save_dir}. Need train_split.csv and test_split.csv")
    
    train_uids = load_split_csv(train_csv)
    test_uids = load_split_csv(test_csv)
    
    logger.info(f"Train samples: {len(train_uids)}, Test samples: {len(test_uids)}")
    
    # uid -> ref index mapping
    uid_to_refidx = {f"{r.tar_path}:{r.prefix}": i for i, r in enumerate(refs)}
    
    train_refidx = [uid_to_refidx[u] for u in train_uids if u in uid_to_refidx]
    test_refidx = [uid_to_refidx[u] for u in test_uids if u in uid_to_refidx]
    
    if len(train_refidx) != len(train_uids):
        logger.warning(f"Some train UIDs not found: {len(train_uids) - len(train_refidx)} missing")
    if len(test_refidx) != len(test_uids):
        logger.warning(f"Some test UIDs not found: {len(test_uids) - len(test_refidx)} missing")
    
    # ==== Build data banks ====
    logger.info("Building train data bank...")
    train_bank = InMemoryTarBank(refs, train_refidx, args.img_size)
    
    logger.info("Building test data bank...")
    test_bank = InMemoryTarBank(refs, test_refidx, args.img_size)
    
    train_ib = list(range(len(train_refidx)))
    test_ib = list(range(len(test_refidx)))
    
    # ==== Load encoder ====
    logger.info(f"Loading encoder from: {args.ckpt_path}")
    
    blocks = parse_int_list(args.blocks, 4)
    dilations = parse_int_list(args.dilations, 4)
    
    encoder = Encoder(
        blocks=blocks,
        dilations=dilations,
        refine_blocks=args.refine_blocks,
        ckpt_segments=args.ckpt_segments
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.ckpt_path, map_location="cpu")
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if "model_q" in checkpoint:
            state_dict = checkpoint["model_q"]
            encoder_state = {k.replace("encoder.", ""): v for k, v in state_dict.items() if k.startswith("encoder.")}
            encoder_state = {k.replace("_orig_mod.", ""): v for k, v in encoder_state.items()}
        else:
            state_dict = checkpoint
            if any(k.startswith("encoder.") for k in state_dict.keys()):
                encoder_state = {k.replace("encoder.", ""): v for k, v in state_dict.items() if k.startswith("encoder.")}
            else:
                encoder_state = state_dict
            encoder_state = {k.replace("_orig_mod.", ""): v for k, v in encoder_state.items()}
    else:
        raise ValueError(f"Unknown checkpoint format: {type(checkpoint)}")
    
    encoder.load_state_dict(encoder_state, strict=True)
    encoder.eval()
    encoder.to(device).to(memory_format=torch.channels_last)
    
    # Apply unit norm
    renorm_unit_per_out_channel_(encoder)
    
    # torch.compile for A100 (optional, can speed up)
    if args.use_compile:
        try:
            encoder = torch.compile(encoder, mode="reduce-overhead")
            logger.info("torch.compile enabled for encoder")
        except Exception as e:
            logger.info(f"torch.compile not available: {e}")
    
    logger.info("Encoder loaded successfully")
    
    # ==== Datasets & Loaders ====
    # A100에서는 큰 배치 사이즈 사용 가능
    train_ds = InMemorySixteenBitDataset(train_bank, train_ib, args.img_size)
    test_ds = InMemorySixteenBitDataset(test_bank, test_ib, args.img_size)
    
    # num_workers > 0 for faster loading (pin_memory=True)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, 
        collate_fn=collate_skip_none, persistent_workers=(args.num_workers > 0)
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_skip_none, persistent_workers=(args.num_workers > 0)
    )
    
    # ==== Extract features for training (no shuffle) ====
    logger.info("Extracting train features (GPU)...")
    train_features_gpu, train_labels_gpu = extract_features_gpu(
        encoder, train_loader, device, grid_size=0, use_bf16=args.use_bf16
    )
    logger.info(f"Train features shape: {train_features_gpu.shape}, device: {train_features_gpu.device}")
    
    # ==== Train linear probe (GPU) ====
    logger.info("Training linear probe (GPU)...")
    linear_probe = train_linear_probe_gpu(
        train_features_gpu, train_labels_gpu, args.num_classes, device, args
    )
    
    # ==== Baseline evaluation (no shuffle) ====
    logger.info("\n" + "=" * 60)
    logger.info("Baseline Evaluation (No Shuffle)")
    logger.info("=" * 60)
    
    test_features_gpu, test_labels_gpu = extract_features_gpu(
        encoder, test_loader, device, grid_size=0, use_bf16=args.use_bf16
    )
    baseline_acc, baseline_preds, test_labels_np = evaluate_probe_gpu(
        linear_probe, test_features_gpu, test_labels_gpu, device
    )
    
    logger.info(f"Baseline Test Accuracy: {baseline_acc * 100:.2f}%")
    
    # Classification report
    class_names = ["Control", "SNCA", "GBA", "LRRK2"]
    report = classification_report(test_labels_np, baseline_preds, target_names=class_names)
    logger.info(f"\nClassification Report:\n{report}")
    
    # Save classification report
    with open(os.path.join(args.output_dir, "baseline_classification_report.txt"), "w") as f:
        f.write(f"Baseline Test Accuracy: {baseline_acc * 100:.2f}%\n\n")
        f.write(report)
    
    # Confusion matrix
    plot_confusion_matrix(
        test_labels_np, baseline_preds, class_names,
        os.path.join(args.output_dir, "baseline_confusion_matrix.png"),
        title=f"Baseline Confusion Matrix (Acc: {baseline_acc*100:.2f}%)"
    )
    
    # ==== Perturbation test with different grid sizes ====
    logger.info("\n" + "=" * 60)
    logger.info("Perturbation Test Results (GPU-accelerated)")
    logger.info("=" * 60)
    
    grid_sizes_str = args.grid_sizes.split(",")
    grid_sizes = [int(g.strip()) for g in grid_sizes_str]
    
    # Include baseline (grid_size=1)
    if 1 not in grid_sizes:
        grid_sizes = [1] + grid_sizes
    grid_sizes = sorted(grid_sizes)
    
    results = []
    accuracies_mean = []
    accuracies_std = []
    
    for grid_size in grid_sizes:
        if grid_size == 1:
            # Baseline
            acc_mean = baseline_acc
            acc_std = 0.0
            logger.info(f"Grid {grid_size}×{grid_size}: {acc_mean*100:.2f}% ± {acc_std*100:.2f}% (baseline)")
        else:
            # Multiple trials with random shuffle (all on GPU)
            trial_accs = []
            
            for trial in range(args.num_shuffle_trials):
                # Extract features with GPU shuffle
                shuffled_features_gpu, shuffled_labels_gpu = extract_features_gpu(
                    encoder, test_loader, device,
                    grid_size=grid_size, use_bf16=args.use_bf16
                )
                
                # Evaluate on GPU
                acc, _, _ = evaluate_probe_gpu(
                    linear_probe, shuffled_features_gpu, shuffled_labels_gpu, device
                )
                trial_accs.append(acc)
                
                # Clear cache after each trial
                del shuffled_features_gpu, shuffled_labels_gpu
                torch.cuda.empty_cache()
            
            acc_mean = np.mean(trial_accs)
            acc_std = np.std(trial_accs)
            logger.info(f"Grid {grid_size}×{grid_size}: {acc_mean*100:.2f}% ± {acc_std*100:.2f}% ({args.num_shuffle_trials} trials)")
        
        results.append({
            "grid_size": grid_size,
            "accuracy_mean": acc_mean,
            "accuracy_std": acc_std,
            "num_trials": args.num_shuffle_trials if grid_size > 1 else 1
        })
        accuracies_mean.append(acc_mean)
        accuracies_std.append(acc_std)
    
    # ==== Save results ====
    save_results_table(results, os.path.join(args.output_dir, "perturbation_results.csv"))
    
    # Plot perturbation curve
    plot_perturbation_curve(
        grid_sizes, accuracies_mean, accuracies_std,
        os.path.join(args.output_dir, "perturbation_curve.png")
    )
    
    # Visualize shuffle examples
    visualize_shuffled_examples(
        test_bank, test_ib, [g for g in grid_sizes if g > 1],
        os.path.join(args.output_dir, "shuffled_examples"),
        num_examples=5
    )
    
    # ==== Print summary table ====
    logger.info("\n" + "=" * 60)
    logger.info("Summary Table")
    logger.info("=" * 60)
    logger.info(f"{'Grid Size':<12} | {'Accuracy (Mean ± Std)':<25}")
    logger.info("-" * 40)
    for r in results:
        gs = r["grid_size"]
        label = f"{gs}×{gs}" if gs > 1 else "1×1 (baseline)"
        logger.info(f"{label:<12} | {r['accuracy_mean']*100:.2f}% ± {r['accuracy_std']*100:.2f}%")
    
    logger.info("\n" + "=" * 60)
    logger.info(f"All results saved to: {args.output_dir}")
    logger.info("=" * 60)
    
    # Memory cleanup
    del train_features_gpu, train_labels_gpu, test_features_gpu, test_labels_gpu
    torch.cuda.empty_cache()
    
    return results


# ==============================================================================
# 10) Arguments
# ==============================================================================
def get_args():
    p = argparse.ArgumentParser("Perturbation Test: Grid Shuffle Evaluation (A100 Optimized)")
    
    # Required paths
    p.add_argument("--ckpt_path", type=str,
                   default="/content/drive/MyDrive/Final_paper/Model_MoCoXBM_PlateLP_LRRK2_L2 norm_hidden2048_resume_bias=True_clean_image/best_model.pt",
                   help="Path to trained model checkpoint (.pt)")
    p.add_argument("--shard_root", type=str, default=DEFAULT_SHARD_ROOT,
                   help="Path to tar shards root directory")
    p.add_argument("--save_dir", type=str,
                   default="/content/drive/MyDrive/Final_paper/Model_MoCoXBM_PlateLP_LRRK2_L2 norm_hidden2048_resume_bias=True_clean_image",
                   help="Directory containing train/val/test split CSVs")
    p.add_argument("--output_dir", type=str, default="/content/perturbation_results",
                   help="Output directory for results")
    
    # Grid shuffle settings
    p.add_argument("--grid_sizes", type=str, default="2,3,4,5,6,8,16",
                   help="Grid sizes to test (comma separated)")
    p.add_argument("--num_shuffle_trials", type=int, default=10,
                   help="Number of random shuffle trials per grid size")
    
    # Data settings
    p.add_argument("--img_size", type=int, default=128, help="Image size")
    p.add_argument("--batch_size", type=int, default=512, help="Batch size (larger for A100)")
    p.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Model architecture
    p.add_argument("--blocks", type=str, default="2,2,2,3", help="Encoder blocks")
    p.add_argument("--dilations", type=str, default="1,1,1,1", help="Encoder dilations")
    p.add_argument("--refine_blocks", type=int, default=1, help="Refine blocks")
    p.add_argument("--ckpt_segments", type=int, default=0, help="Checkpoint segments")
    
    # Linear probe settings
    p.add_argument("--num_classes", type=int, default=4, help="Number of classes")
    p.add_argument("--lp_epochs", type=int, default=10, help="Linear probe epochs")
    p.add_argument("--lp_lr", type=float, default=0.1, help="Linear probe LR")
    p.add_argument("--lp_wd", type=float, default=0.0, help="Linear probe WD")
    p.add_argument("--lp_momentum", type=float, default=0.9, help="Linear probe momentum")
    p.add_argument("--lp_batch_size", type=int, default=8192, help="Linear probe batch size")
    
    # GPU settings (A100 optimized)
    p.add_argument("--device", type=str, default="cuda", help="Device")
    p.add_argument("--use_bf16", action="store_true", default=True, help="Use BF16 (A100)")
    p.add_argument("--use_compile", action="store_true", default=False, help="Use torch.compile")
    
    # Colab notebook mode
    if "ipykernel" in sys.modules:
        return p.parse_args([])
    return p.parse_args()


# ==============================================================================
# 11) Main Entry Point
# ==============================================================================
def main():
    args = get_args()
    
    # Print config
    logger.info("=" * 60)
    logger.info("Perturbation Test Configuration (A100 Optimized)")
    logger.info("=" * 60)
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 60)
    
    # Save config
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    
    # Run test
    results = run_perturbation_test(args)
    
    return results


if __name__ == "__main__":
    main()
