# ==============================================================================
# Perturbation Test: Evaluate CNN's Spatial Dependency via Grid Shuffling
# ==============================================================================
# 이 스크립트는 학습된 SupCon CNN 모델의 공간적 정보 의존도를 평가합니다.
# 이미지를 N×N 그리드로 자른 후 랜덤 셔플하여 분류 성능 변화를 측정합니다.
#
# 사용법:
#   python perturbation_test.py --ckpt_path /path/to/best_model.pt --save_dir /path/to/output
#
# Colab에서 실행 시:
#   args = get_args()  # 모든 default가 Colab 경로로 설정됨
# ==============================================================================

import argparse
import csv
import glob
import io
import json
import logging
import os
import pickle
import random
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.checkpoint import checkpoint_sequential
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
from tqdm.auto import tqdm

try:
    import tifffile
except ImportError:
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "tifffile"])
    import tifffile

from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)

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
# Constants (동일하게 유지)
# ==============================================================================
DEFAULT_SHARD_ROOT = "/content/wds_shards"

LINE_FOLDERS = ["Control_C4", "Control_C18", "Control_C19", "SNCA", "GBA", "LRRK2"]

SUPERCLASS_MAP = {
    "Control_C4": "Control",
    "Control_C18": "Control",
    "Control_C19": "Control",
    "SNCA": "SNCA",
    "GBA": "GBA",
    "LRRK2": "LRRK2",
}
CLASS_TO_LABEL = {"Control": 0, "SNCA": 1, "GBA": 2, "LRRK2": 3}
LABEL_TO_CLASS = {v: k for k, v in CLASS_TO_LABEL.items()}

PLATE_DIR_RE = re.compile(r"plate=(\d{6})")

OUT_DIM = 512


# ==============================================================================
# 1) Reproducibility
# ==============================================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
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
# 2) Tar Index & SampleRef (학습 코드와 동일)
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
            pairs.append(
                (pref, it["tif_off"], it["tif_size"], it["js_off"], it["js_size"])
            )

    with open(idx_path, "wb") as f:
        pickle.dump(pairs, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info(
        f"[tar-index] built {len(pairs)} pairs: {os.path.basename(tar_path)} ({time.time()-t0:.1f}s)"
    )


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
            refs.append(
                SampleRef(
                    tar_path=tp,
                    prefix=pref,
                    tif_off=int(tif_off),
                    tif_size=int(tif_size),
                    js_off=int(js_off),
                    js_size=int(js_size),
                    line=line,
                    superclass=superclass,
                    label=label,
                    plate=plate,
                )
            )

    logger.info(f"Loaded sample refs: {len(refs)}")
    return refs


# ==============================================================================
# 3) Dataset (학습 코드와 동일)
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
    """Instance normalization with std flooring to prevent noise amplification"""

    def __init__(self, threshold: float = 0.01):
        self.threshold = float(threshold)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = tensor.mean(dim=[1, 2], keepdim=True)
        std = tensor.std(dim=[1, 2], keepdim=True).clamp_min(self.threshold)
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

        logger.info(
            f"⚡ Preloading {len(ref_indices)} images into RAM from tar shards..."
        )

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

        logger.info(
            f"Preload done. bad={bad}/{len(ref_indices)} elapsed={(time.time()-t0)/60:.1f} min"
        )


class InMemorySixteenBitDataset(Dataset):
    """Dataset with optional grid shuffle transform"""

    def __init__(
        self,
        bank: InMemoryTarBank,
        indices_in_bank: List[int],
        img_size: int,
        grid_shuffle: Optional[int] = None,
        augment: bool = False,
    ):
        self.bank = bank
        self.ib = indices_in_bank
        self.img_size = int(img_size)
        self.grid_shuffle = grid_shuffle
        self.augment = bool(augment)
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
        x = img.astype(np.float32) / 65535.0
        x = torch.from_numpy(x).permute(2, 0, 1)  # (C, H, W)

        # Apply grid shuffle if specified
        if self.grid_shuffle is not None and self.grid_shuffle > 1:
            x = grid_shuffle_transform(x, self.grid_shuffle)

        # Instance normalize
        x = self.normalize(x)

        return x, y, uid


# ==============================================================================
# 4) Grid Shuffle Transform (using albumentations)
# ==============================================================================
import albumentations as A


class GridShuffleTransform:
    """
    albumentations RandomGridShuffle을 이용한 그리드 셔플.
    128의 약수가 아닌 grid_size도 사용 가능합니다 (나머지 픽셀은 마지막 패치에 포함).
    """

    def __init__(self, grid_size: int = 4):
        self.grid_size = grid_size
        if grid_size > 1:
            self.transform = A.RandomGridShuffle(grid=(grid_size, grid_size), p=1.0)
        else:
            self.transform = None

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: (C, H, W) torch.Tensor
        Returns:
            shuffled: (C, H, W) torch.Tensor
        """
        if self.transform is None or self.grid_size <= 1:
            return img

        # torch (C, H, W) -> numpy (H, W, C)
        img_np = img.permute(1, 2, 0).numpy()

        # Apply albumentations transform
        result = self.transform(image=img_np)
        shuffled_np = result["image"]

        # numpy (H, W, C) -> torch (C, H, W)
        return torch.from_numpy(shuffled_np).permute(2, 0, 1)


def grid_shuffle_transform(img: torch.Tensor, grid_size: int) -> torch.Tensor:
    """
    albumentations RandomGridShuffle을 이용한 그리드 셔플.

    Args:
        img: (C, H, W) torch.Tensor
        grid_size: N for NxN grid shuffle (2 이상)

    Returns:
        shuffled: (C, H, W) torch.Tensor

    Note: 128의 약수가 아닌 grid_size도 사용 가능 (나머지 픽셀은 자동 처리).
    """
    if grid_size <= 1:
        return img

    # torch (C, H, W) -> numpy (H, W, C)
    img_np = img.permute(1, 2, 0).numpy()

    # Apply albumentations RandomGridShuffle
    transform = A.RandomGridShuffle(grid=(grid_size, grid_size), p=1.0)
    result = transform(image=img_np)
    shuffled_np = result["image"]

    # numpy (H, W, C) -> torch (C, H, W)
    return torch.from_numpy(shuffled_np).permute(2, 0, 1)


# ==============================================================================
# 5) Model Definition (학습 코드와 동일한 Encoder)
# ==============================================================================
def parse_int_list(s: str, n: int) -> Tuple[int, ...]:
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    vals = [int(p) for p in parts]
    if len(vals) != n:
        raise ValueError(f"Expected {n} ints, got {len(vals)} from '{s}'")
    return tuple(vals)


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
        self, in_ch, out_ch, n_blocks, dilation, use_ckpt: bool, ckpt_segments: int
    ):
        super().__init__()
        self.use_ckpt = bool(use_ckpt)
        self.ckpt_segments = int(ckpt_segments)
        blocks = [ResBlock(in_ch, out_ch, dilation=dilation)]
        for _ in range(n_blocks - 1):
            blocks.append(ResBlock(out_ch, out_ch, dilation=dilation))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        if (
            self.use_ckpt
            and self.training
            and self.ckpt_segments > 1
            and len(self.blocks) > 1
        ):
            seg = min(self.ckpt_segments, len(self.blocks))
            return checkpoint_sequential(self.blocks, seg, x, use_reentrant=False)
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

        self.stem = nn.Sequential(
            conv2d(3, 64, k=3, stride=2, padding=1, bias=True)
        )  # 128->64
        self.stage2 = Stage(64, 128, b2, d2, use_ckpt=False, ckpt_segments=1)
        self.stage3 = Stage(128, 256, b3, d3, use_ckpt=False, ckpt_segments=1)
        self.stage4 = Stage(
            256, 512, b4, d4, use_ckpt=True, ckpt_segments=ckpt_segments
        )
        self.stage5 = Stage(
            512, OUT_DIM, b5, d5, use_ckpt=True, ckpt_segments=ckpt_segments
        )
        self.refine = Stage(
            OUT_DIM,
            OUT_DIM,
            int(refine_blocks),
            1,
            use_ckpt=True,
            ckpt_segments=ckpt_segments,
        )

        self.trunk = nn.Sequential(
            self.stem, self.stage2, self.stage3, self.stage4, self.stage5, self.refine
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.trunk(x)
        x = self.gap(x).flatten(1)
        return x


# ==============================================================================
# 6) Linear Probe Training
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


def extract_features(
    encoder: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    use_bf16: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encoder를 통해 모든 이미지의 feature를 추출합니다.

    Returns:
        features: (N, OUT_DIM) numpy array
        labels: (N,) numpy array
    """
    encoder.eval()
    all_features = []
    all_labels = []

    autocast_kwargs = dict(device_type="cuda", enabled=torch.cuda.is_available())
    if use_bf16:
        autocast_kwargs["dtype"] = torch.bfloat16

    with torch.inference_mode():
        for batch in tqdm(data_loader, desc="Extracting features", leave=False):
            if batch is None:
                continue
            x, y, uid = batch
            x = x.to(device, non_blocking=True).contiguous(
                memory_format=torch.channels_last
            )

            with torch.amp.autocast(**autocast_kwargs):
                feat = encoder(x)

            # L2 normalize (amount normalization)
            feat = F.normalize(feat, dim=1)

            all_features.append(feat.cpu().numpy())
            all_labels.append(y.numpy())

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    return features, labels


def train_linear_probe(
    features_train: np.ndarray, labels_train: np.ndarray, num_classes: int, args
) -> nn.Module:
    """
    Linear classifier를 학습합니다.

    Args:
        features_train: (N, D) training features
        labels_train: (N,) training labels
        num_classes: number of classes
        args: arguments

    Returns:
        Trained linear probe module
    """
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # bias=False as per standard linear classification
    probe = nn.Linear(OUT_DIM, num_classes, bias=False).to(device)

    # SGD optimizer
    optimizer = optim.SGD(
        probe.parameters(),
        lr=args.lp_lr,
        momentum=args.lp_momentum,
        weight_decay=args.lp_wd,
    )

    criterion = nn.CrossEntropyLoss()

    # Convert to tensors
    X = torch.from_numpy(features_train).float()
    Y = torch.from_numpy(labels_train).long()

    # Create simple dataset
    dataset = torch.utils.data.TensorDataset(X, Y)
    loader = DataLoader(
        dataset, batch_size=args.lp_batch_size, shuffle=True, drop_last=False
    )

    probe.train()
    for epoch in range(1, args.lp_epochs + 1):
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(loader, desc=f"LP Epoch {epoch}/{args.lp_epochs}", leave=False)
        for x_batch, y_batch in pbar:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = probe(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x_batch.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y_batch).sum().item()
            total += x_batch.size(0)

            pbar.set_postfix(
                {"loss": f"{loss.item():.4f}", "acc": f"{correct/total*100:.1f}%"}
            )

        epoch_loss = total_loss / total
        epoch_acc = correct / total
        logger.info(
            f"LP Epoch {epoch}: Loss={epoch_loss:.4f}, Train Acc={epoch_acc*100:.2f}%"
        )

    probe.eval()
    return probe


def evaluate_probe(
    probe: nn.Module, features: np.ndarray, labels: np.ndarray, device: torch.device
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Linear probe를 평가합니다.

    Returns:
        accuracy: float
        predictions: numpy array
        labels: numpy array
    """
    probe.eval()

    X = torch.from_numpy(features).float().to(device)
    Y = torch.from_numpy(labels).long()

    with torch.inference_mode():
        logits = probe(X)
        predictions = logits.argmax(dim=1).cpu().numpy()

    accuracy = accuracy_score(labels, predictions)

    return accuracy, predictions, labels


# ==============================================================================
# 7) Visualization Functions
# ==============================================================================
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: str,
    title: str = "Confusion Matrix",
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
        annot_kws={"size": 14},
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
    save_path: str,
):
    """그리드 크기별 정확도 변화 그래프"""
    plt.figure(figsize=(10, 6))

    # Error bar plot
    plt.errorbar(
        grid_sizes,
        [acc * 100 for acc in accuracies_mean],
        yerr=[std * 100 for std in accuracies_std],
        marker="o",
        markersize=8,
        linewidth=2,
        capsize=5,
        capthick=2,
        color="#2ecc71",
        ecolor="#27ae60",
        label="Test Accuracy",
    )

    plt.xlabel("Grid Size (N×N)", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.title("Perturbation Test: Accuracy vs Grid Shuffle Size", fontsize=14)
    plt.xticks(grid_sizes, [f"{g}×{g}" for g in grid_sizes])
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)

    # Add baseline reference (grid_size=1)
    if 1 in grid_sizes:
        baseline_acc = accuracies_mean[grid_sizes.index(1)] * 100
        plt.axhline(
            y=baseline_acc,
            color="r",
            linestyle="--",
            alpha=0.5,
            label=f"Baseline: {baseline_acc:.1f}%",
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved perturbation curve: {save_path}")


def save_results_table(results: List[Dict], save_path: str):
    """결과 테이블 CSV 저장"""
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["grid_size", "accuracy_mean", "accuracy_std", "num_trials"]
        )
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"Saved results table: {save_path}")


def visualize_shuffled_examples(
    bank: InMemoryTarBank,
    indices: List[int],
    grid_sizes: List[int],
    save_dir: str,
    num_examples: int = 3,
):
    """각 그리드 크기별 셔플된 이미지 예시 저장"""
    os.makedirs(save_dir, exist_ok=True)

    normalize = SafeInstanceNormalize(threshold=0.01)

    for idx in indices[:num_examples]:
        img_np = bank.images[idx]
        if img_np is None:
            continue

        label = bank.labels[idx]
        class_name = LABEL_TO_CLASS[label]

        # 16-bit to float32 [0, 1]
        x = img_np.astype(np.float32) / 65535.0
        x = torch.from_numpy(x).permute(2, 0, 1)  # (C, H, W)

        fig, axes = plt.subplots(
            1, len(grid_sizes) + 1, figsize=(4 * (len(grid_sizes) + 1), 4)
        )

        # Original
        orig_display = x.permute(1, 2, 0).numpy()
        orig_display = (orig_display - orig_display.min()) / (
            orig_display.max() - orig_display.min() + 1e-8
        )
        axes[0].imshow(orig_display)
        axes[0].set_title(f"Original ({class_name})")
        axes[0].axis("off")

        # Shuffled versions
        for i, grid_size in enumerate(grid_sizes):
            shuffled = grid_shuffle_transform(x.clone(), grid_size)
            shuffled_display = shuffled.permute(1, 2, 0).numpy()
            shuffled_display = (shuffled_display - shuffled_display.min()) / (
                shuffled_display.max() - shuffled_display.min() + 1e-8
            )
            axes[i + 1].imshow(shuffled_display)
            axes[i + 1].set_title(f"{grid_size}×{grid_size} Shuffle")
            axes[i + 1].axis("off")

        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, f"example_{idx}_{class_name}.png"),
            dpi=100,
            bbox_inches="tight",
        )
        plt.close()

    logger.info(
        f"Saved {min(num_examples, len(indices))} shuffle examples to {save_dir}"
    )


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
# 9) Main Evaluation Pipeline
# ==============================================================================
def run_perturbation_test(args):
    """전체 perturbation test 실행"""
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # ==== Load refs ====
    logger.info("Loading sample refs...")
    refs = load_all_sample_refs(args.shard_root)

    # ==== Load train/test split from CSV ====
    train_csv = os.path.join(args.save_dir, "train_split.csv")
    test_csv = os.path.join(args.save_dir, "test_split.csv")

    if not os.path.exists(train_csv) or not os.path.exists(test_csv):
        raise FileNotFoundError(
            f"Split CSVs not found in {args.save_dir}. Need train_split.csv and test_split.csv"
        )

    train_uids = load_split_csv(train_csv)
    test_uids = load_split_csv(test_csv)

    logger.info(f"Train samples: {len(train_uids)}, Test samples: {len(test_uids)}")

    # uid -> ref index mapping
    uid_to_refidx = {f"{r.tar_path}:{r.prefix}": i for i, r in enumerate(refs)}

    train_refidx = [uid_to_refidx[u] for u in train_uids if u in uid_to_refidx]
    test_refidx = [uid_to_refidx[u] for u in test_uids if u in uid_to_refidx]

    if len(train_refidx) != len(train_uids):
        logger.warning(
            f"Some train UIDs not found: {len(train_uids) - len(train_refidx)} missing"
        )
    if len(test_refidx) != len(test_uids):
        logger.warning(
            f"Some test UIDs not found: {len(test_uids) - len(test_refidx)} missing"
        )

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
        ckpt_segments=args.ckpt_segments,
    )

    # Load checkpoint
    checkpoint = torch.load(args.ckpt_path, map_location="cpu")

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if "model_q" in checkpoint:
            # Full training checkpoint (resume_ckpt.pt) - extract encoder from model_q
            state_dict = checkpoint["model_q"]
            logger.info(f"Loaded resume checkpoint with {len(state_dict)} keys")
        else:
            # Direct state dict (best_model.pt, last_model.pt)
            state_dict = checkpoint
            logger.info(f"Loaded direct state dict with {len(state_dict)} keys")

        # Debug: print first few keys to understand structure
        sample_keys = list(state_dict.keys())[:5]
        logger.info(f"Sample keys: {sample_keys}")

        # Remove _orig_mod. prefix first (from torch.compile)
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

        # Check if keys have encoder. prefix
        has_encoder_prefix = any(k.startswith("encoder.") for k in state_dict.keys())
        logger.info(f"Has 'encoder.' prefix: {has_encoder_prefix}")

        if has_encoder_prefix:
            # Filter only encoder keys and remove prefix
            encoder_state = {}
            for k, v in state_dict.items():
                if k.startswith("encoder."):
                    new_key = k[len("encoder.") :]  # More reliable than replace
                    encoder_state[new_key] = v
            logger.info(f"Extracted {len(encoder_state)} encoder keys")
        else:
            # Keys don't have encoder. prefix - use as is
            encoder_state = state_dict
            logger.info(f"Using state dict as-is ({len(encoder_state)} keys)")

        # Debug: print first few encoder keys
        sample_encoder_keys = list(encoder_state.keys())[:5]
        logger.info(f"Sample encoder keys: {sample_encoder_keys}")
    else:
        raise ValueError(f"Unknown checkpoint format: {type(checkpoint)}")

    encoder.load_state_dict(encoder_state, strict=True)
    encoder.eval()
    encoder.to(device).to(memory_format=torch.channels_last)

    # Apply unit norm per output channel
    renorm_unit_per_out_channel_(encoder)

    logger.info("Encoder loaded successfully")

    # ==== Datasets & Loaders (baseline, no shuffle) ====
    train_ds = InMemorySixteenBitDataset(
        train_bank, train_ib, args.img_size, grid_shuffle=None, augment=False
    )
    test_ds = InMemorySixteenBitDataset(
        test_bank, test_ib, args.img_size, grid_shuffle=None, augment=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_skip_none,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_skip_none,
    )

    # ==== Extract features for training ====
    logger.info("Extracting train features...")
    train_features, train_labels = extract_features(
        encoder, train_loader, device, args.use_bf16
    )
    logger.info(f"Train features shape: {train_features.shape}")

    # ==== Train linear probe ====
    logger.info("Training linear probe...")
    linear_probe = train_linear_probe(
        train_features, train_labels, args.num_classes, args
    )

    # ==== Baseline evaluation (no shuffle) ====
    logger.info("\n" + "=" * 60)
    logger.info("Baseline Evaluation (No Shuffle)")
    logger.info("=" * 60)

    test_features, test_labels = extract_features(
        encoder, test_loader, device, args.use_bf16
    )
    baseline_acc, baseline_preds, _ = evaluate_probe(
        linear_probe, test_features, test_labels, device
    )

    logger.info(f"Baseline Test Accuracy: {baseline_acc * 100:.2f}%")

    # Classification report
    class_names = ["Control", "SNCA", "GBA", "LRRK2"]
    report = classification_report(
        test_labels, baseline_preds, target_names=class_names
    )
    logger.info(f"\nClassification Report:\n{report}")

    # Save classification report
    with open(
        os.path.join(args.output_dir, "baseline_classification_report.txt"), "w"
    ) as f:
        f.write(f"Baseline Test Accuracy: {baseline_acc * 100:.2f}%\n\n")
        f.write(report)

    # Confusion matrix
    plot_confusion_matrix(
        test_labels,
        baseline_preds,
        class_names,
        os.path.join(args.output_dir, "baseline_confusion_matrix.png"),
        title=f"Baseline Confusion Matrix (Acc: {baseline_acc*100:.2f}%)",
    )

    # ==== Perturbation test with different grid sizes ====
    logger.info("\n" + "=" * 60)
    logger.info("Perturbation Test Results")
    logger.info("=" * 60)

    grid_sizes_str = args.grid_sizes.split(",")
    grid_sizes = [int(g.strip()) for g in grid_sizes_str]

    # Include baseline (no shuffle = grid_size 1)
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
            logger.info(
                f"Grid {grid_size}×{grid_size}: {acc_mean*100:.2f}% ± {acc_std*100:.2f}% (baseline)"
            )
        else:
            # Multiple trials with random shuffle
            trial_accs = []

            for trial in range(args.num_shuffle_trials):
                # Create dataset with shuffle
                test_ds_shuffled = InMemorySixteenBitDataset(
                    test_bank,
                    test_ib,
                    args.img_size,
                    grid_shuffle=grid_size,
                    augment=False,
                )
                test_loader_shuffled = DataLoader(
                    test_ds_shuffled,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=True,
                    collate_fn=collate_skip_none,
                )

                # Extract features with shuffled images
                shuffled_features, shuffled_labels = extract_features(
                    encoder, test_loader_shuffled, device, args.use_bf16
                )

                # Evaluate
                acc, _, _ = evaluate_probe(
                    linear_probe, shuffled_features, shuffled_labels, device
                )
                trial_accs.append(acc)

            acc_mean = np.mean(trial_accs)
            acc_std = np.std(trial_accs)
            logger.info(
                f"Grid {grid_size}×{grid_size}: {acc_mean*100:.2f}% ± {acc_std*100:.2f}% ({args.num_shuffle_trials} trials)"
            )

        results.append(
            {
                "grid_size": grid_size,
                "accuracy_mean": acc_mean,
                "accuracy_std": acc_std,
                "num_trials": args.num_shuffle_trials if grid_size > 1 else 1,
            }
        )
        accuracies_mean.append(acc_mean)
        accuracies_std.append(acc_std)

    # ==== Save results ====
    save_results_table(
        results, os.path.join(args.output_dir, "perturbation_results.csv")
    )

    # Plot perturbation curve
    plot_perturbation_curve(
        grid_sizes,
        accuracies_mean,
        accuracies_std,
        os.path.join(args.output_dir, "perturbation_curve.png"),
    )

    # Visualize shuffle examples
    visualize_shuffled_examples(
        test_bank,
        test_ib,
        [g for g in grid_sizes if g > 1],
        os.path.join(args.output_dir, "shuffled_examples"),
        num_examples=5,
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
        logger.info(
            f"{label:<12} | {r['accuracy_mean']*100:.2f}% ± {r['accuracy_std']*100:.2f}%"
        )

    logger.info("\n" + "=" * 60)
    logger.info(f"All results saved to: {args.output_dir}")
    logger.info("=" * 60)

    return results


# ==============================================================================
# 10) Arguments
# ==============================================================================
def get_args():
    p = argparse.ArgumentParser("Perturbation Test: Grid Shuffle Evaluation")

    # Required paths
    p.add_argument(
        "--ckpt_path",
        type=str,
        default="/content/drive/MyDrive/Final_paper/Model_MoCoXBM_PlateLP_LRRK2_L2 norm_hidden2048_resume_bias=True_clean_image/best_model.pt",
        help="Path to trained model checkpoint (.pt)",
    )
    p.add_argument(
        "--shard_root",
        type=str,
        default=DEFAULT_SHARD_ROOT,
        help="Path to tar shards root directory",
    )
    p.add_argument(
        "--save_dir",
        type=str,
        default="/content/drive/MyDrive/Final_paper/Model_MoCoXBM_PlateLP_LRRK2_L2 norm_hidden2048_resume_bias=True_clean_image",
        help="Directory containing train/val/test split CSVs",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="/content/drive/MyDrive/Final_paper/Model_MoCoXBM_PlateLP_LRRK2_L2 norm_hidden2048_resume_bias=True_clean_image/perturbation_test",
        help="Output directory for results",
    )

    # Grid shuffle settings
    p.add_argument(
        "--grid_sizes",
        type=str,
        default="2,3,4,5,6,8,12,16",
        help="Grid sizes to test (comma separated, e.g., '2,3,4,8,16')",
    )
    p.add_argument(
        "--num_shuffle_trials",
        type=int,
        default=10,
        help="Number of random shuffle trials per grid size",
    )

    # Data settings
    p.add_argument("--img_size", type=int, default=128, help="Image size")
    p.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for feature extraction"
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed")

    # Model architecture (must match training)
    p.add_argument(
        "--blocks", type=str, default="2,2,2,3", help="Encoder blocks configuration"
    )
    p.add_argument("--dilations", type=str, default="1,1,1,1", help="Encoder dilations")
    p.add_argument(
        "--refine_blocks", type=int, default=1, help="Number of refine blocks"
    )
    p.add_argument("--ckpt_segments", type=int, default=0, help="Checkpoint segments")

    # Linear probe settings
    p.add_argument("--num_classes", type=int, default=4, help="Number of classes")
    p.add_argument(
        "--lp_epochs", type=int, default=10, help="Linear probe training epochs"
    )
    p.add_argument(
        "--lp_lr", type=float, default=0.1, help="Linear probe learning rate"
    )
    p.add_argument("--lp_wd", type=float, default=0.0, help="Linear probe weight decay")
    p.add_argument(
        "--lp_momentum", type=float, default=0.9, help="Linear probe momentum (SGD)"
    )
    p.add_argument(
        "--lp_batch_size", type=int, default=4096, help="Linear probe batch size"
    )

    # Device settings
    p.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    p.add_argument("--use_bf16", action="store_true", help="Use bfloat16 for inference")

    # Colab notebook mode
    if "ipykernel" in sys.modules:
        return p.parse_args([])
    return p.parse_args()


# ==============================================================================
# 11) Main Entry Point
# ==============================================================================
def main():
    args = get_args()

    # Print all arguments
    logger.info("=" * 60)
    logger.info("Perturbation Test Configuration")
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
