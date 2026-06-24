######## Early stopping 좀 더 길게하고(15). 원하면 그 뒷부분 그대로 그냥 학습 될 정도로 저장하게끔 코드 수정해서 A100 연산력 끌어내게 빠르게 수정.##################

##### early stopping 멈추는게 linear classification 성능 기준으로 하게. epoch 5 마다 linear classification 성능 평가하게끔 함.

#### 학습 #### XBM + moco

import argparse
import concurrent.futures
import copy
import csv
import glob
import logging
import os
import random
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.nn.utils.parametrizations import weight_norm
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
from tqdm.auto import tqdm

# ==============================================================================
# Logging
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ==============================================================================
# 0. User Configuration (HARDCODED FOR COLAB)
# ==============================================================================
DEFAULT_DATA_ROOTS = {
    "Control": [
        "/content/Control_C4/",
        "/content/Control_C18/",
        "/content/Control_C19/",
    ],
    "SNCA": ["/content/SNCA/"],
    "GBA": ["/content/GBA/"],
    "PINK1": ["/content/PINK1/"],
}


# ==============================================================================
# 1. Configuration & Reproducibility
# ==============================================================================
def get_args():
    parser = argparse.ArgumentParser(
        description="SupCon + Class-Balanced XBM (neg-only, same-class ignored)"
    )

    parser.add_argument(
        "--moco_m",
        type=float,
        default=0.995,
        help="Momentum for key encoder EMA update (0.99~0.999)",
    )

    # Experiment
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/content/drive/MyDrive/Final_paper/Model5",
        help="Save directory",
    )

    # Data
    parser.add_argument(
        "--max_samples", type=int, default=18000, help="Max samples per class"
    )
    parser.add_argument(
        "--test_ratio", type=float, default=1 / 3, help="Test split ratio"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.25, help="Validation split ratio"
    )

    # Training
    parser.add_argument("--img_size", type=int, default=128, help="Input image size")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size (must be 256 as requested)",
    )
    parser.add_argument("--epochs", type=int, default=150, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument(
        "--temp", type=float, default=0.1, help="Temperature for SupCon loss"
    )
    parser.add_argument(
        "--embed_dim", type=int, default=512, help="Projection head dimension"
    )
    parser.add_argument(
        "--patience", type=int, default=15, help="Early stopping patience"
    )

    # ---- Linear probe early stopping ----
    parser.add_argument(
        "--probe_eval_every",
        type=int,
        default=5,
        help="Run linear probe every N epochs (default=5).",
    )
    parser.add_argument(
        "--probe_epochs",
        type=int,
        default=3,
        help="Number of epochs for linear probe training (default=3).",
    )
    parser.add_argument(
        "--probe_patience",
        type=int,
        default=3,
        help="Early stopping patience measured in probe evaluations (default=3).",
    )
    parser.add_argument(
        "--probe_lr",
        type=float,
        default=1e-2,
        help="Learning rate for linear probe (default=1e-2).",
    )
    parser.add_argument(
        "--probe_weight_decay",
        type=float,
        default=0.0,
        help="Weight decay for linear probe (default=0.0).",
    )

    # ---- Speed ----
    parser.add_argument(
        "--channels_last",
        action="store_true",
        help="If set, use channels_last for input tensors (often faster on A100/H100).",
    )

    #### resume
    parser.add_argument(
        "--resume_ckpt",
        type=str,
        default="",
        help="Full checkpoint path to resume (saves model_q/k, optim, sched, scaler, xbm, epoch, rng).",
    )
    parser.add_argument(
        "--ckpt_name",
        type=str,
        default="resume_ckpt.pt",
        help="Filename for latest full checkpoint inside save_dir.",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=1,
        help="Save full checkpoint every N epochs (default=1).",
    )
    parser.add_argument(
        "--no_save_xbm",
        action="store_true",
        help="If set, do NOT store XBM buffers in checkpoint (file gets smaller, but exact resume breaks).",
    )

    # XBM (Cross-Batch Memory)
    parser.add_argument(
        "--xbm_start_epoch",
        type=int,
        default=3,
        help="Start using XBM from this epoch (default=3)",
    )
    parser.add_argument(
        "--xbm_total_capacity",
        type=int,
        default=65536,
        help="Total queue capacity (fp16). Big queue.",
    )
    parser.add_argument(
        "--xbm_sample_per_class",
        type=int,
        default=512,
        help="How many memory negatives to sample per class per step (controls compute).",
    )
    parser.add_argument(
        "--xbm_enqueue_both_views",
        action="store_true",
        help="If set, enqueue both views. Default is enqueue view1 only (recommended).",
    )

    # Jupyter/Colab
    if "ipykernel" in sys.modules:
        return parser.parse_args([])
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Speed mode (non-deterministic)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    logger.info(f"Random Seed set to {seed}")
    logger.info(
        f"cudnn.deterministic={torch.backends.cudnn.deterministic}, cudnn.benchmark={torch.backends.cudnn.benchmark}"
    )


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ==============================================================================
# Data validation & collate
# ==============================================================================
def validate_uint16_rgb_128(img: np.ndarray, filepath: str, img_size: int = 128):
    if img is None:
        raise ValueError("cv2.imread returned None")
    if img.dtype != np.uint16:
        raise ValueError(f"dtype must be uint16, got {img.dtype}")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"shape must be HxWx3, got {img.shape}")
    h, w = img.shape[:2]
    if (h, w) != (img_size, img_size):
        raise ValueError(f"size must be {(img_size, img_size)}, got {(h, w)}")


def log_skip(filepath: str, reason: Exception):
    print(
        f"[DATA_SKIP] {filepath} | {type(reason).__name__}: {reason}",
        file=sys.stderr,
        flush=True,
    )


def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)


# ==============================================================================
# 2. Preprocessing & Dataset
# ==============================================================================
class SafeInstanceNormalize:
    """(x - mu) / max(std, threshold) on CHW tensor."""

    def __init__(self, threshold: float = 0.01):
        self.threshold = float(threshold)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = torch.mean(tensor, dim=[1, 2], keepdim=True)
        std = torch.std(tensor, dim=[1, 2], keepdim=True)
        std = std.clamp_min(self.threshold)
        return (tensor - mean) / std


class InMemorySixteenBitDataset(Dataset):
    """
    - uint16 TIFF (128x128x3) -> float32 [0,1] -> CHW tensor
    - two_crops: return (view1, view2, label)
    - augment: rotation aug ON/OFF
    - preloaded_images: reuse RAM cache
    """

    def __init__(
        self,
        files: List[str],
        labels: List[int],
        img_size: int,
        two_crops: bool,
        augment: bool,
        preloaded_images=None,
    ):
        self.files = files
        self.labels = labels
        self.img_size = img_size
        self.two_crops = two_crops
        self.augment = augment

        if preloaded_images is not None:
            self.images = preloaded_images
        else:
            self.images = [None] * len(self.files)
            print(f"⚡ Loading {len(files)} images into RAM...")

            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
                list(
                    tqdm(
                        executor.map(self._load_file, range(len(files))),
                        total=len(files),
                        leave=False,
                    )
                )

        if self.augment:
            self.aug = transforms.RandomChoice(
                [
                    transforms.Lambda(lambda x: x),
                    transforms.Lambda(lambda x: torch.rot90(x, 1, [1, 2])),
                    transforms.Lambda(lambda x: torch.rot90(x, 2, [1, 2])),
                    transforms.Lambda(lambda x: torch.rot90(x, 3, [1, 2])),
                ]
            )
        else:
            self.aug = transforms.Lambda(lambda x: x)

    def _load_file(self, idx: int):
        filepath = self.files[idx]
        try:
            img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
            validate_uint16_rgb_128(img, filepath, self.img_size)

            # BGR -> RGB (cv2.cvtColor보다 빠르게)
            img = img[..., ::-1].copy()  # HWC uint16, RGB

            # torch tensor: CHW
            x = torch.from_numpy(img).permute(2, 0, 1).contiguous()  # uint16

            # uint16 -> float32 [0,1]
            x = x.to(torch.float32).div_(65535.0)

            # SafeInstanceNormalize를 여기서 1회만 수행 (rotation은 통계값 불변)
            mean = x.mean(dim=(1, 2), keepdim=True)
            std = x.std(dim=(1, 2), keepdim=True).clamp_min_(0.01)
            x = (x - mean) / std

            # 캐시 dtype: float16 권장 (RAM 절약 + H2D 복사량 감소)
            self.images[idx] = x.to(torch.float16)

        except Exception as e:
            log_skip(filepath, e)
            self.images[idx] = None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        x = self.images[idx]
        if x is None:
            return None

        label = int(self.labels[idx])

        if self.two_crops:
            v1 = self.aug(x)
            v2 = self.aug(x)
            return v1, v2, torch.tensor(label, dtype=torch.long)
        else:
            v = self.aug(x)
            return v, torch.tensor(label, dtype=torch.long)


# ==============================================================================
# 3. Data Manager
# ==============================================================================
def save_split_info(files, labels, save_dir, filename):
    path = os.path.join(save_dir, filename)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "label"])
        for fp, lb in zip(files, labels):
            writer.writerow([fp, lb])
    logger.info(f"Saved split info to {path}")


class BalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size, seed=42):
        self.labels = np.asarray(labels)
        self.batch_size = int(batch_size)
        self.seed = int(seed)

        self.classes = np.unique(self.labels).tolist()
        self.num_classes = len(self.classes)
        assert (
            self.batch_size % self.num_classes == 0
        ), "batch_size must be divisible by num_classes"
        self.per_class = self.batch_size // self.num_classes

        self.class_to_indices = {
            c: np.where(self.labels == c)[0].tolist() for c in self.classes
        }
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)

        pools = {}
        for c in self.classes:
            idxs = self.class_to_indices[c].copy()
            rng.shuffle(idxs)
            pools[c] = idxs

        # 만들 수 있는 배치 수 = 가장 적은 클래스 기준
        n_batches = min(len(pools[c]) // self.per_class for c in self.classes)

        for b in range(n_batches):
            batch = []
            for c in self.classes:
                start = b * self.per_class
                batch.extend(pools[c][start : start + self.per_class])
            rng.shuffle(batch)
            yield batch

    def __len__(self):
        return min(len(v) // self.per_class for v in self.class_to_indices.values())


def get_dataloaders(args):
    class_map = {"Control": 0, "SNCA": 1, "GBA": 2, "PINK1": 3}
    all_files, all_labels, stratify_labels = [], [], []
    stratify_counter = 0

    num_control_lines = len(DEFAULT_DATA_ROOTS["Control"])
    target_per_control_line = args.max_samples // num_control_lines

    logger.info("Processing Data Distribution...")

    for class_name, paths in DEFAULT_DATA_ROOTS.items():
        label = class_map[class_name]

        if class_name == "Control":
            for line_idx, line_path in enumerate(paths):
                files = glob.glob(
                    os.path.join(line_path, "**/*.[tT][iI][fF]*"), recursive=True
                )
                files.sort()
                random.shuffle(files)
                files = files[: min(len(files), target_per_control_line)]

                logger.info(f"  [{class_name} Line {line_idx+1}] Count: {len(files)}")
                all_files.extend(files)
                all_labels.extend([label] * len(files))
                stratify_labels.extend([stratify_counter] * len(files))
                stratify_counter += 1
        else:
            files = []
            for p in paths:
                files.extend(
                    glob.glob(os.path.join(p, "**/*.[tT][iI][fF]*"), recursive=True)
                )
            files.sort()
            random.shuffle(files)
            files = files[: min(len(files), args.max_samples)]

            logger.info(f"  [{class_name}] Count: {len(files)}")
            all_files.extend(files)
            all_labels.extend([label] * len(files))
            stratify_labels.extend([stratify_counter] * len(files))
            stratify_counter += 1

    X_temp, X_test, y_temp, y_test, strat_temp, strat_test = train_test_split(
        all_files,
        all_labels,
        stratify_labels,
        test_size=args.test_ratio,
        random_state=args.seed,
        stratify=stratify_labels,
    )
    X_train, X_val, y_train, y_val, _, _ = train_test_split(
        X_temp,
        y_temp,
        strat_temp,
        test_size=args.val_ratio,
        random_state=args.seed,
        stratify=strat_temp,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    save_split_info(X_train, y_train, args.save_dir, "train_split.csv")
    save_split_info(X_val, y_val, args.save_dir, "val_split.csv")
    save_split_info(X_test, y_test, args.save_dir, "test_split.csv")

    logger.info(
        f"Split -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
    )

    # Train: two crops + aug
    train_ds = InMemorySixteenBitDataset(
        X_train, y_train, args.img_size, two_crops=True, augment=True
    )

    # Val: two crops + NO aug
    val_ds = InMemorySixteenBitDataset(
        X_val, y_val, args.img_size, two_crops=True, augment=False
    )

    g = torch.Generator()
    g.manual_seed(args.seed)

    NUM_WORKERS = 0

    train_sampler = BalancedBatchSampler(
        train_ds.labels, batch_size=args.batch_size, seed=args.seed
    )

    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_skip_none,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=collate_skip_none,
        drop_last=True,
    )

    return train_loader, val_loader


# ===============================================================================
# 4.5 학습 재개
# =================================================================================


def _get_rng_state():
    st = {
        "py_random": random.getstate(),
        "np_random": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        st["torch_cuda"] = torch.cuda.get_rng_state_all()
    else:
        st["torch_cuda"] = None
    return st


def _set_rng_state(st):
    random.setstate(st["py_random"])
    np.random.set_state(st["np_random"])
    torch.set_rng_state(st["torch_cpu"])
    if torch.cuda.is_available() and st.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(st["torch_cuda"])


def _pack_xbm(trainer):
    # XBM ring buffers: emb/ptr/full 저장
    xbm_state = []
    for b in trainer.xbm.buffers:
        xbm_state.append(
            {
                "emb": b.emb.detach().cpu(),  # fp16 그대로 CPU로 내림
                "ptr": int(b.ptr),
                "full": bool(b.full),
            }
        )
    return xbm_state


def _unpack_xbm(trainer, xbm_state):
    for buf, st in zip(trainer.xbm.buffers, xbm_state):
        buf.emb.copy_(st["emb"].to(trainer.device, dtype=buf.emb.dtype))
        buf.ptr = int(st["ptr"])
        buf.full = bool(st["full"])


# ==============================================================================
# 4. Model
# ==============================================================================
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(weight_norm(nn.Conv2d(3, 64, 3, 2, 1)), nn.ReLU())
        self.layer2 = nn.Sequential(weight_norm(nn.Conv2d(64, 128, 3, 1, 1)), nn.ReLU())
        self.layer3 = nn.Sequential(
            weight_norm(nn.Conv2d(128, 256, 3, 1, 2, dilation=2)), nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            weight_norm(nn.Conv2d(256, 512, 3, 1, 4, dilation=4)), nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            weight_norm(nn.Conv2d(512, 1024, 3, 1, 2, dilation=2)), nn.ReLU()
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x, return_map=False):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        feat_map = self.layer5(x)
        pooled = self.gap(feat_map).view(feat_map.size(0), -1)
        return (pooled, feat_map) if return_map else pooled


class SupConModel(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.encoder = Encoder()
        self.projector = nn.Sequential(
            weight_norm(nn.Linear(1024, 1024)),
            nn.ReLU(),
            weight_norm(nn.Linear(1024, embed_dim)),
        )

    def forward(self, x):
        pooled = self.encoder(x, return_map=False)
        return F.normalize(self.projector(pooled), dim=1)


# ==============================================================================
# 5. Class-Balanced XBM (per-class quota) + loss
# ==============================================================================
class _RingEmbBuffer:
    def __init__(self, capacity: int, dim: int, device, dtype=torch.float16):
        self.capacity = int(capacity)
        self.dim = int(dim)
        self.device = device
        self.dtype = dtype

        self.emb = torch.empty(
            (self.capacity, self.dim), device=self.device, dtype=self.dtype
        )
        self.ptr = 0
        self.full = False

    @torch.no_grad()
    def reset(self):
        self.ptr = 0
        self.full = False

    @torch.no_grad()
    def enqueue(self, z: torch.Tensor):
        if z is None or z.numel() == 0:
            return
        z = z.detach().to(self.device, dtype=self.dtype)
        n = z.size(0)

        if n >= self.capacity:
            self.emb.copy_(z[-self.capacity :])
            self.ptr = 0
            self.full = True
            return

        end = self.ptr + n
        if end <= self.capacity:
            self.emb[self.ptr : end].copy_(z)
        else:
            first = self.capacity - self.ptr
            self.emb[self.ptr :].copy_(z[:first])
            rest = n - first
            self.emb[:rest].copy_(z[first:])

        self.ptr = (self.ptr + n) % self.capacity
        if self.ptr == 0:
            self.full = True

    def available(self) -> int:
        return self.capacity if self.full else self.ptr

    def get_all(self) -> Optional[torch.Tensor]:
        if self.full:
            return self.emb
        if self.ptr == 0:
            return None
        return self.emb[: self.ptr]


class ClassBalancedXBM:
    """
    - total_capacity를 클래스 수로 나눠 per-class quota 고정
    - get_balanced에서 step당 per-class로 n개 샘플링 (계산량 제어)
    """

    def __init__(
        self,
        total_capacity: int,
        num_classes: int,
        dim: int,
        device,
        dtype=torch.float16,
    ):
        total_capacity = int(total_capacity)
        num_classes = int(num_classes)

        base = total_capacity // num_classes
        rem = total_capacity % num_classes
        caps = [base + (1 if i < rem else 0) for i in range(num_classes)]

        self.num_classes = num_classes
        self.device = device
        self.buffers = [
            _RingEmbBuffer(capacity=caps[c], dim=dim, device=device, dtype=dtype)
            for c in range(num_classes)
        ]

    @torch.no_grad()
    def reset(self):
        for b in self.buffers:
            b.reset()

    @torch.no_grad()
    def enqueue(self, z: torch.Tensor, y: torch.Tensor):
        y = y.detach().to(self.device, dtype=torch.long)
        for c in range(self.num_classes):
            m = y == c
            if m.any():
                self.buffers[c].enqueue(z[m])

    def get_balanced(
        self, n_per_class: int
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        avail = [b.available() for b in self.buffers]
        if min(avail) == 0:
            return None, None

        n = int(n_per_class)
        n = min(n, min(avail))
        if n <= 0:
            return None, None

        zs, ys = [], []
        for c, b in enumerate(self.buffers):
            zc = b.get_all()
            # zc is not None because min(avail)>0
            idx = torch.randint(low=0, high=zc.size(0), size=(n,), device=zc.device)
            zs.append(zc[idx])
            ys.append(torch.full((n,), c, device=self.device, dtype=torch.long))

        return torch.cat(zs, dim=0), torch.cat(ys, dim=0)

    def ready(self, n_per_class: int) -> bool:
        n = int(n_per_class)
        return min(b.available() for b in self.buffers) >= n


def supcon_xbm_crossclass_only(
    z: torch.Tensor,
    y: torch.Tensor,
    z_mem: Optional[torch.Tensor],
    y_mem: Optional[torch.Tensor],
    temperature: float,
) -> torch.Tensor:
    """
    Policy (as requested):
    - positives: ONLY in-batch same-label (both views included)
    - memory same-label pairs: IGNORE (not positive, not negative) => exclude from denominator
    - memory different-label pairs: NEGATIVE-ONLY => include in denominator
    """
    device = z.device
    N = z.size(0)

    z = F.normalize(z, dim=1)

    if z_mem is None or y_mem is None:
        z_all = z
        y_all = y
        M = 0
    else:
        z_mem = F.normalize(z_mem, dim=1).detach()
        y_mem = y_mem.detach().to(device=device, dtype=torch.long)
        z_all = torch.cat([z, z_mem], dim=0)
        y_all = torch.cat([y, y_mem], dim=0)
        M = z_mem.size(0)

    # logits (compute in current autocast dtype, then cast to fp32 for stability ops)
    logits = (z @ z_all.t()).float() / float(temperature)

    # self comparisons only within batch part (first N columns)
    self_mask = torch.zeros((N, N + M), device=device, dtype=torch.bool)
    self_mask[:, :N] = torch.eye(N, device=device, dtype=torch.bool)

    # positives: same-label within batch only
    pos_mask = (y.view(-1, 1) == y_all.view(1, -1)) & (~self_mask)
    if M > 0:
        pos_mask[:, N:] = False  # memory positives 금지

    # denominator: batch(all except self) + memory(cross-class only)
    denom_mask = ~self_mask
    if M > 0:
        cross_class = y.view(-1, 1) != y_mem.view(1, -1)  # [N, M]
        denom_mask[:, N:] = cross_class  # same-class memory는 denom에서 제외 = IGNORE

    # stable masked log-softmax
    logits = logits - logits.max(dim=1, keepdim=True).values
    exp_logits = torch.exp(logits) * denom_mask.float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    pos_cnt = pos_mask.sum(dim=1)
    valid = pos_cnt > 0
    if valid.sum() == 0:
        return torch.zeros((), device=device, dtype=torch.float32)

    mean_log_prob_pos = (pos_mask.float() * log_prob).sum(dim=1) / pos_cnt.clamp_min(1)
    return -mean_log_prob_pos[valid].mean()


# ==============================================================================
# 6. Training
# ==============================================================================
class Trainer:
    def __init__(self, args, model, train_loader, val_loader):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_q = model.to(self.device)

        self.model_k = copy.deepcopy(self.model_q).to(self.device)
        for p in self.model_k.parameters():
            p.requires_grad = False
        self.model_k.eval()

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = optim.AdamW(self.model_q.parameters(), lr=args.lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

        self.num_classes = 4
        self.xbm = ClassBalancedXBM(
            total_capacity=args.xbm_total_capacity,
            num_classes=self.num_classes,
            dim=args.embed_dim,
            device=self.device,
            dtype=torch.float16,
        )

        self.best_loss = float("inf")
        self.patience_counter = 0

        # paths
        self.best_path = os.path.join(
            self.args.save_dir, "best_model.pt"
        )  # query weights only
        self.ckpt_path = os.path.join(
            self.args.save_dir, self.args.ckpt_name
        )  # full resume ckpt

        # ---- Linear probe early stopping state ----
        self.best_probe_acc = 0.0
        self.probe_patience_counter = 0

        # Probe datasets (reuse already-cached tensors in RAM, no reload)
        # - train_loader.dataset is your InMemorySixteenBitDataset with images cached (fp16)
        # - build single-view, no-aug datasets for probe using preloaded_images
        self.probe_train_ds = InMemorySixteenBitDataset(
            files=self.train_loader.dataset.files,
            labels=self.train_loader.dataset.labels,
            img_size=self.args.img_size,
            two_crops=False,
            augment=False,
            preloaded_images=self.train_loader.dataset.images,
        )
        self.probe_val_ds = InMemorySixteenBitDataset(
            files=self.val_loader.dataset.files,
            labels=self.val_loader.dataset.labels,
            img_size=self.args.img_size,
            two_crops=False,
            augment=False,
            preloaded_images=self.val_loader.dataset.images,
        )

        self.probe_train_loader = DataLoader(
            self.probe_train_ds,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_skip_none,
            drop_last=True,
        )
        self.probe_val_loader = DataLoader(
            self.probe_val_ds,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_skip_none,
            drop_last=False,
        )

    def save_full_checkpoint(self, epoch: int):
        os.makedirs(self.args.save_dir, exist_ok=True)

        ckpt = {
            "epoch": int(epoch),
            "best_loss": float(self.best_loss),
            "patience_counter": int(self.patience_counter),
            "model_q": self.model_q.state_dict(),
            "model_k": self.model_k.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "args": vars(self.args),
            "rng": _get_rng_state(),
            "best_probe_acc": float(self.best_probe_acc),
            "probe_patience_counter": int(self.probe_patience_counter),
        }

        if not self.args.no_save_xbm:
            ckpt["xbm"] = _pack_xbm(self)
        else:
            ckpt["xbm"] = None

        torch.save(ckpt, self.ckpt_path)
        tqdm.write(f"[CKPT] saved -> {self.ckpt_path} (epoch={epoch})")

    def load_full_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device)

        self.model_q.load_state_dict(ckpt["model_q"], strict=True)
        self.model_k.load_state_dict(ckpt["model_k"], strict=True)

        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.scaler.load_state_dict(ckpt["scaler"])

        self.best_loss = float(ckpt["best_loss"])
        self.patience_counter = int(ckpt["patience_counter"])

        self.best_probe_acc = float(ckpt.get("best_probe_acc", 0.0))
        self.probe_patience_counter = int(ckpt.get("probe_patience_counter", 0))

        if ckpt.get("xbm") is not None:
            _unpack_xbm(self, ckpt["xbm"])

        if ckpt.get("rng") is not None:
            _set_rng_state(ckpt["rng"])

        start_epoch = int(ckpt["epoch"]) + 1
        tqdm.write(
            f"[RESUME] loaded <- {path} | start_epoch={start_epoch}, best_loss={self.best_loss:.6f}"
        )
        return start_epoch

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        key = m*key + (1-m)*query
        """
        m = float(self.args.moco_m)
        for pq, pk in zip(self.model_q.parameters(), self.model_k.parameters()):
            pk.data.mul_(m).add_(pq.data, alpha=(1.0 - m))

    @torch.no_grad()
    def _sync_key_encoder(self):
        """
        start epoch에서 key encoder를 query encoder로 동기화 (drift 최소화)
        """
        for pq, pk in zip(self.model_q.parameters(), self.model_k.parameters()):
            pk.data.copy_(pq.data)

    def train_epoch(self, epoch: int):
        self.model_q.train()
        self.model_k.eval()

        total_loss = 0.0
        steps = 0

        if hasattr(self.train_loader, "batch_sampler") and hasattr(
            self.train_loader.batch_sampler, "set_epoch"
        ):
            self.train_loader.batch_sampler.set_epoch(epoch)

        pbar = tqdm(
            self.train_loader, desc=f"Train E{epoch}/{self.args.epochs}", leave=True
        )
        for batch in pbar:
            if batch is None:
                continue
            view1, view2, labels = batch
            if labels.numel() < 2:
                continue

            view1 = view1.to(self.device, non_blocking=True)
            view2 = view2.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            images_q = torch.cat([view1, view2], dim=0)  # [2B, C, H, W]
            labels2 = torch.cat([labels, labels], dim=0)  # [2B]

            # -------------------------
            # (A) Make keys for enqueue (NO GRAD, momentum encoder)
            # -------------------------
            z1_k, z2_k = None, None
            if epoch >= self.args.xbm_start_epoch:
                with torch.no_grad():
                    with torch.amp.autocast(
                        device_type=self.device.type, enabled=torch.cuda.is_available()
                    ):
                        # key embeddings for each view
                        z1_k = self.model_k(view1)  # [B, D]
                        if self.args.xbm_enqueue_both_views:
                            z2_k = self.model_k(view2)  # [B, D]

            # -------------------------
            # (B) Sample memory negatives (from XBM)
            # -------------------------
            z_mem, y_mem = None, None
            if epoch >= self.args.xbm_start_epoch and self.xbm.ready(
                self.args.xbm_sample_per_class
            ):
                z_mem, y_mem = self.xbm.get_balanced(self.args.xbm_sample_per_class)

            self.optimizer.zero_grad(set_to_none=True)

            # -------------------------
            # (C) Forward query encoder + loss
            # -------------------------
            with torch.amp.autocast(
                device_type=self.device.type, enabled=torch.cuda.is_available()
            ):
                z_q = self.model_q(images_q)  # [2B, D] (already normalized in model)
                loss = supcon_xbm_crossclass_only(
                    z=z_q,
                    y=labels2,
                    z_mem=z_mem,
                    y_mem=y_mem,
                    temperature=self.args.temp,
                )

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # -------------------------
            # (D) Momentum update key encoder AFTER optimizer step
            # -------------------------
            if epoch >= self.args.xbm_start_epoch:
                with torch.no_grad():
                    self._momentum_update_key_encoder()

            # -------------------------
            # (E) Enqueue keys (momentum embeddings)
            # -------------------------
            if epoch >= self.args.xbm_start_epoch:
                with torch.no_grad():
                    if self.args.xbm_enqueue_both_views:
                        # enqueue both views as keys
                        z_k_all = torch.cat([z1_k, z2_k], dim=0)  # [2B, D]
                        y_k_all = torch.cat([labels, labels], dim=0)  # [2B]
                        self.xbm.enqueue(z_k_all, y_k_all)
                    else:
                        # enqueue view1 keys only
                        self.xbm.enqueue(z1_k, labels)

            cur = float(loss.item())
            total_loss += cur
            steps += 1
            pbar.set_postfix({"loss": f"{cur:.4f}"})

        return 0.0 if steps == 0 else total_loss / steps

    def validate(self, epoch: int):
        # Validation: XBM OFF (as your original design)
        self.model_q.eval()
        total_loss = 0.0
        steps = 0

        pbar = tqdm(
            self.val_loader, desc=f"Val   E{epoch}/{self.args.epochs}", leave=True
        )
        with torch.no_grad():
            for batch in pbar:
                if batch is None:
                    continue
                view1, view2, labels = batch
                if labels.numel() < 2:
                    continue

                view1 = view1.to(self.device, non_blocking=True)
                view2 = view2.to(self.device, non_blocking=True)

                if self.args.channels_last:
                    view1 = view1.contiguous(memory_format=torch.channels_last)
                    view2 = view2.contiguous(memory_format=torch.channels_last)

                labels = labels.to(self.device, non_blocking=True)

                images = torch.cat([view1, view2], dim=0)
                labels2 = torch.cat([labels, labels], dim=0)

                with torch.amp.autocast(
                    device_type=self.device.type, enabled=torch.cuda.is_available()
                ):
                    z = self.model_q(images)
                    loss = supcon_xbm_crossclass_only(
                        z=z,
                        y=labels2,
                        z_mem=None,
                        y_mem=None,
                        temperature=self.args.temp,
                    )

                cur = float(loss.item())
                total_loss += cur
                steps += 1
                pbar.set_postfix({"val_loss": f"{cur:.4f}"})

        return 0.0 if steps == 0 else total_loss / steps

    def _seed_for_probe(self):
        # probe는 매번 동일 조건으로 비교되도록 seed를 고정
        s = int(self.args.seed) + 12345
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
        torch.cuda.manual_seed_all(s)

    @torch.no_grad()
    def _extract_feats(self, loader):
        """
        encoder pooled(1024) -> L2 normalize.
        returns: feats [N,1024] float32(cpu), labels [N] int64(cpu)
        """
        self.model_q.eval()

        feats_all = []
        ys_all = []
        for batch in loader:
            if batch is None:
                continue
            x, y = batch
            x = x.to(self.device, non_blocking=True)
            if self.args.channels_last:
                x = x.contiguous(memory_format=torch.channels_last)

            with torch.amp.autocast(
                device_type=self.device.type, enabled=torch.cuda.is_available()
            ):
                f = self.model_q.encoder(x, return_map=False)  # [B,1024]
                f = F.normalize(f, dim=1)

            feats_all.append(f.float().cpu())
            ys_all.append(y.cpu())

        if len(feats_all) == 0:
            return None, None
        return torch.cat(feats_all, 0), torch.cat(ys_all, 0)

    def run_linear_probe(self) -> float:
        """
        Frozen encoder. Train linear head for args.probe_epochs epochs on train_split,
        evaluate val accuracy. Seed-fixed for comparability.
        """
        self._seed_for_probe()

        # 1) feature extraction (fixed)
        trX, trY = self._extract_feats(self.probe_train_loader)
        vaX, vaY = self._extract_feats(self.probe_val_loader)
        if trX is None or vaX is None:
            return 0.0

        trX = trX.to(self.device, non_blocking=True)
        trY = trY.to(self.device, non_blocking=True)
        vaX = vaX.to(self.device, non_blocking=True)
        vaY = vaY.to(self.device, non_blocking=True)

        # 2) linear head (fresh, deterministic init because seed fixed)
        head = nn.Linear(trX.size(1), 4).to(self.device)
        opt = optim.SGD(
            head.parameters(),
            lr=float(self.args.probe_lr),
            momentum=0.9,
            weight_decay=float(self.args.probe_weight_decay),
        )
        crit = nn.CrossEntropyLoss()

        # 3) quick training (3 epochs default)
        head.train()
        for _ in range(int(self.args.probe_epochs)):
            # shuffle indices deterministically (seed already set)
            idx = torch.randperm(trX.size(0), device=self.device)
            bs = int(self.args.batch_size)
            for i in range(0, trX.size(0), bs):
                j = idx[i : i + bs]
                if j.numel() < 2:
                    continue
                logits = head(trX[j])
                loss = crit(logits, trY[j])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        # 4) eval
        head.eval()
        with torch.no_grad():
            pred = head(vaX).argmax(dim=1)
            acc = (pred == vaY).float().mean().item() * 100.0
        return float(acc)

    def run(self):
        logger.info(f"Starting Training on {self.device}")
        os.makedirs(self.args.save_dir, exist_ok=True)

        start_epoch = 1
        resumed_epoch = 0  # 0이면 새로 시작
        if self.args.resume_ckpt and os.path.exists(self.args.resume_ckpt):
            # load_full_checkpoint가 start_epoch를 반환
            start_epoch = self.load_full_checkpoint(self.args.resume_ckpt)
            resumed_epoch = start_epoch - 1  # ckpt에 저장된 마지막 epoch

        for epoch in range(start_epoch, self.args.epochs + 1):
            tqdm.write(f"\n===== Epoch {epoch}/{self.args.epochs} =====")

            if (
                epoch == self.args.xbm_start_epoch
                and resumed_epoch < self.args.xbm_start_epoch
            ):
                self.xbm.reset()
                self._sync_key_encoder()
                tqdm.write(
                    f"[XBM+MoCo] Enabled from epoch {epoch}. Queue reset + key encoder synced."
                )

            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            tqdm.write(
                f"Epoch {epoch:03d} | Train: {train_loss:.4f} | Val(SupCon): {val_loss:.4f}"
            )

            # ---- (NEW) Linear probe evaluation every N epochs ----
            probe_ran = epoch % self.args.probe_eval_every == 0
            probe_acc = None
            improved = False

            if probe_ran:
                probe_acc = self.run_linear_probe()
                tqdm.write(
                    f"[PROBE] epoch={epoch:03d} | val_acc={probe_acc:.2f}% | best={self.best_probe_acc:.2f}%"
                )

                improved = probe_acc > (self.best_probe_acc + 1e-6)
                if improved:
                    self.best_probe_acc = probe_acc
                    self.probe_patience_counter = 0
                    torch.save(
                        self.model_q.state_dict(), self.best_path
                    )  # best_model.pt by probe
                    tqdm.write(
                        f"  -> Saved Best Model by PROBE (best_probe_acc={self.best_probe_acc:.2f}%)"
                    )
                else:
                    self.probe_patience_counter += 1
                    tqdm.write(
                        f"  -> No probe improvement [{self.probe_patience_counter}/{self.args.probe_patience}]"
                    )

            # ---- scheduler step (same as before) ----
            if isinstance(self.scheduler, optim.lr_scheduler.CosineAnnealingLR):
                if self.scheduler.last_epoch < self.scheduler.T_max:
                    self.scheduler.step()
                else:
                    eta_min = float(self.scheduler.eta_min)
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = eta_min
            else:
                self.scheduler.step()

            # ---- full ckpt save always (same as before) ----
            if (epoch % self.args.save_every) == 0:
                self.save_full_checkpoint(epoch)

            # ---- (NEW) early stop based on probe patience (only after probe ran) ----
            if probe_ran and (self.probe_patience_counter >= self.args.probe_patience):
                tqdm.write("Early Stopping Triggered (by Linear Probe)")
                self.save_full_checkpoint(epoch)
                break


# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)

    train_dl, val_dl = get_dataloaders(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SupConModel(embed_dim=args.embed_dim).to(device)

    trainer = Trainer(args, model, train_dl, val_dl)
    trainer.run()
