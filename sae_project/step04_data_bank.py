# ==============================================================================
# RAM bank dataset (same spirit as yours)
# ==============================================================================
import os
import io
import time
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
from tqdm.auto import tqdm

from sae_project.step02_logging_utils import get_logger

try:
    import tifffile
except Exception:
    raise RuntimeError("tifffile not installed. pip install tifffile")

logger = get_logger("data_bank")


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    import random
    random.seed(worker_seed)


def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)


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
    def __init__(self, threshold: float = 0.01):
        self.threshold = float(threshold)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = tensor.mean(dim=[1, 2], keepdim=True)
        std = tensor.std(dim=[1, 2], keepdim=True).clamp_min(self.threshold)
        return (tensor - mean) / std


class InMemoryTarBank:
    """
    shared RAM bank: preload tar samples into memory as uint16 numpy arrays
    """
    def __init__(self, refs, ref_indices: List[int], img_size: int):
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
            except Exception:
                pass

        logger.info(f"Preload done. bad={bad}/{len(ref_indices)} elapsed={(time.time()-t0)/60:.1f} min")


class InMemorySixteenBitDataset(Dataset):
    def __init__(self, bank: InMemoryTarBank, indices_in_bank: List[int], img_size: int, augment: bool):
        self.bank = bank
        self.ib = indices_in_bank
        self.img_size = int(img_size)
        self.augment = bool(augment)

        if self.augment:
            aug = transforms.RandomChoice([
                transforms.Lambda(lambda x: x),
                transforms.Lambda(lambda x: torch.rot90(x, 1, [1, 2])),
                transforms.Lambda(lambda x: torch.rot90(x, 2, [1, 2])),
                transforms.Lambda(lambda x: torch.rot90(x, 3, [1, 2])),
            ])
        else:
            aug = transforms.Lambda(lambda x: x)

        self.transform = transforms.Compose([
            aug,
            SafeInstanceNormalize(threshold=0.01)
        ])

    def __len__(self):
        return len(self.ib)

    def __getitem__(self, idx: int):
        j = self.ib[idx]
        img = self.bank.images[j]
        if img is None:
            return None

        y = torch.tensor(self.bank.labels[j], dtype=torch.long)
        plate = self.bank.plates[j]
        line = self.bank.lines[j]
        uid = self.bank.uids[j]

        x = (img.astype(np.float32) / 65535.0)
        x = torch.from_numpy(x).permute(2, 0, 1)  # C,H,W
        x = self.transform(x)
        return x, y, plate, line, uid


# ==============================================================================
# Split CSV helpers (reuse your existing split)
# ==============================================================================
def load_split_csv(csv_path: str) -> List[str]:
    df = np.genfromtxt(csv_path, delimiter=",", dtype=str, skip_header=1)
    if df.ndim == 1 and len(df) > 0:
        return [df[0]]
    return df[:, 0].tolist()
