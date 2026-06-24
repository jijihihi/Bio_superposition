# ==============================================================================
# RAM bank dataset (same spirit as yours)
# ==============================================================================
import io
import time
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
from tqdm.auto import tqdm

from sae_project.step02_logging_utils import SUPERCLASS_MAP, get_logger

try:
    import tifffile
except Exception:
    raise RuntimeError("tifffile not installed. pip install tifffile")


import io
import json
import os
import random
import time
from collections import defaultdict, deque

# tif decoder



logger = get_logger("data_bank")


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    import random

    random.seed(worker_seed)


def collate_skip_none(batch):
    """
    Collate function that:
    1. Skips None samples
    2. Handles mixed tensor/string tuples from InMemorySixteenBitDataset
       Dataset returns: (x, y, plate, line, uid) where plate/line/uid are strings
    """
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    # Check if batch contains tuples with strings (from InMemorySixteenBitDataset)
    # Dataset returns: (x, y, plate, line, uid)
    first = batch[0]
    if isinstance(first, (tuple, list)) and len(first) >= 3:
        # Check if element 2 is a string (plate field)
        if isinstance(first[2], str):
            # Separate tensor fields (x, y) from string fields (plate, line, uid)
            tensors = [(b[0], b[1]) for b in batch]  # (x, y)
            collated_tensors = default_collate(tensors)

            # String fields as lists
            plates = [b[2] for b in batch]
            lines = [b[3] for b in batch]
            uids = [b[4] for b in batch]

            return (*collated_tensors, plates, lines, uids)

    # Default: standard collate for tensor-only batches
    return default_collate(batch)


# 클래스 라인별로 이미지 균형있게 뽑는 함수


class StrictPlateBalancedBatchSamplerOnBank(Sampler[List[int]]):
    """
    bank/dataset 인덱스(0..N-1)를 반환하는 strict plate-uniform sampler
    """

    def __init__(self, bank, batch_size: int, seed: int):
        self.bank = bank
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self._epoch = 0

        # group by (superclass, line, plate) using bank arrays
        self.orig = defaultdict(list)
        for j in range(len(bank.images)):
            if bank.images[j] is None:
                continue
            line = bank.lines[j]
            sup = SUPERCLASS_MAP.get(line, line)
            plate = bank.plates[j]
            self.orig[(sup, line, plate)].append(j)

        self.line_plates = defaultdict(list)
        for sup, line, plate in self.orig.keys():
            self.line_plates[(sup, line)].append(plate)
        for k in self.line_plates:
            self.line_plates[k] = sorted(set(self.line_plates[k]))

        self.control_lines = ["Control_C4", "Control_C18", "Control_C19"]

    def __len__(self):
        # rough
        n = sum(len(v) for v in self.orig.values())
        return max(1, n // self.batch_size)

    def _make_working_deques(self, rng: random.Random):
        g = {}
        for k, lst in self.orig.items():
            lst2 = lst[:]
            rng.shuffle(lst2)
            g[k] = deque(lst2)
        return g

    def __iter__(self):
        self._epoch += 1
        rng = random.Random(self.seed + self._epoch)
        g = self._make_working_deques(rng)

        def take_many(sup: str, line: str, plate: str, n: int) -> List[int]:
            dq = g.get((sup, line, plate), None)
            if dq is None or len(dq) == 0:
                return []
            out = []
            for _ in range(n):
                if len(dq) == 0:
                    break
                out.append(dq.popleft())
            return out

        while True:
            bs = self.batch_size

            # superclass allocation
            per = bs // 4
            rem = bs - per * 4
            targets = {"Control": per, "SNCA": per, "GBA": per, "LRRK2": per}
            for k in ["Control", "SNCA", "GBA", "LRRK2"]:
                if rem <= 0:
                    break
                targets[k] += 1
                rem -= 1

            batch = []

            # Control split into 3 lines
            ctl = targets["Control"]
            pcl = ctl // 3
            remc = ctl - pcl * 3
            for li, line in enumerate(self.control_lines):
                need = pcl + (1 if li < remc else 0)
                plates = self.line_plates.get(("Control", line), [])
                if not plates or need <= 0:
                    continue
                plates = list(plates)
                rng.shuffle(plates)

                P = len(plates)
                per_plate = need // P
                remp = need - per_plate * P

                for p in plates:
                    if per_plate > 0:
                        batch.extend(take_many("Control", line, p, per_plate))
                for p in plates[:remp]:
                    batch.extend(take_many("Control", line, p, 1))

            # Mutations
            for sup in ["SNCA", "GBA", "LRRK2"]:
                need = targets[sup]
                line = sup
                plates = self.line_plates.get((sup, line), [])
                if not plates or need <= 0:
                    continue
                plates = list(plates)
                rng.shuffle(plates)

                P = len(plates)
                per_plate = need // P
                remp = need - per_plate * P

                for p in plates:
                    if per_plate > 0:
                        batch.extend(take_many(sup, line, p, per_plate))
                for p in plates[:remp]:
                    batch.extend(take_many(sup, line, p, 1))

            if len(batch) < self.batch_size:
                break

            yield batch[: self.batch_size]


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
            except Exception:
                pass

        logger.info(
            f"Preload done. bad={bad}/{len(ref_indices)} elapsed={(time.time()-t0)/60:.1f} min"
        )


class InMemorySixteenBitDataset(Dataset):
    def __init__(
        self,
        bank: InMemoryTarBank,
        indices_in_bank: List[int],
        img_size: int,
        augment: bool,
        explicit_4x_augment: bool = False,
    ):
        """
        Args:
            bank: InMemoryTarBank with preloaded images
            indices_in_bank: list of indices into the bank
            img_size: image size (e.g., 128)
            augment: if True, apply random rot90 (1 of 4 rotations randomly)
            explicit_4x_augment: if True, explicitly use all 4 rotations per image (4x data)
                                 This creates 4 samples per original image (indices 0-3 for each)
        """
        self.bank = bank
        self.ib = indices_in_bank
        self.img_size = int(img_size)
        self.augment = bool(augment)
        self.explicit_4x_augment = bool(explicit_4x_augment)

        if self.explicit_4x_augment:
            # Explicit 4x: each image appears 4 times with k=0,1,2,3
            # __len__ returns 4x, __getitem__ handles rotation
            self._base_len = len(self.ib)

        self.normalize = SafeInstanceNormalize(threshold=0.01)

    def __len__(self):
        if self.explicit_4x_augment:
            return len(self.ib) * 4
        return len(self.ib)

    def __getitem__(self, idx: int):
        if self.explicit_4x_augment:
            # idx = base_idx * 4 + rot_k
            base_idx = idx // 4
            rot_k = idx % 4
            j = self.ib[base_idx]
        else:
            j = self.ib[idx]
            rot_k = None

        img = self.bank.images[j]
        if img is None:
            return None

        y = torch.tensor(self.bank.labels[j], dtype=torch.long)
        plate = self.bank.plates[j]
        line = self.bank.lines[j]
        uid = self.bank.uids[j]

        x = img.astype(np.float32) / 65535.0
        x = torch.from_numpy(x).permute(2, 0, 1)  # C,H,W

        # Apply rotation
        if self.explicit_4x_augment:
            # Explicit: use the specific rotation k
            if rot_k > 0:
                x = torch.rot90(x, rot_k, [1, 2])
        elif self.augment:
            # Random: pick one of 4 rotations
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                x = torch.rot90(x, k, [1, 2])

        x = self.normalize(x)
        return x, y, plate, line, uid


# ==============================================================================
# Split CSV helpers (reuse your existing split)
# ==============================================================================
def load_split_csv(csv_path: str) -> List[str]:
    df = np.genfromtxt(csv_path, delimiter=",", dtype=str, skip_header=1)
    if df.ndim == 1 and len(df) > 0:
        return [df[0]]
    return df[:, 0].tolist()
