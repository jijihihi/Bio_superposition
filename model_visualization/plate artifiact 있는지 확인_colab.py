# UMAP으로 같은 클래스인데 plate끼리 뭉쳐있는지 확인

# ==============================
# Fast Linear Evaluation (Frozen Encoder) with Feature Cache
# + Plate mixing verification within each class (R2 + Silhouette + Permutation test)
# - encoder forward ONLY ONCE (train/val/test) -> cache features
# - bias=False linear head (head training part currently not used in this script)
# - encoder renorm once + head renorm every step (head training part currently not used)
# - GAP feature L2 normalize -> classifier (state-only)
# - UMAP visualization within each class (colored by plate)
# - NEW: within-class plate mixing quantitative tests
# ==============================

import glob
import io
import json
import os
import pickle
import random
import re
import sys
import tarfile
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, silhouette_score)
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from tqdm.auto import tqdm

cv2.setNumThreads(0)


# -------------------------
# Config
# -------------------------
@dataclass
class CFG:
    # 체크포인트 경로 (Google Drive)
    SAVE_DIR: str = (
        "/content/drive/MyDrive/Final_paper/Model_MoCoXBM_PlateLP_LRRK2_L2 norm_hidden2048_resume"
    )
    CKPT_NAME: str = "best_model.pt"

    # Colab에서 zip 풀어놓은 tar shard 경로
    WDS_ROOT: str = "/content/wds_shards"

    IMG_SIZE: int = 128
    NUM_CLASSES: int = 4
    CLASS_NAMES = ["Control", "SNCA", "GBA", "LRRK2"]

    # MUST match training encoder (moco only - 02.07이코드결정)
    FEAT_DIM: int = 512
    BLOCKS: Tuple[int, int, int, int] = (2, 2, 2, 3)
    DILATIONS: Tuple[int, int, int, int] = (1, 1, 1, 1)
    REFINE_BLOCKS: int = 1

    # extraction (encoder forward)
    IMG_BATCH_SIZE: int = 512
    IMG_NUM_WORKERS: int = 0
    PIN_MEMORY: bool = True
    USE_BF16_EXTRACT: bool = True
    USE_CHANNELS_LAST: bool = True

    SEED: int = 42

    # -------------------------
    # Plate mixing quantitative tests
    # -------------------------
    RUN_PLATE_MIXING_TEST: bool = True
    MIN_SAMPLES_PER_PLATE: int = 5
    MAX_SAMPLES_PER_CLASS_METRIC: int = 20000
    PLATE_R2_N_PERM: int = 200
    PLATE_SIL_N_PERM: int = 50
    SILHOUETTE_SAMPLE_SIZE: int = 5000
    METRIC_SEED: int = 123


# 디렉토리명 -> superclass 매핑
SUPERCLASS_MAP = {
    "Control_C4": "Control",
    "Control_C18": "Control",
    "Control_C19": "Control",
    "SNCA": "SNCA",
    "GBA": "GBA",
    "LRRK2": "LRRK2",
}
LABEL_MAP = {"Control": 0, "SNCA": 1, "GBA": 2, "LRRK2": 3}

# plate 디렉토리 패턴: plate=003001
PLATE_DIR_RE = re.compile(r"plate=(\d{6})")


# -------------------------
# Utils
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def parse_plate_id(filepath: str) -> str:
    base = os.path.basename(filepath)
    return base.split("_")[0]  # "004001" 같은 plate id


def collate_skip_none_with_meta(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    xs = torch.stack([b[0] for b in batch], dim=0)
    ys = torch.stack([b[1] for b in batch], dim=0)
    plates = [b[2] for b in batch]
    paths = [b[3] for b in batch]
    return xs, ys, plates, paths


class SafeInstanceNormalize:
    def __init__(self, threshold: float = 0.01):
        self.threshold = float(threshold)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = tensor.mean(dim=[1, 2], keepdim=True)
        std = tensor.std(dim=[1, 2], keepdim=True).clamp_min(self.threshold)
        return (tensor - mean) / std


class TarShardDataset(Dataset):
    """
    Colab에서 zip -> tar shard 구조를 직접 읽는 Dataset.
    디렉토리 구조: {wds_root}/{class_name}/plate={plate}/{class_name}_plate={plate}-{shard}.tar
    tar 안: {key}.tif + {key}.json
    pkl 인덱스 불필요 — tarfile로 직접 스캔해서 offset 기반 인덱스 생성.
    """

    def __init__(self, wds_root: str, img_size: int):
        self.img_size = int(img_size)
        self.transform = SafeInstanceNormalize(0.01)
        self._fhs = {}  # tar file handles (lazy open)

        # 인덱스 빌드: (tar_path, tif_member_name, offset_data, size, label, plate) 목록
        self.samples = []
        print(f"[TarShardDataset] Scanning {wds_root} ...")

        for class_dir in sorted(os.listdir(wds_root)):
            class_path = os.path.join(wds_root, class_dir)
            if not os.path.isdir(class_path):
                continue

            superclass = SUPERCLASS_MAP.get(class_dir, None)
            if superclass is None:
                print(f"  [skip] Unknown class dir: {class_dir}")
                continue
            label = LABEL_MAP[superclass]

            for plate_dir in sorted(os.listdir(class_path)):
                plate_path = os.path.join(class_path, plate_dir)
                if not os.path.isdir(plate_path):
                    continue

                m = PLATE_DIR_RE.search(plate_dir)
                plate = m.group(1) if m else plate_dir

                tar_files = sorted(glob.glob(os.path.join(plate_path, "*.tar")))
                for tar_path in tar_files:
                    try:
                        with tarfile.open(tar_path, "r") as tf:
                            for member in tf.getmembers():
                                if member.name.endswith(".tif"):
                                    self.samples.append(
                                        (
                                            tar_path,
                                            member.name,
                                            member.offset_data,
                                            member.size,
                                            label,
                                            plate,
                                        )
                                    )
                    except Exception as e:
                        print(f"  [warn] Failed to scan {tar_path}: {e}")

        print(f"[TarShardDataset] Total samples: {len(self.samples)}")

        # 클래스별 통계
        from collections import Counter

        label_cnt = Counter(s[4] for s in self.samples)
        inv_label = {v: k for k, v in LABEL_MAP.items()}
        for lb, cnt in sorted(label_cnt.items()):
            print(f"  {inv_label.get(lb, lb)}: {cnt}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        tar_path, member_name, offset, size, label, plate = self.samples[idx]

        try:
            fh = self._fhs.get(tar_path, None)
            if fh is None:
                fh = open(tar_path, "rb", buffering=0)
                self._fhs[tar_path] = fh
            fh.seek(offset)
            tif_bytes = fh.read(size)

            img = tifffile.imread(io.BytesIO(tif_bytes))
            validate_uint16_rgb_128(
                img, filepath=f"{tar_path}:{member_name}", img_size=self.img_size
            )

            img = img.astype(np.float32) / 65535.0
            x = torch.from_numpy(img).permute(2, 0, 1)
            x = self.transform(x)

            uid = f"{tar_path}:{member_name}"
            return x, torch.tensor(label, dtype=torch.long), plate, uid

        except Exception as e:
            log_skip(f"{tar_path}:{member_name}", e)
            return None

    def __del__(self):
        for fh in self._fhs.values():
            try:
                fh.close()
            except:
                pass


@torch.no_grad()
def renorm_unit_per_out_channel_(model: nn.Module, eps: float = 1e-12):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            w = m.weight.data
            n = w.flatten(1).norm(dim=1, keepdim=True).clamp_min(eps)
            w.div_(n.view(-1, 1, 1, 1))
        elif isinstance(m, nn.Linear):
            w = m.weight.data
            n = w.norm(dim=1, keepdim=True).clamp_min(eps)
            w.div_(n)


@torch.no_grad()
def extract_features(
    encoder: nn.Module,
    loader: DataLoader,
    device: torch.device,
    desc: str,
    use_bf16: bool,
    use_channels_last: bool,
):
    encoder.eval()
    feats, labels = [], []
    plates_all, paths_all = [], []

    pbar = tqdm(loader, desc=desc, leave=False)
    for batch in pbar:
        if batch is None:
            continue
        x, y, plates, paths = batch
        if y.numel() < 1:
            continue

        x = x.to(device, non_blocking=True)
        if use_channels_last and device.type == "cuda":
            x = x.contiguous(memory_format=torch.channels_last)

        if use_bf16 and device.type == "cuda":
            with torch.amp.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=True
            ):
                f = encoder(x)
        else:
            f = encoder(x)

        f = F.normalize(f, dim=1)  # state-only (unit norm)

        feats.append(f.detach().to("cpu", dtype=torch.float16))
        labels.append(y.detach().to("cpu"))
        plates_all.extend(plates)
        paths_all.extend(paths)

    X = torch.cat(feats, dim=0)
    Y = torch.cat(labels, dim=0)
    return X, Y, plates_all, paths_all


# -------------------------
# Encoder definition (MUST match training architecture)
# -------------------------
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


OUT_DIM = 512


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


# -------------------------
# Load checkpoint (robust)
# -------------------------
def load_state_dict(path: str, map_location="cpu") -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location=map_location)
    if (
        isinstance(obj, dict)
        and "state_dict" in obj
        and isinstance(obj["state_dict"], dict)
    ):
        return obj["state_dict"]
    if isinstance(obj, dict):
        return obj
    raise RuntimeError("Unsupported checkpoint format.")


def extract_encoder_sd(full_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in full_sd.items():
        kk = k
        if kk.startswith("module."):
            kk = kk[len("module.") :]
        if ".encoder." in kk:
            out[kk.split(".encoder.", 1)[1]] = v
    if len(out) > 0:
        return out

    out = {}
    for k, v in full_sd.items():
        kk = k
        if kk.startswith("module."):
            kk = kk[len("module.") :]
        if kk.startswith("encoder."):
            out[kk.replace("encoder.", "", 1)] = v
    if len(out) > 0:
        return out

    return {
        k[len("module.") :] if k.startswith("module.") else k: v
        for k, v in full_sd.items()
    }


# -------------------------
# UMAP utils
# -------------------------
def ensure_umap():
    try:
        import umap

        return umap
    except ImportError:
        raise ImportError("umap-learn이 없습니다. Colab이면: !pip install umap-learn")


def plot_umap_by_plate_within_class(
    X: torch.Tensor,
    Y: torch.Tensor,
    plates: list,
    class_names: list,
    seed: int = 42,
    n_neighbors: int = 30,
    min_dist: float = 0.10,
    metric: str = "cosine",
    max_legend: int = 25,
    point_size: float = 6.0,
    alpha: float = 0.75,
    save_dir: str = None,
):
    umap = ensure_umap()
    Xn = X.float().numpy()
    Yn = Y.long().numpy()

    for c, cname in enumerate(class_names):
        idx = np.where(Yn == c)[0]
        if len(idx) < 10:
            print(f"[UMAP] Skip {cname}: too few samples ({len(idx)})")
            continue

        Xc = Xn[idx]
        Pc = [plates[i] for i in idx]

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=seed,
        )
        emb = reducer.fit_transform(Xc)

        cnt = Counter(Pc)
        uniq_plates = [p for p, _ in cnt.most_common()]

        cmap_name = "tab20" if len(uniq_plates) <= 20 else "hsv"
        cmap = plt.get_cmap(cmap_name)

        plate_to_color = {}
        for i, p in enumerate(uniq_plates):
            plate_to_color[p] = cmap(i / max(1, len(uniq_plates) - 1))

        plt.figure(figsize=(9, 8))
        for p in uniq_plates:
            p_idx = [i for i, pp in enumerate(Pc) if pp == p]
            pts = emb[p_idx]
            plt.scatter(
                pts[:, 0],
                pts[:, 1],
                s=point_size,
                c=[plate_to_color[p]],
                alpha=alpha,
                label=f"{p} (n={cnt[p]})",
            )

        plt.title(f"UMAP within class: {cname} (colored by plate)")
        plt.xlabel("UMAP-1")
        plt.ylabel("UMAP-2")

        if len(uniq_plates) <= max_legend:
            plt.legend(loc="best", fontsize=8, markerscale=2)
        else:
            handles, labels = plt.gca().get_legend_handles_labels()
            plt.legend(
                handles[:max_legend],
                labels[:max_legend],
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                fontsize=8,
                markerscale=2,
                title=f"Top {max_legend} plates",
            )
            plt.tight_layout(rect=[0, 0, 0.80, 1])

        plt.tight_layout()
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            out = os.path.join(save_dir, f"UMAP_{cname}_by_plate.png")
            plt.savefig(out, dpi=200)
            print("[UMAP] saved:", out)
        plt.show()


# -------------------------
# NEW: Plate mixing quantitative tests
# -------------------------
def _filter_by_min_plate_samples(
    indices: np.ndarray, plates: List[str], min_count: int
) -> np.ndarray:
    sub_plates = [plates[i] for i in indices]
    cnt = Counter(sub_plates)
    keep = [i for i in indices if cnt[plates[i]] >= min_count]
    return np.array(keep, dtype=np.int64)


def _subsample_indices(indices: np.ndarray, max_n: int, seed: int) -> np.ndarray:
    if len(indices) <= max_n:
        return indices
    rng = np.random.default_rng(seed)
    return rng.choice(indices, size=max_n, replace=False)


def _encode_plates(plates_sub: List[str]) -> Tuple[np.ndarray, List[str]]:
    # returns codes 0..P-1 and unique plate list in code order
    uniq, inv = np.unique(np.array(plates_sub), return_inverse=True)
    return inv.astype(np.int64), uniq.tolist()


def _plate_r2_torch(
    X: torch.Tensor, plate_codes: torch.Tensor, n_plates: int
) -> Tuple[float, float, float]:
    """
    X: (N,D) float32/float16, any device
    plate_codes: (N,) int64 in [0, n_plates-1]
    R2 = 1 - within_ss / total_ss
    """
    X = X.float()
    global_mean = X.mean(dim=0, keepdim=True)
    total_ss = ((X - global_mean) ** 2).sum()

    counts = (
        torch.bincount(plate_codes, minlength=n_plates)
        .clamp_min(1)
        .float()
        .unsqueeze(1)
    )  # (P,1)
    sum_x = torch.zeros((n_plates, X.shape[1]), device=X.device, dtype=X.dtype)
    sum_x.index_add_(0, plate_codes, X)
    plate_mean = sum_x / counts

    diff = X - plate_mean[plate_codes]
    within_ss = (diff**2).sum()

    r2 = 0.0
    if total_ss.item() > 0:
        r2 = float((1.0 - (within_ss / total_ss)).item())
    return r2, float(within_ss.item()), float(total_ss.item())


def _perm_test_r2(
    X: torch.Tensor, plate_codes: torch.Tensor, n_plates: int, n_perm: int, seed: int
) -> Tuple[float, float, float, List[float]]:
    """
    returns (obs_r2, p_value, null_mean, null_list)
    """
    device = X.device
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    obs_r2, _, _ = _plate_r2_torch(X, plate_codes, n_plates)

    null = []
    for _ in tqdm(range(n_perm), desc="perm(R2)", leave=False):
        perm = torch.randperm(plate_codes.numel(), generator=g, device=device)
        pc = plate_codes[perm]
        r2p, _, _ = _plate_r2_torch(X, pc, n_plates)
        null.append(r2p)

    # p-value: P(null >= obs)
    ge = sum([1 for v in null if v >= obs_r2])
    p = (ge + 1.0) / (n_perm + 1.0)
    null_mean = float(np.mean(null)) if len(null) > 0 else float("nan")
    return obs_r2, float(p), null_mean, null


def _silhouette_by_plate(
    X_np: np.ndarray,
    plate_codes_np: np.ndarray,
    metric: str,
    sample_size: int,
    seed: int,
) -> float:
    # sklearn silhouette_score: if sample_size is not None -> random subset approximation
    # needs >=2 labels
    if len(np.unique(plate_codes_np)) < 2:
        return float("nan")
    n = len(plate_codes_np)
    if n < 3:
        return float("nan")
    ss = min(sample_size, n) if sample_size is not None else None
    return float(
        silhouette_score(
            X_np, plate_codes_np, metric=metric, sample_size=ss, random_state=seed
        )
    )


def _perm_test_silhouette(
    X_np: np.ndarray,
    plate_codes_np: np.ndarray,
    metric: str,
    sample_size: int,
    n_perm: int,
    seed: int,
) -> Tuple[float, float, float, List[float]]:
    rng = np.random.default_rng(seed)
    obs = _silhouette_by_plate(
        X_np, plate_codes_np, metric=metric, sample_size=sample_size, seed=seed
    )
    if np.isnan(obs):
        return obs, float("nan"), float("nan"), []

    null = []
    for _ in tqdm(range(n_perm), desc="perm(Sil)", leave=False):
        perm = rng.permutation(plate_codes_np)
        v = _silhouette_by_plate(
            X_np, perm, metric=metric, sample_size=sample_size, seed=seed
        )
        null.append(v)

    ge = sum([1 for v in null if v >= obs])
    p = (ge + 1.0) / (n_perm + 1.0)
    null_mean = float(np.mean(null)) if len(null) > 0 else float("nan")
    return obs, float(p), null_mean, null


def evaluate_plate_mixing(
    X: torch.Tensor,
    Y: torch.Tensor,
    plates: List[str],
    cfg: CFG,
    split_name: str,
    save_dir: str,
):
    """
    같은 클래스 내부에서 plate별로 feature가 뭉치는지(plate effect) 정량 검증.
    - R2(effect size) + permutation p-value
    - Silhouette(by plate) + permutation p-value(비싸서 perm 횟수 적게)
    """
    os.makedirs(save_dir, exist_ok=True)
    Yn = Y.long().numpy()

    rows = []
    print("\n" + "=" * 80)
    print(
        f"[PLATE MIXING TEST] split={split_name}  (computed in ORIGINAL feature space, not UMAP)"
    )
    print("=" * 80)

    for c, cname in enumerate(cfg.CLASS_NAMES):
        idx = np.where(Yn == c)[0]
        n0 = len(idx)
        if n0 < 10:
            print(f"[{cname}] skip: too few samples ({n0})")
            continue

        # 1) min plate count filter
        idx = _filter_by_min_plate_samples(idx, plates, cfg.MIN_SAMPLES_PER_PLATE)
        n1 = len(idx)
        if n1 < 10:
            print(f"[{cname}] skip after min-plate filter: {n1} samples")
            continue

        # 2) subsample for speed (especially permutations)
        idx = _subsample_indices(
            idx, cfg.MAX_SAMPLES_PER_CLASS_METRIC, seed=cfg.METRIC_SEED + c
        )
        n = len(idx)

        plates_sub = [plates[i] for i in idx]
        cnt = Counter(plates_sub)
        n_plates = len(cnt)

        if n_plates < 2:
            print(f"[{cname}] skip: only one plate after filter/subsample")
            continue

        # summary
        counts = np.array(sorted(cnt.values()))
        print(
            f"\n[{cname}] N={n} (orig {n0} -> after_filter {n1}) | plates={n_plates} | "
            f"count(min/med/max)={counts.min()}/{int(np.median(counts))}/{counts.max()}"
        )

        # plate codes
        plate_codes_np, uniq_plates = _encode_plates(plates_sub)
        plate_codes_t = torch.from_numpy(plate_codes_np).long()

        # X on device (CPU is fine; GPU optional if you want)
        Xc = X[idx].float()  # (N,D) on CPU tensor

        # ---- R2 + perm test ----
        obs_r2, p_r2, null_r2_mean, null_r2 = _perm_test_r2(
            Xc,
            plate_codes_t,
            n_plates=len(uniq_plates),
            n_perm=cfg.PLATE_R2_N_PERM,
            seed=cfg.METRIC_SEED + 1000 + c,
        )
        print(
            f"  R2(effect size; plate explains variance): {obs_r2:.6f} | "
            f"perm_p={p_r2:.4f} | null_mean={null_r2_mean:.6f}"
        )

        # ---- Silhouette + perm test (cosine) ----
        # silhouette은 sklearn이 numpy 필요
        Xc_np = Xc.numpy()
        obs_sil, p_sil, null_sil_mean, null_sil = _perm_test_silhouette(
            Xc_np,
            plate_codes_np,
            metric="cosine",
            sample_size=cfg.SILHOUETTE_SAMPLE_SIZE,
            n_perm=cfg.PLATE_SIL_N_PERM,
            seed=cfg.METRIC_SEED + 2000 + c,
        )
        if np.isnan(obs_sil):
            print("  Silhouette(by plate): nan (plates<2 or too few samples)")
        else:
            print(
                f"  Silhouette(by plate, cosine; approx sample={min(cfg.SILHOUETTE_SAMPLE_SIZE, n)}): "
                f"{obs_sil:.6f} | perm_p={p_sil:.4f} | null_mean={null_sil_mean:.6f}"
            )

        # save hist plots (optional)
        try:
            plt.figure(figsize=(7, 4))
            plt.hist(null_r2, bins=30)
            plt.axvline(obs_r2, linewidth=2)
            plt.title(f"Null dist (R2) - {split_name} - {cname}")
            plt.tight_layout()
            outp = os.path.join(save_dir, f"plate_perm_R2_{split_name}_{cname}.png")
            plt.savefig(outp, dpi=200)
            plt.close()
        except Exception as e:
            print("[warn] failed to save R2 hist:", e)

        if len(null_sil) > 0 and not np.isnan(obs_sil):
            try:
                plt.figure(figsize=(7, 4))
                plt.hist(null_sil, bins=30)
                plt.axvline(obs_sil, linewidth=2)
                plt.title(f"Null dist (Silhouette) - {split_name} - {cname}")
                plt.tight_layout()
                outp = os.path.join(
                    save_dir, f"plate_perm_Sil_{split_name}_{cname}.png"
                )
                plt.savefig(outp, dpi=200)
                plt.close()
            except Exception as e:
                print("[warn] failed to save Sil hist:", e)

        rows.append(
            {
                "split": split_name,
                "class": cname,
                "N_used": n,
                "N_orig": n0,
                "N_after_minplate": n1,
                "num_plates": n_plates,
                "plate_count_min": int(counts.min()),
                "plate_count_median": int(np.median(counts)),
                "plate_count_max": int(counts.max()),
                "R2_obs": obs_r2,
                "R2_perm_p": p_r2,
                "R2_null_mean": null_r2_mean,
                "Sil_obs": obs_sil,
                "Sil_perm_p": p_sil,
                "Sil_null_mean": null_sil_mean,
            }
        )

    df = pd.DataFrame(rows)
    out_csv = os.path.join(save_dir, f"plate_mixing_metrics_{split_name}.csv")
    df.to_csv(out_csv, index=False)
    print("\n[PLATE MIXING TEST] saved:", out_csv)
    print(df)


# -------------------------
# Main
# -------------------------
def run_plate_eval(cfg: CFG):
    set_seed(cfg.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        "Device:",
        device,
        "| GPU:",
        torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
    )

    # ---- 1) tar shard에서 전체 데이터 로드 ----
    ds = TarShardDataset(cfg.WDS_ROOT, cfg.IMG_SIZE)
    if len(ds) == 0:
        raise RuntimeError(f"No samples found in {cfg.WDS_ROOT}")

    pin = (device.type == "cuda") and cfg.PIN_MEMORY
    nw = int(cfg.IMG_NUM_WORKERS)
    loader = DataLoader(
        ds,
        batch_size=cfg.IMG_BATCH_SIZE,
        shuffle=False,
        num_workers=nw,
        pin_memory=pin,
        collate_fn=collate_skip_none_with_meta,
    )

    # ---- 2) 체크포인트 로드 ----
    ckpt_path = os.path.join(cfg.SAVE_DIR, cfg.CKPT_NAME)
    print(f"Loading checkpoint: {ckpt_path}")
    full_sd = load_state_dict(ckpt_path, map_location="cpu")
    enc_sd = extract_encoder_sd(full_sd)

    encoder = Encoder(
        blocks=cfg.BLOCKS, dilations=cfg.DILATIONS, refine_blocks=cfg.REFINE_BLOCKS
    )
    missing, unexpected = encoder.load_state_dict(enc_sd, strict=False)
    print(f"Encoder loaded. missing={len(missing)}, unexpected={len(unexpected)}")
    if len(missing) > 0:
        print(f"  ⚠️ Missing keys: {missing[:10]}")
    if len(unexpected) > 0:
        print(f"  ⚠️ Unexpected keys: {unexpected[:10]}")

    # enforce unit-norm weights once
    renorm_unit_per_out_channel_(encoder)

    encoder = encoder.to(device)
    if cfg.USE_CHANNELS_LAST and device.type == "cuda":
        encoder = encoder.to(memory_format=torch.channels_last)

    # ---- 3) feature 캐시 (같은 모델로 재실행 시 빠르게) ----
    cache_dir = os.path.join(cfg.SAVE_DIR, "plate_eval_cache")
    os.makedirs(cache_dir, exist_ok=True)

    cache_key = f"{cfg.CKPT_NAME}__feat{cfg.FEAT_DIM}__blocks{cfg.BLOCKS}__dil{cfg.DILATIONS}__ref{cfg.REFINE_BLOCKS}".replace(
        " ", ""
    )
    cache_path = os.path.join(cache_dir, f"Xy_all__{cache_key}.pt")

    if os.path.exists(cache_path):
        print(f"[cache] Loading cached features: {cache_path}")
        obj = torch.load(cache_path, map_location="cpu")
        Xall, Yall, Pall = obj["X"], obj["Y"], obj["plate"]
    else:
        print("[extract] Extracting features ...")
        Xall, Yall, Pall, _ = extract_features(
            encoder,
            loader,
            device,
            "Extract ALL Features",
            cfg.USE_BF16_EXTRACT,
            cfg.USE_CHANNELS_LAST,
        )
        torch.save({"X": Xall, "Y": Yall, "plate": Pall}, cache_path)
        print(f"[cache] Saved -> {cache_path}")

    print(f"\nTotal features: {Xall.shape[0]} x {Xall.shape[1]}")
    inv_label = {v: k for k, v in LABEL_MAP.items()}
    for lb in sorted(LABEL_MAP.values()):
        cnt = int((Yall == lb).sum())
        print(f"  {inv_label[lb]}: {cnt}")

    # ---- 4) Plate mixing tests ----
    if cfg.RUN_PLATE_MIXING_TEST:
        metric_dir = os.path.join(cfg.SAVE_DIR, "plate_mixing_metrics")
        evaluate_plate_mixing(
            Xall, Yall, Pall, cfg, split_name="all", save_dir=metric_dir
        )

    # ---- 5) UMAP ----
    plot_dir = os.path.join(cfg.SAVE_DIR, "umap_plate_viz")
    plot_umap_by_plate_within_class(
        X=Xall,
        Y=Yall,
        plates=Pall,
        class_names=cfg.CLASS_NAMES,
        seed=cfg.SEED,
        n_neighbors=30,
        min_dist=0.10,
        metric="cosine",
        max_legend=25,
        point_size=6.0,
        alpha=0.75,
        save_dir=plot_dir,
    )

    print("\n" + "=" * 40)
    print("✅ Plate artifact evaluation 완료!")
    print("=" * 40)


if __name__ == "__main__":
    run_plate_eval(CFG())
