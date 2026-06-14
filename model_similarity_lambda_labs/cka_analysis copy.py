# ==============================================================================
# CKA (Centered Kernel Alignment) Analysis for Model Similarity
# ==============================================================================
# 두 모델(다른 seed로 학습)이 비슷한 representation을 학습했는지 측정하는 코드
# - stage5_out에서 feature map 추출
# - adaptive pooling으로 축소
# - CKA로 두 모델 간의 representation 유사도 측정
#
# 이 버전은 sae_project 의존성 없이 독립적으로 실행 가능
# ==============================================================================

import os
import sys
import random
import argparse
import logging
import io
import pickle
import re
import csv
from typing import List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint_sequential
from torchvision import transforms
from tqdm.auto import tqdm

# Workaround: Lambda Labs PyTorch may be built without numpy support
def _to_numpy(t: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy, with fallback for envs without torch-numpy bridge."""
    t = t.cpu().detach().contiguous()
    try:
        return t.numpy()
    except RuntimeError:
        # Fallback: reconstruct from raw storage
        dtype_map = {torch.float32: np.float32, torch.float64: np.float64,
                     torch.float16: np.float16, torch.int64: np.int64, torch.int32: np.int32}
        np_dtype = dtype_map.get(t.dtype, np.float32)
        return np.frombuffer(bytes(t.untyped_storage()), dtype=np_dtype).reshape(t.shape).copy()

try:
    import tifffile
except ImportError:
    print("Installing tifffile...")
    os.system("pip -q install tifffile")
    import tifffile

# ==============================================================================
# Logging
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("CKA_Analysis")

# ==============================================================================
# Constants
# ==============================================================================
DEFAULT_SHARD_ROOT = "/home/ubuntu/model-east3/wds_shards_tar"
OUT_DIM = 512

# Weight renorm (must match training code)
@torch.no_grad()
def renorm_unit_per_out_channel_(model: nn.Module, eps: float = 1e-12):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            w = m.weight.data
            fan_out = w.shape[0]
            w_flat = w.view(fan_out, -1)
            norms = w_flat.norm(dim=1, keepdim=True).clamp_min(eps)
            w_flat.div_(norms)
            m.weight.data.copy_(w_flat.view_as(w))

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
PLATE_DIR_RE = re.compile(r"plate=(\d{6})")

# ==============================================================================
# SampleRef and Data Loading (standalone)
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

    import tarfile
    items = {}
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

    logger.info(f"[tar-index] built {len(pairs)} pairs: {os.path.basename(tar_path)}")

def load_all_sample_refs(shard_root: str) -> List[SampleRef]:
    import glob
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

def build_uid_to_refidx(refs: List[SampleRef]) -> dict:
    return {f"{r.tar_path}:{r.prefix}": i for i, r in enumerate(refs)}

# ==============================================================================
# SafeInstanceNormalize
# ==============================================================================
class SafeInstanceNormalize:
    def __init__(self, threshold: float = 0.01):
        self.threshold = float(threshold)
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = tensor.mean(dim=[1,2], keepdim=True)
        std = tensor.std(dim=[1,2], keepdim=True).clamp_min(self.threshold)
        return (tensor - mean) / std

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

# ==============================================================================
# Encoder Definition (standalone, matches training code)
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
    def __init__(self, blocks=(2,2,4,4), dilations=(1,1,1,1), refine_blocks=1, ckpt_segments=2):
        super().__init__()
        b2,b3,b4,b5 = blocks
        d2,d3,d4,d5 = dilations

        self.stem = nn.Sequential(conv2d(3, 64, k=3, stride=2, padding=1, bias=True))
        self.stage2 = Stage(64, 128, b2, d2, use_ckpt=False, ckpt_segments=1)
        self.stage3 = Stage(128, 256, b3, d3, use_ckpt=False, ckpt_segments=1)
        self.stage4 = Stage(256, 512, b4, d4, use_ckpt=True, ckpt_segments=ckpt_segments)
        self.stage5 = Stage(512, OUT_DIM, b5, d5, use_ckpt=True, ckpt_segments=ckpt_segments)
        self.refine = Stage(OUT_DIM, OUT_DIM, int(refine_blocks), 1, use_ckpt=True, ckpt_segments=ckpt_segments)

        self.trunk = nn.Sequential(self.stem, self.stage2, self.stage3, self.stage4, self.stage5, self.refine)
        self.gap = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        x = self.trunk(x)
        x = self.gap(x).flatten(1)
        return x

    def forward_feature_maps(self, x, which: str = "stage5_out"):
        """Get intermediate feature maps for CKA analysis"""
        x = self.stem(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        if which == "stage5_out":
            return x
        x = self.refine(x)
        return x

# ==============================================================================
# Encoder with CKA Features
# ==============================================================================
class EncoderWithCKAHook(Encoder):
    """Encoder that can extract stage5 features with GAP L2 normalization for CKA
    Also extracts refine_out GAP vector for separate CKA comparison."""
    def __init__(self, *args, use_l2_weighting=True, use_gaussian_blur=True, 
                 blur_sigma=2.0, pooling_size=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_l2_weighting = use_l2_weighting
        self.use_gaussian_blur = use_gaussian_blur
        self.blur_sigma = blur_sigma
        self.adapt_pool = nn.AdaptiveAvgPool2d((pooling_size, pooling_size))
        
        if use_gaussian_blur:
            self.register_buffer('blur_kernel', self._make_gaussian_kernel(blur_sigma))

    def _make_gaussian_kernel(self, sigma: float, kernel_size: int = None):
        if kernel_size is None:
            kernel_size = int(6 * sigma + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
        
        coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel_1d = g / g.sum()
        kernel_2d = kernel_1d[:, None] @ kernel_1d[None, :]
        return kernel_2d.unsqueeze(0).unsqueeze(0)

    @torch.no_grad()
    def get_normalized_features(self, x):
        """Get GAP L2 normalized + pooled features from stage5 output,
        plus refine_out GAP vector (L2-normalized).
        
        Returns:
            stage5_features: (B, C*pooling_size*pooling_size) pooled stage5 features
            refine_gap: (B, C) L2-normalized GAP vector from refine_out
        """
        # Stage5 feature map
        fmap_s5 = self.forward_feature_maps(x, which="stage5_out")
        B, C, H, W = fmap_s5.shape
        
        # Refine feature map (continue from stage5)
        fmap_refine = self.refine(fmap_s5)
        
        # --- Refine GAP (L2-normalized) ---
        refine_gap = fmap_refine.mean(dim=(2, 3))  # (B, C)
        refine_gap = F.normalize(refine_gap, dim=1, eps=1e-12)
        
        # --- Stage5 pooled features (with optional blur/weighting) ---
        # GAP L2 normalization
        gap = fmap_s5.mean(dim=(2, 3))
        gap_norms = gap.norm(dim=1, keepdim=True).view(B, 1, 1, 1).clamp_min(1e-12)
        fmap_s5 = fmap_s5 / gap_norms
        
        # Gaussian blur
        if self.use_gaussian_blur:
            kernel = self.blur_kernel.to(fmap_s5.device)
            pad = kernel.shape[-1] // 2
            fmap_flat = fmap_s5.view(B * C, 1, H, W)
            fmap_flat = F.pad(fmap_flat, (pad, pad, pad, pad), mode='reflect')
            fmap_flat = F.conv2d(fmap_flat, kernel, groups=1)
            fmap_s5 = fmap_flat.view(B, C, H, W)
        
        # L2 weighting
        if self.use_l2_weighting:
            l2_map = fmap_s5.norm(dim=1, keepdim=True)
            fmap_s5 = fmap_s5 * l2_map
        
        # Adaptive Pooling
        pooled = self.adapt_pool(fmap_s5)
        stage5_features = pooled.flatten(1)
        
        return stage5_features, refine_gap

# ==============================================================================
# Dataset for feature extraction
# ==============================================================================
class FeatureExtractionDataset(Dataset):
    """Dataset for extracting features from validation images"""
    def __init__(self, refs: List[SampleRef], ref_indices: List[int], img_size: int):
        self.refs = refs
        self.ref_indices = ref_indices
        self.img_size = int(img_size)
        self.normalize = SafeInstanceNormalize(threshold=0.01)
        
        logger.info(f"Preloading {len(ref_indices)} images for feature extraction...")
        self.images: List = [None] * len(ref_indices)
        self.labels: List[int] = [0] * len(ref_indices)
        self.uids: List[str] = [""] * len(ref_indices)
        
        tar_to_fh = {}
        def read_bytes(tp: str, off: int, size: int) -> bytes:
            fh = tar_to_fh.get(tp, None)
            if fh is None:
                fh = open(tp, "rb", buffering=0)
                tar_to_fh[tp] = fh
            fh.seek(off)
            return fh.read(size)

        bad = 0
        for j, ridx in enumerate(tqdm(ref_indices, desc="preload", leave=True)):
            r = refs[ridx]
            try:
                tif_bytes = read_bytes(r.tar_path, r.tif_off, r.tif_size)
                img = tifffile.imread(io.BytesIO(tif_bytes))
                validate_uint16_rgb_128(img, self.img_size)
                self.images[j] = img
                self.labels[j] = int(r.label)
                self.uids[j] = f"{r.tar_path}:{r.prefix}"
            except Exception:
                bad += 1
                self.images[j] = None

        for fh in tar_to_fh.values():
            try: fh.close()
            except: pass

        logger.info(f"Preload done. bad={bad}/{len(ref_indices)}")

    def __len__(self):
        return len(self.ref_indices)

    def __getitem__(self, idx: int):
        img = self.images[idx]
        if img is None:
            return None
        
        y = self.labels[idx]
        uid = self.uids[idx]
        
        x = (img.astype(np.float32) / 65535.0)
        x = torch.as_tensor(x).permute(2,0,1)
        x = self.normalize(x)
        
        return x, y, uid

def collate_cka(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    
    xs = torch.stack([b[0] for b in batch])
    ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
    uids = [b[2] for b in batch]
    
    return xs, ys, uids

def load_val_split_csv(csv_path: str) -> List[str]:
    """Load UIDs from validation split CSV"""
    uids = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uids.append(row["uid"])
    return uids

# ==============================================================================
# CKA Functions (Torch GPU)
#   - O(n²) double-centering, torch tensors throughout
#   - linear_CKA: center each kernel only once (2 centering calls)
# ==============================================================================
def linear_CKA(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA on torch tensors using the feature-space formulation.
    This avoids the O(N^2) memory explosion of the kernel-space formulation 
    by computing DxD covariance matrices instead of NxN kernel matrices."""
    # 1. Column center X and Y in double precision to prevent floating point inaccuracies
    X_c = X.double()
    X_c = X_c - X_c.mean(dim=0, keepdim=True)
    Y_c = Y.double()
    Y_c = Y_c - Y_c.mean(dim=0, keepdim=True)
    
    # 2. Compute covariance-like matrices
    C_XY = X_c.T @ Y_c  # (D_X, D_Y)
    C_XX = X_c.T @ X_c  # (D_X, D_X)
    C_YY = Y_c.T @ Y_c  # (D_Y, D_Y)
    
    # 3. Frobenius norms squared
    hsic_xy = (C_XY ** 2).sum()
    hsic_xx = (C_XX ** 2).sum()
    hsic_yy = (C_YY ** 2).sum()
    
    return float(hsic_xy / (torch.sqrt(hsic_xx * hsic_yy) + 1e-10))

# ==============================================================================
# Model Loading (Direct Encoder Loading - No sae_project dependency)
# ==============================================================================
def load_encoder_for_cka(ckpt_path: str, args) -> EncoderWithCKAHook:
    """
    Load encoder directly from checkpoint without full model wrapper.
    Handles multiple checkpoint formats:
    - New format: {"model": {...}} with encoder.* keys
    - Old format: {"model_q": {...}} with encoder.* keys
    - Direct state_dict
    """
    blocks = parse_int_list(args.blocks, 4)
    dilations = parse_int_list(args.dilations, 4)
    
    # Create encoder with CKA hooks
    encoder = EncoderWithCKAHook(
        blocks=blocks,
        dilations=dilations,
        refine_blocks=args.refine_blocks,
        ckpt_segments=args.ckpt_segments,
        use_l2_weighting=args.use_l2_weighting,
        use_gaussian_blur=args.use_gaussian_blur,
        blur_sigma=args.blur_sigma,
        pooling_size=args.pooling_size,
    )
    
    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    # Find the state dict
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            state_dict = ckpt["model"]
        elif "model_q" in ckpt:
            state_dict = ckpt["model_q"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt
    
    # Extract encoder weights (remove 'encoder.' prefix if present)
    encoder_sd = {}
    for k, v in state_dict.items():
        if k.startswith("encoder."):
            new_key = k[len("encoder."):]
            encoder_sd[new_key] = v
        elif not k.startswith("projector."):
            # Direct encoder keys (no prefix)
            encoder_sd[k] = v
    
    # Load with non-strict to handle adapt_pool and blur_kernel
    missing, unexpected = encoder.load_state_dict(encoder_sd, strict=False)
    
    if missing:
        # Filter out expected missing keys
        expected_missing = {'adapt_pool.', 'blur_kernel'}
        real_missing = [k for k in missing if not any(k.startswith(e) for e in expected_missing)]
        if real_missing:
            logger.warning(f"Missing keys: {real_missing}")
    
    # Apply renorm (matches bake_unitnorm_encoder in training code)
    renorm_unit_per_out_channel_(encoder)
    encoder.eval()
    logger.info(f"Loaded encoder from: {ckpt_path}")
    return encoder

@torch.no_grad()
def extract_features(encoder: EncoderWithCKAHook, dataloader: DataLoader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """Extract features from encoder for all images.
    
    Returns:
        stage5_features: (N, D_pooled) torch.float32 on CPU
        refine_gap_features: (N, C) torch.float32 on CPU
        labels: (N,) torch.long on CPU
        uids: list of UID strings
    """
    encoder.eval()
    encoder.to(device)
    
    all_stage5 = []
    all_refine_gap = []
    all_labels = []
    all_uids = []
    
    for batch in tqdm(dataloader, desc="Extracting features"):
        if batch is None:
            continue
        
        x, y, uids = batch
        x = x.to(device)
        
        stage5_feat, refine_gap = encoder.get_normalized_features(x)
        
        all_stage5.append(stage5_feat.cpu().float())
        all_refine_gap.append(refine_gap.cpu().float())
        all_labels.append(y)
        all_uids.extend(uids)
    
    stage5_t = torch.cat(all_stage5, dim=0)
    refine_gap_t = torch.cat(all_refine_gap, dim=0)
    labels_t = torch.cat(all_labels, dim=0)
    
    logger.info(f"Extracted stage5: {stage5_t.shape}, refine_gap: {refine_gap_t.shape}")
    return stage5_t, refine_gap_t, labels_t, all_uids

# ==============================================================================
# Balanced Class Sampling
# ==============================================================================
def sample_balanced_per_class(refs: List[SampleRef], ref_indices: List[int], 
                               num_samples_per_class: int, seed: int) -> List[int]:
    rng = random.Random(seed)
    
    class_to_indices = defaultdict(list)
    for idx in ref_indices:
        label = refs[idx].label
        class_to_indices[label].append(idx)
    
    sampled_indices = []
    classes = sorted(class_to_indices.keys())
    
    for class_label in classes:
        indices = class_to_indices[class_label]
        rng.shuffle(indices)
        n_sample = min(num_samples_per_class, len(indices))
        sampled_indices.extend(indices[:n_sample])
        
        class_name = ["Control", "SNCA", "GBA", "LRRK2"][class_label]
        logger.info(f"  Class {class_name}: sampled {n_sample}/{len(indices)} images")
    
    rng.shuffle(sampled_indices)
    return sampled_indices

# ==============================================================================
# Main Analysis (Multi-Model Support)
# ==============================================================================
def run_cka_on_caches(args):
    """Run CKA directly on pre-extracted .npz caches."""
    logger.info(f"Loading {len(args.caches)} caches for CKA analysis...")
    
    all_features = []
    all_labels = None
    model_names = []
    
    for i, path in enumerate(args.caches):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Cache not found: {path}")
            
        data = np.load(path, allow_pickle=True)
        if "X_all" in data:
            X = data["X_all"]
            if "usage_ema" in data:
                alive_mask = data["usage_ema"] >= args.dead_threshold
                X = X[:, alive_mask]
                logger.info(f"[{i+1}] {path}: alive SAE neurons: {alive_mask.sum()}/{len(alive_mask)}")
        elif "X_gap" in data:
            X = data["X_gap"]
            logger.info(f"[{i+1}] {path}: CNN features")
        else:
            raise KeyError(f"Cache {path} missing X_all or X_gap")
            
        y = data["y"]
        if all_labels is None:
            # We only use classes 0, 1, 2, 3 and sample up to args.samples_per_class per class
            target_classes = [0, 1, 2, 3]
            rng = np.random.default_rng(args.seed)
            sub_indices = []
            
            for cls in target_classes:
                cls_idx = np.where(y == cls)[0]
                n_take = min(args.samples_per_class, len(cls_idx))
                if len(cls_idx) > n_take:
                    cls_idx = rng.choice(cls_idx, n_take, replace=False)
                sub_indices.extend(cls_idx)
                
            sub_indices = np.sort(sub_indices)
            all_labels = y[sub_indices]
            logger.info(f"Subsampled {len(sub_indices)} images for CKA (up to {args.samples_per_class} per class for classes {target_classes}).")
            
        else:
            assert np.array_equal(y[sub_indices], all_labels), f"Label mismatch in {path}!"
            
        # Apply subsampling
        X = X[sub_indices]
        
        if args.gap_l2_norm:
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1e-12, norms)
            X = X / norms
            
        all_features.append(torch.tensor(X, dtype=torch.float32))
        
        # Extract name from path
        name = os.path.basename(os.path.dirname(path))
        if not name or name == ".":
            name = f"model_{i}"
        model_names.append(name)
        
    num_models = len(args.caches)
    def compute_and_save_cka(features_list, filename_suffix, title):
        logger.info("\n" + "="*60)
        logger.info(title)
        logger.info("="*60)
        
        cka_matrix = np.eye(num_models)
        for i in range(num_models):
            for j in range(i + 1, num_models):
                cka_val = linear_CKA(features_list[i], features_list[j])
                cka_matrix[i, j] = cka_matrix[j, i] = cka_val
                logger.info(f"  {model_names[i]} vs {model_names[j]}: {cka_val:.4f}")
                
        os.makedirs(args.output_dir, exist_ok=True)
        csv_path = os.path.join(args.output_dir, f"cka_matrix_caches_{filename_suffix}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([""] + model_names)
            for i, name in enumerate(model_names):
                writer.writerow([name] + [f"{cka_matrix[i, j]:.4f}" for j in range(num_models)])
        logger.info(f"Saved CKA matrix to: {csv_path}")
        return cka_matrix
        
    # 1. Global CKA
    global_cka = compute_and_save_cka(all_features, "global", "Computing Global CKA Matrix (Classes 0,1,2,3)")
    
    # 2. Per-class CKA
    class_names_dict = {0: "Control", 1: "SNCA", 2: "GBA", 3: "LRRK2"}
    target_classes = [0, 1, 2, 3]
    for cls in target_classes:
        mask = (all_labels == cls)
        if mask.sum() > 0:
            cls_features = [feat[mask] for feat in all_features]
            cls_name = class_names_dict[cls]
            compute_and_save_cka(cls_features, f"class_{cls_name}", f"Computing CKA Matrix for Class: {cls_name} (n={mask.sum()})")
    
    return global_cka

def run_cka_analysis(args):
    """Run CKA analysis between multiple models (pairwise comparison)"""
    if getattr(args, "caches", None):
        return run_cka_on_caches(args)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    ckpt_paths = args.ckpt_paths
    num_models = len(ckpt_paths)
    logger.info(f"Comparing {num_models} models")
    
    # Load sample refs
    refs = load_all_sample_refs(args.shard_root)
    uid_to_ref_idx = build_uid_to_refidx(refs)
    
    # Load validation + test splits -- try multiple save_dirs or auto-detect from ckpt_paths
    save_dirs = getattr(args, 'save_dirs', []) or []
    if args.save_dir:
        save_dirs.insert(0, args.save_dir)
    # Auto-detect: use ckpt parent dirs as save_dirs
    if not save_dirs:
        save_dirs = list(set(os.path.dirname(p) for p in ckpt_paths))
    
    all_eval_uids = []
    for sd in save_dirs:
        for csv_name in ["val_split.csv", "test_split.csv"]:
            csv_path = os.path.join(sd, csv_name)
            if os.path.exists(csv_path):
                uids = load_val_split_csv(csv_path)
                all_eval_uids.extend(uids)
                logger.info(f"Loaded {len(uids)} images from {csv_path}")
    
    if len(all_eval_uids) > 0:
        seen = set()
        unique_uids = []
        for uid in all_eval_uids:
            if uid not in seen:
                seen.add(uid)
                unique_uids.append(uid)
        
        eval_ref_indices = [uid_to_ref_idx[uid] for uid in unique_uids if uid in uid_to_ref_idx]
        logger.info(f"Total evaluation images: {len(eval_ref_indices)} (val + test combined)")
        
        if args.num_samples > 0:
            num_classes = 4
            num_per_class = args.num_samples // num_classes
            logger.info(f"Sampling {num_per_class} images per class (total: {num_per_class * num_classes})")
            ref_indices = sample_balanced_per_class(refs, eval_ref_indices, num_per_class, args.seed)
        else:
            ref_indices = eval_ref_indices
    else:
        logger.info(f"No val_split.csv or test_split.csv found. Sampling from all images...")
        all_indices = list(range(len(refs)))
        num_classes = 4
        num_per_class = args.num_samples // num_classes
        logger.info(f"Sampling {num_per_class} images per class (total: {num_per_class * num_classes})")
        ref_indices = sample_balanced_per_class(refs, all_indices, num_per_class, args.seed)
    
    logger.info(f"Using {len(ref_indices)} images for CKA analysis")
    
    # Create dataset
    dataset = FeatureExtractionDataset(refs, ref_indices, args.img_size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_cka,
        pin_memory=True
    )
    
    # Extract features from all models
    all_stage5 = []
    all_refine_gap = []
    all_labels = None
    model_names = []
    
    for i, ckpt_path in enumerate(ckpt_paths):
        logger.info(f"\n[{i+1}/{num_models}] Loading Encoder from: {ckpt_path}")
        encoder = load_encoder_for_cka(ckpt_path, args)
        
        stage5_feat, refine_gap_feat, labels, uids = extract_features(encoder, dataloader, device)
        all_stage5.append(stage5_feat)
        all_refine_gap.append(refine_gap_feat)
        
        if all_labels is None:
            all_labels = labels
        
        # Extract model name from path
        model_name = os.path.basename(os.path.dirname(ckpt_path))
        if not model_name or model_name == ".":
            model_name = os.path.basename(ckpt_path).replace(".pt", "")
        model_names.append(model_name)
        
        del encoder
        torch.cuda.empty_cache()
    
    # ================================================================
    # Compute pairwise CKA: stage5_out (pooled spatial features)
    # ================================================================
    logger.info("\n" + "="*60)
    logger.info("Computing Pairwise CKA Matrix [stage5_out pooled]...")
    logger.info("="*60)
    
    cka_matrix = np.eye(num_models)
    for i in range(num_models):
        for j in range(i + 1, num_models):
            cka_val = linear_CKA(all_stage5[i], all_stage5[j])
            cka_matrix[i, j] = cka_matrix[j, i] = cka_val
            logger.info(f"  {model_names[i]} vs {model_names[j]}: {cka_val:.4f}")
    
    # ================================================================
    # Compute pairwise CKA: refine_out GAP (L2-normalized)
    # ================================================================
    logger.info("\n" + "="*60)
    logger.info("Computing Pairwise CKA Matrix [refine_out GAP]...")
    logger.info("="*60)
    
    cka_matrix_gap = np.eye(num_models)
    for i in range(num_models):
        for j in range(i + 1, num_models):
            cka_val = linear_CKA(all_refine_gap[i], all_refine_gap[j])
            cka_matrix_gap[i, j] = cka_matrix_gap[j, i] = cka_val
            logger.info(f"  {model_names[i]} vs {model_names[j]}: {cka_val:.4f}")
    
    # Per-class CKA (if only 2 models)
    class_cka_results = {}
    class_cka_gap_results = {}
    if num_models == 2:
        logger.info("\n=== Per-Class CKA Analysis ===")
        classes = ["Control", "SNCA", "GBA", "LRRK2"]
        for class_label, class_name in enumerate(classes):
            mask = (all_labels == class_label)
            if mask.sum() > 10:
                cka_s5 = linear_CKA(all_stage5[0][mask], all_stage5[1][mask])
                class_cka_results[class_name] = cka_s5
                cka_gap = linear_CKA(all_refine_gap[0][mask], all_refine_gap[1][mask])
                class_cka_gap_results[class_name] = cka_gap
                logger.info(f"  {class_name}: stage5={cka_s5:.4f}, refine_gap={cka_gap:.4f} (n={mask.sum()})")
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = {
        "ckpt_paths": ckpt_paths,
        "model_names": model_names,
        "num_models": num_models,
        "num_samples": len(ref_indices),
        "stage5_feature_dim": int(all_stage5[0].shape[1]),
        "refine_gap_dim": int(all_refine_gap[0].shape[1]),
        "cka_matrix_stage5": cka_matrix.tolist(),
        "cka_matrix_refine_gap": cka_matrix_gap.tolist(),
        "per_class_linear_cka_stage5": class_cka_results if class_cka_results else None,
        "per_class_linear_cka_refine_gap": class_cka_gap_results if class_cka_gap_results else None,
    }
    
    import json
    results_path = os.path.join(args.output_dir, "cka_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved results to: {results_path}")
    
    # Save CKA matrices as CSV
    for tag, mat in [("stage5", cka_matrix), ("refine_gap", cka_matrix_gap)]:
        csv_path = os.path.join(args.output_dir, f"cka_matrix_{tag}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([""] + model_names)
            for i, name in enumerate(model_names):
                writer.writerow([name] + [f"{mat[i, j]:.4f}" for j in range(num_models)])
        logger.info(f"Saved CKA matrix ({tag}) to: {csv_path}")
    
    # Try to create heatmap visualization
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(max(14, num_models * 2), max(6, num_models * 0.8)))
        
        for ax, mat, title_tag in zip(axes, [cka_matrix, cka_matrix_gap],
                                       ["stage5_out (pooled)", "refine_out (GAP)"]):
            im = ax.imshow(mat, cmap='RdYlGn', vmin=0, vmax=1)
            ax.set_xticks(range(num_models))
            ax.set_yticks(range(num_models))
            ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
            ax.set_yticklabels(model_names, fontsize=9)
            for i in range(num_models):
                for j in range(num_models):
                    ax.text(j, i, f'{mat[i, j]:.3f}',
                            ha="center", va="center", color="black", fontsize=8)
            plt.colorbar(im, ax=ax, label='Linear CKA', shrink=0.8)
            ax.set_title(f'CKA: {title_tag}', fontsize=12)
        
        plt.tight_layout()
        heatmap_path = os.path.join(args.output_dir, "cka_heatmap.png")
        plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved heatmap to: {heatmap_path}")
    except ImportError:
        logger.info("matplotlib not available, skipping heatmap generation")
    except Exception as e:
        logger.warning(f"Could not generate heatmap: {e}")
    
    # Print summary
    print("\n" + "="*70)
    print("CKA Analysis Summary")
    print("="*70)
    print(f"Number of models: {num_models}")
    print(f"Number of samples: {len(ref_indices)}")
    print(f"Stage5 feature dim: {all_stage5[0].shape[1]}")
    print(f"Refine GAP dim: {all_refine_gap[0].shape[1]}")
    print("-"*70)
    print("\nModels compared:")
    for i, name in enumerate(model_names):
        print(f"  [{i+1}] {name}")
    
    for tag, mat in [("stage5_out (pooled)", cka_matrix), ("refine_out (GAP)", cka_matrix_gap)]:
        print("-"*70)
        print(f"\nPairwise Linear CKA [{tag}]:")
        col_w = max(16, max(len(n) for n in model_names) + 2)
        header = " " * col_w + "".join([f"{name:>{col_w}}" for name in model_names])
        print(header)
        for i, name in enumerate(model_names):
            row = f"{name:<{col_w}}" + "".join([f"{mat[i,j]:{col_w}.4f}" for j in range(num_models)])
            print(row)
        
        if num_models > 1:
            off_diag = mat[np.triu_indices(num_models, k=1)]
            print(f"  Mean={np.mean(off_diag):.4f}  Std={np.std(off_diag):.4f}  "
                  f"Min={np.min(off_diag):.4f}  Max={np.max(off_diag):.4f}")
    
    if class_cka_results:
        print("\nPer-Class Linear CKA (2-model):")
        print(f"  {'Class':12s}  {'stage5':>10s}  {'refine_gap':>10s}")
        for class_name in class_cka_results:
            s5 = class_cka_results.get(class_name, 0)
            rg = class_cka_gap_results.get(class_name, 0)
            print(f"  {class_name:12s}  {s5:>10.4f}  {rg:>10.4f}")
    
    print("="*70)
    
    # Interpretation
    if num_models == 2:
        cka_s5 = cka_matrix[0, 1]
        cka_rg = cka_matrix_gap[0, 1]
        print(f"\nInterpretation (stage5={cka_s5:.4f}, refine_gap={cka_rg:.4f}):")
        for tag, cka_val in [("stage5", cka_s5), ("refine_gap", cka_rg)]:
            if cka_val >= 0.9:
                print(f"  {tag}: VERY SIMILAR (CKA ≥ 0.9)")
            elif cka_val >= 0.7:
                print(f"  {tag}: SIMILAR (0.7 ≤ CKA < 0.9)")
            elif cka_val >= 0.5:
                print(f"  {tag}: MODERATELY SIMILAR (0.5 ≤ CKA < 0.7)")
            else:
                print(f"  {tag}: DIFFERENT (CKA < 0.5)")
    elif num_models > 2:
        off_s5 = cka_matrix[np.triu_indices(num_models, k=1)]
        off_rg = cka_matrix_gap[np.triu_indices(num_models, k=1)]
        print(f"\nInterpretation (mean off-diag: stage5={np.mean(off_s5):.4f}, refine_gap={np.mean(off_rg):.4f})")
    
    return results


def get_args():
    p = argparse.ArgumentParser("CKA Analysis for Model Similarity (Multi-Model Support)")
    
    # Model checkpoints (supports 2 or more)
    p.add_argument("--ckpt_paths", type=str, nargs="*", default=[],
                   help="Paths to model checkpoints (2 or more). Example: --ckpt_paths model1.pt model2.pt model3.pt")
    
    # Caches (bypasses model loading)
    p.add_argument("--caches", type=str, nargs="*", default=[],
                   help="Paths to pre-extracted .npz caches. Bypasses encoder inference entirely.")
    p.add_argument("--dead_threshold", type=float, default=1e-5)
    p.add_argument("--gap_l2_norm", action="store_true")
    
    # Data paths
    p.add_argument("--shard_root", type=str, default=DEFAULT_SHARD_ROOT,
                   help="Path to WebDataset shards")
    p.add_argument("--save_dir", type=str, default="",
                   help="Save dir of a model (for loading val_split.csv). If empty, auto-detects from ckpt parent dirs.")
    p.add_argument("--save_dirs", type=str, nargs="*", default=[],
                   help="Multiple save dirs to find val/test CSVs")
    
    # Output
    p.add_argument("--output_dir", type=str, default="./cka_results",
                   help="Directory to save results")
    
    # Sampling
    p.add_argument("--num_samples", type=int, default=1000,
                   help="Total samples (num_samples/4 per class), 0=use all validation data")
    p.add_argument("--samples_per_class", type=int, default=5000,
                   help="Number of samples to extract per class specifically for cache mode")
    p.add_argument("--seed", type=int, default=42)
    
    # Encoder architecture (must match all models)
    p.add_argument("--blocks", type=str, default="2,2,2,3")
    p.add_argument("--dilations", type=str, default="1,1,1,1")
    p.add_argument("--refine_blocks", type=int, default=1)
    p.add_argument("--ckpt_segments", type=int, default=0)
    
    # CKA feature extraction settings
    p.add_argument("--use_l2_weighting", action="store_true", default=False)
    p.add_argument("--no_l2_weighting", action="store_false", dest="use_l2_weighting")
    p.add_argument("--use_gaussian_blur", action="store_true", default=False)
    p.add_argument("--no_gaussian_blur", action="store_false", dest="use_gaussian_blur")
    p.add_argument("--blur_sigma", type=float, default=2.0)
    p.add_argument("--pooling_size", type=int, default=8)
    
    # Dataset
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    
    if "ipykernel" in sys.modules:
        return p.parse_args([])
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    run_cka_analysis(args)

