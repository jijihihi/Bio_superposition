# ==============================================================================
# CKA (Centered Kernel Alignment) Analysis for Model Similarity
# ==============================================================================
# 두 모델(다른 seed로 학습)이 비슷한 representation을 학습했는지 측정하는 코드
# - stage5_out에서 feature map 추출 (64x64x512)
# - adaptive pooling으로 8x8로 축소
# - 각 이미지를 8*8*512 = 32768 차원 벡터로 나타냄
# - CKA로 두 모델 간의 representation 유사도 측정
# ==============================================================================

import os
import sys
import random
import argparse
from typing import List, Tuple
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

# Import from sae_project modules
from sae_project.step02_logging_utils import (
    get_logger, 
    DEFAULT_SHARD_ROOT, 
    CLASS_TO_LABEL,
    OUT_DIM
)
from sae_project.step03_data_shards import (
    SampleRef,
    load_all_sample_refs,
    build_uid_to_refidx
)
from sae_project.step04_data_bank import (
    SafeInstanceNormalize,
    validate_uint16_rgb_128,
)
from sae_project.step05_model_encoder import (
    Encoder,
    SupConMoCoModel,
    parse_int_list,
    robust_load_state_dict
)

# For image loading
import io
try:
    import tifffile
except ImportError:
    print("Installing tifffile...")
    os.system("pip -q install tifffile")
    import tifffile


logger = get_logger("CKA_Analysis")


# ==============================================================================
# Helper Functions
# ==============================================================================
def collate_cka(batch):
    """
    Custom collate function for CKA dataset.
    Handles 3-element tuples: (x, y, uid)
    Skips None samples.
    """
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    
    xs = torch.stack([b[0] for b in batch])
    ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
    uids = [b[2] for b in batch]
    
    return xs, ys, uids


def load_val_split_csv(csv_path: str) -> List[str]:
    """Load UIDs from validation split CSV"""
    import csv
    uids = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uids.append(row["uid"])
    return uids


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
        
        # Preload images
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
        x = torch.from_numpy(x).permute(2,0,1)
        x = self.normalize(x)
        
        return x, y, uid


# ==============================================================================
# Encoder with CKA hook (extends base Encoder)
# ==============================================================================
class EncoderWithCKAHook(Encoder):
    """Encoder that can extract stage5 features with GAP L2 normalization, Gaussian blur, and L2 weighting for CKA"""
    def __init__(self, *args, use_l2_weighting=True, use_gaussian_blur=True, 
                 blur_sigma=2.0, pooling_size=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_l2_weighting = use_l2_weighting
        self.use_gaussian_blur = use_gaussian_blur
        self.blur_sigma = blur_sigma
        self.adapt_pool = nn.AdaptiveAvgPool2d((pooling_size, pooling_size))
        
        # Pre-compute Gaussian kernel
        if use_gaussian_blur:
            self.register_buffer('blur_kernel', self._make_gaussian_kernel(blur_sigma))

    def _make_gaussian_kernel(self, sigma: float, kernel_size: int = None):
        """Create a 2D Gaussian kernel for blurring"""
        if kernel_size is None:
            kernel_size = int(6 * sigma + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
        
        coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel_1d = g / g.sum()
        kernel_2d = kernel_1d[:, None] @ kernel_1d[None, :]
        return kernel_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, K, K)

    @torch.no_grad()
    def get_normalized_features(self, x):
        """
        Get GAP L2 normalized + Gaussian blurred + L2 weighted + pooled features from stage5 output.
        1. Get stage5 feature map (B, 512, 64, 64)
        2. Normalize by GAP L2 norm (divide each spatial position by GAP norm)
        3. (Optional) Gaussian blur: smooth spatial differences
        4. (Optional) L2 weighting: scale each position by its L2 norm
        5. Adaptive Pooling (8x8)
        6. Flatten
        """
        # Get stage5 feature map: (B, 512, 64, 64)
        fmap = self.forward_feature_maps(x, which="stage5_out")
        B, C, H, W = fmap.shape
        
        # GAP L2 normalization (same as training/SAE)
        gap = fmap.mean(dim=(2, 3))  # (B, 512)
        gap_norms = gap.norm(dim=1, keepdim=True).view(B, 1, 1, 1).clamp_min(1e-12)
        fmap = fmap / gap_norms  # Divide each spatial position by GAP norm
        
        # Gaussian blur: smooth spatial differences
        if self.use_gaussian_blur:
            # Apply same blur kernel to each channel
            kernel = self.blur_kernel.to(fmap.device)  # (1, 1, K, K)
            pad = kernel.shape[-1] // 2
            # Reshape for group conv: (B*C, 1, H, W)
            fmap_flat = fmap.view(B * C, 1, H, W)
            fmap_flat = F.pad(fmap_flat, (pad, pad, pad, pad), mode='reflect')
            fmap_flat = F.conv2d(fmap_flat, kernel, groups=1)
            fmap = fmap_flat.view(B, C, H, W)
        
        # L2 weighting: scale each position by its L2 norm
        if self.use_l2_weighting:
            l2_map = fmap.norm(dim=1, keepdim=True)  # (B, 1, H, W)
            fmap = fmap * l2_map  # Scale by L2 norm
        
        # Adaptive Pooling (8x8)
        pooled = self.adapt_pool(fmap)  # (B, 512, pool_size, pool_size)
        
        # Flatten
        features = pooled.flatten(1)  # (B, 512*pool_size*pool_size)
        
        return features


class SupConMoCoModelForCKA(nn.Module):
    """Model wrapper for CKA analysis with GAP L2 normalized + Gaussian blurred + L2 weighted features"""
    def __init__(self, blocks, dilations, refine_blocks, ckpt_segments, 
                 use_l2_weighting=True, use_gaussian_blur=True, blur_sigma=2.0, pooling_size=8):
        super().__init__()
        self.encoder = EncoderWithCKAHook(
            blocks=blocks,
            dilations=dilations,
            refine_blocks=refine_blocks,
            ckpt_segments=ckpt_segments,
            use_l2_weighting=use_l2_weighting,
            use_gaussian_blur=use_gaussian_blur,
            blur_sigma=blur_sigma,
            pooling_size=pooling_size
        )

    def get_normalized_features(self, x):
        """Get GAP L2 normalized + L2 weighted features for CKA"""
        return self.encoder.get_normalized_features(x)


# ==============================================================================
# CKA Functions
# ==============================================================================
def centering(K: np.ndarray) -> np.ndarray:
    """Center the kernel matrix"""
    n = K.shape[0]
    unit = np.ones([n, n])
    I = np.eye(n)
    H = I - unit / n
    return H @ K @ H


def rbf(X: np.ndarray, sigma: float = None) -> np.ndarray:
    """RBF (Gaussian) kernel"""
    GX = X @ X.T
    KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
    if sigma is None:
        mdist = np.median(KX[KX != 0])
        sigma = np.sqrt(mdist)
    KX *= -0.5 / (sigma ** 2)
    return np.exp(KX)


def kernel_HSIC(X: np.ndarray, Y: np.ndarray, sigma: float = None) -> float:
    """Hilbert-Schmidt Independence Criterion with RBF kernel"""
    return np.sum(centering(rbf(X, sigma)) * centering(rbf(Y, sigma)))


def linear_HSIC(X: np.ndarray, Y: np.ndarray) -> float:
    """Hilbert-Schmidt Independence Criterion with linear kernel"""
    L_X = X @ X.T
    L_Y = Y @ Y.T
    return np.sum(centering(L_X) * centering(L_Y))


def linear_CKA(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Linear Centered Kernel Alignment
    
    Args:
        X: Feature matrix from model 1, shape (n_samples, n_features)
        Y: Feature matrix from model 2, shape (n_samples, n_features)
    
    Returns:
        CKA similarity score in [0, 1]
    """
    hsic_xy = linear_HSIC(X, Y)
    hsic_xx = linear_HSIC(X, X)
    hsic_yy = linear_HSIC(Y, Y)
    
    return hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-10)


def kernel_CKA(X: np.ndarray, Y: np.ndarray, sigma: float = None) -> float:
    """
    Kernel Centered Kernel Alignment with RBF kernel
    
    Args:
        X: Feature matrix from model 1, shape (n_samples, n_features)
        Y: Feature matrix from model 2, shape (n_samples, n_features)
        sigma: RBF kernel bandwidth (if None, use median heuristic)
    
    Returns:
        CKA similarity score in [0, 1]
    """
    hsic_xy = kernel_HSIC(X, Y, sigma)
    hsic_xx = kernel_HSIC(X, X, sigma)
    hsic_yy = kernel_HSIC(Y, Y, sigma)
    
    return hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-10)


# ==============================================================================
# Model Loading and Feature Extraction
# ==============================================================================
def load_model_for_cka(ckpt_path: str, args) -> SupConMoCoModelForCKA:
    """Load a trained model for CKA analysis"""
    blocks = parse_int_list(args.blocks, 4)
    dilations = parse_int_list(args.dilations, 4)
    
    # First load into full SupConMoCoModel to handle checkpoint format
    temp_model = SupConMoCoModel(
        embed_dim=512,
        blocks=blocks,
        dilations=dilations,
        refine_blocks=args.refine_blocks,
        ckpt_segments=args.ckpt_segments,
        proj_layers=2,
        proj_hidden=2048,
        proj_bn=False,
        proj_dropout=0.0,
    )
    
    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    # Use robust_load_state_dict to handle different formats
    robust_load_state_dict(temp_model, ckpt, strict=False)
    logger.info(f"Loaded checkpoint from: {ckpt_path}")
    
    # Now create our CKA model and copy encoder weights
    model = SupConMoCoModelForCKA(
        blocks=blocks,
        dilations=dilations,
        refine_blocks=args.refine_blocks,
        ckpt_segments=args.ckpt_segments,
        use_l2_weighting=args.use_l2_weighting,
        use_gaussian_blur=args.use_gaussian_blur,
        blur_sigma=args.blur_sigma,
        pooling_size=args.pooling_size,
    )
    
    # Copy encoder state dict (excluding adapt_pool which is new)
    encoder_sd = temp_model.encoder.state_dict()
    model.encoder.load_state_dict(encoder_sd, strict=False)
    
    # Cleanup temp model
    del temp_model
    
    model.eval()
    logger.info(f"Model loaded successfully")
    return model


@torch.no_grad()
def extract_features(model: SupConMoCoModelForCKA, dataloader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract GAP L2 normalized features from a model for all images"""
    model.eval()
    model.to(device)
    
    all_features = []
    all_labels = []
    all_uids = []
    
    for batch in tqdm(dataloader, desc="Extracting features"):
        if batch is None:
            continue
        
        x, y, uids = batch
        x = x.to(device)
        
        # Get GAP L2 normalized features (512-dim per image)
        features = model.get_normalized_features(x)  # (B, 512)
        
        all_features.append(features.cpu().numpy())
        all_labels.extend(y.tolist())
        all_uids.extend(uids)
    
    features_np = np.concatenate(all_features, axis=0)
    labels_np = np.array(all_labels)
    
    logger.info(f"Extracted features: {features_np.shape}")
    return features_np, labels_np, all_uids


# ==============================================================================
# Balanced Class Sampling
# ==============================================================================
def sample_balanced_per_class(refs: List[SampleRef], ref_indices: List[int], 
                               num_samples_per_class: int, seed: int) -> List[int]:
    """
    Sample equal number of images per class from given indices
    
    Args:
        refs: List of all SampleRef objects
        ref_indices: List of ref indices to sample from
        num_samples_per_class: Number of samples per class
        seed: Random seed
    
    Returns:
        List of sampled ref indices with balanced classes
    """
    rng = random.Random(seed)
    
    # Group indices by class
    class_to_indices = defaultdict(list)
    for idx in ref_indices:
        label = refs[idx].label
        class_to_indices[label].append(idx)
    
    # Sample from each class
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
# Main Analysis
# ==============================================================================
def run_cka_analysis(args):
    """Run CKA analysis between two models"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load sample refs
    refs = load_all_sample_refs(args.shard_root)
    uid_to_ref_idx = build_uid_to_refidx(refs)
    
    # Load validation + test splits if available (combine for larger sample size)
    val_csv_path = os.path.join(args.save_dir_1, "val_split.csv")
    test_csv_path = os.path.join(args.save_dir_1, "test_split.csv")
    
    all_eval_uids = []
    
    if os.path.exists(val_csv_path):
        val_uids = load_val_split_csv(val_csv_path)
        all_eval_uids.extend(val_uids)
        logger.info(f"Loaded {len(val_uids)} validation images from val_split.csv")
    
    if os.path.exists(test_csv_path):
        test_uids = load_val_split_csv(test_csv_path)
        all_eval_uids.extend(test_uids)
        logger.info(f"Loaded {len(test_uids)} test images from test_split.csv")
    
    if len(all_eval_uids) > 0:
        # Remove duplicates while preserving order
        seen = set()
        unique_uids = []
        for uid in all_eval_uids:
            if uid not in seen:
                seen.add(uid)
                unique_uids.append(uid)
        
        # Map to ref indices
        eval_ref_indices = [uid_to_ref_idx[uid] for uid in unique_uids if uid in uid_to_ref_idx]
        logger.info(f"Total evaluation images: {len(eval_ref_indices)} (val + test combined)")
        
        # Apply balanced sampling if num_samples specified
        if args.num_samples > 0:
            num_classes = 4
            num_per_class = args.num_samples // num_classes
            logger.info(f"Sampling {num_per_class} images per class (total: {num_per_class * num_classes})")
            ref_indices = sample_balanced_per_class(refs, eval_ref_indices, num_per_class, args.seed)
        else:
            ref_indices = eval_ref_indices
    else:
        logger.info(f"No val_split.csv or test_split.csv found. Sampling from all images...")
        # Sample balanced from all refs
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
    
    # Load models
    logger.info("Loading Model 1...")
    model1 = load_model_for_cka(args.ckpt_path_1, args)
    
    logger.info("Loading Model 2...")
    model2 = load_model_for_cka(args.ckpt_path_2, args)
    
    # Extract features
    logger.info("Extracting features from Model 1...")
    features1, labels1, uids1 = extract_features(model1, dataloader, device)
    
    # Free Model 1 memory
    del model1
    torch.cuda.empty_cache()
    
    logger.info("Extracting features from Model 2...")
    features2, labels2, uids2 = extract_features(model2, dataloader, device)
    
    # Free Model 2 memory
    del model2
    torch.cuda.empty_cache()
    
    # Verify same order
    assert uids1 == uids2, "UIDs don't match between extractions!"
    
    # Compute CKA
    logger.info("Computing CKA similarity...")
    
    # Linear CKA (faster, usually similar to kernel CKA)
    linear_cka = linear_CKA(features1, features2)
    logger.info(f"Linear CKA: {linear_cka:.4f}")
    
    # Kernel CKA (RBF kernel) - can be slow for large feature matrices
    if features1.shape[0] <= 2000:  # Only compute for reasonable sizes
        kernel_cka = kernel_CKA(features1, features2)
        logger.info(f"Kernel CKA (RBF): {kernel_cka:.4f}")
    else:
        logger.info("Skipping Kernel CKA (too many samples, would be slow)")
        kernel_cka = None
    
    # Per-class CKA analysis
    logger.info("\n=== Per-Class CKA Analysis ===")
    classes = ["Control", "SNCA", "GBA", "LRRK2"]
    class_cka_results = {}
    
    for class_label, class_name in enumerate(classes):
        mask = labels1 == class_label
        if mask.sum() > 10:
            f1_class = features1[mask]
            f2_class = features2[mask]
            cka_class = linear_CKA(f1_class, f2_class)
            class_cka_results[class_name] = cka_class
            logger.info(f"  {class_name}: Linear CKA = {cka_class:.4f} (n={mask.sum()})")
        else:
            logger.info(f"  {class_name}: Not enough samples (n={mask.sum()})")
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results = {
        "ckpt_path_1": args.ckpt_path_1,
        "ckpt_path_2": args.ckpt_path_2,
        "num_samples": len(ref_indices),
        "feature_dim": features1.shape[1],
        "linear_cka": float(linear_cka),
        "kernel_cka": float(kernel_cka) if kernel_cka is not None else None,
        "per_class_linear_cka": class_cka_results,
    }
    
    import json
    results_path = os.path.join(args.output_dir, "cka_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to: {results_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("CKA Analysis Summary")
    print("="*60)
    print(f"Model 1: {os.path.basename(args.ckpt_path_1)}")
    print(f"Model 2: {os.path.basename(args.ckpt_path_2)}")
    print(f"Number of samples: {len(ref_indices)}")
    print(f"Feature dimension: {features1.shape[1]} (8x8x512)")
    print("-"*60)
    print(f"Linear CKA:  {linear_cka:.4f}")
    if kernel_cka is not None:
        print(f"Kernel CKA:  {kernel_cka:.4f}")
    print("-"*60)
    print("Per-Class Linear CKA:")
    for class_name, cka_val in class_cka_results.items():
        print(f"  {class_name:12s}: {cka_val:.4f}")
    print("="*60)
    
    # Interpretation
    print("\nInterpretation:")
    if linear_cka >= 0.9:
        print("  → The two models have VERY SIMILAR representations (CKA ≥ 0.9)")
    elif linear_cka >= 0.7:
        print("  → The two models have SIMILAR representations (0.7 ≤ CKA < 0.9)")
    elif linear_cka >= 0.5:
        print("  → The two models have MODERATELY SIMILAR representations (0.5 ≤ CKA < 0.7)")
    else:
        print("  → The two models have DIFFERENT representations (CKA < 0.5)")
    
    return results


def get_args():
    p = argparse.ArgumentParser("CKA Analysis for Model Similarity")
    
    # Model checkpoints
    p.add_argument("--ckpt_path_1", type=str, required=True,
                   help="Path to first model checkpoint (best_model.pt)")
    p.add_argument("--ckpt_path_2", type=str, required=True,
                   help="Path to second model checkpoint (best_model.pt)")
    
    # Data paths
    p.add_argument("--shard_root", type=str, default=DEFAULT_SHARD_ROOT,
                   help="Path to WebDataset shards")
    p.add_argument("--save_dir_1", type=str, default="",
                   help="Save dir of first model (for loading val_split.csv)")
    
    # Output
    p.add_argument("--output_dir", type=str, default="./cka_results",
                   help="Directory to save results")
    
    # Sampling
    p.add_argument("--num_samples", type=int, default=1000,
                   help="Total samples (num_samples/4 per class), 0=use all validation data")
    p.add_argument("--seed", type=int, default=42)
    
    # Encoder architecture (must match both models)
    p.add_argument("--blocks", type=str, default="2,2,2,3")
    p.add_argument("--dilations", type=str, default="1,1,1,1")
    p.add_argument("--refine_blocks", type=int, default=1)
    p.add_argument("--ckpt_segments", type=int, default=0)
    
    # CKA feature extraction settings
    p.add_argument("--use_l2_weighting", action="store_true", default=True,
                   help="Use L2 norm weighting for spatial positions (default: True)")
    p.add_argument("--no_l2_weighting", action="store_false", dest="use_l2_weighting",
                   help="Disable L2 norm weighting")
    p.add_argument("--use_gaussian_blur", action="store_true", default=True,
                   help="Apply Gaussian blur to smooth spatial differences (default: True)")
    p.add_argument("--no_gaussian_blur", action="store_false", dest="use_gaussian_blur",
                   help="Disable Gaussian blur")
    p.add_argument("--blur_sigma", type=float, default=2.0,
                   help="Gaussian blur sigma (default: 2.0)")
    p.add_argument("--pooling_size", type=int, default=8,
                   help="Adaptive pooling size (8 = 8x8, 1 = GAP)")
    
    # Dataset
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    
    # Colab notebook mode
    if "ipykernel" in sys.modules:
        return p.parse_args([])
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    run_cka_analysis(args)
