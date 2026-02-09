# ==============================================================================
# step04_evaluate_transform.py
# ==============================================================================
# Texture & Shape Suppression Metrics (128x128 16-bit 이미지용)
# - step03_hyperparameters.py에서 파라미터 참조
# - step05_texture_shape_eval.py에서 import하여 사용
#
# 메트릭:
#   - LV (Local Variance): 텍스처 억제 측정 (낮을수록 억제 ↑)
#   - HFE (High Frequency Energy): 텍스처 억제 측정 (낮을수록 억제 ↑)
#   - ESSIM (Edge SSIM): 형태 보존 측정 (높을수록 보존 ↑)
#   - GC (Gradient Correlation): 형태 보존 측정 (높을수록 보존 ↑)
# ==============================================================================

import cv2
import numpy as np
from skimage.util import view_as_windows


# ==============================================================================
# Base Metric Functions
# ==============================================================================

def local_variance_map(image_gray: np.ndarray, window_size: int = 6) -> float:
    """
    Compute mean local variance.
    
    Args:
        image_gray: Grayscale image (normalized to [0,1] recommended)
        window_size: Window size for local variance (논문 w=11 → 128x128: 6)
    
    Returns:
        Mean local variance value
    """
    if image_gray.shape[0] < window_size or image_gray.shape[1] < window_size:
        return float(np.var(image_gray))
    windows = view_as_windows(image_gray, (window_size, window_size))
    var_map = np.var(windows, axis=(-2, -1))
    return float(np.mean(var_map))


def high_freq_energy(image_gray: np.ndarray, radius: int = 6) -> float:
    """
    Compute high frequency energy ratio via FFT.
    
    Args:
        image_gray: Grayscale image
        radius: Cutoff radius for high frequency (논문 r=11 → 128x128: 6)
    
    Returns:
        Ratio of high frequency energy to total energy
    """
    f = np.fft.fft2(image_gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift) ** 2
    h, w = magnitude_spectrum.shape
    center = (h // 2, w // 2)
    y, x = np.ogrid[:h, :w]
    mask = ((x - center[1])**2 + (y - center[0])**2) >= radius**2
    high_freq_power = magnitude_spectrum[mask].sum()
    total_power = magnitude_spectrum.sum()
    return float(high_freq_power / (total_power + 1e-12))


def sobel_edge_map(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    """
    Compute Sobel edge magnitude map.
    
    Args:
        img: Input image (grayscale)
        ksize: Sobel kernel size (논문 k=11 → OpenCV 제한으로 5 사용)
               OpenCV Sobel은 1, 3, 5, 7만 지원
    
    Returns:
        Edge magnitude map
    """
    dx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
    dy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
    return np.hypot(dx, dy)


# ==============================================================================
# Normalized Suppression Metrics (all in [0, 1])
# ==============================================================================

def compute_LV(original_gray: np.ndarray, filtered_gray: np.ndarray, 
               window_size: int = 6) -> float:
    """
    Local Variance ratio: LV = min(1, φ_var(filtered) / φ_var(original))
    
    Lower value = more texture suppression.
    
    Args:
        original_gray: Original grayscale image [0,1]
        filtered_gray: Filtered grayscale image [0,1]
        window_size: Window size for LV (논문 w=11 → 128x128: 6)
    """
    var_orig = local_variance_map(original_gray, window_size)
    var_filt = local_variance_map(filtered_gray, window_size)
    if var_orig < 1e-12:
        return 1.0
    return float(min(1.0, var_filt / var_orig))


def compute_HFE(original_gray: np.ndarray, filtered_gray: np.ndarray, 
                radius: int = 6) -> float:
    """
    High Frequency Energy ratio: HFE = min(1, φ_hfe(filtered) / φ_hfe(original))
    
    Lower value = more texture suppression.
    
    Args:
        original_gray: Original grayscale image
        filtered_gray: Filtered grayscale image
        radius: FFT cutoff radius (논문 r=11 → 128x128: 6)
    """
    hfe_orig = high_freq_energy(original_gray, radius)
    hfe_filt = high_freq_energy(filtered_gray, radius)
    if hfe_orig < 1e-12:
        return 1.0
    return float(min(1.0, hfe_filt / hfe_orig))


def compute_ESSIM(original_gray: np.ndarray, filtered_gray: np.ndarray, 
                  ksize: int = 5) -> float:
    """
    Edge Structural Similarity: ESSIM = SSIM(sobel(orig), sobel(filt))
    
    Higher value = better shape preservation. Clamped to [0, 1].
    
    Args:
        original_gray: Original grayscale image
        filtered_gray: Filtered grayscale image
        ksize: Sobel kernel size (논문 k=11 → OpenCV 제한으로 5)
    """
    from skimage.metrics import structural_similarity as ssim
    
    edge_orig = sobel_edge_map(original_gray, ksize=ksize)
    edge_filt = sobel_edge_map(filtered_gray, ksize=ksize)
    
    # Normalize edge maps to [0, 1]
    edge_orig_norm = edge_orig / (edge_orig.max() + 1e-8)
    edge_filt_norm = edge_filt / (edge_filt.max() + 1e-8)
    
    score = ssim(edge_orig_norm, edge_filt_norm, data_range=1.0)
    return float(max(0.0, min(1.0, score)))


def compute_GC(original_gray: np.ndarray, filtered_gray: np.ndarray) -> float:
    """
    Gradient Correlation: GC = 0.5 * [corr(gx) + corr(gy)]
    
    Higher value = better shape preservation. Clamped to [0, 1].
    
    Args:
        original_gray: Original grayscale image
        filtered_gray: Filtered grayscale image
    """
    gx_orig, gy_orig = np.gradient(original_gray.astype(np.float64))
    gx_filt, gy_filt = np.gradient(filtered_gray.astype(np.float64))
    
    def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return 0.0
        return np.corrcoef(a.flatten(), b.flatten())[0, 1]
    
    gc = 0.5 * (safe_corr(gx_orig, gx_filt) + safe_corr(gy_orig, gy_filt))
    return float(max(0.0, min(1.0, gc)))


# ==============================================================================
# Convenience: All Metrics at Once
# ==============================================================================

def compute_all_metrics(original_gray: np.ndarray, filtered_gray: np.ndarray,
                        window_size: int = 6, radius: int = 6, 
                        ksize: int = 5) -> dict:
    """
    Compute all suppression metrics at once.
    
    Args:
        original_gray: Original grayscale image [0,1]
        filtered_gray: Filtered grayscale image [0,1]
        window_size: LV window size
        radius: HFE radius
        ksize: ESSIM Sobel kernel size
    
    Returns:
        Dict with keys: LV, HFE, ESSIM, GC, Texture, Shape
    """
    lv = compute_LV(original_gray, filtered_gray, window_size)
    hfe = compute_HFE(original_gray, filtered_gray, radius)
    essim = compute_ESSIM(original_gray, filtered_gray, ksize)
    gc = compute_GC(original_gray, filtered_gray)
    
    return {
        "LV": lv,
        "HFE": hfe,
        "ESSIM": essim,
        "GC": gc,
        "Texture": (lv + hfe) / 2,
        "Shape": (essim + gc) / 2,
    }