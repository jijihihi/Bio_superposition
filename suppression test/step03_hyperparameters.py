# ==============================================================================
# step03_hyperparameters.py
# ==============================================================================
# Texture & Shape Suppression 실험용 하이퍼파라미터 설정
# - 모든 공간적 파라미터는 128x128 이미지 기준으로 스케일링됨
# - 원본 논문: 224x224 이미지, 스케일 비율: 128/224 ≈ 0.571
#
# 사용법: step05_texture_shape_eval.py에서 import하여 사용
# ==============================================================================

# ==============================================================================
# Image Configuration
# ==============================================================================
IMAGE_SIZE = 128  # 128x128 images
ORIGINAL_IMAGE_SIZE = 224  # Original paper image size
SCALE_RATIO = IMAGE_SIZE / ORIGINAL_IMAGE_SIZE  # ≈ 0.571

# ==============================================================================
# Metric Parameters (논문 Table 1 기준, 128x128 스케일링)
# ==============================================================================
# 논문 원문: w = r = k = 11 for 224x224
# 스케일링: 11 * 0.571 ≈ 6

METRIC_PARAMS = {
    "window_size": 5,  # LV: Local Variance window size (논문 w=11)
    "freq_radius": 5,  # HFE: High Frequency Energy radius (논문 r=11)
    "sobel_ksize": 5,  # ESSIM: Sobel kernel size (논문 k=11, OpenCV 제한으로 5 사용)
}

# ==============================================================================
# Transform Kernel Sizes (128x128 스케일링)
# ==============================================================================
# 원본: [5, 7, 9, 11, 13, 15] → 스케일링 후 홀수로: [3, 5, 5, 7, 7, 9]

KERNEL_SIZES_ORIGINAL = [5, 7, 9, 11, 13, 15]
KERNEL_SIZES_SCALED = [3, 5, 5, 7, 7, 9]  # 128x128 이미지용

# ==============================================================================
# Texture-Suppressing Transforms (Bilateral, Gaussian, NLMeans, Box, Median)
# ==============================================================================

# Bilateral Filter: diagonal sweep (σc, k)
# σc는 intensity 기반이므로 스케일링 불필요, k는 공간 파라미터로 스케일링
BILATERAL_PARAMS = {
    "diagonal_sweep": [
        {"sigma_color": 50, "k": 3},
        {"sigma_color": 80, "k": 3},
        {"sigma_color": 80, "k": 5},
        {"sigma_color": 110, "k": 5},
        {"sigma_color": 140, "k": 5},
        {"sigma_color": 110, "k": 7},
        {"sigma_color": 140, "k": 7},
        {"sigma_color": 170, "k": 7},
        {"sigma_color": 170, "k": 7},
        {"sigma_color": 200, "k": 9},
    ],
    "sigma_space": 43,  # 75 * 0.571 ≈ 43
}

# Gaussian Blur: diagonal sweep (σ, k)
# σ는 커널 대비 비율이므로 유지, k는 스케일링
GAUSSIAN_PARAMS = {
    "diagonal_sweep": [
        {"sigma": 0.66, "k": 3},
        {"sigma": 1.0, "k": 3},
        {"sigma": 0.66, "k": 5},
        {"sigma": 1.0, "k": 5},
        {"sigma": 1.33, "k": 5},
        {"sigma": 1.66, "k": 5},
        {"sigma": 1.33, "k": 7},
        {"sigma": 1.66, "k": 7},
        {"sigma": 2.0, "k": 7},
        {"sigma": 2.33, "k": 7},
        {"sigma": 2.0, "k": 9},
        {"sigma": 2.33, "k": 9},
        {"sigma": 1.66, "k": 3},
        {"sigma": 1.66, "k": 5},
        {"sigma": 1.66, "k": 7},
        {"sigma": 1.66, "k": 9},
    ],
}

# NLMeans Denoising: diagonal sweep (h, k)
# h는 노이즈 대비 상대값이므로 유지, k(patch_size)는 스케일링
NLMEANS_PARAMS = {
    "diagonal_sweep": [
        {"h": 5, "k": 3},
        {"h": 5, "k": 5},
        {"h": 10, "k": 5},
        {"h": 15, "k": 7},
        {"h": 20, "k": 7},
        {"h": 25, "k": 9},
    ],
    "h_scale": 100,  # h는 h/100으로 사용
}

# Box Blur: fixed kernel size
BOX_PARAMS = {
    "k": 5,  # 원본 k=9 → 스케일링 후 5
}

# Median Filter: fixed kernel size
MEDIAN_PARAMS = {
    "k": 5,  # 원본 k=9 → 스케일링 후 5, 홀수 필수
}

# ==============================================================================
# Shape-Suppressing Transforms (PatchShuffle)
# ==============================================================================

# PatchShuffle: grid size
# 224/6 ≈ 37px per patch → 128/4 = 32px per patch (유사 비율)
PATCH_SHUFFLE_PARAMS = {
    "grid_size": 4,  # 원본 grid=6-7 → 스케일링 후 4
}

# ==============================================================================
# Evaluation Settings
# ==============================================================================
EVAL_SETTINGS = {
    "n_per_class": 500,  # 클래스당 샘플 수
    "seed": 42,  # 랜덤 시드
}


# ==============================================================================
# Quick Access Function
# ==============================================================================
def get_all_params():
    """Return all hyperparameters as a single dict for logging."""
    return {
        "image_size": IMAGE_SIZE,
        "scale_ratio": SCALE_RATIO,
        "metric_params": METRIC_PARAMS,
        "bilateral": BILATERAL_PARAMS,
        "gaussian": GAUSSIAN_PARAMS,
        "nlmeans": NLMEANS_PARAMS,
        "box": BOX_PARAMS,
        "median": MEDIAN_PARAMS,
        "patch_shuffle": PATCH_SHUFFLE_PARAMS,
        "eval_settings": EVAL_SETTINGS,
    }


if __name__ == "__main__":
    import json

    print("=" * 60)
    print("Hyperparameters for 128x128 Image Evaluation")
    print("=" * 60)
    print(json.dumps(get_all_params(), indent=2))
