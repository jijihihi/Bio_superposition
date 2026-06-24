#!/usr/bin/env python3
"""
세포사멸율 (Apoptosis Rate) 계산 스크립트 - Rolling Ball + Global Thresholding
픽셀 풀링 방식: 이미지별 평균이 아닌, 클래스별 전체 픽셀을 합산하여 비율 계산

포함된 알고리즘:
    - 기본: Otsu, Li, Yen, Isodata, Mean, Triangle
    - 엔트로피: Shanbhag, Huang
    - 높은 Threshold: Rosin (unimodal), Kittler, Multi-Otsu 3class
    - 통계: Percentile, MAD, IQR 기반
"""

import argparse
import csv
import os
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter
from skimage.filters import (threshold_isodata, threshold_li, threshold_mean,
                             threshold_multiotsu, threshold_otsu,
                             threshold_triangle, threshold_yen)
from skimage.restoration import rolling_ball
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(
        description="Calculate apoptosis rate using Rolling Ball + GLOBAL thresholding (pixel pooling).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default=r"C:\Users\admin\Desktop\세포사멸율_data",
        help="Directory containing QC folders",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=r"C:\Users\admin\Desktop\세포사멸율_pooled_결과_7200_03.27.csv",
        help="Output CSV file path",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--total_samples",
        type=int,
        default=28800,
        help="Total samples (C4:C18:GBA_C19:SNCA_C19:GBA:LRRK2:SNCA = 1000:1000:500:500:3000:3000:3000)",
    )
    parser.add_argument("--gaussian_sigma", type=float, default=1.0)
    parser.add_argument("--rolling_ball_radii", type=int, nargs="+", default=[25])
    parser.add_argument("--light_background", action="store_true", default=False)
    parser.add_argument("--exclude_edge_patches", action="store_true", default=True)
    parser.add_argument("--original_size", type=int, default=1024)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument(
        "--min_nucleus_fraction", type=float, default=0.25
    )  # 전체 픽셀 pooling 즉 핵과 cytoxgreen pooling해서 threshold 잡고 세포사멸율 확인하는거니까 핵 비율 fraction할 필요 없지

    return parser.parse_args()


def is_edge_patch(filename, original_size=1024, patch_size=128):
    match = re.search(r"_x(\d+)_y(\d+)", filename)
    if not match:
        return True
    x, y = int(match.group(1)), int(match.group(2))
    max_coord = original_size - patch_size
    return x == 0 or x == max_coord or y == 0 or y == max_coord


def apply_preprocessing(image, gaussian_sigma=1.0, rolling_ball_radius=0):
    img_float = image.astype(np.float64)
    if gaussian_sigma > 0:
        img_float = gaussian_filter(img_float, sigma=gaussian_sigma)
    if rolling_ball_radius == 0:
        return img_float
    try:
        background = rolling_ball(img_float, radius=rolling_ball_radius)
        return np.maximum(img_float - background, 0)
    except:
        return img_float


# ============================================================================
# HIGH THRESHOLD ALGORITHMS (수학적 근거 있음)
# ============================================================================


def rosin_threshold(data):
    """
    Rosin's Unimodal Thresholding (Rosin, 2001)
    - 분포의 peak에서 tail까지 선을 긋고, 가장 먼 점을 threshold로 사용
    - 배경 >> foreground인 skewed 분포에 적합 (CytoxGreen처럼)
    - 높은 threshold 경향
    """
    hist, bin_edges = np.histogram(data.ravel(), bins=256)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Peak 찾기 (histogram의 최댓값)
    peak_idx = np.argmax(hist)

    # Tail 찾기 (histogram 끝에서 0이 아닌 마지막 bin)
    nonzero_indices = np.where(hist > 0)[0]
    if len(nonzero_indices) == 0:
        return bin_centers[128]
    tail_idx = nonzero_indices[-1]

    if peak_idx >= tail_idx:
        return bin_centers[peak_idx]

    # Peak에서 tail까지 직선 정의
    p1 = np.array([peak_idx, hist[peak_idx]])
    p2 = np.array([tail_idx, hist[tail_idx]])

    # 각 점에서 직선까지의 수직 거리 계산
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)
    if line_len == 0:
        return bin_centers[peak_idx]
    line_unitvec = line_vec / line_len

    max_dist = 0
    best_idx = peak_idx

    for i in range(peak_idx, tail_idx + 1):
        point = np.array([i, hist[i]])
        vec_to_point = point - p1
        # 수직 거리 = 외적 / 선분 길이
        dist = abs(np.cross(line_unitvec, vec_to_point))
        if dist > max_dist:
            max_dist = dist
            best_idx = i

    return bin_centers[best_idx]


def kittler_threshold(data):
    """
    Kittler-Illingworth Minimum Error Thresholding (Kittler & Illingworth, 1986)
    - 두 Gaussian 분포의 혼합으로 가정하고 오분류 오차 최소화
    - 수학적으로 rigorous한 방법
    """
    hist, bin_edges = np.histogram(data.ravel(), bins=256)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    hist = hist.astype(np.float64)

    total = hist.sum()
    if total == 0:
        return bin_centers[128]

    hist = hist / total

    min_criterion = np.inf
    best_thresh = bin_centers[0]

    for t in range(1, 255):
        # 두 클래스의 확률
        p1 = hist[: t + 1].sum()
        p2 = hist[t + 1 :].sum()

        if p1 < 1e-10 or p2 < 1e-10:
            continue

        # 두 클래스의 평균
        mu1 = np.sum(bin_centers[: t + 1] * hist[: t + 1]) / p1
        mu2 = np.sum(bin_centers[t + 1 :] * hist[t + 1 :]) / p2

        # 두 클래스의 분산
        var1 = np.sum((bin_centers[: t + 1] - mu1) ** 2 * hist[: t + 1]) / p1
        var2 = np.sum((bin_centers[t + 1 :] - mu2) ** 2 * hist[t + 1 :]) / p2

        if var1 <= 0 or var2 <= 0:
            continue

        # Kittler criterion
        criterion = p1 * np.log(np.sqrt(var1) / p1) + p2 * np.log(np.sqrt(var2) / p2)

        if criterion < min_criterion:
            min_criterion = criterion
            best_thresh = bin_centers[t]

    return best_thresh


def multiotsu_high_threshold(data):
    """
    Multi-Otsu 3-class thresholding, 더 높은 threshold 선택
    - Otsu의 3-class 확장, 두 개의 threshold 중 높은 것 사용
    """
    try:
        thresholds = threshold_multiotsu(data, classes=3)
        return thresholds[-1]  # 가장 높은 threshold
    except:
        return threshold_otsu(data)


def shanbhag_threshold(data):
    """Shanbhag entropy thresholding"""
    hist, bin_edges = np.histogram(data.ravel(), bins=256)
    hist = hist.astype(np.float64) / (hist.sum() + 1e-10)
    eps = 1e-10
    best_thresh, max_entropy = 0, -np.inf

    for t in range(1, 255):
        p1, p2 = hist[: t + 1].sum(), hist[t + 1 :].sum()
        if p1 < eps or p2 < eps:
            continue
        h1 = hist[: t + 1] / (p1 + eps)
        h2 = hist[t + 1 :] / (p2 + eps)
        entropy = -np.sum(h1[h1 > eps] * np.log(h1[h1 > eps] + eps)) - np.sum(
            h2[h2 > eps] * np.log(h2[h2 > eps] + eps)
        )
        if entropy > max_entropy:
            max_entropy, best_thresh = entropy, t

    return bin_edges[best_thresh]


def huang_threshold(data):
    """Huang's fuzzy thresholding"""
    hist, bin_edges = np.histogram(data.ravel(), bins=256)
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return bin_edges[128]
    hist = hist / total
    S = np.cumsum(hist)
    W = np.cumsum(np.arange(256) * hist)

    best_thresh, min_ent = 0, np.inf
    for t in range(1, 255):
        if S[t] < 1e-10 or (1 - S[t]) < 1e-10:
            continue
        mu1 = W[t] / S[t]
        mu2 = (W[-1] - W[t]) / (1 - S[t])
        ent = 0
        for i in range(256):
            if hist[i] < 1e-10:
                continue
            mu = mu1 if i <= t else mu2
            c = 1 / (1 + abs(i - mu))
            if c > 1e-10 and c < 1:
                ent -= hist[i] * (c * np.log(c) + (1 - c) * np.log(1 - c))
        if ent < min_ent:
            min_ent, best_thresh = ent, t

    return bin_edges[best_thresh]


def calculate_global_thresholds(all_cytox_intensity, radius_label=""):
    """전체 데이터에서 Global Threshold 계산"""
    print(f"\n📊 Global Threshold 계산 중... {radius_label}")
    print(f"   총 핵 픽셀 수: {len(all_cytox_intensity):,}")

    thresholds = {}

    # Basic methods
    for name, func in [
        ("otsu", threshold_otsu),
        ("li", threshold_li),
        ("yen", threshold_yen),
        ("isodata", threshold_isodata),
        ("mean", threshold_mean),
        ("triangle", threshold_triangle),
    ]:
        try:
            thresholds[name] = func(all_cytox_intensity)
        except:
            thresholds[name] = np.nan

    # Entropy-based
    for name, func in [("shanbhag", shanbhag_threshold), ("huang", huang_threshold)]:
        try:
            thresholds[name] = func(all_cytox_intensity)
        except:
            thresholds[name] = np.nan

    # HIGH THRESHOLD ALGORITHMS (수학적 근거)
    for name, func in [
        ("rosin", rosin_threshold),
        ("kittler", kittler_threshold),
        ("multiotsu_high", multiotsu_high_threshold),
    ]:
        try:
            thresholds[name] = func(all_cytox_intensity)
        except:
            thresholds[name] = np.nan

    # Percentile-based
    for p in [75, 90, 95, 99, 99.5, 99.9]:
        key = f"p{int(p)}" if p == int(p) else f"p{p}".replace(".", "_")
        try:
            thresholds[key] = np.percentile(all_cytox_intensity, p)
        except:
            thresholds[key] = np.nan

    # Statistics-based
    try:
        median_val = np.median(all_cytox_intensity)
        mean_val = np.mean(all_cytox_intensity)
        std_val = np.std(all_cytox_intensity)
        mad = np.median(np.abs(all_cytox_intensity - median_val)) * 1.4826
        q1, q3 = np.percentile(all_cytox_intensity, [25, 75])
        iqr = q3 - q1

        thresholds["median_2std"] = median_val + 2 * std_val
        thresholds["median_3std"] = median_val + 3 * std_val
        thresholds["mean_3std"] = mean_val + 3 * std_val
        thresholds["median_3mad"] = median_val + 3 * mad
        thresholds["median_4mad"] = median_val + 4 * mad
        thresholds["q3_1_5iqr"] = q3 + 1.5 * iqr
        thresholds["q3_3iqr"] = q3 + 3 * iqr
    except:
        pass

    # Print comparison (low to high)
    print("\n   [Threshold 비교: 낮음 → 높음]")
    sorted_thresh = sorted(
        [(k, v) for k, v in thresholds.items() if not np.isnan(v)], key=lambda x: x[1]
    )
    for name, val in sorted_thresh[:5]:
        print(f"   {name}: {val:.2f} (낮음)")
    print("   ...")
    for name, val in sorted_thresh[-5:]:
        print(f"   {name}: {val:.2f} (높음)")

    return thresholds


def get_folder_list():
    return [
        "Control_C4",
        "Control_C18",
        "Control_GBA_C19",
        "Control_SNCA_C19",
        "SNCA",
        "GBA",
        "LRRK2",
    ]


def get_sampling_plan(total_samples):
    base_ratio = {
        "Control_C4": 2400,
        "Control_C18": 2400,
        "Control_GBA_C19": 1200,
        "Control_SNCA_C19": 1200,
        "SNCA": 7200,
        "GBA": 7200,
        "LRRK2": 7200,
    }
    scale = total_samples / sum(base_ratio.values())
    return {k: int(v * scale) for k, v in base_ratio.items()}


def get_class_label(folder_name):
    return "Control" if folder_name.startswith("Control") else folder_name


def load_images_from_folder(folder_path, args, is_qc=True):
    mask_dir = folder_path / "stardist_mask"
    cytox_dir = folder_path / "cytoxgreen"

    if not mask_dir.exists() or not cytox_dir.exists():
        return []

    mask_files = list(mask_dir.glob("*_mask.tif")) + list(mask_dir.glob("*_mask.tiff"))

    if args.exclude_edge_patches:
        mask_files = [
            f
            for f in mask_files
            if not is_edge_patch(f.name, args.original_size, args.patch_size)
        ]

    total_pixels = args.patch_size**2
    min_nucleus_pixels = int(total_pixels * args.min_nucleus_fraction)

    images = []
    for mask_file in mask_files:
        try:
            cytox_file = (
                mask_file.parent.parent
                / "cytoxgreen"
                / mask_file.name.replace("_mask", "_cytox")
            )
            if not cytox_file.exists():
                continue

            mask = tifffile.imread(mask_file)
            cytox = tifffile.imread(cytox_file)
            if cytox.ndim == 3:
                cytox = cytox[:, :, 0]

            nucleus_mask = mask > 0
            if np.sum(nucleus_mask) < min_nucleus_pixels:
                continue

            images.append(
                {
                    "filename": mask_file.name,
                    "cytox_image": cytox,
                    "nucleus_mask": nucleus_mask,
                    "total_nucleus_pixels": int(np.sum(nucleus_mask)),
                    "is_qc": is_qc,
                }
            )
        except:
            continue

    return images


def run_apoptosis_calculation(args):
    print("=" * 70)
    print("🧬 세포사멸율 계산 (Rosin/Kittler/Multi-Otsu 포함)")
    print("=" * 70)

    random.seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    radii = args.rolling_ball_radii
    folder_list = get_folder_list()
    sampling_plan = get_sampling_plan(args.total_samples)

    print(f"   Gaussian Sigma: {args.gaussian_sigma}")
    print(f"   Rolling Ball Radii: {radii}")
    print(f"   Total Samples: {args.total_samples}")
    print("=" * 70)

    # Load images
    print("\n[PASS 1] 이미지 로드 및 샘플링...")
    all_image_data = []

    for folder_name in folder_list:
        target_count = sampling_plan.get(folder_name, 0)
        class_label = get_class_label(folder_name)
        folder_images = []

        for subfolder, is_qc in [("QC", True), ("QC_reject", False)]:
            folder = data_dir / folder_name / subfolder
            imgs = load_images_from_folder(folder, args, is_qc=is_qc)
            for img in imgs:
                img["folder"] = folder_name
                img["class"] = class_label
            folder_images.extend(imgs)

        if len(folder_images) > target_count:
            folder_images = random.sample(folder_images, target_count)

        all_image_data.extend(folder_images)
        print(f"   {folder_name}: {len(folder_images)}개")

    if not all_image_data:
        print("❌ 데이터 없음")
        return

    print(f"\n✅ 총 {len(all_image_data)}개 이미지")

    # Process each radius
    pooled_results = []

    for radius in radii:
        radius_label = f"radius={radius}" if radius > 0 else "no_rb"
        print(f"\n{'='*70}\n🔄 {radius_label}")

        all_cytox_intensity = []
        processed_data = []

        for img_data in tqdm(all_image_data, desc=f"   전처리", ncols=100):
            cytox = img_data["cytox_image"]
            mask = img_data["nucleus_mask"]
            cytox_proc = apply_preprocessing(cytox, args.gaussian_sigma, radius)
            cytox_vals = cytox_proc[mask].astype(np.float64)

            all_cytox_intensity.append(cytox_vals)
            processed_data.append({**img_data, "cytox_intensity": cytox_vals})

        all_cytox_concat = np.concatenate(all_cytox_intensity)
        global_thresholds = calculate_global_thresholds(all_cytox_concat, radius_label)

        # Calculate pooled rates per class
        for qc_filter in ["qc_only"]:
            for cls in ["Control", "GBA", "LRRK2", "SNCA"]:
                class_data = [d for d in processed_data if d["class"] == cls]
                if qc_filter == "qc_only":
                    class_data = [d for d in class_data if d["is_qc"]]

                if not class_data:
                    continue

                all_vals = np.concatenate([d["cytox_intensity"] for d in class_data])
                total_pixels = sum(d["total_nucleus_pixels"] for d in class_data)
                total_intensity = np.sum(all_vals)

                result = {
                    "rolling_ball_radius": radius,
                    "qc_filter": qc_filter,
                    "class": cls,
                    "total_images": len(class_data),
                    "total_nucleus_pixels": total_pixels,
                    "total_intensity": total_intensity,
                }

                for algo, thresh in global_thresholds.items():
                    result[f"global_thresh_{algo}"] = thresh
                    if np.isnan(thresh):
                        result[f"rate_{algo}"] = np.nan
                        result[f"intensity_rate_{algo}"] = np.nan
                        result[f"intensity_per_pixel_{algo}"] = np.nan
                    else:
                        apop_pixels = np.sum(all_vals >= thresh)
                        apop_intensity = np.sum(all_vals[all_vals >= thresh])
                        result[f"rate_{algo}"] = apop_pixels / total_pixels
                        result[f"intensity_rate_{algo}"] = (
                            apop_intensity / total_intensity
                            if total_intensity > 0
                            else np.nan
                        )
                        result[f"intensity_per_pixel_{algo}"] = (
                            apop_intensity / total_pixels
                        )

                pooled_results.append(result)

    # Save results
    print("\n💾 저장 중...")
    try:
        with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=pooled_results[0].keys())
            writer.writeheader()
            writer.writerows(pooled_results)
        print(f"✅ 저장: {args.output_csv}")
    except Exception as e:
        print(f"❌ 저장 실패: {e}")

    print("\n✅ 완료!")


if __name__ == "__main__":
    args = get_args()
    run_apoptosis_calculation(args)
