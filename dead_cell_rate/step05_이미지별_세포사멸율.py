#!/usr/bin/env python3
"""
세포사멸율 (Apoptosis Rate) - 이미지별 계산 및 저장
Multi-Otsu threshold 기준 intensity_rate 계산

Purpose:
    - QC 통과한 모든 이미지에 대해 개별 세포사멸율 계산
    - 이미지 이름과 세포사멸율 저장 (DPT Kendall 상관 분석용)

Usage:
    python step05_이미지별_세포사멸율.py
"""

import argparse
import csv
import os
import re
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter
from skimage.restoration import rolling_ball
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(
        description="Calculate per-image apoptosis rate using fixed threshold",
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
        default=r"C:\Users\admin\Desktop\이미지별_세포사멸율_7200_핵제외비율_2.13",
        help="Output CSV file path",
    )

    # Fixed threshold from Multi-Otsu
    parser.add_argument(
        "--threshold",
        type=float,
        default=39217.8199851941,  # 지금 코드에 있는거 기반은 38704.2570189115 이거 나온다. 1.3% 정도 차이로 크지 않다.
        help="Fixed threshold value (Multi-Otsu result)",
    )

    # Preprocessing
    parser.add_argument("--gaussian_sigma", type=float, default=1.0)
    parser.add_argument(
        "--rolling_ball_radius",
        type=int,
        default=25,
        help="Rolling ball radius (0 = no background subtraction)",
    )

    # Edge filtering
    parser.add_argument("--exclude_edge_patches", action="store_true", default=True)
    parser.add_argument("--original_size", type=int, default=1024)
    parser.add_argument("--patch_size", type=int, default=128)

    # Nucleus filtering
    parser.add_argument("--min_nucleus_fraction", type=float, default=0.25)

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


def get_class_label(folder_name):
    return "Control" if folder_name.startswith("Control") else folder_name


def main():
    args = get_args()

    print("=" * 70)
    print("🧬 이미지별 세포사멸율 계산 (intensity_rate)")
    print("=" * 70)
    print(f"   Threshold: {args.threshold:.2f}")
    print(f"   Gaussian Sigma: {args.gaussian_sigma}")
    print(f"   Rolling Ball Radius: {args.rolling_ball_radius}")
    print("=" * 70)

    data_dir = Path(args.data_dir)
    folder_list = get_folder_list()

    total_pixels = args.patch_size**2
    min_nucleus_pixels = int(total_pixels * args.min_nucleus_fraction)

    results = []

    for folder_name in folder_list:
        class_label = get_class_label(folder_name)

        # QC 폴더만 사용
        qc_folder = data_dir / folder_name / "QC"
        mask_dir = qc_folder / "stardist_mask"
        cytox_dir = qc_folder / "cytoxgreen"

        if not mask_dir.exists() or not cytox_dir.exists():
            print(f"⚠️ 폴더 없음: {folder_name}")
            continue

        mask_files = list(mask_dir.glob("*_mask.tif")) + list(
            mask_dir.glob("*_mask.tiff")
        )

        # 가장자리 패치 제외
        if args.exclude_edge_patches:
            mask_files = [
                f
                for f in mask_files
                if not is_edge_patch(f.name, args.original_size, args.patch_size)
            ]

        print(f"\n[{folder_name}] {len(mask_files)}개 이미지 처리중...")

        for mask_file in tqdm(
            mask_files, desc=f"   {folder_name}", leave=True, ncols=100
        ):
            try:
                # cytox 파일 찾기
                cytox_filename = mask_file.name.replace(
                    "_mask.tif", "_cytox.tif"
                ).replace("_mask.tiff", "_cytox.tiff")
                cytox_file = cytox_dir / cytox_filename

                if not cytox_file.exists():
                    continue

                # 이미지 로드
                mask = tifffile.imread(mask_file)
                cytox = tifffile.imread(cytox_file)

                if cytox.ndim == 3:
                    cytox = cytox[:, :, 0]

                # 핵 마스크
                nucleus_mask = mask > 0
                total_nucleus_pixels = np.sum(nucleus_mask)

                if total_nucleus_pixels < min_nucleus_pixels:
                    continue

                # 전처리
                cytox_processed = apply_preprocessing(
                    cytox,
                    gaussian_sigma=args.gaussian_sigma,
                    rolling_ball_radius=args.rolling_ball_radius,
                )

                # 핵 영역의 intensity 추출
                cytox_vals = cytox_processed[nucleus_mask].astype(np.float64)

                # intensity_rate 계산
                total_intensity = np.sum(cytox_vals)
                apoptotic_mask = cytox_vals >= args.threshold
                apoptotic_intensity = np.sum(cytox_vals[apoptotic_mask])

                intensity_rate = (
                    apoptotic_intensity / total_intensity
                    if total_intensity > 0
                    else 0.0
                )

                # 결과 저장
                results.append(
                    {
                        "filename": mask_file.stem,  # _mask 제외한 파일명
                        "folder": folder_name,
                        "class": class_label,
                        "total_nucleus_pixels": int(total_nucleus_pixels),
                        "total_intensity": float(total_intensity),
                        "apoptotic_intensity": float(apoptotic_intensity),
                        "intensity_rate": float(intensity_rate),
                    }
                )

            except Exception as e:
                continue

    # CSV 저장
    print("\n" + "=" * 70)
    print("💾 결과 저장 중...")

    if len(results) == 0:
        print("❌ 결과 없음")
        return

    try:
        with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"✅ 저장 완료: {args.output_csv}")
        print(f"   총 {len(results)}개 이미지")
    except Exception as e:
        print(f"❌ 저장 실패: {e}")

    # 요약 출력
    print("\n" + "=" * 70)
    print("📊 클래스별 요약")
    print("=" * 70)

    class_stats = {}
    for r in results:
        cls = r["class"]
        if cls not in class_stats:
            class_stats[cls] = []
        class_stats[cls].append(r["intensity_rate"])

    print(f"{'Class':<12} {'N':<8} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print("-" * 68)
    for cls in ["Control", "GBA", "LRRK2", "SNCA"]:
        if cls in class_stats:
            rates = class_stats[cls]
            print(
                f"{cls:<12} {len(rates):<8} {np.mean(rates):.6f}   {np.std(rates):.6f}   {np.min(rates):.6f}   {np.max(rates):.6f}"
            )

    print("\n✅ 완료! DPT Kendall 상관 분석에 사용 가능합니다.")


if __name__ == "__main__":
    main()
