import argparse
import os
import shutil

import cv2
import numpy as np
import tifffile
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(
        description="QC: Signal + Laplacian + Linear Scaling BG Check."
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        default=r"D:\From_C_drive\cropped_image",
        help="Directory containing cropped patches.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=r"D:\From_C_drive\cropped_image\Rejected_cropped_image",
        help="Directory to move REJECTED patches to.",
    )

    # [추가] QC를 수행할 특정 하위 폴더 목록 지정 (기본값 설정)
    parser.add_argument(
        "--target_dirs",
        nargs="+",
        default=[
            "GBA_346",
            "GBA_WIMP4",
            "SNCA-G51D",
            "SNCA-G51D_isogenic",
            "SNCAx3_isogenic",
            "alpha_syn_1day",
            "alpha_syn_7day",
        ],
        help="List of specific subfolders inside input_dir to process for QC.",
    )

    # [0] 채널 표준편차 합
    parser.add_argument(
        "--min_std_sum",
        type=float,
        default=3500.0,
        help="Reject if sum of 3 channel std is below this.",
    )

    # [1] 시그널 강도
    parser.add_argument(
        "--min_signal_sum",
        type=float,
        default=8000.0,
        help="Reject if sum of top 15%% pixels is below this.",
    )

    # [2] 라플라시안 분산
    parser.add_argument(
        "--min_laplacian_var",
        type=float,
        default=50000.0,
        help="Reject if Laplacian variance is below this.",
    )

    # [3] 배경 필터링 설정 (Fiji Style Linear Scaling)
    parser.add_argument(
        "--max_saturation_percent",
        type=float,
        default=0.5,
        help="Percentile to saturate at the top (e.g., 0.5 means top 0.5%% pixels are mapped to 65535).",
    )

    parser.add_argument(
        "--min_saturation_percent",
        type=float,
        default=10.0,
        help="Percentile to saturate at the bottom (e.g., 10.0 means bottom 10%% pixels are mapped to 0).",
    )

    parser.add_argument(
        "--bg_threshold",
        type=float,
        default=4000.0,
        help="Pixel value threshold after scaling. Below this is background.",
    )
    parser.add_argument(
        "--max_bg_fraction",
        type=float,
        default=0.65,
        help="Reject if background fraction is > 65%%.",
    )

    return parser.parse_args()


def check_quality_combined(img, args):
    if img.ndim == 2:
        img = img[..., np.newaxis]

    h, w, c = img.shape

    channel_stds = [np.std(img[..., i].astype(np.float32)) for i in range(c)]
    std_sum = sum(channel_stds)

    if std_sum < args.min_std_sum:
        return True, f"Low Variation (std_sum={std_sum:.0f})"

    top_5_idx = int(h * w * 0.15)
    total_signal = 0
    for i in range(c):
        flat_c = img[..., i].flatten()
        if top_5_idx > 0:
            flat_c_part = np.partition(flat_c, -top_5_idx)
            total_signal += np.sum(flat_c_part[-top_5_idx:])

    if total_signal < args.min_signal_sum:
        return True, f"Weak Signal Sum ({total_signal:.0f})"

    flat_max = np.max(img, axis=2).astype(np.float64)
    laplacian_var = cv2.Laplacian(flat_max, cv2.CV_64F).var()

    if laplacian_var < args.min_laplacian_var:
        return True, f"Blurry ({laplacian_var:.0f})"

    scaled_img = np.zeros_like(img, dtype=np.float32)
    target_max = 65535.0
    MIN_STD_THRESHOLD = 655.0

    for i in range(c):
        channel = img[..., i].astype(np.float32)
        raw_std = np.std(channel)

        if raw_std < MIN_STD_THRESHOLD:
            scaled_channel = channel
        else:
            min_cutoff = np.percentile(channel, args.min_saturation_percent)
            max_cutoff = np.percentile(channel, 100 - args.max_saturation_percent)

            if max_cutoff <= min_cutoff:
                scaled_channel = channel * 0
            else:
                channel_shifted = channel - min_cutoff
                scale_factor = target_max / (max_cutoff - min_cutoff)
                scaled_channel = channel_shifted * scale_factor

        scaled_img[..., i] = np.clip(scaled_channel, 0, target_max)

    max_proj = np.max(scaled_img, axis=2)
    is_signal = max_proj > args.bg_threshold

    signal_pixels = np.count_nonzero(is_signal)
    bg_pixels = (h * w) - signal_pixels
    bg_frac = bg_pixels / (h * w)

    if bg_frac > args.max_bg_fraction:
        return True, f"Too Empty (Background: {bg_frac*100:.1f}%)"

    return False, "Pass"


def run_filtering(args):
    print("🚀 Starting Combined QC Pipeline...")
    print(f"   - Targets:         {', '.join(args.target_dirs)}")  # 지정 폴더 출력
    print(f"   0. Std Sum         < {args.min_std_sum}")
    print(f"   1. Signal Sum      < {args.min_signal_sum}")
    print(f"   2. Laplacian Var   < {args.min_laplacian_var}")
    print(f"   3. Background Frac > {args.max_bg_fraction*100:.0f}%")
    print(
        f"      (Scaling: Min {args.min_saturation_percent}% ~ Max Top {args.max_saturation_percent}%)"
    )

    stats = {"Total": 0, "Rejected": 0, "Passed": 0}
    files_to_process = []

    # [수정] 전체 폴더를 돌지 않고 지정한 하위 폴더 경로 목록을 먼저 생성
    target_paths = [os.path.join(args.input_dir, d) for d in args.target_dirs]

    for target_path in target_paths:
        if not os.path.exists(target_path):
            print(f"⚠️ Warning: Target directory not found, skipping: {target_path}")
            continue

        # 지정된 하위 폴더 내부만 순회
        for root, dirs, files in os.walk(target_path):
            # 탈락 폴더가 탐색 범위에 걸리는 것 방지
            if os.path.abspath(args.output_dir) in os.path.abspath(root):
                continue

            for f in files:
                if f.lower().endswith((".tif", ".tiff")):
                    files_to_process.append(os.path.join(root, f))

    print(f"   - Found {len(files_to_process)} patches in target folders.")

    if not files_to_process:
        print("❌ No images to process. Exiting.")
        return

    for file_path in tqdm(files_to_process, desc="Processing"):
        stats["Total"] += 1
        try:
            img = tifffile.imread(file_path)
            is_rejected, reason = check_quality_combined(img, args)

            if is_rejected:
                stats["Rejected"] += 1
                rel_path = os.path.relpath(file_path, args.input_dir)
                dest_path = os.path.join(args.output_dir, rel_path)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.move(file_path, dest_path)
            else:
                stats["Passed"] += 1

        except Exception as e:
            print(f"⚠️ Error: {file_path} - {e}")

    print("\n" + "=" * 50)
    print("📊 Final Summary")
    if stats["Total"] > 0:
        print(
            f"✅ Passed:   {stats['Passed']} ({(stats['Passed']/stats['Total'])*100:.1f}%)"
        )
        print(
            f"❌ Rejected: {stats['Rejected']} ({(stats['Rejected']/stats['Total'])*100:.1f}%)"
        )
    else:
        print("   No images were processed.")
    print("=" * 50)


if __name__ == "__main__":
    args = get_args()
    if os.path.exists(args.input_dir):
        run_filtering(args)
    else:
        print("❌ Input directory not found.")
