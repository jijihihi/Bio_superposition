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
        default=r"C:\Users\admin\Desktop\cropped_image",
        help="Directory containing cropped patches.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=r"C:\Users\admin\Desktop\cropped_image\Rejected_cropped_image",
        help="Directory to move REJECTED patches to.",
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
    # 상위 n% (Max Cutoff용)
    parser.add_argument(
        "--max_saturation_percent",
        type=float,
        default=0.5,
        help="Percentile to saturate at the top (e.g., 0.5 means top 0.5%% pixels are mapped to 65535).",
    )

    # [추가됨] 하위 n% (Min Cutoff용 - 배경 제거 기준)
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
    """
    QC Pipeline:
    1. Signal Sum Check
    2. Laplacian Variance Check
    3. Linear Scaling Background Check (Min-Max Stretching with configurable cutoffs)
    """
    if img.ndim == 2:
        img = img[..., np.newaxis]

    h, w, c = img.shape

    # Step 0: Channel Std Sum Check (배경이 많은 이미지 필터링)
    channel_stds = [np.std(img[..., i].astype(np.float32)) for i in range(c)]
    std_sum = sum(channel_stds)

    if std_sum < args.min_std_sum:
        return True, f"Low Variation (std_sum={std_sum:.0f})"

    # Step 1: Signal Sum Check [cite: 6, 7]
    top_5_idx = int(h * w * 0.15)
    total_signal = 0
    for i in range(c):
        flat_c = img[..., i].flatten()
        if top_5_idx > 0:
            flat_c_part = np.partition(flat_c, -top_5_idx)
            total_signal += np.sum(flat_c_part[-top_5_idx:])

    if total_signal < args.min_signal_sum:
        return True, f"Weak Signal Sum ({total_signal:.0f})"

    # Step 2: Laplacian Variance Check [cite: 8]
    flat_max = np.max(img, axis=2).astype(np.float64)
    laplacian_var = cv2.Laplacian(flat_max, cv2.CV_64F).var()

    if laplacian_var < args.min_laplacian_var:
        return True, f"Blurry ({laplacian_var:.0f})"

    # Step 3: Fiji-Style Linear Scaling [cite: 9, 10, 11, 12, 13, 14]
    scaled_img = np.zeros_like(img, dtype=np.float32)
    target_max = 65535.0
    MIN_STD_THRESHOLD = 655.0

    for i in range(c):
        channel = img[..., i].astype(np.float32)
        raw_std = np.std(channel)

        if raw_std < MIN_STD_THRESHOLD:
            scaled_channel = channel
        else:
            # [변경됨] args에서 설정한 min/max saturation percent 사용
            # Min Cutoff: 하위 n% (배경 제거)
            min_cutoff = np.percentile(channel, args.min_saturation_percent)

            # Max Cutoff: 상위 n% (세포 신호 보존)
            max_cutoff = np.percentile(channel, 100 - args.max_saturation_percent)

            if max_cutoff <= min_cutoff:
                scaled_channel = channel * 0
            else:
                # 배경 빼기 (Min Cutoff 이하 0으로)
                channel_shifted = channel - min_cutoff

                # 스케일링
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
    print(f"   0. Std Sum         < {args.min_std_sum}")
    print(f"   1. Signal Sum      < {args.min_signal_sum}")
    print(f"   2. Laplacian Var   < {args.min_laplacian_var}")
    print(f"   3. Background Frac > {args.max_bg_fraction*100:.0f}%")
    print(
        f"      (Scaling: Min {args.min_saturation_percent}% ~ Max Top {args.max_saturation_percent}%)"
    )  # 설정값 출력 확인용 추가

    stats = {"Total": 0, "Rejected": 0, "Passed": 0}
    files_to_process = []

    for root, dirs, files in os.walk(args.input_dir):
        if os.path.abspath(args.output_dir) in os.path.abspath(root):
            continue
        for f in files:
            if f.lower().endswith((".tif", ".tiff")):
                files_to_process.append(os.path.join(root, f))

    print(f"   - Found {len(files_to_process)} patches.")

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
    print(
        f"✅ Passed:   {stats['Passed']} ({(stats['Passed']/stats['Total'])*100:.1f}%)"
    )
    print(
        f"❌ Rejected: {stats['Rejected']} ({(stats['Rejected']/stats['Total'])*100:.1f}%)"
    )
    print("=" * 50)


if __name__ == "__main__":
    args = get_args()
    if os.path.exists(args.input_dir):
        run_filtering(args)
    else:
        print("❌ Input directory not found.")
