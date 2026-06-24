import argparse
import os
import shutil
import sys

import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(
        description="Filter microscopy images based on QC metrics (CSV) and background analysis, then move rejected files."
    )
    # 경로 설정
    parser.add_argument(
        "--root_dir",
        type=str,
        default=r"D:\From_C_drive\MIP",
        help="Root directory containing image folders.",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=r"D:\From_C_drive\MIP\QC_Output\qc_metrics_raw_per_channel.csv",
        help="Path to the QC metrics CSV file.",
    )
    parser.add_argument(
        "--reject_dir_name",
        type=str,
        default="Rejected_Images",
        help="Name of the folder to store rejected images (created inside root_dir).",
    )

    # 임계값 설정 (User Requirements)
    parser.add_argument(
        "--min_max_laplacian",
        type=float,
        default=800000.0,
        help="Reject if MAX Laplacian variance across channels is below this (Default: 8*10^5). you can change this value.",
    )
    parser.add_argument(
        "--min_avg_intensity",
        type=float,
        default=1000.0,
        help="Reject if AVERAGE intensity across channels is below this.",
    )

    # 배경 필터링 설정
    parser.add_argument(
        "--bg_pixel_threshold",
        type=float,
        default=1000.0,
        help="Sum of scaled RGB channels below this value is considered background.",
    )
    parser.add_argument(
        "--max_bg_fraction",
        type=float,
        default=0.70,
        help="Reject if background pixel fraction is above this (0.70 = 70%).",
    )

    return parser.parse_args()


def check_background_fraction(img_path, bg_pixel_thresh=1000, max_bg_frac=0.70):
    """
    Checks if the image has excessive background noise based on linear scaling.
    Returns: (is_rejected (bool), background_fraction (float))
    """
    try:
        img = tifffile.imread(img_path)

        # Ensure 3D array (H, W, C)
        if img.ndim == 2:
            img = img[..., np.newaxis]

        h, w, c = img.shape
        scaled_channels = []

        # Channel-wise Linear Scaling (ignoring top 0.5% outliers)
        for i in range(c):
            ch_data = img[..., i].astype(np.float32)

            # Robust Max: 99.5th percentile
            robust_max = np.percentile(ch_data, 99.5)

            # Clip and Scale to 0-65535
            ch_data = np.clip(ch_data, 0, robust_max)
            if robust_max > 0:
                ch_data = (ch_data / robust_max) * 65535.0

            scaled_channels.append(ch_data)

        # Sum of scaled channels (R + G + B ...)
        sum_img = np.sum(np.stack(scaled_channels, axis=-1), axis=-1)

        # Calculate background fraction
        # Pixel is background if Sum < threshold
        bg_mask = sum_img < bg_pixel_thresh
        bg_fraction = np.sum(bg_mask) / bg_mask.size

        is_rejected = bg_fraction > max_bg_frac
        return is_rejected, bg_fraction

    except Exception as e:
        print(f"⚠️ Error reading image for bg check: {img_path} ({e})")
        return (
            False,
            0.0,
        )  # Error safe: do not reject if read fails (manual check needed)


def move_to_reject(file_path, root_dir, reject_root):
    """Moves the file to the reject directory, maintaining folder structure."""
    try:
        # Construct relative path to maintain structure (e.g., GroupA/img.tif)
        rel_path = os.path.relpath(file_path, root_dir)
        dest_path = os.path.join(reject_root, rel_path)

        # Create destination directory
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        # Move file
        shutil.move(file_path, dest_path)
        return True
    except Exception as e:
        print(f"❌ Failed to move {file_path}: {e}")
        return False


def run_filtering(args):
    print("🚀 Starting Advanced Image Filtering Pipeline...")

    # 1. Load CSV Data
    if not os.path.exists(args.csv_path):
        print(f"❌ CSV file not found: {args.csv_path}")
        return

    df = pd.read_csv(args.csv_path)
    print(f"📄 Loaded CSV with {len(df)} records.")

    # Prepare Reject Directory
    reject_dir = os.path.join(args.root_dir, args.reject_dir_name)
    if not os.path.exists(reject_dir):
        os.makedirs(reject_dir)
        print(f"📂 Created rejection directory: {reject_dir}")
    else:
        print(f"📂 Using existing rejection directory: {reject_dir}")

    # Determine Channel Columns dynamically
    lap_cols = [c for c in df.columns if "LaplacianVar" in c]
    int_cols = [c for c in df.columns if "Intensity" in c]

    stats = {
        "Total": 0,
        "Rejected_Metric": 0,
        "Rejected_Background": 0,
        "Kept": 0,
        "Missing": 0,
    }

    # Iterate through CSV records
    for idx, row in df.iterrows():
        stats["Total"] += 1

        # 100장마다 진행 상황 화면에 강제 출력
        if stats["Total"] % 100 == 0:
            print(f"⏳ 진행 중... ({stats['Total']}/{len(df)})", end="\r", flush=True)

        full_path = row["Full_Path"]
        filename = row["Filename"]

        # CSV Path Verification
        if not os.path.exists(full_path):
            # Try to recover path if moved or root changed
            # (Assumes structure: root_dir / Group / Filename)
            potential_path = os.path.join(args.root_dir, row["Group"], filename)
            if os.path.exists(potential_path):
                full_path = potential_path
            else:
                stats["Missing"] += 1
                continue

        # --- Criterion 1 & 2: Check Metrics from CSV ---
        # 1. Max Laplacian Variance check
        max_lap = row[lap_cols].max()
        # 2. Average Intensity check
        avg_int = row[int_cols].mean()

        is_metric_rejected = False
        reason = ""

        if max_lap < args.min_max_laplacian:
            is_metric_rejected = True
            reason = f"Low Sharpness (Max Lap {max_lap:.1f} < {args.min_max_laplacian})"
        elif avg_int < args.min_avg_intensity:
            is_metric_rejected = True
            reason = f"Low Intensity (Avg Int {avg_int:.1f} < {args.min_avg_intensity})"

        if is_metric_rejected:
            # Move immediately
            if move_to_reject(full_path, args.root_dir, reject_dir):
                stats["Rejected_Metric"] += 1
            continue  # Skip to next file

        # --- Criterion 3: Check Background (Open Image) ---
        # Only run this if it passed the first check (Efficiency)
        is_bg_rejected, bg_frac = check_background_fraction(
            full_path,
            bg_pixel_thresh=args.bg_pixel_threshold,
            max_bg_frac=args.max_bg_fraction,
        )

        if is_bg_rejected:
            if move_to_reject(full_path, args.root_dir, reject_dir):
                stats["Rejected_Background"] += 1
                # print(f"Rejected {filename}: Background {bg_frac*100:.1f}% > 70%") # Verbose off
        else:
            stats["Kept"] += 1

    # Final Report
    print("\n" + "=" * 50)
    print("📊 Filtering Summary Report")
    print("=" * 50)
    print(f"Total Images Processed: {stats['Total']}")
    print(
        f"❌ Rejected (Metrics):   {stats['Rejected_Metric']} (Low Sharpness/Intensity)"
    )
    print(
        f"❌ Rejected (Backgrnd): {stats['Rejected_Background']} (Empty/Noise > {args.max_bg_fraction*100}%)"
    )
    print(f"✅ Kept (Clean Data):   {stats['Kept']}")
    print(f"⚠️ Missing Files:       {stats['Missing']}")
    print("-" * 50)
    print(f"📁 Rejected files moved to: {reject_dir}")
    print("=" * 50)


if __name__ == "__main__":
    args = get_args()
    run_filtering(args)
