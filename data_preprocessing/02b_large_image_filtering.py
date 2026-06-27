#!/usr/bin/env python3
"""
Microscopy Image Quality Control (QC) and Filtering Pipeline

This script filters microscopy images based on pre-calculated sharpness (Laplacian variance) 
and intensity metrics from a CSV file, combined with a background coverage analysis. 
Low-quality images are automatically isolated into a rejection directory.
"""

import argparse
import shutil
import re
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile


def get_args():
    parser = argparse.ArgumentParser(
        description="Filter microscopy images based on QC metrics and background fraction analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root directory containing raw image folders.",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to the raw QC metrics CSV file.",
    )
    parser.add_argument(
        "--reject_dir_name",
        type=str,
        default="Rejected_Images",
        help="Directory name inside root_dir to store filtered images.",
    )
    parser.add_argument(
        "--min_max_laplacian",
        type=float,
        default=800000.0,
        help="Minimum required maximum Laplacian variance across channels.",
    )
    parser.add_argument(
        "--min_avg_intensity",
        type=float,
        default=1000.0,
        help="Minimum required average intensity across channels.",
    )
    parser.add_argument(
        "--bg_pixel_threshold",
        type=float,
        default=1000.0,
        help="Pixel threshold below which scaled channel sums are classified as background.",
    )
    parser.add_argument(
        "--max_bg_fraction",
        type=float,
        default=0.70,
        help="Maximum allowed background area fraction (e.g., 0.70 for 70%).",
    )
    return parser.parse_args()


def check_background_fraction(img_path: Path, bg_pixel_thresh: float, max_bg_frac: float) -> tuple:
    """Analyzes the fraction of the image area corresponding to uninformative background."""
    img = tifffile.imread(img_path)
    if img.ndim == 2:
        img = img[..., np.newaxis]

    h, w, c = img.shape
    scaled_channels = []

    # Channel-wise linear scaling ignoring the top 0.5% outliers
    for i in range(c):
        ch_data = img[..., i].astype(np.float32)
        robust_max = np.percentile(ch_data, 99.5)
        ch_data = np.clip(ch_data, 0, robust_max)
        
        if robust_max > 0:
            ch_data = (ch_data / robust_max) * 65535.0
        scaled_channels.append(ch_data)

    sum_img = np.sum(np.stack(scaled_channels, axis=-1), axis=-1)
    bg_mask = sum_img < bg_pixel_thresh
    bg_fraction = np.sum(bg_mask) / bg_mask.size
    is_rejected = bg_fraction > max_bg_frac
    
    return is_rejected, bg_fraction


def move_to_reject(file_path: Path, root_dir: Path, reject_root: Path) -> bool:
    """Relocates the target file to the rejection directory while preserving the folder hierarchy."""
    rel_path = file_path.relative_to(root_dir)
    dest_path = reject_root / rel_path
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(file_path), str(dest_path))
    return True


def main():
    args = get_args()
    root_dir = Path(args.root_dir)
    csv_path = Path(args.csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"[-] Specified CSV log does not exist: {csv_path}")

    df = pd.read_csv(csv_path)
    reject_dir = root_dir / args.reject_dir_name
    reject_dir.mkdir(parents=True, exist_ok=True)

    lap_cols = [c for c in df.columns if "LaplacianVar" in c]
    int_cols = [c for c in df.columns if "Intensity" in c]

    stats = {"Total": 0, "Rejected_Metric": 0, "Rejected_Background": 0, "Kept": 0, "Missing": 0}

    print("[*] Initiating advanced image filtration process...")
    for _, row in df.iterrows():
        stats["Total"] += 1
        full_path = Path(row["Full_Path"])

        if not full_path.exists():
            potential_path = root_dir / row["Group"] / row["Filename"]
            if potential_path.exists():
                full_path = potential_path
            else:
                stats["Missing"] += 1
                continue

        # Criterion 1 & 2: Sharpness and Intensity Checks
        max_lap = row[lap_cols].max()
        avg_int = row[int_cols].mean()

        if max_lap < args.min_max_laplacian or avg_int < args.min_avg_intensity:
            if move_to_reject(full_path, root_dir, reject_dir):
                stats["Rejected_Metric"] += 1
            continue

        # Criterion 3: Area Coverage Check via Direct Image Extraction
        is_bg_rejected, _ = check_background_fraction(
            full_path, 
            bg_pixel_thresh=args.bg_pixel_threshold, 
            max_bg_frac=args.max_bg_fraction
        )

        if is_bg_rejected:
            if move_to_reject(full_path, root_dir, reject_dir):
                stats["Rejected_Background"] += 1
        else:
            stats["Kept"] += 1

if __name__ == "__main__":
    main()