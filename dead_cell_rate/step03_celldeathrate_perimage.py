#!/usr/bin/env python3
"""
cell_death Rate Quantification Script - Image-by-Image Analysis

This script calculates the cell_death rate for each individual image patch.
It computes a single global Multi-Otsu threshold from the entire pooled dataset 
and applies it back to each individual image to extract standardized per-image metrics.
"""

import argparse
import csv
import re
from pathlib import Path
import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_multiotsu
from skimage.restoration import rolling_ball
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(
        description="Quantify individual image cell_death rates using a global Multi-Otsu threshold.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the root data directory.")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save the per-image CSV results.")
    parser.add_argument("--gaussian_sigma", type=float, default=1.0)
    parser.add_argument("--rolling_ball_radius", type=int, default=25)
    parser.add_argument("--exclude_edge_patches", type=bool, default=True)
    parser.add_argument("--original_size", type=int, default=1024)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--min_nucleus_fraction", type=float, default=0.25)
    return parser.parse_args()


def is_edge_patch(filename: str, original_size: int, patch_size: int) -> bool:
    match = re.search(r"_x(\d+)_y(\d+)", filename)
    if not match:
        return True
    x, y = int(match.group(1)), int(match.group(2))
    max_coord = original_size - patch_size
    return x == 0 or x == max_coord or y == 0 or y == max_coord


def apply_preprocessing(image: np.ndarray, gaussian_sigma: float, rolling_ball_radius: int) -> np.ndarray:
    img_float = image.astype(np.float64)
    if gaussian_sigma > 0:
        img_float = gaussian_filter(img_float, sigma=gaussian_sigma)
    if rolling_ball_radius > 0:
        background = rolling_ball(img_float, radius=rolling_ball_radius)
        img_float = np.maximum(img_float - background, 0)
    return img_float


def load_images_from_folder(folder_path: Path, args) -> list:
    mask_dir = folder_path / "stardist_mask"
    cytox_dir = folder_path / "cytoxgreen"

    if not mask_dir.exists() or not cytox_dir.exists():
        return []

    mask_files = list(mask_dir.glob("*_mask.tif")) + list(mask_dir.glob("*_mask.tiff"))
    if args.exclude_edge_patches:
        mask_files = [f for f in mask_files if not is_edge_patch(f.name, args.original_size, args.patch_size)]

    min_nucleus_pixels = int((args.patch_size**2) * args.min_nucleus_fraction)
    images = []

    for mask_file in mask_files:
        cytox_file = cytox_dir / mask_file.name.replace("_mask", "_cytox")
        if not cytox_file.exists():
            continue

        mask = tifffile.imread(mask_file)
        cytox = tifffile.imread(cytox_file)
        if cytox.ndim == 3:
            cytox = cytox[:, :, 0]

        nucleus_mask = mask > 0
        total_nucleus_pixels = int(np.sum(nucleus_mask))
        if total_nucleus_pixels < min_nucleus_pixels:
            continue

        images.append({
            "filename": mask_file.name,
            "cytox_image": cytox,
            "nucleus_mask": nucleus_mask,
            "total_nucleus_pixels": total_nucleus_pixels,
        })
    return images


def main():
    args = get_args()
    data_dir_path = Path(args.data_dir)
    target_classes = [d.name for d in data_dir_path.iterdir() if d.is_dir()]
    
    if not target_classes:
        print(f"[-] No folders found in {args.data_dir}")
        return

    print(f"[*] Loading and parsing image datasets from {len(target_classes)} folders...")
    all_image_data = []

    for folder_name in target_classes:
        class_label = "Control" if folder_name.startswith("Control") else folder_name
        folder_path = data_dir_path / folder_name / "QC"
        folder_images = load_images_from_folder(folder_path, args)

        for img in folder_images:
            img["folder"] = folder_name
            img["class"] = class_label
        all_image_data.extend(folder_images)

    if not all_image_data:
        raise ValueError("[-] No valid dataset found inside the specified directory.")

    print(f"[*] Total loaded images: {len(all_image_data)}")
    print("[*] Executing image preprocessing...")
    
    all_pixels_list = []
    for img_data in tqdm(all_image_data, desc="Preprocessing"):
        processed_cytox = apply_preprocessing(img_data["cytox_image"], args.gaussian_sigma, args.rolling_ball_radius)
        nuclear_intensities = processed_cytox[img_data["nucleus_mask"]].astype(np.float64)
        img_data["nuclear_intensities"] = nuclear_intensities
        all_pixels_list.append(nuclear_intensities)

    print("[*] Calculating Global Multi-Otsu (3-class) Threshold...")
    pooled_pixels = np.concatenate(all_pixels_list)
    del all_pixels_list
    
    thresholds = threshold_multiotsu(pooled_pixels, classes=3)
    high_threshold = thresholds[-1]
    print(f"    - Global High Threshold: {high_threshold:.4f}")

    print("[*] Quantifying individual image-level cell death rates...")
    per_image_results = []

    for img_data in all_image_data:
        intensities = img_data["nuclear_intensities"]
        total_nucleus_pixels = img_data["total_nucleus_pixels"]
        total_intensity = np.sum(intensities)

        if total_nucleus_pixels == 0:
            continue

        celldeath_pixels_mask = intensities >= high_threshold
        celldeath_pixels_count = np.sum(celldeath_pixels_mask)
        celldeath_intensity_sum = np.sum(intensities[celldeath_pixels_mask])

        # Core logic applied at the single-image level
        cell_death_rate = celldeath_pixels_count / total_nucleus_pixels
        intensity_rate = celldeath_intensity_sum / total_intensity if total_intensity > 0 else 0.0
        intensity_per_pixel = celldeath_intensity_sum / total_nucleus_pixels

        per_image_results.append({
            "filename": img_data["filename"],
            "folder": img_data["folder"],
            "class": img_data["class"],
            "total_nucleus_pixels": total_nucleus_pixels,
            "total_intensity": total_intensity,
            "global_threshold": high_threshold,
            "cell_death_rate": cell_death_rate,
            "intensity_rate": intensity_rate,
            "intensity_per_pixel": intensity_per_pixel
        })

    print(f"[*] Exporting individual data points to {args.output_csv}...")
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=per_image_results[0].keys())
        writer.writeheader()
        writer.writerows(per_image_results)
        
    print("[*] Analysis completed successfully.")


if __name__ == "__main__":
    main()