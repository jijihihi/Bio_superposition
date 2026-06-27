#!/usr/bin/env python3
"""
Microscopy Image Patch Generation (Tiling) Script

This script partitions high-resolution microscopy images into uniform, 
overlapping or non-overlapping patches while maintaining the original 
subfolder directory hierarchy.
"""

import argparse
import os
from pathlib import Path
import numpy as np
import tifffile
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(
        description="Generate patches from high-resolution microscopy images while preserving directory structure.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Root directory containing the filtered high-resolution input images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Root directory where cropped patches will be consolidated and saved.",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=128,
        help="Dimension (pixels) of the square cropped patch.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Overlap slice between contiguous patches in pixels.",
    )
    parser.add_argument(
        "--exclude_dirs",
        nargs="+",
        default=["Rejected_Images", "QC_Analysis"],
        help="Subdirectory names to completely bypass during iteration.",
    )
    return parser.parse_args()


def crop_and_save(img: np.ndarray, filename: str, save_dir: Path, patch_size: int, overlap: int) -> int:
    """Slices an image array into standardized sub-patches and exports them with coordinate stems."""
    h, w = img.shape[:2]
    stride = patch_size - overlap
    patch_count = 0

    y_steps = range(0, h - patch_size + 1, stride)
    x_steps = range(0, w - patch_size + 1, stride)

    for y in y_steps:
        for x in x_steps:
            patch = img[y : y + patch_size, x : x + patch_size]

            # Enforce strict boundary matching to discard incomplete edges
            if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                continue

            save_name = f"{Path(filename).stem}_x{x}_y{y}.tif"
            save_path = save_dir / save_name

            tifffile.imwrite(save_path, patch)
            patch_count += 1

    return patch_count


def main():
    args = get_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"[-] Input source path does not exist: {input_dir}")

    total_images_processed = 0
    total_patches_generated = 0

    print("[*] Launching structured patch generation pipeline...")
    
    for root, dirs, files in os.walk(input_dir):
        # In-place filtering of directories to control os.walk recursion recursion depth
        dirs[:] = [d for d in dirs if d not in args.exclude_dirs]
        tif_files = [f for f in files if f.lower().endswith((".tif", ".tiff"))]

        if not tif_files:
            continue

        rel_path = Path(root).relative_to(input_dir)
        save_root = output_dir / rel_path
        save_root.mkdir(parents=True, exist_ok=True)

        print(f"[*] Processing partition: {rel_path if str(rel_path) != '.' else 'Root'} ({len(tif_files)} source files)")

        for filename in tqdm(tif_files, desc="Cropping Patches", leave=False):
            img_path = Path(root) / filename

            img = tifffile.imread(img_path)

            # Skip uninformative arrays failing minimal dimension standards
            if img.shape[0] < args.patch_size or img.shape[1] < args.patch_size:
                continue

            n_patches = crop_and_save(
                img,
                filename,
                save_root,
                patch_size=args.patch_size,
                overlap=args.overlap,
            )
            total_patches_generated += n_patches
            total_images_processed += 1

    # Pipeline Performance Summary Execution Analytics
    print("\n" + "=" * 60)
    print("✅ PATCH GENERATION PIPELINE METRICS SUBMISSION")
    print("=" * 60)
    print(f"  - Successfully Processed Source Images : {total_images_processed}")
    print(f"  - Total Standardized Patches Generated : {total_patches_generated}")
    print(f"  - Export Destination Target Path       : {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()