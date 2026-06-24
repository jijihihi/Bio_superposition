import argparse
import os
import sys

import numpy as np
import tifffile
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(
        description="Generate patches from high-resolution microscopy images while preserving directory structure."
    )
    # 경로 설정
    parser.add_argument(
        "--input_dir",
        type=str,
        default=r"D:\From_C_drive\MIP",
        help="Root directory containing the filtered (good) images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=r"D:\From_C_drive\cropped_image",
        help="Root directory where cropped patches will be saved.",
    )

    # 패치 설정
    parser.add_argument(
        "--patch_size",
        type=int,
        default=128,
        help="Size of the cropped patch (default: 128).",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Overlap between patches in pixels (default: 0 for non-overlapping).",
    )

    # 제외할 폴더 설정 (QC 결과 폴더 등)
    parser.add_argument(
        "--exclude_dirs",
        nargs="+",
        default=["Rejected_Images", "결과저장", "QC_Analysis"],
        help="List of folder names to exclude from processing.",
    )

    return parser.parse_args()


def crop_and_save(img, filename, save_dir, patch_size=128, overlap=0):
    """
    Crops the image into patches and saves them with coordinate information.
    """
    h, w = img.shape[:2]
    stride = patch_size - overlap

    # 3채널(RGB)이거나 1채널(Grayscale) 모두 대응
    # 좌표 순회 (y: 높이, x: 너비)
    patch_count = 0

    # Grid Calculation
    # 1024 / 128 = 8 steps (0, 128, 256, ... 896)
    y_steps = range(0, h - patch_size + 1, stride)
    x_steps = range(0, w - patch_size + 1, stride)

    for y in y_steps:
        for x in x_steps:
            # Crop
            patch = img[y : y + patch_size, x : x + patch_size]

            # Sanity Check for Shape (끝부분 자투리 방지)
            if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                continue

            # Naming Convention: OriginalName_x{Coord}_y{Coord}.tif
            # 논문에 쓰기 좋게 직관적인 좌표 명시
            save_name = f"{os.path.splitext(filename)[0]}_x{x}_y{y}.tif"
            save_path = os.path.join(save_dir, save_name)

            tifffile.imwrite(save_path, patch)
            patch_count += 1

    return patch_count


def run_patching(args):
    print(f"🚀 Starting Patch Generation...")
    print(f"   - Input:  {args.input_dir}")
    print(f"   - Output: {args.output_dir}")
    print(f"   - Size:   {args.patch_size}x{args.patch_size}")

    if not os.path.exists(args.input_dir):
        print(f"❌ Input directory not found: {args.input_dir}")
        return

    total_images_processed = 0
    total_patches_generated = 0

    # Walk through input directory
    for root, dirs, files in os.walk(args.input_dir):
        # 제외할 폴더 건너뛰기 (수정: dirs 리스트를 직접 제어해야 os.walk가 안 들어감)
        dirs[:] = [d for d in dirs if d not in args.exclude_dirs]

        tif_files = [f for f in files if f.lower().endswith((".tif", ".tiff"))]

        if not tif_files:
            continue

        # 출력 경로 생성 (구조 복사)
        # 예: MIP/GroupA -> cropped_image/GroupA
        rel_path = os.path.relpath(root, args.input_dir)
        save_root = os.path.join(args.output_dir, rel_path)

        os.makedirs(save_root, exist_ok=True)

        print(f"📂 Processing folder: {rel_path} ({len(tif_files)} images)")

        for filename in tqdm(tif_files, desc="   Cropping", leave=False):
            img_path = os.path.join(root, filename)

            try:
                img = tifffile.imread(img_path)

                # 이미지 크기 검증 (1024x1024가 맞는지)
                if img.shape[0] < args.patch_size or img.shape[1] < args.patch_size:
                    # 너무 작은 이미지는 스킵
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

            except Exception as e:
                print(f"⚠️ Failed to process {filename}: {e}")

    print("\n" + "=" * 50)
    print("✅ Patch Generation Completed!")
    print("=" * 50)
    print(f"   - Original Images: {total_images_processed}")
    print(f"   - Total Patches:   {total_patches_generated}")
    print(f"   - Saved to:        {args.output_dir}")
    print("=" * 50)


if __name__ == "__main__":
    args = get_args()
    run_patching(args)
