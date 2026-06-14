import os
import argparse
import numpy as np
import tifffile
from tqdm import tqdm
import sys

def get_args():
    parser = argparse.ArgumentParser(
        description="Generate patches from high-resolution microscopy images while preserving directory structure."
    )
    # 경로 설정
    parser.add_argument("--input_dir", type=str, default=r"D:\From_C_drive\MIP", 
                        help="Root directory containing the filtered (good) images.")
    parser.add_argument("--output_dir", type=str, default=r"D:\From_C_drive\cropped_image", 
                        help="Root directory where cropped patches will be saved.")
    
    # 패치 설정 (8x8 그리드를 위해 128 설정 유지)
    parser.add_argument("--patch_size", type=int, default=128, help="Size of the cropped patch (default: 128).")
    parser.add_argument("--overlap", type=int, default=0, help="Overlap between patches in pixels (default: 0 for non-overlapping).")
    
    # 처리할 특정 폴더 지정 (기존 exclude_dirs 대신 target_dirs로 변경)
    parser.add_argument("--target_dirs", nargs='+', default=["GBA_346", "GBA_WIMP4", "SNCA-G51D", "SNCA-G51D_isogenic", "SNCAx3_isogenic", "alpha_syn_1day", "alpha_syn_7day"], 
                        help="List of specific subfolders inside input_dir to process.")
    
    return parser.parse_args()

def crop_and_save(img, filename, save_dir, patch_size=128, overlap=0):
    """
    Crops the image into patches and saves them with coordinate information.
    """
    h, w = img.shape[:2]
    stride = patch_size - overlap
    
    patch_count = 0
    
    # Grid Calculation
    # 1024 / 128 = 8 steps (0, 128, 256, 384, 512, 640, 768, 896) -> 총 8개
    y_steps = range(0, h - patch_size + 1, stride)
    x_steps = range(0, w - patch_size + 1, stride)
    
    for y in y_steps:
        for x in x_steps:
            # Crop
            patch = img[y : y + patch_size, x : x + patch_size]
            
            # Sanity Check for Shape
            if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                continue
            
            # Naming Convention: OriginalName_x{Coord}_y{Coord}.tif
            save_name = f"{os.path.splitext(filename)[0]}_x{x}_y{y}.tif"
            save_path = os.path.join(save_dir, save_name)
            
            tifffile.imwrite(save_path, patch)
            patch_count += 1
            
    return patch_count

def run_patching(args):
    print(f"🚀 Starting Patch Generation...")
    print(f"   - Input:  {args.input_dir}")
    print(f"   - Output: {args.output_dir}")
    print(f"   - Size:   {args.patch_size}x{args.patch_size} (8x8 Grid if 1024x1024)")
    print(f"   - Targets: {', '.join(args.target_dirs)}")
    
    if not os.path.exists(args.input_dir):
        print(f"❌ Input directory not found: {args.input_dir}")
        return

    total_images_processed = 0
    total_patches_generated = 0
    
    # [수정] 지정한 3개 폴더만 바로 순회하도록 경로 목록 생성
    target_paths = [os.path.join(args.input_dir, d) for d in args.target_dirs]
    
    for target_path in target_paths:
        if not os.path.exists(target_path):
            print(f"⚠️ Warning: Target directory not found, skipping: {target_path}")
            continue
            
        # 지정된 하위 폴더 내부만 os.walk 수행
        for root, dirs, files in os.walk(target_path):
            tif_files = [f for f in files if f.lower().endswith(('.tif', '.tiff'))]
            
            if not tif_files:
                continue
                
            # 출력 경로 생성 (MIP 기준의 상대 경로 유지)
            rel_path = os.path.relpath(root, args.input_dir)
            save_root = os.path.join(args.output_dir, rel_path)
            
            os.makedirs(save_root, exist_ok=True)
            
            print(f"📂 Processing folder: {rel_path} ({len(tif_files)} images)")
            
            for filename in tqdm(tif_files, desc="   Cropping", leave=False):
                img_path = os.path.join(root, filename)
                
                try:
                    img = tifffile.imread(img_path)
                    
                    if img.shape[0] < args.patch_size or img.shape[1] < args.patch_size:
                        continue
                    
                    n_patches = crop_and_save(
                        img, 
                        filename, 
                        save_root, 
                        patch_size=args.patch_size, 
                        overlap=args.overlap
                    )
                    
                    total_patches_generated += n_patches
                    total_images_processed += 1
                    
                except Exception as e:
                    print(f"⚠️ Failed to process {filename}: {e}")

    print("\n" + "="*50)
    print("✅ Patch Generation Completed!")
    print("="*50)
    print(f"   - Original Images: {total_images_processed}")
    print(f"   - Total Patches:   {total_patches_generated}")
    print(f"   - Saved to:        {args.output_dir}")
    print("="*50)

if __name__ == "__main__":
    args = get_args()
    run_patching(args)
