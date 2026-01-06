import os
import argparse
import numpy as np
import tifffile
from tqdm import tqdm
import sys

def get_args():
    parser = argparse.ArgumentParser(
        description="Verify the integrity and count consistency of cropped image datasets."
    )
    # 경로 설정 (기본값은 사용자 환경에 맞춤, 실행 시 수정 가능)
    parser.add_argument("--source_dir", type=str, default=r"C:\Users\admin\Desktop\MIP", 
                        help="Path to the original high-res image directory (Source).")
    parser.add_argument("--target_dir", type=str, default=r"C:\Users\admin\Desktop\cropped_image", 
                        help="Path to the cropped patch directory (Target).")
    
    # 검증 기준 설정
    parser.add_argument("--expected_ratio", type=int, default=64, 
                        help="Expected ratio of patches per original image (Default: 64 for 1024->128).")
    parser.add_argument("--check_integrity", action='store_true', default=True,
                        help="Enable deep inspection of shape and bit-depth (Recommended).")
    
    # 제외할 폴더
    parser.add_argument("--exclude_dirs", nargs='+', 
                        default=["Rejected_Images", "결과저장", "QC_Analysis", "csv"], 
                        help="Folder names to ignore in the source directory.")
    
    return parser.parse_args()

def count_files_in_dir(root_dir, exclude_list):
    """
    Recursively counts .tif files in directories, preserving relative paths.
    Returns: dict { 'Relative/Path': count }
    """
    counts = {}
    print(f"🔍 Scanning directory: {root_dir}...")
    
    for root, dirs, files in os.walk(root_dir):
        # 제외 폴더 건너뛰기 (Pruning)
        dirs[:] = [d for d in dirs if d not in exclude_list]
        
        tif_files = [f for f in files if f.lower().endswith(('.tif', '.tiff'))]
        
        if tif_files:
            # 루트 경로를 제외한 상대 경로를 키(Key)로 사용 (예: SNCA/Batch1)
            rel_path = os.path.relpath(root, root_dir)
            counts[rel_path] = len(tif_files)
            
    return counts

def check_integrity_worker(file_list):
    """
    Checks shape and dtype for a list of files.
    Returns: (passed_count, failed_files_list)
    """
    passed = 0
    failures = []
    
    for fpath in file_list:
        try:
            img = tifffile.imread(fpath)
            
            # 1. Bit-depth Check (uint16)
            if img.dtype != np.uint16:
                failures.append((fpath, f"Wrong Dtype: {img.dtype}"))
                continue
                
            # 2. Channel Check (Must be 3 channels)
            # Shape can be (128, 128, 3) or (3, 128, 128) depending on save method
            # We assume HWC or CHW, checking if '3' exists in shape dimensions.
            if img.ndim != 3 or (img.shape[2] != 3 and img.shape[0] != 3):
                failures.append((fpath, f"Wrong Shape: {img.shape}"))
                continue
            
            # 3. Size Check (Should be 128x128 in spatial dims)
            # This is a basic check; if strict 128x128 needed, add logic here.
            
            passed += 1
            
        except Exception as e:
            failures.append((fpath, f"Read Error: {e}"))
            
    return passed, failures

def run_verification(args):
    print("🚀 Starting Dataset Verification Protocol\n")
    
    # 1. Source Directory Analysis (Originals)
    source_counts = count_files_in_dir(args.source_dir, args.exclude_dirs)
    if not source_counts:
        print("❌ No images found in source directory.")
        return

    # 2. Target Directory Analysis (Patches)
    # Note: We don't exclude 'Rejected_Images' in target because target shouldn't have them anyway.
    # But strictly, we just scan everything in target.
    target_counts = count_files_in_dir(args.target_dir, [])
    
    # 3. Verification Logic
    print("\n📊 Comparing Source vs Target...")
    print(f"{'Folder Group':<40} | {'Source':<8} | {'Target':<8} | {'Exp. (x64)':<10} | {'Status'}")
    print("-" * 90)
    
    all_passed_counts = True
    total_patches_to_check = 0
    target_files_list = []
    
    # Iterate through Source groups to match with Target
    for group, s_count in sorted(source_counts.items()):
        t_count = target_counts.get(group, 0)
        expected = s_count * args.expected_ratio
        
        status = "✅ Match"
        if t_count != expected:
            status = f"❌ Mismatch (Diff: {t_count - expected})"
            all_passed_counts = False
        
        print(f"{group[:40]:<40} | {s_count:<8} | {t_count:<8} | {expected:<10} | {status}")
        
        # Collect files for integrity check
        if t_count > 0:
            full_group_path = os.path.join(args.target_dir, group)
            # Get all files in this group for later check
            for f in os.listdir(full_group_path):
                if f.lower().endswith('.tif'):
                    target_files_list.append(os.path.join(full_group_path, f))

    # Check for Orphan Folders in Target (Target has folders that Source doesn't)
    orphan_groups = set(target_counts.keys()) - set(source_counts.items())
    # Note: keys match logic slightly adjusted above
    orphan_groups = [g for g in target_counts if g not in source_counts]
    if orphan_groups:
        print("\n⚠️ Warning: Found folders in Target not present in Source:")
        for g in orphan_groups:
            print(f"   - {g} ({target_counts[g]} files)")
            
    # 4. Deep Integrity Check (Bit-depth & Channels)
    if args.check_integrity and target_files_list:
        print(f"\n🔬 Performing Deep Integrity Check on {len(target_files_list)} patches...")
        print("   (Verifying: uint16 dtype, 3 channels)")
        
        passed_count, failures = 0, []
        
        # Progress bar for large number of files
        with tqdm(total=len(target_files_list), unit="img") as pbar:
            # Simple batch processing approach or direct loop
            # To keep code simple, we loop directly but update pbar
            for fpath in target_files_list:
                try:
                    img = tifffile.imread(fpath)
                    
                    is_valid = True
                    # Check 16-bit
                    if img.dtype != np.uint16:
                        failures.append(f"{os.path.basename(fpath)}: Type {img.dtype} != uint16")
                        is_valid = False
                    
                    # Check 3 Channels (HWC or CHW)
                    # Assuming 128x128 patches: shape should contain 3 and 128
                    if is_valid:
                        if img.ndim != 3:
                            failures.append(f"{os.path.basename(fpath)}: Not 3D {img.shape}")
                            is_valid = False
                        elif 3 not in img.shape:
                            failures.append(f"{os.path.basename(fpath)}: No RGB channel found {img.shape}")
                            is_valid = False
                    
                    if is_valid:
                        passed_count += 1
                        
                except Exception as e:
                    failures.append(f"{os.path.basename(fpath)}: Read Error {e}")
                
                pbar.update(1)

        print(f"\n✅ Integrity Check Passed: {passed_count}/{len(target_files_list)}")
        
        if failures:
            print(f"❌ Integrity Failures Found: {len(failures)}")
            print("   First 5 failures:")
            for fail in failures[:5]:
                print(f"   - {fail}")
            all_passed_counts = False # Flag as failed overall
    
    # Final Summary
    print("\n" + "="*50)
    print("🏁 FINAL VERIFICATION RESULT")
    print("="*50)
    if all_passed_counts and not orphan_groups:
        print("🎉 SUCCESS: All data is consistent and valid.")
        print("   - Count Ratio (x64): Consistent")
        print("   - Data Integrity:    Verified (16-bit, RGB)")
    else:
        print("⚠️ WARNING: Issues detected. Review logs above.")
    print("="*50)

if __name__ == "__main__":
    args = get_args()
    if os.path.exists(args.source_dir) and os.path.exists(args.target_dir):
        run_verification(args)
    else:
        print("❌ Error: Check input directories.")