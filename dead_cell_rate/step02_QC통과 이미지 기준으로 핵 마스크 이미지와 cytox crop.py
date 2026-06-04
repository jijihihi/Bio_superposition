#!/usr/bin/env python3
"""
QC 기반 StarDist 마스크 및 CytoxGreen 이미지 Crop 스크립트

QC 통과 이미지를 기준으로 StarDist 마스크와 CytoxGreen 이미지를 
동일 좌표로 crop하여 QC/QC_reject 폴더로 분류합니다.

Usage:
    python "2. QC통과 이미지 기준으로 마스크와 cytox crop.py" --folders Control_C4 SNCA
"""

import os
import re
import argparse
import numpy as np
import tifffile
from pathlib import Path
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(
        description="Crop StarDist masks and CytoxGreen images based on QC-passed images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Path arguments
    parser.add_argument(
        "--qc_passed_dir", type=str,
        default=r"C:\Users\admin\Desktop\cropped_image",
        help="Directory containing QC-passed cropped images"
    )
    parser.add_argument(
        "--mask_dir", type=str,
        default=r"C:\Users\admin\Desktop\MIP_segmentation",
        help="Directory containing StarDist mask images"
    )
    parser.add_argument(
        "--cytox_dir", type=str,
        default=r"C:\Users\admin\Desktop\MIP_cytox",
        help="Directory containing CytoxGreen images"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=r"C:\Users\admin\Desktop\세포사멸율_data_new_class",
        help="Output directory for cropped results"
    )
    
    # Folder selection
    parser.add_argument(
        "--folders", type=str, nargs="+",
        #default=["Control_C4", "Control_C18", "Control_GBA_C19", "Control_SNCA_C19", 
               #  "SNCA", "GBA", "LRRK2"],
        default=["GBA_346", "GBA_WIMP4", "SNCAx3_isogenic", "SNCA-G51D", "SNCA-G51D_isogenic", "SNCAx3"],
        help="List of folder names to process"
    )
    
    # Crop settings
    parser.add_argument(
        "--patch_size", type=int, default=128,
        help="Size of patches (must match original crop size)"
    )
    
    return parser.parse_args()


def parse_qc_filename(filename):
    """
    QC 통과 이미지 파일명에서 plate_position과 x, y 좌표 추출
    
    예: 004001_r02c02f01_Composite_RGB_x0_y0.tif
    -> plate_position: 004001_r02c02f01
    -> x: 0, y: 0
    """
    # x{num}_y{num} 패턴 추출
    match = re.search(r'_x(\d+)_y(\d+)', filename)
    if not match:
        return None, None, None
    
    x = int(match.group(1))
    y = int(match.group(2))
    
    # plate_position 추출 (x_y 이전 부분에서 _Composite_RGB 제거)
    base = filename[:match.start()]
    # 예: 004001_r02c02f01_Composite_RGB -> 004001_r02c02f01
    plate_position_match = re.match(r'^(\d+_[a-zA-Z]\d+c\d+f\d+)', base)
    if not plate_position_match:
        return None, None, None
    
    plate_position = plate_position_match.group(1)
    return plate_position, x, y


def find_mask_file(mask_dir, folder_name, plate_position):
    """
    StarDist 마스크 파일 찾기
    예: MIP_segmentation/Control_C4/mask/004001_r02c02f01_Composite_RGB_mask.tif
    """
    mask_folder = Path(mask_dir) / folder_name / "mask"
    if not mask_folder.exists():
        return None
    
    # 해당 plate_position으로 시작하는 마스크 파일 찾기
    pattern = f"{plate_position}_*_mask.tif"
    matches = list(mask_folder.glob(pattern))
    
    if matches:
        return matches[0]
    return None


def find_cytox_file(cytox_dir, folder_name, plate_position):
    """
    CytoxGreen 파일 찾기
    예: MIP_cytox/Control_C4/004001_r02c02f01_MIP_ch2_Cytox.tif
    """
    cytox_folder = Path(cytox_dir) / folder_name
    if not cytox_folder.exists():
        return None
    
    # 해당 plate_position으로 시작하는 Cytox 파일 찾기
    pattern = f"{plate_position}_*Cytox*.tif"
    matches = list(cytox_folder.glob(pattern))
    
    if matches:
        return matches[0]
    return None


def crop_image(img, x, y, patch_size):
    """이미지에서 지정된 좌표로 patch_size x patch_size 영역 crop"""
    return img[y:y+patch_size, x:x+patch_size]


def run_cropping(args):
    print("=" * 60)
    print("🚀 QC 기반 마스크/CytoxGreen Crop 시작")
    print("=" * 60)
    print(f"   QC 통과 이미지: {args.qc_passed_dir}")
    print(f"   StarDist 마스크: {args.mask_dir}")
    print(f"   CytoxGreen: {args.cytox_dir}")
    print(f"   출력 경로: {args.output_dir}")
    print(f"   처리 폴더: {args.folders}")
    print("=" * 60)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {
        "total_qc_images": 0,
        "matched_mask": 0,
        "matched_cytox": 0,
        "reject_mask": 0,
        "reject_cytox": 0,
    }
    
    # 각 폴더별 처리
    for folder_name in args.folders:
        qc_folder = Path(args.qc_passed_dir) / folder_name
        
        if not qc_folder.exists():
            print(f"⚠️ QC 폴더 없음: {folder_name}")
            continue
        
        print(f"\n📂 폴더 처리 중: {folder_name}")
        
        # 출력 폴더 생성
        qc_mask_out = output_dir / folder_name / "QC" / "stardist_mask"
        qc_cytox_out = output_dir / folder_name / "QC" / "cytoxgreen"
        reject_mask_out = output_dir / folder_name / "QC_reject" / "stardist_mask"
        reject_cytox_out = output_dir / folder_name / "QC_reject" / "cytoxgreen"
        
        qc_mask_out.mkdir(parents=True, exist_ok=True)
        qc_cytox_out.mkdir(parents=True, exist_ok=True)
        reject_mask_out.mkdir(parents=True, exist_ok=True)
        reject_cytox_out.mkdir(parents=True, exist_ok=True)
        
        # QC 통과 이미지 파일 목록
        qc_files = list(qc_folder.glob("*.tif")) + list(qc_folder.glob("*.tiff"))
        
        if not qc_files:
            print(f"   ⚠️ QC 통과 이미지 없음")
            continue
        
        print(f"   📊 QC 통과 이미지: {len(qc_files)}장")
        
        # plate_position별로 원본 이미지 캐싱 (같은 원본에서 여러 crop)
        mask_cache = {}
        cytox_cache = {}
        
        for qc_file in tqdm(qc_files, desc=f"   {folder_name}", leave=False):
            stats["total_qc_images"] += 1
            
            # 파일명 파싱
            plate_position, x, y = parse_qc_filename(qc_file.name)
            if plate_position is None:
                print(f"   ⚠️ 파일명 파싱 실패: {qc_file.name}")
                continue
            
            base_filename = qc_file.stem  # x_y 좌표 포함된 파일명
            
            # === StarDist 마스크 처리 ===
            if plate_position not in mask_cache:
                mask_file = find_mask_file(args.mask_dir, folder_name, plate_position)
                if mask_file:
                    try:
                        mask_cache[plate_position] = tifffile.imread(mask_file)
                    except Exception as e:
                        print(f"   ⚠️ 마스크 로드 실패: {mask_file.name} - {e}")
                        mask_cache[plate_position] = None
                else:
                    mask_cache[plate_position] = None
            
            mask_img = mask_cache.get(plate_position)
            if mask_img is not None:
                try:
                    cropped_mask = crop_image(mask_img, x, y, args.patch_size)
                    if cropped_mask.shape[0] == args.patch_size and cropped_mask.shape[1] == args.patch_size:
                        out_path = qc_mask_out / f"{base_filename}_mask.tif"
                        tifffile.imwrite(out_path, cropped_mask)
                        stats["matched_mask"] += 1
                    else:
                        stats["reject_mask"] += 1
                except Exception as e:
                    stats["reject_mask"] += 1
            else:
                stats["reject_mask"] += 1
            
            # === CytoxGreen 처리 ===
            if plate_position not in cytox_cache:
                cytox_file = find_cytox_file(args.cytox_dir, folder_name, plate_position)
                if cytox_file:
                    try:
                        cytox_cache[plate_position] = tifffile.imread(cytox_file)
                    except Exception as e:
                        print(f"   ⚠️ Cytox 로드 실패: {cytox_file.name} - {e}")
                        cytox_cache[plate_position] = None
                else:
                    cytox_cache[plate_position] = None
            
            cytox_img = cytox_cache.get(plate_position)
            if cytox_img is not None:
                try:
                    cropped_cytox = crop_image(cytox_img, x, y, args.patch_size)
                    if cropped_cytox.shape[0] == args.patch_size and cropped_cytox.shape[1] == args.patch_size:
                        out_path = qc_cytox_out / f"{base_filename}_cytox.tif"
                        tifffile.imwrite(out_path, cropped_cytox)
                        stats["matched_cytox"] += 1
                    else:
                        stats["reject_cytox"] += 1
                except Exception as e:
                    stats["reject_cytox"] += 1
            else:
                stats["reject_cytox"] += 1
        
        # 메모리 정리
        mask_cache.clear()
        cytox_cache.clear()
    
    # 최종 통계
    print("\n" + "=" * 60)
    print("📊 최종 결과")
    print("=" * 60)
    print(f"   총 QC 통과 이미지: {stats['total_qc_images']}장")
    print(f"   ✅ 마스크 매칭 성공: {stats['matched_mask']}장")
    print(f"   ✅ Cytox 매칭 성공: {stats['matched_cytox']}장")
    print(f"   ❌ 마스크 매칭 실패: {stats['reject_mask']}장")
    print(f"   ❌ Cytox 매칭 실패: {stats['reject_cytox']}장")
    print(f"   📁 출력 경로: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    args = get_args()
    run_cropping(args)
