#!/usr/bin/env python3
"""
StarDist Nuclei Segmentation Pipeline

This script applies StarDist 2D nuclei segmentation to microscopy images,
filtering out border and small nuclei, and saves segmentation masks with 
optional visualization outputs.

Usage Examples:
    # Process specific folders with default paths
    python "큰 이미지 stardist해서 masking 된 이미지 같이 저장.py" --folders Control_C18 SNCA GBA
    
    # Process with custom input/output paths
    python "큰 이미지 stardist해서 masking 된 이미지 같이 저장.py" \
        --input_dir "C:/path/to/input" \
        --output_dir "C:/path/to/output" \
        --folders Control_C18 SNCA
    
    # Adjust StarDist parameters
    python "큰 이미지 stardist해서 masking 된 이미지 같이 저장.py" \
        --folders LRRK2 \
        --prob_thresh 0.5 \
        --nms_thresh 0.4 \
        --min_area 50
"""

import argparse
import csv
import numpy as np
import tifffile
from pathlib import Path

# Visualization imports
import matplotlib.pyplot as plt
from skimage.exposure import rescale_intensity
from skimage.measure import regionprops

# StarDist imports
from stardist.models import StarDist2D
from csbdeep.utils import normalize


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="StarDist nuclei segmentation for microscopy images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Path arguments
    parser.add_argument(
        "--input_dir",
        type=str,
        default=r"C:\Users\admin\Desktop\MIP",
        help="Base input directory containing image folders"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=r"C:\Users\admin\Desktop\MIP_segmentation",
        help="Base output directory for segmentation results"
    )
    parser.add_argument(
        "--folders",
        type=str,
        nargs="+",
        #default=["Control_C4", "Control_GBA_C19", "Control_SNCA_C19", "SNCA", "GBA", "LRRK2", "PINK1"],  # 원하는 폴더 이름으로 수정
        default=["GBA_346", "GBA_WIMP4", "SNCAx3_isogenic", "SNCA-G51D", "SNCA-G51D_isogenic", "SNCAx3"],
        help="List of folder names to process (e.g., Control_C18 SNCA GBA)"
    )
    parser.add_argument(
        "--csv_output",
        type=str,
        default=r"C:\Users\admin\Desktop\MIP_segmentation",
        help="Path for CSV results file. If not specified, saves to output_dir/nuclei_count_results.csv"
    )
    
    # StarDist parameters
    parser.add_argument(
        "--prob_thresh",
        type=float,
        default=0.479071,
        help="StarDist probability threshold for nuclei detection"
    )
    parser.add_argument(
        "--nms_thresh",
        type=float,
        default=0.3,
        help="StarDist NMS (non-maximum suppression) threshold"
    )
    
    # Channel and filtering parameters
    parser.add_argument(
        "--nuc_channel",
        type=int,
        default=2,
        help="Index of nucleus channel in multi-channel TIF images"
    )
    parser.add_argument(
        "--min_area",
        type=int,
        default=40,
        help="Minimum nucleus area in pixels (smaller nuclei are filtered)"
    )
    
    # Output options
    parser.add_argument(
        "--save_visualization",
        action="store_true",
        default=True,
        help="Save visualization PNG images"
    )
    parser.add_argument(
        "--no_visualization",
        dest="save_visualization",
        action="store_false",
        help="Disable saving visualization PNG images"
    )
    
    return parser.parse_args()


def segment_nuclei(args):
    """
    Main nuclei segmentation function with Resume capability.
    
    Args:
        args: Parsed command-line arguments
    """
    # Load StarDist model
    print(f"--- 🤖 StarDist 모델 로드 중 (prob={args.prob_thresh}, nms={args.nms_thresh}) ---")
    model = StarDist2D.from_pretrained('2D_versatile_fluo')
    print("✅ StarDist 모델 로드 완료.")
    
    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # [수정] args.csv_output이 없거나 폴더 경로인 경우 자동으로 파일명을 붙여 Path 객체로 만듭니다.
    if args.csv_output is None:
        csv_output_path = output_dir / "nuclei_count_results.csv"
    else:
        temp_path = Path(args.csv_output)
        if temp_path.is_dir() or temp_path.suffix == "":
            csv_output_path = temp_path / "nuclei_count_results.csv"
        else:
            csv_output_path = temp_path
            
    # [수정] 기존 CSV 파일이 있다면 기존 데이터를 로드하여 결과 유실을 방지합니다.
    results_data = []
    existing_records = set() # (폴더명, 파일명) 중복 체크용
    
    if csv_output_path.exists() and csv_output_path.is_file():
        try:
            with open(csv_output_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                header = next(reader, None) # 헤더 건너뛰기
                if header:
                    for row in reader:
                        if len(row) >= 6:
                            results_data.append(row)
                            # 어떤 파일들이 이미 CSV에 기록되었는지 추적
                            existing_records.add((row[0], row[1]))
            print(f"ℹ️ 기존 CSV에서 {len(results_data)}개의 분석 기록을 불러왔습니다.")
        except Exception as e:
            print(f"⚠️ 기존 CSV 파일을 읽는 중 오류 발생 (새로 작성함): {e}")

    # Process each specified folder
    for folder_name in args.folders:
        folder_path = input_dir / folder_name
        
        if not folder_path.exists():
            print(f"‼️ 경고: 폴더를 찾을 수 없습니다: {folder_path}. 건너뜁니다.")
            continue
        
        if not folder_path.is_dir():
            print(f"‼️ 경고: '{folder_name}'은(는) 폴더가 아닙니다. 건너뜁니다.")
            continue
            
        print(f"\n--- 🔬 폴더 처리 중: {folder_name} ---")
        
        # Create output directories for this folder
        mask_output_dir = output_dir / folder_name / "mask"
        vis_output_dir = output_dir / folder_name / "visualization"
        mask_output_dir.mkdir(parents=True, exist_ok=True)
        if args.save_visualization:
            vis_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all TIF files in folder
        tif_files = list(folder_path.glob("*.tif")) + list(folder_path.glob("*.tiff"))
        
        if not tif_files:
            print(f"  - ⚠️ TIF 파일이 없습니다: {folder_name}")
            continue
        
        print(f"  - 발견된 이미지 수: {len(tif_files)}")
        
        for tif_path in tif_files:
            base_filename = tif_path.stem
            expected_mask_path = Path(mask_output_dir) / f"{base_filename}_mask.tif"
            
            # 🔄 이미 처리된 파일인지 체크 (마스크 파일 존재 여부 & CSV 기록 여부)
            if expected_mask_path.exists():
                print(f"  - ⏭️ {tif_path.name}: 이미 처리된 파일입니다. (Skip)")
                continue
                
            try:
                # Load image
                data_cube = tifffile.imread(tif_path)
                
                # Handle different image dimensions
                if data_cube.ndim == 2:
                    nuc_ch_data = data_cube
                elif data_cube.ndim == 3:
                    if data_cube.shape[2] <= 4:  # Likely (H, W, C)
                        if data_cube.shape[2] <= args.nuc_channel:
                            print(f"  - ⚠️ '{tif_path.name}' 파일에서 핵 채널({args.nuc_channel})을 찾을 수 없어 건너뜁니다.")
                            continue
                        nuc_ch_data = data_cube[:, :, args.nuc_channel]
                    else:  # Likely (C, H, W)
                        if data_cube.shape[0] <= args.nuc_channel:
                            print(f"  - ⚠️ '{tif_path.name}' 파일에서 핵 채널({args.nuc_channel})을 찾을 수 없어 건너뜁니다.")
                            continue
                        nuc_ch_data = data_cube[args.nuc_channel, :, :]
                else:
                    print(f"  - ⚠️ '{tif_path.name}' 지원하지 않는 이미지 차원: {data_cube.shape}")
                    continue
                
                # Normalize for StarDist
                normalized_nuc = normalize(nuc_ch_data, 1, 99.8, axis=(0, 1))
                
                # Run StarDist prediction
                labels, _ = model.predict_instances(
                    normalized_nuc,
                    prob_thresh=args.prob_thresh,
                    nms_thresh=args.nms_thresh
                )
                
                # Filter nuclei
                total_nuclei = int(np.max(labels))
                valid_labels_list = []
                border_nuclei_count = 0
                small_nuclei_count = 0
                
                if total_nuclei > 0:
                    height, width = labels.shape
                    props = regionprops(labels)
                    
                    for prop in props:
                        min_r, min_c, max_r, max_c = prop.bbox
                        is_border = (min_r == 0 or min_c == 0 or 
                                    max_r == height or max_c == width)
                        is_small = prop.area <= args.min_area
                        
                        if is_border:
                            border_nuclei_count += 1
                        elif is_small:
                            small_nuclei_count += 1
                        else:
                            valid_labels_list.append(prop.label)
                
                valid_nuclei_count = len(valid_labels_list)
                print(f"  - {tif_path.name}: 총 {total_nuclei}개 -> 유효 핵: {valid_nuclei_count}개 "
                      f"(경계:{border_nuclei_count}, 소형:{small_nuclei_count} 제외)")
                
                # Record results
                results_data.append([
                    folder_name,
                    tif_path.name,
                    total_nuclei,
                    border_nuclei_count,
                    small_nuclei_count,
                    valid_nuclei_count
                ])
                
                # Create filtered segmentation mask
                segmentation_mask = np.zeros_like(labels, dtype=np.uint16)
                for label_id in valid_labels_list:
                    segmentation_mask[labels == label_id] = label_id
                
                # Save mask TIF
                try:
                    tifffile.imwrite(expected_mask_path, segmentation_mask)
                except Exception as tif_error:
                    print(f"  - ‼️ 마스크 TIF 저장 중 오류: {tif_error}")
                
                # Save visualization PNG
                if args.save_visualization:
                    try:
                        nuc_ch_8bit = rescale_intensity(
                            nuc_ch_data, 
                            in_range='image', 
                            out_range=(0, 255)
                        ).astype(np.uint8)
                        rgb_image = np.stack([nuc_ch_8bit] * 3, axis=-1)
                        binary_mask = segmentation_mask > 0
                        rgb_image[binary_mask] = [0, 0, 255]  # Blue overlay
                        
                        output_vis_path = vis_output_dir / f"{base_filename}_visualization.png"
                        plt.imsave(output_vis_path, rgb_image)
                    except Exception as png_error:
                        print(f"  - ‼️ 시각화 PNG 저장 중 오류: {png_error}")
                        
            except Exception as e:
                print(f"  - ‼️ '{tif_path.name}' 파일 분석 중 오류: {e}")
    
    # Save CSV results
    print("\n--- 💾 CSV 파일 저장 중... ---")
    try:
        with open(csv_output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'FolderName',
                'ImageFileName',
                'TotalNuclei',
                'BorderNuclei',
                'SmallNuclei',
                'ValidNuclei_Count'
            ])
            if results_data:
                writer.writerows(results_data)
        print(f"✅ 분석 결과가 '{csv_output_path}' 파일에 성공적으로 저장되었습니다.")
    except Exception as e:
        print(f"‼️ CSV 파일 저장 중 오류 발생: {e}")
    
    print(f"\n--- ✅ 처리 완료 ---")
    print(f"   처리된 폴더: {len(args.folders)}개")
    print(f"   총 누적 이미지: {len(results_data)}장")
    print(f"   출력 경로: {output_dir}")

if __name__ == '__main__':
    args = parse_args()
    segment_nuclei(args)