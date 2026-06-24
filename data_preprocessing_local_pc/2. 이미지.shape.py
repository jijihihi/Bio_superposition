import argparse
import os
import sys

import numpy as np
import tifffile


def check_image_data_integrity(root_dir):
    """
    지정된 디렉토리 내의 TIFF 이미지들의 속성(Shape, Dtype, Value Range)을 검사합니다.
    데이터셋의 일관성을 검증하고 손상된 파일이나 잘못된 포맷을 식별합니다.

    Args:
        root_dir (str): 검사할 데이터셋의 최상위 경로
    """
    print(f"🔍 Starting Data Integrity Check for: '{root_dir}'\n")

    dataset_summary = []
    total_files_checked = 0

    # os.walk로 하위 폴더 재귀 탐색
    for current_root, dirs, files in os.walk(root_dir):
        # .tif 및 .tiff 확장자 필터링 (대소문자 무관)
        tif_files = sorted([f for f in files if f.lower().endswith((".tif", ".tiff"))])

        if not tif_files:
            continue

        print(f"📁 Directory: {os.path.basename(current_root)}")
        print(f"   - Path: {current_root}")
        print(f"   - Count: {len(tif_files)} files")

        all_is_rgb = True
        sample_count = min(5, len(tif_files))  # 최대 5개 샘플링

        for i, filename in enumerate(tif_files[:sample_count]):
            full_path = os.path.join(current_root, filename)

            try:
                img = tifffile.imread(full_path)

                # RGB 형식 확인 (3차원이고 마지막 차원이 3이어야 함)
                is_rgb = img.ndim == 3 and img.shape[-1] == 3
                if not is_rgb:
                    all_is_rgb = False

                # 상세 정보 출력
                print(f"     [{i+1}] {filename}")
                print(
                    f"         Shape: {img.shape} | Dtype: {img.dtype} | Range: {img.min()} ~ {img.max()}"
                )

                # 첫 번째 파일의 정보를 해당 폴더의 대표 정보로 수집
                if i == 0:
                    dataset_summary.append(
                        {
                            "Folder": os.path.basename(current_root),
                            "Shape": img.shape,
                            "Dtype": img.dtype,
                            "Is_RGB": is_rgb,
                        }
                    )

            except Exception as e:
                print(f"     ❌ Error reading file: {filename} ({e})")

        if all_is_rgb:
            print("     ✅ Confirmed: Sampled images are in RGB format.")
        else:
            print("     ⚠️ Warning: Non-RGB images detected in samples.")

        print("-" * 60)
        total_files_checked += len(tif_files)

    # --- 최종 요약 리포트 ---
    print("\n📊 [Dataset Summary Report]")
    if dataset_summary:
        # 데이터 일관성 검사 (모든 폴더의 이미지 Shape가 동일한지)
        shapes = [str(item["Shape"]) for item in dataset_summary]
        dtypes = [str(item["Dtype"]) for item in dataset_summary]

        shape_consistent = len(set(shapes)) == 1
        dtype_consistent = len(set(dtypes)) == 1

        print(f"  - Total Folders Scanned: {len(dataset_summary)}")
        print(f"  - Total Files Found: {total_files_checked}")
        print(
            f"  - Shape Consistency: {'✅ Passed' if shape_consistent else f'⚠️ Failed (Found {len(set(shapes))} variations)'}"
        )
        print(
            f"  - Dtype Consistency: {'✅ Passed' if dtype_consistent else f'⚠️ Failed (Found {len(set(dtypes))} variations)'}"
        )

        if shape_consistent:
            print(f"  - Representative Shape: {shapes[0]}")
            print(f"  - Representative Dtype: {dtypes[0]}")
    else:
        print("  ❌ No TIFF images found in the specified directory.")


if __name__ == "__main__":
    # 터미널 실행을 위한 설정 (기본값은 코드 내에서 수정 가능)
    # 사용법: python script.py --dir "C:/Path/To/Data"
    parser = argparse.ArgumentParser(
        description="Check integrity and consistency of TIFF image datasets."
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=r"C:\Users\admin\Desktop\MIP",
        help="Path to the dataset root directory",
    )

    args = parser.parse_args()

    if os.path.exists(args.dir):
        check_image_data_integrity(args.dir)
    else:
        print(f"❌ Error: The directory '{args.dir}' does not exist.")
