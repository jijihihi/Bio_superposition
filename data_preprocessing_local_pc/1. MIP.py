import os
import re
import numpy as np
import tifffile
import pandas as pd 
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Tuple
from collections import defaultdict


# --- 1. 실험 설계 (Configuration) ---
# 여기에 모든 실험 조건을 미리 적어둡니다. (논문 Methods 파트 작성할 때 매우 유용함)


# INPUT_DIRS = [
#     r"C:\Users\admin\Desktop\professor_data\New_data\003001__2022-05-16T12_26_57-Measurement_1_C19_SNCA\Images",
#     r"C:\Users\admin\Desktop\professor_data\New_data\003001__2022-05-16T15_49_01-Measurement 2_C19_SNCA\Images",
#     r"C:\Users\admin\Desktop\professor_data\New_data\003002__2022-05-16T13_20_55-Measurement 1_C19_SNCA\Images",
#     r"C:\Users\admin\Desktop\professor_data\New_data\003003__2022-05-16T14_09_08-Measurement 1_C19_SNCA\Images",
#     r"C:\Users\admin\Desktop\professor_data\New_data\003004__2022-05-16T14_57_25-Measurement 1_C19_GBA\Images",
#     r"C:\Users\admin\Desktop\professor_data\New_data\003005__2022-05-16T16_25_46-Measurement 1_C19_GBA\Images",
#     r"C:\Users\admin\Desktop\professor_data\New_data\003006__2022-05-16T17_13_39-Measurement 1_C19_GBA\Images",
#     r"C:\Users\admin\Desktop\professor_data\New_data\004001__2022-06-27T11_22_54-Measurement 1_C4_LRRK2\Images",
#     r"C:\Users\admin\Desktop\professor_data\New_data\004002__2022-06-27T12_12_52-Measurement 1_C4_LRRK2\Images",
#     r"C:\Users\admin\Desktop\professor_data\New_data\004003__2022-06-27T13_02_42-Measurement 1_C4_LRRK2\Images",
#     r"C:\Users\admin\Desktop\professor_data\New_data\004004__2022-06-27T14_07_07-Measurement 1_PINK1\Images",
#     r"C:\Users\admin\Desktop\professor_data\New_data\004005__2022-06-27T14_56_55-Measurement 1_PINK1\Images",
#     r"C:\Users\admin\Desktop\professor_data\New_data\004006__2022-06-27T15_46_53-Measurement 1_PINK1\Images",
#     r"C:\Users\admin\Desktop\professor_data\New_data\004007__2022-06-27T17_09_02-Measurement 1_PINK1\Images",
#     r"C:\Users\admin\Desktop\professor_data\New_data\Minee_160822_plate1__2022-08-16T10_54_01-Measurement 1_C18\Images",
#     r"C:\Users\admin\Desktop\professor_data\New_data\Minee_160822_plate2__2022-08-16T11_43_12-Measurement 1_C18\Images",
#     r"C:\Users\admin\Desktop\professor_data\New_data\Minee_160822_plate3__2022-08-16T12_31_25-Measurement 1_C18\Images",
# ]

EXPERIMENTS = {
    # "SNCA": {
    #     "rows": [6, 7],
    #     "input_dirs": [
    #         r"C:\Users\admin\Desktop\professor_data\New_data\003001__2022-05-16T12_26_57-Measurement_1_C19_SNCA\Images",
    #         r"C:\Users\admin\Desktop\professor_data\New_data\003001__2022-05-16T15_49_01-Measurement 2_C19_SNCA\Images",
    #         r"C:\Users\admin\Desktop\professor_data\New_data\003002__2022-05-16T13_20_55-Measurement 1_C19_SNCA\Images",
    #         r"C:\Users\admin\Desktop\professor_data\New_data\003003__2022-05-16T14_09_08-Measurement 1_C19_SNCA\Images",
    #     ]
    # },

    # "GBA": {
    #     "rows": [5, 6, 7], 
    #     "input_dirs": [
    #         r"C:\Users\admin\Desktop\professor_data\New_data\003004__2022-05-16T14_57_25-Measurement 1_C19_GBA\Images",
    #         r"C:\Users\admin\Desktop\professor_data\New_data\003005__2022-05-16T16_25_46-Measurement 1_C19_GBA\Images",
    #         r"C:\Users\admin\Desktop\professor_data\New_data\003006__2022-05-16T17_13_39-Measurement 1_C19_GBA\Images",
    #     ]
    # },

    "Diverse_GBA": {
        "rows": [4], 
        "input_dirs": [
            ##r"C:\Users\admin\Desktop\professor_data\005004__2022-08-16T13_36_04-Measurement 1_diverse\Images",  플레이트 이상
            r"C:\Users\admin\Desktop\professor_data\005005__2022-08-16T14_22_56-Measurement 1_diverse\Images",
            ##r"C:\Users\admin\Desktop\professor_data\005006__2022-08-16T15_10_03-Measurement 1_diverse\Images",  플레이트 이상
            r"C:\Users\admin\Desktop\professor_data\005007__2022-08-16T16_06_39-Measurement 1_diverse\Images",
        ]
    },

    # "PINK1": {
    #     "rows": [2, 3, 4],
    #     "input_dirs": [
    #         r"C:\Users\admin\Desktop\professor_data\New_data\004004__2022-06-27T14_07_07-Measurement 1_PINK1\Images",
    #         r"C:\Users\admin\Desktop\professor_data\New_data\004005__2022-06-27T14_56_55-Measurement 1_PINK1\Images",
    #         r"C:\Users\admin\Desktop\professor_data\New_data\004006__2022-06-27T15_46_53-Measurement 1_PINK1\Images",
    #         r"C:\Users\admin\Desktop\professor_data\New_data\004007__2022-06-27T17_09_02-Measurement 1_PINK1\Images",
    #     ]
    # },

    # "LRRK2": {
    #     "rows": [5, 6, 7], 
    #     "input_dirs": [
    #         r"D:\professor_data\New_data\004001__2022-06-27T11_22_54-Measurement 1_C4_LRRK2\Images",
    #         r"D:\professor_data\New_data\004002__2022-06-27T12_12_52-Measurement 1_C4_LRRK2\Images",
    #         r"D:\professor_data\New_data\004003__2022-06-27T13_02_42-Measurement 1_C4_LRRK2\Images", ### 안에 이미지 없음. 따라서 150 * 3(row 개수) * 2(파일 개수) = 900개
 

    #     ]
    # },

    # "Control_C4": {
    #     "rows": [2, 3, 4], 
    #     "input_dirs": [
    #         r"C:\Users\admin\Desktop\professor_data\New_data\004001__2022-06-27T11_22_54-Measurement 1_C4_LRRK2\Images",
    #         r"C:\Users\admin\Desktop\professor_data\New_data\004002__2022-06-27T12_12_52-Measurement 1_C4_LRRK2\Images",
    #         r"C:\Users\admin\Desktop\professor_data\New_data\004003__2022-06-27T13_02_42-Measurement 1_C4_LRRK2\Images", ### 안에 이미지 없음. 따라서 150 * 3(row 개수) * 2(파일 개수) = 900개
    #     ]
    # },

    # "Control_C18": {
    #     "rows": [2, 3], 
    #     "input_dirs": [
    #         r"C:\Users\admin\Desktop\professor_data\New_data\005001__2022-08-16T10_54_01-Measurement 1_C18\Images",
    #         r"C:\Users\admin\Desktop\professor_data\New_data\005002__2022-08-16T11_43_12-Measurement 1_C18\Images",
    #         r"C:\Users\admin\Desktop\professor_data\New_data\005003__2022-08-16T12_31_25-Measurement 1_C18\Images",
    #     ]
    # },


    "Diverse_C18": {
        "rows": [3], 
        "input_dirs": [
            ##r"C:\Users\admin\Desktop\professor_data\005004__2022-08-16T13_36_04-Measurement 1_diverse\Images",  플레이트 이상
            r"C:\Users\admin\Desktop\professor_data\005005__2022-08-16T14_22_56-Measurement 1_diverse\Images",
            ##r"C:\Users\admin\Desktop\professor_data\005006__2022-08-16T15_10_03-Measurement 1_diverse\Images",  플레이트 이상
            r"C:\Users\admin\Desktop\professor_data\005007__2022-08-16T16_06_39-Measurement 1_diverse\Images",
        ]
    },

    # "Control_SNCA_C19": {
    #     "rows": [2, 3], 
    #     "input_dirs": [
    #         r"C:\Users\admin\Desktop\professor_data\New_data\003001__2022-05-16T12_26_57-Measurement_1_C19_SNCA\Images",
    #         r"C:\Users\admin\Desktop\professor_data\New_data\003001__2022-05-16T15_49_01-Measurement 2_C19_SNCA\Images",
    #         r"C:\Users\admin\Desktop\professor_data\New_data\003002__2022-05-16T13_20_55-Measurement 1_C19_SNCA\Images",
    #         r"C:\Users\admin\Desktop\professor_data\New_data\003003__2022-05-16T14_09_08-Measurement 1_C19_SNCA\Images",
    #     ]
    # },

    # "Control_GBA_C19": {
    #     "rows": [2, 3, 4], 
    #     "input_dirs": [
    #         r"C:\Users\admin\Desktop\professor_data\New_data\003004__2022-05-16T14_57_25-Measurement 1_C19_GBA\Images",
    #         r"C:\Users\admin\Desktop\professor_data\New_data\003005__2022-05-16T16_25_46-Measurement 1_C19_GBA\Images",
    #         r"C:\Users\admin\Desktop\professor_data\New_data\003006__2022-05-16T17_13_39-Measurement 1_C19_GBA\Images",
    #     ]
    # },



}

# --- 1. 데이터 구조 정의 ---

@dataclass(frozen=True)
class ImageMetadata:
    group_name: str 
    folder_id: str  
    row: int
    col: int
    fraction: int
    plane: int
    channel: int
    full_path: str

    @property
    def group_key(self) -> Tuple[str, str, int, int, int, int]:
        """그룹화를 위한 고유 키 (그룹명, ID 포함)"""
        # <--- 3. 여기서 self.group_name을 포함해 총 6개를 반환해야 합니다.
        return (self.group_name, self.folder_id, self.row, self.col, self.fraction, self.channel)

# --- 2. 사용자 설정 ---



# 결과 저장 경로 설정
BASE_OUTPUT_DIR = r"C:\Users\admin\Desktop\MIP"           # 이 아래에 그룹별 폴더가 생성됨
BASE_CYTOX_DIR = r"C:\Users\admin\Desktop\MIP_cytox"


COLS_RANGE = range(2, 12)
Z_STACK_COUNT = 5

# --- 3. 로직 함수 ---

def extract_folder_id(path: str) -> str:
    parent_folder = os.path.basename(os.path.dirname(path))
    match = re.match(r'(\d+)', parent_folder)
    return match.group(1) if match else "Unknown"
# --- 수정된 함수 1: 파일 스캔에도 진행 바 추가 ---
def parse_files(group_name: str, config: dict) -> List[ImageMetadata]:
    """특정 실험 그룹의 설정을 받아 파일을 수집"""
    file_pattern = re.compile(r'r(\d+)c(\d+)f(\d+)p(\d+)-ch(\d+)')
    deduplicated_files = {}
    target_rows = config['rows']
    directories = config['input_dirs']

    print(f"[{group_name}] 파일 스캔 중... (Target Rows: {target_rows})")
    
    # 폴더별로 진행 상황을 보여줍니다
    for folder in directories:
        if not os.path.exists(folder):
            print(f"경고: 경로 없음 - {folder}")
            continue
            
        f_id = extract_folder_id(folder)
        filenames = os.listdir(folder)
        
        # [수정됨] 파일 목록을 순회할 때 tqdm 추가 (leave=False는 완료 후 바를 지워 화면을 깔끔하게 함)
        for filename in tqdm(filenames, desc=f"Scanning {f_id}", leave=False):
            match = file_pattern.search(filename)
            if match:
                r, c, f, p, ch = map(int, match.groups())
                if r in target_rows and c in COLS_RANGE:
                    file_key = (f_id, r, c, f, p, ch)
                    deduplicated_files[file_key] = ImageMetadata(
                        group_name=group_name,
                        folder_id=f_id, row=r, col=c, fraction=f, plane=p, channel=ch,
                        full_path=os.path.join(folder, filename)
                    )
    return list(deduplicated_files.values())

# --- 수정된 함수 2: MIP 계산 단계에도 진행 바 추가 ---
def run_pipeline():
    summary_data = []

    for group_name, config in EXPERIMENTS.items():
        print(f"\n🚀 Processing Group: {group_name}")
        
        save_dir_rgb = os.path.join(BASE_OUTPUT_DIR, group_name)
        save_dir_cytox = os.path.join(BASE_CYTOX_DIR, group_name)
        os.makedirs(save_dir_rgb, exist_ok=True)
        os.makedirs(save_dir_cytox, exist_ok=True)

        # 1. 파일 수집
        images = parse_files(group_name, config)
        if not images:
            print(f"🚨 {group_name} 그룹에 해당하는 파일이 없습니다. 설정을 확인하세요.")
            continue

        # 2. 그룹화
        grouped_data = defaultdict(list)
        for img in images:
            grouped_data[img.group_key].append(img)

        # 3. MIP 처리 (가장 오래 걸리는 부분)
        final_results = defaultdict(dict)
        
        # [수정됨] 계산 루프에 tqdm 추가
        # list(grouped_data.items())로 변환해야 tqdm이 전체 길이를 알고 예상 시간을 계산해줍니다.
        process_loop = tqdm(list(grouped_data.items()), desc="Calculating MIP")
        
        for key, stack in process_loop:
            group, f_id, r, c, f, ch = key
            
            sorted_stack = sorted(stack, key=lambda x: x.plane)[:Z_STACK_COUNT]
            if len(sorted_stack) < Z_STACK_COUNT: continue
            
            # 이미지 로딩 및 Max Projection
            try:
                img_data = np.array([tifffile.imread(m.full_path) for m in sorted_stack])
                mip_img = np.max(img_data, axis=0).astype(np.uint16)
                final_results[(f_id, r, c, f)][ch] = mip_img
            except Exception as e:
                print(f"\nError processing {key}: {e}")

        # 4. 저장 및 요약
        save_count = 0
        # 기존에 있던 저장 단계 tqdm (유지)
        for (f_id, r, c, f), channels in tqdm(final_results.items(), desc=f"Saving {group_name}"):
            filename_prefix = f"{f_id}_r{r:02d}c{c:02d}f{f:02d}"
            
            summary_data.append({
                "Group": group_name,
                "Folder_ID": f_id,
                "Row": r, "Col": c, "Fraction": f,
                "Has_RGB": all(ch in channels for ch in [1, 3, 4]),
                "Has_Cytox": 2 in channels
            })

            if all(ch in channels for ch in [1, 3, 4]):
                rgb = np.stack([channels[3], channels[4], channels[1]], axis=-1)
                tifffile.imwrite(os.path.join(save_dir_rgb, f"{filename_prefix}_Composite_RGB.tif"), rgb)
                save_count += 1

            if 2 in channels:
                tifffile.imwrite(os.path.join(save_dir_cytox, f"{filename_prefix}_MIP_ch2_Cytox.tif"), channels[2])
        
        print(f"✅ {group_name} 완료! 총 {save_count}개의 이미지가 생성되었습니다.")

    df = pd.DataFrame(summary_data)
    # df.to_csv(os.path.join(BASE_OUTPUT_DIR, "processing_summary.csv"), index=False)
    # print(f"\n📄 전체 처리 내역이 'processing_summary.csv'에 저장되었습니다.")

if __name__ == "__main__":
    run_pipeline()