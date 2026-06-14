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
    # "SNCA_AST18": {
    #     "rows": [6, 7],
    #     "input_dirs": [
    #         r"C:\Users\admin\Desktop\professor_data\New_data\003001__2022-05-16T12_26_57-Measurement_1_C19_SNCA\Images",
    #         r"C:\Users\admin\Desktop\professor_data\New_data\003001__2022-05-16T15_49_01-Measurement 2_C19_SNCA\Images",
    #         r"C:\Users\admin\Desktop\professor_data\New_data\003002__2022-05-16T13_20_55-Measurement 1_C19_SNCA\Images",
    #         r"C:\Users\admin\Desktop\professor_data\New_data\003003__2022-05-16T14_09_08-Measurement 1_C19_SNCA\Images",
    #     ]
    # },


########################################################
    # "SNCA_AST18_isogenic": {
    #     "rows": [4, 5],
    #     "input_dirs": [
    #         r"D:\professor_data\New_data\003001__2022-05-16T12_26_57-Measurement_1_C19_SNCA\Images",
    #         r"D:\professor_data\New_data\003001__2022-05-16T15_49_01-Measurement 2_C19_SNCA\Images",
    #         r"D:\professor_data\New_data\003002__2022-05-16T13_20_55-Measurement 1_C19_SNCA\Images",
    #         r"D:\professor_data\New_data\003003__2022-05-16T14_09_08-Measurement 1_C19_SNCA\Images",
    #     ]
    # },


    # "SNCA-G51D": {
    #     "rows": [4, 5],
    #     "input_dirs": [
    #         r"D:\160822\005001_2022-08-16T10_54_01-Measurement 1\Images",
    #         r"D:\160822\005002_2022-08-16T11_43_12-Measurement 1\Images",
    #         r"D:\160822\005003_2022-08-16T12_31_25-Measurement 1\Images",
    #     ]
    # },

    # "SNCA-G51D_isogenic": {
    #     "rows": [6, 7],
    #     "input_dirs": [
    #         r"D:\160822\005001_2022-08-16T10_54_01-Measurement 1\Images",
    #         r"D:\160822\005002_2022-08-16T11_43_12-Measurement 1\Images",
    #         r"D:\160822\005003_2022-08-16T12_31_25-Measurement 1\Images",
    #     ]
    # },
#############################################################

    # "GBA": {
    #     "rows": [5, 6, 7], 
    #     "input_dirs": [
    #         r"C:\Users\admin\Desktop\professor_data\New_data\003004__2022-05-16T14_57_25-Measurement 1_C19_GBA\Images",
    #         r"C:\Users\admin\Desktop\professor_data\New_data\003005__2022-05-16T16_25_46-Measurement 1_C19_GBA\Images",
    #         r"C:\Users\admin\Desktop\professor_data\New_data\003006__2022-05-16T17_13_39-Measurement 1_C19_GBA\Images",
    #     ]
    # },




#######################################  "D:\003008__2022-05-16T18_49_33-Measurement 1_GBA_346"
    # "GBA_346": {
    #     "rows": [5, 6, 7], 
    #     "input_dirs": [
    #         r"D:\003007__2022-05-16T18_01_40-Measurement 1_GBA_346\Images",
    #         r"D:\003008__2022-05-16T18_49_33-Measurement 1_GBA_346\Images",
    #         r"D:\003009__2022-05-16T19_37_25-Measurement 1_GBA_346\Images",
    #     ]
    # },

    # "GBA_WIMP4": {
    #     "rows": [5, 6, 7], 
    #     "input_dirs": [
    #         r"D:\003014__2022-05-17T18_48_31-Measurement 1\Images",
    #         r"D:\003016__2022-05-17T20_24_27-Measurement 1\Images",
    #     ]
    # },
######################################

    
 

    # "Diverse_GBA": {
    #     "rows": [4], 
    #     "input_dirs": [
    #         ##r"C:\Users\admin\Desktop\professor_data\005004__2022-08-16T13_36_04-Measurement 1_diverse\Images",  플레이트 이상
    #         r"C:\Users\admin\Desktop\professor_data\005005__2022-08-16T14_22_56-Measurement 1_diverse\Images",
    #         ##r"C:\Users\admin\Desktop\professor_data\005006__2022-08-16T15_10_03-Measurement 1_diverse\Images",  플레이트 이상
    #         r"C:\Users\admin\Desktop\professor_data\005007__2022-08-16T16_06_39-Measurement 1_diverse\Images",
    #     ]
    # },

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


    # "Diverse_C18": {
    #     "rows": [3], 
    #     "input_dirs": [
    #         ##r"C:\Users\admin\Desktop\professor_data\005004__2022-08-16T13_36_04-Measurement 1_diverse\Images",  플레이트 이상
    #         r"C:\Users\admin\Desktop\professor_data\005005__2022-08-16T14_22_56-Measurement 1_diverse\Images",
    #         ##r"C:\Users\admin\Desktop\professor_data\005006__2022-08-16T15_10_03-Measurement 1_diverse\Images",  플레이트 이상
    #         r"C:\Users\admin\Desktop\professor_data\005007__2022-08-16T16_06_39-Measurement 1_diverse\Images",
    #     ]
    # },

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


   "alpha_syn_1day": {
        "rows": [2, 3, 4, 5, 6, 7], 
        "cols": [8],
        "input_dirs": [
            r"D:\090822\001001__2022-08-09T16_31_17-Measurement 1\Images",
           

        ]
    },

   "alpha_syn_3day": {
        "rows": [2, 3, 4, 5, 6, 7], 
        "cols": [3],
        "input_dirs": [
            r"D:\090822\001001__2022-08-09T16_31_17-Measurement 1\Images",
            

        ]
    },

   "alpha_syn_7day": {
        "rows": [2, 3, 4, 5, 6, 7], 
        "cols": [3],
        "input_dirs": [
            r"D:\090822\001002__2022-08-09T17_35_08-Measurement 1\Images",
           


        ]
    },


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
        return (self.group_name, self.folder_id, self.row, self.col, self.fraction, self.channel)

# --- 3. 기본 사용자 설정 ---
BASE_OUTPUT_DIR = r"D:\From_C_drive\MIP"
BASE_CYTOX_DIR = r"D:\From_C_drive\MIP_cytox"

# 개별 실험 설정에 "cols"가 없을 때 사용할 기본 컬럼 범위
DEFAULT_COLS_RANGE = range(2, 12) 
Z_STACK_COUNT = 5

# --- 4. 로직 함수 ---
def extract_folder_id(path: str) -> str:
    parent_folder = os.path.basename(os.path.dirname(path))
    match = re.match(r'(\d+)', parent_folder)
    return match.group(1) if match else "Unknown"

def parse_files(group_name: str, config: dict) -> List[ImageMetadata]:
    """특정 실험 그룹의 설정을 받아 파일 수집 (Row와 Col 모두 필터링)"""
    file_pattern = re.compile(r'r(\d+)c(\d+)f(\d+)p(\d+)-ch(\d+)')
    deduplicated_files = {}
    
    # 💡 그룹 설정에 'cols'가 있으면 그것을 쓰고, 없으면 기본 범위(2~11)를 사용합니다.
    target_rows = config['rows']
    target_cols = config.get('cols', DEFAULT_COLS_RANGE)
    directories = config['input_dirs']

    print(f"[{group_name}] 파일 스캔 중... (Rows: {list(target_rows)}, Cols: {list(target_cols)})")
    
    for folder in directories:
        if not os.path.exists(folder):
            print(f"경고: 경로 없음 - {folder}")
            continue
            
        f_id = extract_folder_id(folder)
        filenames = os.listdir(folder)
        
        for filename in tqdm(filenames, desc=f"Scanning {f_id}", leave=False):
            match = file_pattern.search(filename)
            if match:
                r, c, f, p, ch = map(int, match.groups())
                
                # 💡 Row 조건과 변경된 Column 조건을 모두 만족하는지 검사합니다.
                if r in target_rows and c in target_cols:
                    file_key = (f_id, r, c, f, p, ch)
                    deduplicated_files[file_key] = ImageMetadata(
                        group_name=group_name,
                        folder_id=f_id, row=r, col=c, fraction=f, plane=p, channel=ch,
                        full_path=os.path.join(folder, filename)
                    )
    return list(deduplicated_files.values())

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

        # 3. MIP 처리
        final_results = defaultdict(dict)
        process_loop = tqdm(list(grouped_data.items()), desc="Calculating MIP")
        
        for key, stack in process_loop:
            group, f_id, r, c, f, ch = key
            
            sorted_stack = sorted(stack, key=lambda x: x.plane)[:Z_STACK_COUNT]
            if len(sorted_stack) < Z_STACK_COUNT: continue
            
            try:
                img_data = np.array([tifffile.imread(m.full_path) for m in sorted_stack])
                mip_img = np.max(img_data, axis=0).astype(np.uint16)
                final_results[(f_id, r, c, f)][ch] = mip_img
            except Exception as e:
                print(f"\nError processing {key}: {e}")

        # 4. 저장 및 요약
        save_count = 0
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

    if summary_data:
        df = pd.DataFrame(summary_data)
        # df.to_csv(os.path.join(BASE_OUTPUT_DIR, "processing_summary.csv"), index=False)

if __name__ == "__main__":
    run_pipeline()