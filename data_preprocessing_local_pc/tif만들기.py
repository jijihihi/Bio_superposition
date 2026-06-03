# import tarfile
# from pathlib import Path

# def compact_specific_folders():
#     # 1. 최상위 작업 디렉토리 설정
#     base_dir = Path(r"C:\Users\admin\Desktop\cropped_image")
    
#     # 2. 압축 대상인 5개 폴더 이름 리스트
#     target_folders = [
#         "GBA_346", 
#         "GBA_WIMP4", 
#         "SNCA_isogenic", 
#         "SNCA-G51D", 
#         "SNCA-G51D_isogenic"
#     ]
    
#     for folder_name in target_folders:
#         folder_path = base_dir / folder_name
        
#         # 폴더가 실제로 존재하는지 확인
#         if not folder_path.exists():
#             print(f"❌ 폴더를 찾을 수 없어 건너뜁니다: {folder_path}")
#             continue
            
#         # 생성할 tar 파일 경로 (예: C:\Users\admin\Desktop\cropped_image\GBA_346.tar)
#         output_tar_path = base_dir / f"{folder_name}.tar"
        
#         print(f"\n📦 [{folder_name}] 압축 시작 중...")
        
#         # tar 파일 생성 시작
#         with tarfile.open(output_tar_path, "w") as tar:
#             # 해당 폴더 내부의 모든 파일 및 하위 폴더 검색
#             for file_path in folder_path.rglob("*"):
#                 # .tif 또는 .tiff 파일만 골라서 추가
#                 if file_path.is_file() and file_path.suffix.lower() in ['.tif', '.tiff']:
#                     # tar 안에서의 상대 경로 설정 (해당 폴더를 최상위로 인식하도록)
#                     arcname = file_path.relative_to(base_dir)
#                     tar.add(file_path, arcname=arcname)
                    
#         print(f"✅ 압축 완료: {output_tar_path}")

# if __name__ == "__main__":
#     compact_specific_folders()

import tarfile
from pathlib import Path

def compact_to_gz():
    base_dir = Path(r"C:\Users\admin\Desktop\cropped_image")
    target_folders = ["GBA_346", "GBA_WIMP4", "SNCA_isogenic", "SNCA-G51D", "SNCA-G51D_isogenic"]
    
    for folder_name in target_folders:
        folder_path = base_dir / folder_name
        if not folder_path.exists(): continue
            
        # 💡 .tar.gz 확장자로 변경하여 파일 생성
        output_tar_path = base_dir / f"{folder_name}.tar.gz"
        print(f"📦 [{folder_name}] 고효율 용량 압축 시작 중...")
        
        # 💡 "w:gz" 모드를 써야 용량이 획기적으로 줄어듭니다.
        with tarfile.open(output_tar_path, "w:gz") as tar:
            for file_path in folder_path.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in ['.tif', '.tiff']:
                    arcname = file_path.relative_to(base_dir)
                    tar.add(file_path, arcname=arcname)
                    
        print(f"✅ 압축 완벽 완료: {output_tar_path}")

if __name__ == "__main__":
    compact_to_gz()