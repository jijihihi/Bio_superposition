import os


def count_images_per_folder(base_dir):
    # 1. 카운트할 이미지 확장자 정의 (TIF 위주 + 기타)
    valid_exts = (".tif", ".tiff")

    # 경로 확인
    if not os.path.exists(base_dir):
        print(f"❌ 경로가 존재하지 않습니다: {base_dir}")
        return

    print(f"📂 분석 경로: {base_dir}")
    print("=" * 50)
    print(f"{'Folder Name':<25} | {'Count':>10}")
    print("-" * 50)

    total_sum = 0

    # 2. 베이스 경로 내의 항목들을 확인
    # 폴더 이름순으로 정렬해서 보기 좋게 출력
    for folder_name in sorted(os.listdir(base_dir)):
        folder_path = os.path.join(base_dir, folder_name)

        # 폴더인 경우에만 진입
        if os.path.isdir(folder_path):
            count = 0

            # 3. 해당 폴더 내부의 모든 파일 검사 (하위 폴더 포함 recursive하게)
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if file.lower().endswith(valid_exts):
                        count += 1

            # 결과 출력
            print(f"{folder_name:<25} | {count:>10} 장")
            total_sum += count

    print("-" * 50)
    print(f"{'TOTAL':<25} | {total_sum:>10} 장")
    print("=" * 50)


# ==========================================
# 실행 부분
# ==========================================
if __name__ == "__main__":
    target_path = r"C:\Users\admin\Desktop\cropped_image"
    count_images_per_folder(target_path)
