import os
import re
import numpy as np
import tifffile
import cv2
import shlex

def parse_windows_paths(raw_input):
    """
    Windows '경로로 복사' 형식("Path1" "Path2")을 파싱하여 리스트로 변환합니다.
    """
    try:
        # shlex는 따옴표로 묶인 문자열을 리스트로 잘 분리해줍니다.
        # 윈도우 경로의 백슬래시(\) 처리를 위해 replace로 이스케이프하거나 shlex 설정을 씁니다.
        # 가장 간단하고 강력한 방법: 정규표현식으로 따옴표 안의 내용만 추출
        paths = re.findall(r'"(.*?)"', raw_input)
        
        # 만약 따옴표가 없는 경우(하나만 복사했거나 다른 방식), 공백으로 분리 시도
        if not paths:
            paths = [p.strip() for p in raw_input.split() if p.strip()]
            
        return paths
    except Exception as e:
        print(f"❌ 경로 파싱 중 오류 발생: {e}")
        return []

def calculate_laplacian_variance(image_path):
    """이미지의 Laplacian Variance (선명도) 계산"""
    try:
        # 이미지 읽기
        img = tifffile.imread(image_path)
        
        # 채널 처리 (RGB -> Max Projection)
        if img.ndim == 3:
            # 밝기나 색상 왜곡 없이 엣지를 가장 잘 살리기 위해 Max Projection 사용
            gray = np.max(img, axis=2)
        else:
            gray = img
            
        # 정밀도를 위해 float64로 변환
        gray = gray.astype(np.float64)
        
        # 라플라시안 필터 적용 및 분산 계산
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        return laplacian_var, img.shape, img.dtype
        
    except Exception as e:
        return None, None, str(e)

def main():
    print("\n" + "="*60)
    print(" 📸 이미지 선명도(Laplacian Variance) 계산기")
    print("="*60)
    print("팁: 파일들을 선택하고 [Ctrl+Shift+C]로 복사한 뒤 아래에 붙여넣으세요.")
    print("-" * 60)
    
    # 사용자 입력 받기
    try:
        raw_input = input(">> 경로 붙여넣기 (Ctrl+V): ")
    except KeyboardInterrupt:
        return

    if not raw_input.strip():
        print("❌ 입력된 내용이 없습니다.")
        return

    # 경로 파싱
    file_paths = parse_windows_paths(raw_input)
    print(f"\n🔍 총 {len(file_paths)}개의 파일이 감지되었습니다.\n")

    print(f"{'Filename':<40} | {'Laplacian Var':<15} | {'Shape':<15} | {'Status'}")
    print("-" * 90)

    # 계산 및 출력
    for fpath in file_paths:
        # 경로의 따옴표나 공백 정리
        fpath = os.path.normpath(fpath.strip())
        filename = os.path.basename(fpath)
        
        if not os.path.exists(fpath):
            print(f"{filename[:37]+'...':<40} | {'-':<15} | {'-':<15} | ❌ File Not Found")
            continue
            
        score, shape, dtype = calculate_laplacian_variance(fpath)
        
        if score is not None:
            # 16-bit 이미지는 숫자가 크므로 지수 표기법이나 콤마 사용
            score_str = f"{score:,.0f}" 
            print(f"{filename[:37]+'...':<40} | {score_str:<15} | {str(shape):<15} | ✅ OK")
        else:
            print(f"{filename[:37]+'...':<40} | {'Error':<15} | {'-':<15} | ⚠️ {dtype}")

    print("-" * 90)
    input("\n엔터 키를 누르면 종료합니다...")

if __name__ == "__main__":
    main()