import numpy as np
import tifffile
import os

def analyze_channel_stats(image_path):
    print(f"run analysis for: {os.path.basename(image_path)}")
    
    try:
        # 1. 이미지 로드 (16-bit 유지)
        img = tifffile.imread(image_path)
        
        # 차원 확인 및 보정 (H, W, C) 형태로 맞춤
        if img.ndim == 2: # (H, W) -> (H, W, 1)
            img = img[..., np.newaxis]
        elif img.shape[0] < img.shape[2]: # (C, H, W) -> (H, W, C) 로 가정
            img = np.transpose(img, (1, 2, 0))
            
        h, w, c = img.shape
        print(f"  - Shape: {img.shape} (H, W, C)")
        
        # 2. [0, 1]로 스케일링 (Instance Norm과 동일한 조건)
        img_float = img.astype(np.float32) / 65535.0
        
        print("-" * 60)
        print(f"{'Channel':^10} | {'Std (표준편차)':^15} | {'Mean (평균)':^15} | {'Max (최대)':^10}")
        print("-" * 60)
        
        # 3. 채널별 통계 계산
        for i in range(c):
            channel_data = img_float[..., i]
            
            std_val = np.std(channel_data)
            mean_val = np.mean(channel_data)
            max_val = np.max(channel_data)
            
            # 채널 이름 추정 (순서가 R, G, B 혹은 Hoechst, TMRM, LysoTracker 등)
            ch_name = f"Ch {i}"
            
            print(f"{ch_name:^10} | {std_val:^15.6f} | {mean_val:^15.6f} | {max_val:^10.4f}")
            
        print("-" * 60)
        print("💡 [분석 팁]")
        print("1. 'Std' 값이 Threshold(예: 0.05)보다 낮으면 노이즈로 간주됩니다.")
        print("2. 배경만 있는 이미지의 Std 값을 확인하고, 그 값보다 살짝 높게 잡으세요.")
        print("-" * 60 + "\n")

    except Exception as e:
        print(f"Error: {e}")

# =========================================================
# ▼ 아래에 테스트하고 싶은 이미지 경로를 입력하세요 ▼
# =========================================================

# 예시 1: 세포가 있는 정상 이미지
path_cell = r"C:\Users\admin\Desktop\cropped_image\Control_C4\004001_r02c02f04_Composite_RGB_x512_y640.tif"

# 예시 2: 거의 배경만 있는 이미지 (노이즈만 있는 경우)
path_bg = r"C:\Users\admin\Desktop\cropped_image\PINK1\004004_r02c02f14_Composite_RGB_x384_y128.tif"

# 실행 (경로가 실제 존재할 때만 실행)
if os.path.exists(path_cell):
    analyze_channel_stats(path_cell)
else:
    print(f"경로를 확인하세요: {path_cell}")

if os.path.exists(path_bg):
    analyze_channel_stats(path_bg)