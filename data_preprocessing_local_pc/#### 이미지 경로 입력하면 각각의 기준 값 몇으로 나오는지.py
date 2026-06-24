import os

import cv2
import numpy as np
import tifffile
from tqdm import tqdm

# ==========================================
# 1. 테스트할 이미지 경로 설정
# ==========================================
target_image_path = r"C:\Users\admin\Desktop\cropped_image\PINK1\004004_r02c02f06_Composite_RGB_x256_y128.tif"

# ==========================================
# 2. 설정값 (사용 중인 값과 동일해야 함)
# ==========================================
CONFIG = {"saturation_percent": 0.5, "bg_threshold": 3000.0, "min_std_threshold": 655.0}


def visualize_background_mask(file_path):
    if not os.path.exists(file_path):
        print("❌ 파일이 없습니다.")
        return

    try:
        img = tifffile.imread(file_path)
    except Exception as e:
        print(f"❌ 로딩 실패: {e}")
        return

    if img.ndim == 2:
        img = img[..., np.newaxis]

    h, w, c = img.shape
    print(f"🔍 Analyzing: {os.path.basename(file_path)}")

    # ---------------------------------------------------------
    # Step 3 로직 그대로 수행 (Linear Scaling)
    # ---------------------------------------------------------
    scaled_img = np.zeros_like(img, dtype=np.float32)
    target_max = 65535.0

    print("\n[Scaling Info]")
    for i in range(c):
        channel = img[..., i].astype(np.float32)
        raw_std = np.std(channel)

        if raw_std < CONFIG["min_std_threshold"]:
            print(f"  - Ch{i}: Std={raw_std:.1f} -> SKIP (Low Info)")
            scaled_channel = channel
        else:
            cutoff = np.percentile(channel, 100 - CONFIG["saturation_percent"])
            if cutoff <= 0:
                scale_factor = 0
            else:
                scale_factor = target_max / cutoff

            print(
                f"  - Ch{i}: Std={raw_std:.1f}, Cutoff={cutoff:.0f} -> Factor={scale_factor:.2f}x"
            )
            scaled_channel = channel * scale_factor

        scaled_img[..., i] = np.clip(scaled_channel, 0, target_max)

    # ---------------------------------------------------------
    # 배경 / 신호 구분
    # ---------------------------------------------------------
    max_proj = np.max(scaled_img, axis=2)

    # Signal Mask (True=세포, False=배경)
    is_signal = max_proj > CONFIG["bg_threshold"]

    signal_count = np.count_nonzero(is_signal)
    bg_count = (h * w) - signal_count
    bg_frac = bg_count / (h * w)

    print(f"\n📊 결과 분석")
    print(f"  - Signal Pixels: {signal_count}")
    print(f"  - BG Pixels: {bg_count}")
    print(f"  - BG Fraction: {bg_frac*100:.2f}%")

    # ---------------------------------------------------------
    # 🎨 시각화 이미지 생성 (배경을 흰색으로 칠하기)
    # ---------------------------------------------------------
    # 1. 보기 좋게 원본을 8비트로 변환 (밝기 보정 없이 단순 압축)
    #    (여기서는 '어디가 배경인지'만 보면 되므로 원본 밝기 유지)
    vis_img = (img.astype(np.float32) / 256.0).astype(np.uint8)

    # 만약 채널이 1개면 RGB로 변환
    if c == 1:
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)
    elif c == 2:  # 채널이 2개면 B 채널 0으로 채워서 3개 맞춤
        empty = np.zeros((h, w, 1), dtype=np.uint8)
        vis_img = np.concatenate([vis_img, empty], axis=2)
    # RGB 순서 주의 (OpenCV는 BGR 사용) -> tifffile은 보통 RGB.
    # 일단 그대로 두고 마스크만 확인

    # 2. 흰색 배경 캔버스 생성
    white_bg = np.ones_like(vis_img) * 255

    # 3. 마스크 적용
    # is_signal이 True인 곳은 원본(vis_img), False인 곳은 흰색(white_bg)
    # 차원 맞추기 (H,W) -> (H,W,3)
    mask_3ch = np.repeat(is_signal[..., np.newaxis], 3, axis=2)

    final_view = np.where(mask_3ch, vis_img, white_bg)

    # 4. 저장
    save_path = "qc_debug_view.png"
    cv2.imwrite(save_path, cv2.cvtColor(final_view, cv2.COLOR_RGB2BGR))

    print(f"\n💾 이미지가 저장되었습니다: {os.path.abspath(save_path)}")
    print(
        "👉 이 이미지를 열어보세요. 흰색이 아닌 부분이 전부 '세포'로 인식된 영역입니다."
    )


if __name__ == "__main__":
    visualize_background_mask(target_image_path)
