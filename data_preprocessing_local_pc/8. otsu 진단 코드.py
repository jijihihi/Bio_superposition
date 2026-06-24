import argparse
import os

import cv2
import numpy as np
import tifffile
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(description="Debug RGB Independent Otsu Logic")
    parser.add_argument(
        "--input_dir",
        type=str,
        default=r"C:\Users\admin\Desktop\cropped_image",
        help="Input directory",
    )
    parser.add_argument(
        "--debug_dir",
        type=str,
        default=r"C:\Users\admin\Desktop\cropped_image\debug",
        help="Output directory for debug images",
    )
    parser.add_argument(
        "--sample_count", type=int, default=50, help="Number of samples"
    )
    return parser.parse_args()


def normalize_to_8bit(img):
    """16bit 이미지를 눈에 잘 보이게 8bit로 스케일링 (시각화용)"""
    img = img.astype(np.float64)
    min_val = np.min(img)
    max_val = np.max(img)
    if max_val - min_val == 0:
        return np.zeros_like(img, dtype=np.uint8)
    return ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)


def process_debug_image(img, filename, save_dir):
    h, w, c = img.shape

    # --- 1. RGB 개별 Otsu 로직 (필터링 코드와 100% 동일) ---
    final_signal_mask = np.zeros((h, w), dtype=np.bool_)

    # 각 채널별 Otsu 수행 내용 기록용
    debug_channels = []

    for i in range(c):
        channel = img[..., i].astype(np.float64)

        # 시각화용 8bit (Max값 기준 정규화)
        max_val = np.max(channel)
        if max_val == 0:
            debug_channels.append(np.zeros((h, w), dtype=np.uint8))
            continue

        channel_8bit = ((channel / max_val) * 255).astype(np.uint8)

        # Otsu 수행
        otsu_val, _ = cv2.threshold(
            channel_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        real_thresh = (otsu_val / 255.0) * max_val

        # 세포(Signal) 판정
        is_signal = channel > real_thresh
        final_signal_mask = np.bitwise_or(final_signal_mask, is_signal)

    # --- 2. 배경 비율 계산 ---
    # 신호가 아닌 곳이 배경
    bg_mask = ~final_signal_mask
    bg_pixel_count = np.sum(bg_mask)
    bg_frac = bg_pixel_count / (h * w)

    # --- 3. 시각화 이미지 생성 ---

    # A. 원본 이미지 (보기 좋게 8bit 변환)
    view_img = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(c):
        view_img[..., i] = normalize_to_8bit(img[..., i])
    # Tifffile은 RGB 순서, OpenCV는 BGR 순서 -> 변환 필요
    view_img_bgr = cv2.cvtColor(view_img, cv2.COLOR_RGB2BGR)

    # B. 마스크 이미지 (세포=흰색, 배경=검은색)
    mask_view = final_signal_mask.astype(np.uint8) * 255
    mask_view_bgr = cv2.cvtColor(mask_view, cv2.COLOR_GRAY2BGR)

    # 세포 영역을 초록색으로 칠하기 (시각적 확인용)
    # 마스크가 있는 곳의 Green 채널을 255로
    overlay = view_img_bgr.copy()
    overlay[final_signal_mask] = [0, 255, 0]  # BGR 기준 Green

    # 원본과 마스크 섞기 (투명도)
    mixed = cv2.addWeighted(view_img_bgr, 0.7, overlay, 0.3, 0)

    # C. 두 이미지 나란히 붙이기
    combined = np.hstack((view_img_bgr, mixed))

    # D. 텍스트 쓰기 (배경 비율)
    text = f"BG: {bg_frac*100:.1f}%"
    color = (0, 0, 255) if bg_frac > 0.7 else (0, 255, 0)  # 70% 넘으면 빨간글씨
    cv2.putText(combined, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # 저장
    save_path = os.path.join(
        save_dir, f"Debug_{bg_frac*100:.0f}pct_{filename[:-4]}.png"
    )
    cv2.imwrite(save_path, combined)


def run_debug(args):
    print("Starting RGB Otsu Debugger...")
    if not os.path.exists(args.debug_dir):
        os.makedirs(args.debug_dir)

    files = []
    for r, d, f in os.walk(args.input_dir):
        for file in f:
            if file.lower().endswith((".tif", ".tiff")):
                files.append(os.path.join(r, file))

    print(f"Total files found: {len(files)}")
    print(f"Processing first {args.sample_count} files...")

    target_files = files[: args.sample_count]

    for fpath in tqdm(target_files):
        try:
            img = tifffile.imread(fpath)
            process_debug_image(img, os.path.basename(fpath), args.debug_dir)
        except Exception as e:
            print(f"Error: {e}")

    print("\nDone! Check the folder.")


if __name__ == "__main__":
    args = get_args()
    run_debug(args)
