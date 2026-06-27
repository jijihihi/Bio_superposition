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
        default=r"./Output/patches",
        help="Input directory",
    )
    parser.add_argument(
        "--debug_dir",
        type=str,
        default=r"./Output/patches/debug",
        help="Output directory for debug images",
    )
    parser.add_argument(
        "--sample_count", type=int, default=50, help="Number of samples"
    )
    return parser.parse_args()


def normalize_to_8bit(img):
    """Scale 16-bit image to 8-bit for visualization"""
    img = img.astype(np.float64)
    min_val = np.min(img)
    max_val = np.max(img)
    if max_val - min_val == 0:
        return np.zeros_like(img, dtype=np.uint8)
    return ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)


def process_debug_image(img, filename, save_dir):
    h, w, c = img.shape

    # --- 1. RGB Independent Otsu Logic (100% identical to filtering code) ---
    final_signal_mask = np.zeros((h, w), dtype=np.bool_)

    # Record Otsu execution per channel
    debug_channels = []

    for i in range(c):
        channel = img[..., i].astype(np.float64)

        # 8-bit for visualization (normalize by Max value)
        max_val = np.max(channel)
        if max_val == 0:
            debug_channels.append(np.zeros((h, w), dtype=np.uint8))
            continue

        channel_8bit = ((channel / max_val) * 255).astype(np.uint8)

        # Perform Otsu
        otsu_val, _ = cv2.threshold(
            channel_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        real_thresh = (otsu_val / 255.0) * max_val

        # Determine cell (Signal)
        is_signal = channel > real_thresh
        final_signal_mask = np.bitwise_or(final_signal_mask, is_signal)

    # --- 2. Calculate Background Ratio ---
    # Non-signal is background
    bg_mask = ~final_signal_mask
    bg_pixel_count = np.sum(bg_mask)
    bg_frac = bg_pixel_count / (h * w)

    # --- 3. Create Visualization Image ---

    # A. Original image (8-bit conversion for viewing)
    view_img = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(c):
        view_img[..., i] = normalize_to_8bit(img[..., i])
    # Tifffile is RGB, OpenCV is BGR -> conversion needed
    view_img_bgr = cv2.cvtColor(view_img, cv2.COLOR_RGB2BGR)

    # B. Mask image (Cell=White, Background=Black)
    mask_view = final_signal_mask.astype(np.uint8) * 255
    mask_view_bgr = cv2.cvtColor(mask_view, cv2.COLOR_GRAY2BGR)

    # Color cell area in green (for visual confirmation)
    # Set Green channel to 255 where mask exists
    overlay = view_img_bgr.copy()
    overlay[final_signal_mask] = [0, 255, 0]  # Green in BGR

    # Blend original and mask (transparency)
    mixed = cv2.addWeighted(view_img_bgr, 0.7, overlay, 0.3, 0)

    # C. Place two images side by side
    combined = np.hstack((view_img_bgr, mixed))

    # D. Write text (Background Ratio)
    text = f"BG: {bg_frac*100:.1f}%"
    color = (0, 0, 255) if bg_frac > 0.7 else (0, 255, 0)  # Red text if over 70%
    cv2.putText(combined, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Save
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
        img = tifffile.imread(fpath)
        process_debug_image(img, os.path.basename(fpath), args.debug_dir)

    print("\nDone! Check the folder.")


if __name__ == "__main__":
    args = get_args()
    run_debug(args)
