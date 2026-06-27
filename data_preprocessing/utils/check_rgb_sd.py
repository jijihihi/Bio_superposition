import os

import numpy as np
import tifffile


def analyze_channel_stats(image_path):
    print(f"run analysis for: {os.path.basename(image_path)}")

    # 1. Load image (keep 16-bit)
    img = tifffile.imread(image_path)

    # Check and adjust dimensions to (H, W, C)
    if img.ndim == 2:  # (H, W) -> (H, W, 1)
        img = img[..., np.newaxis]
    elif img.shape[0] < img.shape[2]:  # (C, H, W) -> (H, W, C)
        img = np.transpose(img, (1, 2, 0))

    h, w, c = img.shape
    print(f"  - Shape: {img.shape} (H, W, C)")

    # 2. Scale to [0, 1] (same as Instance Norm)
    img_float = img.astype(np.float32) / 65535.0

    print("-" * 60)
    print(
        f"{'Channel':^10} | {'Std':^15} | {'Mean':^15} | {'Max':^10}"
    )
    print("-" * 60)

    # 3. Calculate channel statistics
    for i in range(c):
        channel_data = img_float[..., i]

        std_val = np.std(channel_data)
        mean_val = np.mean(channel_data)
        max_val = np.max(channel_data)

        # Estimate channel name
        ch_name = f"Ch {i}"

        print(
            f"{ch_name:^10} | {std_val:^15.6f} | {mean_val:^15.6f} | {max_val:^10.4f}"
        )

    print("-" * 60)
    print("💡 [Analysis Tips]")
    print("1. If 'Std' is lower than threshold (e.g. 0.05), it is considered noise.")
    print(
        "2. Check the Std of background-only images and set threshold slightly higher."
    )
    print("-" * 60 + "\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze image channel statistics (RGB SD).")
    parser.add_argument("image_paths", nargs="+", help="Paths to images to analyze")
    args = parser.parse_args()

    for path in args.image_paths:
        if os.path.exists(path):
            analyze_channel_stats(path)
        else:
            print(f"Path not found: {path}")

if __name__ == "__main__":
    main()
