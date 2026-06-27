import os
import re
import shlex

import cv2
import numpy as np
import tifffile


def parse_windows_paths(raw_input):
    """
    Parses Windows "Copy as path" format into a list.
    """
    # Extract paths enclosed in quotes
    paths = re.findall(r'"(.*?)"', raw_input)

    # Fallback to split by space if no quotes are found
    if not paths:
        paths = [p.strip() for p in raw_input.split() if p.strip()]

    return paths


def calculate_laplacian_variance(image_path):
    """Calculates Laplacian Variance (Sharpness) of an image"""
    # Read image
    img = tifffile.imread(image_path)

    # Channel processing (RGB -> Max Projection)
    if img.ndim == 3:
        # Use Max Projection to preserve edges
        gray = np.max(img, axis=2)
    else:
        gray = img

    # Convert to float64 for precision
    gray = gray.astype(np.float64)

    # Apply Laplacian filter and calculate variance
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    return laplacian_var, img.shape, img.dtype


def main():
    print("\n" + "=" * 60)
    print(" Laplacian Variance Calculator")
    print("=" * 60)
    print("Tip: Copy paths using [Ctrl+Shift+C] and paste below.")
    print("-" * 60)

    # Get user input
    raw_input = input(">> Paste paths (Ctrl+V): ")

    if not raw_input.strip():
        print("No input provided.")
        return

    # Parse paths
    file_paths = parse_windows_paths(raw_input)
    print(f"\nDetected {len(file_paths)} files.\n")

    print(f"{'Filename':<40} | {'Laplacian Var':<15} | {'Shape':<15} | {'Status'}")
    print("-" * 90)

    # Calculate and output
    for fpath in file_paths:
        # Clean up quotes and spaces
        fpath = os.path.normpath(fpath.strip())
        filename = os.path.basename(fpath)

        if not os.path.exists(fpath):
            print(
                f"{filename[:37]+'...':<40} | {'-':<15} | {'-':<15} | File Not Found"
            )
            continue

        score, shape, dtype = calculate_laplacian_variance(fpath)

        if score is not None:
            # Use comma for large numbers
            score_str = f"{score:,.0f}"
            print(
                f"{filename[:37]+'...':<40} | {score_str:<15} | {str(shape):<15} | OK"
            )
        else:
            print(f"{filename[:37]+'...':<40} | {'Error':<15} | {'-':<15} | {dtype}")

    print("-" * 90)
    input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()
