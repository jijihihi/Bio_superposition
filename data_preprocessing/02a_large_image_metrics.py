import argparse
import glob
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tifffile
from tqdm import tqdm


def get_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Quality Control (QC) Analysis Tool for Microscopy Images (Per-Channel)"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=r"./Output/MIP",
        help="Path to the root directory containing image folders.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=r"./Output/MIP_QC_Output",
        help="Directory to save QC results and plots.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.tif",
        help="File pattern to search for (default: *.tif).",
    )
    return parser.parse_args()


def calculate_channel_metrics(img, filename):
    """
    Calculates Mean Intensity and Laplacian Variance for EACH channel independently.
    Maps channel names based on the filename (e.g. Composite_RGB vs Cytox).
    """
    metrics = {}

    # Handle Grayscale (2D) vs Multi-channel (3D)
    if img.ndim == 2:
        channels = [img]
    elif img.ndim == 3:
        # Assuming format is (H, W, C). If (C, H, W), transpose might be needed.
        # Here we assume standard HWC from tifffile/cv2
        channels = [img[:, :, i] for i in range(img.shape[-1])]
    else:
        return None  # Unsupported format

    is_rgb = "Composite_RGB" in filename
    is_cytox = "Cytox" in filename

    for i, channel_data in enumerate(channels):
        # 1. Intensity
        mean_intensity = np.mean(channel_data)

        # 2. Laplacian Variance (Sharpness)
        # Convert to float64 to prevent overflow/underflow during variance calculation
        src = channel_data.astype(np.float64)
        laplacian_var = cv2.Laplacian(src, cv2.CV_64F).var()

        # Map channel indices to biological names
        if is_rgb and len(channels) == 3:
            if i == 0:
                ch_name = "Ch3_Red"
            elif i == 1:
                ch_name = "Ch4_Green"
            elif i == 2:
                ch_name = "Ch1_Blue"
            else:
                ch_name = f"Ch{i}"
        elif is_cytox:
            ch_name = "Ch2_Cytox"
        else:
            ch_name = f"Ch{i}"

        metrics[f"{ch_name}_Intensity"] = mean_intensity
        metrics[f"{ch_name}_LaplacianVar"] = laplacian_var

    metrics["Num_Channels"] = len(channels)
    return metrics


def find_image_files(root_dir, pattern):
    """Recursively finds all files matching the pattern (Case-Insensitive)."""
    files = []
    # Unify pattern to lowercase (e.g. *.tif)
    pattern = pattern.lower()
    for root, _, filenames in os.walk(root_dir):
        for filename in filenames:
            # Compare extension in lowercase
            if filename.lower().endswith(".tif") or filename.lower().endswith(".tiff"):
                files.append(os.path.join(root, filename))
    return sorted(files)


def run_analysis(args):
    """Main execution flow."""

    # Setup directories
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Find Files
    all_files = find_image_files(args.input_dir, args.pattern)


    # 2. Process Images
    results_list = []

    for file_path in tqdm(all_files, desc="Processing Images"):
        img = tifffile.imread(file_path)

        # Extract Group Name (Parent folder name)
        group_name = os.path.basename(os.path.dirname(file_path))
        # Handle standard structure like '.../Group_A/Images/file.tif'
        if group_name.lower() == "images":
            group_name = os.path.basename(
                os.path.dirname(os.path.dirname(file_path))
            )

        # Calculate Metrics
        metrics = calculate_channel_metrics(img, os.path.basename(file_path))

        if metrics:
            record = {
                "Group": group_name,
                "Filename": os.path.basename(file_path),
                "Full_Path": file_path,
            }
            record.update(metrics)
            results_list.append(record)

    # 3. Save Raw Data
    if not results_list:
        print("No data collected.")
        return

    df = pd.DataFrame(results_list)
    
    # Reorder columns explicitly so Intensity/LaplacianVar are grouped together
    base_cols = ["Group", "Filename", "Full_Path", "Num_Channels"]
    ch_cols = sorted([c for c in df.columns if c.endswith("_Intensity") or c.endswith("_LaplacianVar")])
    # Ensure columns exist in dataframe
    final_cols = [c for c in base_cols if c in df.columns] + ch_cols
    df = df[final_cols]

    raw_csv_path = os.path.join(args.output_dir, "qc_metrics_raw_per_channel.csv")
    df.to_csv(raw_csv_path, index=False)
    print(f"\nRaw data saved to: {raw_csv_path}")

    # 4. Generate Visualization (Long-form data for Seaborn)
    # We need to reshape the dataframe to plot "Channel" as a category
    # Identify channel bases dynamically (e.g. Ch1_Blue, Ch2_Cytox)
    channel_bases = sorted(list(set([c.replace("_Intensity", "") for c in df.columns if c.endswith("_Intensity")])))

    plot_data = []
    for _, row in df.iterrows():
        for ch_base in channel_bases:
            if f"{ch_base}_Intensity" in row and not pd.isna(row.get(f"{ch_base}_Intensity")):
                plot_data.append(
                    {
                        "Group": row["Group"],
                        "Filename": row["Filename"],
                        "Channel": ch_base,
                        "Intensity": row[f"{ch_base}_Intensity"],
                        "Laplacian_Variance": row[f"{ch_base}_LaplacianVar"],
                    }
                )

    plot_df = pd.DataFrame(plot_data)

    # 5. Plotting
    sns.set(style="whitegrid", context="paper")

    # Plot A: Boxplot of Laplacian Variance per Channel (To see which channel is sharpest)
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=plot_df, x="Channel", y="Laplacian_Variance", hue="Group", showfliers=False
    )
    plt.yscale("log")
    plt.title("Distribution of Laplacian Variance per Channel")
    plt.ylabel("Laplacian Variance (Log Scale)")
    plt.tight_layout()
    plt.savefig(
        os.path.join(args.output_dir, "QC_Boxplot_Variance_per_Channel.png"), dpi=300
    )
    plt.close()

    # Plot B: Scatter Plot (Intensity vs Variance) - Colored by Channel
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=plot_df,
        x="Intensity",
        y="Laplacian_Variance",
        hue="Channel",
        style="Group",
        alpha=0.7,
        s=40,
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Intensity vs. Sharpness (Per Channel)")
    plt.xlabel("Mean Intensity (Log Scale)")
    plt.ylabel("Laplacian Variance (Log Scale)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(
        os.path.join(args.output_dir, "QC_Scatter_Channel_Comparison.png"), dpi=300
    )
    plt.close()

    # 6. Summary Statistics
    print("\n" + "=" * 60)
    print("QC Summary Report (Per Channel)")
    print("=" * 60)

    summary = plot_df.groupby(["Group", "Channel"])[
        ["Intensity", "Laplacian_Variance"]
    ].mean()
    print(summary)
    summary.to_csv(os.path.join(args.output_dir, "qc_summary_stats.csv"))

    print("\nAll tasks completed successfully.")


if __name__ == "__main__":
    args = get_args()
    if os.path.exists(args.input_dir):
        run_analysis(args)
    else:
        print(f"Input directory does not exist: {args.input_dir}")
        sys.exit(1)
