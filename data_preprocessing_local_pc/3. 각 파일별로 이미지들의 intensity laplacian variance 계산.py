import os
import argparse
import glob
import numpy as np
import tifffile
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import sys

def get_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Quality Control (QC) Analysis Tool for Microscopy Images (Per-Channel)"
    )
    parser.add_argument(
        "--input_dir", 
        type=str, 
        default=r"C:\Users\admin\Desktop\MIP",
        #required=True, 
        help="Path to the root directory containing image folders."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=r"C:\Users\admin\Desktop\MIP\QC_Output", 
        help="Directory to save QC results and plots."
    )
    parser.add_argument(
        "--pattern", 
        type=str, 
        default="*.tif", 
        help="File pattern to search for (default: *.tif)."
    )
    return parser.parse_args()

def calculate_channel_metrics(img):
    """
    Calculates Mean Intensity and Laplacian Variance for EACH channel independently.
    
    Args:
        img (numpy.ndarray): Image array (H, W) or (H, W, C).
        
    Returns:
        dict: Dictionary containing metrics for each channel (e.g., Ch0_Intensity, Ch0_Variance).
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
        return None # Unsupported format

    for i, channel_data in enumerate(channels):
        # 1. Intensity
        mean_intensity = np.mean(channel_data)
        
        # 2. Laplacian Variance (Sharpness)
        # Convert to float64 to prevent overflow/underflow during variance calculation
        src = channel_data.astype(np.float64)
        laplacian_var = cv2.Laplacian(src, cv2.CV_64F).var()
        
        metrics[f"Ch{i}_Intensity"] = mean_intensity
        metrics[f"Ch{i}_LaplacianVar"] = laplacian_var
        
    metrics["Num_Channels"] = len(channels)
    return metrics

def find_image_files(root_dir, pattern):
    """Recursively finds all files matching the pattern (Case-Insensitive)."""
    files = []
    # 패턴을 소문자로 통일 (예: *.tif)
    pattern = pattern.lower() 
    for root, _, filenames in os.walk(root_dir):
        for filename in filenames:
            # 파일 확장자도 소문자로 변경하여 비교
            if filename.lower().endswith('.tif') or filename.lower().endswith('.tiff'):
                files.append(os.path.join(root, filename))
    return sorted(files)

def run_analysis(args):
    """Main execution flow."""
    
    # Setup directories
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"🔍 Starting QC Analysis...")
    print(f"   - Input Root: {args.input_dir}")
    print(f"   - Output Dir: {args.output_dir}")
    
    # 1. Find Files
    all_files = find_image_files(args.input_dir, args.pattern)
    if not all_files:
        print("❌ No files found matching the pattern.")
        return

    print(f"✅ Found {len(all_files)} images. Processing...")

    # 2. Process Images
    results_list = []
    
    for file_path in tqdm(all_files, desc="Processing Images"):
        try:
            img = tifffile.imread(file_path)
            
            # Extract Group Name (Parent folder name)
            group_name = os.path.basename(os.path.dirname(file_path))
            # Handle standard structure like '.../Group_A/Images/file.tif'
            if group_name.lower() == "images":
                group_name = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
            
            # Calculate Metrics
            metrics = calculate_channel_metrics(img)
            
            if metrics:
                record = {
                    "Group": group_name,
                    "Filename": os.path.basename(file_path),
                    "Full_Path": file_path
                }
                record.update(metrics)
                results_list.append(record)
                
        except Exception as e:
            # Print error but continue
            print(f"\n⚠️ Error processing {os.path.basename(file_path)}: {e}")

    # 3. Save Raw Data
    if not results_list:
        print("❌ No data collected.")
        return

    df = pd.DataFrame(results_list)
    raw_csv_path = os.path.join(args.output_dir, "qc_metrics_raw_per_channel.csv")
    df.to_csv(raw_csv_path, index=False)
    print(f"\n📄 Raw data saved to: {raw_csv_path}")

    # 4. Generate Visualization (Long-form data for Seaborn)
    # We need to reshape the dataframe to plot "Channel" as a category
    # Identify channel columns dynamically
    num_channels = int(df["Num_Channels"].max())
    
    plot_data = []
    for _, row in df.iterrows():
        for i in range(num_channels):
            if f"Ch{i}_Intensity" in row:
                plot_data.append({
                    "Group": row["Group"],
                    "Filename": row["Filename"],
                    "Channel": f"Ch{i}", # e.g., Ch0, Ch1
                    "Intensity": row[f"Ch{i}_Intensity"],
                    "Laplacian_Variance": row[f"Ch{i}_LaplacianVar"]
                })
    
    plot_df = pd.DataFrame(plot_data)
    
    # 5. Plotting
    sns.set(style="whitegrid", context="paper")
    
    # Plot A: Boxplot of Laplacian Variance per Channel (To see which channel is sharpest)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=plot_df, x="Channel", y="Laplacian_Variance", hue="Group", showfliers=False)
    plt.yscale("log")
    plt.title("Distribution of Laplacian Variance per Channel")
    plt.ylabel("Laplacian Variance (Log Scale)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "QC_Boxplot_Variance_per_Channel.png"), dpi=300)
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
        s=40
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Intensity vs. Sharpness (Per Channel)")
    plt.xlabel("Mean Intensity (Log Scale)")
    plt.ylabel("Laplacian Variance (Log Scale)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "QC_Scatter_Channel_Comparison.png"), dpi=300)
    plt.close()
    
    # 6. Summary Statistics
    print("\n" + "="*60)
    print("📊 QC Summary Report (Per Channel)")
    print("="*60)
    
    summary = plot_df.groupby(["Group", "Channel"])[["Intensity", "Laplacian_Variance"]].mean()
    print(summary)
    summary.to_csv(os.path.join(args.output_dir, "qc_summary_stats.csv"))
    
    print("\n✅ All tasks completed successfully.")

if __name__ == "__main__":
    args = get_args()
    if os.path.exists(args.input_dir):
        run_analysis(args)
    else:
        print(f"❌ Input directory does not exist: {args.input_dir}")
        sys.exit(1)