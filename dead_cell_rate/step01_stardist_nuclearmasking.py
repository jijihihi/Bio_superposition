#!/usr/bin/env python3
"""
StarDist Nuclei Segmentation Pipeline

This script applies StarDist 2D nuclei segmentation to microscopy images,
filtering out border and small nuclei, and saves segmentation masks with 
optional visualization outputs.
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from csbdeep.utils import normalize
from skimage.exposure import rescale_intensity
from skimage.measure import regionprops
from stardist.models import StarDist2D


def parse_args():
    parser = argparse.ArgumentParser(
        description="StarDist nuclei segmentation for microscopy images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Base input directory containing image folders",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Base output directory for segmentation results",
    )
    parser.add_argument(
        "--folders",
        type=str,
        nargs="+",
        default=None,
        help="List of folder names to process. If not provided, processes all subdirectories in input_dir",
    )
    parser.add_argument(
        "--csv_output",
        type=str,
        default=None,
        help="Path for CSV results file. Defaults to output_dir/nuclei_count_results.csv",
    )
    parser.add_argument(
        "--prob_thresh",
        type=float,
        default=0.479071,
        help="StarDist probability threshold for nuclei detection",
    )
    parser.add_argument(
        "--nms_thresh",
        type=float,
        default=0.3,
        help="StarDist NMS (non-maximum suppression) threshold",
    )
    parser.add_argument(
        "--nuc_channel",
        type=int,
        default=2,
        help="Index of nucleus channel in multi-channel TIF images (e.g., 2 for Blue)",
    )
    parser.add_argument(
        "--min_area",
        type=int,
        default=40,
        help="Minimum nucleus area in pixels (smaller nuclei are filtered)",
    )
    parser.add_argument(
        "--save_visualization",
        action="store_true",
        default=True,
        help="Save visualization PNG images",
    )
    parser.add_argument(
        "--no_visualization",
        dest="save_visualization",
        action="store_false",
        help="Disable saving visualization PNG images",
    )

    return parser.parse_args()


def segment_nuclei(args):
    print(f"[*] Loading StarDist model (prob={args.prob_thresh}, nms={args.nms_thresh})...")
    model = StarDist2D.from_pretrained("2D_versatile_fluo")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.csv_output is None:
        csv_output_path = output_dir / "nuclei_count_results.csv"
    else:
        temp_path = Path(args.csv_output)
        if temp_path.is_dir() or temp_path.suffix == "":
            csv_output_path = temp_path / "nuclei_count_results.csv"
        else:
            csv_output_path = temp_path

    results_data = []
    existing_records = set()

    if csv_output_path.exists() and csv_output_path.is_file():
        with open(csv_output_path, "r", encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader, None)
            if header:
                for row in reader:
                    if len(row) >= 6:
                        results_data.append(row)
                        existing_records.add((row[0], row[1]))
        print(f"[*] Loaded {len(results_data)} records from existing CSV.")

    if not args.folders:
        args.folders = [d.name for d in input_dir.iterdir() if d.is_dir()]
        print(f"[*] Auto-detected {len(args.folders)} folders to process.")

    for folder_name in args.folders:
        folder_path = input_dir / folder_name

        if not folder_path.is_dir():
            print(f"[-] Warning: Folder not found, skipping ({folder_path})")
            continue

        print(f"[*] Processing folder: {folder_name}")

        mask_output_dir = output_dir / folder_name / "mask"
        vis_output_dir = output_dir / folder_name / "visualization"
        mask_output_dir.mkdir(parents=True, exist_ok=True)
        
        if args.save_visualization:
            vis_output_dir.mkdir(parents=True, exist_ok=True)

        tif_files = list(folder_path.glob("*.tif")) + list(folder_path.glob("*.tiff"))

        for tif_path in tif_files:
            base_filename = tif_path.stem
            expected_mask_path = mask_output_dir / f"{base_filename}_mask.tif"

            if expected_mask_path.exists():
                continue

            data_cube = tifffile.imread(tif_path)

            if data_cube.ndim == 3:
                nuc_ch_data = data_cube[:, :, args.nuc_channel]
            else:
                nuc_ch_data = data_cube

            normalized_nuc = normalize(nuc_ch_data, 1, 99.8, axis=(0, 1))

            labels, _ = model.predict_instances(
                normalized_nuc,
                prob_thresh=args.prob_thresh,
                nms_thresh=args.nms_thresh,
            )

            total_nuclei = int(np.max(labels))
            valid_labels_list = []
            border_nuclei_count = 0
            small_nuclei_count = 0

            if total_nuclei > 0:
                height, width = labels.shape
                props = regionprops(labels)

                for prop in props:
                    min_r, min_c, max_r, max_c = prop.bbox
                    is_border = (min_r == 0 or min_c == 0 or max_r == height or max_c == width)
                    is_small = prop.area <= args.min_area

                    if is_border:
                        border_nuclei_count += 1
                    elif is_small:
                        small_nuclei_count += 1
                    else:
                        valid_labels_list.append(prop.label)

            valid_nuclei_count = len(valid_labels_list)

            results_data.append([
                folder_name,
                tif_path.name,
                total_nuclei,
                border_nuclei_count,
                small_nuclei_count,
                valid_nuclei_count,
            ])

            segmentation_mask = np.zeros_like(labels, dtype=np.uint16)
            for label_id in valid_labels_list:
                segmentation_mask[labels == label_id] = label_id

            tifffile.imwrite(expected_mask_path, segmentation_mask)

            if args.save_visualization:
                nuc_ch_8bit = rescale_intensity(nuc_ch_data, in_range="image", out_range=(0, 255)).astype(np.uint8)
                rgb_image = np.stack([nuc_ch_8bit] * 3, axis=-1)
                binary_mask = segmentation_mask > 0
                rgb_image[binary_mask] = [0, 0, 255]

                output_vis_path = vis_output_dir / f"{base_filename}_visualization.png"
                plt.imsave(output_vis_path, rgb_image)

    print("[*] Saving CSV results...")
    with open(csv_output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "FolderName",
            "ImageFileName",
            "TotalNuclei",
            "BorderNuclei",
            "SmallNuclei",
            "ValidNuclei_Count",
        ])
        if results_data:
            writer.writerows(results_data)

    print("[*] Processing complete.")
    print(f"    Total records: {len(results_data)}")
    print(f"    Output directory: {output_dir}")


if __name__ == "__main__":
    args = parse_args()
    segment_nuclei(args)
