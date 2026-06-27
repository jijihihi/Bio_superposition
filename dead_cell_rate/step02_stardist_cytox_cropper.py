import argparse
import logging
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import tifffile
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_args() -> argparse.Namespace:
    """Parses command line arguments for the cropping script."""
    parser = argparse.ArgumentParser(
        description="Crop StarDist masks and CytoxGreen images based on QC-passed images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Path arguments
    parser.add_argument(
        "--qc_passed_dir",
        type=str,
        default=r"D:\From_C_drive\cropped_image",
        help="Directory containing QC-passed cropped images.",
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        default=r"C:\Users\admin\Desktop\MIP_segmentation",
        help="Directory containing StarDist mask images.",
    )
    parser.add_argument(
        "--cytox_dir",
        type=str,
        default=r"C:\Users\admin\Desktop\MIP_cytox",
        help="Directory containing CytoxGreen images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=r"C:\Users\admin\Desktop\세포사멸율_data_new_class",
        help="Output directory for cropped results.",
    )

    parser.add_argument(
        "--folders",
        type=str,
        nargs="+",
        default=None,
        help="List of folder names to process. If not provided, processes all subdirectories in qc_passed_dir.",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=128,
        help="Size of patches (must match original crop size).",
    )

    return parser.parse_args()


def parse_qc_filename(filename: str) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    """Extracts plate position and coordinates from the QC filename.

    Example: '004001_r02c02f01_Composite_RGB_x120_y340.tif'
    Returns: ('004001_r02c02f01', 120, 340)
    """
    coord_match = re.search(r"_x(\d+)_y(\d+)", filename)
    if not coord_match:
        return None, None, None

    x, y = int(coord_match.group(1)), int(coord_match.group(2))
    base = filename[: coord_match.start()]

    plate_match = re.match(r"^(\d+_[a-zA-Z]\d+c\d+f\d+)", base)
    if not plate_match:
        return None, None, None

    return plate_match.group(1), x, y


def find_target_file(base_dir: Path, folder_name: str, plate_position: str, pattern_suffix: str) -> Optional[Path]:
    """Finds a target file matching the plate position and suffix pattern."""
    target_folder = base_dir / folder_name
    if not target_folder.exists():
        return None

    matches = list(target_folder.glob(f"{plate_position}_{pattern_suffix}"))
    return matches[0] if matches else None


def crop_image(img: np.ndarray, x: int, y: int, patch_size: int) -> np.ndarray:
    """Crops a patch from the image at the specified coordinates."""
    return img[y : y + patch_size, x : x + patch_size]


def run_cropping(args: argparse.Namespace) -> None:
    """Main execution block to crop masks and CytoxGreen images based on QC data."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_stats = {"total": 0, "mask_success": 0, "cytox_success": 0}

    if not args.folders:
        qc_passed_path = Path(args.qc_passed_dir)
        if qc_passed_path.exists():
            args.folders = [d.name for d in qc_passed_path.iterdir() if d.is_dir()]
            logger.info(f"Auto-detected {len(args.folders)} folders to process.")
        else:
            logger.error(f"QC passed directory not found: {args.qc_passed_dir}")
            return

    for folder_name in args.folders:
        qc_folder = Path(args.qc_passed_dir) / folder_name
        if not qc_folder.exists():
            logger.warning(f"QC directory not found for folder: {folder_name}")
            continue

        logger.info(f"Processing folder: {folder_name}")

        # Setup structured output directories
        out_dirs = {
            "mask": output_dir / folder_name / "QC" / "stardist_mask",
            "cytox": output_dir / folder_name / "QC" / "cytoxgreen",
        }
        for d in out_dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        qc_files = list(qc_folder.glob("*.tif")) + list(qc_folder.glob("*.tiff"))
        if not qc_files:
            logger.info(f"No QC-passed images found in {folder_name}")
            continue

        # Memory caches for large source images per plate position
        mask_cache: Dict[str, Optional[np.ndarray]] = {}
        cytox_cache: Dict[str, Optional[np.ndarray]] = {}

        for qc_file in tqdm(qc_files, desc=f"Progress ({folder_name})", leave=False):
            summary_stats["total"] += 1
            plate_pos, x, y = parse_qc_filename(qc_file.name)

            if plate_pos is None or x is None or y is None:
                logger.error(f"Failed to parse coordinates from filename: {qc_file.name}")
                continue

            # 1. Process StarDist Mask
            if plate_pos not in mask_cache:
                mask_path = find_target_file(Path(args.mask_dir), f"{folder_name}/mask", plate_pos, "*_mask.tif")
                mask_cache[plate_pos] = tifffile.imread(mask_path) if mask_path else None

            mask_img = mask_cache[plate_pos]
            if mask_img is not None:
                cropped = crop_image(mask_img, x, y, args.patch_size)
                if cropped.shape[:2] == (args.patch_size, args.patch_size):
                    tifffile.imwrite(out_dirs["mask"] / f"{qc_file.stem}_mask.tif", cropped)
                    summary_stats["mask_success"] += 1

            # 2. Process CytoxGreen Image
            if plate_pos not in cytox_cache:
                cytox_path = find_target_file(Path(args.cytox_dir), folder_name, plate_pos, "*Cytox*.tif")
                cytox_cache[plate_pos] = tifffile.imread(cytox_path) if cytox_path else None

            cytox_img = cytox_cache[plate_pos]
            if cytox_img is not None:
                cropped = crop_image(cytox_img, x, y, args.patch_size)
                if cropped.shape[:2] == (args.patch_size, args.patch_size):
                    tifffile.imwrite(out_dirs["cytox"] / f"{qc_file.stem}_cytox.tif", cropped)
                    summary_stats["cytox_success"] += 1

        # Clear memory cache after finishing each folder
        mask_cache.clear()
        cytox_cache.clear()

    # Log execution summary
    logger.info("=" * 50)
    logger.info("Processing complete. Execution Summary:")
    logger.info(f"  Total QC Images Evaluated: {summary_stats['total']}")
    logger.info(f"  Successfully Cropped Masks: {summary_stats['mask_success']}")
    logger.info(f"  Successfully Cropped Cytox: {summary_stats['cytox_success']}")
    logger.info(f"  Output Directory: {output_dir}")
    logger.info("=" * 50)


if __name__ == "__main__":
    run_cropping(get_args())