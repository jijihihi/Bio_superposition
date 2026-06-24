# ==============================================================================
# step05_texture_shape_eval.py
# ==============================================================================
# Texture & Shape Suppression 정량 평가 스크립트
# - step02_transform.py: 변환 클래스
# - step03_hyperparameters.py: 하이퍼파라미터 설정
# - step04_evaluate_transform.py: 메트릭 함수
#
# Usage: python step05_texture_shape_eval.py --val_dir /path/to/model --output_dir /path/to/output
# ==============================================================================

import argparse
import csv
import io
import logging
import os
import pickle
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import tifffile
except ImportError:
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "tifffile"])
    import tifffile

# Add current directory to path for Colab compatibility
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

# Import transforms
from step02_transform import (BilateralFilter, BoxBlur, FastNLMeansDenoising,
                              GaussianBlur, MedianFilter, PatchShuffle)
# Import hyperparameters
from step03_hyperparameters import (BILATERAL_PARAMS, BOX_PARAMS,
                                    EVAL_SETTINGS, GAUSSIAN_PARAMS,
                                    MEDIAN_PARAMS, METRIC_PARAMS,
                                    NLMEANS_PARAMS, PATCH_SHUFFLE_PARAMS,
                                    get_all_params)
# Import metrics
from step04_evaluate_transform import (compute_ESSIM, compute_GC, compute_HFE,
                                       compute_LV)

# ==============================================================================
# Logging
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("TextureShapeEval")


# ==============================================================================
# Transform Configuration Builder
# ==============================================================================
def build_transform_configs():
    """Build transform configurations from step03 hyperparameters."""
    configs = []

    # --- Bilateral Filter ---
    for p in BILATERAL_PARAMS["diagonal_sweep"]:
        configs.append(
            {
                "name": f"Bilateral (σc={p['sigma_color']}, k={p['k']})",
                "type": "texture",
                "category": "bilateral",
                "transform": BilateralFilter(
                    d=p["k"],
                    sigma_color=p["sigma_color"],
                    sigma_space=BILATERAL_PARAMS["sigma_space"],
                    p=1.0,
                ),
            }
        )

    # --- Gaussian Blur ---
    for p in GAUSSIAN_PARAMS["diagonal_sweep"]:
        configs.append(
            {
                "name": f"Gaussian (σ={p['sigma']}, k={p['k']})",
                "type": "texture",
                "category": "gaussian",
                "transform": GaussianBlur(k=p["k"], sigma=p["sigma"], p=1.0),
            }
        )

    # --- NLMeans ---
    h_scale = NLMEANS_PARAMS["h_scale"]
    for p in NLMEANS_PARAMS["diagonal_sweep"]:
        configs.append(
            {
                "name": f"NLMeans (h={p['h']}, k={p['k']})",
                "type": "texture",
                "category": "nlmeans",
                "transform": FastNLMeansDenoising(
                    h=p["h"] / h_scale, patch_size=p["k"], patch_distance=p["k"], p=1.0
                ),
            }
        )

    # --- Box Blur ---
    configs.append(
        {
            "name": f"Box (k={BOX_PARAMS['k']})",
            "type": "texture",
            "category": "box",
            "transform": BoxBlur(k=BOX_PARAMS["k"], p=1.0),
        }
    )

    # --- Median Filter ---
    configs.append(
        {
            "name": f"Median (k={MEDIAN_PARAMS['k']})",
            "type": "texture",
            "category": "median",
            "transform": MedianFilter(k=MEDIAN_PARAMS["k"], p=1.0),
        }
    )

    # --- Patch Shuffle ---
    configs.append(
        {
            "name": f"PatchShuffle (g={PATCH_SHUFFLE_PARAMS['grid_size']})",
            "type": "shape",
            "category": "patch_shuffle",
            "transform": PatchShuffle(
                grid_size=PATCH_SHUFFLE_PARAMS["grid_size"], p=1.0
            ),
        }
    )

    return configs


# ==============================================================================
# Data Loading
# ==============================================================================
def load_split_csv(csv_path: str) -> List[Dict]:
    """Load split CSV."""
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sample_per_class(
    samples: List[Dict], n_per_class: int, seed: int = 42
) -> List[Dict]:
    """Sample n_per_class images per class label."""
    by_label = defaultdict(list)
    for s in samples:
        by_label[s.get("label", "")].append(s)

    rng = np.random.RandomState(seed)
    result = []
    for label, items in sorted(by_label.items()):
        rng.shuffle(items)
        result.extend(items[:n_per_class])

    return result


def build_tar_index(tar_path: str) -> Dict:
    """
    Build tar index from .pkl file or by scanning tar contents.
    Returns dict mapping prefix -> (offset, size) for tif files.
    """
    import tarfile

    # Try .pkl index first
    idx_path = tar_path + ".pkl"
    if os.path.exists(idx_path):
        with open(idx_path, "rb") as f:
            pairs = pickle.load(f)
        return {p[0]: (p[1], p[2]) for p in pairs}

    # Otherwise, build index by scanning tar
    index = {}
    try:
        with tarfile.open(tar_path, "r") as tf:
            for member in tf.getmembers():
                if member.isfile() and member.name.endswith((".tif", ".tiff")):
                    # Extract prefix from filename (e.g., "key.tif" -> "key")
                    prefix = os.path.basename(member.name).rsplit(".", 1)[0]
                    index[prefix] = (member.offset_data, member.size)
    except Exception as e:
        logger.warning(f"Failed to scan tar {tar_path}: {e}")

    return index


# ==============================================================================
# Evaluation
# ==============================================================================
def evaluate_image(img: np.ndarray, configs: List[Dict]) -> List[Dict]:
    """Evaluate single image with all transforms."""
    # Get metric params from step03
    win = METRIC_PARAMS["window_size"]
    rad = METRIC_PARAMS["freq_radius"]

    # Convert to grayscale float
    if img.ndim == 3:
        gray = np.mean(img.astype(np.float64), axis=2)
    else:
        gray = img.astype(np.float64)
    gray_norm = gray / (gray.max() + 1e-8)

    results = []
    for cfg in configs:
        try:
            transform = cfg["transform"]
            if img.ndim == 3:
                filtered = np.stack(
                    [transform.apply(img[:, :, c]) for c in range(img.shape[2])], axis=2
                )
                filt_gray = np.mean(filtered.astype(np.float64), axis=2)
            else:
                filtered = transform.apply(img)
                filt_gray = filtered.astype(np.float64)

            filt_norm = filt_gray / (filt_gray.max() + 1e-8)

            # Compute metrics using step03 params
            lv = compute_LV(gray_norm, filt_norm, win)
            hfe = compute_HFE(gray_norm, filt_norm, rad)
            essim = compute_ESSIM(gray_norm, filt_norm)
            gc = compute_GC(gray_norm, filt_norm)

            results.append(
                {
                    "transform": cfg["name"],
                    "type": cfg["type"],
                    "category": cfg["category"],
                    "LV": lv,
                    "HFE": hfe,
                    "ESSIM": essim,
                    "GC": gc,
                    "Texture": (lv + hfe) / 2,
                    "Shape": (essim + gc) / 2,
                }
            )
        except Exception as e:
            logger.warning(f"Failed {cfg['name']}: {e}")

    return results


def run_evaluation(args):
    """Main evaluation pipeline."""
    # Log hyperparameters
    logger.info("Hyperparameters from step03:")
    for k, v in get_all_params().items():
        logger.info(f"  {k}: {v}")

    # Load validation CSV (auto-find val_split.csv in directory)
    val_csv = os.path.join(args.val_dir, "val_split.csv")
    if not os.path.exists(val_csv):
        raise FileNotFoundError(f"val_split.csv not found in: {args.val_dir}")

    samples = load_split_csv(val_csv)
    logger.info(f"Loaded {len(samples)} validation samples")

    # Sample per class using step03 settings
    n_per_class = args.n_per_class or EVAL_SETTINGS["n_per_class"]
    seed = args.seed or EVAL_SETTINGS["seed"]
    samples = sample_per_class(samples, n_per_class, seed=seed)
    logger.info(f"Sampled {len(samples)} images ({n_per_class} per class)")

    # Build transforms from step03 configs
    configs = build_transform_configs()
    logger.info(f"Evaluating {len(configs)} transform configurations")

    # Group by tar
    tar_to_samples = defaultdict(list)
    for s in samples:
        tar_to_samples[s.get("tar_path", "")].append(s)

    logger.info(f"Found {len(tar_to_samples)} unique tar files")

    all_results = []
    processed_count = 0
    skipped_tar = 0
    skipped_img = 0

    for tar_path, tar_samples in tqdm(tar_to_samples.items(), desc="Processing"):
        if not tar_path or not os.path.exists(tar_path):
            skipped_tar += 1
            if len(tar_to_samples) <= 5:  # Log first few missing tars
                logger.warning(f"Tar not found: {tar_path}")
            continue

        tar_index = build_tar_index(tar_path)

        with open(tar_path, "rb") as fh:
            for sample in tar_samples:
                uid = sample.get("uid", "")
                # Use 'prefix' column directly if available, otherwise extract from uid
                prefix = sample.get("prefix", "")
                if not prefix and ":" in uid:
                    prefix = uid.split(":")[-1]

                if prefix in tar_index:
                    off, size = tar_index[prefix]
                else:
                    # Fallback: try tif_off/tif_size columns (legacy format)
                    try:
                        off = int(sample.get("tif_off", 0))
                        size = int(sample.get("tif_size", 0))
                    except:
                        skipped_img += 1
                        continue

                if size == 0:
                    skipped_img += 1
                    continue

                try:
                    fh.seek(off)
                    img = tifffile.imread(io.BytesIO(fh.read(size)))
                except Exception as e:
                    skipped_img += 1
                    continue

                img_results = evaluate_image(img, configs)
                for r in img_results:
                    r["uid"] = uid
                    r["label"] = sample.get("label", "")
                all_results.extend(img_results)
                processed_count += 1

    logger.info(
        f"Processed {processed_count} images, skipped {skipped_tar} tars, {skipped_img} images"
    )

    # Check if we have results
    if not all_results:
        logger.error("No images were processed! Check:")
        logger.error("  1. val_split.csv has 'tar_path' column with valid paths")
        logger.error("  2. Tar files are accessible from Colab")
        logger.error("  3. CSV has 'tif_off' and 'tif_size' or matching .pkl index")

        # Print sample CSV row for debugging
        if samples:
            logger.error(f"Sample CSV row keys: {list(samples[0].keys())}")
        raise ValueError("No images processed. See logs above for debugging info.")

    # Create DataFrame
    df = pd.DataFrame(all_results)

    # Summary per transform
    summary = (
        df.groupby(["transform", "type", "category"])
        .agg(
            {
                "LV": ["mean", "std"],
                "HFE": ["mean", "std"],
                "ESSIM": ["mean", "std"],
                "GC": ["mean", "std"],
                "Texture": ["mean", "std"],
                "Shape": ["mean", "std"],
            }
        )
        .round(4)
    )
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()

    # Save
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "all_results.csv"), index=False)
    summary.to_csv(os.path.join(output_dir, "summary.csv"), index=False)

    # Print results
    print_results(summary, len(samples), n_per_class, output_dir)

    return df, summary


def print_results(summary, n_images, n_per_class, output_dir):
    """Print formatted results table."""
    print("\n" + "=" * 80)
    print("TEXTURE & SHAPE SUPPRESSION EVALUATION RESULTS")
    print(f"Images: {n_images} ({n_per_class} per class)")
    print(
        f"Metric params: w={METRIC_PARAMS['window_size']}, r={METRIC_PARAMS['freq_radius']}, k={METRIC_PARAMS['sobel_ksize']}"
    )
    print("=" * 80)

    # Print by category
    categories = [
        ("bilateral", "Bilateral Filter"),
        ("gaussian", "Gaussian Blur"),
        ("nlmeans", "NLMeans Denoising"),
        ("box", "Box Blur"),
        ("median", "Median Filter"),
        ("patch_shuffle", "Patch Shuffle"),
    ]

    for cat_key, cat_name in categories:
        cat_df = summary[summary["category"] == cat_key]
        if len(cat_df) == 0:
            continue

        cat_type = cat_df.iloc[0]["type"]
        print(
            f"\n{cat_name} ({'Texture' if cat_type == 'texture' else 'Shape'} Suppression)"
        )
        print("-" * 70)

        cols = [
            "transform",
            "LV_mean",
            "HFE_mean",
            "Texture_mean",
            "ESSIM_mean",
            "GC_mean",
            "Shape_mean",
        ]
        disp = cat_df[cols].copy()
        disp.columns = ["Transform", "LV", "HFE", "Texture", "ESSIM", "GC", "Shape"]

        sort_col = "Texture" if cat_type == "texture" else "Shape"
        disp = disp.sort_values(sort_col)
        print(disp.to_string(index=False))

    print("\n" + "=" * 80)
    print(f"Results saved to: {output_dir}")
    print("=" * 80 + "\n")


# ==============================================================================
# Arguments
# ==============================================================================
def get_args():
    p = argparse.ArgumentParser("Texture & Shape Suppression Evaluation")
    p.add_argument(
        "--val_dir", type=str, required=True, help="Directory containing val_split.csv"
    )
    p.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save evaluation results",
    )
    p.add_argument(
        "--n_per_class",
        type=int,
        default=None,
        help=f"Images per class (default: {EVAL_SETTINGS['n_per_class']})",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help=f"Random seed (default: {EVAL_SETTINGS['seed']})",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    run_evaluation(args)
