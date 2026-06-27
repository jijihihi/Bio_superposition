#!/bin/bash
# ==============================================================================
# Full Data Preprocessing Pipeline
# 01. MIP Generation -> 02. QC & Filtering -> 03. Cropping -> 04. Patch QC
# 
# Usage:
#   bash preprocessing.sh [OUTPUT_DIR]
# Example:
#   bash preprocessing.sh "D:/From_C_drive/code_refactoring_bash"
# ==============================================================================

set -e

# 1. 절대경로 앵커링: 어디서 실행하든 스크립트가 위치한 폴더로 안전하게 이동
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE}")" && pwd)"

VENV_DIR=""
if [ -d "$SCRIPT_DIR/.venv" ]; then
    VENV_DIR="$SCRIPT_DIR/.venv"
elif [ -d "$SCRIPT_DIR/../../.venv" ]; then
    VENV_DIR="$SCRIPT_DIR/../../.venv"
fi

# 2. 가상환경 내부 파이썬 알맹이를 직접 지정 (윈도우/맥/리눅스 자동 호환)
if [ -n "$VENV_DIR" ]; then
    if [ -d "$VENV_DIR/Scripts" ]; then
        VENV_PYTHON="$VENV_DIR/Scripts/python"  # Windows
    elif [ -d "$VENV_DIR/bin" ]; then
        VENV_PYTHON="$VENV_DIR/bin/python"      # Linux / macOS
    fi
    echo "[*] Found Virtual Environment Python: $VENV_PYTHON"
else
    echo "[!] WARNING: Virtual environment not found. Using system default Python."
    VENV_PYTHON="python"
fi

# =========================================================
# 3. 텐서플로우 버전 및 환경 정상 검증
# =========================================================
echo "[*] Checking TensorFlow installation..."
$VENV_PYTHON -c "import tensorflow as tf; print('TensorFlow Version:', tf.__version__)"

$VENV_PYTHON -m pip install -r "${SCRIPT_DIR}/../envs/local_requirements.txt"

cd "${SCRIPT_DIR}/../data_preprocessing"

BASE_OUT="${1:-../Output}"
MIP_DIR="${BASE_OUT}/MIP"
MIP_QC_DIR="${BASE_OUT}/MIP_QC_Output"
PATCH_DIR="${BASE_OUT}/patches"
CYTOX_DIR="${BASE_OUT}/MIP_cytox"

echo "Output directory is set to: $BASE_OUT"


echo "============================================================"
echo "Step 1: Generating MIP (Maximum Intensity Projection)"
echo "============================================================"
$VENV_PYTHON 01_MIP_generation.py \
    --output_dir "${MIP_DIR}" \
    --cytox_dir "${CYTOX_DIR}"

echo "============================================================"
echo "Step 2a: Calculating QC Metrics for Large Images"
echo "============================================================"
$VENV_PYTHON 02a_large_image_metrics.py \
    --input_dir "${MIP_DIR}" \
    --output_dir "${MIP_QC_DIR}"

echo "============================================================"
echo "Step 2b: Filtering Low-Quality Large Images"
echo "============================================================"
$VENV_PYTHON 02b_large_image_filtering.py \
    --root_dir "${MIP_DIR}" \
    --csv_path "${MIP_QC_DIR}/qc_metrics_raw_per_channel.csv" \
    --reject_dir_name "Rejected_Images"

echo "============================================================"
echo "Step 3: Cropping Images into Patches"
echo "============================================================"
$VENV_PYTHON 03_patch_cropping.py \
    --input_dir "${MIP_DIR}" \
    --output_dir "${PATCH_DIR}" \
    --patch_size 128 \
    --overlap 0

echo "============================================================"
echo "Step 4: Quality Control for Patches"
echo "============================================================"
$VENV_PYTHON 04_patch_QC.py \
    --input_dir "${PATCH_DIR}" \
    --output_dir "${PATCH_DIR}/Rejected_cropped_image"

echo "============================================================"
echo "Step 5: StarDist Nuclear Segmentation on Large MIP Images"
echo "============================================================"
cd ../dead_cell_rate
$VENV_PYTHON step01_stardist_nuclearmasking.py \
    --input_dir "${MIP_DIR}" \
    --output_dir "${BASE_OUT}/MIP_nucleus_segmentation"

echo "============================================================"
echo "Step 6: Cropping Masks and Cytox Channels for QC Patches"
echo "============================================================"
$VENV_PYTHON step02_stardist_cytox_cropper.py \
    --qc_passed_dir "${PATCH_DIR}" \
    --mask_dir "${BASE_OUT}/MIP_nucleus_segmentation" \
    --cytox_dir "${CYTOX_DIR}" \
    --output_dir "${BASE_OUT}/CellDeath_QC_patches"

echo "============================================================"
echo "Step 7: Calculating Cell Death Rate per Image"
echo "============================================================"
$VENV_PYTHON step03_celldeathrate_perimage.py \
    --data_dir "${BASE_OUT}/CellDeath_QC_patches" \
    --output_csv "${BASE_OUT}/CellDeath_QC_patches/per_image_celldeath_rate.csv"

echo "============================================================"
echo "Step 8: Packaging Patches into WebDataset Tar Shards"
echo "============================================================"
cd ../data_preprocessing
$VENV_PYTHON step08_create_tar_shards.py \
    --input_dir "${PATCH_DIR}" \
    --output_dir "${BASE_OUT}/wds_shards_tar"

echo "============================================================"
echo "🎉 Full Pipeline Completed Successfully!"
echo "Final patches are located in: ${PATCH_DIR}"
echo "Final cell death rates are saved in: ${BASE_OUT}/CellDeath_QC_patches/per_image_celldeath_rate.csv"
echo "Tar Shards for CNN training are in: ${BASE_OUT}/wds_shards_tar"
echo "============================================================"
