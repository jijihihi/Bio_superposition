#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pointwise Top-K Sparse Autoencoder (SAE) for CNN feature maps
- Targets: Encoder.stage5 output (before refine), Encoder.refine output (before GAP)
- Top-K sparsity per token (k=5)
- Dictionary size: 4096 for d_in=512
- Neuron resampling (dead features)
- Decoder weight row norm fixed to 1
- Uses your saved model_q weights (best_model.pt / last_model.pt) that contain encoder+projector
- Reuses your RAM tar bank + split CSV if available

Example:
python sae_train_cnn_featuremaps.py \
  --shard_root /content/wds_shards \
  --save_dir /content/drive/MyDrive/Final_paper/Model_MoCoXBM_PlateLP_LRRK2_L2\ norm_hidden2048_resume \
  --model_state_path /content/drive/MyDrive/Final_paper/Model_MoCoXBM_PlateLP_LRRK2_L2\ norm_hidden2048_resume/best_model.pt \
  --which_layer refine_out \
  --d_sae 4096 --k 5 --tokens_per_image 1024 --batch_size 32 --epochs 10 \
  --use_bf16
"""

import argparse
import csv
import glob
import io
import logging
import os
import pickle
import random
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
from tqdm.auto import tqdm

# tif decoder
try:
    import tifffile
except Exception:
    raise RuntimeError("tifffile not installed. pip install tifffile")

# ==============================================================================
# Logging
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("SAE_CNN_FMAP")


# ==============================================================================
# Constants / mappings (same as your training code)
# ==============================================================================
DEFAULT_SHARD_ROOT = "/content/wds_shards"

LINE_FOLDERS = [
    "Control_C4", "Control_C18", "Control_C19",
    "SNCA", "GBA", "LRRK2"
]

SUPERCLASS_MAP = {
    "Control_C4":  "Control",
    "Control_C18": "Control",
    "Control_C19": "Control",
    "SNCA":        "SNCA",
    "GBA":         "GBA",
    "LRRK2":       "LRRK2",
}
CLASS_TO_LABEL = {"Control": 0, "SNCA": 1, "GBA": 2, "LRRK2": 3}

PLATE_DIR_RE = re.compile(r"plate=(\d{6})")


# ==============================================================================
# Utils
# ==============================================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)

def validate_uint16_rgb_128(img: np.ndarray, img_size: int):
    if img is None:
        raise ValueError("decoded None")
    if img.dtype != np.uint16:
        raise ValueError(f"dtype must be uint16, got {img.dtype}")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"shape must be HxWx3, got {img.shape}")
    h, w = img.shape[:2]
    if (h, w) != (img_size, img_size):
        raise ValueError(f"size must be {(img_size, img_size)}, got {(h, w)}")

