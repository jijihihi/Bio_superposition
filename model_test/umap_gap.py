# ==============================================================================
# UMAP Visualization of CNN GAP Features (L2 normalized)
#
# Loads encoder, extracts L2-normed GAP features, generates UMAP colored by class
#
# Usage:
#   python -m model_test.umap_gap \
#       --ckpt_path /path/to/best_model.pt \
#       --save_dir /path/to/MoCo_seed42 \
#       --shard_root /path/to/wds_shards_tar \
#       --samples_per_class 5000
# ==============================================================================

# python -m model_test.umap_gap \
#     --ckpt_path /home/ubuntu/model-east3/outputs/MoCo_seed45/best_model.pt \
#     --save_dir /home/ubuntu/model-east3/outputs/MoCo_seed45 \
#     --shard_root /home/ubuntu/model-east3/wds_shards_tar \
#     --samples_per_class 5000

import os
import sys
import random
import argparse
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import matplotlib
_IN_COLAB = "google.colab" in sys.modules
if not _IN_COLAB:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import umap
except ImportError:
    print("Installing umap-learn...")
    os.system("pip -q install umap-learn")
    import umap

from sae_project.step02_logging_utils import get_logger, OUT_DIM
from sae_project.step03_data_shards import load_all_sample_refs, build_uid_to_refidx
from sae_project.step04_data_bank import (
    InMemoryTarBank, InMemorySixteenBitDataset,
    seed_worker, collate_skip_none,
)
from sae_project.step05_model_encoder import (
    Encoder, SupMoCoModel, parse_int_list,
    renorm_unit_per_out_channel_, robust_load_state_dict,
)

logger = get_logger("umap_gap")

CLASS_NAMES = {0: "Control", 1: "SNCA", 2: "GBA", 3: "LRRK2"}
CLASS_COLORS = {
    "Control": "#4C72B0",
    "SNCA": "#DD8452",
    "GBA": "#55A868",
    "LRRK2": "#C44E52",
}


# ==============================================================================
# Load split CSV
# ==============================================================================
def load_split_csv(csv_path: str) -> List[str]:
    import csv
    uids = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uids.append(row["uid"])
    return uids


# ==============================================================================
# Feature extraction
# ==============================================================================
@torch.no_grad()
def extract_features(encoder, loader, device, use_bf16=True):
    """Extract L2-normalized GAP features."""
    encoder.eval()
    autocast_kwargs = dict(device_type="cuda", enabled=torch.cuda.is_available())
    if use_bf16 and torch.cuda.is_available():
        autocast_kwargs["dtype"] = torch.bfloat16

    all_feats, all_labels = [], []
    for batch in tqdm(loader, desc="Extracting features", leave=True):
        if batch is None:
            continue
        x, y, *_ = batch
        if x.numel() < 1:
            continue
        x = x.to(device, non_blocking=True).contiguous(
            memory_format=torch.channels_last)
        with torch.amp.autocast(**autocast_kwargs):
            feat = encoder(x)
        feat = F.normalize(feat, dim=1)  # L2 normalization
        all_feats.append(feat.cpu().float().numpy())
        all_labels.append(y.numpy())

    return np.concatenate(all_feats), np.concatenate(all_labels)


# ==============================================================================
# UMAP plot
# ==============================================================================
def plot_umap(coords, labels, title, output_path, 
              point_size=3.0, alpha=0.4, dpi=200, info_text=""):
    """Save UMAP 2D scatter colored by class."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    plot_order = ["Control", "SNCA", "GBA", "LRRK2"]
    for cls_name in plot_order:
        cls_id = [k for k, v in CLASS_NAMES.items() if v == cls_name][0]
        mask = labels == cls_id
        if mask.sum() == 0:
            continue
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            s=point_size, alpha=alpha,
            label=f"{cls_name} (n={mask.sum()})",
            c=CLASS_COLORS[cls_name], edgecolors="none",
        )

    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    if info_text:
        ax.text(0.02, 0.02, info_text, transform=ax.transAxes,
                fontsize=9, verticalalignment="bottom",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))

    ax.legend(loc="upper right", markerscale=3, fontsize=10, framealpha=0.9)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if _IN_COLAB:
        plt.show()
    plt.close(fig)
    logger.info(f"Saved UMAP: {output_path}")


# ==============================================================================
# Main
# ==============================================================================
def get_args():
    p = argparse.ArgumentParser("UMAP of CNN GAP features (L2 normalized)")

    p.add_argument("--ckpt_path", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="",
                   help="Dir with val/test_split.csv (default: ckpt parent dir)")
    p.add_argument("--shard_root", type=str,
                   default="/home/ubuntu/model-east3/wds_shards_tar")
    p.add_argument("--output_dir", type=str, default="")

    # Sampling
    p.add_argument("--samples_per_class", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)

    # Encoder
    p.add_argument("--blocks", type=str, default="2,2,2,3")
    p.add_argument("--dilations", type=str, default="1,1,1,1")
    p.add_argument("--refine_blocks", type=int, default=1)
    p.add_argument("--ckpt_segments", type=int, default=0)

    # UMAP
    p.add_argument("--n_neighbors", type=int, default=15)
    p.add_argument("--min_dist", type=float, default=0.3)
    p.add_argument("--metric", type=str, default="cosine",
                   choices=["euclidean", "cosine", "correlation"])
    p.add_argument("--n_components", type=int, default=2)

    # Data / plot
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--point_size", type=float, default=3.0)
    p.add_argument("--alpha", type=float, default=0.4)

    return p.parse_args()


def main():
    args = get_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = args.save_dir or os.path.dirname(args.ckpt_path)
    model_name = os.path.basename(save_dir)

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(save_dir, "umap_gap")
    os.makedirs(output_dir, exist_ok=True)

    # Load encoder
    logger.info(f"Loading encoder: {args.ckpt_path}")
    blocks = parse_int_list(args.blocks, 4)
    dilations = parse_int_list(args.dilations, 4)
    model = SupMoCoModel(
        embed_dim=512, blocks=blocks, dilations=dilations,
        refine_blocks=args.refine_blocks,
        ckpt_segments=args.ckpt_segments,
        proj_layers=2, proj_hidden=2048,
    )
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    robust_load_state_dict(model, ckpt, strict=False)
    encoder = model.encoder
    encoder.eval().to(device).to(memory_format=torch.channels_last)
    renorm_unit_per_out_channel_(encoder)
    del model

    # Load refs
    logger.info("Loading sample refs...")
    refs = load_all_sample_refs(args.shard_root)
    uid_to_refidx = build_uid_to_refidx(refs)

    # Load val + test UIDs
    val_csv = os.path.join(save_dir, "val_split.csv")
    test_csv = os.path.join(save_dir, "test_split.csv")
    all_uids = []
    if os.path.exists(val_csv):
        all_uids.extend(load_split_csv(val_csv))
    if os.path.exists(test_csv):
        all_uids.extend(load_split_csv(test_csv))
    if not all_uids:
        raise FileNotFoundError(f"No val/test CSVs in {save_dir}")

    ref_indices = [uid_to_refidx[u] for u in all_uids if u in uid_to_refidx]

    # Balanced subsample
    from collections import defaultdict
    spc = args.samples_per_class
    if spc > 0:
        rng = random.Random(args.seed)
        class_to_idx = defaultdict(list)
        for idx in ref_indices:
            class_to_idx[refs[idx].label].append(idx)

        sampled = []
        for cls in sorted(class_to_idx.keys()):
            idxs = class_to_idx[cls]
            rng.shuffle(idxs)
            n = min(spc, len(idxs))
            sampled.extend(idxs[:n])
            logger.info(f"  {CLASS_NAMES[cls]}: {n}/{len(idxs)}")
        ref_indices = sampled
    
    logger.info(f"Total samples: {len(ref_indices)}")

    # Dataloader
    bank = InMemoryTarBank(refs, ref_indices, args.img_size)
    ib = list(range(len(ref_indices)))
    ds = InMemorySixteenBitDataset(bank, ib, args.img_size, augment=False)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        worker_init_fn=seed_worker, collate_fn=collate_skip_none)

    # Extract features
    logger.info("Extracting L2-normalized GAP features...")
    X, y = extract_features(encoder, loader, device)
    logger.info(f"Features: {X.shape}")

    del encoder, loader, bank, ds
    torch.cuda.empty_cache()

    # Run UMAP
    logger.info(f"Running UMAP (n_neighbors={args.n_neighbors}, "
                f"min_dist={args.min_dist}, metric={args.metric})...")
    reducer = umap.UMAP(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        n_components=args.n_components,
        random_state=args.seed,
        verbose=True,
    )
    coords = reducer.fit_transform(X)
    logger.info(f"UMAP done: {coords.shape}")

    # Save coordinates
    npz_path = os.path.join(output_dir,
        f"umap_coords_{model_name}_nn{args.n_neighbors}.npz")
    np.savez_compressed(npz_path, coords=coords, labels=y,
                        n_neighbors=args.n_neighbors,
                        min_dist=args.min_dist, metric=args.metric)
    logger.info(f"Saved coords: {npz_path}")

    # Plot
    title = (f"UMAP – {model_name} (GAP + L2 norm)\n"
             f"n_neighbors={args.n_neighbors}, min_dist={args.min_dist}")
    info = (f"Samples: {X.shape[0]}\n"
            f"Features: {X.shape[1]}D (GAP + L2 norm)\n"
            f"Metric: {args.metric}")

    png_path = os.path.join(output_dir,
        f"umap_{model_name}_nn{args.n_neighbors}.png")
    plot_umap(coords, y, title, png_path,
              point_size=args.point_size, alpha=args.alpha,
              dpi=args.dpi, info_text=info)

    logger.info("Done!")


if __name__ == "__main__":
    main()
