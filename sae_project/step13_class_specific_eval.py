# ==============================================================================
# Step 13: Class-Specific Concept Evaluation
# - Load trained SAE and class-wise GAP means CSV
# - Filter concepts where max(GAP)/min(GAP) >= threshold (class-specific concepts)
# - Extract image-level features using only selected concepts
# - Train linear classifier and evaluate on test set
# ==============================================================================

import argparse
import csv
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from sae_project.step01_configs import get_args, resolve_paths
from run_CNN.logging_utils import get_logger
from run_CNN.data_shards import (build_uid_to_refidx,
                                            load_all_sample_refs)
from run_CNN.data_bank import (InMemorySixteenBitDataset,
                                          InMemoryTarBank, collate_skip_none,
                                          load_split_csv, seed_worker)
from run_CNN.model_encoder import (SupMoCoModel, parse_int_list,
                                              renorm_unit_per_out_channel_)
from sae_project.step06_gated_sae import GatedSAE

logger = get_logger("class_specific_eval")


# ==============================================================================
# Argument Parser
# ==============================================================================
def get_step13_args():
    p = argparse.ArgumentParser(
        description="Evaluate class-specific concepts from trained Gated SAE"
    )

    # Required inputs
    p.add_argument(
        "--sae_ckpt",
        type=str,
        required=True,
        help="Path to trained SAE checkpoint (.pt file)",
    )
    p.add_argument(
        "--gap_csv",
        type=str,
        required=True,
        help="Path to class-wise GAP means CSV (output from step09)",
    )

    # Concept selection threshold
    p.add_argument(
        "--filter_mode",
        type=str,
        default="gini",
        choices=["ratio", "entropy", "gini"],
        help="Filter mode: 'ratio' (max/min GAP), 'entropy' (Shannon entropy), or 'gini' (Gini impurity)",
    )
    p.add_argument(
        "--min_ratio",
        type=float,
        default=3.0,
        help="[ratio mode] Minimum max(GAP)/min(GAP) ratio for selection",
    )
    p.add_argument(
        "--max_entropy",
        type=float,
        default=1.0,
        help="[entropy mode] Maximum Shannon entropy for selection (lower = more class-specific)",
    )
    p.add_argument(
        "--max_gini_impurity",
        type=float,
        default=0.5,
        help="[gini mode] Maximum Gini impurity for selection (lower = more class-specific, 0=pure, 0.75=uniform for 4 classes)",
    )
    p.add_argument(
        "--min_active_images",
        type=int,
        default=10,
        help="Minimum number of images where concept must be active (GAP>0) in at least one class",
    )
    p.add_argument(
        "--eps",
        type=float,
        default=1e-8,
        help="Small value added to min(GAP) to avoid division by zero",
    )
    p.add_argument(
        "--dead_threshold",
        type=float,
        default=0.0,
        help="If >0, override CSV is_alive using SAE usage_ema >= this threshold. If 0, use CSV is_alive column.",
    )

    # Data paths (inherit from step01_configs defaults)
    p.add_argument("--shard_root", type=str, default=None)
    p.add_argument(
        "--save_dir",
        type=str,
        default="/content/drive/MyDrive/Final_paper/Model_MoCoXBM_PlateLP_LRRK2_L2 norm_hidden2048_resume_bias=True_clean_image",
    )
    p.add_argument(
        "--model_state_path",
        type=str,
        default="/content/drive/MyDrive/Final_paper/Model_MoCoXBM_PlateLP_LRRK2_L2 norm_hidden2048_resume_bias=True_clean_image/best_model.pt",
    )

    # Encoder architecture (must match training)
    p.add_argument("--blocks", type=str, default="2,2,2,3")
    p.add_argument("--dilations", type=str, default="1,1,1,1")
    p.add_argument("--refine_blocks", type=int, default=1)
    p.add_argument("--ckpt_segments", type=int, default=0)
    p.add_argument("--embed_dim", type=int, default=512)
    p.add_argument("--proj_layers", type=int, default=2)
    p.add_argument("--proj_hidden", type=int, default=2048)
    p.add_argument("--proj_bn", action="store_true")
    p.add_argument("--proj_dropout", type=float, default=0.0)

    # Dataset settings
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)

    # Linear probe settings
    p.add_argument("--probe_epochs", type=int, default=30)
    p.add_argument("--probe_lr", type=float, default=0.1)
    p.add_argument("--probe_batch_size", type=int, default=256)

    # Output
    p.add_argument(
        "--output_csv",
        type=str,
        default="",
        help="Path to save results CSV (default: same dir as gap_csv)",
    )

    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


# ==============================================================================
# Linear Probe (same as step09)
# ==============================================================================
class LinearProbe(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.net = nn.Linear(d_in, d_out, bias=False)

    def forward(self, x):
        return self.net(x)


def train_linear_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int = 4,
    epochs: int = 30,
    lr: float = 0.1,
    batch_size: int = 256,
    device: torch.device = None,
) -> dict:
    """Train linear probe with balanced sampling."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_features = X_train.shape[1]
    if n_features == 0:
        return {"train_acc": 0.0, "test_acc": 0.0, "n_concepts": 0}

    # Class indices for balanced sampling
    rng = np.random.default_rng(42)
    class_indices = {c: np.where(y_train == c)[0] for c in range(num_classes)}
    min_class_count = min(len(v) for v in class_indices.values())
    samples_per_class = min_class_count

    # Train probe
    probe = LinearProbe(n_features, num_classes).to(device)
    optimizer = optim.SGD(probe.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    probe.train()
    for epoch in range(epochs):
        # Balanced sampling
        epoch_indices = []
        for c in range(num_classes):
            c_idx = class_indices[c]
            sampled = rng.choice(c_idx, size=samples_per_class, replace=False)
            epoch_indices.extend(sampled.tolist())
        epoch_indices = np.array(epoch_indices)
        rng.shuffle(epoch_indices)

        # Mini-batch training
        for s in range(0, len(epoch_indices), batch_size):
            ii = epoch_indices[s : s + batch_size]
            xb = torch.from_numpy(X_train[ii]).float().to(device)
            yb = torch.from_numpy(y_train[ii]).long().to(device)

            optimizer.zero_grad()
            logits = probe(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    # Evaluate
    probe.eval()
    with torch.no_grad():
        # Train accuracy
        eval_indices = []
        for c in range(num_classes):
            c_idx = class_indices[c]
            sampled = rng.choice(
                c_idx, size=min(samples_per_class, len(c_idx)), replace=False
            )
            eval_indices.extend(sampled.tolist())
        X_tr = torch.from_numpy(X_train[eval_indices]).float().to(device)
        y_tr = torch.from_numpy(y_train[eval_indices]).long().to(device)
        train_pred = probe(X_tr).argmax(dim=1)
        train_acc = (train_pred == y_tr).float().mean().item()

        # Test accuracy
        X_te = torch.from_numpy(X_test).float().to(device)
        y_te = torch.from_numpy(y_test).long().to(device)
        test_pred = probe(X_te).argmax(dim=1)
        test_acc = (test_pred == y_te).float().mean().item()

        # Per-class accuracy
        per_class_acc = {}
        for c in range(num_classes):
            mask = y_te == c
            if mask.sum() > 0:
                per_class_acc[c] = (test_pred[mask] == c).float().mean().item()

    return {
        "train_acc": float(train_acc),
        "test_acc": float(test_acc),
        "n_concepts": n_features,
        "per_class_acc": per_class_acc,
    }


# ==============================================================================
# Load and Filter Concepts
# ==============================================================================
def load_and_filter_concepts(
    gap_csv_path: str,
    filter_mode: str,
    min_ratio: float,
    max_entropy: float,
    max_gini_impurity: float,
    min_active_images: int,
    eps: float,
    alive_mask: np.ndarray = None,
) -> tuple:
    """
    Load GAP means CSV and filter class-specific concepts.

    Args:
        gap_csv_path: Path to CSV with columns [concept_id, is_alive, Control, SNCA, GBA, LRRK2, n_Control, n_SNCA, n_GBA, n_LRRK2, entropy, ...]
        filter_mode: 'ratio', 'entropy', or 'gini'
        min_ratio: Minimum max(GAP)/min(GAP) ratio for ratio mode
        max_entropy: Maximum Shannon entropy for entropy mode
        max_gini_impurity: Maximum Gini impurity for gini mode (0=pure, 0.75=uniform)
        min_active_images: Minimum number of active images in at least one class
        eps: Small value to add to min to avoid division by zero
        alive_mask: Optional numpy array of shape (d_sae,) with True for alive neurons.
                    If provided, overrides CSV is_alive column.

    Returns:
        selected_indices: List of concept indices that pass the filter
        concept_info: List of dicts with concept details
    """
    selected_indices = []
    concept_info = []

    class_cols = ["Control", "SNCA", "GBA", "LRRK2"]
    count_cols = ["n_Control", "n_SNCA", "n_GBA", "n_LRRK2"]

    with open(gap_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            concept_id = int(row["concept_id"])

            # Determine if alive: use alive_mask if provided, else CSV
            if alive_mask is not None:
                is_alive = 1 if alive_mask[concept_id] else 0
            else:
                is_alive = int(row["is_alive"])

            # Skip dead concepts
            if is_alive == 0:
                continue

            # Get GAP values for each class
            gaps = [float(row[c]) for c in class_cols]

            # Get active image counts per class (new columns from step09)
            # Fallback to checking if columns exist
            if count_cols[0] in row:
                active_counts = [int(row[c]) for c in count_cols]
                max_active = max(active_counts)
            else:
                # Old CSV format: use max_gap as proxy
                max_active = 999 if max(gaps) > 0.01 else 0

            # Calculate metrics
            max_gap = max(gaps)
            min_gap = min(gaps)
            ratio = max_gap / (min_gap + eps)

            # Get entropy from CSV (already computed in step09)
            entropy = float(row.get("entropy", 2.0))

            # Compute Gini impurity: 1 - sum(p_i^2)
            # For 4 classes: 0 = pure (only one class), 0.75 = uniform
            total_gap = sum(gaps) + eps
            probs = [g / total_gap for g in gaps]
            gini_impurity = 1.0 - sum(p * p for p in probs)

            # Skip concepts not activated in enough images
            if max_active < min_active_images:
                continue

            # Filter based on mode
            if filter_mode == "ratio":
                passed = ratio >= min_ratio
            elif filter_mode == "gini":
                passed = gini_impurity <= max_gini_impurity
            else:  # entropy
                passed = entropy <= max_entropy

            if passed:
                max_class = class_cols[np.argmax(gaps)]
                selected_indices.append(concept_id)
                concept_info.append(
                    {
                        "concept_id": concept_id,
                        "max_class": max_class,
                        "max_gap": max_gap,
                        "min_gap": min_gap,
                        "ratio": ratio,
                        "entropy": entropy,
                        "gini_impurity": gini_impurity,
                        "gaps": gaps,
                        "active_counts": (
                            active_counts if count_cols[0] in row else None
                        ),
                        "max_active": max_active,
                    }
                )

    return selected_indices, concept_info


# ==============================================================================
# Extract Features with Selected Concepts Only
# ==============================================================================
@torch.no_grad()
def extract_features_selected_concepts(
    encoder,
    sae: GatedSAE,
    loader: DataLoader,
    device: torch.device,
    which_layer: str,
    selected_indices: list,
) -> tuple:
    """
    Extract image-level features using only selected concepts.

    Returns:
        X: (N, n_selected) feature matrix
        y: (N,) labels
    """
    encoder.eval()
    sae.eval()

    # Create mask for selected concepts
    selected_mask = torch.zeros(sae.d_sae, dtype=torch.bool, device=device)
    for idx in selected_indices:
        selected_mask[idx] = True

    autocast_kwargs = dict(device_type="cuda", enabled=torch.cuda.is_available())
    if torch.cuda.is_available():
        autocast_kwargs["dtype"] = torch.bfloat16

    X_list, y_list = [], []

    for batch in tqdm(loader, desc="Extracting features", leave=False):
        if batch is None:
            continue
        x_cpu, y_cpu, *_ = batch
        if x_cpu.numel() < 1:
            continue

        x = x_cpu.to(device, non_blocking=True).contiguous(
            memory_format=torch.channels_last
        )
        y = y_cpu

        with torch.amp.autocast(**autocast_kwargs):
            fmap = encoder.forward_feature_maps(x, which=which_layer)

        curr_batch_size = fmap.size(0)

        # GAP-scalar normalization (match training)
        gap = fmap.mean(dim=(2, 3))
        gap_norm = (
            gap.norm(dim=1, keepdim=True)
            .view(curr_batch_size, 1, 1, 1)
            .clamp_min(1e-12)
        )
        fmap = fmap / gap_norm

        fmap = fmap.permute(0, 2, 3, 1).contiguous()
        C = fmap.shape[-1]

        flat_tokens = fmap.view(-1, C)
        flat_tokens = flat_tokens - flat_tokens.mean(dim=0, keepdim=True)
        flat_tokens = F.normalize(flat_tokens, dim=1, eps=1e-12)

        # Process in chunks
        token_batch_size = 8192
        num_flat_tokens = flat_tokens.size(0)

        acts_list = []
        for start in range(0, num_flat_tokens, token_batch_size):
            end = min(start + token_batch_size, num_flat_tokens)
            chunk = flat_tokens[start:end]
            with torch.amp.autocast(**autocast_kwargs):
                _, chunk_acts, _, _, _ = sae(chunk)
            acts_list.append(chunk_acts)
        acts = torch.cat(acts_list, dim=0)
        del acts_list

        # Select only the specified concepts
        acts = acts.float()
        acts = acts[:, selected_mask]  # (N*H*W, n_selected)

        # Reshape and pool
        H_W = fmap.shape[1] * fmap.shape[2]
        acts = acts.view(curr_batch_size, H_W, -1)
        pooled = acts.mean(dim=1)  # (B, n_selected)
        pooled = F.normalize(pooled, dim=1)

        X_list.append(pooled.cpu().numpy())
        y_list.extend(y.tolist())

    if len(X_list) == 0:
        return np.zeros((0, len(selected_indices)), dtype=np.float32), np.zeros(
            0, dtype=np.int64
        )

    X = np.concatenate(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)

    return X, y


# ==============================================================================
# DataLoader Helper
# ==============================================================================
def make_eval_loader(args, refs, uid_to_refidx, split_csv_path, batch_size):
    if not os.path.exists(split_csv_path):
        return None

    uids = load_split_csv(split_csv_path)
    missing = [u for u in uids if u not in uid_to_refidx]
    if len(missing) > 0:
        raise RuntimeError(f"Missing UIDs: {missing[:5]}")
    refidx = [uid_to_refidx[u] for u in uids]

    bank = InMemoryTarBank(refs, refidx, args.img_size)
    ib = list(range(len(refidx)))
    ds = InMemorySixteenBitDataset(bank, ib, args.img_size, augment=False)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        collate_fn=collate_skip_none,
    )
    return loader


# ==============================================================================
# Main
# ==============================================================================
def main():
    args = get_step13_args()

    # Resolve shard_root
    if args.shard_root is None:
        from run_CNN.logging_utils import DEFAULT_SHARD_ROOT

        args.shard_root = DEFAULT_SHARD_ROOT

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ==== Step 1: Load and filter concepts ====
    logger.info(f"\n{'='*60}")
    logger.info("Step 1: Loading and filtering class-specific concepts")
    logger.info(f"GAP CSV: {args.gap_csv}")
    logger.info(f"Filter mode: {args.filter_mode}")
    if args.filter_mode == "ratio":
        logger.info(f"Minimum ratio: {args.min_ratio}")
    elif args.filter_mode == "gini":
        logger.info(f"Maximum Gini impurity: {args.max_gini_impurity}")
    else:
        logger.info(f"Maximum entropy: {args.max_entropy}")
    logger.info(f"Minimum active images: {args.min_active_images}")

    # Generate alive_mask from SAE if dead_threshold > 0
    alive_mask = None
    if args.dead_threshold > 0:
        logger.info(f"Using custom dead_threshold: {args.dead_threshold}")
        logger.info(f"Loading SAE to get usage_ema from: {args.sae_ckpt}")
        ckpt_for_mask = torch.load(
            args.sae_ckpt, map_location="cpu", weights_only=False
        )
        usage_ema = ckpt_for_mask.get("usage_ema", None)
        if usage_ema is None and "state_dict" in ckpt_for_mask:
            usage_ema = ckpt_for_mask["state_dict"].get("usage_ema", None)
        if usage_ema is not None:
            if isinstance(usage_ema, torch.Tensor):
                usage_ema = usage_ema.cpu().numpy()
            alive_mask = usage_ema >= args.dead_threshold
            n_alive = alive_mask.sum()
            logger.info(
                f"Custom alive mask: {n_alive} / {len(alive_mask)} neurons alive (threshold={args.dead_threshold})"
            )
        else:
            logger.warning(
                "Could not find usage_ema in SAE checkpoint. Using CSV is_alive column instead."
            )
        del ckpt_for_mask

    selected_indices, concept_info = load_and_filter_concepts(
        args.gap_csv,
        args.filter_mode,
        args.min_ratio,
        args.max_entropy,
        args.max_gini_impurity,
        args.min_active_images,
        args.eps,
        alive_mask,
    )

    logger.info(f"Selected {len(selected_indices)} class-specific concepts")

    if len(selected_indices) == 0:
        if args.filter_mode == "ratio":
            logger.error(
                "No concepts passed the filter! Try lowering --min_ratio or --min_active_images"
            )
        elif args.filter_mode == "gini":
            logger.error(
                "No concepts passed the filter! Try increasing --max_gini_impurity or lowering --min_active_images"
            )
        else:
            logger.error(
                "No concepts passed the filter! Try increasing --max_entropy or lowering --min_active_images"
            )
        return

    # Print summary by max_class
    class_counts = {"Control": 0, "SNCA": 0, "GBA": 0, "LRRK2": 0}
    for info in concept_info:
        class_counts[info["max_class"]] += 1

    logger.info(f"Concepts per max-class: {class_counts}")

    # ==== Step 2: Load SAE ====
    logger.info(f"\n{'='*60}")
    logger.info("Step 2: Loading SAE checkpoint")
    logger.info(f"SAE: {args.sae_ckpt}")

    ckpt = torch.load(args.sae_ckpt, map_location="cpu", weights_only=False)
    ckpt_args = ckpt["args"]

    sae = GatedSAE(
        d_in=ckpt_args.get("d_in", 512),
        d_sae=ckpt_args.get("d_sae", 4096),
        tie_weights=ckpt_args.get("tie_gate_weights", False),
        aux_k=ckpt_args.get("aux_k", 32),
    )
    sae.load_state_dict(ckpt["sae"])
    sae.to(device).eval()

    which_layer = ckpt_args.get("which_layer", "refine_out")
    logger.info(f"SAE: d_in={sae.d_in}, d_sae={sae.d_sae}, layer={which_layer}")

    # ==== Step 3: Load encoder ====
    logger.info(f"\n{'='*60}")
    logger.info("Step 3: Loading encoder")

    blocks = parse_int_list(args.blocks, 4)
    dilations = parse_int_list(args.dilations, 4)

    model = SupMoCoModel(
        embed_dim=args.embed_dim,
        blocks=blocks,
        dilations=dilations,
        refine_blocks=args.refine_blocks,
        ckpt_segments=args.ckpt_segments,
        proj_layers=args.proj_layers,
        proj_hidden=args.proj_hidden,
        proj_bn=args.proj_bn,
        proj_dropout=args.proj_dropout,
    )

    sd = torch.load(args.model_state_path, map_location="cpu", weights_only=False)
    from run_CNN.model_encoder import robust_load_state_dict

    robust_load_state_dict(model, sd, strict=True)
    encoder = model.encoder
    renorm_unit_per_out_channel_(encoder)
    encoder.to(device).eval()

    del model, sd

    # ==== Step 4: Load data ====
    logger.info(f"\n{'='*60}")
    logger.info("Step 4: Loading data")

    refs = load_all_sample_refs(args.shard_root)
    uid_to_refidx = build_uid_to_refidx(refs)

    train_csv = os.path.join(args.save_dir, "train_split.csv")
    test_csv = os.path.join(args.save_dir, "test_split.csv")

    train_loader = make_eval_loader(
        args, refs, uid_to_refidx, train_csv, args.batch_size
    )
    test_loader = make_eval_loader(args, refs, uid_to_refidx, test_csv, args.batch_size)

    if train_loader is None or test_loader is None:
        logger.error("Could not load train/test splits!")
        return

    # ==== Step 5: Extract features ====
    logger.info(f"\n{'='*60}")
    logger.info("Step 5: Extracting features (selected concepts only)")

    X_train, y_train = extract_features_selected_concepts(
        encoder, sae, train_loader, device, which_layer, selected_indices
    )
    logger.info(f"Train: {X_train.shape[0]} images, {X_train.shape[1]} concepts")

    X_test, y_test = extract_features_selected_concepts(
        encoder, sae, test_loader, device, which_layer, selected_indices
    )
    logger.info(f"Test: {X_test.shape[0]} images, {X_test.shape[1]} concepts")

    # ==== Step 6: Train linear probe ====
    logger.info(f"\n{'='*60}")
    logger.info("Step 6: Training linear probe")

    results = train_linear_probe(
        X_train,
        y_train,
        X_test,
        y_test,
        num_classes=4,
        epochs=args.probe_epochs,
        lr=args.probe_lr,
        batch_size=args.probe_batch_size,
        device=device,
    )

    logger.info(f"\n{'='*60}")
    logger.info("RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Filter mode: {args.filter_mode}")
    if args.filter_mode == "ratio":
        logger.info(f"Min ratio threshold: {args.min_ratio}")
    else:
        logger.info(f"Max entropy threshold: {args.max_entropy}")
    logger.info(f"Selected concepts: {len(selected_indices)}")
    logger.info(f"Train accuracy: {results['train_acc']:.4f}")
    logger.info(f"Test accuracy:  {results['test_acc']:.4f}")
    logger.info(f"Per-class accuracy: {results['per_class_acc']}")

    # ==== Save results ====
    if args.output_csv == "":
        output_dir = os.path.dirname(args.gap_csv)
        if args.filter_mode == "ratio":
            filename = f"class_specific_eval_{which_layer}_ratio{args.min_ratio}.csv"
        elif args.filter_mode == "gini":
            filename = (
                f"class_specific_eval_{which_layer}_gini{args.max_gini_impurity}.csv"
            )
        else:
            filename = (
                f"class_specific_eval_{which_layer}_entropy{args.max_entropy}.csv"
            )
        output_csv = os.path.join(output_dir, filename)
    else:
        output_csv = args.output_csv

    # If output_csv is a directory, append filename
    if os.path.isdir(output_csv):
        if args.filter_mode == "ratio":
            filename = f"class_specific_eval_{which_layer}_ratio{args.min_ratio}.csv"
        elif args.filter_mode == "gini":
            filename = (
                f"class_specific_eval_{which_layer}_gini{args.max_gini_impurity}.csv"
            )
        else:
            filename = (
                f"class_specific_eval_{which_layer}_entropy{args.max_entropy}.csv"
            )
        output_csv = os.path.join(output_csv, filename)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "gap_csv",
                "sae_ckpt",
                "filter_mode",
                "threshold",
                "n_selected_concepts",
                "n_control",
                "n_snca",
                "n_gba",
                "n_lrrk2",
                "train_acc",
                "test_acc",
                "acc_control",
                "acc_snca",
                "acc_gba",
                "acc_lrrk2",
            ]
        )
        if args.filter_mode == "ratio":
            threshold = args.min_ratio
        elif args.filter_mode == "gini":
            threshold = args.max_gini_impurity
        else:
            threshold = args.max_entropy
        w.writerow(
            [
                os.path.basename(args.gap_csv),
                os.path.basename(args.sae_ckpt),
                args.filter_mode,
                threshold,
                len(selected_indices),
                class_counts["Control"],
                class_counts["SNCA"],
                class_counts["GBA"],
                class_counts["LRRK2"],
                f"{results['train_acc']:.4f}",
                f"{results['test_acc']:.4f}",
                f"{results['per_class_acc'].get(0, 0):.4f}",
                f"{results['per_class_acc'].get(1, 0):.4f}",
                f"{results['per_class_acc'].get(2, 0):.4f}",
                f"{results['per_class_acc'].get(3, 0):.4f}",
            ]
        )

    logger.info(f"\nSaved results to: {output_csv}")

    # Print selected concept details
    logger.info(f"\n{'='*60}")
    logger.info("Selected Concept Details (top 20 by ratio)")
    logger.info(f"{'='*60}")
    sorted_info = sorted(concept_info, key=lambda x: x["ratio"], reverse=True)[:20]
    for info in sorted_info:
        logger.info(
            f"Concept {info['concept_id']:4d}: max_class={info['max_class']:8s} "
            f"ratio={info['ratio']:.2f} gaps={[f'{g:.4f}' for g in info['gaps']]}"
        )


if __name__ == "__main__":
    main()
