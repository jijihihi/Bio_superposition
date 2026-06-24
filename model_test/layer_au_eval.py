# ==============================================================================
# Layer-wise Linear Classification + Alignment & Uniformity Evaluation
#
# Purpose:
#   For each CNN layer × seed, compute:
#     1) Linear classification accuracy (SGD linear probe, train/test split)
#     2) Alignment (Wang & Isola 2020): E[‖f(x)−f(x⁺)‖²], same-class pairs
#     3) Uniformity: log E[e^{-2‖f(x)−f(y)‖²}], all pairs
#
#   Uses pre-extracted NPZ cache files (cnn_gap_{layer}_all.npz)
#   with keys: features (N, d), labels (N,), uids (N,)
#
# Output:
#   {output_dir}/layer_au_results.csv
#   {output_dir}/layer_au_results.json
#
# Usage:
#   python -m model_test.layer_au_eval \
#       --features_cache /path/to/cnn_gap_stage5_out_all.npz \
#       --split_dir /path/to/MoCo_seed87 \
#       --layer_name stage5_out \
#       --seed_name 87 \
#       --gap_l2_norm \
#       --output_dir /path/to/output
# ==============================================================================

import argparse
import csv
import json
import logging
import os
import random
import sys
from typing import Dict, Tuple

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

_IN_COLAB = ("google.colab" in sys.modules) or os.path.isdir("/content")
if not _IN_COLAB:
    matplotlib.use("Agg")

from sae_project.step02_logging_utils import get_logger

logger = get_logger("layer_au_eval")

CLASS_NAMES = {0: "Control", 1: "SNCA", 2: "GBA", 3: "LRRK2"}
NUM_CLASSES = 4


# ==============================================================================
# Load NPZ cache
# ==============================================================================
def load_cache(npz_path: str, gap_l2_norm: bool = True):
    """Load features from NPZ cache.

    Returns
    -------
    features : (N, d) float32
    labels   : (N,)  int
    uids     : (N,)  str
    """
    data = np.load(npz_path, allow_pickle=True)
    features = data["X_gap"].astype(np.float32)  # (N, d)
    labels = data["y"].astype(int)  # (N,)
    uids = data["uids"]  # (N,) str

    if gap_l2_norm:
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        features = features / norms
        logger.info(f"  Applied L2 normalization to features")

    logger.info(f"  Loaded: {npz_path}")
    logger.info(f"  Shape: {features.shape}, classes: {np.unique(labels)}")
    return features, labels, uids


# ==============================================================================
# Train/test split from CSV
# ==============================================================================
def apply_split(features, labels, uids, split_dir):
    """Split features into train+val and test using CSV split files."""
    uid_to_idx = {u: i for i, u in enumerate(uids)}

    def load_csv_uids(csv_path):
        out = []
        if not os.path.exists(csv_path):
            return out
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                out.append(row["uid"])
        return out

    train_uids = load_csv_uids(os.path.join(split_dir, "train_split.csv"))
    val_uids = load_csv_uids(os.path.join(split_dir, "val_split.csv"))
    test_uids = load_csv_uids(os.path.join(split_dir, "test_split.csv"))

    train_idx = [uid_to_idx[u] for u in train_uids + val_uids if u in uid_to_idx]
    test_idx = [uid_to_idx[u] for u in test_uids if u in uid_to_idx]

    if not train_idx or not test_idx:
        logger.warning(
            f"  Split CSVs incomplete in {split_dir}, using 80/20 random split"
        )
        n = len(features)
        perm = np.random.permutation(n)
        split = int(0.8 * n)
        train_idx = perm[:split].tolist()
        test_idx = perm[split:].tolist()

    X_train = features[train_idx]
    y_train = labels[train_idx]
    X_test = features[test_idx]
    y_test = labels[test_idx]

    logger.info(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")
    return X_train, y_train, X_test, y_test


# ==============================================================================
# Linear probe (SGD)
# ==============================================================================
def train_linear_probe(
    X_train,
    y_train,
    X_test,
    y_test,
    num_classes=4,
    lr=0.1,
    momentum=0.9,
    weight_decay=1e-4,
    epochs=50,
    batch_size=512,
    seed=42,
) -> Dict:
    """Train linear probe and return metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.RandomState(seed)

    X_tr = torch.from_numpy(X_train).float()
    y_tr = torch.from_numpy(y_train).long()
    X_te = torch.from_numpy(X_test).float()
    y_te = torch.from_numpy(y_test).long()

    probe = nn.Linear(X_tr.shape[1], num_classes, bias=False).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        probe.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n = X_tr.shape[0]
    probe.train()
    for ep in range(epochs):
        perm = rng.permutation(n)
        X_shuf = X_tr[perm]
        y_shuf = y_tr[perm]
        for i in range(0, n, batch_size):
            xb = X_shuf[i : i + batch_size].to(device)
            yb = y_shuf[i : i + batch_size].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(probe(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

    # Evaluate
    probe.eval()
    with torch.no_grad():
        all_preds, all_true = [], []
        for i in range(0, X_te.shape[0], batch_size):
            xb = X_te[i : i + batch_size].to(device)
            preds = probe(xb).argmax(dim=1).cpu()
            all_preds.append(preds)
            all_true.append(y_te[i : i + batch_size])
        preds = torch.cat(all_preds).numpy()
        true = torch.cat(all_true).numpy()

    acc = float((preds == true).mean())

    # Per-class accuracy
    per_class = {}
    for c in range(num_classes):
        mask = true == c
        if mask.sum() > 0:
            per_class[CLASS_NAMES[c]] = float((preds[mask] == c).mean())

    # Macro F1
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(true, preds):
        cm[t, p] += 1
    f1s = []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        f1s.append(f1)

    return {
        "accuracy": acc,
        "macro_f1": float(np.mean(f1s)),
        "per_class_accuracy": per_class,
    }


# ==============================================================================
# Alignment & Uniformity (Wang & Isola 2020)
# ==============================================================================
def compute_alignment(
    features: np.ndarray,
    labels: np.ndarray,
    alpha: float = 2.0,
    max_pairs: int = 500_000,
) -> float:
    """Alignment: E[‖f(x) - f(x⁺)‖^α] for same-class (positive) pairs.

    Lower is better — features of same-class samples should be close.
    Uses L2-normalized features.

    Parameters
    ----------
    alpha : exponent (default 2, standard in Wang & Isola)
    max_pairs : subsample if too many pairs
    """
    # L2-normalize
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    feats = features / np.maximum(norms, 1e-12)

    # Collect same-class pairs
    dists_sq = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) < 2:
            continue
        # Random pairs within class
        n_pairs = min(len(idx) * (len(idx) - 1) // 2, max_pairs // NUM_CLASSES)
        for _ in range(n_pairs):
            i, j = np.random.choice(idx, 2, replace=False)
            diff = feats[i] - feats[j]
            dists_sq.append(float(np.sum(diff**2)))

    if not dists_sq:
        return float("nan")

    dists_sq = np.array(dists_sq)
    # alignment = E[‖f-f⁺‖^α]
    return float(np.mean(dists_sq ** (alpha / 2)))


def compute_uniformity(
    features: np.ndarray, t: float = 2.0, max_samples: int = 10_000
) -> float:
    """Uniformity: log E[e^{-t·‖f(x) - f(y)‖²}] for all pairs.

    Lower (more negative) is better — features should be uniformly distributed.
    Uses L2-normalized features.

    Parameters
    ----------
    t : temperature (default 2, standard in Wang & Isola)
    max_samples : subsample features if too many (for memory)
    """
    # L2-normalize
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    feats = features / np.maximum(norms, 1e-12)

    # Subsample if needed
    if len(feats) > max_samples:
        idx = np.random.choice(len(feats), max_samples, replace=False)
        feats = feats[idx]

    n = len(feats)
    # Pairwise squared distances: ‖a-b‖² = ‖a‖² + ‖b‖² - 2a·b
    # For L2-normalized: = 2 - 2a·b
    dots = feats @ feats.T  # (n, n)
    sq_dists = 2.0 - 2.0 * dots  # (n, n)

    # Upper triangle (exclude diagonal)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    sq_dists_pairs = sq_dists[mask]

    # log E[exp(-t * ‖f-g‖²)]
    # Use log-sum-exp trick for numerical stability
    exponents = -t * sq_dists_pairs
    max_exp = exponents.max()
    uniformity = max_exp + np.log(np.mean(np.exp(exponents - max_exp)))

    return float(uniformity)


# ==============================================================================
# Argparse
# ==============================================================================
def get_args():
    p = argparse.ArgumentParser(
        description="Layer-wise Linear Classification + Alignment & Uniformity"
    )

    p.add_argument(
        "--features_cache",
        type=str,
        required=True,
        help="NPZ file: cnn_gap_{layer}_all.npz",
    )
    p.add_argument(
        "--split_dir",
        type=str,
        required=True,
        help="Directory containing train_split.csv / test_split.csv",
    )
    p.add_argument(
        "--layer_name",
        type=str,
        required=True,
        help="Layer name (e.g. stage5_mid, stage5_out, refine_out)",
    )
    p.add_argument(
        "--seed_name",
        type=str,
        default="",
        help="CNN encoder seed name for labeling (e.g. 87)",
    )
    p.add_argument(
        "--gap_l2_norm",
        action="store_true",
        help="Apply L2 normalization to GAP features",
    )
    p.add_argument(
        "--output_dir", type=str, default="./layer_au_results", help="Output directory"
    )

    # Linear probe
    p.add_argument("--lp_lr", type=float, default=0.1)
    p.add_argument("--lp_wd", type=float, default=1e-4)
    p.add_argument("--lp_epochs", type=int, default=50)
    p.add_argument("--lp_batch_size", type=int, default=512)

    # A&U
    p.add_argument(
        "--au_alpha",
        type=float,
        default=2.0,
        help="Alignment exponent (Wang & Isola 2020). Default: 2",
    )
    p.add_argument(
        "--au_t", type=float, default=2.0, help="Uniformity temperature. Default: 2"
    )
    p.add_argument(
        "--au_max_samples",
        type=int,
        default=10000,
        help="Max samples for uniformity (memory). Default: 10000",
    )
    p.add_argument(
        "--au_max_pairs",
        type=int,
        default=500000,
        help="Max same-class pairs for alignment. Default: 500000",
    )

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quiet", action="store_true")

    return p.parse_args()


# ==============================================================================
# Main
# ==============================================================================
def main():
    args = get_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.quiet:
        logging.getLogger("layer_au_eval").setLevel(logging.WARNING)

    logger.info(f"\n{'='*60}")
    logger.info(f"  Layer A&U Eval — layer={args.layer_name}, seed={args.seed_name}")
    logger.info(f"{'='*60}")

    # ── 1. Load features ──
    features, labels, uids = load_cache(args.features_cache, args.gap_l2_norm)

    # ── 2. Train/test split ──
    X_train, y_train, X_test, y_test = apply_split(
        features, labels, uids, args.split_dir
    )

    # ── 3. Linear probe ──
    logger.info(
        f"  Training linear probe (SGD, lr={args.lp_lr}, "
        f"epochs={args.lp_epochs})..."
    )
    lp_result = train_linear_probe(
        X_train,
        y_train,
        X_test,
        y_test,
        num_classes=NUM_CLASSES,
        lr=args.lp_lr,
        weight_decay=args.lp_wd,
        epochs=args.lp_epochs,
        batch_size=args.lp_batch_size,
        seed=args.seed,
    )
    logger.info(f"  Linear Probe Accuracy: {lp_result['accuracy']:.4f}")
    logger.info(f"  Macro F1: {lp_result['macro_f1']:.4f}")
    for c, a in lp_result["per_class_accuracy"].items():
        logger.info(f"    {c:>10s}: {a:.4f}")

    # ── 4. Alignment & Uniformity (on ALL features, L2-normalized) ──
    logger.info(f"  Computing Alignment (α={args.au_alpha})...")
    alignment = compute_alignment(
        features, labels, alpha=args.au_alpha, max_pairs=args.au_max_pairs
    )
    logger.info(f"  Alignment: {alignment:.6f}")

    logger.info(f"  Computing Uniformity (t={args.au_t})...")
    uniformity = compute_uniformity(
        features, t=args.au_t, max_samples=args.au_max_samples
    )
    logger.info(f"  Uniformity: {uniformity:.6f}")

    # ── 5. Save results ──
    result = {
        "seed": args.seed_name,
        "layer": args.layer_name,
        "gap_l2_norm": args.gap_l2_norm,
        "linear_accuracy": lp_result["accuracy"],
        "macro_f1": lp_result["macro_f1"],
        "per_class_accuracy": lp_result["per_class_accuracy"],
        "alignment": alignment,
        "uniformity": uniformity,
        "au_alpha": args.au_alpha,
        "au_t": args.au_t,
        "n_total": len(features),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    # JSON
    json_name = f"au_result_{args.layer_name}_seed{args.seed_name}.json"
    json_path = os.path.join(args.output_dir, json_name)
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"  Saved JSON: {json_path}")

    # CSV (append mode — multiple calls accumulate)
    csv_path = os.path.join(args.output_dir, "layer_au_results.csv")
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                [
                    "seed",
                    "layer",
                    "l2_norm",
                    "linear_acc",
                    "macro_f1",
                    "alignment",
                    "uniformity",
                    "n_total",
                    "n_train",
                    "n_test",
                ]
            )
        writer.writerow(
            [
                args.seed_name,
                args.layer_name,
                "Y" if args.gap_l2_norm else "N",
                f"{lp_result['accuracy']:.4f}",
                f"{lp_result['macro_f1']:.4f}",
                f"{alignment:.6f}",
                f"{uniformity:.6f}",
                len(features),
                len(X_train),
                len(X_test),
            ]
        )
    logger.info(f"  Appended to CSV: {csv_path}")

    # ── Summary ──
    logger.info(f"\n  ╔══════════════════════════════════════════╗")
    logger.info(f"  ║ Layer: {args.layer_name:>15s}  Seed: {args.seed_name:>5s} ║")
    logger.info(f"  ╠══════════════════════════════════════════╣")
    logger.info(f"  ║ Linear Acc:  {lp_result['accuracy']:>8.4f}               ║")
    logger.info(f"  ║ Macro F1:    {lp_result['macro_f1']:>8.4f}               ║")
    logger.info(f"  ║ Alignment:   {alignment:>8.6f}  (↓ better)    ║")
    logger.info(f"  ║ Uniformity:  {uniformity:>8.4f}  (↓ better)    ║")
    logger.info(f"  ╚══════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
