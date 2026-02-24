# ==============================================================================
# Linear Classification Evaluation for CNN Encoders
#
# Standard linear evaluation protocol:
#   - Freeze encoder, train linear head with SGD
#   - Train on train+val split, evaluate on test split (no leakage)
#   - Reports accuracy, per-class accuracy, F1, confusion matrix
#   - Supports multi-seed aggregation with mean/std plots
#
# Usage:
#   # Single model
#   python -m model_test.linear_eval \
#       --ckpt_path /path/to/best_model.pt \
#       --save_dir /path/to/MoCo_seed42 \
#       --shard_root /path/to/wds_shards_tar
#
#   # Multiple seeds (table + bar plot)
#   python -m model_test.linear_eval \
#       --ckpt_paths /path/to/MoCo_seed42/best_model.pt \
#                    /path/to/MoCo_seed45/best_model.pt \
#       --save_dirs /path/to/MoCo_seed42 /path/to/MoCo_seed45 \
#       --shard_root /path/to/wds_shards_tar
# ==============================================================================

import os
import sys
import csv
import json
import random
import argparse
import logging
from typing import List, Tuple, Dict
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import matplotlib
_IN_COLAB = "google.colab" in sys.modules
if not _IN_COLAB:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sae_project.step02_logging_utils import get_logger
from sae_project.step03_data_shards import load_all_sample_refs, build_uid_to_refidx
from sae_project.step04_data_bank import (
    InMemoryTarBank, InMemorySixteenBitDataset,
    seed_worker, collate_skip_none,
)
from sae_project.step05_model_encoder import (
    Encoder, SupMoCoModel, parse_int_list,
    renorm_unit_per_out_channel_, robust_load_state_dict,
)
from sae_project.step02_logging_utils import OUT_DIM

logger = get_logger("linear_eval")

CLASS_NAMES = {0: "Control", 1: "SNCA", 2: "GBA", 3: "LRRK2"}
NUM_CLASSES = 4


# ==============================================================================
# Load split CSVs
# ==============================================================================
def load_split_csv(csv_path: str) -> List[str]:
    uids = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uids.append(row["uid"])
    return uids


# ==============================================================================
# Feature extraction (frozen encoder)
# ==============================================================================
@torch.no_grad()
def extract_features(encoder: nn.Module, loader: DataLoader,
                     device: torch.device, use_bf16: bool = True
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Extract L2-normalized GAP features from frozen encoder."""
    encoder.eval()
    autocast_kwargs = dict(device_type="cuda", enabled=torch.cuda.is_available())
    if use_bf16 and torch.cuda.is_available():
        autocast_kwargs["dtype"] = torch.bfloat16

    all_feats, all_labels = [], []
    for batch in tqdm(loader, desc="Extracting features", leave=False):
        if batch is None:
            continue
        x, y, *_ = batch
        if x.numel() < 1:
            continue
        x = x.to(device, non_blocking=True).contiguous(
            memory_format=torch.channels_last)
        with torch.amp.autocast(**autocast_kwargs):
            feat = encoder(x)
        feat = F.normalize(feat, dim=1)
        all_feats.append(feat.cpu().float().numpy())
        all_labels.append(y.numpy())

    X = np.concatenate(all_feats, axis=0)
    y = np.concatenate(all_labels, axis=0)
    return X, y


# ==============================================================================
# Linear probe training (SGD, no data leakage)
# ==============================================================================
def train_linear_probe(X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray,
                       num_classes: int = 4,
                       lr: float = 0.1, momentum: float = 0.9,
                       weight_decay: float = 1e-4,
                       epochs: int = 50, batch_size: int = 512,
                       seed: int = 42,
                       ) -> Dict:
    """Train linear probe with SGD and evaluate on test set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.RandomState(seed)

    X_tr = torch.from_numpy(X_train).float()
    y_tr = torch.from_numpy(y_train).long()
    X_te = torch.from_numpy(X_test).float()
    y_te = torch.from_numpy(y_test).long()

    probe = nn.Linear(X_tr.shape[1], num_classes, bias=False).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(probe.parameters(), lr=lr, momentum=momentum,
                          weight_decay=weight_decay)
    # Cosine annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n = X_tr.shape[0]
    steps_per_epoch = max(1, n // batch_size)

    probe.train()
    for ep in range(epochs):
        perm = rng.permutation(n)
        X_shuf = X_tr[perm]
        y_shuf = y_tr[perm]

        for i in range(0, n, batch_size):
            xb = X_shuf[i:i+batch_size].to(device)
            yb = y_shuf[i:i+batch_size].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = probe(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        scheduler.step()

    # Evaluate
    probe.eval()
    with torch.no_grad():
        all_preds = []
        all_true = []
        for i in range(0, X_te.shape[0], batch_size):
            xb = X_te[i:i+batch_size].to(device)
            yb = y_te[i:i+batch_size]
            logits = probe(xb)
            preds = logits.argmax(dim=1).cpu()
            all_preds.append(preds)
            all_true.append(yb)

        preds = torch.cat(all_preds).numpy()
        true = torch.cat(all_true).numpy()

    # Metrics
    acc = (preds == true).mean()

    per_class_acc = {}
    per_class_n = {}
    for c in range(num_classes):
        mask = true == c
        if mask.sum() > 0:
            per_class_acc[CLASS_NAMES[c]] = float((preds[mask] == c).mean())
            per_class_n[CLASS_NAMES[c]] = int(mask.sum())

    # Confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(true, preds):
        cm[t, p] += 1

    # Macro F1
    f1_scores = []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        f1_scores.append(f1)
    macro_f1 = np.mean(f1_scores)

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "per_class_accuracy": per_class_acc,
        "per_class_n": per_class_n,
        "confusion_matrix": cm,
        "predictions": preds,
        "true_labels": true,
    }


# ==============================================================================
# Confusion matrix plot
# ==============================================================================
def plot_confusion_matrix(cm: np.ndarray, model_name: str,
                          output_path: str, dpi: int = 200):
    """Plot and save confusion matrix."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    classes = [CLASS_NAMES[i] for i in range(cm.shape[0])]

    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes,
           yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12)

    # Normalized percentages
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i + 0.25, f"({cm_norm[i,j]:.1f}%)",
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "gray",
                    fontsize=8)

    acc = np.trace(cm) / cm.sum()
    ax.set_title(f"Confusion Matrix – {model_name}\nAccuracy: {acc:.1%}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if _IN_COLAB:
        plt.show()
    plt.close(fig)
    logger.info(f"Saved confusion matrix: {output_path}")


# ==============================================================================
# Multi-seed accuracy bar plot
# ==============================================================================
def plot_multi_seed_bar(results: List[Dict], model_type: str,
                        output_path: str, dpi: int = 200):
    """Bar plot of per-class accuracy across seeds with mean±std."""
    classes = ["Control", "SNCA", "GBA", "LRRK2"]

    # Gather per-class accuracies
    class_accs = {c: [] for c in classes}
    overall_accs = []
    for r in results:
        overall_accs.append(r["accuracy"])
        for c in classes:
            class_accs[c].append(r["per_class_accuracy"].get(c, 0))

    categories = classes + ["Overall"]
    means = [np.mean(class_accs[c]) for c in classes] + [np.mean(overall_accs)]
    stds = [np.std(class_accs[c]) for c in classes] + [np.std(overall_accs)]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
    x = np.arange(len(categories))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors,
                  edgecolor="black", linewidth=0.5, alpha=0.85)

    # Annotate
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.005,
                f"{m:.1%}±{s:.1%}", ha='center', va='bottom', fontsize=9)

    ax.set_ylim(0, min(1.15, max(means) + max(stds) + 0.08))
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title(f"Linear Evaluation – {model_type}\n"
                 f"({len(results)} seeds, mean±std)",
                 fontsize=13, fontweight="bold")
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if _IN_COLAB:
        plt.show()
    plt.close(fig)
    logger.info(f"Saved bar plot: {output_path}")


# ==============================================================================
# Evaluate single checkpoint
# ==============================================================================
def evaluate_single(ckpt_path: str, save_dir: str, refs, uid_to_refidx,
                    args) -> Dict:
    """Full pipeline: load encoder → extract features → train LP → evaluate."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = os.path.basename(os.path.dirname(ckpt_path))
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating: {model_name}")
    logger.info(f"Checkpoint: {ckpt_path}")

    # Load encoder
    blocks = parse_int_list(args.blocks, 4)
    dilations = parse_int_list(args.dilations, 4)
    model = SupMoCoModel(
        embed_dim=args.embed_dim,
        blocks=blocks, dilations=dilations,
        refine_blocks=args.refine_blocks,
        ckpt_segments=args.ckpt_segments,
        proj_layers=2, proj_hidden=2048,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    robust_load_state_dict(model, ckpt, strict=False)

    encoder = model.encoder
    encoder.eval()
    encoder.to(device).to(memory_format=torch.channels_last)
    renorm_unit_per_out_channel_(encoder)
    del model

    # Load train+val and test splits
    train_csv = os.path.join(save_dir, "train_split.csv")
    val_csv = os.path.join(save_dir, "val_split.csv")
    test_csv = os.path.join(save_dir, "test_split.csv")

    train_uids = []
    if os.path.exists(train_csv):
        train_uids.extend(load_split_csv(train_csv))
    if os.path.exists(val_csv):
        train_uids.extend(load_split_csv(val_csv))
    test_uids = load_split_csv(test_csv) if os.path.exists(test_csv) else []

    if not train_uids or not test_uids:
        raise FileNotFoundError(
            f"Missing split CSVs in {save_dir}. "
            f"Need train_split.csv (or val_split.csv) + test_split.csv")

    # Map UIDs → ref indices
    def uids_to_ref_indices(uids):
        return [uid_to_refidx[u] for u in uids if u in uid_to_refidx]

    train_ref_idx = uids_to_ref_indices(train_uids)
    test_ref_idx = uids_to_ref_indices(test_uids)

    logger.info(f"  Train samples: {len(train_ref_idx)}")
    logger.info(f"  Test samples:  {len(test_ref_idx)}")

    # Create data loaders
    def make_loader(ref_indices, shuffle=False):
        bank = InMemoryTarBank(refs, ref_indices, args.img_size)
        ib = list(range(len(ref_indices)))
        ds = InMemorySixteenBitDataset(bank, ib, args.img_size, augment=False)
        return DataLoader(
            ds, batch_size=args.batch_size, shuffle=shuffle,
            num_workers=args.num_workers, pin_memory=True,
            worker_init_fn=seed_worker, collate_fn=collate_skip_none)

    logger.info("  Loading train data...")
    train_loader = make_loader(train_ref_idx)
    logger.info("  Loading test data...")
    test_loader = make_loader(test_ref_idx)

    # Extract features
    logger.info("  Extracting train features...")
    X_train, y_train = extract_features(encoder, train_loader, device)
    logger.info(f"  Train features: {X_train.shape}")

    logger.info("  Extracting test features...")
    X_test, y_test = extract_features(encoder, test_loader, device)
    logger.info(f"  Test features: {X_test.shape}")

    del encoder, train_loader, test_loader
    torch.cuda.empty_cache()

    # Train linear probe
    logger.info(f"  Training linear probe (SGD, lr={args.lp_lr}, "
                f"epochs={args.lp_epochs})...")
    results = train_linear_probe(
        X_train, y_train, X_test, y_test,
        num_classes=NUM_CLASSES,
        lr=args.lp_lr, momentum=0.9,
        weight_decay=args.lp_wd,
        epochs=args.lp_epochs,
        batch_size=args.lp_batch_size,
        seed=args.seed,
    )

    results["model_name"] = model_name
    results["ckpt_path"] = ckpt_path

    # Log results
    logger.info(f"\n  === Results: {model_name} ===")
    logger.info(f"  Overall Accuracy: {results['accuracy']:.4f}")
    logger.info(f"  Macro F1: {results['macro_f1']:.4f}")
    for c, acc in results["per_class_accuracy"].items():
        n = results["per_class_n"][c]
        logger.info(f"  {c:>10s}: {acc:.4f} (n={n})")

    logger.info(f"\n  Confusion Matrix:")
    cm = results["confusion_matrix"]
    header = "          " + "  ".join(f"{CLASS_NAMES[j]:>8s}" for j in range(4))
    logger.info(f"  {header}")
    for i in range(4):
        row = f"  {CLASS_NAMES[i]:>8s}  " + "  ".join(f"{cm[i,j]:>8d}" for j in range(4))
        logger.info(row)

    return results


# ==============================================================================
# Main
# ==============================================================================
def get_args():
    p = argparse.ArgumentParser("Linear Evaluation for CNN Encoders")

    # Single model
    p.add_argument("--ckpt_path", type=str, default="",
                   help="Single checkpoint path")
    p.add_argument("--save_dir", type=str, default="",
                   help="Save dir for single model (contains split CSVs)")

    # Multiple models
    p.add_argument("--ckpt_paths", type=str, nargs="+", default=[],
                   help="Multiple checkpoint paths")
    p.add_argument("--save_dirs", type=str, nargs="+", default=[],
                   help="Corresponding save dirs")

    # Data
    p.add_argument("--shard_root", type=str,
                   default="/home/ubuntu/model-east3/wds_shards_tar")
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)

    # Encoder architecture
    p.add_argument("--embed_dim", type=int, default=512)
    p.add_argument("--blocks", type=str, default="2,2,2,3")
    p.add_argument("--dilations", type=str, default="1,1,1,1")
    p.add_argument("--refine_blocks", type=int, default=1)
    p.add_argument("--ckpt_segments", type=int, default=0)

    # Linear probe
    p.add_argument("--lp_lr", type=float, default=0.1)
    p.add_argument("--lp_wd", type=float, default=1e-4)
    p.add_argument("--lp_epochs", type=int, default=50)
    p.add_argument("--lp_batch_size", type=int, default=512)

    # Output
    p.add_argument("--output_dir", type=str, default="./linear_eval_results")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def main():
    args = get_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # Build checkpoint list
    ckpt_paths = []
    save_dirs = []
    if args.ckpt_paths:
        ckpt_paths = args.ckpt_paths
        save_dirs = args.save_dirs
        if len(save_dirs) != len(ckpt_paths):
            # Auto-infer save_dirs from ckpt_paths
            save_dirs = [os.path.dirname(p) for p in ckpt_paths]
    elif args.ckpt_path:
        ckpt_paths = [args.ckpt_path]
        save_dirs = [args.save_dir if args.save_dir else
                     os.path.dirname(args.ckpt_path)]
    else:
        raise ValueError("Provide --ckpt_path or --ckpt_paths")

    # Load shard refs once
    logger.info("Loading sample refs...")
    refs = load_all_sample_refs(args.shard_root)
    uid_to_refidx = build_uid_to_refidx(refs)

    # Evaluate each checkpoint
    all_results = []
    for ckpt, sdir in zip(ckpt_paths, save_dirs):
        try:
            result = evaluate_single(ckpt, sdir, refs, uid_to_refidx, args)
            all_results.append(result)
        except Exception as e:
            logger.error(f"Failed: {ckpt}: {e}")
            import traceback
            traceback.print_exc()

    if not all_results:
        logger.error("No successful evaluations!")
        return

    # ── Summary table ──
    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY TABLE")
    logger.info(f"{'='*80}")
    header = f"{'Model':<30s} {'Acc':>7s} {'F1':>7s} {'Ctrl':>7s} {'SNCA':>7s} {'GBA':>7s} {'LRRK2':>7s}"
    logger.info(header)
    logger.info("-" * 80)
    for r in all_results:
        row = (f"{r['model_name']:<30s} "
               f"{r['accuracy']:>7.4f} "
               f"{r['macro_f1']:>7.4f} "
               f"{r['per_class_accuracy'].get('Control',0):>7.4f} "
               f"{r['per_class_accuracy'].get('SNCA',0):>7.4f} "
               f"{r['per_class_accuracy'].get('GBA',0):>7.4f} "
               f"{r['per_class_accuracy'].get('LRRK2',0):>7.4f}")
        logger.info(row)

    if len(all_results) > 1:
        accs = [r["accuracy"] for r in all_results]
        f1s = [r["macro_f1"] for r in all_results]
        logger.info("-" * 80)
        logger.info(f"{'Mean±Std':<30s} "
                    f"{np.mean(accs):>7.4f} "
                    f"{np.mean(f1s):>7.4f}")
        logger.info(f"{'':30s} ±{np.std(accs):.4f} ±{np.std(f1s):.4f}")

    # ── Save CSV ──
    csv_path = os.path.join(args.output_dir, "linear_eval_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Accuracy", "Macro_F1",
                         "Ctrl_Acc", "SNCA_Acc", "GBA_Acc", "LRRK2_Acc"])
        for r in all_results:
            writer.writerow([
                r["model_name"],
                f"{r['accuracy']:.4f}",
                f"{r['macro_f1']:.4f}",
                f"{r['per_class_accuracy'].get('Control',0):.4f}",
                f"{r['per_class_accuracy'].get('SNCA',0):.4f}",
                f"{r['per_class_accuracy'].get('GBA',0):.4f}",
                f"{r['per_class_accuracy'].get('LRRK2',0):.4f}",
            ])
    logger.info(f"\nSaved CSV: {csv_path}")

    # ── Confusion matrix (first model as representative) ──
    cm_path = os.path.join(args.output_dir,
                           f"confusion_matrix_{all_results[0]['model_name']}.png")
    plot_confusion_matrix(
        all_results[0]["confusion_matrix"],
        all_results[0]["model_name"],
        cm_path, args.dpi)

    # ── Multi-seed bar plot ──
    if len(all_results) > 1:
        # Infer model type from name
        first_name = all_results[0]["model_name"]
        if "no_GAPL2norm" in first_name:
            model_type = "MoCo (no GAP L2 norm)"
        else:
            model_type = "MoCo"

        bar_path = os.path.join(args.output_dir,
                                f"accuracy_bar_{model_type.replace(' ', '_')}.png")
        plot_multi_seed_bar(all_results, model_type, bar_path, args.dpi)

    # ── Save full results JSON ──
    json_results = []
    for r in all_results:
        jr = {k: v for k, v in r.items()
              if k not in ("confusion_matrix", "predictions", "true_labels")}
        jr["confusion_matrix"] = r["confusion_matrix"].tolist()
        json_results.append(jr)

    json_path = os.path.join(args.output_dir, "linear_eval_results.json")
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    logger.info(f"Saved JSON: {json_path}")

    logger.info(f"\n{'='*60}")
    logger.info("Linear evaluation complete!")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
