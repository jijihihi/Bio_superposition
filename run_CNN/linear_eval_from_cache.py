import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Tuple

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

_IN_COLAB = "google.colab" in sys.modules
if not _IN_COLAB:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASS_NAMES = {0: "Control", 1: "SNCA", 2: "GBA", 3: "LRRK2"}
NUM_CLASSES = 4

# ==============================================================================
# Helper: Load Split CSVs
# ==============================================================================
def load_split_csv(csv_path: str) -> List[str]:
    uids = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uids.append(row["uid"])
    return set(uids)

# ==============================================================================
# Linear probe training (SGD, exactly matches original linear_eval.py)
# ==============================================================================
def train_linear_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int = 4,
    lr: float = 0.1,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    epochs: int = 50,
    batch_size: int = 512,
    seed: int = 42,
) -> Dict:
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
            xb = X_te[i : i + batch_size].to(device)
            yb = y_te[i : i + batch_size]
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

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(true, preds):
        cm[t, p] += 1

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
    }

# ==============================================================================
# Confusion matrix plot
# ==============================================================================
def plot_confusion_matrix(cm: np.ndarray, output_path: str):
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    classes = [CLASS_NAMES[i] for i in range(cm.shape[0])]

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True label",
        xlabel="Predicted label",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > thresh else "black"
            ax.text(j, i, format(val, "d"), ha="center", va="center", color=color)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser("Linear Eval from Cache")
    parser.add_argument("--save_dir", type=str, required=True, help="Path containing train/test split CSVs")
    parser.add_argument("--cache_path", type=str, required=True, help="Path to cnn_gap_stage5_mid.npz")
    parser.add_argument("--apply_l2_norm", action="store_true", help="Apply L2 normalization before training")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--wd", type=float, default=1e-4, help="Weight decay for SGD")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_cm_plot", action="store_true", help="Save confusion matrix plot")
    args = parser.parse_args()

    print(f"Loading cache from: {args.cache_path}")
    data = np.load(args.cache_path, allow_pickle=True)
    X_gap = data["X_gap"]
    y = data["y"]
    uids = data["uids"]

    # Normalize if requested
    if args.apply_l2_norm:
        print("Applying L2 Normalization to features...")
        X_gap = F.normalize(torch.tensor(X_gap), dim=1).numpy()

    # Load splits
    train_csv = os.path.join(args.save_dir, "train_split.csv")
    test_csv = os.path.join(args.save_dir, "test_split.csv")
    
    # Validation split is typically merged with train for linear probe training or ignored.
    # The original linear_eval.py did: X_tr = loader (which is train+val splits by default if used correctly).
    # We will use train_split.csv and val_split.csv as training data, test_split.csv as testing data.
    val_csv = os.path.join(args.save_dir, "val_split.csv")
    
    train_uids = load_split_csv(train_csv)
    if os.path.exists(val_csv):
        train_uids.update(load_split_csv(val_csv))
    test_uids = load_split_csv(test_csv)

    print(f"Train UIDs: {len(train_uids)}, Test UIDs: {len(test_uids)}")

    # Standardize UIDs to relative paths for O(1) lookup
    def get_rel_uid(uid: str) -> str:
        # Match extract_cnn_gap.py's relative path logic
        for cls_prefix in ["Control/", "SNCA/", "GBA/", "LRRK2/"]:
            idx = uid.find(cls_prefix)
            if idx >= 0:
                return uid[idx:]
        return uid

    # Create O(1) lookup dictionary
    rel_uid_to_idx = {get_rel_uid(uid): i for i, uid in enumerate(uids)}
    
    train_indices = []
    test_indices = []
    
    for csv_uid in train_uids:
        rel_key = get_rel_uid(csv_uid)
        if rel_key in rel_uid_to_idx:
            train_indices.append(rel_uid_to_idx[rel_key])
            
    for csv_uid in test_uids:
        rel_key = get_rel_uid(csv_uid)
        if rel_key in rel_uid_to_idx:
            test_indices.append(rel_uid_to_idx[rel_key])

    X_train, y_train = X_gap[train_indices], y[train_indices]
    X_test, y_test = X_gap[test_indices], y[test_indices]
    
    print(f"Matched Train: {X_train.shape[0]}, Matched Test: {X_test.shape[0]}")

    print("Training Linear Probe...")
    res = train_linear_probe(
        X_train, y_train, X_test, y_test,
        epochs=args.epochs, lr=args.lr, weight_decay=args.wd, batch_size=args.batch_size, seed=args.seed
    )

    print(f"Linear Evaluation Accuracy: {res['accuracy'] * 100:.2f}%")

    # Save results
    res_json = {k: v for k, v in res.items() if k != "confusion_matrix"}
    cm = res["confusion_matrix"]
    res_json["confusion_matrix"] = cm.tolist()
    
    out_json = os.path.join(args.save_dir, "linear_eval_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(res_json, f, indent=2)
        
    print(f"Saved metrics to {out_json}")

    # Plot Confusion Matrix if requested
    if args.save_cm_plot:
        cm_path = os.path.join(args.save_dir, "confusion_matrix.svg")
        plot_confusion_matrix(cm, cm_path)
        print(f"Saved confusion matrix plot to {cm_path}")

if __name__ == "__main__":
    main()
