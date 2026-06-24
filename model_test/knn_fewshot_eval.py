# ==============================================================================
# KNN & Few-Shot Evaluation from Precomputed Feature Caches
#
# Evaluates representations WITHOUT training:
#   1. Weighted KNN classification (distance-inverse-squared weighting)
#   2. Few-shot prototypical classification (1-shot, 5-shot, etc.)
#
# Uses precomputed .npz caches (from extract_features_lambda_labs.py).
# Train/test split via split CSVs in --save_dir.
#
# Usage:
#   python -m model_test.knn_fewshot_eval \
#       --cache_path /path/to/features_cache.npz \
#       --save_dir /path/to/MoCo_seedXX \
#       --eval_modes knn fewshot \
#       --knn_k 5 10 20 \
#       --n_shots 1 5 \
#       --output_dir /path/to/results
# ==============================================================================

import argparse
import csv
import json
import logging
import os
import sys
import time
from collections import defaultdict

import matplotlib
import numpy as np
import torch

_IN_COLAB = "google.colab" in sys.modules
if not _IN_COLAB:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# ── GPU detection ──
_HAS_CUDA = torch.cuda.is_available()
try:
    import faiss

    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False

from apoptosis_prediction.local_knn_std import load_cache
# ── Project imports (maximise reuse, minimise new code) ──
from sae_project.step02_logging_utils import (CLASS_TO_LABEL, SUPERCLASS_MAP,
                                              get_logger)

logger = get_logger("knn_fewshot_eval")

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
logging.getLogger("fontTools").setLevel(logging.WARNING)

CLASS_NAMES = {0: "Control", 1: "SNCA", 2: "GBA", 3: "LRRK2"}
NUM_CLASSES = 4


# ==============================================================================
# Argument Parser
# ==============================================================================
def get_args():
    p = argparse.ArgumentParser(
        description="KNN & Few-Shot Evaluation from precomputed feature caches"
    )
    # Data
    p.add_argument(
        "--cache_path",
        type=str,
        required=True,
        help="Path to .npz feature cache (CNN or SAE)",
    )
    p.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Model output dir (contains train/val/test_split.csv)",
    )
    p.add_argument(
        "--dead_threshold",
        type=float,
        default=1e-5,
        help="SAE dead neuron threshold (default: 1e-5)",
    )

    # Feature preprocessing
    p.add_argument(
        "--norm",
        type=str,
        default="none",
        choices=["none", "log", "log_std"],
        help="Normalization before L2: 'none', 'log', or 'log_std'",
    )
    p.add_argument(
        "--gap_l2_norm", action="store_true", help="L2 normalize feature vectors"
    )

    # Evaluation modes
    p.add_argument(
        "--eval_modes",
        type=str,
        nargs="+",
        default=["knn", "fewshot"],
        choices=["knn", "fewshot"],
        help="Evaluation modes to run. Default: knn fewshot",
    )

    # KNN
    p.add_argument(
        "--knn_k",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10, 20],
        help="K values for weighted KNN. Default: 1 3 5 10 20",
    )
    p.add_argument(
        "--knn_weights",
        type=str,
        default="inv_sq",
        choices=["inv_sq", "distance", "uniform"],
        help="KNN weighting: 'inv_sq' (1/d²), "
        "'distance' (1/d), 'uniform'. Default: inv_sq",
    )

    # Few-shot
    p.add_argument(
        "--n_shots",
        type=int,
        nargs="+",
        default=[1, 5],
        help="Number of support examples per class. Default: 1 5",
    )
    p.add_argument(
        "--n_episodes",
        type=int,
        default=100,
        help="Number of random episodes per n_shot. Default: 100",
    )

    # Output
    p.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory (default: save_dir/knn_fewshot_eval)",
    )
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


# ==============================================================================
# Load split CSVs → UID sets
# ==============================================================================
def load_split_uids(save_dir):
    """Load train (train+val) and test UIDs from split CSVs.
    Returns: train_uids_set, test_uids_set
    """

    def _read_csv(path):
        uids = []
        if not os.path.exists(path):
            return uids
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                uids.append(row["uid"])
        return uids

    train_uids = _read_csv(os.path.join(save_dir, "train_split.csv"))
    val_uids = _read_csv(os.path.join(save_dir, "val_split.csv"))
    test_uids = _read_csv(os.path.join(save_dir, "test_split.csv"))

    logger.info(
        f"  Split CSVs: train={len(train_uids)}, "
        f"val={len(val_uids)}, test={len(test_uids)}"
    )

    return set(train_uids + val_uids), set(test_uids)


def _normalize_uid(uid):
    """Normalize UID to a comparable relative form.
    Handles cross-machine path differences (Lambda Labs vs Colab).
    """
    for prefix in ["Control/", "SNCA/", "GBA/", "LRRK2/"]:
        idx = uid.find(prefix)
        if idx >= 0:
            return uid[idx:]
    return uid


# ==============================================================================
# Split cache into train / test using split CSVs
# ==============================================================================
def split_cache_by_csv(X, lines, uids, save_dir):
    """Split cached features into train/test sets using split CSVs.

    Returns: X_train, y_train, X_test, y_test, source_label
    """
    train_uid_set, test_uid_set = load_split_uids(save_dir)

    # Normalize all UIDs for cross-machine matching
    train_norm = {_normalize_uid(u) for u in train_uid_set}
    test_norm = {_normalize_uid(u) for u in test_uid_set}

    # Map lines → integer labels
    superclasses = np.array([SUPERCLASS_MAP.get(str(ln), str(ln)) for ln in lines])

    train_idx, test_idx = [], []
    uids_arr = np.array(uids) if not isinstance(uids, np.ndarray) else uids

    for i, uid in enumerate(uids_arr):
        norm_uid = _normalize_uid(str(uid))
        if norm_uid in test_norm:
            test_idx.append(i)
        elif norm_uid in train_norm:
            train_idx.append(i)
        # else: not in either split → skip

    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)

    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError(
            f"Split failed: train={len(train_idx)}, test={len(test_idx)}. "
            f"UID path mismatch between cache and split CSVs."
        )

    y_all = np.array([CLASS_TO_LABEL.get(sc, -1) for sc in superclasses])

    X_train = X[train_idx]
    y_train = y_all[train_idx]
    X_test = X[test_idx]
    y_test = y_all[test_idx]

    # Remove any samples with unknown class
    valid_train = y_train >= 0
    valid_test = y_test >= 0
    X_train, y_train = X_train[valid_train], y_train[valid_train]
    X_test, y_test = X_test[valid_test], y_test[valid_test]

    logger.info(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    for c in range(NUM_CLASSES):
        n_tr = (y_train == c).sum()
        n_te = (y_test == c).sum()
        logger.info(f"    {CLASS_NAMES[c]:>8s}: train={n_tr:5d}, test={n_te:5d}")

    return X_train, y_train, X_test, y_test


# ==============================================================================
# FAISS GPU KNN — fast nearest neighbor search
# ==============================================================================
def _faiss_knn(X_train, X_test, k):
    """Find k nearest neighbors using FAISS (GPU if available, else CPU).

    Returns
    -------
    distances : (n_test, k) — L2 distances to neighbors
    indices   : (n_test, k) — neighbor indices into X_train
    """
    d = X_train.shape[1]
    k_actual = min(k, len(X_train))

    # Build index
    if _HAS_CUDA:
        res = faiss.StandardGpuResources()
        index_cpu = faiss.IndexFlatL2(d)
        index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
        logger.info(f"    FAISS: GPU mode (d={d})")
    else:
        index = faiss.IndexFlatL2(d)
        logger.info(f"    FAISS: CPU mode (d={d})")

    index.add(X_train.astype(np.float32))
    distances, indices = index.search(X_test.astype(np.float32), k_actual)
    # FAISS IndexFlatL2 returns **squared** L2 distances → convert to actual L2
    distances = np.sqrt(np.maximum(distances, 0.0))
    return distances, indices


def _sklearn_knn(X_train, X_test, k):
    """Fallback: sklearn NearestNeighbors (CPU, slower)."""
    from sklearn.neighbors import NearestNeighbors

    k_actual = min(k, len(X_train))
    nn = NearestNeighbors(n_neighbors=k_actual, metric="euclidean", n_jobs=-1)
    nn.fit(X_train)
    distances, indices = nn.kneighbors(X_test)
    return distances, indices


def _weighted_vote(distances, indices, y_train, k, weights, num_classes):
    """Weighted vote from KNN distances/indices.

    Parameters
    ----------
    distances : (n_test, k)
    indices : (n_test, k)
    y_train : (n_train,)
    weights : str — 'inv_sq', 'distance', 'uniform'

    Returns
    -------
    y_pred : (n_test,)
    """
    n_test = distances.shape[0]
    neighbor_labels = y_train[indices]  # (n_test, k)

    if weights == "uniform":
        # Simple majority vote
        y_pred = np.zeros(n_test, dtype=np.int64)
        for i in range(n_test):
            counts = np.bincount(neighbor_labels[i], minlength=num_classes)
            y_pred[i] = counts.argmax()
    else:
        # Weighted vote
        if weights == "inv_sq":
            w = 1.0 / np.maximum(distances**2, 1e-12)
        else:  # 'distance'
            w = 1.0 / np.maximum(distances, 1e-12)

        # Accumulate weighted votes per class
        # Use PyTorch for fast scatter_add if GPU available
        if _HAS_CUDA:
            w_t = torch.from_numpy(w).float().cuda()
            labels_t = torch.from_numpy(neighbor_labels).long().cuda()
            votes = torch.zeros(n_test, num_classes, device="cuda")
            votes.scatter_add_(1, labels_t, w_t)
            y_pred = votes.argmax(dim=1).cpu().numpy()
        else:
            votes = np.zeros((n_test, num_classes), dtype=np.float64)
            for c in range(num_classes):
                votes[:, c] = np.sum(w * (neighbor_labels == c), axis=1)
            y_pred = votes.argmax(axis=1)

    return y_pred


# ==============================================================================
# Weighted KNN Evaluation
# ==============================================================================
def evaluate_knn(X_train, y_train, X_test, y_test, k, weights="inv_sq"):
    """Evaluate weighted KNN classification.

    Uses FAISS GPU for fast neighbor search when available,
    falls back to sklearn CPU otherwise.

    weights:
      'inv_sq'   : w = 1/d²  (sharper than 1/d, emphasises nearest)
      'distance' : w = 1/d
      'uniform'  : equal weight (majority vote)
    """
    t0 = time.time()

    # KNN search
    if _HAS_FAISS:
        distances, indices = _faiss_knn(X_train, X_test, k)
    else:
        logger.info("    FAISS not available, using sklearn (slower)")
        distances, indices = _sklearn_knn(X_train, X_test, k)

    # Weighted vote
    y_pred = _weighted_vote(distances, indices, y_train, k, weights, NUM_CLASSES)

    elapsed = time.time() - t0
    logger.info(f"    KNN k={k} done in {elapsed:.1f}s")

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=list(range(NUM_CLASSES)))

    per_class_acc = {}
    for c in range(NUM_CLASSES):
        mask = y_test == c
        if mask.sum() > 0:
            per_class_acc[CLASS_NAMES[c]] = float((y_pred[mask] == c).mean())
        else:
            per_class_acc[CLASS_NAMES[c]] = 0.0

    return {
        "k": k,
        "weights": weights if isinstance(weights, str) else "inv_sq",
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": cm,
    }


# ==============================================================================
# Few-Shot Prototypical Evaluation (GPU-accelerated)
# ==============================================================================
def evaluate_fewshot(X_train, y_train, X_test, y_test, n_shot, n_episodes, seed=42):
    """N-shot prototypical classification with GPU acceleration.

    For each episode:
      1. Sample n_shot support examples per class from train set
      2. Compute class prototype = mean of support vectors
      3. Classify all test samples by nearest prototype (L2 distance)
      4. Compute accuracy

    Uses PyTorch GPU for batched distance computation.
    """
    t0 = time.time()
    rng = np.random.RandomState(seed)
    classes = sorted(np.unique(y_train))
    classes_arr = np.array(classes)
    device = torch.device("cuda" if _HAS_CUDA else "cpu")

    # Group train indices by class
    class_to_idx = {c: np.where(y_train == c)[0] for c in classes}

    # Check feasibility
    for c in classes:
        if len(class_to_idx[c]) < n_shot:
            logger.warning(
                f"  Class {CLASS_NAMES.get(c, c)}: only "
                f"{len(class_to_idx[c])} train samples < n_shot={n_shot}"
            )

    # Move test data to GPU once
    X_test_t = torch.from_numpy(X_test).float().to(device)  # (n_test, d)
    y_test_t = torch.from_numpy(y_test).long()
    X_train_t = torch.from_numpy(X_train).float()  # keep on CPU for indexing

    episode_accs = []
    episode_f1s = []

    for ep in range(n_episodes):
        # Sample support set and compute prototypes
        prototypes = []
        for c in classes:
            pool = class_to_idx[c]
            n_take = min(n_shot, len(pool))
            support_idx = rng.choice(pool, size=n_take, replace=False)
            support_vecs = X_train_t[support_idx]  # (n_take, d)
            prototype = support_vecs.mean(dim=0)  # (d,)
            prototypes.append(prototype)

        proto_t = torch.stack(prototypes).to(device)  # (n_classes, d)

        # Batched L2 distance: (n_test, n_classes)
        # ||x - p||² = ||x||² + ||p||² - 2·x·p^T
        x_sq = (X_test_t**2).sum(dim=1, keepdim=True)  # (n_test, 1)
        p_sq = (proto_t**2).sum(dim=1, keepdim=True).T  # (1, n_classes)
        dists = x_sq + p_sq - 2 * X_test_t @ proto_t.T  # (n_test, n_classes)

        pred_class_idx = dists.argmin(dim=1).cpu().numpy()  # indices into classes
        y_pred = classes_arr[pred_class_idx]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        episode_accs.append(acc)
        episode_f1s.append(f1)

    elapsed = time.time() - t0
    logger.info(
        f"    {n_shot}-shot × {n_episodes} episodes done in {elapsed:.1f}s "
        f"(device={device})"
    )

    mean_acc = float(np.mean(episode_accs))
    std_acc = float(np.std(episode_accs))
    mean_f1 = float(np.mean(episode_f1s))
    std_f1 = float(np.std(episode_f1s))

    return {
        "n_shot": n_shot,
        "n_episodes": n_episodes,
        "accuracy_mean": mean_acc,
        "accuracy_std": std_acc,
        "macro_f1_mean": mean_f1,
        "macro_f1_std": std_f1,
        "episode_accs": episode_accs,
        "episode_f1s": episode_f1s,
    }


# ==============================================================================
# Plotting: KNN accuracy vs K
# ==============================================================================
def plot_knn_k_sweep(knn_results, source_label, output_path, dpi=200):
    """Bar plot of KNN accuracy across K values."""
    ks = [r["k"] for r in knn_results]
    accs = [r["accuracy"] for r in knn_results]
    f1s = [r["macro_f1"] for r in knn_results]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(ks))
    w = 0.35

    bars_acc = ax.bar(
        x - w / 2,
        accs,
        w,
        label="Accuracy",
        color="#4C72B0",
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
    )
    bars_f1 = ax.bar(
        x + w / 2,
        f1s,
        w,
        label="Macro F1",
        color="#DD8452",
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
    )

    for bar, v in zip(bars_acc, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{v:.1%}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    for bar, v in zip(bars_f1, f1s):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{v:.1%}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in ks], fontsize=11)
    ax.set_xlabel("K (number of neighbors)", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, min(1.15, max(accs + f1s) + 0.08))
    ax.set_title(
        f"Weighted KNN — {source_label}\n" f"(weight = 1/d²)",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(output_path.replace(".png", ".svg"), bbox_inches="tight")
    if _IN_COLAB:
        plt.show()
    plt.close(fig)
    logger.info(f"  Saved KNN plot: {output_path}")


# ==============================================================================
# Plotting: Few-shot accuracy distribution
# ==============================================================================
def plot_fewshot_distribution(fewshot_results, source_label, output_path, dpi=200):
    """Violin + strip plot of episode accuracies for each n_shot."""
    n_shots = [r["n_shot"] for r in fewshot_results]
    accs_list = [r["episode_accs"] for r in fewshot_results]

    fig, ax = plt.subplots(figsize=(6, 5))

    positions = list(range(len(n_shots)))
    colors = ["#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]

    parts = ax.violinplot(
        accs_list,
        positions=positions,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i % len(colors)])
        pc.set_alpha(0.3)

    bp = ax.boxplot(
        accs_list,
        positions=positions,
        widths=0.3,
        patch_artist=True,
        showfliers=False,
        zorder=3,
    )
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[i % len(colors)])
        patch.set_alpha(0.6)
    for element in ["whiskers", "caps", "medians"]:
        for line in bp[element]:
            line.set_color("black")
            line.set_linewidth(1.0)

    # Annotate mean ± std
    for i, r in enumerate(fewshot_results):
        ax.text(
            i,
            ax.get_ylim()[1] * 0.98 if i == 0 else max(r["episode_accs"]) + 0.02,
            f"μ={r['accuracy_mean']:.1%}\n±{r['accuracy_std']:.1%}",
            ha="center",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{n}-shot" for n in n_shots], fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(
        f"Few-Shot Prototypical — {source_label}\n"
        f"({fewshot_results[0]['n_episodes']} episodes)",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(output_path.replace(".png", ".svg"), bbox_inches="tight")
    if _IN_COLAB:
        plt.show()
    plt.close(fig)
    logger.info(f"  Saved few-shot plot: {output_path}")


# ==============================================================================
# Plotting: Confusion matrix (reused from linear_eval.py pattern)
# ==============================================================================
def plot_confusion_matrix(cm, title, output_path, dpi=200):
    """Plot and save confusion matrix."""
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
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12,
            )

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i + 0.25,
                f"({cm_norm[i,j]:.1f}%)",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "gray",
                fontsize=8,
            )

    acc = np.trace(cm) / cm.sum()
    ax.set_title(f"{title}\nAccuracy: {acc:.1%}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(output_path.replace(".png", ".svg"), bbox_inches="tight")
    if _IN_COLAB:
        plt.show()
    plt.close(fig)
    logger.info(f"  Saved confusion matrix: {output_path}")


# ==============================================================================
# Main
# ==============================================================================
def main():
    args = get_args()
    np.random.seed(args.seed)

    # ── Output dir ──
    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = os.path.join(args.save_dir, "knn_fewshot_eval")
    os.makedirs(out_dir, exist_ok=True)

    # ── Load cache ──
    logger.info(f"\n{'='*60}")
    logger.info(f"Loading cache: {args.cache_path}")
    X, lines, uids, source_label = load_cache(args.cache_path, args.dead_threshold)
    logger.info(f"  Source: {source_label}, shape: {X.shape}")

    # ── Optional L2 normalization ──
    if args.gap_l2_norm:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-12, norms)
        X = X / norms
        logger.info(f"  Applied L2 normalization")

    # ── Optional log/log_std normalization ──
    if getattr(args, "norm", "none") in ["log", "log_std"]:
        X = np.log1p(np.maximum(X, 0))
        logger.info(f"  Applied log normalization (np.log1p(max(X, 0)))")

    if getattr(args, "norm", "none") == "log_std":
        from sklearn.preprocessing import StandardScaler

        X = StandardScaler().fit_transform(X)
        logger.info(f"  Applied StandardScaler (log_std)")

    # ── Split into train / test ──
    logger.info(f"\nSplitting by CSV...")
    X_train, y_train, X_test, y_test = split_cache_by_csv(X, lines, uids, args.save_dir)

    all_results = {
        "source": source_label,
        "cache_path": args.cache_path,
        "save_dir": args.save_dir,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "gap_l2_norm": args.gap_l2_norm,
        "feature_dim": X.shape[1],
    }

    # ══════════════════════════════════════════════════════════════
    # KNN Evaluation
    # ══════════════════════════════════════════════════════════════
    if "knn" in args.eval_modes:
        logger.info(f"\n{'='*60}")
        logger.info(f"  KNN Evaluation (weights={args.knn_weights})")
        logger.info(f"{'='*60}")

        knn_results = []
        for k in args.knn_k:
            logger.info(f"\n  k={k}...")
            result = evaluate_knn(
                X_train, y_train, X_test, y_test, k=k, weights=args.knn_weights
            )
            knn_results.append(result)

            logger.info(f"    Accuracy: {result['accuracy']:.4f}")
            logger.info(f"    Macro F1: {result['macro_f1']:.4f}")
            for c, acc in result["per_class_accuracy"].items():
                logger.info(f"    {c:>8s}: {acc:.4f}")

        all_results["knn"] = [
            {k: v for k, v in r.items() if k != "confusion_matrix"} for r in knn_results
        ]

        # ── Save confusion matrix CSV + PNG for EVERY k ──
        for r in knn_results:
            k_val = r["k"]
            cm = r["confusion_matrix"]

            # CSV: rows = true class, cols = predicted class
            # Format: source, k, true_class, pred_Control, pred_SNCA, pred_GBA, pred_LRRK2
            cm_csv_path = os.path.join(out_dir, f"cm_knn_k{k_val}_{source_label}.csv")
            with open(cm_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                header = (
                    ["source", "k", "true_class"]
                    + [f"pred_{CLASS_NAMES[j]}" for j in range(NUM_CLASSES)]
                    + ["total", "accuracy"]
                )
                writer.writerow(header)
                for i in range(NUM_CLASSES):
                    row_total = cm[i].sum()
                    row_acc = cm[i, i] / row_total if row_total > 0 else 0
                    writer.writerow(
                        [
                            source_label,
                            k_val,
                            CLASS_NAMES[i],
                            *cm[i].tolist(),
                            int(row_total),
                            f"{row_acc:.4f}",
                        ]
                    )
                # Summary row
                writer.writerow(
                    [
                        source_label,
                        k_val,
                        "TOTAL",
                        *cm.sum(axis=0).tolist(),
                        int(cm.sum()),
                        f"{r['accuracy']:.4f}",
                    ]
                )
            logger.info(f"  Saved CM CSV: {cm_csv_path}")

            # PNG
            cm_png_path = os.path.join(out_dir, f"cm_knn_k{k_val}_{source_label}.png")
            plot_confusion_matrix(
                cm,
                f"KNN (k={k_val}, {args.knn_weights}) — {source_label}",
                cm_png_path,
                args.dpi,
            )

        # K sweep plot
        sweep_path = os.path.join(out_dir, f"knn_sweep_{source_label}.png")
        plot_knn_k_sweep(knn_results, source_label, sweep_path, args.dpi)

        # Summary table
        logger.info(f"\n  ── KNN Summary ──")
        logger.info(
            f"  {'k':>4s}  {'Acc':>7s}  {'F1':>7s}  "
            f"{'Ctrl':>7s}  {'SNCA':>7s}  {'GBA':>7s}  {'LRRK2':>7s}"
        )
        logger.info(f"  {'─'*55}")
        for r in knn_results:
            logger.info(
                f"  {r['k']:4d}  {r['accuracy']:7.4f}  {r['macro_f1']:7.4f}  "
                f"{r['per_class_accuracy'].get('Control',0):7.4f}  "
                f"{r['per_class_accuracy'].get('SNCA',0):7.4f}  "
                f"{r['per_class_accuracy'].get('GBA',0):7.4f}  "
                f"{r['per_class_accuracy'].get('LRRK2',0):7.4f}"
            )

    # ══════════════════════════════════════════════════════════════
    # Few-Shot Evaluation
    # ══════════════════════════════════════════════════════════════
    if "fewshot" in args.eval_modes:
        logger.info(f"\n{'='*60}")
        logger.info(f"  Few-Shot Prototypical Evaluation")
        logger.info(f"{'='*60}")

        fewshot_results = []
        for n_shot in args.n_shots:
            logger.info(f"\n  {n_shot}-shot ({args.n_episodes} episodes)...")
            result = evaluate_fewshot(
                X_train,
                y_train,
                X_test,
                y_test,
                n_shot=n_shot,
                n_episodes=args.n_episodes,
                seed=args.seed,
            )
            fewshot_results.append(result)

            logger.info(
                f"    Accuracy: {result['accuracy_mean']:.4f} "
                f"± {result['accuracy_std']:.4f}"
            )
            logger.info(
                f"    Macro F1: {result['macro_f1_mean']:.4f} "
                f"± {result['macro_f1_std']:.4f}"
            )

        all_results["fewshot"] = [
            {k: v for k, v in r.items() if k not in ("episode_accs", "episode_f1s")}
            for r in fewshot_results
        ]

        # Distribution plot
        dist_path = os.path.join(out_dir, f"fewshot_dist_{source_label}.png")
        plot_fewshot_distribution(fewshot_results, source_label, dist_path, args.dpi)

        # Summary table
        logger.info(f"\n  ── Few-Shot Summary ──")
        logger.info(
            f"  {'N-shot':>6s}  {'Acc (mean±std)':>18s}  "
            f"{'F1 (mean±std)':>18s}  {'Episodes':>8s}"
        )
        logger.info(f"  {'─'*55}")
        for r in fewshot_results:
            logger.info(
                f"  {r['n_shot']:6d}  "
                f"{r['accuracy_mean']:7.4f} ± {r['accuracy_std']:.4f}  "
                f"{r['macro_f1_mean']:7.4f} ± {r['macro_f1_std']:.4f}  "
                f"{r['n_episodes']:8d}"
            )

    # ── Save JSON ──
    json_path = os.path.join(out_dir, f"eval_results_{source_label}.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nSaved results: {json_path}")

    # ── Save CSV summary ──
    csv_path = os.path.join(out_dir, f"eval_summary_{source_label}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Method", "Param", "Accuracy", "Macro_F1", "Ctrl", "SNCA", "GBA", "LRRK2"]
        )
        if "knn" in args.eval_modes:
            for r in all_results.get("knn", []):
                writer.writerow(
                    [
                        "KNN",
                        f"k={r['k']}",
                        f"{r['accuracy']:.4f}",
                        f"{r['macro_f1']:.4f}",
                        f"{r['per_class_accuracy'].get('Control',0):.4f}",
                        f"{r['per_class_accuracy'].get('SNCA',0):.4f}",
                        f"{r['per_class_accuracy'].get('GBA',0):.4f}",
                        f"{r['per_class_accuracy'].get('LRRK2',0):.4f}",
                    ]
                )
        if "fewshot" in args.eval_modes:
            for r in all_results.get("fewshot", []):
                writer.writerow(
                    [
                        "FewShot",
                        f"{r['n_shot']}-shot",
                        f"{r['accuracy_mean']:.4f}±{r['accuracy_std']:.4f}",
                        f"{r['macro_f1_mean']:.4f}±{r['macro_f1_std']:.4f}",
                        "",
                        "",
                        "",
                        "",
                    ]
                )
    logger.info(f"Saved CSV: {csv_path}")

    logger.info(f"\n{'='*60}")
    logger.info("KNN & Few-Shot evaluation complete!")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
