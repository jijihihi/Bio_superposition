import argparse
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap

from model_train.logging_utils import get_logger

logger = get_logger("plot_umap_cnn_vs_sae")

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
sns.set_style("ticks")

# 0 to 8 colors (adjust as needed if there are more classes)
PALETTE = sns.color_palette("tab10", 10)


def load_cache(path, is_sae=False, dead_threshold=1e-5, gap_l2_norm=False):
    data = np.load(path, allow_pickle=True)
    if is_sae:
        if "X_all" in data:
            X = data["X_all"]
        elif "X_gap" in data:
            X = data["X_gap"]
        else:
            raise KeyError(f"SAE cache has keys: {list(data.keys())}")

        if "usage_ema" in data:
            usage_ema = data["usage_ema"]
            alive_mask = usage_ema >= dead_threshold
            X = X[:, alive_mask]
            logger.info(f"Loaded SAE cache: {path}")
            logger.info(
                f"  Shape: {X.shape} (alive neurons: {alive_mask.sum()}/{len(usage_ema)}, thresh={dead_threshold})"
            )
        else:
            logger.warning("No usage_ema found in SAE cache, using all neurons.")
    else:
        if "X_gap" in data:
            X = data["X_gap"]
        else:
            X = data["X_all"]
        logger.info(f"Loaded CNN cache: {path}")
        logger.info(f"  Shape: {X.shape}")

    y = data["y"]

    if gap_l2_norm:
        logger.info("Applying L2 Normalization...")
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-12, norms)
        X = X / norms

    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cnn_cache", type=str, required=True)
    parser.add_argument("--sae_cache", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./")
    parser.add_argument("--dead_threshold", type=float, default=1e-5)
    parser.add_argument(
        "--gap_l2_norm", action="store_true", help="Apply L2 norm to BOTH CNN and SAE"
    )
    parser.add_argument(
        "--cnn_only_l2_norm",
        action="store_true",
        help="Apply L2 norm ONLY to CNN (Recommended)",
    )
    parser.add_argument("--n_neighbors", type=int, default=15)
    parser.add_argument("--min_dist", type=float, default=0.1)
    parser.add_argument(
        "--subsample", type=int, default=5000, help="Max cells per class to plot"
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=[],
        help="Specific class numbers to plot (empty = all)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Data
    cnn_l2 = args.gap_l2_norm or args.cnn_only_l2_norm
    sae_l2 = args.gap_l2_norm

    X_cnn, y_cnn = load_cache(args.cnn_cache, is_sae=False, gap_l2_norm=cnn_l2)
    X_sae, y_sae = load_cache(
        args.sae_cache,
        is_sae=True,
        dead_threshold=args.dead_threshold,
        gap_l2_norm=sae_l2,
    )

    # Validate labels match
    assert np.array_equal(
        y_cnn, y_sae
    ), "Labels do not match between CNN and SAE caches!"

    classes = np.unique(y_cnn)
    if args.classes:
        classes = [c for c in classes if c in args.classes]
    logger.info(f"Classes to plot: {classes}")

    # Subsample for faster UMAP if needed
    indices = []
    for c in classes:
        c_idx = np.where(y_cnn == c)[0]
        if args.subsample > 0 and len(c_idx) > args.subsample:
            c_idx = np.random.choice(c_idx, args.subsample, replace=False)
        indices.extend(c_idx)

    indices = np.array(indices)
    np.random.shuffle(indices)

    X_cnn_sub = X_cnn[indices]
    X_sae_sub = X_sae[indices]
    y_sub = y_cnn[indices]

    logger.info(f"Data to embed: {X_cnn_sub.shape[0]} samples")

    # 2. Run UMAP
    logger.info("Running UMAP for CNN...")
    reducer_cnn = umap.UMAP(
        n_neighbors=args.n_neighbors, min_dist=args.min_dist, random_state=42
    )
    Z_cnn = reducer_cnn.fit_transform(X_cnn_sub)

    logger.info("Running UMAP for SAE...")
    reducer_sae = umap.UMAP(
        n_neighbors=args.n_neighbors, min_dist=args.min_dist, random_state=42
    )
    Z_sae = reducer_sae.fit_transform(X_sae_sub)

    # 3. Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for idx, c in enumerate(classes):
        mask = y_sub == c
        color = PALETTE[idx % len(PALETTE)]

        # Plot CNN
        axes[0].scatter(
            Z_cnn[mask, 0],
            Z_cnn[mask, 1],
            c=[color],
            label=f"Class {c}",
            alpha=0.6,
            s=5,
        )

        # Plot SAE
        axes[1].scatter(
            Z_sae[mask, 0],
            Z_sae[mask, 1],
            c=[color],
            label=f"Class {c}",
            alpha=0.6,
            s=5,
        )

    axes[0].set_title("CNN Features (UMAP)")
    axes[0].axis("off")

    axes[1].set_title("SAE Features (UMAP)")
    axes[1].axis("off")

    # Add legend to the right
    axes[1].legend(loc="center left", bbox_to_anchor=(1, 0.5), markerscale=3)

    plt.tight_layout()
    out_path = os.path.join(args.output_dir, "umap_cnn_vs_sae.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()
