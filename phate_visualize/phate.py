# ==============================================================================
# PHATE 2D Visualization of SAE Concept Activations
#
# Two modes:
#   A. Cache mode (recommended): reads features_cache.npz from extract_features.py
#      python -m kendall_correlation_coefficient.phate \
#          --features_cache /path/to/features_cache_xxx.npz \
#          --knn 5
#
#   B. Encoder mode (legacy): loads encoder + SAE + shard data
#      python -m kendall_correlation_coefficient.phate \
#          --sae_ckpt ... --save_dir ... --model_state_path ... --shard_root ...
# ==============================================================================

# DPT와 뉴런 개수가 다른 이유. DPT에서는 de_eval_split 이 있다. 그 뉴런이 mutation specific 하다는 것 등을 데이터 셋을 나눠서 그 데이터셋에서 선태하고 나머지 데이터 셋에서 DPT 하는등
# 근데 phate는 전체적으로 다 시각화하는거니까 de_eval_split이 필요가 없고 따라서 이걸 안하면 필터링 되는 뉴런 개수가 달라질 수 있다.

## sclens 등 근거로 gap_l2_norm 등 하는데. l2 norm 이후로 min cv de filter 등 작동된다.



# phate.py의 PAGA는 Leiden clustering 없이 superclass 기준으로 합니다: superclass 기준으로 control GBA가 연결되어 있는지 아닌지 등을 판단.

import os
import argparse
import numpy as np
from collections import defaultdict

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm
except ImportError:
    torch = None

import matplotlib
import sys
_IN_COLAB = "google.colab" in sys.modules
if not _IN_COLAB:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import phate

from sae_project.step02_logging_utils import get_logger, SUPERCLASS_MAP

logger = get_logger("phate_vis")


# ==============================================================================
# Argument Parser
# ==============================================================================
def get_args():
    p = argparse.ArgumentParser(description="PHATE 2D visualization of SAE concepts")

    # --- Cache mode (recommended) ---
    p.add_argument("--features_cache", type=str, default="",
                   help="Path to features_cache.npz from extract_features.py. "
                        "If given, skips encoder/SAE/shard loading entirely.")

    # --- Encoder mode (legacy, only if --features_cache not given) ---
    p.add_argument("--sae_ckpt", type=str, default="",
                   help="Path to trained SAE checkpoint (.pt)")
    p.add_argument("--save_dir", type=str, default="",
                   help="Model output dir (contains train/val/test_split.csv)")
    p.add_argument("--model_state_path", type=str, default="",
                   help="Path to best_model.pt")
    p.add_argument("--shard_root", type=str, default="",
                   help="Path to wds_shards_tar")

    # Output
    p.add_argument("--output_dir", type=str, default="",
                   help="Directory to save PNG (default: same dir as cache/SAE)")

    # Dead neuron
    p.add_argument("--dead_threshold", type=float, default=1e-5,
                   help="Neurons with usage_ema < this are dead")
    p.add_argument("--gap_l2_norm", action="store_true",
                   help="Apply L2 normalization to feature vectors (useful for GAP)")

    # Neuron filtering (can combine: --filter_mode cv de)
    p.add_argument("--filter_mode", type=str, nargs="+", default=["none"],
                   help="Neuron filtering, applied sequentially. "
                        "e.g. '--filter_mode cv de' applies CV first, then DE")
    p.add_argument("--max_gini", type=float, default=0.75,
                   help="Max Gini impurity (only with --filter_mode gini)")
    p.add_argument("--min_cv", type=float, default=0.0,
                   help="Min CV to keep (only with --filter_mode cv)")
    p.add_argument("--de_adj_p", type=float, default=0.05,
                   help="BH-adjusted p-value threshold (--filter_mode de)")
    p.add_argument("--de_min_log2fc", type=float, default=0.0,
                   help="Min |log2FC| for DE fislter")
    p.add_argument("--de_mutation", type=str, default="",
                   help="Which mutation for DE filter (e.g. GBA). "
                        "Required with --filter_mode de --de_mode per_mut")
    p.add_argument("--de_mode", type=str, default="union",
                   choices=["union", "per_mut"],
                   help="DE mode: 'union' = same as dpt_kendall (all muts + Ctrl-high), "
                        "'per_mut' = separate PHATE per mutation")

    # PAGA
    p.add_argument("--paga", action="store_true",
                   help="Run PAGA connectivity analysis before PHATE")
    p.add_argument("--paga_n_neighbors", type=int, default=30,
                   help="n_neighbors for PAGA kNN graph")
    p.add_argument("--paga_n_pcs", type=int, default=50,
                   help="Number of PCs for PAGA")

    # Normalization (same as dpt_kendall.py)
    p.add_argument("--norm", type=str, default="none",
                   choices=["none", "log", "median", "std",
                            "log_median", "log_std", "log_IQR", "IQR"],
                   help="Feature normalization before PHATE")

    # Feature extraction (encoder mode only)
    p.add_argument("--restore_token_norm", action="store_true",
                   help="Multiply SAE activations by original per-token L2 norms")

    # Sampling
    p.add_argument("--samples_per_class", type=int, default=10000,
                   help="Max images to sample per class")
    p.add_argument("--seed", type=int, default=42)

    # PHATE
    p.add_argument("--knn", type=int, default=5,
                   help="PHATE k-nearest neighbors")
    p.add_argument("--t", type=str, default="auto",
                   help="PHATE diffusion time. 'auto' uses von Neumann entropy.")
    p.add_argument("--n_components", type=int, default=2,
                   help="PHATE output dimensions")
    p.add_argument("--decay", type=int, default=40,
                   help="PHATE alpha decay parameter")
    p.add_argument("--knn_dist", type=str, default="euclidean",
                   choices=["cosine", "euclidean", "correlation"],
                   help="Distance metric for PHATE kNN graph")
    p.add_argument("--n_pca", type=int, default=100,
                   help="PCA dimensions before PHATE")
    p.add_argument("--n_jobs", type=int, default=-1,
                   help="Parallel jobs for PHATE")

    # Encoder architecture (encoder mode only)
    p.add_argument("--blocks", type=str, default="2,2,2,3")
    p.add_argument("--dilations", type=str, default="1,1,1,1")
    p.add_argument("--refine_blocks", type=int, default=1)
    p.add_argument("--ckpt_segments", type=int, default=0)
    p.add_argument("--embed_dim", type=int, default=512)
    p.add_argument("--proj_layers", type=int, default=2)
    p.add_argument("--proj_hidden", type=int, default=2048)
    p.add_argument("--proj_bn", action="store_true")
    p.add_argument("--proj_dropout", type=float, default=0.0)

    # Data (encoder mode only)
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)

    # Plot
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--point_size", type=float, default=3.0)
    p.add_argument("--alpha", type=float, default=0.5)

    return p.parse_args()


# ==============================================================================
# Feature Extraction (alive neurons only, standard mode)
# ==============================================================================
@torch.no_grad()
def extract_sae_gap_features(
    encoder,
    sae,
    loader,
    device,
    which_layer: str,
    alive_mask,
    restore_token_norm: bool = False,
) -> tuple:
    """
    Extract SAE GAP features for alive neurons only.

    Returns:
        X: (N, d_alive) numpy array of GAP features
        y: (N,) numpy array of labels
    """
    encoder.eval()
    sae.eval()

    alive_mask_dev = alive_mask.to(device)
    d_alive = int(alive_mask.sum().item())

    autocast_kwargs = dict(device_type="cuda", enabled=torch.cuda.is_available())
    if torch.cuda.is_available():
        autocast_kwargs["dtype"] = torch.bfloat16

    X_list, y_list = [], []

    for batch in tqdm(loader, desc="Extracting SAE features", leave=False):
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

        # GAP-scalar normalization (match SAE training)
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

        # Save per-token L2 norms before normalization
        token_l2_norms = flat_tokens.norm(dim=1, keepdim=True).clamp_min(1e-12)  # (N_tokens, 1)

        flat_tokens = F.normalize(flat_tokens, dim=1, eps=1e-12)

        # SAE forward in chunks
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

        # Optionally restore per-token L2 norms
        if restore_token_norm:
            acts = acts.float() * token_l2_norms  # (N_tokens, d_sae)
        else:
            acts = acts.float()

        # Pool over spatial tokens → image-level GAP
        H_W = fmap.shape[1] * fmap.shape[2]
        acts = acts.view(curr_batch_size, H_W, sae.d_sae)
        pooled = acts.mean(dim=1)  # (B, d_sae)
        pooled = F.normalize(pooled, dim=1)

        # Keep only alive neurons
        pooled_alive = pooled[:, alive_mask_dev].cpu().numpy()

        X_list.append(pooled_alive)
        y_list.extend(y.tolist())

    if len(X_list) == 0:
        return np.zeros((0, d_alive), dtype=np.float32), np.zeros(0, dtype=np.int64)

    X = np.concatenate(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)

    return X, y


# ==============================================================================
# Data Loader from val+test with balanced sampling
# ==============================================================================
def make_balanced_loader(args, refs, uid_to_refidx, samples_per_class: int, seed: int):
    """Load val+test, sample up to `samples_per_class` per class, return DataLoader."""
    from sae_project.step04_data_bank import (
        InMemoryTarBank, InMemorySixteenBitDataset, load_split_csv,
        seed_worker, collate_skip_none,
    )

    val_csv = os.path.join(args.save_dir, "val_split.csv")
    test_csv = os.path.join(args.save_dir, "test_split.csv")

    all_uids = []
    for csv_path in [val_csv, test_csv]:
        if os.path.exists(csv_path):
            all_uids.extend(load_split_csv(csv_path))
            logger.info(f"  Loaded {csv_path}: {len(load_split_csv(csv_path))} UIDs")
        else:
            logger.warning(f"  Not found: {csv_path}")

    if not all_uids:
        raise FileNotFoundError(f"No val/test CSVs found in {args.save_dir}")

    # Map UIDs → ref indices, get labels
    refidx_list = []
    for uid in all_uids:
        if uid in uid_to_refidx:
            refidx_list.append(uid_to_refidx[uid])
        else:
            logger.warning(f"  UID missing in shard: {uid}")

    # Group by class
    class_to_indices = defaultdict(list)
    for i, ridx in enumerate(refidx_list):
        label = int(refs[ridx].label)
        class_to_indices[label].append(i)

    # Sample balanced
    rng = np.random.default_rng(seed)
    selected_indices = []
    for cls in sorted(class_to_indices.keys()):
        cls_indices = class_to_indices[cls]
        n_avail = len(cls_indices)
        n_take = min(samples_per_class, n_avail)
        chosen = rng.choice(cls_indices, size=n_take, replace=False)
        selected_indices.extend(chosen.tolist())
        logger.info(f"  Class {cls}: {n_take}/{n_avail} images selected")

    # Re-map: selected_indices index into refidx_list
    selected_refidx = [refidx_list[i] for i in selected_indices]

    bank = InMemoryTarBank(refs, selected_refidx, args.img_size)
    ib = list(range(len(selected_refidx)))
    ds = InMemorySixteenBitDataset(bank, ib, args.img_size, augment=False)

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        collate_fn=collate_skip_none,
    )

    total = len(selected_indices)
    logger.info(f"  Total: {total} images loaded for PHATE")
    return loader


# ==============================================================================
# Plot PHATE
# ==============================================================================
CLASS_NAMES = {0: "Control", 1: "SNCA", 2: "GBA", 3: "LRRK2"}
SUPERCLASS_COLORS = {
    "Control": "#4C72B0",  # blue
    "SNCA": "#DD8452",     # orange
    "GBA": "#55A868",      # green
    "LRRK2": "#C44E52",    # red
}
CLASS_COLORS = {
    0: "#4C72B0",  # blue
    1: "#DD8452",  # orange
    2: "#55A868",  # green
    3: "#C44E52",  # red
}


def plot_phate(
    phate_coords: np.ndarray,
    labels: np.ndarray,
    title: str,
    output_path: str,
    point_size: float = 3.0,
    alpha: float = 0.5,
    dpi: int = 300,
    pca_variance_explained: float = 0.0,
    phate_t: int = 0,
    n_alive: int = 0,
    n_total: int = 0,
):
    """Save publication-ready PHATE 2D scatter plot colored by class."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    for cls in sorted(CLASS_NAMES.keys()):
        mask = labels == cls
        if mask.sum() == 0:
            continue
        n = int(mask.sum())
        ax.scatter(
            phate_coords[mask, 0],
            phate_coords[mask, 1],
            s=point_size,
            alpha=alpha,
            label=f"{CLASS_NAMES[cls]} (n={n:,})",
            c=CLASS_COLORS[cls],
            edgecolors="none",
            rasterized=True,
        )

    ax.set_xlabel("PHATE 1", fontsize=14)
    ax.set_ylabel("PHATE 2", fontsize=14)

    # Remove ticks, spines, grid for clean publication look
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_aspect("equal", adjustable="datalim")

    # Clean legend
    leg = ax.legend(
        loc="best",
        markerscale=4,
        fontsize=11,
        frameon=True,
        framealpha=0.9,
        edgecolor="0.8",
        handletextpad=0.5,
        borderpad=0.6,
    )
    for lh in leg.legend_handles:
        lh.set_alpha(1.0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    pdf_path = output_path.rsplit(".", 1)[0] + ".pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    svg_path = output_path.rsplit(".", 1)[0] + ".svg"
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved PHATE plot: {output_path}")
    logger.info(f"  Saved PHATE PDF:  {pdf_path}")
    logger.info(f"  Saved PHATE SVG:  {svg_path}")


# ==============================================================================
# Plot PHATE (superclass string labels version) — Publication-ready
# ==============================================================================
def plot_phate_superclass(
    phate_coords: np.ndarray,
    superclasses: list,
    title: str,
    output_path: str,
    point_size: float = 3.0,
    alpha: float = 0.5,
    dpi: int = 300,
    info_text: str = "",
):
    """Save publication-ready PHATE 2D scatter plot colored by superclass."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    superclasses_arr = np.array(superclasses)

    # Plot order: Control first (background), then mutations
    plot_order = ["Control", "SNCA", "GBA", "LRRK2"]
    for cls in plot_order:
        mask = superclasses_arr == cls
        if mask.sum() == 0:
            continue
        n = int(mask.sum())
        ax.scatter(
            phate_coords[mask, 0],
            phate_coords[mask, 1],
            s=point_size,
            alpha=alpha,
            label=f"{cls} (n={n:,})",
            c=SUPERCLASS_COLORS.get(cls, "gray"),
            edgecolors="none",
            rasterized=True,
        )

    ax.set_xlabel("PHATE 1", fontsize=14)
    ax.set_ylabel("PHATE 2", fontsize=14)

    # Remove ticks, spines, grid for clean publication look
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_aspect("equal", adjustable="datalim")

    # Clean legend
    leg = ax.legend(
        loc="best",
        markerscale=4,
        fontsize=11,
        frameon=True,
        framealpha=0.9,
        edgecolor="0.8",
        handletextpad=0.5,
        borderpad=0.6,
    )
    # Make legend markers fully opaque
    for lh in leg.legend_handles:
        lh.set_alpha(1.0)

    fig.tight_layout()
    # Save PNG
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    # Save PDF (vector) for publication
    pdf_path = output_path.rsplit(".", 1)[0] + ".pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    # Save SVG for Illustrator editing
    svg_path = output_path.rsplit(".", 1)[0] + ".svg"
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved PHATE plot: {output_path}")
    logger.info(f"  Saved PHATE PDF:  {pdf_path}")
    logger.info(f"  Saved PHATE SVG:  {svg_path}")


# ==============================================================================
# Main
# ==============================================================================
def main():
    args = get_args()
    import gc

    np.random.seed(args.seed)

    # ══════════════════════════════════════════════════════════════════════
    # Mode A: Load from features_cache.npz (no GPU needed)
    # ══════════════════════════════════════════════════════════════════════
    if args.features_cache:
        from kendall_correlation_coefficient.dpt_kendall import load_features_cache

        logger.info(f"\n{'='*60}")
        logger.info("Mode: features_cache (no GPU/encoder needed)")
        logger.info(f"Cache: {args.features_cache}")

        # Auto-detect SAE vs GAP cache
        data = np.load(args.features_cache, allow_pickle=True)
        cache_keys = list(data.keys())
        logger.info(f"  Cache keys: {cache_keys}")

        if "X_gap" in data:
            X = data["X_gap"]
            lines = data["lines"].astype(str) if data["lines"].dtype.kind != 'U' else data["lines"]
            uids = data["uids"].astype(str) if data["uids"].dtype.kind != 'U' else data["uids"]
            which_layer = str(data["which_layer"])
            alive_info = f"GAP raw {X.shape}"
            logger.info(f"  Detected CNN GAP cache: {X.shape}")
        elif "X_all" in data:
            X, y, lines, uids, which_layer, alive_info = load_features_cache(
                args.features_cache, args.dead_threshold
            )
        else:
            raise ValueError(f"Unknown cache format. Keys: {cache_keys}")

        # Optional: L2 normalize
        if args.gap_l2_norm:
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1e-12, norms)
            X = X / norms
            alive_info += " + L2norm"
            logger.info(f"  Applied L2 normalization")

        superclasses = [SUPERCLASS_MAP.get(ln, ln) for ln in lines]
        n_alive = X.shape[1]

        # Report (before filtering)
        unique_sc, sc_counts = np.unique(superclasses, return_counts=True)
        logger.info(f"  Features: {X.shape} ({alive_info})")
        logger.info(f"  Classes: {dict(zip(unique_sc, sc_counts))}")

        # ── Neuron filtering (sequential, on FULL data) ──────────────────
        filter_steps = []
        for fm in args.filter_mode:
            if fm == "none":
                continue

            n_before = X.shape[1]

            if fm == "gini":
                from kendall_correlation_coefficient.dpt_kendall import compute_gini_impurity
                gini = compute_gini_impurity(X, superclasses)
                mask = gini <= args.max_gini
                X = X[:, mask]
                step_info = f"gini≤{args.max_gini:.2f}: {n_before}→{X.shape[1]}"

            elif fm == "cv":
                from kendall_correlation_coefficient.dpt_kendall import compute_cv_per_neuron
                cv = compute_cv_per_neuron(X, superclasses)
                mask = cv >= args.min_cv
                X = X[:, mask]
                step_info = f"cv≥{args.min_cv:.2f}: {n_before}→{X.shape[1]}"

            elif fm == "de":
                # DE is handled per-mutation below (not here)
                step_info = "DE (per-mutation, see below)"

            else:
                logger.warning(f"  Unknown filter: {fm}, skipping")
                continue

            filter_steps.append(step_info)
            logger.info(f"  Filter [{fm}]: {step_info}")

        filter_info = " → ".join(filter_steps) if filter_steps else "none"
        if not filter_steps:
            logger.info("  Filter: none")

        n_alive = X.shape[1]  # update after filtering

        # ── Subsample per class (AFTER neuron filtering) ─────────────────
        spc = args.samples_per_class
        if spc > 0 and X.shape[0] > spc * 4:
            rng = np.random.RandomState(args.seed)
            superclasses_arr_tmp = np.array(superclasses)
            keep_indices = []
            for cls in np.unique(superclasses_arr_tmp):
                cls_idx = np.where(superclasses_arr_tmp == cls)[0]
                n_take = min(spc, len(cls_idx))
                chosen = rng.choice(cls_idx, size=n_take, replace=False)
                keep_indices.extend(chosen.tolist())
                logger.info(f"  Subsample {cls}: {len(cls_idx)} → {n_take}")
            keep_indices = sorted(keep_indices)
            X = X[keep_indices]
            superclasses = [superclasses[i] for i in keep_indices]
            logger.info(f"  After subsampling: {X.shape[0]} samples")

        # ── PAGA connectivity analysis ───────────────────────────────────
        if args.paga:
            import scanpy as sc
            logger.info(f"\n{'='*60}")
            logger.info(f"PAGA connectivity (n_neighbors={args.paga_n_neighbors}, "
                         f"n_pcs={args.paga_n_pcs})")

            adata_paga = sc.AnnData(X.astype(np.float32))
            adata_paga.obs["superclass"] = superclasses
            adata_paga.obs["superclass"] = adata_paga.obs["superclass"].astype("category")

            n_pcs_paga = min(args.paga_n_pcs, X.shape[1] - 1)
            sc.pp.pca(adata_paga, n_comps=n_pcs_paga)
            sc.pp.neighbors(adata_paga, n_neighbors=args.paga_n_neighbors,
                           n_pcs=n_pcs_paga)
            sc.tl.paga(adata_paga, groups="superclass")

            # Log connectivity matrix
            conn = adata_paga.uns["paga"]["connectivities"].toarray()
            groups = adata_paga.obs["superclass"].cat.categories.tolist()
            logger.info(f"  PAGA groups: {groups}")
            logger.info(f"  PAGA connectivity matrix:")
            for i, gi in enumerate(groups):
                for j, gj in enumerate(groups):
                    if j > i:
                        logger.info(f"    {gi} ↔ {gj}: {conn[i,j]:.4f}")

            # Plot PAGA
            fig_paga, ax_paga = plt.subplots(1, 1, figsize=(6, 6))
            sc.pl.paga(
                adata_paga,
                color="superclass",
                ax=ax_paga,
                show=False,
                title=f"PAGA – {os.path.basename(args.features_cache)}\n"
                      f"filter={filter_info}",
                fontsize=10,
            )
            plt.show()
            plt.close(fig_paga)

            del adata_paga
            logger.info("  PAGA analysis complete")

        # Output directory
        if args.output_dir:
            output_dir = args.output_dir
        else:
            output_dir = os.path.join(os.path.dirname(args.features_cache), "phate")
        os.makedirs(output_dir, exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════
    # Mode B: Encoder + SAE extraction (legacy)
    # ══════════════════════════════════════════════════════════════════════
    else:
        if not args.sae_ckpt:
            raise ValueError("Must provide either --features_cache or --sae_ckpt")

        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader
        from sae_project.step03_data_shards import load_all_sample_refs, build_uid_to_refidx
        from sae_project.step04_data_bank import (
            InMemoryTarBank, InMemorySixteenBitDataset, load_split_csv,
            seed_worker, collate_skip_none,
        )
        from sae_project.step05_model_encoder import (
            SupMoCoModel, parse_int_list, renorm_unit_per_out_channel_,
            robust_load_state_dict,
        )
        from sae_project.step06_gated_sae import GatedSAE

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {device}")

        # Load SAE
        logger.info(f"\n{'='*60}")
        logger.info("Loading SAE checkpoint")
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
        d_sae = sae.d_sae

        usage_ema = sae.usage_ema.cpu()
        alive_mask = usage_ema >= args.dead_threshold
        n_alive = int(alive_mask.sum().item())
        logger.info(f"Alive: {n_alive}/{d_sae}")

        # Load encoder
        blocks = parse_int_list(args.blocks, 4)
        dilations = parse_int_list(args.dilations, 4)
        model = SupMoCoModel(
            embed_dim=args.embed_dim, blocks=blocks, dilations=dilations,
            refine_blocks=args.refine_blocks, ckpt_segments=args.ckpt_segments,
            proj_layers=args.proj_layers, proj_hidden=args.proj_hidden,
            proj_bn=args.proj_bn, proj_dropout=args.proj_dropout,
        )
        sd = torch.load(args.model_state_path, map_location="cpu", weights_only=False)
        robust_load_state_dict(model, sd, strict=True)
        encoder = model.encoder
        renorm_unit_per_out_channel_(encoder)
        encoder.to(device).eval()
        del model, sd

        # Load data
        refs = load_all_sample_refs(args.shard_root)
        uid_to_refidx = build_uid_to_refidx(refs)
        loader = make_balanced_loader(
            args, refs, uid_to_refidx,
            samples_per_class=args.samples_per_class, seed=args.seed,
        )

        # Extract features
        X, y = extract_sae_gap_features(
            encoder, sae, loader, device, which_layer, alive_mask,
            restore_token_norm=args.restore_token_norm,
        )
        superclasses = [CLASS_NAMES.get(int(yi), str(yi)) for yi in y]
        filter_info = "N/A (encoder mode)"
        logger.info(f"Features: {X.shape}")

        del encoder, sae, loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if args.output_dir:
            output_dir = args.output_dir
        else:
            output_dir = os.path.dirname(args.sae_ckpt)
        os.makedirs(output_dir, exist_ok=True)

    # ── Normalization (for non-DE mode; DE normalizes per-subset) ────────
    has_de = "de" in args.filter_mode
    if not has_de and args.norm != "none":
        from kendall_correlation_coefficient.dpt_kendall import apply_normalization
        logger.info(f"  Normalization: {args.norm}")
        X = apply_normalization(X, args.norm)
    elif not has_de:
        logger.info("  Normalization: none")

    # ── Helper: run PHATE + plot ─────────────────────────────────────────
    cache_basename = os.path.splitext(
        os.path.basename(args.features_cache or args.sae_ckpt)
    )[0]

    def _run_phate_and_plot(X_in, superclasses_in, suffix="", extra_info=""):
        n_pca = min(args.n_pca, X_in.shape[1])
        t_value = args.t if args.t == "auto" else int(args.t)

        logger.info(f"\n{'='*60}")
        logger.info(f"PHATE{suffix}: knn={args.knn}, t={args.t}, "
                     f"dist={args.knn_dist}, norm={args.norm}, n_pca={n_pca}")
        logger.info(f"Input: {X_in.shape[0]} samples × {X_in.shape[1]} features")

        phate_op = phate.PHATE(
            n_components=args.n_components,
            knn=args.knn,
            t=t_value,
            decay=args.decay,
            knn_dist=args.knn_dist,
            n_pca=n_pca,
            n_jobs=args.n_jobs,
            random_state=args.seed,
            verbose=1,
        )
        X_phate = phate_op.fit_transform(X_in)

        actual_t = phate_op.t
        if hasattr(phate_op, "optimal_t"):
            actual_t = phate_op.optimal_t
        logger.info(f"PHATE done. t={actual_t}, output: {X_phate.shape}")

        clean_suffix = suffix.replace(" ", "_")
        png_name = f"phate_{cache_basename}{clean_suffix}_knn{args.knn}_t{actual_t}.png"
        output_path = os.path.join(output_dir, png_name)

        title = ""  # publication-ready: no title

        plot_phate_superclass(
            X_phate, superclasses_in,
            title=title, output_path=output_path,
            point_size=args.point_size, alpha=args.alpha, dpi=args.dpi,
        )

        npz_path = output_path.replace(".png", "_coords.npz")
        np.savez_compressed(
            npz_path,
            phate_coords=X_phate,
            superclasses=np.array(superclasses_in),
            phate_t=actual_t,
            knn=args.knn,
        )
        logger.info(f"Saved coordinates: {npz_path}")
        return X_phate, actual_t

    # ══════════════════════════════════════════════════════════════════════
    # DE mode
    # ══════════════════════════════════════════════════════════════════════
    if has_de:
        from kendall_correlation_coefficient.dpt_kendall import (
            compute_de_neurons, apply_normalization,
        )
        superclasses_arr = np.array(superclasses)
        mutations = ["SNCA", "GBA", "LRRK2"]
        de_mode = getattr(args, "de_mode", "union")

        if de_mode == "union":
            # ── DE union (same as dpt_kendall) ──────────────────────────
            logger.info(f"\n{'='*60}")
            logger.info("DE union mode (matching dpt_kendall pipeline)")
            logger.info("=" * 60)

            de_masks = []
            # Per-mutation DE (both directions)
            for mut in mutations:
                de_result = compute_de_neurons(
                    X, superclasses, mut,
                    adj_p_threshold=args.de_adj_p,
                    min_log2fc=args.de_min_log2fc,
                )
                de_masks.append(de_result["mask"])
                logger.info(f"    {mut}: {de_result['n_selected']} DE neurons")

            # Control vs AllMut — Control-high (log2fc < 0)
            sc_allm = [("AllMut" if s != "Control" else "Control") for s in superclasses]
            de_ctrl = compute_de_neurons(
                X, sc_allm, "AllMut",
                adj_p_threshold=args.de_adj_p,
                min_log2fc=args.de_min_log2fc,
            )
            ctrl_high_mask = de_ctrl["mask"] & (de_ctrl["log2fc"] < 0)
            de_masks.append(ctrl_high_mask)
            logger.info(f"    Ctrl-high (vs AllMut): {int(ctrl_high_mask.sum())} neurons")

            # Union
            union_mask = np.zeros(X.shape[1], dtype=bool)
            for m in de_masks:
                union_mask |= m
            n_union = int(union_mask.sum())
            logger.info(f"  DE union: {n_union}/{X.shape[1]} neurons")

            if n_union < 3:
                logger.warning("  Too few DE union neurons — running without DE")
                _run_phate_and_plot(X, superclasses)
            else:
                X_de = X[:, union_mask]
                # Normalize after DE selection (same as DPT)
                if args.norm != "none":
                    X_de = apply_normalization(X_de, args.norm)
                    logger.info(f"  Normalization: {args.norm}")

                extra = f"DE union: {n_union} neurons\n"
                _run_phate_and_plot(
                    X_de, superclasses,
                    suffix=f"_DE_union",
                    extra_info=extra,
                )

        else:
            # ── Per-mutation DE (legacy) ────────────────────────────────
            for mut in mutations:
                logger.info(f"\n{'='*60}")
                logger.info(f"DE PHATE: {mut} vs Control")
                logger.info("=" * 60)

                de_result = compute_de_neurons(
                    X, superclasses, mut,
                    adj_p_threshold=args.de_adj_p,
                    min_log2fc=args.de_min_log2fc,
                )
                n_de = de_result["n_selected"]
                if n_de < 3:
                    logger.warning(f"  Only {n_de} DE neurons for {mut} — skipping")
                    continue

                keep = (superclasses_arr == "Control") | (superclasses_arr == mut)
                X_sub = X[keep][:, de_result["mask"]]
                sc_sub = superclasses_arr[keep].tolist()

                if args.norm != "none":
                    X_sub = apply_normalization(X_sub, args.norm)

                extra = f"DE neurons: {n_de}\n"
                _run_phate_and_plot(
                    X_sub, sc_sub,
                    suffix=f"_DE_{mut}",
                    extra_info=extra,
                )

    # ══════════════════════════════════════════════════════════════════════
    # Non-DE mode: global PHATE (all 4 classes)
    # ══════════════════════════════════════════════════════════════════════
    else:
        _run_phate_and_plot(X, superclasses)

    logger.info(f"\n{'='*60}")
    logger.info("PHATE visualization complete!")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()

