# ==============================================================================
# SAE Evaluation Utilities
# Extracted from step09_train_gated_sae.py for modularity.
# Contains: LinearProbe, extract_sae_repr, train_linear_probe,
#           compute_effective_rank, evaluate_concepts_for_sae, DummyTrainer
# ==============================================================================
import os
import csv
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from sae_project.step02_logging_utils import get_logger
from sae_project.step06_gated_sae import GatedSAE

logger = get_logger("sae_eval")


# ==============================================================================
# Linear Probe
# ==============================================================================

class LinearProbe(nn.Module):
    """Simple linear probe for classification."""
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.net = nn.Linear(d_in, d_out, bias=False)

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def extract_sae_repr_for_probe(
    encoder,
    sae: GatedSAE,
    loader: DataLoader,
    device: torch.device,
    which_layer: str,
    token_l2_norm: bool = True,
    strength_weighting: bool = False,
    dead_threshold: float = 1e-6,
) -> tuple:
    """Extract SAE representations for linear probe evaluation (Vectorized)."""
    encoder.eval()
    sae.eval()

    usage = sae.usage_ema.cpu()
    alive_mask = usage >= dead_threshold
    d_alive = int(alive_mask.sum().item())

    if d_alive == 0:
        logger.warning("All neurons are dead! Using all neurons.")
        alive_mask = torch.ones(sae.d_sae, dtype=torch.bool)
        d_alive = sae.d_sae

    alive_mask_dev = alive_mask.to(device)

    autocast_kwargs = dict(device_type="cuda", enabled=torch.cuda.is_available())
    if torch.cuda.is_available():
        autocast_kwargs["dtype"] = torch.bfloat16

    X_list, y_list = [], []

    for batch in tqdm(loader, desc="Extracting SAE repr", leave=False):
        if batch is None:
            continue
        x_cpu, y_cpu, *_ = batch
        if x_cpu.numel() < 1:
            continue

        x = x_cpu.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        y = y_cpu

        with torch.amp.autocast(**autocast_kwargs):
            fmap = encoder.forward_feature_maps(x, which=which_layer)

        curr_batch_size = fmap.size(0)
        norm_mode = getattr(sae, "token_norm_mode", "gap-scalar")
        if norm_mode == "gap-scalar":
            gap = fmap.mean(dim=(2, 3))
            gap_norm = gap.norm(dim=1, keepdim=True).view(curr_batch_size, 1, 1, 1).clamp_min(1e-12)
            fmap = fmap / gap_norm

        fmap = fmap.permute(0, 2, 3, 1).contiguous()
        C = fmap.shape[-1]

        flat_tokens = fmap.view(-1, C)
        flat_tokens = flat_tokens - flat_tokens.mean(dim=0, keepdim=True)

        tok_norm = None
        if strength_weighting:
            tok_norm = flat_tokens.norm(dim=1, keepdim=True).clamp_min(1e-12)

        flat_tokens = F.normalize(flat_tokens, dim=1, eps=1e-12)

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

        acts = acts.float()
        if strength_weighting:
            acts = acts * tok_norm

        acts = acts.view(curr_batch_size, -1, sae.d_sae)
        pooled = acts.float().mean(dim=1)
        pooled = F.normalize(pooled, dim=1)
        pooled_alive = pooled[:, alive_mask_dev].cpu().numpy()

        X_list.append(pooled_alive)
        y_list.extend(y.tolist())

    if len(X_list) == 0:
        return np.zeros((0, d_alive), dtype=np.float32), np.zeros(0, dtype=np.int64), d_alive

    X = np.concatenate(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)

    return X, y, d_alive


def train_linear_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int = 4,
    epochs: int = 50,
    lr: float = 0.1,
    device: torch.device = None,
    batch_size: int = 256,
    balanced_training: bool = True,
    normalize_repr: bool = False,
    verbose: bool = True,
) -> dict:
    """Train linear probe with balanced class sampling and mini-batch updates."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    original_dim = X_train.shape[1]

    feat_std = X_train.std(axis=0)
    alive_mask = feat_std >= 1e-8
    n_alive = int(alive_mask.sum())

    if n_alive == 0:
        return {"train_acc": 0.0, "test_acc": 0.0, "d_probe": 0}

    X_train = X_train[:, alive_mask]
    X_test = X_test[:, alive_mask]

    if normalize_repr:
        mean = X_train.mean(axis=0, keepdims=True)
        std = X_train.std(axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

    if verbose:
        logger.info(f"  [Probe] dim: {original_dim} -> {n_alive} (removed {original_dim - n_alive} zero-var)")
        logger.info(f"  [Probe] Train: {len(y_train)}, Test: {len(y_test)}")

    rng = np.random.default_rng(42)
    class_indices = {c: np.where(y_train == c)[0] for c in range(num_classes)}
    min_class_count = min(len(v) for v in class_indices.values())
    samples_per_class = min_class_count

    if balanced_training and verbose:
        logger.info(f"  [Probe] Balanced: {samples_per_class} samples/class x {num_classes} = {samples_per_class * num_classes}/epoch")

    probe = LinearProbe(n_alive, num_classes).to(device)
    import torch.optim as optim
    optimizer = optim.SGD(probe.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    probe.train()
    for epoch in range(epochs):
        if balanced_training:
            epoch_indices = []
            for c in range(num_classes):
                c_idx = class_indices[c]
                sampled = rng.choice(c_idx, size=samples_per_class, replace=False)
                epoch_indices.extend(sampled.tolist())
            epoch_indices = np.array(epoch_indices)
            rng.shuffle(epoch_indices)
        else:
            epoch_indices = np.arange(len(y_train))
            rng.shuffle(epoch_indices)

        for s in range(0, len(epoch_indices), batch_size):
            ii = epoch_indices[s:s+batch_size]
            xb = torch.from_numpy(X_train[ii]).float().to(device)
            yb = torch.from_numpy(y_train[ii]).long().to(device)

            optimizer.zero_grad()
            logits = probe(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    probe.eval()
    with torch.no_grad():
        if balanced_training:
            eval_indices = []
            for c in range(num_classes):
                c_idx = class_indices[c]
                sampled = rng.choice(c_idx, size=min(samples_per_class, len(c_idx)), replace=False)
                eval_indices.extend(sampled.tolist())
            eval_indices = np.array(eval_indices)
            X_tr_eval = torch.from_numpy(X_train[eval_indices]).float().to(device)
            y_tr_eval = torch.from_numpy(y_train[eval_indices]).long().to(device)
        else:
            X_tr_eval = torch.from_numpy(X_train).float().to(device)
            y_tr_eval = torch.from_numpy(y_train).long().to(device)

        train_pred = probe(X_tr_eval).argmax(dim=1)
        train_acc = (train_pred == y_tr_eval).float().mean().item()

        X_te = torch.from_numpy(X_test).float().to(device)
        y_te = torch.from_numpy(y_test).long().to(device)
        test_pred = probe(X_te).argmax(dim=1)
        test_acc = (test_pred == y_te).float().mean().item()

        per_class_acc = {}
        for c in range(num_classes):
            mask = y_te == c
            if mask.sum() > 0:
                per_class_acc[c] = (test_pred[mask] == c).float().mean().item()

    if verbose:
        logger.info(f"  [Probe] train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")
        logger.info(f"  [Probe] Per-class: {per_class_acc}")

    return {
        "train_acc": float(train_acc),
        "test_acc": float(test_acc),
        "d_probe": n_alive,
        "per_class_acc": per_class_acc,
    }


# ==============================================================================
# Effective Rank
# ==============================================================================

@torch.no_grad()
def compute_effective_rank(sae: GatedSAE, dead_threshold: float = 1e-6) -> dict:
    """Compute effective rank of alive concept weights using SVD entropy."""
    W_dec = sae.W_dec.data.float().cpu()
    usage = sae.usage_ema.cpu()
    alive_mask = usage >= dead_threshold
    d_alive = int(alive_mask.sum().item())

    if d_alive < 2:
        return {"effective_rank": 1.0, "normalized_effective_rank": 1.0, "d_alive": d_alive}

    W_alive = W_dec[alive_mask]
    U, S, Vh = torch.linalg.svd(W_alive, full_matrices=False)
    S = S.clamp(min=1e-12)
    p = S / S.sum()
    entropy = -torch.sum(p * torch.log(p)).item()
    effective_rank = np.exp(entropy)
    max_rank = min(d_alive, W_alive.shape[1])
    normalized_effective_rank = effective_rank / max_rank

    return {
        "effective_rank": float(effective_rank),
        "normalized_effective_rank": float(normalized_effective_rank),
        "d_alive": d_alive,
    }


# ==============================================================================
# Full Concept Evaluation (purity, entropy, GAP CSV, dual probe)
# ==============================================================================

class DummyTrainer:
    """Minimal trainer-like object for evaluation."""
    def __init__(self, encoder, sae, device, best_fvu=0.0):
        self.encoder = encoder
        self.sae = sae
        self.device = device
        self.best_fvu = best_fvu


def evaluate_concepts_for_sae(
    trainer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    args,
    exp_config: dict,
) -> dict:
    """Evaluate concept metrics and linear probe accuracy (Vectorized).
    Uses val+test combined for evaluation metrics."""
    device = trainer.device
    encoder = trainer.encoder
    sae = trainer.sae

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    encoder.eval()
    sae.eval()

    NUM_CLASSES = 4
    d_sae = sae.d_sae

    concept_class_counts = torch.zeros((d_sae, NUM_CLASSES), dtype=torch.float32, device=device)
    class_gap_sum = torch.zeros((NUM_CLASSES, d_sae), dtype=torch.float32, device=device)
    class_img_count = torch.zeros(NUM_CLASSES, dtype=torch.long, device=device)
    class_active_img_count = torch.zeros((NUM_CLASSES, d_sae), dtype=torch.long, device=device)

    all_token_counts_list = []
    image_idx = 0
    total_tokens = 0
    active_tokens = 0

    autocast_kwargs = dict(device_type="cuda", enabled=torch.cuda.is_available())
    if torch.cuda.is_available():
        autocast_kwargs["dtype"] = torch.bfloat16

    eval_loaders = []
    if val_loader is not None:
        eval_loaders.append(val_loader)
    if test_loader is not None:
        eval_loaders.append(test_loader)

    total_eval_images = sum(len(loader.dataset) for loader in eval_loaders)
    logger.info(f"[Eval] Using {len(eval_loaders)} loaders: {total_eval_images} images (val+test)")

    with torch.no_grad():
      for eval_loader in eval_loaders:
        for batch in tqdm(eval_loader, desc="Evaluating concepts", leave=False):
            if batch is None:
                continue
            x_cpu, y_cpu, *_ = batch
            if x_cpu.numel() < 1:
                continue

            x = x_cpu.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            y = y_cpu.to(device)

            with torch.amp.autocast(**autocast_kwargs):
                fmap = encoder.forward_feature_maps(x, which=args.which_layer)

            curr_batch_size = fmap.size(0)
            if getattr(args, "token_norm_mode", "gap-scalar") == "gap-scalar":
                gap = fmap.mean(dim=(2, 3))
                gap_norm = gap.norm(dim=1, keepdim=True).view(curr_batch_size, 1, 1, 1).clamp_min(1e-12)
                fmap = fmap / gap_norm

            fmap = fmap.permute(0, 2, 3, 1).contiguous()
            C = fmap.shape[-1]

            flat_tokens = fmap.view(-1, C)
            flat_tokens = flat_tokens - flat_tokens.mean(dim=0, keepdim=True)
            flat_tokens = F.normalize(flat_tokens, dim=1, eps=1e-12)

            tokens = flat_tokens.view(curr_batch_size, -1, C)
            num_tokens_per_img = tokens.shape[1]

            token_batch_size = 8192
            num_flat_tokens = flat_tokens.size(0)

            image_act_sums = torch.zeros((curr_batch_size, d_sae), device=device, dtype=torch.float32)
            batch_active_token_count = 0
            batch_total_token_count = 0
            batch_token_counts_list = []

            for start in range(0, num_flat_tokens, token_batch_size):
                end = min(start + token_batch_size, num_flat_tokens)
                chunk = flat_tokens[start:end]

                with torch.amp.autocast(**autocast_kwargs):
                    _, chunk_acts, _, _, _ = sae(chunk)

                chunk_acts = chunk_acts.float()
                token_start_idx = start
                token_end_idx = end

                chunk_active = (chunk_acts > 0).sum(dim=1)
                batch_token_counts_list.append(chunk_active.cpu())
                batch_active_token_count += (chunk_active > 0).sum().item()
                batch_total_token_count += chunk_active.size(0)

                for i in range(curr_batch_size):
                    img_start = i * num_tokens_per_img
                    img_end = (i + 1) * num_tokens_per_img
                    rel_start = max(0, img_start - token_start_idx)
                    rel_end = min(end - start, img_end - token_start_idx)
                    if rel_start < rel_end and img_start < token_end_idx and img_end > token_start_idx:
                        image_act_sums[i] += chunk_acts[rel_start:rel_end].sum(dim=0)

                del chunk_acts

            total_tokens += batch_total_token_count
            active_tokens += batch_active_token_count

            if batch_token_counts_list:
                all_token_counts_list.append(torch.cat(batch_token_counts_list))

            for c in range(NUM_CLASSES):
                class_mask = (y == c)
                if class_mask.any():
                    class_act_sums = image_act_sums[class_mask].sum(dim=0)
                    concept_class_counts[:, c] += class_act_sums
                    class_gap_sum[c] += image_act_sums[class_mask].sum(dim=0)
                    class_img_count[c] += class_mask.sum()
                    active_per_concept = (image_act_sums[class_mask] > 0).long().sum(dim=0)
                    class_active_img_count[c] += active_per_concept

            del image_act_sums, flat_tokens, fmap
            image_idx += curr_batch_size

    # --- Process Concept Metrics (CPU) ---
    concept_class_counts_cpu = concept_class_counts.cpu().numpy()
    total_activations = concept_class_counts_cpu.sum(axis=1)

    purity_list = []
    entropy_list = []
    class_specific_breakdown = {0: 0, 1: 0, 2: 0, 3: 0}
    high_purity_count = 0
    low_entropy_count = 0
    very_low_entropy_count = 0

    valid_mask = total_activations >= 5
    if valid_mask.any():
        valid_counts = concept_class_counts_cpu[valid_mask]
        valid_totals = total_activations[valid_mask]

        max_counts = valid_counts.max(axis=1)
        max_classes = valid_counts.argmax(axis=1)
        purities = max_counts / valid_totals
        purity_list = purities.tolist()

        high_purity_mask = purities > 0.9
        high_purity_count = int(high_purity_mask.sum())
        hp_classes = max_classes[high_purity_mask]
        for c in range(NUM_CLASSES):
            class_specific_breakdown[c] = int((hp_classes == c).sum())

        probs = valid_counts / valid_totals[:, None]
        probs_safe = np.maximum(probs, 1e-10)
        entropies = -np.sum(probs * np.log2(probs_safe), axis=1)
        entropy_list = entropies.tolist()

        low_entropy_count = int((entropies < 1.0).sum())
        very_low_entropy_count = int((entropies < 0.5).sum())

    purity_arr = np.array(purity_list) if purity_list else np.array([0.25])
    entropy_arr = np.array(entropy_list) if entropy_list else np.array([2.0])

    # --- Token Metrics ---
    if all_token_counts_list:
        token_counts = torch.cat(all_token_counts_list).numpy()
    else:
        token_counts = np.array([0])

    active_counts = token_counts[token_counts > 0]

    if len(active_counts) > 0:
        mean_concepts = float(active_counts.mean())
        std_concepts = float(active_counts.std())
        max_concepts = int(active_counts.max())
        sorted_counts = np.sort(active_counts)
        n = len(sorted_counts)
        cumulative = np.cumsum(sorted_counts)
        gini = (2 * np.sum((np.arange(1, n + 1) * sorted_counts)) - (n + 1) * cumulative[-1]) / (n * cumulative[-1])
        normalized_mean = (mean_concepts - 1) / (max_concepts - 1) if max_concepts > 1 else 0.0
        pmf_counts = np.zeros(max_concepts + 1)
        vals, counts = np.unique(active_counts, return_counts=True)
        pmf_counts[vals.astype(int)] = counts
        pmf = pmf_counts[1:] / pmf_counts[1:].sum()
        p1 = float(pmf[0]) if len(pmf) > 0 else 0.0
    else:
        mean_concepts = std_concepts = gini = normalized_mean = p1 = 0.0
        max_concepts = 0

    # --- Dual Mode Linear Probe ---
    import gc

    eval_loaders_lp = []
    if val_loader is not None:
        eval_loaders_lp.append(("val", val_loader))
    if test_loader is not None:
        eval_loaders_lp.append(("test", test_loader))

    logger.info("  Extracting SAE representations (Standard)...")
    X_train_std, y_train, _ = extract_sae_repr_for_probe(
        encoder, sae, train_loader, device, args.which_layer,
        token_l2_norm=args.token_l2_norm, strength_weighting=False,
        dead_threshold=float(args.dead_threshold),
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    X_eval_std_parts, y_eval_parts = [], []
    for name, loader in eval_loaders_lp:
        X_part, y_part, _ = extract_sae_repr_for_probe(
            encoder, sae, loader, device, args.which_layer,
            token_l2_norm=args.token_l2_norm, strength_weighting=False,
            dead_threshold=float(args.dead_threshold),
        )
        X_eval_std_parts.append(X_part)
        y_eval_parts.append(y_part)

    X_eval_std = np.concatenate(X_eval_std_parts, axis=0) if X_eval_std_parts else np.zeros((0, X_train_std.shape[1]), dtype=np.float32)
    y_eval = np.concatenate(y_eval_parts, axis=0) if y_eval_parts else np.zeros(0, dtype=np.int64)
    del X_eval_std_parts, y_eval_parts

    logger.info("  Training Linear Probe (Standard)...")
    probe_std = train_linear_probe(
        X_train_std, y_train, X_eval_std, y_eval,
        num_classes=NUM_CLASSES, epochs=50, lr=0.1, batch_size=256,
        normalize_repr=False, device=device, verbose=False
    )

    del X_train_std, X_eval_std
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("  Extracting SAE representations (Strength-weighted)...")
    X_train_str, y_train_str, _ = extract_sae_repr_for_probe(
        encoder, sae, train_loader, device, args.which_layer,
        token_l2_norm=args.token_l2_norm, strength_weighting=True,
        dead_threshold=float(args.dead_threshold),
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    X_eval_str_parts, y_eval_str_parts = [], []
    for name, loader in eval_loaders_lp:
        X_part, y_part, _ = extract_sae_repr_for_probe(
            encoder, sae, loader, device, args.which_layer,
            token_l2_norm=args.token_l2_norm, strength_weighting=True,
            dead_threshold=float(args.dead_threshold),
        )
        X_eval_str_parts.append(X_part)
        y_eval_str_parts.append(y_part)

    X_eval_str = np.concatenate(X_eval_str_parts, axis=0) if X_eval_str_parts else np.zeros((0, X_train_str.shape[1]), dtype=np.float32)
    y_eval_str = np.concatenate(y_eval_str_parts, axis=0) if y_eval_str_parts else np.zeros(0, dtype=np.int64)
    del X_eval_str_parts, y_eval_str_parts

    logger.info("  Training Linear Probe (Strength-weighted)...")
    probe_str = train_linear_probe(
        X_train_str, y_train_str, X_eval_str, y_eval_str,
        num_classes=NUM_CLASSES, epochs=50, lr=0.1, batch_size=256,
        normalize_repr=False, device=device, verbose=False
    )

    del X_train_str, X_eval_str
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(f"  Probe: std={probe_std['test_acc']:.3f}, str={probe_str['test_acc']:.3f} (val+test combined)")

    fvu = getattr(trainer, "best_fvu", None)
    if fvu is None:
        fvu = getattr(trainer, "last_fvu", 0.0)

    eff_rank_results = compute_effective_rank(sae, dead_threshold=float(args.dead_threshold))
    logger.info(f"  Effective Rank: {eff_rank_results['effective_rank']:.2f} (normalized: {eff_rank_results['normalized_effective_rank']:.3f})")

    results = {
        **exp_config,
        "total_images": image_idx,
        "total_tokens": total_tokens,
        "active_tokens": active_tokens,
        "active_token_ratio": active_tokens / max(1, total_tokens),
        "valid_concepts": len(purity_list),
        "high_purity_concepts": high_purity_count,
        "high_purity_Control": class_specific_breakdown[0],
        "high_purity_SNCA": class_specific_breakdown[1],
        "high_purity_GBA": class_specific_breakdown[2],
        "high_purity_LRRK2": class_specific_breakdown[3],
        "purity_mean": float(purity_arr.mean()),
        "purity_std": float(purity_arr.std()),
        "purity_median": float(np.median(purity_arr)),
        "purity_p90": float(np.percentile(purity_arr, 90)) if len(purity_arr) > 0 else 0.0,
        "purity_p95": float(np.percentile(purity_arr, 95)) if len(purity_arr) > 0 else 0.0,
        "entropy_mean": float(entropy_arr.mean()),
        "entropy_std": float(entropy_arr.std()),
        "entropy_min": float(entropy_arr.min()) if len(entropy_arr) > 0 else 0.0,
        "entropy_p05": float(np.percentile(entropy_arr, 5)) if len(entropy_arr) > 0 else 2.0,
        "entropy_p10": float(np.percentile(entropy_arr, 10)) if len(entropy_arr) > 0 else 2.0,
        "low_entropy_concepts": low_entropy_count,
        "very_low_entropy_concepts": very_low_entropy_count,
        "mean_concepts_per_token": mean_concepts,
        "std_concepts_per_token": std_concepts,
        "max_concepts_per_token": max_concepts,
        "gini_coefficient": gini,
        "normalized_mean": normalized_mean,
        "p_single_concept": p1,
        "probe_test_acc_standard": probe_std["test_acc"],
        "probe_test_acc_strength": probe_str["test_acc"],
        "probe_train_acc_standard": probe_std["train_acc"],
        "probe_train_acc_strength": probe_str["train_acc"],
        "fvu": float(fvu) if fvu else 0.0,
        "effective_rank": eff_rank_results["effective_rank"],
        "norm_effective_rank": eff_rank_results["normalized_effective_rank"],
    }

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save class-wise GAP means CSV
    class_gap_mean = torch.zeros((NUM_CLASSES, d_sae), dtype=torch.float32)
    for c in range(NUM_CLASSES):
        if class_img_count[c] > 0:
            class_gap_mean[c] = class_gap_sum[c].cpu() / class_img_count[c].item()

    dead_mask = sae.usage_ema.cpu() < float(args.dead_threshold)
    alive_indices = (~dead_mask).nonzero(as_tuple=True)[0].tolist()

    class_names = ["Control", "SNCA", "GBA", "LRRK2"]
    tie_str = "tied" if sae.tie_weights else "untied"
    sparsity_val = float(exp_config.get("sparsity", 0.0))
    aux_val = float(exp_config.get("aux_coeff", 0.0))
    gap_csv_filename = f"gated_sae_{args.which_layer}_d{d_sae}_sp{sparsity_val}_aux{aux_val}_{tie_str}_class_gap_means.csv"
    gap_csv_path = os.path.join(args.sae_save_dir, gap_csv_filename)

    class_active_img_count_cpu = class_active_img_count.cpu()

    with open(gap_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        active_count_cols = [f"n_{name}" for name in class_names]
        w.writerow(["concept_id", "is_alive"] + class_names + active_count_cols + ["max_class", "class_diff", "entropy", "total_active_imgs"])

        for i in range(d_sae):
            is_alive = int(i in alive_indices)
            gaps = [float(class_gap_mean[c, i]) for c in range(NUM_CLASSES)]
            active_counts_csv = [int(class_active_img_count_cpu[c, i]) for c in range(NUM_CLASSES)]
            total_active = sum(active_counts_csv)
            max_class = class_names[np.argmax(gaps)] if max(gaps) > 0 else "None"
            class_diff = max(gaps) - min(gaps)
            total = sum(gaps) + 1e-10
            probs_csv = [g / total for g in gaps]
            ent = -sum(p * np.log2(max(p, 1e-10)) for p in probs_csv)
            w.writerow([i, is_alive] + gaps + active_counts_csv + [max_class, f"{class_diff:.4f}", f"{ent:.4f}", total_active])

    logger.info(f"  Saved class-wise GAP means to: {gap_csv_path}")
    logger.info(f"  Alive concepts: {len(alive_indices)}/{d_sae}")

    return results
