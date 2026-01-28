# ==============================================================================
# Gated SAE Trainer with Hyperparameter Optimization
# - L2-weighted reconstruction loss
# - Sparsity warmup schedule
# - Automated grid search for sparsity, aux_coeff, tie_weights
# - Linear probe evaluation for classification accuracy
# ==============================================================================
import os
import csv
import random
import numpy as np
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from sae_project.step01_configs import get_args, resolve_paths
from sae_project.step02_logging_utils import get_logger
from sae_project.step03_data_shards import load_all_sample_refs, build_uid_to_refidx
from sae_project.step04_data_bank import (
    InMemoryTarBank, InMemorySixteenBitDataset, load_split_csv,
    seed_worker, collate_skip_none, StrictPlateBalancedBatchSamplerOnBank
)
from sae_project.step05_model_encoder import SupConMoCoModel, parse_int_list, renorm_unit_per_out_channel_
from sae_project.step06_gated_sae import GatedSAE, get_sparsity_coeff, get_aux_coeff_cosine_decay

logger = get_logger("train_gated_sae")


# ==============================================================================
# Linear Probe for Classification
# ==============================================================================

class LinearProbe(nn.Module):
    """Simple linear probe for classification."""
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.net = nn.Linear(d_in, d_out, bias=False)  # ??Match Backbone
    
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
    """
    Extract SAE representations for linear probe evaluation (Vectorized).
    """
    encoder.eval()
    sae.eval()
    
    # Get alive neuron mask
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
        
        # 학습과 동일하게 항상 L2 normalize
        flat_tokens = F.normalize(flat_tokens, dim=1, eps=1e-12)
        
        # [MEM] Process tokens in chunks to avoid OOM
        token_batch_size = 1024  # Reduced from 4096
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
    """
    Train linear probe with balanced class sampling and mini-batch updates.
    
    Args:
        balanced_training: If True, sample equal number from each class per epoch
        
    Returns:
        dict with train_acc, test_acc, d_probe
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    original_dim = X_train.shape[1]
    
    # Remove zero-variance features
    feat_std = X_train.std(axis=0)
    alive_mask = feat_std >= 1e-8
    n_alive = int(alive_mask.sum())
    
    if n_alive == 0:
        return {"train_acc": 0.0, "test_acc": 0.0, "d_probe": 0}
    
    X_train = X_train[:, alive_mask]
    X_test = X_test[:, alive_mask]
    
    # Z-score normalize (Optional, disabled by default for sparse activations)
    if normalize_repr:
        mean = X_train.mean(axis=0, keepdims=True)
        std = X_train.std(axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std
    
    if verbose:
        logger.info(f"  [Probe] dim: {original_dim} -> {n_alive} (removed {original_dim - n_alive} zero-var)")
        logger.info(f"  [Probe] Train: {len(y_train)}, Test: {len(y_test)}")
    
    # Class indices for balanced sampling
    rng = np.random.default_rng(42)
    class_indices = {c: np.where(y_train == c)[0] for c in range(num_classes)}
    min_class_count = min(len(v) for v in class_indices.values())
    samples_per_class = min_class_count
    
    if balanced_training and verbose:
        logger.info(f"  [Probe] Balanced: {samples_per_class} samples/class x {num_classes} = {samples_per_class * num_classes}/epoch")
    
    # Train probe
    probe = LinearProbe(n_alive, num_classes).to(device)
    optimizer = optim.SGD(probe.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    probe.train()
    for epoch in range(epochs):
        if balanced_training:
            # Balanced sampling: equal number from each class
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
        
        # Mini-batch training
        for s in range(0, len(epoch_indices), batch_size):
            ii = epoch_indices[s:s+batch_size]
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
        # Train accuracy (on balanced sample)
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
    
    if verbose:
        logger.info(f"  [Probe] train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")
        logger.info(f"  [Probe] Per-class: {per_class_acc}")
    
    return {
        "train_acc": float(train_acc),
        "test_acc": float(test_acc),
        "d_probe": n_alive,
        "per_class_acc": per_class_acc,
    }


@torch.no_grad()
def compute_effective_rank(
    sae: GatedSAE,
    dead_threshold: float = 1e-6,
) -> dict:
    """
    Compute effective rank of alive concept weights using SVD entropy.
    
    Effective Rank = exp(entropy of normalized singular values)
    
    - 1: All concepts are identical (rank-1)
    - d_alive: All concepts are orthogonal/independent (full rank)
    
    Returns:
        dict with effective_rank, normalized_effective_rank, d_alive
    """
    # Get decoder weights (d_sae, d_in)
    W_dec = sae.W_dec.data.float().cpu()  # (d_sae, d_in)
    
    # Get alive neuron mask
    usage = sae.usage_ema.cpu()
    alive_mask = usage >= dead_threshold
    d_alive = int(alive_mask.sum().item())
    
    if d_alive < 2:
        return {
            "effective_rank": 1.0,
            "normalized_effective_rank": 1.0,
            "d_alive": d_alive,
        }
    
    # Extract alive weights only
    W_alive = W_dec[alive_mask]  # (d_alive, d_in)
    
    # SVD
    U, S, Vh = torch.linalg.svd(W_alive, full_matrices=False)
    
    # Normalize singular values to probabilities
    S = S.clamp(min=1e-12)
    p = S / S.sum()
    
    # Shannon entropy of singular value distribution
    entropy = -torch.sum(p * torch.log(p)).item()
    
    # Effective rank = exp(entropy)
    effective_rank = np.exp(entropy)
    
    # Normalized effective rank (0-1 scale where 1 = full rank)
    max_rank = min(d_alive, W_alive.shape[1])
    normalized_effective_rank = effective_rank / max_rank
    
    return {
        "effective_rank": float(effective_rank),
        "normalized_effective_rank": float(normalized_effective_rank),
        "d_alive": d_alive,
    }


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

# fvu 
@torch.no_grad()
def _sse_sst(x: torch.Tensor, xhat: torch.Tensor):
    diff = xhat - x
    sse = float((diff * diff).sum().item())
    mean = x.mean(dim=0, keepdim=True)
    xc = x - mean
    sst = float((xc * xc).sum().item())
    return sse, sst


@torch.no_grad()
def summarize_usage(usage_ema: torch.Tensor, dead_threshold: float):
    """Compute detailed usage statistics for dead neuron analysis."""
    u = usage_ema.detach().float().flatten()
    d = u.numel()
    
    # Dead neuron counts at various thresholds
    dead_1e6 = int((u < 1e-6).sum().item())
    dead_1e5 = int((u < 1e-5).sum().item())
    dead_1e4 = int((u < 1e-4).sum().item())
    dead_1e3 = int((u < 1e-3).sum().item())
    dead_1e2 = int((u < 1e-2).sum().item())
    dead_custom = int((u < dead_threshold).sum().item())
    
    # Quantiles
    qs = torch.tensor([0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0], device=u.device)
    p = torch.quantile(u, qs).tolist()
    
    # Activation frequency buckets
    very_low = int((u < 0.01).sum().item())  # < 1% activated
    low = int(((u >= 0.01) & (u < 0.1)).sum().item())  # 1-10%
    medium = int(((u >= 0.1) & (u < 0.5)).sum().item())  # 10-50%
    high = int((u >= 0.5).sum().item())  # > 50%
    
    return {
        "total_features": d,
        "dead": dead_custom,
        "dead_frac": float(dead_custom / max(1, d)),
        # Dead counts at various thresholds
        "dead_1e-6": dead_1e6,
        "dead_1e-5": dead_1e5,
        "dead_1e-4": dead_1e4,
        "dead_1e-3": dead_1e3,
        "dead_1e-2": dead_1e2,
        # Activation frequency buckets
        "freq_very_low": very_low,  # < 1%
        "freq_low": low,            # 1-10%
        "freq_medium": medium,      # 10-50%
        "freq_high": high,          # > 50%
        # Quantiles
        "min": float(p[0]), "p01": float(p[1]), "p05": float(p[2]), "p10": float(p[3]),
        "p25": float(p[4]), "p50": float(p[5]), "p75": float(p[6]),
        "p90": float(p[7]), "p95": float(p[8]), "p99": float(p[9]), "max": float(p[10]),
        # Statistics
        "mean": float(u.mean().item()),
        "std": float(u.std().item()),
    }


def format_usage_summary(s):
    """Format detailed usage summary for logging."""
    return (
        f"\n  [Dead Neurons] threshold={s['dead']}/{s['total_features']} ({s['dead_frac']*100:.1f}%)\n"
        f"    dead@1e-6={s['dead_1e-6']} | dead@1e-5={s['dead_1e-5']} | dead@1e-4={s['dead_1e-4']} | "
        f"dead@1e-3={s['dead_1e-3']} | dead@1e-2={s['dead_1e-2']}\n"
        f"  [Activation Freq] very_low(<1%)={s['freq_very_low']} | low(1-10%)={s['freq_low']} | "
        f"medium(10-50%)={s['freq_medium']} | high(>50%)={s['freq_high']}\n"
        f"  [Usage EMA] min={s['min']:.6f} p01={s['p01']:.6f} p05={s['p05']:.6f} p10={s['p10']:.6f} "
        f"p25={s['p25']:.6f} p50={s['p50']:.6f}\n"
        f"              p75={s['p75']:.6f} p90={s['p90']:.6f} p95={s['p95']:.6f} p99={s['p99']:.6f} max={s['max']:.6f}\n"
        f"  [Stats] mean={s['mean']:.6f} std={s['std']:.6f}"
    )


class GatedSAETrainer:
    def __init__(self, args, encoder, train_loader: DataLoader,
                 val_loader: DataLoader | None = None,
                 test_loader: DataLoader | None = None):
        self.args = args
        self.encoder = encoder
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"[env] torch={torch.__version__}, cuda_available={torch.cuda.is_available()}, device={self.device}")
        if torch.cuda.is_available():
            logger.info(f"[env] cuda={torch.version.cuda}, gpu={torch.cuda.get_device_name(0)}")

        self.encoder.eval().to(self.device).to(memory_format=torch.channels_last)
        for p in self.encoder.parameters():
            p.requires_grad = False
        renorm_unit_per_out_channel_(self.encoder)

        # Create Gated SAE
        self.sae = GatedSAE(
            d_in=args.d_in,
            d_sae=args.d_sae,
            tie_weights=bool(getattr(args, "tie_gate_weights", False)),
            aux_k=int(getattr(args, "aux_k", 32)),
            init_scale=args.sae_init_scale,
        ).to(self.device)
        self.sae.token_norm_mode = getattr(args, "token_norm_mode", "gap-scalar")

        # Optionally load clustering initialization
        if getattr(args, "use_clustering_init", False):
            centroid_path = os.path.join(
                args.save_dir, "token_clustering", f"centroids_{args.which_layer}.npy"
            )
            noise_scale = float(getattr(args, "clustering_init_noise", 0.1))
            self.sae.load_clustering_init(centroid_path, noise_scale=noise_scale)

        self.opt = torch.optim.AdamW(
            self.sae.parameters(),
            lr=args.sae_lr,
            betas=(0.9, 0.999),
            weight_decay=args.sae_wd
        )

        self.use_bf16 = bool(args.use_bf16) and (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=(torch.cuda.is_available() and not self.use_bf16))

        self.autocast_kwargs = dict(device_type="cuda", enabled=torch.cuda.is_available())
        if self.use_bf16:
            self.autocast_kwargs["dtype"] = torch.bfloat16

        # Sparsity warmup config
        self.sparsity_warmup_steps = int(getattr(args, "sparsity_warmup_steps", 1000))
        self.final_sparsity_coeff = float(getattr(args, "final_sparsity_coeff", 5.0))
        self.aux_coeff = float(getattr(args, "aux_coeff", 0.1))
        self.initial_aux_coeff = self.aux_coeff  # Store for decay schedule

        # Create unique experiment name
        exp_name = f"gated_sp{self.final_sparsity_coeff}_aux{self.aux_coeff}"
        if getattr(args, "tie_gate_weights", False):
            exp_name += "_tied"
        if getattr(args, "use_clustering_init", False):
            exp_name += "_clust"

        os.makedirs(args.sae_save_dir, exist_ok=True)
        
        # Filename format: {layer}_d{d_sae}_{exp_name}_ep{epoch}.pt
        self.layer_name = args.which_layer
        self.base_filename = f"{self.layer_name}_d{args.d_sae}_{exp_name}"
        self.ckpt_path = os.path.join(args.sae_save_dir, f"{self.base_filename}.pt")
        self.best_ckpt_path = os.path.join(args.sae_save_dir, f"{self.base_filename}_BEST.pt")
        self.log_csv_path = os.path.join(args.sae_save_dir, f"{self.base_filename}_trainlog.csv")

        self._init_log_csv()

        self.sae.renorm_decoder_()
        self.global_step = 0
        self.best_metric = float("inf")

        self.token_batch = int(getattr(args, "token_batch", 8192))
        if self.token_batch <= 0:
            self.token_batch = 8192
        self.shuffle_tokens = bool(getattr(args, "shuffle_tokens", True))

        # Estimate total steps for sparsity schedule
        self.total_steps = args.epochs * len(train_loader)

        logger.info(f"[GatedSAE] tie_weights={getattr(args, 'tie_gate_weights', False)}, "
                    f"aux_coeff={self.aux_coeff}, final_sparsity={self.final_sparsity_coeff}")
        logger.info(f"[GatedSAE] sparsity_warmup_steps={self.sparsity_warmup_steps}, total_steps={self.total_steps}")

    def _init_log_csv(self):
        if not os.path.exists(self.log_csv_path):
            with open(self.log_csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    "epoch", "step",
                    "train_recon", "train_sparsity", "train_aux", "train_total", "train_fvu",
                    "val_recon", "val_sparsity", "val_aux", "val_total", "val_fvu",
                    "test_recon", "test_sparsity", "test_aux", "test_total", "test_fvu",
                    "dead_count", "sparsity_coeff"
                ])


    @torch.no_grad()
    def _extract_tokens_with_l2norms(self, x: torch.Tensor):
        with torch.amp.autocast(**self.autocast_kwargs):
            fmap = self.encoder.forward_feature_maps(x, which=self.args.which_layer)

        curr_batch_size = fmap.size(0)
        norm_mode = getattr(self.args, "token_norm_mode", "gap-scalar")
        if norm_mode == "gap-scalar":
            gap = fmap.mean(dim=(2, 3))
            gap_norms = gap.norm(dim=1, keepdim=True).view(curr_batch_size, 1, 1, 1).clamp_min(1e-12)
            fmap = fmap / gap_norms

        fmap = fmap.permute(0, 2, 3, 1).contiguous()
        _, Hf, Wf, C = fmap.shape
        tokens = fmap.view(curr_batch_size * Hf * Wf, C)

        tokens = tokens - tokens.mean(dim=0, keepdim=True)
        # L2 normalize 전에 원래 norm 저장 (loss 가중치용)
        l2_norms = tokens.norm(dim=1)

        # 항상 L2 normalize: SAE는 방향만 학습하도록
        tokens = F.normalize(tokens, dim=1, eps=1e-12)

        tpi = int(self.args.tokens_per_image)
        if tpi > 0 and tpi < (Hf * Wf):
            tokens_list, l2_list = [], []
            for b_idx in range(curr_batch_size):
                base = b_idx * (Hf * Wf)
                idx = torch.randperm(Hf * Wf, device=tokens.device)[:tpi]
                tokens_list.append(tokens[base + idx])
                l2_list.append(l2_norms[base + idx])
            tokens = torch.cat(tokens_list, dim=0)
            l2_norms = torch.cat(l2_list, dim=0)

        if self.shuffle_tokens and tokens.size(0) > 1:
            perm = torch.randperm(tokens.size(0), device=tokens.device)
            tokens = tokens[perm]
            l2_norms = l2_norms[perm]

        return tokens, l2_norms

    def _compute_loss(
        self, tokens: torch.Tensor, l2_norms: torch.Tensor, sparsity_coeff: float
    ):
        """
        Compute Gated SAE loss with L2-weighted MSE.
        """
        recon, acts, gate_pre, recon_aux, acts_aux = self.sae(tokens)

        tok_f = tokens.float()
        rec_f = recon.float()
        l2_f = l2_norms.float()

        # 1. Reconstruction Loss with L2 weighting
        # CNN이 집중한 토큰(L2 norm 큰)을 더 열심히 학습
        mse = (rec_f - tok_f).pow(2).sum(dim=1)  # (N,)
        
        # L2 norm 기반 가중치: 큰 L2 norm → 더 중요한 토큰
        weights = l2_f / (l2_f.mean() + 1e-8)
        L_recon = (mse * weights).mean()

        # 2. Sparsity Loss: Gated SAE specific sparsity
        L_sparsity = self.sae.compute_sparsity_loss(gate_pre)

        # 3. Aux loss (predicting RESIDUAL using dead neurons)
        if self.aux_coeff > 0 and recon_aux.abs().sum() > 0:
            # Important: Dead neurons should learn to reconstruct the part NOT captured by main path
            residual = tok_f - rec_f.detach()
            aux_error = (recon_aux.float() - residual).pow(2).sum(dim=1)
            L_aux = (aux_error * weights).mean()
        else:
            L_aux = torch.tensor(0.0, device=tokens.device)

        # Total loss
        total_loss = L_recon + sparsity_coeff * L_sparsity + self.aux_coeff * L_aux

        return total_loss, L_recon, L_sparsity, L_aux, acts

    def save_ckpt(self, path: str):
        ckpt = {
            "args": vars(self.args),
            "sae": self.sae.state_dict(),
            "opt": self.opt.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "global_step": int(self.global_step),
            "best_metric": float(self.best_metric),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            }
        }
        torch.save(ckpt, path)

    def cleanup_for_eval(self):
        """
        Release GPU VRAM used by optimizer and training state before evaluation.
        Call this after training completes but before running evaluation.
        """
        import gc
        
        # 1. Delete optimizer (Adam keeps 2x model params in buffers)
        if hasattr(self, "opt") and self.opt is not None:
            del self.opt
            self.opt = None
        
        # 2. Delete scaler
        if hasattr(self, "scaler") and self.scaler is not None:
            del self.scaler
            self.scaler = None
        
        # 3. Clear train/val/test loader DataBank images (huge RAM)
        for loader in [self.train_loader, self.val_loader, self.test_loader]:
            if loader is not None and hasattr(loader, "dataset"):
                ds = loader.dataset
                if hasattr(ds, "bank") and hasattr(ds.bank, "images"):
                    ds.bank.images = None  # Release image array
        
        # 4. Force garbage collection
        gc.collect()
        
        # 5. Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        logger.info("[MEM] Trainer cleanup complete for evaluation")

    @torch.no_grad()
    def eval_epoch(self, loader: DataLoader, tag: str):
        self.sae.eval()

        recon_sum, sparsity_sum, aux_sum, steps = 0.0, 0.0, 0.0, 0
        sse_sum, sst_sum = 0.0, 0.0

        for batch in tqdm(loader, desc=f"GatedSAE {tag}", leave=False):
            if batch is None:
                continue
            x_cpu, *_ = batch
            if x_cpu.numel() < 1:
                continue

            x = x_cpu.to(self.device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            tokens, l2_norms = self._extract_tokens_with_l2norms(x)

            Tb = self.token_batch
            n = tokens.size(0)

            recon_acc, sparsity_acc, aux_acc = 0.0, 0.0, 0.0
            chunks = 0

            for s in range(0, n, Tb):
                tok = tokens[s:s+Tb]
                l2n = l2_norms[s:s+Tb]

                with torch.amp.autocast(**self.autocast_kwargs):
                    _, L_recon, L_sparsity, L_aux, _ = self._compute_loss(
                        tok, l2n, self.final_sparsity_coeff
                    )

                recon_acc += float(L_recon.item())
                sparsity_acc += float(L_sparsity.item())
                aux_acc += float(L_aux.item())
                chunks += 1

                if getattr(self.args, "log_fvu", False):
                    with torch.amp.autocast(**self.autocast_kwargs):
                        recon, *_ = self.sae(tok)
                    sse, sst = _sse_sst(tok.float(), recon.float())
                    sse_sum += sse
                    sst_sum += sst

            if chunks > 0:
                recon_sum += recon_acc / chunks
                sparsity_sum += sparsity_acc / chunks
                aux_sum += aux_acc / chunks
                steps += 1

        self.sae.train()
        if steps == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        recon_avg = recon_sum / steps
        sparsity_avg = sparsity_sum / steps
        aux_avg = aux_sum / steps
        total = recon_avg + self.final_sparsity_coeff * sparsity_avg + self.aux_coeff * aux_avg

        fvu = 0.0
        if getattr(self.args, "log_fvu", False):
            fvu = 0.0 if sst_sum <= 1e-12 else float(sse_sum / sst_sum)

        return recon_avg, sparsity_avg, aux_avg, total, fvu

    def train(self):
        logger.info(f"[GatedSAE] device={self.device}, bf16={self.use_bf16}")
        logger.info(f"[GatedSAE] which_layer={self.args.which_layer}, d_in={self.args.d_in}, d_sae={self.args.d_sae}")
        logger.info(f"[GatedSAE] token_batch={self.token_batch}, shuffle_tokens={self.shuffle_tokens}")

        # Warmup
        if self.device.type == "cuda":
            with torch.no_grad():
                dummy = torch.zeros(2, 3, self.args.img_size, self.args.img_size, device=self.device).contiguous(memory_format=torch.channels_last)
                _ = self.encoder.forward_feature_maps(dummy, which=self.args.which_layer)

        for epoch in range(1, self.args.epochs + 1):
            self.sae.train()
            
            # [NEW] Compute aux coefficient with cosine decay (epoch 2 -> end: decay to 0)
            current_aux = get_aux_coeff_cosine_decay(
                epoch=epoch,
                total_epochs=self.args.epochs,
                initial_aux=self.initial_aux_coeff,
                decay_start_epoch=2,
            )
            self.aux_coeff = current_aux  # Update for _compute_loss

            train_recon_sum, train_sparsity_sum, train_aux_sum, train_steps = 0.0, 0.0, 0.0, 0
            train_sse_sum, train_sst_sum = 0.0, 0.0
            dead_count_epoch = 0

            pbar = tqdm(self.train_loader, desc=f"GatedSAE Train E{epoch}/{self.args.epochs}", leave=True)
            for batch in pbar:
                if batch is None:
                    continue
                x_cpu, *_ = batch
                if x_cpu.numel() < 1:
                    continue

                x = x_cpu.to(self.device, non_blocking=True).contiguous(memory_format=torch.channels_last)
                tokens, l2_norms = self._extract_tokens_with_l2norms(x)

                Tb = self.token_batch
                n = tokens.size(0)
                if n == 0:
                    continue

                self.opt.zero_grad(set_to_none=True)

                # Get current sparsity coefficient
                current_sparsity = get_sparsity_coeff(
                    self.global_step,
                    self.sparsity_warmup_steps,
                    self.final_sparsity_coeff,
                    self.total_steps
                )

                recon_acc, sparsity_acc, aux_acc = 0.0, 0.0, 0.0
                chunks = 0
                chunks_count = (n + Tb - 1) // Tb

                for s in range(0, n, Tb):
                    tok = tokens[s:s+Tb]
                    l2n = l2_norms[s:s+Tb]

                    with torch.amp.autocast(**self.autocast_kwargs):
                        loss, L_recon, L_sparsity, L_aux, acts = self._compute_loss(
                            tok, l2n, current_sparsity
                        )


                    loss = loss / float(chunks_count)

                    if self.use_bf16:
                        loss.backward()
                    else:
                        self.scaler.scale(loss).backward()

                    with torch.no_grad():
                        self.sae.update_usage_ema_(acts.detach(), ema=float(self.args.usage_ema))

                    recon_acc += float(L_recon.item())
                    sparsity_acc += float(L_sparsity.item())
                    aux_acc += float(L_aux.item())
                    chunks += 1

                    if getattr(self.args, "log_fvu", False):
                        with torch.amp.autocast(**self.autocast_kwargs):
                            recon, *_ = self.sae(tok)
                        sse, sst = _sse_sst(tok.float(), recon.float())
                        train_sse_sum += sse
                        train_sst_sum += sst

                # ===== Gradient Projection + Optimizer Step =====
                # 1. Unscale gradients if using mixed precision
                if not self.use_bf16:
                    self.scaler.unscale_(self.opt)
                
                # 2. Project decoder gradients (remove parallel component)
                #    This prevents conflict with momentum-based optimizers (Adam)
                #    by removing gradient components that only change norm magnitude
                self.sae.project_decoder_grads_()
                
                # 3. Gradient clipping
                if self.args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.sae.parameters(), float(self.args.grad_clip))
                
                # 4. Optimizer step
                if self.use_bf16:
                    self.opt.step()
                else:
                    self.scaler.step(self.opt)
                    self.scaler.update()

                # 5. Decoder L2 normalization constraint (every step)
                self.sae.renorm_decoder_()

                dead_count = int((self.sae.usage_ema < float(self.args.dead_threshold)).sum().item())
                dead_count_epoch = dead_count

                self.global_step += 1

                recon_step = recon_acc / max(1, chunks)
                sparsity_step = sparsity_acc / max(1, chunks)
                aux_step = aux_acc / max(1, chunks)

                train_recon_sum += recon_step
                train_sparsity_sum += sparsity_step
                train_aux_sum += aux_step
                train_steps += 1

                pbar.set_postfix({
                    "recon": f"{recon_step:.4f}",
                    "sp": f"{sparsity_step:.4f}",
                    "aux": f"{aux_step:.4f}",
                    "λsp": f"{current_sparsity:.2f}",
                    "αaux": f"{current_aux:.4f}",
                    "dead": dead_count,
                })

                if (self.global_step % int(self.args.save_every)) == 0:
                    self.save_ckpt(self.ckpt_path)

            # Epoch summaries
            if train_steps == 0:
                train_recon_avg = train_sparsity_avg = train_aux_avg = train_total = train_fvu = 0.0
            else:
                train_recon_avg = train_recon_sum / train_steps
                train_sparsity_avg = train_sparsity_sum / train_steps
                train_aux_avg = train_aux_sum / train_steps
                train_total = train_recon_avg + self.final_sparsity_coeff * train_sparsity_avg + self.aux_coeff * train_aux_avg
                train_fvu = 0.0
                if getattr(self.args, "log_fvu", False):
                    train_fvu = 0.0 if train_sst_sum <= 1e-12 else float(train_sse_sum / train_sst_sum)

            # Val/Test
            val_recon = val_sparsity = val_aux = val_total = val_fvu = 0.0
            test_recon = test_sparsity = test_aux = test_total = test_fvu = 0.0

            if self.val_loader is not None:
                val_recon, val_sparsity, val_aux, val_total, val_fvu = self.eval_epoch(self.val_loader, "Val")
            if self.test_loader is not None:
                test_recon, test_sparsity, test_aux, test_total, test_fvu = self.eval_epoch(self.test_loader, "Test")

            metric = val_total if (self.val_loader is not None) else train_total

            usage_stats = summarize_usage(self.sae.usage_ema, float(self.args.dead_threshold))
            dead_count_epoch = usage_stats["dead"]

            current_sparsity = get_sparsity_coeff(
                self.global_step, self.sparsity_warmup_steps, self.final_sparsity_coeff, self.total_steps
            )

            tqdm.write(
                f"Epoch {epoch:03d} | "
                f"Train: recon={train_recon_avg:.6f} sp={train_sparsity_avg:.6f} aux={train_aux_avg:.6f} "
                f"fvu={train_fvu:.6f} total={train_total:.6f} | "
                f"Val: recon={val_recon:.6f} sp={val_sparsity:.6f} aux={val_aux:.6f} "
                f"fvu={val_fvu:.6f} total={val_total:.6f} | "
                f"Test: recon={test_recon:.6f} sp={test_sparsity:.6f} aux={test_aux:.6f} | "
                f"dead={dead_count_epoch} 貫={current_sparsity:.2f}"
            )

            tqdm.write(format_usage_summary(usage_stats))

            with open(self.log_csv_path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    epoch, self.global_step,
                    train_recon_avg, train_sparsity_avg, train_aux_avg, train_total, train_fvu,
                    val_recon, val_sparsity, val_aux, val_total, val_fvu,
                    test_recon, test_sparsity, test_aux, test_total, test_fvu,
                    dead_count_epoch, current_sparsity
                ])

            # Save standard last checkpoint
            self.save_ckpt(self.ckpt_path)
            
            # [NEW] Save per-epoch checkpoint
            epoch_ckpt_path = self.ckpt_path.replace(".pt", f"_ep{epoch:03d}.pt")
            self.save_ckpt(epoch_ckpt_path)
            tqdm.write(f"  -> Saved Epoch {epoch} to {os.path.basename(epoch_ckpt_path)}")

            if getattr(self.args, "save_best", False):
                if metric < self.best_metric:
                    self.best_metric = float(metric)
                    self.best_fvu = test_fvu if test_fvu else val_fvu  # Store best FVU
                    self.save_ckpt(self.best_ckpt_path)
                    tqdm.write(f"  -> Saved BEST to {self.best_ckpt_path} (metric={self.best_metric:.6f})")
            
            # Store last FVU for evaluation
            self.last_fvu = test_fvu if test_fvu else val_fvu

        logger.info(f"[GatedSAE] Done. Saved -> {self.ckpt_path}")


def _make_loader_from_split(args, refs, uid_to_refidx, split_csv_path: str,
                           batch_size: int, augment: bool, shuffle: bool,
                           strict_balance: bool, seed: int):
    if not os.path.exists(split_csv_path):
        return None

    uids = load_split_csv(split_csv_path)
    missing = [u for u in uids if u not in uid_to_refidx]
    if len(missing) > 0:
        raise RuntimeError(f"Some uids in split are missing under current shard_root. ex: {missing[:5]}")
    refidx = [uid_to_refidx[u] for u in uids]

    bank = InMemoryTarBank(refs, refidx, args.img_size)
    ib = list(range(len(refidx)))
    ds = InMemorySixteenBitDataset(bank, ib, args.img_size, augment=augment)

    pin = torch.cuda.is_available()

    if strict_balance:
        sampler = StrictPlateBalancedBatchSamplerOnBank(bank, batch_size=batch_size, seed=seed)
        loader = DataLoader(
            ds,
            batch_sampler=sampler,
            num_workers=int(args.num_workers),
            pin_memory=pin,
            worker_init_fn=seed_worker,
            collate_fn=collate_skip_none,
        )
        return loader

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(args.num_workers),
        pin_memory=pin,
        worker_init_fn=seed_worker,
        collate_fn=collate_skip_none,
        drop_last=False,
    )
    return loader


def run_experiment(args, refs=None, uid_to_refidx=None, encoder=None):
    """
    Run a single Gated SAE training experiment.
    
    Returns:
        trainer: GatedSAETrainer object (with trained SAE)
        ckpt_path: Path to saved checkpoint
    """
    # Load data if not provided
    if refs is None:
        refs = load_all_sample_refs(args.shard_root)
    if uid_to_refidx is None:
        uid_to_refidx = build_uid_to_refidx(refs)

    train_csv = os.path.join(args.save_dir, "train_split.csv")
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"train_split.csv not found: {train_csv}")

    strict = True  # ??Always use strict plate balance for SAE training as requested

    train_loader = _make_loader_from_split(
        args, refs, uid_to_refidx,
        split_csv_path=train_csv,
        batch_size=int(args.batch_size),
        augment=True,  # ??Match Backbone
        shuffle=not strict,
        strict_balance=strict,
        seed=int(args.seed)
    )
    assert train_loader is not None

    val_loader = None
    if getattr(args, "use_val", False):
        val_csv = os.path.join(args.save_dir, "val_split.csv")
        vbs = int(args.batch_size) if int(getattr(args, "val_batch_size", 0)) <= 0 else int(args.val_batch_size)
        val_loader = _make_loader_from_split(
            args, refs, uid_to_refidx,
            split_csv_path=val_csv,
            batch_size=vbs,
            augment=False,
            shuffle=False,
            strict_balance=True,
            seed=int(args.seed) + 1
        )

    test_loader = None
    if getattr(args, "use_test", False):
        test_csv = os.path.join(args.save_dir, "test_split.csv")
        tbs = int(args.batch_size) if int(getattr(args, "test_batch_size", 0)) <= 0 else int(args.test_batch_size)
        test_loader = _make_loader_from_split(
            args, refs, uid_to_refidx,
            split_csv_path=test_csv,
            batch_size=tbs,
            augment=False,
            shuffle=False,
            strict_balance=True,
            seed=int(args.seed) + 2
        )

    # Load encoder if not provided
    if encoder is None:
        blocks = parse_int_list(args.blocks, 4)
        dilations = parse_int_list(args.dilations, 4)

        model = SupConMoCoModel(
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
        from sae_project.step05_model_encoder import robust_load_state_dict
        robust_load_state_dict(model, sd, strict=True)
        encoder = model.encoder

    trainer = GatedSAETrainer(args, encoder, train_loader, val_loader=val_loader, test_loader=test_loader)
    trainer.train()
    
    return trainer, trainer.ckpt_path


# ============== Multi-Layer Training ==============
ALL_LAYERS = ["stage5_out", "refine_out"]


def evaluate_concepts_for_sae(
    trainer: GatedSAETrainer,
    train_loader: DataLoader,
    test_loader: DataLoader,
    args,
    exp_config: dict,
) -> dict:
    """
    Evaluate concept metrics and linear probe accuracy (Vectorized).
    """
    device = trainer.device
    encoder = trainer.encoder
    sae = trainer.sae
    
    # [MEM] Clear VRAM cache before heavy evaluation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    encoder.eval()
    sae.eval()
    
    NUM_CLASSES = 4
    d_sae = sae.d_sae
    
    # Vectorized counters (using float for weighted sum)
    concept_class_counts = torch.zeros((d_sae, NUM_CLASSES), dtype=torch.float32, device=device)
    
    # [NEW] Class-wise GAP accumulators for per-concept average activation
    # gap_sum[c] = sum of GAP vectors for class c, gap_count[c] = number of images
    class_gap_sum = torch.zeros((NUM_CLASSES, d_sae), dtype=torch.float32, device=device)
    class_img_count = torch.zeros(NUM_CLASSES, dtype=torch.long, device=device)
    
    all_token_counts_list = []
    
    image_idx = 0
    total_tokens = 0
    active_tokens = 0
    
    autocast_kwargs = dict(device_type="cuda", enabled=torch.cuda.is_available())
    if torch.cuda.is_available():
        autocast_kwargs["dtype"] = torch.bfloat16
    
    # Use test_loader for concept metrics (faster, relevant)
    # [MEM] Use no_grad only for concept evaluation, NOT for linear probe training
    with torch.no_grad():
      for batch in tqdm(test_loader, desc="Evaluating concepts", leave=False):
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
        # 학습과 동일하게 항상 L2 normalize
        flat_tokens = F.normalize(flat_tokens, dim=1, eps=1e-12)
            
        tokens = flat_tokens.view(curr_batch_size, -1, C)
        num_tokens_per_img = tokens.shape[1]
        
        # [MEM] Compute metrics incrementally in chunks WITHOUT storing full acts tensor
        token_batch_size = 1024  # Reduced from 4096 to prevent OOM
        num_flat_tokens = flat_tokens.size(0)
        
        # Accumulate image-level sums per concept: (B, d_sae)
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
            
            # Figure out which images these tokens belong to
            # tokens are flattened as [img0_tok0, img0_tok1, ..., img1_tok0, ...]
            token_start_idx = start
            token_end_idx = end
            
            # Compute per-token active concept counts for this chunk
            chunk_active = (chunk_acts > 0).sum(dim=1)  # (chunk_size,)
            batch_token_counts_list.append(chunk_active.cpu())
            batch_active_token_count += (chunk_active > 0).sum().item()
            batch_total_token_count += chunk_active.size(0)
            
            # Accumulate to per-image sums
            # Reshape chunk to assign to images
            for i in range(curr_batch_size):
                img_start = i * num_tokens_per_img
                img_end = (i + 1) * num_tokens_per_img
                
                # Find overlap with current chunk
                chunk_img_start = max(0, img_start - token_start_idx)
                chunk_img_end = min(end - start, img_end - token_start_idx)
                
                if chunk_img_start < chunk_img_end and img_start < token_end_idx and img_end > token_start_idx:
                    local_start = max(0, token_start_idx - img_start)
                    local_end = min(num_tokens_per_img, token_end_idx - img_start)
                    
                    # Get the relevant portion of chunk_acts
                    rel_start = max(0, img_start - token_start_idx)
                    rel_end = min(end - start, img_end - token_start_idx)
                    
                    if rel_start < rel_end:
                        image_act_sums[i] += chunk_acts[rel_start:rel_end].sum(dim=0)
            
            del chunk_acts  # Free immediately
        
        total_tokens += batch_total_token_count
        active_tokens += batch_active_token_count
        
        if batch_token_counts_list:
            all_token_counts_list.append(torch.cat(batch_token_counts_list))
        
        # Now image_act_sums: (B, d_sae) - activation strength per concept per image
        for c in range(NUM_CLASSES):
            class_mask = (y == c)
            if class_mask.any():
                class_act_sums = image_act_sums[class_mask].sum(dim=0)
                concept_class_counts[:, c] += class_act_sums
                class_gap_sum[c] += image_act_sums[class_mask].sum(dim=0)
                class_img_count[c] += class_mask.sum()
        
        del image_act_sums, flat_tokens, fmap
        
        image_idx += curr_batch_size
    # END of torch.no_grad() block for concept evaluation
    
    # Process Concept Metrics (on CPU)
    concept_class_counts_cpu = concept_class_counts.cpu().numpy()
    total_activations = concept_class_counts_cpu.sum(axis=1)
    
    purity_list = []
    entropy_list = []
    class_specific_breakdown = {0: 0, 1: 0, 2: 0, 3: 0}
    high_purity_count = 0
    
    # Vectorized metrics
    valid_mask = total_activations >= 5
    if valid_mask.any():
        valid_counts = concept_class_counts_cpu[valid_mask]
        valid_totals = total_activations[valid_mask]
        
        # Purity
        max_counts = valid_counts.max(axis=1)
        max_classes = valid_counts.argmax(axis=1)
        purities = max_counts / valid_totals
        
        purity_list = purities.tolist()
        
        # High purity stats
        high_purity_mask = purities > 0.9
        high_purity_count = int(high_purity_mask.sum())
        
        hp_classes = max_classes[high_purity_mask]
        for c in range(NUM_CLASSES):
            class_specific_breakdown[c] = int((hp_classes == c).sum())
            
        # Entropy (Shannon entropy with base 2)
        # For 4 classes: max entropy = 2.0 (uniform), min = 0 (single class)
        probs = valid_counts / valid_totals[:, None]
        probs_safe = np.maximum(probs, 1e-10)
        entropies = -np.sum(probs * np.log2(probs_safe), axis=1)
        entropy_list = entropies.tolist()
        
        # Low entropy = class-specific concepts
        # Entropy < 1.0 means >50% of activation from one class
        low_entropy_count = int((entropies < 1.0).sum())
        very_low_entropy_count = int((entropies < 0.5).sum())  # >75% from one class
    
    purity_arr = np.array(purity_list) if purity_list else np.array([0.25])
    entropy_arr = np.array(entropy_list) if entropy_list else np.array([2.0])
    
    # Default values if no valid mask
    if not valid_mask.any():
        low_entropy_count = 0
        very_low_entropy_count = 0
    
    # Process Token Metrics
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
        
        if max_concepts > 1:
            normalized_mean = (mean_concepts - 1) / (max_concepts - 1)
        else:
            normalized_mean = 0.0
            
        pmf_counts = np.zeros(max_concepts + 1)
        vals, counts = np.unique(active_counts, return_counts=True)
        pmf_counts[vals.astype(int)] = counts
        pmf = pmf_counts[1:] / pmf_counts[1:].sum()
        p1 = float(pmf[0]) if len(pmf) > 0 else 0.0
    else:
        mean_concepts = std_concepts = gini = normalized_mean = p1 = 0.0
        max_concepts = 0
    
    # ===== Dual Mode Linear Probe Evaluation =====
    import gc
    
    logger.info("  Extracting SAE representations (Standard)...")
    X_train_std, y_train, _ = extract_sae_repr_for_probe(
        encoder, sae, train_loader, device, args.which_layer,
        token_l2_norm=args.token_l2_norm,
        strength_weighting=False,
        dead_threshold=float(args.dead_threshold),
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    X_test_std, y_test, _ = extract_sae_repr_for_probe(
        encoder, sae, test_loader, device, args.which_layer,
        token_l2_norm=args.token_l2_norm,
        strength_weighting=False,
        dead_threshold=float(args.dead_threshold),
    )
    
    # [MEM] Train Standard probe FIRST, then delete unused repr
    logger.info("  Training Linear Probe (Standard)...")
    probe_std = train_linear_probe(
        X_train_std, y_train, X_test_std, y_test,
        num_classes=NUM_CLASSES, epochs=50, lr=0.1, batch_size=256,
        normalize_repr=False, device=device, verbose=False
    )
    
    # [MEM] Delete standard repr before extracting strength repr
    del X_train_std, X_test_std
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info("  Extracting SAE representations (Strength-weighted)...")
    X_train_str, y_train_str, _ = extract_sae_repr_for_probe(
        encoder, sae, train_loader, device, args.which_layer,
        token_l2_norm=args.token_l2_norm,
        strength_weighting=True,
        dead_threshold=float(args.dead_threshold),
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    X_test_str, y_test_str, _ = extract_sae_repr_for_probe(
        encoder, sae, test_loader, device, args.which_layer,
        token_l2_norm=args.token_l2_norm,
        strength_weighting=True,
        dead_threshold=float(args.dead_threshold),
    )
    
    logger.info("  Training Linear Probe (Strength-weighted)...")
    probe_str = train_linear_probe(
        X_train_str, y_train_str, X_test_str, y_test_str,
        num_classes=NUM_CLASSES, epochs=50, lr=0.1, batch_size=256,
        normalize_repr=False, device=device, verbose=False
    )
    
    # [MEM] Cleanup strength repr
    del X_train_str, X_test_str
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info(f"  Probe: std={probe_std['test_acc']:.3f}, str={probe_str['test_acc']:.3f} (test set)")

    
    # Get FVU from trainer (if available)
    fvu = getattr(trainer, "best_fvu", None)
    if fvu is None:
        fvu = getattr(trainer, "last_fvu", 0.0)
    
    # Compute effective rank for concept diversity
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
        # Per-class breakdown (high purity concepts)
        "high_purity_Control": class_specific_breakdown[0],
        "high_purity_SNCA": class_specific_breakdown[1],
        "high_purity_GBA": class_specific_breakdown[2],
        "high_purity_LRRK2": class_specific_breakdown[3],
        # Purity stats
        "purity_mean": float(purity_arr.mean()),
        "purity_std": float(purity_arr.std()),
        "purity_median": float(np.median(purity_arr)),
        "purity_p90": float(np.percentile(purity_arr, 90)) if len(purity_arr) > 0 else 0.0,
        "purity_p95": float(np.percentile(purity_arr, 95)) if len(purity_arr) > 0 else 0.0,
        # Entropy stats (lower = more class-specific)
        "entropy_mean": float(entropy_arr.mean()),
        "entropy_std": float(entropy_arr.std()),
        "entropy_min": float(entropy_arr.min()) if len(entropy_arr) > 0 else 0.0,
        "entropy_p05": float(np.percentile(entropy_arr, 5)) if len(entropy_arr) > 0 else 2.0,
        "entropy_p10": float(np.percentile(entropy_arr, 10)) if len(entropy_arr) > 0 else 2.0,
        "low_entropy_concepts": low_entropy_count,      # entropy < 1.0
        "very_low_entropy_concepts": very_low_entropy_count,  # entropy < 0.5
        # Token activation stats
        "mean_concepts_per_token": mean_concepts,
        "std_concepts_per_token": std_concepts,
        "max_concepts_per_token": max_concepts,
        "gini_coefficient": gini,
        "normalized_mean": normalized_mean,
        "p_single_concept": p1,
        # Probe accuracy
    "probe_test_acc_standard": probe_std["test_acc"],
    "probe_test_acc_strength": probe_str["test_acc"],
    "probe_train_acc_standard": probe_std["train_acc"],
    "probe_train_acc_strength": probe_str["train_acc"],
        # FVU (reconstruction quality)
        "fvu": float(fvu) if fvu else 0.0,
        # Effective rank (concept diversity)
        "effective_rank": eff_rank_results["effective_rank"],
        "norm_effective_rank": eff_rank_results["normalized_effective_rank"],
    }
    
    # [MEM] Final cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # [NEW] Compute and save class-wise GAP means per concept
    class_gap_mean = torch.zeros((NUM_CLASSES, d_sae), dtype=torch.float32)
    for c in range(NUM_CLASSES):
        if class_img_count[c] > 0:
            class_gap_mean[c] = class_gap_sum[c].cpu() / class_img_count[c].item()
    
    # Exclude dead neurons
    dead_mask = sae.usage_ema.cpu() < float(args.dead_threshold)
    alive_indices = (~dead_mask).nonzero(as_tuple=True)[0].tolist()
    
    # Save to CSV: rows = concepts, columns = [concept_id, Control, SNCA, GBA, LRRK2]
    class_names = ["Control", "SNCA", "GBA", "LRRK2"]
    
    # Filename with parameters (matching eval_results naming convention)
    tie_str = "tied" if sae.tie_weights else "untied"
    sparsity_val = float(exp_config.get("sparsity", 0.0))
    aux_val = float(exp_config.get("aux_coeff", 0.0))
    gap_csv_filename = f"gated_sae_{args.which_layer}_d{d_sae}_sp{sparsity_val}_aux{aux_val}_{tie_str}_class_gap_means.csv"
    gap_csv_path = os.path.join(args.sae_save_dir, gap_csv_filename)
    
    with open(gap_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["concept_id", "is_alive"] + class_names + ["max_class", "class_diff", "entropy"])
        
        for i in range(d_sae):
            is_alive = int(i in alive_indices)
            gaps = [float(class_gap_mean[c, i]) for c in range(NUM_CLASSES)]
            
            max_class = class_names[np.argmax(gaps)] if max(gaps) > 0 else "None"
            class_diff = max(gaps) - min(gaps)  # Difference between max and min class
            
            # Entropy of class distribution
            total = sum(gaps) + 1e-10
            probs = [g / total for g in gaps]
            ent = -sum(p * np.log2(max(p, 1e-10)) for p in probs)
            
            w.writerow([i, is_alive] + gaps + [max_class, f"{class_diff:.4f}", f"{ent:.4f}"])
    
    logger.info(f"[NEW] Saved class-wise GAP means to: {gap_csv_path}")
    logger.info(f"      Alive concepts: {len(alive_indices)}/{d_sae}, Columns: {class_names}")
        
    return results


class DummyTrainer:
    """Minimal trainer-like object for evaluation."""
    def __init__(self, encoder, sae, device, best_fvu=0.0):
        self.encoder = encoder
        self.sae = sae
        self.device = device
        self.best_fvu = best_fvu


def main(args_list=None):
    """Main entry point - supports multi-layer training, grid search, and auto-evaluation."""
    args = resolve_paths(get_args(args_list))
    args.log_fvu = True
    args.use_val = True
    args.use_test = True
    args.token_l2_norm = True
    args.save_best = True
    args.augment = True  # ??Enable rot90 to match backbone training
    
    # Linear Probe HP (Match Backbone exactly)
    args.probe_epochs = 50
    args.probe_lr = 0.1
    args.probe_batch_size = 256
    
    # ============== Optimization Flags (Match Backbone) ==============
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except:
        pass

    # ============== Configuration ==============
    run_grid_search = bool(getattr(args, "grid_search", False))
    train_all_layers = bool(getattr(args, "train_all_layers", True))
    eval_gap_random = bool(getattr(args, "eval_gap_random", False))  # GAP@random on/off
    
    # Determine which layers to train
    if train_all_layers:
        layers_to_train = ALL_LAYERS
    else:
        layers_to_train = [args.which_layer]
    
    logger.info("=" * 60)
    logger.info("Gated SAE Training + Concept Evaluation")
    logger.info(f"Layers to train: {layers_to_train}")
    logger.info(f"Grid search: {run_grid_search}")
    logger.info(f"GAP@Random eval: {eval_gap_random}")
    logger.info("=" * 60)
    
    # Load data once
    refs = load_all_sample_refs(args.shard_root)
    uid_to_refidx = build_uid_to_refidx(refs)
    
    # Load encoder once
    blocks = parse_int_list(args.blocks, 4)
    dilations = parse_int_list(args.dilations, 4)
    model = SupConMoCoModel(
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
    from sae_project.step05_model_encoder import robust_load_state_dict
    robust_load_state_dict(model, sd, strict=True)
    encoder = model.encoder
    renorm_unit_per_out_channel_(encoder)  # Weight normalization
    encoder = encoder.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    # [MEM FIX] Delete full model and state_dict to free VRAM
    import gc
    del model
    del sd
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # ============== [NEW] Standalone Evaluation Mode ==============
    if getattr(args, "eval_ckpt", None):
        ckpt_path = args.eval_ckpt
        logger.info(f"\n[STANDALONE EVALUATION] Loading SAE from: {ckpt_path}")
        
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ckpt_args = ckpt["args"]
        
        # Instantiate SAE with ckpt config
        sae = GatedSAE(
            d_in=ckpt_args.get("d_in", args.d_in),
            d_sae=ckpt_args.get("d_sae", args.d_sae),
            tie_weights=ckpt_args.get("tie_gate_weights", False),
            aux_k=ckpt_args.get("aux_k", 32),
        )
        sae.load_state_dict(ckpt["sae"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sae.to(device).eval()
        
        # [MEM] Clear GPU cache after loading models
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"[MEM] After model loading: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated")
        
        # Create dummy trainer for evaluation function
        trainer = DummyTrainer(encoder, sae, device, best_fvu=ckpt.get("best_fvu", 0.0))
        
        # Setup evaluation loaders
        train_csv = os.path.join(args.save_dir, "train_split.csv")
        test_csv = os.path.join(args.save_dir, "test_split.csv")
        
        train_eval_loader = _make_loader_from_split(
            args, refs, uid_to_refidx, split_csv_path=train_csv,
            batch_size=args.batch_size, augment=False, shuffle=False, strict_balance=False, seed=args.seed
        )
        test_eval_loader = _make_loader_from_split(
            args, refs, uid_to_refidx, split_csv_path=test_csv,
            batch_size=args.batch_size, augment=False, shuffle=False, strict_balance=False, seed=args.seed
        )
        
        exp_config = {
            "exp_id": "eval_only",
            "layer": ckpt_args.get("which_layer", args.which_layer),
            "d_sae": sae.d_sae,
            "sparsity": ckpt_args.get("final_sparsity_coeff", 0.0),
            "aux_coeff": ckpt_args.get("aux_coeff", 0.0),
            "tie_weights": sae.tie_weights,
        }
        
        # Ensure args.which_layer matches for extract_sae_repr_for_probe
        args.which_layer = exp_config["layer"]
        
        logger.info("Starting evaluation...")
        results = evaluate_concepts_for_sae(trainer, train_eval_loader, test_eval_loader, args, exp_config)
        
        logger.info(f"\nEvaluation Results for {ckpt_path}:")
        logger.info(f"  Weighted Purity: mean={results['purity_mean']:.3f}, high(>0.9)={results['high_purity_concepts']}")
        logger.info(f"  Entropy: mean={results['entropy_mean']:.3f}, low(<1.0)={results['low_entropy_concepts']}, v.low(<0.5)={results['very_low_entropy_concepts']}")
        logger.info(f"  Probe: test_acc_std={results['probe_test_acc_standard']:.3f}, test_acc_str={results['probe_test_acc_strength']:.3f}")
        logger.info(f"  Effective Rank: {results['effective_rank']:.2f}")
        
        # Save results to a specific file
        eval_results_path = ckpt_path.replace(".pt", "_eval_results.csv")
        with open(eval_results_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(results.keys())
            w.writerow(results.values())
        logger.info(f"Saved eval results to: {eval_results_path}")
        
        return # Exit after evaluation
    # ==============================================================

    # Summary CSV for all experiments
    summary_csv_path = os.path.join(args.sae_save_dir, "all_experiments_summary.csv")
    summary_rows = []
    
    if run_grid_search:
        # ============== Grid Search ==============
        D_SAE_GRID = args.d_sae_grid
        SPARSITY_GRID = args.sparsity_grid
        AUX_COEFF_GRID = args.aux_coeff_grid
        TIE_WEIGHTS_GRID = [bool(tw) for tw in args.tie_weights_grid]
        
        total_experiments = len(layers_to_train) * len(D_SAE_GRID) * len(SPARSITY_GRID) * len(AUX_COEFF_GRID) * len(TIE_WEIGHTS_GRID)
        exp_idx = 0
        
        logger.info(f"Grid: d_sae={D_SAE_GRID}, Sparsity={SPARSITY_GRID}, Aux={AUX_COEFF_GRID}, Tied={TIE_WEIGHTS_GRID}")
        logger.info(f"Total experiments: {total_experiments}")
        
        for layer in layers_to_train:
            for d_sae, sparsity, aux_coeff, tie_weights in itertools.product(
                D_SAE_GRID, SPARSITY_GRID, AUX_COEFF_GRID, TIE_WEIGHTS_GRID
            ):
                exp_idx += 1
                logger.info("\n" + "=" * 60)
                logger.info(f"[Experiment {exp_idx}/{total_experiments}]")
                logger.info(f"  layer={layer}, d_sae={d_sae}, sparsity={sparsity}, aux_coeff={aux_coeff}, tie_weights={tie_weights}")
                logger.info("=" * 60)
                
                import copy
                exp_args = copy.deepcopy(args)
                exp_args.which_layer = layer
                exp_args.d_sae = d_sae
                exp_args.final_sparsity_coeff = sparsity
                exp_args.aux_coeff = aux_coeff
                exp_args.tie_gate_weights = tie_weights
                
                exp_config = {
                    "exp_id": exp_idx,
                    "layer": layer,
                    "d_sae": d_sae,
                    "sparsity": sparsity,
                    "aux_coeff": aux_coeff,
                    "tie_weights": tie_weights,
                }
                
                set_seed(exp_args.seed)
                
                try:
                    # Train SAE
                    trainer, ckpt_path = run_experiment(exp_args, refs, uid_to_refidx, encoder)
                    
                    # [MEM] Clear trainer state to free VRAM before evaluation
                    trainer.cleanup_for_eval()
                    
                    # Evaluate concepts + probe on train/test sets
                    train_csv = os.path.join(args.save_dir, "train_split.csv")
                    test_csv = os.path.join(args.save_dir, "test_split.csv")
                    if os.path.exists(test_csv) and os.path.exists(train_csv):
                        train_eval_loader = _make_loader_from_split(
                            exp_args, refs, uid_to_refidx,
                            split_csv_path=train_csv,
                            batch_size=args.batch_size,
                            augment=False,
                            shuffle=False,
                            strict_balance=True,  # ??Ensure balanced evaluation
                            seed=args.seed + 100
                        )
                        test_eval_loader = _make_loader_from_split(
                            exp_args, refs, uid_to_refidx,
                            split_csv_path=test_csv,
                            batch_size=args.batch_size,
                            augment=False,
                            shuffle=False,
                            strict_balance=True,  # ??Ensure balanced evaluation
                            seed=args.seed + 101
                        )
                        
                        logger.info("Evaluating concept metrics + probe...")
                        results = evaluate_concepts_for_sae(trainer, train_eval_loader, test_eval_loader, exp_args, exp_config)
                        summary_rows.append(results)
                        
                        # Print summary
                        logger.info(f"  Weighted Purity: mean={results['purity_mean']:.3f}, high(>0.9)={results['high_purity_concepts']}")
                        logger.info(f"  Entropy: mean={results['entropy_mean']:.3f}, low(<1.0)={results['low_entropy_concepts']}, v.low(<0.5)={results['very_low_entropy_concepts']}")
                        logger.info(f"  Probe: test_acc_std={results['probe_test_acc_standard']:.3f}, str={results['probe_test_acc_strength']:.3f}")
                    
                except Exception as e:
                    logger.error(f"Experiment failed: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
    else:
        # ============== Train All Layers (no grid search) ==============
        for layer in layers_to_train:
            logger.info("\n" + "=" * 60)
            logger.info(f"Training Gated SAE for layer: {layer}")
            logger.info("=" * 60)
            
            import copy
            exp_args = copy.deepcopy(args)
            exp_args.which_layer = layer
            
            exp_config = {
                "exp_id": 1,
                "layer": layer,
                "sparsity": args.final_sparsity_coeff,
                "aux_coeff": args.aux_coeff,
                "tie_weights": getattr(args, "tie_gate_weights", False),
            }
            
            set_seed(exp_args.seed)
            
            try:
                trainer, ckpt_path = run_experiment(exp_args, refs, uid_to_refidx, encoder)
                
                # [MEM] Clear trainer state to free VRAM before evaluation
                trainer.cleanup_for_eval()
                
                # Evaluate concepts + probe
                train_csv = os.path.join(args.save_dir, "train_split.csv")
                test_csv = os.path.join(args.save_dir, "test_split.csv")
                if os.path.exists(test_csv) and os.path.exists(train_csv):
                    train_eval_loader = _make_loader_from_split(
                        exp_args, refs, uid_to_refidx,
                        split_csv_path=train_csv,
                        batch_size=args.batch_size,
                        augment=False,
                        shuffle=False,
                        strict_balance=True,
                        seed=args.seed + 100
                    )
                    test_eval_loader = _make_loader_from_split(
                        exp_args, refs, uid_to_refidx,
                        split_csv_path=test_csv,
                        batch_size=args.batch_size,
                        augment=False,
                        shuffle=False,
                        strict_balance=True,
                        seed=args.seed + 101
                    )
                    
                    logger.info("Evaluating concept metrics + probe...")
                    results = evaluate_concepts_for_sae(trainer, train_eval_loader, test_eval_loader, exp_args, exp_config)
                    summary_rows.append(results)
                    
                    logger.info(f"  Weighted Purity: mean={results['purity_mean']:.3f}, high(>0.9)={results['high_purity_concepts']}")
                    logger.info(f"  Entropy: mean={results['entropy_mean']:.3f}, low(<1.0)={results['low_entropy_concepts']}, v.low(<0.5)={results['very_low_entropy_concepts']}")
                    logger.info(f"  Probe: test_acc_std={results['probe_test_acc_standard']:.3f}, str={results['probe_test_acc_strength']:.3f}")
                    
            except Exception as e:
                logger.error(f"Layer {layer} training failed: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Save summary CSV (append mode - accumulate results across runs)
    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        file_exists = os.path.exists(summary_csv_path)
        with open(summary_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(summary_rows)
        logger.info(f"\nAppended {len(summary_rows)} experiment(s) to: {summary_csv_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("All experiments complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
