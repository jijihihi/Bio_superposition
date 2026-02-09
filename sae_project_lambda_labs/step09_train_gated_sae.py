# ==============================================================================
# Gated SAE Trainer with Hyperparameter Optimization
# - L2-weighted reconstruction loss
# - Sparsity warmup schedule
# - lr 도 스케쥴러 선택가능. lienar cosine. 나는 cosine으로 학습시킴.
# --dead_threshold로 dead neuron 기준 바꿀 수 있다. 
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
from sae_project.step05_model_encoder import SupMoCoModel, parse_int_list, renorm_unit_per_out_channel_
from sae_project.step06_gated_sae import GatedSAE, get_sparsity_coeff

from sae_project.step09_sae_eval import (
    LinearProbe, extract_sae_repr_for_probe, train_linear_probe,
    compute_effective_rank, evaluate_concepts_for_sae, DummyTrainer,
)

logger = get_logger("train_gated_sae")


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
    
    dead_1e6 = int((u < 1e-6).sum().item())
    dead_1e5 = int((u < 1e-5).sum().item())
    dead_1e4 = int((u < 1e-4).sum().item())
    dead_1e3 = int((u < 1e-3).sum().item())
    dead_1e2 = int((u < 1e-2).sum().item())
    dead_custom = int((u < dead_threshold).sum().item())
    
    qs = torch.tensor([0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0], device=u.device)
    p = torch.quantile(u, qs).tolist()
    
    very_low = int((u < 0.01).sum().item())
    low = int(((u >= 0.01) & (u < 0.1)).sum().item())
    medium = int(((u >= 0.1) & (u < 0.5)).sum().item())
    high = int((u >= 0.5).sum().item())
    
    return {
        "total_features": d,
        "dead": dead_custom,
        "dead_frac": float(dead_custom / max(1, d)),
        "dead_1e-6": dead_1e6,
        "dead_1e-5": dead_1e5,
        "dead_1e-4": dead_1e4,
        "dead_1e-3": dead_1e3,
        "dead_1e-2": dead_1e2,
        "freq_very_low": very_low,
        "freq_low": low,
        "freq_medium": medium,
        "freq_high": high,
        "min": float(p[0]), "p01": float(p[1]), "p05": float(p[2]), "p10": float(p[3]),
        "p25": float(p[4]), "p50": float(p[5]), "p75": float(p[6]),
        "p90": float(p[7]), "p95": float(p[8]), "p99": float(p[9]), "max": float(p[10]),
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
        self.initial_aux_coeff = self.aux_coeff

        # Create unique experiment name
        exp_name = f"gated_sp{self.final_sparsity_coeff}_aux{self.aux_coeff}"
        if getattr(args, "tie_gate_weights", False):
            exp_name += "_tied"
        if getattr(args, "use_clustering_init", False):
            exp_name += "_clust"

        os.makedirs(args.sae_save_dir, exist_ok=True)
        
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

        # Gradient accumulation
        self.grad_accum_steps = int(getattr(args, "gradient_accumulation_steps", 1))
        if self.grad_accum_steps < 1:
            self.grad_accum_steps = 1

        batches_per_epoch = len(train_loader)
        self.total_steps = args.epochs * (batches_per_epoch // self.grad_accum_steps)

        # LR Scheduler
        self.lr_scheduler_type = getattr(args, "lr_scheduler", "cosine")
        self.lr_warmup_fraction = getattr(args, "lr_warmup_fraction", 0.1)
        self.lr_warmup_steps = int(self.total_steps * self.lr_warmup_fraction)
        self.base_lr = args.sae_lr
        
        if self.lr_scheduler_type == "cosine":
            from torch.optim.lr_scheduler import LambdaLR
            import math
            
            def lr_lambda(step):
                if step < self.lr_warmup_steps:
                    return step / max(1, self.lr_warmup_steps)
                else:
                    progress = (step - self.lr_warmup_steps) / max(1, self.total_steps - self.lr_warmup_steps)
                    return 0.5 * (1.0 + math.cos(math.pi * progress))
            
            self.scheduler = LambdaLR(self.opt, lr_lambda)
        elif self.lr_scheduler_type == "linear":
            from torch.optim.lr_scheduler import LambdaLR
            
            def lr_lambda(step):
                if step < self.lr_warmup_steps:
                    return step / max(1, self.lr_warmup_steps)
                else:
                    progress = (step - self.lr_warmup_steps) / max(1, self.total_steps - self.lr_warmup_steps)
                    return max(0.0, 1.0 - progress)
            
            self.scheduler = LambdaLR(self.opt, lr_lambda)
        else:
            self.scheduler = None

        logger.info(f"[GatedSAE] tie_weights={getattr(args, 'tie_gate_weights', False)}, "
                    f"aux_coeff={self.aux_coeff}, final_sparsity={self.final_sparsity_coeff}")
        logger.info(f"[GatedSAE] sparsity_warmup_steps={self.sparsity_warmup_steps}, total_steps={self.total_steps}")
        logger.info(f"[GatedSAE] gradient_accumulation_steps={self.grad_accum_steps}, effective_batch={args.batch_size * self.grad_accum_steps}")
        logger.info(f"[GatedSAE] lr_scheduler={self.lr_scheduler_type}, lr_warmup_steps={self.lr_warmup_steps}, weight_decay={args.sae_wd}")

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
        l2_norms = tokens.norm(dim=1)
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
        """Compute Gated SAE loss with L2-weighted MSE."""
        recon, acts, gate_pre, recon_aux, acts_aux = self.sae(tokens)

        tok_f = tokens.float()
        rec_f = recon.float()
        l2_f = l2_norms.float()

        mse = (rec_f - tok_f).pow(2).sum(dim=1)
        weights = l2_f / (l2_f.mean() + 1e-8)
        L_recon = (mse * weights).mean()

        L_sparsity = self.sae.compute_sparsity_loss(gate_pre)

        if self.aux_coeff > 0 and recon_aux.abs().sum() > 0:
            residual = tok_f - rec_f.detach()
            aux_error = (recon_aux.float() - residual).pow(2).sum(dim=1)
            L_aux = (aux_error * weights).mean()
        else:
            L_aux = torch.tensor(0.0, device=tokens.device)

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
        """Release GPU VRAM used by optimizer and training state before evaluation."""
        import gc
        
        if hasattr(self, "opt") and self.opt is not None:
            del self.opt
            self.opt = None
        
        if hasattr(self, "scaler") and self.scaler is not None:
            del self.scaler
            self.scaler = None
        
        for loader in [self.train_loader, self.val_loader, self.test_loader]:
            if loader is not None and hasattr(loader, "dataset"):
                ds = loader.dataset
                if hasattr(ds, "bank") and hasattr(ds.bank, "images"):
                    ds.bank.images = None
        
        gc.collect()
        
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
            
            current_aux = self.aux_coeff

            train_recon_sum, train_sparsity_sum, train_aux_sum, train_steps = 0.0, 0.0, 0.0, 0
            train_sse_sum, train_sst_sum = 0.0, 0.0
            dead_count_epoch = 0

            accum_recon, accum_sparsity, accum_aux = 0.0, 0.0, 0.0
            accum_count = 0
            self.opt.zero_grad(set_to_none=True)

            pbar = tqdm(self.train_loader, desc=f"GatedSAE Train E{epoch}/{self.args.epochs}", leave=True)
            for batch_idx, batch in enumerate(pbar):
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

                current_sparsity = get_sparsity_coeff(
                    self.global_step,
                    self.sparsity_warmup_steps,
                    self.final_sparsity_coeff,
                    self.total_steps,
                    ramp_fraction=getattr(self.args, 'sparsity_ramp_fraction', 0.1)
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

                    loss = loss / float(chunks_count * self.grad_accum_steps)

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

                recon_step = recon_acc / max(1, chunks)
                sparsity_step = sparsity_acc / max(1, chunks)
                aux_step = aux_acc / max(1, chunks)
                
                accum_recon += recon_step
                accum_sparsity += sparsity_step
                accum_aux += aux_step
                accum_count += 1

                if accum_count >= self.grad_accum_steps:
                    if not self.use_bf16:
                        self.scaler.unscale_(self.opt)
                    
                    self.sae.project_decoder_grads_()
                    
                    if self.args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.sae.parameters(), float(self.args.grad_clip))
                    
                    if self.use_bf16:
                        self.opt.step()
                    else:
                        self.scaler.step(self.opt)
                        self.scaler.update()

                    self.sae.renorm_decoder_()

                    dead_count = int((self.sae.usage_ema < float(self.args.dead_threshold)).sum().item())
                    dead_count_epoch = dead_count

                    if self.scheduler is not None:
                        self.scheduler.step()

                    self.global_step += 1

                    avg_recon = accum_recon / accum_count
                    avg_sparsity = accum_sparsity / accum_count
                    avg_aux = accum_aux / accum_count

                    train_recon_sum += avg_recon
                    train_sparsity_sum += avg_sparsity
                    train_aux_sum += avg_aux
                    train_steps += 1

                    current_lr = self.opt.param_groups[0]['lr']
                    pbar.set_postfix({
                        "recon": f"{avg_recon:.4f}",
                        "sp": f"{avg_sparsity:.4f}",
                        "aux": f"{avg_aux:.4f}",
                        "λsp": f"{current_sparsity:.2f}",
                        "dead": dead_count,
                        "lr": f"{current_lr:.2e}",
                    })

                    if (self.global_step % int(self.args.save_every)) == 0:
                        self.save_ckpt(self.ckpt_path)

                    accum_recon, accum_sparsity, accum_aux = 0.0, 0.0, 0.0
                    accum_count = 0
                    self.opt.zero_grad(set_to_none=True)
                else:
                    pbar.set_postfix({
                        "recon": f"{recon_step:.4f}",
                        "sp": f"{sparsity_step:.4f}",
                        "aux": f"{aux_step:.4f}",
                        "λsp": f"{current_sparsity:.2f}",
                        "αaux": f"{current_aux:.4f}",
                        "acc": f"{accum_count}/{self.grad_accum_steps}",
                    })

            # Handle remaining accumulated gradients
            if accum_count > 0:
                if not self.use_bf16:
                    self.scaler.unscale_(self.opt)
                self.sae.project_decoder_grads_()
                if self.args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.sae.parameters(), float(self.args.grad_clip))
                if self.use_bf16:
                    self.opt.step()
                else:
                    self.scaler.step(self.opt)
                    self.scaler.update()
                self.sae.renorm_decoder_()
                self.global_step += 1
                
                avg_recon = accum_recon / accum_count
                avg_sparsity = accum_sparsity / accum_count
                avg_aux = accum_aux / accum_count
                train_recon_sum += avg_recon
                train_sparsity_sum += avg_sparsity
                train_aux_sum += avg_aux
                train_steps += 1
                self.opt.zero_grad(set_to_none=True)

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

            self.save_ckpt(self.ckpt_path)
            
            epoch_ckpt_path = self.ckpt_path.replace(".pt", f"_ep{epoch:03d}.pt")
            self.save_ckpt(epoch_ckpt_path)
            tqdm.write(f"  -> Saved Epoch {epoch} to {os.path.basename(epoch_ckpt_path)}")

            if getattr(self.args, "save_best", False):
                if metric < self.best_metric:
                    self.best_metric = float(metric)
                    self.best_fvu = test_fvu if test_fvu else val_fvu
                    self.save_ckpt(self.best_ckpt_path)
                    tqdm.write(f"  -> Saved BEST to {self.best_ckpt_path} (metric={self.best_metric:.6f})")
            
            self.last_fvu = test_fvu if test_fvu else val_fvu

        logger.info(f"[GatedSAE] Done. Saved -> {self.ckpt_path}")


def _make_loader_from_split(args, refs, uid_to_refidx, split_csv_path: str,
                           batch_size: int, augment: bool, shuffle: bool,
                           strict_balance: bool, seed: int, explicit_4x_augment: bool = False):
    if not os.path.exists(split_csv_path):
        return None

    uids = load_split_csv(split_csv_path)
    missing = [u for u in uids if u not in uid_to_refidx]
    if len(missing) > 0:
        raise RuntimeError(f"Some uids in split are missing under current shard_root. ex: {missing[:5]}")
    refidx = [uid_to_refidx[u] for u in uids]

    bank = InMemoryTarBank(refs, refidx, args.img_size)
    ib = list(range(len(refidx)))
    ds = InMemorySixteenBitDataset(bank, ib, args.img_size, augment=augment, explicit_4x_augment=explicit_4x_augment)
    
    if explicit_4x_augment:
        logger.info(f"[DataLoader] explicit_4x_augment=True: {len(refidx)} images → {len(ds)} samples (4x)")

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
    if refs is None:
        refs = load_all_sample_refs(args.shard_root)
    if uid_to_refidx is None:
        uid_to_refidx = build_uid_to_refidx(refs)

    train_csv = os.path.join(args.save_dir, "train_split.csv")
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"train_split.csv not found: {train_csv}")

    strict = True
    use_explicit_4x = getattr(args, "explicit_4x_augment", False)

    train_loader = _make_loader_from_split(
        args, refs, uid_to_refidx,
        split_csv_path=train_csv,
        batch_size=int(args.batch_size),
        augment=True if not use_explicit_4x else False,
        shuffle=not strict,
        strict_balance=strict,
        seed=int(args.seed),
        explicit_4x_augment=use_explicit_4x,
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

    if encoder is None:
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
        from sae_project.step05_model_encoder import robust_load_state_dict
        robust_load_state_dict(model, sd, strict=True)
        encoder = model.encoder

    trainer = GatedSAETrainer(args, encoder, train_loader, val_loader=val_loader, test_loader=test_loader)
    trainer.train()
    
    return trainer, trainer.ckpt_path


ALL_LAYERS = ["stage5_out", "refine_out"]


def _make_eval_loaders(args, refs, uid_to_refidx):
    """Create train/val/test loaders for post-training evaluation."""
    train_csv = os.path.join(args.save_dir, "train_split.csv")
    val_csv = os.path.join(args.save_dir, "val_split.csv")
    test_csv = os.path.join(args.save_dir, "test_split.csv")
    
    train_eval_loader = _make_loader_from_split(
        args, refs, uid_to_refidx, split_csv_path=train_csv,
        batch_size=args.batch_size, augment=False, shuffle=False,
        strict_balance=True, seed=args.seed + 100
    )
    val_eval_loader = None
    if os.path.exists(val_csv):
        val_eval_loader = _make_loader_from_split(
            args, refs, uid_to_refidx, split_csv_path=val_csv,
            batch_size=args.batch_size, augment=False, shuffle=False,
            strict_balance=True, seed=args.seed + 102
        )
    test_eval_loader = None
    if os.path.exists(test_csv):
        test_eval_loader = _make_loader_from_split(
            args, refs, uid_to_refidx, split_csv_path=test_csv,
            batch_size=args.batch_size, augment=False, shuffle=False,
            strict_balance=True, seed=args.seed + 101
        )
    return train_eval_loader, val_eval_loader, test_eval_loader


def main(args_list=None):
    """Main entry point - supports multi-layer training, grid search, and auto-evaluation."""
    args = resolve_paths(get_args(args_list))
    args.log_fvu = True
    args.use_val = True
    args.use_test = True
    args.token_l2_norm = True
    args.save_best = True
    args.augment = True
    
    args.probe_epochs = 20
    args.probe_lr = 0.1
    args.probe_batch_size = 256
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except:
        pass

    run_grid_search = bool(getattr(args, "grid_search", False))
    train_all_layers = bool(getattr(args, "train_all_layers", True))
    
    layers_to_train = ALL_LAYERS if train_all_layers else [args.which_layer]
    
    logger.info("=" * 60)
    logger.info("Gated SAE Training + Concept Evaluation")
    logger.info(f"Layers to train: {layers_to_train}")
    logger.info(f"Grid search: {run_grid_search}")
    logger.info("=" * 60)
    
    # Load data once
    refs = load_all_sample_refs(args.shard_root)
    uid_to_refidx = build_uid_to_refidx(refs)
    
    # Load encoder once
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
    from sae_project.step05_model_encoder import robust_load_state_dict
    robust_load_state_dict(model, sd, strict=True)
    encoder = model.encoder
    renorm_unit_per_out_channel_(encoder)
    encoder = encoder.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    import gc
    del model, sd
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Standalone Evaluation Mode
    if getattr(args, "eval_ckpt", None):
        ckpt_path = args.eval_ckpt
        logger.info(f"\n[STANDALONE EVALUATION] Loading SAE from: {ckpt_path}")
        
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ckpt_args = ckpt["args"]
        
        sae = GatedSAE(
            d_in=ckpt_args.get("d_in", args.d_in),
            d_sae=ckpt_args.get("d_sae", args.d_sae),
            tie_weights=ckpt_args.get("tie_gate_weights", False),
            aux_k=ckpt_args.get("aux_k", 32),
        )
        sae.load_state_dict(ckpt["sae"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sae.to(device).eval()
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        trainer = DummyTrainer(encoder, sae, device, best_fvu=ckpt.get("best_fvu", 0.0))
        args.which_layer = ckpt_args.get("which_layer", args.which_layer)
        
        train_eval_loader, val_eval_loader, test_eval_loader = _make_eval_loaders(args, refs, uid_to_refidx)
        
        exp_config = {
            "exp_id": "eval_only",
            "layer": args.which_layer,
            "d_sae": sae.d_sae,
            "sparsity": ckpt_args.get("final_sparsity_coeff", 0.0),
            "aux_coeff": ckpt_args.get("aux_coeff", 0.0),
            "tie_weights": sae.tie_weights,
        }
        
        results = evaluate_concepts_for_sae(trainer, train_eval_loader, val_eval_loader, test_eval_loader, args, exp_config)
        
        logger.info(f"\nEvaluation Results for {ckpt_path}:")
        logger.info(f"  Purity: mean={results['purity_mean']:.3f}, high(>0.9)={results['high_purity_concepts']}")
        logger.info(f"  Entropy: mean={results['entropy_mean']:.3f}, low(<1.0)={results['low_entropy_concepts']}")
        logger.info(f"  Probe: std={results['probe_test_acc_standard']:.3f}, str={results['probe_test_acc_strength']:.3f}")
        
        eval_results_path = ckpt_path.replace(".pt", "_eval_results.csv")
        with open(eval_results_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(results.keys())
            w.writerow(results.values())
        logger.info(f"Saved eval results to: {eval_results_path}")
        return

    # Summary CSV
    summary_csv_path = os.path.join(args.sae_save_dir, "all_experiments_summary.csv")
    summary_rows = []
    
    if run_grid_search:
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
                    "exp_id": exp_idx, "layer": layer, "d_sae": d_sae,
                    "sparsity": sparsity, "aux_coeff": aux_coeff, "tie_weights": tie_weights,
                }
                
                set_seed(exp_args.seed)
                
                try:
                    trainer, ckpt_path = run_experiment(exp_args, refs, uid_to_refidx, encoder)
                    trainer.cleanup_for_eval()
                    
                    train_eval_loader, val_eval_loader, test_eval_loader = _make_eval_loaders(exp_args, refs, uid_to_refidx)
                    if train_eval_loader is not None and test_eval_loader is not None:
                        results = evaluate_concepts_for_sae(trainer, train_eval_loader, val_eval_loader, test_eval_loader, exp_args, exp_config)
                        summary_rows.append(results)
                        logger.info(f"  Purity: mean={results['purity_mean']:.3f}, high(>0.9)={results['high_purity_concepts']}")
                        logger.info(f"  Probe: std={results['probe_test_acc_standard']:.3f}, str={results['probe_test_acc_strength']:.3f}")
                    
                except Exception as e:
                    logger.error(f"Experiment failed: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
    else:
        for layer in layers_to_train:
            logger.info("\n" + "=" * 60)
            logger.info(f"Training Gated SAE for layer: {layer}")
            logger.info("=" * 60)
            
            import copy
            exp_args = copy.deepcopy(args)
            exp_args.which_layer = layer
            
            exp_config = {
                "exp_id": 1, "layer": layer,
                "sparsity": args.final_sparsity_coeff,
                "aux_coeff": args.aux_coeff,
                "tie_weights": getattr(args, "tie_gate_weights", False),
            }
            
            set_seed(exp_args.seed)
            
            try:
                trainer, ckpt_path = run_experiment(exp_args, refs, uid_to_refidx, encoder)
                trainer.cleanup_for_eval()
                
                train_eval_loader, val_eval_loader, test_eval_loader = _make_eval_loaders(exp_args, refs, uid_to_refidx)
                if train_eval_loader is not None and test_eval_loader is not None:
                    results = evaluate_concepts_for_sae(trainer, train_eval_loader, val_eval_loader, test_eval_loader, exp_args, exp_config)
                    summary_rows.append(results)
                    logger.info(f"  Purity: mean={results['purity_mean']:.3f}, high(>0.9)={results['high_purity_concepts']}")
                    logger.info(f"  Probe: std={results['probe_test_acc_standard']:.3f}, str={results['probe_test_acc_strength']:.3f}")
                    
            except Exception as e:
                logger.error(f"Layer {layer} training failed: {e}")
                import traceback
                traceback.print_exc()
                continue
    
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
