# ==============================================================================
# Supervised MoCo (large queue)  + strict Plate/Line balanced batching
# - EMA teacher (model_k) produces keys k (no grad)
# - Large FIFO queue stores (feat, label) -> K, YK
# - Loss uses your supervised_contrastive_q_vs_k 그대로: q vs (k + queue)
# - XBM 제거
# - Optimizer: SGD(+momentum) 추천 (renorm 매 step과 궁합)
# - Optional symmetric loss
# - Per-epoch Linear Probe(3 epochs) for early stopping (그대로 유지)
# ==============================================================================


## CNN encoder 구조.

## stem (3->64. resblock 없음) 그 뒤로 resblock 있음
# stage2 (64 -> 128) 2
# stage3 (128 -> 256) 2
# stage4 (256 -> 512) 3
# stage5 (512 -> 512) 3  resblock은 4개니까. stage4_out
# refine (512 -> 512) 1  refinement block output

# "We extract intermediate feature maps from two locations in the encoder:
# the output of the final backbone stage (f_stage4, 512 channels)
# and the output of the additional residual block (f_refine, 512 channels).
# Each spatial position in these feature maps serves as an input token
# to the Sparse Autoencoder."

# "We extract intermediate feature maps from two specific layers in the encoder:
# the final backbone stage (f_stage4, 512 channels)
# and the subsequent refinement residual block (f_refine, 512 channels).
# Each spatial position within these feature maps is treated as an independent input token
# or training and evaluating the Sparse Autoencoder (SAE)."


import argparse
import csv
import glob
import io
import json
import logging
import math
import os
import pickle
import random
import re
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.checkpoint import checkpoint_sequential
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
from tqdm.auto import tqdm


# --- Inserted Imports from Modularized Files ---
from run_CNN.logging_utils import get_logger, SUPERCLASS_MAP, CLASS_TO_LABEL, PLATE_DIR_RE, DEFAULT_SHARD_ROOT
from run_CNN.data_shards import load_all_sample_refs, SampleRef
from run_CNN.data_bank import (
    InMemoryTarBank, InMemorySixteenBitDataset, StrictPlateBalancedBatchSamplerOnBank,
    load_split_csv, seed_worker, collate_skip_none
)
from run_CNN.model_encoder import (
    Encoder, SupMoCoModel, build_projector, parse_int_list, 
    OUT_DIM, renorm_unit_per_out_channel_
)
# -----------------------------------------------
# ==============================================================================
# Logging
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("SupMoCoQueue_Plate")


def _add_file_logger(save_dir: str):
    """Add a FileHandler so every log line is persisted to save_dir/train.log."""
    log_path = os.path.join(save_dir, "train.log")
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
    )
    logging.getLogger().addHandler(fh)  # root logger
    logger.info(f"Logging to file: {log_path}")


# ==============================================================================
# Dataset layout
# ==============================================================================

LINE_FOLDERS = ["Control_C4", "Control_C18", "Control_C19", "SNCA", "GBA", "LRRK2"]



# ==============================================================================
# Reproducibility & speed
# ==============================================================================
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


# ==============================================================================
# Balanced subset selection (line & plate aware)
# ==============================================================================
def select_balanced_subset(
    refs: List[SampleRef], max_samples_total: int, seed: int
) -> List[int]:
    rng = random.Random(seed)

    by_super = defaultdict(list)
    for i, r in enumerate(refs):
        by_super[r.superclass].append(i)

    per_class = max_samples_total // 4
    targets = {
        "Control": per_class,
        "SNCA": per_class,
        "GBA": per_class,
        "LRRK2": per_class,
    }
    rem = max_samples_total - per_class * 4
    for k in ["Control", "SNCA", "GBA", "LRRK2"]:
        if rem <= 0:
            break
        targets[k] += 1
        rem -= 1

    selected = []

    def pick_line_plate_uniform(line_name: str, sup: str, target_n: int) -> List[int]:
        plate_to_idxs = defaultdict(list)
        for i in by_super[sup]:
            if refs[i].line == line_name:
                plate_to_idxs[refs[i].plate].append(i)
        plates = sorted(plate_to_idxs.keys())
        if len(plates) == 0:
            return []

        for p in plates:
            rng.shuffle(plate_to_idxs[p])

        target_n = min(target_n, sum(len(v) for v in plate_to_idxs.values()))
        per_plate = target_n // len(plates)
        remp = target_n - per_plate * len(plates)

        out = []
        for p in plates:
            out.extend(plate_to_idxs[p][:per_plate])

        ptr = {p: per_plate for p in plates}
        pi = 0
        while remp > 0:
            p = plates[pi % len(plates)]
            if ptr[p] < len(plate_to_idxs[p]):
                out.append(plate_to_idxs[p][ptr[p]])
                ptr[p] += 1
                remp -= 1
            pi += 1
            if pi > 10_000_000:
                break
        return out[:target_n]

    control_lines = ["Control_C4", "Control_C18", "Control_C19"]
    pcl = targets["Control"] // 3
    remc = targets["Control"] - pcl * 3
    for li, line in enumerate(control_lines):
        t = pcl + (1 if li < remc else 0)
        picked = pick_line_plate_uniform(line, "Control", t)
        selected.extend(picked)
        logger.info(f"[subset] Control line={line} picked={len(picked)} target={t}")

    for sup in ["SNCA", "GBA", "LRRK2"]:
        line = sup
        t = targets[sup]
        picked = pick_line_plate_uniform(line, sup, t)
        selected.extend(picked)
        logger.info(f"[subset] {sup} picked={len(picked)} target={t}")

    rng.shuffle(selected)
    return selected


# ==============================================================================
# Split CSV
# ==============================================================================
def save_split_csv(
    uids: List[str],
    labels: List[int],
    refs_by_uid: Dict[str, SampleRef],
    save_dir: str,
    filename: str,
):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["uid", "label", "superclass", "line", "plate", "tar_path", "prefix"]
        )
        for uid, lb in zip(uids, labels):
            r = refs_by_uid[uid]
            w.writerow(
                [uid, int(lb), r.superclass, r.line, r.plate, r.tar_path, r.prefix]
            )
    logger.info(f"Saved split CSV -> {path}")



# ==============================================================================
# MoCo momentum update
# ==============================================================================
@torch.no_grad()
def momentum_update_(model_q: nn.Module, model_k: nn.Module, m: float):
    for p_q, p_k in zip(model_q.parameters(), model_k.parameters()):
        p_k.data.mul_(m).add_(p_q.data, alpha=(1.0 - m))


# ==============================================================================
# Supervised MoCo Queue (FIFO)
# ==============================================================================
class SupervisedMoCoQueue:
    """
    FIFO queue storing (feats, labels)
    feats: fp16 recommended
    labels: int64
    """

    def __init__(
        self, dim: int, capacity: int, device: torch.device, dtype=torch.float16
    ):
        self.dim = int(dim)
        self.capacity = int(capacity)
        self.device = device
        self.dtype = dtype
        self.reset()

    def reset(self):
        self.ptr = 0
        self.full = False
        self.feats = torch.zeros(
            self.capacity, self.dim, device=self.device, dtype=self.dtype
        )
        self.labels = torch.zeros(self.capacity, device=self.device, dtype=torch.long)

    @torch.no_grad()
    def enqueue(self, feats: torch.Tensor, labels: torch.Tensor):
        feats = feats.detach()
        labels = labels.detach()
        b = int(feats.size(0))
        if b <= 0:
            return

        if b > self.capacity:
            feats = feats[-self.capacity :]
            labels = labels[-self.capacity :]
            b = self.capacity

        end = self.ptr + b
        if end <= self.capacity:
            self.feats[self.ptr : end].copy_(feats.to(self.dtype))
            self.labels[self.ptr : end].copy_(labels)
        else:
            first = self.capacity - self.ptr
            second = end - self.capacity
            self.feats[self.ptr :].copy_(feats[:first].to(self.dtype))
            self.labels[self.ptr :].copy_(labels[:first])
            self.feats[:second].copy_(feats[first:].to(self.dtype))
            self.labels[:second].copy_(labels[first:])

        self.ptr = end % self.capacity
        if end >= self.capacity:
            self.full = True

    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.full:
            return self.feats, self.labels
        else:
            return self.feats[: self.ptr], self.labels[: self.ptr]


# ==============================================================================
# SupCon loss (q vs k pool)
# ==============================================================================
def supervised_contrastive_q_vs_k(
    q: torch.Tensor,
    y_q: torch.Tensor,
    k: torch.Tensor,
    y_k: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    q = q.float()
    k = k.float()
    logits = (q @ k.t()) / float(temperature)
    logits = logits - logits.max(dim=1, keepdim=True).values

    with torch.no_grad():
        pos = y_q.view(-1, 1) == y_k.view(1, -1)
        pos_cnt = pos.sum(dim=1).clamp_min(1)

    exp = torch.exp(logits)
    denom = exp.sum(dim=1, keepdim=True).clamp_min(1e-12)
    log_prob = logits - torch.log(denom)

    loss_i = -(log_prob * pos).sum(dim=1) / pos_cnt
    return loss_i.mean()


# ==============================================================================
# Linear Probe + Early stopping
# ==============================================================================
@torch.no_grad()
def bake_unitnorm_encoder(model_q: SupMoCoModel, args, device):
    enc_sd = model_q.encoder.state_dict()
    baked = Encoder(
        blocks=parse_int_list(args.blocks, 4),
        dilations=parse_int_list(args.dilations, 4),
        refine_blocks=int(args.refine_blocks),
        ckpt_segments=int(args.ckpt_segments),
    )
    baked.load_state_dict(enc_sd, strict=True)
    baked.eval().to(device).to(memory_format=torch.channels_last)
    renorm_unit_per_out_channel_(baked)
    return baked


def linear_probe_val_acc(
    args, device, baked_encoder: nn.Module, train_lp_loader, val_lp_loader
) -> float:
    probe = nn.Linear(OUT_DIM, args.num_classes, bias=False).to(device)
    ce = nn.CrossEntropyLoss()
    opt = optim.SGD(
        probe.parameters(),
        lr=args.lp_lr,
        momentum=args.lp_momentum,
        weight_decay=args.lp_wd,
    )

    enc_bs = int(args.lp_enc_bs)
    use_bf16 = bool(args.use_bf16) and (device.type == "cuda")
    probe_dtype = next(probe.parameters()).dtype

    autocast_kwargs = dict(device_type="cuda", enabled=torch.cuda.is_available())
    if use_bf16:
        autocast_kwargs["dtype"] = torch.bfloat16

    probe.train()
    for ep in range(1, args.lp_epochs + 1):
        pbar = tqdm(
            train_lp_loader, desc=f"LP Train ep{ep}/{args.lp_epochs}", leave=False
        )
        for batch in pbar:
            if batch is None:
                continue
            x_cpu, y_cpu, plate, line, uid = batch
            if y_cpu.numel() < 1:
                continue

            for i in range(0, x_cpu.size(0), enc_bs):
                xb = (
                    x_cpu[i : i + enc_bs]
                    .to(device, non_blocking=True)
                    .contiguous(memory_format=torch.channels_last)
                )
                yb = y_cpu[i : i + enc_bs].to(device, non_blocking=True)

                with torch.no_grad():
                    with torch.amp.autocast(**autocast_kwargs):
                        feat = baked_encoder(xb)
                    feat = F.normalize(feat, dim=1)

                feat = feat.to(dtype=probe_dtype)

                opt.zero_grad(set_to_none=True)
                logits = probe(feat)
                loss = ce(logits, yb)
                loss.backward()
                opt.step()

    probe.eval()
    correct = 0
    total = 0
    pbar = tqdm(val_lp_loader, desc="LP Val", leave=False)
    with torch.inference_mode():
        for batch in pbar:
            if batch is None:
                continue
            x_cpu, y_cpu, plate, line, uid = batch
            if y_cpu.numel() < 1:
                continue

            for i in range(0, x_cpu.size(0), enc_bs):
                xb = (
                    x_cpu[i : i + enc_bs]
                    .to(device, non_blocking=True)
                    .contiguous(memory_format=torch.channels_last)
                )
                yb = y_cpu[i : i + enc_bs].to(device, non_blocking=True)

                with torch.amp.autocast(**autocast_kwargs):
                    feat = baked_encoder(xb)
                feat = F.normalize(feat, dim=1).to(dtype=probe_dtype)

                pred = probe(feat).argmax(dim=1)
                correct += int((pred == yb).sum().item())
                total += int(yb.numel())

    del probe, opt, ce
    torch.cuda.empty_cache()
    return 0.0 if total == 0 else (correct / total)


# ==============================================================================
# Warmup + Cosine scheduler helper
# ==============================================================================
def lr_at_step(
    global_step: int, base_lr: float, min_lr: float, warmup_steps: int, total_steps: int
) -> float:
    if global_step < warmup_steps:
        return base_lr * (global_step + 1) / warmup_steps
    t = (global_step - warmup_steps) / max(1, (total_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * t))
    return min_lr + (base_lr - min_lr) * cosine


# ==============================================================================
# Checkpoint (q/k/opt/scaler/queue)
# ==============================================================================
def _get_rng_state():
    st = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def _set_rng_state(st: dict):
    try:
        random.setstate(st["python"])
        np.random.set_state(st["numpy"])
        torch.set_rng_state(st["torch"])
        if torch.cuda.is_available() and ("cuda" in st):
            torch.cuda.set_rng_state_all(st["cuda"])
    except Exception as e:
        logger.info(f"[resume] RNG restore skipped: {e}")


def save_ckpt(
    path,
    epoch,
    global_step,
    model_q,
    model_k,
    opt,
    scaler,
    queue,
    best_acc,
    patience_counter,
    args,
):
    ckpt = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model_q": model_q.state_dict(),
        "model_k": model_k.state_dict(),
        "opt": opt.state_dict(),
        "scaler": (scaler.state_dict() if scaler is not None else None),
        "queue": {
            "feats": queue.feats.detach().cpu(),
            "labels": queue.labels.detach().cpu(),
            "ptr": int(queue.ptr),
            "full": bool(queue.full),
        },
        "best_acc": float(best_acc),
        "patience_counter": int(patience_counter),
        "args": vars(args),
        "rng": _get_rng_state(),
    }
    torch.save(ckpt, path)


def load_ckpt(path, model_q, model_k, opt, scaler, queue, device, strict=False):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    model_q.load_state_dict(ckpt["model_q"], strict=strict)
    model_k.load_state_dict(ckpt["model_k"], strict=strict)
    opt.load_state_dict(ckpt["opt"])

    if scaler is not None and ckpt.get("scaler", None) is not None:
        try:
            scaler.load_state_dict(ckpt["scaler"])
        except Exception as e:
            logger.info(f"[resume] scaler restore skipped: {e}")

    queue.reset()
    queue.feats.copy_(ckpt["queue"]["feats"].to(device=device, dtype=queue.dtype))
    queue.labels.copy_(ckpt["queue"]["labels"].to(device=device))
    queue.ptr = int(ckpt["queue"]["ptr"])
    queue.full = bool(ckpt["queue"]["full"])

    if "rng" in ckpt:
        _set_rng_state(ckpt["rng"])

    epoch = int(ckpt["epoch"])
    global_step = int(ckpt["global_step"])
    best_acc = float(ckpt.get("best_acc", -1.0))
    patience_counter = int(ckpt.get("patience_counter", 0))

    for state in opt.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

    return epoch, global_step, best_acc, patience_counter


# ==============================================================================
# Metrics CSV Logger (for publication figures)
# ==============================================================================
class MetricsCSVLogger:
    """Append-safe CSV logger. Writes header only if file is new/empty."""

    def __init__(self, path: str, columns: List[str]):
        self.path = path
        self.columns = columns
        write_header = (not os.path.exists(path)) or (os.path.getsize(path) == 0)
        self._f = open(path, "a", newline="", encoding="utf-8")
        self._w = csv.writer(self._f)
        if write_header:
            self._w.writerow(columns)
            self._f.flush()

    def log(self, row_dict: dict):
        self._w.writerow([row_dict.get(c, "") for c in self.columns])
        self._f.flush()  # flush every row — crash-safe

    def close(self):
        self._f.close()


# ==============================================================================
# Trainer (Supervised MoCo + large queue)
# ==============================================================================
class Trainer:
    def __init__(
        self,
        args,
        model_q,
        model_k,
        queue,
        train_loader,
        train_lp_loader,
        val_lp_loader,
    ):
        self.args = args
        self.model_q = model_q
        self.model_k = model_k
        self.queue = queue

        self.train_loader = train_loader
        self.train_lp_loader = train_lp_loader
        self.val_lp_loader = val_lp_loader

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_q.to(self.device).to(memory_format=torch.channels_last)
        self.model_k.to(self.device).to(memory_format=torch.channels_last)
        for p in self.model_k.parameters():
            p.requires_grad = False

        # SGD 추천 (renorm 매 step과 궁합)
        self.opt = optim.SGD(
            self.model_q.parameters(),
            lr=float(args.lr),
            momentum=float(args.sgd_momentum),
            weight_decay=float(args.wd),
            nesterov=bool(args.sgd_nesterov),
        )

        self.steps_per_epoch = max(1, len(self.train_loader))
        self.total_steps = max(1, args.epochs * self.steps_per_epoch)
        self.warmup_steps = max(1, args.warmup_epochs * self.steps_per_epoch)

        self.use_bf16 = bool(args.use_bf16) and (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=(torch.cuda.is_available() and not self.use_bf16)
        )

        os.makedirs(args.save_dir, exist_ok=True)
        self.best_path = os.path.join(args.save_dir, "best_model.pt")
        self.last_path = os.path.join(args.save_dir, "last_model.pt")
        self.resume_ckpt_path = os.path.join(args.save_dir, "resume_ckpt.pt")

        # ---- publication metrics CSV ----
        self.epoch_csv = MetricsCSVLogger(
            os.path.join(args.save_dir, "epoch_metrics.csv"),
            [
                "epoch",
                "train_loss",
                "lp_val_acc",
                "best_acc",
                "lr_end",
                "queue_fill",
                "patience",
                "grad_norm_avg",
                "epoch_time_sec",
                "total_elapsed_sec",
                "gpu_mem_peak_mb",
            ],
        )
        self.step_csv = MetricsCSVLogger(
            os.path.join(args.save_dir, "step_metrics.csv"),
            ["global_step", "epoch", "loss", "lr", "queue_fill", "grad_norm"],
        )
        self._train_start_time = time.time()

        renorm_unit_per_out_channel_(self.model_q)
        renorm_unit_per_out_channel_(self.model_k)

        self.start_epoch = 1
        self.global_step = 0
        self.best_acc = -1.0
        self.patience_counter = 0

        resume_path = ""
        if getattr(args, "auto_resume", False) and os.path.exists(
            self.resume_ckpt_path
        ):
            resume_path = self.resume_ckpt_path
        elif getattr(args, "resume_path", ""):
            resume_path = args.resume_path

        if resume_path and os.path.exists(resume_path):
            logger.info(f"[resume] Loading checkpoint: {resume_path}")
            ep, gs, best_acc, pat = load_ckpt(
                resume_path,
                self.model_q,
                self.model_k,
                self.opt,
                self.scaler,
                self.queue,
                self.device,
                strict=bool(getattr(args, "resume_strict", False)),
            )
            self.start_epoch = ep + 1
            self.global_step = gs
            self.best_acc = best_acc
            self.patience_counter = pat
            logger.info(
                f"[resume] start_epoch={self.start_epoch}, global_step={self.global_step}, "
                f"best_acc={self.best_acc*100:.2f}%, patience={self.patience_counter}"
            )

    def train_epoch(self, epoch: int):
        self.model_q.train()
        self.model_k.eval()  # EMA teacher는 eval 유지

        total_loss = 0.0
        total_grad_norm = 0.0
        steps = 0
        epoch_t0 = time.time()

        autocast_kwargs = dict(device_type="cuda", enabled=torch.cuda.is_available())
        if self.use_bf16:
            autocast_kwargs["dtype"] = torch.bfloat16

        pbar = tqdm(
            self.train_loader, desc=f"Train E{epoch}/{self.args.epochs}", leave=True
        )
        for batch in pbar:
            if batch is None:
                continue

            v1, v2, y, plate, line, uid = batch
            if y.numel() < 2:
                continue

            v1 = v1.to(self.device, non_blocking=True).contiguous(
                memory_format=torch.channels_last
            )
            v2 = v2.to(self.device, non_blocking=True).contiguous(
                memory_format=torch.channels_last
            )
            y = y.to(self.device, non_blocking=True)

            lr = lr_at_step(
                self.global_step,
                self.args.lr,
                self.args.min_lr,
                self.warmup_steps,
                self.total_steps,
            )
            for pg in self.opt.param_groups:
                pg["lr"] = lr

            self.opt.zero_grad(set_to_none=True)

            with torch.amp.autocast(**autocast_kwargs):
                q = self.model_q(v1)  # (B, D) grad on
                with torch.no_grad():
                    k = self.model_k(v2)  # (B, D) no grad

                q_queue, y_queue = self.queue.get()  # (Q, D), (Q,)
                # K pool = current batch key + queue
                if q_queue.numel() > 0:
                    K = torch.cat([k, q_queue], dim=0)
                    YK = torch.cat([y, y_queue], dim=0)
                else:
                    K = k
                    YK = y

                loss = supervised_contrastive_q_vs_k(
                    q, y, K, YK, temperature=float(self.args.temp)
                )

                if self.args.symmetric_loss:
                    q2 = self.model_q(v2)
                    with torch.no_grad():
                        k2 = self.model_k(v1)

                    if q_queue.numel() > 0:
                        K2 = torch.cat([k2, q_queue], dim=0)
                        YK2 = torch.cat([y, y_queue], dim=0)
                    else:
                        K2 = k2
                        YK2 = y

                    loss2 = supervised_contrastive_q_vs_k(
                        q2, y, K2, YK2, temperature=float(self.args.temp)
                    )
                    loss = 0.5 * (loss + loss2)

            # Step
            if self.use_bf16:
                loss.backward()
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        self.model_q.parameters(),
                        float(self.args.grad_clip) if self.args.grad_clip > 0 else 1e9,
                    )
                )
                self.opt.step()
                did_step = True
            else:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt)
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        self.model_q.parameters(),
                        float(self.args.grad_clip) if self.args.grad_clip > 0 else 1e9,
                    )
                )
                self.scaler.step(self.opt)
                self.scaler.update()
                did_step = True

            # (중요) queue enqueue는 key(k)로, step/EMA 업데이트와 독립
            with torch.no_grad():
                self.queue.enqueue(k, y)

            # renorm (너는 매 step)
            if did_step and (self.global_step % int(self.args.renorm_every) == 0):
                renorm_unit_per_out_channel_(self.model_q)

            # EMA update: model_q가 업데이트된 다음에 model_k를 따라가게
            with torch.no_grad():
                momentum_update_(self.model_q, self.model_k, float(self.args.moco_m))
                if self.args.renorm_k_every > 0 and (
                    self.global_step % int(self.args.renorm_k_every) == 0
                ):
                    renorm_unit_per_out_channel_(self.model_k)

            cur = float(loss.item())
            total_loss += cur
            total_grad_norm += grad_norm
            steps += 1
            self.global_step += 1

            qfill = int(self.queue.capacity if self.queue.full else self.queue.ptr)
            pbar.set_postfix(
                {
                    "lr": f"{lr:.2e}",
                    "loss": f"{cur:.4f}",
                    "queue": f"{qfill}/{self.queue.capacity}",
                }
            )

            # per-step CSV
            self.step_csv.log(
                {
                    "global_step": self.global_step,
                    "epoch": epoch,
                    "loss": f"{cur:.6f}",
                    "lr": f"{lr:.8f}",
                    "queue_fill": qfill,
                    "grad_norm": f"{grad_norm:.6f}",
                }
            )

        avg_loss = 0.0 if steps == 0 else total_loss / steps
        avg_grad = 0.0 if steps == 0 else total_grad_norm / steps
        epoch_sec = time.time() - epoch_t0
        return avg_loss, avg_grad, epoch_sec

    def run(self):
        logger.info(
            f"Device: {self.device}, bf16={self.use_bf16}, TF32={torch.backends.cuda.matmul.allow_tf32}"
        )

        for epoch in range(self.start_epoch, self.args.epochs + 1):
            tqdm.write(f"\n===== Epoch {epoch}/{self.args.epochs} =====")

            train_loss, avg_grad, epoch_sec = self.train_epoch(epoch)

            baked_encoder = bake_unitnorm_encoder(self.model_q, self.args, self.device)
            val_acc = linear_probe_val_acc(
                self.args,
                self.device,
                baked_encoder,
                self.train_lp_loader,
                self.val_lp_loader,
            )
            del baked_encoder
            torch.cuda.empty_cache()

            # GPU peak memory
            gpu_peak_mb = 0.0
            if torch.cuda.is_available():
                gpu_peak_mb = torch.cuda.max_memory_allocated() / (1024**2)

            lr_end = self.opt.param_groups[0]["lr"]
            qfill = int(self.queue.capacity if self.queue.full else self.queue.ptr)
            total_elapsed = time.time() - self._train_start_time

            tqdm.write(
                f"Epoch {epoch:03d} | TrainLoss: {train_loss:.4f} | LP ValAcc: {val_acc*100:.2f}% | "
                f"GradNorm: {avg_grad:.4f} | Time: {epoch_sec:.0f}s"
            )

            torch.save(self.model_q.state_dict(), self.last_path)

            save_ckpt(
                self.resume_ckpt_path,
                epoch=epoch,
                global_step=self.global_step,
                model_q=self.model_q,
                model_k=self.model_k,
                opt=self.opt,
                scaler=self.scaler,
                queue=self.queue,
                best_acc=self.best_acc,
                patience_counter=self.patience_counter,
                args=self.args,
            )

            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.patience_counter = 0
                torch.save(self.model_q.state_dict(), self.best_path)
                tqdm.write(
                    f"  -> Saved Best (LP ValAcc={val_acc*100:.2f}%) to {self.best_path}"
                )
            else:
                self.patience_counter += 1
                tqdm.write(
                    f"  -> No improve (best={self.best_acc*100:.2f}%) [{self.patience_counter}/{self.args.patience}]"
                )
                if self.patience_counter >= self.args.patience:
                    tqdm.write("Early Stopping Triggered")
                    break

            # per-epoch CSV (논문 figure 용)
            self.epoch_csv.log(
                {
                    "epoch": epoch,
                    "train_loss": f"{train_loss:.6f}",
                    "lp_val_acc": f"{val_acc:.6f}",
                    "best_acc": f"{self.best_acc:.6f}",
                    "lr_end": f"{lr_end:.8f}",
                    "queue_fill": qfill,
                    "patience": self.patience_counter,
                    "grad_norm_avg": f"{avg_grad:.6f}",
                    "epoch_time_sec": f"{epoch_sec:.1f}",
                    "total_elapsed_sec": f"{total_elapsed:.1f}",
                    "gpu_mem_peak_mb": f"{gpu_peak_mb:.1f}",
                }
            )


# ==============================================================================
# Args
# ==============================================================================
def get_args():
    p = argparse.ArgumentParser(
        "Supervised MoCo (large queue) + Plate-balanced + LP ES"
    )

    # Experiment
    p.add_argument("--seed", type=int, default=45)
    p.add_argument(
        "--save_dir",
        type=str,
        default="/home/ubuntu/model-east3/outputs/Model_MoCoXBM_no_cov_loss_no_moco_seed45",
    )
    p.add_argument("--shard_root", type=str, default=DEFAULT_SHARD_ROOT)

    # Data
    p.add_argument("--max_samples", type=int, default=108000)
    p.add_argument("--test_ratio", type=float, default=1 / 3)
    p.add_argument("--val_ratio", type=float, default=0.25)
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=512)

    # Model
    p.add_argument("--embed_dim", type=int, default=512)
    p.add_argument("--blocks", type=str, default="2,2,2,3")
    p.add_argument("--dilations", type=str, default="1,1,1,1")
    p.add_argument("--refine_blocks", type=int, default=1)
    p.add_argument("--ckpt_segments", type=int, default=0)

    # Projector
    p.add_argument("--no_l2_norm_pool", action="store_true", help="Disable L2 normalization on pooled features before projector")
    p.add_argument("--proj_layers", type=int, default=2, choices=[1, 2, 3])
    p.add_argument("--proj_hidden", type=int, default=2048)
    p.add_argument("--proj_bn", action="store_true")
    p.add_argument("--proj_dropout", type=float, default=0.0)

    # Train (SGD)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.1)  # SGD 권장 시작
    p.add_argument("--wd", type=float, default=1e-4)  # SGD L2: 보통 작게
    p.add_argument("--sgd_momentum", type=float, default=0.9)
    p.add_argument("--sgd_nesterov", action="store_true")

    p.add_argument("--temp", type=float, default=0.07)
    p.add_argument("--warmup_epochs", type=int, default=4)
    p.add_argument("--min_lr", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--use_bf16", action="store_true")
    p.add_argument("--symmetric_loss", action="store_true")

    # Renorm
    p.add_argument(
        "--renorm_every", type=int, default=1
    )  # model_q renorm frequency (steps)
    p.add_argument(
        "--renorm_k_every", type=int, default=0
    )  # model_k renorm frequency (0=off)

    # MoCo (EMA + queue)
    p.add_argument("--moco_m", type=float, default=0.995)
    p.add_argument("--queue_size", type=int, default=65536)  # 64k 추천 시작
    p.add_argument("--queue_dtype_fp16", action="store_true")  # on -> fp16 queue

    # Linear Probe
    p.add_argument("--num_classes", type=int, default=4)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--lp_epochs", type=int, default=3)
    p.add_argument("--lp_batch_size", type=int, default=16384)
    p.add_argument("--lp_lr", type=float, default=0.1)
    p.add_argument("--lp_wd", type=float, default=0.0)
    p.add_argument("--lp_momentum", type=float, default=0.9)
    p.add_argument("--lp_enc_bs", type=int, default=128)

    # Resume
    p.add_argument("--resume_path", type=str, default="")
    p.add_argument("--auto_resume", action="store_true")
    p.add_argument("--resume_strict", action="store_true")

    if "ipykernel" in sys.modules:
        return p.parse_args([])
    return p.parse_args()


# ==============================================================================
# Main
# ==============================================================================
def main():
    args = get_args()
    args.use_bf16 = True
    args.symmetric_loss = False
    args.auto_resume = True

    os.makedirs(args.save_dir, exist_ok=True)
    _add_file_logger(args.save_dir)  # persist all logs to save_dir/train.log
    with open(os.path.join(args.save_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    set_seed(args.seed)

    refs = load_all_sample_refs(args.shard_root)

    train_csv = os.path.join(args.save_dir, "train_split.csv")
    val_csv = os.path.join(args.save_dir, "val_split.csv")
    test_csv = os.path.join(args.save_dir, "test_split.csv")

    resume_ckpt_path = os.path.join(args.save_dir, "resume_ckpt.pt")
    will_resume = (args.auto_resume and os.path.exists(resume_ckpt_path)) or (
        args.resume_path != ""
    )

    if (
        will_resume
        and os.path.exists(train_csv)
        and os.path.exists(val_csv)
        and os.path.exists(test_csv)
    ):
        logger.info("[resume] Using existing split CSVs (no re-splitting).")

        X_train = load_split_csv(train_csv)
        X_val = load_split_csv(val_csv)

        uid_to_refidx_all = {f"{r.tar_path}:{r.prefix}": i for i, r in enumerate(refs)}
        missing = [u for u in (X_train + X_val) if u not in uid_to_refidx_all]
        if len(missing) > 0:
            raise RuntimeError(
                f"[resume] Some uids missing in current shard_root. ex: {missing[:5]}"
            )

        train_refidx = [uid_to_refidx_all[u] for u in X_train]
        val_refidx = [uid_to_refidx_all[u] for u in X_val]

    else:
        subset_refidx = select_balanced_subset(refs, args.max_samples, args.seed)
        logger.info(f"Subset size: {len(subset_refidx)}")

        uids = [f"{refs[i].tar_path}:{refs[i].prefix}" for i in subset_refidx]
        labels = [refs[i].label for i in subset_refidx]
        lines = [refs[i].line for i in subset_refidx]

        line_to_id = {ln: i for i, ln in enumerate(sorted(set(lines)))}
        strat = [line_to_id[ln] for ln in lines]

        X_temp, X_test, y_temp, y_test, strat_temp, strat_test = train_test_split(
            uids,
            labels,
            strat,
            test_size=args.test_ratio,
            random_state=args.seed,
            stratify=strat,
        )
        X_train, X_val, y_train, y_val, strat_train, strat_val = train_test_split(
            X_temp,
            y_temp,
            strat_temp,
            test_size=args.val_ratio,
            random_state=args.seed,
            stratify=strat_temp,
        )
        logger.info(
            f"Split -> Train {len(X_train)}, Val {len(X_val)}, Test {len(X_test)}"
        )

        refs_by_uid = {
            f"{refs[i].tar_path}:{refs[i].prefix}": refs[i] for i in subset_refidx
        }
        save_split_csv(X_train, y_train, refs_by_uid, args.save_dir, "train_split.csv")
        save_split_csv(X_val, y_val, refs_by_uid, args.save_dir, "val_split.csv")
        save_split_csv(X_test, y_test, refs_by_uid, args.save_dir, "test_split.csv")

        uid_to_refidx = {
            f"{refs[i].tar_path}:{refs[i].prefix}": i for i in subset_refidx
        }
        train_refidx = [uid_to_refidx[u] for u in X_train]
        val_refidx = [uid_to_refidx[u] for u in X_val]

    train_bank = InMemoryTarBank(refs, train_refidx, args.img_size)
    val_bank = InMemoryTarBank(refs, val_refidx, args.img_size)

    train_ib = list(range(len(train_refidx)))
    val_ib = list(range(len(val_refidx)))

    train_ds = InMemorySixteenBitDataset(
        train_bank, train_ib, args.img_size, two_crops=True, augment=True
    )

    train_lp_ds = InMemorySixteenBitDataset(
        train_bank, train_ib, args.img_size, two_crops=False, augment=False
    )
    val_lp_ds = InMemorySixteenBitDataset(
        val_bank, val_ib, args.img_size, two_crops=False, augment=False
    )

    train_sampler = StrictPlateBalancedBatchSamplerOnBank(
        train_bank, batch_size=args.batch_size, seed=args.seed
    )

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        num_workers=0,
        pin_memory=pin,
        worker_init_fn=seed_worker,
        collate_fn=collate_skip_none,
    )

    train_lp_loader = DataLoader(
        train_lp_ds,
        batch_size=args.lp_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin,
        worker_init_fn=seed_worker,
        collate_fn=collate_skip_none,
        drop_last=False,
    )
    val_lp_loader = DataLoader(
        val_lp_ds,
        batch_size=args.lp_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin,
        worker_init_fn=seed_worker,
        collate_fn=collate_skip_none,
        drop_last=False,
    )

    blocks = parse_int_list(args.blocks, 4)
    dilations = parse_int_list(args.dilations, 4)

    model_q = SupMoCoModel(
        embed_dim=args.embed_dim,
        blocks=blocks,
        dilations=dilations,
        refine_blocks=args.refine_blocks,
        ckpt_segments=args.ckpt_segments,
        proj_layers=args.proj_layers,
        proj_hidden=args.proj_hidden,
        proj_bn=args.proj_bn,
        proj_dropout=args.proj_dropout,
        use_l2_norm_pool=not args.no_l2_norm_pool,
    )
    model_k = SupMoCoModel(
        embed_dim=args.embed_dim,
        blocks=blocks,
        dilations=dilations,
        refine_blocks=args.refine_blocks,
        ckpt_segments=args.ckpt_segments,
        proj_layers=args.proj_layers,
        proj_hidden=args.proj_hidden,
        proj_bn=args.proj_bn,
        proj_dropout=args.proj_dropout,
        use_l2_norm_pool=not args.no_l2_norm_pool,
    )
    model_k.load_state_dict(model_q.state_dict(), strict=True)

    try:
        model_q = torch.compile(model_q, mode="max-autotune")
        logger.info("[compile] torch.compile enabled (q)")
    except Exception as e:
        logger.info(f"[compile] torch.compile not available: {e}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qdtype = torch.float16 if args.queue_dtype_fp16 else torch.float32
    queue = SupervisedMoCoQueue(
        dim=args.embed_dim, capacity=args.queue_size, device=device, dtype=qdtype
    )

    trainer = Trainer(
        args, model_q, model_k, queue, train_loader, train_lp_loader, val_lp_loader
    )
    trainer.run()


if __name__ == "__main__":
    main()
