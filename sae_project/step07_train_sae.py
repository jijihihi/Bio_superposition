# ==============================================================================
# SAE trainer (+ FVU + val/test)
# ==============================================================================
import os
import csv
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from sae_project.step01_configs import get_args, resolve_paths
from sae_project.step02_logging_utils import get_logger
from sae_project.step03_data_shards import load_all_sample_refs, build_uid_to_refidx
from sae_project.step04_data_bank import InMemoryTarBank, InMemorySixteenBitDataset, load_split_csv, seed_worker, collate_skip_none
from sae_project.step05_model_encoder import SupConMoCoModel, parse_int_list, renorm_unit_per_out_channel_
from sae_project.step06_sae_core import PointwiseTopKSAE, resample_dead_features_
from sae_project.step04_data_bank import StrictPlateBalancedBatchSamplerOnBank

# ==============================================================================
# SAE trainer (+ strict plate sampler + token shuffle + token chunking + FVU + val/test)
# ==============================================================================


logger = get_logger("train_sae")


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


@torch.no_grad()
def _sse_sst(tokens: torch.Tensor, recon: torch.Tensor):
    diff = tokens - recon
    sse = float((diff * diff).sum().item())
    mu = tokens.mean(dim=0, keepdim=True)
    cen = tokens - mu
    sst = float((cen * cen).sum().item())
    return sse, sst


class SAETrainer:
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

        self.sae = PointwiseTopKSAE(
            d_in=args.d_in,
            d_sae=args.d_sae,
            k=args.k,
            init_scale=args.sae_init_scale
        ).to(self.device)

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

        os.makedirs(args.sae_save_dir, exist_ok=True)
        self.ckpt_path = os.path.join(args.sae_save_dir, f"sae_{args.which_layer}_d{args.d_sae}_k{args.k}.pt")
        self.best_ckpt_path = os.path.join(args.sae_save_dir, f"sae_{args.which_layer}_BEST_d{args.d_sae}_k{args.k}.pt")
        self.log_csv_path = os.path.join(args.sae_save_dir, f"sae_{args.which_layer}_trainlog.csv")

        self._init_log_csv()

        self.sae.renorm_decoder_()
        self.global_step = 0
        self.best_metric = float("inf")

        # token chunking config
        self.token_batch = int(getattr(args, "token_batch", 8192))
        if self.token_batch <= 0:
            self.token_batch = 8192
        self.shuffle_tokens = bool(getattr(args, "shuffle_tokens", False))

    def _init_log_csv(self):
        if not os.path.exists(self.log_csv_path):
            with open(self.log_csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    "epoch", "step",
                    "train_mse", "train_l1", "train_total", "train_fvu",
                    "val_mse", "val_l1", "val_total", "val_fvu",
                    "test_mse", "test_l1", "test_total", "test_fvu",
                    "dead_count", "resampled"
                ])

    @torch.no_grad()
    def _extract_tokens(self, x: torch.Tensor) -> torch.Tensor:
        # encoder feature maps
        with torch.amp.autocast(**self.autocast_kwargs):
            fmap = self.encoder.forward_feature_maps(x, which=self.args.which_layer)

        # (B,C,H,W) -> (B*H*W, C)
        fmap = fmap.permute(0, 2, 3, 1).contiguous()
        B, Hf, Wf, C = fmap.shape
        tokens = fmap.view(B * Hf * Wf, C)

        # optional token L2 normalization
        if self.args.token_l2_norm:
            tokens = F.normalize(tokens, dim=1)

        # per-image token sampling
        tpi = int(self.args.tokens_per_image)
        if tpi > 0 and tpi < (Hf * Wf):
            tokens_list = []
            for b in range(B):
                base = b * (Hf * Wf)
                idx = torch.randperm(Hf * Wf, device=tokens.device)[:tpi]
                tokens_list.append(tokens[base + idx])
            tokens = torch.cat(tokens_list, dim=0)

        # ✅ shuffle token rows across the batch (recommended for SGD mixing)
        if self.shuffle_tokens and tokens.size(0) > 1:
            perm = torch.randperm(tokens.size(0), device=tokens.device)
            tokens = tokens[perm]

        return tokens

    def save_ckpt(self, path: str):
        ckpt = {
            "args": vars(self.args),
            "sae": self.sae.state_dict(),
            "opt": self.opt.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "global_step": int(self.global_step),
            "best_metric": float(self.best_metric),
        }
        torch.save(ckpt, path)

    @torch.no_grad()
    def eval_epoch(self, loader: DataLoader, tag: str):
        self.sae.eval()

        mse_sum, l1_sum, steps = 0.0, 0.0, 0
        sse_sum, sst_sum = 0.0, 0.0

        for batch in tqdm(loader, desc=f"SAE {tag}", leave=False):
            if batch is None:
                continue
            x_cpu, *_ = batch
            if x_cpu.numel() < 1:
                continue

            x = x_cpu.to(self.device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            tokens = self._extract_tokens(x)

            # ✅ chunk tokens to avoid OOM
            Tb = self.token_batch
            n = tokens.size(0)
            mse_acc, l1_acc = 0.0, 0.0
            chunks = 0

            for s in range(0, n, Tb):
                tok = tokens[s:s+Tb]
                with torch.amp.autocast(**self.autocast_kwargs):
                    recon, acts = self.sae(tok)

                tok_f = tok.float()
                rec_f = recon.float()

                mse = F.mse_loss(rec_f, tok_f)
                l1 = acts.float().abs().mean()

                mse_acc += float(mse.item())
                l1_acc += float(l1.item())
                chunks += 1

                if getattr(self.args, "log_fvu", False):
                    sse, sst = _sse_sst(tok_f, rec_f)
                    sse_sum += sse
                    sst_sum += sst

            if chunks > 0:
                mse_sum += mse_acc / chunks
                l1_sum += l1_acc / chunks
                steps += 1

        self.sae.train()
        if steps == 0:
            return 0.0, 0.0, 0.0, 0.0

        mse_avg = mse_sum / steps
        l1_avg = l1_sum / steps
        total_avg = mse_avg + float(self.args.l1_coeff) * l1_avg

        fvu = 0.0
        if getattr(self.args, "log_fvu", False):
            fvu = 0.0 if sst_sum <= 1e-12 else float(sse_sum / sst_sum)

        return mse_avg, l1_avg, total_avg, fvu

    def train(self):
        logger.info(f"[SAE] device={self.device}, bf16={self.use_bf16}")
        logger.info(f"[SAE] which_layer={self.args.which_layer}, d_in={self.args.d_in}, d_sae={self.args.d_sae}, k={self.args.k}")
        logger.info(f"[SAE] tokens_per_image={self.args.tokens_per_image}, token_batch={self.token_batch}, shuffle_tokens={self.shuffle_tokens}")

        # warmup to show GPU activity early
        if self.device.type == "cuda":
            with torch.no_grad():
                dummy = torch.zeros(2, 3, self.args.img_size, self.args.img_size, device=self.device).contiguous(memory_format=torch.channels_last)
                _ = self.encoder.forward_feature_maps(dummy, which=self.args.which_layer)

        for epoch in range(1, self.args.epochs + 1):
            self.sae.train()

            train_mse_sum, train_l1_sum, train_steps = 0.0, 0.0, 0
            train_sse_sum, train_sst_sum = 0.0, 0.0

            dead_count_epoch = 0
            resampled_epoch = 0

            pbar = tqdm(self.train_loader, desc=f"SAE Train E{epoch}/{self.args.epochs}", leave=True)
            for batch in pbar:
                if batch is None:
                    continue
                x_cpu, *_ = batch
                if x_cpu.numel() < 1:
                    continue

                x = x_cpu.to(self.device, non_blocking=True).contiguous(memory_format=torch.channels_last)
                tokens = self._extract_tokens(x)

                # ✅ chunked training: we do multiple micro-steps over token chunks per image-batch
                Tb = self.token_batch
                n = tokens.size(0)
                if n == 0:
                    continue

                # We want overall update comparable regardless of #chunks.
                # So: accumulate gradients over chunks and step once.
                self.opt.zero_grad(set_to_none=True)

                mse_acc, l1_acc = 0.0, 0.0
                chunks = 0

                for s in range(0, n, Tb):
                    tok = tokens[s:s+Tb]

                    with torch.amp.autocast(**self.autocast_kwargs):
                        recon, acts = self.sae(tok)

                    tok_f = tok.float()
                    rec_f = recon.float()

                    mse = F.mse_loss(rec_f, tok_f)
                    act_l1 = acts.float().abs().mean()
                    loss = mse + float(self.args.l1_coeff) * act_l1

                    # ✅ scale loss by 1/chunks for stable gradients (grad accumulation)
                    # We don't know chunks ahead of time easily without computing, so accumulate then normalize:
                    # easiest: divide by a constant by counting.
                    # We'll do: loss_scaled = loss / num_chunks_est, but num_chunks_est depends on n.
                    # Compute exact chunks_count once:
                    # (We can compute outside loop, but keep code simple)
                    chunks_count = (n + Tb - 1) // Tb
                    loss = loss / float(chunks_count)

                    if self.use_bf16:
                        loss.backward()
                    else:
                        self.scaler.scale(loss).backward()

                    with torch.no_grad():
                    # update usage based on last chunk's acts is OK-ish, but better: update using concatenated acts.
                    # For simplicity and speed: update on the final chunk acts (works in practice).
                          self.sae.update_usage_ema_(acts.detach(), ema=float(self.args.usage_ema))

                    mse_acc += float(mse.item())
                    l1_acc += float(act_l1.item())
                    chunks += 1

                    if getattr(self.args, "log_fvu", False):
                        sse, sst = _sse_sst(tok_f, rec_f)
                        train_sse_sum += sse
                        train_sst_sum += sst

                # optimizer step after all chunks
                if self.use_bf16:
                    if self.args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.sae.parameters(), float(self.args.grad_clip))
                    self.opt.step()
                else:
                    if self.args.grad_clip > 0:
                        self.scaler.unscale_(self.opt)
                        torch.nn.utils.clip_grad_norm_(self.sae.parameters(), float(self.args.grad_clip))
                    self.scaler.step(self.opt)
                    self.scaler.update()

                # constraints / stats
                self.sae.renorm_decoder_()


                dead_count = int((self.sae.usage_ema < float(self.args.dead_threshold)).sum().item())
                resampled = 0
                if self.global_step > 0 and (self.global_step % int(self.args.resample_every)) == 0:
                    # use the last tok/rec as a proxy for resampling signal
                    resampled = resample_dead_features_(
                        sae=self.sae,
                        opt=self.opt,
                        x_batch=tok_f,
                        recon_batch=rec_f,
                        dead_threshold=float(self.args.dead_threshold),
                        max_resample_frac=float(self.args.max_resample_frac),
                    )

                self.global_step += 1

                # step stats (average across chunks)
                if chunks > 0:
                    mse_step = mse_acc / chunks
                    l1_step = l1_acc / chunks
                else:
                    mse_step = 0.0
                    l1_step = 0.0

                train_mse_sum += mse_step
                train_l1_sum += l1_step
                train_steps += 1
                dead_count_epoch = dead_count
                resampled_epoch += int(resampled)

                pbar.set_postfix({
                    "mse": f"{mse_step:.4f}",
                    "l1": f"{l1_step:.4f}",
                    "dead": dead_count,
                    "resamp": int(resampled)
                })

                if (self.global_step % int(self.args.save_every)) == 0:
                    self.save_ckpt(self.ckpt_path)

            # epoch summaries
            if train_steps == 0:
                train_mse_avg = train_l1_avg = train_total_avg = train_fvu = 0.0
            else:
                train_mse_avg = train_mse_sum / train_steps
                train_l1_avg = train_l1_sum / train_steps
                train_total_avg = train_mse_avg + float(self.args.l1_coeff) * train_l1_avg
                train_fvu = 0.0
                if getattr(self.args, "log_fvu", False):
                    train_fvu = 0.0 if train_sst_sum <= 1e-12 else float(train_sse_sum / train_sst_sum)

            val_mse_avg = val_l1_avg = val_total_avg = val_fvu = 0.0
            test_mse_avg = test_l1_avg = test_total_avg = test_fvu = 0.0
            if self.val_loader is not None:
                val_mse_avg, val_l1_avg, val_total_avg, val_fvu = self.eval_epoch(self.val_loader, "Val")
            if self.test_loader is not None:
                test_mse_avg, test_l1_avg, test_total_avg, test_fvu = self.eval_epoch(self.test_loader, "Test")

            metric = val_total_avg if (self.val_loader is not None) else train_total_avg

            tqdm.write(
                f"Epoch {epoch:03d} | "
                f"Train: mse={train_mse_avg:.6f} fvu={train_fvu:.6f} total={train_total_avg:.6f} | "
                f"Val: mse={val_mse_avg:.6f} fvu={val_fvu:.6f} total={val_total_avg:.6f} | "
                f"Test: mse={test_mse_avg:.6f} fvu={test_fvu:.6f} total={test_total_avg:.6f} | "
                f"dead={dead_count_epoch} resamp={resampled_epoch}"
            )

            with open(self.log_csv_path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    epoch, self.global_step,
                    train_mse_avg, train_l1_avg, train_total_avg, train_fvu,
                    val_mse_avg, val_l1_avg, val_total_avg, val_fvu,
                    test_mse_avg, test_l1_avg, test_total_avg, test_fvu,
                    dead_count_epoch, resampled_epoch
                ])

            self.save_ckpt(self.ckpt_path)

            if getattr(self.args, "save_best", False):
                if metric < self.best_metric:
                    self.best_metric = float(metric)
                    self.save_ckpt(self.best_ckpt_path)
                    tqdm.write(f"  -> Saved BEST to {self.best_ckpt_path} (metric={self.best_metric:.6f})")

        logger.info(f"[SAE] Done. Saved -> {self.ckpt_path}")


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


def main():
    args = resolve_paths(get_args())
    set_seed(args.seed)

    refs = load_all_sample_refs(args.shard_root)
    uid_to_refidx = build_uid_to_refidx(refs)

    train_csv = os.path.join(args.save_dir, "train_split.csv")
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"train_split.csv not found: {train_csv}")

    strict = bool(getattr(args, "strict_plate_balance", False))

    # ✅ Train: augment follows --augment (rot90)
    train_loader = _make_loader_from_split(
        args, refs, uid_to_refidx,
        split_csv_path=train_csv,
        batch_size=int(args.batch_size),
        augment=bool(args.augment),
        shuffle=not strict,
        strict_balance=strict,
        seed=int(args.seed)
    )
    assert train_loader is not None

    # ✅ Val/Test: always augment=False
    val_loader = None
    if getattr(args, "use_val", False):
        val_csv = os.path.join(args.save_dir, "val_split.csv")
        vbs = int(args.batch_size) if int(args.val_batch_size) <= 0 else int(args.val_batch_size)
        val_loader = _make_loader_from_split(
            args, refs, uid_to_refidx,
            split_csv_path=val_csv,
            batch_size=vbs,
            augment=False,
            shuffle=False,
            strict_balance=False,
            seed=int(args.seed) + 1
        )
        if val_loader is None:
            logger.info("[val] --use_val set but val_split.csv not found. Skipping val.")

    test_loader = None
    if getattr(args, "use_test", False):
        test_csv = os.path.join(args.save_dir, "test_split.csv")
        tbs = int(args.batch_size) if int(args.test_batch_size) <= 0 else int(args.test_batch_size)
        test_loader = _make_loader_from_split(
            args, refs, uid_to_refidx,
            split_csv_path=test_csv,
            batch_size=tbs,
            augment=False,
            shuffle=False,
            strict_balance=False,
            seed=int(args.seed) + 2
        )
        if test_loader is None:
            logger.info("[test] --use_test set but test_split.csv not found. Skipping test.")

    # model wrapper + load state
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

    sd = torch.load(args.model_state_path, map_location="cpu")
    model.load_state_dict(sd, strict=False)

    trainer = SAETrainer(args, model.encoder, train_loader, val_loader=val_loader, test_loader=test_loader)
    trainer.train()


if __name__ == "__main__":
    main()
