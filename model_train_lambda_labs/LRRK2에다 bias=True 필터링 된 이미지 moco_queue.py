# ==============================================================================
# SupCon + MoCo with Queue (Supervised)
# - q from model_q (train), k from model_k (EMA teacher)
# - Queue stores (key, label) pairs for large negative pool
# - Same class = positive, different class = negative
# ==============================================================================

import os, re, io, json, math, time, glob, random, argparse, logging, sys, pickle, csv
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
from tqdm.auto import tqdm

try:
    import tifffile
except Exception:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", "tifffile"])
    import tifffile

from sklearn.model_selection import train_test_split
from torch.utils.checkpoint import checkpoint_sequential

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("MoCo_Queue")

# ==============================================================================
# Dataset layout
# ==============================================================================
DEFAULT_SHARD_ROOT = "/content/wds_shards"

LINE_FOLDERS = ["Control_C4", "Control_C18", "Control_C19", "SNCA", "GBA", "LRRK2"]
SUPERCLASS_MAP = {"Control_C4": "Control", "Control_C18": "Control", "Control_C19": "Control",
                  "SNCA": "SNCA", "GBA": "GBA", "LRRK2": "LRRK2"}
CLASS_TO_LABEL = {"Control": 0, "SNCA": 1, "GBA": 2, "LRRK2": 3}
PLATE_DIR_RE = re.compile(r"plate=(\d{6})")

# ==============================================================================
# Reproducibility
# ==============================================================================
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False; torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    try: torch.set_float32_matmul_precision("high")
    except: pass

def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed); random.seed(worker_seed)

def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    return default_collate(batch) if batch else None

# ==============================================================================
# Tar index
# ==============================================================================
@dataclass(frozen=True)
class SampleRef:
    tar_path: str; prefix: str; tif_off: int; tif_size: int; js_off: int; js_size: int
    line: str; superclass: str; label: int; plate: str

def _infer_line_and_plate_from_tarpath(tar_path: str) -> Tuple[str, str]:
    parts = tar_path.replace("\\", "/").split("/")
    line = parts[-3]
    m = PLATE_DIR_RE.search(parts[-2])
    return line, (m.group(1) if m else "UNKNOWN")

def build_tar_index_if_needed(tar_path: str):
    idx_path = tar_path + ".pkl"
    if os.path.exists(idx_path): return
    import tarfile
    items = {}
    with tarfile.open(tar_path, "r") as tf:
        for m in tf.getmembers():
            if not m.isreg(): continue
            if m.name.endswith(".tif"):
                items.setdefault(m.name[:-4], {})["tif_off"] = m.offset_data
                items[m.name[:-4]]["tif_size"] = m.size
            elif m.name.endswith(".json"):
                items.setdefault(m.name[:-5], {})["js_off"] = m.offset_data
                items[m.name[:-5]]["js_size"] = m.size
    pairs = [(p, it["tif_off"], it["tif_size"], it["js_off"], it["js_size"]) 
             for p, it in items.items() if "tif_off" in it and "js_off" in it]
    with open(idx_path, "wb") as f: pickle.dump(pairs, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_all_sample_refs(shard_root: str) -> List[SampleRef]:
    tar_paths = sorted(glob.glob(os.path.join(shard_root, "*", "plate=*", "*.tar")))
    if not tar_paths: raise FileNotFoundError(f"No tar shards under: {shard_root}")
    for tp in tar_paths: build_tar_index_if_needed(tp)
    refs = []
    for tp in tar_paths:
        line, plate = _infer_line_and_plate_from_tarpath(tp)
        superclass = SUPERCLASS_MAP.get(line, line)
        label = CLASS_TO_LABEL[superclass]
        with open(tp + ".pkl", "rb") as f: pairs = pickle.load(f)
        for pref, tif_off, tif_size, js_off, js_size in pairs:
            refs.append(SampleRef(tp, pref, int(tif_off), int(tif_size), int(js_off), int(js_size),
                                  line, superclass, label, plate))
    logger.info(f"Loaded {len(refs)} sample refs")
    return refs

# ==============================================================================
# Balanced subset
# ==============================================================================
def select_balanced_subset(refs, max_samples, seed):
    rng = random.Random(seed)
    by_super = defaultdict(list)
    for i, r in enumerate(refs): by_super[r.superclass].append(i)
    per_class = max_samples // 4
    targets = {"Control": per_class, "SNCA": per_class, "GBA": per_class, "LRRK2": per_class}
    selected = []
    
    def pick(line_name, sup, target_n):
        plate_to_idxs = defaultdict(list)
        for i in by_super[sup]:
            if refs[i].line == line_name: plate_to_idxs[refs[i].plate].append(i)
        plates = sorted(plate_to_idxs.keys())
        if not plates: return []
        for p in plates: rng.shuffle(plate_to_idxs[p])
        target_n = min(target_n, sum(len(v) for v in plate_to_idxs.values()))
        per_plate = target_n // len(plates)
        out = []
        for p in plates: out.extend(plate_to_idxs[p][:per_plate])
        ptr = {p: per_plate for p in plates}
        remp = target_n - len(out)
        pi = 0
        while remp > 0:
            p = plates[pi % len(plates)]
            if ptr[p] < len(plate_to_idxs[p]):
                out.append(plate_to_idxs[p][ptr[p]]); ptr[p] += 1; remp -= 1
            pi += 1
            if pi > 1e7: break
        return out[:target_n]
    
    for li, line in enumerate(["Control_C4", "Control_C18", "Control_C19"]):
        t = targets["Control"] // 3 + (1 if li < targets["Control"] % 3 else 0)
        selected.extend(pick(line, "Control", t))
    for sup in ["SNCA", "GBA", "LRRK2"]:
        selected.extend(pick(sup, sup, targets[sup]))
    rng.shuffle(selected)
    return selected

# ==============================================================================
# Batch sampler
# ==============================================================================
class StrictPlateBalancedBatchSamplerOnBank(Sampler):
    def __init__(self, bank, batch_size, seed):
        self.bank, self.batch_size, self.seed, self._epoch = bank, batch_size, seed, 0
        self.orig = defaultdict(list)
        for j in range(len(bank.images)):
            if bank.images[j] is None: continue
            sup = SUPERCLASS_MAP.get(bank.lines[j], bank.lines[j])
            self.orig[(sup, bank.lines[j], bank.plates[j])].append(j)
        self.line_plates = defaultdict(list)
        for (sup, line, plate) in self.orig: self.line_plates[(sup, line)].append(plate)
        for k in self.line_plates: self.line_plates[k] = sorted(set(self.line_plates[k]))
        self.control_lines = ["Control_C4", "Control_C18", "Control_C19"]
    
    def __len__(self): return max(1, sum(len(v) for v in self.orig.values()) // self.batch_size)
    
    def __iter__(self):
        self._epoch += 1
        rng = random.Random(self.seed + self._epoch)
        g = {k: deque(rng.sample(lst, len(lst))) for k, lst in self.orig.items()}
        def take(sup, line, plate, n):
            dq = g.get((sup, line, plate))
            return [dq.popleft() for _ in range(min(n, len(dq) if dq else 0))]
        while True:
            bs = self.batch_size; per = bs // 4; batch = []
            targets = {"Control": per, "SNCA": per, "GBA": per, "LRRK2": per}
            for li, line in enumerate(self.control_lines):
                need = targets["Control"] // 3 + (1 if li < targets["Control"] % 3 else 0)
                plates = self.line_plates.get(("Control", line), [])
                if plates:
                    rng.shuffle(plates)
                    pp = need // len(plates)
                    for p in plates: batch.extend(take("Control", line, p, pp))
                    for p in plates[:need % len(plates)]: batch.extend(take("Control", line, p, 1))
            for sup in ["SNCA", "GBA", "LRRK2"]:
                plates = self.line_plates.get((sup, sup), [])
                if plates:
                    rng.shuffle(plates)
                    pp = targets[sup] // len(plates)
                    for p in plates: batch.extend(take(sup, sup, p, pp))
                    for p in plates[:targets[sup] % len(plates)]: batch.extend(take(sup, sup, p, 1))
            if len(batch) < self.batch_size: break
            yield batch[:self.batch_size]

# ==============================================================================
# Dataset
# ==============================================================================
class SafeInstanceNormalize:
    def __init__(self, threshold=0.01): self.threshold = threshold
    def __call__(self, t):
        mean = t.mean(dim=[1,2], keepdim=True)
        std = t.std(dim=[1,2], keepdim=True).clamp_min(self.threshold)
        return (t - mean) / std

class InMemoryTarBank:
    def __init__(self, refs, ref_indices, img_size):
        self.refs, self.ref_indices, self.img_size = refs, ref_indices, img_size
        n = len(ref_indices)
        self.images, self.labels, self.plates, self.lines, self.uids = [None]*n, [0]*n, [""]*n, [""]*n, [""]*n
        logger.info(f"Preloading {n} images...")
        tar_to_fh = {}
        def read(tp, off, size):
            if tp not in tar_to_fh: tar_to_fh[tp] = open(tp, "rb", buffering=0)
            tar_to_fh[tp].seek(off); return tar_to_fh[tp].read(size)
        for j, ridx in enumerate(tqdm(ref_indices, desc="preload")):
            r = refs[ridx]
            try:
                img = tifffile.imread(io.BytesIO(read(r.tar_path, r.tif_off, r.tif_size)))
                if img.dtype == np.uint16 and img.shape == (img_size, img_size, 3):
                    self.images[j] = img
                    self.labels[j], self.plates[j], self.lines[j] = r.label, r.plate, r.line
                    self.uids[j] = f"{r.tar_path}:{r.prefix}"
            except: pass
        for fh in tar_to_fh.values(): fh.close()

class InMemorySixteenBitDataset(Dataset):
    def __init__(self, bank, indices, img_size, two_crops, augment):
        self.bank, self.ib, self.img_size, self.two_crops = bank, indices, img_size, two_crops
        aug = transforms.RandomChoice([transforms.Lambda(lambda x: x),
              transforms.Lambda(lambda x: torch.rot90(x,1,[1,2])),
              transforms.Lambda(lambda x: torch.rot90(x,2,[1,2])),
              transforms.Lambda(lambda x: torch.rot90(x,3,[1,2]))]) if augment else transforms.Lambda(lambda x: x)
        self.transform = transforms.Compose([aug, SafeInstanceNormalize(0.01)])
    def __len__(self): return len(self.ib)
    def __getitem__(self, idx):
        j = self.ib[idx]; img = self.bank.images[j]
        if img is None: return None
        y = torch.tensor(self.bank.labels[j], dtype=torch.long)
        x = torch.from_numpy(img.astype(np.float32) / 65535.0).permute(2,0,1)
        if self.two_crops: return self.transform(x), self.transform(x), y, self.bank.plates[j], self.bank.lines[j], self.bank.uids[j]
        return self.transform(x), y, self.bank.plates[j], self.bank.lines[j], self.bank.uids[j]

def save_split_csv(uids, labels, refs_by_uid, save_dir, filename):
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, filename), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["uid","label","superclass","line","plate","tar_path","prefix"])
        for uid, lb in zip(uids, labels):
            r = refs_by_uid[uid]; w.writerow([uid, lb, r.superclass, r.line, r.plate, r.tar_path, r.prefix])

def load_split_csv(path):
    with open(path, "r", encoding="utf-8") as f: return [row["uid"] for row in csv.DictReader(f)]

# ==============================================================================
# Model
# ==============================================================================
OUT_DIM = 512

@torch.no_grad()
def renorm_unit_per_out_channel_(model, eps=1e-12):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            w = m.weight.data; n = w.flatten(1).norm(dim=1, keepdim=True).clamp_min(eps); w.div_(n.view(-1,1,1,1))
        elif isinstance(m, nn.Linear):
            w = m.weight.data; n = w.norm(dim=1, keepdim=True).clamp_min(eps); w.div_(n)

def conv2d(in_ch, out_ch, k=3, stride=1, padding=1, dilation=1, bias=True):
    return nn.Conv2d(in_ch, out_ch, k, stride, padding, dilation, bias=bias)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1):
        super().__init__()
        self.c1 = conv2d(in_ch, out_ch, 3, 1, dilation, dilation, True)
        self.c2 = conv2d(out_ch, out_ch, 3, 1, dilation, dilation, True)
        self.proj = conv2d(in_ch, out_ch, 1, 1, 0, bias=False) if in_ch != out_ch else None
    def forward(self, x):
        identity = x; x = F.relu(x, True); x = self.c1(x); x = F.relu(x, True); x = self.c2(x)
        return x + (self.proj(identity) if self.proj else identity)

class Stage(nn.Module):
    def __init__(self, in_ch, out_ch, n_blocks, dilation, use_ckpt, ckpt_seg):
        super().__init__()
        self.use_ckpt, self.ckpt_seg = use_ckpt, ckpt_seg
        self.blocks = nn.Sequential(ResBlock(in_ch, out_ch, dilation), *[ResBlock(out_ch, out_ch, dilation) for _ in range(n_blocks-1)])
    def forward(self, x):
        if self.use_ckpt and self.training and self.ckpt_seg > 1 and len(self.blocks) > 1:
            return checkpoint_sequential(self.blocks, min(self.ckpt_seg, len(self.blocks)), x, use_reentrant=False)
        return self.blocks(x)

class Encoder(nn.Module):
    def __init__(self, blocks=(2,2,4,4), dilations=(1,1,1,1), refine_blocks=1, ckpt_segments=2):
        super().__init__()
        b2,b3,b4,b5 = blocks; d2,d3,d4,d5 = dilations
        self.stem = nn.Sequential(conv2d(3, 64, 3, 2, 1, bias=True))
        self.stage2 = Stage(64, 128, b2, d2, False, 1)
        self.stage3 = Stage(128, 256, b3, d3, False, 1)
        self.stage4 = Stage(256, 512, b4, d4, True, ckpt_segments)
        self.stage5 = Stage(512, OUT_DIM, b5, d5, True, ckpt_segments)
        self.refine = Stage(OUT_DIM, OUT_DIM, refine_blocks, 1, True, ckpt_segments)
        self.trunk = nn.Sequential(self.stem, self.stage2, self.stage3, self.stage4, self.stage5, self.refine)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
    def forward(self, x): return self.gap(self.trunk(x)).flatten(1)

class SupConMoCoModel(nn.Module):
    def __init__(self, embed_dim=512, blocks=(2,2,4,4), dilations=(1,1,1,1), refine_blocks=1, ckpt_segments=2, proj_hidden=2048):
        super().__init__()
        self.encoder = Encoder(blocks, dilations, refine_blocks, ckpt_segments)
        self.projector = nn.Sequential(nn.Linear(OUT_DIM, proj_hidden, bias=False), nn.ReLU(), nn.Linear(proj_hidden, embed_dim, bias=False))
    def forward(self, x):
        pooled = F.normalize(self.encoder(x), dim=1)
        return F.normalize(self.projector(pooled), dim=1)

# ==============================================================================
# MoCo Queue (Supervised - stores keys with labels)
# ==============================================================================
class MoCoQueue:
    """FIFO queue storing (key, label) pairs for supervised contrastive learning."""
    def __init__(self, dim: int, capacity: int, device: torch.device, dtype=torch.float16):
        self.dim, self.capacity, self.device, self.dtype = dim, capacity, device, dtype
        self.reset()
    
    def reset(self):
        self.ptr = 0
        self.full = False
        self.keys = torch.zeros(self.capacity, self.dim, device=self.device, dtype=self.dtype)
        self.labels = torch.zeros(self.capacity, device=self.device, dtype=torch.long)
    
    @torch.no_grad()
    def enqueue(self, keys: torch.Tensor, labels: torch.Tensor):
        """Add new keys and labels to queue (FIFO)."""
        keys, labels = keys.detach(), labels.detach()
        b = keys.size(0)
        if b <= 0: return
        
        # Wrap around if needed
        end = self.ptr + b
        if end <= self.capacity:
            self.keys[self.ptr:end] = keys.to(self.dtype)
            self.labels[self.ptr:end] = labels
        else:
            first = self.capacity - self.ptr
            self.keys[self.ptr:] = keys[:first].to(self.dtype)
            self.labels[self.ptr:] = labels[:first]
            second = end - self.capacity
            self.keys[:second] = keys[first:].to(self.dtype)
            self.labels[:second] = labels[first:]
        
        self.ptr = end % self.capacity
        if end >= self.capacity: self.full = True
    
    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get all stored keys and labels."""
        if self.full:
            return self.keys, self.labels
        return self.keys[:self.ptr], self.labels[:self.ptr]

# ==============================================================================
# Momentum update
# ==============================================================================
@torch.no_grad()
def momentum_update_(model_q, model_k, m):
    for p_q, p_k in zip(model_q.parameters(), model_k.parameters()):
        p_k.data.mul_(m).add_(p_q.data, alpha=1-m)

# ==============================================================================
# Supervised Contrastive Loss (with queue)
# ==============================================================================
def supervised_contrastive_q_vs_k(q, y_q, k, y_k, temperature):
    """SupCon loss: same class = positive, different class = negative."""
    q, k = q.float(), k.float()
    logits = (q @ k.t()) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values  # stability
    
    with torch.no_grad():
        pos_mask = (y_q.view(-1,1) == y_k.view(1,-1))  # same class = positive
        pos_cnt = pos_mask.sum(dim=1).clamp_min(1)
    
    exp = torch.exp(logits)
    log_prob = logits - torch.log(exp.sum(dim=1, keepdim=True).clamp_min(1e-12))
    loss = -(log_prob * pos_mask).sum(dim=1) / pos_cnt
    return loss.mean()

# ==============================================================================
# Checkpoint
# ==============================================================================
def _get_rng_state():
    st = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available(): st["cuda"] = torch.cuda.get_rng_state_all()
    return st

def _set_rng_state(st):
    try:
        random.setstate(st["python"]); np.random.set_state(st["numpy"]); torch.set_rng_state(st["torch"])
        if torch.cuda.is_available() and "cuda" in st: torch.cuda.set_rng_state_all(st["cuda"])
    except: pass

def save_ckpt(path, epoch, step, model_q, model_k, opt, scaler, queue, best_acc, patience, args):
    torch.save({"epoch": epoch, "global_step": step, "model_q": model_q.state_dict(), "model_k": model_k.state_dict(),
                "opt": opt.state_dict(), "scaler": scaler.state_dict() if scaler else None,
                "queue": {"keys": queue.keys.cpu(), "labels": queue.labels.cpu(), "ptr": queue.ptr, "full": queue.full},
                "best_acc": best_acc, "patience_counter": patience, "args": vars(args), "rng": _get_rng_state()}, path)

def load_ckpt(path, model_q, model_k, opt, scaler, queue, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model_q.load_state_dict(ckpt["model_q"]); model_k.load_state_dict(ckpt["model_k"]); opt.load_state_dict(ckpt["opt"])
    if scaler and ckpt.get("scaler"): scaler.load_state_dict(ckpt["scaler"])
    queue.keys.copy_(ckpt["queue"]["keys"].to(device)); queue.labels.copy_(ckpt["queue"]["labels"].to(device))
    queue.ptr, queue.full = ckpt["queue"]["ptr"], ckpt["queue"]["full"]
    if "rng" in ckpt: _set_rng_state(ckpt["rng"])
    for state in opt.state.values():
        for k, v in state.items():
            if torch.is_tensor(v): state[k] = v.to(device)
    return ckpt["epoch"], ckpt["global_step"], ckpt.get("best_acc", -1), ckpt.get("patience_counter", 0)

# ==============================================================================
# Linear Probe
# ==============================================================================
@torch.no_grad()
def bake_encoder(model_q, args, device):
    blocks = tuple(int(x) for x in args.blocks.split(","))
    dilations = tuple(int(x) for x in args.dilations.split(","))
    enc = Encoder(blocks, dilations, args.refine_blocks, args.ckpt_segments)
    enc.load_state_dict(model_q.encoder.state_dict())
    enc.eval().to(device).to(memory_format=torch.channels_last)
    renorm_unit_per_out_channel_(enc)
    return enc

def linear_probe_acc(args, device, encoder, train_loader, val_loader):
    probe = nn.Linear(OUT_DIM, args.num_classes, bias=False).to(device)
    opt = optim.SGD(probe.parameters(), lr=args.lp_lr, momentum=args.lp_momentum, weight_decay=args.lp_wd)
    ce = nn.CrossEntropyLoss()
    autocast_kw = {"device_type": "cuda", "enabled": torch.cuda.is_available()}
    if args.use_bf16: autocast_kw["dtype"] = torch.bfloat16
    
    probe.train()
    for _ in range(args.lp_epochs):
        for batch in train_loader:
            if batch is None: continue
            x, y = batch[0].to(device).contiguous(memory_format=torch.channels_last), batch[1].to(device)
            with torch.no_grad(), torch.amp.autocast(**autocast_kw):
                feat = F.normalize(encoder(x), dim=1)
            opt.zero_grad(); loss = ce(probe(feat.float()), y); loss.backward(); opt.step()
    
    probe.eval(); correct = total = 0
    with torch.inference_mode():
        for batch in val_loader:
            if batch is None: continue
            x, y = batch[0].to(device).contiguous(memory_format=torch.channels_last), batch[1].to(device)
            with torch.amp.autocast(**autocast_kw):
                feat = F.normalize(encoder(x), dim=1)
            correct += (probe(feat.float()).argmax(1) == y).sum().item(); total += y.numel()
    del probe, opt; torch.cuda.empty_cache()
    return correct / total if total else 0

# ==============================================================================
# LR Scheduler
# ==============================================================================
def lr_at_step(step, base_lr, min_lr, warmup_steps, total_steps):
    if step < warmup_steps: return base_lr * (step + 1) / warmup_steps
    t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * t))

# ==============================================================================
# Trainer
# ==============================================================================
class Trainer:
    def __init__(self, args, model_q, model_k, queue, train_loader, train_lp_loader, val_lp_loader):
        self.args, self.model_q, self.model_k, self.queue = args, model_q, model_k, queue
        self.train_loader, self.train_lp_loader, self.val_lp_loader = train_loader, train_lp_loader, val_lp_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model_q.to(self.device).to(memory_format=torch.channels_last)
        self.model_k.to(self.device).to(memory_format=torch.channels_last)
        for p in self.model_k.parameters(): p.requires_grad = False
        
        self.opt = optim.SGD(self.model_q.parameters(), lr=args.lr, momentum=args.sgd_momentum, 
                             weight_decay=args.wd, nesterov=args.sgd_nesterov)
        
        self.steps_per_epoch = max(1, len(train_loader))
        self.total_steps = args.epochs * self.steps_per_epoch
        self.warmup_steps = int(args.warmup_epochs * self.steps_per_epoch)
        
        self.use_bf16 = args.use_bf16 and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available() and not self.use_bf16)
        
        os.makedirs(args.save_dir, exist_ok=True)
        self.best_path = os.path.join(args.save_dir, "best_model.pt")
        self.last_path = os.path.join(args.save_dir, "last_model.pt")
        self.resume_path = os.path.join(args.save_dir, "resume_ckpt.pt")
        
        renorm_unit_per_out_channel_(self.model_q); renorm_unit_per_out_channel_(self.model_k)
        
        self.start_epoch, self.global_step, self.best_acc, self.patience_counter = 1, 0, -1, 0
        
        if args.auto_resume and os.path.exists(self.resume_path):
            ep, gs, ba, pc = load_ckpt(self.resume_path, self.model_q, self.model_k, self.opt, self.scaler, self.queue, self.device)
            self.start_epoch, self.global_step, self.best_acc, self.patience_counter = ep+1, gs, ba, pc
            logger.info(f"[resume] epoch={self.start_epoch}, step={gs}, best={ba*100:.2f}%")
    
    def train_epoch(self, epoch):
        self.model_q.train(); self.model_k.eval()
        total_loss, steps = 0, 0
        autocast_kw = {"device_type": "cuda", "enabled": torch.cuda.is_available()}
        if self.use_bf16: autocast_kw["dtype"] = torch.bfloat16
        
        # Queue warm-up: don't use queue until it has enough samples
        use_queue = epoch >= self.args.queue_start_epoch
        
        pbar = tqdm(self.train_loader, desc=f"E{epoch}/{self.args.epochs}")
        for batch in pbar:
            if batch is None: continue
            v1, v2, y, _, _, _ = batch
            if y.numel() < 2: continue
            
            v1 = v1.to(self.device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            v2 = v2.to(self.device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            y = y.to(self.device, non_blocking=True)
            
            # Momentum update BEFORE forward
            with torch.no_grad(): momentum_update_(self.model_q, self.model_k, self.args.moco_m)
            
            # LR schedule
            lr = lr_at_step(self.global_step, self.args.lr, self.args.min_lr, self.warmup_steps, self.total_steps)
            for pg in self.opt.param_groups: pg["lr"] = lr
            
            self.opt.zero_grad(set_to_none=True)
            
            with torch.amp.autocast(**autocast_kw):
                q = self.model_q(v1)
                with torch.no_grad():
                    k = self.model_k(v2)
                
                # Build key pool: current batch + queue
                if use_queue:
                    q_keys, q_labels = self.queue.get()
                    if q_keys.numel() > 0:
                        all_k = torch.cat([k, q_keys], dim=0)
                        all_y = torch.cat([y, q_labels], dim=0)
                    else:
                        all_k, all_y = k, y
                else:
                    all_k, all_y = k, y
                
                loss = supervised_contrastive_q_vs_k(q, y, all_k, all_y, self.args.temp)
                
                if self.args.symmetric_loss:
                    q2 = self.model_q(v2)
                    with torch.no_grad(): k2 = self.model_k(v1)
                    if use_queue and q_keys.numel() > 0:
                        all_k2 = torch.cat([k2, q_keys], dim=0)
                    else:
                        all_k2 = k2
                    loss2 = supervised_contrastive_q_vs_k(q2, y, all_k2, all_y, self.args.temp)
                    loss = 0.5 * (loss + loss2)
            
            # Backward
            if self.use_bf16:
                loss.backward()
                if self.args.grad_clip > 0: torch.nn.utils.clip_grad_norm_(self.model_q.parameters(), self.args.grad_clip)
                self.opt.step()
            else:
                self.scaler.scale(loss).backward()
                if self.args.grad_clip > 0: self.scaler.unscale_(self.opt); torch.nn.utils.clip_grad_norm_(self.model_q.parameters(), self.args.grad_clip)
                self.scaler.step(self.opt); self.scaler.update()
            
            # Renorm weights
            if self.global_step % self.args.renorm_every == 0:
                renorm_unit_per_out_channel_(self.model_q)
            
            # Enqueue new keys (always, even before queue is used for loss)
            with torch.no_grad():
                self.queue.enqueue(k, y)
            
            total_loss += loss.item(); steps += 1; self.global_step += 1
            qsize = self.queue.capacity if self.queue.full else self.queue.ptr
            pbar.set_postfix({"lr": f"{lr:.2e}", "loss": f"{loss.item():.4f}", "queue": qsize})
        
        return total_loss / steps if steps else 0
    
    def run(self):
        logger.info(f"Device: {self.device}, bf16={self.use_bf16}, queue_size={self.args.queue_size}")
        
        for epoch in range(self.start_epoch, self.args.epochs + 1):
            tqdm.write(f"\n===== Epoch {epoch}/{self.args.epochs} =====")
            train_loss = self.train_epoch(epoch)
            
            enc = bake_encoder(self.model_q, self.args, self.device)
            val_acc = linear_probe_acc(self.args, self.device, enc, self.train_lp_loader, self.val_lp_loader)
            del enc; torch.cuda.empty_cache()
            
            tqdm.write(f"Epoch {epoch} | Loss: {train_loss:.4f} | LP: {val_acc*100:.2f}%")
            
            torch.save(self.model_q.state_dict(), self.last_path)
            save_ckpt(self.resume_path, epoch, self.global_step, self.model_q, self.model_k, 
                     self.opt, self.scaler, self.queue, self.best_acc, self.patience_counter, self.args)
            
            if val_acc > self.best_acc:
                self.best_acc = val_acc; self.patience_counter = 0
                torch.save(self.model_q.state_dict(), self.best_path)
                tqdm.write(f"  -> Best! {val_acc*100:.2f}%")
            else:
                self.patience_counter += 1
                tqdm.write(f"  -> No improve ({self.patience_counter}/{self.args.patience})")
                if self.patience_counter >= self.args.patience:
                    tqdm.write("Early stopping!"); break

# ==============================================================================
# Args
# ==============================================================================
def get_args():
    p = argparse.ArgumentParser("MoCo+Queue (Supervised)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_dir", type=str, default="/content/drive/MyDrive/Final_paper/Model_MoCo_Queue")
    p.add_argument("--shard_root", type=str, default=DEFAULT_SHARD_ROOT)
    p.add_argument("--max_samples", type=int, default=108000)
    p.add_argument("--test_ratio", type=float, default=1/3)
    p.add_argument("--val_ratio", type=float, default=0.25)
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--embed_dim", type=int, default=512)
    p.add_argument("--blocks", type=str, default="2,2,2,3")
    p.add_argument("--dilations", type=str, default="1,1,1,1")
    p.add_argument("--refine_blocks", type=int, default=1)
    p.add_argument("--ckpt_segments", type=int, default=0)
    p.add_argument("--proj_hidden", type=int, default=2048)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--sgd_momentum", type=float, default=0.9)
    p.add_argument("--sgd_nesterov", action="store_true")
    p.add_argument("--temp", type=float, default=0.1)
    p.add_argument("--warmup_epochs", type=int, default=4)
    p.add_argument("--min_lr", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--use_bf16", action="store_true")
    p.add_argument("--renorm_every", type=int, default=1)
    p.add_argument("--symmetric_loss", action="store_true")
    # MoCo
    p.add_argument("--moco_m", type=float, default=0.995)
    # Queue
    p.add_argument("--queue_size", type=int, default=65536, help="Queue capacity (e.g., 65536)")
    p.add_argument("--queue_start_epoch", type=int, default=2, help="Epoch to start using queue for loss")
    # LP
    p.add_argument("--num_classes", type=int, default=4)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--lp_epochs", type=int, default=3)
    p.add_argument("--lp_batch_size", type=int, default=16384)
    p.add_argument("--lp_lr", type=float, default=0.1)
    p.add_argument("--lp_wd", type=float, default=0.0)
    p.add_argument("--lp_momentum", type=float, default=0.9)
    p.add_argument("--auto_resume", action="store_true")
    if "ipykernel" in sys.modules: return p.parse_args([])
    return p.parse_args()

# ==============================================================================
# Main
# ==============================================================================
def main():
    args = get_args()
    args.use_bf16 = True
    args.symmetric_loss = True
    args.auto_resume = True
    
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "args.json"), "w") as f: json.dump(vars(args), f, indent=2)
    
    set_seed(args.seed)
    refs = load_all_sample_refs(args.shard_root)
    
    train_csv, val_csv, test_csv = [os.path.join(args.save_dir, f"{s}_split.csv") for s in ["train","val","test"]]
    resume_ckpt = os.path.join(args.save_dir, "resume_ckpt.pt")
    will_resume = args.auto_resume and os.path.exists(resume_ckpt)
    
    if will_resume and all(os.path.exists(p) for p in [train_csv, val_csv, test_csv]):
        X_train, X_val = load_split_csv(train_csv), load_split_csv(val_csv)
        uid_to_idx = {f"{r.tar_path}:{r.prefix}": i for i, r in enumerate(refs)}
        train_refidx = [uid_to_idx[u] for u in X_train]
        val_refidx = [uid_to_idx[u] for u in X_val]
    else:
        subset = select_balanced_subset(refs, args.max_samples, args.seed)
        uids = [f"{refs[i].tar_path}:{refs[i].prefix}" for i in subset]
        labels = [refs[i].label for i in subset]
        lines = [refs[i].line for i in subset]
        line_id = {l:i for i,l in enumerate(sorted(set(lines)))}
        strat = [line_id[l] for l in lines]
        X_temp, X_test, y_temp, _, st_temp, _ = train_test_split(uids, labels, strat, test_size=args.test_ratio, random_state=args.seed, stratify=strat)
        X_train, X_val, y_train, y_val, _, _ = train_test_split(X_temp, y_temp, st_temp, test_size=args.val_ratio, random_state=args.seed, stratify=st_temp)
        refs_by_uid = {f"{refs[i].tar_path}:{refs[i].prefix}": refs[i] for i in subset}
        save_split_csv(X_train, y_train, refs_by_uid, args.save_dir, "train_split.csv")
        save_split_csv(X_val, y_val, refs_by_uid, args.save_dir, "val_split.csv")
        save_split_csv(X_test, [refs_by_uid[u].label for u in X_test], refs_by_uid, args.save_dir, "test_split.csv")
        uid_to_idx = {f"{refs[i].tar_path}:{refs[i].prefix}": i for i in subset}
        train_refidx = [uid_to_idx[u] for u in X_train]
        val_refidx = [uid_to_idx[u] for u in X_val]
    
    train_bank = InMemoryTarBank(refs, train_refidx, args.img_size)
    val_bank = InMemoryTarBank(refs, val_refidx, args.img_size)
    
    train_ds = InMemorySixteenBitDataset(train_bank, list(range(len(train_refidx))), args.img_size, True, True)
    train_lp_ds = InMemorySixteenBitDataset(train_bank, list(range(len(train_refidx))), args.img_size, False, False)
    val_lp_ds = InMemorySixteenBitDataset(val_bank, list(range(len(val_refidx))), args.img_size, False, False)
    
    sampler = StrictPlateBalancedBatchSamplerOnBank(train_bank, args.batch_size, args.seed)
    train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=0, pin_memory=True, collate_fn=collate_skip_none)
    train_lp_loader = DataLoader(train_lp_ds, batch_size=args.lp_batch_size, shuffle=True, num_workers=0, pin_memory=True, collate_fn=collate_skip_none)
    val_lp_loader = DataLoader(val_lp_ds, batch_size=args.lp_batch_size, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_skip_none)
    
    blocks = tuple(int(x) for x in args.blocks.split(","))
    dilations = tuple(int(x) for x in args.dilations.split(","))
    
    model_q = SupConMoCoModel(args.embed_dim, blocks, dilations, args.refine_blocks, args.ckpt_segments, args.proj_hidden)
    model_k = SupConMoCoModel(args.embed_dim, blocks, dilations, args.refine_blocks, args.ckpt_segments, args.proj_hidden)
    model_k.load_state_dict(model_q.state_dict())
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    queue = MoCoQueue(args.embed_dim, args.queue_size, device, torch.float16)
    
    trainer = Trainer(args, model_q, model_k, queue, train_loader, train_lp_loader, val_lp_loader)
    trainer.run()

if __name__ == "__main__":
    main()
