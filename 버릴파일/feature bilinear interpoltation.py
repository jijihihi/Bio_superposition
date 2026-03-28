## bilinear interpolation으로 피처맵 어디 봤는지 시각화.

# ============================================================
# Feature-map "where it looks" visualization (pos+neg)
# - Scan Top-K per channel by abs-peak: score = max(pos_max, -neg_min)
# - Record why selected: winner=pos/neg + (pos_max, neg_abs)
# - Save per channel K sets; each set has 4 images:
#   01_orig (linear), 02_bright (percentile stretch),
#   03_neg_minmax_colormap (neg emphasized), 04_pos_minmax_colormap (pos emphasized)
# - File name includes line + winner + pos/neg values
# ============================================================

import os, io, csv, json, pickle, hashlib, re
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

import tifffile
from PIL import Image
from torch.utils.checkpoint import checkpoint_sequential

# =========================
# Constants / maps
# =========================
OUT_DIM = 512
PLATE_DIR_RE = re.compile(r"plate=(\d{6})")

# =========================
# Model Definition (bias=True)
# =========================
def conv2d(in_ch, out_ch, k=3, stride=1, padding=1, dilation=1, bias=True):
    return nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=stride, padding=padding, dilation=dilation, bias=bias)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1):
        super().__init__()
        self.c1 = conv2d(in_ch, out_ch, 3, 1, padding=dilation, dilation=dilation, bias=True)
        self.c2 = conv2d(out_ch, out_ch, 3, 1, padding=dilation, dilation=dilation, bias=True)
        self.proj = None
        if in_ch != out_ch:
            self.proj = conv2d(in_ch, out_ch, 1, 1, padding=0, bias=False)

    def forward(self, x):
        identity = x
        x = F.relu(x, inplace=True)
        x = self.c1(x)
        x = F.relu(x, inplace=True)
        x = self.c2(x)
        if self.proj is not None:
            identity = self.proj(identity)
        return x + identity

class Stage(nn.Module):
    def __init__(self, in_ch, out_ch, n_blocks, dilation, use_ckpt: bool, ckpt_segments: int):
        super().__init__()
        self.use_ckpt = bool(use_ckpt)
        self.ckpt_segments = int(ckpt_segments)
        blocks = [ResBlock(in_ch, out_ch, dilation=dilation)]
        for _ in range(n_blocks - 1):
            blocks.append(ResBlock(out_ch, out_ch, dilation=dilation))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        if self.use_ckpt and self.training and self.ckpt_segments > 1 and len(self.blocks) > 1:
            seg = min(self.ckpt_segments, len(self.blocks))
            return checkpoint_sequential(self.blocks, seg, x, use_reentrant=False)
        return self.blocks(x)

class Encoder(nn.Module):
    def __init__(self, blocks=(2,2,4,4), dilations=(1,1,1,1), refine_blocks=1, ckpt_segments=2):
        super().__init__()
        b2,b3,b4,b5 = blocks
        d2,d3,d4,d5 = dilations

        self.stem = nn.Sequential(conv2d(3, 64, k=3, stride=2, padding=1, bias=True))
        self.stage2 = Stage(64, 128, b2, d2, use_ckpt=False, ckpt_segments=1)
        self.stage3 = Stage(128, 256, b3, d3, use_ckpt=False, ckpt_segments=1)
        self.stage4 = Stage(256, 512, b4, d4, use_ckpt=True, ckpt_segments=ckpt_segments)
        self.stage5 = Stage(512, OUT_DIM, b5, d5, use_ckpt=True, ckpt_segments=ckpt_segments)
        self.refine = Stage(OUT_DIM, OUT_DIM, int(refine_blocks), 1, use_ckpt=True, ckpt_segments=ckpt_segments)

        self.trunk = nn.Sequential(self.stem, self.stage2, self.stage3, self.stage4, self.stage5, self.refine)
        self.gap = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        x = self.trunk(x)
        x = self.gap(x).flatten(1)
        return x

def build_projector(in_dim: int, embed_dim: int, proj_layers: int, proj_hidden: int,
                    use_bn: bool, dropout: float) -> nn.Module:
    proj_layers = int(proj_layers)
    proj_hidden = int(proj_hidden)
    dropout = float(dropout)

    def lin(a, b):
        return nn.Linear(a, b, bias=False)

    def bn(d):
        return nn.BatchNorm1d(d) if use_bn else nn.Identity()

    def do():
        return nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    if proj_layers <= 1:
        return nn.Sequential(lin(in_dim, embed_dim))

    if proj_layers == 2:
        return nn.Sequential(
            lin(in_dim, proj_hidden),
            bn(proj_hidden),
            nn.ReLU(inplace=False),
            do(),
            lin(proj_hidden, embed_dim),
        )

    if proj_layers == 3:
        return nn.Sequential(
            lin(in_dim, proj_hidden),
            bn(proj_hidden),
            nn.ReLU(inplace=False),
            do(),
            lin(proj_hidden, proj_hidden),
            bn(proj_hidden),
            nn.ReLU(inplace=False),
            do(),
            lin(proj_hidden, embed_dim),
        )

    raise ValueError(f"Unsupported proj_layers={proj_layers}")

class SupConMoCoModel(nn.Module):
    def __init__(
        self,
        embed_dim=512,
        blocks=(2,2,4,4),
        dilations=(1,1,1,1),
        refine_blocks=1,
        ckpt_segments=2,
        proj_layers: int = 2,
        proj_hidden: int = 2048,
        proj_bn: bool = False,
        proj_dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder = Encoder(
            blocks=blocks,
            dilations=dilations,
            refine_blocks=refine_blocks,
            ckpt_segments=ckpt_segments
        )

        self.projector = build_projector(
            in_dim=OUT_DIM,
            embed_dim=int(embed_dim),
            proj_layers=int(proj_layers),
            proj_hidden=int(proj_hidden),
            use_bn=bool(proj_bn),
            dropout=float(proj_dropout),
        )

    def forward(self, x):
        pooled = self.encoder(x)
        pooled = F.normalize(pooled, dim=1)
        z = self.projector(pooled)
        return F.normalize(z, dim=1)

LINE_FOLDERS = [
    "Control_C4", "Control_C18", "Control_C19",
    "SNCA", "GBA", "LRRK2"
]
SUPERCLASS_MAP = {
    "Control_C4":  "Control",
    "Control_C18": "Control",
    "Control_C19": "Control",
    "SNCA":        "SNCA",
    "GBA":         "GBA",
    "LRRK2":       "LRRK2",
}
CLASS_TO_LABEL = {"Control": 0, "SNCA": 1, "GBA": 2, "LRRK2": 3}
LABEL_TO_CLASS = {v: k for k, v in CLASS_TO_LABEL.items()}

# =========================
# Normalize (same as training)
# =========================
class SafeInstanceNormalize:
    def __init__(self, threshold: float = 0.01):
        self.threshold = float(threshold)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = tensor.mean(dim=[1, 2], keepdim=True)
        std = tensor.std(dim=[1, 2], keepdim=True).clamp_min(self.threshold)
        return (tensor - mean) / std

def validate_uint16_rgb(img: np.ndarray, img_size: int):
    if img is None:
        raise ValueError("decoded None")
    if img.dtype != np.uint16:
        raise ValueError(f"dtype must be uint16, got {img.dtype}")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"shape must be HxWx3, got {img.shape}")
    h, w = img.shape[:2]
    if (h, w) != (img_size, img_size):
        raise ValueError(f"size must be {(img_size, img_size)}, got {(h, w)}")

# =========================
# refs / tar offset index
# =========================
@dataclass(frozen=True)
class SampleRef:
    tar_path: str
    prefix: str
    tif_off: int
    tif_size: int
    js_off: int
    js_size: int
    line: str
    superclass: str
    label: int
    plate: str

def _infer_line_and_plate_from_tarpath(tar_path: str) -> Tuple[str, str]:
    parts = tar_path.replace("\\", "/").split("/")
    line = parts[-3]
    m = PLATE_DIR_RE.search(parts[-2])
    plate = m.group(1) if m else "UNKNOWN"
    return line, plate

def build_tar_index_if_needed(tar_path: str):
    idx_path = tar_path + ".pkl"
    if os.path.exists(idx_path):
        return

    import tarfile
    items = {}
    with tarfile.open(tar_path, "r") as tf:
        for m in tf.getmembers():
            if not m.isreg():
                continue
            name = m.name
            if name.endswith(".tif"):
                pref = name[:-4]
                it = items.get(pref, {})
                it["tif_off"] = m.offset_data
                it["tif_size"] = m.size
                items[pref] = it
            elif name.endswith(".json"):
                pref = name[:-5]
                it = items.get(pref, {})
                it["js_off"] = m.offset_data
                it["js_size"] = m.size
                items[pref] = it

    pairs = []
    for pref, it in items.items():
        if "tif_off" in it and "js_off" in it:
            pairs.append((pref, it["tif_off"], it["tif_size"], it["js_off"], it["js_size"]))

    with open(idx_path, "wb") as f:
        pickle.dump(pairs, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_all_sample_refs(shard_root: str) -> List[SampleRef]:
    import glob
    tar_paths = sorted(glob.glob(os.path.join(shard_root, "*", "plate=*", "*.tar")))
    if len(tar_paths) == 0:
        raise FileNotFoundError(f"No tar shards found under: {shard_root}")

    for tp in tar_paths:
        build_tar_index_if_needed(tp)

    refs: List[SampleRef] = []
    for tp in tar_paths:
        line, plate = _infer_line_and_plate_from_tarpath(tp)
        superclass = SUPERCLASS_MAP.get(line, line)
        label = CLASS_TO_LABEL[superclass]

        with open(tp + ".pkl", "rb") as f:
            pairs = pickle.load(f)

        for pref, tif_off, tif_size, js_off, js_size in pairs:
            refs.append(SampleRef(
                tar_path=tp,
                prefix=pref,
                tif_off=int(tif_off),
                tif_size=int(tif_size),
                js_off=int(js_off),
                js_size=int(js_size),
                line=line,
                superclass=superclass,
                label=label,
                plate=plate
            ))
    return refs

# =========================
# Split CSV loader (uid only)
# =========================
def load_uids_from_split_csv(path: str) -> List[str]:
    uids = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            uids.append(row["uid"])
    return uids

# =========================
# Tar reader (seek+read cache)
# =========================
class TarOffsetReader:
    def __init__(self):
        self._fh = {}  # tar_path -> file handle

    def close(self):
        for fh in self._fh.values():
            try:
                fh.close()
            except:
                pass
        self._fh = {}

    def read_bytes(self, tar_path: str, off: int, size: int) -> bytes:
        fh = self._fh.get(tar_path, None)
        if fh is None:
            fh = open(tar_path, "rb", buffering=0)
            self._fh[tar_path] = fh
        fh.seek(off)
        return fh.read(size)

    def read_uint16_rgb(self, ref: SampleRef, img_size: int) -> np.ndarray:
        tif_bytes = self.read_bytes(ref.tar_path, ref.tif_off, ref.tif_size)
        img = tifffile.imread(io.BytesIO(tif_bytes))
        validate_uint16_rgb(img, img_size)
        return img

# =========================
# Scan Dataset (no aug; only model input)
# =========================
class ScanDataset(Dataset):
    def __init__(self, refs: List[SampleRef], refidx_list: List[int], img_size: int):
        self.refs = refs
        self.refidx_list = refidx_list
        self.img_size = int(img_size)
        self.norm = SafeInstanceNormalize(threshold=0.01)
        self._reader = None  # lazy init per worker

    def _get_reader(self):
        if self._reader is None:
            self._reader = TarOffsetReader()
        return self._reader

    def __len__(self):
        return len(self.refidx_list)

    def __getitem__(self, i: int):
        ridx = self.refidx_list[i]
        ref = self.refs[ridx]
        try:
            reader = self._get_reader()
            img = reader.read_uint16_rgb(ref, self.img_size)  # (H,W,3) uint16

            x = (img.astype(np.float32) / 65535.0)
            x = torch.from_numpy(x).permute(2, 0, 1)  # (3,H,W)
            x = self.norm(x)

            y = torch.tensor(ref.label, dtype=torch.long)
            uid = f"{ref.tar_path}:{ref.prefix}"
            return x, y, ridx, uid
        except Exception:
            return None

def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    xs, ys, ridxs, uids = zip(*batch)
    return torch.stack(xs, 0), torch.stack(ys, 0), torch.tensor(ridxs, dtype=torch.long), list(uids)

# =========================
# Scan TopK per channel with reason (pos/neg)
# score = max(pos_max, neg_abs)
# winner: 1=pos, 0=neg
# =========================
@torch.inference_mode()
def scan_topk_abspeak_with_reason(
    model_q,
    loader,
    device,
    K=15,
    use_bf16=True,
):
    C = OUT_DIM
    top_score = torch.full((C, K), -float("inf"), device=device)
    top_refidx = torch.full((C, K), -1, device=device, dtype=torch.long)
    top_winner = torch.full((C, K), -1, device=device, dtype=torch.long)  # 1=pos, 0=neg
    top_posval = torch.full((C, K), -float("inf"), device=device)
    top_negval = torch.full((C, K), -float("inf"), device=device)

    autocast_kwargs = dict(device_type="cuda", enabled=(device.type == "cuda"))
    if use_bf16 and device.type == "cuda":
        autocast_kwargs["dtype"] = torch.bfloat16

    first_shape_printed = False

    for batch in tqdm(loader, desc="Scan(abs-peak + reason)"):
        if batch is None:
            continue
        x, _, ridx, _ = batch
        x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)

        with torch.amp.autocast(**autocast_kwargs):
            fmap = model_q.encoder.trunk(x)  # (B,C,h,w)

        if not first_shape_printed:
            print("Feature map shape:", tuple(fmap.shape))
            first_shape_printed = True

        fmap = fmap.float()
        posv = fmap.amax(dim=(2, 3))      # (B,C)
        negv = (-fmap.amin(dim=(2, 3)))   # (B,C)  (abs of negative min)
        win = (posv >= negv).to(torch.long)     # (B,C)
        score = torch.maximum(posv, negv)       # (B,C)

        # transpose to (C,B)
        score_t = score.t().contiguous()
        pos_t   = posv.t().contiguous()
        neg_t   = negv.t().contiguous()
        win_t   = win.t().contiguous()

        batch_ids = ridx.to(device, non_blocking=True)             # (B,)
        batch_ids_mat = batch_ids.view(1, -1).expand(C, -1)        # (C,B)

        cand_score = torch.cat([top_score, score_t], dim=1)        # (C,K+B)
        cand_ref   = torch.cat([top_refidx, batch_ids_mat], dim=1)
        cand_pos   = torch.cat([top_posval, pos_t], dim=1)
        cand_neg   = torch.cat([top_negval, neg_t], dim=1)
        cand_win   = torch.cat([top_winner, win_t], dim=1)

        new_score, idx = cand_score.topk(K, dim=1, largest=True, sorted=True)
        top_score  = new_score
        top_refidx = cand_ref.gather(1, idx)
        top_posval = cand_pos.gather(1, idx)
        top_negval = cand_neg.gather(1, idx)
        top_winner = cand_win.gather(1, idx)

    return (
        top_score.detach().cpu(),
        top_refidx.detach().cpu(),
        top_posval.detach().cpu(),
        top_negval.detach().cpu(),
        top_winner.detach().cpu(),
    )

# =========================
# Load weights (flex: compile/DDP prefixes)
# =========================
def load_weights_flex(model, weight_path, device="cpu", strict=True):
    obj = torch.load(weight_path, map_location=device)
    if isinstance(obj, dict) and "model_q" in obj and isinstance(obj["model_q"], dict):
        sd = obj["model_q"]
    else:
        sd = obj

    if any(k.startswith("_orig_mod.") for k in sd.keys()):
        sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
    if strict and (len(missing) > 0 or len(unexpected) > 0):
        print("missing examples:", missing[:5])
        print("unexpected examples:", unexpected[:5])
        raise RuntimeError("State_dict mismatch. See logs above.")
    return model

# =========================
# Visualization utils
# =========================
def linear_uint16_to_uint8_rgb(img_u16: np.ndarray) -> np.ndarray:
    return (img_u16.astype(np.float32) / 65535.0 * 255.0).round().astype(np.uint8)

def robust_uint16_to_uint8_rgb(img_u16: np.ndarray, p_lo=10.0, p_hi=99.5) -> np.ndarray:
    """
    Percentile-based contrast stretching with MIN_STD threshold check.
    Matches QC code's linear scaling logic.
    """
    MIN_STD_THRESHOLD = 655.0  # for uint16 input (same as QC code)
    
    img = img_u16.astype(np.float32)
    out = np.zeros_like(img, dtype=np.uint8)
    for c in range(3):
        v = img[..., c]
        raw_std = np.std(v)
        
        # Skip scaling if channel has very low variance (no meaningful signal)
        if raw_std < MIN_STD_THRESHOLD:
            # Just do simple linear conversion without percentile stretching
            vv = v / 65535.0
        else:
            lo = np.percentile(v, p_lo)
            hi = np.percentile(v, p_hi)
            if hi <= lo + 1e-6:
                hi = lo + 1.0
            vv = np.clip((v - lo) / (hi - lo), 0, 1)
        
        out[..., c] = (vv * 255.0).round().astype(np.uint8)
    return out

def apply_colormap_01(a01: np.ndarray, cmap_name: str = "jet") -> np.ndarray:
    """
    a01: (H,W) float in [0,1]
    return: (H,W,3) uint8
    """
    import matplotlib.cm as cm
    a01 = np.clip(a01.astype(np.float32), 0.0, 1.0)
    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(a01)  # (H,W,4)
    rgb8 = (rgba[..., :3] * 255.0).round().astype(np.uint8)
    return rgb8

def _minmax01(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx <= mn + eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)

def pos_neg_minmax_colormap(
    act_hw: np.ndarray,
    cmap_pos: str = "jet",
    cmap_neg: str = "jet",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    act_hw: (H,W) float
    Returns RGB uint8 maps:
      neg_rgb8 emphasizes most-negative regions (via -act)
      pos_rgb8 emphasizes most-positive regions
    """
    act_hw = act_hw.astype(np.float32)
    pos = np.clip(act_hw, 0.0, None)
    neg = np.clip(-act_hw, 0.0, None)

    pos01 = _minmax01(pos)
    neg01 = _minmax01(neg)

    pos_rgb8 = apply_colormap_01(pos01, cmap_name=cmap_pos)
    neg_rgb8 = apply_colormap_01(neg01, cmap_name=cmap_neg)
    return neg_rgb8, pos_rgb8

# =========================
# Save per-channel K sets (4 images per set)
# - include winner + pos/neg values in filenames
# =========================
@torch.inference_mode()
def save_topk_sets_per_channel_posneg(
    model_q,
    refs: List[SampleRef],
    top_refidx: torch.Tensor,   # (C,K) CPU
    top_posval: torch.Tensor,   # (C,K) CPU
    top_negval: torch.Tensor,   # (C,K) CPU
    top_winner: torch.Tensor,   # (C,K) CPU (1=pos, 0=neg)
    save_root: str,
    img_size: int,
    K=15,
    use_bf16=True,
    split_by_line=False,
    cmap_pos="jet",
    cmap_neg="jet",
):
    os.makedirs(save_root, exist_ok=True)
    device = next(model_q.parameters()).device

    autocast_kwargs = dict(device_type="cuda", enabled=(device.type == "cuda"))
    if use_bf16 and device.type == "cuda":
        autocast_kwargs["dtype"] = torch.bfloat16

    reader = TarOffsetReader()
    norm = SafeInstanceNormalize(threshold=0.01)

    for ch in tqdm(range(OUT_DIM), desc="Save per-channel (4 imgs/set)"):
        ch_dir = os.path.join(save_root, f"ch_{ch:03d}")
        os.makedirs(ch_dir, exist_ok=True)

        ridxs = top_refidx[ch, :K].tolist()

        raw_imgs_u16 = []
        xs = []
        metas = []  # (line, uid_hash, rank, win_str, posv, negv)

        for rank, ridx in enumerate(ridxs, start=1):
            ref = refs[int(ridx)]
            line = ref.line
            uid = f"{ref.tar_path}:{ref.prefix}"
            uid_hash = hashlib.md5(uid.encode("utf-8")).hexdigest()[:10]

            posv = float(top_posval[ch, rank-1].item())
            negv = float(top_negval[ch, rank-1].item())
            win_str = "pos" if int(top_winner[ch, rank-1].item()) == 1 else "neg"

            img_u16 = reader.read_uint16_rgb(ref, img_size)
            raw_imgs_u16.append(img_u16)

            x = (img_u16.astype(np.float32) / 65535.0)
            x = torch.from_numpy(x).permute(2, 0, 1)
            x = norm(x)
            xs.append(x)

            metas.append((line, uid_hash, rank, win_str, posv, negv))

        xb = torch.stack(xs, 0).to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)

        with torch.amp.autocast(**autocast_kwargs):
            fmap = model_q.encoder.trunk(xb)  # (B,C,h,w)

        fmap = fmap.float()
        act = fmap[:, ch, :, :]  # (B,h,w)
        act_up = F.interpolate(
            act.unsqueeze(1),
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False
        ).squeeze(1)  # (B,H,W)

        for i in range(K):
            line, uid_hash, rank, win_str, posv, negv = metas[i]

            out_dir = ch_dir
            if split_by_line:
                out_dir = os.path.join(ch_dir, line)
                os.makedirs(out_dir, exist_ok=True)

            base = (
                f"{line}_ch{ch:03d}_rank{rank:02d}_{uid_hash}"
                f"_win{win_str}_p{posv:.3f}_n{negv:.3f}"
            )

            # 01 orig (linear)
            orig_rgb8 = linear_uint16_to_uint8_rgb(raw_imgs_u16[i])
            Image.fromarray(orig_rgb8, mode="RGB").save(os.path.join(out_dir, f"{base}_01_orig.png"))

            # 02 bright (percentile stretch)
            bright_rgb8 = robust_uint16_to_uint8_rgb(raw_imgs_u16[i], p_lo=1.0, p_hi=99.0)
            Image.fromarray(bright_rgb8, mode="RGB").save(os.path.join(out_dir, f"{base}_02_bright.png"))

            # 03 neg/04 pos (min-max + colormap)
            act_hw = act_up[i].detach().cpu().numpy()
            neg_rgb8, pos_rgb8 = pos_neg_minmax_colormap(act_hw, cmap_pos=cmap_pos, cmap_neg=cmap_neg)

            Image.fromarray(neg_rgb8, mode="RGB").save(os.path.join(out_dir, f"{base}_03_neg_minmax.png"))
            Image.fromarray(pos_rgb8, mode="RGB").save(os.path.join(out_dir, f"{base}_04_pos_minmax.png"))

# =========================
# Main (Execution)
# =========================

# 1) Paths
SAVE_DIR   = "/content/drive/MyDrive/Final_paper/Model_MoCoXBM_PlateLP_LRRK2_L2 norm_hidden2048_resume_bias=True"
SHARD_ROOT = "/content/wds_shards"
WEIGHTS    = os.path.join(SAVE_DIR, "best_model.pt")

TRAIN_CSV = os.path.join(SAVE_DIR, "train_split.csv")
VAL_CSV   = os.path.join(SAVE_DIR, "val_split.csv")
TEST_CSV  = os.path.join(SAVE_DIR, "test_split.csv")

OUT_ROOT = os.path.join(SAVE_DIR, "featmap_top15_posneg_sets_reason")
INDEX_OUT = os.path.join(SAVE_DIR, "featmap_top15_abspeak_reason_index.pt")

# 2) args.json load
with open(os.path.join(SAVE_DIR, "args.json"), "r", encoding="utf-8") as f:
    argsj = json.load(f)

def parse_int_list(s: str, n: int):
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    vals = [int(p) for p in parts]
    assert len(vals) == n
    return tuple(vals)

blocks    = parse_int_list(argsj["blocks"], 4)
dilations = parse_int_list(argsj["dilations"], 4)

# 3) refs
refs = load_all_sample_refs(SHARD_ROOT)

# 4) split uids -> refidx mapping
uid_to_refidx_all = {f"{r.tar_path}:{r.prefix}": i for i, r in enumerate(refs)}

scan_uids = []
for p in [TRAIN_CSV, VAL_CSV, TEST_CSV]:
    if os.path.exists(p):
        scan_uids += load_uids_from_split_csv(p)

missing = [u for u in scan_uids if u not in uid_to_refidx_all]
if len(missing) > 0:
    raise RuntimeError(f"Some uids missing in shard_root. ex={missing[:5]}")

scan_refidx = [uid_to_refidx_all[u] for u in scan_uids]
print("Scan images:", len(scan_refidx))

# 5) dataset/loader
img_size = int(argsj.get("img_size", 128))
ds = ScanDataset(refs, scan_refidx, img_size=img_size)
loader = DataLoader(
    ds,
    batch_size=512,
    shuffle=False,
    num_workers=2,   # if environment is unstable, try 0
    pin_memory=True,
    collate_fn=collate_skip_none,
    drop_last=False
)

# 6) model + weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# NOTE: SupConMoCoModel must already be defined in your notebook/script.
model_q = SupConMoCoModel(
    embed_dim=int(argsj["embed_dim"]),
    blocks=blocks,
    dilations=dilations,
    refine_blocks=int(argsj["refine_blocks"]),
    ckpt_segments=int(argsj["ckpt_segments"]),
    proj_layers=int(argsj.get("proj_layers", 2)),
    proj_hidden=int(argsj.get("proj_hidden", 2048)),
    proj_bn=bool(argsj.get("proj_bn", False)),
    proj_dropout=float(argsj.get("proj_dropout", 0.0)),
)

load_weights_flex(model_q, WEIGHTS, device="cpu", strict=True)
model_q.eval().to(device).to(memory_format=torch.channels_last)

# 7) scan top-K (abs-peak + reason)
K_SETS = 40  # ✅ change this to control number of sets per channel (sets=K, total images per channel = K*4)

top_scores, top_refidx, top_posval, top_negval, top_winner = scan_topk_abspeak_with_reason(
    model_q, loader, device,
    K=K_SETS,
    use_bf16=True
)

torch.save(
    {"top_scores": top_scores, "top_refidx": top_refidx,
     "top_posval": top_posval, "top_negval": top_negval, "top_winner": top_winner},
    INDEX_OUT
)
print("Saved index ->", INDEX_OUT)

# 8) save (4 images per set)
save_topk_sets_per_channel_posneg(
    model_q,
    refs=refs,
    top_refidx=top_refidx,
    top_posval=top_posval,
    top_negval=top_negval,
    top_winner=top_winner,
    save_root=OUT_ROOT,
    img_size=img_size,
    K=K_SETS,
    use_bf16=True,
    split_by_line=False,  # True -> ch_xxx/Control_C4/...
    cmap_pos="jet",
    cmap_neg="jet",
)

print("Done. Output ->", OUT_ROOT)
