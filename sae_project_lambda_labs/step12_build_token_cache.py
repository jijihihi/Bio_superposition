import os, json, math, csv, random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from sae_project.step01_configs import get_args, resolve_paths
from sae_project.step02_logging_utils import get_logger, SUPERCLASS_MAP
from sae_project.step03_data_shards import load_all_sample_refs, build_uid_to_refidx
from sae_project.step04_data_bank import InMemoryTarBank, InMemorySixteenBitDataset, load_split_csv, seed_worker, collate_skip_none
#from sae_project.step02_logging_utils 
from sae_project.step05_model_encoder import SupMoCoModel, OUT_DIM, parse_int_list
#from sae_project.step06_sae_core import PointwiseTopKSAE  # your SAE module (must expose .usage_ema and forward(tok)->(recon_main, acts_main, recon_aux, acts_aux))
from sae_project.step09_train_sae import set_seed
#from collections import defaultdict


logger =  get_logger(__name__)

# ----------------------------
# small helpers
# ----------------------------
def _rot90_batch(x: torch.Tensor, k: int) -> torch.Tensor:
    # x: (B,C,H,W)
    if k == 0:
        return x
    return torch.rot90(x, k, dims=(2, 3))

@torch.no_grad()
def extract_tokens(
    encoder,
    x: torch.Tensor,
    which_layer: str,
    token_center: bool,
    token_l2_norm: bool,
) -> Tuple[torch.Tensor, int, int, int]:
    """
    returns tokens (B*HW, C=512) and (B,H,W)
    """
    fmap = encoder.forward_feature_maps(x, which=which_layer)  # (B,C,H,W)
    B, C, H, W = fmap.shape
    fmap = fmap.permute(0, 2, 3, 1).contiguous()              # (B,H,W,C)
    tokens = fmap.view(B * H * W, C)

    if token_center:
        tokens = tokens - tokens.mean(dim=0, keepdim=True)
    if token_l2_norm:
        tokens = F.normalize(tokens, dim=1)
    return tokens, B, H, W


def ensure_memmap(path: str, shape: Tuple[int, ...], dtype: np.dtype):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # numpy memmap via open_memmap creates .npy with header (portable)
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


# ----------------------------
# balanced image sampler by (class/line/plate)
# ----------------------------
def pick_balanced_images(
    uids: List[str],
    refs_by_uid: Dict[str, object],
    images_target: int,
    seed: int
) -> List[str]:
    """
    균형 기준:
      (superclass, line, plate) group마다 비슷한 image 수를 뽑는다.
    """
    rng = random.Random(seed)
    groups = defaultdict(list)

    for uid in uids:
        r = refs_by_uid[uid]
        sup = getattr(r, "superclass", SUPERCLASS_MAP.get(getattr(r, "line", "UNK"), "UNK"))
        line = getattr(r, "line", "UNK")
        plate = getattr(r, "plate", "UNK")
        groups[(sup, line, plate)].append(uid)

    # shuffle within each group
    for k in groups:
        rng.shuffle(groups[k])

    keys = list(groups.keys())
    rng.shuffle(keys)

    # round-robin take 1 from each group until target met
    out = []
    ptr = {k: 0 for k in keys}
    while len(out) < images_target:
        progressed = False
        for k in keys:
            i = ptr[k]
            if i < len(groups[k]):
                out.append(groups[k][i])
                ptr[k] = i + 1
                progressed = True
                if len(out) >= images_target:
                    break
        if not progressed:
            break

    rng.shuffle(out)
    return out


# ----------------------------
# main builder
# ----------------------------
@dataclass
class CacheSpec:
    split_name: str
    split_csv: str
    out_dir: str
    tokens_target: int
    tokens_per_image: int
    augment_rot90: bool


def build_cache_for_split(
    args,
    encoder,
    refs,
    uid_to_refidx,
    spec: CacheSpec,
    device: torch.device
):
    if not os.path.exists(spec.split_csv):
        logger.info(f"[cache] missing {spec.split_name} split: {spec.split_csv} (skip)")
        return

    os.makedirs(spec.out_dir, exist_ok=True)

    # load uids in split
    uids_all = load_split_csv(spec.split_csv)

    # build refs_by_uid for balancing
    # uid_to_refidx maps uid-> index in refs
    refs_by_uid = {u: refs[uid_to_refidx[u]] for u in uids_all}

    # decide how many images to draw
    tpi = int(spec.tokens_per_image)
    if tpi <= 0:
        raise ValueError("For token-cache, tokens_per_image must be > 0 (avoid full 4096/image explosion).")
    images_target = int(math.ceil(spec.tokens_target / tpi))

    # pick balanced images
    picked_uids = pick_balanced_images(
        uids=uids_all,
        refs_by_uid=refs_by_uid,
        images_target=images_target,
        seed=int(args.seed) + (0 if spec.split_name == "train" else (1 if spec.split_name == "val" else 2))
    )
    if not picked_uids:
        raise RuntimeError(f"[cache] picked_uids empty for split={spec.split_name}")

    # map to refidx list
    refidx = [uid_to_refidx[u] for u in picked_uids]

    # build RAM bank + loader (no augment here; we apply rot90 manually for caching)
    bank = InMemoryTarBank(refs, refidx, args.img_size)
    ib = list(range(len(refidx)))
    ds = InMemorySixteenBitDataset(bank, ib, args.img_size, two_crops=False, augment=False)

    pin = torch.cuda.is_available()
    loader = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(getattr(args, "num_workers", 0)),
        pin_memory=pin,
        worker_init_fn=seed_worker,
        collate_fn=collate_skip_none,
        drop_last=False,
    )

    # allocate memmaps with exact size = images_target * tpi
    N = int(images_target * tpi)
    tok_path = os.path.join(spec.out_dir, f"{spec.split_name}_tokens.f16.npy")
    imgid_path = os.path.join(spec.out_dir, f"{spec.split_name}_imgid.i32.npy")
    y_path = os.path.join(spec.out_dir, f"{spec.split_name}_label.i64.npy")

    tok_mm = ensure_memmap(tok_path, shape=(N, OUT_DIM), dtype=np.float16)
    imgid_mm = ensure_memmap(imgid_path, shape=(N,), dtype=np.int32)
    y_mm = ensure_memmap(y_path, shape=(N,), dtype=np.int64)

    # meta
    meta = {
        "split": spec.split_name,
        "tokens_target": int(spec.tokens_target),
        "tokens_per_image": int(tpi),
        "images_target": int(images_target),
        "which_layer": str(args.which_layer),
        "token_center_for_cache": bool(getattr(args, "token_center", True)),
        "token_l2_norm_for_cache": bool(getattr(args, "token_l2_norm", True)),
        "augment_rot90": bool(spec.augment_rot90),
        "img_size": int(args.img_size),
        "model_state_path": str(args.model_state_path),
        "note": "This cache stores sampled tokens per image and imgid mapping for pooling.",
    }
    with open(os.path.join(spec.out_dir, f"{spec.split_name}_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # build mapping: local image index -> label
    # Here "imgid" is 0..len(picked_uids)-1, representing the selected images in this cache.
    uid_to_imgid = {u: i for i, u in enumerate(picked_uids)}

    # caching loop
    encoder.eval()
    autocast_kwargs = dict(device_type="cuda", enabled=torch.cuda.is_available(), dtype=torch.bfloat16)

    write_ptr = 0
    rng = random.Random(int(args.seed) + 777)

    pbar = tqdm(loader, desc=f"[cache] {spec.split_name}", leave=True)
    for batch in pbar:
        if batch is None:
            continue
        x_cpu, y_cpu, plate, line, uid = batch
        if y_cpu.numel() < 1:
            continue

        B = x_cpu.size(0)
        # move
        x = x_cpu.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)

        # rot90 aug at cache-time
        if spec.augment_rot90:
            k = rng.randint(0, 3)
            x = _rot90_batch(x, k)

        # tokens
        with torch.no_grad():
            with torch.amp.autocast(**autocast_kwargs):
                tokens, B2, H, W = extract_tokens(
                    encoder,
                    x,
                    which_layer=str(args.which_layer),
                    token_center=bool(getattr(args, "token_center", True)),
                    token_l2_norm=bool(getattr(args, "token_l2_norm", True)),
                )
        assert B2 == B
        hw = H * W

        # sample tpi tokens PER IMAGE
        # tokens are ordered by image blocks
        # We'll sample indices per image on GPU then write to memmap.
        for bi in range(B):
            if write_ptr >= N:
                break

            # uid is list of strings length B
            u = uid[bi]
            imgid = uid_to_imgid.get(u, None)
            if imgid is None:
                continue

            base = bi * hw
            if tpi < hw:
                idx = torch.randperm(hw, device=tokens.device)[:tpi]
                sel = tokens[base + idx]  # (tpi,512)
            else:
                sel = tokens[base:base+hw][:tpi]

            m = sel.size(0)
            if m == 0:
                continue

            # clip if overflow
            if write_ptr + m > N:
                m = N - write_ptr
                sel = sel[:m]

            tok_mm[write_ptr:write_ptr+m] = sel.detach().float().cpu().numpy().astype(np.float16)
            imgid_mm[write_ptr:write_ptr+m] = np.int32(imgid)
            y_mm[write_ptr:write_ptr+m] = np.int64(int(y_cpu[bi].item()))

            write_ptr += m

        pbar.set_postfix({"written": f"{write_ptr}/{N}"})
        if write_ptr >= N:
            break

    # flush
    tok_mm.flush(); imgid_mm.flush(); y_mm.flush()
    logger.info(f"[cache] {spec.split_name} done. wrote {write_ptr}/{N} tokens -> {spec.out_dir}")


def main():
    args = resolve_paths(get_args())
    set_seed(args.seed)

    # ---- IMPORTANT: tune these in code (avoid argparse bloat) ----
    CACHE_DIR = os.path.join(args.save_dir, "token_cache")
    TOKENS_TARGET_TRAIN = int(getattr(args, "cache_tokens_train", 100_000_000))  # 1e8 default
    TOKENS_TARGET_VAL   = int(getattr(args, "cache_tokens_val",   10_000_000))   # 1e7 default
    TOKENS_TARGET_TEST  = int(getattr(args, "cache_tokens_test",  10_000_000))   # 1e7 default
    TOKENS_PER_IMAGE = int(getattr(args, "cache_tpi", 512))  # sampled per image, MUST be >0
    AUG_ROT90 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[cache] device={device}")

    # load refs
    refs = load_all_sample_refs(args.shard_root)
    uid_to_refidx = build_uid_to_refidx(refs)

    # load encoder
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
    sd = torch.load(args.model_state_path, map_location="cpu")
    from sae_project.step05_model_encoder import robust_load_state_dict
    robust_load_state_dict(model, sd, strict=True)
    encoder = model.encoder.to(device).eval().to(memory_format=torch.channels_last)

    # splits
    train_csv = os.path.join(args.save_dir, "train_split.csv")
    val_csv   = os.path.join(args.save_dir, "val_split.csv")
    test_csv  = os.path.join(args.save_dir, "test_split.csv")

    # build per split
    specs = [
        CacheSpec("train", train_csv, CACHE_DIR, TOKENS_TARGET_TRAIN, TOKENS_PER_IMAGE, augment_rot90=AUG_ROT90),
        CacheSpec("val",   val_csv,   CACHE_DIR, TOKENS_TARGET_VAL,   TOKENS_PER_IMAGE, augment_rot90=AUG_ROT90),
        CacheSpec("test",  test_csv,  CACHE_DIR, TOKENS_TARGET_TEST,  TOKENS_PER_IMAGE, augment_rot90=False),  # test는 보통 aug 끔
    ]

    for sp in specs:
        build_cache_for_split(args, encoder, refs, uid_to_refidx, sp, device=device)

    logger.info(f"[cache] all done -> {CACHE_DIR}")


if __name__ == "__main__":
    main()
