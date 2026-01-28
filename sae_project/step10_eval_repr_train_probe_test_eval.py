import os, glob, json, math, csv, random
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# ---- reuse your project modules ----
from sae_project.step01_configs import get_args, resolve_paths
from sae_project.step02_logging_utils import get_logger, SUPERCLASS_MAP
from sae_project.step03_data_shards import load_all_sample_refs, build_uid_to_refidx
from sae_project.step04_data_bank import InMemoryTarBank, InMemorySixteenBitDataset, load_split_csv, seed_worker, collate_skip_none, StrictPlateBalancedBatchSamplerOnBank
#from sae_project.step02_logging_utils 
from sae_project.step05_model_encoder import SupConMoCoModel, OUT_DIM, parse_int_list, renorm_unit_per_out_channel_
from sae_project.step06_sae_core import PointwiseTopKSAE  # your SAE module (must expose .usage_ema and forward(tok)->(recon_main, acts_main, recon_aux, acts_aux))
from sae_project.step07_train_sae import set_seed
from collections import defaultdict


logger = get_logger(__name__)

# ----------------------------
# balanced image subset pick (★ 클래스별 강제 균형)
# ----------------------------
SUPERCLASS_ORDER = ["Control", "SNCA", "GBA", "LRRK2"]  # 4개 클래스

def pick_balanced_uids(
    uids: List[str],
    refs_by_uid: Dict[str, object],
    n_target: int,
    seed: int
) -> List[str]:
    """
    ★ 클래스별 강제 균형 샘플링
    - n_target을 4로 나눈 값만큼 각 클래스에서 뽑음
    - 각 클래스 내에서 line별, plate별 균등 분배
    
    예: n_target=4000 → 각 클래스에서 1000개씩
    """
    rng = random.Random(seed)
    num_classes = len(SUPERCLASS_ORDER)
    per_class = n_target // num_classes
    remainder = n_target % num_classes
    
    # 1. 클래스별로 그룹핑
    class_groups = {sup: defaultdict(list) for sup in SUPERCLASS_ORDER}
    for u in uids:
        r = refs_by_uid[u]
        sup = getattr(r, "superclass", SUPERCLASS_MAP.get(getattr(r, "line", "UNK"), "UNK"))
        if sup not in SUPERCLASS_ORDER:
            continue
        line = getattr(r, "line", "UNK")
        plate = getattr(r, "plate", "UNK")
        class_groups[sup][(line, plate)].append(u)
    
    # 2. 각 클래스별로 할당량 계산
    class_quota = {}
    for i, sup in enumerate(SUPERCLASS_ORDER):
        class_quota[sup] = per_class + (1 if i < remainder else 0)
    
    # 3. 각 클래스에서 line/plate 균등하게 뽑기
    out = []
    for sup in SUPERCLASS_ORDER:
        quota = class_quota[sup]
        lp_groups = class_groups[sup]
        keys = list(lp_groups.keys())
        
        if not keys:
            logger.warning(f"[pick_balanced_uids] No samples for class {sup}!")
            continue
        
        # 각 line/plate 그룹 셔플
        rng.shuffle(keys)
        for k in keys:
            rng.shuffle(lp_groups[k])
        
        # 가용 샘플 수 확인
        total_available = sum(len(lp_groups[k]) for k in keys)
        actual_quota = min(quota, total_available)
        
        if actual_quota < quota:
            logger.warning(f"[pick_balanced_uids] Class {sup}: requested {quota}, available {total_available}")
        
        # line/plate별 균등 분배
        num_groups = len(keys)
        per_group = actual_quota // num_groups
        remainder_group = actual_quota % num_groups
        
        group_quotas = {}
        for i, k in enumerate(keys):
            group_quotas[k] = per_group + (1 if i < remainder_group else 0)
        
        # 각 그룹에서 할당량만큼 뽑기
        class_samples = []
        for k in keys:
            available = lp_groups[k]
            take = min(group_quotas[k], len(available))
            class_samples.extend(available[:take])
        
        # 부족분 보충 (다른 그룹에서 더 뽑기)
        if len(class_samples) < actual_quota:
            all_remaining = []
            for k in keys:
                taken = min(group_quotas[k], len(lp_groups[k]))
                all_remaining.extend(lp_groups[k][taken:])
            rng.shuffle(all_remaining)
            need = actual_quota - len(class_samples)
            class_samples.extend(all_remaining[:need])
        
        out.extend(class_samples[:actual_quota])
    
    rng.shuffle(out)
    
    # 최종 분포 확인 로그
    final_dist = defaultdict(int)
    for u in out:
        r = refs_by_uid[u]
        sup = getattr(r, "superclass", SUPERCLASS_MAP.get(getattr(r, "line", "UNK"), "UNK"))
        final_dist[sup] += 1
    logger.info(f"[pick_balanced_uids] Final distribution: {dict(final_dist)} (total={len(out)})")
    
    return out[:n_target]


def make_loader_for_uid_subset(
    args, 
    refs, 
    uid_to_refidx, 
    picked_uids: List[str], 
    batch_size: int,
    use_balanced_sampler: bool = False,
    seed: int = 42,
) -> Tuple[DataLoader, InMemoryTarBank]:
    """
    UID subset으로부터 DataLoader 생성.
    
    Args:
        use_balanced_sampler: True이면 StrictPlateBalancedBatchSamplerOnBank 사용
        seed: balanced sampler용 seed
    
    Returns:
        loader: DataLoader
        bank: InMemoryTarBank (나중에 class distribution 확인 등에 사용)
    """
    refidx = [uid_to_refidx[u] for u in picked_uids]
    bank = InMemoryTarBank(refs, refidx, args.img_size)
    ib = list(range(len(refidx)))
    ds = InMemorySixteenBitDataset(bank, ib, args.img_size, augment=False)

    pin = torch.cuda.is_available()
    
    if use_balanced_sampler:
        # StrictPlateBalancedBatchSamplerOnBank 사용
        sampler = StrictPlateBalancedBatchSamplerOnBank(bank, batch_size=batch_size, seed=seed)
        loader = DataLoader(
            ds,
            batch_sampler=sampler,
            num_workers=int(getattr(args, "num_workers", 0)),
            pin_memory=pin,
            worker_init_fn=seed_worker,
            collate_fn=collate_skip_none,
        )
    else:
        # 기본 sequential loader
        loader = DataLoader(
            ds,
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=int(getattr(args, "num_workers", 0)),
            pin_memory=pin,
            worker_init_fn=seed_worker,
            collate_fn=collate_skip_none,
            drop_last=False,
        )
    
    return loader, bank


def count_class_distribution_in_bank(bank: InMemoryTarBank) -> Dict[str, int]:
    """Bank 내의 클래스별 샘플 수 계산"""
    dist = defaultdict(int)
    for j in range(len(bank.images)):
        if bank.images[j] is None:
            continue
        line = bank.lines[j]
        sup = SUPERCLASS_MAP.get(line, line)
        dist[sup] += 1
    return dict(dist)


# ----------------------------
# SAE ckpt loader (k sweep)
# ----------------------------
def find_sae_ckpt_for_k(sae_root: str, k: int) -> str:
    """
    정확한 k 값 매칭 (k=5가 k=15, k=25와 혼동되지 않도록 regex 사용)
    """
    import re
    # 정확한 k 매칭을 위한 regex 패턴: _k{k}. 또는 _k{k}_ 또는 k={k} 형태만 매칭
    exact_k_pattern = re.compile(rf"(?:_k{k}[._]|_k{k}$|k={k}|\bk{k}\b)")
    
    cands = []
    cands += glob.glob(os.path.join(sae_root, f"k={k}", "*.pt"))
    cands += glob.glob(os.path.join(sae_root, "*", "*.pt"))
    cands += glob.glob(os.path.join(sae_root, "*.pt"))
    
    # 정확한 k 값 매칭 필터링
    filtered = []
    for p in cands:
        basename = os.path.basename(p)
        dirname = os.path.basename(os.path.dirname(p))
        full_check = f"{dirname}/{basename}"
        if exact_k_pattern.search(full_check):
            filtered.append(p)
    
    if not filtered:
        raise FileNotFoundError(f"No SAE checkpoint found for k={k} under {sae_root}")
    
    filtered = sorted(list(set(filtered)))
    
    # 우선순위에 따라 선택
    for pref in ["refine_out", "best", "ckpt", "resume", "last"]:
        for p in filtered:
            if pref in os.path.basename(p).lower():
                return p
                
    return filtered[-1]


def load_sae_from_ckpt(ckpt_path: str, device: torch.device) -> Tuple[PointwiseTopKSAE, Dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    a = ckpt.get("args", {})
    d_in = int(a.get("d_in", OUT_DIM))
    d_sae = int(a.get("d_sae", 4096))
    k = int(a.get("k", 10))
    aux_k = int(a.get("aux_k", 0))
    aux_coeff = float(a.get("aux_coeff", 0.0))
    
    # 로드한 checkpoint에서 k 값 확인 로그
    logger.info(f"[load_sae] ckpt={os.path.basename(ckpt_path)} -> d_in={d_in}, d_sae={d_sae}, k={k}")

    sae = PointwiseTopKSAE(d_in=d_in, d_sae=d_sae, k=k, init_scale=0.02)
    sae.load_state_dict(ckpt["sae"], strict=True)
    sae.to(device).eval()
    return sae, a


# ----------------------------
# repr extraction (full tokens)
# ----------------------------
@torch.no_grad()
def pooled_sae_repr_fulltokens(
    encoder,
    sae: PointwiseTopKSAE,
    loader: DataLoader,
    device: torch.device,
    which_layer: str,
    token_batch: int,
    sae_input_center: bool,
    sae_input_l2_norm: bool,
    strength_weight: bool,
    dead_threshold: float = 1e-6,  # dead neuron 제외 threshold
    exclude_dead: bool = True,     # dead neuron 제외 여부
) -> Tuple[np.ndarray, np.ndarray, int]:  # 반환값에 alive_count 추가
    encoder.eval()
    sae.eval()

    autocast_kwargs = dict(device_type="cuda", enabled=torch.cuda.is_available(), dtype=torch.bfloat16)

    X_list, y_list = [], []
    for batch in tqdm(loader, desc="Repr SAE(full)", leave=False):
        if batch is None:
            continue
        x_cpu, y_cpu, *_ = batch
        if y_cpu.numel() < 1:
            continue

        B = x_cpu.size(0)
        x = x_cpu.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)

        with torch.amp.autocast(**autocast_kwargs):
            fmap = encoder.forward_feature_maps(x, which=which_layer)  # (B,512,H,W)

        B2, C, H, W = fmap.shape
        hw = H * W
        tokens = fmap.permute(0, 2, 3, 1).contiguous().view(B * hw, C).float()
        tokens_raw = tokens

        if sae_input_center:
            tokens_raw = tokens_raw - tokens_raw.mean(dim=0, keepdim=True)

        tok_norm = None
        if strength_weight:
            tok_norm = tokens_raw.norm(dim=1, keepdim=True).clamp_min(1e-12)

        sae_in = tokens_raw
        if sae_input_l2_norm:
            sae_in = F.normalize(sae_in, dim=1)

        d_sae = int(sae.d_sae)
        sum_acts = torch.zeros(B, d_sae, device=device, dtype=torch.float32)
        cnt = torch.zeros(B, device=device, dtype=torch.float32)

        Tb = int(token_batch)
        n = sae_in.size(0)

        for s in range(0, n, Tb):
            tok = sae_in[s:s+Tb]
            with torch.amp.autocast(**autocast_kwargs):
                recon_main, acts_main, recon_aux, acts_aux = sae(tok)

            acts = acts_main.float()
            if strength_weight:
                acts = acts * tok_norm[s:s+Tb]

            m = acts.size(0)
            idx = torch.arange(s, s+m, device=device, dtype=torch.long)
            img = (idx // hw).clamp(max=B-1)

            sum_acts.index_add_(0, img, acts)
            cnt.index_add_(0, img, torch.ones(m, device=device))

        pooled = sum_acts / cnt.clamp_min(1.0).unsqueeze(1)
        X_list.append(pooled.detach().cpu().numpy().astype(np.float32))
        y_list.append(y_cpu.numpy().astype(np.int64))

    X = np.concatenate(X_list, axis=0) if X_list else np.zeros((0, int(sae.d_sae)), np.float32)
    y = np.concatenate(y_list, axis=0) if y_list else np.zeros((0,), np.int64)
    
    # dead neuron 제외 (exclude_dead=True인 경우)
    alive_count = int(sae.d_sae)
    if exclude_dead:
        # usage_ema가 threshold 이상인 neuron만 alive
        alive_mask = sae.usage_ema.cpu().numpy() >= dead_threshold
        alive_count = int(alive_mask.sum())
        dead_count = int(sae.d_sae) - alive_count
        
        logger.info(f"[pooled_sae_repr] d_sae={sae.d_sae}, alive={alive_count}, dead={dead_count}, k={sae.k}")
        
        if alive_count > 0:
            X = X[:, alive_mask]  # (N, alive_count)
        else:
            logger.warning(f"[pooled_sae_repr] All neurons are dead! Using full d_sae.")
            alive_count = int(sae.d_sae)
    
    return X, y, alive_count


@torch.no_grad()
def pooled_randproj_repr_fulltokens(
    encoder,
    loader: DataLoader,
    device: torch.device,
    which_layer: str,
    token_batch: int,
    R: torch.Tensor,          # (512, D)
    token_center: bool,
    use_relu: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    encoder.eval()
    autocast_kwargs = dict(device_type="cuda", enabled=torch.cuda.is_available(), dtype=torch.bfloat16)

    D = R.size(1)
    X_list, y_list = [], []

    for batch in tqdm(loader, desc="Repr RandProj(full)", leave=False):
        if batch is None:
            continue
        x_cpu, y_cpu, *_ = batch
        if y_cpu.numel() < 1:
            continue

        B = x_cpu.size(0)
        x = x_cpu.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)

        with torch.amp.autocast(**autocast_kwargs):
            fmap = encoder.forward_feature_maps(x, which=which_layer)

        B2, C, H, W = fmap.shape
        hw = H * W
        tokens = fmap.permute(0,2,3,1).contiguous().view(B*hw, C).float()

        if token_center:
            tokens = tokens - tokens.mean(dim=0, keepdim=True)

        sum_z = torch.zeros(B, D, device=device, dtype=torch.float32)
        cnt = torch.zeros(B, device=device, dtype=torch.float32)

        Tb = int(token_batch)
        n = tokens.size(0)

        for s in range(0, n, Tb):
            tok = tokens[s:s+Tb]
            z = tok @ R
            if use_relu:
                z = F.relu(z)

            m = z.size(0)
            idx = torch.arange(s, s+m, device=device, dtype=torch.long)
            img = (idx // hw).clamp(max=B-1)

            sum_z.index_add_(0, img, z.float())
            cnt.index_add_(0, img, torch.ones(m, device=device))

        pooled = sum_z / cnt.clamp_min(1.0).unsqueeze(1)
        X_list.append(pooled.detach().cpu().numpy().astype(np.float32))
        y_list.append(y_cpu.numpy().astype(np.int64))

    X = np.concatenate(X_list, axis=0) if X_list else np.zeros((0, D), np.float32)
    y = np.concatenate(y_list, axis=0) if y_list else np.zeros((0,), np.int64)
    return X, y


@torch.no_grad()
def gap_repr(
    encoder, 
    loader: DataLoader, 
    device: torch.device, 
    which_layer: str,
    l2_normalize: bool = True,  # ★ 추가: SupCon 학습 시와 동일하게 L2 정규화
) -> Tuple[np.ndarray, np.ndarray]:
    """
    특정 레이어의 feature map에서 GAP를 추출 (512차원)
    l2_normalize=True이면 SupCon 학습 시와 동일하게 L2 정규화 적용
    """
    encoder.eval()
    autocast_kwargs = dict(device_type="cuda", enabled=torch.cuda.is_available(), dtype=torch.bfloat16)

    X_list, y_list = [], []
    for batch in tqdm(loader, desc=f"GAP({which_layer})", leave=False):
        if batch is None:
            continue
        x_cpu, y_cpu, *_ = batch
        if y_cpu.numel() < 1:
            continue

        x = x_cpu.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        with torch.amp.autocast(**autocast_kwargs):
            fmap = encoder.forward_feature_maps(x, which=which_layer)  # (B, 512, H, W)
        
        # Global Average Pooling
        gap = fmap.mean(dim=[2, 3])  # (B, 512)
        
        # ★ L2 정규화 (SupCon 학습 시와 동일하게!)
        if l2_normalize:
            gap = F.normalize(gap, dim=1)
        
        X_list.append(gap.float().detach().cpu().numpy().astype(np.float32))
        y_list.append(y_cpu.numpy().astype(np.int64))

    X = np.concatenate(X_list, axis=0) if X_list else np.zeros((0, OUT_DIM), np.float32)
    y = np.concatenate(y_list, axis=0) if y_list else np.zeros((0,), np.int64)
    return X, y


@torch.no_grad()
def gap_randproj_repr(
    encoder, 
    loader: DataLoader, 
    device: torch.device, 
    which_layer: str,
    R: torch.Tensor,  # (512, target_dim)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Baseline 1: GAP @ Random Matrix
    - GAP512에서 직접 random projection
    - 차원 증가 효과만 분리하기 위한 baseline
    """
    encoder.eval()
    autocast_kwargs = dict(device_type="cuda", enabled=torch.cuda.is_available(), dtype=torch.bfloat16)
    
    target_dim = R.size(1)
    X_list, y_list = [], []
    
    for batch in tqdm(loader, desc=f"GAP@Rand({which_layer})", leave=False):
        if batch is None:
            continue
        x_cpu, y_cpu, *_ = batch
        if y_cpu.numel() < 1:
            continue

        x = x_cpu.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        with torch.amp.autocast(**autocast_kwargs):
            fmap = encoder.forward_feature_maps(x, which=which_layer)  # (B, 512, H, W)
        
        # GAP -> Random Projection
        gap = fmap.mean(dim=[2, 3]).float()  # (B, 512)
        projected = gap @ R  # (B, target_dim)
        
        X_list.append(projected.detach().cpu().numpy().astype(np.float32))
        y_list.append(y_cpu.numpy().astype(np.int64))

    X = np.concatenate(X_list, axis=0) if X_list else np.zeros((0, target_dim), np.float32)
    y = np.concatenate(y_list, axis=0) if y_list else np.zeros((0,), np.int64)
    return X, y


@torch.no_grad()
def pooled_randproj_topk_repr(
    encoder,
    loader: DataLoader,
    device: torch.device,
    which_layer: str,
    token_batch: int,
    R: torch.Tensor,          # (512, target_dim)
    k: int,                   # Top-K sparsity (same as SAE)
    token_center: bool = True,
    token_l2_norm: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Baseline 2: Token @ Random + Top-K
    - SAE와 동일한 처리 방식 (token-level projection + top-k sparsity + mean pool)
    - SAE의 학습된 feature가 random보다 나은지 테스트
    """
    encoder.eval()
    autocast_kwargs = dict(device_type="cuda", enabled=torch.cuda.is_available(), dtype=torch.bfloat16)

    target_dim = R.size(1)
    X_list, y_list = [], []

    for batch in tqdm(loader, desc=f"Tok@Rand+TopK({which_layer})", leave=False):
        if batch is None:
            continue
        x_cpu, y_cpu, *_ = batch
        if y_cpu.numel() < 1:
            continue

        B = x_cpu.size(0)
        x = x_cpu.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)

        with torch.amp.autocast(**autocast_kwargs):
            fmap = encoder.forward_feature_maps(x, which=which_layer)

        B2, C, H, W = fmap.shape
        hw = H * W
        tokens = fmap.permute(0, 2, 3, 1).contiguous().view(B * hw, C).float()

        # SAE와 동일한 전처리
        if token_center:
            tokens = tokens - tokens.mean(dim=0, keepdim=True)
        if token_l2_norm:
            tokens = F.normalize(tokens, dim=1)

        sum_acts = torch.zeros(B, target_dim, device=device, dtype=torch.float32)
        cnt = torch.zeros(B, device=device, dtype=torch.float32)

        Tb = int(token_batch)
        n = tokens.size(0)

        for s in range(0, n, Tb):
            tok = tokens[s:s+Tb]
            
            # Random projection
            pre = tok @ R  # (chunk, target_dim)
            
            # Top-K sparsity (SAE와 동일)
            # |pre|가 큰 k개만 유지
            topk_vals, topk_idx = torch.topk(pre.abs(), k=k, dim=1, largest=True, sorted=False)
            acts = torch.zeros_like(pre)
            acts.scatter_(1, topk_idx, pre.gather(1, topk_idx))

            m = acts.size(0)
            idx = torch.arange(s, s+m, device=device, dtype=torch.long)
            img = (idx // hw).clamp(max=B-1)

            sum_acts.index_add_(0, img, acts)
            cnt.index_add_(0, img, torch.ones(m, device=device))

        pooled = sum_acts / cnt.clamp_min(1.0).unsqueeze(1)
        X_list.append(pooled.detach().cpu().numpy().astype(np.float32))
        y_list.append(y_cpu.numpy().astype(np.int64))

    X = np.concatenate(X_list, axis=0) if X_list else np.zeros((0, target_dim), np.float32)
    y = np.concatenate(y_list, axis=0) if y_list else np.zeros((0,), np.int64)
    return X, y


# ----------------------------
# probe (train on train_repr, eval on test_repr)
# ----------------------------
class LinearProbe(nn.Module):
    def __init__(self, d_in: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(d_in, num_classes, bias=False)
    def forward(self, x):
        return self.fc(x)

def compute_repr_stats(X: np.ndarray, y: np.ndarray, name: str = "repr") -> Dict:
    """
    Representation 통계 계산: mean, std, class-wise 분리도 등
    """
    stats = {}
    
    # 전체 통계
    stats["shape"] = X.shape
    stats["mean"] = float(np.mean(X))
    stats["std"] = float(np.std(X))
    stats["min"] = float(np.min(X))
    stats["max"] = float(np.max(X))
    stats["n_zeros"] = int(np.sum(X == 0))
    stats["zero_ratio"] = float(np.sum(X == 0) / X.size)
    
    # Per-feature 통계
    feat_mean = np.mean(X, axis=0)
    feat_std = np.std(X, axis=0)
    stats["feat_mean_mean"] = float(np.mean(feat_mean))
    stats["feat_mean_std"] = float(np.std(feat_mean))
    stats["feat_std_mean"] = float(np.mean(feat_std))
    stats["n_zero_features"] = int(np.sum(feat_std < 1e-10))  # 상수 feature 개수
    
    # Class-wise 통계
    unique_classes = np.unique(y)
    class_means = []
    for c in unique_classes:
        mask = y == c
        if np.sum(mask) > 0:
            class_mean = np.mean(X[mask], axis=0)
            class_means.append(class_mean)
    
    if len(class_means) >= 2:
        class_means = np.array(class_means)
        # Inter-class distance (클래스 평균 간 거리)
        from scipy.spatial.distance import pdist
        inter_class_dists = pdist(class_means, metric='euclidean')
        stats["inter_class_dist_mean"] = float(np.mean(inter_class_dists))
        stats["inter_class_dist_min"] = float(np.min(inter_class_dists))
        stats["inter_class_dist_max"] = float(np.max(inter_class_dists))
        
        # Intra-class variance (클래스 내 분산)
        intra_vars = []
        for c in unique_classes:
            mask = y == c
            if np.sum(mask) > 1:
                intra_vars.append(float(np.mean(np.var(X[mask], axis=0))))
        if intra_vars:
            stats["intra_class_var_mean"] = float(np.mean(intra_vars))
        
        # Fisher's criterion (클래스 분리도 지표)
        # = inter-class variance / intra-class variance
        between_class_var = np.var(class_means, axis=0)
        within_class_var = np.mean([np.var(X[y==c], axis=0) for c in unique_classes if np.sum(y==c) > 1], axis=0)
        fisher_ratio = np.mean(between_class_var / (within_class_var + 1e-10))
        stats["fisher_ratio"] = float(fisher_ratio)
    
    return stats


def print_repr_stats(stats: Dict, name: str = "repr"):
    """Representation 통계 출력"""
    print(f"\n{'='*60}")
    print(f"[{name}] Representation Statistics")
    print(f"{'='*60}")
    print(f"  Shape: {stats['shape']}")
    print(f"  Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}")
    print(f"  Min: {stats['min']:.6f}, Max: {stats['max']:.6f}")
    print(f"  Zero ratio: {stats['zero_ratio']:.4f} ({stats['n_zeros']} / {stats['shape'][0]*stats['shape'][1]})")
    print(f"  Zero-variance features: {stats['n_zero_features']}")
    
    if 'inter_class_dist_mean' in stats:
        print(f"\n  Class Separation Metrics:")
        print(f"    Inter-class dist (mean/min/max): {stats['inter_class_dist_mean']:.4f} / {stats['inter_class_dist_min']:.4f} / {stats['inter_class_dist_max']:.4f}")
        if 'intra_class_var_mean' in stats:
            print(f"    Intra-class variance (mean): {stats['intra_class_var_mean']:.6f}")
        print(f"    Fisher ratio: {stats['fisher_ratio']:.4f}")
    print(f"{'='*60}\n")


def verify_train_test_separation(
    train_uids: List[str], 
    test_uids: List[str], 
    refs_by_uid: Dict[str, object]
) -> Dict:
    """Train/Test 분리 및 밸런스 검증"""
    result = {}
    
    # 1. 중복 확인
    train_set = set(train_uids)
    test_set = set(test_uids)
    overlap = train_set & test_set
    result["overlap_count"] = len(overlap)
    result["overlap_uids"] = list(overlap)[:10]  # 최대 10개만
    
    if overlap:
        print(f"[WARNING] Train/Test overlap detected: {len(overlap)} samples!")
        print(f"  First few: {list(overlap)[:5]}")
    else:
        print(f"[OK] No overlap between train ({len(train_uids)}) and test ({len(test_uids)})")
    
    # 2. 클래스별 분포
    def get_distribution(uids):
        dist = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        class_count = defaultdict(int)
        for u in uids:
            r = refs_by_uid.get(u)
            if r:
                sup = getattr(r, "superclass", SUPERCLASS_MAP.get(getattr(r, "line", "UNK"), "UNK"))
                line = getattr(r, "line", "UNK")
                plate = getattr(r, "plate", "UNK")
                dist[sup][line][plate] += 1
                class_count[sup] += 1
        return dist, class_count
    
    train_dist, train_class = get_distribution(train_uids)
    test_dist, test_class = get_distribution(test_uids)
    
    result["train_class_distribution"] = dict(train_class)
    result["test_class_distribution"] = dict(test_class)
    
    print(f"\n[Class Distribution]")
    print(f"  Train: {dict(train_class)}")
    print(f"  Test:  {dict(test_class)}")
    
    # 3. 클래스 밸런스 검사
    train_counts = list(train_class.values())
    test_counts = list(test_class.values())
    
    if train_counts:
        train_imbalance = max(train_counts) / (min(train_counts) + 1e-10)
        result["train_imbalance_ratio"] = train_imbalance
        print(f"  Train imbalance ratio (max/min): {train_imbalance:.2f}")
    
    if test_counts:
        test_imbalance = max(test_counts) / (min(test_counts) + 1e-10)
        result["test_imbalance_ratio"] = test_imbalance
        print(f"  Test imbalance ratio (max/min): {test_imbalance:.2f}")
    
    return result


def train_probe_and_eval_test(
    Xtr: np.ndarray, ytr: np.ndarray,
    Xte: np.ndarray, yte: np.ndarray,
    num_classes: int,
    device: torch.device,
    epochs: int = 5,
    lr: float = 0.1,
    wd: float = 0.0,
    momentum: float = 0.9,
    batch_size: int = 16384,
    method_name: str = "probe",
    verbose: bool = True,
    log_every: int = 1,
    balanced_training: bool = True,  # ★ 클래스 균형 샘플링 사용
    normalize_repr: bool = True,     # ★ 추가: representation 정규화
) -> Tuple[float, Dict]:
    """
    Linear probe train/test with comprehensive debugging.
    
    Args:
        balanced_training: True이면 각 epoch에서 클래스별로 동일한 수의 샘플 사용
        normalize_repr: True이면 representation을 Z-score 정규화
    
    Returns:
        test_acc: float
        debug_info: Dict containing training history and statistics
    """
    original_dim = Xtr.shape[1]
    
    # ★ Zero-variance features 제거 (매우 중요!)
    # Train set 기준으로 variance 계산
    feat_std = Xtr.std(axis=0)
    var_threshold = 1e-8
    alive_mask = feat_std >= var_threshold  # variance가 있는 feature만 선택
    n_alive = int(np.sum(alive_mask))
    n_removed = original_dim - n_alive
    
    if verbose:
        print(f"\n  [FEATURE FILTERING]")
        print(f"    Original dim: {original_dim}") # original_dim이 SAE dictionary 개수 - dead neuron 개수. 근데 이게 GAP@Random 등 SAE 안쓰는 경우에는, 그냥 origianal dim이 SAE에서 계산된 original dim-n_removed가 된다.
        print(f"    Zero-variance features removed: {n_removed}") #분산 적어서 날림
        print(f"    Alive features: {n_alive}") # original_dim - n_removed
    
    # Zero-variance features 제거
    Xtr = Xtr[:, alive_mask]
    Xte = Xte[:, alive_mask]
    
    # Normalize 옵션
    if normalize_repr:
        # Z-score 정규화 (alive features만)
        mean = Xtr.mean(axis=0, keepdims=True)
        std = Xtr.std(axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)  # safety
        
        Xtr = (Xtr - mean) / std
        Xte = (Xte - mean) / std
        
        if verbose:
            print(f"    [NORMALIZED] After z-score: mean={Xtr.mean():.6f}, std={Xtr.std():.6f}")
    
    d_in = Xtr.shape[1]
    probe = LinearProbe(d_in, num_classes).to(device)
    opt = optim.SGD(probe.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    ce = nn.CrossEntropyLoss()

    rng = np.random.default_rng(0)
    
    # ★ 클래스별 인덱스 그룹핑 (balanced sampling용)
    class_indices = {c: np.where(ytr == c)[0] for c in range(num_classes)}
    min_class_count = min(len(v) for v in class_indices.values())
    samples_per_class = min_class_count  # 각 클래스에서 이 개수만큼 샘플링
    
    if balanced_training:
        # balanced 모드: 각 클래스에서 동일한 수의 샘플 사용
        total_samples_per_epoch = samples_per_class * num_classes
    else:
        # 일반 모드: 전체 데이터 사용
        total_samples_per_epoch = len(ytr)
    
    # Debug info 저장
    debug_info = {
        "method": method_name,
        "original_dim": original_dim,
        "n_zero_var_removed": n_removed,
        "d_in": d_in,
        "num_classes": num_classes,
        "n_train": len(ytr),
        "n_test": len(yte),
        "epochs": epochs,
        "lr": lr,
        "wd": wd,
        "momentum": momentum,
        "batch_size": batch_size,
        "balanced_training": balanced_training,
        "samples_per_class": samples_per_class if balanced_training else None,
        "total_samples_per_epoch": total_samples_per_epoch,
        "train_history": [],
    }
    
    # Label 분포 확인
    train_label_counts = np.bincount(ytr, minlength=num_classes)
    test_label_counts = np.bincount(yte, minlength=num_classes)
    debug_info["train_label_distribution"] = train_label_counts.tolist()
    debug_info["test_label_distribution"] = test_label_counts.tolist()
    
    if verbose:
        print(f"\n[{method_name}] Probe Training Debug Info:")
        print(f"  Input dim: {d_in}, Classes: {num_classes}")
        print(f"  Train samples: {len(ytr)}, Test samples: {len(yte)}")
        print(f"  Train label dist: {train_label_counts}")
        print(f"  Test label dist:  {test_label_counts}")
        print(f"  Hyperparams: epochs={epochs}, lr={lr}, wd={wd}, mom={momentum}")
        if balanced_training:
            print(f"  ★ BALANCED TRAINING: {samples_per_class} samples/class x {num_classes} classes = {total_samples_per_epoch} samples/epoch")
        
        # ★ 핵심 디버깅: X 값 자체 확인
        print(f"\n  [X_TRAIN STATS]")
        print(f"    Shape: {Xtr.shape}")
        print(f"    Mean: {np.mean(Xtr):.6f}, Std: {np.std(Xtr):.6f}")
        print(f"    Min: {np.min(Xtr):.6f}, Max: {np.max(Xtr):.6f}")
        print(f"    NaN count: {np.sum(np.isnan(Xtr))}, Inf count: {np.sum(np.isinf(Xtr))}")
        print(f"    Zero ratio: {np.mean(Xtr == 0):.4f}")
        print(f"    Sample row[0][:10]: {Xtr[0][:10]}")
        
        # 클래스별 평균 확인
        for c in range(num_classes):
            mask = ytr == c
            if np.sum(mask) > 0:
                class_mean = np.mean(Xtr[mask])
                class_std = np.std(Xtr[mask])
                print(f"    Class {c}: n={np.sum(mask)}, mean={class_mean:.6f}, std={class_std:.6f}")
    
    # 초기 weight 통계
    with torch.no_grad():
        init_weight = probe.fc.weight.clone()
        debug_info["init_weight_norm"] = float(init_weight.norm().item())
        debug_info["init_weight_std"] = float(init_weight.std().item())
    
    probe.train()
    for ep in range(1, epochs+1):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        n_batches = 0
        
        if balanced_training:
            # ★ Balanced sampling: 각 클래스에서 동일한 수 샘플링
            epoch_indices = []
            for c in range(num_classes):
                c_idx = class_indices[c]
                sampled = rng.choice(c_idx, size=samples_per_class, replace=False)
                epoch_indices.extend(sampled.tolist())
            epoch_indices = np.array(epoch_indices)
            rng.shuffle(epoch_indices)
        else:
            # 일반 모드: 전체 데이터 사용
            epoch_indices = np.arange(len(ytr))
            rng.shuffle(epoch_indices)
        
        for s in range(0, len(epoch_indices), batch_size):
            ii = epoch_indices[s:s+batch_size]
            xb = torch.from_numpy(Xtr[ii]).to(device)
            yb = torch.from_numpy(ytr[ii]).to(device)
            
            opt.zero_grad(set_to_none=True)
            logits = probe(xb)
            loss = ce(logits, yb)
            loss.backward()
            opt.step()
            
            epoch_loss += loss.item()
            pred = logits.argmax(dim=1)
            epoch_correct += int((pred == yb).sum().item())
            epoch_total += int(yb.numel())
            n_batches += 1
        
        avg_loss = epoch_loss / max(n_batches, 1)
        train_acc = epoch_correct / max(epoch_total, 1)
        
        # Test accuracy도 매 epoch 계산
        probe.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for s in range(0, len(yte), batch_size):
                xb = torch.from_numpy(Xte[s:s+batch_size]).to(device)
                yb = torch.from_numpy(yte[s:s+batch_size]).to(device)
                pred = probe(xb).argmax(dim=1)
                test_correct += int((pred == yb).sum().item())
                test_total += int(yb.numel())
        test_acc = test_correct / max(test_total, 1)
        probe.train()
        
        epoch_info = {
            "epoch": ep,
            "train_loss": avg_loss,
            "train_acc": train_acc,
            "test_acc": test_acc,
        }
        debug_info["train_history"].append(epoch_info)
        
        if verbose and (ep % log_every == 0 or ep == epochs):
            print(f"  Epoch {ep:3d}/{epochs}: loss={avg_loss:.4f}, train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")
    
    # 최종 weight 통계
    with torch.no_grad():
        final_weight = probe.fc.weight.clone()
        debug_info["final_weight_norm"] = float(final_weight.norm().item())
        debug_info["final_weight_std"] = float(final_weight.std().item())
        weight_diff = (final_weight - init_weight.to(final_weight.device)).norm().item()
        debug_info["weight_change_norm"] = weight_diff
    
    if verbose:
        print(f"  Weight change: init_norm={debug_info['init_weight_norm']:.4f} -> final_norm={debug_info['final_weight_norm']:.4f}, change={weight_diff:.4f}")
    
    # 최종 test accuracy
    probe.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for s in range(0, len(yte), batch_size):
            xb = torch.from_numpy(Xte[s:s+batch_size]).to(device)
            yb = torch.from_numpy(yte[s:s+batch_size]).to(device)
            pred = probe(xb).argmax(dim=1)
            correct += int((pred == yb).sum().item())
            total += int(yb.numel())
            all_preds.extend(pred.cpu().numpy().tolist())
            all_labels.extend(yb.cpu().numpy().tolist())
    
    final_test_acc = 0.0 if total == 0 else float(correct / total)
    debug_info["final_test_acc"] = final_test_acc
    
    # Per-class accuracy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    per_class_acc = {}
    for c in range(num_classes):
        mask = all_labels == c
        if np.sum(mask) > 0:
            per_class_acc[c] = float(np.mean(all_preds[mask] == c))
    debug_info["per_class_accuracy"] = per_class_acc
    
    if verbose:
        print(f"  Per-class accuracy: {per_class_acc}")
        print(f"  Final test accuracy: {final_test_acc:.4f}")
    
    # Confusion check: 가장 많이 예측된 클래스
    pred_counts = np.bincount(all_preds, minlength=num_classes)
    debug_info["prediction_distribution"] = pred_counts.tolist()
    
    if verbose:
        print(f"  Prediction distribution: {pred_counts}")
        majority_class = np.argmax(pred_counts)
        majority_ratio = pred_counts[majority_class] / len(all_preds)
        if majority_ratio > 0.7:
            print(f"  [WARNING] Model predicts class {majority_class} for {majority_ratio:.1%} of samples!")
    
    return final_test_acc, debug_info


def main():
    args = resolve_paths(get_args())
    set_seed(args.seed)

    # ============================================================
    # HYPERPARAMETERS (쉽게 조절할 수 있도록 상단에 배치)
    # ============================================================
    K_LIST = [5, 10, 15, 20, 25, 30]
    LAYER_LIST = ["stage5_out", "refine_out"]  # 두 레이어 모두 평가
    
    N_TRAIN_IMG = 4800
    N_TEST_IMG  = 3200
    TOKEN_BATCH = int(getattr(args, "token_batch", 65536))
    
    # SAE input policy
    SAE_INPUT_CENTER = True
    SAE_INPUT_L2NORM = True # 피처맵 각 토큰마다 L2 정규화. 크기 아니라 방향에 정보 담김. 물론 크기도 이미지에 대한 정보를 담고 있는데 모델 학습시에는 방향만으로 개념 학습하게 해야해서
                            # 밑에 strength_weight true false 로 둘 다로 해볼 수 있음. 즉 이때는 각 토큰 L2 norm으로 나눠주는데 L2 norm을 기억하고 있다가 SAE후에 각 토큰마다 L2 norm 곱해줘서 어느정도 복원시킴.
    # classification representation variants
    STRENGTH_MODES = [False, True]
    
    # ============================================================
    # PROBE HYPERPARAMETERS (★ 여기를 조절하세요!)
    # ============================================================
    LP_EPOCHS = 50       # epochs
    LP_LR = 0.1          # 학습률 (L2 정규화 시 적당한 수준)
    LP_WD = 0.0         # weight decay (regularization)
    LP_MOM = 0.9         # momentum
    LP_BS = 256          # batch size
    LP_LOG_EVERY = 10    # epoch마다 로그 출력 간격
    LP_VERBOSE = True    # 상세 로그 출력 여부
    LP_NORMALIZE = False # ★ L2 정규화 이미 적용되므로 z-score 불필요
    
    DEAD_THRESHOLD = 1e-6
    DEBUG_REPR_STATS = True  # representation 통계 출력 여부
    # ============================================================

    SAE_ROOT = os.path.join(args.save_dir, "SAE")
    OUT_DIR  = os.path.join(args.save_dir, "eval_train_probe_test_eval")
    os.makedirs(OUT_DIR, exist_ok=True)
    OUT_CSV  = os.path.join(OUT_DIR, "results_comprehensive.csv")
    DEBUG_JSON = os.path.join(OUT_DIR, "debug_info.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[eval] device={device}")
    
    print(f"\n{'#'*70}")
    print(f"# PROBE HYPERPARAMETERS")
    print(f"#   epochs={LP_EPOCHS}, lr={LP_LR}, wd={LP_WD}, mom={LP_MOM}, bs={LP_BS}")
    print(f"{'#'*70}\n")

    # ---- load refs & splits ----
    refs = load_all_sample_refs(args.shard_root)
    uid_to_refidx = build_uid_to_refidx(refs)

    train_csv = os.path.join(args.save_dir, "train_split.csv")
    test_csv  = os.path.join(args.save_dir, "test_split.csv")
    if not os.path.exists(train_csv):
        raise FileNotFoundError(train_csv)
    if not os.path.exists(test_csv):
        raise FileNotFoundError(test_csv)

    u_train = load_split_csv(train_csv)
    u_test  = load_split_csv(test_csv)

    refs_by_uid = {u: refs[uid_to_refidx[u]] for u in (u_train + u_test)}

    train_sub = pick_balanced_uids(u_train, refs_by_uid, N_TRAIN_IMG, seed=args.seed+100)
    test_sub  = pick_balanced_uids(u_test,  refs_by_uid, N_TEST_IMG,  seed=args.seed+200)

    # ============================================================
    # Train/Test 분리 검증
    # ============================================================
    print(f"\n{'='*70}")
    print(f"TRAIN/TEST SEPARATION VERIFICATION")
    print(f"{'='*70}")
    separation_info = verify_train_test_separation(train_sub, test_sub, refs_by_uid)
    print(f"{'='*70}\n")

    # ============================================================
    # Loader 생성 (★ balanced sampler 사용!)
    # ============================================================
    # representation 추출용 loader는 balanced sampler 사용하지 않음 (전체 데이터 순회)
    # linear probe 학습용으로는 추출된 representation에서 balanced sampling
    
    print(f"\n{'='*70}")
    print(f"CREATING DATA LOADERS")
    print(f"{'='*70}")
    
    # 모든 train/test 데이터를 로드 (representation 추출용)
    train_loader, train_bank = make_loader_for_uid_subset(
        args, refs, uid_to_refidx, train_sub, 
        batch_size=args.batch_size,
        use_balanced_sampler=False,  # 추출 시에는 전체 순회
        seed=args.seed+100,
    )
    test_loader, test_bank = make_loader_for_uid_subset(
        args, refs, uid_to_refidx, test_sub, 
        batch_size=args.batch_size,
        use_balanced_sampler=False,  # 추출 시에는 전체 순회
        seed=args.seed+200,
    )
    
    # Bank 내 클래스 분포 출력
    train_dist = count_class_distribution_in_bank(train_bank)
    test_dist = count_class_distribution_in_bank(test_bank)
    print(f"  Train bank class distribution: {train_dist}")
    print(f"  Test bank class distribution:  {test_dist}")
    print(f"{'='*70}\n")

    # ---- load encoder ----
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
    from sae_project.step05_model_encoder import robust_load_state_dict
    robust_load_state_dict(model, sd, strict=True)
    encoder = model.encoder.to(device).eval().to(memory_format=torch.channels_last)
    
    # ★ 중요: LRKK21과 동일하게 weight 정규화 적용!
    renorm_unit_per_out_channel_(encoder)
    
    # ★ Encoder freeze (추론 전용, 학습되지 않도록)
    for p in encoder.parameters():
        p.requires_grad = False
    
    logger.info("[eval] Applied renorm_unit_per_out_channel_() and froze encoder")

    # ---- write CSV header ----
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "which_layer",
            "k",
            "method",
            "strength_weight",
            "test_acc",
            "repr_dim",
            "fisher_ratio",
            "sae_ckpt",
            "note",
        ])
    
    # Debug info 저장용
    all_debug_info = {
        "hyperparams": {
            "epochs": LP_EPOCHS,
            "lr": LP_LR,
            "wd": LP_WD,
            "momentum": LP_MOM,
            "batch_size": LP_BS,
        },
        "separation_info": separation_info,
        "methods": [],
    }

    # ========================================
    # 각 레이어별로 평가
    # ========================================
    for layer in LAYER_LIST:
        print(f"\n{'#'*70}")
        print(f"# EVALUATING LAYER: {layer}")
        print(f"{'#'*70}")
        logger.info(f"===== Evaluating layer: {layer} =====")
        
        # ---------------------------------
        # Baseline 1: GAP512 (해당 레이어)
        # ---------------------------------
        print(f"\n[{layer}] Extracting GAP512 representations...")
        Xtr_gap, ytr = gap_repr(encoder, train_loader, device, which_layer=layer)
        Xte_gap, yte = gap_repr(encoder, test_loader, device, which_layer=layer)
        
        # Representation 통계 출력
        if DEBUG_REPR_STATS:
            stats_tr = compute_repr_stats(Xtr_gap, ytr, f"{layer}_GAP512_train")
            stats_te = compute_repr_stats(Xte_gap, yte, f"{layer}_GAP512_test")
            print_repr_stats(stats_tr, f"{layer}_GAP512_train")
            fisher_gap = stats_tr.get("fisher_ratio", 0.0)
        else:
            fisher_gap = 0.0

        acc, debug = train_probe_and_eval_test(
            Xtr_gap, ytr, Xte_gap, yte,
            num_classes=int(getattr(args,"num_classes",4)),
            device=device,
            epochs=LP_EPOCHS, lr=LP_LR, wd=LP_WD, momentum=LP_MOM, batch_size=LP_BS,
            method_name=f"{layer}_GAP512",
            verbose=LP_VERBOSE,
            log_every=LP_LOG_EVERY,
            balanced_training=True,
            normalize_repr=LP_NORMALIZE,
        )
        all_debug_info["methods"].append(debug)
        
        with open(OUT_CSV, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([layer, "", "gap512", "", acc, 512, fisher_gap, "", "GAP512 baseline"])
        print(f"\n>>> [{layer}] GAP512: acc={acc:.4f}, fisher={fisher_gap:.4f}")

        # ---------------------------------
        # 각 k 값에 대해 SAE 및 matched baselines
        # ---------------------------------
        for k in K_LIST:
            print(f"\n{'-'*50}")
            print(f"[{layer}] Processing k={k}")
            print(f"{'-'*50}")
            
            # SAE checkpoint 찾기 (해당 레이어, 해당 k)
            try:
                sae_ckpt = find_sae_ckpt_for_k(SAE_ROOT, k)
                # layer 이름이 checkpoint 경로에 포함되어 있는지 확인
                if layer not in sae_ckpt and layer.replace("_", "") not in sae_ckpt:
                    # 레이어별 폴더 구조인 경우 시도
                    layer_sae_root = os.path.join(SAE_ROOT, layer)
                    if os.path.exists(layer_sae_root):
                        sae_ckpt = find_sae_ckpt_for_k(layer_sae_root, k)
            except FileNotFoundError:
                logger.warning(f"[{layer}] SAE ckpt not found for k={k}, skipping...")
                continue

            logger.info(f"[{layer}] Loading SAE k={k}: {sae_ckpt}")
            sae, ckpt_args = load_sae_from_ckpt(sae_ckpt, device=device)
            
            # alive neuron 수 확인 (dead neuron 제거)
            alive_mask = sae.usage_ema.cpu().numpy() >= DEAD_THRESHOLD
            alive_count = int(alive_mask.sum())
            dead_count = int(sae.d_sae) - alive_count
            print(f"[{layer}] k={k}: d_sae={sae.d_sae}, alive={alive_count}, dead={dead_count}")

            # ---------------------------------
            # ★ SAE를 먼저 추출하여 zero-variance 제거 후 실제 차원 확인
            # ---------------------------------
            print(f"\n[{layer}] k={k} - Extracting SAE(binary) first to determine actual dimension...")
            Xtr_sae_bin, ytr_sae_bin, _ = pooled_sae_repr_fulltokens(
                encoder, sae, train_loader, device=device,
                which_layer=layer,
                token_batch=TOKEN_BATCH,
                sae_input_center=SAE_INPUT_CENTER,
                sae_input_l2_norm=SAE_INPUT_L2NORM,
                strength_weight=False,  # binary
                dead_threshold=DEAD_THRESHOLD,
                exclude_dead=True,
            )
            
            # Zero-variance features 확인 (train set 기준)
            feat_std = Xtr_sae_bin.std(axis=0)
            zero_var_mask = feat_std < 1e-8
            n_zero_var = int(zero_var_mask.sum())
            actual_dim = alive_count - n_zero_var  # ★ 실제 사용 차원
            
            print(f"[{layer}] k={k}: zero_var_features={n_zero_var}, actual_dim={actual_dim}")

            # ---------------------------------
            # Random matrix 생성 (★ actual_dim 차원에 맞춤!)
            # ---------------------------------
            g = torch.Generator(device=device)
            g.manual_seed(int(args.seed) + k * 100 + hash(layer) % 1000)
            R_matched = (torch.randn(OUT_DIM, actual_dim, device=device, generator=g) / math.sqrt(OUT_DIM)).float()

            # ---------------------------------
            # Baseline 2: GAP @ Random (차원 증가 효과 통제)
            # ---------------------------------
            print(f"\n[{layer}] k={k} - Extracting GAP@Random (dim={actual_dim})...")
            Xtr_gr, ytr_gr = gap_randproj_repr(encoder, train_loader, device, which_layer=layer, R=R_matched)
            Xte_gr, yte_gr = gap_randproj_repr(encoder, test_loader, device, which_layer=layer, R=R_matched)

            if DEBUG_REPR_STATS:
                stats_gr = compute_repr_stats(Xtr_gr, ytr_gr, f"{layer}_k{k}_GAP@Random_train")
                print_repr_stats(stats_gr, f"{layer}_k{k}_GAP@Random_train")
                fisher_gr = stats_gr.get("fisher_ratio", 0.0)
            else:
                fisher_gr = 0.0

            acc_gr, debug_gr = train_probe_and_eval_test(
                Xtr_gr, ytr_gr, Xte_gr, yte_gr,
                num_classes=int(getattr(args,"num_classes",4)),
                device=device,
                epochs=LP_EPOCHS, lr=LP_LR, wd=LP_WD, momentum=LP_MOM, batch_size=LP_BS,
                method_name=f"{layer}_k{k}_GAP@Random",
                verbose=LP_VERBOSE,
                log_every=LP_LOG_EVERY,
                balanced_training=True,
                normalize_repr=LP_NORMALIZE,
            )
            all_debug_info["methods"].append(debug_gr)
            
            with open(OUT_CSV, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([layer, k, "gap_randproj", "", acc_gr, actual_dim, fisher_gr, "", f"GAP@Random({actual_dim}d) - dimension expansion control"])
            print(f">>> [{layer}] k={k} GAP@Random: acc={acc_gr:.4f}, dim={actual_dim}, fisher={fisher_gr:.4f}")

            # ---------------------------------
            # Baseline 3: Token @ Random + Top-K (sparsity 통제)
            # ---------------------------------
            print(f"\n[{layer}] k={k} - Extracting Token@Random+TopK (dim={actual_dim})...")
            Xtr_tk, ytr_tk = pooled_randproj_topk_repr(
                encoder, train_loader, device, which_layer=layer,
                token_batch=TOKEN_BATCH, R=R_matched, k=k,
                token_center=SAE_INPUT_CENTER, token_l2_norm=SAE_INPUT_L2NORM
            )
            Xte_tk, yte_tk = pooled_randproj_topk_repr(
                encoder, test_loader, device, which_layer=layer,
                token_batch=TOKEN_BATCH, R=R_matched, k=k,
                token_center=SAE_INPUT_CENTER, token_l2_norm=SAE_INPUT_L2NORM
            )

            if DEBUG_REPR_STATS:
                stats_tk = compute_repr_stats(Xtr_tk, ytr_tk, f"{layer}_k{k}_Token@Random+TopK_train")
                print_repr_stats(stats_tk, f"{layer}_k{k}_Token@Random+TopK_train")
                fisher_tk = stats_tk.get("fisher_ratio", 0.0)
            else:
                fisher_tk = 0.0

            acc_tk, debug_tk = train_probe_and_eval_test(
                Xtr_tk, ytr_tk, Xte_tk, yte_tk,
                num_classes=int(getattr(args,"num_classes",4)),
                device=device,
                epochs=LP_EPOCHS, lr=LP_LR, wd=LP_WD, momentum=LP_MOM, batch_size=LP_BS,
                method_name=f"{layer}_k{k}_Token@Random+TopK",
                verbose=LP_VERBOSE,
                log_every=LP_LOG_EVERY,
                balanced_training=True,
                normalize_repr=LP_NORMALIZE,
            )
            all_debug_info["methods"].append(debug_tk)
            
            with open(OUT_CSV, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([layer, k, "tok_randproj_topk", "", acc_tk, actual_dim, fisher_tk, "", f"Token@Random+TopK({k}) - SAE pipeline with random weights"])
            print(f">>> [{layer}] k={k} Token@Random+TopK: acc={acc_tk:.4f}, dim={actual_dim}, fisher={fisher_tk:.4f}")

            # ---------------------------------
            # SAE (learned features) - binary는 이미 추출함, intensity만 추출
            # ---------------------------------
            # Binary (이미 추출됨)
            Xte_sae_bin, yte_sae_bin, _ = pooled_sae_repr_fulltokens(
                encoder, sae, test_loader, device=device,
                which_layer=layer,
                token_batch=TOKEN_BATCH,
                sae_input_center=SAE_INPUT_CENTER,
                sae_input_l2_norm=SAE_INPUT_L2NORM,
                strength_weight=False,
                dead_threshold=DEAD_THRESHOLD,
                exclude_dead=True,
            )
            
            if DEBUG_REPR_STATS:
                stats_sae = compute_repr_stats(Xtr_sae_bin, ytr_sae_bin, f"{layer}_k{k}_SAE(binary)_train")
                print_repr_stats(stats_sae, f"{layer}_k{k}_SAE(binary)_train")
                fisher_sae = stats_sae.get("fisher_ratio", 0.0)
            else:
                fisher_sae = 0.0

            acc_sae, debug_sae = train_probe_and_eval_test(
                Xtr_sae_bin, ytr_sae_bin, Xte_sae_bin, yte_sae_bin,
                num_classes=int(getattr(args,"num_classes",4)),
                device=device,
                epochs=LP_EPOCHS, lr=LP_LR, wd=LP_WD, momentum=LP_MOM, batch_size=LP_BS,
                method_name=f"{layer}_k{k}_SAE(binary)",
                verbose=LP_VERBOSE,
                log_every=LP_LOG_EVERY,
                balanced_training=True,
                normalize_repr=LP_NORMALIZE,
            )
            all_debug_info["methods"].append(debug_sae)
            
            with open(OUT_CSV, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([layer, k, "sae_binary", False, acc_sae, actual_dim, fisher_sae, sae_ckpt, f"SAE learned features | strength_weight=False"])
            print(f">>> [{layer}] k={k} SAE(binary): acc={acc_sae:.4f}, dim={actual_dim}, fisher={fisher_sae:.4f}")

            # Intensity
            print(f"\n[{layer}] k={k} - Extracting SAE(intensity)...")
            Xtr_sae_int, ytr_sae_int, _ = pooled_sae_repr_fulltokens(
                encoder, sae, train_loader, device=device,
                which_layer=layer,
                token_batch=TOKEN_BATCH,
                sae_input_center=SAE_INPUT_CENTER,
                sae_input_l2_norm=SAE_INPUT_L2NORM,
                strength_weight=True,  # intensity
                dead_threshold=DEAD_THRESHOLD,
                exclude_dead=True,
            )
            Xte_sae_int, yte_sae_int, _ = pooled_sae_repr_fulltokens(
                encoder, sae, test_loader, device=device,
                which_layer=layer,
                token_batch=TOKEN_BATCH,
                sae_input_center=SAE_INPUT_CENTER,
                sae_input_l2_norm=SAE_INPUT_L2NORM,
                strength_weight=True,
                dead_threshold=DEAD_THRESHOLD,
                exclude_dead=True,
            )
            
            if DEBUG_REPR_STATS:
                stats_sae_int = compute_repr_stats(Xtr_sae_int, ytr_sae_int, f"{layer}_k{k}_SAE(intensity)_train")
                print_repr_stats(stats_sae_int, f"{layer}_k{k}_SAE(intensity)_train")
                fisher_sae_int = stats_sae_int.get("fisher_ratio", 0.0)
            else:
                fisher_sae_int = 0.0

            acc_sae_int, debug_sae_int = train_probe_and_eval_test(
                Xtr_sae_int, ytr_sae_int, Xte_sae_int, yte_sae_int,
                num_classes=int(getattr(args,"num_classes",4)),
                device=device,
                epochs=LP_EPOCHS, lr=LP_LR, wd=LP_WD, momentum=LP_MOM, batch_size=LP_BS,
                method_name=f"{layer}_k{k}_SAE(intensity)",
                verbose=LP_VERBOSE,
                log_every=LP_LOG_EVERY,
                balanced_training=True,
                normalize_repr=LP_NORMALIZE,
            )
            all_debug_info["methods"].append(debug_sae_int)
            
            with open(OUT_CSV, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([layer, k, "sae_intensity", True, acc_sae_int, actual_dim, fisher_sae_int, sae_ckpt, f"SAE learned features | strength_weight=True"])
            print(f">>> [{layer}] k={k} SAE(intensity): acc={acc_sae_int:.4f}, dim={actual_dim}, fisher={fisher_sae_int:.4f}")

            torch.cuda.empty_cache()

    # ---- save meta ----
    meta = {
        "layers": LAYER_LIST,
        "k_list": K_LIST,
        "train_subset_images": len(train_sub),
        "test_subset_images": len(test_sub),
        "token_batch": int(TOKEN_BATCH),
        "sae_input_center": SAE_INPUT_CENTER,
        "sae_input_l2_norm": SAE_INPUT_L2NORM,
        "strength_modes": STRENGTH_MODES,
        "sae_root": SAE_ROOT,
        "dead_threshold": DEAD_THRESHOLD,
        "probe_hyperparams": {
            "epochs": LP_EPOCHS,
            "lr": LP_LR,
            "wd": LP_WD,
            "momentum": LP_MOM,
            "batch_size": LP_BS,
        },
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    
    # Debug info 저장
    with open(DEBUG_JSON, "w", encoding="utf-8") as f:
        json.dump(all_debug_info, f, indent=2, ensure_ascii=False)

    logger.info(f"[eval] done. saved -> {OUT_CSV}")
    logger.info(f"[eval] debug info -> {DEBUG_JSON}")
    
    # ============================================================
    # 최종 요약 출력
    # ============================================================
    print(f"\n{'#'*70}")
    print(f"# FINAL SUMMARY")
    print(f"{'#'*70}")
    print(f"Results CSV: {OUT_CSV}")
    print(f"Debug JSON:  {DEBUG_JSON}")
    print(f"Meta JSON:   {os.path.join(OUT_DIR, 'meta.json')}")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    main()


