# ==============================================================================
# Token Clustering Pipeline (CPU 버전)
# L2 norm 가중치 샘플링 -> PCA(50차원) -> Leiden Clustering -> UMAP 시각화
# 
# 목적: CNN이 집중하는 유의미한 토큰들의 클러스터를 찾고,
#       SAE 초기 가중치로 사용할 centroid 추출
# ==============================================================================

import os
import random
import time
from typing import List, Tuple, Dict
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

try:
    import umap
except ImportError:
    raise ImportError("pip install umap-learn")

try:
    import leidenalg
    import igraph as ig
except ImportError:
    raise ImportError("pip install leidenalg python-igraph")

from sae_project.step01_configs import get_args, resolve_paths
from sae_project.step02_logging_utils import get_logger, OUT_DIM, SUPERCLASS_MAP
from sae_project.step03_data_shards import load_all_sample_refs, build_uid_to_refidx
from sae_project.step04_data_bank import (
    InMemoryTarBank, InMemorySixteenBitDataset, load_split_csv,
    seed_worker, collate_skip_none
)
from sae_project.step05_model_encoder import SupMoCoModel, parse_int_list
from sae_project.step12_build_token_cache import pick_balanced_images

logger = get_logger("token_clustering")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def extract_tokens_with_l2norm(encoder, x: torch.Tensor, which_layer: str):
    """Raw token 추출 + L2 norm 계산"""
    fmap = encoder.forward_feature_maps(x, which=which_layer)
    B, C, H, W = fmap.shape
    tokens = fmap.permute(0, 2, 3, 1).reshape(B * H * W, C)
    l2_norms = tokens.norm(dim=1)
    return tokens, l2_norms, B, H, W


def collect_tokens_weighted_sampling(
    args, encoder, loader, which_layer: str, device: torch.device,
    target_tokens: int, max_images: int
) -> Tuple[np.ndarray, np.ndarray]:
    """L2 norm 비례 샘플링 (2-pass, 메모리 효율적)"""
    encoder.eval()
    autocast_kwargs = dict(device_type="cuda", enabled=torch.cuda.is_available(), dtype=torch.bfloat16)
    
    # Pass 1: L2 norm 합계
    logger.info("[Phase 1a] Computing total L2 norm sum...")
    total_l2_sum, total_count, img_count = 0.0, 0, 0
    
    for batch in tqdm(loader, desc="[Pass1]"):
        if batch is None: continue
        x_cpu, y_cpu, *_ = batch
        if y_cpu.numel() < 1: continue
        
        x = x_cpu.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        with torch.amp.autocast(**autocast_kwargs):
            _, l2_norms, B, H, W = extract_tokens_with_l2norm(encoder, x, which_layer)
        
        total_l2_sum += l2_norms.float().sum().item()
        total_count += len(l2_norms)
        img_count += B
        if max_images and img_count >= max_images: break
    
    logger.info(f"Total: {total_count:,} tokens, L2 sum: {total_l2_sum:.2f}")
    
    # Pass 2: L2 비례 샘플링
    logger.info(f"[Phase 1b] Weighted sampling {target_tokens:,} tokens...")
    sampled_tokens, sampled_l2norms = [], []
    img_count, collected = 0, 0
    np.random.seed(args.seed + 123)
    
    loader2 = DataLoader(loader.dataset, batch_size=loader.batch_size, shuffle=False,
                        num_workers=0, pin_memory=torch.cuda.is_available(),
                        collate_fn=collate_skip_none, drop_last=False)
    
    for batch in tqdm(loader2, desc="[Pass2]"):
        if batch is None: continue
        x_cpu, y_cpu, *_ = batch
        if y_cpu.numel() < 1: continue
        
        x = x_cpu.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        with torch.amp.autocast(**autocast_kwargs):
            tokens, l2_norms, B, H, W = extract_tokens_with_l2norm(encoder, x, which_layer)
        
        tokens_np = tokens.float().cpu().numpy()
        l2_norms_np = l2_norms.float().cpu().numpy()
        
        batch_l2 = l2_norms_np.sum()
        n_sample = min(int(np.ceil((batch_l2 / total_l2_sum) * target_tokens)), len(tokens_np))
        
        if n_sample > 0:
            probs = l2_norms_np / l2_norms_np.sum()
            idx = np.random.choice(len(tokens_np), size=n_sample, replace=False, p=probs)
            sampled_tokens.append(tokens_np[idx])
            sampled_l2norms.append(l2_norms_np[idx])
            collected += n_sample
        
        img_count += B
        if max_images and img_count >= max_images: break
    
    result_tokens = np.concatenate(sampled_tokens, axis=0)
    result_l2norms = np.concatenate(sampled_l2norms, axis=0)
    
    if len(result_tokens) > target_tokens:
        idx = np.random.choice(len(result_tokens), size=target_tokens, replace=False)
        result_tokens, result_l2norms = result_tokens[idx], result_l2norms[idx]
    
    logger.info(f"Sampled: {len(result_tokens):,}, L2 mean: {result_l2norms.mean():.4f}")
    return result_tokens, result_l2norms


def apply_pca(tokens: np.ndarray, n_components: int = 50):
    """PCA 차원 축소"""
    logger.info(f"[Phase 2] PCA: {tokens.shape[1]} -> {n_components} dim...")
    scaler = StandardScaler()
    pca = PCA(n_components=n_components, random_state=42)
    tokens_pca = pca.fit_transform(scaler.fit_transform(tokens))
    logger.info(f"Explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    return tokens_pca, pca


def build_knn_graph_cpu(tokens_pca: np.ndarray, k: int = 15) -> ig.Graph:
    """sklearn CPU 기반 k-NN 그래프 (cosine 유사도)"""
    logger.info(f"[Phase 3] Building {k}-NN graph for {len(tokens_pca):,} points (CPU)...")
    t0 = time.time()
    
    nn = NearestNeighbors(n_neighbors=k+1, metric='cosine', algorithm='brute', n_jobs=-1)
    nn.fit(tokens_pca)
    
    logger.info(f"  Fitting done ({time.time()-t0:.1f}s), searching neighbors...")
    t1 = time.time()
    distances, indices = nn.kneighbors(tokens_pca)
    logger.info(f"  k-NN search done ({time.time()-t1:.1f}s)")
    
    # Edge list (skip self)
    logger.info("  Building edge list...")
    n = len(tokens_pca)
    neighbor_idx = indices[:, 1:]
    neighbor_dist = distances[:, 1:]
    
    src = np.repeat(np.arange(n), k)
    dst = neighbor_idx.flatten()
    weights = np.clip(1.0 - neighbor_dist.flatten(), 0, 1).tolist()
    
    g = ig.Graph(n=n, edges=list(zip(src, dst)), directed=False)
    g.es['weight'] = weights
    g = g.simplify(combine_edges='mean')
    
    logger.info(f"  Graph: {g.vcount()} vertices, {g.ecount()} edges ({time.time()-t0:.1f}s total)")
    return g


def leiden_clustering(g: ig.Graph, resolution: float = 1.0) -> np.ndarray:
    """Leiden 클러스터링"""
    logger.info(f"[Phase 4] Leiden clustering (resolution={resolution})...")
    partition = leidenalg.find_partition(
        g, leidenalg.RBConfigurationVertexPartition,
        weights='weight', resolution_parameter=resolution, seed=42
    )
    clusters = np.array(partition.membership)
    logger.info(f"Found {len(set(clusters))} clusters")
    return clusters


def apply_umap(tokens_pca: np.ndarray) -> np.ndarray:
    """UMAP 2D 임베딩"""
    logger.info("[Phase 5] UMAP visualization...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2,
                       metric='cosine', random_state=42, low_memory=True, verbose=True)
    return reducer.fit_transform(tokens_pca)


def extract_centroids(tokens: np.ndarray, clusters: np.ndarray):
    """클러스터 centroid 추출"""
    logger.info("[Phase 6] Extracting centroids...")
    unique = sorted(set(clusters))
    centroids = np.zeros((len(unique), tokens.shape[1]), dtype=np.float32)
    sizes = {}
    for i, c in enumerate(unique):
        mask = clusters == c
        centroids[i] = tokens[mask].mean(axis=0)
        sizes[c] = int(mask.sum())
    logger.info(f"Extracted {len(unique)} centroids")
    return centroids, sizes


def plot_results(embedding_2d: np.ndarray, clusters: np.ndarray, l2_norms: np.ndarray, save_dir: str):
    """UMAP 시각화"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    scatter1 = axes[0].scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=clusters, cmap='tab20', s=0.3, alpha=0.5)
    axes[0].set_title(f'Leiden Clusters (n={len(set(clusters))})')
    plt.colorbar(scatter1, ax=axes[0])
    
    scatter2 = axes[1].scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=l2_norms, cmap='viridis', s=0.3, alpha=0.5)
    axes[1].set_title('L2 Norm')
    plt.colorbar(scatter2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'umap_results.png'), dpi=200)
    plt.close()
    logger.info(f"Saved plots to {save_dir}")


def create_sae_init_weights(centroids: np.ndarray, n_sae_features: int, noise_scale: float = 0.1):
    """SAE 초기 가중치 생성"""
    n_centroids, d_in = centroids.shape
    logger.info(f"Creating SAE init: {d_in} -> {n_sae_features} ({n_centroids} centroids)")
    
    centroids_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
    
    if n_centroids >= n_sae_features:
        W = centroids_norm[:n_sae_features].T
    else:
        W = np.zeros((d_in, n_sae_features), dtype=np.float32)
        W[:, :n_centroids] = centroids_norm.T
        W[:, n_centroids:] = np.random.randn(d_in, n_sae_features - n_centroids).astype(np.float32) * 0.02
    
    W += np.random.randn(*W.shape).astype(np.float32) * noise_scale
    W /= (np.linalg.norm(W, axis=0, keepdims=True) + 1e-8)
    
    return W


def main():
    args = resolve_paths(get_args())
    set_seed(args.seed)
    
    # ========== Configuration ==========
    # 예상 시간: 3M 토큰 ~ 2-3시간, 5M ~ 5-6시간, 10M ~ 10시간+
    TARGET_TOKENS = 5_000_000      # 샘플링할 토큰 수
    MAX_IMAGES = 3000              # 최대 이미지 수
    PCA_DIM = 50
    KNN_K = 20
    LEIDEN_RESOLUTION = 1.0
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}, Target tokens: {TARGET_TOKENS:,}")
    
    # Load data
    refs = load_all_sample_refs(args.shard_root)
    uid_to_refidx = build_uid_to_refidx(refs)
    
    # Load encoder
    blocks = parse_int_list(args.blocks, 4)
    dilations = parse_int_list(args.dilations, 4)
    model = SupMoCoModel(
        embed_dim=args.embed_dim, blocks=blocks, dilations=dilations,
        refine_blocks=args.refine_blocks, ckpt_segments=args.ckpt_segments,
        proj_layers=args.proj_layers, proj_hidden=args.proj_hidden,
        proj_bn=args.proj_bn, proj_dropout=args.proj_dropout,
    )
    sd = torch.load(args.model_state_path, map_location="cpu")
    from sae_project.step05_model_encoder import robust_load_state_dict
    robust_load_state_dict(model, sd, strict=True)
    encoder = model.encoder.to(device).eval().to(memory_format=torch.channels_last)
    logger.info(f"Loaded encoder: {args.model_state_path}")
    
    # Load train split
    train_csv = os.path.join(args.save_dir, "train_split.csv")
    uids_all = load_split_csv(train_csv)
    valid_uids = [u for u in uids_all if u in uid_to_refidx]
    refs_by_uid = {u: refs[uid_to_refidx[u]] for u in valid_uids}
    
    picked_uids = pick_balanced_images(valid_uids, refs_by_uid, min(MAX_IMAGES, len(valid_uids)), args.seed)
    refidx = [uid_to_refidx[u] for u in picked_uids]
    logger.info(f"Selected {len(refidx)} images")
    
    # Build loader
    bank = InMemoryTarBank(refs, refidx, args.img_size)
    ds = InMemorySixteenBitDataset(bank, list(range(len(refidx))), args.img_size, augment=False)
    loader = DataLoader(ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0,
                       pin_memory=torch.cuda.is_available(), collate_fn=collate_skip_none)
    
    output_dir = os.path.join(args.save_dir, "token_clustering")
    os.makedirs(output_dir, exist_ok=True)
    which_layer = args.which_layer
    logger.info(f"Layer: {which_layer}, Output: {output_dir}")
    
    # === Pipeline ===
    t_start = time.time()
    
    # 1. Collect tokens
    tokens, l2norms = collect_tokens_weighted_sampling(
        args, encoder, loader, which_layer, device, TARGET_TOKENS, MAX_IMAGES
    )
    np.save(os.path.join(output_dir, f"tokens_{which_layer}.npy"), tokens)
    np.save(os.path.join(output_dir, f"l2norms_{which_layer}.npy"), l2norms)
    
    # 2. PCA
    tokens_pca, pca_model = apply_pca(tokens, PCA_DIM)
    import pickle
    with open(os.path.join(output_dir, f"pca_{which_layer}.pkl"), "wb") as f:
        pickle.dump(pca_model, f)
    
    # 3. k-NN graph (CPU)
    graph = build_knn_graph_cpu(tokens_pca, KNN_K)
    
    # 4. Leiden
    clusters = leiden_clustering(graph, LEIDEN_RESOLUTION)
    np.save(os.path.join(output_dir, f"clusters_{which_layer}.npy"), clusters)
    
    # 5. UMAP
    embedding = apply_umap(tokens_pca)
    np.save(os.path.join(output_dir, f"umap_{which_layer}.npy"), embedding)
    
    # 6. Plot
    plot_results(embedding, clusters, l2norms, output_dir)
    
    # 7. Centroids
    centroids, sizes = extract_centroids(tokens, clusters)
    np.save(os.path.join(output_dir, f"centroids_{which_layer}.npy"), centroids)
    
    # 8. SAE init
    W_init = create_sae_init_weights(centroids, args.d_sae)
    np.save(os.path.join(output_dir, f"sae_init_{which_layer}.npy"), W_init)
    
    # Summary
    elapsed = (time.time() - t_start) / 60
    logger.info(f"\n======== DONE ========")
    logger.info(f"Tokens: {len(tokens):,}, Clusters: {len(set(clusters))}")
    logger.info(f"Centroids: {centroids.shape}, SAE init: {W_init.shape}")
    logger.info(f"Total time: {elapsed:.1f} min")
    logger.info(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
