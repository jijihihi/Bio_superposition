# ==============================================================================
# SAE Concept Analysis & Evaluation Metrics
# - Purity: How class-specific each concept is (K/N where K=max class, N=total)
# - Shannon Entropy: Class distribution balance for each concept
# - Token Activation PMF: Distribution of how many concepts each token activates
# - Gini Coefficient: Inequality measure of activation distribution
# ==============================================================================

import os
import csv
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from sae_project.step01_configs import get_args, resolve_paths
from sae_project.step02_logging_utils import get_logger, SUPERCLASS_MAP
from sae_project.step03_data_shards import load_all_sample_refs, build_uid_to_refidx
from sae_project.step04_data_bank import (
    InMemoryTarBank, InMemorySixteenBitDataset, load_split_csv,
    seed_worker, collate_skip_none
)
from sae_project.step05_model_encoder import SupMoCoModel, parse_int_list, renorm_unit_per_out_channel_
from sae_project.step06_gated_sae import GatedSAE

logger = get_logger("sae_concept_eval")

# Class order for consistent indexing
CLASS_ORDER = ["Control", "SNCA", "GBA", "LRRK2"]
NUM_CLASSES = 4


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_gated_sae(ckpt_path: str, device: torch.device) -> GatedSAE:
    """Load Gated SAE from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args_dict = ckpt["args"]
    
    d_in = args_dict.get("d_in", 512)
    d_sae = args_dict.get("d_sae", 4096)
    tie_weights = args_dict.get("tie_gate_weights", False)
    aux_k = args_dict.get("aux_k", 32)
    
    sae = GatedSAE(
        d_in=d_in,
        d_sae=d_sae,
        tie_weights=tie_weights,
        aux_k=aux_k,
    )
    sae.load_state_dict(ckpt["sae"])
    sae.to(device).eval()
    
    logger.info(f"Loaded GatedSAE from {ckpt_path}")
    logger.info(f"  d_in={d_in}, d_sae={d_sae}, tie_weights={tie_weights}")
    
    return sae


def get_class_index(class_name: str) -> int:
    """Get class index from class name."""
    if class_name in CLASS_ORDER:
        return CLASS_ORDER.index(class_name)
    return -1


# ==============================================================================
# Concept Purity and Shannon Entropy
# ==============================================================================

def compute_concept_purity_and_entropy(
    concept_class_counts: Dict[int, Dict[int, int]],
    min_activation_count: int = 5,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Compute purity and Shannon entropy for each concept.
    
    Args:
        concept_class_counts: {concept_id: {class_id: count}}
        min_activation_count: Minimum activations to consider a concept
        
    Returns:
        purity_array: (d_sae,) purity values [0.25, 1.0]
        entropy_array: (d_sae,) entropy values [0, log2(4)]
        stats: Summary statistics
    """
    num_concepts = max(concept_class_counts.keys()) + 1 if concept_class_counts else 0
    
    purity = np.full(num_concepts, np.nan, dtype=np.float32)
    entropy = np.full(num_concepts, np.nan, dtype=np.float32)
    
    valid_concepts = 0
    high_purity_concepts = 0  # purity > 0.9
    class_specific_concepts = defaultdict(int)  # concepts where max class dominates
    
    for concept_id, class_counts in concept_class_counts.items():
        total = sum(class_counts.values())
        
        if total < min_activation_count:
            continue  # Skip rarely activated concepts
        
        valid_concepts += 1
        
        # Purity: K/N where K = max class count
        max_count = max(class_counts.values())
        max_class = max(class_counts, key=class_counts.get)
        purity[concept_id] = max_count / total
        
        if purity[concept_id] > 0.9:
            high_purity_concepts += 1
            class_specific_concepts[max_class] += 1
        
        # Shannon Entropy: -sum(p * log2(p))
        probs = []
        for cls in range(NUM_CLASSES):
            p = class_counts.get(cls, 0) / total
            if p > 0:
                probs.append(p)
        
        if probs:
            entropy[concept_id] = -sum(p * np.log2(p) for p in probs)
        else:
            entropy[concept_id] = 0.0
    
    # Compute summary statistics
    valid_purity = purity[~np.isnan(purity)]
    valid_entropy = entropy[~np.isnan(entropy)]
    
    stats = {
        "total_concepts": num_concepts,
        "valid_concepts": valid_concepts,
        "high_purity_concepts": high_purity_concepts,
        "class_specific_breakdown": dict(class_specific_concepts),
        "purity_mean": float(np.mean(valid_purity)) if len(valid_purity) > 0 else 0.0,
        "purity_std": float(np.std(valid_purity)) if len(valid_purity) > 0 else 0.0,
        "purity_median": float(np.median(valid_purity)) if len(valid_purity) > 0 else 0.0,
        "purity_p90": float(np.percentile(valid_purity, 90)) if len(valid_purity) > 0 else 0.0,
        "purity_p95": float(np.percentile(valid_purity, 95)) if len(valid_purity) > 0 else 0.0,
        "entropy_mean": float(np.mean(valid_entropy)) if len(valid_entropy) > 0 else 0.0,
        "entropy_std": float(np.std(valid_entropy)) if len(valid_entropy) > 0 else 0.0,
        "entropy_min": float(np.min(valid_entropy)) if len(valid_entropy) > 0 else 0.0,
        "max_entropy": float(np.log2(NUM_CLASSES)),  # log2(4) = 2.0
    }
    
    return purity, entropy, stats


# ==============================================================================
# Token Activation Distribution (PMF, Gini, Normalized Expectation)
# ==============================================================================

def compute_token_activation_pmf(
    token_concept_counts: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Compute PMF of how many concepts each active token activates.
    
    Args:
        token_concept_counts: (N,) array where N = number of tokens
                              Each value = number of concepts this token activated
                              
    Returns:
        x_values: (max_count+1,) integer values [0, 1, 2, ...]
        pmf: (max_count+1,) probability mass function (sums to 1)
        stats: Gini coefficient, normalized expectation, etc.
    """
    # Filter to only tokens that activated at least 1 concept
    active_counts = token_concept_counts[token_concept_counts > 0]
    
    if len(active_counts) == 0:
        return np.array([0]), np.array([1.0]), {"gini": 0.0, "mean": 0.0, "normalized_mean": 0.0}
    
    max_count = int(active_counts.max())
    
    # Compute PMF
    x_values = np.arange(1, max_count + 1)  # Start from 1 (active tokens only)
    counts = np.zeros(max_count, dtype=np.float32)
    
    for c in active_counts:
        counts[int(c) - 1] += 1
    
    pmf = counts / counts.sum()
    
    # Statistics
    mean_activation = float(active_counts.mean())
    std_activation = float(active_counts.std())
    
    # Normalized mean (0-1 scale, where 0 = all tokens activate 1 concept, 1 = all activate max)
    if max_count > 1:
        normalized_mean = (mean_activation - 1) / (max_count - 1)
    else:
        normalized_mean = 0.0
    
    # Gini coefficient
    gini = compute_gini(active_counts)
    
    # Entropy of PMF
    pmf_entropy = -sum(p * np.log2(p) for p in pmf if p > 0)
    
    stats = {
        "total_active_tokens": int(len(active_counts)),
        "mean_concepts_per_token": float(mean_activation),
        "std_concepts_per_token": float(std_activation),
        "median_concepts_per_token": float(np.median(active_counts)),
        "max_concepts_per_token": int(max_count),
        "normalized_mean": float(normalized_mean),
        "gini": float(gini),
        "pmf_entropy": float(pmf_entropy),
        "p1": float(pmf[0]) if len(pmf) > 0 else 0.0,  # Fraction activating exactly 1 concept
        "p2": float(pmf[1]) if len(pmf) > 1 else 0.0,  # Fraction activating exactly 2 concepts
        "p3+": float(pmf[2:].sum()) if len(pmf) > 2 else 0.0,  # Fraction activating 3+ concepts
    }
    
    return x_values, pmf, stats


def compute_gini(values: np.ndarray) -> float:
    """
    Compute Gini coefficient (0 = perfect equality, 1 = maximum inequality).
    """
    if len(values) == 0:
        return 0.0
    
    values = np.sort(values)
    n = len(values)
    cumulative = np.cumsum(values)
    
    return (2 * np.sum((np.arange(1, n + 1) * values)) - (n + 1) * cumulative[-1]) / (n * cumulative[-1])


# ==============================================================================
# Main Evaluation Loop
# ==============================================================================

@torch.no_grad()
def evaluate_sae_concepts(
    encoder,
    sae: GatedSAE,
    loader: DataLoader,
    device: torch.device,
    which_layer: str,
    token_l2_norm: bool = True,
    token_batch: int = 8192,
) -> Dict:
    """
    Evaluate SAE concepts on a dataset.
    
    Returns:
        results: Dictionary containing all metrics
    """
    encoder.eval()
    sae.eval()
    
    # Storage
    concept_class_counts = defaultdict(lambda: defaultdict(int))  # {concept: {class: count}}
    concept_image_sets = defaultdict(set)  # {concept: set of image indices}
    concept_image_class = {}  # {(concept, image_idx): class}
    
    all_token_concept_counts = []  # Number of concepts each token activates
    
    image_idx = 0
    total_tokens = 0
    active_tokens = 0
    
    autocast_kwargs = dict(device_type="cuda", enabled=torch.cuda.is_available())
    if torch.cuda.is_available():
        autocast_kwargs["dtype"] = torch.bfloat16
    
    pbar = tqdm(loader, desc="Evaluating SAE concepts")
    for batch in pbar:
        if batch is None:
            continue
        
        x_cpu, y_cpu, *_ = batch
        if x_cpu.numel() < 1:
            continue
        
        x = x_cpu.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        y = y_cpu.numpy()
        
        # Extract features
        with torch.amp.autocast(**autocast_kwargs):
            fmap = encoder.forward_feature_maps(x, which=which_layer)
        
        fmap = fmap.permute(0, 2, 3, 1).contiguous()  # (B,H,W,C)
        B, Hf, Wf, C = fmap.shape
        
        for b in range(B):
            class_idx = int(y[b])
            current_image_idx = image_idx + b
            
            # Extract tokens for this image
            tokens = fmap[b].view(Hf * Wf, C)  # (H*W, C)
            
            # Center and normalize
            tokens = tokens - tokens.mean(dim=0, keepdim=True)
            if token_l2_norm:
                tokens = F.normalize(tokens, dim=1)
            
            # Process in batches
            for s in range(0, tokens.size(0), token_batch):
                tok = tokens[s:s+token_batch].to(device)
                
                with torch.amp.autocast(**autocast_kwargs):
                    # Forward through SAE
                    recon, acts, gate_pre, _, _ = sae(tok)
                
                # Get active concepts (gated SAE: gate > 0)
                gate = (gate_pre > 0).float()  # (num_tokens, d_sae)
                
                # For each token
                for t_idx in range(gate.size(0)):
                    active_concepts = torch.where(gate[t_idx] > 0)[0].cpu().numpy()
                    num_active = len(active_concepts)
                    
                    total_tokens += 1
                    all_token_concept_counts.append(num_active)
                    
                    if num_active > 0:
                        active_tokens += 1
                        
                        for concept_id in active_concepts:
                            concept_id = int(concept_id)
                            # Count class for this concept-image pair (count image once per concept)
                            if current_image_idx not in concept_image_sets[concept_id]:
                                concept_image_sets[concept_id].add(current_image_idx)
                                concept_class_counts[concept_id][class_idx] += 1
                                concept_image_class[(concept_id, current_image_idx)] = class_idx
        
        image_idx += B
        
        pbar.set_postfix({
            "images": image_idx,
            "tokens": total_tokens,
            "active": f"{active_tokens/max(1,total_tokens)*100:.1f}%"
        })
    
    # Compute metrics
    token_concept_counts = np.array(all_token_concept_counts, dtype=np.int32)
    
    # Purity and Entropy
    purity, entropy, purity_stats = compute_concept_purity_and_entropy(
        concept_class_counts, min_activation_count=5
    )
    
    # Token Activation PMF
    x_vals, pmf, pmf_stats = compute_token_activation_pmf(token_concept_counts)
    
    results = {
        "total_images": image_idx,
        "total_tokens": total_tokens,
        "active_tokens": active_tokens,
        "active_token_ratio": active_tokens / max(1, total_tokens),
        "purity": purity,
        "entropy": entropy,
        "purity_stats": purity_stats,
        "pmf_x": x_vals,
        "pmf_y": pmf,
        "pmf_stats": pmf_stats,
        "concept_class_counts": dict(concept_class_counts),
    }
    
    return results


def print_evaluation_results(results: Dict, layer: str):
    """Print formatted evaluation results."""
    print("\n" + "=" * 70)
    print(f"SAE Concept Evaluation Results - Layer: {layer}")
    print("=" * 70)
    
    print(f"\n[Data Summary]")
    print(f"  Total images:     {results['total_images']:,}")
    print(f"  Total tokens:     {results['total_tokens']:,}")
    print(f"  Active tokens:    {results['active_tokens']:,} ({results['active_token_ratio']*100:.2f}%)")
    
    ps = results["purity_stats"]
    print(f"\n[Concept Purity] (K/N where K=max class count, N=total)")
    print(f"  Valid concepts:   {ps['valid_concepts']}/{ps['total_concepts']}")
    print(f"  High purity (>0.9): {ps['high_purity_concepts']}")
    print(f"  Class-specific breakdown: {ps['class_specific_breakdown']}")
    print(f"  Purity mean:      {ps['purity_mean']:.4f} ± {ps['purity_std']:.4f}")
    print(f"  Purity median:    {ps['purity_median']:.4f}")
    print(f"  Purity p90:       {ps['purity_p90']:.4f}")
    print(f"  Purity p95:       {ps['purity_p95']:.4f}")
    
    print(f"\n[Shannon Entropy] (0 = single class, {ps['max_entropy']:.2f} = uniform)")
    print(f"  Entropy mean:     {ps['entropy_mean']:.4f} ± {ps['entropy_std']:.4f}")
    print(f"  Entropy min:      {ps['entropy_min']:.4f}")
    
    ts = results["pmf_stats"]
    print(f"\n[Token Activation Distribution]")
    print(f"  Active tokens:    {ts['total_active_tokens']:,}")
    print(f"  Concepts/token:   {ts['mean_concepts_per_token']:.2f} ± {ts['std_concepts_per_token']:.2f}")
    print(f"  Median:           {ts['median_concepts_per_token']:.1f}")
    print(f"  Max:              {ts['max_concepts_per_token']}")
    print(f"  Normalized mean:  {ts['normalized_mean']:.4f} (0=sparse, 1=dense)")
    print(f"  Gini coefficient: {ts['gini']:.4f} (0=equal, 1=concentrated)")
    print(f"  PMF entropy:      {ts['pmf_entropy']:.4f}")
    print(f"  P(1 concept):     {ts['p1']*100:.1f}%")
    print(f"  P(2 concepts):    {ts['p2']*100:.1f}%")
    print(f"  P(3+ concepts):   {ts['p3+']*100:.1f}%")


def plot_evaluation_results(results: Dict, save_path: str, layer: str):
    """Create visualization plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Purity histogram
    ax1 = axes[0, 0]
    valid_purity = results["purity"][~np.isnan(results["purity"])]
    ax1.hist(valid_purity, bins=50, edgecolor='black', alpha=0.7)
    ax1.axvline(0.25, color='red', linestyle='--', label='Random (0.25)')
    ax1.axvline(results["purity_stats"]["purity_mean"], color='green', linestyle='-', label=f'Mean={results["purity_stats"]["purity_mean"]:.3f}')
    ax1.set_xlabel('Purity (K/N)')
    ax1.set_ylabel('Count')
    ax1.set_title(f'Concept Purity Distribution\n(Higher = More Class-Specific)')
    ax1.legend()
    
    # 2. Entropy histogram
    ax2 = axes[0, 1]
    valid_entropy = results["entropy"][~np.isnan(results["entropy"])]
    ax2.hist(valid_entropy, bins=50, edgecolor='black', alpha=0.7)
    ax2.axvline(np.log2(4), color='red', linestyle='--', label='Max=2.0 (uniform)')
    ax2.axvline(results["purity_stats"]["entropy_mean"], color='green', linestyle='-', label=f'Mean={results["purity_stats"]["entropy_mean"]:.3f}')
    ax2.set_xlabel('Shannon Entropy')
    ax2.set_ylabel('Count')
    ax2.set_title(f'Concept Entropy Distribution\n(Lower = More Class-Specific)')
    ax2.legend()
    
    # 3. Token activation PMF
    ax3 = axes[1, 0]
    x = results["pmf_x"]
    pmf = results["pmf_y"]
    ax3.bar(x, pmf, edgecolor='black', alpha=0.7)
    ax3.set_xlabel('Number of Concepts Activated')
    ax3.set_ylabel('Probability')
    ax3.set_title(f'Token Activation PMF\nMean={results["pmf_stats"]["mean_concepts_per_token"]:.2f}, Gini={results["pmf_stats"]["gini"]:.3f}')
    if len(x) > 10:
        ax3.set_xlim(0, min(20, x.max()))
    
    # 4. Purity vs Entropy scatter
    ax4 = axes[1, 1]
    valid_mask = ~(np.isnan(results["purity"]) | np.isnan(results["entropy"]))
    if valid_mask.sum() > 0:
        ax4.scatter(results["purity"][valid_mask], results["entropy"][valid_mask], alpha=0.3, s=5)
        ax4.set_xlabel('Purity')
        ax4.set_ylabel('Shannon Entropy')
        ax4.set_title('Purity vs Entropy (Each point = 1 concept)')
    
    plt.suptitle(f'SAE Concept Analysis - {layer}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved plots to {save_path}")


def save_results_csv(results: Dict, save_path: str):
    """Save detailed results to CSV."""
    # Per-concept results
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["concept_id", "purity", "entropy", "total_images"] + 
                   [f"class_{c}_count" for c in CLASS_ORDER])
        
        for concept_id in range(len(results["purity"])):
            purity = results["purity"][concept_id]
            entropy = results["entropy"][concept_id]
            
            if np.isnan(purity):
                continue
            
            class_counts = results["concept_class_counts"].get(concept_id, {})
            total = sum(class_counts.values())
            
            row = [concept_id, f"{purity:.4f}", f"{entropy:.4f}", total]
            for cls_idx in range(NUM_CLASSES):
                row.append(class_counts.get(cls_idx, 0))
            
            w.writerow(row)
    
    logger.info(f"Saved per-concept results to {save_path}")


# ==============================================================================
# Main Entry Point
# ==============================================================================

def main(args_list=None):
    args = resolve_paths(get_args(args_list))
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # Load data
    refs = load_all_sample_refs(args.shard_root)
    uid_to_refidx = build_uid_to_refidx(refs)
    
    # Load test split
    test_csv = os.path.join(args.save_dir, "test_split.csv")
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"test_split.csv not found: {test_csv}")
    
    uids = load_split_csv(test_csv)
    refidx = [uid_to_refidx[u] for u in uids if u in uid_to_refidx]
    
    bank = InMemoryTarBank(refs, refidx, args.img_size)
    ds = InMemorySixteenBitDataset(bank, list(range(len(refidx))), args.img_size, augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, 
                       num_workers=0, pin_memory=torch.cuda.is_available(),
                       collate_fn=collate_skip_none)
    
    logger.info(f"Loaded {len(refidx)} test images")
    
    # Load encoder
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
    encoder = model.encoder.to(device).eval()
    renorm_unit_per_out_channel_(encoder)
    
    # Find and evaluate SAE checkpoints
    sae_root = args.sae_save_dir
    layers = ["stage5_out", "refine_out"]
    
    for layer in layers:
        # Find Gated SAE checkpoint
        pattern = f"gated_sae_{layer}_*.pt"
        import glob
        ckpts = glob.glob(os.path.join(sae_root, pattern))
        
        if not ckpts:
            logger.warning(f"No Gated SAE checkpoint found for {layer}")
            continue
        
        # Use the first (or best) checkpoint
        ckpt_path = ckpts[0]
        for c in ckpts:
            if "BEST" in c:
                ckpt_path = c
                break
        
        logger.info(f"\nEvaluating: {ckpt_path}")
        
        # Load SAE
        sae = load_gated_sae(ckpt_path, device)
        
        # Update args for this layer
        args.which_layer = layer
        
        # Evaluate
        results = evaluate_sae_concepts(
            encoder, sae, loader, device, layer,
            token_l2_norm=True,
            token_batch=args.token_batch,
        )
        
        # Print results
        print_evaluation_results(results, layer)
        
        # Save plots
        plot_path = os.path.join(sae_root, f"concept_analysis_{layer}.png")
        plot_evaluation_results(results, plot_path, layer)
        
        # Save CSV
        csv_path = os.path.join(sae_root, f"concept_analysis_{layer}.csv")
        save_results_csv(results, csv_path)
        
        # Save summary
        summary_path = os.path.join(sae_root, f"concept_summary_{layer}.json")
        import json
        summary = {
            "layer": layer,
            "checkpoint": ckpt_path,
            "total_images": results["total_images"],
            "total_tokens": results["total_tokens"],
            "active_token_ratio": results["active_token_ratio"],
            "purity_stats": results["purity_stats"],
            "pmf_stats": results["pmf_stats"],
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
