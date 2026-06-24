# ==============================================================================
# Batch Activation Maximization Sweep
#
# Runs AM across multiple concepts × hyperparameter combos × seeds.
# Results are saved in organized directories matching step14 naming.
#
# Output structure:
#   output_dir/
#     concept_0018_LRRK2/
#       AM_decay1.0_jitter8/
#         seed_42.png
#         seed_43.png
#         seed_44.png
#         seed_42.npz
#         ...
#     concept_0037_SNCA/
#       ...
#
# Usage (Colab):
#   import sys
#   sys.argv = [
#       "batch_am_sweep",
#       "--sae_ckpt", "/path/to/sae.pt",
#       "--model_state_path", "/path/to/best_model.pt",
#       "--concepts", "0018:LRRK2,0037:SNCA,0152:GBA",
#       "--output_dir", "/path/to/am_sweep_output",
#       "--decay_powers", "1.0,1.5,2.0",
#       "--jitter_pxs", "4,8,12",
#       "--seeds", "42,43,44",
#   ]
#   from activation_maximization.batch_am_sweep import main
#   main()
# ==============================================================================

import argparse
import itertools
import os
import re
import shutil
import sys

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

_IN_COLAB = "google.colab" in sys.modules
if not _IN_COLAB:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from activation_maximization_file.activation_maximization import (
    get_spatial_activation_map, run_activation_maximization)
from sae_project.step02_logging_utils import get_logger
from sae_project.step05_model_encoder import (SupMoCoModel, parse_int_list,
                                              renorm_unit_per_out_channel_,
                                              robust_load_state_dict)
from sae_project.step06_gated_sae import GatedSAE

logger = get_logger("batch_am_sweep")


# ==============================================================================
# Visualization (R, G, B, Composite, Heatmap) — single image
# ==============================================================================
def visualize_am_with_heatmap(
    img_tensor,  # (3, H, W) float tensor
    neuron_idx,
    activation,
    heatmap,  # (H, W) numpy array
    output_path,
    dpi=150,
):
    """
    5-panel visualization: R, G, B, Composite, Spatial Heatmap.
    """
    img = img_tensor.numpy()  # (3, H, W)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    channel_names = ["TMRM (Red)", "Lysotracker (Green)", "Hoechst (Blue)"]
    channel_cmaps = ["Reds", "Greens", "Blues"]

    for i in range(3):
        ch = img[i]
        ch_min, ch_max = ch.min(), ch.max()
        if ch_max - ch_min > 1e-8:
            ch_disp = (ch - ch_min) / (ch_max - ch_min)
        else:
            ch_disp = np.zeros_like(ch)
        axes[i].imshow(ch_disp, cmap=channel_cmaps[i], vmin=0, vmax=1)
        axes[i].set_title(
            f"{channel_names[i]}\n[{ch_min:.3f}, {ch_max:.3f}]", fontsize=9
        )
        axes[i].axis("off")

    # Composite
    rgb = np.zeros((img.shape[1], img.shape[2], 3), dtype=np.float32)
    for i in range(3):
        ch = img[i]
        ch_min, ch_max = ch.min(), ch.max()
        if ch_max - ch_min > 1e-8:
            rgb[:, :, i] = (ch - ch_min) / (ch_max - ch_min)

    axes[3].imshow(rgb)
    axes[3].set_title(f"Composite\nact={activation:.4f}", fontsize=9)
    axes[3].axis("off")

    # Heatmap overlay
    axes[4].imshow(rgb, alpha=0.3)
    im = axes[4].imshow(
        heatmap, cmap="hot", alpha=0.7, vmin=0, vmax=heatmap.max() + 1e-8
    )
    axes[4].set_title(
        f"Spatial Activation\n[{heatmap.min():.3f}, {heatmap.max():.3f}]", fontsize=9
    )
    axes[4].axis("off")
    fig.colorbar(im, ax=axes[4], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"SAE Neuron {neuron_idx:04d} — act={activation:.4f}",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# Argument Parser
# ==============================================================================
def get_args():
    p = argparse.ArgumentParser(
        description="Batch AM sweep: concepts × hyperparameters × seeds"
    )
    # Model
    p.add_argument("--sae_ckpt", type=str, required=True)
    p.add_argument("--model_state_path", type=str, required=True)
    p.add_argument(
        "--which_layer",
        type=str,
        default="",
        help="Encoder layer (default: from SAE ckpt)",
    )

    # Target concepts: two modes
    #  1) --concept_dir: auto-discover from step14 output (scan concept_XXXX_CLASS dirs)
    #  2) --concepts: manual "id:class,id:class,..." e.g. "0018:LRRK2,0037:SNCA"
    p.add_argument(
        "--concept_dir",
        type=str,
        default="",
        help="Path to step14 output dir to auto-discover concepts "
        "(scans concept_XXXX_CLASS folders). AM results saved inside each.",
    )
    p.add_argument(
        "--concepts",
        type=str,
        default="",
        help="Manual: comma-separated id:class pairs, e.g. '0018:LRRK2,0037:SNCA'",
    )

    # Output
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument(
        "--drive_backup_dir",
        type=str,
        default="",
        help="Google Drive path to copy results to (e.g. /content/drive/MyDrive/AM_results)",
    )

    # Optimization
    p.add_argument("--steps", type=int, default=512)
    p.add_argument("--lr", type=float, default=10.0)
    p.add_argument("--img_size", type=int, default=128)

    # Sweep parameters (comma-separated lists)
    p.add_argument(
        "--decay_powers",
        type=str,
        default="1.0",
        help="Comma-separated decay_power values to sweep",
    )
    p.add_argument(
        "--jitter_pxs",
        type=str,
        default="8",
        help="Comma-separated jitter_px values to sweep",
    )
    p.add_argument(
        "--seeds", type=str, default="42,43,44", help="Comma-separated seed values"
    )

    # Fixed regularization
    p.add_argument("--l2_weight", type=float, default=0.0)
    p.add_argument("--l1_weight", type=float, default=0.0)
    p.add_argument("--rotate_deg", type=float, default=15.0)
    p.add_argument("--scale_lo", type=float, default=0.9)
    p.add_argument("--scale_hi", type=float, default=1.1)
    p.add_argument("--init_std", type=float, default=0.01)

    # Encoder architecture
    p.add_argument("--blocks", type=str, default="2,2,2,3")
    p.add_argument("--dilations", type=str, default="1,1,1,1")
    p.add_argument("--refine_blocks", type=int, default=1)
    p.add_argument("--ckpt_segments", type=int, default=0)
    p.add_argument("--embed_dim", type=int, default=512)
    p.add_argument("--proj_layers", type=int, default=2)
    p.add_argument("--proj_hidden", type=int, default=2048)
    p.add_argument("--proj_bn", action="store_true")
    p.add_argument("--proj_dropout", type=float, default=0.0)

    p.add_argument("--dpi", type=int, default=150)

    return p.parse_args()


# ==============================================================================
# Parse concept string "0018:LRRK2,0037:SNCA" → [(18, "LRRK2"), (37, "SNCA")]
# ==============================================================================
def parse_concepts(s):
    """Parse 'id:class,id:class,...' or 'id,id,...' (no class label)."""
    result = []
    for item in s.split(","):
        item = item.strip()
        if ":" in item:
            idx_str, cls = item.split(":", 1)
            result.append((int(idx_str), cls.strip()))
        else:
            result.append((int(item), ""))
    return result


def parse_float_list(s):
    return [float(x.strip()) for x in s.split(",")]


def parse_int_list_csv(s):
    return [int(x.strip()) for x in s.split(",")]


# ==============================================================================
# Auto-discover concepts from step14 output directory
# ==============================================================================
_CONCEPT_DIR_RE = re.compile(r"concept_(\d{4})(?:_(.+))?$")


def auto_discover_concepts(concept_dir):
    """
    Scan concept_dir for subdirectories matching 'concept_XXXX' or 'concept_XXXX_CLASS'.
    Returns list of (neuron_idx, class_label, full_dir_path).
    """
    results = []
    for name in sorted(os.listdir(concept_dir)):
        full_path = os.path.join(concept_dir, name)
        if not os.path.isdir(full_path):
            continue
        m = _CONCEPT_DIR_RE.match(name)
        if m:
            neuron_idx = int(m.group(1))
            class_label = m.group(2) or ""
            results.append((neuron_idx, class_label, full_path))
    return results


# ==============================================================================
# Main
# ==============================================================================
def main():
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Parse concepts: auto-discover or manual ──
    if args.concept_dir:
        discovered = auto_discover_concepts(args.concept_dir)
        if not discovered:
            raise ValueError(f"No concept_XXXX directories found in {args.concept_dir}")
        # (neuron_idx, class_label, existing_dir_path)
        concepts_with_dirs = discovered
        logger.info(
            f"Auto-discovered {len(concepts_with_dirs)} concepts from {args.concept_dir}"
        )
        for idx, cls, path in concepts_with_dirs:
            logger.info(f"  concept_{idx:04d}_{cls}" if cls else f"  concept_{idx:04d}")
    elif args.concepts:
        parsed = parse_concepts(args.concepts)
        # Create dirs in output_dir
        concepts_with_dirs = []
        for idx, cls in parsed:
            if cls:
                cdir = os.path.join(args.output_dir, f"concept_{idx:04d}_{cls}")
            else:
                cdir = os.path.join(args.output_dir, f"concept_{idx:04d}")
            concepts_with_dirs.append((idx, cls, cdir))
        logger.info(f"Manual concepts: {len(concepts_with_dirs)}")
    else:
        raise ValueError("Either --concept_dir or --concepts must be specified")

    decay_powers = parse_float_list(args.decay_powers)
    jitter_pxs = parse_int_list_csv(args.jitter_pxs)
    seeds = parse_int_list_csv(args.seeds)

    logger.info(f"Sweep: decay_powers={decay_powers}, jitter_pxs={jitter_pxs}")
    logger.info(f"Seeds: {seeds}")

    hp_combos = list(zip(decay_powers, jitter_pxs))
    if len(decay_powers) != len(jitter_pxs):
        logger.warning(
            f"decay_powers ({len(decay_powers)}) and jitter_pxs ({len(jitter_pxs)}) "
            f"have different lengths! Using zip (shorter list determines count)."
        )
    total_runs = len(concepts_with_dirs) * len(hp_combos) * len(seeds)
    logger.info(
        f"Total runs: {len(concepts_with_dirs)} concepts × {len(hp_combos)} HP combos × {len(seeds)} seeds = {total_runs}"
    )

    # ── Load SAE ──
    logger.info(f"\n{'='*60}")
    logger.info(f"Loading SAE: {args.sae_ckpt}")
    ckpt = torch.load(args.sae_ckpt, map_location="cpu", weights_only=False)
    ckpt_args = ckpt["args"]

    sae = GatedSAE(
        d_in=ckpt_args.get("d_in", 512),
        d_sae=ckpt_args.get("d_sae", 4096),
        tie_weights=ckpt_args.get("tie_gate_weights", False),
        aux_k=ckpt_args.get("aux_k", 32),
    )
    sae.load_state_dict(ckpt["sae"])
    sae.to(device).eval()
    for p in sae.parameters():
        p.requires_grad_(False)

    which_layer = args.which_layer or ckpt_args.get("which_layer", "refine_out")
    d_sae = sae.d_sae
    logger.info(f"SAE: d_sae={d_sae}, layer={which_layer}")

    # ── Load encoder ──
    logger.info(f"\n{'='*60}")
    logger.info("Loading encoder")
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
    robust_load_state_dict(model, sd, strict=True)
    encoder = model.encoder
    renorm_unit_per_out_channel_(encoder)
    encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    del model, sd
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Run sweep ──
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    run_count = 0

    for neuron_idx, class_label, concept_dir in concepts_with_dirs:
        if neuron_idx >= d_sae:
            logger.warning(f"Neuron {neuron_idx} >= d_sae={d_sae}, skipping")
            continue

        for decay_power, jitter_px in hp_combos:
            # Hyperparameter subdirectory
            hp_name = f"AM_decay{decay_power}_jitter{jitter_px}"
            hp_dir = os.path.join(concept_dir, hp_name)
            os.makedirs(hp_dir, exist_ok=True)

            for seed in seeds:
                run_count += 1

                # Resume: skip if already done
                npz_path = os.path.join(hp_dir, f"seed_{seed}.npz")
                if os.path.exists(npz_path):
                    logger.info(
                        f"  [{run_count}/{total_runs}] SKIP (exists): {npz_path}"
                    )
                    continue

                logger.info(f"\n{'='*60}")
                logger.info(
                    f"[{run_count}/{total_runs}] Neuron {neuron_idx:04d} "
                    f"({class_label}) | decay={decay_power}, jitter={jitter_px} "
                    f"| seed={seed}"
                )
                logger.info("=" * 60)

                # Run AM
                img, act = run_activation_maximization(
                    encoder=encoder,
                    sae=sae,
                    which_layer=which_layer,
                    target_neuron_idx=neuron_idx,
                    device=device,
                    img_size=args.img_size,
                    steps=args.steps,
                    lr=args.lr,
                    l2_weight=args.l2_weight,
                    l1_weight=args.l1_weight,
                    decay_power=decay_power,
                    jitter_px=jitter_px,
                    rotate_deg=args.rotate_deg,
                    scale_range=(args.scale_lo, args.scale_hi),
                    init_std=args.init_std,
                    seed=seed,
                )

                logger.info(f"  activation = {act:.4f}")

                # Compute spatial activation heatmap
                heatmap = get_spatial_activation_map(
                    img,
                    encoder,
                    sae,
                    which_layer,
                    neuron_idx,
                    device,
                )

                # Save visualization
                png_path = os.path.join(hp_dir, f"seed_{seed}.png")
                visualize_am_with_heatmap(
                    img,
                    neuron_idx,
                    act,
                    heatmap,
                    png_path,
                    dpi=args.dpi,
                )

                # Save raw data
                npz_path = os.path.join(hp_dir, f"seed_{seed}.npz")
                np.savez_compressed(
                    npz_path,
                    img=img.numpy(),
                    heatmap=heatmap,
                    neuron_idx=neuron_idx,
                    activation=act,
                    decay_power=decay_power,
                    jitter_px=jitter_px,
                    seed=seed,
                )

                logger.info(f"  Saved: {png_path}")

    logger.info(f"\n{'='*60}")
    logger.info(f"Batch AM sweep complete! {run_count} runs")
    logger.info(f"Output: {args.output_dir}")
    logger.info("=" * 60)

    # ── Backup to Google Drive ──
    if args.drive_backup_dir:
        logger.info(f"\nBacking up to Drive: {args.drive_backup_dir}")
        if os.path.exists(args.drive_backup_dir):
            shutil.rmtree(args.drive_backup_dir)
        shutil.copytree(args.output_dir, args.drive_backup_dir)

        # Verify: count files in source and dest
        src_files = sum(len(f) for _, _, f in os.walk(args.output_dir))
        dst_files = sum(len(f) for _, _, f in os.walk(args.drive_backup_dir))
        if src_files == dst_files:
            logger.info(f"  ✓ Backup verified: {dst_files} files copied")
        else:
            logger.warning(f"  ⚠ File count mismatch! src={src_files}, dst={dst_files}")


if __name__ == "__main__":
    main()
