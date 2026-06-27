# ==============================================================================
# Activation Maximization for SAE Neurons
#
# Generates synthetic input images that maximally activate chosen SAE neurons.
# Pipeline: learnable_input → CNN encoder → GAP-scalar norm → batch centering
#           → token L2 norm → SAE → target neuron activation (maximize)
#
# Regularizations:
#   - Spatial jitter (translation robustness)
#   - Random rotation (±15°)
#   - Random scale (0.9–1.1×)
#   - Total Variation (TV) loss for spatial smoothness
#   - L2 pixel norm decay
#   - Gaussian blur every N steps (frequency penalization)
#   - Optional: channel-decorrelation parameterization
#
# Usage (Colab):
#   import sys
#   sys.argv = [
#       "activation_maximization",
#       "--sae_ckpt", "/path/to/sae.pt",
#       "--model_state_path", "/path/to/best_model.pt",
#       "--concepts", "0018,0037,0152",
#       "--output_dir", "/path/to/am_output",
#       "--steps", "512",
#   ]
#   from Activation_maximization.activation_maximization import main
#   main()
# ==============================================================================


## AM할때 gated SAE는 eval할때 gradient가 0이 된다.
# eval 모드

# eval 모드 (self.training = False)
# gate = (gate_pre > 0).float()  # ← step function, gradient = 0!
# acts = gate * mag


# 피처맵 bilinear interpolation할때 [0, 1]로 정규화 하지 않는다.
# # step14 방식
# a01 = minmax_normalize(spatial_act)   # [0, 1]로 정규화
# heatmap_rgb = apply_colormap_01(a01, "jet")  # jet colormap
# overlay = create_overlay(base_rgb, heatmap_rgb, alpha=0.5)


import argparse
import os
import sys

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_IN_COLAB = "google.colab" in sys.modules
if not _IN_COLAB:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_train.logging_utils import get_logger
from model_train.model_encoder import (SupMoCoModel, parse_int_list,
                                              renorm_unit_per_out_channel_,
                                              robust_load_state_dict)
from sae_project.step06_gated_sae import GatedSAE

logger = get_logger("activation_max")


# ==============================================================================
# Regularizations
# ==============================================================================
def total_variation_loss(x):
    """Total Variation: encourages spatial smoothness."""
    tv_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).pow(2).mean()
    tv_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).pow(2).mean()
    return tv_h + tv_w


def random_jitter(x, max_px=8):
    """Random spatial translation."""
    pad = max_px
    x_pad = F.pad(x, [pad] * 4, mode="reflect")
    h, w = x.shape[2], x.shape[3]
    dy = torch.randint(0, 2 * pad + 1, (1,)).item()
    dx = torch.randint(0, 2 * pad + 1, (1,)).item()
    return x_pad[:, :, dy : dy + h, dx : dx + w]


def random_rotate(x, max_deg=15.0):
    """Random rotation by small angle."""
    angle = (torch.rand(1).item() * 2 - 1) * max_deg
    theta = torch.tensor(angle * np.pi / 180.0)
    cos_a, sin_a = torch.cos(theta), torch.sin(theta)
    # Affine matrix for rotation
    rot = torch.tensor(
        [
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
        ],
        dtype=x.dtype,
        device=x.device,
    ).unsqueeze(0)
    grid = F.affine_grid(rot, x.shape, align_corners=False)
    return F.grid_sample(
        x, grid, align_corners=False, mode="bilinear", padding_mode="reflection"
    )


def random_scale(x, lo=0.9, hi=1.1):
    """Random uniform scaling."""
    s = torch.empty(1).uniform_(lo, hi).item()
    theta = torch.tensor(
        [
            [s, 0, 0],
            [0, s, 0],
        ],
        dtype=x.dtype,
        device=x.device,
    ).unsqueeze(0)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(
        x, grid, align_corners=False, mode="bilinear", padding_mode="reflection"
    )


def gaussian_blur(x, kernel_size=3, sigma=1.0):
    """Gaussian blur for frequency regularization."""
    C = x.shape[1]
    # Create 1D Gaussian kernel
    coords = torch.arange(kernel_size, dtype=x.dtype, device=x.device)
    coords -= kernel_size // 2
    g = torch.exp(-coords.pow(2) / (2 * sigma**2))
    g /= g.sum()
    # Separable 2D kernel
    kernel = g.view(1, 1, -1, 1) * g.view(1, 1, 1, -1)  # (1, 1, k, k)
    kernel = kernel.expand(C, 1, -1, -1)
    pad = kernel_size // 2
    return F.conv2d(x, kernel, padding=pad, groups=C)


# ==============================================================================
# Soft-gated SAE forward (differentiable for AM optimization)
# ==============================================================================
def soft_gated_sae_forward(sae, tokens, temperature=10.0):
    """
    Differentiable SAE forward using soft sigmoid gate instead of hard step.

    GatedSAE eval mode uses: gate = (gate_pre > 0).float()  ← zero gradient!
    Here we replace with:    gate = sigmoid(gate_pre * T)   ← smooth gradient

    Args:
        sae: GatedSAE model
        tokens: (N, d_in) input tokens
        temperature: sigmoid sharpness (higher = closer to hard gate)

    Returns:
        acts: (N, d_sae) soft-gated activations
    """
    # Gate pre-activations
    gate_pre = tokens @ sae.W_gate_effective + sae.b_gate  # (N, d_sae)

    # Magnitude pre-activations
    mag_pre = tokens @ sae.W_mag + sae.b_mag  # (N, d_sae)

    # Soft gate: sigmoid with temperature (differentiable everywhere)
    gate = torch.sigmoid(gate_pre * temperature)

    # Magnitude: ReLU activation
    mag = F.relu(mag_pre)

    # Final activation: gate ⊙ magnitude
    acts = gate * mag  # (N, d_sae)

    return acts


# ==============================================================================
# Gradient-enabled encoder forward (bypass @torch.no_grad on forward_feature_maps)
# ==============================================================================
def encoder_forward_with_grad(encoder, x, which_layer):
    """
    forward_feature_maps()에 @torch.no_grad()가 걸려있어서
    AM에서 gradient가 흐르지 않음.
    encoder의 레이어를 직접 호출하여 gradient를 유지.
    """
    x = encoder.stem(x)
    x = encoder.stage2(x)
    x = encoder.stage3(x)
    x = encoder.stage4(x)
    if which_layer == "stage5_mid":
        for block in list(encoder.stage5.blocks)[:-1]:
            x = block(x)
        return x
    x = encoder.stage5(x)
    if which_layer == "stage5_out":
        return x
    x = encoder.refine(x)
    if which_layer == "refine_out":
        return x
    raise ValueError(f"Unknown which_layer={which_layer}")


# ==============================================================================
# Forward pass: Input → Encoder → SAE → target neuron activation
# ==============================================================================
def forward_to_sae_neuron(
    x_input,  # (1, 3, H, W), requires_grad
    encoder,
    sae,
    which_layer,
    target_neuron_idx,  # int or list[int]
    pooling="gap",  # "gap" or "max"
    temperature=10.0,  # soft gate sharpness
    restore_token_norm=True,  # multiply back per-token L2 norms
):
    """
    Full forward pass from pixel input to SAE neuron activation.
    Uses soft-gated SAE forward for differentiable optimization.

    Pipeline:
      1. encoder layers (gradient enabled) → fmap
      2. GAP-scalar normalization
      3. Save per-token L2 norms
      4. Token L2 normalization
      5. Soft-gated SAE forward → acts
      6. Restore token norms: acts × ||token||  (pixel magnitude matters!)
      7. Pool (GAP or max) over spatial → target neuron activation

    Returns: scalar activation for optimization
    """
    # Gradient-enabled encoder forward (bypass @torch.no_grad)
    fmap = encoder_forward_with_grad(encoder, x_input, which_layer)

    # GAP-scalar normalization
    gap = fmap.mean(dim=(2, 3))
    gap_norm = gap.norm(dim=1, keepdim=True).view(1, 1, 1, 1).clamp_min(1e-12)
    fmap = fmap / gap_norm

    # (B, C, H, W) → (B, H, W, C) → flatten to tokens
    fmap = fmap.permute(0, 2, 3, 1).contiguous()
    C = fmap.shape[-1]

    flat_tokens = fmap.view(-1, C)
    # NOTE: Per-image centering 제거 (AM 전용)
    # extraction에서는 배치(~64장) 전체 토큰의 평균을 빼므로 개별 이미지 DC 성분 유지됨
    # AM에서 1장만 넣으면 mean(dim=0) = 이 이미지의 공간 평균 = GAP 벡터 자체이므로
    # DC 성분이 완전히 제거되어 SAE가 학습 시 보던 분포와 달라짐

    # Save per-token L2 norms before normalization
    token_l2_norms = flat_tokens.norm(dim=1, keepdim=True).clamp_min(1e-12)

    # Token L2 normalization
    flat_tokens = F.normalize(flat_tokens, dim=1, eps=1e-12)

    # Soft-gated SAE forward (differentiable!)
    acts = soft_gated_sae_forward(sae, flat_tokens, temperature=temperature)

    # Restore per-token L2 norms → pixel magnitude affects activation
    if restore_token_norm:
        acts = acts * token_l2_norms  # (H*W, d_sae)

    # Target neuron(s)
    if isinstance(target_neuron_idx, (list, tuple)):
        target_acts = acts[:, target_neuron_idx]  # (H*W, n_targets)
    else:
        target_acts = acts[:, target_neuron_idx].unsqueeze(1)  # (H*W, 1)

    # Pool over spatial
    if pooling == "gap":
        activation = target_acts.mean(dim=0)  # (n_targets,)  ## = sum / H*W
    else:  # max
        activation = target_acts.max(dim=0).values

    return activation.sum()


# ==============================================================================
# Spatial Activation Map (for heatmap visualization)
# ==============================================================================
@torch.no_grad()
def get_spatial_activation_map(
    img_tensor,  # (3, H, W) cpu float tensor (AM result)
    encoder,
    sae,
    which_layer,
    target_neuron_idx,
    device,
    temperature=10.0,
    restore_token_norm=True,
):
    """
    Get per-spatial-location SAE neuron activation for the AM result image.
    Returns heatmap upsampled to input image size via bilinear interpolation.

    Returns:
        heatmap: (H_img, W_img) numpy array, bilinear-upsampled activation map
    """
    x = img_tensor.unsqueeze(0).to(device)  # (1, 3, H, W)
    H_img, W_img = x.shape[2], x.shape[3]

    # Encoder forward (no grad needed for visualization)
    fmap = encoder.forward_feature_maps(x, which=which_layer)
    H_feat, W_feat = fmap.shape[2], fmap.shape[3]

    # GAP-scalar normalization
    gap = fmap.mean(dim=(2, 3))
    gap_norm = gap.norm(dim=1, keepdim=True).view(1, 1, 1, 1).clamp_min(1e-12)
    fmap = fmap / gap_norm

    # Reshape to tokens
    fmap = fmap.permute(0, 2, 3, 1).contiguous()
    C = fmap.shape[-1]
    flat_tokens = fmap.view(-1, C)

    # Token L2 norms
    token_l2_norms = flat_tokens.norm(dim=1, keepdim=True).clamp_min(1e-12)

    # Token L2 normalization
    flat_tokens = F.normalize(flat_tokens, dim=1, eps=1e-12)

    # Soft-gated SAE forward
    acts = soft_gated_sae_forward(sae, flat_tokens, temperature=temperature)

    # Restore token norms
    if restore_token_norm:
        acts = acts * token_l2_norms

    # Target neuron spatial activations
    spatial_acts = acts[:, target_neuron_idx]  # (H_feat*W_feat,)
    spatial_map = spatial_acts.view(1, 1, H_feat, W_feat)  # (1, 1, H, W)

    # Bilinear interpolation to input image size
    heatmap = F.interpolate(
        spatial_map, size=(H_img, W_img), mode="bilinear", align_corners=False
    )
    return heatmap.squeeze().cpu().numpy()  # (H_img, W_img)


# ==============================================================================
# Fourier Preconditioning: 1/f decay mask
# ==============================================================================
def make_fourier_decay_mask(h, w, decay_power=1.0, device="cpu"):
    """
    Create 1/f^decay_power mask in frequency domain.
    Natural images have power spectrum ~ 1/f, so this makes optimization
    stay within the natural image manifold.

    Args:
        h, w: spatial dimensions
        decay_power: higher = stronger high-freq suppression (1.0 = natural images)
    Returns:
        mask: (1, 1, h, w//2+1) for rfft2 output shape
    """
    fy = torch.fft.fftfreq(h, device=device).view(-1, 1)  # (h, 1)
    fx = torch.fft.rfftfreq(w, device=device).view(1, -1)  # (1, w//2+1)
    freq = (fx**2 + fy**2).sqrt().clamp_min(1.0 / max(h, w))
    mask = 1.0 / freq.pow(decay_power)
    mask = mask / mask.max()  # normalize to [0, 1]
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, h, w//2+1)


def fourier_to_image(coeffs, decay_mask):
    """
    Convert learnable Fourier coefficients → pixel image.
    coeffs: (1, 3, h, w//2+1) complex tensor
    decay_mask: (1, 1, h, w//2+1) real tensor
    Returns: (1, 3, h, w) real image
    """
    # Apply 1/f decay to suppress high frequencies
    scaled = coeffs * decay_mask
    # Inverse FFT → pixel space
    return torch.fft.irfft2(scaled, s=(coeffs.shape[2], (coeffs.shape[3] - 1) * 2))


# ==============================================================================
# Main Optimization Loop — Fourier Preconditioning
# ==============================================================================
def run_activation_maximization(
    encoder,
    sae,
    which_layer,
    target_neuron_idx,
    device,
    img_size=128,
    steps=512,
    lr=0.05,
    # Regularization weights
    l2_weight=1e-4,
    l1_weight=1e-4,
    decay_power=1.0,
    # Transformation robustness
    jitter_px=8,
    rotate_deg=15.0,
    scale_range=(0.9, 1.1),
    # Initialization
    init_std=0.01,
    seed=42,
    # Legacy (ignored, kept for CLI compatibility)
    tv_weight=0.0,
    blur_every=0,
    blur_sigma=0.0,
):
    """
    Run activation maximization for a single SAE neuron.
    Uses Fourier preconditioning: optimizes in frequency space with 1/f decay.
    """
    torch.manual_seed(seed)

    # ── Fourier preconditioning setup ──
    # Learnable parameter = Fourier coefficients (complex)
    rfft_shape = (1, 3, img_size, img_size // 2 + 1)
    fourier_coeffs = (
        torch.randn(*rfft_shape, dtype=torch.cfloat, device=device) * init_std
    )
    fourier_coeffs = fourier_coeffs.requires_grad_(True)

    # 1/f decay mask (precomputed, not learnable)
    decay_mask = make_fourier_decay_mask(
        img_size, img_size, decay_power=decay_power, device=device
    )

    optimizer = torch.optim.Adam([fourier_coeffs], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

    best_act = -float("inf")
    best_img = None

    for step in range(steps):
        optimizer.zero_grad()

        # ── Convert Fourier → pixel ──
        img_raw = fourier_to_image(fourier_coeffs, decay_mask)

        # ── Pixel clipping: match SafeInstanceNormalize range ──
        # 학습 때 IN 적용하여 ~[-3, 3] 분포 → clamp로 OOD 방지
        img = torch.clamp(img_raw, -3.5, 3.5)

        # ── Transformation robustness ──
        x = img

        # Random jitter
        if jitter_px > 0:
            x = random_jitter(x, max_px=jitter_px)

        # Random rotation
        if rotate_deg > 0:
            x = random_rotate(x, max_deg=rotate_deg)

        # Random scale
        if scale_range[0] < 1.0 or scale_range[1] > 1.0:
            x = random_scale(x, lo=scale_range[0], hi=scale_range[1])

        # ── Forward pass (with token norm restoration) ──
        activation = forward_to_sae_neuron(
            x,
            encoder,
            sae,
            which_layer,
            target_neuron_idx,
            pooling="gap",
            restore_token_norm=True,
        )

        # ── Loss = maximize activation + regularizations ──
        loss_act = -activation
        loss_l2 = l2_weight * fourier_coeffs.abs().pow(2).mean()
        loss_l1 = l1_weight * img.abs().mean()  # sparse dark backgrounds
        loss = loss_act + loss_l2 + loss_l1

        loss.backward()
        optimizer.step()
        scheduler.step()

        # Track best (convert to pixel for saving)
        act_val = activation.item()
        if act_val > best_act:
            best_act = act_val
            best_img = img.detach().clone()

        if (step + 1) % 100 == 0 or step == 0:
            img_range = f"[{img.min().item():.3f}, {img.max().item():.3f}]"
            logger.info(
                f"  step {step+1:4d}/{steps}: act={act_val:.4f}, "
                f"l2={loss_l2.item():.6f}, l1={loss_l1.item():.6f}, "
                f"px_range={img_range}, lr={scheduler.get_last_lr()[0]:.5f}"
            )

    return best_img.squeeze(0).cpu(), best_act


# ==============================================================================
# Visualization
# ==============================================================================
def visualize_am_result(
    img_tensor,  # (3, H, W) float tensor
    neuron_idx,
    activation,
    output_path,
    heatmap=None,  # (H, W) numpy array, optional spatial activation map
    dpi=150,
):
    """
    Visualize AM result. Shows per-channel, composite, and spatial heatmap.
    img_tensor is raw (un-normalized). We scale for display.
    """
    img = img_tensor.numpy()  # (3, H, W)

    n_cols = 5 if heatmap is not None else 4
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

    channel_names = ["TMRM (Red)", "Lysotracker (Green)", "Hoechst (Blue)"]
    channel_cmaps = ["Reds", "Greens", "Blues"]

    for i in range(3):
        ch = img[i]
        # Normalize to [0, 1] for display
        ch_min, ch_max = ch.min(), ch.max()
        if ch_max - ch_min > 1e-8:
            ch_disp = (ch - ch_min) / (ch_max - ch_min)
        else:
            ch_disp = np.zeros_like(ch)

        axes[i].imshow(ch_disp, cmap=channel_cmaps[i], vmin=0, vmax=1)
        axes[i].set_title(
            f"{channel_names[i]}\n[{ch_min:.3f}, {ch_max:.3f}]", fontsize=10
        )
        axes[i].axis("off")

    # Composite: merge channels as RGB
    rgb = np.zeros((img.shape[1], img.shape[2], 3), dtype=np.float32)
    for i in range(3):
        ch = img[i]
        ch_min, ch_max = ch.min(), ch.max()
        if ch_max - ch_min > 1e-8:
            rgb[:, :, i] = (ch - ch_min) / (ch_max - ch_min)

    axes[3].imshow(rgb)
    axes[3].set_title(f"Composite\nact={activation:.4f}", fontsize=10)
    axes[3].axis("off")

    # Spatial activation heatmap
    if heatmap is not None:
        axes[4].imshow(rgb, alpha=0.3)  # dim composite background
        im = axes[4].imshow(
            heatmap, cmap="hot", alpha=0.7, vmin=0, vmax=heatmap.max() + 1e-8
        )
        axes[4].set_title(
            f"Spatial Activation\n[{heatmap.min():.3f}, {heatmap.max():.3f}]",
            fontsize=10,
        )
        axes[4].axis("off")
        fig.colorbar(im, ax=axes[4], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Activation Maximization — SAE Neuron {neuron_idx:04d}",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    logger.info(f"  Saved: {output_path}")


# ==============================================================================
# Multi-neuron grid
# ==============================================================================
def visualize_am_grid(results, output_path, dpi=200):
    """Grid of AM results: one row per neuron (R, G, B, composite)."""
    n = len(results)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    channel_names = ["PI (R)", "AnnV (G)", "Hoechst (B)"]
    channel_cmaps = ["Reds", "Greens", "Blues"]

    for row, (neuron_idx, img_tensor, act) in enumerate(results):
        img = img_tensor.numpy()
        for i in range(3):
            ch = img[i]
            ch_min, ch_max = ch.min(), ch.max()
            if ch_max - ch_min > 1e-8:
                ch_disp = (ch - ch_min) / (ch_max - ch_min)
            else:
                ch_disp = np.zeros_like(ch)

            axes[row, i].imshow(ch_disp, cmap=channel_cmaps[i], vmin=0, vmax=1)
            if row == 0:
                axes[row, i].set_title(channel_names[i], fontsize=10)
            axes[row, i].set_ylabel(
                f"N{neuron_idx:04d}\nact={act:.3f}",
                fontsize=9,
                rotation=0,
                labelpad=60,
                va="center",
            )
            axes[row, i].set_xticks([])
            axes[row, i].set_yticks([])

        # Composite
        rgb = np.zeros((img.shape[1], img.shape[2], 3), dtype=np.float32)
        for i in range(3):
            ch = img[i]
            ch_min, ch_max = ch.min(), ch.max()
            if ch_max - ch_min > 1e-8:
                rgb[:, :, i] = (ch - ch_min) / (ch_max - ch_min)
        axes[row, 3].imshow(rgb)
        if row == 0:
            axes[row, 3].set_title("Composite", fontsize=10)
        axes[row, 3].set_xticks([])
        axes[row, 3].set_yticks([])

    fig.suptitle("SAE Neuron Activation Maximization", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    logger.info(f"  Saved grid: {output_path}")


# ==============================================================================
# Argument Parser
# ==============================================================================
def get_args():
    p = argparse.ArgumentParser(description="Activation Maximization for SAE neurons")
    # Model
    p.add_argument("--sae_ckpt", type=str, required=True)
    p.add_argument("--model_state_path", type=str, required=True)
    p.add_argument(
        "--which_layer",
        type=str,
        default="",
        help="Encoder layer (default: from SAE ckpt)",
    )

    # Target neurons
    p.add_argument(
        "--concepts",
        type=str,
        required=True,
        help="Comma-separated neuron indices, e.g. '0018,0037,0152'",
    )

    # Output
    p.add_argument("--output_dir", type=str, required=True)

    # Optimization
    p.add_argument("--steps", type=int, default=512)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)

    # Regularization
    p.add_argument(
        "--l2_weight",
        type=float,
        default=1e-4,
        help="L2 Fourier coefficient decay weight",
    )
    p.add_argument(
        "--l1_weight",
        type=float,
        default=1e-4,
        help="L1 pixel sparsity weight (dark backgrounds)",
    )
    p.add_argument(
        "--decay_power",
        type=float,
        default=1.0,
        help="Fourier 1/f decay power (higher = smoother, 1.0 = natural images)",
    )
    p.add_argument(
        "--jitter_px", type=int, default=8, help="Max random jitter in pixels (0=off)"
    )
    p.add_argument(
        "--rotate_deg",
        type=float,
        default=15.0,
        help="Max random rotation degrees (0=off)",
    )
    p.add_argument("--scale_lo", type=float, default=0.9)
    p.add_argument("--scale_hi", type=float, default=1.1)
    p.add_argument(
        "--init_std",
        type=float,
        default=0.01,
        help="Std of initial Fourier coefficients",
    )
    # Legacy (kept for backward compat, ignored by Fourier mode)
    p.add_argument("--tv_weight", type=float, default=0.0)
    p.add_argument("--blur_every", type=int, default=0)
    p.add_argument("--blur_sigma", type=float, default=0.0)

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

    p.add_argument("--dpi", type=int, default=200)

    return p.parse_args()


# ==============================================================================
# Main
# ==============================================================================
def main():
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Parse target neuron indices ──
    concepts = [int(c.strip()) for c in args.concepts.split(",")]
    logger.info(f"Target neurons: {concepts}")

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

    # Freeze SAE
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

    # Freeze encoder
    for p in encoder.parameters():
        p.requires_grad_(False)

    del model, sd
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Output directory ──
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Run AM for each neuron ──
    results = []  # (neuron_idx, img_tensor, activation)

    for neuron_idx in concepts:
        if neuron_idx >= d_sae:
            logger.warning(f"  Neuron {neuron_idx} >= d_sae={d_sae}, skipping")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Activation Maximization: Neuron {neuron_idx:04d}")
        logger.info("=" * 60)

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
            decay_power=args.decay_power,
            jitter_px=args.jitter_px,
            rotate_deg=args.rotate_deg,
            scale_range=(args.scale_lo, args.scale_hi),
            init_std=args.init_std,
            seed=args.seed + neuron_idx,  # Different seed per neuron
        )

        results.append((neuron_idx, img, act))
        logger.info(f"  Neuron {neuron_idx:04d}: activation = {act:.4f}")

        # Compute spatial activation heatmap
        heatmap = get_spatial_activation_map(
            img,
            encoder,
            sae,
            which_layer,
            neuron_idx,
            device,
        )
        logger.info(f"  Heatmap range: [{heatmap.min():.4f}, {heatmap.max():.4f}]")

        # Save individual plot
        out_path = os.path.join(args.output_dir, f"am_neuron_{neuron_idx:04d}.png")
        visualize_am_result(
            img, neuron_idx, act, out_path, heatmap=heatmap, dpi=args.dpi
        )

        # Save raw tensor
        np.savez_compressed(
            os.path.join(args.output_dir, f"am_neuron_{neuron_idx:04d}.npz"),
            img=img.numpy(),
            heatmap=heatmap,
            neuron_idx=neuron_idx,
            activation=act,
        )

    # ── Grid visualization ──
    if len(results) > 1:
        grid_path = os.path.join(args.output_dir, "am_grid.png")
        visualize_am_grid(results, grid_path, dpi=args.dpi)

    logger.info(f"\n{'='*60}")
    logger.info("Activation Maximization complete!")
    logger.info(f"Output: {args.output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
