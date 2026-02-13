# ==============================================================================
# Gated Sparse Autoencoder (Gated SAE)
# - Weight tying option (W_gate = W_mag)
# - L1 sparsity on relu(gate_pre) * decoder (detached)
# - Aux loss with detached decoder
# - Decoder row norm = 1 constraint
# - Clustering-based initialization with optional noise
# ==============================================================================

from __future__ import annotations
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GatedSAE(nn.Module):
    """
    Gated Sparse Autoencoder
    
    Loss = L_recon (L2-weighted) + λ * L_sparsity + α * L_aux
    
    - L_recon: 각 토큰의 L2 norm으로 가중치 적용
    - L_sparsity: L1 penalty on relu(gate_pre) * W_dec.detach()
    - L_aux: Dead neuron 방지 (detached decoder)
    - λ: Warmup 스케줄 (0 → final_sparsity)
    
    Args:
        d_in: Input dimension
        d_sae: SAE hidden dimension (dictionary size)
        tie_weights: If True, share W_gate = W_mag for parameter efficiency
        aux_k: Number of features for auxiliary loss path
        init_scale: Scale for random initialization
    """

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        tie_weights: bool = True,
        aux_k: int = 32,
        init_scale: float = 0.02,
    ):
        super().__init__()
        self.d_in = int(d_in)
        self.d_sae = int(d_sae)
        self.tie_weights = bool(tie_weights)
        self.aux_k = int(aux_k)

        # Magnitude encoder (W_mag, b_mag)
        self.W_mag = nn.Parameter(torch.randn(self.d_in, self.d_sae) * float(init_scale))
        self.b_mag = nn.Parameter(torch.zeros(self.d_sae)) # 일반적인 bias ReLU에 들어가는.

        # Gating encoder (W_gate, b_gate)
        # We initialize b_gate to 1.0 (positive) to prevent neurons from being dead 
        # at the beginning of training when L1 penalty is not yet active.
        if self.tie_weights:
            # W_gate shares with W_mag (handled by property W_gate_effective)
            self.b_gate = nn.Parameter(torch.ones(self.d_sae) * 0.1) # 이 gate를 켤지 말지 결정. 
            # Register dummy so state_dict knows we're tying
            self.register_buffer("_tied_weights_flag", torch.tensor(1))
        else:
            self.W_gate = nn.Parameter(torch.randn(self.d_in, self.d_sae) * float(init_scale))
            self.b_gate = nn.Parameter(torch.ones(self.d_sae) * 0.1)

        # Decoder (untied from encoder)
        self.W_dec = nn.Parameter(torch.randn(self.d_sae, self.d_in) * float(init_scale))
        self.b_dec = nn.Parameter(torch.zeros(self.d_in))

        # Initialize decoder as transpose of magnitude encoder
        with torch.no_grad():
            self.W_dec.copy_(self.W_mag.t().contiguous())

        # Usage EMA buffer for dead feature monitoring
        self.register_buffer("usage_ema", torch.zeros(self.d_sae), persistent=True)

        # Keep decoder rows unit norm at init
        self.renorm_decoder_()

    @property
    def W_gate_effective(self) -> torch.Tensor:
        """Return the effective W_gate (shared with W_mag if tied)."""
        if self.tie_weights:
            return self.W_mag
        return self.W_gate

    @torch.no_grad()
    def renorm_decoder_(self, eps: float = 1e-12):
        """Normalize decoder rows to unit L2 norm."""
        n = self.W_dec.norm(dim=1, keepdim=True).clamp_min(eps)
        self.W_dec.div_(n)

    @torch.no_grad()
    def update_usage_ema_(self, acts: torch.Tensor, ema: float = 0.99):
        """
        Update usage EMA for dead feature monitoring.
        acts: (N, d_sae) sparse tensor
        """
        used = (acts != 0).float().mean(dim=0)
        self.usage_ema.mul_(ema).add_(used, alpha=(1.0 - ema))

    def project_decoder_grads_(self, eps: float = 1e-12):
        """
        Project decoder gradients to remove the component parallel to decoder weights.
        
        This should be called AFTER backward() and BEFORE optimizer.step().
        
        Why: Gradient components parallel to W_dec only change the magnitude (norm)
        of the decoder rows. Since we normalize to unit norm anyway, these components
        are wasted and can conflict with momentum-based optimizers (Adam, etc).
        
        Math: grad_projected = grad - (grad · w_hat) * w_hat
              where w_hat = w / ||w|| is the unit vector along each decoder row
        """
        if self.W_dec.grad is None:
            return
        
        with torch.no_grad():
            W = self.W_dec       # (d_sae, d_in)
            G = self.W_dec.grad  # (d_sae, d_in)
            
            # Normalize each row to get unit vectors
            W_norm = W.norm(dim=1, keepdim=True).clamp_min(eps)  # (d_sae, 1)
            W_hat = W / W_norm  # (d_sae, d_in)
            
            # Compute parallel component: (G · W_hat) for each row
            # dot product per row: sum(G * W_hat, dim=1)
            parallel_component = (G * W_hat).sum(dim=1, keepdim=True)  # (d_sae, 1)
            
            # Remove parallel component from gradient
            G_projected = G - parallel_component * W_hat  # (d_sae, d_in)
            
            # Update gradient in-place
            self.W_dec.grad.copy_(G_projected)

    def forward(self, tokens: torch.Tensor):
        """
        Forward pass.
        
        Args:
            tokens: (N, d_in) input tokens
            
        Returns:
            recon: (N, d_in) reconstructed tokens
            acts: (N, d_sae) gated activations
            gate_pre: (N, d_sae) gate pre-activations (for sparsity loss)
            recon_aux: (N, d_in) auxiliary reconstruction (for aux loss)
            acts_aux: (N, d_sae) auxiliary activations
        """
        # Gate pre-activations
        gate_pre = tokens @ self.W_gate_effective + self.b_gate  # (N, d_sae)
        
        # Magnitude pre-activations
        mag_pre = tokens @ self.W_mag + self.b_mag  # (N, d_sae)
        
        # Gating: binary gate based on gate_pre > 0
        # Using straight-through estimator: forward uses binary, backward uses sigmoid
        gate = (gate_pre > 0).float()
        if self.training:
            # STE: Gradient behaves like sigmoid, but forward remains binary 0/1
            gate = gate_pre.sigmoid() + (gate - gate_pre.sigmoid()).detach()
        
        # Magnitude: ReLU activation
        mag = F.relu(mag_pre)
        
        # Final activation: gate ⊙ magnitude
        acts = gate * mag  # (N, d_sae)
        
        # Main reconstruction
        recon = acts @ self.W_dec + self.b_dec  # (N, d_in)
        
        # Auxiliary path for dead neurons (top aux_k by magnitude excluding active)
        if self.aux_k > 0 and self.training:
            acts_aux = self._compute_aux_activations(mag, gate, self.aux_k)
            # Use detached decoder for aux loss to prevent gating from corrupting decoder
            recon_aux = acts_aux @ self.W_dec.detach() + self.b_dec.detach()
        else:
            acts_aux = torch.zeros_like(acts)
            recon_aux = torch.zeros_like(recon)
        
        return recon, acts, gate_pre, recon_aux, acts_aux

    def _compute_aux_activations(
        self, mag: torch.Tensor, gate: torch.Tensor, aux_k: int
    ) -> torch.Tensor:
        """
        Compute auxiliary activations from features not selected by gating.
        """
        # Features where gate is off (potential dead neurons we want to train)
        inactive_mask = (gate == 0).float()
        
        # Masked magnitude (only inactive features)
        masked_mag = mag * inactive_mask
        
        # Select top aux_k by magnitude from inactive features
        if aux_k >= self.d_sae:
            return masked_mag
        
        # Get top-k indices from masked magnitudes
        _, idx = torch.topk(masked_mag, k=min(aux_k, self.d_sae), dim=1, largest=True, sorted=False)
        
        # Create sparse aux activations
        acts_aux = torch.zeros_like(mag)
        acts_aux.scatter_(1, idx, masked_mag.gather(1, idx))
        
        return acts_aux

    def compute_sparsity_loss(self, gate_pre: torch.Tensor) -> torch.Tensor:
        """
        Compute L1 sparsity loss on relu(gate_pre) * decoder (detached).
        
        This encourages sparse gating while not affecting decoder learning.
        """
        gate_relu = F.relu(gate_pre)  # (N, d_sae)
        
        # L1 norm of gate_relu weighted by decoder column norms (detached)
        # Using detached decoder to prevent sparsity loss from affecting decoder
        W_dec_detached = self.W_dec.detach()  # (d_sae, d_in)
        decoder_norms = W_dec_detached.norm(dim=1)  # (d_sae,)
        
        # Weighted L1: sum over features of (relu(gate_pre) * decoder_norm)
        weighted_gate = gate_relu * decoder_norms.unsqueeze(0)  # (N, d_sae)
        
        return weighted_gate.mean()

    def load_clustering_init(
        self,
        centroid_path: str,
        noise_scale: float = 0.1,
    ):
        """
        Initialize encoder weights from clustering centroids with optional noise.
        
        Args:
            centroid_path: Path to centroids .npy file
            noise_scale: Scale of random noise to add
        """
        if not os.path.exists(centroid_path):
            print(f"[GatedSAE] Centroid file not found: {centroid_path}")
            return False
        
        centroids = np.load(centroid_path)  # (n_clusters, d_in)
        n_centroids, d_in = centroids.shape
        
        if d_in != self.d_in:
            print(f"[GatedSAE] Centroid d_in mismatch: {d_in} vs {self.d_in}")
            return False
        
        print(f"[GatedSAE] Loading {n_centroids} centroids from {centroid_path}")
        
        # Normalize centroids
        centroids_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
        
        with torch.no_grad():
            W_init = np.zeros((self.d_in, self.d_sae), dtype=np.float32)
            
            if n_centroids >= self.d_sae:
                # Use first d_sae centroids
                W_init = centroids_norm[:self.d_sae].T
            else:
                # Use all centroids, fill rest with random
                W_init[:, :n_centroids] = centroids_norm.T
                W_init[:, n_centroids:] = np.random.randn(
                    self.d_in, self.d_sae - n_centroids
                ).astype(np.float32) * 0.02
            
            # Add noise
            if noise_scale > 0:
                W_init += np.random.randn(*W_init.shape).astype(np.float32) * noise_scale
            
            # Normalize columns
            W_init /= (np.linalg.norm(W_init, axis=0, keepdims=True) + 1e-8)
            
            # Copy to W_mag (and W_gate if not tied)
            self.W_mag.copy_(torch.from_numpy(W_init))
            if not self.tie_weights:
                self.W_gate.copy_(torch.from_numpy(W_init))
            
            # Re-initialize decoder as transpose
            self.W_dec.copy_(self.W_mag.t().contiguous())
            self.renorm_decoder_()
        
        print(f"[GatedSAE] Initialized with centroids + noise_scale={noise_scale}")
        return True


def get_sparsity_coeff(
    step: int,
    warmup_steps: int,
    final_coeff: float,
    total_steps: int = None,
    ramp_fraction: float = 0.1,
) -> float:
    """
    Compute sparsity coefficient with warmup + ramp + maintain schedule.
    
    Schedule:
    - Steps 0 to warmup_steps-1: return 0
    - warmup_steps to warmup_steps + (total_steps * ramp_fraction): linear ramp 0 → final_coeff
    - After ramp: maintain final_coeff
    
    Args:
        step: Current training step
        warmup_steps: Number of steps to keep at 0
        final_coeff: Final sparsity coefficient
        total_steps: Total training steps
        ramp_fraction: Fraction of total_steps for linear ramp (default: 0.1 = 10%)
    """
    if step < warmup_steps:
        return 0.0
    
    if total_steps is None:
        return final_coeff
    
    # Ramp duration = ramp_fraction of total_steps
    ramp_steps = max(1, int(total_steps * ramp_fraction))
    ramp_end = warmup_steps + ramp_steps
    
    if step >= ramp_end:
        # Maintain final coefficient
        return final_coeff
    
    # Linear ramp from warmup_steps to ramp_end
    progress = (step - warmup_steps) / ramp_steps
    return final_coeff * progress


