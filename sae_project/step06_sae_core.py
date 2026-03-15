# ==============================================================================
# Pointwise Top-K SAE + AuxK (no argparse options)
# - Decoder untied (learned independently)
# - Tied init only: W_dec initialized as W_enc.T at init, then trained untied
# - AuxK loss path: trains additional features beyond TopK without changing inference
# - Decoder row norm = 1 constraint
# ==============================================================================

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointwiseTopKSAE(nn.Module):
    """
    Pointwise SAE operating on tokens of shape (N, d_in).
    Forward returns:
      recon_main, acts_main, recon_aux, acts_aux

    Inference-time recon should use recon_main only.
    Aux path is training-only.
    """

    # ---- hardcoded defaults (no argparse bloat) ----
    AUX_K: int = 32          # auxiliary K (must be > k)
    AUX_COEFF: float = 0.10  # auxiliary loss coefficient
    TIED_INIT_ONLY: bool = True

    def __init__(self, d_in: int, d_sae: int, k: int, init_scale: float = 0.02):
        super().__init__()
        self.d_in = int(d_in)
        self.d_sae = int(d_sae)
        self.k = int(k)

        # encoder/decoder weights (untied)
        self.W_enc = nn.Parameter(torch.randn(self.d_in, self.d_sae) * float(init_scale))
        self.b_enc = nn.Parameter(torch.zeros(self.d_sae))
        self.W_dec = nn.Parameter(torch.randn(self.d_sae, self.d_in) * float(init_scale))
        self.b_dec = nn.Parameter(torch.zeros(self.d_in))

        # tied init only (W_dec <- W_enc.T), then untied training
        if self.TIED_INIT_ONLY:
            with torch.no_grad():
                self.W_dec.copy_(self.W_enc.t().contiguous())

        # usage EMA buffer (for dead feature monitoring)
        self.register_buffer("usage_ema", torch.zeros(self.d_sae), persistent=True)

        # keep decoder rows unit norm at init
        self.renorm_decoder_()

        # aux config
        self.aux_k = int(self.AUX_K)
        self.aux_coeff = float(self.AUX_COEFF)
        if self.aux_k <= self.k:
            # make aux inactive safely
            self.aux_k = self.k
            self.aux_coeff = 0.0

    @torch.no_grad()
    def renorm_decoder_(self, eps: float = 1e-12):
        # row-wise norm = 1  (each feature vector)
        w = self.W_dec.data
        n = w.norm(dim=1, keepdim=True).clamp_min(eps)
        w.div_(n)

    @torch.no_grad()
    def update_usage_ema_(self, acts: torch.Tensor, ema: float = 0.99):
        """
        acts: (N, d_sae) sparse tensor.
        We track fraction of non-zeros per feature.
        """
        # nonzero rate per feature
        used = (acts != 0).float().mean(dim=0)
        self.usage_ema.mul_(ema).add_(used, alpha=(1.0 - ema))

    @staticmethod
    def _topk_masked(pre: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        pre: (N, d_sae)
        Returns:
          acts: masked sparse (N, d_sae)
          idx: indices of selected features (N, k)
        Selection is by |pre|, values keep original sign/magnitude.
        """
        # idx: (N, k)
        idx = torch.topk(pre.abs(), k=k, dim=1, largest=True, sorted=False).indices
        acts = torch.zeros_like(pre)
        acts.scatter_(1, idx, pre.gather(1, idx))
        return acts, idx

    @staticmethod
    def _auxk_excluding_topk(pre: torch.Tensor, idx_topk: torch.Tensor, aux_k: int) -> torch.Tensor:
        """
        Build aux activations by taking top aux_k by |pre|,
        then removing those already in topk, so aux uses "other" features.
        """
        if aux_k <= idx_topk.size(1):
            return torch.zeros_like(pre)

        idx_aux = torch.topk(pre.abs(), k=aux_k, dim=1, largest=True, sorted=False).indices
        aux = torch.zeros_like(pre)
        aux.scatter_(1, idx_aux, pre.gather(1, idx_aux))

        # zero out topk positions (exclude topk from aux)
        # create a mask of ones at topk positions
        mask = torch.zeros_like(pre)
        mask.scatter_(1, idx_topk, 1.0)
        aux = aux * (1.0 - mask)
        return aux

    def forward(self, tokens: torch.Tensor):
        """
        tokens: (N, d_in)
        returns recon_main, acts_main, recon_aux, acts_aux
        """
        # preactivations
        pre = tokens @ self.W_enc + self.b_enc  # (N, d_sae)

        # main top-k
        acts_main, idx_topk = self._topk_masked(pre, self.k)
        recon_main = acts_main @ self.W_dec + self.b_dec  # (N, d_in)

        # aux path (training-only)
        if self.aux_coeff > 0.0 and self.aux_k > self.k:
            acts_aux = self._auxk_excluding_topk(pre, idx_topk, self.aux_k)
            recon_aux = acts_aux @ self.W_dec + self.b_dec
        else:
            acts_aux = torch.zeros_like(acts_main)
            recon_aux = torch.zeros_like(recon_main)

        return recon_main, acts_main, recon_aux, acts_aux
