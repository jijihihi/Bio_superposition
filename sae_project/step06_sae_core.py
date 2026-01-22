# ==============================================================================
# Pointwise Top-K SAE
# ==============================================================================
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def topk_relu(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    x: (N, d_sae) after ReLU
    keep top-k per row, zero others
    """
    if k <= 0:
        return torch.zeros_like(x)
    if k >= x.size(1):
        return x
    vals, idx = torch.topk(x, k, dim=1, largest=True, sorted=False)
    out = torch.zeros_like(x)
    out.scatter_(1, idx, vals)
    return out


class PointwiseTopKSAE(nn.Module):
    """
    token x in R^{d_in}
    acts = TopK(ReLU(x W_enc + b_enc))
    recon = acts W_dec + b_dec
    decoder rows are unit norm enforced
    """
    def __init__(self, d_in: int, d_sae: int, k: int, init_scale: float = 0.02):
        super().__init__()
        self.d_in = int(d_in)
        self.d_sae = int(d_sae)
        self.k = int(k)

        self.W_enc = nn.Parameter(torch.randn(self.d_in, self.d_sae) * init_scale)
        self.b_enc = nn.Parameter(torch.zeros(self.d_sae))
        self.W_dec = nn.Parameter(torch.randn(self.d_sae, self.d_in) * init_scale)
        self.b_dec = nn.Parameter(torch.zeros(self.d_in))

        self.register_buffer("usage_ema", torch.zeros(self.d_sae), persistent=True)

    @torch.no_grad()
    def renorm_decoder_(self, eps: float = 1e-8):
        norms = self.W_dec.data.norm(dim=1, keepdim=True).clamp_min(eps)
        self.W_dec.data.div_(norms)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = x @ self.W_enc + self.b_enc
        h = F.relu(h)
        h = topk_relu(h, self.k)
        return h

    def decode(self, a: torch.Tensor) -> torch.Tensor:
        return a @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor):
        a = self.encode(x)
        recon = self.decode(a)
        return recon, a

    @torch.no_grad()
    def update_usage_ema_(self, a: torch.Tensor, ema: float = 0.99):
        active = (a > 0).float().mean(dim=0)
        self.usage_ema.mul_(ema).add_(active, alpha=(1.0 - ema))


# ==============================================================================
# Neuron resampling (dead feature replacement)
# ==============================================================================
@torch.no_grad()
def resample_dead_features_(
    sae: PointwiseTopKSAE,
    opt: torch.optim.Optimizer,
    x_batch: torch.Tensor,          # (N, d_in)
    recon_batch: torch.Tensor,      # (N, d_in)
    dead_threshold: float,
    max_resample_frac: float = 0.05,
    eps: float = 1e-8,
) -> int:
    """
    - Identify dead features using sae.usage_ema
    - Replace subset with normalized residual directions
    - Reset optimizer moments for those indices
    """
    usage = sae.usage_ema
    dead = (usage < dead_threshold).nonzero(as_tuple=False).flatten()
    if dead.numel() == 0:
        return 0

    max_n = int(max(1, sae.d_sae * max_resample_frac))
    if dead.numel() > max_n:
        dead = dead[torch.randperm(dead.numel(), device=dead.device)[:max_n]]

    resid = (x_batch - recon_batch).detach()
    n = dead.numel()

    if resid.size(0) < n:
        idx = torch.randint(0, resid.size(0), (n,), device=resid.device)
        vecs = resid[idx]
    else:
        idx = torch.randperm(resid.size(0), device=resid.device)[:n]
        vecs = resid[idx]

    vecs = vecs / (vecs.norm(dim=1, keepdim=True).clamp_min(eps))

    sae.W_enc.data[:, dead] = vecs.t()
    sae.W_dec.data[dead, :] = vecs
    sae.b_enc.data[dead] = 0.0

    sae.renorm_decoder_()
    sae.usage_ema[dead] = 0.0

    # reset optimizer state (Adam moments)
    for group in opt.param_groups:
        for p in group["params"]:
            st = opt.state.get(p, None)
            if st is None:
                continue
            for key in ["exp_avg", "exp_avg_sq"]:
                if key in st and torch.is_tensor(st[key]):
                    buf = st[key]
                    if p is sae.W_enc:
                        buf[:, dead] = 0
                    elif p is sae.W_dec:
                        buf[dead, :] = 0
                    elif p is sae.b_enc:
                        buf[dead] = 0

    return int(n)
