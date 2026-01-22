# ==============================================================================
# Model: Encoder (same as yours) + minimal loader wrapper
# ==============================================================================

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential

from sae_project.step02_logging_utils import get_logger, OUT_DIM

logger = get_logger("model_encoder")


def parse_int_list(s: str, n: int) -> Tuple[int, ...]:
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    vals = [int(p) for p in parts]
    if len(vals) != n:
        raise ValueError(f"Expected {n} ints, got {len(vals)} from '{s}'")
    return tuple(vals)


@torch.no_grad()
def renorm_unit_per_out_channel_(model: nn.Module, eps: float = 1e-12):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            w = m.weight.data
            n = w.flatten(1).norm(dim=1, keepdim=True).clamp_min(eps)
            w.div_(n.view(-1, 1, 1, 1))
        elif isinstance(m, nn.Linear):
            w = m.weight.data
            n = w.norm(dim=1, keepdim=True).clamp_min(eps)
            w.div_(n)


def conv2d(in_ch, out_ch, k=3, stride=1, padding=1, dilation=1, bias=False):
    return nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=stride,
                     padding=padding, dilation=dilation, bias=bias)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1):
        super().__init__()
        self.c1 = conv2d(in_ch, out_ch, 3, 1, padding=dilation, dilation=dilation, bias=False)
        self.c2 = conv2d(out_ch, out_ch, 3, 1, padding=dilation, dilation=dilation, bias=False)
        self.proj = None
        if in_ch != out_ch:
            self.proj = conv2d(in_ch, out_ch, 1, 1, padding=0, bias=False)

    def forward(self, x):
        identity = x
        x = F.relu(x, inplace=False)
        x = self.c1(x)
        x = F.relu(x, inplace=False)
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
    def __init__(self, blocks=(2, 2, 4, 4), dilations=(1, 1, 1, 1), refine_blocks=1, ckpt_segments=2):
        super().__init__()
        b2, b3, b4, b5 = blocks
        d2, d3, d4, d5 = dilations

        self.stem = nn.Sequential(conv2d(3, 64, k=3, stride=2, padding=1, bias=False))  # 128->64
        self.stage2 = Stage(64, 128, b2, d2, use_ckpt=False, ckpt_segments=1)
        self.stage3 = Stage(128, 256, b3, d3, use_ckpt=False, ckpt_segments=1)
        self.stage4 = Stage(256, 512, b4, d4, use_ckpt=True, ckpt_segments=ckpt_segments)
        self.stage5 = Stage(512, OUT_DIM, b5, d5, use_ckpt=True, ckpt_segments=ckpt_segments)
        self.refine = Stage(OUT_DIM, OUT_DIM, int(refine_blocks), 1, use_ckpt=True, ckpt_segments=ckpt_segments)

        self.trunk = nn.Sequential(self.stem, self.stage2, self.stage3, self.stage4, self.stage5, self.refine)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.trunk(x)
        x = self.gap(x).flatten(1)
        return x

    @torch.no_grad()
    def forward_feature_maps(self, x, which: str):
        """
        which in {"stage5_out", "refine_out"}
        returns feature map (B, C=512, H, W)
        """
        x = self.stem(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        if which == "stage5_out":
            return x
        x = self.refine(x)
        if which == "refine_out":
            return x
        raise ValueError(f"Unknown which={which}")


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
    """
    Minimal wrapper to load your saved model_q.state_dict() (encoder+projector keys).
    For SAE we only use model.encoder.
    """
    def __init__(
        self,
        embed_dim=512,
        blocks=(2, 2, 4, 4),
        dilations=(1, 1, 1, 1),
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
