# ==============================================================================
# Model: Encoder (same as yours) + minimal loader wrapper
# ==============================================================================

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential

from run_CNN.logging_utils import OUT_DIM, get_logger

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


def conv2d(in_ch, out_ch, k=3, stride=1, padding=1, dilation=1, bias=True):
    return nn.Conv2d(
        in_ch,
        out_ch,
        kernel_size=k,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
    )


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1):
        super().__init__()
        self.c1 = conv2d(
            in_ch, out_ch, 3, 1, padding=dilation, dilation=dilation, bias=True
        )
        self.c2 = conv2d(
            out_ch, out_ch, 3, 1, padding=dilation, dilation=dilation, bias=True
        )
        self.proj = None
        if in_ch != out_ch:
            self.proj = conv2d(in_ch, out_ch, 1, 1, padding=0, bias=False)

    def forward(self, x):
        identity = x
        x = F.relu(x, inplace=True)
        x = self.c1(x)
        x = F.relu(x, inplace=True)
        x = self.c2(x)
        if self.proj is not None:
            identity = self.proj(identity)
        return x + identity


class Stage(nn.Module):
    def __init__(
        self, in_ch, out_ch, n_blocks, dilation, use_ckpt: bool, ckpt_segments: int
    ):
        super().__init__()
        self.use_ckpt = bool(use_ckpt)
        self.ckpt_segments = int(ckpt_segments)
        blocks = [ResBlock(in_ch, out_ch, dilation=dilation)]
        for _ in range(n_blocks - 1):
            blocks.append(ResBlock(out_ch, out_ch, dilation=dilation))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        if (
            self.use_ckpt
            and self.training
            and self.ckpt_segments > 1
            and len(self.blocks) > 1
        ):
            seg = min(self.ckpt_segments, len(self.blocks))
            return checkpoint_sequential(self.blocks, seg, x, use_reentrant=False)
        return self.blocks(x)


class Encoder(nn.Module):
    def __init__(
        self,
        blocks=(2, 2, 2, 3),
        dilations=(1, 1, 1, 1),
        refine_blocks=1,
        ckpt_segments=2,
    ):
        super().__init__()
        b2, b3, b4, b5 = blocks
        d2, d3, d4, d5 = dilations

        self.stem = nn.Sequential(
            conv2d(3, 64, k=3, stride=2, padding=1, bias=True)
        )  # 128->64
        self.stage2 = Stage(64, 128, b2, d2, use_ckpt=False, ckpt_segments=1)
        self.stage3 = Stage(128, 256, b3, d3, use_ckpt=False, ckpt_segments=1)
        self.stage4 = Stage(
            256, 512, b4, d4, use_ckpt=True, ckpt_segments=ckpt_segments
        )
        self.stage5 = Stage(
            512, OUT_DIM, b5, d5, use_ckpt=True, ckpt_segments=ckpt_segments
        )
        self.refine = Stage(
            OUT_DIM,
            OUT_DIM,
            int(refine_blocks),
            1,
            use_ckpt=True,
            ckpt_segments=ckpt_segments,
        )

        self.trunk = nn.Sequential(
            self.stem, self.stage2, self.stage3, self.stage4, self.stage5, self.refine
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.trunk(x)
        x = self.gap(x).flatten(1)
        return x

    @torch.no_grad()
    def forward_feature_maps(self, x, which: str):
        """
        which in {"stage5_mid", "stage5_out", "refine_out"}
        - stage5_mid: after stage5 blocks[:-1] (penultimate ResBlock in stage5)
        - stage5_out: after all stage5 blocks
        - refine_out: after refine block
        returns feature map (B, C, H, W)
        """
        x = self.stem(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        if which == "stage5_mid":
            for block in list(self.stage5.blocks)[:-1]:
                x = block(x)
            return x
        x = self.stage5(x)
        if which == "stage5_out":
            return x
        x = self.refine(x)
        if which == "refine_out":
            return x
        raise ValueError(f"Unknown which={which}")


def build_projector(
    in_dim: int,
    embed_dim: int,
    proj_layers: int,
    proj_hidden: int,
    use_bn: bool,
    dropout: float,
) -> nn.Module:
    proj_layers = int(proj_layers)
    proj_hidden = int(proj_hidden)
    dropout = float(dropout)

    def lin(a, b):
        return nn.Linear(
            a, b, bias=False
        )  # BN 쓰면 bias=False 하는게 일반적. 이게 표준

    def bn(d):
        return nn.BatchNorm1d(d) if use_bn else nn.Identity()

    def do():  # 뒤에 dropout 안 씀. p.add_argument("--proj_dropout", type=float, default=0.0) 이고 결국 proj_dropout이 들어가는데 설정에서 0.0으로 해놨음. 선택만 가능 일반적으로 projector에 dropout 안씀
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


class SupMoCoModel(nn.Module):
    """
    Minimal wrapper to load your saved model_q.state_dict() (encoder+projector keys).
    For SAE we only use model.encoder.
    """

    def __init__(
        self,
        embed_dim=512,
        blocks=(2, 2, 2, 3),
        dilations=(1, 1, 1, 1),
        refine_blocks=1,
        ckpt_segments=0,
        proj_layers: int = 2,
        proj_hidden: int = 2048,
        proj_bn: bool = False,
        # dropout 안씀
        proj_dropout: float = 0.0,
        use_l2_norm_pool: bool = True,
    ):
        super().__init__()
        self.encoder = Encoder(
            blocks=blocks,
            dilations=dilations,
            refine_blocks=refine_blocks,
            ckpt_segments=ckpt_segments,
        )
        self.use_l2_norm_pool = use_l2_norm_pool
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
        if self.use_l2_norm_pool:
            pooled = F.normalize(pooled, dim=1)
        z = self.projector(pooled)
        return F.normalize(z, dim=1)


def robust_load_state_dict(model: nn.Module, state_dict: dict, strict: bool = False):
    """
    Robustly load state_dict by:
    1. Extracting 'model_q' if the state_dict is a full checkpoint dictionary.
    2. Stripping common prefixes like 'module.' (DataParallel) or '_orig_mod.' (torch.compile).
    """
    if "model_q" in state_dict:
        sd = state_dict["model_q"]
    elif "model" in state_dict:
        sd = state_dict["model"]
    else:
        sd = state_dict

    # Strip prefixes
    new_sd = {}
    for k, v in sd.items():
        name = k
        if name.startswith("module."):
            name = name[7:]
        if name.startswith("_orig_mod."):
            name = name[10:]
        new_sd[name] = v

    return model.load_state_dict(new_sd, strict=strict)
