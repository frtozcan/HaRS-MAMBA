"""HaRS-Mamba: Hair-aware Robust Scanning Mamba U-Net for skin lesion segmentation.

Hair prior: a two-scale black top-hat + a small CNN map the image to an artifact
            probability a, which is concatenated to the input and gates every SSM layer.
Encoder   : conv stages at high resolution, then Hybrid (AGSS) stages with the four
            cross-scan routes; every SSM step size is shrunk where a is high.
Decoder   : Hybrid (AGSS) blocks with the same four routes at 1/8 and 1/4 resolution,
            convolutional blocks at 1/2 and full resolution.
Heads     : deep-supervision heads at every decoder level, a boundary head, and a final
            segmentation head conditioned on the predicted boundary.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .artifact import HairPrior
from .blocks import ConvBNAct, HybridBlock, ResConvBlock
from .scans import FIXED_ROUTES

CONFIGS = {
    #            channels per stage (1, 1/2, 1/4, 1/8, 1/16)   blocks per stage
    "tiny": dict(dims=(8, 16, 32, 64, 128), depths=(1, 1, 1, 1, 2), headdim=16, d_state=16),
    "base": dict(dims=(16, 32, 64, 128, 256), depths=(1, 1, 1, 2, 2), headdim=32, d_state=16),
    "large": dict(dims=(24, 48, 96, 192, 384), depths=(1, 1, 2, 2, 2), headdim=24, d_state=16),
}
SSM_FROM_STAGE = 2      # stages >= 2 (1/4 resolution and coarser) use state-space blocks
SSM_DEC_STAGES = (3, 2)  # decoder levels (1/8, 1/4) that use state-space blocks


class HaRSMamba(nn.Module):
    def __init__(self, size: str = "base", in_ch: int = 3, use_artifact: bool = True,
                 drop_path: float = 0.1, chunk: int = 64):
        super().__init__()
        cfg = CONFIGS[size]
        dims, depths = cfg["dims"], cfg["depths"]
        ssm_kw = dict(headdim=cfg["headdim"], d_state=cfg["d_state"], chunk=chunk,
                      use_artifact=use_artifact)
        self.use_artifact = use_artifact
        self.cfg = dict(size=size, use_artifact=use_artifact)

        self.hair = HairPrior() if use_artifact else None
        stem_in = in_ch + (1 if use_artifact else 0)

        n_ssm = sum(depths[SSM_FROM_STAGE:]) + len(SSM_DEC_STAGES)
        dpr = iter(torch.linspace(0, drop_path, n_ssm).tolist())

        # ---------------- encoder
        self.enc = nn.ModuleList()
        self.down = nn.ModuleList()
        for i, (d, n) in enumerate(zip(dims, depths)):
            if i == 0:
                self.down.append(ConvBNAct(stem_in, d))
            else:
                self.down.append(ConvBNAct(dims[i - 1], d, k=3, s=2))
            if i < SSM_FROM_STAGE:
                self.enc.append(nn.ModuleList([ResConvBlock(d, d) for _ in range(n)]))
            else:
                self.enc.append(nn.ModuleList(
                    [HybridBlock(d, FIXED_ROUTES, drop_path=next(dpr), **ssm_kw) for _ in range(n)]))

        # ---------------- decoder (levels 3, 2, 1, 0)
        self.up_proj = nn.ModuleDict()
        self.fuse = nn.ModuleDict()
        self.dec = nn.ModuleDict()
        self.aux = nn.ModuleDict({"4": nn.Conv2d(dims[4], 1, 1)})
        for i in range(len(dims) - 2, -1, -1):
            d = dims[i]
            self.up_proj[str(i)] = ConvBNAct(dims[i + 1], d, k=1)
            self.fuse[str(i)] = ConvBNAct(2 * d, d)
            if i in SSM_DEC_STAGES:
                self.dec[str(i)] = HybridBlock(d, FIXED_ROUTES, drop_path=next(dpr), **ssm_kw)
            else:
                self.dec[str(i)] = ResConvBlock(d, d)
            if i > 0:
                self.aux[str(i)] = nn.Conv2d(d, 1, 1)

        d0 = dims[0]
        self.boundary_head = nn.Sequential(ConvBNAct(d0, d0), nn.Conv2d(d0, 1, 1))
        self.seg_head = nn.Sequential(ConvBNAct(d0 + 1, d0), nn.Conv2d(d0, 1, 1))

    # ------------------------------------------------------------------------------
    def forward(self, x):
        out = {"aux": []}
        art = None
        if self.hair is not None:
            hair_logits, art = self.hair(x)
            out["hair"] = hair_logits
            x = torch.cat([x, art], dim=1)

        skips = []
        for i, (down, blocks) in enumerate(zip(self.down, self.enc)):
            x = down(x)
            for blk in blocks:
                x = blk(x, art) if isinstance(blk, HybridBlock) else blk(x)
            skips.append(x)

        coarse = self.aux["4"](x)
        out["aux"].append(coarse)
        for i in range(len(skips) - 2, -1, -1):
            skip = skips[i]
            up = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = self.fuse[str(i)](torch.cat([self.up_proj[str(i)](up), skip], dim=1))
            blk = self.dec[str(i)]
            x = blk(x, art) if isinstance(blk, HybridBlock) else blk(x)
            if str(i) in self.aux:
                coarse = self.aux[str(i)](x)
                out["aux"].append(coarse)

        boundary = self.boundary_head(x)
        out["boundary"] = boundary
        out["seg"] = self.seg_head(torch.cat([x, torch.sigmoid(boundary)], dim=1))
        return out


def build_model(size="base", **kw) -> HaRSMamba:
    return HaRSMamba(size=size, **kw)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
