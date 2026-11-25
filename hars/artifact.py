"""Hair / artifact prior.

Dark, thin, elongated hairs are the typical occluder in dermoscopy.  A morphological
black top-hat (closing minus image) highlights exactly such structures: closing erases
dark details thinner than the structuring element, so the difference is large on hairs.
The responses at two scales are refined by a tiny CNN into an artifact map a in [0, 1].

The map is (i) concatenated to the RGB input and (ii) fed to every Artifact-Gated
Selective Scan, where it shrinks the SSM step size Delta on occluded tokens so the
recurrent state "steps over" them instead of being overwritten (see blocks.AGSSM).

It is weakly supervised by the synthetic hairs drawn during augmentation
(``data.SyntheticHair``), for which the exact hair mask is known.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _odd(k: float) -> int:
    k = max(3, int(round(k)))
    return k if k % 2 == 1 else k + 1


def dilate(x, k):
    return F.max_pool2d(x, k, stride=1, padding=k // 2)


def erode(x, k):
    return -F.max_pool2d(-x, k, stride=1, padding=k // 2)


class HairPrior(nn.Module):
    def __init__(self, scales=(7, 15), ref_size: int = 256, hidden: int = 8):
        super().__init__()
        self.scales = scales
        self.ref_size = ref_size
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)
        self.refine = nn.Sequential(
            nn.Conv2d(len(scales) + 1, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=2, dilation=2),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, img_norm: torch.Tensor):
        """img_norm: ImageNet-normalised RGB (b, 3, H, W) -> (logits, prob), both (b, 1, H, W)."""
        rgb = img_norm * self.std + self.mean
        lum = (0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]).clamp(0, 1)
        s = img_norm.shape[-1] / self.ref_size
        feats = []
        for k in self.scales:
            k = _odd(k * s)
            closing = erode(dilate(lum, k), k)
            bth = (closing - lum).clamp_min(0)
            bth = bth / (bth.amax(dim=(2, 3), keepdim=True) + 1e-4)       # per-image normalisation
            feats.append(bth)
        feats.append(lum)
        logits = self.refine(torch.cat(feats, dim=1))
        return logits, torch.sigmoid(logits)
