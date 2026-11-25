"""Training objective: BCE + Dice with deep supervision, boundary loss and hair supervision."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_dice_loss(logits, target, eps=1.0):
    """Per-image soft Dice loss, averaged over the batch."""
    p = torch.sigmoid(logits).flatten(1)
    t = target.flatten(1)
    inter = (p * t).sum(1)
    return (1 - (2 * inter + eps) / (p.sum(1) + t.sum(1) + eps)).mean()


def bce_dice(logits, target):
    return F.binary_cross_entropy_with_logits(logits, target) + soft_dice_loss(logits, target)


@torch.no_grad()
def boundary_target(mask, width: int = 3):
    """Morphological gradient of the mask: a band of about `width` px around the border."""
    k = 2 * (width // 2) + 1
    dil = F.max_pool2d(mask, k, stride=1, padding=k // 2)
    ero = -F.max_pool2d(-mask, k, stride=1, padding=k // 2)
    return (dil - ero).clamp(0, 1)


class HaRSLoss(nn.Module):
    def __init__(self, aux_weights=(0.1, 0.2, 0.3, 0.4), w_boundary=0.5, w_hair=0.1):
        super().__init__()
        self.aux_weights = aux_weights      # for the aux outputs ordered coarse -> fine
        self.w_boundary = w_boundary
        self.w_hair = w_hair

    def forward(self, out, mask, hair_mask=None):
        logs = {}
        loss = bce_dice(out["seg"].float(), mask)
        logs["seg"] = loss.item()

        aux_total = 0.0
        for w, a in zip(self.aux_weights, out["aux"]):
            t = F.interpolate(mask, size=a.shape[-2:], mode="area")
            aux_total = aux_total + w * bce_dice(a.float(), t)
        loss = loss + aux_total
        logs["aux"] = float(aux_total.detach()) if torch.is_tensor(aux_total) else aux_total

        bt = boundary_target(mask)
        pos = bt.mean().clamp_min(1e-3)
        lb = F.binary_cross_entropy_with_logits(out["boundary"].float(), bt,
                                                pos_weight=((1 - pos) / pos).clamp(max=50))
        loss = loss + self.w_boundary * lb
        logs["boundary"] = lb.item()

        if self.w_hair > 0 and hair_mask is not None and "hair" in out:
            has = hair_mask.flatten(1).amax(1) > 0                  # images that got synthetic hair
            if has.any():
                hl, ht = out["hair"][has].float(), hair_mask[has]
                pos = ht.mean().clamp_min(1e-3)
                lh = F.binary_cross_entropy_with_logits(hl, ht, pos_weight=((1 - pos) / pos).clamp(max=50))
                loss = loss + self.w_hair * lh
                logs["hair"] = lh.item()
        logs["total"] = loss.item()
        return loss, logs
