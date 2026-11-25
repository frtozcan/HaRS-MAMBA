"""Segmentation metrics.

Reported in two ways:
* image-wise  : metric computed on every image, then averaged (ISIC challenge style).
* dataset-wise: TP/FP/FN/TN pooled over all pixels of the split, then the metric.

For foreground Dice and IoU it always holds that IoU = Dice / (2 - Dice) <= Dice.
A reported IoU larger than Dice means the IoU was computed differently (for example,
averaged with the background class).
"""
from __future__ import annotations

import numpy as np
import torch

METRICS = ("dice", "iou", "sensitivity", "specificity", "accuracy")


@torch.no_grad()
def confusion(pred: torch.Tensor, target: torch.Tensor):
    """pred/target: (b, 1, H, W) binary -> (b, 4) float64 [tp, fp, fn, tn]."""
    p = pred.flatten(1).bool()
    t = target.flatten(1).bool()
    tp = (p & t).sum(1)
    fp = (p & ~t).sum(1)
    fn = (~p & t).sum(1)
    tn = (~p & ~t).sum(1)
    return torch.stack([tp, fp, fn, tn], 1).double().cpu()


def scores(tp, fp, fn, tn):
    """Works on scalars or numpy arrays.  Empty-vs-empty images count as perfect."""
    tp, fp, fn, tn = (np.asarray(v, dtype=np.float64) for v in (tp, fp, fn, tn))
    both_empty = (tp + fp + fn) == 0
    with np.errstate(divide="ignore", invalid="ignore"):
        dice = np.where(both_empty, 1.0, 2 * tp / (2 * tp + fp + fn))
        iou = np.where(both_empty, 1.0, tp / (tp + fp + fn))
        sen = np.where(tp + fn == 0, 1.0, tp / (tp + fn))
        spe = np.where(tn + fp == 0, 1.0, tn / (tn + fp))
        acc = (tp + tn) / (tp + fp + fn + tn)
    return {"dice": dice, "iou": iou, "sensitivity": sen, "specificity": spe, "accuracy": acc}


class MetricTracker:
    def __init__(self):
        self.rows = []          # (b, 4) blocks

    def update(self, pred, target):
        self.rows.append(confusion(pred, target).numpy())

    def per_image(self):
        c = np.concatenate(self.rows, 0)
        return c, scores(*c.T)

    def summary(self):
        c, s = self.per_image()
        res = {f"{k}_img": float(np.mean(v)) for k, v in s.items()}
        g = scores(*c.sum(0))
        res.update({f"{k}_global": float(v) for k, v in g.items()})
        res["n_images"] = int(c.shape[0])
        return res
