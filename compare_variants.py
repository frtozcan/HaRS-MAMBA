"""Ablation tables (Tables 5 and 6 of the paper) from the evaluated runs.

    python compare_variants.py                       # reads $HARS_RUNS/{hars_mamba,no_synthetic_hair,no_agss}
    python compare_variants.py --root results        # reads the released per-image results in this repository

Hair-heavy subset (model-independent): black top-hat of the luminance (max over 7 and 15 px closings) on the
test images; pixels with response > 0.15 on the skin outside the lesion (lesion dilated by 7 px) are grouped
into 8-connected components, and only long, thin components (bounding-box side >= 20 px, fill ratio < 0.35)
are counted.  The 200 test images (20%) with the largest share of such pixels form the subset
(results/hair_heavy_scores.csv; recomputed with --rescore, which needs the PNG cache).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from config import N_FOLDS, REPO, RUNS, SIZE

VARIANTS = [("hars_mamba", "HaRS-Mamba", "✓", "✓"),
            ("no_synthetic_hair", "w/o synthetic hair", "✓", "—"),
            ("no_agss", "w/o AGSS", "—", "—")]
COMPARISONS = [("hars_mamba", "no_agss", "AGSS with synthetic hair"),
               ("no_synthetic_hair", "no_agss", "AGSS alone"),
               ("hars_mamba", "no_synthetic_hair", "synthetic hair")]
SCORES = REPO / "results" / "hair_heavy_scores.csv"


def per_image(path):
    with open(path, newline="") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def holm(p):
    p = np.asarray(p, float); m = len(p); adj = np.empty(m); prev = 0.0
    for rank, i in enumerate(np.argsort(p)):
        prev = max(prev, min(1.0, (m - rank) * p[i])); adj[i] = prev
    return adj


def hair_scores():
    """Model-independent hairiness of each test image (needs the 256 px PNG cache)."""
    import torch
    import torch.nn.functional as F
    from scipy import ndimage as ndi
    from config import CACHE, DATASET
    from hars.data import SkinLesionDataset
    ds = SkinLesionDataset(CACHE, DATASET, SIZE, "test", train=False)
    res = {}
    for i in range(len(ds)):
        it = ds[i]
        rgb = (it["image"] * ds.std + ds.mean).clamp(0, 1)
        lum = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2])[None, None]
        bth = torch.zeros_like(lum)
        for k in (7, 15):
            clo = -F.max_pool2d(-F.max_pool2d(lum, k, 1, k // 2), k, 1, k // 2)
            bth = torch.maximum(bth, (clo - lum).clamp_min(0))
        skin = (F.max_pool2d(it["mask"][None], 15, 1, 7)[0, 0] <= 0.5).numpy()
        b = (bth[0, 0].numpy() > 0.15) & skin
        lab, _ = ndi.label(b, structure=np.ones((3, 3)))
        kept = 0
        for j, sl in enumerate(ndi.find_objects(lab), 1):
            h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
            area = int((lab[sl] == j).sum())
            if max(h, w) >= 20 and area / (h * w) < 0.35:
                kept += area
        res[it["id"]] = kept / max(int(skin.sum()), 1)
    with open(SCORES, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["id", "score"])
        for k, v in sorted(res.items(), key=lambda t: -t[1]):
            w.writerow([k, f"{v:.6f}"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="folder holding <run>/fold_k and <run>/ensemble (default $HARS_RUNS)")
    ap.add_argument("--rescore", action="store_true", help="recompute the hair-heavy scores from the image cache")
    a = ap.parse_args()
    root = Path(a.root) if a.root else RUNS
    if a.rescore or not SCORES.exists():
        hair_scores()
    with open(SCORES, newline="") as f:
        hair = [r["id"] for r in csv.DictReader(f)][:200]

    ens, rows = {}, []
    for name, label, agss, syn in VARIANTS:
        ev = [json.loads((root / name / f"fold_{k}" / "test_eval.json").read_text()) for k in range(1, N_FOLDS + 1)]
        dn = [json.loads((root / name / f"fold_{k}" / "done.json").read_text()) for k in range(1, N_FOLDS + 1)]
        E = per_image(root / name / "ensemble" / "test_per_image.csv")
        ids = sorted(E)
        dice = np.array([float(E[i]["dice"]) for i in ids])
        area = np.array([(int(E[i]["tp"]) + int(E[i]["fn"])) / SIZE ** 2 for i in ids])
        d = np.array([e["dice_img"] for e in ev]); iou = np.array([e["iou_img"] for e in ev])
        ens[name] = ({i: float(E[i]["dice"]) for i in ids}, d)
        rows.append([label, agss, syn, f"{np.mean([x['best_val_dice_img'] for x in dn]):.4f}",
                     f"{d.mean():.4f} ± {d.std(ddof=1):.4f}", f"{iou.mean():.4f} ± {iou.std(ddof=1):.4f}",
                     f"{dice.mean():.4f}", f"{dice[area > 0.6].mean():.4f}",
                     f"{np.mean([float(E[i]['dice']) for i in hair]):.4f}", str(int((dice < 0.5).sum()))])

    comps, p_all, p_hair = [], [], []
    for a_, b_, what in COMPARISONS:
        A, fa = ens[a_]; B, fb = ens[b_]; ids = sorted(A)
        x = np.array([A[i] for i in ids]); y = np.array([B[i] for i in ids]); dlt = x - y
        xh = np.array([A[i] for i in hair]); yh = np.array([B[i] for i in hair])
        p_all.append(wilcoxon(x, y).pvalue); p_hair.append(wilcoxon(xh, yh).pvalue)
        comps.append([f"{a_} vs {b_}", what, f"{dlt.mean():+.4f} / {np.median(dlt):+.4f}",
                      f"{(dlt > 0).sum()} / {(dlt < 0).sum()}", f"{(fa > fb).sum()} / {N_FOLDS}", (xh - yh).mean()])
    pa, ph = holm(p_all), holm(p_hair)

    md = ["## Table 5. Ablation (official ISIC 2018 test set, 5 folds)", "",
          "| Configuration | AGSS | Synthetic hair | Val Dice | Test Dice (fold mean ± SD) | Test IoU (fold mean ± SD) "
          "| Ensemble Dice | Ensemble Dice, lesions > 60% | Ensemble Dice, hair-heavy 20% | Images with Dice < 0.5 |",
          "|" + "---|" * 10]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    md += ["", "## Table 6. Paired comparisons of the ensembles (Wilcoxon signed-rank, Holm-adjusted over 3 tests)", "",
           "| Comparison | Isolates | Δ Dice, mean / median | Images better / worse | Folds better | p | Δ Dice, hair-heavy (p) |",
           "|" + "---|" * 7]
    for c, pa_, ph_ in zip(comps, pa, ph):
        md.append(f"| {c[0]} | {c[1]} | {c[2]} | {c[3]} | {c[4]} | {pa_:.2g} | {c[5]:+.4f} ({ph_:.2g}) |")
    text = "\n".join(md) + "\n"
    (root / "ablation_tables.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
