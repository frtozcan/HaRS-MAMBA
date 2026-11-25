"""Extra statistics for the manuscript: bootstrap CIs, paired tests, hair-map statistics, speed.
Writes summary/paper_stats.json and summary/paper_stats.md."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon

from config import CACHE, DATASET, SIZE, run_dir
from hars import build_model, count_parameters
from hars.data import SkinLesionDataset

rng = np.random.default_rng(0)


def col(path, key="dice"):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {r["id"]: float(r[key]) for r in rows}


def boot(v, n=10000):
    v = np.asarray(v)
    idx = rng.integers(0, len(v), (n, len(v)))
    m = v[idx].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


_ap = argparse.ArgumentParser()
_ap.add_argument("--name", default="hars_mamba", help="run name under $HARS_RUNS")
_ap.add_argument("--root", default=None, help="override the run folder")
_a = _ap.parse_args()
ROOT = Path(_a.root) if _a.root else run_dir(_a.name)
(ROOT / "summary").mkdir(exist_ok=True)
out = {"root": str(ROOT)}
E = ROOT / "ensemble"
ens, tta = col(E / "test_per_image.csv"), col(E / "test_per_image_tta.csv")
ens_iou, tta_iou = col(E / "test_per_image.csv", "iou"), col(E / "test_per_image_tta.csv", "iou")
ids = sorted(ens)
for name, d in (("ensemble_dice", ens), ("ensemble_tta_dice", tta), ("ensemble_iou", ens_iou), ("ensemble_tta_iou", tta_iou)):
    v = np.array([d[i] for i in ids])
    out[name] = {"mean": float(v.mean()), "ci95": boot(v), "median": float(np.median(v)),
                 "q1": float(np.percentile(v, 25)), "q3": float(np.percentile(v, 75)), "sd": float(v.std(ddof=1))}
folds = {k: col(ROOT / f"fold_{k}" / "test_per_image.csv") for k in range(1, 6)}
out["wilcoxon_ensemble_vs_fold"] = {}
for k, d in folds.items():
    a, b = np.array([ens[i] for i in ids]), np.array([d[i] for i in ids])
    out["wilcoxon_ensemble_vs_fold"][k] = {"mean_diff": float((a - b).mean()), "p": float(wilcoxon(a, b).pvalue)}
a, b = np.array([tta[i] for i in ids]), np.array([ens[i] for i in ids])
out["wilcoxon_tta_vs_ensemble"] = {"mean_diff": float((a - b).mean()), "p": float(wilcoxon(a, b).pvalue)}

# hair-map statistics with the best fold model on all 1000 test images
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_k = max(range(1, 6), key=lambda k: json.loads((ROOT / f"fold_{k}" / "test_eval.json").read_text())["dice_img"])
ck = torch.load(ROOT / f"fold_{best_k}" / "best.pt", map_location="cpu", weights_only=False)
model = build_model("base", use_artifact=not ck["args"].get("no_artifact", False)).to(dev).eval()
model.load_state_dict(ck["model"])
ds = SkinLesionDataset(CACHE, DATASET, SIZE, "test")
dl = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
in_l, out_l, frac_in, frac_out = [], [], [], []
with torch.no_grad():
    for bt in dl:
        o = model(bt["image"].to(dev))
        h = torch.sigmoid(o["hair"].float()).cpu()
        m = bt["mask"] > 0.5
        for j in range(h.shape[0]):
            hi, mi = h[j, 0], m[j, 0]
            if mi.any():
                in_l.append(float(hi[mi].mean())); frac_in.append(float((hi[mi] > 0.5).float().mean()))
            if (~mi).any():
                out_l.append(float(hi[~mi].mean())); frac_out.append(float((hi[~mi] > 0.5).float().mean()))
out["hair_map"] = {"fold": best_k, "mean_prob_inside_lesion": float(np.mean(in_l)),
                   "mean_prob_outside_lesion": float(np.mean(out_l)),
                   "frac_px_gt05_inside_lesion": float(np.mean(frac_in)),
                   "frac_px_gt05_outside_lesion": float(np.mean(frac_out))}

# size and speed of the trained configuration (batch 1, 256x256)
x = torch.randn(1, 3, SIZE, SIZE, device=dev)
with torch.no_grad():
    for _ in range(20):
        model(x)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(200):
        model(x)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / 200 * 1000
from torch.utils.flop_counter import FlopCounterMode
with torch.no_grad(), FlopCounterMode(display=False) as fc:
    model(torch.randn(1, 3, SIZE, SIZE, device=dev))
out["model"] = {"params_M": count_parameters(model) / 1e6, "GMAC_256": fc.get_total_flops() / 2e9,
                "ms_per_image_bs1_fp32": ms, "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"}

(ROOT / "summary" / "paper_stats.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
