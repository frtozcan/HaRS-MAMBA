"""Evaluate the 5 fold models of a run on the official ISIC 2018 TEST set (1000 images).

    python evaluate_folds.py --name hars_mamba

* each fold's best.pt (selected on that fold's validation split, never on test);
* the 5-model ensemble (mean probability), without and with flip TTA;
* the ensemble at the ORIGINAL mask resolution (stricter, challenge-style).
Outputs (all under $HARS_RUNS/<name>/):
  fold_<k>/test_eval.json, test_per_image.csv, test_pred/*.png
  ensemble/test_eval*.json, test_per_image*.csv, test_pred/*.png, test_prob/*.png
  summary/test_results.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from config import CACHE, DATASET, N_FOLDS, RAW, SIZE, run_dir
from hars import build_model
from hars.data import SkinLesionDataset, find_sources
from hars.metrics import METRICS, MetricTracker


def write_eval(tracker, ids, json_path, csv_path, extra):
    s = tracker.summary()
    conf, per = tracker.per_image()
    s["thresholded_jaccard"] = float(np.mean(np.where(per["iou"] >= 0.65, per["iou"], 0.0)))
    s["n_dice_lt_0.5"] = int((per["dice"] < 0.5).sum())
    s.update(extra)
    json_path.write_text(json.dumps(s, indent=2))
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", *METRICS, "tp", "fp", "fn", "tn"])
        for i, sid in enumerate(ids):
            w.writerow([sid, *[f"{per[m][i]:.4f}" for m in METRICS], *conf[i].astype(int).tolist()])
    return s


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N test images")
    ap.add_argument("--name", default="hars_mamba", help="run name under $HARS_RUNS")
    ap.add_argument("--root", default=None, help="override the run folder")
    a = ap.parse_args()
    root = Path(a.root) if a.root else run_dir(a.name)
    fdir = lambda k: root / f"fold_{k}"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = dev.type == "cuda"
    ds = SkinLesionDataset(CACHE, DATASET, SIZE, "test")
    if a.limit:
        ds.ids = ds.ids[: a.limit]
    dl = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)
    n = len(ds)
    ens = torch.zeros(n, 1, SIZE, SIZE)
    ens_tta = torch.zeros(n, 1, SIZE, SIZE)
    gts = torch.zeros(n, 1, SIZE, SIZE, dtype=torch.bool)
    ids, rows, folds_used = list(ds.ids), [], []

    for k in range(1, N_FOLDS + 1):
        ck_path = fdir(k) / "best.pt"
        if not ck_path.exists():
            print(f"fold {k}: no best.pt, skipped")
            continue
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        ca = ck["args"]
        model = build_model(ca.get("model", "base"), use_artifact=not ca.get("no_artifact", False)).to(dev).eval()
        model.load_state_dict(ck["model"])
        pdir = fdir(k) / "test_pred"
        pdir.mkdir(exist_ok=True)
        tr, j, n_fp32 = MetricTracker(), 0, [0]
        for b in dl:
            x = b["image"].to(dev)
            # fp16 inference; any view with non-finite logits (fp16 overflow, seen in models that finished
            # training in fp32 after a NaN rollback) is recomputed in fp32.
            def seg(v):
                with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
                    o = model(v)["seg"].float()
                bad = ~torch.isfinite(o).flatten(1).all(1)
                if bad.any():
                    n_fp32[0] += int(bad.sum())
                    o[bad] = model(v[bad].float())["seg"].float()
                return torch.sigmoid(o)
            p = seg(x)
            pt = (p + seg(x.flip(-1)).flip(-1) + seg(x.flip(-2)).flip(-2)) / 3
            m = b["mask"] > 0.5
            tr.update(p.cpu() > 0.5, m)
            bs = x.shape[0]
            ens[j:j + bs] += p.float().cpu()
            ens_tta[j:j + bs] += pt.float().cpu()
            gts[j:j + bs] = m
            for sid, pm in zip(b["id"], (p > 0.5).cpu()):
                Image.fromarray(pm[0].numpy().astype(np.uint8) * 255).save(pdir / f"{sid}.png")
            j += bs
        s = write_eval(tr, ids, fdir(k) / "test_eval.json", fdir(k) / "test_per_image.csv",
                       {"fold": k, "best_epoch": ck["epoch"] + 1, "val_dice_img": ck["val"]["dice_img"]})
        rows.append({"model": f"fold_{k}", **s})
        folds_used.append(k)
        print(f"fold {k}: test Dice {s['dice_img']:.4f}  IoU {s['iou_img']:.4f}  (fp32 fallback on {n_fp32[0]} views)")

    if not folds_used:
        raise SystemExit("no fold checkpoints found")
    nf = len(folds_used)
    ens /= nf
    ens_tta /= nf
    ed = root / "ensemble"
    (ed / "test_pred").mkdir(parents=True, exist_ok=True)
    (ed / "test_prob").mkdir(parents=True, exist_ok=True)
    for name, P, suffix in (("ensemble", ens, ""), ("ensemble_tta", ens_tta, "_tta")):
        tr = MetricTracker()
        for i in range(0, n, 64):
            tr.update(P[i:i + 64] > 0.5, gts[i:i + 64])
        s = write_eval(tr, ids, ed / f"test_eval{suffix}.json", ed / f"test_per_image{suffix}.csv",
                       {"folds": folds_used, "tta": bool(suffix)})
        rows.append({"model": name, **s})
        print(f"{name}: test Dice {s['dice_img']:.4f}  IoU {s['iou_img']:.4f}")
    for i, sid in enumerate(ids):
        Image.fromarray((ens[i, 0].numpy() > 0.5).astype(np.uint8) * 255).save(ed / "test_pred" / f"{sid}.png")
        Image.fromarray((ens[i, 0].numpy() * 255).round().astype(np.uint8)).save(ed / "test_prob" / f"{sid}.png")

    # ensemble at the original resolution of the ground-truth masks
    _, msk_src = find_sources(Path(RAW), DATASET, "test")
    tr = MetricTracker()
    for i, sid in enumerate(ids):
        gt = np.asarray(msk_src.open(sid).convert("L")) > 127
        p = F.interpolate(ens[i:i + 1], size=gt.shape, mode="bilinear", align_corners=False) > 0.5
        tr.update(p, torch.from_numpy(gt)[None, None])
    s = write_eval(tr, ids, ed / "test_eval_origres.json", ed / "test_per_image_origres.csv",
                   {"folds": folds_used, "orig_res": True})
    rows.append({"model": "ensemble_origres", **s})
    print(f"ensemble (original resolution): test Dice {s['dice_img']:.4f}  IoU {s['iou_img']:.4f}")

    sd = root / "summary"
    sd.mkdir(exist_ok=True)
    keys = ["model", "dice_img", "iou_img", "sensitivity_img", "specificity_img", "accuracy_img",
            "dice_global", "iou_global", "thresholded_jaccard", "n_dice_lt_0.5", "n_images"]
    with open(sd / "test_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for r in rows:
            w.writerow([r["model"]] + [f"{r[k]:.4f}" if isinstance(r[k], float) else r[k] for k in keys[1:]])
    print(f"-> {sd / 'test_results.csv'}")


if __name__ == "__main__":
    main()
