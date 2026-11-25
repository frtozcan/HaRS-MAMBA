"""Build 5 folds over the official ISIC2018 train (2594) + val (100) images.

* The official TEST set (1000) is not used here.
* Stratified by lesion-area quintile, so large lesions (the weak case) are spread evenly.
* Near-duplicate images (16x16 difference hash, Hamming distance <= HAM_MAX of 256 bits, also under flips/rot90)
  are grouped and kept in the same fold to limit leakage between train and val.
  Lesion IDs are not available for these ISIC images, so this is the best available guard.
"""
from __future__ import annotations

import csv
import json

import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from config import CACHE, DATASET, FOLDS_JSON, N_FOLDS, SIZE

SEED = 42
HAM_MAX = 20


def dhash(img: Image.Image, n=16):
    g = np.asarray(img.convert("L").resize((n + 1, n), Image.BILINEAR), dtype=np.float32)
    return (g[:, 1:] > g[:, :-1]).flatten()


def main():
    out = FOLDS_JSON.parent
    out.mkdir(parents=True, exist_ok=True)
    base = CACHE / f"{DATASET}_{SIZE}"
    ids, source, area, hashes = [], {}, [], []
    for split in ("train", "val"):
        for p in sorted((base / split / "images").glob("*.png")):
            sid = p.stem
            ids.append(sid)
            source[sid] = split
            m = np.asarray(Image.open(base / split / "masks" / f"{sid}.png")) > 127
            area.append(float(m.mean()))
            im = Image.open(p)
            variants = [im, im.transpose(Image.FLIP_LEFT_RIGHT), im.transpose(Image.FLIP_TOP_BOTTOM),
                        im.transpose(Image.ROTATE_90), im.transpose(Image.ROTATE_180), im.transpose(Image.ROTATE_270)]
            hashes.append(np.stack([dhash(v) for v in variants]))
    n = len(ids)
    H = np.stack(hashes)                                   # (n, 6, 64)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    P = np.packbits(H, axis=-1)                            # (n, 6, 8) uint8
    popcnt = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    pairs = []
    for v in range(P.shape[1]):
        d = popcnt[P[:, None, 0, :] ^ P[None, :, v, :]].sum(-1, dtype=np.int32)   # (n, n)
        ii, jj = np.where(np.triu(d <= HAM_MAX, 1))
        pairs += list(zip(ii.tolist(), jj.tolist()))
    for a, b in set(pairs):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    groups = np.array([find(i) for i in range(n)])
    dup_groups = {g: [ids[i] for i in np.where(groups == g)[0]] for g in set(groups.tolist())}
    dup_groups = {g: v for g, v in dup_groups.items() if len(v) > 1}

    area = np.array(area)
    bins = np.digitize(area, np.quantile(area, [0.2, 0.4, 0.6, 0.8]))
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = {}
    for k, (tr, va) in enumerate(sgkf.split(np.zeros(n), bins, groups), start=1):
        folds[str(k)] = {"train": [ids[i] for i in tr], "val": [ids[i] for i in va]}

    (out / "folds.json").write_text(json.dumps({"seed": SEED, "n_images": n, "source": source,
                                                "folds": folds}, indent=1))
    (out / "near_duplicate_groups.json").write_text(json.dumps(list(dup_groups.values()), indent=1))
    with open(out / "folds_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold", "n_train", "n_val", "val_from_official_val", "val_mean_lesion_area",
                    "val_n_lesion_area_gt_60pct"])
        idx = {s: i for i, s in enumerate(ids)}
        for k, d in folds.items():
            va = [idx[s] for s in d["val"]]
            w.writerow([k, len(d["train"]), len(d["val"]), sum(source[s] == "val" for s in d["val"]),
                        f"{area[va].mean():.3f}", int((area[va] > 0.6).sum())])
    sizes = sorted((len(v) for v in dup_groups.values()), reverse=True)
    print("largest near-duplicate groups:", sizes[:10])
    print(f"{n} images, {len(dup_groups)} near-duplicate groups "
          f"({sum(len(v) for v in dup_groups.values())} images); folds -> {out}")
    for k, d in folds.items():
        print(f"  fold {k}: train {len(d['train'])}  val {len(d['val'])}")


if __name__ == "__main__":
    main()
