"""Resize raw ISIC 2017 / ISIC 2018 / PH2 data once into a PNG cache.

Examples (run from the HaRS-Mamba folder):
    python prepare_data.py --root ..  --dataset isic2018 --size 256
    python prepare_data.py --root ..  --dataset isic2017 --size 256
    python prepare_data.py --root ..\\PH2 --dataset ph2 --size 256

--root is the Skin_Canser folder (the one that contains ISIC2017 / ISIC2018).  Image
folders or the original .zip files both work; zips are read directly.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from hars.data import SPECS, cache_dir, find_sources

_IMG = _MSK = None
_OUT = None
_SIZE = None


def _init(img_src, msk_src, out_dir, size):
    global _IMG, _MSK, _OUT, _SIZE
    _IMG, _MSK, _OUT, _SIZE = img_src, msk_src, Path(out_dir), size


def _process(sid):
    out_img = _OUT / "images" / f"{sid}.png"
    out_msk = _OUT / "masks" / f"{sid}.png"
    if out_img.exists() and out_msk.exists():
        return sid, None
    img = _IMG.open(sid, size_hint=2 * _SIZE).convert("RGB")
    msk = _MSK.open(sid).convert("L")
    orig = msk.size
    img = img.resize((_SIZE, _SIZE), Image.BICUBIC)
    m = np.asarray(msk.resize((_SIZE, _SIZE), Image.BILINEAR), dtype=np.uint8)
    m = ((m > 127) * 255).astype(np.uint8)
    img.save(out_img)
    Image.fromarray(m).save(out_msk)
    return sid, orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Skin_Canser folder (or the PH2 folder)")
    ap.add_argument("--dataset", required=True, choices=list(SPECS))
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--cache", default="cache", help="output cache folder")
    ap.add_argument("--splits", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="only the first N samples per split (quick tests)")
    args = ap.parse_args()

    splits = args.splits or list(SPECS[args.dataset]["splits"])
    out_root = cache_dir(args.cache, args.dataset, args.size)
    index = {"dataset": args.dataset, "size": args.size, "splits": {}, "orig_size": {}}
    for split in splits:
        t0 = time.time()
        img_src, msk_src = find_sources(Path(args.root), args.dataset, split)
        ids = sorted(set(img_src.items) & set(msk_src.items))
        missing = len(img_src.items) - len(ids)
        if args.limit:
            ids = ids[: args.limit]
        out = out_root / split
        (out / "images").mkdir(parents=True, exist_ok=True)
        (out / "masks").mkdir(parents=True, exist_ok=True)
        print(f"[{args.dataset}/{split}] {len(ids)} pairs  (images: {img_src.path.name}, "
              f"masks: {msk_src.path.name}; {missing} images without mask skipped)", flush=True)
        done = 0
        with ProcessPoolExecutor(max_workers=max(1, args.workers), initializer=_init,
                                 initargs=(img_src, msk_src, out, args.size)) as ex:
            for sid, orig in ex.map(_process, ids, chunksize=8):
                if orig is not None:
                    index["orig_size"][sid] = list(orig)
                done += 1
                if done % 200 == 0 or done == len(ids):
                    print(f"   {done}/{len(ids)}  {time.time() - t0:.0f}s", flush=True)
        index["splits"][split] = len(ids)

    old = out_root / "index.json"
    if old.exists():
        prev = json.loads(old.read_text())
        prev["splits"].update(index["splits"])
        prev.setdefault("orig_size", {}).update(index["orig_size"])
        index = prev
    old.write_text(json.dumps(index))
    print("cache ready:", out_root.resolve())


if __name__ == "__main__":
    main()
