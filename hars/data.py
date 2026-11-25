"""Datasets: ISIC 2017, ISIC 2018 (Task 1) and PH2.

Raw data can be either extracted folders or the original .zip files (read directly, no
extraction needed).  ``prepare_data.py`` resizes everything once into a small PNG cache;
training/evaluation then read only that cache.
"""
from __future__ import annotations

import io
import json
import math
import random
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter

from .artifact import IMAGENET_MEAN, IMAGENET_STD

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

SPECS = {
    "isic2018": {
        "subdir": "ISIC2018",
        "splits": {
            "train": ("ISIC2018_Task1-2_Training_Input", "ISIC2018_Task1_Training_GroundTruth"),
            "val": ("ISIC2018_Task1-2_Validation_Input", "ISIC2018_Task1_Validation_GroundTruth"),
            "test": ("ISIC2018_Task1-2_Test_Input", "ISIC2018_Task1_Test_GroundTruth"),
        },
    },
    "isic2017": {
        "subdir": "ISIC2017",
        "splits": {
            "train": ("ISIC-2017_Training_Data", "ISIC-2017_Training_Part1_GroundTruth"),
            "val": ("ISIC-2017_Validation_Data", "ISIC-2017_Validation_Part1_GroundTruth"),
            "test": ("ISIC-2017_Test_v2_Data", "ISIC-2017_Test_v2_Part1_GroundTruth"),
        },
    },
    # PH2: one folder (any name containing "PH2") with IMDxxx/IMDxxx_Dermoscopic_Image + IMDxxx_lesion
    "ph2": {"subdir": "PH2", "splits": {"test": (None, None)}},
}


# ----------------------------------------------------------------------------- raw sources
class Source:
    """A folder or a .zip with images, addressed by sample id."""

    def __init__(self, path: Path, kind: str):
        self.path = Path(path)
        self.kind = kind                     # "image" | "mask" | "ph2_image" | "ph2_mask"
        self.is_zip = self.path.suffix.lower() == ".zip"
        self._zf = None
        names = self._list()
        self.items = {}
        for n in names:
            p = Path(n)
            if p.suffix.lower() not in IMG_EXT:
                continue
            sid = self._sample_id(p)
            if sid:
                self.items[sid] = n

    def _list(self):
        if self.is_zip:
            with zipfile.ZipFile(self.path) as zf:
                return [n for n in zf.namelist() if not n.endswith("/")]
        return [str(p) for p in self.path.rglob("*") if p.is_file()]

    def _sample_id(self, p: Path):
        stem = p.stem
        if self.kind == "image":
            return None if ("superpixels" in stem or not stem.startswith("ISIC_")) else stem
        if self.kind == "mask":
            return stem[: -len("_segmentation")] if stem.endswith("_segmentation") else None
        if self.kind == "ph2_image":
            return stem if p.parent.name.endswith("_Dermoscopic_Image") else None
        if self.kind == "ph2_mask":
            return stem[: -len("_lesion")] if stem.endswith("_lesion") else None
        raise ValueError(self.kind)

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_zf"] = None
        return d

    def open(self, sid: str, size_hint: int | None = None) -> Image.Image:
        name = self.items[sid]
        if self.is_zip:
            if self._zf is None:
                self._zf = zipfile.ZipFile(self.path)
            img = Image.open(io.BytesIO(self._zf.read(name)))
        else:
            img = Image.open(name)
        if size_hint and img.format == "JPEG":
            img.draft("RGB", (size_hint, size_hint))     # fast JPEG decoding at reduced scale
        return img


def _find(base: Path, name: str):
    """Locate `name` as a folder (preferred) or `name`.zip below base."""
    for cand in (base / name, base / name / name):
        if cand.is_dir() and any(cand.iterdir()):
            return cand
    z = base / f"{name}.zip"
    if z.is_file():
        return z
    return None


def find_sources(root: Path, dataset: str, split: str):
    root = Path(root)
    spec = SPECS[dataset]
    if dataset == "ph2":
        cands = [root] + [p for p in root.iterdir() if p.is_dir() and "PH2" in p.name.upper()]
        for c in cands:
            imgs = Source(c, "ph2_image")
            if imgs.items:
                return imgs, Source(c, "ph2_mask")
        raise FileNotFoundError(f"PH2 images not found below {root}")
    img_name, mask_name = spec["splits"][split]
    for base in (root / spec["subdir"], root):
        ip, mp = _find(base, img_name), _find(base, mask_name)
        if ip is not None and mp is not None:
            return Source(ip, "image"), Source(mp, "mask")
    raise FileNotFoundError(
        f"{dataset}/{split}: could not find '{img_name}' and '{mask_name}' (folder or .zip) under {root}")


# ----------------------------------------------------------------------------- cached dataset
def cache_dir(cache_root, dataset, size):
    return Path(cache_root) / f"{dataset}_{size}"


def read_split(cache_root, dataset, size, split):
    d = cache_dir(cache_root, dataset, size) / split
    ids = sorted(p.stem for p in (d / "images").glob("*.png"))
    if not ids:
        raise FileNotFoundError(f"No cached images in {d}. Run prepare_data.py first.")
    return d, ids


def _to_tensor(img: Image.Image) -> torch.Tensor:
    a = torch.from_numpy(np.asarray(img, dtype=np.uint8).copy())
    if a.dim() == 2:
        a = a[None]
    else:
        a = a.permute(2, 0, 1)
    return a.float() / 255.0


class SyntheticHair:
    """Draw random dark curved hairs; returns the image and the exact hair mask."""

    def __init__(self, max_hairs=20, p=0.5):
        self.max_hairs = max_hairs
        self.p = p

    def __call__(self, img: Image.Image):
        w, h = img.size
        hair = Image.new("L", (w, h), 0)
        if random.random() > self.p:
            return img, hair
        draw = ImageDraw.Draw(hair)
        s = w / 256.0
        for _ in range(random.randint(2, self.max_hairs)):
            x0, y0 = random.uniform(-0.1, 1.1) * w, random.uniform(-0.1, 1.1) * h
            ang = random.uniform(0, 2 * math.pi)
            length = random.uniform(0.2, 0.9) * w
            x2, y2 = x0 + length * math.cos(ang), y0 + length * math.sin(ang)
            bend = random.uniform(-0.35, 0.35) * length
            cx = (x0 + x2) / 2 - bend * math.sin(ang)
            cy = (y0 + y2) / 2 + bend * math.cos(ang)
            pts = []
            for t in np.linspace(0, 1, 24):
                pts.append(((1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t ** 2 * x2,
                            (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t ** 2 * y2))
            draw.line(pts, fill=255, width=max(1, int(round(random.uniform(0.8, 2.6) * s))), joint="curve")
        soft = np.asarray(hair.filter(ImageFilter.GaussianBlur(0.6 * s)), dtype=np.float32) / 255.0
        arr = np.asarray(img, dtype=np.float32)
        dark = np.percentile(arr.mean(-1), 3)
        colour = np.array([random.uniform(0.15, 0.6), random.uniform(0.1, 0.45), random.uniform(0.05, 0.35)])
        colour = colour / colour.max() * dark * random.uniform(0.4, 1.0)
        alpha = (soft * random.uniform(0.7, 1.0))[..., None]
        arr = arr * (1 - alpha) + colour[None, None] * alpha
        img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
        return img, Image.fromarray(((soft > 0.25) * 255).astype(np.uint8))


class SkinLesionDataset(torch.utils.data.Dataset):
    def __init__(self, cache_root, dataset, size, split, train=False, synthetic_hair=0.5,
                 ids=None):
        self.dir, all_ids = read_split(cache_root, dataset, size, split)
        self.ids = ids if ids is not None else all_ids
        self.train = train
        self.size = size
        self.hair = SyntheticHair(p=synthetic_hair) if (train and synthetic_hair > 0) else None
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __len__(self):
        return len(self.ids)

    def _augment(self, img, mask, hair):
        # geometric (identical for image, mask and hair mask)
        if random.random() < 0.5:
            img, mask, hair = img.flip(-1), mask.flip(-1), hair.flip(-1)
        if random.random() < 0.5:
            img, mask, hair = img.flip(-2), mask.flip(-2), hair.flip(-2)
        k = random.randint(0, 3)
        if k:
            img, mask, hair = (torch.rot90(t, k, (-2, -1)) for t in (img, mask, hair))
        if random.random() < 0.5:
            a = math.radians(random.uniform(-30, 30))
            sc = random.uniform(0.85, 1.2)
            tx, ty = random.uniform(-0.08, 0.08), random.uniform(-0.08, 0.08)
            theta = torch.tensor([[math.cos(a) / sc, -math.sin(a) / sc, tx],
                                  [math.sin(a) / sc, math.cos(a) / sc, ty]], dtype=torch.float32)
            grid = F.affine_grid(theta[None], (1, 1, self.size, self.size), align_corners=False)
            stack = torch.cat([img, mask, hair], 0)[None]
            stack = F.grid_sample(stack, grid, mode="bilinear", padding_mode="reflection",
                                  align_corners=False)[0]
            img, mask, hair = stack[:3], (stack[3:4] > 0.5).float(), (stack[4:5] > 0.5).float()
        # photometric
        if random.random() < 0.8:
            img = img * random.uniform(0.8, 1.2)                                    # brightness
            m = img.mean()
            img = (img - m) * random.uniform(0.8, 1.2) + m                          # contrast
            g = img.mean(0, keepdim=True)
            img = (img - g) * random.uniform(0.8, 1.2) + g                          # saturation
            img = img.clamp(0, 1)
        return img, mask, hair

    def __getitem__(self, i):
        sid = self.ids[i]
        img = Image.open(self.dir / "images" / f"{sid}.png").convert("RGB")
        mask = Image.open(self.dir / "masks" / f"{sid}.png").convert("L")
        if self.hair is not None:
            img, hair = self.hair(img)
        else:
            hair = Image.new("L", img.size, 0)
        img, mask, hair = _to_tensor(img), (_to_tensor(mask) > 0.5).float(), (_to_tensor(hair) > 0.5).float()
        if self.train:
            img, mask, hair = self._augment(img, mask, hair)
        img = (img - self.mean) / self.std
        return {"image": img, "mask": mask, "hair": hair, "id": sid}


def load_index(cache_root, dataset, size):
    p = cache_dir(cache_root, dataset, size) / "index.json"
    return json.loads(p.read_text()) if p.exists() else {}
