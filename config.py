"""Shared paths for the 5-fold experiments.

Set the data and output locations with environment variables (defaults in brackets):
    HARS_DATA  folder that contains ISIC2018/ and the PNG cache made by prepare_data.py   [./data]
    HARS_RUNS  folder for checkpoints, logs and predictions                                [./runs]
"""
from __future__ import annotations

import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent
DATA = Path(os.environ.get("HARS_DATA", REPO / "data"))
RUNS = Path(os.environ.get("HARS_RUNS", REPO / "runs"))
CACHE = DATA / "cache"                 # made by: python prepare_data.py --root $HARS_DATA --dataset isic2018 --cache $HARS_DATA/cache
RAW = DATA                             # contains ISIC2018/
FOLDS_JSON = REPO / "results" / "folds" / "folds.json"
DATASET, SIZE = "isic2018", 256
N_FOLDS = 5


def run_dir(name: str) -> Path:
    return RUNS / name


def load_folds():
    return json.loads(FOLDS_JSON.read_text())


def make_trainval_dataset(ids, source, train, synthetic_hair=0.5):
    """Dataset over ids that live in either the official train or val cache split."""
    import torch
    from hars.data import SkinLesionDataset
    parts = []
    for split in ("train", "val"):
        sub = [i for i in ids if source[i] == split]
        if sub:
            parts.append(SkinLesionDataset(CACHE, DATASET, SIZE, split, train=train,
                                           synthetic_hair=synthetic_hair if train else 0, ids=sub))
    return parts[0] if len(parts) == 1 else torch.utils.data.ConcatDataset(parts)
