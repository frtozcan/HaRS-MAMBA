"""Scan routes that turn a 2D feature map into 1D token sequences.

The four fixed routes of a VMamba-style cross-scan are used throughout HaRS-Mamba:

    row, row_rev, col, col_rev

A route is a permutation of the flattened feature map, so it costs one ``gather`` before
the scan and one inverse ``gather`` after it.  Any other route that can be written as a
permutation (shared by the batch, shape (L,), or per image, shape (b, L)) can be passed to
the mixer in the same way.
"""
from __future__ import annotations

import torch

FIXED_ROUTES = ("row", "row_rev", "col", "col_rev")

_cache: dict = {}


def fixed_route(name: str, h: int, w: int, device) -> torch.Tensor:
    """Permutation (L,) such that seq = flat[perm]."""
    key = (name, h, w, str(device))
    if key not in _cache:
        idx = torch.arange(h * w, device=device)
        if name == "row":
            perm = idx
        elif name == "row_rev":
            perm = idx.flip(0)
        elif name == "col":
            perm = idx.view(h, w).t().reshape(-1)
        elif name == "col_rev":
            perm = idx.view(h, w).t().reshape(-1).flip(0)
        else:
            raise ValueError(name)
        _cache[key] = perm
    return _cache[key]


def inverse_perm(perm: torch.Tensor) -> torch.Tensor:
    """Inverse of a permutation along the last dim."""
    inv = torch.empty_like(perm)
    src = torch.arange(perm.shape[-1], device=perm.device).expand_as(perm)
    return inv.scatter_(-1, perm, src)
