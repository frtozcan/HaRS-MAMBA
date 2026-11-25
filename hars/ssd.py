"""Pure-PyTorch selective state-space scan (Mamba-2 / SSD formulation).

No CUDA/Triton extension is needed (unlike ``mamba_ssm``), so the same code
runs on NVIDIA GPUs, CPUs, Apple MPS and DirectML.  The chunked algorithm
follows the "state space duality" decomposition of Dao & Gu (2024):
intra-chunk terms are computed as small masked matrix products and the
inter-chunk recurrence is carried by a tiny chunk-level scan.

Shapes used throughout
    x  : (b, l, h, p)  input sequence, h heads with head dimension p
    dt : (b, l, h)     positive step sizes (Delta)
    A  : (h,)          negative per-head decay rates
    B  : (b, l, h, n)  input projections  (state size n)
    C  : (b, l, h, n)  output projections
returns y : (b, l, h, p)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _segsum(x: torch.Tensor) -> torch.Tensor:
    """Stable segment sum.  x: (..., T) -> (..., T, T) with
    out[..., i, j] = sum_{k=j+1..i} x[..., k] for i >= j, -inf above the diagonal."""
    T = x.size(-1)
    x = x.unsqueeze(-1).expand(*x.shape, T)                     # (..., T, T): x[..., i, j] = x_i
    strict = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=-1)
    x = x.masked_fill(~strict, 0.0)
    out = torch.cumsum(x, dim=-2)
    lower = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=0)
    return out.masked_fill(~lower, float("-inf"))


def ssd_scan(x, dt, A, B, C, chunk: int = 64) -> torch.Tensor:
    """Chunked selective scan h_t = exp(dt_t A) h_{t-1} + dt_t B_t x_t,  y_t = C_t h_t.

    Always computed in float32 for numerical safety (also under autocast)."""
    in_dtype = x.dtype
    with torch.autocast(device_type=x.device.type, enabled=False):
        x, dt, B, C = x.float(), dt.float(), B.float(), C.float()
        A = A.float()
        b, l, h, p = x.shape
        n = B.shape[-1]
        pad = (-l) % chunk
        if pad:
            # zero-padding at the END never influences earlier (causal) outputs
            x = F.pad(x, (0, 0, 0, 0, 0, pad))
            dt = F.pad(dt, (0, 0, 0, pad))
            B = F.pad(B, (0, 0, 0, 0, 0, pad))
            C = F.pad(C, (0, 0, 0, 0, 0, pad))
        L = l + pad
        c = L // chunk

        xd = (x * dt.unsqueeze(-1)).view(b, c, chunk, h, p)        # discretised input
        a = (dt * A.view(1, 1, h)).view(b, c, chunk, h)             # log-decay per step (<= 0)
        Bc = B.view(b, c, chunk, h, n)
        Cc = C.view(b, c, chunk, h, n)

        a = a.permute(0, 3, 1, 2)                                   # (b, h, c, T)
        a_cum = torch.cumsum(a, dim=-1)

        # 1) intra-chunk (diagonal blocks)
        Lmat = torch.exp(_segsum(a))                                # (b, h, c, T, T)
        CB = torch.einsum("bclhn,bcshn->bhcls", Cc, Bc)            # (b, h, c, T, T)
        y_diag = torch.einsum("bhcls,bcshp->bclhp", CB * Lmat, xd)

        # 2) final state of every chunk
        decay_states = torch.exp(a_cum[..., -1:] - a_cum)           # (b, h, c, T)
        states = torch.einsum("bclhn,bhcl,bclhp->bchpn", Bc, decay_states, xd)

        # 3) recurrence across chunks
        init = torch.zeros_like(states[:, :1])
        states = torch.cat([init, states], dim=1)                   # (b, c+1, h, p, n)
        decay_chunk = torch.exp(_segsum(F.pad(a_cum[..., -1], (1, 0))))  # (b, h, c+1, c+1)
        new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
        states = new_states[:, :-1]                                 # state entering each chunk

        # 4) contribution of the incoming state
        y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", Cc, states, torch.exp(a_cum))

        y = (y_diag + y_off).reshape(b, L, h, p)[:, :l]
    return y.to(in_dtype)


def ssd_reference(x, dt, A, B, C) -> torch.Tensor:
    """Slow step-by-step recurrence, used only to unit-test :func:`ssd_scan`."""
    b, l, h, p = x.shape
    n = B.shape[-1]
    state = x.new_zeros(b, h, p, n)
    ys = []
    for t in range(l):
        decay = torch.exp(dt[:, t] * A.view(1, h))                  # (b, h)
        state = state * decay[..., None, None] + \
            (dt[:, t, :, None, None] * x[:, t, :, :, None] * B[:, t, :, None, :])
        ys.append(torch.einsum("bhpn,bhn->bhp", state, C[:, t]))
    return torch.stack(ys, dim=1)
