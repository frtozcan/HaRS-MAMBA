"""Quick correctness checks.  Run:  python selftest.py   (takes < 1 min on CPU)"""
from __future__ import annotations

import torch

from hars import build_model, count_parameters
from hars.losses import HaRSLoss
from hars.scans import FIXED_ROUTES, fixed_route, inverse_perm
from hars.ssd import ssd_reference, ssd_scan


def check_ssd():
    torch.manual_seed(0)
    for (b, l, h, p, n, chunk) in [(2, 50, 3, 4, 5, 16), (1, 64, 2, 8, 16, 64), (2, 130, 4, 4, 8, 32)]:
        x = torch.randn(b, l, h, p, dtype=torch.float64)
        dt = torch.rand(b, l, h, dtype=torch.float64) * 0.5
        A = -torch.rand(h, dtype=torch.float64) * 4 - 0.5
        B = torch.randn(b, l, h, n, dtype=torch.float64)
        C = torch.randn(b, l, h, n, dtype=torch.float64)
        ref = ssd_reference(x, dt, A, B, C)
        out = ssd_scan(x.float(), dt.float(), A.float(), B.float(), C.float(), chunk=chunk).double()
        err = (out - ref).abs().max().item() / ref.abs().max().item()
        assert err < 1e-4, f"SSD mismatch {err}"
    print("ok  chunked SSD == step-by-step recurrence")


def check_skip():
    """Delta = 0 at token t  =>  the scan ignores x_t completely (artifact gating rationale)."""
    torch.manual_seed(1)
    b, l, h, p, n = 1, 40, 2, 4, 8
    x, B, C = torch.randn(b, l, h, p), torch.randn(b, l, h, n), torch.randn(b, l, h, n)
    dt, A = torch.rand(b, l, h) * 0.5 + 0.05, -torch.rand(h) - 0.5
    t = 17
    dt[:, t] = 0.0
    y1 = ssd_scan(x, dt, A, B, C, chunk=16)
    x2, B2 = x.clone(), B.clone()
    x2[:, t] += 100.0
    B2[:, t] += 100.0
    y2 = ssd_scan(x2, dt, A, B2, C, chunk=16)
    assert torch.allclose(y1[:, t + 1:], y2[:, t + 1:], atol=1e-4)
    print("ok  zero step size skips an occluded token")


def check_routes():
    for name in FIXED_ROUTES:
        perm = fixed_route(name, 7, 9, "cpu")
        assert torch.equal(perm.sort().values, torch.arange(63))
        assert torch.equal(perm[inverse_perm(perm)], torch.arange(63))
    print("ok  scan routes are valid permutations")


def check_model():
    torch.manual_seed(0)
    crit = HaRSLoss()
    for size, kw in [("tiny", {}), ("base", {}), ("base", {"use_artifact": False})]:
        m = build_model(size, **kw)
        x = torch.randn(2, 3, 128, 128)
        y = (torch.rand(2, 1, 128, 128) > 0.5).float()
        hm = torch.zeros(2, 1, 128, 128)
        hm[0, :, 60:62, :] = 1
        out = m(x)
        assert out["seg"].shape == (2, 1, 128, 128)
        loss, logs = crit(out, y, hm)
        loss.backward()
        assert torch.isfinite(loss)
        nograd = [n for n, p in m.named_parameters() if p.grad is None]
        assert not nograd, nograd[:5]
        print(f"ok  {size:5s} {str(kw):24s} params {count_parameters(m) / 1e6:.3f}M  loss {loss.item():.3f}")


if __name__ == "__main__":
    check_ssd()
    check_skip()
    check_routes()
    check_model()
    print("all checks passed")
