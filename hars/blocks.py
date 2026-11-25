"""Building blocks: convolutional blocks and the Artifact-Gated Selective Scan (AGSS)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .scans import FIXED_ROUTES, fixed_route, inverse_perm
from .ssd import ssd_scan


# ----------------------------------------------------------------------------- utils
class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty((x.shape[0],) + (1,) * (x.dim() - 1)).bernoulli_(keep)
        return x * mask / keep


class ConvBNAct(nn.Sequential):
    def __init__(self, cin, cout, k=3, s=1, groups=1, act=True):
        super().__init__(
            nn.Conv2d(cin, cout, k, s, k // 2, groups=groups, bias=False),
            nn.BatchNorm2d(cout),
            nn.GELU() if act else nn.Identity(),
        )


class ResConvBlock(nn.Module):
    """3x3 conv -> 3x3 conv with an identity (or 1x1) shortcut.  Used at high resolution."""

    def __init__(self, cin, cout):
        super().__init__()
        self.body = nn.Sequential(ConvBNAct(cin, cout), ConvBNAct(cout, cout, act=False))
        self.short = nn.Identity() if cin == cout else ConvBNAct(cin, cout, k=1, act=False)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.body(x) + self.short(x))


class RMSNormGated(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x, z):
        dtype = x.dtype
        x = x.float() * F.silu(z.float())                      # float32: safe under fp16 autocast
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def _take(t: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """Reorder dim 1 of t (b, L, ...) with perm of shape (L,) or (b, L)."""
    if perm.dim() == 1:
        return t[:, perm]
    idx = perm.view(perm.shape[0], perm.shape[1], *([1] * (t.dim() - 2))).expand_as(t)
    return torch.gather(t, 1, idx)


# ----------------------------------------------------------------------------- AGSS mixer
class AGSSM(nn.Module):
    """Artifact-Gated multi-route Selective State-Space Mixer.

    * Heads are split into route groups; each group scans the 2D map along its own route
      (the four fixed cross-scan routes, or custom permutations passed at call time).  All
      groups are evaluated in ONE batched SSD call, so four routes cost the same as one.
    * Artifact gating: Delta_t <- Delta_t * (1 - beta_h * a_t), a_t = artifact probability.
      With a small Delta the update h_t = exp(Delta A) h_{t-1} + Delta B x_t keeps the state
      (exp(.) -> 1) and ignores the occluded input (Delta B x -> 0): the scan steps over hair.
    """

    def __init__(self, dim, routes, expand=2, headdim=32, d_state=16, chunk=64,
                 use_artifact=True, dt_min=1e-3, dt_max=1e-1):
        super().__init__()
        self.routes = tuple(routes)
        self.d_inner = expand * dim
        self.headdim = headdim
        self.nheads = self.d_inner // headdim
        self.ngroups = len(self.routes)
        assert self.d_inner % headdim == 0 and self.nheads % self.ngroups == 0, \
            f"heads ({self.nheads}) must be divisible by routes ({self.ngroups})"
        self.d_state = d_state
        self.chunk = chunk
        self.use_artifact = use_artifact

        conv_dim = self.d_inner + 2 * self.ngroups * d_state
        self.in_proj = nn.Linear(dim, self.d_inner + conv_dim + self.nheads, bias=False)
        self.conv = nn.Conv2d(conv_dim, conv_dim, 3, padding=1, groups=conv_dim)

        dt = torch.exp(torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))          # inverse softplus
        self.A_log = nn.Parameter(torch.log(torch.empty(self.nheads).uniform_(1, 16)))
        self.D = nn.Parameter(torch.ones(self.nheads))
        if use_artifact:
            self.beta_logit = nn.Parameter(torch.zeros(self.nheads))            # beta = 0.5 at init
        self.norm = RMSNormGated(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

    def _perms(self, h, w, device, custom):
        perms = []
        for r in self.routes:
            if r in FIXED_ROUTES:
                perms.append(fixed_route(r, h, w, device))
            else:
                if custom is None or r not in custom:
                    raise RuntimeError(f"route '{r}' is not a fixed route; pass its permutation in `custom`")
                perms.append(custom[r])
        return perms

    def forward(self, x, artifact=None, custom=None):
        """x: (b, h, w, c) channels-last.  artifact: (b, 1, H, W) probabilities or None.
        custom: optional dict route name -> permutation (L,) or (b, L) for non-fixed routes."""
        b, h, w, _ = x.shape
        L = h * w
        H, P, G, N = self.nheads, self.headdim, self.ngroups, self.d_state

        z, xbc, dt = torch.split(self.in_proj(x), [self.d_inner, self.d_inner + 2 * G * N, H], dim=-1)
        xbc = F.silu(self.conv(xbc.permute(0, 3, 1, 2))).permute(0, 2, 3, 1)
        xs, Bm, Cm = torch.split(xbc, [self.d_inner, G * N, G * N], dim=-1)

        xs = xs.reshape(b, L, H, P)
        Bm = Bm.reshape(b, L, G, N)
        Cm = Cm.reshape(b, L, G, N)
        dt = F.softplus(dt.reshape(b, L, H).float() + self.dt_bias.float())
        if self.use_artifact and artifact is not None:
            a = F.adaptive_avg_pool2d(artifact.float(), (h, w)).reshape(b, L, 1)
            dt = dt * (1.0 - torch.sigmoid(self.beta_logit).view(1, 1, H) * a)
        dt = dt.clamp_min(1e-4)
        A = -torch.exp(self.A_log.float())

        perms = self._perms(h, w, x.device, custom)
        hr = H // G
        X, DT, Bh, Ch = [], [], [], []
        for g, perm in enumerate(perms):
            sl = slice(g * hr, (g + 1) * hr)
            X.append(_take(xs[:, :, sl], perm))
            DT.append(_take(dt[:, :, sl], perm))
            Bh.append(_take(Bm[:, :, g], perm).unsqueeze(2).expand(b, L, hr, N))
            Ch.append(_take(Cm[:, :, g], perm).unsqueeze(2).expand(b, L, hr, N))
        X, DT, Bh, Ch = (torch.cat(t, dim=2) for t in (X, DT, Bh, Ch))

        y = ssd_scan(X, DT, A, Bh, Ch, chunk=self.chunk)
        y = y + X * self.D.view(1, 1, H, 1).to(y.dtype)

        out = []
        for g, perm in enumerate(perms):
            out.append(_take(y[:, :, g * hr:(g + 1) * hr], inverse_perm(perm)))
        y = torch.cat(out, dim=2).reshape(b, h, w, self.d_inner)
        y = self.norm(y, z)
        return self.out_proj(y)


class ConvFFN(nn.Module):
    def __init__(self, dim, ratio=2.0):
        super().__init__()
        hid = int(dim * ratio)
        self.fc1 = nn.Linear(dim, hid)
        self.dw = nn.Conv2d(hid, hid, 3, padding=1, groups=hid)
        self.fc2 = nn.Linear(hid, dim)

    def forward(self, x):  # (b, h, w, c)
        x = self.fc1(x)
        x = self.dw(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return self.fc2(F.gelu(x))


class HybridBlock(nn.Module):
    """Pre-norm residual block: AGSS mixer + depth-wise convolutional FFN."""

    def __init__(self, dim, routes, mlp_ratio=2.0, drop_path=0.0, **ssm_kw):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mixer = AGSSM(dim, routes, **ssm_kw)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = ConvFFN(dim, mlp_ratio)
        self.dp = DropPath(drop_path)

    def forward(self, x, artifact=None, custom=None):  # (b, c, h, w)
        t = x.permute(0, 2, 3, 1)
        t = t + self.dp(self.mixer(self.norm1(t), artifact, custom))
        t = t + self.dp(self.ffn(self.norm2(t)))
        return t.permute(0, 3, 1, 2).contiguous()
