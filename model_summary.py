"""Parameters, FLOPs, memory and speed of the HaRS-Mamba variants.

    python model_summary.py --size 256
"""
from __future__ import annotations

import argparse
import time

import torch
from torch.utils.flop_counter import FlopCounterMode

from hars import CONFIGS, build_model, count_parameters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--models", nargs="*", default=list(CONFIGS))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    for name in args.models:
        model = build_model(name).to(device).eval()
        x = torch.randn(1, 3, args.size, args.size, device=device)
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            model(x)
        gflops = fc.get_total_flops() / 2e9          # multiply-accumulates, as usually reported
        with torch.no_grad():
            for _ in range(3):
                model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            n = 10
            for _ in range(n):
                model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
        ms = (time.time() - t0) / n * 1000
        print(f"{name:6s} params {count_parameters(model) / 1e6:6.3f} M   "
              f"GFLOPs(MAC) {gflops:6.2f}   {ms:7.1f} ms/img on {device}")


if __name__ == "__main__":
    main()
