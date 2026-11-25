"""Train one fold of HaRS-Mamba.

    python train_fold.py --fold 1                                   # HaRS-Mamba
    python train_fold.py --fold 1 --name no_synthetic_hair --synthetic-hair 0
    python train_fold.py --fold 1 --name no_agss --no-artifact      # no hair prior, gate or synthetic hair

* early stopping on the fold's validation Dice (image-wise): patience 40, never before epoch 60;
* NaN/Inf guard: non-finite batches are skipped; non-finite weights roll back to the last healthy
  epoch and the fold continues in float32 (AMP off);
* resumable (last.pt), never overwrites a finished fold (done.json).
Outputs: $HARS_RUNS/<name>/fold_<k>/best.pt, last.pt, log.csv, events.log, done.json
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import time

import numpy as np
import torch

from config import N_FOLDS, load_folds, make_trainval_dataset, run_dir
from hars import build_model, count_parameters
from hars.losses import HaRSLoss
from hars.metrics import MetricTracker


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--name", default="hars_mamba", help="run name; outputs go to $HARS_RUNS/<name>/fold_<k>")
    ap.add_argument("--model", default="base")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--min-epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--synthetic-hair", type=float, default=0.5)
    ap.add_argument("--no-artifact", action="store_true", help="ablation: no hair prior, no step-size gate, no synthetic hair")
    ap.add_argument("--seed", type=int, default=None, help="default 42 + fold")
    ap.add_argument("--max-steps", type=int, default=0, help="smoke test: stop each epoch after N steps")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: use N train / N val images")
    ap.add_argument("--out", default=None, help="override the output folder")
    return ap.parse_args()


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def all_finite(model):
    for t in list(model.parameters()) + list(model.buffers()):
        if t.is_floating_point() and not torch.isfinite(t).all():
            return False
    return True


@torch.no_grad()
def validate(model, loader, device, amp):
    model.eval()
    tr = MetricTracker()
    for b in loader:
        x, y = b["image"].to(device, non_blocking=True), b["mask"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            out = model(x)
        tr.update(torch.sigmoid(out["seg"].float()) > 0.5, y > 0.5)
    return tr.summary()


def main():
    a = get_args()
    assert 1 <= a.fold <= N_FOLDS
    seed = a.seed if a.seed is not None else 42 + a.fold
    seed_all(seed)
    out = run_dir(a.name) / f"fold_{a.fold}" if a.out is None else __import__("pathlib").Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "done.json").exists():
        print(f"fold {a.fold} already finished -> {out / 'done.json'}")
        return
    ev = open(out / "events.log", "a", encoding="utf-8")

    def event(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
        print(line, flush=True)
        ev.write(line + "\n")
        ev.flush()

    F = load_folds()
    src, fd = F["source"], F["folds"][str(a.fold)]
    tr_ids, va_ids = fd["train"], fd["val"]
    if a.limit:
        tr_ids, va_ids = tr_ids[: a.limit], va_ids[: a.limit]
    syn = 0.0 if a.no_artifact else a.synthetic_hair
    train_ds = make_trainval_dataset(tr_ids, src, train=True, synthetic_hair=syn)
    val_ds = make_trainval_dataset(va_ids, src, train=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    pin = device.type == "cuda"
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=a.batch_size, shuffle=True,
                                           num_workers=a.workers, pin_memory=pin, drop_last=True,
                                           persistent_workers=a.workers > 0)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=a.batch_size, shuffle=False,
                                         num_workers=a.workers, pin_memory=pin)

    model = build_model(a.model, use_artifact=not a.no_artifact).to(device)
    n_params = count_parameters(model)
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if (p.dim() < 2 or n.endswith(("A_log", "dt_bias", "D", "beta_logit"))) else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=a.lr)
    spe = max(1, len(train_dl) if not a.max_steps else min(len(train_dl), a.max_steps))
    total, warm = a.epochs * spe, a.warmup * spe

    def lr_at(step):
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm))) * 0.99 + 0.01

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    crit = HaRSLoss(w_hair=0.0 if a.no_artifact else 0.1)

    start, best, best_ep, step, rollbacks = 0, -1.0, -1, 0, 0
    if (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        amp = ck.get("amp", amp)
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        if amp:
            scaler.load_state_dict(ck["scaler"])
        start, best, best_ep, step, rollbacks = (ck["epoch"] + 1, ck["best"], ck["best_epoch"],
                                                 ck["step"], ck.get("rollbacks", 0))
        event(f"resumed fold {a.fold} at epoch {start + 1} (best {best:.4f} @ {best_ep + 1}, amp={amp})")

    event(f"fold {a.fold}: train {len(train_ds)}  val {len(val_ds)}  params {n_params / 1e6:.3f}M  "
          f"seed {seed}  amp {amp}  device {device}  no_artifact {a.no_artifact}  synthetic_hair {syn}")
    new_log = not (out / "log.csv").exists()
    logf = open(out / "log.csv", "a", newline="")
    wr = csv.writer(logf)
    if new_log:
        wr.writerow(["epoch", "lr", "loss", "val_dice_img", "val_iou_img", "val_dice_global",
                     "val_iou_global", "skipped_batches", "amp", "sec"])

    def snapshot():
        return {"model": copy.deepcopy(model.state_dict()), "opt": copy.deepcopy(opt.state_dict()),
                "sched": copy.deepcopy(sched.state_dict())}

    good = snapshot()
    stop_reason = "max_epochs"
    epoch = start
    while epoch < a.epochs:
        model.train()
        t0, run_loss, nb, skipped, broken = time.time(), 0.0, 0, 0, False
        for i, b in enumerate(train_dl):
            if a.max_steps and i >= a.max_steps:
                break
            x, y, hm = (b[k].to(device, non_blocking=True) for k in ("image", "mask", "hair"))
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                o = model(x)
            loss, _ = crit(o, y, hm)
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                skipped += 1
                sched.step(); step += 1
                if skipped > 20:
                    broken = True
                    break
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step(); step += 1
            run_loss += loss.item(); nb += 1
            if step % 50 == 0 and not all_finite(model):
                broken = True
                break
        if broken or not all_finite(model):
            rollbacks += 1
            event(f"epoch {epoch + 1}: non-finite loss/weights (skipped {skipped}); rollback #{rollbacks} "
                  f"to last healthy epoch, continuing with AMP OFF")
            if rollbacks > 3:
                stop_reason = "too_many_rollbacks"
                break
            model.load_state_dict(good["model"]); opt.load_state_dict(good["opt"]); sched.load_state_dict(good["sched"])
            amp = False
            scaler = torch.amp.GradScaler("cuda", enabled=False)
            continue                                    # retry the same epoch in float32
        if skipped:
            event(f"epoch {epoch + 1}: skipped {skipped} non-finite batch(es)")

        res = validate(model, val_dl, device, amp)
        if res["dice_img"] > best:
            best, best_ep = res["dice_img"], epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "val": res, "params": n_params,
                        "fold": a.fold, "seed": seed, "args": {**vars(a), "size": 256}}, out / "best.pt")
        dt = time.time() - t0
        wr.writerow([epoch + 1, f"{sched.get_last_lr()[0]:.2e}", f"{run_loss / max(1, nb):.4f}",
                     f"{res['dice_img']:.4f}", f"{res['iou_img']:.4f}", f"{res['dice_global']:.4f}",
                     f"{res['iou_global']:.4f}", skipped, int(amp), f"{dt:.0f}"])
        logf.flush()
        print(f"fold {a.fold}  epoch {epoch + 1}/{a.epochs}  loss {run_loss / max(1, nb):.4f}  "
              f"val Dice {res['dice_img']:.4f}  IoU {res['iou_img']:.4f}  "
              f"(best {best:.4f} @ {best_ep + 1})  {dt:.0f}s", flush=True)
        good = snapshot()
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "best": best, "best_epoch": best_ep,
                    "step": step, "amp": amp, "rollbacks": rollbacks}, out / "last.pt")
        epoch += 1
        if epoch >= a.min_epochs and epoch - 1 - best_ep >= a.patience:
            stop_reason = f"early_stopping (no improvement for {a.patience} epochs)"
            break

    logf.close()
    done = {"fold": a.fold, "best_val_dice_img": best, "best_epoch": best_ep + 1, "epochs_run": epoch,
            "stop_reason": stop_reason, "rollbacks": rollbacks, "seed": seed, "params": n_params,
            "n_train": len(train_ds), "n_val": len(val_ds), "no_artifact": a.no_artifact, "synthetic_hair": syn, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    if stop_reason != "too_many_rollbacks":
        (out / "done.json").write_text(json.dumps(done, indent=2))
    event(f"fold {a.fold} finished: {done}")


if __name__ == "__main__":
    main()
