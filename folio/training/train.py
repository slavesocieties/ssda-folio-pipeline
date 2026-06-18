"""Fine-tune one head and export a TorchScript checkpoint for inference.

Orientation (self-supervised, no labels needed):
  python -m folio.training.train --task orientation \
      --data /data/upright_pages --out weights/orientation4_convnextv2.pt

Folio-count (labelled folders one_folio/ two_folios/ reject/, or --manifest):
  python -m folio.training.train --task count \
      --data /data/count_dataset --out weights/folio_count_convnextv2.pt
"""
from __future__ import annotations
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from .datasets import OrientationDataset, FolioCountDataset
from .models import OrientationNet, FolioCountNet, export_torchscript


def _split(ds, val_frac, seed):
    n_val = max(1, int(len(ds) * val_frac))
    n_tr = len(ds) - n_val
    g = torch.Generator().manual_seed(seed)
    return random_split(ds, [n_tr, n_val], generator=g)


def run_epoch(model, loader, task, device, opt=None, scaler=None):
    train = opt is not None
    model.train(train)
    tot, correct, loss_sum = 0, 0, 0.0
    crit = torch.nn.CrossEntropyLoss()
    for batch in loader:
        if task == "orientation":
            x, y = batch; x, y = x.to(device), y.to(device); aux = None
        else:
            x, aux, y = batch; x, aux, y = x.to(device), aux.to(device), y.to(device)
        with torch.set_grad_enabled(train), torch.autocast(device_type=device.split(":")[0],
                                                           enabled=(device != "cpu")):
            logits = model(x) if task == "orientation" else model(x, aux)
            loss = crit(logits, y)
        if train:
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        loss_sum += float(loss) * x.size(0); tot += x.size(0)
        correct += int((logits.argmax(1) == y).sum())
    return loss_sum / tot, correct / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["orientation", "count"], required=True)
    ap.add_argument("--data", required=False)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.task == "orientation":
        full = OrientationDataset(args.data, size=args.size, train=True, seed=args.seed)
        model = OrientationNet(pretrained=True)
    else:
        full = FolioCountDataset(root=args.data, manifest=args.manifest,
                                 size=args.size, train=True, seed=args.seed)
        model = FolioCountNet(pretrained=True)
    tr, va = _split(full, args.val_frac, args.seed)
    va.dataset.train = False  # disable jitter for val view (shared underlying ds)
    dl_tr = DataLoader(tr, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                       pin_memory=True, drop_last=True)
    dl_va = DataLoader(va, batch_size=args.bs, shuffle=False, num_workers=args.workers)

    model.to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.device != "cpu"))

    best = 0.0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for ep in range(args.epochs):
        tl, ta = run_epoch(model, dl_tr, args.task, args.device, opt, scaler)
        vl, vacc = run_epoch(model, dl_va, args.task, args.device)
        sched.step()
        print(f"epoch {ep+1}/{args.epochs}  train_acc={ta:.3f}  val_acc={vacc:.3f}")
        if vacc >= best:
            best = vacc
            export_torchscript(model, args.task, args.size, args.out, args.device)
            print(f"  saved {args.out}  (val_acc={vacc:.3f})")
    print(f"done. best val_acc={best:.3f} -> {args.out}")


if __name__ == "__main__":
    main()
