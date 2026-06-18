"""Evaluate an exported TorchScript head: accuracy + confusion matrix.

  python -m folio.training.eval --task orientation \
      --data /data/upright_holdout --weights weights/orientation4_convnextv2.pt
"""
from __future__ import annotations
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from .datasets import OrientationDataset, FolioCountDataset
from .labels import ORIENTATION_DEGREES, COUNT_CLASSES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["orientation", "count"], required=True)
    ap.add_argument("--data", required=False)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model = torch.jit.load(args.weights, map_location=args.device).eval()
    if args.task == "orientation":
        ds = OrientationDataset(args.data, size=args.size, train=False)
        names = [str(d) for d in ORIENTATION_DEGREES]; n = 4
    else:
        ds = FolioCountDataset(root=args.data, manifest=args.manifest,
                               size=args.size, train=False)
        names = COUNT_CLASSES; n = 3
    dl = DataLoader(ds, batch_size=args.bs)
    cm = np.zeros((n, n), int); correct = tot = 0
    with torch.inference_mode():
        for batch in dl:
            if args.task == "orientation":
                x, y = batch; logits = model(x.to(args.device))
            else:
                x, aux, y = batch; logits = model(x.to(args.device), aux.to(args.device))
            pred = logits.argmax(1).cpu().numpy()
            for t, p in zip(y.numpy(), pred):
                cm[t, p] += 1; correct += int(t == p); tot += 1
    print(f"accuracy: {correct/tot:.4f}  (n={tot})")
    print("confusion (rows=true, cols=pred):", names)
    print(cm)


if __name__ == "__main__":
    main()
