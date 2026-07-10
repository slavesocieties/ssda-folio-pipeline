#!/usr/bin/env python3
"""benchmark_orientation.py — measure orientation-correction accuracy on a labelled
set of KNOWN-UPRIGHT folios.

Method (self-labelling): rotate each upright folio by 0/90/180/270, run it through the
full pipeline, and check the pipeline restores it to upright. Ground-truth oracle is
framing-matched: the pipeline's output for the un-rotated input (C0) is a tight crop, so
it is directly comparable to the tight crops from the rotated inputs (correlating against
the loosely-framed *original* gives false failures — the pipeline crops tighter than it).

    python tools/benchmark_orientation.py <upright_dir> [--limit N] [--device cuda|cpu]

Getting a labelled upright set: the SSDA validation set lives in Google Drive under
`upright_training/upright` (folios named `*-folio-upright.jpg`). Pull it with:

    pip install gdown
    python -m gdown --folder "https://drive.google.com/drive/folders/<UPRIGHT_FOLDER_ID>" -O ./upright

Then: `python tools/benchmark_orientation.py ./upright`.

Reports per-rotation accuracy (0/90/180/270) and overall. Note: automated orientation
oracles are themselves noisy on dense/faint cursive, so treat a few percent as a floor
and confirm borderline cases visually.
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from folio.process import make_config, build_pipeline, find_legacy_weights  # noqa: E402

S = 200


def _prep(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    g = cv2.resize(g, (S, S)).astype(np.float32)
    return (g - g.mean()) / (g.std() + 1e-6)


def _aligned_to(C, ref):
    r = _prep(ref)
    return int(np.argmax([float((_prep(np.rot90(C, k)) * r).mean()) for k in range(4)]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("upright_dir", help="folder of known-upright folio images (jpg/png)")
    ap.add_argument("--limit", type=int, default=None, help="use at most N folios")
    ap.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    args = ap.parse_args(argv)

    files = sorted(glob.glob(os.path.join(args.upright_dir, "*.jpg"))
                   + glob.glob(os.path.join(args.upright_dir, "*.png")))
    if args.limit:
        files = files[:args.limit]
    if not files:
        sys.exit(f"no images in {args.upright_dir}")

    cfg = make_config(device=args.device)
    pipe, mode = build_pipeline(cfg, find_legacy_weights(None, args.upright_dir))
    try:
        from folio.stages.ocr_orient import OCRUpDownVerifier
        ocr = OCRUpDownVerifier(device=cfg.model.device)
        ocr_ok = ocr.available
    except Exception:
        ocr, ocr_ok = None, False
    print(f"device: {cfg.model.device} | mode: {mode} | ocr: {ocr_ok} | folios: {len(files)}", flush=True)

    per_k = {k: [0, 0] for k in range(4)}
    for i, fp in enumerate(files, 1):
        U = cv2.imread(fp)
        if U is None:
            continue
        outs = {}
        for k in range(4):
            res = pipe.process_image(f"t{k}.jpg", np.ascontiguousarray(np.rot90(U, k)))
            outs[k] = (max(res.folios, key=lambda f: f.crop.shape[0]*f.crop.shape[1]).crop
                       if res.folios else None)
        C0 = outs[0]
        if C0 is not None:
            per_k[0][1] += 1
            up0 = True
            if ocr_ok:
                sf, mg = ocr.flip_verdict(C0)
                up0 = not (sf and mg >= 0.30)
            per_k[0][0] += up0
        for k in (1, 2, 3):
            per_k[k][1] += 1
            if outs[k] is not None and C0 is not None and _aligned_to(outs[k], C0) == 0:
                per_k[k][0] += 1
        if i % 10 == 0:
            tot = sum(v[1] for v in per_k.values()); cor = sum(v[0] for v in per_k.values())
            print(f"  {i}/{len(files)}  running {100*cor/max(1,tot):.1f}%", flush=True)

    print("\n=== orientation-correction accuracy ===")
    names = {0: "0deg passthrough (OCR-checked)", 1: "90deg", 2: "180deg (hard)", 3: "270deg"}
    tc = tn = 0
    for k in range(4):
        c, n = per_k[k]; tc += c; tn += n
        print(f"  applied {names[k]:<30} {c}/{n} = {100*c/max(1,n):.1f}%")
    print(f"  OVERALL: {tc}/{tn} = {100*tc/max(1,tn):.1f}%")


if __name__ == "__main__":
    main()
