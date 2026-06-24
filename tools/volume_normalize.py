"""Per-volume consistency pass (Daniel's idea): lock a single page size per volume
from its high-confidence crops, standardize every crop in the volume to that size,
and flag size outliers (likely mis-crops) for QA.

    python tools/volume_normalize.py <folio_out_dir> [--pad 255]

Reads <dir>/folios/*.jpg + <dir>/manifest.csv, groups by volume (the part of the
name before the first '-'), and writes <dir>/folios_normalized/*.jpg where every
crop in a volume has IDENTICAL dimensions. Padding only — content is never cropped
(the full folio is preserved, centred on a uniform canvas). Writes
volume_report.csv with each crop's deviation from its volume's norm.

Why this aids precision, not just consistency: the target size comes from the
volume's *confident* crops (text-rich, not review-flagged), so sparse / torn /
hard pages inherit the volume's reliable page size instead of a per-image guess,
and any crop whose own size is far from the volume norm is surfaced as suspect.
"""
from __future__ import annotations
import argparse, csv, os, glob
from collections import defaultdict
import numpy as np
import cv2


def volume_of(name: str) -> str:
    return name.split("-")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--pad", type=int, default=-1,
                    help="pad gray value 0-255; default -1 = blend with each crop's own border")
    ap.add_argument("--outlier", type=float, default=0.12,
                    help="flag crops whose aspect deviates from the volume median by more than this")
    args = ap.parse_args()

    folios = os.path.join(args.out_dir, "folios")
    man = os.path.join(args.out_dir, "manifest.csv")
    review = {}
    if os.path.exists(man):
        for r in csv.DictReader(open(man, encoding="utf-8")):
            review[r["folio"]] = (r.get("needs_review") == "True")

    files = sorted(p for p in glob.glob(os.path.join(folios, "*.jpg")) if "_enhanced" not in p)
    by_vol = defaultdict(list)
    dims = {}
    for p in files:
        im = cv2.imread(p)
        if im is None:
            continue
        h, w = im.shape[:2]
        dims[p] = (w, h)
        by_vol[volume_of(os.path.basename(p))].append(p)

    out_dir = os.path.join(args.out_dir, "folios_normalized")
    os.makedirs(out_dir, exist_ok=True)
    rep = open(os.path.join(args.out_dir, "volume_report.csv"), "w", newline="")
    wtr = csv.writer(rep); wtr.writerow(["folio", "volume", "w", "h", "aspect",
                                         "vol_aspect", "aspect_dev", "canvas_w", "canvas_h", "suspect"])
    n_norm = n_suspect = 0
    for vol, ps in by_vol.items():
        ws = np.array([dims[p][0] for p in ps]); hs = np.array([dims[p][1] for p in ps])
        ar = ws / hs
        # target aspect from CONFIDENT crops (fall back to all)
        conf = [i for i, p in enumerate(ps) if not review.get(os.path.basename(p), False)]
        idx = conf if conf else list(range(len(ps)))
        vol_ar = float(np.median(ar[idx]))
        # canvas big enough to hold every crop at the target aspect (never crop)
        cw = int(max(ws.max(), round(hs.max() * vol_ar)))
        ch = int(round(cw / vol_ar))
        ch = max(ch, int(hs.max())); cw = max(cw, int(round(ch * vol_ar)))
        for p in ps:
            w, h = dims[p]
            dev = abs(ar[ps.index(p)] - vol_ar)
            suspect = dev > args.outlier
            n_suspect += int(suspect)
            im = cv2.imread(p)
            if args.pad >= 0:
                fill = (args.pad, args.pad, args.pad)
            else:  # blend: median colour of the crop's 1-px border
                b = np.concatenate([im[0], im[-1], im[:, 0], im[:, -1]])
                fill = tuple(int(v) for v in np.median(b, axis=0))
            canvas = np.full((ch, cw, 3), fill, np.uint8)
            x = (cw - w) // 2; y = (ch - h) // 2
            canvas[y:y + h, x:x + w] = im
            cv2.imwrite(os.path.join(out_dir, os.path.basename(p)), canvas)
            n_norm += 1
            wtr.writerow([os.path.basename(p), vol, w, h, f"{w/h:.3f}", f"{vol_ar:.3f}",
                          f"{dev:.3f}", cw, ch, suspect])
    rep.close()
    print(f"normalized {n_norm} crops across {len(by_vol)} volumes -> {out_dir}")
    print(f"size/aspect outliers flagged (suspect mis-crops): {n_suspect}")
    print(f"report -> {os.path.join(args.out_dir, 'volume_report.csv')}")


if __name__ == "__main__":
    main()
