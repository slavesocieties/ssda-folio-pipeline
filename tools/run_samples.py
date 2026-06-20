"""Smoke-test the pipeline on a folder of real scans using the classical
fallback (no GPU / no weights). Produces, per image:
  - crops/<stem>[-A|-B].jpg   the upright, split, cropped page(s)
  - overlays/<stem>.jpg       original with page boxes + gutter seam drawn
and overall:
  - montage_overlays.jpg      grid of all overlays (eyeball the splits)
  - montage_crops.jpg         grid of all output crops
  - report.json               per-image decisions + provenance

Usage:
  python tools/run_samples.py "/path/to/sample images" --outdir ./sample_out
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from folio.config import PipelineConfig
from folio.pipeline import FolioPipeline
from folio.models.classical import (ClassicalSegmenter, ClassicalCounter,
                                     ClassicalOrienter)


def draw_overlay(image, boxes, seam):
    ov = image.copy()
    for i, b in enumerate(boxes):
        color = (0, 0, 255) if i == 0 else (255, 0, 0)
        cv2.rectangle(ov, (b.x1, b.y1), (b.x2, b.y2), color, 4)
    if seam is not None:
        pts = np.array([[x, y] for y, x in enumerate(seam)], dtype=np.int32)
        cv2.polylines(ov, [pts], False, (0, 255, 0), 4)
    return ov


def montage(images, cols, cell=420, pad=8, bg=30):
    if not images:
        return None
    rows = math.ceil(len(images) / cols)
    canvas = np.full((rows * (cell + pad) + pad, cols * (cell + pad) + pad, 3),
                     bg, np.uint8)
    for idx, im in enumerate(images):
        r, c = divmod(idx, cols)
        h, w = im.shape[:2]
        s = min(cell / w, cell / h)
        rim = cv2.resize(im, (max(int(w * s), 1), max(int(h * s), 1)))
        y0 = pad + r * (cell + pad) + (cell - rim.shape[0]) // 2
        x0 = pad + c * (cell + pad) + (cell - rim.shape[1]) // 2
        canvas[y0:y0+rim.shape[0], x0:x0+rim.shape[1]] = rim
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--outdir", default="./sample_out")
    ap.add_argument("--valley", type=float, default=0.6,
                    help="two-folio gutter-valley threshold")
    ap.add_argument("--osd", action="store_true",
                    help="use Tesseract OSD orientation (slower; better on print)")
    ap.add_argument("--neural", action="store_true",
                    help="use SAM2.1+RT-DETR + trained heads (needs GPU + weights)")
    ap.add_argument("--weights-dir", default=None,
                    help="dir holding the .pt weights (overrides config paths)")
    ap.add_argument("--hybrid", action="store_true",
                    help="recommended: reuse legacy .pth models in our split/crop logic")
    ap.add_argument("--legacy-weights", default=None,
                    help="dir with legacy .pth files (for --hybrid)")
    ap.add_argument("--sam-fallback", action="store_true",
                    help="with --hybrid, escalate weak masks to SAM 2.1")
    args = ap.parse_args()

    indir = Path(args.input)
    out = Path(args.outdir)
    (out / "crops").mkdir(parents=True, exist_ok=True)
    (out / "overlays").mkdir(parents=True, exist_ok=True)

    cfg = PipelineConfig()
    if args.weights_dir:
        import os
        wd = args.weights_dir
        cfg.model.sam_checkpoint = os.path.join(wd, "sam2.1_hiera_large.pt")
        cfg.model.detector_weights = os.path.join(wd, "rtdetr_page.pt")
        cfg.model.folio_count_weights = os.path.join(wd, "folio_count_convnextv2.pt")
        cfg.model.orientation_weights = os.path.join(wd, "orientation4_convnextv2.pt")

    if args.hybrid:
        from folio.models.hybrid import build_hybrid_pipeline
        pipe = build_hybrid_pipeline(cfg, args.legacy_weights or args.weights_dir,
                                     device=cfg.model.device,
                                     use_sam_fallback=args.sam_fallback)
        seg = pipe.segmenter
    elif args.neural:
        # production path: foundation segmentation + trained heads
        from folio.models.segmentation import PageSegmenter
        from folio.models.classifiers import FolioCountClassifier, OrientationClassifier
        seg = PageSegmenter(cfg.model)
        pipe = FolioPipeline(cfg, segmenter=seg,
                             counter=FolioCountClassifier(cfg.model),
                             orienter=OrientationClassifier(cfg.model))
    else:
        seg = ClassicalSegmenter(cfg.model, two_folio_valley=args.valley)
        orienter = (__import__("folio.models.osd_orient", fromlist=["OSDOrienter"]).OSDOrienter()
                    if args.osd else ClassicalOrienter())
        pipe = FolioPipeline(cfg, segmenter=seg,
                             counter=ClassicalCounter(two_folio_valley=args.valley),
                             orienter=orienter)

    files = sorted([p for p in indir.iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff")])
    overlays, crops, report = [], [], []
    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            report.append({"file": p.name, "error": "unreadable"})
            continue
        res = pipe.process_image(p.name, img)
        # rebuild boxes/seam for the overlay (same analysis, cached)
        boxes = seg.detect(img)
        seam = np.array(res.gutter_seam) if res.gutter_seam else None
        ov = draw_overlay(img, boxes, seam)
        cv2.imwrite(str(out / "overlays" / f"{p.stem}.jpg"), ov)
        overlays.append(ov)

        for f in res.folios:
            suffix = f"-{f.label}" if f.label else ""
            cp = out / "crops" / f"{p.stem}{suffix}.jpg"
            cv2.imwrite(str(cp), f.crop)
            crops.append(f.crop)

        report.append({
            "file": p.name,
            "count": res.page_count.value,
            "count_conf": round(res.count_conf, 3),
            "n_folios": len(res.folios),
            "gutter_median_x": (int(np.median(seam)) if seam is not None else None),
            "img_w": img.shape[1],
            "folios": [{"label": f.label,
                        "rotation_deg": round(f.rotation_deg, 2),
                        "orient_conf": round(f.orientation_conf, 3),
                        "is_blank": f.is_blank,
                        "blank_conf": round(f.blank_conf, 3),
                        "review": f.needs_review,
                        "reasons": f.review_reasons,
                        "out_w": f.crop.shape[1], "out_h": f.crop.shape[0]}
                       for f in res.folios],
            "error": res.error,
        })
        print(f"{p.name:22s} {res.page_count.value:11s} "
              f"folios={len(res.folios)} "
              f"gutter_x={'%4d'%int(np.median(seam)) if seam is not None else '  - '} "
              f"(W={img.shape[1]})")

    mo = montage(overlays, cols=5)
    mc = montage(crops, cols=6)
    if mo is not None:
        cv2.imwrite(str(out / "montage_overlays.jpg"), mo)
    if mc is not None:
        cv2.imwrite(str(out / "montage_crops.jpg"), mc)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nWrote {len(crops)} crops, {len(overlays)} overlays to {out}")


if __name__ == "__main__":
    main()
