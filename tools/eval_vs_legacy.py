"""Score the FRESH pipeline against the LEGACY .pth baseline on a folder of
images. Produces, per source image, a side-by-side strip (legacy crops | fresh
crops) plus an aggregate report.

Metrics:
  * folio-count agreement (legacy vs fresh)
  * #folios produced by each
  * if a ground-truth CSV is supplied (cols: file,count[,orient]) -> per-system
    accuracy for count (and orientation if labelled)

Legacy needs torch + the .pth files. Fresh defaults to the classical fallback
(no weights); pass --fresh-neural --weights-dir to use SAM2.1 + trained heads.

Usage:
  python tools/eval_vs_legacy.py "/path/scans" \
      --legacy-weights /path/legacy_pth --outdir eval_out [--gt gt.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import numpy as np
import cv2
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from folio.config import PipelineConfig
from folio.pipeline import FolioPipeline
from folio.models.classical import ClassicalSegmenter, ClassicalCounter, ClassicalOrienter
from folio.models.legacy import LegacyPipeline


def strip(images, cell=300, pad=6, bg=40, label=None):
    if not images:
        images = [np.full((cell, cell, 3), 60, np.uint8)]
    h = cell + pad * 2 + (18 if label else 0)
    w = pad + len(images) * (cell + pad)
    canvas = np.full((h, w, 3), bg, np.uint8)
    for i, im in enumerate(images):
        ih, iw = im.shape[:2]; s = min(cell / iw, cell / ih)
        r = cv2.resize(im, (max(int(iw*s),1), max(int(ih*s),1)))
        y0 = pad + (18 if label else 0) + (cell - r.shape[0]) // 2
        x0 = pad + i * (cell + pad) + (cell - r.shape[1]) // 2
        canvas[y0:y0+r.shape[0], x0:x0+r.shape[1]] = r
    if label:
        cv2.putText(canvas, label, (pad, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    return canvas


def build_fresh(args):
    cfg = PipelineConfig()
    if args.fresh_neural:
        from folio.models.segmentation import PageSegmenter
        from folio.models.classifiers import FolioCountClassifier, OrientationClassifier
        if args.weights_dir:
            import os; wd = args.weights_dir
            cfg.model.sam_checkpoint = os.path.join(wd, "sam2.1_hiera_large.pt")
            cfg.model.detector_weights = os.path.join(wd, "rtdetr_page.pt")
            cfg.model.folio_count_weights = os.path.join(wd, "folio_count_convnextv2.pt")
            cfg.model.orientation_weights = os.path.join(wd, "orientation4_convnextv2.pt")
        return FolioPipeline(cfg, segmenter=PageSegmenter(cfg.model),
                             counter=FolioCountClassifier(cfg.model),
                             orienter=OrientationClassifier(cfg.model))
    return FolioPipeline(cfg, segmenter=ClassicalSegmenter(cfg.model),
                         counter=ClassicalCounter(), orienter=ClassicalOrienter())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--legacy-weights", required=True)
    ap.add_argument("--outdir", default="eval_out")
    ap.add_argument("--gt", default=None, help="CSV: file,count[,orient]")
    ap.add_argument("--fresh-neural", action="store_true")
    ap.add_argument("--weights-dir", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out = Path(args.outdir); (out / "compare").mkdir(parents=True, exist_ok=True)
    gt = {}
    if args.gt:
        for row in csv.DictReader(open(args.gt)):
            gt[row["file"]] = row

    fresh = build_fresh(args)
    legacy = LegacyPipeline(args.legacy_weights, device=args.device)

    rows, agree = [], 0
    leg_c_ok = fr_c_ok = n_gt = 0
    files = sorted(p for p in Path(args.input).iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff"))
    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            continue
        leg = legacy.process(img)
        fr = fresh.process_image(p.name, img)
        fr_count = fr.page_count.value
        fr_crops = [f.crop for f in fr.folios]
        same = (leg["count"] == fr_count)
        agree += int(same)

        row = {"file": p.name, "legacy_count": leg["count"], "fresh_count": fr_count,
               "count_agree": same, "legacy_folios": len(leg["crops"]),
               "fresh_folios": len(fr_crops)}
        if p.name in gt:
            n_gt += 1
            gtc = gt[p.name]["count"]
            row["gt_count"] = gtc
            leg_c_ok += int(leg["count"] == gtc); fr_c_ok += int(fr_count == gtc)
        rows.append(row)

        top = strip(leg["crops"], label=f"LEGACY  {leg['count']}")
        bot = strip(fr_crops, label=f"FRESH   {fr_count}")
        w = max(top.shape[1], bot.shape[1])
        def padw(a): 
            c = np.full((a.shape[0], w, 3), 40, np.uint8); c[:, :a.shape[1]] = a; return c
        cv2.imwrite(str(out / "compare" / f"{p.stem}.jpg"),
                    np.vstack([padw(top), padw(bot)]))
        print(f"{p.name:20s} legacy={leg['count']:10s} fresh={fr_count:10s} {'AGREE' if same else 'DIFFER'}")

    summary = {"n_images": len(rows), "count_agreement": agree / max(len(rows), 1)}
    if n_gt:
        summary["legacy_count_accuracy"] = leg_c_ok / n_gt
        summary["fresh_count_accuracy"] = fr_c_ok / n_gt
        summary["n_ground_truth"] = n_gt
    (out / "report.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print("\nSUMMARY:", json.dumps(summary, indent=2))
    print(f"side-by-side comparisons -> {out/'compare'}")


if __name__ == "__main__":
    main()
