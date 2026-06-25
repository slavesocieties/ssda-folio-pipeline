"""Entry (record) segmentation — apply the trained entry segmenter to folio crops.

    python tools/segment_entries.py <crops_dir> <out_dir> [--weights ...] [--min-area 0.004]

For each image it predicts the entry-region mask, splits it into individual entry
boxes (connected components, sorted top-to-bottom then left-to-right), writes a
red-overlay preview and a <stem>.json of boxes. This is the record-level layout
for downstream transcription — separate from the folio crop/orient/clean pipeline.
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np
import cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from folio.models.folio_seg import FolioSegmenter


def entry_boxes(prob, min_area_frac, thresh=0.5):
    h, w = prob.shape
    m = (prob > thresh).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, lbl, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    boxes = []
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < min_area_frac * h * w:
            continue
        x, y = int(st[i, cv2.CC_STAT_LEFT]), int(st[i, cv2.CC_STAT_TOP])
        bw, bh = int(st[i, cv2.CC_STAT_WIDTH]), int(st[i, cv2.CC_STAT_HEIGHT])
        boxes.append((x, y, bw, bh))
    # reading order: top-to-bottom, then left-to-right within a row band
    boxes.sort(key=lambda b: (b[1] // max(1, h // 20), b[0]))
    return boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("crops"); ap.add_argument("out")
    ap.add_argument("--weights",
                    default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                         "weights", "entry_seg_unet.pt.ts.pt"))
    ap.add_argument("--min-area", type=float, default=0.004)
    args = ap.parse_args()

    seg = FolioSegmenter(args.weights)
    os.makedirs(args.out, exist_ok=True)
    files = [p for p in sorted(glob.glob(os.path.join(args.crops, "*.jpg"))) if "_enhanced" not in p]
    total = 0
    for p in files:
        im = cv2.imread(p)
        if im is None:
            continue
        boxes = entry_boxes(seg.page_prob(im), args.min_area)
        total += len(boxes)
        stem = os.path.splitext(os.path.basename(p))[0]
        vis = im.copy()
        for j, (x, y, w, h) in enumerate(boxes, 1):
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), max(2, im.shape[1] // 400))
            cv2.putText(vis, str(j), (x + 5, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
        cv2.imwrite(os.path.join(args.out, stem + "_entries.jpg"), vis)
        json.dump({"image": os.path.basename(p), "n_entries": len(boxes),
                   "boxes_xywh": boxes}, open(os.path.join(args.out, stem + ".json"), "w"), indent=2)
    print(f"{len(files)} crops -> {total} entries total ({total/max(1,len(files)):.1f}/crop) -> {args.out}")


if __name__ == "__main__":
    main()
