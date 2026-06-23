"""Evaluate crop tightness against the supervisor's ground-truth text rectangles.

Metrics per image:
  coverage  = GT-content area kept inside the crop / total GT area   (want ~1.00 = no clipped text)
  meaningful= GT-content area inside crop / crop area                (target ~0.75)

Run candidate crop-box methods and print aggregate. The GT rects come from the
representative-set CSV (region annotations); the union of an image's rects is the
"meaningful information" region. NB: a few top-level images are spreads with only
one page annotated, so their meaningful score is pessimistic (production splits
first); judge mainly on coverage (no clipping) and the single-folio cases.
"""
import os, csv, sys
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from folio.stages import content

BASE = r"C:\Users\mahajar\Downloads\sample images\_repr_images\Representative Images"
CSVF = os.path.join(BASE, "ea3kvhrhn28abu (1).csv")


def load_gt():
    rects = {}
    for row in csv.DictReader(open(CSVF, encoding="utf-8")):
        d = dict(kv.split("=") for kv in row["ANCHOR"].replace("rect:", "").split(","))
        rects.setdefault(row["FILE"], []).append(
            (int(d["x"]), int(d["y"]), int(d["w"]), int(d["h"])))
    return rects


# ---- candidate crop-box methods: image -> (x0,y0,x1,y1) ----
def m_full(im):
    h, w = im.shape[:2]; return (0, 0, w, h)

def m_paperbox(im):
    b = content.paper_box(im)
    h, w = im.shape[:2]
    return b if b else (0, 0, w, h)

def evaluate(method, rects, detail=False):
    cov, mean, rows = [], [], []
    for fn, rs in rects.items():
        im = cv2.imread(os.path.join(BASE, fn))
        if im is None:
            continue
        h, w = im.shape[:2]
        x0, y0, x1, y1 = method(im)
        gt = np.zeros((h, w), np.uint8)
        for (x, y, rw, rh) in rs:
            gt[y:y+rh, x:x+rw] = 1
        gt_area = int(gt.sum())
        x0, y0, x1, y1 = method(im)
        inside = int(gt[y0:y1, x0:x1].sum())
        crop_area = max(1, (x1 - x0) * (y1 - y0))
        c, m = inside / max(1, gt_area), inside / crop_area
        cov.append(c); mean.append(m)
        rows.append((fn, c, m, w / h))
    if detail:
        for fn, c, m, ar in sorted(rows, key=lambda r: r[1]):
            tag = "SPREAD?" if ar > 0.9 else ""
            print(f"    {fn:18s} cov={c:.2f} mean={m:.2f} aspect={ar:.2f} {tag}")
    return np.array(cov), np.array(mean)


if __name__ == "__main__":
    rects = load_gt()
    print(f"{len(rects)} annotated images\n")
    for name, fn in [("full-frame (loose)", m_full),
                     ("paper_box (current)", m_paperbox)]:
        cov, mean = evaluate(fn, rects, detail=False)
        print(f"{name:24s}  coverage={cov.mean():.3f} (min {cov.min():.2f})   "
              f"meaningful={mean.mean():.3f}   "
              f"clipping<0.98: {int((cov<0.98).sum())}/{len(cov)}")
