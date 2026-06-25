"""QA summary + montage for a folio output dir.

    python tools/review_output.py <out_dir> [montage.jpg]

Prints page-count split, aspect distribution (flags any non-portrait / over-ratio
/ dark crops), review/blank counts, and confirms the sidecar provenance
(source_size + crop_quad_norm) is present. Writes a sampled montage.
"""
import os, csv, glob, json, sys
from collections import Counter
import numpy as np
import cv2

out = sys.argv[1]
mon = sys.argv[2] if len(sys.argv) > 2 else out.rstrip("\\/") + "_montage.jpg"

rows = list(csv.DictReader(open(os.path.join(out, "manifest.csv"), encoding="utf-8")))
print(f"source images: {len(set(r['source'] for r in rows))}   folios: {len(rows)}")
print("page_count:", dict(Counter(r["page_count"] for r in rows)))
print("needs_review:", sum(r["needs_review"] == "True" for r in rows),
      "  is_blank:", sum(r["is_blank"] == "True" for r in rows))

files = sorted(p for p in glob.glob(os.path.join(out, "folios", "*.jpg")) if "_enhanced" not in p)
ars, bad = [], []
for p in files:
    im = cv2.imread(p)
    if im is None:
        continue
    h, w = im.shape[:2]; ar = w / h
    ars.append(ar)
    long_short = max(h, w) / min(h, w)
    if ar >= 1.0 or long_short > 2.4 + 0.01 or cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).mean() < 45:
        bad.append((os.path.basename(p), round(ar, 2), round(long_short, 2)))
ars = np.array(ars)
print(f"aspect w/h: median={np.median(ars):.2f} min={ars.min():.2f} max={ars.max():.2f}  "
      f"portrait(<1)={int((ars<1).sum())}/{len(ars)}  >10:24 ratio={sum(1 for _ in bad)}")
if bad:
    print("FLAGGED:", bad[:10])

# provenance check on sidecars
side = glob.glob(os.path.join(out, "sidecars", "*.json"))
ok = 0
for s in side:
    d = json.load(open(s))
    for f in d.get("folios", []):
        if f.get("source_size") and f.get("crop_quad_norm"):
            ok += 1
print(f"sidecars with provenance (source_size + crop_quad_norm): {ok} folios across {len(side)} files")

step = max(1, len(files) // 60)
sample = files[::step][:60]
tiles = [cv2.resize(cv2.imread(p), (150, 210)) for p in sample]
g = [np.hstack(tiles[i:i+10]) for i in range(0, len(tiles), 10)]
W = max(r.shape[1] for r in g)
g = [cv2.copyMakeBorder(r, 2, 2, 0, W-r.shape[1], cv2.BORDER_CONSTANT, value=(150, 150, 150)) for r in g]
cv2.imwrite(mon, np.vstack(g))
print(f"{len(files)} crops -> sampled montage {mon}")
