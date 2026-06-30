"""Montage the crops that need a human look: needs_review, extreme aspect, tiny,
or very dark. Helps characterize the remaining failure modes after a run."""
import os, csv, glob, sys, cv2, numpy as np
out, mon = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(os.path.join(out, "manifest.csv"), encoding="utf-8")))
review = {r["source"] for r in rows if r["needs_review"] == "True"}

flagged = []
for p in sorted(glob.glob(os.path.join(out, "folios", "*.jpg"))):
    name = os.path.basename(p)
    if "_enhanced" in name: continue
    im = cv2.imread(p)
    if im is None: continue
    h, w = im.shape[:2]; ar = w / h
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    src = name.split("-")[0] + "-" + name.split("-")[1].split(".")[0].split("_")[0]
    reasons = []
    if ar < 0.5: reasons.append("sliver")
    if ar >= 1.0: reasons.append("landscape")
    if min(h, w) < 200: reasons.append("tiny")
    if gray.mean() < 70: reasons.append("dark")
    if (gray >= 250).mean() > 0.55: reasons.append("mostly-white")
    if any(src.startswith(s.rsplit('-',1)[0]) for s in review) and reasons == []:
        pass
    if reasons:
        flagged.append((name, ar, reasons, p))

print(f"flagged crops: {len(flagged)}")
from collections import Counter
rc = Counter(r for _,_,rs,_ in flagged for r in rs)
print("reasons:", dict(rc))
for name, ar, rs, _ in flagged[:25]:
    print(f"  {name:<24} ar={ar:.2f}  {','.join(rs)}")

if flagged:
    tiles = []
    for name, ar, rs, p in flagged[:40]:
        t = cv2.resize(cv2.imread(p), (150, 210))
        cv2.putText(t, ",".join(rs)[:16], (3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,220), 1)
        tiles.append(t)
    g = [np.hstack(tiles[i:i+8]) for i in range(0, len(tiles), 8)]
    W = max(r.shape[1] for r in g)
    g = [cv2.copyMakeBorder(r, 2, 2, 0, W-r.shape[1], cv2.BORDER_CONSTANT, value=(160,160,160)) for r in g]
    cv2.imwrite(mon, np.vstack(g))
    print(f"montage -> {mon}")
