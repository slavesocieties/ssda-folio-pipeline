"""Find white-out NOTCHES: white pixels sitting INSIDE the page's convex hull — i.e.
regions the background-removal erased that are surrounded by page (a bite out of the
folio). Distinct from edge-clipping. Reports prevalence + montages the worst."""
import os, glob, sys, cv2, numpy as np
folios = sys.argv[1]; mon = sys.argv[2]
rows = []
for p in sorted(glob.glob(os.path.join(folios, "*.jpg"))):
    if "_enhanced" in p: continue
    im = cv2.imread(p)
    if im is None: continue
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    page = (g < 248).astype(np.uint8)                    # non-white = page/ink
    n, lab, stats, _ = cv2.connectedComponentsWithStats(page, 8)
    if n < 2: continue
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    pm = (lab == big).astype(np.uint8)
    pts = cv2.findNonZero(pm)
    if pts is None or len(pts) < 50: continue
    hull = cv2.convexHull(pts)
    hm = np.zeros_like(pm); cv2.fillConvexPoly(hm, hull, 1)
    white = (g >= 250).astype(np.uint8)
    bite = ((hm == 1) & (white == 1)).astype(np.uint8)   # white INSIDE the page hull
    # keep only sizeable bites (drop 1px seam noise)
    bite = cv2.morphologyEx(bite, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    frac = bite.sum() / max(1, hm.sum())
    rows.append((os.path.basename(p), frac, p))
rows.sort(key=lambda r: r[1], reverse=True)
vals = np.array([r[1] for r in rows])
for thr in (0.03, 0.05, 0.10):
    print(f"  notch-bite >= {thr:.0%}: {int((vals>=thr).sum())} crops ({100*(vals>=thr).mean():.1f}%)")
print(f"median bite={np.median(vals):.4f}  max={vals.max():.3f}")
print("\ntop 24 (bite-frac):")
for name, f, _ in rows[:24]:
    print(f"  {name:<22} {f:.3f}")
tiles = [cv2.putText(cv2.resize(cv2.imread(p),(180,250)), f"{f*100:.0f}%", (4,20),
         cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,230),2) for name,f,p in rows[:24]]
g2 = [np.hstack(tiles[i:i+6]) for i in range(0,len(tiles),6)]
W = max(r.shape[1] for r in g2)
g2 = [cv2.copyMakeBorder(r,2,2,0,W-r.shape[1],cv2.BORDER_CONSTANT,value=(150,150,150)) for r in g2]
cv2.imwrite(mon, np.vstack(g2))
print(f"\nworst-24 notch montage -> {mon}")
