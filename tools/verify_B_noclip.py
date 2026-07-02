"""Algorithmic clip test for approach B (tight, no white-out) WITHOUT needing a
pixel-aligned ground truth for B. Chain:
  * A (white-out) and KEEPALL (FOLIO_KEEP_ALL, no white-out) share identical
    geometry -> pixel-aligned. A's DARK pixels are exactly the folio ink (its
    background is white), so A gives an isolated folio-ink map in KEEPALL coords.
  * B is a tighter rectangle cut from the same warped page as KEEPALL and has the
    SAME real background, so B locates reliably inside KEEPALL by template matching
    (matching B against the *whited* A fails -- different backgrounds).
Locate B in KEEPALL, then measure how much of A's folio ink (same coords) falls
OUTSIDE B's window = folio text B clipped away.

  python verify_B_noclip.py <A_whiteout_dir> <keepall_dir> <B_tight_dir> <montage.jpg>
"""
import cv2, numpy as np, os, glob, sys
from collections import Counter
Adir, Gdir, Bdir, mon = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
INK = 110
TARGET_W = 500          # matching resolution (downscale for speed)
rows = []
n = 0
for ap in sorted(glob.glob(os.path.join(Adir, "*.jpg"))):
    nm = os.path.basename(ap)
    if "_enhanced" in nm:
        continue
    gp = os.path.join(Gdir, nm); bp = os.path.join(Bdir, nm)
    if not (os.path.exists(gp) and os.path.exists(bp)):
        continue
    A = cv2.imread(ap, cv2.IMREAD_GRAYSCALE)
    G = cv2.imread(gp, cv2.IMREAD_GRAYSCALE)      # keepall: same geometry as A, real bg
    B = cv2.imread(bp, cv2.IMREAD_GRAYSCALE)
    if A is None or G is None or B is None or A.shape != G.shape:
        continue
    n += 1
    sA = TARGET_W / G.shape[1]
    Gs = cv2.resize(G, (TARGET_W, max(1, int(G.shape[0] * sA))))
    As = cv2.resize(A, (Gs.shape[1], Gs.shape[0]))
    Bs = cv2.resize(B, (max(1, int(B.shape[1] * sA)), max(1, int(B.shape[0] * sA))))
    if Bs.shape[0] > Gs.shape[0] or Bs.shape[1] > Gs.shape[1]:
        Bs = Bs[:min(Bs.shape[0], Gs.shape[0]), :min(Bs.shape[1], Gs.shape[1])]
    res = cv2.matchTemplate(Gs, Bs, cv2.TM_CCOEFF_NORMED)   # B vs keepall (same bg)
    _, score, _, loc = cv2.minMaxLoc(res)
    x0, y0 = loc
    x1, y1 = x0 + Bs.shape[1], y0 + Bs.shape[0]
    win = np.zeros_like(As, dtype=bool)
    win[y0:y1, x0:x1] = True
    # A's isolated folio ink. A and keepall are the same geometry, but the orientation
    # head can flip 0-vs-180 between runs on ambiguous pages -> A may be upside down
    # vs the B/keepall frame. Try both; the true alignment is the one that fills B's
    # window (min ink outside). A genuine clip loses ink in BOTH orientations.
    def outside(ink):
        return int((ink & ~win).sum()) / max(1, int(ink.sum()))
    ink0 = As < INK
    ink180 = (cv2.rotate(As, cv2.ROTATE_180) < INK)
    frac = min(outside(ink0), outside(ink180))
    clipped = int(round(frac * int(ink0.sum())))
    rows.append((nm, frac, clipped, score, ap, bp, (x0, y0, x1, y1), Gs.shape))
rows.sort(key=lambda r: r[1], reverse=True)
vals = np.array([r[1] for r in rows]) if rows else np.array([0.0])
print(f"pages compared (A<->B): {n}")
for thr in (0.005, 0.01, 0.03, 0.05):
    print(f"  B clips >= {thr:.1%} of folio ink: {int((vals >= thr).sum())} pages")
print(f"  median clipped = {np.median(vals)*100:.3f}%   p99 = {np.percentile(vals,99)*100:.2f}%   max = {vals.max()*100:.2f}%")
lowscore = [r for r in rows if r[3] < 0.5]
print(f"  low match-confidence pages (score<0.5, result unreliable): {len(lowscore)}")
bad = [r for r in rows if r[1] >= 0.01 and r[3] >= 0.5]
print(f"  volumes with a real clip >=1%: {dict(Counter(r[0].split('-')[0] for r in bad))}")
print("\ntop 12 by clipped-ink fraction:")
for nm, fr, cp, sc, *_ in rows[:12]:
    tag = "  (LOW-CONF match)" if sc < 0.5 else ""
    print(f"  {nm:<22} clipped {fr*100:5.2f}%  match={sc:.2f}{tag}")
top = [r for r in rows if r[1] >= 0.005][:12]
if top:
    tiles = []
    for nm, fr, cp, sc, ap, bp, box, shp in top:
        A = cv2.resize(cv2.imread(ap), (shp[1], shp[0]))
        cv2.rectangle(A, (box[0], box[1]), (box[2]-1, box[3]-1), (0, 0, 255), 2)
        cv2.putText(A, f"{fr*100:.1f}% m{sc:.2f}", (3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 220), 1)
        B = cv2.resize(cv2.imread(bp), (shp[1], shp[0]))
        tiles.append(np.hstack([A, np.full((A.shape[0], 6, 3), 200, np.uint8), B]))
    Wm = max(t.shape[1] for t in tiles); hh = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, hh-t.shape[0], 0, Wm-t.shape[1], cv2.BORDER_CONSTANT, value=(150,150,150)) for t in tiles]
    grid = [np.hstack(tiles[i:i+3]) for i in range(0, len(tiles), 3)]
    Wm = max(g.shape[1] for g in grid)
    grid = [cv2.copyMakeBorder(g, 3, 3, 0, Wm-g.shape[1], cv2.BORDER_CONSTANT, value=(120,120,120)) for g in grid]
    cv2.imwrite(mon, np.vstack(grid))
    print(f"\nworst-clip montage (A+matched-box | B) -> {mon}")
else:
    print("\nNO page has B clipping >=0.5% of A's folio ink -> B loses no folio text.")
