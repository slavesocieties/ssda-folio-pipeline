"""Full-dataset no-text-loss check. Compares a white-out crop set (arg1) against a
pixel-aligned ZERO-erasure ground-truth (arg2, FOLIO_KEEP_ALL). For every crop:
  erased = (crop is white) AND (ground truth has ink)   -> what the white-out removed
It then splits the erased ink into FOLIO-BODY (central 64%) vs PERIMETER (gutter/bg).
Perimeter erasure is intended (background removal); central erasure is the thing we
must not have (real folio text). Reports the distribution + flags/montages any crop
with meaningful central erasure so residual text loss cannot hide.

  python gt_full_verify.py <whiteout_dir> <keepall_dir> <montage.jpg>
"""
import cv2, numpy as np, os, glob, sys
wdir, gdir, mon = sys.argv[1], sys.argv[2], sys.argv[3]
rows = []
n_pairs = 0
for gp in sorted(glob.glob(os.path.join(gdir, "*.jpg"))):
    nm = os.path.basename(gp)
    if "_enhanced" in nm:
        continue
    wp = os.path.join(wdir, nm)
    if not os.path.exists(wp):
        continue
    g = cv2.imread(gp); w = cv2.imread(wp)
    if g is None or w is None or g.shape != w.shape:
        continue
    n_pairs += 1
    gg = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY); wg = cv2.cvtColor(w, cv2.COLOR_BGR2GRAY)
    ink = gg < 110
    erased = (wg >= 250) & ink
    er = cv2.morphologyEx(erased.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    H, W = er.shape
    cy0, cy1, cx0, cx1 = int(H*.18), int(H*.82), int(W*.18), int(W*.82)
    central_px = int(er[cy0:cy1, cx0:cx1].sum())
    # central erasure as a FRACTION of the folio ink that lives in the central region
    central_ink = int((ink[cy0:cy1, cx0:cx1]).sum())
    central_frac = central_px / max(1, central_ink)
    rows.append((nm, central_frac, central_px, gp, wp, er))
rows.sort(key=lambda r: r[1], reverse=True)
vals = np.array([r[1] for r in rows]) if rows else np.array([0.0])
print(f"pairs compared: {n_pairs}")
for thr in (0.005, 0.01, 0.03):
    print(f"  central(folio-body) erasure >= {thr:.1%}: {int((vals>=thr).sum())} crops")
print(f"  median central erasure = {np.median(vals)*100:.3f}%   max = {vals.max()*100:.2f}%")
from collections import Counter
bad = [r for r in rows if r[1] >= 0.01]
print(f"  volumes with any crop >=1% central erasure: {dict(Counter(r[0].split('-')[0] for r in bad))}")
print("\ntop 12 by central-body erasure:")
for nm, cf, px, _, _, _ in rows[:12]:
    print(f"  {nm:<22} central-erasure {cf*100:5.2f}%  ({px}px)")
top = [r for r in rows if r[1] >= 0.005][:15]
if top:
    tiles = []
    for nm, cf, px, gp, wp, er in top:
        g = cv2.resize(cv2.imread(gp), (150, 210)); w = cv2.resize(cv2.imread(wp), (150, 210))
        erc = cv2.cvtColor(cv2.resize((er*255).astype(np.uint8), (150, 210)), cv2.COLOR_GRAY2BGR)
        erc[:, :, 0] = 0; erc[:, :, 1] = 0
        cv2.putText(w, f"{cf*100:.1f}%", (2, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 220), 2)
        tiles.append(np.hstack([g, w, erc]))
    grid = [np.hstack(tiles[i:i+2]) for i in range(0, len(tiles), 2)]
    Wm = max(r.shape[1] for r in grid)
    grid = [cv2.copyMakeBorder(r, 3, 3, 0, Wm-r.shape[1], cv2.BORDER_CONSTANT, value=(150,150,150)) for r in grid]
    cv2.imwrite(mon, np.vstack(grid)); print(f"\nworst central-erasure montage (GT|crop|erased-red) -> {mon}")
else:
    print("\nNO crop has >=0.5% central (folio-body) erasure — no folio text lost.")
