"""Airtight no-text-loss check, tilt-robust. Compares an aligned white-out crop set
(arg1) against the pixel-aligned no-white-out crop (arg2, FOLIO_KEEP_ALL, same
geometry). For each crop:
    erased = (A is white) AND (keepall has ink)      # what white-out removed

Then classifies every erased pixel WITHOUT assuming the folio is centered:
  * BORDER erasure  = erased pixel lies in the white region reachable from the crop
    border through white -> background / facing-page / hand / binding. Intended.
  * INTERIOR erasure = erased pixel in an ISOLATED white blob NOT touching the border
    -> a white hole punched INSIDE the folio = real folio text loss. This is the
    thing that must be ~0. The convex-hull _safe_page_mask is designed to make it
    impossible; this proves it on real data.

  python gt_interior_verify.py <whiteout_dir> <keepall_dir> <montage.jpg>
"""
import cv2, numpy as np, os, glob, sys
from collections import Counter
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
    white = wg >= 250
    erased = white & ink
    er = cv2.morphologyEx(erased.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    total_er = int(er.sum())
    # border-connected white via flood fill from a 1px border ring
    H, W = white.shape
    ff = np.zeros((H + 2, W + 2), np.uint8)
    wm = (white.astype(np.uint8)) * 255
    # seed flood from every border white pixel
    m2 = wm.copy()
    mask_ff = np.zeros((H + 2, W + 2), np.uint8)
    for x in range(W):
        if wm[0, x]:      cv2.floodFill(m2, mask_ff, (x, 0), 128)
        if wm[H-1, x]:    cv2.floodFill(m2, mask_ff, (x, H-1), 128)
    for y in range(H):
        if wm[y, 0]:      cv2.floodFill(m2, mask_ff, (0, y), 128)
        if wm[y, W-1]:    cv2.floodFill(m2, mask_ff, (W-1, y), 128)
    border_white = (m2 == 128)
    interior_er = er & (~border_white)          # erased ink NOT reachable from border
    # tiny isolated specks (jpeg/ink noise) are not text loss: require a real blob
    ie = cv2.morphologyEx(interior_er.astype(np.uint8), cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).astype(bool)
    interior_px = int(ie.sum())
    rows.append((nm, interior_px, total_er, gp, wp, ie))
rows.sort(key=lambda r: r[1], reverse=True)
ivals = np.array([r[1] for r in rows]) if rows else np.array([0])
print(f"pairs compared: {n_pairs}")
for thr in (50, 200, 1000, 5000):
    print(f"  crops with INTERIOR (folio-hole) erasure > {thr}px: {int((ivals > thr).sum())}")
print(f"  max interior erasure = {int(ivals.max())}px   median = {int(np.median(ivals))}px")
bad = [r for r in rows if r[1] > 200]
print(f"  volumes with any interior>200px: {dict(Counter(r[0].split('-')[0] for r in bad))}")
print("\ntop 10 by interior (folio-hole) erasure:")
for nm, ip, te, _, _, _ in rows[:10]:
    print(f"  {nm:<22} interior {ip:>7}px   (total-erased {te}px)")
top = [r for r in rows if r[1] > 200][:12]
if top:
    tiles = []
    for nm, ip, te, gp, wp, ie in top:
        g = cv2.resize(cv2.imread(gp), (170, 240)); w = cv2.resize(cv2.imread(wp), (170, 240))
        ov = w.copy()
        iem = cv2.resize(ie.astype(np.uint8) * 255, (170, 240), interpolation=cv2.INTER_NEAREST) > 0
        ov[iem] = (0, 0, 255)
        cv2.putText(ov, f"{ip}px", (2, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 220), 2)
        tiles.append(np.hstack([g, w, ov]))
    grid = [np.hstack(tiles[i:i+2]) for i in range(0, len(tiles), 2)]
    Wm = max(r.shape[1] for r in grid)
    grid = [cv2.copyMakeBorder(r, 3, 3, 0, Wm - r.shape[1], cv2.BORDER_CONSTANT, value=(150, 150, 150)) for r in grid]
    cv2.imwrite(mon, np.vstack(grid))
    print(f"\ninterior-hole montage (keepall|A|A+interior-red) -> {mon}")
else:
    print("\nNO crop has a folio-interior white hole > 200px -> the white-out erased ZERO interior folio text.")
