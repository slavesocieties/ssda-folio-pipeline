"""Definitive loss detector. Compares two crop sets produced by the SAME geometry
(pixel-aligned): one with white-out (safe), one with white-out disabled (ground
truth = every page pixel kept). Any pixel that is WHITE in the safe crop but INK in
the no-white-out crop is erased text. Reports per-crop erased-ink fraction, the
volumes affected, and montages the worst so residual loss can be fixed."""
import os, glob, sys, cv2, numpy as np
safe_dir, nowo_dir, mon = sys.argv[1], sys.argv[2], sys.argv[3]
WHITE, INK = 250, 110
rows = []
for sp in sorted(glob.glob(os.path.join(safe_dir, "*.jpg"))):
    name = os.path.basename(sp)
    if "_enhanced" in name:
        continue
    npth = os.path.join(nowo_dir, name)
    if not os.path.exists(npth):
        continue
    s = cv2.imread(sp); n = cv2.imread(npth)
    if s is None or n is None:
        continue
    if s.shape != n.shape:                       # align if aspect-cap padded differently
        n = cv2.resize(n, (s.shape[1], s.shape[0]))
    sg = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY); ng = cv2.cvtColor(n, cv2.COLOR_BGR2GRAY)
    ink = (ng < INK)
    erased = (sg >= WHITE) & ink
    # drop 1px seam noise
    er = cv2.morphologyEx(erased.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    frac = er.sum() / max(1, ink.sum())          # fraction of page ink that got erased
    rows.append((name, frac, int(er.sum()), sp, npth))

rows.sort(key=lambda r: r[1], reverse=True)
vals = np.array([r[1] for r in rows])
print(f"crops compared: {len(rows)}")
for thr in (0.005, 0.02, 0.05):
    print(f"  erased-ink >= {thr:.1%} of page text: {int((vals>=thr).sum())} crops")
print(f"  median erased={np.median(vals):.4f}  max={vals.max():.4f}")
from collections import Counter
aff = Counter(r[0].split("-")[0] for r in rows if r[1] >= 0.02)
print("  volumes with >=2% erasure:", dict(aff))
print("\ntop 15 residual-loss crops:")
for name, f, px, _, _ in rows[:15]:
    print(f"  {name:<22} erased={f:.3f} ({px}px)")

top = [r for r in rows if r[1] >= 0.01][:18]
if top:
    tiles = []
    for name, f, px, sp, npth in top:
        s = cv2.resize(cv2.imread(sp), (150, 210)); n = cv2.resize(cv2.imread(npth), (150, 210))
        sep = np.full((210, 3, 3), (0, 0, 200), np.uint8)
        t = np.hstack([s, sep, n])
        cv2.putText(t, f"{f*100:.0f}%", (2, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 220), 2)
        tiles.append(t)
    g = [np.hstack(tiles[i:i+4]) for i in range(0, len(tiles), 4)]
    W = max(r.shape[1] for r in g)
    g = [cv2.copyMakeBorder(r, 2, 2, 0, W-r.shape[1], cv2.BORDER_CONSTANT, value=(150,150,150)) for r in g]
    cv2.imwrite(mon, np.vstack(g))
    print(f"\nworst residual-loss (safe|nowo) montage -> {mon}")
else:
    print("\nNO residual loss >= 1% — safe white-out erases no text.")
