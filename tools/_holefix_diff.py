"""Show which crops the hole-fill changed: for crops in both dirs, count near-white
pixels in each; the ones where v2 has materially FEWER white pixels are holes that
got filled back in. Writes a side-by-side montage of the top-changed crops."""
import os, glob, sys, cv2, numpy as np
v1, v2, mon = sys.argv[1], sys.argv[2], sys.argv[3]

def white_frac(im):
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return float((g >= 250).mean())

rows = []
for p2 in sorted(glob.glob(os.path.join(v2, "folios", "*.jpg"))):
    name = os.path.basename(p2)
    if "_enhanced" in name: continue
    p1 = os.path.join(v1, "folios", name)
    if not os.path.exists(p1): continue
    a, b = cv2.imread(p1), cv2.imread(p2)
    if a is None or b is None: continue
    d = white_frac(a) - white_frac(b)          # positive => v2 filled holes
    rows.append((d, name, p1, p2))

rows.sort(reverse=True)
changed = [r for r in rows if r[0] > 0.01]
print(f"crops compared: {len(rows)}   materially de-holed (>1% less white): {len(changed)}")
for d, name, _, _ in changed[:15]:
    print(f"  {name:<22} -{d*100:5.1f}% white")

top = changed[:8]
if top:
    tiles = []
    for d, name, p1, p2 in top:
        a = cv2.resize(cv2.imread(p1), (150, 210)); b = cv2.resize(cv2.imread(p2), (150, 210))
        sep = np.full((210, 4, 3), (0, 0, 200), np.uint8)   # red divider: before|after
        tiles.append(np.hstack([a, sep, b]))
    g = [np.hstack(tiles[i:i+4]) for i in range(0, len(tiles), 4)]
    W = max(r.shape[1] for r in g)
    g = [cv2.copyMakeBorder(r, 3, 3, 0, W-r.shape[1], cv2.BORDER_CONSTANT, value=(160,160,160)) for r in g]
    cv2.imwrite(mon, np.vstack(g))
    print(f"before|after montage -> {mon}")
else:
    print("no crops changed materially")
