"""Detect over-cropping / clipped text: with background whited out, a good folio has
a white margin, so DARK (ink) pixels in the outermost border band mean the page/text
was cut at that edge. Reports the distribution + flags worst offenders + montage."""
import os, glob, sys, cv2, numpy as np
folios = sys.argv[1]
mon = sys.argv[2]
BAND = 0.012          # outer 1.2% of each side
DARK = 110            # ink threshold (0-255)
FLAG = 0.04           # >4% of a border band being ink => likely clipped

rows = []
for p in sorted(glob.glob(os.path.join(folios, "*.jpg"))):
    if "_enhanced" in p:
        continue
    im = cv2.imread(p)
    if im is None:
        continue
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    bh, bw = max(2, int(h * BAND)), max(2, int(w * BAND))
    dark = (g < DARK).astype(np.float32)
    e = {"T": dark[:bh, :].mean(), "B": dark[-bh:, :].mean(),
         "L": dark[:, :bw].mean(), "R": dark[:, -bw:].mean()}
    worst = max(e, key=e.get)
    rows.append((os.path.basename(p), e[worst], worst, p))

rows.sort(key=lambda r: r[1], reverse=True)
vals = np.array([r[1] for r in rows])
flagged = [r for r in rows if r[1] >= FLAG]
print(f"crops analyzed: {len(rows)}")
print(f"max-edge-ink: median={np.median(vals):.3f}  p90={np.percentile(vals,90):.3f}  "
      f"p99={np.percentile(vals,99):.3f}  max={vals.max():.3f}")
print(f"flagged (>= {FLAG:.0%} ink on an edge): {len(flagged)} ({100*len(flagged)/len(rows):.1f}%)")
from collections import Counter
print("worst edge among flagged:", dict(Counter(r[2] for r in flagged)))
print("\ntop 20 (edge, ink-frac):")
for name, v, edge, _ in rows[:20]:
    print(f"  {name:<24} {edge} {v:.3f}")

# montage the 24 worst, drawing a red bar on the clipped edge
top = rows[:24]
tiles = []
for name, v, edge, p in top:
    t = cv2.resize(cv2.imread(p), (200, 280))
    H, W = t.shape[:2]
    bar = 6
    if edge == "T": t[:bar, :] = (0, 0, 255)
    if edge == "B": t[-bar:, :] = (0, 0, 255)
    if edge == "L": t[:, :bar] = (0, 0, 255)
    if edge == "R": t[:, -bar:] = (0, 0, 255)
    cv2.putText(t, f"{edge}{v:.2f}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 220), 2)
    tiles.append(t)
g = [np.hstack(tiles[i:i+6]) for i in range(0, len(tiles), 6)]
Wm = max(r.shape[1] for r in g)
g = [cv2.copyMakeBorder(r, 2, 2, 0, Wm-r.shape[1], cv2.BORDER_CONSTANT, value=(150,150,150)) for r in g]
cv2.imwrite(mon, np.vstack(g))
print(f"\nworst-24 montage -> {mon}")
