"""Which crops materially changed between OLD (eroded) and NEW (safe) — i.e. which
pages had text restored and therefore need re-transcription. Same geometry, so crops
are pixel-aligned; a drop in white area = content recovered. Groups by volume."""
import os, glob, sys, cv2, numpy as np
old_dir, new_dir = sys.argv[1], sys.argv[2]
THR = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0   # % white recovered to count as changed
from collections import defaultdict
per_vol_changed = defaultdict(int); per_vol_total = defaultdict(int)
changed = []
for op in sorted(glob.glob(os.path.join(old_dir, "*.jpg"))):
    name = os.path.basename(op)
    if "_enhanced" in name: continue
    npth = os.path.join(new_dir, name)
    if not os.path.exists(npth): continue
    o = cv2.imread(op); n = cv2.imread(npth)
    if o is None or n is None: continue
    if o.shape != n.shape: n = cv2.resize(n, (o.shape[1], o.shape[0]))
    ow = (cv2.cvtColor(o, cv2.COLOR_BGR2GRAY) >= 250).mean() * 100
    nw = (cv2.cvtColor(n, cv2.COLOR_BGR2GRAY) >= 250).mean() * 100
    vol = name.split("-")[0]
    per_vol_total[vol] += 1
    recovered = ow - nw
    if recovered >= THR:
        per_vol_changed[vol] += 1
        changed.append((name, recovered))
changed.sort(key=lambda r: r[1], reverse=True)
print(f"materially-changed crops (>= {THR}% content recovered): {len(changed)} / {sum(per_vol_total.values())}")
print(f"{'vol':<9}{'changed':>9}{'total':>7}{'pct':>7}")
for v in sorted(per_vol_total):
    c = per_vol_changed[v]; t = per_vol_total[v]
    print(f"{v:<9}{c:>9}{t:>7}{100*c/t:>6.0f}%")
# save the list of changed crop names (re-transcription set) per volume
outdir = os.path.join(os.path.dirname(new_dir), "_retranscribe")
os.makedirs(outdir, exist_ok=True)
byv = defaultdict(list)
for name, _ in changed: byv[name.split("-")[0]].append(name)
for v, names in byv.items():
    open(os.path.join(outdir, f"{v}.txt"), "w").write("\n".join(sorted(names)))
print(f"\nchanged-crop lists per volume -> {outdir}")
print("top 12 most-recovered:")
for name, r in changed[:12]:
    print(f"  {name:<22} +{r:.0f}% content")
