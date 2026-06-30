import os, csv, glob, sys, cv2, numpy as np
from collections import defaultdict, Counter
out = sys.argv[1]; mondir = sys.argv[2]
os.makedirs(mondir, exist_ok=True)
rows = list(csv.DictReader(open(os.path.join(out, "manifest.csv"), encoding="utf-8")))
byvol_rows = defaultdict(list)
for r in rows: byvol_rows[r["source"].split("-")[0]].append(r)
files = sorted(p for p in glob.glob(os.path.join(out, "folios", "*.jpg")) if "_enhanced" not in p)
byvol_f = defaultdict(list)
for p in files: byvol_f[os.path.basename(p).split("-")[0]].append(p)

print(f"{'vol':<10}{'imgs':>5}{'folios':>7}{'1/2-folio':>12}{'med_ar':>8}{'land':>6}{'review':>8}{'blank':>7}")
for vol in sorted(byvol_f):
    rs = byvol_rows[vol]; fs = byvol_f[vol]
    pc = Counter(r["page_count"] for r in rs)
    ars = []; land = 0
    for p in fs:
        im = cv2.imread(p); ar = im.shape[1]/im.shape[0]; ars.append(ar)
        if ar >= 1.0: land += 1
    rev = sum(r["needs_review"]=="True" for r in rs); bl = sum(r["is_blank"]=="True" for r in rs)
    print(f"{vol:<10}{len(set(r['source'] for r in rs)):>5}{len(fs):>7}"
          f"{str(pc.get('one_folio',0))+'/'+str(pc.get('two_folios',0)):>12}"
          f"{np.median(ars):>8.2f}{land:>6}{rev:>8}{bl:>7}")
    # montage for this volume
    tiles = [cv2.resize(cv2.imread(p),(150,210)) for p in fs]
    per = 8
    g = [np.hstack(tiles[i:i+per]) for i in range(0,len(tiles),per)]
    W = max(r.shape[1] for r in g)
    g = [cv2.copyMakeBorder(r,2,2,0,W-r.shape[1],cv2.BORDER_CONSTANT,value=(160,160,160)) for r in g]
    cv2.imwrite(os.path.join(mondir, f"vol_{vol}.jpg"), np.vstack(g))
print(f"\nper-volume montages -> {mondir}")
