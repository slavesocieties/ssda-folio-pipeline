"""For the worst edge-ink crops, draw the crop quad on the ORIGINAL image (via the
coordinate artifacts) so we can SEE whether the crop boundary sits on the physical
page edge (fine) or cuts through text (real clip). Also measures dark 'page' ink in
a band JUST OUTSIDE the quad on the flagged edge = content the crop left behind."""
import os, glob, json, sys, cv2, numpy as np

src_dir = sys.argv[1]      # full_src (originals)
coords = sys.argv[2]       # out/full/coords
folios = sys.argv[3]       # out/full/folios
mon = sys.argv[4]
# worst crops from the edge-clip pass (name, edge)
targets = [("375062-0181-A", "B"), ("375062-0143-A", "B"), ("375062-0220-A", "B"),
           ("375062-0123-A", "B"), ("375062-0150-A", "B"), ("375062-0097-A", "T"),
           ("375062-0067-A", "B"), ("375062-0016-A", "B")]

tiles = []
for name, edge in targets:
    cj = os.path.join(coords, name + ".json")
    if not os.path.exists(cj):
        continue
    d = json.load(open(cj))
    src = d.get("source_image"); size = d.get("source_size"); quad = d.get("crop_quad_norm")
    if not (src and size and quad):
        continue
    orig = cv2.imread(os.path.join(src_dir, os.path.basename(src)))
    if orig is None:
        continue
    H, W = orig.shape[:2]
    pts = np.array([[int(x * W), int(y * H)] for x, y in quad], np.int32)
    vis = orig.copy()
    cv2.polylines(vis, [pts], True, (0, 255, 0), max(2, W // 400))
    # measure dark ink just OUTSIDE the quad on the flagged edge (a thin outward band)
    mask = np.zeros((H, W), np.uint8); cv2.fillPoly(mask, [pts], 255)
    k = max(4, int(0.02 * min(H, W)))
    outward = cv2.dilate(mask, np.ones((k, k), np.uint8)) & ~mask
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    band_ink = ((gray < 110) & (outward > 0)).sum() / max(1, (outward > 0).sum())
    tag = f"{name} out-ink={band_ink:.2f}"
    vis = cv2.resize(vis, (300, int(300 * H / W)))
    cv2.putText(vis, tag, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 230), 2)
    tiles.append(vis)
    print(f"{name}: dark ink in 2% band OUTSIDE crop quad = {band_ink:.3f}  (high => content left outside)")

if tiles:
    hh = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, hh-t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(150,150,150)) for t in tiles]
    g = [np.hstack(tiles[i:i+4]) for i in range(0, len(tiles), 4)]
    Wm = max(r.shape[1] for r in g)
    g = [cv2.copyMakeBorder(r, 3, 3, 0, Wm-r.shape[1], cv2.BORDER_CONSTANT, value=(150,150,150)) for r in g]
    cv2.imwrite(mon, np.vstack(g))
    print(f"\noverlay montage -> {mon}")
