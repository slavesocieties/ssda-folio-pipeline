"""Compare the current folio segmenter vs a faded-augmented candidate, specifically
on LOW-CONTRAST pages (the failure mode that caused the white-out erosion).

- IoU on the seed-0 held-out val split (held out for both, since both train seed 0),
  at increasing fade levels (page bleached toward white).
- Visual mask overlay on the REAL faded pages that failed (no GT, qualitative).
"""
import torch, cv2, numpy as np, glob, os, random, sys

REPO = r"C:\Users\mahajar\Downloads\sample images\ssda-folio-pipeline"
DATA = r"C:\Users\mahajar\Downloads\sample images\_preprocessed\preprocessed"
FULLSRC = r"C:\Users\mahajar\Downloads\sample images\_task2\full_src"
IMEAN = np.array([0.485, 0.456, 0.406], np.float32); ISTD = np.array([0.229, 0.224, 0.225], np.float32)
SIZE = 512
dev = "cuda" if torch.cuda.is_available() else "cpu"


def pairs(imd, mkd):
    out = []
    for ip in sorted(glob.glob(os.path.join(imd, "*"))):
        s = os.path.splitext(os.path.basename(ip))[0]
        for e in (".png", ".jpg", ".jpeg"):
            mp = os.path.join(mkd, s + e)
            if os.path.exists(mp): out.append((ip, mp)); break
    return out


def prep(im):
    x = cv2.cvtColor(cv2.resize(im, (SIZE, SIZE)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = (x - IMEAN) / ISTD
    return torch.from_numpy(x.transpose(2, 0, 1))[None].to(dev)


def prob(model, im):
    with torch.no_grad():
        return torch.sigmoid(model(prep(im)))[0, 0].cpu().numpy()


def fade(im, mk_bool, f):
    faded = np.clip(im.astype(np.float32) * f + 255 * (1 - f), 0, 255).astype(np.uint8)
    out = im.copy(); out[mk_bool] = faded[mk_bool]; return out


def load(p):
    return torch.jit.load(p).eval().to(dev) if os.path.exists(p) else None


models = {"current": load(os.path.join(REPO, "weights", "folio_seg_unet.pt.ts.pt")),
          "faded":   load(os.path.join(REPO, "weights", "folio_seg_unet_faded.pt.ts.pt"))}

ps = pairs(os.path.join(DATA, "images"), os.path.join(DATA, "masks"))
random.seed(0); random.shuffle(ps); nval = max(1, int(len(ps) * 0.15)); val = ps[:nval]
print(f"held-out val: {len(val)} pairs\n")
print(f"{'model':<9}{'normal':>9}{'fade0.6':>9}{'fade0.45':>10}{'fade0.3':>9}")
for name, m in models.items():
    if m is None: print(f"{name}: weight missing"); continue
    cols = []
    for f in (1.0, 0.6, 0.45, 0.30):
        ious = []
        for ip, mp in val:
            im = cv2.resize(cv2.imread(ip), (SIZE, SIZE))
            mk = cv2.resize(cv2.imread(mp, 0), (SIZE, SIZE), interpolation=cv2.INTER_NEAREST) > 127
            x = im if f == 1.0 else fade(im, mk, f)
            p = prob(m, x) > 0.5
            ious.append((np.logical_and(p, mk).sum() + 1) / (np.logical_or(p, mk).sum() + 1))
        cols.append(np.mean(ious))
    print(f"{name:<9}{cols[0]:>9.3f}{cols[1]:>9.3f}{cols[2]:>10.3f}{cols[3]:>9.3f}")

# visual: masks on real faded source pages
reals = ["29597-0078", "201991-0163", "29597-0128", "375062-0181"]
tiles = []
for r in reals:
    ip = os.path.join(FULLSRC, r + ".jpg")
    if not os.path.exists(ip): continue
    im = cv2.resize(cv2.imread(ip), (SIZE, SIZE))
    row = [cv2.putText(im.copy(), r, (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 230), 2)]
    for name, m in models.items():
        if m is None: continue
        p = (prob(m, im) > 0.5).astype(np.uint8) * 255
        ov = im.copy(); ov[p > 0] = (0.5 * ov[p > 0] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
        cv2.putText(ov, name + f" cov={100*(p>0).mean():.0f}%", (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 0), 2)
        row.append(ov)
    tiles.append(np.hstack(row))
if tiles:
    out = os.path.join(r"C:\Users\mahajar\Downloads\sample images\_task2\_audit", "_seg_faded_masks.jpg")
    cv2.imwrite(out, np.vstack(tiles)); print(f"\nmask overlay (src | current | faded) -> {out}")
