"""3-way folio-segmenter comparison: current vs fade-aug (p=0.6) vs fade-aug (p=0.35).
Synthetic-fade IoU on the held-out val split + coverage on real faded & clean pages."""
import torch, cv2, numpy as np, glob, os, random
REPO = r"C:\Users\mahajar\Downloads\sample images\ssda-folio-pipeline"
DATA = r"C:\Users\mahajar\Downloads\sample images\_preprocessed\preprocessed"
SRC = r"C:\Users\mahajar\Downloads\sample images\_task2\full_src"
IMEAN = np.array([0.485,0.456,0.406],np.float32); ISTD = np.array([0.229,0.224,0.225],np.float32); S=512
dev = "cuda" if torch.cuda.is_available() else "cpu"

def load(n):
    p = os.path.join(REPO, "weights", n)
    return torch.jit.load(p).eval().to(dev) if os.path.exists(p) else None
M = {"current": load("folio_seg_unet.pt.ts.pt"),
     "fade.60":  load("folio_seg_unet_faded.pt.ts.pt"),
     "fade.35":  load("folio_seg_unet_faded2.pt.ts.pt")}
M = {k:v for k,v in M.items() if v is not None}

def prep(im):
    x = cv2.cvtColor(cv2.resize(im,(S,S)),cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    return torch.from_numpy(((x-IMEAN)/ISTD).transpose(2,0,1))[None].to(dev)
def pmask(m, im):
    with torch.no_grad(): return torch.sigmoid(m(prep(im)))[0,0].cpu().numpy() > 0.5
def fade(im, mk, f):
    fd = np.clip(im.astype(np.float32)*f + 255*(1-f), 0, 255).astype(np.uint8)
    o = im.copy(); o[mk] = fd[mk]; return o

def pairs(a,b):
    out=[]
    for ip in sorted(glob.glob(os.path.join(a,"*"))):
        s=os.path.splitext(os.path.basename(ip))[0]
        for e in (".png",".jpg",".jpeg"):
            mp=os.path.join(b,s+e)
            if os.path.exists(mp): out.append((ip,mp)); break
    return out
ps = pairs(os.path.join(DATA,"images"),os.path.join(DATA,"masks"))
random.seed(0); random.shuffle(ps); val = ps[:max(1,int(len(ps)*0.15))]

print("== held-out val IoU (GT) ==")
print(f"{'model':<9}{'normal':>8}{'fade.6':>8}{'fade.45':>9}{'fade.3':>8}")
for name,m in M.items():
    r=[]
    for f in (1.0,0.6,0.45,0.30):
        io=[]
        for ip,mp in val:
            im=cv2.resize(cv2.imread(ip),(S,S)); mk=cv2.resize(cv2.imread(mp,0),(S,S),interpolation=cv2.INTER_NEAREST)>127
            p=pmask(m, im if f==1.0 else fade(im,mk,f))
            io.append((np.logical_and(p,mk).sum()+1)/(np.logical_or(p,mk).sum()+1))
        r.append(np.mean(io))
    print(f"{name:<9}{r[0]:>8.3f}{r[1]:>8.3f}{r[2]:>9.3f}{r[3]:>8.3f}")

faded_pgs=["29597-0078","201991-0163","29597-0128","29597-0188","201991-0159"]
clean_pgs=["375062-0181","176899-0021","176899-0148","176899-0010","201991-0250"]
print("\n== coverage % on REAL pages (higher = less under-segmentation on faded) ==")
hdr = f"{'page':<15}" + "".join(f"{k:>10}" for k in M); print(hdr)
def cov_row(pg):
    im=cv2.imread(os.path.join(SRC,pg+".jpg"))
    if im is None: return None
    return pg, {k: pmask(m,im).mean()*100 for k,m in M.items()}
for grp,label in ((faded_pgs,"FADED"),(clean_pgs,"clean")):
    print(f"-- {label} --")
    for pg in grp:
        r=cov_row(pg)
        if r: print(f"{r[0]:<15}" + "".join(f"{r[1][k]:>10.1f}" for k in M))
