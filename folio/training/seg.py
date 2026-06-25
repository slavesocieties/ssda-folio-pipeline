"""Folio (page) segmentation training.

Trains a U-Net (ImageNet-pretrained encoder via smp) to predict the single-folio
PAGE mask from a scan — the robust, background-agnostic page boundary that the
classical paper detector can't get on light-on-light scans. Small dataset (the
~151 preprocessed folio-level mask pairs), so we lean on a pretrained encoder and
strong augmentation, especially colour/illumination jitter so it generalises
across dark- and light-background scans.

    python -m folio.training.seg --images <dir> --masks <dir> \
        --out weights/folio_seg_unet.pt --epochs 60 --bs 8 --size 512

Windows: keep it single-process (no DataLoader workers). Saves the best-val-IoU
state dict and a TorchScript export (<out> and <out>.ts.pt).
"""
from __future__ import annotations
import argparse, glob, os, random
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _pairs(images_dir, masks_dir):
    out = []
    for ip in sorted(glob.glob(os.path.join(images_dir, "*"))):
        stem = os.path.splitext(os.path.basename(ip))[0]
        for ext in (".png", ".jpg", ".jpeg"):
            mp = os.path.join(masks_dir, stem + ext)
            if os.path.exists(mp):
                out.append((ip, mp)); break
    return out


class SegDS(Dataset):
    def __init__(self, pairs, size, train):
        self.pairs, self.size, self.train = pairs, size, train

    def __len__(self):
        return len(self.pairs)

    def _aug(self, im, mk):
        # geometric: flip + small rotation (page can be slightly skewed)
        if random.random() < 0.5:
            im, mk = im[:, ::-1], mk[:, ::-1]
        if random.random() < 0.7:
            ang = random.uniform(-12, 12); h, w = im.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, random.uniform(0.92, 1.10))
            im = cv2.warpAffine(im, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            mk = cv2.warpAffine(mk, M, (w, h), flags=cv2.INTER_NEAREST)
        # photometric: brightness/contrast/gamma + hue/sat (generalise backgrounds)
        if random.random() < 0.8:
            a = random.uniform(0.75, 1.3); b = random.uniform(-25, 25)
            im = np.clip(im.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
        if random.random() < 0.5:
            hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV).astype(np.int16)
            hsv[..., 0] = (hsv[..., 0] + random.randint(-12, 12)) % 180
            hsv[..., 1] = np.clip(hsv[..., 1] + random.randint(-30, 30), 0, 255)
            im = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return np.ascontiguousarray(im), np.ascontiguousarray(mk)

    def __getitem__(self, i):
        ip, mp = self.pairs[i]
        im = cv2.imread(ip); mk = cv2.imread(mp, 0)
        if im.shape[:2] != (self.size, self.size):
            im = cv2.resize(im, (self.size, self.size))
        if mk.shape[:2] != (self.size, self.size):
            mk = cv2.resize(mk, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        if self.train:
            im, mk = self._aug(im, mk)
        x = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(x.transpose(2, 0, 1))
        y = torch.from_numpy((mk > 127).astype(np.float32))[None]
        return x, y


def iou(logits, y):
    p = (torch.sigmoid(logits) > 0.5).float()
    inter = (p * y).sum((1, 2, 3)); union = ((p + y) > 0).float().sum((1, 2, 3))
    return ((inter + 1) / (union + 1)).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True); ap.add_argument("--masks", required=True)
    ap.add_argument("--out", default="weights/folio_seg_unet.pt")
    ap.add_argument("--encoder", default="tu-resnet34")
    ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--size", type=int, default=512); ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    pairs = _pairs(args.images, args.masks)
    random.shuffle(pairs)
    nval = max(1, int(len(pairs) * args.val_frac))
    val, train = pairs[:nval], pairs[nval:]
    print(f"pairs: {len(pairs)} -> train {len(train)} / val {len(val)}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    import segmentation_models_pytorch as smp
    model = smp.Unet(encoder_name=args.encoder, encoder_weights="imagenet",
                     in_channels=3, classes=1).to(dev)
    tl = DataLoader(SegDS(train, args.size, True), batch_size=args.bs, shuffle=True, num_workers=0)
    vl = DataLoader(SegDS(val, args.size, False), batch_size=args.bs, shuffle=False, num_workers=0)

    dice = smp.losses.DiceLoss(mode="binary"); bce = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    best = 0.0
    for ep in range(args.epochs):
        model.train()
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            out = model(x)
            loss = dice(out, y) + bce(out, y)
            loss.backward(); opt.step()
        sched.step()
        model.eval(); ious = []
        with torch.no_grad():
            for x, y in vl:
                ious.append(iou(model(x.to(dev)), y.to(dev)))
        miou = float(np.mean(ious))
        print(f"ep {ep+1:02d}/{args.epochs}  val_iou={miou:.4f}" + ("  *best" if miou > best else ""))
        if miou > best:
            best = miou
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "encoder": args.encoder,
                        "size": args.size, "val_iou": best}, args.out)
            model.eval()
            ex = torch.randn(1, 3, args.size, args.size, device=dev)
            ts = torch.jit.trace(model, ex)
            ts.save(args.out + ".ts.pt")
    print(f"done. best val IoU = {best:.4f}  -> {args.out} (+ .ts.pt)")


if __name__ == "__main__":
    main()
