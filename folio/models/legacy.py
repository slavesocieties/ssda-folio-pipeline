"""Faithful reconstruction of the LEGACY pipeline so we can score the fresh
approach against it on the same images (see tools/eval_vs_legacy.py).

Mirrors the original image-processing scripts: ResNet-18 folio-count and
upside-down classifiers + a from-scratch U-Net for folio segmentation, with the
hardcoded W//2 two-folio split and 180-only orientation. Torch/torchvision are
imported lazily so the package still imports without them.

Download the .pth files from the project Drive into one folder and pass it as
weights_dir:
  folio_count_classifier.pth   upside_down.pth (or folio_upside_down.pth)
  unet_folio_split.pth
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import numpy as np
import cv2

IMG_SIZE = (640, 960)  # (W,H) used during legacy U-Net training


def _torch():
    import torch
    return torch


class _ResNet18Binary:
    def __init__(self, model_path, class_names, device):
        torch = _torch()
        from torchvision import models
        from torchvision.models import ResNet18_Weights
        self.device = torch.device(device)
        self.class_names = class_names
        net = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        net.fc = torch.nn.Linear(net.fc.in_features, 2)
        net.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model = net.eval().to(self.device)

    def predict(self, bgr) -> Tuple[str, float]:
        torch = _torch()
        from torchvision import transforms
        from PIL import Image
        tf = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        t = tf(pil).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            p = torch.softmax(self.model(t), 1)[0]
        i = int(p.argmax())
        return self.class_names[i], float(p[i])


def _build_unet():
    torch = _torch()
    nn = torch.nn

    class DoubleConv(nn.Module):
        def __init__(s, i, o):
            super().__init__()
            s.seq = nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True),
                                  nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True))
        def forward(s, x): return s.seq(x)

    class UNet(nn.Module):
        def __init__(s, n_ch=3, n_cls=2):
            super().__init__()
            s.inc = DoubleConv(n_ch, 64)
            s.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
            s.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
            s.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
            s.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 512))
            s.up1 = nn.ConvTranspose2d(512, 512, 2, stride=2); s.conv1 = DoubleConv(1024, 256)
            s.up2 = nn.ConvTranspose2d(256, 256, 2, stride=2); s.conv2 = DoubleConv(512, 128)
            s.up3 = nn.ConvTranspose2d(128, 128, 2, stride=2); s.conv3 = DoubleConv(256, 64)
            s.up4 = nn.ConvTranspose2d(64, 64, 2, stride=2);   s.conv4 = DoubleConv(128, 64)
            s.outc = nn.Conv2d(64, n_cls, 1)
        def forward(s, x):
            x1 = s.inc(x); x2 = s.down1(x1); x3 = s.down2(x2); x4 = s.down3(x3); x5 = s.down4(x4)
            x = s.up1(x5); x = s.conv1(torch.cat([x, x4], 1))
            x = s.up2(x);  x = s.conv2(torch.cat([x, x3], 1))
            x = s.up3(x);  x = s.conv3(torch.cat([x, x2], 1))
            x = s.up4(x);  x = s.conv4(torch.cat([x, x1], 1))
            return s.outc(x)
    return UNet()


class _LegacyUNet:
    def __init__(self, model_path, device):
        torch = _torch()
        self.device = torch.device(device)
        self.model = _build_unet().to(self.device)
        state = torch.load(model_path, map_location=self.device, weights_only=True)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        self.model.load_state_dict(state, strict=False)
        self.model.eval()

    def bbox(self, bgr):
        torch = _torch()
        h0, w0 = bgr.shape[:2]
        resized = cv2.resize(bgr, IMG_SIZE, interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(resized.transpose(2, 0, 1)).float()[None] / 255.0
        with torch.inference_mode():
            mask = torch.argmax(self.model(t.to(self.device)), 1)[0].cpu().numpy().astype(np.uint8)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        sx, sy = w0 / IMG_SIZE[0], h0 / IMG_SIZE[1]
        return int(x * sx), int(y * sy), int((x + w) * sx), int((y + h) * sy)

    def full_mask(self, bgr):
        """Full-resolution binary folio mask (uint8 0/1) from the legacy U-Net."""
        torch = _torch()
        h0, w0 = bgr.shape[:2]
        resized = cv2.resize(bgr, IMG_SIZE, interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(resized.transpose(2, 0, 1)).float()[None] / 255.0
        with torch.inference_mode():
            m = torch.argmax(self.model(t.to(self.device)), 1)[0].cpu().numpy().astype(np.uint8)
        return cv2.resize(m, (w0, h0), interpolation=cv2.INTER_NEAREST)


class LegacyPipeline:
    """Reproduces the legacy split/orient/crop behaviour for benchmarking."""

    def __init__(self, weights_dir: str, device: str = "cuda"):
        wd = Path(weights_dir)
        self.count = _ResNet18Binary(wd / "folio_count_classifier.pth",
                                     ["one_folio", "two_folios"], device)
        ori = wd / "upside_down.pth"
        if not ori.exists():
            ori = wd / "folio_upside_down.pth"
        self.orient = _ResNet18Binary(ori, ["right_side", "upside_down"], device)
        self.seg = _LegacyUNet(wd / "unet_folio_split.pth", device)

    def _fix_global(self, bgr, two):
        h, w = bgr.shape[:2]
        if two and h > w:
            return np.ascontiguousarray(np.rot90(bgr, 3))
        if not two and w > h:
            return np.ascontiguousarray(np.rot90(bgr, 3))
        return bgr

    def process(self, bgr) -> dict:
        """Returns {count, crops:[bgr...], labels:[...], orientations:[...]} ."""
        label, cconf = self.count.predict(bgr)
        two = label == "two_folios"
        page = self._fix_global(bgr, two)
        h, w = page.shape[:2]
        regions = []
        if two:
            mid = w // 2; ov = int(0.05 * w)
            left = page[:, :mid + ov]; right = page[:, mid - ov:]
            for sfx, crop, xoff in [("A", left, 0), ("B", right, mid - ov)]:
                bb = self.seg.bbox(crop)
                if bb and (bb[2]-bb[0])*(bb[3]-bb[1]) > 0.1*w*h:
                    regions.append((sfx, crop[bb[1]:bb[3], bb[0]:bb[2]]))
        else:
            bb = self.seg.bbox(page)
            if bb and (bb[2]-bb[0])*(bb[3]-bb[1]) > 0.1*w*h:
                regions.append(("", page[bb[1]:bb[3], bb[0]:bb[2]]))
            else:
                regions.append(("", page))
        crops, labels, oris = [], [], []
        for sfx, crop in regions:
            olab, _ = self.orient.predict(crop)
            if olab == "upside_down":
                crop = np.ascontiguousarray(np.rot90(crop, 2))
            crops.append(crop); labels.append(sfx); oris.append(olab)
        return {"count": label, "count_conf": cconf, "crops": crops,
                "labels": labels, "orientations": oris}
