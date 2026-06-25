"""Learned folio (page) segmenter — inference wrapper.

Loads the TorchScript U-Net trained by ``folio.training.seg`` and returns a
binary PAGE mask (the single-folio region) at the input image's resolution. This
is the background-agnostic page boundary the classical detector couldn't get on
light-on-light scans; the pipeline uses it as the crop mask for a precise,
full-folio crop. Lazy/torch-optional and CPU/GPU aware.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import cv2

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


class FolioSegmenter:
    def __init__(self, ts_weights: str, size: int = 512, device: Optional[str] = None):
        import torch
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.size = size
        self.model = torch.jit.load(ts_weights, map_location=self.device).eval()

    def page_prob(self, image: np.ndarray) -> np.ndarray:
        """Per-pixel page probability at the input resolution (float32 0..1)."""
        h, w = image.shape[:2]
        im = cv2.resize(image, (self.size, self.size))
        x = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = (x - _MEAN) / _STD
        t = self.torch.from_numpy(x.transpose(2, 0, 1))[None].to(self.device)
        with self.torch.no_grad():
            prob = self.torch.sigmoid(self.model(t))[0, 0].float().cpu().numpy()
        return cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)

    def page_mask(self, image: np.ndarray, thresh: float = 0.5) -> np.ndarray:
        """Binary page mask (uint8 0/1) at the input resolution, largest component."""
        prob = self.page_prob(image)
        m = (prob > thresh).astype(np.uint8)
        n, lbl, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if n <= 1:
            return m
        big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        return (lbl == big).astype(np.uint8)
