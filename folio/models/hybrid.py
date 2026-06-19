"""Recommended PRODUCTION configuration: reuse the trained legacy models where
they are genuinely good, inside our better split/orient/crop logic, and call
SAM 2.1 only on the hard cases. This is higher quality than the legacy pipeline
AND far cheaper than running SAM 2.1 on all 750k images.

Components:
  * UNetFolioSegmenter  - legacy unet_folio_split.pth produces the page mask
    (cheap, trained); our seam-energy spine split + oriented crop replace the
    legacy W//2 split. Optional SAM 2.1 fallback when the U-Net mask is weak.
  * LegacyCountAdapter  - legacy folio_count_classifier.pth (one/two), with our
    aspect/shadow cue as a tie-breaker.
  * LegacyOrientAdapter - legacy upside_down.pth for 0 vs 180, plus a landscape
    rule for the rare 90/270 case.

All torch use is lazy/inside the legacy wrappers, so the package still imports
without torch.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import numpy as np
import cv2

from ..schemas import PageBox, PageCount
from ..stages import foreground
from .classical import ClassicalSegmenter
from .legacy import _ResNet18Binary, _LegacyUNet


class UNetFolioSegmenter:
    """detect() via our robust foreground/aspect boxes; segment() via the
    trained legacy U-Net mask (tighter than classical). Optional SAM fallback."""

    def __init__(self, weights_dir: str, cfg=None, device: str = "cuda",
                 sam_fallback=None, min_mask_frac: float = 0.12):
        self.cfg = cfg
        self._boxes = ClassicalSegmenter(cfg)
        self._unet = _LegacyUNet(Path(weights_dir) / "unet_folio_split.pth", device)
        self._sam = sam_fallback          # e.g. PageSegmenter; used only if weak
        self._min_mask_frac = min_mask_frac
        self._cache = {}

    def detect(self, image: np.ndarray, max_pages: int = 2) -> List[PageBox]:
        return self._boxes.detect(image, max_pages)

    def _mask(self, image):
        key = (image.shape, int(image[::97, ::97].sum()))
        if key not in self._cache:
            if len(self._cache) > 6:
                self._cache.clear()
            self._cache[key] = self._unet.full_mask(image)
        return self._cache[key]

    def segment(self, image: np.ndarray, boxes: List[PageBox]) -> List[np.ndarray]:
        mask = self._mask(image)
        h, w = mask.shape
        # Restrict the page mask to each box's horizontal TERRITORY (bounded by
        # the midpoints between adjacent box centres), full height — NOT the tight
        # detection box. A box that under-covers the page (e.g. foreground detection
        # stopping short of a faint outer edge) would otherwise clip real text; the
        # territory keeps the full page mask, and the downstream seam split refines
        # the inner (gutter) edge.
        centres = sorted((int((b.x1 + b.x2) // 2), i) for i, b in enumerate(boxes))
        terr = {}
        for rank, (c, i) in enumerate(centres):
            lo = 0 if rank == 0 else (centres[rank - 1][0] + c) // 2
            hi = w if rank == len(centres) - 1 else (c + centres[rank + 1][0]) // 2
            terr[i] = (lo, hi)
        out = []
        for i, b in enumerate(boxes):
            lo, hi = terr[i]
            sub = np.zeros((h, w), np.uint8)
            sub[:, lo:hi] = mask[:, lo:hi]
            frac = sub.sum() / float(max((hi - lo) * h, 1))
            if frac < self._min_mask_frac and self._sam is not None:
                # U-Net unsure on this page -> escalate to SAM 2.1
                try:
                    sub = self._sam.segment(image, [b])[0]
                except Exception:
                    pass
            out.append(sub)
        return out


class LegacyCountAdapter:
    """one/two folio from the legacy ResNet, tie-broken by our aspect/shadow."""

    def __init__(self, weights_dir: str, device: str = "cuda"):
        wd = Path(weights_dir)
        self._m = _ResNet18Binary(wd / "folio_count_classifier.pth",
                                  ["one_folio", "two_folios"], device)

    def predict(self, image: np.ndarray) -> Tuple[PageCount, float]:
        label, conf = self._m.predict(image)
        # tie-break low-confidence calls with the geometric signal
        if conf < 0.75:
            _, _, shadow = foreground.detect_pages(image)
            h, w = image.shape[:2]
            geo_two = (w > 1.15 * h) or (w > 1.05 * h and shadow < 0.92)
            label = "two_folios" if geo_two else "one_folio"
        return (PageCount.TWO if label == "two_folios" else PageCount.ONE), conf


class LegacyOrientAdapter:
    """0-vs-180 from the legacy upside-down ResNet; landscape adds 90/270."""

    def __init__(self, weights_dir: str, device: str = "cuda"):
        wd = Path(weights_dir)
        ori = wd / "upside_down.pth"
        if not ori.exists():
            ori = wd / "folio_upside_down.pth"
        self._m = _ResNet18Binary(ori, ["right_side", "upside_down"], device)

    def predict_probs(self, image: np.ndarray) -> np.ndarray:
        label, conf = self._m.predict(image)
        p = np.zeros(4, np.float64)
        if label == "upside_down":
            p[2] = conf; p[0] = 1 - conf
        else:
            p[0] = conf; p[2] = 1 - conf
        h, w = image.shape[:2]
        if w > 1.15 * h:                 # sideways capture -> allow quarter turns
            p = np.array([0.1, 0.4, 0.1, 0.4])
        return p / p.sum()


def build_hybrid_pipeline(cfg, weights_dir: str, device: str = "cuda",
                          use_sam_fallback: bool = False):
    """FolioPipeline wired to the recommended hybrid configuration."""
    from ..pipeline import FolioPipeline
    sam = None
    if use_sam_fallback:
        from .segmentation import PageSegmenter
        sam = PageSegmenter(cfg.model)
    seg = UNetFolioSegmenter(weights_dir, cfg.model, device, sam_fallback=sam)
    return FolioPipeline(cfg, segmenter=seg,
                         counter=LegacyCountAdapter(weights_dir, device),
                         orienter=LegacyOrientAdapter(weights_dir, device))
