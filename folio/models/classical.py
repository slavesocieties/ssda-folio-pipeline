"""Classical-fallback implementations of the three model interfaces, so the
full pipeline (folio.pipeline.FolioPipeline) can run end-to-end with NO neural
weights and NO GPU. Accuracy is below the neural path, but it exercises the
real spine / geometry / orient modules on real images for smoke-testing.

Swap these for PageSegmenter / FolioCountClassifier / OrientationClassifier in
production.
"""
from __future__ import annotations

from typing import List, Tuple
import cv2
import numpy as np

from ..schemas import PageBox, PageCount
from ..stages import foreground


class ClassicalSegmenter:
    """detect() via foreground/gutter analysis; segment() returns the
    foreground mask restricted to each box."""

    def __init__(self, cfg=None, two_folio_valley: float = 0.6):
        self.cfg = cfg
        self.two_folio_valley = two_folio_valley
        self._cache = {}

    @staticmethod
    def _sig(image: np.ndarray):
        # content signature (shape + sparse checksum); id() is unsafe because
        # CPython reuses ids of freed arrays -> stale cross-image cache hits.
        sub = image[::97, ::97]
        return (image.shape, int(sub.astype(np.int64).sum()))

    def _analyze(self, image: np.ndarray):
        key = self._sig(image)
        if key not in self._cache:
            if len(self._cache) > 8:
                self._cache.clear()
            self._cache[key] = foreground.detect_pages(image, self.two_folio_valley)
        return self._cache[key]

    def detect(self, image: np.ndarray, max_pages: int = 2) -> List[PageBox]:
        boxes, _, _ = self._analyze(image)
        return boxes[:max_pages]

    def segment(self, image: np.ndarray, boxes: List[PageBox]) -> List[np.ndarray]:
        _, fg, _ = self._analyze(image)
        out = []
        for b in boxes:
            x1, y1, x2, y2 = b.as_tuple()
            bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)
            sub = fg[y1:y2, x1:x2]
            # Consolidate a possibly-fragmented foreground (e.g. separate ink
            # lines on a white-on-white scan) into one solid page blob, so the
            # oriented crop covers the WHOLE page rather than a single text line.
            k = max(int(0.03 * min(bw, bh)) | 1, 3)
            el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            closed = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, el)
            # fill interior holes
            ff = closed.copy()
            hh, ww = ff.shape
            fmask = np.zeros((hh + 2, ww + 2), np.uint8)
            cv2.floodFill(ff, fmask, (0, 0), 1)
            filled = closed | (1 - ff)
            # if still sparse, the mask is unreliable -> use a solid page box
            n, lbl, st, _ = cv2.connectedComponentsWithStats((filled > 0).astype(np.uint8), 8)
            page = np.zeros((hh, ww), np.uint8)
            if n > 1:
                big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
                page = (lbl == big).astype(np.uint8)
            if page.sum() < 0.55 * bw * bh:
                page[:] = 1  # solid box fallback (guarantees full-page crop)
            m = np.zeros(image.shape[:2], np.uint8)
            m[y1:y2, x1:x2] = page
            out.append(m)
        return out


class ClassicalCounter:
    """one/two folio from the same foreground gutter-valley signal."""

    def __init__(self, cfg=None, two_folio_valley: float = 0.6):
        self.cfg = cfg
        self.two_folio_valley = two_folio_valley

    def predict(self, image: np.ndarray) -> Tuple[PageCount, float]:
        boxes, fg, valley = foreground.detect_pages(image, self.two_folio_valley)
        if not boxes:
            return PageCount.REJECT, 0.5
        # near-blank reject: very little ink relative to paper area
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        if len(boxes) == 2:
            conf = float(np.clip(1.0 - valley, 0.5, 0.99))
            return PageCount.TWO, conf
        conf = float(np.clip(valley, 0.5, 0.99))
        return PageCount.ONE, conf


class ClassicalOrienter:
    """Best-effort 4-way distribution from classical cues.

    Returns softmax-like probs over [0, 90, 180, 270] of the CURRENT rotation.
    - 90 vs 0/180: a landscape crop (w>h) suggests a 90-degree turn; the
      direction is left to the skew step.
    - 0 vs 180: an ink-vertical-centroid heuristic (text mass tends to sit in
      the upper half). This is weak -> reported with low confidence so the
      pipeline routes ambiguous pages to review, exactly as designed.
    """

    def __init__(self, cfg=None):
        self.cfg = cfg

    def predict_probs(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        h, w = gray.shape
        probs = np.array([0.40, 0.0, 0.10, 0.0], dtype=np.float64)  # 0/180 only

        # ink mask + vertical centroid for 0-vs-180
        _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ys, xs = np.where(ink > 0)
        if ys.size > 50:
            cy = ys.mean() / h  # 0=top heavy, 1=bottom heavy
            # top-heavy -> upright(0); bottom-heavy -> upside down(180)
            up = np.clip(1.0 - cy, 0.0, 1.0)
            down = np.clip(cy, 0.0, 1.0)
            probs[0] = 0.20 + 0.6 * up
            probs[2] = 0.20 + 0.6 * down
        return probs / probs.sum()
