"""Orientation backend that uses Tesseract OSD (a real trained orientation/
script detector) as the primary signal, with the classical ink-centroid cue as
a fallback when OSD is unavailable or low-confidence.

Returns a softmax-like distribution over the CURRENT rotation [0, 90, 180, 270]
so it is a drop-in for OrientationClassifier / ClassicalOrienter.

Note on handwriting: Tesseract OSD is trained on printed text, so confidence on
manuscript hands is modest; we threshold on its confidence and defer to the
classical cue + review routing below it. The production-grade fix is the
fine-tuned 4-way head in folio.models.classifiers.
"""
from __future__ import annotations

import numpy as np
import cv2

from .classical import ClassicalOrienter

try:
    import pytesseract
    from pytesseract import Output
    _HAS_TESS = True
except Exception:  # pragma: no cover
    _HAS_TESS = False


# OSD 'rotate' R = degrees clockwise to CORRECT the image.
# current rotation C = (360 - R) % 360 ; map C -> index in [0,90,180,270]
_R_TO_CURRENT_IDX = {0: 0, 90: 3, 180: 2, 270: 1}


class OSDOrienter:
    def __init__(self, cfg=None, min_conf: float = 2.0, max_side: int = 1600):
        self.cfg = cfg
        self.min_conf = min_conf
        self.max_side = max_side
        self._fallback = ClassicalOrienter(cfg)

    def _osd_probs(self, image: np.ndarray):
        if not _HAS_TESS:
            return None
        img = image
        h, w = img.shape[:2]
        s = self.max_side / max(h, w)
        if s < 1.0:
            img = cv2.resize(img, (int(w * s), int(h * s)))
        try:
            osd = pytesseract.image_to_osd(img, output_type=Output.DICT)
        except Exception:
            return None
        rotate = int(osd.get("rotate", 0)) % 360
        conf = float(osd.get("orientation_conf", 0.0))
        if rotate not in _R_TO_CURRENT_IDX or conf < self.min_conf:
            return None
        idx = _R_TO_CURRENT_IDX[rotate]
        # confidence -> peakiness; cap so a single source never reads as certain
        peak = float(np.clip(0.5 + conf / 20.0, 0.5, 0.9))
        probs = np.full(4, (1.0 - peak) / 3.0)
        probs[idx] = peak
        return probs / probs.sum()

    def predict_probs(self, image: np.ndarray) -> np.ndarray:
        p = self._osd_probs(image)
        if p is None:
            p = self._fallback.predict_probs(image)
        # Portrait scans are realistically only 0 or 180; 90/270 from OSD on
        # handwriting are almost always spurious, so suppress them. (Landscape
        # pages keep the full 4-way set.)
        h, w = image.shape[:2]
        if h >= w:
            p = p.copy(); p[1] = 0.0; p[3] = 0.0
            ssum = p.sum()
            p = p / ssum if ssum > 1e-9 else np.array([0.6, 0.0, 0.4, 0.0])
        return p
