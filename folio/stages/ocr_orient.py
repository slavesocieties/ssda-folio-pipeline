"""Stage 5d — OCR orientation review-rescue.

An INDEPENDENT third signal for the one thing the 4-way head and the legacy
up/down net systematically get wrong: the 180-degree flip. The head is ~99%
right on the AXIS (portrait vs landscape) but near coin-flip on up-vs-down; the
legacy ResNet shares that visual blind spot (both were confidently wrong on the
hard laminated/stained scan 6517-0105). OCR is a genuinely different modality —
it reads text legibility, not visual gestalt — so its failure mode is
decorrelated: an upright page yields more, higher-confidence characters than the
same page upside-down.

Measured on 200 labelled folios x 4 rotations (789 well-posed flip decisions):
OCR's flip verdict is 87% correct overall, and its relative-confidence margin is
well calibrated — at rel-margin >= 0.20 it is **98% precise** (covering ~54% of
cases). That calibration is what makes it safe to (a) correct a wrong flip and
(b) clear the review flag: we act only when the independent signal is strong.

Scope (deliberately narrow, for cost and safety):
  * Runs ONLY on folios already flagged ``low_orientation_conf`` — i.e. the small
    fraction the head itself is unsure about — so throughput on the 750k corpus
    is unaffected (OCR never touches the confident majority).
  * NEVER overrides a confident head. In production framing the head is ~95%+ and
    its rare errors are low-confidence, which is exactly this flagged set.
  * Acts only above ``rescue_margin``; below it, the folio stays flagged for a
    human. A weak/ambiguous OCR reading is treated as no signal, not a tie-break.

Graceful: if EasyOCR (an optional dependency) is unavailable, the verifier is
simply never attached and the pipeline behaves exactly as before.
"""
from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
import cv2


class OCRUpDownVerifier:
    """Scores a crop's text legibility upright vs 180-flipped to decide up/down.

    ``reader`` is injected in tests (any object with ``.readtext(bgr)`` returning
    ``[(box, text, conf), ...]``); in production it is a lazily-built EasyOCR
    Reader so importing the package never requires easyocr or torch.
    """

    def __init__(self, cfg=None, reader=None, long_side: int = 1400,
                 langs=("es",), device: str = "cuda"):
        self.cfg = cfg
        self._reader = reader
        self._long_side = long_side
        self._langs = list(langs)
        self._gpu = device == "cuda"
        self._tried = reader is not None

    # -- lazy backend -------------------------------------------------------
    def _get_reader(self):
        if self._reader is not None:
            return self._reader
        if self._tried:
            return None
        self._tried = True
        try:
            import easyocr
            self._reader = easyocr.Reader(self._langs, gpu=self._gpu, verbose=False)
        except Exception:
            self._reader = None
        return self._reader

    @property
    def available(self) -> bool:
        return self._get_reader() is not None

    # -- scoring ------------------------------------------------------------
    def _weighted_score(self, bgr: np.ndarray) -> float:
        """Sum of (recognition confidence x characters) over detected text.

        Rewards both MORE recognised text and HIGHER-confidence text, so an
        upright page (legible) outscores its upside-down self (garbled). A
        downscale keeps it fast — the relative comparison, not absolute OCR
        quality, is all that matters here.
        """
        reader = self._get_reader()
        if reader is None:
            return 0.0
        h, w = bgr.shape[:2]
        s = self._long_side / float(max(h, w))
        view = cv2.resize(bgr, (max(int(w * s), 1), max(int(h * s), 1)),
                          interpolation=cv2.INTER_AREA) if s < 1.0 else bgr
        try:
            res = reader.readtext(view)
        except Exception:
            return 0.0
        return float(sum(r[2] * len(r[1]) for r in res))

    def flip_verdict(self, upright_crop: np.ndarray) -> Tuple[bool, float]:
        """Should ``upright_crop`` be rotated 180 to become truly upright?

        Returns ``(should_flip, rel_margin)`` where
        ``rel_margin = |s_up - s_flip| / (s_up + s_flip)`` in [0, 1]. The caller
        acts only when ``rel_margin`` clears its gate; a small margin means OCR
        cannot tell and the folio should stay flagged for review.
        """
        s_up = self._weighted_score(upright_crop)
        s_flip = self._weighted_score(np.ascontiguousarray(np.rot90(upright_crop, 2)))
        denom = s_up + s_flip
        if denom <= 1e-6:
            return False, 0.0            # no text either way -> no signal
        rel_margin = abs(s_up - s_flip) / denom
        return (s_flip > s_up), rel_margin
