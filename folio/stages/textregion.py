"""Learned text-region detection for tight cropping.

Classical morphology can't reliably isolate the text block on archival folios
(scattered stains/show-through stretch the box to the page edges; aggressive
denoising fragments the handwriting). A text *detector* trained on large corpora
localizes the writing directly, so the union of its detections — plus a margin —
gives the supervisor's "tight" crop (~0.75 meaningful-pixel ratio) without the
fragility.

We use EasyOCR's CRAFT detector in *detection only* mode (no recognition): it is
language-agnostic and finds character/word regions, which we union into one box.
Everything is lazy-imported and optional — if EasyOCR or its model isn't present
the detector reports unavailable and the pipeline keeps its looser, never-clip
crop. Detection runs on GPU when torch CUDA is available.
"""
from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np

_READER = None
_TRIED = False


def _get_reader():
    """Lazily build (and cache) the EasyOCR reader; None if unavailable."""
    global _READER, _TRIED
    if _TRIED:
        return _READER
    _TRIED = True
    try:
        import torch
        import easyocr
        gpu = bool(torch.cuda.is_available())
        # verbose=False suppresses EasyOCR's download progress bar, whose Unicode
        # block glyph crashes on a cp1252 (Windows) console.
        _READER = easyocr.Reader(["en"], gpu=gpu, verbose=False)
    except Exception:
        _READER = None
    return _READER


def available() -> bool:
    return _get_reader() is not None


def detect_boxes(image: np.ndarray, low_text: float = 0.3,
                 text_threshold: float = 0.5) -> List[Tuple[int, int, int, int]]:
    """Axis-aligned text boxes (x0, y0, x1, y1). Empty list if unavailable.

    Lower ``low_text`` widens regions to capture faint strokes; the defaults are
    tuned slightly permissive so light handwriting is still localized."""
    reader = _get_reader()
    if reader is None:
        return []
    horizontal, free = reader.detect(
        image, low_text=low_text, text_threshold=text_threshold,
        link_threshold=0.4, add_margin=0.1)
    boxes: List[Tuple[int, int, int, int]] = []
    for grp in (horizontal or []):
        for b in grp:
            x0, x1, y0, y1 = b[0], b[1], b[2], b[3]
            boxes.append((int(x0), int(y0), int(x1), int(y1)))
    for grp in (free or []):
        for poly in grp:
            pts = np.array(poly).reshape(-1, 2)
            boxes.append((int(pts[:, 0].min()), int(pts[:, 1].min()),
                          int(pts[:, 0].max()), int(pts[:, 1].max())))
    return boxes


def text_crop_box(image: np.ndarray, margin_frac: float = 0.02,
                  trim_outliers: bool = False, min_keep: float = 0.45
                  ) -> Optional[Tuple[int, int, int, int]]:
    """Tight crop box = union of detected text regions + margin, in image coords.

    Safety first — this must never clip real content:
      * ``trim_outliers`` is OFF by default. Dropping a detection that is far from
        the main mass risks clipping a real but isolated line (a signature, a
        bottom-margin note), so by default we union *every* detection: at worst a
        slightly looser crop, never a clip.
      * ``min_keep`` rejects an implausibly small box. On faint pages the detector
        may fire on only a band of the page; cropping to that band would clip the
        rest. If the proposed box keeps < ``min_keep`` of either dimension we
        return None, and the caller keeps its looser (paper) crop.
    Returns None when no text is found, or the box is too small to trust."""
    boxes = detect_boxes(image)
    if not boxes:
        return None
    h, w = image.shape[:2]
    arr = np.array(boxes, dtype=np.float32)  # (n,4): x0,y0,x1,y1
    if trim_outliers and len(arr) >= 4:
        cx = (arr[:, 0] + arr[:, 2]) / 2.0
        cy = (arr[:, 1] + arr[:, 3]) / 2.0
        mx, my = np.median(cx), np.median(cy)
        dist = np.hypot(cx - mx, cy - my)
        keep = dist <= (np.median(dist) + 3.0 * (np.median(np.abs(dist - np.median(dist))) + 1e-6))
        if keep.sum() >= 1:
            arr = arr[keep]
    x0, y0 = float(arr[:, 0].min()), float(arr[:, 1].min())
    x1, y1 = float(arr[:, 2].max()), float(arr[:, 3].max())
    m = margin_frac * float(np.hypot(h, w))
    bx0, by0 = max(0, int(x0 - m)), max(0, int(y0 - m))
    bx1, by1 = min(w, int(x1 + m)), min(h, int(y1 + m))
    # guard against under-detection: a box that keeps too little of the page is
    # almost certainly missed (faint) text, not a tight crop -> keep looser crop.
    if (bx1 - bx0) < min_keep * w or (by1 - by0) < min_keep * h:
        return None
    return (bx0, by0, bx1, by1)
