"""Contrast-robust content (ink / handwriting) detection.

Archival folios carry faint iron-gall ink, pencil, and light pressed text on
aged, textured, unevenly-lit paper. A single global or mean-C threshold misses
those low-contrast strokes, so a page of real (but light) handwriting reads as
"blank" — which then mis-routes the orientation gate and, if used for cropping,
would let us trim real content as if it were background.

`content_mask` combines several complementary contrast tools and unions their
responses, so a stroke that is invisible to one method is caught by another:

  1. CLAHE  — local histogram equalization flattens uneven lighting and lifts
     faint ink off the paper before any thresholding.
  2. Sauvola local threshold (integral-image, no extra deps) — keys on local
     mean/variance, the standard for low-contrast handwritten document
     binarization; far better than mean-C on light ink.
  3. Black-hat morphology — isolates dark strokes thinner than the structuring
     element against a brighter background, rescuing hairline pen strokes that
     have almost no global contrast.

A contrast floor and small-speck removal keep bare paper *texture* from reading
as ink, so a genuinely blank page still measures as blank. Pure NumPy + OpenCV.
"""
from __future__ import annotations

from typing import Tuple
import cv2
import numpy as np


def _odd(n: int) -> int:
    n = int(n)
    return n if n % 2 == 1 else n + 1


def _sauvola(gray: np.ndarray, window: int, k: float = 0.15,
             R: float = 128.0) -> np.ndarray:
    """Sauvola binarization via integral images. Returns uint8 0/255 ink mask
    (ink = dark strokes on lighter paper). threshold = mean*(1 + k*(std/R - 1))."""
    win = _odd(max(3, window))
    g = gray.astype(np.float64)
    # local mean and mean-of-squares over a win x win box via integral images
    mean = cv2.boxFilter(g, ddepth=-1, ksize=(win, win),
                         borderType=cv2.BORDER_REPLICATE)
    sqmean = cv2.boxFilter(g * g, ddepth=-1, ksize=(win, win),
                           borderType=cv2.BORDER_REPLICATE)
    var = np.clip(sqmean - mean * mean, 0, None)
    std = np.sqrt(var)
    thresh = mean * (1.0 + k * (std / R - 1.0))
    return ((g < thresh) & (std > 2.0)).astype(np.uint8) * 255


def _blackhat_ink(gray: np.ndarray, ksize: int) -> np.ndarray:
    """Dark thin strokes via black-hat (closing - image). Thresholded with a
    floor so paper texture doesn't pass."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(ksize), _odd(ksize)))
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)
    if bh.max() <= 0:
        return np.zeros_like(gray)
    # Otsu on the positive response, but never below a small absolute floor so a
    # near-uniform (blank) page can't have its faint noise promoted to "ink".
    t, _ = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = max(float(t), 12.0)
    return (bh >= t).astype(np.uint8) * 255


def content_mask(image: np.ndarray, *, mask: np.ndarray | None = None
                 ) -> Tuple[np.ndarray, float]:
    """Robust binary content mask (uint8 0/255) + coverage fraction.

    ``mask`` (optional, 0/1 or 0/255) restricts measurement to the page region
    so scanner background is never counted as content. ``coverage`` is the
    content-pixel fraction *within that region* (or the whole image if no mask).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = gray.shape[:2]
    m = min(h, w)
    if m < 8:
        # too small to threshold meaningfully
        cov = 0.0
        return np.zeros((h, w), np.uint8), cov

    # 1. illumination-normalize + lift faint ink
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)

    # 2. Sauvola local threshold (window ~ several stroke widths, scaled to size)
    win = int(np.clip(0.03 * m, 15, 51))
    ink = _sauvola(clahe, win, k=0.15)

    # 3. rescue faint hairline strokes the local threshold missed
    bh = _blackhat_ink(clahe, ksize=int(np.clip(0.02 * m, 9, 31)))
    ink = cv2.bitwise_or(ink, bh)

    # clean: drop isolated single-pixel speckle (paper grain), keep thin strokes
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    # remove tiny connected components (noise), keep anything stroke-sized
    ink = _drop_specks(ink, min_area=max(6, int(2e-6 * h * w)))

    if mask is not None:
        region = (mask > 0)
        ink[~region] = 0
        denom = float(region.sum()) or float(h * w)
    else:
        denom = float(h * w)
    coverage = float((ink > 0).sum()) / denom
    return ink, coverage


def _drop_specks(mask: np.ndarray, min_area: int) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(mask)
    # Vectorized: build a per-label keep flag, then index it by the label image
    # in ONE pass. The old per-component `labels == i` loop was O(components x
    # pixels) and dominated runtime on speckled full-res scans (~45 s/image).
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False  # label 0 is background
    return (keep[labels].astype(np.uint8) * 255)


def trim_background_border(crop: np.ndarray, pad_frac: float = 0.01
                           ) -> Tuple[int, int, int, int]:
    """Remove uninformative *background* border from a finished page crop and
    return the kept box (x0, y0, x1, y1).

    Targets two kinds of non-information pixels the supervisor flagged: black
    warp padding (when the oriented quad ran past the image edge) and a uniform
    scanner-background strip left by a slightly loose page mask. It eats border
    rows/cols from the outside in *only while* they are both (a) close to the
    background colour estimated from the crop's corners and (b) free of detected
    content — so it stops the instant it reaches paper or ink and never clips a
    real pixel. Paper margins (different colour from background) are kept, so the
    folio itself is never trimmed.
    """
    h, w = crop.shape[:2]
    if min(h, w) < 16:
        return 0, 0, w, h
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    ink, _ = content_mask(crop)

    # Reference = the PAGE colour, taken from the crop centre (which is folio, not
    # background). A border row/col counts as background only if it differs from
    # the page colour AND carries no content. This keeps uniform paper margins
    # (same colour as the page) and trims only genuinely non-paper border: black
    # warp padding, or a differently-coloured scanner-bed strip.
    cy0, cy1 = int(0.3 * h), int(0.7 * h)
    cx0, cx1 = int(0.3 * w), int(0.7 * w)
    page_val = float(np.median(gray[cy0:cy1, cx0:cx1]))
    tol = 40.0

    def is_bg_row(r):
        return abs(float(np.median(gray[r])) - page_val) > tol and not ink[r].any()

    def is_bg_col(cc):
        return abs(float(np.median(gray[:, cc])) - page_val) > tol and not ink[:, cc].any()

    y0, y1, x0, x1 = 0, h, 0, w
    while y0 < y1 - 16 and is_bg_row(y0):
        y0 += 1
    while y1 > y0 + 16 and is_bg_row(y1 - 1):
        y1 -= 1
    while x0 < x1 - 16 and is_bg_col(x0):
        x0 += 1
    while x1 > x0 + 16 and is_bg_col(x1 - 1):
        x1 -= 1

    pad = int(pad_frac * float(np.hypot(h, w)))
    x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
    x1 = min(w, x1 + pad); y1 = min(h, y1 + pad)
    return x0, y0, x1, y1


def paper_box(crop: np.ndarray, margin_frac: float = 0.012
              ) -> Tuple[int, int, int, int] | None:
    """Tight box around the folio PAPER, excluding dark non-information regions
    that the page mask let through — book binding, scanner bed, fingers/clamps.

    The folio paper is the bright region; binding, background and hardware are
    dark. An Otsu split on a blurred luminance isolates the largest bright blob
    (the paper), then the box is *expanded* to enclose any detected ink that sits
    outside it, so marginalia/text is never clipped. Returns None (keep original)
    when no confident paper region is found, or when paper already fills the crop.
    """
    h, w = crop.shape[:2]
    if min(h, w) < 32:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    blur = cv2.GaussianBlur(gray, (0, 0), max(1.5, 0.004 * min(h, w)))
    t, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = (blur > t).astype(np.uint8)
    # close text holes / small gaps so the page becomes one blob
    k = max(3, int(0.02 * min(h, w)) | 1)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    if n <= 1:
        return None
    i = 1 + int(np.argmax([stats[j, cv2.CC_STAT_AREA] for j in range(1, n)]))
    area = stats[i, cv2.CC_STAT_AREA]
    if area < 0.20 * h * w:
        return None  # no dominant bright page -> don't risk it
    x = stats[i, cv2.CC_STAT_LEFT]; y = stats[i, cv2.CC_STAT_TOP]
    bw = stats[i, cv2.CC_STAT_WIDTH]; bh = stats[i, cv2.CC_STAT_HEIGHT]
    x0, y0, x1, y1 = x, y, x + bw, y + bh

    # never clip real ink: expand to include content pixels, but only those ON
    # the paper (a dilated copy of the page blob). This keeps outer-margin text /
    # marginalia while ignoring dark-on-dark "ink" the detector finds in the
    # binding or scanner bed (which would otherwise re-expand to the full frame).
    paper = (labels == i).astype(np.uint8)
    grow = max(3, int(0.04 * min(h, w)) | 1)
    paper = cv2.dilate(paper, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow)))
    ink, _ = content_mask(crop)
    ink = cv2.bitwise_and(ink, ink, mask=paper)
    ys = np.where(ink.any(axis=1))[0]
    xs = np.where(ink.any(axis=0))[0]
    if xs.size and ys.size:
        x0 = min(x0, int(xs.min())); x1 = max(x1, int(xs.max()) + 1)
        y0 = min(y0, int(ys.min())); y1 = max(y1, int(ys.max()) + 1)

    # if it barely shrinks the crop, treat as already-tight (no-op)
    if (x1 - x0) >= 0.97 * w and (y1 - y0) >= 0.97 * h:
        return None
    m = int(margin_frac * float(np.hypot(h, w)))
    return (max(0, x0 - m), max(0, y0 - m), min(w, x1 + m), min(h, y1 + m))


def enhance_faint(image: np.ndarray) -> np.ndarray:
    """Contrast-normalized derivative for faint/light-ink pages: CLAHE on the
    luminance channel so faint handwriting becomes legible without crushing the
    paper. Returns a BGR image the same size as the input. Used for the optional
    `_enhanced` output that helps downstream transcription read light ink."""
    if image.ndim == 2:
        return cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(image)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
