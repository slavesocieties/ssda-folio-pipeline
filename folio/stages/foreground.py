"""Classical, background-agnostic foreground/page detection.

This is the dependency-light stand-in for the RT-DETR detector used in the
classical-fallback mode (no GPU / no weights). It estimates the background
colour from the image border, builds a foreground (paper) mask that works on
dark, coloured, or light backgrounds, then proposes 1 or 2 page boxes by
testing for a central gutter valley.

Used only by folio.models.classical; the production path uses the neural
detector + SAM 2.1 instead. Pure NumPy + OpenCV, fully testable.
"""
from __future__ import annotations

from typing import List, Tuple
import cv2
import numpy as np

from ..schemas import PageBox


def _largest_components(mask: np.ndarray, min_area_frac: float, top_k: int = 4):
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    h, w = mask.shape
    out = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area_frac * h * w:
            out.append((area, stats[i], cents[i]))
    out.sort(key=lambda t: -t[0])
    return out[:top_k]


def foreground_mask(image: np.ndarray) -> np.ndarray:
    """Binary mask (uint8 0/1) of the paper region, robust to background colour.

    Strategy: estimate background from a border frame, threshold the per-pixel
    colour distance from it; if that fails (paper ~ background, e.g. white page
    on a white desk), fall back to an ink/edge-density mask.
    """
    h, w = image.shape[:2]
    bgr = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    # border frame samples -> background colour
    b = max(2, int(0.02 * min(h, w)))
    frame = np.concatenate([
        lab[:b, :].reshape(-1, 3), lab[-b:, :].reshape(-1, 3),
        lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3),
    ], axis=0)
    bg = np.median(frame, axis=0)
    dist = np.linalg.norm(lab - bg[None, None, :], axis=2)
    dist_n = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, fg = cv2.threshold(dist_n, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    fg = _clean(fg)
    cover = fg.mean() / 255.0
    if cover < 0.15 or cover > 0.97:
        # low contrast paper-vs-bg: fall back to ink/edge density
        fg2 = _ink_edge_mask(bgr)
        if 0.15 <= fg2.mean() / 255.0 <= 0.99:
            fg = fg2
    return (fg > 0).astype(np.uint8)


def _ink_edge_mask(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    dens = cv2.dilate(edges, k)
    dens = cv2.morphologyEx(dens, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45)))
    return _clean(dens)


def _clean(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    k = max(3, int(0.01 * min(h, w)) | 1)
    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, el)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, el)
    # keep large blobs only
    out = np.zeros_like(m)
    for _, st, _ in _largest_components((m > 0).astype(np.uint8), 0.02, top_k=4):
        x, y, ww, hh = st[cv2.CC_STAT_LEFT], st[cv2.CC_STAT_TOP], \
            st[cv2.CC_STAT_WIDTH], st[cv2.CC_STAT_HEIGHT]
        out[y:y+hh, x:x+ww] |= m[y:y+hh, x:x+ww]
    return out


def trim_facing_sliver(gray, fg, fx1, fy1, fx2, fy2):
    """If a single-page capture includes a thin strip of the FACING page along
    one outer edge (separated by a dark gutter), trim it off. Looks only in the
    outer 22% bands and only removes a strip narrower than 25% of the width.
    Returns a possibly-tightened (fx1, fx2).
    """
    region = gray[fy1:fy2 + 1, fx1:fx2 + 1].astype(np.float32)
    fgreg = fg[fy1:fy2 + 1, fx1:fx2 + 1].astype(bool)
    region[~fgreg] = np.nan
    bw = fx2 - fx1
    if bw < 20:
        return fx1, fx2
    with np.errstate(all="ignore"):
        col = np.nanmean(region, axis=0)
    med = float(np.nanmedian(col)) if np.isfinite(np.nanmedian(col)) else 0.0
    col = np.nan_to_num(col, nan=med)
    col_s = cv2.GaussianBlur(col.reshape(1, -1), (0, 0),
                             sigmaX=max(bw * 0.006, 1)).ravel()
    off = max(int(0.10 * bw), 4)
    med = float(np.median(col_s)) + 1e-6

    def deepest(band_lo, band_hi):
        idx = np.arange(band_lo, band_hi)
        if idx.size == 0:
            return None
        left = col_s[np.clip(idx - off, 0, bw - 1)]
        right = col_s[np.clip(idx + off, 0, bw - 1)]
        prom = np.minimum(left, right) - col_s[idx]
        b = int(np.argmax(prom))
        return band_lo + b, float(prom[b])

    new_l, new_r = fx1, fx2
    # left band [0,0.22]
    res = deepest(int(0.04 * bw), int(0.22 * bw))
    if res and res[1] > 0.14 * med:
        new_l = fx1 + res[0]
    # right band [0.78,0.96]
    res = deepest(int(0.78 * bw), int(0.96 * bw))
    if res and res[1] > 0.14 * med:
        new_r = fx1 + res[0]
    # only accept trims that remove a NARROW strip (facing-page sliver)
    if (new_l - fx1) > 0.16 * bw:
        new_l = fx1
    if (fx2 - new_r) > 0.16 * bw:
        new_r = fx2
    return new_l, new_r


def detect_pages(image: np.ndarray, two_folio_valley: float = 0.6
                 ) -> Tuple[List[PageBox], np.ndarray, float]:
    """Return (page_boxes, foreground_mask, gutter_valley_ratio).

    Decides 1 vs 2 pages from the depth of a central vertical valley in the
    foreground column profile. Boxes are split at the valley; spine.detect_gutter
    later refines the exact (curved) seam.
    """
    h, w = image.shape[:2]
    fg = foreground_mask(image)
    comps = _largest_components(fg, min_area_frac=0.05, top_k=4)
    if not comps:
        return [], fg, 1.0

    # overall foreground bbox
    xs = np.where(fg.any(axis=0))[0]
    ys = np.where(fg.any(axis=1))[0]
    fx1, fx2 = int(xs.min()), int(xs.max())
    fy1, fy2 = int(ys.min()), int(ys.max())

    # A bound book photographed open is CONTINUOUS paper across the spread, so
    # the gutter is NOT a gap in the foreground - it is the SPINE SHADOW, a dark
    # vertical valley in intensity. Detect the spine via the darkest central
    # column within the page region (restricted to the central band).
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    region = gray[fy1:fy2+1, fx1:fx2+1].astype(np.float32)
    fgreg = fg[fy1:fy2+1, fx1:fx2+1].astype(bool)
    region[~fgreg] = np.nan  # ignore background columns
    with np.errstate(all="ignore"):
        col = np.nanmean(region, axis=0)
    col = np.nan_to_num(col, nan=float(np.nanmedian(region)))
    bw = fx2 - fx1
    col_s = cv2.GaussianBlur(col.reshape(1, -1), (0, 0),
                             sigmaX=max(bw * 0.008, 1)).ravel()
    # The spine is a LOCALIZED dark dip flanked by brighter pages - not simply
    # the globally darkest column (page edges / vignetting can be darker). Score
    # each candidate by its PROMINENCE: how much darker it is than the brighter
    # of its two flanks, sampled ~12% of the spread width to either side.
    off = max(int(0.12 * bw), 5)
    lo, hi = int(0.32 * bw), int(0.68 * bw)
    if hi > lo and col_s.size > 2 * off + 2:
        idx = np.arange(lo, hi)
        left_flank = col_s[np.clip(idx - off, 0, bw - 1)]
        right_flank = col_s[np.clip(idx + off, 0, bw - 1)]
        flank = np.minimum(left_flank, right_flank)   # brighter side must exceed dip
        prominence = flank - col_s[idx]               # large => sharp dark spine
        best = int(np.argmax(prominence))
        valley_x = fx1 + lo + best
        med = float(np.median(col_s)) + 1e-6
        shadow = float(col_s[lo + best] / med)
        # if no real dip stands out, treat as weak (low two-folio confidence)
        if prominence[best] < 0.04 * med:
            shadow = max(shadow, 0.95)
    else:
        shadow = 1.0
        valley_x = (fx1 + fx2) // 2

    aspect = (fx2 - fx1) / float(max(fy2 - fy1, 1))
    # Two folios when the spread is landscape; a shadow valley confirms it and
    # rescues borderline-aspect spreads. Single pages are portrait.
    is_two = (aspect > 1.15) or (aspect > 1.05 and shadow < 0.92)

    if is_two:
        conf = float(np.clip(1.0 - shadow + 0.5, 0.5, 0.99))
        left = PageBox(fx1, fy1, valley_x, fy2, conf)
        right = PageBox(valley_x, fy1, fx2, fy2, conf)
        return [left, right], fg, shadow
    tl, tr = trim_facing_sliver(gray, fg, fx1, fy1, fx2, fy2)
    return [PageBox(tl, fy1, tr, fy2, 1.0)], fg, shadow
