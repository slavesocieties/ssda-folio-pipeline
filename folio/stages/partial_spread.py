"""partial_spread.py — rescue crops of a book photographed part-way open.

When a volume is shot with the facing leaf only partly open, the count head
correctly reports ONE folio (only one complete folio is present) and the segmenter
keeps the contiguous paper, which drags the partial facing leaf into the crop. The
result is too wide — a folio in these registers is ~0.77 w/h, these land at
0.95-1.37 — and trips the `unexpected_aspect` quality gate.

The rescue finds the spine in the finished crop and cuts there, keeping the
complete folio. It fires ONLY on a crop whose aspect is already out of range, so
it cannot disturb a crop that is currently fine.

Finding the spine
-----------------
The spine is scored per column by the FRACTION OF ROWS on which that column is
locally darker than the paper to either side. A spine shadow is dark at every row
down the page; a column of handwriting is dark only where text sits, and a page
edge only where the page is. Measured on volume 265705: real spines score
0.29-0.42 against page medians of 0.17-0.20, so the decision uses the RATIO of
peak to the page's own median — an absolute cutoff fires on everything or nothing,
because "dark" depends on that page's handwriting density.

Calibration (volume 265705, 76 flagged crops, 10 hand-labelled): 9/10 correct,
1 correctly declined, 0 wrong, and 0 false fires on 60 correctly-cropped controls.
The `min_spine` gap is narrow — the one wrong case scored 1.416 and the weakest
correct case 1.582 — so the default is deliberately conservative: a crop with no
confident spine keeps its `unexpected_aspect` flag and goes to review unchanged.
"""
from typing import List, Optional, Tuple

import cv2
import numpy as np


def spine_score(gray: np.ndarray, off_frac: float = 0.05,
                dark: float = 6.0) -> np.ndarray:
    """Per-column "is this the spine?" score in [0,1] (fraction of dark rows).

    A column counts as dark on a row when it is darker than the local horizontal
    average. A two-sided variant (darker than BOTH flanks) was tried, to reject
    the ragged edge of a text block, which is darker than a window straddling
    text and margin. It is more principled but measurably worse here: on volume
    265705 it dropped agreement with the hand labels from 9/10 to 7/10, losing
    one correct rescue and flipping another. The one-sided form is kept because
    the evidence favours it; revisit if a text-block edge is ever seen winning.
    """
    g = cv2.GaussianBlur(gray, (0, 0), 2).astype(np.float32)
    k = max(int(off_frac * gray.shape[1]) | 1, 3)          # odd kernel
    bg = cv2.blur(g, (k, 1), borderType=cv2.BORDER_REPLICATE)
    return ((bg - g) > dark).mean(axis=0)


def find_spine(gray: np.ndarray, lo: float = 0.18, hi: float = 0.82,
               smooth: float = 0.01) -> Tuple[Optional[int], float]:
    """Best spine column within [lo,hi] of the width.

    Returns (x, ratio) where ratio is the peak score over the image's own median.
    """
    w = gray.shape[1]
    s = spine_score(gray)
    s = cv2.GaussianBlur(s.reshape(1, -1).astype(np.float32), (0, 0),
                         sigmaX=max(w * smooth, 1)).ravel()
    a, b = int(lo * w), int(hi * w)
    if b <= a:
        return None, 0.0
    seg = s[a:b]
    i = int(np.argmax(seg))
    # Floor the denominator: on a page whose columns are unusually uniform the
    # median score approaches zero and the ratio would explode, forcing a fire on
    # what may be no spine at all. Real pages sit near 0.17, well above the floor.
    den = max(float(np.median(s)), 0.05)
    return a + i, float(seg[i] / den)


def _ink(gray: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(gray, (0, 0), 2)
    bg = cv2.medianBlur(g, 31)
    return (np.clip(bg.astype(np.int16) - g.astype(np.int16), 0, 255) > 18)


def split_quad(quad: List[List[float]], t: float, keep_left: bool) -> List[List[float]]:
    """Cut a crop quad at fraction `t` of its width, keeping one side.

    quad is [TL, TR, BR, BL]. The new corners are linear interpolations along the
    top and bottom edges. Exact for an affine warp and a close approximation for a
    mild perspective one, which is what a flat-ish page photographed square is.
    """
    (tl, tr, br, bl) = [np.asarray(p, dtype=float) for p in quad]
    top = tl + (tr - tl) * t
    bot = bl + (br - bl) * t
    if keep_left:
        return [tl.tolist(), top.tolist(), bot.tolist(), bl.tolist()]
    return [top.tolist(), tr.tolist(), br.tolist(), bot.tolist()]


def rescue_partial_spread(bgr: np.ndarray, quad: Optional[List[List[float]]] = None,
                          aspect_hi: float = 0.95, target_aspect: float = 0.77,
                          min_spine: float = 1.5, min_ink: float = 0.02):
    """Split an over-wide one-folio crop at the spine, keeping the complete folio.

    Returns ``(kept_bgr, kept_quad, info)``. ``kept_bgr`` is None when the rule
    declines, in which case the caller must leave the crop untouched.
    """
    h, w = bgr.shape[:2]
    aspect = w / float(h)
    info = {"aspect_before": round(aspect, 3), "fired": False}
    if aspect <= aspect_hi:
        info["why"] = "aspect already in range"
        return None, quad, info

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    ink = _ink(gray)
    page_ink = float(ink.mean())
    info["page_ink"] = round(page_ink, 4)
    if page_ink < min_ink:
        # Covers, endpapers and near-blank leaves: no text to reason about and no
        # spread to split. Inventing a gutter in a blank field helps nobody.
        info["why"] = f"too little text ({page_ink:.4f}) - cover/blank, not a spread"
        return None, quad, info

    x, score = find_spine(gray)
    info["spine_x_frac"] = round(x / float(w), 3) if x is not None else None
    info["spine_score"] = round(score, 3)
    if x is None or score < min_spine:
        info["why"] = f"no spine-like column (score {score:.3f})"
        return None, quad, info

    cands = []
    for keep_left in (True, False):
        kw = x if keep_left else w - x
        if kw < 0.2 * w:
            continue
        cands.append((abs(kw / float(h) - target_aspect), keep_left, kw))
    if not cands:
        info["why"] = "spine too close to an edge"
        return None, quad, info

    # With the spine located, the side to keep is the one shaped like a folio.
    # (Valley depth and ink density were both tried as proxies and both mispicked.)
    _d, keep_left, kw = min(cands, key=lambda c: c[0])
    kept = bgr[:, :x] if keep_left else bgr[:, x:]
    info.update(fired=True, kept="left" if keep_left else "right",
                aspect_after=round(kw / float(h), 3))
    new_quad = split_quad(quad, x / float(w), keep_left) if quad is not None else None
    return kept, new_quad, info
