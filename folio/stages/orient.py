"""Stage 5 - orientation. Coarse 4-way (model-driven, with a classical
text-line tie-breaker) plus a final sub-degree skew correction.

The classical pieces here are dependency-light and unit-testable; the learned
4-way classifier lives in ``folio.models.orientation`` and is consulted by the
pipeline. ``estimate_skew`` and ``text_line_score`` are pure NumPy/OpenCV.
"""
from __future__ import annotations

from typing import Tuple
import cv2
import numpy as np


def _binary_ink(gray: np.ndarray) -> np.ndarray:
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return bw


def _text_ink(gray: np.ndarray) -> np.ndarray:
    """Ink mask for the skew objective via a LOCAL (adaptive) threshold.

    A global Otsu on a page crop separates page-from-margin, not ink-from-paper,
    so the whole folio reads as one solid blob and the skew search locks onto the
    page rectangle -- railing to +/-max_deg. An adaptive (mean-C) threshold keys
    on dark strokes/ruled lines instead, which is what actually carries the skew
    signal. Measured on real crops this drops mean deskew error ~10deg -> ~1deg
    and eliminates railing (45% -> 0%). Falls back to Otsu for tiny images.
    """
    h, w = gray.shape[:2]
    m = min(h, w)
    bs = 31 if m >= 31 else (m if m % 2 == 1 else m - 1)
    if bs < 3:
        return _binary_ink(gray)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY_INV, bs, 10)


def text_line_score(gray: np.ndarray) -> float:
    """How strongly do horizontal text lines line up? Variance of the
    horizontal projection profile (row ink counts). High == well-aligned rows.
    Used as a tie-breaker between 0 and 180 / 90 and 270 candidates.
    """
    bw = _binary_ink(gray)
    proj = bw.sum(axis=1).astype(np.float64)
    if proj.size == 0:
        return 0.0
    return float(np.var(proj / (bw.shape[1] * 255.0 + 1e-9)))


def _skew_scores(bw: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Projection-profile variance for each candidate angle (higher == better)."""
    h, w = bw.shape
    cx, cy = w / 2.0, h / 2.0
    scores = np.empty(len(angles), np.float64)
    for i, angle in enumerate(angles):
        M = cv2.getRotationMatrix2D((cx, cy), float(angle), 1.0)
        rot = cv2.warpAffine(bw, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        scores[i] = np.var(rot.sum(axis=1))
    return scores


def estimate_skew(gray: np.ndarray, max_deg: float = 15.0, step: float = 0.25,
                  coarse_step: float = 1.0, downscale_to: int = 700,
                  topk: int = 3) -> float:
    """Fine in-plane skew via projection-profile variance maximisation.

    Returns the angle (deg) to rotate the image by to make text lines
    horizontal. More stable than the legacy sum-of-squared-diffs because
    variance is scale-normalised across angles.

    The objective keys on a LOCAL-threshold text mask (``_text_ink``) rather than
    a global Otsu, which is what stops the search from railing to +/-max_deg on
    page crops (see ``_text_ink``).

    Coarse-to-fine for speed WITHOUT sacrificing precision: a cheap coarse sweep
    (on a downscaled mask) localises the peak, then the angle is resolved at the
    FULL ``step`` resolution on the full-size mask around the top-``topk`` coarse
    candidates. The fine sweep evaluates exactly the scores the brute-force sweep
    would, so the returned angle matches an exhaustive search whenever the global
    optimum lies within ``coarse_step`` of one of the top-k coarse peaks (true for
    the smooth, single-lobed text-projection landscape); the top-k window guards
    the rare multi-lobed case. A final parabolic interpolation around the discrete
    peak yields sub-``step`` precision.
    """
    bw = _text_ink(gray).astype(np.float32)
    h, w = bw.shape
    if bw.sum() < 1.0:                       # blank / textless: nothing to align
        return 0.0

    # 1) coarse sweep, on a downscaled copy when the page is large (the peak
    #    angle of the row-projection variance is preserved under downscaling).
    scale = downscale_to / float(max(h, w)) if max(h, w) > downscale_to else 1.0
    bw_coarse = (cv2.resize(bw, (max(int(w * scale), 1), max(int(h * scale), 1)),
                            interpolation=cv2.INTER_AREA) if scale < 1.0 else bw)
    coarse_angles = np.arange(-max_deg, max_deg + coarse_step, coarse_step)
    coarse_scores = _skew_scores(bw_coarse, coarse_angles)

    # 2) refine at full resolution + full step around the best coarse candidates.
    k = max(1, min(topk, len(coarse_angles)))
    top_idx = np.argsort(coarse_scores)[::-1][:k]
    fine_set = set()
    for ci in top_idx:
        c = float(coarse_angles[ci])
        lo = max(-max_deg, c - coarse_step)
        hi = min(max_deg, c + coarse_step)
        fine_set.update(np.round(np.arange(lo, hi + step, step), 6).tolist())
    fine_angles = np.array(sorted(fine_set), np.float64)
    fine_scores = _skew_scores(bw, fine_angles)
    j = int(np.argmax(fine_scores))
    best = float(fine_angles[j])

    # 3) sub-step parabolic peak, only when both neighbours are contiguous (+/-step)
    #    and the local curve is concave (a genuine interior maximum).
    if 0 < j < len(fine_angles) - 1 \
            and abs(fine_angles[j] - fine_angles[j - 1] - step) < 1e-6 \
            and abs(fine_angles[j + 1] - fine_angles[j] - step) < 1e-6:
        y0, y1, y2 = fine_scores[j - 1], fine_scores[j], fine_scores[j + 1]
        denom = y0 - 2.0 * y1 + y2
        if denom < 0.0:
            delta = 0.5 * (y0 - y2) / denom
            if abs(delta) <= 1.0:
                best = float(fine_angles[j] + delta * step)
    return best


def resolve_quarter_turn(probs: np.ndarray, gray: np.ndarray,
                         conf_margin: float = 0.15) -> Tuple[int, float]:
    """Combine the 4-way classifier distribution with the classical text-line
    score to pick k in {0,1,2,3} (rotation needed = -90*k to make upright).

    probs: softmax over [0, 90, 180, 270] degrees of *current* rotation.
    Returns (k_to_undo, confidence).
    """
    probs = np.asarray(probs, dtype=np.float64).ravel()
    order = np.argsort(probs)[::-1]
    top, second = int(order[0]), int(order[1])
    conf = float(probs[top])
    if conf - float(probs[second]) >= conf_margin:
        return top, conf
    # ambiguous: use text-line score to break ties between the two candidates
    cand_scores = {}
    for k in (top, second):
        rotated = np.rot90(gray, k=(-k) % 4)
        cand_scores[k] = text_line_score(rotated)
    best_k = max(cand_scores, key=cand_scores.get)
    return best_k, conf
