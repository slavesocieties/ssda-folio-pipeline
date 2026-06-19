"""Stage 4 - Dynamic spine / gutter detection.

Replaces the legacy hardcoded ``mid = W // 2`` split. The gutter is *found*,
not assumed, via a minimum-energy vertical seam through a "gutter-likeness"
energy field. The seam may bend, so warped / curved / shadowed gutters on
off-center books are handled correctly.

Pure NumPy + OpenCV; no GPU, no learned weights required. Fully unit-testable.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple
import cv2
import numpy as np


def _normalize(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-6:
        return np.zeros_like(a, dtype=np.float32)
    return (a - lo) / (hi - lo)


def gutter_energy(
    gray: np.ndarray,
    mask: Optional[np.ndarray] = None,
    weights: Sequence[float] = (0.45, 0.35, 0.20),
) -> np.ndarray:
    """Per-pixel "this looks like a gutter" energy in [0, 1].

    Combines three complementary cues (all higher == more gutter-like):
      * darkness   : spines sit in a shadow valley   -> 1 - intensity
      * smoothness : gutters carry no text/ink        -> 1 - |grad|
      * mask gap   : gutters fall between page masks   -> 1 - mask
    """
    w_dark, w_smooth, w_gap = weights
    g = gray.astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), sigmaX=2.0)

    e_dark = 1.0 - _normalize(g)

    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    grad = cv2.GaussianBlur(grad, (0, 0), sigmaX=3.0)
    e_smooth = 1.0 - _normalize(grad)

    if mask is not None:
        m = _normalize(mask.astype(np.float32))
        e_gap = 1.0 - m
    else:
        e_gap = np.zeros_like(e_dark)
        w_gap = 0.0

    total_w = w_dark + w_smooth + w_gap
    energy = (w_dark * e_dark + w_smooth * e_smooth + w_gap * e_gap) / max(total_w, 1e-6)
    return energy.astype(np.float32)


def find_min_energy_seam(energy: np.ndarray, smoothness: float = 1.0) -> np.ndarray:
    """Vertical minimum-energy seam via dynamic programming.

    Returns one x-coordinate per row (length == energy.shape[0]). ``smoothness``
    penalises lateral moves so the seam stays coherent (a near-straight spine
    yields a near-constant seam; a curved spine bends with the book).

    Recurrence:
        C(x, y) = E(x, y) + min( C(x-1,y-1)+p, C(x,y-1), C(x+1,y-1)+p )
    """
    h, w = energy.shape
    if w == 0 or h == 0:
        return np.zeros(h, dtype=np.int32)
    cost = energy.astype(np.float64).copy()
    back = np.zeros((h, w), dtype=np.int8)  # -1 / 0 / +1 chosen offset
    p = float(smoothness)

    for y in range(1, h):
        prev = cost[y - 1]
        left = np.empty(w, dtype=np.float64)
        left[0] = np.inf
        left[1:] = prev[:-1] + p
        right = np.empty(w, dtype=np.float64)
        right[-1] = np.inf
        right[:-1] = prev[1:] + p
        center = prev
        stacked = np.stack([left, center, right], axis=0)  # 0=left,1=center,2=right
        choice = np.argmin(stacked, axis=0)
        cost[y] += np.take_along_axis(stacked, choice[None], 0)[0]
        back[y] = choice.astype(np.int8) - 1  # map {0,1,2} -> {-1,0,+1}

    seam = np.zeros(h, dtype=np.int32)
    seam[h - 1] = int(np.argmin(cost[h - 1]))
    for y in range(h - 1, 0, -1):
        seam[y - 1] = int(np.clip(seam[y] + back[y, seam[y]], 0, w - 1))
    return seam


def detect_gutter(
    image: np.ndarray,
    left_box: Tuple[int, int, int, int],
    right_box: Tuple[int, int, int, int],
    mask: Optional[np.ndarray] = None,
    band_margin_frac: float = 0.04,
    seam_smoothness: float = 0.3,
    energy_weights: Sequence[float] = (0.45, 0.35, 0.20),
) -> np.ndarray:
    """Find the full-resolution gutter seam between two detected page boxes.

    The search band is the *gap between the pages* (right edge of the left page
    to left edge of the right page), widened by ``band_margin_frac`` of image
    width. This is what makes off-center books work: there is no central-window
    assumption.

    Returns: x per image row (length == H), in full-resolution coordinates.
    """
    h, w = image.shape[:2]
    lx2 = left_box[2]
    rx1 = right_box[0]
    # robust to overlapping/contained boxes: fall back to a window around the
    # midpoint of the two box centers (still data-driven, never a blind W//2).
    if rx1 <= lx2:
        c = (left_box[0] + left_box[2] + right_box[0] + right_box[2]) // 4
        half = max(int(0.08 * w), 8)
        band_l, band_r = c - half, c + half
    else:
        margin = int(band_margin_frac * w)
        band_l = lx2 - margin
        band_r = rx1 + margin
    band_l = max(0, band_l)
    band_r = min(w, band_r)
    if band_r - band_l < 3:
        # pathological; return straight midline of the band
        x0 = (band_l + band_r) // 2
        return np.full(h, x0, dtype=np.int32)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    band = gray[:, band_l:band_r]
    band_mask = mask[:, band_l:band_r] if mask is not None else None

    energy = gutter_energy(band, band_mask, energy_weights)
    # The seam must follow MAXIMUM gutter-likeness. find_min_energy_seam
    # minimises cumulative cost, so feed it the inverse (1 - energy): low cost
    # where the gutter is most likely. This is the crucial sign convention.
    cost = 1.0 - energy
    seam_local = find_min_energy_seam(cost, smoothness=seam_smoothness)
    return (seam_local + band_l).astype(np.int32)


def split_along_seam(
    image: np.ndarray, seam: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Split an image into (left_page, right_page) along a per-row seam.

    Pixels right of the seam in the left output (and left of it in the right
    output) are set to the local background mean so downstream segmentation /
    crop is unaffected by the jagged seam edge.
    """
    h, w = image.shape[:2]
    xs = np.arange(w)[None, :]
    seam_col = seam[:, None]
    left_keep = xs <= seam_col       # (H, W) bool
    right_keep = xs > seam_col

    if image.ndim == 3:
        left_keep = left_keep[:, :, None]
        right_keep = right_keep[:, :, None]

    bg = int(np.median(image)) if image.ndim == 2 else \
        np.median(image.reshape(-1, image.shape[2]), axis=0).astype(image.dtype)

    left = np.where(left_keep, image, bg).astype(image.dtype)
    right = np.where(right_keep, image, bg).astype(image.dtype)

    # tight-crop each side to its content column range to drop the dead half
    max_seam = int(seam.max())
    min_seam = int(seam.min())
    left = left[:, : max_seam + 1]
    right = right[:, min_seam + 1 :]
    return left, right
