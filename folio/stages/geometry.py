"""Stage 3 boundary math + Stage 5 transform composition.

All geometry is computed and applied at full resolution. Crops are derived
from an oriented minimum-area rectangle of the page mask, dilated outward by a
margin so frayed edges and marginalia are never clipped, then sampled with a
single warpPerspective (one resample => minimal interpolation loss).
"""
from __future__ import annotations

from typing import Tuple
import cv2
import numpy as np


def largest_component_mask(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected foreground blob (drops speckle)."""
    m = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return m
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest).astype(np.uint8)


def oriented_page_quad(mask: np.ndarray, margin_frac: float = 0.015) -> np.ndarray:
    """Return the 4 corners (TL,TR,BR,BL) of the page's oriented bounding box,
    expanded outward by ``margin_frac`` of the page diagonal.
    """
    m = largest_component_mask(mask)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        h, w = mask.shape[:2]
        return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    rect = cv2.minAreaRect(max(cnts, key=cv2.contourArea))  # ((cx,cy),(w,h),angle)
    (cx, cy), (rw, rh), ang = rect
    diag = float(np.hypot(rw, rh))
    grow = margin_frac * diag
    rect_grown = ((cx, cy), (rw + 2 * grow, rh + 2 * grow), ang)
    box = cv2.boxPoints(rect_grown)            # 4x2, arbitrary order
    return _order_quad(box.astype(np.float32))


def _order_quad(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as TL, TR, BR, BL."""
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def crop_homography(quad: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """Perspective transform that maps an oriented page quad to an upright
    axis-aligned rectangle. Returns (H, out_w, out_h)."""
    tl, tr, br, bl = quad
    out_w = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))) + 1
    out_h = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))) + 1
    out_w, out_h = max(out_w, 1), max(out_h, 1)
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
                   dtype=np.float32)
    H = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return H, out_w, out_h


def rotation_matrix_3x3(angle_deg: float, w: int, h: int) -> Tuple[np.ndarray, int, int]:
    """3x3 homogeneous rotation about the image center with expand=True bounds."""
    rad = np.deg2rad(angle_deg)
    cos, sin = abs(np.cos(rad)), abs(np.sin(rad))
    new_w = int(round(h * sin + w * cos))
    new_h = int(round(h * cos + w * sin))
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    M[0, 2] += (new_w - w) / 2.0
    M[1, 2] += (new_h - h) / 2.0
    M3 = np.vstack([M, [0, 0, 1]]).astype(np.float64)
    return M3, new_w, new_h


def quarter_turn_matrix(k: int, w: int, h: int) -> Tuple[np.ndarray, int, int]:
    """Exact 90*k rotation as a 3x3 matrix (no interpolation loss)."""
    k = k % 4
    if k == 0:
        return np.eye(3), w, h
    return rotation_matrix_3x3(90.0 * k, w, h)


def compose_and_warp(
    image: np.ndarray,
    crop_H: np.ndarray,
    crop_w: int,
    crop_h: int,
    quarter_k: int = 0,
    skew_deg: float = 0.0,
    interp: int = cv2.INTER_CUBIC,
    border: int = cv2.BORDER_REPLICATE,
    border_value: int = 0,
) -> np.ndarray:
    """Compose crop homography, 90*k turn and fine skew into ONE matrix and
    apply it with a single warpPerspective. This avoids the legacy pattern of
    chained rotate(expand=True) calls that compound blur and re-pad repeatedly.

    ``interp``/``border``/``border_value`` are exposed so the same transform can
    warp a *mask* (NEAREST + CONSTANT 0) in lockstep with the image crop.
    """
    # after crop the canvas is crop_w x crop_h
    q_M, w1, h1 = quarter_turn_matrix(quarter_k, crop_w, crop_h)
    s_M, w2, h2 = rotation_matrix_3x3(skew_deg, w1, h1)
    T = s_M @ q_M @ crop_H
    return cv2.warpPerspective(image, T, (w2, h2), flags=interp,
                               borderMode=border, borderValue=border_value)
