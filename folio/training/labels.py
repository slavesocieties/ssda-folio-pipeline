"""Label conventions shared by training, eval and inference. Torch-free so it
can be unit-tested anywhere.

ORIENTATION: label k in {0,1,2,3} means the image is currently rotated k
counter-clockwise quarter-turns from upright (the np.rot90 convention). To
restore upright the pipeline applies geometry.compose_and_warp(quarter_k=(-k)%4),
which is verified to invert np.rot90(k) (see tests/test_orientation_convention).

COUNT: 0=one_folio, 1=two_folios, 2=reject  (matches
folio.schemas.PageCount order used by FolioCountClassifier.CLASSES).
"""
from __future__ import annotations
import numpy as np

ORIENTATION_DEGREES = [0, 90, 180, 270]   # index == label k
COUNT_CLASSES = ["one_folio", "two_folios", "reject"]


def apply_orientation(image: np.ndarray, k: int) -> np.ndarray:
    """Rotate an upright image to orientation-label k (np.rot90 convention)."""
    return np.ascontiguousarray(np.rot90(image, k % 4))


def correction_quarter_k(k: int) -> int:
    """The geometry quarter_k that restores upright from label k."""
    return (-k) % 4
