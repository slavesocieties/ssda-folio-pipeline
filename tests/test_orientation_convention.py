"""Lock the orientation label convention: a page rotated to label k must be
restored to upright by the pipeline's geometry correction quarter_k=(-k)%4.
This is what makes the trained 4-way head's outputs actually fix orientation.
Pure NumPy/OpenCV - no torch."""
import numpy as np
import cv2
from folio.stages import geometry as G
from folio.training.labels import apply_orientation, correction_quarter_k


def _ident_crop(img):
    h, w = img.shape[:2]
    quad = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    return G.crop_homography(quad)


def test_quarter_turn_matches_rot90():
    up = np.zeros((120, 80, 3), np.uint8)
    up[:30, :] = (255, 0, 0); up[:, :15] = (0, 255, 0)
    H, cw, ch = _ident_crop(up)
    for qk in range(4):
        out = G.compose_and_warp(up, H, cw, ch, quarter_k=qk, skew_deg=0.0)
        ref = np.rot90(up, qk)
        assert out.shape[:2] == ref.shape[:2]
        assert np.mean(np.abs(out.astype(int) - ref.astype(int))) < 3


def test_label_correction_round_trip():
    up = np.zeros((120, 80, 3), np.uint8)
    up[:30, :] = (255, 0, 0); up[:, :15] = (0, 255, 0)
    for k in range(4):
        cur = apply_orientation(up, k)             # image now at label k
        H, cw, ch = _ident_crop(cur)
        out = G.compose_and_warp(cur, H, cw, ch,
                                 quarter_k=correction_quarter_k(k), skew_deg=0.0)
        assert out.shape[:2] == up.shape[:2]
        assert np.mean(np.abs(out.astype(int) - up.astype(int))) < 3
