"""Provenance quad un-rotation + 10:24 aspect padding."""
import numpy as np
from folio.pipeline import _unrotate_quad, _fit_max_aspect


def test_unrotate_recovers_original_point():
    H0, W0 = 7, 5
    for k in range(4):
        O = np.zeros((H0, W0), np.uint8)
        oy, ox = 2, 3                      # marker at (col=3, row=2)
        O[oy, ox] = 255
        W = np.rot90(O, k)
        yw, xw = map(int, np.argwhere(W == 255)[0])  # (row, col) in working
        (rx, ry), = _unrotate_quad([(xw, yw)], k, W0, H0)
        assert (rx, ry) == (ox, oy), f"k={k}: got {(rx, ry)} want {(ox, oy)}"


def test_fit_max_aspect_pads_tall_image():
    img = np.zeros((300, 100, 3), np.uint8)   # ratio 3.0 (long:short)
    out = _fit_max_aspect(img, 2.4)
    h, w = out.shape[:2]
    assert h == 300 and w > 100               # padded width, height unchanged (no crop)
    assert max(h, w) / min(h, w) <= 2.4 + 1e-6


def test_fit_max_aspect_noop_within_ratio():
    img = np.zeros((280, 200, 3), np.uint8)   # ratio 1.4 < 2.4
    out = _fit_max_aspect(img, 2.4)
    assert out.shape == img.shape


def test_fit_max_aspect_disabled():
    img = np.zeros((300, 100, 3), np.uint8)
    assert _fit_max_aspect(img, 0).shape == img.shape
