"""Tests for the adaptive-threshold deskew (the railing fix)."""
import numpy as np
import cv2

from folio.stages import orient


def _text_page(h=900, w=600, tilt=0.0, border=False):
    """A portrait page of ruled text lines, optionally tilted, optionally with a
    dark scan border (the case that used to rail the global-Otsu objective)."""
    img = np.full((h, w), 245, np.uint8)
    for y in range(80, h - 80, 28):
        img[y:y + 4, 70:w - 70] = 35
    if tilt:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), tilt, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderValue=245)
    if border:
        img[:18, :] = 20; img[-18:, :] = 20
        img[:, :18] = 20; img[:, -18:] = 20
    return img


def test_recovers_known_tilt():
    est = orient.estimate_skew(_text_page(tilt=6.0))
    assert abs(est - (-6.0)) <= 1.5


def test_no_railing_on_bordered_upright_page():
    """Upright page with a dark border must NOT rail to +/-max_deg."""
    est = orient.estimate_skew(_text_page(tilt=0.0, border=True))
    assert abs(est) < 3.0          # near 0, not +/-15


def test_blank_page_returns_zero():
    blank = np.full((500, 400), 250, np.uint8)
    assert orient.estimate_skew(blank) == 0.0


def test_text_ink_is_binary_and_sparse_on_blank():
    page = _text_page()
    mask = orient._text_ink(page)
    assert set(np.unique(mask)).issubset({0, 255})
    blank = np.full((400, 300), 250, np.uint8)
    assert orient._text_ink(blank).mean() < 5.0   # almost no ink
