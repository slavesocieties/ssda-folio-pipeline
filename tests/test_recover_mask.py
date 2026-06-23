"""Sparse-page mask recovery: a band mask on a tall bright page is expanded to
the full page; a correct full-page mask is left alone."""
import numpy as np
import cv2

from folio.pipeline import FolioPipeline

R = FolioPipeline._recover_page_mask  # staticmethod, no models needed


def _page_on_black(h=1200, w=800, pad=80):
    """A tall bright page centred on a black background (like an archival scan)."""
    img = np.zeros((h, w, 3), np.uint8)
    img[pad:h - pad, pad:w - pad] = (210, 210, 210)
    return img


def test_recovers_band_mask_to_full_page():
    img = _page_on_black()
    h, w = img.shape[:2]
    # legacy segmenter masked only a middle band (one text block) of the page
    mask = np.zeros((h, w), np.uint8)
    mask[520:680, 120:680] = 1
    out = R(img, mask)
    ys = np.where(out.any(axis=1))[0]
    # recovered mask should now cover most of the page height, not just the band
    assert (ys.max() - ys.min()) > 0.7 * (h - 2 * 80)


def test_noop_on_full_page_mask():
    img = _page_on_black()
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    mask[80:h - 80, 80:w - 80] = 1            # already the whole page
    out = R(img, mask)
    before = mask.sum()
    after = out.sum()
    # essentially unchanged (no spurious expansion)
    assert abs(int(after) - int(before)) < 0.1 * before


def test_noop_when_empty_mask():
    img = _page_on_black()
    mask = np.zeros(img.shape[:2], np.uint8)
    out = R(img, mask)
    assert out.sum() == 0
