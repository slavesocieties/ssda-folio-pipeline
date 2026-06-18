"""Tests for the dynamic gutter detector (Stage 4). These prove the seam
search finds the TRUE spine even when it is off-center and curved - exactly
the cases the legacy W//2 split fails on."""
import numpy as np
import pytest
from folio.stages import spine


def _synthetic_spread(w=400, h=300, gutter_x=None, curve=0):
    """White pages with a dark vertical gutter at gutter_x (optionally curved)."""
    img = np.full((h, w, 3), 235, np.uint8)
    rng = np.random.default_rng(0)
    img = (img.astype(np.int16) + rng.integers(-8, 8, img.shape)).clip(0, 255).astype(np.uint8)
    gx = gutter_x if gutter_x is not None else w // 2
    seam_true = np.zeros(h, np.int32)
    for y in range(h):
        cx = int(gx + curve * np.sin(np.pi * y / h))
        seam_true[y] = cx
        img[y, max(cx - 3, 0):cx + 3] = 30  # dark spine band
    # add fake text columns so "smoothness" cue is meaningful
    for tx in (gx // 2, gx + (w - gx) // 2):
        img[40:h - 40:6, max(tx - 40, 0):tx + 40] = 60
    return img, seam_true


def test_finds_centered_gutter():
    img, true = _synthetic_spread(gutter_x=200)
    seam = spine.detect_gutter(img, (10, 0, 195, 300), (205, 0, 390, 300))
    assert abs(float(np.median(seam)) - 200) <= 8


def test_finds_off_center_gutter():
    # gutter at x=140, NOT the midpoint 200 -> legacy W//2 would cut a page
    img, true = _synthetic_spread(gutter_x=140)
    seam = spine.detect_gutter(img, (10, 0, 135, 300), (145, 0, 390, 300))
    assert abs(float(np.median(seam)) - 140) <= 6
    assert abs(float(np.median(seam)) - 200) > 40  # clearly not the midpoint


def test_follows_curved_gutter():
    img, true = _synthetic_spread(gutter_x=200, curve=25)
    seam = spine.detect_gutter(img, (10, 0, 195, 300), (205, 0, 390, 300))
    # seam must bend, tracking the true curved spine within tolerance
    err = np.abs(seam - true)
    assert float(np.mean(err)) < 8
    assert int(seam.max() - seam.min()) > 10  # it actually curved


def test_seam_is_monotonic_in_motion():
    """DP seam may move at most 1 px laterally per row (coherence guarantee)."""
    energy = np.random.default_rng(1).random((50, 30)).astype(np.float32)
    seam = spine.find_min_energy_seam(energy, smoothness=1.0)
    assert np.all(np.abs(np.diff(seam)) <= 1)
    assert seam.shape[0] == 50


def test_split_shapes():
    img, _ = _synthetic_spread(gutter_x=160)
    seam = spine.detect_gutter(img, (10, 0, 155, 300), (165, 0, 390, 300))
    left, right = spine.split_along_seam(img, seam)
    assert left.shape[0] == img.shape[0] and right.shape[0] == img.shape[0]
    assert left.shape[1] + right.shape[1] <= img.shape[1] + 5
