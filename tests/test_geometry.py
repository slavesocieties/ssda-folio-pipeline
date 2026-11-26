"""Tests for boundary cropping + transform composition (Stages 3 & 5)."""
import numpy as np
import cv2
from folio.stages import geometry, orient


def test_largest_component_drops_speckle():
    m = np.zeros((100, 100), np.uint8)
    m[20:80, 20:80] = 1          # big square
    m[0:3, 0:3] = 1              # speckle
    out = geometry.largest_component_mask(m)
    assert out[50, 50] == 1
    assert out[1, 1] == 0


def test_oriented_quad_covers_mask_with_margin():
    m = np.zeros((200, 300), np.uint8)
    m[40:160, 60:240] = 1
    quad = geometry.oriented_page_quad(m, margin_frac=0.02)
    xs, ys = quad[:, 0], quad[:, 1]
    # quad should enclose the mask bbox (margin grows outward)
    assert xs.min() <= 60 and xs.max() >= 240
    assert ys.min() <= 40 and ys.max() >= 160


def test_oriented_quad_fits_true_trapezoid_under_perspective():
    """A page photographed at a real angle (not just in-plane rotation) has a
    genuinely trapezoidal outline that no rotated rectangle can represent.
    Confirmed on a real keystoned crop from the corpus: minAreaRect either
    clips real content or pads in background. The quad fit should recover the
    actual corners, not the bounding rectangle."""
    m = np.zeros((200, 300), np.uint8)
    true_quad = np.array([[60, 40], [240, 40], [210, 160], [90, 160]], np.int32)
    cv2.fillPoly(m, [true_quad], 1)
    quad = geometry.oriented_page_quad(m, margin_frac=0.0)
    assert np.allclose(quad, true_quad, atol=1.0)
    # a minAreaRect fit would have to be at least as wide as the top edge
    # (240-60=180) by the full height (120) -- the true trapezoid is smaller
    rect_area = 180 * 120
    assert cv2.contourArea(quad.astype(np.float32)) < 0.9 * rect_area


def test_oriented_quad_keeps_rect_path_for_ordinary_pages():
    """An ordinary (non-keystoned) page must still use the plain rotated-rect
    fit -- the quad path only activates on real perspective distortion."""
    m = np.zeros((200, 300), np.uint8)
    m[40:160, 60:240] = 1
    quad = geometry.oriented_page_quad(m, margin_frac=0.0)
    expected = np.array([[60, 40], [239, 40], [239, 159], [60, 159]], np.float32)
    assert np.allclose(quad, expected, atol=1.0)


def test_quarter_turn_dims():
    M, w, h = geometry.quarter_turn_matrix(1, 300, 200)
    assert (w, h) == (200, 300)  # 90deg swaps dims
    M0, w0, h0 = geometry.quarter_turn_matrix(0, 300, 200)
    assert (w0, h0) == (300, 200)
    assert np.allclose(M0, np.eye(3))


def test_compose_warp_upright_portrait():
    # landscape page -> ask for a 90deg turn -> expect portrait output
    img = np.full((200, 300, 3), 200, np.uint8)
    mask = np.zeros((200, 300), np.uint8)
    mask[10:190, 10:290] = 1
    quad = geometry.oriented_page_quad(mask, 0.0)
    H, cw, ch = geometry.crop_homography(quad)
    out = geometry.compose_and_warp(img, H, cw, ch, quarter_k=1, skew_deg=0.0)
    assert out.shape[0] > out.shape[1]  # portrait


def test_estimate_skew_recovers_known_tilt():
    # build horizontal text lines, tilt by +6deg, expect estimator ~ -6 to undo
    img = np.full((300, 300), 255, np.uint8)
    for y in range(40, 260, 20):
        img[y:y + 4, 40:260] = 0
    M = cv2.getRotationMatrix2D((150, 150), 6.0, 1.0)
    tilted = cv2.warpAffine(img, M, (300, 300), borderValue=255)
    est = orient.estimate_skew(tilted, max_deg=15, step=0.5)
    assert abs(est - (-6.0)) <= 1.5


def test_resolve_quarter_turn_confident():
    probs = np.array([0.9, 0.04, 0.03, 0.03])
    gray = np.zeros((50, 50), np.uint8)
    k, conf = orient.resolve_quarter_turn(probs, gray)
    assert k == 0 and conf == 0.9
