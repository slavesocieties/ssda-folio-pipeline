"""Two-folio asymmetric margin: outer edges get breathing room, the spine edge
stays clamped at the seam (no facing-page sliver)."""
from types import SimpleNamespace
import numpy as np

from folio.pipeline import FolioPipeline

# _margin_to_seam only reads self.cfg.geom.crop_margin_frac
_FAKE = SimpleNamespace(cfg=SimpleNamespace(geom=SimpleNamespace(crop_margin_frac=0.05)))


def _run(mask, sel):
    return FolioPipeline._margin_to_seam(_FAKE, mask, sel)


def test_spine_edge_clamped_at_seam():
    H, W = 200, 220
    mask = np.zeros((H, W), np.uint8)
    mask[20:180, 10:210] = 1            # page paint spans both sides
    sel = np.zeros((H, W), np.uint8)
    sel[:, :101] = 1                    # this page is the LEFT side, seam at col 100
    out = _run(mask, sel)
    cols = np.where(out.any(axis=0))[0]
    # spine (right) edge never crosses the seam
    assert cols.max() <= 100
    # outer (left) edge got the margin (dilated past the original col 10)
    assert cols.min() < 10
    # vertical edges got the margin too
    rows = np.where(out.any(axis=1))[0]
    assert rows.min() < 20 and rows.max() > 179


def test_right_page_spine_on_left():
    H, W = 200, 220
    mask = np.zeros((H, W), np.uint8)
    mask[20:180, 10:210] = 1
    sel = np.zeros((H, W), np.uint8)
    sel[:, 100:] = 1                    # this page is the RIGHT side, seam at col 100
    out = _run(mask, sel)
    cols = np.where(out.any(axis=0))[0]
    assert cols.min() >= 100            # spine (left) edge stays at the seam
    assert cols.max() > 210            # outer (right) edge dilated


def test_empty_side_is_safe():
    mask = np.zeros((100, 100), np.uint8)
    sel = np.zeros((100, 100), np.uint8); sel[:, :50] = 1
    assert _run(mask, sel).sum() == 0


def test_enforce_aspect_extends_square_mask():
    # a square content mask in a tall image -> extended vertically toward portrait,
    # width unchanged, never cropped
    img_hw = (1000, 600)
    mask = np.zeros(img_hw, np.uint8)
    mask[400:600, 100:500] = 1               # 400x200, w/h = 2.0 (very wide)
    out = FolioPipeline._enforce_page_aspect(mask, img_hw, 0.80)
    ys = np.where(out.any(axis=1))[0]; xs = np.where(out.any(axis=0))[0]
    w = xs.max()-xs.min()+1; h = ys.max()-ys.min()+1
    assert w == 400                          # width unchanged
    assert w / h <= 0.80 + 0.02              # now portrait enough
    assert h > 200                           # extended (added area, never cropped)


def test_enforce_aspect_noop_on_portrait():
    img_hw = (1000, 600)
    mask = np.zeros(img_hw, np.uint8)
    mask[100:900, 150:450] = 1               # 300x800, w/h = 0.375 (already portrait)
    out = FolioPipeline._enforce_page_aspect(mask, img_hw, 0.80)
    assert int(out.sum()) == int(mask.sum())  # unchanged
