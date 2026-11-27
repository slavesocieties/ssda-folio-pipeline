"""Tests for classical background-agnostic page detection (Stage 1/3)."""
import numpy as np
from folio.stages import foreground as fg


def test_foreground_mask_finds_a_clear_page():
    img = np.full((200, 150, 3), 40, np.uint8)       # dark background border
    img[20:180, 15:135] = 230                        # light page
    mask = fg.foreground_mask(img)
    assert mask[100, 75] == 1       # page interior
    assert mask[5, 5] == 0          # background border


def test_foreground_mask_falls_back_to_full_frame_when_both_signals_fail(monkeypatch):
    """Real, severely degraded pages (tears, heavy staining) can make BOTH
    signals untrustworthy: primary colour-distance finds nothing (page colour
    barely differs from its background) and the ink/edge fallback finds the
    damage texture everywhere, failing its own sanity bound too (measured
    ~99.5-99.8% on three such real pages). Without a further fallback here,
    foreground_mask returns empty and the pipeline produces ZERO output for
    a real page (no_page_detected) -- confirmed this used to happen on those
    pages before this fallback was added."""
    h, w = 200, 150
    img = np.full((h, w, 3), 200, np.uint8)  # uniform -> primary mask finds nothing
    monkeypatch.setattr(fg, "_ink_edge_mask",
                        lambda bgr: np.full(bgr.shape[:2], 255, np.uint8))
    mask = fg.foreground_mask(img)
    b = max(2, int(0.02 * min(h, w)))
    assert mask[b:h - b, b:w - b].all()      # full frame...
    assert mask[0, 0] == 0 and mask[-1, -1] == 0   # ...minus a thin border
