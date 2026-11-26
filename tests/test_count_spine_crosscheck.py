"""Pins the trained-count / classical-spine cross-check: a strong, sharp
spine dip is trusted enough to actually auto-correct a wrong "one folio"
call into a real two-folio split; a landscape-shaped frame with no real
gutter (a single page in a wide frame, or a book-spine photo) is NOT --
it only gets flagged for review, never auto-split. Calibrated on 173 real
disagreements from the corpus: a clean gap in spine-prominence separated
138 real missed spreads from 35 false triggers.
"""
import numpy as np
from folio.config import PipelineConfig
from folio.pipeline import FolioPipeline
from folio.schemas import PageBox, PageCount


class OneCounter:
    """Simulates the trained model confidently (and wrongly) calling ONE."""
    def predict(self, image):
        return PageCount.ONE, 0.99


class HalfSplitSegmenter:
    """Two page boxes on either side of x=230, matching the synthetic gutter."""
    def detect(self, image, max_pages=2):
        h, w = image.shape[:2]
        gx = 230
        if max_pages == 1:
            return [PageBox(8, 8, w - 8, h - 8, 0.97)]
        return [PageBox(8, 8, gx - 5, h - 8, 0.97),
                PageBox(gx + 5, 8, w - 8, h - 8, 0.96)]

    def segment(self, image, boxes):
        h, w = image.shape[:2]
        out = []
        for b in boxes:
            m = np.zeros((h, w), np.uint8)
            m[b.y1:b.y2, b.x1:b.x2] = 1
            out.append(m)
        return out


class UprightOrienter:
    def predict_probs(self, image):
        return np.array([0.95, 0.02, 0.02, 0.01])


def _page_on_background(w, h, margin=20, fill=235, bg=0):
    """A page rectangle on a distinct background border, so foreground_mask's
    border-sampled background estimate actually finds a page to detect (a
    flat, edge-to-edge fill has no border/page contrast at all)."""
    img = np.full((h, w, 3), bg, np.uint8)
    img[margin:h - margin, margin:w - margin] = fill
    return img


def _strong_gutter_image(w=600, h=400, gutter_x=230):
    """A real, sharp dark spine dip -> foreground.detect_pages should find
    strong_spine=True (high prominence/med). The gutter fill must stay
    closer in colour to the page than to the background, or the
    foreground-mask step (correctly) reads it as background, not a dip."""
    img = _page_on_background(w, h)
    img[:, max(gutter_x - 4, 0):gutter_x + 4] = 140
    return img


def _landscape_no_gutter_image(w=600, h=400):
    """Landscape-shaped frame (aspect > 1.15) but NO real spine dip -- a
    single page shot wide, or a book-spine photo. Plain-aspect alone
    triggers is_two inside detect_pages, but strong_spine stays False."""
    return _page_on_background(w, h)


def _pipe():
    cfg = PipelineConfig()
    return FolioPipeline(cfg, segmenter=HalfSplitSegmenter(),
                         counter=OneCounter(), orienter=UprightOrienter())


def test_strong_spine_auto_corrects_to_two_folios():
    img = _strong_gutter_image()
    res = _pipe().process_image("x.jpg", img)
    assert res.error is None, res.error
    assert res.page_count == PageCount.TWO
    assert len(res.folios) == 2
    for f in res.folios:
        assert "count_auto_corrected_to_two" in f.review_reasons
        assert f.needs_review is True
        assert "count_disagrees_with_spine" not in f.review_reasons


def test_weak_landscape_only_flags_never_splits():
    img = _landscape_no_gutter_image()
    res = _pipe().process_image("x.jpg", img)
    assert res.error is None, res.error
    assert res.page_count == PageCount.ONE          # never auto-split
    assert len(res.folios) == 1
    f = res.folios[0]
    assert "count_disagrees_with_spine" in f.review_reasons
    assert "count_auto_corrected_to_two" not in f.review_reasons
