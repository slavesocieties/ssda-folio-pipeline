"""Tests for the landscape orientation pre-pass and the low-text review gate
(stub models, no torch/GPU)."""
import numpy as np
from folio.config import PipelineConfig
from folio.pipeline import FolioPipeline
from folio.schemas import PageBox, PageCount


class OneCounter:
    def predict(self, image):
        return PageCount.ONE, 0.99


class FullSegmenter:
    """One page box covering the central region; rectangular mask."""
    def detect(self, image, max_pages=1):
        h, w = image.shape[:2]
        return [PageBox(20, 20, w - 20, h - 20, 0.98)]
    def segment(self, image, boxes):
        h, w = image.shape[:2]
        m = np.zeros((h, w), np.uint8)
        b = boxes[0]
        m[b.y1:b.y2, b.x1:b.x2] = 1
        return [m]


class UprightOrienter:
    def predict_probs(self, image):
        return np.array([0.95, 0.02, 0.02, 0.01])


class CoarseStub:
    """Returns a fixed 4-way distribution to drive the pre-pass."""
    def __init__(self, probs):
        self.probs = np.asarray(probs, float)
    def predict_probs(self, image):
        return self.probs


def _page(h=420, w=300, fill=235):
    return np.full((h, w, 3), fill, np.uint8)


def _pipe(coarse=None):
    cfg = PipelineConfig()
    p = FolioPipeline(cfg, segmenter=FullSegmenter(), counter=OneCounter(),
                      orienter=UprightOrienter())
    p.coarse_orienter = coarse
    return p


def test_prepass_rotates_sideways_scan():
    # argmax = 1 (90deg), confident -> whole image uprighted, pre_rotation_k set
    pipe = _pipe(CoarseStub([0.1, 0.7, 0.1, 0.1]))
    res = pipe.process_image("x.jpg", _page(h=300, w=420))  # landscape input
    assert res.pre_rotation_k == (-1) % 4 == 3


def test_prepass_leaves_portrait_scan():
    # argmax = 0 (upright) -> no pre-rotation
    pipe = _pipe(CoarseStub([0.9, 0.03, 0.05, 0.02]))
    res = pipe.process_image("x.jpg", _page())
    assert res.pre_rotation_k == 0


def test_prepass_ignores_180():
    # argmax = 2 (upside down) is left to the per-folio stage, not pre-rotated
    pipe = _pipe(CoarseStub([0.1, 0.1, 0.7, 0.1]))
    res = pipe.process_image("x.jpg", _page())
    assert res.pre_rotation_k == 0


def test_low_text_gate_flags_blank_page():
    pipe = _pipe()
    res = pipe.process_image("blank.jpg", _page(fill=240))  # near-blank
    assert res.folios
    f = res.folios[0]
    assert f.needs_review
    assert "low_text_for_orientation" in f.review_reasons


def test_text_dense_page_not_low_text_flagged():
    img = _page()
    img[40:380:2, 40:260] = 30          # dense horizontal ink -> high text_frac
    pipe = _pipe()
    res = pipe.process_image("dense.jpg", img)
    assert res.folios
    assert "low_text_for_orientation" not in res.folios[0].review_reasons
