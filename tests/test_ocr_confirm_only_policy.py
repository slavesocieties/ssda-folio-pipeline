"""Pins the confirm-only OCR rescue policy: an override (flip) verdict must
never auto-rotate the crop or clear the review flag, only a confirm verdict
may. Regression guard for the silent-wrong-flip failure found on the control
sample (a confident override flipped an already-upright page and shipped it
unflagged)."""
import numpy as np
from folio.config import PipelineConfig
from folio.pipeline import FolioPipeline
from folio.schemas import PageBox, PageCount


class OneCounter:
    def predict(self, image):
        return PageCount.ONE, 0.99


class FullSegmenter:
    def detect(self, image, max_pages=1):
        h, w = image.shape[:2]
        return [PageBox(20, 20, w - 20, h - 20, 0.98)]

    def segment(self, image, boxes):
        h, w = image.shape[:2]
        m = np.zeros((h, w), np.uint8)
        b = boxes[0]
        m[b.y1:b.y2, b.x1:b.x2] = 1
        return [m]


class AmbiguousOrienter:
    """Near-even 0-vs-180 split -> low_orientation_conf, like the real head's
    documented blind spot."""
    def predict_probs(self, image):
        return np.array([0.55, 0.0, 0.45, 0.0])


class StubOCR:
    def __init__(self, should_flip, margin):
        self.available = True
        self._verdict = (should_flip, margin)

    def flip_verdict(self, crop):
        return self._verdict


def _dense_page(h=420, w=300, fill=235):
    img = np.full((h, w, 3), fill, np.uint8)
    img[40:380:2, 40:260] = 30          # dense ink -> clears low_text_for_orientation
    return img


def _pipe(ocr_verifier):
    cfg = PipelineConfig()
    p = FolioPipeline(cfg, segmenter=FullSegmenter(), counter=OneCounter(),
                      orienter=AmbiguousOrienter())
    p.ocr_verifier = ocr_verifier
    return p


def test_confident_override_does_not_flip_or_clear_flag():
    img = _dense_page()
    baseline = FolioPipeline(PipelineConfig(), segmenter=FullSegmenter(),
                             counter=OneCounter(), orienter=AmbiguousOrienter())
    before = baseline.process_image("x.jpg", img.copy()).folios[0]
    assert "low_orientation_conf" in before.review_reasons   # sanity: ambiguous as intended

    pipe = _pipe(StubOCR(should_flip=True, margin=0.9))       # confident override verdict
    res = pipe.process_image("x.jpg", img.copy())
    f = res.folios[0]
    assert "low_orientation_conf" in f.review_reasons          # stays flagged
    assert f.needs_review is True
    assert f.ocr_rescued is False
    assert f.ocr_suspected_flip is True                        # surfaced for human priority
    assert f.rotation_deg == before.rotation_deg                # crop never rotated
    assert np.array_equal(f.crop, before.crop)                  # pixels untouched


def test_confident_confirm_clears_flag():
    img = _dense_page()
    pipe = _pipe(StubOCR(should_flip=False, margin=0.9))       # confident confirm verdict
    res = pipe.process_image("x.jpg", img.copy())
    f = res.folios[0]
    assert "low_orientation_conf" not in f.review_reasons
    assert f.ocr_rescued is True
    assert f.ocr_suspected_flip is False


def test_weak_margin_leaves_flag_untouched():
    img = _dense_page()
    pipe = _pipe(StubOCR(should_flip=False, margin=0.05))      # below confirm gate
    res = pipe.process_image("x.jpg", img.copy())
    f = res.folios[0]
    assert "low_orientation_conf" in f.review_reasons
    assert f.ocr_rescued is False
