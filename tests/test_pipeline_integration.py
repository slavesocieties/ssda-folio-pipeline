"""End-to-end wiring test using stub models (no torch/GPU required).
Proves the orchestrator splits a two-folio spread at the TRUE off-center
gutter and emits two upright crops + a provenance sidecar."""
import numpy as np
from folio.config import PipelineConfig
from folio.pipeline import FolioPipeline
from folio.schemas import PageBox, PageCount


def _two_folio_image(w=600, h=400, gutter_x=230):
    img = np.full((h, w, 3), 235, np.uint8)
    img[:, max(gutter_x - 4, 0):gutter_x + 4] = 30  # dark off-center spine
    return img, gutter_x


class StubCounter:
    def predict(self, image):
        return PageCount.TWO, 0.99


class StubSegmenter:
    """Two page boxes on either side of x=230, full-height masks per side."""
    def __init__(self, gutter_x):
        self.g = gutter_x
    def detect(self, image, max_pages=2):
        h, w = image.shape[:2]
        return [PageBox(8, 8, self.g - 5, h - 8, 0.97),
                PageBox(self.g + 5, 8, w - 8, h - 8, 0.96)]
    def segment(self, image, boxes):
        h, w = image.shape[:2]
        out = []
        for b in boxes:
            m = np.zeros((h, w), np.uint8)
            m[b.y1:b.y2, b.x1:b.x2] = 1
            out.append(m)
        return out


class StubOrienter:
    def predict_probs(self, image):
        return np.array([0.95, 0.02, 0.02, 0.01])  # already upright


def test_two_folio_split_off_center():
    img, gx = _two_folio_image(gutter_x=230)
    cfg = PipelineConfig()
    pipe = FolioPipeline(cfg, segmenter=StubSegmenter(gx),
                         counter=StubCounter(), orienter=StubOrienter())
    res = pipe.process_image("vol/DSC_0013.JPG", img)

    assert res.error is None, res.error
    assert res.page_count == PageCount.TWO
    assert len(res.folios) == 2
    assert {f.label for f in res.folios} == {"A", "B"}

    # gutter found near the TRUE spine (230), NOT the midpoint (300)
    seam = np.array(res.gutter_seam)
    assert abs(float(np.median(seam)) - 230) <= 10
    assert abs(float(np.median(seam)) - 300) > 50   # legacy W//2 would be 300

    # crops exist and are non-trivial
    for f in res.folios:
        assert f.crop is not None and f.crop.size > 0

    # sidecar serialises cleanly (provenance)
    sc = res.sidecar()
    assert sc["page_count"] == "two_folios"
    assert sc["gutter_seam_summary"]["mean_x"] < 280
