"""OCR orientation review-rescue (folio.stages.ocr_orient + pipeline wiring).

Uses a STUB reader (no EasyOCR / no torch) so it runs anywhere. The stub scores
'legibility' from the pixels themselves, so np.rot90(img, 2) genuinely changes the
verdict — mirroring how real OCR scores a legible page above its upside-down self.
"""
import numpy as np
import pytest

from folio.stages.ocr_orient import OCRUpDownVerifier


class StubReader:
    """readtext() 'legibility' == brightness of the TOP half of the image. A real
    upright page reads well (content-top at image-top); rotating it 180 moves that
    content to the bottom, dropping the score. This is rotation-consistent: it
    tracks the actual pixels, so np.rot90(img, 2) genuinely changes the verdict."""
    def readtext(self, bgr):
        h = bgr.shape[0]
        top = float(bgr[: h // 2].mean())
        conf = min(top / 255.0, 1.0)              # in [0,1]
        return [((0, 0, 0, 0), "x" * 10, conf)]   # conf*len tokens


def _img(bright_top):
    """Portrait crop with a bright band in the top (upright) or bottom (upside-down)."""
    im = np.zeros((40, 30, 3), np.uint8)
    if bright_top:
        im[:10] = 255
    else:
        im[-10:] = 255
    return im


def test_verifier_prefers_more_legible_orientation():
    v = OCRUpDownVerifier(reader=StubReader())
    # bright top == upright; rot180 darkens the top -> upright wins, no flip
    should_flip, margin = v.flip_verdict(_img(bright_top=True))
    assert should_flip is False
    assert margin > 0.5


def test_verifier_recommends_flip_when_flip_more_legible():
    v = OCRUpDownVerifier(reader=StubReader())
    # bright bottom == upside-down; rot180 brings it to the top -> flip recommended
    should_flip, margin = v.flip_verdict(_img(bright_top=False))
    assert should_flip is True
    assert margin > 0.5


def test_verifier_no_text_gives_no_signal():
    class Empty:
        def readtext(self, bgr):
            return []
    v = OCRUpDownVerifier(reader=Empty())
    should_flip, margin = v.flip_verdict(_img(True))
    assert should_flip is False and margin == 0.0


def test_unavailable_when_no_reader_and_no_easyocr(monkeypatch):
    # force the lazy import to fail -> verifier reports unavailable, never raises
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "easyocr":
            raise ImportError("no easyocr")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    v = OCRUpDownVerifier(reader=None)
    assert v.available is False
