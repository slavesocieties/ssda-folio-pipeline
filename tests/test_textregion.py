"""Text-region tight-crop logic. The detector is mocked so these run without
EasyOCR / a GPU and stay deterministic."""
import numpy as np
from folio.stages import textregion


def test_none_when_no_text(monkeypatch):
    monkeypatch.setattr(textregion, "detect_boxes", lambda im, **k: [])
    assert textregion.text_crop_box(np.zeros((100, 100, 3), np.uint8)) is None


def test_unions_detected_boxes(monkeypatch):
    # text spanning most of the page -> tight union, nothing clipped
    boxes = [(10, 10, 30, 190), (50, 10, 95, 190)]
    monkeypatch.setattr(textregion, "detect_boxes", lambda im, **k: boxes)
    b = textregion.text_crop_box(np.zeros((200, 200, 3), np.uint8),
                                 margin_frac=0.0, min_keep=0.0)
    assert b == (10, 10, 95, 190)


def test_rejects_tiny_box_under_detection(monkeypatch):
    # detector fired on only a small band (faint page) -> reject, keep looser crop
    monkeypatch.setattr(textregion, "detect_boxes",
                        lambda im, **k: [(40, 40, 60, 60), (42, 45, 62, 65)])
    b = textregion.text_crop_box(np.zeros((1000, 1000, 3), np.uint8), min_keep=0.45)
    assert b is None


def test_outlier_trim_off_by_default_no_clip(monkeypatch):
    # a far bottom line must NOT be dropped by default (would clip real content)
    boxes = [(40, 40, 960, 300), (40, 900, 960, 980)]
    monkeypatch.setattr(textregion, "detect_boxes", lambda im, **k: boxes)
    b = textregion.text_crop_box(np.zeros((1000, 1000, 3), np.uint8), margin_frac=0.0)
    assert b is not None and b[3] >= 980  # bottom line kept


def test_margin_clamped_to_image(monkeypatch):
    monkeypatch.setattr(textregion, "detect_boxes", lambda im, **k: [(5, 5, 95, 95)])
    b = textregion.text_crop_box(np.zeros((100, 100, 3), np.uint8),
                                 margin_frac=0.5, min_keep=0.0)
    assert b == (0, 0, 100, 100)          # never exceeds the image bounds
