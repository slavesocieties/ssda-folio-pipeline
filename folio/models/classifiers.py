"""Stage 1 (folio count + layout) and Stage 5 (4-way orientation) classifiers.

Both are compact ConvNeXt-V2-Tiny heads. The count head additionally consumes
two cheap geometric priors (aspect ratio + central gutter-valley ratio) to fix
the legacy "skinny two-folio / fat one-folio" misfires. Torch is imported
lazily; ``geometric_priors`` is pure NumPy and unit-testable on its own.
"""
from __future__ import annotations

from typing import List, Tuple
import cv2
import numpy as np

from ..config import ModelConfig
from ..schemas import PageCount


def geometric_priors(image: np.ndarray) -> Tuple[float, float]:
    """(aspect_ratio, gutter_valley_ratio).

    aspect_ratio = W / H.
    gutter_valley_ratio = min column-intensity in the central third divided by
    the median column intensity. A deep central valley (ratio << 1) implies a
    two-page opening regardless of overall shape.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = gray.shape[:2]
    col = gray.mean(axis=0).astype(np.float32)
    col = cv2.GaussianBlur(col.reshape(1, -1), (0, 0), sigmaX=max(w * 0.01, 1)).ravel()
    med = float(np.median(col)) + 1e-6
    lo, hi = int(0.40 * w), int(0.60 * w)
    central_min = float(col[lo:hi].min()) if hi > lo else med
    return (w / float(h), central_min / med)


class FolioCountClassifier:
    """one_folio / two_folios / reject."""
    CLASSES = [PageCount.ONE, PageCount.TWO, PageCount.REJECT]

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        self._torch = torch
        self._model = torch.jit.load(self.cfg.folio_count_weights,
                                     map_location=cfg_device(self.cfg)).eval()

    def predict(self, image: np.ndarray) -> Tuple[PageCount, float]:
        self._load()
        torch = self._torch
        t = _preprocess(image, self.cfg.classifier_size)
        ar, valley = geometric_priors(image)
        priors = torch.tensor([[ar, valley]], dtype=t.dtype, device=t.device)
        with torch.inference_mode():
            logits = self._model(t.to(cfg_device(self.cfg)), priors)
            probs = torch.softmax(logits, dim=1)[0].float().cpu().numpy()
        idx = int(probs.argmax())
        return self.CLASSES[idx], float(probs[idx])


class OrientationClassifier:
    """4-way: probability of current rotation being [0, 90, 180, 270]."""

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        self._torch = torch
        self._model = torch.jit.load(self.cfg.orientation_weights,
                                     map_location=cfg_device(self.cfg)).eval()

    def predict_probs(self, image: np.ndarray) -> np.ndarray:
        self._load()
        torch = self._torch
        t = _preprocess(image, self.cfg.classifier_size)
        with torch.inference_mode():
            logits = self._model(t.to(cfg_device(self.cfg)))
            probs = torch.softmax(logits, dim=1)[0].float().cpu().numpy()
        return probs


def cfg_device(cfg: ModelConfig) -> str:
    return cfg.device


def _preprocess(image: np.ndarray, size: int):
    import torch
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else \
        cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    arr = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    t = torch.from_numpy(arr.transpose(2, 0, 1))[None]
    return t
