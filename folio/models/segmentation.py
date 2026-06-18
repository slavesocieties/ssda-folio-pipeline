"""Stage 3 - page detection + segmentation: RT-DETR (page boxes) -> SAM 2.1
(pixel-precise masks). Heavy deps are imported lazily so the rest of the
package imports cleanly on machines without GPU/torch.

The wrapper exposes:
    detect(image) -> list[PageBox]                 # 1 or 2 page boxes
    segment(image, boxes) -> list[mask]            # full-res binary masks
Both accept batches for GPU efficiency where the backends allow it.
"""
from __future__ import annotations

from typing import List, Optional
import numpy as np

from ..schemas import PageBox
from ..config import ModelConfig


class PageSegmenter:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self._detector = None
        self._sam = None

    # --- lazy backends -----------------------------------------------------
    def _load(self):
        if self._sam is not None:
            return
        import torch  # noqa
        from ultralytics import RTDETR  # detector
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from sam2.build_sam import build_sam2

        self._torch = torch
        self._detector = RTDETR(self.cfg.detector_weights)
        sam_model = build_sam2(self.cfg.sam_model_cfg, self.cfg.sam_checkpoint,
                               device=self.cfg.device)
        self._sam = SAM2ImagePredictor(sam_model)

    # --- inference ---------------------------------------------------------
    def detect(self, image: np.ndarray, max_pages: int = 2) -> List[PageBox]:
        """Return up to ``max_pages`` page boxes sorted left-to-right."""
        self._load()
        res = self._detector.predict(image, imgsz=self.cfg.detector_size,
                                     verbose=False)[0]
        boxes = []
        for b in res.boxes:
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
            boxes.append(PageBox(x1, y1, x2, y2, float(b.conf[0])))
        boxes.sort(key=lambda p: -p.score)
        boxes = boxes[:max_pages]
        boxes.sort(key=lambda p: p.x1)  # left-to-right
        return boxes

    def segment(self, image: np.ndarray, boxes: List[PageBox]) -> List[np.ndarray]:
        """Pixel-precise full-res binary masks, one per box. The SAM image
        embedding is computed once and reused across the (<=2) box prompts."""
        self._load()
        self._sam.set_image(image)  # encoder runs once; embedding cached
        masks = []
        for box in boxes:
            m, scores, _ = self._sam.predict(
                box=np.array(box.as_tuple(), dtype=np.float32)[None],
                multimask_output=False,
            )
            masks.append((m[0] > 0).astype(np.uint8))
        return masks

    def full_page_mask(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Convenience: union mask of all detected pages (for skew/gutter)."""
        boxes = self.detect(image)
        if not boxes:
            return None
        masks = self.segment(image, boxes)
        out = np.zeros(image.shape[:2], dtype=np.uint8)
        for m in masks:
            out |= m
        return out
