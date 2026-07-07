"""Structured records that flow through the pipeline and are emitted as
per-image JSON sidecars for provenance, QA sampling and active learning."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Optional, Tuple
import numpy as np


class PageCount(str, Enum):
    ONE = "one_folio"
    TWO = "two_folios"
    REJECT = "reject"


@dataclass
class PageBox:
    """One detected page within a source image (axis-aligned, full-res px)."""
    x1: int
    y1: int
    x2: int
    y2: int
    score: float

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass
class FolioResult:
    """A single output page plus everything we decided about it."""
    label: str                       # "A" (recto) / "B" (verso) / "" (single)
    crop: Optional[np.ndarray] = None  # full-res upright portrait crop (BGR)
    rotation_deg: float = 0.0        # total applied rotation (90k + skew)
    orientation_conf: float = 0.0
    mask_area_frac: float = 0.0
    text_frac: float = 0.0           # fraction of text-ink pixels (orient reliability)
    is_blank: bool = False           # blank/non-content page (skip downstream transcription)
    blank_conf: float = 0.0
    needs_review: bool = False
    review_reasons: List[str] = field(default_factory=list)
    # ---- OCR orientation review-rescue (an independent text-legibility signal
    # consulted ONLY on folios the 4-way head flagged low_orientation_conf) ----
    ocr_rescued: bool = False        # OCR resolved the flip -> flag cleared
    ocr_flipped: bool = False        # OCR's verdict rotated the crop 180 to upright
    ocr_margin: float = 0.0          # OCR relative-confidence margin |a-b|/(a+b)
    # ---- provenance: where this folio sits in the ORIGINAL source image, so a
    # transcription can be matched back to its location later ----
    source_size: Optional[List[int]] = None        # original image [width, height] px
    crop_quad_norm: Optional[List[List[float]]] = None  # folio region corners
    #   [TL, TR, BR, BL] as (x, y) ratios in [0,1] of the original image

    def meta(self) -> dict:
        d = asdict(self)
        d.pop("crop", None)
        return d


@dataclass
class ImageResult:
    """Everything produced for one source S3 object."""
    source_key: str
    page_count: PageCount = PageCount.REJECT
    count_conf: float = 0.0
    global_skew_deg: float = 0.0
    pre_rotation_k: int = 0                    # whole-image quarter-turns applied
                                              # before count/segment (landscape fix)
    gutter_seam: Optional[List[int]] = None   # x per row, full-res (two-folio)
    folios: List[FolioResult] = field(default_factory=list)
    error: Optional[str] = None
    version: str = ""

    def sidecar(self) -> dict:
        return {
            "source_key": self.source_key,
            "page_count": self.page_count.value,
            "count_conf": round(self.count_conf, 4),
            "global_skew_deg": round(self.global_skew_deg, 3),
            "pre_rotation_k": self.pre_rotation_k,
            "gutter_seam_summary": _seam_summary(self.gutter_seam),
            "folios": [f.meta() for f in self.folios],
            "error": self.error,
            "version": self.version,
        }


def _seam_summary(seam: Optional[List[int]]) -> Optional[dict]:
    if not seam:
        return None
    arr = np.asarray(seam)
    return {
        "min_x": int(arr.min()),
        "max_x": int(arr.max()),
        "mean_x": float(arr.mean()),
        "curvature_px": int(arr.max() - arr.min()),  # straight gutter -> ~0
    }
