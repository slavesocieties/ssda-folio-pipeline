"""Central, typed configuration for the whole pipeline.

Every tunable lives here so behaviour is reproducible and overridable from a
single YAML/env source. Defaults are production-sane.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Tuple
import os


@dataclass
class S3Config:
    input_bucket: str = "ssda-raw"
    output_bucket: str = "ssda-folios"
    input_prefix: str = ""
    output_prefix: str = "folios/"
    review_prefix: str = "review/"
    region: str = "us-east-1"
    # streaming concurrency (Stage 0). Auto-tuned at runtime around these.
    list_page_size: int = 1000
    download_concurrency: int = 32
    upload_concurrency: int = 16
    # bounded queues = backpressure; keeps RAM flat regardless of dataset size.
    decode_queue_size: int = 256
    result_queue_size: int = 256
    multipart_threshold_mb: int = 64


@dataclass
class ModelConfig:
    device: str = "cuda"
    dtype: str = "fp16"                  # fp16 | fp32 | bf16
    compile: bool = True                 # torch.compile / TensorRT graph
    gpu_batch_size: int = 16             # micro-batch size for the GPU
    # weights (resolved from a local cache or s3://… at startup)
    detector_weights: str = "weights/rtdetr_page.pt"
    sam_checkpoint: str = "weights/sam2.1_hiera_large.pt"
    sam_model_cfg: str = "sam2.1_hiera_l.yaml"
    folio_count_weights: str = "weights/folio_count_convnextv2.pt"
    orientation_weights: str = "weights/orientation4_convnextv2.pt"
    blank_weights: str = "weights/blank_convnextv2.pt"   # content/blank classifier
    folio_seg_weights: str = "weights/folio_seg_unet.pt.ts.pt"  # learned page segmenter (TorchScript)
    dewarp_weights: str = "weights/uvdoc.pt"
    # inference image sizes (geometry is always applied at full res)
    detector_size: int = 1024
    classifier_size: int = 384


@dataclass
class GeometryConfig:
    # outward crop margin so frayed edges / marginalia are never clipped,
    # expressed as a fraction of the page diagonal (Stage 3 boundary math).
    crop_margin_frac: float = 0.045
    # trim leftover black warp-padding / uniform background border from the
    # finished crop (ink-guarded, never clips content). Safety net for loose masks.
    trim_background: bool = True
    # tighten the finished crop to the detected text region (learned CRAFT
    # detector) for the supervisor's "tight" look. No-ops gracefully when EasyOCR
    # is unavailable or no text is found (so it can never clip content).
    tight_crop: bool = False
    tight_crop_margin_frac: float = 0.012
    # Anti-over-crop: a sparse page (text not filling the sheet) can crop square.
    # If the crop is wider than this w/h, EXTEND it (vertically, into the page)
    # toward a portrait shape so the full folio is kept -- only ever adds area,
    # never crops tighter (no info lost). 0 disables.
    max_crop_aspect: float = 0.80
    # white-out every non-folio pixel (background, facing-page sliver, binding)
    # using the precise learned page mask, leaving only the folio. Needs the
    # learned segmenter; no-op without it.
    mask_background: bool = True
    # Stage 4 gutter search
    gutter_band_margin_frac: float = 0.04   # widen inter-page gap by this much
    seam_smoothness: float = 0.3            # DP diagonal penalty weight
    energy_weights: Tuple[float, float, float] = (0.45, 0.35, 0.20)  # dark, smooth, gap
    # Stage 5 fine skew
    skew_max_deg: float = 15.0
    skew_step_deg: float = 0.25
    # Stage 6 gating
    dewarp_curvature_thresh: float = 0.03


@dataclass
class QualityConfig:
    min_count_conf: float = 0.80
    min_orientation_conf: float = 0.70
    min_page_area_frac: float = 0.10        # crop must cover >=10% of source
    portrait_aspect_range: Tuple[float, float] = (0.4, 0.95)  # h>w expected
    # below this fraction of text-ink pixels the 4-way orientation head is
    # unreliable (sparse/near-blank pages, where it can be confidently 180-wrong)
    # -> flag for review instead of trusting it. Calibrated on the sample set to
    # catch the known sparse-page failures; recall-oriented (a few correct sparse
    # pages are flagged too). Re-tune against a labelled validation set.
    min_text_frac_for_orient: float = 0.075


@dataclass
class PipelineConfig:
    s3: S3Config = field(default_factory=S3Config)
    model: ModelConfig = field(default_factory=ModelConfig)
    geom: GeometryConfig = field(default_factory=GeometryConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    version: str = "folio-1.0.0"
    enable_dewarp: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        cfg = cls()
        cfg.s3.input_bucket = os.getenv("FOLIO_INPUT_BUCKET", cfg.s3.input_bucket)
        cfg.s3.output_bucket = os.getenv("FOLIO_OUTPUT_BUCKET", cfg.s3.output_bucket)
        cfg.s3.region = os.getenv("AWS_REGION", cfg.s3.region)
        cfg.model.device = os.getenv("FOLIO_DEVICE", cfg.model.device)
        return cfg
