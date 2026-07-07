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
    # safety margin (fraction of page diagonal) the two-folio split is allowed to
    # extend PAST the gutter seam, so an imprecise spine split never clips the
    # folio's own inner text. A thin sliver past the gutter is removed by white-out.
    gutter_safety_frac: float = 0.02
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
    # APPROACH A (white-out): blank every non-folio pixel (background, facing-page
    # sliver, binding) to white using the precise learned page mask, leaving only the
    # folio on white. OFF by default -- the white-out ERASES pixels it judges non-page,
    # so on hard pages (ink bleed-through, water damage) it can over-crop and eat real
    # content. Enable with --white-out. Needs the learned segmenter.
    mask_background: bool = False
    # APPROACH B (tight bounding-box crop) -- THE DEFAULT (supervisor-approved). Instead
    # of blanking the background, CROP the finished crop to the bounding box of the same
    # safe learned folio-half mask (hull-union-full-mask + margin). Excludes the facing
    # page / binding (outside the folio half) while keeping every folio pixel -- NO pixel
    # is ever altered, only the rectangle is tightened, so it cannot erase text (verified
    # 0px folio-text loss). Needs the learned segmenter; ignored when mask_background is on.
    crop_to_folio_mask: bool = True
    # cap the output aspect ratio (long:short) so no crop is more extreme than the
    # transcription backend accepts (SSDA HTR/Gemini wants <= 10:24, i.e. 24/10).
    # Padded with white, never cropped, so no content is lost. 0 disables.
    max_output_ratio: float = 2.4
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
    # OCR orientation review-rescue: on a folio flagged ``low_orientation_conf``,
    # an independent OCR pass decides the up-vs-down flip. Two gates, because the
    # two actions carry very different risk (measured on a held-out labelled set,
    # folio.stages.ocr_orient):
    #   CONFIRM  (OCR agrees with the head -> clear the flag, keep orientation):
    #     low risk (the head's answer is kept), ~100% precise at margin >= 0.20.
    #   OVERRIDE (OCR disagrees -> rotate the crop 180): high risk (a wrong flip
    #     silently breaks a correct page AND clears its review flag, so an undetected
    #     upside-down page reaches paid transcription). 0.30 was held-out-validated
    #     100%-precise on the design set (cleaner tight crops); on harder/faded real
    #     corpora the flips cluster near that edge on pages OCR reads weakly, so the
    #     shipped default is a slightly more conservative 0.35 -- it defers the
    #     riskiest ~quartile of flips (0.30-0.35 band) to human review instead of
    #     acting on them. Lower it toward 0.30 for higher coverage on legible corpora.
    # A weak margin is treated as no signal and the folio stays flagged.
    ocr_confirm_margin: float = 0.20
    ocr_override_margin: float = 0.35


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
