"""Core driver for turning SSDA scans into single-folio, upright, cropped images.

This is the reusable engine behind the ``folio`` CLI and the desktop GUI: it
builds the recommended hybrid pipeline (with the trained 4-way orientation head,
the landscape pre-pass, the adaptive-threshold deskew, and the low-text review
gate), then runs it over a single image, a folder, or an S3 prefix.

Everything here is import-safe without torch (models are built lazily) and the
small helpers (`find_legacy_weights`, `parse_s3`, `build_pipeline`) are unit
tested in ``tests/``.
"""
from __future__ import annotations

import asyncio
import csv
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2

from .config import PipelineConfig
from .pipeline import FolioPipeline

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
_REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- env
def auto_device() -> str:
    """cuda if a working CUDA torch is present, else cpu."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def find_legacy_weights(explicit: Optional[str] = None,
                        input_path: Optional[str] = None) -> Optional[str]:
    """Locate the legacy .pth folder so the common case needs no flag.

    Search order: explicit arg, $FOLIO_LEGACY_WEIGHTS, ./legacy_weights, the repo
    and its parent, and a ``legacy_weights`` folder next to the input.
    """
    cands: List[Path] = []
    if explicit:
        cands.append(Path(explicit))
    if os.environ.get("FOLIO_LEGACY_WEIGHTS"):
        cands.append(Path(os.environ["FOLIO_LEGACY_WEIGHTS"]))
    if input_path:
        here = Path(input_path)
        base = here if here.is_dir() else here.parent
        cands.append(base / "legacy_weights")
    cands += [Path.cwd() / "legacy_weights", _REPO / "legacy_weights",
              _REPO.parent / "legacy_weights"]
    for c in cands:
        if c and (c / "unet_folio_split.pth").exists():
            return str(c)
    return None


def parse_s3(uri: str) -> Tuple[str, str]:
    """``s3://bucket/prefix`` -> ``(bucket, prefix)``."""
    if not uri.startswith("s3://"):
        raise ValueError(f"not an s3 uri: {uri}")
    bucket, _, prefix = uri[len("s3://"):].partition("/")
    return bucket, prefix


def resolve_orient_weight(cfg: PipelineConfig) -> None:
    """Make the default relative orientation weight path absolute (repo-rooted)
    so the tool works from any working directory."""
    if not Path(cfg.model.orientation_weights).is_absolute():
        cfg.model.orientation_weights = str(_REPO / cfg.model.orientation_weights)


# ---------------------------------------------------------------------- pipeline
def build_pipeline(cfg: PipelineConfig, legacy_weights: Optional[str],
                   prepass: bool = True) -> Tuple[FolioPipeline, str]:
    """Recommended hybrid pipeline; falls back to classical when weights absent.

    Returns ``(pipeline, mode_description)``.
    """
    have_legacy = bool(legacy_weights) and Path(legacy_weights).is_dir() and \
        (Path(legacy_weights) / "unet_folio_split.pth").exists()
    orient_ok = Path(cfg.model.orientation_weights).exists()

    if have_legacy:
        from .models.hybrid import build_hybrid_pipeline
        pipe = build_hybrid_pipeline(cfg, legacy_weights, device=cfg.model.device)
        mode = "hybrid"
        if orient_ok:
            from .models.classifiers import OrientationClassifier
            head = OrientationClassifier(cfg.model)
            pipe.orienter = head
            mode = "hybrid + trained-4way-orient"
            if prepass:
                pipe.coarse_orienter = head
                mode += " + landscape pre-pass"
        return pipe, mode

    from .models.classical import (ClassicalSegmenter, ClassicalCounter,
                                    ClassicalOrienter)
    pipe = FolioPipeline(cfg, segmenter=ClassicalSegmenter(cfg.model),
                         counter=ClassicalCounter(), orienter=ClassicalOrienter())
    return pipe, "classical (no legacy weights found)"


def make_config(device: Optional[str] = None,
                orient_weights: Optional[str] = None) -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.model.device = device or auto_device()
    if orient_weights:
        cfg.model.orientation_weights = str(Path(orient_weights))
    else:
        resolve_orient_weight(cfg)
    return cfg


# ----------------------------------------------------------------- local results
MANIFEST_FIELDS = ["source", "folio", "page_count", "count_conf", "pre_rotation_k",
                   "rotation_deg", "orient_conf", "text_frac", "needs_review",
                   "review_reasons"]


def _row(source: str, folio: str, res, f) -> dict:
    return {
        "source": source, "folio": folio, "page_count": res.page_count.value,
        "count_conf": round(res.count_conf, 3), "pre_rotation_k": res.pre_rotation_k,
        "rotation_deg": round(f.rotation_deg, 2), "orient_conf": round(f.orientation_conf, 3),
        "text_frac": round(f.text_frac, 4), "needs_review": f.needs_review,
        "review_reasons": ";".join(f.review_reasons),
    }


@dataclass
class RunStats:
    images: int = 0
    folios: int = 0
    review: int = 0
    errors: int = 0
    manifest: List[dict] = field(default_factory=list)


def write_image_result(out: Path, source_name: str, stem: str, res,
                       stats: RunStats) -> None:
    """Persist one ImageResult: sidecar + crops (+ review copies) + manifest rows."""
    (out / "sidecars" / f"{stem}.json").write_text(json.dumps(res.sidecar(), indent=2))
    if res.error and not res.folios:
        stats.errors += 1
        return
    for f in res.folios:
        sfx = f"-{f.label}" if f.label else ""
        rel = f"{stem}{sfx}.jpg"
        dst = out / "folios" / rel
        cv2.imwrite(str(dst), f.crop)
        if f.needs_review:
            shutil.copyfile(dst, out / "review" / rel)
            stats.review += 1
        stats.folios += 1
        stats.manifest.append(_row(source_name, rel, res, f))


def list_images(input_path) -> List[Path]:
    """Image files under a path, or expanded from a list of files/folders
    (used by the GUI for drag-and-drop of mixed items)."""
    if isinstance(input_path, (list, tuple)):
        out: List[Path] = []
        for item in input_path:
            out.extend(list_images(item))
        return out
    p = Path(input_path)
    if p.is_file():
        return [p] if p.suffix.lower() in IMAGE_EXTS else []
    if p.is_dir():
        return sorted(q for q in p.iterdir() if q.suffix.lower() in IMAGE_EXTS)
    return []


def _ensure_dirs(out: Path) -> None:
    for sub in ("folios", "sidecars", "review"):
        (out / sub).mkdir(parents=True, exist_ok=True)


def write_manifest(out: Path, stats: RunStats) -> None:
    with open(out / "manifest.csv", "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        wtr.writeheader()
        wtr.writerows(stats.manifest)


# ---------------------------------------------------------------- parallel worker
_WORKER = {}


def _worker_init(legacy, prepass, orient_weights):
    cfg = make_config(device="cpu", orient_weights=orient_weights)  # CPU per worker
    pipe, _ = build_pipeline(cfg, legacy, prepass=prepass)
    _WORKER["pipe"] = pipe


def _worker_run(path_str):
    p = Path(path_str)
    img = cv2.imread(str(p))
    if img is None:
        return (p.name, p.stem, None)
    res = _WORKER["pipe"].process_image(p.name, img)
    return (p.name, p.stem, res)


# ------------------------------------------------------------------- orchestration
def run_local(input_path, out, *, device=None, legacy=None, prepass=True,
              orient_weights=None, jobs=1, resume=False, limit=None,
              on_start=None, on_item=None) -> Tuple[RunStats, str]:
    """Process a local image or folder. Returns ``(stats, mode)``.

    ``jobs>1`` runs a CPU process pool (the work is CPU-bound; the GPU models are
    light). ``resume`` skips images whose primary crop already exists.
    ``on_start(n, mode, device)`` and ``on_item(i, n, name, res)`` are optional
    progress callbacks (used by the GUI and CLI).
    """
    out = Path(out)
    _ensure_dirs(out)
    files = list_images(input_path)
    if limit:
        files = files[:limit]
    if resume:
        files = [p for p in files
                 if not (out / "folios" / f"{p.stem}.jpg").exists()
                 and not (out / "folios" / f"{p.stem}-A.jpg").exists()]

    discover_hint = (input_path if isinstance(input_path, (str, Path))
                     else (files[0] if files else None))
    legacy = find_legacy_weights(legacy, str(discover_hint) if discover_hint else None)
    stats = RunStats(images=len(files))

    if jobs and jobs > 1 and len(files) > 1:
        mode = f"parallel x{jobs} (cpu workers)"
        if on_start:
            on_start(len(files), mode, "cpu")
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(jobs, initializer=_worker_init,
                      initargs=(legacy, prepass, orient_weights)) as pool:
            for i, (name, stem, res) in enumerate(
                    pool.imap_unordered(_worker_run, [str(p) for p in files]), 1):
                if res is None:
                    stats.errors += 1
                else:
                    write_image_result(out, name, stem, res, stats)
                if on_item:
                    on_item(i, len(files), name, res)
    else:
        cfg = make_config(device=device, orient_weights=orient_weights)
        pipe, mode = build_pipeline(cfg, legacy, prepass=prepass)
        if on_start:
            on_start(len(files), mode, cfg.model.device)
        for i, p in enumerate(files, 1):
            img = cv2.imread(str(p))
            if img is None:
                stats.errors += 1
                res = None
            else:
                res = pipe.process_image(p.name, img)
                write_image_result(out, p.name, p.stem, res, stats)
            if on_item:
                on_item(i, len(files), p.name, res)

    write_manifest(out, stats)
    return stats, mode


def run_s3(input_uri, out_uri, *, device=None, legacy=None, prepass=True,
           orient_weights=None, region=None, limit=None, shard=None,
           resume=False) -> Tuple[dict, str]:
    """Stream-process an S3 prefix back to S3. Returns ``(stats, mode)``.
    ``shard=(i, n)`` processes only worker i-of-n's keys (for EC2/Batch fan-out);
    ``resume`` skips inputs whose output already exists."""
    cfg = make_config(device=device, orient_weights=orient_weights)
    cfg.s3.input_bucket, cfg.s3.input_prefix = parse_s3(input_uri)
    cfg.s3.output_bucket, out_prefix = parse_s3(out_uri)
    cfg.s3.output_prefix = out_prefix or "folios/"
    if region:
        cfg.s3.region = region
    legacy = find_legacy_weights(legacy, None)
    pipe, mode = build_pipeline(cfg, legacy, prepass=prepass)
    stats = asyncio.run(pipe.run_s3(limit=limit, shard=shard, resume=resume))
    return stats, mode
