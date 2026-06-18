"""folio_process — turn ANY SSDA scan into clean single-folio, upright, cropped
page image(s). This is the production entry point for Task 1 (pre-processing).

It runs the recommended **hybrid** configuration: the trained legacy models reused
inside our better split / orient / crop logic, plus

  * a coarse orientation PRE-PASS that uprights sideways (landscape) scans before
    segmentation, and
  * a trained 4-way ConvNeXt-V2 orientation head + sub-degree deskew,

and writes, for every input image:
  * folios/<stem>[-A|-B].jpg   the upright portrait single-folio crop(s)
  * sidecars/<stem>.json       provenance (counts, rotation, confidences, review)
  * review/<stem>...           a copy of any crop flagged for human review
  * manifest.csv               one row per output folio

Usage
-----
  # one image (weights auto-discovered, device auto -> just works)
  python tools/folio_process.py page.jpg

  # a whole folder, custom output dir
  python tools/folio_process.py /scans --out /out

  # S3 batch (the full corpus); --limit N for a safe dry run on a subset
  python tools/folio_process.py s3://ssda-raw/vols/ --out s3://ssda-folios/folios/ --limit 20

Legacy weights are auto-discovered (./legacy_weights, the repo, $FOLIO_LEGACY_WEIGHTS,
or next to the input); pass --legacy-weights to override. If none are found it
falls back to the dependency-free classical config so it still produces output
for any image (lower quality). Device is auto-selected (CUDA if available, else CPU).
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from folio.config import PipelineConfig
from folio.pipeline import FolioPipeline

_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
_REPO = Path(__file__).resolve().parents[1]


def _auto_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _find_legacy_weights(explicit, input_path) -> str | None:
    """Locate the legacy .pth folder so the common case needs no flag."""
    cands = []
    if explicit:
        cands.append(Path(explicit))
    if os.environ.get("FOLIO_LEGACY_WEIGHTS"):
        cands.append(Path(os.environ["FOLIO_LEGACY_WEIGHTS"]))
    here = Path(input_path)
    base = here if here.is_dir() else here.parent
    cands += [Path.cwd() / "legacy_weights", _REPO / "legacy_weights",
              _REPO.parent / "legacy_weights", base / "legacy_weights"]
    for c in cands:
        if c and (c / "unet_folio_split.pth").exists():
            return str(c)
    return None


def _parse_s3(uri: str):
    """s3://bucket/prefix -> (bucket, prefix)."""
    rest = uri[len("s3://"):]
    bucket, _, prefix = rest.partition("/")
    return bucket, prefix


def _build_pipeline(cfg: PipelineConfig, legacy_weights, prepass: bool):
    """Recommended hybrid pipeline; falls back to classical if weights absent."""
    have_legacy = legacy_weights and Path(legacy_weights).is_dir() and \
        (Path(legacy_weights) / "unet_folio_split.pth").exists()
    orient_ok = Path(cfg.model.orientation_weights).exists()

    if have_legacy:
        from folio.models.hybrid import build_hybrid_pipeline
        pipe = build_hybrid_pipeline(cfg, legacy_weights, device=cfg.model.device)
        mode = "hybrid"
        if orient_ok:
            from folio.models.classifiers import OrientationClassifier
            head = OrientationClassifier(cfg.model)
            pipe.orienter = head                       # trained 4-way head
            mode = "hybrid + trained-4way-orient"
            if prepass:
                pipe.coarse_orienter = head            # upright sideways scans first
                mode += " + landscape pre-pass"
        return pipe, mode

    # fallback: no weights -> classical (CPU, no GPU/weights needed)
    from folio.models.classical import (ClassicalSegmenter, ClassicalCounter,
                                         ClassicalOrienter)
    pipe = FolioPipeline(cfg, segmenter=ClassicalSegmenter(cfg.model),
                         counter=ClassicalCounter(), orienter=ClassicalOrienter())
    return pipe, "classical (no weights found)"


def main():
    ap = argparse.ArgumentParser(description="Single-folio, upright, cropped derivatives from SSDA scans.")
    ap.add_argument("input", help="an image file, a directory of images, or s3://bucket/prefix")
    ap.add_argument("--out", default="./folio_out", help="output dir, or s3://bucket/prefix for S3 mode")
    ap.add_argument("--legacy-weights", default=None,
                    help="dir with legacy .pth (auto-discovered if omitted)")
    ap.add_argument("--orient-weights", default=None,
                    help="trained 4-way orientation .pt (default: config path)")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    ap.add_argument("--no-prepass", action="store_true",
                    help="disable the sideways/landscape orientation pre-pass")
    ap.add_argument("--region", default=None, help="AWS region for S3 mode")
    ap.add_argument("--limit", type=int, default=None,
                    help="process at most N images (safe dry run, esp. for S3)")
    args = ap.parse_args()

    cfg = PipelineConfig()
    cfg.model.device = args.device or _auto_device()
    if args.orient_weights:
        cfg.model.orientation_weights = str(Path(args.orient_weights))
    elif not Path(cfg.model.orientation_weights).is_absolute():
        # resolve the default relative weight against the repo so it works anywhere
        cfg.model.orientation_weights = str(_REPO / cfg.model.orientation_weights)

    legacy = _find_legacy_weights(args.legacy_weights, args.input)

    # ---- S3 batch mode (the 750k run) ----------------------------------------
    if args.input.startswith("s3://"):
        if not args.out.startswith("s3://"):
            print("S3 input requires an s3:// --out", file=sys.stderr); sys.exit(2)
        cfg.s3.input_bucket, cfg.s3.input_prefix = _parse_s3(args.input)
        cfg.s3.output_bucket, out_prefix = _parse_s3(args.out)
        cfg.s3.output_prefix = out_prefix or "folios/"
        if args.region:
            cfg.s3.region = args.region
        pipe, mode = _build_pipeline(cfg, legacy, prepass=not args.no_prepass)
        print(f"config : {mode}   device={cfg.model.device}")
        print(f"S3     : s3://{cfg.s3.input_bucket}/{cfg.s3.input_prefix} "
              f"-> s3://{cfg.s3.output_bucket}/{cfg.s3.output_prefix}\n")
        try:
            stats = asyncio.run(pipe.run_s3(limit=args.limit))
        except Exception as e:
            print(f"S3 run failed: {type(e).__name__}: {e}", file=sys.stderr)
            print("  check AWS credentials (env / ~/.aws), bucket names, and region.",
                  file=sys.stderr)
            sys.exit(1)
        print(f"\ndone: {stats}")
        return

    pipe, mode = _build_pipeline(cfg, legacy, prepass=not args.no_prepass)

    inp = Path(args.input)
    if not inp.exists():
        print(f"input not found: {inp}", file=sys.stderr)
        sys.exit(2)
    files = ([inp] if inp.is_file() else
             sorted(p for p in inp.iterdir() if p.suffix.lower() in _EXT))
    if not files:
        print(f"no images found at {inp}", file=sys.stderr)
        sys.exit(2)
    if args.limit:
        files = files[:args.limit]

    out = Path(args.out)
    for sub in ("folios", "sidecars", "review"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    print(f"config : {mode}   device={cfg.model.device}")
    print(f"input  : {len(files)} image(s) -> {out}\n")

    manifest = []
    n_folios = n_review = n_err = 0
    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            n_err += 1
            print(f"  {p.name:28s} UNREADABLE")
            continue
        res = pipe.process_image(p.name, img)
        (out / "sidecars" / f"{p.stem}.json").write_text(json.dumps(res.sidecar(), indent=2))
        if res.error and not res.folios:
            n_err += 1
            print(f"  {p.name:28s} {res.error}")
            continue
        for f in res.folios:
            sfx = f"-{f.label}" if f.label else ""
            rel = f"{p.stem}{sfx}.jpg"
            dst = out / "folios" / rel
            cv2.imwrite(str(dst), f.crop)
            if f.needs_review:
                shutil.copyfile(dst, out / "review" / rel)
                n_review += 1
            n_folios += 1
            manifest.append({
                "source": p.name, "folio": rel, "page_count": res.page_count.value,
                "count_conf": round(res.count_conf, 3),
                "pre_rotation_k": res.pre_rotation_k,
                "rotation_deg": round(f.rotation_deg, 2),
                "orient_conf": round(f.orientation_conf, 3),
                "text_frac": round(f.text_frac, 4),
                "needs_review": f.needs_review,
                "review_reasons": ";".join(f.review_reasons),
            })
        flag = "  REVIEW" if any(f.needs_review for f in res.folios) else ""
        print(f"  {p.name:28s} {res.page_count.value:11s} folios={len(res.folios)}{flag}")

    with open(out / "manifest.csv", "w", newline="") as fh:
        if manifest:
            wtr = csv.DictWriter(fh, fieldnames=list(manifest[0].keys()))
            wtr.writeheader(); wtr.writerows(manifest)

    print(f"\ndone: {n_folios} folio crop(s) from {len(files)} image(s); "
          f"{n_review} flagged for review; {n_err} error(s).")
    print(f"  crops    -> {out/'folios'}")
    print(f"  review   -> {out/'review'}")
    print(f"  manifest -> {out/'manifest.csv'}")


if __name__ == "__main__":
    main()
