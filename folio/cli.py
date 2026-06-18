"""Command-line entry point.

Examples
--------
# Process a whole S3 bucket (streaming, GPU-batched, writes crops back to S3):
    python -m folio.cli s3 --input-bucket ssda-raw --output-bucket ssda-folios

# Process a single local image (debug / golden-set):
    python -m folio.cli local path/to/DSC_0013.JPG --outdir ./out
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from .config import PipelineConfig
from .pipeline import FolioPipeline


def _cmd_s3(args):
    cfg = PipelineConfig.from_env()
    if args.input_bucket:
        cfg.s3.input_bucket = args.input_bucket
    if args.output_bucket:
        cfg.s3.output_bucket = args.output_bucket
    if args.prefix:
        cfg.s3.input_prefix = args.prefix
    cfg.enable_dewarp = args.dewarp
    pipe = FolioPipeline(cfg)
    stats = asyncio.run(pipe.run_s3())
    print(json.dumps(stats, indent=2))


def _cmd_local(args):
    import cv2
    cfg = PipelineConfig.from_env()
    cfg.enable_dewarp = args.dewarp
    pipe = FolioPipeline(cfg)
    pipe._ensure_models()
    img = cv2.imread(args.image)
    res = pipe.process_image(Path(args.image).name, img)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem
    for f in res.folios:
        suffix = f"-{f.label}" if f.label else ""
        sub = "review" if f.needs_review else "output"
        (outdir / sub).mkdir(exist_ok=True)
        cv2.imwrite(str(outdir / sub / f"{stem}{suffix}.jpg"), f.crop)
    (outdir / f"{stem}.json").write_text(json.dumps(res.sidecar(), indent=2))
    print(json.dumps(res.sidecar(), indent=2))


def main(argv=None):
    p = argparse.ArgumentParser(prog="folio", description="Folio pipeline")
    sub = p.add_subparsers(required=True)

    s = sub.add_parser("s3", help="stream-process an S3 bucket")
    s.add_argument("--input-bucket")
    s.add_argument("--output-bucket")
    s.add_argument("--prefix", default="")
    s.add_argument("--dewarp", action="store_true")
    s.set_defaults(func=_cmd_s3)

    l = sub.add_parser("local", help="process a single local image")
    l.add_argument("image")
    l.add_argument("--outdir", default="./out")
    l.add_argument("--dewarp", action="store_true")
    l.set_defaults(func=_cmd_local)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
