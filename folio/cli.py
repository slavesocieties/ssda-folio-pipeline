"""``folio`` command-line entry point.

Turn ANY SSDA scan into clean single-folio, upright, cropped page image(s).

    folio page.jpg                       # one image (weights auto-discovered)
    folio /scans --out /out --jobs 6     # a folder, 6 parallel workers
    folio s3://ssda-raw/v/ --out s3://ssda-folios/f/ --limit 20   # S3 batch

Outputs (local): folios/ (crops), sidecars/ (provenance json), review/ (flagged
for human QA), manifest.csv. Device auto-selects CUDA, else CPU. With no legacy
weights found it falls back to a dependency-free classical mode.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import process as P


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="folio",
                                 description="Single-folio, upright, cropped derivatives from SSDA scans.")
    ap.add_argument("input", help="an image file, a directory, or s3://bucket/prefix")
    ap.add_argument("--out", default="./folio_out", help="output dir, or s3://bucket/prefix")
    ap.add_argument("--legacy-weights", default=None, help="legacy .pth dir (auto-discovered if omitted)")
    ap.add_argument("--orient-weights", default=None, help="trained 4-way orientation .pt")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    ap.add_argument("--no-prepass", action="store_true", help="disable landscape orientation pre-pass")
    ap.add_argument("--jobs", type=int, default=1, help="parallel CPU workers for folders")
    ap.add_argument("--resume", action="store_true", help="skip images already processed in --out")
    ap.add_argument("--limit", type=int, default=None, help="process at most N images (safe dry run)")
    ap.add_argument("--enhance", action="store_true",
                    help="also write a contrast-boosted *_enhanced.jpg for faint/light-ink pages")
    ap.add_argument("--tight-crop", action="store_true",
                    help="crop tight to the detected text region (learned). Default OFF: "
                         "keep the FULL folio (Daniel's preferred behavior).")
    ap.add_argument("--region", default=None, help="AWS region for S3 mode")
    ap.add_argument("--shard", default=None, metavar="i/N",
                    help="S3 only: process worker i of N (e.g. 0/8) for EC2/Batch fan-out")
    ap.add_argument("--quiet", action="store_true", help="only print the final summary")
    return ap


def _parse_shard(s):
    if not s:
        return None
    i, n = s.split("/")
    i, n = int(i), int(n)
    if not (0 <= i < n):
        raise SystemExit(f"--shard out of range: {s} (need 0 <= i < N)")
    return (i, n)


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)

    # ---- S3 batch mode ----
    if args.input.startswith("s3://"):
        if not args.out.startswith("s3://"):
            print("S3 input requires an s3:// --out", file=sys.stderr)
            return 2
        try:
            stats, mode = P.run_s3(args.input, args.out, device=args.device,
                                   legacy=args.legacy_weights, prepass=not args.no_prepass,
                                   orient_weights=args.orient_weights, region=args.region,
                                   limit=args.limit, shard=_parse_shard(args.shard),
                                   resume=args.resume, tight_crop=args.tight_crop)
        except Exception as e:  # boto/credentials/region issues
            print(f"S3 run failed: {type(e).__name__}: {e}", file=sys.stderr)
            print("  check AWS credentials (env / ~/.aws), bucket names, and region.", file=sys.stderr)
            return 1
        print(f"config : {mode}")
        print(f"done   : {stats}")
        return 0

    # ---- local mode ----
    inp = Path(args.input)
    if not inp.exists():
        print(f"input not found: {inp}", file=sys.stderr)
        return 2

    def on_start(n, mode, device):
        if not args.quiet:
            print(f"config : {mode}   device={device}")
            print(f"input  : {n} image(s) -> {args.out}\n")

    def on_item(i, n, name, res):
        if args.quiet or res is None:
            if res is None and not args.quiet:
                print(f"  [{i}/{n}] {name:28s} UNREADABLE")
            return
        flag = "  REVIEW" if any(f.needs_review for f in res.folios) else ""
        msg = res.error or f"{res.page_count.value:11s} folios={len(res.folios)}{flag}"
        print(f"  [{i}/{n}] {name:28s} {msg}")

    stats, mode = P.run_local(args.input, args.out, device=args.device,
                              legacy=args.legacy_weights, prepass=not args.no_prepass,
                              orient_weights=args.orient_weights, jobs=args.jobs,
                              resume=args.resume, limit=args.limit, enhance=args.enhance,
                              tight_crop=args.tight_crop,
                              on_start=on_start, on_item=on_item)
    out = Path(args.out)
    print(f"\ndone: {stats.folios} folio crop(s) from {stats.images} image(s); "
          f"{stats.review} flagged for review; {stats.errors} error(s).")
    print(f"  crops    -> {out/'folios'}")
    print(f"  review   -> {out/'review'}")
    print(f"  manifest -> {out/'manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

