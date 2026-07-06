#!/usr/bin/env python3
"""fetch_weights.py — download the trained model weights into ./weights/.

The weights are too large to track in git, so they're published as assets on a
GitHub Release. This pulls the four files the default (approach B) crop pipeline
needs. Standard library only — no extra dependencies.

    python tools/fetch_weights.py            # download missing weights
    python tools/fetch_weights.py --force    # re-download all
    python tools/fetch_weights.py --list     # just print what/where

Set FOLIO_WEIGHTS_BASE_URL to override the download location (e.g. a private
mirror or an S3/HTTPS endpoint hosting the same filenames).
"""
import argparse
import os
import sys
import urllib.request
from pathlib import Path

REPO = "slavesocieties/ssda-folio-pipeline"
TAG = "weights-v1"
DEFAULT_BASE = f"https://github.com/{REPO}/releases/download/{TAG}"

# The four weights the default crop path loads. (SAM/RT-DETR are only for the
# optional 'neural' path and are fetched separately by tools/setup_weights.)
WEIGHTS = [
    "folio_seg_unet.pt.ts.pt",       # learned page segmenter — drives the crop
    "orientation4_convnextv2.pt",    # 4-way orientation
    "folio_count_convnextv2.pt",     # one/two/reject count
    "blank_convnextv2.pt",           # content vs. blank
]

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"


def _download(url, dest):
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                pct = 100 * done / total
                sys.stdout.write(f"\r  {dest.name}: {done>>20}/{total>>20} MB ({pct:.0f}%)")
                sys.stdout.flush()
    tmp.replace(dest)
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    ap.add_argument("--list", action="store_true", help="print sources and exit")
    args = ap.parse_args()

    base = os.environ.get("FOLIO_WEIGHTS_BASE_URL", DEFAULT_BASE).rstrip("/")
    WEIGHTS_DIR.mkdir(exist_ok=True)
    if args.list:
        print(f"target dir: {WEIGHTS_DIR}")
        for w in WEIGHTS:
            print(f"  {base}/{w}")
        return 0

    print(f"weights -> {WEIGHTS_DIR}\n(source: {base})")
    for w in WEIGHTS:
        dest = WEIGHTS_DIR / w
        if dest.exists() and not args.force:
            print(f"  {w}: already present ({dest.stat().st_size>>20} MB) — skip")
            continue
        try:
            _download(f"{base}/{w}", dest)
        except Exception as e:  # network / 404 / auth
            print(f"\n[!] failed to fetch {w}: {type(e).__name__}: {e}", file=sys.stderr)
            print(f"    try downloading it manually to {dest}", file=sys.stderr)
            return 1
    print("done. The default `folio` crop pipeline can now find its weights.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
