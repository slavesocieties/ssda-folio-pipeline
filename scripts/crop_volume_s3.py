#!/usr/bin/env python3
"""crop_volume_s3.py — the production cropping runner.

Pull every image of one VOLUME from a source S3 bucket, crop it with the folio
pipeline, and push the cropped single-folio images to a target S3 bucket — across
two (possibly different) AWS accounts, with the image bytes streamed in memory and
NEVER written to local disk.

This is the "master script": it does the S3 orchestration and calls the pipeline
(`folio.process.build_pipeline` -> `pipe.process_image`). To change the model, edit
the `folio/` package or swap the weights in `weights/` (see fetch_weights.py) — this
script does not need to change.

Credentials are AWS *profile names* (from `aws configure` / ~/.aws), never keys, so
nothing secret lives in the script or the command line.

Examples
--------
  # crop volume 176899 from the source account into the target account
  python scripts/crop_volume_s3.py \
      --source-profile ssda-read  --source-bucket legacy-ssda-jpgs-... \
      --volume 176899 \
      --target-profile ssda-write --target-bucket ssda-archivault-crops-...

  python scripts/crop_volume_s3.py ... --jobs 16              # CPU-parallel at scale
  python scripts/crop_volume_s3.py ... --write-coords          # also push coord JSON
  python scripts/crop_volume_s3.py ... --dry-run               # list keys, do nothing
  python scripts/crop_volume_s3.py ... --white-out             # approach A (default is B)

Scale-out: this runs ONE volume per invocation, so the natural way to parallelise
across machines is one process per volume. On a single box, `--jobs N` fans the work
across N CPU worker processes (the crop is CPU-bound; the GPU models are light), which
is the big throughput lever.
"""
import argparse
import json
import sys
from pathlib import Path

import boto3
import cv2
import numpy as np

# Import the pipeline whether or not the package is pip-installed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from folio.process import make_config, build_pipeline, find_legacy_weights  # noqa: E402

IMG_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

# ---- per-worker state (built once per process) ------------------------------
_W = {}


def _s3(profile):
    return boto3.Session(profile_name=profile).client("s3")


def _worker_init(cfg_kw, legacy, src_profile, tgt_profile):
    cfg = make_config(**cfg_kw)
    pipe, mode = build_pipeline(cfg, legacy)
    _W.update(pipe=pipe, mode=mode,
              src=_s3(src_profile), tgt=_s3(tgt_profile))


def _crop_and_push(task):
    """Fetch one source image (in memory), crop it, push the crops. Returns a dict."""
    key, src_bucket, tgt_bucket, tgt_prefix, coords_prefix = task
    src, tgt, pipe = _W["src"], _W["tgt"], _W["pipe"]
    try:
        body = src.get_object(Bucket=src_bucket, Key=key)["Body"].read()
        img = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return {"key": key, "crops": 0, "error": "unreadable"}
        stem = Path(key).stem
        res = pipe.process_image(Path(key).name, img)
        out = []
        for f in res.folios:
            if f.crop is None:
                continue
            name = f"{stem}{('-' + f.label) if f.label else ''}.jpg"
            ok, buf = cv2.imencode(".jpg", f.crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                continue
            tgt.put_object(Bucket=tgt_bucket, Key=tgt_prefix + name,
                           Body=buf.tobytes(), ContentType="image/jpeg")
            if coords_prefix is not None:
                coord = {
                    "crop": name, "source_key": key,
                    "source_size": [int(img.shape[1]), int(img.shape[0])],
                    "crop_quad_norm": f.crop_quad_norm,
                    "corner_order": "TL,TR,BR,BL (x,y ratios of source image)",
                    "label": f.label, "page_count": res.page_count.value,
                    "rotation_deg": f.rotation_deg, "is_blank": f.is_blank,
                    "needs_review": f.needs_review,
                }
                tgt.put_object(Bucket=tgt_bucket, Key=coords_prefix + name + ".json",
                               Body=json.dumps(coord).encode(), ContentType="application/json")
            out.append(name)
        return {"key": key, "crops": len(out), "review": any(f.needs_review for f in res.folios),
                "error": res.error}
    except Exception as e:  # keep the run alive; report the failure
        return {"key": key, "crops": 0, "error": f"{type(e).__name__}: {e}"}


def list_volume_keys(s3, bucket, prefix):
    keys = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].lower().endswith(IMG_EXT):
                keys.append(obj["Key"])
    return sorted(keys)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-profile", required=True, help="AWS profile to READ the source bucket")
    ap.add_argument("--source-bucket", required=True)
    ap.add_argument("--volume", required=True, help="volume id (used as the source key prefix)")
    ap.add_argument("--source-prefix", default=None,
                    help="explicit source key prefix (default: the volume id)")
    ap.add_argument("--target-profile", required=True, help="AWS profile to WRITE the target bucket")
    ap.add_argument("--target-bucket", required=True)
    ap.add_argument("--target-prefix", default="folios/", help="key prefix for crops (default folios/)")
    ap.add_argument("--write-coords", action="store_true",
                    help="also push a coord JSON per crop (provenance -> original image)")
    ap.add_argument("--coords-prefix", default="coords/", help="key prefix for coord JSON")
    ap.add_argument("--white-out", action="store_true",
                    help="approach A (white-out background) instead of the default tight crop (B)")
    ap.add_argument("--jobs", type=int, default=1, help="CPU worker processes (crop is CPU-bound)")
    ap.add_argument("--limit", type=int, default=None, help="process at most N images (dry run/testing)")
    ap.add_argument("--dry-run", action="store_true", help="list what would be processed, do nothing")
    ap.add_argument("--device", default=None, help="cuda|cpu (default: auto for --jobs 1, cpu for workers)")
    args = ap.parse_args(argv)

    prefix = args.source_prefix if args.source_prefix is not None else args.volume
    src = _s3(args.source_profile)
    keys = list_volume_keys(src, args.source_bucket, prefix)
    if args.limit:
        keys = keys[:args.limit]
    print(f"volume {args.volume}: {len(keys)} image(s) under "
          f"s3://{args.source_bucket}/{prefix}")
    print(f"  -> crops to s3://{args.target_bucket}/{args.target_prefix}"
          + (f" + coords {args.coords_prefix}" if args.write_coords else ""))
    if not keys:
        print("nothing to do."); return 0
    if args.dry_run:
        for k in keys[:10]:
            print("  ", k)
        print(f"[dry-run] {len(keys)} images; approach {'A (white-out)' if args.white_out else 'B (tight)'};"
              f" jobs={args.jobs}. Nothing pushed.")
        return 0

    coords_prefix = args.coords_prefix if args.write_coords else None
    legacy = find_legacy_weights(None, str(Path(__file__).resolve().parent.parent))
    tasks = [(k, args.source_bucket, args.target_bucket, args.target_prefix, coords_prefix) for k in keys]

    n_img = n_crop = n_err = n_rev = 0
    if args.jobs and args.jobs > 1:
        cfg_kw = dict(device=args.device or "cpu",
                      mask_background=args.white_out, crop_to_folio_mask=not args.white_out)
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.jobs, initializer=_worker_init,
                      initargs=(cfg_kw, legacy, args.source_profile, args.target_profile)) as pool:
            for i, r in enumerate(pool.imap_unordered(_crop_and_push, tasks), 1):
                n_img += 1; n_crop += r["crops"]; n_err += bool(r.get("error")); n_rev += bool(r.get("review"))
                if r.get("error"):
                    print(f"  [!] {r['key']}: {r['error']}")
                if i % 50 == 0:
                    print(f"  {i}/{len(tasks)}  crops={n_crop} errors={n_err}", flush=True)
    else:
        _worker_init(dict(device=args.device, mask_background=args.white_out,
                          crop_to_folio_mask=not args.white_out),
                     legacy, args.source_profile, args.target_profile)
        print(f"config: {_W['mode']}")
        for i, t in enumerate(tasks, 1):
            r = _crop_and_push(t)
            n_img += 1; n_crop += r["crops"]; n_err += bool(r.get("error")); n_rev += bool(r.get("review"))
            if r.get("error"):
                print(f"  [!] {r['key']}: {r['error']}")
            if i % 25 == 0:
                print(f"  {i}/{len(tasks)}  crops={n_crop} errors={n_err}", flush=True)

    print(f"\ndone: {n_crop} crop(s) from {n_img} image(s); {n_rev} flagged for review; {n_err} error(s).")
    print(f"  target: s3://{args.target_bucket}/{args.target_prefix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
