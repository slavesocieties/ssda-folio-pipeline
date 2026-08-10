#!/usr/bin/env python3
"""eval_partial_spread.py — evaluate a rescue for the "partially open book" crops.

The failure
-----------
When a volume is photographed with the facing leaf only partly open, the trained
count head confidently returns ONE folio (correctly — only one complete folio is
present), and the segmenter then keeps the contiguous paper, which includes the
partial facing leaf. The crop comes out too wide: a folio is ~0.70 w/h, these land
at 0.95-1.37, tripping `unexpected_aspect` (config ceiling 0.95).

Note this is NOT decided by detect_pages' aspect/spine thresholds — production runs
the trained counter, so those thresholds never see these images.

The rescue
----------
Fire only on a one-folio crop that is already too wide: look for the spine shadow
(a prominent dark column valley) anywhere in the middle of the crop, cut there, and
keep the side carrying more text. Because it is conditional on an aspect that is
already out of range, it cannot alter crops that are currently fine.

Usage
-----
  python tools/eval_partial_spread.py 265705
  python tools/eval_partial_spread.py 265705 --limit 30 --outdir eval_out
  python tools/eval_partial_spread.py 265705 --control   # also run on GOOD crops
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
import cv2
import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts"))

PORTRAIT_LO, PORTRAIT_HI = 0.4, 0.95     # folio.config.portrait_aspect_range

# Single source of truth: the promoted stage module.
from folio.stages.partial_spread import (spine_score, find_spine,  # noqa: E402
                                         rescue_partial_spread, split_quad)


def _s3(profile):
    return boto3.Session(profile_name=profile or None).client("s3")


def load_cfg():
    import run_prefix as rp
    return rp.load_config()


def review_report(cfg, volume):
    tc = cfg["transcripts"]
    key = f"review/{tc.get('prefix','')}{volume}.json"
    return json.loads(_s3(tc["profile"]).get_object(
        Bucket=tc["bucket"], Key=key)["Body"].read())


def fetch_crops(cfg, keys):
    cc = cfg["crops"]
    s3 = _s3(cc["profile"])

    def one(k):
        try:
            b = s3.get_object(Bucket=cc["bucket"], Key=k)["Body"].read()
            return k, cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            return k, None
    with ThreadPoolExecutor(max_workers=16) as ex:
        return dict(ex.map(one, keys))



def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("volume")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--outdir", default="eval_partial_out")
    ap.add_argument("--montages", type=int, default=10)
    ap.add_argument("--control", action="store_true",
                    help="also run the rule over crops that were NOT flagged")
    args = ap.parse_args(argv)

    cfg = load_cfg()
    rep = review_report(cfg, args.volume)
    flagged = [r["crop"] for r in rep["review"]
               if "unexpected_aspect" in r["review_reasons"]]
    if args.limit:
        flagged = flagged[:args.limit]
    print(f"volume {args.volume}: {rep['crops_total']} crops, "
          f"{rep['flagged_for_review']} flagged, "
          f"{len(flagged)} with unexpected_aspect (evaluating these)\n")

    crops = fetch_crops(cfg, flagged)
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    fired = inrange = 0
    infos = {}
    montaged = 0
    for k, img in crops.items():
        if img is None:
            continue
        kept, _q, info = rescue_partial_spread(img)
        infos[k] = info
        if info["fired"]:
            fired += 1
            inrange += bool(0.65 <= info["aspect_after"] <= PORTRAIT_HI)
            if montaged < args.montages:
                hh = 900
                a = cv2.resize(img, (int(img.shape[1] * hh / img.shape[0]), hh))
                b = cv2.resize(kept, (int(kept.shape[1] * hh / kept.shape[0]), hh))
                pad = np.full((hh, 24, 3), 255, np.uint8)
                cv2.imwrite(str(out / f"{Path(k).stem}_rescue.jpg"),
                            np.hstack([a, pad, b]), [cv2.IMWRITE_JPEG_QUALITY, 88])
                montaged += 1

    n = len(infos)
    print(f"{'metric':<38}{'count':>8}")
    print("-" * 46)
    print(f"{'flagged crops evaluated':<38}{n:>8}")
    print(f"{'rescue fired':<38}{fired:>8}  ({100*fired/max(n,1):.0f}%)")
    print(f"{'aspect in folio band 0.65-0.95':<38}{inrange:>8}  ({100*inrange/max(n,1):.0f}%)")

    befores = [i["aspect_before"] for i in infos.values()]
    afters = [i["aspect_after"] for i in infos.values() if i.get("fired")]
    if befores:
        print(f"\naspect before: min={min(befores):.3f} median={np.median(befores):.3f} "
              f"max={max(befores):.3f}")
    if afters:
        print(f"aspect after : min={min(afters):.3f} median={np.median(afters):.3f} "
              f"max={max(afters):.3f}")
    gx = [i["gutter_x_frac"] for i in infos.values() if i.get("gutter_x_frac")]
    if gx:
        print(f"gutter position (fraction of width): min={min(gx):.2f} "
              f"median={np.median(gx):.2f} max={max(gx):.2f}")

    if args.control:
        good = [r for r in rep.get("blank", [])] and []
        ok_keys = [k for k in _all_crop_keys(cfg, args.volume)
                   if k not in set(flagged)][:args.limit or 60]
        ctrl = fetch_crops(cfg, ok_keys)
        cf = sum(1 for _k, im in ctrl.items() if im is not None
                 and rescue_partial_spread(im)[2]["fired"])
        print(f"\ncontrol (unflagged crops): {len(ctrl)} tested, rescue fired on {cf}")

    (out / "rescue_results.json").write_text(json.dumps(infos, indent=2))
    print(f"\nmontages + results -> {out.resolve()}")
    return 0


def _all_crop_keys(cfg, volume):
    cc = cfg["crops"]
    s3 = _s3(cc["profile"])
    keys = []
    for page in s3.get_paginator("list_objects_v2").paginate(
            Bucket=cc["bucket"], Prefix=cc.get("prefix", "") + volume):
        for o in page.get("Contents", []):
            if o["Key"].lower().endswith(".jpg"):
                keys.append(o["Key"])
    return sorted(keys)


if __name__ == "__main__":
    raise SystemExit(main())
