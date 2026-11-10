"""Automated, ground-truth-free pass/fail verdicts for a `folio.cli` output dir.

This is NOT a substitute for human QA -- it's a floor, not a ceiling. It can
catch geometrically/physically implausible crops without needing a human to
look at every image, but it CANNOT tell a plausible-but-subtly-wrong crop
(e.g. a spine a few percent off, excluded marginalia) from a correct one. Any
pass-rate this script reports is a PROXY metric: necessary-condition checks,
not a certification of correctness. Treat it as "nothing is obviously broken",
and still spot-check a human-reviewed sample before trusting a number from it.

Checks (none rely on the pipeline's own confidence scores, so they're an
independent cross-check, not circular):
  - needs_review          the pipeline's own flag (baseline floor)
  - tiny_crop             shorter output side < min_side_px
  - extreme_aspect        long:short ratio outside a generous sanity bound
  - degenerate_brightness crop is ~solid white or ~solid black (empty mask)
  - sibling_orientation_mismatch
                          two-folio spread's A/B sides disagree on rotation
                          by something other than plausible shared skew --
                          both sides were photographed in one shot, so they
                          must share orientation; a mismatch means one is
                          wrong, with no ground truth needed to know that.

Usage:
  python tools/auto_verify.py <out_dir> [--min-side-px 300] [--json report.json]
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def _rotation_mismatch(ra: float, rb: float, tol_deg: float = 15.0) -> bool:
    """True if two sibling rotations disagree by something other than shared
    skew -- i.e. neither (ra - rb) nor (ra - rb - 180) is near 0 mod 360."""
    d = (ra - rb) % 360.0
    d = min(d, 360.0 - d)
    d180 = abs(d - 180.0)
    return d > tol_deg and d180 > tol_deg


def verify(out_dir: Path, min_side_px: int = 300,
          max_long_short: float = 3.2, min_long_short: float = 1.02) -> dict:
    rows = list(csv.DictReader(open(out_dir / "manifest.csv", encoding="utf-8")))
    by_source = defaultdict(list)
    for r in rows:
        by_source[r["source"]].append(r)

    verdicts = []          # one per folio
    suspicious = []        # folios worth a human looking at first

    for source, group in by_source.items():
        rotations = {}
        for r in group:
            fname = r["folio"]
            path = out_dir / "folios" / fname
            reasons = []
            im = cv2.imread(str(path))
            if im is None:
                reasons.append("unreadable_output")
            else:
                h, w = im.shape[:2]
                short, long_ = min(h, w), max(h, w)
                if short < min_side_px:
                    reasons.append("tiny_crop")
                ratio = long_ / max(short, 1)
                if ratio > max_long_short:
                    reasons.append("extreme_aspect")
                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                mean, std = float(gray.mean()), float(gray.std())
                if mean > 250 or std < 3:
                    reasons.append("degenerate_brightness")
            if r["needs_review"] == "True":
                reasons.append("needs_review:" + r["review_reasons"])
            rotations[fname] = float(r["rotation_deg"])
            verdicts.append({"source": source, "folio": fname,
                            "passed": len(reasons) == 0, "reasons": reasons})

        # sibling consistency: only meaningful for an actual two-folio split
        names = sorted(rotations)
        if len(names) == 2:
            ra, rb = rotations[names[0]], rotations[names[1]]
            if _rotation_mismatch(ra, rb):
                for v in verdicts[-2:]:
                    v["reasons"].append("sibling_orientation_mismatch")
                    v["passed"] = False
                suspicious.append({"source": source, "check": "sibling_orientation_mismatch",
                                   "rotations": [ra, rb]})

    # surface OCR's own "I suspect this is upside-down but won't act" signal,
    # whether or not it already carries needs_review, as a human-priority item
    sidecar_dir = out_dir / "sidecars"
    if sidecar_dir.exists():
        for sc in sidecar_dir.glob("*.json"):
            data = json.loads(sc.read_text())
            for f in data.get("folios", []):
                if f.get("ocr_suspected_flip"):
                    suffix = f"-{f['label']}" if f.get("label") else ""
                    suspicious.append({"source": data["source_key"],
                                       "check": "ocr_suspected_flip",
                                       "folio": f"{sc.stem}{suffix}.jpg"})

    n = len(verdicts)
    n_pass = sum(v["passed"] for v in verdicts)
    reason_counts = defaultdict(int)
    for v in verdicts:
        for r in v["reasons"]:
            key = r.split(":")[0]
            reason_counts[key] += 1

    return {
        "out_dir": str(out_dir),
        "n_folios": n,
        "n_images": len(by_source),
        "n_pass": n_pass,
        "pass_rate": round(n_pass / n, 4) if n else None,
        "fail_reason_counts": dict(sorted(reason_counts.items(), key=lambda kv: -kv[1])),
        "suspicious_for_human_review": suspicious,
        "verdicts": verdicts,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--min-side-px", type=int, default=300)
    ap.add_argument("--json", type=Path, default=None,
                    help="write the full per-folio report here")
    args = ap.parse_args()

    report = verify(args.out_dir, min_side_px=args.min_side_px)
    print(f"{report['out_dir']}")
    print(f"  images={report['n_images']}  folios={report['n_folios']}")
    print(f"  PROXY pass rate: {report['pass_rate']*100:.1f}%  "
          f"({report['n_pass']}/{report['n_folios']})")
    print(f"  fail reasons: {report['fail_reason_counts']}")
    print(f"  suspicious-for-human items: {len(report['suspicious_for_human_review'])}")
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
        print(f"  full report -> {args.json}")


if __name__ == "__main__":
    main()
