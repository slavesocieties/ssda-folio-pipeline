"""Hybrid pipeline with a swappable orientation head, for A/B-ing the legacy
0/180-only orient classifier against our trained 4-way ConvNeXt-V2 head.

  python tools/run_hybrid_orient_demo.py "<dir>" --legacy-weights <pth_dir> \
      --orient {legacy|neural} --outdir out

Writes crops/ + montage_crops.jpg + report.json. With --orient neural it loads
cfg.model.orientation_weights (weights/orientation4_convnextv2.pt by default).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from folio.config import PipelineConfig
from folio.models.hybrid import build_hybrid_pipeline
from folio.models.classifiers import OrientationClassifier
from tools.run_samples import montage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--legacy-weights", required=True)
    ap.add_argument("--orient", choices=["legacy", "neural"], default="neural")
    ap.add_argument("--outdir", default="./hybrid_orient_out")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--prepass", action="store_true",
                    help="coarse orient the whole image before segmenting (fixes sideways scans)")
    args = ap.parse_args()

    out = Path(args.outdir); (out / "crops").mkdir(parents=True, exist_ok=True)
    cfg = PipelineConfig(); cfg.model.device = args.device

    pipe = build_hybrid_pipeline(cfg, args.legacy_weights, device=args.device)
    if args.orient == "neural":
        head = OrientationClassifier(cfg.model)            # trained 4-way head
        pipe.orienter = head
        if args.prepass:
            pipe.coarse_orienter = head                    # upright sideways scans first

    files = sorted(p for p in Path(args.input).iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff"))
    crops, report = [], []
    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            continue
        res = pipe.process_image(p.name, img)
        for f in res.folios:
            sfx = f"-{f.label}" if f.label else ""
            cv2.imwrite(str(out / "crops" / f"{p.stem}{sfx}.jpg"), f.crop)
            crops.append(f.crop)
        report.append({"file": p.name, "count": res.page_count.value,
                       "folios": [{"label": f.label, "rotation_deg": round(f.rotation_deg, 2),
                                   "orient_conf": round(f.orientation_conf, 3)}
                                  for f in res.folios]})
        rot = ",".join(f"{f.rotation_deg:+.0f}" for f in res.folios) or "-"
        print(f"{p.name:28s} {res.page_count.value:11s} rot_deg=[{rot}]")

    mc = montage(crops, cols=6)
    if mc is not None:
        cv2.imwrite(str(out / "montage_crops.jpg"), mc)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\n[{args.orient}] wrote {len(crops)} crops -> {out}")


if __name__ == "__main__":
    main()
