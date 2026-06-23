"""Per-image throughput benchmark + full-corpus projection.

    python tools/benchmark.py <dir> [--n 20] [--device cuda|cpu] [--no-tight-crop]

Builds the pipeline once, warms up on one image, then times process_image over
the rest (decode + full pipeline, the per-image cost a real run pays). Prints
mean sec/image and projects the ~750k corpus on 1 and N workers.
"""
from __future__ import annotations
import argparse, time, statistics, sys
from pathlib import Path
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from folio import process as P

CORPUS = 750_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-tight-crop", action="store_true")
    args = ap.parse_args()

    files = P.list_images(args.dir)[: args.n + 1]
    if len(files) < 2:
        print("need >=2 images"); return 1
    legacy = P.find_legacy_weights(None, str(args.dir))
    cfg = P.make_config(device=args.device, tight_crop=not args.no_tight_crop)
    pipe, mode = P.build_pipeline(cfg, legacy, prepass=True)
    print(f"config: {mode}  device={cfg.model.device}  tight_crop={not args.no_tight_crop}")

    # warmup (model load, CRAFT model, CUDA kernels) — not timed
    img0 = cv2.imread(str(files[0]))
    pipe.process_image(files[0].name, img0)

    times = []
    for p in files[1:]:
        img = cv2.imread(str(p))
        if img is None:
            continue
        t = time.perf_counter()
        pipe.process_image(p.name, img)
        times.append(time.perf_counter() - t)

    mean = statistics.mean(times)
    med = statistics.median(times)
    print(f"\nimages timed: {len(times)}")
    print(f"sec/image: mean={mean:.3f}  median={med:.3f}  "
          f"min={min(times):.3f}  max={max(times):.3f}")
    print(f"throughput: {1/mean:.2f} img/s/worker")
    hrs = CORPUS * mean / 3600
    print(f"\n~750k projection (mean):")
    for w in (1, 8, 16, 32):
        print(f"  {w:2d} worker(s): {hrs/w:8.1f} h  ({hrs/w/24:.1f} days)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
