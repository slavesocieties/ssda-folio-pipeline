# Folio Pipeline

Production system that turns raw archival book/document photographs into
**cropped, upright, portrait single pages** — one image in, the corrected
folio(s) out — locally or at cloud scale.

Replaces the legacy U-Net + `W//2` pipeline. Full design rationale and the
math for every stage live in **[ARCHITECTURE.md](ARCHITECTURE.md)**; the
end-user tool guide is **[README_TOOL.md](README_TOOL.md)**.

## Quick start

```bash
pip install -e . --no-deps            # installs the `folio` + `folio-gui` commands
folio page.jpg                        # one image  -> ./folio_out/folios/
folio /scans --out /out --jobs 6      # a folder, 6 parallel workers
folio-gui                             # drag-and-drop desktop app
```

Weights are auto-discovered; device auto-selects CUDA then CPU; with no weights
it falls back to a dependency-free classical mode so it always produces output.
See **[README_TOOL.md](README_TOOL.md)** for the full tool/GUI/S3 guide.

## What it fixes (vs. the legacy scripts)

| Legacy problem | This system |
|---|---|
| Spine split hardcoded to `W // 2` (breaks on off-center books) | **Seam-energy gutter detection** — finds the true, possibly curved spine |
| From-scratch U-Net @640×960, lossy bbox crop | **RT-DETR → SAM 2.1** masks, oriented crop at original resolution + outward margin so marginalia is never clipped |
| 180°-only flip + fragile aspect-ratio rule | **4-way (0/90/180/270) classifier + projection-profile skew** |
| "Skinny two-folio / fat one-folio" misfires | Count classifier fed **geometric priors** (aspect + central gutter-valley) |
| Download whole bucket to disk, synchronously | **aioboto3 streaming**, in-RAM, GPU-batched, backpressured |

## Layout

```
folio/
  config.py            typed, single-source configuration
  schemas.py           data records + JSON provenance sidecars
  io/s3_async.py       async streaming S3 reader/writer (Stage 0)
  models/
    segmentation.py    RT-DETR detector + SAM 2.1 masks (Stage 3)
    classifiers.py     folio-count + 4-way orientation heads (Stages 1, 5)
  stages/
    spine.py           dynamic gutter detection via seam carving (Stage 4)
    geometry.py        oriented crop + single-warp transform composition (Stages 3, 5)
    orient.py          4-way resolution + fine skew (Stage 5)
  pipeline.py          per-image orchestrator + async S3 runner
  cli.py               `python -m folio.cli {s3,local}`
tests/                 pure-CV math is fully unit-tested (no GPU needed)
```

## Install

```bash
pip install -r requirements.txt
# SAM 2.1 from Meta's repo:
pip install "git+https://github.com/facebookresearch/sam2.git"
```

Place model weights where `folio/config.py:ModelConfig` points (or override via
env / a config file).

## Run

```bash
# Whole S3 bucket -> cropped pages + JSON sidecars back to S3
python -m folio.cli s3 --input-bucket ssda-raw --output-bucket ssda-folios

# Single image (debug / golden set)
python -m folio.cli local sample_images/DSC_0013.JPG --outdir ./out
```

## Test

```bash
pytest -q          # 31 tests; all CPU-only, no GPU/weights needed
```

The tests cover the spine math (synthetic **off-center** and **curved** gutters),
geometry/orientation conventions, the adaptive-threshold deskew (recovers known
tilt, no railing on bordered pages), the landscape orientation **pre-pass**, the
low-text **review gate**, the S3-URI / weight-discovery helpers, and an
end-to-end wiring test with stub models (no GPU required).

## Scale-out

`pipeline.run_s3()` already overlaps S3 I/O with GPU work via asyncio +
threads. For multi-GPU / multi-node, shard the input prefix across **Ray**
actors or **AWS Batch** array jobs, each running the same `run_s3()` over its
shard; a completed-key manifest makes the job idempotent and spot-safe.

## Classical-fallback mode (no weights, no GPU)

For smoke-testing without the neural weights, the pipeline can run with
classical stand-ins that satisfy the same interfaces:

```python
from folio.models.classical import ClassicalSegmenter, ClassicalCounter, ClassicalOrienter
pipe = FolioPipeline(cfg, segmenter=ClassicalSegmenter(cfg.model),
                     counter=ClassicalCounter(), orienter=ClassicalOrienter())
```

- `folio/stages/foreground.py` — background-agnostic page detection (border
  background estimate + Otsu; ink/edge fallback for white-on-white). Spine is
  found by **local-prominence shadow scoring**, not a global-darkest column.
- Run over a folder and get crops + overlays + montages + a JSON report:

```bash
python tools/run_samples.py "/path/to/sample images" --outdir ./sample_out
```

This mode splits two-folio spreads, crops/deskews pages and routes low-confidence
results to review. It is intentionally **below neural accuracy** — in particular
180° orientation is a heuristic guess; the production path uses SAM 2.1 +
fine-tuned heads. Use it to validate plumbing and geometry on real images.

## Production (neural) path — run on a GPU

This sandbox/laptop session has no GPU, so the neural path runs on your machine
or an AWS GPU instance. Two custom heads must be trained first; SAM 2.1 + RT-DETR
use pretrained weights.

### 1. Install
```bash
pip install -r requirements.txt
pip install "git+https://github.com/facebookresearch/sam2.git"
```

### 2. Get foundation weights
```bash
bash tools/setup_weights.sh weights      # downloads SAM 2.1; preps RT-DETR
```

### 3. Train the two heads
Orientation is **self-supervised** — point it at a folder of correctly-oriented
page crops (e.g. vetted pipeline output) and it manufactures 0/90/180/270 labels:
```bash
python -m folio.training.train --task orientation \
    --data /data/upright_pages --out weights/orientation4_convnextv2.pt
```
Folio-count uses labelled folders `one_folio/ two_folios/ reject/` (or `--manifest a.csv`):
```bash
python -m folio.training.train --task count \
    --data /data/count_dataset --out weights/folio_count_convnextv2.pt
python -m folio.training.eval  --task count \
    --data /data/count_holdout --weights weights/folio_count_convnextv2.pt
```
Exports are TorchScript and load directly in `folio.models.classifiers`.

### 4. Run the neural pipeline
On a folder (writes crops/overlays/montages like the classical harness):
```bash
python tools/run_samples.py "/path/to/scans" --neural --weights-dir weights --outdir out
```
At scale over S3 (async streaming + GPU batching, see ARCHITECTURE.md §5):
```bash
python -m folio.cli s3 --input-bucket ssda-raw --output-bucket ssda-folios
```

### Orientation label convention
Label `k ∈ {0,1,2,3}` = the image is rotated `k` CCW quarter-turns from upright
(`np.rot90`). The pipeline restores upright with `quarter_k=(-k)%4`; this
round-trip is verified in `tests/test_orientation_convention.py`. Training and
inference share `folio/training/labels.py`, so the head's outputs map correctly
onto the geometry correction.
