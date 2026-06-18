# SSDA Folio Pipeline — project context for Claude Code

## What this is
Production system that turns raw SSDA archival book/document photographs into
clean **single-folio, upright (portrait), cropped** page images, to improve
downstream Archivault output. Corpus size: **~750,000 images**.

This implements "Task 1: Image pre-processing": (1) split two-folio images into
two single-folio derivatives, (2) make every folio upright, (3) crop to the
folio (tight cropping is "nice to have," not strictly required). Text-region
segmentation from the old prototype is intentionally dropped.

## Pipeline (per image)
1. Folio count: one / two / reject
2. Page detect + segment (the "where")
3. Dynamic spine/gutter split for two-folio spreads (NOT a fixed midpoint)
4. Oriented crop at full resolution (margin so marginalia isn't clipped)
5. Orientation: 4-way (0/90/180/270) + fine deskew
6. Write upright portrait crop + JSON provenance sidecar; low-confidence -> review

## Three runnable configurations (same FolioPipeline, swappable components)
- **classical** (no weights, CPU): `tools/run_samples.py "<dir>" --outdir out`
  Stand-ins in `folio/models/classical.py` + `folio/stages/foreground.py`.
  Used for smoke-testing on real images without a GPU.
- **hybrid (RECOMMENDED for the 750k run)**: reuse the trained legacy models
  where good, inside our better split/crop/orient logic; SAM 2.1 only as a
  fallback. `folio/models/hybrid.py::build_hybrid_pipeline`.
  `tools/run_samples.py "<dir>" --hybrid --legacy-weights <pth_dir> [--sam-fallback]`
- **neural (highest ceiling)**: SAM 2.1 + RT-DETR + our trained heads.
  `tools/run_samples.py "<dir>" --neural --weights-dir weights`

## Key modules
- `folio/stages/spine.py`      seam-energy gutter detection (core contribution)
- `folio/stages/geometry.py`   oriented crop + single-warp transform composition
- `folio/stages/orient.py`     4-way resolution + projection-profile deskew
- `folio/stages/foreground.py` classical background-agnostic page detection
- `folio/models/segmentation.py` SAM 2.1 + RT-DETR wrapper (neural)
- `folio/models/classifiers.py`  ConvNeXt-V2 count + 4-way orientation (neural)
- `folio/models/hybrid.py`       legacy-U-Net + legacy classifiers + our logic
- `folio/models/legacy.py`       faithful legacy pipeline (baseline only)
- `folio/io/s3_async.py`         async streaming S3 (no download-all-to-disk)
- `folio/pipeline.py`            orchestrator + async S3 runner
- `folio/training/`              self-supervised orientation + count training
- `tools/run_samples.py`         folder runner -> crops + overlays + montages
- `tools/eval_vs_legacy.py`      head-to-head fresh-vs-legacy scoring
- `ARCHITECTURE.md`              full design rationale + §9 scale + §10 decision

## Conventions
- All geometry computed/applied at FULL resolution; transforms composed into ONE
  warp to avoid compounding interpolation loss.
- Orientation label k = np.rot90 CCW quarter-turns from upright; correction is
  quarter_k=(-k)%4. Verified in tests/test_orientation_convention.py. Keep this
  convention if you touch orientation.
- Legacy `.pth` models are a BASELINE; do not depend on them except in the
  hybrid path. Download IDs in tools/legacy_model_ids.txt.

## Commands
```bash
pip install -r requirements.txt          # core
pip install -r requirements-train.txt    # training (GPU)
pytest -q                                 # 14 tests, CPU-only, must stay green
python tools/run_samples.py "<dir>" --outdir out          # classical smoke test
```
GPU notes: this targets an RTX 5080 (Blackwell) — needs CUDA 12.8+ and a recent
torch build. SAM 2.1: `pip install "git+https://github.com/facebookresearch/sam2.git"`.

## Status (as of handoff)
- Classical path validated on 25 real samples: 24/25 count correct, spines
  found, clean portrait crops. 14 unit tests green.
- Hybrid + neural + training code written and syntax-clean but NOT yet run on a
  GPU (the build sandbox had none).

## Next steps
1. Set up GPU (CUDA 12.8 + torch) on the 5080; `pip install` deps.
2. Download legacy `.pth` (tools/legacy_model_ids.txt) -> run `--hybrid` on samples.
3. Run `tools/eval_vs_legacy.py` to confirm we beat the baseline.
4. Train orientation (self-supervised) + count heads (folio/training/) for `--neural`.
5. Benchmark per-image speed; plan the full 750k run (laptop multi-day or cloud burst).

## Guardrails for future edits
- Keep `pytest -q` green; the spine/geometry/orientation math is covered there.
- Don't reintroduce a fixed-midpoint split or chained rotations.
- Keep model imports lazy so the package imports without torch.
