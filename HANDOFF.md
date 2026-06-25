# Project handoff / context — SSDA Folio Pipeline

Read this first if you're a new session or a fresh agent. It captures the whole
state of the project so you can continue without re-deriving anything.

---

## 1. What this is

A production tool implementing **Task 1: image pre-processing** for the Slave
Societies Digital Archive (SSDA). It turns **any** archival book/document scan
into clean **single-folio, upright (portrait), cropped** page images, to improve
downstream Archivault transcription. Corpus: **~750,000 images**.

Three things per image: (1) split two-folio spreads into two single-folio
derivatives, (2) make every folio upright, (3) crop tightly to the folio.

- **User:** Ronak Mahajan (ronak.mahajan@vanderbilt.edu) — research assistant.
- **Supervisor:** Daniel Genkins (daniel.genkins@gmail.com) — owns the data/S3.
- Production target: **EC2 on AWS** (local-first is fine to start).

## 2. Current status — DONE and working

| Capability | Status | Measured (labelled, in-distribution) |
|---|---|---|
| Folio count (one/two) | ✅ | 100% (also 100% on a 126-img held-out test) |
| Two-folio split | ✅ | 100% |
| Orientation (upright) | ✅ | 98.8% (upright 100 / upside-down 97.5 / landscape 100) |
| Deskew | ✅ | matches exhaustive search exactly, no railing |
| Cropping (page) | ✅ | 0/25 clipped (full page width captured) |
| **Learned folio segmentation** | ✅ | U-Net page-boundary, **val IoU 0.96** (Daniel's 151 masks); precise full-folio crops on ANY background (fixes light-on-light); on by default, falls back to classical mask if smp/weight absent |
| **Full-folio crop (default)** | ✅ | tight to the page, never over-crops; square sparse-page crops 5→0 on Daniel's sample |
| Tight crop (text region) | ✅ opt-in | learned CRAFT crop via `--tight-crop`; meaningful ~0.5–0.6, coverage 1.00 |
| **Per-volume consistency** | ✅ | `tools/volume_normalize.py` → identical size per volume (pad-only); flags size-outliers |
| **Faint-ink enhance** | ✅ | `--enhance` writes a CLAHE `*_enhanced.jpg` for faint pages |
| **Throughput** | ✅ | ~2.6 s/image (RTX 5080, all on); 16 GPU workers ≈ 1.4 days for 750k |
| Tests | ✅ | 51 passing, CPU-only |
| Packaging | ✅ | `pip install -e .` → `folio` + `folio-gui`; tight crop needs `requirements-tight.txt` (EasyOCR) |
| GUI | ✅ | drag-and-drop desktop app, verified |
| S3 batch | ✅ wired | `--shard i/N`, `--resume`; NOT live-tested (needs AWS creds) |

**The one open model item:** sparse/near-blank-page orientation can still be
wrong on rare cross-volume pages (flagged for review, not silently shipped). The
fix is a supervised fine-tune on a sparse image set — **blocked on the supervisor
delivering those images from S3** (see §9).

## 3. Environment & how to run (Windows 11, RTX 5080 laptop)

- **Python (real one):** `C:\Users\mahajar\AppData\Local\Programs\Python\Python312\python.exe`
  (the bare `python`/`python3` on PATH are Windows Store stubs — don't use them).
- **Repo:** `C:\Users\mahajar\Downloads\sample images\ssda-folio-pipeline`
- **Working-dir parent (data lives here):** `C:\Users\mahajar\Downloads\sample images`
- **GPU:** RTX 5080 (Blackwell, sm_120). Torch must be the **cu128** build:
  `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`
  → `torch 2.11.0+cu128`, `torch.cuda.is_available() == True`.
  ⚠️ Re-running `pip install -r requirements.txt` re-installs the **+cpu** build
  and breaks CUDA. Reinstall cu128 if `cuda.is_available()` turns False.
  ⚠️ `pip install -e .` MUST use `--no-deps` to avoid the same clobber.

Run the tool:
```bash
PY="C:\Users\mahajar\AppData\Local\Programs\Python\Python312\python.exe"
$PY -m pytest -q                                  # 51 tests
# tight crop needs EasyOCR (optional): pip install --no-deps easyocr;
#   pip install python-bidi Shapely pyclipper ninja scikit-image PyYAML
# on Windows set PYTHONUTF8=1 so EasyOCR's download bar doesn't crash cp1252
$PY -m folio.cli "<image|folder>" --out <dir>     # or: folio <...> if installed
$PY -m folio.gui                                  # or: folio-gui  (drag-and-drop)
# S3 (needs AWS creds / IAM role):
$PY -m folio.cli s3://in/pre/ --out s3://out/pre/ --shard 0/8 --resume --region us-east-1
```
Weights are auto-discovered; device auto-selects CUDA→CPU; with no legacy weights
it falls back to a dependency-free classical mode.

## 4. The tool — code map

- `folio/cli.py` — flat CLI: `folio <image|dir|s3://>`. Flags: `--out --jobs
  --resume --limit --no-prepass --shard i/N --device --orient-weights --legacy-weights`.
- `folio/gui.py` — tkinter drag-and-drop app (tkinterdnd2 + Pillow); thumbnail
  grid, review-flagged crops outlined red.
- `folio/process.py` — engine: `build_pipeline`, weight auto-discovery, s3-uri
  parse, `run_local` (with parallel CPU `--jobs`), `run_s3`, progress callbacks.
- `folio/pipeline.py` — `FolioPipeline.process_image`: pre-pass → count → detect
  → segment → spine split → per-folio crop/orient/deskew → review gates → sidecar.
- `folio/models/hybrid.py` — recommended config: legacy U-Net segmenter (with
  TERRITORY masking), legacy count adapter, **trained 4-way orient head** swapped in.
- `folio/models/classifiers.py` — `OrientationClassifier`, `FolioCountClassifier`
  (ConvNeXt-V2, TorchScript), `geometric_priors`.
- `folio/stages/`: `spine.py` (seam gutter), `geometry.py` (oriented crop +
  single warp), `orient.py` (4-way resolve + adaptive deskew), `foreground.py`.
- `tools/folio_process.py` — back-compat shim → `folio.cli`.
- `tools/evaluate.py` — end-to-end labelled eval (count/split/orientation/review).
- `tools/find_sparse_folios.py` — scans transcription JSONs → sparse-image list.
- `tools/legacy_model_ids.txt` — Google Drive IDs for the legacy `.pth`.

## 5. Models & weights

- `weights/orientation4_convnextv2.pt` — trained 4-way orientation head (val
  0.983; self-supervised on 3,160 upright pages). Backup: `…prev.pt`.
- `weights/folio_count_convnextv2.pt` — trained count head (100% held-out).
  NOTE: the hybrid pipeline currently uses the LEGACY count adapter, not this —
  both are ~100%, so it's a non-issue, but the trained head exists if wanted.
- `legacy_weights/` (in the data parent dir): `unet_folio_split.pth`,
  `folio_count_classifier.pth`, `upside_down.pth`, `folio_upside_down.pth`.
- Weights are gitignored (large). `tools/setup_weights.sh` / `legacy_model_ids.txt`
  document re-download.

## 6. Training data (in the data parent dir, gitignored)

- `train_data/rightside_up/` (1527 upright single pages — real Drive labels)
- `train_data/upside_down/` (1633 real 180°-rotated single pages)
- `train_data/two_folios/` (280 spreads)
- `train_data/upright_all_512/` (3160 = rightside_up + upside_down-flipped, 512px)
- `train_data/count_dataset/{train,test}/` (count head data + held-out test)
- 25 sample scans: loose `*.jpg` in the data parent dir.
- Training: `python -m folio.training.train --task {orientation|count} --data <dir>
  --out weights/... --epochs 12 --bs 32 --workers 0 --size 384`
  (Windows: `--workers 0`; `--bs` small enough that train//bs ≥ 1.)

## 7. Key bugs found & fixed this project (so you don't re-introduce them)

1. **Orientation framing bug (biggest win): 50% → 98.8%.** The pipeline fed the
   4-way head the *tight oriented crop*, which flips its 0-vs-180 call. Fix:
   `pipeline._orientation_view` decides orientation on a full-height page view
   (the head's training framing); quarter-turn still applies to the oriented crop.
2. **Two-folio split under-segmentation: 90% → 100%.** When count says two but
   detection returns one box, halve the **full frame** (`_halve_box`).
3. **Page clipping (user-caught on 388248-0099-B).** The segmenter masked to the
   *tight detection box*; a box covering 73% width clipped the right page. Fix:
   `UNetFolioSegmenter.segment` masks by box **TERRITORY** (midpoints between box
   centres, full height) so a narrow box can never truncate the page. Verified:
   all 25 samples capture 100% of page width.
4. **Adaptive deskew.** Global Otsu railed to ±15° on ~39% of pages (it
   thresholded page-vs-margin). Now `orient._text_ink` uses a local adaptive
   threshold; deskew matches exhaustive search exactly, 0% railing.
5. **Landscape pre-pass.** Sideways scans broke segmentation; a coarse
   orient-before-segment pass (k∈{1,3} only) uprights the whole image first.
6. **Low-text review gate.** `QualityConfig.min_text_frac_for_orient=0.075` flags
   sparse pages (where orientation is unreliable) for human review.

7. **Sparse multi-block page → landscape band crop.** The legacy segmenter masks
   only the densest block on a sparse page; `oriented_page_quad` takes the largest
   connected component, cropping a portrait folio to a landscape band (e.g.
   225290-0182 → 1826×576, lost 2/3). Fix: `pipeline._recover_page_mask` expands a
   band mask to the full bright page within the folio's columns. Portrait crops
   28/30 → 30/30; broad sample 56/56 portrait.
8. **6× speed regression in `_drop_specks`.** It looped over every connected
   component doing `labels == i` (O(components×pixels)) — 86% of runtime, 15 s/image
   on speckled scans. Vectorized to one label lookup → **2.6 s/image**. Identical
   output. (`tools/benchmark.py` measures per-image throughput + 750k projection.)

Reliable clip-detection signal = **mask coverage** (crop vs U-Net page width),
NOT crop edge-ink (over-flags on sparse/full-bleed pages).

### Tight crop & faint ink (this session)
- `folio/stages/textregion.py` — EasyOCR/CRAFT detector (lazy, optional, GPU);
  `text_crop_box` unions detections + margin → tight crop. Safety: outlier-trim
  OFF by default (never drop a real line); `min_keep` guard rejects tiny boxes
  (faint under-detection) → falls back to looser crop. Default ON; `--no-tight-crop`.
- `folio/stages/content.py` — faint-ink detector (CLAHE + Sauvola + black-hat);
  `paper_box`/`trim_background` conservative crop; `enhance_faint` (CLAHE) for `--enhance`.
- Known limits (see docs/METRICS.md): rare spread→one_folio mis-count (unflagged);
  heavy-tilt spreads crop poorly (flagged). Both pre-existing count/tilt issues.

## 8. Reproduce the metrics

```bash
# end-to-end (count/split/orientation by input orientation):
python tools/evaluate.py --from-folders ../train_data --n 80 --landscape \
    --legacy-weights ../legacy_weights
# count head held-out:
python -m folio.training.eval --task count --data ../train_data/count_dataset/test \
    --weights weights/folio_count_convnextv2.pt
```
Docs: `docs/METRICS.md`, `docs/SUPERVISOR_SUMMARY.md` (+ `supervisor_evidence.png`),
`docs/DEPLOY.md` (EC2/Batch), `README_TOOL.md` (usage), `ARCHITECTURE.md` (design).

## 9. Sparse folios (the open loop with the supervisor)

Why it matters: orientation reads up/down from text structure; near-blank pages
have too little signal → rare confident-wrong 180°. We need a sparse set to
measure the true rate and fine-tune.

DONE: `tools/find_sparse_folios.py` scanned the supervisor's full transcription
set (232 volumes, 62,320 folios) → **`sparse_folios/`** in the data parent dir:
- `sparse_images.txt` (2,448 ids ≤120 chars), `sparse_le30.txt` (791),
  `sparse_le0.txt` (111 blank), `all_folios.csv` (per-image char counts).
- Recommended set to pull from S3: the **791-image ≤30-char set**.

NEXT (when images arrive): `python tools/evaluate.py` on them for a real
sparse-page orientation number → supervised fine-tune of the orientation head if
needed → re-tune `min_text_frac_for_orient`.

Transcription source: shared Google Drive folder
`1LiVAvvB6ot7mH_LmOkgpog_a4FDFBcCg` (subfolders `json/`, `md/`); also extracted
locally at `transcriptions/json/`. The Drive connector returns file contents into
context (can't bulk-pull to disk) — run the scanner on local/zip'd JSONs instead.

## 10. What's left

1. **Sparse-page orientation fine-tune** — blocked on supervisor's S3 images.
2. **Live S3 smoke test** — needs AWS creds; do `--limit 20` against a real prefix.
3. (Optional) train/reuse the `reject` (non-page) class — no reject data exists.
4. (Optional) cosmetic: narrow spread-halves trip `unexpected_aspect` review flag
   (conservative over-flag); gutter crops include a thin neighbor sliver (safe).

## 11. Gotchas / operational notes

- **OLED laptop:** disable system sleep during long runs but let the SCREEN sleep
  (`powercfg -change -monitor-timeout-ac 5 -standby-timeout-ac 0`), and RESTORE
  defaults after (`-monitor-timeout-ac 10 -standby-timeout-ac 30`). A 5-hour
  "stuck" training was the laptop sleeping and killing the job.
- **Background training** can be killed by sleep; pre-resize images to ~512px so
  epochs run in minutes (`train_data/upright_all_512`).
- **gdown** can't pull the supervisor's Drive folders (not "anyone with link"); he
  zips the data instead — that's the working path.
- Git repo is on branch `main`; commit messages end with the Co-Authored-By
  trailer. Weights/train_data/scratch outputs are gitignored.
- Persistent memory for this project lives in the agent's memory dir
  (`…/memory/MEMORY.md` + `folio-*.md`) and auto-loads each session.

## 12. One-line summary

The tool is complete and works (count/split/orientation/crop all verified); the
only remaining work is a sparse-page orientation fine-tune that waits on the
supervisor's S3 image set, plus a live S3 credential smoke test before the full
750k EC2 run.
