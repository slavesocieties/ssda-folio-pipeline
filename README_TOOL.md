# Folio Processor — SSDA single-folio pre-processing tool

Turns **any** SSDA scan into clean **single-folio, upright (portrait), cropped**
page image(s). Implements Task 1: (1) split two-folio spreads into two
derivatives, (2) make every folio upright, (3) crop to the folio.

Two ways to use it: a **drag-and-drop desktop app** (`folio-gui`) and a
**command line** (`folio`) that also does S3 at corpus scale.

## Install on a new machine (any lab computer)
Works on Windows, macOS, Linux. Needs **Python 3.10–3.12**.
```bash
git clone <repo-url> ssda-folio-pipeline
cd ssda-folio-pipeline
python -m pip install -r requirements.txt   # dependencies (first time only)
python -m pip install -e . --no-deps         # the `folio` + `folio-gui` commands
python tools/setup_weights.py                # fetch the model weights (see below)
```
GPU is optional but ~10× faster. For an NVIDIA card install the matching CUDA
torch, e.g. cu128 for RTX 50-series:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```
> Re-running `pip install -r requirements.txt` can replace a CUDA torch with a
> CPU build. If `torch.cuda.is_available()` turns False, reinstall from the cu128
> index above. CPU-only machines work fine (auto-detected), just slower.

**Windows double-click app:** `Folio Processor.bat` in the repo opens the GUI with
no terminal (uses `py`/`python` on PATH — copy it to the Desktop if you like).

## Weights (one-time) — they are NOT in git
The weights are too large for git, so a fresh clone has none. `tools/setup_weights.py`
fetches them: the legacy `.pth` (Google Drive IDs in `tools/legacy_model_ids.txt`)
→ `../legacy_weights/`, and the three trained heads → `weights/`
(`orientation4_convnextv2.pt`, `folio_count_convnextv2.pt`, `blank_convnextv2.pt`).

To host the trained heads for the whole lab, attach them to a **GitHub Release**,
then everyone runs:
```bash
FOLIO_WEIGHTS_BASE=https://github.com/<org>/ssda-folio-pipeline/releases/download/v1.0 \
  python tools/setup_weights.py
```
Weights are then **auto-discovered** from `./weights`, `./legacy_weights`, the repo,
the input folder, or `$FOLIO_LEGACY_WEIGHTS` — no flags needed.

## Desktop app (recommended for ad-hoc use)
```bash
folio-gui            # or:  python -m folio.gui
```
Drag images or a folder onto the window (or use the buttons), press **Process**,
and the corrected crops appear as thumbnails — pages the tool is unsure about are
outlined in red and copied to a `review/` folder for a quick human check.

## Command line
```bash
folio page.jpg                                   # one image -> ./folio_out
folio /scans --out /out                          # a folder
folio /scans --out /out --jobs 6                 # 6 parallel CPU workers (big folders)
folio /scans --out /out --resume                 # skip already-processed images
folio /scans --out /out --enhance                # also write *_enhanced.jpg for faint pages
folio s3://ssda-raw/v/ --out s3://ssda-folios/f/ --limit 20   # S3, 20-image dry run
```
(`python -m folio.cli ...` and `python tools/folio_process.py ...` are equivalent.)

- Device auto-selects CUDA, else CPU. `--device cpu` to force.
- `--no-prepass` disables the sideways/landscape auto-uprighting.
- `--enhance` writes a CLAHE contrast-boosted `*_enhanced.jpg` next to each faint /
  light-ink crop, so downstream transcription can read faint handwriting.
- `--limit N` processes at most N images (safe dry run, esp. for S3).
- S3 mode needs AWS credentials (env or `~/.aws`); it streams with bounded
  concurrency so memory stays flat regardless of corpus size (built for 750k).

## Output (`--out`, default `./folio_out`)
```
folios/    <stem>[-A|-B].jpg   the upright single-folio crop(s)
sidecars/  <stem>.json         provenance per source image
review/    copies of crops the tool is unsure about (human QA)
manifest.csv                   one row per output folio
```
Each row records page count + confidence, any whole-image pre-rotation, the
applied rotation/skew, orientation confidence, text fraction, **`is_blank`**
(content vs blank/non-content page — so Archivault can skip blanks), and whether
it was flagged `needs_review` (and why).

## What it does internally (recommended hybrid config)
1. **Coarse orientation pre-pass** — uprights sideways/landscape scans before
   anything else, so segmentation never sees a rotated page.
2. **Count + segment** — legacy U-Net page mask inside robust foreground boxes.
3. **Dynamic spine split** — seam-energy gutter (not a fixed midpoint) for spreads.
4. **Oriented crop** at full resolution (margin so marginalia isn't clipped).
5. **Orientation** — trained 4-way ConvNeXt-V2 head + sub-degree adaptive deskew.
6. **Review gate** — low-confidence / sparse / odd-aspect folios routed to `review/`.

## Accuracy (held-out)
| Head | Held-out accuracy | Notes |
|---|---|---|
| Folio count | **100%** (126 images) | reserved test split, never trained on; no `reject` class |
| 4-way orientation | **98.3%** | held-out split during training (in-distribution) |

On **new volumes** the orientation head is reliable on text-rich pages but can be
confidently wrong on **sparse/near-blank** pages; those are caught by the low-text
**review gate** (flagged, not silently shipped). Re-tune `min_text_frac_for_orient`
on a labelled validation set before a full-corpus run.

## Known limitations (honest)
- **Sparse/near-blank page orientation**: both the neural and legacy models can
  agree on the wrong 180°. Not catchable by confidence; mitigated by routing
  low-text pages to `review/`. The real fix is supervised up/down labels.
- The `reject` (non-page) class is not trained — every input yields folio(s).
