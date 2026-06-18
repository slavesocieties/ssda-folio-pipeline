# folio_process — SSDA single-folio pre-processing tool

Turns **any** SSDA scan into clean **single-folio, upright (portrait), cropped**
page image(s). Implements Task 1: (1) split two-folio spreads into two
derivatives, (2) make every folio upright, (3) crop to the folio.

## Install
```bash
pip install -r requirements.txt
# GPU (recommended): install the CUDA build of torch for your card, e.g. cu128:
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## Weights (one-time)
Download the legacy `.pth` models into one folder (IDs in
`tools/legacy_model_ids.txt`) and keep the trained orientation head at
`weights/orientation4_convnextv2.pt` (the default path).

## Run
```bash
# simplest: just an image in -> corrected folio image(s) out (no flags needed)
python tools/folio_process.py page.jpg

# a whole folder
python tools/folio_process.py /scans --out /out

# S3 batch (the full corpus): stream in, write crops + sidecars back to S3
python tools/folio_process.py s3://ssda-raw/volumes/ --out s3://ssda-folios/folios/ --region us-east-1
```
- **Weights are auto-discovered** (`./legacy_weights`, repo, `$FOLIO_LEGACY_WEIGHTS`,
  or next to the input); pass `--legacy-weights DIR` to override.
- Device is auto-selected (CUDA if available, else CPU).
- If legacy weights aren't found it falls back to a dependency-free classical
  mode so it still produces output for any image (lower quality).
- `--no-prepass` disables the sideways/landscape auto-uprighting.
- S3 mode needs AWS credentials (env or `~/.aws`); it streams with bounded
  concurrency so memory stays flat regardless of corpus size (built for 750k).

## Output (`--out`, default `./folio_out`)
```
folios/    <stem>[-A|-B].jpg   the upright single-folio crop(s)
sidecars/  <stem>.json         provenance per source image
review/    copies of crops the tool is unsure about (human QA)
manifest.csv                   one row per output folio
```
Each folio row records page count + confidence, any whole-image pre-rotation,
the applied rotation/skew, orientation confidence, text fraction, and whether it
was flagged `needs_review` (and why).

## What it does internally (recommended hybrid config)
1. **Coarse orientation pre-pass** — uprights sideways/landscape scans before
   anything else, so segmentation never sees a rotated page.
2. **Count + segment** — legacy U-Net page mask inside robust foreground boxes.
3. **Dynamic spine split** — seam-energy gutter (not a fixed midpoint) for spreads.
4. **Oriented crop** at full resolution (margin so marginalia isn't clipped).
5. **Orientation** — trained 4-way ConvNeXt-V2 head + sub-degree projection deskew.
6. **Review gate** — low-confidence / blank / odd-aspect folios routed to `review/`.

## Known limitations (honest)
- Rare **confident-wrong** orientation on sparse/near-blank pages (both the neural
  and legacy models can agree on the wrong 180°); these are not catchable by the
  confidence gate and need more training data. Measure the rate on a labeled set.
- The `reject` (non-page) class is not trained.
