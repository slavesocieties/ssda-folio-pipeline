# Folio Processor — summary

A tool that turns **any** SSDA scan into clean **single-folio, upright, cropped**
page images. It implements Task 1 (split two-folio spreads, make every folio
upright, crop to the folio) and runs on one image, a folder, or the full S3
corpus.

![results on real scans](supervisor_evidence.png)

## How to use it
- **Desktop app:** `folio-gui` — drag scans onto the window, press Process, get
  thumbnails (pages it's unsure about are outlined in red).
- **Command line:** `folio page.jpg` · `folio /scans --out /out --jobs 6`
- **Whole corpus (S3):** `folio s3://ssda-raw/v/ --out s3://ssda-folios/f/`

Outputs: `folios/` (the crops), `sidecars/` (per-image JSON provenance),
`review/` (auto-flagged for a human glance), `manifest.csv`.

## What it does well (end-to-end, measured on labelled data)
- **Count** (one vs two folios): **100%**.
- **Two-folio split:** **100%** — dynamic seam/gutter split (not a fixed
  midpoint); splits even when detection under-segments a spread.
- **Orientation:** **98.8%** (upright 100%, upside-down 97.5%, landscape 90/270
  100%). 4-way head + an orient-before-segment pre-pass for the landscape
  volumes + an adaptive deskew that matches an exhaustive search exactly and
  never rails.
- **Robust:** auto-discovers weights, auto-selects GPU/CPU, classical fallback
  with no weights, one bad image never kills a run.
- **Scales:** ~0.5 s/image; S3 streaming with `--shard i/N` (EC2/Batch fan-out)
  and `--resume`; ~half a day for 750k at 8× (CPU-bound — see `DEPLOY.md`).

## Honest limitations
- **Sparse/near-blank pages** are the remaining orientation edge case: a few can
  still be confidently mis-oriented. The tool **flags low-text pages for review**
  rather than shipping them silently. A labelled sparse set (we can build it from
  the transcriptions — see below) would let us measure and fine-tune this away.
- The `reject` (non-page) class isn't trained — every input yields folio(s).
- Numbers above are in-distribution (same volumes as training); a fresh labelled
  set gives the true cross-volume figure. The harness (`tools/evaluate.py`) is
  ready to run on it.

## What I need from you
1. A **labelled validation set** for a true cross-volume number — and the
   **sparse subset** specifically. `tools/find_sparse_folios.py` already turns the
   automated transcriptions into a sparse-image list (verified on the real
   transcription format); point it at the `json/` folder, or send the JSONs and
   I'll generate the list to pull from S3.
2. The **run target** is EC2 — the S3 + `--shard` + `--resume` path is ready
   (`DEPLOY.md`).

See `README_TOOL.md` (usage), `METRICS.md` (numbers + how to reproduce),
`ARCHITECTURE.md` (design).
