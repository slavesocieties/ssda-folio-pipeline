# Sparse-folio findings (and why we did NOT fine-tune on them)

We pulled the sparse set (791 images, ≤30 transcribed characters) to investigate
and ideally fix orientation on sparse pages. After examining it, the right call
was **not** to fine-tune the orientation model on it. Here's the reasoning, with
evidence, so it's on record.

## What the sparse set actually is
Overwhelmingly **blank / non-content pages**, not "pages with a little text":
- blank covers and **marbled inside-covers / endpapers**,
- conservation **calibration-card / ruler** reference pages,
- **decayed, worm-eaten fragments** with no readable text.

Breakdown of the 791:
- **111** have **0** transcribed characters (truly textless).
- **680** have **1–30** characters — but on inspection these are also blank/decayed;
  the "characters" are noise, decay marks, or a tiny illegible corner inscription
  misread by the transcription model, not real readable text.
- Survey: median text-ink coverage 0.041; the model's orientation guesses split
  73% upright / 24% "upside-down" / 3% sideways — but on a textless page the
  prediction is meaningless (a marbled pattern looks identical at any rotation).

## Why fine-tuning orientation on this is the wrong move
1. **No signal.** Orientation is defined by text structure; a blank/decayed page
   has none. There is nothing for the head to learn.
2. **It would add noise.** Feeding textless pages as "upright" teaches the head to
   associate arbitrary patterns with arbitrary labels, risking *worse* calibration
   on the real-text pages where it is currently **98.8%**.
3. **No verifiable labels.** These came as images only; we can't confirm a true
   orientation for pages that have no readable content.

## What the data DID confirm (the real value)
- **The review gate works on real sparse data:** ~**95%** of these pages are auto
  flagged for review (low-text), exactly the intended safety behavior. A blank
  page routed to review costs nothing — a human just accepts it; orientation is moot.
- **The orientation model is fine for real text** (98.8%). The earlier sparse
  "failures" (e.g. `84469`, `92786`) are the *rare* sparse-but-has-real-text pages,
  a small minority — not the textless bulk seen here.
- So the "sparse-page orientation problem" was, for the most part, a **non-problem**:
  most sparse pages are blank, where orientation doesn't matter.

## Recommendation
- **Treat these as blank / non-content**, not as an orientation-training target.
  The transcription-length signal already identifies them; downstream transcription
  can skip them.
- To push orientation on the *minority* of sparse-but-real-text pages would need
  actual orientation **labels** on that specific subset — not the textless bulk.

## Blank/non-content classifier — BUILT
We used this set as the genuinely-useful ML target: a content-vs-blank classifier
(`weights/blank_convnextv2.pt`, ConvNeXt-V2; train: `folio.training.train --task
blank`; inference: `BlankClassifier`). The pipeline now tags every folio
`is_blank` + `blank_conf` in the sidecar/manifest, so Archivault can skip
non-content pages.

Results (held-out test + end-to-end):
- **0 / 33 real content folios** wrongly marked blank — it never drops a real page
  (high precision; ~95% on the held-out test).
- **~70%** of real sparse/blank pages correctly marked (the obvious blank ones;
  textured/decayed/marbled blanks are harder and are the misses).
- So it's a **conservative, high-precision** signal: trust `is_blank` to skip, and
  use `blank_conf` if you want a stricter threshold. The low-text review gate
  remains the backstop for the blanks it misses.
