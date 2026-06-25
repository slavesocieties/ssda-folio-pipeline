# Metrics

Honest, reproducible numbers for the folio pipeline. Two kinds are reported:
**held-out** (a reserved split the model never trained on) and **cross-source**
(the 25 hand-collected sample scans, from volumes outside the training data —
the closest proxy for new-corpus generalisation).

## Held-out accuracy

| Head | Accuracy | Test set | Notes |
|---|---|---|---|
| **Folio count** (one vs two) | **100%** | 126 images (84 one-folio, 42 two-folio) reserved from training | no `reject` class trained |
| **4-way orientation** (0/90/180/270) | **98.3%** | 15% split held out during training (in-distribution) | best-checkpoint selection adds mild optimism |

Reproduce:
```bash
# count head on its reserved test split
python -m folio.training.eval --task count \
    --data train_data/count_dataset/test --weights weights/folio_count_convnextv2.pt
# orientation val accuracy is printed each epoch during training
```

## End-to-end pipeline (labelled, in-distribution)

`tools/evaluate.py` runs the WHOLE pipeline per image and scores it against
labels (the labelled folders, with some singles synthetically rotated to 90/270
to exercise the landscape pre-pass). On 200 images:

| Metric | Result |
|---|---|
| Folio count | **100%** |
| Two-folio split (==2 folios) | **100%** |
| Orientation upright (overall) | **98.8%** |
| — upright input (stay upright) | 100% |
| — upside-down input (flip 180) | 97.5% |
| — landscape 90/270 (pre-pass) | 100% |

> This used to read ~50% before two fixes: (1) the pipeline fed the 4-way head
> the *tight oriented crop*, which flips its 0-vs-180 call — it now decides on a
> full-page view (the head's training framing); (2) the harness was
> double-rotating the already-180 `upside_down` files. Reproduce:
> `python tools/evaluate.py --from-folders ../train_data --n 80 --landscape --legacy-weights ../legacy_weights`

These are **in-distribution** (same volumes as training). Cross-source below.

## Cross-source (the 25 sample scans, new volumes)

End-to-end with the production hybrid config:

| Task | Result |
|---|---|
| Two-folio **split** | 7/7 spreads split into A/B; 32/32 output crops portrait; 0 unsplit spreads |
| **Deskew** | small, sane corrections (mean ~1.7°, max ~5°); 0 railing |
| **Orientation** (text-rich pages) | correct (upright, readable) |
| **Orientation** (sparse/near-blank pages) | ~2–3 of 32 folios confidently wrong (180°) — **all routed to `review/`** by the low-text gate |

So the tool does not silently ship a mis-oriented page: sparse pages it cannot
orient reliably are flagged for a human glance.

## Deskew correctness (regression guard)

The adaptive-threshold deskew returns the **same angle as an exhaustive search**
(max |Δ| = 0.000° on real + synthetically-tilted pages) while running ~5× faster,
and recovers known tilts within the unit-test tolerance. Earlier (global-Otsu)
objective railed to ±15° on ~39% of page crops; the current objective: **0%**.

## Learned folio segmentation (page boundary)

Trained a U-Net (ImageNet-pretrained encoder, `segmentation_models_pytorch`) on
Daniel's **151 folio-level mask pairs** to predict the page region directly —
the background-agnostic boundary the classical paper detector can't get on
light-on-light scans. Strong colour/illumination augmentation to generalise
across dark/light backgrounds.

| | |
|---|---|
| **Val IoU** | **0.9604** (22-image held-out split) |
| Train / val | 129 / 22 of 151 pairs, 512×512 |
| Use | when present, its mask drives the crop (precise full folio); seam split still divides spreads. Falls back to the classical/legacy mask if absent |

Verified: the torn light-background document (`113754`) that classical failed on
(empty paper mask) now crops tight to the full folio; dark-bg spreads still split
cleanly. Train: `python -m folio.training.seg --images <dir> --masks <dir>`.
Reproduces from `preprocessed/{images,masks}` (Daniel's mask set).

## Cropping tightness (vs the supervisor's ground-truth text rects)

Crop quality is measured as the **meaningful-pixel ratio** (ground-truth text area
÷ crop area; Daniel's conceptual target ≈ 0.75) and **coverage** (GT text kept
inside the crop; must stay ~1.00 = nothing clipped). `tools/eval_crop.py`:

| Crop | meaningful-pixel ratio | content clipped |
|---|---|---|
| Full-frame (loose) | 0.30 | none |
| `paper_box` (classical) | 0.40 | none |
| **CRAFT text-region (default)** | **~0.5–0.6 / page** | **none (coverage 1.00)** |

The learned crop (EasyOCR/CRAFT, `folio/stages/textregion.py`) removes binding,
scanner bed, colour-calibration cards and blank margins. It **never clips**: it
unions every detection (no outlier-drop) and rejects an implausibly small box
(faint-page under-detection), falling back to the looser crop. On a 36-image
broad sample (56 folios): **56/56 portrait, 0 errors, 0 landscape over-crops.**

Sparse multi-block pages (text in separated blocks) used to crop to one band
(landscape); `_recover_page_mask` now expands a band mask to the full bright page
→ portrait crops **28/30 → 30/30** on the 20-image sample.

## Throughput

**~2.6 s/image** end-to-end on an RTX 5080 with everything on (hybrid segment +
4-way orientation + deskew + blank-detect + CRAFT tight crop), measured by
`tools/benchmark.py` on large scans. For 750k: ~22 days single-worker, **~1.4 days
on 16 GPU workers**, ~0.7 days on 32. `--no-tight-crop` skips CRAFT.

> Was 15 s/image until a pathological `_drop_specks` loop (O(components×pixels) on
> speckled scans) was vectorized — a 6× speedup. Re-benchmark on the target EC2
> instance before sizing the fleet.

## Known limitations (found in broad-sample verification)

- **Spread mis-counted as one folio (rare, silent):** a two-page spread with a
  subtle gutter can be classified `one_folio` and left unsplit (e.g. `740004-0014`).
  It is *not* flagged (the near-square aspect overlaps legitimate single folios, so
  there is no safe geometric gate). A count-classifier limitation; would need
  retraining on more spread examples to fix.
- **Heavily tilted spreads (~14°+):** deskew rails at its ±15° cap and the crop
  stays skewed/square (e.g. `701241-0057`) — but these *are* flagged for review
  (low-text + unexpected-aspect), so not silently shipped.
- `is_blank` can false-positive on such degenerate crops (dense text read as blank);
  it never drops a real page on a clean crop (0/33 in earlier testing).

## Caveats

- The held-out count/orientation splits are from the **same volumes** as training;
  the honest new-corpus signal is the cross-source section above.
- Before a full-corpus run, **re-tune `min_text_frac_for_orient`** (the review-gate
  threshold) on a labelled validation set drawn from the real corpus mix — the 25
  samples are unusually sparse-heavy.
