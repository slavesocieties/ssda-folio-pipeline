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

## Throughput

~0.5 s/image single-process on an RTX 5080 (CPU-bound classical stages dominate;
the GPU is lightly used). For 750k images: ~4.5 days single-process, or ~18 h at
6× parallelism (`--jobs 6`, or shard the S3 prefix across workers).

## Caveats

- The held-out count/orientation splits are from the **same volumes** as training;
  the honest new-corpus signal is the cross-source section above.
- Before a full-corpus run, **re-tune `min_text_frac_for_orient`** (the review-gate
  threshold) on a labelled validation set drawn from the real corpus mix — the 25
  samples are unusually sparse-heavy.
