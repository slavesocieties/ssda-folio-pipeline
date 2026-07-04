# Folio Pipeline — Architecture Proposal

**A production system for classifying, de-warping, splitting, orienting, and cropping archival book/document photography at scale.**

Author: Principal CV Engineering review
Target: replacement for the legacy `image-processing` U-Net pipeline
Compute: AWS GPU (S3-resident data)
Date: June 2026

---

## 1. Executive summary

The legacy pipeline works as a proof of concept but has four structural weaknesses that cap its accuracy and throughput:

1. **Spine/gutter splitting is hardcoded to `W // 2`.** Any book photographed off-center, with an asymmetric mount, or with unequal page widths is split through text. The 5% overlap band is a patch, not a fix.
2. **Segmentation is a from-scratch U-Net at 640×960.** Boundaries are recovered from `cv2.boundingRect(max(contour))` of an upscaled low-res mask. This loses true-resolution boundary fidelity per side, clips marginalia, and is brittle on frayed edges, shadow gradients, and tight margins.
3. **Orientation is a single 180° classifier plus an aspect-ratio heuristic for 90°.** It cannot correct in-plane skew, cannot distinguish 90° vs 270°, and the aspect-ratio rule misfires on "skinny two-folio" and "fat one-folio" scans (a problem the team already flagged in `next_steps.txt`).
4. **I/O downloads the entire bucket to local disk first, synchronously.** This is the dominant bottleneck at scale: it serializes network and compute, needs disk equal to the dataset, and leaves the GPU idle during transfer.

This proposal replaces the pipeline with a **precision-first hybrid**: a strong promptable foundation segmenter (**SAM 2.1**) localized by a lightweight detector, **geometry-driven gutter detection** that finds the *true* spine via a learned page mask plus a seam-energy search, full **4-way + continuous skew** orientation correction, optional **neural dewarping** (UVDoc/DocTr++ class) for curved pages, and a fully **asynchronous, streaming, GPU-batched** execution engine that never touches local disk and saturates the GPU. The design targets millions of images on AWS with horizontal scaling via Ray or AWS Batch.

The remainder of this document specifies the framework choices and the mathematics for each stage, then the corresponding code modules ship in `folio/`.

---

## 2. Design principles

- **Operate on the original resolution for all geometry.** Models run on downscaled tensors for speed, but every crop, rotation, and boundary is computed and applied in full-resolution coordinates. Sub-pixel transforms are composed once and applied once to avoid repeated resampling loss.
- **Decouple "where" from "what".** Detection/segmentation answer *where the page is*; classification heads answer *what state it is in* (count, orientation). Each is independently swappable and independently evaluable.
- **Foundation models for generalization, small heads for precision.** SAM 2.1 gives layout-agnostic masks with no per-collection training; compact fine-tuned heads handle the narrow, high-frequency decisions (folio count, 4-way orientation) where a specialist beats a generalist at a fraction of the FLOPs.
- **Streaming, never staging.** Bytes flow S3 → memory → GPU → S3. No "download everything first" phase. Backpressure keeps memory bounded.
- **Every decision is observable.** Confidence scores, the chosen gutter column, rotation angle, and mask IoU are emitted per image as structured JSON for QA sampling and active-learning loops.

---

## 3. Stage-by-stage architecture

### Stage 0 — Ingestion & orchestration

**Framework:** `aioboto3` (async S3) + a bounded `asyncio` producer/consumer pipeline, scaled horizontally with **Ray** (or AWS Batch array jobs for the simplest operational model).

**Why not the legacy approach:** the legacy `s3_download.py` lists then `download_file`s every key to disk via a 10-thread pool. At millions of images this needs petabyte-scale local disk and blocks compute. Instead:

- A single **lister coroutine** paginates `list_objects_v2` and feeds keys into an `asyncio.Queue` (bounded — this is the backpressure valve).
- **N downloader coroutines** `get_object` straight into RAM (`bytes`), decode with `PyTurboJPEG` / `cv2.imdecode`, and push decoded arrays onto a second bounded queue.
- A **batcher** assembles GPU micro-batches (see Stage 4).
- **Writer coroutines** stream output crops back to S3 with `put_object` (multipart for large TIFFs).

**Math/throughput model.** With per-image GPU time `t_g` and per-image network time `t_n`, the synchronous legacy cost is `sum(t_n + t_g)`. The async pipeline overlaps them, so steady-state throughput is `1 / max(t_n/P_n, t_g/B)` where `P_n` is download concurrency and `B` the GPU batch size. Sizing `P_n` so that `t_n/P_n <= t_g/B` makes the system **GPU-bound** — the correct target. We expose both as config and auto-tune `P_n` from observed latencies.

### Stage 1 — Page-count & coarse layout classification

**Decision:** one folio vs. two folios (book opening), plus a "junk/cover/blank" reject class.

**Framework:** a fine-tuned **ConvNeXt-V2-Tiny** (or EfficientNetV2-S) classifier — a stronger, modern backbone than the legacy ResNet-18, still tiny and fast in `fp16`/`int8`.

**Key fix for the "skinny two-folio / fat one-folio" problem:** do **not** rely on aspect ratio. We feed the classifier the image *plus* two cheap geometric priors: (a) the normalized aspect ratio, and (b) the **vertical projection profile's central-third energy ratio**, which spikes when a dark gutter runs down the middle. For the column-intensity profile `c(x) = mean_y I(x,y)`, the gutter prior is

```
g = min_{x in [0.4W, 0.6W]} c_smooth(x)  /  median_x c_smooth(x)
```

A low `g` (a dark central valley) is strong evidence of a two-page opening regardless of aspect ratio.

### Stage 2 — Global de-skew / coarse rectification (before splitting)

We rectify the *whole spread* before we try to find the spine, because a tilted book makes the gutter a diagonal, not a column.

**Framework:** classical, robust, and cheap — estimate the dominant page-edge orientation via a `minAreaRect` fit on the foreground mask (from Stage 3's low-res pass), or the radon-transform peak of the gradient field. Apply a single affine de-rotation. Heavy non-planar curvature is deferred to Stage 6 (neural dewarp) only when needed.

### Stage 3 — Page segmentation (the "where")

**Framework:** **SAM 2.1 (Hiera-L image predictor)** prompted by a lightweight **detector** (`RT-DETR` or a small YOLO) that proposes one box per page. This "detect → prompt SAM" pattern is the current best accuracy-per-effort: the detector gives robust instance separation (1 or 2 page boxes); SAM 2.1 returns a pixel-precise, layout-agnostic mask with far better boundary fidelity than a from-scratch U-Net, with **no per-collection training**.

**Why this beats the legacy U-Net:**

- Trained on billions of masks → generalizes to unseen layouts, frayed edges, and shadow gradients out of the box.
- We run SAM on a **high-resolution tile** and refine the mask edge with a guided filter, so the boundary is recovered at original resolution, not upscaled from 640×960.
- Marginalia preservation: we take the page *support* mask (the physical leaf), not a text box, so nothing in the margins is clipped. Text-region detection becomes an *optional, separate* output, never a crop constraint.

**Boundary math.** From the binary mask `M` we take the largest connected component, compute its **oriented minimum-area rectangle**, and then *dilate the crop outward* by a learned margin `delta` (default 1.5% of the page diagonal) clamped to image bounds, so frayed edges and marginal notes are never cut. The final crop is the rotated rectangle, sampled once via `cv2.warpPerspective`.

### Stage 4 — Dynamic spine / gutter detection (core contribution)

This replaces `mid = W // 2` entirely. The gutter is found, not assumed.

**Inputs:** the de-skewed spread (Stage 2) and the foreground page mask (Stage 3).

**Algorithm — learned prior + seam-energy search:**

1. **Region of interest.** The gutter lies *between* the two detected page boxes. The search band is the gap between the right edge of the left page and the left edge of the right page (with a margin), not a fixed central window. This already handles off-center books.
2. **Energy field.** Build a vertical "gutter-likeness" energy `E(x,y)` that is high where the spine is. Three complementary cues, normalized and summed:
   - **Darkness/shadow:** spines sit in a shadow valley → `E_dark = 1 - I_norm(x,y)`.
   - **Texture absence:** the gutter has no text → low local gradient magnitude → `E_smooth = 1 - grad_norm(x,y)`.
   - **Mask gap:** between the two page masks → `E_gap = 1 - M(x,y)`.
3. **Seam carving for the true (possibly curved) gutter.** Rather than a single column, we find the minimum-energy **vertical seam** `s(y)` — one `x` per row — via dynamic programming:

   ```
   C(x,y) = E(x,y) + min( C(x-1,y-1), C(x,y-1), C(x+1,y-1) )
   ```

   Backtracking from `argmin_x C(x,H)` yields a seam that **bends with the book**, correctly separating warped/curved gutters where a straight line would cut into a page. This is the key robustness win over a hardcoded midpoint.
4. **Split & assign.** Each page is cropped on its side of the seam; the seam path defines the inner boundary, the page mask the outer three. Pages are labeled recto/verso (A/B) *after* orientation is resolved (Stage 5), fixing the legacy bug where a 180° flip silently mislabeled A/B.

This degrades gracefully: if only one page box is found, the spread is single-folio and Stage 4 is skipped.

### Stage 5 — Orientation correction (4-way + continuous skew)

**Two layers:**

1. **Coarse 4-way (0/90/180/270):** a fine-tuned compact classifier (ConvNeXt-V2-Tiny head). The 2025 literature ("Seeing Straight") confirms a CNN classifier on the four canonical angles is the reliable approach — far better than the legacy aspect-ratio guess, and it distinguishes 90° from 270°, which the legacy code cannot. We additionally use a **text-line angle estimator** as a tie-breaker: the dominant text-line orientation from the projection-profile variance peak resolves low-confidence cases.
2. **Fine skew (±15°):** maximize the variance of the **horizontal projection profile** over rotation angle — sharp, well-separated text-line peaks ⇒ correct deskew. This is the legacy `deslant` idea, but applied as the *final* sub-degree correction after the page is already coarse-aligned, and using projection-profile *variance* (more stable than the legacy sum-of-squared-differences).

**Transform composition.** Coarse rotation `R_90k`, fine skew `R_theta`, and the Stage-3/4 crop homography `H` are composed into a **single** matrix `T = R_theta @ R_90k @ H` and applied with **one** `warpPerspective` at original resolution — avoiding the legacy pattern of multiple `rotate(expand=True)` calls that compound interpolation blur and re-pad backgrounds repeatedly.

### Stage 6 — Optional neural dewarping (curved/bound volumes)

For tightly bound volumes where pages curve toward the gutter, a planar crop still bows the text. When the Stage-3 mask's boundary curvature exceeds a threshold, route the page through a **UVDoc / DocTr++-class** unwarping network that regresses a 2D deformation grid and remaps the page flat. This is gated (most flat archival shots skip it) to keep throughput high.

### Stage 7 — Output & provenance

Each output page is written to S3 as the upright, portrait, full-resolution crop, alongside a JSON sidecar recording: source key, page label, gutter seam, all rotation angles, every model's confidence, mask IoU/area ratio, and processing version. This powers QA sampling and an active-learning loop that routes low-confidence images to human review and back into fine-tuning sets.

**Background handling — tight crop (default) vs. white-out.** The crop is stage 1 of a pipeline whose final stage is *paid* transcription, so anything the crop removes is text that is permanently lost and never billed for. Two modes share the identical split/upright/deskew geometry and differ only in the background:

- **Approach B — tight bounding-box crop (`crop_to_folio_mask`, the DEFAULT).** The finished crop is reduced to the bounding box of the learned folio-half mask (its convex-hull-safe form, plus a small margin). It keeps the natural background but **alters no pixel** — only the rectangle is tightened — so it *cannot* erase folio text, while the facing page/binding fall outside the half-mask and are excluded. Verified at **0 px of folio-text loss** across the evaluation set (`tools/verify_B_noclip.py` locates each crop inside a pixel-aligned keep-everything reference and measures any folio ink outside it; `tools/gt_interior_verify.py` checks for interior white holes).
- **Approach A — white-out (`mask_background`, `--white-out`).** Blanks every non-folio pixel to white using the mask. Cleaner input, but because it *erases* pixels judged non-page, on hard pages (ink bleed-through, water damage, faded parchment where the segmenter under-covers the sheet) it over-crops and eats real content.

**Decision:** B is the default. A was trialled first and produced a clean background, but review of difficult volumes (heavy bleed-through / water damage) showed the white-out shaving genuine text; B removes that failure mode entirely at the cost of a slightly busier background for the transcriber, which downstream models tolerate well. Selectable per-run via `--white-out` / `--crop-to-mask`.

---

## 4. Recommended stack

| Concern | Legacy | Proposed | Rationale |
|---|---|---|---|
| Page count / layout | ResNet-18 | ConvNeXt-V2-Tiny + geometric priors | Layout-aware, fixes skinny/fat misfires |
| Segmentation | U-Net @640×960 from scratch | RT-DETR → **SAM 2.1** (Hiera-L) | Zero-shot generalization, original-res boundaries |
| Spine split | `W // 2` | **Seam-energy gutter search** | Handles off-center, warped, shadowed gutters |
| Orientation | 180° classifier + aspect heuristic | 4-way CNN + projection-profile skew | Full 0/90/180/270 + sub-degree deskew |
| Dewarp | none | UVDoc/DocTr++ (gated) | Flattens curved bound pages |
| Inference runtime | eager PyTorch | **TensorRT / `torch.compile` + fp16**, micro-batched | 3–6× GPU throughput |
| I/O | download-all-to-disk | **aioboto3 streaming**, in-RAM | Removes the dominant bottleneck |
| Scale-out | single process | **Ray** / AWS Batch arrays | Linear horizontal scaling |
| Decode | PIL | **PyTurboJPEG / nvJPEG** | Faster JPEG decode, optional GPU decode |

---

## 5. Scaling & performance engineering

- **GPU saturation via micro-batching.** Variable image sizes are bucketed by aspect ratio and padded within a bucket so each batch is a clean tensor; this keeps GPU utilization high without wasting FLOPs on extreme padding.
- **Mixed precision + compiled graphs.** All models run `fp16` (or `int8` for the classifiers) under `torch.compile`/TensorRT. SAM 2.1's image encoder is the cost center; we cache its embedding per image and reuse it across prompts.
- **Zero-copy decode path.** `get_object` → `bytes` → `PyTurboJPEG.decode` (or nvJPEG on-GPU) avoids temp files entirely. Optional **GPU decode** keeps the image on-device from decode through inference.
- **Backpressure & bounded memory.** Every queue is bounded; if the GPU stalls, downloaders pause — memory stays flat regardless of dataset size.
- **Idempotent, resumable.** A manifest of completed keys (in DynamoDB or an S3 manifest) lets array jobs resume after spot-instance interruption without reprocessing.
- **Cost lever.** Spot GPU fleet + checkpointed manifest; the embarrassingly parallel workload tolerates interruptions cleanly.

**Expected effect:** the legacy serial download+infer loop is replaced by a GPU-bound streaming engine. On a single A10G/L4, micro-batched fp16 inference plus overlapped S3 streaming should move the bottleneck entirely onto the GPU; throughput then scales linearly with GPU count under Ray/Batch.

---

## 6. Failure handling & quality gates

- Confidence thresholds per stage; anything below routes to a `review/` prefix in S3 instead of `output/`.
- Sanity invariants: output aspect ratio within expected portrait band; crop area ≥ X% of detected page; seam monotonic and within the inter-page gap. Violations flag for review.
- Golden-set regression: a fixed labeled set runs every release; boundary IoU, split accuracy, and orientation accuracy are tracked over time.

---

## 7. Migration path

1. Stand up the streaming I/O engine and run the **legacy models inside it** — immediate throughput win, zero accuracy change, low risk.
2. Swap segmentation to RT-DETR→SAM 2.1; A/B against the U-Net on the golden set.
3. Replace `W//2` with the seam-energy gutter detector.
4. Replace orientation with the 4-way + skew module.
5. Enable gated dewarping last.

Each step is independently shippable and measurable.

---

## 8. References

- SAM 2 — *Segment Anything in Images and Videos*, Meta AI (ICLR 2025).
- *Seeing Straight: Document Orientation Detection for Efficient OCR* (arXiv:2511.04161).
- *UVDoc: Neural Grid-based Document Unwarping* (SIGGRAPH Asia 2023); *DocTr++* unified rectification.
- *Efficient Document Image Dewarping via Hybrid Deep Learning and Cubic Polynomial Geometry Restoration* (arXiv:2501.03145).
- Seam carving: Avidan & Shamir, *Seam Carving for Content-Aware Image Resizing* (SIGGRAPH 2007) — adapted here for gutter localization.

---

## 9. Scale, baseline, and the 750k processing plan

### Corpus
The full SSDA corpus is **~750,000 digital images**. The legacy team's
`object_keys_remaining.txt` (a flat list of image keys) is reused directly as
the **resumable work manifest**: each worker claims keys, writes outputs, and
records completion, so spot-instance interruptions never cause reprocessing.

### Fresh approach vs. legacy baseline
We use the fresh stack; the legacy `.pth` models serve only as a baseline to
beat (see `tools/eval_vs_legacy.py`).

| Concern | Legacy model (`.pth`) | Fresh approach |
|---|---|---|
| Folio count | `folio_count_classifier.pth` (ResNet-18, aspect-driven) | ConvNeXt-V2 head + geometric priors |
| Orientation | `upside_down.pth` / `folio_upside_down.pth` (180° only) | 4-way head (0/90/180/270) + deskew |
| Page split/segmentation | `unet_folio_split.pth` (U-Net @640×960, `W//2` split) | SAM 2.1 + detector, seam-energy gutter |
| Text regions | `unet_text_segmentation.pth` | dropped (not required by Task 1) |
| Page matching | `lawson_siamese_model.pth` (Siamese) | out of scope (split/orient/crop only) |

### Throughput & cost (order-of-magnitude)
SAM 2.1's image encoder dominates per-image cost.

| Venue | Rate (batched) | Time for 750k | Notes |
|---|---|---|---|
| RTX 5080 laptop (16 GB) | ~0.5–1.5 s/img | ~4–13 days continuous | great for dev/train + moderate batches |
| AWS, 8× L4/A10G (spot) | parallel | ~12–36 h | ~hundreds of USD on spot |

**Cost-cutting lever (recommended):** route the *easy majority* through a cheap
fast path (detector box + classical refine, no SAM) and reserve SAM 2.1 for the
hard cases (low mask confidence, frayed/curved edges, ambiguous gutter). On a
clean archival corpus this typically removes SAM from the large majority of
images, cutting full-run time/cost several-fold. The router is a confidence
threshold on the detector + a quick boundary-curvature check.

### Recommended rollout
1. **Develop & train on the laptop** (5080 handles SAM 2.1 inference + head training).
2. **Benchmark** real per-image time on the 5080 → firm up the 4–13 day estimate.
3. **Validate** against the legacy baseline on the 25-image golden set (`eval_vs_legacy.py`).
4. **Full 750k run**: laptop for a multi-day grind, or burst to spot GPUs for ~a day; manifest makes it resumable either way.

---

## 10. Final build decision: hybrid (reuse the good legacy parts)

After weighing reuse vs. rebuild per component, the **recommended production
configuration is a hybrid** (`folio/models/hybrid.py`,
`build_hybrid_pipeline(...)`):

| Component | Decision | Why |
|---|---|---|
| `unet_folio_split.pth` | **Reuse** as the cheap default page-mask generator | Trained, tiny, fast — avoids running SAM 2.1 on all 750k images |
| Spine split & crop | **Fresh** (seam-energy + oriented crop) | Fixes the legacy `W//2` split and low-res crop |
| `folio_count_classifier.pth` | **Reuse**, tie-broken by aspect/shadow | Trained & free; geometry rescues its skinny/fat misfires |
| `upside_down.pth` | **Reuse** for 0-vs-180 + landscape→90/270 rule | Trained; covers the common case, rule covers the gap |
| SAM 2.1 | **Fallback only** | Used where the U-Net mask is weak (frayed/curved) — keeps cost down |
| text-seg U-Net, Siamese | **Drop** | Not part of split/orient/crop |

Net effect: higher quality than the legacy pipeline (true-spine split, precise
upright crops, 90/270 coverage) at a fraction of an all-SAM run's cost — the
right trade for a 750k corpus. Run it with:
```
python tools/run_samples.py "/scans" --hybrid --legacy-weights /legacy_pth --outdir out
# add --sam-fallback to escalate hard masks to SAM 2.1
```
The fully-fresh path (SAM 2.1 + trained heads, `--neural`) remains available and
is the higher-ceiling option once the custom heads are trained.
