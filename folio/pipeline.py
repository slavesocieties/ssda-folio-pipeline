"""Orchestrator: ties Stages 1-7 into one per-image flow, and provides the
async streaming runner that drives it over an entire S3 bucket.

The per-image logic (`process_image`) is deliberately model-light at the type
level: it depends on small interfaces (detect/segment/predict) so it can be
unit-tested with stubs and so backends are swappable.
"""
from __future__ import annotations

import asyncio
from typing import List, Optional
import numpy as np

from .config import PipelineConfig
from .schemas import ImageResult, FolioResult, PageBox, PageCount
from .stages import spine, geometry, orient


class FolioPipeline:
    def __init__(self, cfg: PipelineConfig, segmenter=None, counter=None,
                 orienter=None, dewarper=None, coarse_orienter=None,
                 blank_classifier=None, folio_segmenter=None):
        self.cfg = cfg
        # injected so tests can pass stubs; built lazily for production runs
        self.segmenter = segmenter
        self.counter = counter
        self.orienter = orienter
        self.dewarper = dewarper
        # optional LEARNED folio/page segmenter (U-Net). When set, its precise
        # page mask drives the crop (background-agnostic, works on light-on-light
        # scans the classical detector can't), replacing the legacy crop mask.
        self.folio_segmenter = folio_segmenter
        # optional content/blank classifier; when set, each folio is tagged
        # is_blank in the sidecar so Archivault can skip non-content pages.
        self.blank_classifier = blank_classifier
        # optional 4-way head run on the WHOLE image to upright sideways
        # (landscape) scans BEFORE count/segment, so the segmenter never sees a
        # rotated page. Must be a true 4-way classifier (not the legacy
        # aspect-ratio orient adapter, which flags any wide image as sideways).
        self.coarse_orienter = coarse_orienter

    # ------------------------------------------------------------------ build
    def _ensure_models(self):
        from .models.segmentation import PageSegmenter
        from .models.classifiers import FolioCountClassifier, OrientationClassifier
        if self.segmenter is None:
            self.segmenter = PageSegmenter(self.cfg.model)
        if self.counter is None:
            self.counter = FolioCountClassifier(self.cfg.model)
        if self.orienter is None:
            self.orienter = OrientationClassifier(self.cfg.model)

    # ---------------------------------------------------------------- per img
    def process_image(self, key: str, image: np.ndarray) -> ImageResult:
        res = ImageResult(source_key=key, version=self.cfg.version)
        if image is None or image.size == 0:
            res.error = "decode_failed"
            return res
        orig_h, orig_w = image.shape[:2]   # ORIGINAL size, before any pre-pass rotation
        try:
            # Stage 0b: coarse orientation pre-pass. Only acts on sideways
            # (landscape, k in {1,3}) scans, which the upright-trained segmenter
            # cannot handle; 0/180 are left to the per-folio orientation stage.
            if self.coarse_orienter is not None:
                cp = np.asarray(self.coarse_orienter.predict_probs(image)).ravel()
                ck = int(np.argmax(cp))
                if ck in (1, 3) and float(cp[ck]) >= 0.5:
                    image = np.ascontiguousarray(np.rot90(image, k=(-ck) % 4))
                    res.pre_rotation_k = (-ck) % 4

            # Stage 1: count + layout
            count, cconf = self.counter.predict(image)
            res.page_count, res.count_conf = count, cconf
            if count == PageCount.REJECT:
                res.error = "rejected_non_page"
                return res

            # Stage 3: detect + segment pages
            boxes = self.segmenter.detect(image, max_pages=2 if count == PageCount.TWO else 1)
            if not boxes:
                res.error = "no_page_detected"
                return res
            # count says two folios but detection under-segmented to one box:
            # halve the FULL FRAME (not the under-covering detected box, which can
            # clip a page's outer edge) so segment()'s page mask — which reaches
            # further than the box — is captured on both sides; the seam search
            # then refines the true spine.
            if count == PageCount.TWO and len(boxes) == 1:
                h, w = image.shape[:2]
                boxes = _halve_box(PageBox(0, 0, w, h, boxes[0].score))
            masks = self.segmenter.segment(image, boxes)

            # Learned folio segmenter (precise, background-agnostic page boundary)
            # supersedes the legacy crop mask. It returns the whole paper region
            # (both pages on a spread); the seam split below divides it per folio.
            learned = None
            if self.folio_segmenter is not None:
                lp = self.folio_segmenter.page_mask(image)
                if lp.any() and float(lp.mean()) > 0.05:
                    masks = [lp for _ in masks]
                    learned = lp   # precise page region, for the background white-out

            # Stage 2 global skew estimate (from union mask) -> recorded only;
            # fine skew is applied per-folio in Stage 5 for accuracy.
            union = np.zeros(image.shape[:2], np.uint8)
            for m in masks:
                union |= m

            # Stage 4: gutter split for two-folio spreads
            page_specs = []
            if count == PageCount.TWO and len(boxes) == 2:
                seam = spine.detect_gutter(
                    image, boxes[0].as_tuple(), boxes[1].as_tuple(), mask=union,
                    band_margin_frac=self.cfg.geom.gutter_band_margin_frac,
                    seam_smoothness=self.cfg.geom.seam_smoothness,
                    energy_weights=self.cfg.geom.energy_weights,
                )
                res.gutter_seam = seam.tolist()
                # restrict each page's mask to its side of the seam, then bake an
                # ASYMMETRIC crop margin into the mask: dilate to give the 3 outer
                # edges their protective margin (frayed edges / marginalia), then
                # re-clip at the seam so the SPINE edge stays exactly at the gutter
                # — no facing-page sliver. _finish_page then crops with margin 0.
                xs = np.arange(image.shape[1])[None, :]
                left_sel = (xs <= seam[:, None]).astype(np.uint8)
                right_sel = 1 - left_sel
                # white-out mask = the TRUE learned page on this side (no outer
                # margin), so binding/background go but marginalia stays
                wmA = (learned * left_sel) if learned is not None else None
                wmB = (learned * right_sel) if learned is not None else None
                page_specs = [
                    ("A", self._margin_to_seam(masks[0], left_sel), 0.0, wmA),
                    ("B", self._margin_to_seam(masks[1], right_sel), 0.0, wmB),
                ]
            else:
                page_specs = [("", masks[0], None, None)]

            # Stages 3b/5/6/7 per page
            for label, mask, mfrac, wmask in page_specs:
                folio = self._finish_page(image, mask, label, margin_frac=mfrac,
                                          white_mask=wmask)
                if folio is not None:
                    # provenance: map the crop quad (in the possibly pre-rotated
                    # working frame) back to the ORIGINAL image, as [0,1] ratios.
                    if folio.crop_quad_norm is not None:
                        q = _unrotate_quad(folio.crop_quad_norm, res.pre_rotation_k,
                                           orig_w, orig_h)
                        folio.crop_quad_norm = [
                            [round(min(1.0, max(0.0, x / orig_w)), 5),
                             round(min(1.0, max(0.0, y / orig_h)), 5)] for x, y in q]
                    folio.source_size = [int(orig_w), int(orig_h)]
                    res.folios.append(folio)

            if not res.folios:
                res.error = "no_valid_folio"
            return res
        except Exception as e:  # robustness: one bad image never kills the run
            res.error = f"exception:{type(e).__name__}:{e}"
            return res

    def _orientation_view(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Full-height view of the folio's horizontal slice, for the 4-way head.

        The head was trained on FULL single-folio scans (page + scan border), so
        the tight oriented crop — which strips that context — flips its 0-vs-180
        decision (verified: head is 0.99-correct on the full image, wrong on the
        tight crop). For a single folio this returns ~the whole image; for a
        two-folio side it returns that side's full-height slice. Preserves image
        up/down, so the chosen quarter-turn still applies to the oriented crop."""
        h, w = image.shape[:2]
        cols = np.where(mask.any(axis=0))[0]
        if cols.size == 0:
            return image
        x0, x1 = int(cols.min()), int(cols.max())
        # wide margin so a single folio gets ~the whole scan (page + border),
        # which is the framing the head was trained on
        pad = int(0.5 * (x1 - x0))
        return image[:, max(0, x0 - pad):min(w, x1 + pad)]

    @staticmethod
    def _recover_page_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Guard against the legacy segmenter under-covering a sparse, multi-block
        page (masking only the densest text block, which then crops a portrait
        folio down to a landscape band). If the masked region is only a band of a
        much taller bright page, expand it to that full bright page WITHIN the
        folio's own column span (so a two-folio side never pulls in its neighbour).
        A no-op whenever the mask already covers the page."""
        import cv2
        # Use the mask's LARGEST CONNECTED COMPONENT, since that is what
        # oriented_page_quad crops to. On a sparse page the mask fragments into
        # separate text blocks, so its overall bbox can span the page while the
        # largest component is only one band.
        nm, lm, sm, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
        if nm <= 1:
            return mask
        mbig = 1 + int(np.argmax(sm[1:, cv2.CC_STAT_AREA]))
        mx0 = int(sm[mbig, cv2.CC_STAT_LEFT]); mx1 = mx0 + int(sm[mbig, cv2.CC_STAT_WIDTH]) - 1
        mask_h = int(sm[mbig, cv2.CC_STAT_HEIGHT])
        band = image[:, mx0:mx1 + 1]
        gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY) if band.ndim == 3 else band
        blur = cv2.GaussianBlur(gray, (0, 0), max(1.5, 0.004 * min(gray.shape[:2])))
        _, bright = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bk = max(3, int(0.02 * min(gray.shape[:2])) | 1)
        bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bk, bk)))
        n, lb, st, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
        if n <= 1:
            return mask
        big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        bh = int(st[big, cv2.CC_STAT_HEIGHT])
        area = int(st[big, cv2.CC_STAT_AREA])
        # recover only when the bright page is much taller than the masked band
        # AND is a substantial region (a real page, not a stripe of noise)
        if bh > 1.5 * max(mask_h, 1) and area > 0.30 * band.shape[0] * band.shape[1]:
            new = np.zeros(mask.shape[:2], np.uint8)
            new[:, mx0:mx1 + 1] = (lb == big).astype(np.uint8)
            return new
        return mask

    @staticmethod
    def _enforce_page_aspect(mask: np.ndarray, img_hw, max_aspect: float) -> np.ndarray:
        """Anti-over-crop: if the page mask is wider than ``max_aspect`` (w/h) —
        a sparse page where text doesn't fill the sheet, so the crop would come
        out square — EXTEND it vertically (into the rest of the folio, clamped to
        the image) toward a portrait shape. Only ever adds area, so no content is
        lost. Biases the extension to whichever side has room (content at the top
        of the page extends downward). No-op when already portrait enough."""
        if not max_aspect or max_aspect <= 0:
            return mask
        ys = np.where(mask.any(axis=1))[0]
        xs = np.where(mask.any(axis=0))[0]
        if ys.size == 0 or xs.size == 0:
            return mask
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        w, h = x1 - x0 + 1, y1 - y0 + 1
        if w <= max_aspect * h:
            return mask                              # already portrait enough
        H = int(img_hw[0])
        extra = int(round(w / max_aspect)) - h
        top, bot = extra // 2, extra - extra // 2
        top = min(top, y0); bot = min(bot, H - 1 - y1)
        deficit = extra - (top + bot)                # redistribute to the side with room
        if deficit > 0:
            add_top = min(deficit, y0 - top); top += add_top
            bot = min(H - 1 - y1, bot + (deficit - add_top))
        out = mask.copy()
        out[y0 - top:y1 + bot + 1, x0:x1 + 1] = 1
        return out

    def _margin_to_seam(self, mask: np.ndarray, sel: np.ndarray) -> np.ndarray:
        """Build a two-folio page mask with an ASYMMETRIC margin: keep only this
        page's side of the seam (``sel`` = 1 on this side), dilate by the crop
        margin to give the outer edges breathing room, then re-clip at the seam so
        the spine edge is exactly at the gutter (no facing-page sliver)."""
        import cv2
        m = (mask.astype(bool) & sel.astype(bool)).astype(np.uint8)
        ys = np.where(m.any(axis=1))[0]
        xs = np.where(m.any(axis=0))[0]
        if ys.size == 0 or xs.size == 0:
            return m
        diag = float(np.hypot(int(ys.max()) - int(ys.min()), int(xs.max()) - int(xs.min())))
        k = max(1, int(self.cfg.geom.crop_margin_frac * diag))
        d = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
        return (d.astype(bool) & sel.astype(bool)).astype(np.uint8)

    def _finish_page(self, image: np.ndarray, mask: np.ndarray,
                     label: str, margin_frac: Optional[float] = None,
                     white_mask: Optional[np.ndarray] = None) -> Optional[FolioResult]:
        g = self.cfg.geom
        q = self.cfg.quality
        if margin_frac is None:
            margin_frac = g.crop_margin_frac
        # recover a band mask the legacy segmenter may have produced on a sparse page
        mask = self._recover_page_mask(image, mask)
        # the precise page region — used to white-out every non-folio pixel
        # (background, facing-page sliver, binding). For two-folio sides the caller
        # passes the un-margined learned half so binding goes but marginalia stays.
        true_mask = white_mask if white_mask is not None else mask.copy()
        # keep the full folio: extend a square (sparse-page) crop toward portrait
        mask = self._enforce_page_aspect(mask, image.shape[:2], g.max_crop_aspect)
        # oriented crop quad (Stage 3 boundary math) + single warp. ``margin_frac``
        # is 0 for two-folio sides (the margin is already baked into the mask,
        # asymmetrically, so the spine edge stays tight at the gutter).
        quad = geometry.oriented_page_quad(mask, margin_frac=margin_frac)
        crop_H, cw, ch = geometry.crop_homography(quad)
        # provisional upright crop (used for the final geometry + skew)
        provisional = geometry.compose_and_warp(image, crop_H, cw, ch)
        import cv2

        # Stage 5a: 4-way. Decide orientation on a GENEROUS page view, not the
        # tight oriented crop: the tight crop loses the page-context cues the head
        # was trained on and can flip its 0-vs-180 call (verified). Both views
        # preserve image up/down, so the chosen quarter-turn applies to either.
        orient_view = self._orientation_view(image, mask)
        probs = self.orienter.predict_probs(orient_view)
        gray = cv2.cvtColor(provisional, cv2.COLOR_BGR2GRAY)
        k, oconf = orient.resolve_quarter_turn(probs, gray)
        # Stage 5b: fine skew on the (mentally) rotated page
        rotated_gray = np.rot90(gray, k=(-k) % 4)
        skew = orient.estimate_skew(rotated_gray, max_deg=g.skew_max_deg,
                                    step=g.skew_step_deg)

        # final single warp: crop -> quarter turn -> skew, at full res
        final = geometry.compose_and_warp(image, crop_H, cw, ch,
                                          quarter_k=(-k) % 4, skew_deg=skew)

        # White-out non-folio pixels using the precise LEARNED page mask: every
        # pixel outside the folio (background, facing-page sliver, binding, the
        # portrait aspect-padding) becomes white, leaving only the folio. Only
        # with the learned segmenter (its mask is precise enough to trust);
        # dilated a hair so the page edge is never clipped.
        masked_out = False
        if getattr(g, "mask_background", True) and self.folio_segmenter is not None:
            fm = geometry.compose_and_warp(
                (true_mask > 0).astype(np.uint8) * 255, crop_H, cw, ch,
                quarter_k=(-k) % 4, skew_deg=skew,
                interp=cv2.INTER_NEAREST, border=cv2.BORDER_CONSTANT, border_value=0)
            # Never erase page interior: fill enclosed holes AND concavities (convex
            # hull of the page region). Faded parchment under-segments into open bays
            # that reach the margin; the raw mask would white-out real text there.
            fm = _safe_page_mask(fm)
            grow = max(3, int(0.004 * min(final.shape[:2])) | 1)
            fm = cv2.dilate(fm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow)))
            final[fm < 127] = 255
            masked_out = True

        # Background tightening (skipped when we white-out — the mask is
        # authoritative): drop dark non-information the page mask let through.
        if not masked_out and getattr(g, "trim_background", True):
            from .stages import content as _content
            box = _content.paper_box(final)
            if box is None:
                tx0, ty0, tx1, ty1 = _content.trim_background_border(final)
            else:
                tx0, ty0, tx1, ty1 = box
            # Anti-over-crop: the background trim removes side margins, but on a
            # sparse page it would also crop the blank balance of the sheet and
            # leave a square. Don't let it make the crop squarer than max_crop_aspect
            # -- keep the page height (final already spans the full folio), only
            # trim the sides. Adds no pixels beyond `final`, so never invents content.
            ma = g.max_crop_aspect
            if ma and (tx1 - tx0) > ma * (ty1 - ty0):
                need = int(round((tx1 - tx0) / ma))
                cy = (ty0 + ty1) // 2
                ty0 = max(0, cy - need // 2)
                ty1 = min(final.shape[0], ty0 + need)
                ty0 = max(0, ty1 - need)
            if (tx1 - tx0) >= 16 and (ty1 - ty0) >= 16:
                final = final[ty0:ty1, tx0:tx1]

        # Tight crop to the detected text region (learned). Runs last, on the
        # upright single-folio crop. Gracefully no-ops (keeps the looser crop)
        # when EasyOCR is unavailable or finds no text -> never clips content.
        if getattr(g, "tight_crop", False):
            from .stages import textregion as _tr
            tb = _tr.text_crop_box(final, margin_frac=g.tight_crop_margin_frac)
            if tb is not None and (tb[2] - tb[0]) >= 16 and (tb[3] - tb[1]) >= 16:
                final = final[tb[1]:tb[3], tb[0]:tb[2]]

        # Cap the output aspect (long:short) for the transcription backend
        # (<= 10:24); pad with white, never crop, so nothing is lost.
        final = _fit_max_aspect(final, getattr(g, "max_output_ratio", 0.0))

        folio = FolioResult(label=label, crop=final,
                            rotation_deg=(90.0 * ((-k) % 4) + skew),
                            orientation_conf=oconf)
        # provenance: the crop quad in the (working) frame; process_image maps it
        # back to ORIGINAL-image [0,1] ratios. quad order is TL, TR, BR, BL.
        folio.crop_quad_norm = quad.tolist()
        # Stage 5c: content vs blank/non-content (so Archivault can skip blanks)
        if self.blank_classifier is not None:
            folio.is_blank, folio.blank_conf = self.blank_classifier.predict(final)

        # Stage 6: optional dewarp gate
        if self.cfg.enable_dewarp and self.dewarper is not None:
            curv = _boundary_curvature(mask)
            if curv > g.dewarp_curvature_thresh:
                folio.crop = self.dewarper.unwarp(folio.crop)

        # quality gates (Stage 7)
        h, w = final.shape[:2]
        area_frac = float(mask.sum()) / float(image.shape[0] * image.shape[1])
        folio.mask_area_frac = area_frac
        reasons = []
        # near-blank pages carry too little text for the 4-way head to be
        # trusted (it can be confidently 180-wrong) -> route to human review.
        text_frac = float(orient._text_ink(rotated_gray).mean()) / 255.0
        folio.text_frac = text_frac
        if text_frac < q.min_text_frac_for_orient:
            reasons.append("low_text_for_orientation")
        if oconf < q.min_orientation_conf:
            reasons.append("low_orientation_conf")
        if area_frac < q.min_page_area_frac:
            reasons.append("page_too_small")
        aspect = w / float(h)
        if not (q.portrait_aspect_range[0] <= aspect <= q.portrait_aspect_range[1]):
            reasons.append("unexpected_aspect")
        folio.needs_review = len(reasons) > 0
        folio.review_reasons = reasons
        return folio

    # -------------------------------------------------------------- async run
    async def run_s3(self, limit: Optional[int] = None,
                     shard: Optional[tuple] = None, resume: bool = False) -> dict:
        """Stream the input bucket through the pipeline, writing crops + sidecars
        back to S3. Returns aggregate counters. ``limit`` caps how many source
        images are processed (safe dry run). ``shard=(i, n)`` processes only the
        keys assigned to worker ``i`` of ``n`` (stable hash partition) so N
        EC2 / AWS Batch workers can split one corpus with no coordination.
        ``resume`` skips inputs whose output already exists (spot-safe reruns)."""
        import zlib
        from .io.s3_async import S3Streamer
        self._ensure_models()
        streamer = S3Streamer(self.cfg)
        sem = asyncio.Semaphore(self.cfg.s3.upload_concurrency)
        stats = {"processed": 0, "folios": 0, "review": 0, "errors": 0, "skipped": 0}
        shard_i, shard_n = (shard or (0, 1))
        done = await streamer.collect_done_stems() if resume else set()

        def _mine(key: str) -> bool:
            return shard_n <= 1 or (zlib.crc32(key.encode()) % shard_n) == shard_i

        async def handle(item):
            # GPU work is sync; offload to a thread so the event loop keeps
            # downloading/uploading concurrently.
            res = await asyncio.to_thread(self.process_image, item.key, item.image)
            if res.error:
                stats["errors"] += 1
            sidecar = res.sidecar()
            async with sem:
                for i, f in enumerate(res.folios):
                    suffix = f"-{f.label}" if f.label else ""
                    out_key = f"{_stem(item.key)}{suffix}.jpg"
                    await streamer.upload(out_key, f.crop, {**sidecar, "folio_index": i},
                                          review=f.needs_review)
                    stats["folios"] += 1
                    stats["review"] += int(f.needs_review)
            stats["processed"] += 1

        tasks: List[asyncio.Task] = []
        submitted = 0
        async for item in streamer.stream_images():
            if not _mine(item.key):          # not this shard's key -> skip
                continue
            if done and _stem(item.key) in done:   # already processed (resume)
                stats["skipped"] += 1
                continue
            tasks.append(asyncio.create_task(handle(item)))
            submitted += 1
            if len(tasks) >= self.cfg.model.gpu_batch_size * 4:
                done = [t for t in tasks if t.done()]
                for t in done:
                    await t
                tasks = [t for t in tasks if not t.done()]
            if limit is not None and submitted >= limit:
                break
        if tasks:
            await asyncio.gather(*tasks)
        return stats


def _halve_box(b: PageBox) -> List[PageBox]:
    """Split one page box into left/right halves at its horizontal midpoint
    (used when a two-folio spread was detected as a single box; the seam search
    then refines the true gutter inside the overlap)."""
    mid = (b.x1 + b.x2) // 2
    ov = int(0.04 * (b.x2 - b.x1))         # small overlap so the seam can wander
    return [PageBox(b.x1, b.y1, min(mid + ov, b.x2), b.y2, b.score),
            PageBox(max(mid - ov, b.x1), b.y1, b.x2, b.y2, b.score)]


def _safe_page_mask(fm: np.ndarray) -> np.ndarray:
    """Make a white-out mask that can NEVER erase page interior: fill enclosed holes,
    then take the convex hull of the substantial page region(s). On faded parchment the
    segmenter under-covers the page (open bays reaching the margin) and the raw mask
    would white-out real text; the hull bridges those gaps. Slight over-extension at
    torn corners (keeps a little background) is acceptable — never losing text is not."""
    import cv2, os
    if os.environ.get("FOLIO_KEEP_ALL"):          # QA ground truth: keep every pixel,
        return np.full_like(fm, 255)              # same geometry as safe, zero erasure
    fm = _fill_mask_holes(fm)
    num, lab, stats, _ = cv2.connectedComponentsWithStats((fm > 0).astype(np.uint8), 8)
    if num < 2:
        return fm
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    thr = 0.12 * stats[big, cv2.CC_STAT_AREA]              # keep components >=12% of largest
    keep = np.zeros_like(fm)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= thr:
            keep[lab == i] = 255
    pts = cv2.findNonZero(keep)
    if pts is None or len(pts) < 3:
        return fm
    hull = cv2.convexHull(pts)
    out = np.zeros_like(fm)
    cv2.fillConvexPoly(out, hull, 255)
    # union the hull with the full page mask so NO detected page fragment (incl. small
    # components the hull-of-substantial-parts would drop) is ever erased -> safe >= raw mask
    return cv2.bitwise_or(out, fm)


def _fill_mask_holes(fm: np.ndarray) -> np.ndarray:
    """Fill interior holes in a binary (0/255) page mask. Background fully enclosed
    by the page is a segmentation error (a bleached/faded patch read as non-page);
    flood-fill the exterior from the border and treat whatever it can't reach as
    page, so the white-out never punches a hole through the folio."""
    import cv2
    pad = cv2.copyMakeBorder(fm, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    ff = pad.copy()
    m = np.zeros((ff.shape[0] + 2, ff.shape[1] + 2), np.uint8)
    cv2.floodFill(ff, m, (0, 0), 255)            # exterior background -> 255
    holes = cv2.bitwise_not(ff)                  # only interior holes remain set
    out = cv2.bitwise_or(pad, holes)
    return out[1:-1, 1:-1]


def _fit_max_aspect(img: np.ndarray, max_ratio: float, fill: int = 255) -> np.ndarray:
    """Pad (with white) so the long:short side ratio never exceeds ``max_ratio``
    (e.g. 24/10 for the HTR backend's 10:24 limit). Padding only — never crops."""
    if not max_ratio or max_ratio <= 0:
        return img
    import cv2
    h, w = img.shape[:2]
    long_, short = max(h, w), min(h, w)
    if long_ <= max_ratio * short:
        return img
    target = int(np.ceil(long_ / max_ratio))
    pad = target - short
    a, b = pad // 2, pad - pad // 2
    if h >= w:                                   # tall -> pad width
        return cv2.copyMakeBorder(img, 0, 0, a, b, cv2.BORDER_CONSTANT, value=(fill, fill, fill))
    return cv2.copyMakeBorder(img, a, b, 0, 0, cv2.BORDER_CONSTANT, value=(fill, fill, fill))


def _unrotate_quad(quad, k: int, orig_w: int, orig_h: int):
    """Map quad points from a working image that was ``np.rot90(orig, k)`` back to
    the ORIGINAL image's pixel coordinates."""
    k = int(k) % 4
    out = []
    for x, y in quad:
        if k == 0:
            ox, oy = x, y
        elif k == 1:
            ox, oy = orig_w - 1 - y, x
        elif k == 2:
            ox, oy = orig_w - 1 - x, orig_h - 1 - y
        else:
            ox, oy = y, orig_h - 1 - x
        out.append((ox, oy))
    return out


def _stem(key: str) -> str:
    base = key.rsplit("/", 1)[-1]
    return base.rsplit(".", 1)[0]


def _boundary_curvature(mask: np.ndarray) -> float:
    """Rough curvature proxy: deviation of the page's top edge from a straight
    line, normalised by width. Cheap gate for the (expensive) dewarp model."""
    ys = []
    h, w = mask.shape[:2]
    for x in range(0, w, max(w // 64, 1)):
        col = np.where(mask[:, x] > 0)[0]
        ys.append(col[0] if col.size else np.nan)
    ys = np.array(ys, dtype=np.float64)
    ys = ys[~np.isnan(ys)]
    if ys.size < 4:
        return 0.0
    line = np.linspace(ys[0], ys[-1], ys.size)
    return float(np.max(np.abs(ys - line)) / max(h, 1))
