"""End-to-end labelled evaluation of the full folio pipeline.

Unlike ``folio.training.eval`` (which scores one head in isolation on synthetic
rotations), this runs the WHOLE production pipeline per image and scores it
against ground-truth labels: folio **count**, **orientation** (did the output
come out upright?), two-folio **split**, and the **review** flag rate.

Label sources
-------------
1. ``--from-folders ROOT`` (default ROOT = train_data): uses the labelled
   folders we already have —
     rightside_up/  -> one_folio, upright (input_k=0)
     upside_down/   -> one_folio, 180     (input_k=2)
     two_folios/    -> two_folios
   With ``--landscape`` a fraction of the singles are synthetically rotated to
   90/270 to exercise the landscape orient-before-segment pre-pass.
2. ``--manifest CSV`` with columns ``path,count,input_k`` where count is
   ``one_folio``/``two_folios`` and input_k in {0,1,2,3} = quarter-turns the
   input is rotated CCW from upright (np.rot90 convention).

Drop in a fresh labelled validation set the same way to get true cross-source
numbers.

    python tools/evaluate.py --from-folders ../train_data --n 60 --landscape \
        --legacy-weights ../legacy_weights --out eval_pipeline
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from folio import process as P
from folio.schemas import PageCount


def _applied_quarters(res, folio) -> int:
    """Total upright-correction quarter-turns the pipeline applied (CCW)."""
    return (res.pre_rotation_k + round(folio.rotation_deg / 90.0)) % 4


def _build_labels(args):
    """Returns (path, input_k, count_label, synth, rotate_by).

    input_k  = the input's TRUE orientation (quarter-turns CCW from upright).
    rotate_by = how much to rotate the loaded file to realise input_k.
      - rightside_up files are already upright (k=0); rotate_by==input_k.
      - upside_down files are already at 180 (k=2); rotate_by==0 (do NOT rotate
        again — that was a bug that double-flipped them to upright).
    """
    items = []
    if args.manifest:
        for row in csv.DictReader(open(args.manifest)):
            k = int(row.get("input_k", 0))
            items.append((row["path"], k, row["count"], k != 0, k))
        return items
    root = Path(args.from_folders)
    rng = random.Random(0)

    def sample(folder, n):
        fs = sorted((root / folder).glob("*.jpg"))
        rng.shuffle(fs)
        return fs[:n]

    for p in sample("rightside_up", args.n):
        k = 0
        if args.landscape and rng.random() < 0.5:   # synthesise 90/270 inputs
            k = rng.choice([1, 3])
        items.append((str(p), k, "one_folio", k != 0, k))   # rotate_by = k
    for p in sample("upside_down", args.n):
        items.append((str(p), 2, "one_folio", False, 0))    # already at 180
    for p in sample("two_folios", args.n // 2):
        items.append((str(p), 0, "two_folios", False, 0))
    rng.shuffle(items)
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-folders", default="../train_data")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--n", type=int, default=60, help="samples per class")
    ap.add_argument("--landscape", action="store_true",
                    help="synthetically rotate some singles to 90/270")
    ap.add_argument("--legacy-weights", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="eval_pipeline")
    args = ap.parse_args()

    cfg = P.make_config(device=args.device)
    legacy = P.find_legacy_weights(args.legacy_weights, args.from_folders)
    pipe, mode = P.build_pipeline(cfg, legacy, prepass=True)
    print(f"config: {mode}  device={cfg.model.device}")

    items = _build_labels(args)
    print(f"evaluating {len(items)} labelled images...\n")

    # counters
    cnt_ok = cnt_tot = 0
    ori_ok = ori_tot = 0
    ori_ok_land = ori_tot_land = 0
    split_ok = split_tot = 0
    review = 0
    # orientation accuracy broken down by the input's true orientation
    by_in = {0: [0, 0], 2: [0, 0], "land": [0, 0]}   # bucket -> [ok, tot]
    rev_by_in = {0: 0, 2: 0, "land": 0}
    rows = []
    cm_count = {("one_folio", "one_folio"): 0, ("one_folio", "two_folios"): 0,
                ("two_folios", "one_folio"): 0, ("two_folios", "two_folios"): 0}

    for path, input_k, count_lbl, synth, rotate_by in items:
        img = cv2.imread(path)
        if img is None:
            continue
        if rotate_by:                             # synthesise the rotated input
            img = np.ascontiguousarray(np.rot90(img, rotate_by))
        res = pipe.process_image(Path(path).name, img)
        pred_count = res.page_count.value
        cnt_tot += 1
        cnt_ok += int(pred_count == count_lbl)
        if (count_lbl, pred_count) in cm_count:
            cm_count[(count_lbl, pred_count)] += 1

        if count_lbl == "two_folios":
            split_tot += 1
            split_ok += int(len(res.folios) == 2)
        elif res.folios:                          # single folio -> orientation
            f = res.folios[0]
            upright = (input_k + _applied_quarters(res, f)) % 4 == 0
            ori_tot += 1; ori_ok += int(upright)
            bucket = "land" if input_k in (1, 3) else input_k
            by_in[bucket][1] += 1; by_in[bucket][0] += int(upright)
            rev_by_in[bucket] += int(f.needs_review)
            if synth:
                ori_tot_land += 1; ori_ok_land += int(upright)
            review += int(f.needs_review)
        rows.append({"file": Path(path).name, "input_k": input_k,
                     "count_label": count_lbl, "count_pred": pred_count,
                     "n_folios": len(res.folios)})

    def pct(a, b):
        return f"{100*a/max(b,1):.1f}% ({a}/{b})"

    print("=== RESULTS ===")
    print(f"folio count accuracy : {pct(cnt_ok, cnt_tot)}")
    print(f"  confusion (true->pred): "
          f"1->1 {cm_count[('one_folio','one_folio')]}, 1->2 {cm_count[('one_folio','two_folios')]}, "
          f"2->1 {cm_count[('two_folios','one_folio')]}, 2->2 {cm_count[('two_folios','two_folios')]}")
    print(f"two-folio split (==2): {pct(split_ok, split_tot)}")
    print(f"orientation upright  : {pct(ori_ok, ori_tot)}")
    print(f"  by input orientation:")
    print(f"    upright input  (0,   should stay) : {pct(*by_in[0])}   review {pct(rev_by_in[0], by_in[0][1])}")
    print(f"    upside-down    (180, should flip) : {pct(*by_in[2])}   review {pct(rev_by_in[2], by_in[2][1])}")
    print(f"    landscape      (90/270, pre-pass) : {pct(*by_in['land'])}   review {pct(rev_by_in['land'], by_in['land'][1])}")
    print(f"single-folio review-flag rate: {pct(review, ori_tot)}")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    summary = {"count_acc": cnt_ok / max(cnt_tot, 1), "count_n": cnt_tot,
               "split_acc": split_ok / max(split_tot, 1), "split_n": split_tot,
               "orientation_acc": ori_ok / max(ori_tot, 1), "orientation_n": ori_tot,
               "orientation_landscape_acc": ori_ok_land / max(ori_tot_land, 1),
               "orientation_landscape_n": ori_tot_land,
               "review_rate": review / max(ori_tot, 1)}
    (out / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    with open(out / "eval_rows.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out/'eval_summary.json'}")


if __name__ == "__main__":
    main()
