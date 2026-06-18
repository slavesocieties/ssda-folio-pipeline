"""Find SPARSE folios from the automated transcriptions.

A "sparse" folio is a page with little or no writing (blank versos, ledger
pages with a couple of names, covers, back-matter). These are where the
orientation head's up/down signal is weak, so this list lets us (a) measure the
true residual orientation-error rate and (b) build a targeted training/validation
set for the sparse case.

Input: a folder of per-volume transcription JSONs, each a list of records:
    [{"file": "74234-0002.jpg", "transcription": "Rosa\\nMaria\\nJose", ...}, ...]
(the SSDA Archivault transcription output format).

Output: a CSV of images sorted by transcribed-text length, with everything below
``--max-chars`` flagged sparse, plus a plain image-id list for the sparse ones
(ready to pull from S3).

    python tools/find_sparse_folios.py /path/to/transcriptions/json \\
        --max-chars 120 --out sparse_folios
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def _text_len(rec) -> tuple[int, int]:
    t = (rec.get("transcription") or "").strip()
    chars = len(re.sub(r"\s+", " ", t))          # collapse whitespace
    words = len(t.split())
    return chars, words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_dir", help="folder of per-volume transcription .json files")
    ap.add_argument("--max-chars", type=int, default=120,
                    help="images with <= this many transcribed chars are 'sparse'")
    ap.add_argument("--out", default="sparse_folios")
    args = ap.parse_args()

    rows = []
    files = sorted(Path(args.json_dir).glob("*.json"))
    for jf in files:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"skip {jf.name}: {e}")
            continue
        vol = jf.stem
        for rec in data:
            img = rec.get("file")
            if not img:
                continue
            chars, words = _text_len(rec)
            rows.append({"image": img, "volume": vol, "chars": chars, "words": words,
                         "sparse": chars <= args.max_chars})

    rows.sort(key=lambda r: r["chars"])
    sparse = [r for r in rows if r["sparse"]]

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    with open(out / "all_folios.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["image", "volume", "chars", "words", "sparse"])
        w.writeheader(); w.writerows(rows)
    (out / "sparse_images.txt").write_text("\n".join(r["image"] for r in sparse), encoding="utf-8")

    n = len(rows)
    print(f"scanned {len(files)} volume(s), {n} folio(s)")
    print(f"sparse (<= {args.max_chars} chars): {len(sparse)} ({100*len(sparse)/max(n,1):.1f}%)")
    print("lowest-text examples:")
    for r in rows[:12]:
        print(f"   {r['image']:20s} chars={r['chars']:5d} words={r['words']:3d}")
    print(f"\nwrote {out/'sparse_images.txt'} ({len(sparse)} ids) and {out/'all_folios.csv'}")


if __name__ == "__main__":
    main()
