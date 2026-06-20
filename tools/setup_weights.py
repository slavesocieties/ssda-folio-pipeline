"""Fetch the model weights a fresh clone needs (they're gitignored — too large
for git). Cross-platform; run once after installing.

    python tools/setup_weights.py

Downloads:
  * legacy .pth (segmenter + count + orient adapters) -> ../legacy_weights/
    from the Google Drive IDs in tools/legacy_model_ids.txt
  * the trained heads (orientation / count / blank) -> weights/
    from $FOLIO_WEIGHTS_BASE if set, else prints where to get them.

Hosting the trained heads for the lab (pick one):
  * GitHub Release assets (recommended): attach the three .pt files to a release,
    then set FOLIO_WEIGHTS_BASE to the release download URL prefix, e.g.
      https://github.com/<org>/ssda-folio-pipeline/releases/download/v1.0
  * Google Drive: add their IDs to tools/legacy_model_ids.txt-style lines below.
"""
from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TRAINED = ["orientation4_convnextv2.pt", "folio_count_convnextv2.pt", "blank_convnextv2.pt"]


def _gdown(file_id: str, dest: Path):
    import gdown
    dest.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(id=file_id, output=str(dest), quiet=False)


def fetch_legacy():
    ids = REPO / "tools" / "legacy_model_ids.txt"
    out = REPO.parent / "legacy_weights"
    print(f"\n== legacy .pth -> {out} ==")
    for line in ids.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        name, fid = parts[0], parts[1]
        dest = out / name
        if dest.exists():
            print(f"  have {name}")
            continue
        try:
            _gdown(fid, dest)
        except Exception as e:
            print(f"  FAILED {name}: {e} (is gdown installed? pip install gdown)")


def fetch_trained():
    out = REPO / "weights"
    out.mkdir(parents=True, exist_ok=True)
    base = os.environ.get("FOLIO_WEIGHTS_BASE", "").rstrip("/")
    print(f"\n== trained heads -> {out} ==")
    for name in TRAINED:
        dest = out / name
        if dest.exists():
            print(f"  have {name}")
            continue
        if base:
            url = f"{base}/{name}"
            try:
                print(f"  downloading {url}")
                urllib.request.urlretrieve(url, dest)
            except Exception as e:
                print(f"  FAILED {name}: {e}")
        else:
            print(f"  MISSING {name} — set FOLIO_WEIGHTS_BASE to a URL prefix, or "
                  f"copy {name} into {out} manually.")


def main():
    fetch_legacy()
    fetch_trained()
    missing = [n for n in TRAINED if not (REPO / "weights" / n).exists()]
    print("\nDone." + (f" Still missing: {missing}" if missing else " All weights present."))
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
