"""Backward-compatible shim. The tool now lives in the package: ``folio.cli``.

    python tools/folio_process.py <image|dir|s3://...> --out ...
is equivalent to
    python -m folio.cli <image|dir|s3://...> --out ...
or, once installed (`pip install -e .`), simply:
    folio <image|dir|s3://...> --out ...
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from folio.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
