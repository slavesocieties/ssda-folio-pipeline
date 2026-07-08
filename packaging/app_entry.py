"""Entry point for the packaged Folio Processor executable (PyInstaller).

When frozen, all Python deps are bundled. This entry:
  * points the pipeline at a `weights/` folder NEXT TO the .exe (writable),
  * downloads the model weights there on first run (~416 MB, from the GitHub release),
  * launches the local web app and opens the browser.

Not used in normal source runs (that's run_app.py / folio-web).
"""
import os
import sys
import urllib.request
from pathlib import Path

RELEASE = "https://github.com/slavesocieties/ssda-folio-pipeline/releases/download/weights-v1"
WEIGHTS = [
    "folio_seg_unet.pt.ts.pt",
    "orientation4_convnextv2.pt",
    "folio_count_convnextv2.pt",
    "blank_convnextv2.pt",
]


def _base_dir() -> Path:
    # folder containing the .exe (onedir bundle) — writable, user-visible
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def _ensure_weights(wdir: Path) -> None:
    wdir.mkdir(parents=True, exist_ok=True)
    missing = [w for w in WEIGHTS if not (wdir / w).exists()]
    if not missing:
        return
    print(f"First run: downloading {len(missing)} model file(s) (~416 MB) to {wdir}")
    print("This happens once. Please keep this window open…\n")
    for w in missing:
        dest, tmp = wdir / w, wdir / (w + ".part")
        print(f"  {w} …", end="", flush=True)
        with urllib.request.urlopen(f"{RELEASE}/{w}") as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length", 0)); done = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk); done += len(chunk)
                if total:
                    print(f"\r  {w} … {done * 100 // total}%", end="", flush=True)
        tmp.replace(dest)
        print(f"\r  {w} … done      ")


def main() -> None:
    base = _base_dir()
    wdir = base / "weights"
    os.environ["FOLIO_WEIGHTS_DIR"] = str(wdir)
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    try:
        _ensure_weights(wdir)
    except Exception as e:
        print(f"\n[!] Could not download weights automatically: {e}")
        print(f"    Manually download these into {wdir} and re-run:")
        for w in WEIGHTS:
            print(f"      {RELEASE}/{w}")
        input("\nPress Enter to exit."); return
    from folio.webapp import main as web_main
    web_main([])


if __name__ == "__main__":
    main()
