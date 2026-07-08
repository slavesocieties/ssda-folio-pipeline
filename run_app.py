#!/usr/bin/env python3
"""One-step launcher for the Folio Processor.

Double-click "Run Folio App.bat" (Windows) or "run_folio_app.command" (macOS),
or run `python run_app.py`. It installs anything missing INTO YOUR CURRENT Python
(so it reuses an existing GPU torch build instead of replacing it), downloads the
model weights the first time, then opens the web app in your browser.

No coding, no LLM — this is the normal way to run the tool.
"""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def have(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except Exception:
        return False


def pip(*args: str) -> None:
    print(f"\n>>> installing: {' '.join(args)}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", *args])


def main() -> int:
    print("Folio Processor — setup & launch")
    print(f"Python: {sys.executable}\n")

    # 1) Heavy core deps (torch/opencv/etc.) — only if genuinely missing, so an
    #    existing CUDA torch build is left untouched.
    if not (have("cv2") and have("torch") and have("numpy")):
        pip("-e", str(ROOT))            # pulls requirements.txt (torch, opencv, …)
    else:
        pip("-e", str(ROOT), "--no-deps")   # register the package only

    # 2) Learned page segmenter (approach B needs it) — installed without deps so
    #    it can't disturb the torch build.
    if not have("segmentation_models_pytorch"):
        pip("--no-deps", "segmentation-models-pytorch")
    for extra in ("timm", "safetensors", "huggingface_hub"):
        if not have(extra):
            pip(extra)

    # 3) Web app dependency
    if not have("flask"):
        pip("-r", str(ROOT / "requirements-web.txt"))

    # 4) Model weights (one-time, ~416 MB from the GitHub release)
    seg = ROOT / "weights" / "folio_seg_unet.pt.ts.pt"
    if not seg.exists():
        print("\n>>> downloading model weights (~416 MB, one time)…")
        subprocess.check_call([sys.executable, str(ROOT / "tools" / "fetch_weights.py")])

    # 5) Launch. Import from the repo directly so we don't depend on the editable
    #    install being visible in this exact process.
    sys.path.insert(0, str(ROOT))
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    print("\n>>> starting the Folio Processor web app…\n")
    from folio.webapp import main as web_main
    web_main([])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as e:
        print(f"\n[setup failed] {e}\n"
              "Fix: make sure Python 3.10+ and pip work, then re-run. For a GPU build\n"
              "of torch, install it first (see README) — this launcher won't overwrite it.")
        raise SystemExit(1)
    except KeyboardInterrupt:
        raise SystemExit(0)
