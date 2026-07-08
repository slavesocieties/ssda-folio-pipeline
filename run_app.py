#!/usr/bin/env python3
"""One-step launcher for the Folio Processor.

Double-click "Run Folio App.bat" (Windows) or "run_folio_app.command" (macOS), or
run `python run_app.py`. It installs anything missing INTO YOUR CURRENT Python, then
opens the web app in your browser. No coding, no LLM.

GPU aware: if you have an NVIDIA GPU it installs a CUDA build of PyTorch (much faster
for this workload); otherwise it installs the CPU build. An existing torch is reused
untouched, so it never downgrades a working GPU setup.

    python run_app.py                         # set up + open the app
    FOLIO_TORCH_INDEX=cu128 python run_app.py  # force a specific CUDA build
"""
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Default CUDA build. cu124 covers most recent NVIDIA GPUs; 50-series/Blackwell needs
# cu128 (set FOLIO_TORCH_INDEX=cu128). Override with the env var if your driver differs.
CUDA_TAG = os.environ.get("FOLIO_TORCH_INDEX", "cu124")


def have(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except Exception:
        return False


def pip(*args: str) -> None:
    print(f"\n>>> installing: {' '.join(args)}", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", *args])


def has_nvidia_gpu() -> bool:
    if not shutil.which("nvidia-smi"):
        return False
    try:
        subprocess.run(["nvidia-smi"], capture_output=True, timeout=15, check=True)
        return True
    except Exception:
        return False


def install_torch() -> None:
    if have("torch"):
        print(">>> PyTorch already installed — reusing it.")
        return
    if has_nvidia_gpu():
        idx = f"https://download.pytorch.org/whl/{CUDA_TAG}"
        print(f">>> NVIDIA GPU detected — installing CUDA PyTorch ({CUDA_TAG}).")
        try:
            pip("--index-url", idx, "torch", "torchvision")
            return
        except subprocess.CalledProcessError:
            print(f"[!] CUDA torch ({CUDA_TAG}) install failed — falling back to CPU. "
                  f"(For a 50-series/Blackwell GPU try FOLIO_TORCH_INDEX=cu128.)")
    else:
        print(">>> No NVIDIA GPU found — installing CPU PyTorch.")
    pip("--index-url", "https://download.pytorch.org/whl/cpu", "torch", "torchvision")


def main() -> int:
    print("Folio Processor — setup & launch")
    print(f"Python: {sys.executable}\n")

    # 1) PyTorch (GPU build when available)
    install_torch()

    # 2) Lean runtime deps for the local app (skip the cloud/optional heavies:
    #    ultralytics, ray, aioboto3/boto3 are lazy-imported and not needed here).
    for mod, spec in [("cv2", "opencv-python-headless"), ("numpy", "numpy"),
                      ("timm", "timm"), ("safetensors", "safetensors"),
                      ("huggingface_hub", "huggingface_hub"), ("flask", "flask")]:
        if not have(mod):
            pip(spec)
    if not have("segmentation_models_pytorch"):
        pip("--no-deps", "segmentation-models-pytorch")

    # 3) Optional: EasyOCR enables the orientation review-rescue. Best-effort — the
    #    pipeline no-ops gracefully without it, so a failure here is not fatal.
    if not have("easyocr"):
        try:
            pip("--no-deps", "easyocr", "python-bidi", "pyclipper", "shapely",
                "scikit-image", "PyYAML", "ninja")
        except subprocess.CalledProcessError:
            print("[i] EasyOCR not installed (optional) — orientation rescue disabled.")

    # 4) Register the package (source-importable) without disturbing torch
    pip("-e", str(ROOT), "--no-deps")

    # 5) Model weights (one-time, ~416 MB from the GitHub release)
    if not (ROOT / "weights" / "folio_seg_unet.pt.ts.pt").exists():
        print("\n>>> downloading model weights (~416 MB, one time)…")
        subprocess.check_call([sys.executable, str(ROOT / "tools" / "fetch_weights.py")])

    # 6) Launch
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
        print(f"\n[setup failed] {e}\nMake sure Python 3.10+ and pip work, then re-run.")
        raise SystemExit(1)
    except KeyboardInterrupt:
        raise SystemExit(0)
