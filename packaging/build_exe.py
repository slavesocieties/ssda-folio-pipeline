"""Build a standalone Folio Processor executable with PyInstaller.

Creates an isolated build venv with a CPU torch (so the .exe runs on any machine,
no CUDA needed and no risk to a local GPU build), installs the lean runtime set
(no ultralytics/ray/easyocr/boto — not needed for local cropping), and produces a
onedir bundle at packaging/dist/FolioProcessor/.

    python packaging/build_exe.py

Zip that folder for distribution. First run of the .exe downloads the weights
(~416 MB) next to it. ~1 GB build; takes several minutes.
"""
import os
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "packaging"
VENV = PKG / "build_venv"


def vpy() -> Path:
    sub = "Scripts" if os.name == "nt" else "bin"
    return VENV / sub / ("python.exe" if os.name == "nt" else "python")


def run(*args) -> None:
    print(">>>", " ".join(str(a) for a in args), flush=True)
    subprocess.check_call([str(a) for a in args])


def main() -> int:
    if not VENV.exists():
        print("Creating build venv…", flush=True)
        venv.create(VENV, with_pip=True)
    py = vpy()
    run(py, "-m", "pip", "install", "-q", "--upgrade", "pip", "wheel")
    # CPU torch for portability
    run(py, "-m", "pip", "install", "-q",
        "--index-url", "https://download.pytorch.org/whl/cpu", "torch", "torchvision")
    # lean runtime + build tool
    run(py, "-m", "pip", "install", "-q", "opencv-python-headless", "numpy",
        "timm", "safetensors", "huggingface_hub", "flask", "pyinstaller")
    run(py, "-m", "pip", "install", "-q", "--no-deps", "segmentation-models-pytorch")
    run(py, "-m", "pip", "install", "-q", "--no-deps", "-e", str(ROOT))  # register folio

    run(py, "-m", "PyInstaller", "--noconfirm", "--clean", "--name", "FolioProcessor",
        "--collect-all", "torch",
        "--collect-all", "torchvision",
        "--collect-all", "segmentation_models_pytorch",
        "--collect-all", "timm",
        "--collect-all", "cv2",
        "--collect-submodules", "folio",
        "--paths", str(ROOT),
        "--distpath", str(PKG / "dist"),
        "--workpath", str(PKG / "build"),
        "--specpath", str(PKG),
        str(PKG / "app_entry.py"))

    out = PKG / "dist" / "FolioProcessor"
    exe = out / ("FolioProcessor.exe" if os.name == "nt" else "FolioProcessor")
    print(f"\nBUILD DONE -> {exe}" if exe.exists() else f"\nBUILD FINISHED but {exe} not found — check log")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
