# Packaging — standalone Windows executable

Build a self-contained **Folio Processor.exe** that end users run with **no Python
install, no pip, no setup** — double-click, the browser opens, drop scans, get crops.

## Build

```bash
python packaging/build_exe.py
```

This creates an isolated build venv with a **CPU** torch (so the bundle runs on any
machine and never touches a local GPU build), installs the lean runtime set, and runs
PyInstaller. Output: `packaging/dist/FolioProcessor/` (a onedir bundle, ~1 GB).

The build excludes the cloud/optional pieces not needed for local cropping
(ultralytics, ray, aioboto3/boto3, easyocr). The OCR orientation-rescue is therefore
off in the packaged app; the crop itself is identical (approach B by default).

## Distribute

1. Zip `packaging/dist/FolioProcessor/` → `FolioProcessor-win64.zip`.
2. Attach it to a **GitHub Release** (a zipped onedir is well under the 2 GB asset limit).
3. Users: download, unzip, double-click **`FolioProcessor.exe`**.

On first run the app downloads the model weights (~416 MB) into a `weights/` folder
**next to the exe** (from the `weights-v1` release), then opens
`http://127.0.0.1:8000`. Subsequent runs are offline and instant.

## Notes

- `app_entry.py` is the frozen entry point: it sets `FOLIO_WEIGHTS_DIR` to the
  adjacent `weights/` folder, fetches weights if missing, then launches the web app.
- CPU-only, so large batches are slower than a GPU source install. For heavy/scale
  work, use the source install (`pip install -e .` + a CUDA torch) and the `folio`
  CLI / `folio-web`.
- macOS/Linux: the same `build_exe.py` produces a onedir bundle for the host OS, but
  it is untested there — the source `run_app.py` launcher is the supported path.
