# scripts/ — crop → transcribe workflow helpers

Reproducible, secret-free helpers. No credentials are hard-coded: AWS uses your
configured *profile names* (`aws configure`), and the Archivault password comes
from `$ARCHIVAULT_PASSWORD` or an interactive prompt.

## `crop_volume_s3.py` — the production cropping runner

The master script for the real workload: pull one **volume** from a source S3
bucket, crop every image, and push the crops to a target S3 bucket — **across two
AWS accounts**, streamed **in memory** (image bytes never hit local disk).

```bash
python scripts/crop_volume_s3.py \
    --source-profile ssda-read  --source-bucket legacy-ssda-jpgs-...  --volume 176899 \
    --target-profile ssda-write --target-bucket ssda-archivault-crops-... \
    --write-coords --jobs 16
```

- **Source/target are AWS profile names** (two boto3 sessions) → different accounts, no keys in the script.
- **`--volume`** is used as the source key prefix (override with `--source-prefix`).
- **Approach B** (tight crop) by default; `--white-out` for A.
- **`--write-coords`** also pushes a per-crop provenance JSON (crop → original-image quad).
- **`--jobs N`** fans work across N CPU worker processes (the crop is CPU-bound; the
  big throughput lever); `--jobs 1` uses the GPU in one process.
- **`--dry-run`** lists the volume's keys and does nothing; **`--limit N`** for a smoke test.
- **Scale-out** = one invocation per volume, so shard volumes across machines.

It calls the pipeline (`folio.process.build_pipeline` → `pipe.process_image`), so to
change the model just edit `folio/` or swap the weights — this script doesn't change.
The desktop GUI (`folio-gui`) and web app (`folio-web`) are the QA front-ends over the
same pipeline.

## End-to-end flow

1. **Crop** (approach B is the default — see the top-level README):
   ```bash
   folio s3://raw-bucket/volume/ --out s3://crops-bucket/ --jobs 16
   # or locally, then sync the crops up:
   folio ./raw --out ./out --jobs 16
   aws s3 sync ./out/folios s3://crops-bucket/folios/
   ```

2. **Provenance coordinates** (crop → original-image mapping) for the coordinates bucket:
   ```bash
   python tools/export_for_archivault.py ./out          # writes ./out/coords/<crop>.json
   aws s3 sync ./out/coords s3://coords-bucket/folios/   # (rename to <crop>.jpg.json if your bucket uses that)
   ```

3. **Transcribe**, one volume at a time (you supply the password; nothing is stored):
   ```bash
   python scripts/transcribe_volumes.py \
       --bucket crops-bucket --email you@example.edu --submit ./submit_job.py \
       --volumes 176899 201991 --key-prefix "folios/{vol}-"
   ```
   - `--dry-run` lists what would submit and spends nothing.
   - It writes keys to a **temp keys-file** so large volumes don't overflow the OS
     command-line limit (Windows caps argv at ~32 KB — a ~1,000-key volume blows past
     it, raising `WinError 206`). Pass `--inline-keys` only for small volumes.
   - Underscore-style corpora: `--key-prefix "folios/{vol}_" --title-strip GMRV_`.
   - The default `--instructions` is the sacramental-register, main-body-only prompt;
     override with `--instructions` / `--instructions-file`.

## Requirements / notes

- **`submit_job.py` is SSDA's own script** (from
  [`slavesocieties/ssda-archivault`](https://github.com/slavesocieties/ssda-archivault)),
  not vendored here — pass its path via `--submit`. `transcribe_volumes.py` uses its
  `--keys-file <path>` option (one S3 key per line); if your copy predates that option,
  add a couple of lines to read the file into `args.keys`, or use `--inline-keys`.
- **Security:** rotate any shared Archivault password + AWS key after a run. Never commit
  credentials; both are read at runtime only.
