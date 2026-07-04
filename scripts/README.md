# scripts/ — crop → transcribe workflow helpers

Reproducible, secret-free helpers for taking cropped folios through the SSDA
Archivault transcription API. No credentials are hard-coded: AWS uses your
configured profile (`aws configure`), and the Archivault password comes from
`$ARCHIVAULT_PASSWORD` or an interactive prompt.

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
