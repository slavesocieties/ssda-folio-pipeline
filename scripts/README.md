# scripts/ — crop → transcribe workflow helpers

Reproducible, secret-free helpers. No credentials are hard-coded: AWS uses your
configured *profile names* (`aws configure`), and the Archivault password comes
from `$ARCHIVAULT_PASSWORD` or an interactive prompt.

## `run_prefix.py` — the one-command entry point (START HERE)

Crop → submit → archive, for one source key prefix (i.e. one volume):

```bash
python scripts/run_prefix.py 176899
python scripts/run_prefix.py 176899 201991 205410      # several volumes, in order
```

Everything that doesn't change between runs lives in **`pipeline.toml`** at the repo
root (buckets, AWS profile names, models, steps, transcription instructions), so the
prefix is the only per-run parameter. That file is **gitignored** — it names your
buckets and AWS account. Start from the tracked template:

```bash
cp pipeline.example.toml pipeline.toml
```

`run_prefix.py` also imports **`submit_job.py`** from the repo root. That is SSDA's own
script (from [`slavesocieties/ssda-archivault`](https://github.com/slavesocieties/ssda-archivault)),
deliberately not vendored here, so drop a copy in place before the first run.

**Profiles.** `[source]` and `[crops]` name AWS profiles. On `[coords]` and
`[transcripts]` an empty `profile` means *"use the crops profile"* — those three
buckets live in one account. It does **not** mean boto3's default credential chain,
which would be the source account.

**Preflight.** Every run first probes each bucket it will touch, with a real
put-then-delete on the destinations — listing a bucket can succeed while `PutObject`
is denied, so a read-only check would not catch a misconfigured profile until after an
hour of cropping. Skip it with `--no-preflight`.

Per prefix it:

1. **Crops** every source image under `s3://<source.bucket>/<prefix>-` and pushes the
   crops to the crops bucket, streamed in memory (bytes never hit local disk).
2. **Submits** the crop keys through the API's **S3 import** flow, so the API copies
   the crops server-side out of the crops bucket — nothing is re-uploaded from here.
   The API's account therefore needs `s3:GetObject` on the crops bucket.
3. **Archives** the returned artifacts straight into the transcripts bucket
   (`<volume>.json`, `<volume>.md`) plus two run records under their own key
   prefixes: `manifest/<volume>.json` (job id, counts, timings) and
   `review/<volume>.json` (see below).

**Blank folios are not transcribed.** The pipeline already decides a folio is blank;
those crops are still written to the crops bucket, but they are held back from the API
job rather than paying to transcribe an empty page. Every held-back crop is listed in
the review report.

**Review report** — `review/<volume>.json`, always written, and written *before*
submission so it survives a failed job. It lists each blank crop held back and each
crop the pipeline flagged for review, with `review_reasons` and the relevant
confidences. The full verdict is also stored as S3 user metadata on each crop object
(`folio-blank`, `folio-review`, `folio-reasons`, `folio-source`, `folio-label`,
`folio-text-frac`, `folio-rotation`, and the two confidences), so a **resumed** run
recovers every reported field for crops it did not reprocess and the report still
covers the whole volume. Crops written before that metadata existed are counted under
`status_unknown` rather than being silently reported as clean.

**Provenance coordinates.** Set `[coords] enabled = true` to also write a per-crop JSON
(the crop's quad on the original image, plus rotation/blank/review flags) to a
**dedicated coords bucket**. Coord keys mirror the crop keys with the image extension
replaced, so the crop `176899-0002A.jpg` gets `176899-0002A.json`. The coords bucket is
in the same account as the crops bucket, so `[coords] profile` can be left empty and
falls back to the crops profile.

**Partial-spread rescue (opt-in).** A volume photographed part-way open produces
one-folio crops that drag in the partial facing leaf, coming out too wide and flagged
`unexpected_aspect`. Setting `quality.rescue_partial_spread = True` re-cuts such a crop
at the detected spine and keeps the complete folio, adjusting the provenance quad to
match. Crops with no confident spine (and covers/blanks) keep their flag and are left
untouched. **Off by default** — calibrated on volume 265705 only (9/10 hand-labelled
correct, 0 wrong, 0 false fires on 60 controls); confirm on a second volume first.
Rescued crops are marked `partial_spread_rescued` in the review report so they stay
auditable. See `folio/stages/partial_spread.py` and `tools/eval_partial_spread.py`.

**Artifact naming.** The job title is the volume id, and the transcript keys are built
from that title plus the artifact *kind* — `<volume>.json` and `<volume>.md`. Whatever
filename the API uses in its own `s3_key` is deliberately ignored, so a missing or
surprising extension upstream can't leak into the target bucket. Any other artifact
kind (notably the tables ZIP) is discarded, and a missing JSON or MD artifact is
reported as a warning rather than passing silently.

**Key naming.** The source bucket is flat (`{volume}-{0000}.jpg`) and the crops bucket
mirrors it exactly: a single folio keeps its key verbatim, and a two-folio spread
splits into `<stem>A.jpg` (left/recto) and `<stem>B.jpg` (right/verso). No `folios/`
prefix. The `-` between volume and image id is appended to the prefix when listing, so
volume `1768` cannot match `17689-0001.jpg`.

**Resume** is on by default: a re-run skips any source image whose crops are already
in the crops bucket, so an interrupted run costs only what it hadn't reached. It still
submits the volume's full crop set. `--no-resume` forces a re-crop.

Other flags: `--dry-run` (list the work, spend nothing), `--limit N` (smoke test),
`--skip-crop` / `--skip-submit` (run one stage), `--continue-on-error` (by default a
failing prefix stops the run so a systemic error can't burn credits across dozens of
volumes — just re-run to resume).

It imports `submit_job.py` as a module rather than shelling out to it, which removes
the OS argv length limit on the key list (a ~1k-image volume overflows Windows' ~32 KB
cap) and keeps the password out of the process table.

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
