# Deploying on AWS (EC2 / Batch)

The pipeline streams images from S3, processes them, and writes crops + JSON
sidecars back to S3 — the same `folio` command you run locally, just with
`s3://` paths. Memory stays flat regardless of corpus size (bounded queues).

## Quick version (one EC2 instance)
```bash
# on the instance (Python 3.10+; for GPU use the cu128 torch build)
pip install -e . --no-deps && pip install -r requirements.txt
# IAM instance role provides credentials — no keys needed
folio s3://ssda-raw/volumes/ --out s3://ssda-folios/folios/ --region us-east-1 --limit 50   # dry run
folio s3://ssda-raw/volumes/ --out s3://ssda-folios/folios/ --region us-east-1               # full
```
Outputs land under the output prefix (`folios/<stem>[-A|-B].jpg` + `.json`);
review-flagged crops go under the `review/` prefix.

## Measured throughput (RTX 5080 laptop GPU, `tools/benchmark.py`)
**~2.6 s/image** end-to-end with everything on (hybrid segment + 4-way orientation
+ deskew + blank-detect + learned tight crop), on large scans (2.7k–4.5k px).
CRAFT tight cropping adds ~0.5–3 s on the biggest images; `--no-tight-crop` drops
it. (An earlier pathological `_drop_specks` loop made this 15 s — now fixed.)

| workers | ~750k wall-clock |
|---|---|
| 1  | ~22 days |
| 8  | ~2.8 days |
| 16 | ~1.4 days |
| 32 | ~0.7 days |

## Instance choice
The neural heads (orientation, blank, count) **and** the CRAFT tight-crop detector
run on the GPU, so this is no longer CPU-bound. Options:
- **GPU instances** (`g5.xlarge` / `g6`), one `folio … --shard i/N` each — matches
  the ~2.6 s/image above. Simplest path to the table.
- **CPU instances** (`c7i`) are viable for the bulk if you pass `--no-tight-crop`
  (skips CRAFT); the remaining heads are small. Tight cropping on CPU is slow.
- Re-run `tools/benchmark.py <dir>` on your chosen instance to get its real number
  before sizing the fleet.

## Horizontal fan-out with `--shard`
Each worker takes `--shard i/N` and processes only its share of the keys
(stable CRC32 partition — no coordination, no overlap, complete coverage):
```bash
# worker 0 of 8, worker 1 of 8, ...  (one per instance / Batch array index)
folio s3://ssda-raw/volumes/ --out s3://ssda-folios/folios/ --shard 0/8 --region us-east-1
```
On **AWS Batch**, set N = array size and pass `--shard ${AWS_BATCH_JOB_ARRAY_INDEX}/N`.
16 GPU workers ≈ the full 750k in ~1.4 days; scale N to hit your deadline.

## IAM policy (least privilege)
```json
{"Version":"2012-10-17","Statement":[
  {"Effect":"Allow","Action":["s3:GetObject","s3:ListBucket"],
   "Resource":["arn:aws:s3:::ssda-raw","arn:aws:s3:::ssda-raw/*"]},
  {"Effect":"Allow","Action":["s3:PutObject"],
   "Resource":["arn:aws:s3:::ssda-folios/*"]}
]}
```

## Weights
Bake the legacy `.pth` + trained `weights/*.pt` into the AMI/container, or sync
them from S3 on boot to the paths in `folio/config.py` (auto-discovered from
`./legacy_weights` and `weights/`).

## Tight cropping (optional)
Tight cropping (on by default) uses EasyOCR's CRAFT detector. For the EC2 run,
`pip install -r requirements-tight.txt` and pre-download the CRAFT model into the
AMI (`python -c "import easyocr; easyocr.Reader(['en'])"`) so workers don't each
fetch it. Or pass `--no-tight-crop` to skip it (keeps the looser page crop). It
adds ~0.1–0.3 s/folio on GPU; the detector never clips content (it no-ops when it
finds no text).

## Idempotency / resume
The run is shard-stable, so a failed/spot-killed worker can just be relaunched
with the same `--shard i/N`. Add `--resume` to skip inputs whose output already
exists (one listing of the output prefix, then in-memory checks) — so a restart
only does the remaining work:
```bash
folio s3://ssda-raw/v/ --out s3://ssda-folios/f/ --shard 3/8 --resume --region us-east-1
```
