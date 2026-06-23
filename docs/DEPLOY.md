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

## Instance choice — it's CPU-bound
The neural models are light; the classical CV stages dominate and the GPU sits
~10% utilised. So:
- A single big GPU instance is **underutilised**. Fine for simplicity, wasteful at scale.
- Cheaper/faster: several **CPU instances** (e.g. `c7i`) or **AWS Batch** array
  jobs, each processing a **shard** of the corpus. ~0.5 s/image/core.
- A small GPU instance (`g5.xlarge`) works if you prefer one box.

## Horizontal fan-out with `--shard`
Each worker takes `--shard i/N` and processes only its share of the keys
(stable CRC32 partition — no coordination, no overlap, complete coverage):
```bash
# worker 0 of 8, worker 1 of 8, ...  (one per instance / Batch array index)
folio s3://ssda-raw/volumes/ --out s3://ssda-folios/folios/ --shard 0/8 --region us-east-1
```
On **AWS Batch**, set N = array size and pass `--shard ${AWS_BATCH_JOB_ARRAY_INDEX}/N`.
8 workers ≈ the full 750k in ~half a day; scale N to hit your deadline.

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
