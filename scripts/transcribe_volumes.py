#!/usr/bin/env python3
"""transcribe_volumes.py — submit cropped folios to the Archivault API, one volume
at a time, with NO secrets hard-coded.

This is the reproducible, genericized version of the ad-hoc submission scripts used
during the SSDA cropping/transcription task. It:
  * lists a volume's crop keys straight from the crops bucket (by key prefix),
  * writes them to a temp keys-file (so a big volume never overflows the OS
    command-line length limit — Windows caps argv at ~32 KB, which ~1000+ keys blow
    past; see --keys-file below),
  * submits via the SSDA `submit_job.py` with your chosen steps/model/instructions,
  * renames the returned artifacts to <title>.json / <title>.md.

Credentials are never written here or to disk:
  * AWS comes from your configured profile (boto3 default chain — `aws configure`).
  * The Archivault password comes from $ARCHIVAULT_PASSWORD or an interactive
    getpass prompt.

Examples
--------
  # dash-style keys (folios/<vol>-*.jpg), title = volume id
  python transcribe_volumes.py --bucket my-crops --email me@x.edu \
      --submit ../submit_job.py --volumes 176899 201991 --key-prefix "folios/{vol}-"

  # underscore-style keys (folios/<vol>_*.jpg), strip a title prefix
  python transcribe_volumes.py --bucket my-crops --email me@x.edu \
      --submit ../submit_job.py --volumes GMRV_005090007 \
      --key-prefix "folios/{vol}_" --title-strip GMRV_

  python transcribe_volumes.py ... --dry-run     # list only, submit nothing

NOTE: `submit_job.py` must accept `--keys-file <path>` (one S3 key per line). If your
copy only accepts inline `--keys`, add a small reader, or pass --inline-keys to send
them on the command line (fine for small volumes only).

Rotate any shared password + AWS key after a run.
"""
import os
import sys
import argparse
import getpass
import subprocess
import tempfile

try:
    import boto3
except ImportError:
    sys.exit("boto3 is required: pip install boto3")

DEFAULT_INSTRUCTIONS = (
    "This is an image of a sacramental register. *Transcribe only the main body "
    "of each entry in the register*, ignoring any identifying information in the "
    "lefthand margin.")


def keys_for(s3, bucket, prefix):
    out = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].lower().endswith(".jpg"):
                out.append(obj["Key"])
    return sorted(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bucket", required=True, help="crops S3 bucket (Archivault-authenticated)")
    ap.add_argument("--email", required=True, help="your Archivault account email")
    ap.add_argument("--submit", required=True, help="path to SSDA submit_job.py")
    ap.add_argument("--volumes", nargs="+", required=True, help="volume ids to transcribe")
    ap.add_argument("--key-prefix", default="folios/{vol}-",
                    help="S3 key prefix template; {vol} is substituted (default 'folios/{vol}-')")
    ap.add_argument("--title-strip", default="",
                    help="strip this leading string from each volume id to form the job title")
    ap.add_argument("--steps", nargs="+", default=["foliate", "metadata", "transcribe", "ner"])
    ap.add_argument("--transcription-model", default="gemini-3.1-pro")
    ap.add_argument("--instructions", default=DEFAULT_INSTRUCTIONS,
                    help="project transcription instructions (<=500 chars)")
    ap.add_argument("--instructions-file", default=None,
                    help="read instructions from this file instead of --instructions")
    ap.add_argument("--out-base", default="./transcription_out")
    ap.add_argument("--delete-data", action="store_true", default=True,
                    help="ask Archivault to delete uploaded data after processing (default on)")
    ap.add_argument("--inline-keys", action="store_true",
                    help="pass keys on the command line instead of a keys-file (small volumes only)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    instructions = args.instructions
    if args.instructions_file:
        with open(args.instructions_file, encoding="utf-8") as f:
            instructions = f.read().strip()
    if len(instructions) > 500:
        sys.exit(f"--instructions is {len(instructions)} chars; Archivault caps at 500.")

    s3 = boto3.client("s3")  # default credential chain — no keys in this file
    password = os.environ.get("ARCHIVAULT_PASSWORD") or (
        None if args.dry_run else getpass.getpass("Archivault password: "))

    for vol in args.volumes:
        prefix = args.key_prefix.format(vol=vol)
        keys = keys_for(s3, args.bucket, prefix)
        title = vol[len(args.title_strip):] if args.title_strip and vol.startswith(args.title_strip) else vol
        if not keys:
            print(f"[!] {vol}: no crops under s3://{args.bucket}/{prefix} — skipping.")
            continue
        vol_dir = os.path.join(args.out_base, vol)
        os.makedirs(vol_dir, exist_ok=True)
        print(f"\n--- {vol} (title {title}): {len(keys)} crops -> {vol_dir} ---")
        if args.dry_run:
            print(f"[dry-run] steps={args.steps} model={args.transcription_model}")
            print(f"[dry-run] instructions: {instructions!r}")
            continue

        base = ["python", args.submit, "--source-bucket", args.bucket, "--email", args.email,
                "--password", password, "--title", title, "--steps", *args.steps,
                "--transcription-model", args.transcription_model,
                "--transcription-instructions", instructions, "--out-dir", vol_dir]
        if args.delete_data:
            base.append("--delete-data")

        keys_file = None
        try:
            if args.inline_keys:
                cmd = base + ["--keys", *keys]
            else:
                fd, keys_file = tempfile.mkstemp(prefix=f"keys_{title}_", suffix=".txt")
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write("\n".join(keys))
                cmd = base + ["--keys-file", keys_file]
            r = subprocess.run(cmd, capture_output=True, text=True)
        finally:
            if keys_file and os.path.exists(keys_file):
                os.remove(keys_file)

        if r.returncode != 0:
            print(f"[!] {vol} FAILED — stopping so a systemic error can't burn credits.")
            print(r.stdout[-2000:]); print(r.stderr[-2000:])
            break
        for src, dst in [("result.json", f"{title}.json"), ("report.md", f"{title}.md")]:
            sp = os.path.join(vol_dir, src)
            if os.path.exists(sp):
                os.replace(sp, os.path.join(vol_dir, dst))
        print(f"[*] {vol} done.")

    print("\nDone. Rotate any shared Archivault password + AWS key after the run.")


if __name__ == "__main__":
    main()
