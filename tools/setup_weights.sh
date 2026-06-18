#!/usr/bin/env bash
# Download the pretrained foundation-model weights for the neural path.
# Run on the GPU box (AWS). RT-DETR weights auto-download via ultralytics on
# first use, so only SAM 2.1 needs fetching here. Your two trained heads
# (orientation4_convnextv2.pt, folio_count_convnextv2.pt) come from
# folio.training.train and should be copied into the same WEIGHTS dir.
set -euo pipefail
WEIGHTS="${1:-weights}"
mkdir -p "$WEIGHTS"

# SAM 2.1 Hiera-Large checkpoint (Meta)
SAM_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
echo "Downloading SAM 2.1 Hiera-Large -> $WEIGHTS/"
curl -L -o "$WEIGHTS/sam2.1_hiera_large.pt" "$SAM_URL"

# RT-DETR base (ultralytics) - pull a generic checkpoint to fine-tune or use.
# For page detection you will typically fine-tune on a small box-labelled set;
# as a starting point, ultralytics fetches rtdetr-l.pt automatically:
python - <<'PY'
try:
    from ultralytics import RTDETR
    RTDETR("rtdetr-l.pt")  # triggers download into the ultralytics cache
    print("RT-DETR base ready (rtdetr-l.pt). Fine-tune for page boxes, then save as rtdetr_page.pt")
except Exception as e:
    print("ultralytics not installed yet:", e)
PY

echo "Done. Expected files in $WEIGHTS/:"
echo "  sam2.1_hiera_large.pt           (downloaded)"
echo "  rtdetr_page.pt                  (your fine-tuned page detector)"
echo "  orientation4_convnextv2.pt      (from folio.training.train --task orientation)"
echo "  folio_count_convnextv2.pt       (from folio.training.train --task count)"
