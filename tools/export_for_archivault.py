"""Turn a folio-pipeline output dir into Archivault upload artifacts.

    python tools/export_for_archivault.py <out_dir>

Reads <out>/folios/*.jpg + <out>/sidecars/*.json and writes:
  <out>/coords/<crop>.json  — one crop-coordinate artifact per crop, mapping it
                              back to the ORIGINAL source image (source_size +
                              crop_quad_norm corners), for the coordinates bucket.
Prints the list of crop filenames (the --keys for submit_job.py) per volume.
Crops themselves stay in <out>/folios/ for upload to the crops bucket.
"""
import os, csv, glob, json, sys
from collections import defaultdict

out = sys.argv[1]
coords = os.path.join(out, "coords")
os.makedirs(coords, exist_ok=True)

# map crop filename -> its FolioResult meta (from the per-source sidecars)
side = {}
for s in glob.glob(os.path.join(out, "sidecars", "*.json")):
    d = json.load(open(s))
    src = d.get("source_key", os.path.basename(s).replace(".json", ".jpg"))
    stem = os.path.splitext(os.path.basename(src))[0]
    for f in d.get("folios", []):
        crop = f"{stem}{('-' + f['label']) if f.get('label') else ''}.jpg"
        side[crop] = (src, d.get("page_count"), f)

by_vol = defaultdict(list)
n = 0
for p in sorted(glob.glob(os.path.join(out, "folios", "*.jpg"))):
    crop = os.path.basename(p)
    if "_enhanced" in crop:
        continue
    vol = crop.split("-")[0]
    by_vol[vol].append(crop)
    info = side.get(crop)
    if info:
        src, pc, f = info
        rec = {
            "crop": crop,
            "source_image": src,
            "source_size": f.get("source_size"),
            "crop_quad_norm": f.get("crop_quad_norm"),
            "corner_order": "TL,TR,BR,BL (x,y ratios of source image)",
            "label": f.get("label"),
            "page_count": pc,
            "rotation_deg": round(f.get("rotation_deg", 0.0), 3),
            "is_blank": f.get("is_blank"),
            "needs_review": f.get("needs_review"),
        }
        json.dump(rec, open(os.path.join(coords, crop.replace(".jpg", ".json")), "w"), indent=2)
        n += 1

print(f"wrote {n} coordinate artifacts -> {coords}")
for vol, crops in sorted(by_vol.items()):
    print(f"volume {vol}: {len(crops)} crops")
    # keys file (one crop name per line) for convenience
    open(os.path.join(out, f"_keys_{vol}.txt"), "w").write("\n".join(crops))
