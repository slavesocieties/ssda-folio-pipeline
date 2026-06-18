"""Build a count dataset (one_folio/ two_folios/) at 512px with a held-out
test split recorded, from the labelled folders we have. No reject class
(we have no reject examples) -> binary in practice."""
import cv2, sys, random
from pathlib import Path

base = Path(sys.argv[1])                  # train_data
out = base / "count_dataset"
cap = 512
random.seed(0)

def resize_to(img, cap):
    h, w = img.shape[:2]; s = cap / max(h, w)
    return cv2.resize(img, (max(int(w*s),1), max(int(h*s),1)), interpolation=cv2.INTER_AREA) if s < 1 else img

# two_folios: all 280
two = sorted((base / "two_folios").glob("*.jpg"))
# one_folio: subsample rightside_up + upside_down (single pages) to ~2x two
ones = sorted((base / "rightside_up").glob("*.jpg")) + sorted((base / "upside_down").glob("*.jpg"))
random.shuffle(ones)
ones = ones[: 2 * len(two)]

def split_write(files, label):
    n_test = max(1, int(0.15 * len(files)))
    test = set(files[:n_test])
    for sub in ("train", "test"):
        (out / sub / label).mkdir(parents=True, exist_ok=True)
    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            continue
        sub = "test" if p in test else "train"
        cv2.imwrite(str(out / sub / label / p.name), resize_to(img, cap), [cv2.IMWRITE_JPEG_QUALITY, 90])

split_write(two, "two_folios")
split_write(ones, "one_folio")
for sub in ("train", "test"):
    n1 = len(list((out / sub / "one_folio").glob("*.jpg")))
    n2 = len(list((out / sub / "two_folios").glob("*.jpg")))
    print(f"{sub}: one_folio={n1}  two_folios={n2}")
