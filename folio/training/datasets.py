"""Datasets for the two heads. Torch is imported lazily at module load (these
modules are only used on the training box).

OrientationDataset is SELF-SUPERVISED: point it at any folder of correctly
oriented page crops (e.g. a vetted subset of pipeline output) and it manufactures
balanced 0/90/180/270 examples on the fly. No manual labels.

FolioCountDataset reads either class subfolders (one_folio/, two_folios/,
reject/) or a CSV manifest with columns: path,label.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import csv
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset

from .labels import COUNT_CLASSES
from ..models.classifiers import geometric_priors

IMNET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMNET_STD = np.array([0.229, 0.224, 0.225], np.float32)
_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def _to_tensor(bgr: np.ndarray, size: int) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    arr = (rgb.astype(np.float32) / 255.0 - IMNET_MEAN) / IMNET_STD
    return torch.from_numpy(arr.transpose(2, 0, 1))


def _photometric(bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # mild brightness/contrast jitter; never geometric (would corrupt labels)
    a = 1.0 + rng.uniform(-0.15, 0.15)
    b = rng.uniform(-15, 15)
    return np.clip(bgr.astype(np.float32) * a + b, 0, 255).astype(np.uint8)


class OrientationDataset(Dataset):
    def __init__(self, root: str, size: int = 384, train: bool = True,
                 small_skew_deg: float = 4.0, seed: int = 0):
        self.paths = [p for p in Path(root).rglob("*") if p.suffix.lower() in _EXT]
        self.size = size
        self.train = train
        self.small_skew = small_skew_deg
        self.rng = np.random.default_rng(seed)
        if not self.paths:
            raise FileNotFoundError(f"no images under {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i: int):
        img = cv2.imread(str(self.paths[i]))
        if img is None:
            img = np.full((self.size, self.size, 3), 255, np.uint8)
        k = int(self.rng.integers(0, 4))            # the label
        rot = np.ascontiguousarray(np.rot90(img, k))
        if self.train:
            rot = _photometric(rot, self.rng)
            if self.small_skew > 0:                 # tiny in-plane jitter only
                ang = float(self.rng.uniform(-self.small_skew, self.small_skew))
                h, w = rot.shape[:2]
                M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
                rot = cv2.warpAffine(rot, M, (w, h), borderValue=(255, 255, 255))
        return _to_tensor(rot, self.size), k


class FolioCountDataset(Dataset):
    def __init__(self, root: str = None, manifest: str = None, size: int = 384,
                 train: bool = True, seed: int = 0):
        self.size = size
        self.train = train
        self.rng = np.random.default_rng(seed)
        self.items: List[Tuple[str, int]] = []
        if manifest:
            with open(manifest) as f:
                for row in csv.DictReader(f):
                    self.items.append((row["path"], COUNT_CLASSES.index(row["label"])))
        elif root:
            for ci, cls in enumerate(COUNT_CLASSES):
                d = Path(root) / cls
                if d.is_dir():
                    for p in d.rglob("*"):
                        if p.suffix.lower() in _EXT:
                            self.items.append((str(p), ci))
        if not self.items:
            raise FileNotFoundError("no labelled images found (need folders or manifest)")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i: int):
        path, label = self.items[i]
        img = cv2.imread(path)
        if img is None:
            img = np.full((self.size, self.size, 3), 255, np.uint8)
        ar, valley = geometric_priors(img)
        if self.train:
            img = _photometric(img, self.rng)
        aux = torch.tensor([ar, valley], dtype=torch.float32)
        return _to_tensor(img, self.size), aux, label
