"""Unit tests for the folio.process driver helpers (no torch/GPU/weights)."""
import os
import numpy as np
import cv2
import pytest

from folio import process as P
from folio.pipeline import FolioPipeline


def test_parse_s3_basic():
    assert P.parse_s3("s3://bucket/prefix/x") == ("bucket", "prefix/x")
    assert P.parse_s3("s3://bucket") == ("bucket", "")
    assert P.parse_s3("s3://b/") == ("b", "")


def test_parse_s3_rejects_non_s3():
    with pytest.raises(ValueError):
        P.parse_s3("/local/path.jpg")


def test_find_legacy_weights_explicit(tmp_path):
    wd = tmp_path / "lw"
    wd.mkdir()
    (wd / "unet_folio_split.pth").write_bytes(b"x")
    assert P.find_legacy_weights(str(wd), None) == str(wd)


def test_find_legacy_weights_env(tmp_path, monkeypatch):
    wd = tmp_path / "envlw"
    wd.mkdir()
    (wd / "unet_folio_split.pth").write_bytes(b"x")
    monkeypatch.setenv("FOLIO_LEGACY_WEIGHTS", str(wd))
    assert P.find_legacy_weights(None, None) == str(wd)


def test_find_legacy_weights_requires_marker(tmp_path):
    # an explicit dir WITHOUT the marker .pth is never returned as-is
    empty = tmp_path / "nope"
    empty.mkdir()
    assert P.find_legacy_weights(str(empty), None) != str(empty)


def test_list_images(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "b.PNG").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("x")
    got = [p.name for p in P.list_images(tmp_path)]
    assert got == ["a.jpg", "b.PNG"]
    single = tmp_path / "a.jpg"
    assert P.list_images(single) == [single]


def test_parse_shard():
    from folio.cli import _parse_shard
    assert _parse_shard(None) is None
    assert _parse_shard("0/8") == (0, 8)
    assert _parse_shard("7/8") == (7, 8)
    import pytest as _pt
    with _pt.raises(SystemExit):
        _parse_shard("8/8")          # i must be < N


def test_shard_partition_is_complete_and_disjoint():
    """Every key lands in exactly one shard across N workers (the run_s3 rule)."""
    import zlib
    keys = [f"vol/{i}-{j}.jpg" for i in range(300) for j in range(3)]
    N = 6
    seen = {}
    for shard in range(N):
        for k in keys:
            if (zlib.crc32(k.encode()) % N) == shard:
                seen[k] = seen.get(k, 0) + 1
    assert len(seen) == len(keys)            # complete coverage
    assert set(seen.values()) == {1}         # disjoint (each key once)


def test_build_pipeline_classical_fallback():
    """No legacy weights -> classical pipeline, no torch required."""
    cfg = P.make_config(device="cpu")
    pipe, mode = P.build_pipeline(cfg, legacy_weights=None, prepass=True)
    assert isinstance(pipe, FolioPipeline)
    assert "classical" in mode
    assert pipe.coarse_orienter is None  # no pre-pass without a neural head


def test_run_local_classical_end_to_end(tmp_path, monkeypatch):
    """A whole local run on a synthetic page produces a crop, sidecar, manifest.
    Force classical (no weights) so the test is hermetic on any machine."""
    monkeypatch.setattr(P, "find_legacy_weights", lambda *a, **k: None)
    img = np.full((400, 300, 3), 235, np.uint8)
    img[60:340, 40:260] = 200            # a page region
    for y in range(80, 320, 24):         # some 'text' rows
        img[y:y + 3, 60:240] = 40
    src = tmp_path / "page.jpg"
    cv2.imwrite(str(src), img)
    out = tmp_path / "out"
    stats, mode = P.run_local(src, out, device="cpu", legacy=None)
    assert stats.images == 1
    assert stats.folios >= 1
    assert (out / "manifest.csv").exists()
    assert list((out / "sidecars").glob("*.json"))
    assert list((out / "folios").glob("*.jpg"))
