"""Indexer + progress: both geometries, provenance, incremental gate, hostile files."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import xxhash
from PIL import Image

from imgtag.core import models, store
from imgtag.core.indexer import index, scan
from imgtag.core.progress import read_job

HAS_PECORE = models.find_artifact(models.registry()["pecore-s16-384"], "pecore-s16-384-vision.onnx") is not None
needs_model = pytest.mark.skipif(not HAS_PECORE, reason="pecore-s16-384 artifacts not on this host")
PROFILE = {"precision": "fp32", "intra_op": 1, "batch": 2, "cores": 4, "mem_available_mb": 4096,
           "geometry": "central", "worker_intra_op": 1}


@pytest.fixture
def imgs(tmp_path):
    d = tmp_path / "imgs"
    d.mkdir()
    for i in range(6):
        Image.new("RGB", (120 + i, 90), (i * 40 % 255, 90, 200)).save(d / f"{i}.jpg", quality=90)
    return d


@needs_model
@pytest.mark.parametrize("geometry", ["central", "worker"])
def test_index_end_to_end(tmp_path, imgs, geometry):
    home = tmp_path / "home"
    s = index(imgs, "ds", profile={**PROFILE, "geometry": geometry}, home=home, workers=2)
    assert s["indexed"] == 6 and s["failed"] == 0

    snap = store.open_snapshot("ds", home)
    assert snap.count == 6 == len(snap.ids) == snap.emb.shape[0]
    assert snap.emb.dtype == np.float32
    norms = np.linalg.norm(np.asarray(snap.emb), axis=1)
    assert 0.999 <= norms.mean() <= 1.001  # L2 invariant (ORACLE §6)
    for r in snap.ids:  # B18 provenance: id is xxhash64 of the file bytes, path exists
        assert r["dataset"] == "ds" and Path(r["path"]).is_file()
        assert r["image_id"] == xxhash.xxh64(Path(r["path"]).read_bytes()).hexdigest()
        assert r["w"] > 0 and r["h"] == 90
    assert sorted(r["row"] for r in snap.ids) == list(range(6))

    job = read_job(s["job_id"], home)
    assert job["state"] == "done" and job["done"] == 6 and job["inflight"] == 0
    assert job["done"] == store.read_manifest("ds", home)["count"]  # progress == durable count


@needs_model
def test_reindex_unchanged_reembeds_nothing(tmp_path, imgs):
    home = tmp_path / "home"
    index(imgs, "ds", profile=PROFILE, home=home, workers=2)
    again = index(imgs, "ds", profile=PROFILE, home=home, workers=2)
    assert again["indexed"] == 0 and again["skipped"] == 6  # B12 compute-leak gate
    assert store.open_snapshot("ds", home).count == 6
    # a touched file IS re-embedded
    Image.new("RGB", (60, 90), (7, 7, 7)).save(imgs / "0.jpg")
    third = index(imgs, "ds", profile=PROFILE, home=home, workers=2)
    assert third["indexed"] == 1 and third["skipped"] == 5


@needs_model
def test_hostile_files_are_counted_not_fatal(tmp_path, imgs):
    home = tmp_path / "home"
    (imgs / "empty.jpg").write_bytes(b"")
    (imgs / "truncated.jpg").write_bytes((imgs / "0.jpg").read_bytes()[:200])
    Image.new("RGB", (40, 40), (1, 2, 3)).save(imgs / "actually_png.jpg", "PNG")  # lying extension
    s = index(imgs, "ds", profile=PROFILE, home=home, workers=2)
    assert s["indexed"] >= 6  # every valid file made it (the PNG-in-a-.jpg decodes fine)
    assert s["failed"] == 2 and s["indexed"] + s["failed"] == len(scan(imgs))
    job = read_job(s["job_id"], home)
    assert {Path(f["path"]).name for f in job["failures"]} == {"empty.jpg", "truncated.jpg"}
    assert all(f["reason"] for f in job["failures"])  # every failure is named, none silent


@needs_model
def test_search_while_indexing_sees_a_consistent_prefix(tmp_path):
    """B11-shaped smoke test: a snapshot taken mid-job is never torn."""
    home = tmp_path / "home"
    d = tmp_path / "many"
    d.mkdir()
    for i in range(40):
        Image.new("RGB", (64, 64), (i * 6 % 255, 10, 10)).save(d / f"{i}.jpg")
    seen = []

    def watcher(_):
        pass

    s = index(d, "ds", profile=PROFILE, home=home, workers=2, on_progress=watcher)
    snap = store.open_snapshot("ds", home)
    assert snap.count == 40 == len(snap.ids)
    assert s["stages_ms_per_img"]["decode"] > 0  # per-stage telemetry is real
    assert "infer" in s["stages_ms_per_img"] and "queue_wait" in s["stages_ms_per_img"]


def test_scan_finds_images_recursively(tmp_path):
    (tmp_path / "a" / "b").mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(tmp_path / "a" / "b" / "x.PNG")
    (tmp_path / "a" / "notes.txt").write_text("hi")
    assert [p.name for p in scan(tmp_path)] == ["x.PNG"]
    assert scan(tmp_path, recursive=False) == []
