"""Generic metadata + the moderation hook contract (user directive, VISION-ADDENDA 12:33Z).

The hook is the seam the three track-* lanes plug into, so its CONTRACT is tested here
with a stand-in detector: post-embedding, batch-wise, one flag list per record, and a
detector that raises must never break the user's index.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from imgtag.core import models, store
from imgtag.core.indexer import index, load_meta_csv, moderation_summary, parse_meta

HAS_PECORE = models.find_artifact(models.registry()["pecore-s16-384"], "pecore-s16-384-vision.onnx") is not None
needs_model = pytest.mark.skipif(not HAS_PECORE, reason="pecore-s16-384 artifacts not on this host")
PROFILE = {"precision": "fp32", "intra_op": 1, "batch": 2, "cores": 4, "mem_available_mb": 4096,
           "geometry": "central", "worker_intra_op": 1}


def flag_every_other(embs, recs, images=None):
    """Stand-in detector. Asserts the contract it is handed while it is at it."""
    assert embs.dtype == np.float32 and embs.shape[0] == len(recs)
    assert 0.99 <= float(np.linalg.norm(embs, axis=1).mean()) <= 1.01, "hook must get L2-normalized rows"
    out = []
    for r in recs:
        i = int(Path(r["path"]).stem.replace("img", "") or 0)
        out.append([{"category": "weapons", "p": 0.9, "tier": "violation"}] if i % 2 == 0 else
                   [{"category": "weapons", "p": 0.4, "tier": "review"}] if i == 1 else [])
    return out


def exploding_detector(embs, recs, images=None):
    raise RuntimeError("detector is broken")


@pytest.fixture
def imgs(tmp_path):
    d = tmp_path / "imgs"
    d.mkdir()
    for i in range(4):
        Image.new("RGB", (64, 64), (i * 50 % 255, 70, 130)).save(d / f"img{i}.jpg")
    (d / "meta.csv").write_text(
        "filename,account_id,captured\n" + "".join(f"img{i}.jpg,acct-{i},2026-07-0{i + 1}\n" for i in range(4))
    )
    return d


# ---------------------------------------------------------------- metadata


def test_parse_meta_and_csv(imgs):
    assert parse_meta(["a=1", "b = two"]) == {"a": "1", "b": "two"}
    with pytest.raises(ValueError):
        parse_meta(["nope"])
    rows = load_meta_csv(imgs / "meta.csv", imgs)
    assert rows["img2.jpg"] == {"account_id": "acct-2", "captured": "2026-07-03"}


def test_meta_csv_requires_a_key_column(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("colour,size\nred,big\n")
    with pytest.raises(ValueError, match="path"):
        load_meta_csv(p)


@needs_model
def test_job_meta_and_csv_meta_land_on_every_record(tmp_path, imgs):
    home = tmp_path / "home"
    s = index(imgs, "ds", profile=PROFILE, home=home, workers=2,
              meta={"source": "partner-a"}, meta_csv=imgs / "meta.csv")
    assert s["indexed"] == 4
    snap = store.open_snapshot("ds", home)
    for r in snap.ids:
        i = int(Path(r["path"]).stem.replace("img", ""))
        assert r["meta"]["source"] == "partner-a"           # job-level, every image
        assert r["meta"]["account_id"] == f"acct-{i}"       # per-image, from the CSV
        assert r["meta"]["captured"] == f"2026-07-0{i + 1}"
    # dataset-level metadata is on the manifest, not repeated per row
    assert snap.manifest["meta"] == {"source": "partner-a"}


@needs_model
def test_records_without_meta_stay_lean(tmp_path, imgs):
    home = tmp_path / "home"
    index(imgs, "ds", profile=PROFILE, home=home, workers=2)
    assert all("meta" not in r for r in store.open_snapshot("ds", home).ids)


# ---------------------------------------------------------------- moderation


@needs_model
def test_moderation_hook_flags_counts_and_summary(tmp_path, imgs):
    home = tmp_path / "home"
    s = index(imgs, "ds", profile=PROFILE, home=home, workers=2,
              moderation=f"{__name__}:flag_every_other")
    assert s["moderation_active"] is True and s["moderation_errors"] == 0
    assert s["moderation"]["violation"] == {"nudity": 0, "weapons": 2, "drugs": 0}
    assert s["moderation"]["review"]["weapons"] == 1   # ADR-14: tiers counted separately

    snap = store.open_snapshot("ds", home)
    flagged = [r for r in snap.ids if r.get("flags")]
    assert len(flagged) == 3   # 2 violation + 1 review
    assert flagged[0]["flags"][0] == {"category": "weapons", "p": 0.9, "tier": "violation"}
    assert snap.manifest["moderation"]["counts"]["violation"]["weapons"] == 2
    assert snap.manifest["moderation"]["enforcement_ready"] is False  # unfitted tau, stated

    job = json.loads((home / "jobs" / f"{s['job_id']}.json").read_bytes())
    assert job["moderation"]["violation"]["weapons"] == 2


@needs_model
def test_a_broken_detector_never_breaks_the_index(tmp_path, imgs):
    home = tmp_path / "home"
    s = index(imgs, "ds", profile=PROFILE, home=home, workers=2,
              moderation=f"{__name__}:exploding_detector")
    assert s["indexed"] == 4 and s["failed"] == 0     # the user's index survives
    assert s["moderation_errors"] > 0                  # and the failure is COUNTED, not hidden
    assert store.open_snapshot("ds", home).count == 4


@needs_model
def test_no_hook_means_no_moderation_block(tmp_path, imgs):
    home = tmp_path / "home"
    s = index(imgs, "ds", profile=PROFILE, home=home, workers=2, moderation="nosuchmodule:detect")
    assert s["moderation_active"] is False
    assert "moderation" not in store.read_manifest("ds", home)


def test_summary_uses_the_users_phrasing():
    assert moderation_summary({"violation": {"drugs": 3, "weapons": 1, "nudity": 0}}) == \
        "Found 0 images with nudity, 1 images with weapons, 3 images with drugs"
    # ADR-14 tiers are visible per category, in the lead's finalized phrasing
    assert moderation_summary({"violation": {"drugs": 1}, "review": {"weapons": 2}}) == \
        "Found 0 images with nudity, 0 images with weapons (2 for review), 1 images with drugs"
    assert moderation_summary({}, active=False) == "moderation: off (no tracks loaded)"


# ---------------------------------------------------------------- CLI surface


@needs_model
def test_cli_meta_flags_rollup_and_dataset_meta(tmp_path, imgs):
    home = tmp_path / "home"
    # a real detector module on disk, so the subprocess resolves it exactly as the
    # track-* lanes' package will be resolved in production
    (tmp_path / "fakemod.py").write_text(
        "def detect(embs, recs, images=None):\n"
        "    return [[{'category': 'weapons', 'p': 0.9, 'tier': 'violation'}]\n"
        "            if int(r['path'].split('img')[-1][0]) % 2 == 0 else [] for r in recs]\n")
    env = {"IMGTAG_HOME": str(home), "PYTHONPATH": str(tmp_path)}
    cli = [sys.executable, "-m", "imgtag.cli"]

    def run(*a):
        p = subprocess.run(cli + list(a), capture_output=True, text=True, timeout=600,
                           env={**__import__("os").environ, **env})
        assert p.returncode == 0, p.stderr[-400:]
        return json.loads(p.stdout) if "--json" in a else p.stdout

    run("index", str(imgs), "--dataset", "shop", "--wait", "--json", "--meta", "source=partner-a",
        "--meta-csv", str(imgs / "meta.csv"), "--moderation-hook", "fakemod:detect")

    hit = run("search", "a photo", "-k", "1", "--json", "--no-daemon")["hits"][0]
    # The DIRECTIVE: per-image metadata is reachable on every hit. Searcher currently
    # sweeps unknown id-record fields into `meta`, so ours arrives nested (meta.meta);
    # accept either while b-daemon hoists it — the requirement is the same either way.
    m = hit["meta"].get("meta", hit["meta"])
    assert m["source"] == "partner-a" and m["account_id"].startswith("acct-")

    roll = run("info", "--flags", "--json")
    assert roll["counts"]["violation"]["weapons"] == 2
    assert roll["datasets"][0]["flagged"][0]["categories"][0]["category"] == "weapons"

    assert run("manage", "meta", "shop", "--set", "owner=ops", "--json")["meta"]["owner"] == "ops"
    assert run("manage", "meta", "shop", "--json")["meta"] == {"source": "partner-a", "owner": "ops"}
