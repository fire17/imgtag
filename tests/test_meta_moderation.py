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


# the tier bands the read path uses to DERIVE tiers from the stored weapons scores
flag_every_other.specs = {"weapons": {"tau_violation": 0.5, "tau_review": 0.2}}


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
    # category-agnostic (the 100-track law applied to the tests): assert the ONE track we
    # drove, never a fixed-key dict — other lanes' tracks may be present or absent
    assert s["moderation"]["violation"]["weapons"] == 2
    assert s["moderation"]["review"]["weapons"] == 1   # ADR-14: tiers counted separately

    # T1: scores are stored per image in a dense sidecar; tiers derive from them
    snap = store.open_snapshot("ds", home)
    assert snap.tracks["weapons"] is not None and len(snap.tracks["weapons"]) == snap.count
    flags = store.dataset_flags("ds", home)
    tiers = flags["weapons"]["tiers"]
    assert tiers.count("violation") == 2 and tiers.count("review") == 1
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


class _LegacyHead:
    """A head from before the seam ruling: tensor-only score(), flagged instead of tier."""

    def score(self, batch):
        raise TypeError("score() takes 2 positional arguments but 4 were given")

    def score_images(self, images):
        return [[{"category": "weapons", "p": 0.8, "flagged": True}] for _ in images]


@needs_model
def test_legacy_head_is_tolerated_and_deprecation_fires_once(tmp_path, imgs, monkeypatch):
    """Ruling: keep score_images and flagged->violation as TOLERATED adapters, each with
    a deprecation line — said once per job, not once per batch (a log that repeats per
    batch is a log nobody reads)."""
    import sys
    import types

    fake = types.ModuleType("imgtag.moderation")
    fake.load_heads = lambda profile: {"weapons": _LegacyHead()}
    monkeypatch.setitem(sys.modules, "imgtag.moderation", fake)

    lines = []
    s = index(imgs, "ds", profile=PROFILE, home=tmp_path / "home", workers=2,
              moderation=True, log=lines.append)

    assert s["moderation"]["violation"]["weapons"] == 4      # legacy path still counts
    seam = [ln for ln in lines if "DEPRECATED" in ln and "score_images" in ln]
    tier = [ln for ln in lines if "DEPRECATED" in ln and "tier" in ln]
    assert len(seam) == 1, f"seam deprecation should fire once per job, got {len(seam)}"
    assert len(tier) == 1, f"tier deprecation should fire once per job, got {len(tier)}"


def test_summary_uses_the_users_phrasing():
    # category-agnostic: assert the user's per-category phrasing is PRESENT, not that the
    # sentence has exactly N clauses — new tracks add clauses and must not break this
    line = moderation_summary({"violation": {"drugs": 3, "weapons": 1}})
    assert line.startswith("Found ")
    assert "3 images with drugs" in line and "1 images with weapons" in line
    # review tier is shown per category in the finalized "(M for review)" form
    assert "(2 for review)" in moderation_summary({"violation": {"drugs": 1}, "review": {"weapons": 2}})
    # the alert tier, when present, leads and cannot be scrolled past
    alert = moderation_summary({"alert": {"safety": 2}, "violation": {"drugs": 1}})
    assert alert.startswith("\u26a0 2 ALERTS") and "3 images" not in alert
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
        "            if int(r['path'].split('img')[-1][0]) % 2 == 0 else [] for r in recs]\n"
        "detect.specs = {'weapons': {'tau_violation': 0.5, 'tau_review': 0.2}}\n")
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
    assert roll["counts"]["violation"]["weapons"] == 2   # derived from the score sidecar
    flagged = roll["datasets"][0]["flagged"]
    assert any(c["category"] == "weapons" for c in flagged[0]["categories"])

    assert run("manage", "meta", "shop", "--set", "owner=ops", "--json")["meta"]["owner"] == "ops"
    assert run("manage", "meta", "shop", "--json")["meta"] == {"source": "partner-a", "owner": "ops"}


# ---------------------------------------------------------------- the free-view fast path


@needs_model
def test_prebuilt_view_matches_reopen_bit_for_bit(tmp_path):
    """track-nudity's perf offer: on a 384 backend the worker's drafted decode also serves
    the nudity view, so the head skips its re-open. The whole point is that scores are
    UNCHANGED — a fast path that moves a score is a silent regression, not a win."""
    pytest.importorskip("imgtag.moderation.nudity")
    from imgtag.moderation import nudity

    head = nudity.load_nudity_head({})
    if head is None or not hasattr(nudity, "make_view"):
        pytest.skip("nudity head/make_view not present on this host")

    src = tmp_path / "imgs"
    src.mkdir()
    paths = []
    for i in range(5):
        p = src / f"v{i}.jpg"
        Image.new("RGB", (900, 600), (i * 40 % 255, i * 25 % 255, 120)).save(p, quality=92)
        paths.append(p)

    # index on the 384 default -> the view fast path is taken (bit-parity precondition met)
    s = index(src, "vv", profile=PROFILE, home=tmp_path / "home", workers=2, moderation=True)
    assert s["moderation_active"]
    view_scores = np.asarray(store.open_snapshot("vv", tmp_path / "home").tracks["nudity"])

    # reference: the head re-opening each file itself (images=None)
    ref = head.score(np.zeros((len(paths), 512), np.float32), None, [{"path": str(p)} for p in paths])
    ref_scores = np.array([(f[0] if isinstance(f, list) else f)["p"] for f in ref], np.float32)

    # per image (both enumerate the same folder, sorted) the scores are identical
    np.testing.assert_allclose(sorted(view_scores), sorted(ref_scores), atol=1e-6)


# ---------------------------------------------------------------- tier taxonomy (P1 regression)


def content_and_unknown_detector(embs, recs, images=None):
    """Emits a CONTENT tier (match) and an UNKNOWN tier — neither may crash the index nor
    inflate the enforcement counts (the sports.py 'match' KeyError, generalized). Keyed on
    the filename so it is deterministic across batch boundaries."""
    out = []
    for r in recs:
        i = int(Path(r["path"]).stem.replace("img", ""))
        out.append([
            {"category": "people", "p": float(i), "tier": "match"},          # content tier, every image
            {"category": "weapons", "p": 0.9, "tier": "violation"} if i == 0 else
            {"category": "future", "p": 0.5, "tier": "some_new_tier"},        # unknown tier, i>0
        ])
    return out


content_and_unknown_detector.specs = {"people": {"tau_review": 0.0}, "weapons": {"tau_violation": 0.5}}


def test_split_and_accumulate_are_defensive():
    from imgtag.core.indexer import accumulate, split_tiers

    # a brand-new tier the engine has never heard of is counted, never a KeyError
    got = accumulate(accumulate({}, {"violation": {"a": 1}}), {"totally_new": {"b": 2}})
    assert got == {"violation": {"a": 1}, "totally_new": {"b": 2}}
    enf, content = split_tiers({"violation": {"a": 1}, "match": {"b": 2}, "weird": {"c": 3}})
    assert enf == {"violation": {"a": 1}}
    assert content == {"match": {"b": 2}, "weird": {"c": 3}}  # everything non-enforcement


@needs_model
def test_content_tier_head_does_not_crash_and_routes_correctly(tmp_path, imgs):
    """P1: a head emitting 'match' (or any non-enforcement tier) must not KeyError, its
    counts must NOT appear in the moderation totals, and the summary must stay clean."""
    from imgtag.core.indexer import moderation_summary

    home = tmp_path / "home"
    s = index(imgs, "ds", profile=PROFILE, home=home, workers=2,
              moderation=f"{__name__}:content_and_unknown_detector")
    assert s["moderation_active"] and s["moderation_errors"] == 0
    # enforcement bucket has weapons; content bucket has the match + unknown tiers
    assert s["moderation"]["violation"]["weapons"] == 1
    assert "match" not in s["moderation"] and "some_new_tier" not in s["moderation"]
    assert s["content"]["match"]["people"] == 4
    assert s["content"]["some_new_tier"]["future"] == 3

    man = store.read_manifest("ds", home)
    assert "match" not in man["moderation"]["counts"]           # never pollutes enforcement
    assert man["content"]["counts"]["match"]["people"] == 4     # lands in its own bucket
    # the user's summary line is enforcement-only — a content tier can't crowd it
    line = moderation_summary(store.read_manifest("ds", home)["moderation"]["counts"])
    assert "people" not in line and "1 images with weapons" in line


# ---------------------------------------------------------------- multi-column tracks (people-counting)


def people_counting_detector(embs, recs, images=None):
    """A [N,4] counting track (track-people's shape): one multi-role record per image plus
    a derived 'match' chip when >=1 person. col_roles order is the on-disk order."""
    out = []
    for r in recs:
        n_p = float(int(Path(r["path"]).stem.replace("img", "")) % 3)   # img0->0 img1->1 img2->2 img3->0
        rec = [{"category": "people",
                "cols": {"n_persons": n_p, "n_faces": max(0.0, n_p - 1),
                         "n_persons_conf": 0.9, "n_faces_conf": 0.8},
                "tier": "none"}]
        if n_p >= 1:  # per-image chip -> content bucket, separate single-value category
            rec.append({"category": "one-person" if n_p == 1 else "multi-person", "p": 1.0, "tier": "match"})
        out.append(rec)
    return out


people_counting_detector.col_roles = {"people": ["n_persons", "n_faces", "n_persons_conf", "n_faces_conf"]}
people_counting_detector.specs = {"people": {"scorer": "cascade", "col_roles":
                                             ["n_persons", "n_faces", "n_persons_conf", "n_faces_conf"]}}


@needs_model
def test_multi_column_people_track_stores_N_by_4(tmp_path, imgs):
    home = tmp_path / "home"
    s = index(imgs, "ds", profile=PROFILE, home=home, workers=2,
              moderation=f"{__name__}:people_counting_detector")
    assert s["moderation_active"]
    snap = store.open_snapshot("ds", home)

    people = np.asarray(snap.tracks["people"])
    assert people.shape == (4, 4), people.shape           # [N, C] dense
    # THE alignment invariant: row i of the column belongs to row i of the ids, whatever
    # order the workers delivered them in — so check each row against its OWN image
    for i, r in enumerate(snap.ids):
        want = int(Path(r["path"]).stem.replace("img", "")) % 3
        assert people[i, 0] == want, (i, r["path"], people[i, 0], want)   # n_persons, col 0
    np.testing.assert_allclose(people[:, 2], 0.9, atol=1e-6)  # n_persons_conf constant

    meta = store.read_track_meta("ds", "people", home)
    assert meta["cols"] == 4
    assert meta["col_roles"] == ["n_persons", "n_faces", "n_persons_conf", "n_faces_conf"]

    # the derived chips (match tier) land in the content bucket, never enforcement
    assert "match" not in s["moderation"]
    assert s["content"]["match"]["one-person"] == 1     # i=1
    assert s["content"]["match"]["multi-person"] == 1   # i=2
    # and no vestigial all-NaN bare column now that pre-seeding is gone (ask #2)
    assert set(snap.tracks) == {"people", "one-person", "multi-person"}


@needs_model
def test_cli_info_image_tracks_matches_the_per_image_object(tmp_path, imgs):
    """B20 parity with GET /api/image/<ds>/<id>/tracks: `info --image <id> --tracks`
    returns the exact per-image all-tracks object (delegated to the one owner, Searcher)."""
    home = tmp_path / "home"
    (tmp_path / "fakemod.py").write_text(
        "def detect(embs, recs, images=None):\n"
        "    return [[{'category':'weapons','p':0.9,'tier':'violation'}] for _ in recs]\n"
        "detect.specs = {'weapons': {'tau_violation': 0.5}}\n")
    import os as _os
    env = {**_os.environ, "IMGTAG_HOME": str(home), "PYTHONPATH": str(tmp_path)}
    cli = [sys.executable, "-m", "imgtag.cli"]

    def run(*a):
        p = subprocess.run(cli + list(a), capture_output=True, text=True, timeout=600, env=env)
        assert p.returncode == 0, p.stderr[-400:]
        return json.loads(p.stdout)

    run("index", str(imgs), "--dataset", "shop", "--wait", "--json", "--moderation-hook", "fakemod:detect")
    obj = run("info", "--image", _read_first_id(home, "shop"), "--tracks", "--dataset", "shop", "--json")
    assert set(obj) >= {"image_id", "dataset", "path", "tracks"}
    assert obj["tracks"] and all({"category", "kind", "tier", "scored"} <= set(t) for t in obj["tracks"])
    # a missing image is a clean exit 4, not a crash
    p = subprocess.run(cli + ["info", "--image", "0" * 16, "--tracks", "--dataset", "shop", "--json"],
                       capture_output=True, text=True, env=env)
    assert p.returncode == 4


def _read_first_id(home, dataset):
    import os as _os

    from imgtag.core import store
    old = _os.environ.get("IMGTAG_HOME")
    _os.environ["IMGTAG_HOME"] = str(home)
    try:
        return store.open_snapshot(dataset).ids[0]["image_id"]
    finally:
        if old:
            _os.environ["IMGTAG_HOME"] = old
