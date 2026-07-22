"""ADR-6 storage tests: exclusion, durability, torn-tail recovery, snapshot isolation."""

import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pytest

from imgtag.core import store


class FakeModel:
    model_id = "fake-8"
    model_sha = "0" * 64
    dim = 8


def _w(home, ds="ds"):
    return store.Writer(ds, FakeModel(), home=home)


def _rows(n, dim=8, seed=0, base=0):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((n, dim)).astype(np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    recs = [{"image_id": f"{base + i:016x}", "path": f"/x/{base + i}.jpg", "dataset": "ds", "w": 4, "h": 4}
            for i in range(n)]
    return e, recs


def test_append_flush_and_snapshot(tmp_path):
    with _w(tmp_path) as w:
        e, r = _rows(10)
        w.append(e, r)
        w._flush_pending()
        assert w.count == 10
    snap = store.open_snapshot("ds", home=tmp_path)
    assert snap.count == 10
    assert snap.emb.shape == (10, 8)
    assert snap.emb.dtype == np.float32
    assert len(snap.ids) == 10
    assert snap.ids[3]["row"] == 3 and snap.ids[3]["path"] == "/x/3.jpg"
    np.testing.assert_allclose(np.asarray(snap.emb), e, rtol=0, atol=0)


def test_reap_stale_uses_the_flock_not_the_status_file(tmp_path):
    """ADR-6: a job frozen at 'running' by a killed process is a corpse the moment its
    writer lock is free — liveness is the kernel's answer, never the status file."""
    from imgtag.core.progress import Job, reap_stale

    Job("ghost1", "ds", 0, home=tmp_path, state="running").start()          # never held a lock
    Job("ghost2", "ds", 0, home=tmp_path, state="queued", pid=2 ** 31 - 1)  # dead pid
    with _w(tmp_path) as w:                                                  # a REAL live writer
        # while a writer holds the lock, nothing for this dataset is reaped
        assert reap_stale("ds", home=tmp_path, keep=w.job_id) == []
    # lock released -> both ghosts are provably dead
    closed = set(reap_stale("ds", home=tmp_path))
    assert {"ghost1", "ghost2"} <= closed
    from imgtag.core.progress import read_job
    assert read_job("ghost1", tmp_path)["state"] == "failed"


def test_queued_job_within_grace_is_not_reaped(tmp_path):
    """A queued job whose recorded pid is alive is loading a model, not dead."""
    import os

    from imgtag.core.progress import Job, is_corpse

    j = Job("starting", "ds", 0, home=tmp_path, state="queued", pid=os.getpid()).state
    assert is_corpse(j, lock_free=True) is False   # our own pid is alive
    j["pid"] = 2 ** 31 - 1
    assert is_corpse(j, lock_free=True) is True     # dead pid


def test_two_writer_exclusion(tmp_path):
    with _w(tmp_path):
        with pytest.raises(store.LockedError):
            with _w(tmp_path):
                pass
    with _w(tmp_path):  # lock released on exit -> second writer succeeds
        pass


def test_model_mismatch_refuses(tmp_path):
    with _w(tmp_path) as w:
        w.append(*_rows(2))
    other = FakeModel()
    other.model_sha = "1" * 64
    with pytest.raises(store.ModelMismatchError):
        with store.Writer("ds", other, home=tmp_path):
            pass


def test_snapshot_isolation_during_append(tmp_path):
    """A snapshot taken mid-job keeps its own count and never sees later rows."""
    with _w(tmp_path) as w:
        w.append(*_rows(5))
        w._flush_pending()
        snap = store.open_snapshot("ds", home=tmp_path)
        assert snap.count == 5 and len(snap.ids) == 5
        w.append(*_rows(7, seed=1, base=100))
        w._flush_pending()
        assert snap.count == 5  # frozen
        assert len(np.asarray(snap.emb)) == 5
        assert store.open_snapshot("ds", home=tmp_path).count == 12  # fresh sees all


def test_writer_refuses_a_duplicate_id(tmp_path):
    """IA.md makes xxhash64-of-bytes the identity, so one dataset may never hold the same
    id twice — the writer refuses rather than trusting callers to dedup (b-daemon's ask)."""
    with _w(tmp_path) as w:
        w.append(*_rows(3))
        with pytest.raises(ValueError, match="duplicate image_id"):
            w.append(*_rows(1))            # same id, already committed
        e, r = _rows(2)
        with pytest.raises(ValueError, match="duplicate image_id"):
            w.append(np.concatenate([e, e]), r + r)   # duplicate WITHIN one batch
    # a fresh writer on the same dataset still knows the ids on disk
    with _w(tmp_path) as w:
        with pytest.raises(ValueError, match="duplicate image_id"):
            w.append(*_rows(1))


def test_track_sidecars_round_trip_and_stay_aligned(tmp_path):
    """ADR-15 T1: dense f32 per track, row-aligned to the shards; RAW scores stored."""
    with _w(tmp_path) as w:
        e, r = _rows(4)
        w.append(e, r, tracks={"nudity": np.array([0.1, 0.9, 0.5, 0.02], np.float32)})
        w._flush_pending()
        e2, r2 = _rows(2, base=50)
        w.append(e2, r2, tracks={"nudity": np.array([0.7, 0.3], np.float32)})
    snap = store.open_snapshot("ds", home=tmp_path)
    col = snap.tracks["nudity"]
    assert len(col) == snap.count == 6
    np.testing.assert_allclose(col[:4], [0.1, 0.9, 0.5, 0.02], atol=1e-6)
    # row alignment is the invariant: row i of the column IS row i of the ids
    assert snap.ids[1]["image_id"] == f"{1:016x}" and col[1] == np.float32(0.9)


def test_a_track_added_late_pads_earlier_rows_as_not_scored(tmp_path):
    with _w(tmp_path) as w:
        w.append(*_rows(3))
        w._flush_pending()
        e, r = _rows(2, base=50)
        w.append(e, r, tracks={"weapons": np.array([0.8, 0.1], np.float32)})
    col = store.open_snapshot("ds", home=tmp_path).tracks["weapons"]
    assert len(col) == 5
    assert np.isnan(col[:3]).all()      # never scored != scored 0.0
    np.testing.assert_allclose(col[3:], [0.8, 0.1], atol=1e-6)


def test_tier_derivation_is_deterministic_and_honest(tmp_path):
    spec = {"tau_alert": 0.9, "tau_violation": 0.5, "tau_review": 0.2}
    got = store.derive_tiers([0.95, 0.6, 0.3, 0.05, float("nan")], spec)
    assert got == ["alert", "violation", "review", "none", "unknown"]
    assert got == store.derive_tiers([0.95, 0.6, 0.3, 0.05, float("nan")], spec)  # B25d
    assert store.derive_tiers([0.05, 0.5], {"tau": 0.02}) == ["violation", "violation"]


def test_torn_tail_recovery(tmp_path):
    with _w(tmp_path) as w:
        w.append(*_rows(6))
        w._flush_pending()
        name = w.name
    d = store.dataset_dir("ds", tmp_path)
    with open(d / name, "ab") as f:  # simulate a crash mid-shard-write
        f.write(b"\x00" * 97)
    with open(d / store._ids_name(name), "ab") as f:
        f.write(b'{"image_id":"deadbe')
    with _w(tmp_path) as w:  # open-for-write recovers
        assert any("truncated torn tail" in a for a in w.recovery)
        assert w.count == 6
    snap = store.open_snapshot("ds", home=tmp_path)
    assert snap.count == 6 and len(snap.ids) == 6 and snap.emb.shape == (6, 8)


def test_short_file_fails_loud_and_quarantines(tmp_path):
    with _w(tmp_path) as w:
        w.append(*_rows(6))
        w._flush_pending()
        name = w.name
    d = store.dataset_dir("ds", tmp_path)
    os.truncate(d / name, 4)
    with pytest.raises(store.CorruptIndexError):
        with _w(tmp_path):
            pass
    assert (d / "trash" / f"SHORT-{name}").exists()


def test_orphan_shard_moved_to_trash(tmp_path):
    with _w(tmp_path) as w:
        w.append(*_rows(3))
    d = store.dataset_dir("ds", tmp_path)
    (d / "shard-deadbeef-0000.f32").write_bytes(b"junk")
    with _w(tmp_path) as w:
        assert any("orphan" in a for a in w.recovery)
    assert (d / "trash" / "shard-deadbeef-0000.f32").exists()


CRASHER = textwrap.dedent(
    """
    import sys, time, numpy as np
    sys.path.insert(0, {src!r})
    from imgtag.core import store
    class M: model_id="fake-8"; model_sha="0"*64; dim=8
    home = {home!r}
    with store.Writer("ds", M(), home=__import__("pathlib").Path(home)) as w:
        i = 0
        while True:
            e = np.full((50, 8), 0.35355339, np.float32)
            recs = [{{"image_id": f"{{i+j:016x}}", "path": f"/x/{{i+j}}.jpg",
                     "dataset": "ds", "w": 1, "h": 1}} for j in range(50)]
            w.append(e, recs)
            i += 50
            if w.count >= 500:
                print("READY", w.count, flush=True)
            time.sleep(0.02)
    """
)


def test_crash_mid_flush_kill9(tmp_path):
    """kill -9 a live writer; the manifest must stay consistent and recovery must
    make the dataset writable + readable again (B21 restart survival, ADR-6)."""
    src = str(Path(__file__).resolve().parents[1] / "src")
    code = CRASHER.format(src=src, home=str(tmp_path))
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    line = p.stdout.readline()  # wait until it has durably committed >= 500 rows
    assert line.startswith("READY"), line
    time.sleep(0.05)  # land in the middle of the next flush cycle
    os.kill(p.pid, signal.SIGKILL)
    p.wait(timeout=10)

    d = store.dataset_dir("ds", tmp_path)
    man = json.loads((d / "manifest.json").read_bytes())  # never torn: atomic rename
    assert man["count"] >= 500 and man["count"] == sum(s["rows"] for s in man["shards"])

    with _w(tmp_path) as w:  # dead pid's flock released by the kernel; no heuristics
        assert w.count == man["count"]
    snap = store.open_snapshot("ds", home=tmp_path)
    assert snap.count == man["count"] == len(snap.ids)
    assert snap.emb.shape == (man["count"], 8)
    np.testing.assert_allclose(np.linalg.norm(np.asarray(snap.emb), axis=1), 1.0, atol=1e-4)


def test_track_sidecar_header_is_the_read_authority(tmp_path):
    """b-daemon's derivation layer reads tracks/<cat>.json, not the .f32 stat — byte
    counts there are authoritative, and spec_sha/model_sha let it refuse a stale sidecar."""
    class M:
        model_id = "fake-8"
        model_sha = "0" * 64
        dim = 8

    with store.Writer("ds", M(), home=tmp_path) as w:
        w._track_specs["nudity"] = {"tau_violation": 0.5, "tau_review": 0.2,
                                    "scorer": "marqo-384", "model_sha": "a" * 64}
        e, r = _rows(4)
        w.append(e, r, tracks={"nudity": np.array([0.1, 0.9, 0.3, 0.6], np.float32)})
    meta = store.read_track_meta("ds", "nudity", tmp_path)
    assert meta["rows"] == 4 and meta["cols"] == 1 and meta["dtype"] == "float32"
    assert meta["col_roles"] == ["p"] and meta["scorer"] == "marqo-384"
    assert meta["bytes"] == (tmp_path / "datasets/ds/tracks/nudity.f32").stat().st_size
    # spec_sha is order-independent so both writer and reader hash identically
    assert meta["spec_sha"] == store.spec_sha({"tau_review": 0.2, "scorer": "marqo-384",
                                               "tau_violation": 0.5, "model_sha": "a" * 64})


def test_fitted_tau_wins_over_recorded_spec(tmp_path, monkeypatch):
    """P1-2: derivation must read the CURRENT fitted file (fitted wins), not a τ baked at
    index time — the store-side/daemon-side split-brain that made every weapons row 'none'."""
    import json as _json

    data = tmp_path / "data" / "moderation"
    data.mkdir(parents=True)
    (data.parent / "moderation").mkdir(exist_ok=True)
    monkeypatch.setattr(store, "_DATA", tmp_path / "data")
    # a fitted file with a real threshold; base model id = "m" (from "m-fp32".rsplit)
    (data / "weapons-m.json").write_text(_json.dumps(
        {"category": "weapons", "tau_violation": 0.8, "tau_review": 0.1, "calibration": "fitted"}))

    cfg = store.resolve_track_cfg("weapons", "m-fp32", spec={"tau_violation": 0.0})  # fitted wins
    assert cfg["tau_violation"] == 0.8 and cfg["tau_review"] == 0.1
    # a caller spec overrides the moderation.json base (general-consumer guarantee) but a
    # fitted file still wins over the caller — the T1 law
    assert store.resolve_track_cfg("nofit", "m", spec={"tau_violation": 0.42})["tau_violation"] == 0.42
    # a score above the fitted violation tau derives 'violation', not 'none'
    assert store.derive_tiers([0.99, 0.5, 0.05], cfg) == ["violation", "review", "none"]
    # absent fitted file -> falls back to the recorded spec, still deterministic
    assert store.resolve_track_cfg("weapons", "other", spec={"tau_violation": 0.3})["tau_violation"] == 0.3


def test_dataset_flags_uses_fitted_tau(tmp_path, monkeypatch):
    import json as _json

    class M:
        model_id = "m-fp32"
        model_sha = "0" * 64
        dim = 8

    data = tmp_path / "data" / "moderation"
    data.mkdir(parents=True)
    monkeypatch.setattr(store, "_DATA", tmp_path / "data")
    (data / "weapons-m.json").write_text(_json.dumps({"tau_violation": 0.5, "tau_review": 0.1}))

    with store.Writer("ds", M(), home=tmp_path) as w:
        e, r = _rows(3)
        w.append(e, r, tracks={"weapons": np.array([0.99, 0.2, 0.01], np.float32)})
    # manifest recorded NO spec (tracks_spec absent) — fitted file must still drive tiers
    tiers = store.dataset_flags("ds", tmp_path)["weapons"]["tiers"]
    assert tiers == ["violation", "review", "none"], tiers
