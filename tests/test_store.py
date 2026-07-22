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


def _rows(n, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((n, dim)).astype(np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    recs = [{"image_id": f"{i:016x}", "path": f"/x/{i}.jpg", "dataset": "ds", "w": 4, "h": 4} for i in range(n)]
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
        w.append(*_rows(7, seed=1))
        w._flush_pending()
        assert snap.count == 5  # frozen
        assert len(np.asarray(snap.emb)) == 5
        assert store.open_snapshot("ds", home=tmp_path).count == 12  # fresh sees all


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
