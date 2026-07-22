"""b-daemon lane: daemon.py — socket lifecycle (ADR-13), endpoints, errors, isolation."""

from __future__ import annotations

import json
import os
import socket
import threading
import time

import numpy as np
import pytest
from PIL import Image

from imgtag import daemon as D
from imgtag.core.store import Writer
from test_search import FakeBackend  # same deterministic backend


@pytest.fixture()
def short_tmp():
    """AF_UNIX sun_path is ~104 bytes — pytest's tmp_path is far too long for a socket."""
    import shutil
    import tempfile
    from pathlib import Path

    d = Path(tempfile.mkdtemp(prefix="it", dir="/tmp"))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


@pytest.fixture()
def live(short_tmp, monkeypatch):
    """A daemon serving a real 6-row dataset over a real UNIX socket."""
    tmp_path = short_tmp
    home = tmp_path / "home"
    home.mkdir()
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    be = FakeBackend()
    # cats are RARE in the corpus: the provisional free-text calibration scores a hit by how
    # far it stands out from THIS corpus (per-query z-score), so a 1-in-3 class is not a hit.
    labels = ["cat", "cat"] + ["car"] * 14 + ["tree"] * 14
    recs, embs = [], []
    for i, lab in enumerate(labels):
        p = imgs / f"{i}.jpg"
        Image.new("RGB", (64, 48), (10 * i, 90, 140)).save(p, "JPEG")
        recs.append({"image_id": f"{i:016x}", "path": str(p), "dataset": "d1", "w": 64, "h": 48})
        embs.append(be._vec(lab))
    with Writer("d1", be, home) as w:
        w.append(np.stack(embs), recs)

    monkeypatch.setenv("IMGTAG_HOME", str(home))
    t = threading.Thread(target=D.serve, kwargs={"home": home, "log": lambda *_: None}, daemon=True)
    t.start()
    sock = home / "daemon.sock"
    for _ in range(100):
        if sock.exists():
            break
        time.sleep(0.05)
    assert sock.exists(), "daemon never bound its socket"
    D.Handler.daemon.searcher._backend = be  # no real model on a test box
    yield home
    D.request("POST", "/api/shutdown", home=home)
    t.join(timeout=10)


def get(home, path):
    return D.request("GET", path, home=home)


# ---------------------------------------------------------------- lifecycle


def test_socket_lifecycle_and_single_instance(live):
    home = live
    sock, lock, rec = D.daemon_paths(home)
    assert sock.is_socket() and (os.stat(sock).st_mode & 0o777) == 0o600  # B22
    assert (os.stat(home).st_mode & 0o777) == 0o700
    record = json.loads(rec.read_bytes())
    assert record["pid"] == os.getpid() and record["http_port"] is None
    assert record["socket"] == str(sock)

    # a second daemon must refuse (flock held) with exit code 3, and must NOT unlink the socket
    assert D.serve(home=home, log=lambda *_: None) == 3
    assert sock.is_socket()
    st, body = get(home, "/api/hello")
    assert st == 200 and body["version"] == D.VERSION


def test_stale_socket_takeover(short_tmp):
    tmp_path = short_tmp
    """A socket file left by a dead daemon is unlinked by the next lock holder (ADR-13)."""
    home = tmp_path / "h"
    home.mkdir()
    sock, _, _ = D.daemon_paths(home)
    sock.write_bytes(b"")  # a stale FILE where the socket belongs
    t = threading.Thread(target=D.serve, kwargs={"home": home, "log": lambda *_: None}, daemon=True)
    t.start()
    for _ in range(100):
        if sock.is_socket():
            break
        time.sleep(0.05)
    assert sock.is_socket()
    st, _ = D.request("GET", "/api/hello", home=home)
    assert st == 200
    D.request("POST", "/api/shutdown", home=home)
    t.join(timeout=10)
    assert not sock.exists()  # cleaned up on the way out


def test_shutdown_removes_socket_and_record(short_tmp):
    tmp_path = short_tmp
    home = tmp_path / "h2"
    home.mkdir()
    t = threading.Thread(target=D.serve, kwargs={"home": home, "log": lambda *_: None}, daemon=True)
    t.start()
    sock, _, rec = D.daemon_paths(home)
    for _ in range(100):
        if sock.exists():
            break
        time.sleep(0.05)
    D.request("POST", "/api/shutdown", home=home)
    t.join(timeout=10)
    assert not sock.exists() and not rec.exists()


# ---------------------------------------------------------------- endpoints


def test_search_endpoint_provenance_and_no_match(live):
    st, r = get(live, "/api/search?q=cat&dataset=d1&k=5")
    assert st == 200 and r["hits"]
    for h in r["hits"]:  # B18: never null
        assert h["image_id"] and h["path"] and h["dataset"] == "d1" and h["dataset_slug"] == "d1"
        assert h["why"]["path"] in ("tag", "text")
    assert r["coverage"] == {"indexed": 30, "total": 30}

    st, r = get(live, "/api/search?q=boat&dataset=d1")
    assert st == 200 and r["hits"] == [] and r["no_match"] is True  # honest no-match, HTTP 200


def test_datasets_and_jobs_endpoints(live):
    st, r = get(live, "/api/datasets")
    assert st == 200 and [d["dataset"] for d in r["datasets"]] == ["d1"]
    assert r["datasets"][0]["count"] == 30  # B18(f): the manifest count, exactly
    st, r = get(live, "/api/jobs")
    assert st == 200 and r["jobs"] == []


def test_thumb_endpoint_caches_on_disk(live):
    st, body = D.request("GET", "/api/thumb/d1/0000000000000000?s=64", home=live)
    assert st == 200 and body[:2] == b"\xff\xd8"  # JPEG SOI
    cached = live / "thumbs" / "d1" / "0000000000000000-64.jpg"
    assert cached.is_file()
    st2, body2 = D.request("GET", "/api/thumb/d1/0000000000000000?s=64", home=live)
    assert body2 == body
    st3, err = D.request("GET", "/api/thumb/d1/ffffffffffffffff?s=64", home=live)
    assert st3 == 404 and err["exit_code"] == 4


def test_error_contract(live):
    st, r = get(live, "/api/search?q=cat&dataset=ghost")
    assert st == 404 and r["exit_code"] == 4 and r["code"] == "UnknownDatasetError"
    st, r = get(live, "/api/search")
    assert st == 400 and r["exit_code"] == 1
    st, r = get(live, "/api/nope")
    assert st == 404
    st, r = D.request("POST", "/api/index", {"dataset": "x"}, home=live)
    assert st == 400 and r["exit_code"] == 1
    st, r = D.request("POST", "/api/index", {"dataset": "x", "path": "/no/such/dir"}, home=live)
    assert st == 404 and r["exit_code"] == 4


def test_events_stream_is_fresh(live):
    """SSE emits job state within ~1s of the status file changing."""
    from imgtag.core.progress import Job

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(6)
    sock.connect(str(D.daemon_paths(live)[0]))
    sock.sendall(b"GET /api/events HTTP/1.1\r\nHost: localhost\r\n\r\n")
    assert b"text/event-stream" in sock.recv(4096)
    j = Job("testjob1", "d1", total=7, home=live)
    j.update(3, 1, force=True)
    t0 = time.time()
    buf = b""
    while time.time() - t0 < 4 and b"testjob1" not in buf:
        buf += sock.recv(4096)
    sock.close()
    assert b"event: job" in buf and b"testjob1" in buf
    payload = json.loads(buf.split(b"data: ")[-1].split(b"\n\n")[0])
    assert payload["total"] == 7 and payload["done"] == 3


def test_search_while_appending_stays_consistent(live):
    """Snapshot isolation through the HTTP door: rows appear only after their commit."""
    be = D.Handler.daemon.searcher._backend
    stop, errs, counts = threading.Event(), [], []

    def hammer():
        while not stop.is_set():
            try:
                st, r = get(live, "/api/search?q=cat&dataset=d1&k=50")
                assert st == 200
                counts.append(r["coverage"]["indexed"])
                assert all(h["path"] and h["image_id"] for h in r["hits"])
            except Exception as e:  # pragma: no cover
                errs.append(e)
                return

    t = threading.Thread(target=hammer)
    t.start()
    try:
        with Writer("d1", be, live) as w:
            for i in range(30, 54, 6):
                w.append(
                    np.stack([be._vec("cat")] * 6),
                    [{"image_id": f"{j:016x}", "path": f"/x/{j}.jpg", "dataset": "d1", "w": 1, "h": 1}
                     for j in range(i, i + 6)],
                )
                time.sleep(0.3)
    finally:
        stop.set()
        t.join(timeout=10)
    assert not errs, errs
    assert counts and max(counts) >= 30 and counts == sorted(counts)  # never goes backwards
    st, r = get(live, "/api/search?q=cat&dataset=d1&k=100")
    assert r["coverage"]["indexed"] == 54


def test_idle_ttl_evicts_the_model_but_not_the_daemon(live):
    """ADR-5 (revised): eviction drops the MODEL; the daemon keeps serving."""
    d = D.Handler.daemon
    released = []
    d.searcher._backend.release_text = lambda: released.append(True)
    d.searcher.last_query = time.time() - 100
    t = threading.Thread(target=D._memory_watch, args=(d, 1.0, 0.0), daemon=True)
    t.start()
    for _ in range(60):
        if d.evictions:
            break
        time.sleep(0.05)
    assert released, "model was never released"
    assert d.evictions and d.evictions[-1]["reason"] == "idle"
    st, r = get(live, "/api/hello")
    assert st == 200 and r["text_tower"] == "unloaded"  # daemon still serving, model gone


def test_rss_watermark_evicts_the_model(live):
    """The memory-watermark half of the policy: a crossed watermark evicts, TTL or not."""
    d = D.Handler.daemon
    released = []
    d.searcher._backend.release_text = lambda: released.append(True)
    d.searcher.text_loaded = True
    t = threading.Thread(target=D._memory_watch, args=(d, 0.0, 0.001, 0.2), daemon=True)  # 1KB watermark, 0.2s period
    t.start()
    for _ in range(60):
        if d.evictions:
            break
        time.sleep(0.1)
    assert released and d.evictions[-1]["reason"] == "watermark"
    assert D.rss_mb() > 1.0  # the probe itself works on this platform


def test_named_tag_query_never_wakes_the_text_tower(live):
    """ADR-5 revised: a query that NAMES a tag is served with zero text-encoder work."""
    import json as _json

    import numpy as _np

    from test_search import DIM

    be = D.Handler.daemon.searcher._backend
    d = live / "models" / be.model_sha
    d.mkdir(parents=True, exist_ok=True)
    _np.stack([be._vec("cat")]).astype(_np.float32).tofile(d / "tags.f32")
    (d / "tags.json").write_text(_json.dumps({"names": ["cat"], "dim": DIM,
                                              "model_sha": be.model_sha, "tier": ["calibrated"],
                                              "tau": [0.5], "platt": [[-12.0, 6.0]]}))
    s = D.Handler.daemon.searcher
    s._tags.clear()
    boom = []
    s._backend.embed_texts = lambda *a, **k: boom.append(a) or (_ for _ in ()).throw(
        AssertionError("text tower was used for a named-tag query"))

    st, r = get(live, "/api/search?q=cat&dataset=d1&k=5&text=auto")
    assert st == 200 and r["hits"] and not boom
    assert r["text_tower"] == "skipped" and r["text_tower_load_ms"] == 0.0
    assert r["hits"][0]["why"]["path"] == "tag" and r["hits"][0]["score"] > 0

    # text=never + an unnamed query = an honest no-match, still no encoder
    st, r = get(live, "/api/search?q=some+unnamed+thing&dataset=d1&text=never")
    assert st == 200 and r["hits"] == [] and r["no_match"] is True and not boom


def test_status_and_images_endpoints(live):
    """The app's health strip + gallery listing (b-app calls both)."""
    st, r = get(live, "/api/status")
    assert st == 200 and r["rss"] > 0 and r["rss_mb"] > 0
    assert r["text_tower"] in ("loaded", "unloaded") and r["jobs"] == 0
    assert [d["dataset"] for d in r["datasets"]] == ["d1"]
    assert r["datasets"][0]["bytes"] > 0  # index-on-disk metric

    st, r = get(live, "/api/images?dataset=d1&offset=0&limit=4")
    assert st == 200 and r["total"] == 30 and len(r["items"]) == 4
    for it in r["items"]:  # B18 provenance on the gallery path too
        assert it["image_id"] and it["path"] and it["dataset"] == "d1" and it["dataset_slug"] == "d1"
    st, r2 = get(live, "/api/images?dataset=d1&offset=2&limit=2")
    assert [i["image_id"] for i in r2["items"]] == [i["image_id"] for i in r["items"][2:4]]  # stable paging
    st, r = get(live, "/api/images")
    assert st == 400 and r["exit_code"] == 1
    st, r = get(live, "/api/images?dataset=ghost")
    assert st == 404 and r["exit_code"] == 4


def test_images_contract(live):
    """b-app's gallery source: stable manifest order, durable total, exists flag, cap 500."""
    st, r = get(live, "/api/images?dataset=d1&offset=0&limit=999")
    assert st == 200 and r["limit"] == 500 and r["total"] == 30
    first5 = [i["image_id"] for i in r["items"][:5]]
    st, r2 = get(live, "/api/images?dataset=d1&offset=2&limit=3")
    assert [i["image_id"] for i in r2["items"]] == first5[2:5]  # stable, deterministic paging
    it = r2["items"][0]
    assert set(it) >= {"image_id", "path", "dataset", "w", "h", "exists"}
    assert it["exists"] is True
    st, r3 = get(live, "/api/images?dataset=d1&offset=0&limit=1")
    assert r3["items"][0]["exists"] is True


def test_datasets_and_status_contract(live):
    st, r = get(live, "/api/datasets")
    d = r["datasets"][0]
    assert d["index_bytes"] > 0 and d["index_bytes"] == d["bytes"]
    assert d["total"] >= d["count"] == 30
    assert d["root_path"]  # common parent of the indexed files
    st, r = get(live, "/api/status")
    assert {"rss_mb", "uptime_s", "models_loaded", "text_tower_resident"} <= set(r)
    assert isinstance(r["models_loaded"], list) and isinstance(r["text_tower_resident"], bool)


def test_app_assets_resolve_at_root_and_under_app(live):
    """b-app may reference assets relatively (/app.js) or absolutely (/app/app.js)."""
    for path in ("/", "/app/app.js", "/app.js", "/app.css"):
        st, body = D.request("GET", path, home=live)
        assert st == 200 and body, path
