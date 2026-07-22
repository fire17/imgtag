"""Resident search daemon — ADR-5 (resident set, revised) + ADR-13 (lifecycle), stdlib only.

Resident set at IDLE = mmap'd shards + the tag table (+ b-engine's binary tokenizer) and
NO model: measured 51.9MB fresh, 181MB after an eviction, against B8's 350MB idle cap.
A free-text query lazy-loads the model (measured 427MB resident, 489ms one-time) and the
eviction policy — idle ``--text-ttl`` OR the ``--rss-watermark-mb`` high-water mark — drops
it again. The DAEMON itself never exits on idle (immich's model_ttl anti-pattern). A query
that NAMES a tag is served from the tag table with zero text-encoder involvement.

OWNER: b-daemon. ``ThreadingHTTPServer`` over a per-user UNIX socket
(``~/.imgtag/daemon.sock``, 0600); ``--tcp PORT`` is the app's opt-in loopback door and
refuses any non-loopback bind. Single instance via ``fcntl.flock`` on
``~/.imgtag/daemon.lock`` held for the process lifetime — the same kernel-owned pattern as
the index writer (ADR-6), never pid heuristics. The endpoint record is ``daemon.json``.

Endpoints::

    GET  /api/hello                              version, model_shas, rss_mb, text_tower,
                                                 eviction policy + last evictions
    GET  /api/search?q=&dataset=&k=&strict=&text= Search API contract (briefs §Search API);
                                                 text=auto|never|always picks the encoder
                                                 policy, and the reply labels text_tower +
                                                 text_tower_load_ms so a warm budget (B3)
                                                 can exclude the one-time load
    GET  /api/datasets                           fleet view (B18f: exactly what is on disk)
    GET  /api/status                             footprint + fleet + job count (app health)
    GET  /api/moderation?dataset=&limit=&source=  source=current-scan (default) | stored
                                                 ADR-14 two-tier counts per category
                                                 ("N violations, M for review") + flagged
                                                 images when limit>0. LIVE scan; b-engine's
                                                 stored index-time flags are a separate,
                                                 separately-labelled source.
    GET  /api/images?dataset=&offset=&limit=     paged snapshot listing (gallery view)
    GET  /api/jobs                               job status files (progress.list_jobs)
    GET  /api/events                             SSE, <=1s freshness, from the job files
    GET  /api/thumb/<dataset>/<image_id>?s=256   draft-decoded JPEG, LRU disk cache <=200MB
    POST /api/index {path, dataset}              spawns `imgtag index`, returns {job_id}
    POST /api/shutdown                           ADR-13 version/model-upgrade restart
    GET  /  /app/*                               b-app's static files, when present

Errors are JSON ``{error, code, exit_code}`` with the B20 exit codes; the daemon never
prints a traceback at a client.
"""

from __future__ import annotations

import argparse
import fcntl
import http.client
import io
import json
import os
import socket
import socketserver
import subprocess
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .core import models as _models
from .core.progress import Job, list_jobs, read_job
from .core.search import CalibrationMismatchError, Searcher
from .core.store import (
    CorruptIndexError,
    LockedError,
    ModelMismatchError,
    UnknownDatasetError,
    imgtag_home,
    list_datasets,
    read_manifest,
)

VERSION = "0.1.0"
THUMB_CAP_BYTES = 200 * 1024 * 1024  # B8: thumbs are <=200MB of the 500MB disk budget
EVENT_POLL_S = 0.4  # SSE freshness <=1s
DEFAULT_WATERMARK_MB = 1200.0  # evict the model above this; B8's under-load cap is 1.5GB
APP_DIR = Path(__file__).resolve().parent / "app"

# exception -> (HTTP status, B20 exit code)
ERRORS = {
    UnknownDatasetError: (404, 4),
    LockedError: (423, 3),
    ModelMismatchError: (409, 5),
    CalibrationMismatchError: (409, 5),
    CorruptIndexError: (500, 6),
    _models.ModelUnavailableError: (503, 7),
    FileNotFoundError: (404, 4),
    ValueError: (400, 1),
}


PARAMS = {  # unknown query params are a 400, never silently ignored (b-app found ?source=)
    "/api/search": {"q", "dataset", "k", "strict", "text", "track", "tier"},
    "/api/moderation": {"dataset", "limit", "source"},
    "/api/images": {"dataset", "offset", "limit"},
    "/api/thumb": {"s"},
}


def daemon_paths(home: Path | None = None) -> tuple[Path, Path, Path]:
    h = home or imgtag_home()
    return h / "daemon.sock", h / "daemon.lock", h / "daemon.json"


# ---------------------------------------------------------------- thumbnails


def thumb_bytes(path: str, size: int) -> bytes:
    """Draft-decode (JPEG DCT scaling) — never a full decode for a 256px tile."""
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im.draft("RGB", (size, size))  # JPEG-only fast path; a no-op elsewhere
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((size, size), Image.Resampling.BILINEAR)
        out = io.BytesIO()
        im.save(out, "JPEG", quality=82, optimize=False)
    return out.getvalue()


class ThumbCache:
    """LRU disk cache under ``~/.imgtag/thumbs`` (B8 <=200MB). Swept by mtime, cheaply:
    the total is tracked in memory and a sweep only runs when the cap is crossed."""

    def __init__(self, home: Path, cap: int = THUMB_CAP_BYTES):
        self.dir = home / "thumbs"
        self.dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.cap = cap
        self._lock = threading.Lock()
        self._bytes = sum(p.stat().st_size for p in self.dir.rglob("*.jpg"))

    def get(self, dataset: str, image_id: str, path: str, size: int) -> bytes:
        f = self.dir / dataset / f"{image_id}-{size}.jpg"
        try:
            data = f.read_bytes()
            os.utime(f, None)  # touch: mtime is the LRU key
            return data
        except OSError:
            pass
        data = thumb_bytes(path, size)
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, f)
        with self._lock:
            self._bytes += len(data)
            if self._bytes > self.cap:
                self._sweep()
        return data

    def _sweep(self) -> None:
        files = sorted(self.dir.rglob("*.jpg"), key=lambda p: p.stat().st_mtime)
        total = sum(p.stat().st_size for p in files)
        target = int(self.cap * 0.8)
        for p in files:
            if total <= target:
                break
            try:
                total -= p.stat().st_size
                p.unlink()
            except OSError:
                pass
        self._bytes = total


# ---------------------------------------------------------------- server


class Daemon:
    """Process-wide state: the resident Searcher (warm text tower) + thumb cache."""

    def __init__(self, home: Path, backend: str | None = None,
                 text_ttl: float = 0.0, watermark_mb: float = 0.0):
        self.home = home
        self.text_ttl, self.watermark_mb = text_ttl, watermark_mb
        self.started = time.time()
        self.searcher = Searcher(home, backend=_models.load_backend(backend) if backend else None)
        self.thumbs = ThumbCache(home)
        self.stop = threading.Event()
        self.evictions: list[dict] = []  # ADR-5 model-eviction audit trail (last 20)

    def eviction_policy(self) -> str:
        """ADR-5 ruled default: KEEP the model while RSS stays under the watermark. A time
        TTL is the opt-in override, so `text_ttl_s: null` means "no timer", not "evict now"
        — paying a ~400ms load after every idle period is immich's sin with extra steps."""
        if self.text_ttl > 0:
            return f"idle-ttl {self.text_ttl:g}s or rss>{self.watermark_mb:g}MB"
        return f"keep-while-rss<{self.watermark_mb:g}MB" if self.watermark_mb > 0 else "keep"

    def model_shas(self) -> dict:
        b = self.searcher._backend
        return {b.model_id: b.model_sha} if b else {}

    def datasets(self) -> list[dict]:
        """Fleet view: exactly the datasets on disk, counts straight from their manifests."""
        out = []
        jobs = list_jobs(self.home)
        for name in list_datasets(self.home):
            try:
                m = read_manifest(name, self.home)
            except UnknownDatasetError:
                continue
            nbytes = sum(sh.get("emb_bytes", 0) + sh.get("ids_bytes", 0) for sh in m.get("shards", []))
            mine = [j for j in jobs if j.get("dataset") == name]
            last = max(mine, key=lambda j: float(j.get("started") or 0), default=None)
            # `total` = files the last job SAW on disk; `count` = rows durably indexed. They
            # differ exactly when a job is running or something failed — which is the point.
            total = max(int((last or {}).get("total") or 0), int(m.get("count", 0)))
            out.append(
                {
                    "dataset": name,
                    "count": m.get("count", 0),
                    "model_id": m.get("model_id"),
                    "model_sha": m.get("model_sha"),
                    "dim": m.get("dim"),
                    "total": total,
                    "shards": len(m.get("shards", [])),
                    "bytes": nbytes,
                    "index_bytes": nbytes,
                    "root_path": m.get("root") or (last or {}).get("source") or self.root_of(name),
                    "calibrated": bool(m.get("calib_sha")),
                    "created": m.get("created"),
                    "updated": m.get("updated"),
                }
            )
        return out

    def root_of(self, dataset: str) -> str:
        """Common parent of the indexed files — the dataset's root as the app shows it."""
        try:
            paths = [r["path"] for r in self.searcher.snapshot(dataset).ids[:200]]
        except Exception:
            return ""
        return os.path.commonpath(paths) if paths else ""

    def jobs(self) -> list[dict]:
        """Job list with mirrored engine records folded away — one row per real job."""
        js = list_jobs(self.home)
        mirrored = {j.get("engine_job_id") for j in js if j.get("engine_job_id")}
        return [j for j in js if j.get("job_id") not in mirrored]

    def status(self) -> dict:
        """One call for the app's health strip: footprint + fleet + job count."""
        mb = rss_mb()
        return {
            "version": VERSION, "pid": os.getpid(),
            "uptime_s": round(time.time() - self.started, 1),
            "rss": int(mb * 1024 * 1024), "rss_mb": round(mb, 1),
            "text_tower": "loaded" if self.searcher.text_loaded else "unloaded",
            "text_tower_resident": bool(self.searcher.text_loaded),
            "models_loaded": sorted(self.model_shas()),
            "eviction_policy": self.eviction_policy(),
            "text_ttl_s": self.text_ttl or None, "rss_watermark_mb": self.watermark_mb,
            "datasets": self.datasets(), "jobs": len(self.jobs()),
        }

    def images(self, dataset: str, offset: int, limit: int) -> dict:
        """Paged listing straight off the snapshot — the gallery view's source. Rows are
        manifest order (= index order), so paging is stable while a job appends."""
        snap = self.searcher.snapshot(dataset)
        ids = snap.ids[offset : offset + limit]
        return {
            # total is the DURABLE manifest count (ADR-6 progress authority), so a gallery
            # scrollbar never disagrees with the coverage banner.
            "dataset": dataset, "total": int(snap.manifest.get("count", len(snap.ids))),
            "offset": offset, "limit": limit,
            "items": [
                {"image_id": r["image_id"], "path": r["path"],
                 "dataset": r.get("dataset") or dataset,          # B18: never null
                 "dataset_slug": r.get("dataset") or dataset,
                 "w": r.get("w"), "h": r.get("h"),
                 # B18(b): a moved/deleted file is TOMBSTONED, never a 404 in the grid
                 "exists": os.path.exists(r["path"])}
                for r in ids
            ],
        }

    def record(self, ids, dataset: str, image_id: str) -> dict:
        for rec in ids:
            if rec["image_id"] == image_id:
                return rec
        raise FileNotFoundError(f"image {image_id!r} not in dataset {dataset!r}")

    def start_index(self, path: str, dataset: str, meta: dict | None = None,
                    moderation: bool = False) -> dict:
        """Spawn `imgtag index` and return its job id (<=500ms, B20).

        The engine accepts `--job-id` now, so the daemon mints the id, opens the job record
        and hands the SAME id to the CLI — one record, no mirroring. A watcher only exists
        to mark the job failed if the child dies before writing its own status.
        """
        import importlib.util
        import uuid

        src = Path(path).expanduser()
        if not src.exists():
            raise FileNotFoundError(f"path does not exist: {src}")
        if importlib.util.find_spec("imgtag.cli") is None:
            raise NotImplementedError("imgtag.cli is not installed — nothing to spawn")
        job_id = uuid.uuid4().hex[:8]
        job = Job(job_id, dataset, total=0, home=self.home, source=str(src), origin="daemon")
        argv = [sys.executable, "-m", "imgtag.cli", "index", str(src), "--dataset", dataset,
                "--wait", "--job-id", job_id]
        for k, v in (meta or {}).items():  # generic index-time metadata (VISION 12:33Z)
            argv += ["--meta", f"{k}={v}"]
        if moderation:
            argv.append("--moderation")
        proc = subprocess.Popen(
            argv, env={**os.environ, "IMGTAG_HOME": str(self.home)},
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, start_new_session=True)
        threading.Thread(target=self._reap, args=(proc, job), daemon=True).start()
        return {"job_id": job_id, "dataset": dataset, "path": str(src), "pid": proc.pid}

    def _reap(self, proc, job: Job) -> None:
        """Only failure handling: a child that dies without writing its own status would
        otherwise leave the job 'queued' forever."""
        err = (proc.communicate()[1] or b"").decode()[-500:]
        if proc.returncode:
            fresh = read_job(job.state["job_id"], self.home) or {}
            if fresh.get("state") not in ("done", "failed"):
                job.fail(f"imgtag index exited {proc.returncode}: {err.strip()}")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = f"imgtag/{VERSION}"
    daemon: Daemon = None  # injected by serve()

    # -- plumbing --------------------------------------------------
    def log_message(self, fmt, *a):  # quiet by default; the job files are the log
        if os.environ.get("IMGTAG_DAEMON_VERBOSE"):
            sys.stderr.write("%s %s\n" % (self.address_string(), fmt % a))

    def address_string(self) -> str:
        return "unix" if isinstance(self.client_address, tuple) and self.client_address[0] == "unix" else str(self.client_address[0])

    def _send(self, status: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def json(self, obj, status: int = 200) -> None:
        self._send(status, json.dumps(obj).encode(), "application/json")

    def fail(self, exc: BaseException) -> None:
        status, code = next(
            ((s, c) for t, (s, c) in ERRORS.items() if isinstance(exc, t)), (500, 1)
        )
        if isinstance(exc, NotImplementedError):
            status, code = 501, 1
        self.json({"error": str(exc), "code": type(exc).__name__, "exit_code": code}, status)

    # -- routes ----------------------------------------------------
    def do_GET(self) -> None:
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        d = self.daemon
        try:
            known = PARAMS.get(u.path if not u.path.startswith("/api/thumb/") else "/api/thumb")
            unknown = sorted(set(q) - known) if known else []
            if unknown:
                raise ValueError(
                    f"unknown query parameter(s) {', '.join(unknown)} for {u.path} "
                    f"— accepted: {', '.join(sorted(known))}")
            if u.path == "/api/hello":
                self.json({"version": VERSION, "model_shas": d.model_shas(), "pid": os.getpid(),
                           "socket": str(daemon_paths(d.home)[0]),
                           "uptime_s": round(time.time() - d.started, 1),
                           # ADR-5 revised: resident set is shards + tokenizer + tag table at
                           # idle; the model is a transient the eviction policy owns.
                           "rss_mb": round(rss_mb(), 1),
                           "text_tower": "loaded" if d.searcher.text_loaded else "unloaded",
                           "eviction_policy": d.eviction_policy(),
                           "text_ttl_s": d.text_ttl or None, "rss_watermark_mb": d.watermark_mb,
                           "evictions": d.evictions[-5:]})
            elif u.path == "/api/search":
                query = (q.get("q") or [""])[0]
                trk = (q.get("track") or [None])[0] or None
                k = max(1, min(500, int((q.get("k") or [50])[0])))
                if not query.strip() and trk:
                    # "show me the flagged images": a track with no query is a browse of
                    # that track, ranked by track probability instead of query relevance.
                    ds = (q.get("dataset") or [None])[0]
                    names = [ds] if ds else list_datasets(d.home)
                    want_tier = (q.get("tier") or [None])[0] or None
                    hits = [h for nm in names
                            for h in d.searcher.moderation(nm, limit=k).get("flagged", [])
                            if h["category"] == trk and (not want_tier or h["tier"] == want_tier)]
                    hits.sort(key=lambda h: (h["tier"] != "violation", -h["p"], h["image_id"]))
                    info = d.searcher.track_state(names[0], trk) if names else {}
                    self.json({"query": "", "track": trk, "tookMs": 0.0, "hits": hits[:k],
                               "no_match": not hits,
                               "calibration": info.get("calibration", "unfitted"),
                               "track_calibration": info.get("calibration", "unfitted"),
                               "spec_calibration": info.get("spec_calibration", "unfitted"),
                               "enforcement_ready": info.get("enforcement_ready", False),
                               "coverage": {"indexed": sum(d.searcher.snapshot(n).count
                                                           for n in names)},
                               "datasets": names})
                    return
                if not query.strip():
                    raise ValueError("q is required (or pass track= to browse a track)")
                if trk:  # the track's own calibration travels with a track-filtered query
                    ds0 = (q.get("dataset") or [None])[0] or (list_datasets(d.home) or [None])[0]
                    info = d.searcher.track_state(ds0, trk) if ds0 else {}
                else:
                    info = {}
                res = d.searcher.search(
                    query,
                    dataset=(q.get("dataset") or [None])[0] or None,
                    k=k,
                    strict=(q.get("strict") or ["0"])[0] in ("1", "true", "yes"),
                    text=(q.get("text") or ["auto"])[0],
                    track=trk,
                    tier=(q.get("tier") or [None])[0] or None,
                )
                if info:
                    res["track_calibration"] = info["calibration"]
                    res["spec_calibration"] = info["spec_calibration"]
                    res["enforcement_ready"] = info["enforcement_ready"]
                self.json(res)
            elif u.path == "/api/moderation":
                ds = (q.get("dataset") or [None])[0]
                limit = max(0, min(200, int((q.get("limit") or [0])[0])))
                names = [ds] if ds else list_datasets(d.home)
                src = (q.get("source") or ["current-scan"])[0]
                if src not in ("current-scan", "stored"):
                    raise ValueError("source must be 'current-scan' or 'stored'")
                per = [(d.searcher.stored_moderation(nm, limit) if src == "stored"
                        else d.searcher.moderation(nm, limit)) for nm in names]
                totals: dict[str, dict] = {}
                for r in per:  # ADR-14: violations and review counts never merge into one
                    for cat, c in r["counts"].items():
                        t = totals.setdefault(cat, {"violation": 0, "review": 0})
                        t["violation"] += c["violation"]
                        t["review"] += c["review"]
                self.json({"datasets": per, "totals": totals,
                           "indexed": sum(r["indexed"] for r in per),
                           # "stored" = flagged at indexing (survives a threshold change);
                           # "current-scan" = today's detectors over today's embeddings
                           "source": src,
                           # per-category, straight from each track's spec (a fitted track
                           # says so; an unfitted one can never claim otherwise)
                           "calibration": {c: v for r in per for c, v in r["calibration"].items()},
                           "enforcement_ready": {c: v for r in per
                                                 for c, v in r["enforcement_ready"].items()}})
            elif u.path == "/api/status":
                self.json(d.status())
            elif u.path == "/api/images":
                ds = (q.get("dataset") or [None])[0]
                if not ds:
                    raise ValueError("dataset is required")
                self.json(d.images(ds, max(0, int((q.get("offset") or [0])[0])),
                                   max(1, min(500, int((q.get("limit") or [300])[0])))))
            elif u.path == "/api/datasets":
                self.json({"datasets": d.datasets()})
            elif u.path == "/api/jobs":
                self.json({"jobs": d.jobs()})
            elif u.path == "/api/events":
                self.events()
            elif u.path.startswith("/api/thumb/"):
                self.thumb(u.path, q)
            elif u.path == "/" or (APP_DIR / u.path.lstrip("/")).is_file() or u.path.startswith("/app/"):
                self.static(u.path)  # assets resolve at BOTH / and /app/ (b-app relative refs)
            else:
                self.json({"error": f"no route {u.path}", "code": "NotFound", "exit_code": 1}, 404)
        except BrokenPipeError:
            pass
        except Exception as e:  # never leak a traceback at a client
            self.fail(e)

    def do_POST(self) -> None:
        u = urllib.parse.urlparse(self.path)
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}") if n else {}
            if u.path == "/api/index":
                if not body.get("path") or not body.get("dataset"):
                    raise ValueError("path and dataset are required")
                self.json(self.daemon.start_index(
                    str(body["path"]), str(body["dataset"]),
                    meta=body.get("meta") or None, moderation=bool(body.get("moderation"))), 202)
            elif u.path == "/api/shutdown":
                self.json({"stopping": True, "pid": os.getpid()})
                self.daemon.stop.set()
                threading.Thread(target=self.server.shutdown, daemon=True).start()
            else:
                self.json({"error": f"no route {u.path}", "code": "NotFound", "exit_code": 1}, 404)
        except BrokenPipeError:
            pass
        except Exception as e:
            self.fail(e)

    # -- heavier handlers ------------------------------------------
    def thumb(self, path: str, q: dict) -> None:
        parts = path.split("/")  # /api/thumb/<dataset>/<image_id>
        if len(parts) != 5:
            raise ValueError("usage: /api/thumb/<dataset>/<image_id>?s=256")
        dataset, image_id = urllib.parse.unquote(parts[3]), urllib.parse.unquote(parts[4])
        size = max(16, min(1024, int((q.get("s") or [256])[0])))
        snap = self.daemon.searcher.snapshot(dataset)
        rec = self.daemon.record(snap.ids, dataset, image_id)
        data = self.daemon.thumbs.get(dataset, image_id, rec["path"], size)
        self._send(200, data, "image/jpeg", {"Cache-Control": "public, max-age=86400"})

    def events(self) -> None:
        """SSE over the job status files — no polling thread in the engine (B10d)."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        seen: dict[str, float] = {}
        last_beat = 0.0
        while not self.daemon.stop.is_set():
            for j in self.daemon.jobs():
                jid, upd = j.get("job_id"), float(j.get("updated") or 0)
                if jid and seen.get(jid) != upd:
                    seen[jid] = upd
                    self.wfile.write(f"event: job\ndata: {json.dumps(j)}\n\n".encode())
                    self.wfile.flush()
            now = time.time()
            if now - last_beat > 5:
                self.wfile.write(b": hb\n\n")
                self.wfile.flush()
                last_beat = now
            time.sleep(EVENT_POLL_S)

    def static(self, path: str) -> None:
        rel = "index.html" if path == "/" else path[len("/app/"):] if path.startswith("/app/") else path.lstrip("/")
        f = (APP_DIR / rel).resolve()
        if not str(f).startswith(str(APP_DIR.resolve())) or not f.is_file():
            raise FileNotFoundError(f"no app file {rel!r} (b-app has not landed it yet)")
        ctypes = {".html": "text/html", ".js": "text/javascript", ".css": "text/css",
                  ".json": "application/json", ".svg": "image/svg+xml", ".png": "image/png"}
        self._send(200, f.read_bytes(), ctypes.get(f.suffix, "application/octet-stream"))


class UnixHTTPServer(ThreadingHTTPServer):
    address_family = socket.AF_UNIX
    daemon_threads = True

    def server_bind(self) -> None:
        socketserver.TCPServer.server_bind(self)  # HTTPServer's getfqdn() chokes on a path
        self.server_name, self.server_port = "unix", 0

    def get_request(self):
        conn, _ = self.socket.accept()
        return conn, ("unix", 0)


class LoopbackHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


# ---------------------------------------------------------------- lifecycle


def rss_mb(pid: int | None = None) -> float:
    """This process's current RSS in MB — stdlib only (no psutil; ADR-7)."""
    pid = pid or os.getpid()
    try:
        with open(f"/proc/{pid}/status") as f:  # Linux (the primary target)
            for ln in f:
                if ln.startswith("VmRSS:"):
                    return int(ln.split()[1]) / 1024
    except OSError:
        pass
    try:  # BSD/macOS dev box
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True,
                             text=True, timeout=5).stdout.strip()
        return int(out) / 1024 if out else 0.0
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0.0


def _memory_watch(d: "Daemon", ttl: float, watermark_mb: float, period: float = 0.0) -> None:
    """ADR-5 (revised on measured RSS): the DAEMON stays up forever; only the MODEL is
    evicted — on idle TTL or when RSS crosses the watermark. Never immich's idle-exit."""
    period = period or (min(ttl / 4, 30.0) if ttl > 0 else 30.0)
    while not d.stop.wait(period):
        s = d.searcher
        if s._backend is None:
            continue
        idle = ttl > 0 and s.last_query and time.time() - s.last_query > ttl
        heavy = watermark_mb > 0 and rss_mb() > watermark_mb
        if idle or heavy:
            before = rss_mb()
            if s.release_text():
                d.evictions.append({"at": time.time(), "reason": "idle" if idle else "watermark",
                                    "rss_before_mb": round(before, 1), "rss_after_mb": round(rss_mb(), 1)})
                del d.evictions[:-20]


def serve(home: Path | None = None, tcp: int | None = None, backend: str | None = None,
          log=print, text_ttl: float = 0.0, watermark_mb: float = DEFAULT_WATERMARK_MB) -> int:
    """Run the daemon. Returns a process exit code (3 = another instance holds the lock)."""
    home = home or imgtag_home()
    home.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(home, 0o700)  # B22: ~/.imgtag is 0700 even if it predates us
    sock_p, lock_p, rec_p = daemon_paths(home)
    if len(str(sock_p).encode()) >= 100:  # AF_UNIX sun_path is 104 (BSD) / 108 (Linux) bytes
        raise ValueError(
            f"socket path is too long for AF_UNIX ({len(str(sock_p))} chars): {sock_p} "
            "— point IMGTAG_HOME at a shorter directory"
        )

    lock_fd = os.open(lock_p, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(lock_fd)
        log(f"another imgtag daemon already holds {lock_p}")
        return 3
    os.ftruncate(lock_fd, 0)
    os.write(lock_fd, json.dumps({"pid": os.getpid(), "since": time.time()}).encode())

    # --tcp is STICKY: the port lives in daemon.json (ADR-13's endpoint record), so a
    # flag-free restart keeps the browser app alive. `--tcp 0` disables, `--tcp N` overrides
    # and persists. Three outages came from a lane restarting without the flag.
    if tcp is None:
        try:
            tcp = json.loads(rec_p.read_bytes()).get("http_port")
        except (OSError, ValueError):
            tcp = None
    if tcp == 0:
        tcp = None
    if sock_p.exists():  # ADR-13: only the flock holder may unlink the socket
        sock_p.unlink()
    d = Daemon(home, backend, text_ttl, watermark_mb)
    Handler.daemon = d
    servers = []
    if tcp is not None:
        servers.append(LoopbackHTTPServer(("127.0.0.1", tcp), Handler))  # loopback ONLY (B22)
    # Bind on a side path and rename into place: bind() creates the socket file BEFORE
    # listen(), so a client polling for daemon.sock could connect in that window and take
    # an ECONNREFUSED (observed once in the test suite). Renaming publishes an already
    # listening socket — anything that sees the path can talk to it.
    staging = sock_p.with_name(f".{sock_p.name}.{os.getpid()}")
    staging.unlink(missing_ok=True)
    umask = os.umask(0o177)  # socket is created 0600
    try:
        unix = UnixHTTPServer(str(staging), Handler)
    finally:
        os.umask(umask)
    servers.append(unix)
    port = servers[0].server_address[1] if tcp is not None else None

    # daemon.json is "the endpoint record every door reads" (ADR-13) — it must exist BEFORE
    # the socket is reachable, or a client that polls the socket path wins the race and
    # reads a missing record.
    rec = {"pid": os.getpid(), "version": VERSION, "socket": str(sock_p), "http_port": port,
           "running": True, "started_at": d.started, "models": d.model_shas()}
    tmp = rec_p.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1))
    os.chmod(tmp, 0o600)
    os.replace(tmp, rec_p)
    os.rename(staging, sock_p)  # publish last: whoever sees the socket sees a ready daemon
    unix.server_address = str(sock_p)
    log(f"imgtag daemon {VERSION} pid={os.getpid()} socket={sock_p}"
        + (f" http=127.0.0.1:{port}" if port else ""))

    if text_ttl > 0 or watermark_mb > 0:
        threading.Thread(target=_memory_watch, args=(d, text_ttl, watermark_mb), daemon=True).start()
    threads = [threading.Thread(target=s.serve_forever, kwargs={"poll_interval": 0.2}, daemon=True)
               for s in servers]
    for t in threads:
        t.start()
    try:
        while any(t.is_alive() for t in threads) and not d.stop.is_set():
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        d.stop.set()
        for s in servers:
            s.shutdown()
            s.server_close()
        try:
            sock_p.unlink()  # ADR-13: the flock holder owns the socket, live or dead
        except OSError:
            pass
        # The record SURVIVES as a stopped record: it is what makes --tcp sticky across a
        # flag-free restart. `running: false` + no socket is how a door knows it is dead.
        try:
            rec_p.write_text(json.dumps({**rec, "pid": None, "running": False,
                                         "stopped_at": time.time()}, indent=1))
            os.chmod(rec_p, 0o600)
        except OSError:
            pass
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
    return 0


# ---------------------------------------------------------------- client


class UnixHTTPConnection(http.client.HTTPConnection):
    """http.client over an AF_UNIX socket — what every door (CLI, skill, tests) uses."""

    def __init__(self, path: str, timeout: float = 30.0):
        super().__init__("localhost", timeout=timeout)
        self.unix_path = path

    def connect(self) -> None:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(self.timeout)
        s.connect(self.unix_path)
        self.sock = s


def request(method: str, path: str, body=None, home: Path | None = None, timeout: float = 30.0):
    """One request against the running daemon. Returns (status, parsed-body)."""
    sock_p, _, _ = daemon_paths(home)
    c = UnixHTTPConnection(str(sock_p), timeout)
    try:
        payload = json.dumps(body).encode() if body is not None else None
        c.request(method, path, payload, {"Content-Type": "application/json"} if payload else {})
        r = c.getresponse()
        raw = r.read()
        ctype = r.getheader("Content-Type", "")
        return r.status, (json.loads(raw) if ctype.startswith("application/json") else raw)
    finally:
        c.close()


def ensure_daemon(home: Path | None = None, timeout: float = 2.0, backend: str | None = None) -> bool:
    """ADR-13 client algorithm: connect, else take the lock and spawn, else wait for the
    other client's daemon. Returns False on timeout — the caller falls back in-process."""
    home = home or imgtag_home()
    sock_p, lock_p, _ = daemon_paths(home)

    def connect() -> bool:
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(0.5)
            s.connect(str(sock_p))
            s.close()
            return True
        except OSError:
            return False

    if connect():
        return True
    home.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(lock_p, os.O_RDWR | os.O_CREAT, 0o600)
    spawned = False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if sock_p.exists():  # stale socket: its owner is dead, or we would not hold the lock
            sock_p.unlink()
        fcntl.flock(fd, fcntl.LOCK_UN)
        cmd = [sys.executable, "-m", "imgtag.daemon"] + (["--backend", backend] if backend else [])
        subprocess.Popen(cmd, env={**os.environ, "IMGTAG_HOME": str(home)},
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        spawned = True
    except OSError:
        pass  # another client is starting one — just wait for it
    finally:
        os.close(fd)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if connect():
            return True
        time.sleep(0.025)
    return spawned and connect()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("imgtag-daemon", description="resident imgtag search daemon")
    ap.add_argument("--tcp", type=int, metavar="PORT",
                    help="also bind 127.0.0.1:PORT (opt-in, loopback only) for the web app. "
                         "STICKY: persisted in daemon.json and inherited by a flag-free "
                         "restart; --tcp 0 turns it off.")
    ap.add_argument("--home", type=Path, default=None, help="override ~/.imgtag")
    ap.add_argument("--backend", default=None, help="preload this model backend by name")
    ap.add_argument("--idle-timeout", type=float, default=0.0,
                    help="0 = never exit (ADR-13 default; model_ttl is the proven anti-pattern)")
    ap.add_argument("--rss-watermark-mb", type=float, default=DEFAULT_WATERMARK_MB, metavar="MB",
                    help="evict the model when RSS crosses this (0 = never). Default keeps us "
                         "under B8's 1.5GB under-load cap without ever unloading on a timer.")
    ap.add_argument("--text-ttl", type=float, default=0.0, metavar="SECONDS",
                    help="release the text tower after SECONDS idle (ADR-5 revised: 0 = never "
                         "on desktop, 300 on the 8GB server). The daemon stays up either way.")
    a = ap.parse_args(argv)
    if a.idle_timeout:
        print("--idle-timeout is accepted but ignored: ADR-13 pins the default at 0", file=sys.stderr)
    return serve(a.home, a.tcp, a.backend, text_ttl=a.text_ttl, watermark_mb=a.rss_watermark_mb)


if __name__ == "__main__":
    raise SystemExit(main())
