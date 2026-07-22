"""Resident search daemon — ADR-5 (warm towers) + ADR-13 (lifecycle), stdlib only.

OWNER: b-daemon. ``ThreadingHTTPServer`` over a per-user UNIX socket
(``~/.imgtag/daemon.sock``, 0600); ``--tcp PORT`` is the app's opt-in loopback door and
refuses any non-loopback bind. Single instance via ``fcntl.flock`` on
``~/.imgtag/daemon.lock`` held for the process lifetime — the same kernel-owned pattern as
the index writer (ADR-6), never pid heuristics. The endpoint record is ``daemon.json``.

Endpoints::

    GET  /api/hello                              {version, model_shas, pid, socket, uptime_s}
    GET  /api/search?q=&dataset=&k=&strict=      Search API contract (briefs §Search API)
    GET  /api/datasets                           fleet view (B18f: exactly what is on disk)
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
from .core.progress import Job, list_jobs
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

    def __init__(self, home: Path, backend: str | None = None):
        self.home = home
        self.started = time.time()
        self.searcher = Searcher(home, backend=_models.load_backend(backend) if backend else None)
        self.thumbs = ThumbCache(home)
        self.stop = threading.Event()

    def model_shas(self) -> dict:
        b = self.searcher._backend
        return {b.model_id: b.model_sha} if b else {}

    def datasets(self) -> list[dict]:
        """Fleet view: exactly the datasets on disk, counts straight from their manifests."""
        out = []
        for name in list_datasets(self.home):
            try:
                m = read_manifest(name, self.home)
            except UnknownDatasetError:
                continue
            out.append(
                {
                    "dataset": name,
                    "count": m.get("count", 0),
                    "model_id": m.get("model_id"),
                    "model_sha": m.get("model_sha"),
                    "dim": m.get("dim"),
                    "shards": len(m.get("shards", [])),
                    "calibrated": bool(m.get("calib_sha")),
                    "created": m.get("created"),
                    "updated": m.get("updated"),
                }
            )
        return out

    def jobs(self) -> list[dict]:
        """Job list with mirrored engine records folded away — one row per real job."""
        js = list_jobs(self.home)
        mirrored = {j.get("engine_job_id") for j in js if j.get("engine_job_id")}
        return [j for j in js if j.get("job_id") not in mirrored]

    def record(self, ids, dataset: str, image_id: str) -> dict:
        for rec in ids:
            if rec["image_id"] == image_id:
                return rec
        raise FileNotFoundError(f"image {image_id!r} not in dataset {dataset!r}")

    def start_index(self, path: str, dataset: str) -> dict:
        """Spawn `imgtag index` as a subprocess and return a job id (<=500ms, B20).

        The engine mints its own job id only once its Writer exists (seconds later, after
        the model loads), so the daemon opens the job record itself and a watcher mirrors
        the engine's numbers into it. One record per job for the client either way.
        (Deletable the day `imgtag index` accepts `--job-id`; that is b-engine's file.)
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
        t0 = time.time()
        proc = subprocess.Popen(
            [sys.executable, "-m", "imgtag.cli", "--json", "index", str(src), dataset],
            env={**os.environ, "IMGTAG_HOME": str(self.home)},
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, start_new_session=True,
        )
        threading.Thread(target=self._mirror, args=(proc, job, dataset, t0), daemon=True).start()
        return {"job_id": job_id, "dataset": dataset, "path": str(src), "pid": proc.pid}

    def _mirror(self, proc, job: Job, dataset: str, t0: float) -> None:
        """Copy the engine job's durable counts into the daemon's record until it exits."""
        job.start()
        eng_id = None
        while True:
            for j in list_jobs(self.home):
                if eng_id is None and j.get("dataset") == dataset and j.get("job_id") != job.state["job_id"] \
                        and float(j.get("started") or 0) >= t0 - 1:
                    eng_id = j["job_id"]
                if j.get("job_id") == eng_id:
                    job.state.update(total=j.get("total", 0), failed=j.get("failed", 0),
                                     engine_job_id=eng_id, stages_ms=j.get("stages_ms", {}))
                    job.update(int(j.get("done", 0)), int(j.get("inflight", 0)), force=True)
            if proc.poll() is not None:
                break
            time.sleep(0.5)
        err = (proc.stderr.read() or b"").decode()[-500:] if proc.stderr else ""
        if proc.returncode == 0:
            job.finish(engine_job_id=eng_id)
        else:
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
            if u.path == "/api/hello":
                self.json({"version": VERSION, "model_shas": d.model_shas(), "pid": os.getpid(),
                           "socket": str(daemon_paths(d.home)[0]), "uptime_s": round(time.time() - d.started, 1)})
            elif u.path == "/api/search":
                query = (q.get("q") or [""])[0]
                if not query.strip():
                    raise ValueError("q is required")
                self.json(d.searcher.search(
                    query,
                    dataset=(q.get("dataset") or [None])[0] or None,
                    k=max(1, min(500, int((q.get("k") or [50])[0]))),
                    strict=(q.get("strict") or ["0"])[0] in ("1", "true", "yes"),
                ))
            elif u.path == "/api/datasets":
                self.json({"datasets": d.datasets()})
            elif u.path == "/api/jobs":
                self.json({"jobs": d.jobs()})
            elif u.path == "/api/events":
                self.events()
            elif u.path.startswith("/api/thumb/"):
                self.thumb(u.path, q)
            elif u.path in ("/", "/index.html") or u.path.startswith("/app/"):
                self.static(u.path)
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
                self.json(self.daemon.start_index(str(body["path"]), str(body["dataset"])), 202)
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
        rel = "index.html" if path in ("/", "/index.html") else path[len("/app/"):]
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


def _text_ttl_watch(d: "Daemon", ttl: float) -> None:
    """ADR-5 (revised on measured RSS): the DAEMON stays up; only the text tower is
    evicted after `ttl` idle seconds. Never an idle-exit — that is immich's sin."""
    while not d.stop.wait(min(ttl / 4, 30.0)):
        be = d.searcher._backend
        if be is not None and d.searcher.last_query and time.time() - d.searcher.last_query > ttl:
            be.release_text()
            d.searcher.last_query = 0.0


def serve(home: Path | None = None, tcp: int | None = None, backend: str | None = None,
          log=print, text_ttl: float = 0.0) -> int:
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

    if sock_p.exists():  # ADR-13: only the flock holder may unlink the socket
        sock_p.unlink()
    d = Daemon(home, backend)
    Handler.daemon = d
    servers = []
    if tcp is not None:
        servers.append(LoopbackHTTPServer(("127.0.0.1", tcp), Handler))  # loopback ONLY (B22)
    umask = os.umask(0o177)  # socket is created 0600
    try:
        unix = UnixHTTPServer(str(sock_p), Handler)
    finally:
        os.umask(umask)
    servers.append(unix)
    port = servers[0].server_address[1] if tcp is not None else None

    rec = {"pid": os.getpid(), "version": VERSION, "socket": str(sock_p), "http_port": port,
           "started_at": d.started, "models": d.model_shas()}
    tmp = rec_p.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1))
    os.chmod(tmp, 0o600)
    os.replace(tmp, rec_p)
    log(f"imgtag daemon {VERSION} pid={os.getpid()} socket={sock_p}"
        + (f" http=127.0.0.1:{port}" if port else ""))

    if text_ttl > 0:
        threading.Thread(target=_text_ttl_watch, args=(d, text_ttl), daemon=True).start()
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
        for p in (sock_p, rec_p):
            try:
                p.unlink()
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
                    help="also bind 127.0.0.1:PORT (opt-in, loopback only) for the web app")
    ap.add_argument("--home", type=Path, default=None, help="override ~/.imgtag")
    ap.add_argument("--backend", default=None, help="preload this model backend by name")
    ap.add_argument("--idle-timeout", type=float, default=0.0,
                    help="0 = never exit (ADR-13 default; model_ttl is the proven anti-pattern)")
    ap.add_argument("--text-ttl", type=float, default=0.0, metavar="SECONDS",
                    help="release the text tower after SECONDS idle (ADR-5 revised: 0 = never "
                         "on desktop, 300 on the 8GB server). The daemon stays up either way.")
    a = ap.parse_args(argv)
    if a.idle_timeout:
        print("--idle-timeout is accepted but ignored: ADR-13 pins the default at 0", file=sys.stderr)
    return serve(a.home, a.tcp, a.backend, text_ttl=a.text_ttl)


if __name__ == "__main__":
    raise SystemExit(main())
