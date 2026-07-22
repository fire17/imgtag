"""Shard/manifest storage — ADR-6 is the law; contracts in .deify/wave-b-briefs.md.

OWNER: b-engine. Consumers (b-daemon, b-bench) code against open_snapshot()/Writer only.

Layout (``~/.imgtag/datasets/<slug>/``)::

    shard-<jobid8>-<seq:04d>.f32   append-only f32 [rows, dim]
    ids-<jobid8>-<seq:04d>.jsonl   line i  <->  row i of the shard (WRITTEN INVARIANT)
    manifest.json                  atomic (tmp+fsync+rename+dir fsync); byte counts authoritative
    .writer.lock                   fcntl flock(LOCK_EX|LOCK_NB) held for the whole job
    trash/                         superseded / orphan shards; never unlinked inline

Readers never stat() a shard: every read is capped at the manifest's byte counts.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import signal
import threading
import time
import uuid
from pathlib import Path

import numpy as np

MANIFEST_VERSION = 1
FLUSH_ROWS = 500  # ADR-6 flush cadence: pending >= N rows ...
FLUSH_SECONDS = 1.5  # ... or T elapsed with pending > 0 (inside B11's 2s visibility)


class LockedError(RuntimeError):
    """Another writer holds the dataset lock (CLI exit 3)."""


class UnknownDatasetError(RuntimeError):
    """Dataset has no manifest (CLI exit 4)."""


class ModelMismatchError(RuntimeError):
    """Manifest model_sha != loaded model (CLI exit 5)."""


class CorruptIndexError(RuntimeError):
    """Shard shorter than its manifest byte count, or byte counts inconsistent (CLI exit 6)."""


def imgtag_home() -> Path:
    return Path(os.environ.get("IMGTAG_HOME", str(Path.home() / ".imgtag")))


def dataset_dir(dataset: str, home: Path | None = None) -> Path:
    return (home or imgtag_home()) / "datasets" / dataset


def list_datasets(home: Path | None = None) -> list[str]:
    root = (home or imgtag_home()) / "datasets"
    if not root.is_dir():
        return []
    return sorted(d.name for d in root.iterdir() if (d / "manifest.json").is_file())


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _fsync_dir(d: Path) -> None:
    fd = os.open(d, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def writer_lock_free(dataset: str, home: Path | None = None) -> bool:
    """True when NO writer holds this dataset — the kernel's answer, not a heuristic.

    ADR-6 makes the flock the sole authority on liveness: a job status file frozen at
    "queued" by a killed process is a corpse, and anything that trusts it (delete, info,
    recovery) inherits a lie. Acquiring the lock for an instant is the only honest test.
    """
    p = dataset_dir(dataset, home) / ".writer.lock"
    if not p.exists():
        return True
    try:
        fd = os.open(p, os.O_RDWR)
    except OSError:
        return True
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return True
    except OSError:
        return False
    finally:
        os.close(fd)


def read_manifest(dataset: str, home: Path | None = None) -> dict:
    p = dataset_dir(dataset, home) / "manifest.json"
    try:
        return json.loads(p.read_bytes())
    except FileNotFoundError:
        raise UnknownDatasetError(f"no index for dataset {dataset!r} ({p})") from None


# ---------------------------------------------------------------- track sidecars
#
# ADR-15/T1: every track scores EVERY image. Scores live in a dense f32 sidecar per
# track per dataset, ROW-ALIGNED to the shards (row i of `tracks/<cat>.f32` is row i of
# the concatenated shards), and RAW SCORES ARE WHAT IS STORED — tiers are derived at read
# from a versioned spec, so a policy change costs a re-derivation, never a re-scan.
# NaN means "not scored" (a head that failed a batch), which is an honest answer and
# distinguishable from a real 0.0.

TRACKS_DIR = "tracks"
NOT_SCORED = np.float32("nan")


def track_path(dataset: str, category: str, home: Path | None = None) -> Path:
    return dataset_dir(dataset, home) / TRACKS_DIR / f"{category}.f32"


def read_track(dataset: str, category: str, home: Path | None = None, manifest: dict | None = None):
    """Memmap one track column. Capped at the manifest's row count — never stat()."""
    man = manifest or read_manifest(dataset, home)
    rec = (man.get("tracks") or {}).get(category)
    if not rec or not rec["rows"]:
        return None
    p = track_path(dataset, category, home)
    cols = rec.get("cols", 1)
    shape = (rec["rows"],) if cols == 1 else (rec["rows"], cols)
    return np.memmap(p, np.float32, "r", shape=shape)


def dataset_flags(dataset: str, home: Path | None = None, snap=None) -> dict:
    """Derive the per-image tier view from the stored SCORE sidecars (ADR-15 T1).

    Flags are no longer stored — they are computed here from raw scores and the tier
    spec recorded in the manifest, so a threshold change re-derives for free. Returns
    ``{category: {"scores": memmap, "tiers": [...]}}`` for every scored track.
    """
    snap = snap or open_snapshot(dataset, home)
    specs = ((snap.manifest.get("tracks_spec") or {}).get("tiers")) or {}
    out = {}
    for cat, col in snap.tracks.items():
        if col is None:
            continue
        out[cat] = {"scores": col, "tiers": derive_tiers(col, specs.get(cat, {}))}
    return out


def derive_tiers(scores, spec: dict) -> list[str]:
    """Raw scores + a track spec -> ADR-14/15 tier labels. Pure, deterministic, and the
    ONE place the mapping lives, so daemon, CLI and bench cannot disagree (B25d).

    Bands are read from the spec, highest first; a score below every band is "none",
    and NaN (not scored) is "unknown" rather than a silently-passing "none".
    """
    bands = [(t, float(spec[k])) for t, k in
             (("alert", "tau_alert"), ("violation", "tau_violation"), ("review", "tau_review"))
             if spec.get(k) is not None]
    if not bands and spec.get("tau") is not None:  # single-threshold spec
        bands = [("violation", float(spec["tau"]))]
    bands.sort(key=lambda b: -b[1])
    out = []
    for v in np.asarray(scores, np.float32).reshape(-1):
        if np.isnan(v):
            out.append("unknown")
            continue
        out.append(next((t for t, tau in bands if v >= tau), "none"))
    return out


def _ids_name(shard_name: str) -> str:
    return "ids-" + shard_name[len("shard-") : -len(".f32")] + ".jsonl"


# ---------------------------------------------------------------- snapshot


class ShardArray:
    """Read-only [N, D] f32 view over the mmap'd shards named in one manifest.

    Duck-compatible with the ``np.memmap`` the contract names (``.shape``, ``.dtype``,
    ``x @ q``, ``x[i]``, ``np.asarray(x)``) while keeping every shard file-backed and
    page-evictable — concatenating into one heap array would cost 205MB at 100k rows
    and forfeit exactly the property ADR-2 buys.
    """

    def __init__(self, mms: list[np.memmap], dim: int):
        self._mms = mms
        self._offs: list[int] = []
        n = 0
        for m in mms:
            self._offs.append(n)
            n += len(m)
        self.shape = (n, dim)
        self.dtype = np.dtype(np.float32)

    def __len__(self) -> int:
        return self.shape[0]

    def __matmul__(self, q):
        if not self._mms:
            return np.zeros((0,) + np.shape(q)[1:], np.float32)
        if len(self._mms) == 1:
            return self._mms[0] @ q
        return np.concatenate([m @ q for m in self._mms])

    def __getitem__(self, i):
        if isinstance(i, slice) or not np.isscalar(i):
            return np.asarray(self)[i]
        if i < 0:
            i += self.shape[0]
        for off, m in zip(self._offs, self._mms):
            if i < off + len(m):
                return m[i - off]
        raise IndexError(i)

    def __array__(self, dtype=None, copy=None):
        a = np.concatenate(self._mms) if self._mms else np.zeros(self.shape, np.float32)
        return a.astype(dtype) if dtype else a


class Snapshot:
    """Point-in-time view: one manifest read, every shard opened EAGERLY (ADR-6)."""

    def __init__(self, dataset: str, home: Path | None = None):
        self.dataset = dataset
        self.home = home
        self.dir = dataset_dir(dataset, home)
        self.manifest = self._open_all(retry=True)

    def _open_all(self, retry: bool) -> dict:
        man = json.loads((self.dir / "manifest.json").read_bytes())
        dim = man["dim"]
        mms: list[np.memmap] = []
        ids: list[dict] = []
        try:
            for s in man["shards"]:
                if s["rows"]:
                    mms.append(np.memmap(self.dir / s["name"], np.float32, "r", shape=(s["rows"], dim)))
                with open(self.dir / _ids_name(s["name"]), "rb") as f:
                    blob = f.read(s["ids_bytes"])  # capped at the manifest count; never stat()
                ids.extend(json.loads(ln) for ln in blob.splitlines() if ln.strip())
        except FileNotFoundError:
            if retry:  # compaction may have swapped the manifest under us — re-read ONCE
                return self._open_all(retry=False)
            raise
        self.emb = ShardArray(mms, dim)
        self.ids = ids
        self.tracks = {c: read_track(self.dataset, c, self.home, man)
                       for c in (man.get("tracks") or {})}
        return man

    @property
    def count(self) -> int:
        return int(self.manifest["count"])


def open_snapshot(dataset: str, home: Path | None = None) -> Snapshot:
    try:
        return Snapshot(dataset, home)
    except FileNotFoundError:
        raise UnknownDatasetError(f"no index for dataset {dataset!r}") from None


# ---------------------------------------------------------------- writer


class Writer:
    """Exclusive, durable appender. Context manager; the flock holder is the SOLE
    manifest writer (decode/embed workers never touch manifest.json)."""

    def __init__(self, dataset: str, model, home: Path | None = None, job_id: str | None = None):
        self.dataset = dataset
        self.model = model
        self.home = home
        self.dir = dataset_dir(dataset, home)
        if job_id and not re.fullmatch(r"[0-9A-Za-z_-]{1,32}", job_id):
            # the id becomes a FILENAME (shard-<id>-0000.f32) — never let a caller's
            # string reach the filesystem unchecked
            raise ValueError(f"invalid job_id {job_id!r}: expected [0-9A-Za-z_-]{{1,32}}")
        self.job_id = job_id or uuid.uuid4().hex[:8]
        self.name = f"shard-{self.job_id}-0000.f32"
        self._buf_e: list[np.ndarray] = []
        self._buf_r: list[dict] = []
        self._buf_t: dict[str, list[np.ndarray]] = {}
        self._ids: set[str] = set()
        self._cv = threading.Condition()
        self._stop = False
        self._err: BaseException | None = None
        self._lock_fd: int | None = None
        self._thread: threading.Thread | None = None
        self.flushes = 0

    # -- lifecycle -------------------------------------------------
    def __enter__(self) -> "Writer":
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "trash").mkdir(exist_ok=True)
        self._acquire()
        self.manifest = self._load_or_init()
        if self.manifest["count"]:  # content-addressed ids are unique per dataset (IA.md)
            try:
                self._ids = {r["image_id"] for r in Snapshot(self.dataset, self.home).ids}
            except (OSError, ValueError, KeyError):
                self._ids = set()
        self.recovery = recover(self.dir, self.manifest)  # torn tails + orphans, write-open only
        # We hold the exclusive lock, so every OTHER queued/running record for this
        # dataset is a corpse by definition (ADR-6: the flock is the authority).
        from .progress import reap_stale

        for jid in reap_stale(self.dataset, self.home, keep=self.job_id):
            self.recovery.append(f"closed stale job record {jid}")
        self._commit()
        self._install_signals()
        self._thread = threading.Thread(target=self._loop, name="imgtag-flusher", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        if self._thread:
            self._thread.join(timeout=60)
        self._flush_pending()  # belt & braces: nothing may stay in RAM
        self._release()
        if self._err and exc[0] is None:
            raise self._err

    def _acquire(self) -> None:
        path = self.dir / ".writer.lock"
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            try:
                held = path.read_text()[:200]
            except OSError:
                held = ""
            os.close(fd)
            raise LockedError(f"dataset {self.dataset!r} is locked by another writer: {held}") from None
        os.ftruncate(fd, 0)
        os.write(fd, json.dumps({"pid": os.getpid(), "job": self.job_id, "since": _now()}).encode())
        os.fsync(fd)
        self._lock_fd = fd

    def _release(self) -> None:
        if self._lock_fd is not None:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            os.close(self._lock_fd)
            self._lock_fd = None

    def _install_signals(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        for sig in (signal.SIGTERM, signal.SIGINT):
            prev = signal.getsignal(sig)

            def handler(signum, frame, _prev=prev):
                self._flush_pending()
                if callable(_prev):
                    _prev(signum, frame)
                else:
                    raise SystemExit(128 + signum)

            try:
                signal.signal(sig, handler)
            except ValueError:  # pragma: no cover
                pass

    def _load_or_init(self) -> dict:
        p = self.dir / "manifest.json"
        if p.is_file():
            man = json.loads(p.read_bytes())
            if man.get("model_sha") != self.model.model_sha:
                raise ModelMismatchError(
                    f"index built with {man.get('model_id')} ({man.get('model_sha')!r}); "
                    f"loaded {self.model.model_id} ({self.model.model_sha!r}) — reindex or switch model"
                )
            return man
        return {
            "version": MANIFEST_VERSION,
            "dataset": self.dataset,
            "model_id": self.model.model_id,
            "model_sha": self.model.model_sha,
            "dim": int(self.model.dim),
            "count": 0,
            "shards": [],
            "created": _now(),
            "updated": _now(),
        }

    # -- append / flush --------------------------------------------
    def append(self, embs: np.ndarray, recs: list[dict], tracks: dict | None = None) -> None:
        """Append rows. ``tracks`` maps category -> f32 [n] (or [n, cols]) of RAW scores,
        written to the row-aligned sidecars (ADR-15 T1)."""
        if self._err:
            raise self._err
        embs = np.ascontiguousarray(embs, np.float32)
        if embs.shape[0] != len(recs):
            raise ValueError("embs/recs length mismatch")
        if embs.shape[1] != self.manifest["dim"]:
            raise ValueError(f"dim mismatch: {embs.shape[1]} != {self.manifest['dim']}")
        ids = {r["image_id"] for r in recs}
        if len(ids) != len(recs):
            raise ValueError("duplicate image_id within one append batch")
        if ids & self._ids:
            raise ValueError(f"duplicate image_id already in {self.dataset!r}: "
                             f"{sorted(ids & self._ids)[:3]} — ids are content-addressed (IA.md)")
        self._ids |= ids
        with self._cv:
            self._buf_e.append(embs)
            self._buf_r.extend(recs)
            for cat, col in (tracks or {}).items():
                col = np.asarray(col, np.float32).reshape(len(recs), -1)
                self._buf_t.setdefault(cat, []).append(col)
            if len(self._buf_r) >= FLUSH_ROWS:
                self._cv.notify_all()

    @property
    def count(self) -> int:
        """Durable manifest count — the ONLY progress authority (ADR-6)."""
        return int(self.manifest["count"])

    @property
    def pending(self) -> int:
        with self._cv:
            return len(self._buf_r)

    def _loop(self) -> None:
        while True:
            with self._cv:
                if not self._stop and len(self._buf_r) < FLUSH_ROWS:
                    self._cv.wait(timeout=FLUSH_SECONDS)
                stop = self._stop
            try:
                self._flush_pending()
            except BaseException as e:  # fail loud, stop accepting rows
                self._err = e
                with self._cv:
                    self._stop = True
                return
            if stop:
                return

    def _flush_pending(self) -> None:
        with self._cv:
            embs, recs, tracks = self._buf_e, self._buf_r, self._buf_t
            self._buf_e, self._buf_r, self._buf_t = [], [], {}
        if not recs:
            return
        self._write(np.concatenate(embs), recs,
                    {c: np.concatenate(v) for c, v in tracks.items()})

    def _write(self, embs: np.ndarray, recs: list[dict], tracks: dict | None = None) -> None:
        rec = next((s for s in self.manifest["shards"] if s["name"] == self.name), None)
        if rec is None:
            rec = {"name": self.name, "rows": 0, "emb_bytes": 0, "ids_bytes": 0}
            self.manifest["shards"].append(rec)
        base = self.manifest["count"]
        lines = b"".join(
            json.dumps({**r, "row": base + i}, separators=(",", ":")).encode() + b"\n"
            for i, r in enumerate(recs)
        )
        blob = embs.tobytes()
        # (1) buffered write to shard  (2) fsync
        with open(self.dir / self.name, "ab", buffering=1024 * 1024) as f:
            f.write(blob)
            f.flush()
            os.fsync(f.fileno())
        # (3) ids lines  (4) fsync
        with open(self.dir / _ids_name(self.name), "ab", buffering=1024 * 1024) as f:
            f.write(lines)
            f.flush()
            os.fsync(f.fileno())
        # (3b) track sidecars — same durability discipline, before the manifest names them
        for cat, col in (tracks or {}).items():
            self._write_track(cat, col, base, len(recs))
        rec["rows"] += len(recs)
        rec["emb_bytes"] += len(blob)
        rec["ids_bytes"] += len(lines)
        self.manifest["count"] = base + len(recs)
        # (5) manifest.tmp + fsync  (6) rename  (7) fsync dirfd
        self._commit()
        self.flushes += 1

    def _write_track(self, category: str, col: np.ndarray, base: int, n: int) -> None:
        """Append one track column, padding any gap so row alignment can never drift."""
        t = self.manifest.setdefault("tracks", {}).setdefault(
            category, {"name": f"{category}.f32", "rows": 0, "cols": int(col.shape[1]), "bytes": 0})
        d = self.dir / TRACKS_DIR
        d.mkdir(exist_ok=True)
        gap = base - t["rows"]
        if gap > 0:  # rows written before this track existed are "not scored", not 0.0
            pad = np.full((gap, t["cols"]), NOT_SCORED, np.float32)
            col = np.concatenate([pad, col])
        with open(d / t["name"], "ab", buffering=1024 * 1024) as f:
            f.write(col.tobytes())
            f.flush()
            os.fsync(f.fileno())
        t["rows"] = base + n
        t["bytes"] = t["rows"] * t["cols"] * 4

    def _commit(self) -> None:
        self.manifest["updated"] = _now()
        tmp = self.dir / "manifest.json.tmp"
        with open(tmp, "wb") as f:
            f.write(json.dumps(self.manifest, indent=1).encode())
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.dir / "manifest.json")
        _fsync_dir(self.dir)


# ---------------------------------------------------------------- recovery


def recover(d: Path, manifest: dict, log=None) -> list[str]:
    """Open-for-WRITE recovery (never on read). Truncates torn tails, fails loud on
    short files, moves orphans to trash/. Returns the actions taken."""
    say = log or (lambda m: None)
    actions: list[str] = []
    dim = manifest["dim"]
    known: set[str] = set()
    total = 0
    for s in manifest["shards"]:
        if s["emb_bytes"] % (dim * 4) or s["emb_bytes"] // (dim * 4) != s["rows"]:
            raise CorruptIndexError(
                f"{s['name']}: emb_bytes={s['emb_bytes']} inconsistent with rows={s['rows']} dim={dim}"
            )
        total += s["rows"]
        for fn, key in ((s["name"], "emb_bytes"), (_ids_name(s["name"]), "ids_bytes")):
            known.add(fn)
            p = d / fn
            want = s[key]
            have = p.stat().st_size if p.exists() else -1
            if have < 0 and want == 0:
                continue
            if have < want:
                q = d / "trash" / f"SHORT-{fn}"
                if p.exists():
                    p.rename(q)
                raise CorruptIndexError(
                    f"{fn}: {have} bytes on disk < {want} in manifest — quarantined to {q}"
                )
            if have > want:
                os.truncate(p, want)
                actions.append(f"truncated torn tail {fn}: {have} -> {want} bytes")
                say(actions[-1])
    for cat, t in (manifest.get("tracks") or {}).items():
        p = d / TRACKS_DIR / t["name"]
        want = t["rows"] * t.get("cols", 1) * 4
        have = p.stat().st_size if p.exists() else -1
        if have < 0 and want == 0:
            continue
        if have < want:
            raise CorruptIndexError(f"track {cat}: {have} bytes on disk < {want} in manifest")
        if have > want:
            os.truncate(p, want)
            actions.append(f"truncated torn tail tracks/{t['name']}: {have} -> {want} bytes")
            say(actions[-1])
    if total != manifest["count"]:
        raise CorruptIndexError(f"manifest count {manifest['count']} != sum(shard rows) {total}")
    for p in d.iterdir():
        if p.is_file() and (p.name.startswith(("shard-", "ids-"))) and p.name not in known:
            p.rename(d / "trash" / p.name)
            actions.append(f"orphan {p.name} -> trash/")
            say(actions[-1])
    return actions
