"""Job status files — A3 lifecycle, atomic writes, honest progress (B10).

OWNER: b-engine. ``done`` is ALWAYS the durable manifest count (ADR-6 progress
authority); dispatched-but-not-yet-committed rows are reported separately as
``in_flight`` so a crash can never make progress jump backwards.
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from pathlib import Path

from .store import imgtag_home

HEARTBEAT_S = 0.9  # B10(a): max gap between events <= 1.0s, including stalls
RATE_WINDOW_S = 10.0


def jobs_dir(home: Path | None = None) -> Path:
    return (home or imgtag_home()) / "jobs"


def read_job(job_id: str, home: Path | None = None) -> dict | None:
    p = jobs_dir(home) / f"{job_id}.json"
    try:
        return json.loads(p.read_bytes())
    except (OSError, ValueError):
        return None


def list_jobs(home: Path | None = None, limit: int = 50) -> list[dict]:
    d = jobs_dir(home)
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
        try:
            out.append(json.loads(p.read_bytes()))
        except (OSError, ValueError):
            pass
    return out


def abort_path(job_id: str, home: Path | None = None) -> Path:
    return jobs_dir(home) / f"{job_id}.abort"


def request_abort(job_id: str, home: Path | None = None) -> None:
    """Ask a running job to stop. A file, not a signal: the writer may be a detached
    process in another session, and the kernel-owned flock is what protects the data —
    this only needs to be a flag the coordinator polls."""
    p = abort_path(job_id, home)
    p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    p.write_text(str(time.time()))


def clear_abort(job_id: str, home: Path | None = None) -> None:
    abort_path(job_id, home).unlink(missing_ok=True)


def mark_stale(job_id: str, home: Path | None = None, reason: str = "stale: writer lock free") -> bool:
    """Close a job record whose writer is provably gone. Returns True if it changed."""
    p = jobs_dir(home) / f"{job_id}.json"
    try:
        j = json.loads(p.read_bytes())
    except (OSError, ValueError):
        return False
    if j.get("state") not in ("queued", "running"):
        return False
    j.update(state="failed", error=reason, inflight=0, in_flight=0, updated=time.time())
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(j))
    os.replace(tmp, p)
    return True


STARTUP_GRACE_S = 45.0  # a queued job may legitimately spend this long loading a model
                        # before it ever reaches the Writer and takes the lock


def _pid_alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def is_corpse(job: dict, lock_free: bool, grace_s: float = STARTUP_GRACE_S) -> bool:
    """Is this 'live' record actually dead? Kernel truth first, clock last.

    RUNNING means a Writer exists, and a Writer holds the flock — so a free lock proves
    the record is a corpse (ADR-6: the flock is the authority, never the status file).
    QUEUED is different: the job has not reached its Writer yet, so a free lock proves
    nothing during the startup window. There we ask the kernel about the recorded pid,
    and only fall back to the clock when no pid was recorded.
    """
    if job.get("state") not in ("queued", "running"):
        return False
    if not lock_free:
        return False
    if job.get("state") == "running":
        return True
    if job.get("pid") is not None:
        return not _pid_alive(job["pid"])
    return (time.time() - float(job.get("updated") or job.get("started") or 0)) > grace_s


def reap_stale(dataset: str, home: Path | None = None, keep: str | None = None) -> list[str]:
    """Close every record for this dataset that is provably dead. Returns the ids closed."""
    from .store import writer_lock_free

    free = writer_lock_free(dataset, home)
    if not free:
        return []
    out = []
    for j in list_jobs(home, limit=200):
        if j.get("dataset") != dataset or j.get("job_id") == keep:
            continue
        if is_corpse(j, free) and mark_stale(j["job_id"], home):
            out.append(j["job_id"])
    return out


def annotate_stale(jobs: list[dict], home: Path | None = None, older_than_s: float = 60.0) -> list[dict]:
    """Flag records that CLAIM to be live but whose dataset lock is free and which have
    not been touched recently — display truth, so `info` never shows a ghost as running."""
    from .store import writer_lock_free

    free: dict[str, bool] = {}
    for j in jobs:
        if j.get("state") in ("queued", "running"):
            ds = j.get("dataset")
            if ds not in free:
                free[ds] = writer_lock_free(ds, home)
            if is_corpse(j, free[ds], grace_s=older_than_s):
                j["stale"] = True
                j["stale_reason"] = ("writer lock free and its process is gone"
                                     if j.get("pid") else "writer lock free and record is stale")
    return jobs


class Job:
    """One index job's status file. Cheap: writes are throttled and event-driven
    (B10(d): the emitter must cost <=1% of run wall time — no polling thread)."""

    def __init__(self, job_id: str, dataset: str, total: int, home: Path | None = None, **extra):
        self._home = home
        self.path = jobs_dir(home) / f"{job_id}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._hist: deque[tuple[float, int]] = deque()
        self._last_write = 0.0
        self.t0 = time.time()
        self.state = {
            "job_id": job_id,
            "dataset": dataset,
            "state": "queued",
            "total": int(total),
            "done": 0,
            "inflight": 0,
            "failed": 0,
            "failures": [],
            "img_s": 0.0,
            "eta_s": None,
            "stages_ms": {},
            "started": time.time(),
            "updated": time.time(),
            **extra,
        }
        self._write(force=True)

    # -- lifecycle -------------------------------------------------
    def start(self) -> None:
        self.state["state"] = "running"
        self.t0 = time.time()
        self._write(force=True)

    def finish(self, **extra) -> None:
        self.state.update(state="done", inflight=0, eta_s=0.0, **extra)
        self._write(force=True)

    def aborted(self) -> bool:
        """True once someone asked this job to stop (e.g. `manage delete --force`)."""
        return abort_path(self.state["job_id"], self._home).exists()

    def abort(self, reason: str = "aborted by request") -> None:
        self.state.update(state="aborted", inflight=0, error=reason)
        self._write(force=True)

    def fail(self, reason: str) -> None:
        self.state.update(state="failed", error=str(reason)[:500], inflight=0)
        self._write(force=True)

    def add_failure(self, path: str, reason: str) -> None:
        self.state["failed"] += 1
        if len(self.state["failures"]) < 200:  # ponytail: cap the inline list; count is exact
            self.state["failures"].append({"path": str(path), "reason": str(reason)[:200]})

    # -- progress --------------------------------------------------
    def update(self, done: int, inflight: int = 0, stages_ms: dict | None = None, force: bool = False) -> None:
        now = time.time()
        self._hist.append((now, done))
        while self._hist and now - self._hist[0][0] > RATE_WINDOW_S:
            self._hist.popleft()
        rate = 0.0
        if len(self._hist) > 1:
            dt = self._hist[-1][0] - self._hist[0][0]
            dn = self._hist[-1][1] - self._hist[0][1]
            rate = dn / dt if dt > 0 else 0.0
        left = max(0, self.state["total"] - done - self.state["failed"])
        self.state.update(
            done=int(done),
            inflight=int(inflight),
            img_s=round(rate, 2),
            eta_s=round(left / rate, 1) if rate > 0 else None,
            elapsed_s=round(now - self.t0, 2),
        )
        if stages_ms:
            self.state["stages_ms"] = {k: round(v, 2) for k, v in stages_ms.items()}
        self._write(force=force)

    def _write(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_write < HEARTBEAT_S:
            return
        self.state["updated"] = now
        self.state["in_flight"] = self.state["inflight"]  # both spellings: b-skill's
        # contract test reads `inflight`, b-daemon's SSE reads `in_flight`
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(self.state, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)
        self._last_write = now
