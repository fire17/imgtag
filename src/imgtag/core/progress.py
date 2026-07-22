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


class Job:
    """One index job's status file. Cheap: writes are throttled and event-driven
    (B10(d): the emitter must cost <=1% of run wall time — no polling thread)."""

    def __init__(self, job_id: str, dataset: str, total: int, home: Path | None = None, **extra):
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
            "in_flight": 0,
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
        self.state.update(state="done", in_flight=0, eta_s=0.0, **extra)
        self._write(force=True)

    def fail(self, reason: str) -> None:
        self.state.update(state="failed", error=str(reason)[:500], in_flight=0)
        self._write(force=True)

    def add_failure(self, path: str, reason: str) -> None:
        self.state["failed"] += 1
        if len(self.state["failures"]) < 200:  # ponytail: cap the inline list; count is exact
            self.state["failures"].append({"path": str(path), "reason": str(reason)[:200]})

    # -- progress --------------------------------------------------
    def update(self, done: int, in_flight: int = 0, stages_ms: dict | None = None, force: bool = False) -> None:
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
            in_flight=int(in_flight),
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
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(self.state, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)
        self._last_write = now
