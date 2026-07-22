"""First-run autotune + machine profile — ADR-10d/ADR-11.

OWNER: b-engine. "Generic and ready" means the engine adapts itself to the host, not
that we guessed: `imgtag doctor` micro-benches precision × intra_op × batch on the real
machine (~30s) and writes ~/.imgtag/profile.json. Until it has run, precision is fp32
(ADR-10e: int8 is enabled only by a measured win on THIS host — on pre-VNNI AVX2 it is
routinely a slowdown).
"""

from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from .store import imgtag_home

PROFILE_VERSION = 1
WORKER_RSS_MB = 130  # spawn start-method: each decode worker is a fresh interpreter + PIL
SESSION_RSS_MB = 420  # central ORT vision session headroom (spike: 329MB fp32 @384 b1)
INDEX_RSS_BUDGET_MB = 1024  # B8 hard: peak tree RSS while indexing


def usable_cores() -> tuple[int, str]:
    """EFFECTIVE cores (ADR-11) — never os.cpu_count() alone: a shared server is very
    likely a cgroup-limited container where cpu_count() reports the HOST's cores."""
    try:
        n = len(os.sched_getaffinity(0))
        src = "sched_getaffinity"
    except AttributeError:
        n, src = os.cpu_count() or 2, "cpu_count"
    for path, parse in (
        ("/sys/fs/cgroup/cpu.max", lambda s: None if s.split()[0] == "max" else int(s.split()[0]) / int(s.split()[1])),
        ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", None),
    ):
        try:
            if parse:
                q = parse(Path(path).read_text().strip())
                src2 = "cgroup v2 cpu.max"
            else:
                quota = int(Path(path).read_text().strip())
                period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text().strip())
                q = None if quota <= 0 else quota / period
                src2 = "cgroup v1 cfs_quota"
            if q:
                q = max(1, int(q))
                if q < n:
                    n, src = q, src2
        except (OSError, ValueError, ZeroDivisionError):
            pass
    return max(1, n), src


def mem_available_mb() -> int:
    try:  # Linux (+ cgroup limit if tighter)
        info = dict(
            (k.strip(), v) for k, v in (ln.split(":", 1) for ln in Path("/proc/meminfo").read_text().splitlines())
        )
        avail = int(info["MemAvailable"].split()[0]) // 1024
        try:
            lim = Path("/sys/fs/cgroup/memory.max").read_text().strip()
            if lim != "max":
                avail = min(avail, int(lim) // (1 << 20))
        except OSError:
            pass
        return avail
    except (OSError, KeyError, ValueError):
        pass
    try:  # macOS dev box: no MemAvailable, use total as a coarse stand-in
        import subprocess

        total = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout)
        return int(total / (1 << 20) * 0.5)
    except Exception:
        return 4096


def worker_count(cores: int | None = None, avail_mb: int | None = None, full_speed: bool = False) -> int:
    """POLITE (default): clamp(usable-2, 2, 8), capped by the memory arithmetic (ADR-11)."""
    n = cores if cores is not None else usable_cores()[0]
    if full_speed:
        return max(1, n)
    avail = avail_mb if avail_mb is not None else mem_available_mb()
    budget = min(INDEX_RSS_BUDGET_MB, max(256, avail // 2)) - SESSION_RSS_MB
    by_mem = max(1, int(budget // WORKER_RSS_MB))
    return max(1, min(max(2, min(n - 2, 8)), by_mem, n))


def default_profile() -> dict:
    cores, src = usable_cores()
    return {
        "version": PROFILE_VERSION,
        "measured": False,
        "cores": cores,
        "cores_source": src,
        "mem_available_mb": mem_available_mb(),
        "machine": f"{platform.machine()} {platform.system()}",
        "precision": "fp32",  # ADR-10e: unprofiled host => fp32, always
        "intra_op": 2,
        "text_intra_op": 2,
        "batch": 2,
        "workers": worker_count(cores),
        "geometry": "central",
    }


def profile_path(home: Path | None = None) -> Path:
    return (home or imgtag_home()) / "profile.json"


def load_profile(home: Path | None = None) -> dict:
    p = profile_path(home)
    if p.is_file():
        try:
            prof = json.loads(p.read_bytes())
            if prof.get("version") == PROFILE_VERSION:
                return prof
        except (OSError, ValueError):
            pass
    return default_profile()


def save_profile(prof: dict, home: Path | None = None) -> Path:
    p = profile_path(home)
    p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(prof, indent=1))
    os.replace(tmp, p)
    return p


def autotune(backend_name: str = None, timebox_s: float = 45.0, n_images: int = 8, log=print) -> dict:
    """~30s micro-bench of precision × intra_op × batch on THIS machine (ADR-10d).

    Geometry stays `central` (one ORT session fed by decode workers) — the per-worker-
    session geometry costs SESSION_RSS_MB per worker and does not fit B8 on the 8GB
    target; it remains a bench-swept axis, not a doctor knob.
    """
    from .models import DEFAULT_BACKEND, ModelBackend, registry

    name = backend_name or DEFAULT_BACKEND
    spec = registry()[name]
    prof = default_profile()
    rng = np.random.default_rng(0)
    imgs = rng.integers(0, 255, (n_images, spec["size"], spec["size"], 3), dtype=np.uint8)
    rows, t0 = [], time.time()
    for precision in [p for p in ("fp32", "int8") if p in spec["vision"]]:
        for intra in (1, 2, 4):
            if intra > prof["cores"]:
                continue
            try:
                be = ModelBackend(name, spec, {**prof, "precision": precision, "intra_op": intra})
            except Exception as e:  # artifact for this precision missing
                log(f"  skip {precision} intra={intra}: {e}")
                break
            be.embed_images(imgs[:1])  # warm-up
            for batch in (1, 2):
                t = time.perf_counter()
                for i in range(0, n_images, batch):
                    be.embed_images(imgs[i : i + batch])
                dt = time.perf_counter() - t
                rows.append(
                    {"precision": precision, "intra_op": intra, "batch": batch, "img_s": round(n_images / dt, 2)}
                )
                log(f"  {precision} intra={intra} batch={batch}: {rows[-1]['img_s']} img/s")
            del be
            if time.time() - t0 > timebox_s:
                log("  timebox reached, stopping sweep")
                break
        if time.time() - t0 > timebox_s:
            break
    if not rows:
        raise RuntimeError(f"no runnable configuration for backend {name!r}")
    best = max(rows, key=lambda r: r["img_s"])
    fp32_best = max((r for r in rows if r["precision"] == "fp32"), key=lambda r: r["img_s"], default=None)
    if fp32_best and best["precision"] == "int8" and best["img_s"] < fp32_best["img_s"] * 1.05:
        best = fp32_best  # int8 must EARN it on this host (ADR-10e)
    prof.update(
        measured=True,
        backend=name,
        precision=best["precision"],
        intra_op=best["intra_op"],
        batch=best["batch"],
        sweep=rows,
        tuned_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        tune_seconds=round(time.time() - t0, 1),
        loadavg=[round(x, 2) for x in os.getloadavg()],
    )
    prof["workers"] = worker_count(prof["cores"], prof["mem_available_mb"])
    return prof
