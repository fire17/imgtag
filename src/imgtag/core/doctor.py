"""First-run autotune + machine profile — ADR-10d/ADR-11.

OWNER: b-engine. "Generic and ready" means the engine adapts itself to the host, not
that we guessed: `imgtag doctor` micro-benches precision × intra_op × batch on the real
machine (~30s) and writes ~/.imgtag/profile.json.

Precision is fp32 — before AND after tuning. RULING 2026-07-22 (B24 two-tier gate):
v1 ships fp32 vision everywhere; doctor measures the int8 speed-up and reports it as an
offer (`int8_offer`), but only `--allow-int8` / `--precision int8` enables it, and only
after B24 tier-2 passes on that arch.
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
WORKER_SESSION_RSS_MB = 210  # decode worker that ALSO owns a session (spike: 188MB int8 b1)
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


def worker_count(
    cores: int | None = None, avail_mb: int | None = None, full_speed: bool = False, geometry: str = "central"
) -> int:
    """POLITE (default): clamp(usable-2, 2, 8), capped by the memory arithmetic (ADR-11).

    geometry=worker puts one ORT session in every worker (measured 188MB for PE-Core-S
    int8 b1, 329MB fp32) so the per-worker cost — and therefore the cap — is much higher.
    """
    n = cores if cores is not None else usable_cores()[0]
    avail = avail_mb if avail_mb is not None else mem_available_mb()
    per_worker = WORKER_SESSION_RSS_MB if geometry == "worker" else WORKER_RSS_MB
    budget = min(INDEX_RSS_BUDGET_MB, max(256, avail // 2)) - (0 if geometry == "worker" else SESSION_RSS_MB)
    by_mem = max(1, int(budget // per_worker))
    if full_speed:
        return max(1, min(n, by_mem))
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


def autotune(backend_name: str = None, timebox_s: float = 45.0, n_images: int = 8, log=print,
             allow_int8: bool = False) -> dict:
    """~30s micro-bench of precision × intra_op × batch on THIS machine (ADR-10d).

    Also picks the pipeline geometry: `central` (one session fed by decode workers) vs
    `worker` (a session per decode worker, memory permitting), projecting the latter from
    the measured intra_op=1 row × the memory-derived worker count. `bench index
    --geometry` settles it for real; doctor only has to pick a sane default in ~30s.
    """
    from .models import DEFAULT_BACKEND, ModelBackend, registry

    name = backend_name or DEFAULT_BACKEND
    spec = registry()[name]
    prof = default_profile()
    rng = np.random.default_rng(0)
    imgs = rng.integers(0, 255, (n_images, spec["size"], spec["size"], 3), dtype=np.uint8)
    rows, t0 = [], time.time()
    # precision axis = the UNGATED vision variants only (RULING: a downloaded/naive int8
    # vision file is never swept; b-bench promotes one through B24 first)
    for precision in [p for p in ("fp32", "fp16", "int8") if p in spec["vision"]]:
        for intra in (1, 2, 4):
            if intra > prof["cores"]:
                continue
            try:
                be = ModelBackend(name, spec, {**prof, "precision": precision, "precision_explicit": True, "intra_op": intra})
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
    fp32_best = max((r for r in rows if r["precision"] == "fp32"), key=lambda r: r["img_s"], default=None)
    int8_best = max((r for r in rows if r["precision"] == "int8"), key=lambda r: r["img_s"], default=None)
    # RULING 2026-07-22 (B24 two-tier gate): v1 ships fp32 vision EVERYWHERE. int8 is an
    # opt-in speed lane that must first pass B24 tier-2 on this arch — doctor measures it
    # and reports the offer, but never enables it on its own.
    best = fp32_best or int8_best
    if allow_int8 and int8_best and (not fp32_best or int8_best["img_s"] > fp32_best["img_s"] * 1.05):
        best = int8_best
    # geometry: central (one session at the best intra_op) vs worker (W sessions at
    # intra_op=1). Projected from the measured intra=1 row of the winning precision —
    # derived from measurement, not guessed; `bench index --geometry` settles it.
    w_workers = worker_count(prof["cores"], prof["mem_available_mb"], geometry="worker")
    solo = next((r["img_s"] for r in rows if r["precision"] == best["precision"] and r["intra_op"] == 1
                 and r["batch"] == best["batch"]), None)
    proj = round(solo * w_workers, 2) if solo else 0.0
    geometry = "worker" if proj > best["img_s"] * 1.15 else "central"
    prof.update(
        measured=True,
        backend=name,
        precision=best["precision"],
        intra_op=best["intra_op"],
        batch=best["batch"],
        geometry=geometry,
        geometry_projection={"central_img_s": best["img_s"], "worker_img_s_projected": proj,
                             "worker_processes": w_workers, "basis": "measured intra_op=1 row x workers"},
        worker_intra_op=1,
        sweep=rows,
        int8_offer=(
            {"img_s": int8_best["img_s"], "speedup_vs_fp32": round(int8_best["img_s"] / fp32_best["img_s"], 2),
             "intra_op": int8_best["intra_op"], "batch": int8_best["batch"],
             "status": "opt-in only until it passes B24 tier-2 on this arch (--allow-int8 / --precision int8)"}
            if int8_best and fp32_best else None),
        tuned_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        tune_seconds=round(time.time() - t0, 1),
        loadavg=[round(x, 2) for x in os.getloadavg()],
    )
    prof["workers"] = worker_count(prof["cores"], prof["mem_available_mb"], geometry=geometry)
    prof["text_resident_rss_mb"] = text_resident_rss(name, log=log)
    return prof


def text_resident_rss(backend: str, log=print) -> float | None:
    """Peak RSS of a process that loads ONLY this backend's text tower + tokenizer.

    Measured in a fresh child (a delta inside this process would be masked by the vision
    session's high-water mark). Feeds candidate eligibility: PE-Core's int8 text tower is
    154MB resident, SigLIP2's is ~757MB — B8 precedence, not taste, picks the default.
    """
    import subprocess
    import sys

    code = (
        "import resource,sys;from imgtag.core.models import load_backend;"
        f"be=load_backend({backend!r},{{'intra_op':1}},vision=False);be.embed_texts(['a photo of a dog']);"
        "u=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;"
        "print(u/1e6 if sys.platform=='darwin' else u/1e3)"
    )
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180)
        return round(float(r.stdout.strip().splitlines()[-1]), 1)
    except Exception as e:  # text tower unavailable (no tokenizer binary / missing file)
        log(f"  text RSS probe skipped: {str(e)[:100]}")
        return None
