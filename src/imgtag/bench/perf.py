"""Per-config throughput + peak RSS, ONE FRESH PROCESS PER CONFIG.

Fresh process is mandatory (spike-siglip2 §3 burned itself on this): `ru_maxrss` is a
monotonic high-water mark, so a sweep inside one process reports the max of everything
that ran before, not the config. Ported from `.scratch/pecore/rssbench.py`.

Worker:  python -m imgtag.bench.perf <candidate_id> <precision> <intra> <batch>
         -> one JSON line on stdout.
Driver:  `run_matrix()` — median of >=3 fresh processes per row (BUDGETS bench protocol).
"""
from __future__ import annotations

import glob
import json
import os
import statistics
import subprocess
import sys
import time

import numpy as np

from . import candidates as C

IMAGES_PER_REP = 16  # equal work across batch sizes -> comparable ms/img


def _worker(cand_id: str, prec: str, intra: int, batch: int) -> dict:
    c = C.CANDIDATES[cand_id]
    path = c.path(prec)
    if not path:
        return {"error": f"artifact missing for {cand_id}/{prec}"}

    rss_base = C.peak_rss_mb()
    pool = sorted(glob.glob(os.path.join(C.DATA, "quick500/images/*.jpg")))[:batch]
    X = np.stack([C.preprocess(p, c) for p in pool])
    if len(X) < batch:  # pad by tiling (never fewer rows than the shape under test)
        X = np.concatenate([X] * (batch // len(X) + 1))[:batch]
    rss_data = C.peak_rss_mb()

    t0 = time.perf_counter()
    sess = C.session(path, intra)
    load_s = time.perf_counter() - t0
    rss_load = C.peak_rss_mb()

    feed = {"pixel_values": X}
    for _ in range(2):
        sess.run(None, feed)

    nbatch = max(3, IMAGES_PER_REP // batch)
    load_before = os.getloadavg()[0]
    t = time.perf_counter()
    for _ in range(nbatch):
        sess.run(None, feed)
    elapsed = time.perf_counter() - t
    load_after = os.getloadavg()[0]

    n_img = nbatch * batch
    return {
        "candidate": cand_id, "precision": prec, "intra": intra, "batch": batch,
        "ms_per_img": round(elapsed / n_img * 1000, 2),
        "img_s": round(n_img / elapsed, 2),
        "peak_rss_mb": round(C.peak_rss_mb(), 1),
        "rss_base_mb": round(rss_base, 1), "rss_data_mb": round(rss_data, 1),
        "rss_afterload_mb": round(rss_load, 1),
        "session_load_s": round(load_s, 2),
        "artifact_mb": round(os.path.getsize(path) / 1e6, 1),
        "load_before": round(load_before, 2), "load_after": round(load_after, 2),
        "n_img": n_img,
    }


PERF_CACHE = os.path.join(C.ROOT, "bench", "cache", "perf")


def _cache_path(cand_id: str, prec: str, intra: int, batch: int) -> str:
    return os.path.join(PERF_CACHE, f"{cand_id}-{prec}-i{intra}-b{batch}.json")


def run_config(cand_id: str, prec: str, intra: int, batch: int, repeats: int = 3,
               timeout: int = 900, use_cache: bool = True) -> dict:
    """Spawn `repeats` fresh processes; return the MEDIAN row + spread + advisory flag.

    Perf rows are cached so a later quality-only run composes them into one report (perf
    and quality run in separate passes; without this the B8 column reads empty). A cached
    ADVISORY row is replaced whenever a fresh run beats its loadavg (a quiet-window pass
    supersedes a contended one); an OK row is never overwritten by an ADVISORY one.
    """
    cache = _cache_path(cand_id, prec, intra, batch)
    if use_cache and os.path.exists(cache):
        try:
            return json.load(open(cache))
        except ValueError:
            pass
    cores = C.usable_cores()[0]
    rows, errs = [], []
    for _ in range(repeats):
        p = subprocess.run(
            [sys.executable, "-m", "imgtag.bench.perf", cand_id, prec, str(intra), str(batch)],
            capture_output=True, text=True, timeout=timeout,
            env={**os.environ, "PYTHONWARNINGS": "ignore"})
        line = (p.stdout or "").strip().splitlines()
        try:
            r = json.loads(line[-1])
        except (ValueError, IndexError):
            errs.append((p.stderr or p.stdout or "no output")[-300:])
            continue
        if "error" in r:
            errs.append(r["error"])
            continue
        rows.append(r)
    if not rows:
        return {"candidate": cand_id, "precision": prec, "intra": intra, "batch": batch,
                "status": "BLOCKED", "error": errs[-1] if errs else "no rows"}

    med = dict(sorted(rows, key=lambda r: r["ms_per_img"])[len(rows) // 2])
    ms = [r["ms_per_img"] for r in rows]
    loads = [r["load_before"] for r in rows]
    med["runs"] = len(rows)
    med["ms_per_img_min"] = min(ms)
    med["ms_per_img_max"] = max(ms)
    med["spread_pct"] = round((max(ms) - min(ms)) / statistics.median(ms) * 100, 1)
    med["peak_rss_mb"] = max(r["peak_rss_mb"] for r in rows)
    med["loadavg_max"] = max(loads)
    # BUDGETS noise protocol: never fabricate a quiet-machine number.
    med["status"] = "ADVISORY" if max(loads) > cores * 0.6 else "OK"
    # Cache, but never let an ADVISORY row overwrite a cached OK one.
    prior = json.load(open(cache)) if os.path.exists(cache) else None
    if not (prior and prior.get("status") == "OK" and med["status"] == "ADVISORY"):
        os.makedirs(PERF_CACHE, exist_ok=True)
        tmp = cache + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(med, fh)
        os.replace(tmp, cache)
    return med


def load_cached(cand_id: str, precisions=("fp32", "int8"), intras=(1, 2, 4),
                batches=(1, 2, 8)) -> list[dict]:
    """Cached perf rows only (no spawning) — lets a quality-only run show the B8 column."""
    out = []
    for prec in precisions:
        for intra in intras:
            for batch in batches:
                p = _cache_path(cand_id, prec, intra, batch)
                if os.path.exists(p):
                    try:
                        out.append(json.load(open(p)))
                    except ValueError:
                        pass
    return out


def run_matrix(cand_id: str, precisions=("fp32", "int8"), intras=(1, 2, 4),
               batches=(1, 2, 8), repeats: int = 3, log=print) -> list[dict]:
    out = []
    for prec in precisions:
        if not C.CANDIDATES[cand_id].path(prec):
            log(f"  {cand_id}/{prec}: artifact missing — skipped")
            continue
        for intra in intras:
            for batch in batches:
                r = run_config(cand_id, prec, intra, batch, repeats=repeats)
                out.append(r)
                log(f"  {cand_id} {prec} intra={intra} b={batch}: "
                    f"{r.get('img_s', '—')} img/s  {r.get('ms_per_img', '—')} ms/img  "
                    f"rss={r.get('peak_rss_mb', '—')}MB  [{r['status']}]")
    return out


if __name__ == "__main__":
    print(json.dumps(_worker(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))))
