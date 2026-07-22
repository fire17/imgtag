"""Indexing pipeline — ADR-11 resource policy + transport.

OWNER: b-engine.

    decode workers (processes)                 coordinator (this process)
    read bytes once -> xxhash64 -> decode      free_q slot -> ready_q {slot, id, path, w, h}
    -> preprocess -> uint8 into a shm slot     -> batch B views -> ORT session -> Writer

Queues carry INTEGERS AND METADATA ONLY; pixels move through a
``multiprocessing.shared_memory`` slab (a pickling Queue of tensors is ~106MB/s of
serialize+pipe+deserialize at 60 img/s — on the one axis this project claims as its
edge). Backpressure is structural: a worker blocks on ``free_q.get()``.
"""

from __future__ import annotations

import io
import multiprocessing as mp
import os
import queue as _q
import signal
import time
from pathlib import Path

import numpy as np

from . import models
from .doctor import load_profile, worker_count
from .progress import Job
from .store import Writer, open_snapshot

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".heic", ".heif"}
JPEG_EXTS = {".jpg", ".jpeg"}
HEIC_EXTS = {".heic", ".heif"}
MAX_PIXELS = 80_000_000  # B21: the decompression bomb is refused by a pixel cap, not by OOM
FILE_TIMEOUT_S = 5  # B21: every file resolved or skipped within 5s


def parse_meta(pairs) -> dict:
    """``--meta key=value`` (repeatable) -> dict. Job-level: applies to every image."""
    out = {}
    for p in pairs or []:
        if "=" not in p:
            raise ValueError(f"--meta expects key=value, got {p!r}")
        k, v = p.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def load_meta_csv(path, root: Path | None = None) -> dict:
    """CSV -> {image key: {column: value}}. One column must be `path` or `filename`;
    every other column becomes per-image metadata (account ids, dates, whatever).

    Keys are matched later by absolute path, then by basename, so a CSV written against
    relative paths on another machine still lands.
    """
    import csv

    rows: dict[str, dict] = {}
    with open(path, newline="") as f:
        rdr = csv.DictReader(f)
        key_col = next((c for c in (rdr.fieldnames or []) if c and c.lower() in ("path", "filename", "file")), None)
        if not key_col:
            raise ValueError(f"{path}: needs a `path`, `filename` or `file` column; got {rdr.fieldnames}")
        for row in rdr:
            raw = (row.get(key_col) or "").strip()
            if not raw:
                continue
            meta = {k: v for k, v in row.items() if k != key_col and k and v not in (None, "")}
            p = Path(raw).expanduser()
            if not p.is_absolute() and root:
                p = (root / p)
            rows[str(p.resolve() if p.exists() else p)] = meta
            rows.setdefault(Path(raw).name, meta)  # basename fallback
    return rows


def scan(root: Path, recursive: bool = True) -> list[Path]:
    root = Path(root).expanduser()
    if root.is_file():
        return [root]
    it = root.rglob("*") if recursive else root.glob("*")
    return sorted(p for p in it if p.suffix.lower() in IMAGE_EXTS and p.is_file())


# ---------------------------------------------------------------- moderation hook

MODERATION_CATEGORIES = ("nudity", "weapons", "drugs")


def load_moderation_hook(spec=None, profile: dict | None = None, log=lambda m: None):
    """Resolve the index-time moderation detector. Returns a callable or None.

    PRIMARY PATH — the conductor-owned dispatcher (`imgtag.moderation.load_heads`):
    each track exports a Head with `.score(embeddings, images, ids) -> [{category, p,
    tier}]` per ADR-14. This function loads every head that can load on this machine and
    fans a batch across them; a track whose model files are absent simply is not there.

    OVERRIDE PATH — `spec="module:function"` (or $IMGTAG_MODERATION), a single callable
    `detect(embs, recs, images=None)` with the same per-record return shape. Used by
    tests and by anyone plugging in a detector outside the package.

    Either way the detector is called POST-EMBEDDING, BATCH-WISE, on the coordinator
    (never in a decode worker), so it may reuse the L2-normalized embeddings it is handed
    and cost ~0, or run its own model. It must never raise: exceptions are caught,
    counted, and indexing continues — moderation is a summary layer, not a gate on the
    user's index. Detectors needing pixels set `wants_images = True`; under
    geometry="worker" no pixels exist on the coordinator and they are handed None.
    """
    import importlib

    spec = spec if isinstance(spec, str) else os.environ.get("IMGTAG_MODERATION")
    if spec:
        try:
            mod, _, fn = spec.partition(":")
            return getattr(importlib.import_module(mod), fn or "detect")
        except (ImportError, AttributeError) as e:
            log(f"moderation hook {spec!r} unavailable ({type(e).__name__}) — indexing without it")
            return None
    try:
        heads = importlib.import_module("imgtag.moderation").load_heads(profile or {})
    except (ImportError, AttributeError, Exception) as e:  # a track's loader may itself fail
        if not isinstance(e, (ImportError, AttributeError)):
            log(f"moderation dispatcher failed ({type(e).__name__}: {e}) — indexing without it")
        return None
    if not heads:
        return None
    log(f"moderation heads loaded: {', '.join(sorted(heads))}")

    def detect(embs, recs, images=None):
        per: list[list[dict]] = [[] for _ in recs]
        for name, head in heads.items():
            try:
                out = _call_head(head, embs, images, recs) or []
            except Exception as e:
                log(f"moderation head {name!r} raised {type(e).__name__}: {e} — skipped for this batch")
                continue
            for i, flags in enumerate(out[: len(per)]):
                for f in (flags if isinstance(flags, list) else [flags]):
                    if f:
                        per[i].append(f)
        return per

    detect.wants_images = any(getattr(h, "wants_images", False) or hasattr(h, "score_images")
                              for h in heads.values())
    detect.heads = heads
    return detect


def _call_head(head, embs, images, recs):
    """Call a track head through whichever shape it actually implements.

    The documented seam is `score(embeddings, images, ids) -> [{category,p,tier}]`. The
    nudity track ships `score_images(list[PIL.Image]) -> [{category,p,flagged}]` plus a
    tensor-only `score(batch)`; the PIL path is UNAMBIGUOUS so it is supported here. A
    bare `score(batch)` is NOT guessed at — feeding it embeddings would silently produce
    numbers that mean nothing, which is worse than the loud skip the caller gets.
    """
    try:
        return head.score(embs, images, recs)          # the finalized seam
    except TypeError:
        if not hasattr(head, "score_images"):
            raise
    if images is None:                                  # legacy PIL-only head
        raise RuntimeError("head needs pixels but this batch has none (geometry=worker)")
    from PIL import Image as _Image

    return head.score_images([_Image.fromarray(im) for im in images])


def _apply_moderation(hook, embs, recs, images, counts: dict, log) -> int:
    """Attach ADR-14 flags to recs and accumulate per-TIER, per-category counts.

    Returns 1 if the detector failed for this batch (counted, never fatal).
    """
    try:
        wants = getattr(hook, "wants_images", False)
        out = hook(embs, recs, images if (images is not None or not wants) else None)
    except Exception as e:  # never let a detector break the index
        log(f"moderation hook raised {type(e).__name__}: {e} — flags skipped for this batch")
        return 1
    for rec, flags in zip(recs, out or []):
        keep = []
        for f in flags or []:
            # ADR-14: tier is the carrier. Legacy `flagged: True` maps to violation so an
            # older detector still counts for something instead of vanishing silently.
            tier = f.get("tier") or ("violation" if f.get("flagged") else "none")
            if tier == "none":
                continue
            keep.append({"category": f["category"], "p": round(float(f.get("p", 1.0)), 4), "tier": tier})
            counts.setdefault(tier, {})
            counts[tier][f["category"]] = counts[tier].get(f["category"], 0) + 1
        if keep:
            rec["flags"] = keep
    return 0


def empty_counts() -> dict:
    return {t: {c: 0 for c in MODERATION_CATEGORIES} for t in ("violation", "review")}


def merge_counts(counts: dict) -> dict:
    """Fill the fixed ADR-14 shape so consumers never branch on a missing key."""
    out = empty_counts()
    for tier, per_cat in (counts or {}).items():
        for cat, n in (per_cat or {}).items():
            out.setdefault(tier, {})
            out[tier][cat] = out[tier].get(cat, 0) + n
    return out


def moderation_summary(counts: dict, active: bool = True) -> str:
    """The job-end line, in the user's phrasing with ADR-14's tiers made visible:
    "Found N images with drugs (M for review), ...". Heads absent is a real answer."""
    if not active:
        return "moderation: off (no tracks loaded)"
    c = merge_counts(counts)
    order = list(MODERATION_CATEGORIES) + [k for k in sorted(c.get("violation", {})) if k not in MODERATION_CATEGORIES]
    return "Found " + ", ".join(
        f"{c['violation'].get(k, 0)} images with {k}" + (f" ({r} for review)" if (r := c["review"].get(k, 0)) else "")
        for k in order)


# ---------------------------------------------------------------- workers


def _polite(nice_level: int = 10) -> None:
    try:
        os.nice(nice_level)  # set IN the worker, never merely inherited (ADR-11)
    except OSError:
        pass
    try:  # we die first, not the co-tenant (B15)
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write("500")
    except OSError:
        pass


class _Timeout(Exception):
    pass


def _worker(task_q, ready_q, free_q, shm_name, size, squash, resample_name, sem, nice_level, worker_be=None):
    """geometry=central: decode into a shm slot, the coordinator runs the one session.
    geometry=worker  : this process owns an ORT session and returns the embedding
    (measured 1.7× on CPU — process-level parallelism beats intra-op threading — at the
    cost of one session's RSS per worker, which is why doctor gates it on memory)."""
    from multiprocessing.shared_memory import SharedMemory

    import xxhash
    from PIL import Image

    from .models import load_backend, preprocess_image

    _polite(nice_level)
    Image.MAX_IMAGE_PIXELS = MAX_PIXELS
    try:  # ADR-7: HEIC is an OPTIONAL extra, never a silent gap in the index
        import pillow_heif

        pillow_heif.register_heif_opener()
        heic_ok = True
    except ImportError:
        heic_ok = False
    resample = getattr(Image.Resampling, resample_name)
    be = load_backend(*worker_be) if worker_be else None
    shm = slab = None
    if be is None:
        shm = SharedMemory(name=shm_name)
        nslots = shm.size // (size * size * 3)
        slab = np.ndarray((nslots, size, size, 3), np.uint8, buffer=shm.buf)
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, lambda *a: (_ for _ in ()).throw(_Timeout("decode timeout")))
    try:
        while True:
            path = task_q.get()
            if path is None:
                return
            p = Path(path)
            suffix = p.suffix.lower()
            heavy = suffix not in JPEG_EXTS  # PNG/HEIC full decodes are the RAM term
            slot = None
            try:
                if suffix in HEIC_EXTS and not heic_ok:
                    raise RuntimeError("HEIC support is an optional extra — install `imgtag[heic]` (pillow-heif)")
                if hasattr(signal, "SIGALRM"):
                    signal.alarm(FILE_TIMEOUT_S)
                t_dec = time.perf_counter()
                st = p.stat()
                with open(p, "rb") as f:
                    buf = f.read()  # ONE read: hash and decode share this buffer
                    try:
                        os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)  # B15 cache hygiene
                    except (AttributeError, OSError):
                        pass
                image_id = xxhash.xxh64(buf).hexdigest()
                im = Image.open(io.BytesIO(buf))
                w, h = im.size
                if heavy:
                    sem.acquire()
                try:
                    arr = preprocess_image(im, size, squash, resample)
                finally:
                    if heavy:
                        sem.release()
                decode_ms = (time.perf_counter() - t_dec) * 1000
                emb = None
                if be is not None:
                    t_inf = time.perf_counter()
                    emb = be.embed_images(arr[None])[0].tobytes()
                    infer_ms = (time.perf_counter() - t_inf) * 1000
                else:
                    infer_ms = 0.0
                    slot = free_q.get()  # blocks when the consumer is behind -> backpressure
                    slab[slot] = arr
                ready_q.put(
                    {
                        "slot": slot,
                        "emb": emb,
                        "infer_ms": infer_ms,
                        "image_id": image_id,
                        "path": str(p),
                        "w": w,
                        "h": h,
                        "mtime": st.st_mtime,
                        "size": st.st_size,
                        "decode_ms": decode_ms,
                        "pid": os.getpid(),
                    }
                )
            except BaseException as e:  # never kill the run for one bad file (B21)
                if slot is not None:
                    free_q.put(slot)
                ready_q.put({"error": f"{type(e).__name__}: {e}"[:200].replace(str(p), p.name),
                             "path": str(p), "pid": os.getpid()})
            finally:
                if hasattr(signal, "SIGALRM"):
                    signal.alarm(0)
    finally:
        if shm is not None:
            shm.close()


# ---------------------------------------------------------------- coordinator


def _sweep_stale_segments() -> None:
    """resource_tracker leaks segments on abnormal exit; reap ours whose pgid is dead."""
    d = Path("/dev/shm")
    if not d.is_dir():
        return
    for p in d.glob("imgtag-*"):
        try:
            pgid = int(p.name.split("-")[1])
            os.killpg(pgid, 0)
        except (ValueError, IndexError):
            continue
        except ProcessLookupError:
            try:
                p.unlink()
            except OSError:
                pass
        except PermissionError:
            pass


def index(
    path,
    dataset: str,
    backend: str | None = None,
    profile: dict | None = None,
    home: Path | None = None,
    full_speed: bool = False,
    workers: int | None = None,
    job_id: str | None = None,
    recursive: bool = True,
    meta: dict | None = None,
    meta_csv=None,
    moderation: str | None = None,
    log=lambda m: None,
    on_progress=None,
) -> dict:
    """Index a folder into a dataset. Returns a summary dict (also on the job file)."""
    from multiprocessing.shared_memory import SharedMemory

    files = scan(Path(path), recursive)
    root = Path(path).expanduser()
    job_meta = dict(meta or {})
    per_image_meta = load_meta_csv(meta_csv, root if root.is_dir() else root.parent) if meta_csv else {}
    mod_counts: dict[str, dict] = {}
    mod_errors = 0
    prof = dict(profile or load_profile(home))
    hook = load_moderation_hook(moderation, prof, log) if moderation else None
    name = backend or prof.get("backend") or models.DEFAULT_BACKEND
    geometry = prof.get("geometry", "central")
    be = models.load_backend(name, prof, vision=(geometry != "worker"))
    workers = workers or worker_count(prof.get("cores"), prof.get("mem_available_mb"), full_speed, geometry)
    batch = int(prof.get("batch", 2))
    nslots = min(128, max(16, 4 * batch, 4 * workers))
    slot_bytes = be.size * be.size * 3
    if hook is not None and getattr(hook, "wants_images", False) and geometry == "worker":
        # a detector that needs pixels cannot be fed under the per-worker geometry (only
        # embeddings cross that boundary), so the JOB drops to the central session — a
        # measured throughput cost, taken deliberately and stated, never a silent skip
        log("moderation needs pixels — using the central-session geometry for this job")
        geometry = "central"
        be = models.load_backend(name, prof, vision=True)
        workers = worker_count(prof.get("cores"), prof.get("mem_available_mb"), full_speed, geometry)
    worker_be = None
    if geometry == "worker":
        # each worker owns a session at intra_op=1; the coordinator only stores
        worker_be = (name, {**prof, "intra_op": prof.get("worker_intra_op", 1)})
        nslots = 1  # no pixels cross the boundary in this geometry

    # incremental gate (B12 compute leak): unchanged files are never re-embedded
    known: dict[str, tuple[float, int]] = {}
    try:
        snap = open_snapshot(dataset, home)
        known = {r["path"]: (r.get("mtime", -1), r.get("size", -1)) for r in snap.ids}
        del snap
    except Exception:
        pass
    todo = [p for p in files if known.get(str(p), (None, None)) != (p.stat().st_mtime, p.stat().st_size)]
    skipped = len(files) - len(todo)

    ctx = mp.get_context("spawn")
    _sweep_stale_segments()
    shm = SharedMemory(create=True, size=nslots * slot_bytes, name=f"imgtag-{os.getpgrp()}-{os.getpid()}")
    slab = np.ndarray((nslots, be.size, be.size, 3), np.uint8, buffer=shm.buf)
    task_q, ready_q, free_q = ctx.Queue(), ctx.Queue(), ctx.Queue()
    sem = ctx.BoundedSemaphore(4)  # non-JPEG full decodes capped (ADR-11)
    procs: list[mp.Process] = []
    t_decode = t_infer = t_queue = 0.0
    summary: dict = {}

    with Writer(dataset, be, home=home, job_id=job_id) as w:
        job = Job(
            w.job_id,
            dataset,
            len(files),
            home=home,
            model_id=be.model_id,
            model_sha=be.model_sha,
            workers=workers,
            batch=batch,
            precision=be.precision,
            root=str(Path(path).expanduser()),
            skipped=skipped,
        )
        try:
            for s in range(nslots):
                free_q.put(s)
            for p in todo:
                task_q.put(str(p))
            for _ in range(workers):
                task_q.put(None)
            for _ in range(workers):
                pr = ctx.Process(
                    target=_worker,
                    args=(task_q, ready_q, free_q, shm.name, be.size, be.squash, be.spec["resample"], sem,
                          0 if full_speed else 10, worker_be),
                    daemon=True,
                )
                pr.start()
                procs.append(pr)
            job.start()
            log(f"indexing {len(todo)} images ({skipped} unchanged) with {workers} workers, "
                f"{be.model_id}, batch {batch}, intra_op {prof.get('intra_op')}")

            received = 0
            pend_slots: list[int] = []
            pend_embs: list[np.ndarray] = []
            pend_recs: list[dict] = []
            prior = w.count
            aborted = False
            t_start = time.time()

            def flush_batch():
                nonlocal t_infer
                if not pend_recs:
                    return
                nonlocal mod_errors
                if worker_be:  # workers already embedded; we only store
                    embs = np.stack(pend_embs)
                    images = None
                else:
                    images = np.stack([slab[s] for s in pend_slots])
                    t = time.perf_counter()
                    embs = be.embed_images(images)
                    t_infer += time.perf_counter() - t
                if hook is not None:
                    mod_errors += _apply_moderation(hook, embs, pend_recs, images, mod_counts, log)
                w.append(embs, pend_recs.copy())
                for s in pend_slots:
                    free_q.put(s)
                pend_slots.clear()
                pend_embs.clear()
                pend_recs.clear()

            while received < len(todo):
                if job.aborted():  # `manage delete --force`: stop cleanly, keep the index consistent
                    log(f"job {w.job_id} aborted by request after {w.count - prior} images")
                    aborted = True
                    break
                t = time.perf_counter()
                try:
                    msg = ready_q.get(timeout=0.5)
                except _q.Empty:
                    t_queue += time.perf_counter() - t
                    flush_batch()  # never let a partial batch sit while the queue starves
                    job.update(w.count, inflight=received - w.count)
                    if not any(pr.is_alive() for pr in procs) and ready_q.empty():
                        log("all decode workers exited early")
                        break
                    continue
                t_queue += time.perf_counter() - t
                received += 1
                if "error" in msg:
                    job.add_failure(msg["path"], msg["error"])
                else:
                    t_decode += msg.get("decode_ms", 0.0)
                    t_infer += msg.get("infer_ms", 0.0) / 1000.0
                    if msg.get("emb") is not None:
                        pend_embs.append(np.frombuffer(msg["emb"], np.float32))
                    else:
                        pend_slots.append(msg["slot"])
                    rec = {
                        "image_id": msg["image_id"],
                        "path": msg["path"],
                        "dataset": dataset,
                        "w": msg["w"],
                        "h": msg["h"],
                        "mtime": msg["mtime"],
                        "size": msg["size"],
                    }
                    m = {**job_meta, **per_image_meta.get(msg["path"], {}),
                         **per_image_meta.get(Path(msg["path"]).name, {})}
                    if m:
                        rec["meta"] = m
                    pend_recs.append(rec)
                    if len(pend_recs) >= batch:
                        flush_batch()
                job.update(w.count, inflight=received - w.count)
            flush_batch()
            w._flush_pending()  # make the tail durable before we report done
            wall = time.time() - t_start
            n = w.count
            summary = {
                "job_id": w.job_id,
                "dataset": dataset,
                "indexed": n - prior,
                "count": n,
                "skipped": skipped,
                "failed": job.state["failed"],
                "total_files": len(files),
                "seconds": round(wall, 2),
                "img_s": round((n - prior) / wall, 2) if wall > 0 else 0.0,
                "model_id": be.model_id,
                "model_sha": be.model_sha,
                "workers": workers,
                "batch": batch,
                "precision": be.precision,
                "intra_op": prof.get("intra_op"),
                "stages_ms_per_img": {
                    "decode": round(t_decode / max(1, n - prior), 2),  # summed across workers
                    "infer": round(t_infer * 1000 / max(1, n - prior), 2),
                    "queue_wait": round(t_queue * 1000 / max(1, n - prior), 2),
                },
                "flushes": w.flushes,
                "meta": job_meta,
                "moderation": merge_counts(mod_counts),
                "moderation_active": hook is not None,
                "moderation_requested": bool(moderation),
                "moderation_errors": mod_errors,
            }
            if job_meta:  # dataset-level metadata lives in the manifest, merged across jobs
                w.manifest["meta"] = {**w.manifest.get("meta", {}), **job_meta}
            if hook is not None:
                prev = (w.manifest.get("moderation") or {}).get("counts", {})
                total = merge_counts(prev)
                for tier, per_cat in merge_counts(mod_counts).items():
                    for cat, n in per_cat.items():
                        total[tier][cat] = total[tier].get(cat, 0) + n
                w.manifest["moderation"] = {
                    "counts": total, "schema": "ADR-14",
                    # honesty: tau is unfitted until b-bench fits it on labeled ground truth
                    "calibration": "unfitted", "enforcement_ready": False,
                    "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            if job_meta or hook is not None:
                w._commit()
            if moderation:
                log(moderation_summary(mod_counts, active=hook is not None))
            job.update(w.count, 0, summary["stages_ms_per_img"], force=True)
            job.state["moderation"] = summary["moderation"]
            job.state["meta"] = job_meta
            summary["aborted"] = aborted
            if aborted:
                job.abort(f"aborted after {n - prior} images")
            else:
                job.finish(indexed=n, seconds=summary["seconds"], img_s=summary["img_s"])
        except BaseException as e:
            job.fail(f"{type(e).__name__}: {e}")
            raise
        finally:
            for pr in procs:
                if pr.is_alive():
                    pr.terminate()
            for pr in procs:
                pr.join(timeout=5)
            for q in (task_q, ready_q, free_q):
                q.close()
                q.join_thread()
            slab = None  # release the shm export WITHOUT unbinding the name: `del` empties
            # flush_batch's closure cell, so any future flush-on-abort would NameError
            shm.close()
            shm.unlink()  # always, in a finally (ADR-11 segment hygiene)
            if on_progress:
                on_progress(summary)
    return summary
