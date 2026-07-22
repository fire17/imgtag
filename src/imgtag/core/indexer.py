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
MAX_PIXELS = 80_000_000  # B21: the decompression bomb is refused by a pixel cap, not by OOM
FILE_TIMEOUT_S = 5  # B21: every file resolved or skipped within 5s


def scan(root: Path, recursive: bool = True) -> list[Path]:
    root = Path(root).expanduser()
    if root.is_file():
        return [root]
    it = root.rglob("*") if recursive else root.glob("*")
    return sorted(p for p in it if p.suffix.lower() in IMAGE_EXTS and p.is_file())


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
            heavy = p.suffix.lower() not in JPEG_EXTS  # PNG/HEIC full decodes are the RAM term
            slot = None
            try:
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
                ready_q.put({"error": f"{type(e).__name__}: {e}"[:200], "path": str(p), "pid": os.getpid()})
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
    recursive: bool = True,
    log=lambda m: None,
    on_progress=None,
) -> dict:
    """Index a folder into a dataset. Returns a summary dict (also on the job file)."""
    from multiprocessing.shared_memory import SharedMemory

    files = scan(Path(path), recursive)
    prof = dict(profile or load_profile(home))
    name = backend or prof.get("backend") or models.DEFAULT_BACKEND
    geometry = prof.get("geometry", "central")
    be = models.load_backend(name, prof, vision=(geometry != "worker"))
    workers = workers or worker_count(prof.get("cores"), prof.get("mem_available_mb"), full_speed, geometry)
    batch = int(prof.get("batch", 2))
    nslots = min(128, max(16, 4 * batch, 4 * workers))
    slot_bytes = be.size * be.size * 3
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

    with Writer(dataset, be, home=home) as w:
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
            t_start = time.time()

            def flush_batch():
                nonlocal t_infer
                if not pend_recs:
                    return
                if worker_be:  # workers already embedded; we only store
                    embs = np.stack(pend_embs)
                else:
                    t = time.perf_counter()
                    embs = be.embed_images(np.stack([slab[s] for s in pend_slots]))
                    t_infer += time.perf_counter() - t
                w.append(embs, pend_recs.copy())
                for s in pend_slots:
                    free_q.put(s)
                pend_slots.clear()
                pend_embs.clear()
                pend_recs.clear()

            while received < len(todo):
                t = time.perf_counter()
                try:
                    msg = ready_q.get(timeout=0.5)
                except _q.Empty:
                    t_queue += time.perf_counter() - t
                    flush_batch()  # never let a partial batch sit while the queue starves
                    job.update(w.count, in_flight=received - w.count)
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
                    pend_recs.append(
                        {
                            "image_id": msg["image_id"],
                            "path": msg["path"],
                            "dataset": dataset,
                            "w": msg["w"],
                            "h": msg["h"],
                            "mtime": msg["mtime"],
                            "size": msg["size"],
                        }
                    )
                    if len(pend_recs) >= batch:
                        flush_batch()
                job.update(w.count, in_flight=received - w.count)
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
            }
            job.update(w.count, 0, summary["stages_ms_per_img"], force=True)
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
            del slab
            shm.close()
            shm.unlink()  # always, in a finally (ADR-11 segment hygiene)
            if on_progress:
                on_progress(summary)
    return summary
