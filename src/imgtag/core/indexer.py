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
import json
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

MODERATION_CATEGORIES = ("nudity", "weapons", "drugs", "safety")
#: ADR-14 / VISION-ADDENDA 14:16Z tier ranking: alert > violation > review > match > none.
#: ENFORCEMENT tiers drive the "Found N images with drugs…" summary and the moderation
#: totals. CONTENT tiers (match) are per-image category chips for the detail view — a
#: counting/content track is not a policy violation, so its tiers route to a SEPARATE
#: content bucket and never pollute the enforcement counts.
ENFORCEMENT_TIERS = ("alert", "violation", "review")
CONTENT_TIERS = ("match",)
#: ADR-14 amendment: "alert" is the highest tier (safety track — person-down + danger
#: context); plain person-down is "review". Ordered highest-first for reporting.
MODERATION_TIERS = ENFORCEMENT_TIERS  # back-compat alias (empty_counts / summary shape)


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

    warned: set[str] = set()   # deprecation notices fire ONCE per job, not per batch

    def detect(embs, recs, images=None, views=None):
        per: list[list[dict]] = [[] for _ in recs]
        for name, head in heads.items():
            try:
                out = _call_head(head, embs, images, recs, name, warned, log, views) or []
            except Exception as e:
                log(f"moderation head {name!r} raised {type(e).__name__}: {e} — skipped for this batch")
                continue
            for i, flags in enumerate(out[: len(per)]):
                per[i].extend(f for f in as_flag_list(flags) if f)
        return per

    detect.wants_images = any(getattr(h, "wants_images", False) or hasattr(h, "score_images")
                              for h in heads.values())
    detect.heads = heads
    detect.specs = {c: track_tier_spec(c, h) for c, h in heads.items()}
    detect.scorer_meta = {c: track_scorer_meta(c, h) for c, h in heads.items()}
    return detect


def as_flag_list(x) -> list[dict]:
    """One head's answer for ONE image, normalized to a list.

    The seam says `-> [{category, p, tier}]`, i.e. one dict PER IMAGE; a multi-signal
    track may return several. Accepting both shapes is what stops a single-dict return
    from being iterated as its own keys (a str has no .get — loudly, but pointlessly).
    """
    if x is None:
        return []
    return list(x) if isinstance(x, (list, tuple)) else [x]


def _call_head(head, embs, images, recs, name="head", warned=None, log=lambda m: None, views=None):
    """Call a track head through the ONE documented seam, with one tolerated legacy path.

    Seam (ruling 2026-07-22): `score(embeddings, images, ids) -> [{category, p, tier}]`.
    TOLERATED UNTIL THE TRACK CONFORMS: `score_images(list[PIL.Image])`, which is
    unambiguous. A bare `score(batch)` is NOT guessed at — feeding an image model's
    tensor input with embeddings would silently produce numbers that mean nothing.
    """
    try:
        if views is not None and getattr(head, "view_key", None) in views:
            # the free-view fast path: hand the pre-made view so the head skips its re-open
            return head.score(embs, images, recs, views=views)
        return head.score(embs, images, recs)          # the documented seam
    except TypeError:
        if not hasattr(head, "score_images"):
            raise
    if warned is not None and f"seam:{name}" not in warned:
        warned.add(f"seam:{name}")
        log(f"DEPRECATED: moderation head {name!r} has no score(embeddings, images, ids) — "
            f"using its legacy score_images(); the track should migrate to the documented seam")
    if images is None:                                  # legacy PIL-only head
        raise RuntimeError("head needs pixels but this batch has none (geometry=worker)")
    from PIL import Image as _Image

    return head.score_images([_Image.fromarray(im) for im in images])


def track_scorer_meta(category: str, head=None) -> dict:
    """Provenance for a track's sidecar header: which model produced the scores."""
    return {"scorer": getattr(head, "model_id", None) if head is not None else None,
            "model_sha": getattr(head, "model_sha", None) if head is not None else None}


def track_tier_spec(category: str, head=None) -> dict:
    """The τ bands that DERIVE tiers for one track (ADR-15 T3: versioned spec, not code).

    Read from the head's fitted attributes first, then the category's entry in
    data/moderation.json, so the read path can derive tiers WITHOUT loading a model. A
    band that is not yet fitted is simply absent — its scores derive to "unknown", never
    a silently-passing "none" (enforcement_ready stays false until τ is fitted).
    """
    spec = {}
    for k in ("tau_alert", "tau_violation", "tau_review", "tau"):
        v = getattr(head, k, None) if head is not None else None
        if v is not None:
            spec[k] = float(v)
    try:
        import json

        cat = json.loads((models.DATA / "moderation.json").read_bytes()).get("categories", {}).get(category, {})
        for k in ("tau_alert", "tau_violation", "tau_review", "tau"):
            spec.setdefault(k, float(cat[k])) if k in cat else None
    except Exception:
        pass
    return spec


def _apply_moderation(hook, embs, recs, images, counts: dict, log, warned=None, views=None) -> tuple[int, dict]:
    """Collect RAW per-image scores for every track (ADR-15 T1) and count derived tiers.

    Returns ``(errors, columns)`` where columns maps category -> f32[len(recs)]: one score
    per image, NaN where a track did not answer. SCORES, not flags, are what gets stored —
    a later policy change then re-derives tiers for free (T1), and the per-tier counts
    computed here are only for the job summary, never the durable answer.
    """
    n = len(recs)
    # a dense NaN column per KNOWN category up front: every image gets a slot in every
    # track, so the sidecars stay row-aligned even in a batch where a track never fires
    known = list(getattr(hook, "specs", {}) or getattr(hook, "heads", {}) or [])
    columns: dict[str, np.ndarray] = {c: np.full(n, np.float32("nan"), np.float32) for c in known}
    try:
        wants = getattr(hook, "wants_images", False)
        imgs = images if (images is not None or not wants) else None
        out = hook(embs, recs, imgs, views) if views is not None else hook(embs, recs, imgs)
    except Exception as e:  # never let a detector break the index
        log(f"moderation hook raised {type(e).__name__}: {e} — scores skipped for this batch")
        return 1, columns

    for i, flags in enumerate(out or []):
        if i >= n:
            break
        for f in flags or []:
            cat = f.get("category")
            if not cat:
                continue
            col = columns.setdefault(cat, np.full(n, np.float32("nan"), np.float32))
            col[i] = float(f.get("p", float("nan")))
            # ADR-14/15: tier is the carrier. Legacy {flagged: bool} maps to violation so
            # an older track still counts for something instead of vanishing silently.
            tier = f.get("tier")
            if tier is None:
                tier = "violation" if f.get("flagged") else "none"
                if warned is not None and "tier" not in warned:
                    warned.add("tier")
                    log(f"DEPRECATED: moderation flag for {cat!r} has no ADR-14 `tier` — "
                        f"mapping flagged->violation; the track should emit tier")
            if tier != "none":
                counts.setdefault(tier, {})
                counts[tier][cat] = counts[tier].get(cat, 0) + 1
    return 0, columns


def empty_counts() -> dict:
    return {t: {c: 0 for c in MODERATION_CATEGORIES} for t in MODERATION_TIERS}


def accumulate(total: dict, add: dict) -> dict:
    """total += add over the {tier: {category: n}} shape, defensively — an UNKNOWN tier
    (a track lane shipping a new tier before the engine knows it) is counted under its own
    name, never a KeyError. Forward-compatible, exactly like b-daemon's reader."""
    for tier, per_cat in (add or {}).items():
        t = total.setdefault(tier, {})
        for cat, n in (per_cat or {}).items():
            t[cat] = t.get(cat, 0) + n
    return total


def merge_counts(counts: dict) -> dict:
    """Fill the fixed enforcement shape so consumers never branch on a missing key."""
    return accumulate(empty_counts(), counts)


def split_tiers(counts: dict) -> tuple[dict, dict]:
    """(enforcement, content) — content (match + any tier outside ENFORCEMENT_TIERS) is
    kept apart so it can never inflate the moderation totals or the summary line."""
    enf, content = {}, {}
    for tier, per_cat in (counts or {}).items():
        (enf if tier in ENFORCEMENT_TIERS else content)[tier] = dict(per_cat or {})
    return enf, content


def _moderation_spec_version():
    try:
        import json

        return json.loads((models.DATA / "moderation.json").read_bytes()).get("version")
    except Exception:
        return None


def moderation_summary(counts: dict, active: bool = True) -> str:
    """The job-end line in the user's phrasing, with ADR-14's tiers visible. Alerts lead
    when present — the whole point of the tier is that it cannot be scrolled past."""
    if not active:
        return "moderation: off (no tracks loaded)"
    c = merge_counts(counts)
    order = list(MODERATION_CATEGORIES) + [k for k in sorted(c.get("violation", {}))
                                           if k not in MODERATION_CATEGORIES]
    line = "Found " + ", ".join(
        f"{c['violation'].get(k, 0)} images with {k}" + (f" ({r} for review)" if (r := c["review"].get(k, 0)) else "")
        for k in order)
    if (n := sum(c.get("alert", {}).values())):
        who = ", ".join(f"{v} {k}" for k, v in sorted(c["alert"].items()) if v)
        line = f"\u26a0 {n} ALERTS ({who}) \u00b7 " + line
    return line


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


def _worker(task_q, ready_q, free_q, shm_name, size, squash, resample_name, sem, nice_level, worker_be=None,
            known_ids=frozenset(), view_spec=None):
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
    make_view = None
    if view_spec and view_spec.get("size") == size:  # bit-parity precondition (make_view
        import importlib                              # docstring): our decode is drafted@size

        mod, _, fn = view_spec["builder"].partition(":")
        try:
            make_view = getattr(importlib.import_module(mod), fn)
        except (ImportError, AttributeError):
            make_view = None
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
                if image_id in known_ids:
                    # same BYTES already in this dataset: never decode, never embed, never
                    # append a second row (IA: duplicates collapse to one row; B12: no
                    # re-embedding of content we already have)
                    ready_q.put({"duplicate": True, "image_id": image_id, "path": str(p), "pid": os.getpid()})
                    continue
                im = Image.open(io.BytesIO(buf))
                w, h = im.size
                if heavy:
                    sem.acquire()
                try:
                    arr = preprocess_image(im, size, squash, resample)
                    # the moderation view, built off the SAME drafted decode (zero extra
                    # decode, one resize) — only when the draft scale matches (gated above)
                    mod_view = make_view(im).tobytes() if make_view is not None else None
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
                        "mod_view": mod_view,
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
    mod_warned: set[str] = set()
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
    view_spec = None
    if hook is not None:
        for h in (getattr(hook, "heads", {}) or {}).values():
            g = getattr(h, "view_geometry", None)
            vk = getattr(h, "view_key", None)
            builder = getattr(h, "view_builder", None) or (
                "imgtag.moderation.nudity:make_view" if vk == "nudity-384crop" else None)
            # PRECONDITION (make_view docstring): only when the backend's decode is drafted
            # at the head's required scale — otherwise the view differs and we re-open
            if g and vk and builder and g.get("requires_draft") == be.size:
                view_spec = {"builder": builder, "key": vk, "size": be.size}
                break
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
    known_ids: frozenset = frozenset()
    try:
        snap = open_snapshot(dataset, home)
        known = {r["path"]: (r.get("mtime", -1), r.get("size", -1)) for r in snap.ids}
        known_ids = frozenset(r["image_id"] for r in snap.ids)
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
        if hook is not None:
            for c in getattr(hook, "specs", {}):
                w._track_specs[c] = {**(hook.specs.get(c) or {}), **(getattr(hook, "scorer_meta", {}).get(c) or {})}
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
                          0 if full_speed else 10, worker_be, known_ids, view_spec),
                    daemon=True,
                )
                pr.start()
                procs.append(pr)
            job.start()
            log(f"indexing {len(todo)} images ({skipped} unchanged) with {workers} workers, "
                f"{be.model_id}, batch {batch}, intra_op {prof.get('intra_op')}")

            received = 0
            duplicates = 0
            pend_views: list = []
            seen_ids = set(known_ids)   # + everything this job appends, so a folder that
            # contains the same photo twice still yields exactly one row
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
                cols = None
                if hook is not None:
                    views = None
                    if view_spec and pend_views and all(v is not None for v in pend_views):
                        views = {view_spec["key"]: np.stack([
                            np.frombuffer(v, np.uint8).reshape(view_spec["size"], view_spec["size"], 3)
                            for v in pend_views])}
                    err, cols = _apply_moderation(hook, embs, pend_recs, images, mod_counts, log,
                                                  mod_warned, views=views)
                    mod_errors += err
                w.append(embs, pend_recs.copy(), tracks=cols)
                for s in pend_slots:
                    free_q.put(s)
                pend_slots.clear()
                pend_embs.clear()
                pend_recs.clear()
                pend_views.clear()

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
                if msg.get("duplicate"):
                    duplicates += 1
                    continue
                if "error" in msg:
                    job.add_failure(msg["path"], msg["error"])
                else:
                    t_decode += msg.get("decode_ms", 0.0)
                    t_infer += msg.get("infer_ms", 0.0) / 1000.0
                    # dedup BEFORE the batch buffers: two workers can carry the same
                    # content concurrently, and appending its embedding first would leave
                    # embs and recs different lengths
                    if msg["image_id"] in seen_ids:
                        duplicates += 1
                        if msg.get("slot") is not None:
                            free_q.put(msg["slot"])
                        job.update(w.count, inflight=received - w.count)
                        continue
                    seen_ids.add(msg["image_id"])
                    if msg.get("emb") is not None:
                        pend_embs.append(np.frombuffer(msg["emb"], np.float32))
                    else:
                        pend_slots.append(msg["slot"])
                    pend_views.append(msg.get("mod_view"))
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
            _enf, _content = split_tiers(mod_counts)  # enforcement vs content (match) tiers
            summary = {
                "job_id": w.job_id,
                "dataset": dataset,
                "indexed": n - prior,
                "count": n,
                "skipped": skipped,
                "duplicates": duplicates,
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
                "moderation": merge_counts(_enf),
                "content": _content,
                "moderation_active": hook is not None,
                "moderation_requested": bool(moderation),
                "moderation_errors": mod_errors,
            }
            if job_meta:  # dataset-level metadata lives in the manifest, merged across jobs
                w.manifest["meta"] = {**w.manifest.get("meta", {}), **job_meta}
            if hook is not None:
                w.manifest["tracks_spec"] = {
                    # WHICH spec produced these scores — derivation at read is only
                    # reproducible if the version that scored them is recorded (T3)
                    "moderation_version": _moderation_spec_version(),
                    "categories": sorted(w.manifest.get("tracks", {})),
                    "tiers": getattr(hook, "specs", {}),
                }
                stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                prev = (w.manifest.get("moderation") or {}).get("counts", {})
                w.manifest["moderation"] = {
                    "counts": accumulate(merge_counts(prev), _enf), "schema": "ADR-14",
                    # honesty: tau is unfitted until b-bench fits it on labeled ground truth
                    "calibration": "unfitted", "enforcement_ready": False, "updated": stamp,
                }
                if _content:  # per-image content chips (match tier) — a separate bucket,
                    # NEVER folded into the enforcement totals or the "Found…" summary
                    prevc = (w.manifest.get("content") or {}).get("counts", {})
                    w.manifest["content"] = {
                        "counts": accumulate(accumulate({}, prevc), _content),
                        "tiers": list(CONTENT_TIERS), "updated": stamp,
                    }
            if job_meta or hook is not None:
                w._commit()
            if moderation:
                log(moderation_summary(_enf, active=hook is not None))
            job.update(w.count, 0, summary["stages_ms_per_img"], force=True)
            job.state["moderation"] = summary["moderation"]
            job.state["content"] = _content
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


# ---------------------------------------------------------------- track backfill


def track_add(dataset: str, category: str, home: Path | None = None, profile: dict | None = None,
              job_id: str | None = None, log=lambda m: None, batch: int = 256) -> dict:
    """Score ONE track over an existing index — no re-embedding (ADR-15 T3 / `track add`).

    Embedding-space tracks read the shards that are already on disk, so adding track #101
    is a matvec pass over f32 rows: milliseconds per thousand images, and indexing time for
    everyone else is unchanged. A track that needs pixels re-reads only its own images.
    """
    from .doctor import load_profile
    from .progress import Job
    from .store import open_snapshot

    prof = dict(profile or load_profile(home))
    snap = open_snapshot(dataset, home)
    heads = load_moderation_hook(True, prof, log)
    if heads is None or category not in getattr(heads, "heads", {}):
        raise ModerationTrackUnavailable(
            f"no head for track {category!r} — available: "
            f"{sorted(getattr(heads, 'heads', {}) or [])}")
    head = heads.heads[category]
    spec = {**(getattr(heads, "specs", {}).get(category) or {}),
            **(getattr(heads, "scorer_meta", {}).get(category) or {})}
    n = snap.count
    job = Job(job_id or f"track-{category}", dataset, n, home=home, kind="track-add", track=category)
    job.start()
    scores = np.full(n, np.float32("nan"), np.float32)
    t0 = time.time()
    try:
        for i in range(0, n, batch):
            recs = snap.ids[i : i + batch]
            embs = np.ascontiguousarray(np.asarray(snap.emb)[i : i + batch], np.float32)
            images = None
            if getattr(head, "wants_images", False):
                images = None  # the head re-reads its own pixels from rec["path"]
            out = _call_head(head, embs, images, recs, category, set(), log) or []
            for j, flags in enumerate(out):
                for f in as_flag_list(flags):
                    if f.get("category") == category:
                        scores[i + j] = float(f.get("p", float("nan")))
            job.update(min(i + batch, n))
        _write_track_column(dataset, category, scores, home, spec)
        job.finish(indexed=n, seconds=round(time.time() - t0, 2))
    except BaseException as e:
        job.fail(f"{type(e).__name__}: {e}")
        raise
    return {"dataset": dataset, "track": category, "rows": n, "job_id": job.state["job_id"],
            "seconds": round(time.time() - t0, 2), "scored": int((~np.isnan(scores)).sum())}


class ModerationTrackUnavailable(RuntimeError):
    """No head can score the requested track on this machine (CLI exit 7)."""


def _write_track_column(dataset: str, category: str, scores: np.ndarray, home=None, spec=None) -> None:
    """Swap in a whole track column atomically, under the dataset's writer lock.

    Backfill replaces one column and nothing else — no shard is touched, no embedding is
    recomputed, and readers either see the old column or the new one (ADR-6's rename).
    """
    import fcntl

    from .store import TRACKS_DIR, dataset_dir

    d = dataset_dir(dataset, home)
    lock = d / ".writer.lock"
    fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        from .store import LockedError

        raise LockedError(f"{dataset} is being written — track backfill refused") from None
    try:
        (d / TRACKS_DIR).mkdir(exist_ok=True)
        tmp = d / TRACKS_DIR / f"{category}.f32.tmp"
        with open(tmp, "wb") as f:
            f.write(np.ascontiguousarray(scores, np.float32).tobytes())
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, d / TRACKS_DIR / f"{category}.f32")
        man = json.loads((d / "manifest.json").read_bytes())
        man.setdefault("tracks", {})[category] = {
            "name": f"{category}.f32", "rows": int(scores.shape[0]), "cols": 1,
            "bytes": int(scores.nbytes), "backfilled": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        mtmp = d / "manifest.json.tmp"
        mtmp.write_text(json.dumps(man, indent=1))
        os.replace(mtmp, d / "manifest.json")
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
