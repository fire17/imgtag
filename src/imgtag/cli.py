"""imgtag CLI — verbs index/info/manage/search/doctor (B20 machine-API law).

OWNER: b-engine. --json puts VALID JSON on stdout and nothing else; all human text
goes to stderr. Exit codes: 0 ok · 2 usage · 3 dataset-locked · 4 unknown-dataset ·
5 model/manifest-mismatch · 6 corrupt-index · 7 model-unavailable-offline.
Zero results above tau is exit 0 with ``no_match: true``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

from .core import store
from .core.doctor import autotune, load_profile, save_profile, usable_cores
from .core.progress import list_jobs, read_job

# Mapped BY NAME so `models` (and with it onnxruntime, ~15ms) stays out of the import
# path of info/status/manage — B20 gives info a 200ms budget and it should not be spent
# loading an inference runtime it never uses.
EXIT = {"LockedError": 3, "UnknownDatasetError": 4, "ModelMismatchError": 5,
        "CorruptIndexError": 6, "ModelUnavailableError": 7, "CalibrationMismatchError": 5,
        "ModerationTrackUnavailable": 7}


def _out(args, obj: dict, human: str = "") -> None:
    if args.json:
        json.dump(obj, sys.stdout)
        sys.stdout.write("\n")
    elif human:
        print(human)


def _err(msg: str) -> None:
    print(msg, file=sys.stderr)


# ---------------------------------------------------------------- verbs


def cmd_index(args) -> int:
    from .core.indexer import index

    t = time.time()
    ds = args.dataset_flag or args.dataset or Path(args.path).expanduser().resolve().name
    if not args.wait:
        # B20: `index` returns a job id in <=500ms and does not block. The real work
        # runs in a detached child that holds the dataset lock (ADR-6).
        import subprocess
        import uuid

        from .core.progress import Job

        job_id = uuid.uuid4().hex[:8]
        job = Job(job_id, ds, 0, root=str(Path(args.path).expanduser()), state="queued")
        argv = [sys.executable, "-m", "imgtag.cli", "index", str(args.path), "--dataset", ds,
                "--wait", "--job-id", job_id]
        for flag, v in (("--model", args.model), ("--geometry", args.geometry), ("--workers", args.workers),
                        ("--batch", args.batch), ("--intra-op", args.intra_op), ("--precision", args.precision),
                        ("--meta-csv", args.meta_csv), ("--moderation-hook", args.moderation_hook)):
            if v:
                argv += [flag, str(v)]
        if args.moderation:
            argv.append("--moderation")
        for kv in args.meta or []:
            argv += ["--meta", kv]
        if args.full_speed:
            argv.append("--full-speed")
        if args.no_recursive:
            argv.append("--no-recursive")
        child = subprocess.Popen(argv, start_new_session=True, stdin=subprocess.DEVNULL,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        job.state["pid"] = child.pid      # liveness becomes a kernel question, not a clock
        job._write(force=True)
        obj = {"job_id": job_id, "dataset": ds, "path": str(Path(args.path).expanduser()),
               "status": "queued", "state": "queued", "queued": True,
               "tookMs": round((time.time() - t) * 1000, 1)}
        _out(args, obj, f"job {job_id} queued for {ds} — poll: imgtag info --job {job_id}")
        return 0
    prof = load_profile()
    for k, v in (("geometry", args.geometry), ("workers", args.workers), ("batch", args.batch),
                 ("intra_op", args.intra_op), ("precision", args.precision)):
        if v:
            prof[k] = v
    if args.precision:
        prof["precision_explicit"] = True
    from .core.indexer import parse_meta

    s = index(
        args.path,
        ds,
        meta=parse_meta(args.meta),
        meta_csv=args.meta_csv,
        moderation=args.moderation_hook or args.moderation,
        backend=args.model,
        profile=prof,
        workers=args.workers,
        job_id=args.job_id,
        full_speed=args.full_speed,
        recursive=not args.no_recursive,
        log=_err if not args.json else (lambda m: None),
    )
    s["tookMs"] = round((time.time() - t) * 1000, 1)
    _out(
        args,
        s,
        f"indexed {s['indexed']} images ({s['skipped']} unchanged, {s['failed']} failed) "
        f"in {s['seconds']}s = {s['img_s']} img/s  [{s['model_id']}, {s['workers']} workers, "
        f"batch {s['batch']}, stages/img {s['stages_ms_per_img']}]"
        + (f"\n{_moderation_line(s['moderation'], s.get('moderation_active', False))}"
           if s.get("moderation_requested") else ""),
    )
    return 0


def _moderation_line(counts: dict, active: bool = True) -> str:
    from .core.indexer import moderation_summary

    return moderation_summary(counts or {}, active)


def _flags_rollup(dataset: str, limit: int = 200) -> dict:
    """Per-dataset moderation rollup, DERIVED from the stored score sidecars (ADR-15 T1).

    No model is loaded: raw scores + the tier spec recorded in the manifest produce the
    per-tier counts and the flagged list. A threshold change re-derives this for free.
    """
    from .core.indexer import MODERATION_TIERS, merge_counts

    snap = store.open_snapshot(dataset)
    flags = store.dataset_flags(dataset, snap=snap)
    counts: dict[str, dict] = {}
    per_image: dict[int, list] = {}
    for cat, d in flags.items():
        for i, tier in enumerate(d["tiers"]):
            if tier in MODERATION_TIERS:            # alert/violation/review — a real flag
                counts.setdefault(tier, {})
                counts[tier][cat] = counts[tier].get(cat, 0) + 1
                per_image.setdefault(i, []).append(
                    {"category": cat, "p": round(float(d["scores"][i]), 4), "tier": tier})
    flagged = []
    for i in sorted(per_image)[:limit]:
        r = snap.ids[i]
        flagged.append({"image_id": r["image_id"], "path": r["path"],
                        "dataset": r.get("dataset", dataset),
                        "categories": per_image[i], "meta": r.get("meta", {})})
    return {"dataset": dataset, "total": snap.count, "counts": merge_counts(counts),
            "calibration": (snap.manifest.get("moderation") or {}).get("calibration", "unfitted"),
            "enforcement_ready": (snap.manifest.get("moderation") or {}).get("enforcement_ready", False),
            "tracks": sorted(flags), "flagged": flagged, "truncated": len(per_image) > limit}


def _backend_names() -> list[str]:
    """Registry names straight from the bundled data file — listing backends must not
    import an inference runtime."""
    data = Path(__file__).resolve().parent / "data" / "backends.json"
    return sorted(k for k in json.loads(data.read_bytes()) if not k.startswith("_"))


def _daemon_state() -> dict:
    """Endpoint record published by the daemon (ADR-13); b-daemon owns writing it."""
    p = store.imgtag_home() / "daemon.json"
    sock = str(store.imgtag_home() / "daemon.sock")
    try:
        d = json.loads(p.read_bytes())
        up = round(max(0.0, time.time() - float(d.get("started_at") or 0)), 1) if d.get("started_at") else None
        return {"running": True, "pid": d.get("pid"), "version": d.get("version"),
                "uptimeSec": up, "socket": d.get("socket", sock), **d}
    except (OSError, ValueError):
        return {"running": False, "pid": None, "version": None, "uptimeSec": None, "socket": sock}


def cmd_info(args) -> int:
    t0 = time.perf_counter()
    args.dataset = getattr(args, "dataset_flag_info", None) or args.dataset
    if getattr(args, "image", None):
        # per-image all-tracks (B20 parity with GET /api/image/<ds>/<id>/tracks) —
        # delegate to the ONE owner (core.search.Searcher) so the object is identical
        from .core.search import Searcher

        s = Searcher()
        for ds in ([args.dataset] if args.dataset else store.list_datasets()):
            try:
                obj = s.image_tracks(ds, args.image)
            except (FileNotFoundError, store.UnknownDatasetError):
                continue
            obj["tookMs"] = round((time.perf_counter() - t0) * 1000, 2)
            _out(args, obj, "\n".join(
                f"{e['category']:14s} {e.get('tier') or 'none':10s} "
                f"p={e['p'] if e.get('scored') else 'pending'}" for e in obj["tracks"]))
            return 0
        _err(f"image {args.image!r} not found in any dataset")
        if args.json:
            json.dump({"error": "UnknownImage", "image_id": args.image, "exit": 4}, sys.stdout)
            sys.stdout.write("\n")
        return 4
    if getattr(args, "job", None):
        job = read_job(args.job)
        if job is None:
            _err(f"no such job {args.job}")
            if args.json:
                json.dump({"error": "UnknownJob", "job_id": args.job, "exit": 4}, sys.stdout)
                sys.stdout.write("\n")
            return 4
        job.setdefault("imgsPerSec", job.get("img_s", 0.0))   # b-skill's spelling
        job.setdefault("etaSec", job.get("eta_s"))
        job.setdefault("errors", job.get("failures", []))
        job["tookMs"] = round((time.perf_counter() - t0) * 1000, 2)
        _out(args, job, json.dumps(job, indent=1))
        return 0
    if getattr(args, "flags", False):
        names = [args.dataset] if args.dataset else store.list_datasets()
        rolls = [_flags_rollup(n) for n in names]
        from .core.indexer import merge_counts

        total = merge_counts({})
        for r in rolls:
            for tier, per_cat in r["counts"].items():
                for cat, n in per_cat.items():
                    total[tier][cat] = total[tier].get(cat, 0) + n
        obj = {"datasets": rolls, "counts": total, "tookMs": round((time.perf_counter() - t0) * 1000, 2)}
        _out(args, obj, "\n".join(f"{r['dataset']:24s} " + _moderation_line(r["counts"]) for r in rolls)
             or "no datasets")
        return 0
    if args.dataset:
        man = store.read_manifest(args.dataset)
        from .core.progress import annotate_stale

        jobs = annotate_stale([j for j in list_jobs() if j.get("dataset") == args.dataset][:5])
        obj = {"dataset": args.dataset, "manifest": man, "jobs": jobs, "daemon": _daemon_state(),
               "datasets": [{"dataset": args.dataset, "count": man["count"], "model_id": man["model_id"],
                             "model_sha": man["model_sha"], "dim": man["dim"],
                             "bytes": sum(s["emb_bytes"] + s["ids_bytes"] for s in man["shards"]),
                             "updated": man.get("updated")}],
               "tookMs": round((time.perf_counter() - t0) * 1000, 2)}
        _out(args, obj, f"{args.dataset}: {man['count']} images, model {man['model_id']} "
                        f"({man['model_sha'][:12]}), dim {man['dim']}, {len(man['shards'])} shard(s)")
        return 0
    ds = []
    for name in store.list_datasets():
        m = store.read_manifest(name)
        ds.append({"dataset": name, "count": m["count"], "model_id": m["model_id"],
                   "model_sha": m["model_sha"], "dim": m["dim"],
                   "updated": m.get("updated"), "bytes": sum(s["emb_bytes"] + s["ids_bytes"] for s in m["shards"])})
    from .core.progress import annotate_stale

    obj = {"datasets": ds, "jobs": annotate_stale(list_jobs(limit=10)), "home": str(store.imgtag_home()),
           "daemon": _daemon_state(), "profile": load_profile(), "backends": _backend_names(),
           "tookMs": round((time.perf_counter() - t0) * 1000, 2)}
    _out(args, obj, "\n".join(f"{d['dataset']:24s} {d['count']:>7} imgs  {d['model_id']}" for d in ds) or "no datasets")
    return 0


def _dir_bytes(d: Path) -> int:
    return sum(p.stat().st_size for p in d.rglob("*") if p.is_file())


def cmd_manage(args) -> int:
    home = store.imgtag_home()
    t0 = time.perf_counter()
    if args.action == "list":
        ds = [{"dataset": n, **{k: store.read_manifest(n)[k] for k in ("count", "model_id", "model_sha", "dim")}}
              for n in store.list_datasets()]
        _out(args, {"datasets": ds, "tookMs": round((time.perf_counter() - t0) * 1000, 2)},
             "\n".join(f"{d['dataset']:24s} {d['count']:>7} imgs  {d['model_id']}" for d in ds) or "no datasets")
        return 0
    if args.action == "meta":
        man = store.read_manifest(args.dataset, home)
        if args.set:
            from .core.indexer import parse_meta

            man["meta"] = {**man.get("meta", {}), **parse_meta(args.set)}
            man["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            d = store.dataset_dir(args.dataset, home)
            tmp = d / "manifest.json.tmp"
            tmp.write_text(json.dumps(man, indent=1))
            os.replace(tmp, d / "manifest.json")  # same atomic swap the writer uses
        _out(args, {"dataset": args.dataset, "meta": man.get("meta", {}),
                    "tookMs": round((time.perf_counter() - t0) * 1000, 2)},
             json.dumps(man.get("meta", {}), indent=1))
        return 0
    if args.action == "verify":
        # ADR-6 integrity: byte counts are the authority; recompute against them.
        man = store.read_manifest(args.dataset)
        d = store.dataset_dir(args.dataset, home)
        problems = []
        for sh in man["shards"]:
            for fn, key in ((sh["name"], "emb_bytes"), (store._ids_name(sh["name"]), "ids_bytes")):
                have = (d / fn).stat().st_size if (d / fn).exists() else -1
                if have < sh[key]:
                    problems.append({"file": fn, "expected": sh[key], "found": have})
        snap = store.open_snapshot(args.dataset, home)
        rows = len(snap.ids)
        if rows != man["count"]:
            problems.append({"file": "ids", "expected": man["count"], "found": rows})
        import numpy as np

        norms = np.linalg.norm(np.asarray(snap.emb), axis=1) if snap.count else np.ones(1)
        if not (0.99 <= float(norms.mean()) <= 1.01):
            problems.append({"file": "shards", "expected": "L2-normalized rows", "found": float(norms.mean())})
        obj = {"dataset": args.dataset, "count": man["count"], "ok": not problems, "problems": problems,
               "model_id": man["model_id"], "model_sha": man["model_sha"],
               "tookMs": round((time.perf_counter() - t0) * 1000, 2)}
        _out(args, obj, f"{args.dataset}: {'OK' if not problems else 'PROBLEMS: ' + json.dumps(problems)}")
        return 0 if not problems else 6
    if args.action == "create":
        d = store.dataset_dir(args.dataset, home)
        d.mkdir(parents=True, exist_ok=True)
        _out(args, {"dataset": args.dataset, "created": True, "path": str(d)}, f"created {d}")
    elif args.action == "rename":
        new = args.to or args.new
        if not new:
            _err("rename needs a target name: imgtag manage rename <old> <new>")
            return 2
        src, dst = store.dataset_dir(args.dataset, home), store.dataset_dir(new, home)
        if not (src / "manifest.json").is_file():
            raise store.UnknownDatasetError(args.dataset)
        if dst.exists():
            _err(f"{new} already exists")
            return 2
        src.rename(dst)
        man = json.loads((dst / "manifest.json").read_bytes())
        man["dataset"] = new
        (dst / "manifest.json").write_text(json.dumps(man, indent=1))
        _out(args, {"dataset": new, "renamed_from": args.dataset,
                    "tookMs": round((time.perf_counter() - t0) * 1000, 2)}, f"renamed -> {new}")
    elif args.action == "reindex":
        from .core.indexer import index

        d = store.dataset_dir(args.dataset, home)
        man = store.read_manifest(args.dataset)
        root = args.path or next((j.get("root") for j in list_jobs() if j.get("dataset") == args.dataset and j.get("root")), None)
        if not root:
            _err("reindex needs --path (no previous job recorded a root)")
            return 2
        shutil.rmtree(d)  # full rebuild: the model or the corpus changed
        s = index(root, args.dataset, backend=args.model, log=_err)
        _out(args, s, f"reindexed {s['indexed']} images from {root} ({s['img_s']} img/s)")
    elif args.action == "delete":
        d = store.dataset_dir(args.dataset, home)
        # A delete racing an index job must be UNAMBIGUOUS: refuse with the documented
        # dataset-locked code rather than delete bytes a live (or queued) writer is about
        # to recreate — that would silently undo the user's delete and leave orphans.
        from .core.progress import reap_stale, request_abort

        # LIVENESS = THE FLOCK, never a status file (ADR-6). A record frozen at "queued"
        # by a killed process must not be able to block a delete forever, which is exactly
        # what trusting the record did.
        for jid in reap_stale(args.dataset, home):
            _err(f"closed stale job record {jid} (writer lock is free — its process is gone)")
        live = [j for j in list_jobs()
                if j.get("dataset") == args.dataset and j.get("state") in ("queued", "running")]
        if live and not args.force:
            j = live[0]
            raise store.LockedError(
                f"{args.dataset} has an active job {j['job_id']} ({j['state']}) — wait for it "
                f"to finish, or `manage delete {args.dataset} --force` to abort it and delete"
            )
        if live:  # --force: abort first, THEN delete, so no writer can resurrect the dataset
            for j in live:
                request_abort(j["job_id"])
            deadline = time.time() + 20
            while time.time() < deadline:
                still = [j for j in list_jobs() if j.get("dataset") == args.dataset
                         and j.get("state") in ("queued", "running")]
                if not still:
                    break
                # a job that never took the lock (or died holding nothing) is a corpse:
                # re-test reality each round instead of waiting out the full timeout
                if reap_stale(args.dataset, home):
                    continue
                time.sleep(0.1)
            else:
                raise store.LockedError(
                    f"{args.dataset}: job(s) {[j['job_id'] for j in live]} did not stop within 20s — "
                    f"nothing deleted"
                )
            _err(f"aborted job(s) {', '.join(j['job_id'] for j in live)} before deleting")
        if not d.exists():
            raise store.UnknownDatasetError(args.dataset)
        lock = d / ".writer.lock"
        if lock.exists():  # kernel-owned truth, no pid heuristics (ADR-6)
            import fcntl

            fd = os.open(lock, os.O_RDWR)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                raise store.LockedError(f"{args.dataset} is being written right now — delete refused") from None
            finally:
                os.close(fd)
        freed = _dir_bytes(d)
        shutil.rmtree(d)  # leaves 0 orphan bytes: everything lives under this dir (B20)
        _out(args, {"dataset": args.dataset, "deleted": True, "freedBytes": freed,
                    "tookMs": round((time.perf_counter() - t0) * 1000, 2)},
             f"deleted {args.dataset} ({freed/1e6:.1f} MB freed)")
    return 0


def _search_via_daemon(args, t0: float):
    """Return an exit code if the daemon answered, else None (caller falls back)."""
    import http.client
    import socket
    import urllib.parse

    sock_p = store.imgtag_home() / "daemon.sock"
    qs = {"q": args.query, "k": args.k}
    if args.dataset:
        qs["dataset"] = args.dataset
    path = "/api/search?" + urllib.parse.urlencode(qs)

    def ask(timeout: float):
        # 8 lines of stdlib instead of `from .daemon import request`: importing the daemon
        # module pulls onnxruntime + core.search (~40ms) into the hot query path, which is
        # exactly the cost the daemon exists to remove.
        c = http.client.HTTPConnection("localhost", timeout=timeout)
        c.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        c.sock.settimeout(timeout)
        c.sock.connect(str(sock_p))
        try:
            c.request("GET", path)
            r = c.getresponse()
            return r.status, json.loads(r.read())
        finally:
            c.close()

    try:
        status, body = ask(30.0)
    except OSError:  # no daemon yet: ADR-13 says start one, then retry
        try:
            from .daemon import ensure_daemon

            if not ensure_daemon(backend=args.model):
                _err("daemon unavailable — answering in-process (cold model load)")
                return None
            status, body = ask(30.0)
        except Exception as e:
            _err(f"daemon start failed ({type(e).__name__}) — answering in-process")
            return None
    except Exception as e:  # socket died mid-flight: never fail the user's query (ADR-13)
        _err(f"daemon request failed ({type(e).__name__}) — answering in-process")
        return None
    if status != 200:
        # the daemon reports {error: message, code: ExceptionName, exit_code: n}
        b = body if isinstance(body, dict) else {}
        code = int(b.get("exit_code") or EXIT.get(b.get("code") or b.get("error"), 1))
        if args.json:
            json.dump(body if isinstance(body, dict) else {"error": "DaemonError", "exit": code}, sys.stdout)
            sys.stdout.write("\n")
        _err(f"error: {(body or {}).get('message', body)}")
        return code
    body["clientMs"] = round((time.perf_counter() - t0) * 1000, 2)
    # `served_by` names the TRANSPORT (which process answered) — that is what the ADR-13
    # client contract is about. The daemon's own label for its tower state ("warm-tower",
    # "cold-load") is preserved beside it instead of overwriting the transport answer.
    detail = body.get("served_by")
    body["served_by"] = "daemon"
    if detail and detail != "daemon":
        body["served_detail"] = detail
    _out(args, body, "\n".join(f"{h['score']:.4f}  {h['dataset']:12s} {h['image_id']}  {h['path']}"
                               for h in body.get("hits", [])) or "no match")
    return 0


def cmd_search(args) -> int:
    """ADR-13 client: talk to the resident daemon, else start one, else answer in-process.

    Both paths return the SAME schema because both are produced by the same owner —
    `core.search.Searcher` (b-daemon's file). The CLI has no query implementation of its
    own; a second scan here would mean two sigmoids, two sets of keys, and two answers to
    the same question. `--no-daemon` chooses the transport, never the semantics.
    """
    t0 = time.perf_counter()
    if args.dataset:
        store.read_manifest(args.dataset)   # a NAMED unknown dataset is exit 4, always —
        # "nothing is indexed" may not swallow "that dataset does not exist"
    if not args.dataset and not store.list_datasets():  # nothing indexed is an honest empty answer
        _out(args, {"query": args.query, "tookMs": round((time.perf_counter() - t0) * 1000, 2),
                    "coverage": {"indexed": 0, "total": 0}, "hits": [], "no_match": True},
             "no datasets indexed")
        return 0

    if not args.no_daemon:
        served = _search_via_daemon(args, t0)
        if served is not None:
            return served

    from .core.search import Searcher

    obj = Searcher().search(args.query, dataset=args.dataset, k=args.k)
    obj["clientMs"] = round((time.perf_counter() - t0) * 1000, 2)
    detail = obj.get("served_by")
    obj["served_by"] = "in-process"
    if detail and detail != "in-process":
        obj["served_detail"] = detail
    _out(args, obj, "\n".join(f"{h['score']:.4f}  {h['dataset']:12s} {h['image_id']}  {h['path']}"
                              for h in obj.get("hits", [])) or "no match")
    return 0


def cmd_track(args) -> int:
    """ADR-15: add/refresh ONE track column over an existing index (no re-embedding)."""
    from .core.indexer import track_add

    t0 = time.perf_counter()
    if args.action == "recount":
        # ADR-15 T1: re-derive stored counts from the sidecars under CURRENT fitted tau —
        # no re-embed, no re-score. Fixes indexes whose counts were empty at index time.
        # recount takes NO category, so its positional names the dataset: `recount <ds>`.
        from .core.indexer import track_recount

        target = args.dataset or args.category
        names = [target] if target else store.list_datasets()
        res = [track_recount(ds) for ds in names]
        _out(args, {"results": res, "tookMs": round((time.perf_counter() - t0) * 1000, 2)},
             "\n".join(f"{r['dataset']}: " + _moderation_line(r["moderation"]) for r in res))
        return 0
    if args.action == "list":
        out = []
        for ds in ([args.dataset] if args.dataset else store.list_datasets()):
            man = store.read_manifest(ds)
            out.append({"dataset": ds, "count": man["count"],
                        "tracks": {c: {k: t[k] for k in ("rows", "cols") if k in t}
                                   for c, t in (man.get("tracks") or {}).items()},
                        "spec": man.get("tracks_spec")})
        _out(args, {"datasets": out, "tookMs": round((time.perf_counter() - t0) * 1000, 2)},
             "\n".join(f"{d['dataset']:24s} {', '.join(d['tracks']) or '(none)'}" for d in out))
        return 0
    names = [args.dataset] if args.dataset else store.list_datasets()
    res = [track_add(ds, args.category, log=_err if not args.json else (lambda m: None)) for ds in names]
    _out(args, {"results": res, "tookMs": round((time.perf_counter() - t0) * 1000, 2)},
         "\n".join(f"{r['dataset']}: scored {r['scored']}/{r['rows']} for {r['track']} in {r['seconds']}s"
                   for r in res))
    return 0


def cmd_doctor(args) -> int:
    from .core import models

    if args.fetch:
        paths = models.fetch(args.fetch, log=_err)
        _out(args, {"fetched": [str(p) for p in paths]}, f"fetched {len(paths)} files")
        return 0
    cores, src = usable_cores()
    if args.show:
        prof = load_profile()
    else:
        _err(f"autotuning on {cores} usable cores ({src}) — ~30s ...")
        prof = autotune(args.model, log=_err if not args.json else (lambda m: None), allow_int8=args.allow_int8)
        save_profile(prof)
    _out(args, prof,
         f"cores={prof['cores']} ({prof['cores_source']}) workers={prof['workers']} "
         f"precision={prof['precision']} intra_op={prof['intra_op']} batch={prof['batch']} "
         f"geometry={prof['geometry']} measured={prof['measured']}")
    return 0


def cmd_status(args) -> int:
    """ADR-13: daemon pid/version/socket/uptime + what the engine would load."""
    t0 = time.perf_counter()
    prof = load_profile()
    obj = {"daemon": _daemon_state(), "home": str(store.imgtag_home()),
           "datasets": [{"dataset": n, "count": store.read_manifest(n)["count"]} for n in store.list_datasets()],
           "profile": {k: prof.get(k) for k in ("precision", "intra_op", "batch", "workers", "geometry",
                                                "cores", "cores_source", "measured")},
           "version": __import__("imgtag").__version__ if hasattr(__import__("imgtag"), "__version__") else "0.1.0",
           "jobs_running": [j["job_id"] for j in list_jobs() if j.get("state") == "running"],
           "tookMs": round((time.perf_counter() - t0) * 1000, 2)}
    _out(args, obj, f"daemon: {'up' if obj['daemon']['running'] else 'down'} · "
                    f"{len(obj['datasets'])} dataset(s) · profile {obj['profile']}")
    return 0


def cmd_job(args) -> int:
    obj = read_job(args.job_id) if args.job_id else {"jobs": list_jobs()}
    if obj is None:
        _err(f"no such job {args.job_id}")
        return 4
    _out(args, obj, json.dumps(obj, indent=1))
    return 0


# ---------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    # --json is accepted before OR after the verb (agents write it either way)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", help="machine output on stdout (B20)")
    p = argparse.ArgumentParser("imgtag", description="CPU-only semantic image search", parents=[common])
    sub = p.add_subparsers(dest="verb", required=True)

    i = sub.add_parser("index", help="index a folder into a dataset", parents=[common])
    i.add_argument("path")
    i.add_argument("dataset", nargs="?", help="dataset slug (defaults to the folder name)")
    i.add_argument("--dataset", dest="dataset_flag")
    i.add_argument("--wait", action="store_true", help="run in the foreground instead of returning a job id")
    i.add_argument("--job-id", dest="job_id", metavar="ID",
                   help="use this job id instead of minting one (the daemon mints ids so it "
                        "can answer POST /api/index inside B20's 500ms); [0-9A-Za-z_-]{1,32}")
    i.add_argument("--model")
    i.add_argument("--full-speed", action="store_true")
    i.add_argument("--no-recursive", action="store_true")
    i.add_argument("--geometry", choices=["central", "worker"], help="override the tuned pipeline geometry")
    i.add_argument("--workers", type=int)
    i.add_argument("--batch", type=int)
    i.add_argument("--intra-op", type=int, dest="intra_op")
    i.add_argument("--precision", choices=["fp32", "fp16", "int8"])
    i.add_argument("--meta", action="append", metavar="KEY=VALUE",
                   help="metadata applied to every image in this job (repeatable)")
    i.add_argument("--meta-csv", dest="meta_csv", metavar="FILE",
                   help="per-image metadata: CSV with a path/filename column + any other columns")
    i.add_argument("--moderation", action="store_true",
                   help="run the moderation tracks (imgtag.moderation.load_heads) during indexing")
    i.add_argument("--moderation-hook", dest="moderation_hook", metavar="MODULE:FN",
                   help="override the dispatcher with one detect() callable (dev/tests)")
    i.set_defaults(fn=cmd_index)

    n = sub.add_parser("info", help="datasets, jobs, profile", parents=[common])
    n.add_argument("dataset", nargs="?")
    n.add_argument("--dataset", dest="dataset_flag_info")
    n.add_argument("--job", help="report one job's live status")
    n.add_argument("--flags", action="store_true", help="moderation rollup (all datasets, or one with --dataset)")
    n.add_argument("--image", metavar="IMAGE_ID", help="per-image all-tracks panel (with --tracks)")
    n.add_argument("--tracks", action="store_true", help="with --image: every track's confidence for it")
    n.set_defaults(fn=cmd_info)

    m = sub.add_parser("manage", help="create/rename/reindex/delete a dataset", parents=[common])
    m.add_argument("action", choices=["list", "create", "rename", "reindex", "delete", "verify", "meta"])
    m.add_argument("dataset", nargs="?")
    m.add_argument("new", nargs="?", help="target name for `manage rename <old> <new>`")
    m.add_argument("--to")
    m.add_argument("--path")
    m.add_argument("--model")
    m.add_argument("--yes", "-y", action="store_true", help="no-op: imgtag never prompts (B20)")
    m.add_argument("--set", action="append", metavar="KEY=VALUE",
                   help="meta: set dataset-level metadata (repeatable)")
    m.add_argument("--force", action="store_true",
                   help="delete: abort any in-flight job for this dataset first, then delete")
    m.set_defaults(fn=cmd_manage)

    s = sub.add_parser("search", help="semantic search", parents=[common])
    s.add_argument("query")
    s.add_argument("--dataset")
    s.add_argument("-k", type=int, default=50)
    s.add_argument("--model")
    s.add_argument("--no-daemon", action="store_true",
                   help="answer in-process instead of via the resident daemon (ADR-13 fallback)")
    s.set_defaults(fn=cmd_search)

    d = sub.add_parser("doctor", help="autotune this machine (~30s)", parents=[common])
    d.add_argument("--show", action="store_true", help="print the stored profile, do not re-tune")
    d.add_argument("--model")
    d.add_argument("--fetch", metavar="BACKEND", help="download a backend's artifacts")
    d.add_argument("--allow-int8", action="store_true",
                   help="let the tune select int8 vision (B24: opt-in speed lane, v1 default is fp32)")
    d.set_defaults(fn=cmd_doctor)

    tr = sub.add_parser("track", help="add/refresh a track column over an existing index",
                        parents=[common])
    tr.add_argument("action", choices=["add", "list", "recount"])
    tr.add_argument("category", nargs="?", help="track name, e.g. weapons")
    tr.add_argument("--dataset", help="one dataset (default: every dataset on disk)")
    tr.set_defaults(fn=cmd_track)

    st = sub.add_parser("status", help="daemon + engine state (ADR-13)", parents=[common])
    st.set_defaults(fn=cmd_status)

    j = sub.add_parser("job", help="job status", parents=[common])
    j.add_argument("job_id", nargs="?")
    j.set_defaults(fn=cmd_job)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except Exception as e:
        code = EXIT.get(type(e).__name__)
        if code is None:
            raise
        if args.json:
            json.dump({"error": type(e).__name__, "message": str(e), "exit": code}, sys.stdout)
            sys.stdout.write("\n")
        _err(f"error: {e}")
        return code
    except KeyboardInterrupt:
        _err("interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
