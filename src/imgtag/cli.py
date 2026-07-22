"""imgtag CLI — verbs index/info/manage/search/doctor (B20 machine-API law).

OWNER: b-engine. --json puts VALID JSON on stdout and nothing else; all human text
goes to stderr. Exit codes: 0 ok · 2 usage · 3 dataset-locked · 4 unknown-dataset ·
5 model/manifest-mismatch · 6 corrupt-index · 7 model-unavailable-offline.
Zero results above tau is exit 0 with ``no_match: true``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from .core import models, store
from .core.doctor import autotune, load_profile, save_profile, usable_cores
from .core.progress import list_jobs, read_job

EXIT = {
    store.LockedError: 3,
    store.UnknownDatasetError: 4,
    store.ModelMismatchError: 5,
    store.CorruptIndexError: 6,
    models.ModelUnavailableError: 7,
}
NO_MATCH_FLOOR = 0.20  # provisional dense floor; b-daemon replaces it with the calibrated tau


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
        Job(job_id, ds, 0, root=str(Path(args.path).expanduser()), state="queued")
        argv = [sys.executable, "-m", "imgtag.cli", "index", str(args.path), "--dataset", ds,
                "--wait", "--job-id", job_id]
        for flag, v in (("--model", args.model), ("--geometry", args.geometry), ("--workers", args.workers),
                        ("--batch", args.batch), ("--intra-op", args.intra_op), ("--precision", args.precision)):
            if v:
                argv += [flag, str(v)]
        if args.full_speed:
            argv.append("--full-speed")
        if args.no_recursive:
            argv.append("--no-recursive")
        subprocess.Popen(argv, start_new_session=True, stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        obj = {"job_id": job_id, "dataset": ds, "state": "queued",
               "tookMs": round((time.time() - t) * 1000, 1)}
        _out(args, obj, f"job {job_id} queued for {ds} — poll: imgtag info --job {job_id}")
        return 0
    prof = load_profile()
    for k, v in (("geometry", args.geometry), ("workers", args.workers), ("batch", args.batch),
                 ("intra_op", args.intra_op), ("precision", args.precision)):
        if v:
            prof[k] = v
    s = index(
        args.path,
        ds,
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
        f"batch {s['batch']}, stages/img {s['stages_ms_per_img']}]",
    )
    return 0


def _daemon_state() -> dict:
    """Endpoint record published by the daemon (ADR-13); b-daemon owns writing it."""
    p = store.imgtag_home() / "daemon.json"
    try:
        d = json.loads(p.read_bytes())
        return {"running": True, **d}
    except (OSError, ValueError):
        return {"running": False, "socket": str(store.imgtag_home() / "daemon.sock")}


def cmd_info(args) -> int:
    t0 = time.perf_counter()
    if getattr(args, "job", None):
        job = read_job(args.job)
        if job is None:
            _err(f"no such job {args.job}")
            if args.json:
                json.dump({"error": "UnknownJob", "job_id": args.job, "exit": 4}, sys.stdout)
                sys.stdout.write("\n")
            return 4
        job["tookMs"] = round((time.perf_counter() - t0) * 1000, 2)
        _out(args, job, json.dumps(job, indent=1))
        return 0
    if args.dataset:
        man = store.read_manifest(args.dataset)
        jobs = [j for j in list_jobs() if j.get("dataset") == args.dataset][:5]
        obj = {"dataset": args.dataset, "manifest": man, "jobs": jobs, "daemon": _daemon_state(),
               "datasets": [{"dataset": args.dataset, "count": man["count"], "model_id": man["model_id"]}],
               "tookMs": round((time.perf_counter() - t0) * 1000, 2)}
        _out(args, obj, f"{args.dataset}: {man['count']} images, model {man['model_id']} "
                        f"({man['model_sha'][:12]}), dim {man['dim']}, {len(man['shards'])} shard(s)")
        return 0
    ds = []
    for name in store.list_datasets():
        m = store.read_manifest(name)
        ds.append({"dataset": name, "count": m["count"], "model_id": m["model_id"], "dim": m["dim"],
                   "updated": m.get("updated"), "bytes": sum(s["emb_bytes"] + s["ids_bytes"] for s in m["shards"])})
    obj = {"datasets": ds, "jobs": list_jobs(limit=10), "home": str(store.imgtag_home()),
           "daemon": _daemon_state(), "profile": load_profile(), "backends": sorted(models.registry()),
           "tookMs": round((time.perf_counter() - t0) * 1000, 2)}
    _out(args, obj, "\n".join(f"{d['dataset']:24s} {d['count']:>7} imgs  {d['model_id']}" for d in ds) or "no datasets")
    return 0


def cmd_manage(args) -> int:
    home = store.imgtag_home()
    if args.action == "create":
        d = store.dataset_dir(args.dataset, home)
        d.mkdir(parents=True, exist_ok=True)
        _out(args, {"dataset": args.dataset, "created": True, "path": str(d)}, f"created {d}")
    elif args.action == "rename":
        src, dst = store.dataset_dir(args.dataset, home), store.dataset_dir(args.to, home)
        if not (src / "manifest.json").is_file():
            raise store.UnknownDatasetError(args.dataset)
        if dst.exists():
            _err(f"{args.to} already exists")
            return 2
        src.rename(dst)
        man = json.loads((dst / "manifest.json").read_bytes())
        man["dataset"] = args.to
        (dst / "manifest.json").write_text(json.dumps(man, indent=1))
        _out(args, {"dataset": args.to, "renamed_from": args.dataset}, f"renamed -> {args.to}")
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
        if not d.exists():
            raise store.UnknownDatasetError(args.dataset)
        shutil.rmtree(d)  # leaves 0 orphan bytes: everything lives under this dir (B20)
        _out(args, {"dataset": args.dataset, "deleted": True}, f"deleted {args.dataset}")
    return 0


def cmd_search(args) -> int:
    """Thin f32 scan. ponytail: b-daemon owns core/search.py (calibration, tag fusion,
    hypernym expansion); this fallback keeps the CLI + e2e honest until it lands, and
    labels its probability as uncalibrated."""
    import numpy as np

    t0 = time.perf_counter()
    names = [args.dataset] if args.dataset else store.list_datasets()
    if not names:  # nothing indexed yet is an honest empty answer, not an error
        _out(args, {"query": args.query, "tookMs": round((time.perf_counter() - t0) * 1000, 2),
                    "coverage": {"indexed": 0, "total": 0}, "hits": [], "no_match": True,
                    "calibrated": False}, "no datasets indexed")
        return 0
    snaps = [store.open_snapshot(n) for n in names]
    # the manifest decides the model AND its precision — the query must live in the same
    # embedding space as the index (int8 vs fp32 vision differ at cos ~0.94)
    mid = snaps[0].manifest["model_id"]
    name, prec = (args.model, None) if args.model else mid.rsplit("-", 1)
    be = models.load_backend(name, {**load_profile(), **({"precision": prec} if prec else {})})
    for s in snaps:
        if s.manifest["model_sha"] != be.model_sha:
            raise store.ModelMismatchError(
                f"{s.dataset} was indexed with {s.manifest['model_id']} ({s.manifest['model_sha'][:12]}); "
                f"loaded {be.model_id} ({be.model_sha[:12]})"
            )
    q = be.embed_texts([args.query])[0]
    hits = []
    for s in snaps:
        if not s.count:
            continue
        scores = np.asarray(s.emb @ q, np.float32)
        k = min(args.k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        for i in top[np.argsort(-scores[top], kind="stable")]:
            r = s.ids[int(i)]
            hits.append({
                "image_id": r["image_id"], "path": r["path"], "dataset": r["dataset"],
                "score": round(float(scores[i]), 6),
                "p": round(float(1 / (1 + np.exp(-(float(scores[i]) - 0.22) * 40))), 4),
                "why": {"path": "text", "tag": None, "calibrated": False},
            })
    hits.sort(key=lambda h: (-h["score"], h["image_id"]))  # deterministic ties (B18e)
    hits = hits[: args.k]
    obj = {
        "query": args.query,
        "tookMs": round((time.perf_counter() - t0) * 1000, 2),
        "coverage": {"indexed": sum(s.count for s in snaps), "total": sum(s.count for s in snaps)},
        "hits": hits,
        "no_match": not hits or hits[0]["score"] < NO_MATCH_FLOOR,
        "calibrated": False,
    }
    _out(args, obj, "\n".join(f"{h['score']:.4f}  {h['dataset']:12s} {h['image_id']}  {h['path']}" for h in hits)
         or "no match")
    return 0


def cmd_doctor(args) -> int:
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
    i.add_argument("--job-id", dest="job_id", help=argparse.SUPPRESS)
    i.add_argument("--model")
    i.add_argument("--full-speed", action="store_true")
    i.add_argument("--no-recursive", action="store_true")
    i.add_argument("--geometry", choices=["central", "worker"], help="override the tuned pipeline geometry")
    i.add_argument("--workers", type=int)
    i.add_argument("--batch", type=int)
    i.add_argument("--intra-op", type=int, dest="intra_op")
    i.add_argument("--precision", choices=["fp32", "fp16", "int8"])
    i.set_defaults(fn=cmd_index)

    n = sub.add_parser("info", help="datasets, jobs, profile", parents=[common])
    n.add_argument("dataset", nargs="?")
    n.add_argument("--job", help="report one job's live status")
    n.set_defaults(fn=cmd_info)

    m = sub.add_parser("manage", help="create/rename/reindex/delete a dataset", parents=[common])
    m.add_argument("action", choices=["create", "rename", "reindex", "delete"])
    m.add_argument("dataset")
    m.add_argument("--to")
    m.add_argument("--path")
    m.add_argument("--model")
    m.add_argument("--yes", "-y", action="store_true", help="no-op: imgtag never prompts (B20)")
    m.set_defaults(fn=cmd_manage)

    s = sub.add_parser("search", help="semantic search", parents=[common])
    s.add_argument("query")
    s.add_argument("--dataset")
    s.add_argument("-k", type=int, default=50)
    s.add_argument("--model")
    s.set_defaults(fn=cmd_search)

    d = sub.add_parser("doctor", help="autotune this machine (~30s)", parents=[common])
    d.add_argument("--show", action="store_true", help="print the stored profile, do not re-tune")
    d.add_argument("--model")
    d.add_argument("--fetch", metavar="BACKEND", help="download a backend's artifacts")
    d.add_argument("--allow-int8", action="store_true",
                   help="let the tune select int8 vision (B24: opt-in speed lane, v1 default is fp32)")
    d.set_defaults(fn=cmd_doctor)

    j = sub.add_parser("job", help="job status", parents=[common])
    j.add_argument("job_id", nargs="?")
    j.set_defaults(fn=cmd_job)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except tuple(EXIT) as e:
        code = EXIT[type(e)]
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
