"""`imgtag bench *` — the BUDGETS test-command surface (phase 2), wired to the engine.

OWNER: b-bench. The engine's `cli.py` (b-engine) dispatches its `bench` verb here via
`bench.cli.main(argv)`; everything below reuses the PUBLIC engine API (`store.open_snapshot`,
`models.ModelBackend`) so throughput/latency/quality are measured through the real pipeline,
not a bench-only mock.

Every subcommand honors B20's --json law: valid JSON to stdout, human text to stderr, and
the BUDGETS bench-protocol header (machine, loadavg, corpus tag, mode) on every result.
Rows measured at 1-min load > usable_cores×0.6 carry `"status":"ADVISORY"` — never a
fabricated quiet-machine number.

    imgtag bench search   --dataset cocoval2017 --queries 200   (B3)
    imgtag bench resources --dataset cocoval2017                (B8, search-path RSS)
    imgtag bench candidates [--all]                             (phase-1 matrix)
    imgtag bench quality  --dataset cocoval2017 [--negatives|--hypernym|--retrieval]
    imgtag bench headtohead --dataset cocoval2017               (rclip search parity)
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

import numpy as np

from . import candidates as C


def _stderr(*a):
    print(*a, file=sys.stderr, flush=True)


def _header(dataset: str | None = None, mode: str = "POLITE") -> dict:
    h = C.machine_header()
    ok, load = C.load_ok()
    h.update({"loadavg": __import__("os").getloadavg(), "mode": mode,
              "load_ok": ok, "status": "OK" if ok else "ADVISORY"})
    if dataset:
        h["dataset"] = dataset
    return h


# ── B3: warm search latency, e2e through the engine ──────────────────────────
def _query_pool(n: int) -> list[str]:
    """DISTINCT never-pre-warmed queries (B3): COCO caption fragments + tag names."""
    from . import corpus as X
    from . import textsets as T

    a = X.corpus_a()
    caps = [c for _, c in a["captions"]]
    tags = [T.prompt(k) for k in a["pos"]]
    pool = tags + caps
    # deterministic spread, de-duplicated, capped at n
    seen, out = set(), []
    step = max(1, len(pool) // n)
    for s in pool[::step]:
        if s not in seen:
            seen.add(s)
            out.append(s)
        if len(out) >= n:
            break
    return out


def cmd_search(args) -> dict:
    from ..core import doctor, models, store

    snap = store.open_snapshot(args.dataset)
    emb = np.ascontiguousarray(snap.emb)  # B3: scan array MUST be f32
    assert emb.dtype == np.float32, f"scan array is {emb.dtype}, not float32 (ADR-2)"
    model_id = snap.manifest.get("model_id", "").replace("-fp32", "").replace("-int8", "")
    spec = models.registry().get(model_id)
    if not spec:
        return {"error": f"backend '{model_id}' not in registry", "exit": 5}
    be = models.ModelBackend(model_id, spec, doctor.load_profile(), vision=False)

    queries = _query_pool(args.queries)
    _stderr(f"[bench search] {args.dataset} N={emb.shape[0]} queries={len(queries)} "
            f"dim={emb.shape[1]}")
    # warm the tower/arena (NOT the query cache — the test set is never pre-warmed)
    be.embed_texts(queries[:2])

    lat = []
    for q in queries:
        t = time.perf_counter()
        qe = be.embed_texts([q])[0]
        scores = emb @ qe                       # exact brute-force scan (ADR-2)
        np.argpartition(scores, -min(50, len(scores)))[-50:]
        lat.append((time.perf_counter() - t) * 1000)
    lat.sort()

    def pct(p):
        return round(lat[min(len(lat) - 1, int(len(lat) * p / 100))], 2)

    h = _header(args.dataset)
    return {**h, "budget": "B3", "n_queries": len(queries), "scan_n": int(emb.shape[0]),
            "scan_dtype": str(emb.dtype),
            "embed_plus_scan_ms": {"p50": pct(50), "p95": pct(95), "p99": pct(99),
                                   "min": round(lat[0], 2), "max": round(lat[-1], 2)},
            "gate": {"p50<=50": pct(50) <= 50, "p95<=120": pct(95) <= 120},
            "note": "embed_text + full scan per query, cold cache; daemon path is separate"}


# ── B8 search-path: resident RSS of the search process ───────────────────────
def cmd_resources(args) -> dict:
    from ..core import doctor, models, store

    base = C.peak_rss_mb()
    snap = store.open_snapshot(args.dataset)
    emb = np.ascontiguousarray(snap.emb)
    after_snap = C.peak_rss_mb()
    model_id = snap.manifest.get("model_id", "").replace("-fp32", "").replace("-int8", "")
    be = models.ModelBackend(model_id, models.registry()[model_id],
                             doctor.load_profile(), vision=False)
    be.embed_texts(["warm the text tower"])
    after_text = C.peak_rss_mb()
    _ = emb @ be.embed_texts(["a photo of a car"])[0]
    peak = C.peak_rss_mb()
    return {**_header(args.dataset), "budget": "B8-search",
            "scan_array_mb": round(emb.nbytes / 1e6, 1),
            "rss_base_mb": round(base, 1), "rss_after_snapshot_mb": round(after_snap, 1),
            "rss_after_text_tower_mb": round(after_text, 1),
            "peak_rss_mb": round(peak, 1),
            "text_tower_resident_mb": round(after_text - after_snap, 1),
            "gate": {"search_resident<=350": peak <= 350},
            "note": "search-path only (idle daemon proxy); indexing peak is `bench index`"}


# ── B1/B2: e2e index throughput (QUIET-WINDOW bench — it is a timed measurement) ─
CORPUS_PATHS = {
    "A": "data/coco/val2017",       # CORPUS-A coco5k
    "B": "data/unsplash-b",         # CORPUS-B photo10k (partial on disk)
    "quick": "data/quick500/images",
}


def cmd_index(args) -> dict:
    import os
    import shutil

    from ..core import doctor, indexer, store

    ok, load = C.load_ok()
    corpora = [c.strip() for c in args.corpus.split(",") if c.strip()]
    rows = []
    for tag in corpora:
        path = CORPUS_PATHS.get(tag)
        if not path or not os.path.isdir(os.path.join(C.ROOT, path)):
            rows.append({"corpus": tag, "error": f"path missing: {path}"})
            continue
        abspath = os.path.join(C.ROOT, path)
        ds = f"benchtmp-{tag.lower()}-{os.getpid()}"
        prof = doctor.load_profile()
        if args.full_speed:
            prof["full_speed"] = True
        _stderr(f"[bench index] {tag} → {abspath} (dataset {ds}, "
                f"{'FULL' if args.full_speed else 'POLITE'})")
        t = time.perf_counter()
        try:
            s = indexer.index(abspath, ds, profile=prof, full_speed=args.full_speed,
                              recursive=True, log=lambda m: None)
        except Exception as e:  # noqa: BLE001 — a failed corpus is a row, not a crash
            rows.append({"corpus": tag, "error": f"{type(e).__name__}: {e}"[:200]})
            continue
        wall = time.perf_counter() - t
        row = {"corpus": tag, "path": path, "indexed": s.get("indexed"),
               "skipped": s.get("skipped"), "failed": s.get("failed"),
               "seconds": s.get("seconds"), "img_s": s.get("img_s"),
               "wall_s": round(wall, 1), "model_id": s.get("model_id"),
               "workers": s.get("workers"), "batch": s.get("batch"),
               "stages_ms_per_img": s.get("stages_ms_per_img"),
               "mode": "FULL" if args.full_speed else "POLITE"}
        if args.headtohead:
            row["rclip"] = _rclip_index(abspath)
        rows.append(row)
        # throwaway dataset — never leave bench state in the user's registry
        try:
            shutil.rmtree(store.dataset_dir(ds), ignore_errors=True)
        except Exception:  # noqa: BLE001
            pass
    return {**_header(mode="FULL" if args.full_speed else "POLITE"),
            "budget": "B1/B2", "loadavg_1min": load,
            "quiet_gate_ok": ok, "rows": rows,
            "note": "e2e index throughput; POLITE headline per B1. ADVISORY unless run in "
                    "a quiet window (load <= cores*0.6). Throwaway datasets deleted after."}


def _rclip_index(dirpath: str) -> dict:
    import shutil
    import subprocess

    if not shutil.which("rclip"):
        return {"error": "rclip not installed"}
    t = time.perf_counter()
    # rclip builds its index on first run over the dir; time that as its index throughput.
    r = subprocess.run(["rclip", "-n", "xyzzy-warm-index"], cwd=dirpath,
                       capture_output=True, text=True, timeout=1800)
    return {"index_s": round(time.perf_counter() - t, 1), "rc": r.returncode}


# ── rclip head-to-head (B1/B17 baseline) ─────────────────────────────────────
def cmd_headtohead(args) -> dict:
    import shutil
    import subprocess

    if not shutil.which("rclip"):
        return {"error": "rclip not installed (l-logistics: uv tool install rclip)", "exit": 7}
    ver = subprocess.run(["rclip", "--version"], capture_output=True, text=True)
    # rclip indexes on first search; we time a single query as the search-latency baseline.
    from ..core import store
    snap = store.open_snapshot(args.dataset)
    dirpath = snap.ids[0]["path"].rsplit("/", 1)[0] if snap.ids else "."
    t = time.perf_counter()
    r = subprocess.run(["rclip", "-n", "-f", "a photo of a car"], cwd=dirpath,
                       capture_output=True, text=True, timeout=600)
    dt = time.perf_counter() - t
    return {**_header(args.dataset), "budget": "B1-headtohead",
            "rclip_version": (ver.stdout or ver.stderr).strip(),
            "rclip_first_query_s": round(dt, 2),
            "rclip_rc": r.returncode,
            "note": "rclip indexes-on-first-query (CoreML); our CPU e2e is `bench index`. "
                    "This is the search-side baseline; the index-throughput head-to-head "
                    "needs the engine indexer (phase-2 --index path, conductor-coordinated)"}


# ── phase-1 passthroughs ─────────────────────────────────────────────────────
def cmd_candidates(args) -> int:
    from . import matrix

    argv = []
    if args.all:
        argv.append("--all")
    if args.skip_perf:
        argv.append("--skip-perf")
    if args.candidates:
        argv += ["--candidates", args.candidates]
    return matrix.main(argv)


def cmd_quality(args) -> dict:
    """CORPUS-A quality of the dataset's OWN indexed embeddings (no re-embed)."""
    from ..core import doctor, models, store
    from . import quality as Q

    from . import corpus as X

    snap = store.open_snapshot(args.dataset)
    img = np.ascontiguousarray(snap.emb)
    model_id = snap.manifest.get("model_id", "").replace("-fp32", "").replace("-int8", "")
    be = models.ModelBackend(model_id, models.registry()[model_id],
                             doctor.load_profile(), vision=False)
    # ground truth aligned to THIS snapshot's row order (not corpus_a's) — mandatory for a
    # pre-indexed dataset, else pos indices point at the wrong rows.
    gt = X.align_to_ids([dict(r) for r in snap.ids])
    ts = Q.text_sets()
    ts = {**ts, "captions": [c for _, c in gt["captions"]],
          "caption_img_idx": [i for i, _ in gt["captions"]]}
    groups = {"cat": ts["cat_prompts"], "sup": ts["sup_prompts"], "abs": ts["abs_prompts"]}
    if not args.no_retrieval:
        groups["cap"] = ts["captions"]
    txt = {k: be.embed_texts(v) for k, v in groups.items()}
    _stderr(f"[bench quality] {args.dataset} N={img.shape[0]} "
            f"coco-coverage={gt['coverage']}")
    res = Q.score_all(img, txt, ts, gt=gt)
    return {**_header(args.dataset), "budget": "B5/B6/B7/B17",
            "coco_coverage": gt["coverage"], "results": res}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="imgtag bench")
    sub = ap.add_subparsers(dest="bench_cmd", required=True)

    s = sub.add_parser("search", help="B3 warm search latency (e2e)")
    s.add_argument("--dataset", default="cocoval2017")
    s.add_argument("--queries", type=int, default=200)
    s.add_argument("--json", action="store_true")

    r = sub.add_parser("resources", help="B8 search-path resident RSS")
    r.add_argument("--dataset", default="cocoval2017")
    r.add_argument("--json", action="store_true")

    q = sub.add_parser("quality", help="B5/B6/B7/B17 on an indexed dataset")
    q.add_argument("--dataset", default="cocoval2017")
    q.add_argument("--no-retrieval", action="store_true")
    q.add_argument("--json", action="store_true")

    h = sub.add_parser("headtohead", help="rclip search baseline (B1)")
    h.add_argument("--dataset", default="cocoval2017")
    h.add_argument("--json", action="store_true")

    ix = sub.add_parser("index", help="B1/B2 e2e index throughput (quiet-window bench)")
    ix.add_argument("--corpus", default="A", help="comma list of A,B,quick")
    ix.add_argument("--headtohead", action="store_true", help="also time rclip on the dir")
    ix.add_argument("--full-speed", action="store_true")
    ix.add_argument("--json", action="store_true")

    c = sub.add_parser("candidates", help="phase-1 candidate matrix")
    c.add_argument("--all", action="store_true")
    c.add_argument("--candidates", default="")
    c.add_argument("--skip-perf", action="store_true")

    a = ap.parse_args(argv)
    if a.bench_cmd == "candidates":
        return cmd_candidates(a)

    fn = {"search": cmd_search, "resources": cmd_resources, "quality": cmd_quality,
          "headtohead": cmd_headtohead, "index": cmd_index}[a.bench_cmd]
    out = fn(a)
    print(json.dumps(out, indent=None if getattr(a, "json", False) else 1))
    return out.get("exit", 0) if isinstance(out, dict) else 0


if __name__ == "__main__":
    sys.exit(main())
