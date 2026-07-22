"""THE CANDIDATE MATRIX — phase 1 of the ADR-4 bench. Standalone; no engine required.

    uv run python -m imgtag.bench.matrix --all
    uv run python -m imgtag.bench.matrix --candidates pecore-t16-384 --skip-perf

Emits `bench/results/<date>-candidates.json` (machine header + every row) and a ranked
`research/candidates.md` with a WINNER recommendation under the B8 precedence law.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

from . import candidates as C
from . import corpus as X
from . import perf as P
from . import quality as Q
from . import quant as QT
from . import textsets as T

CACHE = T.CACHE
RESULTS = os.path.join(C.ROOT, "bench", "results")
FIDELITY_N = 200  # B24 gate set: 200 quick500 images (brief)

# B8: peak indexing tree-RSS <= 1.0GB. Phase 1 is single-process, so the projection is
# per-worker RSS x the POLITE worker count the target would run (ADR-11 clamp, 8GB/4-core
# class box -> 2 workers). Projection, clearly labelled — not a measured tree number.
B8_INDEX_MB = 1000
TARGET_WORKERS = 2
B9_TOTAL_MB = 150


def log(*a):
    print(*a, flush=True)


# ── embeddings (cached) ───────────────────────────────────────────────────────
def corpus_embeddings(cid: str, prec: str, paths: list[str], intra: int = 4,
                      batch: int = 2, tag: str = "corpusA") -> np.ndarray | None:
    c = C.CANDIDATES[cid]
    path = c.path(prec)
    if not path:
        return None
    os.makedirs(CACHE, exist_ok=True)
    dst = os.path.join(CACHE, f"emb-{tag}-{cid}-{prec}.npy")
    if os.path.exists(dst):
        e = np.load(dst)
        if len(e) == len(paths):
            return e
    t0 = time.perf_counter()
    sess = C.session(path, intra)
    e = C.embed_images(sess, paths, c, batch=batch,
                       progress=lambda i, n: log(f"      {cid}/{prec} {i}/{n} "
                                                 f"({i/max(1e-9, time.perf_counter()-t0):.1f} img/s)")
                       if tag == "corpusA" else None)
    np.save(dst, e)
    log(f"    embedded {len(e)} imgs in {time.perf_counter()-t0:.0f}s -> {dst}")
    return e


def text_embeddings(cid: str, ts: dict, with_captions: bool = True) -> dict | None:
    """Text tower embeddings for one candidate. None when no usable text tower."""
    c = C.CANDIDATES[cid]
    tpath = c.text.get("fp32") or c.text.get("int8")
    if not tpath or not os.path.exists(tpath):
        log(f"    {cid}: no text tower on disk — quality BLOCKED")
        return None
    dst = os.path.join(CACHE, f"txt-{cid}.npz")
    if os.path.exists(dst):
        z = np.load(dst)
        return {k: z[k] for k in z.files}
    try:
        groups = {"cat": ts["cat_prompts"], "sup": ts["sup_prompts"],
                  "abs": ts["abs_prompts"]}
        if with_captions:
            groups["cap"] = ts["captions"]
        toks = {k: T.tokenize(v, c.tok, c.ctx) for k, v in groups.items()}
    except RuntimeError as e:
        log(f"    {cid}: tokenizer BLOCKED — {e}")
        return None
    sess = C.session(tpath, 4)
    out = {k: C.embed_texts(sess, v, c.out_idx) for k, v in toks.items()}
    np.savez_compressed(dst, **out)
    return out


# ── per-candidate run ─────────────────────────────────────────────────────────
def run_candidate(cid: str, do_perf: bool, do_quality: bool, quality_precs: tuple,
                  repeats: int) -> dict:
    c = C.CANDIDATES[cid]
    log(f"\n=== {cid} ({c.license}) — {c.note}")
    rec: dict = {"candidate": cid, "res": c.res, "dim": c.dim, "license": c.license,
                 "note": c.note, "artifacts": {}, "perf": [], "fidelity": {},
                 "quality": {}, "blocked": []}

    # 1. artifacts — self-quantize int8 with the ADR-4 weight-only recipe
    fp32 = c.path("fp32")
    if fp32 and not c.path("int8") and c.vision.get("int8", "").endswith("-int8-wo.onnx"):
        log(f"  quantizing int8 (ADR-4 weight-only recipe) …")
        r = QT.quantize_weight_only(fp32, c.vision["int8"])
        log(f"    {r}")
        if not r.get("ok"):
            rec["blocked"].append(f"int8 quantize: {r.get('error')}")
    for p in ("fp32", "int8", "int8wo"):
        q = c.path(p)
        if q:
            rec["artifacts"][f"vision_{p}_mb"] = round(os.path.getsize(q) / 1e6, 1)
    for p, q in c.text.items():
        if os.path.exists(q):
            rec["artifacts"][f"text_{p}_mb"] = round(os.path.getsize(q) / 1e6, 1)
    avail = c.available()
    if not avail:
        rec["blocked"].append("no vision artifact on disk")
        log("  BLOCKED: no vision artifact")
        return rec

    # 2. perf matrix — precision x intra x batch, fresh process per config
    if do_perf:
        log("  perf matrix (fresh process per config):")
        rec["perf"] = P.run_matrix(cid, precisions=tuple(avail), repeats=repeats, log=log)

    # 3. B24 fidelity — int8 vs fp32 on 200 quick500 images
    quick = sorted(glob.glob(os.path.join(C.DATA, "quick500/images/*.jpg")))[:FIDELITY_N]
    if "fp32" in avail and quick:
        ref = corpus_embeddings(cid, "fp32", quick, tag="quick200")
        for prec in [p for p in ("int8", "int8wo") if c.path(p)]:
            cand = corpus_embeddings(cid, prec, quick, tag="quick200")
            if cand is not None and ref is not None:
                rec["fidelity"][prec] = QT.fidelity(ref, cand)
                f = rec["fidelity"][prec]
                log(f"  B24 {prec}: cos={f['mean_cos']:.4f} min={f['min_cos']:.4f} "
                    f"nn_agree={f['nn_agree']:.3f} -> {'PASS' if f['pass'] else 'FAIL'}")
    elif "fp32" not in avail:
        rec["blocked"].append("B24 ungateable: no fp32 reference artifact for this model")

    # 4. quality on CORPUS-A
    if do_quality:
        a = X.corpus_a()
        ts = Q.text_sets()
        txt = text_embeddings(cid, ts)
        if txt is None:
            rec["blocked"].append("quality: no text tower / tokenizer")
        else:
            for prec in [p for p in quality_precs if p in avail or c.path(p)]:
                log(f"  quality {cid}/{prec} on {a['tag']} ({a['n']} imgs) …")
                img = corpus_embeddings(cid, prec, a["paths"])
                if img is None:
                    continue
                rec["quality"][prec] = Q.score_all(img, txt, ts)
                q = rec["quality"][prec]
                log(f"    B6 p@k mean={q['b6_category_precision']['mean']:.3f} "
                    f"min={q['b6_category_precision']['min']:.3f} | "
                    f"B5 p@100={q['b5_hypernym']['mean_p_at_100']:.3f} "
                    f"minchild={q['b5_hypernym']['min_child_recall']:.3f} | "
                    f"B17 R@10={q.get('b17_retrieval', {}).get('R@10', float('nan')):.1f} | "
                    f"B7 leak={q['b7_negatives']['leakage_rate']:.3f}")
    return rec


# ── ranking + report ──────────────────────────────────────────────────────────
def best_perf(rec: dict, prec: str) -> dict | None:
    rows = [r for r in rec["perf"] if r.get("precision") == prec and "img_s" in r]
    return max(rows, key=lambda r: r["img_s"]) if rows else None


def stream_perf(rec: dict, prec: str) -> dict | None:
    """The config the engine would actually run: batch<=2 streaming (spike-pecore §4)."""
    rows = [r for r in rec["perf"]
            if r.get("precision") == prec and r.get("batch", 9) <= 2 and "img_s" in r]
    return max(rows, key=lambda r: r["img_s"]) if rows else None


def shippable_precision(rec: dict) -> str:
    """int8 only if it passes B24; otherwise fp32 (ADR-4: failing precision never ships)."""
    f = rec.get("fidelity", {}).get("int8")
    return "int8" if f and f.get("pass") else "fp32"


def summarize(rec: dict) -> dict:
    prec = shippable_precision(rec)
    sp = stream_perf(rec, prec) or stream_perf(rec, "fp32")
    bp = best_perf(rec, prec)
    art = rec["artifacts"]
    # B9 shipping config (team-lead): fp32 vision + int8 text (text int8 passes fidelity
    # on every family, 0.98–0.99) + binary tokenizer (~11MB) + tag table (T×dim×4).
    vis = art.get(f"vision_{prec}_mb") or art.get("vision_fp32_mb") or 0
    txt = art.get("text_int8_mb") or art.get("text_fp32_mb") or 0
    tags_mb = round(2177 * rec["dim"] * 4 / 1e6, 1)
    tokenizer_mb = 11.0
    q = rec["quality"].get(prec) or rec["quality"].get("fp32") or {}
    per_worker = sp["peak_rss_mb"] if sp else None
    proj = per_worker * TARGET_WORKERS if per_worker else None
    # How many POLITE workers the B8 ceiling can actually afford, and what that buys.
    affordable = int(B8_INDEX_MB // per_worker) if per_worker else 0
    workers = min(TARGET_WORKERS, affordable)
    proj_throughput = round(sp["img_s"] * workers, 1) if sp and workers else None

    # B24 semantics (team-lead ruling): the gate tests a QUANTIZED artifact vs the same
    # model's fp32 reference. A ships-fp32 row IS the reference → passes trivially.
    fint8 = rec.get("fidelity", {}).get("int8")
    int8_tier = QT.b24_tier(fint8) if fint8 else None
    if prec == "fp32":
        b24_label = "✅ (ref)" + ("" if fint8 else "")
    else:
        b24_label = {"default": "✅ default", "optin": "◐ opt-in",
                     "banned": "❌ banned"}.get(int8_tier, "—")
    # int8 quality delta vs fp32 (tier-2 decides), when both quality rows exist.
    qi, qf = rec["quality"].get("int8"), rec["quality"].get("fp32")
    int8_delta = None
    if qi and qf and qi.get("b17_retrieval") and qf.get("b17_retrieval"):
        int8_delta = {
            "d_r10": round(qi["b17_retrieval"]["R@10"] - qf["b17_retrieval"]["R@10"], 1),
            "d_p_at_k": round(qi["b6_category_precision"]["mean"]
                              - qf["b6_category_precision"]["mean"], 3),
        }
    return {
        "candidate": rec["candidate"], "license": rec["license"],
        "shippable_precision": prec,
        "b24_label": b24_label, "int8_tier": int8_tier, "int8_delta": int8_delta,
        "b24": rec.get("fidelity", {}).get("int8", {}).get("pass"),
        "img_s_stream": sp["img_s"] if sp else None,
        "ms_img_stream": sp["ms_per_img"] if sp else None,
        "stream_config": f"intra={sp['intra']},b={sp['batch']}" if sp else None,
        "img_s_best": bp["img_s"] if bp else None,
        "per_worker_rss_mb": per_worker,
        "projected_index_rss_mb": proj,
        "workers_affordable_under_b8": affordable,
        "projected_polite_img_s": proj_throughput,
        "b8_eligible": (proj is not None and proj <= B8_INDEX_MB),
        "artifact_total_mb": round(vis + txt + tags_mb + tokenizer_mb, 1),
        "artifact_breakdown": {"vision_fp32": vis, "text_int8": txt,
                               "tag_table": tags_mb, "tokenizer": tokenizer_mb},
        "b9_ok": round(vis + txt + tags_mb + tokenizer_mb, 1) <= B9_TOTAL_MB,
        "b6_mean": q.get("b6_category_precision", {}).get("mean"),
        "b6_min": q.get("b6_category_precision", {}).get("min"),
        "b5_p100": q.get("b5_hypernym", {}).get("mean_p_at_100"),
        "b5_min_child": q.get("b5_hypernym", {}).get("min_child_recall"),
        "b17_r10": q.get("b17_retrieval", {}).get("R@10"),
        "b7_leak": q.get("b7_negatives", {}).get("leakage_rate"),
        "advisory": any(r.get("status") == "ADVISORY" for r in rec["perf"]),
        "blocked": rec["blocked"],
    }


def markdown(header: dict, summaries: list[dict], control_r10: float | None) -> str:
    def f(v, spec=".3f", dash="—"):
        return dash if v is None else format(v, spec)

    L = []
    L.append("# research/candidates.md — ADR-4 candidate matrix (phase 1)\n")
    L.append(f"> Generated {header['generated']} · b-bench · git {header['git_sha']}\n")
    L.append("> **EVERY NUMBER IS A PROXY.** Bench host is Apple M3 Max (arm64/NEON); the "
             "primary target is shared Linux x86_64, 8GB, no GPU. Per ADR-10e int8 "
             "speed/accuracy does NOT transfer across ISAs. No 🐧 row may lock on these.\n")
    L.append(f"> Machine: {header['machine']['platform']} · usable_cores="
             f"{header['machine']['usable_cores']} "
             f"({header['machine']['usable_cores_source']}) · ORT "
             f"{header['machine']['ort_version']} · EP=CPUExecutionProvider · "
             f"numpy {header['machine']['numpy']} · Pillow {header['machine']['pillow']}\n")
    L.append(f"> Corpora: quality/CORPUS-A = coco5k (5,000 val2017 + exhaustive 80-class "
             f"truth) · B24 fidelity = 200 quick500 · perf = quick500 tiles. "
             f"Mode: FULL (phase 1 is a model bench, not an engine bench — POLITE/FULL "
             f"resource policy applies to `bench index`, phase 2).\n")
    L.append(f"> Protocol: median of {header['repeats']} FRESH processes per perf row; "
             "`os.getloadavg()` recorded per run; rows measured at 1-min load > "
             "usable_cores x 0.6 are marked **ADVISORY** (the swarm was live — advisory "
             "rows are honest, not quiet-machine, numbers).\n")

    L.append("\n## Ranked table\n")
    L.append("| # | candidate | ships | B24 | img/s 1-proc | ms/img | per-worker RSS | "
             "workers ≤B8 | proj. POLITE img/s | proj. index RSS | B8 | artifacts | B9 | "
             "B6 p@k | B5 p@100 | B5 min-child | B17 R@10 | B7 leak |")
    L.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|---:|---:|---:|"
             "---:|---:|")
    for i, s in enumerate(summaries, 1):
        L.append(
            f"| {i} | `{s['candidate']}` | {s['shippable_precision']} | {s['b24_label']} | "
            f"{f(s['img_s_stream'], '.1f')}{'*' if s['advisory'] else ''} | "
            f"{f(s['ms_img_stream'], '.1f')} | {f(s['per_worker_rss_mb'], '.0f')}MB | "
            f"{s['workers_affordable_under_b8']} | "
            f"{f(s['projected_polite_img_s'], '.0f')} | "
            f"{f(s['projected_index_rss_mb'], '.0f')}MB | "
            f"{'✅' if s['b8_eligible'] else '🔴 INELIGIBLE-DEFAULT'} | "
            f"{f(s['artifact_total_mb'], '.0f')}MB | {'✅' if s['b9_ok'] else '❌'} | "
            f"{f(s['b6_mean'])} | {f(s['b5_p100'])} | {f(s['b5_min_child'])} | "
            f"{f(s['b17_r10'], '.1f')} | {f(s['b7_leak'])} |")
    L.append("\n`*` = ADVISORY (machine under swarm load during the timed run).")
    L.append("`ships` = default precision. v1 = **fp32 vision** everywhere (no int8 vision "
             "artifact clears B24's DEFAULT nn@200≥0.90 bar). B24 col: `✅ (ref)` = a "
             "fp32 row IS its own reference; int8 arms classified `✅ default` / `◐ opt-in` "
             "(nn 0.60–0.90, printed deltas) / `❌ banned` (below tier-1 cos 0.95 & "
             "nn 0.60). int8 opt-in deltas vs fp32:")
    for s in summaries:
        if s.get("int8_tier"):
            d = s.get("int8_delta")
            dd = (f"ΔR@10 {d['d_r10']:+.1f}, Δp@k {d['d_p_at_k']:+.3f}"
                  if d else "quality delta not measured")
            L.append(f"  - `{s['candidate']}` int8 = **{s['int8_tier']}** ({dd})")
    L.append("`artifacts` = B9 shipping sum: fp32 vision + int8 text + tag table (T×dim×4) "
             "+ ~11MB binary tokenizer.")
    L.append("`workers ≤B8` = how many streaming workers of this size fit under B8's 1.0GB "
             "indexing ceiling; `proj. POLITE img/s` = single-process img/s x "
             f"min({TARGET_WORKERS}, that) — {TARGET_WORKERS} being ADR-11's POLITE clamp "
             "`clamp(ncpu−2,2,8)` on a 4-core target. Both are **projections** from a "
             "single-process bench, not a measured process tree (that is B8/B1's own "
             "bench, phase 2), and they assume the near-linear process scaling the "
             "runtime lane measured — which is itself an ARM proxy result.")
    if control_r10 is not None:
        L.append(f"\n**B17 control** (`openclip-vitb32`, openai weights, same corpus/run): "
                 f"R@10 = {control_r10:.1f}. Gate: default must reach "
                 f"**{control_r10 + 12:.1f}**.")
    return "\n".join(L) + "\n"


def detail_md(recs: list[dict]) -> str:
    L = ["\n## Per-candidate detail\n"]
    for r in recs:
        L.append(f"### `{r['candidate']}` — {r['license']} · res {r['res']} · dim {r['dim']}")
        L.append(f"*{r['note']}*\n")
        if r["blocked"]:
            L.append("**BLOCKED:** " + "; ".join(r["blocked"]) + "\n")
        if r["artifacts"]:
            L.append("Artifacts: " + " · ".join(f"`{k}` {v}MB"
                                                for k, v in r["artifacts"].items()) + "\n")
        if r["fidelity"]:
            L.append("**B24 fidelity vs own fp32** (n=200 quick500; gate mean cos ≥0.995, "
                     "min ≥0.97, top-1 NN agreement ≥0.90):\n")
            L.append("| precision | mean cos | min cos | p1 cos | NN agree | top-3 overlap | verdict |")
            L.append("|---|---:|---:|---:|---:|---:|---|")
            for p, f in r["fidelity"].items():
                L.append(f"| {p} | {f['mean_cos']:.4f} | {f['min_cos']:.4f} | "
                         f"{f['p1_cos']:.4f} | {f['nn_agree']:.3f} | {f['top3_overlap']:.3f} | "
                         f"{'✅ PASS' if f['pass'] else '❌ FAIL'} |")
            L.append("")
        if r["perf"]:
            L.append("**Perf matrix** (median of fresh processes; ms/img, img/s, peak RSS):\n")
            L.append("| precision | intra | batch | img/s | ms/img | peak RSS | spread | load | status |")
            L.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
            for p in r["perf"]:
                if "img_s" not in p:
                    L.append(f"| {p['precision']} | {p['intra']} | {p['batch']} | — | — | — | — | — | "
                             f"BLOCKED: {p.get('error', '')[:60]} |")
                    continue
                L.append(f"| {p['precision']} | {p['intra']} | {p['batch']} | {p['img_s']} | "
                         f"{p['ms_per_img']} | {p['peak_rss_mb']:.0f}MB | {p['spread_pct']}% | "
                         f"{p['loadavg_max']} | {p['status']} |")
            L.append("")
        for prec, q in r["quality"].items():
            b6, b5 = q["b6_category_precision"], q["b5_hypernym"]
            L.append(f"**Quality / {prec}** on CORPUS-A/coco5k:\n")
            L.append(f"- **B6** precision@min(10,N_pos): mean **{b6['mean']:.3f}** · "
                     f"min **{b6['min']:.3f}** · zeros: "
                     f"{', '.join(b6['zeros']) or 'none'} → "
                     f"{'✅' if b6['pass'] else '❌'} (gate mean ≥0.90, min ≥0.70, no zeros)")
            L.append("  - worst 8: " + ", ".join(f"`{x['category']}` {x['p_at_k']:.2f}"
                                                 f"(k={x['k']})" for x in b6["rows"][:8]))
            L.append(f"- **B5** hypernym: p@100 mean **{b5['mean_p_at_100']:.3f}** · "
                     f"child recall@R mean **{b5['mean_child_recall']:.3f}** min "
                     f"**{b5['min_child_recall']:.3f}** · children absent from top-100: "
                     f"{', '.join(b5['children_absent_from_top100']) or 'none'} → "
                     f"{'✅' if b5['pass'] else '❌'}")
            for s in b5["rows"]:
                worst = sorted(s["children"], key=lambda c: c["recall_at_R"])[:3]
                L.append(f"  - `{s['supercat']}` R={s['R']} p@100={s['precision_at_100']:.2f}"
                         f" · weakest children: " +
                         ", ".join(f"{c['child']} {c['recall_at_R']:.2f}" for c in worst))
            if "b17_retrieval" in q:
                b17 = q["b17_retrieval"]
                L.append(f"- **B17** text→image on {b17['corpus']}, {b17['n_queries']} "
                         f"caption queries: R@1 **{b17['R@1']:.1f}** · R@5 "
                         f"**{b17['R@5']:.1f}** · R@10 **{b17['R@10']:.1f}** · median rank "
                         f"{b17['median_rank']:.0f}")
            b7 = q["b7_negatives"]
            if b7.get("unfittable"):
                L.append(f"- **B7** negatives: UNFITTABLE — recall@10 ceiling "
                         f"{b7['recall_at_10_ceiling']:.3f} < 0.70 at any τ. "
                         f"margin(present−absent) = {b7['margin_present_minus_absent']:.4f}")
            else:
                L.append(f"- **B7** negatives (τ fitted in-run to hold recall@10 ≥0.70): "
                         f"τ={b7['tau']:.4f} · recall@10 {b7['recall_at_10_at_tau']:.3f} · "
                         f"leakage **{b7['leakage_rate']:.3f}** over "
                         f"{b7['n_absent_queries']} auto-derived absent queries · "
                         f"margin(present−absent) {b7['margin_present_minus_absent']:.4f} → "
                         f"{'✅' if b7['pass'] else '❌'} (gate ≤0.02)")
            L.append("")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--skip-perf", action="store_true")
    ap.add_argument("--skip-quality", action="store_true")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--quality-precisions", default="fp32,int8")
    ap.add_argument("--wait-quiet", type=int, default=0,
                    help="poll-wait up to N seconds for load <= cores*0.6 before perf")
    a = ap.parse_args(argv)

    ids = ([k for k in C.CANDIDATES if C.CANDIDATES[k].available()] if a.all
           else [s for s in a.candidates.split(",") if s])
    if not ids:
        log("no candidates with artifacts on disk")
        return 1

    if a.wait_quiet and not a.skip_perf:
        ok, l1 = C.wait_for_quiet(a.wait_quiet)
        log(f"load gate: {'quiet' if ok else 'STILL LOADED -> rows ADVISORY'} (1-min {l1})")

    header = {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "machine": C.machine_header(),
        "loadavg_at_start": os.getloadavg(),
        "repeats": a.repeats,
        "git_sha": subprocess.run(["git", "-C", C.ROOT, "rev-parse", "--short", "HEAD"],
                                  capture_output=True, text=True).stdout.strip() or "?",
        "corpora": {"quality": "CORPUS-A/coco5k", "fidelity": "quick500[:200]"},
    }
    recs = [run_candidate(cid, not a.skip_perf, not a.skip_quality,
                          tuple(a.quality_precisions.split(",")), a.repeats)
            for cid in ids]

    summaries = [summarize(r) for r in recs]
    control = next((s["b17_r10"] for s in summaries
                    if s["candidate"] == "openclip-vitb32"), None)

    def key(s):  # B8 precedence: ineligible candidates rank below every eligible one
        return (0 if s["b8_eligible"] else 1, -(s["b17_r10"] or 0), -(s["b6_mean"] or 0))

    summaries.sort(key=key)

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, f"{dt.date.today()}-candidates.json")
    with open(out, "w") as fh:
        json.dump({"header": header, "summaries": summaries, "records": recs}, fh, indent=1)
    log(f"\nwrote {out}")

    md = os.path.join(C.ROOT, "research", "candidates.md")
    prev = ""
    if os.path.exists(md):  # keep a hand-written WINNER/FINDINGS block across re-runs
        old = open(md).read()
        if "<!-- HANDWRITTEN -->" in old:
            prev = old.split("<!-- HANDWRITTEN -->", 1)[1]
    with open(md, "w") as fh:
        fh.write(markdown(header, summaries, control) + detail_md(recs) +
                 "\n<!-- HANDWRITTEN -->" + prev)
    log(f"wrote {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
