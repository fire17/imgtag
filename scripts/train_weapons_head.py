#!/usr/bin/env python3
"""Train + evaluate the weapons moderation head on the Open Images slice.

TRAIN = OI `test` split · HELD-OUT EVAL = OI `validation` split (disjoint by construction).
Compares every approach on the SAME held-out split and writes the winner into
src/imgtag/data/moderation/ so the head ships with the package (a few KB of floats).

Usage:
  .venv/bin/python scripts/train_weapons_head.py --backend pecore-s16-384 [--save]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from imgtag.core.models import load_backend                       # noqa: E402
from imgtag.core.tags import fit_platt, platt_apply               # noqa: E402
from imgtag.moderation import weapons as W                        # noqa: E402

SLICE = ROOT / "data" / "oi-weapons"
CACHE = ROOT / ".scratch" / "weapons-emb"


def probe_records() -> list[dict]:
    f = SLICE / "probe.jsonl"
    if not f.is_file():
        return []
    recs = [json.loads(ln) for ln in f.read_text().splitlines()]
    return [r for r in recs if (SLICE / "validation" / f"{r['id']}.jpg").is_file()]


def records(split: str) -> list[dict]:
    recs = [json.loads(ln) for ln in (SLICE / f"{split}.jsonl").read_text().splitlines()]
    return [r for r in recs if (SLICE / split / f"{r['id']}.jpg").is_file()]


def embed_split(backend, split: str, recs: list[dict], batch: int = 8) -> np.ndarray:
    """Embed a split, cached by (model_sha, split, n). Decode threads + batched infer."""
    CACHE.mkdir(parents=True, exist_ok=True)
    # Keyed by the CONTENT of the id list, not its length: a single re-fetched image used
    # to silently invalidate a 4,600-image cache and cost ~50 CPU-minutes for no new data.
    key = hashlib.sha256("\x00".join(r["id"] for r in recs).encode()).hexdigest()[:12]
    cp = CACHE / f"{backend.model_sha[:12]}-{split}-{len(recs)}-{key}.npy"
    if cp.is_file():
        return np.load(cp)
    sub = "validation" if split == "probe" else split
    paths = [str(SLICE / sub / f"{r['id']}.jpg") for r in recs]

    def prep(p):
        with Image.open(p) as im:
            return backend.preprocess(im.convert("RGB"))

    out, t0 = [], time.time()
    with ThreadPoolExecutor(4) as ex:
        buf = []
        for i, a in enumerate(ex.map(prep, paths), 1):
            buf.append(a)
            if len(buf) == batch:
                out.append(backend.embed_images(np.stack(buf)))
                buf = []
            if i % 500 == 0:
                print(f"  {split} {i}/{len(paths)} ({i/(time.time()-t0):.1f} img/s)", flush=True)
        if buf:
            out.append(backend.embed_images(np.stack(buf)))
    e = np.concatenate(out).astype(np.float32)
    np.save(cp, e)
    return e


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval. n is small here (392 positives, 177 random negatives) and a
    bare point estimate on that would overclaim — 0/177 is not 'zero', it is '<1.7%'."""
    if n == 0:
        return (0.0, 1.0)
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * ((ph * (1 - ph) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def cross_corpus(head, backend_name: str) -> None:
    """FP rate on COCO val2017 — a DIFFERENT corpus, 5,000 images, zero OI overlap.

    This is the honest per-corpus false-alarm number: OI-derived negatives were chosen
    adversarially, COCO was not. Reuses b-bench's embedding cache, so it costs nothing.
    """
    cache = ROOT / "bench" / "cache" / f"emb-corpusA-{backend_name}-fp32.npy"
    if not cache.is_file():
        print(f"\ncross-corpus check SKIPPED (no {cache.name}; run `imgtag bench` first)")
        return
    emb = np.load(cache)
    if emb.shape[1] != head.dim:
        print("\ncross-corpus check SKIPPED (dim mismatch)")
        return
    p = head.probs(emb)
    n = len(p)
    print(f"\nCROSS-CORPUS FP rate — COCO val2017, n={n}, non-OI imagery:")
    print("| tier | τ | flagged | FP rate [95% CI] | P̂@π=10% | P̂@π=1% | P̂@π=0.1% |")
    print("|---|---|---|---|---|---|---|")
    for tname, tt in (("violation", head.tau_violation), ("review", head.tau_review)):
        k = int((p >= tt).sum())
        lo, hi = wilson(k, n)
        rec = head.metrics[tname]["recall"]
        cells = [pi * rec / max(pi * rec + (1 - pi) * hi, 1e-12) for pi in (0.10, 0.01, 0.001)]
        print(f"| {tname} | {tt:.4f} | {k}/{n} | {k/n:.4f} [{lo:.4f}–{hi:.4f}] | "
              + " | ".join(f"{c:.3f}" for c in cells) + " |")

    try:                                    # WHICH COCO classes trip it — the FP structure
        from imgtag.bench.corpus import corpus_a
        pos = corpus_a()["pos"]
    except Exception as e:
        print(f"  (class breakdown unavailable: {type(e).__name__})")
        return
    flag = p >= head.tau_review
    base = float(flag.mean())
    rows = [(c, len(s), float(flag[np.array(sorted(s))].mean()))
            for c, s in pos.items() if len(s) >= 25]
    rows.sort(key=lambda x: -x[2])
    print(f"\n  COCO classes most flagged at review-τ (baseline {base:.3f}):")
    for c, nn, r in rows[:8]:
        print(f"    {c:16s} n={nn:4d}  {r:.3f}  ({r/max(base,1e-9):.1f}x)")
    if "knife" in pos:
        s = np.array(sorted(pos["knife"]))
        r = float(flag[s].mean())
        print(f"    {'knife (COCO)':16s} n={len(s):4d}  {r:.3f}  ({r/max(base,1e-9):.1f}x) "
              "← kitchen-knife boundary check")


def per_class(recs, p, y, tau) -> list[tuple]:
    by = defaultdict(lambda: [0, 0])
    for r, pi in zip(recs, p):
        if r["y"] != 1:
            continue
        for c in r["classes"]:
            by[c][0] += 1
            by[c][1] += pi >= tau
    return sorted(((c, n, int(k), k / n) for c, (n, k) in by.items()), key=lambda x: x[3])


def neg_by_class(recs, p, tau) -> list[tuple]:
    """FP rate per hard-negative object class — the honest 'what can't it tell apart' table."""
    by = defaultdict(lambda: [0, 0])
    for r, pi in zip(recs, p):
        if r["y"] == 1:
            continue
        for c in r["classes"] or ["(unlabeled)"]:
            by[c][0] += 1
            by[c][1] += pi >= tau
    return sorted(((c, n, int(k), k / n) for c, (n, k) in by.items() if n >= 5),
                  key=lambda x: -x[3])


def neg_by_tier(recs, p, tau) -> list[tuple]:
    by = defaultdict(lambda: [0, 0])
    for r, pi in zip(recs, p):
        if r["y"] == 1:
            continue
        t = r.get("tier", "?")
        by[t][0] += 1
        by[t][1] += pi >= tau
    return sorted((t, n, int(k), k / n) for t, (n, k) in by.items())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="pecore-s16-384")
    ap.add_argument("--target-recall", type=float, default=0.95)
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()

    b = load_backend(a.backend)
    print(f"backend={a.backend} dim={b.dim} sha={b.model_sha[:12]}")

    tr, ev = records("test"), records("validation")
    print(f"train n={len(tr)} pos={sum(r['y'] for r in tr)} | "
          f"eval n={len(ev)} pos={sum(r['y'] for r in ev)}")
    Xtr, Xev = embed_split(b, "test", tr), embed_split(b, "validation", ev)
    pr = probe_records()
    probe = embed_split(b, "probe", pr) if pr else None
    if probe is not None:
        print(f"clean-corpus probe: {len(pr)} images")
    ytr = np.array([r["y"] for r in tr])
    yev = np.array([r["y"] for r in ev])

    pos, neg = W.prompt_matrices(b)
    b.release_text()
    rows, scored = [], {}

    # A — zero-shot, max positive-prompt cosine only (no background contrast)
    for name, s_tr, s_ev in (
        ("A zero-shot cos (pos-max)", (Xtr @ pos.T).max(1), (Xev @ pos.T).max(1)),
        ("B zero-shot margin (pos-max − bg-max)",
         W.zero_shot_margin(Xtr, pos, neg), W.zero_shot_margin(Xev, pos, neg)),
    ):
        ab = fit_platt(s_tr, ytr)                  # calibrate on TRAIN, threshold on EVAL
        p = platt_apply(s_ev, ab)
        tau = W.tau_for_recall(p, yev, a.target_recall)
        m = W.prf(p, yev, tau)
        m["ap"] = W.average_precision(p, yev)
        rows.append((name, m))
        scored[name] = (p, tau)

    # C — trained logistic head on the embedding
    head = W.train(Xtr, ytr, b, val=(Xev, yev), review_recall=a.target_recall)
    p = head.probs(Xev)
    rows.append(("C trained head (logistic)", head.metrics))
    scored["C trained head (logistic)"] = (p, head.tau_review)

    # D — trained head on [embedding | zero-shot margin] (does the prompt view add?)
    mtr = W.zero_shot_margin(Xtr, pos, neg)[:, None]
    mev = W.zero_shot_margin(Xev, pos, neg)[:, None]
    w, bb = W.fit_logistic(np.hstack([Xtr, mtr]), ytr)
    str_ = np.hstack([Xtr, mtr]) @ w + bb
    ab = fit_platt(str_, ytr)
    p = platt_apply(np.hstack([Xev, mev]) @ w + bb, ab)
    tau = W.tau_for_recall(p, yev, a.target_recall)
    m = W.prf(p, yev, tau)
    m["ap"] = W.average_precision(p, yev)
    rows.append(("D trained head + margin feature", m))
    scored["D trained head + margin feature"] = (p, tau)

    print(f"\n| approach | AP | τ@R{a.target_recall:.2f} | precision | recall | flag-rate | FP |")
    print("|---|---|---|---|---|---|---|")
    for name, m in rows:
        print(f"| {name} | {m['ap']:.3f} | {m['tau']:.4f} | {m['precision']:.3f} | "
              f"{m['recall']:.3f} | {m['flag_rate']:.3f} | {m['fp']} |")

    best = max(rows, key=lambda r: r[1]["ap"])[0]
    print(f"\nbest by AP: {best}")
    p, tau = scored["C trained head (logistic)"]
    print("\nper-weapon-class recall @ head τ (held-out):")
    for c, n, k, r in per_class(ev, p, yev, tau):
        print(f"  {c:15s} {k:3d}/{n:3d}  {r:.2f}")
    print("\nfalse-positive rate by negative tier @ head review-τ:")
    for t, n, k, r in neg_by_tier(ev, p, tau):
        print(f"  {t:9s} {k:3d}/{n:4d}  {r:.3f}")
    print("\nfalse-positive rate by hard-negative CLASS @ head review-τ (n>=5):")
    for c, n, k, r in neg_by_class(ev, p, tau):
        print(f"  {c:16s} {k:3d}/{n:4d}  {r:.3f}")
    # The operator table. Precision on THIS split is meaningless to a site owner (33%
    # prevalence, adversarially-chosen negatives); recall + FPR project to any prevalence.
    print("\nOPERATOR TABLE — recall, FPR, and projected precision by real prevalence:")
    print("| tier | τ | recall [95% CI] | FPR-random [95% CI] | P@π=10% | P@π=1% | P@π=0.1% | flags/10k @π=0.1% |")
    print("|---|---|---|---|---|---|---|---|")
    rnd = np.array([r.get("tier") == "random" for r in ev]) & (yev == 0)
    p_rnd = p[rnd]
    if probe is not None:                      # clean-corpus probe widens the FPR sample
        p_rnd = np.concatenate([p_rnd, head.probs(probe)])
        print(f"  (FPR sample = {int(rnd.sum())} eval-random + {len(probe)} clean-corpus probe)")
    for tname, tt in (("violation", head.tau_violation), ("review", head.tau_review)):
        m = W.prf(p, yev, tt)
        nr = len(p_rnd)
        kr = int((p_rnd >= tt).sum())
        lo_r, hi_r = wilson(kr, nr)
        rlo, rhi = wilson(m["tp"], m["tp"] + m["fn"])
        cells = []
        for pi in (0.10, 0.01, 0.001):     # worst case of the FPR CI — never the point est.
            cells.append(pi * m["recall"] / max(pi * m["recall"] + (1 - pi) * hi_r, 1e-12))
        print(f"| {tname} | {tt:.4f} | {m['recall']:.3f} [{rlo:.3f}–{rhi:.3f}] | "
              f"{kr}/{nr}={kr/max(nr,1):.3f} [{lo_r:.3f}–{hi_r:.3f}] | "
              + " | ".join(f"{c:.3f}" for c in cells)
              + f" | {10000 * (0.001 * m['recall'] + 0.999 * hi_r):.0f} |")
    print("  (Projected precision uses the UPPER bound of the FPR interval — the honest")
    print("   worst case. 0/177 flagged is not 'zero FPs', it is 'below ~2%'.)")
    print("  (FPR-random is the honest per-corpus false-alarm rate; the `hard`/`verified`")
    print("   tiers are adversarially selected and over-represent confusions ~4-8x.)")

    cross_corpus(head, a.backend)

    print("\nADR-14 tier split (held-out):")
    for name, tt in (("violation", head.tau_violation), ("review", head.tau_review)):
        m = W.prf(p, yev, tt)
        print(f"  {name:9s} tau={tt:.4f} precision={m['precision']:.3f} "
              f"recall={m['recall']:.3f} flag_rate={m['flag_rate']:.3f} fp={m['fp']}")

    # operating-point sweep — the operator's real knob
    print("\nrecall-target sweep (trained head, held-out):")
    print("| target recall | τ | precision | recall | flag-rate |")
    print("|---|---|---|---|---|")
    for t in (0.80, 0.90, 0.95, 0.99):
        tt = W.tau_for_recall(p, yev, t)
        m = W.prf(p, yev, tt)
        print(f"| {t:.2f} | {tt:.4f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['flag_rate']:.3f} |")

    if a.save:
        head.metrics["per_class_recall"] = {c: r for c, _, _, r in per_class(ev, p, yev, head.tau_review)}
        head.metrics["fp_by_tier"] = {t: r for t, _, _, r in neg_by_tier(ev, p, head.tau_review)}
        head.metrics["eval_split"] = "openimages-validation"
        head.metrics["train_split"] = "openimages-test"
        head.model_id = a.backend
        print("saved", W.save(head, ROOT / "src" / "imgtag" / "data" / "moderation"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
