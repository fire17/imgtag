#!/usr/bin/env python
"""Measure AND refit the DRUGS track. Honest by construction; writes its own constants.

REFIT v2 (2026-07-22) — after b-daemon + b-app reported four measured defects:
  1. tau_review > tau made the review tier unreachable        -> review is now a BAND BELOW
     violation on the same score, and the invariant is asserted here and in policy().
  2. a vape landed at violation                               -> tier arbitration + this
     script's acceptance set now regress it.
  3. a raspberry/bramble leaf scored p=0.92                   -> serrated/compound-leaf and
     benign-object negatives added, AND the root cause fixed: that same image was
     MISLABELLED as a positive in my ground truth (see labels.json hard_negatives).
  4. the logistic saturated: 218 violations all at p=0.99     -> fit on the full real-photo
     pool, plus a hard evidence cap P_MAX = (n+1)/(n+2), which makes p>=0.95 unreachable.

Positive slices, kept separate because they mean different things:
  A. `drug`  — 17 hand-verified drug images (18 minus the bramble mislabel), full-res audited.
  B. `proxy` — 26 LVIS val2017 tobacco/medicine paraphernalia + 10 Open Images `Syringe`.
  C. `amb`   — hand-marked ambiguous; scored, never counted right or wrong.
Negatives: every indexed real-photo corpus available (COCO val2017 + Unsplash pulls) plus
the non-drug images from the keyword probe. LVIS is federated, so the FP rate is an UPPER
bound; the top-scoring negatives are hand-inspected.

    uv run python scripts/eval_drugs.py [--write] [--datasets a,b,c]
`--write` patches the fitted constants back into src/imgtag/moderation/drugs.py.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imgtag.core import models, store  # noqa: E402
from imgtag.moderation import drugs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LVIS = ROOT / "data/lvis/lvis_val2017_only.json"
OI_SYRINGE = ROOT / "data/oi-drugs/syringe"
PROBE = ROOT / "data/drug-probe"
PROXY_CATS = {24: "ashtray", 140: "pipe_bowl", 258: "cigar_box", 259: "cigarette",
              260: "cigarette_case", 567: "hookah", 678: "matchbox", 683: "medicine",
              810: "tobacco_pipe", 1047: "syringe"}

# The six-image acceptance set is the cross-track regression suite. These two are ours.
ACCEPTANCE = {"O5BSKKHYiEU.jpg": "review",     # a person exhaling vape — ADR-14 review tier
              "lhPLeHgox9Q.jpg": "none"}       # raspberry/bramble leaf — must not flag


def lvis_positives() -> dict[str, str]:
    d = json.loads(LVIS.read_bytes())
    keep = {a["image_id"]: PROXY_CATS[a["category_id"]]
            for a in d["annotations"] if a["category_id"] in PROXY_CATS}
    return {f"{i:012d}.jpg": c for i, c in keep.items()}


def ap(p: np.ndarray, y: np.ndarray) -> float:
    o = np.argsort(-p)
    ys = np.asarray(y, bool)[o]
    tp = np.cumsum(ys)
    prec = tp / np.arange(1, len(ys) + 1)
    return float((prec * ys).sum() / max(ys.sum(), 1))


def rec_at_fpr(s: np.ndarray, y: np.ndarray, f: float) -> float:
    t = float(np.quantile(s[~y], 1 - f))
    return float((s[y] >= t).mean())


def ridge_platt(s, y, lam: float = 1e-3):
    """Platt scaling with an L2 penalty on the SLOPE.

    Unregularized Newton on 17 positives produced a razor-thin transition band — any
    corpus with a slightly heavier margin tail then saturated to p=0.99 (b-app's defect
    #4). The penalty keeps the band wide enough to survive a corpus shift; the evidence
    cap in drugs.P_MAX is the belt to this pair of braces.
    """
    s = np.asarray(s, float)
    y = np.asarray(y, bool)
    npos, nneg = float(y.sum()), float((~y).sum())
    if npos == 0 or nneg == 0:
        return 0.0, 0.0
    t = np.where(y, (npos + 1) / (npos + 2), 1 / (nneg + 2))   # Platt's prior-corrected targets
    A, B = 1.0, 0.0
    for _ in range(300):
        p = 1 / (1 + np.exp(-(A * s + B)))
        d = p - t
        w = np.maximum(p * (1 - p), 1e-9)
        g = np.array([d @ s + lam * A, d.sum()])
        H = np.array([[w @ (s * s) + lam, w @ s], [w @ s, w.sum()]]) + 1e-9 * np.eye(2)
        step = np.linalg.solve(H, g)
        A, B = A - step[0], B - step[1]
        if np.abs(step).max() < 1e-11:
            break
    return float(A), float(B)


def embed_dir(backend, paths: list[Path], cache: Path) -> np.ndarray:
    if cache.is_file():
        a = np.load(cache)
        if len(a) == len(paths):
            return a
    from PIL import Image
    out = []
    for p in paths:
        with Image.open(p) as im:
            out.append(backend.preprocess(im))
    e = np.asarray(backend.embed_images(np.stack(out)), np.float32)
    np.save(cache, e)
    return e


def main() -> int:
    a = argparse.ArgumentParser()
    a.add_argument("--datasets", default="cocoval2017,unsplash-demo,unsplashb",
                   help="indexed real-photo corpora used as the negative pool")
    a.add_argument("--model", default="pecore-s16-384")
    a.add_argument("--lam", type=float, default=1e-3)
    a.add_argument("--fp-budget", type=float, default=0.005, help="violation FP rate target")
    a.add_argument("--top", type=int, default=15)
    a.add_argument("--write", action="store_true", help="patch constants into drugs.py")
    args = a.parse_args()

    backend = models.load_backend(args.model, {})
    tag = backend.model_id
    emb, names, kind = [], [], []

    lv = lvis_positives()
    labels = json.loads((PROBE / "labels.json").read_bytes())
    tob_lab = set(labels.get("tobacco", []))
    # name -> kind for every hand-labelled drug-probe image, so it keeps its label no
    # matter which corpus it turns up in first (drug-probe ⊂ Unsplash by photo id).
    probe_kind = {n: "drug" for n in labels["drug"]}
    probe_kind.update({n: "tobacco" for n in tob_lab})
    probe_kind.update({n: "amb" for n in labels["ambiguous"]})
    probe_kind.update({n: "neg" for n in labels.get("verified_negatives", [])})
    # DEDUPE BY FILENAME: the Unsplash corpora overlap each other, and a labelled positive
    # appearing a second time under a different dataset was silently counted as a NEGATIVE
    # (it is what put a real cannabis photo at the top of the "false positive" list).
    seen: set[str] = set()
    for ds in [d for d in args.datasets.split(",") if d]:
        try:
            snap = store.open_snapshot(ds)
        except Exception as e:
            print(f"  (skipping {ds}: {type(e).__name__})")
            continue
        E = np.asarray(snap.emb, np.float32)
        keep = []
        for i, r in enumerate(snap.ids):
            n = Path(r["path"]).name
            if n in seen:
                continue
            seen.add(n)
            keep.append(i)
            names.append(n)
            # drug-probe labels WIN over corpus membership: the Unsplash corpora contain
            # the same photo ids as data/drug-probe, so a labelled cannabis photo would
            # otherwise be dropped into the negative pool by whichever corpus indexed it
            # first (it was — WOs7WulAfPw sat atop the "false positives"). Label first.
            kind.append(probe_kind.get(n, "tobacco" if n in tob_lab
                                       else "proxy" if n in lv else "neg"))
        emb.append(E[keep])
        print(f"  {ds}: {len(snap.ids)} rows, {len(keep)} new after dedupe")

    oi = sorted(OI_SYRINGE.glob("*.jpg"))
    if oi:
        emb.append(embed_dir(backend, oi, PROBE / f".oi-{tag}.npy"))
        names += [f"OI:{p.name}" for p in oi]
        kind += ["proxy"] * len(oi)

    for sub in ("strong", "med"):
        ps = sorted((PROBE / sub).glob("*.jpg"))
        if not ps:
            continue
        E = embed_dir(backend, ps, PROBE / f".{sub}-{tag}.npy")
        keep = []
        for i, pth in enumerate(ps):
            if pth.name in seen:
                continue
            seen.add(pth.name)
            keep.append(i)
            names.append(pth.name)
            kind.append("drug" if pth.name in labels["drug"] else
                        "tobacco" if pth.name in tob_lab else
                        "amb" if pth.name in labels["ambiguous"] else
                        "policy" if sub == "med" else "neg")
        emb.append(E[keep])

    emb = np.concatenate(emb)
    kind = np.array(kind)
    counts = {k: int((kind == k).sum()) for k in ("drug", "proxy", "tobacco", "amb", "policy", "neg")}
    print("slices:", counts)

    scorer = drugs.DrugsScorer.build(backend)
    cp = emb @ scorer.pos.T
    bgm = (emb @ scorer.bg.T).max(1)
    s = cp.max(1) - bgm                        # violation margin
    st = (emb @ scorer.tob.T).max(1) - bgm     # tobacco margin
    y, neg = kind == "drug", kind == "neg"
    m = y | neg

    out: dict = {"model": tag, "n": len(kind), "counts": counts,
                 "negative_pool": args.datasets,
                 "margin_negatives": {q: round(float(np.quantile(s[neg], q / 100)), 4)
                                      for q in (50, 90, 99, 99.9)} | {"max": round(float(s[neg].max()), 4)},
                 "margin_positives": {"min": round(float(s[y].min()), 4),
                                      "median": round(float(np.median(s[y])), 4),
                                      "max": round(float(s[y].max()), 4)}}
    for slice_ in ("drug", "proxy"):
        yy = kind == slice_
        mm = yy | neg
        out[slice_] = {"n_pos": int(yy.sum()), "AP": round(ap(s[mm], yy[mm]), 4),
                       "R@fpr1%": round(rec_at_fpr(s[mm], yy[mm], 0.01), 3),
                       "R@fpr5%": round(rec_at_fpr(s[mm], yy[mm], 0.05), 3)}
        print(slice_, out[slice_])

    # ── refit ────────────────────────────────────────────────────────────────
    A, B = ridge_platt(s[m], y[m], args.lam)
    P_MAX = drugs.P_MAX
    p = np.minimum(1 / (1 + np.exp(-(A * s + B))), P_MAX)
    pr = np.minimum(1 / (1 + np.exp(-(A * st + B))), P_MAX)

    tau = float(np.quantile(p[neg], 1 - args.fp_budget))
    tau = min(tau, float(np.sort(p[y])[::-1][max(0, int(np.ceil(0.90 * y.sum())) - 1)]))
    tau_review = float(np.quantile(p[neg], 1 - 3 * args.fp_budget))
    if tau_review >= tau:                       # defect #1: the band must not invert
        tau_review = tau * drugs.REVIEW_BAND
    out["platt"] = [round(A, 4), round(B, 4)]
    out["p_max_evidence_cap"] = round(P_MAX, 4)
    out["tau"], out["tau_review"] = round(tau, 4), round(tau_review, 4)
    assert tau_review < tau, "review band must sit BELOW violation"

    bins = [0, .01, .05, .1, .2, .3, .5, .7, .9, 1.0001]
    hist, _ = np.histogram(p[neg], bins=bins)
    out["p_histogram_negatives"] = {f"{bins[i]:g}-{bins[i+1]:g}": int(hist[i]) for i in range(len(hist))}
    out["p_histogram_drug"] = {f"{bins[i]:g}-{bins[i+1]:g}": int(h)
                               for i, h in enumerate(np.histogram(p[y], bins=bins)[0])}
    print("\nfit A=%.2f B=%.2f  tau=%.4f  tau_review=%.4f  (cap %.3f)" % (A, B, tau, tau_review, P_MAX))
    print("p histogram, negatives:", out["p_histogram_negatives"])
    print("p histogram, drug     :", out["p_histogram_drug"])

    # ── tiering with the shipped arbitration, so the numbers match the product ──
    viol = (p >= tau) & (s >= st + drugs.TIER_MARGIN)
    review = ~viol & ((pr >= tau_review) | (p >= tau))
    out["operating"] = {
        "violation_rate_neg": round(float(viol[neg].mean()), 4),
        "violation_count_neg": int(viol[neg].sum()),
        "review_rate_neg": round(float(review[neg].mean()), 4),
        "recall_drug_violation": round(float(viol[y].mean()), 3),
        "recall_drug_surfaced": round(float((viol | review)[y].mean()), 3),
        "recall_proxy_surfaced": round(float((viol | review)[kind == "proxy"].mean()), 3),
        "recall_tobacco_surfaced": round(float((viol | review)[kind == "tobacco"].mean()), 3)
        if (kind == "tobacco").any() else None,
        "tobacco_wrongly_violation": int(viol[kind == "tobacco"].sum()),
        "flag_rate_ambiguous": round(float((viol | review)[kind == "amb"].mean()), 3),
    }
    print("operating:", json.dumps(out["operating"]))

    # ── acceptance set (ours: vape -> review, leaf -> none) ──
    idx = {n: i for i, n in enumerate(names)}
    acc = {}
    for n, want in ACCEPTANCE.items():
        if n not in idx:
            acc[n] = {"expected": want, "got": "ABSENT"}
            continue
        i = idx[n]
        got = "violation" if viol[i] else "review" if review[i] else "none"
        acc[n] = {"expected": want, "got": got, "p": round(float(p[i]), 4),
                  "p_review": round(float(pr[i]), 4), "pass": got == want}
    out["acceptance"] = acc
    print("acceptance:", json.dumps(acc, indent=1))

    order = np.nonzero(neg)[0]
    order = order[np.argsort(-p[order])][: args.top]
    out["top_negatives"] = [{"name": names[i], "p": round(float(p[i]), 4),
                             "why": scorer.names[int(cp[i].argmax())]} for i in order]
    print("\ntop scoring negatives (hand-check these):")
    for r in out["top_negatives"]:
        print(f"  {r['p']:.3f} {r['name']:30s} {r['why']}")

    (ROOT / "research/eval-drugs.json").write_text(json.dumps(out, indent=1))
    if args.write:
        f = ROOT / "src/imgtag/moderation/drugs.py"
        src = f.read_text()
        src = re.sub(r"PLATT_A, PLATT_B = [-\d.]+, [-\d.]+",
                     f"PLATT_A, PLATT_B = {A:.4f}, {B:.4f}", src, count=1)
        src = re.sub(r"^TAU = [\d.]+", f"TAU = {tau:.4f}", src, count=1, flags=re.M)
        src = re.sub(r"^TAU_REVIEW = [\d.]+", f"TAU_REVIEW = {tau_review:.4f}", src, count=1, flags=re.M)
        f.write_text(src)
        print(f"\npatched constants into {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
