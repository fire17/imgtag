#!/usr/bin/env python
"""Measure the DRUGS track on every label that actually exists. Honest by construction.

Three labelled positive slices, kept SEPARATE in every number because they mean different
things:

  A. `drug`  — 16 Unsplash photos of real drug imagery (cannabis plants/buds, bongs, joints
     being smoked, a vape cartridge), hand-verified by eye from keyword candidates.
     THE ONLY slice that is the category the user actually asked about.
  B. `proxy` — 26 LVIS val2017 images labelled ashtray / cigarette / cigarette_case /
     matchbox / tobacco_pipe / medicine + 10 Open Images `Syringe` test images. Smoking and
     medical paraphernalia: adjacent, human-labelled, and mostly TINY objects in scenes.
  C. `amb`   — 10 hand-marked ambiguous images (a lighter, unidentifiable leaves, someone
     exhaling smoke). Scored and reported, never counted as right or wrong.

Negatives: COCO val2017 minus the LVIS positives (verified-ish, federated caveat) PLUS the
174 non-drug images from the Unsplash keyword pull — hard negatives, since they are what a
'weed'/'hemp' keyword search actually returns (ferns, houseplants, fields).

    uv run python scripts/eval_drugs.py [--dataset cocoval2017] [--model pecore-s16-384]
"""
from __future__ import annotations

import argparse
import json
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
# LVIS ids, drug-ADJACENT only (mushroom/spoon/pipe excluded: food and plumbing)
PROXY_CATS = {24: "ashtray", 140: "pipe_bowl", 258: "cigar_box", 259: "cigarette",
              260: "cigarette_case", 567: "hookah", 678: "matchbox", 683: "medicine",
              810: "tobacco_pipe", 1047: "syringe"}


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


def tau_for_recall(p: np.ndarray, y: np.ndarray, target: float) -> float:
    pos = np.sort(p[np.asarray(y, bool)])[::-1]
    k = max(1, int(np.ceil(target * len(pos))))
    return float(pos[min(k, len(pos)) - 1])


def rec_at_fpr(s: np.ndarray, y: np.ndarray, f: float) -> float:
    t = float(np.quantile(s[~y], 1 - f))
    return float((s[y] >= t).mean())


def embed_dir(backend, paths: list[Path], cache: Path) -> np.ndarray:
    """Embed a folder once; cache beside it (re-runs are free)."""
    if cache.is_file():
        a = np.load(cache)
        if len(a) == len(paths):
            return a
    from PIL import Image
    out = []
    for p in paths:
        with Image.open(p) as im:
            out.append(backend.preprocess(im))
        if len(out) % 64 == 0:
            print(f"  embedding {cache.stem}: {len(out)}/{len(paths)}", flush=True)
    e = np.asarray(backend.embed_images(np.stack(out)), np.float32)
    np.save(cache, e)
    return e


def main() -> int:
    a = argparse.ArgumentParser()
    a.add_argument("--dataset", default="cocoval2017")
    a.add_argument("--model", default="pecore-s16-384")
    a.add_argument("--top", type=int, default=15)
    a.add_argument("--tobacco", action="store_true")
    args = a.parse_args()

    backend = models.load_backend(args.model, {})
    tag = backend.model_id

    snap = store.open_snapshot(args.dataset)
    emb = [np.asarray(snap.emb, np.float32)]
    names = [Path(r["path"]).name for r in snap.ids]
    lv = lvis_positives()
    kind = ["proxy" if n in lv else "neg" for n in names]
    print(f"{args.dataset}: {len(names)} rows · LVIS proxy positives present: "
          f"{sum(k == 'proxy' for k in kind)}")

    oi = sorted(OI_SYRINGE.glob("*.jpg"))
    if oi:
        emb.append(embed_dir(backend, oi, PROBE / f".oi-{tag}.npy"))
        names += [f"OI:{p.name}" for p in oi]
        kind += ["proxy"] * len(oi)

    labels = json.loads((PROBE / "labels.json").read_bytes())
    for sub in ("strong", "med"):
        ps = sorted((PROBE / sub).glob("*.jpg"))
        if not ps:
            continue
        emb.append(embed_dir(backend, ps, PROBE / f".{sub}-{tag}.npy"))
        for p in ps:
            names.append(f"{sub}:{p.name}")
            kind.append("drug" if p.name in labels["drug"] else
                        "amb" if p.name in labels["ambiguous"] else
                        "policy" if sub == "med" else "neg")

    emb = np.concatenate(emb)
    kind = np.array(kind)
    print({k: int((kind == k).sum()) for k in ("drug", "proxy", "amb", "policy", "neg")})

    scorer = drugs.DrugsScorer.build(backend, tobacco=args.tobacco)
    cp = emb @ scorer.pos.T
    s = cp.max(1) - (emb @ scorer.bg.T).max(1)          # the shipped feature

    out: dict = {"model": tag, "n": len(kind), "tobacco": args.tobacco,
                 "counts": {k: int((kind == k).sum()) for k in set(kind)}}
    neg = kind == "neg"
    for slice_ in ("drug", "proxy"):
        y = kind == slice_
        m = y | neg
        out[slice_] = {
            "n_pos": int(y.sum()),
            "AP": round(ap(s[m], y[m]), 4),
            "R@fpr1%": round(rec_at_fpr(s[m], y[m], 0.01), 3),
            "R@fpr5%": round(rec_at_fpr(s[m], y[m], 0.05), 3),
            "R@fpr10%": round(rec_at_fpr(s[m], y[m], 0.10), 3),
        }
        print(slice_, out[slice_])

    # ── ship the thresholds: recall-first per ADR-14 tier ──
    y = kind == "drug"
    m = y | neg
    from imgtag.core.tags import fit_platt
    A, B = fit_platt(s[m], y[m])
    A, B = -A, -B                       # tags.fit_platt is sigmoid(-(As+B))
    p = drugs._sigmoid(A * s + B)
    for target in (0.90, 0.95):
        t = tau_for_recall(p[m], y[m], target)
        out[f"violation@r{int(target * 100)}"] = {
            "tau": round(t, 4),
            "fp_rate_neg": round(float((p[neg] >= t).mean()), 4),
            "fp_count_neg": int((p[neg] >= t).sum()),
            "recall_drug": round(float((p[y] >= t).mean()), 3),
            "recall_proxy": round(float((p[kind == "proxy"] >= t).mean()), 3),
            "flag_rate_ambiguous": round(float((p[kind == "amb"] >= t).mean()), 3),
            "flag_rate_tobacco_medicine_keywords": round(float((p[kind == "policy"] >= t).mean()), 3),
        }
        print(f"violation r{int(target*100)}", out[f"violation@r{int(target * 100)}"])
    out["platt"] = [round(A, 4), round(B, 4)]

    # REVIEW tier (tobacco/vape, ADR-14): same logistic, tobacco bank, fitted on the
    # human-labelled LVIS smoking-paraphernalia slice.
    tob = drugs.concept_vectors(backend, drugs.TOBACCO)
    sr = (emb @ tob.T).max(1) - (emb @ scorer.bg.T).max(1)
    pr = drugs._sigmoid(A * sr + B)
    yp = kind == "proxy"
    mp = yp | neg
    out["review_tier"] = {"n_pos": int(yp.sum()), "AP": round(ap(sr[mp], yp[mp]), 4)}
    for target in (0.80, 0.90):
        t = tau_for_recall(pr[mp], yp[mp], target)
        out["review_tier"][f"@r{int(target * 100)}"] = {
            "tau": round(t, 4),
            "fp_rate_neg": round(float((pr[neg] >= t).mean()), 4),
            "recall_lvis_tobacco": round(float((pr[yp] >= t).mean()), 3),
            "flag_rate_tobacco_keywords": round(float((pr[kind == "policy"] >= t).mean()), 3),
        }
    print("review", json.dumps(out["review_tier"]))

    tau = tau_for_recall(p[m], y[m], 0.95)
    top = np.nonzero(neg)[0]
    top = top[np.argsort(-p[top])][: args.top]
    out["top_false_positives"] = [
        {"name": names[i], "p": round(float(p[i]), 4),
         "why": scorer.names[int(cp[i].argmax())]} for i in top]
    print("\ntop scoring negatives (hand-check these):")
    for r in out["top_false_positives"]:
        print(f"  {r['p']:.3f} {r['name']:34s} {r['why']}")
    print(f"\nshipped tau (recall .95 on the drug slice) = {tau:.4f}")

    dest = ROOT / f"research/eval-drugs{'-tobacco' if args.tobacco else ''}.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
