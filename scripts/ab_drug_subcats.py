#!/usr/bin/env python
"""A/B every taxonomy candidate prompt for TP-vs-FP SEPARATION. improve-track round 1.

The gate (chaser's law: precision dies in the tail): a candidate prompt is PROMOTED only
if adding it to its subcategory raises the drug-vs-negative separation (AUROC) OR fills a
subcategory with weak coverage, WITHOUT lifting the negative FP band (p99 of the margin
over 15k real photos). A prompt that only adds recall on the labelled 17 while lifting the
FP band is rejected — it would flag more ordinary photos in production.

One model load, one embed of the full pool + all candidate prompts, then pure numpy A/B.
Writes research/ab-drug-subcats.json. Does NOT mutate drugs.py — the conductor/agent reads
the verdict and promotes winners by hand (versioned, reviewable).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imgtag.core import models, store  # noqa: E402
from imgtag.moderation import drugs  # noqa: E402
from scripts.eval_drugs import auroc, lvis_positives  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "data/drug-probe"
POOL = ("cocoval2017", "unsplash-demo", "unsplashb")


def load_pool(backend):
    labels = json.loads((PROBE / "labels.json").read_bytes())
    lv, tob = lvis_positives(), set(labels.get("tobacco", []))
    pk = {n: "drug" for n in labels["drug"]}
    pk.update({n: "neg" for n in labels.get("verified_negatives", [])})
    emb, kind, seen = [], [], set()
    for ds in POOL:
        try:
            snap = store.open_snapshot(ds)
        except Exception:
            continue
        E = np.asarray(snap.emb, np.float32)
        keep = []
        for i, r in enumerate(snap.ids):
            n = Path(r["path"]).name
            if n in seen:
                continue
            seen.add(n)
            keep.append(i)
            kind.append(pk.get(n, "tobacco" if n in tob else "proxy" if n in lv else "neg"))
        emb.append(E[keep])
    for sub in ("strong", "med"):
        ps = sorted((PROBE / sub).glob("*.jpg"))
        cache = PROBE / f".{sub}-{backend.model_id}.npy"
        if not (ps and cache.is_file()):
            continue
        E = np.load(cache)
        keep = []
        for i, pth in enumerate(ps):
            if pth.name in seen:
                continue
            seen.add(pth.name)
            keep.append(i)
            kind.append("drug" if pth.name in labels["drug"] else
                        "tobacco" if pth.name in tob else
                        "amb" if pth.name in labels["ambiguous"] else
                        "neg")
        emb.append(E[keep])
    return np.concatenate(emb), np.array(kind)


def main() -> int:
    backend = models.load_backend("pecore-s16-384", {})
    E, kind = load_pool(backend)
    y, neg = kind == "drug", kind == "neg"
    print(f"pool {len(kind)}  drug {int(y.sum())}  neg {int(neg.sum())}")

    tax = json.loads((PROBE / "taxonomy.json").read_bytes())["subcategories"]
    bg = drugs.concept_vectors(backend, drugs.BACKGROUND)          # fixed negative bank
    bgm = (E @ bg.T).max(1)

    def sep(pos_prompts):
        V = drugs.concept_vectors(backend, pos_prompts)
        s = (E @ V.T).max(1) - bgm
        m = y | neg
        return (auroc(s[m], y[m]), float(np.quantile(s[neg], 0.99)),
                float(np.median(s[y])))

    shipped = [c for g in drugs.CONCEPTS.values() for c in g]
    base_au, base_fp99, base_tp = sep(shipped)
    print(f"BASELINE  AUROC {base_au:.4f}  FP-p99 {base_fp99:.4f}  TP-median {base_tp:.4f}")

    out = {"baseline": {"auroc": round(base_au, 4), "fp_p99": round(base_fp99, 4),
                        "tp_median": round(base_tp, 4), "n_prompts": len(shipped)},
           "candidates": [], "promote": [], "reject": []}

    # test each candidate INDIVIDUALLY (added to the full shipped set): does it help?
    for subcat, spec in tax.items():
        for cand in spec.get("candidates", []):
            au, fp99, tp = sep(shipped + [cand])
            d_au, d_fp = au - base_au, fp99 - base_fp99
            # PROMOTE: separation up (or flat) AND FP band not lifted more than a hair
            ok = (d_au >= -0.0005) and (d_fp <= 0.0010)
            rec = {"subcat": subcat, "prompt": cand, "d_auroc": round(d_au, 4),
                   "d_fp_p99": round(d_fp, 4), "verdict": "promote" if ok else "reject"}
            out["candidates"].append(rec)
            (out["promote"] if ok else out["reject"]).append(cand)
            print(f"  [{'PROMOTE' if ok else 'reject '}] {subcat:16s} dAUROC{d_au:+.4f} "
                  f"dFP99{d_fp:+.4f}  {cand[:48]}")

    # the WHOLE promoted set together (interactions can differ from one-at-a-time)
    if out["promote"]:
        au, fp99, tp = sep(shipped + out["promote"])
        out["promoted_together"] = {"auroc": round(au, 4), "d_auroc": round(au - base_au, 4),
                                    "fp_p99": round(fp99, 4), "d_fp_p99": round(fp99 - base_fp99, 4),
                                    "n_added": len(out["promote"])}
        print(f"\nPROMOTED TOGETHER (+{len(out['promote'])}): AUROC {au:.4f} "
              f"(d{au - base_au:+.4f})  FP-p99 {fp99:.4f} (d{fp99 - base_fp99:+.4f})")

    # per-subcategory separation, baseline (which kinds are weakest -> need the candidates most)
    grp = np.array([g for g, cs in drugs.CONCEPTS.items() for _ in cs])
    Vs = drugs.concept_vectors(backend, shipped)
    cp = E @ Vs.T
    best_grp = grp[cp.argmax(1)]
    out["per_subcat_auroc_baseline"] = {}
    for g in drugs.CONCEPTS:
        sel = y & (best_grp == g)
        if sel.sum() >= 1:
            s = cp.max(1) - bgm
            mm = sel | neg
            out["per_subcat_auroc_baseline"][g] = {
                "n_tp": int(sel.sum()), "auroc": round(auroc(s[mm], sel[mm]), 3)}
    print("per-subcat baseline:", json.dumps(out["per_subcat_auroc_baseline"]))

    (ROOT / "research/ab-drug-subcats.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote research/ab-drug-subcats.json  |  promote {len(out['promote'])}  "
          f"reject {len(out['reject'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
