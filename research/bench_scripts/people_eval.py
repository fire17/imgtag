#!/usr/bin/env python3
"""track-people evaluation: can the SHARED EMBEDDING count people, or do we need optics?

TRACKS.md T2 forces this question to be asked before any dedicated model is proposed:
rung 1 (embedding matvec) is "the default and the only instrument that is unconditionally
allowed", so a dedicated detector must be JUSTIFIED BY MEASUREMENT, not by intuition.

Measured here, all first-party, all on COCO val2017 ground truth:
  A. zero-shot prompt ensemble          (rung 1, free, untrained)
  B. trained ordinal cascade probe      (rung 1/2, free, trained on our own embeddings)
  C. [people_yunet.py] YuNet face optics (rung 3, dedicated, budgeted)

Embeddings are NOT recomputed: the `cocoval2017` dataset is already indexed with
pecore-s16-384-fp32, so this whole sweep is a few matmuls.

    .venv/bin/python research/bench_scripts/people_eval.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from imgtag.core.store import open_snapshot            # noqa: E402
from imgtag.moderation.weapons import fit_logistic     # noqa: E402  (reuse, ADR-7)
from imgtag.core.tags import fit_platt, platt_apply    # noqa: E402
from people_gt import build as build_gt                # noqa: E402

SEED = 20260722
DATASET = "cocoval2017"

#: Zero-shot probes. "one person" vs "many people" is deliberately phrased the way a
#: user would ask, because that is the claim under test: does the embedding carry COUNT?
ZS_PROMPTS = {
    "person>=1": (["a photo of a person", "a photo of people", "a person",
                   "a photo of a man", "a photo of a woman", "a portrait of someone"],
                  ["a photo of an empty landscape", "a photo of an empty room",
                   "a photo of food", "a photo of an animal", "a photo of a building",
                   "a photo of an object", "a photo of a vehicle"]),
    "person>=2": (["a photo of two people", "a photo of a group of people",
                   "a photo of a crowd", "several people together", "many people",
                   "a photo of three people"],
                  ["a photo of one person alone", "a portrait of a single person",
                   "a photo of a person by themselves", "a photo with nobody in it",
                   "an empty scene"]),
    "face>=1": (["a photo of a face", "a close-up of a person's face",
                 "a visible human face", "a portrait showing a face"],
                ["a photo of a person from behind", "the back of someone's head",
                 "a photo with no faces", "an empty landscape", "a photo of an object"]),
    "face>=2": (["a photo of two faces", "several faces visible",
                 "a group photo showing many faces", "multiple people facing the camera"],
                ["a photo of one face", "a single portrait",
                 "a photo of a person from behind", "a photo with no faces"]),
}


def load_embeddings() -> tuple[np.ndarray, list[str]]:
    """[N,D] float32 + the COCO file_name aligned to each row (TRACKS.md alignment law)."""
    snap = open_snapshot(DATASET)
    emb = np.asarray(snap.emb, np.float32)
    names = [Path(r["path"]).name for r in snap.ids]
    assert len(names) == emb.shape[0], "row alignment broken"
    return emb, names


def targets(gt: dict, names: list[str]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Row mask of CLEAN (non-crowd) rows + the four binary targets, row-aligned."""
    by_name = {v["file_name"]: v for v in gt.values()}
    keep, y = [], {k: [] for k in ("person>=1", "person>=2", "face>=1", "face>=2")}
    for i, n in enumerate(names):
        g = by_name.get(n)
        if g is None or g["crowd"]:
            continue
        keep.append(i)
        y["person>=1"].append(g["n_persons"] >= 1)
        y["person>=2"].append(g["n_persons"] >= 2)
        y["face>=1"].append(g["n_faces"] >= 1)
        y["face>=2"].append(g["n_faces"] >= 2)
    return np.asarray(keep), {k: np.asarray(v, bool) for k, v in y.items()}


def split(n: int, frac: float = 0.6) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    p = rng.permutation(n)
    c = int(n * frac)
    return p[:c], p[c:]


def metrics(p: np.ndarray, y: np.ndarray, tau: float) -> dict:
    yp = p >= tau
    tp = int((yp & y).sum()); fp = int((yp & ~y).sum()); fn = int((~yp & y).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    return {"precision": prec, "recall": rec,
            "f1": 2 * prec * rec / max(prec + rec, 1e-12),
            "acc": float((yp == y).mean()), "tau": float(tau)}


def average_precision(p: np.ndarray, y: np.ndarray) -> float:
    o = np.argsort(-p); ys = y[o]
    tp = np.cumsum(ys); prec = tp / np.arange(1, len(ys) + 1)
    return float((prec * ys).sum() / max(ys.sum(), 1))


def best_tau(p: np.ndarray, y: np.ndarray) -> float:
    """Max-F1 threshold — the GENEROUS choice, so a failure here is a real failure."""
    cand = np.quantile(p, np.linspace(0.01, 0.99, 99))
    return float(max(cand, key=lambda t: metrics(p, y, t)["f1"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    gt = build_gt()
    emb, names = load_embeddings()
    keep, y = targets(gt, names)
    X = emb[keep]
    n = len(keep)
    tr, te = split(n)
    print(f"rows {n} (clean)  ·  train {len(tr)}  ·  held-out {len(te)}  ·  dim {X.shape[1]}\n")

    out: dict = {"n": n, "n_train": int(len(tr)), "n_test": int(len(te)), "tasks": {}}

    # ---- A. zero-shot prompt ensemble (free, untrained) ----------------------
    zs: dict[str, dict] = {}
    try:
        from imgtag.core.models import load_backend
        be = load_backend("pecore-s16-384")
        for task, (pos, neg) in ZS_PROMPTS.items():
            P = be.embed_texts(pos).astype(np.float32)
            N = be.embed_texts(neg).astype(np.float32)
            m = (X @ P.T).max(1) - (X @ N.T).max(1)
            yt = y[task]
            zs[task] = {"ap": average_precision(m[te], yt[te]),
                        **metrics(m[te], yt[te], best_tau(m[tr], yt[tr]))}
    except Exception as e:                                   # noqa: BLE001
        print(f"zero-shot skipped: {type(e).__name__}: {e}\n")

    # ---- B. trained ordinal cascade probe (free, trained) --------------------
    probe: dict[str, dict] = {}
    for task in y:
        yt = y[task]
        w, b = fit_logistic(X[tr], yt[tr].astype(np.float64))
        s_tr, s_te = X[tr] @ w + b, X[te] @ w + b
        pl = fit_platt(s_tr, yt[tr])
        p_te, p_tr = platt_apply(s_te, pl), platt_apply(s_tr, pl)
        probe[task] = {"ap": average_precision(p_te, yt[te]),
                       **metrics(p_te, yt[te], best_tau(p_tr, yt[tr])),
                       "prevalence": float(yt[te].mean())}

    hdr = f"{'task':<12} {'prev':>6} | {'zs AP':>7} {'zs F1':>6} {'zs rec':>7} | {'probe AP':>9} {'F1':>6} {'rec':>6} {'prec':>6} {'acc':>6}"
    print(hdr); print("-" * len(hdr))
    for task in ("person>=1", "person>=2", "face>=1", "face>=2"):
        z, pr = zs.get(task, {}), probe[task]
        print(f"{task:<12} {pr['prevalence']:>6.1%} | "
              f"{z.get('ap', float('nan')):>7.3f} {z.get('f1', float('nan')):>6.3f} {z.get('recall', float('nan')):>7.3f} | "
              f"{pr['ap']:>9.3f} {pr['f1']:>6.3f} {pr['recall']:>6.3f} {pr['precision']:>6.3f} {pr['acc']:>6.3f}")
        out["tasks"][task] = {"zeroshot": z, "probe": pr}

    # ---- exact-count accuracy of the DERIVED categories ----------------------
    # The user's question is not "is there a person" but "is there EXACTLY one".
    # A cascade that is individually strong can still be weak on the conjunction.
    print("\nDERIVED four-category accuracy on held-out (cascade: >=1 AND NOT >=2):")
    cas = {}
    for kind in ("person", "face"):
        p1 = probe[f"{kind}>=1"]; p2 = probe[f"{kind}>=2"]
        w1, b1 = fit_logistic(X[tr], y[f"{kind}>=1"][tr].astype(np.float64))
        w2, b2 = fit_logistic(X[tr], y[f"{kind}>=2"][tr].astype(np.float64))
        pl1 = fit_platt(X[tr] @ w1 + b1, y[f"{kind}>=1"][tr])
        pl2 = fit_platt(X[tr] @ w2 + b2, y[f"{kind}>=2"][tr])
        q1 = platt_apply(X[te] @ w1 + b1, pl1) >= p1["tau"]
        q2 = platt_apply(X[te] @ w2 + b2, pl2) >= p2["tau"]
        pred_one, pred_multi = q1 & ~q2, q2
        true_one = y[f"{kind}>=1"][te] & ~y[f"{kind}>=2"][te]
        true_multi = y[f"{kind}>=2"][te]
        for lab, pd, td in (("one", pred_one, true_one), ("multi", pred_multi, true_multi)):
            m = metrics(pd.astype(float), td, 0.5)
            cas[f"{lab}-{kind}"] = m
            print(f"  {lab}-{kind:<8} P={m['precision']:.3f} R={m['recall']:.3f} "
                  f"F1={m['f1']:.3f}  (true n={int(td.sum())})")
    out["derived"] = cas

    if args.json:
        args.json.write_text(json.dumps(out, indent=1))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
