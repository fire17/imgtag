#!/usr/bin/env python3
"""Fit + save the people person-cascade AND produce the measured report tables.

TRAIN/EVAL SPLIT: COCO val2017 crowd-free images, 60/40 by a fixed seed. Faces come from
the cached YuNet sweep (.scratch/yunet-coco.json — raw detections, thresholded here so
every tau is a re-read, TRACKS.md T1); persons come from the cascade over the pecore
embedding the index already computed. Ground truth is COCO instances (persons, exhaustive)
+ keypoints (faces, PROXY — its blind spot is measured, never hidden).

    .venv/bin/python scripts/train_people_head.py --save     # writes the shipped cascade
    .venv/bin/python scripts/train_people_head.py --report .scratch/people-report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research/bench_scripts"))

from imgtag.core.store import open_snapshot                       # noqa: E402
from imgtag.core.tags import fit_platt, platt_apply               # noqa: E402
from imgtag.moderation.people import PersonCascade, cascade_path  # noqa: E402
from imgtag.moderation.weapons import (                           # noqa: E402  (reuse)
    fit_logistic, prf, tau_for_recall,
)
from people_gt import build as build_gt                           # noqa: E402

SEED = 20260722
DATASET = "cocoval2017"
SWEEP = ROOT / ".scratch/yunet-coco.json"


def load() -> tuple[np.ndarray, list[str], dict]:
    snap = open_snapshot(DATASET)
    emb = np.asarray(snap.emb, np.float32)
    names = [Path(r["path"]).name for r in snap.ids]
    return emb, names, build_gt()


def aligned(emb, names, gt, sweep):
    """Crowd-free rows with (emb, gt, n_faces@thresholds)."""
    by = {v["file_name"]: v for v in gt.values()}
    rows = []
    for i, n in enumerate(names):
        g = by.get(n)
        if g is None or g["crowd"]:
            continue
        det = sweep.get(n, {"scores": []})
        rows.append((i, g, np.asarray(det["scores"], np.float32)))
    return rows


def faces_at(scores: np.ndarray, tau: float) -> int:
    return int((scores >= tau).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()

    emb, names, gt = load()
    sweep = json.loads(SWEEP.read_text()) if SWEEP.is_file() else {}
    rows = aligned(emb, names, gt, sweep)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(rows))
    cut = int(len(rows) * 0.6)
    tr_idx, te_idx = perm[:cut], perm[cut:]

    R = {"n": len(rows), "n_train": int(cut), "n_test": len(rows) - cut}

    # ---- person cascade (free, over the shared embedding) --------------------
    E = np.stack([emb[rows[i][0]] for i in range(len(rows))])
    y1 = np.array([rows[i][1]["n_persons"] >= 1 for i in range(len(rows))], bool)
    y2 = np.array([rows[i][1]["n_persons"] >= 2 for i in range(len(rows))], bool)
    w1, b1 = fit_logistic(E[tr_idx], y1[tr_idx].astype(np.float64))
    w2, b2 = fit_logistic(E[tr_idx], y2[tr_idx].astype(np.float64))
    s1_tr, s1_te = E[tr_idx] @ w1 + b1, E[te_idx] @ w1 + b1
    s2_tr, s2_te = E[tr_idx] @ w2 + b2, E[te_idx] @ w2 + b2
    pl1, pl2 = fit_platt(s1_tr, y1[tr_idx]), fit_platt(s2_tr, y2[tr_idx])
    p1_te, p2_te = platt_apply(s1_te, pl1), platt_apply(s2_te, pl2)
    # tau1: recall-first (a missed person is the worst error for a counting track).
    # tau2: chosen to maximize 3-WAY bucket accuracy (0 / 1 / 2+), because ONE n_persons
    # is stored per image, so the metric that matters is the bucket that count lands in —
    # not either binary's F1. A max-F1-on->=2 tau over-fires multi and steals one-person;
    # a precision-leaning tau does the reverse. Bucket accuracy is the honest middle, and
    # the operator re-derives at read (T1) if their site wants a different boundary.
    tau1 = tau_for_recall(platt_apply(s1_tr, pl1), y1[tr_idx], 0.92)
    p1_tr, p2_tr = platt_apply(s1_tr, pl1), platt_apply(s2_tr, pl2)
    faces_tr = np.array([faces_at(rows[i][2], 0.6) for i in tr_idx])
    pgb_tr = np.minimum(np.array([rows[i][1]["n_persons"] for i in tr_idx]), 2)
    best = (0.5, -1.0)
    for q in np.linspace(0.30, 0.98, 69):
        t2 = float(np.quantile(p2_tr, q))
        casc = np.where(p1_tr < tau1, 0, np.where(p2_tr < t2, 1, 2))
        acc = float((np.minimum(np.maximum(faces_tr, casc), 2) == pgb_tr).mean())
        if acc > best[1]:
            best = (t2, acc)
    tau2 = best[0]
    R["person>=1"] = prf(p1_te, y1[te_idx], tau1)
    R["person>=2"] = prf(p2_te, y2[te_idx], tau2)

    # Key by BACKEND NAME (weapons convention), NOT the -fp32 model_id: the dispatcher
    # resolves a head from the machine profile's backend alone, without a model load.
    from imgtag.core.models import DEFAULT_BACKEND
    backend_name = DEFAULT_BACKEND
    cascade = PersonCascade(model_id=backend_name, dim=int(E.shape[1]),
                            w1=w1, b1=float(b1), platt1=list(pl1),
                            w2=w2, b2=float(b2), platt2=list(pl2),
                            tau1=float(tau1), tau2=float(tau2),
                            metrics={"person>=1": R["person>=1"], "person>=2": R["person>=2"],
                                     "n_test": R["n_test"], "source": "coco-val2017 60/40"})

    # ---- face threshold sweep vs the COCO keypoint proxy ---------------------
    face_gt = np.array([rows[i][1]["n_faces"] for i in range(len(rows))])
    face_tbl = {}
    for tau in (0.5, 0.6, 0.7, 0.8, 0.9):
        pred = np.array([faces_at(rows[i][2], tau) for i in range(len(rows))])
        te = te_idx
        exact = float((pred[te] == face_gt[te]).mean())
        # >=1 and >=2 detection quality vs proxy
        d1 = prf((pred >= 1).astype(float)[te], (face_gt >= 1)[te], 0.5)
        d2 = prf((pred >= 2).astype(float)[te], (face_gt >= 2)[te], 0.5)
        mae = float(np.abs(pred[te] - face_gt[te]).mean())
        face_tbl[f"{tau:.1f}"] = {"exact_count_acc": exact, "count_mae": mae,
                                  "face>=1": {k: d1[k] for k in ("precision", "recall", "f1")},
                                  "face>=2": {k: d2[k] for k in ("precision", "recall", "f1")}}
    R["face_tau_sweep"] = face_tbl

    # ---- the four DERIVED categories at the chosen operating point -----------
    TAU_FACE = 0.6
    faces_pred = np.array([faces_at(rows[i][2], TAU_FACE) for i in range(len(rows))])
    casc = np.where(p1_te < tau1, 0, np.where(p2_te < tau2, 1, 2))
    faces_te = faces_pred[te_idx]
    persons_pred = np.maximum(faces_te, casc)
    persons_gt = np.array([rows[i][1]["n_persons"] for i in range(len(rows))])[te_idx]
    faces_gt_te = face_gt[te_idx]

    def cat_prf(pred_on, true_on):
        return prf(pred_on.astype(float), true_on, 0.5)

    R["derived"] = {
        "one-person": cat_prf(persons_pred == 1, persons_gt == 1),
        "multi-person": cat_prf(persons_pred >= 2, persons_gt >= 2),
        "one-face": cat_prf(faces_te == 1, faces_gt_te == 1),
        "multi-face": cat_prf(faces_te >= 2, faces_gt_te >= 2),
    }
    R["person_exact_count_acc"] = float((persons_pred == np.minimum(persons_gt, 2)
                                         if False else (persons_pred == persons_gt)).mean())
    R["person_bucket_acc"] = float((np.minimum(persons_pred, 2) == np.minimum(persons_gt, 2)).mean())

    # ---- back-view cost: persons with NO visible face (the user's hard case) -
    backview = [i for i in te_idx if rows[i][1]["n_persons"] >= 1 and rows[i][1]["n_faces"] == 0]
    if backview:
        bp = np.array([max(faces_at(rows[i][2], TAU_FACE),
                           0 if platt_apply(emb[rows[i][0]] @ w1 + b1, pl1) < tau1
                           else (1 if platt_apply(emb[rows[i][0]] @ w2 + b2, pl2) < tau2 else 2))
                       for i in backview])
        R["backview"] = {"n": len(backview),
                         "recovered_by_cascade": int((bp >= 1).sum()),
                         "recall": float((bp >= 1).mean()),
                         "faces_alone_would_get": 0.0}

    if args.save:
        p = cascade_path(cascade.model_id, ROOT / "src/imgtag/data/moderation")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(cascade.to_json()))
        print(f"saved cascade -> {p}")

    print(json.dumps({k: v for k, v in R.items() if k != "face_tau_sweep"}, indent=1, default=float))
    print("\nface tau sweep (exact-count acc / MAE / f1@>=1 / f1@>=2):")
    for t, d in face_tbl.items():
        print(f"  tau={t}: acc={d['exact_count_acc']:.3f} mae={d['count_mae']:.3f} "
              f"f1>=1={d['face>=1']['f1']:.3f} f1>=2={d['face>=2']['f1']:.3f}")
    if args.report:
        args.report.write_text(json.dumps(R, indent=1, default=float))
        print(f"\nwrote {args.report}")


if __name__ == "__main__":
    main()
