#!/usr/bin/env python
"""Person-presence pre-filter for safetyprobe, then re-measure alert separation (round 3).

Round 2 found the probe's keyword TP labels are polluted by NON-PERSON images (a kitten on
a sofa, a laptop flat-lay, a first-aid kit) — 4/4 diagnostic views mislabelled. Ruling
(team-lead 2026-07-22): gate the pull on person-present using track-people's detector, T4-
clean (mechanical, no hand labels), then re-run the separation. If the alert CI lower bound
clears 0.5, the alert tier ships.

PERSON-PRESENCE = (YuNet face detected) OR (person-presence prompt margin over the SAME
embedding ≥ τ). YuNet is face-only (misses backs/cropped bodies — the user's explicit case);
the free embedding margin catches faceless people. τ is fitted on COCO val2017 (person-
keypoint images = present, empty images = absent) at a conservative 95%-precision point, so
the filter DROPS no-person noise without dropping real faceless people. The fitted cascade
in track-people is absent for this backend, so the prompt margin is its stand-in — noted,
not hidden.

    uv run python scripts/filter_safetyprobe_persons.py [--json OUT]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imgtag.core import models, store  # noqa: E402
from imgtag.moderation import people, safety  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "data/safety-probe"

PERSON = ["a photo of a person", "a person", "people", "a human being",
          "a man", "a woman", "a child", "a person seen from behind"]
NO_PERSON = ["an empty room", "a landscape with no people", "an object", "food on a plate",
             "an animal", "a piece of furniture", "a laptop on a table", "a cat", "a dog",
             "an empty street", "a close-up of an object", "a first aid kit"]

BENIGN_LYING = ("person_down", "sunbathing")


def _vecs(backend, texts):
    v = np.asarray(backend.embed_texts(texts), np.float32)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def fit_person_tau(backend, target_prec=0.95) -> tuple[float, np.ndarray, np.ndarray]:
    """τ for the person-presence margin, fitted on COCO keypoint(present) vs empty(absent)."""
    import math
    kp = json.loads((ROOT / "data/coco/annotations/person_keypoints_val2017.json").read_bytes())
    inst = json.loads((ROOT / "data/coco/annotations/instances_val2017.json").read_bytes())
    imgs = {im["id"] for im in kp["images"]}
    has_person = {a["image_id"] for a in inst["annotations"] if a["category_id"] == 1}
    empty = imgs - has_person
    snap = store.open_snapshot("cocoval2017")
    emb = np.asarray(snap.emb, np.float32)
    row = {int(os.path.basename(r["path"])[:-4]): r["row"] for r in snap.ids}
    P, N = _vecs(backend, PERSON), _vecs(backend, NO_PERSON)
    m = (emb @ P.T).max(1) - (emb @ N.T).max(1)
    y = np.zeros(len(emb), bool)
    for i in has_person:
        if i in row:
            y[row[i]] = True
    neg = np.zeros(len(emb), bool)
    for i in empty:
        if i in row:
            neg[row[i]] = True
    sel = y | neg
    o = np.argsort(-m[sel])
    ys = y[sel][o]
    tp = np.cumsum(ys)
    prec = tp / np.arange(1, len(ys) + 1)
    ok = np.where(prec >= target_prec)[0]
    tau = float(m[sel][o][ok[-1]]) if len(ok) else float(np.median(m[sel]))
    recall = float((m[y] >= tau).mean())
    print(f"person-margin τ fit (COCO): τ={tau:.4f} @ {target_prec:.0%} precision, "
          f"present-recall {recall:.3f} ({int(y.sum())} present / {int(neg.sum())} empty)")
    return tau, P, N


def main() -> int:
    a = argparse.ArgumentParser()
    a.add_argument("--json", default="")
    a.add_argument("--model", default="pecore-s16-384")
    args = a.parse_args()

    backend = models.load_backend(args.model, {}, vision=False)
    snap = store.open_snapshot("safetyprobe")
    emb = np.asarray(snap.emb, np.float32)
    paths = [ROOT / r["path"] for r in snap.ids]
    names = [os.path.basename(str(p)) for p in paths]

    tau_p, P, N = fit_person_tau(backend)
    pm = (emb @ P.T).max(1) - (emb @ N.T).max(1)
    margin_present = pm >= tau_p

    # YuNet face pass (person evidence the embedding margin can miss / confirm)
    head = people.load_people_head({"model": args.model})
    if head is None:
        print("WARN: YuNet head unavailable — filtering on the embedding margin alone")
        face_present = np.zeros(len(emb), bool)
    else:
        from PIL import Image
        face_present = np.zeros(len(emb), bool)
        for i, p in enumerate(paths):
            try:
                with Image.open(p) as im:
                    nf, _ = head.faces(im.convert("RGB"))
                face_present[i] = nf >= 1
            except Exception:
                pass
            if (i + 1) % 200 == 0:
                print(f"  YuNet {i + 1}/{len(paths)}")

    person = margin_present | face_present
    print(f"person-present: {int(person.sum())}/{len(person)} "
          f"(face {int(face_present.sum())}, margin {int(margin_present.sum())}, "
          f"both {int((face_present & margin_present).sum())})")

    # labels + filtered membership
    lab = json.loads((PROBE / "labels.json").read_bytes())
    sub = {}
    for k, v in lab.items():
        if k.startswith("_"):
            continue
        for nm in v:
            sub[nm] = k
    subarr = np.array([sub.get(n, "unlabelled") for n in names])

    # how much of each lying subcat is actually person-present (the pollution measurement)
    print("\nperson-present rate by subcategory:")
    keep = collections.OrderedDict()
    for k in ("alert_tp", "person_down", "sunbathing", "injury_context",
              "danger_context", "destruction"):
        m = subarr == k
        if not m.any():
            continue
        rate = float(person[m].mean())
        keep[k] = {"n": int(m.sum()), "person_present": int(person[m].sum()),
                   "rate": round(rate, 3)}
        print(f"  {k:16s} {int(person[m].sum()):3d}/{int(m.sum()):3d}  ({rate:.0%})")

    # ── re-run the alert separation on the PERSON-PRESENT subset ─────────────────────────
    sc = safety.SafetyScorer.build(backend)
    pl = safety.lying_prob(sc.lying_margin(emb))
    pd = safety.danger_prob(sc.danger_margin(emb))
    import importlib.util
    spec = importlib.util.spec_from_file_location("sep", ROOT / "scripts/eval_safety_separation.py")
    sep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sep)

    def separation(mask):
        alert = (subarr == "alert_tp") & mask
        benign = np.isin(subarr, BENIGN_LYING) & mask
        s = alert | benign
        if alert.sum() < 3 or benign.sum() < 3:
            return None
        ap = sep.ap(pd[s], alert[s])
        ci = sep.boot_ci(pd[s], alert[s], sep.ap)
        return {"n_alert": int(alert.sum()), "n_benign": int(benign.sum()),
                "AP": round(ap, 4), "CI95": [round(ci[0], 4), round(ci[1], 4)]}

    before = separation(np.ones(len(emb), bool))
    after = separation(person)
    print(f"\nSEPARATION (alert_tp vs benign-lying, p_danger):")
    print(f"  BEFORE filter: {before}")
    print(f"  AFTER  filter: {after}")
    ci_lo = (after or {}).get("CI95", [0])[0]
    verdict = "SHIP alert" if ci_lo > 0.5 else "WITHHOLD alert (still)"
    print(f"  VERDICT: {verdict} (CI lower {ci_lo:.3f}; gate 0.5)")

    out = {"model": backend.model_id, "person_tau": round(tau_p, 4),
           "person_present_total": int(person.sum()), "n": len(emb),
           "by_subcategory": keep, "separation_before": before,
           "separation_after": after, "verdict": verdict}
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
