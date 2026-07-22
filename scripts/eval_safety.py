#!/usr/bin/env python
"""Measure the SAFETY track (people lying down + danger escalation) on real labels.

VISION-ADDENDA 13:20Z: "identify people lying down (even if part of their body is
obstructed) and even higher flagging if either detecting injury, things broken,
distruction distress high stress or anything dangorous".

GROUND TRUTH — built here, from two INDEPENDENT human-annotated sources on disk, because
neither alone is trustworthy for a POSE question:

  1. `person_keypoints_val2017.json` — GEOMETRY. Torso vector (shoulder-mid -> hip-mid)
     within 30 deg of horizontal, PLUS the leg vector (hip-mid -> ankle/knee-mid) within
     45 deg of horizontal. The leg constraint is not optional: torso-only geometry labels
     every crouched jockey, surfer and skateboarder as "lying" (verified by eye —
     000000080659 is three jockeys, and torso-only geometry called it lying).
  2. `captions_val2017.json` — HUMAN CONSENSUS. Five independent captions per image; a
     caption counts when it contains a lying/sleeping verb AND a person word, with an
     animal guard (COCO is full of cats and cows lying down).

PRIMARY LABEL = human consensus (>=2 of 5 captions). Geometry is NOT the primary label:
measured here, geometry and captions agree on only 15 images while geometry alone claims
30 — a 50% disagreement rate, and eye-checking the disagreements showed geometry wrong.
Geometry earns its keep as the OCCLUSION STRATIFIER: whether a positive has resolvable
keypoints is a direct, annotator-supplied measure of how visible that body is, which is
exactly the "even if part of their body is obstructed" requirement made measurable.

    stratum A  kp-resolved-lying   both signals agree, body clearly visible
    stratum B  kp-present-disagree keypoints exist but the pose geometry is ambiguous
    stratum C  NO usable keypoints the annotator could not place a torso = heavily
                                   occluded / covered / cropped (the hard case)

DANGER is measured on caption-derived labels too, and the honest headline is that COCO
val2017 contains ~17 danger images and ZERO images of a person lying down IN a danger
context. Alert-tier precision is therefore NOT measurable on COCO — it is measured, at
much lower confidence, on the hand-checked Unsplash probe (`data/safety-probe`) or not at
all. Every table below says which.

    uv run python scripts/eval_safety.py [--dataset cocoval2017] [--model pecore-s16-384]
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imgtag.core import models, store  # noqa: E402
from imgtag.moderation import safety  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ANN = ROOT / "data/coco/annotations"
PROBE = ROOT / "data/safety-probe"

LIE_VERB = re.compile(
    r"\b(lying|laying|lies|sleeping|asleep|sleeps|napping|sprawled|reclining|"
    r"passed out|lounging|sunbathing|lays)\b"
)
PERSON = re.compile(
    r"\b(man|woman|person|people|boy|girl|guy|child|children|baby|lady|someone|"
    r"men|women|kid|kids|toddler|couple|human|male|female)\b"
)
ANIMAL = re.compile(
    r"\b(cat|cats|dog|dogs|kitten|puppy|bear|zebra|giraffe|sheep|cow|cows|horse|"
    r"bird|elephant|animal|teddy|doll)\b"
)
# Danger words a caption writer actually uses. "broken" and "hospital" are DELIBERATELY
# excluded: measured, they are dominated by "a broken umbrella" / "a hospital bed in a
# clean room" and add 25 false danger labels to a 17-image slice.
DANGER_WORD = re.compile(
    r"\b(crash|crashed|wreck|wreckage|accident|injur\w*|bleeding|blood|ambulance|"
    r"paramedic|stretcher|firefighter|fire truck|burning|on fire|flames|shattered|"
    r"debris|destroyed|destruction|collapsed|rubble|disaster|flood|unconscious)\b"
)


def _angle(v) -> float:
    """Degrees from HORIZONTAL: 0 = flat, 90 = upright. Sign-free (up/down irrelevant)."""
    return abs(math.degrees(math.atan2(abs(v[1]), abs(v[0]))))


def _pose(ann) -> tuple[float, float | None, int] | None:
    """(torso angle, leg angle or None, visible-keypoint count) for one person."""
    k = np.array(ann["keypoints"]).reshape(17, 3)

    def mid(ix):
        vs = [i for i in ix if k[i, 2] > 0]
        return k[vs, :2].mean(0) if vs else None

    sh, hip, knee, ankle = mid([5, 6]), mid([11, 12]), mid([13, 14]), mid([15, 16])
    if sh is None or hip is None:
        return None
    torso = sh - hip
    if math.hypot(*torso) < 8:          # sub-8px torso: annotation noise, not a pose
        return None
    low = ankle if ankle is not None else knee
    leg = _angle(low - hip) if low is not None and math.hypot(*(low - hip)) > 8 else None
    return _angle(torso), leg, int((k[:, 2] > 0).sum())


def ground_truth() -> dict:
    """Build every labelled slice from COCO val2017. Pure disk, no network, no guessing."""
    kp = json.loads((ANN / "person_keypoints_val2017.json").read_bytes())
    inst = json.loads((ANN / "instances_val2017.json").read_bytes())
    caps = json.loads((ANN / "captions_val2017.json").read_bytes())
    imgs = {im["id"]: im for im in kp["images"]}

    by = collections.defaultdict(list)
    for a in caps["annotations"]:
        by[a["image_id"]].append(a["caption"].lower())

    npersons = collections.Counter(
        a["image_id"] for a in inst["annotations"] if a["category_id"] == 1
    )

    def human_lying_votes(i: int) -> int:
        n = 0
        for c in by[i]:
            lm, pm = LIE_VERB.search(c), PERSON.search(c)
            if not lm or not pm:
                continue
            am = ANIMAL.search(c)
            # animal guard: only count when the PERSON is plausibly the lying subject
            if am and not (pm.start() < lm.start() and (am.start() > lm.start()
                                                        or pm.start() < am.start())):
                continue
            n += 1
        return n

    votes = {i: human_lying_votes(i) for i in by}
    cap2 = {i for i, n in votes.items() if n >= 2}      # human consensus = the label
    cap1 = {i for i, n in votes.items() if n >= 1}      # any mention = excluded from NEG

    geo, upright, has_kp = {}, set(), set()
    for a in kp["annotations"]:
        if a.get("num_keypoints", 0) == 0:
            continue
        im = imgs[a["image_id"]]
        areafrac = a["bbox"][2] * a["bbox"][3] / (im["width"] * im["height"])
        p = _pose(a)
        if p is None:
            continue
        torso, leg, nvis = p
        has_kp.add(a["image_id"])
        flat_legs = leg is not None and leg <= 45
        if torso <= 30 and areafrac >= 0.02 and (flat_legs or (leg is None and torso <= 20)):
            geo[a["image_id"]] = max(geo.get(a["image_id"], 0), nvis)
        if torso >= 60:
            upright.add(a["image_id"])

    danger = {i for i, cs in by.items() if any(DANGER_WORD.search(c) for c in cs)}
    g = set(geo)
    return {
        "captions": by,
        "pos": cap2,
        "geo": g,
        # negatives exclude ANY lying mention and any geometric lying: a negative must be
        # negative under BOTH signals, or the FP rate is measuring label noise.
        "neg_upright": upright - cap1 - g,
        "neg_empty": set(imgs) - set(npersons) - cap1 - g,
        "danger": danger,
        # occlusion strata of the positives (the "part of their body is obstructed" axis)
        "occ_visible": cap2 & g,
        "occ_ambiguous": (cap2 & has_kp) - g,
        "occ_hidden": cap2 - has_kp,
        "npersons": npersons,
    }


def ap(s: np.ndarray, y: np.ndarray) -> float:
    o = np.argsort(-s)
    ys = np.asarray(y, bool)[o]
    tp = np.cumsum(ys)
    return float(((tp / np.arange(1, len(ys) + 1)) * ys).sum() / max(ys.sum(), 1))


def rec_at_fpr(s: np.ndarray, y: np.ndarray, neg: np.ndarray, f: float) -> tuple[float, float]:
    t = float(np.quantile(s[neg], 1 - f))
    return float((s[y] >= t).mean()), t


def prec_at_tau(s: np.ndarray, y: np.ndarray, neg: np.ndarray, t: float) -> float:
    hit = s >= t
    tp = int((hit & y).sum())
    return tp / max(int((hit & (y | neg)).sum()), 1)


def main() -> int:
    a = argparse.ArgumentParser()
    a.add_argument("--dataset", default="cocoval2017")
    a.add_argument("--model", default="pecore-s16-384")
    a.add_argument("--json", default="")
    args = a.parse_args()

    gt = ground_truth()
    backend = models.load_backend(args.model, {}, vision=False)
    snap = store.open_snapshot(args.dataset)
    emb = np.asarray(snap.emb, np.float32)
    row = {int(os.path.basename(r["path"])[:-4]): r["row"] for r in snap.ids}

    def mask(ids) -> np.ndarray:
        m = np.zeros(len(emb), bool)
        for i in ids:
            if i in row:
                m[row[i]] = True
        return m

    y = mask(gt["pos"])
    neg = mask(gt["neg_upright"] | gt["neg_empty"])
    yd = mask(gt["danger"])
    out: dict = {"model": backend.model_id, "dataset": args.dataset, "n": len(emb),
                 "counts": {"lying_pos": int(y.sum()), "neg": int(neg.sum()),
                            "danger": int(yd.sum()),
                            "lying_AND_danger": int((y & yd).sum())}}
    print(f"{args.dataset}: {len(emb)} rows · lying {y.sum()} · neg {neg.sum()} · "
          f"danger {yd.sum()} · lying∩danger {(y & yd).sum()}")

    sc = safety.SafetyScorer.build(backend)
    s_lie = sc.lying_margin(emb)
    s_dan = sc.danger_margin(emb)

    # ── lying: the headline table ────────────────────────────────────────────────────
    sel = y | neg
    out["lying"] = {"AP": round(ap(s_lie[sel], y[sel]), 4), "at_fpr": {}}
    print(f"\nLYING  AP={out['lying']['AP']:.3f}   (chance = {y.sum()/sel.sum():.4f})")
    for f in (0.01, 0.02, 0.05, 0.10, 0.20):
        r, t = rec_at_fpr(s_lie, y, neg, f)
        p = prec_at_tau(s_lie, y, neg, t)
        out["lying"]["at_fpr"][str(f)] = {"recall": round(r, 4), "tau": round(t, 5),
                                          "precision": round(p, 4)}
        print(f"  FPR {f:4.0%}  tau {t:+.4f}  recall {r:.3f}  precision {p:.3f}")

    # ── occlusion robustness: recall per stratum at the shipped tau ──────────────────
    tau = safety.TAU_REVIEW
    out["occlusion"] = {}
    print(f"\nOCCLUSION ROBUSTNESS at shipped tau_review={tau} (p>={safety.TAU_REVIEW})")
    for name, ids in (("A visible (kp-resolved)", gt["occ_visible"]),
                      ("B ambiguous (kp, geom disagrees)", gt["occ_ambiguous"]),
                      ("C hidden (no usable keypoints)", gt["occ_hidden"])):
        m = mask(ids)
        if not m.any():
            continue
        p = safety.lying_prob(s_lie[m])
        r = float((p >= tau).mean())
        out["occlusion"][name.split()[0]] = {"n": int(m.sum()), "recall": round(r, 4)}
        print(f"  {name:34s} n={int(m.sum()):3d}  recall {r:.3f}")

    # ── danger: tiny slice, wide CI, said so everywhere ─────────────────────────────
    dneg = ~yd
    out["danger"] = {"n_pos": int(yd.sum()), "AP": round(ap(s_dan, yd), 4), "at_fpr": {}}
    print(f"\nDANGER  AP={out['danger']['AP']:.3f}  on n={yd.sum()} positives "
          f"(⚠️ tiny slice — wide CI, treat as directional)")
    for f in (0.01, 0.05, 0.10):
        r, t = rec_at_fpr(s_dan, yd, dneg, f)
        out["danger"]["at_fpr"][str(f)] = {"recall": round(r, 4), "tau": round(t, 5)}
        print(f"  FPR {f:4.0%}  tau {t:+.4f}  recall {r:.3f}")

    # ── the alert tier: honest non-measurement ───────────────────────────────────────
    n_alert = int((y & yd).sum())
    out["alert"] = {"measurable_on_coco": n_alert > 0, "n": n_alert,
                    "note": "COCO val2017 contains ZERO images of a person lying down in "
                            "a danger context: every one of its lying people is benign "
                            "(sleeping, beach, sofa). Alert-tier precision/recall CANNOT "
                            "be measured on this corpus."}
    print(f"\nALERT tier: lying∩danger = {n_alert} images on COCO → "
          f"{'measurable' if n_alert else 'NOT MEASURABLE (see report §alert)'}")
    print(f"BENIGN-vs-DANGER split of the {int(y.sum())} lying positives: "
          f"{int(y.sum()) - n_alert} benign / {n_alert} danger")

    # ── Unsplash probe (hand-checked, weaker labels — reported separately, never merged)
    probe = _probe(backend, sc)
    if probe:
        out["probe"] = probe

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))
        print(f"\nwrote {args.json}")
    return 0


def _probe(backend, sc) -> dict | None:
    """Score the hand-checked Unsplash probe: the ONLY danger-lying images we have."""
    labels = PROBE / "labels.json"
    if not labels.is_file():
        return None
    lab = json.loads(labels.read_bytes())
    paths, kinds = [], []
    for kind, names in lab.items():
        if kind.startswith("_"):
            continue
        for n in names:
            for sub in ("lie", "dan"):
                p = PROBE / sub / n
                if p.is_file():
                    paths.append(p)
                    kinds.append(kind)
                    break
    if not paths:
        return None
    cache = PROBE / f".emb-{backend.model_id}.npz"
    key = [p.name for p in paths]
    if cache.is_file():
        z = np.load(cache, allow_pickle=True)
        emb = z["emb"] if list(z["key"]) == key else None
    else:
        emb = None
    if emb is None:
        from PIL import Image
        buf = []
        for p in paths:
            with Image.open(p) as im:
                buf.append(backend.preprocess(im.convert("RGB")))
        emb = np.asarray(backend.embed_images(np.stack(buf)), np.float32)
        np.savez(cache, emb=emb, key=np.array(key))
    kinds = np.array(kinds)
    sl, sd = sc.lying_margin(emb), sc.danger_margin(emb)
    pl, pd = safety.lying_prob(sl), safety.danger_prob(sd)
    tier = sc.tiers(pl, pd)
    res = {"n": len(paths), "counts": {k: int((kinds == k).sum()) for k in set(kinds)},
           "by_kind": {}}
    print(f"\nUNSPLASH PROBE (hand-checked labels, n={len(paths)}) — "
          f"the only danger-lying images available")
    for k in sorted(set(kinds)):
        m = kinds == k
        res["by_kind"][k] = {"n": int(m.sum()),
                             "alert": int((tier[m] == "alert").sum()),
                             "review": int((tier[m] == "review").sum()),
                             "none": int((tier[m] == "none").sum()),
                             "mean_p_lying": round(float(pl[m].mean()), 4),
                             "mean_p_danger": round(float(pd[m].mean()), 4)}
        r = res["by_kind"][k]
        print(f"  {k:16s} n={r['n']:3d}  alert={r['alert']:3d} review={r['review']:3d} "
              f"none={r['none']:3d}  p_lie={r['mean_p_lying']:.3f} p_dan={r['mean_p_danger']:.3f}")
    return res


if __name__ == "__main__":
    raise SystemExit(main())
