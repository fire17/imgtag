#!/usr/bin/env python3
"""COCO ground truth for track-people: exhaustive person counts + a visible-face proxy.

WHY COCO IS USABLE AS TRUTH HERE (and where it is not):
  * `instances_val2017.json` person annotations are EXHAUSTIVE per image — every person
    is boxed. That makes `n_persons` real ground truth for categories (a)/(b), which is
    exactly what a caption corpus could never give us.
  * `person_keypoints_val2017.json` carries 17 keypoints per person; the first five are
    nose / left_eye / right_eye / left_ear / right_ear with a visibility flag
    v ∈ {0 = unlabeled, 1 = labeled-but-occluded, 2 = labeled-and-visible}.
    A visible FACE is therefore only ever a PROXY here — COCO never annotated "faces".
    §Proxy limits below is written into the report verbatim; no number derived from it
    may be presented as face ground truth.

CROWD ANNOTATIONS ARE THE TRAP. 227 val2017 images carry an `iscrowd=1` person region:
ONE annotation covering an unknown number of people. Counting it as 1 person understates
badly; dropping it silently would quietly delete the hardest crowd images from the eval.
So it is neither: crowd images are kept, labelled `crowd_unbounded`, and every count
table reports them as their own column instead of folding them into an accuracy number.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ANN = ROOT / "data" / "coco" / "annotations"

#: COCO keypoint indices for the face. Order is fixed by the dataset spec.
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
VISIBLE = 2  # v flag meaning "labeled AND visible"


def _kp_visible(kp: list, idx: int) -> bool:
    return len(kp) > idx * 3 + 2 and kp[idx * 3 + 2] == VISIBLE


def face_is_visible(kp: list) -> bool:
    """Proxy: >=2 of {nose, left_eye, right_eye} marked visible.

    Chosen over 'nose visible' alone because a face at an ANGLE (the user's explicit
    requirement) shows nose + one eye while the far eye is occluded — a 2-of-3 rule
    accepts profile views, which a 3-of-3 rule would reject and a 1-of-3 rule would
    over-accept (a bare nose keypoint also survives heavy occlusion of the rest).
    Sensitivity to this choice is measured and reported, never assumed.
    """
    return sum(_kp_visible(kp, i) for i in (NOSE, L_EYE, R_EYE)) >= 2


def face_visible_strict(kp: list) -> bool:
    """3-of-3 variant — reported alongside the primary proxy as a sensitivity bound."""
    return all(_kp_visible(kp, i) for i in (NOSE, L_EYE, R_EYE))


def face_visible_loose(kp: list) -> bool:
    """1-of-5 variant (any face keypoint incl. ears) — the upper bound of the proxy."""
    return any(_kp_visible(kp, i) for i in (NOSE, L_EYE, R_EYE, L_EAR, R_EAR))


def build(split: str = "val2017") -> dict[int, dict]:
    """-> {coco_image_id: {file_name, n_persons, crowd, n_faces, n_faces_strict,
                           n_faces_loose, n_kp_annotated, n_persons_no_kp}}"""
    inst = json.loads((ANN / f"instances_{split}.json").read_text())
    person_cat = next(c["id"] for c in inst["categories"] if c["name"] == "person")

    gt: dict[int, dict] = {
        im["id"]: {"file_name": im["file_name"], "n_persons": 0, "crowd": False,
                   "n_faces": 0, "n_faces_strict": 0, "n_faces_loose": 0,
                   "n_kp_annotated": 0, "n_persons_no_kp": 0}
        for im in inst["images"]
    }
    for a in inst["annotations"]:
        if a["category_id"] != person_cat:
            continue
        if a.get("iscrowd"):
            gt[a["image_id"]]["crowd"] = True      # unbounded: NOT added to n_persons
        else:
            gt[a["image_id"]]["n_persons"] += 1

    kp_ann = json.loads((ANN / f"person_keypoints_{split}.json").read_text())
    for a in kp_ann["annotations"]:
        if a.get("iscrowd"):
            continue
        g = gt[a["image_id"]]
        if a.get("num_keypoints", 0) == 0:
            # A person too small/occluded for COCO to annotate keypoints AT ALL. This is
            # absence of evidence, not evidence of absence — tracked separately so the
            # face proxy's blind spot is visible in every table instead of silently
            # deflating recall.
            g["n_persons_no_kp"] += 1
            continue
        kp = a["keypoints"]
        g["n_kp_annotated"] += 1
        g["n_faces"] += face_is_visible(kp)
        g["n_faces_strict"] += face_visible_strict(kp)
        g["n_faces_loose"] += face_visible_loose(kp)
    return gt


def categories(rec: dict) -> dict[str, bool]:
    """The four USER-FACING categories (VISION-ADDENDA 13:28Z), DERIVED from raw counts.

    This function is the reference implementation of the derivation that
    imgtag.moderation.people performs at read time (TRACKS.md T1).
    """
    n_p, n_f = rec["n_persons"], rec["n_faces"]
    return {"one-person": n_p == 1, "multi-person": n_p >= 2,
            "one-face": n_f == 1, "multi-face": n_f >= 2}


if __name__ == "__main__":
    import collections

    gt = build()
    n = len(gt)
    clean = {k: v for k, v in gt.items() if not v["crowd"]}
    print(f"images {n}  ·  crowd-tainted {n - len(clean)}  ·  clean {len(clean)}")

    hp = collections.Counter(min(v["n_persons"], 6) for v in clean.values())
    hf = collections.Counter(min(v["n_faces"], 6) for v in clean.values())
    print("\n n | n_persons | n_faces(proxy)")
    for k in range(7):
        lab = f"{k}" if k < 6 else "6+"
        print(f"{lab:>3}| {hp[k]:>9} | {hf[k]:>9}")

    cats = collections.Counter()
    for v in clean.values():
        for c, on in categories(v).items():
            cats[c] += on
    print("\nderived categories (clean images):")
    for c, k in sorted(cats.items()):
        print(f"  {c:<14} {k:>5}  ({k / len(clean):.1%})")

    no_kp = sum(v["n_persons_no_kp"] for v in clean.values())
    kp_ok = sum(v["n_kp_annotated"] for v in clean.values())
    print(f"\nproxy coverage: {kp_ok} persons keypoint-annotated · "
          f"{no_kp} persons with NO keypoints (proxy blind) · "
          f"{no_kp / max(kp_ok + no_kp, 1):.1%} blind")
