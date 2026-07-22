"""CORPUS-A ground truth (COCO val2017 + LVIS-on-val2017). No number exists without one.

Quality metrics are computed ONLY against downloaded ground truth, never eyeballed
(ORACLE §6). Everything here is deterministic and sorted so two runs of the bench compare
the same rows.
"""
from __future__ import annotations

import functools
import json
import os

from . import candidates as C

COCO = os.path.join(C.DATA, "coco")
ANN = os.path.join(COCO, "annotations")
LVIS = os.path.join(C.DATA, "lvis", "lvis_val2017_only.json")

# B5 supercategory suite (BUDGETS: vehicle, animal, food, furniture, appliance, sports).
SUPERCATS = ("vehicle", "animal", "food", "furniture", "appliance", "sports")

# 5 absurdities (B7) — deliberately not derivable from any label file.
ABSURD = ("a photorealistic dragon breathing fire",
          "the interior of a nuclear fusion reactor",
          "a medieval knight riding a motorcycle on mars",
          "an MRI scan of a human brain",
          "a screenshot of a spreadsheet")


@functools.lru_cache(maxsize=1)
def corpus_a() -> dict:
    """5,000 COCO val2017 images + exhaustive 80-class truth + captions."""
    inst = json.load(open(os.path.join(ANN, "instances_val2017.json")))
    imgs = sorted(inst["images"], key=lambda i: i["id"])
    paths = [os.path.join(COCO, "val2017", i["file_name"]) for i in imgs]
    idx = {i["id"]: n for n, i in enumerate(imgs)}

    cats = {c["id"]: c for c in inst["categories"]}
    pos: dict[str, set[int]] = {c["name"]: set() for c in cats.values()}
    for a in inst["annotations"]:
        if a["image_id"] in idx:
            pos[cats[a["category_id"]]["name"]].add(idx[a["image_id"]])

    supers: dict[str, list[str]] = {}
    for c in cats.values():
        supers.setdefault(c["supercategory"], []).append(c["name"])
    for v in supers.values():
        v.sort()

    caps = json.load(open(os.path.join(ANN, "captions_val2017.json")))
    captions = [(idx[a["image_id"]], a["caption"].strip())
                for a in sorted(caps["annotations"], key=lambda a: a["id"])
                if a["image_id"] in idx]

    return {
        "tag": "CORPUS-A/coco5k",
        "paths": paths,
        "image_ids": [i["id"] for i in imgs],
        "n": len(paths),
        "pos": {k: sorted(v) for k, v in sorted(pos.items())},
        "supers": {k: supers[k] for k in SUPERCATS if k in supers},
        "captions": captions,
    }


def _cocoid_from_path(path: str) -> int | None:
    """COCO image id from a val2017 filename (000000000139.jpg → 139)."""
    stem = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    try:
        return int(stem)
    except ValueError:
        return None


def align_to_ids(ids: list[dict]) -> dict:
    """Ground truth aligned to a SNAPSHOT's own row order (not corpus_a's order).

    A pre-indexed dataset stores rows in indexer order, so corpus_a()'s positional `pos`
    indices don't apply. This rebuilds pos/supers/captions against the row `i` of each id
    record — the only correct way to score an already-indexed dataset.
    """
    inst = json.load(open(os.path.join(ANN, "instances_val2017.json")))
    cats = {c["id"]: c for c in inst["categories"]}
    per_img: dict[int, set[str]] = {}
    for a in inst["annotations"]:
        per_img.setdefault(a["image_id"], set()).add(cats[a["category_id"]]["name"])

    row_of_cocoid: dict[int, int] = {}
    for i, rec in enumerate(ids):
        cid = _cocoid_from_path(rec.get("path", ""))
        if cid is not None:
            row_of_cocoid[cid] = i

    pos = {c["name"]: [] for c in cats.values()}
    for cid, row in row_of_cocoid.items():
        for name in per_img.get(cid, ()):  # image annotated but maybe not in this dataset
            pos[name].append(row)
    pos = {k: sorted(v) for k, v in pos.items()}

    supers: dict[str, list[str]] = {}
    for c in cats.values():
        supers.setdefault(c["supercategory"], []).append(c["name"])
    for v in supers.values():
        v.sort()

    caps = json.load(open(os.path.join(ANN, "captions_val2017.json")))
    captions = [(row_of_cocoid[a["image_id"]], a["caption"].strip())
                for a in sorted(caps["annotations"], key=lambda a: a["id"])
                if a["image_id"] in row_of_cocoid]

    return {"pos": pos, "supers": {k: supers[k] for k in SUPERCATS if k in supers},
            "captions": captions, "n": len(ids),
            "coverage": len(row_of_cocoid)}


@functools.lru_cache(maxsize=1)
def absent_concepts(n: int = 25) -> list[str]:
    """B7 absent list, AUTO-DERIVED: LVIS categories with zero annotations on val2017.

    LVIS v1 val is annotated over 4,809 of the 5,000 val2017 images, so a category with
    zero annotations here is absent from the corpus by the dataset's own labelling.
    Deterministic: sorted by name, evenly spread across the alphabet.
    """
    lv = json.load(open(LVIS))
    seen = {a["category_id"] for a in lv["annotations"]}
    zero = sorted(c["name"].replace("_", " ") for c in lv["categories"]
                  if c["id"] not in seen)
    if not zero:
        return []
    step = max(1, len(zero) // n)
    return zero[::step][:n]
