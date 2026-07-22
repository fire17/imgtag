#!/usr/bin/env python3
"""Build the weapons TRUE-POSITIVE probe (`weaponprobe`) — real weapon images spanning an
authored SUBCATEGORY TAXONOMY, drawn from the HELD-OUT OI `validation` split only.

Why held-out only: the shipped head (`weapons-pecore-s16-384.json`) was trained on OI
`test`. Measuring true-positive confidence on `test` images would be optimistic (the head
saw them). Every probe image here is from `validation`, which the head never saw — so the
TP confidence numbers in eval_weapons.py are honest.

Output (all versioned DATA, no hot-path code — TRACKS.md T3):
  data/weapon-probe/taxonomy.json   — subcategory taxonomy + per-image subcat membership
  data/weapon-probe/images/<id>.jpg — the probe images, copied so they index like any folder

Then index it into the gallery exactly like drugprobe:
  uv run imgtag index data/weapon-probe/images --dataset weaponprobe --wait --moderation

Re-runnable with zero agents (T4). Deterministic: sampling is seeded and sorted.

    .venv/bin/python scripts/build_weapons_probe.py [--cap 25] [--seed 17]
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SLICE = ROOT / "data" / "oi-weapons"
OUT = ROOT / "data" / "weapon-probe"

# ── The authored subcategory taxonomy (the user's explicit list, 2026-07-22 13:58Z) ──
# Each subcategory maps to the OI 600-class `Weapon`-subtree labels that realise it. Open
# Images does NOT label sub-types below its flat classes, so several user subcategories are
# subsumed (assault/hunting/submachine → Rifle; crossbow → Bow and arrow; machete → Sword)
# and TWO cannot be probed from OI at all — recorded as GAPS, never silently dropped.
TAXONOMY: dict[str, dict] = {
    "handguns_pistols_revolvers": {
        "oi_classes": ["Handgun"],
        "note": "pistols, revolvers, semi-autos — one flat OI class",
    },
    "rifles": {
        "oi_classes": ["Rifle"],
        "note": "hunting + assault-style + submachine/automatic all fall here; "
                "OI has no sub-type label to separate them (GAP: submachine/automatic)",
    },
    "shotguns": {
        "oi_classes": ["Shotgun"],
        "note": "pump/break-action shotguns",
    },
    "knives_threat": {
        "oi_classes": ["Knife", "Dagger"],
        "note": "blades in the OI Weapon subtree — Kitchen-knife is deliberately OUTSIDE "
                "it (kitchen-context is the FP contrast, measured against the FP band)",
    },
    "swords_machetes_axes": {
        "oi_classes": ["Sword", "Axe"],
        "note": "edged long weapons; machete subsumed into Sword (no OI machete label)",
    },
    "bows_crossbows": {
        "oi_classes": ["Bow and arrow"],
        "note": "crossbow subsumed into Bow and arrow (no OI crossbow label)",
    },
    "explosives_grenades": {
        "oi_classes": ["Bomb"],
        "note": "grenades/explosive devices — OI class `Bomb`",
    },
    "heavy_ordnance": {
        "oi_classes": ["Tank", "Cannon", "Missile"],
        "note": "military heavy weapons",
    },
    "generic_display": {
        "oi_classes": ["Weapon"],
        "note": "generic/multi-weapon scenes (gun racks, weapon displays) tagged only "
                "with the parent `Weapon` label",
    },
}


def records(split: str) -> list[dict]:
    """On-disk y==1 records for a split, with their OI class list."""
    recs = [json.loads(l) for l in (SLICE / f"{split}.jsonl").read_text().splitlines()]
    return [r for r in recs
            if r["y"] == 1 and (SLICE / split / f"{r['id']}.jpg").is_file()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=25,
                    help="max images per subcategory (take all if fewer available)")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--split", default="validation",
                    help="OI split to draw from — MUST be the held-out one (validation)")
    a = ap.parse_args()

    rng = random.Random(a.seed)
    pos = records(a.split)

    # class -> list of image ids that carry it (an image may carry several)
    by_class: dict[str, list[str]] = defaultdict(list)
    for r in pos:
        for c in r["classes"]:
            by_class[c].append(r["id"])

    chosen: dict[str, list[str]] = {}       # subcat -> sampled image ids
    gaps: list[dict] = []
    for sub, spec in TAXONOMY.items():
        ids = sorted({i for c in spec["oi_classes"] for i in by_class.get(c, [])})
        if not ids:
            gaps.append({"subcategory": sub, "oi_classes": spec["oi_classes"],
                         "reason": "0 images with these labels in the held-out split"})
            chosen[sub] = []
            continue
        rng.shuffle(ids)
        chosen[sub] = sorted(ids[: a.cap])

    # Union of every sampled id → the probe image set (dedup; a multi-label image is one file
    # but is counted in every subcategory it belongs to during evaluation).
    probe_ids = sorted({i for ids in chosen.values() for i in ids})

    # image -> the subcategories it realises (for per-subcat eval grouping)
    img_subcats: dict[str, list[str]] = {}
    for iid in probe_ids:
        subs = [sub for sub, ids in chosen.items() if iid in ids]
        img_subcats[f"{iid}.jpg"] = subs

    # ── copy images into the probe dir ──
    imgdir = OUT / "images"
    imgdir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for iid in probe_ids:
        src = SLICE / a.split / f"{iid}.jpg"
        dst = imgdir / f"{iid}.jpg"
        if not dst.is_file():
            shutil.copy2(src, dst)
            copied += 1

    taxonomy_out = {
        "version": 1,
        "created": "2026-07-22",
        "source": f"Open Images v5 `{a.split}` split (HELD-OUT — head trained on `test`); "
                  "image-level `Weapon`-subtree labels. CC-BY licensed, data/ gitignored.",
        "purpose": "TRUE-POSITIVE probe for the weapons track (user directive 13:58Z): "
                   "real weapon images per subcategory, so TP confidence can be measured "
                   "against the current false-positive band in user datasets and a per-tier "
                   "ratio threshold set. Pattern = data/drug-probe.",
        "cap_per_subcategory": a.cap,
        "seed": a.seed,
        "subcategories": {
            sub: {"oi_classes": spec["oi_classes"], "note": spec["note"],
                  "n": len(chosen[sub]), "images": [f"{i}.jpg" for i in chosen[sub]]}
            for sub, spec in TAXONOMY.items()
        },
        "gaps": gaps,
        "n_images": len(probe_ids),
        "image_subcategories": img_subcats,
    }
    (OUT / "taxonomy.json").write_text(json.dumps(taxonomy_out, indent=1))

    print(f"probe: {len(probe_ids)} images ({copied} newly copied) -> {imgdir}")
    for sub in TAXONOMY:
        print(f"  {sub:28s} {len(chosen[sub]):3d}")
    if gaps:
        print("GAPS (no held-out TP imagery — reported, not hidden):")
        for g in gaps:
            print(f"  {g['subcategory']:28s} {g['oi_classes']}  {g['reason']}")
    print(f"\nnext: uv run imgtag index {imgdir} --dataset weaponprobe --wait --moderation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
