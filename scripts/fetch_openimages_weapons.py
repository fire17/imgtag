#!/usr/bin/env python3
"""Build the OI weapons train/eval slice (track-weapons, ADR-3 §calibration).

Open Images image-level labels are the ONLY ground truth here (`Confidence` 1 = verified
present, 0 = verified ABSENT — the second is what makes a clean negative possible; a
missing label means "not verified", never "absent").

Split law: OI `test` -> TRAIN, OI `validation` -> HELD-OUT EVAL. Disjoint by construction.

Images stream from the public S3 mirror (no auth, verified 2026-07-22):
  https://open-images-dataset.s3.amazonaws.com/<split>/<ImageID>.jpg

Usage: .venv/bin/python scripts/fetch_openimages_weapons.py [--neg-ratio 3] [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OI = ROOT / "data" / "openimages"
OUT = ROOT / "data" / "oi-weapons"
S3 = "https://open-images-dataset.s3.amazonaws.com"

# OI 600-class `Weapon` subtree (bbox_labels_600_hierarchy.json), verbatim members.
WEAPON = {
    "/m/083kb": "Weapon", "/m/0gxl3": "Handgun", "/m/06c54": "Rifle",
    "/m/06nrc": "Shotgun", "/m/04ctx": "Knife", "/m/02gzp": "Dagger",
    "/m/06y5r": "Sword", "/m/01g3x7": "Bow and arrow", "/m/0c2jj": "Axe",
    "/m/020kz": "Cannon", "/m/07cmd": "Tank", "/m/04ylt": "Missile",
    "/m/0ct4f": "Bomb",
}
# Confusable non-weapons: blade-shaped, held-in-hand, or weapon-adjacent context.
HARD_NEG = {
    "/m/058qzx": "Kitchen knife", "/m/01lsmm": "Scissors", "/m/03l9g": "Hammer",
    "/m/01j4z9": "Chainsaw", "/m/07k1x": "Tool", "/m/01d380": "Drill",
    "/m/01bms0": "Screwdriver", "/m/01j5ks": "Wrench", "/m/0_dqb": "Chisel",
    "/m/02bm9n": "Ratchet", "/m/05bm6": "Nail", "/m/03g8mr": "Baseball bat",
    "/m/0dv9c": "Racket", "/m/071p9": "Ski", "/m/06_fw": "Skateboard",
    "/m/0138tl": "Toy", "/m/07dd4": "Torch", "/m/01kb5b": "Flashlight",
    "/m/0dv5r": "Camera", "/m/0dt3t": "Fork", "/m/0cmx8": "Spoon",
    "/m/02pdsw": "Cutting board", "/m/09rvcxw": "Rocket", "/m/0k5j": "Aircraft",
    "/m/09ct_": "Helicopter", "/m/0h8jyh6": "Grinder", "/m/0hnnb": "Umbrella",
    "/m/0342h": "Guitar",
}


def labels(split: str) -> tuple[dict, dict, list]:
    """(pos, neg, all_ids) — image_id -> label names, over the classes we care about."""
    pos, neg, seen = defaultdict(set), defaultdict(set), {}
    f = OI / f"{split}-annotations-human-imagelabels-boxable.csv"
    with f.open() as fh:
        for row in csv.DictReader(fh):
            ln, iid = row["LabelName"], row["ImageID"]
            seen[iid] = None
            if ln in WEAPON or ln in HARD_NEG:
                (pos if row["Confidence"] == "1" else neg)[iid].add(ln)
    return pos, neg, list(seen)


def build_split(split: str, neg_ratio: int, seed: int) -> list[dict]:
    pos, neg, all_ids = labels(split)
    rng = random.Random(seed)
    weapon_ids = {i for i, ls in pos.items() if ls & WEAPON.keys()}

    recs = [{"id": i, "y": 1, "classes": sorted(WEAPON[c] for c in pos[i] & WEAPON.keys())}
            for i in sorted(weapon_ids)]

    # Negatives in three tiers, most-informative first:
    #   hard      — a confusable object is VERIFIED PRESENT (kitchen knife, tool, toy…)
    #   verified  — a weapon class is VERIFIED ABSENT (Confidence 0)
    #   random    — no weapon label at all (unverified-absent; small label noise, see report)
    hard = sorted({i for i, ls in pos.items() if ls & HARD_NEG.keys()} - weapon_ids)
    verified = sorted({i for i, ls in neg.items() if ls & WEAPON.keys()} - weapon_ids - set(hard))
    taken = weapon_ids | set(hard) | set(verified)
    rand = sorted(set(all_ids) - taken)
    for pool in (hard, verified, rand):
        rng.shuffle(pool)

    want = len(recs) * neg_ratio
    n_hard = min(len(hard), want // 2)
    tiers = [("hard", hard[:n_hard]), ("verified", verified[: want - n_hard])]
    tiers.append(("random", rand[: want - n_hard - len(tiers[1][1])]))
    for tier, ids in tiers:
        for i in ids:
            recs.append({"id": i, "y": 0, "tier": tier,
                         "classes": sorted(HARD_NEG[c] for c in pos.get(i, set()) & HARD_NEG.keys())})
    return recs


def download(split: str, recs: list[dict], workers: int = 32) -> tuple[int, int]:
    d = OUT / split
    d.mkdir(parents=True, exist_ok=True)
    todo = [r["id"] for r in recs if not (d / f"{r['id']}.jpg").is_file()]
    ok = fail = 0

    def get(iid: str) -> bool:
        p = d / f"{iid}.jpg"
        r = subprocess.run(
            ["curl", "-fsSL", "--retry", "2", "--max-time", "60", "-o", str(p),
             f"{S3}/{split}/{iid}.jpg"], capture_output=True)
        if r.returncode != 0 or not p.is_file() or p.stat().st_size < 1024:
            p.unlink(missing_ok=True)
            return False
        return True

    with ThreadPoolExecutor(workers) as ex:
        for n, got in enumerate(ex.map(get, todo), 1):
            ok, fail = ok + got, fail + (not got)
            if n % 200 == 0:
                print(f"  {split}: {n}/{len(todo)} ok={ok} fail={fail}", flush=True)
    return ok, fail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--neg-ratio", type=int, default=2)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--probe", type=int, default=0,
                    help="extra CLEAN-CORPUS probe: N more random validation negatives, "
                         "written to probe.jsonl. Tightens the FPR confidence interval — "
                         "the one number a site operator actually sizes a queue on.")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    for split, role in (("test", "train"), ("validation", "eval")):
        recs = build_split(split, a.neg_ratio, a.seed)
        npos = sum(r["y"] for r in recs)
        print(f"{split} ({role}): {npos} pos / {len(recs) - npos} neg "
              f"(hard={sum(1 for r in recs if r.get("tier")=="hard")} verified={sum(1 for r in recs if r.get("tier")=="verified")} random={sum(1 for r in recs if r.get("tier")=="random")})")
        (OUT / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
        if a.dry_run:
            continue
        ok, fail = download(split, recs)
        print(f"  downloaded ok={ok} fail={fail}")

    if a.probe:
        pos, _, all_ids = labels("validation")
        used = {json.loads(x)["id"] for x in (OUT / "validation.jsonl").read_text().splitlines()}
        pool = sorted(set(all_ids) - used - {i for i, ls in pos.items() if ls & WEAPON.keys()})
        random.Random(a.seed + 1).shuffle(pool)
        recs = [{"id": i, "y": 0, "tier": "random", "classes": [],
                 "note": "clean-corpus FPR probe"} for i in pool[: a.probe]]
        (OUT / "probe.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
        print(f"probe: {len(recs)} clean-corpus negatives")
        if not a.dry_run:
            ok, fail = download("validation", recs)
            print(f"  downloaded ok={ok} fail={fail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
