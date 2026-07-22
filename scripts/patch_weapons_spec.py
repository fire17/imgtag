#!/usr/bin/env python3
"""Deepen the WEAPONS entry in src/imgtag/data/moderation.json with the authored
subcategory taxonomy (user directive 2026-07-22 13:58Z) — as versioned DATA, never
hot-path code (TRACKS.md T3). Idempotent + collision-safe (rule 7): refuses to write
unless moderation.json has been quiescent for >=QUIET seconds, edits ONLY
`categories.weapons`, writes atomically. Re-runnable with zero agents (T4).

    .venv/bin/python scripts/patch_weapons_spec.py [--quiet 180] [--force]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MOD = ROOT / "src" / "imgtag" / "data" / "moderation.json"

# ── The deepened weapons spec — subcategory taxonomy as prompt-set DATA ──
# Enforcement ships via the TRAINED head (weapons-<backend>.json, a logistic over the
# embedding); these prompt sets drive the zero-shot fallback + the app's live scoring and
# are the human-readable taxonomy record. Toy/replica stays REVIEW tier (ADR-14).
SUBCATEGORIES = {
    "handguns_pistols_revolvers": [
        "a handgun", "a pistol", "a revolver", "a semi-automatic pistol",
        "a person holding a handgun", "a holstered pistol on a belt"],
    "rifles": [
        "a rifle", "an assault rifle", "a hunting rifle", "a bolt-action rifle",
        "a soldier carrying a rifle", "a person aiming a rifle"],
    "submachine_automatic": [
        "a submachine gun", "a machine gun", "an automatic firearm", "a mounted machine gun"],
    "shotguns": [
        "a shotgun", "a pump-action shotgun", "a double-barrel shotgun"],
    "ammunition": [
        "live ammunition and bullets", "a loaded ammunition magazine", "a belt of ammunition"],
    "knives_threat": [
        "a knife held as a weapon in a threatening way", "a combat knife",
        "a hunting knife", "a dagger", "a switchblade", "a bayonet"],
    "swords_machetes_axes": [
        "a sword", "a katana", "a sabre", "a machete", "a battle axe as a weapon"],
    "bows_crossbows": [
        "a bow and arrow", "a crossbow", "a person aiming a bow and arrow"],
    "explosives_grenades": [
        "a hand grenade", "an explosive device", "a bomb", "a stick of dynamite",
        "a landmine"],
    "heavy_ordnance": [
        "a military tank", "an artillery cannon", "a missile", "a rocket launcher",
        "a mortar", "a howitzer"],
}

WEAPONS_ENTRY = {
    "label": "weapons",
    # violation = the real policy breach, deepened across every subcategory (flattened
    # from `subcategories` below so the existing reader keeps working unchanged).
    "violation": sorted({p for ps in SUBCATEGORIES.values() for p in ps}),
    # review = toy/replica/prop — ADR-14 keeps these at REVIEW tier, never violation.
    "review": [
        "a toy gun", "plastic toy soldiers with toy rifles", "a replica or prop weapon",
        "a water pistol", "a nerf blaster toy", "a cap gun toy",
        "a video game controller shaped like a gun",
        "a museum display of antique swords"],
    # negatives = the MEASURED false-positive classes (research/track-weapons.md §5.2/§5.3):
    # baseball scene is the #1 FP (10.9x), long-lens camera reads as a rifle, military
    # airframes read as ordnance. Kitchen-knife stays a negative (OI boundary, ADR-14).
    "negatives": [
        "a kitchen knife cutting vegetables on a board", "cutlery on a dining table",
        "a person holding a phone", "a power drill or hand tool",
        "people playing baseball on a field", "a baseball glove and bat",
        "a tennis racket", "a person holding a guitar",
        "a camera with a long telephoto lens",
        "a military jet aircraft flying", "a helicopter in the sky",
        "a construction site with tools"],
    # subcategory taxonomy (user directive 13:58Z) — the versioned record; companion to
    # data/weapon-probe/taxonomy.json which carries the TP probe image ids per subcategory.
    "subcategories": SUBCATEGORIES,
    "toy_replica_tier": "review",
    "taxonomy_source": "data/weapon-probe/taxonomy.json",
    "measurement": "research/eval-weapons.json (TP-vs-FP-band separation, per subcategory)",
    "note": "Enforcement ships via the trained head (weapons-<backend>.json); these prompt "
            "sets are the zero-shot fallback + taxonomy record. Toy/replica = review "
            "(ADR-14); real weapons = violation. GAPS (no held-out OI imagery): "
            "explosives_grenades (0 `Bomb` images), submachine/automatic + crossbow have "
            "no distinct OI label so are unmeasured separately.",
}


def spec_sha(entry: dict) -> str:
    payload = json.dumps({k: entry[k] for k in ("violation", "review", "negatives",
                                                "subcategories")}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", type=int, default=180,
                    help="require moderation.json untouched for this many seconds (rule 7)")
    ap.add_argument("--force", action="store_true", help="skip the quiescence guard")
    a = ap.parse_args()

    age = time.time() - MOD.stat().st_mtime
    if age < a.quiet and not a.force:
        print(f"ABORT: moderation.json touched {age:.0f}s ago (< {a.quiet}s). "
              "A parallel lane may be writing it — retry when quiet (rule 7).")
        return 3

    d = json.loads(MOD.read_text())          # re-read immediately before writing
    entry = copy.deepcopy(WEAPONS_ENTRY)
    entry["spec_sha"] = spec_sha(entry)
    if d["categories"].get("weapons") == entry:
        print("no change — weapons entry already up to date (idempotent).")
        return 0
    before = d["categories"].get("weapons", {})
    d["categories"]["weapons"] = entry

    tmp = MOD.with_suffix(".json.weapons-tmp")
    tmp.write_text(json.dumps(d, indent=1, ensure_ascii=False))
    os.replace(tmp, MOD)
    print(f"patched categories.weapons (violation {len(before.get('violation', []))}"
          f"->{len(entry['violation'])} prompts, {len(SUBCATEGORIES)} subcategories, "
          f"spec_sha {entry['spec_sha']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
