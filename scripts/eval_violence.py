#!/usr/bin/env python
"""Measure the VIOLENCE / ABUSE track on everything that CAN be measured here.

EVAL DATA LAW — obeyed and load-bearing. No graphic-violence corpus is fetched to this
machine. Consequences, stated once and honoured everywhere below:

  * Every first-party number produced by this script is **FALSE-POSITIVE-side only**.
  * There is no recall number. τ is a FALSE-POSITIVE BUDGET (a quantile of the safe-corpus
    margin distribution), never a recall fit.
  * True-positive evidence is CITED from published model evaluations in
    `research/track-violence.md` and is labelled as cited every time it appears.

WHAT IS MEASURED
  A. COCO val2017 (5,000) — the baseline safe corpus, already embedded by the engine.
     Its 5-captions-per-image human annotations are ALSO mined for violence words, using
     the same technique track-safety uses for DANGER, so "COCO is a clean negative" is a
     MEASURED claim rather than an assumption.
  B. Unsplash keyword slices — the confusable classes a violence detector actually fails
     on. `contact-sport` (boxing / martial arts / wrestling / fencing / rugby) is the
     headline one; also costume-horror, protest, medical, red-liquid, military, misc-hard.

    uv run python scripts/eval_violence.py [--fetch] [--json research/eval-violence.json]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imgtag.core import models, store  # noqa: E402
from imgtag.moderation import violence  # noqa: E402

SLICE_DIR = ROOT / "data/unsplash-slices"
IMG_DIRS = (ROOT / "data/unsplash/images", ROOT / "data/unsplash-b", SLICE_DIR)
CACHE = ROOT / "research/bench_scripts/.violence_emb.npz"

#: Keyword slices. Every one is a class where a violence prompt bank can plausibly fire
#: for the wrong reason; `contact-sport` is the one the brief singles out.
SLICES: dict[str, list[str]] = {
    "contact-sport": ["boxing", "boxer", "martial arts", "karate", "judo", "wrestling",
                      "fencing", "rugby", "kickboxing", "taekwondo", "fight"],
    "team-sport": ["football", "hockey", "soccer", "basketball"],
    "costume-horror": ["halloween", "costume", "horror", "mask", "makeup", "scary"],
    "protest": ["protest", "riot", "demonstration", "activism", "police"],
    "medical": ["injury", "bandage", "hospital", "surgery", "ambulance", "doctor"],
    "red-liquid": ["ketchup", "paint", "wine", "tomato", "strawberry", "red"],
    "military": ["military", "soldier", "war", "army", "weapon", "gun", "tank"],
    "misc-hard": ["butcher", "meat", "graffiti", "fire", "explosion", "smoke", "tattoo",
                  "crowd", "dance", "concert"],
}

#: Caption-mined violence probe over COCO val2017. Same shape as track-safety's
#: DANGER_WORD regex, so the two COCO-derived label sets are comparable. Deliberately
#: WIDE (recall-first on the LABEL side): a word that over-fires costs us a manual look,
#: a word that under-fires would let us claim a clean negative we had not verified.
VIOLENCE_WORD = re.compile(
    r"\b(fight\w*|punch\w*|hit(?:ting|s)?|kick\w*|attack\w*|assault\w*|violen\w*|"
    r"beat(?:ing|en|s)?|stab\w*|shoot\w*|shot|gun|rifle|pistol|blood\w*|bleed\w*|"
    r"wound\w*|injur\w*|corpse|dead body|riot\w*|brawl\w*|strangl\w*|choking|abuse\w*)\b",
    re.I,
)
#: Words that make a VIOLENCE_WORD hit benign. Measured necessity: COCO captions are full
#: of "shooting a photo", "a blood orange", "hitting a baseball", "shot of the skyline".
BENIGN_CTX = re.compile(
    r"\b(photo|picture|camera|photograph\w*|shot of|screenshot|baseball|tennis|golf|"
    r"basketball|soccer|hockey|frisbee|ball|bat|racket|orange|moon|eclipse|star|"
    r"video game|movie|television|tv|painting|graffiti|sign|poster)\b",
    re.I,
)


# ── corpora ───────────────────────────────────────────────────────────────────────────
def unsplash_index() -> dict[str, Path]:
    out: dict[str, Path] = {}
    for d in IMG_DIRS:
        if d.is_dir():
            for f in d.iterdir():
                if f.suffix.lower() == ".jpg":
                    out.setdefault(f.stem, f)
    return out


def slice_ids() -> dict[str, set[str]]:
    """keyword -> photo_ids, from the Unsplash Lite keyword table (local metadata)."""
    want = {k for ks in SLICES.values() for k in ks}
    kw: dict[str, set[str]] = defaultdict(set)
    with open(ROOT / "data/unsplash/keywords.tsv000", encoding="utf8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r["keyword"] in want:
                kw[r["keyword"]].add(r["photo_id"])
    return kw


def build_slices() -> dict[str, list[Path]]:
    have, kw = unsplash_index(), slice_ids()
    out: dict[str, list[Path]] = {}
    for name, keys in SLICES.items():
        ids = sorted(set().union(*[kw.get(k, set()) for k in keys]) & set(have))
        if ids:
            out[name] = [have[i] for i in ids]
    return out


def coco_captions() -> dict[str, list[str]]:
    p = ROOT / "data/coco/annotations/captions_val2017.json"
    if not p.is_file():
        return {}
    d = json.loads(p.read_bytes())
    out: dict[str, list[str]] = defaultdict(list)
    for a in d["annotations"]:
        out[f"{a['image_id']:012d}.jpg"].append(a["caption"])
    return dict(out)


def caption_violence(caps: dict[str, list[str]], votes: int = 1) -> dict[str, list[str]]:
    """Images whose captions mention violence, benign-context words removed.

    `votes` = how many of the 5 independent captions must agree. 1 = the widest possible
    net, which is what we want when the CLAIM being tested is "this corpus is clean".
    """
    hit: dict[str, list[str]] = {}
    for name, cs in caps.items():
        got = [c for c in cs if VIOLENCE_WORD.search(c) and not BENIGN_CTX.search(c)]
        if len(got) >= votes:
            hit[name] = got
    return hit


# ── embedding ─────────────────────────────────────────────────────────────────────────
def embed_paths(backend, paths: list[Path], batch: int = 8) -> np.ndarray:
    from PIL import Image

    out, buf = [], []
    for i, p in enumerate(paths):
        try:
            with Image.open(p) as im:
                buf.append(backend.preprocess(im.convert("RGB")))
        except Exception:
            buf.append(np.zeros((backend.size, backend.size, 3), np.uint8))
        if len(buf) == batch or i == len(paths) - 1:
            out.append(backend.embed_images(np.stack(buf)))
            buf = []
            if (i + 1) % 400 < batch:
                print(f"  embedded {i + 1}/{len(paths)}", flush=True)
    return np.concatenate(out) if out else np.zeros((0, backend.dim), np.float32)


def slice_embeddings(model: str, refresh: bool = False):
    """(names, membership dict, [N,D]) for the union of all slice members, cached."""
    slices = build_slices()
    members = sorted({p for ps in slices.values() for p in ps})
    key = [str(p) for p in members]
    if CACHE.is_file() and not refresh:
        z = np.load(CACHE, allow_pickle=True)
        if list(z["paths"]) == key and str(z["model"]) == model:
            return slices, members, z["emb"]
    print(f"embedding {len(members)} slice images with {model} …", flush=True)
    backend = models.load_backend(model, {"intra_op": 4}, vision=True)
    emb = embed_paths(backend, members)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, paths=np.array(key), emb=emb, model=model)
    return slices, members, emb


# ── reporting ─────────────────────────────────────────────────────────────────────────
def q(a: np.ndarray, *ps) -> list[float]:
    return [round(float(np.quantile(a, p)), 4) for p in ps]


def rates(tier: np.ndarray) -> dict:
    n = max(len(tier), 1)
    return {t: round(float((tier == t).mean()) * 100, 2) for t in ("alert", "violation", "review")} | {"n": int(n)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cocoval2017")
    ap.add_argument("--model", default="pecore-s16-384")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--json", default=str(ROOT / "research/eval-violence.json"))
    a = ap.parse_args()

    out: dict = {"model": a.model, "law": "FP-side only; no violence corpus on this machine"}

    # -- A. COCO baseline ------------------------------------------------------------
    snap = store.open_snapshot(a.dataset)
    coco_emb = np.asarray(snap.emb, np.float32)
    coco_names = [Path(r["path"]).name for r in snap.ids]
    backend = models.load_backend(a.model, {}, vision=False)
    sc = violence.ViolenceScorer.build(backend)
    c = sc.score(coco_emb)
    m_all = c["margin"]
    print(f"\nCOCO {a.dataset}: n={len(coco_emb)}  margin "
          f"p50={q(m_all,.5)[0]} p95={q(m_all,.95)[0]} p99={q(m_all,.99)[0]} "
          f"p99.5={q(m_all,.995)[0]} p99.9={q(m_all,.999)[0]} max={m_all.max():.4f}")
    print(f"  p distribution: {q(c['p'],.05,.5,.95,.99,.999)}  (saturation check: "
          f"frac p>0.9 = {float((c['p'] > 0.9).mean()):.4f})")
    out["coco"] = {"n": int(len(coco_emb)),
                   "margin_q": dict(zip(["p50", "p90", "p95", "p99", "p995", "p999", "max"],
                                        q(m_all, .5, .9, .95, .99, .995, .999) + [round(float(m_all.max()), 4)])),
                   "p_q": dict(zip(["p05", "p50", "p95", "p99", "p999"], q(c["p"], .05, .5, .95, .99, .999))),
                   "p_over_0.9": round(float((c["p"] > 0.9).mean()), 5),
                   "tiers": rates(c["tier"])}
    print(f"  tier rates: {out['coco']['tiers']}")

    # -- A2. is COCO actually a clean negative? (measured, not assumed) ---------------
    caps = coco_captions()
    if caps:
        hit1 = caption_violence(caps, 1)
        hit2 = caption_violence(caps, 2)
        idx = {n: i for i, n in enumerate(coco_names)}
        rows = [(n, float(m_all[idx[n]]), hit1[n][0][:90]) for n in hit1 if n in idx]
        rows.sort(key=lambda r: -r[1])
        out["coco_caption_probe"] = {
            "n_captioned": len(caps), "hits_1of5": len(hit1), "hits_2of5": len(hit2),
            "top": [{"file": n, "margin": round(m, 4), "caption": c} for n, m, c in rows[:15]],
        }
        print(f"\nCOCO caption probe: {len(hit1)} images with >=1/5 violence-word captions, "
              f"{len(hit2)} with >=2/5 (of {len(caps)})")
        for n, m, cap in rows[:10]:
            print(f"   margin {m:+.4f}  {n}  {cap}")

    # -- B. confusable slices --------------------------------------------------------
    slices, members, emb = slice_embeddings(a.model, a.refresh)
    pos = {p: i for i, p in enumerate(members)}
    s = sc.score(emb)
    out["slices"] = {}
    print(f"\n{'slice':16s} {'n':>5s} {'mean m':>8s} {'max m':>8s} "
          f"{'alert%':>7s} {'viol%':>7s} {'rev%':>7s}")
    for name, paths in sorted(slices.items()):
        i = np.array([pos[p] for p in paths])
        r = rates(s["tier"][i])
        r |= {"mean_margin": round(float(s["margin"][i].mean()), 4),
              "max_margin": round(float(s["margin"][i].max()), 4),
              "p_mean": round(float(s["p"][i].mean()), 4)}
        out["slices"][name] = r
        print(f"{name:16s} {r['n']:5d} {r['mean_margin']:8.4f} {r['max_margin']:8.4f} "
              f"{r['alert']:7.2f} {r['violation']:7.2f} {r['review']:7.2f}")

    # top scorers per slice, named by their own Unsplash description (no image inspection)
    desc: dict[str, str] = {}
    with open(ROOT / "data/unsplash/photos.tsv000", encoding="utf8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            desc[row["photo_id"]] = (row.get("ai_description") or row.get("photo_description") or "")[:80]
    out["slice_tails"] = {}
    for name, paths in sorted(slices.items()):
        i = np.array([pos[p] for p in paths])
        order = i[np.argsort(-s["margin"][i])][:5]
        out["slice_tails"][name] = [
            {"id": members[j].stem, "margin": round(float(s["margin"][j]), 4),
             "tier": str(s["tier"][j]), "why": s["concept"][j],
             "desc": desc.get(members[j].stem, "")}
            for j in order
        ]

    # -- C. tier assignment is p-space bands (post-2026-07-22-incident) --------------
    # The head no longer arbitrates on separate margins; ADR-15 stores ONE score `p` and
    # tiers are ascending p-space bands (== store.derive_tiers, B25d). Confirm the head's
    # tier matches derive_tiers on this corpus — the invariant the nudityprobe fix rests on.
    from imgtag.core.store import derive_tiers
    spec = violence.track_spec()
    out["derive_tiers_consistent"] = bool(list(c["tier"]) == derive_tiers(c["p"], spec))
    print(f"\nhead tier == store.derive_tiers on COCO: {out['derive_tiers_consistent']}")

    # -- D. threshold sweep in `p`-space (the space tiers are actually banded in) -----
    out["p_sweep"] = {}
    print("\nflag-rate vs p-tau (FP budget, % of corpus at/above tau in p-space)")
    hdr = ["tau", "coco"] + sorted(slices)
    print("  " + " ".join(f"{h[:12]:>12s}" for h in hdr))
    for tau in (0.30, 0.40, 0.46, 0.55, 0.65, 0.75, 0.85, 0.90, 0.95):
        row = {"coco": round(float((c["p"] >= tau).mean()) * 100, 2)}
        for name, paths in sorted(slices.items()):
            i = np.array([pos[p] for p in paths])
            row[name] = round(float((s["p"][i] >= tau).mean()) * 100, 2)
        out["p_sweep"][f"{tau:g}"] = row
        print("  " + f"{tau:>12g}" + " ".join(f"{row[h]:12.2f}" for h in hdr[1:]))

    Path(a.json).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
