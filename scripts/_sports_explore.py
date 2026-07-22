#!/usr/bin/env python
"""Measurement harness for the sports content track. Not shipped — reproduces every
number in research/track-sports.md against CORPUS-A (COCO val2017, 5000 imgs).

  uv run python scripts/_sports_explore.py [--fit]

Ground truth
  primary : COCO `sports` supercategory (10 children) — exhaustive, 938/5000 positives.
  extra   : LVIS v1 val sports categories restricted to val2017 — the sports COCO lumps
            into `sports ball` or does not annotate at all (basketball, volleyball,
            golf_club, hockey_stick, dumbbell, boxing_glove, ...).
  weak    : Unsplash Lite keyword rows (labeled WEAK — photographer/AI keywords, not
            exhaustive) for the activity-only classes COCO cannot score.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from imgtag.core import models, store                      # noqa: E402
from imgtag.core.tags import fit_platt                      # noqa: E402
from imgtag.moderation import sports                       # noqa: E402

BACKEND = "pecore-s16-384"
CORPUS = "cocoval2017"

# LVIS categories that are sport but are NOT their own COCO child (or are not COCO at
# all). Used as EXTRA ground truth, never as a negative.
LVIS_SPORT = {
    "basketball", "basketball_backboard", "volleyball", "soccer_ball", "softball",
    "baseball", "tennis_ball", "ping-pong_ball", "beachball", "football_(American)",
    "football_helmet", "golf_club", "hockey_stick", "dumbbell", "barbell",
    "boxing_glove", "bowling_ball", "ice_skate", "roller_skate", "water_ski",
    "table-tennis_table", "scoreboard", "home_plate_(baseball)", "mound_(baseball)",
    "baseball_base", "ski_pole", "ski_boot", "racket", "paddle", "kayak",
    "mat_(gym_equipment)",
}
# LVIS classes that share a scene with sport but are NOT sport — the hard-negative probe.
LVIS_HARD_NEG = {"bicycle", "helmet", "jersey", "glove", "ball", "kite", "horse"}


def coco_gt() -> tuple[dict, set]:
    d = json.loads((ROOT / "data/coco/annotations/instances_val2017.json").read_bytes())
    sport = {c["id"]: c["name"] for c in d["categories"] if c["supercategory"] == "sports"}
    per: dict[str, set] = {}
    for a in d["annotations"]:
        if a["category_id"] in sport:
            per.setdefault(f"{a['image_id']:012d}.jpg", set()).add(sport[a["category_id"]])
    return per, set(sport.values())


def lvis_gt() -> tuple[dict, dict]:
    d = json.loads((ROOT / "data/lvis/lvis_val2017_only.json").read_bytes())
    cats = {c["id"]: c["name"] for c in d["categories"]}
    imgs = {i["id"]: Path(i.get("coco_url", str(i["id"]))).name for i in d["images"]}
    pos: dict[str, set] = {}
    neg: dict[str, set] = {}
    for a in d["annotations"]:
        n = cats.get(a["category_id"], "")
        f = imgs.get(a["image_id"])
        if not f:
            continue
        if n in LVIS_SPORT:
            pos.setdefault(f, set()).add(n)
        elif n in LVIS_HARD_NEG:
            neg.setdefault(f, set()).add(n)
    return pos, neg


def ap(p, y):
    return sports.average_precision(p, y)


def r_at_fpr(s, y, f):
    t = float(np.quantile(s[~y], 1 - f))
    return float((s[y] >= t).mean())


def report(name, s, y):
    r = {"AP": round(ap(s, y), 4), "R@fpr1%": round(r_at_fpr(s, y, 0.01), 3),
         "R@fpr5%": round(r_at_fpr(s, y, 0.05), 3),
         "R@fpr10%": round(r_at_fpr(s, y, 0.10), 3)}
    print(f"  {name:28s} {r}")
    return r


def main() -> None:
    fit_mode = "--fit" in sys.argv
    backend = models.load_backend(BACKEND, {}, vision=False)
    snap = store.open_snapshot(CORPUS)
    emb = np.ascontiguousarray(np.asarray(snap.emb, np.float32))
    names = [Path(r["path"]).name for r in snap.ids]
    n = len(names)

    per, children = coco_gt()
    y = np.array([nm in per for nm in names])
    lv_pos, lv_neg = lvis_gt()
    y_lvis = np.array([nm in lv_pos for nm in names])
    y_any = y | y_lvis
    print(f"CORPUS-A n={n}  COCO-sports pos={int(y.sum())} ({y.mean():.1%})  "
          f"+LVIS-sports pos={int(y_any.sum())} ({y_any.mean():.1%})")

    # ── bank ablation: equipment vs activity vs both, raw cosine vs background margin ──
    texts, labels = sports.flat_prompts()
    bg = sports.background_prompts()
    P = np.asarray(backend.embed_texts(texts), np.float32)
    B = np.asarray(backend.embed_texts(bg), np.float32)
    ACTIVITY = {"swimming", "running", "gym", "climbing", "cycling", "martial arts",
                "gymnastics", "equestrian", "motorsport", "stadium game", "team sport"}
    eq_idx = [i for i, l in enumerate(labels) if l not in ACTIVITY]
    ac_idx = [i for i, l in enumerate(labels) if l in ACTIVITY]

    cp_all = emb @ P.T
    bgmax = (emb @ B.T).max(1)
    print("\n[1] bank ablation vs COCO sports-supercategory truth")
    res = {}
    for key, idx in (("equipment", eq_idx), ("activity", ac_idx), ("all", list(range(len(labels))))):
        c = cp_all[:, idx].max(1)
        res[f"{key}/raw"] = report(f"{key}/raw-cosine", c, y)
        res[f"{key}/margin"] = report(f"{key}/bg-margin", c - bgmax, y)

    s = cp_all.max(1) - bgmax
    print("\n[2] same score vs COCO+LVIS sports truth (the fairer denominator)")
    res["all/margin@lvis"] = report("all/bg-margin", s, y_any)

    # ── why the margin loses: WHICH background prompts steal it from positives ──
    print("\n[1b] background-bank ablation (all-positives bank, COCO truth)")
    NEAR_SPORT = [                      # REJECTED: background prompts naming a sport SCENE
        "an empty sports stadium with no people", "an empty gymnasium",
        "an empty green grass field", "a beach with people sunbathing",
        "a snowy mountain landscape with no people", "a swimming pool with nobody in it",
        "a bicycle parked on a city street", "a person in athletic clothing standing on a street",
        "a toy ball on the floor", "a children's playground with a slide",
    ]
    FAR_HARD = [                        # REJECTED: hard negatives that are not sport scenes
        "a concert in a stadium with stage lights", "a locker room with lockers",
        "a fashion photo of sportswear", "a sports logo printed on a t-shirt",
        "a scoreboard-shaped advertising billboard", "a person walking a dog in a park",
        "a car parked on the road", "a group of people posing for a photo",
    ]
    GENERIC = list(sports.NEGATIVE_PROMPTS)                  # the SHIPPED bank
    BORDER = [q for qs in sports.BORDERLINE_PROMPTS.values() for q in qs]
    variants = {
        "none (raw cosine)": [],
        "generic only": GENERIC,
        "generic + borderline (SHIPPED)": sports.background_prompts(),
        "generic + far-hard": GENERIC + FAR_HARD,
        "full bank (v0 draft)": GENERIC + FAR_HARD + NEAR_SPORT,
        "full + borderline (v0 draft)": GENERIC + FAR_HARD + NEAR_SPORT + BORDER,
    }
    cmax = cp_all.max(1)
    for vname, bank in variants.items():
        if not bank:
            report(vname, cmax, y)
            continue
        Bv = np.asarray(backend.embed_texts(bank), np.float32)
        cb = emb @ Bv.T
        report(vname + " /max", cmax - cb.max(1), y)
        if len(bank) >= 3:
            report(vname + " /mean-top3", cmax - np.sort(cb, 1)[:, -3:].mean(1), y)

    # ── calibration + tau on a held-out half ─────────────────────────────────
    rng = np.random.default_rng(17)
    perm = rng.permutation(n)
    tr, va = perm[: n // 2], perm[n // 2:]
    head = sports.SportsHead.build(backend)
    head = sports.fit(head, emb[tr], y[tr], val=(emb[va], y[va]))
    print(f"\n[3] fitted head (platt on train n={len(tr)}, tau on held-out n={len(va)})")
    print("  " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                             for k, v in head.metrics.items() if k != "spread"}))
    print("  spread " + json.dumps(head.metrics["spread"]))
    p_all = head.probs(emb)
    hist, edges = np.histogram(p_all, bins=20, range=(0, 1))
    print("  p-histogram (all 5000, 20 bins 0..1):")
    for i, h in enumerate(hist):
        print(f"    {edges[i]:.2f}-{edges[i+1]:.2f} {'#' * max(1, int(60 * h / hist.max())) if h else ''} {h}")

    # precision/recall sweep so the threshold ruling is auditable
    print("\n[4] operating points on the held-out half (COCO truth)")
    for tgt in (0.60, 0.70, 0.80, 0.90):
        t = sports.tau_for_precision(p_all[va], y[va], tgt)
        m = sports.prf(p_all[va], y[va], t)
        print(f"  target_prec={tgt:.2f} tau={t:.4f} prec={m['precision']:.3f} "
              f"rec={m['recall']:.3f} f1={m['f1']:.3f} match_rate={m['match_rate']:.3f}")

    tau = head.tau_match
    # ── per-child recall ─────────────────────────────────────────────────────
    print(f"\n[5] per-COCO-child recall at tau={tau:.4f}")
    for c in sorted(children):
        idx = [i for i, nm in enumerate(names) if nm in per and c in per[nm]]
        if idx:
            print(f"  {c:16s} n={len(idx):4d} recall={np.mean(p_all[idx] >= tau):.3f}")

    print(f"\n[6] per-LVIS-extra recall at tau={tau:.4f} (sports COCO does not name)")
    counts: dict[str, list] = {}
    for i, nm in enumerate(names):
        for c in lv_pos.get(nm, ()):
            counts.setdefault(c, []).append(i)
    for c, idx in sorted(counts.items(), key=lambda kv: -len(kv[1])):
        if len(idx) >= 5:
            print(f"  {c:24s} n={len(idx):4d} recall={np.mean(p_all[idx] >= tau):.3f}")

    # ── false positives: what does it match that GT says is not sport? ───────
    print(f"\n[7] false-positive anatomy at tau={tau:.4f} (COCO+LVIS truth)")
    fp = np.flatnonzero((p_all >= tau) & ~y_any)
    tp = np.flatnonzero((p_all >= tau) & y_any)
    print(f"  matched={int((p_all>=tau).sum())} tp={len(tp)} fp={len(fp)} "
          f"precision={len(tp)/max(len(tp)+len(fp),1):.3f}")
    best = cp_all.argmax(1)
    by: dict[str, int] = {}
    for i in fp:
        by[labels[best[i]]] = by.get(labels[best[i]], 0) + 1
    for k, v in sorted(by.items(), key=lambda kv: -kv[1])[:12]:
        print(f"    {k:18s} {v}")
    print("  hard-negative classes (LVIS bicycle/horse/kite/... with NO sport ann):")
    for c in sorted(LVIS_HARD_NEG):
        idx = [i for i, nm in enumerate(names) if c in lv_neg.get(nm, ()) and not y_any[i]]
        if len(idx) >= 5:
            print(f"    {c:12s} n={len(idx):4d} fp_rate={np.mean(p_all[idx] >= tau):.3f}")
    print("  top-20 FP filenames (for eyeballing):")
    for i in fp[np.argsort(-p_all[fp])][:20]:
        print(f"    {names[i]} p={p_all[i]:.3f} -> {labels[best[i]]}")

    # ── sport-label distribution on true positives ───────────────────────────
    print("\n[8] reported `sport` label on matched true positives")
    by = {}
    for i in tp:
        by[labels[best[i]]] = by.get(labels[best[i]], 0) + 1
    for k, v in sorted(by.items(), key=lambda kv: -kv[1]):
        print(f"    {k:18s} {v}")

    print("\n[8b] FP samples per group (for the manual audit)")
    for g in ("motorsport", "equestrian", "kite", "water sports"):
        ids = [i for i in fp if labels[best[i]] == g]
        ids = sorted(ids, key=lambda i: -p_all[i])[:: max(1, len(ids) // 4)][:4]
        print(f"    {g:14s} " + " ".join(names[i] for i in ids))

    # ── [9] weak-label cross-domain probe: Unsplash Lite keywords ────────────
    print("\n[9] WEAK-LABEL probe on unsplashb (photographer/AI keywords, NOT exhaustive)."
          "\n    Each bank is RE-FITTED on COCO (platt on train, tau at precision 0.80 on"
          "\n    held-out) before it is evaluated here — otherwise the banks are on"
          "\n    different margin scales and the comparison is meaningless.")
    try:
        us = store.open_snapshot("unsplashb")
    except Exception as e:                                   # noqa: BLE001
        print(f"    skipped: {e}")
        us = None
    if us is not None:
        uemb = np.ascontiguousarray(np.asarray(us.emb, np.float32))
        uids = [Path(r["path"]).stem for r in us.ids]
        want = set(uids)
        kw: dict[str, set] = {}
        with open(ROOT / "data/unsplash/keywords.tsv000", encoding="utf8", errors="replace") as fh:
            next(fh)
            for line in fh:
                p = line.split("\t")
                if len(p) > 1 and p[0] in want:
                    kw.setdefault(p[0], set()).add(p[1])
        SPORT_KW = {"sport", "sports", "athlete", "soccer", "football", "basketball",
                    "tennis", "swimming", "running", "marathon", "gym", "fitness",
                    "workout", "climbing", "cycling", "surfing", "skiing", "golf",
                    "boxing", "skateboard", "stadium"}
        GROUPS = {
            "sport-keyword (weak +)": SPORT_KW,
            "concert / festival / music": {"concert", "festival", "music", "band"},
            "crowd (no sport kw)": {"crowd"},
            "hiking (borderline, OUT)": {"hiking"},
            "yoga (borderline, OUT)": {"yoga"},
            "chess (borderline, OUT)": {"chess"},
            "food": {"food"}, "architecture": {"architecture"},
        }
        for bname, bank in (("generic + borderline (SHIPPED)", sports.background_prompts()),
                            ("generic only", GENERIC),
                            ("generic + far-hard", GENERIC + FAR_HARD),
                            ("full + borderline (v0 draft)",
                             GENERIC + FAR_HARD + NEAR_SPORT + BORDER)):
            Bv = np.asarray(backend.embed_texts(bank), np.float32)
            mc = cp_all.max(1) - (emb @ Bv.T).max(1)         # re-fit THIS bank on COCO
            pl = fit_platt(mc[tr], y[tr])
            pc = sports.squash(mc, pl)
            tb = sports.tau_for_precision(pc[va], y[va], sports.MATCH_PRECISION)
            mcoco = sports.prf(pc[va], y[va], tb)
            m = (uemb @ P.T).max(1) - (uemb @ Bv.T).max(1)
            pu = sports.squash(m, pl)
            print(f"    -- background = {bname}: COCO AP={sports.average_precision(pc[va], y[va]):.4f} "
                  f"tau={tb:.4f} prec={mcoco['precision']:.3f} rec={mcoco['recall']:.3f}"
                  f" | unsplash n={len(uids)} overall match_rate={np.mean(pu >= tb):.3f}")
            tau_b = tb
            for gname, terms in GROUPS.items():
                idx = [i for i, u in enumerate(uids)
                       if kw.get(u, set()) & terms
                       and (gname.startswith("sport") or not (kw.get(u, set()) & SPORT_KW))]
                if len(idx) >= 8:
                    print(f"       {gname:28s} n={len(idx):4d} match_rate="
                          f"{np.mean(pu[idx] >= tau_b):.3f}")

    if fit_mode:
        full = sports.SportsHead.build(backend)
        full = sports.fit(full, emb[tr], y[tr], val=(emb[va], y[va]))
        p = sports.save(full)
        print(f"\nsaved {p} ({p.stat().st_size/1024:.0f} KB)")

    np.save("/tmp/sports_p.npy", p_all)
    (ROOT / ".scratch/sports_names.json").write_text(json.dumps(names))


if __name__ == "__main__":
    main()
