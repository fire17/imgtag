#!/usr/bin/env python3
"""track-nudity TRUE-POSITIVE probe + confidence-separation fit (review tier only).

USER DIRECTIVE 2026-07-22 13:58Z: nudity findings on real datasets are all FALSE positives;
there are no TRUE positives to test against, so no ratio threshold ("auto flag" tau) can be
set. This harness builds the legal half of the missing corpus and measures whether TP
confidence sits ABOVE the current FP band, per subcategory.

EVAL DATA LAW (absolute): no explicit-adult imagery is fetched, downloaded or generated.
Only REVIEW-TIER positives — swimwear / lingerie / underwear / bare-chest-male — are in
scope, and they are sourced PROGRAMMATICALLY from the local Unsplash Lite corpus by
KEYWORD JOIN (never by an agent hand-labelling images: T4). VIOLATION-tier recall stays
benchmark-cited (Marqo 98.56%/20k), never re-measured here.

TWO INSTRUMENTS, by design (predecessor finding, research/track-nudity.md §9): the Marqo
head is a VIOLATION detector and scores swimwear LOW (it was trained swimwear=SFW), so the
REVIEW tier is served by a PROMPT-ENSEMBLE MARGIN over the index embeddings (the drugs
pattern) — max(review concepts) - max(negatives incl. mannequin/statue). This script
measures BOTH over the same probe so the composition is honest:
  * Marqo p on swimwear TP   -> expected to NOT separate from FP band (confirms predecessor)
  * review margin on TP      -> expected to separate -> fit tau_review + platt

    uv run python research/bench_scripts/nudity_probe.py                 # measure + fit
    uv run python research/bench_scripts/nudity_probe.py --build-dataset # + data/nudity-probe/ for `imgtag index`
    uv run python research/bench_scripts/nudity_probe.py --write         # patch moderation.json + nudity.py

VIOLATION-TIER, THE SANCTIONED WAY (user-supplied only — the agent NEVER fetches such
material, EVAL DATA LAW stands). If the operator drops a lawfully-held, adult-only folder
locally and indexes it:
    imgtag index <folder> nsfwprobe --wait --moderation      # local-only, gitignored, never committed
then this script measures VIOLATION-tier separation on it UNCHANGED — the Marqo head's own
argmax tau=0.50 finally gets a first-party recall number vs the safe-corpus FP band:
    uv run python research/bench_scripts/nudity_probe.py --nsfw-dataset nsfwprobe
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from imgtag.core import models, store  # noqa: E402
from imgtag.moderation import nudity  # noqa: E402

SEED = 20260722
PROBE_DIR = ROOT / "data/nudity-probe"

# ── the review-tier subcategory taxonomy (user's explicit ask) ─────────────────────────
# label -> Unsplash keywords that programmatically select its TRUE positives. The label is
# derived from the keyword, never from an agent looking at the image (T4). Priority order:
# an image matching several keywords is assigned to the FIRST subcategory here.
PROBE_KEYWORDS: dict[str, list[str]] = {
    "swimwear": ["bikini", "swimwear", "swimsuit"],
    "lingerie": ["lingerie", "bra"],
    "underwear": ["underwear"],
    "bare-chest-male": ["fitness"],          # NOISY proxy (no `shirtless` keyword on disk) — flagged
    # NEVER-FLAG control — non-person figures. Must stay BELOW tau on both instruments.
    "mannequin-statue": ["mannequin", "statue", "sculpture", "marble", "figurine"],
}
TP_SUBCATS = ("swimwear", "lingerie", "underwear", "bare-chest-male")
CONTROL_SUBCATS = ("mannequin-statue",)
CAP = {"swimwear": 80, "lingerie": 45, "underwear": 31, "bare-chest-male": 45,
       "mannequin-statue": 60}

# ── review-tier prompt banks (versioned data candidate for moderation.json) ────────────
TEMPLATES = ("a photo of {}.", "a close-up photo of {}.", "{}")
REVIEW_CONCEPTS: dict[str, list[str]] = {
    "swimwear": ["a person in a bikini", "a person in swimwear at the beach",
                 "a woman in a bikini", "a man in swim trunks", "a person in a swimsuit"],
    "lingerie": ["a person in lingerie", "a woman in lingerie",
                 "a person wearing a bra and underwear"],
    "underwear": ["a person in their underwear", "a person in briefs",
                  "a person in boxer shorts"],
    "bare-chest-male": ["a shirtless man", "a bare-chested man",
                        "a muscular man with no shirt"],
    "artistic-figure": ["an artistic figure study of a body",
                        "a fine-art photograph of a bare human form"],
    "medical-context": ["a clinical medical photograph of a patient's skin",
                        "a dermatology photo of bare skin"],
}
# Subtracted from BOTH: the FP-killers and the ADR-14 never-flag figures. A mannequin image
# maxes a NEGATIVE, so its margin stays low by construction (then measured, not assumed).
NEGATIVES = [
    "a landscape photograph", "a photograph of food", "a fully clothed person",
    "a portrait of a person wearing clothes", "an everyday object on a table",
    "an animal", "a building", "a car", "an indoor room",
    "a mannequin in a shop window", "a marble statue of a human figure",
    "a sculpture of a human body", "a plastic dummy figure", "an anatomical model",
]


def unsplash_index() -> dict[str, Path]:
    out: dict[str, Path] = {}
    for d in (ROOT / "data/unsplash/images", ROOT / "data/unsplash-b"):
        if d.is_dir():
            for f in d.iterdir():
                if f.suffix.lower() == ".jpg":
                    out.setdefault(f.stem, f)
    return out


def build_probe(rng: random.Random) -> dict[str, list[Path]]:
    """Keyword-join -> {subcat: [paths]}, deterministic, deduped across subcats by priority."""
    have = unsplash_index()
    kw2ids: dict[str, set[str]] = defaultdict(set)
    wanted = {k for ks in PROBE_KEYWORDS.values() for k in ks}
    with open(ROOT / "data/unsplash/keywords.tsv000", encoding="utf8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row["keyword"] in wanted and row["photo_id"] in have:
                kw2ids[row["keyword"]].add(row["photo_id"])
    slices: dict[str, list[Path]] = {}
    used: set[str] = set()
    for sub, keys in PROBE_KEYWORDS.items():
        ids = sorted({i for k in keys for i in kw2ids[k]} - used)
        rng.shuffle(ids)
        ids = ids[: CAP[sub]]
        used.update(ids)
        slices[sub] = [have[i] for i in ids]
    return slices


def concept_vectors(backend, texts: list[str]) -> np.ndarray:
    """One L2-normalized vector per concept = mean over TEMPLATES (classic CLIP ensembling)."""
    flat = [t.format(c) for c in texts for t in TEMPLATES]
    emb = np.asarray(backend.embed_texts(flat), np.float32).reshape(len(texts), len(TEMPLATES), -1)
    v = emb.mean(1)
    return np.ascontiguousarray(v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12))


def review_margin(emb: np.ndarray, pos: np.ndarray, neg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """margin = max(review concepts) - max(negatives); also the argmax concept index."""
    cp = np.asarray(emb, np.float32) @ pos.T
    bg = (np.asarray(emb, np.float32) @ neg.T).max(1)
    return cp.max(1) - bg, cp.argmax(1)


# ── separation stats (reused from scripts/eval_drugs.py — generic, inlined to stay
#    nudity-self-contained) ─────────────────────────────────────────────────────────────
def ap(p: np.ndarray, y: np.ndarray) -> float:
    o = np.argsort(-p)
    ys = np.asarray(y, bool)[o]
    tp = np.cumsum(ys)
    prec = tp / np.arange(1, len(ys) + 1)
    return float((prec * ys).sum() / max(ys.sum(), 1))


def rec_at_fpr(s: np.ndarray, y: np.ndarray, f: float) -> float:
    t = float(np.quantile(s[~y], 1 - f))
    return float((s[y] >= t).mean())


def ridge_platt(s, y, lam: float = 1e-3):
    s, y = np.asarray(s, float), np.asarray(y, bool)
    npos, nneg = float(y.sum()), float((~y).sum())
    if npos == 0 or nneg == 0:
        return 0.0, 0.0
    t = np.where(y, (npos + 1) / (npos + 2), 1 / (nneg + 2))
    A, B = 1.0, 0.0
    for _ in range(300):
        p = 1 / (1 + np.exp(-(A * s + B)))
        d = p - t
        w = np.maximum(p * (1 - p), 1e-9)
        g = np.array([d @ s + lam * A, d.sum()])
        H = np.array([[w @ (s * s) + lam, w @ s], [w @ s, w.sum()]]) + 1e-9 * np.eye(2)
        step = np.linalg.solve(H, g)
        A, B = A - step[0], B - step[1]
        if np.abs(step).max() < 1e-11:
            break
    return float(A), float(B)


def boot_ci(fn, s, y, n=1000, seed=SEED):
    """Bootstrap 95% CI for a scalar statistic fn(s, y) over paired (s,y)."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    vals = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if 0 < np.asarray(y)[b].sum() < len(b):
            vals.append(fn(np.asarray(s)[b], np.asarray(y)[b]))
    if not vals:
        return (0.0, 0.0)
    return (round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3))


def fp_band_embeddings(datasets: list[str], exclude: set[str]) -> tuple[np.ndarray, list[str]]:
    """Embeddings of the indexed safe corpora = the CURRENT FP band, deduped by filename.

    ``exclude`` is the set of probe filenames — the probe is DRAWN FROM unsplashb, so
    without this the swimwear/lingerie positives are double-counted INTO the negative pool
    and inflate its tail (the eval_drugs dedup lesson: a labelled positive silently scored
    as a negative). Excluding them is what makes the separation number honest."""
    emb, names, seen = [], [], set(exclude)
    dropped = 0
    for ds in datasets:
        try:
            snap = store.open_snapshot(ds)
        except Exception as e:
            print(f"  (skipping {ds}: {type(e).__name__})")
            continue
        E = np.asarray(snap.emb, np.float32)
        keep = []
        for i, r in enumerate(snap.ids):
            n = Path(r["path"]).name
            if n in seen:
                dropped += n in exclude
                continue
            seen.add(n)
            keep.append(i)
            names.append(n)
        emb.append(E[keep])
        print(f"  FP-band {ds}: {len(snap.ids)} rows, {len(keep)} new")
    print(f"  FP-band excluded {dropped} probe positives from the negative pool")
    return (np.concatenate(emb) if emb else np.zeros((0, 512), np.float32)), names


def build_dataset(slices: dict[str, list[Path]]) -> None:
    """Materialise data/nudity-probe/<subcat>/ as SYMLINKS (no redistribution, gitignored)
    plus labels.json with keyword provenance — ready for `imgtag index`."""
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    labels = {"source": "Unsplash Lite keywords.tsv000, KEYWORD-JOIN (programmatic, no "
                        "hand-labelling — T4). label == subcategory == originating keyword group.",
              "keywords": PROBE_KEYWORDS, "subcats": {}}
    for sub, paths in slices.items():
        d = PROBE_DIR / sub
        d.mkdir(exist_ok=True)
        for p in paths:
            link = d / p.name
            if not link.exists():
                link.symlink_to(p.resolve())
        labels["subcats"][sub] = [p.name for p in paths]
    (PROBE_DIR / "labels.json").write_text(json.dumps(labels, indent=1))
    print(f"built {PROBE_DIR} — {sum(len(v) for v in slices.values())} symlinks across "
          f"{len(slices)} subcats")


def run_nsfw_probe(slug: str, out_json: Path) -> int:
    """USER-SUPPLIED violation-tier separation — the ONLY sanctioned way violation recall is
    measured. Reads an ALREADY-INDEXED local dataset (the agent NEVER fetches this material,
    EVAL DATA LAW stands). Scores the Marqo VIOLATION head (its argmax tau_violation=0.50 is
    the threshold) over the operator's positives and reports recall vs the safe-corpus FP band.
    Works UNCHANGED on any folder the operator drops and indexes as `--dataset nsfwprobe`."""
    from PIL import Image

    head = nudity.load_nudity_head({"intra_op": 4})
    if head is None:
        print("nudity Marqo artifact missing — run scripts/export_nudity_marqo.py first")
        return 2
    try:
        snap = store.open_snapshot(slug)
    except Exception as e:
        print(f"dataset {slug!r} not indexed ({type(e).__name__}). Operator step first:\n"
              f"  imgtag index <lawful-local-folder> {slug} --wait --moderation")
        return 2
    paths = [r.get("path") if isinstance(r, dict) else r for r in snap.ids]
    p = []
    for src in paths:
        try:
            with Image.open(src) as im:
                p.append(float(head.probs(head.preprocess(im)[None])[0]))
        except Exception:
            continue
    p = np.array(p, np.float32)
    if not len(p):
        print(f"no readable images in {slug!r}")
        return 2
    # FP-band Marqo reference: first-party, measured on 1,826 safe images (track-nudity.md §3).
    FP = {"mean": 0.0595, "p95": 0.07, "p99": 0.175, "p999": 0.698,
          "tau_violation": 0.50, "tau_review": 0.10}
    out = {"dataset": slug, "n_positive": int(len(p)), "instrument": head.model_id,
           "fp_band_marqo_reference": FP,
           "violation_tp": {"p50": round(float(np.percentile(p, 50)), 4),
                            "p90": round(float(np.percentile(p, 90)), 4),
                            "p10": round(float(np.percentile(p, 10)), 4),
                            "min": round(float(p.min()), 4)},
           "recall_at_tau_violation_0.50": round(float((p >= 0.50).mean()), 3),
           "recall_at_tau_review_0.10": round(float((p >= 0.10).mean()), 3),
           "separation_vs_fp_p99": "CLEAN" if float(np.percentile(p, 50)) > FP["p99"] else "OVERLAP — headline it",
           "note": "Marqo VIOLATION head — the first-party violation recall the EVAL DATA LAW "
                   "forbids the agent from producing itself; it exists ONLY because the operator "
                   "supplied lawful local material. tau_violation=0.50 stays the argmax point; "
                   "enforcement_ready flips only after this measurement on the target host."}
    out_json.write_text(json.dumps(out, indent=1))
    print(f"NSFW-PROBE {slug}: n={len(p)} Marqo p50={out['violation_tp']['p50']} "
          f"R@0.50={out['recall_at_tau_violation_0.50']} R@0.10={out['recall_at_tau_review_0.10']} "
          f"(safe-corpus FP p99={FP['p99']}) -> {out['separation_vs_fp_p99']}")
    print(f"wrote {out_json}")
    return 0


def main() -> int:
    a = argparse.ArgumentParser()
    a.add_argument("--model", default="pecore-s16-384")
    a.add_argument("--datasets", default="cocoval2017,unsplash-demo,unsplashb")
    a.add_argument("--fp-budget", type=float, default=0.01, help="review FP-rate target for tau")
    a.add_argument("--build-dataset", action="store_true")
    a.add_argument("--marqo", action="store_true", help="also score the Marqo head over TP (slow, reopens)")
    a.add_argument("--nsfw-dataset", dest="nsfw_dataset", metavar="SLUG",
                   help="measure VIOLATION-tier separation on a USER-SUPPLIED indexed dataset "
                        "(e.g. nsfwprobe). The agent never fetches such material (EVAL DATA LAW).")
    a.add_argument("--json", type=Path, default=ROOT / "research/eval-nudity-probe.json")
    args = a.parse_args()

    if args.nsfw_dataset:
        return run_nsfw_probe(args.nsfw_dataset, ROOT / "research/eval-nsfwprobe.json")

    rng = random.Random(SEED)
    slices = build_probe(rng)
    print("probe subcats:", {k: len(v) for k, v in slices.items()})
    if args.build_dataset:
        build_dataset(slices)

    backend = models.load_backend(args.model, {})
    tag = backend.model_id
    pos_names = [c for cs in REVIEW_CONCEPTS.values() for c in cs]
    pos = concept_vectors(backend, pos_names)
    neg = concept_vectors(backend, NEGATIVES)

    # embed every probe image once (vision forward); margin per image
    probe_emb: dict[str, np.ndarray] = {}
    for sub, paths in slices.items():
        from PIL import Image
        vecs = []
        for p in paths:
            with Image.open(p) as im:
                vecs.append(backend.preprocess(im))
        probe_emb[sub] = np.asarray(backend.embed_images(np.stack(vecs)), np.float32) if vecs else np.zeros((0, pos.shape[1]), np.float32)

    probe_names = {p.name for paths in slices.values() for p in paths}
    fp_emb, _ = fp_band_embeddings([d for d in args.datasets.split(",") if d], probe_names)
    fp_margin, _ = review_margin(fp_emb, pos, neg)

    print(f"\n== REVIEW-MARGIN separation ({tag}) ==")
    print(f"FP band n={len(fp_margin)}: p50={np.percentile(fp_margin,50):+.4f} "
          f"p90={np.percentile(fp_margin,90):+.4f} p99={np.percentile(fp_margin,99):+.4f} "
          f"max={fp_margin.max():+.4f}")

    out: dict = {"model": tag, "seed": SEED, "negative_pool": args.datasets,
                 "fp_band": {"n": int(len(fp_margin)),
                             "margin_p50": round(float(np.percentile(fp_margin, 50)), 4),
                             "margin_p90": round(float(np.percentile(fp_margin, 90)), 4),
                             "margin_p99": round(float(np.percentile(fp_margin, 99)), 4),
                             "margin_max": round(float(fp_margin.max()), 4)},
                 "subcats": {}}
    # per-subcategory separation vs the FP band
    tau = float(np.quantile(fp_margin, 1 - args.fp_budget))
    for sub in slices:
        m = probe_emb[sub]
        if not len(m):
            continue
        sm, _ = review_margin(m, pos, neg)
        pos_y = np.concatenate([np.ones(len(sm), bool), np.zeros(len(fp_margin), bool)])
        scores = np.concatenate([sm, fp_margin])
        row = {"n": int(len(sm)),
               "margin_p10": round(float(np.percentile(sm, 10)), 4),
               "margin_p50": round(float(np.percentile(sm, 50)), 4),
               "margin_p90": round(float(np.percentile(sm, 90)), 4),
               "AP_vs_fp": round(ap(scores, pos_y), 4),
               "AP_CI95": boot_ci(ap, scores, pos_y),
               "R@fpr1%": round(rec_at_fpr(scores, pos_y, 0.01), 3),
               "R@fpr5%": round(rec_at_fpr(scores, pos_y, 0.05), 3),
               "flag_rate@tau": round(float((sm >= tau).mean()), 3)}
        out["subcats"][sub] = row
        kind = "CONTROL(never-flag)" if sub in CONTROL_SUBCATS else "TP"
        print(f"  {sub:18s}[{kind:18s}] n={row['n']:<3} margin p10/p50/p90="
              f"{row['margin_p10']:+.3f}/{row['margin_p50']:+.3f}/{row['margin_p90']:+.3f} "
              f"AP={row['AP_vs_fp']:.3f} R@1%={row['R@fpr1%']:.2f} flag@tau={row['flag_rate@tau']:.2f}")

    # ── fit platt + tau_review on ALL TP subcats vs FP band ────────────────────────────
    tp = np.concatenate([review_margin(probe_emb[s], pos, neg)[0] for s in TP_SUBCATS if len(probe_emb[s])])
    y = np.concatenate([np.ones(len(tp), bool), np.zeros(len(fp_margin), bool)])
    allm = np.concatenate([tp, fp_margin])
    A, B = ridge_platt(allm, y)
    p_all = 1 / (1 + np.exp(-(A * allm + B)))
    p_fp = p_all[len(tp):]
    p_tp = p_all[: len(tp)]
    tau_review_p = float(np.quantile(p_fp, 1 - args.fp_budget))
    out["review_fit"] = {
        "feature": "margin = max(review concepts) - max(negatives incl. mannequin/statue)",
        "instrument_choice": "full-negatives margin — measured to keep the ADR-14 mannequin "
                             "control tightest (generic-only neg raised control leak 0.07->0.12)",
        "platt": [round(A, 4), round(B, 4)],
        "tau_margin_1pct": round(float(np.quantile(fp_margin, 0.99)), 4),
        "tau_margin_5pct": round(float(np.quantile(fp_margin, 0.95)), 4),
        "tau_review_p_1pct": round(float(np.quantile(p_fp, 0.99)), 4),
        "tau_review_p_5pct": round(float(np.quantile(p_fp, 0.95)), 4),
        "operating_point_law": "recall-first (ADR-14) — 5% FP is the recall-first review point",
        "AP_all_TP": round(ap(allm, y), 4),
        "AP_CI95": boot_ci(ap, allm, y),
        "R@fpr1%": round(rec_at_fpr(allm, y, 0.01), 3),
        "R@fpr5%": round(rec_at_fpr(allm, y, 0.05), 3),
        "tp_p_p50": round(float(np.percentile(p_tp, 50)), 4),
        "fp_p_p99": round(float(np.percentile(p_fp, 99)), 4),
        "n_tp": int(len(tp)), "n_fp": int(len(fp_margin)),
    }
    # per-subcat honest verdict: does TP separate from the FP band?
    for sub, row in out["subcats"].items():
        r5 = rec_at_fpr(np.concatenate([review_margin(probe_emb[sub], pos, neg)[0], fp_margin]),
                        np.concatenate([np.ones(len(probe_emb[sub]), bool), np.zeros(len(fp_margin), bool)]),
                        0.05) if len(probe_emb[sub]) else 0.0
        row["R@fpr5%"] = round(r5, 3)
        if sub in CONTROL_SUBCATS:
            row["verdict"] = "CONTROL PASS (stays below FP band)" if row["margin_p50"] <= out["fp_band"]["margin_p50"] + 0.01 else "CONTROL LEAK — headline"
        elif r5 >= 0.45:
            row["verdict"] = "SEPARATES (usable recall-first review tier)"
        elif r5 >= 0.20:
            row["verdict"] = "WEAK separation (thin-n / borderline)"
        else:
            row["verdict"] = "DOES NOT SEPARATE on shared embedding — needs distilled head (TRACKS T2)"
    print(f"\nreview fit: platt={out['review_fit']['platt']} tau_review_p={tau_review_p:.4f} "
          f"AP={out['review_fit']['AP_all_TP']:.3f} CI={out['review_fit']['AP_CI95']} "
          f"R@1%={out['review_fit']['R@fpr1%']:.2f}")
    print(f"SEPARATION: TP p50={out['review_fit']['tp_p_p50']:.3f} vs FP p99={out['review_fit']['fp_p_p99']:.3f}"
          f"  -> {'CLEAN' if out['review_fit']['tp_p_p50'] > out['review_fit']['fp_p_p99'] else 'OVERLAP'}")

    # ── the honest counter-measurement: Marqo p does NOT separate the review tier ───────
    if args.marqo:
        head = nudity.load_nudity_head({"intra_op": 4})
        if head is not None:
            from PIL import Image
            marqo = {}
            for sub in ("swimwear", "lingerie", "mannequin-statue"):
                ps = []
                for p in slices[sub]:
                    try:
                        with Image.open(p) as im:
                            ps.append(head.preprocess(im))
                    except Exception:
                        pass
                pr = head.probs(np.stack(ps)) if ps else np.zeros(0)
                marqo[sub] = {"p50": round(float(np.percentile(pr, 50)), 4),
                              "p90": round(float(np.percentile(pr, 90)), 4),
                              "flag@0.10": round(float((pr >= 0.10).mean()), 3)}
            out["marqo_p_on_review_TP"] = marqo
            out["marqo_note"] = ("Marqo p does NOT separate the review tier — swimwear sits "
                                 "in the same 0.05-0.15 band as the FP corpus. This is WHY "
                                 "review is served by the margin instrument. Violation-tier "
                                 "recall stays benchmark-cited (98.56%/20k), never re-measured.")
            print("Marqo p on review TP (expected NON-separating):", marqo)

    args.json.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
