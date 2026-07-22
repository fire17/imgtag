#!/usr/bin/env python3
"""track-nudity FALSE-POSITIVE-side evaluation + threshold fit.

EVAL DATA LAW: no explicit-adult corpus is downloaded to this machine. Everything below
measures the FP side only, on the safe corpora already on disk, and the report labels the
TP side as "published, not reproduced here".

Slices are built by joining the Unsplash Lite keywords.tsv000 to the images actually
downloaded, plus a COCO val2017 sample. The "hard" slices (bikini/swimwear/lingerie/
underwear/beach/portrait/yoga/dance/sculpture) are where a nudity detector is most likely
to fire wrongly; the "easy" slices (landscape/coco) establish the floor.

    python3 research/bench_scripts/nudity_eval.py            # ONNX head
    python3 research/bench_scripts/nudity_eval.py --zeroshot # + CLIP prompt baseline
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

HARD = ["bikini", "swimwear", "lingerie", "underwear", "yoga", "dance", "sculpture",
        "portrait", "beach", "child", "baby", "shower", "bath", "tattoo", "muscle"]
#: ADR-14's REQUIRED negative: non-person figures must never flag.
FIGURES = ["mannequin", "statue", "doll", "figurine", "marble", "torso"]
EASY = ["landscape", "architecture", "food", "car"]
CAP = 200  # per-slice cap keeps the sweep honest about slice imbalance and quick


def unsplash_index() -> dict[str, Path]:
    out: dict[str, Path] = {}
    for d in (ROOT / "data/unsplash/images", ROOT / "data/unsplash-b"):
        if d.is_dir():
            for f in d.iterdir():
                if f.suffix.lower() == ".jpg":
                    out.setdefault(f.stem, f)
    return out


def build_slices(rng: random.Random) -> dict[str, list[Path]]:
    have = unsplash_index()
    want = set(HARD) | set(EASY) | set(FIGURES)
    kw: dict[str, list[str]] = defaultdict(list)
    with open(ROOT / "data/unsplash/keywords.tsv000", encoding="utf8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row["keyword"] in want and row["photo_id"] in have:
                kw[row["keyword"]].append(row["photo_id"])
    slices: dict[str, list[Path]] = {}
    for k, ids in kw.items():
        ids = sorted(set(ids))
        rng.shuffle(ids)
        if len(ids) >= 20:
            slices[k] = [have[i] for i in ids[:CAP]]
    coco = sorted((ROOT / "data/coco/val2017").glob("*.jpg")) or sorted((ROOT / "data/coco").rglob("*.jpg"))
    if coco:
        rng.shuffle(coco)
        slices["coco-val2017"] = coco[:500]
    return dict(sorted(slices.items()))


def run_onnx(slices: dict[str, list[Path]], intra: int) -> tuple[dict[str, np.ndarray], float]:
    from imgtag.moderation.nudity import load_nudity_head

    head = load_nudity_head({"intra_op": intra})
    if head is None:
        sys.exit("nudity artifact missing — run scripts/export_nudity_marqo.py first")
    print(f"head {head.model_id} sha {head.model_sha[:12]} tau_v {head.tau_violation} "
          f"tau_r {head.tau_review} intra_op {intra}")
    scores, t_total, n_total = {}, 0.0, 0
    for name, paths in slices.items():
        vals, t0 = [], time.perf_counter()
        for i in range(0, len(paths), 8):
            batch = []
            for p in paths[i : i + 8]:
                try:
                    with Image.open(p) as im:
                        batch.append(head.preprocess(im))
                except Exception as e:  # poison-corpus behaviour is b-engine's problem
                    print(f"  skip {p.name}: {type(e).__name__}", file=sys.stderr)
            if batch:
                vals.append(head.probs(np.stack(batch)))
        dt = time.perf_counter() - t0
        t_total += dt
        n_total += len(paths)
        scores[name] = np.concatenate(vals) if vals else np.zeros(0, np.float32)
        print(f"  {name:<16} n={len(scores[name]):<4} mean={scores[name].mean():.4f} "
              f"p95={np.percentile(scores[name], 95):.4f} max={scores[name].max():.4f} "
              f"({1000*dt/max(1,len(paths)):.0f} ms/img incl. decode)")
    return scores, t_total / max(1, n_total)


def run_zeroshot(slices: dict[str, list[Path]], backend_name: str) -> dict[str, np.ndarray]:
    from imgtag.core.models import load_backend
    from imgtag.moderation.nudity import ZeroShotNudityHead

    be = load_backend(backend_name, {"intra_op": 4})
    zs = ZeroShotNudityHead(be)
    out = {}
    for name, paths in slices.items():
        vals = []
        for i in range(0, len(paths), 8):
            batch = []
            for p in paths[i : i + 8]:
                try:
                    with Image.open(p) as im:
                        batch.append(be.preprocess(im))
                except Exception:
                    pass
            if batch:
                vals.append(zs.margins(be.embed_images(np.stack(batch))))
        out[name] = np.concatenate(vals) if vals else np.zeros(0, np.float32)
        print(f"  [zs] {name:<16} n={len(out[name]):<4} mean={out[name].mean():+.4f} "
              f"p95={np.percentile(out[name], 95):+.4f} max={out[name].max():+.4f}")
    return out


def negative_control(intra: int) -> int:
    """PERMANENT negative control (team-lead ruling 2026-07-22): content-free input must
    never reach the violation tier, and solid colours/gradients must not flag at all."""
    from imgtag.moderation.nudity import load_nudity_head, structure

    h = load_nudity_head({"intra_op": intra})
    rng = np.random.default_rng(0)
    g = np.tile(np.linspace(0, 255, 384, dtype=np.uint8)[None, :, None], (384, 1, 3))
    probes = {
        "solid-black": np.zeros((384, 384, 3), np.uint8),
        "solid-white": np.full((384, 384, 3), 255, np.uint8),
        "solid-flesh": np.full((384, 384, 3), (222, 180, 150), np.uint8),
        "solid-worst": np.full((384, 384, 3), (255, 192, 255), np.uint8),
        "gradient": g,
        "gradient-flesh": np.clip(g * 0.6 + np.array([120, 70, 50]), 0, 255).astype(np.uint8),
        "gradient-radial": (np.hypot(*(np.indices((384, 384)) - 192)) / 272 * 255).astype(np.uint8)[:, :, None].repeat(3, 2),
        "noise-uniform": rng.integers(0, 256, (384, 384, 3), dtype=np.uint8),
        "noise-gauss": np.clip(rng.normal(128, 30, (384, 384, 3)), 0, 255).astype(np.uint8),
    }
    batch = np.stack(list(probes.values()))
    flags, st, bad = h._flags(batch), structure(batch), 0
    print("\nNEGATIVE CONTROL (content-free probes)")
    for name, f, s_ in zip(probes, flags, st):
        note = " content_free" if f.get("content_free") else ""
        print(f"  {name:16s} raw p={f['p']:.4f} structure={s_:7.3f} -> {f['tier']}{note}")
        if f["tier"] == "violation":
            bad += 1
    print(f"  violation-tier leaks: {bad} (must be 0)")
    return bad


def sweep(scores: dict[str, np.ndarray], taus: list[float]) -> None:
    allv = np.concatenate([v for v in scores.values() if len(v)])
    hard = np.concatenate([v for k, v in scores.items() if k in HARD and len(v)])
    print(f"\nFLAG RATE vs threshold (safe corpus only — every flag here is a FALSE POSITIVE\n"
          f"except in the swimwear/lingerie slices, where a strict site rule may call some real):")
    print(f"{'tau':>6} {'all':>8} {'hard':>8}  " + "  ".join(f"{k[:9]:>9}" for k in scores))
    figs = [v for k, v in scores.items() if k in FIGURES and len(v)]
    figs = np.concatenate(figs) if figs else np.zeros(0, np.float32)
    for t in taus:
        row = f"{t:>6.2f} {100*(allv>=t).mean():>7.2f}% {100*(hard>=t).mean():>7.2f}%  "
        row += "  ".join(f"{100*(v>=t).mean():>8.2f}%" for v in scores.values())
        print(row)
    if len(figs):  # ADR-14's required negative, called out on its own line
        print("\nnon-person figures (mannequin/statue/doll/figurine/marble/torso), n=%d:" % len(figs)
              + "  " + "  ".join(f"tau {t}: {100*(figs>=t).mean():.2f}%" for t in (0.10, 0.30, 0.50)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zeroshot", action="store_true")
    ap.add_argument("--skip-onnx", action="store_true")
    ap.add_argument("--backend", default="openclip-vitb32")
    ap.add_argument("--intra", type=int, default=int(os.environ.get("INTRA", 4)))
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()

    rng = random.Random(20260722)
    slices = build_slices(rng)
    print("slices:", {k: len(v) for k, v in slices.items()})
    if not a.skip_onnx:
        scores, per_img = run_onnx(slices, a.intra)
        print(f"\nmean {1000*per_img:.1f} ms/img (decode+preprocess+forward, intra_op={a.intra}, "
              f"load {os.getloadavg()[0]:.1f})")
        sweep(scores, [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 0.90])
    else:
        scores = {}
    if a.zeroshot:
        zs = run_zeroshot(slices, a.backend)
        sweep(zs, [0.0, 0.01, 0.02, 0.03, 0.05, 0.08])
    if not a.skip_onnx and negative_control(a.intra):
        print("NEGATIVE CONTROL FAILED", file=sys.stderr)
        return 1
    if a.json:
        a.json.write_text(json.dumps({k: v.tolist() for k, v in scores.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
