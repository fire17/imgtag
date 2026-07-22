#!/usr/bin/env python3
"""Weapons track — TRUE-POSITIVE vs FALSE-POSITIVE-BAND confidence separation.

Answers the user's directive (2026-07-22 13:58Z) with numbers: does the shipped weapons
head score REAL weapon images (the `weaponprobe` gallery, held-out OI validation, per
subcategory) at HIGHER confidence than the current false-positive band in the user's own
datasets (unsplash-demo, unsplashb, cocoval2017)? If yes, a per-tier "ratio threshold for
auto flagging" can be set between the two distributions. If any subcategory's true
positives do NOT dominate the FP band, that is a headline finding — printed, never buried.

Everything is a matmul over embeddings that already exist (the probe's own snapshot + the
user datasets' snapshots). No re-embedding, no agents, re-runnable (TRACKS.md T4).

    .venv/bin/python scripts/eval_weapons.py [--fp-datasets a,b,c] [--write-json]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from imgtag.core import store                       # noqa: E402
from imgtag.moderation import weapons as W          # noqa: E402

PROBE = ROOT / "data" / "weapon-probe"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval — small-n honest (0/15890 is '<0.02%', not 'zero')."""
    if n == 0:
        return (0.0, 1.0)
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * ((ph * (1 - ph) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def ap(p: np.ndarray, y: np.ndarray) -> float:
    """Average precision (ranking quality, prevalence-independent)."""
    o = np.argsort(-np.asarray(p))
    ys = np.asarray(y, bool)[o]
    tp = np.cumsum(ys)
    prec = tp / np.arange(1, len(ys) + 1)
    return float((prec * ys).sum() / max(ys.sum(), 1))


def dist(p: np.ndarray) -> dict:
    p = np.asarray(p, float)
    q = lambda x: round(float(np.quantile(p, x)), 4)      # noqa: E731
    return {"n": int(len(p)), "min": round(float(p.min()), 4), "p05": q(.05),
            "p25": q(.25), "median": q(.5), "p75": q(.75), "p90": q(.90),
            "p99": q(.99), "p999": q(.999), "max": round(float(p.max()), 4)}


def main() -> int:
    a = argparse.ArgumentParser()
    a.add_argument("--fp-datasets", default="unsplash-demo,unsplashb,cocoval2017",
                   help="indexed USER datasets forming the current false-positive band")
    a.add_argument("--probe", default="weaponprobe", help="indexed TP probe dataset")
    a.add_argument("--backend", default="pecore-s16-384")
    a.add_argument("--fp-budget-v", type=float, default=0.001,
                   help="violation: fraction of the FP band allowed above tau_v")
    a.add_argument("--fp-budget-r", type=float, default=0.01,
                   help="review: fraction of the FP band allowed above tau_r")
    a.add_argument("--write-json", action="store_true")
    args = a.parse_args()

    head = W.load_head(args.backend)
    if head is None:
        print(f"no weapons head for {args.backend}", file=sys.stderr)
        return 2
    print(f"head {args.backend} dim={head.dim} shipped tau_v={head.tau_violation:.4f} "
          f"tau_r={head.tau_review:.4f} AP(OI-val)={head.metrics.get('ap'):.4f}")

    # ── TRUE POSITIVES: the weaponprobe snapshot (what the gallery actually shows) ──
    tax = json.loads((PROBE / "taxonomy.json").read_bytes())
    img_subs = tax["image_subcategories"]              # "id.jpg" -> [subcat,...]
    snap = store.open_snapshot(args.probe)
    tp_emb = np.asarray(snap.emb, np.float32)
    tp_names = [Path(r["path"]).name for r in snap.ids]
    p_tp = head.probs(tp_emb)
    name2p = dict(zip(tp_names, p_tp))
    print(f"probe {args.probe}: {len(tp_names)} images scored")

    # per-subcategory TP probability vectors
    sub_p: dict[str, list[float]] = defaultdict(list)
    for n, pv in zip(tp_names, p_tp):
        for s in img_subs.get(n, []):
            sub_p[s].append(float(pv))

    # ── FALSE-POSITIVE BAND: user datasets, deduped by filename ──
    fp_p, seen = [], set()
    per_ds = {}
    for ds in [d for d in args.fp_datasets.split(",") if d]:
        try:
            s = store.open_snapshot(ds)
        except Exception as e:
            print(f"  (skip {ds}: {type(e).__name__})")
            continue
        E = np.asarray(s.emb, np.float32)
        keep = [i for i, r in enumerate(s.ids)
                if Path(r["path"]).name not in seen and not seen.add(Path(r["path"]).name)]
        pv = head.probs(E[keep])
        fp_p.append(pv)
        per_ds[ds] = {"n": len(keep),
                      "ge_shipped_tau_v": int((pv >= head.tau_violation).sum()),
                      "ge_shipped_tau_r": int((pv >= head.tau_review).sum())}
        print(f"  FP band {ds}: {len(keep)} new rows")
    fp_p = np.concatenate(fp_p)
    fp = dist(fp_p)

    # ── proposed ratio thresholds: set tau between the FP band and the TP mass ──
    tau_v_prop = float(np.quantile(fp_p, 1 - args.fp_budget_v))
    tau_r_prop = float(np.quantile(fp_p, 1 - args.fp_budget_r))
    kv, kr = int((fp_p >= tau_v_prop).sum()), int((fp_p >= tau_r_prop).sum())
    lo_v, hi_v = wilson(kv, len(fp_p))
    lo_r, hi_r = wilson(kr, len(fp_p))

    out: dict = {
        "backend": args.backend,
        "shipped": {"tau_violation": round(head.tau_violation, 4),
                    "tau_review": round(head.tau_review, 4),
                    "ap_oi_val": round(float(head.metrics.get("ap", 0)), 4)},
        "fp_band": {"datasets": args.fp_datasets, "per_dataset": per_ds,
                    "n_total": int(len(fp_p)), "distribution": fp},
        "tp_overall": dist(p_tp),
        "separation_overall_ap": round(ap(np.concatenate([p_tp, fp_p]),
                                          np.r_[np.ones(len(p_tp)), np.zeros(len(fp_p))]), 4),
        "proposed": {
            "tau_violation": round(tau_v_prop, 4),
            "tau_violation_fp_rate": f"{kv}/{len(fp_p)}",
            "tau_violation_fp_ci95": [round(lo_v, 5), round(hi_v, 5)],
            "tau_review": round(tau_r_prop, 4),
            "tau_review_fp_rate": f"{kr}/{len(fp_p)}",
            "tau_review_fp_ci95": [round(lo_r, 5), round(hi_r, 5)],
            "fp_budget_v": args.fp_budget_v, "fp_budget_r": args.fp_budget_r,
        },
        "subcategories": {},
        "gaps": tax.get("gaps", []),
    }

    # ── per-subcategory separation ──
    for sub in tax["subcategories"]:
        ps = np.array(sub_p.get(sub, []), float)
        if len(ps) == 0:
            out["subcategories"][sub] = {"n": 0, "status": "NO PROBE IMAGERY (gap)"}
            continue
        y = np.r_[np.ones(len(ps)), np.zeros(len(fp_p))]
        pall = np.r_[ps, fp_p]
        dominates = float(np.quantile(ps, .05)) > fp["p99"]      # TP p05 above FP p99
        out["subcategories"][sub] = {
            "n": int(len(ps)),
            "tp_distribution": dist(ps),
            "ap_vs_fp_band": round(ap(pall, y), 4),
            "recall_at_shipped_tau_v": round(float((ps >= head.tau_violation).mean()), 3),
            "recall_at_shipped_tau_r": round(float((ps >= head.tau_review).mean()), 3),
            "recall_at_proposed_tau_v": round(float((ps >= tau_v_prop).mean()), 3),
            "tp_p05_over_fp_p99": dominates,
            "separation_margin_median_minus_fp_p99": round(float(np.median(ps)) - fp["p99"], 4),
        }

    # ── print: the two distributions, the proposal, the per-subcat table ──
    print(f"\nFP BAND (current 'everything scored now'), n={len(fp_p)}:")
    print(f"  median={fp['median']:.4f}  p90={fp['p90']:.4f}  p99={fp['p99']:.4f}  "
          f"p99.9={fp['p999']:.4f}  max={fp['max']:.4f}")
    print(f"  >= shipped tau_v(0.811): {int((fp_p>=head.tau_violation).sum())}/{len(fp_p)}   "
          f">= shipped tau_r(0.087): {int((fp_p>=head.tau_review).sum())}/{len(fp_p)}")
    tp = out["tp_overall"]
    print(f"\nTRUE POSITIVES (weaponprobe), n={tp['n']}:")
    print(f"  min={tp['min']:.4f}  p05={tp['p05']:.4f}  median={tp['median']:.4f}  "
          f"max={tp['max']:.4f}")
    print(f"  overall TP-vs-FP separation AP = {out['separation_overall_ap']:.4f}")

    print(f"\nPROPOSED RATIO THRESHOLDS (FP band budget v={args.fp_budget_v} r={args.fp_budget_r}):")
    print(f"  tau_violation = {tau_v_prop:.4f}   FP {kv}/{len(fp_p)} [{lo_v:.5f}-{hi_v:.5f}]"
          f"   (shipped {head.tau_violation:.4f})")
    print(f"  tau_review    = {tau_r_prop:.4f}   FP {kr}/{len(fp_p)} [{lo_r:.5f}-{hi_r:.5f}]"
          f"   (shipped {head.tau_review:.4f})")

    print("\nPER-SUBCATEGORY SEPARATION:")
    print("| subcategory | n | TP median | TP p05 | AP vs FP | R@τv(.811) | R@τv_prop | dominates FP band |")
    print("|---|---|---|---|---|---|---|---|")
    fails = []
    for sub, d in out["subcategories"].items():
        if d.get("n", 0) == 0:
            print(f"| {sub} | 0 | — | — | — | — | — | ⚠ GAP (no imagery) |")
            continue
        td = d["tp_distribution"]
        dom = "✅ yes" if d["tp_p05_over_fp_p99"] else "❌ NO"
        if not d["tp_p05_over_fp_p99"]:
            fails.append(sub)
        print(f"| {sub} | {d['n']} | {td['median']:.3f} | {td['p05']:.3f} | "
              f"{d['ap_vs_fp_band']:.3f} | {d['recall_at_shipped_tau_v']:.2f} | "
              f"{d['recall_at_proposed_tau_v']:.2f} | {dom} |")

    if fails:
        print(f"\n⚠ HEADLINE — TP p05 does NOT clear the FP-band p99 for: {', '.join(fails)}")
        print("  (still may separate at median; see AP-vs-FP and the recall columns).")
    if out["gaps"]:
        print("⚠ GAPS — subcategories with no held-out TP imagery (cannot be tested here):")
        for g in out["gaps"]:
            print(f"    {g['subcategory']} {g['oi_classes']}: {g['reason']}")

    if args.write_json:
        (ROOT / "research" / "eval-weapons.json").write_text(json.dumps(out, indent=1))
        print(f"\nwrote research/eval-weapons.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
