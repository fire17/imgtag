#!/usr/bin/env python
"""Alert-tier separation on the indexed `safetyprobe` dataset (USER LAW 13:58Z).

The COCO eval (scripts/eval_safety.py) could not measure the ALERT tier — COCO val2017
has exactly 1 person-down-in-danger image. This does, on the hand-fetched Unsplash probe:
57 `alert_tp` (person-down IN danger) vs the benign-lying band (`person_down` +
`sunbathing`). The question the user's law asks: does the alert-TP confidence measurably
DOMINATE the benign-FP band, per subcategory? If it does not, the alert tier stays
withheld — a valid honest result, not a failure to hide.

WHAT SEPARATES WHAT. All of {alert_tp, person_down, sunbathing} contain a lying body, so
p_lying does NOT tell them apart — p_DANGER does. So the alert-tier fit is a fit of the
DANGER gate: at what τ_danger does alert_tp sit above the benign-lying band? τ_review (the
person-down gate) is cross-checked here on the 183 real lying images, but its FALSE-
positive rate stays measured on COCO (the probe has no clean upright/empty negatives).

Labels are WEAK (fetch-keyword-derived, T4 — a "first aid" tag can be a kit not a victim).
Every number is directional, reported with a bootstrap CI and the weak-label caveat, and
never promoted to enforcement.

    uv run python scripts/eval_safety_separation.py [--dataset safetyprobe] [--json OUT]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imgtag.core import models, store  # noqa: E402
from imgtag.moderation import safety  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "data/safety-probe"

BENIGN_LYING = ("person_down", "sunbathing")   # the band alert_tp must beat
DANGER_NOBODY = ("injury_context", "danger_context", "destruction")  # danger, no body down


def load_labels() -> dict[str, str]:
    d = json.loads((PROBE / "labels.json").read_bytes())
    out = {}
    for k, v in d.items():
        if k.startswith("_"):
            continue
        for name in v:
            out[name] = k
    return out


def ap(score: np.ndarray, y: np.ndarray) -> float:
    o = np.argsort(-score)
    ys = np.asarray(y, bool)[o]
    tp = np.cumsum(ys)
    return float(((tp / np.arange(1, len(ys) + 1)) * ys).sum() / max(ys.sum(), 1))


def boot_ci(score: np.ndarray, y: np.ndarray, stat, n=1000, seed=0) -> tuple[float, float]:
    """Percentile bootstrap CI. seed fixed — Math.random is unavailable and flaky anyway."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    vals = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if 0 < y[b].sum() < len(b):
            vals.append(stat(score[b], y[b]))
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def recall_at_precision(score, y, target_prec):
    """Highest recall whose precision >= target (precision-first operating point)."""
    o = np.argsort(-score)
    ys = np.asarray(y, bool)[o]
    tp = np.cumsum(ys)
    prec = tp / np.arange(1, len(ys) + 1)
    rec = tp / max(ys.sum(), 1)
    ok = prec >= target_prec
    if not ok.any():
        return 0.0, float("inf")
    i = np.where(ok)[0][-1]                       # last index still meeting precision
    thr = float(score[o][i])
    return float(rec[i]), thr


def main() -> int:
    a = argparse.ArgumentParser()
    a.add_argument("--dataset", default="safetyprobe")
    a.add_argument("--model", default="pecore-s16-384")
    a.add_argument("--json", default="")
    args = a.parse_args()

    labels = load_labels()
    backend = models.load_backend(args.model, {}, vision=False)
    snap = store.open_snapshot(args.dataset)
    emb = np.asarray(snap.emb, np.float32)
    names = [os.path.basename(r["path"]) for r in snap.ids]
    sub = np.array([labels.get(n, "unlabelled") for n in names])
    print(f"{args.dataset}: {len(emb)} indexed · subcategories: "
          + ", ".join(f"{k}={int((sub == k).sum())}" for k in sorted(set(sub))))

    sc = safety.SafetyScorer.build(backend)
    pl = safety.lying_prob(sc.lying_margin(emb))
    pd = safety.danger_prob(sc.danger_margin(emb))
    tier = sc.tiers(pl, pd)

    out: dict = {"dataset": args.dataset, "model": backend.model_id, "n": len(emb),
                 "per_subcategory": {}, "shipped_tau": {"review": sc.pol["tau"],
                                                        "danger": sc.pol["tau_danger"]}}

    # ── per-subcategory distributions + tier assignment (the eyeball table) ──────────────
    print("\nsubcategory            n   p_lying(med)  p_danger(med)   alert  review  none")
    for k in ("alert_tp", "person_down", "sunbathing", "injury_context",
              "danger_context", "destruction", "unlabelled"):
        m = sub == k
        if not m.any():
            continue
        row = {"n": int(m.sum()),
               "p_lying_med": round(float(np.median(pl[m])), 4),
               "p_danger_med": round(float(np.median(pd[m])), 4),
               "alert": int((tier[m] == "alert").sum()),
               "review": int((tier[m] == "review").sum()),
               "none": int((tier[m] == "none").sum())}
        out["per_subcategory"][k] = row
        print(f"{k:20s} {row['n']:4d}   {row['p_lying_med']:.4f}       "
              f"{row['p_danger_med']:.4f}       {row['alert']:4d}   {row['review']:4d}  {row['none']:4d}")

    # ── SEPARATION 1: alert_tp vs benign-lying band, on p_danger (the alert discriminator)
    alert = sub == "alert_tp"
    benign = np.isin(sub, BENIGN_LYING)
    sel = alert | benign
    if alert.sum() and benign.sum():
        ap_d = ap(pd[sel], alert[sel])
        ci = boot_ci(pd[sel], alert[sel], ap)
        rec90, thr90 = recall_at_precision(pd[sel], alert[sel], 0.90)
        rec80, thr80 = recall_at_precision(pd[sel], alert[sel], 0.80)
        out["separation_alert_vs_benign_lying"] = {
            "discriminator": "p_danger", "n_alert": int(alert.sum()),
            "n_benign_lying": int(benign.sum()),
            "AP": round(ap_d, 4), "AP_CI95": [round(ci[0], 4), round(ci[1], 4)],
            "alert_p_danger_med": round(float(np.median(pd[alert])), 4),
            "benign_p_danger_med": round(float(np.median(pd[benign])), 4),
            "recall_at_prec90": round(rec90, 4), "tau_danger_at_prec90": round(thr90, 5),
            "recall_at_prec80": round(rec80, 4), "tau_danger_at_prec80": round(thr80, 5),
        }
        s = out["separation_alert_vs_benign_lying"]
        print(f"\nSEPARATION (alert_tp vs benign-lying, discriminator=p_danger):")
        print(f"  AP {s['AP']:.3f}  CI95 [{s['AP_CI95'][0]:.3f}, {s['AP_CI95'][1]:.3f}]  "
              f"(alert med {s['alert_p_danger_med']:.3f} vs benign med {s['benign_p_danger_med']:.3f})")
        print(f"  PRECISION-FIRST tau_danger: @prec0.90 recall {rec90:.3f} tau {thr90:.4f} · "
              f"@prec0.80 recall {rec80:.3f} tau {thr80:.4f}")

    # ── SEPARATION 2: person-down detection — real lying (probe) vs danger-no-body ──────
    # cross-corpus recall check for tau_review; FP rate stays on COCO (no clean neg here)
    lying_any = np.isin(sub, ("alert_tp", "person_down", "sunbathing"))
    nobody = np.isin(sub, DANGER_NOBODY)
    if lying_any.sum() and nobody.sum():
        selL = lying_any | nobody
        ap_l = ap(pl[selL], lying_any[selL])
        ciL = boot_ci(pl[selL], lying_any[selL], ap)
        out["separation_lying_vs_nobody"] = {
            "discriminator": "p_lying", "n_lying": int(lying_any.sum()),
            "n_danger_nobody": int(nobody.sum()),
            "AP": round(ap_l, 4), "AP_CI95": [round(ciL[0], 4), round(ciL[1], 4)],
            "lying_recall_at_shipped_tau_review": round(float((pl[lying_any] >= sc.pol["tau"]).mean()), 4),
            "note": "cross-corpus recall check for tau_review; the review-tier FP rate is "
                    "measured on COCO (scripts/eval_safety.py), not here — the probe has no "
                    "clean upright/empty negatives.",
        }
        s = out["separation_lying_vs_nobody"]
        print(f"\nPERSON-DOWN (lying vs danger-no-body, discriminator=p_lying):")
        print(f"  AP {s['AP']:.3f}  CI95 [{s['AP_CI95'][0]:.3f}, {s['AP_CI95'][1]:.3f}]  "
              f"lying recall @shipped tau_review {s['lying_recall_at_shipped_tau_review']:.3f}")

    # ── the boots/puddle question: are benign-lying images reaching alert? ──────────────
    benign_alert = int((tier[benign] == "alert").sum())
    out["benign_lying_false_alerts"] = {"n_benign_lying": int(benign.sum()),
                                        "reached_alert": benign_alert,
                                        "rate": round(benign_alert / max(int(benign.sum()), 1), 4)}
    print(f"\nBENIGN-LYING FALSE ALERTS at shipped tau: {benign_alert}/{int(benign.sum())} "
          f"({out['benign_lying_false_alerts']['rate']:.1%})")

    # ── verdict ─────────────────────────────────────────────────────────────────────────
    ap_d = out.get("separation_alert_vs_benign_lying", {}).get("AP", 0.0)
    ci_lo = out.get("separation_alert_vs_benign_lying", {}).get("AP_CI95", [0])[0]
    verdict = ("SHIP" if ci_lo > 0.5 else "WITHHOLD")
    out["verdict"] = {"decision": verdict,
                      "rule": "ship the alert tier only if alert_tp dominates benign-lying "
                              "(AP CI95 lower bound > 0.5 = better than a coin flip on the "
                              "person-down subset)",
                      "ap_ci95_lower": ci_lo}
    print(f"\nVERDICT: {verdict}  (alert-vs-benign AP CI95 lower bound = {ci_lo:.3f}; "
          f"rule: >0.5 to ship)")

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
