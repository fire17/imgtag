"""Weight-only dynamic int8 — the ADR-4 recipe, verbatim, and the B24 fidelity gate.

ADR-4 (runtime lane, measured): "SELF-quantized dynamic weight-only int8, MatMul-only,
`MatMulConstBOnly`, QUInt8, per-tensor". Naive full-graph quantize_dynamic scored cos 0.94
on the PE-Core vision tower (spike-pecore §finding 2) = FAILS B24. This module is the
KEY EXPERIMENT: does the weight-only recipe recover it?

FIDELITY GATE (ADR-4 + B24): mean cos >= 0.995 (ADR-4 CI floor 0.98), min cos >= 0.97,
top-1 NN ranking agreement >= 0.90. Ranking agreement is the metric that matters — mean
cosine hides rank flips.
"""
from __future__ import annotations

import os
import time

import numpy as np

RECIPE = dict(op_types_to_quantize=["MatMul"], per_channel=False,
              extra_options={"MatMulConstBOnly": True})


def quantize_weight_only(src: str, dst: str, force: bool = False) -> dict:
    """ADR-4 recipe. Returns {ok, seconds, mb, error?}. Idempotent unless force."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    if os.path.exists(dst) and not force:
        return {"ok": True, "cached": True, "mb": os.path.getsize(dst) / 1e6, "seconds": 0.0}
    t = time.perf_counter()
    try:
        quantize_dynamic(src, dst, weight_type=QuantType.QUInt8, **RECIPE)
    except Exception as e:  # noqa: BLE001 — a failed candidate is a row, not a crash
        return {"ok": False, "error": f"{type(e).__name__}: {e}"[:300]}
    return {"ok": True, "cached": False, "seconds": round(time.perf_counter() - t, 1),
            "mb": round(os.path.getsize(dst) / 1e6, 1)}


# ── B24 gate ──────────────────────────────────────────────────────────────────
GATE = {"mean_cos": 0.995, "min_cos": 0.97, "nn_agree": 0.90}


def fidelity(ref: np.ndarray, cand: np.ndarray) -> dict:
    """Per-image cosine + top-1 NN ranking agreement between two L2-normed embed sets."""
    cos = (ref * cand).sum(1)
    Sr, Sc = ref @ ref.T, cand @ cand.T
    np.fill_diagonal(Sr, -9.0)
    np.fill_diagonal(Sc, -9.0)
    nn_r, nn_c = np.argmax(Sr, 1), np.argmax(Sc, 1)
    top3_r, top3_c = np.argsort(-Sr, 1)[:, :3], np.argsort(-Sc, 1)[:, :3]
    ov = float(np.mean([len(set(a) & set(b)) / 3 for a, b in zip(top3_r, top3_c)]))
    r = {
        "n": int(len(cos)),
        "mean_cos": float(cos.mean()),
        "min_cos": float(cos.min()),
        "p1_cos": float(np.percentile(cos, 1)),
        "nn_agree": float(np.mean(nn_r == nn_c)),
        "top3_overlap": ov,
    }
    r["pass"] = bool(r["mean_cos"] >= GATE["mean_cos"] and r["min_cos"] >= GATE["min_cos"]
                     and r["nn_agree"] >= GATE["nn_agree"])
    r["pass_ci_floor"] = bool(r["mean_cos"] >= 0.98 and r["nn_agree"] >= 0.90)
    return r
