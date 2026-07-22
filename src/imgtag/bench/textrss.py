"""Isolated text-tower + tokenizer resident-RSS probe (ADR-5 revision, feeds B8).

Fresh process per measurement: baseline RSS → +text ORT session → RSS delta = the text
tower's resident cost. Contention-immune (no timing), so it needs no quiet window.

    python -m imgtag.bench.textrss <candidate_id> [precision]  -> one JSON line
    run_all() drives every candidate.

The tokenizer number answers spike-siglip2's flag (tokenizer.json JSON-parse ~551MB); the
shipped binary tokenizer (b-engine) should be re-probed here once it lands.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

from . import candidates as C


def _probe(cand_id: str, prec: str) -> dict:
    c = C.CANDIDATES[cand_id]
    tpath = c.text.get(prec) or c.text.get("int8") or c.text.get("fp32")
    if not tpath or not os.path.exists(tpath):
        return {"error": f"no text tower for {cand_id}/{prec}"}
    base = C.peak_rss_mb()
    sess = C.session(tpath, 2)
    loaded = C.peak_rss_mb()
    # run once so lazy allocations (arena) materialize into the resident number.
    ids = np.zeros((1, c.ctx), np.int64)
    sess.run(None, {"input_ids": ids})
    ran = C.peak_rss_mb()
    return {
        "candidate": cand_id, "precision_text": prec,
        "text_artifact_mb": round(os.path.getsize(tpath) / 1e6, 1),
        "rss_base_mb": round(base, 1),
        "text_resident_rss_mb": round(ran - base, 1),
        "rss_after_load_mb": round(loaded, 1),
        "rss_after_run_mb": round(ran, 1),
    }


def run_all(log=print) -> list[dict]:
    import subprocess

    out = []
    for cid, c in C.CANDIDATES.items():
        prec = "int8" if c.text.get("int8") else "fp32"
        if not (c.text.get(prec) and os.path.exists(c.text[prec])):
            continue
        p = subprocess.run([sys.executable, "-m", "imgtag.bench.textrss", cid, prec],
                           capture_output=True, text=True, timeout=300,
                           env={**os.environ, "PYTHONWARNINGS": "ignore"})
        try:
            r = json.loads(p.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            r = {"candidate": cid, "error": (p.stderr or "no output")[-200:]}
        out.append(r)
        log(f"  {cid}/{r.get('precision_text', '?')}: text resident "
            f"{r.get('text_resident_rss_mb', '—')}MB "
            f"(artifact {r.get('text_artifact_mb', '—')}MB)")
    return out


if __name__ == "__main__":
    print(json.dumps(_probe(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "int8")))
