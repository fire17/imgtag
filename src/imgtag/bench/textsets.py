"""Text sets + offline tokenization cache.

ADR-7 keeps tokenizers out of the runtime env, so the bench tokenizes OFFLINE in the
export venv (open_clip SimpleTokenizer for PE-Core/OpenCLIP; the spike's hand-rolled
Gemma BPE for SigLIP2) and caches int64 id arrays as .npz. The bench process itself
only ever loads numpy arrays — no transformers, no tokenizers, no 34MB JSON parse.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess

import numpy as np

from . import candidates as C

CACHE = os.path.join(C.ROOT, "bench", "cache")
EXPORT_PY = os.environ.get(
    "IMGTAG_EXPORT_PY",
    os.path.expanduser("~/Creations/ImgTag/.scratch/pecore/.venv/bin/python"))

# Tokenizer family -> (backend, spec). open_clip covers CLIP-BPE and SigLIP-v1's
# sentencepiece; SigLIP2 uses Gemma BPE, read from the downloaded model dir.
TOK_SPEC = {
    "clip": ("open_clip", "hf-hub:timm/PE-Core-S-16-384"),
    "siglip1": ("open_clip", "hf-hub:timm/ViT-B-16-SigLIP"),
    "gemma": ("hf", os.path.join(C.MODELS, "siglip2-base")),
}

_TOK_CODE = r"""
import sys, json, numpy as np
backend, spec, ctx, inp, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
texts = json.load(open(inp))
if backend == "open_clip":
    import open_clip
    ids = np.asarray(open_clip.get_tokenizer(spec)(texts, context_length=ctx), dtype=np.int64)
else:
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(spec)
    ids = np.asarray(tk(texts, padding="max_length", max_length=ctx, truncation=True,
                        return_tensors="np")["input_ids"], dtype=np.int64)
np.savez_compressed(out, ids=ids)
print(ids.shape)
"""


def _sha8(texts: list[str]) -> str:
    h = hashlib.sha256("\x00".join(texts).encode()).hexdigest()
    return h[:8]


def tokenize(texts: list[str], family: str, ctx: int) -> np.ndarray:
    """Cached offline tokenization. Raises RuntimeError for unsupported families."""
    os.makedirs(CACHE, exist_ok=True)
    dst = os.path.join(CACHE, f"tokens-{family}-{ctx}-{_sha8(texts)}.npz")
    if os.path.exists(dst):
        return np.load(dst)["ids"]
    entry = TOK_SPEC.get(family)
    if not entry:
        raise RuntimeError(f"tokenizer family '{family}' not wired (BLOCKED, not faked)")
    backend, spec = entry
    tmp = dst + ".texts.json"
    with open(tmp, "w") as f:
        json.dump(texts, f)
    p = subprocess.run([EXPORT_PY, "-c", _TOK_CODE, backend, spec, str(ctx), tmp, dst],
                       capture_output=True, text=True, timeout=1800)
    os.unlink(tmp)
    if p.returncode:
        raise RuntimeError(f"tokenize failed: {(p.stderr or p.stdout)[-300:]}")
    return np.load(dst)["ids"]


# ── prompt construction ───────────────────────────────────────────────────────
# ONE prompt template across every candidate — a prompt ensemble would be a per-model
# tunable and this bench compares models, not prompts. The shipped engine may ensemble
# (ADR-3 prompt_ensemble_sha); that is a later, separately-measured win.
def prompt(name: str) -> str:
    n = name.replace("_", " ").strip()
    article = "an" if n[:1].lower() in "aeiou" else "a"
    return f"a photo of {article} {n}."
