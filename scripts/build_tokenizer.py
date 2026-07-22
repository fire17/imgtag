#!/usr/bin/env python3
"""OFFLINE build of the compact tokenizer binaries imgtag ships (HARD REQUIREMENT).

Measured by the tokenizer spike: a 34MB ``tokenizer.json`` costs **0.61s cold and 551MB
resident** (16× the file) once Python turns it into dicts of str. The binary below is
~11MB resident and loads in ~0, because nothing becomes a Python object:

    vocab_blob   uint8   concatenated UTF-8 token bytes
    vocab_off    int32   [n+1] offsets into vocab_blob (token i = blob[off[i]:off[i+1]])
    vocab_sort   int32   [n] token ids ordered by their bytes -> str lookup = searchsorted
    merges       int32   [m,3] (rank, new_id, unused) aligned with merge_key
    merge_key    int64   [m] sorted (left<<32|right) -> pair lookup = searchsorted

`new_id` (the id of left+right) is carried so the whole BPE loop runs on INTEGERS —
no str objects are ever created for vocabulary entries.

Usage:
    uv run python scripts/build_tokenizer.py clip                 # CLIP BPE (PE-Core, OpenCLIP)
    uv run python scripts/build_tokenizer.py hf <tokenizer.json> <out.npz>   # any HF BPE

`clip` fetches openai/clip-vit-base-patch32's vocab.json+merges.txt (the same BPE PE-Core
and OpenCLIP ViT-B/32 use: 49,408 tokens, sot 49406, eot 49407).
"""

import json
import sys
from pathlib import Path

import numpy as np

BASE = "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main"
ROOT = Path(__file__).resolve().parents[1]


def pack(tokens: list[str], merges: list[tuple[str, str]], out: Path, **extra) -> None:
    ids = {t: i for i, t in enumerate(tokens)}
    enc = [t.encode() for t in tokens]
    off = np.zeros(len(enc) + 1, np.int32)
    off[1:] = np.cumsum([len(b) for b in enc], dtype=np.int32)
    blob = np.frombuffer(b"".join(enc), np.uint8)
    order = sorted(range(len(enc)), key=lambda i: enc[i])
    rows, keys = [], []
    for rank, (a, b) in enumerate(merges):
        if a in ids and b in ids and (a + b) in ids:
            rows.append((rank, ids[a + b], 0))
            keys.append((ids[a] << 32) | ids[b])
    rows = np.asarray(rows, np.int32)
    keys = np.asarray(keys, np.int64)
    o = np.argsort(keys, kind="stable")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, vocab_blob=blob, vocab_off=off, vocab_sort=np.asarray(order, np.int32),
                        merges=rows[o], merge_key=keys[o],
                        meta=np.frombuffer(json.dumps(extra).encode(), np.uint8))
    print(f"wrote {out} ({out.stat().st_size/1e6:.2f} MB) — {len(tokens)} tokens, {len(rows)} merges")


def build_clip() -> int:
    import httpx

    with httpx.Client(follow_redirects=True, timeout=60) as c:
        vocab = c.get(f"{BASE}/vocab.json").raise_for_status().json()
        merges = c.get(f"{BASE}/merges.txt").raise_for_status().text
    toks = [t for t, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
    pairs = [tuple(ln.split()) for ln in merges.splitlines()[1:] if ln.strip()]
    assert len(toks) == 49408 and len(pairs) == 48894, (len(toks), len(pairs))
    pack(toks, pairs, ROOT / "src/imgtag/data/clip-bpe.npz", kind="clip-bpe",
         source="openai/clip-vit-base-patch32", sot="<|startoftext|>", eot="<|endoftext|>")
    return 0


def build_hf(src: Path, out: Path) -> int:
    """Any HF `tokenizer.json` whose model.type is BPE (e.g. the Gemma tokenizer SigLIP2 uses)."""
    d = json.loads(src.read_bytes())
    m = d["model"]
    if m["type"] != "BPE":
        raise SystemExit(f"{src}: model.type={m['type']!r} — only BPE is supported by this packer")
    toks = [t for t, _ in sorted(m["vocab"].items(), key=lambda kv: kv[1])]
    pairs = [tuple(p) if isinstance(p, list) else tuple(p.split(" ", 1)) for p in m["merges"]]
    norm = d.get("normalizer") or {}
    pack(toks, pairs, out, kind="hf-bpe", source=str(src),
         byte_fallback=bool(m.get("byte_fallback")), unk=m.get("unk_token"),
         replace=[norm.get("pattern", {}).get("String"), norm.get("content")] if norm.get("type") == "Replace" else None,
         eos=next((a["content"] for a in d.get("added_tokens", []) if a["content"] == "<eos>"), None))
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "hf":
        sys.exit(build_hf(Path(sys.argv[2]), Path(sys.argv[3])))
    sys.exit(build_clip())
