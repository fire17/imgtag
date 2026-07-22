#!/usr/bin/env python3
"""OFFLINE build of the compact CLIP-BPE tokenizer binary bundled with imgtag.

Runtime must never parse a 34MB tokenizer.json (measured 0.64s) — this writes
``src/imgtag/data/clip-bpe.npz`` (~1MB) holding two newline-joined UTF-8 blobs:
the 49,408-entry vocab in id order and the BPE merge list in rank order.

    uv run python scripts/build_tokenizer.py

Source of truth: openai/clip-vit-base-patch32 (vocab.json + merges.txt) — the same
BPE PE-Core and OpenCLIP ViT-B/32 use (vocab_size 49408, sot 49406, eot 49407).
"""

import json
import sys
from pathlib import Path

import httpx
import numpy as np

BASE = "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main"
OUT = Path(__file__).resolve().parents[1] / "src/imgtag/data/clip-bpe.npz"


def main() -> int:
    with httpx.Client(follow_redirects=True, timeout=60) as c:
        vocab = c.get(f"{BASE}/vocab.json").raise_for_status().json()
        merges = c.get(f"{BASE}/merges.txt").raise_for_status().text
    toks = [t for t, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
    assert len(toks) == 49408, len(toks)
    lines = [ln for ln in merges.splitlines()[1:] if ln.strip()]
    assert len(lines) == 48894, len(lines)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT,
        vocab=np.frombuffer("\n".join(toks).encode(), np.uint8),
        merges=np.frombuffer("\n".join(lines).encode(), np.uint8),
    )
    print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.2f} MB), {len(toks)} tokens, {len(lines)} merges")
    return 0


if __name__ == "__main__":
    sys.exit(main())
