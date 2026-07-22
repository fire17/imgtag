"""Content-moderation tracks (VISION-ADDENDA 12:33Z + policy ADR-14 — two-tier flags).

One module per track, one owner per module (F2). Tracks choose their own instrument:
weapons/drugs score the embeddings the index already computed (one matmul, no extra
model); nudity runs a dedicated 5.6M-param head (+~4.5 GFLOPs/img) because recall-first
enforcement needs citable metrics a prompt ensemble cannot provide. Per-image output
schema, all tracks: {"category": str, "p": float, "tier": "violation"|"review"|"none"}.

CONDUCTOR-OWNED seam (this file only): the dispatcher b-engine calls.
Each track exports load_<category>_head(profile) -> Head | None (None = model files
absent / track disabled; indexing proceeds without it, counted as "moderation: off").
A Head exposes .score(embeddings, images, ids) -> list[dict] per the schema above.
"""

from importlib import import_module

_TRACKS = ("nudity", "weapons", "drugs")


def load_heads(profile):
    """Return {category: Head} for every track that can load on this machine."""
    heads = {}
    for name in _TRACKS:
        try:
            mod = import_module(f"imgtag.moderation.{name}")
        except ImportError:
            continue
        loader = getattr(mod, f"load_{name}_head", None)
        if loader is None:
            continue
        head = loader(profile)
        if head is not None:
            heads[name] = head
    return heads
