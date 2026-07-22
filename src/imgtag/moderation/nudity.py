"""Nudity / NSFW moderation track — recall-first site-rule enforcement.

VISION-ADDENDA 12:33Z (verbatim): "we dont want images with nudity, weapons or drugs …
these are very important to indentify correctly". Policy: ADR-14 two tiers —
``violation`` = human nudity/explicit · ``review`` = swimwear/lingerie · non-person nude
figures (statues, mannequins) = no flag.

OWNER: track-nudity. Research, ranked alternatives and every measured number:
research/track-nudity.md.

INSTRUMENT — a dedicated classifier, not a prompt ensemble. Marqo/nsfw-image-detection-384
(Apache-2.0, ViT-Tiny/16 @384, 5.6M params, published 98.56% accuracy on a 20k held-out
split), self-exported to ONNX by scripts/export_nudity_marqo.py (torch↔ORT parity
9.8e-07, one 22.5MB file + sha256). Weapons/drugs can ride the index embeddings because
"a rifle" is an object CLIP was trained to name; "nudity vs a swimsuit" is a boundary the
same embedding does not separate — measured on our own corpus, a CLIP prompt ensemble
ranks bikini and beach photos indistinguishably (research/track-nudity.md §baseline).
Cost of being right: one extra forward, ~4.5 GFLOPs/img (~¼ of PE-Core-S16-384). No new
runtime dependency — onnxruntime + numpy + Pillow only (ADR-7 intact).

OPERATING POINTS (both PROVISIONAL — ADR-14 keeps ``enforcement_ready`` false until τ is
fitted on labeled ground truth, which cannot happen on this machine):
  * ``violation`` τ=0.50 — the model's OWN argmax point, the only threshold the published
    98.56%/20k evaluation actually describes. Measured 0.18% flag rate on our 2.2k-image
    safe corpus (all false positives by construction).
  * ``review``    τ=0.10 — recall-first extension below it, sited just above the SFW mass
    (the safe corpus p95 is 0.07). Measured 3.8% flag rate on the safe corpus.
EVAL DATA LAW: no explicit-adult corpus is fetched here, so TRUE-POSITIVE RECALL IS NOT
MEASURED — it rests on Marqo's published evaluation and is labelled as such everywhere.
Only the false-positive side below is first-party measurement.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

# reuse, never re-implement: the engine already owns artifact lookup, ORT session
# construction (thread pinning) and the EXIF/draft-aware preprocess.
from ..core.models import _session, file_sha256, find_artifact, preprocess_image

CATEGORY = "nudity"
ARTIFACT = "nudity-marqo-384.onnx"
SPEC = {"subdir": "moderation"}
SIZE = 384  # Marqo pretrained_cfg: 384², bicubic, crop_pct 1.0 + crop_mode center
SQUASH = False  # crop_pct 1.0 center == resize shortest edge then centre-crop
NSFW_INDEX = 0  # config.json label_names == ["NSFW", "SFW"] — NOT alphabetical
TAU_VIOLATION = 0.50
TAU_REVIEW = 0.10

#: Zero-shot prompt ensemble — the research BASELINE and the no-artifact fallback, never
#: the default. Scored against whichever text tower the index used, so it costs one text
#: batch once and one dot product per image; its recall is UNMEASURABLE here, so anything
#: it produces is marked ``calibrated: false`` and may only reach the review tier.
PROMPTS_POSITIVE = (
    "a nude person",
    "a naked person",
    "explicit sexual content",
    "pornography",
    "a topless woman",
    "bare buttocks",
    "exposed breasts",
    "exposed genitals",
)
PROMPTS_NEGATIVE = (
    "a photo",
    "a landscape photograph",
    "a person wearing clothes",
    "a portrait of a clothed person",
    "a person at the beach in a swimsuit",
    "a fashion photograph",
    "an object on a table",
    "an animal",
)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def tier_of(p: float, tau_violation: float = TAU_VIOLATION, tau_review: float = TAU_REVIEW) -> str:
    """ADR-14 tier for one probability. ``none`` is a real answer, not a missing one."""
    return "violation" if p >= tau_violation else "review" if p >= tau_review else "none"


class NudityHead:
    """The dispatcher-facing head (imgtag.moderation.load_heads contract).

    ``score(embeddings, images, ids) -> list[dict]`` — one dict per record, schema
    ``{category, p, tier}`` plus provenance. Embeddings are ignored: this track answers
    from pixels, which is exactly why it can answer at all.
    """

    category = CATEGORY
    wants_images = True
    size, squash = SIZE, SQUASH
    resample = Image.Resampling.BICUBIC

    def __init__(self, path: Path, profile: dict | None = None):
        profile = profile or {}
        self.path = path
        self.model_sha = file_sha256(path)
        self.model_id = "marqo-nsfw-384-fp32"
        self._s = _session(path, profile.get("intra_op", 2))
        self._in = self._s.get_inputs()[0].name
        calib = path.with_suffix(".calib.json")
        self.calib = json.loads(calib.read_bytes()) if calib.is_file() else {}
        self.tau_violation = float(profile.get("nudity_tau_violation")
                                   or self.calib.get("tau_violation") or TAU_VIOLATION)
        self.tau_review = float(profile.get("nudity_tau_review")
                                or self.calib.get("tau_review") or TAU_REVIEW)
        #: ADR-14: τ was never fitted against labeled nudity ground truth (EVAL DATA LAW),
        #: so this track never claims enforcement readiness, however good the FP side looks.
        self.calibrated = False

    # ── pixels ────────────────────────────────────────────────────────────────
    def preprocess(self, im: Image.Image) -> np.ndarray:
        """PIL -> uint8 [384,384,3] in THIS model's geometry, not the backend's."""
        return preprocess_image(im, self.size, self.squash, self.resample)

    # ── scoring ───────────────────────────────────────────────────────────────
    def probs(self, batch: np.ndarray) -> np.ndarray:
        """uint8 [n,384,384,3] -> f32 [n] p(NSFW)."""
        x = np.asarray(batch, np.uint8).astype(np.float32) / 127.5 - 1.0  # (x/255-.5)/.5
        x = np.ascontiguousarray(x.transpose(0, 3, 1, 2))
        return _softmax(np.asarray(self._s.run(None, {self._in: x})[0], np.float32))[:, NSFW_INDEX]

    def _flag(self, p: float) -> dict:
        return {"category": CATEGORY, "p": round(float(p), 4), "tier": tier_of(p, self.tau_violation, self.tau_review),
                "model_id": self.model_id, "calibrated": self.calibrated}

    def score(self, embeddings, images=None, ids=None) -> list[dict]:
        n = len(ids) if ids is not None else (len(images) if images is not None else len(embeddings))
        if images is not None and images.shape[1:] == (self.size, self.size, 3) and self.squash:
            return [self._flag(v) for v in self.probs(np.asarray(images, np.uint8))]
        pix, slots = self._reopen(ids, n)
        out = [{"category": CATEGORY, "p": 0.0, "tier": "none", "model_id": self.model_id,
                "calibrated": self.calibrated, "unreadable": True} for _ in range(n)]
        if pix is None:
            return out
        p = self.probs(pix)
        for value, i in zip(p, slots):
            out[i] = self._flag(value)
        return out

    def _reopen(self, ids, n) -> tuple[np.ndarray | None, list[int]]:
        """The coordinator's slab carries the BACKEND's geometry (squashed, sometimes
        224²) — a domain shift this model was never trained for — so unless that geometry
        already matches ours the file is re-opened and preprocessed properly. ``draft()``
        makes that a partial JPEG decode, not a full one."""
        pix, slots = [], []
        for i, rec in enumerate(ids or []):
            src = rec.get("path") if isinstance(rec, dict) else rec
            try:
                with Image.open(src) as im:
                    pix.append(self.preprocess(im))
                slots.append(i)
            except Exception:
                continue  # an unreadable file is unflagged, never a crashed index
        return (np.stack(pix) if pix else None), slots

    def score_images(self, images: list[Image.Image]) -> list[dict]:
        """Convenience path for callers holding PIL images (tests, CLI one-offs)."""
        if not images:
            return []
        return [self._flag(v) for v in self.probs(np.stack([self.preprocess(im) for im in images]))]


class ZeroShotNudityHead:
    """Research baseline and offline fallback: prompt-ensemble margin over the embeddings
    the index already computed. Marginal cost ~0 — but its threshold is a FLAG BUDGET over
    a safe corpus, not a calibrated probability, and it demonstrably cannot separate
    swimwear from nudity. review tier only, always ``calibrated: false``."""

    category = CATEGORY
    wants_images = False
    model_id = "zeroshot-prompt-ensemble"
    calibrated = False

    def __init__(self, backend, tau: float = 0.02):
        self.backend = backend
        self.model_sha = getattr(backend, "model_sha", "")
        pos = backend.embed_texts(list(PROMPTS_POSITIVE))
        neg = backend.embed_texts(list(PROMPTS_NEGATIVE))
        self._t = np.ascontiguousarray(np.vstack([pos, neg]).T.astype(np.float32))
        self._npos = len(PROMPTS_POSITIVE)
        self.tau = float(tau)

    def margins(self, embs: np.ndarray) -> np.ndarray:
        """f32 [n,D] L2-normalized image embeddings -> f32 [n] margin. The background
        prompts are what stop a beach photo from winning on skin tone alone."""
        s = np.asarray(embs, np.float32) @ self._t
        return s[:, : self._npos].max(1) - s[:, self._npos :].max(1)

    def score(self, embeddings, images=None, ids=None) -> list[dict]:
        return [{"category": CATEGORY, "p": round(float(v), 4),
                 "tier": "review" if v >= self.tau else "none",
                 "model_id": self.model_id, "calibrated": False}
                for v in self.margins(embeddings)]


def load_nudity_head(profile: dict | None = None) -> NudityHead | None:
    """The ONNX head, or None when its artifact is absent — a missing track is simply not
    loaded (the dispatcher reports moderation heads by name), never a silent zero."""
    path = find_artifact(SPEC, ARTIFACT)
    return None if path is None else NudityHead(path, profile)
