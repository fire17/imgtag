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

#: CONTENT-FREE GUARD. Measured 2026-07-22: on content-free input this model degenerates
#: to a COLOUR prior — a solid flesh tone filling the frame scores p=0.5498 and a
#: flesh-toned linear gradient p=0.7612, i.e. VIOLATION tier from an image containing no
#: subject at all. Reproduced identically in torch, and our preprocessed tensor matches
#: timm's own transform bit-for-bit, so this is the model's out-of-distribution behaviour,
#: not a bug on our side (research/track-nudity.md §9).
#:
#: Structure = mean |discrete Laplacian| of the preprocessed frame. Second-order on
#: purpose: a solid colour AND a linear gradient both have zero second derivative, while
#: every photograph has texture. Measured over 1,826 real photographs: min **1.171**,
#: p0.1 1.413, p50 14.68. Synthetic probes: solid 0.000 · linear gradient 0.667 ·
#: flesh gradient 0.637 · radial gradient 0.994 · noise 168. The floor sits in the gap:
#: 1.006× above the highest synthetic probe, 1.171× below the lowest real photograph.
#: Below it the record is re-TIERED to "none" and marked ``content_free`` — p is still
#: reported and nothing is dropped, so an operator can always query what was set aside.
#: KNOWN RESIDUAL RECALL HOLE: an image deliberately smoothed below this floor is set
#: aside. Documented in research/track-nudity.md §9 rather than hidden.
MIN_STRUCTURE = 1.0

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


def structure(batch: np.ndarray) -> np.ndarray:
    """uint8 [n,H,W,3] -> f32 [n] mean |discrete Laplacian|, summed over both axes.
    Two second-difference passes over 384² — sub-millisecond against a ~90ms forward."""
    g = np.asarray(batch, np.uint8).astype(np.float32).mean(-1)
    return (np.abs(g[:, :-2] - 2 * g[:, 1:-1] + g[:, 2:]).mean((1, 2))
            + np.abs(g[:, :, :-2] - 2 * g[:, :, 1:-1] + g[:, :, 2:]).mean((1, 2)))


#: The pre-made-view contract (ruling 2026-07-22). b-engine's decode worker may hand this
#: track a second uint8 view at OUR geometry, eliminating the re-open. To make transform
#: drift structurally impossible, the worker calls ``make_view`` BELOW — one implementation,
#: not two-plus-a-parity-test.
VIEW_KEY = "nudity-384crop"


def make_view(im: Image.Image) -> np.ndarray:
    """THE nudity view: PIL -> uint8 [384,384,3], shortest-side→384, centre-crop, BICUBIC.

    ⚠️ **DRAFT-SCALE PRECONDITION — measured, and the whole correctness of the fast path.**
    ``preprocess_image`` calls ``im.draft()``, which sets the JPEG's DCT-domain decode
    scale, and that scale is part of the result. Measured on real photos:

    ============================ =========================================
    decode state when called     result vs a fresh open at our geometry
    ============================ =========================================
    drafted (384,384)            **bit-identical**
    drafted (224,224) first      differs, max 33/255, mean 1.3  ← WRONG
    fully decoded, never drafted differs, max 9/255
    ============================ =========================================

    So the worker may pass a view ONLY when its own decode was drafted at 384 — i.e. when
    the active backend's size is 384 (pecore-s16-384, the default). Then ONE decode serves
    both and both are bit-identical (verified on 4 images: our view AND the backend's own
    view are each unchanged).

    It is NOT free to force this for smaller backends: drafting at 384 to serve us
    perturbs a 224-backend's OWN embeddings by up to **83/255** — corrupting the index to
    speed up moderation. When the backend is not 384, ship no view; this track re-opens
    (a partial 384 decode) and stays exactly as calibrated.
    """
    return preprocess_image(im, SIZE, SQUASH, Image.Resampling.BICUBIC)


class NudityHead:
    """The dispatcher-facing head (imgtag.moderation.load_heads contract).

    ``score(embeddings, images, ids, views=None) -> list[dict]`` — one dict per record,
    schema ``{category, p, tier}`` plus provenance. Embeddings are ignored: this track
    answers from pixels, which is exactly why it can answer at all.

    Pixel source, in priority order: an offered ``views[VIEW_KEY]`` → the coordinator slab
    when its geometry already matches ours → re-open from ``rec["path"]``.
    """

    category = CATEGORY
    wants_images = True
    size, squash = SIZE, SQUASH
    resample = Image.Resampling.BICUBIC
    #: Declared so the worker knows what to build and under which precondition.
    view_key = VIEW_KEY
    view_geometry = {"size": SIZE, "squash": SQUASH, "resample": "BICUBIC",
                     "requires_draft": SIZE}

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
        self.min_structure = float(profile.get("nudity_min_structure")
                                   or self.calib.get("min_structure") or MIN_STRUCTURE)
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

    def _flags(self, batch: np.ndarray) -> list[dict]:
        """probs + the content-free guard, in one pass over the batch."""
        out = []
        for p, s in zip(self.probs(batch), structure(batch)):
            f = {"category": CATEGORY, "p": round(float(p), 4),
                 "tier": tier_of(p, self.tau_violation, self.tau_review),
                 "model_id": self.model_id, "calibrated": self.calibrated}
            if s < self.min_structure:  # no photograph of a person lives down here
                f["tier"], f["content_free"] = "none", True
            out.append(f)
        return out

    def _offered(self, views, n: int) -> np.ndarray | None:
        """A pre-made view, if one was offered AND it is the right shape. Shape/dtype are
        checked, not trusted — a mis-shaped view must fall back, never be fed to the model.
        Its DECODE state cannot be checked here; that precondition is ``make_view``'s
        contract and is covered by the parity test."""
        v = (views or {}).get(VIEW_KEY)
        if v is None:
            return None
        v = np.asarray(v)
        if v.dtype != np.uint8 or v.shape != (n, self.size, self.size, 3):
            return None
        return v

    def score(self, embeddings, images=None, ids=None, views=None) -> list[dict]:
        n = len(ids) if ids is not None else (len(images) if images is not None else len(embeddings))
        offered = self._offered(views, n)
        if offered is not None:
            return self._flags(offered)
        if images is not None and images.shape[1:] == (self.size, self.size, 3) and self.squash:
            return self._flags(np.asarray(images, np.uint8))
        pix, slots = self._reopen(ids, n)
        out = [{"category": CATEGORY, "p": 0.0, "tier": "none", "model_id": self.model_id,
                "calibrated": self.calibrated, "unreadable": True} for _ in range(n)]
        if pix is None:
            return out
        for f, i in zip(self._flags(pix), slots):
            out[i] = f
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
        return self._flags(np.stack([self.preprocess(im) for im in images]))


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


# ── REVIEW-TIER instrument: prompt-ensemble margin over the index embeddings ───────────
# Measured 2026-07-22 (research/track-nudity.md §10, research/eval-nudity-probe.json): the
# Marqo head is a VIOLATION detector and scores swimwear in the SAME 0.05-0.15 band as
# mannequins (swimwear p50 0.056 == mannequin p50 0.056), so it CANNOT carry the review
# tier. Review is served instead by a margin over the embedding the index already computed
# (TRACKS T1, ~0 extra cost): margin = max(review concepts) - max(negatives incl. the
# ADR-14 never-flag figures). This COMPOSES with the Marqo head — violation from Marqo
# (pixels), review from the margin (embeddings) — and is fitted on data/nudity-probe.
# SEPARATION IS SWIMWEAR-ONLY: lingerie/underwear/bare-chest do NOT separate on the shared
# embedding (a distilled head is owed, TRACKS T2). tau_review is a recall-first review-QUEUE
# threshold in MARGIN space, never enforcement — calibrated stays False.


def _review_spec() -> dict:
    """The ``nudity`` entry of data/moderation.json (conductor-owned file, this key ours).
    Never raises — a missing/broken spec just means the review instrument is unavailable."""
    try:
        data = Path(__file__).resolve().parent.parent / "data" / "moderation.json"
        return json.loads(data.read_bytes())["categories"]["nudity"]
    except (OSError, ValueError, KeyError):
        return {}


def _concept_vectors(backend, texts: list[str], templates) -> np.ndarray:
    """One L2-normalized vector per concept = mean over templates (classic CLIP ensembling)."""
    flat = [t.format(c) for c in texts for t in templates]
    emb = np.asarray(backend.embed_texts(flat), np.float32).reshape(len(texts), len(templates), -1)
    v = emb.mean(1)
    return np.ascontiguousarray(v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12))


class NudityReviewScorer:
    """Review-tier margin instrument. Build once per (model, spec); score any dataset for ~0.

    ``score(embeddings, images, ids) -> [{category, p, tier: review|none, subcategory}]``.
    NEVER emits ``violation`` (that tier belongs to the Marqo head). ``calibrated`` is False:
    ``tau_review`` is a recall-first review-QUEUE threshold in margin space, not enforcement.
    """

    category = CATEGORY
    wants_images = False
    model_id = "nudity-review-margin"
    calibrated = False

    def __init__(self, pos, groups, neg, platt, tau_review, model_sha=""):
        self.pos, self.groups, self.neg = pos, groups, neg
        self.platt = tuple(platt)
        self.tau_review = float(tau_review)
        self.model_sha = model_sha

    @classmethod
    def build(cls, backend, spec: dict | None = None, profile: dict | None = None) -> "NudityReviewScorer":
        spec = _review_spec() if spec is None else spec
        profile = profile or {}
        subs = spec.get("review_subcategories") or {"review": list(spec.get("review") or PROMPTS_POSITIVE)}
        templates = spec.get("templates") or ["a photo of {}.", "a close-up photo of {}.", "{}"]
        names, groups = [], []
        for g, cs in subs.items():
            names += list(cs)
            groups += [g] * len(cs)
        neg = list(spec.get("negatives") or PROMPTS_NEGATIVE)
        platt = spec.get("platt") or [1.0, 0.0]
        tau = float(profile.get("nudity_tau_review") or spec.get("tau_review") or TAU_REVIEW)
        return cls(_concept_vectors(backend, names, templates), groups,
                   _concept_vectors(backend, neg, templates), platt, tau,
                   getattr(backend, "model_sha", ""))

    def margins(self, embs) -> tuple[np.ndarray, np.ndarray]:
        """(margin, argmax-concept-index). margin = max(review) - max(negatives)."""
        e = np.asarray(embs, np.float32)
        cp = e @ self.pos.T
        return cp.max(1) - (e @ self.neg.T).max(1), cp.argmax(1)

    def probs(self, embs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        m, best = self.margins(embs)
        a, b = self.platt
        return 1.0 / (1.0 + np.exp(-(a * m + b))), m, best

    def score(self, embeddings, images=None, ids=None, views=None) -> list[dict]:
        p, m, best = self.probs(embeddings)
        return [{"category": CATEGORY, "p": round(float(pi), 4),
                 "tier": "review" if mi >= self.tau_review else "none",
                 "subcategory": self.groups[bi], "margin": round(float(mi), 4),
                 "model_id": self.model_id, "calibrated": False}
                for pi, mi, bi in zip(p, m, best)]


def load_nudity_review_scorer(backend, profile: dict | None = None) -> NudityReviewScorer | None:
    """The review instrument, or None if the spec/backend is unavailable.

    Kept OUT of the auto-loaded runtime head (``load_nudity_head``) DELIBERATELY: separation
    is swimwear-only (measured), so wiring it into the index-time composition is a flagged
    next step — earned when a subcategory clears the bar or a distilled head is trained
    (research/track-nudity.md §10). Until then it is an on-demand, calibratable instrument
    the operator (or b-daemon's tag path) can run to elevate swimwear above the FP band."""
    try:
        return NudityReviewScorer.build(backend, profile=profile)
    except Exception:
        return None


def load_nudity_head(profile: dict | None = None) -> NudityHead | None:
    """The ONNX head, or None when its artifact is absent — a missing track is simply not
    loaded (the dispatcher reports moderation heads by name), never a silent zero."""
    path = find_artifact(SPEC, ARTIFACT)
    return None if path is None else NudityHead(path, profile)
