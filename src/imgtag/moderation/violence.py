"""VIOLENCE / ABUSE moderation track — embedding-space prompt-ensemble detector.

VISION-ADDENDA 2026-07-22 13:29Z (verbatim): *"and one track for general violence or
abuse"*. Policy: ORACLE ADR-14 (tiers) + ADR-15 / TRACKS.md (the scaling law).
OWNER: track-violence. Sibling tracks own their own files; `moderation.json`'s
`violence` key is the only shared surface this lane writes.

WHY EMBEDDING-SPACE, NOT A MODEL (TRACKS.md T2)
----------------------------------------------
The B25 dedicated-model budget (Σ dedicated FLOPs ≤ 30% of the encoder) is already ~25%
consumed by nudity's Marqo head. This track therefore takes instrument tier 1 — one
`[N,D]·[D,P]` matmul over the embedding the index ALREADY computed — which costs ~0 and
keeps the 100-track invariant true. If a permissively-licensed still-image violence model
ever proves worth it, TRACKS.md T2 mandates it enter as an offline TEACHER for a distilled
head over these same embeddings, never as a second forward pass.

WHAT THIS IS, HONESTLY
----------------------
A **recall-first triage** scorer, not a classifier. Two structural facts bound every claim
this module makes, and both are stated in `research/track-violence.md` in full:

  * **EVAL DATA LAW — obeyed.** No graphic-violence corpus was fetched to this machine.
    Every first-party number is therefore FALSE-POSITIVE-side only, measured on safe
    corpora already on disk (COCO val2017 + Unsplash keyword slices).
  * **No labelled positives exist here.** τ is consequently a FALSE-POSITIVE BUDGET
    (a quantile of the safe-corpus margin), never a recall fit — so `calibrated` is False
    and `enforcement_ready` is False, permanently, until labelled ground truth exists on
    the target host. `p` is a monotone triage score, NOT a calibrated probability.

SCORING (ADR-3 §2 — the commissioned background-margin experiment)
------------------------------------------------------------------
Each concept is a prompt ENSEMBLE: mean of its templated text embeddings, L2-normalized.
FOUR banks, and keeping them apart is the whole design:

    SEVERE     graphic gore / heavy blood / mutilation          → ALERT tier
    VIOLENT    depicted interpersonal violence, assault, abuse  → the score `p`
    CONTEXT    staged & clinical near-twins (halloween SFX, film
               gore, surgery, butchery, property damage)        → REVIEW prompts (reader)
    BACKGROUND visually DISTINCT confusables (contact sports,
               peaceful protest, red food/paint, parades)       → SUBTRACTED

    margin = max_cos(image, SEVERE∪VIOLENT) - max_cos(image, BACKGROUND)
    p      = sigmoid(PLATT_A·margin + PLATT_B)      # the ONE score ADR-15 stores

The subtraction is what makes the number comparable across images (absolute CLIP cosines
are dominated by per-image norm/entropy effects, not content). The split between
BACKGROUND (subtracted) and CONTEXT (never subtracted) is the drugs lane's most expensive
lesson applied forward: subtracting concepts that are *visually identical* to the positives
subtracts the signal itself (their AP collapsed 0.58 → 0.04 when clinical syringes were
subtracted from drug syringes). Halloween gore and surgical blood are exactly that kind of
twin, so they live in the CONTEXT/`review` prompt set, never in BACKGROUND.

TIERS ARE `p`-SPACE BANDS, NOT A SEPARATE ARBITER (the 2026-07-22 nudityprobe lesson).
ADR-15 stores ONE scalar (`p`) per image; ``store.derive_tiers`` bands it (B25d: one
mapping so head, daemon reader and CLI recount cannot disagree). An earlier design gated
`alert`/`violation` on SEPARATE margins in MARGIN space with a CONTEXT arbiter — that logic
was silently bypassed by both shipping paths, and worse, its margin-space taus (~0.05) were
then applied by derive_tiers to the `p`-space stored score, flagging 16 false `alert` + 79
`violation` on a 202-image swimwear probe. Fixed: `p`-space ASCENDING taus (review <
violation < alert), so a swimwear photo whose intimate-pose `p` reaches ~0.5 lands at
`review`, never `alert`. Staged/clinical demotion now rides the conservative violation/alert
taus (measured: halloween → review) plus the reader's own exceedance competition, not a
per-image arbiter a single stored scalar cannot carry.

BOUNDARY WITH track-safety (agreed, ORACLE ADR-14 `alert` semantics): safety owns
person-DOWN ∧ danger-context (the victim's state); this track owns depicted interpersonal
violence, and `alert` (the top `p` band, τ=0.95) is reserved for the genuinely extreme.
Both tracks score every image independently (ADR-15); tier counts are per category, so no
double-count arises (b-daemon's `alert_images` dedupes any cross-category fleet total).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CATEGORY = "violence"

#: Three templates. Same as drugs/sports — measured gain past three was noise there, and
#: this lane has no labelled data with which to justify a different number.
TEMPLATES = ("a photo of {}.", "a close-up photo of {}.", "{}")

# ── SEVERE: the ALERT tier. Graphic imagery ITSELF, not the act. Deliberately narrow:
# `alert` sorts above `violation` in every count and view (ADR-14), so a wide alert bank
# would swamp the most severe signal the product has. ──────────────────────────────────
SEVERE: list[str] = [
    "a graphic photo of a severe bleeding wound",
    "a gory scene with blood everywhere",
    "a mutilated human body",
    "a dead body with visible injuries",
    "a pool of blood on the floor",
    "a person covered in blood",
    "a severed limb",
    "graphic gore and viscera",
]

# ── VIOLENT: the VIOLATION tier. Grouped so an operator sees WHY, and so a future refit
# can drop or retune one subcategory without touching the others. ─────────────────────
VIOLENT: dict[str, list[str]] = {
    "fighting": [
        "two people fighting and punching each other",
        "a street fight between two men",
        "a person punching another person in the face",
        "a violent brawl between several people",
        "a person kicking someone who is on the ground",
    ],
    "assault": [
        "a person violently attacking another person",
        "a man grabbing a woman by the throat",
        "a person being strangled",
        "a person raising a fist to strike someone",
        "a person shoving another person violently",
    ],
    "armed_threat": [
        "a person pointing a gun at another person",
        "a person holding a knife to someone's throat",
        "an armed robbery in progress",
        "a person threatening someone with a weapon",
        "a person aiming a rifle at a person",
    ],
    "abuse": [
        "domestic violence between partners",
        "a child cowering from an adult raising a hand to hit them",
        "a person cowering in fear from an attacker",
        "a person being bullied and pushed against a wall",
        "a person held down against their will",
    ],
    "injury_detail": [
        "a bleeding facial injury on a person",
        "a bruised and beaten face",
        "a bloody wound on a person's body",
        "a person with a bleeding head injury",
    ],
    "aftermath": [
        "the aftermath of a violent attack with blood on the floor",
        "a bloodstained wall at a crime scene",
    ],
    "crowd_violence": [
        "riot police beating protesters with batons",
        "a violent riot with people attacking each other",
        "a person being beaten by an angry crowd",
    ],
    "armed_conflict": [
        "soldiers shooting at people in combat",
        "a civilian wounded in a war zone",
    ],
}

# ── CONTEXT: the REVIEW tier AND the arbitration bank. Visually near-IDENTICAL twins of
# the banks above. NEVER subtracted — see the module docstring for why that would cost
# the recall this track exists to buy. ────────────────────────────────────────────────
CONTEXT: list[str] = [
    # staged / fictional
    "a horror movie still with fake blood",
    "a halloween zombie costume with special effects makeup",
    "a person wearing scary halloween makeup with fake wounds",
    "a violent video game screenshot",
    "a staged fight scene in a film with stunt actors",
    "a historical battle reenactment with costumed actors",
    # clinical / occupational — real blood, no violence
    "a surgeon operating in an operating theatre",
    "a doctor treating a wound with bandages",
    "a first aid kit and a bandaged arm",
    "raw meat on a butcher's counter",
    # ambiguous human conflict, no contact
    "two people shouting angrily at each other",
    "an angry person yelling at the camera",
    # damage without people
    "a car wrecked after a road accident",
    "a broken window and destroyed property",
]

# ── BACKGROUND: subtracted. Visually DISTINCT confusables plus generic scene prompts, so
# the margin is a real contrast rather than an absolute cosine (ADR-3). The contact-sport
# block is lifted VERBATIM from `sports.py`'s martial-arts / team-sport banks so the two
# tracks stay in the same concept space: a sanctioned bout is exculpatory evidence, and
# the composition ruling is `sports: match(martial arts)` + `violence: none-or-review`.
BACKGROUND: list[str] = [
    # contact sports — THE classic false-positive class for this category
    "two boxers fighting in a boxing ring",
    "a photo of boxing gloves",
    "martial artists sparring in a dojo",
    "a judo throw on a mat",
    "a wrestling match on a mat",
    "a fencing bout with masks and foils",
    "a rugby tackle during a match",
    "an american football tackle on a field",
    "an ice hockey game on the rink",
    "a soccer match on a pitch",
    "a basketball game in an arena",
    "athletes competing in a stadium with a crowd",
    # peaceful assembly — protest is not violence
    "a peaceful protest march with banners and placards",
    "a crowd of people at an outdoor demonstration",
    "a music concert crowd with raised hands",
    # ceremonial / display military — not combat
    "a military parade with soldiers marching",
    "a museum display of historical weapons",
    "a war memorial statue",
    # red things that are not blood
    "ketchup on a plate of food",
    "spilled red paint",
    "a glass of red wine",
    "ripe tomatoes and strawberries",
    "red fabric and red paint on a canvas",
    # generic scene background
    "a photo of a person",
    "a portrait of a smiling person",
    "a photo of a landscape",
    "a photo of food",
    "a photo of an animal",
    "a photo of a car",
    "a photo of a building",
    "a photo of a document",
    "an abstract pattern",
]

#: Mapping margin -> the stored score `p`. Fitted to the SPREAD of the safe corpus (COCO
#: val2017, n=5000, `pecore-s16-384-fp32`; see `research/track-violence.md` §5), NOT to
#: labels: it maps the safe-corpus median margin (-0.0075) to p≈0.05 and the p99.9 margin
#: (0.0874) to p≈0.90. Measured result: only 0.1% of COCO maps above p=0.9 — the score
#: distribution does NOT saturate (the b-daemon defect that got the drugs proxy logistic
#: rolled back). `p` is a monotone triage score. It is NOT a probability, ever. This `p`
#: is what b-engine writes to the ADR-15 sidecar; the tiers below are derived from IT.
PLATT_A, PLATT_B = 54.0, -2.54

#: FALSE-POSITIVE BUDGET tier thresholds — in the SAME SPACE as the stored score `p`
#: ([0,1] after PLATT), ASCENDING, so ``store.derive_tiers`` (the ADR-15/B25d single
#: source of truth that bands the ONE stored score) assigns the correct severity:
#: p≥alert → alert, p≥violation → violation, p≥review → review, else none.
#:
#: LAW: these are `p`-space quantiles of the SAFE corpus, NEVER margin values. The
#: 2026-07-22 nudityprobe incident (16 false `alert` + 79 `violation` on 202 swimwear
#: images) was exactly a UNIT bug — margin-space taus (~0.05) applied by derive_tiers to
#: `p`-space scores, with tau_alert < tau_violation inverting the severity order. Fixed
#: by moving to ascending `p`-space bands. Measured FP on COCO val2017 in parens:
TAU_ALERT = 0.95        # COCO p99.98 (~0.02% FP) — reserved for genuinely extreme imagery
TAU_VIOLATION = 0.85    # COCO ~p99.7 (~0.3% FP)
TAU_REVIEW = 0.46       # COCO p95 (~5% FP) — the wide recall-first net
#: verified on the OOD swimwear corpus that triggered the incident: nudityprobe (n=202,
#: max stored p=0.82) → 0 alert, 0 violation (0.82 < 0.85), ~10 review. Was 16 / 79.

TIERS = ("alert", "violation", "review")


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))


def tier_of(p, pol: dict):
    """`p`-space bands → ADR-14 tier(s). Ascending taus, highest first — byte-identical in
    ordering to ``store.derive_tiers`` so the head and the recount/daemon never disagree.
    Accepts a scalar or an array; returns the same shape (dtype=object for arrays)."""
    ta, tv, tr = float(pol["tau_alert"]), float(pol["tau_violation"]), float(pol["tau_review"])
    a = np.asarray(p, np.float64)
    out = np.where(a >= ta, "alert",
                   np.where(a >= tv, "violation",
                            np.where(a >= tr, "review", "none")))
    return out if out.ndim else str(out)


# ── the user-rulable knobs live in CONFIG, not in code ────────────────────────────────
DEFAULTS = {"tau_alert": TAU_ALERT, "tau_violation": TAU_VIOLATION,
            "tau_review": TAU_REVIEW}


def policy(config: dict | None = None) -> dict:
    """Effective policy = module defaults ← moderation.json `categories.violence` ← arg.

    Unknown/invalid values fall back to the default rather than raising: a typo in a
    config file must not take moderation offline.
    """
    p = dict(DEFAULTS)
    if config is None:
        try:
            data = Path(__file__).resolve().parent.parent / "data" / "moderation.json"
            config = json.loads(data.read_bytes())["categories"][CATEGORY]
        except (OSError, ValueError, KeyError):
            config = {}
    for k in DEFAULTS:
        try:
            if config.get(k) is not None:
                p[k] = float(config[k])
        except (TypeError, ValueError):
            pass
    return p


def prompts() -> tuple[list[str], list[str], list[str], list[str], list[str]]:
    """(severe, violent, violent-group-labels, context, background)."""
    vio, groups = [], []
    for g, cs in VIOLENT.items():
        vio += cs
        groups += [g] * len(cs)
    return list(SEVERE), vio, groups, list(CONTEXT), list(BACKGROUND)


def spec_sha() -> str:
    sev, vio, _, ctx, bg = prompts()
    blob = "\x00".join(list(TEMPLATES) + sev + vio + ctx + bg)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def concept_vectors(backend, texts: list[str]) -> np.ndarray:
    """One L2-normalized vector per concept = mean over TEMPLATES. One text batch."""
    flat = [t.format(c) for c in texts for t in TEMPLATES]
    emb = np.asarray(backend.embed_texts(flat), np.float32).reshape(len(texts), len(TEMPLATES), -1)
    v = emb.mean(1)
    return np.ascontiguousarray(v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12))


@dataclass
class ViolenceScorer:
    """Holds the four prompt banks for one model. Build once, score any dataset.

    >>> s = ViolenceScorer.build(backend)     # one text-tower batch
    >>> out = s.score(snapshot.emb)           # one [N, P] matmul
    """

    severe: np.ndarray
    violent: np.ndarray
    groups: list[str]
    context: np.ndarray
    bg: np.ndarray
    names: list[str]
    ctx_names: list[str]
    pol: dict = field(default_factory=policy)

    @classmethod
    def build(cls, backend, config: dict | None = None) -> "ViolenceScorer":
        sev, vio, groups, ctx, bg = prompts()
        return cls(severe=concept_vectors(backend, sev),
                   violent=concept_vectors(backend, vio), groups=groups,
                   context=concept_vectors(backend, ctx), bg=concept_vectors(backend, bg),
                   names=vio, ctx_names=ctx, pol=policy(config))

    def score(self, emb: np.ndarray) -> dict:
        """emb: [N, D] L2-normalized image embeddings → per-image violence payload.

        Arrays throughout — a whole dataset costs four small matmuls and no decode.
        `p` (max of the severe and violent background-margins, mapped through PLATT) is the
        ONE score ADR-15 stores densely; **tiers are derived from `p` by ascending
        `p`-space bands — byte-identical to ``store.derive_tiers`` (B25d: one mapping, so
        the head, the daemon reader and the CLI recount cannot disagree)**. The CONTEXT
        bank is scored for the reader's exceedance path and for the `context` diagnostic,
        but it does NOT arbitrate here: a single stored scalar cannot carry a separate
        demotion, so staged/clinical demotion rides the conservative violation/alert taus
        (measured acceptable) plus the reader's own review-tier competition.
        """
        pol = self.pol
        emb = np.asarray(emb, np.float32)
        if emb.ndim != 2:
            raise ValueError(f"violence scorer expects [N,D], got {np.shape(emb)}")
        bg = (emb @ self.bg.T).max(1)                     # [N]
        cv = emb @ self.violent.T                         # [N, P]
        m_sev = (emb @ self.severe.T).max(1) - bg
        m_vio = cv.max(1) - bg
        m_ctx = (emb @ self.context.T).max(1) - bg
        best = cv.argmax(1)
        m = np.maximum(m_sev, m_vio)                      # the worst thing it looks like
        p = _sigmoid(PLATT_A * m + PLATT_B)               # the stored score (ADR-15)
        tier = tier_of(p, pol)                            # p-space bands, == derive_tiers
        return {"category": CATEGORY, "p": p, "tier": tier,
                "margin": m, "margin_severe": m_sev, "margin_violent": m_vio,
                "margin_context": m_ctx,
                "concept": [self.names[i] for i in best],
                "group": [self.groups[i] for i in best],
                "calibration": "fp-budget",   # never claim more than the fit supports
                "tau_alert": pol["tau_alert"], "tau_violation": pol["tau_violation"],
                "tau_review": pol["tau_review"]}

    def per_image(self, emb: np.ndarray, i: int) -> dict:
        out = self.score(np.asarray(emb, np.float32)[i : i + 1])
        return {"category": CATEGORY, "p": float(out["p"][0]), "tier": str(out["tier"][0]),
                "why": out["concept"][0], "group": out["group"][0]}


# ── ADR-14 head seam: what imgtag.moderation.load_heads() asks every track for ────────


class ViolenceHead:
    """`score(embeddings, images, ids) -> [{category, p, tier, ...}]`.

    Costs one text-tower batch ONCE per (model, prompt-set), cached to
    ``~/.imgtag/models/<model_sha>/violence-<spec_sha>.npz`` — an index run on the 8GB
    target then loads a ~250KB file instead of a text tower (ADR-5's resident-set law).
    """

    category = CATEGORY
    wants_images = False        # dispatcher hint: never re-decode for this track
    calibrated = False          # no labelled positives exist on this machine
    enforcement_ready = False   # ADR-14: stays false until tau is fitted on ground truth

    def __init__(self, scorer: ViolenceScorer, model_sha: str, model_id: str = ""):
        self.scorer, self.model_sha = scorer, model_sha
        self.model_id = model_id or f"violence-prompts-{spec_sha()}"

    def probs(self, embeddings):
        """Arrays for a whole dataset: (p, tier) — what b-daemon's track_scores() wants."""
        out = self.scorer.score(embeddings)
        return out["p"], out["tier"]

    def score(self, embeddings, images=None, ids=None) -> list[dict]:
        out = self.scorer.score(embeddings)
        return [{"category": CATEGORY, "p": round(float(p), 4), "tier": str(t),
                 "why": w, "group": g, "model_id": self.model_id,
                 "calibrated": False, "enforcement_ready": False}
                for p, t, w, g in zip(out["p"], out["tier"], out["concept"], out["group"])]


def _cache_path(model_sha: str, root=None) -> Path:
    from ..core.store import imgtag_home

    d = (Path(root) if root else imgtag_home() / "models") / model_sha
    return d / f"violence-{spec_sha()}.npz"


def load_violence_head(profile=None, backend=None, root=None, config: dict | None = None):
    """Loader called by `imgtag.moderation.load_heads(profile)`. None = track unavailable.

    Never raises: a missing model or an unreadable cache means "no violence track on this
    machine", which the indexer reports honestly, rather than a broken index run.
    """
    try:
        from ..core import models as _models

        if backend is None:
            name = (profile or {}).get("model") or (profile or {}).get("backend") \
                or _models.DEFAULT_BACKEND
            backend = _models.load_backend(name, profile or {}, vision=False)
        cache = _cache_path(backend.model_sha, root)
        sev, vio, groups, ctx, bg = prompts()
        if cache.is_file():
            z = np.load(cache)
            sc = ViolenceScorer(severe=z["severe"], violent=z["violent"], groups=groups,
                                context=z["context"], bg=z["bg"], names=vio, ctx_names=ctx,
                                pol=policy(config))
        else:
            sc = ViolenceScorer.build(backend, config=config)
            cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache.parent / (cache.stem + ".tmp.npz")   # np.savez appends .npz
            np.savez(tmp, severe=sc.severe, violent=sc.violent, context=sc.context, bg=sc.bg)
            tmp.replace(cache)                               # atomic, ADR-6 discipline
        return ViolenceHead(sc, backend.model_sha, getattr(backend, "model_id", ""))
    except Exception:                                        # never break an index run
        return None


def track_spec() -> dict:
    """The `violence` entry of src/imgtag/data/moderation.json (v2 schema, conductor-owned
    file, this key owned by track-violence).

    `negatives` are subtracted; `policy_neighbours` MUST NOT be — the CONTEXT bank is
    published there as well as under `review` so a consumer that only annotates (rather
    than tiers) still sees the staged/clinical twins by name.
    """
    sev, vio, _, ctx, bg = prompts()
    return {
        "label": "violence / abuse",
        "alert": sev,
        "violation": vio,
        "review": ctx,
        "negatives": bg,
        "policy_neighbours": ctx,
        "templates": list(TEMPLATES),
        "scorer": "margin",
        "platt": [PLATT_A, PLATT_B],
        "tau_alert": TAU_ALERT,
        "tau_violation": TAU_VIOLATION,
        "tau_review": TAU_REVIEW,
        # NOT "fitted": tau is a false-positive budget, never a recall fit. search.py
        # gates only on `calibration == "fitted"`, which is exactly the behaviour this
        # track wants — it must never gate until labelled ground truth exists.
        "calibration": "fp-budget",
        "enforcement_ready": False,
        "spec_sha": spec_sha(),
    }
