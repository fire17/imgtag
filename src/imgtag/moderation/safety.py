"""SAFETY moderation track — people lying down, escalated by danger context.

OWNER: track-safety. Sibling tracks own their own files; `moderation.json`'s `safety` key
is the only shared surface this lane writes.

VISION-ADDENDA 13:20Z (verbatim): "make another track to identify people lying down (even
if part of their body is obstructed) and even higher flagging if either detecting injury,
things broken, distruction distress high stress or anything dangorous".

This is WELFARE monitoring, not rule enforcement. The other three tracks answer "did
someone break the site rules?"; this one answers "does someone need help?". That flips the
cost asymmetry: a missed nudity flag embarrasses a site, a missed person-down could be a
person on the floor of a warehouse at 3am. Recall-first is not a preference here.

WHAT THIS IS, HONESTLY
----------------------
A **recall-first triage** scorer over the embedding the index already computed. No pose
model, no second network, no new dependency (ADR-7/ADR-9/ADR-10 — the target is a shared
8GB CPU Linux box whose co-tenants are sacred). Measured on COCO val2017 against ground
truth built from TWO independent human-annotated sources (see `scripts/eval_safety.py`):

  * lying-person detection — MEASURED, AP 0.53 on 41 human-consensus positives vs 4058
    negatives (chance 0.010). Recall 0.78 at 2% FP.
  * OCCLUSION ROBUSTNESS — MEASURED, and the headline result: recall on the heavily
    occluded stratum (positives where COCO's own annotator could NOT place a torso —
    covered by blankets, cropped to a head) is 0.79 at the shipped threshold, BELOW the
    fully-visible stratum's 0.92 but ABOVE the geometry-ambiguous stratum's 0.64 — i.e.
    hiding the body does NOT make it the worst case. At a slightly looser (5%-FP) point
    the hidden stratum reaches 0.93, matching fully-visible. Whole-image embeddings do not
    need visible joints — and a keypoint pose model scores ~0 on this stratum BY
    CONSTRUCTION (no keypoints to place). This is exactly the property the user asked for,
    and the one a pose model structurally cannot have (see research/track-safety.md §5).
  * danger-context detection — MEASURED but WEAK and on a TINY slice: AP 0.144 on the 19
    danger images COCO val2017 contains. Directional only.
  * the ALERT tier (person-down IN danger context) — essentially UNMEASURABLE: COCO
    val2017 contains exactly ONE image of a person lying down in a danger context (an
    injured man, facial wounds, image 000000354307) out of 5000; the shipped scorer DOES
    tier it `alert` with danger_why "an injured person bleeding". One true positive is a
    sanity check, not a precision estimate. The other 40 lying people are benign
    (sleeping, beach, sofa). Alert precision is UNMEASURED — the tier is shipped
    recall-first and `enforcement_ready` is false.

SCORING (ADR-3 §2 — probability space; the commissioned background-margin feature)
----------------------------------------------------------------------------------
Two INDEPENDENT margins, never mixed into one number, because they answer two questions
and the user asked for an escalation, not a blend:

    LYING     person-down concepts                       -> the review tier
    BACKGROUND upright/empty/animal-lying look-alikes     -> SUBTRACTED from lying
    DANGER    injury / breakage / destruction / distress  -> the escalation signal
    DANGER_BACKGROUND ordinary scenes, benign fire        -> SUBTRACTED from danger
    BENIGN_CONTEXT  beach, bed, sofa, sunbathing          -> NEVER subtracted; annotates

    p_lying  = sigmoid(A_l * (max_cos(LYING) - max_cos(BACKGROUND)) + B_l)
    p_danger = sigmoid(A_d * (max_cos(DANGER) - max_cos(DANGER_BACKGROUND)) + B_d)

    tier = alert   if p_lying >= tau AND p_danger >= tau_danger
           review  if p_lying >= tau
           none    otherwise

BENIGN_CONTEXT follows the drugs track's `policy_neighbours` law: a sunbather and a
collapsed person are the SAME pose, separated only by context, so subtracting the benign
bank would destroy the signal it is meant to refine. It is scored to ANNOTATE a flag
("nearest benign context: sunbathing on a beach") so a reviewer clears a false alarm in
one glance — which matters more here than anywhere, because the measured truth is that
the overwhelming majority of lying people in real photos are FINE.

ANIMALS. The requirement says PEOPLE. Cats and dogs lying on furniture were the largest
measured false-positive class, so animal-lying phrasings sit in BACKGROUND. Full-set AP
rises 0.453 -> 0.534 with them, but that gain is NOT stable across a 2-fold split
(fold A +0.14, fold B -0.02; n=41 cannot resolve it). They are shipped on SPEC grounds —
an animal is not a person — and the honest headline range is AP 0.45-0.53. Never claim
0.53 alone.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CATEGORY = "safety"

#: Two templates. A third was measured on the drugs track as noise; the same holds here.
TEMPLATES = ("a photo of {}.", "{}")

# ── positive bank: person-down concepts ───────────────────────────────────────────────
# Deliberately spans the whole benign..emergency range. The track's job is to FIND the
# pose; the danger bank decides how loudly to shout. Occlusion phrasings ("partially
# covered", "the legs of") are in the bank because the requirement names occlusion — and
# they cost nothing, since the measured occluded-stratum recall is already 0.93.
LYING: list[str] = [
    "a person lying on the ground",
    "a person lying down",
    "a person lying on their back",
    "a person sleeping",
    "a person asleep in bed",
    "someone collapsed on the floor",
    "a fallen person lying on the ground",
    "a person lying on a sofa",
    "a person reclining on a couch",
    "a person sunbathing lying on the beach",
    "the legs of a person lying on the floor",
    "a partially covered person lying down",
]

# ── SUBTRACTED negatives: what an upright/empty/animal scene looks like ───────────────
# Every entry earns its place from a MEASURED false-positive class, not from intuition.
# NOTE (measured, and the reason this bank is short): adding "surfer/jockey/skateboarder"
# phrasings — the crouched-athlete FP class — made AP WORSE (0.453 -> 0.440). Prompt banks
# do not reward accretion. Anything added here must beat the base on a re-run of
# scripts/eval_safety.py, or it does not ship.
BACKGROUND: list[str] = [
    "a person standing",
    "a person walking",
    "people standing upright",
    "a group of people standing",
    "a person sitting on a chair",
    "an empty room",
    "an outdoor landscape",
    "a photo of an object",
    "a close-up of food",
    "a street with cars",
    "a building exterior",
    "an animal",
    # animal-lying: the largest measured FP class (a cat asleep on a bed is not a person)
    "a cat sleeping on a bed",
    "a dog lying on a couch",
    "an animal lying on the ground",
    "a sleeping cat curled up",
]

# ── the escalation bank: the user's list, made concrete ────────────────────────────────
# "injury, things broken, distruction distress high stress or anything dangorous"
DANGER: list[str] = [
    "a car crash accident",
    "a wrecked crashed car",
    "a burning building on fire",
    "fire and smoke",
    "blood on the ground",
    "shattered broken glass",
    "a scene of destruction and debris",
    "an injured person bleeding",
    "an emergency scene with an ambulance",
    "a collapsed building and rubble",
    "a natural disaster scene",
    "a person in distress",
    "a flooded street",
    "a firefighter fighting a fire",
    "an industrial accident",
    "a damaged wrecked vehicle",
    "a person receiving first aid",
]

# Benign fire/smoke is the obvious danger FP class — a campfire, a candle, a barbecue and
# a sunset are not emergencies.
DANGER_BACKGROUND: list[str] = [
    "a normal everyday scene",
    "an ordinary photograph",
    "a person standing",
    "a clean tidy room",
    "food on a plate",
    "a landscape",
    "a building exterior",
    "people at an event",
    "an animal",
    "a vehicle parked normally",
    "a sunny day outdoors",
    "a bedroom",
    "a campfire",
    "a sunset sky",
    "a barbecue grill",
    "a candle flame",
]

# ── ground-level FALSE-POSITIVE killers (b-daemon, MEASURED on unsplash-demo 2026-07-22) ─
# b-daemon's naive draft put 12 images in `alert`, top two: a photographer's boots beside
# ice shot downward (p=0.959), and a night puddle reflection with legs at the frame edge
# (p=0.858). The failure mode: a whole-image embedding keys on GROUND-LEVEL VIEWPOINT +
# limbs, not on a person's posture. These phrases give the subtraction something to latch
# that exact viewpoint onto. Added to the SPEC negatives (reader path, corpus-relative
# thresholding, no Platt dependency); folding them into the code BACKGROUND bank + re-fit
# is the first post-quiet-window task (ledger) — done unmeasured they would invalidate the
# frozen Platt constants above, so they are NOT in BACKGROUND yet.
GROUND_LEVEL_FP: list[str] = [
    "boots and feet photographed from above on the ground",
    "a ground-level view of legs and shoes",
    "a puddle reflection on wet pavement at night",
    "an empty floor or pavement",
    "shoes on the floor with no person",
]

# ── reader-path ALERT bank: person-down AND danger in ONE phrase (b-daemon's exceedance
# model). The daemon's reader assigns each row to the single tier it exceeds by the most —
# it cannot express the code head's two-margin conjunction. A COMBINED phrase scores high
# only when BOTH concepts are present, so exceedance-over-this-bank approximates the AND:
# an injured-lying image exceeds these more than the benign `review` bank; a peaceful
# sleeper exceeds `review`; boots-on-ice exceeds neither. Used ONLY by track_spec() /
# moderation.json — the Python head keeps the two independent margins (LYING × DANGER).
ALERT_PHRASES: list[str] = [
    "an injured person lying on the ground",
    "a person collapsed on the floor",
    "an unconscious person lying down",
    "a person lying on the ground next to a wrecked car",
    "a person lying injured in the street after an accident",
    "a person lying on the ground bleeding",
    "a fallen person on the ground amid debris",
    "a person lying motionless at a disaster scene",
    "a person receiving first aid lying on the ground",
    "a person lying hurt on the floor",
]

# ── NEVER subtracted (drugs-track `policy_neighbours` law) ─────────────────────────────
# A sunbather and an unconscious person are the same pose. Subtracting these would remove
# most true positives. Scored only to explain a flag to the human who reviews it.
BENIGN_CONTEXT: list[str] = [
    "a person sunbathing on a beach towel",
    "a person napping on a sofa at home",
    "a person asleep in their own bed",
    "a baby sleeping in a crib",
    "a person doing yoga lying on a mat",
    "a person lying on the grass in a park",
    "people lying on towels at a crowded beach",
    "a person getting a massage on a table",
    "a patient resting in a hospital bed",
    "a person lying down posing for a photo shoot",
]

# ── fitted on COCO val2017; numbers written by scripts/eval_safety.py, never by hand ───
# p = sigmoid(A * margin + B).
LYING_PLATT_A, LYING_PLATT_B = 99.3941, -5.2050
DANGER_PLATT_A, DANGER_PLATT_B = 37.8043, -5.3679

#: review tier — the 2%-FP operating point: recall 0.780. Recall-first but not reckless:
#: 5% FP buys only +0.05 recall (0.829) for 2.5x the queue. Both are in FIT; the knob is
#: in moderation.json, so an operator who wants the wider net changes one number.
TAU_REVIEW = 0.0695
#: alert escalation — the 10%-FP danger point (recall 0.526). Deliberately LOOSE: it only
#: ever re-tiers images ALREADY flagged as person-down, so its false positives cost a
#: louder label on an image a human is looking at anyway, while its misses cost an
#: unescalated emergency. Recall-first, exactly as the brief requires at the alert level.
TAU_DANGER = 0.0076

FIT = {
    "model": "pecore-s16-384-fp32",
    "feature": "margin = max(positive prompts) - max(background prompts), per bank",
    "corpus": "COCO val2017 (5000 images), labels from person_keypoints_val2017 geometry "
              "+ captions_val2017 human consensus",
    "lying": "AP 0.534 full-set (0.453 without the animal negatives; the difference is "
             "NOT fold-stable — see module docstring). 41 human-consensus positives vs "
             "4058 doubly-verified negatives; chance AP 0.010. Operating points: "
             "recall .659 @1% FP · .780 @2% FP (SHIPPED) · .829 @5% FP · .878 @10% FP.",
    "occlusion": "recall at the shipped tau_review by visibility stratum: fully visible "
                 "(n=13) 0.923 · geometry-ambiguous (n=14) 0.643 · NO usable keypoints, "
                 "i.e. heavily occluded (n=14) 0.786. Hiding the body does NOT make it "
                 "the worst case (the ambiguous stratum is); at a 5%-FP point the hidden "
                 "stratum reaches 0.929, matching fully-visible. A keypoint pose model "
                 "scores ~0 on the no-keypoint stratum by construction.",
    "danger": "WEAK AND SAID SO: AP 0.144 on the 19 danger images in COCO val2017. "
              "recall .316 @2% FP · .421 @5% FP · .526 @10% FP (shipped). A 19-image "
              "slice cannot support a stronger claim than 'better than chance'.",
    "alert": "UNMEASURED. COCO val2017 contains ZERO person-down-in-danger images "
             "(lying ∩ danger = 0 of 5000); all 41 lying positives are benign. Alert "
             "precision and recall are therefore unknown, not estimated.",
    "caveat": "41 positives is a small set with wide CIs, on ONE corpus, with the "
              "negative bank chosen after inspecting this corpus's false positives. "
              "Treat every number as directional until a labelled safety corpus exists.",
}

# ── policy questions only the user can answer (do not guess these) ─────────────────────
AMBIGUITIES = [
    "1. SLEEPING PEOPLE. The measured truth is that ~all lying people in real photos are "
    "fine — asleep in bed, sunbathing, napping on a sofa. This track flags them at "
    "`review` by design (a welfare monitor that ignores still bodies is useless), but on "
    "a site full of beach or bedroom imagery that is a large queue. Options: keep, or "
    "raise tau, or (best) rule that only person-down WITH danger surfaces at all.",
    "2. DANGER WITHOUT A PERSON. A burning building or a wrecked car with nobody visible "
    "currently produces NO flag: the user's words gate escalation on people lying down. "
    "`danger_alone_tier` in moderation.json defaults to \"none\" for that reason. For a "
    "site-safety monitor 'review' is arguably right — the user rules, we do not guess.",
    "3. WHAT COUNTS AS DISTRESS. 'high stress' is in the requirement but is largely "
    "invisible to a whole-image embedding: a distressed face is a handful of pixels. "
    "Facial-expression signals are NOT implemented and NOT measured. Do not read the "
    "danger score as a distress detector.",
    "4. MEDICAL CONTEXT. A patient in a hospital bed is a person lying down, often with "
    "danger-adjacent surroundings. Currently: review, and the benign-context annotator "
    "usually names it. A care-home deployment probably wants the opposite ruling.",
    "5. ANIMALS. Excluded by spec (the requirement says people). A site monitoring animal "
    "welfare would want them included — one prompt-bank move, no retrain.",
    "6. CHILDREN PLAYING / YOGA / EXERCISE / PHOTO SHOOTS. Lying on purpose, perfectly "
    "safe, visually identical to collapse. In the benign-context bank as an annotation "
    "only; they will still reach the review queue.",
    "7. WHAT DOES A FLAG DO? This is calibrated for a review QUEUE. An ALERT that pages a "
    "human at 3am needs a second, much higher threshold AND a labelled corpus — neither "
    "exists yet. `enforcement_ready` is false for exactly this reason.",
    "8. NO LABELLED SAFETY CORPUS EXISTS on this machine. The single largest improvement "
    "available is 200-500 hand-labelled real person-down images from the deployment "
    "site itself; every number here would tighten by an order of magnitude.",
]

TIERS = ("alert", "review", "none")
DANGER_ALONE_TIERS = ("none", "review")
DEFAULTS = {"tau": TAU_REVIEW, "tau_danger": TAU_DANGER, "danger_alone_tier": "none"}


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))


def lying_prob(margin) -> np.ndarray:
    return _sigmoid(LYING_PLATT_A * np.asarray(margin) + LYING_PLATT_B)


def danger_prob(margin) -> np.ndarray:
    return _sigmoid(DANGER_PLATT_A * np.asarray(margin) + DANGER_PLATT_B)


def policy(config: dict | None = None) -> dict:
    """Effective policy = module defaults ← moderation.json `categories.safety` ← argument.

    Invalid values fall back to the default rather than raising: a typo in a config file
    must never take safety monitoring offline.
    """
    p = dict(DEFAULTS)
    if config is None:
        try:
            data = Path(__file__).resolve().parent.parent / "data" / "moderation.json"
            config = json.loads(data.read_bytes())["categories"]["safety"]
        except (OSError, ValueError, KeyError):
            config = {}
    for k in ("tau", "tau_danger"):
        try:
            if config.get(k) is not None:
                p[k] = float(config[k])
        except (TypeError, ValueError):
            pass
    if config.get("danger_alone_tier") in DANGER_ALONE_TIERS:
        p["danger_alone_tier"] = config["danger_alone_tier"]
    return p


def spec_sha() -> str:
    parts = list(TEMPLATES) + LYING + BACKGROUND + DANGER + DANGER_BACKGROUND + ALERT_PHRASES
    return hashlib.sha256("\x00".join(parts).encode()).hexdigest()[:16]


def concept_vectors(backend, texts: list[str]) -> np.ndarray:
    """One L2-normalized vector per concept = mean over TEMPLATES. One text batch."""
    flat = [t.format(c) for c in texts for t in TEMPLATES]
    emb = np.asarray(backend.embed_texts(flat), np.float32)
    emb = emb.reshape(len(texts), len(TEMPLATES), -1).mean(1)
    return np.ascontiguousarray(emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12))


@dataclass
class SafetyScorer:
    """Holds the prompt banks for one model. Build once, score any dataset.

    >>> s = SafetyScorer.build(backend)     # one text-tower batch
    >>> out = s.score(snapshot.emb)         # four small matmuls
    """

    lying: np.ndarray
    bg: np.ndarray
    danger: np.ndarray
    dbg: np.ndarray
    benign: np.ndarray | None = None
    pol: dict = field(default_factory=policy)

    @classmethod
    def build(cls, backend, benign: bool = True, config: dict | None = None) -> "SafetyScorer":
        return cls(
            lying=concept_vectors(backend, LYING),
            bg=concept_vectors(backend, BACKGROUND),
            danger=concept_vectors(backend, DANGER),
            dbg=concept_vectors(backend, DANGER_BACKGROUND),
            benign=concept_vectors(backend, BENIGN_CONTEXT) if benign else None,
            pol=policy(config),
        )

    # ── the two independent margins ──────────────────────────────────────────────────
    def lying_margin(self, emb: np.ndarray) -> np.ndarray:
        emb = np.asarray(emb, np.float32)
        return (emb @ self.lying.T).max(1) - (emb @ self.bg.T).max(1)

    def danger_margin(self, emb: np.ndarray) -> np.ndarray:
        emb = np.asarray(emb, np.float32)
        return (emb @ self.danger.T).max(1) - (emb @ self.dbg.T).max(1)

    def tiers(self, p_lying: np.ndarray, p_danger: np.ndarray) -> np.ndarray:
        """ADR-14 tier vocabulary, `alert` highest. Escalation, never a blend.

        A person down IS the flag; danger only decides how loudly. `danger_alone_tier`
        (default "none") is the un-ruled policy question in AMBIGUITIES #2 — a burning
        building with nobody visible produces no safety flag until the user rules.
        """
        down = np.asarray(p_lying) >= self.pol["tau"]
        danger = np.asarray(p_danger) >= self.pol["tau_danger"]
        tier = np.where(down & danger, "alert", np.where(down, "review", "none"))
        if self.pol["danger_alone_tier"] == "review":
            tier = np.where(~down & danger, "review", tier)
        return tier

    def score(self, emb: np.ndarray) -> dict:
        """emb: [N, D] L2-normalized image embeddings → per-image safety payload."""
        emb = np.asarray(emb, np.float32)
        ml, md = self.lying_margin(emb), self.danger_margin(emb)
        pl, pd = lying_prob(ml), danger_prob(md)
        cl = emb @ self.lying.T
        out = {
            "category": CATEGORY,
            # `p` is the PERSON-DOWN probability — the thing the track detects. The danger
            # score rides alongside as the escalation reason, never averaged in.
            "p": pl,
            "p_danger": pd,
            "tier": self.tiers(pl, pd),
            "margin": ml,
            "why": [LYING[i] for i in cl.argmax(1)],
            "tau": self.pol["tau"],
            "tau_danger": self.pol["tau_danger"],
            "calibration": "proxy-fitted",     # never claim more than FIT says
        }
        cd = emb @ self.danger.T
        out["danger_why"] = [DANGER[i] for i in cd.argmax(1)]
        if self.benign is not None:            # review aid, never part of the score
            nb = emb @ self.benign.T
            out["nearest_benign"] = [BENIGN_CONTEXT[i] for i in nb.argmax(1)]
        return out

    def per_image(self, emb: np.ndarray, i: int) -> dict:
        """The contract shape for one row."""
        out = self.score(np.asarray(emb, np.float32)[i : i + 1])
        r = {"category": CATEGORY, "p": float(out["p"][0]),
             "p_danger": float(out["p_danger"][0]), "tier": str(out["tier"][0]),
             "why": out["why"][0]}
        if str(out["tier"][0]) == "alert":
            r["danger_why"] = out["danger_why"][0]
        if "nearest_benign" in out:
            r["nearest_benign"] = out["nearest_benign"][0]
        return r


# ── ADR-14 head seam: what imgtag.moderation.load_heads() asks every track for ─────────


class SafetyHead:
    """`score(embeddings, images, ids) -> [{category, p, tier}]` — the pipeline contract.

    Costs one text-tower batch ONCE per (model, prompt-set), cached to
    ``~/.imgtag/models/<model_sha>/safety-<spec_sha>.npz`` — an index run on the 8GB
    target loads a ~150KB file instead of a text tower (ADR-5's resident-set law).
    """

    wants_images = False               # dispatcher hint: never re-decode for this track
    category = CATEGORY

    def __init__(self, scorer: SafetyScorer, model_sha: str):
        self.scorer, self.model_sha = scorer, model_sha

    @property
    def enforcement_ready(self) -> bool:
        return False                   # no labelled safety corpus exists — FIT says so

    def probs(self, embeddings):
        """Arrays for a whole dataset: (p, tier) — what b-daemon's track_scores() wants."""
        out = self.scorer.score(embeddings)
        return out["p"], out["tier"]

    def score(self, embeddings, images=None, ids=None) -> list[dict]:
        out = self.scorer.score(embeddings)
        rows = []
        for i in range(len(out["p"])):
            r = {"category": CATEGORY, "p": round(float(out["p"][i]), 4),
                 "tier": str(out["tier"][i]), "why": out["why"][i],
                 "p_danger": round(float(out["p_danger"][i]), 4)}
            if r["tier"] == "alert":
                r["danger_why"] = out["danger_why"][i]
            if "nearest_benign" in out:
                r["nearest_benign"] = out["nearest_benign"][i]
            rows.append(r)
        return rows


def _cache_path(model_sha: str, root=None) -> Path:
    from ..core.store import imgtag_home

    d = (Path(root) if root else imgtag_home() / "models") / model_sha
    return d / f"safety-{spec_sha()}.npz"


def load_safety_head(profile=None, backend=None, root=None, config: dict | None = None):
    """Loader called by `imgtag.moderation.load_heads(profile)`. None = track unavailable.

    Never raises: a missing model or an unreadable cache means "no safety track on this
    machine", which the indexer reports honestly, rather than a broken index run.
    """
    try:
        from ..core import models as _models

        pol = policy(config)
        if backend is None:
            name = (profile or {}).get("model") or _models.DEFAULT_BACKEND
            backend = _models.load_backend(name, profile or {}, vision=False)
        cache = _cache_path(backend.model_sha, root)
        if cache.is_file():
            z = np.load(cache)
            sc = SafetyScorer(lying=z["lying"], bg=z["bg"], danger=z["danger"],
                              dbg=z["dbg"], benign=z["benign"] if "benign" in z else None,
                              pol=pol)
        else:
            sc = SafetyScorer.build(backend, config=config)
            cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache.parent / (cache.stem + ".tmp.npz")   # np.savez appends .npz
            np.savez(tmp, lying=sc.lying, bg=sc.bg, danger=sc.danger, dbg=sc.dbg,
                     **({"benign": sc.benign} if sc.benign is not None else {}))
            tmp.replace(cache)                                # atomic, ADR-6 discipline
        return SafetyHead(sc, backend.model_sha)
    except Exception:                                         # never break an index run
        return None


def track_spec(ship_alert: bool = False) -> dict:
    """The `safety` entry of src/imgtag/data/moderation.json (v2 schema, conductor-owned
    file, this key owned by track-safety).

    `negatives` are subtracted; `policy_neighbours` MUST NOT be — a sunbather and a
    collapsed person are the same pose, so subtracting the benign bank removes the
    positives (the drugs track measured the identical collapse, AP 0.58 -> 0.04).

    THE ALERT TIER IS WITHHELD (`ship_alert=False`, the default and current state). Ruling
    2026-07-22 (team-lead): ship the REVIEW tier now — person-down is COCO-measured and
    occlusion-robust (§3), and review is a human queue where recall-first is correct — but
    keep `alert` OUT until a clean TP set shows separation (the safetyprobe round measured
    alert_tp-vs-benign-lying AP 0.454, CI95 [0.349, 0.590]; lower bound < 0.5 = not shippable,
    and 4/4 diagnostic views were mislabelled — the labels, not the threshold, are the
    blocker; research/track-safety.md §5b). The `alert` block is ABSENT, not zeroed, so the
    reader derives tiers `[review]` only. Flip `ship_alert=True` once a person-presence-
    filtered pull clears the 0.5 gate — the ALERT_PHRASES / platt_danger / tau_danger are
    ready and validated by test_safety.py; only the go decision is withheld.
    """
    pol = policy({})                   # module defaults, not whatever is on disk now
    spec = {
        "label": "safety / person lying down",
        # `review` = person-down (benign or not). b-daemon's reader assigns by EXCEEDANCE
        # over tier prompt-sets; with only `review` present it derives the tier `[review]`.
        "review": list(LYING),
        # negatives are subtracted as background. GROUND_LEVEL_FP kills b-daemon's measured
        # boots/puddle FPs; animal + standing/seated come from BACKGROUND.
        "negatives": list(BACKGROUND) + list(DANGER_BACKGROUND) + list(GROUND_LEVEL_FP),
        "policy_neighbours": list(BENIGN_CONTEXT),
        "templates": list(TEMPLATES),
        "scorer": "margin",
        # NOTE: no `violation` key — this track's tiers are alert|review|none only.
        "platt": [LYING_PLATT_A, LYING_PLATT_B],
        "tau": pol["tau"],
        # proxy-fitted, NOT fitted: no labelled safety corpus exists (COCO has n=1
        # person-down-in-danger). The reader must corpus-relative-threshold this, not
        # trust tau — enforcement_ready stays false until a real fit on real positives.
        "calibration": "proxy-fitted",
        "enforcement_ready": False,
        "policy_questions": len(AMBIGUITIES),
        "fit": FIT,
    }
    if ship_alert:
        # b-daemon's reader: `alert` = person-down-AND-danger COMBINED phrases (highest
        # ADR-14 tier, 13:20Z), written to CONTRAST `review`, not nest. Withheld by default.
        spec["alert"] = list(ALERT_PHRASES)
        spec["platt_danger"] = [DANGER_PLATT_A, DANGER_PLATT_B]
        spec["tau_danger"] = pol["tau_danger"]
        spec["danger_alone_tier"] = pol["danger_alone_tier"]
    else:
        spec["alert_withheld"] = ("alert_tp vs benign-lying AP 0.454 CI95 [0.349,0.590] "
                                  "< 0.5 ship gate; labels not clean TPs (§5b). review "
                                  "tier ships; alert re-armed via track_spec(ship_alert=True).")
    spec["spec_sha"] = spec_sha()
    return spec
