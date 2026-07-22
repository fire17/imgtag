"""DRUGS moderation track — zero-shot prompt-ensemble detector (VISION-ADDENDA 12:33Z).

OWNER: track-drugs. Sibling tracks own their own files; `moderation.json`'s `drugs` key is
the only shared surface this lane writes.

WHAT THIS IS, HONESTLY
----------------------
A **recall-first triage** scorer, not a classifier. It answers "does this image look like
it belongs in a drug-context review queue?" — nothing stronger. The category is
context-heavy (a pharmacy pill bottle and a recreational staging of the same pills differ
by *intent*, which is not in the pixels) and NO drug-labelled corpus exists on disk, so the
honest numbers we can publish are:

  * FP/specificity on 5k real photos with human labels (COCO val2017 / LVIS) — MEASURED.
  * recall on a 26-image LVIS smoking-paraphernalia proxy + 10 OI syringe images — MEASURED
    but tiny; a proxy for the category, not the category.
  * recall on illicit-drug imagery (cocaine, heroin kits, cannabis buds, bongs) — **NOT
    MEASURABLE HERE**. Zero labelled positives exist in any corpus we are allowed to use.

Everything in `research/track-drugs.md` is labelled measured-vs-not on exactly that split.

SCORING (ADR-3 §2 — probability space, and the commissioned background-margin experiment)
-----------------------------------------------------------------------------------------
Each concept is a prompt ENSEMBLE: the mean of its templated text embeddings, L2-normalized
(classic CLIP zero-shot ensembling — cuts single-phrasing noise). FOUR banks, and keeping
them apart is the whole design:

    CONCEPTS          drug-context concepts, grouped by subcategory  → VIOLATION tier
    TOBACCO           cigarettes / cigars / vapes / hookah           → REVIEW tier (ADR-14)
    BACKGROUND        visually DISTINCT lookalikes (snow, plumbing, ferns) → SUBTRACTED
    POLICY_NEIGHBOURS visually IDENTICAL benign twins (a clinical syringe) → NEVER
                      subtracted; they only annotate a flag for the reviewer

    margin = max_cos(image, CONCEPTS) - max_cos(image, BACKGROUND)

The margin is the feature ADR-3 commissioned ("score minus max over K generic negative
prompts"): measured AP 0.726 on the drug slice, vs 0.31 for the mean-vector form
search.py uses today, and vs 0.04 if POLICY_NEIGHBOURS are subtracted too — that collapse
is why the two negative banks exist. p = sigmoid(A·margin + B) fitted on the labelled
slice; `flagged = p >= tau`.

TOBACCO is a POLICY KNOB, not a fact. ADR-14 (user ruling 12:50Z) puts it at the REVIEW
tier, so a cigarette is surfaced but never counted as a drugs violation. The knob lives in
CONFIG — `moderation.json` → `categories.drugs.tobacco_tier` ∈ review|violation|none — so a
changed ruling is a one-word edit with no retrain and no re-embedding. See AMBIGUITIES for
what is still un-ruled; the user rules, we do not guess.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np

CATEGORY = "drugs"

# Templates for the ensemble. Three is enough — measured gain past three was noise.
TEMPLATES = ("a photo of {}.", "a close-up photo of {}.", "{}")

# ── positive bank: drug-context concepts, grouped so `why` can name the subcategory ──
CONCEPTS: dict[str, list[str]] = {
    "cannabis": [
        "marijuana buds",
        "a cannabis plant with leaves and buds",
        "a bag of marijuana buds",
        "a jar of cannabis buds",
        "a grinder full of ground marijuana",
        "a person smoking a marijuana joint",
        "a rolled joint and rolling papers",
    ],
    "smoking_apparatus": [
        "a glass bong for smoking marijuana",
        "a glass bong with a bowl and a stem",  # NOT "water pipe": fires on plumbing (measured)
        "a glass pipe for smoking drugs",
        "a crack pipe",
        "a meth pipe with smoke",
        "aluminium foil with burnt drug residue",
    ],
    "powder": [
        "lines of white powder cocaine on a mirror with a rolled banknote",
        "a pile of white powder drugs on a table",
        "small plastic baggies of white powder",
        "a digital scale weighing white powder in a bag",
        "a brick of packaged cocaine",
    ],
    "pills_context": [
        "a pile of colorful ecstasy pills",
        "loose pills scattered on a table next to a plastic baggie",
        "recreational drug pills in a plastic bag",
        "sheets of LSD blotter tabs",
    ],
    "injection": [
        "a syringe with a spoon and a lighter for heroin",
        "a used needle and drug injection kit",
        "a syringe and a tourniquet on an arm",
    ],
    "psychedelics": [
        "dried psilocybin magic mushrooms",
        "a bag of dried magic mushrooms",
    ],
    "paraphernalia": [
        "drug paraphernalia laid out on a table",
        "illegal drugs and drug equipment",
    ],
}

# ── tobacco = the REVIEW tier (ADR-14 / VISION-ADDENDA 12:50Z ruling: "Tobacco / vaping /
# smoking imagery → Flag for review"). Scored as its own bank with its own threshold, so a
# cigarette never counts as a drugs VIOLATION and never gets silently dropped either. ──
TOBACCO: list[str] = [
    "a person smoking a cigarette",
    "a lit cigarette in an ashtray",
    "a cigar",
    "an e-cigarette vape pen",
    "a hookah water pipe with shisha",
]

# ── TWO negative banks, and the split is the whole design (measured, see §Measured) ──
#
# BACKGROUND — visually DISTINCT from drug imagery, but lexically/texturally confusable to
# CLIP: snow reads as cocaine, plumbing reads as a "water pipe", any leaf reads as cannabis.
# These are SUBTRACTED (max-pooled): a real drug photo does not match them strongly, so the
# subtraction costs no recall and centres the score across corpora.
BACKGROUND: list[str] = [
    # generic anchors — "is this just an ordinary photo?"
    "a photograph of a scene",
    "a photo of an ordinary everyday object",
    "a person",
    "an indoor room",
    "an outdoor landscape",
    "food on a plate",
    "an animal",
    "a vehicle",
    # white-powder lookalikes (COCO FP: a ski slope scored as cocaine)
    "fresh snow on the ground",
    "a snowy mountain slope",
    "sugar spilled on a kitchen counter",
    "flour on a baking table",
    "salt in a bowl",
    "baking soda",
    "chalk dust on a blackboard",
    "spices in small bowls",
    # pipe / tube lookalikes (COCO FP: bathroom plumbing scored as a bong)
    "plumbing pipes on a wall",
    "a bathroom with tiles",
    "electrical cables and wires",
    "laboratory glassware",
    # green lookalikes (the Unsplash 'hemp'/'weed' keyword noise is 90% this)
    "a green houseplant in a pot",
    "fresh basil and parsley herbs",
    "a leafy green plant",
    "a fern",
    "loose leaf tea",
    "a bowl of edible mushrooms",
    # smoke / fire that is not drugs
    "incense sticks burning",
    "a birthday cake with lit candles",
    "a campfire",
    "a barbecue grill with smoke",
]

# POLICY_NEIGHBOURS — visually IDENTICAL to positives; only intent/context separates them
# (a clinical syringe *is* a syringe). Subtracting these destroys the signal it is meant to
# refine — MEASURED: AP 0.58 → 0.04 when they are max-subtracted. So they are NEVER
# subtracted. They are scored only to ANNOTATE a flag ("nearest neighbour: pharmacy shelf"),
# which is what lets a reviewer clear a false alarm in one glance, and to make the policy
# questions in AMBIGUITIES concrete.
POLICY_NEIGHBOURS: list[str] = [
    "a pharmacy shelf with medicine boxes",
    "a prescription pill bottle",
    "a medicine cabinet with vitamins and supplements",
    "a doctor giving a vaccine injection in a clinic",
    "an insulin pen for diabetes",
    "a medical syringe on a sterile tray",
    "a first aid kit",
    "a glass of beer or wine",
]

# ── fitted on the labelled slices (research/track-drugs.md §Measured; numbers written by
# scripts/eval_drugs.py, never by hand). p = sigmoid(A * margin + B).
#   VIOLATION tier — fitted on 18 hand-verified drug images vs COCO val2017 negatives.
#   REVIEW tier    — fitted on the 26-image LVIS tobacco/medicine slice (human labels).
PLATT_A, PLATT_B = 105.2162, -6.6182
TAU = 0.0191              # violation: recall-first — 18/18 on the drug slice, 1.54% FP
TAU_REVIEW = 0.0316       # review (tobacco): set by a 1% FP budget, NOT by recall (see FIT)
TAU_PRECISION = 0.0373    # the alternative violation point: recall .94, 0.91% FP
TIER_MARGIN = 0.01        # violation must beat the tobacco bank by this (see score())
FIT = {
    "model": "pecore-s16-384-fp32",
    "feature": "margin = max(positive concepts) - max(background concepts)",
    "corpus": "5000 COCO val2017 + 328 Unsplash keyword-probe images",
    "violation": "AP 0.726 · recall .944 at 1% FP · fitted on 18 hand-verified drug "
    "images vs 5145 negatives. tau=0.0191 → 18/18 recall, 79/5145 (1.54%) flagged.",
    "tier_arbitration": "violation requires the drug bank to beat the tobacco bank by "
    "TIER_MARGIN: drug-slice violations 18/18 -> 15/18 (the other 3 are people smoking, "
    "surfaced at review), COCO violations 79 -> 46 (0.9%), tobacco-keyword photos scored "
    "as violations 22/128 -> 10/128, and the vape-exhale case demotes to review.",
    "review": "WEAK AND SAID SO: tobacco recall at a 1% FP budget is 0.17 (LVIS "
    "smoking labels) / 0.15 (Unsplash smoking photos). A cigarette is usually a "
    "20-pixel object and whole-image embeddings do not see it.",
    "caveat": "small positive sets (18 violation / 36 review). Wide-CI estimates on ONE "
    "corpus, not a benchmark. Recall on cocaine/heroin/meth imagery specifically is "
    "UNMEASURED — no labelled image of it exists in any corpus we may use.",
}

# ── policy questions only the user can answer (do not guess these) ──
# #1 and #2 were RULED on 2026-07-22 12:50Z (VISION-ADDENDA) and are recorded as settled.
AMBIGUITIES = [
    "1. TOBACCO — RULED 12:50Z: 'flag for review'. Implemented as the review tier; "
    "cigarettes/cigars/vapes never count as a drugs violation.",
    "2. VAPE vs. cannabis vape pen: visually near-identical, so both land at review tier. "
    "A cannabis cartridge that looks like a nicotine one CANNOT be separated — accepted "
    "consequence of ruling #1.",
    "3. ALCOHOL: currently NOT a drug track at all (beer/wine/spirits never flag). "
    "Say the word and it becomes its own track — do not fold it into drugs.",
    "4. LEGAL CANNABIS (dispensary shelf, medical marijuana, CBD product shots): pixels "
    "cannot distinguish legal from illegal. Default: flags. Accept or exempt?",
    "5. MEDICAL SYRINGES / vaccination / insulin / IV drips: default NOT flagged "
    "(confusable bank). Enforcement on a health-topic site may want the opposite.",
    "6. PRESCRIPTION MEDICINE (pill bottles, blister packs, pharmacy shots): default NOT "
    "flagged. 'Pills spilled on a table' is the grey zone and WILL sometimes flag.",
    "7. DRUG-AWARENESS / harm-reduction / news / anti-drug campaign imagery: visually "
    "identical to the thing it depicts. Flags. Needs a human, always.",
    "8. HISTORICAL / ARTISTIC (opium-den painting, poppy fields, Amsterdam street shots, "
    "hemp rope, hemp-seed food): edge cases with no visual tell.",
    "9. KITCHEN POWDERS (sugar/flour/salt) and CULINARY MUSHROOMS: in the confusable bank, "
    "but a white-powder-on-a-dark-surface photo is genuinely ambiguous to any model.",
    "10. WHAT DOES 'FLAGGED' DO? review queue vs. auto-hide. This detector is calibrated "
    "for a review QUEUE (recall-first). Auto-hide needs a second, much higher threshold.",
]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))


# ── the user-rulable knobs live in CONFIG, not in code ────────────────────────────────
# A ruling ("vaping is/ isn't a violation on my sites") must be a moderation.json edit —
# never a retrain, never a code change. The prompt banks are the same either way, so
# flipping `tobacco_tier` re-labels flags instantly with no re-embedding of anything.
TOBACCO_TIERS = ("review", "violation", "none")
DEFAULTS = {"tau": None, "tau_review": None, "tobacco_tier": "review"}


def policy(config: dict | None = None) -> dict:
    """Effective policy = module defaults ← moderation.json `categories.drugs` ← argument.

    Unknown/invalid values fall back to the default rather than raising: a typo in a
    config file must not take moderation offline.
    """
    p = {"tau": TAU, "tau_review": TAU_REVIEW, "tobacco_tier": DEFAULTS["tobacco_tier"]}
    if config is None:
        try:
            from pathlib import Path

            data = Path(__file__).resolve().parent.parent / "data" / "moderation.json"
            config = json.loads(data.read_bytes())["categories"]["drugs"]
        except (OSError, ValueError, KeyError):
            config = {}
    for k in ("tau", "tau_review"):
        try:
            if config.get(k) is not None:
                p[k] = float(config[k])
        except (TypeError, ValueError):
            pass
    if config.get("tobacco_tier") in TOBACCO_TIERS:
        p["tobacco_tier"] = config["tobacco_tier"]
    return p


def prompts(tobacco: bool = False) -> tuple[list[str], list[str], list[str]]:
    """(positive concepts, their subcategory labels, BACKGROUND concepts).

    `tobacco=True` promotes the review-tier bank into the VIOLATION bank — for a site
    whose rules do ban smoking imagery outright. It is never subtracted in either mode:
    a joint and a cigarette look alike, so subtracting one costs real recall on the other.
    """
    pos, groups = [], []
    for g, cs in CONCEPTS.items():
        pos += cs
        groups += [g] * len(cs)
    if tobacco:
        pos += TOBACCO
        groups += ["tobacco"] * len(TOBACCO)
    return pos, groups, BACKGROUND


def spec_sha(tobacco: bool = False) -> str:
    p, _, n = prompts(tobacco)
    return hashlib.sha256("\x00".join(list(TEMPLATES) + p + n).encode()).hexdigest()[:16]


def concept_vectors(backend, texts: list[str]) -> np.ndarray:
    """One L2-normalized vector per concept = mean over TEMPLATES. One text batch."""
    flat = [t.format(c) for c in texts for t in TEMPLATES]
    emb = np.asarray(backend.embed_texts(flat), np.float32).reshape(len(texts), len(TEMPLATES), -1)
    v = emb.mean(1)
    return np.ascontiguousarray(v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12))


@dataclass
class DrugsScorer:
    """Holds the two prompt banks for one model. Build once, score any dataset.

    >>> s = DrugsScorer.build(backend)          # one text-tower batch
    >>> out = s.score(snapshot.emb)             # one [N, P] matmul
    """

    pos: np.ndarray
    groups: list[str]
    bg: np.ndarray
    names: list[str]
    tob: np.ndarray | None = None
    neighbours: np.ndarray | None = None
    tobacco: bool = False
    pol: dict = field(default_factory=policy)

    @classmethod
    def build(cls, backend, tobacco: bool | None = None, neighbours: bool = True,
              config: dict | None = None) -> "DrugsScorer":
        """`tobacco` is derived from CONFIG unless explicitly overridden (see `policy`)."""
        pol = policy(config)
        if tobacco is None:
            tobacco = pol["tobacco_tier"] == "violation"
        p, groups, bg = prompts(tobacco)
        return cls(pos=concept_vectors(backend, p), groups=groups,
                   bg=concept_vectors(backend, bg), names=p, tobacco=tobacco,
                   tob=None if tobacco else concept_vectors(backend, TOBACCO),
                   neighbours=concept_vectors(backend, POLICY_NEIGHBOURS) if neighbours else None,
                   pol=pol)

    def score(self, emb: np.ndarray) -> dict:
        """emb: [N, D] L2-normalized image embeddings → per-image drugs payload.

        Two tiers per ADR-14: `p` is the violation probability (illegal drugs and
        paraphernalia), `p_review` the tobacco/vape one. `tier` is the ADR-14 carrier.
        Arrays throughout — one dataset costs two small matmuls.
        """
        tau, tau_r = self.pol["tau"], self.pol["tau_review"]
        emb = np.asarray(emb, np.float32)
        cp = emb @ self.pos.T                       # [N, P]
        bg = (emb @ self.bg.T).max(1)               # [N]
        best = cp.argmax(1)
        margin = cp.max(1) - bg
        p = _sigmoid(PLATT_A * margin + PLATT_B)
        out = {
            "category": CATEGORY,
            "p": p,
            "flagged": p >= tau,
            "margin": margin,
            "concept": [self.names[i] for i in best],
            "group": [self.groups[i] for i in best],
            "tau": tau,
            "calibration": "proxy-fitted",   # never claim more than FIT says
        }
        if self.tob is not None:
            mt = (emb @ self.tob.T).max(1) - bg
            pr = _sigmoid(PLATT_A * mt + PLATT_B)
            out["p_review"] = pr
            second = "none" if self.pol["tobacco_tier"] == "none" else "review"
            # ARBITRATION (measured): a joint and a cigarette look alike, so `p >= tau`
            # alone made every vape exhale a VIOLATION — the exact v0 failure b-daemon hit
            # at p=.964. A violation now also has to be explained BETTER by the drug bank
            # than by the tobacco bank, by TIER_MARGIN. Nothing leaves the queue: what
            # loses the arbitration lands at review, where ADR-14 says a human decides.
            wins = margin >= mt + TIER_MARGIN
            out["tier"] = np.where(out["flagged"] & wins, "violation",
                                   np.where((pr >= tau_r) | out["flagged"], second, "none"))
            out["flagged"] = out["tier"] == "violation"
            out["tau_review"] = tau_r
        else:   # tobacco promoted into the violation bank: no separate review tier
            out["tier"] = np.where(out["flagged"], "violation", "none")
        if self.neighbours is not None:      # review aid, never part of the score
            nb = emb @ self.neighbours.T
            out["nearest_benign"] = [POLICY_NEIGHBOURS[i] for i in nb.argmax(1)]
        return out

    def per_image(self, emb: np.ndarray, i: int) -> dict:
        """The contract shape the brief asks for: {category, p, flagged} for one row."""
        out = self.score(np.asarray(emb, np.float32)[i : i + 1])
        r = {"category": CATEGORY, "p": float(out["p"][0]),
             "flagged": bool(out["flagged"][0]), "tier": str(out["tier"][0]),
             "why": out["concept"][0], "group": out["group"][0]}
        if "nearest_benign" in out:
            r["nearest_benign"] = out["nearest_benign"][0]
        return r


# ── ADR-14 head seam: what imgtag.moderation.load_heads() asks every track for ─────────


class DrugsHead:
    """`score(embeddings, images, ids) -> [{category, p, tier}]` — the pipeline contract.

    Costs one text-tower batch ONCE per (model, prompt-set), cached to
    ``~/.imgtag/models/<model_sha>/drugs-<spec_sha>.npz`` — an index run on the 8GB target
    then loads a ~200KB file instead of a text tower (ADR-5's resident-set law).
    """

    wants_images = False

    def __init__(self, scorer: DrugsScorer, model_sha: str):
        self.scorer, self.model_sha = scorer, model_sha

    def probs(self, embeddings):
        """Arrays for a whole dataset: (p, tier) — what b-daemon's track_scores() wants."""
        out = self.scorer.score(embeddings)
        return out["p"], out["tier"]

    def score(self, embeddings, images=None, ids=None) -> list[dict]:
        out = self.scorer.score(embeddings)
        return [{"category": CATEGORY, "p": round(float(p), 4), "tier": str(t),
                 "why": w, "group": g}
                for p, t, w, g in zip(out["p"], out["tier"], out["concept"], out["group"])]


def _cache_path(model_sha: str, tobacco: bool, root=None):
    from pathlib import Path

    from ..core.store import imgtag_home

    d = (Path(root) if root else imgtag_home() / "models") / model_sha
    return d / f"drugs-{spec_sha(tobacco)}.npz"


def load_drugs_head(profile=None, backend=None, tobacco: bool | None = None, root=None,
                    config: dict | None = None):
    """Loader called by `imgtag.moderation.load_heads(profile)`. None = track unavailable.

    Never raises: a missing model or an unreadable cache means "no drugs track on this
    machine", which the indexer reports honestly, rather than a broken index run.
    """
    try:
        from ..core import models as _models

        pol = policy(config)
        if tobacco is None:
            tobacco = pol["tobacco_tier"] == "violation"
        if backend is None:
            name = (profile or {}).get("model") or _models.DEFAULT_BACKEND
            backend = _models.load_backend(name, profile or {}, vision=False)
        cache = _cache_path(backend.model_sha, tobacco, root)
        p, groups, _ = prompts(tobacco)
        if cache.is_file():
            z = np.load(cache)
            sc = DrugsScorer(pos=z["pos"], groups=groups, bg=z["bg"], names=p,
                             tob=z["tob"] if "tob" in z else None,
                             neighbours=z["nb"] if "nb" in z else None, tobacco=tobacco,
                             pol=pol)
        else:
            sc = DrugsScorer.build(backend, tobacco=tobacco, config=config)
            cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache.parent / (cache.stem + ".tmp.npz")  # np.savez appends .npz itself
            np.savez(tmp, pos=sc.pos, bg=sc.bg, **({"tob": sc.tob} if sc.tob is not None else {}),
                     **({"nb": sc.neighbours} if sc.neighbours is not None else {}))
            tmp.replace(cache)                    # atomic, ADR-6 discipline
        return DrugsHead(sc, backend.model_sha)
    except Exception:                             # never break an index run
        return None


def track_spec() -> dict:
    """The `drugs` entry of src/imgtag/data/moderation.json (v2 schema, conductor-owned
    file, this key owned by track-drugs). `negatives` are subtracted; `policy_neighbours`
    MUST NOT be — subtracting them was measured to collapse AP 0.58 → 0.04."""
    pol = policy({})          # module defaults, not whatever is on disk right now
    pos, _, bg = prompts()
    return {
        "label": "drugs / drug paraphernalia",
        "violation": pos,
        "review": list(TOBACCO),
        "negatives": bg,
        "policy_neighbours": POLICY_NEIGHBOURS,
        "templates": list(TEMPLATES),
        "scorer": "margin",
        "platt": [PLATT_A, PLATT_B],
        "tau": TAU,
        "tau_review": TAU_REVIEW,
        # THE user-rulable knob: "review" (ADR-14 default) | "violation" | "none".
        # Changing it here re-tiers every flag with no retrain and no re-embedding.
        "tobacco_tier": pol["tobacco_tier"],
        "calibration": "proxy-fitted",
        "enforcement_ready": False,
        "spec_sha": spec_sha(),
        "fit": FIT,
        "policy_questions": len(AMBIGUITIES),
    }
