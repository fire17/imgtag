"""Sports CONTENT track — "is this image about sport?" (VISION-ADDENDA 13:23Z).

Not a moderation track. Nothing here is a policy breach; the answer is a *content
label*, so the tier vocabulary is ``match`` | ``none`` (ADR-14, `match` added 13:23Z)
and the result NEVER enters a moderation total — b-daemon routes ``content_track``
heads into their own ``content`` bucket.

Instrument (TRACKS.md T2 rung 1, the only unconditionally-allowed one): a prompt
ensemble over the SAME embedding the index already computed. Score =
``max cos(image, sport prompt) - max cos(image, background prompt)``. One [N,D]·[D,k]
matmul per image, no second model, no new dependency. The background subtraction is
what makes the number comparable across images (weapons.py established this shape;
this track reuses it deliberately — see ADR-3).

Sport is *nameable* content, which is exactly where an open-vocabulary text tower is
strongest, so unlike nudity this track needs no dedicated model — and it gets a free
bonus the moderation tracks do not have: the argmax prompt names WHICH sport, so a
match carries a ``sport`` label ("tennis", "skiing", …) at zero extra cost.

Two paths, same seam:
  * ``SportsHead``          — Platt-calibrated on labeled ground truth (COCO val2017
                              sports supercategory, 938/5000 — research/track-sports.md),
                              τ fitted on a held-out split. Loadable from the machine
                              profile alone: the prompt matrices are baked into the
                              fitted file, so no text tower runs at load.
  * ``ZeroShotSportsHead``  — same margin, no fit: a monotone squash and a default τ.
                              ``calibrated`` is False and its p is a RANKING, not a
                              probability.

Borderline sports (chess, hiking, fishing, …) are a documented, configurable ruling —
see ``BORDERLINE_PROMPTS``. They are OUT by default.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..core.tags import fit_platt, platt_apply, prompt_ensemble_sha

CATEGORY = "sports"
DATA = Path(__file__).resolve().parent.parent / "data" / "moderation"

#: Sport vocabulary, grouped by the label a match reports. Two families, both needed:
#: EQUIPMENT (what COCO's `sports` supercategory and LVIS actually annotate — the part
#: this track can be measured on) and ACTIVITY/SCENE (a stadium crowd, a marathon, a
#: gym — sport that COCO cannot score because it annotates no object for it).
#: Group keys are the reported ``sport``; every prompt in a group is scored and the
#: group's best cosine represents it.
SPORT_PROMPTS: dict[str, list[str]] = {
    # ── equipment: the ten COCO `sports` children ────────────────────────────
    "frisbee": ["a photo of a frisbee", "a person throwing a frisbee",
                "a dog catching a flying disc"],
    "skiing": ["a photo of skis", "a skier going down a snowy slope",
               "a person skiing with ski poles", "a downhill ski race"],
    "snowboarding": ["a photo of a snowboard", "a snowboarder jumping in the air",
                     "a person riding a snowboard down a mountain"],
    "kite": ["a photo of a kite flying in the sky", "a person flying a kite on the beach",
             "a kitesurfer on the water"],
    "baseball": ["a photo of a baseball bat", "a photo of a baseball glove",
                 "a baseball player swinging a bat", "a pitcher throwing a baseball",
                 "a baseball game on a diamond"],
    "skateboarding": ["a photo of a skateboard", "a skateboarder doing a trick",
                      "a person riding a skateboard at a skate park"],
    "surfing": ["a photo of a surfboard", "a surfer riding a wave",
                "a person carrying a surfboard on the beach"],
    "tennis": ["a photo of a tennis racket", "a tennis player hitting a ball",
               "a tennis match on a court", "a photo of a tennis ball"],
    # ── equipment: LVIS sports categories COCO lumps into `sports ball` ──────
    "soccer": ["a photo of a soccer ball", "a soccer player kicking a ball",
               "a football match on a green pitch", "a goalkeeper diving for the ball"],
    "basketball": ["a photo of a basketball", "a basketball player dunking",
                   "a basketball game on an indoor court", "a basketball hoop and backboard"],
    "volleyball": ["a photo of a volleyball", "a beach volleyball game over a net",
                   "a volleyball player spiking the ball"],
    "american football": ["a photo of an american football",
                          "american football players tackling on a field",
                          "a photo of a football helmet"],
    "golf": ["a photo of a golf club", "a golfer swinging on a fairway",
             "a golf ball on a tee"],
    "hockey": ["a photo of a hockey stick", "ice hockey players on the rink",
               "a field hockey match"],
    "table tennis": ["a ping-pong table with paddles", "a person playing table tennis"],
    "bowling": ["a photo of a bowling ball", "a bowling alley with pins"],
    "skating": ["a photo of ice skates", "a figure skater on the ice",
                "a person on roller skates"],
    "water sports": ["a photo of a kayak", "a person paddling a kayak in a river",
                     "a rowing team in a racing boat", "a water skier behind a boat"],
    # ── activity / scene: sport with no annotatable object ───────────────────
    "swimming": ["a swimmer in a swimming pool", "a swimming race in lanes",
                 "a person doing the freestyle stroke in a pool"],
    "running": ["a marathon race with many runners", "a runner sprinting on a track",
                "people jogging in a road race with race numbers"],
    "gym": ["a person lifting weights in a gym", "a barbell and dumbbells in a gym",
            "a weightlifter at a squat rack", "a person on a treadmill in a fitness club"],
    "climbing": ["a rock climber on a cliff", "a person climbing an indoor climbing wall",
                 "a climber with a harness and ropes"],
    "cycling": ["a road cycling race with a peloton", "a mountain biker on a trail",
                "a cyclist racing in a helmet and jersey"],
    "martial arts": ["two boxers fighting in a boxing ring", "a photo of boxing gloves",
                     "martial artists sparring in a dojo", "a judo throw on a mat"],
    "gymnastics": ["a gymnast performing on a balance beam",
                   "a gymnast doing a floor routine"],
    "equestrian": ["a horse and rider jumping an obstacle", "a horse race at a racetrack"],
    "motorsport": ["a race car on a circuit", "a motorcycle road race"],
    "stadium game": ["a packed sports stadium during a game",
                     "a crowd watching a match in an arena",
                     "athletes competing in front of spectators",
                     "a scoreboard at a sports game"],
    "team sport": ["a sports team celebrating a victory",
                   "athletes in matching jerseys on a field",
                   "a referee on the pitch during a match"],
}

#: Prompts that are their own group but are NOT sport by default. The user ruling on
#: each is *configurable*, not baked: these read as sport to some sites and as leisure
#: to others (research/track-sports.md §Rulings). Default OUT — they are appended to
#: the background bank so they actively push the margin DOWN, which is the honest
#: default for a content filter someone will use to find "sports photos".
#: Turn them on with ``SportsHead.build(..., borderline=True)`` / spec
#: ``categories.sports.borderline: "match"``.
BORDERLINE_PROMPTS: dict[str, list[str]] = {
    "chess": ["a chess board with pieces", "two people playing chess at a table"],
    "hiking": ["a hiker with a backpack on a mountain trail",
               "people trekking through the countryside"],
    "fishing": ["a person fishing with a rod by a lake"],
    "darts / pool": ["a game of darts on a dartboard", "a person playing pool at a table"],
    "esports": ["people playing video games at a computer tournament"],
    "yoga": ["a person doing a yoga pose on a mat"],
    "dance": ["dancers performing on a stage"],
}

#: Background bank — GENERIC SCENE PROMPTS ONLY, and that is a measured decision, not
#: laziness. Its one job is to make the score a CONTRAST rather than an absolute cosine.
#:
#: ⚠️ Do NOT add scene-level hard negatives here. weapons.py's bank is full of them and
#: they work there, because a kitchen knife is genuinely far from a rifle. On THIS track
#: the equivalent prompts ("an empty sports stadium", "a snowy mountain landscape",
#: "a beach with people sunbathing", "a swimming pool with nobody in it") sit right next
#: to the true positives, so ``max(background)`` rises on the images we want to keep and
#: the margin collapses. Measured on COCO val2017 (research/track-sports.md §3.2):
#: AP 0.9321 → 0.9037 and recall-at-precision-0.80 0.947 → 0.889 when they are added.
#: The FP classes they were meant to kill (stadium concert, crowd) already sit at a low
#: ≤0.11 weak-label match rate without them (corpus-dependent; research/track-sports.md §6).
NEGATIVE_PROMPTS = [
    "a photo of a person",
    "a portrait of a person",
    "a photo of a landscape",
    "a photo of food",
    "a photo of an animal",
    "a photo of a building",
    "a photo of a city street",
    "a photo of an indoor room",
    "a photo of furniture",
    "a photo of a document",
    "a photo of a computer",
    "an abstract pattern",
]

#: Uncalibrated MARGIN default for the zero-shot path — a FLAG BUDGET, not an operating
#: point (τ_match gates in margin space; a 0.5 *probability* here would flag nothing, since
#: margins top out near 0.16). The fitted head replaces this with a measured margin-τ.
ZERO_SHOT_TAU = 0.05

#: A content track is not enforcement, so it is NOT recall-first (weapons/nudity are —
#: a missed weapon goes live, a missed sports photo just does not show up in a filter).
#: τ is fitted where precision and recall are both defensible: the smallest threshold
#: whose PRECISION reaches this target on held-out ground truth. Rationale in
#: research/track-sports.md §Threshold.
MATCH_PRECISION = 0.80


def flat_prompts(borderline: bool = False) -> tuple[list[str], list[str]]:
    """(prompt texts, per-prompt sport label) for the positive bank, in matrix order."""
    groups = dict(SPORT_PROMPTS)
    if borderline:
        groups.update(BORDERLINE_PROMPTS)
    texts, labels = [], []
    for label, prompts in groups.items():
        texts.extend(prompts)
        labels.extend([label] * len(prompts))
    return texts, labels


def background_prompts(borderline: bool = False) -> list[str]:
    """Background bank. Borderline sports live here unless the ruling turns them on."""
    if borderline:
        return list(NEGATIVE_PROMPTS)
    return NEGATIVE_PROMPTS + [p for ps in BORDERLINE_PROMPTS.values() for p in ps]


def margin(emb: np.ndarray, pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """max-positive cosine minus max-background cosine. Higher = more sport."""
    e = np.asarray(emb, np.float32)
    return (e @ pos.T).max(1) - (e @ neg.T).max(1)


def squash(m: np.ndarray, platt: list | None = None) -> np.ndarray:
    """Margin → [0,1]. With a fitted Platt pair this is a probability; without one it
    is a MONOTONE convenience and must not be read as one."""
    if platt:
        return platt_apply(m, platt)
    return 1.0 / (1.0 + np.exp(-np.asarray(m, np.float64) * 20.0))


def tier_of(margin_score: float, tau_match: float) -> str:
    """Content-track tier. Gates in MARGIN space (`margin >= tau_match`), because that is
    the shared serving contract: b-daemon's fitted reader (search.py `_margin_p` / the
    `fitted` branch) does `margin - tau >= 0` and treats Platt as a COSMETIC reported
    probability. τ_match is therefore a margin threshold, NOT a probability — fitting it in
    probability space silently ships a track the reader flags 0 images for (the max margin
    never reaches a 0.18 probability-τ). ``none`` is a real answer, not a missing one."""
    return "match" if margin_score >= tau_match else "none"


def _pack(a: np.ndarray) -> dict:
    """Prompt matrices → fp16 base64 (ADR: fp16 is bit-equivalent WEIGHT STORAGE for
    retrieval, and a text embedding is the same kind of vector). Keeps the head ~165KB
    instead of ~600KB of JSON floats, self-contained so no text tower runs at load."""
    a = np.ascontiguousarray(a, np.float16)
    return {"b64": base64.b64encode(a.tobytes()).decode(), "shape": list(a.shape)}


def _unpack(d) -> np.ndarray:
    """Read a matrix back as float32 (legacy list form still accepted)."""
    if isinstance(d, dict):
        return np.frombuffer(base64.b64decode(d["b64"]), np.float16).reshape(d["shape"]).astype(np.float32)
    return np.asarray(d, np.float32)


# ── fitting ───────────────────────────────────────────────────────────────────
def tau_for_precision(p: np.ndarray, y: np.ndarray, target: float = MATCH_PRECISION) -> float:
    """Smallest threshold whose precision >= target (so recall is maximal at that
    precision). Falls back to the max-precision threshold if the target is unreachable
    — the caller sees the achieved precision in ``metrics`` and must not assume it hit."""
    p = np.asarray(p, np.float64)
    y = np.asarray(y, bool)
    order = np.argsort(-p)
    ys = y[order]
    prec = np.cumsum(ys) / np.arange(1, len(ys) + 1)
    ok = np.flatnonzero(prec >= target)
    if len(ok) == 0:
        return float(p[order[0]])
    return float(p[order[ok[-1]]])


def prf(p: np.ndarray, y: np.ndarray, tau: float) -> dict:
    """precision / recall / f1 / match-rate at tau."""
    yp = np.asarray(p) >= tau
    y = np.asarray(y, bool)
    tp, fp, fn = int((yp & y).sum()), int((yp & ~y).sum()), int((~yp & y).sum())
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"tau": float(tau), "tp": tp, "fp": fp, "fn": fn, "precision": prec,
            "recall": rec, "f1": 2 * prec * rec / max(prec + rec, 1e-12),
            "match_rate": float(yp.mean())}


def average_precision(p: np.ndarray, y: np.ndarray) -> float:
    """Area under the precision-recall curve (prevalence-independent ranking quality)."""
    order = np.argsort(-np.asarray(p))
    ys = np.asarray(y, bool)[order]
    tp = np.cumsum(ys)
    return float(((tp / np.arange(1, len(ys) + 1)) * ys).sum() / max(ys.sum(), 1))


def spread(p: np.ndarray) -> dict:
    """Quantiles of the probability column. b-daemon's lesson: a track whose p is
    saturated at 0/1 is a broken fit even when its AP looks fine — publish the spread."""
    q = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    return {"quantiles": {str(k): round(float(v), 4)
                          for k, v in zip(q, np.quantile(np.asarray(p, np.float64), q))},
            "frac_below_0.02": float(np.mean(np.asarray(p) < 0.02)),
            "frac_above_0.98": float(np.mean(np.asarray(p) > 0.98))}


# ── head ──────────────────────────────────────────────────────────────────────
@dataclass
class SportsHead:
    """Fitted sports content head for ONE model (dispatcher-facing).

    ``score(embeddings, images, ids) -> list[dict]`` — one dict per record. Embeddings
    are all it needs; ``images``/``ids`` are accepted for contract symmetry and ignored,
    which is exactly why this track costs one matmul and no second decode.
    """

    model_id: str
    model_sha: str
    dim: int
    pos: np.ndarray                    # [k,D] L2-normed sport-prompt embeddings
    neg: np.ndarray                    # [m,D] L2-normed background-prompt embeddings
    labels: list                       # [k] sport name per positive prompt
    tau_match: float
    platt: list | None = None          # [A,B] margin -> probability
    borderline: bool = False
    prompts_sha: str = ""
    metrics: dict = field(default_factory=dict)

    category = CATEGORY
    wants_images = False               # dispatcher hint: never re-decode for this track
    content_track = True               # b-daemon: route to `content`, not moderation

    @property
    def calibrated(self) -> bool:
        return bool(self.platt) and bool(self.metrics.get("held_out"))

    #: A content label never gates enforcement — it is not a policy breach (ADR-14).
    enforcement_ready = False

    def margins(self, emb: np.ndarray) -> np.ndarray:
        emb = np.asarray(emb, np.float32)
        if emb.ndim != 2 or emb.shape[1] != self.dim:
            raise ValueError(f"sports head expects [N,{self.dim}], got {emb.shape}")
        return margin(emb, self.pos, self.neg)

    def probs(self, emb: np.ndarray) -> np.ndarray:
        """The reported (cosmetic) probability — Platt over the margin. NOT the gate:
        the tier is decided on the raw margin vs τ_match (see ``score``)."""
        return squash(self.margins(emb), self.platt)

    def score(self, embeddings, images=None, ids=None) -> list[dict]:
        emb = np.asarray(embeddings, np.float32)
        if emb.ndim != 2 or emb.shape[1] != self.dim:
            raise ValueError(f"sports head expects [N,{self.dim}], got {emb.shape}")
        cp = emb @ self.pos.T
        m = cp.max(1) - (emb @ self.neg.T).max(1)      # margin — the GATED quantity
        p = squash(m, self.platt)                       # probability — cosmetic display
        best = cp.argmax(1)
        out = []
        for i in range(len(m)):
            tier = tier_of(float(m[i]), self.tau_match)   # gate in MARGIN space (reader contract)
            d = {"category": CATEGORY, "p": round(float(p[i]), 4), "tier": tier,
                 "model_id": self.model_id, "calibrated": self.calibrated,
                 "enforcement_ready": False, "content_track": True}
            if tier == "match":
                # ``label`` is the cross-track field b-daemon's reader emits (argmax
                # concept); ``sport`` is kept as this track's explicit alias.
                d["label"] = d["sport"] = self.labels[int(best[i])]
            out.append(d)
        return out

    def to_json(self) -> dict:
        # ``calibration``/``scorer`` are the keys b-daemon's spec reader honours off the
        # fitted file; the rest is what THIS module's loader needs to score without a
        # text tower. One file, two consumers — each ignores the other's extra keys.
        return {"category": CATEGORY, "model_id": self.model_id, "model_sha": self.model_sha,
                "dim": self.dim, "labels": list(self.labels),
                "pos": _pack(self.pos), "neg": _pack(self.neg),
                "tau_match": float(self.tau_match), "platt": self.platt,
                "scorer": "margin",
                "calibration": "fitted" if self.calibrated else "unfitted",
                "borderline": bool(self.borderline), "prompts_sha": self.prompts_sha,
                "metrics": self.metrics}

    @classmethod
    def from_json(cls, d: dict) -> SportsHead:
        return cls(model_id=d["model_id"], model_sha=d["model_sha"], dim=d["dim"],
                   pos=_unpack(d["pos"]), neg=_unpack(d["neg"]),
                   labels=d["labels"], tau_match=float(d["tau_match"]),
                   platt=d.get("platt"), borderline=bool(d.get("borderline", False)),
                   prompts_sha=d.get("prompts_sha", ""), metrics=d.get("metrics", {}))

    @classmethod
    def build(cls, backend, borderline: bool = False, tau_match: float = ZERO_SHOT_TAU,
              platt: list | None = None) -> SportsHead:
        """Embed the banks with ``backend``'s text tower. Uncalibrated until ``fit``."""
        texts, labels = flat_prompts(borderline)
        bg = background_prompts(borderline)
        # Round to fp16 in memory too, so the τ/Platt we fit are fit on the SAME values
        # that ship in the head file — no train/serve skew from the storage cast.
        pos = np.asarray(backend.embed_texts(texts), np.float16).astype(np.float32)
        neg = np.asarray(backend.embed_texts(bg), np.float16).astype(np.float32)
        return cls(model_id=getattr(backend, "name", "?"), model_sha=backend.model_sha,
                   dim=int(pos.shape[1]), pos=pos, neg=neg, labels=labels,
                   tau_match=tau_match, platt=platt, borderline=borderline,
                   prompts_sha=prompt_ensemble_sha(texts + bg))


def fit(head: SportsHead, emb: np.ndarray, y: np.ndarray, val: tuple | None = None,
        precision: float = MATCH_PRECISION) -> SportsHead:
    """Calibrate + fit τ. ``val`` = held-out (emb, y); τ is fitted THERE when given —
    fitting an operating point on the same scores you calibrated on is how a track
    quietly ships a precision it does not have.

    τ_match is fitted in **MARGIN space** — the space the reader gates in. Platt is fitted
    too but only feeds the reported (cosmetic) probability; it never decides a tier. AP and
    prf are computed on the margin (a monotone transform, so ranking-identical to the
    probability), and the reported-probability spread is published for the saturation check.
    """
    m = head.margins(emb)
    head.platt = fit_platt(m, y)
    ve, vy = val if val is not None else (emb, y)
    vm = head.margins(ve)                                  # MARGIN space — the reader's gate
    head.tau_match = tau_for_precision(vm, vy, precision)  # τ_match is a MARGIN threshold
    vp = squash(vm, head.platt)                            # reported probability (cosmetic)
    met = {"target_precision": precision, "tau_space": "margin", **prf(vm, vy, head.tau_match)}
    met.update(ap=average_precision(vm, vy), held_out=val is not None,
               n=int(len(vy)), n_pos=int(np.sum(vy)), spread=spread(vp))
    head.metrics = met
    return head


# ── persistence ───────────────────────────────────────────────────────────────
def head_path(model_id: str, root: Path | None = None) -> Path:
    """Keyed by BACKEND NAME so the dispatcher can find one from the machine profile
    alone, without paying a model load. model_sha inside is the integrity check."""
    return (root or DATA) / f"sports-{model_id}.json"


def save(head: SportsHead, root: Path | None = None) -> Path:
    p = head_path(head.model_id, root)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(head.to_json()))
    tmp.replace(p)
    return p


def load_head(model_id: str, root: Path | None = None,
              model_sha: str | None = None) -> SportsHead | None:
    p = head_path(model_id, root)
    if not p.is_file():
        return None
    h = SportsHead.from_json(json.loads(p.read_text()))
    if model_sha is not None and h.model_sha != model_sha:   # ADR-6's refusal shape
        raise ValueError(f"sports head model_sha {h.model_sha[:12]} != loaded model "
                         f"{model_sha[:12]} — re-fit the head, never score across models")
    return h


def load_sports_head(profile: dict | None = None) -> SportsHead | None:
    """Dispatcher entry point (imgtag.moderation.load_heads contract).

    None when no head has been fitted for this machine's backend — a missing track is
    simply not loaded and is reported by name, never a silent zero.
    """
    from ..core.models import DEFAULT_BACKEND
    model_id = (profile or {}).get("backend") or DEFAULT_BACKEND
    return load_head(model_id)


# ── zero-shot path (works on any backend, uncalibrated) ───────────────────────
class ZeroShotSportsHead:
    """Same margin, no fit. ``p`` is a RANKING, not a probability: ``calibrated`` is
    False forever and τ is a flag budget, not a measured operating point."""

    category = CATEGORY
    wants_images = False
    calibrated = False
    enforcement_ready = False
    content_track = True

    def __init__(self, backend, tau: float = ZERO_SHOT_TAU, borderline: bool = False):
        self.backend = backend
        self.model_id = f"zeroshot:{getattr(backend, 'name', '?')}"
        self.tau = tau
        self.borderline = borderline
        self._head = None

    def _lazy(self) -> SportsHead:
        if self._head is None:
            self._head = SportsHead.build(self.backend, self.borderline, self.tau)
        return self._head

    def probs(self, emb: np.ndarray) -> np.ndarray:
        return self._lazy().probs(emb)

    def score(self, embeddings, images=None, ids=None) -> list[dict]:
        out = self._lazy().score(embeddings, images, ids)
        for d in out:
            d["model_id"] = self.model_id
            d["calibrated"] = False
        return out
