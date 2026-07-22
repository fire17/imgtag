"""Weapons moderation track — recall-first site-rule enforcement.

VISION-ADDENDA 12:33Z: "we dont want images with nudity, weapons or drugs … these are
very important to indentify correctly". This track answers ONE question per image:
does it contain a weapon (firearm, blade-as-weapon, bow, ordnance)?

Design (ADR-3, ADR-7): the score is a function of the SAME embedding the index already
computed. No detector, no second model, no new dependency — one [N,D]·[D,k] matmul,
marginal cost ~0, and it runs on the 8GB CPU target unchanged (ADR-10).

Two scoring paths, both shipped:
  * ``WeaponsHead``          — a logistic head trained on Open Images `Weapon`-subtree
                               labels (measured; see research/track-weapons.md). Loaded
                               when a head exists for this machine's backend. Calibrated.
  * ``ZeroShotWeaponsHead``  — prompt-ensemble margin (max positive-prompt cos MINUS max
                               background-prompt cos). Works for ANY backend with no
                               training, but is UNCALIBRATED → ``review`` tier only,
                               ``enforcement_ready: false`` forever.

ADR-14 two tiers, both recall-first: τ is the SMALLEST threshold meeting a target recall
on the held-out split, never max-F1. Moderation misses cost more than false alarms — a
flagged image goes to a human, a missed one goes live. ``review`` is the wide net
(0.95 recall); ``violation`` is the confident band a site could act on unattended.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..core.tags import fit_platt, platt_apply, prompt_ensemble_sha

CATEGORY = "weapons"
DATA = Path(__file__).resolve().parent.parent / "data" / "moderation"

#: Weapon vocabulary — object names, carry/handling context, and ordnance. Deliberately
#: wide: recall-first. Each is scored with the model's own text tower at build time.
POSITIVE_PROMPTS = [
    "a photo of a gun",
    "a photo of a handgun",
    "a photo of a pistol",
    "a photo of a revolver",
    "a photo of a rifle",
    "a photo of an assault rifle",
    "a photo of a shotgun",
    "a photo of a machine gun",
    "a photo of a submachine gun",
    "a photo of a sniper rifle",
    "a photo of a firearm",
    "a person holding a gun",
    "a person aiming a firearm",
    "a person pointing a gun at the camera",
    "a soldier carrying a rifle",
    "a holstered pistol on a belt",
    "a photo of ammunition and bullets",
    "a photo of a knife used as a weapon",
    "a photo of a combat knife",
    "a photo of a hunting knife",
    "a photo of a dagger",
    "a photo of a switchblade",
    "a person holding a knife threateningly",
    "a photo of a machete",
    "a photo of a sword",
    "a photo of a katana",
    "a photo of a sabre",
    "a photo of a bayonet",
    "a photo of an axe as a weapon",
    "a photo of a bow and arrow",
    "a photo of a crossbow",
    "a photo of a hand grenade",
    "a photo of a bomb",
    "a photo of an explosive device",
    "a photo of a missile",
    "a photo of a rocket launcher",
    "a photo of a military tank",
    "a photo of a cannon",
    "a photo of an artillery gun",
    "a display of weapons",
]

#: Background prompts. Two jobs: (1) hard negatives that share a weapon's shape or
#: handling — the kitchen-knife / tool / toy / sports families are where an embedding
#: model actually fails; (2) generic scene prompts, so the margin is a real contrast
#: rather than an absolute cosine (ADR-3's commissioned background-margin experiment).
NEGATIVE_PROMPTS = [
    "a kitchen knife on a cutting board",
    "chopping vegetables with a kitchen knife",
    "a chef preparing food in a kitchen",
    "cutlery on a dining table",
    "a pair of scissors",
    "a hammer and nails",
    "a screwdriver",
    "a power drill",
    "a chainsaw cutting wood",
    "hand tools in a workshop",
    "a toy gun",
    "a plastic water pistol",
    "a toy soldier figurine",
    "a video game screenshot",
    "a baseball bat",
    "a hockey stick",
    "a tennis racket",
    "ski poles in the snow",
    "a camera with a long lens",
    "a flashlight",
    "an umbrella",
    "a guitar",
    "a fishing rod",
    "a metal pipe",
    "a construction site",
    "a firework display",
    "a photo of a person",
    "a photo of a landscape",
    "a photo of food",
    "a photo of an animal",
    "a photo of a car",
    "a photo of a building",
    "a photo of a document",
    "an abstract pattern",
]

#: Conservative zero-shot default, used only when no trained head exists for the model.
#: NOT a measured operating point for an arbitrary backend — see report §blind spots.
ZERO_SHOT_TAU = 0.5

#: ADR-14 recall targets per tier. `review` is the recall-first net (miss as little as
#: possible); `violation` is the confident band a site could act on without a human.
REVIEW_RECALL = 0.95
VIOLATION_RECALL = 0.60


def tier_of(p: float, tau_violation: float, tau_review: float) -> str:
    """ADR-14 tier for one probability. ``none`` is a real answer, not a missing one."""
    return "violation" if p >= tau_violation else "review" if p >= tau_review else "none"


@dataclass
class WeaponsHead:
    """Trained, calibrated weapons head for ONE model (dispatcher-facing, ADR-14).

    ``score(embeddings, images, ids) -> list[dict]`` — one dict per record. Embeddings are
    all it needs: ``images``/``ids`` are accepted for contract symmetry and ignored, which
    is exactly why this track costs one matmul and no second decode.
    """

    model_id: str
    model_sha: str
    dim: int
    w: np.ndarray                      # [D] logistic weights over the L2-normed embedding
    b: float
    platt: list                        # [A, B] mapping the logit to a probability
    tau_violation: float
    tau_review: float
    prompts_sha: str = ""
    metrics: dict = field(default_factory=dict)

    category = CATEGORY
    wants_images = False               # dispatcher hint: never re-decode for this track

    @property
    def calibrated(self) -> bool:
        return bool(self.metrics.get("held_out"))

    @property
    def enforcement_ready(self) -> bool:
        """B-brief law (d): false until tau is fitted on labeled ground truth."""
        return self.calibrated and self.tau_review > 0.0

    def scores(self, emb: np.ndarray) -> np.ndarray:
        emb = np.asarray(emb, np.float32)
        if emb.ndim != 2 or emb.shape[1] != self.dim:
            raise ValueError(f"weapons head expects [N,{self.dim}], got {emb.shape}")
        return emb @ self.w + self.b

    def probs(self, emb: np.ndarray) -> np.ndarray:
        return platt_apply(self.scores(emb), self.platt)

    def _flag(self, p: float) -> dict:
        return {"category": CATEGORY, "p": round(float(p), 4),
                "tier": tier_of(p, self.tau_violation, self.tau_review),
                "model_id": self.model_id, "calibrated": self.calibrated,
                "enforcement_ready": self.enforcement_ready}

    def score(self, embeddings, images=None, ids=None) -> list[dict]:
        return [self._flag(v) for v in self.probs(embeddings)]

    def to_json(self) -> dict:
        return {"category": CATEGORY, "model_id": self.model_id, "model_sha": self.model_sha,
                "dim": self.dim, "w": [float(x) for x in self.w], "b": float(self.b),
                "platt": list(self.platt), "tau_violation": float(self.tau_violation),
                "tau_review": float(self.tau_review), "prompts_sha": self.prompts_sha,
                "metrics": self.metrics}

    @classmethod
    def from_json(cls, d: dict) -> WeaponsHead:
        return cls(model_id=d["model_id"], model_sha=d["model_sha"], dim=d["dim"],
                   w=np.asarray(d["w"], np.float32), b=float(d["b"]), platt=d["platt"],
                   tau_violation=float(d["tau_violation"]), tau_review=float(d["tau_review"]),
                   prompts_sha=d.get("prompts_sha", ""), metrics=d.get("metrics", {}))


# ── prompt ensemble (zero-shot path) ──────────────────────────────────────────
def prompt_matrices(backend) -> tuple[np.ndarray, np.ndarray]:
    """(positive[k,D], background[m,D]) text embeddings — L2-normalized by the backend."""
    return (backend.embed_texts(POSITIVE_PROMPTS).astype(np.float32),
            backend.embed_texts(NEGATIVE_PROMPTS).astype(np.float32))


def zero_shot_margin(emb: np.ndarray, pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """max-positive cosine minus max-background cosine, per image. Higher = more weapon.

    The subtraction is what makes the number comparable across images: absolute CLIP
    cosines are dominated by per-image norm/entropy effects, not by content.
    """
    e = np.asarray(emb, np.float32)
    return (e @ pos.T).max(1) - (e @ neg.T).max(1)


def zero_shot_prob(margin: np.ndarray, platt: list | None = None) -> np.ndarray:
    """Squash the margin to [0,1]. With a fitted Platt pair this is calibrated; without
    one it is a MONOTONE convenience only — never treat it as a probability."""
    if platt:
        return platt_apply(margin, platt)
    return 1.0 / (1.0 + np.exp(-np.asarray(margin, np.float64) * 20.0))


# ── training ──────────────────────────────────────────────────────────────────
def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = 1e-2, iters: int = 400,
                 lr: float = 1.0) -> tuple[np.ndarray, float]:
    """L2-regularized logistic regression, numpy only (ADR-7: no sklearn at runtime).

    Full-batch gradient descent with momentum on class-balanced weights — the positive
    class is the minority and recall is what we are buying.
    """
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64).ravel()
    n, d = x.shape
    npos, nneg = max(y.sum(), 1.0), max((1 - y).sum(), 1.0)
    sw = np.where(y > 0, n / (2 * npos), n / (2 * nneg))       # balanced sample weights
    w, b, vw, vb = np.zeros(d), 0.0, np.zeros(d), 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(x @ w + b)))
        g = sw * (p - y)
        gw, gb = x.T @ g / n + l2 * w, g.sum() / n
        vw, vb = 0.9 * vw - lr * gw, 0.9 * vb - lr * gb
        w, b = w + vw, b + vb
    return w.astype(np.float32), float(b)


def tau_for_recall(p: np.ndarray, y: np.ndarray, target: float = 0.95) -> float:
    """Lowest threshold whose recall is >= target (recall-first, not max-F1).

    Ties are broken toward MORE recall: with equal probabilities we flag.
    """
    p = np.asarray(p, np.float64)
    y = np.asarray(y, bool)
    if not y.any():
        raise ValueError("no positives")
    pos = np.sort(p[y])
    k = int(np.floor((1.0 - target) * len(pos)))
    return float(pos[min(k, len(pos) - 1)])


def prf(p: np.ndarray, y: np.ndarray, tau: float) -> dict:
    """precision / recall / f1 / flag-rate at tau."""
    yp = np.asarray(p) >= tau
    y = np.asarray(y, bool)
    tp, fp, fn = int((yp & y).sum()), int((yp & ~y).sum()), int((~yp & y).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {"tau": float(tau), "tp": tp, "fp": fp, "fn": fn,
            "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / max(prec + rec, 1e-12),
            "flag_rate": float(yp.mean())}


def average_precision(p: np.ndarray, y: np.ndarray) -> float:
    """Area under the precision-recall curve (prevalence-independent ranking quality)."""
    order = np.argsort(-np.asarray(p))
    ys = np.asarray(y, bool)[order]
    tp = np.cumsum(ys)
    prec = tp / np.arange(1, len(ys) + 1)
    return float((prec * ys).sum() / max(ys.sum(), 1))


def train(emb: np.ndarray, y: np.ndarray, backend, val: tuple | None = None,
          review_recall: float = REVIEW_RECALL,
          violation_recall: float = VIOLATION_RECALL) -> WeaponsHead:
    """Fit head + Platt + BOTH ADR-14 tier thresholds. ``val`` = held-out (emb, y).

    Both taus are fitted on the HELD-OUT split when one is given — fitting an operating
    point on training scores is how a moderation system quietly ships a 0.7 real recall.
    """
    w, b = fit_logistic(emb, y)
    s = np.asarray(emb, np.float32) @ w + b
    platt = fit_platt(s, y)
    ve, vy = val if val is not None else (emb, y)
    vp = platt_apply(np.asarray(ve, np.float32) @ w + b, platt)
    tau_r = tau_for_recall(vp, vy, review_recall)
    tau_v = max(tau_for_recall(vp, vy, violation_recall), tau_r)
    m = prf(vp, vy, tau_r)
    m["ap"] = average_precision(vp, vy)
    m["held_out"] = val is not None
    m["n"], m["n_pos"] = int(len(vy)), int(np.sum(vy))
    m["review"] = {"target_recall": review_recall, **prf(vp, vy, tau_r)}
    m["violation"] = {"target_recall": violation_recall, **prf(vp, vy, tau_v)}
    return WeaponsHead(model_id=getattr(backend, "name", "?"), model_sha=backend.model_sha,
                       dim=int(emb.shape[1]), w=w, b=b, platt=platt,
                       tau_violation=tau_v, tau_review=tau_r,
                       prompts_sha=prompt_ensemble_sha(POSITIVE_PROMPTS + NEGATIVE_PROMPTS),
                       metrics=m)


# ── persistence ───────────────────────────────────────────────────────────────
def head_path(model_id: str, root: Path | None = None) -> Path:
    """Heads are keyed by BACKEND NAME so the dispatcher can find one from the machine
    profile alone, without paying a model load. model_sha inside is the integrity check."""
    return (root or DATA) / f"weapons-{model_id}.json"


def save(head: WeaponsHead, root: Path | None = None) -> Path:
    p = head_path(head.model_id, root)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(head.to_json()))
    tmp.replace(p)
    return p


def load_head(model_id: str, root: Path | None = None,
              model_sha: str | None = None) -> WeaponsHead | None:
    p = head_path(model_id, root)
    if not p.is_file():
        return None
    h = WeaponsHead.from_json(json.loads(p.read_text()))
    if model_sha is not None and h.model_sha != model_sha:   # ADR-6's refusal shape
        raise ValueError(f"weapons head model_sha {h.model_sha[:12]} != loaded model "
                         f"{model_sha[:12]} — re-train the head, never score across models")
    return h


def load_weapons_head(profile: dict | None = None) -> WeaponsHead | None:
    """Dispatcher entry point (imgtag.moderation.load_heads contract).

    None when no head has been trained for this machine's backend — a missing track is
    simply not loaded and is reported by name, never a silent zero.
    """
    from ..core.models import DEFAULT_BACKEND
    model_id = (profile or {}).get("backend") or DEFAULT_BACKEND
    return load_head(model_id)


# ── zero-shot path (research baseline + offline fallback) ─────────────────────
class ZeroShotWeaponsHead:
    """Prompt-ensemble margin over the embeddings the index already computed.

    Works for ANY backend with no training, but its threshold is a FLAG BUDGET, not a
    calibrated probability: ``calibrated`` and ``enforcement_ready`` are always False and
    it may only ever produce the ``review`` tier (ADR-3 tiering law, ADR-14).
    """

    category = CATEGORY
    wants_images = False
    calibrated = False
    enforcement_ready = False

    def __init__(self, backend, tau: float = ZERO_SHOT_TAU):
        self.backend = backend
        self.model_id = f"zeroshot:{getattr(backend, 'name', '?')}"
        self.tau = tau
        self._pos = self._neg = None

    def probs(self, emb: np.ndarray) -> np.ndarray:
        if self._pos is None:
            self._pos, self._neg = prompt_matrices(self.backend)
        return zero_shot_prob(zero_shot_margin(emb, self._pos, self._neg))

    def score(self, embeddings, images=None, ids=None) -> list[dict]:
        return [{"category": CATEGORY, "p": round(float(v), 4),
                 "tier": "review" if v >= self.tau else "none",
                 "model_id": self.model_id, "calibrated": False,
                 "enforcement_ready": False}
                for v in self.probs(embeddings)]
