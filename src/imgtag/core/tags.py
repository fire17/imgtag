"""Tag table build + Platt calibration + dataset-layer stats spec — ADR-3.

OWNER: b-bench (wave-b-briefs). b-engine's indexer reads the table, b-daemon reads the
calibration; only this module writes either.

ADR-3 says the tag vocabulary is ~4–8k names scored by the SAME text encoder at index
time, in TWO TIERS:
  * `calibrated`   — the tag has ground truth (COCO exhaustive 80 / LVIS frequent+common),
                     so a per-tag Platt sigmoid + tau can be fitted. ONLY these may
                     hard-gate or produce an honest "no match".
  * `uncalibrated` — everything else (LVIS rare, Open Images, curated). May boost rank and
                     explain a hit; may NEVER gate or veto.
The planner's chaser is law here: "the tag table wants to stay SMALL and curated …
precision dies in the tail". This builder emits ~4k and grows only on measured precision.

On-disk contract (ADR-3, one owner, one location):
    ~/.imgtag/models/<model_sha>/tags.f32    float32 [T, dim], L2-normalized, C-order
    ~/.imgtag/models/<model_sha>/tags.json   {names, dim, model_sha, prompt_ensemble_sha,
                                              tier, tau, platt, provenance}
`tau` and `platt` are null for every tag until a CAL-SET fit runs — a null is an honest
"not calibrated", never a guessed threshold.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from dataclasses import dataclass

import numpy as np

DATA = os.path.join(os.environ.get("IMGTAG_ROOT",
                                   os.path.expanduser("~/Creations/ImgTag")), "data")
HOME = os.path.expanduser("~/.imgtag")

CALIBRATED, UNCALIBRATED = "calibrated", "uncalibrated"

# Curated additions: scene/context words no object-detection vocabulary carries, which
# real photo-library queries lean on. Deliberately short (chaser: small and curated).
CURATED = [
    "beach", "sunset", "sunrise", "mountain", "forest", "desert", "snow", "city street",
    "skyline", "night sky", "portrait", "selfie", "crowd", "wedding", "birthday party",
    "concert", "graduation", "hiking trail", "waterfall", "lake", "river", "ocean wave",
    "rain", "fog", "rainbow", "fireworks", "campfire", "picnic", "market stall",
    "construction site", "office desk", "classroom", "hospital room", "gym",
    "swimming pool", "playground", "garden", "farm field", "vineyard", "barn",
    "bridge", "tunnel", "railway station", "airport terminal", "parking lot",
    "restaurant interior", "cafe", "bar", "library", "museum", "church interior",
    "temple", "castle", "ruins", "lighthouse", "harbor", "sailboat", "kayak",
    "ski slope", "surfing", "scuba diving", "road trip", "traffic jam", "storefront",
    "graffiti wall", "neon sign", "silhouette", "reflection in water", "aerial view",
    "close-up of food", "latte art", "birthday cake", "christmas tree", "halloween",
    "baby", "toddler", "elderly person", "family group photo", "pet on a couch",
]


def _norm(name: str) -> str:
    n = re.sub(r"[_\-]+", " ", str(name)).strip().lower()
    n = re.sub(r"\s*\(.*?\)\s*", " ", n)  # LVIS disambiguators: "bat (animal)"
    return re.sub(r"\s+", " ", n).strip()


@dataclass
class TagTable:
    names: list[str]
    tier: list[str]
    provenance: list[str]
    dim: int = 0
    model_sha: str = ""
    prompt_ensemble_sha: str = ""
    emb: np.ndarray | None = None
    tau: list | None = None
    platt: list | None = None

    def __len__(self) -> int:
        return len(self.names)


def build_tag_table(max_tags: int = 4200) -> TagTable:
    """COCO 80 + LVIS (names + synonyms) + Open Images 600 + curated, deduped.

    Order is stable: calibrated sources first, then uncalibrated, then curated — so a
    truncation at `max_tags` drops the least-trustworthy tail, never a COCO class.
    """
    rows: list[tuple[str, str, str]] = []  # (name, tier, provenance)
    seen: set[str] = set()

    def add(name: str, tier: str, prov: str):
        n = _norm(name)
        if n and n not in seen and len(n) <= 40:
            seen.add(n)
            rows.append((n, tier, prov))

    coco = os.path.join(DATA, "coco/annotations/instances_val2017.json")
    if os.path.exists(coco):
        for c in json.load(open(coco))["categories"]:
            add(c["name"], CALIBRATED, "coco80")

    lvis = os.path.join(DATA, "lvis/lvis_val2017_only.json")
    if os.path.exists(lvis):
        cats = json.load(open(lvis))["categories"]
        for freq, tier in (("f", CALIBRATED), ("c", CALIBRATED), ("r", UNCALIBRATED)):
            for c in cats:
                if c["frequency"] == freq:
                    add(c["name"], tier, f"lvis-{freq}")
        for c in cats:  # synonyms ride at the parent's tier, flagged as aliases
            tier = CALIBRATED if c["frequency"] in "fc" else UNCALIBRATED
            for s in c.get("synonyms", []):
                add(s, tier, f"lvis-{c['frequency']}-syn")

    oi = os.path.join(DATA, "openimages/oidv7-class-descriptions-boxable.csv")
    if os.path.exists(oi):
        with open(oi) as fh:
            for r in csv.DictReader(fh):
                add(r["DisplayName"], UNCALIBRATED, "openimages600")

    for c in CURATED:
        add(c, UNCALIBRATED, "curated")

    rows = rows[:max_tags]
    return TagTable(names=[r[0] for r in rows], tier=[r[1] for r in rows],
                    provenance=[r[2] for r in rows],
                    tau=[None] * len(rows), platt=[None] * len(rows))


# ── Platt calibration ─────────────────────────────────────────────────────────
def fit_platt(scores: np.ndarray, labels: np.ndarray, iters: int = 100) -> list:
    """Per-tag Platt scaling: p = sigmoid(-(A*s + B)). Newton/IRLS, numpy only.

    Uses Platt's prior-corrected targets so a tag with few positives cannot fit a
    degenerate step function. Returns [A, B]; [0.0, 0.0] means UNFITTABLE.
    """
    s = np.asarray(scores, np.float64).ravel()
    y = np.asarray(labels, np.float64).ravel() > 0
    n_pos, n_neg = float(y.sum()), float((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return [0.0, 0.0]
    hi, lo = (n_pos + 1) / (n_pos + 2), 1 / (n_neg + 2)
    t = np.where(y, hi, lo)

    A, B = 0.0, float(np.log((n_neg + 1) / (n_pos + 1)))
    for _ in range(iters):
        f = A * s + B
        p = 1.0 / (1.0 + np.exp(f))          # p = sigmoid(-f)
        d = p - t
        w = np.maximum(p * (1 - p), 1e-12)
        g = np.array([np.dot(d, s), d.sum()])
        H = np.array([[np.dot(w, s * s), np.dot(w, s)],
                      [np.dot(w, s), w.sum()]]) + 1e-10 * np.eye(2)
        step = np.linalg.solve(H, g)
        A, B = A + step[0], B + step[1]
        if np.abs(step).max() < 1e-9:
            break
    return [float(A), float(B)]


def platt_apply(scores: np.ndarray, ab) -> np.ndarray | None:
    if not ab:
        return None
    return 1.0 / (1.0 + np.exp(ab[0] * np.asarray(scores, np.float64) + ab[1]))


def max_f1_tau(p: np.ndarray, y: np.ndarray) -> tuple:
    """Threshold on calibrated probability maximizing F1. Returns (tau, f1)."""
    order = np.argsort(-p)
    ys = np.asarray(y, bool)[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(~ys)
    fn = ys.sum() - tp
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1e-12)
    i = int(np.argmax(f1))
    return float(p[order][i]), float(f1[i])


def calibrate(table: TagTable, scores: np.ndarray, labels: np.ndarray) -> TagTable:
    """Fit per-tag Platt + max-F1 tau on a HELD-OUT set. scores/labels: [N, T].

    Only `calibrated`-tier tags with both classes present get a fit; every other tag keeps
    tau=None (ADR-3: uncalibrated tags may never gate). Uncalibrated tags inherit nothing.
    """
    n_tags = len(table)
    table.platt = [None] * n_tags
    table.tau = [None] * n_tags
    for j in range(n_tags):
        if table.tier[j] != CALIBRATED:
            continue
        y = labels[:, j]
        if y.sum() == 0 or y.sum() == len(y):
            continue
        ab = fit_platt(scores[:, j], y)
        if ab == [0.0, 0.0]:
            continue
        p = platt_apply(scores[:, j], ab)
        table.platt[j], table.tau[j] = ab, max_f1_tau(p, y)[0]
    return table


CALSET_DIR = os.path.join(DATA, "cocotrain2k")


def calset_status() -> dict:
    """CAL-SET readiness. The fit sits BEHIND this interface, never faked."""
    imgs = os.path.join(CALSET_DIR, "images")
    n = len(os.listdir(imgs)) if os.path.isdir(imgs) else 0
    return {"ready": n >= 500, "n_images": n, "dir": CALSET_DIR,
            "note": "CAL-SET (cocotrain2k) is a HELD-OUT split — never benched (BUDGETS)."}


# ── persistence ───────────────────────────────────────────────────────────────
def prompt_ensemble_sha(prompts) -> str:
    return hashlib.sha256("\x00".join(prompts).encode()).hexdigest()[:16]


def table_dir(model_sha: str) -> str:
    return os.path.join(HOME, "models", model_sha)


def save(table: TagTable, model_sha: str, root: str | None = None) -> str:
    d = os.path.join(root, model_sha) if root else table_dir(model_sha)
    os.makedirs(d, exist_ok=True)
    if table.emb is None:
        raise ValueError("tag table has no embeddings — embed before saving")
    emb = np.ascontiguousarray(table.emb, dtype=np.float32)
    emb.tofile(os.path.join(d, "tags.f32"))
    meta = {
        "names": table.names, "dim": int(emb.shape[1]), "model_sha": model_sha,
        "prompt_ensemble_sha": table.prompt_ensemble_sha, "tier": table.tier,
        "tau": table.tau, "platt": table.platt,
        "provenance": {"sources": sorted(set(table.provenance)),
                       "per_tag": table.provenance,
                       "n_calibrated": sum(t == CALIBRATED for t in table.tier),
                       "calset": calset_status()},
    }
    tmp = os.path.join(d, "tags.json.tmp")
    with open(tmp, "w") as fh:
        json.dump(meta, fh)
    os.replace(tmp, os.path.join(d, "tags.json"))  # atomic, ADR-6 discipline
    return d


def load(model_sha: str, root: str | None = None) -> TagTable:
    d = os.path.join(root, model_sha) if root else table_dir(model_sha)
    meta = json.load(open(os.path.join(d, "tags.json")))
    emb = np.fromfile(os.path.join(d, "tags.f32"), dtype=np.float32)
    return TagTable(names=meta["names"], tier=meta["tier"],
                    provenance=meta["provenance"]["per_tag"], dim=meta["dim"],
                    model_sha=meta["model_sha"],
                    prompt_ensemble_sha=meta["prompt_ensemble_sha"],
                    emb=emb.reshape(len(meta["names"]), meta["dim"]),
                    tau=meta["tau"], platt=meta["platt"])


# ── dataset-layer stats spec (ADR-3 layer 2) ──────────────────────────────────
# Accumulated FREE during the index matmul by b-engine; stored in the manifest as
# `tag_stats`. Effective threshold: tau_eff = max(tau_tag, mean + k*std), k default 3.
DATASET_STATS_SPEC = {
    "per_tag": ["n", "sum", "sumsq", "p99"],
    "derived": {"mean": "sum/n", "std": "sqrt(sumsq/n - mean^2)",
                "tau_eff": "max(tau_tag, mean + k*std)  # k=3"},
    "manifest_key": "tag_stats",
    "approx_bytes": "4k tags x 4 float32 = ~64KB",
    "note": "streaming/one-pass so it costs nothing extra; recomputed on every reindex.",
}
