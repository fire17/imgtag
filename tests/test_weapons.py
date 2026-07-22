"""Weapons moderation track — unit + contract tests (track-weapons).

Fast by construction: everything below runs on synthetic embeddings. The MEASURED
quality numbers live in research/track-weapons.md and are produced by
scripts/train_weapons_head.py against the Open Images slice — a test must never
re-download 1.4GB to assert a metric.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from imgtag.moderation import weapons as W


# ── prompt ensemble hygiene ───────────────────────────────────────────────────
def test_prompt_lists_are_clean():
    for lst in (W.POSITIVE_PROMPTS, W.NEGATIVE_PROMPTS):
        assert len(lst) >= 20
        assert len(set(lst)) == len(lst), "duplicate prompt"
        assert all(p == p.strip() and p == p.lower() and not p.endswith(".") for p in lst)
    assert not set(W.POSITIVE_PROMPTS) & set(W.NEGATIVE_PROMPTS)


def test_hard_negative_families_are_present():
    """The families where an embedding model actually fails must be in the background set."""
    blob = " ".join(W.NEGATIVE_PROMPTS)
    for family in ("kitchen knife", "toy", "baseball bat", "scissors", "hand tools"):
        assert family in blob, f"missing hard-negative family: {family}"


# ── zero-shot margin ──────────────────────────────────────────────────────────
def _unit(a):
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def test_zero_shot_margin_is_signed_contrast():
    pos = _unit(np.array([[1.0, 0.0], [0.9, 0.1]], np.float32))
    neg = _unit(np.array([[0.0, 1.0]], np.float32))
    emb = _unit(np.array([[1.0, 0.0], [0.0, 1.0]], np.float32))
    m = W.zero_shot_margin(emb, pos, neg)
    assert m[0] > 0 > m[1], "weapon-like must outscore background-like"
    assert m.shape == (2,)


def test_zero_shot_prob_is_monotone_in_margin():
    m = np.array([-0.3, 0.0, 0.2, 0.5])
    p = W.zero_shot_prob(m)
    assert np.all(np.diff(p) > 0)
    assert 0.0 <= p.min() and p.max() <= 1.0


# ── training math ─────────────────────────────────────────────────────────────
def _toy(n=400, d=16, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=d)
    x = _unit(rng.normal(size=(n, d)).astype(np.float32))
    y = (x @ w > np.quantile(x @ w, 0.75)).astype(int)      # 25% positive
    return x, y


def test_fit_logistic_separates_a_linear_problem():
    x, y = _toy()
    w, b = W.fit_logistic(x, y)
    s = x @ w + b
    assert W.average_precision(s, y) > 0.95


def test_fit_logistic_handles_extreme_imbalance():
    """Balanced sample weights: a 2%-positive problem must still be learned."""
    rng = np.random.default_rng(1)
    d = 16
    w0 = rng.normal(size=d)
    x = _unit(rng.normal(size=(2000, d)).astype(np.float32))
    y = (x @ w0 > np.quantile(x @ w0, 0.98)).astype(int)
    w, b = W.fit_logistic(x, y)
    assert W.average_precision(x @ w + b, y) > 0.8


def test_tau_for_recall_hits_its_target():
    p = np.linspace(0, 1, 200)
    y = (np.arange(200) % 3 == 0).astype(int)
    for target in (0.80, 0.90, 0.95, 0.99, 1.0):
        tau = W.tau_for_recall(p, y, target)
        assert W.prf(p, y, tau)["recall"] >= target - 1e-9


def test_tau_for_recall_is_monotone():
    rng = np.random.default_rng(2)
    p, y = rng.random(500), (rng.random(500) < 0.2).astype(int)
    taus = [W.tau_for_recall(p, y, t) for t in (0.99, 0.95, 0.90, 0.80)]
    assert taus == sorted(taus), "a higher recall target must never raise tau"


def test_tau_for_recall_refuses_without_positives():
    with pytest.raises(ValueError):
        W.tau_for_recall(np.array([0.1, 0.2]), np.array([0, 0]))


def test_prf_counts():
    p = np.array([0.9, 0.8, 0.2, 0.1])
    y = np.array([1, 0, 1, 0])
    m = W.prf(p, y, 0.5)
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 1)
    assert m["precision"] == 0.5 and m["recall"] == 0.5
    assert m["flag_rate"] == 0.5


def test_average_precision_bounds():
    y = np.array([1, 1, 0, 0])
    assert W.average_precision(np.array([1.0, 0.9, 0.1, 0.0]), y) == pytest.approx(1.0)
    assert W.average_precision(np.array([0.0, 0.1, 0.9, 1.0]), y) < 0.6


# ── head lifecycle ────────────────────────────────────────────────────────────
class _FakeBackend:
    """Minimal ModelBackend surface: model_sha + embed_texts (models.py contract)."""

    name = "fake"
    model_sha = "deadbeef" * 8
    dim = 16

    def embed_texts(self, texts):
        rng = np.random.default_rng(abs(hash(tuple(texts))) % 2**32)
        return _unit(rng.normal(size=(len(texts), self.dim)).astype(np.float32))

    def release_text(self):
        pass


def _head(seed=0):
    x, y = _toy(seed=seed)
    n = len(x) // 2
    return W.train(x[:n], y[:n], _FakeBackend(), val=(x[n:], y[n:])), x, y


def test_train_produces_a_calibrated_recall_first_head():
    head, x, y = _head()
    assert head.dim == 16 and head.model_sha == _FakeBackend.model_sha
    assert head.metrics["held_out"] is True and head.calibrated and head.enforcement_ready
    assert head.metrics["review"]["recall"] >= W.REVIEW_RECALL - 1e-9
    assert head.metrics["violation"]["recall"] >= W.VIOLATION_RECALL - 1e-9
    assert head.tau_violation >= head.tau_review, "violation must be the stricter tier"
    p = head.probs(x)
    assert p.shape == (len(x),) and 0.0 <= p.min() and p.max() <= 1.0


def test_violation_tier_is_more_precise_than_review():
    head, x, y = _head()
    v, r = head.metrics["violation"], head.metrics["review"]
    assert v["precision"] >= r["precision"]
    assert v["recall"] <= r["recall"]


def test_tier_of_is_a_total_function():
    assert W.tier_of(0.9, 0.8, 0.4) == "violation"
    assert W.tier_of(0.5, 0.8, 0.4) == "review"
    assert W.tier_of(0.1, 0.8, 0.4) == "none"
    assert W.tier_of(0.8, 0.8, 0.4) == "violation", "tau is inclusive"


def test_head_roundtrips_through_json(tmp_path):
    head, x, _ = _head()
    p = W.save(head, tmp_path)
    assert json.loads(p.read_text())["category"] == "weapons"
    back = W.load_head(head.model_id, tmp_path)
    assert back is not None
    np.testing.assert_allclose(back.probs(x), head.probs(x), rtol=1e-6)
    assert (back.tau_review, back.tau_violation) == (head.tau_review, head.tau_violation)


def test_load_head_missing_is_none(tmp_path):
    assert W.load_head("no-such-backend", tmp_path) is None


def test_head_refuses_a_foreign_model_sha(tmp_path):
    """Same loud-refusal law as the index manifest (ADR-6): never score across models."""
    head, _, _ = _head()
    p = W.save(head, tmp_path)
    d = json.loads(p.read_text())
    d["model_sha"] = "0" * 64
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        W.load_head(head.model_id, tmp_path, model_sha=head.model_sha)


# ── the dispatcher contract (what indexer/daemon call) ────────────────────────
SCHEMA = {"category", "p", "tier", "model_id", "calibrated", "enforcement_ready"}


def test_head_score_matches_the_dispatcher_schema(tmp_path):
    head, x, _ = _head()
    out = head.score(x[:5], images=None, ids=None)
    assert len(out) == 5
    for r in out:
        assert set(r) == SCHEMA
        assert r["category"] == "weapons"
        assert r["tier"] in ("violation", "review", "none")
        assert isinstance(r["p"], float) and 0.0 <= r["p"] <= 1.0
        assert r["tier"] == W.tier_of(r["p"], head.tau_violation, head.tau_review)


def test_head_ignores_images_and_ids():
    """wants_images is False: passing pixels must change nothing (no second decode)."""
    head, x, _ = _head()
    a = head.score(x[:4])
    b = head.score(x[:4], images=np.zeros((4, 8, 8, 3), np.uint8), ids=[{}] * 4)
    assert a == b and head.wants_images is False


def test_head_refuses_a_dim_mismatch():
    head, _, _ = _head()
    with pytest.raises(ValueError):
        head.score(np.zeros((2, 32), np.float32))


def test_load_weapons_head_returns_none_without_a_trained_head(monkeypatch, tmp_path):
    monkeypatch.setattr(W, "DATA", tmp_path)
    assert W.load_weapons_head({"backend": "pecore-s16-384"}) is None


def test_load_weapons_head_finds_the_profile_backend(tmp_path, monkeypatch):
    head, _, _ = _head()
    head.model_id = "pecore-s16-384"
    W.save(head, tmp_path)
    monkeypatch.setattr(W, "DATA", tmp_path)
    got = W.load_weapons_head({"backend": "pecore-s16-384"})
    assert got is not None and got.model_id == "pecore-s16-384"


# ── zero-shot head: review tier only, never enforcement-ready ─────────────────
def test_zero_shot_head_is_review_only_and_uncalibrated():
    z = W.ZeroShotWeaponsHead(_FakeBackend(), tau=0.0)
    out = z.score(_unit(np.random.default_rng(3).normal(size=(4, 16)).astype(np.float32)))
    assert len(out) == 4
    for r in out:
        assert set(r) == SCHEMA
        assert r["tier"] == "review", "an uncalibrated signal may never claim violation"
        assert r["calibrated"] is False and r["enforcement_ready"] is False
    assert W.ZeroShotWeaponsHead(_FakeBackend(), tau=1.1).score(
        _unit(np.ones((1, 16), np.float32)))[0]["tier"] == "none"


def test_zero_shot_head_embeds_prompts_once():
    calls = []

    class _Counting(_FakeBackend):
        def embed_texts(self, texts):
            calls.append(len(texts))
            return _FakeBackend.embed_texts(self, texts)

    z = W.ZeroShotWeaponsHead(_Counting())
    x = _unit(np.random.default_rng(4).normal(size=(3, 16)).astype(np.float32))
    z.score(x)
    z.score(x)
    assert len(calls) == 2, "one positive batch + one background batch, then cached"


# ── shipped artifacts ─────────────────────────────────────────────────────────
def test_shipped_heads_are_wellformed():
    """Every head in the package is loadable, calibrated, and carries its measurement."""
    if not W.DATA.is_dir():
        pytest.skip("no shipped heads yet")
    heads = sorted(W.DATA.glob("weapons-*.json"))
    if not heads:
        pytest.skip("no shipped heads yet")
    for path in heads:
        d = json.loads(path.read_text())
        h = W.WeaponsHead.from_json(d)
        assert len(h.w) == h.dim and h.platt
        assert 0 < h.tau_review <= h.tau_violation < 1
        assert h.metrics.get("held_out") is True, "tau fitted on train = a lie about recall"
        assert h.metrics["review"]["recall"] >= W.REVIEW_RECALL - 1e-9
        assert h.metrics["violation"]["recall"] >= W.VIOLATION_RECALL - 1e-9
        assert h.enforcement_ready is True
        assert path.name == W.head_path(h.model_id).name
        assert h.metrics["eval_split"] == "openimages-validation"
        assert h.metrics["train_split"] == "openimages-test"
        assert h.metrics["n_pos"] >= 300, "too few positives to claim a recall number"


def test_shipped_head_quality_does_not_regress():
    """Locks the MEASURED quality of the shipped head (research/track-weapons.md §5).

    A re-train that quietly loses ranking quality or firearm recall fails here rather than
    in production. Bounds are floors, not equalities — improving the head is always fine.
    """
    h = W.load_head("pecore-s16-384")
    if h is None:
        pytest.skip("default head not built (run scripts/train_weapons_head.py --save)")
    assert h.metrics["ap"] >= 0.87, f"AP regressed to {h.metrics['ap']}"
    assert h.metrics["violation"]["precision"] >= 0.90, "violation tier must stay actionable"
    per_class = h.metrics.get("per_class_recall", {})
    for firearm in ("Rifle", "Shotgun", "Handgun"):
        assert per_class.get(firearm, 0) >= 0.95, f"{firearm} recall regressed"
    assert per_class.get("Knife", 0) >= 0.80, "knife is the weakest class; floor it"


# ── subcategory taxonomy + TP-probe separation (user directive 13:58Z) ────────
from pathlib import Path  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]

# The user's explicit subcategory list — the taxonomy MUST cover every one of these.
_REQUIRED_SUBCATS = {
    "handguns_pistols_revolvers", "rifles", "shotguns", "knives_threat",
    "swords_machetes_axes", "bows_crossbows", "explosives_grenades", "heavy_ordnance",
}


def test_probe_taxonomy_covers_the_user_subcategories():
    """data/weapon-probe/taxonomy.json is versioned DATA and must span the asked-for list."""
    f = _ROOT / "data" / "weapon-probe" / "taxonomy.json"
    if not f.is_file():
        pytest.skip("probe not built (run scripts/build_weapons_probe.py)")
    tax = json.loads(f.read_text())
    assert _REQUIRED_SUBCATS <= set(tax["subcategories"]), "taxonomy misses a user subcategory"
    # every non-gap subcategory lists images; every image maps back to >=1 subcategory
    gaps = {g["subcategory"] for g in tax.get("gaps", [])}
    for name, spec in tax["subcategories"].items():
        assert spec["oi_classes"], f"{name} has no OI class mapping"
        if name not in gaps:
            assert spec["n"] == len(spec["images"]) > 0, f"{name} claims images it lacks"
    for img, subs in tax["image_subcategories"].items():
        assert subs, f"{img} belongs to no subcategory"


def test_moderation_spec_weapons_is_deepened_and_toys_stay_review():
    """The moderation.json weapons entry carries the subcategory taxonomy as DATA, and
    ADR-14's toy/replica-is-review boundary is asserted, not assumed."""
    f = _ROOT / "src" / "imgtag" / "data" / "moderation.json"
    w = json.loads(f.read_text())["categories"]["weapons"]
    assert _REQUIRED_SUBCATS <= set(w.get("subcategories", {}))
    assert w.get("toy_replica_tier") == "review"
    # every subcategory's prompts are flattened into the violation set (no toy leakage)
    flat = {p for ps in w["subcategories"].values() for p in ps}
    assert flat <= set(w["violation"]), "a subcategory prompt is missing from violation"
    toys = " ".join(w["review"]).lower()
    assert "toy" in toys and "replica" in toys, "toy/replica must live in review, not violation"
    assert not any("toy" in p or "replica" in p for p in w["violation"])


def test_separation_result_is_ordered_and_honest():
    """If eval-weapons.json exists, its proposed tiers must be well-ordered and its overall
    true-positive mass must beat the false-positive band — the whole point of the exercise."""
    f = _ROOT / "research" / "eval-weapons.json"
    if not f.is_file():
        pytest.skip("separation not measured (run scripts/eval_weapons.py --write-json)")
    r = json.loads(f.read_text())
    assert 0 < r["proposed"]["tau_review"] < r["proposed"]["tau_violation"] < 1
    assert r["tp_overall"]["median"] > r["fp_band"]["distribution"]["p99"], \
        "true positives must out-score the FP band's tail — else the track cannot be trusted"
    assert r["separation_overall_ap"] >= 0.85, "overall TP-vs-FP separation regressed"
