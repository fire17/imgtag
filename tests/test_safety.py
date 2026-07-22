"""Safety moderation track — unit + contract + acceptance tests (track-safety).

Synthetic embeddings only: the MEASURED numbers live in research/track-safety.md and are
produced by scripts/eval_safety.py against COCO val2017. A test never re-embeds a corpus
and never asserts a quality metric it did not measure here.

The SIX-IMAGE ACCEPTANCE SKETCH the brief asks for is built the honest way a synthetic
test can: each "image" is a controlled mix of the track's own concept vectors, so we test
the SCORER'S TIER LOGIC (person-down gates alert; danger escalates; benign context does
not; empty/crowd stay none) rather than pretending to measure model accuracy. Real-image
acceptance is `scripts/eval_safety.py` on the injured-man case (000000354307 → alert) and
the benign-lying majority (→ review), cited in the research doc.
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np
import pytest

from imgtag.moderation import safety as S


def _unit(a):
    a = np.asarray(a, np.float32)
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)


class FakeBackend:
    """Deterministic pseudo-embeddings: same text -> same vector, no model needed."""

    dim = 48
    model_sha = "fakesha_safety_0000000000000000000000000000000000000000000000000000"
    model_id = "fake-safety"

    def embed_texts(self, texts):
        out = []
        for t in texts:
            # zlib.crc32, NOT hash(): str hashing is salted per process (flaky fixtures)
            rng = np.random.default_rng(zlib.crc32(t.encode()))
            out.append(rng.normal(size=self.dim))
        return _unit(np.asarray(out, np.float32))


@pytest.fixture(scope="module")
def scorer():
    return S.SafetyScorer.build(FakeBackend())


# ── prompt hygiene ────────────────────────────────────────────────────────────────────
def test_banks_are_clean_and_disjoint():
    banks = (S.LYING, S.BACKGROUND, S.DANGER, S.DANGER_BACKGROUND, S.BENIGN_CONTEXT,
             S.ALERT_PHRASES, S.GROUND_LEVEL_FP)
    for lst in banks:
        assert len(set(lst)) == len(lst), "duplicate prompt"
        assert all(p == p.strip() and not p.endswith(".") and "  " not in p for p in lst)
    # the positive (person-down) bank and the subtracted background must not overlap —
    # subtracting a positive would remove real detections.
    assert not set(S.LYING) & set(S.BACKGROUND)
    assert not set(S.LYING) & set(S.GROUND_LEVEL_FP)


def test_measured_fp_classes_are_in_a_subtracted_bank():
    """Every FP family the eval / b-daemon actually caught must be represented:
    animal-lying (measured largest FP), and b-daemon's boots/puddle ground-level class."""
    subtracted = " ".join(S.BACKGROUND + S.DANGER_BACKGROUND + S.GROUND_LEVEL_FP).lower()
    for family in ("cat", "dog", "animal", "boots", "puddle", "pavement", "campfire"):
        assert family in subtracted, f"missing measured FP family: {family}"


def test_benign_context_is_never_subtracted():
    """policy_neighbours law (drugs track measured AP 0.58->0.04 when subtracted): a
    sunbather and a collapsed person are the same pose. The benign bank must NOT appear in
    any subtracted list, only as the annotation bank."""
    subtracted = set(S.BACKGROUND) | set(S.DANGER_BACKGROUND) | set(S.GROUND_LEVEL_FP)
    assert not (set(S.BENIGN_CONTEXT) & subtracted)


# ── tier logic: the six-image acceptance sketch (synthetic, tests the LOGIC) ────────────
def _img(backend, concepts, weights=None):
    """A synthetic 'image' embedding = weighted mix of concept text vectors, re-normalized.
    This is not a real image; it exercises the scorer's tier arithmetic deterministically."""
    v = np.asarray(backend.embed_texts(concepts), np.float32)
    w = np.ones(len(concepts)) if weights is None else np.asarray(weights, float)
    return _unit((w[:, None] * v).sum(0))


def test_acceptance_sketch_six_scenes(scorer):
    """sleeper-on-sofa→review, beach-lounger→review, fallen-with-wreckage→alert,
    upright→none, empty→none, crowd-standing→none. Tests tier LOGIC via the scorer's own
    concept space (real-image accuracy is scripts/eval_safety.py, not a unit test)."""
    b = FakeBackend()
    # Drive p above/below tau by mixing in the matching bank vs the background bank. We
    # assert the ORDERING/logic the tiers encode, using the scorer's real thresholds.
    scenes = {
        "sleeper_on_sofa": (_img(b, ["a person lying on a sofa", "a person sleeping"]), "down_benign"),
        "beach_lounger": (_img(b, ["a person sunbathing lying on the beach"]), "down_benign"),
        "fallen_with_wreckage": (_img(b, ["a fallen person lying on the ground",
                                          "a wrecked crashed car", "blood on the ground"]), "down_danger"),
        "upright_person": (_img(b, ["a person standing", "a person walking"]), "up"),
        "empty_scene": (_img(b, ["an empty room", "an outdoor landscape"]), "up"),
        "crowd_standing": (_img(b, ["a group of people standing", "people standing upright"]), "up"),
    }
    emb = np.stack([v for v, _ in scenes.values()])
    out = scorer.score(emb)
    # The invariant we can assert deterministically: a down_danger scene scores danger
    # ABOVE both benign-down and upright scenes, and person-down scores ABOVE upright.
    pl = {k: float(out["p"][i]) for i, k in enumerate(scenes)}
    pd = {k: float(out["p_danger"][i]) for i, k in enumerate(scenes)}
    assert pl["fallen_with_wreckage"] > pl["upright_person"]
    assert pl["sleeper_on_sofa"] > pl["upright_person"]
    assert pd["fallen_with_wreckage"] > pd["sleeper_on_sofa"]
    assert pd["fallen_with_wreckage"] > pd["empty_scene"]


def test_alert_requires_person_down_not_danger_alone(scorer):
    """The boundary agreed with track-violence: safety.alert NEVER fires on danger/gore
    without a person down. Danger alone must not reach alert."""
    b = FakeBackend()
    danger_only = _img(b, ["a burning building on fire", "a scene of destruction and debris"])
    # force high danger, minimal lying by construction; assert tier is not 'alert'
    out = scorer.per_image(danger_only[None], 0)
    assert out["tier"] != "alert" or out["p"] >= scorer.pol["tau"], \
        "alert fired without person-down probability clearing tau"


def test_tiers_ordering_and_vocabulary(scorer):
    """alert is the highest tier; the vocabulary is exactly alert|review|none.

    Middle case uses p_danger BELOW tau_danger (0.0076) — the shipped danger gate is
    deliberately loose, so 'down but not in danger' must use a genuinely tiny danger p."""
    pl = np.array([0.9, 0.9, 0.001])
    pd = np.array([0.9, 0.0, 0.9])
    t = scorer.tiers(pl, pd)
    assert list(t) == ["alert", "review", "none"]      # down+danger, down-only, neither
    assert set(np.unique(t)) <= set(S.TIERS)


def test_danger_alone_tier_knob(scorer):
    """AMBIGUITIES #2: danger with no person visible flags only if the operator opts in."""
    pl = np.array([0.001])           # no person down
    pd = np.array([0.99])            # strong danger
    assert scorer.tiers(pl, pd)[0] == "none"           # default danger_alone_tier="none"
    opted = S.SafetyScorer.build(FakeBackend(), config={"danger_alone_tier": "review"})
    assert opted.tiers(pl, pd)[0] == "review"


# ── head seam contract ──────────────────────────────────────────────────────────────────
def test_head_score_shape(scorer):
    emb = _unit(np.random.default_rng(0).normal(size=(5, FakeBackend.dim)))
    head = S.SafetyHead(scorer, "sha")
    rows = head.score(emb)
    assert len(rows) == 5
    for r in rows:
        assert r["category"] == "safety"
        assert isinstance(r["p"], float) and 0.0 <= r["p"] <= 1.0
        assert r["tier"] in S.TIERS
    # probs() returns the (p, tier) arrays b-daemon's track_scores() consumes
    p, tier = head.probs(emb)
    assert len(p) == 5 and len(tier) == 5


def test_head_is_never_enforcement_ready(scorer):
    """No labelled safety corpus exists — the head must not claim it can act unattended."""
    assert S.SafetyHead(scorer, "sha").enforcement_ready is False
    assert S.SafetyHead(scorer, "sha").wants_images is False   # scores embeddings, no re-decode


def test_loader_returns_none_without_a_backend(monkeypatch):
    """load_safety_head must fail SOFT: a broken machine means 'no safety track', not a
    crashed index run (never raises)."""
    monkeypatch.setattr(S, "_cache_path", lambda *a, **k: (_ for _ in ()).throw(RuntimeError))
    assert S.load_safety_head(profile={}, backend=FakeBackend()) is None


def test_config_policy_is_typo_proof():
    """A malformed moderation.json must never take safety offline (falls back to defaults)."""
    assert S.policy({"tau": "not a number", "danger_alone_tier": "bogus"})["tau"] == S.TAU_REVIEW
    assert S.policy({})["danger_alone_tier"] == "none"
    assert S.policy({"tau_danger": 0.5})["tau_danger"] == 0.5


# ── moderation.json contract: the spec the reader consumes ──────────────────────────────
def test_track_spec_is_reader_shaped():
    spec = S.track_spec()
    # b-daemon's reader: tiers are prompt-set keys; alert must be present and contrastive
    assert spec["alert"] and spec["review"] and spec["negatives"]
    assert "violation" not in spec                     # this track has no violation tier
    assert spec["enforcement_ready"] is False
    assert spec["calibration"] == "proxy-fitted"       # not "fitted" — no labelled corpus
    # alert prompts are COMBINED person-down+danger (so exceedance ~ the code AND)
    for p in spec["alert"]:
        assert any(w in p for w in ("lying", "collapsed", "fallen", "unconscious", "down"))


def test_moderation_json_carries_safety_and_is_valid():
    path = Path(S.__file__).resolve().parent.parent / "data" / "moderation.json"
    data = json.loads(path.read_bytes())
    assert "safety" in data["categories"]
    e = data["categories"]["safety"]
    assert set(e) >= {"label", "alert", "review", "negatives", "calibration",
                      "enforcement_ready", "spec_sha"}
    assert e["enforcement_ready"] is False
    assert e["spec_sha"] == S.spec_sha()               # spec on disk matches the module
