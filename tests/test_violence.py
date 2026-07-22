"""Violence / abuse moderation track — unit + contract tests (track-violence).

Fast by construction: everything runs on synthetic embeddings from a deterministic
FakeBackend. The MEASURED false-positive numbers live in research/track-violence.md and
are produced by scripts/eval_violence.py against COCO + Unsplash slices — a test must
never re-embed a corpus to assert a rate.

The invariants under test are the ones the drugs/nudity lanes paid for in defects:
  * the CONTEXT (staged/clinical) bank is NEVER subtracted — only arbitrated;
  * a flag that loses arbitration DEMOTES to review, never drops (nothing leaves the queue);
  * the score distribution does not saturate at 0.99;
  * alert is reserved for the SEVERE bank — a bloodless fight can never reach alert;
  * enforcement_ready / calibrated stay false (no labelled positives on this machine);
  * moderation.json's `violence` key never drifts from this module.
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np
import pytest

from imgtag.moderation import violence as V


def _unit(a):
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)


class FakeBackend:
    """Deterministic pseudo-embeddings: same text -> same vector, no model needed."""

    dim = 32
    model_sha = "fake" * 16
    model_id = "fake-backend"

    def embed_texts(self, texts):
        out = []
        for t in texts:
            # zlib.crc32, NOT hash(): str hashing is salted per process (flaky fixtures)
            rng = np.random.default_rng(zlib.crc32(t.encode()))
            out.append(rng.normal(size=self.dim))
        return _unit(np.asarray(out, np.float32))


# ── prompt bank hygiene ───────────────────────────────────────────────────────
def test_banks_are_clean_and_disjoint():
    vio = [c for g in V.VIOLENT.values() for c in g]
    for lst in (V.SEVERE, vio, V.CONTEXT, V.BACKGROUND):
        assert len(lst) >= 2
        assert len(set(lst)) == len(lst), "duplicate prompt"
        assert all(p == p.strip() and not p.endswith(".") for p in lst)
    # a prompt may not sit in two banks — that would make its role ambiguous
    banks = [set(V.SEVERE), set(vio), set(V.CONTEXT), set(V.BACKGROUND)]
    for i, a in enumerate(banks):
        for b in banks[i + 1:]:
            assert not a & b, f"prompt shared across banks: {a & b}"


def test_contact_sports_are_in_the_subtracted_background():
    """THE classic FP class. Boxing/martial-arts/wrestling/rugby MUST be subtracted, not
    left to arbitration — a sanctioned bout is exculpatory, and the sports track owns the
    positive label for it (composition ruling)."""
    blob = " ".join(V.BACKGROUND).lower()
    for family in ("boxing", "martial artists", "wrestling", "rugby", "hockey", "soccer"):
        assert family in blob, f"missing contact-sport background: {family}"


def test_staged_and_clinical_are_context_not_background():
    """Halloween SFX and surgery are visually IDENTICAL twins of gore — subtracting them
    would subtract the signal (the drugs lane's AP 0.58 -> 0.04 collapse). They belong in
    the arbitrated CONTEXT bank, never in the subtracted BACKGROUND bank."""
    ctx = " ".join(V.CONTEXT).lower()
    bg = " ".join(V.BACKGROUND).lower()
    for twin in ("halloween", "fake blood", "surgeon", "butcher"):
        assert twin in ctx, f"{twin} must be an arbitrated context concept"
        assert twin not in bg, f"{twin} must NOT be subtracted"


# ── scorer ────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def scorer():
    return V.ViolenceScorer.build(FakeBackend())


def test_vectors_are_l2_normalized(scorer):
    for m in (scorer.severe, scorer.violent, scorer.context, scorer.bg):
        assert np.allclose(np.linalg.norm(m, axis=1), 1.0, atol=1e-5)
    assert len(scorer.names) == len(scorer.groups) == len(scorer.violent)


def test_score_contract(scorer):
    emb = _unit(np.random.default_rng(0).normal(size=(9, FakeBackend.dim)))
    out = scorer.score(emb)
    assert out["category"] == "violence"
    assert out["p"].shape == (9,) and ((0 <= out["p"]) & (out["p"] <= 1)).all()
    assert set(np.unique(out["tier"])) <= set(V.TIERS) | {"none"}
    assert len(out["concept"]) == 9 and all(c in scorer.names for c in out["concept"])
    assert set(out["group"]) <= set(V.VIOLENT)
    one = scorer.per_image(emb, 4)
    assert set(one) >= {"category", "p", "tier", "why", "group"}
    assert one["p"] == pytest.approx(float(out["p"][4]))


def test_probability_is_monotonic_in_margin(scorer):
    """A violence-concept-aligned image must outscore a background-aligned one."""
    emb = _unit(np.stack([scorer.violent[0], scorer.bg[0]]))
    out = scorer.score(emb)
    assert out["margin"][0] > out["margin"][1]
    assert out["p"][0] > out["p"][1]


def test_p_mapping_spreads_over_the_designed_margin_range():
    """b-daemon's rolled-back-defect guard: the margin->p map must SPREAD over the real
    operating range, not pin to 0.99. The real-corpus ceiling (frac p>0.9 on COCO) is a
    corpus property measured by scripts/eval_violence.py; here we test the fitted PLATT
    map directly over the safe-corpus margin band (~ -0.05 .. p99.9 ~ 0.10)."""
    m = np.linspace(-0.05, 0.12, 40)
    p = V._sigmoid(V.PLATT_A * m + V.PLATT_B)
    assert p.min() < 0.30, "safe-corpus floor should map well below 0.5"
    assert p.max() > 0.90, "the tail should be able to reach a high score"
    assert p.std() > 0.15, "p must spread across the band, not saturate"
    assert np.all(np.diff(p) > 0), "p must be monotone in margin"


# ── tiers are p-space ASCENDING bands (the nudityprobe unit-bug fix) ──────────
def test_taus_are_p_space_ascending():
    """LAW after the 2026-07-22 incident: taus live in the SAME [0,1] space as the stored
    score `p`, ASCENDING (review < violation < alert). Margin-space values (~0.05) applied
    to `p` by derive_tiers is exactly what flagged 16 false alerts on swimwear."""
    assert 0.0 < V.TAU_REVIEW < V.TAU_VIOLATION < V.TAU_ALERT <= 1.0
    assert V.TAU_REVIEW >= 0.2, "a p-space review band must be well above margin-space ~0.05"


def test_tier_of_bands_p_by_severity():
    """alert = the TOP p band; a mid p is violation; a low-but-nonzero p is review."""
    pol = V.DEFAULTS
    assert V.tier_of(0.99, pol) == "alert"
    assert V.tier_of(0.90, pol) == "violation"   # in [violation, alert)
    assert V.tier_of(0.50, pol) == "review"       # in [review, violation)
    assert V.tier_of(0.10, pol) == "none"
    arr = V.tier_of(np.array([0.99, 0.90, 0.50, 0.10]), pol)
    assert list(arr) == ["alert", "violation", "review", "none"]


def test_head_tier_matches_derive_tiers(scorer):
    """B25d: the head's tier and store.derive_tiers must be byte-identical on the same `p`
    — one mapping, so head / daemon recount / CLI cannot disagree."""
    from imgtag.core.store import derive_tiers
    emb = _unit(np.random.default_rng(7).normal(size=(64, FakeBackend.dim)))
    out = scorer.score(emb)
    spec = V.track_spec()
    assert list(out["tier"]) == derive_tiers(out["p"], spec), "head tier != derive_tiers"


def test_alert_sorts_above_violation():
    assert V.TIERS.index("alert") < V.TIERS.index("violation") < V.TIERS.index("review")


# ── the nudityprobe incident regression: elevated-but-not-extreme p never alerts ──
def test_elevated_p_never_reaches_alert():
    """The 2026-07-22 failure: 202 swimwear images (max stored p=0.82) produced 16 false
    `alert`. With ascending p-space bands, nothing below tau_alert=0.95 can reach alert —
    a swimwear intimate-pose p in the 0.5–0.85 band lands at review/violation, never the
    UI's loudest tier. Asserts the band math directly on the incident's score range."""
    pol = V.DEFAULTS
    swimwear_like = np.linspace(0.40, 0.82, 40)   # the measured nudityprobe p range
    tiers = set(V.tier_of(swimwear_like, pol).tolist())
    assert "alert" not in tiers, "an elevated non-extreme p must never reach alert"
    assert tiers <= {"none", "review", "violation"}


# ── config-driven policy (a ruling is an edit, never a retrain) ───────────────
def test_thresholds_are_config_driven():
    assert V.policy({})["tau_violation"] == V.TAU_VIOLATION
    assert V.policy({"tau_violation": 0.9})["tau_violation"] == 0.9
    assert V.policy({"tau_alert": "oops"})["tau_alert"] == V.TAU_ALERT     # typo != outage
    assert "tier_margin" not in V.policy({}), "tier_margin retired with the arbiter"


# ── the shared daemon surface must not silently drift from this module ────────
def test_moderation_json_violence_track_matches_this_module():
    p = Path(__file__).resolve().parents[1] / "src/imgtag/data/moderation.json"
    track = json.loads(p.read_bytes())["categories"]["violence"]  # v2 schema
    spec = V.track_spec()
    for k in ("alert", "violation", "review", "negatives", "scorer",
              "tau_alert", "tau_violation", "tau_review", "platt", "spec_sha"):
        assert track[k] == spec[k], f"moderation.json drifted from violence.py on {k!r}"
    assert track["calibration"] != "fitted", "violence is an FP budget; never claim fitted"
    assert track["enforcement_ready"] is False
    # the measured law: staged/clinical twins are explained, never subtracted
    assert not set(track["policy_neighbours"]) & set(track["negatives"])


def test_head_reports_unfitted():
    head = V.ViolenceHead(V.ViolenceScorer.build(FakeBackend()), "fakesha")
    assert head.calibrated is False and head.enforcement_ready is False
    emb = _unit(np.random.default_rng(5).normal(size=(3, FakeBackend.dim)))
    rows = head.score(emb)
    assert len(rows) == 3
    for r in rows:
        assert r["category"] == "violence" and r["tier"] in set(V.TIERS) | {"none"}
        assert r["calibrated"] is False and r["enforcement_ready"] is False
    p_arr, t_arr = head.probs(emb)
    assert p_arr.shape == (3,) and len(t_arr) == 3


def test_loader_none_on_missing_model(tmp_path):
    """No backend for this machine -> None, never a crash (a missing track is reported,
    never a silent zero)."""
    assert V.load_violence_head({"model": "no-such-model-xyz"}, root=tmp_path) is None
