"""Drugs moderation track — unit + contract tests (track-drugs).

Synthetic embeddings only: the MEASURED numbers live in research/track-drugs.md and are
produced by scripts/eval_drugs.py against the labelled slices. A test never re-embeds a
corpus, and never asserts a quality metric it did not measure here.
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np
import pytest

from imgtag.moderation import drugs as D


def _unit(a):
    a = np.asarray(a, np.float32)
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


class FakeBackend:
    """Deterministic pseudo-embeddings: same text -> same vector, no model needed."""

    dim = 32

    def embed_texts(self, texts):
        out = []
        for t in texts:
            # zlib.crc32, NOT hash(): str hashing is salted per process (flaky fixtures)
            rng = np.random.default_rng(zlib.crc32(t.encode()))
            out.append(rng.normal(size=self.dim))
        return _unit(np.asarray(out, np.float32))


# ── prompt hygiene ────────────────────────────────────────────────────────────
def test_banks_are_clean_and_disjoint():
    pos = [c for g in D.CONCEPTS.values() for c in g]
    for lst in (pos, D.BACKGROUND, D.POLICY_NEIGHBOURS, D.TOBACCO):
        assert len(set(lst)) == len(lst), "duplicate prompt"
        # lowercase except acronyms (LSD); no stray whitespace or trailing period
        assert all(p == p.strip() and not p.endswith(".") and "  " not in p for p in lst)
    assert not set(pos) & set(D.BACKGROUND)
    assert not set(pos) & set(D.POLICY_NEIGHBOURS)
    assert not set(D.BACKGROUND) & set(D.POLICY_NEIGHBOURS)


def test_measured_false_positive_families_are_in_the_background_bank():
    """Every FP family the eval actually caught must be represented (research/track-drugs.md):
    snow-as-cocaine, plumbing-as-bong, houseplant-as-cannabis, incense-as-smoke."""
    blob = " ".join(D.BACKGROUND)
    for family in ("snow", "plumbing", "houseplant", "incense", "sugar", "flour"):
        assert family in blob, f"missing measured FP family: {family}"


def test_policy_neighbours_are_never_subtracted():
    """The whole ADR-3-adjacent finding: subtracting visually-identical benign concepts
    (a clinical syringe) destroyed AP 0.58 -> 0.04. They must not reach the score."""
    assert "a medical syringe on a sterile tray" in D.POLICY_NEIGHBOURS
    _, _, bg = D.prompts()
    assert not set(D.POLICY_NEIGHBOURS) & set(bg)


def test_tobacco_is_a_switch_not_a_negative():
    pos_off, groups_off, bg_off = D.prompts(tobacco=False)
    pos_on, groups_on, bg_on = D.prompts(tobacco=True)
    assert not set(D.TOBACCO) & set(pos_off), "tobacco must not be positive by default"
    assert not set(D.TOBACCO) & set(bg_off), "tobacco must not be subtracted either"
    assert set(D.TOBACCO) <= set(pos_on)
    assert "tobacco" in groups_on and "tobacco" not in groups_off
    assert bg_on == bg_off
    assert D.spec_sha(True) != D.spec_sha(False)


def test_ambiguities_are_stated_for_the_user():
    assert len(D.AMBIGUITIES) >= 8
    blob = " ".join(D.AMBIGUITIES).lower()
    for topic in ("tobacco", "alcohol", "cannabis", "syringe", "prescription"):
        assert topic in blob


# ── scorer ────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def scorer():
    return D.DrugsScorer.build(FakeBackend())


def test_vectors_are_l2_normalized(scorer):
    for m in (scorer.pos, scorer.bg, scorer.neighbours):
        assert np.allclose(np.linalg.norm(m, axis=1), 1.0, atol=1e-5)
    assert len(scorer.names) == len(scorer.groups) == len(scorer.pos)


def test_score_contract_and_flag_rule(scorer):
    emb = _unit(np.random.default_rng(0).normal(size=(7, FakeBackend.dim)))
    out = scorer.score(emb)
    assert out["category"] == "drugs"
    assert out["p"].shape == (7,) and ((0 <= out["p"]) & (out["p"] <= 1)).all()
    assert (out["flagged"] == (out["p"] >= out["tau"])).all()
    assert len(out["concept"]) == 7 and all(c in scorer.names for c in out["concept"])
    assert set(out["group"]) <= set(D.CONCEPTS) | {"tobacco"}
    assert len(out["nearest_benign"]) == 7  # review aid always present

    one = scorer.per_image(emb, 3)
    assert set(one) >= {"category", "p", "flagged", "why"}
    assert one["p"] == pytest.approx(float(out["p"][3]))
    assert one["flagged"] is bool(out["flagged"][3])  # a real bool, not np.bool_


def test_probability_is_monotonic_in_margin(scorer):
    """A drug-concept-aligned image must outscore a background-aligned one."""
    emb = _unit(np.stack([scorer.pos[0], scorer.bg[0]]))
    out = scorer.score(emb)
    assert out["margin"][0] > out["margin"][1]
    assert out["p"][0] > out["p"][1]


def test_neighbours_do_not_change_the_score(scorer):
    plain = D.DrugsScorer.build(FakeBackend(), neighbours=False)
    emb = _unit(np.random.default_rng(1).normal(size=(5, FakeBackend.dim)))
    assert np.allclose(plain.score(emb)["p"], scorer.score(emb)["p"])
    assert "nearest_benign" not in plain.score(emb)


# ── the shared daemon surface must not silently drift from this module ────────
def test_moderation_json_drugs_track_matches_this_module():
    p = Path(__file__).resolve().parents[1] / "src/imgtag/data/moderation.json"
    track = json.loads(p.read_bytes())["categories"]["drugs"]  # v2 schema
    spec = D.track_spec()
    for k in ("violation", "review", "negatives", "scorer", "tau", "tau_review",
              "platt", "spec_sha"):
        assert track[k] == spec[k], f"moderation.json drifted from drugs.py on {k!r}"
    assert track["calibration"] != "fitted", "drugs is proxy-fitted; never claim more"
    assert track["enforcement_ready"] is False
    # the measured law: benign look-alikes are explained, never subtracted
    assert not set(track["policy_neighbours"]) & set(track["negatives"])


# ── the tobacco boundary is CONFIG, not code (lead's ask: a ruling, not a retrain) ────
def test_tobacco_tier_is_config_driven():
    assert D.policy({})["tobacco_tier"] == "review"          # ADR-14 default
    assert D.policy({"tobacco_tier": "violation"})["tobacco_tier"] == "violation"
    assert D.policy({"tobacco_tier": "none"})["tobacco_tier"] == "none"
    assert D.policy({"tobacco_tier": "banana"})["tobacco_tier"] == "review"  # typo ≠ outage
    assert D.policy({"tau": 0.5})["tau"] == 0.5
    assert D.policy({"tau": "oops"})["tau"] == D.TAU


def test_config_tier_none_never_emits_review():
    sc = D.DrugsScorer.build(FakeBackend(), config={"tobacco_tier": "none", "tau": 0.99})
    emb = _unit(np.stack([sc.tob[0], sc.pos[0]]))
    assert "review" not in set(sc.score(emb)["tier"])


def test_violation_requires_beating_the_tobacco_bank():
    """The measured v0 failure: a vape exhale scored as a drugs VIOLATION. A violation now
    has to be explained better by the drug bank than by the tobacco bank (TIER_MARGIN)."""
    sc = D.DrugsScorer.build(FakeBackend(), config={"tau": 0.0, "tau_review": 0.0})
    tobacco_like = _unit(sc.tob[0][None])
    out = sc.score(tobacco_like)
    assert out["tier"][0] == "review" and not out["flagged"][0]
    drug_like = sc.score(_unit(sc.pos[0][None]))
    assert drug_like["tier"][0] == "violation"


def test_nothing_that_passes_tau_leaves_the_queue():
    """Losing the arbitration must DEMOTE to review, never drop the image."""
    sc = D.DrugsScorer.build(FakeBackend(), config={"tau": 0.0, "tau_review": 1.0})
    emb = _unit(sc.tob[:2])
    assert set(sc.score(emb)["tier"]) <= {"violation", "review"}


# ── refit v2: the four measured defects b-daemon/b-app reported, each pinned ───────────
def test_review_band_is_below_violation():
    """Defect #1: tau_review > tau made the review tier unreachable. Never again."""
    assert D.TAU_REVIEW < D.TAU
    assert D.track_spec()["tau_review"] < D.track_spec()["tau"]
    # and a config that tries to invert it is repaired, not served
    pol = D.policy({"tau": 0.02, "tau_review": 0.5})
    assert pol["tau_review"] < pol["tau"]


def test_evidence_cap_makes_p099_unreachable():
    """Defect #4: 218 violations all at p=0.99. The cap is (n+1)/(n+2); nothing may exceed it."""
    assert D.P_MAX == (D.N_POSITIVES + 1) / (D.N_POSITIVES + 2)
    assert D.P_MAX < 0.95
    sc = D.DrugsScorer.build(FakeBackend())
    # even an embedding identical to a positive concept cannot reach 0.99
    out = sc.score(_unit(sc.pos[:3]))
    assert out["p"].max() <= D.P_MAX + 1e-9


def test_serrated_leaf_and_benign_object_negatives_present():
    """Defect #3: raspberry leaf scored p=0.92. The FP families are now named negatives."""
    blob = " ".join(D.BACKGROUND)
    for fam in ("raspberry", "japanese maple", "serrated", "fire hydrant", "soap bubbles"):
        assert fam in blob, f"missing b-app FP family: {fam}"


def test_n_positives_matches_labels_file():
    """The evidence cap is only honest if N_POSITIVES tracks the real label count."""
    p = Path(__file__).resolve().parents[1] / "data/drug-probe/labels.json"
    if p.is_file():
        assert D.N_POSITIVES == len(json.loads(p.read_bytes())["drug"])


# ── gate-safe declaration + arbitration-preserving storage (conductor ruling 2026-07-22) ──
def test_gate_safe_declaration_is_measured_not_asserted():
    s = D.track_spec()
    assert s["gate_safe"] is True and s["calibration"] == "proxy-fitted"  # honest label kept
    assert s["evidence_cap"] == round(D.P_MAX, 4)
    ge = s["gate_evidence"]
    assert ge["auroc_tp_vs_fp"] == D.FIT_AUROC and ge["p_negatives_ge_0.9"] == 0
    assert ge["acceptance"] == {"vape": "review", "raspberry_leaf": "none"}


def test_arbitrated_storage_contract_present():
    a = D.track_spec()["arbitrated_storage"]
    assert a["col_roles"] == ["margin", "margin_review"]
    assert a["scorer"] == "margin_arbitrated" and a["tier_margin"] == D.TIER_MARGIN


def test_head_emits_both_arbitration_margins_but_stays_single_col_until_flip():
    """The head carries the arbitration inputs now; storage stays single-col until b-engine's
    derive_tiers honours them (col_roles is the switch, still None)."""
    h = D.DrugsHead(D.DrugsScorer.build(FakeBackend()), "sha")
    assert h.col_roles is None
    emb = _unit(np.random.default_rng(3).normal(size=(4, FakeBackend.dim)))
    for f in h.score(emb):
        assert set(f["cols"]) == {"margin", "margin_review"}
        assert f["tier"] in ("violation", "review", "none")
