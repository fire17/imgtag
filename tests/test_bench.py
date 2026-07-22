"""Tests for the b-bench lane: tag table, Platt calibration, fidelity + quality math.

Metric code that is wrong is worse than no metric — it produces confident wrong rankings.
Every metric here is checked against a case whose answer is known by construction.
"""
from __future__ import annotations

import numpy as np
import pytest

from imgtag.bench import quality as Q
from imgtag.bench import quant as QT
from imgtag.core import tags as G


def _unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


# ── fidelity (B24) ────────────────────────────────────────────────────────────
def test_fidelity_identical_is_perfect():
    e = _unit(np.random.default_rng(0).standard_normal((32, 16)).astype(np.float32))
    r = QT.fidelity(e, e)
    assert r["mean_cos"] == pytest.approx(1.0, abs=1e-5)
    assert r["nn_agree"] == 1.0 and r["pass"]


def test_fidelity_catches_rank_flips_that_cosine_hides():
    """The whole point of the NN-agreement clause: high cos, broken neighbours."""
    rng = np.random.default_rng(1)
    ref = _unit(rng.standard_normal((64, 8)).astype(np.float32))
    cand = _unit(ref + rng.standard_normal(ref.shape).astype(np.float32) * 0.06)
    r = QT.fidelity(ref, cand)
    assert r["mean_cos"] > 0.95          # cosine still looks fine
    assert r["nn_agree"] < 0.90          # neighbours are not
    assert not r["pass"]


def test_fidelity_records_pool_size():
    """POOL-SIZE LAW: nn_agree is meaningless without n, so n must always be present."""
    e = _unit(np.random.default_rng(2).standard_normal((10, 4)).astype(np.float32))
    assert QT.fidelity(e, e)["n"] == 10


# ── B6 / B5 / B17 / B7 on constructed data ────────────────────────────────────
def _toy():
    """4 images, 2 categories; text vectors point exactly at their positives."""
    img = _unit(np.array([[1, 0], [1, 0.05], [0, 1], [0.05, 1]], np.float32))
    txt = _unit(np.array([[1, 0], [0, 1]], np.float32))
    pos = {"a": [0, 1], "b": [2, 3]}
    return img, txt, pos


def test_category_precision_perfect_case():
    img, txt, pos = _toy()
    r = Q.category_precision(img, txt, ["a", "b"], pos)
    assert r["mean"] == 1.0 and r["min"] == 1.0 and r["pass"]
    assert r["rows"][0]["k"] == 2  # k = min(10, N_pos), N_pos = 2


def test_category_precision_k_is_capped_by_positives():
    """B6's k=min(10,N_pos) exists because toaster has 8 positives in val2017."""
    img = _unit(np.random.default_rng(3).standard_normal((50, 4)).astype(np.float32))
    txt = _unit(img[:1].copy())
    r = Q.category_precision(img, txt, ["rare"], {"rare": [0]})
    assert r["rows"][0]["k"] == 1 and r["rows"][0]["p_at_k"] == 1.0


def test_category_precision_flags_zeros():
    img, txt, _ = _toy()
    r = Q.category_precision(img, txt, ["a", "b"], {"a": [2, 3], "b": [0, 1]})  # swapped
    assert r["zeros"] == ["a", "b"] and not r["pass"]


def test_hypernym_recall_at_R_and_absent_children():
    img, txt, pos = _toy()
    sup = _unit(np.array([[1.0, 1.0]], np.float32))  # equidistant supercategory query
    r = Q.hypernym(img, sup, {"super": ["a", "b"]}, pos)
    row = r["rows"][0]
    assert row["R"] == 4                       # union of both children's positives
    assert row["precision_at_100"] == 0.04     # 4 relevant of a 100-slot window
    assert all(c["recall_at_R"] == 1.0 for c in row["children"])
    assert r["children_absent_from_top100"] == []


def test_retrieval_ranks_are_zero_based_and_exact():
    img, txt, _ = _toy()
    caps = np.concatenate([txt, txt])           # 4 caption queries
    r = Q.retrieval(img, caps, [0, 2, 1, 3])
    assert r["R@10"] == 100.0 and r["n_queries"] == 4
    assert "NOT Karpathy" in r["corpus"]        # honesty label must survive


def test_retrieval_detects_a_broken_encoder():
    rng = np.random.default_rng(4)
    img = _unit(rng.standard_normal((200, 8)).astype(np.float32))
    caps = _unit(rng.standard_normal((200, 8)).astype(np.float32))  # unrelated
    assert Q.retrieval(img, caps, list(range(200)))["R@10"] < 15.0  # ~chance (5%)


def test_negatives_separates_present_from_absent():
    img, txt, pos = _toy()
    absent = _unit(np.array([[1.0, -1.0]], np.float32))  # orthogonal-ish to everything
    r = Q.negatives(img, txt, absent, ["a", "b"], pos, ["nonsense"])
    assert r["margin_present_minus_absent"] > 0
    assert r["leakage_rate"] == 0.0 and r["pass"]


def test_negatives_reports_unfittable_instead_of_a_meaningless_tau():
    """If recall@10 can't reach 0.70 at ANY tau, say so — never emit a fake threshold."""
    rng = np.random.default_rng(5)
    img = _unit(rng.standard_normal((300, 8)).astype(np.float32))
    txt = _unit(rng.standard_normal((2, 8)).astype(np.float32))
    pos = {"a": list(range(0, 40)), "b": list(range(40, 80))}
    r = Q.negatives(img, txt, txt[:1], ["a", "b"], pos, ["x"])
    assert r["unfittable"] is True and r["tau"] is None and not r["pass"]


# ── tag table + calibration ───────────────────────────────────────────────────
def test_tag_table_is_deduped_ordered_and_tiered():
    t = G.build_tag_table()
    assert len(t) == len(set(t.names)) > 1000
    assert t.names[:3] == ["person", "bicycle", "car"]   # COCO first = truncation-safe
    assert set(t.tier) == {G.CALIBRATED, G.UNCALIBRATED}
    assert sum(x == G.CALIBRATED for x in t.tier) > 1000
    # ADR-3: nothing is calibrated until a CAL-SET fit runs.
    assert all(x is None for x in t.tau)


def test_tag_names_are_normalized():
    t = G.build_tag_table()
    assert all(n == n.lower().strip() for n in t.names)
    assert not any("(" in n or "_" in n for n in t.names)


def test_tag_table_truncation_keeps_coco():
    t = G.build_tag_table(max_tags=90)
    assert len(t) == 90 and "person" in t.names and t.tier[0] == G.CALIBRATED


def test_platt_fit_separates_and_max_f1_tau_is_sane():
    rng = np.random.default_rng(6)
    s = np.r_[rng.normal(0.30, 0.03, 80), rng.normal(0.18, 0.03, 920)]
    y = np.r_[np.ones(80), np.zeros(920)]
    ab = G.fit_platt(s, y)
    p = G.platt_apply(s, ab)
    assert p[:80].mean() > 0.8 and p[80:].mean() < 0.1
    tau, f1 = G.max_f1_tau(p, y)
    assert 0.0 < tau < 1.0 and f1 > 0.9


def test_platt_is_unfittable_without_both_classes():
    assert G.fit_platt(np.linspace(0, 1, 10), np.zeros(10)) == [0.0, 0.0]
    assert G.platt_apply(np.array([0.5]), None) is None


def test_calibrate_never_touches_uncalibrated_tier():
    t = G.TagTable(names=["a", "b"], tier=[G.CALIBRATED, G.UNCALIBRATED],
                   provenance=["coco80", "curated"])
    rng = np.random.default_rng(7)
    scores = np.c_[np.r_[rng.normal(0.4, .02, 50), rng.normal(0.1, .02, 50)],
                   rng.normal(0.2, .05, 100)]
    labels = np.c_[np.r_[np.ones(50), np.zeros(50)], np.r_[np.ones(50), np.zeros(50)]]
    G.calibrate(t, scores, labels)
    assert t.tau[0] is not None and t.platt[0] is not None
    assert t.tau[1] is None and t.platt[1] is None   # uncalibrated may NEVER gate


def test_tag_table_roundtrip(tmp_path):
    t = G.build_tag_table(max_tags=50)
    t.emb = _unit(np.random.default_rng(8).standard_normal((50, 12)).astype(np.float32))
    t.prompt_ensemble_sha = G.prompt_ensemble_sha(["a photo of a {}."])
    G.save(t, "deadbeef", root=str(tmp_path))
    b = G.load("deadbeef", root=str(tmp_path))
    assert b.names == t.names and b.tier == t.tier and b.dim == 12
    assert np.allclose(b.emb, t.emb) and b.prompt_ensemble_sha == t.prompt_ensemble_sha


def test_save_refuses_a_table_without_embeddings(tmp_path):
    with pytest.raises(ValueError):
        G.save(G.build_tag_table(max_tags=5), "x", root=str(tmp_path))


def test_calset_status_is_honest_when_absent():
    s = G.calset_status()
    assert s["ready"] is (s["n_images"] >= 500)


# ── B24 two-tier gate (team-lead ruling) ──────────────────────────────────────
def test_b24_tier_classification():
    assert QT.b24_tier({"mean_cos": 0.99, "nn_agree": 0.92}) == "default"
    assert QT.b24_tier({"mean_cos": 0.99, "nn_agree": 0.80}) == "optin"
    assert QT.b24_tier({"mean_cos": 0.94, "nn_agree": 0.95}) == "banned"  # tier-1 cos floor
    assert QT.b24_tier({"mean_cos": 0.99, "nn_agree": 0.55}) == "banned"  # tier-1 nn floor


def test_platt_no_longer_overflows_on_separable_data():
    import warnings
    rng = np.random.default_rng(11)
    s = np.r_[rng.normal(0.5, 0.01, 100), rng.normal(0.0, 0.01, 900)]
    y = np.r_[np.ones(100), np.zeros(900)]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any RuntimeWarning becomes a failure
        ab = G.fit_platt(s, y)
        p = G.platt_apply(s, ab)
    assert p[:100].mean() > 0.9 and p[100:].mean() < 0.1
