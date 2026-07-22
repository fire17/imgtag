"""Sports CONTENT track — unit + contract tests (track-sports).

Fast by construction: everything runs on synthetic embeddings. The MEASURED quality
numbers live in research/track-sports.md and are produced by scripts/_sports_explore.py
against CORPUS-A (COCO val2017) — a test must never re-embed 5000 images to assert a metric.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from imgtag.moderation import sports as S


def _unit(a):
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)


# ── prompt-bank hygiene ───────────────────────────────────────────────────────
def test_prompt_banks_are_clean():
    texts, labels = S.flat_prompts()
    assert len(texts) == len(labels) >= 80
    assert len(set(texts)) == len(texts), "duplicate sport prompt"
    assert all(p == p.strip() and not p.endswith(".") for p in texts)
    bg = S.background_prompts()
    assert len(set(bg)) == len(bg), "duplicate background prompt"
    # positives and background must not overlap — a prompt can't be sport and not-sport
    assert not set(texts) & set(bg)


def test_labels_are_parallel_to_prompts():
    texts, labels = S.flat_prompts()
    assert len(texts) == len(labels)
    # every declared sport group is represented, in contiguous order
    assert set(labels) == set(S.SPORT_PROMPTS)
    for g, prompts in S.SPORT_PROMPTS.items():
        idx = [i for i, l in enumerate(labels) if l == g]
        assert [texts[i] for i in idx] == prompts


def test_borderline_is_out_by_default_and_in_when_asked():
    off = S.background_prompts(borderline=False)
    on_texts, _ = S.flat_prompts(borderline=True)
    border = [p for ps in S.BORDERLINE_PROMPTS.values() for p in ps]
    # default: borderline prompts live in the BACKGROUND bank (suppressed)
    assert set(border) <= set(off)
    # borderline=True: they move to the POSITIVE bank instead
    assert set(border) <= set(on_texts)
    assert not set(border) & set(S.background_prompts(borderline=True))


def test_negatives_are_generic_only():
    """The load-bearing measured result: no scene hard-negatives in the base bank."""
    blob = " ".join(S.NEGATIVE_PROMPTS)
    for banned in ("stadium", "gymnasium", "mountain", "beach", "pool", "field"):
        assert banned not in blob, f"scene hard-negative leaked into base bank: {banned}"


# ── scorer math ───────────────────────────────────────────────────────────────
def test_margin_is_signed_contrast():
    pos = _unit(np.array([[1.0, 0.0], [0.9, 0.1]], np.float32))
    neg = _unit(np.array([[0.0, 1.0]], np.float32))
    emb = _unit(np.array([[1.0, 0.0], [0.0, 1.0]], np.float32))
    m = S.margin(emb, pos, neg)
    assert m[0] > 0 > m[1]
    assert m.shape == (2,)


def test_squash_monotone_and_bounded():
    m = np.array([-0.3, 0.0, 0.2, 0.5])
    p = S.squash(m)
    assert np.all(np.diff(p) > 0)
    assert 0.0 <= p.min() and p.max() <= 1.0


def test_tier_of():
    assert S.tier_of(0.9, 0.5) == "match"
    assert S.tier_of(0.5, 0.5) == "match"        # ties flag (>=)
    assert S.tier_of(0.4, 0.5) == "none"


def test_tau_for_precision_hits_target():
    rng = np.random.default_rng(0)
    y = rng.random(2000) < 0.2
    p = np.clip(0.5 * y + 0.25 * rng.random(2000), 0, 1)   # separable-ish
    tau = S.tau_for_precision(p, y, 0.80)
    m = S.prf(p, y, tau)
    assert m["precision"] >= 0.80 - 1e-9
    # and it is the SMALLEST such threshold: nudging down breaks the target
    below = S.prf(p, y, tau - 1e-3)
    assert below["recall"] >= m["recall"]


# ── head contract (the dispatcher-facing seam) ────────────────────────────────
class _Backend:
    name = "toy"
    model_sha = "toysha"

    def __init__(self, d=16, seed=1):
        self.rng = np.random.default_rng(seed)
        self.d = d
        self._cache = {}

    def embed_texts(self, texts):
        out = []
        for t in texts:
            if t not in self._cache:
                self._cache[t] = self.rng.normal(size=self.d)
            out.append(self._cache[t])
        return _unit(np.asarray(out, np.float32))


def _head():
    return S.SportsHead.build(_Backend())


def test_score_schema_and_content_routing():
    h = _head()
    emb = _unit(np.random.default_rng(2).normal(size=(20, 16)).astype(np.float32))
    flags = h.score(emb)
    assert len(flags) == 20
    for f in flags:
        assert f["category"] == "sports"
        assert f["tier"] in ("match", "none")
        assert f["content_track"] is True          # routes to the content bucket
        assert f["enforcement_ready"] is False      # a content label never enforces
        assert 0.0 <= f["p"] <= 1.0
        if f["tier"] == "match":
            assert f["label"] == f["sport"]         # cross-track alias
            assert f["label"] in S.SPORT_PROMPTS
        else:
            assert "label" not in f                 # no sport name on a non-match


def test_head_declares_content_track():
    h = _head()
    assert h.content_track is True
    assert h.enforcement_ready is False
    assert S.SportsHead.wants_images is False       # never re-decode images


def test_score_rejects_wrong_dim():
    h = _head()
    with pytest.raises(ValueError):
        h.score(np.zeros((3, 999), np.float32))


def test_fit_calibrates_and_sets_tau():
    rng = np.random.default_rng(3)
    b = _Backend()
    h = S.SportsHead.build(b)
    # synthesize labels correlated with the margin so a fit is meaningful
    emb = _unit(rng.normal(size=(1000, 16)).astype(np.float32))
    m = h.margins(emb)
    y = m > np.median(m)
    tr, va = slice(0, 500), slice(500, 1000)
    S.fit(h, emb[tr], y[tr], val=(emb[va], y[va]))
    assert h.calibrated is True
    assert h.platt is not None
    assert h.metrics["held_out"] is True
    # τ_match is a MARGIN threshold (the reader gates in margin space), so it lives in the
    # margin's range [-1, 1] — NOT a probability. This is the guard against the reader-0-match
    # bug: a probability-space τ (~0.18) exceeds the max attainable margin → 0 matches.
    assert h.metrics["tau_space"] == "margin"
    assert -1.0 <= h.tau_match <= 1.0
    assert h.tau_match <= float(m.max())          # some image can actually clear it


def test_gates_in_margin_space_not_probability():
    """Regression for the head≠reader divergence: the tier MUST be decided on the raw margin
    vs τ_match (what b-daemon's fitted reader does: `margin - tau >= 0`), never on the Platt
    probability. A steep Platt makes probability-space and margin-space gating disagree, and
    the reader is the authority."""
    rng = np.random.default_rng(11)
    h = S.SportsHead.build(_Backend())
    emb = _unit(rng.normal(size=(400, 16)).astype(np.float32))
    y = h.margins(emb) > np.median(h.margins(emb))
    S.fit(h, emb, y)
    m = h.margins(emb)
    flags = h.score(emb)
    # head's match set == the reader's margin-space gate, exactly
    head_match = np.array([f["tier"] == "match" for f in flags])
    reader_gate = m >= h.tau_match
    assert np.array_equal(head_match, reader_gate)
    # and the reported p is Platt(margin), decoupled from the gate
    assert np.allclose([f["p"] for f in flags], np.round(h.probs(emb), 4), atol=1e-4)


# ── persistence: fp16 round-trip + b-daemon's fitted-file keys ────────────────
def test_json_roundtrip_fp16(tmp_path):
    h = _head()
    S.fit(h, *(_fit_data(h)))
    p = S.save(h, root=tmp_path)
    assert p.stat().st_size < 400_000            # fp16 keeps it small (not 600KB of floats)
    d = json.loads(p.read_text())
    # keys b-daemon's spec reader honours off the fitted file
    assert d["scorer"] == "margin"
    assert d["calibration"] == "fitted"
    assert "tau_match" in d and len(d["platt"]) == 2
    h2 = S.load_head(h.model_id, root=tmp_path)
    assert h2.pos.shape == h.pos.shape and h2.labels == h.labels
    # fp16 storage is faithful enough that tiers do not flip
    emb = _unit(np.random.default_rng(4).normal(size=(50, 16)).astype(np.float32))
    t1 = [f["tier"] for f in h.score(emb)]
    t2 = [f["tier"] for f in h2.score(emb)]
    assert t1 == t2


def _fit_data(h):
    rng = np.random.default_rng(5)
    emb = _unit(rng.normal(size=(600, h.dim)).astype(np.float32))
    y = h.margins(emb) > np.median(h.margins(emb))
    return emb, y, (emb[300:], y[300:])


def test_model_sha_refusal(tmp_path):
    h = _head()
    S.save(h, root=tmp_path)
    with pytest.raises(ValueError):
        S.load_head(h.model_id, root=tmp_path, model_sha="a-different-sha")


def test_unfitted_head_reads_unfitted(tmp_path):
    h = _head()                                    # never fit
    p = S.save(h, root=tmp_path)
    assert json.loads(p.read_text())["calibration"] == "unfitted"


# ── spec ⇄ module agreement (the two scorers must embed identical strings) ────
def test_moderation_spec_matches_module():
    spec = json.loads((Path(S.__file__).resolve().parent.parent
                       / "data" / "moderation.json").read_text())
    if "sports" not in spec["categories"]:
        pytest.skip("sports entry not yet committed to the shared moderation.json "
                    "(written in the working tree; landing is sequenced with other lanes)")
    sp = spec["categories"]["sports"]
    assert sp["content_track"] is True
    assert sp["enforcement_ready"] is False
    texts, labels = S.flat_prompts()
    assert sp["match"] == texts
    assert sp["match_labels"] == labels
    # negatives in the spec == the shipped background bank (borderline folded in)
    assert sp["negatives"] == S.background_prompts()


# ── zero-shot path ────────────────────────────────────────────────────────────
def test_zero_shot_is_never_calibrated():
    z = S.ZeroShotSportsHead(_Backend())
    assert z.calibrated is False and z.enforcement_ready is False and z.content_track is True
    emb = _unit(np.random.default_rng(6).normal(size=(10, 16)).astype(np.float32))
    for f in z.score(emb):
        assert f["calibrated"] is False
        assert f["model_id"].startswith("zeroshot:")
