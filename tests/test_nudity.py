"""track-nudity — contract, policy and artifact tests.

The 22.5MB ONNX head is gitignored (models/**/*.onnx), so every test that needs it is
skipped rather than failed on a host that has not run scripts/export_nudity_marqo.py.
The contract and ADR-14 policy tests need no artifact at all and always run.
"""

import numpy as np
import pytest
from PIL import Image

from imgtag.moderation import load_heads, nudity

HAS_ARTIFACT = nudity.find_artifact(nudity.SPEC, nudity.ARTIFACT) is not None
needs_head = pytest.mark.skipif(not HAS_ARTIFACT, reason="nudity ONNX artifact not on this host")


class FakeBackend:
    """Text tower stand-in: 'nude'-ish prompts point one way, everything else another."""

    model_sha = "fake"
    dim = 4

    def embed_texts(self, texts):
        v = np.zeros((len(texts), self.dim), np.float32)
        for i, t in enumerate(texts):
            v[i, 0 if any(w in t for w in ("nude", "naked", "porn", "topless", "bare", "exposed", "explicit")) else 1] = 1.0
        return v


# ---------------------------------------------------------------- ADR-14 policy


def test_tier_boundaries_are_adr14_not_a_single_flag():
    assert nudity.tier_of(0.99) == "violation"
    assert nudity.tier_of(nudity.TAU_VIOLATION) == "violation"       # inclusive
    assert nudity.tier_of(nudity.TAU_VIOLATION - 1e-6) == "review"
    assert nudity.tier_of(nudity.TAU_REVIEW) == "review"
    assert nudity.tier_of(nudity.TAU_REVIEW - 1e-6) == "none"
    assert nudity.tier_of(0.0) == "none"


def test_review_tau_sits_below_violation_tau():
    assert 0.0 < nudity.TAU_REVIEW < nudity.TAU_VIOLATION < 1.0


# ---------------------------------------------------------------- zero-shot baseline


def test_zero_shot_head_is_never_calibrated_and_never_reaches_violation():
    zs = nudity.ZeroShotNudityHead(FakeBackend(), tau=0.5)
    embs = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)  # "nude"-aligned, then not
    out = zs.score(embs)
    assert [f["tier"] for f in out] == ["review", "none"]
    assert all(f["calibrated"] is False for f in out)
    assert all(f["category"] == "nudity" for f in out)
    assert "violation" not in {f["tier"] for f in out}


def test_zero_shot_margin_is_positive_minus_background():
    zs = nudity.ZeroShotNudityHead(FakeBackend())
    m = zs.margins(np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32))
    assert m[0] == pytest.approx(1.0) and m[1] == pytest.approx(-1.0)


# ---------------------------------------------------------------- the real head


@needs_head
def test_head_loads_and_declares_what_it_needs():
    h = nudity.load_nudity_head({"intra_op": 1})
    assert h.category == "nudity" and h.wants_images is True
    assert h.size == 384 and h.squash is False
    assert h.calibrated is False, "tau was never fitted on labeled nudity ground truth"
    assert len(h.model_sha) == 64


@needs_head
def test_preprocess_geometry_matches_the_models_own_config():
    h = nudity.load_nudity_head({"intra_op": 1})
    a = h.preprocess(Image.new("RGB", (900, 300), (10, 200, 30)))
    assert a.shape == (384, 384, 3) and a.dtype == np.uint8


@needs_head
def test_probabilities_are_a_two_class_softmax_in_range():
    h = nudity.load_nudity_head({"intra_op": 1})
    batch = np.stack([h.preprocess(Image.new("RGB", (64, 64), c))
                      for c in ((0, 0, 0), (255, 255, 255), (120, 90, 70))])
    p = h.probs(batch)
    assert p.shape == (3,) and p.dtype == np.float32
    assert np.all((p >= 0) & (p <= 1))


@needs_head
def test_label_order_sanity_flat_colours_are_not_nudity():
    """If NSFW_INDEX were flipped, every blank image would score ~0.95 instead of ~0.05."""
    h = nudity.load_nudity_head({"intra_op": 1})
    p = h.probs(np.stack([h.preprocess(Image.new("RGB", (256, 256), c))
                          for c in ((0, 0, 0), (255, 255, 255), (30, 120, 200))]))
    assert p.max() < 0.5, f"flat colours scored {p} — label index or normalization is wrong"


@needs_head
def test_score_returns_the_dispatcher_schema_per_record(tmp_path):
    h = nudity.load_nudity_head({"intra_op": 1})
    paths = []
    for i, c in enumerate(((10, 20, 30), (200, 180, 160))):
        p = tmp_path / f"img{i}.jpg"
        Image.new("RGB", (128, 96), c).save(p)
        paths.append({"path": str(p)})
    out = h.score(np.zeros((2, 512), np.float32), None, paths)
    assert len(out) == 2
    for f in out:
        assert set(("category", "p", "tier")) <= set(f)
        assert f["category"] == "nudity"
        assert 0.0 <= f["p"] <= 1.0
        assert f["tier"] in ("violation", "review", "none")


@needs_head
def test_an_unreadable_file_is_unflagged_never_an_exception(tmp_path):
    h = nudity.load_nudity_head({"intra_op": 1})
    bad = tmp_path / "truncated.jpg"
    bad.write_bytes(b"\xff\xd8\xff\xe0 not really a jpeg")
    good = tmp_path / "ok.jpg"
    Image.new("RGB", (64, 64), (5, 5, 5)).save(good)
    out = h.score(np.zeros((2, 512), np.float32), None, [{"path": str(bad)}, {"path": str(good)}])
    assert len(out) == 2
    assert out[0]["tier"] == "none" and out[0].get("unreadable") is True
    assert "unreadable" not in out[1]


@needs_head
def test_slab_pixels_are_only_used_when_their_geometry_matches_ours():
    """A 224² backend slab must NOT be fed to a 384-center-crop model as if it were."""
    h = nudity.load_nudity_head({"intra_op": 1})
    slab = np.zeros((2, 224, 224, 3), np.uint8)
    out = h.score(np.zeros((2, 512), np.float32), slab, [{"path": "/nonexistent-a.jpg"},
                                                         {"path": "/nonexistent-b.jpg"}])
    # geometry mismatch -> reopen path -> both files missing -> unflagged, not crashed
    assert [f["tier"] for f in out] == ["none", "none"]
    assert all(f.get("unreadable") for f in out)


# ---------------------------------------------------------------- negative control


def _probe_batch():
    """Content-free probes. The model scores a solid flesh tone at p=0.55 and a flesh
    gradient at 0.76 (its own OOD behaviour, reproduced in torch) — the guard, not the
    threshold, is what must stop them."""
    rng = np.random.default_rng(0)
    g = np.tile(np.linspace(0, 255, 384, dtype=np.uint8)[None, :, None], (384, 1, 3))
    return {
        "black": np.zeros((384, 384, 3), np.uint8),
        "white": np.full((384, 384, 3), 255, np.uint8),
        "grey": np.full((384, 384, 3), 128, np.uint8),
        "flesh": np.full((384, 384, 3), (222, 180, 150), np.uint8),
        "worst-solid": np.full((384, 384, 3), (255, 192, 255), np.uint8),
        "gradient": g,
        "flesh-gradient": np.clip(g * 0.6 + np.array([120, 70, 50]), 0, 255).astype(np.uint8),
        "noise": rng.integers(0, 256, (384, 384, 3), dtype=np.uint8),
    }


def test_structure_separates_synthetic_from_photographs():
    p = _probe_batch()
    s = dict(zip(p, nudity.structure(np.stack(list(p.values())))))
    assert s["black"] == s["white"] == s["grey"] == s["flesh"] == 0.0
    # second-order on purpose: a LINEAR gradient is as content-free as a solid colour
    assert s["gradient"] < nudity.MIN_STRUCTURE
    assert s["flesh-gradient"] < nudity.MIN_STRUCTURE
    assert s["noise"] > 50  # unstructured but far from content-free
    # the floor sits below the minimum measured over 1,826 real photographs (1.171)
    assert nudity.MIN_STRUCTURE < 1.171


@needs_head
def test_negative_control_content_free_images_never_flag():
    """PERMANENT negative control (team-lead 2026-07-22): flat colours, gradients and
    noise must produce ZERO flags at the operating point."""
    h = nudity.load_nudity_head({"intra_op": 1})
    probes = _probe_batch()
    by = dict(zip(probes, h._flags(np.stack(list(probes.values())))))

    # NOTHING content-free may ever reach the violation tier — this is the hard line.
    for name, f in by.items():
        assert f["tier"] != "violation", f"{name} reached violation at p={f['p']}"

    # Solid colours and gradients — the class that DID reach violation on raw p
    # (flesh 0.55, flesh-gradient 0.76) — are silenced outright, and say why.
    for name in ("black", "white", "grey", "flesh", "worst-solid", "gradient", "flesh-gradient"):
        assert by[name]["tier"] == "none", f"{name} flagged {by[name]['tier']} at p={by[name]['p']}"
    assert by["flesh"]["content_free"] is True and by["flesh"]["p"] > 0.5
    assert by["flesh-gradient"]["content_free"] is True and by["flesh-gradient"]["p"] > 0.5

    # White noise is the documented exception: it has genuine spatial structure, so the
    # guard correctly does not claim it, and the model puts it at ~0.16 — review tier, by
    # design (review is a human queue). Distorting τ to chase an input no camera produces
    # would cost real recall. research/track-nudity.md §9.
    assert "content_free" not in by["noise"]
    assert by["noise"]["p"] < h.tau_violation


@needs_head
def test_a_real_photograph_is_never_marked_content_free(tmp_path):
    h = nudity.load_nudity_head({"intra_op": 1})
    rng = np.random.default_rng(1)
    p = tmp_path / "photo.jpg"
    Image.fromarray(rng.integers(60, 200, (300, 400, 3), dtype=np.uint8)).save(p, quality=92)
    f = h.score(np.zeros((1, 512), np.float32), None, [{"path": str(p)}])[0]
    assert "content_free" not in f


# ---------------------------------------------------------------- pre-made view (B16)


@pytest.fixture
def photo(tmp_path):
    """A real-ish JPEG big enough that draft() actually engages (>2× our 384)."""
    rng = np.random.default_rng(7)
    a = rng.integers(0, 256, (60, 100, 3), dtype=np.uint8).repeat(16, 0).repeat(16, 1)
    p = tmp_path / "photo.jpg"
    Image.fromarray(a).save(p, quality=95)
    return p


def test_worker_view_is_bit_identical_to_our_own_preprocess(photo):
    """B16: the worker's view MUST equal what this track would have computed itself.
    Guaranteed by construction — the worker calls our make_view — and asserted anyway, so
    a future reimplementation on either side is caught here and not in production."""
    with Image.open(photo) as im:
        worker = nudity.make_view(im)
    with Image.open(photo) as im:
        ours = nudity.preprocess_image(im, nudity.SIZE, nudity.SQUASH, Image.Resampling.BICUBIC)
    assert np.array_equal(worker, ours)
    assert worker.dtype == np.uint8 and worker.shape == (384, 384, 3)


def test_the_draft_precondition_is_real_and_would_change_the_pixels(photo):
    """The hazard bit-parity alone cannot catch: the SAME transform on a differently
    drafted decode is a different image. Documents why the worker may only ship a view
    when its decode was drafted at 384 (measured: 224-first differs by up to 33/255)."""
    with Image.open(photo) as im:
        ref = nudity.make_view(im)
    with Image.open(photo) as im:
        im.draft("RGB", (224, 224))          # a 224 backend's decode
        wrong = nudity.make_view(im)
    assert not np.array_equal(ref, wrong), "draft scale no longer affects the view — re-check the contract"
    with Image.open(photo) as im:
        im.draft("RGB", (384, 384))          # a 384 backend's decode — the allowed case
        right = nudity.make_view(im)
    assert np.array_equal(ref, right)


@needs_head
def test_an_offered_view_is_used_and_matches_the_reopen_path(photo):
    """The whole point: taking the offered view must not change a single score."""
    h = nudity.load_nudity_head({"intra_op": 1})
    ids = [{"path": str(photo)}]
    with Image.open(photo) as im:
        view = nudity.make_view(im)[None]
    viaview = h.score(np.zeros((1, 512), np.float32), None, ids, views={nudity.VIEW_KEY: view})
    reopen = h.score(np.zeros((1, 512), np.float32), None, ids)
    assert viaview[0]["p"] == reopen[0]["p"] and viaview[0]["tier"] == reopen[0]["tier"]


@needs_head
def test_a_malformed_view_falls_back_instead_of_being_fed_to_the_model(photo):
    h = nudity.load_nudity_head({"intra_op": 1})
    ids = [{"path": str(photo)}]
    good = h.score(np.zeros((1, 512), np.float32), None, ids)[0]
    for bad in (np.zeros((1, 224, 224, 3), np.uint8),      # wrong geometry
                np.zeros((2, 384, 384, 3), np.uint8),      # wrong batch length
                np.zeros((1, 384, 384, 3), np.float32)):   # wrong dtype
        f = h.score(np.zeros((1, 512), np.float32), None, ids, views={nudity.VIEW_KEY: bad})[0]
        assert f["p"] == good["p"], f"malformed view {bad.shape}/{bad.dtype} was not rejected"
    # an unrelated key is simply ignored
    f = h.score(np.zeros((1, 512), np.float32), None, ids, views={"something-else": None})[0]
    assert f["p"] == good["p"]


@needs_head
def test_score_still_works_when_the_caller_passes_no_views_at_all(photo):
    """b-engine's current 3-positional-arg call site must keep working unchanged."""
    h = nudity.load_nudity_head({"intra_op": 1})
    assert h.score(np.zeros((1, 512), np.float32), None, [{"path": str(photo)}])[0]["category"] == "nudity"


@needs_head
def test_the_dispatcher_finds_this_track():
    heads = load_heads({"intra_op": 1})
    assert "nudity" in heads and heads["nudity"].wants_images is True


# ---------------------------------------------------------------- review-tier instrument
# (track-nudity3 2026-07-22): the Marqo head cannot carry the review tier (swimwear p50
# 0.056 == mannequin p50 0.056, measured) — review is a prompt-ensemble MARGIN over the
# index embeddings, taxonomy-driven from moderation.json. These tests need NO ONNX artifact
# and NO real backend: an axis-aligned fake text space makes the margin exactly checkable.


class ReviewFakeBackend:
    """One axis per concept family, so margin = max(review) - max(neg) is exact:
    swimwear/lingerie/underwear/bare-chest each own a POSITIVE axis; clothed / mannequin /
    generic negatives each own a NEGATIVE axis."""

    model_sha = "fake-review"
    AXES = ["swim", "lingerie", "underwear", "barechest", "skin", "clothed", "mannequin", "generic"]

    def _axis(self, t: str) -> int:
        t = t.lower()
        if any(w in t for w in ("bikini", "swim", "swimsuit")):
            return 0
        if "lingerie" in t or "bra" in t:
            return 1
        if "underwear" in t or "briefs" in t or "boxer" in t:
            return 2
        if "shirtless" in t or "bare-chested" in t or "muscular" in t:
            return 3
        if any(w in t for w in ("artistic", "fine-art", "bare human form", "medical", "dermatology", "skin")):
            return 4
        if "clothed" in t or "wearing clothes" in t:
            return 5
        if any(w in t for w in ("mannequin", "statue", "sculpture", "dummy", "anatomical", "marble")):
            return 6
        return 7

    def embed_texts(self, texts):
        v = np.zeros((len(texts), len(self.AXES)), np.float32)
        for i, t in enumerate(texts):
            v[i, self._axis(t)] = 1.0
        return v


def _onehot(axis: int, dim: int = 8) -> np.ndarray:
    v = np.zeros((1, dim), np.float32)
    v[0, axis] = 1.0
    return v


def test_the_review_taxonomy_is_versioned_data_in_moderation_json():
    """The user's explicit ask: per-subcategory prompts live as DATA, not code."""
    spec = nudity._review_spec()
    subs = spec.get("review_subcategories") or {}
    assert {"swimwear", "lingerie", "underwear", "bare-chest-male"} <= set(subs)
    assert spec.get("scorer") == "margin" and "platt" in spec
    assert spec.get("enforcement_ready") is False  # never fitted on labeled nudity GT


def test_review_scorer_flags_swimwear_and_names_the_subcategory():
    sc = nudity.NudityReviewScorer.build(ReviewFakeBackend())
    out = sc.score(_onehot(0))  # a swimwear-axis image
    assert out[0]["tier"] == "review"
    assert out[0]["subcategory"] == "swimwear"
    assert out[0]["category"] == "nudity" and out[0]["calibrated"] is False


def test_review_scorer_keeps_the_mannequin_control_at_none():
    """ADR-14 hard requirement: non-person figures never flag. The mannequin/statue
    concepts are in the SUBTRACTED negatives, so a mannequin image maxes a negative and its
    margin goes negative -> none."""
    sc = nudity.NudityReviewScorer.build(ReviewFakeBackend())
    assert sc.score(_onehot(6))[0]["tier"] == "none"   # mannequin axis
    assert sc.score(_onehot(5))[0]["tier"] == "none"   # clothed axis


def test_review_scorer_never_emits_violation():
    sc = nudity.NudityReviewScorer.build(ReviewFakeBackend())
    embs = np.eye(8, dtype=np.float32)  # one image per axis
    tiers = {f["tier"] for f in sc.score(embs)}
    assert "violation" not in tiers and tiers <= {"review", "none"}


def test_review_margin_is_max_positive_minus_max_negative():
    sc = nudity.NudityReviewScorer.build(ReviewFakeBackend())
    m, best = sc.margins(_onehot(0))
    assert m[0] == pytest.approx(1.0)          # swim concept dot 1, every negative dot 0
    m2, _ = sc.margins(_onehot(6))
    assert m2[0] == pytest.approx(-1.0)        # mannequin: no positive, one negative dot 1
