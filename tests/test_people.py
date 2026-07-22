"""People/face counting track — unit + contract + acceptance tests (track-people).

Fast by construction: the graph tests below build a tiny fake ONNX session and synthetic
embeddings, so the suite never downloads a model or an image. The MEASURED quality numbers
live in research/track-people.md and are produced by scripts/train_people_head.py against
COCO val2017; a test must never re-derive a metric from 5,000 images.

The one test that touches real artifacts (`test_acceptance_*`) is skipped cleanly when the
YuNet model or the COCO index is absent, so CI without the data still passes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from imgtag.moderation import people as P

ROOT = Path(__file__).resolve().parents[1]


# ── the DERIVED categories (TRACKS.md T1: derived from raw, at read) ──────────
def test_derive_is_a_pure_function_of_counts():
    assert P.derive(0, 0) == {"one-person": False, "multi-person": False,
                              "one-face": False, "multi-face": False}
    assert P.derive(1, 0) == {"one-person": True, "multi-person": False,
                              "one-face": False, "multi-face": False}   # back-view hiker
    assert P.derive(1, 1)["one-face"] and P.derive(1, 1)["one-person"]
    assert P.derive(2, 2) == {"one-person": False, "multi-person": True,
                              "one-face": False, "multi-face": True}    # couple selfie
    assert P.derive(9, 5)["multi-person"] and P.derive(9, 5)["multi-face"]
    # exactly-one is exclusive with multi, by construction
    for n in range(6):
        d = P.derive(n, n)
        assert not (d["one-person"] and d["multi-person"])
        assert not (d["one-face"] and d["multi-face"])


# ── decode: the numpy reimplementation of YuNet's post-process ────────────────
def _fake_outputs(size=P.INPUT_SIZE, hits=()):
    """Build zeroed YuNet head tensors, then plant `hits` = [(stride, cell, score)]."""
    outs = {}
    for s in P.STRIDES:
        g = size // s
        outs[f"cls_{s}"] = np.zeros((1, g * g, 1), np.float32)
        outs[f"obj_{s}"] = np.zeros((1, g * g, 1), np.float32)
        outs[f"bbox_{s}"] = np.zeros((1, g * g, 4), np.float32)
        outs[f"kps_{s}"] = np.zeros((1, g * g, 10), np.float32)
    for s, cell, score in hits:
        outs[f"cls_{s}"][0, cell, 0] = score
        outs[f"obj_{s}"][0, cell, 0] = score
        # a real box so NMS keeps it: width/height ~ one stride
        outs[f"bbox_{s}"][0, cell] = [0.5, 0.5, 0.0, 0.0]
    return outs


def test_decode_score_is_geometric_mean_of_cls_and_obj():
    outs = _fake_outputs(hits=[(8, 100, 0.81)])
    boxes, scores = P.decode(outs)
    top = scores.max()
    assert abs(top - np.sqrt(0.81 * 0.81)) < 1e-5   # sqrt(cls*obj) == 0.81
    assert boxes.shape[0] == scores.shape[0] == 6400 + 1600 + 400


def test_decode_box_geometry_is_centre_offset_log_size():
    # cell 0 at stride 8, offsets 0.5/0.5, log-size 0 -> centre (4,4), size (8,8)
    outs = _fake_outputs(hits=[(8, 0, 0.9)])
    boxes, scores = P.decode(outs)
    i = int(scores.argmax())
    x, y, w, h = boxes[i]
    assert (abs(w - 8) < 1e-4 and abs(h - 8) < 1e-4)
    assert (abs((x + w / 2) - 4) < 1e-4 and abs((y + h / 2) - 4) < 1e-4)


def test_nms_collapses_duplicates_and_keeps_distinct():
    # two boxes on the same cell-neighbourhood overlap; a far one stays
    boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [500, 500, 10, 10]], np.float32)
    scores = np.array([0.9, 0.8, 0.7], np.float32)
    keep = P.nms(boxes, scores, iou_thr=0.3)
    assert len(keep) == 2 and 0 in keep and 2 in keep


def test_nms_empty_is_empty():
    assert P.nms(np.empty((0, 4), np.float32), np.empty(0, np.float32)) == []


# ── count confidence: both error directions are priced ───────────────────────
def test_count_confidence_prices_the_weakest_accept_and_strongest_reject():
    # clean: a strong kept det, nothing rejected near the line -> high confidence
    assert P.count_confidence(np.array([0.95]), below=0.0) > 0.9
    # a strong reject just under tau erodes confidence in the count
    assert P.count_confidence(np.array([0.95]), below=0.9) < 0.15
    # empty frame with nothing near the line -> confident zero
    assert P.count_confidence(np.empty(0), below=0.0) == 1.0
    # empty frame with a near-miss -> less confident it is really zero
    assert P.count_confidence(np.empty(0), below=0.55) < 0.5


# ── head contract (ADR-14 seam, TRACKS.md T1 every-image law) ─────────────────
class _FakeSession:
    """Minimal onnxruntime stand-in: returns planted YuNet tensors for any input."""

    def __init__(self, hits=()):
        self._hits = hits
        self._names = [f"{k}_{s}" for k in ("cls", "obj", "bbox", "kps") for s in P.STRIDES]

    def get_inputs(self):
        return [type("I", (), {"name": "input"})()]

    def get_outputs(self):
        return [type("O", (), {"name": n})() for n in self._names]

    def run(self, out_names, feed):
        o = _fake_outputs(hits=self._hits)
        return [o[n] for n in out_names]


def test_head_emits_raw_multicol_record_every_image():
    """T1: every image gets the RAW `people` multi-col record (col_roles order), tier none.

    An empty frame satisfies no category, so it emits exactly the one raw record — never
    silently absent, never a spurious chip.
    """
    head = P.PeopleHead(_FakeSession(), cascade=None)
    out = head.score(np.zeros((1, 512), np.float32), images=[None],
                     ids=[{"path": "/does/not/exist.jpg"}])
    assert len(out) == 1
    recs = out[0]
    raw = recs[0]
    assert raw["category"] == "people" and raw["tier"] == "none"
    assert list(raw["cols"]) == P.PeopleHead.col_roles   # dict in the on-disk column order
    assert raw["enforcement_ready"] is False             # counting track never authorizes
    # a no-person frame emits ONLY the raw record (no satisfied chip)
    assert [r for r in recs if r["tier"] == "match"] == []
    # the four user categories are a pure read-time function of the two count columns
    d = P.derive(int(raw["cols"]["n_persons"]), int(raw["cols"]["n_faces"]))
    assert set(d) == set(P.DERIVED)


def test_col_roles_matches_dispatcher_contract():
    """The head is the single authority for its column schema (b-engine reads this attr)."""
    assert P.PeopleHead.col_roles == ["n_persons", "n_faces", "n_persons_conf", "n_faces_conf"]
    spec = P.PeopleHead(_FakeSession(), cascade=None).spec
    assert set(spec["bands"]) == set(P.DERIVED) and spec["bands"]["multi-person"] == [2, None]


def test_unreadable_file_never_raises_and_marks_the_record():
    head = P.PeopleHead(_FakeSession(), cascade=None)
    out = head.score(np.zeros((1, 512), np.float32), images=[None],
                     ids=[{"path": "/nope.jpg"}])[0]
    assert out[0]["category"] == "people" and out[0]["tier"] == "none"
    assert out[0].get("unreadable") is True
    assert out[0]["cols"]["n_persons"] == 0.0 and out[0]["cols"]["n_faces"] == 0.0
    assert [r for r in out if r["tier"] == "match"] == []   # nothing satisfied


def test_head_counts_faces_from_optics(tmp_path):
    """Three planted detections at distinct cells -> three faces -> multi-face match."""
    hits = [(8, 100, 0.95), (8, 4000, 0.95), (16, 50, 0.95)]
    head = P.PeopleHead(_FakeSession(hits), cascade=None, tau_face=0.6)

    from PIL import Image
    p = tmp_path / "x.png"
    Image.new("RGB", (64, 48), (120, 120, 120)).save(p)
    out = head.score(np.zeros((1, 512), np.float32), images=[None], ids=[{"path": str(p)}])[0]
    raw = out[0]
    assert raw["cols"]["n_faces"] == 3.0                  # raw count is the sidecar value
    assert raw["cols"]["n_persons"] >= 3.0                # faces lower-bound persons
    chips = {r["category"] for r in out if r["tier"] == "match"}
    assert "multi-face" in chips and "multi-person" in chips   # satisfied categories fire


# ── cascade persistence roundtrip ─────────────────────────────────────────────
def test_cascade_json_roundtrip():
    c = P.PersonCascade(model_id="m", dim=4,
                        w1=np.arange(4, dtype=np.float32), b1=0.1, platt1=[1.0, 0.0],
                        w2=np.arange(4, dtype=np.float32), b2=-0.2, platt2=[1.0, 0.0],
                        tau1=0.4, tau2=0.6)
    c2 = P.PersonCascade.from_json(json.loads(json.dumps(c.to_json())))
    assert c2.dim == 4 and c2.tau1 == 0.4 and c2.tau2 == 0.6
    assert np.allclose(c2.w1, c.w1)


def test_cascade_p2_never_exceeds_p1():
    """P(>=2) <= P(>=1) is an ordinal invariant — a broken fit must not invert it."""
    c = P.PersonCascade(model_id="m", dim=3,
                        w1=np.zeros(3, np.float32), b1=5.0, platt1=[1.0, 0.0],   # p1 ~ 1
                        w2=np.zeros(3, np.float32), b2=5.0, platt2=[1.0, 0.0],   # p2 ~ 1
                        tau1=0.5, tau2=0.5)
    p1, p2 = c.probs(np.zeros((4, 3), np.float32))
    assert np.all(p2 <= p1 + 1e-9)


# ── the shipped cascade is real and consistent with the report ───────────────
def test_shipped_cascade_loads_and_matches_backend():
    from imgtag.core.models import DEFAULT_BACKEND
    p = P.cascade_path(DEFAULT_BACKEND, ROOT / "src/imgtag/data/moderation")
    if not p.is_file():
        pytest.skip("cascade not fitted on this checkout")
    c = P.PersonCascade.from_json(json.loads(p.read_text()))
    assert c.model_id == DEFAULT_BACKEND
    assert c.dim == 512 and 0.0 < c.tau1 < 1.0 and 0.0 < c.tau2 < 1.0


# ── acceptance: the real detector on the four named cases (data-gated) ────────
ACCEPTANCE = {
    "000000456496.jpg": ("back-view person", "one-person", "no-one-face"),
    "000000308394.jpg": ("single person + face", "one-person", "one-face"),
    "000000171190.jpg": ("crowd", "multi-person", "multi-face"),
    "000000037777.jpg": ("empty landscape", "none", "none"),
}


@pytest.mark.parametrize("fname,expect", list(ACCEPTANCE.items()))
def test_acceptance_real_yunet(fname, expect):
    art = P.MODELS / P.ARTIFACT
    img = ROOT / "data/coco/val2017" / fname
    if not art.is_file() or not img.is_file():
        pytest.skip("YuNet artifact or COCO image absent")
    head = P.load_people_head({})
    assert head is not None
    from PIL import Image
    with Image.open(img) as im:
        n_faces, _ = head.faces(im.convert("RGB"))
    label, person_cat, face_cat = expect
    if person_cat == "none":
        assert n_faces == 0, f"{label}: expected no faces, got {n_faces}"
    elif person_cat == "multi-person":
        assert n_faces >= 2, f"{label}: expected a crowd, got {n_faces}"
    # single/back-view: the face count alone must not over-count a lone subject
    if face_cat == "one-face":
        assert n_faces == 1, f"{label}: expected one face, got {n_faces}"
    if face_cat == "no-one-face":
        assert n_faces == 0, f"{label}: back view must show no face, got {n_faces}"
