"""b-daemon lane: core/search.py — calibration, fusion, isolation, provenance (B18)."""

from __future__ import annotations

import json
import threading

import numpy as np
import pytest

from imgtag.core import search as S
from imgtag.core.store import Writer

DIM = 8


class FakeBackend:
    """Deterministic stand-in for a ModelBackend: one axis per vocabulary word."""

    model_id, dim = "fake-fp32", DIM
    model_sha = "f" * 64
    WORDS = ["cat", "dog", "car", "tree", "boat", "cup", "sky", "misc"]

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(DIM, np.float32)
        for i, w in enumerate(self.WORDS):
            if w in text.lower():
                v[i] += 1.0
        if not v.any():
            v[-1] = 1.0
        return v / np.linalg.norm(v)

    def embed_texts(self, texts):
        return np.stack([self._vec(t) for t in texts])


@pytest.fixture()
def home(tmp_path):
    return tmp_path


def build(home, dataset, labels):
    be = FakeBackend()
    embs = np.stack([be._vec(x) for x in labels])
    recs = [
        {"image_id": f"{i:016x}", "path": f"/img/{dataset}/{i}.jpg", "dataset": dataset, "w": 4, "h": 4}
        for i in range(len(labels))
    ]
    with Writer(dataset, be, home) as w:
        w.append(embs, recs)
    return be


def searcher(home, be):
    return S.Searcher(home, backend=be)


# ---------------------------------------------------------------- text path


def test_free_text_path_ranks_and_carries_provenance(home):
    be = build(home, "d1", ["cat"] * 3 + ["car"] * 10 + ["tree"] * 10)
    r = searcher(home, be).search("cat", "d1", k=5)
    assert r["hits"], r
    for h in r["hits"]:  # B18(a): provenance is never null
        assert h["image_id"] and h["path"] and h["dataset"] == "d1" and h["dataset_slug"] == "d1"
        assert h["why"]["path"] == "text"
    assert r["coverage"] == {"indexed": 23, "total": 23}
    assert r["no_match"] is False


def test_provenance_complete_over_20_queries(home):
    labels = ["misc"] * 60 + ["cat", "dog", "car", "tree", "boat", "cup", "sky"] * 2
    be = build(home, "d1", labels)
    s = searcher(home, be)
    words = FakeBackend.WORDS[:7]
    queries = words + [f"a {w}" for w in words] + [f"the {w}" for w in words[:6]]
    assert len(queries) == 20
    seen = 0
    for q in queries:
        r = s.search(q, "d1", k=10)
        assert r["hits"], f"{q!r} should match its 2 rows"
        for h in r["hits"]:
            assert h["image_id"] and h["path"] and h["dataset"] and h["dataset_slug"]
            assert set(h) >= {"image_id", "path", "dataset", "score", "p", "why"}
            seen += 1
    assert seen >= 20


def test_no_match_is_honest(home):
    """A query orthogonal to every row must return zero hits AND say so."""
    be = build(home, "d1", ["cat"] * 20)
    r = searcher(home, be).search("boat", "d1", k=5)
    assert r["hits"] == [] and r["no_match"] is True


def test_determinism_ties_broken_by_image_id(home):
    be = build(home, "d1", ["cat"] * 10 + ["misc"] * 10)
    s = searcher(home, be)
    runs = [json.dumps(s.search("cat", "d1", k=10)["hits"]) for _ in range(5)]
    assert len(set(runs)) == 1  # B18(e)
    ids = [h["image_id"] for h in json.loads(runs[0])]
    assert ids == sorted(ids)  # equal p -> ordered by id


def test_empty_dataset_and_unknown_dataset(home):
    be = FakeBackend()
    with Writer("empty", be, home):
        pass
    s = searcher(home, be)
    r = s.search("cat", "empty", k=5)
    assert r["hits"] == [] and r["no_match"] is True and r["coverage"]["indexed"] == 0
    from imgtag.core.store import UnknownDatasetError

    with pytest.raises(UnknownDatasetError):
        s.search("cat", "ghost")


# ---------------------------------------------------------------- isolation


def test_snapshot_isolation_during_concurrent_append(home):
    """A search running while a writer appends sees a consistent point-in-time view,
    and the newly committed rows become visible to the NEXT query (B11)."""
    be = build(home, "live", ["cat"] * 10)
    s = searcher(home, be)
    first = s.search("cat", "live", k=50)
    assert first["coverage"]["indexed"] == 10

    stop = threading.Event()
    errors = []

    def hammer():
        while not stop.is_set():
            try:
                r = s.search("cat", "live", k=50)
                n = r["coverage"]["indexed"]
                assert len(r["hits"]) <= n
                assert all(h["path"] for h in r["hits"])  # no torn/partial records
            except Exception as e:  # pragma: no cover - failure detail
                errors.append(e)
                return

    t = threading.Thread(target=hammer)
    t.start()
    try:
        with Writer("live", be, home) as w:
            for i in range(10, 60, 10):
                embs = np.stack([be._vec("cat") for _ in range(10)])
                recs = [
                    {"image_id": f"{j:016x}", "path": f"/img/live/{j}.jpg", "dataset": "live", "w": 1, "h": 1}
                    for j in range(i, i + 10)
                ]
                w.append(embs, recs)
    finally:
        stop.set()
        t.join(timeout=10)
    assert not errors, errors
    assert s.search("cat", "live", k=100)["coverage"]["indexed"] == 60  # visible after commit


# ---------------------------------------------------------------- tag path


def write_tags(home, model_sha, names, tiers, platt, taus, extra=None):
    """Platt pairs follow tags.py's convention: p = sigmoid(-(A*s + B)), so a tag that
    should fire on a high cosine has a NEGATIVE A (e.g. [-12, 6] -> p(1.0)=0.997)."""
    d = home / "models" / model_sha
    d.mkdir(parents=True, exist_ok=True)
    be = FakeBackend()
    np.stack([be._vec(n) for n in names]).astype(np.float32).tofile(d / "tags.f32")
    (d / "tags.json").write_text(
        json.dumps({"names": names, "dim": DIM, "model_sha": model_sha, "tier": tiers,
                    "tau": taus, "platt": platt, "calib_sha": "abc", **(extra or {})})
    )


def test_tag_path_wins_and_explains(home):
    be = build(home, "d1", ["cat"] * 5 + ["car"] * 15)
    write_tags(home, be.model_sha, ["cat", "car"], ["calibrated"] * 2,
               [[-12.0, 6.0]] * 2, [0.5, 0.5])
    r = searcher(home, be).search("cat", "d1", k=5)
    assert r["hits"] and r["hits"][0]["why"]["path"] == "tag"
    assert r["hits"][0]["why"]["tag"] == "cat" and r["hits"][0]["why"]["tier"] == "calibrated"
    assert 0.0 <= r["hits"][0]["p"] <= 1.0  # probability space only


def test_uncalibrated_tag_never_gates(home):
    """An uncalibrated tag may boost/explain but may not create an honest no-match verdict."""
    be = build(home, "d1", ["cat"] * 20)
    write_tags(home, be.model_sha, ["boat"], ["uncalibrated"], [[-12.0, 6.0]], [0.5])
    r = searcher(home, be).search("boat", "d1", k=5)
    assert r["no_match"] is True and r["hits"] == []


def test_calibration_mismatch_refuses_loudly(home):
    be = build(home, "d1", ["cat"] * 5)
    write_tags(home, be.model_sha, ["cat"], ["calibrated"], [[-12.0, 6.0]], [0.5])
    man_p = home / "datasets" / "d1" / "manifest.json"
    man = json.loads(man_p.read_bytes())
    man["calib_sha"] = "a-different-calibration"
    man_p.write_text(json.dumps(man))
    with pytest.raises(S.CalibrationMismatchError):
        searcher(home, be).search("cat", "d1", k=5)


def test_model_sha_mismatch_refuses_loudly(home):
    from imgtag.core.store import ModelMismatchError

    build(home, "d1", ["cat"] * 5)

    class Other(FakeBackend):
        model_sha = "0" * 64

    with pytest.raises(ModelMismatchError):
        searcher(home, Other()).search("cat", "d1", k=5)


# ---------------------------------------------------------------- query rules


def test_compound_query_never_inherits_a_component_tag(home):
    assert S.is_compound("my dog wearing a santa hat")
    assert not S.is_compound("a dog")
    be = build(home, "d1", ["dog"] * 10)
    write_tags(home, be.model_sha, ["dog"], ["calibrated"], [[-12.0, 6.0]], [0.5],
               extra={"theta_syn": 0.5})
    s = searcher(home, be)
    simple = s.search("dog", "d1", k=3)
    assert simple["hits"] and simple["hits"][0]["why"]["path"] == "tag"
    # "santa hat" is not in the vocabulary: the compound must not borrow `dog`'s calibration
    compound = s.search("dog wearing santa hat", "d1", k=3)
    assert all(h["why"]["path"] == "text" for h in compound["hits"])


def test_hypernym_expansion_from_the_static_table():
    kids = S.hierarchy()
    assert kids, "scripts/build_hierarchy.py must have produced src/imgtag/data/hierarchy.json"
    assert "dog" in S.expand("animal")
    assert "car" in S.expand("vehicle")
    assert S.expand("nonexistent-term-xyz") == ["nonexistent-term-xyz"]


# ---------------------------------------------------------- ALL-SOME-ANY spectrum


def test_all_some_any_spectrum(home):
    """VISION-ADDENDA 2026-07-22 ~12:12Z (verbatim): space-separated tags rank ALL first,
    then descending by how many of the m tags were found, then ANY."""
    labels = ["cat dog car", "cat dog", "cat", "dog", "car", "tree", "misc", "misc"]
    be = build(home, "d1", labels)
    write_tags(home, be.model_sha, ["cat", "dog", "car"], ["calibrated"] * 3,
               [[-12.0, 6.0]] * 3, [0.5] * 3)
    r = searcher(home, be).search("cat dog car", "d1", k=8)
    got = [(h["why"]["tags_matched"], h["why"]["spectrum"]) for h in r["hits"]]
    assert got[0] == (3, "all"), got                       # ALL first
    assert [m for m, _ in got] == sorted([m for m, _ in got], reverse=True)  # then descending
    assert got[-1][0] == 1 and got[-1][1] == "any"         # ANY last
    assert r["hits"][0]["why"]["tags_total"] == 3
    # every row that owns at least one of the three tags is present; "tree"/"misc" are not
    assert len(r["hits"]) == 5


def test_spectrum_only_for_real_tag_lists(home):
    """A natural-language query keeps ADR-3 §3: it is NOT split into tags."""
    be = build(home, "d1", ["dog"] * 5 + ["misc"] * 15)
    write_tags(home, be.model_sha, ["dog", "cat"], ["calibrated"] * 2, [[-12.0, 6.0]] * 2, [0.5] * 2)
    s = searcher(home, be)
    assert s.concepts(s.tags(s.snapshot("d1").manifest), "dog cat") != []
    assert s.concepts(s.tags(s.snapshot("d1").manifest), "dog wearing santa hat") == []
    r = s.search("dog wearing santa hat", "d1", k=3)
    assert all("tags_matched" not in h["why"] for h in r["hits"])


def test_unfitted_tags_fall_back_to_dataset_layer_and_never_gate(home):
    """b-bench ships tau/platt as null until CAL-SET runs: the tag must still be usable
    for ranking (dataset-layer z-score) and must never gate or claim calibration."""
    be = build(home, "d1", ["cat"] * 2 + ["misc"] * 30)
    d = home / "models" / be.model_sha
    d.mkdir(parents=True, exist_ok=True)
    np.stack([be._vec("cat")]).astype(np.float32).tofile(d / "tags.f32")
    (d / "tags.json").write_text(json.dumps({
        "names": ["cat"], "dim": DIM, "model_sha": be.model_sha,
        "tier": ["calibrated"], "tau": [None], "platt": [None]}))  # tier says yes, fit says no
    s = searcher(home, be)
    table = s.tags(s.snapshot("d1").manifest)
    assert table.calibrated(0) is False  # no fit -> not gating-eligible, whatever the tier
    r = s.search("cat", "d1", k=5)
    assert r["hits"] and r["hits"][0]["why"]["path"] == "tag"  # still ranks
    assert r["hits"][0]["p"] > 0.5


def test_all_outranks_a_higher_probability_any(home):
    """The verbatim ordering claim: tag COUNT beats probability. A row with 2/2 tags sits
    above a row with 1/2 even when the 1/2 row scores higher."""
    labels = ["cat dog"] + ["cat"] * 4 + ["misc"] * 20
    be = build(home, "d1", labels)
    write_tags(home, be.model_sha, ["cat", "dog"], ["calibrated"] * 2, [[-12.0, 6.0]] * 2, [0.5] * 2)
    r = searcher(home, be).search("cat dog", "d1", k=6)
    top, rest = r["hits"][0], r["hits"][1:]
    assert top["why"]["tags_matched"] == 2 and top["why"]["spectrum"] == "all"
    assert rest and max(h["p"] for h in rest) > top["p"]  # a lower-ranked row scores HIGHER
    assert all(h["why"]["tags_matched"] == 1 for h in rest)
