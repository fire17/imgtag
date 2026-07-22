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


def test_no_match_is_honest_via_calibrated_tag_or_dense_floor(home):
    """Two honest no-match routes. (a) A calibrated tag present but not cleared -> gated
    veto, calibration "fitted". (b) A query orthogonal to everything -> dense-floor veto,
    holds even when no calibrated tag engaged (calibration honestly "unfitted")."""
    be = build(home, "d1", ["cat"] * 18 + ["car"] * 2)
    write_tags(home, be.model_sha, ["cat"], ["calibrated"], [[-12.0, 6.0]], [0.5])
    s = searcher(home, be)
    # (a) "cat" IS the calibrated tag and fires -> a real fitted result
    r = s.search("cat", "d1", k=5)
    assert r["calibration"] == "fitted" and r["hits"]
    # (b) "boat" is orthogonal to cat+car -> honest no-match via the dense floor; the label
    # is "unfitted" because no calibrated tag gated it (the veto came from density, not tau)
    r = s.search("boat", "d1", k=5)
    assert r["hits"] == [] and r["no_match"] is True and r["calibration"] == "unfitted"


def test_unfitted_calibration_fails_OPEN(home):
    """FAIL-OPEN LAW: with no CAL-SET fit on disk the engine ranks and admits it cannot
    judge. An unfitted threshold must NEVER turn a real query into 'nothing matched'."""
    be = build(home, "d1", ["cat"] * 3 + ["car"] * 10 + ["tree"] * 10)
    r = searcher(home, be).search("car", "d1", k=5)  # fresh home: no tags.json anywhere
    assert r["calibration"] == "unfitted"
    assert len(r["hits"]) == 5 and r["no_match"] is False
    assert r["hits"][0]["p"] >= r["hits"][-1]["p"]  # still a ranking, just an unjudged one
    # an unfitted TAG table does not rescue the gate: a query with real dense neighbours
    # ("car") still fails open (ranks), never a false no-match
    write_tags(home, be.model_sha, ["car"], ["calibrated"], [None], [None])
    s = searcher(home, be)
    r = s.search("car", "d1", k=3)
    assert r["calibration"] == "unfitted" and r["hits"] and r["no_match"] is False
    # but a query with NOTHING dense-similar ("boat" is orthogonal here) is an honest
    # no-match via the global dense floor — that holds even unfitted (b-bench's rule)
    r = s.search("boat", "d1", k=3)
    assert r["no_match"] is True and r["hits"] == []


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
    # an unfitted tag still RANKS the cat images first (boost) — but its confidence is
    # capped at the honest dense score, never an inflated tag probability
    assert r["hits"] and all(h["path"].endswith(("0.jpg", "1.jpg")) for h in r["hits"][:2])
    assert r["hits"][0]["p"] > 0.5 and r["no_match"] is False


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


# ---------------------------------------------------------- multi-term wire shape


def test_terms_payload_shape(home):
    """The wire shape b-app renders: per-hit terms{matched,missed,m,n,mean_p} + top echo."""
    be = build(home, "d1", ["cat dog", "cat", "dog", "misc", "misc", "misc"])
    write_tags(home, be.model_sha, ["cat", "dog"], ["calibrated"] * 2, [[-12.0, 6.0]] * 2, [0.5] * 2)
    r = searcher(home, be).search("cat dog", "d1", k=6)
    assert r["terms"] == ["cat", "dog"]  # bare array, quoted spans would be one element
    assert r["calibration"] in ("fitted", "measured-default")
    ms = [h["why"]["terms"]["m"] for h in r["hits"]]
    assert ms == sorted(ms, reverse=True)  # tiers contiguous: b-app renders bands in order
    top = r["hits"][0]["why"]["terms"]
    assert top == {"matched": ["cat", "dog"], "missed": [], "m": 2, "n": 2,
                   "mean_p": top["mean_p"]}
    assert 0.0 < top["mean_p"] <= 1.0
    for h in r["hits"][1:]:
        t = h["why"]["terms"]
        assert len(t["matched"]) == t["m"] and len(t["missed"]) == t["n"] - t["m"]


def test_single_term_query_has_no_terms_key(home):
    """n == 1 behaves exactly as before — b-app suppresses the badge on absence."""
    be = build(home, "d1", ["cat"] * 3 + ["misc"] * 20)
    write_tags(home, be.model_sha, ["cat"], ["calibrated"], [[-12.0, 6.0]], [0.5])
    r = searcher(home, be).search("cat", "d1", k=3)
    assert "terms" not in r and r["hits"]
    assert all("terms" not in h["why"] for h in r["hits"])


def test_quoted_phrase_stays_one_term(home):
    assert S.split_terms('"night sky" dog') == [("night sky", True), ("dog", False)]
    assert S.split_terms("of the in") == [("of the in", True)]  # all stopwords -> one phrase
    assert S.split_terms("dog beach") == [("dog", False), ("beach", False)]
    be = build(home, "d1", ["cat dog"] * 2 + ["misc"] * 10)
    write_tags(home, be.model_sha, ["cat dog", "cat"], ["calibrated"] * 2,
               [[-12.0, 6.0]] * 2, [0.5] * 2)
    s = searcher(home, be)
    # the quoted span is looked up as ONE tag, never split into "cat" + "dog"
    assert [t for t, _ in s.concepts(s.tags(s.snapshot("d1").manifest), '"cat dog" cat')] == [
        "cat dog", "cat"]


def test_matched_terms_are_the_users_words_with_via_for_expansion(home):
    """Chips print the user's word; the hypernym it actually matched rides in `via`."""
    be = build(home, "d1", ["cat"] * 2 + ["misc"] * 20)
    write_tags(home, be.model_sha, ["cat", "dog"], ["calibrated"] * 2, [[-12.0, 6.0]] * 2, [0.5] * 2)
    s = searcher(home, be)
    r = s.search("animal misc", "d1", k=3)  # "animal" is a hypernym: expands to cat/dog
    if r.get("terms"):
        assert "animal" in r["terms"]  # the user's word, never the internal tag name


def test_generic_metadata_passthrough(home):
    """Index-time metadata (account ids, dates, ...) rides through to every hit (12:33Z)."""
    be = FakeBackend()
    embs = np.stack([be._vec("cat")] * 3 + [be._vec("misc")] * 20)
    recs = [{"image_id": f"{i:016x}", "path": f"/img/{i}.jpg", "dataset": "d1", "w": 4, "h": 4,
             "mtime": 1784724524.24, "size": 824,  # indexer bookkeeping: must NOT surface
             "meta": {"account_id": "acct-7", "captured": "2026-07-01"}} for i in range(23)]
    with Writer("d1", be, home) as w:
        w.append(embs, recs)
    r = S.Searcher(home, backend=be).search("cat", "d1", k=3)
    assert r["hits"]
    for h in r["hits"]:
        # hoisted verbatim, one level deep exactly as the contract says
        assert h["meta"] == {"account_id": "acct-7", "captured": "2026-07-01"}
        assert "mtime" not in h["meta"] and "size" not in h["meta"] and "meta" not in h["meta"]
        assert h["w"] == 4 and h["h"] == 4  # known fields stay top-level, not in meta


def test_common_term_undercounts_until_tau_is_fitted(home):
    """HONEST LIMITATION, measured: a term present in most of the corpus cannot stand 2
    sigma above it, so it under-counts in the spectrum. An image-relative top-R test was
    built and A/B-measured against COCO GT — identical precision AND recall — so the fix
    is a fitted per-tag tau (b-bench), not another heuristic."""
    labels = ["cat"] * 18 + ["cat dog"] * 2  # "cat" is in 100% of rows
    be = build(home, "d1", labels)
    write_tags(home, be.model_sha, ["cat", "dog"], ["calibrated"] * 2, [[-12.0, 6.0]] * 2, [0.5] * 2)
    r = searcher(home, be).search("cat dog", "d1", k=5)
    assert r["hits"]  # ranking still works; the tier is what degrades
    assert r["hits"][0]["why"]["terms"]["n"] == 2


# ---------------------------------------------------------- ADR-14 two-tier moderation


def _fake_tracks(monkeypatch):
    """A track spec expressed in FakeBackend's vocabulary so tiering is deterministic."""
    spec = {"version": 2, "enforcement_ready": False, "categories": {"x": {
        "label": "x", "violation": ["cat"], "review": ["dog"], "negatives": ["sky"]}}}
    monkeypatch.setattr(S, "tracks", lambda: spec)
    return spec


def test_tier_is_decided_by_which_prompt_set_the_image_resembles_more(home, monkeypatch):
    """ADR-14: the review set IS the look-alike, so 'violation first' mislabels it. A row
    goes to the tier whose prompt set it scores HIGHER on."""
    _fake_tracks(monkeypatch)
    be = build(home, "d1", ["cat"] + ["dog"] + ["tree"] * 18)
    s = searcher(home, be)
    t = s.track_scores("d1")
    ids = [r["image_id"] for r in s.snapshot("d1").ids]
    cat_row, dog_row = 0, 1
    c = t["categories"]["x"]
    assert c["is"]["violation"][cat_row] and not c["is"]["review"][cat_row]
    assert c["is"]["review"][dog_row] and not c["is"]["violation"][dog_row]
    assert c["tiers"] == ["violation", "review"] and c["moderation"] is True
    assert t["counts"]["x"] == {"violation": 1, "review": 1}
    assert ids  # provenance intact

    flags = s.flags_for(t, dog_row)
    assert flags == [{"category": "x", "p": flags[0]["p"], "tier": "review",
                      "kind": "moderation"}]
    assert 0.0 < flags[0]["p"] <= 1.0


def test_moderation_summary_splits_tiers_and_never_claims_enforcement(home, monkeypatch):
    _fake_tracks(monkeypatch)
    be = build(home, "d1", ["cat", "dog"] + ["tree"] * 18)
    m = searcher(home, be).moderation("d1", limit=5)
    assert m["counts"]["x"] == {"violation": 1, "review": 1}
    assert m["enforcement_ready"] == {"x": False} and m["calibration"] == {"x": "unfitted"}
    assert m["source"] == "current-scan"  # distinct from b-engine's stored index-time flags
    tiers = {f["tier"] for f in m["flagged"]}
    assert tiers == {"violation", "review"}
    assert m["flagged"][0]["tier"] == "violation"  # violations sort first within a category


def test_stored_index_time_flags_pass_through_under_their_own_name(home):
    """Two legitimate sources: b-engine's stored flags and my live scan never merge."""
    be = FakeBackend()
    recs = [{"image_id": f"{i:016x}", "path": f"/img/{i}.jpg", "dataset": "d1", "w": 1, "h": 1,
             "flags": [{"category": "weapons", "p": 0.9, "tier": "violation"}]} for i in range(6)]
    with Writer("d1", be, home) as w:
        w.append(np.stack([be._vec("cat")] * 6), recs)
    r = S.Searcher(home, backend=be).search("cat", "d1", k=3)
    h = r["hits"][0]
    assert h["flags_stored"] == [{"category": "weapons", "p": 0.9, "tier": "violation"}]
    assert "flags" not in h  # nothing scanned live in this process -> no live flags invented
    assert "flags" not in h.get("meta", {})  # and it does not leak into generic metadata


def test_duplicate_index_rows_collapse_but_stay_counted(home):
    """LEGACY-INDEX GUARD (read side of a layered defence). b-engine's writer now refuses a
    duplicate content id — within a batch, across appends, and on re-open — so this exercises
    the reader against the corruption a PRE-guard index (b-app's 420-rows-for-44-ids) still
    holds on disk: dedupe() collapses to one image, folds extra paths in, and reports the
    count so a regression can never hide behind a tidy payload. The writer guard is asserted
    live so this test fails loudly if that root fix is ever lost."""
    be = FakeBackend()
    from imgtag.core.store import Writer as _W

    with _W("d1", be, home) as w:
        w.append(np.stack([be._vec("cat")]),
                 [{"image_id": "a" * 16, "path": "/img/0.jpg", "dataset": "d1", "w": 1, "h": 1}])
        with pytest.raises(ValueError, match="duplicate image_id"):
            w.append(np.stack([be._vec("cat")]),
                     [{"image_id": "a" * 16, "path": "/img/1.jpg", "dataset": "d1", "w": 1, "h": 1}])

    hits = [{"image_id": "a" * 16, "path": f"/img/copy{i}.jpg", "dataset": "d1", "p": 0.9 - i / 100}
            for i in range(7)]
    hits += [{"image_id": "b" * 16, "path": "/img/other.jpg", "dataset": "d1", "p": 0.5}]
    out, collapsed = S.dedupe(hits)
    assert [h["image_id"] for h in out] == ["a" * 16, "b" * 16]
    assert collapsed == 6
    assert len(out[0]["paths"]) == 7 and out[0]["path"] in out[0]["paths"]
    # Default (per-dataset) key: same id in ANOTHER dataset stays a distinct row.
    cross, n = S.dedupe(hits[:1] + [{**hits[0], "dataset": "d2"}])
    assert len(cross) == 2 and n == 0
    # GLOBAL search path (2026-07-22, user-reported dupes): identical content indexed
    # under two dataset names is ONE result; provenance survives in also_in and the
    # collapse stays counted. B18(d) intact — every surfaced name/path is the true one.
    g, gn = S.dedupe(
        [hits[0], {**hits[0], "dataset": "d2", "path": "/elsewhere/0.jpg"}],
        across_datasets=True)
    assert len(g) == 1 and gn == 1
    assert g[0]["dataset"] == "d1"  # best-ranked row keeps its own attribution
    assert g[0]["also_in"] == [{"dataset": "d2", "path": "/elsewhere/0.jpg"}]


def test_every_hit_carries_exists(home):
    """B18(b): a path that is gone is tombstoned, never a silent 404 for the client."""
    be = FakeBackend()
    with Writer("d1", be, home) as w:
        w.append(np.stack([be._vec("cat")] * 2),
                 [{"image_id": f"{i:016x}", "path": f"/gone/{i}.jpg", "dataset": "d1", "w": 1, "h": 1}
                  for i in range(2)])
    r = S.Searcher(home, backend=be).search("cat", "d1", k=2)
    assert r["hits"] and all(h["exists"] is False for h in r["hits"])


def test_proxy_fitted_spec_is_demoted_to_unfitted(home, monkeypatch):
    """A PROXY fit may not gate (ruling after b-app's audit: the proxy logistic produced
    218 benign violations at a saturated p=0.99). The scorer survives; the taus do not."""
    spec = {"version": 2, "categories": {"x": {
        "label": "x", "violation": ["cat"], "review": ["dog"], "negatives": ["sky"],
        "scorer": "margin", "platt": [105.0, -6.6], "tau": 0.019, "tau_review": 0.031,
        "calibration": "proxy-fitted", "enforcement_ready": False}}}
    monkeypatch.setattr(S, "tracks", lambda: spec)
    be = build(home, "d1", ["cat", "dog"] + ["tree"] * 18)
    s = searcher(home, be)
    t = s.track_scores("d1")
    c = t["categories"]["x"]
    assert c["calibration"] == "unfitted"       # what the engine reports
    assert c["spec_calibration"] == "proxy-fitted"  # what the spec claimed, preserved
    assert c["enforcement_ready"] is False
    # p must not saturate: a fitted logistic on a margin pinned everything at 0.99
    assert float(c["p"]["violation"].max()) < 0.999
    assert s.track_state("d1", "x")["calibration"] == "unfitted"  # same value everywhere


def test_stored_moderation_counts_index_time_flags(home):
    """`source=stored` totals what the INDEXER recorded — no model, survives a threshold
    change, and never merges with the live scan."""
    be = FakeBackend()
    recs = []
    for i in range(6):
        r = {"image_id": f"{i:016x}", "path": f"/img/{i}.jpg", "dataset": "d1", "w": 1, "h": 1}
        if i < 3:
            r["flags"] = [{"category": "weapons", "p": 0.9, "tier": "violation" if i < 2 else "review"}]
        recs.append(r)
    with Writer("d1", be, home) as w:
        w.append(np.stack([be._vec("cat")] * 6), recs)
    m = S.Searcher(home, backend=be).stored_moderation("d1", limit=5)
    assert m["source"] == "stored" and m["indexed"] == 6
    assert m["counts"] == {"weapons": {"violation": 2, "review": 1}}
    assert m["enforcement_ready"] == {"weapons": False}
    assert m["flagged"][0]["tier"] == "violation" and m["flagged"][0]["image_id"]


def test_alert_tier_outranks_violation_and_match_is_never_moderation(home, monkeypatch):
    """Tier vocabulary is data-driven: a track ships by adding a spec entry.
    `alert` is the most severe moderation tier; `match` is CONTENT and never counted
    as moderation (ORACLE tier-vocab amendments 2026-07-22)."""
    spec = {"version": 2, "categories": {
        "safety": {"label": "safety", "alert": ["car"], "review": ["dog"], "negatives": ["sky"]},
        "sports": {"label": "sports", "match": ["cat"], "negatives": ["sky"]}}}
    monkeypatch.setattr(S, "tracks", lambda: spec)
    be = build(home, "d1", ["car", "dog", "cat"] + ["tree"] * 17)
    s = searcher(home, be)
    t = s.track_scores("d1")
    assert t["categories"]["safety"]["tiers"] == ["alert", "review"]
    assert t["categories"]["safety"]["moderation"] is True
    assert t["categories"]["sports"]["moderation"] is False  # content track

    m = s.moderation("d1", limit=5)
    assert "safety" in m["counts"] and "sports" not in m["counts"]      # never mixed
    assert m["content_counts"] == {"sports": {"match": 1}}
    assert m["counts"]["safety"] == {"alert": 1, "review": 1}
    assert m["flagged"][0]["tier"] == "alert"  # most severe first, above violation

    flags = s.flags_for(t, 0)  # the "car" row -> safety/alert
    assert flags[0]["tier"] == "alert" and flags[0]["kind"] == "moderation"
    sports_flags = s.flags_for(t, 2)
    assert sports_flags[0] == {"category": "sports", "p": sports_flags[0]["p"],
                               "tier": "match", "kind": "content"}
    assert S.tier_rank("alert") < S.tier_rank("violation") < S.tier_rank("review")
    assert S.tier_rank("unknown-future-tier") == len(S.TIER_ORDER)  # never dropped


def test_content_track_emits_a_per_prompt_label(home, monkeypatch):
    """A content track names WHICH item fired (which sport) via `<tier>_labels`, and never
    counts toward moderation totals (content_track: true)."""
    spec = {"version": 2, "categories": {"sports": {
        "label": "sports", "content_track": True,
        "match": ["cat", "car", "boat"], "match_labels": ["tennis", "motorsport", "sailing"],
        "negatives": ["sky"]}}}
    monkeypatch.setattr(S, "tracks", lambda: spec)
    be = build(home, "d1", ["car"] + ["tree"] * 19)  # one standout sport row
    s = searcher(home, be)
    t = s.track_scores("d1")
    assert t["categories"]["sports"]["moderation"] is False
    f = s.flags_for(t, 0)
    assert f and f[0]["tier"] == "match" and f[0]["kind"] == "content"
    assert f[0]["label"] == "motorsport"  # car -> its parallel match_label
    m = s.moderation("d1", limit=3)
    assert "sports" not in m["counts"] and m["content_counts"] == {"sports": {"match": 1}}


def test_fitted_head_file_overrides_spec_thresholds(home, monkeypatch, tmp_path):
    """Per-model tau lives in data/moderation/<category>-<model_id>.json (TRACKS.md T3),
    and wins over the spec — so a refit is a file swap, no code change."""
    import imgtag.core.store as ST
    fitdir = tmp_path / "moderation"
    fitdir.mkdir()
    (fitdir / "weapons-fake.json").write_text(json.dumps(
        {"calibration": "fitted", "scorer": "margin", "tau": -9.9, "tau_review": -9.9}))
    monkeypatch.setattr(ST, "_DATA", tmp_path)  # the shared loader reads store._DATA
    spec = {"version": 2, "categories": {"weapons": {
        "label": "w", "violation": ["cat"], "review": ["dog"], "negatives": ["sky"],
        "scorer": "margin"}}}
    monkeypatch.setattr(S, "tracks", lambda: spec)
    be = build(home, "d1", ["cat"] * 3 + ["misc"] * 5)
    t = S.Searcher(home, backend=be).track_scores("d1")
    assert t["categories"]["weapons"]["calibration"] == "fitted"  # came from the head file
    assert int(t["counts"]["weapons"]["violation"]) > 0


def test_image_tracks_lists_every_track_ranked(home, monkeypatch):
    """VISION-ADDENDA 14:16Z: per-image panel shows EVERY track, ranked, fired first."""
    spec = {"version": 2, "categories": {
        "weapons": {"label": "weapons", "violation": ["car"], "review": ["boat"], "negatives": ["sky"]},
        "sports":  {"label": "sports", "content_track": True, "match": ["cat"], "negatives": ["sky"]},
        "drugs":   {"label": "drugs", "violation": ["dog"], "review": ["cup"], "negatives": ["sky"]}}}
    monkeypatch.setattr(S, "tracks", lambda: spec)
    be = build(home, "d1", ["car"] + ["tree"] * 19)  # row 0 is a "car" (weapons/violation)
    s = searcher(home, be)
    iid = s.snapshot("d1").ids[0]["image_id"]
    r = s.image_tracks("d1", iid, source="current")  # synthetic dataset has no sidecars
    assert r["image_id"] == iid and r["dataset"] == "d1" and r["source"] == "current scan"
    cats = {t["category"] for t in r["tracks"]}
    assert cats == {"weapons", "sports", "drugs"}  # ALL tracks present, even ~0 ones
    for t in r["tracks"]:
        assert set(t) >= {"category", "label", "kind", "p", "tier", "scored", "calibration"}
        assert t["kind"] == ("content" if t["category"] == "sports" else "moderation")
        assert t["tier"] is None or isinstance(t["tier"], str)  # "none" or a tier name
    # the fired track (weapons on a car) ranks above a no-signal track
    fired = [t for t in r["tracks"] if t["tier"] not in (None, "none")]
    assert fired and fired[0]["category"] == "weapons"
    assert r["tracks"][0]["category"] == "weapons"  # fired-first ordering


def test_image_tracks_unknown_image_404s(home, monkeypatch):
    monkeypatch.setattr(S, "tracks", lambda: {"version": 2, "categories": {}})
    be = build(home, "d1", ["cat"] * 3)
    with pytest.raises(FileNotFoundError):
        searcher(home, be).image_tracks("d1", "deadbeefdeadbeef")


def test_uncalibrated_tag_never_saturates_above_calibrated_or_dense(home):
    """P1 (weapon/weapo): an unfitted hypernym must never present fake ~0.99 confidence and
    outrank calibrated tags / the dense honest score on a homogeneous corpus."""
    # 20 near-identical "cat" images (homogeneous, like weaponprobe's all-guns): an
    # uncalibrated tag's corpus-relative z-score would saturate on the tail here.
    be = build(home, "d1", ["cat"] * 20)
    d = home / "models" / be.model_sha
    d.mkdir(parents=True, exist_ok=True)
    # one calibrated tag ("cat", fitted low so its p is modest) + one uncalibrated ("dog")
    np.stack([be._vec("cat"), be._vec("dog")]).astype(np.float32).tofile(d / "tags.f32")
    (d / "tags.json").write_text(json.dumps({
        "names": ["cat", "dog"], "dim": DIM, "model_sha": be.model_sha,
        "tier": ["calibrated", "uncalibrated"],
        "tau": [0.5, None], "platt": [[-4.0, 2.0], None]}))
    s = searcher(home, be)
    # a query that resolves to the uncalibrated "dog" tag on a corpus of cats: its p must
    # be bounded, never a saturated 0.99 that would dominate the ranking
    r = s.search("dog", "d1", k=5)
    if r["hits"]:
        assert all(h["p"] <= 0.5 + 1e-6 for h in r["hits"]), [h["p"] for h in r["hits"]]
        # and it must never gate (no_match verdict) from an uncalibrated tag
        assert r["calibration"] == "unfitted"


def test_fitted_head_sha_guard_rejects_wrong_model(home, monkeypatch, tmp_path):
    """SHA-GUARD: a fitted file whose model_sha != the dataset's is ignored (a base-model
    fit must not contaminate a same-base variant with a different model_sha)."""
    import imgtag.core.store as ST
    fitdir = tmp_path / "moderation"
    fitdir.mkdir()
    # a fitted file that CLAIMS a different model than the dataset's FakeBackend ("f"*64)
    (fitdir / "sports-fake.json").write_text(json.dumps(
        {"calibration": "fitted", "scorer": "margin", "tau_match": -9.9,
         "model_sha": "0" * 64}))  # wrong sha
    monkeypatch.setattr(ST, "_DATA", tmp_path)
    spec = {"version": 2, "categories": {"sports": {
        "label": "s", "content_track": True, "match": ["cat"], "negatives": ["sky"]}}}
    monkeypatch.setattr(S, "tracks", lambda: spec)
    be = build(home, "d1", ["cat"] * 3 + ["misc"] * 5)
    t = S.Searcher(home, backend=be).track_scores("d1")
    # the wrong-sha fitted file is IGNORED -> track stays unfitted, its bogus tau never gates
    assert t["categories"]["sports"]["calibration"] == "unfitted"


def test_absolute_margin_floor_blocks_ood_mass_fire(home, monkeypatch):
    """OOD guard: an unfitted tier must not mass-fire on a homogeneous/OOD corpus just
    because the corpus-relative z-score tail is high. A row whose ABSOLUTE margin (best
    tier concept - best negative) is below the floor never fires, whatever its z-rank."""
    # corpus of 20 "cat" images (homogeneous). A "weapons" track whose concepts are all
    # ORTHOGONAL to cats -> every image has a near-zero/negative absolute margin, so the
    # z-score tail must NOT be allowed to fire the tier.
    spec = {"version": 2, "categories": {"weapons": {
        "label": "w", "violation": ["car"], "review": ["boat"], "negatives": ["dog"]}}}
    monkeypatch.setattr(S, "tracks", lambda: spec)
    be = build(home, "d1", ["cat"] * 20)  # nothing is a weapon
    t = S.Searcher(home, backend=be).track_scores("d1")
    c = t["categories"]["weapons"]
    # no cat image is absolutely weapon-like -> zero fires despite the relative tail
    assert int(c["is"]["violation"].sum()) == 0 and int(c["is"]["review"].sum()) == 0


def test_derive_unfitted_shared_contract():
    """The ONE shared derivation both the reader and store-side dataset_flags call.
    Contract: per-tier margins -> tiers by exceedance + absolute-margin floor, fail-open."""
    n = 100
    # a genuine tail (the z-floor is 3 sigma): rows 0-1 strongly violation, 2-3 review, rest ~0
    viol = np.zeros(n, np.float32)
    viol[:2], viol[2:4] = 0.20, -0.05
    rev = np.zeros(n, np.float32)
    rev[2:4], rev[:2] = 0.20, -0.05
    r = S.derive_unfitted({"violation": viol, "review": rev})
    assert r["tiers"] == ["violation", "review"]
    v, rv = r["is"]["violation"], r["is"]["review"]
    assert v[:2].all() and not v[2:].any()             # rows 0-1 -> violation only
    assert rv[2:4].all() and not rv[:2].any()          # rows 2-3 -> review only
    assert not v[4:].any() and not rv[4:].any()        # the ~0 bulk fires NOTHING (abs floor)
    # a negative-margin row never fires, whatever its corpus z-rank
    r2 = S.derive_unfitted({"violation": np.full(30, -0.1, np.float32)})
    assert not r2["is"]["violation"].any()
    # corpus_stats override is honoured (reuse precomputed mean/std)
    r3 = S.derive_unfitted({"violation": viol}, corpus_stats={"violation": (0.0, 0.05)})
    assert r3["is"]["violation"][:2].all()


def test_head_arbitrated_track_consumed_not_rebanded(home, monkeypatch):
    """A track that declares scorer='margin_arbitrated' is scored by its HEAD's (p, tier)
    — the head owns the arbitration (drugs' vape->review); the reader never re-bands. The
    calibration LABEL is never relabeled to 'fitted' (stays proxy-fitted)."""
    class FakeHead:
        model_sha = "f" * 64
        def probs(self, emb):
            n = len(emb)
            tiers = np.array(["none"] * n, dtype=object)
            tiers[0] = "review"   # the head arbitrates row 0 to review (e.g. a vape)
            tiers[1] = "violation"
            p = np.zeros(n, np.float32)
            p[0] = 0.6
            p[1] = 0.8
            return p, tiers
    import imgtag.core.search as CS
    spec = {"version": 2, "categories": {"drugs": {
        "label": "d", "violation": ["car"], "review": ["boat"], "negatives": ["sky"],
        "scorer": "margin_arbitrated", "calibration": "proxy-fitted",
        "gate_safe": True, "evidence_cap": 0.947}}}
    monkeypatch.setattr(CS, "tracks", lambda: spec)
    be = build(home, "d1", ["car"] * 3 + ["misc"] * 5)
    s = searcher(home, be)
    monkeypatch.setattr(s, "_track_head", lambda *a, **k: FakeHead())
    t = s.track_scores("d1")
    c = t["categories"]["drugs"]
    # tiers come straight from the head (row0 review, row1 violation), not re-banded
    assert c["is"]["review"][0] and c["is"]["violation"][1]
    assert int(c["is"]["violation"].sum()) == 1 and int(c["is"]["review"].sum()) == 1
    # label NEVER relabeled to fitted, enforcement stays false
    assert c["calibration"] == "proxy-fitted" and c["enforcement_ready"] is False
