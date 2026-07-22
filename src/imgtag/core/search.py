"""Query path: f32 scan + calibration + probability fusion — ADR-3 contract.

OWNER: b-daemon. Consumes store.open_snapshot() + models.ModelBackend; never writes them.

Two paths, fused in PROBABILITY SPACE ONLY (ADR-3 calibration contract §2):

* **tag path** — precomputed tag table (b-bench writes ``~/.imgtag/models/<model_sha>/
  {tags.f32,tags.json}``, we only read): per-tag Platt fit -> ``p_tag``, effective
  threshold ``max(tau_tag, mean + k*std)`` from the manifest's dataset-layer ``tag_stats``.
  Only *calibrated* tags may gate or produce an honest no-match; uncalibrated tags may
  boost rank and explain a hit, never veto.
* **free-text path** — per-query corpus z-score -> global logistic -> ``p_text``.
  Works standalone: with no tag table on disk every query takes this path.

``p = max(p_tag, p_text)`` and the winning path is the ``why`` payload. Zero results above
tau is an honest ``no_match``, never an empty list pretending to be a miss.
"""

from __future__ import annotations

import json
import re
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from .store import (
    UnknownDatasetError,
    dataset_dir,
    imgtag_home,
    list_datasets,
    open_snapshot,
)

# -- calibration constants (ADR-3). PROVISIONAL until b-bench's fit lands; every one of
# them is overridable per-model by tags.json so the fitted values win automatically.
TEXT_A, TEXT_B = 1.6, -4.0  # p_text = sigmoid(A*x + B); x is TEXT_FEATURE
TEXT_FEATURE = "z"  # "z" = per-query corpus z-score (ADR-3 as written) | "cos" = raw cosine.
# MEASURED 2026-07-22 on unsplash-demo (N=2000, pecore-s16-384-fp32), 15 real vs 15 nonsense
# queries: best single threshold separates at 77% on "cos" but only 60% on "z" (chance = 50%)
# — nonsense max-z (med 4.16) actually EXCEEDS real max-z (med 3.81), so no (A,B) on the z
# feature can produce an honest no-match. Escalated; the default stays ADR-3's "z" until the
# conductor rules and b-bench fits the winner on CAL-SET. Both are shipped, tags.json picks.
TAU = 0.5  # probability floor for "this is a match"
THETA_SYN = 0.90  # near-tag rule: inherit a tag's calibration only at cos >= theta_syn
K_STD = 3.0  # dataset-layer effective tau = max(tau_tag, mean + K_STD*std)
# Counting a tag as PRESENT for the ALL-SOME-ANY spectrum is ranking, not gating, and gets
# its own (softer) floor. MEASURED on unsplash-demo (N=2000): at 3 sigma NO image clears two
# tags at once — a single CLIP embedding splits its mass across concepts — so beach+sunset
# co-occurs 0 times at 3.0, 1 at 2.5, 16 at 2.0. A gate floor would make ALL unreachable and
# the whole spectrum collapse to ANY. Applies ONLY to unfitted tags; a CAL-SET-fitted tau
# replaces it. Uncalibrated tags still may never gate or veto (ADR-3 tiers).
SPECTRUM_K = 2.0
MAX_TAGS = 32  # candidate tags per query (bounds the [N,C] tag matmul)
NGRAM = 3  # longest multi-word tag a space-separated tag list may name ("night sky")
DATA = Path(__file__).resolve().parent.parent / "data"


class CalibrationMismatchError(RuntimeError):
    """Manifest calib_sha/calib_model_sha != the tag table on disk (CLI exit 5)."""


def _order(h: dict) -> tuple:
    """Result order: more query tags matched first (ALL > SOME > ANY), then probability,
    then image id — the id tiebreak is what makes a result list byte-identical (B18e)."""
    return (-int((h.get("why") or {}).get("tags_matched", 0)), -h["p"], h["image_id"])


def _sigmoid(x) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float32)))


# ---------------------------------------------------------------- query text

_STOP = {
    "a", "an", "the", "my", "your", "his", "her", "its", "their", "of", "in", "on", "at",
    "to", "for", "with", "and", "or", "is", "are", "was", "were", "be", "some", "this",
    "that", "these", "those", "there", "photo", "picture", "image",
}


def content_tokens(q: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", q.lower()) if t not in _STOP]


def is_compound(q: str) -> bool:
    """ADR-3 §3: a compound query NEVER inherits a component tag's threshold
    ("my dog wearing a santa hat" must not borrow `dog`'s calibration)."""
    return len(content_tokens(q)) > 2


# ---------------------------------------------------------------- hypernyms


@lru_cache(maxsize=1)
def hierarchy() -> dict[str, list[str]]:
    """Static hypernym table built offline by scripts/build_hierarchy.py (ADR-3/ADR-7)."""
    try:
        return json.loads((DATA / "hierarchy.json").read_bytes())["children"]
    except (OSError, ValueError, KeyError):
        return {}


def expand(term: str, depth: int = 3) -> list[str]:
    """term + its hyponym closure, cycle-safe and depth-capped. 'animal' -> dog, cat, …"""
    kids = hierarchy()
    seen, frontier = [term], [term]
    for _ in range(depth):
        nxt = [c for t in frontier for c in kids.get(t, []) if c not in seen]
        if not nxt:
            break
        seen.extend(dict.fromkeys(nxt))
        frontier = nxt
    return seen


# ---------------------------------------------------------------- tag table


class TagTable:
    """Read-only view of b-bench's tag table. Schema (ADR-3):
    ``{names[], dim, model_sha, prompt_ensemble_sha, tier[], tau[], platt[], provenance{}}``
    beside a ``tags.f32`` [T, dim] L2-normalized matrix."""

    def __init__(self, d: Path):
        meta = json.loads((d / "tags.json").read_bytes())
        self.dir = d
        self.names: list[str] = [str(n).lower() for n in meta["names"]]
        n = len(self.names)
        self.dim = int(meta["dim"])
        self.model_sha = meta.get("model_sha")
        self.calib_sha = meta.get("calib_sha") or meta.get("sha")
        self.tier: list[str] = list(meta.get("tier") or ["uncalibrated"] * n)
        # tau/platt are per-tag and may be null: "not calibrated yet", never a guessed
        # threshold (tags.py's law). A null tag falls back to dataset-layer stats below
        # and can never gate, whatever its tier says.
        self.tau: list = list(meta.get("tau") or [None] * n)
        self.platt: list = list(meta.get("platt") or [None] * n)
        self.theta_syn = float(meta.get("theta_syn", THETA_SYN))
        self.text_logistic = tuple(meta.get("text_logistic") or (TEXT_A, TEXT_B))
        self.text_feature = meta.get("text_feature", TEXT_FEATURE)
        self.emb = np.memmap(d / "tags.f32", np.float32, "r", shape=(n, self.dim))
        self.index = {name: i for i, name in enumerate(self.names)}

    def calibrated(self, i: int) -> bool:
        """Gating-eligible: calibrated TIER *and* an actual fit on disk (ADR-3)."""
        return self.tier[i] == "calibrated" and bool(self.platt[i]) and self.tau[i] is not None

    @classmethod
    def load(cls, model_sha: str, home: Path | None = None) -> "TagTable | None":
        d = (home or imgtag_home()) / "models" / model_sha
        if not ((d / "tags.json").is_file() and (d / "tags.f32").is_file()):
            return None  # free-text path works standalone (ADR-3)
        return cls(d)


# ---------------------------------------------------------------- searcher


class Searcher:
    """Resident query engine: snapshot cache + warm text tower + LRU query-vector cache.

    One instance per process (the daemon holds it for its lifetime, ADR-5). Snapshots are
    re-opened only when the manifest changes, so rows committed by a running index job
    become visible within one query of their commit (B11) without re-mmapping every request.
    """

    def __init__(self, home: Path | None = None, backend=None, cache: int = 256):
        self.home = home or imgtag_home()
        self._backend = backend
        self._snaps: dict[str, tuple[tuple, object]] = {}
        self._tags: dict[str, TagTable | None] = {}
        self._qvec = lru_cache(maxsize=cache)(self._embed_uncached)
        self._own_backend = backend is None  # only drop what we loaded ourselves
        self.text_loaded = False
        self.last_query = 0.0  # for the daemon's eviction policy (ADR-5, revised)

    # -- resources -------------------------------------------------
    def _manifest_stamp(self, dataset: str) -> tuple:
        st = (dataset_dir(dataset, self.home) / "manifest.json").stat()
        return (st.st_mtime_ns, st.st_size)

    def snapshot(self, dataset: str):
        try:
            stamp = self._manifest_stamp(dataset)
        except FileNotFoundError:
            raise UnknownDatasetError(f"no index for dataset {dataset!r}") from None
        hit = self._snaps.get(dataset)
        if hit is None or hit[0] != stamp:
            self._snaps[dataset] = (stamp, open_snapshot(dataset, self.home))
        return self._snaps[dataset][1]

    def backend(self, manifest: dict):
        """The text tower for this manifest's model. Loud refusal on sha mismatch (ADR-6)."""
        if self._backend is None:
            from . import models as _m

            self._backend = _m.load_backend(manifest["model_id"].rsplit("-", 1)[0])
        if self._backend.model_sha != manifest.get("model_sha"):
            from .store import ModelMismatchError

            raise ModelMismatchError(
                f"index built with {manifest.get('model_id')} ({manifest.get('model_sha')!r}); "
                f"this process holds {self._backend.model_id} ({self._backend.model_sha!r}) "
                f"— restart the daemon or reindex"
            )
        return self._backend

    def tags(self, manifest: dict) -> TagTable | None:
        sha = manifest["model_sha"]
        if sha not in self._tags:
            t = TagTable.load(sha, self.home)
            if t is not None:
                want_model, want_calib = manifest.get("calib_model_sha"), manifest.get("calib_sha")
                if (want_model and want_model != t.model_sha) or (want_calib and want_calib != t.calib_sha):
                    raise CalibrationMismatchError(
                        f"tag table {t.dir} (model_sha={t.model_sha!r}, calib_sha={t.calib_sha!r}) "
                        f"does not match manifest (calib_model_sha={want_model!r}, "
                        f"calib_sha={want_calib!r}) — recalibrate; the tag path refuses rather than lie"
                    )
            self._tags[sha] = t
        return self._tags[sha]

    def _embed_uncached(self, sha: str, query: str) -> np.ndarray:
        return np.ascontiguousarray(self._backend.embed_texts([query])[0], np.float32)

    def embed(self, manifest: dict, query: str) -> tuple[np.ndarray, float]:
        """Query vector + the ms spent loading the text tower (0 when it was already warm).

        The load cost is returned, not hidden: it belongs in ``tookMs`` AND in its own
        labeled field so a warm-latency budget (B3) never silently absorbs a B13-shaped
        one-time cost.
        """
        t0 = time.perf_counter()
        cold = self._backend is None or getattr(self._backend, "_ts", None) is None
        self.backend(manifest)  # refuses on mismatch before any work
        v = self._qvec(manifest["model_sha"], query)
        self.text_loaded = True
        return v, round((time.perf_counter() - t0) * 1000, 2) if cold else 0.0

    def release_text(self) -> bool:
        """Drop the resident model. ADR-5 (revised): the IDLE resident set is shards +
        tokenizer + tag table only — no text tower, and no vision session either (the
        daemon never indexes; that runs in the `imgtag index` subprocess)."""
        be, self.text_loaded = self._backend, False
        if be is None:
            return False
        be.release_text()
        if self._own_backend:  # we loaded it -> we may drop it whole (vision session too)
            self._backend = None
        self._qvec.cache_clear()
        return True

    # -- tag path --------------------------------------------------
    def concepts(self, table: TagTable, query: str) -> list[tuple[str, list[int]]]:
        """Split a SPACE-SEPARATED TAG LIST into concepts — the ALL-SOME-ANY spectrum.

        VISION-ADDENDA 2026-07-22 ~12:12Z (verbatim): "in the search i should be able to
        use space to enter more tags (show results that have all tags first, then
        decending for higher n/m found tags number, then finally results from each tag
        (like any) … its like a specturm between ALL-SOME-ANY in tag-search".

        Multi-tag mode needs EVERY content token to name a tag (directly or through the
        hypernym table). That test is what keeps ADR-3 §3 intact: "my dog wearing a santa
        hat" contains non-tag words, so it stays one natural-language query on the text
        path and never inherits `dog`'s threshold — while "dog beach sunset" is three
        exact tag names and is scored as three concepts.
        """
        toks = content_tokens(query)
        if len(toks) < 2:
            return []
        out: list[tuple[str, list[int]]] = []
        i = 0
        while i < len(toks):
            # greedy longest match first: the vocabulary has multi-word tags ("ocean wave",
            # "night sky"), so "ocean wave sunset" is TWO concepts, not three unknown words.
            for span in range(min(NGRAM, len(toks) - i), 0, -1):
                term = " ".join(toks[i : i + span])
                idxs = [table.index[n] for n in expand(term) if n in table.index]
                if idxs:
                    out.append((term, idxs[:MAX_TAGS]))
                    i += span
                    break
            else:
                return []  # a non-tag word -> not a tag list; text path owns this query
        return out if len(out) > 1 else []

    def _name_candidates(self, table: TagTable, query: str) -> list[tuple[int, str]]:
        """Tags the query NAMES — exact or via the static hypernym table. No embedding
        involved, so a named-tag query is served with zero text-encoder involvement."""
        out: dict[int, str] = {}
        # A compound query NEVER inherits a COMPONENT tag's calibration (ADR-3 §3): only the
        # whole query string may name a tag; single-concept queries also match per token.
        terms = [query.lower().strip()] + ([] if is_compound(query) else content_tokens(query))
        for term in terms:
            for name in expand(term):
                i = table.index.get(name)
                if i is not None:
                    out.setdefault(i, "exact" if name == term else "hypernym")
        return sorted(out.items())[:MAX_TAGS]

    def _near_candidates(self, table: TagTable, query: str, q: np.ndarray) -> list[tuple[int, str]]:
        """Near-tag rule (ADR-3 §3): inherit a tag's calibration at cos >= theta_syn.
        Needs the query vector, so it only ever runs when the text path ran anyway."""
        if is_compound(query):
            return []
        cos = np.asarray(table.emb @ q, np.float32)
        out = []
        for i in np.argsort(-cos)[:MAX_TAGS]:
            if cos[i] < table.theta_syn:
                break
            out.append((int(i), "near"))
        return out

    def _tag_scores(self, snap, table: TagTable, idx: list[int], manifest: dict):
        """[N,C] calibrated probabilities + each tag's effective threshold.

        Two layers, exactly as ADR-3 orders them. A tag WITH a model-layer Platt fit uses
        it (tags.py's convention, ``p = sigmoid(-(A*s+B))`` — imported, never re-derived)
        and its fitted max-F1 tau, raised by the dataset layer when the manifest carries
        per-tag stats. A tag WITHOUT a fit gets the dataset layer alone: its own score
        distribution over THIS corpus, z-scored through the same global logistic. No
        per-tag number is ever invented, and an unfitted tag can never gate.
        """
        from .tags import platt_apply

        T = np.ascontiguousarray(np.asarray(table.emb)[idx].T, np.float32)  # [D, C]
        cos = np.asarray(snap.emb @ T, np.float32).reshape(len(snap.ids), len(idx))
        a, b = table.text_logistic
        stats = manifest.get("tag_stats") or {}
        P = np.empty_like(cos)
        eff = np.empty(len(idx), np.float32)
        present = np.empty(len(idx), np.float32)  # softer floor, ranking only
        for j, i in enumerate(idx):
            st = stats.get(table.names[i]) or {}
            if table.platt[i]:  # model layer: fitted per-tag sigmoid + max-F1 tau
                P[:, j] = platt_apply(cos[:, j], table.platt[i])
                floor = float(table.tau[i]) if table.tau[i] is not None else TAU
                if st:  # dataset layer can only RAISE the bar, never lower it
                    floor = max(floor, float(st.get("mean", 0.0)) + K_STD * float(st.get("std", 0.0)))
                eff[j] = present[j] = floor  # a fitted tau is the right bar for both
            else:  # dataset layer alone
                mu = float(st["mean"]) if "mean" in st else float(cos[:, j].mean())
                sd = float(st["std"]) if "std" in st else float(cos[:, j].std())
                P[:, j] = _sigmoid(a * ((cos[:, j] - mu) / max(sd, 1e-6)) + b)
                eff[j] = float(_sigmoid(a * K_STD + b))
                present[j] = float(_sigmoid(a * SPECTRUM_K + b))
        return P, eff, cos, present

    # -- the query -------------------------------------------------
    def search_one(self, query: str, dataset: str, k: int = 50, strict: bool = False,
                   text: str = "auto") -> dict:
        """One dataset. ``text`` selects the encoder policy (ADR-5, revised resident set):

        * ``auto`` (default) — a query that NAMES a tag is served from the tag table alone,
          never waking the text tower; anything else loads it.
        * ``never`` — the text tower is never loaded; unnamed queries honestly no-match.
        * ``always`` — always fuse both paths (what a bench harness wants).

        The choice depends only on (query, tag table, policy), never on cache state, so
        results stay deterministic across restarts (B18e).
        """
        snap = self.snapshot(dataset)
        man = snap.manifest
        n = len(snap.ids)
        if n == 0:
            return {"hits": [], "gated": False, "indexed": 0, "best_p_text": 0.0,
                    "text_tower": "skipped", "load_ms": 0.0}
        table = self.tags(man)
        cands = self._name_candidates(table, query) if table is not None else []
        groups = self.concepts(table, query) if table is not None else []
        if groups:
            # A tag LIST is not a natural-language compound: each token is itself a tag, so
            # every group's tags are candidates (ADR-3 §3 still bars borrowing a threshold
            # through the near-tag rule — see concepts()).
            named = {i: "tag-list" for _, idxs in groups for i in idxs}
            named.update(dict(cands))
            cands = sorted(named.items())[:MAX_TAGS]
        use_text = text == "always" or (text != "never" and not cands)

        q, load_ms, s, z = None, 0.0, None, np.zeros(n, np.float32)
        p_text = np.zeros(n, np.float32)
        if use_text:
            q, load_ms = self.embed(man, query)
            s = np.asarray(snap.emb @ q, np.float32)  # cosine: both sides are L2-normalized
            a, b = table.text_logistic if table else (TEXT_A, TEXT_B)
            z = (s - s.mean()) / max(float(s.std()), 1e-6)
            feature = table.text_feature if table else TEXT_FEATURE
            p_text = _sigmoid(a * (s if feature == "cos" else z) + b)
            if table is not None:
                cands = sorted(dict(cands + self._near_candidates(table, query, q)).items())[:MAX_TAGS]

        p_tag = np.zeros(n, np.float32)
        col = np.full(n, -1, np.int32)  # winning tag column per row, -1 = no tag path
        gated = False  # a CALIBRATED tag cleared its effective tau somewhere
        tag_cos = None
        matched = np.zeros(n, np.int32)  # ALL-SOME-ANY: how many query tags this row has
        n_concepts = len(groups)
        if cands:
            idx = [i for i, _ in cands]
            P, eff, tag_cos, present = self._tag_scores(snap, table, idx, man)
            col = P.argmax(1).astype(np.int32)
            p_tag = P[np.arange(n), col].astype(np.float32)
            calib = np.asarray([table.calibrated(i) for i in idx])
            if calib.any():
                gated = bool((P[:, calib] >= eff[calib]).any())
            if n_concepts:  # rank by ALL first, then n/m descending, then ANY
                pos = {i: j for j, i in enumerate(idx)}
                for _, tag_idxs in groups:
                    cols = [pos[i] for i in tag_idxs if i in pos]
                    if cols:
                        matched += (P[:, cols] >= present[cols]).any(1)
            if strict:  # --strict: calibrated tags become a hard AND
                keep = np.ones(n, bool)
                for j, i in enumerate(idx):
                    if table.calibrated(i):
                        keep &= P[:, j] >= eff[j]
                p_tag = np.where(keep, p_tag, 0.0).astype(np.float32)
                p_text = np.where(keep, p_text, 0.0).astype(np.float32)

        p = np.maximum(p_tag, p_text)
        # ALL-SOME-ANY ordering (VISION-ADDENDA): tag-count is the PRIMARY key, probability
        # the tiebreak. p is in [0,1), so `matched + p` orders both in one pass.
        rank = matched.astype(np.float32) + p
        top = np.argpartition(-rank, min(k, n) - 1)[:k] if k < n else np.arange(n)
        hits = []
        for i in top:
            i = int(i)
            if p[i] < TAU:
                continue
            rec = snap.ids[i]
            if p_tag[i] > p_text[i] and col[i] >= 0:
                ti, how = cands[int(col[i])]
                why = {"path": "tag", "tag": table.names[ti], "match": how, "tier": table.tier[ti]}
                # tag-path score is cos(image, tag vector) — there may be no query vector at all
                score = float(tag_cos[i, int(col[i])])
            else:
                why = {"path": "text"}
                score = float(s[i])
            why.update(p_tag=round(float(p_tag[i]), 4), p_text=round(float(p_text[i]), 4), z=round(float(z[i]), 3))
            if n_concepts:
                why.update(tags_matched=int(matched[i]), tags_total=n_concepts,
                           spectrum="all" if matched[i] == n_concepts else
                                    ("some" if matched[i] > 1 else "any"))
            hits.append(
                {
                    "image_id": rec["image_id"],
                    "path": rec["path"],
                    "dataset": rec.get("dataset") or dataset,  # B18: provenance NEVER null
                    "dataset_slug": rec.get("dataset") or dataset,
                    "score": round(score, 6),
                    "p": round(float(p[i]), 4),
                    "why": why,
                }
            )
        hits.sort(key=_order)  # ALL-SOME-ANY, then p, then image id (B18e determinism)
        return {"hits": hits[:k], "gated": gated, "indexed": n, "best_p_text": float(p_text.max()),
                "text_tower": ("loaded" if load_ms else "warm") if use_text else "skipped",
                "load_ms": load_ms}

    def search(self, query: str, dataset: str | None = None, k: int = 50, strict: bool = False,
               text: str = "auto") -> dict:
        """Search one dataset, or every dataset on disk when ``dataset`` is None."""
        t0 = time.perf_counter()
        self.last_query = time.time()
        names = [dataset] if dataset else list_datasets(self.home)
        hits: list[dict] = []
        indexed, gated, best_text, load_ms = 0, False, 0.0, 0.0
        tower = "skipped"
        for name in names:
            r = self.search_one(query, name, k=k, strict=strict, text=text)
            hits += r["hits"]
            indexed += r["indexed"]
            gated |= r["gated"]
            best_text = max(best_text, r["best_p_text"])
            load_ms += r["load_ms"]
            tower = r["text_tower"] if tower == "skipped" else tower
        hits.sort(key=_order)
        return {
            "query": query,
            "tookMs": round((time.perf_counter() - t0) * 1000, 2),
            # the one-time tower load is INSIDE tookMs and labeled here, so a warm-latency
            # budget (B3) can exclude it instead of silently absorbing a B13-shaped cost.
            "text_tower": tower,
            "text_tower_load_ms": round(load_ms, 2),
            "coverage": {"indexed": indexed, "total": total_expected(names, indexed, self.home)},
            "datasets": names,
            "hits": hits[:k],
            # honest no-match: no calibrated tag cleared its tau AND nothing cleared the
            # free-text floor. Uncalibrated tags may never veto (ADR-3 tiers).
            "no_match": not hits and not gated and best_text < TAU,
        }


def total_expected(datasets: list[str], indexed: int, home: Path | None = None) -> int:
    """Coverage denominator: indexed rows + whatever a live index job still owes us."""
    from .progress import list_jobs

    pending = sum(
        max(0, int(j.get("total", 0)) - int(j.get("done", 0)))
        for j in list_jobs(home)
        if j.get("dataset") in datasets and j.get("state") in ("queued", "running")
    )
    return indexed + pending
