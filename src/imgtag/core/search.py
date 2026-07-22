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
TEXT_A, TEXT_B = 1.6, -4.0  # p_text = sigmoid(A*z + B) over the per-query corpus z-score
TAU = 0.5  # probability floor for "this is a match"
THETA_SYN = 0.90  # near-tag rule: inherit a tag's calibration only at cos >= theta_syn
K_STD = 3.0  # dataset-layer effective tau = max(tau_tag, mean + K_STD*std)
MAX_TAGS = 32  # candidate tags per query (bounds the [N,C] tag matmul)
DATA = Path(__file__).resolve().parent.parent / "data"


class CalibrationMismatchError(RuntimeError):
    """Manifest calib_sha/calib_model_sha != the tag table on disk (CLI exit 5)."""


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
        self.tau = np.asarray(meta.get("tau") or [TAU] * n, np.float32)
        self.platt = np.asarray(meta.get("platt") or [[1.0, 0.0]] * n, np.float32)
        self.theta_syn = float(meta.get("theta_syn", THETA_SYN))
        self.text_logistic = tuple(meta.get("text_logistic") or (TEXT_A, TEXT_B))
        self.emb = np.memmap(d / "tags.f32", np.float32, "r", shape=(n, self.dim))
        self.index = {name: i for i, name in enumerate(self.names)}

    def calibrated(self, i: int) -> bool:
        return self.tier[i] == "calibrated"

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
        self.last_query = 0.0  # for the daemon's --text-ttl eviction (ADR-5, revised)

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

    def embed(self, manifest: dict, query: str) -> np.ndarray:
        self.backend(manifest)  # refuses on mismatch before any work
        return self._qvec(manifest["model_sha"], query)

    # -- tag path --------------------------------------------------
    def _candidate_tags(self, table: TagTable, query: str, q: np.ndarray) -> list[tuple[int, str]]:
        """Exact + hypernym-expanded name matches, then the near-tag rule (ADR-3 §3)."""
        out: dict[int, str] = {}
        compound = is_compound(query)
        # A compound query NEVER inherits a COMPONENT tag's calibration (ADR-3 §3): only the
        # whole query string may name a tag; single-concept queries also match per token.
        terms = [query.lower().strip()] + ([] if compound else content_tokens(query))
        for term in terms:
            for name in expand(term):
                i = table.index.get(name)
                if i is not None:
                    out.setdefault(i, "exact" if name == term else "hypernym")
        if not compound:  # …and never borrows one via the near-tag rule either
            cos = np.asarray(table.emb @ q, np.float32)
            for i in np.argsort(-cos)[:MAX_TAGS]:
                if cos[i] < table.theta_syn:
                    break
                out.setdefault(int(i), "near")
        return sorted(out.items())[:MAX_TAGS]

    def _tag_scores(self, snap, table: TagTable, idx: list[int], manifest: dict):
        T = np.ascontiguousarray(np.asarray(table.emb)[idx].T, np.float32)  # [D, C]
        cos = np.asarray(snap.emb @ T, np.float32).reshape(len(snap.ids), len(idx))
        P = _sigmoid(cos * table.platt[idx, 0] + table.platt[idx, 1])  # [N, C]
        stats = manifest.get("tag_stats") or {}
        eff = np.asarray(
            [
                max(
                    float(table.tau[i]),
                    float((stats.get(table.names[i]) or {}).get("mean", 0.0))
                    + K_STD * float((stats.get(table.names[i]) or {}).get("std", 0.0)),
                )
                for i in idx
            ],
            np.float32,
        )
        return P, eff

    # -- the query -------------------------------------------------
    def search_one(self, query: str, dataset: str, k: int = 50, strict: bool = False) -> dict:
        snap = self.snapshot(dataset)
        man = snap.manifest
        n = len(snap.ids)
        if n == 0:
            return {"hits": [], "gated": False, "indexed": 0, "best_p_text": 0.0}
        q = self.embed(man, query)
        s = np.asarray(snap.emb @ q, np.float32)  # cosine: both sides are L2-normalized
        table = self.tags(man)
        a, b = table.text_logistic if table else (TEXT_A, TEXT_B)
        z = (s - s.mean()) / max(float(s.std()), 1e-6)
        p_text = _sigmoid(a * z + b)

        p_tag = np.zeros(n, np.float32)
        col = np.full(n, -1, np.int32)  # winning tag column per row, -1 = no tag path
        gated = False  # a CALIBRATED tag cleared its effective tau somewhere
        cands = self._candidate_tags(table, query, q) if table is not None else []
        if cands:
            idx = [i for i, _ in cands]
            P, eff = self._tag_scores(snap, table, idx, man)
            col = P.argmax(1).astype(np.int32)
            p_tag = P[np.arange(n), col].astype(np.float32)
            calib = np.asarray([table.calibrated(i) for i in idx])
            if calib.any():
                gated = bool((P[:, calib] >= eff[calib]).any())
            if strict:  # --strict: calibrated tags become a hard AND
                keep = np.ones(n, bool)
                for j, i in enumerate(idx):
                    if table.calibrated(i):
                        keep &= P[:, j] >= eff[j]
                p_tag = np.where(keep, p_tag, 0.0).astype(np.float32)
                p_text = np.where(keep, p_text, 0.0).astype(np.float32)

        p = np.maximum(p_tag, p_text)
        top = np.argpartition(-p, min(k, n) - 1)[:k] if k < n else np.arange(n)
        hits = []
        for i in top:
            i = int(i)
            if p[i] < TAU:
                continue
            rec = snap.ids[i]
            if p_tag[i] > p_text[i] and col[i] >= 0:
                ti, how = cands[int(col[i])]
                why = {"path": "tag", "tag": table.names[ti], "match": how, "tier": table.tier[ti]}
            else:
                why = {"path": "text"}
            why.update(p_tag=round(float(p_tag[i]), 4), p_text=round(float(p_text[i]), 4), z=round(float(z[i]), 3))
            hits.append(
                {
                    "image_id": rec["image_id"],
                    "path": rec["path"],
                    "dataset": rec.get("dataset") or dataset,  # B18: provenance NEVER null
                    "dataset_slug": rec.get("dataset") or dataset,
                    "score": round(float(s[i]), 6),
                    "p": round(float(p[i]), 4),
                    "why": why,
                }
            )
        hits.sort(key=lambda h: (-h["p"], h["image_id"]))  # B18(e) determinism: ties by id
        return {"hits": hits[:k], "gated": gated, "indexed": n, "best_p_text": float(p_text.max())}

    def search(self, query: str, dataset: str | None = None, k: int = 50, strict: bool = False) -> dict:
        """Search one dataset, or every dataset on disk when ``dataset`` is None."""
        t0 = time.perf_counter()
        self.last_query = time.time()
        names = [dataset] if dataset else list_datasets(self.home)
        hits: list[dict] = []
        indexed, gated, best_text = 0, False, 0.0
        for name in names:
            r = self.search_one(query, name, k=k, strict=strict)
            hits += r["hits"]
            indexed += r["indexed"]
            gated |= r["gated"]
            best_text = max(best_text, r["best_p_text"])
        hits.sort(key=lambda h: (-h["p"], h["image_id"]))
        return {
            "query": query,
            "tookMs": round((time.perf_counter() - t0) * 1000, 2),
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
