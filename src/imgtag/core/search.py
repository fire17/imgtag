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
tau is an honest ``no_match``, never an empty list pretending to be a miss — but ONLY when
a real fit exists. **Fail-open law:** with no CAL-SET fit for the loaded model_sha the
engine returns the ranking labeled ``"calibration": "unfitted"`` and never no-matches;
"I cannot judge" must never be dressed up as "nothing matched".
"""

from __future__ import annotations

import json
import os
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
TEXT_FEATURE = "cos"  # "cos" = raw cosine | "z" = per-query corpus z-score.
# ADR-3 §2 was RE-RULED 2026-07-22 on this lane's measurement: on unsplash-demo (N=2000,
# pecore-s16-384-fp32), 15 real vs 15 nonsense queries, the best single threshold separates
# 77% on "cos" but only 60% on "z" (chance = 50%) — nonsense max-z (med 4.16) EXCEEDS real
# max-z (med 3.81), so no (A,B) on z can produce an honest no-match.
TEXT_A, TEXT_B = 60.0, -17.6  # p_text = sigmoid(A*x + B); x is TEXT_FEATURE
# (A,B) place p=0.5 at cos 0.2934 — the measured best-separating threshold above — with a
# slope that spans the measured real/nonsense gap (real med 0.300 vs nonsense med 0.277).
# MODEL-SPECIFIC and PROVISIONAL: cosine scale differs per model, so these hold only for
# pecore-s16-384 until b-bench's CAL-SET fit lands in tags.json per model_sha. Any result
# scored this way is labeled `"calibration": "unfitted"` — never "fitted" — and, per the
# FAIL-OPEN law, an unfitted threshold NEVER vetoes: it ranks and says it cannot judge.
Z_A, Z_B = 1.6, -4.0  # the Z-SCALE pair: p = sigmoid(Z_A*z + Z_B). Used by the dataset
# layer (an unfitted tag's own score distribution over this corpus) and by the "z" text
# feature. Kept SEPARATE from (TEXT_A, TEXT_B) because the two features live on different
# scales — feeding a z-score through the cosine pair saturates every probability to 1.0.
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
# An image-relative presence test (term inside the image's own top-R tags) was built and
# A/B-MEASURED against COCO ground truth on cocoval2017, 10 tag pairs x top-20: identical
# ALL-tier precision (61/169 = 36%) and recall (61/66 = 92%) with and without it. Inert,
# so it was deleted rather than kept as decoration. The real lever is a FITTED per-tag tau.
MAX_TAGS = 32  # candidate tags per query (bounds the [N,C] tag matmul)
NGRAM = 3  # longest multi-word tag a space-separated tag list may name ("night sky")
DATA = Path(__file__).resolve().parent.parent / "data"


class CalibrationMismatchError(RuntimeError):
    """Manifest calib_sha/calib_model_sha != the tag table on disk (CLI exit 5)."""


def dedupe(hits: list[dict]) -> tuple[list[dict], int]:
    """Collapse rows that describe the SAME image (IA.md: the id is xxhash64 of the bytes,
    so duplicates are the same picture). Keeps the best-ranked row, folds the other paths
    into `paths`, and RETURNS the count collapsed so the caller can surface it instead of
    hiding an indexer bug (b-app found 200 hits carrying 181 unique ids, one id x21)."""
    out: list[dict] = []
    seen: dict[tuple, dict] = {}
    for h in hits:
        key = (h["dataset"], h["image_id"])
        first = seen.get(key)
        if first is None:
            seen[key] = h
            out.append(h)
            continue
        paths = first.setdefault("paths", [first["path"]])
        if h["path"] not in paths:
            paths.append(h["path"])
    return out, len(hits) - len(out)


def _order(h: dict) -> tuple:
    """Result order: coverage tier first (ALL > SOME > ANY), then the MEAN probability over
    the matched terms, then probability, then image id. Tiers therefore arrive contiguous
    in the payload (b-app renders them as bands) and the id tiebreak keeps a result list
    byte-identical across runs (B18e)."""
    t = (h.get("why") or {}).get("terms") or {}
    return (-int(t.get("m", 0)), -float(t.get("mean_p", 0.0)), -h["p"], h["image_id"])


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


_QUOTED = re.compile(r'"([^"]*)"')


def split_terms(query: str) -> list[tuple[str, bool]]:
    """Split a query into (text, is_quoted) segments for multi-term search.

    Double-quoted spans survive whole ("night city" stays one term); everything else
    splits on whitespace with stopwords dropped. A query made ENTIRELY of stopwords (or
    of one quoted span) yields a single phrase segment, so it can never be shredded.
    """
    segs: list[tuple[str, bool]] = []
    pos = 0
    for m in _QUOTED.finditer(query):
        segs += [(t, False) for t in content_tokens(query[pos : m.start()])]
        phrase = " ".join(m.group(1).lower().split())
        if phrase:
            segs.append((phrase, True))
        pos = m.end()
    segs += [(t, False) for t in content_tokens(query[pos:])]
    if not segs:
        return [(" ".join(query.lower().split()), True)]
    return segs


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


# ---------------------------------------------------------------- moderation


@lru_cache(maxsize=1)
def tracks() -> dict:
    """Moderation track definitions (VISION-ADDENDA 2026-07-22 ~12:33Z).

    A track is a PROMPT SET, not a tag-table lookup: measured on b-bench's 2177-tag
    vocabulary, 'nudity'/'nude'/'naked' do not exist in it at all — only clothing items,
    and a swimsuit is not nudity. Scoring prompts directly is the only honest route.
    Every track is UNFITTED until b-bench calibrates it, and says so in the payload.
    """
    try:
        return json.loads((DATA / "moderation.json").read_bytes())
    except (OSError, ValueError):
        return {"tracks": {}, "enforcement_ready": False}


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
        # A fit exists only if someone FITTED it: an explicit text logistic in the file, or
        # at least one tag carrying both a Platt pair and a tau (b-bench's CAL-SET output).
        self.fitted = bool(meta.get("text_logistic")) or any(
            bool(a) and t is not None for a, t in zip(self.platt, self.tau))
        self.z_logistic = tuple(meta.get("z_logistic") or (Z_A, Z_B))
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
        self._tracks: dict[str, tuple[tuple, dict]] = {}  # moderation scores per generation
        self._track_vecs: dict = {}
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

    # -- moderation ------------------------------------------------
    def track_scores(self, dataset: str) -> dict:
        """Per-image scores for every moderation category, in BOTH ADR-14 tiers.

        Two spec shapes are honoured, per category:

        * **fitted** (``scorer: "margin"`` + ``tau``/``tau_review``, e.g. the drugs track):
          feature = max(positive concepts) - max(background concepts), thresholded at the
          FITTED taus in margin space. ``policy_neighbours`` are scored for EXPLANATION
          ONLY and never subtracted — measured by that lane, subtracting them collapses AP
          0.58 to 0.04, because a clinical syringe is visually identical to a drug syringe.
        * **unfitted**: the same margin feature, z-scored against THIS corpus, tiered by
          which prompt set the image resembles more (ADR-14).
        """
        snap = self.snapshot(dataset)
        stamp = self._manifest_stamp(dataset)
        hit = self._tracks.get(dataset)
        if hit and hit[0] == stamp:
            return hit[1]
        spec = tracks().get("categories") or {}
        if not spec or not len(snap.ids):
            out = {"categories": {}, "counts": {}, "n": len(snap.ids), "floor": 1.0}
            self._tracks[dataset] = (stamp, out)
            return out
        table = self.tags(snap.manifest)
        za, zb = table.z_logistic if table else (Z_A, Z_B)
        zfloor = float(_sigmoid(za * K_STD + zb))
        groups, cols = self._track_concepts(snap.manifest, spec)
        cos = np.asarray(snap.emb @ cols.T, np.float32).reshape(len(snap.ids), cols.shape[0])

        def best(sl):  # max similarity over a concept group (empty group -> -inf)
            return cos[:, sl].max(1) if sl.stop > sl.start else np.full(len(snap.ids), -1e9, np.float32)

        cats, counts = {}, {}
        for name, g in groups.items():
            cfg = spec[name]
            # A PROXY fit does not gate. Ruling 2026-07-22 after b-app's audit: the drugs
            # track's proxy-fitted logistic put 218 violations on this corpus (vs nudity 5,
            # weapons 10) with the first 21 eyeballed all benign at a SATURATED p=0.99 — a
            # fit that is worse than no fit gets rolled back, per staleness-worse-than-
            # absence. Only `calibration: "fitted"` may use its own taus; a proxy spec keeps
            # its declared scorer but falls back to the dataset layer and says "unfitted".
            trusted = cfg.get("calibration") == "fitted"
            fitted = trusted and cfg.get("scorer") == "margin" and cfg.get("tau") is not None
            if fitted:
                # margin ONLY where the spec was fitted for it: its negative set is a
                # background for a max-margin, not a mean to subtract. Applying margin to a
                # spec written the other way MEASURABLY broke nudity — "a clothed person"
                # dominates the background of any photo containing a person, so a genuine
                # topless image scored no flag at all.
                bg = best(g["negatives"])
                mv, mr = best(g["violation"]) - bg, best(g["review"]) - bg
                tv = float(cfg["tau"])
                tr_ = float(cfg.get("tau_review", tv))
                over_v, over_r = mv - tv, mr - tr_
                # tier by which threshold is exceeded MORE. `tau_review` above `tau` would
                # otherwise make the review tier unreachable (everything clearing the higher
                # review bar has already cleared the lower violation bar) — measured on the
                # drugs spec, where a vape landed in `violation` against ADR-14.
                is_v = (over_v >= 0) & (over_v >= over_r)
                is_r = (over_r >= 0) & ~is_v
                pv, pr = self._margin_p(mv, cfg), self._margin_p(mr, cfg)
            elif cfg.get("scorer") == "margin":  # declared margin, untrusted thresholds:
                # keep the feature, drop the fitted taus, z-score against THIS corpus
                bg = best(g["negatives"])
                mv, mr = best(g["violation"]) - bg, best(g["review"]) - bg
                pv = _sigmoid(za * ((mv - mv.mean()) / max(float(mv.std()), 1e-6)) + zb)
                pr = _sigmoid(za * ((mr - mr.mean()) / max(float(mr.std()), 1e-6)) + zb)
                fires = np.maximum(pv, pr) >= zfloor
                is_v = fires & (pv > pr)
                is_r = fires & ~is_v
            else:  # unfitted spec: mean-of-prompts minus half the mean of the negatives,
                # z-scored against THIS corpus (the shape those prompt sets were written for)
                neg = cos[:, g["negatives"]].mean(1) if g["negatives"].stop > g["negatives"].start else 0.0
                mv = (cos[:, g["violation"]].mean(1) if g["violation"].stop > g["violation"].start
                      else np.zeros(len(snap.ids), np.float32)) - 0.5 * neg
                mr = (cos[:, g["review"]].mean(1) if g["review"].stop > g["review"].start
                      else np.zeros(len(snap.ids), np.float32)) - 0.5 * neg
                pv = _sigmoid(za * ((mv - mv.mean()) / max(float(mv.std()), 1e-6)) + zb)
                pr = _sigmoid(za * ((mr - mr.mean()) / max(float(mr.std()), 1e-6)) + zb)
                fires = np.maximum(pv, pr) >= zfloor
                is_v = fires & (pv > pr)
                is_r = fires & ~is_v
            cats[name] = {"violation_p": pv, "review_p": pr, "is_violation": is_v,
                          "is_review": is_r,
                          # a proxy fit is reported as UNFITTED, with its claim preserved
                          "calibration": cfg.get("calibration", "unfitted") if trusted else "unfitted",
                          "spec_calibration": cfg.get("calibration", "unfitted"),
                          "enforcement_ready": bool(cfg.get("enforcement_ready", False)),
                          # explanation-only signal, never subtracted, never a gate
                          "neighbour": best(g["policy_neighbours"]) if g["policy_neighbours"].stop
                          > g["policy_neighbours"].start else None}
            counts[name] = {"violation": int(is_v.sum()), "review": int(is_r.sum())}
        out = {"categories": cats, "counts": counts, "floor": zfloor, "n": len(snap.ids),
               "labels": {n: spec[n].get("label", n) for n in spec}}
        self._tracks[dataset] = (stamp, out)
        return out

    @staticmethod
    def _margin_p(m: np.ndarray, cfg: dict) -> np.ndarray:
        """Fitted margin -> probability via that track's Platt pair (project convention:
        ``p = sigmoid(-(A*s + B))``, tags.platt_apply). Cosmetic next to the fitted tau,
        which does the gating in margin space."""
        ab = cfg.get("platt")
        if not ab:
            return _sigmoid(m)
        from .tags import platt_apply

        return np.asarray(platt_apply(m, ab), np.float32)

    def _track_concepts(self, manifest: dict, spec: dict):
        """Embed every concept of every category ONCE (templates expanded and averaged).

        Returns (groups, matrix): groups[name][role] is a slice into the concept matrix, so
        scoring is one [N, C] matmul and `max over a role` is a slice-max.
        """
        key = (manifest["model_sha"], tracks().get("version"), tuple(sorted(spec)))
        if self._track_vecs.get("key") == key:
            return self._track_vecs["groups"], self._track_vecs["cols"]
        roles = ("violation", "review", "negatives", "policy_neighbours")
        texts, groups, n = [], {}, 0
        for name, cfg in spec.items():
            tpl = cfg.get("templates") or ["{}"]
            groups[name] = {}
            for role in roles:
                start = n
                for concept in cfg.get(role) or []:
                    texts.append([t.format(concept) for t in tpl])
                    n += 1
                groups[name][role] = slice(start, n)
        self.backend(manifest)
        flat = [t for group in texts for t in group]
        emb = np.asarray(self._backend.embed_texts(flat), np.float32)
        self.text_loaded = True
        rows, at = [], 0
        for group in texts:  # average the templates, renormalize: one vector per concept
            v = emb[at : at + len(group)].mean(0)
            at += len(group)
            rows.append(v / max(float(np.linalg.norm(v)), 1e-12))
        cols = np.ascontiguousarray(rows, np.float32) if rows else np.zeros((0, emb.shape[1]), np.float32)
        self._track_vecs = {"key": key, "groups": groups, "cols": cols}
        return groups, cols

    def flags_for(self, tr: dict, i: int) -> list[dict]:
        """ADR-14 per-image payload: [{category, p, tier}] for whatever actually fired."""
        out = []
        for name, c in (tr.get("categories") or {}).items():
            if c["is_violation"][i]:
                out.append({"category": name, "p": round(float(c["violation_p"][i]), 4),
                            "tier": "violation"})
            elif c["is_review"][i]:
                out.append({"category": name, "p": round(float(c["review_p"][i]), 4),
                            "tier": "review"})
        return out

    def track_state(self, dataset: str, category: str) -> dict:
        """One category's calibration state — the SAME value /api/moderation reports, so
        two screens of the app can never disagree about a threshold."""
        c = (self.track_scores(dataset).get("categories") or {}).get(category)
        if not c:
            raise ValueError(f"unknown moderation category {category!r}")
        return {"category": category, "calibration": c["calibration"],
                "spec_calibration": c["spec_calibration"],
                "enforcement_ready": c["enforcement_ready"]}

    def stored_moderation(self, dataset: str, limit: int = 0) -> dict:
        """Aggregate the INDEX-TIME flags b-engine wrote onto each ids record.

        Answers a different question from the live scan — "what did we record when we
        indexed it" — and needs no model at all, so it is cheap and survives a threshold
        change. Never merged with current-scan numbers (they carry different labels).
        """
        snap = self.snapshot(dataset)
        counts: dict[str, dict] = {}
        flagged: list[dict] = []
        for rec in snap.ids:
            for f in rec.get("flags") or []:
                cat, tier = f.get("category"), f.get("tier")
                if not cat or tier not in ("violation", "review"):
                    continue
                counts.setdefault(cat, {"violation": 0, "review": 0})[tier] += 1
                if limit:
                    flagged.append({**f, "image_id": rec["image_id"], "path": rec["path"],
                                    "dataset": rec.get("dataset") or dataset,
                                    "dataset_slug": rec.get("dataset") or dataset})
        flagged.sort(key=lambda f: (f["category"], f["tier"] != "violation",
                                    -float(f.get("p") or 0), f["image_id"]))
        out = {"dataset": dataset, "indexed": len(snap.ids), "counts": counts,
               "source": "stored", "labels": {}, "threshold": None,
               "calibration": {c: "unfitted" for c in counts},
               "enforcement_ready": {c: False for c in counts}}
        if limit:
            out["flagged"] = flagged[: limit * max(1, 2 * len(counts))]
        return out

    def moderation(self, dataset: str, limit: int = 0) -> dict:
        """Per-dataset summary: "N violations, M for review" per category (ADR-14)."""
        snap = self.snapshot(dataset)
        t = self.track_scores(dataset)
        out = {
            "dataset": dataset, "indexed": t["n"], "counts": t.get("counts", {}),
            "labels": t.get("labels", {}), "threshold": t.get("floor"),
            "source": "current-scan",  # live: today's detectors over today's embeddings
            # per-category, and false until per-TIER tau is fitted (ADR-14 item 3)
            "calibration": {n: c["calibration"] for n, c in (t.get("categories") or {}).items()},
            "enforcement_ready": {n: c["enforcement_ready"]
                                  for n, c in (t.get("categories") or {}).items()},
        }
        if limit and t.get("categories"):
            flagged = []
            for name, c in t["categories"].items():
                for tier, p, mask in (("violation", c["violation_p"], c["is_violation"]),
                                      ("review", c["review_p"], c["is_review"])):
                    idx = np.argsort(-np.where(mask, p, -1.0))[:limit]
                    for i in idx:
                        if not mask[i]:
                            break
                        r = snap.ids[int(i)]
                        flagged.append({"category": name, "tier": tier,
                                        "p": round(float(p[i]), 4),
                                        "image_id": r["image_id"], "path": r["path"],
                                        "dataset": r.get("dataset") or dataset,
                                        "dataset_slug": r.get("dataset") or dataset})
            flagged.sort(key=lambda f: (f["category"], f["tier"] != "violation", -f["p"], f["image_id"]))
            out["flagged"] = flagged
        return out

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
        segs = split_terms(query)
        if len(segs) < 2:
            return []
        out: list[tuple[str, list[int]]] = []
        i = 0
        while i < len(segs):
            text, quoted = segs[i]
            if quoted:  # a quoted span is ONE term, verbatim — never merged or split
                idxs = [table.index[n] for n in expand(text) if n in table.index]
                if not idxs:
                    return []
                out.append((text, idxs[:MAX_TAGS]))
                i += 1
                continue
            # unquoted: greedy longest match first, because the vocabulary has multi-word
            # tags ("ocean wave", "night sky") — "ocean wave sunset" is TWO terms, not three.
            span_max = 0
            while span_max < NGRAM and i + span_max < len(segs) and not segs[i + span_max][1]:
                span_max += 1
            for span in range(span_max, 0, -1):
                term = " ".join(t for t, _ in segs[i : i + span])
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
        za, zb = table.z_logistic  # the dataset layer is ALWAYS a z-score, never a cosine
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
                P[:, j] = _sigmoid(za * ((cos[:, j] - mu) / max(sd, 1e-6)) + zb)
                eff[j] = float(_sigmoid(za * K_STD + zb))
                present[j] = float(_sigmoid(za * SPECTRUM_K + zb))
        return P, eff, cos, present

    # -- the query -------------------------------------------------
    def search_one(self, query: str, dataset: str, k: int = 50, strict: bool = False,
                   text: str = "auto", track: str | None = None,
                   tier: str | None = None) -> dict:
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
            return {"hits": [], "gated": False, "indexed": 0, "collapsed": 0, "best_p_text": 0.0,
                    "text_tower": "skipped", "load_ms": 0.0, "terms": [],
                    "calibration": "unfitted"}
        table = self.tags(man)
        # FAIL-OPEN LAW: an UNFITTED threshold may never veto a result. Without a CAL-SET
        # fit for this model_sha the engine returns the honest ranking and says so; it does
        # not pretend that "nothing matched" when what it really means is "I cannot judge".
        fitted = bool(table and table.fitted)
        floor = TAU if fitted else -1.0
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
            if feature == "cos":
                p_text = _sigmoid(a * s + b)
            else:
                za, zb = table.z_logistic if table else (Z_A, Z_B)
                p_text = _sigmoid(za * z + zb)
            if table is not None:
                cands = sorted(dict(cands + self._near_candidates(table, query, q)).items())[:MAX_TAGS]

        p_tag = np.zeros(n, np.float32)
        col = np.full(n, -1, np.int32)  # winning tag column per row, -1 = no tag path
        gated = False  # a CALIBRATED tag cleared its effective tau somewhere
        tag_cos = None
        matched = np.zeros(n, np.int32)  # ALL-SOME-ANY: how many query terms this row has
        mean_p = np.zeros(n, np.float32)  # mean p over the MATCHED terms (secondary sort)
        n_concepts = len(groups)
        if cands:
            idx = [i for i, _ in cands]
            P, eff, tag_cos, present = self._tag_scores(snap, table, idx, man)
            col = P.argmax(1).astype(np.int32)
            p_tag = P[np.arange(n), col].astype(np.float32)
            calib = np.asarray([table.calibrated(i) for i in idx])
            if calib.any():
                gated = bool((P[:, calib] >= eff[calib]).any())
            if n_concepts:  # rank by ALL first, then m/n descending, then ANY
                pos = {i: j for j, i in enumerate(idx)}
                term_hit = np.zeros((n, n_concepts), bool)
                term_p = np.zeros((n, n_concepts), np.float32)
                term_via: list[dict] = []
                for gi, (term, tag_idxs) in enumerate(groups):
                    cols = [pos[i] for i in tag_idxs if i in pos]
                    if not cols:
                        term_via.append({})
                        continue
                    best = P[:, cols].argmax(1)
                    term_p[:, gi] = P[np.arange(n), [cols[b] for b in best]]
                    term_hit[:, gi] = (P[:, cols] >= present[cols]).any(1)
                    names = {table.names[idx[c]] for c in cols}
                    term_via.append({} if term in names else {"via": sorted(names)[:4]})
                matched = term_hit.sum(1).astype(np.int32)
                mean_p = np.where(matched > 0,
                                  (term_p * term_hit).sum(1) / np.maximum(matched, 1), 0.0)
            if strict:  # --strict: calibrated tags become a hard AND
                keep = np.ones(n, bool)
                for j, i in enumerate(idx):
                    if table.calibrated(i):
                        keep &= P[:, j] >= eff[j]
                p_tag = np.where(keep, p_tag, 0.0).astype(np.float32)
                p_text = np.where(keep, p_text, 0.0).astype(np.float32)

        p = np.maximum(p_tag, p_text)
        # Moderation costs a text-tower batch, so a plain query NEVER triggers it: scores
        # are used only when explicitly filtered on, or when this generation is already in
        # cache (an earlier /api/moderation). ADR-5's no-tower named-tag path stays intact.
        tr = {"categories": {}, "floor": 1.0}
        if track:
            tr = self.track_scores(dataset)
        else:
            cached = self._tracks.get(dataset)
            if cached and cached[0] == self._manifest_stamp(dataset):
                tr = cached[1] or tr
        if track:  # moderation filter: only images this category flags, ranked by the query
            c = (tr.get("categories") or {}).get(track)
            if c is None:
                raise ValueError(
                    f"unknown moderation category {track!r}; known: {sorted(tr.get('categories') or {})}")
            keep = c["is_violation"] | c["is_review"] if tier is None else (
                c["is_violation"] if tier == "violation" else c["is_review"])
            p = np.where(keep, p, 0.0).astype(np.float32)
        # ALL-SOME-ANY ordering (VISION-ADDENDA): tag-count is the PRIMARY key, probability
        # the tiebreak. p is in [0,1), so `matched + p` orders both in one pass.
        rank = matched.astype(np.float32) + (mean_p if n_concepts else p)
        top = np.argpartition(-rank, min(k, n) - 1)[:k] if k < n else np.arange(n)
        hits = []
        for i in top:
            i = int(i)
            if p[i] < floor:
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
                terms_hit = [groups[g][0] for g in range(n_concepts) if term_hit[i, g]]
                terms_missed = [groups[g][0] for g in range(n_concepts) if not term_hit[i, g]]
                payload = {"matched": terms_hit, "missed": terms_missed,
                           "m": int(matched[i]), "n": n_concepts,
                           "mean_p": round(float(mean_p[i]), 4)}
                via = {groups[g][0]: term_via[g]["via"] for g in range(n_concepts)
                       if term_hit[i, g] and term_via[g].get("via")}
                if via:  # the user's word matched through a hypernym/synonym: show which
                    payload["via"] = via
                why.update(terms=payload, tags_matched=int(matched[i]), tags_total=n_concepts,
                           spectrum="all" if matched[i] == n_concepts else
                                    ("some" if matched[i] > 1 else "any"))
            flags = self.flags_for(tr, i) if tr.get("categories") else []
            # The user's metadata is exactly `rec["meta"]` (VISION-ADDENDA 12:33Z). It is
            # HOISTED, not swept: sweeping unknown fields buried it one level deep AND
            # leaked the indexer's internal bookkeeping (mtime/size) into a public field.
            extra = rec.get("meta") or {}
            hits.append(
                {
                    "image_id": rec["image_id"],
                    "path": rec["path"],
                    "dataset": rec.get("dataset") or dataset,  # B18: provenance NEVER null
                    "dataset_slug": rec.get("dataset") or dataset,
                    "score": round(score, 6),
                    "p": round(float(p[i]), 4),
                    "w": rec.get("w"),
                    "h": rec.get("h"),
                    # B18(b): a moved/deleted file is TOMBSTONED, never a silent 404 later
                    "exists": os.path.exists(rec["path"]),
                    "why": why,
                    # LIVE scan: categories firing now, [{category, p, tier}] (ADR-14)
                    **({"flags": flags} if flags else {}),
                    # STORED: what the indexer recorded when this row was written. Kept
                    # under its own name so a UI can never show two numbers as one.
                    **({"flags_stored": rec["flags"]} if rec.get("flags") else {}),
                    # generic index-time metadata (account ids, dates, ...) verbatim
                    **({"meta": extra} if extra else {}),
                }
            )
        hits.sort(key=_order)  # ALL-SOME-ANY, then p, then image id (B18e determinism)
        hits, collapsed = dedupe(hits)
        return {"hits": hits[:k], "gated": gated, "indexed": n, "collapsed": collapsed,
                "best_p_text": float(p_text.max()),
                "text_tower": ("loaded" if load_ms else "warm") if use_text else "skipped",
                "load_ms": load_ms, "terms": [t for t, _ in groups],
                "calibration": "fitted" if fitted else "unfitted"}

    def search(self, query: str, dataset: str | None = None, k: int = 50, strict: bool = False,
               text: str = "auto", track: str | None = None, tier: str | None = None) -> dict:
        """Search one dataset, or every dataset on disk when ``dataset`` is None."""
        t0 = time.perf_counter()
        self.last_query = time.time()
        names = [dataset] if dataset else list_datasets(self.home)
        hits: list[dict] = []
        indexed, gated, best_text, load_ms, collapsed = 0, False, 0.0, 0.0, 0
        tower, terms, calibration = "skipped", [], "unfitted"
        for name in names:
            r = self.search_one(query, name, k=k, strict=strict, text=text, track=track, tier=tier)
            hits += r["hits"]
            indexed += r["indexed"]
            gated |= r["gated"]
            best_text = max(best_text, r["best_p_text"])
            load_ms += r["load_ms"]
            collapsed += r["collapsed"]
            tower = r["text_tower"] if tower == "skipped" else tower
            terms = terms or r["terms"]
            calibration = "fitted" if r["calibration"] == "fitted" else calibration
        hits.sort(key=_order)
        hits, extra_collapsed = dedupe(hits)
        collapsed += extra_collapsed
        return {
            "query": query,
            "tookMs": round((time.perf_counter() - t0) * 1000, 2),
            # the one-time tower load is INSIDE tookMs and labeled here, so a warm-latency
            # budget (B3) can exclude it instead of silently absorbing a B13-shaped cost.
            "text_tower": tower,
            "text_tower_load_ms": round(load_ms, 2),
            # honest latency attribution: a query that paid a model load says so, so a
            # warm-latency budget (B3) never silently absorbs a B13-shaped cost.
            "served_by": ("tag-table" if tower == "skipped" else
                          "warm-tower" if not load_ms else "cold-load"),
            # how the probabilities were produced: a CAL-SET fit, or this build's measured
            # defaults. Never let a caller mistake one for the other.
            "calibration": calibration,
            "coverage": {"indexed": indexed, "total": total_expected(names, indexed, self.home)},
            # duplicate index rows for one content-addressed id: collapsed here, COUNTED so
            # the indexer bug behind them stays visible (never silently swallowed)
            **({"collapsed_duplicates": collapsed} if collapsed else {}),
            "datasets": names,
            # parsed multi-term query (quoted spans stay ONE element); absent when n < 2
            **({"terms": terms} if terms else {}),
            "hits": hits[:k],
            # An honest no-match requires a real threshold to have rejected everything:
            # no calibrated tag cleared its tau AND nothing cleared the free-text floor.
            # Unfitted calibration can only ever report "no rows", never "no match".
            "no_match": (not hits) if calibration != "fitted"
                        else (not hits and not gated and best_text < TAU),
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
