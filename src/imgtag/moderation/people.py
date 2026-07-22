"""People/face COUNTING track — VISION-ADDENDA 2026-07-22 13:28Z (verbatim):

    "i want track to be able to categorize images if they have 1 person in them (even if
     its their back with no face), more then one person, 1 visible face - and more than
     one visible face (even at angles for any)"

This is a CONTENT track, not an enforcement track: its tier is ``match``, never
``violation``/``review``, so it never enters the ADR-14 moderation counts.

WHY THIS TRACK CANNOT RIDE THE EMBEDDING ALONE (measured, research/track-people.md §2).
TRACKS.md T2 makes embedding-space the default and forces a dedicated model to be
justified by measurement. It was: on 4,773 crowd-free COCO val2017 images, a trained
logistic probe over the SAME pecore-s16-384 embedding the index already computed scores
AP 0.969 for "is there a person" but collapses to F1 0.497 on "is there EXACTLY one
person" — the actual question the user asked. Presence is in the embedding; CARDINALITY
is not. Faces are worse: the zero-shot prompt ensemble lands AP 0.277 against a 34.8%
prevalence, i.e. anti-correlated with the thing it is meant to find.

THE INSTRUMENT (TRACKS.md T2 rung 3, budgeted). YuNet (OpenCV-Zoo face_detection_yunet,
anchor-free, strides 8/16/32) at 640x640 = 0.688 GFLOPs/img MEASURED off its own ONNX
graph — see research/track-people.md §5 for the B25 arithmetic. It is decoded here in
numpy because cv2 is NOT a runtime dependency of this project and may not become one
(ADR-7); onnxruntime + numpy + Pillow are all this file uses.

THE HYBRID, and why persons cost ZERO extra FLOPs. Every detected face is positive
evidence of a person, so the face count is a hard LOWER BOUND on the person count. The
residual question — "are there people the face detector cannot see?" (the user's explicit
back-view case) — is exactly the PRESENCE question the embedding answers well. So
n_persons = max(n_faces, cascade over the embedding). One detector pass serves both
counts; no second dedicated model is loaded, and B25 stays satisfied.

Per-image sidecar values (TRACKS.md T1 — RAW is stored, categories are DERIVED at read):
``n_persons``, ``n_persons_conf``, ``n_faces``, ``n_faces_conf``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..core.tags import platt_apply

CATEGORY = "people"
DATA = Path(__file__).resolve().parent.parent / "data" / "moderation"
MODELS = Path(__file__).resolve().parents[3] / "models" / "moderation"
ARTIFACT = "face-yunet-640.onnx"

#: The four DERIVED categories, in the order the user named them.
DERIVED = ("one-person", "multi-person", "one-face", "multi-face")

INPUT_SIZE = 640           # the ONNX graph is STATIC at 640x640 — not a tunable
STRIDES = (8, 16, 32)
TAU_FACE = 0.6             # calibrated in research/track-people.md §3
NMS_IOU = 0.3
#: Person-cascade operating points. Fitted on a COCO val2017 held-out split; overridden
#: by the fitted head file when one exists for this backend.
TAU_PERSON_1 = 0.5
TAU_PERSON_2 = 0.5


# ── preprocessing ─────────────────────────────────────────────────────────────
def preprocess(im, size: int = INPUT_SIZE) -> tuple[np.ndarray, float]:
    """PIL RGB -> ([1,3,size,size] float32 BGR 0-255, scale).

    Two choices that are measured, not folklore (research/track-people.md §3):
      * **BGR**, because YuNet was trained through OpenCV, whose images are BGR. Feeding
        RGB measurably loses faces (COCO 000000301135: 3 detections BGR vs 1 RGB).
      * **Letterbox**, not squash: the graph is square and fixed, and squashing a 640x426
        photo distorts every face by 1.5x horizontally. Letterboxing preserves aspect and
        measured strictly better on the crowd images (7/7 vs 6/7 on 000000171190).
    No mean/std normalization: YuNet consumes raw 0-255, as OpenCV's blobFromImage
    default (scalefactor=1.0, no mean subtraction) hands it.
    """
    from PIL import Image

    w, h = im.size
    scale = min(size / w, size / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(im.resize((nw, nh), Image.BILINEAR), (0, 0))
    a = np.asarray(canvas, np.float32)[:, :, ::-1]           # RGB -> BGR
    return np.ascontiguousarray(a.transpose(2, 0, 1)[None]), float(scale)


# ── YuNet decode (numpy reimplementation of the OpenCV post-process) ──────────
def decode(outs: dict[str, np.ndarray], size: int = INPUT_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """Raw YuNet head outputs -> (boxes[N,4] xywh in letterboxed pixels, scores[N]).

    Anchor-free, one prior per feature-map cell. For stride s the cell grid is
    (size/s)^2 in ROW-MAJOR order, and the box regression is centre-offset + log-size:
        cx = (col + bbox[0]) * s        w = exp(bbox[2]) * s
        cy = (row + bbox[1]) * s        h = exp(bbox[3]) * s
    The confidence is the GEOMETRIC MEAN of the classification and objectness heads,
    sqrt(cls * obj) — YuNet trains them separately and multiplies at inference; the
    square root keeps the result on the same scale as either head alone.
    """
    boxes, scores = [], []
    for s in STRIDES:
        cls = np.clip(outs[f"cls_{s}"][0, :, 0], 0.0, 1.0)
        obj = np.clip(outs[f"obj_{s}"][0, :, 0], 0.0, 1.0)
        bb = outs[f"bbox_{s}"][0]
        g = size // s
        rows, cols = np.divmod(np.arange(g * g), g)          # row-major cell order
        cx = (cols + bb[:, 0]) * s
        cy = (rows + bb[:, 1]) * s
        w = np.exp(bb[:, 2]) * s
        h = np.exp(bb[:, 3]) * s
        boxes.append(np.stack([cx - w / 2, cy - h / 2, w, h], 1))
        scores.append(np.sqrt(cls * obj))
    return np.concatenate(boxes).astype(np.float32), np.concatenate(scores).astype(np.float32)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = NMS_IOU) -> list[int]:
    """Greedy IoU NMS. Counting is the product here, so a merge error is a COUNT error —
    this is deliberately the plain, auditable version rather than a clever one."""
    if len(boxes) == 0:
        return []
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = x1 + boxes[:, 2], y1 + boxes[:, 3]
    areas = np.maximum(boxes[:, 2], 0) * np.maximum(boxes[:, 3], 0)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thr]
    return keep


def count_confidence(kept: np.ndarray, below: float) -> float:
    """Confidence that the count is EXACTLY len(kept).

    Two ways a count can be wrong, so both are priced: the weakest detection we accepted
    might be spurious (s_k), and the strongest one we rejected might be real (below).
        conf = s_k * (1 - below),  with s_0 := 1.0 for the empty case.
    This is why a clean zero-person image reports high confidence rather than none.
    """
    weakest = float(kept.min()) if len(kept) else 1.0
    return float(np.clip(weakest * (1.0 - below), 0.0, 1.0))


# ── the person cascade over the shared embedding (free) ───────────────────────
@dataclass
class PersonCascade:
    """P(>=1 person) and P(>=2 persons) as two logistic probes over ONE embedding.

    Ordinal-by-cascade rather than 3-way softmax: the two questions have very different
    difficulty (AP 0.969 vs 0.813) and separate thresholds let each be set on its own
    measured operating point instead of averaging the easy one into the hard one.
    """

    model_id: str
    dim: int
    w1: np.ndarray
    b1: float
    platt1: list
    w2: np.ndarray
    b2: float
    platt2: list
    tau1: float = TAU_PERSON_1
    tau2: float = TAU_PERSON_2
    metrics: dict = field(default_factory=dict)

    def probs(self, emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        e = np.asarray(emb, np.float32)
        if e.ndim != 2 or e.shape[1] != self.dim:
            raise ValueError(f"people cascade expects [N,{self.dim}], got {e.shape}")
        p1 = platt_apply(e @ self.w1 + self.b1, self.platt1)
        p2 = platt_apply(e @ self.w2 + self.b2, self.platt2)
        return np.asarray(p1, np.float64), np.minimum(np.asarray(p2, np.float64), p1)

    def to_json(self) -> dict:
        return {"model_id": self.model_id, "dim": self.dim,
                "w1": [float(x) for x in self.w1], "b1": float(self.b1), "platt1": list(self.platt1),
                "w2": [float(x) for x in self.w2], "b2": float(self.b2), "platt2": list(self.platt2),
                "tau1": float(self.tau1), "tau2": float(self.tau2), "metrics": self.metrics}

    @classmethod
    def from_json(cls, d: dict) -> PersonCascade:
        return cls(model_id=d["model_id"], dim=d["dim"],
                   w1=np.asarray(d["w1"], np.float32), b1=float(d["b1"]), platt1=d["platt1"],
                   w2=np.asarray(d["w2"], np.float32), b2=float(d["b2"]), platt2=d["platt2"],
                   tau1=float(d.get("tau1", TAU_PERSON_1)), tau2=float(d.get("tau2", TAU_PERSON_2)),
                   metrics=d.get("metrics", {}))


def cascade_path(model_id: str, root: Path | None = None) -> Path:
    return (root or DATA) / f"people-{model_id}.json"


def load_cascade(model_id: str, root: Path | None = None) -> PersonCascade | None:
    p = cascade_path(model_id, root)
    return PersonCascade.from_json(json.loads(p.read_text())) if p.is_file() else None


# ── derivation (TRACKS.md T1: categories are DERIVED from raw, at read) ───────
def derive(n_persons: int, n_faces: int) -> dict[str, bool]:
    """The four user-facing categories. Pure function of the raw counts, no model.

    Kept deliberately separate from scoring so a policy change ("multi means >=3")
    re-reads the sidecar instead of re-running the detector.
    """
    return {"one-person": n_persons == 1, "multi-person": n_persons >= 2,
            "one-face": n_faces == 1, "multi-face": n_faces >= 2}


# ── the head ──────────────────────────────────────────────────────────────────
class PeopleHead:
    """Dispatcher-facing head (imgtag.moderation.load_heads contract).

    ``score(embeddings, images, ids) -> list[list[dict]]`` — one LIST per image. First
    element is always the RAW multi-column record (``category="people"``, a ``cols`` dict
    in ``col_roles`` order → ``people.f32 [N,4]``, tier ``none``); then zero or more
    ``match`` chips, one per satisfied derived category. Every image gets the raw record,
    so T1's "every track scores every image" holds even for empty frames.
    """

    category = CATEGORY
    wants_images = True        # pixels required: this track answers from optics

    #: Multi-column sidecar schema (b-engine's [N,C] write path). INERT until the engine's
    #: multi-col consumer lands: it reads this once per job, writes it verbatim into
    #: ``tracks/people.json`` as ``col_roles``, and canonicalizes each record's ``cols``
    #: dict into this order (a missing role → NaN, an honest "not scored"). The head is the
    #: single authority for its own column schema — never repeated per record, never in the
    #: shared moderation.json. Order here IS the on-disk column order.
    col_roles = ["n_persons", "n_faces", "n_persons_conf", "n_faces_conf"]

    def __init__(self, session, cascade: PersonCascade | None = None,
                 tau_face: float = TAU_FACE, model_id: str = "yunet-640"):
        self.sess = session
        self.cascade = cascade
        self.tau_face = float(tau_face)
        self.model_id = model_id
        self._in = session.get_inputs()[0].name
        self._out = [o.name for o in session.get_outputs()]

    @property
    def spec(self) -> dict:
        """Versioned scoring params (b-engine folds this into the header's ``spec_sha``,
        so a stale sidecar is refused without touching a shared file — rule-7 safe). Carries
        the derive() BAND EDGES as data (TRACKS.md T3: a future "crowd = 10+" is a new band
        over the SAME column, no re-score), so b-daemon's reader derives identically."""
        return {"version": 1, "scorer": "yunet-640 + pecore-cascade",
                "tau_face": round(float(self.tau_face), 4),
                "cascade_tau1": round(float(self.cascade_tau1), 4),
                "cascade_tau2": round(float(self.cascade_tau2), 4),
                "cascade_model": (self.cascade.model_id if self.cascade else None),
                "bands": {"one-person": [1, 1], "multi-person": [2, None],
                          "one-face": [1, 1], "multi-face": [2, None]}}

    # -- calibration honesty ---------------------------------------------------
    @property
    def calibrated(self) -> bool:
        """Face tau is fitted against a keypoint PROXY, never against face ground truth
        (COCO has none), so this stays False until real labels exist on the target host."""
        return False

    @property
    def enforcement_ready(self) -> bool:
        return False           # a counting track never authorizes an action

    # -- optics ----------------------------------------------------------------
    def faces(self, im) -> tuple[int, float]:
        """(n_faces, confidence) for one PIL image."""
        x, _ = preprocess(im)
        boxes, scores = decode(dict(zip(self._out, self.sess.run(self._out, {self._in: x}))))
        m = scores >= self.tau_face
        kept_idx = nms(boxes[m], scores[m]) if m.any() else []
        kept = scores[m][kept_idx] if len(kept_idx) else np.empty(0, np.float32)
        below = float(scores[~m].max()) if (~m).any() else 0.0
        return len(kept), count_confidence(kept, below)

    def _records(self, n_persons: int, n_faces: int, cp: float, cf: float,
                 extra: dict | None = None) -> list[dict]:
        """Per image: ONE raw multi-column record + one ``match`` chip per satisfied category.

        RAW (`category="people"`, tier ``none``): a dense ``cols`` dict in ``col_roles``
        order → the engine writes ``people.f32 [N,4]`` (b-daemon's single-column ask). This
        is the DURABLE, re-derivable source of truth; the four user categories are DERIVED
        at read via ``derive(n_persons, n_faces)`` (TRACKS.md T1), so a future "multi means
        >=3" is a free re-read. Tier ``none`` keeps a count out of the ADR-14 enforcement
        accounting — a person count is not a policy breach.

        CHIPS (tier ``match``, one per SATISFIED category — multi-label, both
        multi-person AND one-face can fire): the per-image flags the 14:16Z detail view
        ranks, and the numbers behind the ``content`` bucket rollup. Emitted only when true,
        so an unsatisfied category leaves no spurious column; membership stays a pure
        function of the stored counts.
        """
        prov = {"model_id": self.model_id, "calibrated": self.calibrated,
                "enforcement_ready": self.enforcement_ready, **(extra or {})}
        out = [{"category": CATEGORY,
                "cols": {"n_persons": float(n_persons), "n_faces": float(n_faces),
                         "n_persons_conf": round(float(cp), 4),
                         "n_faces_conf": round(float(cf), 4)},
                "tier": "none",
                "n_persons": int(n_persons), "n_faces": int(n_faces), **prov}]
        on = derive(n_persons, n_faces)
        conf = {"one-person": cp, "multi-person": cp, "one-face": cf, "multi-face": cf}
        out += [{"category": c, "p": round(float(conf[c]), 4), "tier": "match",
                 "n_persons": int(n_persons), "n_faces": int(n_faces), **prov}
                for c in DERIVED if on[c]]
        return out

    def score(self, embeddings, images=None, ids=None) -> list[list[dict]]:
        from PIL import Image

        recs = list(ids or [])
        n = len(recs) if recs else (len(embeddings) if embeddings is not None else 0)

        # persons: the free cascade over the embedding the index already computed
        if self.cascade is not None and embeddings is not None and len(embeddings):
            p1, p2 = self.cascade.probs(embeddings)
        else:
            p1 = p2 = np.zeros(n)

        out: list[list[dict]] = []
        for i in range(n):
            rec = recs[i] if i < len(recs) else {}
            # PIXEL GEOMETRY (the one thing to get right, per nudity's §7). The coordinator's
            # slab carries the BACKEND's geometry — squashed, 384². YuNet needs 640²
            # LETTERBOXED BGR from the ORIGINAL frame, so the slab is never the right input
            # and is deliberately ignored; the head re-opens from rec["path"] exactly like
            # nudity. Only a genuine PIL image (never handed by the current coordinator) is
            # taken directly — a numpy slab has a `.size` INT, so `isinstance` is the guard,
            # not `hasattr`.
            src = images[i] if (images is not None and i < len(images)) else None
            if not isinstance(src, Image.Image):
                src = None
            try:
                if src is None:
                    path = rec.get("path") if isinstance(rec, dict) else None
                    if path is None:
                        out.append(self._records(0, 0, 0.0, 0.0, {"no_pixels": True}))
                        continue
                    with Image.open(path) as im:
                        im.draft("RGB", (INPUT_SIZE, INPUT_SIZE))   # partial JPEG decode
                        n_faces, cf = self.faces(im.convert("RGB"))
                else:
                    n_faces, cf = self.faces(src.convert("RGB"))
            except Exception:                        # never break the index (ADR-14)
                out.append(self._records(0, 0, 0.0, 0.0, {"unreadable": True}))
                continue

            # cascade -> 0 / 1 / 2+, then the face count as a hard lower bound
            casc = 0 if p1[i] < self.cascade_tau1 else (1 if p2[i] < self.cascade_tau2 else 2)
            n_persons = max(n_faces, casc)
            cp = self._person_conf(p1[i], p2[i], n_persons, n_faces, cf)
            out.append(self._records(n_persons, n_faces, cp, cf))
        return out

    @property
    def cascade_tau1(self) -> float:
        return self.cascade.tau1 if self.cascade else TAU_PERSON_1

    @property
    def cascade_tau2(self) -> float:
        return self.cascade.tau2 if self.cascade else TAU_PERSON_2

    def _person_conf(self, p1: float, p2: float, n_persons: int, n_faces: int, cf: float) -> float:
        """Confidence in the person count, priced from whichever evidence decided it.

        When faces carried the count (n_persons == n_faces > 0) the optics are the
        evidence and their confidence is the honest one; otherwise the cascade decided
        and its own posterior is reported. No cascade at all -> faces only, and the
        caller sees ``persons_from_faces_only``.
        """
        if self.cascade is None:
            return cf
        if n_faces and n_persons == n_faces:
            return float(max(cf, p1 if n_persons >= 1 else 1.0 - p1))
        if n_persons == 0:
            return float(1.0 - p1)
        if n_persons == 1:
            return float(max(p1 - p2, 0.0))
        return float(p2)


def load_people_head(profile: dict | None = None) -> PeopleHead | None:
    """Dispatcher entry point. None when the YuNet artifact is absent — a missing track
    is simply not loaded and is reported by name, never a silent zero (seam law)."""
    import onnxruntime as ort

    prof = profile or {}
    art = Path(prof.get("people_model") or (MODELS / ARTIFACT))
    if not art.is_file():
        return None
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(prof.get("people_threads", 2))
    sess = ort.InferenceSession(str(art), so, providers=["CPUExecutionProvider"])

    from ..core.models import DEFAULT_BACKEND
    cascade = load_cascade(prof.get("backend") or DEFAULT_BACKEND)
    return PeopleHead(sess, cascade, float(prof.get("people_tau_face", TAU_FACE)))
