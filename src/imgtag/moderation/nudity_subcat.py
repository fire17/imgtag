"""Sub-category labeling for the nudity track (VISION-ADDENDA 12:33Z + user 2026-07-22:
"no sub-category labeling"). OWNER: track-nudity. Research: research/track-nudity.md §10.

WHY A SECOND INSTRUMENT. The primary head (Marqo, nudity.py) is a BINARY NSFW/SFW
classifier — it produces one probability and CANNOT name a sub-category. Sub-category
labels ("exposed breast" vs "covered buttocks" vs "swimwear") need a detector with a
body-part vocabulary. NudeNet v3 320n (deepghs/nudenet_onnx, **Apache-2.0**, YOLOv8 @320,
12MB) is that vocabulary: 18 anatomical-region classes, each exposed/covered.

This head is OPT-IN and runs ONLY on images the primary head already flagged (or on demand)
— it is the explainer, not the gate. So it does not touch the 100-track scaling budget for
the common (unflagged) case: a normal image pays the Marqo forward and stops.

TIER MAPPING is ADR-14, and it is DATA (below), not folklore: the exposed genital/anus/
breast/buttocks classes are `violation`; covered-but-revealing classes (the swimwear/
lingerie surface) are `review`; face/feet/belly/armpit classes are context, `none`.

No new runtime dependency: onnxruntime + numpy + Pillow (ADR-7 intact). NMS is ~30 lines
of numpy — the shipped nms-yolov8.onnx is not needed (one fewer session, one fewer graph).
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from ..core.models import _session, file_sha256, find_artifact

ARTIFACT = "nudenet-320n.onnx"
SPEC = {"subdir": "moderation"}
INPUT = 320

#: NudeNet v3 class order — VERBATIM from the model's own README label table, NOT guessed.
CLASSES = (
    "FEMALE_GENITALIA_COVERED", "FACE_FEMALE", "BUTTOCKS_EXPOSED", "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED", "MALE_BREAST_EXPOSED", "ANUS_EXPOSED", "FEET_EXPOSED",
    "BELLY_COVERED", "FEET_COVERED", "ARMPITS_COVERED", "ARMPITS_EXPOSED", "FACE_MALE",
    "BELLY_EXPOSED", "MALE_GENITALIA_EXPOSED", "ANUS_COVERED", "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED",
)

#: ADR-14 tier per class, as data. violation = exposed sexual anatomy; review = the
#: covered-but-revealing surface that IS the swimwear/lingerie signal; none = neutral
#: context (a face or a foot is not a moderation event).
CLASS_TIER = {
    "FEMALE_GENITALIA_EXPOSED": "violation", "MALE_GENITALIA_EXPOSED": "violation",
    "FEMALE_BREAST_EXPOSED": "violation", "BUTTOCKS_EXPOSED": "violation",
    "ANUS_EXPOSED": "violation",
    "FEMALE_GENITALIA_COVERED": "review", "FEMALE_BREAST_COVERED": "review",
    "BUTTOCKS_COVERED": "review", "ANUS_COVERED": "review", "BELLY_EXPOSED": "review",
    "MALE_BREAST_EXPOSED": "review", "ARMPITS_EXPOSED": "review",
}


def _letterbox(im: Image.Image, size: int = INPUT) -> np.ndarray:
    """PIL -> f32 [3,size,size] in [0,1], aspect-preserving pad (YOLO convention).
    Padding grey 114 matches the training preprocess; wrong padding shifts every box."""
    im = im.convert("RGB")
    w, h = im.size
    s = size / max(w, h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    r = im.resize((nw, nh), Image.Resampling.BILINEAR)
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[:nh, :nw] = np.asarray(r, np.uint8)
    return np.ascontiguousarray(canvas.transpose(2, 0, 1).astype(np.float32) / 255.0)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> list[int]:
    """Plain greedy NMS. boxes xyxy [n,4]. ~15 lines vs a whole second ONNX graph."""
    x1, y1, x2, y2 = boxes.T
    area = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep = []
    while len(order):
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = (xx2 - xx1).clip(0) * (yy2 - yy1).clip(0)
        ovr = inter / (area[i] + area[order[1:]] - inter + 1e-9)
        order = order[1:][ovr <= iou]
    return keep


class NudeNetSubcategoryHead:
    """Body-part detector → per-image sub-category labels + ADR-14 tier.

    ``detect(im) -> list[{class, tier, p, box}]`` (raw), and ``label(im) -> dict`` for the
    per-image moderation payload: the max-tier reached + the sub-categories that drove it.
    """

    category = "nudity"
    model_id = "nudenet-320n"

    def __init__(self, path, profile: dict | None = None, score_threshold: float = 0.25,
                 iou_threshold: float = 0.45):
        profile = profile or {}
        self.path = path
        self.model_sha = file_sha256(path)
        self._s = _session(path, profile.get("intra_op", 2))
        self._in = self._s.get_inputs()[0].name
        self.score_threshold = float(profile.get("nudity_subcat_score") or score_threshold)
        self.iou_threshold = float(iou_threshold)

    def detect(self, im: Image.Image) -> list[dict]:
        w, h = im.convert("RGB").size
        x = _letterbox(im)[None]
        out = self._s.run(None, {self._in: x})[0][0]        # [22, N]
        preds = out.T                                        # [N, 22] = 4 box + 18 cls
        cls = preds[:, 4:]
        conf = cls.max(1)
        keep0 = conf >= self.score_threshold
        preds, cls, conf = preds[keep0], cls[keep0], conf[keep0]
        if not len(preds):
            return []
        cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
        cid = cls.argmax(1)
        out_dets, scale = [], max(w, h) / INPUT
        for c in np.unique(cid):
            m = cid == c
            for i in _nms(boxes[m], conf[m], self.iou_threshold):
                b = boxes[m][i] * scale
                name = CLASSES[int(c)]
                out_dets.append({"class": name, "tier": CLASS_TIER.get(name, "none"),
                                 "p": round(float(conf[m][i]), 4),
                                 "box": [round(float(v), 1) for v in b]})
        return out_dets

    def label(self, im: Image.Image) -> dict:
        """One per-image sub-category verdict: the highest tier any body part reached, and
        the distinct sub-categories that reached a flagging tier (the explanation surface)."""
        dets = self.detect(im)
        order = {"violation": 2, "review": 1, "none": 0}
        tier = "none"
        subs: dict[str, float] = {}
        for d in dets:
            if order[d["tier"]] > order[tier]:
                tier = d["tier"]
            if d["tier"] != "none":
                subs[d["class"]] = max(subs.get(d["class"], 0.0), d["p"])
        return {"category": "nudity", "tier": tier, "model_id": self.model_id,
                "subcategories": [{"class": k, "p": v} for k, v in sorted(subs.items(), key=lambda kv: -kv[1])],
                "detections": dets}


def load_subcategory_head(profile: dict | None = None) -> NudeNetSubcategoryHead | None:
    path = find_artifact(SPEC, ARTIFACT)
    return None if path is None else NudeNetSubcategoryHead(path, profile)
