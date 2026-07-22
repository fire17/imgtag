"""Build + install the calibrated tag table for a shipping model — ADR-3 layer 1.

    uv run python -m imgtag.bench.calibrate pecore-s16-384

Embeds the tag prompts with the model's text tower, embeds the HELD-OUT CAL-SET
(data/coco-train2k, COCO-annotated) with its vision tower, scores cos per (image, tag),
fits a per-tag Platt sigmoid + max-F1 tau on the tags that have ground truth there (the 80
COCO categories — LVIS-tier tags keep tau=null, an honest "not yet calibrated"), then
writes tags.f32 + tags.json under `~/.imgtag/models/<model_sha>/` (ADR-3 one-owner path).

model_sha = sha256(shipping vision artifact bytes). Coordinated with b-daemon: whatever
its loaded ModelBackend reports as `.model_sha` must equal this, else the tag path refuses
loudly (ADR-3 calib_model_sha mismatch). --root overrides the install dir for testing.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np

from ..core import tags as G
from . import candidates as C
from . import textsets as T

CAL_DIR = os.path.join(C.DATA, "coco-train2k")
CAL_JSON = os.path.join(CAL_DIR, "instances_cal.json")


def model_sha(cand: C.Candidate, prec: str = "fp32") -> str:
    with open(cand.path(prec), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def cal_labels(names: list[str]) -> tuple[list[str], np.ndarray]:
    """CAL-SET image paths + [N, T] binary label matrix over the tag `names`."""
    inst = json.load(open(CAL_JSON))
    imgs = sorted(inst["images"], key=lambda i: i["id"])
    on_disk = {i["id"]: os.path.join(CAL_DIR, i["file_name"]) for i in imgs
               if os.path.exists(os.path.join(CAL_DIR, i["file_name"]))}
    imgs = [i for i in imgs if i["id"] in on_disk]
    idx = {i["id"]: n for n, i in enumerate(imgs)}
    col = {G._norm(n): j for j, n in enumerate(names)}
    cats = {c["id"]: G._norm(c["name"]) for c in inst["categories"]}
    labels = np.zeros((len(imgs), len(names)), np.float32)
    for a in inst["annotations"]:
        if a["image_id"] in idx and cats[a["category_id"]] in col:
            labels[idx[a["image_id"]], col[cats[a["category_id"]]]] = 1.0
    return [on_disk[i["id"]] for i in imgs], labels


def build(cand_id: str, root: str | None = None, cal_intra: int = 4) -> dict:
    c = C.CANDIDATES[cand_id]
    if not c.path("fp32"):
        raise SystemExit(f"{cand_id}: no fp32 vision artifact")
    tpath = c.text.get("fp32") or c.text.get("int8")
    if not tpath or not os.path.exists(tpath):
        raise SystemExit(f"{cand_id}: no text tower")

    table = G.build_tag_table()
    prompts = [T.prompt(n) for n in table.names]
    table.prompt_ensemble_sha = G.prompt_ensemble_sha(prompts)
    print(f"tags: {len(table)} ({sum(t == G.CALIBRATED for t in table.tier)} calibrated-tier)")

    # 1. tag embeddings (text tower)
    toks = T.tokenize(prompts, c.tok, c.ctx)
    ts = C.session(tpath, 4)
    tag_emb = C.embed_texts(ts, toks, c.out_idx)
    table.emb = tag_emb
    print(f"tag embeddings: {tag_emb.shape}")

    # 2. CAL-SET image embeddings (vision fp32)
    paths, labels = cal_labels(table.names)
    print(f"CAL-SET: {len(paths)} images, {int(labels.sum())} positive labels over "
          f"{int((labels.sum(0) > 0).sum())} tags with ground truth")
    vs = C.session(c.path("fp32"), cal_intra)
    t0 = time.perf_counter()
    img_emb = C.embed_images(vs, paths, c, batch=2,
                             progress=lambda i, n: print(f"  {i}/{n} "
                                                         f"({i/max(1e-9,time.perf_counter()-t0):.1f} img/s)")
                             if i % 400 == 0 else None)

    # 3. score + Platt/tau fit
    scores = (img_emb @ tag_emb.T).astype(np.float32)  # [N, T] cosine (both L2-normed)
    G.calibrate(table, scores, labels)
    n_fit = sum(t is not None for t in table.tau)
    print(f"calibrated {n_fit} tags (Platt + max-F1 tau)")

    # 4. install
    sha = model_sha(c)
    d = G.save(table, sha, root=root)
    meta = json.load(open(os.path.join(d, "tags.json")))
    meta["calib"] = {"cal_set": "coco-train2k", "n_cal_images": len(paths),
                     "n_calibrated_tags": n_fit, "cal_model_sha": sha,
                     "shipping_precision": "fp32-vision"}
    with open(os.path.join(d, "tags.json"), "w") as f:
        json.dump(meta, f)
    print(f"installed -> {d}  (model_sha {sha[:16]}…)")
    return {"dir": d, "model_sha": sha, "n_tags": len(table), "n_calibrated": n_fit,
            "prompt_ensemble_sha": table.prompt_ensemble_sha}


if __name__ == "__main__":
    cand = sys.argv[1] if len(sys.argv) > 1 else "pecore-s16-384"
    root = sys.argv[2] if len(sys.argv) > 2 else None
    print(json.dumps(build(cand, root=root), indent=1))
