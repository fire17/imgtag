#!/usr/bin/env python3
"""Run YuNet over the whole COCO val2017 corpus and CACHE RAW DETECTIONS.

Raw scores are cached, never thresholded counts, so every confidence threshold in the
report is a re-read of this file rather than a re-run of the model (the same
store-raw-derive-at-read discipline TRACKS.md T1 imposes on the sidecars themselves).

    .venv/bin/python research/bench_scripts/people_yunet_sweep.py --out .scratch/yunet-coco.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from people_gt import build as build_gt  # noqa: E402

from imgtag.moderation.people import (  # noqa: E402
    decode, nms, preprocess,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models/moderation/face-yunet-640.onnx"))
    ap.add_argument("--out", type=Path, default=ROOT / ".scratch/yunet-coco.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--keep", type=float, default=0.2, help="cache every det above this")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    sess = ort.InferenceSession(args.model, so, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]

    gt = build_gt()
    items = sorted(gt.values(), key=lambda v: v["file_name"])
    if args.limit:
        items = items[: args.limit]
    root = ROOT / "data/coco/val2017"

    def load(v):
        with Image.open(root / v["file_name"]) as im:
            return v, preprocess(im.convert("RGB"))

    out: dict[str, dict] = {}
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as ex:
        for k, (v, (x, scale)) in enumerate(ex.map(load, items)):
            raw = sess.run(out_names, {in_name: x})
            boxes, scores = decode(dict(zip(out_names, raw)))
            m = scores >= args.keep
            boxes, scores = boxes[m], scores[m]
            keep = nms(boxes, scores) if len(boxes) else []
            out[v["file_name"]] = {
                "scores": [round(float(s), 4) for s in scores[keep]],
                "boxes": [[round(float(c), 1) for c in b] for b in boxes[keep]],
                "scale": scale,
            }
            if (k + 1) % 500 == 0:
                el = time.perf_counter() - t0
                print(f"{k + 1}/{len(items)}  {el:.0f}s  {(k + 1) / el:.1f} img/s", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out))
    el = time.perf_counter() - t0
    print(f"done {len(out)} imgs in {el:.0f}s ({len(out) / el:.1f} img/s) -> {args.out}")


if __name__ == "__main__":
    main()
