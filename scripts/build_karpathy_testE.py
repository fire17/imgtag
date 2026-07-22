#!/usr/bin/env python3
"""CORPUS-E "karpathy-test" list builder — the canonical Karpathy COCO TEST split (5,000
val2014 images), for b-bench B17 "within-2pts-of-card" retrieval (our val2017 overlaps the
Karpathy test set only 593/5000, so the card number isn't reproducible without these).

Reads data/karpathy/dataset_coco.json (on disk). Extracts images with split=="test"
(all 5,000 live in val2014), deterministic by cocoid. Writes only a dest<TAB>url list;
fetching is fetch_karpathy_test.sh's job (curl, certifi law / ORACLE §4). No network here.
"""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT = os.path.join(ROOT, "data", "karpathy", "dataset_coco.json")
OUT = os.path.join(ROOT, "data", "karpathy-test")
os.makedirs(OUT, exist_ok=True)

d = json.load(open(SPLIT))
test = sorted((i for i in d["images"] if i["split"] == "test"), key=lambda i: i["cocoid"])
assert all(i["filepath"] == "val2014" for i in test), "unexpected non-val2014 test image"

with open(os.path.join(OUT, ".fetch.tsv"), "w") as fh:
    for i in test:
        fn = i["filename"]                       # COCO_val2014_000000391895.jpg
        fh.write(f"{OUT}/{fn}\thttp://images.cocodataset.org/val2014/{fn}\n")

# tiny id manifest so the split membership is auditable without re-parsing 144MB
json.dump({"corpus": "CORPUS-E", "name": "karpathy-test",
           "split": "karpathy test", "source": "data/karpathy/dataset_coco.json",
           "count": len(test), "cocoids": [i["cocoid"] for i in test]},
          open(os.path.join(OUT, "karpathy_test_ids.json"), "w"))

print(f"CORPUS-E: {len(test)} test-split val2014 images -> {OUT}/.fetch.tsv "
      f"(+ karpathy_test_ids.json)")
