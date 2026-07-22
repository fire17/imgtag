#!/usr/bin/env python3
"""CAL-SET "cocotrain2k" sampler — deterministic stratified 2k split of COCO train2017.

HELD-OUT calibration split (B7 τ fitting) — never benched, never overlaps CORPUS-A
(val2017). Selection is fully deterministic (no RNG): for each of the 80 categories take
the 25 LOWEST image ids that contain >=1 instance of it, union them, dedupe.

Outputs:
  data/coco-train2k/.fetch.tsv        dest<TAB>url list for fetch_coco_train2k.sh
  data/coco-train2k/instances_cal.json filtered annotations, same schema as COCO
Reads data/coco/annotations/instances_train2017.json (already on disk, 470MB).
No network here; fetching is the shell script's job (curl, per ORACLE §4).
"""
import json, os, sys, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANN = os.path.join(ROOT, "data", "coco", "annotations", "instances_train2017.json")
OUT = os.path.join(ROOT, "data", "coco-train2k")
PER_CAT = int(os.environ.get("PER_CAT", "25"))
os.makedirs(OUT, exist_ok=True)

print("loading", ANN, "(470MB, ~40s)...", flush=True)
d = json.load(open(ANN))

by_cat = collections.defaultdict(set)
for a in d["annotations"]:
    by_cat[a["category_id"]].add(a["image_id"])

# 25/cat alone yields ~1195 unique images (low-id COCO images are multi-category, so the
# 80x25 slices overlap heavily). Deterministic top-up: keep widening every category's
# slice by one, round-robin in category order, until the union reaches TARGET.
TARGET = int(os.environ.get("TARGET", "2000"))
ordered = {c: sorted(by_cat[c]) for c in sorted(by_cat)}
keep, k = set(), 0
while k < max(len(v) for v in ordered.values()):
    for c, ids in ordered.items():
        if k < PER_CAT:                       # phase 1: the mandated 25 per category
            keep.update(ids[:PER_CAT])
        elif k < len(ids):
            keep.add(ids[k])
        if k >= PER_CAT and len(keep) >= TARGET:
            break
    k = PER_CAT if k < PER_CAT else k + 1
    if len(keep) >= TARGET:
        break

imgs = [im for im in d["images"] if im["id"] in keep]
anns = [a for a in d["annotations"] if a["image_id"] in keep]
cal = {
    "info": dict(d.get("info", {}), description="CAL-SET cocotrain2k — held-out calibration split"),
    "licenses": d.get("licenses", []),
    "images": sorted(imgs, key=lambda i: i["id"]),
    "annotations": sorted(anns, key=lambda a: a["id"]),
    "categories": d["categories"],
}
json.dump(cal, open(os.path.join(OUT, "instances_cal.json"), "w"))

with open(os.path.join(OUT, ".fetch.tsv"), "w") as fh:
    for im in cal["images"]:
        fh.write(f"{OUT}/{im['file_name']}\thttp://images.cocodataset.org/train2017/{im['file_name']}\n")

print(f"categories={len(d['categories'])} per_cat={PER_CAT} images={len(imgs)} annotations={len(anns)}")
print(f"wrote {OUT}/instances_cal.json ({os.path.getsize(os.path.join(OUT,'instances_cal.json'))/1e6:.1f}MB)"
      f" and .fetch.tsv ({len(imgs)} urls)")
