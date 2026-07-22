#!/usr/bin/env bash
# IMGTAG dataset fetch — verified 2026-07-22
set -euo pipefail
DATA="${IMGTAG_DATA:-$HOME/Creations/ImgTag/data}"
mkdir -p "$DATA"/{coco,lvis,openimages,unsplash,caltech101}
get(){ # url outfile
  [ -s "$2" ] && { echo "skip $(basename "$2")"; return; }
  echo "get  $(basename "$2")"; curl -fL -C - --retry 3 --retry-delay 2 -o "$2" "$1"
}

## ---------- TIER 1: PRIMARY BENCH (COCO val2017 + LVIS) ~1.06 GiB ----------
get http://images.cocodataset.org/zips/val2017.zip                              "$DATA/coco/val2017.zip"                      # 815,585,330 B
get http://images.cocodataset.org/annotations/annotations_trainval2017.zip      "$DATA/coco/annotations_trainval2017.zip"     # 252,907,541 B
get https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip                    "$DATA/lvis/lvis_v1_val.json.zip"             #  64,026,968 B

unzip -n -q "$DATA/coco/val2017.zip"                  -d "$DATA/coco"     # -> coco/val2017/*.jpg   (5,000)
unzip -n -q "$DATA/coco/annotations_trainval2017.zip" -d "$DATA/coco"     # -> coco/annotations/instances_val2017.json, captions_val2017.json
unzip -n -q "$DATA/lvis/lvis_v1_val.json.zip"         -d "$DATA/lvis"     # -> lvis/lvis_v1_val.json (201 MB)

## ---------- TIER 1b: hierarchies (tiny, ~100 KB) ----------
get https://storage.googleapis.com/openimages/2018_04/bbox_labels_600_hierarchy.json "$DATA/openimages/bbox_labels_600_hierarchy.json"  # 86,291 B
get https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv "$DATA/openimages/oidv7-class-descriptions-boxable.csv" # 12,064 B

## ---------- TIER 2: QUICK SET (no extra download; deterministic 500-image subset) ----------
python3 - "$DATA" <<'PY'
import json, os, shutil, sys
D = sys.argv[1]
ann = json.load(open(f"{D}/coco/annotations/instances_val2017.json"))
keep = sorted(im["id"] for im in ann["images"])[:500]          # deterministic: lowest 500 image ids
ks = set(keep)
by_id = {im["id"]: im for im in ann["images"]}
os.makedirs(f"{D}/quick500/images", exist_ok=True)
for i in keep:
    src = f"{D}/coco/val2017/{by_id[i]['file_name']}"
    dst = f"{D}/quick500/images/{by_id[i]['file_name']}"
    if not os.path.exists(dst): shutil.copy2(src, dst)         # swap copy2 -> os.symlink to save ~80 MB
sub = {"info": ann.get("info", {}), "licenses": ann.get("licenses", []),
       "images": [by_id[i] for i in keep],
       "annotations": [a for a in ann["annotations"] if a["image_id"] in ks],
       "categories": ann["categories"]}
json.dump(sub, open(f"{D}/quick500/instances_quick500.json", "w"))
print("quick500:", len(sub["images"]), "images,", len(sub["annotations"]), "annotations")
PY

## ---------- TIER 2b: LVIS restricted to the val2017 images we actually have ----------
python3 - "$DATA" <<'PY'
import json, sys
D = sys.argv[1]
lv = json.load(open(f"{D}/lvis/lvis_v1_val.json"))
imgs = [im for im in lv["images"] if "val2017" in (im.get("coco_url") or "")]
ids  = {im["id"] for im in imgs}
out = {"images": imgs,
       "annotations": [a for a in lv["annotations"] if a["image_id"] in ids],
       "categories": lv["categories"]}                          # each cat: name, synset, def, frequency
json.dump(out, open(f"{D}/lvis/lvis_val2017_only.json", "w"))
print("LVIS∩val2017:", len(imgs), "images,", len(out['annotations']), "anns,", len(out['categories']), "cats")
PY

## ---------- TIER 3: DEMO GALLERY (Unsplash Lite) ----------
get https://unsplash.com/data/lite/latest "$DATA/unsplash/unsplash-lite-latest.zip"   # 320,024,071 B
unzip -n -q "$DATA/unsplash/unsplash-lite-latest.zip" -d "$DATA/unsplash"             # -> photos.tsv000, keywords.tsv000, ...
N_DEMO="${N_DEMO:-2000}"                                                              # ~250 KB each -> ~0.5 GB
python3 - "$DATA" "$N_DEMO" <<'PY'
import csv, os, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
D, N = sys.argv[1], int(sys.argv[2])
os.makedirs(f"{D}/unsplash/images", exist_ok=True)
rows = []
with open(f"{D}/unsplash/photos.tsv000", encoding="utf-8") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        rows.append((r["photo_id"], r["photo_image_url"]))
        if len(rows) >= N: break
def fetch(t):
    pid, url = t
    dst = f"{D}/unsplash/images/{pid}.jpg"
    if os.path.exists(dst): return
    req = urllib.request.Request(f"{url}?w=1080&q=80", headers={"User-Agent": "imgtag-research/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r, open(dst, "wb") as o: o.write(r.read())
with ThreadPoolExecutor(max_workers=12) as ex: list(ex.map(fetch, rows))   # be polite: <=12 conns
print("demo images:", len(os.listdir(f"{D}/unsplash/images")))
PY

## ---------- OPTIONAL: independent single-label sanity set (Caltech-101, CC BY 4.0) ----------
# NOTE: HEAD 403s on this signed URL — GET works. 137,414,764 B
# get https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip "$DATA/caltech101/caltech-101.zip"
# unzip -n -q "$DATA/caltech101/caltech-101.zip" -d "$DATA/caltech101"

echo "done -> $DATA"
