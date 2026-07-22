# IMGTAG — dataset research (bench + quick + demo)

Lane: research-datasets · date verified: **2026-07-22** · all URL checks done live this day
with `curl -sIL` (HEAD) or `curl -sL -r 0-0` (1-byte GET range, used where HEAD is refused).
Sizes = exact `Content-Length` / `Content-Range` total bytes off the wire. No guessed numbers.

---

## TL;DR — the three picks

| Role | Dataset | Download | Wire size | Images | Ground truth |
|---|---|---|---|---|---|
| **PRIMARY bench** | **COCO val2017** + `annotations_trainval2017` **+ LVIS v1 val (restricted to val2017)** | 3 URLs, all ✅ verified | 778 MiB + 241 MiB + 61 MiB = **1.06 GiB** | 5,000 | 80 cats / 12 supercats (`vehicle` = 8) + 1,203 LVIS cats w/ WordNet synsets on 4,809 of those same images |
| **QUICK iterate** | **COCO-val2017-500** (scripted deterministic subset of primary) | 0 extra bytes (script) | ~80 MiB on disk | 500 | same annotations, filtered |
| **DEMO gallery** | **Unsplash Lite** (metadata TSV → pull N pretty images from CDN) | 1 URL ✅ verified | 305 MiB TSV zip + ~N×250 KB images | 25,000 available (pull 2–5k) | keywords.tsv (weak labels), not a bench |

Killer property of the primary: **one image download, two label granularities, two hierarchies.**
COCO gives coarse 80-class multi-label + a hand-made supercategory tree (perfect for
"car" precision and "vehicle" hypernym recall). LVIS v1 val overlaps 4,809 of the exact
same 5,000 val2017 files with 1,203 fine-grained classes, each carrying a **WordNet
synset id** → free unlimited hypernym chains (`sports_car.n.01` → `car` → `motor_vehicle`
→ `vehicle`) without hand-authoring a taxonomy.

---

## Verification log (empirical, today)

| URL | Method | Result |
|---|---|---|
| `http://images.cocodataset.org/zips/val2017.zip` | HEAD | 200 · **815,585,330 B** (777.8 MiB) · application/zip |
| `http://images.cocodataset.org/annotations/annotations_trainval2017.zip` | HEAD | 200 · **252,907,541 B** (241.2 MiB) |
| `http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip` | HEAD | 200 · 860,725,834 B (821 MiB) — optional |
| `http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip` | HEAD | 200 · 1,148,688,564 B (1.07 GiB) — optional |
| `http://images.cocodataset.org/zips/test2017.zip` | HEAD | 200 · 6,646,970,404 B — unlabeled, skip |
| `https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip` | HEAD + **full download + parse** | 200 · **64,026,968 B** (61.1 MiB) → `lvis_v1_val.json` **201,235,232 B** |
| `https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip` | range GET (HEAD 403 — signed-URL method mismatch) | 206 · **137,414,764 B** (131 MiB) |
| `https://data.caltech.edu/records/nyy15-4j048/files/256_ObjectCategories.tar` | range GET | 206 · **1,183,006,720 B** (1.10 GiB) |
| `http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar` | HEAD | 301 → `https://thor.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar` → 200 · **1,999,639,040 B** (1.86 GiB) |
| `https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip` | range GET | 206 · **1,115,419,746 B** (1.04 GiB) |
| `https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip` | range GET | 206 · 2,340,801 B |
| `https://huggingface.co/datasets/nlphuji/flickr30k/resolve/main/flickr30k-images.zip` | range GET | 206 · **4,390,240,817 B** (4.09 GiB) · no auth needed |
| `https://unsplash.com/data/lite/latest` | HEAD | 302 → `https://unsplash-datasets.s3.amazonaws.com/lite/latest/unsplash-research-dataset-lite-latest.zip` → 200 · **320,024,071 B** (305 MiB) |
| `https://images.unsplash.com/photo-<id>?w=1080` (CDN sample) | range GET | 206 · image/jpeg · 826,694 B — dynamic resize works |
| `https://open-images-dataset.s3.amazonaws.com/tar/validation.tar.gz` | range GET | 206 · **12,844,530,798 B** (11.96 GiB) — too big for primary |
| `https://open-images-dataset.s3.amazonaws.com/validation/<id>.jpg` (sample) | range GET | 206 · image/jpeg — **per-image pull works, no auth** |
| `https://storage.googleapis.com/openimages/2018_04/bbox_labels_600_hierarchy.json` | range GET | 206 · **86,291 B** — the 600-class hierarchy |
| `https://storage.googleapis.com/openimages/v5/validation-annotations-human-imagelabels-boxable.csv` | range GET | 206 · 10,649,275 B |
| `https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv` | range GET | 206 · 12,064 B |
| `https://raw.githubusercontent.com/openimages/dataset/master/downloader.py` | range GET | 206 · 4,244 B |
| `https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip` | range GET | 206 · 36,745,453 B (Karpathy splits, coco/f8k/f30k captions) |
| `https://huggingface.co/datasets/imagenet-1k/resolve/main/data/val_images.tar.gz` | range GET | **401 — gated, needs HF login** ❌ |
| `https://dl.fbaipublicfiles.com/LVIS/lvis_v1_minival.json.zip` | range GET | **403 — does not exist at that path** ❌ |

Two facts proved by actually parsing files, not by reading docs:

1. **COCO supercategory tree** (from `panoptic_coco_categories.json`, parsed live):
   80 thing categories in 12 supercategories —
   `vehicle → bicycle, car, motorcycle, airplane, bus, train, truck, boat` (8),
   `animal → bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe` (10),
   `food → banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake` (10),
   plus person / outdoor(5) / accessory(5) / sports(10) / kitchen(7) / furniture(6) /
   electronic(6) / appliance(5) / indoor(7). **Exactly the hypernym test the brief wants.**
2. **LVIS v1 val composition** (grep over the downloaded 201 MB json):
   **4,809 distinct `val2017/*.jpg`** + 15,000 distinct `train2017/*.jpg` = 19,809 images,
   and **1,203 `"synset"` entries** (one per category). → Restricting LVIS to val2017
   costs **0 extra image bytes** and still yields 4.8k images × fine-grained labels.
   (Using full LVIS val would drag in train2017 = 19.3 GB. Don't.)

---

## Per-candidate evaluation

### 1. COCO val2017 — ✅ **PRIMARY**
- **URLs:** `zips/val2017.zip` (777.8 MiB), `annotations/annotations_trainval2017.zip` (241.2 MiB).
- **Images:** 5,000 · natural photos, multi-object, cluttered — realistic for false-positive measurement.
- **Labels:** `instances_val2017.json` — per-image instance list over 80 categories
  (COCO json: `images[]`, `annotations[{image_id, category_id, bbox, area, iscrowd}]`,
  `categories[{id, name, supercategory}]`). Also `captions_val2017.json` (5 captions/img,
  useful as a natural-language relevance proxy) and `person_keypoints_val2017.json`.
- **Hierarchy:** built-in 2-level `supercategory` (12 nodes) — verified above.
- **License:** annotations CC BY 4.0 (COCO Consortium); images remain under their Flickr
  terms — COCO does not own image copyright. **Local research/benchmark use: fine.
  Redistribution of images: don't.**
- **Auth:** none. Plain HTTP host (`images.cocodataset.org`), no HTTPS on the zip host
  redirect chain — it's fine, but checksum after download.
- **Verdict:** best objective ground truth per byte. `car` → exact positive set; every other
  image is a true negative → FP rate is directly computable. `vehicle` → union of 8 children.

### 2. LVIS v1 val (restricted to val2017) — ✅ **PRIMARY add-on**
- **URL:** `https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip` (61.1 MiB → 201 MB json).
- **Labels:** 1,203 fine-grained categories, federated annotation (each image labeled
  exhaustively only for a subset of categories — **read `not_exhaustive_category_ids` and
  `neg_category_ids` per image; scoring FPs without them will over-count false positives**).
- **Hierarchy:** every category carries `synset` (e.g. `car_(automobile).n.01` style
  WordNet ids) + `def` + `frequency` (f/c/r). WordNet hypernym closure via `nltk.corpus.wordnet`
  gives arbitrary-depth "vehicle" style queries, free.
- **License:** LVIS annotations CC BY 4.0 (FAIR); images = COCO images (same terms).
- **Caveat:** only 4,809/5,000 val2017 images are in LVIS val. Filter, don't join blindly.
- **Verdict:** the fine-grained + hypernym-depth layer. Take it.

### 3. Open Images V7 — ⚠️ **hierarchy YES, images NO (too big)**
- Validation split = 41,620 images; `tar/validation.tar.gz` = **11.96 GiB** → over budget.
  But per-image S3 pull works unauthenticated → a scripted 3–5k subset is possible if a
  second bench set is ever wanted.
- **Take the hierarchy file anyway** (86 KB): `bbox_labels_600_hierarchy.json` is a real
  multi-level tree (Vehicle → Land vehicle → Car → Limousine …) — a ready-made
  hypernym oracle that beats COCO's 2-level tree for depth testing, and maps onto COCO
  names by string match for extra query expansion.
- **License:** annotations CC BY 4.0 (Google); images CC BY 2.0 (per-image verification advised).
- **Verdict:** grab hierarchy + class descriptions (98 KB total). Defer the images.

### 4. Caltech-101 — ✅ **optional standalone quick set**
- **URL:** `https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip` — 131 MiB,
  ~9,144 images, 101 object categories + background, ~300×200 px, one object per image.
- **License:** **CC BY 4.0** (verified from the CaltechDATA record API) — cleanest license here.
- **Labels:** directory-per-class = single-label ground truth, trivially parsed. No hierarchy.
- **Gotcha:** HEAD returns 403 (signed S3 URL rejects HEAD) — use `curl -L` GET, works fine.
- **Verdict:** good cheap sanity set (precision@1 by folder name), but single-object,
  low-res, and no multi-label → weaker than a COCO subset for FP measurement.
  Take only if a *fully independent* dataset is wanted for overfit checks.

### 5. Caltech-256 — ⚠️ 1.10 GiB, 30,607 images, 257 classes, CC BY 4.0. Same shape as 101,
  8× the bytes, still no hierarchy. Skip unless class-count breadth matters.

### 6. PASCAL VOC2012 — ⚠️ **usable, redundant**
- `https://thor.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar` (1.86 GiB;
  the old `host.robots.ox.ac.uk` URL 301s here — use the thor URL directly).
- 20 classes, XML per-image annotations, informal 4-group hierarchy (person/animal/vehicle/indoor)
  — `vehicle` = aeroplane, bicycle, boat, bus, car, motorbike, train.
- License: "flickr terms", non-commercial research framing, no explicit CC.
- **Verdict:** COCO strictly dominates it (more classes, cleaner json, official supercats). Skip.

### 7. Flickr8k / Flickr30k — ❌ **wrong tool**
- Flickr8k images 1.04 GiB (GitHub mirror ✅), Flickr30k images 4.09 GiB (HF, ungated ✅).
- Captions only — no object labels, no categories → cannot compute a clean FP rate
  (absence from a caption ≠ absence from the image). License murky (Flickr-terms, mirrors).
- **Verdict:** skip for bench. COCO captions already cover the caption angle.

### 8. Unsplash Lite — ✅ **DEMO gallery**
- `https://unsplash.com/data/lite/latest` → 305 MiB zip of TSVs (photos, keywords,
  collections, conversions, colors). **No images in the zip** — `photos.tsv000` has
  `photo_image_url` (dynamic CDN URL: append `?w=1080&q=80` to size it) + `photo_description`
  + `ai_description`; `keywords.tsv000` has photo↔keyword pairs with AI/human confidence.
- **License:** Unsplash Dataset terms — Lite is usable **commercially and non-commercially**,
  but **images may not be redistributed** (local demo = fine; shipping a mirror = not).
- **Verdict:** the pretty/diverse demo set. Pull 2,000–5,000 at w=1080 (≈0.25 MB each →
  0.5–1.2 GB) with a bounded-parallel fetcher. Keywords give a weak-label smoke test for free.

### 9. ImageNet val — ❌ **blocked**
- HF `imagenet-1k` returns **401** (gated, login+terms). image-net.org requires registration.
  Kaggle mirrors require login. **Do not use** — fails the no-auth requirement.

### 10. CIFAR — ❌ 32×32, per brief. Skip.

---

## How to score with these (the reason for the picks)

- **Precision / FP:** query `"car"` → predicted set P. Ground truth G = images whose
  `instances_val2017` annotations contain `category.name == "car"`. FP = |P \ G|,
  and every val2017 image not in G is a genuine negative (COCO's 80 classes are
  exhaustively annotated → absence is real absence, unlike caption data).
- **Hypernym recall:** `"vehicle"` → G = union over the 8 children. Report recall +
  child-level breakdown (does the embedding find `truck` as readily as `car`?).
- **Fine-grained + deep hierarchy:** LVIS synsets + WordNet closure, restricted to the
  4,809 shared images. Honour `neg_category_ids` / `not_exhaustive_category_ids` when
  counting FPs, else the federated design inflates them.
- **Cross-taxonomy sanity:** Open Images' 600-class tree as a second, independently
  authored hierarchy — if hypernym recall ranks the same under both trees, the metric
  isn't a taxonomy artifact.
- **Query list:** derive from `categories[]` (80 names + 12 supercats) — no hand-authoring.

---

## Ready-to-run download script

Paste as-is. Idempotent (`curl -C -` resumes, skips finished files). Default target
`~/Creations/ImgTag/data`. Primary tier ≈ **1.06 GiB** down, ≈ **1.9 GB** on disk.

```bash
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
```

Disk budget: COCO 778 MiB zip + ~780 MiB extracted, LVIS 61 MiB + 201 MB json,
quick500 ~80 MiB (or ~0 with symlinks), Unsplash 305 MiB TSV + ~0.5 GB images.
**Total ≈ 2.7 GB** with the demo tier, **≈ 1.9 GB** bench-only. Delete the zips after
extraction to reclaim ~1.1 GB.

## Notes / risks

- `images.cocodataset.org` serves over plain HTTP (HTTPS also answers). Zips are unsigned —
  verify by count after extraction (`ls coco/val2017 | wc -l` must be 5000).
- LVIS federated annotation: **must** use `neg_category_ids` / `not_exhaustive_category_ids`
  or the false-positive numbers are wrong (too pessimistic). COCO's 80 classes have no such caveat.
- Unsplash: images may not be redistributed — keep the demo set local / re-fetchable by script.
- Open Images images deferred (11.96 GiB); only the 98 KB taxonomy files are pulled.
- ImageNet val and any Kaggle-hosted mirror are **auth-gated** → excluded by requirement.
