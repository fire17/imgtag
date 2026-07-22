# Corpus status — CORPUS-B / B12 / D / CAL-SET

> Owner: corpus-builder lane · snapshot 2026-07-22 14:05 local · implements BUDGETS.md §Corpora.
> Everything lands under `data/` (gitignored — verified: `git check-ignore data/…` hits).
> **Unsplash images are LOCAL RESEARCH ONLY — never redistributed, never committed.**

## CORPUS-E karpathy-test — 5,000 val2014 (Karpathy TEST split, B17) — ✅ DONE 5000/5000 (18:50)

> 18:50 checkpoint: **5,000/5,000** fetched (798MB), gap of 5 closed by idempotent re-run.
> Verify sweep n=200 ok=0 bad, width median 640, mean 0.16MB. **NOT committed — publish freeze
> active**; this update + the entry below are held in the working tree, to be committed once the
> team-lead lifts the freeze.

Location `data/karpathy-test/`. b-bench needs the canonical Karpathy test split for B17's
"within-2pts-of-card" retrieval clause — our CORPUS-A (val2017) overlaps it only 593/5000, so
the published card number isn't reproducible without these. Extracted from
`data/karpathy/dataset_coco.json` (`split=="test"` → exactly **5,000 images, all val2014**),
deterministic by cocoid. URL `http://images.cocodataset.org/val2014/<filename>` (206-probe
confirmed). Scripts: `scripts/build_karpathy_testE.py` (writes `.fetch.tsv` +
`karpathy_test_ids.json` — the auditable cocoid list) · `scripts/fetch_karpathy_test.sh`
(POLITE PAR=8, idempotent/resumable, curl/certifi).

```
sample n=20: 20/20 ok, width median 511, mean 0.13MB → 5,000 ≈ 650MB. launched nohup 17:51.
```
Held-out note: Karpathy test is a val2014 slice; disjoint from CAL-SET (train2017). Overlaps
CORPUS-A only where a val2017 image shares a cocoid with the test set (593) — that overlap is
expected and is exactly why E exists.

## ✅ CHECKPOINT 2026-07-22 17:10 — ALL FOUR CORPORA COMPLETE

| Tag | Name | Location | Target | Final | Size | Sample verify |
|---|---|---|---|---|---|---|
| CORPUS-B | photo10k | `data/unsplash-b/` | 10,000 @ w=3200 | **9,998/10,000** ✅ | 18GB | n=200 ok, w=3200, MP 4.9–18.2 |
| CORPUS-B12 | fullres300 | `data/unsplash-fullres/` | 300 native ≥12MP | **300/300** ✅ | 1.1GB | n=300 ok, **MP 12.0–74.6, all ≥3000px** |
| CORPUS-D | poison | `data/poison/` | ~120 hostile | **170 / 18 classes** ✅ | 28MB | every class probe-verified hostile |
| CAL-SET | cocotrain2k | `data/coco-train2k/` | ~2,000 train2017 | **2,000/2,000** ✅ | 341M | n=200 ok, 80/80 cats |

**b-bench index f1d3a1ca decode-timeouts (2FxgoQ0qcW4.jpg, 2GAvdXQ6Xp8.jpg) — WRITTEN OFF as
NOT corpus defects.** Post-window decode check (17:14): both decode cleanly and fast —
`OK RGB (3200,5689) 184ms` and `OK RGB (3200,2133) 42ms`. The timeouts were transient index-side
(decode-timeout threshold tripped under concurrent 3-way fetch + index load), not truncated/corrupt
files. No re-fetch; both remain valid members of CORPUS-B. If b-bench still times out on them,
raise its per-file decode timeout rather than re-fetching.

Gaps closed by idempotent re-run (17:09): CAL 1966→2000, B 9934→9998. The **2 remaining B files
are unrecoverable** — their `photo_image_url` in `photos.tsv000` is malformed (e.g.
`images.unsplash.com_TheBeach.jpg`, no scheme/path → `curl (6) could not resolve host`); dead at
source, not a fetch bug. 9,998/10,000 = 99.98%, accepted. Disk at checkpoint: 46GB free (b-bench
index job is consuming disk concurrently; still above the 25GB guard).

---

### Original launch table (2026-07-22 14:05)

| Tag | Name | Location | Target | Status @14:05 | Size (proj.) | ETA |
|---|---|---|---|---|---|---|
| CORPUS-B | photo10k | `data/unsplash-b/` | 10,000 @ w=3200 | 🟡 fetching, 1,935 | 3.5GB now → **~18GB** | ~14:57 |
| CORPUS-B12 | fullres300 | `data/unsplash-fullres/` | 300 native ≥12MP | 🟡 fetching, 23 | 99MB now → **~1.0GB** | ~15:05 |
| CORPUS-D | poison | `data/poison/` | ~120 hostile | ✅ **DONE — 170 files, 18 classes** | 28MB | — |
| CAL-SET | cocotrain2k | `data/coco-train2k/` | ~2,000 train2017 | 🟡 fetching, 510 | 105MB now → **~420MB** | ~14:25 |

Scripts (all idempotent, resumable, ≤12 connections, log to `data-fetch-corpus.log`):
`scripts/fetch_photo10k.sh` · `scripts/fetch_fullres300.sh` · `scripts/build_poison.py` ·
`scripts/build_coco_train2k.py` + `scripts/fetch_coco_train2k.sh` · `scripts/verify_corpus.py`
(shared verifier: dims/MP/bytes + size projection, `--min-width` gate).

## CORPUS-B photo10k — sample verification (n=50, PASSED)

```
files=50 ok=50 bad=0
width  min/median/max = 3200/3200/3200
MP     min/median/max = 2.5/6.8/15.4      (median 6.8MP — above the ≈5MP spec)
bytes  mean=1.53MB → projected 15.3GB ; live avg after 1,935 files = 1.8MB → ~18GB
```
Source `data/unsplash/photos.tsv000` rows 1–10,000, URL = `photo_image_url?w=3200&q=85`.
Throughput 50 imgs/12.6s solo (≈240/min); ≈155/min with the other two fetches running.

## CORPUS-B12 fullres300 — probe verification (n=5, PASSED)

```
files=5 ok=5 bad=0 (--min-width 3000)
width  min/median/max = 4636/5139/5757
MP     min/median/max = 14.2/18.6/19.0    ⇒ native originals confirmed ≥12MP
bytes  mean=3.21MB → 300 imgs ≈ 1.0GB
```
Raw `photo_image_url` with **no width param** returns the native original.
Candidate window = TSV rows **10,001–11,000** (zero overlap with photo10k), metadata-prefiltered
to `photo_width≥3000 AND w*h≥12MP` (349 of the first 400 rows pass width, 307 pass 12MP — the
window is widened to 1,000 rows so 360 candidates exist for a 300 keep), then every downloaded
file is dimension-verified ≥3000px and surplus/failed candidates are deleted.

## CORPUS-D poison — DONE, 170 files, 18 classes, 28MB

Built offline from copies of `data/coco/val2017` + Pillow + macOS `sips`. Manifest:
`data/poison/manifest.json` (`{corpus,name,purpose,source,count,classes,files[{file,class,bytes,note}]}`).
Class list follows the **widened** BUDGETS.md §Corpora spec (classes 13–17 added after that edit).

| class | n | live probe result (verified) |
|---|---|---|
| truncated_jpeg | 12 | `OSError: image file is truncated` |
| corrupt_jpeg | 12 | 7/12 raise; 5 decode to garbage (**silent-corruption** case, deliberate) |
| zero_byte | 12 | `UnidentifiedImageError` |
| wrong_extension | 12 | PNG bytes as `.jpg` (10) + JPEG bytes as `.png` (2) — decode OK, sniffing required |
| cmyk_jpeg | 10 | `mode=CMYK` |
| png16 | 10 | `mode=I;16` |
| tiny_1px | 10 | 1×1 |
| huge_dims | 10 | 20000×20000 → `DecompressionBombError` under default Pillow guard |
| decompression_bomb | 6 | 17500×17500 = 306MP (>178MP guard) |
| exif_rotated | 16 | **all 8 orientations present, 2 each** (verified via `getexif()[274]`) |
| heic | 10 | real `ftypheic` container (`sips -s format heic`), Pillow cannot decode |
| progressive_eof | 10 | progressive JPEG cut at 55% → EOF mid-scan |
| filename_hostile | 12 | spaces · unicode/emoji · leading dash · tab · quotes · no extension · UPPER.JPG · longest-possible name |
| palette_alpha | 6 | mode-P PNG with transparency index |
| animated_gif | 6 | 4-frame GIF |
| symlink_valid | 4 | symlink to a real JPEG outside the dir |
| symlink_loop | 8 | `OSError [Errno 62] Too many levels of symbolic links` |
| perm_denied | 4 | `PermissionError` (mode 000) |

Deviations, recorded honestly:
- **300-char filename → 251-char.** APFS `NAME_MAX` is 255 **bytes**; `open()` returns
  `ENAMETOOLONG` at 300. The longest name the filesystem can hold is used and the generator
  documents why. A real 300-char path is untestable on this machine.
- Huge-dims files are ~700KB on disk but ~1.2GB decoded — that is the point (B8/B21 RAM bomb).
- `perm_denied` files are mode 000: `rm -rf data/poison` still works (dir is writable), but a
  `find -type f -exec` run as another user will hit them — intended.

## CAL-SET cocotrain2k — 2,000 images, annotations DONE, fetch running

```
categories=80/80 · images=2000 · annotations=15427 · min images-per-category=41
data/coco-train2k/instances_cal.json (9.2MB, COCO schema: info/licenses/images/annotations/categories)
sample fetch n=20: 20/20 ok, width median 640, mean 0.21MB → 2,000 ≈ 420MB
```
Deterministic, no RNG: per category take the 25 lowest image ids containing that category
(the mandated stratum), union → **1,195 unique** (low-id COCO images are multi-category, so the
80×25 slices overlap ~40%). A deterministic round-robin top-up then widens each category's slice
by one id at a time until the union hits 2,000. Held-out: train2017 is disjoint from CORPUS-A
(val2017) by construction — **never benched, calibration only (B7 τ)**.

## Operational notes

- Disk: 73GB free of 926GB at snapshot; total corpus footprint ≈19.5GB. Guard threshold 25GB —
  re-check `df -h` before adding any further corpus.
- All three fetches run concurrently under `nohup` (12/12/8 connections); per-file failures are
  retried 3× then skipped (`*.part` removed), so a re-run of any script repairs gaps — the
  scripts are the resume mechanism, no separate resume state.
- Re-verify at completion with:
  `python3 scripts/verify_corpus.py data/unsplash-b 200` ·
  `python3 scripts/verify_corpus.py data/unsplash-fullres 300 --min-width 3000` ·
  `python3 scripts/verify_corpus.py data/coco-train2k 200`
