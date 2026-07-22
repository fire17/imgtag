# track-people — the people / face counting track

> Mandate: VISION-ADDENDA 2026-07-22 **13:28Z** (verbatim) — *"i want track to be able to
> categorize images if they have 1 person in them (even if its their back with no face),
> more then one person, 1 visible face - and more than one visible face (even at angles
> for any)"*.
> Constitution: TRACKS.md (this track is the archetype of T1 — **store raw counts, derive
> categories at read**). Budget: ORACLE ADR-15 + B25. Escalation: ORACLE §7.
>
> **This is a CONTENT / counting track, not an enforcement track.** Its tier is never
> `violation`/`review`; the four categories are derived from stored counts. It authorizes
> no action (`enforcement_ready = False`, always).

---

> **Checkpoint 2026-07-22 ~14:30Z.** Track COMPLETE and committed (`6342eae`): module +
> shipped cascade + 16 tests (green) + 3 bench scripts + fetch script + this doc. All
> numbers below are measured first-party on COCO val2017 and verified end-to-end through
> the real CLI. Open items are coordination-only (not blocking): b-engine to register a
> non-counting `match` tier (14:16Z-sanctioned) so per-image category chips can flag;
> b-daemon to derive the four labels from the raw count sidecars at read. Darwin item
> D-people-1 (distil a person-count head to lift one-person past 0.561) is logged, not owed.
> Latency re-measure on the 🐧 target still pending (FLOPs is the trustworthy budget number).

---

## 0. TL;DR

- Counting is **provably not in a single global embedding** (measured §2): presence is
  (person≥1 probe AP 0.969) but **cardinality is not** (one-person F1 collapses to 0.497,
  face≥1 zero-shot AP 0.277 — *anti-correlated*). A dedicated instrument is justified by
  measurement, exactly as TRACKS.md T2 demands before spending a dedicated forward.
- The instrument: **YuNet** (OpenCV-Zoo `face_detection_yunet`, anchor-free), decoded in
  **numpy** (cv2 is not and may not become a runtime dep — ADR-7). **0.688 GFLOPs/img**
  measured off its own ONNX graph.
- **The hybrid, and why persons cost zero extra FLOPs.** Faces are a hard lower bound on
  persons; the residual "people with no visible face" (the user's back-view case) is the
  *presence* question the embedding answers well. So `n_persons = max(n_faces, cascade)` —
  one detector pass serves both counts, no second dedicated model.
- **B25: FITS.** Encoder 31.32 GFLOPs; dedicated spend nudity 7.83 + people 0.688 = 8.52 =
  **27.2%** of encoder, under the 30% cap (§5). People's marginal is **2.2%**.
- Measured on **COCO val2017** (exhaustive person boxes = real count truth; keypoints = a
  face-visibility **proxy**, its blind spot quantified, never hidden).

---

## 1. Why an embedding cannot count — the falsification (measured, first-party)

Corpus: **4,773 crowd-free COCO val2017 images** (227 `iscrowd` person images excluded —
one annotation of an unbounded crowd is not a count; they are tracked separately, never
folded into an accuracy number). 60/40 train/held-out, seed 20260722. Embeddings are the
`pecore-s16-384-fp32` vectors the index **already computed** — this whole section is a few
matmuls, zero new forwards. Harness: `research/bench_scripts/people_eval.py`.

| task | prevalence | zero-shot ensemble AP | zero-shot F1 | trained probe AP | trained probe F1 |
|---|---|---|---|---|---|
| person ≥ 1 | 51.0% | 0.894 | 0.782 | **0.969** | 0.894 |
| person ≥ 2 | 29.7% | 0.628 | 0.615 | 0.813 | 0.741 |
| face ≥ 1 | 34.8% | **0.277** | 0.515 | 0.887 | 0.853 |
| face ≥ 2 | 14.0% | 0.480 | 0.493 | 0.725 | 0.644 |

**The ordering is wrong in the way that matters.** The zero-shot face≥1 AP of **0.277**
against a 34.8% prevalence is *below chance ranking* — the global embedding cannot even
tell "a face is visible" apart from "a person is present but facing away". And the DERIVED
"exactly one person" category, built from the two best *trained* probes, lands at
**F1 0.497** — a coin flip on the user's headline category.

This is the on-corpus falsification of *"counting can just score the embeddings the index
already computed"*. It says nothing about weapons/drugs (nameable objects); it says
everything about **cardinality**, which a single pooled vector does not carry. (Consistent
with the vision addendum's own note: "person" ranked 800/2177 on a bus scene full of
people.)

---

## 2. The instrument — YuNet, decoded in numpy

**Artifact:** `models/moderation/face-yunet-640.onnx`, 232 KB. **License: Apache-2.0
(`libfacedetection`)** — permissive, commercial-safe. The `.onnx` and its `.sha256` sidecar
are gitignored (as nudity's is), so the integrity anchor lives here, committed:

```
curl -sSL -o models/moderation/face-yunet-640.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
# sha256 (verify on every fetch):
# 8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4
```
The graph is **static 640×640**, anchor-free, three stride levels (8/16/32 → 6400+1600+400
= 8400 priors), heads `cls_{s}`, `obj_{s}`, `bbox_{s}`, `kps_{s}`.

**No cv2.** `cv2.FaceDetectorYN` is the usual decoder; cv2 is not a runtime dependency of
this project and may not become one (ADR-7). The full post-process is reimplemented in
numpy (`imgtag.moderation.people.decode` / `nms`), validated against real images (§4):

- **Score = `sqrt(cls · obj)`** — YuNet trains classification and objectness separately and
  multiplies at inference; the square root keeps the fused score on the same scale.
- **Box = centre-offset + log-size** per prior: `cx=(col+bbox0)·s`, `cy=(row+bbox1)·s`,
  `w=exp(bbox2)·s`, `h=exp(bbox3)·s`, cells in row-major order.
- **Greedy IoU NMS** at 0.3 — deliberately the plain auditable version, because for a
  counting track a merge error *is* a count error.

**Preprocessing is measured, not folklore** (§4): **BGR** (YuNet was trained through
OpenCV) and **letterbox** (the graph is square; squashing a 640×426 photo distorts every
face 1.5× and measurably loses detections). Raw 0–255, no mean/std — OpenCV's
`blobFromImage` default.

**Pixel geometry (the one thing to get right).** Like nudity, the head **ignores the
coordinator's slab** (backend geometry — squashed 384²) and re-opens the original frame
from `rec["path"]` at 640² letterbox; `draft()` makes that a partial JPEG decode. A numpy
slab has a `.size` *int*, so the PIL guard is `isinstance(x, Image.Image)`, never
`hasattr(x,"size")` — the bug that first zeroed every column until it was caught end-to-end
(§6 field note).

---

## 3. Measured — count accuracy vs COCO ground truth

### 3a. Faces (YuNet) vs the keypoint proxy — threshold sweep

Face confidence τ swept over the cached raw detections (`research/bench_scripts/
people_yunet_sweep.py` → `.scratch/yunet-coco.json`; raw scores cached, thresholds re-read
— the same store-raw-derive discipline T1 imposes on the sidecars). Held-out, n=1,910:

| τ_face | exact-count acc | count MAE | face≥1 F1 | face≥2 F1 |
|---|---|---|---|---|
| 0.5 | 0.770 | 0.326 | 0.825 | 0.792 |
| **0.6** (shipped) | **0.814** | 0.232 | 0.841 | 0.832 |
| 0.7 | 0.830 | 0.209 | 0.838 | 0.858 |
| 0.8 | 0.825 | 0.242 | 0.809 | 0.782 |
| 0.9 | 0.733 | 0.446 | 0.562 | 0.401 |

τ=0.6 ships (recall-leaning, honoring "even at angles"); τ=0.7 is the exact-count optimum.
Overridable per install via `profile["people_tau_face"]`. **Exact face count is right 81%
of the time** — versus the embedding's 0.644 F1 ceiling on the *easier* binary face≥2.

### 3b. Persons — the cascade over the shared embedding (free)

Two logistic probes over the one embedding, P(≥1) and P(≥2), Platt-calibrated on the
train split. τ₁ recall-first (0.92 target: a missed person is the worst error), τ₂ chosen
to **maximize 3-way bucket accuracy (0 / 1 / 2+)** — the metric a single stored `n_persons`
actually serves; max-F1-on-≥2 over-fires multi and steals one-person, precision-leaning
does the reverse.

| probe | precision | recall | F1 | τ |
|---|---|---|---|---|
| person ≥ 1 | 0.857 | 0.920 | 0.888 | 0.382 |
| person ≥ 2 | 0.819 | 0.526 | 0.640 | 0.687 |

### 3c. The four DERIVED user categories, `n_persons = max(n_faces, cascade)`

Held-out, τ_face 0.6. **This is the deliverable table** — the four categories the user named:

| category | precision | recall | F1 |
|---|---|---|---|
| **one-person** | 0.456 | 0.730 | **0.561** |
| **multi-person** | 0.824 | 0.670 | **0.739** |
| **one-face** | 0.685 | 0.671 | **0.678** |
| **multi-face** | 0.814 | 0.850 | **0.832** |

Person 3-way bucket accuracy **0.742**; exact person count 0.603.

**Honest reading.** Faces are counted well (multi-face 0.832). Persons split cleanly at the
crowd boundary (multi-person 0.739). **one-person (0.561) is the hard cell** and this is a
real, measured limit, not a bug: "exactly one" is the conjunction "≥1 AND NOT ≥2", and the
≥2 signal from a global embedding has recall 0.53 — every true-2 it misses and every true-0
it over-fires lands in one-person. It still beats the pure-embedding derivation (0.497)
because YuNet's exact face count disambiguates many singles. Closing it further needs a
dedicated **person detector** (per-box), which does **not** fit the budget hole (§5) — so
it is logged as darwin item **D-people-1**: distill a person-count head, or re-budget.

### 3d. Back-view persons — the user's explicit hard case (measured)

309 held-out images with **≥1 person and zero visible face** (all keypoint-annotated, so
the proxy is not blind on them). **Faces alone score every one of them 0** — they are
invisible to any face detector.

| instrument | recall on faceless-person images |
|---|---|
| YuNet faces alone | **0.000** (0 / 309) |
| **hybrid (cascade recovers them)** | **0.841** (260 / 309) |

This is the hybrid earning its place on exactly the case the user called out — *"even if
its their back with no face"*. The confidence for these is the cascade's own posterior, and
the record carries `n_faces = 0` honestly.

---

## 4. Decode validation (why the numpy re-implementation is trusted)

Ran on real COCO images before landing (`.scratch/yunet_proto.py`), across preprocessing
variants. Representative:

| image | GT faces / persons | letterbox+BGR det | note |
|---|---|---|---|
| 000000171190 (crowd) | 7 / 10 | **7** | exact face match |
| 000000386912 | 1 / 1 | 1 (BGR) vs 2 (RGB) | BGR is correct; RGB spuriously splits |
| 000000456496 (back view) | 0 / 1 | **0** | correct silence — no face to find |
| 000000037777 (empty) | 0 / 0 | **0** | correct silence |

BGR and letterbox win consistently; both are the shipped defaults. The decode is exercised
by unit tests on synthetic head tensors (geometric-mean score, centre-offset/log-size box,
NMS collapse) so a regression in the math fails fast without any model download.

---

## 5. FLOPs budget (B25 / ADR-15) — measured, and a handed number corrected

All measured off the ONNX graphs (Conv MACs + attention/GEMM MACs via symbolic shape
inference), load-independent, `2·MAC = FLOPs`:

| model | GFLOPs/img | share of encoder |
|---|---|---|
| **encoder** PE-Core-S16-384 vision | **31.32** | (denominator) |
| nudity marqo-384 (dedicated) | 7.83 | 25.0% |
| **people YuNet-640 (dedicated)** | **0.688** | **2.2%** |
| **Σ dedicated** | **8.52** | **27.2%** |
| B25 cap | 9.40 | 30% |

**VERDICT: FITS**, margin 2.8 points (≈0.88 GFLOPs headroom).

> **§7(b) correction, recorded honestly.** A first naive pass counted only Conv+GEMM and
> got encoder 23.15 GFLOPs → *false BREACH*. The encoder is a ViT: its 13 attention layers
> add **6.14 GFLOPs** of QKᵀ/AV matmuls that a Conv-only counter misses. With attention
> included the encoder is **31.32 GFLOPs** and nudity is **exactly 25.0%** of it — which is
> the "~¼ of PE-Core-S16" ratio the nudity report already claims (its *absolute* "4.5
> GFLOPs" figure is stale; the real number is 7.83, but the ratio and its conclusion stand).
> Verifying the denominator myself rather than inheriting it is what turned a phantom breach
> back into a real 2.8-point margin.

**The scaling invariant (ADR-15).** Persons ride the shared embedding (rung 1, ~0 marginal
— the cascade is two D-length dot products). Only **faces** pay a dedicated forward, and at
2.2% of the encoder the whole track is a rounding error against the 30% budget. If a future
host makes even a handful of dedicated heads too costly, the escape hatch is the same shared
multi-head backbone nudity's report names; and D-people-1 (distil YuNet→embedding head) is
the mandated long-term fate (T2 rung 2).

---

## 6. Integration contract (for b-engine + b-daemon)

`imgtag.moderation.load_people_head(profile)` — `PeopleHead | None` (None when the YuNet
artifact is absent; a missing track is reported by name, never a silent zero). Already
registered in `_TRACKS` and found by `load_heads`.

- `PeopleHead.wants_images = True` — answers from pixels.
- `PeopleHead.col_roles = ["n_persons","n_faces","n_persons_conf","n_faces_conf"]` — the
  head is the single authority for its column schema; the engine reads it once and writes
  `tracks/people.json`'s `col_roles`. `PeopleHead.spec` (property) carries the derive band
  edges as data → folded into the header `spec_sha` (no shared-file write, rule-7 safe).
- `score(embeddings, images, ids) -> list[list[dict]]`, per image:
  - **ONE raw record** `{"category":"people", "cols":{n_persons, n_faces, n_persons_conf,
    n_faces_conf}, "tier":"none"}` → the engine writes a **`people.f32 [N,4]`** dense
    sidecar in `col_roles` order (b-daemon's single-column ask). Tier `none`: a count is
    not a moderation flag, so it never enters the ADR-14 enforcement accounting. Emitted
    for **every** image incl. empty/`unreadable`/`no_pixels` (T1), so confidence is present
    for every image.
  - **Zero-to-four `match` chips** `{"category":"one-person"|…, "p":conf, "tier":"match"}`,
    one per SATISFIED category (multi-label — multi-person AND one-face can co-fire). These
    populate the manifest **`content` bucket** (`{match:{one-person:N,…}}`, verified) and
    the per-image flags the 14:16Z detail view ranks. Unsatisfied → no chip, no spurious
    membership.
- **The four USER categories DERIVE from the two count columns** at read via
  `people.derive(n_persons, n_faces)` (T1): `one-person = n_persons==1`, `multi-person ≥2`,
  `one-face = n_faces==1`, `multi-face ≥2`. A future "multi means ≥3" is a free re-read of
  `people.f32`. **b-daemon / b-app: read `people.f32` cols 0/1 and call `derive()`** — the
  chip columns (`one-person.f32` …) are a convenience for the content bucket, not the
  source of truth; the counts are.

**Coordination:**
1. **b-engine** — ✅ `match` content-tier landed (`CONTENT_TIERS`, `accumulate()`); ✅
   multi-col `[N,C]` write path landed and verified for people. ⚠️ **OPEN BUG** reported:
   `_apply_moderation` (indexer.py:275) does `roles.get(cat, ["p"])[0]`, which is
   `None[0]` for any single-value track whose head has no `col_roles` (weapons/nudity/…),
   because `detect.col_roles` stores `None` for them and `.get(cat, default)` skips the
   default when the key exists-with-`None`. Fix: `(roles.get(cat) or ["p"])[0]`. Blocks the
   full-multi-track moderation index; people verified in isolation until it lands.
2. **b-daemon** — expose the four categories in search/tag vocab via `derive()` over
   `people.f32`; "sort gallery by n_persons/n_faces" is a direct column read. Validation
   `.npy` for `cocoval2017` offered.

---

## 7. Verification status (honest)

- ✅ Embedding cannot count — **measured, first-party** on 4,773 COCO images (§1).
- ✅ Face count vs COCO keypoint proxy — measured, exact-count 0.814 @ τ0.6 (§3a).
- ✅ Person cascade vs exhaustive COCO boxes — measured, bucket acc 0.742 (§3b/c).
- ✅ Back-view recovery — measured, **0.841 recall on 309 faceless-person images vs 0.000
  for faces alone** (§3d).
- ✅ Numpy decode matches expected YuNet behaviour on real crowd/single/empty images (§4).
- ✅ FLOPs / B25 — measured off graphs, **27.2% < 30%**, denominator independently
  re-derived after a false-breach scare (§5).
- ✅ **End-to-end through the real CLI** — `imgtag index … --moderation` persists
  `people.n_persons/.n_faces/*_conf` sidecars; the four acceptance images derive correctly
  (back-view→one-person, single→one-person+one-face, crowd→multi+multi, empty→none), with
  **zero moderation-count pollution and no crash**.
- ✅ 16 unit/contract/acceptance tests green (`tests/test_people.py`).
- ⚠️ **Latency NOT reliably measured** — dev box load 75.4 / 16 cores (4.7× the ORACLE
  honesty threshold of 9.6), so the ~15 ms/img @ 640² samples are **UNRELIABLE** and must
  be re-measured idle, then on the 🐧 8GB-no-GPU target. FLOPs (0.688 G, load-independent)
  is the trustworthy budget number.
- ❌ **`calibrated = False`, `enforcement_ready = False`** — τ_face is fitted against a
  keypoint PROXY, never against face ground truth (COCO has none). A counting track never
  authorizes an action regardless (§Proxy limits).
- 🔭 **D-people-1 (darwin):** distil a person-count head to lift one-person past 0.561
  without a per-box person detector (which does not fit the budget), or re-budget with
  numbers.

### Proxy limits (stated plainly, never hidden)

COCO never annotated "faces". The face proxy is **≥2 of {nose, left-eye, right-eye}
keypoints marked visible** (v=2) — chosen so a face at an ANGLE (nose + one eye, far eye
occluded) counts, honoring "even at angles"; strict 3-of-3 and loose 1-of-5 variants are
computed alongside as sensitivity bounds. Two honest holes:

- **35.1% of COCO persons carry no keypoints at all** (too small/occluded for COCO to
  annotate). The proxy is blind to those faces; §3d's back-view eval is restricted to
  keypoint-annotated persons so the number is not deflated by proxy blindness.
- The proxy is a *lower bound on visibility, upper bound on nothing* — it can call a face
  "visible" that a human would debate. Every face number here is therefore "vs the COCO
  visible-keypoint proxy", never "vs face ground truth", which does not exist on this
  machine. Real face-recall calibration waits for lawfully-held labels on the target host.

---

## 8. Reproduce

```
# ground truth (person counts + face-visibility proxy)
.venv/bin/python research/bench_scripts/people_gt.py
# embedding-cannot-count falsification (free; uses the existing index)
.venv/bin/python research/bench_scripts/people_eval.py --json .scratch/people-eval.json
# YuNet over all of COCO val2017, cache raw detections
.venv/bin/python research/bench_scripts/people_yunet_sweep.py --out .scratch/yunet-coco.json
# fit + save the shipped cascade, emit every measured table
.venv/bin/python scripts/train_people_head.py --save --report .scratch/people-report.json
```

Acceptance sketch (VISION-ADDENDA 13:28Z): back-view hiker → `one-person` + zero-face ✓ ·
couple selfie → `multi-person` + `multi-face` ✓ · empty landscape → none ✓ · crowd →
`multi-person` ✓ — all four verified end-to-end through the real CLI (§7).
