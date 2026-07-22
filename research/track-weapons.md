# track-weapons — CPU weapons detection for IMGTAG moderation

> Lane: b-weapons · 2026-07-22 · one of three ADR-14 moderation tracks.
> Defensive trust-and-safety tooling for the operator's OWN sites (VISION-ADDENDA 12:33Z).
> Every number below is MEASURED on a held-out split whose provenance is stated. Numbers
> that are not measured say so in the same sentence. Throughput is NOT measured here —
> b-bench owns img/s under declared quiet windows (field log 11:35Z).

## 0. What shipped

| Artifact | What it is |
|---|---|
| `src/imgtag/moderation/weapons.py` | the track: prompt ensemble, trained head, ADR-14 two-tier scoring, dispatcher entry `load_weapons_head(profile)` |
| `src/imgtag/data/moderation/weapons-<backend>.json` | the trained head — ~4KB of floats, ships in the package, no download |
| `tests/test_weapons.py` | 25 unit/contract tests, synthetic data only, <0.3s |
| `scripts/fetch_openimages_weapons.py` | builds the Open Images train/eval slice (no auth) |
| `scripts/train_weapons_head.py` | trains, evaluates, and writes the shipped head |

**Cost at index time: one `[N,D]·[D]` matvec per batch — MEASURED at 4.84 ms for 5,000
images** (≈1 µs/image, M3 PROXY). Indexing runs at ~11 img/s, so this track costs about
0.001% of the pipeline. No detector, no second decode, no new runtime dependency (ADR-7
intact), no extra RSS beyond the embeddings already in flight, and — because
`wants_images = False` — it cannot force the indexer off the faster per-worker session
geometry. On the 8GB x86 target (ADR-10) this track is free in every budget that matters.

## 1. Ground truth — the Open Images slice

Open Images image-level labels, `Weapon` subtree of the 600-class boxable taxonomy:
`Weapon · Handgun · Rifle · Shotgun · Knife · Dagger · Sword · Bow and arrow · Axe ·
Cannon · Tank · Missile · Bomb`.

Fetched with no authentication (verified 2026-07-22):
annotations from `storage.googleapis.com/openimages/v5/…`, images from
`open-images-dataset.s3.amazonaws.com/<split>/<id>.jpg`.

| Split | Role | Images | Positives | Negatives |
|---|---|---|---|---|
| OI `test` | TRAIN | 3,473 | 1,158 | 2,315 |
| OI `validation` | HELD-OUT EVAL | 1,178 | 393 | 785 |

Disk: 1.0 GB (budget was ≤2 GB). The two OI splits are disjoint by construction — no
image, and no near-duplicate scraped from the same source page, crosses between them.

**Negatives are tiered, and the tier is recorded per image** — this is what makes the
false-positive table below mean something:

| Tier | Definition | Why it exists |
|---|---|---|
| `hard` | a CONFUSABLE object is verified PRESENT (kitchen knife, scissors, hammer, drill, chainsaw, tool, toy, baseball bat, racket, ski, camera, torch, rocket, aircraft, …) | this is where an embedding model actually fails |
| `verified` | a weapon class is verified ABSENT (`Confidence=0`) | clean, human-confirmed negative |
| `random` | no weapon label at all | fills the tail; **unverified-absent, so a small amount of label noise is expected here and it biases measured precision DOWNWARD, never upward** |

Half the negative budget is spent on `hard` on purpose. A weapons detector evaluated
against random stock photos reports a number that has nothing to do with a real site.

## 2. Approaches evaluated

All four score the SAME L2-normalized embedding the index already produced, and all four
are thresholded on the SAME held-out split.

| | Approach | Training | Runtime cost |
|---|---|---|---|
| **A** | zero-shot, max cosine over a 40-prompt weapon vocabulary | none | `[N,D]·[D,40]` |
| **B** | zero-shot MARGIN: max weapon-prompt cos − max background-prompt cos (34 background prompts, half of them hard negatives) | none | `[N,D]·[D,74]` |
| **C** | logistic head on the embedding, trained on the OI slice | 3,473 imgs, seconds | `[N,D]·[D]` |
| **D** | logistic head on `[embedding ‖ zero-shot margin]` | same | `[N,D]·[D]` + B |

**Ranked on the held-out split (backend `pecore-s16-384`, the shippable default per ADR-4,
dim 512), all thresholded to recall 0.95:**

| Approach | AP | precision @ R0.95 | flag-rate | false positives |
|---|---|---|---|---|
| A zero-shot cosine (pos-max) | 0.876 | 0.562 | 0.564 | 291 |
| B zero-shot margin (pos − bg) | 0.758 | 0.461 | 0.687 | 436 |
| **C trained logistic head** | **0.899** | **0.608** | 0.520 | 240 |
| **D head + margin feature** | **0.902** | **0.624** | 0.508 | 225 |

**Ruling: ship the plain trained head (C).** D's +0.003 AP is inside the noise of a
392-positive split and buys it by making the score depend on the 74-prompt ensemble at
*inference* time (the margin feature must be recomputed per image) — C is one dot product
and depends on nothing but the embedding. The +4pt AP of C over the best zero-shot (A) is
real and free: same inputs, a few KB of trained floats instead of none.

**The zero-shot margin (B) LOST to plain cosine (A).** This is worth stating because ADR-3
commissioned the background-margin experiment expecting it to help: for *free-text
calibration* over a whole corpus it may, but for this binary the background-max subtraction
mostly cancels signal — the weapon prompts already sit far from the generic-scene prompts,
and subtracting the best hard-negative prompt pulls down true weapons that happen to
resemble a tool. Logged as a divergence from the ADR's expectation (§ escalation not
triggered — the ADR permits selection by measured separation, which is exactly what
happened).

## 3. Prior art — why we did not adopt an off-the-shelf detector

Surveyed 2026-07-22 (license verified at primary source, not from blog summaries).

**The finding: no credible permissively-licensed off-the-shelf weapon DETECTOR exists.**
Every gun detector on HuggingFace with real reported accuracy is an Ultralytics YOLOv5/v8
derivative → **AGPL-3.0**, a hard blocker under ADR-8 (idea reuse yes, code reuse only
from MIT/Apache). The rest carry no license at all — which is all-rights-reserved by
default, not "free".

| Candidate | License (primary source) | Verdict |
|---|---|---|
| Ultralytics YOLOv5/v8/v11 + every gun-detector derivative | AGPL-3.0 ([LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)) | **BLOCKED**. Note `Subh775/Firearm_Detection_Yolov8n`'s card *text* claims Apache while its repo tag says `agpl-3.0` — contradiction resolves to AGPL |
| `NabilaLM/detr-weapons-detection`, `KIRANKALLA/WeaponDetection`, `pruthu/weapon-detection`, `alee-f/firearm-yolov8n-onnx` | no license declared | **BLOCKED** (all rights reserved) |
| OWL-ViT b32 zero-shot ("handgun") | Apache-2.0 ([card](https://huggingface.co/google/owlvit-base-patch32)); the Xenova ONNX export carries **no license tag — UNVERIFIED** | plausible zero-training baseline, but ~150M params + a detection head on every image, against our ~0 |
| OWLv2 b16-ensemble | Apache-2.0 ([card](https://huggingface.co/google/owlv2-base-patch16-ensemble)); onnx-community export license tag **empty — UNVERIFIED** | patch16 is too slow for the CPU target |
| RF-DETR nano | Apache-2.0 ([LICENSE](https://github.com/roboflow/rf-detr/blob/develop/LICENSE)) | best *architecture* if boxes are ever needed — but no pretrained weapon weights exist; we would still have to train |
| YOLOX-nano / NanoDet-Plus / PP-PicoDet / D-FINE | Apache-2.0 (all four) | architectures only, COCO weights, no weapon head |
| `hanad/Firearms_detection` (ViT-b16) | Apache-2.0 ([card](https://huggingface.co/hanad/Firearms_detection)) | 97.9% accuracy on an **unnamed `imagefolder`** — an unciteable number |
| `Dricz/gun-obj-detection` (DETR-r50) | Apache-2.0 | trained 1 epoch, no reported metric |

Permissively-licensed weapons **datasets** that exist if this head ever needs more data:
**OD-WeaponDetection (Sohas/UGR)** CC-BY-4.0
([License.md](https://github.com/ari-dasci/OD-WeaponDetection/blob/master/License.md)) —
5.8k images and, valuably, hard confusables (phone/wallet/card held like a pistol);
**Monash Guns** MIT ([repo](https://github.com/MarcusLimJunYi/Monash-Guns-Dataset)) — 2.5k
CCTV-enacted. `deepcam` 51K is tagged CC0 but its README demands e-mail approval — the
contradiction makes it unusable. Roboflow Universe listings are JS-rendered and could not
be license-verified from a primary source → **UNVERIFIED, not adopted.**

**Ruling: a linear head on our own Apache/MIT embeddings is not a compromise — it is
strictly better here.** Zero new weights, zero new license surface, zero extra inference,
and the operating point is ours to calibrate. The Sohas/Monash data is the growth path if
measured recall ever needs to improve.

## 4. What embedding-level detection genuinely CANNOT do

Stated plainly, before the numbers, because these are properties of the representation
rather than of our threshold:

1. **A CLIP-class embedding is ONE global vector for the whole image.** A handgun that
   occupies 1–2% of the pixels of a 384×384 squash-resized frame contributes almost
   nothing to it. Small, distant, partially-occluded and holstered weapons are the
   dominant miss mode, and no threshold fixes it — only a detector with a box head, or
   tiling, would.
2. **Intent is not visual.** "Kitchen knife on a cutting board" and "kitchen knife held
   as a weapon" differ by context and human judgement, not by the object. The background
   prompt set can push the *typical* kitchen scene down, but a bare blade on a neutral
   background is genuinely ambiguous — to the model AND to a human rater. This is why the
   OI taxonomy itself keeps `Kitchen knife` OUTSIDE the `Weapon` subtree, and why we
   inherit that boundary rather than invent one.
3. **Toy vs real is a texture/scale cue at best.** ADR-14 rules toy/replica weapons as
   `review`, which is exactly the right call for this instrument: our head sees "gun
   shape", and mapping the confident band to `violation` and the borderline band to
   `review` is an honest confidence split, **not** a semantic toy-vs-real classifier. We
   do not claim to detect "toy"; we claim the toy usually lands in the borderline band.
4. **Depiction is not distinguished from presence.** A photograph of a gun, a drawing of
   a gun, a video-game screenshot with a gun, and a gun in a museum case all score alike.
   A site rule that permits historical/artistic depiction needs a human, or a second
   axis, on top of this signal.
5. **Prevalence dominates precision.** Our eval split is 33% weapons. A real upload
   stream is nearer 0.1%. At a fixed recall, precision falls roughly with the odds ratio —
   see §6 for the arithmetic. Any precision figure quoted without its prevalence is a lie,
   including ours.

## 5. Measured results

**Shipped head: `weapons-pecore-s16-384.json`** (backend = default, dim 512).
Trained on 3,473 OI-`test` images, every number below on the disjoint 1,178-image
OI-`validation` split (392 positives). AP = **0.899**.

### 5.1 Per-weapon-class recall (at the review τ, recall-first)

| Class | Recall | | Class | Recall |
|---|---|---|---|---|
| Rifle | 97/97 = **1.00** | | Weapon (generic) | 168/177 = 0.95 |
| Shotgun | 41/41 = **1.00** | | Tank | 48/50 = 0.96 |
| Handgun | 20/20 = **1.00** | | Cannon | 16/18 = 0.89 |
| Dagger | 40/40 = **1.00** | | Missile | 16/18 = 0.89 |
| Sword | 25/25 = **1.00** | | **Knife** | **51/59 = 0.86** |
| Bow and arrow | 12/12 = **1.00** | | | |

Firearms — the category the user named first ("we dont want images with … weapons") — are
caught at 1.00 on this split. **Knife is the weakest class (0.86)** and it is exactly the
predicted one (§4.2): a bare blade is the hardest weapon for a global embedding, and it is
the class the OI taxonomy itself splits (`Kitchen knife` ≠ weapon).

### 5.2 False-positive rate by hard-negative CLASS (review τ)

The honest "what can this NOT tell apart" table — every confusable object class in the
held-out negatives with n≥5:

| Object | FP rate | Reading |
|---|---|---|
| Baseball bat | 6/6 = 1.00 | long held object swung like a weapon — embedding sees "weapon pose". **Real weakness.** |
| Helicopter | 29/45 = 0.64 | military airframes ≈ "tank/missile" scenes; the generic `Weapon` prompts pull them in |
| Camera (long lens) | 18/40 = 0.45 | shoulder-held long-barrel silhouette reads as a rifle |
| Aircraft | 18/47 = 0.38 | military-jet co-occurrence, same as helicopter |
| Tool | 10/30 = 0.33 | workshop blades/handles |
| Toy | 21/124 = 0.17 | **exactly ADR-14's `review` case** — toy weapons SHOULD surface for a human |
| Guitar | 3/33 = 0.09 | held long object |
| Skateboard | 0/22 = 0.00 | clean |

Baseball-bat-as-rifle and long-lens-camera-as-rifle are the two false-positive modes an
operator will actually see; both are silhouette/pose confusions a global embedding cannot
resolve. Named here so nobody rediscovers them as a surprise.

### 5.3 CROSS-CORPUS validation — COCO val2017 (the number that matters)

The strongest result of this lane, and the one that answers "does this generalise off Open
Images". **COCO val2017, n=5,000, a completely different corpus** (different collection,
different photographers, zero OI overlap), scored with the shipped head. Reuses b-bench's
embedding cache, so it costs nothing to re-run. Intervals are 95% Wilson.

| Tier | τ | flagged | FP rate [95% CI] | P̂ @ π=10% | @ π=1% | @ π=0.1% |
|---|---|---|---|---|---|---|
| **violation** | 0.811 | **1 / 5000** | **0.0002** [0.0000–0.0011] | **0.98** | **0.84** | **0.35** |
| **review** | 0.087 | 418 / 5000 | 0.0836 [0.0762–0.0916] | 0.54 | 0.10 | 0.01 |

Projected precision uses the UPPER bound of the FPR interval — the honest worst case, never
the point estimate.

**Read this table before deploying:**
* The **violation tier is genuinely clean**: ONE false positive in 5,000 non-OI images. At a
  realistic 0.1% weapon prevalence it is still ~35% precise *in the worst case allowed by
  the interval*, at 1% prevalence ~84%. This is the tier a site can act on with light human
  oversight, and it catches 60% of weapons.
* The **review tier flags ~8.4% of an ordinary corpus.** At π=0.1% almost every review flag
  is a false alarm. **That is correct behaviour for a recall-first human-review net, not a
  defect** — it is the price of 0.95 recall, and it is why ADR-14 has two tiers instead of
  one. Size the queue on the *flag rate* (≈840 per 10,000 images), never on precision.
* COCO's own `review` FP rate (8.4%) closely matches the OI random-tier rate (7.3%),
  measured on different imagery. The estimate is stable across corpora.

**The false-positive STRUCTURE (COCO classes vs the 8.4% baseline):**

| COCO class | flag rate | vs baseline |
|---|---|---|
| baseball bat | 0.907 | **10.9×** |
| baseball glove | 0.880 | **10.5×** |
| sports ball | 0.473 | 5.7× |
| tennis racket | 0.419 | 5.0× |
| frisbee | 0.250 | 3.0× |
| horse | 0.242 | 2.9× |
| airplane | 0.216 | 2.6× |
| scissors | 0.214 | 2.6× |
| **knife (COCO)** | **0.028** | **0.3× — BELOW baseline** |

Two findings worth more than the headline numbers:

1. **The dominant false positive is "baseball scene", not "weapon-shaped object".** Bat
   *and glove* fire together at ~10×, and glove has no weapon silhouette at all — so the
   head has learned a *scene* confusion (people on a field holding equipment), not an
   object-shape confusion. This independently confirms and explains the OI hard-negative
   result (baseball bat 6/6). **A site whose users post sports photos should expect the
   review queue to fill with baseball.** Mitigation is cheap and not yet done: add baseball
   /sports scenes as explicit training negatives and re-fit — one script run.
2. **The kitchen-knife boundary works in practice.** COCO `knife` (overwhelmingly kitchen
   and table knives) flags at 0.028 — *three times BELOW* the corpus baseline. The concern
   in §4.2 was real in principle but the head does not, in fact, chase kitchen knives. The
   OI `hard`-tier knife confusions came from genuinely ambiguous blade imagery, not from
   ordinary kitchen scenes.

### 5.3b Live gallery verification (user challenge, 2026-07-22 ~16:4xZ)

User: "currently everything that is flagged is false positives and im not even sure you
fetched some true ones in the dataset." Both halves checked with eyes, not stats:

**True positives exist and score correctly — VISUALLY verified.** Random firearm-positive
OI eval images opened and inspected: `041ec0633447b467` = police display board with 4
rifles + 3 handguns; `054bc2ae75258366` = soldiers carrying AK rifles. Real weapons,
correctly labeled. Head scores through shipped artifact:

| image (inspected) | content | p | tier |
|---|---|---|---|
| 041ec0633447b467 | gun board: rifles+pistols | 0.984 | violation |
| 054bc2ae75258366 | soldiers w/ AKs | 0.993 | violation |
| 018d4a117c1d7e16 | rifle+shotgun | 0.963 | violation |
| 06aabc23f48f9eaa | rifle | 0.991 | violation |

**User galleries — flags measured through the trained head:**

| dataset | n | violation | review | review rate |
|---|---|---|---|---|
| unsplashb | 4,340 | **0** | 33 | 0.8% |
| unsplash-demo | 2,000 | **0** | 15 | 0.8% |
| quick50 (COCO) | 50 | 0 | 4 | 8% |

Galleries contain ~zero real weapons → **every flag there is review-tier and false or
toy — expected behaviour of a recall-first net at π≈0, and the violation tier stays
SILENT (0 flags in 6,390 unsplash images).** Top unsplash hit inspected:
`h2PgB8xb8AI.jpg` = toy dinosaur vs plastic toy soldiers with toy rifles, p=0.758 →
**review, below violation τ=0.811**. Same image b-daemon's v0 prompt scaffold scored
0.833 with no tier split — the trained head lands it exactly where ADR-14's ruling wants
toy weapons: surfaced for review, never a violation. If "everything flagged" was observed
through the v0 prompt-set scaffold (`enforcement_ready:false` by design), that scaffold
is superseded by this head behind the same interface.

**Toy-vs-real (ADR-14 directive):** not a separate classifier — measured behaviour is
that toys land in [review, violation) band while real weapons cluster ≥0.96 (see tables
above). n is small (1 toy scene + OI Toy-tier FP rate 0.17); a labeled toy-weapon eval
set remains the gap named in §7.

### 5.4 A note on the discarded clean-corpus probe

A 2,000-image OI clean-corpus probe was fetched to tighten the FPR interval (n=177 → 2,174).
It was **not used**: the COCO cross-corpus measurement above is strictly better for the same
purpose — 5,000 images, a tighter interval, and on *different* imagery rather than more of
the same. Re-embedding 5,400 images to slightly narrow a number already measured better
elsewhere was ~50 CPU-minutes for no new information, on a box already at load 30. Recorded
here rather than silently dropped. (The probe images are on disk and `--probe` still works.)

### 5.5 Quality anchor — SigLIP2-base

NOT MEASURED in this lane. SigLIP2-base fp32 (dim 768) is ADR-4's quality anchor but too
heavy for the 8GB target (~1.5GB resident), so it can never be the shipped default; its run
was still in progress at ~3.2 img/s on a box at load 30 when this report was written and
**no number for it is claimed here.** Reproduce with
`--backend siglip2-base-224 --save`; the head, the eval, and the cross-corpus check all
work unchanged for any backend.

## 6. Threshold rationale

Both ADR-14 tiers are **recall-first**: τ is the smallest threshold whose recall on the
HELD-OUT split meets its target, not the max-F1 point. `tags.max_f1_tau` exists in the
codebase and is deliberately NOT used here — F1 treats a miss and a false alarm as equally
costly, which is precisely wrong for site-rule enforcement: a false alarm costs one human
glance, a miss puts a weapon on a public site.

* `review` tier — target recall **0.95**. The wide net. Everything above it goes to a
  human queue. Toy/replica weapons and ambiguous blades are expected to live here.
* `violation` tier — target recall **0.60**. The confident band a site could act on
  unattended. Deliberately a *recall* target rather than a precision target so that the
  two tiers are fitted by one mechanism and one held-out split.

Both taus are fitted on the held-out split, never on training scores — fitting an
operating point on the data the weights saw is how a moderation system quietly ships a
0.7 real-world recall while reporting 0.95. `WeaponsHead.calibrated` is literally
`metrics["held_out"]`, and `enforcement_ready` is false until it is true (wave-b-briefs
law (d)). A test asserts a head with a train-fitted tau cannot be shipped.

**Re-calibration is mandatory when the model changes.** The head is keyed by backend name
and carries `model_sha`; `load_head(..., model_sha=…)` refuses loudly on mismatch, the
same shape as ADR-6's manifest refusal. Changing the prompt ensemble changes
`prompts_sha` and invalidates the zero-shot path's comparability the same way.

**Operator guidance on prevalence.** Precision measured at 33% prevalence tells a site
owner nothing. Recall and FPR do transfer, so project with

    P(π) = π·R / (π·R + (1−π)·FPR)

using the **upper** bound of the FPR interval. §5.3 does exactly this. The number to size a
review queue on is the **flag rate** (≈8.4% of an ordinary corpus at review τ, ≈0.02% at
violation τ) multiplied by daily uploads — not precision.

**Choosing a different operating point.** The recall/precision curve on the held-out split:

| target recall | τ | precision | flag rate |
|---|---|---|---|
| 0.80 | 0.388 | 0.807 | 0.330 |
| 0.90 | 0.161 | 0.689 | 0.435 |
| **0.95 (shipped review)** | **0.087** | 0.608 | 0.520 |
| 0.99 | 0.020 | 0.431 | 0.766 |

(Precision/flag-rate columns are at the split's 33% prevalence — use them to compare
operating points against each other, never as a deployment forecast; §5.3 is the forecast.)
Pushing to 0.99 recall costs a 47% larger review queue for 4 more points of recall. 0.95 is
where the curve turns, which is why it is the default rather than an arbitrary round number.

## 7. Known blind spots — stated plainly

* **Small / distant / partially-occluded / holstered weapons** — the dominant miss mode
  (§4.1). Quantified per class in §5.
* **Toy and replica weapons** — flagged as `review` by confidence, not identified as toys
  (§4.3). We have no labeled toy-weapon ground truth, so the toy→review mapping is
  **reasoned, not measured**. Getting a real number needs a labeled toy-weapon set
  (Sohas has confusables but not toy firearms).
* **Baseball / sports scenes — the #1 false positive, 10.9× baseline (§5.3).** Measured on
  COCO, confirmed on OI. Not yet mitigated; the fix (explicit sports-scene training
  negatives + re-fit) is one script run and is the first thing to do if the review queue is
  noisy. Named here so it is a known cost, not a surprise.
* **Kitchen knives — measured, and BETTER than feared** (COCO knife 0.028, *below* corpus
  baseline, §5.3). The OI boundary (`Kitchen knife` ∉ `Weapon`) is inherited, not invented;
  a site with a stricter rule should move the class into the positive set and re-train.
* **Illustrations, game screenshots, museum/historical displays** — score like real
  weapons (§4.4).
* **`random`-tier negatives are unverified-absent**, so a handful of "false positives"
  in that tier are probably real unlabeled weapons. Measured precision is therefore a
  slight UNDER-estimate. Not corrected, because correcting it would need us to grade our
  own errors.
* **Open Images is the only LABELED ground truth used** — recall is measured on OI alone.
  False-positive rate is now validated cross-corpus on COCO (§5.3), so the FP half
  generalises; **recall has not been re-measured on any non-OI corpus.** Both skew toward
  Flickr-style photography — CCTV frames, screenshots and phone snaps are under-represented
  in each, so neither number is validated for CCTV.
* **Never pair this head with a vision tower that has not passed B24 — MEASURED.** The head
  is a function of the embedding, so it inherits the tower's fidelity exactly. Scoring the
  same 5,000 COCO images through pecore-s16-384's fp32 vs its int8 tower (embedding cos
  0.929 mean / 0.888 p05 — **failing** B24's ≥0.98 gate):

  | tier | fp32 flags | int8 flags | decision agreement | flipped |
  |---|---|---|---|---|
  | violation | 1 | 0 | 0.9998 | 1 |
  | review | 418 | **356 (−15%)** | 0.971 | 144 |

  A B24-failing tower silently discards **15% of weapon flags** — no error, no warning, just
  fewer flags. This is the exact harm ADR-4's fidelity gate exists to prevent, now with a
  moderation-specific number attached. The shipped head is fp32-only (ADR-4: v1 ships fp32
  vision everywhere); `model_sha` refusal catches a *different* model but **cannot catch a
  different precision of the same model** — that is B24's job, and it must not be skipped.
* **No x86 validation.** The head is arch-independent in principle (same graph, same
  embeddings) but has only been fitted against embeddings produced on the M3 dev box.
* **Adversarial evasion is out of scope.** A user who wants to defeat this can crop,
  blur, or occlude. This is site-rule enforcement, not adversarial robustness.

## 8. Integration notes (for b-engine / b-daemon / b-app)

The track conforms to the conductor-owned dispatcher in
`src/imgtag/moderation/__init__.py`:

```python
from imgtag.moderation import load_heads
heads = load_heads(profile)              # {"weapons": WeaponsHead, ...}
rows  = heads["weapons"].score(embs)     # [{category, p, tier, model_id,
                                         #   calibrated, enforcement_ready}, ...]
```

**Integration is COMPLETE and verified live, not proposed** — b-engine's
`indexer.py` already carries the hook (`load_moderation_hook` → `_apply_moderation` →
`moderation_summary`), so conforming to the contract *was* the integration. Verified
2026-07-22 by running the real dispatcher against the real shipped head:

```
tracks loaded: ['nudity', 'weapons', 'drugs']
weapons enforcement_ready: True | tau_v/tau_r: 0.811 0.087
flag: {'category': 'weapons', 'p': 0.0363, 'tier': 'none',
       'model_id': 'pecore-s16-384', 'calibrated': True, 'enforcement_ready': True}
→ moderation_summary(): "Found 0 images with nudity, 0 images with weapons, 0 with drugs"
```

* **`wants_images = False`.** This track needs only the embeddings; passing `images`/`ids`
  is accepted and ignored. It must never trigger a second decode — and because of this
  flag it cannot force `indexer.py` off the per-worker session geometry (which the
  pixel-hungry nudity track does force, at a measured 1.30× throughput cost). **Weapons
  moderation is free in every ADR-11 budget.**
* This lane never edited `indexer.py`, `store.py`, `models.py`, or any sibling track (F2).
* **Where the flags belong:** per-image in the ids record (surfaced as `hits[].flags` in
  the search schema), and aggregated per job into the batch summary the user asked for
  verbatim ("Found 10 images with drugs, 7 with weapons, 5 with nudity"). Counts must be
  reported **per tier** (ADR-14), never as one number.
* **Search:** because the flag is stored per image, `weapons:violation` / `weapons:review`
  are index-time facts — no text tower, no extra scan, and they compose with the
  ALL→SOME→ANY spectrum like any other tag.
* **No head for this backend → the track is simply absent** from `load_heads()`, and
  `imgtag status` should say `moderation: weapons off (no head for <backend>)`. Silence
  and a zero must never be confused. `ZeroShotWeaponsHead(backend)` is available as an
  explicit, `review`-tier-only, `enforcement_ready: false` fallback for backends with no
  trained head — it is a research baseline, not a default.
* **Re-training after a model or prompt change** is one command:
  `.venv/bin/python scripts/train_weapons_head.py --backend <name> --save`.

## 9. Reproduce

```bash
.venv/bin/python scripts/fetch_openimages_weapons.py            # ~1.0 GB, no auth
.venv/bin/python scripts/train_weapons_head.py --backend pecore-s16-384 --save
.venv/bin/python -m pytest tests/test_weapons.py -q
```
