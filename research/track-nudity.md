# track-nudity — the nudity/NSFW moderation track

> Mandate: VISION-ADDENDA 2026-07-22 12:33Z (verbatim) — *"we dont want images with
> nudity, weapons or drugs … these are very important to indentify correctly"*.
> Policy: ORACLE ADR-14 (two tiers). Deps: ADR-7. Escalation: ORACLE §7.
> Defensive trust-and-safety tooling for the user's own public sites.
>
> **OPERATING POINT LAW — recall-first.** A missed nude reaches a public site; a false
> flag costs one human glance.
>
> **EVAL DATA LAW — obeyed.** No explicit-adult corpus was downloaded to this machine.
> Every number below labelled *measured* is first-party and comes from **safe** corpora
> already on disk, and therefore describes the **FALSE-POSITIVE side only**. True-positive
> recall is **not reproduced here**; it is cited from the model's published evaluation and
> labelled as such every time it appears. Nothing in this document is a recall claim.

---

## 1. Phase 1 — ranked approaches

| # | Approach | License | CPU cost | Integration fit | Accuracy evidence | Verdict |
|---|---|---|---|---|---|---|
| **1** | **Marqo/nsfw-image-detection-384** — ViT-Tiny/16 @384, 5.6M params, purpose-trained | **Apache-2.0** | ~4.5 GFLOPs/img (≈¼ of PE-Core-S16-384); 22.5MB fp32 ONNX | standalone → backend-agnostic; survives whichever model wins the b-bench | **98.56%** acc on a 20k held-out split (10k NSFW / 10k SFW); training set 220k incl. photos, drawings, Rule 34, memes, AI-generated | **CHOSEN** |
| 2 | Falconsai/nsfw_image_detection — ViT-base/16 @224, 86M params | Apache-2.0 | ~15× the params of #1 | standalone | 98.04% acc on its own set — but Freepik measures it at **31.25%** on *mild* NSFW and 78.54% on *medium* | rejected: 15× cost, and its published weakness is exactly the recall-first tier |
| 3 | Freepik/nsfw_image_detector — EVA-02-base @448, 4 severity levels | MIT | 448²/patch14 = 1024 tokens, ~87M params; published **28 ms/img on an RTX 3090** | standalone; severity levels map neatly onto ADR-14 tiers | best published table of the set (99.54/97.02/98.31/99.87 by level) | rejected for v1: a GPU-timed 87M-param model at 448² is the wrong shape for an 8GB no-GPU box. Best *future* upgrade if the target host proves fast enough |
| 4 | LAION CLIP-based-NSFW-Detector (`clip_autokeras_nsfw_b32`) — tiny MLP over CLIP ViT-B/32 embeddings | MIT (`license.md`; GitHub reports NOASSERTION) | ~0 **if** B/32 is already the index backend — otherwise a whole second 88M-param CLIP tower | **space-locked.** Our `openclip-vitb32` is genuinely `openai/clip-vit-base-patch32` (config `_name_or_path`), so the space matches — but B/32 is the ADR-4 **control**, not the expected winner. If PE-Core or SigLIP2 wins, this costs *more* than #1 | none published: no accuracy, AUC or threshold in the repo | rejected: artifact is AutoKeras/TensorFlow (TF at runtime violates ADR-7; extracting to numpy is a side quest), zero published metrics, unmaintained since 2023-05, and its cheapness evaporates unless the control model wins |
| 5 | NudeNet v3 (320n/640m ONNX detector) | MIT | ONNX, no TF since v3 — genuinely light | standalone; gives boxes, which we do not need | no published held-out metrics found | rejected: no citable evaluation. Its box output is a real future asset for explainability |
| 6 | GantMan/nsfw_model | MIT | Keras/TF, InceptionV3/MobileNetV2 lineage | TF at runtime = ADR-7 violation | 2019-era | rejected |
| 7 | **Zero-shot prompt ensemble over our own embeddings** (the free baseline) | n/a | **exactly 0** — one text batch, one dot product | perfect | **measured here, and it fails** — see §4 | rejected as the instrument; kept as the offline fallback + the baseline of record |

Not evaluated: `AdamCodd/vit-base-nsfw-detector` (Apache-2.0, 0.9654 acc / 0.9948 AUC,
ViT-base @384 — dominated by #1 on both size and published accuracy).

---

## 2. Chosen design

**A dedicated classifier, not the index embeddings.** Weapons and drugs can ride the
embeddings the index already computed, because "a rifle" is an object CLIP was trained to
name. "Nudity vs a swimsuit" is a *boundary*, and §4 shows measurably that the same
embedding does not carry it. Recall-first enforcement on a public site needs an
instrument with citable metrics; a prompt ensemble has none and — because the EVAL DATA
LAW forbids fetching positives — can never be honestly calibrated on this machine.

- Artifact: `models/moderation/nudity-marqo-384.onnx`, 22.5MB fp32, **self-exported**
  by `scripts/export_nudity_marqo.py` in a throwaway venv (torch/timm never enter the
  runtime env — ADR-7/B23 intact). Export is trusted by measurement, not assumption:
  **max |torch − ORT| = 9.8e-07** on a random batch, printed by the script on every run.
  Weights are inlined into the single `.onnx` (torch 2.13's dynamo exporter externalises
  them by default — a sha256 over the 80KB graph alone would have proved nothing).
- Preprocess is the model's own config, as data, never folklore: 384², **bicubic**,
  `crop_pct 1.0` + `crop_mode center` (= resize shortest edge → centre-crop),
  mean = std = 0.5. Implemented by *reusing* `core.models.preprocess_image(im, 384,
  squash=False, BICUBIC)` — the EXIF/`draft()`-aware path the engine already owns.
- Label order is a trap and is pinned in code: `config.json` `label_names ==
  ["NSFW", "SFW"]`, i.e. **index 0 is NSFW**, not alphabetical. A regression test asserts
  flat colour images score low, which is exactly what fails if this flips.
- **No new runtime dependency.** onnxruntime + numpy + Pillow, all already in ADR-7.

Code: `src/imgtag/moderation/nudity.py` · tests: `tests/test_nudity.py` (20, all green).

---

## 3. Measured — the false-positive side (first-party)

Corpus: **1,826 unique safe images** — COCO val2017 (500) + Unsplash slices built by
joining `data/unsplash/keywords.tsv000` to the images actually on disk. The "hard" slices
are where a nudity detector is most likely to fire wrongly. Harness:
`research/bench_scripts/nudity_eval.py` (seed 20260722).

Score distribution over all 1,826: mean 0.0595 · p50 0.0532 · p99 0.1754 · p99.9 0.6983.

**Flag rate vs threshold** (every flag here is a false positive, except in the
swimwear/lingerie/underwear slices where a strict site rule may legitimately call some of
them review-tier — so those columns are an *upper bound* on FP, not FP):

| τ | all (1826) | landscape | architecture | car | food | beach | coco-val | portrait | sculpture | bikini | swimwear | lingerie | underwear | baby | child |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 66.3% | 60.0% | 62.5% | 73.2% | 71.5% | 53.0% | 71.6% | 62.0% | 68.2% | 70.5% | 73.2% | 76.9% | 71.0% | 58.1% | 71.1% |
| **0.10** | **3.07%** | 0.00% | 1.00% | 0.51% | 3.50% | 1.50% | 6.00% | 4.50% | 2.27% | 6.82% | 7.04% | 3.85% | 6.45% | 6.45% | 6.02% |
| 0.15 | 1.42% | 0.00% | 1.00% | 0.00% | 1.50% | 0.00% | 2.60% | 3.00% | 0.00% | 2.27% | 4.23% | 3.85% | 6.45% | 6.45% | 4.82% |
| 0.20 | 0.82% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 1.80% | 2.00% | 0.00% | 2.27% | 2.82% | 3.85% | 6.45% | 6.45% | 3.61% |
| 0.30 | 0.44% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.80% | 1.50% | 0.00% | 0.00% | 0.00% | 0.00% | 3.23% | 6.45% | 2.41% |
| **0.50** | **0.22%** | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.40% | 1.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| 0.70 | 0.11% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.20% | 0.50% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| 0.90 | 0.00% | — | — | — | — | — | — | — | — | — | — | — | — | — | — |

The **all** column is over the 1,826 *unique* images; the per-slice columns are over the
2,229 slice memberships (an image with both `beach` and `bikini` keywords appears in both,
by design — slices are views, not a partition). Over slice memberships the τ=0.10 rate is
3.81% rather than 3.07%.

(τ=0.05 sits *inside* the SFW mass — the whole safe corpus piles up at ~0.053. It is in
the table only to show where the floor is; it is not a candidate operating point.)

*Landscape/architecture/car must be silent:* **landscape flags 0.00% at every τ ≥ 0.10**;
car 0.51%, architecture 1.00%.

### ADR-14's required negative: non-person figures (measured)

The team-lead named this "your detector's hardest required negative", after the v0
prompt-set scorer flagged **a nude mannequin at p=0.957**. Unsplash keyword slices,
scored by this head:

| slice | n | mean | max | ≥ 0.10 (review) | ≥ 0.50 (violation) |
|---|---|---|---|---|---|
| **mannequin** | 26 | 0.0918 | **0.4515** | 4 | **0** |
| statue | 68 | 0.0559 | 0.1172 | 2 | **0** |
| sculpture | 88 | 0.0559 | 0.1172 | 2 | **0** |
| marble | 48 | 0.0620 | 0.2108 | 2 | **0** |
| figurine | 55 | 0.0628 | 0.3563 | 3 | **0** |
| doll | 19 | 0.0722 | 0.3563 | 1 | **0** |
| torso | 10 | 0.1226 | 0.4515 | 2 | **0** |
| **union** | **314** | 0.0641 | **0.4515** | 16 (5.1%) | **0 (0.00%)** |

**Zero violation-tier flags across all 314 non-person figures**, worst case 0.4515 — the
v0 scorer's 0.957 mannequin becomes at most a review-tier hit here. The residual 5.1%
review rate on this class is the honest cost of a recall-first review band; it is the
class most worth re-checking if the operator ever supplies labeled data.

### The false-positive tail, characterised

Top scorers, described from the corpora's **own** caption metadata (COCO
`captions_val2017.json`, Unsplash `ai_description`) — no image inspection needed:

| p | slice | what it actually is |
|---|---|---|
| 0.8247 | coco-val2017 | *"A woman holding a baby next to a bird cage."* |
| 0.8019 | portrait | *"beige sea creature underwater photography"* — **a dugong** |
| 0.6763 | coco-val2017 | *"Many apples and oranges are stacked near to each other."* |
| 0.5613 | portrait | *"time lapse photography of water hitting left palm"* |
| 0.4911 | coco-val2017 | *"it looks like a hand of a small child with pink and white top"* |
| 0.3988 | coco-val2017 | *"a bathroom with a tub and a shower curtain"* |
| 0.3671 / 0.3563 | baby | *"grayscale photo of woman lying on bed"* · *"Dog and baby watch the outside world together."* |
| 0.2805 | bikini | *"Hands draw patterns in the beach sand."* |
| 0.27–0.18 | coco-val2017 | bananas, toilets, giraffes |

**The FP class is smooth flesh-toned organic surface** — skin close-ups, babies, bathroom
porcelain, ripe fruit, and one marine mammal. It is *not* people-in-swimwear, which is the
reassuring result: the head is not simply a skin-tone detector. It is also the reason the
review queue must show the image to a human and never auto-act — 4 images in 1,826 crossed
0.50 and **all four were benign**.

---

## 4. Measured — why the free zero-shot baseline was rejected

Same 1,826 images, same harness (`--zeroshot`), prompt-ensemble margin (max positive-prompt
cosine − max background-prompt cosine) over `openclip-vitb32` embeddings:

| slice | mean margin | | slice | mean margin |
|---|---|---|---|---|
| coco-val2017 | **−0.0132** | | bikini | −0.0254 |
| food | **−0.0191** | | swimwear | −0.0258 |
| tattoo | −0.0191 | | sculpture | −0.0269 |
| **child** | **−0.0201** | | yoga | −0.0289 |
| lingerie | −0.0222 | | architecture | −0.0325 |
| shower | −0.0231 | | beach | −0.0362 |
| underwear | −0.0239 | | landscape | −0.0388 |
| baby | −0.0242 | | | |

**The ordering is wrong in the way that matters.** Food, tattoos and *children* rank above
lingerie, underwear and bikini. At a τ that flags any positive margin, coco-val2017 flags
17.8% while swimwear flags 1.41% and bikini 2.27% — the baseline is anti-correlated with
the thing it is supposed to find. Compare the dedicated head at τ=0.10: bikini 6.82%,
swimwear 7.04%, underwear 6.45% vs landscape **0.00%**, car 0.51%, architecture 1.00% —
the correct ordering.

This is the falsification, on our own corpus, of "moderation can just score the embeddings
the index already computed" **for this category**. It says nothing about weapons or drugs,
where the target is a nameable object.

The zero-shot path is kept in `nudity.py` as `ZeroShotNudityHead`: the offline fallback
when the artifact is absent, permanently marked `calibrated: false`, and structurally
incapable of emitting a `violation` tier.

---

## 5. Thresholds and the rationale

| tier | τ | measured flag rate on the safe corpus | why this number |
|---|---|---|---|
| `violation` | **0.50** | 0.22% (4 / 1826) | The model's **own argmax point** — the *only* threshold the published 98.56% / 20k evaluation actually describes. Any other τ would be a number with no evidence attached to its recall side. |
| `review` | **0.10** | 3.07% (56 / 1826) | Recall-first extension *below* the citable point, sited just above the SFW mass: the safe-corpus p95 is 0.07 and p99 is 0.175, so 0.10 is the first threshold that clears the bulk without landing in the tail. |
| `none` | < 0.10 | 96.9% | |

Dialling, for the operator: τ_review 0.15 → 1.42%, 0.20 → 0.82%, 0.30 → 0.44%. Overridable
per install without a code change via `profile["nudity_tau_review"]` /
`["nudity_tau_violation"]`, or a `nudity-marqo-384.calib.json` sidecar next to the artifact.

**`calibrated` is hard-coded `False` and `enforcement_ready` stays false for this category.**
τ was never fitted against labeled nudity ground truth — it *cannot* be on this machine —
so per ADR-14 this track produces review signal, never an enforcement decision.

### Published true-positive metrics (cited, NOT reproduced here)

> Marqo/nsfw-image-detection-384 model card, huggingface.co/Marqo/nsfw-image-detection-384
> (Apache-2.0, retrieved 2026-07-22): *"achieves a superior accuracy of **98.56%** on our
> dataset"* — proprietary, 220k images (100k NSFW / 100k SFW train; **10k NSFW / 10k SFW
> test**), spanning "real photos, drawings, Rule 34 material, memes, and AI-generated
> images". Precision/recall curves and a threshold sweep are published as figures
> (`images/PrecisionRecallCurves.png`, `images/ThresholdEvals.png`); the model card gives
> **no recommended threshold** and warns the definition of NSFW is contextual.

That 98.56% is measured at the argmax point, which is why `violation` sits there. **We did
not reproduce it, we cannot reproduce it here, and no recall number in this project may be
attributed to first-party measurement until labeled ground truth exists on the target host.**

---

## 6. Cost (⚠️ UNRELIABLE — machine was 2.6× oversubscribed)

Forward-only, random 384² input, this Mac, **1-min load average 41.5 on 16 cores**. ORACLE's
bench-honesty rule (refuse/mark above cores × 0.6 = 9.6) applies: **these rows are
UNRELIABLE and must be re-measured**, on an idle machine and then on the 🐧 target.

| intra_op | batch 1 | batch 2 | batch 8 |
|---|---|---|---|
| 1 | 140.3 ms | 168.4 ms | 153.6 ms |
| **2** | 98.6 ms | **89.3 ms** | 96.0 ms |
| 4 | 136.5 ms | 117.9 ms | 102.7 ms |

Load-independent facts: 4.5 GFLOPs/img, ~¼ of PE-Core-S16-384's vision forward, 22.5MB of
weights, batch size does not matter (consistent with ADR's batch-1/2 streaming finding).
Expect the moderation pass to add roughly a quarter to index wall-clock — **bounded, and
the price of an instrument that can actually answer the question**.

---

## 6b. The 100-track scaling invariant (VISION-ADDENDA 13:26Z) — where this track stands

> Verbatim: *"for each track … i want a confidence score for each track for every image …
> even if we have 100 tracks, the times for indexing and inferencing should remain
> relatively the same so this system can continue to scale."*

**Confidence for every image: satisfied.** Every record this head returns carries a `p`,
including `content_free` and `unreadable` records — nothing is silently absent.

**Scaling invariant: this track is the deliberate, bounded exception, and that is a design
fact the conductor needs, not a defect to hide.** The invariant holds *by construction* for
tracks that ride the shared embedding: the image is embedded **once**, and each such track
adds one `[N,D]·[D,k]` matmul (~0 — weapons and drugs are exactly this). 100 embedding-head
tracks ≈ the cost of 1. **Nudity cannot be one of them**, for two measured reasons:
(§4) the CLIP embedding provably does not separate nudity from swimwear/skin, and (EVAL DATA
LAW) a *trained* embedding head — the weapons approach — cannot be fitted or validated here
because no positive corpus may be fetched. So nudity pays one dedicated forward
(~4.5 GFLOPs, ~¼ of the index model).

The invariant therefore reads, honestly: **`t_index ≈ t_embed + Σ_tracks t_track`, where
embedding-head tracks contribute ~0 and dedicated-model tracks each contribute a bounded
constant.** 100 tracks stay flat *only if* almost all of them are embedding heads. The
architectural rule this implies (conductor's to enforce): **a track earns a dedicated model
only when the shared embedding provably cannot carry its signal** — nudity clears that bar;
most future tracks will not, and should be linear probes on the one embedding. If a
future host is fast enough that even a handful of dedicated heads blow the budget, the
escape hatch is a shared multi-head backbone (one small ViT, N linear classifier heads) —
noted here so it is not rediscovered.

## 7. Integration notes for b-engine

`imgtag.moderation.load_heads(profile)` already finds this track. Contract as implemented:

- `load_nudity_head(profile) -> NudityHead | None` — **None** when the artifact is absent
  (gitignored). A missing track is simply not loaded; it never returns a silent zero.
- `NudityHead.wants_images = True`.
- `NudityHead.score(embeddings, images, ids) -> list[dict]`, one dict per record:
  `{"category": "nudity", "p": float, "tier": "violation"|"review"|"none",
    "model_id": "marqo-nsfw-384-fp32", "calibrated": False}`, plus `"content_free": True`
  on records the §9 guard set aside and `"unreadable": True` on files that would not open.
  Both are additive markers — `category`/`p`/`tier` are always present, so the API, the UI
  and the counts need no change to consume them.
  `embeddings` are **ignored** — this track answers from pixels, which is why it can answer.
- **Pre-made view (landed 2026-07-22, commit `8cd2778`).** `score()` accepts an optional
  `views=` kwarg and uses `views["nudity-384crop"]` when offered. The worker builds it by
  calling **our** `nudity.make_view(im)` — one implementation, so transform drift is
  impossible by construction rather than policed by a test. The head declares
  `view_key` + `view_geometry`. Offered views are shape/dtype-checked and fall back to
  re-open if malformed. The old 3-positional call site is unchanged.
  **PRECONDITION, measured — `draft()` sets the JPEG DCT decode scale and that scale is
  part of the pixels.** Same transform, different decode state: drafted (384,384) →
  **bit-identical**; drafted (224,224) first → differs **max 33/255**; full decode, no
  draft → differs max 9/255. So a view may be shipped ONLY when the worker's decode was
  drafted at 384, i.e. the backend's size is 384 (`pecore-s16-384`, the default) — verified
  on 4 images that one decode then serves both bit-identically. **Forcing it for smaller
  backends is not free and must not be done:** drafting at 384 to serve this track perturbs
  a 224-backend's OWN embeddings by up to **83/255** — corrupting the index to speed up
  moderation. Not-384 backend → ship no view, this track re-opens, calibration holds.
- **Pixel geometry — the one thing to get right.** The coordinator's slab carries the
  *backend's* geometry (squashed, sometimes 224²). That is a domain shift this model was
  never trained for, so unless the slab is already 384² **and** the backend squashes, the
  head re-opens the file from `rec["path"]` and preprocesses properly. `draft()` makes that
  a partial JPEG decode, not a full one. Under `geometry="worker"` (`images is None`) this
  is the only available path and it works unchanged.
  → *If b-engine can cheaply hand the head a second 384-shortest-crop uint8 view from the
  decode worker (one extra resize off the already-open PIL image, no second decode), that
  is strictly better and the head will take it — say the word and the slab fast path
  widens to accept it.*
- Unreadable file → `{"tier": "none", "unreadable": True}`, never an exception. Verified by
  test.
- Batch-summary plumbing needs nothing from this track: tiers and counts already flow
  through `_apply_moderation` / `moderation_summary`.
- Search/app side: `p` is always present, so "sort the gallery by nudity score" works
  without re-running anything.

## 8. Provenance, escalations, honest verification status

**Escalated (sent to team-lead 15:50Z):** the original `moderation/__init__.py` asserted all
three tracks score the index embeddings; nudity does not. The conductor has since rewritten
that file to describe both instruments — resolved.

**Divergence from the track brief, deliberate:** the brief specified output
`{category, p, flagged}`. The landed and tested seam (`tests/test_meta_moderation.py`,
ADR-14) uses `{category, p, tier}`. The landed contract wins; `flagged` is not emitted
(`_apply_moderation` maps a legacy `flagged` to `violation`, which would silently promote
review-tier hits — emitting it would be worse than useless).

**Verification status (honest):**
- ✅ ONNX export matches torch (9.8e-07) — measured, reproducible by re-running the script.
- ✅ False-positive behaviour on 1,826 safe images — measured, first-party, reproducible.
- ✅ ADR-14 statue/mannequin and landscape boundaries — measured, satisfied.
- ✅ Zero-shot baseline is unfit for this category — measured, first-party.
- ✅ 20 tests green (`tests/test_nudity.py`), incl. the label-order and geometry traps and
  the permanent content-free negative control.
- ✅ ADR-14's required negative measured: **0 violation flags across 314 non-person figures**
  (mannequin/statue/doll/figurine/marble/torso), worst case 0.4515 — vs the v0 scorer's
  0.957 mannequin.
- ✅ Preprocessing proven correct against timm's own transform (bit-identical tensors);
  the solid-colour FP is the model's OOD colour prior, not our bug (§9).
- ❌ **True-positive recall — NOT verified here.** Published metric only, cited above.
- ❌ **τ not fitted on labeled ground truth** — `calibrated: False`, `enforcement_ready` false.
- ⚠️ **Latency UNRELIABLE** (load 41.5 / 16 cores). Re-measure idle, then on the 🐧 target.
- ✅ **End-to-end through the real CLI** — `imgtag index … --moderation` on 4 files loads the
  head via `load_heads`, and `imgtag info --flags --json` returns
  `p = 0.8247 / 0.8019 / 0.6763` for the three FP-tail images — **bit-identical to the
  offline bench**, which proves the re-open pixel path in production matches the measured
  one. `calibration: "unfitted"` and `enforcement_ready: false` propagate to the rollup.
  (Those three files were chosen *because* they are the known FP tail; three violations is
  the harness working, not a quality claim.)
- ⚠️ Two tests in b-engine's `tests/test_meta_moderation.py` fail on this checkout
  (`test_summary_uses_the_users_phrasing`, `test_cli_meta_flags_rollup_and_dataset_meta`) —
  their own expectation vs their own `moderation_summary` wording, mid-flight in that lane.
  Not touched (F2); reported to team-lead.

**Next, in order:** (1) re-measure §6 idle and on the target host; (2) if the operator can
supply a *labeled, lawfully-held* in-house sample on the target machine, fit τ there and only
then flip `enforcement_ready`; (3) revisit Freepik/EVA-02 (severity levels map onto ADR-14
tiers directly) once target-host throughput is known.

---

## 9. The out-of-distribution colour prior, and the content-free guard

**Reported by b-engine:** 6 synthetic solid-colour JPEGs wired through this head produced
4 nudity flags at p = 0.34–0.41. Investigated, reproduced, and worse than reported.

**Measured.** Sweeping the RGB cube (125 solid colours, 384², n=125):
mean **0.273**, 99% score ≥ 0.10, 4% score ≥ 0.50. Worst cases:

| probe | raw p |
|---|---|
| solid flesh tone (222,180,150) | **0.5498** — violation tier, from an empty frame |
| solid (255,192,255) | 0.5480 |
| **flesh-toned linear gradient** | **0.7612** |
| solid black | 0.3590 · solid white 0.2040 · solid grey 0.2100 |
| solid green (0,255,0) | 0.0840 (the floor) |
| uniform noise | 0.1259 · gaussian noise 0.0927 · checkerboard 0.0553 |

The ordering is by **colour**, not content: pinks and flesh tones at the top, greens and
blues at the bottom. On content-free input the model degenerates to a colour prior.

**It is not our preprocessing** — this was the prime suspect and it was eliminated by
direct comparison against timm's own transform in the export venv:

| image | max &#124;our tensor − timm tensor&#124; | p (ours) | p (timm tensor → ONNX) | p (timm tensor → torch) |
|---|---|---|---|---|
| coco 000000185599 | **0.0000** | 0.6763 | 0.6763 | 0.6763 |
| solid flesh 512² | **0.0000** | 0.5498 | 0.5498 | 0.5498 |
| coco 000000463618 | 1.051 | 0.8247 | 0.8257 | 0.8257 |
| unsplash 5L47XYRvGOo | 0.235 | 0.8019 | 0.8168 | 0.8168 |

Bit-identical where the image needs no EXIF rotation and no `draft()` rescale; where
`draft()` (the engine's partial-JPEG-decode speed win) does apply, it costs **≤ 0.015 p**.
The published 98.56% simply never covered content-free input — this is the model's own
out-of-distribution behaviour.

**The guard.** `structure()` = mean |discrete Laplacian| of the preprocessed frame.
Second-order deliberately: a solid colour *and* a linear gradient both have zero second
derivative, while every photograph has texture.

| | measured |
|---|---|
| 1,826 real photographs | min **1.171** · p0.1 1.413 · p1 2.396 · p50 14.68 |
| solid colour | 0.000 |
| linear gradient | 0.667 · flesh gradient 0.637 · radial gradient 0.994 |
| white noise | 168.2 |

`MIN_STRUCTURE = 1.0` sits in the gap — above every synthetic probe, **1.171× below the
lowest real photograph in the corpus**. Below the floor a record is re-**tiered** to
`none` and marked `content_free: true`; **p is still reported and nothing is dropped**, so
an operator can always query what was set aside. **Zero of the 1,826 real images are
affected.** Cost is two second-difference passes over 384² — sub-millisecond against a
~90ms forward. Overridable via `profile["nudity_min_structure"]`.

**Two honest limits, neither hidden:**
- **White noise is NOT claimed by the guard** — it has genuine spatial structure — and the
  model puts it at p≈0.16, i.e. *review* tier. That is by design: review is a human queue,
  and moving τ_review above 0.16 to win against an input no camera produces would cost
  real recall on the borderline cases the tier exists for. The permanent negative-control
  test asserts the hard line (**nothing content-free may ever reach violation**) and full
  silence on the solid/gradient class that actually breached it.
- **Residual recall hole:** an image deliberately smoothed below structure 1.0 is set
  aside. Bounded by the fact that such an image carries almost no visual information —
  and it is recorded, not deleted.

**A note on the review tier and swimwear.** ADR-14 puts swimwear/lingerie at review tier.
This head is measured *not* to be a good swimwear detector — Marqo trained swimwear as SFW,
so bikini tops out at 0.28 and the whole class sits in the 0.05–0.15 band. The tag path
(b-daemon's `Searcher.track_scores`, whose vocabulary *does* carry swimsuit/bra/underwear)
is the better instrument for that tier, and the two compose: **violation from this head,
swimwear-review from the tag path**. This head's own review band should be read as
"possible nudity, look at it", not as "swimwear".

---

## 10. Datasets & sub-category labeling (user 2026-07-22: "no true positives and no sub-category labeling")

> **READER MAP (merge ruling 2026-07-22).** This §10 is the **VIOLATION branch** of the one
> nudity taxonomy tree — anatomy-region subcategory *labels* via the NudeNet body-part
> detector, run as an explainer on Marqo-gated images. §11 is the **REVIEW branch** —
> clothing-context confidence *separation* + the ratio *threshold* + the indexed TP dataset.
> §11.6 has the unified two-branch table. Both are versioned data in the `moderation.json`
> nudity entry (`violation_subcategories` + `review_subcategories`).

Both gaps are real; the TP one was raised as honest-gap #1. They are DISTINCT problems
and one is architectural.

### 10.1 Sub-category labeling needs a second instrument — the binary head cannot do it

The Marqo head is a **2-class NSFW/SFW** classifier. It emits one probability and
**structurally cannot name a sub-category** ("exposed breast" vs "covered buttocks" vs
"swimwear"). Sub-category labels need a detector with a body-part vocabulary. Added:
**NudeNet v3 320n** (`deepghs/nudenet_onnx`, **Apache-2.0**, YOLOv8 @320, 12MB ONNX),
`src/imgtag/moderation/nudity_subcat.py`. 18 anatomical classes, each exposed/covered,
mapped to ADR-14 tiers **as data** (`CLASS_TIER`): exposed genital/anus/breast/buttocks →
`violation`; covered-but-revealing → `review`; face/feet → `none`. NMS is ~15 lines of
numpy (the shipped `nms-yolov8.onnx` is not needed — one fewer graph). No new runtime dep.

**It is an EXPLAINER, run only on already-flagged images — NOT a standalone gate.** Two
measured reasons:
- **NudeNet has no concept of statue/mannequin.** On 40 CC0 nude marble sculptures it fired
  **violation on 28%** (FEMALE_GENITALIA_EXPOSED ×7) — it detects anatomy in marble. That
  is the ADR-14 hard-negative failure in the open. The Marqo head, by contrast, flags **0%
  violation** on photographic statues. So the gate stays Marqo (+ its content-free guard);
  NudeNet adds the *why* on images already flagged, and its statue-blindness is documented,
  not shipped as a decision.
- Running a detector on every image would break the scaling invariant (§6b). Gated behind a
  flag, the common (unflagged) image pays only the Marqo forward.

**Sub-category labeling works and is measured** on lawful non-explicit images. NudeNet on
50 Unsplash swimwear/bikini photos: `FEMALE_BREAST_COVERED` ×7 (review), `ARMPITS_EXPOSED`,
`BELLY_EXPOSED`, and one `MALE_GENITALIA_EXPOSED` false positive (a swimsuit at score 0.25).
Content-free probes: **0 detections** — the detector is immune to the colour prior that
fools the binary head (§9), so it needs no structure guard of its own.

Per-image sub-category payload (`NudeNetSubcategoryHead.label`):
`{category:"nudity", tier, subcategories:[{class,p}], detections:[{class,tier,p,box}]}` —
`tier` composes into the same ADR-14 counts; `subcategories` is the human-readable
explanation; `box` supports later blur/redact.

### 10.2 True positives — a lawful non-explicit proxy now, the real set escalated

**EVAL DATA LAW still binds: no explicit-adult corpus was fetched to this machine.** What
was fetchable lawfully and safely: **112 CC0 public-domain works from the Met Open Access
API** (`scripts/fetch_art_proxy.py`, reproducible) — 72 classical nudes + 40 nude
sculptures. Adult art only, non-explicit, non-photographic. First positive-side signal on
the nudity boundary:

| set (n) | Marqo recall@0.50 | Marqo @0.10 | NudeNet violation | note |
|---|---|---|---|---|
| painted/sculpted nudes (72) | 17% | 42% | 10% | **weak — painting is a domain shift from photos** |
| nude marble statues (40) | 10% | 28% | **28%** | ADR-14 = NO FLAG; NudeNet fails it, Marqo weakly fires too on explicit classical nudity |

**This is a PROXY and under-reads real-photo recall** — both models trained on photographs,
so painted nudity is out of distribution and the 17–42% is a *floor*, not the recall a
photographic set would show. It is honest positive-side signal, not a recall claim. The
Met statue result also sharpens ADR-14: full-frontal classical *sculpture* is exactly where
"non-person figure = no flag" is hardest, and it is where a future statue/mannequin
discriminator earns its place (both instruments want it).

**Review-tier TRUE positives, measured on lawful data now:** the 197 Unsplash
swimwear/bikini/lingerie/underwear/bra/panties images ARE ADR-14 review positives. Marqo's
recall on them at τ_review=0.10 is **6.1%** — confirming quantitatively that the binary head
is the wrong review-tier instrument and the tag path owns that tier (§7 note). NudeNet's
`*_COVERED` classes are the better review signal and are now available.

### 10.3 What still cannot be closed here — ESCALATED (ORACLE §7, rule 2)

1. **Photographic violation-tier recall.** Needs a lawfully-held, labeled photographic
   nudity set. That decision is not mine and must not be a casual fetch — the baby/child
   slices scoring non-zero (§3) make any explicit-set fetch CSAM-adjacent and unacceptable
   without an explicit, lawful, adult-only provenance. **Recommended path (unchanged):** the
   operator supplies an in-house labeled sample **on the target host**, τ is fitted there,
   and only then does `enforcement_ready` flip. Alternative: a research-DUA academic
   benchmark, on the target host, never this Mac.
2. **Policy boundary: is depicted (painted/drawn) nudity a `violation`, or `none` like a
   statue?** ADR-14 rules non-person *figures* no-flag; it does not rule on art *of* people.
   The proxy is only interpretable once this is ruled. Question routed to the user.

### 10.4 Files added this round

`src/imgtag/moderation/nudity_subcat.py` · `tests/test_nudity_subcat.py` (7 tests) ·
`scripts/fetch_art_proxy.py`. NudeNet artifact + art proxy are gitignored (models/**/*.onnx,
/data/); regen from `scripts/fetch_art_proxy.py`. (The §10.2 proxy numbers were produced by an
inline eval, not a committed script; re-derivable from the fetch script + `nudity_subcat.py`.)

---

## 11. Review-tier TP probe + confidence separation + ratio-threshold (track-nudity3, 2026-07-22)

> Mandate: VISION-ADDENDA 13:58Z (verbatim) — *"i dont see true positives in the datasets and
> all findings are false positives … whatever is scored now should … be at lower confidence
> results compared to the true positive results and then we will be able to set the ratio
> threshold for auto flagging for each track"*. This round answers the **confidence-separation
> + ratio-threshold** half of the directive, and delivers a **real indexed TP dataset** the
> user can feel-test. It COMPOSES with §10 (which answers the subcategory-*labeling* half via
> the NudeNet body-part explainer) — different instruments, different halves, no runtime
> overlap. **EVAL DATA LAW obeyed: only review-tier (non-explicit) TPs, keyword-sourced from
> the local Unsplash corpus; violation-tier recall stays benchmark-cited, never re-measured.**

### 11.1 The deliverable that directly fixes "no TPs to test": `nudityprobe`

`data/nudity-probe/` — **202 local Unsplash review-tier positives, assembled PROGRAMMATICALLY
by keyword-join (T4: no agent hand-labelling; ≤20 spot-views only)**, indexed as a real
dataset (`imgtag index data/nudity-probe nudityprobe --wait --moderation`). Subcategories &
counts: swimwear 77 · lingerie 16 · underwear 4 · bare-chest-male 45 (noisy `fitness` proxy) ·
**mannequin-statue control 60** (never-flag). Symlinked, gitignored, never redistributed.
Builder+harness: `research/bench_scripts/nudity_probe.py` (seed 20260722, re-runnable).

**This reproduced the user's complaint as the honest BEFORE picture:** on a dataset that is
100% review-tier TP, the current Marqo-only runtime flagged only **13/202 review, 0 violation**
— and those 13 are the §9 colour-prior FPs, *not* the swimwear. The track genuinely could not
be tested until this corpus existed.

### 11.2 The review tier is a MARGIN over the index embeddings — NOT the Marqo head

Confirmed, first-party, on the probe: the Marqo p **cannot separate the review tier** —
swimwear p50 **0.056** == mannequin p50 **0.056** == the FP corpus band. So review is served
by a prompt-ensemble margin over the pecore-s16-384 embedding the index already computed
(TRACKS T1, ~0 extra cost): `margin = max(review concepts) − max(negatives incl. the ADR-14
never-flag figures)`. Instrument choice was **measured**: full-negatives (mannequin/statue in
the subtracted bank) keeps the ADR-14 control tightest — generic-only negatives raised the
control leak 0.07→0.12, so the never-flag law wins over marginal recall.

### 11.3 Confidence separation, per subcategory (FP band = 14,734 indexed safe images)

FP-band review-margin: p50 **−0.021** · p90 +0.019 · p99 +0.066.

| subcategory | n | margin p50 | R@fpr5% | R@fpr1% | AP (CI95) | verdict |
|---|---|---|---|---|---|---|
| **swimwear** | 77 | **+0.045** | **0.571** | 0.299 | 0.111 (0.063–0.18) | **SEPARATES — usable recall-first review tier** |
| lingerie | 16 | −0.035 | 0.00 | 0.00 | 0.001 | does NOT separate on shared embedding |
| underwear | 4 | −0.000 | 0.00 | 0.00 | 0.001 | does NOT separate (also n=4, too thin) |
| bare-chest-male | 45 | −0.007 | 0.133 | 0.022 | 0.007 | does NOT separate (noisy `fitness` proxy) |
| **mannequin-statue** (control) | 60 | −0.016 | 0.067 | **0.000** | — | **CONTROL PASS — below FP median, 0 flags at τ₁%** |

End-to-end via the real scorer (`load_nudity_review_scorer`, reading the taxonomy from
`moderation.json`): swimwear **22/30** review-tier vs mannequin control **2/30** — the
elevation the Marqo-only runtime could not produce.

### 11.4 Proposed ratio thresholds ("auto-flag τ"), per tier

- **violation** — Marqo head, **τ=0.50** (its argmax, the only point the published 98.56%/20k
  evaluation describes). NOT locally re-measured (EVAL DATA LAW). Honest margin: the safe-corpus
  Marqo **p99 is 0.175**, so 0.50 sits far above the FP band. `enforcement_ready` stays false.
- **review (swimwear)** — margin instrument, recall-first (ADR-14): **τ_review = 0.0339** (5% FP
  band → R@5%=0.57) or **0.0658** (1% FP band → R@1%=0.30). Platt [28.96, −4.53] → confidence p.
  Overridable per install via `profile["nudity_tau_review"]`. Dial as the operator's queue
  budget allows.

### 11.5 Honest limits (headlined per directive: "if separation fails anywhere, headline it")

- **Review recall is swimwear-ONLY.** lingerie / underwear / bare-chest do **not** separate on
  the shared pecore embedding (R@5% ≤ 0.13) — the same lesson as §4's violation-tier
  falsification, now measured for these review subcats. Each **earns a distilled head** (TRACKS
  T2) or a dedicated instrument; recorded as darwin item **D-nudity-review-distill**. NudeNet's
  `*_COVERED` classes (§10) are the immediately-available better signal for these.
- **Violation-tier recall is benchmark-cited, never locally reproduced** — and cannot be, on
  this machine, under the EVAL DATA LAW. That honesty is the deliverable, not a gap.
- **Small positive sets** (swimwear 77, lingerie 16, underwear 4) ⇒ wide CI; the fit is a
  review-queue calibration, not an enforcement claim.
- **Composition wiring is a flagged NEXT STEP, not silently shipped.** `NudityReviewScorer`
  exists as a tested, calibratable instrument, but is deliberately kept OUT of the auto-loaded
  `load_nudity_head` runtime composition until a subcategory clears the bar or a distilled head
  is trained (TRACKS T2 earn-it rule). The Marqo violation path is untouched. Handoff to
  b-engine: compose `violation←Marqo(pixels)` + `review←NudityReviewScorer(embeddings)` in the
  index-time head when green-lit (the embeddings already flow into `score()`).

### 11.6 The unified taxonomy tree — §10 and §11 are ONE story (conductor merge ruling 2026-07-22)

§10 and §11 were built concurrently by two nudity-lane agents (§10 = the original track-nudity,
resurrected by a stray sibling message then stopped; field-logged 14:18Z). **RULING: compose,
don't discard — one lane owner, ONE taxonomy tree, two branches, versioned data in the
`moderation.json` nudity entry:**

| branch | subcategories | instrument | where |
|---|---|---|---|
| **REVIEW** (clothing-context) | swimwear · lingerie · underwear · bare-chest-male · artistic-figure · medical-context | prompt-ensemble **margin** over index embeddings (`review_subcategories`) | §11 |
| **VIOLATION** (anatomy-region) | exposed genitalia · exposed breast · exposed buttocks/anus (+ COVERED classes → review) | **NudeNet v3** body-part detector, run as an EXPLAINER on Marqo-gated images (`violation_subcategories`) | §10 |
| **gate + control** | — | **Marqo head** (pixels) gates violation; §9 content-free guard; mannequin/statue never-flag control | §5, §9, §11 |

They share no runtime path by design (NudeNet uses `CLASS_TIER` constants and runs only on
flagged images; the margin reads `moderation.json.review_subcategories`; Marqo is the gate).
**The composition SEAM — `violation ← Marqo(pixels)` gate + NudeNet labels, `review ←
NudityReviewScorer(margin over embeddings)` — is a flagged HANDOFF to b-engine, not wired
here** (the embeddings and the flagged-image trigger already exist in the pipeline; the
conductor is routing the seam).

Files this lane owns (merged): `nudity.py` (`NudityReviewScorer`) ·
`nudity_subcat.py` (NudeNet head, adopted with credit) · `moderation.json` nudity entry
(taxonomy tree + fit) · `tests/test_nudity.py` (+5) · `tests/test_nudity_subcat.py` (7) ·
`scripts/fetch_art_proxy.py` (Met CC0 proxy) · `research/bench_scripts/nudity_probe.py` ·
`research/eval-nudity-probe.json` · `data/nudity-probe/` + `nsfwprobe` slot (both gitignored).
Darwin item **D-nudity-review-distill** is cited in `DARWIN.md`.

### 11.7 Violation-tier: the operator-supplied path (`nsfwprobe`)

Violation-tier recall is the one number the EVAL DATA LAW forbids the agent from producing
(no explicit corpus is ever fetched). The **sanctioned** way it gets measured first-party: the
operator drops a **lawfully-held, adult-only** folder locally and indexes it —
`imgtag index <folder> nsfwprobe --wait --moderation` (local-only, gitignored, never
committed, never redistributed) — then `nudity_probe.py --nsfw-dataset nsfwprobe` measures the
Marqo head's violation recall at τ=0.50 vs the safe-corpus FP band **unchanged**. Until that
folder exists, τ_violation=0.50 stays benchmark-cited (98.56%/20k) and `enforcement_ready`
stays false. The harness slot is built and waiting; the agent supplies nothing.

> **WHERE I AM (2026-07-22, track-nudity3) — what's next.** DONE: `nudityprobe` (202 review-tier
> TPs) built + indexed; review-margin separation measured (swimwear SEPARATES R@5%=0.57;
> lingerie/underwear/bare-chest do not; mannequin control passes); τ proposed per tier;
> taxonomy + `NudityReviewScorer` shipped as data+code; 32 tests green; `nsfwprobe` operator
> slot wired. NEXT: (1) on ALL-CLEAR, re-verify idle; (2) if the operator supplies `nsfwprobe`,
> run `--nsfw-dataset` to get first-party violation recall; (3) darwin item **D-nudity-review-
> distill** — train a distilled head for lingerie/underwear/bare-chest (shared embedding
> cannot carry them); (4) conductor to dedupe §10 (NudeNet anatomy taxonomy) with §11
> (clothing-context taxonomy) into one composition.
