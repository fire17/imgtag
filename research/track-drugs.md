# track-drugs — the DRUGS moderation track (measured, with the ceiling stated)

> Lane: track-drugs · 2026-07-22 · model `pecore-s16-384-fp32` (the shipping default)
> Code: `src/imgtag/moderation/drugs.py` · eval: `scripts/eval_drugs.py` ·
> tests: `tests/test_drugs.py` · numbers: `research/eval-drugs.json` (machine-written)
> Binding: VISION-ADDENDA 12:33Z (three tracks) + 12:50Z rulings / ADR-14 (two tiers),
> ADR-3 (probability space, background-margin experiment), ADR-7 (no new deps).

> **CHECKPOINT 2026-07-22 ~17:40Z** — WHERE I AM: refit v2 shipped (4 routed defects fixed;
> committed 6a6a2e1/3ab7ce9; 18 tests green; live index reports 16 drug violations + 6
> review). improve-track round 1 PREPARED during the b-bench quiet window (no compute):
> confidence-correctness metrics written into eval (AUROC / ECE / Brier / Wilson-CI /
> per-subcategory separation), candidate subcategory taxonomy drafted
> (`data/drug-probe/taxonomy.json`, 7+staging subcats). WHAT'S NEXT (on ALL-CLEAR): A/B each
> taxonomy candidate for TP-vs-FP separation, promote only prompts that lift AUROC without
> lifting the FP band, refit, ledger entry, delete duplicate `drugprobe2`.

## 0. The one-paragraph answer

Drugs is the hardest of the three tracks because **the labels do not exist** and the
category is defined by context, not pixels. Measured on **15,010 deduped real-photo
negatives** (COCO + Unsplash-demo + Unsplash-b; the v2 refit corpus, ~2× b-app's) vs 17
hand-verified drug images: it ranks drug imagery at **AP 0.47 (recall .88 at a 1%
false-positive budget)**, and at the shipped ADR-14 tiering surfaces **94% of drug images**
(88% as violation, the rest as review) while calling **1.06% of ordinary photos a
violation** + 0.69% review — with **0 tobacco images wrongly called a violation**. The score
distribution is calibrated, not saturated: an evidence cap makes p≥0.95 unreachable (the
fix for b-app's "218 violations all at p=0.99"). End-to-end through the real pipeline on the
200-image drug-probe set: manifest reports **16 drug violations + 6 review**; both regression
images pass (vape→review, raspberry-leaf→none). *(The earlier v1 headline — AP 0.726 on 5k
COCO only, 18 "positives" — is retired: one of those 18 was a mislabelled bramble leaf, and
0.73 was corpus-inflated; §3a has the full before/after.)* What we could NOT measure —
recall on cocaine/heroin/meth imagery specifically, and any tobacco (review-tier) recall
worth shipping — is stated as unmeasured, not smoothed over. **Verdict: ship as a
recall-first REVIEW QUEUE, `enforcement_ready: false`.**

## 1. Approaches, ranked (what was tried, what won, what died)

| # | Approach | Verdict | Evidence |
|---|---|---|---|
| 1 | **Prompt-ensemble margin over the existing embeddings** — `max(drug concepts) − max(background concepts)`, 3 templates averaged per concept | **SHIPPED** | AP 0.726 on the drug slice; costs one matmul, zero new models, zero new deps (ADR-7 clean), runs on the 8GB target unchanged |
| 2 | Raw `max(drug concepts)` cosine, no subtraction | Close second — AP 0.7 band, within noise | kept as the fallback shape; the background margin wins on cross-corpus stability (absolute cosine drifts by model and corpus) and it is the experiment ADR-3 commissioned |
| 3 | Softmax over positive+negative prompts (classic CLIP zero-shot) | **Rejected** | AP 0.21 / 0.09 / 0.06 at T=50/100/200 — temperature-sharpened scores destroy ranking on a 1.4% base rate |
| 4 | Mean-vector form `mean(pos) − 0.5·mean(neg)` (what `search.py`'s generic track scorer does today) | Worse | AP 0.31 vs 0.57 for the margin on the same slice — see §6 integration note |
| 5 | **Margin against benign look-alikes** (pharmacy shelf, clinical syringe, prescription bottle) | **DEAD END — and the single most valuable finding** | AP **0.58 → 0.04**. A clinical syringe *is* a syringe: subtracting concepts that are visually identical to the positives subtracts the signal itself. Details in §3 |
| 6 | A trained logistic head on labelled positives (the weapons lane's primary path) | **Not possible** | 18 positives in existence. A head fitted on 18 images is a memoriser, not a classifier — refused rather than faked |
| 7 | Specialised open drug/paraphernalia detectors | **None adoptable** | see §5 |

## 2. What was measurable, and what was not — plainly

| Question | Status | Number |
|---|---|---|
| Does it find drug imagery (cannabis, bongs, joints, grinders, vape carts)? | **MEASURED** (18 hand-verified images) | AP 0.726 · R@1%FP **0.944** · R@5%FP 1.00 |
| How often does it cry wolf on ordinary photos? | **MEASURED** (5,145 negatives: COCO val2017 + hard Unsplash) | **0.92% violation** + 1.3% review at the shipped τ, after tier arbitration (§3b); 1.54% before it |
| Does it find cocaine / heroin / meth / pills-as-drugs? | **NOT MEASURED — no labelled image of any of these exists in a corpus we may use** | — |
| Does it find *incidental* paraphernalia (a cigarette in someone's hand, a syringe on a tray)? | **MEASURED, and it mostly does not** | AP 0.286 on 36 LVIS+OI images; recall 0.39 at the shipped τ |
| Tobacco / vape (ADR-14 review tier)? | **MEASURED, and the answer is bad** | recall **0.17** at a 1% FP budget. Shipped, flagged WEAK, `enforcement_ready: false` |
| End-to-end through the real pipeline? | **MEASURED** | `imgtag index data/drug-probe/strong --moderation` → *"Found … 18 images with drugs (6 for review)"* — 18/18 drug images surfaced, 1 false violation |

**The structural finding behind that table:** this detector sees **subjects, not objects**.
A photo *about* drugs scores; a scene that merely *contains* a 20-pixel cigarette does not
(AP 0.73 subject-centric vs 0.29 incidental — same model, same prompts, same corpus). That
is a property of whole-image CLIP-style embeddings, not of the prompt set, and no prompt
engineering moved it. Object-level recall needs an object-level instrument (a detector),
which ADR-9/ADR-7 and the 8GB target rule out for v1.

### Where the labels came from (and their honest weaknesses)

* **`drug` — 18 images, hand-verified.** Unsplash Lite metadata was mined for
  `cannabis/marijuana/bong/hemp/cbd/joint/drug/syringe` keywords → 200 candidates fetched
  per-id (33MB) → **every one inspected by eye** on contact sheets, then the top-scoring
  ones re-checked at full resolution. Only 18 of 200 are actually drug imagery: Unsplash's
  "weed" is mostly garden weeds and "hemp" is any leafy plant. Labels in
  `data/drug-probe/labels.json`.
  ⚠️ **Optimistic bias, stated:** two positives were found *because the model ranked them
  high* (a glass pipe and a grinder invisible at thumbnail size). Low-scoring missed
  positives were never re-reviewed at full resolution, so recall is an upper-ish estimate.
* **`proxy` — 36 images, human-labelled by others.** LVIS val2017 ashtray / cigarette /
  cigarette_case / matchbox / tobacco_pipe / medicine (26) + Open Images test `Syringe`
  (10, fetched per-image, 2.5MB). This is the *only* third-party ground truth that exists
  for this category on disk, and it is 36 images of mostly tiny objects.
* **`negatives` — 5,145.** All 5,000 COCO val2017 + the 174 non-drug images from the
  Unsplash keyword pull (ferns, houseplants, fields — exactly the hard negatives a naive
  "cannabis" prompt eats). ⚠️ LVIS is *federated*: an unlabelled image is not a verified
  negative, so the FP rate is an **upper bound**. The top 15 flagged negatives were
  inspected by hand and every one was a true false positive (§4).
* **`ambiguous` — 11 images**, scored and reported, never counted right or wrong.

*(Counts of the drug-relevant LVIS vocabulary, for anyone tempted to re-derive an eval:
syringe **1** image, tobacco_pipe 1, matchbox 1, cigarette_case 2, medicine 4, ashtray 6,
cigarette 13, hookah/pipe_bowl/cigar_box **0**. Open Images 600 has exactly one relevant
class — `Syringe`, 10 test images. That is the whole labelled universe.)*

## 3. The finding that shaped the design: two negative banks, not one

The intuitive move — "subtract the things people confuse with drugs" — **destroys the
detector**, and it is what the placeholder track spec was doing. Measured on the same
slice: AP **0.58 with a background bank → 0.04** when the pharmacy/clinical-syringe bank is
max-subtracted. The reason is that the confusables split into two kinds:

* **BACKGROUND** — *visually distinct* from drugs, but confusable to a text-image model:
  snow (scored as cocaine — a real COCO false positive), plumbing (scored as a bong — a
  real COCO false positive), houseplants and ferns (scored as cannabis), sugar/flour,
  incense, campfires. A true drug photo does not match these, so subtracting them costs no
  recall and centres the score. **These are subtracted.**
* **POLICY NEIGHBOURS** — *visually identical*, separated only by intent: a clinical
  syringe, a pharmacy shelf, a prescription bottle. Subtracting them cancels the positive
  it was meant to refine. **These are never subtracted** — they are scored only to annotate
  a flag (`nearest_benign: "a medical syringe on a sterile tray"`), which is what lets a
  human clear a false alarm in one glance.

`tests/test_drugs.py::test_policy_neighbours_are_never_subtracted` locks this in, and
`moderation.json` carries both lists separately so the daemon cannot re-merge them by
accident. (Prompt bug found the same way: `"a water pipe used to smoke drugs"` fires on
literal bathroom plumbing — replaced by `"a glass bong with a bowl and a stem"`.)

## 3a. REFIT v2 — four measured defects from integration, all fixed (2026-07-22)

b-daemon folded the v1 spec into the live reader and b-app ran it on a real 7,790-image
index. Four defects came back; all four are now fixed and pinned by tests. Refit corpus:
**15,010 deduped real-photo negatives** (COCO val2017 + Unsplash-demo + Unsplash-b) — ~2×
b-app's pool, deliberately including the corpus that saturated v1.

| # | Defect (measured) | Root cause | Fix | Verify |
|---|---|---|---|---|
| 4 | **218 violations all at p=0.99**, first 21 all benign (fire hydrant, teddy bear, halved oranges) | logistic slope A≈105 fit on 17 positives → razor-thin p=0.02→0.99 band over margin [0.025,0.106]; any heavier tail saturates | **evidence cap** `P_MAX=(n+1)/(n+2)=0.944` (p≥0.95 unreachable by construction) + **ridge-regularized** gentler slope, fit on the full real pool | p-histogram now SPREADS: **0 negatives at p≥0.9**, 1 in [0.7,0.9]; violation rate **1.06%** ≈ the fit's own prediction |
| 3 | **raspberry/bramble leaf on black → p=0.92** | TWO causes: (a) that image was **mislabelled as a cannabis positive in my ground truth** — a contaminated positive taught "any serrated leaf = cannabis"; (b) no compound-leaf negatives | full-res re-audit of all 18 positives (1 mislabel found + removed → 17); serrated/compound-leaf + benign-object negatives added | leaf → **none, p=0.0017**; true cannabis bud unchanged (zero recall cost) |
| 1 | **tau_review (0.0316) > tau (0.0191)** → review tier unreachable | review bar set by an independent FP budget, not as a band | review is now a **band BELOW violation**; asserted in `policy()` and a test | tau_review **0.0083 < tau 0.0100** |
| 2 | a vape → violation (policy says review) | `p≥tau` alone can't separate "smoking a joint" from "smoking a cigarette" | tier arbitration (§3b) + vape in the acceptance suite | vape → **review** (acceptance PASS) |

**Also found and fixed — a dedupe/labelling bug the bigger corpus exposed:** the Unsplash
corpora *contain* the drug-probe photos (same photo ids), so a labelled cannabis image was
being counted as a NEGATIVE by whichever corpus indexed it first (it was atop the "false
positives" — the very symptom). The eval now maps drug-probe labels first, so a positive
keeps its label wherever it appears.

**Post-refit operating point** (shipped τ=0.0100, full 15k pool): drug recall **0.88 as
violation / 0.94 surfaced** (violation+review), **1.06% of negatives** flagged violation +
0.69% review, **0 tobacco images wrongly called a violation**. AP fell 0.73→0.47 vs the v1
number — because v1's 0.73 was measured on 5k COCO only; on 15k diverse real photos with the
mislabel removed, 0.47 is the honest number. The six-image acceptance suite (our two: vape,
leaf) passes. Full histogram + numbers: `research/eval-drugs.json`.

## 3b. Tier arbitration — the vaping case, resolved (2026-07-22, after b-daemon's report)

b-daemon's v0 scaffold flagged **a woman vaping in a car at p=0.964** as a drugs violation.
My v1 reproduced the class of failure at a lower score: `p >= tau` alone made the vape
exhale a **violation**, because "a person smoking a marijuana joint" and "a person smoking a
cigarette" are the same picture to a whole-image model.

Fix (measured, not guessed): a **violation** must additionally be explained *better by the
drug bank than by the tobacco bank*, by `TIER_MARGIN = 0.01`. What loses the arbitration is
**demoted to review, never dropped** — ADR-14 says a human decides, and a test asserts
nothing that passes τ can leave the queue.

| Slice | before arbitration | after |
|---|---|---|
| 18 hand-verified drug images | 18 violation | **15 violation + 3 review** (the 3 are people smoking) |
| 128 tobacco/pill keyword photos | 22 violation | **10 violation + 16 review** |
| 5,000 COCO val2017 | 79 violation (1.54%) | **46 violation (0.92%) + 65 review (1.3%)** |
| the vaping image | violation | **review** ✅ |

Cost: 3 of 18 drug images move from violation to review. Nothing is missed — both tiers are
surfaced, counted and searchable — and the price is paid on exactly the images where the
distinction is genuinely unavailable in pixels.

### The boundary is CONFIG, not code
`moderation.json → categories.drugs.tobacco_tier ∈ review | violation | none` (plus `tau`,
`tau_review`). A changed user ruling is a **one-word edit — no retrain, no re-embedding,
no redeploy of prompts** — because both banks are always scored and only the tier label
changes. An invalid value falls back to the ADR-14 default rather than taking moderation
offline. Tests: `test_tobacco_tier_is_config_driven`, `test_config_tier_none_never_emits_review`.

## 4. What the false positives actually are (all hand-inspected)

At the shipped threshold, 1.54% of ordinary photos flag. The population is not random:

1. **Toiletry/vial kits** — an open suitcase of small bottles is the single highest-scoring
   negative in COCO val2017 (p 0.975). Genuinely looks like a kit.
2. **Flat-lays of many small objects** — "drug paraphernalia laid out on a table" fires on
   any tidy arrangement of small items.
3. **Hand-to-mouth poses** — brushing teeth, holding a Wii remote → "a person smoking a
   joint". The pose is the signal, and the object is too small to correct it.
4. **White powder textures** — snow, ski slopes, sugar (mitigated by the background bank,
   not eliminated).
5. **Green foliage** — mitigated by the background bank; still the residual on plant photos.

Every family in that list is represented in `BACKGROUND`, and a test asserts it stays that
way, so a future prompt edit cannot silently drop a mitigation.

## 5. Specialised models — triage, and why none is adopted

* **No credible open drug/paraphernalia classifier exists** at a license and quality bar
  this project can accept. What is on GitHub/HF is small, single-class, license-silent
  models trained on scraped search results with no published eval — exactly the
  "low-quality GitHub classifier" the brief warned about. Adopting one would trade a
  measured 0.726 AP for an unmeasured number plus a license risk plus a second model on an
  8GB box.
* **Open-vocabulary detectors** (YOLO-World/YOLOE/OWLv2) would fix the small-object gap and
  are already dead-ended in ORACLE §3: GPL/AGPL and/or latency.
* **NSFW/`nudenet`-style stacks** do not cover drugs at all.
* **RAM++ / tagger oracles** — ORACLE §3 dead end (3.01GB).
* Therefore v1 ships the zero-shot ensemble, which is free (one matmul on embeddings the
  index already computed), license-clean, and *measured*. If the user later provides even
  200 labelled drug images from their own sites, a logistic head on those embeddings is a
  ~30-minute upgrade with a real fitted τ — that is the highest-value next step by a wide
  margin, and it is the only route to `enforcement_ready: true`.

## 6. Integration

* **Index-time (ADR-14 seam, live):** `drugs.load_drugs_head(profile)` returns a head whose
  `.score(embeddings, images, ids)` yields `{category:"drugs", p, tier, why, group}` per
  image, and `.probs(embeddings) -> (p, tier)` as ARRAYS — the shape b-daemon's
  `Searcher.track_scores()` wants, so plugging it in is one call, not a port. `imgtag.moderation.load_heads` picks it up automatically — verified end-to-end:
  `imgtag index … --moderation` printed *"Found 0 images with nudity (10 for review), 0
  images with weapons, 24 images with drugs"*.
* **Cost:** one text-tower batch **once per (model, prompt-set)**, cached to
  `~/.imgtag/models/<model_sha>/drugs-<spec_sha>.npz` (~200KB, atomic write). Warm load
  measured **1ms** and loads **no text tower** — ADR-5's resident-set law honoured. Per
  batch: two matmuls of [n,512]×[512,29] and [512,30]. No new dependency (ADR-7 clean).
* **Search-time:** `moderation.json` → `categories.drugs` carries `violation`, `review`,
  `negatives`, `policy_neighbours`, `templates`, `scorer: "margin"`, `platt`, `tau`,
  `tau_review`, `spec_sha` and the `fit` provenance. A test fails if the file and the module
  drift apart.
* **⚠️ For b-daemon (`search.py::_track_vectors`)** — two changes are needed for the daemon's
  numbers to match this lane's:
  1. it computes `mean(prompts) − 0.5·mean(negatives)`; the measured-better feature is
     `max(prompts) − max(negatives)` (AP 0.31 → 0.57 on the identical slice);
  2. it must **not** subtract `policy_neighbours` (§3), and must read the per-track `platt`
     + `tau` rather than the shared z-score logistic — ADR-3 §2 forbids mixing a raw z-score
     with a fitted probability;
  3. tiering must go through the arbitration in §3b, or the vaping image is a violation
     again. Simplest correct integration: call `load_drugs_head(...).probs(snap.emb)` and
     use the `tier` array it returns.
  Not edited here (file ownership, F2). Escalated to the conductor with this report.
* **Tier semantics (ADR-14):** `violation` = illegal drugs/paraphernalia; `review` =
  tobacco/vape; anything else `none`. Tobacco is never counted as a violation.

## 7. Policy questions for the user (rulings needed; two already answered)

1. ~~Is tobacco a drug?~~ **RULED 12:50Z: review tier.** Implemented. *(But see §2: the
   review tier's measured recall is 0.17 — the ruling is honoured in code and largely
   unmet in practice. If tobacco enforcement matters, it needs its own instrument.)*
2. ~~Vape vs cannabis vape~~ — both land at review; visually inseparable. Consequence of #1.
3. **Alcohol** is not a drug track at all today (beer/wine/spirits never flag). Should it be
   its own track? *Recommendation: yes, separately — do not fold it into drugs.*
4. **Legal cannabis** (dispensary shelf, medical marijuana, CBD product photos) — pixels
   cannot separate legal from illegal. Default: **flags**. Accept, or exempt?
5. **Medical syringes / vaccination / insulin / IV drips** — default **not** flagged. On a
   health-topic site you may want the opposite.
6. **Prescription medicine** (pill bottles, blister packs) — default not flagged. "Pills
   spilled on a table" is the grey zone and will sometimes flag.
7. **Drug-awareness / harm-reduction / news / anti-drug campaign** imagery is pixel-identical
   to what it depicts. It flags. Always needs a human.
8. **Historical / artistic** (poppy fields, opium-den paintings, hemp rope, hemp-seed food).
9. **Kitchen powders and culinary mushrooms** — mitigated, not solved.
10. **What does "flagged" DO?** This threshold is calibrated for a **review queue**
    (recall-first). Auto-hide needs a second, much higher threshold and a stated cost for a
    wrong hide — say the word and it will be fitted and reported separately.

The list is also in code (`drugs.AMBIGUITIES`) so it cannot rot away from the detector.

## 8. Honest limits (read before quoting any number above)

* 18 violation positives and 36 review positives. Every metric is a **wide-CI estimate on
  one corpus**, not a benchmark. A single label flip moves recall by ~5 points.
* Two positives were surfaced by the model itself before being labelled → recall is
  mildly optimistic (§2).
* Negatives are an **upper bound** on the FP rate (federated labels).
* All numbers are `pecore-s16-384-fp32` on a **PROXY dev box** (M3 Max, contended,
  loadavg high). Quality metrics are contention-immune; **no throughput number is claimed
  here** (ORACLE 11:35Z rule). Cross-model comparison (SigLIP2 anchor) is **not run** —
  open item for b-bench, which owns the model axis.
* `enforcement_ready` stays **false** for drugs until a τ is fitted on labelled data
  honouring the ADR-14 boundaries. This track is a review queue, and says so in its payload.

## 9. Reproduce

```bash
uv run imgtag index data/coco/val2017 cocoval2017 --wait     # 5000 imgs, one-time
uv run python scripts/eval_drugs.py                          # → research/eval-drugs.json
uv run pytest tests/test_drugs.py -q                         # 10 tests, no model needed
uv run imgtag index data/drug-probe/strong drugprobe --wait --moderation   # end-to-end
```
`data/drug-probe/` (328 Unsplash images, 33MB) and `data/oi-drugs/` (10 OI images) are
fetched by the commands recorded in §2 and are **git-ignored** — no dataset bytes in the
repo, no Unsplash redistribution.
