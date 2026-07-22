# research/track-safety.md — the SAFETY track (people lying down + danger escalation)

> OWNER: track-safety (predecessor: track-safety; current: track-safety2).
> User law, VISION-ADDENDA 13:20Z (verbatim): *"make another track to identify people
> lying down (even if part of their body is obstructed) and even higher flagging if
> either detecting injury, things broken, distruction distress high stress or anything
> dangorous"*.
> Binds: ORACLE ADR-14 (`alert` = highest tier, sorts above violation; enforcement_ready
> false until τ fitted on labeled GT) · ADR-15 / TRACKS.md T1–T4 (dense f32 sidecar, raw
> scores stored, tiers derived at read; embedding-space instrument; agents verify, never
> operate) · §7 escalation contract (never guess a measurable number; stop-and-say-so).

This document is the track's report. It records **what was built, what is measured, what
is NOT, and the exact reason the versioned spec is being WITHHELD** from
`data/moderation.json` until it beats a known failure set. A weak `alert` — the app's most
prominent chip — is worse than no `alert`.

---

## 1. What this track is (and is not)

This is a **welfare monitor**, not a rule-enforcer. The other three moderation tracks
answer *"did someone break the site rules?"*; this one answers *"does someone need help?"*.
That flips the cost asymmetry: a missed nudity flag embarrasses a site; a missed
person-down could be a person on a warehouse floor at 3am. So the base detector is
**recall-first**. The `alert` tier is the exception — see §6: an `alert` that pages a human
must be **precision-first**, because a false alert on the highest chip erodes trust in the
whole tier.

Two INDEPENDENT signals, never blended into one number (the user asked for an *escalation*,
not an average):

```
p_lying  = sigmoid(A_l · (max_cos(LYING) − max_cos(BACKGROUND)) + B_l)   → the review tier
p_danger = sigmoid(A_d · (max_cos(DANGER) − max_cos(DANGER_BG)) + B_d)   → the escalation

tier = alert   if p_lying ≥ τ_review  AND  p_danger ≥ τ_danger      (person-down IN danger)
       review  if p_lying ≥ τ_review                                (person-down, benign)
       none     otherwise
```

A person-down IS the flag; danger only decides how loudly to shout. `danger_alone_tier`
(default `none`) is the un-ruled policy question (AMBIGUITIES #2): a burning building with
nobody visible produces no safety flag until the user rules — the user's words gate
escalation on *people lying down*.

---

## 2. Instrument evaluation — the person-down detector (brief item 1)

**Decision: the runtime instrument is embedding-space margins. Keypoint/pose geometry is
kept OFFLINE ONLY, as a ground-truth builder and occlusion stratifier — never in the hot
path.** This was evaluated, not assumed:

- The predecessor's `scripts/eval_safety.py` contains a full keypoint-pose classifier
  (`_pose()`: torso vector within 30° of horizontal AND leg vector within 45° of
  horizontal, sub-8px torso rejected as annotation noise). It is used to **build labels
  and stratify occlusion**, and it is deliberately NOT the runtime detector.
- Why keypoints cannot ship in the runtime: (a) TRACKS.md T2 — a per-image pose model is a
  second forward pass; the deploy target is a shared **8GB CPU Linux box** whose co-tenants
  are sacred (ADR-10), and the dedicated-model FLOPs budget (B25) is already fully spent by
  nudity's Marqo head. (b) T4 — the runtime must be 100% programmatic and identical with
  zero agents; a pose model adds a dependency and cost with no measured benefit here.
- Why keypoints are also **worse** here, measured: geometry and human-caption consensus
  agree on only 15 images while geometry alone claims 30 (a ~50% disagreement rate on COCO
  val2017); eyeballing the disagreements showed the *geometry* wrong (crouched jockeys /
  surfers / skateboarders labeled "lying" by torso angle). Human consensus is the primary
  label; geometry earns its keep ONLY as the occlusion axis.
- The headline result that vindicates the embedding choice: **occlusion does not make the
  hidden body the worst case.** Recall on the heavily-occluded stratum (positives where
  COCO's own annotator could not place a torso — covered by a blanket, cropped to a head)
  is **0.786** at the shipped threshold — below the fully-visible stratum's **0.923** but
  ABOVE the geometry-ambiguous stratum's **0.643**; at a looser 5%-FP point it reaches
  **0.929**, matching fully-visible. Whole-image embeddings do not need visible joints —
  which is *exactly* the "even if part of their body is obstructed" clause the user named,
  and it is the property a keypoint pose model structurally CANNOT have (no keypoints → no
  pose → no label → recall ~0 on this stratum by construction).

**Conclusion:** no new runtime instrument is needed or wanted. The predecessor made the
correct call. This track's improvement budget goes to the DANGER/ALERT signal and to
fitting honest thresholds, not to a pose model.

---

## 3. Measured on COCO val2017 (predecessor, committed in `safety.py` FIT)

Model: `pecore-s16-384-fp32`. Ground truth: two independent human-annotated sources —
`person_keypoints_val2017` (geometry) + `captions_val2017` (≥2 of 5 captions = consensus).
41 human-consensus lying positives vs 4058 doubly-verified negatives (chance AP 0.010).

| signal | metric | value | note |
|---|---|---|---|
| lying | AP | **0.45–0.53** | 0.453 w/o animal negs, 0.534 with; NOT fold-stable (n=41) — report the range, never 0.53 alone |
| lying | recall @1% FP | 0.659 | |
| lying | recall @2% FP | **0.780** | the SHIPPED review operating point (τ_review=0.0695) |
| lying | recall @5% / @10% FP | 0.829 / 0.878 | 5% buys only +0.05 recall for 2.5× the queue |
| occlusion | recall @ shipped τ, visible / ambiguous / hidden | 0.923 / 0.643 / **0.786** | hidden ≥ ambiguous (not the worst case); hidden 0.929 @ 5%-FP; a pose model = ~0 on hidden by construction |
| danger | AP | **0.144** | WEAK, SAID SO — only 19 danger images in COCO val2017; directional |
| danger | recall @2/5/10% FP | 0.316 / 0.421 / 0.526 | shipped τ_danger=0.0076 is the 10%-FP point |
| **alert** | precision, recall | **UNMEASURABLE** | COCO val2017 has exactly **1** person-down-in-danger image (000000354307, injured man) of 5000 — the scorer DOES tier it `alert`; 1 TP caught = sanity check, not a metric. The other 40 lying positives are benign |

Platt fits (written by the eval, never by hand): lying `(99.3941, −5.2050)`, danger
`(37.8043, −5.3679)`.

**The gap that defines this track's work:** the `alert` tier — the whole point of the
13:20Z directive — is essentially unmeasurable on COCO (n=1). That is why the TP probe
(§5) exists, and why the spec is withheld (§6).

> Correction (consolidation, this session): the predecessor FIT said "0" person-down-in-
> danger images; a re-check found exactly **1** (000000354307), and the shipped scorer
> tiers it `alert` with `danger_why: "an injured person bleeding"`, `nearest_benign: "a
> patient resting in a hospital bed"`. `safety.py` FIT + docstring corrected to match. One
> caught TP is a sanity check, not a precision estimate — the withhold (§6) stands.

---

## 4. Honest gaps (do not paper over — §7f)

1. **Alert precision is unknown**, not estimated. Shipped recall-first with
   `enforcement_ready=false`.
2. **Distress is not implemented.** "high stress" is largely invisible to a whole-image
   embedding — a distressed face is a handful of pixels. There is no distress signal; the
   danger score must not be read as one. No fetch keyword targets it (§5).
3. **No labeled safety corpus exists on the deploy machine.** The single largest available
   improvement is 200–500 hand-labeled real person-down images from the deployment site.
4. **41 positives is a small set, one corpus, negatives chosen after inspecting this
   corpus's FPs.** Every COCO number is directional until a labeled corpus exists.

---

## 5. The TRUE-POSITIVE probe — `safetyprobe` dataset (brief item 3, VISION-ADDENDA 13:58Z)

The user's 13:58Z law (given for weapons/nudity, applied here): *bring real true positives
so monitoring is truly tested; current findings should score at LOWER confidence than the
TPs, and the ratio between them sets the auto-flag threshold per track.*

**EVAL DATA LAW (hard):** policy-safe TPs only — people lying down, falls, accidents
**WITHOUT gore**. Source is **Unsplash stock** (curated, no gore corpora fetched). Graphic
violence / gore corpora are never fetched. `blood`/`injury` search terms on Unsplash return
mild staged stock, not gore.

**Corpus on disk:** `data/safety-probe/` — `lie/` (183 imgs) + `dan/` (614 imgs) = 797,
fetched by keyword. `keywords.json` maps each image → its fetch query terms. **The fetch
keyword IS the weak label** — labeling is PROGRAMMATIC (TRACKS.md T4: agents never
hand-categorize; ≤20 diagnostic views/round only, reserved for spot-checking the top alert
candidates after scoring). This is the drugprobe/weaponprobe pattern.

**`labels.json` (built this session, disjoint subcategories):**

| subcategory | n | fetch keywords | role |
|---|---|---|---|
| `person_down` | 118 | asleep, sleeping, nap, lying, lying down | benign person-down → p_lying recall + must NOT alert |
| `sunbathing` | 8 | sunbathing | benign outdoor lying (the boots-on-ice-adjacent benign class) |
| **`alert_tp`** | **57** | any lying kw ∩ any danger kw | **person-down IN danger — the alert-tier TP COCO lacked** (43 strong: lying + stretcher/first-aid/injury/rescue/ambulance) |
| `injury_context` | 72 | injury, injured, blood, first aid | injury danger, not lying → p_danger; per policy should NOT alert (no person-down) |
| `danger_context` | 25 | ambulance, stretcher, rescue | emergency-response danger, not lying |
| `destruction` | 517 | rubble, earthquake, disaster, destruction, broken glass, crash, wreck | structural/vehicle danger, not lying |
| `distress` | 0 | — | NO keyword; unmeasured gap (§4.2) |

Derivation rule is recorded verbatim in `labels.json` `_subcategory_keywords` +
`_priority` (alert_tp > person_down/sunbathing > injury_context > danger_context >
destruction), so the labels rebuild deterministically from `keywords.json` with no new
committed script.

**The 57 `alert_tp` images are the headline:** they are person-down-in-danger, the exact
class COCO val2017 has zero of. They make alert-tier separation measurable for the first
time. Labels are WEAK (a "first aid" tag can mean a kit, not a victim) — treated as
directional, spot-checked ≤20/round, never as a precision ground truth.

**Indexing as the `safetyprobe` dataset is HELD** until the conductor's ALL-CLEAR (quiet
window: no `imgtag index` / bulk embedding while the main index builds — §8 loadavg gate).

---

## 6. Separation + threshold fit — the plan, and why the spec is WITHHELD (brief items 2, 4)

**Standing law (from b-daemon, adopted):** the versioned spec ships to
`data/moderation.json` `categories.safety` ONLY when the alert-tier TP confidence
**dominates the benign false-positive band** per subcategory. b-daemon's draft prompts
alert-flagged **12 benign images** (boots by ice, a night puddle) — a weak alert is worse
than none, so b-daemon rightly withheld the spec. This track inherits that gate.

**Why boots-on-ice can currently reach `alert`:** to alert, an image needs *both*
p_lying ≥ τ_review AND p_danger ≥ τ_danger. The shipped τ_danger is **0.0076** (the 10%-FP
point — deliberately loose because on COCO it only ever re-tiered images already flagged
person-down). On benign scenes with a low object on the ground (boots, a puddle at night),
p_lying can misfire ("the legs of a person lying") while the loose danger gate passes on
icy/wet/dark texture. Two weak signals ANDed at loose thresholds is not enough for the
highest chip.

**The fit (executes on ALL-CLEAR):**
1. Score the whole probe (797) + b-daemon's 12-image benign-FP set with the current
   scorer → per-subcategory distributions of `p_lying` and `p_danger`.
2. **Separation test per subcategory** (person-down / injury-context / destruction /
   distress / danger-context): does the `alert_tp` (57) `p_danger`·`p_lying` band sit
   ABOVE the benign-FP band (boots-on-ice class + benign `person_down`/`sunbathing`)? If
   the bands overlap, the signal cannot carry an `alert` tier and the tier stays withheld —
   that is a valid, honest result (§7f), not a failure to hide.
3. **Fit τ precision-first for alert, recall-first for review:**
   - `τ_review` — keep recall-first (the 2%-FP person-down point), re-fit on the probe's
     real person-down images (183) as a cross-corpus check on the COCO fit.
   - `τ_alert` (= τ_lying ∧ τ_danger operating point) — **precision-first**: set the danger
     gate at the operating point where benign-FP precision is high (target ≥0.9 on the
     boots-on-ice band), even at the cost of alert recall. A false alert on the top chip
     costs trust; a missed alert degrades to `review` (still flagged), not to `none`.
   - **CIs by bootstrap** over the probe (1000 resamples) on AP and on recall/precision at
     the chosen τ — reported with n and the "weak-label, one-corpus" caveat, never a bare
     point estimate.
4. Only if separation holds: write `categories.safety` to `data/moderation.json`.
   `track_spec()` in `safety.py` already emits the reader-shaped entry; the withheld piece
   is the *fitted, validated* τ_alert. **Reader-compat reshape (this session, per b-daemon):**
   the daemon's reader assigns each row to the single tier it EXCEEDS by the most (argmax
   over tier prompt-sets) — it cannot express the code head's two-margin AND. So the spec's
   `alert` bank is written as COMBINED person-down+danger phrases (`ALERT_PHRASES`: "an
   injured person lying on the ground", "a person collapsed next to a wrecked car") so
   exceedance-over-alert ≈ the AND; `review` = benign person-down; the sets contrast, not
   nest. `negatives` gain `GROUND_LEVEL_FP` (boots-from-above / puddle-on-pavement / empty
   floor) to kill b-daemon's measured boots-on-ice alert FP class. `calibration:
   "proxy-fitted"` → the reader corpus-relative-thresholds it, never trusts τ.
   **STAGED, NOT COMMITTED:** the entry is generated by `track_spec()` and validated by
   `test_safety.py`, but is NOT written into `data/moderation.json` — the withhold stands
   until the ALL-CLEAR separation run proves alert TP dominates the benign-FP band.

`enforcement_ready` stays **false** regardless (ADR-14): a proxy-fitted, weak-label τ on a
Unsplash probe is a review-queue threshold, not a page-a-human threshold.

---

## 7. Status

| deliverable | state |
|---|---|
| person-down instrument evaluated (item 1) | ✅ done — embedding-space runtime confirmed correct; keypoints offline-only |
| `safetyprobe` corpus + `labels.json` (item 3) | ✅ built (797 imgs, 57 alert-TPs, 6 subcategories); indexing HELD for ALL-CLEAR |
| separation per subcategory (item 4) | ⏳ blocked on (a) ALL-CLEAR embedding, (b) b-daemon's 12-image benign-FP band |
| fitted τ_alert + τ_review + CI (item 4) | ⏳ same block |
| spec → `data/moderation.json` (item 2) | ⛔ WITHHELD by design until separation beats the benign-FP band |
| ledger entry (item 5) | ⏳ lands with the measured round (before/after numbers — never fabricated, improve-track law) |

**Next actions (in order):** (1) receive b-daemon failure analysis → define benign-FP
band; (2) on ALL-CLEAR → `imgtag index` the probe as `safetyprobe` + run `eval_safety.py`;
(3) fit + separation test; (4) if it passes, ship spec + ledger entry; if not, report the
overlap honestly and keep the tier withheld.

> 2026-07-22 · track-safety2 · created (predecessor died before reporting). COCO numbers
> cited from the committed `safety.py` FIT (predecessor's measured run). Probe counts +
> taxonomy measured this session (pure JSON, no embedding). All τ/separation/CI numbers
> marked ⏳ are UNMEASURED and will not be quoted until the ALL-CLEAR run produces them.
>
> 2026-07-22 · track-safety (consolidated, sole owner) · predecessor turned out alive and
> producing; successor stopped; lanes merged into this one. Reconciled: (a) corrected the
> alert case from "0" to the real n=1 (000000354307), caught by the scorer — `safety.py`
> FIT + docstring + this report fixed; (b) corrected the occlusion numbers to the shipped
> module + threshold (0.923/0.643/0.786 @ shipped τ, hidden 0.929 @ 5%-FP); (c) added the
> reader-compat reshape — `ALERT_PHRASES` (combined person-down+danger) + `GROUND_LEVEL_FP`
> negatives — and `test_safety.py` (12 unit/contract/acceptance tests, incl. the six-scene
> sketch). Preserved the successor's WITHHOLD decision and `labels.json` taxonomy verbatim.
> Coordinated the alert boundary with track-violence (safety.alert = person-down∧danger;
> violence.alert = gore) and handed b-daemon the final DANGER_WORD regex. Everything ⏳
> remains blocked on the quiet-window ALL-CLEAR.
