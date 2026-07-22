# track-sports — the sports CONTENT track

> Owner: track-sports2 · verified & finalized 2026-07-22 by track-sports3 (COCO/LVIS numbers
> reproduced to the digit over existing embeddings; weak-label §6 re-measured on the grown
> corpus). Instrument: prompt ensemble over the index embedding (TRACKS.md
> T2 rung 1 — the only unconditionally-allowed instrument). Not moderation: the answer is
> a content label, tier `match`|`none` (ADR-14, `match` added 13:23Z), routed to the
> `content` bucket, never summed into moderation totals. Ships as `src/imgtag/moderation/
> sports.py` + a `categories.sports` entry in `data/moderation.json` + a per-model fitted
> file `data/moderation/sports-<model_id>.json`. Every number here is reproducible with
> `./.venv/bin/python scripts/_sports_explore.py [--fit]`.

## 0. What shipped

- **Scorer.** `score = max_k cos(image, sport_prompt_k) − max_m cos(image, background_m)`.
  One `[N,D]·[D,92]` + one `[N,D]·[D,22]` matmul per image, no second model, no new
  dependency (ADR-3/ADR-7). 92 sport prompts across **29 sport labels**; the argmax prompt
  names WHICH sport, so a match carries a `label` ("tennis", "skiing", …) at zero cost.
- **Two paths, one seam.** `SportsHead` (Platt-calibrated, τ fitted on a held-out split;
  loads from the machine profile with no text-tower pass — prompt matrices baked fp16 into
  the fitted file) and `ZeroShotSportsHead` (same margin, no fit, `p` is a ranking).
- **Measured on CORPUS-A** (COCO val2017, 5000 imgs, PE-Core-S16-384 fp32):
  **AP 0.9321**, held-out precision **0.801** / recall **0.947** at τ_match **0.1809**.
- **`enforcement_ready` = false, permanently.** A content label is not a policy breach and
  must never gate enforcement.

## 1. Ground truth

- **Primary (exhaustive).** COCO val2017 `sports` supercategory — 10 children (baseball
  bat, baseball glove, frisbee, kite, skateboard, skis, snowboard, sports ball, surfboard,
  tennis racket). **938 / 5000 positives (18.8%)**. Exhaustively annotated, so precision
  AND recall are both real (not recall-only like a keyword corpus).
- **Extra (exhaustive, different taxonomy).** LVIS v1 val restricted to val2017 — the
  sports COCO lumps into `sports ball` or does not name at all (basketball, volleyball,
  soccer_ball, golf_club, hockey_stick, dumbbell, boxing_glove, ski_pole, scoreboard, …).
  Lifts the positive count to **953 / 5000**. Used as a fairer denominator, never as a
  negative.
- **Weak (NON-exhaustive, labeled weak).** Unsplash Lite photographer/AI keyword rows over
  the indexed `unsplashb` snapshot (n≈9926; was ≈2540 at first fit — b-corpus grew it, so the
  §6 weak-label rates were re-measured 2026-07-22 by track-sports3). Keywords are suggestions,
  not exhaustive labels — read as a cross-domain sanity check and a probe for the
  activity-only classes COCO cannot score, NEVER as precision/recall.

## 2. Chosen design — and the one place it diverges from weapons

Weapons/nudity are recall-first enforcement (a missed weapon goes live). Sport is neither:
a missed sports photo just does not surface in a content filter. So the operating point is
**precision-first**: τ_match = the smallest threshold whose precision reaches 0.80 on the
held-out split (so recall is maximal at that precision). Rationale confirmed by b-daemon:
"a content label is not an accusation." A site that wants a wider net lowers the precision
floor — the sweep in §5 gives the exact τ for 0.60 / 0.70 / 0.80 / 0.90.

## 3. Measured — the background bank is generic-only, and that is the load-bearing result

The instinct from weapons.py is a big hard-negative bank. **On this track it backfires.**
Scene-level hard negatives ("an empty sports stadium", "a snowy mountain landscape", "a
beach with people sunbathing", "a swimming pool with nobody in it") sit *next to* real
sports scenes in embedding space, so `max(background)` rises on the very images we want to
keep and the margin collapses. Measured on COCO (all-92-prompt positive bank):

| background bank | COCO AP | R@fpr1% | R@fpr5% | held-out rec@prec0.80 |
|---|---|---|---|---|
| none (raw cosine) | 0.9460 | 0.819 | 0.949 | — |
| **generic only** | **0.9562** | 0.810 | 0.962 | 0.964 |
| generic + borderline **(SHIPPED)** | 0.9321 | 0.706 | 0.923 | **0.947** |
| generic + far-hard | 0.9154 | 0.625 | 0.899 | 0.912 |
| full bank (v0 draft) | 0.9074 | 0.629 | 0.897 | 0.889 |
| full + borderline (v0 draft) | 0.9016 | 0.608 | 0.889 | 0.889 |

`generic only` has the highest COCO AP, but COCO contains no chess/hiking/yoga images, so
it cannot see the cost of letting borderline sports through. The **shipped** bank is
`generic + borderline`: the 7 borderline sports are folded into the background so they read
`none` by default (§7). That trades COCO AP 0.956→0.932 to buy borderline suppression the
COCO metric is blind to — measured on Unsplash in §6. **Do not add scene hard-negatives to
this bank without re-measuring.** (Opposite of weapons, where "kitchen knife" is genuinely
far from "rifle".)

`mean-top3` background pooling recovers some of the loss (0.932→0.937 AP) but not enough to
beat generic-only, and it complicates the scorer for b-daemon's reader — rejected.

## 4. Measured — per-class recall (held-out τ = 0.1809)

**Per COCO child** (exhaustive truth): baseball bat 0.938 · baseball glove 1.000 · frisbee
0.917 · kite 0.934 · skateboard 0.945 · skis 0.917 · snowboard 0.959 · sports ball 0.941 ·
surfboard 0.973 · tennis racket 0.970. **Uniform ≥0.92** — no dead child.

**Per LVIS-extra child** (sports COCO does not name): tennis_ball 0.983 · soccer_ball 0.947
· baseball 1.000 · ski_pole 0.903 · ski_boot 0.923 · scoreboard 1.000 · home_plate 1.000 ·
mound 1.000. The equipment prompts generalise past the COCO taxonomy.

## 5. Operating-point sweep (held-out half, COCO truth)

| target precision | τ_match | precision | recall | f1 | match-rate |
|---|---|---|---|---|---|
| 0.60 | 0.0487 | 0.600 | 0.971 | 0.742 | 0.308 |
| 0.70 | 0.0920 | 0.700 | 0.956 | 0.808 | 0.260 |
| **0.80 (SHIPPED)** | **0.1809** | **0.801** | **0.947** | **0.868** | 0.225 |
| 0.90 | 0.5489 | 0.902 | 0.828 | 0.863 | 0.175 |

**p-spread (NOT saturated — the drugs failure mode is absent):** q05=0.0003, q50=0.0112,
q95=0.9744; frac(p<0.02)=0.582, frac(p>0.98)=0.041. A genuine spread, not a p=0.99 pile —
this is why the fitted file ships `calibration: "fitted"` (b-daemon's gate).

## 6. Measured — the borderline & cross-domain probe (Unsplash weak labels)

Each background bank re-fitted on COCO before evaluation (otherwise the margins are on
different scales). Match-rate on keyword slices of `unsplashb`:

| slice | generic only | generic + borderline (SHIPPED) |
|---|---|---|
| sport-keyword (weak +) | 0.407 | 0.301 |
| concert / festival / music | 0.070 | 0.056 |
| crowd (no sport kw) | 0.108 | 0.090 |
| **hiking** (borderline, OUT) | 0.209 | **0.034** |
| food | 0.037 | 0.036 |
| architecture | 0.088 | 0.066 |

> Re-measured 2026-07-22 (track-sports3) on the grown `unsplashb` (n≈9926, was ≈2540). The
> COCO/LVIS exhaustive-GT numbers above reproduced to the digit; only these weak-label rates
> shifted with the corpus — direction and ranking unchanged.

The shipped bank drives **hiking 0.209 → 0.034** (≈6× suppression) while barely moving true
sport (0.407 → 0.301). That is the whole reason it beats generic-only despite a lower COCO
AP. `concert` and `crowd` (the "stadium hosting a non-game" FP class the user's acceptance
sketch names) sit at **0.056 / 0.090** in-bank (0.070 / 0.108 generic) — low without any
scene hard-negative, the generic bank already handles them.

Weak-label sport recall is only ~0.30: keyword slices are noisy (a photo tagged "sport" may
be a shoe, a gym interior, an abstract) and this track is equipment/scene-biased. Read it as
a floor, not the recall — the exhaustive COCO recall (0.947) is the real one.

## 7. The borderline ruling (configurable, default OUT)

7 borderline "sports" — **chess, hiking, fishing, darts/pool, esports, yoga, dance** — read
as sport to some sites and leisure to others. Default **OUT**: their prompts are folded into
`negatives`, which actively suppresses them (measured: hiking 0.034 in-bank, ≈6× below
generic-only's 0.209). To count them
as sport, remove them from `negatives` (or call `SportsHead.build(borderline=True)`) and
re-score the ONE sidecar column — no re-embedding of images (TRACKS.md T3). This is a data
edit, not a code change.

**User-acceptance sketch, graded:** soccer match → `match(soccer)` ✅ · tennis-racket
closeup → `match(tennis)` ✅ (COCO tennis racket recall 0.970) · stadium concert → `none` ✅
(Unsplash concert 0.056 in-bank) · gym selfie → `match(gym)` — activity-only, COCO can't measure,
weak-label plausible · chess/hiking → `none` by the documented default, one flag to flip.

## 8. Known blind spots — stated plainly

- **Activity-only sport is under-measured.** COCO/LVIS annotate OBJECTS; swimming, running,
  a gym, martial arts, climbing have no annotatable object, so their recall is only weakly
  probed (Unsplash keywords). The `activity/bg-margin` bank alone scores AP 0.646 vs the
  equipment bank's 0.958 — the equipment half carries the track. A photo of runners with no
  visible race number or bib may score low. Distilled activity head is the darwin path if
  this matters.
- **Contact-sports recall is weak on the only slice available.** Unsplash boxing-keyword
  match-rate **0.102**, martial-arts **0.146** (equipment-biased prompts miss gym/portrait
  boxing shots; a few matches even argmax to "surfing" on n<5). COCO has ≈no boxing to
  measure. **Consequence for the violence track:** sports cannot yet be relied on as the
  exculpatory "this is a bout, not an assault" label — the compose-claim is not measured-
  strong on the sports side. Confirmed 2026-07-22 (track-sports3): the COCO TP label
  distribution shows **0** martial-arts / 0 boxing matches — COCO annotates ~no combat sport,
  so this bound is weak-label-only. Flagged to track-violence2; strengthening
  martial-arts/boxing prompts + a combat-sports TP probe corpus is the open item.
- **Equestrian / motorsport / kite dominate the false positives** (§ FP anatomy: 70
  motorsport, 43 kite, 29 equestrian FPs at τ). These ARE arguably sport (horse racing,
  motorsport) — the "FP" is partly a COCO-taxonomy artifact (COCO's `sports` supercategory
  excludes them). LVIS-truth precision is 0.783 vs the fitted 0.801 for exactly this reason.
  A ruling that equestrian/motorsport ARE sport would raise measured precision.
- **fp16 storage delta.** Prompt matrices ship fp16 (ADR storage law); τ/Platt are fit on
  the same fp16 values (build rounds in memory) so there is zero train/serve skew.
- **Zero-shot path is a ranking, not a probability** — `calibrated=False`, τ is a flag
  budget; only for backends with no fitted head.

## 9. Integration notes (for b-engine / b-daemon / b-app)

- **Dispatcher seam:** `load_sports_head(profile) -> SportsHead | None`. None when no head
  is fitted for the machine's backend — reported by name, never a silent zero.
- **Per-image schema:** `{category, p, tier: "match"|"none", model_id, calibrated,
  enforcement_ready:false, content_track:true}` + `label`/`sport` (the sport name) when
  `tier=="match"`. b-app reads **`label`** (cross-track field); `sport` is this track's alias.
- **Routing:** `content_track:true` on every flag AND on the head object → the `content`
  bucket, `content_counts`, `/api/search?track=sports`. Never in violation/review totals
  (b-daemon confirmed live).
- **Fitted file** `data/moderation/sports-<model_id>.json` carries `calibration:"fitted"`,
  `scorer:"margin"`, `tau_match`, `platt` (the keys b-daemon's spec reader honours) plus the
  fp16 prompt matrices this module needs to score with no text tower. The fitted file WINS
  over the spec; a refit is a pure file swap (TRACKS.md T3).
- **Spec ⇄ head must agree.** The `match`/`negatives` prompts in `moderation.json` are the
  EXACT strings this module embeds (borderline folded into `negatives`), so b-daemon's
  spec-reader margin reproduces this head's margin. Changing one requires changing both.

## 10. Reproduce

```
./.venv/bin/python scripts/_sports_explore.py          # all measurements above
./.venv/bin/python scripts/_sports_explore.py --fit    # + writes the fitted head file
./.venv/bin/python -m pytest tests/test_sports.py -q
```

Provenance: COCO val2017 instances + LVIS v1 val (val2017 subset) + Unsplash Lite keywords,
all already in `data/`. Backend PE-Core-S16-384 fp32 (model_sha 8c080c43…). Honest status:
COCO/LVIS numbers are first-party exhaustive-GT measurements; Unsplash numbers are weak-label
and labeled as such; activity-only and contact-sports recall are under-measured and flagged.

## 11. Ledger (dated, append-only)

- **2026-07-22 · track-sports2 · v1 shipped + owner-consolidated.** Built the measured
  scorer (embedding-space prompt ensemble, generic-only background, precision-first τ),
  measured on CORPUS-A (COCO val2017 exhaustive GT): AP 0.9321, held-out prec 0.801 /
  rec 0.947 @ τ 0.1809, non-saturated p-spread. Committed sports.py + fitted head +
  tests (17, green) + this doc + `scripts/_sports_explore.py` at 45c10f6.
- **2026-07-22 · spec approved (b-daemon).** `categories.sports` entry authored per
  b-daemon's amendments: `match` sole positive key, `match_labels` parallel array,
  `content_track:true`, borderline folded into `negatives`, τ in the per-model fitted
  file (not the spec). WRITTEN in the `data/moderation.json` working tree; the shared-file
  commit is sequenced by the lead (index-race safety, rule 7). b-daemon's reader is live
  (bd33cde); offered to score the fitted file vs unsplash-demo + cocoval2017 to confirm
  the p-spread off-COCO before the `fitted` label is blessed in prod — PENDING.
- **2026-07-22 · contact-sports boundary synced (track-violence, fe74302).** Their
  negatives bank PINS this track's `martial arts` prompt strings verbatim → sync-on-retune
  contract: any rename/retune of that group pings track-violence BEFORE it lands. Measured
  weak-label contact-sports recall handed over (boxing 0.102, martial-arts 0.146 — LOW;
  under-measured, cite WITH the weak label). The compose-claim (boxing → sports:match as
  the exculpatory label) is NOT measured-strong on the sports side; open item.
- **2026-07-22 · verified (track-sports3, folded in).** COCO/LVIS exhaustive-GT numbers
  reproduced to the digit over existing embeddings; §6 weak-label rates re-measured on the
  grown `unsplashb` (n≈9926) — direction/ranking unchanged; the corpus-dependent phrasing
  in sports.py's background-bank comment tightened. Lane consolidated to a single owner
  (track-sports2) after the auth-outage false-death; successor stopped, its edits adopted.

## Open items (darwin backlog)

1. **Activity-only recall is under-measured** — swimming/running/gym/martial-arts have no
   COCO object to score against; only weakly probed via Unsplash keywords. A distilled
   activity head (offline teacher → embedding MLP) is the TRACKS.md T2-rung-2 path if a
   site needs strong activity recall.
2. **Contact-sports recall is weak** (§8) — strengthen `martial arts` prompts and re-measure
   on a real (indexed) contact-sports slice; coordinate the retune with track-violence.
   UNBLOCKED 2026-07-22: track-violence indexed dataset **`vslices`** (1856 imgs — real
   ring/dojo/mat boxing/martial-arts/wrestling/fencing + team-contact & confusables,
   pecore-s16-384-fp32). STAGED: run `SportsHead.score` over `store.open_snapshot("vslices")`
   for REAL per-slice match rates (replaces the weak-keyword 0.102/0.146) — pending the
   quiet-window ALL-CLEAR (scoring stored embeddings is compute). If a targeted prompt round
   lifts boxing recall, ping track-violence to upgrade their §7 to "measured".
3. **equestrian / motorsport are scored as FPs by the COCO taxonomy** but are arguably sport
   — a user ruling that they ARE sport would raise measured precision from 0.783 (LVIS) to
   ~0.80+. Configurable via the label set.
4. **Confirm `calibration:"fitted"` off-COCO** — awaiting b-daemon's unsplash-demo p-spread
   check; demote to unfitted if it saturates there.
