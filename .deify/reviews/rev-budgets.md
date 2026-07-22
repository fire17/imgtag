# rev-budgets.md — ADVERSARIAL review of BUDGETS.md (B1–B17)

> Reviewer: rev-budgets (adversarial lane). Date 2026-07-22. Mandate: REFUTE, not check.
> Sources read: BUDGETS.md · VISION.md (verbatim law) · UNKNOWNS.md · ORACLE.md §5–6 ·
> research/priorart.md §3 + §5 · research/measured-numbers.md · IA.md · data/ on disk.
> Verdict up front: **BUDGETS.md is not lockable.** One budget is mathematically
> impossible, one contradicts the project's own measured number, three vision
> deliverables have zero rows, and four "test commands" cannot assert their threshold.
> Counts: **8 CRITICAL · 12 IMPORTANT · 7 MINOR**.

---

## CRITICAL

### C-1 · B5 is mathematically impossible. Proven on disk.

B5: `"vehicle" recall@100 ≥0.80` on COCO supercategory queries.

Measured just now from `data/coco/annotations/instances_val2017.json` (5,000 images):

| supercategory | images containing it | recall@100 ceiling |
|---|---:|---:|
| vehicle | **1,160** | **0.086** |
| person | 2,693 | 0.037 |
| furniture | 1,257 | 0.080 |
| animal | 1,016 | 0.098 |
| appliance | 320 | 0.313 |

A perfect oracle returning 100 results out of 1,160 relevant scores recall@100 = 0.086.
B5 demands 0.80 — **9.3× above the ceiling**. It cannot be passed by any system, ever.
A budget that can never go green is worse than no budget: the first builder who runs it
will either (a) silently redefine the metric to make it pass, or (b) mark it "known-red"
and the whole table loses force.

Second defect, independent of the math: B5 measures an **aggregate**, but the vision's
actual demand is **child coverage** — "if a search for 'vehicle' i should also find all
of these images with the cars as well as all motocycles and other vehicles". An aggregate
recall number is passable while motorcycles (159 imgs) and boats (121 imgs) score zero,
because cars (535) dominate. UNKNOWNS I1 already says "report per-child recall breakdown";
BUDGETS does not enforce it. Reporting is not a gate.

**Exact fix — replace B5 with three assertions at one operating point:**

```
| B5 | "sematically flexible … 'vehicle' … all motocycles and other vehicles" |
  hypernym retrieval on COCO supercategory queries, corpus CORPUS-A (coco5k) |
  (a) precision@100 ≥0.85, mean over the 6 supercats {vehicle, animal, food, furniture,
      appliance, sports};
  (b) per-child recall@R (R = |relevant| for that supercat): mean over children ≥0.55,
      MIN over children ≥0.35, and NO child at 0.00;
  (c) all children of the queried supercat present ≥1× in top-100 |
  `uv run imgtag bench quality --hypernym` | provisional |
```

Rationale for (b): recall@R (R-precision style) is the only recall variant that is not
capped by an arbitrary K, and the min-over-children clause is the literal encoding of the
vision sentence. Per-child table is emitted with the verdict, always.

### C-2 · No corpus is named anywhere → 11 of 17 budgets are unfalsifiable. And the corpus that would make B1/B2 honest does not exist on disk.

Not one row says what images the number is measured on. `bench index --n 100` on
COCO val2017 (median **640×480**, max dimension 640 — verified) versus on real 12MP
camera JPEGs differ by **1–2 orders of magnitude in decode cost** — priorart §1.2
measured 163–287 ms/img full-res decode vs ~4–8 ms for a 640×480. That is the single
largest term in B1/B2 (ORACLE C3, the chaser's #1 danger).

Verified corpus inventory on disk right now:

| dir | images | note |
|---|---:|---|
| data/coco/val2017 | 5,000 | 640×480 median, max side 640 |
| data/unsplash/images | 2,000 | **1080 px wide** — not big either |
| data/quick500 | 500 | COCO subset (instances_quick500.json) |
| data/lvis, data/openimages, data/caltech101 | **0** | annotations only |

Consequences, all of them real:
1. **B1/B2 will be measured on the easiest possible images** and the resulting img/s will
   not survive contact with a user's photo folder. That is exactly the "regression with
   better docs" failure priorart §5 action 1 warns about, arriving through the back door.
2. **B3 says "@10k". There is no 10k corpus.** Unique images available = 7,000
   (5,000 + 2,000; quick500 ⊂ coco). B3, B8's 10k disk cap, and B2's 10k run cannot be
   executed today.
3. Two budgets silently disagree about which world they live in and nobody will notice.

**Exact fix — add a `Corpus` column to the table and define the corpora in a new §Corpora:**

```
CORPUS-A  coco5k     5,000 COCO val2017 (640×480 median). Ground truth: 80 exhaustive
                     classes. Used by: all quality/FP budgets.
CORPUS-B  photo10k   10,000 real photos, p50 ≥8 MP, p95 ≥12 MP, mixed EXIF orientation.
                     MUST BE FETCHED — does not exist yet (Unsplash full-res via
                     scripts/fetch_unsplash_demo.sh with the full-size URL, not the 1080px
                     one). Used by: B1(hard), B2, B8, B10, B11, B12, B15.
CORPUS-C  mixed10k   coco5k + 5k from photo10k. Used by: B3, B13, B14.
CORPUS-D  poison     ~120 hostile files (see B21). Used by: B21.
```

Every perf row then carries BOTH numbers where they differ (small-image and
real-photo). A single img/s figure with no image-size distribution is marketing.

### C-3 · B1 is set 2.6× BELOW a number already measured on this exact machine, and its stretch is anchored to a figure the project's own ORACLE flags as stale.

`research/measured-numbers.md` + priorart §3 row **0b**: **157.9 img/s, CPU-only,
CLIP-B/32 int8 ONNX, batch 8, on this M3 Max** — and the lane labels it a *lower bound*
(taken under load). priorart §3 row 2 hands the project an explicit target: **"Target
≥150 img/s on the dev target, ⌂ ≥10 img/s on the edge floor."** priorart §5 action 1:
*"A budget that ships something several times slower than an existing MIT CLI is not
'blazing fast' — it is a regression with better docs."*

BUDGETS B1 says **≥60 img/s**, stretch ≥120. The re-derivation note claims it was raised
"30→60 (rclip evidence)" — but it stopped at less than half the handed target and below
the encode number already in hand. A pipeline hitting 60 img/s on 640×480 COCO images
while its encoder alone does 158 is **63% waste**, which is precisely the inefficiency
this project exists to attack in immich.

Second defect: the stretch reads `≥120 = beat rclip-CoreML`. ORACLE §8 field log,
2026-07-22 10:20: *"rclip README's 119 img/s conflicts with its own PR #249 (~180 img/s
CoreML) — README is stale, PR is measured."* B1's stretch is built on the number the
project's own oracle already retired. The real bar is **180**.

Third defect: B1 has no head-to-head clause, so it is absolute-only. An absolute number
can be met on easy images while still losing to rclip on the same folder.

**Exact fix:**

```
| B1 | "blazing fast … processing and indexing" | sustained end-to-end index throughput
  (files-on-disk → searchable), CPU-only, DEFAULT (polite) config |
  CORPUS-A ≥150 img/s (stretch ≥180 = rclip's measured CoreML rate, PR #249, beaten on
  pure CPU) · CORPUS-B ≥60 img/s · ⌂-proxy ≥10 img/s (see §⌂) ·
  **HEAD-TO-HEAD GATE: red if imgtag img/s < rclip img/s on the same corpus, same
  machine, same run — regardless of the absolute number** |
  `uv run imgtag bench index --corpus {A,B} --headtohead rclip` | provisional |
```

The head-to-head clause is the part that cannot be gamed by choosing easy images, and it
is the direct discharge of priorart §5 action 1.

### C-4 · B7 is a free parameter. The FP guard is fake, and it is gameable against B5/B6 with no cost.

B7: *"≤2% of returned results above score threshold."* The threshold is chosen by the
builder. Set τ = 0.99 → nothing is ever returned above it → B7 = 0.0%, green, forever.

Does anything push back? No. B5 and B6 are **ranking** metrics (recall@K, precision@10) —
they consume the sorted score list and never look at τ. So τ can be raised without moving
B5/B6 by a single point. **A builder can pass B7 perfectly while destroying the product's
actual recall, and the budget table will be all green.** This is the exact
game-one-against-the-other hole the review was asked to find, and the guard does not exist.

The reverse is equally open: τ = 0 makes every query return top-K, B7 explodes, but
nothing else notices — meaning B7 also fails to *protect* anything.

**Exact fix — one frozen operating point, both sides asserted in the same run:**

```
| B7 | "lack of false positives" / "minimization of any false positives" |
  calibrated no-match threshold τ, FROZEN in the model manifest, fitted on a HELD-OUT
  split (COCO train2017 sample — never val). At that same τ, in one run:
  (a) absent-concept leakage ≤2% — fraction of the ≥20 absent-concept queries returning
      ≥1 result above τ. Absent list auto-derived: OI-600 / LVIS labels with 0 annotations
      on CORPUS-A (both annotation files already on disk), plus 5 hand-written absurdities;
  (b) AT THE SAME τ, mean recall@10 over the 80 present COCO categories ≥0.70;
  (c) τ is recorded in the bench output and in the index manifest; a τ change without a
      re-run of (a)+(b) is a red.
  Passing (a) by sacrificing (b) is RED, not a trade. |
  `uv run imgtag bench quality --negatives` | provisional |
```

Cross-ref: ORACLE's chaser calls the honest no-match "the single most breakable
differentiator" and demands re-calibration whenever the model or prompts change. Clause
(c) is what turns that warning into a gate.

### C-5 · Vision sentence with ZERO budget coverage: result provenance (dataset + path + id).

VISION, verbatim: *"get exactly both from which of the datasets the resulting images are
from and obviously the image path and or id"*.

This is the most literal, most testable, least ambiguous requirement in the entire vision
statement, and **BUDGETS.md has no row for it**. IA.md §View tier 2 mentions it as a UI
element ("vision-mandated"), but IA is a design doc, not a gate — nothing fails a build if
a hit comes back with a null dataset or a stale path. And this is a *correctness* surface
with real failure modes: shards from two datasets in one search, an id that no longer
resolves after a file moves, a path relative to the wrong root.

**Exact fix — new row:**

```
| B18 | "get exactly both from which of the datasets the resulting images are from and
  obviously the image path and or id" | search-result provenance completeness &
  resolvability, over 200 queries × top-50 on CORPUS-C |
  100% of hits carry non-null {dataset_slug, path, image_id}; 100% of paths exist on
  disk; 100% of ids satisfy xxhash64(file bytes) == id; 100% of dataset_slug values match
  the manifest owning that shard; 0 cross-dataset misattribution. Any single failure = red
  (this budget has no tolerance band — it is a correctness invariant). |
  `uv run imgtag bench provenance` | provisional |
```

### C-6 · Vision deliverable with ZERO budget coverage: the global agent skill.

VISION, verbatim: *"plus a globally available skill so agents will be able to use this to
either tag (process/index) some photos or new datasets, get info about datasets, manage
them, and run searches."*

Four verbs — tag, info, **manage**, search — and not one budget row. priorart §4 move 8
calls the agent-native lane *"the single least-contested lane in the entire survey"* — the
project's strongest strategic position is completely unbudgeted, so it will be the first
thing that ships half-done under time pressure. "manage them" in particular (create /
rename / re-index / delete a dataset) has no gate anywhere in the project, and delete is
also the storage-leak vector (see I-6).

**Exact fix — new row:**

```
| B20 | "a globally available skill so agents will be able to … tag … get info about
  datasets, manage them, and run searches" | machine-API conformance across all 4 verbs |
  Every verb (index, info, manage{create,rename,reindex,delete}, search) is callable
  headlessly with `--json`: valid JSON on stdout (human text on stderr only, zero ANSI in
  stdout), documented exit codes for the 5 error cases (unknown dataset · model/manifest
  hash mismatch · corrupt index · model unavailable offline · zero results above τ), zero
  interactive prompts, and per-call latency: search ≤ B3 p95 + 50 ms · info ≤200 ms ·
  index returns a job id ≤500 ms and does not block · delete leaves 0 orphan bytes under
  ~/.imgtag (byte-diff before/after). |
  `uv run imgtag bench skill-contract` | provisional |
```

### C-7 · B14 and B4 have no harness. Their "test command" cannot assert their threshold, which breaks the file's own law.

BUDGETS header law: *"A regression is a build failure, not a ticket."* ORACLE §6:
*"`uv run imgtag bench all` = the full budget table; exits nonzero on any budget red."*

- **B14** test command is literally `manual + devtools trace`. A human looking at a flame
  chart cannot be in `bench all`, cannot exit nonzero, cannot fail a build. B14 is not a
  budget; it is an intention.
- **B4** claims `uv run imgtag bench app-search` measures *"keystroke→results-painted
  p95"*. A Python CLI has no access to a browser's paint timeline. As written it will
  degenerate into "measure the HTTP endpoint and call it painted" — which is B3 with extra
  steps and a 100 ms fudge, i.e. a fabricated number for the metric the user will actually
  feel.

Also unresolvable as written: **16 ms/frame at p95 is not achievable in a browser** across
GC pauses; a real spec separates p50 from tail.

**Exact fix — one real harness, dev-only dependency (playwright is a dev dep, ADR-7's
runtime dep list stays intact):**

```
| B4 | "google photos in app search … instantly found" | keystroke→results-painted,
  CORPUS-C, warm daemon, Chromium via Playwright/CDP. App emits
  performance.mark('imgtag:key') on keydown and performance.mark('imgtag:painted') inside
  the requestAnimationFrame AFTER the results grid commits; bench reads
  performance.getEntriesByName over ≥50 trials × 10 distinct queries (no repeats — see
  I-3) | p50 ≤80 ms · p95 ≤150 ms | `uv run imgtag bench app-search` (playwright, dev dep)
  | provisional |

| B14 | UI quality bar (impeccable) | scripted scroll of a 10k-thumbnail virtualized grid
  at 1000 px/s for 5 s under CDP Tracing (devtools.timeline), Chromium, 1440×900, 6× CPU
  throttle OFF (report a 4× throttled run separately) | frame interval p50 ≤16.7 ms ·
  p95 ≤20 ms · **0 long tasks >50 ms** · DOM node count <5,000 at any point (proves
  virtualization) | `uv run imgtag bench ui` | provisional |
```

If the project will not take a Playwright dev dep, then B4/B14 must be **deleted and
replaced with what the CLI can actually measure** (server-side time-to-first-byte for the
results payload) plus an explicit honesty line saying the perceived-latency budget is
NOT measured. Either is defensible. Keeping a threshold with no instrument is not.

### C-8 · The quantization-quality guard is missing. B16 does not cover it, and B1 is passed *through* quantization.

B16 asserts fast-path **preprocessing** parity vs the reference pipeline. Quantization is
not preprocessing. ORACLE ADR/C2 makes precision sweep (fp32 / int8-dynamic / int8-static)
the primary speed lever; priorart row 0b's 157.9 img/s *is* an int8 number. So the main
route to passing B1 is a precision change that **no budget checks for quality loss within
the same model**:

- B16: compares preprocessing paths → blind to weight quantization.
- B17: only demands ≥ +5 R@10 over a 2021 ViT-B/32-openai control → a SigLIP2 int8 model
  that lost 4 points to quantization still clears that bar comfortably.
- B6/B5: measured on whatever model ships — they see the *absolute* level, never the
  *delta from fp32*, so a 3-point self-inflicted loss is invisible.

This is exactly the clip.cpp failure mode (31.4% vs 66.6% top-1, identical weights) that
UNKNOWNS C4 and the ORACLE chaser both name as the field's classic trap — and the
current table guards one half of it (interpolation) while leaving the other half
(precision) open.

**Exact fix — new row:**

```
| B24 | anti-silent-quality-loss, precision axis (C2/C4 lesson) | shipped precision vs the
  SAME model at fp32, CORPUS-A | mean cos(v_quant, v_fp32) ≥0.995 AND min ≥0.97 AND
  |Δ precision@10| ≤0.01 abs AND Δ R@10 ≥ −1.0 pt. A precision that fails this is not
  shipped as default even if it is faster (it may be offered behind an explicit flag with
  the measured deltas printed). | `uv run imgtag bench parity --precision` | provisional |
```

---

## IMPORTANT

### I-1 · B1 and B2 measure the same quantity and disagree by 2.16×.

B1 ≥60 img/s ⇒ 10k in **2.78 min**. B2 says target ≤3 min, **pass ≤6 min** — i.e. 27.8
img/s, less than half of B1's hard floor. Two rows, one physical quantity, two thresholds;
a report can truthfully say "B2 pass" while B1 is red. Any softer number in the table
becomes the one that gets quoted.

**Fix:** B2 stops being an independent threshold and becomes the *validation* of B1's
projection (see I-2). Delete "pass ≤6min". `t_10k ≤ 10000 / B1_floor` by construction.

### I-2 · B2's "100-img test → 10k projection" is an unvalidated extrapolation.

The vision sanctions the shortcut verbatim (*"tests scales to 100 to not waste time"*), so
the shortcut stays — but nothing in BUDGETS validates that the projection is honest. At
n=100 there is no thermal steady state, no page-cache pressure, no shard-flush/manifest
tail, no fragmentation. Real pipelines are routinely 20–40% slower at 10k than a 100-image
sprint predicts, and that is the number the user cares about.

**Fix:**

```
| B2 | "time to process 10,000 images on cpu (tests scales to 100 to not waste time)" |
  wall time for a REAL 10,000-image end-to-end run + projection fidelity |
  CORPUS-B: t_10k ≤ 6 min (target ≤3 min) · CORPUS-A: t_10k ≤ 2 min ·
  **projection gate: |t_10k_projected_from_n=100 − t_10k_actual| / t_10k_actual ≤ 0.15**
  (this is what licenses the n=100 shortcut for day-to-day CI; the full run is required
  once per lock and once per darwin round) |
  `uv run imgtag bench index --n 100` (CI) · `--n 10000` (lock/darwin) | provisional |
```

### I-3 · B3 is gameable by the query cache the architecture explicitly ships.

ADR-5 / UNKNOWNS I3 mandate an LRU query-embedding cache. B3 says "warm". Run the same
20 queries 100× each and p50 → ~1 ms (scan only, 0.47 ms measured), p95 trivially green,
and the number is meaningless — the *first* time a user types a query is the case that
matters, and it is the case the cache never covers.

**Fix:**

```
| B3 | "especially during inference/search … instantly" | search e2e latency, warm daemon,
  CORPUS-C | over ≥200 DISTINCT never-before-seen queries (cache cold per query, cache
  never pre-warmed with the test set): p50 ≤50 ms · p95 ≤120 ms. Cache-HIT latency
  reported separately as an informational number, never as the budget. Also assert the
  daemon is resident (cold-start path is B13, not B3). |
  `uv run imgtag bench search --queries 200 --no-cache-prewarm` | provisional |
```

### I-4 · B6's "mean over categories" hides total category failure — and two COCO categories cannot reach 0.90 at all.

Mean precision@10 ≥0.90 over 80 categories is satisfied by 72 categories at 1.00 and
**8 categories at 0.00**. A model that is completely blind to 8 of 80 concepts ships green.

Separately, a ground-truth ceiling: `toaster` has **8** positive images in val2017 and
`hair drier` has **9**. precision@10 for toaster is capped at 0.80 — the metric is
ill-defined for them.

**Fix:**

```
| B6 | "when i search for 'car' - all of the images with one or more cars" | per-category
  precision@k, k = min(10, N_pos), over all 80 COCO categories, CORPUS-A |
  mean ≥0.90 AND 10th-percentile ≥0.70 AND **min ≥0.40** AND zero categories at 0.00.
  Full 80-row table emitted with the verdict. |
  `uv run imgtag bench quality` | provisional |
```

### I-5 · B8's RSS is undefined for a multi-process design, and its idle number is gameable against B3/B13.

Two holes:
1. ADR/UNKNOWNS C3 + the ORACLE risk register call for **process-pool decode workers**.
   `peak RSS ≤1.5 GB` measured on the parent only understates real memory by ~Nx. mmap'd
   shards also make RSS ambiguous (mapped-and-touched pages count).
2. `server idle RSS ≤400 MB` is trivially passed by **unloading the model when idle** —
   which is precisely immich's documented sin (`model_ttl=300` → 60–70 s cold search) that
   ADR-5 exists to avoid. Passing B8 by breaking B3/B13 is the second gameable pair in the
   table.

**Fix:**

```
| B8 | "not taking too many resources … old computers or edge devices" |
  (a) peak RSS during indexing = **sum of RSS across the whole process tree**, sampled
      ≥10 Hz, CORPUS-B; (b) daemon idle RSS after 10 min idle; (c) model+index+thumb-cache
      on disk for 10k |
  ≤1.5 GB · ≤400 MB · ≤500 MB (thumbs ≤200 MB of that, LRU-capped) —
  **and immediately after the idle-RSS sample, a search must still meet B3 p95
  (asserted in the same run): the text tower must be resident, not unloaded.** |
  `uv run imgtag bench resources` | provisional |
```

### I-6 · B12 tests 2 of the 6 leak classes the file itself enumerates, and its soak is too short to detect a leak.

The §"Leak classes" block names six: storage, memory, fd, CPU, compute, disk-bloat. B12
asserts only memory (RSS drift) and fd. VISION says *"no data or performance or compute or
leaks of any kind"* — **compute** is called out by the user by name and has no metric.

And duration: "full-dataset run" on CORPUS-A at even 60 img/s is **83 seconds**. No leak
worth the name is visible in 83 seconds; a 1 MB/1000-images leak is 5 MB and inside noise.

**Fix:**

```
| B12 | "no data or performance or compute or leaks of any kind" | ≥30-min continuous soak
  (corpus looped), 6 leak classes |
  memory: OLS slope of process-tree RSS over the run, post-warmup, ≤5% total growth AND
  slope 95% CI includes ≤0.5 MB/min · fd: 0 drift · threads: 0 drift ·
  storage: 0 files left under ~/.imgtag/tmp and 0 orphan thumbnails after job end and
  after a dataset delete (byte-diff) · disk-bloat: index bytes/image within 5% of the
  theoretical dim×dtype size · CPU: idle-daemon CPU ≤0.5% over 60 s (proves event-driven,
  no spin-polling) |
  `uv run imgtag bench soak --minutes 30` | provisional |

| B25 | "no … compute … leaks of any kind" (compute-leak class, user-named) |
  re-index of an UNCHANGED dataset | 0 images re-embedded (mtime+hash gate), completes in
  ≤ 5 s per 10k, and the manifest is byte-identical except a timestamp |
  `uv run imgtag bench reindex-noop` | provisional |
```

### I-7 · B15 (politeness) silently taxes B1, and BUDGETS never says which config B1 is measured in. On the edge floor the tax is 25%.

B15 requires "≥1 core left free unless `--full-speed`". On the 16-core dev target that is
a ~6% throughput tax. On the ⌂ 4-core edge floor it is **25%**. If B1 is measured with
`--full-speed` and B15 is measured separately, then the shipped default configuration
achieves neither number and both rows are green. Classic pair-gaming.

Also, B15's own assertion is currently unstated as a procedure. It *is* measurable — here
is the concrete form, which the row should carry:

**Fix:**

```
| B15 | indexing politeness ("run … even on old computers") | OS priority, reserved
  capacity, and machine responsiveness during indexing |
  (a) every process in the indexer tree has nice ≥10 (os.getpriority per pid);
  (b) worker_count × intra_op_threads ≤ cpu_count − 1;
  (c) system-wide CPU utilization sampled 10 Hz for 30 s ≤ ((N−1)/N)×100% + 5 pp;
  (d) responsiveness probe: an external control process running a fixed 200 ms busy-loop
      sees ≤1.30× wall-time inflation while indexing (this is the metric that actually
      encodes "the machine stays usable");
  (e) `--full-speed` run asserts nice == 0 and workers == N (proves the escape hatch works)
  | `uv run imgtag bench politeness` | provisional |
```

**And add to B1's row:** *"measured in the DEFAULT (polite) configuration; a `--full-speed`
number may be reported additionally, clearly labelled, and never as the headline."*

### I-8 · B16's mean cosine is the wrong statistic — it is blind to exactly the failures it exists to catch.

`mean cosine ≥0.99 on quick500`: with 500 images, ten images at cosine 0.30 still yield a
mean of 0.976 — and if the parity bug is narrow (EXIF-rotated images, CMYK JPEGs,
grayscale, palette PNGs with alpha) the affected set *is* small. That is a
catastrophic-but-rare failure sailing past a mean. The clip.cpp disaster was a *systematic*
shift, which a mean catches; the draft()-decode risks in this project are *conditional*,
which a mean does not.

`"quality deltas within noise"` is not a number and cannot fail a build.

**Fix:**

```
| B16 | anti-silent-quality-loss (clip.cpp lesson), preprocessing axis | fast-path vs
  reference-pipeline embeddings on quick500 + the EXIF/format subset of CORPUS-D |
  mean cos ≥0.995 AND p1 cos ≥0.99 AND **min cos ≥0.98** AND |Δ precision@10| ≤0.01 abs
  AND Δ hypernym min-child recall ≥ −0.02 (bootstrap 95% CI over 1000 resamples reported).
  Any fast path failing this is disabled, not shipped. |
  `uv run imgtag bench parity` | provisional |
```

### I-9 · B17 is passable by a free model swap, and its split is not comparable to any published number — so it is not "auditable".

Two defects:

1. **Too soft.** `≥ +5 R@10 over OpenCLIP ViT-B/32-openai`. priorart §4 move 4 records
   immich's own measurement that `ViT-B-32__laion2b-s34b-b79k` is **+7.7 recall points for
   +0.03 ms** over that exact control — a swap requiring zero engineering. B17 as written
   is cleared by picking a different 2022 checkpoint and doing nothing else, while the
   modern tier (SigLIP2-B/16, 84.86 vs 69.90 = **+15**) is 3× further out. A budget below
   a free upgrade is not a budget.
2. **Not auditable.** priorart §5 action 3 asked for recall pinned *against a public
   benchmark* so quality is externally checkable. B17 uses a self-run control on an
   unspecified COCO split. R@10 numbers in the literature are reported on the **Karpathy
   5k test split**; whether COCO val2017 is identical to it is widely assumed but must be
   verified by image-id intersection before any comparability is claimed.

**Fix:**

```
| B17 | "state of the art" quality, auditable | COCO-caption text→image R@1/R@5/R@10 on
  the Karpathy 5k test split (identity with val2017 VERIFIED by image-id intersection and
  recorded in the bench output; if it does not hold, val2014 + karpathy json are fetched) |
  default model R@10 ≥ control + **12 pts** (control = OpenCLIP ViT-B/32-openai run through
  OUR pipeline, same corpus, same run) AND absolute R@10 within 2 pts of the model card's
  published figure (proves our pipeline is not degrading the model) AND both numbers plus
  hardware published in the results table |
  `uv run imgtag bench quality --retrieval` | provisional |
```

### I-10 · B10's ETA spec is ambiguous in the direction that makes it free to pass, and cannot be tested at n=100.

*"ETA within ±20%"* — ±20% **of what**? Of remaining time (hard, meaningful) or of total
elapsed+remaining (trivially easy near the end)? *"after 10% done"* — at n=100 that is 10
images, where rate variance is enormous. And *"≤1s stale max"* does not say stale relative
to what, nor does it require a heartbeat when a single image takes >1 s (a 12 MP decode
plus a queue stall can).

**Fix:**

```
| B10 | "see live progress, how many images a sec its indexing, projected etas" |
  progress-stream freshness, rate accuracy, ETA error — CORPUS-B (n≥10,000; the n=100
  bench asserts only freshness) |
  (a) freshness: max gap between consecutive progress events ≤1.0 s INCLUDING idle/stall
      periods (heartbeat required), and max(event_ts − manifest_commit_ts) ≤1.0 s;
  (b) rate accuracy: reported rolling-10 s img/s vs actual from manifest counts, mean
      absolute error ≤10%;
  (c) ETA error at the 25/50/75% marks:
      |eta_predicted_finish − actual_finish| / (actual_finish − t_mark) ≤ 0.20;
      at the 10% mark ≤0.35 (honest about early-run variance);
  (d) progress emitter CPU time ≤1% of run wall time (proves event-driven, no spin-poll) |
  `uv run imgtag bench progress` | provisional |
```

### I-11 · B11's "zero blocking" is not a measurable predicate, and there is no under-load search budget at all.

"Zero blocking" cannot be asserted — you cannot prove the absence of a lock from the
outside. What you *can* assert is behavioural. Worse, the user's real experience during
indexing is unbudgeted: B3 is measured warm and idle, but 16 cores are saturated during
a job, so search latency during indexing — the exact scenario the vision demands
(*"instantly search … while the processing is still ongoing"*) — has no number.

**Fix:**

```
| B11 | "instantly search … while the processing is still ongoing" | visibility latency +
  behavioural non-blocking, during a full CORPUS-B index with a searcher issuing 1 query
  per 100 ms from a distinct query pool |
  (a) visibility: every 500 images, the newest indexed image is retrievable by a
      known-true query within ≤2.0 s of its manifest commit;
  (b) reader not blocked: p99 search latency during indexing ≤ 2× B3 p95, 0 queries >1 s,
      0 errors/stale-manifest exceptions;
  (c) writer not blocked: index throughput during the query storm ≥95% of the
      query-free run;
  (d) coverage honesty: the reported "indexed so far" count equals the manifest count
      exactly at every sample (no over- or under-claim) |
  `uv run imgtag bench concurrent` | provisional |
```

### I-12 · B9's "primary model" is gameable by splitting towers, and ≤150 MB is soft against a documented 85.6 MB.

priorart §3 row 6: clip.cpp already ships **85.6 MB** at 4-bit, and the lane's own note is
*"aim to beat 85.6 MB while beating ViT-B/32 quality"*. measured-numbers records Ente
shipping 143 MB image + 67 MB text = 210 MB total. B9 says "≤150 MB **primary** model" —
under that wording, a 145 MB image tower plus a 90 MB text tower is green at 235 MB.
priorart §4 move 10 also *wants* the towers split (so a search loads ~35 MB) — good design,
but it means the budget must be stated on the **sum**.

**Fix:**

```
| B9 | "super state of the art small but powerfull" | sum of ALL model artifacts required
  for index + search (image tower + text tower + tokenizer + tag-vocab table), on disk,
  in the shipped precision | ≤150 MB total (stretch ≤90 MB = beat clip.cpp's 85.6 MB while
  beating ViT-B/32 quality per B17) · additionally: search-only resident footprint
  ≤50 MB of model bytes (tower split, priorart move 10) | `uv run imgtag bench artifacts`
  (not "inspect artifact" — must be machine-asserted) | provisional |
```

### I-13 · No robustness budget. "make sure it all works" and "increased reliablility correctness" have no gate.

Nothing in the table survives a bad file. A single truncated JPEG that raises inside a
decode worker can kill a 10k job at image 9,998 and every budget stays green because
every budget runs on clean, curated academic corpora.

**Fix:**

```
| B21 | "make sure it all works" / "increased reliablility correctness" | poison-corpus
  robustness, CORPUS-D (~120 hostile files: truncated JPEG, 0-byte, CMYK JPEG, 16-bit and
  palette+alpha PNG, all 8 EXIF orientations, HEIC, animated GIF, .jpg that is really a
  PNG, unicode/emoji filenames, symlink, symlink loop, 300 MP decompression bomb,
  read-permission-denied file) |
  0 crashes · 0 hangs (every file resolved or skipped within 5 s) · every failure recorded
  as {path, reason} in the job status file · ≥99.5% of the VALID files indexed · process
  exits 0 with a nonzero `failed` count, never a traceback · the decompression bomb is
  refused by a pixel-count cap, not by OOM |
  `uv run imgtag bench robustness` | provisional |
```

### I-14 · "no data … leaks" is read only as memory/fd. The data-egress reading is unbudgeted.

VISION: *"no **data** or performance or compute or leaks of any kind"*, in a project whose
first law is *"using only the cpu"* and whose users are indexing private photo libraries.
"Data leak" most naturally means *bytes leaving the machine*. B12 covers RSS and fds. There
is no assertion that a local photo indexer never phones home.

**Fix:**

```
| B22 | "no data … leaks of any kind" (egress reading) | network egress in steady state |
  0 connections to any non-loopback address across the whole process tree during index +
  search (psutil.net_connections sampled 1 Hz for the full bench run). The first-run model
  download is the ONLY permitted egress: it is explicitly announced, logged with the exact
  URLs, and must not recur once the cache is warm (second run = 0 external connections). |
  `uv run imgtag bench egress` | provisional |
```

### I-15 · The ⌂ edge-floor honesty protocol: the 4-thread throttle is NOT a defensible proxy. Position taken.

UNKNOWNS I6 promises ⌂ numbers are *"projected via 4-thread throttled run (documented
proxy), NOT yet live-verified on real old hardware"*. The honesty *labelling* is good and
must stay. The **proxy itself is not honest** — it models one variable (core count) and
holds every other variable at 2024-flagship level:

| axis | M3 Max @ 4 threads | real 2015 4-core x86 | ratio |
|---|---|---|---|
| memory bandwidth | ~400 GB/s unified | DDR3-1600 dual-ch ~25 GB/s | **~16×** |
| SIMD/ML kernels | ARM NEON + MLAS ARM path | SSE4.2/AVX2, no VNNI, different MLAS kernels | not a scalar factor — *different code* |
| single-core IPC/clock | ~4.0 GHz, wide OoO | ~2.6 GHz Haswell-class | ~2–2.5× |
| storage feeding decode | NVMe ~5 GB/s | SATA SSD/HDD 0.1–0.5 GB/s | 10–50× |
| cache | 16 MB+ L2 per cluster | 6–8 MB L3 total | ~2× |

A throttled M3 Max is not a slow computer; it is a fast computer being polite.
Compounding the ratios, the proxy plausibly **overestimates a real 2015 4-core by 4–8×**,
and it overestimates in the flattering direction — which is the definition of
self-deception, not conservatism. ORACLE's chaser rule ("believe your measurement") does
not apply, because this is not a measurement.

The evidence needed already exists in the research and is being ignored: **real x86 CPU
data points are on hand** — photofield-ai 20 img/s on a 2014 6-core i7, and rclip 1.9 img/s
on a 2016 Celeron J3455 with fp32 ViT-B/32 (priorart §3 rows 0, 1).

**Position + exact fix — three changes:**

1. **Rename the proxy so it cannot be misread.** Not "⌂ projected" but
   `⌂-ub` = *upper bound of the edge estimate (M3 Max, 4 threads) — NOT a prediction of
   any real machine*. An upper bound is an honest object; a "projection" is a claim.
2. **Add a second row that stays empty until it is real.** `⌂-real: NOT MEASURED` with the
   target hardware named (any x86 without AVX-512, 4 cores). An honest blank beats a
   flattering number, and it creates the itch that gets it filled.
3. **Fill it cheaply — this is a cents-and-minutes problem, not a hardware problem.**
   Any x86 box with `docker run --cpus=4 --memory=8g` (a friend's machine, a CI runner, a
   $0.02/hr old-generation cloud instance) yields ONE honest x86 datapoint. Until it
   exists, derive the ⌂ *expectation* from the real anchors above (photofield-ai's 20 img/s
   on 6 older cores ⇒ ~13 img/s on 4, times an int8 factor to be measured), not from the
   throttle. **And hard rule: no ⌂ number appears in any README, landing page, or public
   claim until `⌂-real` is populated.** ORACLE §5's row "Old-machine claims challenged"
   currently accepts the label as the mitigation; the label is not enough, the empty row is.

### I-16 · The "lightweight" install footprint is in UNKNOWNS but not in BUDGETS.

UNKNOWNS I8 sets *"Target install ≤150MB wheel-tree excluding models"* and forbids
torch/transformers at runtime. VISION: *"very very lightweight poc app"*, *"small but
powerfull"*. Neither is a budget row, so neither can fail a build — and dependency creep
is the single most common way a "lightweight" project stops being one.

**Fix:**

```
| B23 | "very very lightweight" / "small but powerfull" | installed footprint & dependency
  discipline | `uv sync --no-dev` tree ≤150 MB excluding models · runtime import set is a
  subset of ADR-7 {onnxruntime, numpy, Pillow, certifi/httpx, micro-server} · asserting
  `import torch` and `import transformers` both FAIL in the runtime env · wheel builds and
  installs on a machine with no Docker, no Postgres, no GPU |
  `uv run imgtag bench footprint` | provisional |
```

### I-17 · B13 does not say what "cold" means, and the two meanings differ by ~5×.

`≤2 s cold start incl. model load` — with the model file hot in the OS page cache
(the state after literally any previous run) this is a memcpy; with a cold page cache
(the state after a reboot, i.e. the user's actual "first search of the day", which is the
exact wound priorart §3 row 5 says we are attacking) it is a 150 MB disk read plus graph
init. Unstated, the builder will measure the easy one.

**Fix:** `≤2 s with warm page cache (budget) AND ≤4 s with purged page cache (recorded,
`sudo purge` / `vmtouch -e` documented in the bench); both printed, the cold number is the
one quoted in any public claim.`

---

## MINOR

- **M-1 · No lock ceremony.** The header says budgets go `provisional → locked` and "may
  only tighten after locking" — but nothing says who locks, what artifact records the
  locking run, or where the measured numbers live. **Fix:** locking requires a committed
  `bench/results/<date>-<git-sha>.json` containing every budget's measured value, the
  hardware string, the model id+sha, and the corpus ids; the Status cell becomes
  `locked@<sha>`. Without this, "tighten only" is unenforceable because there is no
  recorded prior value.
- **M-2 · B1 "sustained" is undefined.** No warm-up exclusion, no measurement window, and
  no statement of whether shard flush + manifest commit + thumbnail generation are inside
  the timed region. **Fix:** "wall clock from process start to the manifest commit that
  makes the last image searchable, divided by N; first 5% discarded as warm-up; all I/O
  included."
- **M-3 · B14 names no device or viewport.** 16 ms/frame on what — 1440×900 desktop
  Chromium, or a 2015 laptop at 1366×768? **Fix:** covered in the C-7 rewrite (Chromium,
  1440×900, plus a 4× CPU-throttled run reported separately).
- **M-4 · No determinism budget.** Same query + same index must yield the same ordering;
  float non-determinism across thread counts and tie-breaking on equal scores can reorder
  results between runs, which makes every quality budget subtly irreproducible and makes
  darwin-round comparisons noisy. **Fix:** add to B18 or a new row: "identical query on an
  identical manifest returns a byte-identical result list across 10 runs and across
  thread-count settings; ties broken by image id."
- **M-5 · The gallery is only budgeted for frame rate.** VISION: *"i can see the gallery of
  the datasets that have been indexed"* — B14 covers smoothness, nothing covers
  correctness: that every indexed dataset appears, and that per-dataset counts equal the
  manifest. **Fix:** fold into B18 as clause (e): "fleet view lists exactly the datasets
  present under ~/.imgtag/datasets and every displayed count equals its manifest count."
- **M-6 · B3 says "@10k" but the corpus does not exist** (7,000 unique images on disk).
  Covered by C-2; flagged separately because it makes B3 unrunnable *today*, not merely
  ambiguous.
- **M-7 · The 100k-scale claim in UNKNOWNS §3 has no budget row.** "The bench also runs a
  100k synthetic-scale scan test so claims degrade gracefully" — a promise with no
  threshold. **Fix:** either add a row (scan p95 ≤15 ms at 100k synthetic — §1.1 measured
  7.4 ms, so this is comfortable and worth locking) or delete the promise from UNKNOWNS.
  A stated intention with no number is the softest thing in the project.

---

## Summary of the gameable pairs (the review's central structural finding)

| pair | how a builder passes one by breaking the other | is the guard real? |
|---|---|---|
| B7 vs B5/B6 | raise τ → FP rate 0%, ranking metrics untouched | **NO** — fixed by C-4 (shared frozen operating point) |
| B8-idle vs B3/B13 | unload the model when idle → idle RSS green, cold search 60 s | **NO** — fixed by I-5 (search asserted right after the RSS sample) |
| B1 vs B15 | measure B1 with `--full-speed`, ship polite defaults | **NO** — fixed by I-7 (B1 measured in default config) |
| B1 vs B24 (quantization) | quantize for speed, lose quality within-model | **NO — no budget existed**; fixed by C-8 |
| B1 vs B16 (preprocessing) | draft()-decode shortcut degrades embeddings | **YES** — B16 exists; strengthened by I-8 (min/p1, not mean) |
| B1 vs B8-peak | more workers + bigger batches → throughput up, RSS up | **YES** — B8 caps it; only needs I-5's process-tree definition |
| B1 vs B2 | quote the softer row | **N/A — they contradict each other**; fixed by I-1 |
| B1/B2 vs everything | measure on 640×480 COCO, ship for 12 MP photos | **NO** — fixed by C-2 (corpus column + CORPUS-B) |

Three of eight pairs had no guard at all, and two more were unguarded because the budget
did not exist. This is the part of BUDGETS.md that most needs rewriting before any builder
touches it — a budget table whose rows can be traded against each other is a table that
reports green while the product gets worse.

## Bottom line

The table's *shape* is right — vision phrase → metric → threshold → command → status is
exactly the correct discipline, and B16's existence (preprocessing parity) shows real
lesson-absorption from the research. But as it stands it cannot be locked: **B5 can never
pass, B1 is below a number already in hand, B7 protects nothing, three vision deliverables
have no rows, and four thresholds have no instrument.** Post-fix the table is 25 rows
(B1–B25) with a Corpus column, a §Corpora block, a lock ceremony, and — the one thing that
makes the whole thing self-calibrating — the rclip head-to-head gate in B1.

---
---

# AMENDMENT REVIEW — 2026-07-22 10:47, Linux 8GB shared-server primary target

> Second adversarial pass, triggered by VISION-ADDENDA.md (verbatim: *"a 8gb ram (not
> powerful) linux server (that also has other things running and we cant slow down the
> server while we are doing both processing and infrence work)"*) and the 🐧 edits to the
> BUDGETS header, B8 and B15.
> Counts this pass: **5 CRITICAL · 8 IMPORTANT · 4 MINOR.**
> Verdict: the amendment is directionally right and moves fast, but it patched **2 of 17
> rows** for a change that invalidates the machine assumption under **all 17**. As it
> stands the table now measures a machine the product does not run on, and the two rows it
> did update rely on Linux mechanisms that may be inert on the actual target.

## Direct answers to the three questions asked

**Q: Is the ≥85% co-workload probe test well-defined?**
**No — and the number itself is un-sourced.** Four independent under-specifications each
decide the verdict on their own (which probe, when the solo baseline is taken, over what
window, and with no floor on our own throughput so the budget is passable by doing
nothing). Worse, `85%` is not derived from the user's words: the verbatim is *"we cant
slow down the server"* — an absolute — and 85% silently licenses a permanent 15% tax on
someone else's workload for however many hours our index runs. Full spec in **A-C2**.

**Q: Should primary-target thresholds exist beside the proxy ones?**
**Yes, and their absence is the single biggest defect of the amendment.** Only B8 and B15
got 🐧. Every *speed* budget — B1, B2, B3, B13, and B11's under-load behaviour — is still
an M3 Max number with no primary-target counterpart. The result: after an amendment whose
entire point was "optimize for the Linux server, not this machine", **the Linux server has
no throughput budget, no latency budget, and no cold-start budget.** Rows in **A-C1**.

**Q: Is the proxy-labeling protocol honest enough?**
**Half. The labelling is honest; the belief that a throttled M3 Max is a proxy is not.**
The header's own hedge ("even that is an optimistic proxy for old x86") is correct but
under-stated: M3 Max is **ARM**, the target is **x86 AVX2 without VNNI**. These are not
the same kernel, the same quantization economics, or even the same *direction* of result —
clip.cpp's int8-slower-than-fp32 anomaly was measured on an Intel Mac, i.e. exactly the
target's ISA class. The proxy cannot predict the target's precision choice even in sign.
Plus two mechanical gaps: proxy tags live in the header but not in the threshold cells a
builder actually reads, and nothing forbids **locking a 🐧 budget on proxy numbers**.
Fixes in **A-I3** and **A-C3**.

---

## CRITICAL (amendment)

### A-C1 · 15 of 17 rows still assume the proxy machine. The primary target has no speed budget at all.

The header now says the primary target is a weak shared Linux box. B1 still reads
`≥60 img/s (M3 Max)`. B2 still reads `≤3min per 10k`. B3 `p50 ≤50ms`. B13 `≤2s`. None
carries 🐧, none has a target-side counterpart, and none is achievable on the machine the
product will actually run on.

Order-of-magnitude reality, from anchors already in `research/`:

| anchor | number | source |
|---|---|---|
| photofield-ai, 2014 6-core i7, e2e | 20 img/s ⇒ **~3.3 img/s per core** | priorart §3 row 1 |
| UForm-small ONNX, server x86, batch 128 | **~2 img/s per core** | priorart §3 row 2 |
| immich, i5-10500 / N100 | 0.4 / 1.11 img/s | measured-numbers |

Now apply the amendment's own B15 default (`workers ≤ cores/2`) to a plausible "not
powerful" 4-core box: **2 usable workers × ~3 img/s/core ≈ 6 img/s**, i.e. **10k images in
~28 minutes**, not 3. B1 and B2 are off by roughly **10×** for the primary target — and
because they carry no 🐧, a builder will tune against them, hit them on the M3, and ship
something whose real-world behaviour was never budgeted.

There is also a doctrinal consequence nobody has written down: **the user's constraint
makes B15 dominate B1 by decree.** "we cant slow down the server" is absolute; "blazing
fast" is an adjective. On the primary target these two now conflict directly (every core
we take is a core the co-tenant loses), and the table has no precedence rule, so a builder
resolving the conflict will resolve it the flattering way.

**Exact fix — (a) a precedence law in the header, (b) 🐧 counterpart rows:**

```
> PRECEDENCE LAW (from VISION-ADDENDA verbatim "we cant slow down the server"):
> on the primary target, when a speed budget (B1/B2) conflicts with co-tenant protection
> (B15), **B15 wins**. Indexing throughput is the variable; co-tenant impact is the
> constraint. `--full-speed` is the only way to invert this, and it requires an explicit
> user flag, never a default, never a benchmark headline.
```

```
| B1 🐧 | "blazing fast … processing and indexing" | sustained e2e index throughput per
  USABLE core (cgroup-aware, see A-C4), at defaults (polite) |
  PRIMARY (Linux x86 AVX2, shared): **≥3.0 img/s per usable core** on CORPUS-A,
  ≥1.2 img/s per usable core on CORPUS-B — e.g. ≥6 img/s / ≥2.4 img/s on a 4-core box at
  the ≤cores/2 default. Head-to-head gate stands: red if < rclip on the same box.
  PROXY (M3 Max, informational only, never a headline): ≥150 img/s CORPUS-A.
  ⌂-real: NOT MEASURED. | `imgtag bench index --corpus {A,B}` | provisional |

| B2 🐧 | "time to process 10,000 images on cpu" | wall time for a real 10k run at defaults |
  PRIMARY: ≤45 min (CORPUS-A) / ≤2 h (CORPUS-B) at defaults on a 4-usable-core box;
  ≤20 min with `--full-speed`. PROXY: ≤2 min. Projection gate (±15%) as in I-2. |
  `imgtag bench index --n 10000` | provisional |

| B3 🐧 | "especially during inference/search … instantly" | search e2e latency, warm daemon,
  ≥200 distinct uncached queries | PRIMARY, idle box: p50 ≤120 ms · p95 ≤250 ms.
  PRIMARY, **while indexing at defaults**: p95 ≤400 ms (this is the number users will
  actually live with and it exists nowhere today). PROXY: p50 ≤50 ms · p95 ≤120 ms. |
  `imgtag bench search` | provisional |

| B13 🐧 | "very very lightweight poc app" | cold start → first searchable response |
  PRIMARY: ≤6 s warm page cache · ≤15 s cold page cache (a shared box's page cache is
  evicted by co-tenants constantly — the cold number is the realistic one and must be the
  one quoted). PROXY: ≤2 s / ≤4 s. | `imgtag bench coldstart` | provisional |
```

The per-core formulation matters: it is the only form that transfers across an unknown
server, and it is the form the head-to-head gate can be checked against without knowing
the hardware in advance.

**Honest reframing that should be written into the project's story:** on the primary
target the claim is no longer "10,000 images in 3 minutes". It is **"10,000 images
overnight without anyone noticing, and instant search the whole time"** — still 5–15×
every self-hosted incumbent (immich 0.4–1.11 img/s), still the only project publishing the
number, and *true*. The current table promises the first sentence and will deliver the
second.

### A-C2 · B15's ≥85% probe is not a test. Four unspecified choices each decide the verdict, and one of them makes the budget passable by shipping a useless product.

B15 now reads: *"reference co-workload (CPU+IO probe) keeps ≥85% of its solo throughput
while we index at defaults + search concurrently."* Good instinct — it measures the right
*kind* of thing (impact on the other guy, not our own niceness). But as written:

1. **The probe is unnamed, and the probe choice picks the answer.** A CPU-spin probe is
   almost perfectly protected by `nice` — that is what nice does — so it passes at ~98%
   trivially and proves nothing. A memory-bandwidth-bound probe is **not** protected by
   nice at all (the scheduler allocates CPU time, not LLC or DRAM bandwidth), and our
   decode+embed loop is bandwidth-hungry; it may fail at 60%. A page-cache-sensitive probe
   fails for a third reason entirely (A-I1). Whoever writes `bench politeness` chooses the
   verdict. **A budget whose result is determined by an unspecified fixture is not a
   budget.**
2. **The solo baseline drifts.** "≥85% of its solo throughput" — measured when? On a
   *shared* server the solo baseline moves with the other tenants' load; a baseline taken
   at 09:00 and compared to a run at 14:00 measures the other tenants, not us.
3. **No window, no percentile.** 85% of the *mean* over a 30-minute run is satisfied by
   perfect behaviour plus one 4-minute stall. Users experience the stall.
4. **No floor on our own work.** Nothing stops a builder passing B15 by indexing at
   0.3 img/s. B15 and B1 must be asserted **in the same run** or they are a free pass.
   (Same structural defect as B7 in the first pass: a constraint with no opposing force.)
5. **85% is not sourced.** The verbatim is *"cant slow down the server"*. 85% licenses a
   15% permanent tax for hours. The default profile should be far more conservative, with
   85% as the floor of an opt-in faster mode.

**Exact fix — a fully specified, versioned probe and a two-sided assertion:**

```
| B15 🐧 | "cant slow down the server while we are doing both processing and infrence work"
  | co-tenant impact, measured with the FIXED reference probe `cotenant-probe-v1`
  (versioned, committed under bench/probes/, three concurrent components pinned to their
  own cores: (i) CPU — sha256 over a 1 MB resident buffer; (ii) MEMORY — pointer-chase
  over a 256 MB working set, the component nice cannot protect; (iii) IO+PAGE-CACHE —
  4 KB random reads over a 512 MB file pre-warmed into page cache, reporting hit latency).
  Baseline is measured **interleaved**, A/B/A/B in 30 s windows across the whole run, so
  co-tenant drift cancels; never a single before-run baseline. |
  DEFAULT profile: each probe component ≥**95%** of its interleaved solo throughput
  (mean) AND ≥90% in EVERY 30 s window AND probe p99 op-latency ≤1.5× solo p99.
  `--profile fast` (opt-in): ≥85% / ≥75% / ≤2.0×.
  `--full-speed`: unbounded, requires an explicit flag, prints the measured co-tenant cost.
  **AND, asserted in the SAME run: our own throughput ≥ B1🐧 floor** (a polite indexer
  that does no work fails B15, it does not pass it).
  **AND: 0 increase in system swap-in/swap-out over the run** (A-C5).
  Mechanism assertions (necessary, not sufficient — see A-C4): nice ≥10 on every pid in
  the tree · cgroup cpu.weight lowered when we own a cgroup · IO politeness verified by
  EFFECT (component iii), not by the presence of an ionice call. |
  `uv run imgtag bench politeness` | provisional |
```

The two structural additions are (a) the memory-bandwidth component — the one thing that
`nice` provably does not protect and that our workload provably stresses — and (b) the
same-run B1 floor, without which B15 is free.

### A-C3 · The project's entire speed thesis is int8, and int8 may be worth nothing — or be negative — on AVX2-without-VNNI. The ARM proxy cannot detect this.

The 157.9 img/s in `measured-numbers.md` is **int8 ONNX on an M3 Max**. ORACLE C2 makes
the precision sweep the primary lever. But int8 inference speed on x86 comes almost
entirely from **AVX-512 VNNI** (`vpdpbusd`); the header itself now says *"do NOT assume
AVX512/VNNI"*. On AVX2, MLAS emulates int8 GEMM via `vpmaddubsw`/`vpmaddwd` sequences
with requantization overhead, and the win over fp32 can be small, zero, or **negative**.

The evidence is already in the research and nobody connected it to the amendment:
priorart's clip.cpp counter-anomaly — *f32 272 ms < q5_1 322 < q8_0 334 < q4_0 539* —
was measured on an **Intel Mac**, i.e. an AVX2 machine with no VNNI. **That is the primary
target's ISA class.** UNKNOWNS C2 recorded the anomaly as a general warning; the amendment
turns it into the base case.

Consequences:
- DeepSparse's 1230 img/s (the headline int8 anchor) is on **64-core AVX512-VNNI** — it
  transfers to the target not at all.
- The M3 Max proxy cannot tell us which way this goes, because ARM int8 (SDOT/NEON) has
  entirely different economics from AVX2 int8. **The proxy is not merely optimistic here;
  it is uninformative about the sign of the result.**
- A single "chosen recipe" in the index manifest is now wrong.

**Exact fix — three changes:**
1. **Per-ISA recipes.** The manifest records `{isa: avx2|avx512-vnni|neon, precision,
   threads, batch}` and the engine selects by detected ISA at load. One global "we chose
   int8" is a bug on a heterogeneous fleet.
2. **The precision sweep re-runs on the target.** No precision decision is inherited from
   the proxy. Add to the ORACLE playbook: *"int8 slower than fp32 on the Linux target"* is
   the **expected** case, not an anomaly — ship fp32/fp16 there and say so.
3. **Add the ISA to every bench result line** (ORT `get_available_providers` +
   `cpuinfo` flags), so no number in this project is ever quoted without the instruction
   set it was measured on. This costs three lines and permanently prevents the category of
   error that produced B1's original 60.

### A-C4 · The 🐧 rows size themselves from the wrong numbers, and their two politeness mechanisms may be inert on the actual target.

B15 says `workers ≤ cores/2`; B8 budgets against 8 GB. Both imply reading `os.cpu_count()`
and total RAM. On a *shared* Linux server — which is the stated target, and which very
often means a container or a cgroup slice — both are wrong, and wrong in the dangerous
direction:

- **cgroup v2 `cpu.max`**: in a 2-CPU slice on a 32-core host, `os.cpu_count()` returns
  32 → `cores/2` spawns **16 workers** into a 2-CPU quota. Result: massive throttling,
  scheduler churn, and the co-tenant harm the budget exists to prevent. This is the single
  most common Linux-container bug in existence and B15 walks straight into it.
- **`cpuset.cpus.effective`** may pin us to a subset; **`MemAvailable`** (not MemTotal) is
  the only meaningful memory number on a box with other tenants.
- **`nice` is inert across cgroups.** Nice values are compared *within* a CPU cgroup. If
  the co-tenants live in a different cgroup (systemd services almost always do), our
  `nice 19` gives them **nothing**; only `cpu.weight`/`cpu.max` on our own slice does.
  B15's headline mechanism can be a complete no-op on the real server.
- **`ionice` is a no-op on most modern servers.** It requires CFQ/BFQ; NVMe and most
  current distros default to `none` or `mq-deadline`, where ionice is silently ignored.
  B15 currently asserts *"nice ≥10 + ionice"* — i.e. it asserts two calls were made, not
  that either had an effect.

**Exact fix:**
```
Resource discovery (a single committed helper, unit-tested against synthetic cgroup trees):
  usable_cores = min(len(sched_getaffinity(0)),
                     cgroup_v2 cpu.max quota/period if set,
                     cpuset.cpus.effective count)          # NEVER os.cpu_count()
  usable_mem   = min(cgroup memory.max − memory.current, /proc/meminfo MemAvailable)
                                                           # NEVER MemTotal
Politeness is asserted BY EFFECT (B15's probe), never by mechanism presence. Mechanisms
applied best-effort in this order, each logged as applied/ignored/unavailable:
  cgroup cpu.weight (if we own a cgroup) → nice → ionice (if scheduler is cfq/bfq) →
  explicit self-throttle (sleep-based duty cycle) as the ALWAYS-AVAILABLE fallback.
`bench politeness` prints which mechanisms were actually effective on this host.
```
The duty-cycle self-throttle is the important addition: it is the only mechanism
guaranteed to work regardless of scheduler, cgroup layout, or privileges, and it is what
makes B15 achievable on a box where we control nothing.

### A-C5 · 8 GB shared + our 1.5 GB assumes an empty box — which contradicts the user's own sentence. Swap and the OOM killer are the real co-tenant catastrophe, and neither is budgeted.

B8's new note reads *"≤1.5GB total under load (leaves ≥6.5GB to co-tenants)"*. The
verbatim constraint says the server *"also has other things running"*. If co-tenants
already hold 5 GB, our 1.5 GB does not leave 6.5 GB — it pushes the box into **swap**, and
a swapping shared server is not 15% slower for the co-tenant, it is 10× slower. Every CPU
politeness mechanism in B15 is irrelevant in that regime.

Second, worse: on Linux, a 1.5 GB spike on a loaded 8 GB box invites the **OOM killer**,
which by default kills the largest RSS — plausibly *us* (fine) or plausibly the co-tenant
(catastrophic, and precisely the "slow down the server" outcome, maximally).

Neither failure mode has a budget, and B8 as written is a static cap that cannot see them.

**Exact fix:**
```
| B8 🐧 | "not taking too many resources" + the 8GB-shared-server law |
  (a) peak RSS across the whole process tree, sampled ≥10 Hz;
  (b) daemon idle RSS (with the text tower still resident — see I-5);
  (c) ADAPTIVE cap: at startup the engine sizes itself to
      min(1.0 GB, 25% of usable_mem at that moment) and logs the chosen cap;
  (d) **0 increase in system swap-in/swap-out (/proc/vmstat pswpin/pswpout) over the run**
      — this is the hard one and it is the one that actually protects the co-tenant;
  (e) `oom_score_adj = +500` on every worker: if the box runs out, **we die first, never
      the co-tenant** (this is a one-line change and it is the most co-tenant-protective
      thing in the whole project);
  (f) memory-gated model load with requeue instead of OOM (Photonix pattern, priorart
      steal list) — if usable_mem is below the model's requirement, refuse with a clear
      message and retry later; never load-and-pray;
  (g) model+index+thumbs on disk for 10k ≤500 MB.
  Thresholds: ≤1.0 GB · ≤350 MB · adaptive · 0 swap delta · asserted +500 · asserted · ≤500 MB |
  `uv run imgtag bench resources` | provisional |
```

---

## IMPORTANT (amendment)

### A-I1 · Page-cache eviction: the invisible way we slow the co-tenant, and B15's probe as written would miss it.

Indexing 10k images streams several GB through the page cache and **evicts the
co-tenants' hot data** — their database working set, their web app's files. The victim's
CPU time looks unchanged; their latency triples. This is one of the most common causes of
"the server got slow when the backup ran" and it is invisible to any pure CPU probe.
**Fix:** (a) probe component (iii) in A-C2 exists precisely to catch it; (b) structural
prevention — `posix_fadvise(POSIX_FADV_DONTNEED)` on each image file after reading it, and
`O_DIRECT`-style streaming for the shard writer where available, so our bulk scan does not
pollute a cache we do not own. Assert in `bench politeness` that the probe's page-cache hit
latency stays within 1.5× of solo.

### A-I2 · B8🐧 is now a model-selection gate, and it collides with B17. A new gameable pair the amendment created.

priorart §3 row 10: SigLIP2 at f32 is **~3 GB peak RSS**. Under the new B8 (≤1.0 GB) the
highest-quality candidate in the roster is **disqualified on the primary target** — not
"watch it", *disqualified*. Meanwhile B17 demands the quality points that model provides.
Two budgets now pull in opposite directions with no stated precedence, exactly like B1/B15.

**Fix:** (a) add a **hard pre-filter to the candidate bench**: any candidate whose measured
peak RSS on the primary target exceeds B8 is marked `INELIGIBLE-DEFAULT` before quality is
even scored (it may still be benched as a ceiling, like MobileCLIP2 for licensing);
(b) state the precedence in the header: *"on the primary target B8 is a hard constraint;
B17 is maximized subject to it. A quality win that does not fit in memory is not a win."*
(c) this makes the quantized/small-model path **mandatory**, not optional — which should
be said out loud, because it also reactivates A-C3 (int8 may not be fast on AVX2, but it
is still needed for *size*; speed and size are now separate justifications and must be
argued separately).

### A-I3 · Proxy labelling: right in the header, absent where it is read; and nothing forbids locking a 🐧 budget on proxy numbers.

Two mechanical gaps:
1. **Tags are not where the eyes are.** A builder reads a *row*, not the preamble. B1's
   cell says `≥60 img/s (M3 Max…)` — the machine is mentioned, but nothing marks it as
   non-authoritative, and B2/B3/B13 do not mention a machine at all. **Fix:** every
   proxy-derived threshold is prefixed inline, e.g. `PROXY ≥150 img/s`, and every 🐧 row
   carries an explicit `PRIMARY: … / NOT MEASURED` cell. A blank that says NOT MEASURED is
   the most valuable cell in the table.
2. **No lock law for the amendment.** The header says budgets go provisional→locked. It
   does not say a 🐧 budget may not be locked on proxy numbers — so the first person under
   deadline will. **Fix, one line, load-bearing:**
   ```
   > LOCK LAW: no 🐧 budget may move to `locked` on proxy numbers. Ever. A 🐧 row locks
   > only from a bench run executed on the real primary target, recorded in
   > bench/results/<date>-<sha>.json with its /proc/cpuinfo flags, cgroup limits and
   > MemAvailable. Until then a 🐧 row is `provisional (proxy-only)`.
   ```
3. **Inverted marker semantics.** Marking only B8/B15 with 🐧 implies the other 15 rows
   are *not* primary-target concerns — the opposite of the truth after this amendment.
   **Fix:** every row is primary-target by default; mark the handful that are genuinely
   proxy-or-UI-only (B14, B4) instead.

### A-I4 · The stated throttle profile is not implementable on the proxy machine as described.

The header says the throttled profile is *"4 threads + memory-capped"*. On macOS there is
no cgroup; `ulimit -v` caps address space (which breaks mmap'd shards and ORT arenas
rather than emulating a small machine), and there is no supported RSS cap. So the
"memory-capped" half of the proxy profile likely cannot be built, and if someone builds
something and calls it that, it will not model an 8 GB box.
**Fix:** state honestly that the proxy models **core count only**; drop the memory-cap
claim; get the memory behaviour from the real target (or from a Linux box with
`docker run --cpus=2 --memory=1g`, which *does* model it correctly and is available for
free on any x86 machine). Note the corollary: **a Linux x86 container is a categorically
better proxy than a throttled M3 Max for everything in this amendment** — same ISA class,
real cgroups, real MemAvailable, real page cache. If the real server is not yet reachable,
that is where the proxy should move.

### A-I5 · "Both processing and inference" is in the verbatim; only processing is budgeted for politeness.

The constraint says *"while we are doing both processing and infrence work"*. B15's
threshold text does say "index at defaults + search concurrently", which is good — but
there is no budget on **search-side** resource use at all: no cap on text-encoder threads,
no cap on concurrent query handling, no queue bound. A burst of agent-driven searches
(and the agent skill, B20, makes bursts likely) can saturate the box with no politeness
mechanism whatsoever, because all of B15's mechanisms are described for the indexer.
**Fix:** extend B15 to the daemon: search-path thread count ≤ usable_cores/2, a bounded
query queue with backpressure (reject/queue rather than fan out), and the probe run must
include a **search-only** phase (no indexing) that still meets the ≥95% default.

### A-I6 · Shared server ⇒ multi-user ⇒ a new, literal reading of "no data leaks" that has no row.

On a shared box, other humans have shell accounts. `~/.imgtag/` holding a full index of
someone's private photo library — paths, filenames, thumbnails, embeddings from which
thumbnails can be approximately reconstructed — world-readable is a real data leak, and it
is the reading of the vision phrase that the amendment newly makes relevant.
**Fix — extend B22 (from the first pass):** `~/.imgtag` and every file under it is created
`0700`/`0600`; the daemon binds **loopback only** (never `0.0.0.0` — on a shared server
that exposes one user's photo search to every other tenant *and* to the network); if a
non-loopback bind is requested it requires an explicit flag plus a printed warning; the
socket/port is per-user; `bench egress` additionally asserts the listening socket is
127.0.0.1 and the mode bits are correct.

### A-I7 · No ISA/portability floor, and the failure mode is a SIGILL on a shared production box.

"Assume AVX2" is now a stated assumption with nothing enforcing or checking it. A
pre-2013 or heavily-virtualized host without AVX2 will not degrade — an AVX2 kernel on a
non-AVX2 CPU **crashes with SIGILL**, which on a shared server is the worst available
outcome. Nextcloud Recognize's degraded-tier fallback on missing AVX is already in the
priorart steal list and unadopted.
**Fix:** startup capability check (`/proc/cpuinfo` flags) → if AVX2 is absent, either
select a compatible ORT build or **refuse with a clear message naming the missing
instruction set**; never SIGILL. Also pin and state the portability floor: manylinux
glibc ≥2.17, Python ≥3.10, no Docker required, no root required, no privileged port. Add
these as assertions to B23 (footprint).

### A-I8 · Deployment reality for a headless shared box has no coverage anywhere.

The primary target is headless, shared, and not ours: the daemon must survive without root
(systemd **user** unit or a supervised process), restart on crash, not require a privileged
port, tolerate the box rebooting under it, and never leave a wedged process holding
memory. None of this is in BUDGETS, IA, or ORACLE. It is also where "no leaks" meets
operations: a crashed-and-restarted daemon that leaks 300 MB per restart eventually is the
co-tenant problem.
**Fix:** add to B20/B21 a deployment clause — install and run as an unprivileged user with
a provided systemd user unit; survive `kill -9` mid-index with the manifest intact
(ADR-6's atomic rename makes this true by design — assert it); on restart, resume without
re-embedding already-indexed images (ties to B25's compute-leak gate); RSS after
N=5 restart cycles within 5% of the first.

---

## MINOR (amendment)

- **A-M1 · Every result line must carry the machine.** Add to the bench output header:
  ISA flags, usable_cores, cgroup quota, MemAvailable, kernel, glibc, ORT version+EP. Three
  lines of code; permanently prevents the class of error that produced the original B1.
- **A-M2 · B4/B14 now cross a network.** With the app served from the server, "keystroke→
  painted" includes RTT to it. Specify which: budget on `localhost` (SSH tunnel) as the
  measured case, and record a LAN number separately. Unspecified, it will drift.
- **A-M3 · The changelog calls B8 "tightened".** Under the new law it is not the same
  quantity tightened — it is a **different machine's** budget. The lock/tighten law must
  be machine-scoped: a budget's identity is `(metric, target)`, and a target change resets
  it to provisional rather than counting as a tighten. Otherwise the "never loosen"
  guarantee is unfalsifiable across target changes.
- **A-M4 · ORACLE and UNKNOWNS have not been re-derived against the amendment.** ADR-1
  (ORT CPU EP) survives — it is arguably *strengthened*, since MLAS/AVX2 is exactly where
  ORT wins and the OpenVINO EP bench slot on x86 is now primary-target-relevant, not a
  curiosity. But ADR-2 (brute-force scan) should be re-checked against 8 GB (10k×512 f16
  = 10 MB, fine; 100k = 100 MB, still fine — it holds, and should be *said* to hold),
  UNKNOWNS I6's edge-floor protocol is now superseded by the 🐧 protocol, and I8's
  dependency budget should name the manylinux/glibc floor. The three surfaces should be
  updated in the same pass per the project's own coherence invariant (ORACLE §6).

---

## Revised summary of gameable pairs (after the amendment)

| pair | status |
|---|---|
| B15 vs B1 (polite vs fast) | **NEW, created by the amendment** — no precedence rule; fixed by A-C1's precedence law + A-C2's same-run B1 floor |
| B8🐧 vs B17 (fits in 8 GB vs highest quality) | **NEW** — no precedence rule; fixed by A-I2 (hard pre-filter + explicit precedence) |
| proxy vs primary (all speed rows) | **NEW** — proxy numbers are the only ones present, so they become the de-facto target; fixed by A-C1 rows + A-I3 lock law |
| B7 vs B5/B6, B8-idle vs B3/B13, B1 vs B24, B1 vs B2 | unchanged from the first pass — all still open |

## Bottom line on the amendment

The re-anchoring is the right call and it was made fast, which is worth more than it was
made completely. But two rows is not the change: the constraint invalidates the machine
assumption under the whole table, creates two new unguarded budget conflicts (B15↔B1,
B8↔B17), and — the finding I would put in front of the user first — **quietly threatens
the project's core speed thesis**, because the int8 win that produced 157.9 img/s is an
ARM/VNNI phenomenon and the primary target has neither. The cheapest, highest-value next
action is not more budget-writing: it is **one bench run inside `docker run --cpus=2
--memory=1g` on any x86 machine**, which converts A-C1, A-C3, A-C4 and A-C5 from argument
into measurement in about an hour, and would tell the project whether its headline number
survives contact with the machine it is actually for.

