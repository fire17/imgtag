# BUDGETS.md — vision adjectives → enforced numbers

> Law (grand-start Phase 2): every quality adjective in VISION.md becomes a numeric
> budget measured on the real target. A regression is a build failure, not a ticket.
> `provisional` until the candidate bench locks them; budgets only tighten after locking.
>
> **PRIMARY DEPLOY TARGET (VISION-ADDENDA.md, verbatim): shared Linux x86 server, no GPU,
> 8GB RAM, "not powerful", co-tenant workloads must never be slowed.** AVX2 baseline, no
> AVX512/VNNI assumed. Bench machine M3 Max = PROXY (all its numbers labeled so).
>
> **Marker semantics (rev round 1):** *every row is a primary-target row by default.*
> 🐧 marks the rows whose threshold is target-specific; 🖥 marks proxy/UI-only rows
> (`B*-dev`, B4, B14). `PROXY` prefixes any threshold derived from the M3 Max. **Every speed
> budget is split in two:** a `B*-dev` PROXY row (M3 Max, gates CI today) and a `B*🐧` target
> row carrying an *interim floor* that locks only at the first real-server bench. Reporting a
> proxy number without its label — or comparing one against a 🐧 floor — is a stop-and-escalate
> offense (ORACLE §7 clause g).
> ⌂-ub = **upper bound of the edge estimate** (M3 Max, 4 threads) — *not* a prediction of
> any real machine. ⌂-real = the real old-x86 datapoint, **NOT MEASURED** until someone runs
> `docker run --cpus=4 --memory=8g` on any x86 box. **No ⌂ number appears in any README,
> landing page, or public claim until ⌂-real is populated.**
> The 4-thread throttle models **core count only** — it does not model memory bandwidth,
> ISA, storage, or an 8GB ceiling (macOS has no cgroup; `ulimit -v` breaks mmap rather than
> emulating a small box). A Linux x86 container (`docker run --cpus=2 --memory=1g`) is a
> categorically better proxy than a throttled M3 Max and is where the proxy should move
> until the real server is reachable.
>
> **LOCK LAW:** no 🐧 budget may move to `locked` on proxy numbers. Ever. A 🐧 row locks only
> from a bench run executed on the real primary target, recorded in
> `bench/results/<date>-<git-sha>.json` with its `/proc/cpuinfo` flags, cgroup limits and
> MemAvailable. Until then a 🐧 row is `provisional (proxy-only)`. Locking ANY row requires
> that same committed results file (every budget's measured value + hardware string + model
> id/sha + corpus ids); the Status cell then reads `locked@<sha>`. A budget's identity is
> **(metric, target)** — changing the target resets the row to `provisional`, it does not
> count as a tighten, and "never loosen" is asserted only within one identity.
>
> **PRECEDENCE (when budgets pull against each other):** on the primary target **B8 is a hard
> constraint and B17 is maximized subject to it** — a quality win that does not fit in memory
> is not a win. Any candidate whose measured peak tree-RSS on the primary target exceeds B8 is
> marked `INELIGIBLE-DEFAULT` in the candidate bench *before* quality is scored (it may still
> be benched as a ceiling, like MobileCLIP2 for licensing). This makes the small/quantized
> model path mandatory rather than optional — and note that quantization now has **two
> independent justifications, size and speed, which must be argued separately** (int8 may buy
> size on AVX2 while buying no speed at all).
>
> **Bench protocol (rev round 1):** every row = median of ≥3 runs; every row records
> `os.getloadavg()`; runs refuse or mark UNRELIABLE when 1-min load > cores×0.6 (on the shared
> target `bench all` records loadavg and marks rows ADVISORY instead of refusing — the target's
> honest number is a polite number); every number carries its corpus tag + resource mode
> (POLITE/FULL). Every result header line carries the machine: ISA flags, usable_cores (=
> effective cores, ORACLE ADR-11), cgroup quota, MemAvailable, kernel, glibc, ORT version+EP.
> Darwin loops gate on all of it.

## Corpora (no number exists without one)

| Tag | Name | Contents | Status |
|---|---|---|---|
| CORPUS-A | coco5k | 5,000 COCO val2017 (640×480 median) + exhaustive 80-class truth | ✅ on disk |
| CORPUS-B | photo10k | 10,000 Unsplash @w=3200 (≈5MP) — realistic photo sizes | fetch queued (≈18GB) |
| CORPUS-B12 | fullres300 | 300 native ≥12MP originals — the decode-bound case | fetch queued |
| CORPUS-C | mixed10k | coco5k + 5k of photo10k | derived |
| CORPUS-D | poison | ~120 hostile files: truncated/corrupt JPEG, 0-byte, CMYK JPEG, 16-bit + palette-with-alpha PNG, all 8 EXIF orientations, HEIC, animated GIF, `.jpg` that is really a PNG, unicode/emoji filenames, symlink + symlink loop, 300MP decompression bomb, read-permission-denied file, huge-dims | build queued |
| CAL-SET | cocotrain2k | ~2,000 COCO train2017 imgs (per-image fetch) — HELD-OUT calibration split, never benched | fetch queued |

## Budgets

| # | Vision phrase (verbatim anchor) | Metric | Threshold | Test command | Status |
|---|---|---|---|---|---|
| B1-dev 🖥 | "blazing fast … processing and indexing" | sustained e2e index throughput (files→searchable), CPU-only, **POLITE mode** (headline), M3 Max PROXY. *Sustained* = wall clock from process start to the manifest commit that makes the LAST image searchable, ÷ N; first 5% discarded as warm-up; shard flush, manifest commit and thumbnails all inside the timed region | PROXY: CORPUS-A ≥150 img/s · CORPUS-B ≥60 img/s · stretch (FULL, labeled) ≥180 = beat rclip's measured CoreML rate on pure CPU · ⌂-ub ≥10 img/s · **HEAD-TO-HEAD GATE: red if slower than rclip on same corpus+machine+run** | `uv run imgtag bench index --corpus A,B --headtohead rclip` | provisional |
| B1 🐧 | same phrase, primary target | same metric, POLITE mode, on the real shared Linux x86 server | **interim floor ≥8 img/s** (photofield-ai did 20 on a 2014 6-core at FULL speed; we run memory-capped workers at nice 10) — **locked only at the first real-server bench; until then NO proxy number may be published as a product claim** | same command, on the target host | provisional — NOT MEASURED |
| B2-dev 🖥 | "time to process 10,000 images on cpu (tests scales to 100…)" | real 10,000-image wall time **+ projection fidelity** (what licenses the n=100 CI shortcut), PROXY | CORPUS-B t_10k ≤ 10000/B1-dev floor by construction (≤3min at ≥60 img/s) · CORPUS-A t_10k ≤2min · **projection gate: \|t_10k projected from n=100 − t_10k actual\| / t_10k actual ≤0.15**. Full 10k run required once per lock and once per darwin round; n=100 is CI-only | `uv run imgtag bench index --n 100` (CI) · `--n 10000` (lock/darwin) | provisional |
| B2 🐧 | same phrase, primary target | 10k wall time on the real server | **interim floor ≤25min POLITE (≤12min FULL)** — locked only at the first real-server bench; no proxy number published as the product claim | same command, on the target host | provisional — NOT MEASURED |
| B3-dev 🖥 | "especially during inference/search … instantly" | search e2e warm-daemon latency over ≥200 DISTINCT never-before-seen queries (cache never pre-warmed with the test set), CORPUS-C, PROXY | p50 ≤50ms · p95 ≤120ms · cache-HIT latency reported separately, never as the budget · daemon asserted resident (cold path is B13) · scan array asserted f32 · 100k synthetic-scale scan p95 ≤15ms (graceful-degradation claim, §1.1 measured 7.4ms) | `uv run imgtag bench search --queries 200 --no-cache-prewarm` | provisional |
| B3 🐧 | same phrase, primary target | same metric on the real server | **interim floor p50 ≤80ms · p95 ≤200ms** — locked only at the first real-server bench | same command, on the target host | provisional — NOT MEASURED |
| B4 🖥 | "google photos in app search … instantly found" | keystroke→painted via performance.mark pairs, Chromium/Playwright (DEV dep), 1440×900, CORPUS-C, warm daemon, ≥50 trials × 10 distinct queries | p50 ≤80ms · p95 ≤150ms · measured over **localhost** (SSH tunnel when the app is served from the server); a LAN number may be recorded separately, never as the budget | `uv run imgtag bench app-search` | provisional |
| B5 | "sematically flexible … 'vehicle' … all motocycles and other vehicles" | hypernym retrieval, CORPUS-A supercategory queries {vehicle, animal, food, furniture, appliance, sports} | (a) precision@100 mean ≥0.85 · (b) per-child recall@R: mean ≥0.55, MIN ≥0.35, NO child at 0.00 · (c) every child present ≥1× in top-100 · per-child table always emitted | `uv run imgtag bench quality --hypernym` | provisional |
| B6 | "when i search for 'car' - all of the images with one or more cars" | per-category precision@k, **k = min(10, N_pos)** (toaster has 8 positives in val2017, hair drier 9 — p@10 is ill-defined for them), all 80 COCO categories, CORPUS-A | mean ≥0.90 AND min ≥0.70 AND **zero categories at 0.00** · full 80-row table emitted with the verdict | `uv run imgtag bench quality` | provisional |
| B7 | "lack of false positives" / "minimization of any false positives" | calibrated no-match threshold τ FROZEN in manifest, fitted on CAL-SET (held out); both sides asserted in ONE run | (a) absent-concept leakage ≤2% over ≥20 auto-derived absent queries + 5 absurdities · (b) at the SAME τ: mean recall@10 over 80 present categories ≥0.70 · (c) τ recorded in bench output + manifest; τ change without re-run of (a)+(b) = red. Passing (a) by sacrificing (b) is RED | `uv run imgtag bench quality --negatives` | provisional |
| B8 🐧 | "not taking too many resources" + 8GB-shared-server law | **peak process-TREE RSS** (sum over the process group, sampled ≥2Hz, max over the run; nothing subtracted for shared pages) — indexing peak / daemon idle after 10min / total under load · model+index+thumb-cache disk @10k · **per-worker RSS is a first-class reported number** (a total without its geometry is not a result) | ≤1.0GB · ≤350MB · ≤1.5GB · ≤500MB (thumbs ≤200MB of that, LRU-capped) · **anti-gaming: immediately after the idle-RSS sample a search must still meet B3 p95 in the SAME run** — the text tower must be resident, not unloaded (immich's `model_ttl` sin) · measured at the defaults `imgtag doctor` selects, with worker count + per-worker RSS printed alongside · this budget measures OUR total footprint — ≤1.5GB regardless of what co-tenants use — and makes no claim about free memory on a box we do not own | `uv run imgtag bench resources --tree` | provisional (proxy-only) |
| B9 | "super state of the art small but powerfull" | **sum of ALL model artifacts** required for index + search (image tower + text tower + tokenizer + tag-vocab table + calibration), on disk, in the shipped precision | ≤150MB total · stretch ≤90MB (beats clip.cpp's 85.6MB while beating ViT-B/32 quality per B17) · additionally: search-only resident model bytes ≤50MB (tower split) | `uv run imgtag bench artifacts` | provisional |
| B10 | "see live progress, how many images a sec its indexing, projected etas" | progress-stream freshness, rate accuracy, ETA error — CORPUS-B (n≥10,000; the n=100 bench asserts freshness only) | (a) freshness: max gap between consecutive progress events ≤1.0s **including idle/stall periods** (heartbeat required) and max(event_ts − manifest_commit_ts) ≤1.0s · (b) rate accuracy: reported rolling-10s img/s vs actual from manifest counts, MAE ≤10% · (c) ETA error at the 25/50/75% marks: \|eta_predicted_finish − actual_finish\| / (actual_finish − t_mark) ≤0.20; at the 10% mark ≤0.35 · (d) progress-emitter CPU ≤1% of run wall time (proves event-driven) · progress `done` = durable manifest count; in-flight reported as a separate field | `uv run imgtag bench progress` | provisional |
| B11 | "instantly search … while the processing is still ongoing" | visibility + behavioural non-blocking during a full CORPUS-B index (POLITE) with a searcher issuing 1 query/100ms from a distinct pool | (a) every 500 images, the newest indexed image is retrievable by a known-true query ≤2.0s after its manifest commit · (b) reader not blocked: search p95 ≤200ms, p99 ≤2× B3 p95, 0 queries >1s, 0 stale-manifest exceptions, 0 torn reads over ≥5k concurrent searches · (c) writer not blocked: index throughput during the query storm ≥95% of the query-free run · (d) coverage honesty: reported "indexed so far" equals the manifest count exactly at every sample | `uv run imgtag bench concurrent` | provisional |
| B12 🐧 | "no data or performance or compute or leaks of any kind" | **≥30-min continuous soak** (corpus looped): full CORPUS-B index + 10k searches + 3 reindex cycles; all 6 enumerated leak classes | memory: OLS slope of tree-RSS post-warmup ≤5% total growth AND slope 95% CI includes ≤0.5MB/min · fd drift 0 · thread drift 0 · storage: 0 files under `~/.imgtag/tmp`, 0 orphan bytes after a dataset delete (byte-diff) · disk-bloat: index bytes/image within 5% of dim×dtype · CPU: idle-daemon CPU ≤0.5% over 60s (proves event-driven, no spin-poll) · compute: **re-index of an UNCHANGED dataset re-embeds 0 images (mtime+hash gate), completes ≤5s per 10k, manifest byte-identical except timestamp** | `uv run imgtag bench soak --minutes 30` | provisional |
| B13-dev 🖥 | "very very lightweight poc app" | server cold start → first searchable response (incl. model load, no daemon) @10k, f32 open-cost included, PROXY | ≤2s **warm page cache** · the purged-page-cache number (`sudo purge`/`vmtouch -e`, documented in the bench) is also printed; **the cold number is the one quoted in any public claim** · ORT-optimized graph cached (`optimized_model_filepath`) so session creation is not re-paid | `uv run imgtag bench coldstart` | provisional |
| B13 🐧 | same phrase, primary target | cold start on the real server — **a contended shared disk, not an idle NVMe: honest cold numbers there will be worse and that is the number that counts** | **interim floor ≤4s cold disk, ≤2s warm** — locked only at the first real-server bench · cold start is rare by design (long-lived daemon, ADR-13) | same command, on the target host | provisional — NOT MEASURED |
| B14 🖥 | UI quality bar (impeccable) | scripted 10k-thumb virtualized-grid scroll 1000px/s 5s, Chromium 1440×900, CDP tracing (DEV dep) | frame p50 ≤16.7ms · p95 ≤20ms · 0 long tasks >50ms · DOM nodes <5,000 · a 4×-CPU-throttled run reported separately (never as the budget) | `uv run imgtag bench ui` | provisional |
| B15 🐧 | "cant slow down the server while we are doing both processing and infrence work" | co-tenant protection asserted from the SAME run as B1: reference co-workload keeps ≥85% solo throughput while indexing at POLITE defaults + concurrent searches; per-process nice sampled at 1Hz | POLITE policy: workers=clamp(ncpu−2,2,8) (memory-derived cap applies, ADR-10c), ORT intra=2 inter=1, total threads ≤ ncpu−1, nice ≥10 asserted per pid in the whole tree · **search side is polite too**: search-path threads ≤ usable_cores/2, bounded query queue with backpressure, and the probe includes a **search-only phase** (no indexing) that also holds ≥85% · **cache hygiene**: `posix_fadvise(DONTNEED)` after each image read, shards mmap'd read-only `MADV_RANDOM` — the co-tenant's page-cache hit latency stays within 1.5× of solo · **post-run recovery**: the co-workload probe runs 60s AFTER our job ends and must return to ≥95% of its solo baseline (a during-run pass alone does not satisfy B15) · **we die first**: workers set `oom_score_adj` high and an `RLIMIT_AS` from the memory budget · escape hatch proven: a `--full-speed` run asserts nice==0 and workers==ncpu | same run as `bench index` (+ `bench politeness` probe) | provisional (proxy-only) |
| B16 | anti-silent-quality-loss, preprocessing axis | fast decode path vs model-reference pipeline (per the model's OWN preprocessor_config) on quick500 **+ the EXIF/format subset of CORPUS-D** (a mean is blind to narrow, conditional failures — which is exactly what draft()/EXIF/CMYK bugs are) | mean cos ≥0.995 AND p1 cos ≥0.99 AND **min cos ≥0.98** AND top-1 NN agreement ≥0.90 AND \|Δ precision@10\| ≤0.01 abs AND Δ hypernym min-child recall ≥ −0.02 (bootstrap 95% CI over 1000 resamples reported) · a fast path failing this is disabled, not shipped | `uv run imgtag bench parity` | provisional |
| B17 | "state of the art" quality, auditable | COCO-caption text→image R@1/R@5/R@10 on the **Karpathy 5k test split** — identity with val2017 VERIFIED by image-id intersection and recorded in the bench output (if it does not hold, val2014 + karpathy json are fetched) | default model R@10 ≥ control **+12 pts** (control = OpenCLIP ViT-B/32-openai through OUR pipeline, same corpus, same run — +5 was below a free checkpoint swap: laion2b is +7.7 over that control for +0.03ms) AND absolute R@10 within 2 pts of the model card's published figure (proves our pipeline is not degrading the model) AND both numbers + hardware published in the results table · **maximized subject to B8 on the primary target** | `uv run imgtag bench quality --retrieval` | provisional |
| B18 | "get exactly both from which of the datasets the resulting images are from and obviously the image path and or id" | provenance completeness + result determinism + gallery correctness, 200 queries × top-50, CORPUS-C | (a) 100% hits carry non-null {dataset_slug, path, image_id} · (b) 100% paths exist (or are tombstoned `exists:false`, never a 404'd result set) · (c) 100% ids == xxhash64(bytes) · (d) 0 cross-dataset misattribution · (e) **determinism**: identical query on an identical manifest returns a byte-identical result list across 10 runs and across thread-count settings; ties broken by image id · (f) **gallery correctness**: fleet view lists exactly the datasets present under `~/.imgtag/datasets` and every displayed count equals its manifest count · **zero tolerance — any failure = red** | `uv run imgtag bench provenance` | provisional |
| B20 | "a globally available skill so agents will be able to … tag … info … manage … search" | machine-API conformance, all 4 verbs headless `--json` | valid JSON stdout (ANSI-free; human text → stderr) · documented exit codes for 5 error classes · 0 interactive prompts · search ≤ B3 p95+50ms · info ≤200ms · index returns job id ≤500ms non-blocking · delete leaves 0 orphan bytes (byte-diff) · **deployment**: installs and runs as an unprivileged user from a provided systemd **user** unit, no root, no privileged port | `uv run imgtag bench skill-contract` | provisional |
| B21 | "make sure it all works" / "increased reliablility correctness" | robustness over CORPUS-D (~120 hostile files) + crash/restart survival | 0 crashes · 0 hangs (every file resolved or skipped within a 5s per-file timeout) · every failure recorded as {path, reason} in the job status file and visible in `imgtag status` · ≥99.5% of the VALID files indexed · process exits 0 with a nonzero `failed` count, never a traceback · the decompression bomb is refused by a pixel-count cap, not by OOM · **survives `kill -9` mid-index with the manifest intact (ADR-6 atomic rename — asserted, not assumed); on restart resumes without re-embedding indexed images; tree-RSS after 5 restart cycles within 5% of the first** | `uv run imgtag bench robustness` | provisional |
| B22 🐧 | "no **data** … leaks of any kind" (egress + multi-user reading) | network egress + on-disk exposure across the whole process tree | 0 connections to any non-loopback address during index + search (`net_connections` sampled 1Hz for the full bench run) · the first-run model download is the ONLY permitted egress: announced, logged with exact URLs, and must not recur once cached (second run = 0 external connections) · `~/.imgtag` and every file under it created `0700`/`0600` · the daemon binds a per-user UNIX socket (or loopback only); a non-loopback bind requires an explicit flag + printed warning; the bench asserts the listening socket and the mode bits | `uv run imgtag bench egress` | provisional |
| B23 | "very very lightweight" / "small but powerfull" | installed footprint, dependency discipline, portability floor | `uv sync --no-dev` tree ≤150MB excluding models · runtime import set ⊆ ADR-7 {onnxruntime, numpy, Pillow, certifi/httpx, xxhash, micro-server} · `import torch` and `import transformers` both FAIL in the runtime env · wheel builds and installs with no Docker, no Postgres, no GPU, no root · **portability floor: manylinux glibc ≥2.17, Python ≥3.10; startup checks `/proc/cpuinfo` flags and, if AVX2 is absent, selects a compatible ORT build or refuses with a message naming the missing instruction set — never SIGILL** | `uv run imgtag bench footprint` | provisional |
| B24 | anti-silent-quality-loss, precision axis — TWO-TIER (re-ruled 2026-07-22 after bench falsified the single-tier gate both directions) | shipped precision vs SAME model fp32, CORPUS-A; parity pool FIXED at n=200 (nn-agreement is pool-size-dependent — n recorded on every row, cross-harness comparisons only at equal n) | **TIER-1 HARD FLOOR** (fail ⇒ banned everywhere, incl. opt-in): mean cos ≥0.95 AND nn@200 ≥0.60 · **TIER-2 QUALITY GATE** (the decider, full CORPUS-A suite): \|Δ mean p@k\| ≤0.01 · ΔR@10 ≥ −1.0pt · Δ hypernym recall ≥ −0.02 · **DEFAULT precision additionally requires nn@200 ≥0.90** · passes 1+2 but nn <0.90 ⇒ opt-in speed lane, measured deltas printed at enable time · re-gated PER-ARCH on the target | `uv run imgtag bench parity --precision` | provisional |

## Leak classes enumerated at design time (E5)
storage (orphan shards/thumbnails/tmp) · memory (decode buffers, ORT arena, query cache) ·
fd (files/sockets) · CPU (spin-polling forbidden; event-driven only) · compute (re-embedding
unchanged files — mtime+hash gated) · disk-bloat (compaction). Each: structural prevention +
monitor in `bench soak`/`bench resources`. The **data-egress** reading of "no data leaks" is
B22, not a leak class.

> 2026-07-22: v1 (14 budgets, provisional) → post-research re-derivation (17) → rev round 1
> criticals (B5 replaced — recall@100 ceiling was 0.086; corpus registry; B1 polite +
> head-to-head; B7 calibration contract; B18/B20/B21/B24 added; Playwright harnesses; noise
> protocol) → **rev round 1 IMPORTANT/MINOR (editor pass): lock law + budget identity +
> precedence + proxy semantics in the header; B22 egress/exposure and B23 footprint/portability
> added; B2 projection gate, B3 distinct-query + 100k scan, B6 k=min(10,N_pos), B8 anti-idle-
> gaming + per-worker RSS, B9 sum-of-artifacts, B10 ETA direction, B11 behavioural clauses,
> B12 30-min soak + compute-leak numbers, B13 cold/warm split, B15 search-side + cache hygiene
> + post-run recovery, B16 min-cos, B17 +12pts on the Karpathy split, B18 determinism + gallery,
> B21 restart survival tightened.** → **rev-oracle ROUND 2 (same pass): B1/B2/B3/B13 each split
> into a `-dev` PROXY row and a 🐧 target row with interim floors (≥8 img/s · ≤25min POLITE /
> ≤12min FULL · p50 ≤80ms p95 ≤200ms · ≤4s cold disk), none lockable off the proxy; B15 gains
> the ≥95%-of-solo post-run page-cache recovery clause; B8 restated as our footprint ≤1.5GB
> regardless of co-tenants.** Full adjudication: `.deify/reviews/ADJUDICATION.md`;
> application log: `.deify/reviews/EDITOR-LOG.md`. Numbering keeps review references stable
> (B19 intentionally unassigned).
