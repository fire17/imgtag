# EDITOR-LOG.md — rev round 1, IMPORTANT/MINOR application pass (2026-07-22)

Editor agent `editor-reviews`. Law order followed: ADJUDICATION.md → current
BUDGETS/ORACLE/UNKNOWNS/IA (post-critical state) → the three reviews.
Legend: **APPLIED** (fix landed) · **SKIP-COVERED** (critical-fix pass already covers it) ·
**CONFLICT** (needs a conductor ruling — nothing improvised).

## rev-oracle.md

| id | verdict | note |
|---|---|---|
| I-1 | APPLIED | risk row response → "ship fp32 (fp16 ONNX will NOT load on CPU EP)"; new dead end "fp16 ONNX on the ORT CPU EP" |
| I-2 | APPLIED | ADR-6: flock holder = SOLE manifest writer, workers report row counts; playbook "Indexed count < images processed" |
| I-3 | APPLIED | ADR-3 two tag tiers (calibrated may gate; uncalibrated boost/explain only, tier in tag table + why-this-matched); risk row added |
| I-4 | APPLIED | ADR-3 default fusion = dense-recall-first tag-boosted, `--strict` opt-in, veto only via honest-no-match; playbook "Precision/FP great but recall dropped" |
| I-5 | APPLIED (a,c) / SKIP-COVERED (b) | ADR-5 resident-set rule + `--text-ttl`; B8 already tree-RSS + per-target; B9 was already restated as sum-of-all-search-artifacts (ambiguity gone) — added ≤90MB stretch + ≤50MB search-resident |
| I-6 | APPLIED | field log: 75.5/113.4/157.9/181.2 img/s **UNRECONCILED**, no published claim may cite any |
| I-7 | APPLIED | ADR-2 crossover restated as measured band ~500k–1M (lanes disagree) + honest ANN build-cost range |
| I-8 | APPLIED | ADR-2: ANN breaks B11's lock-free guarantee; build-once-after-indexing only, re-pass `bench concurrent` |
| I-9 | APPLIED | ADR-7 `pillow-heif` optional extra; playbooks "draft() isn't working" (JPEG-only + format mix) and "Some images never appear" (skip ledger); CORPUS-D row now enumerates the format zoo |
| I-10 | SKIP-COVERED | adjudication ruling 12 — offline hypernym table already in ADR-3 |
| I-11 | APPLIED | four playbooks: disk-full, moved/deleted sources (`exists:false` + `imgtag verify`), model change → reindex, duplicate paths |
| M-1 | APPLIED | §6: OI = taxonomy-only hierarchy, no OI imagery; caltech101 not in the bench (empty) |
| M-2 | APPLIED | risk row for B13 cold start (pre-warm, cached optimized graph, cold/warm split) |
| M-3 | APPLIED | playbook: ship the faster **that passes the fidelity gate** |
| M-4 | APPLIED | dead end now reads "6–22× slower depending on shape/lane — both measured" |
| M-5 | APPLIED | every §8 field-log timestamp suffixed Z; header states UTC |
| M-6 | APPLIED | "three surfaces" wording fixed (collided with the Creations registry invariant); **sha corrected `…3200` → `…0200`**, re-verified on disk against `.deify/vision.sha256` |
| M-7 | APPLIED | dead end: faiss + usearch in one process (unreproduced exit 139) |
| R2-6 | APPLIED | ADR-1: OpenVINO revisit ACTIVE — one REQUIRED bench slot on the target host |
| R2-7 | APPLIED | ADR-4: fp32 is the default on any host not through `doctor`; playbook says fp32 winning on non-VNNI x86 is normal, not an anomaly |
| R2-8 | APPLIED (merged with rev-arch I-5) | new **ADR-13** daemon lifecycle |
| R2-9 | APPLIED | B13 split: dev ≤2s warm / 🐧 ≤5s cold-disk, ≤2s warm; cold is the quoted number; cached optimized graph |
| R2-10 | APPLIED (merged with rev-arch I-7) | ADR-6 flush cadence: timed flusher thread, N=500 or T=1.5s, polite profile never fsyncs more than once per 1.5s (1.5s chosen over R2-10's 2s so B11's ≤2s visibility still holds) |
| R2-11 | APPLIED | ADR-4: target-profile ranking ≠ quality ranking; SigLIP2 = anchor, default must fit the memory ceiling; quality-per-MB-of-RSS column |

## rev-budgets.md

| id | verdict | note |
|---|---|---|
| I-1 | SKIP-COVERED | "pass ≤6min" already gone; B2 is now B1's projection validator |
| I-2 | APPLIED | B2: real 10k run + projection gate ≤0.15, full run once per lock and per darwin round, n=100 is CI-only |
| I-3 | APPLIED | B3: ≥200 distinct never-seen queries, no cache pre-warm, hits reported separately, daemon residency asserted |
| I-4 | APPLIED (partial) | k = min(10, N_pos) + "zero categories at 0.00" adopted; **kept the existing min ≥0.70** rather than the review's ≥0.40 — lowering it would loosen a locked-direction budget |
| I-5 | APPLIED | B8 anti-gaming clause (search must meet B3 p95 right after the idle sample, same run); tree-RSS sampling ≥2Hz; per-worker RSS reported |
| I-6 | APPLIED (merged) | B12 = ≥30-min soak with all 6 leak classes + slope CI + idle-CPU ≤0.5%; the proposed B25 folded INTO B12's compute-leak clause (0 re-embeds, ≤5s/10k, byte-identical manifest) rather than added as a duplicate row |
| I-7 | SKIP-COVERED + APPLIED (e) | polite-mode B1 + same-run B15 already ruled; added the `--full-speed` escape-hatch assertion (nice==0, workers==ncpu) |
| I-8 | APPLIED | B16: mean ≥0.995, p1 ≥0.99, min ≥0.98, bootstrap CI, EXIF/format subset of CORPUS-D |
| I-9 | APPLIED | B17: +12 pts, Karpathy-5k identity verified by image-id intersection, absolute R@10 within 2 pts of the model card |
| I-10 | APPLIED | B10 rewritten: freshness incl. heartbeat, rate MAE ≤10%, ETA error defined against REMAINING time at 25/50/75% (≤0.20) and 10% (≤0.35), emitter CPU ≤1% |
| I-11 | APPLIED | B11 (a)–(d) behavioural clauses incl. writer ≥95% of query-free throughput and exact coverage honesty |
| I-12 | APPLIED | B9 = sum of ALL artifacts, stretch ≤90MB, search-only resident ≤50MB, machine-asserted `bench artifacts` |
| I-13 | SKIP-COVERED + APPLIED | B21 existed (ruling 9); tightened with ≥99.5% valid indexed, 5s per-file timeout, exit-0-with-failed-count, pixel-cap bomb refusal |
| I-14 | APPLIED | new **B22** egress (0 non-loopback connections; model download is the only announced egress) |
| I-15 | APPLIED | ⌂ renamed ⌂-ub (upper bound, not a prediction) + ⌂-real NOT MEASURED + no ⌂ number in any public claim until it is filled; UNKNOWNS I6 marked superseded |
| I-16 | APPLIED | new **B23** footprint + dependency discipline |
| I-17 | APPLIED (merged with R2-9) | B13 warm vs purged-page-cache definitions, both printed |
| M-1 | APPLIED | lock ceremony: `bench/results/<date>-<sha>.json`, Status → `locked@<sha>` |
| M-2 | APPLIED | B1 "sustained" defined (start → last searchable commit, first 5% warm-up dropped, all I/O inside) |
| M-3 | APPLIED | B14 = Chromium 1440×900 + separate 4×-throttled run |
| M-4 | APPLIED | determinism folded into B18(e) (byte-identical results across runs/thread counts, ties by id) |
| M-5 | APPLIED | gallery correctness folded into B18(f) |
| M-6 | SKIP-COVERED | corpus registry (ruling 3); B3 now names CORPUS-C explicitly |
| M-7 | APPLIED | 100k synthetic scan p95 ≤15ms added to B3; UNKNOWNS §3 promise now points at it |
| A-I1 | APPLIED | B15 cache hygiene (`posix_fadvise(DONTNEED)`, MADV_RANDOM, probe cache-hit latency ≤1.5× solo); ORACLE playbook "server got slower and it wasn't CPU" |
| A-I2 | APPLIED | header PRECEDENCE law: B8 hard on the primary target, B17 maximized subject to it; `INELIGIBLE-DEFAULT` pre-filter; size-vs-speed justifications separated |
| A-I3 | APPLIED | inline `PROXY`/`PRIMARY: NOT MEASURED` labelling, **LOCK LAW**, 🐧/🖥 marker semantics inverted (every row is primary-target by default) |
| A-I4 | APPLIED | header: the throttle models core count only; memory-cap claim dropped; docker `--cpus/--memory` named as the better proxy |
| A-I5 | APPLIED | B15 extended to the search path (threads ≤ usable_cores/2, bounded query queue, search-only probe phase) |
| A-I6 | APPLIED | B22: 0700/0600, per-user UNIX socket / loopback-only, mode bits asserted; UNKNOWNS §3 multi-user row |
| A-I7 | APPLIED | B23: AVX2 capability check → compatible build or clear refusal, never SIGILL; manylinux glibc ≥2.17, Python ≥3.10 |
| A-I8 | APPLIED | B20 deployment clause (systemd **user** unit, unprivileged, no privileged port) + B21 `kill -9` survival, resume without re-embedding, RSS after 5 restarts; ADR-13 ships the unit |
| A-M1 | APPLIED | bench result header carries ISA flags, usable_cores, cgroup quota, MemAvailable, kernel, glibc, ORT version+EP |
| A-M2 | APPLIED | B4 budgeted on localhost (SSH tunnel); LAN recorded separately |
| A-M3 | APPLIED | budget identity = (metric, target); a target change resets to provisional, it is not a "tighten" |
| A-M4 | APPLIED | ADR-2 states it holds at 8GB (20.5MB @10k, 205MB @100k, mmap'd); UNKNOWNS I6 superseded, I8 names the manylinux/glibc floor + the two new deps |

## rev-architecture.md

| id | verdict | note |
|---|---|---|
| I-1 | SKIP-COVERED (+detail) | B8 already process-TREE RSS (ruling 8); added ≥2Hz sampling, no shared-page subtraction, `bench resources --tree` |
| I-2 | APPLIED | ADR-11 now specifies ONE transport: shared-memory uint8 slot ring + `free_q`/`ready_q` of indices only, zero-copy ndarray view, deterministic segment names + pgid sweep + `finally` unlink, `slot→owner_pid` reclaim. Pickled tensor queues explicitly forbidden |
| I-3 | SKIP-COVERED (budget) + APPLIED (structure) | B11's search-under-load p95 ≤200ms is stricter than the proposed 250ms, so no new row; the structural fix landed in ADR-5 (separate text session, own thread, query-pending check between batches) |
| I-4 | APPLIED | ADR-6: `open_snapshot()` eagerly opens+mmaps every shard, ENOENT → one manifest re-read; compaction moves to trash/ with the `max(60s, 2× longest query)` grace, asserted under a synthetic slow reader |
| I-5 | APPLIED | **ADR-13** (merged with R2-8): UNIX socket 0600 + `daemon.json` endpoint record, flock singleton (same pattern as the index writer), stale-socket takeover by the flock holder only, `hello` version handshake → shutdown+respawn on model/version skew, `--idle-timeout 0`, systemd user unit |
| I-6 | APPLIED | ADR-6 progress authority = durable manifest count, `in_flight` separate; mirrored in IA core API + Jobs KPIs and B10 |
| I-7 | APPLIED (merged with R2-10) | timed flusher thread, flush on job end/drain/SIGTERM/SIGINT, stalled-tail case in `bench concurrent` |
| I-8 | APPLIED (per adjudication, not per review) | conductor ruling: **xxhash** wins over the review's stdlib blake2b — declared in ADR-7 as an adopted runtime dep, id stays `xxhash64(file_bytes)` (IA), plus the review's real second half: bytes read ONCE, hashed and decoded from the same buffer (saves ~30GB I/O per 10k cold run) |
| M-1 | APPLIED | ids-line ↔ shard-row promoted to an ORACLE §6 invariant with the self-describing `row` field assertion |
| M-2 | APPLIED | IA Dataset identity: slug collision rule with `-<blake2b6(abspath)>`, manifest records `root_path` |
| M-3 | APPLIED | ADR-4: dynamic export axis, fixed runtime batch, pad the final partial batch |
| M-4 | APPLIED | ADR-3: tag table owner = b-engine, location `~/.imgtag/models/<model_sha>/`, one schema; b-bench writes calibration in, b-daemon reads only |
| M-5 | APPLIED | IA gains the frozen five-call core API + daemon wire protocol |

## ROUND 2 (rev-oracle §ROUND 2 + the spike-siglip2 ruling — conductor brief extension)

All rulings ADOPTED as instructed. Note: R2's re-read was of commit `80112d2` (pre-fix), so
several of its "still open" items were already closed at `66227ff`; those are marked
SKIP-COVERED with what closed them. The one real leftover it caught — ADR-4's stale
"fp16 shards / narrow-store-wide-compute" text — is fixed.

| id | verdict | note |
|---|---|---|
| R2 leftover (ADR-4 fp16 shards) | APPLIED | sentence now reads: storage is **f32** per ADR-2; the fp16-shard refinement is marked SUPERSEDED, and the falls-off-BLAS measurement is kept as the *reason* f16 was dropped (6–48× slower matmul + 9.3ms@10k / 97ms@100k convert) |
| R2-1 | APPLIED | ADR-11 POLITE workers = **min(clamp(ncpu−2,2,8), floor(mem_budget_MB / per_worker_RSS_MB))**, per-worker RSS MEASURED by `imgtag doctor` on first run and reported first-class by `bench resources`; ORT loads weights in **external-data format** so weight pages are file-backed/page-cache-shared across workers (also stated in ADR-10c) |
| R2-2 | APPLIED | ADR-10(d) autotune axes now name **worker×thread GEOMETRY** (per-worker sessions vs central session, workers×intra) alongside precision/threads/batch/per-worker-RSS; the 12×1 result is labeled **"M3 E-core artifact, NON-PORTABLE"** at its citation in ADR-4 |
| R2-3 | APPLIED | ADR-11 "shared-box hygiene" clause: `posix_fadvise(DONTNEED)` after each decode, `madvise(MADV_RANDOM)` on shard mmaps, `oom_score_adj=+500` (we die first), optional `RLIMIT_AS` 2GB; B15 post-run probe ≥95% of solo 60s AFTER the job; B8 restated as "our total footprint ≤1.5GB regardless of what co-tenants use" (no ≥6.5GB-free claim existed post-critical-pass) |
| R2-4 | APPLIED | ADR-11 'cores' = `sched_getaffinity()` → cgroup v2 `cpu.max` quota/period → (v1 fallback) → `os.cpu_count()` LAST, with the container-lies note and `doctor` printing the resolved value + source |
| R2-5 | APPLIED | BUDGETS: **B1/B2/B3/B13 split into `-dev` PROXY rows and 🐧 target rows** with interim floors ≥8 img/s · ≤25min POLITE (≤12min FULL) · p50 ≤80ms/p95 ≤200ms · ≤4s cold disk, each "locked only at the first real-server bench, no proxy number published as a product claim"; header states the split + 🖥/🐧 semantics; ORACLE §7 gains **clause (g)** (unlabeled proxy number = stop-and-escalate); the B1/B2-unreachable risk row was already added in round 1 and now points at the split |
| R2-6 | SKIP-COVERED (round 1) | ADR-1: OpenVINO REQUIRED bench slot on the Linux target, not on the Mac |
| R2-7 | SKIP-COVERED (round 1) + APPLIED | ADR-4 already sets fp32-until-`doctor`; the ruling's "(ADR-10d)" home now says it too, inside the autotune axes |
| R2-8 | SKIP-COVERED (round 1) + APPLIED | ADR-13 existed (socket-only, flock singleton, stale takeover, version/model-upgrade restart); this pass renamed the opt-in to **`--tcp` (127.0.0.1 only)** and marked the systemd user unit an **optional** install extra |
| R2-9 | APPLIED | B13🐧 carries the contended-shared-disk honesty note explicitly ("honest cold numbers there will be worse and that is the number that counts") |
| R2-10 | APPLIED (revised) | ADR-6 flush cadence moved from T=1.5s to the ruling's **`min(2.0s, 500 rows)`**, fsync batched per flush never per row, POLITE never fsyncs more than once per 2s. Safe because B11's ≤2s visibility is measured FROM the manifest commit, not from decode |
| R2-11 | SKIP-COVERED (round 1) | ADR-4 already carries "target-profile ranking ≠ quality ranking" incl. SigLIP2's ~1.5GB fp32 pair = the whole B8 budget |
| spike-siglip2 ruling | APPLIED | (1) ADR-4 SigLIP2 line: official int8 vision = SPEED variant that **FAILS B24** (cos 0.7846 / p05 0.65; car recall@5 0.40 vs 0.80), quality anchor is fp32, any int8 must be self-quantized weight-only + pass B24. (2) Blacklist generalized: **downloaded int8 VISION towers fail as a CLASS, 2 for 2** — every quantized vision artifact passes B24 before ANY use; text-tower int8 passed everywhere measured (0.98–0.99), so fp32-vision/int8-text is the safe default shape. (3) Field log `2026-07-22 11:25Z` — the gate caught it on day one, before any index existed |

Note: `BUDGETS.md` was concurrently edited by another writer between the two passes (inline
🐧 planning figures on B2/B3). Those numbers were PRESERVED and promoted into the split rows —
nothing reverted.

## Counts

Round 1: **APPLIED 57 · SKIP-COVERED 7 · CONFLICT 0.**
Round 2: **APPLIED 9 · SKIP-COVERED 4 (2 with added detail) · CONFLICT 0.**
Total: **APPLIED 66 · SKIP-COVERED 11 · CONFLICT 0.**

## Notes for the conductor (no ruling needed, but worth knowing)

1. **rev-arch I-8** was applied against the adjudication (xxhash), not against the review text
   (blake2b8). ADR-7 now carries the dependency decision explicitly.
2. **rev-budgets I-4** would have *loosened* B6's per-category minimum (0.70 → 0.40). Kept the
   stricter existing floor; the review's real defect (mean hides a zero, and p@10 is undefined
   for 8-positive categories) is fixed by `k = min(10, N_pos)` + "zero categories at 0.00".
3. **Two IMPORTANT-class fixes required text from R2-CRITICALs that the critical pass did not
   land** (they are not in ADJUDICATION's rulings): effective-cores resolution (R2-4) is now in
   ADR-11, and the memory-derived worker formula + external-data weight sharing (R2-1) is now in
   ADR-10(c) — both were load-bearing for B8/B15 clauses this pass added, so leaving them
   dangling would have made BUDGETS reference rules that did not exist. R2-3's `oom_score_adj`/
   `RLIMIT_AS` and page-cache clauses landed inside B15 via A-I1/A-I5. **R2-5 (per-profile
   speed rows) landed only partially** in round 1 — **CLOSED in round 2**: B1/B2/B3/B13 are now
   split dev/🐧 with interim floors. B4 stays single-threshold on purpose (🖥 UI-only, measured
   over localhost).
4. `bench` verbs referenced by the new rows and not previously named: `bench artifacts`,
   `bench egress`, `bench footprint`, `bench politeness`, `bench resources --tree`,
   `bench index --headtohead`, `bench search --no-cache-prewarm`. b-bench's brief should carry
   this list.
