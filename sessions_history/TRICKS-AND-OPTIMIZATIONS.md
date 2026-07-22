# TRICKS & OPTIMIZATIONS — the paid-for knowledge

> Every entry below was **measured on this project**, not imported from folklore.
> Format: the trick → why it must be that way → where it lives. If you change the
> related code without reading its entry, you will likely re-buy the lesson at full price.

## Inference & models (CPU)

1. **Weight-only dynamic int8, MatMul-only, is the ONLY quantization recipe that
   survived.** Naive full int8 quantization of vision towers fails parity catastrophically
   (cos 0.94 → recall halved). Two *official* downloaded int8 vision towers were broken
   out of the box (cos 0.78, 0.008). Rule: **quantize weights only
   (`MatMulConstBOnly`, QUInt8, per-tensor), and gate EVERY quantized artifact through
   the B24 parity bench before it may ship.** → `src/imgtag/bench/quant.py`, ORACLE ADR-4/B24.
2. **Mean-cosine parity LIES; nearest-neighbor agreement is the real gate.** int8 text
   read cos 0.982 vs fp32 — looked safe — while **28% of query neighbor-sets shifted**
   (R@10 −3.1pt). Gate on `nn_agree`, never on mean cos. (Learned twice. Don't be third.)
3. **fp16 weights on CPU are a disk trick, not a RAM trick.** ORT's CPU EP has no fp16
   MatMul kernels: every fp16 weight Casts→fp32 per call, so peak RSS *rises* (938MB vs
   622MB fp32). Bit-equivalent output, half disk, worse memory. Use only if disk-bound.
4. **Text towers are RAM bombs — keep them lazy and int8.** fp32 text resident ≈ 850MB;
   int8 ≈ 183MB. Load lazily, keep-while-under-watermark, release on pressure.
   Named-tag queries don't need the tower at all (see 7). → ADR-5.
5. **Batch size 1–2 streaming beats big batches on CPU.** Batch-8 ≈ 2× RSS for ~zero
   throughput gain. intra_op threads have a knee (4–6 on M3-class); autotune per machine
   (`doctor`), never `cpu_count()`. → ADR-11/13.
6. **Normalize embeddings IN the embed function, always.** PE-Core emits unnormalized
   vectors (norms 5–21); forgetting L2 silently poisons every downstream cosine.
7. **Pre-tokenize to a compact binary.** Parsing a 34MB tokenizer.json costs 0.64s per
   process. We ship an ~11MB mmap-able binary (UTF-8 blob + int32 offsets; merges as
   int64-packed sorted pairs + np.searchsorted). → `scripts/build_tokenizer.py`.
8. **A tag table makes common queries tower-free.** Precomputed tag embeddings serve
   named-tag searches from mmap (`served_by: "tag-table"`) — that's how idle RSS stays
   ~52MB and warm search stays single-digit ms.

## Search quality & calibration

9. **Fuse paths in PROBABILITY space only, and fail OPEN.** Per-tag Platt fits (on a
   calibration set) + a dense-path logistic; `p = max(p_tag, p_text)`. An unfitted
   threshold may rank but NEVER veto — "I cannot judge" must never be dressed as
   "nothing matched". Label every result `fitted`/`unfitted` per-query. → ADR-3.
10. **Uncalibrated tags may boost, never dominate.** Corpus-relative scoring saturates on
    homogeneous data: an uncalibrated hypernym expansion ("weapon"→sword/missile) hit
    fake p≈0.99 and buried the honest results. Cap any uncalibrated tag's p at the best
    honest signal for that image (calibrated-anchor p, else dense p). Same root cause as
    the OOD tier guard (13). → `search.py` + regression test.
11. **One global embedding cannot separate co-occurring objects — don't tune, tile.**
    Multi-object AND-query precision has a measured ceiling of 0.281 even with a
    val-fitted (cheating) threshold. No calibration beats it; region/tile embeddings
    (D1) are the only real fix. Ship recall-first + honest labels meanwhile.
12. **Free-text feature = raw cosine, not z-score.** Z-scores inverted separability on
    real data (nonsense max-z EXCEEDED real max-z): 60% vs 77% separation. Measure your
    feature choice; the "obvious" normalization lost.
13. **Absolute floor before any corpus-relative tier.** On a topically-uniform dataset
    (all guns, all swimwear) relative z-tails fire en masse (160/166 phantom flags).
    An absolute-margin floor (≈0.02) before relative tiering kills the class.

## Moderation architecture (the 100-track laws)

14. **Store raw scores; derive tiers at read.** One f32 sidecar column per track per
    dataset, row-aligned to shards. Threshold/policy changes are then FREE and
    retroactive (`track recount`, no re-scoring). → ADR-15, `TRACKS.md` T1.
15. **Track cost must not grow with the encoder.** Instrument ladder: embedding matvec
    (default; 100 tracks ≈ microseconds) → distilled head (same cost class) → dedicated
    model ONLY with a measured zero-shot failure, budgeted ≤30% encoder FLOPs total.
    Escape hatch if dedicated heads multiply: one shared small backbone + N linear heads.
16. **READER PARITY: a track is done when the SERVING path reproduces its numbers.**
    τ must live in the score space the reader gates in — a prob-space τ gated a
    margin-space reader to 0/1856 while the lane's own harness said 137. Assert
    head-counts == derived-counts on a probe slice. Three bugs, one law. → TRACKS (e).
17. **Tiers-as-data.** A track's prompt-set keys ARE its tiers; adding a track or a tier
    is a JSON edit, no reader change. Severity: alert > violation > review; `match` is
    content, NEVER summed into moderation totals. Assignment by exceedance, not
    severity-order (else the review tier becomes unreachable).
18. **Heads with internal arbitration must be CONSUMED, never re-banded.** Drugs decides
    vape→review inside its head; τ-banding its raw score resurrects vape→violation —
    a user-policy violation. Store both margins (multi-col sidecar + col_roles) so the
    stored derivation reproduces the arbitration exactly.
19. **Probe datasets are how thresholds get earned.** Index a labeled true-positive set
    as a real dataset; TP-vs-FP separation on YOUR corpus yields τ with a Wilson CI.
    TP-scores must dominate the FP band or the instrument isn't ready (say so).
20. **Keyword mining fails for pose/context classes.** "gun"→guns works; "person lying
    down"→kittens/flat-lays (4/4 mislabeled sample). Pre-filter pulls with a person/
    object detector, or hand-verify — and let a weak alert tier stay WITHHELD: a false
    "someone may be hurt" is worse than none.

## Storage & durability

21. **The 7-step durable flush** (data fsync → ids fsync → manifest tmp+fsync+rename →
    dir fsync) + truncate-on-open recovery + byte-count-as-authority (readers cap at the
    manifest's counts, never stat files). Survived kills and a mid-day history rewrite.
22. **Content-addressed ids (xxhash64 of bytes) + dedup BEFORE decode.** Duplicate work
    skipped cheaply; global-search collapse (`also_in`) and writer-side duplicate refusal
    both fall out of it. Dense columns even for sparse detectors (NaN-pad per append,
    finalize to count) or mmap alignment breaks under worker-arrival timing.
23. **flock is the liveness authority.** Not pidfiles, not status strings — the kernel.
    (Queued jobs get recorded-pid + grace as refinement.)

## Operations (single box, co-tenants, agent teams)

24. **Politeness is architecture:** nice 10, oom_score_adj +500, workers =
    clamp(ncpu−2, 2, 8) memory-derived, posix_fadvise(DONTNEED), non-JPEG decode
    semaphore. FULL speed is opt-in. A co-tenant index at --full-speed contaminated a
    benchmark window in minutes — the polite default exists for a reason.
25. **img/s claims need a quiet window with an ACK handshake.** Load gates get silently
    unmet; a GO message can sit batched in an inbox while you *think* measurement is
    running. Protocol: holds → GO → explicit "WINDOW OPEN, firing" → DONE → release.
    And know your floor: a desktop's own baseline can make strict gates structurally
    unreachable — record per-row loadavg and label honestly instead of waiting forever.
26. **Multi-agent git on one worktree:** commit with explicit pathspecs
    (`git commit -- <paths>`) — plain `git commit` sweeps whatever OTHER sessions staged
    into your commit (bit four separate lanes in one afternoon). Never `git add -A`,
    never bare `git stash`. Messaging a "dead" agent RESUMES it — check before spawning
    a successor, or you get two owners racing one lane.
27. **Distrust self-reports; verify through the surface the user touches.** The panel
    "not working" was a stale browser cache + a stale daemon process — the code was fine.
    Serve `Cache-Control: no-store` during live development; restart daemons after
    landing code; then verify with a real browser click, not a code read.

## When you extend this system

Read `ORACLE.md` §Dead Ends before building anything you think is missing — several
"obvious improvements" are already measured failures (image-relative presence tests:
inert; scene hard-negatives for sports: −3pts; beach negatives for safety: hurt true
positives; z-score text feature: inverted). The map marks the cliffs so you don't have to.
