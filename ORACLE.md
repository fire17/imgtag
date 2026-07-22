# ORACLE.md — the IMGTAG pseudo-oracle (wartable output)

> **Consult this before improvising. If reality diverges from this oracle: STOP, append the
> divergence to §8 Field log, and escalate to the orchestrator — do not improvise past a
> broken map.** Forged 2026-07-22 by the planning session at maximum context (post-research,
> pre-build). Companions: VISION.md (verbatim law) · UNKNOWNS.md (blind spots) · BUDGETS.md
> (numbers) · IA.md (views) · research/ (evidence). Dated entries; derived content prunable.

## 1. Context capsule

IMGTAG = CPU-only, blazing-fast, open-vocabulary semantic image search over local datasets
(~10k images; must age well to 100k+ and degrade to old machines). One core engine, three
doors: CLI, local web app, global agent skill. The user cares about under-the-hood > UI.
WHY: every incumbent is 0.2–2 img/s self-hosted with unpublished quality and no honest
numbers; the field has a measurement vacuum. IMGTAG wins by (a) treating decode as the
engine, (b) shipping a modern model by default, (c) publishing the first honest CPU bench,
(d) calibrated correctness (an honest "no match"), (e) live observability + search-while-
indexing, (f) being the only agent-native engine. The deeper why: this is fire17's
highest-bar protocol — verbatim vision, budgets as tests, honest verification, and a
60-min darwin self-improvement loop after delivery.

## 2. Decision records (ADR — chose X over Y because W; revisit-if)

- **ADR-1 Runtime = ONNX Runtime CPU EP.** Over ggml (ORT MLAS measured 2–6× faster on
  CPU; Ente migrated ggml→ORT), over torch (weight, cold start), over OpenVINO-as-primary
  (Intel-centric), over CoreML (violates CPU-only law; opt-in future).
  **OpenVINO's revisit is now ACTIVE**
  (rev round 1): the target IS x86, and non-VNNI AVX2 int8 is MLAS's weakest configuration
  and roughly OpenVINO's strongest relative footing — OpenVINO gets one REQUIRED bench slot
  **on the target host** (not on the Mac). If it wins ≥1.5× there it becomes the x86 EP behind
  the same pluggable-backend seam ADR-4 provides, and its ~100MB install weight is then a
  measured trade, not a guess. Revisit if: bench shows another EP ≥1.5× on target hardware.
- **ADR-2 No vector database. Exact brute-force scan over L2-normalized contiguous
  mmap'd f32 shards. f16 is DROPPED entirely** (rev round 1 adjudication: numpy has no
  f16 BLAS — f16 matmul measured 48× slower, f16→f32 convert 9.3ms@10k/97ms@100k; disk
  saving is trivial at our scale, and mmap'd f32 pages are file-backed/evictable — kinder
  to the 8GB box than any heap mirror). Measured f32 scan: 0.28ms @10k, 3.2ms @100k.
  `bench search` asserts `dtype == float32` on the array actually handed to the matmul.
  Memory holds on the 8GB target: 10k×512 f32 = 20.5MB, 100k = 205MB, mmap'd and evictable —
  scale is not the memory problem, worker processes are (ADR-10c).
  Revisit only past the **measured crossover band (~500k–1M vectors; lanes disagree — priorart
  says 100k–500k, runtime says ≥1M; re-measure before acting)** → binary-quantized coarse pass
  + f32 rerank, then usearch HNSW i8. ANN build cost at 10k–100k measured 0.2–105s depending on
  library and config, to save ≤0.3ms/query. ⚠️ **Any ANN adoption BREAKS the lock-free
  search-while-indexing guarantee (B11)** — every insert mutates a shared graph. An ANN path
  must therefore be build-once-after-indexing with brute force serving during jobs, and may not
  be adopted without re-passing `bench concurrent`.
  NEVER pgvector/sqlite-vec for the hot path at this scale.
- **ADR-3 Hybrid retrieval architecture.** Embedding index (free-text recall) + tag
  vocabulary (~4–8k tags from COCO/LVIS/OI names + curated) scored via the SAME text
  encoder at index time (one matmul, marginal cost ~0) + query-time hypernym expansion.
  Over embedding-only (fails: multi-object recall, superordinate queries, no honest
  no-match) and over heavy tagger models (RAM++ dead end). Tags are the FP gate and the
  explanation surface — but **only in the tiered, rank-boosting form below**.
  **Two tag tiers (rev round 1):** *calibrated* tags (τ_t fitted for max-F1 on COCO/LVIS
  ground truth — at most ~1,283 of the 4–8k vocabulary can ever be) are the ONLY tags allowed
  to hard-gate or to produce an honest "no match"; *uncalibrated* tags get a conservative
  family-prior τ, may boost rank and explain a hit, and may NEVER gate or veto. The tier is
  stored in the tag table and shown in `why-this-matched`.
  **Default fusion = dense-recall-first, tag-boosted** (tags raise rank; they do not filter).
  `--strict` flips calibrated tags to a hard AND. The honest-no-match path is the ONLY place a
  threshold vetoes results: zero results only if no calibrated tag in the expanded query set
  passes τ AND the best dense score is below the global no-match floor. (A hard AND by default
  trades B5/B6 recall for a B7 win — the vision says "**all** of the images with cars".)
  **Tag table ownership:** one owner (b-engine), one location
  `~/.imgtag/models/<model_sha>/{tags.f32,tags.json}`, one schema
  `{names[], dim, model_sha, prompt_ensemble_sha, tier[], tau[], platt[], provenance{}}`;
  b-bench writes calibration INTO that file, b-daemon only reads.
  **Hypernym data is a PRECOMPUTED static table** built offline from
  LVIS synsets + OI 600-class tree + COCO supercats (all on disk) — no nltk/WordNet at
  runtime (ADR-7 intact). **CALIBRATION CONTRACT (rev round 1, rev-arch C-7 — mandatory):**
  (1) two-layer thresholds: model-layer per-tag Platt fit on COCO+LVIS (held-out CAL-SET)
  → `p_tag`, shipped per model_sha; dataset-layer streaming stats (per-tag mean/std/p99
  over THIS corpus, accumulated free during the index matmul, stored in manifest ~96KB)
  → effective τ = max(τ_tag, mean+k·std). (2) **Fusion in probability space ONLY** —
  never max-pool raw cosine against calibrated probabilities. Free-text calibration
  feature (RE-RULED 2026-07-22 — the original per-query z-score spec was FALSIFIED with
  data: max-z is bounded by corpus shape, ~3–4 for every query regardless of truth;
  nonsense out-z-scored real queries, 60% separation vs chance 50): **default feature =
  raw absolute cosine → logistic (A,B) + τ fit on CAL-SET per model_sha** (77% measured
  separation); `text_feature` is selectable in tags.json ("cos"|"z"); COMMISSIONED
  EXPERIMENT: background-prompt margin (score minus max over K generic negative prompts,
  one cached text batch) — if it beats cos on CAL-SET separation, it becomes the default.
  Fuse p = max(p_tag, p_text), record which path won (= the "why this matched" payload). (3) Near-tag rule: query inherits a
  tag's calibration only if cos(q, tag) ≥ θ_syn (fit on COCO synonym pairs); compound
  queries NEVER inherit a component tag's threshold — unit test with "my dog wearing a
  santa hat". (4) Manifest records calib_sha + calib_model_sha; tag-path search REFUSES
  loudly on mismatch (same mechanism as the model/manifest refusal). Re-calibration is
  structurally impossible to forget, not a discipline.
- **ADR-4 Pluggable model backends; Apache-2.0 default.** Bench roster (2026-07-22):
  PE-Core-S16-384 + T16-384 (Apache; export spike first), SigLIP2-base-224 (Apache, official
  ONNX — **the QUALITY ANCHOR is its fp32**; the official onnx-community **int8 vision export
  is a SPEED variant that FAILS B24**: measured 2026-07-22 by spike-siglip2 at mean cos 0.7846
  / p05 0.65 vs its own fp32, with real quality damage — COCO "car" recall@5 0.40 int8 vs 0.80
  fp32, every GT rank worse or equal. Any int8 we ship must be SELF-quantized weight-only and
  pass B24), SigLIP-v1-base (Apache, small text tower),
  UForm (Apache, Matryoshka 64d dark horse), OpenCLIP ViT-B/32 (MIT control),
  MobileCLIP2-S0/S2 (apple-amlr — REFERENCE CEILING ONLY, opt-in plugin, never default,
  never in published artifacts), rclip (system-level head-to-head baseline).
  Winner chosen by bench on: index img/s, search ms, precision@10, hypernym recall, FP
  rate, RSS, disk. Revisit if: FG-CLIP2/OpenVision v1 export cheaply and beat the set.
  **Measured refinements (runtime lane, 2026-07-22, research/runtime.md — first-party):**
  quantization status (re-ruled 2026-07-22): the weight-only recipe was REFUTED as a
  general fix — on PE-Core vision every int8 variant swept (weight-only per-tensor/
  per-channel, QInt8/QUInt8, full-graph) fails B24 tier-1-default at n=200; the earlier
  0.96 agreement was an n=24 pool artifact (pool-size law: agreement falls with n — B24
  now fixes n=200). **v1 ships fp32 vision everywhere**; int8 remains a per-arch opt-in
  candidate under B24's two-tier gate (openclip-B/32 int8 measurably costs nothing at
  n=200 quality — the gate now distinguishes harmless from harmful instead of banning by
  proxy metric). Static/calibrated quant remains unexplored on the TARGET arch only. **FIDELITY GATE (CI): cos ≥0.98 AND
  top-1 NN ranking agreement ≥0.90 vs fp32** — ranking agreement is the metric that
  matters, mean cosine hides rank flips (per-channel scored 0.83 agreement at cos 0.955).
  ☠️ BLACKLIST — **downloaded int8 VISION towers fail fidelity as a CLASS (2 for 2 measured):**
  `Xenova/mobileclip_s0` (cos 0.008 vs its own fp32, AND 3.4× slower) and onnx-community's
  official `SigLIP2-base` int8 vision (cos 0.785, car recall halved). **Every quantized vision
  artifact, whatever its source, passes B24 before ANY use** — official ≠ audited. Text-tower
  int8 has passed everywhere measured (0.98–0.99), so the fp32-vision / int8-text split (Ente's
  choice) is the safe default shape; MobileCLIP vision towers stay fp32 or self-quantized
  behind the gate.
  Parallelism geometry is **host-probed, not fixed**: the 12 workers × 1 ORT intra-op thread
  = 181 img/s result is an **M3 E-CORE ARTIFACT, NON-PORTABLE** — its stated cause (ORT's
  intra-op pool scheduled onto efficiency cores) is Apple-specific and has no x86 analogue;
  label it that way wherever it is cited. `imgtag doctor` sweeps `workers × intra_op × batch ×
  precision` AND the session geometry itself (per-worker sessions vs one central session) under
  the memory ceiling (ADR-10c); on ≤4-core hosts expect single-process/multi-thread to win on
  memory alone.
  **Default precision on any host that has not been through `doctor` is fp32** — int8 is
  enabled only by a measured win on that host (ADR-10e); the recipe above is how we quantize,
  not a promise that we will (pre-VNNI x86 losing to quantization is the normal case there,
  not an anomaly). Batch axis: export with a **dynamic** batch axis, run with a **fixed** batch
  B from the bench, and pad the final partial batch to B (discarding padded rows) — one shape
  for the entire run, no MLAS re-planning. **Embedding storage is f32** (ADR-2; the earlier
  "fp16 shards / narrow-store-wide-compute" refinement is SUPERSEDED — f16 was measured
  lossless as *storage* but numpy f16/int8 matmul falls off BLAS entirely, 6–48× slower, and
  the convert step costs 9.3ms@10k/97ms@100k, so the narrow store bought nothing at our scale.
  The falls-off-BLAS fact stands and is exactly why f16 was dropped.)
  **Target-profile ranking ≠ quality ranking:** SigLIP2 is the quality ANCHOR (reference for
  how good we could be — its fp32 pair is ~1.5GB, the entire B8 budget for one model); the
  shippable default on the 8GB profile is the best model whose vision tower + resident text
  path fits the memory ceiling (PE-Core-T16/S16, SigLIP-v1's 111MB int8 text tower, UForm).
  The bench reports a quality-per-MB-of-RSS column and the default is chosen on the TARGET
  profile, not the dev box (BUDGETS precedence law: B8 hard, B17 maximized subject to it).
- **ADR-4b DEFAULT MODEL DECIDED (conductor, 2026-07-22 13:47Z, from the completed
  CORPUS-A quality matrix — contention-immune numbers).** Quality table (B6 mean / B5
  p100 / B17 R@10 / B7 leak@unfitted): siglip2-base .925/.937/77.5/.2 · siglip-v1
  .893/.917/80.5/.2 · pecore-s16 .893/.927/77.2/.3 · pecore-t16 .841/.908/70.9/.47 ·
  openclip control .775/.832/65.2/.6. Every candidate clears B17's +5-over-control.
  Under the B8 PRECEDENCE law (hard memory cap on the 8GB target), eligible defaults are
  pecore-s16 (850MB proj) and pecore-t16 (459MB); s16 dominates t16 on every quality
  axis. **DEFAULT = pecore-s16-384 (fp32 vision + int8 text).** Doctor MAY select
  siglip2-base-224 as a quality step-up on roomy machines (its .925 B6 is best-in-matrix
  but B8-INELIGIBLE on the target); pecore-t16 remains the edge-floor option. Caveats
  honestly attached: B6-min/B5-min-child/B7 clauses are red across ALL candidates at
  unfitted τ — the CAL-SET fit (pending) + per-tag calibration are expected to lift
  them; budget-lock happens only after the fit re-run. Revisit if the fitted re-run
  reorders the eligible set.
- **ADR-5 Resident daemon + warm text tower.** Anti-pattern proven: immich unloads models
  after 300s → 60–70s cold search. LRU query cache, tag table precomputed. CLI talks to the
  daemon when present, else in-process (still ≤2s cold, B13).
  **Resident set (rev round 1, the rule ADR-5's "few hundred MB" was hiding):** vision session
  (indexing only, released when idle) + **precomputed tag table** (this is what makes tag-path
  search need no text encoder at all) + text tower **lazily loaded, LRU-evicted after
  `--text-ttl`** (default: never on desktop, 300s on the 8GB server). B8's idle cap is asserted
  together with a B3-passing search in the same run, so eviction can never be used to game it.
  **The text tower gets its OWN ORT session** (`intra_op=1..2`) and its own thread — never
  shared with the image session — and the batch dispatcher checks a "query pending" flag
  between batches; otherwise a 32-image batch is ~530ms of head-of-line blocking and search
  during indexing (B11b) silently dies while B3 stays green.
- **ADR-6 Storage = append-only shards + atomic manifest + EXCLUSIVE WRITER + DURABLE
  FLUSH** (rev round 1: the original "falls out free" claim was false — two writers or a
  crash could silently permute row↔id mapping forever). `~/.imgtag/datasets/<slug>/`:
  - Files: `shard-<jobid8>-<seq:04d>.f32` + `ids-<jobid8>-<seq>.jsonl` (generation-scoped
    names — two writers can never land in one file) + `manifest.json`. Manifest per-shard
    record: `{name, rows, emb_bytes, ids_bytes}` — **byte counts are the authority; readers
    never stat() shard files** and cap reads at manifest counts. Line i of ids ↔ row i of
    shard is a WRITTEN invariant; each ids line also carries `"row": <global_index>`.
  - **Writer exclusion:** `.writer.lock`, `fcntl.flock(LOCK_EX|LOCK_NB)` held for the whole
    job; failure = exit 3 with pid/job/since message. Kernel releases on any death — NO
    pid-liveness heuristics. Lock contents (JSON) are advisory for the error message only.
    Readers take no lock ever. **The flock holder is the SOLE manifest writer** (the
    coordinator); decode/embed workers never touch `manifest.json` — they hand durable row
    counts back to the coordinator, which merges them. Two processes calling
    `os.replace("manifest.json")` is last-write-wins and silently loses shard counts.
  - **Flush cadence:** the flusher is its own thread on a `Condition.wait(timeout=T)` loop and
    flushes at `min(T, N-rows)` boundaries — default **T=2.0s** (aligned with B11's ≤2s
    visibility, which is measured FROM the manifest commit) or N=500 rows, whichever comes
    first. **fsync is batched per flush, never per row**, and the POLITE profile never fsyncs
    more than once per 2s: `ionice` does not throttle fsync, and frequent fsyncs on a shared
    disk hurt co-tenants measurably. Also flush on job end, on
    queue-drain, and on SIGTERM/SIGINT (install handlers). Without the timer the tail of a job
    (slow TIFFs, a stalled queue) stays invisible far past B11 — possibly forever if aborted.
    `bench politeness` reports commits/s; `bench concurrent` includes a **stalled-tail** case.
  - **Progress authority:** progress `done` = **durable manifest count**, never dispatched
    count; in-flight is a separate field, rendered differently. The IA coverage banner and the
    progress bar read the same number by construction (else a crash makes progress jump
    backwards and poisons the ETA model).
  - **Flush protocol (order mandatory):** (1) buffered write() batch to shard — never
    mmap-write; (2) fsync(shard); (3) write() ids lines; (4) fsync(ids); (5) write
    manifest.tmp + fsync; (6) rename→manifest.json; (7) **fsync(dataset dirfd)** — the
    step everyone omits.
  - **Recovery (on every open-for-WRITE, never on read):** truncate any shard/ids file
    longer than its manifest byte count (torn tail from crash); shorter than manifest =
    FAIL LOUDLY + quarantine; assert emb_bytes % (D×4) == 0 and == rows; log every
    truncation. Orphan shards not in manifest → moved to trash/ at job start, never read,
    never deleted inline.
  - **Snapshot opens eagerly:** `open_snapshot()` opens and mmaps EVERY shard named in the
    manifest before returning — POSIX protects already-open fds, not future opens, so a lazy
    open racing a compaction is a `FileNotFoundError` mid-search. `ENOENT` during snapshot →
    re-read the manifest once and retry; second failure = loud error.
  - Compaction writes new files then swaps manifest; superseded shards are MOVED to trash/
    (never `unlink`ed inline), swept at the next job start once older than
    `max(60s, 2× longest observed query)`. `bench concurrent` asserts this by compacting
    under a synthetic slow reader.
- **ADR-7 Engine deps = onnxruntime + numpy + Pillow + certifi/httpx + xxhash + micro-server.**
  NO torch/transformers at runtime (export tooling may use them offline). uv-managed venv.
  **`xxhash` is an explicit, adopted dependency** (rev round 1, adjudicated): the image id is
  `xxhash64(file_bytes)` (IA.md), the wheel is tiny and permissive, and a stdlib fallback
  (`zlib`/`hashlib`) was considered and rejected as slower or wider — declaring the dep here is
  what stops two builders inventing two different ids. The worker reads each file's bytes ONCE
  and both hashes and decodes from that same buffer (`Image.open(io.BytesIO(buf))`) — hashing
  then re-reading costs ~30GB of extra I/O per 10k cold run, straight out of B1/B2.
  **`pillow-heif` is an OPTIONAL extra** (`imgtag[heic]`); without it HEIC files are skipped
  with a named, counted, actionable error — never a silent gap in the index. Portability floor:
  manylinux glibc ≥2.17, Python ≥3.10, no Docker, no root, no privileged port (B23).
- **ADR-8 Idea reuse: concepts from AGPL tools, code only from MIT/Apache.** (UNKNOWNS I9.)
- **ADR-9 CPU-only is law** (VISION verbatim: "using only the cpu"). Accel lanes (CoreML)
  are future opt-in flags, never required, never the benchmarked claim.
- **ADR-10 Primary deploy profile = shared Linux x86 server, 8GB, no GPU, co-tenants
  sacred** (user constraint 2026-07-22, VISION-ADDENDA.md verbatim). Consequences:
  (a) optimize for x86 AVX2 baseline — never assume AVX512/VNNI, never assume Apple
  silicon; M3 Max numbers are PROXY, labeled as such, until the bench runs on the real
  server; (b) politeness-first defaults: nice ≥10 + ionice, workers ≤ cores/2, bounded
  queues, `--full-speed` opt-in (B15 is a hard budget with a co-workload probe test);
  (c) memory ceiling is structural: streaming decode, bounded buffers, small-model bias
  (PE-Core-S16/T16, SigLIP-v1, UForm gain rank vs SigLIP2's fat text tower; B8 tightened
  to ≤1.0GB indexing / ≤1.5GB total under load). **Worker count is memory-derived, not
  core-derived:** `workers = clamp(1, min(effective_cores−2, floor((RSS_budget − daemon_RSS) /
  measured_per_worker_RSS)), 8)` — each worker is a separate interpreter with its own ORT
  session (spawn ⇒ no COW sharing), realistically 200–350MB int8 / 450–600MB fp32, so the 1.0GB
  ceiling admits ~2–3 int8 workers or ONE fp32 worker on the target. `bench resources` reports
  per-worker RSS as a first-class number and `imgtag doctor` measures it on the host before
  choosing geometry. Mitigation that makes multi-process affordable: export with external-data
  weights and load them mmap'd/read-only shared across workers, `arena_extend_strategy=
  kSameAsRequested`, mem-pattern off (external-data weights are file-backed, so those pages are
  page-cache-shared across workers instead of copied per process); (d) **first-run autotune**:
  `imgtag doctor` runs a ~30s micro-bench on the actual deploy machine — fp32 vs int8 (**and
  on an unprofiled x86 host the default stays fp32 until int8 proves itself here**), thread
  count, batch size, per-worker RSS, and **worker×thread GEOMETRY (per-worker sessions vs one
  central session, workers × intra_op)** — the axis most likely to invert between the ARM dev
  box and the x86 target — then stores the recipe in the machine profile; "generic and ready"
  means the engine adapts itself, not that we guessed;
  (e) quant decisions are PER-ARCH — NEON results never transfer to AVX2 (clip.cpp
  anomaly was x86; DeepSparse win was AVX512-VNNI; measure on target). Revisit if: the
  user later asks for Mac-local optimization (add profile, change no defaults).

- **ADR-11 Resource policy — ONE place, both throughput and politeness measured from the
  SAME run** (rev round 1, rev-arch C-4/C-5/C-6). POLITE (default): decode_workers =
  **min(clamp(ncpu−2, 2, 8), floor(mem_budget_MB / per_worker_RSS_MB))** — the CPU clamp AND
  the memory cap, whichever is smaller; `per_worker_RSS_MB` is MEASURED by `imgtag doctor` on
  first run (never assumed) and is a first-class `bench resources` number. ORT sessions load
  weights in **external-data format** so weight pages are file-backed and page-cache-shared
  across workers rather than copied per interpreter. · ORT intra_op=2, inter_op=1 · total
  threads ≤ ncpu−1 asserted at
  startup · os.nice(10) set IN EACH worker initializer (not only inherited). FULL
  (`--full-speed`): workers=ncpu, no nice. B1 headline = POLITE run; FULL number always
  labeled. **"ncpu" means EFFECTIVE cores**, resolved in this order: `len(os.sched_getaffinity(0))`
  → cgroup v2 `/sys/fs/cgroup/cpu.max` quota/period → cgroup v1 `cpu.cfs_quota_us/cfs_period_us`
  → `os.cpu_count()`. Never `os.cpu_count()` alone — a shared server is very likely a
  cgroup-limited container where it reports the HOST's cores and we spawn 8 workers against a
  2-CPU quota (the exact harm the user forbade). `imgtag doctor` prints the resolved value, its
  source, and the resulting geometry.
  **Transport (SPECIFIED — the design is not buildable without this):** a
  `multiprocessing.shared_memory` slab of S slots × H×W×3 **uint8** (≈442KB/slot at 384²;
  S = 4×batch, default 128 slots ≈57MB) plus two small `multiprocessing.Queue`s carrying
  **integers and metadata only** — `free_q` (slot indices available to workers) and `ready_q`
  (`{slot, image_id, path, w, h}`). Backpressure is structural: a worker blocks on
  `free_q.get()` when the consumer is behind. The consumer builds the ORT input as
  `np.ndarray(shape, np.uint8, buffer=shm.buf, offset=slot*SZ)` — a view, no copy — batches B
  contiguous slots, runs, and returns the slots to `free_q`. Normalization is FUSED into the
  ONNX graph (Ente trick), so workers need Pillow only, no numpy. **Never a pickling
  `multiprocessing.Queue` of tensors** — that is 106MB/s of serialize+pipe+deserialize at
  60 img/s, syscall-bound, on the one axis the project claims as its edge. **Segment naming and
  cleanup are mandatory** (Python's `resource_tracker` leaks segments on abnormal exit): name
  them `imgtag-<pgid>-<n>`; on daemon/CLI start sweep and unlink any `imgtag-*` segment whose
  pgid is dead; always `unlink()` in a `finally`. Track `slot→owner_pid` and reclaim slots when
  `Process.exitcode is not None` — otherwise a dead worker's slots leak from the free list and
  a long run silently degrades to a hang. Non-JPEG decodes capped
  by semaphore(4) (draft() is JPEG-only; PNG/HEIC full decodes are the biggest RAM term).
  Pipeline geometry (central session vs per-worker sessions) is a `doctor`-swept axis; the
  policy's totals govern either. Memory arithmetic vs B8 is re-measured, never assumed
  (spawn start-method means zero COW sharing — 16 workers ≈ 1.4GB of interpreters alone;
  hence the clamp).
  **Shared-box hygiene (non-negotiable on the primary target — `nice` and `ionice` do NOT
  cover any of it):** `posix_fadvise(POSIX_FADV_DONTNEED)` on each image file after decoding it
  (streaming 10k images through the page cache evicts a co-tenant's hot working set — their CPU
  looks unchanged while their latency triples, and a during-run CPU probe sees nothing);
  `madvise(MADV_RANDOM)` on shard mmaps; `oom_score_adj = +500` on every process we own — **we
  die first, the co-tenant never does**; optional `RLIMIT_AS` at 2GB as a hard backstop.
  B15 asserts the other half: the co-workload probe is re-measured for 60s AFTER our job ends
  and must be back to ≥95% of solo (page-cache recovery) — a during-run pass alone does not
  satisfy B15.

- **ADR-13 Daemon lifecycle — one contract, verbatim-implementable** (rev round 1, rev-arch I-5
  + rev-oracle R2-8; ADR-12 left unassigned so review references stay stable). Unspecified,
  four builders invent four answers and the agent door (b-skill) silently pays a 2s cold start
  per call, destroying ADR-5's whole rationale.
  - **Transport: UNIX domain socket** `~/.imgtag/daemon.sock`, mode 0600, per user — never a
    TCP port on a shared box (port conflicts, multi-user collisions, and a privacy leak:
    binding TCP exposes one tenant's photo search to the others). **`--tcp` is an opt-in flag**
    for the app's browser use; it binds 127.0.0.1 ONLY and refuses any non-loopback bind. `~/.imgtag` is 0700 (B22).
  - **Single instance:** the daemon holds `fcntl.flock(LOCK_EX)` on `~/.imgtag/daemon.lock`
    for its lifetime — the SAME kernel-owned pattern as the index writer (ADR-6), no pid
    heuristics. It publishes `~/.imgtag/daemon.json` = `{pid, version, socket, http_port|null,
    started_at, models:{id,sha}}` — the endpoint record every door reads.
  - **Client algorithm:** (1) connect to the socket; (2) on ENOENT/ECONNREFUSED try
    `flock(LOCK_EX|LOCK_NB)` — if acquired: unlink the stale socket, fork-exec the daemon,
    release, poll connect ≤2s; if not acquired another client is already starting one: poll
    connect ≤2s with 25ms backoff; (3) on timeout **fall back to in-process** and print one
    line saying so — never fail the user's query. **Only the flock holder may unlink the
    socket** (this is what stops a client killing a live daemon's socket).
  - **Version / model-upgrade restart:** first request is `hello{client_version}`; the daemon
    replies `{version, model_shas}`. Mismatch → client sends `shutdown`, waits for socket
    removal (≤2s), respawns. A stale daemon serving model A while the manifest says model B is
    the one case ADR-6's loud refusal cannot catch (it fires inside one process only).
  - **Idle policy:** default `--idle-timeout 0` (never exit) — immich's `model_ttl=300` is the
    proven anti-pattern; RAM protection comes from the CLI's in-process mode and `--text-ttl`
    (ADR-5), not from evicting the daemon.
  - **Deployment:** an OPTIONAL systemd **user** unit install extra (`imgtag --install-service`,
    never required to run) with `MemoryMax=`,
    `CPUWeight=`, `Nice=10`; no root, no system-wide install, restart-on-crash, survives a
    reboot. `imgtag status` reports daemon pid, version, socket, uptime, loaded models + shas,
    warm/cold, tree-RSS — B13/B8 are measured through it.

- **Playbook addition:** exporting openai-CLIP/SigLIP-v1 towers via torch.onnx needs
  `torch.backends.mha.set_fastpath_enabled(False)` first, else export fails (bench lane,
  2026-07-22).

- **ADR-14 Moderation policy (user rulings 2026-07-22 12:50Z, VISION-ADDENDA).** Two-tier
  flags: `violation` = human nudity/explicit · real weapons · illegal drugs/paraphernalia;
  `review` = swimwear/lingerie · toy/replica weapons · tobacco/vape/smoking. Non-person
  nude figures (mannequins/statues) = no flag. Per-image schema: {category, p, tier:
  alert|violation|review|match|none} — `alert` added 13:20Z; `match` added 13:23Z
  (non-severity CONTENT label for classification tracks like sports — never counted in
  moderation totals, surfaced as its own content filter) (user safety-track directive): the
  HIGHEST tier, reserved for safety-class signals (person-down + danger context);
  sorts above violation in every count, view, and summary; counts/API/UI always report tiers separately; recall-first
  operating points per tier; enforcement_ready stays false per category until its τ is
  fitted on labeled ground truth honoring these boundaries. The measured v0 failures
  re-grade under policy: bikini/toy-rifle/vape flags were CORRECT at review tier; the
  mannequin is the true FP class to eliminate.

- **ADR-15 The track scaling law (user directive 13:26Z — TRACKS.md is the constitution).**
  Every track scores EVERY image (dense f32 sidecar per track per dataset; raw scores
  stored, tiers/labels DERIVED at read from versioned specs). Instrument hierarchy:
  embedding-matvec (default, unconditionally allowed) > distilled embedding-MLP head
  (mandated fate of every dedicated model — teacher offline, MLP in the hot path) >
  dedicated per-image model (budgeted exception: Σ dedicated FLOPs ≤30% of encoder, B25;
  distillation logged as owed). Track upgrades re-score ONE sidecar column (ms), never
  re-embed. At 100 tracks indexing/inference stay ~flat — bench-enforced (bench tracks).

## 3. Dead ends (do not rediscover)

- **fp16 as a COMPUTE format on ORT CPU EP — dead** (no native fp16 kernels; casts to
  fp32; speed samples contended→unproven but the mechanism is documented). **DISTINCT and
  ALIVE: fp16 as WEIGHT STORAGE** — measured bit-equivalent to fp32 for retrieval (cos
  0.9999986 mean, min 0.9999968; recall identical on all 14 classes) at HALF the disk
  (186MB vs 372MB SigLIP2 vision). Pending: RSS-resident probe + quiet-window speed (if
  ORT converts weights once at session init, RSS may NOT halve — measure, never assume).
  Loader law: SigLIP2 official fp16 export needs graph_optimization_level=ENABLE_EXTENDED
  (ENABLE_ALL crashes, SimplifiedLayerNormFusion bug, ORT 1.27 — one deterministic line).
- **Batch>1 for CLIP-class vision on CPU** — batch=1/2 streaming is the DEFAULT, justified
  by the contention-immune half of the evidence: batching ~doubles peak RSS on both measured
  families (pecore int8 b1 188MB → b8 412MB) for no demonstrated throughput gain. The
  "batching HURTS throughput" claim is DOWNGRADED to plausible-unproven — every img/s
  sample behind it was taken on a contended box (load 23–131; see field log 11:35Z) and
  was retracted by its own measurer. The quiet-window bench decides the throughput half;
  RSS alone already forbids batch-32 on the 8GB target.
- ggml/clip.cpp path — dormant, slower on CPU than ORT MLAS, quant regresses on x86.
- ORT CoreML EP — partition thrash (14 round-trips on Pad-reflect), silent fp16, minutes
  of recompile without ModelCacheDirectory. Even the accel lane should use coremltools.
- OpenVision **2** — no text encoder, cannot do text→image. Version number is a trap; v1 only.
- Apple FastVLM — VLM, no shared embedding space. Out of scope.
- RAM++ as hot-path tagger — 3.01GB ckpt (`14m` in filename = 14M images, NOT params),
  ~104 GFLOPs. Offline oracle use only, maybe never.
- ImageNet-val / Kaggle datasets — auth-gated, excluded by requirement.
- ANN indexes at ≤100k (hnswlib/usearch/faiss-HNSW) — seconds-to-minutes build for
  negative latency benefit; hnswlib returns 67% recall where exact is 1.0.
- Naive numpy int8 scan — 6–22× slower than f32 BLAS depending on shape/lane (both numbers
  measured: runtime 1.60 vs 0.25ms, priorart 22×). Either way, never do narrow-dtype matmul
  in numpy; int8 scan needs SimSIMD-class kernels or don't bother.
- **fp16 ONNX on the ORT CPU EP** — does not load at all (`SimplifiedLayerNormFusion` throws);
  fp16 ONNX files are a GPU/WebGPU artifact. Never ship a codepath that expects them.
- Loading **faiss and usearch in one process** — one unreproduced exit 139 under memory
  pressure (runtime R10). If you ever must, isolate them in separate processes.
- urllib on this machine's framework Python — SSL-dead. curl/certifi only.
- YOLO-World/YOLOE (GPL/AGPL) and OWLv2-class open-vocab detectors as the core — license
  and/or latency; detection is not needed for the search product.

## 4. Playbooks (symptom-keyed)

- **`SSL: CERTIFICATE_VERIFY_FAILED` from any python fetch** → you are outside the venv or
  certifi is missing. Fix: `uv run` inside the project (venv bundles certifi) or fetch via
  `curl -fL --retry 3`. Never `sudo`, never Install-Certificates.command (user's machine).
- **HEAD request returns 403/405 on a dataset URL that "should work"** → signed-URL hosts
  (Caltech) reject HEAD. Use `curl -sL -r 0-0` (1-byte range GET) to verify, plain GET to
  download.
- **PE-Core ONNX export fails / graph contains unsupported ops** → timebox 1 day, then:
  bench proceeds with SigLIP2-base int8 (official onnx-community export, verified to
  exist) as quality anchor; PE-Core moves to §8 field log with the exact error; revisit
  post-MVP. Do NOT hand-write graph surgery in bench phase.
- **int8 model slower than fp32 in your measurement** → expected possibility (see C2).
  Record both, ship the faster **that passes the fidelity gate (ADR-4)** — a faster model with
  <0.90 NN agreement is not a candidate. On x86 without VNNI, expect fp32 to win and treat that
  as the normal case, not an anomaly; do not "fix" it mid-bench, log it.
- **Indexing is slower than the bench and `draft()` "isn't working"** → `draft()` is a libjpeg
  DCT feature, **JPEG-only** — a no-op for PNG/WebP/HEIC, so a non-JPEG corpus silently decodes
  at the naive rate. Check the format mix (`imgtag stats --formats`) and always report the
  format mix beside the number.
- **Some images never appear in results** → read the skip ledger: unreadable, truncated,
  unsupported format (HEIC without the `imgtag[heic]` extra), 0-byte, CMYK, or over
  `--max-pixels` (decompression-bomb guard). Every skip is counted, reasoned, and visible in
  `imgtag status`; a silent skip is a bug (B21, CORPUS-D is the fixture).
- **Disk fills during indexing** → the writer fsyncs-then-publishes, so a failed append leaves
  rows past `count` that readers already ignore. Fail the job naming the remaining bytes; never
  leave a manifest claiming rows that aren't durable.
- **Source images moved or deleted after indexing** → hits carry stale paths; the API returns
  `exists: false` per hit rather than 404-ing the whole result set; `imgtag verify <dataset>`
  re-stats and tombstones. Ids are content-addressed, so a moved file keeps its id.
- **The same file is present twice** → id = xxhash64(bytes), so duplicates collapse to one row
  by construction; every path is kept in the ids record.
- **Indexed count < images processed** → two writers raced the manifest. Only the flock holder
  (coordinator) may write it (ADR-6); workers report row counts, they never `os.replace`.
- **Address already in use / daemon won't start** → stale socket or a second instance.
  `imgtag status` names the holder pid from `daemon.json`; only the flock holder may unlink the
  socket, and we NEVER fall back to a random port (ADR-13).
- **The server got slower and it wasn't CPU** → page cache or swap. `nice`/`ionice` do not cover
  either (`ionice` is a no-op on most NVMe multiqueue setups). Check `free -m` before/after and
  `vmstat si/so`; the `posix_fadvise(DONTNEED)` path and the RSS cap are the mechanisms (B15).
- **Throughput far below what the core count promises on the server** → you are inside a CPU
  quota. Check `sched_getaffinity` and cgroup `cpu.max`, re-run `imgtag doctor`; expect
  throttling, not parallelism (ADR-11 effective cores).
- **Precision/FP looks great but recall dropped** → you hard-ANDed the tags. Check the fusion
  mode (default is dense-recall-first, tag-boosted; `--strict` is opt-in). B5/B6 and B7 must be
  read together, never one alone.
- **Embeddings don't match reference / quality bench suddenly drops** → run the parity
  gate: `uv run imgtag bench parity`. Prime suspects: resize interpolation (must match
  the model's OWN preprocessor_config — e.g. SigLIP2 mandates resample=2 = BILINEAR,
  measured 2026-07-22; never assume bicubic), normalization constants, RGB/BGR, draft()
  decode scale, missing L2-normalize (MobileCLIP2 exports are UNNORMALIZED — assert
  ‖v‖≈1 in tests). The config file wins over folklore, always.
- **"Decode is the engine" vs "model is the engine" — BOTH are true, corpus-scoped**
  (measured 2026-07-22): on 640×480 COCO, int8 inference (33.5ms) is 9× decode (3.7ms);
  on 12MP photos, decode (163–287ms) dwarfs inference. The pipeline must profile per-
  dataset (`bench index --profile` reports the split) and publish corpus-labeled claims
  only. A 12MP real-photo decode case is a REQUIRED bench-suite member.
- **"vehicle" recall looks great but FP rate exploded** → threshold calibration drifted;
  re-run `bench quality --negatives`; check LVIS scorer honors `neg_category_ids` (I2).
- **Search returns results from a different model than the index** → manifest model-hash
  mismatch must refuse loudly. If you see garbage similarity scores, check manifest
  `model_sha` vs loaded model FIRST.
- **Indexing throughput collapses on a real photo folder** → check decode path: are you
  full-decoding 12MP frames? `draft()` scale active? EXIF orientation handled? Run
  `bench index --profile` (per-stage timings are built into the pipeline).
- **RSS grows across a long run** → mmap + arena behavior. `bench soak` isolates: decode
  pool vs ORT arena vs shard writer. ORT: set `arena_extend_strategy=kSameAsRequested`
  (immich-proven), cap decode pool queue depth (backpressure).
- **App search feels slow but bench says 50ms** → you're measuring cold start or the
  daemon isn't resident; `imgtag status` shows daemon state + model warmth. B4 measures
  keystroke→painted, not just engine time.
- **HuggingFace download stalls/fails mid-model** → resume with the same command (fetcher
  uses ranged requests + sha256 verify); offline → clear error naming the file + expected
  hash, never a partial-model run.

## 5. Risk register (branch table: likelihood · blast · detection · response)

| Risk | L | Blast | Detection | Pre-approved response |
|---|---|---|---|---|
| PE-Core export fails | M | bench slot | export spike errors | fallback SigLIP2 anchor (playbook) |
| All int8 paths regress (M3 or AVX2-no-VNNI) | M | miss B1 stretch | bench matrix | ship **fp32** (fp16 ONNX will NOT load on the CPU EP — dead end); recover size via weight-only int8 on whichever ops do win, or drop to a smaller model (PE-Core-T16) |
| Uncalibrated tags gate results | M | B6/B7 + the honest-no-match claim | quality bench per-tag report | only calibrated tags may gate; asserted in code (ADR-3 tiers) |
| B13 cold start blown on a contended-disk server | M | first-impression latency | `bench coldstart` cold-disk run | pre-warm on daemon start; cache the ORT-optimized graph (`optimized_model_filepath`); B13 stays split cold vs warm, cold is the quoted number |
| B1/B2 unreachable on the primary target under B15 politeness | H | headline claims, the definition of "done" | first target-host bench | publish per-profile budgets (PROXY dev vs 🐧 polite target); escalate to the user with the honest two-column table; NEVER loosen a budget or quietly report the proxy as the product number |
| Decode parallelism doesn't scale to 16 cores (GIL/IO) | M | B1/B2 | `bench index --profile` per-stage | process-pool decode workers (not threads); pre-sized shared buffers |
| Quality budgets fail for every Apache model (B5/B6) | L | B5–B7 | `bench quality` | escalate to user with honest table incl. MobileCLIP2 ceiling; do NOT silently relax budgets |
| Search-while-indexing race (manifest swap mid-read) | L | wrong coverage count | `bench concurrent` asserts snapshot isolation | manifest read = single open+parse of one path; never re-stat mid-query |
| 10k thumbnails bloat disk (B8) | M | B8 | `bench resources` | on-demand draft-decode thumbs + LRU cap (≤200MB); no eager thumb farm |
| App jank at 10k images (B14) | M | UX | devtools trace | virtualized grid mandatory; no full-DOM gallery |
| Old-machine claims challenged | M | credibility | — | ⌂ numbers always labeled "projected, NOT live-verified" until real hardware run |
| A lane half-succeeds and lies (looks done, isn't) | M | compounding | orchestrator spot-verification | every builder claim re-verified by independent run (fable credo #13; F-verify) |
| Darwin loop optimizes a metric by breaking another | M | regression | full bench suite each round | DARWIN.md logs per-round FULL budget table; any red = revert round |
| Bench numbers are noise from machine load (measured: thread sweep swung 3× at load 47) | H | wrong optimization decisions | record `os.getloadavg()` with every bench row | **bench refuses to run (or marks rows UNRELIABLE) when 1-min load > cores×0.6; darwin loops gate on this** |
| Zero x86 validation exists (runtime lane R6) while primary target IS x86 | H | wrong quant/thread defaults shipped | — | `imgtag doctor` autotunes on the real machine (ADR-10d); offer user: run research/bench_scripts/ + bench on the actual Linux server (one command) — the single highest-value validation available |

## 6. Invariants & verification recipes

- Project docs stay coherent: **VISION.md is sealed** (sha256 `9240e8b1…0200`, verified
  matching on disk 2026-07-22); later user constraints append VERBATIM to VISION-ADDENDA.md;
  BUDGETS/IA/ORACLE/UNKNOWNS/SWARM are updated in the same pass as any design change.
  (Distinct from the Creations registry's identically-named three-surface invariant.)
- Every embedding row L2-normalized: unit test asserts mean ‖v‖ ∈ [0.999, 1.001].
- **Line i of `ids-*.jsonl` ↔ row i of its shard** — load-bearing for every result's identity;
  each line is self-describing (`{"row": <global_idx>, "id":…, "path":…}`) and `row == expected`
  is asserted while loading a snapshot. One integer per image, and it is what silently breaks
  in every writer/compaction bug.
- Index manifest always names {model_id, model_sha, dim, count, shards[]}; loader refuses
  mismatches loudly.
- `uv run imgtag bench all` = the full budget table; exits nonzero on any budget red.
  This is the project's reconcile.py-equivalent: run before every "done" claim.
- Quality metrics computed only against downloaded ground truth (never eyeballed): COCO
  exhaustive 80 classes = FP source of truth; LVIS federated protocol for depth; OI's
  600-class **hierarchy** as a taxonomy-only cross-check — no OI imagery is downloaded, and
  caltech101 is NOT part of the bench (empty on disk).
- No dataset bytes in git (data/ ignored); no Unsplash redistribution; no apple-amlr
  weights in defaults or published artifacts.
- Every spawned agent: explicit non-Fable model + effort; MODEL line first in reports.
- `git add -A` is FORBIDDEN in this repo — explicit paths only (784MB lesson, field log 11:48Z).
- Bare `git stash` is FORBIDDEN in the shared worktree — it sweeps EVERY lane's uncommitted
  work (13:38Z incident). Bisect by copying your files aside or `git stash push -- <your files>` only.
- Nothing leaves the machine without fire17's explicit confirmation (registry law #2).

## 7. Escalation contract (for every worker on this project)

STOP and report (symptom + what you tried + which oracle entries you consulted) when:
(a) a budget test fails twice for the same cause; (b) reality contradicts an ADR or a
research number you were handed; (c) you are about to add a RUNTIME dependency outside ADR-7 or
any GPL/AGPL/amlr-licensed code (dev-deps for the bench harness — playwright, psutil-free
sampling — are allowed and marked dev-only); (d) you are about to relax a budget, delete
user data, or write outside the allowed surfaces: the ImgTag tree, ~/.imgtag,
~/.claude/skills/imgtag* (the vision-mandated skill install), and the session scratchpad;
(e) an export/quantization fight exceeds its timebox; (f) you catch yourself guessing a
number you could measure; **(g) you are about to report a PROXY number without its PROXY
label, or to compare a dev-machine number against a target-profile (🐧) budget.**
"I stopped because X" is a success report. Silent grinding is the only failure.

## 8. Field log (append-only: symptom → what worked; timestamps are UTC, always suffixed Z)

- 2026-07-22 11:25Z · **SigLIP2's OFFICIAL onnx-community int8 VISION export FAILS parity**
  (spike-siglip2): mean cos 0.7846, p05 0.65 vs its own fp32 (healthy ≥0.99), COCO "car"
  recall@5 0.40 int8 vs 0.80 fp32, every GT rank worse-or-equal. Second broken OFFICIAL int8
  after `Xenova/mobileclip_s0` — downloaded int8 vision towers now fail as a CLASS, 2 for 2.
  The B24 gate caught it on day one, before a single index was built. Lesson: the fidelity gate
  pays for itself before the product exists; official ≠ audited, ever.
- 2026-07-22 11:05Z · **UNRECONCILED — ViT-B/32-class int8 vision throughput measured at
  75.5 / 113.4 / 157.9 / 181.2 img/s** by different lanes under different geometry and machine
  load (runtime §2.1 thr4-bs8, runtime §3.2 self-quantized thr4-bs4, priorart §1.3 intra_op=8
  bs8, runtime §6.3 e2e 12×1). A 2.4× spread on the most decision-relevant number in the
  project — and B1 sits inside the band. **No published claim may cite any of them**; B1 is
  decided by ONE controlled sweep under the load-validity gate, reported with its full tuple
  (corpus, geometry, precision, mode, loadavg).
- 2026-07-22 10:15Z · Framework-Python SSL failure killed Unsplash pull → curl-based
  fetcher (scripts/fetch_unsplash_demo.sh); SIGPIPE from `tail|head` under pipefail →
  single-awk restructure. Both patterns now in playbooks.
- 2026-07-22 10:09Z · Cross-lane license conflict on PE-Core (NC vs Apache) → resolved by
  fetching the repo + HF card live: LICENSE.PE = Apache-2.0; FAIR-NC is PLM's. Lesson:
  two lanes disagreeing = fetch the primary source, never vote.
- 2026-07-22 10:31Z · Two research reports arrived under generic agent id (name lost in
  routing) → content cross-verified against disk artifacts before adoption. Lesson: trust
  content provenance (verifiable claims + URLs), not sender labels.
- 2026-07-22 10:20Z · rclip README's 119 img/s conflicts with its own PR #249 (~180 img/s
  CoreML) — README is stale, PR is measured. Lesson: PRs > READMEs for numbers.
- 2026-07-22 10:45Z · User constraint mid-mission: primary deploy = shared Linux x86 8GB
  no-GPU server, co-tenants must not slow. → ADR-10, B8/B15 tightened, Wave A briefs
  amended by broadcast. Lesson: the harbor moved; the map moved with it same-pass.
- 2026-07-22 10:50Z · runtime lane's completion message + report were delayed ~20min in
  routing (report on disk 10:29Z, message landed 10:49Z); orchestrator had declared the
  lane complete-via-cross-checks in the gap. Lesson: disk truth led message truth by 20
  minutes — check the file BEFORE pinging, and re-check before declaring a lane missing.

- 2026-07-22 12:52Z · Truncated-fetch incident closed: openclip/siglip-base fp32 towers
  were truncated by the parallel fetch storm; SHAs recorded the truncated files as
  baseline; INVALID_PROTOBUF was the symptom; re-fetch produced full files (sizes match
  params×4B arithmetic) verified by ORT load+forward. LAW: fetch validation must include
  expected-size-or-load-test — ">1MB and binary" passes truncated files. SHA256SUMS.fp32-
  verified is the new baseline in both dirs.
- 2026-07-22 13:38Z · track-nudity ran bare `git stash` in the shared worktree — swept 15
  files across lanes (~4 min blast). Its recovery was exemplary: restored the 9 whose
  worktree copies were clean, REFUSED to overwrite the 3 re-modified since (indexer.py,
  drugs.py, uv.lock — live edits), preserved stash@{0} as sole copy, reported immediately
  with exact recovery commands. New invariant added (§6). Silver lining: the bisect
  exposed that test_meta_moderation hard-codes the 3-category world — the 100-track
  scaling law arriving in test form.
- 2026-07-22 13:30Z · 🚨 RULE-4 TRIPWIRE FIRED: track-sports' MODEL line read
  claude-fable-5 DESPITE an explicit model:"opus" at spawn — the harness has drifted
  model overrides all session (haiku→opus4.8, opus→opus4.6, opus→FABLE). Agent stopped
  within one message of detection; respawned as track-sports2. LAW REINFORCED: the
  MODEL-line-first tripwire is the only reliable guard — explicit spawn params are NOT
  sufficient; every first report gets checked, every violation gets an immediate stop.
- 2026-07-22 12:25Z · b-daemon falsified ADR-3's z-score free-text calibration with a
  15-real-vs-15-nonsense probe (nonsense max-z median 4.16 > real 3.81) and shipped the
  fix behind a selector without touching the ADR — escalation contract §7(b) honored
  again. Also: idle daemon measured 51.9MB; named-tag search 0.2–1.8ms with the text
  tower never loaded — the tag-table-first resident-set law is paying exactly as designed.
- 2026-07-22 11:48Z · 784MB of model weights found COMMITTED (conductor's own `git add
  -A` at bc1f70f swept freshly-downloaded files before the subdir gitignore existed;
  caught by spike-siglip2's hygiene check). Untracked same hour; small json/config files
  retained deliberately. **OPEN DEBT: `git filter-repo` history rewrite (611MB .git) —
  MUST run when all lanes quiesce and BEFORE any publish** (publish is a rule-2
  confirmation event anyway). Standing law: `git add -A` is FORBIDDEN in this repo —
  every lane adds explicit paths (wave-b-briefs already says so; now it's an invariant).
- 2026-07-22 11:40Z · Isolated RSS probes (contention-immune): tokenizer.json 551MB
  resident, SigLIP2 int8 text 757MB, PE-Core int8 text 154MB. ADR-5 premise falsified as
  written → revised same hour (resident set = tag table + binary tokenizer; text tower
  lazy+TTL). Resident-daemon-as-designed was 1.51GB before decoding one image; tokenizer
  fix alone → ~0.97GB. Lesson: RSS is a property of the RUNTIME representation, not the
  file — 16× and 2.7× amplifications hid in "load the file".
- 2026-07-22 11:35Z · CONTENTION EVENT + honored escalation: spike-siglip2 STOPPED per
  §7(b)/(f) rather than publish img/s from a box at load 32.6/16 cores (two sibling lanes
  ~735% CPU; its matrix process was also killed externally at 19/45). ALL its img/s
  retracted incl. "clean" rounds; parity/recall/RSS/loadability stand (contention-immune).
  RULING: throughput measurement is CONSOLIDATED under b-bench as sole owner, timed runs
  only inside DECLARED QUIET WINDOWS (conductor pauses CPU-heavy lanes; loadavg gate
  refuses > cores×0.6, recorded per row). The bench-noise risk row predicted this on day
  one; the system worked — a false B1 projection never reached BUDGETS.
- 2026-07-22 12:05Z · b-engine, three graph-level traps the design docs could not know.
  (1) **HF ONNX exports put `last_hidden_state` FIRST** — `run(None, …)[0]` on SigLIP2's
  vision graph returns the [n,196,768] token grid, not the [n,768] pooled embedding, and
  nothing errors: it just indexes garbage. models.py now selects the output BY NAME
  (image_embeds/text_embeds/pooler_output) and REFUSES loudly when the best output is not
  rank-2 (siglip-base's quantized vision export exposes only last_hidden_state).
  (2) **PE-Core-B16-224's embed dim is 1024, not 512** (measured from the graph:
  `image_embeds [2,1024]`); the graph now overrides the config for `dim`.
  (3) Several fp32 artifacts fetched into `models/{openclip-vitb32,siglip-base}` are
  INVALID_PROTOBUF and their sizes disagree with the lane's own SHA256SUMS (e.g.
  openclip text_model.onnx: 120,320,255 on disk vs 117,403,648 recorded) — the
  *_quantized ones load fine. Re-fetch needed; flagged to l-logistics, not repaired here.
- 2026-07-22 12:05Z · b-engine, ADR-11 geometry measured rather than assumed: on quick500
  (500 COCO imgs, M3 Max PROXY, loadavg ~24, PE-Core-S16-384 int8, POLITE 4 workers) the
  per-worker-session geometry ran **11.31 img/s** vs **8.72 img/s** for the central-session
  geometry (1.30×; research predicted 1.7× on a quiet box). Both geometries are implemented
  behind one policy and `imgtag doctor` picks between them from its own sweep, projecting
  the worker geometry from the measured intra_op=1 row × the memory-derived worker count.
  Memory arithmetic is what gates it: a session per worker costs ~188MB (int8) so POLITE
  caps at 4 workers inside B8's 1.0GB.
- 2026-07-22 11:25Z · l-logistics spawned as haiku reported MODEL: claude-opus-4-8 —
  model-line tripwire caught a spawn-vs-actual mismatch (non-Fable either way; rule 4
  intact). All 4 model repos fetched + validated (rclip 2.1.6; uform ONNX lives in the
  uform3-image-text-english-base repo, NOT uform-vl-english — recorded for re-fetch).
- 2026-07-22 14:05Z · **Rule-4 tripwire #2: track-weapons self-reported MODEL:
  claude-fable-5** despite an explicit opus spawn (same drift class as track-sports 13:3xZ).
  TaskStop'd on receipt; its committed work (741f5a9 — OI-head verification, toys→review)
  is content-verified and STAYS. Successor track-weapons3 spawned explicit opus. Law
  reconfirmed: the MODEL first line is the only spawn-model truth; harness overrides drift.
- 2026-07-22 14:10Z · **User-reported live dupes in search** → dedupe() keyed
  (dataset,image_id) so identical content in two datasets shipped twice on the global path.
  Fix: `across_datasets=True` on multi-dataset merges — collapse by image_id, fold extras
  into `hits[].also_in`, count in collapsed_duplicates. B18(d) intact (attribution stays
  true). Test extended; verified over HTTP. Owner b-daemon notified (conductor edit while
  the lane was down).
- 2026-07-22 14:12Z · **b-bench architectural ruling accepted**: fitted ALL-tier AND-query
  precision ceiling is 0.281 even with a val-fitted τ (oracle bound) — a single global
  embedding conflates co-occurring objects; calibration cannot separate what the embedding
  merged. The 36%→70% acceptance is RETIRED as a v1 gate and moves to D1 (region/tile
  embeddings) exit criteria. v1 ships: honest no-match, rank-boost, rare/mid-tag FP-gating
  (pizza P0.93, bus P0.89, dog P0.72), recall-first fusion. Never publish a 70% AND number
  before D1.
- 2026-07-22 14:12Z · **ADR-4b addendum (B9)**: pecore-s16-384 shipping sum 182MB > B9
  150MB. The fp16-WEIGHTS vision arm (bit-equivalent, expands to fp32 at load) is
  LOAD-BEARING for shipping S16 (102→51MB ⇒ 130MB total ✅). Pending its RSS+speed bench
  (b-bench). B6 min-clause re-based: evaluate at k=min(10,N_pos) with N_pos≥25 floor —
  rare-class sparsity (toaster n=8) was measuring the dataset, not the model.
- 2026-07-22 14:20Z · **14:12Z fp16-weights clause FALSIFIED by measurement** (b-bench):
  fp16 weights load compact (293MB) and are bit-equivalent, but CPU EP has NO fp16 MatMul
  kernel — every weight Casts→fp32 per MatMul, so peak RUN RSS is 938MB vs fp32's 622MB.
  fp16-weights = disk-only win with a RAM regression; it rescues nothing into B8 and
  self-converted pecore fp16 doesn't even load (mixed-type Div). RULING: **B9 relaxed to
  ≤200MB "honest fp32-vision floor" for v1** — B9's 150 assumed a quantizable vision tower
  and B24 took that off the table (2 of 2 official int8 towers broken + int8 pairing
  silently drops 15% of weapons review flags). pecore-s16-384 fp32 (182MB) SHIPS; t16
  (~112MB) remains the strict-disk edge floor. fp16-weights arm dropped from perf matrix.
  Revisit trigger: ORT CPU EP gains fp16 MatMul kernels, or a B24-passing int8 vision
  tower appears.
- 2026-07-22 14:18Z · **RESURRECTION FOOTGUN (operational law)**: a SendMessage to a
  completed/apparently-dead teammate RESUMES it from its transcript. During the auth
  outage the conductor spawned successors (safety2/violence2/sports3) for "dead" lanes —
  then sibling lanes' contract messages resurrected the ORIGINALS (safety, violence,
  sports2, nudity), producing three duplicated lanes racing the same files. Caught via ps
  before any file collision; successors TaskStop'd, originals kept (richer context).
  LAW: before spawning a successor for a dead lane, `ps aux | grep 'agent-id <name>'`
  AND TaskStop the original name first; a "dead" lane is one message away from alive.
- 2026-07-22 16:00Z · **B17 RULING — gate re-based to the box that exists** (b-bench §7b
  escalation): the +12pt R@10 gate was measured with fp32 TEXT (850MB resident — blows B8
  on the 8GB target); shipping int8 text = R@10 74.2 = **+9.0pt over control**, and NO
  roster model can satisfy both B8 and +12 simultaneously. Ruled option (a): B17 becomes
  +9pt on the 8GB shipping config; fp32-text 77.2 stays published as the labeled
  reference ceiling. Lesson (2nd occurrence): mean-cos parity HIDES rank shifts —
  int8 text cos 0.982 yet 28% of query nearest-neighbor sets shifted; nn_agree is the
  real gate metric. New darwin item D13: static/calibrated int8 text quant (or smaller
  text projection) to close the 3pt gap. B24's nn-based clause vindicated again.
- 2026-07-22 16:00Z · **violence 16-false-alerts root cause = a UNIT bug**, not the
  intimate-pose residual: margin-space taus (≈0.05) banded a PLATT-p sidecar ([0,1]) and
  tau_alert < tau_violation even INVERTED severity. Fix: p-space ascending taus via
  per-model fitted file (recount-free re-derive); nudityprobe 16/79/9 → 0/0/10; every
  confusable slice 0.00% alert. LAW REINFORCED: a tier τ must state its SCORE SPACE and
  the derive path must assert it — the sports lesson's twin, now covered from both sides.
- 2026-07-22 14:52Z · **git add -A sweeps recurred 3× in one afternoon** under parallel
  committers (45c10f6 hijacked sports2's stage; dd906c9 [b-bench] swept nudity3's test;
  755f8e9 [b-corpus] swept b-engine's indexer/store edits). All benign THIS time (content
  verified in HEAD each time) but the failure mode is committing a sibling's half-finished
  edit under your name. Invariant re-affirmed + offenders notified: `git add <explicit
  paths>` + `git commit -- <paths>`, never -A, never -a. Also: quiet-window postmortem —
  bench's perf pass NEVER fired (GO ping sat batched in its inbox; the 17:43 results file
  was cached-report regeneration). Ruling: proxy img/s stays ADVISORY, B1 locks on the 🐧
  server per LOCK LAW; future windows require the "WINDOW OPEN" ACK handshake before the
  hold counts as spent.

---

## The final chaser (personal, from the planning mind — 2026-07-22)

To whoever builds this: three places the danger actually sleeps, none of them where the
checklists point. First — the decode pipeline. Everyone's instinct (mine included) is to
obsess over the model; but the measured story of this entire field is that the model was
never the bottleneck, plumbing was. If IMGTAG ends up slow, I am nearly certain you will
find a full-resolution decode, a single-threaded queue, or a Python loop where a batch
should be — look there FIRST, profile before touching the model. Second — quality
regressions from speed wins. The draft()-decode and int8 paths are seductive and the
parity gate is the only thing standing between "2× faster" and "silently 30 points worse"
(clip.cpp lived that fate for years without noticing). Run `bench parity` after every
preprocessing change, no exceptions, even the "obviously safe" ones. Third — the honest
no-match. Every incumbent returns top-K unconditionally; our calibrated threshold is the
single most breakable differentiator because every knob-turn elsewhere (new model, new
quant, new prompt ensemble) silently moves the score distribution. Re-calibrate whenever
the model or prompts change; it's one bench flag, and forgetting it converts our proudest
claim into our most embarrassing bug. One more intuition I couldn't formalize: the tag
table wants to stay SMALL and curated. The temptation will be to throw 20k LVIS-plus-
everything tags at it; precision dies in the tail of that distribution. Start with ~4k
good tags; grow only with measured precision. And a process note: the research lanes were
uncommonly good today — when a number in research/ contradicts what you measure, believe
your measurement, log the divergence, and feel zero guilt. The bench is the only oracle
above this one. — the planner, at maximum context, before the first line of engine code.
