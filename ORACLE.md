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
  never max-pool raw cosine against calibrated probabilities; free-text path calibrated
  via per-query corpus z-score → global logistic; fuse p = max(p_tag, p_text), record
  which path won (= the "why this matched" payload). (3) Near-tag rule: query inherits a
  tag's calibration only if cos(q, tag) ≥ θ_syn (fit on COCO synonym pairs); compound
  queries NEVER inherit a component tag's threshold — unit test with "my dog wearing a
  santa hat". (4) Manifest records calib_sha + calib_model_sha; tag-path search REFUSES
  loudly on mismatch (same mechanism as the model/manifest refusal). Re-calibration is
  structurally impossible to forget, not a discipline.
- **ADR-4 Pluggable model backends; Apache-2.0 default.** Bench roster (2026-07-22):
  PE-Core-S16-384 + T16-384 (Apache; export spike first), SigLIP2-base-224 (Apache,
  official ONNX — quality anchor; its official int8 files must THEMSELVES pass the B24
  fidelity gate vs their fp32 before being trusted — official ≠ audited, per the broken
  Xenova int8 precedent), SigLIP-v1-base (Apache, small text tower),
  UForm (Apache, Matryoshka 64d dark horse), OpenCLIP ViT-B/32 (MIT control),
  MobileCLIP2-S0/S2 (apple-amlr — REFERENCE CEILING ONLY, opt-in plugin, never default,
  never in published artifacts), rclip (system-level head-to-head baseline).
  Winner chosen by bench on: index img/s, search ms, precision@10, hypernym recall, FP
  rate, RSS, disk. Revisit if: FG-CLIP2/OpenVision v1 export cheaply and beat the set.
  **Measured refinements (runtime lane, 2026-07-22, research/runtime.md — first-party):**
  quantization recipe = SELF-quantized dynamic weight-only int8, MatMul-only,
  `MatMulConstBOnly`, QUInt8, per-tensor (measured 113 vs 40 img/s fp32, ranking
  agreement 0.96 — beats off-the-shelf HF int8 on speed AND accuracy; static/calibrated
  quant = the one untested axis, gets a bench slot). **FIDELITY GATE (CI): cos ≥0.98 AND
  top-1 NN ranking agreement ≥0.90 vs fp32** — ranking agreement is the metric that
  matters, mean cosine hides rank flips (per-channel scored 0.83 agreement at cos 0.955).
  ☠️ BLACKLIST: `Xenova/mobileclip_s0` vision int8 ONNX — numerically broken (cos 0.008
  vs its own fp32) AND 3.4× slower; MobileCLIP vision towers stay fp32 or self-quantized
  behind the gate (Ente's fp32-vision/int8-text split independently corroborates).
  Parallelism geometry is **host-probed, not fixed**: the 12 workers × 1 ORT intra-op thread
  = 181 img/s result is an **ARM PROXY** whose stated cause (ORT's intra-op pool being
  scheduled onto efficiency cores) is Apple-specific and does not transfer to x86. `imgtag
  doctor` sweeps `workers × intra_op × batch × precision` under the memory ceiling (ADR-10c);
  on ≤4-core hosts expect single-process/multi-thread to win on memory alone.
  **Default precision on any host that has not been through `doctor` is fp32** — int8 is
  enabled only by a measured win on that host (ADR-10e); the recipe above is how we quantize,
  not a promise that we will (pre-VNNI x86 losing to quantization is the normal case there,
  not an anomaly). Batch axis: export with a **dynamic** batch axis, run with a **fixed** batch
  B from the bench, and pad the final partial batch to B (discarding padded rows) — one shape
  for the entire run, no MLAS re-planning. Embeddings are stored f32 (ADR-2; f16 dropped).
  **Target-profile ranking ≠ quality ranking:** SigLIP2 is the quality ANCHOR (reference for
  how good we could be — its fp32 pair is ~1.5GB, the entire B8 budget for one model); the
  shippable default on the 8GB profile is the best model whose vision tower + resident text
  path fits the memory ceiling (PE-Core-T16/S16, SigLIP-v1's 111MB int8 text tower, UForm).
  The bench reports a quality-per-MB-of-RSS column and the default is chosen on the TARGET
  profile, not the dev box (BUDGETS precedence law: B8 hard, B17 maximized subject to it).
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
    flushes when `pending ≥ N` **or** `T` elapses with `pending > 0` (default N=500 rows,
    T=1.5s — inside B11's 2s visibility with headroom for the fsync pair; on the POLITE profile
    the writer coalesces and never fsyncs more often than once per 1.5s, because `ionice` does
    not throttle fsync and frequent fsyncs hurt co-tenants). Also flush on job end, on
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
  kSameAsRequested`, mem-pattern off; (d) **first-run autotune**: `imgtag
  doctor` runs a ~30s micro-bench on the actual deploy machine (fp32 vs int8, thread
  count, batch size — int8 winners differ per ISA) and stores the recipe in the machine
  profile; "generic and ready" means the engine adapts itself, not that we guessed;
  (e) quant decisions are PER-ARCH — NEON results never transfer to AVX2 (clip.cpp
  anomaly was x86; DeepSparse win was AVX512-VNNI; measure on target). Revisit if: the
  user later asks for Mac-local optimization (add profile, change no defaults).

- **ADR-11 Resource policy — ONE place, both throughput and politeness measured from the
  SAME run** (rev round 1, rev-arch C-4/C-5/C-6). POLITE (default): decode_workers =
  clamp(ncpu−2, 2, 8) · ORT intra_op=2, inter_op=1 · total threads ≤ ncpu−1 asserted at
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
  Pipeline geometry (central session vs per-worker sessions) is a bench-swept axis; the
  policy's totals govern either. Memory arithmetic vs B8 is re-measured, never assumed
  (spawn start-method means zero COW sharing — 16 workers ≈ 1.4GB of interpreters alone;
  hence the clamp).

- **ADR-13 Daemon lifecycle — one contract, verbatim-implementable** (rev round 1, rev-arch I-5
  + rev-oracle R2-8; ADR-12 left unassigned so review references stay stable). Unspecified,
  four builders invent four answers and the agent door (b-skill) silently pays a 2s cold start
  per call, destroying ADR-5's whole rationale.
  - **Transport: UNIX domain socket** `~/.imgtag/daemon.sock`, mode 0600, per user — never a
    TCP port on a shared box (port conflicts, multi-user collisions, and a privacy leak:
    binding TCP exposes one tenant's photo search to the others). `--http 127.0.0.1:PORT` is
    opt-in for the local app and refuses non-loopback binds. `~/.imgtag` is 0700 (B22).
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
  - **Deployment:** ship a systemd **user** unit (`imgtag --install-service`) with `MemoryMax=`,
    `CPUWeight=`, `Nice=10`; no root, no system-wide install, restart-on-crash, survives a
    reboot. `imgtag status` reports daemon pid, version, socket, uptime, loaded models + shas,
    warm/cold, tree-RSS — B13/B8 are measured through it.

## 3. Dead ends (do not rediscover)

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
number you could measure.
"I stopped because X" is a success report. Silent grinding is the only failure.

## 8. Field log (append-only: symptom → what worked; timestamps are UTC, always suffixed Z)

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
