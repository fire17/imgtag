# rev-architecture — adversarial review of the IMGTAG system architecture

> Reviewer: rev-architecture (Wave A, adversarial — job is to REFUTE, not to check).
> Date 2026-07-22. Targets: ORACLE.md ADR-1..9 + §4 playbooks, IA.md (data flow, entities),
> UNKNOWNS.md C3/I3/I4/I7, BUDGETS.md B1/B3/B8/B10/B11/B13/B15.
> Every finding = state → action → wrong outcome, plus the missing design decision
> specified tightly enough for a cold builder to implement without asking.
> Two numbers below are MEASURED on this machine (2026-07-22, uv/py3.10/numpy) and marked ⊕.
> Everything else is derived arithmetic or spec-reading, marked as such — no invented numbers.

**Counts: 12 CRITICAL · 16 IMPORTANT · 10 MINOR.**
(Round 1 = C-1..C-7 / I-1..I-8 / M-1..M-5. Round 2, after the ADR-10 brief amendment
[shared Linux x86, 8GB, co-tenants sacred] = C-8..C-12 / I-9..I-16 / M-6..M-10 — see
**§ Amendment** at the end. One round-1 fix is **partially reversed** there: see C-10.)

Verdict up front: the storage design is *directionally* right and the retrieval design is
*ambitious but underspecified*. The architecture as written cannot be built by four
independent agents without producing incompatible halves, and three of the CRITICALs
(C-1 f16 scan, C-2 double-writer, C-3 no fsync/truncate recovery) are silent-corruption or
silent-budget-miss classes — they will not announce themselves in any existing bench.

---

## CRITICAL

### C-1 · "f16 on disk, f32 accumulate" is a 20–30× search regression, and B3 dies at 100k
**Where:** ADR-2 ("exact brute-force scan over L2-normalized contiguous mmap (f16 on disk,
f32 accumulate). Measured 0.47ms @10k×512, 7.4ms @100k"), IA data flow, I3.

**Failure scenario.** State: 100k-image dataset, shards are `shard-XXXX.f16`, daemon mmaps
them. Action: user types a query; daemon does `shard.astype(np.float32) @ q` (the literal
reading of "f16 on disk, f32 accumulate"). Wrong outcome: search takes ~97ms of pure
conversion+scan instead of the 7.4ms the ADR promises — B3 (p50 ≤50ms) is red from the scan
alone, before the text encoder has been charged a single millisecond. At 10k it is 9.3ms
instead of 0.47ms: 20× the budgeted scan, quietly eating a fifth of B3.

⊕ **Measured on this machine** (numpy 2.x, M3 Max, 10k×512 / 100k×512, float32 query):

| op | 10k×512 | 100k×512 |
|---|---|---|
| f32 mmap `A @ q` (BLAS) | **0.277 ms** | **3.17 ms** |
| f16 `A16 @ q16` (numpy half loop, NO BLAS) | **13.48 ms** | — |
| f16 → f32 convert then dot | **9.29 ms** | **96.9 ms** |

The 0.47ms/7.4ms figures in ADR-2 are f32-resident numbers. numpy has no float16 BLAS path;
`np.dot` on f16 falls into the naive half-precision loop (48× slower here). The ADR silently
assumes a conversion that costs more than the thing it was optimizing.

**Fix (specify this).** Keep f16 as the *disk* format (it halves I/O and disk, which is the
real point) but the daemon MUST hold an **f32 mirror it extends incrementally, never
re-converts**:
- `Snapshot` owns `emb32: np.ndarray[(cap, D), float32]` allocated with geometric growth
  (`cap = max(4096, next_pow2(count))`).
- On every manifest bump, convert only `manifest.count - cached.count` new rows:
  `emb32[old:new] = shard_f16_view[old:new].astype(np.float32)`. At 60 img/s with a 2s flush
  that is 120 rows × 512 = 61k elements per bump — microseconds. Full re-conversion per bump
  is FORBIDDEN (at 100k that is 97ms of CPU every 2 seconds during indexing).
- Memory cost is explicit and budgeted: `count × D × 4 B` → 10k×512 = **20.5 MB**, 100k×512
  = **205 MB**. This must be added to B8's idle-RSS line (see C-4) — at 100k the f32 mirror
  alone is half the ≤400 MB idle budget.
- CLI-without-daemon (ADR-5 in-process mode) converts once at snapshot open and accepts the
  cost; document it as part of the ≤2s B13 cold path (9.3ms at 10k, 97ms at 100k — fine).
- The bench must measure the scan **from the shipped code path**, not from a synthetic f32
  array, or C-1 reappears the day the bench is written. Add an assertion in `bench search`:
  `assert index.dtype == np.float32` on the array actually handed to the matmul.

---

### C-2 · Two indexing jobs on one dataset silently permute every result, forever
**Where:** ADR-6, I4, IA "IndexJob" entity. No writer-exclusion mechanism exists anywhere in
the corpus. `flock`, "lock", "single writer" appear zero times.

**Failure scenario.** State: `imgtag index ~/Photos --dataset photos` is running (job A,
appending to `shard-0007.f16` and `ids-0007.jsonl`). Action: the user (or the app's "index"
button, or an agent calling the skill, or a watch-folder trigger from feature-idea #5) starts
job B on the same dataset — nothing rejects it. Both processes append to the same shard and
the same jsonl. Wrong outcome: the OS interleaves the two append streams. Embedding row *i*
in the shard no longer corresponds to line *i* of the ids file. **Every search from then on
returns the correct-looking similarity score attached to the wrong file path**, for the whole
dataset, permanently, with no error, no warning, and nothing in any bench that would catch it
(`bench quality` runs on a single-writer corpus). Then both jobs `tmp+rename` the manifest;
last writer wins and the loser's rows become unaccounted orphan bytes.

The blast radius is the worst in the whole design: it destroys correctness invisibly and the
"honest no-match" differentiator becomes an honest-looking lie.

**Fix (specify this).**
1. **Exclusive writer lock, whole job duration.** `~/.imgtag/datasets/<slug>/.writer.lock`,
   opened `O_CREAT|O_RDWR`, `fcntl.flock(fd, LOCK_EX|LOCK_NB)`. Acquisition failure = hard
   fail with `dataset 'photos' is being indexed by pid <p> since <iso8601> (job <id>)`, exit
   code 3. The fd is held open for the entire job; the kernel releases it on crash/SIGKILL,
   so stale locks are impossible — do NOT implement pid-liveness heuristics on top.
   Lock file *contents* (JSON: `{pid, job_id, started_at, argv}`) are advisory, for the error
   message only; the flock is the authority.
2. **Generation-scoped shard names.** `shard-<jobid8>-<seq:04d>.f16` / `ids-<jobid8>-<seq>.jsonl`.
   Even if the lock is ever bypassed, two writers cannot land in the same file. The manifest's
   `shards[]` array carries the order; row index is global across the ordered list.
3. **Readers take no lock at all** (the whole point of ADR-6) — they never touch `.writer.lock`.

---

### C-3 · "Crash recovery falls out free" is false — there is no fsync and no truncation rule
**Where:** ADR-6 ("Search-while-indexing and crash recovery fall out free"), I4.

**Failure scenario A (torn tail).** State: writer has flushed 4,800 rows; manifest says 4,800.
It appends 120 more rows to the shard and 120 lines to the jsonl. Action: the process is
SIGKILLed (OOM, user ^C^C, laptop sleep-kill) *before* the manifest rename. Wrong outcome:
the shard file is 4,920 rows long, the jsonl has 4,917 complete lines and a torn 4,918th
(no trailing newline), the manifest says 4,800. Now the user re-runs the job. If the writer
appends at EOF (the obvious implementation), row 4,920 becomes index 4,920 while the manifest
counts from 4,800 → **all subsequent rows are offset by 120 relative to their ids** — C-2's
silent permutation, reached by a plain crash. "Free recovery" only exists if truncation is
explicit, and it is specified nowhere.

**Failure scenario B (power loss / no fsync).** State: manifest renamed to say 4,920 rows.
Action: power loss 200ms later. Wrong outcome: on macOS/APFS, `rename()` is atomic w.r.t.
ordering but the *shard data* written earlier is not guaranteed durable without an fsync, and
the rename itself is not durable without a directory fsync. Post-reboot the manifest can claim
4,920 rows while the shard file physically ends at 4,800 → reads past EOF on the mmap →
**SIGBUS**, or (if the file was extended but pages not flushed) 120 rows of **zeros**, which
L2-normalize to NaN and produce NaN similarity scores that sort arbitrarily. `fsync` does not
appear anywhere in the corpus.

**Fix (specify this — the exact flush protocol, in order, no steps optional).**
Manifest per-shard record becomes `{name, rows, emb_bytes, ids_bytes}` (byte counts are the
authority, not `stat()`).

Flush sequence (writer):
1. `write()` the batch's embedding bytes to the shard (buffered `write`, **never** mmap-write
   — extending a file under an existing reader mapping risks SIGBUS on the reader).
2. `os.fsync(shard_fd)`.
3. `write()` the ids lines; each line ends `\n`; each line carries its own `"row": <global_index>`.
4. `os.fsync(ids_fd)`.
5. Write `manifest.json.tmp` with the new counts; `os.fsync(tmp_fd)`.
6. `os.rename(tmp, manifest.json)`.
7. `os.fsync(dirfd_of_dataset)` — **step 7 is the one everyone omits**; without it the rename
   can be lost and you fall back to the previous manifest, which recovery (below) handles
   correctly. With it omitted *and* recovery omitted, you get scenario B.

Recovery (runs on every dataset **open-for-write**, before the first append; never on read):
```
for shard in manifest.shards:
    if size(shard.emb) > shard.emb_bytes: os.truncate(shard.emb, shard.emb_bytes)
    if size(shard.ids) > shard.ids_bytes: os.truncate(shard.ids, shard.ids_bytes)
    if size(shard.emb) < shard.emb_bytes: FAIL LOUDLY  # torn durability; quarantine shard
assert shard.emb_bytes % (D*2) == 0 and shard.emb_bytes//(D*2) == shard.rows
log("recovered: truncated N bytes from <shard>")   # never silent
```
Readers cap every read at `manifest.rows` / `emb_bytes` and **never** call `stat()` on shard
files. Shard files present on disk but absent from `manifest.shards[]` are orphans → moved to
`trash/` at job start, never read, never deleted inline (see C-5).

---

### C-4 · The memory math does not fit B8. 16 decode workers + ORT arena ≈ 2.2 GB vs a 1.5 GB cap
**Where:** C3 ("parallel decode worker pool feeding batched inference", ORACLE risk row
"process-pool decode workers (not threads)"), B8 (peak RSS indexing ≤1.5 GB, idle ≤400 MB),
BUDGETS dev target "16 cores".

**Failure scenario.** State: default config on the dev target = one worker per core = 16
process-pool workers, batch 32, PE-Core-S16-**384** (per ADR-4). Action: run `bench resources`
on a folder of 12 MP JPEGs. Wrong outcome: B8 red by ~50%, and on the ⌂ edge target (8 GB,
4 cores) the same defaults page or OOM.

**The arithmetic** (derived; per-item figures are conservative estimates, the structure is the
point — the builder must re-measure, but must not ship the 16-worker default unmeasured):

| Component | Per unit | ×N | Subtotal |
|---|---|---|---|
| Worker interpreter + numpy + Pillow, **spawn** start method | ~60–90 MB | ×16 | **0.96–1.44 GB** |
| Draft-decoded JPEG buffer in flight (12 MP → 1/8 scale, 500×375×3) | 0.56 MB | ×16 | 9 MB |
| **Non-JPEG** (PNG/HEIC/TIFF) full decode — `draft()` is JPEG-only | 36 MB | ×16 | **0.58 GB** |
| Preprocessed f32 tensors in the queue (384²×3×4B) | 1.77 MB | ×32×(4 batches) | **0.23 GB** |
| ORT weights, ViT-S/16 fp32 | — | — | ~0.09 GB |
| ORT arena, batch-32 ViT-S/16@384 (578 tokens; attention 578²×heads) | — | — | **0.3–0.8 GB** |
| mmap shards touched, 10k×512 f16 | — | — | 0.01 GB |
| **Total, JPEG happy path** | | | **≈ 1.6–2.5 GB** |
| **Total, PNG/HEIC path** | | | **≈ 2.2–3.1 GB** |

⊕ **Measured on this machine:** `multiprocessing.get_start_method()` → **`spawn`** (macOS
default since 3.8). This is load-bearing: spawn means every worker re-imports numpy and Pillow
into its own address space with **zero copy-on-write sharing** — the 16× multiplier on the
first row is real, not pessimism. A fork-based mental model (which the design implicitly
assumes) would have been ~1/5 of it.

**Fixes (all three are needed).**
1. **Ship uint8 across the boundary, not f32 — fuse normalization into the ONNX graph** (the
   Ente trick already recorded in `research/measured-numbers.md` §"Ente production stack" and
   §"Design implications" #3). Consequences: the queued tensor becomes 384×384×3 **uint8 =
   442 KB** (4× smaller), and workers no longer need numpy at all (Pillow → `img.tobytes()`),
   dropping per-worker RSS to roughly interpreter+Pillow. This single change removes ~0.4 GB
   of workers and 0.17 GB of queue.
2. **Default worker count is NOT ncpu.** Specify `workers = clamp(ncpu - 2, 2, 8)` by default
   (also satisfies B15's "≥1 core free" — see C-5), `--workers N` to override, and make the
   bench sweep worker-count as a first-class axis (`research/measured-numbers.md` #4 already
   asks for "8 workers × 2 threads" geometry — the architecture must expose the knob).
3. **Bound the non-JPEG path explicitly.** Specify: for non-JPEG inputs, decode with
   `Image.thumbnail(target, reducing_gap=2.0)` (Pillow's reducing-gap path avoids
   materializing the full-res float pipeline where possible) and **cap concurrent non-JPEG
   decodes to 4** via a separate semaphore, since 16 × 36 MB is the single largest term in the
   table.
4. **B8 must be re-stated as tree-RSS.** See I-1 below — as currently written it is not even
   measurable for this architecture.

---

### C-5 · B15 (politeness, ≥1 core free) directly contradicts C3/ADR (ORT session, all cores)
**Where:** C3 "Indexing pipeline = (N decode workers) → (batch queue) → (**ORT session, all
cores**)" vs B15 "niced ≥10 by default, **≥1 core left free** unless `--full-speed`".

**Failure scenario.** State: b-engine reads C3 and configures
`SessionOptions.intra_op_num_threads = 16` plus 16 decode workers. b-bench reads B15 and
writes `bench politeness` asserting ≥1 idle core. Action: both land in the same repo. Wrong
outcome: `bench politeness` is red against b-engine's default and someone "fixes" it by
changing whichever file is easier — the two agents have built genuinely incompatible halves
from two documents that are both normative. Worse variant: b-engine keeps all-cores, and B1
(≥60 img/s) is measured in that config, while the *shipped default* reserves a core — the
published headline number is then unreproducible by any user, on the one project whose entire
thesis is "the field has no honest numbers".

**Fix (specify the single resource policy, one place, both docs point to it).**
```
POLITE mode (default):
  decode_workers   = clamp(ncpu - 2, 2, 8)
  ort intra_op     = 2      # per measured-numbers #4: per-image parallelism > intra-op
  ort inter_op     = 1
  total threads    = workers + intra_op  ->  <= ncpu - 1 asserted at startup
  os.nice(10) applied IN EACH WORKER and in the ORT-owning process, at start
FULL mode (--full-speed):
  decode_workers   = ncpu
  ort intra_op     = 2 ; no nice
```
Amend C3's sentence "ORT session, all cores" → "ORT session (thread geometry per the resource
policy; intra-op is small by design — per-image parallelism beats intra-op on many-core)".
`os.nice()` is inherited by children, but with `spawn` the child is exec'd fresh from the
parent's *current* niceness — that works; still, set it explicitly in the worker initializer
so a re-nice of the parent mid-run cannot desync.

---

### C-6 · B1 and B15 are measured by different commands on different runs → the throughput number is gameable
**Where:** B1 `bench index --n 100` vs B15 `bench politeness`. Nothing binds them to the same
process configuration.

**Failure scenario.** State: `bench index` runs the pipeline at 16 workers, un-niced (fastest
path — whoever writes the bench wants a green B1). `bench politeness` spawns a separate short
run in polite mode and asserts nice/core-reservation. Action: `bench all` runs both; both
green. Wrong outcome: the project publishes "≥60 img/s CPU-only" as its headline claim while
the default install, in polite mode with ~8 workers, does materially less — and IMGTAG's
entire differentiator per ORACLE §1 is *honest published numbers*. This is self-inflicted
credibility damage of exactly the kind the project exists to attack in others.

**Fix.** One run, two assertions. `bench index` must:
- run in **POLITE mode** (the shipped default) and report that number as B1;
- sample, during that same run, `nice` value of every process in the tree and per-core
  utilisation (`psutil`-free: `ps -o pid,ni,pcpu` sampling at 1 Hz is enough) and assert B15
  from those samples — same run, same clock;
- additionally report `B1-full` from a second `--full-speed` run, labeled as the stretch
  number and never as the headline;
- write both into the manifest/bench provenance with the resource policy that produced them
  (C5-provenance principle, already the project's own rule).

Restate B1's threshold as: **≥60 img/s in POLITE mode**; the ≥120 stretch is explicitly a
`--full-speed` number and must be labeled as such wherever it is published.

---

### C-7 · Calibrated tag thresholds max-pooled against raw cosine = confidently wrong answers
**Where:** ADR-3 ("per-tag calibrated thresholds + query-time hypernym expansion (WordNet
closure + supercategory tables, **max-pooled**)"), I3, B7, and the ORACLE chaser's own
"single most breakable differentiator".

Three distinct holes, one failure:

**(a) Calibrated on WHAT?** The only labeled data in the project is COCO/LVIS/OI. A threshold
τ_tag fitted on COCO's score distribution is applied to a user's personal photo corpus whose
score distribution is different (different image statistics, different tag priors, different
capture devices). Concrete: τ("car") fitted where 40% of images contain vehicles, applied to a
wedding-photo folder where 0.5% do → either the honest-no-match never fires (FP flood, B7's
≤2% blown in the field though green in the bench) or it always fires (empty results, the
product looks broken).

**(b) WHEN is it re-run?** The chaser says "re-calibrate whenever the model or prompts change"
— but nothing in the *architecture* enforces it. State: user upgrades from SigLIP2 to a new
default model; the calibration file on disk still belongs to the old model. Action: search.
Wrong outcome: thresholds from a different score distribution → silently wrong no-match
behaviour, with no analogue of I7's loud model/manifest refusal.

**(c) A query that is neither a tag nor near one.** "my dog wearing a santa hat". State: the
tag path finds nearest tag `dog` and scores it *calibrated-confident*; the embedding path
scores true matches at raw cosine ~0.21. Action: `max-pool`. Wrong outcome: **max() over two
different scales is meaningless** — the calibrated tag probability (0..1, well-separated)
dominates the raw cosine, so every dog photo is returned as a confident match for a query
about santa hats, *and* the honest-no-match cannot fire because the tag score is high. The
architecture's proudest feature inverts into its most embarrassing bug — precisely the
scenario the planner flagged and did not close.

**Fix (specify the full calibration contract).**
1. **Two-layer thresholds.**
   - *Model layer* (offline, once per model): per-tag Platt/logistic fit on COCO (exhaustive,
     FP source of truth) + LVIS federated (depth) → `p_tag(score)` ∈ [0,1] and τ_tag at the
     operating point that meets B7. Shipped as
     `~/.imgtag/models/<model_sha>/tags.json` alongside `tags.f16`.
   - *Dataset layer* (online, free): during indexing you already compute the full
     `images × tags` matmul. Accumulate streaming per-tag `mean`, `M2`(→std), and a p99
     reservoir over **this dataset's own images**; store in the manifest as
     `tag_stats: {tag: [mean, std, p99]}` (~8k tags × 3 floats = 96 KB — negligible).
     Effective threshold = `max(τ_tag, mean_tag + k·std_tag)` with `k` fitted on COCO.
     Zero extra inference; adapts to corpus. Recomputed on compaction; carried through
     append-only flushes as running statistics.
2. **Binding + loud refusal.** Manifest records `calib_sha` and `calib_model_sha`. The daemon
   refuses tag-path search when `calib_model_sha != loaded_model_sha` — identical mechanism
   and identical loudness to I7's index/model refusal. `bench quality` fails if the calibration
   file's `model_sha`, preprocessing-recipe hash, or prompt-ensemble hash differs from the
   engine's current values. Calibration is thereby *structurally* impossible to forget.
3. **Never max-pool raw scores — pool calibrated probabilities.**
   - tag path → `p_tag` (layer 1+2 above).
   - free-text path → `p_text = Φ((s - μ_q)/σ_q)` where `μ_q, σ_q` are the mean/std of the
     query's scores **over the whole corpus** (free: you just computed all N of them), then
     mapped through a global free-text logistic fitted on COCO-captions retrieval (B17 already
     downloads that ground truth). This makes "is 0.21 good?" answerable: 0.21 at 6σ above the
     corpus mean is an excellent match; 0.21 at the mean is noise.
   - fuse: `p = max(p_tag, p_text)` over *probabilities*, and record which path won → this is
     exactly the payload the "why this matched" feature (UNKNOWNS §4.1) needs, for free.
4. **Near-tag rule, stated numerically.** If `cos(q_emb, nearest_tag_emb) ≥ θ_syn` (fit θ_syn
   on COCO synonym pairs; expect ~0.9) treat the query as that tag and inherit its calibration;
   otherwise it is a pure free-text query and **only** the free-text path may fire. A
   compound query ("dog + santa hat") must never inherit a component tag's threshold — assert
   this with a unit test using that exact example.

---

## IMPORTANT

### I-1 · B8's RSS is unmeasurable as written for a process-pool architecture
`resource.getrusage(RUSAGE_SELF)` excludes children **by definition**; `RUSAGE_CHILDREN` only
counts *reaped* children and reports high-water marks, not concurrent peak. With 16 spawned
workers, the parent's RSS is the smallest number in the system and the naive measurement will
report ~15% of the truth — B8 green while the machine swaps.
**Fix:** define B8 as **peak tree-RSS**, sampled at ≥2 Hz across the process group:
`ps -o rss= -g <pgid>` summed, or read `task_info` per pid; take the max over the run. Subtract
nothing for shared pages (conservative is correct here). Add `bench resources --tree` and make
the number's definition part of the published methodology — this project's headline is honesty
about measurement.

### I-2 · The decode→inference transport is not specified at all — the design is not buildable as written
C3 says "(N decode workers) → (batch queue) → (ORT session)". Nothing states *how* a decoded
tensor crosses the process boundary. The default (`multiprocessing.Queue`) pickles: at 60 img/s
× 1.77 MB f32 = **106 MB/s** of serialize + pipe + deserialize (212 MB/s at the 120 img/s
stretch), through a 64 KB pipe buffer → syscall-bound, plus allocator churn in both processes.
That is a plausible 20–40% of a core spent moving bytes that never needed to move, on the one
axis (decode throughput) the whole project claims as its edge.
**Fix (specify the ring):**
- `multiprocessing.shared_memory.SharedMemory` slab of `S` slots × `384×384×3` **uint8**
  (442 KB/slot after C-4's fix). `S = 4 × batch` (default 128 slots ≈ 57 MB).
- Two small `multiprocessing.Queue`s carrying **integers and metadata only**: `free_q`
  (slot indices available to workers) and `ready_q` (`{slot, image_id, path, w, h}`).
  Backpressure is structural: a worker blocks on `free_q.get()` when the consumer is behind.
- Consumer builds the ORT input as `np.ndarray(shape, np.uint8, buffer=shm.buf, offset=slot*SZ)`
  — a view, no copy — batches `B` contiguous slots, runs, then returns the slots to `free_q`.
- **Segment naming and cleanup are mandatory** (Python's `resource_tracker` leaks
  `shared_memory` segments on abnormal exit — a known, still-open failure mode): name segments
  deterministically `imgtag-<pgid>-<n>`; on daemon/CLI start, sweep and unlink any
  `imgtag-*` segment whose pgid is not alive; always `unlink()` in a `finally`.
- If a worker dies mid-slot, its slot leaks from the free list. Track `slot→owner_pid`; on
  `Process.exitcode is not None`, reclaim its slots. Without this, a long run silently
  degrades to zero throughput and looks like a hang.

### I-3 · Search latency **during** indexing is budgeted nowhere
B3 is "warm" (idle). B11 measures *visibility* (≤2s) and "zero blocking" — not latency. State:
indexing at full rate saturates the thread pool; a search arrives. Wrong outcome: text-tower
encode queues behind decode workers and a 32-image batch; B3's p95 ≤120ms becomes 400–900ms
while both budgets stay green. The product feels broken in exactly the mode the project
advertises as its differentiator ("instantly search while the processing is still ongoing").
**Fix:** add **B3-concurrent: p95 ≤250 ms while indexing at ≥80% of B1**, measured in
`bench concurrent`. Structurally: the text tower gets its **own ORT session** with
`intra_op_num_threads=1..2` and its own thread, never sharing the image session; the daemon
serves queries on a thread that is not the flusher thread; and the batch dispatcher checks a
"query pending" flag between batches (a 32-image batch at 60 img/s is ~530 ms of head-of-line
blocking if it cannot be interleaved — with a separate session it can).

### I-4 · Compaction can unlink a shard a live reader has not opened yet
ADR-6: "compaction writes new files then swaps manifest". State: reader parses manifest v7,
then lazily opens `shard-0003.f16` when the scan reaches it. Action: compaction swaps to v8 and
unlinks the v7 shards in between. Wrong outcome: `FileNotFoundError` mid-search (POSIX protects
*already-open* fds, not future opens). The ADR's own mitigation ("never re-stat mid-query")
does not cover the lazy-open window because nothing says the open is eager.
**Fix:** `snapshot()` **opens and mmaps every shard named in the manifest before returning**;
after that, unlink is harmless. `ENOENT` during snapshot → re-read manifest once and retry;
second failure → loud error. Plus a deletion grace: compaction moves superseded shards to
`trash/` (never `unlink` inline); a sweeper removes `trash/` entries older than
`max(60s, 2× longest observed query)` at the *next* job start. `bench concurrent` must assert
this by compacting under a synthetic slow reader.

### I-5 · Daemon lifecycle is entirely unspecified — four builders will invent four answers
ADR-5 gives one sentence ("CLI talks to the daemon when present, else in-process"). Unanswered:
who starts it, what it listens on, what happens on version skew, what happens to a second
instance. Concrete wrong outcomes: (a) **stale daemon after upgrade** — daemon holds model A,
user upgrades, new CLI talks to old daemon, results computed with model A while the manifest
says model B; I7's loud refusal only fires *inside* one process, so this passes silently.
(b) **b-skill** (the agent door) shells out and gets in-process mode every call → 2s cold start
per agent invocation, which destroys the entire warm-tower rationale for the one door
"agent-native" is the differentiator for. (c) **two daemons** after a crash → 2× resident model
= B8 idle ≤400 MB blown.
**Fix (full contract, verbatim-implementable):**
- **Transport: unix domain socket** `~/.imgtag/daemon.sock`, mode `0600`. Not TCP — this kills
  port conflicts, multi-user collisions, and accidental network exposure in one decision.
- **Singleton:** daemon holds `flock(LOCK_EX)` on `~/.imgtag/daemon.lock` for its lifetime;
  lock contents `{pid, version, socket, started_at, models:{id,sha}}`.
- **Client algorithm:** (1) connect to socket; (2) on ENOENT/ECONNREFUSED → try
  `flock(LOCK_EX|LOCK_NB)`; if acquired: unlink stale socket, fork-exec daemon, release, poll
  connect ≤2 s; if not acquired: another client is starting one — poll connect ≤2 s with 25 ms
  backoff; (3) on timeout → **fall back to in-process** and print one line saying so (never
  fail the user's query). Only the flock holder may unlink the socket — this is what prevents
  a client from killing a live daemon's socket.
- **Version handshake:** first request is `hello{client_version}`; daemon replies
  `{version, model_shas}`. Mismatch → client sends `shutdown`, waits for socket removal
  (≤2 s), respawns. This is the *only* correct answer to the stale-daemon-after-upgrade case.
- **Idle policy:** default `--idle-timeout 0` (never exit) — ADR-5 exists precisely because
  immich's `model_ttl=300` is the proven anti-pattern. RAM protection comes from the CLI's
  in-process mode, not from evicting the daemon.
- **`imgtag status`** reports: daemon pid, version, socket, uptime, loaded models + shas,
  warm/cold, RSS. B13/B8 are measured through it.

### I-6 · Progress can run ahead of durability and lie after a crash (B10)
B10 wants ≤1 s staleness and ETA ±20%. With a decode pool, the tempting counter is "images
dispatched". State: 4,920 dispatched, 4,800 durable. Action: crash; user re-runs. Wrong
outcome: progress jumped backwards by 120 and the ETA model's history is poisoned; worse, a
UI that showed "4,920 searchable" contradicted the manifest for the whole run, breaking B11's
coverage banner ("searching 3,214 of 5,000").
**Fix:** exactly one authority — **progress `done` = manifest count (durable)**. In-flight is
reported as a *separate* field (`in_flight`) and rendered differently (ghosted). ETA is
computed from durable-count deltas over a rolling 10 s window. The IA's "coverage %" and the
progress bar read the same number, by construction.

### I-7 · The flush trigger's "N images OR T seconds" needs a real timer, or B11 fails at the tail
I4 says flush "~every N images or T seconds". State: 9,880 of 10,000 images done; the last 120
are slow (huge TIFFs) or the queue stalls on a bad file. Action: user searches for something in
the last batch. Wrong outcome: those images are invisible for far longer than B11's ≤2 s —
potentially forever if the job is aborted, because the count-trigger never fires and no timer
exists.
**Fix:** the flusher is its own thread with a `Condition.wait(timeout=T)` loop; it flushes when
`pending ≥ N` **or** `T` elapses with `pending > 0` (default `N=128, T=1.5s` — under B11's 2 s
with headroom for the fsync pair). Flush on job end, on SIGTERM/SIGINT (install handlers), and
on queue-drain. `bench concurrent` must include a *stalled-tail* case, not just steady state.

### I-8 · Image id = xxhash64 of file bytes: undeclared dependency + a second full read of every file
IA declares `xxhash64 of file bytes`; ADR-7's dependency list (`onnxruntime + numpy + Pillow +
certifi/httpx + micro-server`) does not include `xxhash`. Two builders → one adds the dep
(violating ADR-7 and triggering the escalation contract clause (c)), the other invents a
different id. Separately, hashing file bytes and *then* decoding reads every file twice: 10k ×
~3 MB = ~30 GB of extra I/O on the first cold run, which lands squarely on B1/B2.
**Fix:** (a) use stdlib `hashlib.blake2b(data, digest_size=8)` — no new dependency, ponytail's
stdlib rung, ~GB/s; (b) the worker reads the file bytes **once** into memory and both hashes
and decodes from that same buffer (`Image.open(io.BytesIO(buf))`); (c) state the id as an
invariant in IA: `image_id = blake2b8(file_bytes)` — content-addressed, so duplicate files
dedupe for free (which also gives feature-idea #4 a head start) and a moved file keeps its id.

---

## MINOR

### M-1 · `ids-jsonl` line *i* ↔ shard row *i* is load-bearing but never written down as an invariant
It is the thing that silently breaks in C-2, C-3, and any future compaction bug. **Fix:** add
to ORACLE §6 Invariants; make each jsonl line self-describing (`{"row": <global_idx>, "id":…,
"path":…}`) and assert `row == expected` while loading a snapshot. Cost: one integer per image.

### M-2 · Dataset slug collisions are undefined
Two folders both named `photos` → same slug → the second job either merges into the first
dataset silently or trips C-2's lock and looks broken. **Fix:** `slug = user_arg or
slugify(basename)`; on collision with a different `root_path`, append `-<blake2b6(abspath)>`;
manifest records `root_path`; results carry dataset slug + root.

### M-3 · Fixed vs dynamic ONNX batch axis is unspecified
A partial final batch against a fixed-shape export either crashes or forces ORT to re-plan for
a new shape (MLAS re-tuning cost, per-shape). **Fix:** export with a dynamic batch axis, run
with a **fixed** batch `B` from the bench, and **pad** the final partial batch to `B`,
discarding padded rows. One shape for the entire run.

### M-4 · Tag-table ownership and format are unassigned across three builders
b-engine computes it, b-bench calibrates it, b-daemon serves it. **Fix:** one owner
(b-engine), one location `~/.imgtag/models/<model_sha>/{tags.f16,tags.json}`, one schema
(`{names[], dim, model_sha, prompt_ensemble_sha, tau[], platt[], provenance{}}`). b-bench
writes calibration *into* that file; b-daemon only reads.

### M-5 · Core-lib API surface exists in prose only
"CLI/app/skill all through one core lib" (ORACLE §1, IA §"one core, three doors") with four
agents (b-engine, b-daemon, b-app, b-skill) building against zero declared signatures.
**Fix — freeze this minimal contract before Wave B:**
```python
open_snapshot(slug) -> Snapshot          # eager mmap of all shards (I-4); .count, .model_sha
search(q: str, *, dataset=None, k=20, min_p=None) -> list[Hit]
                                          # Hit: id, path, dataset, score, p, why(tags[])
index(paths, *, dataset, workers=None, full_speed=False, on_progress=None) -> JobId
job_status(job_id) -> {state, done, in_flight, total, img_s, eta_s, failures[]}
status() -> {daemon, models, datasets[], rss}
```
Daemon wire protocol = these five as JSON-lines over the unix socket, plus a
`subscribe_progress` stream emitting `job_status` at ≥1 Hz (B10). Anything not in this list is
not cross-door API.

---

## The three seams that would most certainly produce incompatible halves

1. **Resource policy** (C-5/C-6) — C3 says all-cores, B15 says leave a core; b-engine and
   b-bench read different documents and both are "correct". Fix = one POLITE/FULL policy block,
   referenced from both files.
2. **Decode→inference transport** (I-2) — literally unspecified; b-engine will pick pickled
   queues (the obvious default) and the throughput thesis quietly dies. Fix = the shm ring,
   spelled out above.
3. **Daemon lifecycle + core API** (I-5/M-5) — four doors, no contract, no socket, no version
   handshake. Fix = the unix-socket singleton contract + the five-call surface.

## What I could NOT refute (survives attack)

- **ADR-2's no-vector-DB call** — the exact-scan arithmetic holds decisively at ≤100k once the
  dtype bug (C-1) is fixed: ⊕ 3.17 ms for a 100k×512 f32 scan on this machine. No ANN index
  can justify its build cost against that.
- **ADR-6's append-only + atomic-rename shape** — the right primitive. Every CRITICAL above is
  a *missing rule around* it (locking, fsync, truncation, deletion grace), not a flaw in the
  choice.
- **ADR-5's resident-daemon rationale** — the immich `model_ttl` anti-pattern is well evidenced;
  only the lifecycle mechanics are missing, not the decision.
- **C3's "decode is the engine" thesis** — the strongest claim in the corpus and it survives
  every angle I attacked it from; the risk is entirely in the unspecified plumbing beneath it.
