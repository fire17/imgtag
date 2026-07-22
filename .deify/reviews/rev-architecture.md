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

---
---

# § AMENDMENT — round 2, under ADR-10 (shared Linux x86, 8GB, no GPU, co-tenants sacred)

> Trigger: brief amendment 2026-07-22 ~10:45Z. Grounded against `VISION-ADDENDA.md` (user
> verbatim: *"a 8gb ram (not powerful) linux server (that also has other things running and we
> cant slow down the server while we are doing both processing and infrence work)"*) and
> ORACLE ADR-10(a–e) as they exist on disk.
> New budget: **B8-linux ≤1.0 GB indexing peak / ≤1.5 GB total under load.**
> Everything below is derived arithmetic or documented platform behavior, labeled as such.
> **None of it is measured on the target** — there is no x86 Linux box in this session, and
> ADR-10a already concedes "zero x86 validation exists". Treat every number here as a
> *prediction to be falsified by `imgtag doctor` on the real server*, which is exactly the
> honesty protocol the project claims (I6/⌂).

**The headline reversal:** the round-1 constraint was *speed*; the ADR-10 constraint is
*residency*. Three round-1 conclusions change sign under it, and the single largest memory
term in the whole system turns out to be one nobody has named: **attention buffer size scales
with token-count², so input resolution — not parameter count — dominates B8-linux.**

---

## CRITICAL (round 2)

### C-8 · `fork` after the ORT session is created: deadlock, plus COW decay that silently doubles RSS
**Where:** C3 / ORACLE risk row "process-pool decode workers (not threads)". No start method,
and no *ordering*, is specified anywhere.

Round 1 measured `mp.get_start_method()` → `spawn` **on macOS**. On Linux the default is
**`fork`** (CPython ≤3.13). That single platform difference invalidates the round-1 memory
model in both directions and introduces a correctness bug that does not exist on the Mac:

**Failure scenario A (deadlock — correctness).** State: parent creates the ORT
`InferenceSession`, which spins up its intra-op thread pool. Action: parent then forks decode
workers. Wrong outcome: `fork()` copies only the calling thread; the child inherits the
address space with ORT's (and glibc malloc's) mutexes frozen in whatever state the *other*
threads left them — possibly locked by a thread that does not exist in the child. The child
deadlocks on its first allocation or first ORT touch. This is intermittent, load-dependent,
and will present as "indexing randomly hangs on the server but never on the Mac" — the worst
possible debugging shape. CPython 3.12+ emits `DeprecationWarning: This process is
multi-threaded, use of fork() may lead to deadlocks in the child`; **that warning is the
design telling you it is wrong.**

**Failure scenario B (COW decay — memory).** State: parent has loaded model weights, the ORT
arena, and (per round-1 C-1) an f32 index mirror; then forks 2–4 workers. Action: the workers
run for minutes. Wrong outcome: fork's copy-on-write sharing decays continuously because
CPython's refcounts live *inside* object headers — merely traversing a shared list unshares
its pages. Over a long run each worker's private RSS drifts upward toward a fraction of the
parent's heap. On a 64 GB Mac nobody notices; against **≤1.0 GB** it is the difference between
green and OOM, and it is invisible at t=0 (the bench's short runs will pass; the user's 10k
job at minute 30 will not — this is precisely B12's soak class, but B12 monitors the parent).

**Failure scenario C (silent behavior change on upgrade).** CPython **3.14 changes the default
start method on Linux to `forkserver`**. State: the profile and all budgets were locked under
`fork`. Action: the deploy box upgrades Python. Wrong outcome: worker startup cost, per-worker
RSS, and the inheritance semantics all change under the project's feet, with no code change
and no signal. A design that relies on a platform default it never names has an undeclared
dependency on the interpreter's minor version.

**Fix (specify all four; ordering is normative).**
1. **Pin the start method explicitly, never inherit the default:**
   `ctx = multiprocessing.get_context("forkserver")` and use `ctx` for every Process/Queue.
   `forkserver` gives fork's cheap startup without inheriting the parent's threads, is stable
   across 3.10→3.14, and is identical in behavior on Linux and macOS. Do not use bare `fork`
   (scenarios A/B) and do not use `spawn` on Linux (pays a full re-import per worker for no
   benefit over forkserver).
2. **Preload the worker's imports into the forkserver:**
   `ctx.set_forkserver_preload(["PIL.Image"])` — and, per round-1 C-4 fix #1, workers must
   **not** import numpy at all (uint8 out via `img.tobytes()`; normalization fused into the
   ONNX graph). This keeps per-worker RSS to interpreter+Pillow and makes it *stable* rather
   than decaying.
3. **Hard ordering rule, stated as an invariant:** *the worker pool is created BEFORE the ORT
   session exists in that process.* Add a runtime assert — the pool constructor records
   `ort_session is None` and raises `RuntimeError("worker pool must be started before the ORT
   session (C-8)")` otherwise. This is the kind of rule that survives only if it is executable.
4. Add to `bench soak`: **per-worker RSS sampled over the full run**, asserting drift ≤5%
   per worker (not just parent — see round-1 I-1). Scenario B is otherwise undetectable.

---

### C-9 · The memory ceiling is set by **tokens² × batch**, not by parameter count — 384-res @ batch 32 cannot fit B8-linux
**Where:** ADR-4 (bench roster ranks by params/quality; **PE-Core-S16-384 and T16-384 are the
lead candidates and are 384-resolution**), ADR-10c ("small-model bias … PE-Core-S16/T16,
SigLIP-v1, UForm gain rank vs SigLIP2's fat text tower"), B8-linux ≤1.0 GB, C3 ("batch queue").

ADR-10c reasons about model size the way the field does — parameters and text-tower weight.
That is the wrong variable for a memory ceiling. The transformer's peak *activation* memory is
dominated by the attention score matrix, which is **quadratic in token count**:

| model geometry | tokens (patch 16, +CLS) | attn scores / image (tokens²×heads×4B, 6 heads) | MLP intermediate / image (tokens×4·dim×4B, dim 384) |
|---|---|---|---|
| **@384** (PE-Core-S16-**384**) | 24×24+1 = **577** | **7.99 MB** | 3.54 MB |
| **@224** (SigLIP2-base-224 geometry) | 14×14+1 = **197** | **0.93 MB** | 1.21 MB |
| ratio | 2.93× | **8.58×** | 2.93× |

Now the two configurations, on a 4-core / 8 GB shared box (derived; ORT's arena is
allocator-dependent, so these are the *floor* of what the arena must hold, not a promise):

| term | **NAIVE** (384-res, batch 32, workers=ncpu=4, f16 shards) | **SAFE** (224-res, batch 8, workers=2, f32 shards) |
|---|---|---|
| parent: interp + numpy + Pillow + ORT lib | 100 MB | 100 MB |
| weights (image fp32 88 + text fp32 67) / (int8 ~22 + ~67) | 155 MB | 90 MB |
| **ORT arena** (attn ≥2 live buffers + MLP, ×batch) | **≈ 500–900 MB** | **≈ 60–120 MB** |
| decode workers (forkserver, Pillow-only) | 4 × 27 = 108 MB | 2 × 27 = 55 MB |
| non-JPEG full-decode worst case (uncapped / capped at 2) | 144 MB | 72 MB |
| shm ring (4×batch slots × 442 KB) | 57 MB | 14 MB |
| f32 index mirror @10k (round-1 C-1) / f32 mmap, file-backed | 20 MB anon | 20 MB **evictable** |
| **indexing peak** | **≈ 1.08–1.48 GB → B8-linux RED** | **≈ 410–470 MB → green, ~2× headroom** |

**Failure scenario.** State: the bench runs on the M3 Max (64 GB), where batch 32 @384 is the
throughput winner and B8's old 1.5 GB cap is met. Action: that recipe is written into the
manifest as the chosen recipe (C2's provenance rule) and deployed to the 8 GB server. Wrong
outcome: indexing peaks near or above 1.4 GB on a box that must also host co-tenants; the
kernel reclaims aggressively or the OOM killer fires (and per C-12 it does not fire at us).
The project's own tuning process selects the configuration that violates its own primary
constraint, because throughput is measured and residency is not.

**Fix.**
1. **Make resolution a first-class axis of ADR-4's selection, ranked by `tokens² × batch`.**
   Under ADR-10, a 224-model is worth ~8.6× its 384 sibling in arena terms — that is a larger
   effect than every other memory lever combined, and it is currently invisible in the roster.
   Concretely: **PE-Core-S16-384's 384 resolution is a liability on the primary target**;
   if PE-Core has a 224 variant it must be benched, and if not, SigLIP2-base-224 /
   SigLIP-v1-base / UForm rise on memory grounds independent of their quality scores.
   ADR-10c's "small-model bias" must be restated as **"small-*token* bias"**.
2. **Batch is budget-derived, not throughput-derived.** Specify:
   `batch = max(1, floor(arena_budget / (tokens² × heads × 4B × 2.5)))`, evaluated at startup
   from the machine profile, then clamped by the doctor's measured optimum. Cap batch ≤8 on
   the linux profile until measurement says otherwise.
3. **`bench resources` must report peak RSS per (model × batch × resolution)** as a matrix,
   and `bench all` must fail if the *chosen* recipe's measured peak exceeds B8-linux — not
   merely if some run somewhere did. Provenance must record which budget profile it was
   validated against (`dev-m3` vs `linux-8g`); an M3-validated recipe is **not** deployable.
4. Restate B8 as two profiles in BUDGETS.md rather than one tightened number, so an M3 run
   cannot ever be read as validating the server: `B8-dev` (≤1.5 GB) and **`B8-linux` (≤1.0 GB
   indexing / ≤1.5 GB total under load)**, with the linux row marked *projected until run on
   target* per the ⌂ honesty protocol.

---

### C-10 · f16-on-disk is now actively harmful — it converts evictable page cache into unevictable anonymous memory (partial reversal of round-1 C-1's fix)
**Where:** ADR-6 (`shard-XXXX.f16`), round-1 C-1's prescribed f32 mirror.

Round 1 established that f16 shards force an f32 copy (⊕ measured: f16 dot is 48× slower;
f16→f32+dot at 100k = 96.9 ms) and prescribed a resident, incrementally-extended f32 mirror.
**On an 8 GB shared box that fix is the wrong shape**, and I am flagging my own round-1
recommendation:

**Failure scenario.** State: 100k-image dataset; daemon holds a 205 MB f32 mirror as an
**anonymous** allocation. Action: the co-tenant's workload spikes and the kernel needs memory.
Wrong outcome: anonymous pages cannot be dropped — they can only be swapped (and a "not
powerful" shared server may have little or no swap). So the kernel reclaims *the co-tenant's*
page cache instead, or invokes the OOM killer. We have converted a cost the kernel could have
absorbed silently into a cost the co-tenant pays. That is a direct violation of the verbatim
constraint "we cant slow down the server".

**Fix — store shards as f32 on disk and mmap them directly; delete the mirror.**
- Disk cost is trivially affordable: 10k×512×4 = **20.5 MB**, 100k = **205 MB**. B8's
  ≤500 MB disk line at 10k is untouched.
- `np.dot` runs BLAS directly on the mmap'd f32 array (⊕ round-1 measurement: **0.277 ms**
  @10k, **3.17 ms** @100k) — zero conversion, zero mirror, zero warm-up.
- Those pages are **file-backed and clean** → under pressure the kernel drops them for free
  and re-faults from page cache on the next query. The system degrades *gracefully* instead of
  OOMing, which is precisely the co-tenant contract.
- `madvise(MADV_RANDOM)` on the mapping to suppress readahead; after each flush, the writer
  issues `posix_fadvise(POSIX_FADV_DONTNEED)` on the bytes it just fsynced so the write path
  does not squat cache it will never re-read.
- Keep f16 as an **opt-in `--compact`** for genuinely disk-bound installs, documented with its
  9.3 ms/97 ms conversion tax. It is a disk optimization, and it must stop being described as
  a memory or speed one.
- Round-1 C-1's incremental-extension rule still applies verbatim **if** `--compact` is used.
  Under the new default it is simply unnecessary — the better fix is to not need it.

---

### C-11 · `imgtag doctor` (ADR-10d) can silently invalidate the calibrated no-match on every machine it tunes
**Where:** ADR-10d ("fp32 vs int8, thread count, batch size … stores the recipe in the machine
profile"), ADR-10e ("quant decisions are PER-ARCH"), against I7 (model_sha refusal), B16
(parity), and round-1 C-7 (calibration).

This is the round-2 finding I rate highest, because it is a *new* mechanism that breaks the
project's proudest differentiator, and it fires automatically, on first run, on every install.

**Failure scenario.** State: release ships PE-Core with per-tag thresholds calibrated on
COCO/LVIS against the **fp32** artifact; the manifest and the refusal logic key on
`model_sha`. Action: `imgtag doctor` runs on the server, measures int8-dynamic as faster on
that ISA (exactly what ADR-10e says will happen and vary per-arch), and writes `quant: int8`
into the machine profile. Wrong outcome: the model_sha is **unchanged** — it is the same model,
a different artifact — so I7's loud refusal never fires, B16's parity gate is not consulted at
runtime, and the engine now scores images with a quantized encoder whose cosine distribution
is shifted relative to the one every τ_tag was fitted on. The honest-no-match threshold is
silently miscalibrated on every machine the tuner touched, in a machine-dependent direction,
and the failure is *invisible* because both the tag path and the embedding path still return
plausible-looking results. B7 (≤2% FP) is green in the lab and unknown in the field.

**Fix.**
1. **Introduce `recipe_sha` and make it the calibration key everywhere `model_sha` is used
   today.** `recipe_sha = sha256(model_sha ‖ quant ‖ preprocessing_recipe ‖ prompt_ensemble_sha
   ‖ ort_version_major)`. The manifest records it; the tag table records it; the calibration
   file records it; **the daemon refuses to serve tag-path search on mismatch, with I7's exact
   loudness.** An index built under recipe A must refuse queries under recipe B — quantization
   is a semantic change, not a deployment detail.
2. **Doctor may only select recipes that have shipped calibration + parity artifacts.**
   At release time, generate and ship calibration for *every* quant variant the tuner is
   allowed to pick (fp32 / int8-dynamic / int8-static — 3 small JSON files keyed by
   recipe_sha). Doctor's candidate set is defined as *"recipes with a calibration artifact"*;
   anything else is benchable but not selectable. If a variant has no artifact, doctor may
   report it as faster and must not choose it — printing "int8-static was 1.4× faster but has
   no calibration artifact; not selected".
3. **Doctor must run `bench parity` (B16) on its winner before committing the profile**, on
   the shipped synthetic corpus (M-8): mean cosine ≥0.99 vs the fp32 reference. A recipe that
   wins on speed and fails parity is discarded on the spot, on that machine. This makes B16 a
   *runtime* gate on the deploy box, not just a CI gate on the dev box — which is the only
   place it can catch a per-arch quantization anomaly (ADR-10e's whole point).
4. **Changing the profile invalidates existing indexes built under the old recipe.** Specify
   the behavior explicitly rather than leaving it to discovery: doctor detects existing
   datasets whose manifest `recipe_sha` differs, and either (a) refuses to change the recipe
   while indexes exist unless `--reindex` is passed, or (b) marks those datasets
   `stale: true` and search refuses loudly with a re-index instruction. Silent cross-recipe
   search is the same class of footgun I7 already forbids for models.

---

### C-12 · `nice` + `ionice` + `workers ≤ cores/2` do **not** protect co-tenants on a modern Linux server — six mechanisms, four of which are inert as specified
**Where:** ADR-10b, B15. This is the direct answer to the amendment's question — the honest
verdict is that the specified mechanics protect against the *least* likely harm and leave the
*most* likely ones wide open.

| # | Harm channel | What ADR-10b/B15 specifies | Why it is insufficient — concrete |
|---|---|---|---|
| 1 | CPU contention **within** our cgroup | `nice ≥10` | Works. This is the one that works. |
| 2 | CPU contention **across** cgroups | `nice ≥10` | **Inert.** Under systemd/cgroup-v2 the co-tenant is in a different slice; CFS arbitrates *between* slices by `cpu.weight`, and nice only orders tasks *within* a leaf cgroup. Niced to 19 in `user.slice`, we still take our slice's full weighted share from `system.slice`. |
| 3 | Disk I/O contention | `ionice` | **Usually inert.** `ionice` classes are honored by **BFQ/CFQ only**. Modern distros default NVMe/SSD to `none` or `mq-deadline`, where ionice is a no-op. Check `/sys/block/<dev>/queue/scheduler` before believing it. |
| 4 | **Page-cache eviction** — the biggest real harm | *nothing* | Indexing 10k photos streams **~30 GB** of file bytes through the page cache on an 8 GB box. That evicts the co-tenant's hot working set (their DB pages, their mmap'd files). Their p99 latency degrades for the entire run and recovers slowly afterward. No nice value, no ionice class, and no worker count affects this at all. |
| 5 | **Memory pressure / OOM** | *nothing* | With no cgroup memory limit, an imgtag spike triggers the **global** OOM killer, which selects by `oom_score` ≈ largest RSS — on a shared box that is very likely **the co-tenant's database, not us**. Our memory bug kills their service. This is the single worst outcome in the entire design and nothing currently prevents it. |
| 6 | Thread oversubscription | `workers ≤ cores/2` | Insufficient — see I-9 (BLAS/OMP each spawn `ncpu` threads *per process*) and I-10 (`os.cpu_count()` reports **host** cores inside a container). "cores/2 workers" can still mean 3×ncpu runnable threads. |

**Failure scenario (channel 4+5 together, the realistic one).** State: 8 GB server running a
co-tenant Postgres with a 2 GB hot set; admin starts `imgtag index ~/photos` (10k images) with
the specified politeness. Action: 30 GB of image reads stream through page cache; imgtag's own
RSS climbs to ~1 GB. Wrong outcome: Postgres's cached pages are evicted, its query latency
rises 10–50×, and if imgtag then spikes, the OOM killer picks Postgres as the largest RSS
process. The politeness budget was green throughout — `nice` was 10, `ionice` was set,
workers were `cores/2`. **B15 as written can pass while the exact harm the user forbade is
occurring.**

**Fix — five mechanisms, all cheap, in priority order.**
1. **`posix_fadvise(POSIX_FADV_DONTNEED)` on every image file after reading it** (and
   `POSIX_FADV_SEQUENTIAL` before). This is the highest-value line of code in the entire
   politeness story: it stops us from evicting the co-tenant's cache with bytes we will read
   exactly once. Same for shard writes after fsync (see C-10). Two `os.posix_fadvise` calls.
2. **`oom_score_adj = +500`** for the indexer process group, written at startup
   (`/proc/self/oom_score_adj`). If the box does go OOM, the kernel kills **us**, not the
   co-tenant. One line; converts the worst-case outcome from "we killed their database" to
   "our job died and says so". Pair it with a resumable job (the append-only design already
   gives this for free, once C-3's recovery exists).
3. **Self-imposed memory ceiling with a watchdog**, since we cannot assume cgroup access:
   a thread samples tree RSS (round-1 I-1's method) every 2 s; at >0.8× `B8-linux` it reduces
   batch then workers (and logs it); at 1.0× it stops the job with a clear error. **Do not use
   `RLIMIT_AS`** — it counts virtual address space and will kill us for merely mmap'ing shards.
4. **Cgroup v2 as the documented real answer** (the amendment asked; this is the answer):
   nice/ionice are best-effort *hints*; only cgroups are enforcement. `imgtag doctor` should
   detect cgroup v2 (`/sys/fs/cgroup/cgroup.controllers`) and **emit a ready-to-paste systemd
   drop-in**, applying it only behind an explicit flag:
   ```ini
   # /etc/systemd/system/imgtag.service.d/limits.conf   (imgtag doctor --print-limits)
   [Service]
   CPUWeight=20         # vs default 100 — real cross-slice protection (channel 2)
   MemoryHigh=1G        # throttle+reclaim before OOM  (channel 5)
   MemoryMax=1.5G       # hard ceiling
   IOWeight=20          # real I/O protection where ionice is inert (channel 3)
   Nice=10
   ```
   Also honor them if already present: read `memory.max` / `cpu.max` from our own cgroup and
   derive the machine profile from those, never from the host's totals (see I-10).
5. **Keep nice + ionice** — they are free and they do help in the single-cgroup case — but
   **B15's assertion must change**. Asserting "nice ≥10 and ≥1 core free" tests our
   *intentions*. Specify the test to measure the *outcome*: `bench politeness` runs a
   **co-workload probe** — a small fixed latency-sensitive benchmark process (e.g. a loop doing
   4 KB random reads + a fixed matmul, reporting p99) — first alone, then during a full-rate
   index. **B15 passes iff the probe's p99 degrades ≤25% and its page-cache hit rate degrades
   ≤10%.** ADR-10b already gestures at "a co-workload probe test"; this makes it the
   *definition* of the budget rather than an addendum to it, and it is the only formulation
   that can catch channels 3, 4, and 6.

---

## IMPORTANT (round 2)

### I-9 · BLAS/OpenMP thread explosion — `cores/2 workers` can mean 3×ncpu runnable threads
NumPy on Linux links OpenBLAS, which defaults its thread pool to the **core count, per
process**. State: 4-core server, 2 decode workers, each importing numpy, plus the parent.
Action: any numpy op in a worker. Wrong outcome: up to 3 × 4 = 12 BLAS threads plus ORT's pool
plus decode work, all contending — B15's co-workload probe tanks and B1 *also* drops (context
switching). Additionally, `OPENBLAS_NUM_THREADS` is read **at import time**, so setting it
after `import numpy` does nothing — a classic silent no-op.
**Fix:** set `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
`NUMEXPR_NUM_THREADS=1` in `os.environ` **before the first numpy import**, in both the CLI
entrypoint and the forkserver preload. Per round-1 C-4/C-8, workers should not import numpy at
all — this is the belt to that fix's braces. The parent's scan parallelism is then controlled
explicitly by one knob in the machine profile, not by three libraries guessing. Add a startup
assertion via `threadpoolctl` (dev-only) or by reading `/proc/self/status:Threads` in
`bench politeness` and asserting it is ≤ the profile's declared total.

### I-10 · `os.cpu_count()` lies inside containers — the profile must come from cgroup quota
State: imgtag runs in a Docker/k8s container with `cpu.max = 200000 100000` (2 cores) on a
64-core host. Action: any of `os.cpu_count()`, ORT's default `intra_op_num_threads`, or
OpenBLAS's default pool sizing. Wrong outcome: **64** threads scheduled onto 2 cores of quota
→ the container is throttled every period, latency becomes sawtooth, and the co-tenants on
that host see the worst behavior of all (a throttled, thrashing neighbor). ADR-10b's
"workers ≤ cores/2" computes 32 workers on a 2-core quota.
**Fix:** one helper, used everywhere, never `os.cpu_count()` directly:
```
effective_cores() = min(
   len(os.sched_getaffinity(0)),                       # cpuset/taskset
   cgroup_v2_quota(),   # /sys/fs/cgroup/cpu.max -> quota/period, or inf
   cgroup_v1_quota(),   # cfs_quota_us/cfs_period_us
)
```
The machine profile stores `effective_cores` and total RAM from `memory.max` (falling back to
`MemTotal`), and every downstream default — workers, intra_op, batch — derives from those.
`imgtag status` prints them so a wrong value is visible in one command.

### I-11 · Politeness must be **asymmetric** and applied **before** the ORT session, or it kills search latency
Two ordering/scoping bugs in one: (a) B15 says "indexer OS priority", but C3 puts indexing and
(per I-12) serving in one process — a single `os.nice(10)` therefore deprioritizes **searches
too**, so under co-tenant load B3's p95 ≤120 ms and the new B3-concurrent both blow, and the
"instantly search while indexing" claim dies on the primary target. (b) On Linux, nice is
per-task and new threads inherit the creator's value **at creation time** — so `os.nice(10)`
called *after* `InferenceSession(...)` leaves ORT's whole intra-op pool at nice 0, silently
un-policing the largest CPU consumer in the system.
**Fix:** specify the priority map and enforce it with per-thread `setpriority(PRIO_PROCESS,
tid, n)` (Linux tids from `threading.get_native_id()`):

| task | nice | rationale |
|---|---|---|
| daemon query/serve thread + text-tower session | **0** | B3/B3-concurrent are user-facing |
| flusher / dispatcher | +5 | durable-progress work, latency-tolerant |
| image-tower ORT pool | +10 | the bulk consumer |
| decode worker processes | +12 | plus `ionice -c3` where the scheduler honors it |

And the invariant: **`os.nice()` / `setpriority` is applied before any thread pool is created**
— same ordering family as C-8's rule, and worth one shared "process policy applied at startup,
before sessions and pools" module that both b-engine and b-daemon call.

### I-12 · Indexer and daemon as separate processes duplicate the model weights — +90–155 MB on the tightest target (new seam)
Nothing states whether `imgtag index` runs the engine in its own process or submits to the
daemon. b-engine and b-daemon are separate builders; the natural independent choice is "each
owns its own engine". State: daemon resident with both towers (~90–155 MB of weights) while
`imgtag index` runs its own engine with its own copy. Wrong outcome: weights are resident
twice, plus two ORT arenas, on the box with a **1.5 GB total-under-load** ceiling — 10–15% of
the entire budget burned on a duplicate that also *doubles* the co-tenant impact of C-12's
channels 4 and 5.
**Fix:** **the indexer is a mode of the daemon, not a peer process.** `imgtag index` submits a
job over the unix socket (round-1 I-5 contract) and streams progress back; the daemon owns the
single ORT session pair, the single writer lock (round-1 C-2), and the flusher. The in-process
path survives only as the no-daemon fallback for one-shot CLI use, and it must **refuse to run
if the daemon holds the dataset write lock** (which C-2's `flock` gives for free). This also
resolves round-1 I-3 cleanly: with one process owning both towers, giving the text tower its
own small session and its own nice-0 thread is a local change, not a cross-process protocol.

### I-13 · ADR-2's "revisit if >~300k" crossover is a 64 GB-Mac number applied to an 8 GB shared box
ADR-2 states a single global trigger for abandoning exact scan. Under ADR-10 the binding
constraint is not scan *time* (⊕ 3.17 ms @100k — still excellent) but scan *residency*: a
100k f32 index is 205 MB of page cache that must coexist with co-tenants on 8 GB, and per C-12
channel 4 anything we keep resident is something they lose.
**Fix:** make the trigger **profile-dependent and memory-based, not count-based**:
`switch to binary/scalar-quantized coarse pass + f32 rerank of top-1000 when
index_bytes > 0.25 × B8_total_budget`. On `linux-8g` that fires at ~375 MB f32 ≈ **180k
images** (or ~90k if you want the index at ≤10% of budget — pick one and state it); on
`dev-m3` it stays at ADR-2's ~300k. Binary quantization cuts resident bytes **32×** (512 bits
= 64 B/vector → 100k = 6.4 MB) with f32 rerank restoring precision; ADR-2 already names this
as the designed escape, so only the trigger needs fixing. Update ADR-2's revisit line rather
than leaving a number that is right for the machine we are not deploying to.

### I-14 · Doctor's 30 s budget across a quant×threads×batch grid selects the winner by noise
ADR-10d: "~30s micro-bench (fp32 vs int8, thread count, batch size)". A full grid is
3 quant × 4 thread-counts × 3 batches = **36 configs → 0.83 s each**. ORT session creation
alone is ~100–500 ms per config (a new session is required per quant *and* per thread count),
leaving ~100–300 ms of actual measurement — a handful of images, dominated by first-call
allocation and MLAS warm-up.
**Failure scenario.** Action: run doctor twice on the same idle machine. Wrong outcome: two
different winners. Two identical servers get different profiles; published numbers become
irreproducible — on the project whose thesis is *"the field has no honest numbers"*.
**Fix:**
- **Coarse-to-fine, not a grid.** Stage 1: fix `batch=8, threads=effective_cores/2`, sweep
  quant (3 sessions). Stage 2: winning quant, sweep threads `{1, 2, cores/2}` (3 sessions).
  Stage 3: winning pair, sweep batch `{1, 4, 8}` (3 sessions). **9 sessions, not 36.**
- **Per config: ≥0.5 s discarded warm-up + ≥1.5 s timed steady state**, ≥5 repeats. ~25–30 s
  total — the ADR's budget is achievable, but only for 9 configs, not 36.
- **Report confidence and break ties by memory.** If the top-2 are within 5% (overlapping CIs),
  choose the lower-memory config and record `tie_broken_by: memory`. On an 8 GB shared box that
  tie-break is the correct default, and it must be explicit rather than incidental.
- Store the winner's **predicted img/s** in the profile; the engine compares actual vs
  predicted on every real run and flags >2× deviation (this is what makes I-15/I-16
  detectable at zero cost).
- Provide `--thorough` (~5 min, full grid) for when the admin can afford it, and print the
  tune's duration + confidence so a rushed profile is visibly a rushed profile.

### I-15 · A profile keyed to the machine goes stale invisibly on VM migration, resize, or quota change
State: profile written on a host with AVX-512/VNNI, where int8-static won decisively
(the DeepSparse regime). Action: the VM is live-migrated to an AVX2-only host, or resized
2→8 vCPU, or its container quota is halved. Wrong outcome: the frozen recipe now runs a
VNNI-tuned int8 path on AVX2 (ADR-10e's exact warning — "quant decisions are PER-ARCH"), or
schedules `cores/2 = 4` workers against a 2-core quota → sustained throttling and co-tenant
harm, which is the forbidden outcome, arrived at through a *stale file*.
**Fix:** profile key = `sha256` of `{cpu_model, cpu_flags ∩ {avx2, avx512f, avx512_vnni, f16c,
fma}, effective_cores (I-10), mem_total_effective, ort_version, imgtag_version,
model_sha, recipe_sha}`. On every start, recompute the key in ~1 ms and compare. Mismatch →
**do not use the profile**: fall back to conservative safe defaults (int8-off, threads=2,
batch=4, workers=2), warn once, and suggest `imgtag doctor`. Add `--auto-retune` (default off
on shared servers — retuning unattended is itself a politeness event). `imgtag status` prints
the profile's key, its age, and whether it currently matches.

### I-16 · Tuning *while the co-tenant is busy* bakes a permanently pessimal profile
State: admin installs at 09:00; the co-tenant's nightly batch is saturating all 4 cores.
Action: doctor's micro-bench measures threads=1 and batch=1 as "best" (extra parallelism only
adds contention under load) and freezes that. Wrong outcome: at 03:00, when the box is idle,
imgtag runs single-threaded at a fraction of achievable throughput — **forever**, with no
symptom, because the profile is presumed authoritative. The inverse is equally bad: tuning on
an idle box yields a profile that is impolite the moment the co-tenant wakes.
**Fix:**
- Doctor samples system idle (`/proc/stat` deltas + `/proc/loadavg`) **before and during** the
  tune and records `tuned_under_load: {idle_pct_min, loadavg}` in the profile.
- If mean idle < 70%, doctor **refuses by default** with a clear message ("system is 55% busy;
  a profile tuned now will be pessimal when it is idle — re-run when quiet, or pass
  `--tune-anyway` to accept a profile marked low-confidence").
- Profiles carry `confidence: high|low`; low-confidence profiles are re-validated
  automatically the first time the engine observes both (a) system idle >70% and (b) measured
  throughput deviating >2× from the profile's stored prediction (I-14).
- The deeper fix is that **one static profile cannot serve a shared box**: pair it with C-12's
  adaptive worker count (sample idle every 5 s; shrink toward 1 worker when idle <25%, grow
  back with hysteresis and a ≥30 s dwell). The profile then sets the *ceiling*, and runtime
  adaptation sets the *actual* — which is the only structure that honors "we cant slow down
  the server" across a full day/night cycle.

---

## MINOR (round 2)

- **M-6 · Profile location and precedence on a multi-user server.** `~/.imgtag/` is per-user:
  the admin runs `imgtag doctor` as themselves, the service runs as `imgtag`, and the tune is
  invisible to the thing it was for. **Fix:** precedence `$IMGTAG_PROFILE > ./.imgtag/profile.json
  > ~/.imgtag/profile.json > /etc/imgtag/profile.json`; `imgtag doctor --system` writes the
  `/etc` one; `imgtag status` prints the active path. Same treatment for the model cache
  (`~/.imgtag/models/` duplicated per user = 2–3× the disk on the tightest box) → allow
  `IMGTAG_MODELS_DIR` / `/var/lib/imgtag/models`.
- **M-7 · Doctor must itself obey the politeness policy.** A 30 s max-threads micro-bench on a
  shared production box is exactly the harm we promised to avoid, executed at install time.
  **Fix:** doctor runs under POLITE by default; sweeping aggressive configs requires
  `--full-speed`; it prints a one-line consent notice ("will use ~N cores for ~30 s") and
  honors `--yes` for automation.
- **M-8 · The tune corpus must be fixed and shipped, not "whatever images are around."**
  Otherwise profiles are incomparable across machines (defeating "generic and ready") and the
  first run is contaminated by cold page cache. **Fix:** generate a deterministic synthetic set
  (fixed seed; a spread of 0.5/2/12 MP JPEGs + one PNG) into a temp dir, read it once to warm,
  then time. Ship the generator, not the images.
- **M-9 · Every emitted number must carry its profile.** ADR-10a says M3 numbers are PROXY;
  make the bench *enforce* it — every result row, JSON field, and showcase table carries
  `profile: dev-m3-proxy | linux-8g-measured`, and the publish path refuses to render a proxy
  number without the label. Round-1's honesty rules are only as strong as the field that
  carries them.
- **M-10 · `--full-speed` needs a co-tenant guard.** On the primary target, `--full-speed` is
  the one flag that can violate the founding constraint. **Fix:** when the machine profile is
  `linux-shared`, `--full-speed` prints what it will do and requires confirmation (or
  `--yes`), and is refused entirely if a cgroup memory/cpu limit is detected that it would
  exceed.

---

## What survives round 2 unrefuted

- **ADR-10's diagnosis itself** — the four consequences (a) x86/AVX2 baseline, (b)
  politeness-first, (c) structural memory ceiling, (e) per-arch quant are all correctly
  reasoned; my attacks land on the *mechanisms*, not the direction. (d) autotune is the right
  instinct (it is the only honest answer to "we have never run on the target") and needs the
  guardrails in C-11/I-14/I-15/I-16 to be trustworthy rather than merely present.
- **The append-only + atomic-rename storage shape** survives the memory constraint intact —
  and gets *better* under C-10's f32 mmap, since file-backed pages are exactly what a
  memory-pressured shared box wants.
- **Exact brute-force scan** survives at the target scale (⊕ 3.17 ms @100k); only the
  *crossover trigger* was mis-specified (I-13).
- **"Decode is the engine"** survives again and is reinforced: on a 4-core server the decode:
  inference ratio widens, so the pipeline design matters *more* than on the 16-core Mac, not
  less.

## Round-2 seams (would produce incompatible halves)

1. **Process topology** (I-12) — is the indexer a daemon mode or its own process? b-engine and
   b-daemon will answer differently and the weights get duplicated on the tightest target.
2. **Process policy module** (C-8 ordering + I-9 env + I-10 effective_cores + I-11 nice map) —
   four separate ordering-sensitive rules that must run in one place, before any pool or
   session exists. Ship it as `imgtag/_policy.py:apply_process_policy()` called first thing by
   every entrypoint, or each builder will implement a partial version.
3. **`recipe_sha`** (C-11) — b-engine writes manifests, b-bench writes calibration, b-daemon
   enforces refusal. If they do not agree on the exact hashed tuple, the refusal is either
   spurious or absent. Freeze the field list before Wave B.
