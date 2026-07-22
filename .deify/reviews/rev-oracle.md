# rev-oracle — adversarial review of ORACLE.md

> Reviewer: rev-oracle (Wave A) · 2026-07-22 · target `ORACLE.md` (14,842 B, 197 lines)
> Evidence base read in full: VISION.md, VISION-ADDENDA.md, UNKNOWNS.md, BUDGETS.md, IA.md,
> SWARM.md, research/{datasets,models,tagging,priorart,runtime,measured-numbers,
> apple-runtimes-note}.md, plus disk truth (`data/`, `models/`, `.gitignore`, git status)
> and one live external verification (HF PE-Core-S16-384 license).
> Mandate: REFUTE. Verdict counts: **7 CRITICAL · 11 IMPORTANT · 7 MINOR**.
>
> Headline: the oracle is well-built for the *research it was forged from* — but it was
> forged against **the wrong machine** and it **dropped the runtime lane's single most
> important finding** (quantization fidelity). Both are fixable in one editing pass.

---

## CRITICAL

### C-1. The oracle plans for an M3 Max. The product runs on an 8 GB shared Linux x86 server.
**Defect.** `VISION-ADDENDA.md` (2026-07-22 ~10:45Z, verbatim user constraint) states the final
system runs on *"a linux server without a gpu … 8gb ram (not powerful) … that also has other
things running and we cant slow down the server."* ORACLE.md does not mention this **anywhere**:
not in §1 context capsule, not in any ADR, not in the risk register, not in the playbooks, and
`VISION-ADDENDA.md` is not even in the "Companions:" list of the header. Meanwhile
`research/runtime.md` R6 states plainly: *"No x86 measurement exists in this report. Every x86
statement is literature or extrapolation… The 'old computers' requirement is therefore
**unvalidated**."*

**Why it matters.** Every load-bearing measured number behind ADR-1/ADR-2/ADR-4/ADR-5 was taken
on Apple ARM: ORT's ARM int8 SDOT/I8MM kernels, the efficiency-core thread-scheduling trap, the
~70 GB/s memory-bandwidth scan model, `draft()` decode on ARM libjpeg. ORT's own docs (quoted in
runtime.md §3.2) warn that **x86 without VNNI can get *worse* performance from quantization** —
i.e. the project's central speed lever may invert on the deployment target. A cold worker reading
ORACLE alone will optimize, bench, and *publish* numbers for a machine the product never runs on.
This is the single largest gap between the oracle and reality.

**Exact fix.** (a) Add `VISION-ADDENDA.md` to the header Companions line. (b) Insert a new ADR:

> - **ADR-10 Deployment target ≠ dev machine (VISION-ADDENDA 2026-07-22 10:45Z).** Production =
>   **x86-64 Linux, no GPU, ~8 GB RAM, shared with other workloads that must not be slowed.**
>   The M3 Max is a *development* box only. Consequences, binding on every lane: (1) no ARM-only
>   optimization may become a default (ARM int8 SDOT/I8MM wins, efficiency-core thread caps,
>   Apple-specific decode paths) — they are runtime-probed, not assumed; (2) every published
>   number carries its host label, and **no x86 number may be reported until measured on x86**
>   (research/runtime.md R6: zero x86 measurements exist today); (3) memory budgets are sized for
>   8 GB *shared*, not 64 GB (see B8/B15); (4) politeness (B15: nice ≥10, ≥1 core free) is a
>   **production requirement**, not a nicety — "cant slow down the server" is verbatim law;
>   (5) the bench harness must run unmodified on Linux x86 (no macOS-only calls, no `sysctl`
>   core probing without a Linux fallback). Revisit if: the user redirects the target.

(c) Add a risk row: `| Target-transfer failure (x86 lacks VNNI / int8 inverts) | M | B1,B2,B9 | first x86 bench run | fp32 fallback auto-gated by a startup micro-benchmark (runtime.md §3.3 rec 2); publish per-host numbers, never one blended figure |`
(d) Add a playbook: **`Numbers on the Linux server are 3–10× worse than the Mac numbers`** →
expected, not a bug. ARM/x86 kernel gap + memory bandwidth (~10–25 GB/s vs ~70). Re-run the
thread/precision sweep *on that host*; the engine stores the winning recipe per host in the
manifest (C2). Never carry a Mac recipe to a Linux host.

---

### C-2. ADR-2 mandates f16 shards but quotes f32 measurements — and omits the chunked cast that keeps it from being 30× slower.
**Defect.** ADR-2 says *"Exact brute-force scan over L2-normalized contiguous mmap (f16 on disk,
f32 accumulate). Measured 0.47 ms @10k×512, 7.4 ms @100k on this machine."* Those two numbers are
the **f32** rows of priorart.md §1.1. The measured f16 path is in runtime.md §5.3 rec 2: *"Store
fp16 on disk, cast to fp32 **in 8k-row chunks** … 2× smaller for ~1.9× the latency (6.9 ms vs
3.4 ms @100k)"*, and R5: *"numpy int8/fp16 matmul falls off BLAS and is **6–30× SLOWER** than
fp32 … A naive 'we'll save memory' change would be a 6× latency regression."*

**Why it matters (weakest-reader test).** A cold Haiku reads "f16 on disk, f32 accumulate" and
writes `np.memmap(..., dtype=np.float16) @ q`. That is exactly the 30×-slower path R5 warns
about; it will silently blow B3 and there is **no playbook keyed to it**. Worse, the trade is bad
on its own terms at the stated scale: at 10k×512, f16 saves **10 MB** of disk and costs ~1.9×
search latency.

**Exact fix.** Replace ADR-2's first sentence and add a playbook.

> - **ADR-2 No vector database. Exact brute-force scan** over an L2-normalized contiguous mmap.
>   **Default dtype = f32** (measured 0.47 ms @10k, 3.8–7.4 ms @100k; index is only 20.5 MB at
>   10k — the memory saving of f16 is not worth buying). `--f16-shards` is an opt-in flag for
>   RAM-constrained hosts (the 8 GB Linux target above ~200k images): it is **bit-for-bit lossless
>   for retrieval** (cos(f32, f16-stored) = 1.000000, measured) but costs ~1.9× latency and MUST
>   be implemented as **store narrow / compute wide: cast an 8k-row chunk to f32, then BLAS**.
>   Never hand numpy an f16 (or int8) matmul — it falls off BLAS and is 6–30× slower (R5).

Playbook: **`Search got much slower after an index format change`** → you are doing arithmetic in
a narrow dtype. numpy has no BLAS path for f16/int8. Cast chunk-wise to f32 first; assert the
scan kernel receives an f32 array (`assert X.dtype == np.float32` in the hot path).

---

### C-3. The runtime lane's self-declared "most important finding" — quantization fidelity — is entirely absent from the oracle, and ADR-4 endorses exactly the artifact class it warns against.
**Defect.** runtime.md §3 (titled *"the most important finding in this report"*) measured that
**downloaded int8 ONNX files damage retrieval**: `Xenova/clip-vit-base-patch32
vision_model_int8.onnx` keeps cos 0.955 but **flips 25 % of nearest-neighbour rankings**;
`Xenova/mobileclip_s0 vision_model_int8.onnx` produces **noise** (cos 0.008–0.16, ‖e‖ 28× off) and
is 3.4× *slower*; literature corroborates −4.6 pp zero-shot for full-QDQ int8. The lane's fix is
explicit: self-quantize **weight-only, per-tensor U8, MatMul only, `MatMulConstBOnly`** (measured
strictly better than the downloaded file: 113 vs 105 img/s, cos 0.985 vs 0.955, NN agreement
**0.96 vs 0.75**) plus a **CI fidelity gate: cos > 0.98 AND top-1 NN agreement > 0.90 on a fixed
100-image holdout** (R1: *"Any pipeline that grabs 'the int8 one' from HF will silently produce a
useless index"*).

ORACLE contains **none of this**. There is no quantization ADR, no dead end for the broken files,
no fidelity invariant, no budget. And ADR-4 + the PE-Core playbook both nominate
*"SigLIP2-base-224 (Apache, **official int8 ONNX** — quality anchor)"* — i.e. a downloaded int8
file is designated **the quality reference**, the one role in which a silent 25 % ranking flip is
maximally destructive (every other candidate would be scored against a corrupted anchor).

**Why it matters.** B16 (parity) covers *preprocessing* fast paths only. Nothing in the budget
table, the invariants, or the playbooks would catch a degraded quantized model. This is the
project's stated nightmare — *"invisible in a smoke test and shows up as 'why didn't it find that
car'"* — and it is the one failure class the oracle does not cover.

**Exact fix.** (a) New ADR:

> - **ADR-11 Quantization = self-quantized weight-only int8; downloaded `*_int8/_q4/_fp16.onnx`
>   are untrusted inputs.** Recipe (measured best on both speed and fidelity):
>   `quantize_dynamic(weight_type=QUInt8, per_channel=False, op_types_to_quantize=["MatMul"],
>   extra_options={"MatMulConstBOnly": True})` — weights int8, **activations stay fp32** (full QDQ
>   is where the collapse lives). Keep the fp32 graph available and auto-fall-back via a startup
>   micro-benchmark when int8 is not faster on the host (pre-VNNI x86, hybrid conv nets).
>   Revisit if: a measured recipe beats it on the fidelity gate AND speed.

(b) New invariant: *"**Fidelity gate (blocking).** Any quantized/exported model, ours or
downloaded, must pass `cos(fp32, quant) ≥ 0.98` mean AND `top-1 NN agreement ≥ 0.90` on a fixed
100-image holdout before it may index anything or appear in a bench table. Mean cosine alone is
NOT the metric — ranking agreement is."*
(c) New budget row for BUDGETS.md: `| B18 | anti-silent-quantization-loss (R1/R2) | quantized-model fidelity vs its own fp32 graph | mean cos ≥0.98 AND top-1 NN agreement ≥0.90 (100-img holdout) | uv run imgtag bench fidelity | provisional |`
(d) New dead ends: *"`Xenova/mobileclip_s0` `vision_model_int8.onnx` — embeddings are noise
(cos 0.008–0.16) and 3.4× slower. `…_q4.onnx` degraded (cos 0.67). Blacklisted by name."* and
*"Full-QDQ (activation) quantization of a retrieval encoder — −4.6 pp zero-shot, 25 % NN flips."*
(e) Amend ADR-4: SigLIP2's role becomes *"quality anchor — **fp32 graph is the reference**;
its official int8 export is a candidate like any other and must clear the fidelity gate."*

---

### C-4. "All cores" vs "12 workers × 1 thread": the oracle never adjudicates a 1.7–3× contradiction inside its own corpus.
**Defect.** UNKNOWNS C3 (the resolution a builder will implement) says the pipeline is
*"(N decode workers) → (batch queue) → (**ORT session, all cores**)"*, and priorart.md §9 lists
*"`inter_op=1/intra_op=2` → **all-core ORT**"* as a win to take from immich. Measured reality in
runtime.md contradicts both: **R4** *"More threads is slower. ViT-B/32 int8: 55.5 img/s @4 threads
→ 17.9 @8. Do not default to `os.cpu_count()`"*; **§6.3** *"Process-level parallelism beats
intra-op threading by 1.7× … Recommended: `workers = performance_core_count`,
`intra_op_num_threads = 1`, batch 4–8 per worker, each worker appending to its own shard file."*
(1×8 = 104 img/s vs 12×1 = 181 img/s.) ORACLE's §2/§3/§4 say nothing about thread geometry at
all, and its only throughput playbook points at decode.

**Why it matters.** This is the difference between meeting B1's stretch and missing B1 outright,
and the wrong choice looks *reasonable* (more threads = faster is the universal prior). It also
interacts with C-1: the efficiency-core explanation is Apple-specific, so the *conclusion* must be
re-derived on the Linux target rather than copied.

**Exact fix.** New ADR + playbook.

> - **ADR-12 Parallelism geometry = many single-threaded worker processes, not one many-threaded
>   session.** Measured (M3 Max, ViT-B/32 w8, e2e): 1×8 = 104 img/s · 4×2 = 154 · **12×1 = 181**.
>   Default: `workers = performance-core count` (probed, Linux-safe fallback = `os.cpu_count()-1`),
>   `intra_op_num_threads = 1`, `inter_op = 1`, batch 4–8 per worker, **one shard file per worker**.
>   Python decode is GIL-bound → processes are required anyway. Never default to `os.cpu_count()`
>   threads (R4). Re-probe the geometry per host (ADR-10); store the winning geometry in the
>   manifest. Supersedes the "ORT session, all cores" phrasing in UNKNOWNS C3 — fix that line too.

Playbook: **`Throughput drops when I add threads`** → expected. ORT's intra-op pool schedules onto
efficiency/SMT cores and contends with the decode pool. Sweep `workers × intra_op` (bench does
this) and pick the measured winner; never assume monotonic.

---

### C-5. No corpus on disk can produce a decode-bound measurement, so B1's "≥60 img/s" is unfalsifiable — and the whole "first honest CPU bench" claim rides on it.
**Defect.** ORACLE §1 stakes the project on *"publishing the first honest CPU bench"* and the
final chaser says decode is where the danger sleeps. But every image available locally is small:
`data/coco/val2017` + `quick500` are COCO JPEGs (~150–330 KB, 640×480); `data/unsplash/images`
are ~130–320 KB web-sized. The 12 MP regime that produced the 287 ms / 80.9 ms decode numbers
**does not exist in the corpus**, and B1/B2 state a threshold (`≥60 img/s`, `≤3 min per 10k`)
with **no corpus, resolution, or format qualifier**. A worker who benches on COCO will report a
big number that is honest about nothing a user with a real photo library cares about — the exact
sin the project exists to correct. (`data/caltech101/` is also empty on disk, so any plan that
assumed a third image corpus is already wrong.)

**Why it matters.** It converts the differentiator into a liability: the first outsider who runs
IMGTAG on 12 MP phone photos gets a number 5–20× worse than the published one. Credibility loss
here is unrecoverable and is precisely what the oracle's own risk row "Old-machine claims
challenged" tries to prevent for hardware, while leaving the corpus dimension unguarded.

**Exact fix.** (a) Invariant: *"**Every throughput number is reported as (host, corpus, median
image megapixels, format mix, threads/workers, batch, precision).** A number without that tuple
may not be published, logged as a budget result, or compared across candidates."* (b) Bench must
include a **`bigphoto` corpus**: ≥200 images at ≥8 MP (upscale/re-encode COCO or fetch Unsplash
full-size under the fetch-only rule) so the decode-bound regime is measured, not theorized.
(c) Restate B1: *"≥60 img/s on the **standard corpus** (COCO val2017, ~0.3 MP) **and** ≥20 img/s
on `bigphoto` (≥8 MP) — both reported, neither alone."* (d) Add a risk row:
`| Bench corpus unrepresentative (all ≤1 MP) | H | credibility, B1/B2 | corpus tuple missing from a result | bigphoto corpus mandatory in bench all; refuse to print a headline img/s without it |`

---

### C-6. Benchmark noise is unmanaged, and the 60-minute darwin loop is designed to optimize it.
**Defect.** runtime.md R9: *"My own thread sweep swung **3× between back-to-back runs at load 47**.
Sibling agents on this machine will corrupt any benchmark. **Gate the darwin/self-improvement
loops on machine load**, or their 'improvements' will be measuring noise."* priorart §1.3 carries
the same caveat (*loadavg 19→60, treat as lower bounds*). SWARM.md schedules **14 agents**, the
user will be feel-testing concurrently, and VISION mandates a 60-min darwin loop afterwards.
ORACLE has one darwin risk row ("optimizes a metric by breaking another") and **nothing about
measurement validity**: budgets are single-run pass/fail (`bench all` exits nonzero on any red),
so a noisy run both *fails* good builds and *passes* bad ones.

**Why it matters.** Compounding: a darwin round that "wins" on noise gets kept, the next round
builds on it, and the honest-numbers thesis quietly becomes fiction. It also makes the C-5 fix
worthless (a correctly-specified number measured under load 40 is still garbage).

**Exact fix.** (a) Invariant: *"**Measurement validity gate.** Every timing result records
`loadavg1`, core count, and thermal state; each budget metric is the **median of ≥5 runs** with
the IQR reported; a run started at `loadavg1 > 0.5 × cores` is marked INVALID and not used to
pass, fail, or compare anything. `bench all` refuses to print a headline table from invalid
runs."* (b) Risk row: `| Bench numbers corrupted by concurrent agents/user load | H | every speed budget + darwin | loadavg recorded per run; IQR > 20% of median | mark INVALID, re-run when quiet; darwin loop hard-gated on loadavg before each round |` (c) Playbook:
**`Two runs of the same bench disagree by >20%`** → you are measuring the machine, not the code.
Check `loadavg`, other agents, thermal throttling; re-run quiet; never adopt a darwin round whose
delta is inside the IQR.

---

### C-7. The escalation contract forbids the deliverable. `b-skill` cannot install the globally-available skill without violating clause (d).
**Defect.** §7 clause (d): *"STOP and report … when you are about to … touch anything outside the
ImgTag tree + `~/.imgtag`."* VISION demands *"a globally available skill so agents will be able to
use this"*, and SWARM.md agent #12 is *"b-skill … global agent skill (`~/.claude/skills/imgtag`)"*.
`~/.claude/skills/` is outside both allowed roots. Same problem for the Creations vault law
(`~/Creations/Skills/`) and for `uv`'s caches.

**Why it matters (weakest-reader test).** The literal reading has exactly two outcomes, both bad:
the worker stops and burns a round on an escalation that was always sanctioned, or it decides the
contract is soft and starts ignoring clause (d) generally — which is the clause protecting the
user's machine.

**Exact fix.** Rewrite clause (d):

> (d) you are about to relax a budget, delete or overwrite user data, or write **outside the
> allowed write-set**: the ImgTag tree · `~/.imgtag/` (indexes, models, jobs) · and — for the
> vision-mandated global skill only — `~/.claude/skills/imgtag/` plus its Creations-vault sync
> (`~/Creations/Skills/`), which are **pre-approved, additive, and non-destructive** (create or
> update only; never delete or modify another skill). Anything else: STOP.

---

## IMPORTANT

### I-1. The risk register's pre-approved response "ship fp32/f16" is impossible on the CPU EP.
Row 2 (*"All int8 paths regress on M3 → ship **fp32/f16**, note"*) contradicts runtime.md R7 and
§2.1: *"**fp16 ONNX does not load on the ORT CPU EP** — `SimplifiedLayerNormFusion` throws … fp16
ONNX files are a GPU/WebGPU artifact. Don't ship a codepath that expects them."* Pre-approved
responses are executed without thinking; this one dead-ends. **Fix:** change the response to
*"ship fp32 (fp16 ONNX will not load on the CPU EP — R7); recover the size loss via weight-only
int8 on whichever ops do win (ADR-11), or drop to a smaller model (PE-Core-T16)"*, and add
"fp16 ONNX on the ORT CPU EP" to §3 Dead ends.

### I-2. ADR-6 has no story for concurrent manifest writers, but ADR-12/§6.3 mandates N writer processes.
ADR-6 says *"writers append + rename"*; the validated experiment (runtime §6.2) used **one**
writer. With `workers = 12` each owning a shard and each doing `os.replace("manifest.json")`,
updates are last-write-wins and shard counts are silently lost — a data-integrity bug that
presents as "the index says 8,400 images but 10,000 were processed" and would also corrupt B10's
ETA and B11's coverage. **Fix:** amend ADR-6: *"Exactly **one** process owns `manifest.json`
(the coordinator). Workers publish progress by writing their own `shard-<worker>.commit` file
(atomic rename, contains durable row count); the coordinator merges commits into the manifest on
a timer. Workers never write the manifest."* Add playbook: **`Indexed count < images processed`**
→ two writers raced the manifest; check that only the coordinator writes it.

### I-3. Per-tag thresholds have no label source for 90 % of the tag vocabulary.
ADR-3 mandates *"per-tag calibrated thresholds"* over *"~4–8k tags"*. tagging.md §7 states the
calibration sample *"is the expensive part … plan for ~200–500 hand-labelled images across the
top ~50 tags, or bootstrap from RAM++ offline"*. On disk we have labels for COCO's 80 classes and
LVIS's 1,203 — i.e. at best ~1,283 of 4,000–8,000 tags can ever be calibrated, and only on COCO
imagery. ORACLE offers no rule for the uncalibrated majority. **Fix:** add to ADR-3: *"Tags come
in two tiers: **calibrated** (τ_t fitted for max-F1 on COCO/LVIS ground truth — the only tags
allowed to hard-gate or to produce an honest 'no match') and **uncalibrated** (default
conservative τ from the family prior; may boost rank and explain a hit, may never gate or veto).
A tag's tier is stored in the tag table and shown in `why-this-matched`."* Add risk row:
`| Uncalibrated tags gate results | M | B6/B7 + honest-no-match claim | quality bench per-tag report | only calibrated tags may gate; assert in code |`

### I-4. "Tags are the FP gate" under-specifies the fusion default; the corpus recommends the opposite default.
tagging.md §7 is explicit: *"Recommended default: **dense-recall-first, tag-boosted** (tags raise
rank rather than hard-filter) with a user-visible 'strict mode' that flips tags to a hard AND"*,
and warns the τ boundary creates false negatives (*"a real car scoring 0.63 against τ=0.65
vanishes"*). ORACLE's ADR-3 says only *"Tags are the FP gate"*. A cold reader implements a hard
AND, B5/B6 recall collapses, and — because the FP budget B7 improves — the bench looks like a
win on one axis while breaking the vision's headline promise ("**all** of the images with cars").
**Fix:** append to ADR-3: *"**Default fusion = dense-recall-first, tag-boosted** (tags raise rank;
they do not filter). `--strict` flips calibrated tags to a hard AND. The honest-no-match path is
the ONLY place a threshold vetoes results: if no calibrated tag in the expanded query set passes
τ **and** the best dense score is below the global no-match floor, return zero results."* Add
playbook: **`Precision/FP looks great but recall dropped`** → you hard-ANDed the tags. Check the
fusion mode; B5/B6 and B7 must be read together, never one alone.

### I-5. B8 (idle RSS ≤400 MB) collides with ADR-5 (resident text tower) — worse on the 8 GB target — and B9's "primary model" is ambiguous.
models.md: SigLIP2-base text tower = **1.13 GB fp32 / 283 MB int8**; vision 94.6 MB int8. A daemon
holding a warm text tower + a vision session + mmap'd shards will not fit ≤400 MB idle if SigLIP2
wins the bench, and ADR-5's parenthetical *"(few hundred MB max)"* is doing load-bearing work with
no rule behind it. B9 (*"model on-disk size ≤150 MB primary model"*) never says whether "model"
means the vision tower or the vision+text pair — SigLIP2 int8 is 94.6 MB as vision-only (passes)
and ~378 MB as a pair (fails), so the same candidate can be ruled in or out by reader mood.
**Fix:** (a) ADR-5 gains: *"Resident set = vision session (indexing only, released when idle) +
**precomputed tag table** + text tower **lazily loaded and LRU-evicted after `--text-ttl`
(default: never on desktop, 300 s on the 8 GB server)**. The tag table is what makes tag-path
search need no text encoder at all."* (b) B9 restated: *"≤150 MB **vision tower** on disk;
≤450 MB vision+text combined; both reported."* (c) B8 gains a per-target column (M3 dev vs 8 GB
Linux) per ADR-10.

### I-6. Three mutually inconsistent throughput measurements exist for the same model class; the oracle cites none of them and B1 sits inside the disagreement band.
For ViT-B/32-class int8 vision on this machine, the corpus reports **75.5 img/s** (runtime §2.1,
thr4 bs8), **113.4 img/s** (runtime §3.2, self-quantized, thr4 bs4), **157.9 img/s** (priorart
§1.3, intra_op=8 bs8), and **181.2 img/s** (runtime §6.3 e2e, 12×1). That is a 2.4× spread across
lanes, partially but not fully explained by geometry and load. ORACLE's §8 field log carefully
records *other* cross-lane conflicts (PE-Core license, rclip 119 vs 180) but not this one — the
most decision-relevant number in the project (B1 = 60, stretch 120, sits inside the band).
**Fix:** add a field-log entry: *"2026-07-22 · ViT-B/32-int8 throughput measured at 75.5 / 113.4 /
157.9 / 181.2 img/s by different lanes under different geometry and machine load. **Unreconciled.**
No published claim may cite any of them; B1 is decided by one controlled sweep under the C-6
validity gate, with the tuple from C-5."*

### I-7. ADR-2's "measured crossover ~300k" and "0.3–105 s build" are not in the evidence.
The corpus says: *"ANN only above a measured ~100k–500k crossover"* (priorart §1.1) and
*"Adopt only past ~1M vectors" / "numpy … fine to 1M, acceptable to ~2M"* (runtime §5.3). "~300k"
appears nowhere and is presented in ADR-2 as *measured*. Likewise "0.3–105 s build": the tables
show minimum ANN build times of 0.2 s (usearch HNSW i8 @10k) up to 105 s, but the same tables
show faiss `IndexFlatIP` building in 17 ms — the range is an artifact of mixing tables. On a
project whose whole differentiator is honest numbers, an invented midpoint in the oracle is
self-inflicted. **Fix:** *"Revisit if: corpus exceeds the **measured crossover band (~500k–1M;
lanes disagree, re-measure before acting)** → binary-quantized coarse pass + f32 rerank, then
usearch HNSW i8. ANN build cost at 10k–100k measured 0.2–105 s depending on library and config,
to save ≤0.3 ms/query."*

### I-8. The ANN escalation path silently breaks B11 (search-while-indexing).
runtime.md R11: *"HNSW is architecturally incompatible with cheap search-while-indexing. If
someone later 'adds an index for speed', they will break the concurrency guarantee that §6
validated. **Document this.**"* ADR-2's revisit clause names usearch as the escalation without
mentioning it. **Fix:** append to ADR-2: *"⚠️ Any ANN adoption **breaks the lock-free
search-while-indexing guarantee** (B11) — every insert mutates a shared graph. An ANN path must
therefore be build-once-after-indexing, with brute force serving during jobs. Do not adopt one
without re-passing `bench concurrent`."*

### I-9. No format story: HEIC/PNG/WebP/CMYK/corrupt files are absent from ADR-7, the playbooks, and the risk register — and `draft()` is JPEG-only.
The vision is *"like google photos"* over a user's photos; on macOS/iPhone that is **HEIC**, and
Pillow cannot open HEIC without `pillow-heif`, which is not in ADR-7's dependency list. Worse, the
entire decode strategy (`Image.draft()`, the 1.7–2.1× win) is a **libjpeg DCT feature — it is a
no-op for PNG/WebP/HEIC**, so throughput silently falls to the naive path on a non-JPEG corpus
with no diagnostic. Nothing covers unreadable/truncated files, CMYK JPEGs, animated GIFs, 0-byte
files, or non-image files in a folder. **Fix:** (a) ADR-7 gains: *"`pillow-heif` is an **optional
extra** (`imgtag[heic]`); without it HEIC files are skipped with a named, counted, actionable
error — never a silent gap in the index."* (b) Playbook: **`Indexing is slower than the bench and
draft() 'isn't working'`** → `draft()` is JPEG-only; check the format mix (`imgtag stats
--formats`). Non-JPEG corpora decode at the naive rate; report the format mix with the number.
(c) Playbook: **`Some images never appear in results`** → check the skip ledger: unreadable,
truncated, unsupported format, 0-byte, or >`--max-pixels` (decompression-bomb guard). Every skip
is counted, reasoned, and visible in `imgtag status`; a silent skip is a bug.

### I-10. Hypernym expansion needs WordNet at query time, but WordNet is not an allowed dependency.
ADR-3 specifies *"query-time hypernym expansion (WordNet closure + supercategory tables)"*.
datasets.md §: *"WordNet hypernym closure via `nltk.corpus.wordnet`"*. ADR-7's allowed runtime deps
are `onnxruntime + numpy + Pillow + certifi/httpx + micro-server` — no nltk, and nltk additionally
requires a runtime **corpus download**. tagging.md already solved it (*"cached to a static JSON"*)
but the oracle doesn't say so, so a cold worker either adds nltk (ADR-7 violation → escalation) or
stalls. **Fix:** amend ADR-3: *"…query-time hypernym expansion against a **static, precomputed
hierarchy JSON shipped with the package** (built offline from WordNet + COCO supercategories + OI's
600-class tree; nltk is an **export-time-only** tool, never a runtime dependency)."*

### I-11. Operational failure modes with no entry anywhere: disk-full mid-index, moved/deleted source images, re-index after model change, duplicate paths.
Each is likely within the first real user session and each currently has undefined behaviour.
**Fix:** one playbook block: *"**Disk fills during indexing** → the writer must fsync-then-publish;
a failed append leaves rows past `count` which readers already ignore. Fail the job with the
remaining-bytes number; never leave a manifest claiming rows that aren't durable.
**Source images moved/deleted after indexing** → hits carry stale paths; the API returns
`exists: false` rather than 404-ing the whole result set; `imgtag verify <dataset>` re-stats and
tombstones. **Model changed** → the manifest's `model_sha` mismatch refuses the search loudly
(ADR-6) and `imgtag reindex` is the only path; never mix embedding spaces.
**Same file present twice** → id is xxhash64 of bytes (IA.md), so duplicates collapse to one row
by construction; keep all paths in the ids record."*

---

## MINOR

- **M-1. "OI tree cross-check" over-promises what's on disk.** `data/openimages/` holds only
  `bbox_labels_600_hierarchy.json` + class descriptions (100 KB) — **no OI images, no OI image
  labels**. The cross-check is taxonomy-only (applying an independent tree to COCO images), which
  is defensible but not what §6's phrasing implies. `data/caltech101/` is **empty**. Fix: §6
  reads *"OI's 600-class **hierarchy** (taxonomy-only cross-check — no OI imagery is downloaded);
  caltech101 is not part of the bench (empty on disk)."*
- **M-2. B13 (≤2 s cold start incl. model load) has no evidence and no risk row.** ADR-5 asserts
  it parenthetically. On the 8 GB Linux target with a cold page cache and a 95–283 MB model, ORT
  session creation plus graph optimization plausibly exceeds it. Fix: add a risk row with the
  pre-approved response *"pre-warm on daemon start; cache the ORT-optimized graph
  (`optimized_model_filepath`); if still over, B13 splits into cold (first-ever) vs warm-disk"*.
- **M-3. Playbook "ship the faster" ignores fidelity.** *"int8 model slower than fp32 → Record
  both, ship the faster"* must read *"ship the faster **that passes the fidelity gate (ADR-11)**;
  a faster model with <0.90 NN agreement is not a candidate."*
- **M-4. Two numbers for the same fact (naive int8 scan).** §3 dead ends says *"22× slower"*
  (priorart), runtime §5 measured *"6× slower"* (1.60 vs 0.25 ms). The oracle picked the more
  dramatic without noting the other. Fix: *"6–22× slower depending on shape/lane — both measured;
  either way, never do narrow-dtype matmul in numpy."*
- **M-5. Field-log timestamps are timezone-ambiguous.** §8 uses bare `10:15`, `10:09` while the
  research capsules use `10:14Z`, `10:31Z` and VISION-ADDENDA uses `~10:45Z`. Ordering matters
  when reconstructing who knew what. Fix: suffix Z (or state local offset) on every entry.
- **M-6. Companions list is stale and the "three surfaces" wording collides with the Creations
  registry's identically-named invariant.** §6's first bullet says *"Three-surface project docs"*
  then enumerates four+ (VISION/BUDGETS/IA/ORACLE, plus UNKNOWNS, SWARM, VISION-ADDENDA). Fix:
  *"Project docs stay coherent: VISION.md is sealed (sha256 `9240e8b1…0200`; **verified matching
  on disk 2026-07-22**); later user constraints append verbatim to VISION-ADDENDA.md; BUDGETS/IA/
  ORACLE/UNKNOWNS/SWARM are updated in the same pass as any design change."*
- **M-7. R10 (unreproduced segfault, faiss+usearch in one process) has no entry.** Zero-cost to
  add to §3 dead ends: *"Loading faiss and usearch in one process — one unreproduced exit 139
  under memory pressure; if you ever must, isolate them in separate processes."*

---

## Attacks attempted that FAILED (the oracle held)

1. **"PE-Core is not really Apache-2.0."** Live-fetched `huggingface.co/facebook/PE-Core-S16-384`
   during this review: license reads **apache-2.0**. ADR-4, UNKNOWNS C1 and the §8 field-log entry
   are correct; models.md's own row 13 (FAIR-NC) is the stale one and is already self-corrected at
   models.md:261-266. No finding.
2. **"ADR-2's no-vector-DB call is over-confident."** Attacked from three directions: priorart's
   scan table, runtime's independent library sweep (`vb2.py`), and the memory-bandwidth model that
   predicts measurement within 25 % and extrapolates 10k → ~2 ms even on a ~10 GB/s DDR3 laptop.
   Plus recall evidence (HNSW 0.64–0.91 vs exact 1.000 on clustered data) and autofaiss's own
   decision table. The decision survives every angle; only its *numbers* and *crossover* needed
   correcting (C-2, I-7).
3. **"The append-only/atomic-manifest concurrency design is a hypothesis dressed as fact."**
   Refuted by runtime §6.2: two real processes, 20,000 rows, **9,021 concurrent searches, 0 torn
   reads, 0 wrong results, 1.6 ms median**. `os.replace` atomicity holds on POSIX and Windows.
   ADR-6's core stands (the multi-*writer* gap is I-2, a different claim).
4. **"OpenVision 2 disqualification is a version-number misreading."** Independently stated twice
   in models.md (rows 67 and 116) with the quoted training-objective language ("removes the text
   encoder and contrastive loss"). Dead end is correct.
5. **"The VISION sha256 invariant is decorative / already broken."** Recomputed:
   `shasum -a 256 VISION.md` = `9240e8b1dd74799f829fbce44f23bed26be27df1491e642d68917cb3193e0200`,
   byte-identical to `.deify/vision.sha256` and to the oracle's abbreviation. Seal intact; the
   verbatim-preservation discipline is real, and VISION-ADDENDA.md is the correct mechanism (which
   is exactly why C-1 — the oracle not citing it — is the failure, not the seal).
6. **"The CoreML-EP dead end is FUD."** Cross-corroborated by the apple-runtimes capsule
   (partition thrash, silent fp16, recompile-per-run; coremltools is the proven path) and consistent
   with the CPU-only law. Holds.
7. **"`draft()` 1.75× is a synthetic-noise artifact that won't survive real photos."** Two
   independent lanes measured it on different content (163.6 ms vs 286.7 ms full-decode; 38.8 ms vs
   80.9 ms) and agree on the direction and magnitude (1.7–2.1×). Holds — though the oracle omits
   the *best* measured variant (`draft` + `thumbnail(reducing_gap=2.0)`, 2.1× vs plain draft 1.7×);
   worth adding to ADR-3/§4 as a one-line free win.

---

---
---

# ROUND 2 — re-attack under the amended target profile (shared Linux x86, 8 GB, AVX2, co-tenants)

> Re-read ORACLE.md at commit `80112d2` (ADR-10 + ADR-4 measured refinements + load-gating +
> corpus-scoped decode playbook + x86-gap risk row now present — Round-1 C-1/C-3/C-4/C-5/C-6
> substantially landed; C-2 (f16 default), C-7 (escalation clause d), and most IMPORTANTs are
> still open as written above). New verdict for this lens: **5 CRITICAL · 6 IMPORTANT**.
>
> Bottom line: ADR-10 states the target correctly but the rest of the oracle has not been
> re-derived *through* it. Three of its own load-bearing decisions (worker geometry, memory
> ceiling, politeness) were measured on a 16-core/64 GB ARM box and **invert or become
> impossible** on a 2–4-core/8 GB shared AVX2 server, and two budgets (B1, B2) are now
> arithmetically unreachable under the politeness law the same document just added.

## R2-CRITICAL

### R2-1. The memory math does not close: ADR-4's N-process geometry × B8's ≤1.0 GB is unsatisfiable, and nobody has written down per-worker RSS.
ADR-4 mandates *"N worker processes × 1 ORT intra-op thread (12×1 = 181 img/s)"*. Each worker is
a **separate Python interpreter with its own ORT session** — model initializers are private per
process (no COW sharing after ORT's own copy/alignment), plus arena, plus interpreter+numpy+Pillow.
Realistic per-worker RSS: ~60–90 MB interpreter/libs + ~96 MB int8 weights (352 MB if `doctor`
picks fp32 on non-VNNI x86 — see R2-4) + activation arena at batch 4–8. Call it **200–350 MB
(int8) / 450–600 MB (fp32) per worker**. B8🐧 caps peak indexing RSS at **≤1.0 GB**. So the
target admits roughly **2–3 int8 workers, or ONE fp32 worker** — and nowhere in ORACLE or BUDGETS
is worker count tied to the memory budget, nor is per-worker RSS listed as a thing to measure.
A cold builder sets `workers = cores/2` per ADR-10(b), lands at 4 on an 8-core VM, and blows B8
on the primary target while passing it on the Mac.
**Fix.** (a) ADR-10(c) gains: *"**Worker count is memory-derived, not core-derived:**
`workers = clamp(1, min(cores//2, floor((RSS_budget − daemon_RSS) / measured_per_worker_RSS)))`.
`bench resources` MUST report per-worker RSS as a first-class number; `imgtag doctor` measures it
on the host before choosing geometry."* (b) Add the mitigation that makes multi-process affordable:
*"Export models with **external-data weights** and load with `session_options.add_session_config_entry`
mmap/`use_ort_model_bytes_directly` so weight pages are shared read-only across workers; set
`arena_extend_strategy=kSameAsRequested` and disable mem-pattern for bounded arenas."*
(c) B8 gains a row-level note: *"measured at the defaults `doctor` selects, with worker count and
per-worker RSS printed alongside the total — a total without its geometry is not a result."*

### R2-2. "N processes × 1 thread" is an Apple-silicon artifact presented as a portable law, and `doctor` does not autotune the axis that matters most.
runtime.md §6.3 attributes the 1.7× win to (a) GIL-bound decode, (b) *"ORT's intra-op pool scales
poorly past the performance-core count **and gets scheduled onto efficiency cores**"*, (c) one
session per process. Reason (b) **does not exist on an x86 server** (no P/E asymmetry; SMT
behaves differently), and on a 2–4-core box with `workers ≤ cores/2` the process fan-out has 1–2
slots — while each extra process costs a full model copy (R2-1). On that machine, **one process
with 2–3 intra-op threads may strictly dominate** (shared weights, one arena, no IPC, no queue).
ADR-10(d)'s `imgtag doctor` sweeps *"fp32 vs int8, thread count, batch size"* — it does **not**
sweep **worker×thread geometry**, the single decision this contradiction turns on.
**Fix.** ADR-4's geometry sentence becomes: *"Parallelism geometry is **host-probed, not fixed**.
The 12×1 = 181 img/s result is an **ARM PROXY** whose stated cause (efficiency-core scheduling) is
Apple-specific and does not transfer to x86. `imgtag doctor` sweeps the full grid
`workers × intra_op × batch × precision` under the memory ceiling of R2-1 and stores the winner in
the machine profile. On ≤4-core hosts expect single-process/multi-thread to win on memory alone."*
Add to ADR-10(d)'s autotune list: *"…worker×thread geometry (the axis most likely to invert
between ARM dev and x86 target)"*.

### R2-3. B15 protects the co-tenant's CPU and IO but not its RAM or its page cache — the two things an 8 GB shared box actually dies of.
`nice` governs CPU scheduling; `ionice` governs block-IO queueing (and only under CFQ/BFQ — it is
a **no-op on most NVMe/multiqueue setups**, which is worth knowing before it is trusted). Neither
touches: (1) **page-cache eviction** — streaming 10k images (GBs) through `read()` evicts the
co-tenant's hot working set, so the co-tenant is slow *after* our job finishes and B15's
during-run probe can pass while we did real damage; (2) **RSS pressure** — our 1.0–1.5 GB on a box
whose co-tenants already hold several GB is what pushes it into swap or wakes the OOM killer.
B8's own parenthetical *"leaves ≥6.5 GB to co-tenants"* is internally contradictory with the
premise that co-tenants are **already running** — there is no empty 8 GB to divide.
**Fix.** (a) B15 gains two clauses: *"**Cache hygiene:** after decoding an image the reader calls
`posix_fadvise(POSIX_FADV_DONTNEED)` on its range (no-op elsewhere); index shards are mmap'd
read-only and `MADV_RANDOM`; bulk reads never warm the cache with data we will not reuse.
**Post-run recovery:** the co-workload probe is measured for 60 s **after** our job ends and must
return to within 5 % of its solo baseline — a during-run pass alone does not satisfy B15."*
(b) *"**We die first:** worker processes set `oom_score_adj` high (+500) and an `RLIMIT_AS` derived
from the memory budget, so kernel pressure kills IMGTAG, never the co-tenant."* (c) B8's
parenthetical becomes *"≤1.5 GB total under load — measured as **our** footprint; the budget makes
no claim about free memory on a box we do not own, and `imgtag doctor` refuses defaults that would
take more than 25 % of currently-free RAM."* (d) Playbook: **`The server got slower and it wasn't
CPU`** → page cache or swap. Check `free -m` before/after, `vmstat si/so`, and whether the
co-tenant's cache was evicted; nice/ionice do not cover this — the fadvise path and the RSS cap do.

### R2-4. `cores` is undefined on the target class, and `os.cpu_count()` lies inside containers/cgroups.
ADR-10(b) and B15 both key defaults off *"cores"* (`workers ≤ cores/2`). A shared Linux server is
very likely a container or a cgroup-limited VM where `os.cpu_count()` reports the **host's** CPU
count while `cpu.max` grants a fraction of it. The classic outcome is the exact harm the user
forbade: we spawn 8 workers against a 2-CPU quota, get throttled, thrash, and starve the
co-tenant. Nothing in the oracle mentions cgroups, quotas, or affinity.
**Fix.** ADR-10(b) gains: *"**'cores' means effective cores**, resolved in this order:
`len(os.sched_getaffinity(0))` → cgroup v2 `/sys/fs/cgroup/cpu.max` quota/period → cgroup v1
`cpu.cfs_quota_us/cfs_period_us` → `os.cpu_count()`. Never `os.cpu_count()` alone. `imgtag doctor`
prints the resolved value, its source, and the resulting geometry; a mismatch between reported and
effective cores is logged loudly."* Playbook: **`Throughput is far below the core count's promise
on the server`** → you are inside a CPU quota. Check `sched_getaffinity` and `cpu.max`; re-run
`doctor`; expect throttling, not parallelism.

### R2-5. B1 and B2 are now unreachable on the primary target by ~an order of magnitude, and no risk row or pre-approved response exists for that.
B1 (≥60 img/s, stretch ≥120) and B2 (≤3 min/10k) remain **un-🐧'd, M3-anchored, single-threshold**
rows. Under ADR-10/B15 on a "not powerful" 2–4-core AVX2 server with `workers ≤ cores/2`, nice 10,
and (per R2-4) possibly a CPU quota, a realistic ceiling is **single-digit to low-teens img/s**
— 10k images in **15–30+ minutes**. Every plausible outcome from here is bad in a different way:
a worker silently loosens B1 (forbidden), or reports the M3 number as the product claim (the
half-succeeds-and-lies row), or `bench all` exits nonzero forever and "done" becomes unreachable.
**Fix.** (a) Split every speed budget into two rows with two statuses:
`B1-dev (M3 PROXY, full speed) ≥60 img/s` and `B1🐧 (target, polite defaults) ≥TBD — locked only
by a measurement on the real server; provisional planning figure ≥10 img/s (photofield-ai's
2014 6-core did 20 at full speed; we run at half cores and nice 10)`. Same for B2, B3, B4, B13.
(b) Add a risk row: `| B1/B2 unreachable on target under B15 politeness | H | headline claims, "done" definition | first target-host bench | publish per-profile budgets (dev-ceiling vs polite-target); escalate to user with the honest two-column table; NEVER loosen a budget or quietly report the proxy as the product number |`
(c) Add to the escalation contract: *"(g) you are about to report a PROXY number without its label,
or compare a dev-machine number against a target-profile budget."*

## R2-IMPORTANT

- **R2-6. OpenVINO's rejection rationale inverted with the target.** ADR-1 defers OpenVINO as
  *"Intel-centric; keep as bench slot"* — the target **is** x86, and non-VNNI AVX2 int8 is ORT
  MLAS's weakest configuration and roughly OpenVINO's strongest relative footing (runtime.md:
  *"Strong on Intel x86 … worth a later A/B on old x86 boxes only"*). Fix: promote to *"ADR-1
  revisit is now ACTIVE: OpenVINO gets one **required** bench slot on the target host (not on the
  Mac). If it wins ≥1.5× there, it becomes the x86 EP behind the same pluggable-backend seam
  ADR-4 already provides — install weight (~100 MB) is then a measured trade, not a guess."*
- **R2-7. The default precision on an unprofiled x86 host should be fp32, and the oracle still
  implies int8.** The corpus's only x86 quantization datapoints both point the wrong way for the
  target: clip.cpp's q8_0 *slower* than f32 on an Intel Mac, and ORT's own docs warning that
  pre-VNNI x86 can lose from quantization. ADR-4's recipe reads as a settled default. Fix: add
  one sentence — *"**On any host that has not been through `doctor`, the default is fp32.** int8
  is enabled only by a measured win on that host (ADR-10e). The recipe above is how we quantize,
  not a promise that we will."* Also amend the playbook *"int8 model slower than fp32 → ship the
  faster"* → *"…ship the faster **that passes the fidelity gate**; on x86 without VNNI expect fp32
  to win and treat that as the normal case, not an anomaly."*
- **R2-8. No Linux daemon lifecycle story anywhere** (ADR-5 assumes a desktop). Missing: how the
  daemon starts on a server (systemd user unit? `--daemon` fork? nothing?), single-instance
  enforcement, **UNIX socket vs TCP port** (a shared box may already use your port, and binding
  TCP exposes search to other tenants — a privacy leak on a multi-user server), permissions on
  `~/.imgtag`, behavior for two users running IMGTAG simultaneously, and restart-on-crash. Fix:
  add ADR-13: *"Daemon binds a **UNIX socket** at `~/.imgtag/daemon.sock` (mode 0600) by default —
  never a TCP port on the shared server; `--http 127.0.0.1:PORT` is opt-in for the local app and
  refuses non-loopback binds. Single instance per user via a flock on the socket dir; a stale
  socket is detected and replaced, never silently reused. Ship a systemd **user** unit
  (`--install-service`) with `MemoryMax=`, `CPUWeight=`, `Nice=10`; no root, no system-wide
  install."* Add playbook: **`Address already in use / daemon won't start`** → stale socket or a
  second instance; `imgtag status` names the holder pid; never fall back to a random port.
- **R2-9. B13 (≤2 s cold start) is a desktop budget on a contended-disk server** and now collides
  with ADR-5's resident text tower under the 350 MB idle-RSS cap. Fix: 🐧-split it —
  *"B13-dev ≤2 s (warm page cache); B13🐧 ≤5 s cold-disk, ≤2 s warm — and cold start is rare by
  design because the daemon is a long-lived systemd user service (ADR-13)."* Also cache the
  ORT-optimized graph (`optimized_model_filepath`) so session creation isn't re-paid each start.
- **R2-10. fsync-per-batch is an IO tax on a shared box.** The validated writer pattern fsyncs per
  250-row commit; `ionice` does not throttle fsync, and frequent fsyncs on a shared disk hurt
  co-tenants measurably. Fix: add to ADR-6: *"Commit cadence is **time-based and configurable**
  (default: every ~2 s or 500 rows, whichever is later — B11 needs ≤2 s visibility, nothing
  faster). On the polite profile the writer coalesces commits and never fsyncs more than once per
  2 s. Report commits/s in `bench politeness`."*
- **R2-11. The 8 GB ceiling changes model ranking, but ADR-4's roster still leads with the fat
  one.** ADR-10(c) correctly says small models "gain rank", yet ADR-4's roster order and the
  PE-Core-failure playbook both make **SigLIP2-base** (94.6 MB vision int8 + **283 MB int8 /
  1.13 GB fp32 text tower**) the anchor/fallback — and if `doctor` picks fp32 on non-VNNI x86
  (R2-7), SigLIP2's fp32 pair is ~1.5 GB, i.e. the entire B8 budget for one model. Fix: state the
  consequence explicitly in ADR-4: *"**Target-profile ranking differs from quality ranking.**
  SigLIP2 is the quality ANCHOR (reference for how good we could be); the shippable default on the
  8 GB profile is the best model whose fp32 vision tower + resident text path fits the memory
  ceiling — PE-Core-T16/S16, SigLIP-v1 (111 MB int8 text), UForm. The bench reports a
  quality-per-MB-of-RSS column, and the default is chosen on the TARGET profile, not the dev box."*

## Round-2 attacks that FAILED (the amended oracle held)

1. *"ADR-10 is decorative — the numbers behind it weren't touched."* Partly refuted: BUDGETS B8/B15
   were genuinely re-derived (tightened ceilings, co-workload probe, 🐧 markers, proxy-labeling
   protocol) and the field log records the constraint with its timestamp. The failure is scope
   (B1/B2/B3/B13 not re-derived — R2-5), not sincerity.
2. *"The brute-force scan collapses on a weak x86 box."* Refuted by the bandwidth model in
   runtime.md §5.1 — the scan is bandwidth-bound and extrapolates to ~0.8 ms @10k on a ~25 GB/s
   DDR4 laptop and ~2 ms on ~10 GB/s DDR3. Even a bad shared server keeps a ~25–60× margin under
   B3. ADR-2's *architecture* survives the target change untouched (its dtype default, R1 C-2,
   is a separate matter).
3. *"8 GB can't hold the index at 100k."* Refuted: 100k × 512 f32 = 205 MB (10k = 20.5 MB), and it
   is mmap'd and evictable. Scale is not the memory problem — **worker processes** are (R2-1).
4. *"The fidelity gate is ARM-specific and won't transfer."* Refuted: cos/NN-agreement are
   numerical properties of the graph, not the ISA. The gate transfers as-is; only the *speed*
   half of the quantization decision is per-arch, which ADR-10(e) already says.
5. *"Politeness and the load-gate contradict each other (you can never get a quiet bench on a
   shared server)."* Survives as a real tension but not a defect: the load-gate governs the **dev**
   bench (where quiet is achievable), and the target-host run is explicitly a
   politeness/co-tenant measurement (B15), not a speed record. Worth one clarifying sentence in the
   load-gate risk row — *"on the shared target, `bench all` records loadavg and marks rows
   ADVISORY rather than refusing to run; the target's honest number is a polite number"* — but
   the oracle is not wrong here.

---

## Suggested application order

C-1 and C-3 first (they change what gets built and what "quality anchor" means), then C-4/C-2
(they change the hot path), then C-5/C-6 (they change what any number is worth), then C-7 (it
unblocks `b-skill`). The IMPORTANTs are all local text edits to ADRs/risk rows/playbooks and can
land in one pass. Nothing here requires new research — every fix is quoted from the existing
corpus or from disk truth.
