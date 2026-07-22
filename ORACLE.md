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
  (Intel-centric; keep as bench slot), over CoreML (violates CPU-only law; opt-in future).
  Revisit if: bench shows another EP ≥1.5× on target hardware.
- **ADR-2 No vector database. Exact brute-force scan** over L2-normalized contiguous
  mmap (f16 on disk, f32 accumulate). Measured 0.47ms @10k×512, 7.4ms @100k on this
  machine; every ANN option costs 0.3–105s build to save ≤0.3ms. Revisit if: corpus
  >~300k measured crossover → binary-quantized coarse pass + f32 rerank (design ready),
  then usearch. NEVER pgvector/sqlite-vec for the hot path at this scale.
- **ADR-3 Hybrid retrieval architecture.** Embedding index (free-text recall) + tag
  vocabulary (~4–8k tags from COCO/LVIS/OI names + curated) scored via the SAME text
  encoder at index time (one matmul, marginal cost ~0) with per-tag calibrated thresholds
  + query-time hypernym expansion (WordNet closure + supercategory tables, max-pooled).
  Over embedding-only (fails: multi-object recall, superordinate queries, no honest
  no-match) and over heavy tagger models (RAM++ = 3GB/750M params, 24× PE-T FLOPs — dead
  end). Tags are the FP gate and the explanation surface.
- **ADR-4 Pluggable model backends; Apache-2.0 default.** Bench roster (2026-07-22):
  PE-Core-S16-384 + T16-384 (Apache; export spike first), SigLIP2-base-224 (Apache,
  official int8 ONNX — quality anchor), SigLIP-v1-base (Apache, small text tower),
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
  Parallelism geometry: **N worker processes × 1 ORT intra-op thread** (12×1 = 181 img/s
  e2e proxy; more ORT threads REGRESSES past perf-core count). Embedding storage: fp16
  shards (measured lossless, cos 1.000000); compute in fp32 BLAS chunks (narrow-store/
  wide-compute; numpy int8/fp16 matmul falls off BLAS = 6–30× slower).
- **ADR-5 Resident daemon + warm text tower.** Anti-pattern proven: immich unloads models
  after 300s → 60–70s cold search. We keep the text tower resident (few hundred MB max),
  LRU query cache, tag table precomputed. CLI talks to the daemon when present, else
  in-process (still ≤2s cold, B13).
- **ADR-6 Storage = append-only shards + atomic manifest.** `~/.imgtag/datasets/<slug>/`:
  `shard-XXXX.f16` (embeddings, mmap), `ids-XXXX.jsonl` (image id/path/dims), 
  `manifest.json` (atomic tmp+rename; counts, model id+hash, shard list). Readers snapshot
  the manifest; writers append + rename. Search-while-indexing and crash recovery fall out
  free. Never mutate published shards; compaction writes new files then swaps manifest.
- **ADR-7 Engine deps = onnxruntime + numpy + Pillow + certifi/httpx + micro-server.**
  NO torch/transformers at runtime (export tooling may use them offline). uv-managed venv.
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
  to ≤1.0GB indexing / ≤1.5GB total under load); (d) **first-run autotune**: `imgtag
  doctor` runs a ~30s micro-bench on the actual deploy machine (fp32 vs int8, thread
  count, batch size — int8 winners differ per ISA) and stores the recipe in the machine
  profile; "generic and ready" means the engine adapts itself, not that we guessed;
  (e) quant decisions are PER-ARCH — NEON results never transfer to AVX2 (clip.cpp
  anomaly was x86; DeepSparse win was AVX512-VNNI; measure on target). Revisit if: the
  user later asks for Mac-local optimization (add profile, change no defaults).

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
- Naive numpy int8 scan — 22× slower than f32 BLAS (measured). int8 scan needs SimSIMD-
  class kernels or don't bother.
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
  Record both, ship the faster; do not "fix" the anomaly mid-bench, log it.
- **Embeddings don't match reference / quality bench suddenly drops** → run the parity
  gate: `uv run imgtag bench parity`. Prime suspects: resize interpolation (must match
  model card — usually bicubic), normalization constants, RGB/BGR, draft() decode scale,
  missing L2-normalize (MobileCLIP2 exports are UNNORMALIZED — assert ‖v‖≈1 in tests).
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
| All int8 paths regress on M3 | M | miss B1 stretch | bench matrix | ship fp32/f16, note; edge story via smaller model |
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

- Three-surface project docs stay coherent: VISION sha256 `9240e8b1…3200` never changes;
  BUDGETS/IA/ORACLE updated in the same pass as design changes.
- Every embedding row L2-normalized: unit test asserts mean ‖v‖ ∈ [0.999, 1.001].
- Index manifest always names {model_id, model_sha, dim, count, shards[]}; loader refuses
  mismatches loudly.
- `uv run imgtag bench all` = the full budget table; exits nonzero on any budget red.
  This is the project's reconcile.py-equivalent: run before every "done" claim.
- Quality metrics computed only against downloaded ground truth (never eyeballed): COCO
  exhaustive 80 classes = FP source of truth; LVIS federated protocol for depth; OI tree
  cross-check.
- No dataset bytes in git (data/ ignored); no Unsplash redistribution; no apple-amlr
  weights in defaults or published artifacts.
- Every spawned agent: explicit non-Fable model + effort; MODEL line first in reports.
- Nothing leaves the machine without fire17's explicit confirmation (registry law #2).

## 7. Escalation contract (for every worker on this project)

STOP and report (symptom + what you tried + which oracle entries you consulted) when:
(a) a budget test fails twice for the same cause; (b) reality contradicts an ADR or a
research number you were handed; (c) you are about to add a dependency outside ADR-7 or
any GPL/AGPL/amlr-licensed code; (d) you are about to relax a budget, delete user data,
or touch anything outside the ImgTag tree + ~/.imgtag; (e) an export/quantization fight
exceeds its timebox; (f) you catch yourself guessing a number you could measure.
"I stopped because X" is a success report. Silent grinding is the only failure.

## 8. Field log (append-only: symptom → what worked)

- 2026-07-22 10:15 · Framework-Python SSL failure killed Unsplash pull → curl-based
  fetcher (scripts/fetch_unsplash_demo.sh); SIGPIPE from `tail|head` under pipefail →
  single-awk restructure. Both patterns now in playbooks.
- 2026-07-22 10:09 · Cross-lane license conflict on PE-Core (NC vs Apache) → resolved by
  fetching the repo + HF card live: LICENSE.PE = Apache-2.0; FAIR-NC is PLM's. Lesson:
  two lanes disagreeing = fetch the primary source, never vote.
- 2026-07-22 10:31 · Two research reports arrived under generic agent id (name lost in
  routing) → content cross-verified against disk artifacts before adoption. Lesson: trust
  content provenance (verifiable claims + URLs), not sender labels.
- 2026-07-22 10:20 · rclip README's 119 img/s conflicts with its own PR #249 (~180 img/s
  CoreML) — README is stale, PR is measured. Lesson: PRs > READMEs for numbers.
- 2026-07-22 10:45 · User constraint mid-mission: primary deploy = shared Linux x86 8GB
  no-GPU server, co-tenants must not slow. → ADR-10, B8/B15 tightened, Wave A briefs
  amended by broadcast. Lesson: the harbor moved; the map moved with it same-pass.
- 2026-07-22 10:50 · runtime lane's completion message + report were delayed ~20min in
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
