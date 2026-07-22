# PRIOR ART — local / self-hosted semantic image search

> Lane: **PRIOR ART** (IMGTAG research phase). Compiled **2026-07-22**.
> Founding-vision mandate this file serves: *"learn what others have accomplished before
> and make it even better than anyone has done before"* (`VISION.md`).
>
> **Law of this file: no fabricated numbers.** Every number carries a source and its
> hardware context. Where the project publishes nothing, the row says
> **`no measured number found`** — and that absence is itself the single biggest finding
> in this field (see §0).
>
> Repo facts (stars / last push / license / archived) come from the **GitHub REST API,
> queried 2026-07-22**, not from memory. Every URL in §Appendix A was HTTP-checked (200).

---

## 0. The headline finding — the field publishes no numbers

Across ~25 serious projects surveyed, **almost nobody publishes reproducible CPU
benchmarks** for indexing throughput or search latency. The exceptions are counted on one
hand:

| Who | Published CPU number | Source |
|---|---|---|
| **rclip** | **84,725 photos in 15 h on a Celeron J3455 (1.57 img/s)**; **50,000 in 7 min on an M1 Max (119 img/s)**; 1.28 M in 3 h on the same MacBook | [README](https://github.com/yurijmikhalevich/rclip) (verified verbatim) |
| photofield-ai | **~20 req/s on an i7-5820K (6-core, 2014)**, ~200 req/s on GTX 1070 Ti | [README](https://github.com/SmilyOrg/photofield-ai) |
| open-clip-inference-rs | per-model CPU ms/img for MobileCLIP2 + SigLIP2 (⚠️ CPU model unstated) | [README](https://github.com/RuurdBijlsma/open-clip-inference-rs) |
| rclip release notes | **180 img/s (M1 Max, CoreML) · 1.9 img/s (Celeron J3455) · 15 s text search over 80k images** — before/after numbers for a real change | [PR #249](https://github.com/yurijmikhalevich/rclip/pull/249) |
| memery | 9× indexer speedup after removing 46 M GPU syncs (Apple MPS, not CPU) | [PR #41](https://github.com/deepfates/memery/pull/41) |
| clip-retrieval | 1,400–1,500 sample/s on an RTX 3080; 7,000 sample/s on 8×A100 (**GPU only; CPU: none**) | [README](https://github.com/rom1504/clip-retrieval) |
| clip-as-service | full QPS table — **all on a TITAN RTX; no CPU numbers exist** | [benchmark](https://clip-as-service.jina.ai/user-guides/benchmark/) |
| Queryable (iOS) | *"about 2000 photos per minute on an iPhone 12 mini"*; 35,000-photo search in 2.8 s | [mazzzystar.com](https://mazzzystar.com/2022/12/29/Run-CLIP-on-iPhone-to-Search-Photos/) |
| UForm (unum-cloud) | **325.4 img/s (small, ONNX) / 212.8 img/s (base, ONNX)** on a *160-core dual-socket Intel Emerald Rapids*, batch 128 | [BENCHMARKS.md](https://github.com/unum-cloud/uform/blob/main/BENCHMARKS.md) |
| clip.cpp | **4-bit quantized CLIP = 85.6 MB** on disk (size, not speed) | [README](https://github.com/monatis/clip.cpp) |
| Nextcloud Recognize (user report) | "cranking through **10k–20k photos per day**" (≈0.12–0.23 img/s) | [HN 44426233](https://news.ycombinator.com/item?id=44426233) |
| **immich (docs)** | **per-model CPU execution time + peak RSS + recall**, measured on a 7800X3D bare-metal Linux at f32 — the best dataset in the field (§2.1.1) | [docs/features/searching.md](https://github.com/immich-app/immich/blob/main/docs/docs/features/searching.md) |

Everything else — PhotoPrism, Lap, Ente, PixFinder, Desktop Docs, LibrePhotos, Photonix and
the 2026 wave of "local AI photo search" apps — ships **feature lists, not measurements**
(immich publishes per-model *forward-pass* times but no end-to-end throughput; rclip
publishes throughput but no search latency). Two apps
we fetched directly ([PixFinder](https://pixfinder.app/), Software Mansion's on-device
[ExecuTorch series](https://swmansion.com/blog/on-device-image-semantic-search-recreating-apple-google-photos-in-react-native-part-4-1aeedc044289/))
state *zero* quantitative specs.

**Consequence for IMGTAG:** the benchmark ground is *unoccupied*. A project that ships a
reproducible `bench` command with published, hardware-labelled numbers (BUDGETS.md B1–B14)
is instantly the most credible tool in the category — before a single millisecond of
optimisation. That is the cheapest world-beating move available, and nobody has taken it.

---

## 1. Own measurements (this machine, today)

Because the field publishes so little, this lane produced its own reference numbers on the
declared dev target. **Apple M3 Max, 16 cores, macOS, numpy 2.4.3, Python 3, 2026-07-22.**
Reproduce with the snippets in Appendix B.

### 1.1 Brute-force vector scan (the "do we even need an ANN index?" question)

Cosine scan over the full corpus + `argpartition` top-50, single query, f32:

| corpus | dim | dtype | scan + top-50 | index RAM |
|---:|---:|---|---:|---:|
| 10,000 | 512 | f32 | **0.47 ms** | 20.5 MB |
| 10,000 | 256 | f32 | **0.09 ms** | 10.2 MB |
| 10,000 | 64 | f32 | 0.09 ms | 2.6 MB |
| 100,000 | 512 | f32 | 7.40 ms | 204.8 MB |
| 1,000,000 | 512 | f32 | 86.0 ms | 2,048 MB |
| 10,000 | 512 | int8 (naive numpy upcast) | 10.64 ms ⚠️ | 5.1 MB |

Three conclusions, each of which contradicts what the incumbents built:

1. **At 10k images an ANN index is dead weight.** 0.47 ms brute force is ~100× under the
   B3 budget (p50 ≤50 ms). immich ships pgvector/VectorChord + HNSW; Lap and rclip carry a
   DB; all of that complexity buys nothing below ~500k images. Brute force is also
   *exact* — no recall loss, no index build time, no "rebuild after model change" step
   (a real immich pain: changing the model forces re-embedding + re-index).
2. **Search latency in this class is not search — it is the text encoder.** If scan is
   0.5 ms and the budget is 50 ms, then ~99% of the budget is text-encode + IPC + process
   overhead. Optimising the vector store is optimising the wrong thing; **caching /
   quantising / warm-starting the *text tower*** is where the win is.
3. **int8 must be a real SIMD kernel, not numpy.** Naive int8→int16 upcast is *22× slower*
   than f32 BLAS. Quantised scan only pays with usearch/SimSIMD-class kernels — otherwise
   f32 BLAS wins. (Memory savings are real: 4× smaller.)

**Cross-checked independently.** A second lane ran the same question against real index
libraries on the same class of machine (M3 Max, dim 512, k=20, faiss-cpu 1.14.3,
usearch 2.26.0, hnswlib 0.8.0, sqlite-vec 0.1.9) and reproduced the conclusion:

| layer | query @10k | query @100k | index build @100k |
|---|---:|---:|---:|
| numpy `X @ q` + argpartition | **0.275 ms** | 3.96 ms | 0 |
| numpy, batched 100 queries | 0.111 ms/q | 1.89 ms/q | 0 |
| faiss `IndexFlatIP` (1 thread) | 0.511 ms | 3.69 ms | 17 ms |
| faiss `IndexHNSWFlat` M=16 | 0.131 ms | 0.088 ms | **4,346 ms** |
| usearch HNSW | 0.256 ms | 0.742 ms | **30,997 ms** |
| hnswlib M=16 ef=64 | 0.846 ms | 1.033 ms | **105,476 ms** |
| sqlite-vec `vec0` | 4.87 ms | 53.0 ms | 1,354 ms |
| packed-bit Hamming (numpy) | 0.787 ms | 9.43 ms | 0 — and **0.64 MB** at 10k |

At 10k×512 every ANN option spends **0.3–105 seconds of build time to save ≤0.3 ms per
query**. Published third-party numbers agree on shape: vectorlite's dim-512 benchmark
(i5-12600KF) shows sqlite-vec exact at 989 µs @3k / 7,778 µs @20k **at recall 1.00**, while
hnswlib returns in 781 µs **at 67.2% recall** ([vectorlite benchmark](https://github.com/1yefuwang1/vectorlite#benchmark));
faiss's own wiki puts flat CPU search at ~0.91 ms/query on SIFT1M with recall 1.0000, and
recommends flat/IVF below 1M ([faiss wiki](https://github.com/facebookresearch/faiss/wiki/Indexing-1M-vectors)).
`ann-benchmarks` does not even test a corpus as small as ours (smallest set is 60k).
**Decision: exact brute force over an L2-normalised contiguous `memmap`; sqlite-vec only if
SQL metadata and persistence are worth ~4.9 ms; ANN only above a measured ~100k–500k
crossover, with binary-quantised coarse pass + f32 rerank as the first escalation.**

### 1.2 JPEG decode is a first-class bottleneck

Decode + resize-to-224 of 12 MP JPEGs (synthetic worst case: random-noise content,
~10 MB/file — real photos compress better, so treat these as a *floor*), single thread:

| path | per image | single-thread rate |
|---|---:|---:|
| full decode → resize | 286.7 ms | 3.5 img/s |
| `Image.draft()` (DCT-scaled decode) → resize | **163.6 ms** | **6.1 img/s** |

`draft()` — decoding the JPEG at 1/2, 1/4, 1/8 scale inside the DCT — is a **1.75× free
win** here, and larger on typical photos. At 16 threads this projects to roughly
50–100 img/s of *decode capacity*, i.e. the same order as the model itself. **Any design
that treats preprocessing as "the easy part" is wrong**: at B1 (≥30 img/s) decode and
inference must be pipelined and both must be parallel. Two further multipliers nobody in
the surveyed field exploits systematically: (a) reading the **embedded EXIF thumbnail**
(160×120–1024px, already in most camera/phone JPEGs) instead of decoding the full frame;
(b) libvips/`fast_image_resize` SIMD resamplers instead of PIL.

### 1.3 Measured encoder throughput on this machine (CPU-only)

A sibling lane ran ONNX Runtime 1.27.0 with `CPUExecutionProvider` and
`intra_op_num_threads=8` on the same M3 Max (⚠️ **the machine was under heavy load from
parallel agents — `loadavg` 19→60 — so these are lower bounds**):

| build | batch | ms/img | img/s | projected 10k |
|---|---:|---:|---:|---:|
| CLIP-B/32 vision **fp32** (352 MB) | 1 | 112.89 | 8.9 | 18.8 min |
| CLIP-B/32 vision fp32 | 8 | 17.55 | 57.0 | 2.9 min |
| CLIP-B/32 vision **int8** (89 MB) | 1 | 7.98 | 125.3 | 1.3 min |
| **CLIP-B/32 vision int8** | **8** | **6.33** | **157.9** | **1.1 min** |
| wd-convnext-tagger-v3 fp32 | 1 | 556 | 1.8 | 1.5 h |
| wd-eva02-large-tagger-v3 fp32 | 1 | 2,999 | 0.33 | 8.3 h |

Two build rules fall out, both actionable today:
- **Ship int8.** 2.8–14× faster and 4× smaller than fp32 on ARM — and *every* competitor surveyed ships f32 (immich, rclip, photofield-ai, LibrePhotos, Photonix, Recognize). Lap ships quantized ONNX and is the sole exception.
- **Set `intra_op_num_threads` explicitly.** ORT's default pool under load produced 50–126 ms/img with wild variance versus ~24 ms pinned. rclip already does this; it is a free correctness-of-measurement win.
- Third-party corroboration that quantised CLIP is the right lever, and that at 10k an ANN index is not: criteo's **autofaiss** picks `Flat` below 1,000 vectors and `HNSW15` below **10,000** in its own `get_optimal_index_keys_v2` decision table — the tool that exists purely to choose indexes says *don't* at our scale.

---

## 2. The competitors

### 2.1 immich — the giant (108,427★)

- **Repo:** https://github.com/immich-app/immich · AGPL-3.0 · TypeScript · last push 2026-07-22 · **108,427★** (API, 2026-07-22)
- **Shape:** server + Postgres + a separate `immich-machine-learning` Python container; smart search = CLIP-family embeddings in Postgres vector extension, HNSW index; per-asset embeddings written as assets are processed (so partial results *are* searchable during a run).
- **Where it hurts (user-reported, not our opinion):** on the 2025 HN "self-hosted photo library" thread a user says immich's AI "really starts to show its age", "embeddings seem really poor, and has lots of misses and false positives" ([HN 44426233](https://news.ycombinator.com/item?id=44426233)). Mobile app performance is acknowledged by the devs as "a well known problem" in the same thread.
- **Verified from source (not from docs prose), 2026-07-22:**
  - Default smart-search model is **`ViT-B-32__openai`** — `server/src/config.ts:304`. The 2021 OpenAI CLIP. Out of the box, the 108k-star leader of this category runs a five-year-old model.
  - Supported set (`machine-learning/immich_ml/models/constants.py`) *does* include the modern tier — `ViT-B-16-SigLIP2__webli`, `ViT-SO400M-16-SigLIP2-{256,384,512}__webli`, `nllb-clip-*-siglip`, `XLM-Roberta-*` — but the user must find, paste and then **re-process the entire library** to use them (docs step 6: *"Click 'All' next to 'Smart Search' to begin re-processing your assets"*).
  - Vector store: **VectorChord** (Postgres ext), tables `smart_search` / `face_search` (`server/src/constants.ts:39-42`); dim varies with the model (512 / 768 / 1024 / 1152).
  - ML service: Python + ONNX Runtime, `MACHINE_LEARNING_*` env prefix, `request_threads = os.cpu_count()`, `model_inter_op_threads` / `model_intra_op_threads` exposed, and **`model_ttl: int = 300`** — the model is *unloaded after 5 minutes idle* (`machine-learning/immich_ml/config.py`). A search after an idle period pays a full model reload. Precision default is **f32** (their own docs: *"All testing and evaluation was done at f32 precision (the default in Immich)"*).

#### 2.1.1 immich's model benchmark table — the single most useful measurement in the field

Immich publishes, in-repo, memory + CPU execution time + retrieval recall for every supported
model. Method (their words): *"Memory and execution time estimates were obtained without
acceleration on a **7800x3D processor running bare metal Linux**… at f32 precision"*;
execution time = *"after warming up the model with one pass, the mean execution time of 100
passes"*; memory = *"peak RSS … does not include image decoding, concurrent processing, the
web server"*; recall = mean of recall@{1,5,10} on **Crossmodal-3600 (+XTD-10, Flickr30k)**.
English table, selected rows:

| Model | Exec time (ms) | Peak RSS (MiB) | Recall (%) | Note |
|---|---:|---:|---:|---|
| `ViT-B-32__openai` | **2.26** | 1004 | **69.9** | ← **immich's default** |
| `RN50__openai` | 2.39 | 913 | 69.02 | |
| `ViT-B-32__laion2b-s34b-b79k` | 2.29 | 1001 | 77.62 | +7.7 recall, same speed as the default |
| **`ViT-B-32-SigLIP2-256__webli`** | **3.31** | 3061 | **82.28** | +12.4 recall for +1.05 ms |
| **`ViT-B-16-SigLIP2__webli`** | **5.81** | 3038 | **84.86** | +15.0 recall for +3.55 ms |
| `ViT-L-16-SigLIP2-256__webli` | 23.77 | 2830 | 85.03 | |
| `ViT-SO400M-16-SigLIP2-384__webli` | 56.57 | 3854 | **85.99** | best quality published |
| `ViT-L-14__openai` | 19.91 | 2212 | 72.99 | big *and* worse than B-16-SigLIP2 |

⚠️ **Caveat on the exec-time column (raised by the runtime lane, unresolved):** 5.81 ms for a
ViT-B/16 image tower implies ~3 TFLOPS f32 on a 7800X3D — at or above its AVX-512 peak — so
the column may be the **text** tower or a fused figure. Counter-evidence: internal ratios
track image-tower FLOPs (B/16 5.81 → L/16-256 23.77 = 4.1×, vs ~4.6× FLOPs). immich ships
**no benchmark script** (repo searched). **Treat as relative-ordering truth; absolute values
unverified.** Re-derive locally before quoting in a public claim.

Four things fall out of this table, and they set IMGTAG's whole strategy:

1. **The default is nearly the worst model on the list** (69.9% vs 85.99% best; only the yfcc15m/cc12m relics are worse). This is the mechanical explanation for the field complaint *"embeddings seem really poor, and has lots of misses and false positives"* — it is not a mystery, it is a default.
2. **Quality is almost free.** `ViT-B-16-SigLIP2` costs **3.55 ms more per image** than the default and returns **+15 recall points**. At 10k images that is +35 seconds of indexing for a categorically better search. Nobody in §2 has taken this trade.
3. **The model is not the bottleneck — decode is.** 5.81 ms/img of inference vs our measured **163–287 ms/img** of full-resolution JPEG decode (§1.2, worst-case files; expect tens of ms on typical photos). Inference is one-to-two orders of magnitude cheaper than getting the pixels ready. Every project in §2 optimises the model. **The race is won in the decode pipeline.**
4. **f32 SigLIP2 costs ~3 GB RSS**, which blows B8 (≤1.5 GB peak) on its own — so quantisation/precision work is a *budget requirement* for us, not a nicety, and immich has never measured a quantised variant.

#### 2.1.2 What immich actually achieves in the field — and why it is ~1000× off its own bench

The 2.26 ms figure is a bare model forward pass. What users measure end-to-end:

| Field rate | Hardware | Model | Source |
|---|---|---|---|
| **28,000 assets ≈ 7 h CPU-only (~1.11 img/s)** → 10k ≈ 2.5 h; ~55 min with OpenVINO | Intel N100 | n/s | [sumguy.com, 2026-05-11](https://sumguy.com/immich-hardware-acceleration/) |
| **~0.4 img/s @2 concurrent, ~0.44 @4** — *identical CPU vs OpenVINO* | i5-10500 + UHD 630 | `nllb-clip-large-siglip__v1` | [disc #8104](https://github.com/immich-app/immich/discussions/8104) |
| **~0.25 img/s** ("~1 img/4 s") | user report | `nllb-clip-large-siglip__mrl` | [disc #11862](https://github.com/immich-app/immich/discussions/11862) |
| 80 min vs 270 min for the same library | Ryzen 2600 + P600 | `ViT-B-32__laion2b_e16` vs `ViT-B-16-SigLIP-384__webli` | [disc #11862](https://github.com/immich-app/immich/discussions/11862) |
| **20 days for 320k photos** | i7-1355 + OpenVINO | `nllb-clip-large-siglip__v1` | [liutyi wiki](https://wiki.liutyi.info/display/DEVOPS/Immich+CLIP+model+test) |
| **First search after 1 h idle = 60–70 s**, then instant. Collaborator: *"The delay is to load in the machine learning model."* Fix = `PRELOAD__CLIP__*` + `MODEL_TTL=0` | 9,272 photos, 7840HS | — | [disc #14547](https://github.com/immich-app/immich/discussions/14547) |

**~2.26 ms of model vs ~900 ms of wall clock per image.** The causes are all visible in
source (lane-verified 2026-07-22):

- **Batch size is hardcoded to 1** — `models/clip/visual.py::_predict` runs `session.run(...)[0][0]` per image. No CLIP batching anywhere.
- **ORT threads are deliberately throttled** — `sessions/ort.py::_sess_options_default` sets `inter_op_num_threads = 1`, `intra_op_num_threads = 2` for the CPU provider, commented *"avoid thread contention between models"*. Parallelism comes only from job concurrency (**SmartSearch concurrency = 2** by default).
- **The wrong image is decoded** — `getForClipEncoding` feeds the **Preview JPEG (1440 px, q80)**, not the existing 250 px thumbnail, to produce a 224 px tensor.
- **Two-process hop per image** — server reads the file and POSTs it as HTTP multipart to the ML container.
- **f32 only.** No int8 path exists.
- **RAM floor 6 GB** (4 GB only with ML disabled).

**And immich's own code confirms §1.1's brute-force conclusion:** `database.repository.ts`
`targetListCount` returns **`lists = 1`** below **128,000 assets**, with
`probes = ceil(1/8) = 1`. Below 128k rows immich's "vector index" *is* a full scan — it
just pays Postgres, VectorChord and a container to perform it.

### 2.2 photofield + photofield-ai — the speed obsessive (599★ / 30★)

- **Repos:** https://github.com/SmilyOrg/photofield (599★, Go, last push 2026-06-21) · https://github.com/SmilyOrg/photofield-ai (30★, last push 2026-06-07)
- **Stated goal, verbatim:** *"to be as fast or faster than Google Photos on commodity hardware while displaying more photos at the same time"* — the only project in the field whose README states a *speed* thesis. Single binary, non-invasive (doesn't import your files), SQLite cache, zoomable tile renderer, 43k-image demo on an i7-5820K + NVMe.
- **The one honest CPU number in the category:** photofield-ai README — *"~20 req/sec (i7-5820K CPU), ~200 req/sec (GTX 1070 Ti GPU)"*.
- **Model/runtime:** ONNX Runtime; default visual/textual models are `clip-vit-base-patch32-{visual,textual}-float16.onnx` from `mlunar/clip-variants`; `PHOTOFIELD_AI_RUNTIME=cpu|all`; faces = RetinaFace + EdgeFace-XXS. README warns *"the `qint8` models don't seem to work right now, so use `quint8` ones instead"* — i.e. **their int8 path is broken**, a concrete gap.
- **Steal:** single-binary distribution; separate-service ML so the core stays lean; progressive multi-resolution loading; the *"faster than Google Photos"* framing.
- **Beat:** ViT-B/32 float16 in 2026 is a stale model choice (see §3); ML is a second service over HTTP with multipart uploads (per-image HTTP round-trip is pure overhead); 20 img/s on 6 cores is our floor to exceed, not our ceiling.

### 2.3 Lap — the closest architectural twin (1,354★, GPL-3.0)

- **Repo:** https://github.com/julyx10/lap · GPL-3.0-or-later · Rust/Tauri · v0.3.0 · last push 2026-07-21 · **1,354★** · notarized macOS build + Homebrew cask.
- **Verified stack (from `src-tauri/Cargo.toml` + `scripts/download_models.sh`):** Tauri 2 · `rusqlite` (bundled SQLite) · **`ort` 2.0.0-rc.10 (ONNX Runtime)** · `tokenizers` · `ndarray` · `fast_image_resize` · `blake3` · `kamadak-exif` · `jxl-oxide` · `rstar`. Models: **`Xenova/clip-vit-base-patch32` `vision_model_quantized.onnx` + `text_model_quantized.onnx`**, plus InsightFace `buffalo_s` (`det_500m`, `w600k_mbf`) for faces. Similarity is computed in `t_sqlite.rs`/`t_cluster.rs` (cosine in SQL/Rust) — **no ANN index crate**, i.e. brute force, consistent with §1.1.
- **Claims:** "optimized for smooth browsing across libraries with 100k+ files", multilingual search in 50+ languages. **No measured number found** anywhere in README or docs.
- **This is the project IMGTAG most resembles** (local-first, single machine, SQLite, ONNX, brute force) — and it is a *photo manager* first, so its search quality rides on quantized ViT-B/32.
- **Beat:** model tier (quantized ViT-B/32 → MobileCLIP2/SigLIP2 class), published benchmarks (they have none), GPL-3 vs a permissive license, and an agent-facing API/skill (they have none — it is a GUI app).

### 2.4 rclip — the CLI incumbent (979★, MIT)

- **Repo:** https://github.com/yurijmikhalevich/rclip · MIT · Python · v3.2.4 · last push 2026-07-21 · **979★** · snap/AppImage/brew/pip distribution.
- **Verified stack (`pyproject.toml`, `rclip/model.py`, `rclip/main.py`, `model_download.py`):** the model is **`ViT-B-32-256/DataComp-1B` (`ViT-B-32-256-datacomp_s34b_b86k`, MIT, 72.7% ImageNet zero-shot)** — *not* plain OpenAI ViT-B/32 — at `VECTOR_SIZE = 512`. Runtime: **ONNX Runtime CPU everywhere for text + query, CoreML for visual *indexing* on macOS only** (env escape `RCLIP_USE_ONNX_ON_MACOS`), with `intra_op_num_threads` set explicitly from `sched_getaffinity`. Weights are **fp32 only** — `visual.onnx` 351.8 MB + `textual.onnx` 254.3 MB. Preprocessing is hand-rolled **pure PIL+numpy** (torchvision dropped). One global SQLite at `$datadir/rclip/db.sqlite3`, vectors as raw `float32.tobytes()` BLOBs, a `db_version` that **wipes the cache when the model changes**, `(mtime,size)` incremental skip, and an `indexing`-flag sweep for deletions. `MAX_IMAGE_LOADING_WORKERS = 16`, `LOOKAHEAD_BATCHES = 3`, default batch 8. RAW handled by reading the **embedded preview** rather than demosaicing (PR #285: *"more than 3x speedup"*).
- **⚠️ rclip is the real speed bar, and it is far above our provisional budget.** README, verbatim (verified in-source 2026-07-22): *"it took **15 hours to process 84,725 photos** on a NAS with an **old Intel Celeron J3455**, **7 minutes to index 50,000 images** on a **MacBook with an M1 Max**, and **3 hours to process 1.28 million images** on the same MacBook."*
  - Celeron J3455 (4-core, 2016, ~10 W): **1.57 img/s** — pure CPU.
  - M1 Max: **50,000 / 420 s = 119 img/s**, and the 1.28 M run independently gives **118.5 img/s** — the two numbers agree, which is a good sign they are real.
  - Caveat that matters: on macOS rclip's runtime deps include `coremltools`, so the Apple-silicon figure is very likely **CoreML (ANE/GPU-assisted)**, not strictly CPU-only. The honest CPU-only datapoint from rclip is the Celeron's 1.57 img/s.
  - **Implication for BUDGETS.md B1 (≥30 img/s on M3 Max):** an existing MIT CLI already claims ~119 img/s on an M1 Max. B1 as written would ship something **4× slower than the incumbent** on comparable hardware. B1 should be re-derived from a head-to-head rclip run on our machine before it is locked — see §5.
- **Their own release notes give the sharpest numbers of all** ([PR #249](https://github.com/yurijmikhalevich/rclip/pull/249), the 2026-04 switch to ONNX/CoreML):
  - **M1 Max:** indexing *"around 160 img/sec"* → *"180 img/sec (12.5% faster)"*; text search *"~2 sec / 623 MB RAM"* → *"~0.5 s / 672 MB (4× faster)"*; image search 4 s → 0.6 s.
  - **Intel J3455 (Linux, CPU-only):** *"~1.8 img/s"* → *"~1.9 img/s"*; **text search over 80,000 images: 14–15 s, unchanged.**
- **⚠️ The 15-second search is the field's most exploitable bug.** `model.py::compute_similarities_to_text` does `sorted(zip(similarities, range(n)))` — a **full Python sort over every row**, then slices top-k. The matmul itself is ~3 ms at 80k (§1.1). `np.argpartition` alone would make that search ~500× faster. On top of it, rclip **re-reads every vector out of SQLite into a Python list and `np.stack`s them on each invocation** — no mmap, no cached matrix, no daemon. **The category leader in CLI photo search spends 15 seconds doing 3 milliseconds of work.**
- **Steal:** the 16-worker lookahead decode pipeline; the hand-rolled PIL preprocess (small install, fast cold start — they even lazy-load libraw/libheif *just* to shave text-search startup, PR #274); `db_version`-coupled cache invalidation on model change; the `(mtime,size)` + deletion-flag incremental scheme; embedded-RAW-preview decoding; arithmetic/mixed queries; `release_indexing_resources()`; the packaging spread (snap + AppImage + brew + pip).
- **Also weak (open issues, all exploitable):** keyed by **filepath**, so a rename forces a full re-encode ([#10](https://github.com/yurijmikhalevich/rclip/issues/10), open since 2021 — content-hash keying fixes it); **no index-only mode**, you must issue a query to index ([#92](https://github.com/yurijmikhalevich/rclip/issues/92)); one **global** DB in the app data dir rather than beside the photos ([#36](https://github.com/yurijmikhalevich/rclip/issues/36)); no daemon, no watch mode, no server, no live progress.
- **Beat:** ViT-B/32 (69.9% recall by immich's measurement, vs 84.86% available for +3.55 ms); CLI-only (no gallery, no live progress semantics, no agent API); no published *search* latency; Python process start dominates one-shot search (B13 cold start ≤2 s is a real differentiator).

### 2.5 Ente — on-device CLIP done as a product (27,929★)

- **Repo:** https://github.com/ente-io/ente · AGPL-3.0 · Dart · last push 2026-07-22 · **27,929★**
- **Stack:** per Ente's own docs/help, *magic search* runs **MobileCLIP** on-device via **ONNX Runtime**, with YOLOv5Face + MobileFaceNet for faces; indexes are stored locally and E2E-encrypted server-side. They also published [*Image search on the Edge*](https://ente.com/blog/image-search-with-clip-ggml/) describing a **GGML/clip.cpp** path with *"quantisation (up to 4-bit)"*. That post contains **no measured numbers** (verified by fetch).
- **Steal:** MobileCLIP as the on-device model choice (validated at product scale, not just in a paper); desktop-indexes-for-mobile split; the honest "indexing is faster on desktop" UX.
- **Beat:** E2E crypto forces a full re-index on model change (a criticised trade-off in the HN thread); it is a cloud-product architecture, not an embeddable engine; nothing agent-callable.

### 2.6 PhotoPrism (39,991★) and the classic self-hosted set

**PhotoPrism** — https://github.com/photoprism/photoprism · Go · 39,991★ · alive (2026-07-22) · AGPL-3.0 + CLA/trademark + proprietary "Plus" tier.
- **It has no semantic search at all, in 2026.** Issue [#1287 "AI: Add a CLIP-powered semantic search"](https://github.com/photoprism/photoprism/issues/1287) opened **2021-05-12** and is **still open** (~5 years). `internal/ai/vision/model_types.go` enumerates exactly `labels | nsfw | face | caption | generate` — no `embedding`. Search is SQL `LIKE` over keywords a CNN or an LLM wrote (`internal/entity/search/photos.go`; no FULLTEXT, no trigram, no vector) — so "dog" matches "dogma".
- **Models:** NASNet-A Mobile 224 (ImageNet-1k, exactly **1,000 labels**; a 21,843-label file is downloaded and never wired), Yahoo-lineage NSFW, FaceNet 512-d, ONNX SCRFD detector (replaced Pigo 2026-05-24). Runtime **TensorFlow 2.18 C library** (the libtensorflow-1.15 pin finally died 2025-04-25), ~187 MB of models to serve a 1,000-word vocabulary.
- **Their own CLIP plan is DOA by their own acceptance criteria:** default UForm3-multilingual-base ONNX, but *"embeddings stored, CLIP searches not possible"* on SQLite and MariaDB <11.8, and *"Phase 1: no Go fallback scorer"* — i.e. they blocked a 20 MB matmul behind a database vector extension.
- **Measured indexing (community, all URL-cited):** ~**2 photos/s** on a 2017 laptop ([disc #2183](https://github.com/photoprism/photoprism/discussions/2183)); **4–5 s/image** on an i3-6100 with local SSD, **~6 s/image** on a Celeron J4025, **~18 s/image** over CIFS ([disc #4638](https://github.com/photoprism/photoprism/discussions/4638)); **50,000 photos in ~24 h ≈ 1.7 s/photo** with 4 workers ([mustafa.net, 2026-03-06](https://mustafa.net/2026/03/06/photoprism-tips-tricks-every-self-hoster-should-know/)); a reported degradation from ~0.02 s/image at 1k to ~2 s/image at 1.75M, unchanged with TensorFlow disabled — i.e. DB/IO-bound, not model-bound (⚠️ source link `photoprism/discussions/4771` now 404s — **unverified, re-source before quoting**). **Read: ~2–6 s/photo typical ⇒ 10k ≈ 6–17 h.** No official throughput number exists.
- **Their captioning path is unusable on CPU:** own comparison table (Ollama 0.7.0, hardware unstated) — Qwen2.5-VL-7B **36.34 s/image**, 3B **24.12 s/image**, Gemma 3 **31.02 s/image** ⇒ 10k images = 67–100 h.
- **Steal (this is the best idea-mine in the field):** the two-stage thumbnail cascade (original decoded **once**, everything derives from `fit_720`/`fit_1920`); the **3-crop feed** (`tile_224` + `left_224` + `right_224` = free test-time augmentation); the **3×3 "colors" thumbnail** powering `color:`/`mono:`/`chroma:` search at ~0 CPU; the per-hash index mutex; the 0–7 **quality heuristic** with a "Quality Review" triage inbox (zero ML, high perceived intelligence); `rules.yml` threshold/priority rewriting over raw softmax; and **logging the worker count with a reason string** (`4 (sqlite-cap)`, `1 (low-memory)`).

**LibrePhotos** — https://github.com/LibrePhotos/librephotos · **MIT** · 8,005★ · alive (v1.0.3, 2026-06-27). The only other project here with real text→image.
- CLIP **ViT-B/32** via `sentence_transformers`, 512-d, torch, served as per-model Flask sidecars with lazy load + explicit `unload()` + `/health`. Tagging has already moved to **`onnx-community/siglip2-base-patch16-384-ONNX`** zero-shot against a `tags.txt`, with tag embeddings cached to **`tag_embeddings.npy`** — they know torch is the problem and stopped half-way (search still ViT-B/32).
- Retrieval = **faiss-cpu `IndexFlatIP(512)` — exact brute force**, one in-RAM index per user, embeddings stored as a Postgres **JSONField** and rebuilt at boot. Semantic search is **off by default** (`semantic_search_topk = 0`) and *"first search may take up to a minute"*.
- Measured: **77 photos in 6 h** (~13/h) on a 1.2 GHz i5 ([issue #203, 2021](https://github.com/LibrePhotos/librephotos/issues/203)); nothing newer — [a 2025 "system requirements" issue closed unanswered](https://github.com/LibrePhotos/librephotos/issues/1515).
- **Steal:** semantic **∪** keyword fusion (a bad embedding still degrades to text); the declarative model registry with user-selectable accuracy tiers; the precomputed tag-embedding cache.

**Photonix** — https://github.com/photonixapp/photonix · AGPL-3.0 · 1,953★. Dead 2023–2025 (**1 commit in 2023, 0 in 2024, 0 in 2025**), revived January 2026 (**195 commits**, AI-assisted solo rewrite — the repo now carries a `CLAUDE.md`). **Stable channel is still v0.24.0 from 2021-11-18** — `docker pull …:latest` gets a 4.5-year-old build. SSD-MobileNetV2 (Open Images v4) + MTCNN/FaceNet; **no CLIP, no embeddings, no text→image** (grep `clip|embed|semantic` = 0 hits). **No CPU number found.**
- **Steal:** **memory-gated model loading** — check `psutil.virtual_memory().available` against a per-model `approx_ram_mb` + a 500 MB buffer, then *requeue* instead of OOM-killing; and the zero-ML classifiers (dominant-colour palette, EXIF→event, offline reverse-geocode) that buy real search facets for ~0 CPU.

**Nextcloud Recognize** — https://github.com/nextcloud/recognize · AGPL-3.0 · 692★ · alive (v12.0.0). EfficientNetV2 / EfficientNet-Lite4 + `@vladmandic/face-api` (128-d), **tfjs-node or WASM**, PHP spawning a Node subprocess per batch, **`setInterval(60*5)` — a 5-minute cron cadence**. **No CLIP, no text→image.** Requirements: *"~4 GB of free RAM"*, *"we recommend 10-20 cores"*. Field datapoint: **10k–20k photos/day ≈ 0.12–0.23 img/s** ([HN 44426233](https://news.ycombinator.com/item?id=44426233)); a 2022 report of 5 node processes, >6 GB swap and load >50 on a 4 GiB box ([issue #596](https://github.com/nextcloud/recognize/issues/596)).
- **Steal:** the **degraded-tier fallback** (no AVX ⇒ swap to a smaller model, smaller input, smaller batch — never hard-fail); per-image **timeout budgets** so one poison file cannot stall a library; `rules.yml` class→tag rewriting with aliasing.

**Common weakness of the classic set:** fixed label taxonomies rather than open-vocabulary
embeddings, so they *structurally cannot* satisfy the vision's `car → vehicle → motorcycle`
hypernym requirement (B5). Where CLIP does exist it is bolted onto Django/Postgres/Docker.

### 2.6.1 Self-hosted scoreboard (verified 2026-07-22)

| | immich | PhotoPrism | LibrePhotos | Photonix | NC Recognize |
|---|---|---|---|---|---|
| Stars | 108,427 | 39,991 | 8,005 | 1,953 | 692 |
| License | AGPL-3.0 | AGPL-3.0 + proprietary Plus | **MIT** | AGPL-3.0 | AGPL-3.0 |
| Alive | v3.0.3 (2026-07-15) | 2026-06-01 | v1.0.3 (2026-06-27) | code yes / **releases stuck at 2021** | v12.0.0 (2026-04-07) |
| **Text→image search** | **YES** (`ViT-B-32__openai`) | **NO** — open 5 yrs | **YES** (off by default) | **NO** | **NO** |
| Runtime | ONNX Runtime | TF 2.18 C (+ORT for faces) | torch (+ORT for tags) | TF 2.21 | tfjs-node / WASM |
| Vector store | Postgres + VectorChord | none | JSONField + faiss `IndexFlatIP` | annoy (faces only) | none |
| Index at 10k | **`lists=1, probes=1` = full scan** | n/a | exact brute force | n/a | n/a |
| Precision | f32 only | f32 | f32 | f32 | f32 |
| Image batch | **1** | n/a | 1 | 64 | 100 |
| Best CPU 10k estimate | ~2.5 h (N100) | 6–17 h | no 2025-26 number | none found | none first-party |
| Cold search | **60–70 s** after idle | n/a (SQL) | "up to a minute" | n/a | 5-min cron |
| RAM floor | 6 GB | 3 GB | 4 GB | ~2.6 GB | 4 GB |

### 2.7 The 2024–2026 indie wave (all feature-list, no numbers)

| Project | Link | Stars / status | Stack | Numbers |
|---|---|---|---|---|
| Desktop Docs (commercial) | [HN 44118023](https://news.ycombinator.com/item?id=44118023) · desktopdocs.com | 597 pts HN, Mac-only | Rust + Tauri + **`ort` crate, ONNX Runtime bundled**; rewrite cut app 1 GB → ~172 MB | **no measured number found** (dev explicitly gave none in-thread); biggest reported pain = *"bundling the onnxruntime into the app and making sure everything was signed"* |
| PixFinder | https://pixfinder.app/ | HN 2026-07-15 | "SigLip2 + OCR", Windows, 100% offline | **no measured number found** (fetched: zero specs) |
| Facet | https://github.com/ncoevoet/facet | 169★, alive today | Python/FastAPI + Angular; 9-axis scoring, zero-shot tags, semantic search, CPU-only supported | no measured number found |
| harperreed/photo-similarity-search | https://github.com/harperreed/photo-similarity-search | 502★, last push 2024-05 | MLX (Apple Silicon) CLIP | no measured number found |
| CLIP-Finder2 | https://github.com/fguzman82/CLIP-Finder2 | 95★, 2024 | **MobileCLIP-S0 → CoreML on the Neural Engine**, MPSGraph preprocessing + dot-product, Core Data store, **batch size 512**, incremental add/delete on launch | latency "consistent with the MobileCLIP paper" (no independent numbers) |
| haltakov/natural-language-image-search | https://github.com/haltakov/natural-language-image-search | 1,041★, **dead since 2022-10** | Unsplash + CLIP notebook | n/a |
| ldqk/ImageSearch | https://github.com/ldqk/ImageSearch | 1,317★, alive | .NET 10 desktop, **perceptual-hash** reverse-image (not semantic text), integrates with *Everything* for directory scanning | claims *"千万级图片秒级检索"* (10M images, second-scale retrieval) — vendor claim, unverified, and image→image only |
| memery | https://github.com/deepfates/memery | 576★, last push 2026-07-04 | CLIP + notebook-era Python | no measured number found |
| clip-retrieval | https://github.com/rom1504/clip-retrieval | 2,787★, last push 2026-03-28 | inference worker + autofaiss + knn service (the LAION pipeline) | GPU-scale throughput claims; not a CPU/desktop design |
| CLIP-as-service (Jina) | https://github.com/jina-ai/clip-as-service | 12,831★, **last push 2024-01-23 → effectively dormant** | gRPC/HTTP CLIP service | n/a |

**Pattern:** the indie wave keeps re-implementing the same three-part recipe (CLIP ONNX +
SQLite + brute force) and keeps not measuring it. Nobody has combined *modern model* +
*measured speed* + *agent API* + *live-progress operations*.

### 2.7.1 Native CPU-first engines (Rust / C++) — the technical frontier

**`open_clip_inference` (RuurdBijlsma)** — canonical repo https://github.com/RuurdBijlsma/open-clip-inference-rs (the `open-clip-rs` path redirects) · MIT · last push 2026-04-26 · **5★** · crate `open_clip_inference` v0.4.0 (tiny, but the single most relevant benchmark table found). Rust + `ort` (ONNX Runtime), CPU by default, auto-downloads pre-converted ONNX by HF id. Published table (⚠️ *"measured on my CPU"* — **CPU model not stated**; vision times include 10–20 ms preprocessing):

| Model | ImageNet zero-shot | vision ms/img | text ms |
|---|---:|---:|---:|
| ViT-gopt-16-SigLIP2-384 | 85.0% | 2354 | 128 |
| ViT-SO400M-16-SigLIP2-384 | 84.1% | 988 | 136 |
| **MobileCLIP2-S3** | **80.7%** | **116** | 35 |
| MobileCLIP2-S4 | 79.4% | 192 | 38 |
| **MobileCLIP2-S2** | **77.2%** | **75** | **19** |

The author separately verified embedding equality against the Python reference
([clip-model-research](https://github.com/RuurdBijlsma/clip-model-research)) — the
preprocessing-parity discipline §4 recommends, already done by a 5-star repo and by nobody
with 100k stars.

**`clip.cpp` (monatis)** — https://github.com/monatis/clip.cpp · MIT · **564★** · last push 2025-06-19 (13 months stale). ggml, quant q4_0…f32, zero dependencies, stdlib-only Python binding, `examples/image-search` backed by **usearch**. **4-bit CLIP = 85.6 MB.** Measured in [PR #57](https://github.com/monatis/clip.cpp/pull/57) (ViT-B-32 laion2b f16 over ImageNet-1k, **hardware not stated**): *"47904 images encoded in 5594410.00 ms (116.78 ms per image)"*.
- **Quality red flag, same PR:** ImageNet top-1 measured **0.0805 → 0.3137** after a `ggml_conv_2d` batch fix, against open_clip's reference **66.6%** for the same weights. Maintainer attributes the gap to test protocol, tokenizer, and **linear-instead-of-bicubic resize**. **Steal the architecture; do not inherit the numerics.** This is the strongest single argument for the preprocessing-parity gate in §4.

**`osmarks/meme-search-engine`** — https://github.com/osmarks/meme-search-engine · MIT · **115★** · Rust; SQLite holds images *and* vectors; in-memory FAISS *"scales to perhaps ~1e5 items"*, DiskANN beyond. Author's write-up ([osmarks.net/memescale](https://osmarks.net/memescale/), hardware stated: Ryzen 5 5500 / 64 GB / RTX 3090) reports 230 M images, ~2 kB per embedding, dot-product **~180 ns → ~60 ns** at 1152-d after optimisation, graph build *"slightly over six days"*. Steal: online reindex without a full rebuild; weighted multi-term ± queries.

**Edge ports worth copying (mobile, but the engineering transfers):**
- **Queryable** (iOS, Swift, MIT, 2,969★) — MobileCLIP-S2 CoreML with **split text/image encoder files** so the text tower alone loads at query time. Author's measurements ([mazzzystar.com](https://mazzzystar.com/2022/12/29/Run-CLIP-on-iPhone-to-Search-Photos/)): *"about **2000 photos per minute** on an iPhone 12 mini"* (≈33 img/s **on a phone**); *"For fewer than 10,000 photos, it takes **less than 1 second**"*; *"35,000 photos … about **2.8 seconds**"*; 300 MB of models, 512-d.
- **PicQuery** (Android, Kotlin, MIT, 505★) — **int8-quantized** CLIP/MobileCLIP as `.ort` (ONNX Runtime Mobile); README: *"Show results in less than 1 second when searching for 8,000+ photos"*.
- **apple/ml-mobileclip** — latency column is **iPhone 12 Pro Max / ANE**, not CPU: S0 1.5 ms img + 3.3 ms txt (IN 71.5%), S2 3.6+3.3 (77.2%), S3 8.0+6.6 (80.7%). Use `open-clip-rs` for CPU truth.

**`Eventual-Inc/local-image-search`** — https://github.com/Eventual-Inc/local-image-search · **9★** · last push 2026-01-29 · **no licence file** · *"Privacy-first, 100% local MCP server for macOS. Uses MLX CLIP for embeddings, Daft for batch processing, and Lance for vector storage."* **The one agent-facing local image search found in the entire survey** — and it is macOS-only, GPU-only (MLX never touches the ANE), unlicensed, and 9 stars. Measured by a sibling lane's sources: **260 img/s on an M4 Max** (11,843 images in ~39 s), with *scanning* taking 26 s against 39 s of embedding — i.e. **decode/IO is ~40% of its wall clock**, independently confirming §1.2. This is the closest thing to a competitor for IMGTAG's agent-skill deliverable, and the bar it sets is low everywhere except raw throughput.

**Go:** nothing serious and alive in semantic image search beyond photofield (which delegates ML to a Python sidecar). The best Go hits are ≤6★ toys. **An open field.**

### 2.7.2 screenpipe — the dog that did not bark

https://github.com/screenpipe/screenpipe (moved from `mediar-ai/`) · **20,379★** · alive
2026-07-22 · **license flipped to a proprietary "Screenpipe Commercial License" (2026-06-10)
— design is stealable, code is not.** Verified by code search: `moondream` **0 hits**,
`florence` **0 hits**; the only `embedding` code is *audio speaker diarization*. Its vision
path is OCR + accessibility-tree text into SQLite FTS; the one vision-ML crate
(`screenpipe-rfdetr-mlx`) is **MLX/Apple-GPU only and excluded from the workspace**.
Published resource envelope (README): *"5-10% cpu usage · 0.5-3gb ram · ~20gb storage/month"*.
**Lesson:** the largest always-on capture product in the world deliberately avoided
CLIP-over-pixels because text extraction is cheaper — which leaves CPU semantic-over-pixels
at 10k images genuinely unoccupied.

### 2.8 The infrastructure everyone builds on (verified alive)

`usearch` 4,228★ · `sqlite-vec` 7,919★ (last push 2026-05-18) · `hnswlib` 5,288★ ·
`faiss` 40,565★ · `lancedb` 10,954★ · `qdrant` 33,489★ · `pgvector` 22,303★ ·
`VectorChord` 1,747★ (the pgvecto.rs successor — **pgvecto.rs itself last pushed
2025-02-26, i.e. abandoned**) · `candle` 20,702★ · `tract` 3,002★ · `ort` (Rust ORT) ·
`onnxruntime` 21,160★ · `openvino` 10,561★ · `mlx` 27,645★ · `libvips` 11,508★.
Per §1.1, at 10k–100k images **none of the vector databases are required**.

### 2.9 Lane provenance

Four parallel research lanes fed this file, plus one sibling lane's overlap. **Merged:**
the self-hosted-manager lane (§2.1, §2.6, §2.6.1); the native/HN/vector-index lane (§1.1
cross-check, §2.7.1, §2.7.2, §2.10); the CLIP-tooling lane (§2.4 internals, §2.11 licences,
§2.12 taggers, §2.13, plus the measured int8 numbers in §3 row 0b); and the Apple-runtime
sibling lane's prior-art overlap (rclip PR #249, `local-image-search`, immich's lack of any
macOS acceleration path); and the model/runtime-benchmark lane (§2.14–§2.18 — the MobileCLIP
CPU inversion, immich field throughput + FLOPs span, quantisation evidence, decode routing,
ANN recall collapse). **All four lanes plus the sibling overlap are merged**; §5.8 lists the
eight experiments they collectively proved nobody in this field has run.
**Coverage caveat, stated honestly:** the session-wide WebSearch budget (200 calls) was
exhausted mid-lane, so late verification leaned on the GitHub REST API, direct source
reads, and WebFetch. Every claim in this file is sourced; the *breadth* of the last two
lanes may be thinner than the first two.

---

### 2.10 What the users actually demand (Hacker News, 2024–2026)

The field's *requirements document* is written in HN comments, and it maps almost perfectly
onto VISION.md. Verbatim, with links:

- **No Docker, no server — one double-clickable thing.** *"can it please just be an exe I can double click … If it involves maintaining a server or faffing about with Docker I'm probably not going to bother."* ([44430416](https://news.ycombinator.com/item?id=44430416))
- **Publish benchmarks or be disbelieved.** *"You mention performance in this post but not at all on the landing page … Would want benchmarks on various machines"* ([44119032](https://news.ycombinator.com/item?id=44119032))
- **The model-upgrade tax is the known killer.** *"What happens if there's a new, better model? You'd need to re-download, decrypt, and run inference on all your past media"* ([44438033](https://news.ycombinator.com/item?id=44438033)) — and immich's docs confirm a model change means reprocessing everything; one user notes re-indexing **deletes old data first**, leaving search incomplete for days ([44426745](https://news.ycombinator.com/item?id=44426745)).
- **CLIP quality is openly contested.** Against: *"Embeddings seem really poor, and has lots of misses and false positives"* ([44429763](https://news.ycombinator.com/item?id=44429763)); *"CLIP is out of date, to put it mildly"* ([44118572](https://news.ycombinator.com/item?id=44118572)). For: photofield's author — *"For semantic search CLIP and cosine similarity are just fine … Vector DBs are cool, but what is cooler is **writing float arrays to sqlite**"* ([44429236](https://news.ycombinator.com/item?id=44429236)). Both sides are satisfied by *a modern encoder + a flat float array*, which is exactly §4's plan.
- **Indexing must not eat the machine.** Nobody in the corpus runs indexing at low OS priority; users ask for it ([48419626](https://news.ycombinator.com/item?id=48419626) — *"Fans go crazy"*). Free differentiator, ~10 lines of `nice`/QoS.
- **Slow local tools get destroyed in public.** Show HN "Sisi" (node-mlx CLIP, 580★, dead since 2024-09): *"Uses only 1 core 100% under linux, 10 images, each ~20 kb size, took more than 10 minutes to index"* → *"Wow that's atrocious performance. So there's no chance to use this on real photos"* ([41555976](https://news.ycombinator.com/item?id=41555976), [41559075](https://news.ycombinator.com/item?id=41559075)). Performance *is* the product in this category.
- **The GPU crutch is the incumbent's admission.** Desktop Docs' author, in his own Show HN: *"Desktop Docs needs a GPU to work well"* ([44118740](https://news.ycombinator.com/item?id=44118740)).

**Negative result worth not repeating:** VLM-captioning-as-index. Smart Photo Finder's own
README table: 10 images = 3–5 min, 50 = 15–20 min, 200 = 1–2 h on CPU (**~18–36 s/image**)
— three orders of magnitude off CLIP's 2–116 ms. PhotoPrism's own caption benchmark agrees
(24–36 s/image). **Captioning is an opt-in enrichment for selected images, never the index
path.** Likewise RAM++/recognize-anything (3,690★, stale since 2025-02, no CPU number
published anywhere) and YOLO-World (6,475★, stale, **GPL-3.0**, latency published only for a
T4 GPU) are not CPU candidates.

### 2.11 ⚠️ The licensing minefield — read before choosing a model

Several of the fastest, best-looking options **cannot legally ship** in a distributed tool.
Verified against the licence files themselves:

| Model family | Licence | Usable in a shipped IMGTAG? |
|---|---|---|
| **MobileCLIP / MobileCLIP2 weights** (Apple) | code MIT, **weights `apple-amlr`** — [LICENSE_MODELS](https://raw.githubusercontent.com/apple/ml-mobileclip/main/LICENSE_MODELS) verbatim: *"exclusively for Research Purposes… does not include any commercial exploitation"* | ❌ **research only** — this disqualifies the otherwise-attractive MobileCLIP2-S2 (§2.7.1) for anything shipped |
| **jina-clip-v2**, jina-embeddings-v3/v4/v5 | **CC-BY-NC-4.0** | ❌ non-commercial |
| jina-clip-**v1** | Apache-2.0 | ✅ (but no better than OpenCLIP/SigLIP2 at its size) |
| **SigLIP / SigLIP2** (`onnx-community/siglip2-*`) | **Apache-2.0** | ✅ |
| **OpenCLIP** checkpoints incl. `ViT-B-32-256/DataComp-1B` | MIT | ✅ |
| WD/wd-tagger v3 family | Apache-2.0 | ✅ (but see §2.12 — wrong tool) |
| YOLO-World | **GPL-3.0** | ⚠️ viral |

**Consequence:** the ship-safe modern-quality pick is the **SigLIP2 family, int8 ONNX**
(`onnx-community/siglip2-base-patch32-256-ONNX`, vision int8 **95.9 MB**, Apache-2.0,
immich-measured sibling **3.31 ms / 82.28 recall**). MobileCLIP numbers stay in this report
as *evidence about what small architectures can do*, not as a shippable candidate. Also
note: Jina AI was acquired by Elastic (2025-10-09) and **clip-as-service's last real commit
was 2023-12-20** with `onnxruntime<=1.13.1` pinned on macOS — it will not install on a 2026
stack. Do not adopt it.

### 2.12 Taggers are the other paradigm — and they lose on CPU

- **WD/wd-tagger (SmilingWolf)** — Apache-2.0, 12 models on HF, **newest is `wd-eva02-large-tagger-v3`, last modified 2024-07-28; nothing in 2025 or 2026**. Vocabulary counted, not guessed: v3 `selected_tags.csv` = **10,861 rows** (4 rating + 8,106 general + **2,751 anime character names**), with a thin real-world tail (`dog` 24,867 · `car` 16,519 · `pizza` 4,153) and no `photo_(medium)` concept. Macro-F1: vit-v3 0.4402, swinv2-v3 0.4541, eva02-large-v3 0.4772. ONNX exports are **fp32 only**, 448² NHWC. **Rigorous CPU img/s with stated hardware: no measured number found** anywhere in the ecosystem.
- **DeepDanbooru** — MIT, but functionally frozen: last code commit 2024-08-27, weights frozen at **v3-20211112**. Strictly dominated.
- **The cost gap is structural, ~90–100× per image**: a 448² backbone with a 10,861-way sigmoid head in fp32, versus a 224²/256² transformer that quantises cleanly.
- **The good idea instead:** run a fixed tag vocabulary through the **text tower once at build time** (a few thousand phrases = seconds) and dot it against the image embeddings you already have. Facets, autocomplete, clustering and "tags" for free — no second model, no anime prior, no extra pass. This is the cheapest feature-per-CPU-cycle idea found in the entire survey.
- (General-domain **RAM++ / recognize-anything** — Apache-2.0, 4,585 tags — exists but is `.pth`-only with **no official ONNX** and no published CPU number, and has been stale since 2025-02.)

### 2.13 Three more competitors worth naming

- **sist2** — https://github.com/sist2app/sist2 · **1,276★** · GPL-3.0 · last push **2025-07-05** (going stale). Architecturally the closest twin found: SQLite backend with embeddings searched **brute-force, documented in its own README as O(n)**, with Elasticsearch as the O(log n) alternative. Independent confirmation that flat scan is the sane default at this scale.
- **Eagle** (closed source, commercial) — offline "AI Search" plugin with text and reverse-image search. The only number they publish is a GPU comparison: *"overall processing speed can be up to 30–50× faster than CPU-only mode"* ([blog](https://en.eagle.cool/blog/post/eagle-plugin-ai-search)). Model, indexing rate and RAM are **not disclosed** — the proprietary end of the field is even more opaque than the open end.
- **Hydrus Network** (3,132★) and **digiKam** (8.8.0, 2025-10-19): the brief's assumption of a "Hydrus CLIP addon" **could not be verified** — grepping Hydrus's changelog for `clip|semantic|embedding` returns zero hits; it does perceptual-hash duplicates and DeepDanbooru-style tagging. digiKam ships OpenCV DNN faces and HAAR similarity, **no CLIP**; semantic search is an open request (KDE bug 497938). ("Lykos" turned out to be Stability Matrix, a Stable-Diffusion launcher, not a tagger; no local-CLIP project could be found under "Hyperspace"/"Sharkey".) **Two whole desktop ecosystems have no semantic search at all.**

### 2.14 ⚠️ The MobileCLIP trap — correcting an earlier section of this file

§2.7.1 quotes open-clip-inference-rs putting MobileCLIP2-S2 at 75 ms/img, the best
quality-per-ms in that table. **On x86 CPU that ranking inverts.** Measured, PyTorch,
10,000-iteration average, [HF discussion](https://huggingface.co/apple/MobileCLIP-S2-OpenCLIP/discussions/3):

| Model | i7-12700K (CPU) | RTX 3090 |
|---|---:|---:|
| MobileCLIP-S2 | **170.62 ms/frame** | 18.91 ms |
| ViT-B-32-256 (OpenCLIP) | **114.05 ms/frame** | 6.31 ms |

**MobileCLIP-S2 is 1.5× slower than plain ViT-B-32-256 on a desktop CPU.** Apple's own
reply in that thread: *"We benchmarked our models on the neural engine of the iPhone 12 Pro
Max using Core ML."* Their famous "1.5 ms" is **iPhone 12 Pro Max, Core ML, batch 1, ANE** —
not a CPU number, and the architecture (depthwise/grouped convs, FastViT) is tuned for
exactly that. Worse for us: the MobileCLIP2 paper concedes its DFNDR-2B pretraining
*"[does] not always achieve state-of-the-art retrieval performance"* because the dataset is
biased toward ImageNet classification, while *"models trained on DataComp, WebLI, and their
derivatives may achieve higher retrieval performance."* **IMGTAG is a retrieval product.**
Verdict: MobileCLIP is a mobile-ANE story. Correct only if we ship an **fp16 CoreML** path.

**And that path is wide open.** rclip's CoreML export uses
`compute_precision=ct.precision.FLOAT32` — **ANE is fp16-only, so rclip never engages it**,
yet still hits 180 img/s on an M1 Max. Nobody in this survey has measured an fp16 CoreML
CLIP export. Largest unclaimed differentiator found.

### 2.15 More immich field throughput — the "ordinary old CPU" reality

| Hardware | Model | Rate | 10k projection |
|---|---|---:|---:|
| Ryzen 2600 + P600, 32 GB | `ViT-B-32__laion2b_e16` | **16.7 img/s** (80k in 80 min) | 10 min |
| same | `ViT-B-16-SigLIP-384__webli` | **4.9 img/s** (80k in 270 min) | 34 min |
| **Synology DS920+ (Celeron J4125)**, concurrency 4 | `ViT-B-16-SigLIP2__webli` | **~2 img/s**, ~1.45 GB | **83 min** |
| same | `ViT-SO400M-16-SigLIP2-384__webli` | **~0.17 img/s**, ~7 GB | **16 h** |
| i5-10500 | `nllb-clip-large-siglip__v1` | 0.40–0.44 img/s | ~7 h |

Sources: immich [disc #11862](https://github.com/immich-app/immich/discussions/11862),
[disc #17135](https://github.com/immich-app/immich/discussions/17135), [disc #8104](https://github.com/immich-app/immich/discussions/8104).
**~10× spread from model choice alone on one box** — and `MaxBatchSize` in immich covers
only `facial_recognition` and `ocr`, never CLIP (dev, 2024-03-21: *"Inputs to the GPU aren't
batched right now"*). FLOPs frame the whole spread (open_clip `model_profile.csv`):
ViT-B/32 **8.82 GFLOPs** → ViT-B/16 35.13 → ViT-L/14 162.03 → ViT-H-14-378 **1006.96** =
**114× span**. Also note SigLIP2's **256,000-token Gemma vocab** (~197M params in the text
embedding table alone) — that is why its RSS is ~3 GB despite an 86M-param vision tower, and
it is a real cost for a resident interactive text encoder.

### 2.16 Quantization — the evidence, split into two very different decisions

**Model quantization (risky, and nobody ships it for CLIP image towers):**

| Method | Model | Quality delta | Speed |
|---|---|---|---|
| naive `quantize_dynamic` ([arXiv 2605.26415](https://arxiv.org/html/2605.26415v1), 2026-05) | CLIP ViT-B/32 | 63.3% → **58.72%** IN-1k (**−4.6 pp**) | — |
| DeepSparse sparse-int8 (clip-retrieval) | ViT-B/32 | 72.8% → **71.1%** (−1.7 pp) | **1230 img/s**, 64-core AVX-512 VNNI Xeon, **2.84×** over fp32 |
| OpenVINO NNCF PTQ | CLIP-B/16 | — | 285→144 MB, **1.64×** |
| OpenVINO NNCF PTQ | SigLIP | — | 387→201 MB, **1.91×** |

That paper also documents **QIRC** (quantization-induced representation collapse): layer-wise
noise-to-signal climbing *"from below 10% in shallow blocks to 52% at Layer 11"* — CLIP
degrades differently from CNNs. ORT's own docs warn *"it is not rare to get worse performance
on old devices"* — our exact target — and ViT's conv patch-embed is the part that quantizes
badly ([ORT #12925](https://github.com/microsoft/onnxruntime/issues/12925)). **Ecosystem
signal: immich's 64 HF model repos are all fp32; Ente quantizes only its text tower.**
(This tempers §1.3's measured 157.9 img/s int8 result: the *speed* is real and measured here,
the *recall cost* is not yet measured here — B6/B7 must gate it.)

**Embedding quantization (safe, post-hoc, and unclaimed):** [HF/mixedbread](https://github.com/huggingface/blog/blob/main/embedding-quantization.md)
— int8 = 4× memory, **~99.3%** performance, 3.66× mean CPU speedup; binary = 32× memory,
**~96%** *with* rescoring (92.5% without), 24.76× mean speedup. Recipe: binary Hamming top-(k×4)
→ rescore with f32 query × int8 docs. Matryoshka truncation is similarly cheap where the model
supports it (jina-clip-v2 1024→256 = *">99% of retrieval quality"*; nomic 768→256 = 98.0%).
⚠️ Do **not** stack binary with aggressive MRL: 512-bit 90.8% → 256-bit 79.5% → 128-bit 60.3%.
⚠️ **All of these retention numbers are TEXT embeddings (MTEB/BEIR). No CLIP-image equivalent
exists — measure it ourselves before relying on it.**

### 2.17 Decode — route by size; the folk wisdom is wrong at small sizes

[ternaus/imread_benchmark](https://github.com/ternaus/imread_benchmark) (run 2026-05-20, 50k
ImageNet JPEGs ~500×400, single thread, decode only, img/s):

| library | EPYC 9B14 | Xeon Plat 8581C | Neoverse-V2 |
|---|---:|---:|---:|
| simplejpeg | **690** | **735** | **662** |
| opencv | 664 | 721 | 645 |
| pillow | 537 | 577 | 551 |
| **pyvips** | **420** | **462** | **413** |

**pyvips is the slowest mainstream option on small images** — per-op pipeline overhead
dominates. It wins only when the image is large *and* shrink-on-load applies:
`vipsthumbnail` on a 10,000×10,000 JPEG is **0.317 s** with shrink-on-load versus **4.660 s**
with `--linear` (which disables it) — **14.7×**, though `--linear` also forces float
linear-light math, so treat it as an upper bound. On 6000×4000 JPEGs libvips does ~4 ms/img
threaded vs ImageMagick's ~70 ms. Apple Silicon reference (M4 Max, [arXiv 2501.13131](https://arxiv.org/html/2501.13131v1)):
OpenCV 1016 img/s, Pillow 775 — and **Pillow-SIMD is dead** (last release 9.5.0.post2, based
on Pillow 9.5, SSE4/AVX2 x86 only — useless on ARM). **Pillow does not release the GIL during
decode** ([#2635](https://github.com/python-pillow/Pillow/issues/2635)) → multiprocess, ~3.4×
at 8 workers, or let libvips thread itself via `VIPS_CONCURRENCY`.
🔴 **Nobody has ever published a `PIL.Image.draft()` or `tjbench -scale` speedup measurement.**
§1.2's 1.75× is, as far as this survey found, **the only measured number in existence** for
that lever. Publish it.

### 2.18 ANN recall collapses earlier than folklore says

[vectorlite benchmark](https://github.com/1yefuwang1/vectorlite/blob/main/README.md)
(i5-12600KF, WSL, **20,000 vectors**, ef=100): recall **85.7% at 128-d**, **59.5% at 1536-d**.
Insert cost: brute force **2.73 µs/vec** vs HNSW **820.8 µs/vec (300×)** → a 20k HNSW build is
~16 s against 0.055 s. At 1M, a serial pgvector HNSW build took **1h27m**
([Supabase](https://supabase.com/blog/pgvector-fast-builds)). faiss's own guidelines: *"If you
plan to perform only a few searches (say 1000-10000), the index building time will not be
amortized"* → use `Flat`. **The escalation ladder, in order: fp32 flat → MRL truncation →
binary + rescore → HNSW last.**

## 3. To beat the world — the bar, in numbers

Every entry is a real, sourced number an incumbent has to defend, mapped to a BUDGETS.md row.

| # | The bar to exceed | Where it comes from | IMGTAG budget |
|---|---|---|---|
| **0** | **⚠️ rclip: 180 img/s on an M1 Max** (PR #249, CoreML batch 8; its README's older PyTorch-era figure was 119 img/s) and **1.9 img/s on a 2016 Celeron J3455** (pure CPU). Plus `local-image-search`: **260 img/s on an M4 Max** (MLX, GPU) | rclip [PR #249](https://github.com/yurijmikhalevich/rclip/pull/249) + README (§2.4); §2.7.1 | **B1 (≥30 img/s) is 6× below the incumbent on comparable hardware.** Note both leaders reach those rates via **CoreML/MLX accelerators**, which a strictly CPU-only reading of the vision excludes — so the honest CPU-only bar is lower, and IMGTAG must state which game it is playing. Either way, measure rclip head-to-head before locking B1 (§5) |
| **0b** | **157.9 img/s measured on an M3 Max, CPU-only** — CLIP-B/32 vision **int8 ONNX, batch 8, 6.33 ms/img** (fp32 batch 8 = 57 img/s; int8 batch 1 = 125 img/s) | sibling lane's direct measurement on this machine (⚠️ taken under load — a *lower* bound) | **This is the most important number in the file: ~158 img/s CPU-only makes B1's ≥30 img/s look timid, and puts "10,000 images in ~63 seconds" within reach on CPU alone.** int8 also cuts the model 352 MB → 89 MB (B9) |
| 1 | **20 img/s CPU indexing on a 6-core 2014 i7** | photofield-ai README (measured) | The edge floor ⌂ ≥5 img/s is too soft — a 2014 6-core already does 20 |
| 2 | **~2 img/s per core** (325 img/s ÷ 160 cores, UForm-small ONNX, batch 128) | UForm BENCHMARKS.md | A *scaling sanity check* on row 0b: naive per-core extrapolation gives ~32 img/s on 16 cores, yet the direct int8 measurement here is 157.9 img/s — the difference is precision and batching, which is precisely the lever nobody else pulls. Target **≥150 img/s** on the dev target, ⌂ ≥10 img/s on the edge floor |
| 3 | **The self-hosted status quo is ~0.2–2 img/s**: immich **1.11 img/s** (N100, CPU-only), immich **0.4 img/s** (i5-10500), PhotoPrism **0.17–0.6 img/s** (2–6 s/photo), Recognize **0.12–0.23 img/s** | §2.1.2, §2.6 | B1's ≥30 img/s is a **15–250× jump over every self-hosted incumbent**. This is the headline claim, and it is defensible because their bottlenecks are structural (batch=1, 2 threads, 1440 px decode, HTTP hop), not physical |
| 4 | **10k images fully indexed** — best documented CPU-only comparators: immich ≈2.5 h (N100), PhotoPrism 6–17 h, rclip ≈1.8 h (Celeron J3455), photofield-ai ≈8m20s (i7-5820K) | §2.1.2, §2.4, §2.6, §0 | B2 says ≤6 min. Row 0b's measured 157.9 img/s implies **~63 s of encode** for 10k, so B2 is achievable *if decode keeps up* (§1.2) — **a sub-2-minute, CPU-only, end-to-end 10k index would be the fastest published number in this field by a wide margin** |
| 5 | **Search latency: nobody publishes one — and the leader's cold search is 60–70 s** | [immich disc #14547](https://github.com/immich-app/immich/discussions/14547); LibrePhotos "first search may take up to a minute" | B3 p50 ≤50 ms / B4 p95 ≤150 ms keystroke→painted. §1.1 proves the scan is 0.47 ms; the *only* ways to lose are text-encode and cold start. **A warm resident text tower turns the category's worst wound into our headline demo** |
| 6 | **Model size 85.6 MB (4-bit CLIP, clip.cpp)** | clip.cpp README | B9 ≤150 MB is comfortable; aim to *beat 85.6 MB while beating ViT-B/32 quality* |
| 7 | **Quality: "poor embeddings, misses and false positives" (immich, user)** | HN 44426233 | B5/B6/B7 are the differentiator — the incumbents ship no quality metrics at all |
| 8 | **Retrieval recall 69.9%** (immich default `ViT-B-32__openai`, Crossmodal-3600 avg of R@1/5/10) — and **85.99%** for the best model anyone in this field ships | immich docs table (§2.1.1) | Shipping a **≥82–85%-recall model *by default*** puts IMGTAG 12–16 recall points above the category leader's out-of-box behaviour on day one |
| 9 | **Model inference 2.26 ms (B-32) / 5.81 ms (B-16-SigLIP2) per image on a 7800X3D** | immich docs table | ⇒ inference alone allows >170 img/s single-stream. Anyone indexing at 20 img/s is losing ~90% of the time somewhere else (decode, I/O, HTTP, Python). **That gap is the prize.** |
| 10 | **~3 GB peak RSS for SigLIP2 at f32** | immich docs table | B8 caps peak at 1.5 GB ⇒ we must land modern-tier quality *inside half the memory immich needs*. No competitor has published a quantised measurement — uncontested ground |

## 4. The twelve moves that make IMGTAG objectively better

Ranked by (leverage × how few competitors do it). Each is grounded in a specific finding above.

1. **Publish the benchmark nobody else has.** `imgtag bench {index,search,quality,resources,soak}` producing a hardware-labelled table, plus a public results file. Cost: near zero. Effect: instantly the most credible project in §2. Nobody — not immich, not Lap, not rclip — can currently answer "how many images per second?" (§0).
2. **Drop the vector database entirely at this scale — the incumbents' own code proves it.** 0.47 ms exact brute force at 10k, 7.4 ms at 100k (§1.1). immich's `targetListCount` sets `lists=1, probes=1` below 128k assets, i.e. it *already* full-scans while paying for Postgres + VectorChord + a container; LibrePhotos uses faiss `IndexFlatIP` (exact); PhotoPrism blocked its entire CLIP feature behind a MariaDB 11.8 vector type for what is a 20 MB matmul. Shipping brute force removes a dependency class, removes recall loss, removes index-rebuild-on-model-change, and *shrinks* the code. Add ANN only above a measured crossover (~500k) — and publish that crossover.
3. **Optimise the text tower, not the index.** Since scan is ~1% of B3, latency is text-encode + cold start. Concrete: a resident daemon with the text encoder warm; an LRU cache of query embeddings; a quantized/distilled text tower; a ≤2 s cold start (B13). No surveyed competitor treats query-side latency as the primary metric: rclip pays a full Python start per invocation, photofield-ai pays an HTTP multipart round-trip, and **immich actively unloads its model after 300 s idle (`model_ttl`)** — so the first search of the day is the slowest one, by design. Keeping the *text* tower resident costs a few hundred MB and buys the whole B3/B4 budget.
4. **Ship the good model *by default*.** Two concrete, measured candidates the field has left on the table: **`ViT-B-32__laion2b-s34b-b79k`** — by immich's own bench it costs **+0.03 ms** over their default and returns **+7.7 recall points** (77.62 vs 69.90), i.e. a *free* upgrade nobody took; and **`ViT-B-16-SigLIP2__webli`** at 5.81 ms / 84.86%. ⚠️ **Do not pick MobileCLIP for CPU** — see §2.14: it is 1.5× *slower* than ViT-B-32-256 on x86 CPU, and its weights are research-only. Ship-safe modern pick: **SigLIP2-B/32-256 or B/16, int8 ONNX** (Apache-2.0, vision **95.9 MB**). Above them sits `ViT-B-16-SigLIP2` at 5.81 ms / 84.86%. Note that `ViT-L-14__openai` (19.91 ms) scores *worse* (72.99) than `ViT-B-32` laion2b at 2.29 ms — **bigger is not better here**, which is exactly the kind of claim the candidate bench must verify rather than inherit. Meanwhile rclip, Lap, photofield-ai and immich all default to **CLIP ViT-B/32 openai (2021)** — 69.9% recall by immich's own measurement — while a SigLIP2-B/16 at **84.86%** costs **+3.55 ms/image** (§2.1.1). Ente (MobileCLIP) and PixFinder (SigLIP2) are the only ones on a modern tier, and neither publishes numbers. Defaulting to the modern tier is a **+15-recall-point, ~35-second-per-10k-images** trade that nobody has taken, and it is exactly what B5's hypernym requirement (`car → vehicle → motorcycle`) needs, since hypernymy lives in the text tower's semantics, not in the search code. (Watch B8: SigLIP2 at f32 is ~3 GB RSS — the quantised/precision path is mandatory, and it is unmeasured territory for everyone.)
5. **Make decode the engine, not the glue — this is where the race is actually won.** immich's own table says inference is **2.26–5.81 ms/img**; our §1.2 measurement says full-res JPEG decode is **163–287 ms/img** single-threaded on worst-case files. Preprocessing outweighs the model by 1–2 orders of magnitude, and *every* project in §2 treats it as plumbing. Concretely: DCT-scaled `draft()` decode (1.75× measured), EXIF-thumbnail-first decoding, libvips/`fast_image_resize` SIMD resamplers, rclip's 16-worker lookahead pipeline, and never decoding a 12 MP frame to produce a 224×224 tensor. A project that indexes at 20 img/s while its model needs 5 ms is spending ~90% of its time not doing ML — that waste is our headroom.
6. **Ship correctness as a product feature — the anti-false-positive layer.** The loudest field complaint is quality ("lots of misses and false positives"), and *no competitor returns a calibrated score or ever says "no match"*: they all return top-K unconditionally. A calibrated absent-category threshold (B7 ≤2%) plus published precision/recall on COCO ground truth (B6) is a claim literally no one else in §2 can make.
7. **Operational transparency as a feature (live progress + search-during-index).** The vision demands img/s, ETA, and searchable-while-indexing (B10/B11). immich gets partial-results-while-indexing for free from its per-asset writes but exposes no rate/ETA worth the name; the desktop apps show a spinner. An event-driven progress stream (≤1 s stale, ETA ±20%, zero spin-polling) is both a UX differentiator and the honest proof that the engine has no leaks (B12).
8. **Own the agent-native niche — it has exactly one, tiny incumbent.** Every project in §2 is a GUI or a human CLI, with one exception: `Eventual-Inc/local-image-search` (9★, macOS-only, MLX/GPU-only, **no licence file**, MCP server). The vision demands "a globally available skill so agents will be able to … tag, get info about datasets, manage them, and run searches." A cross-platform, CPU-only, licensed, dataset-scoped machine API (results carrying dataset id + path/id, exactly as the vision spells out) beats that incumbent on every axis except raw M4-GPU throughput. **This is the single least-contested lane in the entire survey.**

9. **Take the whole-pipeline win the incumbents left on the table.** immich's structural losses are enumerable and each is ours to invert: batch=1 → **batched inference**; `inter_op=1/intra_op=2` → **all-core ORT**; decoding a 1440 px preview to make a 224 px tensor → **decode at target scale**; HTTP-multipart hop between containers → **in-process**; f32-only → **int8/fp16**; 6 GB RAM floor → **B8's 1.5 GB**. Their own bench (2.26 ms/img) versus their own field rate (~1.11 img/s on an N100) says roughly **99.8% of wall-clock is not the model**. That is not a tuning opportunity, it is a different architecture — and it is the single most defensible reason IMGTAG can claim a 15–250× jump without a faster chip.

11. **Batch the encoder — the leader does not.** immich's `MaxBatchSize` covers only faces and OCR; CLIP runs one image per `session.run`. Measured here: int8 batch 1 → batch 8 = 125 → **158 img/s**; fp32 batch 1 → 8 = 8.9 → **57 img/s (6.4×)**. CLIP-Finder2 uses batch 512 on ANE; rclip's CoreML export is fixed batch 8. Batching plus an async decode→encode pipeline is the cheapest large multiplier available, and it is unexploited by the 108k-star incumbent.
12. **Quantize the *embeddings*, not (yet) the model.** Model int8 costs 1.7–4.6 pp of accuracy depending on method, and ORT warns it can be *slower* on old devices — our target. Embedding int8 keeps ~99.3% and binary+rescore ~96%, at 3.7×/24.8× measured speedups, needs no re-embedding if f32 is stored, and is reversible. Store f32 at full dim; derive int8/binary tiers on top. (⚠️ those retention numbers are text-embedding measurements — verify on CLIP image vectors as part of B6/B7.)

**Steal list (attributed, ranked by value-per-effort):**
`tag_embeddings.npy` precomputed zero-shot label cache (LibrePhotos) · two-stage thumbnail
cascade + 3-crop 224 feed + 3×3 colour thumbnail + 0–7 quality heuristic + per-hash index
mutex + worker-count-with-a-reason logging (PhotoPrism) · memory-gated model load with
requeue instead of OOM (Photonix) · degraded-tier fallback on missing AVX and per-image
timeout budgets (Nextcloud Recognize) · 16-worker lookahead decode pipeline, arithmetic
queries, `release_indexing_resources()` (rclip) · single-binary non-invasive distribution
(photofield) · `PRELOAD` + `MODEL_TTL=0` as an explicit hot mode, split textual/visual
models, and the per-language mem/latency/recall model table (immich — their docs page is the
best competitive artifact in the field; we should ship a better one) · batch-512 incremental
add/delete on launch (CLIP-Finder2) · semantic ∪ keyword fusion (LibrePhotos).

10. **Win the operational details the whole field ignores.** Each is small, each is demanded in public, none is claimed: **split the text and image towers** into separate model files so a search loads ~35 MB not ~300 MB (Queryable); **never delete-then-reindex** on model change — write the new embedding column beside the old so search stays live (immich's documented sin, HN 44426745); **run indexing at low OS scheduling priority** so the machine stays usable (asked for on HN, implemented by nobody); **make re-embedding cheap by design** (the "model-upgrade tax", HN 44438033); and **put the benchmark table on the landing page** (HN 44119032 — the exact complaint levelled at a competitor).

**Two bonus moves, cheap and unclaimed:**
- **Single self-contained binary/wheel with the model resolved on first run** (photofield's single-binary distribution + Lap's `download_models.sh` pattern), so "runs on an old computer" is true without Docker or Postgres.
- **Preprocessing parity tests.** clip.cpp openly documents that it substitutes linear for bicubic interpolation and therefore *does not reproduce reference numbers* — a silent-quality-loss trap. A test asserting our preprocessing matches the reference implementation's embeddings within ε converts a hidden failure mode into a CI gate (protects B6/B7).

---

## 5. Actions this lane hands to the project (not opinions — consequences)

1. **Re-derive B1 before locking it.** rclip reports **180 img/s on an M1 Max** (PR #249) and a sibling lane measured **157.9 img/s CPU-only int8 on this very M3 Max** — against B1's ≥30 img/s. Run rclip head-to-head on the chosen bench corpus, record its img/s honestly, and set B1 above it. A budget that ships something several times slower than an existing MIT CLI is not "blazing fast" — it is a regression with better docs.
2. **Raise the edge floor (⌂ ≥5 img/s).** A 2014 6-core i7 already sustains 20 img/s (photofield-ai) and a 2016 Celeron J3455 does 1.9 img/s with fp32 ViT-B/32 (rclip PR #249) — int8 should roughly double that. ⌂ should be ≥10 img/s on 4 modern cores, with the Celeron-class figure reported separately and honestly.
3. **Add a quality budget with a published baseline.** Nothing in BUDGETS.md pins retrieval recall against a *public* benchmark. immich publishes Crossmodal-3600 recall per model (69.9% default → 85.99% best). Add "retrieval recall on a public set ≥ the model card, ≥82% English" alongside B5/B6/B7 so quality claims are auditable, not internal.
4. **Add a preprocessing-parity gate.** clip.cpp's measured ImageNet top-1 of 31.4% vs open_clip's 66.6% for identical weights — caused partly by linear-instead-of-bicubic resize — is the cautionary tale. Assert our embeddings match the reference implementation within ε in CI.
5. **Add two budgets the field has taught us to want:** a *cold-search* budget (immich: 60–70 s after idle) and an *indexing-politeness* budget (low OS priority; HN's "fans go crazy").
6. **Add a licence gate to the candidate bench (§2.11).** MobileCLIP/MobileCLIP2 weights are Apple research-only and jina-clip-v2 is CC-BY-NC — both would look like winners on a pure speed/quality bench and both are unshippable. The candidate table needs a **licence column**, and any non-shippable model must be labelled "reference only".
7. **Beat rclip's 15-second search on day one.** `sorted()` over the whole corpus instead of `np.argpartition`, plus re-materialising every vector from SQLite per invocation (§2.4). Our B3/B4 budgets already imply the fix; the point is that the *demo* — "80,000 images searched in under a millisecond of scan while the incumbent takes 15 seconds" — writes itself.
8. **Run the 8 experiments nobody in this field has run.** Every one is cheap, and each closes a gap where *no measured number exists anywhere* — meaning whoever measures it first owns the claim:
   1. `PIL.Image.draft()` / `tjbench -scale` speedup table (**literally unpublished; §1.2 is the only measurement found**) — ~30 min.
   2. **fp16 CoreML CLIP export on M-series** (rclip ships fp32 → never touches the ANE) — the single largest unclaimed differentiator.
   3. Re-derive immich's exec-time column locally (image tower vs text tower, §2.1.1) before quoting it publicly.
   4. ORT CoreML EP vs CPU EP on the same ONNX — no such CLIP measurement exists.
   5. MLX CLIP img/s on any M-chip — none published, from anyone.
   6. NEON int8-vs-f32 dot product (all published int8 SIMD wins are AVX-512 VNNI; `simsimd` measures it in minutes).
   7. CLIP recall@k vs input resolution (224/160/128/112) — decides whether the EXIF-thumbnail fast path is safe.
   8. Recall@10 of int8/binary/MRL-truncated **CLIP image** vectors vs f32 — every published retention number is text-only.
9. **Also unmeasured and worth owning:** HEIC decode cost on Apple Silicon (ImageIO vs libheif vs vips — Apple libraries are mostly HEIC since iOS 11), decode throughput on M1/M2/M3 (only M4 Max is published), and whether ImageIO reaches the `AppleJPEGDriver` hardware JPEG block at all.
9. **The one thing not to lose:** every headline number in this file is either quoted from a source with its hardware, or measured here and labelled. Keep it that way — the strategy in §4 move 1 (be the project that publishes numbers) only works if our own numbers are beyond reproach.

---

## Appendix A — verified repo table (GitHub REST API, 2026-07-22)

| Repo | Stars | Last push | License | Lang |
|---|---:|---|---|---|
| immich-app/immich | 108,427 | 2026-07-22 | AGPL-3.0 | TypeScript |
| photoprism/photoprism | 39,991 | 2026-07-22 | NOASSERTION | Go |
| ente-io/ente | 27,929 | 2026-07-22 | AGPL-3.0 | Dart |
| screenpipe/screenpipe | 20,379 | 2026-07-22 | NOASSERTION | Rust |
| jina-ai/clip-as-service | 12,831 | **2024-01-23 (dormant)** | NOASSERTION | Python |
| LibrePhotos/librephotos | 8,005 | 2026-07-21 | MIT | Python |
| rom1504/clip-retrieval | 2,787 | 2026-03-28 | MIT | Jupyter |
| photonixapp/photonix | 1,953 | 2026-07-21 | AGPL-3.0 | Python |
| julyx10/lap | 1,354 | 2026-07-21 | GPL-3.0-or-later | Rust |
| ldqk/ImageSearch | 1,317 | 2026-07-07 | — | C# |
| unum-cloud/UForm | 1,243 | 2025-10-30 | Apache-2.0 | Python |
| haltakov/natural-language-image-search | 1,041 | **2022-10-13 (dead)** | MIT | Jupyter |
| yurijmikhalevich/rclip | 979 | 2026-07-21 | MIT | Python |
| nextcloud/recognize | 692 | 2026-07-22 | AGPL-3.0 | PHP |
| SmilyOrg/photofield | 599 | 2026-06-21 | — | Go |
| deepfates/memery | 576 | 2026-07-04 | MIT | Python |
| monatis/clip.cpp | 564 | 2025-06-19 | MIT | C++ |
| harperreed/photo-similarity-search | 502 | 2024-05-17 | — | Python |
| ncoevoet/facet | 169 | 2026-07-22 | MIT | Python |
| fguzman82/CLIP-Finder2 | 95 | 2024-07-25 | — | Swift |
| SmilyOrg/photofield-ai | 30 | 2026-06-07 | — | Python |

Also verified this session: sist2app/sist2 1,276 (2025-07-05, GPL-3.0) · criteo/autofaiss 907 (2025-11-04, Apache-2.0) ·
DIVISIO-AI/stag 150 (2025-08-19, Apache-2.0) · Eventual-Inc/local-image-search 9 (2026-01-29, **no licence**) ·
Pankaj4152/smart-photo-finder 12 (2025-12-12, MIT) · illegal-instruction-co/rememex 66 (2026-02-19, no licence).

Native / edge / adjacent (same query, same day): mazzzystar/Queryable 2,969 (2026-03-29, MIT) ·
pykeio/ort 2,414 (2026-07-16, Apache-2.0) · frost-beta/sisi 580 (**dead 2024-09-16**, MIT) ·
greyovo/PicQuery 505 (2026-04-25, MIT) · 1yefuwang1/vectorlite 360 (2026-07-01, Apache-2.0) ·
osmarks/meme-search-engine 115 (2026-03-09, MIT) · apetersson/immich_ml_balancer 70 (2025-08-19, MIT) ·
immich-app/ml-models 13 (2026-07-22, AGPL-3.0 — where immich's bench table comes from) ·
RuurdBijlsma/open-clip-inference-rs 5 (2026-04-26, MIT) · tracel-ai/burn 15,622 ·
xinyu1205/recognize-anything 3,690 (**stale 2025-02-18**) · AILab-CVC/YOLO-World 6,475 (**stale 2025-02-26, GPL-3.0**).

Infrastructure: faiss 40,565 · qdrant 33,489 · mlx 27,645 · pgvector 22,303 · onnxruntime 21,160 ·
candle 20,702 · transformers.js 16,206 · open_clip 14,012 · libvips 11,508 · lancedb 10,954 ·
openvino 10,561 · sqlite-vec 7,919 · hnswlib 5,288 · marqo 5,018 · usearch 4,228 · tract 3,002 ·
VectorChord 1,747 · apple/ml-mobileclip 1,593 · pgvecto.rs 2,179 (**abandoned 2025-02-26**).

**Link check:** every GitHub, docs and blog URL cited in this file was fetched on
2026-07-22 with `curl -L -o /dev/null -w %{http_code}` and returned **200**, with two
noted exceptions: `photoprism/discussions/4771` returns **404** (the citing lane's number
for it — "0.02 s/image at 1k → ~2 s/image at 1.75M" — is therefore **flagged unverified**
and should not be quoted without re-sourcing), and `news.ycombinator.com` item pages
rate-limited (**429**) on bulk checking while individual fetches succeeded; two HN quotes
were additionally re-verified verbatim through the Algolia item API
(comments 41555976 and 44429236 — both matched exactly).

## Appendix B — reproducing §1

```bash
# B.1 brute-force scan
python3 - <<'PY'
import numpy as np, time
n, dim = 10_000, 512
X = np.random.randn(n, dim).astype(np.float32); q = np.random.randn(dim).astype(np.float32)
for _ in range(3): s = X @ q
t = time.perf_counter()
for _ in range(10):
    s = X @ q; top = np.argpartition(-s, 50)[:50]
print((time.perf_counter()-t)/10*1000, "ms")
PY

# B.2 JPEG decode: full vs DCT-scaled draft()
python3 - <<'PY'
from PIL import Image; import time, glob
files = glob.glob('/path/to/*.jpg')
def run(draft):
    t=time.perf_counter()
    for f in files:
        im = Image.open(f)
        if draft: im.draft('RGB', (224,224))
        im.convert('RGB').resize((224,224), Image.BILINEAR)
    return (time.perf_counter()-t)/len(files)*1000
print('full', run(False), 'ms/img'); print('draft', run(True), 'ms/img')
PY
```

---

> 2026-07-22: created by the PRIOR ART lane. Session-wide WebSearch budget was exhausted
> mid-lane (200/200); GitHub REST API, direct source reads, and WebFetch carried the rest —
> noted for honesty about coverage, not as an excuse.
