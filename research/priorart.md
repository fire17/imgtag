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
| photofield-ai | **~20 req/s on an i7-5820K (6-core, 2014)**, ~200 req/s on GTX 1070 Ti | [README](https://github.com/SmilyOrg/photofield-ai) |
| UForm (unum-cloud) | **325.4 img/s (small, ONNX) / 212.8 img/s (base, ONNX)** on a *160-core dual-socket Intel Emerald Rapids*, batch 128 | [BENCHMARKS.md](https://github.com/unum-cloud/uform/blob/main/BENCHMARKS.md) |
| clip.cpp | **4-bit quantized CLIP = 85.6 MB** on disk (size, not speed) | [README](https://github.com/monatis/clip.cpp) |
| Nextcloud Recognize (user report) | "cranking through **10k–20k photos per day**" (≈0.12–0.23 img/s) | [HN 44426233](https://news.ycombinator.com/item?id=44426233) |
| **immich (docs)** | **per-model CPU execution time + peak RSS + recall**, measured on a 7800X3D bare-metal Linux at f32 — the best dataset in the field (§2.1.1) | [docs/features/searching.md](https://github.com/immich-app/immich/blob/main/docs/docs/features/searching.md) |

Everything else — immich, PhotoPrism, rclip, Lap, Ente, PixFinder, Desktop Docs, the 2026
wave of "local AI photo search" apps — ships **feature lists, not measurements**. Two apps
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
- **Verified stack (`pyproject.toml`, `rclip/model.py`, `rclip/main.py`):** OpenCLIP **ViT-B/32**, `VECTOR_SIZE = 512`; runtime **`onnxruntime` ≥1.24 + `coremltools` on macOS** (they dropped torch from the runtime deps — dev-only); SQLite store; `pi-heif` + `rawpy` for HEIC/RAW; **`MAX_IMAGE_LOADING_WORKERS = 16` and `LOOKAHEAD_BATCHES = 3`** — a genuine decode/inference pipeline with in-flight futures; incremental indexing; arithmetic queries (`"2:golden retriever" + ./pool.jpg - fruit`).
- **⚠️ rclip is the real speed bar, and it is far above our provisional budget.** README, verbatim (verified in-source 2026-07-22): *"it took **15 hours to process 84,725 photos** on a NAS with an **old Intel Celeron J3455**, **7 minutes to index 50,000 images** on a **MacBook with an M1 Max**, and **3 hours to process 1.28 million images** on the same MacBook."*
  - Celeron J3455 (4-core, 2016, ~10 W): **1.57 img/s** — pure CPU.
  - M1 Max: **50,000 / 420 s = 119 img/s**, and the 1.28 M run independently gives **118.5 img/s** — the two numbers agree, which is a good sign they are real.
  - Caveat that matters: on macOS rclip's runtime deps include `coremltools`, so the Apple-silicon figure is very likely **CoreML (ANE/GPU-assisted)**, not strictly CPU-only. The honest CPU-only datapoint from rclip is the Celeron's 1.57 img/s.
  - **Implication for BUDGETS.md B1 (≥30 img/s on M3 Max):** an existing MIT CLI already claims ~119 img/s on an M1 Max. B1 as written would ship something **4× slower than the incumbent** on comparable hardware. B1 should be re-derived from a head-to-head rclip run on our machine before it is locked — see §5.
- **Steal:** the 16-worker lookahead decode pipeline; arithmetic/mixed queries (an actual differentiator on quality-of-life); `release_indexing_resources()` — dropping the vision session after indexing to shrink RSS; terminal image previews; the packaging spread (snap + AppImage + brew + pip).
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
- **Measured indexing (community, all URL-cited):** ~**2 photos/s** on a 2017 laptop ([disc #2183](https://github.com/photoprism/photoprism/discussions/2183)); **4–5 s/image** on an i3-6100 with local SSD, **~6 s/image** on a Celeron J4025, **~18 s/image** over CIFS ([disc #4638](https://github.com/photoprism/photoprism/discussions/4638)); **50,000 photos in ~24 h ≈ 1.7 s/photo** with 4 workers ([mustafa.net, 2026-03-06](https://mustafa.net/2026/03/06/photoprism-tips-tricks-every-self-hoster-should-know/)); degradation from **0.02 s/image at 1k to ~2 s/image at 1.75M**, unchanged with TensorFlow disabled — i.e. DB/IO-bound, not model-bound ([disc #4771](https://github.com/photoprism/photoprism/discussions/4771)). **Read: ~2–6 s/photo typical ⇒ 10k ≈ 6–17 h.** No official throughput number exists.
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

**`open-clip-rs` / `open_clip_inference` (RuurdBijlsma)** — https://github.com/RuurdBijlsma/open-clip-rs · MIT · created 2026-01-27, last push 2026-04-26 · **5★** (tiny, but the single most relevant benchmark table found). Rust + `ort` (ONNX Runtime), CPU by default, auto-downloads pre-converted ONNX by HF id. Published table (⚠️ *"measured on my CPU"* — **CPU model not stated**; vision times include 10–20 ms preprocessing):

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

### 2.9 Merged lane reports

*(immich/PhotoPrism internals · rclip/memery/clip-retrieval/Jina/WD14 · Rust-Go/clip.cpp/HN
· model+runtime CPU benchmarks — merged below as the four research lanes land.)*

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

## 3. To beat the world — the bar, in numbers

Every entry is a real, sourced number an incumbent has to defend, mapped to a BUDGETS.md row.

| # | The bar to exceed | Where it comes from | IMGTAG budget |
|---|---|---|---|
| **0** | **⚠️ 119 img/s on an M1 Max (50,000 images in 7 min), and 1.57 img/s on a 2016 Celeron J3455** — the highest credible throughput any local tool claims | **rclip README**, verified verbatim in-repo (§2.4). Apple-silicon figure likely CoreML-assisted; the Celeron figure is pure CPU | **This is the number to beat, and B1 (≥30 img/s) is below it.** Either B1 rises to ≥150 img/s on the dev target, or IMGTAG must be honest that it trades throughput for quality/latency. Recommend: measure rclip head-to-head on the same corpus before locking B1 (§5) |
| 1 | **20 img/s CPU indexing on a 6-core 2014 i7** | photofield-ai README (measured) | The edge floor ⌂ ≥5 img/s is too soft — a 2014 6-core already does 20 |
| 2 | **~2 img/s per core** (325 img/s ÷ 160 cores, UForm-small ONNX, batch 128) | UForm BENCHMARKS.md | On 16 cores that projects to ~32 img/s for a *small* model — B1's ≥30 img/s is roughly par, so par is not a win. Target **≥60 img/s** with a MobileCLIP/SigLIP2-class model + draft decode |
| 3 | **The self-hosted status quo is ~0.2–2 img/s**: immich **1.11 img/s** (N100, CPU-only), immich **0.4 img/s** (i5-10500), PhotoPrism **0.17–0.6 img/s** (2–6 s/photo), Recognize **0.12–0.23 img/s** | §2.1.2, §2.6 | B1's ≥30 img/s is a **15–250× jump over every self-hosted incumbent**. This is the headline claim, and it is defensible because their bottlenecks are structural (batch=1, 2 threads, 1440 px decode, HTTP hop), not physical |
| 4 | **10k images fully indexed** | vision: "time to process 10,000 images on cpu" | B2 ≤6 min ⇒ ≥28 img/s sustained. At 20 img/s (photofield-ai) the same job takes 8m20s. **Sub-3-minute 10k is a world-first claim if measured and published** |
| 5 | **Search latency: nobody publishes one — and the leader's cold search is 60–70 s** | [immich disc #14547](https://github.com/immich-app/immich/discussions/14547); LibrePhotos "first search may take up to a minute" | B3 p50 ≤50 ms / B4 p95 ≤150 ms keystroke→painted. §1.1 proves the scan is 0.47 ms; the *only* ways to lose are text-encode and cold start. **A warm resident text tower turns the category's worst wound into our headline demo** |
| 6 | **Model size 85.6 MB (4-bit CLIP, clip.cpp)** | clip.cpp README | B9 ≤150 MB is comfortable; aim to *beat 85.6 MB while beating ViT-B/32 quality* |
| 7 | **Quality: "poor embeddings, misses and false positives" (immich, user)** | HN 44426233 | B5/B6/B7 are the differentiator — the incumbents ship no quality metrics at all |
| 8 | **Retrieval recall 69.9%** (immich default `ViT-B-32__openai`, Crossmodal-3600 avg of R@1/5/10) — and **85.99%** for the best model anyone in this field ships | immich docs table (§2.1.1) | Shipping a **≥82–85%-recall model *by default*** puts IMGTAG 12–16 recall points above the category leader's out-of-box behaviour on day one |
| 9 | **Model inference 2.26 ms (B-32) / 5.81 ms (B-16-SigLIP2) per image on a 7800X3D** | immich docs table | ⇒ inference alone allows >170 img/s single-stream. Anyone indexing at 20 img/s is losing ~90% of the time somewhere else (decode, I/O, HTTP, Python). **That gap is the prize.** |
| 10 | **~3 GB peak RSS for SigLIP2 at f32** | immich docs table | B8 caps peak at 1.5 GB ⇒ we must land modern-tier quality *inside half the memory immich needs*. No competitor has published a quantised measurement — uncontested ground |

## 4. The eight moves that make IMGTAG objectively better

Ranked by (leverage × how few competitors do it). Each is grounded in a specific finding above.

1. **Publish the benchmark nobody else has.** `imgtag bench {index,search,quality,resources,soak}` producing a hardware-labelled table, plus a public results file. Cost: near zero. Effect: instantly the most credible project in §2. Nobody — not immich, not Lap, not rclip — can currently answer "how many images per second?" (§0).
2. **Drop the vector database entirely at this scale — the incumbents' own code proves it.** 0.47 ms exact brute force at 10k, 7.4 ms at 100k (§1.1). immich's `targetListCount` sets `lists=1, probes=1` below 128k assets, i.e. it *already* full-scans while paying for Postgres + VectorChord + a container; LibrePhotos uses faiss `IndexFlatIP` (exact); PhotoPrism blocked its entire CLIP feature behind a MariaDB 11.8 vector type for what is a 20 MB matmul. Shipping brute force removes a dependency class, removes recall loss, removes index-rebuild-on-model-change, and *shrinks* the code. Add ANN only above a measured crossover (~500k) — and publish that crossover.
3. **Optimise the text tower, not the index.** Since scan is ~1% of B3, latency is text-encode + cold start. Concrete: a resident daemon with the text encoder warm; an LRU cache of query embeddings; a quantized/distilled text tower; a ≤2 s cold start (B13). No surveyed competitor treats query-side latency as the primary metric: rclip pays a full Python start per invocation, photofield-ai pays an HTTP multipart round-trip, and **immich actively unloads its model after 300 s idle (`model_ttl`)** — so the first search of the day is the slowest one, by design. Keeping the *text* tower resident costs a few hundred MB and buys the whole B3/B4 budget.
4. **Ship the good model *by default*.** rclip, Lap, photofield-ai, immich (!) and most of the indie wave all default to **CLIP ViT-B/32 (2021)** — 69.9% recall by immich's own measurement — while a SigLIP2-B/16 at **84.86%** costs **+3.55 ms/image** (§2.1.1). Ente (MobileCLIP) and PixFinder (SigLIP2) are the only ones on a modern tier, and neither publishes numbers. Defaulting to the modern tier is a **+15-recall-point, ~35-second-per-10k-images** trade that nobody has taken, and it is exactly what B5's hypernym requirement (`car → vehicle → motorcycle`) needs, since hypernymy lives in the text tower's semantics, not in the search code. (Watch B8: SigLIP2 at f32 is ~3 GB RSS — the quantised/precision path is mandatory, and it is unmeasured territory for everyone.)
5. **Make decode the engine, not the glue — this is where the race is actually won.** immich's own table says inference is **2.26–5.81 ms/img**; our §1.2 measurement says full-res JPEG decode is **163–287 ms/img** single-threaded on worst-case files. Preprocessing outweighs the model by 1–2 orders of magnitude, and *every* project in §2 treats it as plumbing. Concretely: DCT-scaled `draft()` decode (1.75× measured), EXIF-thumbnail-first decoding, libvips/`fast_image_resize` SIMD resamplers, rclip's 16-worker lookahead pipeline, and never decoding a 12 MP frame to produce a 224×224 tensor. A project that indexes at 20 img/s while its model needs 5 ms is spending ~90% of its time not doing ML — that waste is our headroom.
6. **Ship correctness as a product feature — the anti-false-positive layer.** The loudest field complaint is quality ("lots of misses and false positives"), and *no competitor returns a calibrated score or ever says "no match"*: they all return top-K unconditionally. A calibrated absent-category threshold (B7 ≤2%) plus published precision/recall on COCO ground truth (B6) is a claim literally no one else in §2 can make.
7. **Operational transparency as a feature (live progress + search-during-index).** The vision demands img/s, ETA, and searchable-while-indexing (B10/B11). immich gets partial-results-while-indexing for free from its per-asset writes but exposes no rate/ETA worth the name; the desktop apps show a spinner. An event-driven progress stream (≤1 s stale, ETA ±20%, zero spin-polling) is both a UX differentiator and the honest proof that the engine has no leaks (B12).
8. **Be the first agent-native image search.** Every project in §2 is a GUI or a human CLI. The vision demands "a globally available skill so agents will be able to … tag, get info about datasets, manage them, and run searches." A stable, machine-readable API (dataset-scoped results with dataset id + path/id, as the vision spells out) makes IMGTAG the *only* engine an agent can drive — a category with, as far as this survey found, **zero incumbents**.

9. **Take the whole-pipeline win the incumbents left on the table.** immich's structural losses are enumerable and each is ours to invert: batch=1 → **batched inference**; `inter_op=1/intra_op=2` → **all-core ORT**; decoding a 1440 px preview to make a 224 px tensor → **decode at target scale**; HTTP-multipart hop between containers → **in-process**; f32-only → **int8/fp16**; 6 GB RAM floor → **B8's 1.5 GB**. Their own bench (2.26 ms/img) versus their own field rate (~1.11 img/s on an N100) says roughly **99.8% of wall-clock is not the model**. That is not a tuning opportunity, it is a different architecture — and it is the single most defensible reason IMGTAG can claim a 15–250× jump without a faster chip.

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

**Two bonus moves, cheap and unclaimed:**
- **Single self-contained binary/wheel with the model resolved on first run** (photofield's single-binary distribution + Lap's `download_models.sh` pattern), so "runs on an old computer" is true without Docker or Postgres.
- **Preprocessing parity tests.** clip.cpp openly documents that it substitutes linear for bicubic interpolation and therefore *does not reproduce reference numbers* — a silent-quality-loss trap. A test asserting our preprocessing matches the reference implementation's embeddings within ε converts a hidden failure mode into a CI gate (protects B6/B7).

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

Infrastructure: faiss 40,565 · qdrant 33,489 · mlx 27,645 · pgvector 22,303 · onnxruntime 21,160 ·
candle 20,702 · transformers.js 16,206 · open_clip 14,012 · libvips 11,508 · lancedb 10,954 ·
openvino 10,561 · sqlite-vec 7,919 · hnswlib 5,288 · marqo 5,018 · usearch 4,228 · tract 3,002 ·
VectorChord 1,747 · apple/ml-mobileclip 1,593 · pgvecto.rs 2,179 (**abandoned 2025-02-26**).

**Link check:** all URLs cited above returned HTTP 200 on 2026-07-22 (`curl -L -o /dev/null -w %{http_code}`).

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

> 2026-07-22: created by the PRIOR ART lane. Session-wide WebSearch budget was exhausted
> mid-lane (200/200); GitHub REST API, direct source reads, and WebFetch carried the rest —
> noted for honesty about coverage, not as an excuse.
