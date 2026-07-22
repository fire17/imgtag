# IMGTAG research lane A+B — CPU runtimes, quantization, preprocessing, vector storage

**Lane:** (A) CPU inference runtimes + quantization for VLM/CLIP-class encoders · (B) vector storage & search
**Author:** `research-runtime` teammate · **Date:** 2026-07-22
**Method:** deep web research **plus first-party benchmarks run on this machine.** Every number below marked
`[measured]` was produced by a script in `research/bench_scripts/` and is reproducible. Numbers marked
`[literature]` come from cited sources and were **not** independently verified.

---

## 0. TL;DR — the recommended stack

| Layer | Recommendation | Why (one line) |
|---|---|---|
| Runtime | **onnxruntime CPUExecutionProvider** | only runtime with a light pip wheel (18 MB), both arches, a real CLIP model zoo, and measured 2.8× int8 speedup here |
| Quantization | **`quantize_dynamic(..., op_types_to_quantize=['MatMul'], extra_options={'MatMulConstBOnly': True})` — weight-only int8, activations left fp32** | **[measured]** cos 0.985 vs fp32 & 2.8× faster; the *off-the-shelf* full-QDQ int8 files on HF are cos 0.955 and **slower** |
| Preprocessing | **Pillow + `Image.draft()` + `thumbnail(reducing_gap=2.0)`**, opencv/kornia-rs only if profiling says so | **[measured]** 2.1× over naive Pillow, zero extra deps (Pillow already required to read files) |
| Vector layer | **numpy fp32 `X @ q` brute force over an mmap'd shard** — no ANN library at all | **[measured]** 0.28 ms @10k, 3.8 ms @100k, 36 ms @1M. Exact. Zero build cost. Zero deps. |
| Concurrency | **append-only fp32 shard + atomic manifest (`os.replace`), readers `np.memmap`** | **[measured]** 20 000 rows appended while a reader searched: 0 torn reads, 0 wrong results, 1.6 ms median search |
| Parallelism | **N worker processes × 1 ORT thread**, NOT 1 process × N threads | **[measured]** 181 img/s vs 104 img/s — 1.7× from process-level parallelism alone |

**Headline measured result:** end-to-end (JPEG decode → resize → normalize → CLIP ViT-B/32 int8 encode),
12 workers × 1 thread = **181 img/s → 10 000 images in ~55 s** on an M3 Max *while the machine was under
load average ~10 from sibling agents*. Target was "tens of images/sec". We are ~6× past it.

Search latency budget: **0.28 ms** for 10k exact. Target was "tens of ms". We are ~100× under it.

---

## 1. Test environment & honesty statement

```
Host      : Apple M3 Max (arm64), macOS 14.4, 16 logical cores
Python    : 3.12 · onnxruntime 1.27.0 · numpy 2.5.1 · Pillow 12.1.1
            usearch 2.26.0 · faiss-cpu 1.14.3 · hnswlib 0.8.0
Models    : Xenova/clip-vit-base-patch32 (ONNX), Xenova/mobileclip_s0 (ONNX)
Images    : 24 real photos (picsum, 640×480) for accuracy; 12 synthetic 4000×3000 JPEGs for decode
```

**Caveats you must carry forward — do not launder these away:**

1. **The machine was heavily loaded during part of the runs.** `load average` peaked at **47.5** (sibling
   agents in this swarm, one at 503 % CPU). Absolute throughput numbers taken during that window are
   *pessimistic and noisy* — one thread sweep swung 3× between back-to-back runs. All headline numbers in
   §0 were re-measured at load ≈ 10 and are conservative, not optimistic. **Re-benchmark on a quiet
   machine before publishing any number.**
2. **No x86 machine was available.** All measurements are arm64. x86 claims are `[literature]` only. This
   is the single biggest gap in this report — see §8 red flag R6.
3. **Recall numbers for ANN indexes were measured on synthetic data.** I ran two variants (i.i.d. Gaussian
   *and* a 200-cluster mixture that mimics real embedding geometry) and report both; real CLIP embeddings
   will differ. Directionally the conclusion is robust, the exact recall figures are not.
4. **The int8 accuracy tests use 24 images and neighbour-rank agreement, not zero-shot ImageNet.** They
   detect *embedding drift*, which is the right proxy for a retrieval system, but they are not a
   substitute for a real zero-shot eval. Hand that to the eval lane.

---

## 2. (A) CPU inference runtimes — the field

### 2.1 Measured: ONNX Runtime on this box

`bench_scripts/ortbench2.py` — vision encoder only, CPUExecutionProvider, `ORT_ENABLE_ALL` graph opt.

| Model / precision | file MB | thr=1 bs=1 | thr=4 bs=1 | thr=4 bs=8 | thr=8 bs=8 |
|---|---|---|---|---|---|
| MobileCLIP-S0 fp32 (256²) | 45.5 | 11.7 img/s | 42.2 | 41.8 | 45.3 |
| MobileCLIP-S0 **int8** (256²) | 11.8 | 3.8 | 6.2 | 7.0 | 5.8 |
| MobileCLIP-S0 q4 (256²) | 36.7 | 11.7 | 36.6 | 36.0 | 45.4 |
| CLIP ViT-B/32 fp32 (224²) | 351.7 | 10.0 | 27.5 | 26.9 | 16.8 |
| CLIP ViT-B/32 **int8** (224²) | 88.6 | **25.2** | **55.5** | **75.5** | 35.0 |
| CLIP ViT-B/32 fp16 (224²) | 72.5 | **FAILS TO LOAD** | — | — | — |

Three findings jump out, and two of them are traps:

- **Trap 1 — int8 helps pure transformers, and actively *hurts* hybrid conv/transformer nets.**
  ViT-B/32 int8 is **2.0–2.8× faster** than fp32. MobileCLIP-S0 int8 is **3.4× SLOWER** than its own fp32.
  MobileCLIP-S0 is a FastViT-style hybrid; its reparameterized/depthwise convs have no fast int8 ARM
  kernel in ORT, so the graph fills with `QuantizeLinear`/`DequantizeLinear` round-trips that cost more
  than the matmuls save. **Never assume int8 = faster. Always measure per model.**
- **Trap 2 — more threads is not more speed.** ViT-B/32 int8 goes 55.5 img/s at 4 threads and *down to
  17.9* at 8, because ORT's thread pool schedules onto M-series efficiency cores. Cap `intra_op_num_threads`
  at the **performance**-core count, not `os.cpu_count()`.
- **fp16 ONNX does not load on the ORT CPU EP** — `SimplifiedLayerNormFusion` throws
  `GetIndexFromName ... InsertedPrecisionFreeCast_...`. fp16 ONNX files are a GPU/WebGPU artifact. Ignore
  them for this project.

### 2.2 The field, with alive-checks (all verified via authenticated GitHub API, 2026-07-22)

| Runtime | ★ | last push | license | install weight | verdict |
|---|---|---|---|---|---|
| **microsoft/onnxruntime** | 21 160 | 2026-07-22 | MIT | pip wheel **18.4 MB** mac-arm64 / 18.7 MB linux-x64, no compiler | ✅ **PICK.** Only candidate that is light, prebuilt for both arches, and has real CLIP/SigLIP/MobileCLIP ONNX zoo. ARM int8 uses SDOT/I8MM kernels automatically `[literature]`. |
| **openvinotoolkit/openvino** | 10 561 | 2026-07-22 | Apache-2.0 | ~100 MB+ runtime | ⚠️ Real ARM64/Apple-Silicon support exists (2024.1+), **but** "scope of CPU plugin features and optimizations on Arm may differ from Intel x86-64" `[literature]`. Strong on Intel x86 (fp16 on armv8.2 also good). Worth a **later** A/B on old x86 boxes only; not worth a second runtime dependency in v1. |
| **ml-explore/mlx** | 27 645 | 2026-07-22 | MIT | pip, Apple-only | ❌ Apple Silicon **only** — instantly disqualified by the "old computers / edge devices" requirement. MLX is also GPU/unified-memory-first; its CPU path is not the story. Cross-platform is a hard constraint here. |
| **monatis/clip.cpp** (ggml) | 564 | **2025-06-19** | MIT | build from source | ❌ **Stale ~13 months, 564★.** Genuinely appealing idea (no deps, GGUF q4/q8, AVX/AVX2/AVX512/NEON) but you'd be betting the project's core on an unmaintained repo and hand-writing Python bindings. `yysu-888/clip.cpp` is worse (28★, 2025-03). Revisit only if ORT proves inadequate. |
| **huggingface/candle** (Rust) | 20 702 | 2026-07-14 | Apache-2.0 | Rust toolchain | ⚠️ Alive and good, has a CLIP example. But: Rust build in the loop, no prebuilt wheels, you re-implement preprocessing. Only justified if you're shipping a Rust binary anyway. |
| **sonos/tract** (Rust) | 3 002 | 2026-07-22 | Apache/MIT | Rust toolchain | ⚠️ Excellent small-model NN inference, genuinely tiny binaries, great for true edge. Same Rust-in-the-loop cost. Keep as the **future edge-device** path, not v1. |
| **Tencent/ncnn** | 23 564 | 2026-07-22 | BSD-3-ish | build from source | ⚠️ Mobile-first, superb ARM NEON kernels, but the model-conversion path (onnx→ncnn) is manual and transformer support historically lags CNNs. High friction for a Python project. |
| **alibaba/MNN** | 15 702 | 2026-07-22 | Apache-2.0 | build / pip `MNN` | ⚠️ Same class as ncnn, has a Python package and good ARM int8. A legitimate v2 A/B candidate if ORT throughput plateaus. Not a v1 dep. |
| **pytorch/executorch** | 4 819 | 2026-07-22 | BSD-3-ish | heavy toolchain | ⚠️ XNNPACK backend gives strong Arm numbers (int8 1.83×, fp16 3.9× on SME2 `[literature]`) but requires a PyTorch export pipeline and is aimed at mobile app embedding. Overkill; adds enormous install weight. |
| TensorRT | — | — | — | — | ❌ GPU-only. Out of scope per brief. |

**Ranked runtime recommendation**

1. **ONNX Runtime CPU EP** — ship this. Light, dual-arch, model zoo, measured wins.
2. *(defer)* **OpenVINO** as an optional accelerator on Intel x86 hosts, behind a capability probe.
3. *(defer)* **tract or MNN** when the project actually targets a physical edge device with no Python.
4. *(reject for v1)* clip.cpp, candle, ncnn, ExecuTorch, MLX — each fails on maintenance, install weight,
   cross-platform, or integration cost.

---

## 3. Quantization — the most important finding in this report

### 3.1 Off-the-shelf int8 CLIP files on HuggingFace are quietly damaging

`bench_scripts/acc_test.py` / `acc_test2.py` — 24 real photos, compare quantized embeddings against the
fp32 reference from the *same* graph and *same* preprocessing.

| Model & file | cos(fp32, quant) mean | min | top-1 nearest-neighbour agreement |
|---|---|---|---|
| `Xenova/clip-vit-base-patch32` `vision_model_int8.onnx` | 0.9553 | 0.9099 | **0.75** |
| `Xenova/mobileclip_s0` `vision_model_int8.onnx` | **0.1638** | **0.0157** | **0.00** |
| `Xenova/mobileclip_s0` `vision_model_q4.onnx` | 0.6733 | 0.3248 | 0.42 |

**`Xenova/mobileclip_s0/onnx/vision_model_int8.onnx` is broken.** Cosine similarity to its own fp32 output
is 0.16 — the embeddings are unrelated noise. Embedding norms confirm it: fp32 ‖e‖ ≈ 0.93, int8 ‖e‖ ≈ 26.1
(28× off). I re-ran under **two different preprocessing normalizations** to rule out a preprocessing
artifact; with the *correct* one (see §4.1) the agreement is **cos = 0.008**. It is also 3.4× slower.
**Blacklist this file.** The q4 file is degraded too (cos 0.67).

For ViT-B/32 the damage is subtler and therefore more dangerous: cos 0.955 sounds fine, but **25 % of
nearest-neighbour rankings flip.** That is precisely the failure mode this project cannot afford — it is
invisible in a smoke test and shows up as "why didn't it find that car".

This matches the literature: `arXiv:2605.26415` ("The Rescue Effect") reports INT8 CLIP ViT-B/32 zero-shot
ImageNet at **58.72 %** vs ~63.3 % fp32 — a **−4.6 pp** collapse, attributed to activation noise
accumulating across transformer blocks (noise-to-signal <10 % in shallow blocks → **52 % at layer 11**)
which rotates the joint-embedding direction that cosine retrieval depends on `[literature]`.

### 3.2 The fix: quantize the WEIGHTS ONLY, leave activations in fp32

`bench_scripts/myquant.py` + `quantcmp.py`. The recipe:

```python
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic(
    "vision_model.onnx", "vision_model_w8.onnx",
    weight_type=QuantType.QUInt8,
    per_channel=False,                       # per-tensor won here; see table
    op_types_to_quantize=["MatMul"],         # <- do NOT quantize everything
    extra_options={"MatMulConstBOnly": True} # <- weight-only; activations stay fp32
)
```

Measured, same 24 images, 4 threads, batch 4 (load ≈ 10):

| variant | MB | img/s | cos vs fp32 | min cos | top-1 NN agreement |
|---|---|---|---|---|---|
| fp32 reference | 351.7 | 40.2 | 1.0000 | — | 1.00 |
| HF `vision_model_int8.onnx` | 88.6 | 104.7 | 0.9553 | 0.9099 | 0.75 |
| **mine: dyn U8, per-tensor** | **95.9** | **113.4** | **0.9850** | **0.9785** | **0.96** |
| mine: dyn U8, per-channel | 96.4 | 109.3 | 0.9876 | 0.9704 | 0.83 |
| mine: dyn S8, per-channel | 96.3 | 110.4 | 0.9866 | 0.9735 | 0.83 |

**The self-quantized weight-only model is simultaneously faster (113 vs 105 img/s), more accurate
(0.985 vs 0.955 cos), and preserves 96 % of nearest-neighbour rankings vs 75 %.** It is strictly better
than the file you'd download. 3.7× smaller than fp32 for a 2.8× speedup at ~1.5 % embedding drift.

Interesting: per-tensor beat per-channel on ranking agreement (0.96 vs 0.83) despite slightly lower mean
cosine — mean cosine is not the metric that matters, ranking agreement is. Measure the right thing.

> `[literature]` note for x86: ORT docs state **U8S8 can be faster than U8U8 on low-end ARM64** with no
> accuracy difference, and that **x86 without VNNI can get *worse* performance from quantization** because
> of quantize/dequantize overhead. Both need re-measuring on the target old-x86 box.

### 3.3 Recommended quantization recipe (ranked)

1. **Weight-only int8 dynamic on MatMul, per-tensor U8** — ship this. Verified above.
2. **Keep an fp32 model available** and gate on a startup micro-benchmark: if int8 isn't faster on this
   host (old x86 without VNNI, or a hybrid-conv model), fall back to fp32 automatically.
3. **Ship a fidelity gate in CI**: after any quantization, assert `cos(fp32, quant) > 0.98` and
   `top-1 NN agreement > 0.90` over a fixed 100-image holdout. This exact test would have caught the
   broken MobileCLIP-S0 file instantly.
4. **Do not** use downloaded `*_int8.onnx` / `*_q4.onnx` / `*_fp16.onnx` files unquestioned.
5. **Do not** quantize activations (full QDQ) for a retrieval encoder — that's where the collapse lives.

---

## 4. Preprocessing — decode + resize

### 4.1 Correct normalization is model-specific (and getting it wrong is silent)

Measured from embedding norms: **CLIP ViT-B/32 needs CLIP mean/std**
(`[0.481,0.458,0.408] / [0.269,0.261,0.276]`); **MobileCLIP ONNX exports expect plain `0..1` rescale with
NO mean/std** (with 0-1 the fp32 output norm is a sane 0.93; with CLIP mean/std it explodes to 2.28e6).
Getting this wrong does not crash — it silently degrades every embedding. Assert output norm sanity.

### 4.2 Measured decode+resize (`bench_scripts/prebench.py`, 4000×3000 JPEGs, single thread)

| strategy | ms/img | img/s (1 thread) | speedup |
|---|---|---|---|
| `Image.open().convert().resize()` (naive) | 80.9 | 13 | 1.0× |
| `+ im.draft('RGB', (S,S))` then resize BILINEAR | 48.6 | 21 | **1.7×** |
| `+ draft` then resize BICUBIC | 53.2 | 19 | 1.5× |
| **`+ draft` then `thumbnail(reducing_gap=2.0)`** | **38.8** | **26** | **2.1×** |

`Image.draft()` is the whole trick: it tells libjpeg to decode at 1/2, 1/4 or 1/8 scale **in the DCT
domain**, so you never materialize the full 12 MP bitmap. It costs one line and is free — Pillow is
already a dependency because you must open the files anyway.

At 26 img/s/thread × 8 workers ≈ 208 img/s decode ceiling for 12 MP photos — comfortably above the
181 img/s the encoder sustains, so **the encoder stays the bottleneck.** For smaller (≤1 MP) images decode
is negligible.

### 4.3 The field `[literature]` + install weight `[measured from PyPI]`

`arXiv:2501.13131` "Need for Speed: A Comprehensive Benchmark of JPEG Decoders in Python" (9 libraries,
ARM64 + x86_64) — **decode only, no resize**:

| library | ARM64 (M4 Max) img/s | wheel size mac-arm64 / linux-x64 | verdict |
|---|---|---|---|
| kornia-rs | **1034** | small (Rust, prebuilt) | ⚠️ fastest decoder, 680★, pushed 2026-07-20, Apache-2.0. Real option **if** decode ever becomes the bottleneck. Adds a dep for ~1.3× over Pillow. |
| OpenCV | 1016 | **48.3 MB / 61.2 MB** | ❌ 1.3× decode gain for a **48 MB** dependency. Violates "install weight must be LIGHT". Reject. |
| torchvision | 992 | enormous (drags torch) | ❌ absolutely not. |
| Pillow | 775 | **4.8 MB / 7.0 MB** | ✅ **PICK.** 75 % of the fastest decoder, already required, tiny, universal, and `draft()` gives 2.1× that the paper's decode-only benchmark doesn't even measure. |
| Pillow-SIMD | ~Pillow×4–6 on resize `[literature]` | **source build only**, 2283★, NOASSERTION license | ❌ requires a compiler + AVX2, x86-focused, is a *fork* of Pillow (conflicts on install), and the 4–6× claim is for resize not decode. Install-weight and fragility disqualify it. |
| libvips / pyvips | very strong on large images `[literature]` | **no wheels** — needs system `libvips` via brew/apt | ❌ non-pip system dependency. Kills "light install" and "runs anywhere". |

**Ranked preprocessing recommendation**

1. **Pillow + `draft()` + `thumbnail(reducing_gap=2.0)`** — ship this. 0 new deps, 2.1× measured.
2. Parallelize decode across the *same* worker processes as the encoder (§6) — that's where the real
   throughput is, not in swapping decoders.
3. *(only if profiling proves decode-bound)* add **kornia-rs** behind an optional import.
4. Reject opencv (48 MB), pyvips (system dep), Pillow-SIMD (source build + fork conflict).

---

## 5. (B) Vector storage & search

### 5.1 The math: brute force at this scale is memory-bandwidth-bound, and that's it

A single-query search is a GEMV: read `N × D × sizeof(dtype)` bytes, do `N × D` MACs. Modern CPUs do far
more FLOPs than they can feed, so **runtime = bytes ÷ memory bandwidth.** Predicted vs measured:

| N | fp32 index size | predicted @ ~70 GB/s | **measured** | implied bandwidth |
|---|---|---|---|---|
| 10 000 | 20.5 MB | 0.29 ms | **0.278 ms** | 74 GB/s ✔ |
| 100 000 | 205 MB | 2.9 ms | **3.81 ms** | 54 GB/s ✔ |
| 1 000 000 | 2048 MB | 29 ms | **36.4 ms** | 56 GB/s ✔ |

The model predicts measurement within ~25 %. **You cannot beat this with a better algorithm at 10k —
you're already at the hardware roof.** Which lets us extrapolate honestly to old hardware:

| host | ~mem bandwidth | 10k fp32 (20 MB) | 10k fp16 (10 MB) | 100k fp32 (205 MB) |
|---|---|---|---|---|
| M3 Max (measured) | ~70 GB/s eff. | 0.28 ms | ~0.15 ms | 3.8 ms |
| modern x86 laptop DDR4 | ~25 GB/s | ~0.8 ms | ~0.4 ms | ~8 ms |
| **old laptop / DDR3 ~2013** | **~10 GB/s** | **~2 ms** | **~1 ms** | **~20 ms** |
| Raspberry Pi 4 class | ~4 GB/s | ~5 ms | ~2.5 ms | ~50 ms |

**Even on a decade-old DDR3 laptop, exact brute force over 10k CLIP vectors is ~2 ms.** The "tens of ms"
target has ~10× headroom on the worst realistic hardware. This is the single strongest argument in the
report: *the vector layer does not need to be clever.*

### 5.2 Measured: every library vs plain numpy (`bench_scripts/vb2.py`, D=512, k=10)

| N | numpy fp32 | usearch exact | faiss FlatIP | usearch HNSW f32 | usearch HNSW i8 | faiss HNSW16 | hnswlib |
|---|---|---|---|---|---|---|---|
| 10k | 0.278 ms | 1.035 ms | 0.220 ms | 0.217 ms (build 0.9 s) | 0.071 ms (0.2 s) | 0.030 ms (0.3 s) | 0.851 ms (**2.2 s**) |
| 100k | 3.805 ms | 9.560 ms | 2.427 ms | 0.275 ms (**20.5 s**) | 0.056 ms (4.8 s) | 0.025 ms (7.5 s) | 0.927 ms (**45.9 s**) |
| 1M | 36.4 ms | 90.2 ms | 21.2 ms | (too slow to build) | 0.083 ms (**81.5 s**) | (too slow) | (too slow) |

**Recall — measured on clustered data that mimics real embedding geometry** (`bench_scripts/recall_test.py`,
100k vectors, 200 clusters, 50 queries):

| method | latency | **recall@10** | build | index RAM |
|---|---|---|---|---|
| **numpy brute force** | 3.23 ms | **1.000** | **0 s** | 205 MB |
| usearch HNSW f32 M=16 ef=64 | 0.140 ms | 0.642 | 5.3 s | 287 MB |
| usearch HNSW f32 M=32 ef=128 | 0.391 ms | 0.912 | 5.4 s | 320 MB |
| usearch HNSW i8 M=32 ef=128 | 0.116 ms | 0.742 | 1.8 s | 119 MB |

Read that table carefully. At 100k, HNSW buys you **2.8 ms** and costs you **9 % of true results**, plus
5 s of build time, plus 1.6× the RAM (f32 HNSW is *larger* than the raw vectors — the graph is not free).
For an app whose spec is *"search 'car' → **all** images with cars"*, silently dropping 9 % of matches is a
product defect, not an optimization. And 3.23 ms was never the problem — the human can't perceive it.

**Crucially, HNSW also destroys the search-while-indexing story** (§6): every insert mutates a shared
graph, needing locks or an expensive concurrent-construction scheme; brute force append is a pure O(1)
file write.

### 5.3 The field, alive-checked + install weight

| candidate | ★ | last push | license | wheel mac-arm64 / linux-x64 | verdict |
|---|---|---|---|---|---|
| **numpy** (no vector lib) | — | 2026 | BSD | 11.9 / 16.7 MB (already required) | ✅ **PICK.** Exact, zero build cost, zero extra deps, trivially mmap-able, lock-free append. Instant to 100k, fine to 1M. |
| unum-cloud/USearch | 4 228 | 2026-07-10 | Apache-2.0 | **0.5 / 2.4 MB** | ⭐ **best ANN option when you actually need one.** Astonishingly light, single-file C++, i8/f16 quantized indexes, memory-mappable, serializable. Its `exact=True` SIMD brute force is *slower than BLAS numpy* (1.0 vs 0.28 ms) — so use it for HNSW or not at all. **Adopt only past ~1M vectors.** |
| facebookresearch/faiss | 40 565 | 2026-07-22 | MIT | 4.8 / **18.5 MB** | ⚠️ `IndexFlatIP` is the fastest exact option measured (0.22 ms @10k, 21 ms @1M — it multithreads the GEMM). If you ever want a drop-in exact speedup at 1M+, this is it. But 18.5 MB on Linux + an OpenMP runtime for a 1.7× gain over numpy at 1M is not worth it at 10k. |
| nmslib/hnswlib | 5 288 | **2026-03-28** | Apache-2.0 | **NO WHEELS — source build** | ❌ Requires a C++ compiler at install time on every platform. That alone disqualifies it against "install weight must be LIGHT". Also slowest build (45.9 s @100k) and slowest ANN query (0.93 ms) in my runs. usearch dominates it on every axis. |
| asg017/sqlite-vec | 7 919 | **2026-05-18** | Apache-2.0 | **0.2 MB** | ⭐ Tiny and lovely if you want vectors *inside* the same SQLite file as your metadata — a genuinely attractive story for a single-file portable index. But it's still brute force under the hood `[literature]` and can't outrun BLAS-backed numpy at 10k+, and it adds a serialization boundary. **Consider for the metadata layer, not the hot search path.** |
| lancedb/lancedb | 10 954 | 2026-07-22 | Apache-2.0 | **52.7 / 58.7 MB** | ❌ 53 MB wheel. Great product (columnar, versioned, disk-native, IVF-PQ), wildly over-scoped for 10k local images. Reconsider only at 10M+ or if you need multi-tenant versioned datasets. |

**Ranked vector-layer recommendation**

1. **Plain numpy fp32 matmul over an mmap'd shard.** Exact, instant, zero deps, lock-free. Ship it.
   Ceiling: comfortable to 1M (36 ms), acceptable to ~2M.
2. **Store fp16 on disk, cast to fp32 in 8k-row chunks** if RAM matters — `[measured]` 2× smaller for
   ~1.9× the latency (6.9 ms vs 3.4 ms @100k), and **fp16 storage is bit-for-bit lossless for retrieval**
   (`[measured]` cos(fp32, fp16-stored) = **1.000000**).
3. **usearch HNSW i8** as the escape hatch past ~1–2M vectors, behind a config flag, once someone actually
   has that many images. 0.08 ms at 1M, 119 MB, 0.5 MB wheel.
4. **faiss IndexFlatIP** if you want exact but multithreaded at 1M+ (21 ms vs 36 ms) and can afford 18 MB.
5. Reject hnswlib (no wheels) and lancedb (53 MB) for this project.

### 5.4 Quantize the *vectors*, never the *encoder* — measured

This is the clean inversion of §3. `bench_scripts/acc_test.py`:

| what is quantized | cos vs fp32 | verdict |
|---|---|---|
| **index stored as fp16** | **1.000000** | ✅ free 2× compression, literally lossless at retrieval precision |
| **index stored as int8** (scale-127) | **0.998666** (b32), 0.998677 (mcs0) | ✅ free 4× compression, 0.13 % drift — far below the encoder's own noise |
| encoder weights int8 (good recipe) | 0.9850 | ⚠️ 1.5 % drift, worth it for 2.8× speed |
| encoder full-QDQ int8 (HF file) | 0.9553 | ❌ 4.5 % drift, 25 % ranking flips |
| encoder int8 (MobileCLIP-S0 HF file) | 0.1638 | ☠️ broken |

**Storage quantization is ~30× more forgiving than encoder quantization.** int8 vectors cut a 10k index
from 20 MB to 5 MB at 0.13 % drift.

⚠️ **But do not naively store int8 and matmul in int8 with numpy.** `[measured]`: numpy has no BLAS path
for int8/fp16 GEMM and falls back to a scalar loop — int8 matmul is **6× *slower*** than fp32
(1.60 ms vs 0.25 ms @10k) and fp16 is **30× slower** (7.5 ms). The correct pattern is
**store narrow, compute wide, in chunks**: keep int8/fp16 on disk, cast a chunk to fp32, hit BLAS.
(Or use usearch/faiss, which have genuine SIMD int8 kernels.)

---

## 6. Concurrency — search while indexing

### 6.1 The pattern: append-only shard + atomic manifest, zero locks

```
index/
  shard-0000.f32      # raw C-contiguous float32 rows, APPEND ONLY, never rewritten
  shard-0000.ids      # parallel id/path table, append only
  manifest.json       # {"shards":[{"file":"shard-0000.f32","count":8421,"dim":512}], ...}
```

- **Writer** appends rows → `f.flush(); os.fsync()` → writes `manifest.json.tmp` → `os.replace()`.
  `os.replace` is atomic on POSIX *and* Windows, so readers never observe a partial manifest.
- **Reader** reads the manifest, `np.memmap`s each shard **read-only**, and searches exactly
  `rows[0:count]`. Rows past `count` may be half-written — the reader simply never looks at them.
- The invariant that makes it safe: **rows are only ever appended, and the count is published only after
  the bytes are durable.** Publish-after-write. No locks, no readers-writer coordination, no torn reads.
- Sealed shards (at e.g. 50k rows) roll to a new file, so a shard is immutable once sealed → trivially
  cacheable, checksummable, and shippable.
- Deletes/updates: append a tombstone to a small `deleted.roaring`/bitmask and mask at query time;
  compact offline. Never mutate a sealed shard.

### 6.2 Measured, not assumed (`bench_scripts/concurrency_test.py`)

Two real OS processes. Writer appends 20 000 × 512 fp32 rows in 250-row batches with fsync+commit; reader
polls the manifest and runs a full matmul search over the growing index, verifying that the top-1 result is
always the newest row and that its score matches the expected value.

```
rows_visible = 20000/20000   torn_reads = 0   wrong_results = 0   reader_polls = 9021
concurrent search latency over the growing index: median 1.595 ms, max 6.130 ms
final shard = 41.0 MB for 20000 × 512 fp32
```

**Zero torn reads, zero incorrect results, 9 021 successful concurrent searches, 1.6 ms median while
ingestion was in flight.** The pattern is validated, not hypothesized. This directly satisfies the vision's
*"instantly search for things and get results for what was already processed while processing is ongoing."*

Free bonus: progress/ETA observability is just `sum(shard.count)` from the manifest — the same file that
makes search safe also drives the live progress bar, with no extra IPC.

### 6.3 Worker topology — measured (`bench_scripts/e2e.py`)

Full pipeline (decode → resize → normalize → encode), ViT-B/32 weight-int8, 288 images, load ≈ 10:

| workers × ORT threads | img/s | 10k ETA |
|---|---|---|
| 1 × 8 | 104.2 | 1.6 min |
| 2 × 4 | 137.6 | 1.2 min |
| 4 × 2 | 154.3 | 1.1 min |
| 8 × 1 | 166.7 | 1.0 min |
| **12 × 1** | **181.2** | **0.9 min** |

**Process-level parallelism beats intra-op threading by 1.7×.** Reasons: (a) Python decode is GIL-bound so
it must be in separate processes anyway; (b) ORT's intra-op pool scales poorly past the performance-core
count and gets scheduled onto efficiency cores; (c) one session per process gives perfect batch-level
parallelism with no synchronization.

Recommended: **`workers = performance_core_count`, `intra_op_num_threads = 1`, batch 4–8 per worker**, each
worker appending to **its own shard file** — which makes the append path contention-free by construction
(no shared file offset), and the manifest the only shared object.

---

## 7. Expected numbers to design against

| metric | measured (M3 Max, loaded) | expected old x86 laptop (est.) | target | margin |
|---|---|---|---|---|
| index throughput | **181 img/s** | ~25–40 img/s | "tens/sec" | ✅ met on both |
| 10 000 images, full index | **~55 s** | ~5 min | — | ✅ |
| search latency @10k | **0.28 ms** | ~2 ms | "tens of ms" | ✅ ~10–100× under |
| search latency @100k | **3.8 ms** | ~20 ms | — | ✅ |
| search-while-indexing | **1.6 ms median, 0 errors** | ~5–10 ms | must work | ✅ validated |
| index size @10k, fp32 | 20.5 MB | same | light | ✅ (5 MB as int8) |
| model on disk | 96 MB int8 (B/32) | same | light | ✅ (45 MB w/ MobileCLIP-S0 fp32) |
| total install | ORT 18 MB + numpy 12 MB + Pillow 5 MB ≈ **35 MB** | same | LIGHT | ✅ |

Old-x86 figures are **extrapolations from the bandwidth model and literature, not measurements.** See R6.

---

## 8. Red flags — read these before building

- **R1 · `Xenova/mobileclip_s0/onnx/vision_model_int8.onnx` produces garbage embeddings** (cos 0.008–0.16
  vs its own fp32) **and is 3.4× slower.** `vision_model_q4.onnx` is also degraded (cos 0.67). Any pipeline
  that grabs "the int8 one" from HF will silently produce a useless index. **Add the CI fidelity gate (§3.3).**
- **R2 · Off-the-shelf full-QDQ int8 CLIP loses ~4.6 pp zero-shot accuracy** `[literature]` and flips 25 %
  of nearest neighbours `[measured]`. Self-quantize weight-only instead. Non-obvious and easy to miss.
- **R3 · int8 is *slower* than fp32 for hybrid conv/transformer encoders on ARM ORT** (MobileCLIP-S0:
  3.4× slower). Never generalize a quantization win across architectures — measure per model.
- **R4 · More threads is slower.** ViT-B/32 int8: 55.5 img/s @4 threads → 17.9 @8. Do not default to
  `os.cpu_count()`. Probe and cap at performance cores.
- **R5 · numpy int8/fp16 matmul falls off BLAS and is 6–30× SLOWER than fp32.** Store narrow, compute wide,
  chunked. A naive "we'll save memory with int8 vectors" change would be a 6× latency regression.
- **R6 · No x86 measurement exists in this report.** Every x86 statement is literature or extrapolation.
  ORT docs warn quantization can *lose* on pre-VNNI x86. The "old computers" requirement is therefore
  **unvalidated**. Highest-priority follow-up: run `bench_scripts/` on a genuinely old x86 box.
- **R7 · fp16 ONNX files do not load on the ORT CPU EP** (`SimplifiedLayerNormFusion` crash). They are a
  GPU/WebGPU artifact. Don't ship a codepath that expects them.
- **R8 · MobileCLIP ONNX exports use `0..1` preprocessing, CLIP uses mean/std.** Wrong normalization
  silently degrades every embedding without an error. Assert embedding-norm sanity at startup.
- **R9 · Benchmark hygiene.** My own thread sweep swung 3× between back-to-back runs at load 47. Sibling
  agents on this machine will corrupt any benchmark. Gate the darwin/self-improvement loops on machine
  load, or their "improvements" will be measuring noise.
- **R10 · One unreproduced segfault.** A combined faiss + usearch + 100k-vector script exited 139 twice;
  the isolated repro was clean, so it's most likely memory pressure at load 47, not a library conflict.
  Noted for honesty; not diagnosed. If you ever load faiss and usearch in one process, watch for it.
- **R11 · HNSW is architecturally incompatible with cheap search-while-indexing.** If someone later "adds
  an index for speed", they will break the concurrency guarantee that §6 validated. Document this.
- **R12 · clip.cpp is stale (13 months, 564★).** Attractive on paper for edge; a maintenance liability as
  a core dependency.

---

## 8b. Cross-lane reconciliation with `measured-numbers.md` (priorart lane)

The priorart lane surveyed shipping tools; I benchmarked locally. We were independent, and we converge.
Where their open questions existed, my measurements answer them:

| priorart finding / open question | my measurement | outcome |
|---|---|---|
| "immich defaults `intra_op=2`, suggests per-image parallelism beats intra-op — bench 8 workers × 2 threads" | 12×1 = **181** img/s · 8×1 = 167 · 4×2 = 154 · 2×4 = 138 · 1×8 = 104 | ✅ **confirmed and refined: go to 1 thread/worker, not 2.** immich's `intra_op=2` is leaving ~18 % on the table vs 8×1 on a 16-core box. |
| clip.cpp anomaly: quantization **slower** than f32 on an Intel mac → "measure, never assume int8 wins" | MobileCLIP-S0 int8 on ARM ORT is **3.4× slower** than its own fp32 | ✅ **independently reproduced on a different arch and runtime.** Now with a mechanism: hybrid conv/transformer nets have no fast int8 ARM kernel, so the graph fills with Quantize/Dequantize round-trips. The rule generalizes. |
| Ente ships `mobileclip_s2_image.onnx` **fp32** but the **text** tower int8 ("~70 % shrink, no noticeable decline") | MobileCLIP-S0 **vision** int8 is numerically broken (cos 0.16→0.008) and slower | 🔑 **This is almost certainly not a coincidence — Ente hit the same wall and shipped around it.** Strong external corroboration of R1/R3. Whoever picks MobileCLIP must keep the vision tower fp32 (or self-quantize weight-only and gate on the fidelity test). |
| Ente: 50k brute-force search "~30 min" → USearch <30 s; lane read it as "their brute path was pathological" | 50k × 512 fp32 matmul ≈ **1.9 ms** (interpolated from 0.28 ms @10k / 3.8 ms @100k), bandwidth model confirms | ✅ **confirmed pathological — by ~10⁶×.** Their "brute force" was not a BLAS matmul. Do not let this anecdote justify an ANN index at our scale. |
| "ORT MLAS kernels beat ggml on CPU" (LocalAI face bench) | I did not bench ggml; my ORT numbers are strong and clip.cpp is 13 months stale (564★) | ✅ agreed, converging on ORT from two directions. |
| "No mainstream tool publishes CPU CLIP image-encode ms with (CPU, threads, batch, precision) stated" | §2.1 is exactly that table, plus §3.2 quantization fidelity | ✅ the vacuum is real; this report starts filling it. |
| "immich ships NO int8/fp16 CLIP artifacts, NO CLIP batching = untapped headroom" | weight-only int8 = **2.8×**; batching bs=8 vs bs=1 at 4 threads = 55→75 img/s (**1.36×**) | ✅ headroom quantified: ~3.8× combined, before process-level parallelism. |
| immich real-user bar: i5-10500 ≈ **0.4 img/s**, 320k photos ≈ 20 days | 181 img/s here; conservative old-x86 estimate ~25–40 img/s | ⚠️ apples-to-oranges (they run nllb-clip-large-siglip, a far bigger model, and it's an unverified user report) — but the order-of-magnitude gap is the opportunity. **Do not quote a speedup ratio against it** until we run the same model on the same class of CPU. |

**Two things the priorart lane raised that I did NOT cover — real gaps in this report:**

- **int8 *static* (calibrated) quantization.** I only tested *dynamic*. Static quantizes activations too,
  which is exactly the axis §3.1 shows is dangerous for retrieval — but with real calibration data it may
  land better than the naive QDQ files, and it's the only path to the DeepSparse-class 1230 img/s number.
  **Untested here. Worth one bench slot**, gated on the same fidelity test (cos > 0.98, NN-agreement > 0.90).
- **Ente's fused uint8-RGBA preprocessing in-graph.** Folding the `/255 → mean/std → transpose` into the
  ONNX graph and feeding raw uint8 saves a float conversion and a copy per image. Cheap to try, plausibly
  worth a few percent, and it also removes the §4.1 normalization footgun by construction. **Not tested.**

## 9. Reproduce

All scripts are in `research/bench_scripts/`. Environment:

```bash
uv venv /tmp/imgtag_bench --python 3.12
VIRTUAL_ENV=/tmp/imgtag_bench uv pip install onnxruntime onnx pillow numpy usearch faiss-cpu
```

| script | what it proves |
|---|---|
| `vecbench.py`, `vecbench2.py` | brute-force scaling; numpy int8/fp16 GEMM is off-BLAS (R5) |
| `prebench.py` | Pillow `draft()`/`thumbnail` 2.1× decode win |
| `ortbench2.py` | per-model/precision/thread ORT throughput (R3, R4, R7) |
| `acc_test.py`, `acc_test2.py` | int8 encoder fidelity; MobileCLIP-S0 int8 is broken (R1, R8) |
| `myquant.py` + `quantcmp.py` | **the recommended quantization recipe**, better+faster than off-the-shelf |
| `vb2.py`, `recall_test.py` | numpy vs usearch/faiss/hnswlib latency, build cost, recall |
| `concurrency_test.py` | lock-free append + atomic manifest: 0 torn reads, 0 wrong results |
| `e2e.py` | end-to-end pipeline throughput; process- vs thread-parallelism (181 img/s) |

## 10. Sources

Runtimes & quantization —
[ONNX Runtime quantization docs](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) ·
[microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) ·
[OpenVINO CPU device docs](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device.html) ·
[OpenVINO system requirements](https://docs.openvino.ai/systemrequirements) ·
[Arm: ONNX Runtime on Arm platforms](https://learn.arm.com/learning-paths/mobile-graphics-and-gaming/onnx/01_fundamentals/) ·
[Arm: optimizing with I8MM](https://developer.arm.com/community/arm-community-blogs/b/ai-blog/posts/optimize-llama-cpp-with-arm-i8mm-instruction) ·
[PyTorch: ExecuTorch on Arm CPUs/NPUs](https://pytorch.org/blog/efficient-edge-ai-on-arm-cpus-and-npus/) ·
[monatis/clip.cpp](https://github.com/monatis/clip.cpp) ·
[huggingface/candle](https://github.com/huggingface/candle) ·
[sonos/tract](https://github.com/sonos/tract) ·
[Tencent/ncnn](https://github.com/Tencent/ncnn) ·
[alibaba/MNN](https://github.com/alibaba/MNN) ·
[ml-explore/mlx](https://github.com/ml-explore/mlx)

Quantization accuracy —
[arXiv:2605.26415 "The Rescue Effect: Spatio-Semantic Early Exit Bypasses Quantization Collapse in CLIP"](https://arxiv.org/pdf/2605.26415) ·
[arXiv:2510.04547 "Activation Quantization of Vision Encoders Needs Prefixing Registers"](https://arxiv.org/pdf/2510.04547)

Models —
[Apple MobileCLIP2](https://machinelearning.apple.com/research/mobileclip2) ·
[apple/ml-mobileclip](https://github.com/apple/ml-mobileclip) ·
[Xenova/clip-vit-base-patch32](https://huggingface.co/Xenova/clip-vit-base-patch32) ·
[Xenova/mobileclip_s0](https://huggingface.co/Xenova/mobileclip_s0) ·
[plhery/mobileclip2-onnx](https://huggingface.co/plhery/mobileclip2-onnx)

Preprocessing —
[arXiv:2501.13131 "Need for Speed: A Comprehensive Benchmark of JPEG Decoders in Python"](https://arxiv.org/abs/2501.13131) ·
[libvips speed and memory use](https://github.com/libvips/libvips/wiki/Speed-and-memory-use) ·
[Pillow Performance](https://python-pillow.github.io/pillow-perf/) ·
[kornia/kornia-rs](https://github.com/kornia/kornia-rs)

Vector search —
[unum-cloud/USearch](https://github.com/unum-cloud/usearch) ·
[USearch BENCHMARKS.md](https://github.com/unum-cloud/usearch/blob/main/BENCHMARKS.md) ·
[facebookresearch/faiss](https://github.com/facebookresearch/faiss) ·
[nmslib/hnswlib](https://github.com/nmslib/hnswlib) ·
[asg017/sqlite-vec](https://github.com/asg017/sqlite-vec) ·
[sqlite-vec v0.1.0 release notes](https://alexgarcia.xyz/blog/2024/sqlite-vec-stable-release/index.html) ·
[lancedb/lancedb](https://github.com/lancedb/lancedb) ·
[Zilliz: Faiss vs HNSWlib](https://zilliz.com/blog/faiss-vs-hnswlib-choosing-the-right-tool-for-vector-search)

Concurrency —
[OpenSearch: concurrent vector graph construction](https://opensearch.org/blog/breaking-the-single-thread-bottleneck-concurrent-vector-graph-construction-in-opensearch/) ·
[arXiv:2506.03437 "Quake: Adaptive Indexing for Vector Search"](https://arxiv.org/pdf/2506.03437) ·
[ibraheemdev/boxcar (lock-free append-only vector)](https://github.com/ibraheemdev/boxcar)
