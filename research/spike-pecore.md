# Spike — PE-Core → ONNX → onnxruntime CPU

> Empirical spike, 2026-07-22. Answers ORACLE.md §playbook "PE-Core ONNX export fails"
> and closes tagging.md's #1 flagged unknown ("PE-Core-S16 / T16 → ONNX export **actually
> works** (biggest unverified assumption)").
>
> **Deploy-target framing (brief amendment):** primary target is a **shared Linux x86,
> 8 GB RAM, no GPU** server with a **≤1.0 GB indexing memory budget**. This M3 Max is a
> **proxy bench only**. Low-thread configs (2, 4) are therefore headlined; 16-thread is
> reported as secondary/negative evidence. Peak RSS is reported for every config.

---

## VERDICT: ✅ EXPORTABLE — export viability confirmed, risk closed

**PE-Core-S16-384 exports to ONNX cleanly, both towers, in 18 seconds, with bit-exact
numerics (cos = 1.000000 vs torch), working dynamic batch, and correct zero-shot
retrieval.** The ORACLE fallback playbook (drop to SigLIP2-base int8) is **not needed**.
ADR-2's "export spike first" condition is satisfied — PE-Core-S16-384 stands as primary
encoder candidate. *This verdict is architecture-independent* — it is a property of the
graph, not of the CPU, so it transfers to Linux x86 unchanged.

Three independent confirmations, strongest first:

| # | Evidence | Status |
|---|----------|--------|
| 1 | **We exported it ourselves** — `open_clip` → `torch.onnx.export` opset 17, vision + text | ✅ works, numerics verified |
| 2 | **`onnx-community/PE-Core-B16-224-ONNX` + `PE-Core-L14-336-ONNX` exist on HF** (2025-04-24), full quant matrix (fp16/int8/uint8/q4/q4f16/bnb4) | ✅ exists, but ⚠️ fixed batch=2 (see §Reuse) |
| 3 | **timm ships PE-Core natively** — `timm/vit_pe_core_{tiny,small,base,large,gigantic}_*` + open_clip mirrors `timm/PE-Core-{T,S,B,L,bigG}-*` | ✅ standard export path, no custom code |

**Correction for research/tagging.md:** PE-Core-S16-384 vision tower is **23.78 M params**
(measured), not 20 M. Full CLIP (vision + text) = **87.19 M**. Embed dim **512**.

---

## 🔴 Portability law — NEON int8 results DO NOT transfer to AVX2

Every int8 number below was produced by ORT's **ARM NEON / MLAS** kernels on Apple
silicon. Per **ORACLE ADR-10e (per-arch quant law)**, these **must not be carried over to
the Linux x86 deploy target**:

- Dynamic int8 on x86 dispatches to a **different kernel family** (AVX2 `VPMADDUBSW`, or
  AVX-512 VNNI / AMX where available). Speedup ratios routinely differ by **2× or more**
  in *either* direction versus NEON.
- **On AVX2 without VNNI, int8 dynamic quantization is frequently a net *slowdown*** —
  the u8s8 requantize path costs more than it saves. NEON's `SDOT` has no cheap AVX2
  equivalent.
- Accuracy is *also* not guaranteed identical: x86 dynamic quant commonly uses **uint8
  activations** (vs int8 on ARM), which changes the zero-point/saturation behaviour.

**What DOES transfer:** the export verdict; parity of the fp32 graph; output shapes and
the un-normalized-embedding fact; **peak RSS** (memory is architecture-stable, ±10%);
the *shape* of the thread curve (a real oversubscription effect, though its exact knee
moves with core topology).

**Required follow-up:** re-run `bench.py` / `rssbench.py` on the actual Linux x86 box
before any latency number enters BUDGETS.md. Also benchmark **fp16** and **static
per-channel int8** there — on AVX2 those often beat dynamic int8.

---

## Measured numbers — M3 Max PROXY BENCH, CPU-only, onnxruntime 1.27.0

⚠️ **Contention caveat.** Runs happened under **load average 23–131** (parallel agent
swarm). Each config below ran in a **fresh process** to keep RSS clean; the `best-of-9`
column is the honest latency estimator (least contention-polluted) and is still an
*upper bound* on true idle latency. Median is shown to expose the noise.

### ▶ HEADLINE — vision tower `PE-Core-S16-384`, low-thread configs

Batch-1 (per-image latency) and batch-8 (throughput), with peak process RSS:

| precision | intra_op | b1 best | b1 median | **b8 ms/img** | **img/s** | **peak RSS b1** | **peak RSS b8** |
|-----------|---------:|--------:|----------:|--------------:|----------:|----------------:|----------------:|
| fp32 | 1 | 382.8 ms | 386.4 | 410.7 | 2.4 | 336 MB | 726 MB |
| **fp32** | **2** | **200.7 ms** | 204.0 | **212.6** | **4.7** | **392 MB** | **742 MB** |
| **fp32** | **4** | **113.1 ms** | 116.2 | **116.6** | **8.6** | **329 MB** | **714 MB** |
| int8 | 1 | 214.6 ms | 219.7 | 219.8 | 4.5 | 161 MB | 441 MB |
| **int8** | **2** | **113.9 ms** | 118.8 | **126.2** | **7.9** | **187 MB** | **395 MB** |
| **int8** | **4** | **72.3 ms** | 125.1 | **80.3** | **12.5** | **188 MB** | **412 MB** |

### ▷ Secondary — high-thread configs (negative evidence)

| precision | intra_op | b1 best | b8 ms/img | img/s | peak RSS b8 |
|-----------|---------:|--------:|----------:|------:|------------:|
| fp32 | 16 | 347.0 ms | 242.6 | 4.1 | 767 MB |
| int8 | 16 | 221.7 ms | 304.7 | 3.3 | 449 MB |

**16 threads is 3.1× slower than 4 threads (int8, b1)** and 3.8× slower at b8. Full
sweep, single image, median of 10:

| model | th1 | th2 | **th4** | **th6** | **th8** | th12 | th16 |
|-------|----:|----:|--------:|--------:|--------:|-----:|-----:|
| fp32 | 402 | 239 | **149** | **150** | 219 | 428 | 635 |
| int8 | 232 | 129 | **84** | **78** | **77** | 170 | 539 |

M3 Max is 12 performance + 4 efficiency cores — spilling ORT's intra-op pool onto E-cores
makes every op wait on the slowest lane, and past 8 threads sync overhead dominates.

**Engine recommendation: `intra_op_num_threads = 4`, never `os.cpu_count()`.** On a
*shared* Linux box, `2` is the better-citizen default — it costs ~1.6× latency but halves
core occupancy, and a shared server's real bottleneck is co-tenancy, not single-request
latency. Make it a config value with a small startup auto-probe; the knee will sit at a
different thread count on the x86 target.

### Text tower `PE-Core-S16-384`, context_length 32

| precision | intra_op | b1 best | b8 ms/caption | **peak RSS** |
|-----------|---------:|--------:|--------------:|-------------:|
| fp32 | 1 | 27.4 ms | 26.2 | 743 MB |
| fp32 | 2 | 14.7 ms | 13.8 | 760 MB |
| fp32 | 4 | 8.4 ms | 6.9 | **850 MB** 🔴 |
| fp32 | 16 | 14.9 ms | 39.6 | 828 MB |
| **int8** | **2** | **5.9 ms** | — | **177 MB** ✅ |
| **int8** | **4** | **4.3 ms** | — | **154 MB** ✅ |

### 🔴🔴 Memory bombshell — the fp32 text tower alone blows the 1.0 GB budget

The 242 MB text `.onnx` expands to **743–866 MB resident**. Combined realistic engine
footprint, measured in one process:

| configuration | loaded RSS | peak RSS after b8 vision + b8 text |
|---|---:|---:|
| **int8 vision + fp32 text** | 785 MB | **830 MB** 🔴 83% of budget, no headroom |
| **int8 vision + int8 text** | 292 MB | **425 MB** ✅ 43% of budget |
| fp32 vision + fp32 text (implied) | ~1.1 GB | **>1.4 GB** 🔴 over budget |

**Two mitigations, both cheap, use both:**

1. **Quantize the text tower.** 242 MB → **61 MB** file, 850 MB → **154 MB** RSS
   (5.5× reduction), *and it gets faster* (8.4 → 4.3 ms). Fidelity is **excellent**:
   cos = **0.9884 / 0.9827 / 0.9892** vs torch — far better than the vision tower's
   int8 (0.94). The text tower is the *safe* one to quantize.
2. **Never hold both towers resident during indexing.** Indexing is vision-only; the text
   tower is query-path only. Load lazily and release. Vision-int8-only at b8 peaks at
   **412 MB** — leaving 588 MB of the budget free.

⚠️ **Batch size is the other memory lever:** b1 → b8 roughly **doubles** peak RSS
(int8: 188 → 412 MB; fp32: 329 → 714 MB) from activation buffers. On an 8 GB shared box,
batch-8 buys only ~1.1× throughput over batch-1 for int8 (72.3 → 80.3 ms/img — batch-8 is
actually *slower per image* here) — **so batching is not worth its memory on this target.
Recommend batch-1 or batch-2 streaming.** This alone keeps indexing under 200 MB.

### Cross-check — reusing the existing `onnx-community` export

| model | precision | th | best | ms/img | img/s |
|-------|-----------|---:|-----:|-------:|------:|
| `onnx-community/PE-Core-B16-224` vision | int8 | 6 | 61 ms / batch-2 | 30.3 | **33.0** |

B16-224 is *faster* than S16-384 despite 4× the params — 224² gives 196 patch tokens vs
384²'s 576 (2.9×), and attention is quadratic in tokens. **Resolution dominates params on
CPU.** If bench accuracy allows 224, B16-224 is the throughput winner. Worth a bench slot.

### Correctness

| check | result |
|-------|--------|
| Vision output shape / dtype | `(batch, 512)` float32 ✅ |
| Text output shape | `(batch, 512)` float32 ✅ |
| Dynamic batch (exported 1, ran 8) | ✅ works, both towers |
| **fp32 ONNX vs torch — vision** | **cos = 1.000000, max\|Δ\| = 6.26e-06** ✅ |
| **fp32 ONNX vs torch — text** | **cos = 1.000000, max\|Δ\| = 9.54e-06** ✅ |
| int8 ONNX vs torch — vision | cos = **0.9495 / 0.9358 / 0.9337** ⚠️ |
| int8 ONNX vs torch — text | cos = **0.9884 / 0.9827 / 0.9892** ✅ |
| **L2 norm of raw output** | ⚠️ **NOT normalized** — vision norms 5.23–6.50, text norms 21.31 |
| 3-image × 3-caption retrieval, fp32 v + fp32 t | ✅ **3/3 correct** |
| 3-image × 3-caption retrieval, int8 v + int8 t | ✅ **3/3 correct** |

**Retrieval sanity matrix** (real quick500 COCO images, captions from their sole category;
`a photo of a <cat>`; cosine after manual L2-norm):

```
  img[bear    ]  bear=+0.3354  zebra=+0.1793  airplane=+0.1839   → bear     OK
  img[zebra   ]  bear=+0.2123  zebra=+0.3571  airplane=+0.1728   → zebra    OK
  img[airplane]  bear=+0.1556  zebra=+0.1925  airplane=+0.2989   → airplane OK
  full-int8 diagonal: 0.3269 / 0.3425 / 0.3046  (vs fp32 0.3354 / 0.3571 / 0.2989)  OK
```

Margins are healthy (+0.10 to +0.16 over the runner-up) and survive full int8.

---

## ⚠️ Findings the engine must act on

1. **Embeddings are NOT L2-normalized.** `encode_image` / `encode_text` return raw
   projections (norms ~5–6 and ~21). Every cosine/dot-product path **must normalize
   explicitly**, and the two towers' norms differ by 4× so an un-normalized dot product is
   meaningless. Bake normalization into the embed function or fold a node into the graph.
2. **Vision int8 costs real fidelity: cos ≈ 0.93–0.95 vs fp32** (text int8 is fine at
   0.99). That is large drift for an embedding space. Retrieval survived this trivial
   3-way probe, but 0.94 will move real top-k neighbours. **Validate with recall@k on the
   bench set before adopting** — and note the speedup measured here (1.6×) is a **NEON**
   number that may vanish or invert on AVX2.
3. **`intra_op_num_threads` must not default to `cpu_count()`** — 4 (or 2 on a shared
   box), auto-probed at startup.
4. **Batch-1/2 streaming, not batch-8** — batching doubles RSS for ~zero throughput gain.
5. **Preprocessing must be `squash` resize, bilinear, mean = std = 0.5** (NOT the CLIP/
   ImageNet constants), per `open_clip_config.json`. Getting this wrong silently degrades
   every embedding.
6. **Reusing the `onnx-community` exports has a trap:** fixed batch dimensions (B16-224
   vision is hardcoded `[2, 3, 224, 224]`) and no config/tokenizer/README. **Export
   ourselves** — reuse the *path*, not the artifact.
7. **PE-Core-T16-384 (10 M, edge tier) NOT exported** — timebox. Same script, swap hub id
   to `timm/PE-Core-T-16-384`. Expect ~2–3× faster, ~1/2 the RSS.

---

## Repro — exact commands

```bash
# throwaway venv (NEVER the project venv — torch is 2 GB)
mkdir -p ~/Creations/ImgTag/.scratch/pecore && cd ~/Creations/ImgTag/.scratch/pecore
uv venv --python 3.12 .venv
VIRTUAL_ENV=$PWD/.venv uv pip install torch open_clip_torch onnx onnxruntime pillow numpy
# torch 2.13.0 · open_clip 3.3.0 · timm 1.0.28 · onnxruntime 1.27.0 · Python 3.12

# export both towers (~18 s incl. 15 s model download/load)
.venv/bin/python export.py                      # defaults to hf-hub:timm/PE-Core-S-16-384
.venv/bin/python export.py hf-hub:timm/PE-Core-T-16-384 ~/Creations/ImgTag/models pecore-t16-384 384 32

# int8 dynamic quant (~1 s each)
.venv/bin/python -c "from onnxruntime.quantization import quantize_dynamic,QuantType; M='../../models'; \
  quantize_dynamic(f'{M}/pecore-s16-384-vision.onnx', f'{M}/pecore-s16-384-vision-int8.onnx', weight_type=QuantType.QInt8); \
  quantize_dynamic(f'{M}/pecore-s16-384-text.onnx',   f'{M}/pecore-s16-384-text-int8.onnx',   weight_type=QuantType.QInt8)"

.venv/bin/python bench.py                                  # thread/precision timing table
.venv/bin/python verify.py                                 # torch parity + retrieval + thread sweep
.venv/bin/python rssbench.py <model.onnx> <threads> <batch> # one config per process: timing + peak RSS
```

Scripts live in `.scratch/pecore/` (`export.py`, `bench.py`, `verify.py`, `rssbench.py`) —
throwaway, gitignored, kept so numbers are re-derivable. **`rssbench.py` is the one to
port to the Linux x86 box** — it is dependency-light (numpy, PIL, onnxruntime) and needs
no torch. The export is ~30 lines: wrap `model.encode_image` / `model.encode_text` in an
`nn.Module`, `torch.onnx.export(..., opset_version=17, dynamo=False,
dynamic_axes={0: "batch"})`. No custom ops, no patches, no graph surgery. Two benign
warnings only (legacy-exporter deprecation; `aten::index` advanced-indexing note —
harmless here, numerics verified bit-exact).

*RSS caveat:* `ru_maxrss` is read on macOS (bytes). On Linux the same field is in
**kilobytes** — `rssbench.py` divides by 1e6 and will need `/1e3` on the target, or use
`psutil` for portability.

## Artifacts

`/Users/magic/Creations/ImgTag/models/` (gitignored via `models/*.onnx`):

| file | size | peak RSS (b1, th4) |
|------|-----:|-------------------:|
| `pecore-s16-384-vision.onnx` (fp32) | 98 MB | 329 MB |
| `pecore-s16-384-vision-int8.onnx` | 31 MB | 188 MB |
| `pecore-s16-384-text.onnx` (fp32) | 242 MB | 850 MB 🔴 |
| `pecore-s16-384-text-int8.onnx` | 61 MB | 154 MB ✅ |
| `pecore-b16-224-vision-int8.onnx` (downloaded, onnx-community) | 90 MB | — |

## Extrapolation to the 10k-image budget (BUDGETS.md input)

⚠️ **ARM proxy figures — do not commit until re-measured on Linux x86.**

Using the headline int8 th=4 batch-1 figure (**72.3 ms/img, 13.8 img/s**): 10 000 images
≈ **12.1 min** single-process, at **188 MB** peak RSS. fp32 th=4 (113.1 ms/img, 8.8 img/s)
≈ **18.9 min** at 329 MB. Both sit in/near tagging.md's predicted "~6–11 min" band for
PE-Core-S16 INT8 — the research estimate holds within ~1.2×.

**The memory headroom is the real opportunity:** at 188 MB/process, the 1.0 GB budget
fits **~4 concurrent worker processes** (each pinned to `intra_op_num_threads=2`). Given
the thread-sweep result — intra-op scaling collapses past 4–8 threads while per-process
cost stays flat — **process-level parallelism is strictly the better lever than thread
scaling.** 4 workers × 2 threads should approach ~4× the single-process rate (≈3 min for
10k) on a box with ≥8 free cores, which the fp32 text tower's 850 MB would have made
impossible. Validate on the x86 target.
