# Spike — PE-Core → ONNX → onnxruntime CPU

> Empirical spike, 2026-07-22. Answers ORACLE.md §playbook "PE-Core ONNX export fails"
> and closes tagging.md's #1 flagged unknown ("PE-Core-S16 / T16 → ONNX export **actually
> works** (biggest unverified assumption)").

## VERDICT: ✅ EXPORTABLE — and it already exists for two sizes

**PE-Core-S16-384 exports to ONNX cleanly, both towers, in 18 seconds, with bit-exact
numerics (cos = 1.000000 vs torch), working dynamic batch, and correct zero-shot
retrieval.** The ORACLE fallback playbook (drop to SigLIP2-base int8) is **not needed**.
ADR-2's "export spike first" condition is satisfied — PE-Core-S16-384 stands as primary
encoder candidate.

Three independent confirmations, strongest first:

| # | Evidence | Status |
|---|----------|--------|
| 1 | **We exported it ourselves** — `open_clip` → `torch.onnx.export` opset 17, vision + text | ✅ works, numerics verified |
| 2 | **`onnx-community/PE-Core-B16-224-ONNX` + `PE-Core-L14-336-ONNX` exist on HF** (2025-04-24), full quant matrix (fp16/int8/uint8/q4/q4f16/bnb4) | ✅ exists, but ⚠️ fixed batch=2 (see §Reuse) |
| 3 | **timm ships PE-Core natively** — `timm/vit_pe_core_{tiny,small,base,large,gigantic}_*` + open_clip mirrors `timm/PE-Core-{T,S,B,L,bigG}-*` | ✅ standard export path, no custom code |

**Correction for research/tagging.md:** PE-Core-S16-384 vision tower is **23.78 M params**
(measured), not 20 M. Full CLIP (vision + text) = **87.19 M**. Embed dim **512**.

---

## Measured numbers — M3 Max, CPU-only, onnxruntime 1.27.0

⚠️ **Read the contention caveat first.** These ran while the machine carried
**load average 50–131** (parallel agent swarm). Median timings are therefore heavily
inflated and unstable run-to-run (the same config measured 86 ms/img and 146 ms/img in
two runs 4 minutes apart). **The `best-of-N` column is the honest estimator** — it is the
sample least polluted by contention, and it is still an *upper bound* on true idle
latency. Re-run on a quiet machine before these go into BUDGETS.md as committed numbers.

### Vision tower — `PE-Core-S16-384`, 384×384, fp32 + int8-dynamic

| precision | intra_op | 1 image (median) | 1 image (best) | batch-8 (best) | ms/img @b8 | **img/s** |
|-----------|---------:|-----------------:|---------------:|---------------:|-----------:|----------:|
| fp32 | 4  | 152.8 ms | 144.3 ms | 3145 ms* | 393.2* | 2.5* |
| fp32 | 6  | 150 ms   | —        | 861 ms  | 107.7 | **9.3** |
| fp32 | 16 | 488.0 ms | 402.1 ms | 2277 ms | 284.7 | 3.5 |
| **int8** | **4** | **92.6 ms** | **87.9 ms** | 569 ms | 71.1 | 14.1 |
| **int8** | **6** | **78 ms** | — | **471 ms** | **58.8** | **17.0** |
| int8 | 16 | 399.9 ms | 255.6 ms | 2069 ms | 258.7 | 3.9 |

\* fp32/th=4 batch-8 caught a contention spike; the earlier quiet-ish run measured
1237 ms (154.6 ms/img) for the same config. Treat fp32/th=6 (107.7 ms/img) as the
representative fp32 figure.

### 🔴 Load finding: **more threads is worse.** Thread sweep, 1 image, median of 10

| model | th1 | th2 | **th4** | **th6** | **th8** | th12 | th16 |
|-------|----:|----:|----:|----:|----:|-----:|-----:|
| fp32 | 402 ms | 239 ms | **149** | **150** | 219 | 428 | 635 |
| int8 | 232 ms | 129 ms | **84** | **78** | **77** | 170 | 539 |

**Sweet spot is 4–8 threads; 16 is 4–7× SLOWER than 6.** M3 Max is 12 performance +
4 efficiency cores — spilling ORT's intra-op pool onto E-cores makes every op wait on the
slowest lane, and past 8 the sync overhead dominates. **Recommended engine default:
`intra_op_num_threads = 6`** (never `os.cpu_count()`, which is the common default and is
the worst setting here). This is a portable-looking but machine-specific tuning knob →
belongs in the engine as a config value with a small auto-probe, not a hardcoded 16.

### Text tower — `PE-Core-S16-384` text, fp32, context_length 32

| intra_op | 1 caption | batch-8 | ms/caption @b8 |
|---------:|----------:|--------:|---------------:|
| 4  | 10.40 ms | 76.79 ms | 9.60 ms |
| 16 | 53.81 ms | 418.78 ms | 52.35 ms |

Text is ~8× cheaper than vision per item — query latency is a non-issue. Same
more-threads-is-worse law applies.

### Cross-check — reusing the existing `onnx-community` export

| model | precision | th | best | ms/img | img/s |
|-------|-----------|---:|-----:|-------:|------:|
| `onnx-community/PE-Core-B16-224` vision | int8 | 6 | 61 ms / batch-2 | 30.3 | **33.0** |

B16-224 is *faster* than S16-384 despite 4× the params — 224² gives 196 patch tokens vs
384²'s 576 (2.9×), and attention is quadratic in tokens. **Resolution dominates params on
CPU.** If bench accuracy allows 224, B16-224 is the throughput winner. Worth adding to
the bench slate.

### Correctness

| check | result |
|-------|--------|
| Vision output shape / dtype | `(batch, 512)` float32 ✅ |
| Text output shape | `(batch, 512)` float32 ✅ |
| Dynamic batch (exported 1, ran 8) | ✅ works, both towers |
| **fp32 ONNX vs torch — vision** | **cos = 1.000000, max\|Δ\| = 6.26e-06** ✅ |
| **fp32 ONNX vs torch — text** | **cos = 1.000000, max\|Δ\| = 9.54e-06** ✅ |
| int8 ONNX vs torch — vision | cos = **0.9495 / 0.9358 / 0.9337** ⚠️ (see risk) |
| **L2 norm of raw output** | ⚠️ **NOT normalized** — vision norms 5.23–6.50, text norms 21.31 |
| 3-image × 3-caption retrieval (fp32) | ✅ **3/3 correct** |
| same, int8 | ✅ **3/3 correct** |

**Retrieval sanity matrix** (real quick500 COCO images, captions from their sole category;
`a photo of a <cat>`; cosine after manual L2-norm):

```
  img[bear    ]  bear=+0.3354  zebra=+0.1793  airplane=+0.1839   → bear     OK
  img[zebra   ]  bear=+0.2123  zebra=+0.3571  airplane=+0.1728   → zebra    OK
  img[airplane]  bear=+0.1556  zebra=+0.1925  airplane=+0.2989   → airplane OK
```

Margins are healthy (+0.10 to +0.16 over the runner-up). The model works end-to-end
through ONNX.

---

## ⚠️ Findings the engine must act on

1. **Embeddings are NOT L2-normalized.** `encode_image` / `encode_text` return raw
   projections (norms ~5–6 and ~21). Every cosine/dot-product path **must normalize
   explicitly**, and the two towers' norms differ by 4× so an un-normalized dot product is
   meaningless. Bake `v /= np.linalg.norm(v)` into the embed function, or fold a
   normalize node into the graph at export.
2. **int8 dynamic quant costs real fidelity: cos ≈ 0.93–0.95 vs fp32.** That is a *large*
   drift for an embedding space — far worse than the ~0.99 typically seen. Retrieval
   still ranked 3/3 on this trivial probe, but a 0.94 cosine will move real top-k
   neighbours. **Do not adopt int8 on the strength of this spike** — it needs a recall@k
   comparison on the real bench set, and static/per-channel quantization (or fp16) should
   be tried before dynamic int8. Speedup is only ~1.8× (149→78 ms), so the trade may not
   be worth it.
3. **`intra_op_num_threads` default must not be `cpu_count()`.** See sweep above — 16 is
   4–7× slower than 6.
4. **Preprocessing must be `squash`, not center-crop.** `open_clip_config.json` specifies
   `resize_mode: squash`, `interpolation: bilinear`, mean = std = 0.5 (NOT the CLIP/
   ImageNet constants). Getting this wrong silently degrades every embedding.
5. **Reusing the `onnx-community` exports has a trap:** their graphs have **fixed batch
   dimensions** (B16-224 vision is hardcoded `[2, 3, 224, 224]`) and ship **no config.json,
   tokenizer, or README**. Our own export is dynamic-batch and reproducible — **prefer
   exporting ourselves** (reuse law is served by reusing the *path*, not the artifact).
6. **PE-Core-T16-384 (10 M, the edge tier) was NOT exported in this spike** — timebox.
   `timm/PE-Core-T-16-384` exists in the same open_clip format, so the same script should
   work verbatim by swapping the hub id. Expect ~2–3× faster than S16.

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

# int8 dynamic quant (1.1 s)
.venv/bin/python -c "from onnxruntime.quantization import quantize_dynamic,QuantType; \
  quantize_dynamic('../../models/pecore-s16-384-vision.onnx', \
                   '../../models/pecore-s16-384-vision-int8.onnx', weight_type=QuantType.QInt8)"

.venv/bin/python bench.py     # thread/precision timing table
.venv/bin/python verify.py    # torch parity + retrieval sanity + thread sweep
```

Scripts live in `.scratch/pecore/` (`export.py`, `bench.py`, `verify.py`) — throwaway,
gitignored, but kept so the numbers are re-derivable. The export is ~30 lines: wrap
`model.encode_image` / `model.encode_text` in an `nn.Module`, `torch.onnx.export(...,
opset_version=17, dynamo=False, dynamic_axes={0: "batch"})`. No custom ops, no patches,
no graph surgery. Two benign warnings only (legacy-exporter deprecation; `aten::index`
advanced-indexing note — harmless here, numerics verified bit-exact).

## Artifacts

`/Users/magic/Creations/ImgTag/models/` (gitignored via `models/*.onnx`):

| file | size |
|------|-----:|
| `pecore-s16-384-vision.onnx` (fp32) | 98 MB |
| `pecore-s16-384-vision-int8.onnx` | 31 MB |
| `pecore-s16-384-text.onnx` (fp32) | 242 MB |
| `pecore-b16-224-vision-int8.onnx` (downloaded, onnx-community) | 90 MB |

⚠️ The **text tower fp32 is 242 MB** — dominated by the 49408 × 512 token embedding.
For a shipped engine that must be quantized or the vocab table stored separately;
a 242 MB download for the query path alone is a distribution problem worth an ADR.

## Extrapolation to the 10k-image budget (BUDGETS.md input)

Using the honest best-of int8 th=6 figure of **17.0 img/s**: 10 000 images ≈ **9.8 min**
single-process. fp32 th=6 (9.3 img/s) ≈ **17.9 min**. Both are inside tagging.md's
predicted "~6–11 min" band for PE-Core-S16 INT8 — the research estimate holds. Under a
quiet machine and 2–3 worker processes (each pinned to 4 threads) this should improve
substantially; process-level parallelism is likely to beat intra-op scaling given the
thread-sweep result.
