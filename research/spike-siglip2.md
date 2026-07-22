# SPIKE — SigLIP2-base-patch16-224 ONNX, honest CPU numbers

> **STATUS: PARTIAL / LIVE — bench sweep still running.** Written mid-run per orchestrator
> request (partial + honest > complete + late). Rows below are measured, not projected.
> This file is rewritten in full when the sweep completes.
> Spike agent: MODEL `claude-opus-4-8`. Date 2026-07-22.

> **EVERY NUMBER HERE IS A PROXY.** Bench host = Apple M3 Max (arm64, NEON, 16 core =
> 12P+4E), macOS 14.4. Primary deploy target = shared **Linux x86_64, 8 GB, no GPU**
> (AVX2/AVX-512). Dynamic-int8 kernels differ per arch (ORACLE ADR-10e, per-arch quant
> law): NEON ≠ AVX2. Nothing here transfers to the Linux box without a real-server bench.

## 0. What was measured

| | |
|---|---|
| Model | `onnx-community/siglip2-base-patch16-224-ONNX` (Apache-2.0, official export) |
| Runtime | onnxruntime **1.27.0**, CPUExecutionProvider only, `inter_op=1`, `ORT_SEQUENTIAL`, `ORT_ENABLE_ALL` |
| Python / numpy / Pillow | 3.12.13 / 2.5.1 / 12.3.0 |
| Images | 100 real COCO images, `data/quick500/images/` (first 100 by filename, ~640×480 JPEG) |
| Ground truth | `data/quick500/instances_quick500.json` (COCO instances) |
| Deps | onnxruntime + numpy + Pillow + huggingface_hub **only** (ADR-7 respected; tokenizer hand-rolled, see §5) |

## 1. Artifacts downloaded (verified sha256)

Landed at `/Users/magic/Creations/ImgTag/models/siglip2-base/` (gitignored — `.gitignore`
extended with `models/**/*.onnx`, `models/**/*.bin`, `models/**/tokenizer.json`).

| File | Bytes | sha256 |
|---|---:|---|
| `vision_model.onnx` (fp32) | 371,807,752 | `c0573e3f4140c3a7c4e9cc5912bd6b26a033b46a6a8e8af26cbea262b163bcad` |
| `vision_model_int8.onnx` | 94,553,333 | `0dd31785a2713f1113ef2272472165c69d580473dae38d7b47568ac587795e70` |
| `text_model_int8.onnx` | 283,438,275 | `3a0603d3a00c05a80a6ded4743c16aaac7b1e62cdcc7e362e7ce418659b96400` |
| `tokenizer.json` | 34,363,039 | `cb9140fae3ac5122c972d37adf83e1248471a38147ad76f8215c8872c6fd8322` |
| `config.json` | 435 | `e43a9f7692d3819886a82cb2097048258d444f123c67d37ec825f9345b019cf2` |
| `preprocessor_config.json` | 394 | `9b36b57ebaf20f09bf4c22100ccc21877ea6bfe5aead0c00c59f8af8ccefacfc` |
| `tokenizer_config.json` | 47,240 | `7c3a247599e741bceba1a3fe0285aea88d1044dc1fad2caa1e48cdd9fd25f630` |
| `special_tokens_map.json` | 636 | `baec30ea10906f16adb8c18af7a34023002c1746542612b8b41c9f09e1351351` |
| `quantize_config.json` | 310 | `9b60b5877b9a1687b5cddbc06124a974cb7536af9dbf5295279bf73c3823170c` |

Sizes confirm `research/models.md` §2 (372 MB fp32 vis / 94.6 MB int8 vis / 283 MB int8 txt).
**research/models.md verified accurate on disk footprint.**

### IO signatures (observed, not assumed)

```
vision_model{,_int8}.onnx  IN  pixel_values (batch,3,H,W) float32
                           OUT last_hidden_state (b,196,768) | pooler_output (b,768)
text_model_int8.onnx       IN  input_ids (batch,seq) int64
                           OUT last_hidden_state (b,seq,768) | pooler_output (b,768)
```
Embedding = `pooler_output`, dim **768**. Outputs are **unnormalized** → L2-normalize before cosine.

### Preprocessing law (from `preprocessor_config.json`, not guessed)

`size 224×224 · resample=2 → PIL BILINEAR · rescale 1/255 · mean 0.5 · std 0.5 · NCHW`

⚠️ **Divergence from ORACLE §4 parity playbook**, which says "usually bicubic". SigLIP2's own
preprocessor config says **BILINEAR**. Believe the config. Logged for ORACLE §8.

## 2. Decode — measured separately from inference (single thread, best of 3)

| Path | s / 100 img | ms / img | img/s |
|---|---:|---:|---:|
| Pillow `draft()` → 224 BILINEAR | 0.3655 | **3.66** | **273.6** |
| Full decode → 224 BILINEAR | 0.5749 | 5.75 | 174.0 |

`draft()` = **1.57× faster** decode on COCO-sized JPEG.

### ⚠️ The project thesis is REFUTED on this dataset

ORACLE §1(a) + the planner's chaser assert *decode is the engine / the bottleneck is plumbing*.
On **640×480 COCO JPEGs** that is **false**: decode costs **3.66 ms/img** while the best
observed int8 inference so far is **33.5 ms/img** — inference is **9× the decode cost**.

Honest scoping of the refutation: COCO images are small. The thesis was formed against
consumer photo libraries (12 MP, ~4000×3000 = ~26× the pixels). Decode there would be
~50–100 ms/img and would dominate. **The thesis is dataset-dependent, not universal** —
and quick500 (the current bench corpus) sits firmly on the *model-bound* side.
Consequence: on COCO-class inputs, model/thread geometry is the lever, not the decode pool.
A follow-up spike must re-measure decode on real 12 MP photos before the claim is published.

## 3. Vision encode matrix — PARTIAL (sweep in flight)

100 real images, pre-decoded once (decode excluded — pure inference), best of 2 reps,
1 warmup batch. `peakRSS` column here is a **process high-water mark and therefore
cumulative/monotonic** — it is NOT per-config. Isolated per-config RSS runs are queued
(§7 TODO); trust only those.

| Precision | intra_op | batch | img/s | ms/img | cum. peak RSS |
|---|---:|---:|---:|---:|---:|
| int8 | 1 | 1 | 6.72 | 148.80 | 511 MB |
| int8 | 1 | 8 | 6.77 | 147.81 | 554 MB |
| int8 | 1 | 32 | 5.92 | 168.84 | 1068 MB |
| int8 | 4 | 1 | 10.68 | 93.61 | 1068 MB |
| int8 | 4 | 8 | 17.18 | 58.22 | 1068 MB |
| int8 | 4 | 32 | **18.43** | 54.25 | 1201 MB |
| int8 | 8 | 1 | 25.96 | 38.53 | 1201 MB |
| int8 | 8 | 8 | **29.87** | 33.48 | 1201 MB |
| int8 | 8 | 32 | *running* | | |
| int8 | 16 | 1/8/32 | *queued* | | |
| fp32 | 1/4/8/16 × 1/8/32 | | *queued* | | |
| int8+fp32 | **2** × 1/8/32 | | *queued (brief amendment)* | | |

### Early reads (subject to change when sweep completes)

- **intra=1 is catastrophic**: 6.7 img/s. Batching does nothing at 1 thread (6.72 → 6.77);
  batch=32 at intra=1 is *worse* (5.92). Ente's intra=1 default (research/measured-numbers.md)
  would be a disaster here.
- **immich's intra=2 default is not obviously right on 16 cores either** — intra=8 is 3.9×
  intra=1. The intra=2 row is now explicitly queued (brief amendment) since the Linux box
  is the real target.
- **Superlinear anomaly**: intra=4→8 at batch=1 gives 10.68 → 25.96 img/s (2.43× for 2×
  threads). Not explainable by thread count alone. Candidates: P-core vs E-core scheduling
  (12P+4E — intra=4 may land partly on E-cores), MLAS blocking thresholds, or turbo
  residency. **Flagged as unexplained; a confirmation re-run is queued.** Not reported as
  fact until reproduced.
- **RSS is already a problem for the 8 GB Linux box**: cumulative peak hit 1068 MB during
  int8 batch=32 and 1201 MB after. Tightened budget is ≤1.0 GB indexing / ≤1.5 GB total.
  Even allowing that this is a cumulative mark, batch=32 is the suspect. Isolated runs pending.

## 4. Text encode — PENDING

## 5. Tokenizer — hand-rolled, ADR-7 respected ✅

SigLIP2's tokenizer is **GemmaTokenizer** (BPE, 256k vocab, 580,604 merges, byte-fallback,
`<eos>` appended, pad id 0, max_len 64, `tokenizer.json` = 34 MB).
No `transformers`/`tokenizers` dependency was added. Pure-python BPE implemented
(`bench.py::BPE`, ~40 lines): normalizer `" " → "▁"`, greedy lowest-rank merge, byte fallback.

Verified against the vocab:

```
'car'                → [2269, 1]                                  ['car', '<eos>']
'dog'                → [12240, 1]                                 ['dog', '<eos>']
'pizza'              → [46722, 1]                                 ['pizza', '<eos>']
'a photo of a dog.'  → [235250, 2686, 576, 476, 5929, 235265, 1]
                       ['a','▁photo','▁of','▁a','▁dog','.','<eos>']
```

**Finding:** `tokenizer.json` load = **0.64 s** and holds a 580k-entry merge dict in RAM.
For a resident daemon (ADR-5) that is a one-time cost, but the engine should ship a
**compacted binary tokenizer** (merges → ranked array, vocab → trie) rather than parse 34 MB
of JSON per cold start. Text-tower RSS is measured both with and without the tokenizer loaded.

## 6. Retrieval sanity (precision@5 vs COCO ground truth) — PENDING

## 7. Repro

```bash
mkdir -p ~/Creations/scratchpad_dir/imgtag-spike-siglip2 && cd $_
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python onnxruntime numpy pillow huggingface_hub
.venv/bin/python dl.py        # downloads + sha256 into ImgTag/models/siglip2-base/
.venv/bin/python -u bench.py  # decode + vision matrix + text + retrieval
.venv/bin/python bench2.py intra2                   # intra_op=2 tier
.venv/bin/python bench2.py rss  int8 32 8           # isolated per-config peak RSS
.venv/bin/python bench2.py textrss 4                # text tower RSS (with tokenizer)
.venv/bin/python bench2.py textrss_notok 4          # text tower RSS (session only)
```

## 8. B1 projection — PENDING (will be stated for BOTH profiles: M3 Max measured, Linux x86 unknown-until-benched)
