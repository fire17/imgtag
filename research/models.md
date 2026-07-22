# IMGTAG — Research Lane: Image–Text Embedding Models for CPU-Only Semantic Search

**Date:** 2026-07-22
**Lane:** Embedding models (CLIP-family + successors) viable for fast CPU inference
**Target workload:** ~10,000 local images, open-vocabulary text→image search, CPU-only,
must degrade gracefully onto old machines / edge devices later.
**Method:** live web research (WebSearch + WebFetch). Every model link below was fetched
and confirmed to resolve. Numbers are attributed to their measurement context — see
[§0 Reading the numbers](#0-reading-the-numbers-honesty-notes) before trusting any latency figure.

---

## TL;DR — the three things that decide this project

1. **The three most modern SOTA families are all non-commercial.** Apple MobileCLIP/MobileCLIP2
   (`apple-amlr`, *explicitly* "does not include any commercial exploitation, product development
   or use in any commercial product or service"), Meta MetaCLIP 2 (`cc-by-nc-4.0`), and Meta
   Perception Encoder (FAIR Noncommercial Research License) are **all barred from commercial use**.
   The best *shippable* frontier is Google SigLIP 2 (Apache-2.0).
2. **Published "latency" numbers for the mobile models are Neural-Engine numbers, not CPU numbers.**
   MobileCLIP2's own reported figures are iPhone 12 Pro Max ANE latencies (1.5 ms image encode for
   S0). The same paper's reported *CPU* average latency is **~495 ms** vs 9.74 ms on NPU — a ~50×
   gap. Nothing in this table is a substitute for benchmarking on the actual target CPU.
3. **Asymmetry is the biggest engineering lever available.** The image encoder runs 10,000×
   (indexing); the text encoder runs **once per query**. Rank candidates primarily on *image*-tower
   cost. A fat text tower costs disk/RAM, not throughput — and can be eliminated entirely at search
   time by precomputing a tag-vocabulary embedding table. See [§4](#4-architectural-levers-that-beat-model-choice).

---

## 0. Reading the numbers (honesty notes)

| Caveat | Detail |
|---|---|
| Apple latencies | Measured on **iPhone 12 Pro Max Neural Engine**, image-encoder + text-encoder split. Not x86 CPU, not ARM CPU. |
| TinyCLIP throughput (pairs/s) | Reported from the paper's accelerator benchmark, **not CPU**. Useful only as *relative* ordering. |
| ImageNet zero-shot top-1 | Consistently reported and comparable across rows. This is the most trustworthy column. |
| ONNX file sizes | **Directly observed** by fetching the HuggingFace file listings. These are hard facts. |
| Published x86 CPU img/s | **Essentially nonexistent in the literature.** I searched specifically for this and found no credible published img/s figures for these models on desktop CPU. This is a genuine gap the ImgTag bench must fill — do not let anyone fabricate it. |

---

## 1. Ranked table

Ranking weights ImgTag's actual priorities: **index throughput (image tower) > license shippability
> retrieval quality > semantic flexibility > edge footprint.**

Legend — License gate: 🟢 commercial-safe · 🔴 non-commercial / research-only · 🟡 ambiguous

| # | Model / variant | Params (img+txt) | Emb dim | IN-1k 0-shot | Disk (ONNX) | ONNX / quant | License | Alive? | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **SigLIP 2 base-patch16-224** | 86M vision (+ 256k-vocab text tower) | 768 | **78.4%** | 372 MB fp32 vis / **94.6 MB int8** vis; text 1.13 GB fp32 / 283 MB int8 | ✅ official `onnx-community` int8/fp16/q4 | 🟢 **Apache-2.0** | ✅ open_clip v3.3.0 (Feb 2026) | **Best shippable quality.** Strongest text tower ⇒ best hypernym behaviour. Text tower is disk-fat but runtime-cheap (mostly vocab lookup). |
| 2 | **MobileCLIP2-S0** | 11.4M + 63.4M | 512 | 71.5% | **43 MB total (both towers)** | ✅ community ONNX (`plhery`, `Xenova`, `RuteNL`) | 🔴 `apple-amlr` **research-only** | ✅ Apple, TMLR Aug 2025 | **The speed/size frontier, full stop.** 43 MB for both encoders is unmatched. Use as the performance *ceiling reference*. Legal blocker if ImgTag ships publicly. |
| 3 | **MobileCLIP2-S2** | 35.7M + 63.4M | 512 | **77.2%** | 136 MB total | ✅ community ONNX | 🔴 `apple-amlr` | ✅ | Matches SigLIP2-base quality at ~1/3 the disk. Same license blocker. Best quality-per-byte in existence. |
| 4 | **SigLIP (v1) base-patch16-224** | 93M vision + 110M text (32k vocab) | 768 | ~76% | vis 94.1 MB int8 / txt **111 MB int8** (≈205 MB total int8) | ✅ `Xenova` int8/fp16/q4 | 🟢 **Apache-2.0** | ✅ mature, widely deployed | **The pragmatic commercial-safe pick.** English-only 32k vocab makes the text tower 2.5× smaller than SigLIP2's. Best Apache-2.0 *edge* candidate. |
| 5 | **TinyCLIP ViT-45M/32 (auto)** | 45M + 18M | 512 | 61.4% | ~63M params (≈250 MB fp32, ~65 MB int8) | ⚠️ export yourself (OpenCLIP-compatible) | 🟢 **MIT** | ⚠️ ICCV-2023 vintage, lives in `microsoft/Cream` | Commercial-safe *speed* tier. Quality is a real step down (61.4%). Good floor/baseline, not the winner. |
| 6 | **TinyCLIP ViT-63M/32 (auto)** | 63M + 31M | 512 | 63.9% | ~94M params | ⚠️ self-export | 🟢 MIT | ⚠️ 2023 | Slightly better than #5, same caveats. |
| 7 | **OpenVision v1 (tiny→huge, 26 variants)** | 5.9M → 632M vision | varies | up to 79.9% (SoViT-400M) | n/a published | ⚠️ self-export | 🟢 **Apache-2.0** | ✅ ICCV 2025, 487★ | **Only fully-open family with a genuine tiny tier (5.9M).** Has a text tower in v1. Under-documented; needs hands-on validation. Highest upside for edge. |
| 8 | **FG-CLIP 2 base** | ~0.4B total | — | beats SigLIP2/MetaCLIP2 (claimed) | — | ⚠️ self-export | 🟢 **Apache-2.0** | ✅ Oct 2025, `360CVGroup` | Apache-2.0 *and* claims to beat SigLIP2. Fine-grained + bilingual EN/ZH. Too big for edge but a strong quality-tier contender. Verify claims independently. |
| 9 | **Nomic Embed Vision v1.5** | 92M vision (+137M text) | 768 | — | — | ⚠️ self-export | 🟢 **Apache-2.0** (relicensed) | ✅ | Small, commercial-safe, shares a latent space with `nomic-embed-text` — unusual and useful if ImgTag ever wants unified text+image retrieval. |
| 10 | **UForm (unum-cloud)** | tiny | **64–768 (Matryoshka)** | — | small | ✅ **native ONNX**, XNNPACK/OpenVINO/CoreML | 🟢 **Apache-2.0** | ✅ | Purpose-built for exactly this use case. **64-dim Matryoshka embeddings** = dramatically faster ANN search. Underrated dark horse for edge. Quality unverified. |
| 11 | **OpenCLIP ViT-B/32 (LAION-2B / DataComp-XL)** | 88M + 63M | 512 | ~66–73% | ~600 MB fp32 | ✅ well-trodden | 🟢 **MIT** | ✅ 14k★, v3.3.0 Feb 2026 | The boring, bulletproof baseline. Every tool supports it. Include purely as the control arm. |
| 12 | **EVA-02-CLIP-B/16** | 149M | 512 | ~74% | — | ⚠️ self-export | 🟡 verify per-checkpoint | ✅ BAAI | Strong quality/param, but no first-class CPU story and license needs per-file confirmation. |
| 13 | **Meta Perception Encoder (PE-Core B/16)** | 86M+ | — | beats SigLIP2 (claimed SOTA) | — | ⚠️ | 🔴 **FAIR Noncommercial** | ✅ NeurIPS 2025 | Genuinely SOTA on retrieval. Non-commercial. Research reference only. |
| 14 | **MetaCLIP 2 (worldwide)** | ViT-H/14 = 2B | — | 81.3% | huge | ⚠️ | 🔴 **CC-BY-NC-4.0** | ✅ Jul 2025 | Too big for CPU *and* non-commercial. Double disqualification. |
| 15 | **jina-clip-v2** | 865M | 1024 (Matryoshka) | — | huge | ⚠️ | 🔴 **CC-BY-NC-4.0** | ✅ | 89 languages, 8k context — impressive, wrong tool. Too big + NC. |
| — | **OpenVision 2** | — | — | — | — | — | 🟢 Apache-2.0 | ✅ CVPR 2026 | ❌ **DISQUALIFIED — no text encoder.** Caption-only generative training "removes the text encoder and contrastive loss". Cannot do text→image search. Do not be seduced by the newer version number. |
| — | **Apple FastVLM** | — | — | — | — | — | — | — | ❌ **Out of scope.** It is a VLM (generates text), not a dual-tower embedding model. No shared image/text embedding space ⇒ no ANN index. |

---

## 2. Per-candidate notes

### 🥇 SigLIP 2 (Google) — `Apache-2.0`
- **Links:** [paper arXiv:2502.14786](https://arxiv.org/abs/2502.14786) · [HF collection](https://huggingface.co/collections/google/siglip2-67b5dcef38c175486e240107) · [ONNX](https://huggingface.co/onnx-community/siglip2-base-patch16-224-ONNX) · [HF blog](https://huggingface.co/blog/siglip2)
- Sizes: base (86M vision), large (303M), so400m (400M), giant (1B). NAFlex dynamic-resolution variants for base and so400m.
- OpenCLIP-reported zero-shot IN-1k: **B/32-256 = 73.9%, B/16-224 = 78.4%** (Ross Wightman, OpenCLIP maintainer).
- Trained on **109 languages**; adds captioning pretraining, self-distillation, masked prediction, online data curation over SigLIP v1.
- **Observed ONNX disk footprint (base-patch16-224):** vision 372 MB fp32 → **94.6 MB int8**; text **1.13 GB fp32 → 283 MB int8**. The text tower is enormous *because of the 256k multilingual vocab embedding table* — that is a memory-lookup cost, **not** a FLOPs cost. Runtime text latency is far cheaper than the file size suggests.
- **Verdict:** the highest-quality model ImgTag can legally ship. Its rich multilingual text tower is also the best bet for hypernym queries like "vehicle".

### 🥈 Apple MobileCLIP2 — `apple-amlr` ⚠️ RESEARCH-ONLY
- **Links:** [paper arXiv:2508.20691](https://arxiv.org/abs/2508.20691) · [GitHub apple/ml-mobileclip](https://github.com/apple/ml-mobileclip) (1.6k★, code MIT / **weights research-only**) · [HF MobileCLIP2-S0](https://huggingface.co/apple/MobileCLIP2-S0) · [ONNX exports](https://huggingface.co/plhery/mobileclip2-onnx)
- Full family (params img+txt · ANE latency ms img+txt · IN-1k):

  | Variant | Params | Latency (ANE) | IN-1k |
  |---|---|---|---|
  | MobileCLIP2-S0 | 11.4M + 63.4M | 1.5 + 3.3 ms | 71.5% |
  | MobileCLIP2-S2 | 35.7M + 63.4M | 3.6 + 3.3 ms | 77.2% |
  | MobileCLIP2-B | 86.3M + 63.4M | 10.4 + 3.3 ms | 79.4% |
  | MobileCLIP2-S3 | 125.1M + 123.6M | 8.0 + 6.6 ms | 80.7% |
  | MobileCLIP2-S4 | 321.6M + 123.6M | 19.6 + 6.6 ms | 81.9% |
  | MobileCLIP2-L/14 | 304.3M + 123.6M | 57.9 + 6.6 ms | 81.9% |

- **ONNX disk (observed):** S0 = **43 MB**, S2 = 136 MB, B = 330 MB, L-14 = 1.1 GB. Embedding dim 512 (768 for L-14). Outputs are **unnormalized** — L2-normalize before cosine similarity. Text: 49,408-token vocab, fixed 77-token sequences.
- MobileCLIP2-S4 matches SigLIP-SO400M/14 on IN-1k at **2× smaller**; beats DFN ViT-L/14 at 2.5× lower latency.
- **🔴 THE LICENSE PROBLEM — verified by reading the license text directly:** permits use "exclusively for Research Purposes", defined as "non-commercial scientific research and academic development activities", and explicitly states this "does not include any commercial exploitation, product development or use in any commercial product or service." The `timm/MobileCLIP2-*-OpenCLIP` re-uploads carry the **same** `apple-amlr` license — re-hosting does not launder it.
- **Verdict:** technically the best answer to ImgTag's brief and legally the worst. Fine for fire17's private local use; a hard blocker the moment ImgTag is published as a tool others run.

### SigLIP v1 (Google) — `Apache-2.0` — the pragmatic edge pick
- **Observed ONNX disk:** vision **94.1 MB int8**, text **111 MB int8** → **~205 MB int8 total**, vs SigLIP2's 378 MB. The saving is entirely the 32k English vocab vs 256k multilingual.
- Same sigmoid-loss architecture and ~76% IN-1k. Loses multilingual + SigLIP2's semantic upgrades.
- **Verdict:** if the Apache-2.0 constraint holds *and* edge footprint matters, this beats SigLIP2 on bytes-per-point-of-accuracy.

### TinyCLIP (Microsoft) — `MIT` ✅
- **Links:** [GitHub microsoft/Cream/TinyCLIP](https://github.com/microsoft/Cream/tree/main/TinyCLIP) · [arXiv:2309.12314](https://arxiv.org/abs/2309.12314) · [HF checkpoints](https://huggingface.co/wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M) · [model zoo](https://github.com/wkcn/TinyCLIP-model-zoo)
- **License confirmed MIT by reading the LICENSE file** — commercial use permitted.
- Zoo (accuracy · params): ViT-63M/32+Text-31M 63.9% · ViT-61M/32+Text-29M 62.4% · ViT-45M/32+Text-18M 61.4% · ViT-40M/32+Text-19M 59.8% (84.2M total) · ViT-22M/32+Text-10M 53.7% · ResNet-30M 59.1% · ResNet-19M 56.4% · ViT-8M/16 41.1%.
- *(Note: the "/32" and "/16" are patch sizes, not embedding dims — embedding dim is 512 for the ViT-32 family. Some secondary sources get this wrong.)*
- ⚠️ **Age is the risk.** ICCV 2023. Distilled from OpenAI/LAION CLIP, so it inherits 2021-era semantics. Quality is meaningfully below MobileCLIP2-S0 at comparable size — MobileCLIP2-S0 gets 71.5% with 11.4M vision params vs TinyCLIP's 61.4% with 45M.
- **Verdict:** the honest MIT-licensed speed floor. Include as a baseline; unlikely to win.

### OpenVision (UCSC-VLAA) — `Apache-2.0` — highest edge upside
- **Links:** [GitHub](https://github.com/UCSC-VLAA/OpenVision) (487★) · [arXiv:2505.04601](https://arxiv.org/html/2505.04601v1) · [project page](https://ucsc-vlaa.github.io/OpenVision/) · [HF tiny variant](https://huggingface.co/UCSC-VLAA/openvision-vit-tiny-patch8-224)
- **26 variants from 5.9M to 632.1M params** — by far the widest small-end ladder of any fully-open family. ViT-L/14-224 = 78.5% IN-1k; SoViT-400M = 79.9%.
- **⚠️ CRITICAL VERSION TRAP:** OpenVision **v1** is CLIP-style (vision **+ text** tower ⇒ zero-shot retrieval works). OpenVision **2** (CVPR 2026) "removes the text encoder and contrastive loss, keeping only the captioning objective" — **it cannot do text→image search at all.** Use v1 only.
- **Verdict:** the only Apache-2.0 family with a genuine sub-10M-param tier. If ImgTag's edge ambition is real, this deserves a bench slot despite thin documentation.

### UForm (unum-cloud) — `Apache-2.0` — dark horse
- **Links:** [GitHub unum-cloud/uform](https://github.com/unum-cloud/uform) · [HF](https://huggingface.co/unum-cloud/uform-vl-english)
- **Matryoshka embeddings from 64 to 768 dims.** A 64-dim embedding shrinks the ANN index ~12× vs 768-dim and makes brute-force search over 10k images essentially free.
- **Native ONNX** with XNNPACK / OpenVINO / CoreML / DirectML execution providers — the best out-of-the-box CPU deployment story of any candidate.
- Claims 2–4× faster inference than competitors due to small size. Quality benchmarks are thin and self-reported — **verify independently.**
- **Verdict:** purpose-built for exactly ImgTag's problem. Under-researched by the field. Worth a bench slot.

### Non-commercial SOTA (reference only)
- **Meta Perception Encoder / PE-Core** — [GitHub](https://github.com/facebookresearch/perception_models), [HF PE-Core-B16-224](https://huggingface.co/facebook/PE-Core-B16-224), NeurIPS 2025. Genuinely beats SigLIP2 on image CLIP benchmarks; first model in 3+ years to top all zero-shot categories simultaneously without JFT-3B/WebLI. **FAIR Noncommercial Research License.**
- **MetaCLIP 2** — [arXiv:2507.22062](https://arxiv.org/abs/2507.22062), ViT-H/14 = 81.3% IN-1k, 2B params, `cc-by-nc-4.0`. Too big for CPU anyway.
- **jina-clip-v2** — [arXiv:2412.08802](https://arxiv.org/abs/2412.08802), 865M, 89 languages, 8k context, Matryoshka 64–1024. `cc-by-nc-4.0`.
- **DFN-CLIP (Apple)** — [HF DFN2B-CLIP-ViT-B-16](https://huggingface.co/apple/DFN2B-CLIP-ViT-B-16). 🟡 License is inconsistent across revisions (`apple-amlr` on one, `apple-sample-code-license` on another). **Treat as unusable until legal clarity is established.**

### Adjacent techniques worth stealing (not models to bench)
- **LLM2CLIP (Microsoft, AAAI 2026 Outstanding Paper)** — [GitHub](https://github.com/microsoft/LLM2CLIP), [arXiv:2411.04997](https://arxiv.org/pdf/2411.04997). Uses an LLM as a text teacher. Lifts SigLIP-2 long-caption retrieval by **+14.8/+15.8** and multilingual by +11.9/+15.2. Directly relevant: better text-side semantics ⇒ better hypernym handling. Checkpoints are EVA02-based and large, but the *technique* informs ImgTag's query-side design.
- **RECALL — "Empowering Multimodal Embedding for Edge Devices"** — [arXiv:2409.15342](https://arxiv.org/abs/2409.15342). On-device multimodal embedding system: coarse embeddings + query-based filtering for refinement. **This is a two-stage retrieval architecture matched exactly to ImgTag's constraints — read it before finalizing the index design.**
- **clip.cpp (ggml)** — [GitHub monatis/clip.cpp](https://github.com/monatis/clip.cpp), 564★, **MIT**, no dependencies, 4/5/8-bit quantization, "4-bit quantized CLIP is only 85.6 MB". A viable pure-C++ inference path if ONNX Runtime proves too heavy for old machines. Note: supports OpenAI/LAION-format CLIP, so it constrains model choice.
- **OpenVINO** has a first-party [MobileCLIP visual-search notebook](https://docs.openvino.ai/2024/notebooks/mobileclip-video-search-with-output.html) and an `OpenVINOClipEmbedding` class — the fastest path to good x86 CPU numbers on Intel hardware.
- **ONNX int8 quantization** reportedly gives ~**3.08× speedup on text encoders** and 20–40% latency reduction for SigLIP-2/JinaCLIP-v2 vs PyTorch eager. ⚠️ Counter-evidence exists: an open ONNX Runtime issue reports `quantize_static` to uint8 being *slower* on some hardware. **Must be measured, not assumed.**

---

## 3. Recommended top 3 to bench

### 1. SigLIP 2 base-patch16-224 (int8 ONNX) — *the shippable quality anchor*
**Why:** highest zero-shot accuracy (78.4%) available under a commercial-safe licence, official
`onnx-community` int8 exports already exist (94.6 MB vision), and its 109-language,
256k-vocab text tower is the single best asset for the "vehicle finds cars + motorcycles + trucks"
requirement — hypernym generalization is a *text-tower* property, and this is the richest text
tower ImgTag can legally use. Its scary 1.13 GB fp32 text file is vocab-embedding weight, not
compute; int8 brings it to 283 MB and it runs **once per query**.
**Bench:** index throughput (img/s, 1/2/4/8 threads), text latency cold + warm, hypernym recall.

### 2. MobileCLIP2-S0 and -S2 (ONNX) — *the performance ceiling reference*
**Why:** 43 MB for *both* towers at 71.5% IN-1k (S0), and 136 MB at 77.2% (S2) — S2 essentially
matches SigLIP2-base quality at a third of the disk. Nothing else in the field is close on
quality-per-byte, and this is the only family explicitly engineered for the "old computers / edge
devices" clause. Bench it to establish **what the achievable frontier actually is**, so the
commercial-safe choice is made with the cost of that choice measured, not guessed.
**⚠️ License gate:** research-only. Fine for private local use; **must not** ship inside a public
ImgTag release. Treat its numbers as the target the Apache-2.0 stack should chase.

### 3. UForm (Apache-2.0, native ONNX, 64-dim Matryoshka) — *the edge dark horse*
**Why:** it is the only candidate designed from the start for this exact job — pocket-sized,
native ONNX across XNNPACK/OpenVINO/CoreML, and **64-dimensional Matryoshka embeddings** that
shrink the ANN index ~12× and make 10k-image search effectively instantaneous. If its retrieval
quality lands anywhere near SigLIP v1, it wins the edge tier outright on total system cost.
Its quality is the least independently verified of the three — which is precisely why it needs a
bench rather than a guess.
**Bench alongside:** SigLIP v1 base int8 (205 MB total, Apache-2.0) as the commercial-safe
fallback, and OpenCLIP ViT-B/32 (MIT) as the boring control arm.

**Alternates if a slot frees up:** OpenVision **v1** tiny/small tier (Apache-2.0, 5.9M params —
highest edge upside, thinnest documentation) and FG-CLIP 2 base (Apache-2.0, claims to beat
SigLIP2 and MetaCLIP2 — verify independently).

---

## 4. Architectural levers that beat model choice

These will likely move ImgTag's metrics more than swapping models. Flagging them for the build lanes.

1. **Exploit the 10,000:1 asymmetry.** The image tower runs 10,000×; the text tower runs once per
   query. Optimize, quantize, and thread the *image* path aggressively. A slow text tower is nearly
   free.
2. **Precompute the text side entirely.** For a fixed tag vocabulary, embed every tag once at build
   time and ship a small embedding table. At search time ImgTag then needs **no text encoder in
   memory at all** — this deletes SigLIP2's 283 MB text tower from the edge footprint. Keep the
   live text encoder only for true free-form open-vocabulary queries, and lazy-load it.
3. **Hypernym handling is a query-side problem with a cheap fix.** "vehicle" → cars + motorcycles +
   trucks does not require a bigger model. Expand the query into a prompt ensemble (WordNet
   hyponyms, or a small static expansion table), embed each, and max- or mean-pool the similarity
   scores. Prompt ensembling is well-established for lifting zero-shot accuracy, and it converts a
   model-capability problem into a lookup. **This is the highest-leverage single trick available for
   the brief's central requirement.**
4. **Matryoshka dims where available** (UForm 64–768, jina 64–1024). Truncating to 64–128 dims
   shrinks the index 6–12× with modest recall loss, and enables a two-stage search: cheap 64-dim
   scan → rerank survivors at full dim. This is exactly the RECALL paper's architecture.
5. **Normalize explicitly.** MobileCLIP2 ONNX exports emit **unnormalized** embeddings. L2-normalize
   before cosine similarity or results will be silently wrong — a classic false-positive source.
6. **ONNX Runtime + OpenVINO on Intel; XNNPACK on ARM.** Do not assume int8 is faster — measure it.
   There is documented counter-evidence of uint8 static quantization *regressing* on some hardware.

---

## 5. Red flags

| # | Risk | Detail | Mitigation |
|---|---|---|---|
| 1 | 🔴 **The best models are legally unshippable** | MobileCLIP2, PE-Core, MetaCLIP2, jina-clip-v2 are all non-commercial. Apple's licence text explicitly excludes "product development or use in any commercial product or service". | Decide the licensing posture **before** benching. If ImgTag ships publicly → SigLIP2 / SigLIP v1 / TinyCLIP / OpenVision v1 / UForm only. Consider a pluggable backend so private users can opt into MobileCLIP2 themselves. |
| 2 | 🔴 **Headline latencies are NPU numbers** | Apple's "1.5 ms" is iPhone Neural Engine. MobileCLIP2's own reported **CPU** average latency is ~495 ms vs 9.74 ms NPU — a ~50× gap. | Never quote ANE latency as a CPU expectation. Bench locally; treat all published ms figures as ordering hints only. |
| 3 | 🔴 **OpenVision 2 cannot do text→image search** | Caption-only generative training "removes the text encoder and contrastive loss". The newer version number is a trap. | Use OpenVision **v1** only. |
| 4 | 🟡 **No published x86 CPU img/s exists for any of these** | Searched specifically; found none credible. Any such number in a plan is fabricated. | ImgTag's own bench is the only source of truth. This is a genuine contribution the project can make. |
| 5 | 🟡 **DFN-CLIP licence is inconsistent** | Same HF repo shows `apple-amlr` on one revision and `apple-sample-code-license` on another. | Exclude until legally clarified. |
| 6 | 🟡 **SigLIP2's text tower is 1.13 GB fp32** | 256k multilingual vocab. Shocking on disk; cheap at runtime. | int8 → 283 MB, or precompute the tag table and drop the tower entirely (§4.2). Or use SigLIP v1 (111 MB int8 text). |
| 7 | 🟡 **TinyCLIP is 2023 vintage** | Distilled from 2021-era CLIP; 61.4% vs MobileCLIP2-S0's 71.5% at 4× the vision params. | Keep as MIT baseline, not as the expected winner. |
| 8 | 🟡 **int8 quantization is not reliably faster** | Open ONNX Runtime issue documents uint8 static quantization being *slower*. | Measure fp32 / fp16 / int8 per model per machine. Never assume. |
| 9 | 🟡 **Unnormalized ONNX outputs** | MobileCLIP2 exports emit unnormalized embeddings. | L2-normalize in the pipeline; add a unit test asserting ‖v‖≈1. |
| 10 | 🟡 **Third-party ONNX re-uploads are unaudited** | `plhery`, `Xenova`, `RuteNL`, `memojo` are community exports, not vendor-official. Licence does not change by re-hosting. | Prefer `onnx-community` (SigLIP2) where official; otherwise export from source and checksum. |

---

## Sources

[MobileCLIP2 arXiv:2508.20691](https://arxiv.org/abs/2508.20691) ·
[Apple ml-mobileclip GitHub](https://github.com/apple/ml-mobileclip) ·
[apple/MobileCLIP2-S0](https://huggingface.co/apple/MobileCLIP2-S0) ·
[timm/MobileCLIP2-S2-OpenCLIP](https://huggingface.co/timm/MobileCLIP2-S2-OpenCLIP) ·
[plhery/mobileclip2-onnx](https://huggingface.co/plhery/mobileclip2-onnx) ·
[Apple MobileCLIP2 research page](https://machinelearning.apple.com/research/mobileclip2) ·
[SigLIP 2 arXiv:2502.14786](https://arxiv.org/abs/2502.14786) ·
[SigLIP2 HF collection](https://huggingface.co/collections/google/siglip2-67b5dcef38c175486e240107) ·
[SigLIP2 HF blog](https://huggingface.co/blog/siglip2) ·
[onnx-community/siglip2-base-patch16-224-ONNX](https://huggingface.co/onnx-community/siglip2-base-patch16-224-ONNX) ·
[Xenova/siglip-base-patch16-224](https://huggingface.co/Xenova/siglip-base-patch16-224) ·
[TinyCLIP arXiv:2309.12314](https://arxiv.org/abs/2309.12314) ·
[microsoft/Cream TinyCLIP](https://github.com/microsoft/Cream/tree/main/TinyCLIP) ·
[wkcn/TinyCLIP-model-zoo](https://github.com/wkcn/TinyCLIP-model-zoo) ·
[MetaCLIP 2 arXiv:2507.22062](https://arxiv.org/abs/2507.22062) ·
[facebook/metaclip-2-worldwide-huge-378](https://huggingface.co/facebook/metaclip-2-worldwide-huge-378) ·
[facebookresearch/perception_models](https://github.com/facebookresearch/perception_models) ·
[facebook/PE-Core-B16-224](https://huggingface.co/facebook/PE-Core-B16-224) ·
[OpenVision GitHub](https://github.com/UCSC-VLAA/OpenVision) ·
[OpenVision arXiv:2505.04601](https://arxiv.org/html/2505.04601v1) ·
[OpenVision 2 project page](https://ucsc-vlaa.github.io/OpenVision2/) ·
[FG-CLIP 2 arXiv:2510.10921](https://arxiv.org/abs/2510.10921) ·
[qihoo360/fg-clip2-base](https://huggingface.co/qihoo360/fg-clip2-base) ·
[jina-clip-v2 arXiv:2412.08802](https://arxiv.org/abs/2412.08802) ·
[Nomic Embed Vision](https://www.nomic.ai/news/nomic-embed-vision) ·
[unum-cloud/uform](https://github.com/unum-cloud/uform) ·
[LLM2CLIP GitHub](https://github.com/microsoft/LLM2CLIP) ·
[RECALL arXiv:2409.15342](https://arxiv.org/abs/2409.15342) ·
[monatis/clip.cpp](https://github.com/monatis/clip.cpp) ·
[mlfoundations/open_clip](https://github.com/mlfoundations/open_clip) ·
[OpenVINO MobileCLIP notebook](https://docs.openvino.ai/2024/notebooks/mobileclip-video-search-with-output.html) ·
[EVA-CLIP arXiv:2303.15389](https://arxiv.org/abs/2303.15389) ·
[apple/DFN2B-CLIP-ViT-B-16](https://huggingface.co/apple/DFN2B-CLIP-ViT-B-16)

---

## ⚠️ CORRECTION (2026-07-22, main session — live-verified, supersedes row 13)

Row 13 (**Meta Perception Encoder / PE-Core**) wrongly lists 🔴 FAIR Noncommercial. Live check
of github.com/facebookresearch/perception_models shows **two separate licenses**: LICENSE.PE =
**Apache-2.0 for all PE checkpoints** (badge: "Model License: Apache 2.0"); the FAIR
Noncommercial Research License applies to **PLM only** (the language model). HF model card
facebook/PE-Core-S16-384 metadata also reads `apache-2.0`. → **PE-Core is commercial-safe**
and PE-Core-S16-384 (20M vision, 72.7% IN1k) / PE-Core-T16-384 (10M) join the shippable top
tier per the tagging lane's ranking. (Cross-lane conflict caught 10:07Z, resolved 10:09Z.)
