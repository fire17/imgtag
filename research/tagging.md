# IMGTAG Research Lane — Explicit Image Tagging / Recognition Models

**Lane:** tagging & recognition models · tag-vs-embedding architecture tradeoffs
**Date:** 2026-07-22 · **Author:** research-tagging (teammate lane)
**Scope:** ~10k local photos, **CPU-only**, open-vocabulary text→image search, hypernym-flexible
("vehicle" ⊇ car + motorcycle), edge-viable later.

---

## 0. Verdict first (read this, skip rest if busy)

**Winner: HYBRID — small CLIP-family embedding index as the substrate, PLUS a scored tag
vocabulary living in the SAME embedding space, PLUS training-free hierarchy expansion at
query time (CHiLS/SHiNe pattern).**

Not (a) pure embeddings, not (b) explicit tagger model. Reasons, in one line each:

- **Pure embedding search breaks on multi-object photos.** CLIP's global CLS token is
  *dominated by the most prominent object* — documented, reproduced, and the entire premise
  of TagCLIP (AAAI-24). A photo of a street with 1 small motorcycle scores low on
  "motorcycle" because the building dominates. Recall hole.
- **Pure embedding search also breaks on superordinate queries** — exactly fire17's
  "vehicle" case. CHiLS measured this: expanding a coarse label into its hyponyms and
  max/reweight-pooling gained **+25.7% on CIFAR20, +32.2% on ObjectNet, +18.5% on
  Entity30**. That gap IS the "vehicle" problem, quantified.
- **A dedicated tagger (RAM++) is genuinely more accurate at tagging** (+9.2 mAP over CLIP
  on OpenImages-uncommon, +13.0 on common) — but the only released checkpoint is
  **3.01 GB fp32 / ~750M params / Swin-L@384 ≈ 104 GFLOPs**, roughly **24× the compute of a
  small CLIP encoder**. On CPU for 10k images that is the difference between minutes and
  hours. Repo untouched since **2025-02-18**. It is not the blazing-fast edge answer.
- **The hybrid gets tags for free.** You do not need a second model. Encode a tag
  vocabulary's *text* once with the same text encoder; scoring 4,000 tags against a cached
  image embedding is a single 4000×D matmul — **sub-millisecond, done at index time**.
  RAM++'s own paper concedes the decoder path costs 0.007s for 4,585 categories; a plain
  matmul is cheaper still. You get explicit, thresholdable tags at ~0 marginal cost.
- **Explicit tags are the only credible false-positive gate.** Cosine similarity has no
  absolute meaning across queries (0.28 is a hit for one query, noise for another). A tag
  with a **per-tag calibrated threshold** is a boolean you can AND against. This is the
  single strongest argument for keeping tags in the loop, and it is an *operational*
  argument, not an accuracy one.

**Concrete stack recommendation:** `PE-Core-S16-384` (Apache-2.0, 20M vision params,
72.7% IN1k zero-shot) or `PE-Core-T16-384` (10M params, 62.1%) as the encoder, ONNX-INT8,
+ a ~4–8k tag vocabulary scored in-space at index time, + LLM/WordNet-derived hyponym
expansion at query time, + hybrid rank fusion. Details in §6.

**Do NOT ship:** YOLO-World (GPL-3.0), YOLOE (AGPL-3.0), Ultralytics runtime (AGPL-3.0),
MobileCLIP/MobileCLIP2 weights (**apple-amlr — research-only, commercial use forbidden**).
Full license analysis in §4. These are real landmines for a project fire17 intends to
publish.

---

## 1. Method & honesty notes

- Every GitHub/HF link below was **verified live** (HTTP 200) on 2026-07-22 via `curl -L`;
  repo stars / last-push / SPDX license pulled from the **GitHub API**, model file sizes and
  licenses from the **HuggingFace API** (`?blobs=true`). Those numbers are observed, not
  recalled.
- **Parameter counts I could not find published, I derived from fp32 checkpoint size**
  (bytes/4) and labelled *derived*. Cross-check: RAM++'s `ram_plus_tag_embedding_class_4585_des_51.pth`
  is 479 MB, and 4585 tags × 51 descriptions × 512 dims × 4 B = 479 MB exactly — the
  derivation method validates.
- **GFLOPs are published architecture figures** (Swin paper Table 1; ViT/DeiT standard
  tables), used as a CPU-cost *proxy*. They are NOT measured wall-clock on fire17's machine.
- **I did not run a single one of these models.** Every latency claim below is an estimate
  or a third-party figure. §8 lists exactly what must be benchmarked before any of this is
  treated as fact. Flagging this loudly because the ranking's top rows depend on it.
- One correction worth recording: the checkpoint name `ram_plus_swin_large_14m.pth` makes
  automated summarizers (and an early WebFetch in this lane) report **"14M parameters."**
  Wrong. `14m` = the **14-million-image training set**. Actual size 3.01 GB. If another lane
  reports RAM++ as a 14M-param model, that is this error propagating.

---

## 2. Ranked candidates

Ranked **for this project** (CPU-only, 10k images, open-vocab, edge-later), not by paper SOTA.

| # | Candidate | Type | Size / params | CPU viability | Accuracy claim | License | Alive? | Verdict |
|---|---|---|---|---|---|---|---|---|
| **1** | **PE-Core-S16-384** (Meta Perception Encoder) | embedding | **20M vision** (0.02B), ~16 GFLOPs @384 | ✅ **Excellent** | 72.7% IN1k ZS; PE-Core beats SigLIP2 on image benchmarks | **Apache-2.0** | ✅ pushed 2026-04-13, 2.3k★, 1.09M dl/mo | **Primary encoder.** Best accuracy-per-FLOP with a *clean* license. |
| **2** | **PE-Core-T16-384** | embedding | **10M vision** (0.01B), ~4.3 GFLOPs | ✅ **Best-in-class** | 62.1% IN1k ZS (≈ CLIP ViT-B/32's 63.3% at ~1/9 the params) | **Apache-2.0** | ✅ same repo | **Edge/fallback tier.** The genuine edge-device answer. |
| **3** | **SigLIP 2 base/16-224** | embedding | 86M vision (92M w/ head), 17.6 GFLOPs | ✅ Good | ~78% IN1k ZS; +2–3pp over SigLIP at matched size; multilingual | **Apache-2.0** | ✅ 1.6M+ dl/mo, Feb-2025 | **Quality tier.** Strongest permissive encoder if latency budget allows. Multilingual is a real bonus. |
| **4** | **Tag vocabulary scored in-space** (CHiLS / SHiNe / TagCLIP pattern) | *technique*, not a model | +0 params (reuses #1–3 text encoder) | ✅ **~0 cost** (4k×D matmul, <1 ms) | CHiLS: **+25.7 / +32.2 / +18.5%** on coarse-label sets; training-free | technique — CHiLS/SHiNe/TagCLIP code MIT/permissive | ✅ TagCLIP MIT; SHiNe (naver) live | **THE key idea.** Delivers explicit tags + hypernyms with no second model. |
| **5** | **RAM++** (Recognize Anything Plus) | tagger | **3.01 GB fp32 ≈ 750M** *(derived)*; Swin-L@384 ≈104 GFLOPs | ⚠️ **Poor** — ~24× PE-T compute; no ONNX; no small ckpt released | OpenImages-common **86.6** vs CLIP 73.6; uncommon **75.4** vs CLIP 66.2; 4,585-tag vocab | Apache-2.0 (code) | ⚠️ 3.7k★ but **last push 2025-02-18** — stale 17 mo | **Accuracy oracle, not the runtime.** Use offline to *distill/validate* the tag vocab. See §7. |
| **6** | **Florence-2-base** | VLM (tagging mode) | ~230M; ONNX **q4 vision enc = 81 MB**, int8 = 94 MB | ⚠️ Marginal — **autoregressive decode**, "several seconds/image" on CPU | Strong dense-caption / region-caption quality | **MIT** | ✅ ONNX community repo, May-2025 | **Too slow for 10k index.** Decode loop kills throughput. Good for *enriching* a few hundred hero images. |
| **7** | **WD taggers v3** (SmilingWolf) | tagger | ViT 378 MB / SwinV2 468 MB / EVA02-L 1.26 GB (≈95–315M *derived*) | ✅ Good (ONNX, batched, non-fixed batch dim) | F1 ≈ 0.44 on Danbooru tags | **Apache-2.0** | ✅ 430k dl/mo (swinv2) | ❌ **Wrong domain.** Trained on 7.2M *anime* images. Excellent engineering, useless for real photos. Excluded. |
| **8** | **OWLv2-base/16** | open-vocab **detector** | 620 MB fp32 ≈ 155M *(derived)* | ⚠️ Detector cost + per-query text | Strong zero-shot detection; ~50% cheaper than OWL-ViT | **Apache-2.0** | ✅ **1.58M dl/mo** — very alive | **Optional precision layer.** Gives *counts* + boxes ("2 cars"). Not the index. |
| **9** | **Grounding DINO** | open-vocab detector | Swin-T ~172M | ❌ Heavy for CPU | Strong; 1.5/1.6 Pro are **API-only** | Apache-2.0 (OSS ver.) | ⚠️ 10.4k★, **last push 2024-08-12** | **Skip.** OSS branch stale; the good versions are closed API. |
| **10** | **YOLO-World** | open-vocab detector | v2.1 (Feb-2025) | ✅ Fast | 35.4 AP LVIS @52 FPS (V100) | ❌ **GPL-3.0** | ⚠️ push 2025-02-26 | ❌ **License landmine.** GPL-3.0 infects a published project. |
| **11** | **YOLOE** | open-vocab detect+seg | S/M edge-suitable | ✅ Fast, CoreML/TFLite/ONNX | Re-parameterizes to zero extra inference cost | ❌ **AGPL-3.0** (+ Ultralytics AGPL) | ✅ push 2025-06-26 | ❌ **License landmine.** AGPL is worse than GPL for a hosted app. |
| **12** | **MobileCLIP2-S0/S2** | embedding | S0: 11.4M img + 42.4M txt, 71.5% IN1k; S2: 35.7M+63.4M, 77.2% | ✅ Excellent on paper | S0 ≈ OpenAI ViT-B/16 at 4.8× faster / 2.8× smaller | ❌ **apple-amlr — RESEARCH ONLY** | ✅ Oct-2025 | ❌ **Disqualified on license.** See §4 — this one is a trap. |
| — | **Tag2Text** | tagger+caption | Swin-B, 3,400 tags | ⚠️ | ICLR-2024; superseded | Apache-2.0 | superseded by RAM++ | Historical. RAM++ dominates it. |
| — | **OTTER** (arXiv 2510.00652, Oct-2025) | open-tagging | CLIP/SigLIP backbone | ? | benchmarks vs RAM++ & CLIP | CC-BY-4.0 (paper) | ⚠️ **no repo found** | **Watch item.** Newest published open-tagging work; no verifiable code → not a candidate today. |

---

## 3. Candidate detail cards (the ones that matter)

### #1/#2 — Meta Perception Encoder (PE-Core) ⭐ the find of this lane
`github.com/facebookresearch/perception_models` · `huggingface.co/facebook/PE-Core-S16-384`

| Variant | Vision params | Res | IN1k ZS |
|---|---|---|---|
| PE-Core-**T**/16 | **0.01B (10M)** | 384 | 62.1% |
| PE-Core-**S**/16 | **0.02B (20M)** | 384 | 72.7% |
| PE-Core-B/16 | 0.09B (90M) | 224 | 78.4% |
| PE-Core-L/14 | 0.32B | 336 | 83.5% |
| PE-Core-G/14 | 1.88B | 448 | 85.4% |

Why this is the answer nobody expected: **PE-Core-S16 hits 72.7% IN1k zero-shot with 20M
vision params under Apache-2.0.** That is MobileCLIP2-S0 territory (71.5%) *without* the
research-only license. Repo is actively maintained (push 2026-04-13), Meta released PE-AV in
Dec-2025 — this is a living line, unlike RAM++.

- ✅ Apache-2.0 weights AND code. Commercial-safe. No asterisk.
- ✅ Tiny. T/16 at ~4.3 GFLOPs is *cheaper than CLIP ViT-B/32* (4.4) and far more accurate
  per FLOP.
- ✅ OpenCLIP-compatible loading + `timm/PE-Core-B-16` mirror → standard ONNX export path.
- ⚠️ **README documents no ONNX export.** Export is a standard `torch.onnx`/`optimum` job
  but is *unverified* — this is the #1 integration risk in the recommendation. Budget a
  half-day; have SigLIP2 (#3, well-trodden ONNX path) as the fallback.
- ⚠️ 384px input for T/S means 576 tokens — resolution partly offsets the small param count.
  Benchmark T@384 vs SigLIP2-B@224 head-to-head; do not assume the smaller model wins.

### #5 — RAM++ (the honest teardown)
`github.com/xinyu1205/recognize-anything` · Apache-2.0 · 3.69k★ · **last push 2025-02-18**

The accuracy is real and it is the reason to take explicit tagging seriously at all:

| Model | OpenImages-common | OpenImages-uncommon | ImageNet-multi | HICO |
|---|---|---|---|---|
| CLIP | 73.6 | 66.2 | 56.6 | 26.8 |
| RAM | 86.5 | 68.8 | 71.4 | 32.9 |
| **RAM++** | **86.6** | **75.4** | **72.4** | **37.7** |

**+9.2 mAP over CLIP on uncommon categories.** That is the single strongest evidence in this
entire lane that explicit tagging supervision beats raw CLIP similarity for *tag assignment*.

But for THIS project:

- ❌ Only released checkpoint is 3.01 GB fp32, `swin_large`, 384px → **~104 GFLOPs**.
- ❌ No ONNX, no quantized release, no small checkpoint. Everything is a port you write.
- ❌ Repo stale 17 months.
- 🤔 **Paper's own ablation contradicts the release:** RAM++'s main implementation is
  **SwinBase**, and the authors report Swin**Large** *decreasing* open-set performance
  (86.4 common but 75.0 uncommon, 53.4 ImageNet-uncommon — worse than SwinBase). **The
  checkpoint they shipped is the configuration their paper says is worse at the open-set
  task we care about.** Anyone adopting RAM++ naively inherits that.
- ✅ Genuinely useful architectural lesson: the recognition decoder is **0.007s for 4,585
  categories** vs 86.76s for ITM-based scoring. Cost is the backbone, never the vocabulary.
  This is precisely why bolting a tag vocabulary onto a cheap encoder (§6) is sound.

**Use:** run RAM++ **offline, once**, on a sample of the corpus to auto-derive and validate
the tag vocabulary + per-tag thresholds; then throw it away. Do not put it in the hot path.

### #4 — The technique tier: CHiLS / SHiNe / TagCLIP
Three training-free papers that together *are* the architecture.

- **CHiLS** (arXiv 2302.02551) — coarse class → subclass set (WordNet lookup, or GPT-3
  "generate 10 types of X"), classify over the union, map back to parent, combine by
  **multiplicative reweighting** (`p_sub × p_super`) — which downweights super/sub
  disagreement and defers to subclasses when the superclass score is flat. **+25.7% CIFAR20,
  +32.2% ObjectNet, +18.5% Entity30** with a true hierarchy; only +0.3–5.4% with GPT-3
  generated ones. *Stated limitation: only helps where a hierarchy plausibly exists; no
  clean way to validate in a true zero-shot setting.* **← this is the "vehicle" fix, measured.**
  The true-vs-generated gap says: **curate the hierarchy for the top ~200 query terms; do not
  rely on LLM generation alone.**
- **SHiNe** (CVPR-2024, `github.com/naver/shine`) — same idea for open-vocab *detection*:
  fuses a WordNet/LLM hierarchy into a single "nexus" classifier, training-free, modest
  inference overhead, robust across vocabulary granularity.
- **TagCLIP** (AAAI-2024, MIT, `github.com/linyq2117/TagCLIP`) — training-free fix for CLIP's
  multi-label weakness: drop the final attention op, use **local patch features** instead of
  the global CLS token. Motivation quoted directly: *the global feature is dominated by the
  most prominent class.* Stale (Jan-2024) and small (115★) — mine it for the idea, don't
  depend on the code.

---

## 4. License landmine map ⚠️

| Model | License | Commercial? | Note |
|---|---|---|---|
| PE-Core (all) | Apache-2.0 | ✅ | Clean, weights + code |
| SigLIP 2 | Apache-2.0 | ✅ | Clean |
| RAM / RAM++ / Tag2Text | Apache-2.0 | ✅ | Clean, but stale + heavy |
| WD taggers v3 | Apache-2.0 | ✅ | Clean, wrong domain |
| Florence-2 | MIT | ✅ | Clean |
| OWLv2 | Apache-2.0 | ✅ | Clean |
| Grounding DINO (OSS) | Apache-2.0 | ✅ | 1.5/1.6 Pro are closed API |
| **YOLO-World** | **GPL-3.0** | ❌ | Copyleft infection |
| **YOLOE** | **AGPL-3.0** | ❌ | Worse — network-use clause |
| **Ultralytics runtime** | **AGPL-3.0** | ❌ | Catches you even via the YOLOE tutorial path |
| **MobileCLIP / MobileCLIP2 weights** | **apple-amlr** | ❌ | See below |

**The MobileCLIP trap, in detail.** `apple/ml-mobileclip`'s root `LICENSE` is **MIT** — and
the GitHub API reports the repo as `NOASSERTION`, and a naive fetch of that LICENSE file
reports "MIT, commercial use permitted." All true, and all about the **code**. The *weights*
ship under a separate `LICENSE_MODELS`, and HF tags `apple/MobileCLIP2-S0` as `apple-amlr`:

> "...limited license, to use, copy, modify, distribute, and create Model Derivatives ...
> **exclusively for Research Purposes.** ... 'Research Purposes' **does not include any
> commercial exploitation, product development or use in any commercial product or service.**"

Also note *revocable*, and the restriction transitively binds any Model Derivative — so a
distilled or fine-tuned MobileCLIP2 is still research-only. MobileCLIP2 is otherwise the most
attractive small encoder in this survey (S0: 71.5% IN1k at 11.4M params, 1.5 ms image
latency). **It must not go into a shipped IMGTAG.** PE-Core-S16 is the Apache-2.0 substitute
at ~equal accuracy, which is why it tops the ranking.

---

## 5. The core question, answered with evidence

> For semantic flexibility (hypernyms: "vehicle" ⊇ car, motorcycle) — is (a) CLIP-style
> embedding search, (b) explicit tags + WordNet hierarchy, or (c) hybrid the SOTA approach?

**(c) hybrid — and the strong form of the claim is that (a) and (b) each fail in a way the
other repairs.**

**Where pure embeddings (a) fail — three independent, documented failures:**

1. **Multi-object dilution.** CLIP's global feature is dominated by the most prominent class
   (TagCLIP's founding observation; FreeSeg states it flatly: *"CLIP is not good at zero-shot
   multi-label classification, because one image often contains multiple objects, while the
   text descriptions usually miss some of them"*). Small/secondary objects are recall holes.
   **Direct hit on fire17's spec:** *"all of the images with one or more cars"* — a photo
   where a car is incidental is exactly the case that gets missed.
2. **Superordinate weakness.** CLIP degrades on abstract/superordinate categories vs concrete
   ones; zero-shot transfer work finds gains from replacing abstract category names with
   concrete ones. CHiLS quantifies the recoverable gap at **+18–32%** on coarse-label
   benchmarks.
3. **No calibrated threshold.** Cosine scores are not probabilities; 0.80 does not mean
   "80% similar," and the usable cutoff drifts per query and per corpus. Mitigations exist
   (adaptive mean−1σ over top-k; largest-consecutive-score-drop; conformal prediction) but
   all are *relative* — none give a stable absolute "is there a car in this photo, yes/no."

**Where pure explicit tagging (b) fails:**

1. **Closed vocabulary at index time.** Anything outside the 4,585 tags is unsearchable.
   fire17 wants free-text search; a fixed tag list cannot serve "sunset at the beach" or
   "someone eating cake indoors."
2. **Cost.** Best tagger available is ~750M params / ~104 GFLOPs with no ONNX path.
3. **Hierarchy is manual.** WordNet gives `car → motor_vehicle → vehicle` cleanly, but
   WordNet's photo-domain coverage is patchy and its senses are ambiguous (`crane`:
   bird or machine).
4. **Real-world proof:** **PhotoPrism** is the tags-only design — labels are searchable
   keywords, and the documented limitation is exactly *"No semantic search. You need exact
   labels."* **Immich** is the embeddings-only design (CLIP + pgvecto.rs), praised for
   natural-language recall and correspondingly weak at precise "does this contain X"
   filtering. **The two leading self-hosted photo apps have each implemented one half of the
   answer, and each has the predicted weakness.** IMGTAG's opening is to be the first to do
   both.

**Why hybrid wins, and why it is nearly free:** the tag vocabulary does not need a second
model. Encode 4–8k tag strings *once* with the encoder's own text tower. Then per image,
scoring the full vocabulary is one `[4000×D] @ [D]` matmul against an embedding you already
computed — **microseconds, at index time, amortized to zero.** RAM++'s own efficiency
result (0.007s for 4,585 categories vs 86.76s for ITM) confirms vocabulary scoring is never
the bottleneck; the backbone is. So the hybrid's marginal cost over embedding-only is
**~0**, and it buys: explicit thresholdable tags, hypernym expansion, and a
false-positive gate. There is no version of this tradeoff where embedding-only is correct.

Production retrieval practice agrees independently: the standard 2025–26 pattern is
**broad recall first-stage → precise rerank second-stage**, sparse+dense fused. Tags are the
sparse signal; embeddings are the dense one. This is textbook hybrid search applied to pixels.

---

## 6. Recommended architecture for IMGTAG

```
INDEX TIME (per image, one encoder pass — the only real cost)
  image ──► PE-Core-S16-384 (ONNX INT8) ──► v (D-dim, L2-normed)
                                             │
              ┌──────────────────────────────┼──────────────────────────────┐
              ▼                              ▼                              ▼
      store v in ANN index          T · v  ──► tag scores            store EXIF/path/
      (hnswlib / usearch)           (T = [N_tags × D] tag-text        dataset id
                                     matrix, encoded ONCE)
                                          │
                                          ▼
                                  per-tag calibrated threshold τ_t
                                          │
                                          ▼
                                  sparse tag posting lists (roaring bitmaps)

QUERY TIME
  "vehicle" ──► text encoder ──► q
       │
       ├─► DENSE:  ANN top-K on v·q                     (recall — free-text, any phrasing)
       │
       └─► SPARSE: expand "vehicle" → {car, truck, motorcycle, bus, van, bicycle…}
                   via curated WordNet hyponyms ∪ LLM-generated (CHiLS)
                   → union/max over tag posting lists   (precision — thresholded, gateable)

  FUSE (RRF or weighted) ──► ranked results + per-result WHY (which tag fired, what score)
```

**Component choices**

| Slot | Pick | Why |
|---|---|---|
| Encoder | **PE-Core-S16-384**, ONNX INT8 | Apache-2.0, 20M params, 72.7% IN1k. Fallback **SigLIP2-B/16** (better ONNX track record), edge tier **PE-Core-T16-384** |
| Tag vocab | **~4–8k**, seeded from OpenImages/RAM++'s 4,585 list, pruned to photo-relevant | Proven-useful vocabulary; RAM++'s list is Apache-2.0 and reusable as *data* |
| Tag scoring | `T @ v` at index time, prompt-ensembled tag text ("a photo of a {tag}") | ~0 marginal cost; prompt ensembling is the standard cheap accuracy win |
| Thresholds | **Per-tag τ_t**, calibrated on a labelled sample | The false-positive gate. §7 |
| Hierarchy | Curated WordNet hyponyms for top ~200 query terms + LLM fill-in, cached to a static JSON | CHiLS: true hierarchies give +18–32%, LLM-generated only +0.3–5.4% — curate the head, generate the tail |
| Aggregation | CHiLS multiplicative reweight (`p_sub × p_super`) or max-over-hyponyms | Both training-free; A/B them |
| ANN | hnswlib / usearch, 10k vectors | 10k × 512 fp32 = 20 MB. Fits in L3-ish. Brute force is honestly viable too — benchmark before adding an index |
| Sparse | Roaring bitmaps per tag | Instant boolean AND/OR, trivially incremental |
| Fusion | RRF, weight tunable | Standard, no training |

**Fits the spec's operational asks:** both structures are **incrementally appendable**, so
"search while indexing is still running" works by construction — the ANN index and the
bitmaps are both valid at any prefix of the corpus. Per-image cost is one encoder pass, so
images/sec and ETA are honest to report.

---

## 7. False-positive control — the analysis the brief asked for

fire17 explicitly wants "minimization of any false positives." This is where explicit tags
earn their place, and it is worth being precise about *why*.

**The embedding-only failure mode.** ANN returns top-K by cosine, always. Search "car" in a
corpus with zero cars and you get K confidently-ranked non-cars. There is no "no results"
state. Documented in open_clip's own discussions (*"Handling Irrelevant Results in CLIP-Based
Image Retrieval When No Match Exists"*) and in threshold-consistency work: choosing one
threshold across diverse classes and distributions requires per-dataset manual tuning and
still leaks. Score scales also shift with the encoder, the prompt, and the corpus — so any
threshold you tune is fragile to every one of those changing.

**The tag gate.** A per-tag threshold τ_t is calibrated **once per tag** against a labelled
sample, not per query. `car` becomes a stable boolean. Then:

- **AND-gating:** dense top-K ∩ `has_tag(car)` → dense provides recall, the tag kills the
  tail of confident-but-wrong hits.
- **Empty-result honesty:** if no image passes τ for any expanded hyponym, return **zero
  results** instead of K plausible wrong ones. This is a UX capability embedding-only search
  structurally cannot offer, and it is what "correctness" means to a user typing "car."
- **Asymmetric τ per tag:** rare/high-risk tags get conservative thresholds, common ones
  permissive. Impossible with a single global cosine cutoff.
- **Explainability:** "matched because tag `motorcycle` scored 0.71 ≥ τ=0.65, and
  `motorcycle` is a hyponym of your query `vehicle`." Debuggable, tunable, and it makes the
  candidate-benchmark system fire17 asked for actually meaningful.

**The tradeoff, stated honestly.** Tags gate false positives but introduce **false
negatives** at the τ boundary — a real car scoring 0.63 against τ=0.65 vanishes. That is
exactly why the dense arm must stay in the fusion rather than being replaced by tags. Run
both, fuse, and expose a precision/recall knob. Recommended default: **dense-recall-first,
tag-boosted** (tags raise rank rather than hard-filter) with a user-visible "strict mode"
that flips tags to a hard AND. Calibrate τ per tag by sweeping for max F1 on a labelled
sample — and note the sample is the expensive part, so plan for ~200–500 hand-labelled
images across the top ~50 tags, or bootstrap the labels from RAM++ run offline (§5's
"accuracy oracle" role).

---

## 8. CPU throughput & edge viability — estimates, and what to actually measure

**Relative compute (published GFLOPs, image encoder only, single image):**

| Model | GFLOPs | ×PE-T | Note |
|---|---|---|---|
| PE-Core-T/16 @384 | ~4.3 | 1.0× | derived from ViT-Ti scaling |
| CLIP ViT-B/32 @224 | 4.4 | ~1.0× | reference point |
| PE-Core-S/16 @384 | ~16 | ~3.7× | derived from ViT-S scaling |
| SigLIP2-B/16 @224 | 17.6 | ~4.1× | published ViT-B/16 |
| Swin-B @384 (RAM++ paper config) | 47.0 | ~11× | published |
| **Swin-L @384 (RAM++ released ckpt)** | **103.9** | **~24×** | published |
| CLIP ViT-L/14 @336 | ~191 | ~44× | published |

**Third-party CPU anchors (bracketing, not measured here):**
- Unoptimized CLIP ViT-B/32 ONNX on CPU: **~8 img/s** (batch 64).
- INT8-quantized ViT-B/32 @256, 64-core AVX512 + VNNI: **~1230 img/s** — showing quantization
  + wide SIMD is worth 2 orders of magnitude, and that hardware dominates any model choice.
- FastEmbed claims ~12× CPU speedup via ONNX, 10–15× throughput on M1/Snapdragon.

**Rough projection for 10k images** (fp32-to-INT8 ≈ 2–4× assumed, modern multicore laptop —
**estimate, must be verified**):

| Stack | Est. throughput | Est. 10k wall-clock |
|---|---|---|
| PE-Core-T16 INT8 | ~40–80 img/s | **~2–4 min** |
| PE-Core-S16 INT8 | ~15–30 img/s | **~6–11 min** |
| SigLIP2-B/16 INT8 | ~12–25 img/s | ~7–14 min |
| RAM++ Swin-L fp32 | ~1–3 img/s | **~1–3 hours** ❌ |
| Florence-2-base (AR decode) | <1 img/s | **>3 hours** ❌ |

The RAM++ and Florence-2 rows are the whole argument. "Blazing fast on CPU" and
"RAM++ in the hot path" are mutually exclusive.

**Query latency** is a non-issue at this scale for every candidate: one text-encoder pass
(~1–5 ms) + 10k×512 brute-force dot products (~20 MB, ~1–3 ms) + bitmap ops (µs). **Sub-10 ms
end-to-end without an ANN index at all.** Recommendation: **ship brute-force first**, add
hnswlib only when the corpus outgrows ~100k. Fewer moving parts, and it is genuinely faster
at 10k than an index with its traversal overhead.

**Edge viability ranking:** PE-Core-T16 (10M) > MobileCLIP2-S0 (11.4M, *license-barred*) >
PE-Core-S16 (20M) > SigLIP2-B (86M) >> Florence-2 (230M) >> RAM++ (~750M, not viable).

**⚠️ Every number in this section is an estimate or a third-party figure.** Before any of it
is treated as fact, benchmark on fire17's actual hardware:
1. PE-Core-S16 / T16 → ONNX export **actually works** (biggest unverified assumption).
2. INT8 quantized throughput, img/s, on 100 images, single-thread and all-core.
3. Accuracy delta fp32 vs INT8 on a labelled sample (quantization can cost more than the
   speed is worth — Florence-2 guidance notes generic calibration data leaves accuracy on
   the table; use 200–500 in-domain images to calibrate).
4. PE-Core vs SigLIP2 head-to-head on real retrieval, not IN1k — IN1k top-1 is a weak proxy
   for multi-object photo retrieval, which is the actual task.

---

## 9. Gaps & watch items (honest)

- **OTTER** (arXiv 2510.00652, Oct-2025) benchmarks against RAM++ and CLIP but **I could not
  locate a code repo.** Newest open-tagging work found; unverifiable → not ranked. Re-check.
- **No open-source tagger has displaced RAM++ since Oct-2023.** I searched several framings
  for 2026 releases and found none. The field's energy moved to general vision encoders
  (PE, SigLIP2) — which is itself the strongest signal that the hybrid, not the dedicated
  tagger, is where SOTA lives now. Stated as a negative result, not a certainty: absence of
  search evidence is not proof none exists.
- **PE-Core ONNX export is unverified.** Single largest technical risk in this
  recommendation. SigLIP2-B/16 is the de-risked fallback.
- **RAM++ param count is derived, not published.** 3.01 GB ÷ 4 B ≈ 750M. Derivation method
  validated against the tag-embedding file, but the split across backbone/decoder/text
  encoder is unknown.
- **GFLOPs for PE-Core T/S at 384 are extrapolated** from ViT-Ti/ViT-S scaling, not published
  by Meta. Treat ±30%.
- Not investigated this lane (other lanes / follow-up): vector-index engines, quantization
  toolchains, dataset acquisition, PQ/binary embedding compression (would matter above ~1M
  images, not at 10k).

---

## 10. Sources (all verified 200 OK, 2026-07-22)

**Models & repos**
- [xinyu1205/recognize-anything](https://github.com/xinyu1205/recognize-anything) — RAM/RAM++/Tag2Text, Apache-2.0, 3.69k★, last push 2025-02-18
- [recognize-anything-plus-model (HF)](https://huggingface.co/xinyu1205/recognize-anything-plus-model) — 3.01 GB ckpt
- [facebookresearch/perception_models](https://github.com/facebookresearch/perception_models) — PE, Apache-2.0, 2.33k★, push 2026-04-13
- [PE-Core README (variant table)](https://github.com/facebookresearch/perception_models/blob/main/apps/pe/README.md) · [facebook/PE-Core-S16-384](https://huggingface.co/facebook/PE-Core-S16-384) · [PE-Core-T16-384](https://huggingface.co/facebook/PE-Core-T16-384)
- [google/siglip2-base-patch16-224](https://huggingface.co/google/siglip2-base-patch16-224) — Apache-2.0
- [apple/ml-mobileclip](https://github.com/apple/ml-mobileclip) · [LICENSE_MODELS (apple-amlr, research-only)](https://raw.githubusercontent.com/apple/ml-mobileclip/main/LICENSE_MODELS) · [apple/MobileCLIP2-S0](https://huggingface.co/apple/MobileCLIP2-S0)
- [onnx-community/Florence-2-base](https://huggingface.co/onnx-community/Florence-2-base) — MIT, quantized ONNX variants
- [google/owlv2-base-patch16-ensemble](https://huggingface.co/google/owlv2-base-patch16-ensemble) — Apache-2.0, 1.58M dl/mo
- [SmilingWolf/wd-swinv2-tagger-v3](https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3) · [wd-vit-tagger-v3](https://huggingface.co/SmilingWolf/wd-vit-tagger-v3) — Apache-2.0, anime domain
- [AILab-CVC/YOLO-World](https://github.com/AILab-CVC/YOLO-World) — GPL-3.0 · [THU-MIG/yoloe](https://github.com/THU-MIG/yoloe) — AGPL-3.0 · [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) — Apache-2.0, stale
- [linyq2117/TagCLIP](https://github.com/linyq2117/TagCLIP) — MIT · [naver/shine](https://github.com/naver/shine)

**Papers**
- [RAM++ — Open-Set Image Tagging with Multi-Grained Text Supervision (2310.15200)](https://ar5iv.labs.arxiv.org/html/2310.15200)
- [RAM — Recognize Anything (2306.03514)](https://arxiv.org/abs/2306.03514) · [CVPRW-2024 PDF](https://openaccess.thecvf.com/content/CVPR2024W/MMFM/papers/Zhang_Recognize_Anything_A_Strong_Image_Tagging_Model_CVPRW_2024_paper.pdf)
- [CHiLS — Zero-Shot Image Classification with Hierarchical Label Sets (2302.02551)](https://ar5iv.labs.arxiv.org/html/2302.02551)
- [SHiNe — Semantic Hierarchy Nexus for Open-Vocabulary Object Detection, CVPR-2024 (2405.10053)](https://arxiv.org/pdf/2405.10053)
- [TagCLIP — AAAI-2024 (2312.12828)](https://arxiv.org/abs/2312.12828)
- [Perception Encoder (2504.13181)](https://arxiv.org/abs/2504.13181) · [MobileCLIP2 (2508.20691)](https://arxiv.org/html/2508.20691v1) · [SigLIP 2 (2502.14786)](https://arxiv.org/pdf/2502.14786)
- [OTTER — Open-Tagging via Text-Image Representation (2510.00652)](https://arxiv.org/pdf/2510.00652) — no repo found
- [YOLO-World (2401.17270)](https://arxiv.org/abs/2401.17270) · [YOLOE docs](https://docs.ultralytics.com/models/yoloe)

**Systems & practice**
- [Immich Smart Search (CLIP embeddings)](https://pixelunion.eu/blog/2026/04/immich-smart-search/) · [Immich vs PhotoPrism 2026](https://selfhostable.dev/blog/immich-vs-photoprism-photo-management-2026/)
- [open_clip — Handling Irrelevant Results When No Match Exists](https://github.com/mlfoundations/open_clip/discussions/1058)
- [Threshold-Consistent Margin Loss (2307.04047)](https://arxiv.org/html/2307.04047v2)
- [CLIP-ONNX CPU benchmarks](https://github.com/Lednik7/CLIP-ONNX/blob/main/benchmark.md) · [neuralmagic quantized CLIP ViT-B/32](https://huggingface.co/neuralmagic/CLIP-ViT-B-32-256x256-DataComp-s34B-b86K-quant-ds)
- [Hybrid search: BM25 + vector + reranking (2026)](https://www.digitalapplied.com/blog/hybrid-search-bm25-vector-reranking-reference-2026)
- [Florence-2 quantized ONNX on-device](https://mvpfactory.io/blog/quantized-vision-transformers-on-android-running-florence-2-with-onnx-runtime/)
