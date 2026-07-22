# Violence / Abuse still-image models — research for `track-violence`

Scope: ImgTag is CPU-only (onnxruntime + numpy + Pillow), FLOPs budget nearly spent.
A NEW dedicated per-image violence model is almost certainly unaffordable, so the framing
below is: **can any permissively-licensed still-image model serve as an OFFLINE TEACHER**
that labels unlabeled images on a dev box, so we distil a tiny MLP head over our existing
CLIP embeddings? Everything ranked by that lens.

Adoption filter (hard): permissive license ONLY (Apache-2.0 / MIT / BSD). GPL / AGPL /
NC / research-only / custom-restricted (e.g. Gemma) = REJECT for adoption. A model card
with NO published metrics is itself a rejection reason (recorded as such).

Research-only lane. No weights downloaded, no corpus downloaded. Every number carries its
source link. Verified live 2026-07-22.

---

## Ranked table (teacher-candidate framing)

| # | Model / artifact | Arch / params | License | Violence/gore coverage | Published metrics | Alive? | ONNX | Verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | [LAION-AI/CLIP-based-NSFW-Detector](https://github.com/LAION-AI/CLIP-based-NSFW-Detector) — `violence_detection_vit_b_32.npy` / `_vit_l_14.npy` | numpy weight over CLIP ViT-B/32 (512-d) / ViT-L/14 (768-d) embeddings | **MIT** | violence signal, undocumented meaning | **none published** | 471★, repo live | n/a (numpy; trivially our-own-runtime) | TEACHER-CANDIDATE — architectural bullseye, but UNDOCUMENTED + no metrics; must be validated locally before trust |
| 2 | [jaranohaal/vit-base-violence-detection](https://huggingface.co/jaranohaal/vit-base-violence-detection) | ViT-base, 85.8M | **Apache-2.0** | binary violent / non-violent; **fights/assault, NOT gore** | "Test accuracy 98.80%" (in-domain only, no external eval) | 2,287 dl/mo | not shipped; standard ViT → exportable in throwaway venv | TEACHER-CANDIDATE (narrow: interpersonal-violence slice only) |
| 3 | LAION safety-pipeline zero-shot CLIP concept list ([main.py](https://github.com/LAION-AI/safety-pipeline/blob/main/main.py)) | text-prompt similarity, no weights | MIT (code) | gun/knife/blood/gore/horror/terror concepts | none | live | n/a | NOT a model — but a zero-cost baseline we already can run + prompt-vocab source |
| 4 | [Ali7880/multihead-content-moderator](https://huggingface.co/Ali7880/multihead-content-moderator) | ViT (Falconsai base) + 2 heads | **Apache-2.0** | violence head (safe/violence) | "0.9075 acc / 0.9076 F1" — **no dataset, no split, training data undisclosed** | **6 dl/mo** | `.pt` only, no ONNX | REJECTED — unaccountable provenance, near-zero adoption |
| 5 | [ShieldGemma 2](https://huggingface.co/papers/2504.01081) | Gemma-3 4B VLM | **Gemma license (NON-permissive)** | explicit "Violence & Gore" policy | internal P/R/F1 80.3/90.4/85.0; UnsafeBench-Violence 1-FPR 95.9% | live (Google) | n/a (4B, far over CPU budget) | REJECTED for adoption (license + 4B size); **best taxonomy + gore definition** |
| — | [Freepik/nsfw_image_detector](https://huggingface.co/Freepik/nsfw_image_detector) | ViT/timm | **MIT** | NSFW severity only, **no violence/gore class** | none on card | 60+ likes | none | REJECTED — out of scope (nudity, not violence) |
| — | [prithivMLmods](https://huggingface.co/blog/prithivMLmods/image-guard-models) SigLIP2 guard family | SigLIP2 ~93M | mixed | explicit/NSFW classes only; **no violence/gore label** | none tied to dataset+split | very active | none | REJECTED — no violence/gore class |
| — | Falconsai / Marqo / AdamCodd NSFW detectors | ViT / SigLIP | apache/other | **NSFW only** | Falconsai "98% acc" (nsfw); Marqo "98.56%" (nsfw) | huge (Falconsai 80M dl) | some | REJECTED — no violence/gore signal |

---

## Per-candidate notes

### 1. LAION-AI/CLIP-based-NSFW-Detector — `violence_detection_*.npy`  ← the real "CLIP-embedding heads" lead
- Verified files in repo: `violence_detection_vit_b_32.npy`, `violence_detection_vit_l_14.npy`,
  plus the NSFW autokeras heads. Repo: <https://github.com/LAION-AI/CLIP-based-NSFW-Detector> (471★, 36 forks).
- The NSFW head is documented as "a lightweight Autokeras model taking CLIP embeddings as inputs"
  → binary 0-1. The **violence** npy files carry **no documentation, no training-data description,
  and no metrics**. Format = numpy array (512-d for ViT-B/32, 768-d for ViT-L/14) — i.e. a linear
  weight applied to a CLIP embedding.
- License (quoted verbatim): "Permission is hereby granted, free of charge, to any person obtaining
  a copy of this software … to deal in the Software without restriction …" — **MIT**, © 2022 Christoph Schuhmann.
- **Why it ranks #1 for us:** ImgTag already ships `models/openclip-vitb32/`. If the npy is a linear
  weight over OpenAI/OpenCLIP ViT-B/32 space, it is a literal drop-in tiny head over our existing
  embeddings — **zero added encoder FLOPs**, no ONNX needed (numpy dot product in our runtime).
- **Why the heavy caveat:** cannot verify what "violence" means to it, what it was trained on, or how
  well it works — **no published metrics = a rejection reason per our own rule**. Treat as a
  *candidate to trial and locally validate*, not a trusted teacher. Embedding-space must be checked
  for compatibility with our ViT-B/32 export (OpenAI-CLIP vs OpenCLIP preprocessing differ).

### 2. jaranohaal/vit-base-violence-detection  — best *documented* permissive per-image classifier
- <https://huggingface.co/jaranohaal/vit-base-violence-detection> — ViT-base, 85.8M params,
  base `google/vit-base-patch16-224-in21k`.
- License: card states **"apache-2.0"** — clean for adoption/teacher use.
- Training data: **"Real Life Violence Situations" dataset (Kaggle)** — this is a *video* fights
  dataset (frames); "violence" here means **street fights / physical assault**, NOT gore, blood,
  injury, or aftermath. Binary violent / non-violent.
- Metrics: card reports "Test accuracy of 98.80%" — **in-domain test split of the same dataset
  only, no external benchmark, high leakage risk** (frames from same clips). No P/R/F1 by class.
- Alive: 2,287 downloads/month → real usage. No ONNX shipped, but a stock ViT exports cleanly in a
  throwaway torch venv → onnx (our runtime constraint satisfied).
- **Teacher verdict:** usable as an OFFLINE TEACHER for the **interpersonal-violence / fighting**
  slice of the track only. Will NOT teach gore/graphic-blood/aftermath. Its 98.8% is not credible
  as an out-of-domain estimate — expect large drop on your own site images.

### 3. LAION safety-pipeline zero-shot concept list (prompt-vocab, not a model)
- <https://github.com/LAION-AI/safety-pipeline/blob/main/main.py> uses zero-shot open_clip
  similarity against violence concepts: **"gun", "weapon", "knife", "handcuff(s)", "tank, war",
  "pistol", "blood, bloody", "blade", "rifle", "horror", "gore", "terror"** — softmax over
  `100.0 * image_features @ text_features.T`.
- ImgTag already does CLIP zero-shot → this is a **near-zero-FLOP baseline we can run today** and
  the cleanest source for our prompt vocabulary (see taxonomy section). MIT code.

### 4. Ali7880/multihead-content-moderator  (rejected)
- <https://huggingface.co/Ali7880/multihead-content-moderator> — ViT (Falconsai/nsfw base) + two
  heads: NSFW {nsfw,safe} and **Violence {safe,violence}**. License "apache-2.0".
- Metric: violence "0.9075 accuracy / 0.9076 F1" — but **no dataset named, no split, training data
  not disclosed**, ships `multihead_model.pt` (no ONNX), and **6 downloads/month**.
- REJECTED: unaccountable provenance + effectively unadopted. (This is a ViT multihead, NOT the
  CLIP-embedding-heads artifact the earlier lane referred to — that is candidate #1, LAION.)

### 5. ShieldGemma 2 (rejected for adoption; taxonomy gold)
- <https://huggingface.co/papers/2504.01081> — Gemma-3 **4B** image content-moderation VLM.
- Violence & Gore policy (verbatim): *"The image shall not contain content that depicts shocking,
  sensational, or gratuitous violence (e.g., excessive blood and gore, gratuitous violence against
  animals, extreme injury or moment of death)."* — this is the best written gore definition found.
- Metrics: internal Violence policy **P/R/F1 = 80.3 / 90.4 / 85.0**; external **UnsafeBench-Violence
  1-FPR = 95.9%**.
- REJECTED for adoption: **Gemma custom license (not Apache/MIT/BSD)** + **4B params** (orders of
  magnitude over CPU/FLOPs budget, and its output-use restrictions make it unsafe as a distillation
  teacher under our permissive-only rule). Keep only for taxonomy + as the quality bar others miss.

---

## Recommended teacher — honest verdict

**No single permissively-licensed still-image model covers the whole track with published metrics.**
Split by sub-concept:

- **Interpersonal violence / fighting / assault:** `jaranohaal/vit-base-violence-detection`
  (Apache-2.0, ViT, ONNX-exportable) is the only real, documented, permissive per-image classifier.
  Usable as a **weak offline teacher for the fights slice only**; its 98.8% is in-domain and will
  not hold on your images — treat labels as noisy, hand-audit a sample.
- **Gore / graphic blood / injury / aftermath / corpses:** **NONE adoptable.** The only strong
  gore labeler found (ShieldGemma 2) is Gemma-licensed + 4B → fails both the license and the budget
  filter. There is no permissively-licensed, published-metric still-image gore model. This is the
  biggest gap.
- **CLIP-embedding drop-in head:** LAION's `violence_detection_*.npy` (MIT) is the exact
  architecture we want and adds ~zero FLOPs, but is undocumented with no metrics → **trial-and-
  locally-validate only**, never trust blind.

**Pragmatic recommendation for ImgTag:**
1. Baseline first, ~free: run **zero-shot CLIP** with a prompt vocabulary derived from the taxonomy
   below (ImgTag already does this; no new encoder cost). Calibrate thresholds per tier.
2. If a learned head is wanted, distil a **tiny MLP over existing CLIP embeddings** using teacher
   labels assembled from: jaranohaal (fights slice) + zero-shot CLIP concept scores + a small
   HAND-LABELED gore seed set (NeuralShell Gore-Blood + your own corpus). No off-the-shelf gore
   teacher exists to lean on — the gore labels must be human-curated.
3. Optionally A/B the LAION violence npy as a free extra feature into the MLP, gated on a local
   validation pass — it costs nothing to try and nothing to drop.

Do NOT budget for a dedicated per-image violence encoder — unaffordable and unjustified given the
absence of an adoptable teacher that beats zero-shot CLIP + a distilled MLP.

---

## Taxonomy notes for our prompt vocabulary

Commercial taxonomies are NOT adoptable but their published label hierarchies are the best guide for
our zero-shot prompt vocabulary and tier boundaries.

- **AWS Rekognition** (4 top categories: Explicit Nudity, Suggestive, **Violence**, **Visually
  Disturbing**; hierarchical L1-L3). Violence child seen in API: **"Weapon Violence"** (parent
  "Violence"); coverage described as **blood, wounds, weapons, self-injury, corpses**.
  → gives us a clean split: *Violence* (weapons, physical violence, self-injury) vs *Visually
  Disturbing* (blood/gore, corpses, emaciation, air crash, explosions).
  Docs: <https://docs.aws.amazon.com/rekognition/latest/dg/moderation.html> ·
  2019 launch (violence/weapons/self-injury):
  <https://aws.amazon.com/about-aws/whats-new/2019/08/amazon-rekognition-now-detects-violence-weapons-and-self-injury-in-images-and-videos-improves-accuracy-for-nudity-detection/>
- **Google Cloud Vision SafeSearch**: single `violence` likelihood on a 5-level scale
  (VERY_UNLIKELY / UNLIKELY / POSSIBLE / LIKELY / VERY_LIKELY) — maps cleanly to graded tiers.
  <https://gcloud.readthedocs.io/en/latest/vision-safe-search.html>
- **Hive Moderation**: 50+ granular classes (weapons vs drug paraphernalia vs gore, etc.) — deepest
  taxonomy; use as a checklist of sub-concepts we might want distinct prompts for.
  (overview via Eden AI: <https://www.edenai.co/post/top-10-explicit-content-detection-apis>)
- **ShieldGemma 2** "Violence & Gore" definition (quoted above) — best single sentence for the gore
  boundary: shocking/sensational/gratuitous violence, excessive blood & gore, animal cruelty,
  extreme injury, moment of death.
- **LAION concept list** (ready-to-use CLIP prompts): gun, weapon, knife, pistol, rifle, blade,
  handcuffs, "tank, war", "blood, bloody", gore, horror, terror.

Suggested tier boundary (synthesizing the above): **T1 explicit gore/graphic-blood/corpses** (AWS
"Visually Disturbing" + ShieldGemma gore) → hard block; **T2 depicted physical violence/assault/
weapons-in-use** (AWS "Weapon/Physical Violence", jaranohaal positive) → review; **T3 mild/ambiguous
injury or weapon presence** → flag/soft.

---

## Evaluation datasets + access terms (we will NOT download these)

The field measures on datasets that are mostly request-only or video — we cannot reproduce their
metrics offline, which is itself a reason our own numbers must be self-generated on a hand-labeled
holdout.

| Dataset | Type | What it measures | Access / license |
|---|---|---|---|
| [UnsafeBench](https://unsafebench.github.io/) ([arXiv 2405.03486](https://arxiv.org/pdf/2405.03486), ACM CCS 2025) | still images, 10K real + AI-gen, 11 unsafe categories incl. **violence** | the standard image-safety-classifier benchmark (ShieldGemma reports on it) | **request-only on HF, research/education + responsible commercial use**; code MIT. Cannot redistribute; gated. |
| [NeuralShell/Gore-Blood-Dataset-v1.0](https://huggingface.co/datasets/NeuralShell/Gore-Blood-Dataset-v1.0) | still images, <1K, blood/death/gore | small gore seed set (for hand-labeled eval or teacher-seed) | **MIT, open access**, flagged Not-For-All-Audiences. Small → eval/seed only, not training. |
| Real Life Violence Situations (Kaggle) | **VIDEO** (frames) | street fights / physical violence, binary | Kaggle terms; jaranohaal trained on it |
| RWF-2000 | **VIDEO** | real-world fight vs non-fight (2000 clips) | request-only (agreement form). Video only. |
| Hockey Fight | **VIDEO** | fight vs no-fight in hockey | research; video only |
| UCF-Crime | **VIDEO** | weakly-labeled anomaly/crime incl. assault, fighting | research; video only |

Note on video models: **RWF-2000, Hockey Fight, UCF-Crime and Real Life Violence Situations are
video / action-recognition benchmarks** (temporal models: 3D-CNN, I3D, Flow-gated). They are NOT
still-image models. jaranohaal is the notable case of a *per-frame still* ViT trained off the video
frames of one of them — the only still variant found. Video temporal models are out of scope for
ImgTag (single still image, no frames).

Zero-shot-CLIP-for-violence literature (CLIP-embedding-based, directly relevant to our approach) —
cited for method, metrics behind paywalls so treated as "no verifiable published number here":
- "Efficient Violence Detection Using the CLIP Model" (Springer 2025):
  <https://link.springer.com/chapter/10.1007/978-3-032-07915-2_29>
- "Zero-Shot Harmful Image Recognition Based On Innovative Dataset Construction and CLIP Embedding"
  (ACM 2023, covers violence among pornography/gambling/drugs):
  <https://dl.acm.org/doi/10.1145/3660043.3660102>

---

## Red flags / honest caveats

1. **Gore has no permissive, published-metric model.** The only strong gore labeler (ShieldGemma 2)
   is Gemma-licensed + 4B → unusable. Gore labels must be human-curated.
2. **jaranohaal's 98.8% is in-domain video-frame accuracy** — not an out-of-domain estimate; expect
   a large drop on real site images, and it covers fights only, not gore/aftermath.
3. **LAION violence npy = zero documentation, zero metrics** — cannot be trusted without a local
   validation pass; embedding-space compatibility with our ViT-B/32 export is unverified.
4. **Ali7880** metric has no dataset/split and undisclosed training data + 6 dl/mo → not credible.
5. **Field benchmarks are request-only (UnsafeBench) or video (RWF-2000/Hockey/UCF-Crime)** → we
   cannot reproduce standard numbers offline; our track's real accuracy must come from a self-built,
   hand-labeled holdout.
6. NSFW moderation models are abundant and permissive (Falconsai/Marqo/AdamCodd/Freepik/prithivMLmods)
   but **none carry a violence or gore class** — do not mistake them for violence coverage.
