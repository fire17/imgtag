# BUDGETS.md — vision adjectives → enforced numbers

> Law (grand-start Phase 2): every quality adjective in VISION.md becomes a numeric
> budget measured on the real target. A regression is a build failure, not a ticket.
> `provisional` until the candidate bench locks them; budgets only tighten after locking.
>
> **PRIMARY DEPLOY TARGET (VISION-ADDENDA.md, verbatim): shared Linux x86 server, no GPU,
> 8GB RAM, "not powerful", co-tenant workloads must never be slowed.** AVX2 baseline, no
> AVX512/VNNI assumed. Bench machine M3 Max = PROXY (all its numbers labeled so).
> 🐧 = primary-target budget. ⌂ = old-machine floor: "projected via 4-thread throttled
> proxy, NOT live-verified" until run on real hardware — always labeled.
>
> **Bench protocol (rev round 1):** every row = median of ≥3 runs; every row records
> `os.getloadavg()`; runs refuse or mark UNRELIABLE when 1-min load > cores×0.6; every
> number carries its corpus tag + resource mode (POLITE/FULL). Darwin loops gate on this.

## Corpora (no number exists without one)

| Tag | Name | Contents | Status |
|---|---|---|---|
| CORPUS-A | coco5k | 5,000 COCO val2017 (640×480 median) + exhaustive 80-class truth | ✅ on disk |
| CORPUS-B | photo10k | 10,000 Unsplash @w=3200 (≈5MP) — realistic photo sizes | fetch queued (≈18GB) |
| CORPUS-B12 | fullres300 | 300 native ≥12MP originals — the decode-bound case | fetch queued |
| CORPUS-C | mixed10k | coco5k + 5k of photo10k | derived |
| CORPUS-D | poison | ~120 hostile files: truncated/corrupt JPEG, HEIC, CMYK, 0-byte, PNG16, huge-dims, EXIF-rotated | build queued |
| CAL-SET | cocotrain2k | ~2,000 COCO train2017 imgs (per-image fetch) — HELD-OUT calibration split, never benched | fetch queued |

## Budgets

| # | Vision phrase (verbatim anchor) | Metric | Threshold | Test command | Status |
|---|---|---|---|---|---|
| B1 🐧 | "blazing fast … processing and indexing" | sustained e2e index throughput (files→searchable), CPU-only, **POLITE mode** (headline) | CORPUS-A ≥150 img/s · CORPUS-B ≥60 img/s · ⌂ ≥10 img/s · stretch (FULL, labeled): ≥180 = beat rclip's measured CoreML rate on pure CPU · **HEAD-TO-HEAD GATE: red if slower than rclip on same corpus+machine+run** | `uv run imgtag bench index --corpus A,B --headtohead rclip` | provisional |
| B2 | "time to process 10,000 images on cpu (tests scales to 100…)" | wall time, CORPUS-B 100-img sample → full-run validation at lock time | ≤3min per 10k projected AND one full CORPUS-B run within 15% of projection | `uv run imgtag bench index --corpus B` | provisional |
| B3 | "especially during inference/search … instantly" | search e2e warm, cache-MISS queries only (cache hits reported separately), @10k | p50 ≤50ms · p95 ≤120ms · scan array asserted f32 | `uv run imgtag bench search` | provisional |
| B4 | "google photos in app search … instantly found" | keystroke→painted via performance.mark pairs, Chromium/Playwright (DEV dep), CORPUS-C, warm daemon, ≥50 trials × 10 distinct queries | p50 ≤80ms · p95 ≤150ms | `uv run imgtag bench app-search` | provisional |
| B5 | "sematically flexible … 'vehicle' … all motocycles and other vehicles" | hypernym retrieval, CORPUS-A supercategory queries {vehicle, animal, food, furniture, appliance, sports} | (a) precision@100 mean ≥0.85 · (b) per-child recall@R: mean ≥0.55, MIN ≥0.35, NO child at 0.00 · (c) every child present ≥1× in top-100 · per-child table always emitted | `uv run imgtag bench quality --hypernym` | provisional |
| B6 | "when i search for 'car' - all of the images with one or more cars" | per-category precision/recall on CORPUS-A ground truth | precision@10 ≥0.90 mean AND ≥0.70 min over the 80 categories (per-category table emitted) | `uv run imgtag bench quality` | provisional |
| B7 | "lack of false positives" / "minimization of any false positives" | calibrated no-match threshold τ FROZEN in manifest, fitted on CAL-SET (held out); both sides asserted in ONE run | (a) absent-concept leakage ≤2% over ≥20 auto-derived absent queries + 5 absurdities · (b) at the SAME τ: mean recall@10 over 80 present categories ≥0.70 · (c) τ recorded in bench output + manifest; τ change without re-run of (a)+(b) = red. Passing (a) by sacrificing (b) is RED | `uv run imgtag bench quality --negatives` | provisional |
| B8 🐧 | "not taking too many resources" + 8GB-shared-server law | **process-TREE RSS** (indexing peak / daemon idle / total under load) · model+index disk @10k | ≤1.0GB · ≤350MB · ≤1.5GB · ≤500MB | `uv run imgtag bench resources` | provisional |
| B9 | "super state of the art small but powerfull" | total model artifacts loaded for SEARCH (all towers + tag table + calib) | ≤150MB | manifest inspection in `bench resources` | provisional |
| B10 | "see live progress, how many images a sec its indexing, projected etas" | progress freshness · ETA accuracy: after 20% of CORPUS-B, |actual−predicted| remaining time | ≤1s stale · ETA error ≤20% · progress row count never exceeds durable manifest count | `uv run imgtag bench progress` | provisional |
| B11 | "instantly search … while the processing is still ongoing" | index-visible latency · search p95 DURING active indexing (POLITE mode) | ≤2s visibility · search-under-load p95 ≤200ms · 0 torn reads over ≥5k concurrent searches | `uv run imgtag bench concurrent` | provisional |
| B12 🐧 | "no data or performance or compute or leaks of any kind" | soak: full CORPUS-B index + 10k searches + 3 reindex cycles; all 6 enumerated leak classes | RSS drift ≤5% · fd drift 0 · orphan bytes under ~/.imgtag = 0 after delete · tmp files 0 · no re-embed of unchanged files (compute leak) · disk returns to baseline after compaction | `uv run imgtag bench soak` | provisional |
| B13 | "very very lightweight poc app" | server cold start → first searchable response (incl. model load, no daemon) | ≤2s @10k (f32 open-cost included) | `uv run imgtag bench coldstart` | provisional |
| B14 | UI quality bar (impeccable) | scripted 10k-thumb virtualized-grid scroll 1000px/s 5s, CDP tracing (DEV dep) | frame p50 ≤16.7ms · p95 ≤20ms · 0 long tasks >50ms · DOM nodes <5,000 | `uv run imgtag bench ui` | provisional |
| B15 🐧 | "cant slow down the server while we are doing both processing and infrence work" | co-tenant protection asserted from the SAME run as B1: reference co-workload keeps ≥85% solo throughput while indexing at POLITE defaults + concurrent searches; per-process nice sampled at 1Hz | POLITE policy: workers=clamp(ncpu−2,2,8), ORT intra=2 inter=1, total threads ≤ ncpu−1, nice ≥10 in every worker · FULL only via `--full-speed` | same run as `bench index` (+`bench politeness` probe) | provisional |
| B16 | anti-silent-quality-loss, preprocessing axis | fast decode path vs model-reference pipeline (per model's OWN preprocessor_config) on quick500 | mean cos ≥0.99 AND top-1 NN agreement ≥0.90 AND quality deltas within noise | `uv run imgtag bench parity` | provisional |
| B17 | "state of the art" quality, auditable | COCO-caption t2i retrieval vs control, CORPUS-A | default model ≥ +5 R@10 pts over OpenCLIP ViT-B/32-openai control, both measured same harness same corpus | `uv run imgtag bench quality --retrieval` | provisional |
| B18 | "get exactly both from which of the datasets the resulting images are from and obviously the image path and or id" | provenance completeness over 200 queries × top-50, CORPUS-C | 100% hits carry non-null {dataset_slug, path, image_id} · 100% paths exist · 100% ids == xxhash64(bytes) · 0 cross-dataset misattribution · **zero tolerance — any failure = red** | `uv run imgtag bench provenance` | provisional |
| B20 | "a globally available skill so agents will be able to … tag … info … manage … search" | machine-API conformance, all 4 verbs headless `--json` | valid JSON stdout (ANSI-free; human text → stderr) · documented exit codes for 5 error classes · 0 interactive prompts · search ≤ B3 p95+50ms · info ≤200ms · index returns job id ≤500ms non-blocking · delete leaves 0 orphan bytes (byte-diff) | `uv run imgtag bench skill-contract` | provisional |
| B21 | "make sure it all works" / "increased reliablility" | robustness over CORPUS-D (~120 hostile files) | 0 crashes · 0 hangs (per-file timeout) · every failure skipped+logged with reason · job completes · summary counts match | `uv run imgtag bench robustness` | provisional |
| B24 | anti-silent-quality-loss, precision axis — applies to ALL quantized artifacts incl. official downloads | shipped precision vs SAME model fp32, CORPUS-A | mean cos ≥0.995 · min ≥0.97 · |Δ precision@10| ≤0.01 · ΔR@10 ≥ −1.0pt · failing precision never ships as default (flag-only, deltas printed) | `uv run imgtag bench parity --precision` | provisional |

## Leak classes enumerated at design time (E5)
storage (orphan shards/thumbnails/tmp) · memory (decode buffers, ORT arena, query cache) ·
fd (files/sockets) · CPU (spin-polling forbidden; event-driven only) · compute (re-embedding
unchanged files — mtime+hash gated) · disk-bloat (compaction). Each: structural prevention +
monitor in `bench soak`/`bench resources`.

> 2026-07-22: v1 (14 budgets, provisional) → post-research re-derivation (17) → **rev round 1
> (this rewrite): B5 was mathematically impossible (recall@100 ceiling 0.086 — replaced per
> rev-budgets C-1); corpus registry added (C-2); B1 restated polite-mode + head-to-head
> (C-3/C-6); B7 calibration contract (C-4/C-7); B18/B20/B21/B24 added (C-5/C-6/I-13/C-8);
> B4/B14 got real Playwright harnesses (C-7); noise protocol added (rev-oracle C-6).
> Full adjudication: .deify/reviews/ADJUDICATION.md. Numbering keeps review references
> stable (B19/B22/B23 intentionally unassigned).**
