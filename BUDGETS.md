# BUDGETS.md — vision adjectives → enforced numbers

> Law (grand-start Phase 2): every quality adjective in VISION.md becomes a numeric
> budget measured on the real target. A regression is a build failure, not a ticket.
> Status column: `provisional` until the candidate bench lands real numbers, then
> `locked`. Budgets may only tighten after locking, never loosen silently.
>
> **PRIMARY DEPLOY TARGET (user constraint 2026-07-22, VISION-ADDENDA.md): shared Linux
> x86 server, no GPU, 8GB RAM, "not powerful", other workloads running — must never be
> slowed by our indexing OR inference.** Assume AVX2, do NOT assume AVX512/VNNI.
> Bench machine: Apple M3 Max (proxy only — all its numbers labeled PROXY; the throttled
> profile is 4 threads + memory-capped, and even that is an optimistic proxy for old x86).
> Real-server numbers land the day the bench runs on the actual server (one command).
> Budgets marked 🐧 = primary-target budget (proxy-measured until then).

| # | Vision phrase (verbatim anchor) | Metric | Threshold | Test command | Status |
|---|---|---|---|---|---|
| B1 | "blazing fast … processing and indexing" | sustained index throughput, CPU-only | ≥60 img/s (M3 Max; stretch ≥120 = beat rclip-CoreML on pure CPU) · ⌂ ≥10 img/s | `uv run imgtag bench index --n 100` | provisional |
| B2 | "time to process 10,000 images on cpu (tests scales to 100…)" | wall time, 100-img test → 10k projection | ≤3min per 10k (pass ≤6min) | `uv run imgtag bench index --n 100` | provisional |
| B3 | "especially during inference/search … instantly" | search e2e latency warm (text-encode+scan+rank @10k) | p50 ≤50ms · p95 ≤120ms | `uv run imgtag bench search` | provisional |
| B4 | "google photos in app search … instantly found" | app keystroke→results-painted | p95 ≤150ms | `uv run imgtag bench app-search` | provisional |
| B5 | "sematically flexible … 'vehicle' … all motocycles and other vehicles" | hypernym recall: COCO supercategory queries | "vehicle" recall@100 ≥0.80 | `uv run imgtag bench quality` | provisional |
| B6 | "when i search for 'car' - all of the images with one or more cars" | category precision/recall on COCO ground truth | precision@10 ≥0.90 (mean over categories) | `uv run imgtag bench quality` | provisional |
| B7 | "lack of false positives" / "minimization of any false positives" | absent-category test: queries for objects NOT in corpus | ≤2% of returned results above score threshold | `uv run imgtag bench quality --negatives` | provisional |
| B8 🐧 | "not taking too many resources" + 8GB-shared-server law | peak RSS indexing · daemon idle RSS · total under load · model+index disk (10k) | ≤1.0GB · ≤350MB · ≤1.5GB (leaves ≥6.5GB to co-tenants) · ≤500MB | `uv run imgtag bench resources` | provisional |
| B9 | "super state of the art small but powerfull" | model on-disk size (quantized) | ≤150MB primary model | inspect artifact | provisional |
| B10 | "see live progress, how many images a sec its indexing, projected etas" | progress freshness · ETA error after 10% done | ≤1s stale max · ETA within ±20% | `uv run imgtag bench progress` | provisional |
| B11 | "instantly search … while the processing is still ongoing" | index-visible latency (image indexed → searchable) · reader/writer blocking | ≤2s · zero blocking | `uv run imgtag bench concurrent` | provisional |
| B12 | "no data or performance or compute or leaks of any kind" | soak: RSS drift over full-dataset run · fd count drift | ≤5% RSS drift · 0 fd drift | `uv run imgtag bench soak` | provisional |
| B13 | "very very lightweight poc app" | server cold start → first searchable response (incl. model load) | ≤2s | `uv run imgtag bench coldstart` | provisional |
| B14 | UI quality bar (impeccable) | interaction frame budget · gallery scroll | ≤16ms/frame, no jank at 10k thumbnails (virtualized) | manual + devtools trace | provisional |
| B15 🐧 | "cant slow down the server while we are doing both processing and infrence work" | co-tenant protection: reference co-workload (CPU+IO probe) keeps ≥85% of its solo throughput while we index at defaults + search concurrently | nice ≥10 + ionice + workers ≤ cores/2 by default; `--full-speed` opt-in | `uv run imgtag bench politeness` | provisional |
| B16 | anti-silent-quality-loss (clip.cpp lesson) | fast-path preprocessing parity vs reference pipeline | mean cosine ≥0.99 on quick500 AND quality deltas within noise | `uv run imgtag bench parity` | provisional |
| B17 | "state of the art" quality, auditable | COCO-caption t2i retrieval vs control | default model ≥ +5 R@10 points over OpenCLIP ViT-B/32-openai control | `uv run imgtag bench quality --retrieval` | provisional |

## Leak classes enumerated at design time (E5)
storage (orphan thumbnails/tmp), memory (embedding buffers, decoded images), fd (image files, sockets),
CPU (spin-polling — forbidden; event-driven progress only), compute (re-embedding unchanged images — mtime/hash-gated),
disk-bloat (index compaction). Each gets a structural prevention + a monitor in `bench soak`/`bench resources`.

> 2026-07-22: created (provisional) — to be locked after candidate bench (research → /unknowns → /wargame → selection → bench).
> 2026-07-22 (post-research re-derivation): B1 raised 30→60 img/s + stretch (rclip evidence), edge floor 5→10 (photofield-ai 2014-i7 does 20), B2 target ≤3min, added B15 politeness / B16 parity / B17 public-auditable quality. Still provisional until the candidate bench locks them.
> 2026-07-22 10:47: PRIMARY TARGET re-anchored per user constraint (VISION-ADDENDA.md): shared Linux x86 8GB no-GPU server; B8 tightened (≤1.0GB/≤350MB/≤1.5GB-total), B15 rewritten as co-tenant-protection probe (≥85%), 🐧 markers added, proxy-labeling protocol for all M3 Max numbers.
