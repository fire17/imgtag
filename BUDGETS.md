# BUDGETS.md — vision adjectives → enforced numbers

> Law (grand-start Phase 2): every quality adjective in VISION.md becomes a numeric
> budget measured on the real target. A regression is a build failure, not a ticket.
> Status column: `provisional` until the candidate bench lands real numbers, then
> `locked`. Budgets may only tighten after locking, never loosen silently.
>
> Dev target: Apple M3 Max, 16 cores, 64GB (CPU-only — no GPU/ANE paths).
> Edge floor: assume 4-core 2015-era x86, 8GB — projected budgets marked ⌂.

| # | Vision phrase (verbatim anchor) | Metric | Threshold | Test command | Status |
|---|---|---|---|---|---|
| B1 | "blazing fast … processing and indexing" | sustained index throughput, CPU-only | ≥30 img/s (M3 Max) · ⌂ ≥5 img/s | `uv run imgtag bench index --n 100` | provisional |
| B2 | "time to process 10,000 images on cpu (tests scales to 100…)" | wall time, 100-img test → 10k projection | ≤4s per 100 → ≤6min per 10k | `uv run imgtag bench index --n 100` | provisional |
| B3 | "especially during inference/search … instantly" | search e2e latency warm (text-encode+scan+rank @10k) | p50 ≤50ms · p95 ≤120ms | `uv run imgtag bench search` | provisional |
| B4 | "google photos in app search … instantly found" | app keystroke→results-painted | p95 ≤150ms | `uv run imgtag bench app-search` | provisional |
| B5 | "sematically flexible … 'vehicle' … all motocycles and other vehicles" | hypernym recall: COCO supercategory queries | "vehicle" recall@100 ≥0.80 | `uv run imgtag bench quality` | provisional |
| B6 | "when i search for 'car' - all of the images with one or more cars" | category precision/recall on COCO ground truth | precision@10 ≥0.90 (mean over categories) | `uv run imgtag bench quality` | provisional |
| B7 | "lack of false positives" / "minimization of any false positives" | absent-category test: queries for objects NOT in corpus | ≤2% of returned results above score threshold | `uv run imgtag bench quality --negatives` | provisional |
| B8 | "not taking too many resources … old computers or edge devices" | peak RSS indexing · server idle RSS · model+index disk (10k) | ≤1.5GB · ≤400MB · ≤500MB | `uv run imgtag bench resources` | provisional |
| B9 | "super state of the art small but powerfull" | model on-disk size (quantized) | ≤150MB primary model | inspect artifact | provisional |
| B10 | "see live progress, how many images a sec its indexing, projected etas" | progress freshness · ETA error after 10% done | ≤1s stale max · ETA within ±20% | `uv run imgtag bench progress` | provisional |
| B11 | "instantly search … while the processing is still ongoing" | index-visible latency (image indexed → searchable) · reader/writer blocking | ≤2s · zero blocking | `uv run imgtag bench concurrent` | provisional |
| B12 | "no data or performance or compute or leaks of any kind" | soak: RSS drift over full-dataset run · fd count drift | ≤5% RSS drift · 0 fd drift | `uv run imgtag bench soak` | provisional |
| B13 | "very very lightweight poc app" | server cold start → first searchable response (incl. model load) | ≤2s | `uv run imgtag bench coldstart` | provisional |
| B14 | UI quality bar (impeccable) | interaction frame budget · gallery scroll | ≤16ms/frame, no jank at 10k thumbnails (virtualized) | manual + devtools trace | provisional |

## Leak classes enumerated at design time (E5)
storage (orphan thumbnails/tmp), memory (embedding buffers, decoded images), fd (image files, sockets),
CPU (spin-polling — forbidden; event-driven progress only), compute (re-embedding unchanged images — mtime/hash-gated),
disk-bloat (index compaction). Each gets a structural prevention + a monitor in `bench soak`/`bench resources`.

> 2026-07-22: created (provisional) — to be locked after candidate bench (research → /unknowns → /wargame → selection → bench).
