# IA.md — information architecture (ladder-aware)

> Entities, view tiers, default views, KPIs + data sources — declared before views are
> built (grand-start Phase 3). CLI-first (I2): every view is a lens over the same core
> API; the app, the CLI, and the agent skill are three doors into ONE engine.

## Entities

| Entity | Identity | Source of truth |
|---|---|---|
| **Dataset** | slug (e.g. `coco-val2017`) | `~/.imgtag/datasets/<slug>/manifest.json` |
| **Image** | stable id = xxhash64 of file bytes; carries path + dataset slug | dataset manifest + index shards |
| **IndexJob** | job id | on-disk job status file (A3 lifecycle: queued→running→done/failed, atomic writes) |
| **Query** | text (+ optional dataset filter) | ephemeral; recent queries ring buffer (observability) |
| **Candidate** (bench) | method slug (model×runtime×quant) | `bench/candidates/<slug>.json` (frozen results, provenance C5) |

## View tiers (zoom ladder: fleet → dataset → image)

1. **Home / fleet view** — all datasets as cards: image count, indexed %, live job chip
   (img/s + ETA when running), index size, last-updated. KPI: total images searchable.
   One global search box on top — search is never more than one keystroke away.
2. **Search results** — grid of hits; EVERY hit shows: thumbnail, **dataset badge**,
   **path + id** (vision-mandated), similarity score. Filter chips per dataset. Partial-
   coverage banner when a job is mid-flight ("searching 3,214 of 5,000 indexed so far").
3. **Dataset view** — virtualized thumbnail gallery (10k-safe), indexing progress strip,
   per-dataset search. Drill: click image → detail.
4. **Image detail** — full image, id, path, dataset, dims, its top matching tags/queries
   (reverse lookup: "what would find this image").
5. **Jobs & health** (observability, vision-mandated) — live jobs with img/s + ETA + eta
   error band, recent queries + latencies, RSS/index sizes. No leaks visible = leak found.
6. **Bench / candidates** (under-the-hood pride) — candidate compare table: speed ×
   quality × false-positive rate; the empirical reason the winning method won.

## Data flow (one core, three doors)

```
images → decode/resize → encoder (CPU, quantized) → embeddings → shard store (append-only, mmap)
                                                              ↘ manifest (atomic rename)
query text → text encoder (cached) → scored scan → ranked hits (dataset, path, id, score)
doors: CLI (imgtag …) · HTTP app (localhost) · agent skill (global) — all call the same core lib
```

## KPIs per view (fed by real data only — no placeholders, ever)
- Fleet: images indexed total, img/s live, datasets count, index bytes.
- Search: latency ms (shown honestly in UI footer), hits count, coverage % during jobs.
- Jobs: img/s rolling-10s, ETA ±band, completed/total, failures (with reasons).
- Bench: per-candidate index img/s, search p50/p95, precision@10, hypernym recall, FP rate.

> 2026-07-22: created. UI register: product (design serves the tool) — impeccable init to
> run at UI-build phase. Ladder here is guidance (full /ladder-abstraction not invoked).
