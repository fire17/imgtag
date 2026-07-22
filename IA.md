# IA.md — information architecture (ladder-aware)

> Entities, view tiers, default views, KPIs + data sources — declared before views are
> built (grand-start Phase 3). CLI-first (I2): every view is a lens over the same core
> API; the app, the CLI, and the agent skill are three doors into ONE engine.

## Entities

| Entity | Identity | Source of truth |
|---|---|---|
| **Dataset** | `slug = user_arg or slugify(basename)`; on collision with a DIFFERENT `root_path`, append `-<blake2b6(abspath)>` (never silently merge two folders both called `photos`); the manifest records `root_path` and every result carries slug + root | `~/.imgtag/datasets/<slug>/manifest.json` |
| **Image** | stable id = **xxhash64 of file bytes** (ADR-7 declares the dep); bytes are read ONCE and both hashed and decoded from that buffer. Content-addressed ⇒ duplicates collapse to one row and a moved file keeps its id; all paths kept in the ids record | dataset manifest + index shards |
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

## Core API — frozen before Wave B (the one contract all four doors build against)

```python
open_snapshot(slug) -> Snapshot   # EAGER mmap of every shard in the manifest; .count, .model_sha
search(q: str, *, dataset=None, k=20, min_p=None) -> list[Hit]
                                  # Hit: id, path, dataset, score, p, exists, why(tags[])
index(paths, *, dataset, workers=None, full_speed=False, on_progress=None) -> JobId
job_status(job_id) -> {state, done, in_flight, total, img_s, eta_s, failures[]}
status() -> {daemon, models, datasets[], rss}
```

`done` is the DURABLE manifest count; `in_flight` is separate and rendered ghosted — the
coverage banner and the progress bar read the same number by construction (ORACLE ADR-6).
Daemon wire protocol = these five as JSON-lines over the UNIX socket (ADR-13), plus a
`subscribe_progress` stream emitting `job_status` at ≥1Hz (B10). Anything not in this list is
not cross-door API.

## KPIs per view (fed by real data only — no placeholders, ever)
- Fleet: images indexed total, img/s live, datasets count, index bytes.
- Search: latency ms (shown honestly in UI footer), hits count, coverage % during jobs.
- Jobs: img/s rolling-10s, ETA ±band, durable done + in-flight (never merged), failures with
  reasons (the skip ledger — a silent skip is a bug, B21).
- Bench: per-candidate index img/s, search p50/p95, precision@10, hypernym recall, FP rate.

> 2026-07-22: created. UI register: product (design serves the tool) — impeccable init to
> run at UI-build phase. Ladder here is guidance (full /ladder-abstraction not invoked).
> 2026-07-22 (rev round 1, editor pass): dataset-slug collision rule, xxhash64 id invariant +
> single-read, frozen five-call core API + daemon wire protocol, durable-vs-in-flight progress.
