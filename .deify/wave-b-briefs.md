# Wave B — builder briefs & interface contracts (the seams)

> Conductor-authored 2026-07-22 post-review-round-1. Every builder: (1) activate Skill
> `ponytail` (full) as an EARLY action (b-app also activates `impeccable`); (2) read
> ORACLE.md WHOLE (ADRs+playbooks+escalation are law), BUDGETS.md, this file; (3) consult
> research/spike-*.md + research/runtime.md for measured facts; (4) MODEL line first in
> every report; (5) escalation contract ORACLE §7 binds you — divergence → STOP+log+report.
> File ownership is EXCLUSIVE (F2): never edit another lane's files; integration happens
> at the contracts below. All code in the uv project (pyproject committed); runtime deps
> ONLY per ADR-7 + xxhash; dev-group deps allowed for bench/tests.

## Package skeleton (committed by conductor — build INTO it)

```
src/imgtag/
  core/store.py      b-engine   ADR-6 exactly: Manifest/Snapshot/Writer, flock, 7-step flush, recovery
  core/models.py     b-engine   ModelBackend registry; per-model preprocess from bundled config; L2-norm ALWAYS
  core/indexer.py    b-engine   decode workers → bounded uint8 queue → session; progress events; ADR-11 policy
  core/doctor.py     b-engine   first-run autotune (precision×threads×geometry×batch), machine profile JSON
  core/progress.py   b-engine   job status files (atomic), A3 lifecycle, img/s + ETA calc
  core/search.py     b-daemon   scan (f32 mmap @ q), calibration consumption, probability fusion, hypernym expansion
  core/tags.py       b-bench    tag table build + Platt fitting on CAL-SET + dataset-layer stats spec
  cli.py             b-engine   verbs: index/info/manage/search/doctor/bench dispatch; --json law (B20)
  daemon.py          b-daemon   stdlib ThreadingHTTPServer over UNIX socket (~/.imgtag/daemon.sock); --tcp 127.0.0.1 opt-in; SSE /api/events
  app/               b-app      static vanilla JS+CSS, virtualized grid, served by daemon
  bench/             b-bench    every `imgtag bench *` in BUDGETS; candidate matrix; rclip head-to-head
skill/               b-skill    source of ~/.claude/skills/imgtag (installed by its installer script)
```

## Contracts (breaking one = escalate, never improvise)

**store.py** (b-engine owns; b-daemon/b-bench consume):
- `open_snapshot(dataset: str) -> Snapshot` — Snapshot: `.emb  (np.memmap f32 [N,D] L2-normalized)`, `.ids (list[IdRec])`, `.manifest (Manifest)`; IdRec = `{image_id: str(hex16), path: str, dataset: str, w: int, h: int}`. Reads manifest ONCE; never stats shards; N from manifest counts.
- `Writer(dataset, model: ModelBackend)` ctx-mgr — flock or exit-3; `.append(embs: f32[n,D], recs: list[IdRec])`; flush per ADR-6 protocol at min(2s, 512 rows); `.job_id`.
- Manifest JSON fields: `{version, dataset, model_id, model_sha, dim, count, shards:[{name,rows,emb_bytes,ids_bytes}], tag_stats?, calib_sha?, calib_model_sha?, created, updated}`.

**models.py** (b-engine owns):
- `load_backend(name: str, profile: MachineProfile) -> ModelBackend` — ModelBackend: `.model_id .model_sha .dim`, `.preprocess(PIL.Image) -> np.uint8 [H,W,3]` (per-model config: size/resample/squash — SigLIP2 BILINEAR, PE-Core squash+0.5/0.5; normalization fused in graph where possible), `.embed_images(uint8 batch) -> f32 [n,D] L2-NORMALIZED`, `.embed_texts(list[str]) -> f32 [n,D] L2-NORMALIZED` (lazy-loads text tower; releasable via `.release_text()`). Backends: `pecore-s16-384`, `pecore-b16-224`, `pecore-t16-384`, `siglip2-base-224`, `siglip-base-224`, `openclip-vitb32` (+ `mobileclip2-s0/s2` opt-in plugin, never default). Model files under `~/.imgtag/models/<model_id>/` with sha256 manifest; downloader = httpx+certifi, ranged resume, hard sha check.
- Tokenizers: bundled COMPACT binary (msgpack/npz of merges+vocab) built offline — never parse 34MB tokenizer.json at runtime (measured 0.64s).

**Search API (daemon HTTP + CLI --json share this schema):**
`GET /api/search?q=&dataset=&k=50` → `{query, tookMs, coverage: {indexed, total}, hits: [{image_id, path, dataset, score, p, why: {path: "tag|text", tag?}}], no_match: bool}` — provenance fields NEVER null (B18). `GET /api/datasets` → fleet view data. `GET /api/jobs` + SSE `/api/events` (progress ≤1s). `GET /api/thumb/<dataset>/<image_id>?s=256` → JPEG (draft-decoded, LRU disk cache ≤200MB). `POST /api/index {path, dataset}` → `{job_id}` ≤500ms.
**Skill/CLI verbs (B20):** `imgtag index|info|manage|search|doctor --json`; exit codes: 0 ok · 3 dataset-locked · 4 unknown-dataset · 5 model/manifest-mismatch · 6 corrupt-index · 7 model-unavailable-offline; zero-results-above-τ is exit 0 with `no_match: true`.

**Measured facts binding the build** (from spikes — cite, don't re-derive): PE-Core embeds are UNNORMALIZED (norms 5–21) — normalize in embed fn; text tower int8-only + lazy (fp32 text = 850MB RSS bomb); batch-1/2 streaming (batch-8 ≈ 2× RSS, ~0 gain); intra_op default from doctor (never cpu_count; knee at 4–6 on M3); vision int8 must pass B24 via the weight-only MatMul recipe (naive full quant = cos 0.94 FAILS); B16-224 vs S16-384: resolution dominates params — bench decides default.

## Launch order
1. b-engine (critical path; owns pyproject deps if additions needed → escalate first)
2. b-bench phase 1 (candidate matrix standalone on spike/runtime scripts) — parallel with 1
3. l-logistics (haiku): rclip install (`uv tool install rclip`), model zoo fetch per models.py list, checksums → reports blockers, never improvises
4. b-daemon + b-app + b-skill after b-engine's store/models land (contracts above let them start on their own files immediately — integration at the end)
