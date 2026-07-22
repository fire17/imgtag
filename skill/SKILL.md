---
name: imgtag
description: Semantic image search and indexing over local photo datasets — CPU-only, blazing fast, agent-callable. Use when the user says "search my images/photos", "find photos of X", "semantic image search", "index this folder of images", "tag/process this dataset of photos", "imgtag", "what datasets are indexed", "how many images are indexed", "delete/rename an image dataset", or asks for photos by content ("cars", "vehicles", "food on a table") rather than by filename. Handles the four verbs — index (tag/process a folder), info (datasets, jobs, live progress/ETA), manage (list/delete/rename/verify), search (semantic query with dataset + path + image_id provenance). Every verb answers headless JSON, so agents can use it directly.
argument-hint: "<index|info|manage|search> [args]"
allowed-tools: Bash
---

# imgtag — semantic image search for agents

Local, CPU-only semantic search over image datasets. One embedding index per dataset under
`~/.imgtag/datasets/<slug>/`; a long-lived daemon (`~/.imgtag/daemon.sock`) keeps models warm
so a search is milliseconds, not a cold start. Nothing leaves the machine (the first-run model
download is the only egress, and it is announced).

**Engine command** — prefer the installed console script; fall back to the repo:

```bash
imgtag <verb> --json ...                              # if on PATH
uv run --project ~/Creations/ImgTag imgtag <verb> --json ...   # always works
```

## ⛔ Never fabricate a result

Every answer you give about a dataset, a job, or a search MUST come from a command you actually
ran **this turn**, and you must quote its real output. No remembered counts, no invented paths,
no guessed latencies.

## The four verbs

### 1. `index` — tag/process a folder (non-blocking)

```bash
imgtag index /path/to/photos --dataset my-photos --json
```

Returns immediately (≤500ms) with a job id; embedding continues in the background.

```json
{"job_id":"7f3a91c2","dataset":"my-photos","path":"/path/to/photos",
 "status":"running","queued":10432,"tookMs":214}
```

Flags: `--dataset <slug>` (required) · `--full-speed` (drop the polite nice/worker caps; default
is polite so co-tenants are not starved) · `--wait` (block until done — for scripts only, never
when an agent is reporting progress).

### 2. `info` — datasets, jobs, live progress

```bash
imgtag info --json                       # fleet: every dataset + daemon state
imgtag info --dataset my-photos --json   # one dataset's manifest summary
imgtag info --job 7f3a91c2 --json        # live progress of an index job
```

```json
{"datasets":[{"dataset":"my-photos","count":10432,"model_id":"pecore-s16-384",
  "model_sha":"…","dim":512,"bytes":21364736,"updated":"2026-07-22T14:02:11Z"}],
 "daemon":{"running":true,"pid":40127,"version":"0.1.0","uptimeSec":812,
  "socket":"~/.imgtag/daemon.sock"},"tookMs":18}
```

Job shape (`--job`) — `done` is the **durable manifest count**, `inflight` is separate:

```json
{"job_id":"7f3a91c2","dataset":"my-photos","state":"running",
 "done":4210,"inflight":16,"failed":3,"total":10432,
 "imgsPerSec":62.4,"etaSec":99.7,"started":"…","updated":"…",
 "errors":[{"path":"/path/broken.jpg","reason":"truncated"}],"tookMs":9}
```

`state` ∈ `queued|running|done|failed|aborted`. Poll this — do not busy-loop; every ~2s is plenty
(progress events are ≤1s fresh). `failed` files are recorded, never fatal: a job with failures
still exits 0.

### 3. `manage` — dataset lifecycle

```bash
imgtag manage list --json
imgtag manage delete my-photos --yes --json     # --yes required; there are no prompts
imgtag manage rename my-photos holiday-2026 --json
imgtag manage verify my-photos --json           # manifest ↔ shard integrity
```

```json
{"dataset":"my-photos","deleted":true,"freedBytes":21364736,"tookMs":41}
```

Delete removes the dataset directory entirely (0 orphan bytes). It is not undoable — **ask the
user before deleting anything you were not explicitly told to delete.**

### 4. `search` — semantic query

```bash
imgtag search "cars parked at night" --json
imgtag search "vehicle" --dataset my-photos -k 50 --json
```

```json
{"query":"vehicle","tookMs":37,"coverage":{"indexed":4210,"total":10432},
 "hits":[{"image_id":"9c1f…","path":"/path/img_004.jpg","dataset":"my-photos",
   "score":0.2914,"p":0.83,"why":{"path":"tag","tag":"car"}}],
 "no_match":false}
```

- Semantically flexible by design: `vehicle` also returns cars, motorcycles, trucks.
- `coverage` tells you the index is still filling — searching during an index job is supported
  and returns what is durable so far. **Always report coverage when `indexed < total`.**
- `why` explains the hit (`"path": "tag"` with the matching tag, or `"path": "text"` for a pure
  dense-embedding hit). `image_id`, `path`, `dataset` are **never null**.
- `no_match: true` with `hits: []` is a **real, correct answer** — the calibrated threshold said
  nothing in the dataset matches. Exit code is still 0. Report it as "no matching images",
  never as an error or a failure.

### Auxiliary

```bash
imgtag doctor --json     # machine profile + autotuned threads/batch/precision (first run)
imgtag status --json     # daemon pid/version/uptime/loaded models/RSS
```

## Exit codes

| Code | Meaning | What to do |
|---|---|---|
| 0 | OK (includes `no_match: true` and jobs with `failed > 0`) | Report the JSON |
| 2 | Usage error (bad flags/args) | Fix the command |
| 3 | Dataset locked — another index job holds the writer lock | Report the holding pid/job from the error; wait or pick another dataset |
| 4 | Unknown dataset | `imgtag info --json` to list real slugs |
| 5 | Model / manifest mismatch — index built with a different model | Re-index, or search with the matching model; never mix |
| 6 | Corrupt index | `imgtag manage verify <dataset>`; re-index if unrecoverable |
| 7 | Model unavailable offline (not cached, no network) | Report it; the user must allow the one-time model download |

Errors print human text to **stderr**; `--json` keeps stdout pure JSON (ANSI-free), so
`imgtag ... --json | jq` is always safe.

## Operating laws for agents

1. **Run it, then report it.** Quote real values — including `tookMs` — from the actual run.
2. **Index is asynchronous.** Never claim a folder is "indexed" off the `index` call alone;
   that only proves a job started. Confirm with `imgtag info --job <id>` → `state: done`.
3. **Respect `no_match`.** It is the anti-false-positive mechanism working, not a bug.
4. **Report coverage during indexing** so the user knows results are partial.
5. **Provenance always.** Give the user `dataset`, `path`, and `image_id` for every hit — that
   is what the JSON is for.
6. **Ask before destructive `manage` operations** (delete, rename).
7. **Don't hand-roll around it.** No shelling into `~/.imgtag` internals, no reading manifests
   directly — the CLI is the contract.
