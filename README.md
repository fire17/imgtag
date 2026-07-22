# imgtag

**Blazing-fast, CPU-only semantic image search & content moderation for local photo datasets.**
No GPU. No cloud. Nothing leaves your machine.

Search 10,000 photos by *meaning* in milliseconds — `"vehicle"` finds cars, motorcycles,
trucks; `"a bowl of broccoli"` finds exactly that — while a scalable track system scores
every image for nudity, weapons, drugs, safety, sports, violence, and people-count, each
with an honest confidence and a policy tier.

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green)
![CPU only](https://img.shields.io/badge/GPU-not%20required-orange)

---

## Why imgtag

- **CPU-first, co-tenant-polite.** Designed for a shared 8GB Linux box that's busy doing
  other things: polite scheduling by default (`nice`, bounded workers, memory watermarks),
  full-speed opt-in. Apple Silicon dev boxes are first-class too.
- **Honest by construction.** Every probability is labeled `fitted` or `unfitted` — the
  engine never dresses "resembles the prompt set" as a verified finding. Zero results
  above a *fitted* threshold is an honest `no_match`, never an error.
- **Search while indexing.** Kick off a 10k-image index job, search immediately — partial
  results with coverage reporting, live img/s + ETA.
- **Scales in tracks, not cost.** Adding moderation track #101 changes indexing time by
  ~0%: tracks score the embeddings the index already computed (one matvec), raw scores are
  stored per image, and tiers derive at read time — so changing a policy threshold is free
  and retroactive.
- **Agent-native.** Every verb answers pure JSON (`--json`), and a ready-made agent skill
  (`skill/`) drives the whole system headlessly.

## Quickstart

```bash
git clone https://github.com/fire17/imgtag && cd imgtag
uv sync                                  # or: pip install -e .

imgtag index ~/Pictures --dataset my-photos     # returns a job id immediately
imgtag info --job <id> --json                   # live progress: img/s, ETA, failures
imgtag search "sunset over the ocean" --json    # milliseconds, with provenance
```

First run downloads the default model (PE-Core-S16-384, Apache-2.0, ~180MB total) and
autotunes threads/batch for your machine (`imgtag doctor`).

### The web app

```bash
imgtag daemon start        # unix socket + http://127.0.0.1:8899
```

Gallery per dataset · global semantic search with ALL→SOME→ANY multi-tag ranking ·
live job progress · moderation dashboard with per-tier counts · click any image for its
**ranked per-track confidence panel**.

## Search that understands meaning

```bash
imgtag search "vehicle"            # cars + motorcycles + trucks (hypernym expansion)
imgtag search "beach sunset dog"   # ALL matches first, then 2/3, then any — ranked tiers
imgtag search "a red sports car"   # free-text dense path, calibrated when fitted
```

Every hit carries `dataset`, `path`, `image_id` (content-addressed), a `why` explanation
(which tag / which path matched), and its calibration status. Identical images indexed
under multiple datasets collapse into one result with `also_in` provenance.

Measured on a dev box (M3, 5,000-image dataset): search p50 **6.8ms** / p95 7.4ms warm.

## Content moderation tracks

Seven built-in tracks, each producing a confidence for **every** image, with two-tier
policy semantics (`violation` vs `review` — look-alikes route to humans) plus an `alert`
tier for safety and a `match` tier for content classification (never counted as
moderation):

| track | instrument | status (measured) |
|---|---|---|
| weapons | trained head over OI classes | TP median 0.929 vs FP-band 0.008; AP 0.938; 10-subcategory taxonomy |
| drugs | calibrated prompt ensemble | AUROC 0.998; FP 0.44%; vape/tobacco → review per policy |
| nudity | dedicated pixel head + margin scorer | violation benchmark-cited; swimwear-review separates; mannequin control 0-flag |
| sports | fitted prompt ensemble (content) | precision 0.80 / recall 0.95, independently verified |
| violence | prompt ensemble + context arbiter | contact sports: 0% false-violation (n=262) |
| safety | person-down + danger context | review tier shipped; alert tier deliberately withheld pending better ground truth |
| people | face detector + embedding cascade | back-view person recall 0.841 where faces-alone = 0.0 |

Batch summaries at index time (`"Found 10 images with drugs, 7 with weapons…"`), per-image
stored flags, generic metadata (`--meta account_id=…`), and a user-supplied probe-dataset
workflow to measure any track's thresholds on **your** data.

## Architecture (the short version)

ONNX Runtime CPU inference · f32 memory-mapped shards with crash-safe durable writes ·
content-addressed image ids · brute-force vector scan (no ANN needed at this scale) ·
stdlib HTTP daemon over a unix socket (TCP opt-in) · vanilla-JS app · per-track f32 score
sidecars with derive-at-read tiers. Runtime dependencies are deliberately minimal.

Design records live in-repo: `ORACLE.md` (architecture decisions + playbooks),
`BUDGETS.md` (every performance claim as a numbered, tested budget), `TRACKS.md` (the
scaling laws), `VISION.md` (the founding brief, verbatim).

## The agent skill

`skill/` packages the four verbs (index / info / manage / search) for AI agents with
strict honesty laws (never fabricate results, always report coverage, respect no_match).
Install: `skill/install.sh` → drops into `~/.claude/skills/imgtag`.

## Status & roadmap

v0.1.0 — engine, daemon, app, skill, 7 tracks, calibration, benchmarks: working and
measured on dev hardware; Linux-server numbers lock on target-hardware benches next.
Roadmap lives in `DARWIN.md`: region/tile embeddings (multi-object precision), distilled
track heads, OpenVINO/int8 lanes (currently refused: every tested int8 vision tower
failed the fidelity gate — that refusal is a feature).

## License

Apache-2.0. Default model weights (Meta PE-Core) are Apache-2.0; optional models keep
their own licenses and are never bundled.
