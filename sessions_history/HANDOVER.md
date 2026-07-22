# HANDOVER — take it from here

> Written by the founding conductor at publish time, for whoever comes next.
> Everything below is *observed working*, not aspirational. Where something is
> unfinished or uncertain, it says so — that honesty is the house style; keep it.

## What you're inheriting (30 seconds)

A CPU-only semantic image search + content-moderation engine that indexes ~10k-image
datasets, answers open-vocabulary searches in single-digit milliseconds warm, and scores
**every image on every moderation track** (nudity/weapons/drugs/safety/sports/violence/
people) with honest, calibration-labeled confidences and policy tiers. A stdlib HTTP
daemon serves a vanilla-JS app (gallery, search, moderation dashboard, per-image ranked
track panel). An agent skill drives everything headlessly in JSON. It was designed for a
shared 8GB Linux box it must never slow down — politeness is architectural, not a flag.

## Run it now

```bash
git clone https://github.com/fire17/imgtag && cd imgtag
uv sync
uv run imgtag doctor --json          # autotunes threads/batch; first-run model download
uv run imgtag index <photos-dir> --dataset mine        # returns job id instantly
uv run imgtag info --job <id> --json                   # live img/s + ETA
uv run imgtag search "sunset over the ocean" --json
uv run imgtag daemon start           # app at http://127.0.0.1:8899
```

## Verify it yourself (never trust a handover, including this one)

```bash
uv run pytest -q                      # full suite (95+ green at handover)
uv run imgtag bench search            # B3: p50 ≤50ms gate (we measured 6.8ms)
uv run imgtag bench resources         # B8-search: ≤350MB gate (we measured 240MB)
uv run imgtag bench quality           # B5/B6/B17 on an indexed dataset
uv run imgtag track recount <dataset> # stored tier counts re-derived under current τ
```

Every budget's command is in `BUDGETS.md`. If a number you measure contradicts a number
you read here, **believe your measurement and log the divergence** — that rule is in the
ORACLE and it saved us repeatedly.

## The system map (who owns what)

```
src/imgtag/core/store.py     shards, manifests, flock writer, 7-step durable flush,
                             score sidecars, tier derivation (ONE shared path)
src/imgtag/core/models.py    ONNX backends, per-model preprocess, L2-norm ALWAYS
src/imgtag/core/indexer.py   decode workers → bounded queue → session; polite mode
src/imgtag/core/search.py    tag path + free-text path, PROBABILITY-space fusion,
                             calibration consumption, hypernyms, dedupe/also_in
src/imgtag/core/tags.py      2,177-tag table, Platt fits, max-F1 τ
src/imgtag/moderation/*.py   one module per track; conductor owns __init__ dispatcher
src/imgtag/data/moderation.json  every track's spec AS DATA (prompts, tiers, τ, fits)
src/imgtag/daemon.py         stdlib ThreadingHTTPServer, unix socket + TCP opt-in
src/imgtag/app/              vanilla JS/CSS; the ranked track panel lives here
src/imgtag/bench/            every budget's measurement harness
skill/                       the agent skill (install.sh → ~/.claude/skills/imgtag)
```

## State at handover (the honest ledger)

**Solid and verified:** search (fitted calibration live, honest no_match, cross-dataset
collapse), all 7 tracks measured with TP-probe datasets, stored counts correct
(weapons 114/166 with TP-median 0.929 vs FP 0.008), the per-image tracks endpoint +
panel, install-from-repo, the published history is clean (2.4MB).

**Known-open (each has an owner-note in DARWIN.md or ORACLE §8):**
- Sidecar read hook in the daemon: blocked only on column-header metadata population;
  until then first-click-per-large-dataset pays a one-time ~7s live scan (then cached).
- Drugs gallery visibility: gate VERIFIED safe independently; wiring waits on arbitrated
  consumption (never re-band a head that has tier arbitration — vape→review is policy).
- Perf numbers are ADVISORY on the dev Mac (desktop baseline makes the honest load gate
  unreachable there); B1/B2 lock on the real Linux target. The bench harness is ready.
- D1 (region/tile embeddings) is the ceiling-breaker for multi-object AND-precision
  (measured ceiling 0.281 with ONE global embedding — no fit can beat it; don't try).

## The five rules that kept this project honest (keep them)

1. **Budgets, not adjectives.** "Fast" becomes a number with a command that proves it.
2. **Fail-open calibration.** An unfitted threshold ranks and admits it cannot judge —
   it never turns a real query into "nothing matched".
3. **Reader parity.** A scoring change is done when the SERVING path reproduces it,
   not when your own harness does. (Three bugs taught this: `TRACKS.md` clause (e).)
4. **Store scores, derive tiers.** Policy changes must be free and retroactive.
5. **User policy is never approximable.** If the user ruled vape→review, no interim
   shortcut may ever show vape→violation, even briefly.

## A final word

We left the map better than we found it: every dead end is labeled (ORACLE Dead Ends),
every surprise is field-logged, every number carries its measurement. If you're about to
do something clever, grep the ORACLE first — there's a decent chance we already tried it,
measured it, and wrote down where the cliff is. Build well.
