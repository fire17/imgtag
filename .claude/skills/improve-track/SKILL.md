---
name: improve-track
description: Brief a track subagent to improve its own IMGTAG track (autoresearch + refit
  + measured deltas), and meta-improve this briefing protocol itself by observing the
  rate of improvement it produces. Use when the user says "improve the <X> track",
  "run a track improvement round", "/improve-track <track>", or a track's metrics
  need a push. Project-local (ImgTag); generalization to other tracks-having systems
  is deliberately deferred (see §Globalize later).
argument-hint: "<track-name> [rounds]"
---

# /improve-track — the track self-improvement protocol (inner loop + outer loop)

> User law (VISION-ADDENDA 13:42Z, verbatim there; TRACKS.md T3/T4): track agents improve
> their OWN tracks programmatically; agents may eyeball only selective test samples, NEVER
> perform the track's job; the runtime stays 100% agent-free. This skill is BOTH the
> briefing we hand a track agent (inner loop) AND a self-improving artifact whose fitness
> is the change in rate of improvement it produces across invocations (outer loop).

## Invoking (conductor side)

1. Read `.deify/track-improvement-ledger.json` — the track's past rounds (metric deltas,
   tokens, wall time). Read the track's `research/track-<name>.md` + its spec in
   `data/moderation.json` + its acceptance set.
2. Spawn (or resume) the track's agent — explicit non-Fable model, MODEL-line tripwire —
   with the INNER BRIEF below, filling the placeholders from the ledger + report.
3. On completion: append a ledger entry (see schema) with the measured deltas. Never
   accept a round without before/after numbers from `bench` commands.

## The INNER BRIEF (hand this to the track agent, filled in)

You own the <TRACK> track. Improve it in ≤N rounds; every round = the loop:
(a) GROUND — run the track's eval suite (`uv run imgtag bench quality --track <TRACK>` +
    its acceptance set + negative controls); record baseline numbers.
(b) AUTORESEARCH — WebSearch/WebFetch for better instruments within TRACKS.md law:
    richer prompt/negative vocabularies, better teachers for distilled heads, subcategory
    taxonomies (expand depth: e.g. drugs → cannabis / pills-in-drug-context / powders /
    paraphernalia / psychedelics / staging), fitted-threshold techniques. Cheap before
    expensive; embedding-space before dedicated models (a dedicated model must be EARNED
    with a measured zero-shot failure, TRACKS.md T2).
(c) IMPROVE — spec/prompt/negative/fit changes as VERSIONED DATA (never code in the hot
    path); refit on held-out ground truth; re-score ONE sidecar column.
(d) VERIFY — re-run (a). Gates: acceptance set green · p-distribution SPREAD (saturation
    = automatic revert, the drugs lesson) · confidence CORRECTNESS improved (calibration
    error / separation, not just rates) · no other track touched · agents-never-operate
    honored (T4: you may open ≤20 sampled images per round to diagnose, none to score).
(e) REPORT — before/after table, what changed and why, honest gaps; a round that
    regresses gets reverted and reported as such (a reverted round is a valid result).
Escalation: ORACLE §7 binds; stop-and-say-so beats grinding.

## The OUTER LOOP (this skill improving itself)

- Every invocation appends to `.deify/track-improvement-ledger.json`:
  `{ts, track, rounds, before: {...}, after: {...}, delta, tokens_est, wall_min,
    brief_version}`.
- Fitness of THIS SKILL = d(delta/round)/d(invocation) across tracks — are briefs
  producing faster improvement over time? (Inner loop improves tracks; outer loop
  improves the improving.)
- Periodically (each darwin round, or on 3+ new ledger entries) the conductor runs a
  darwin/autoresearch cycle ON THIS FILE: vary the inner brief (ordering, gate strictness,
  research depth, sample budgets), A/B across comparable invocations via the ledger,
  keep what raises the improvement rate, bump `brief_version`. Regressions revert;
  the ledger is the arena.

## Hard laws (bind every invocation)

- T4: programmatic runtime, agent-free after the round; ≤20 diagnostic image views/round.
- Spread p, never saturated; store raw scores, derive tiers (T1).
- No new runtime deps; dedicated-model FLOPs budget (B25) respected.
- Never fabricate a measurement; reverted rounds are reported, not hidden.

## Globalize later (user reminder — deliberately deferred)

The user asked to be REMINDED to consider making this generic/global once we understand
what "tracks" abstracts to in other systems (any per-item scored-classifier registry).
Generalization notes accumulate here; the reminder fires at delivery + at darwin-loop end.

> brief_version: 1 · created 2026-07-22 per VISION-ADDENDA 13:42Z
