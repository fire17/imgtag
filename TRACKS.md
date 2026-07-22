# TRACKS.md — the track system constitution

> User law (VISION-ADDENDA 13:26Z, verbatim there): every track — present or future —
> produces a confidence score for EVERY image; tracks are specialized and may auto-improve
> themselves; and **even at 100 tracks, indexing and inference times must remain
> relatively the same. The system scales in track count.** This file is where that law
> lives and how it is made structurally true. Companion: ORACLE.md ADR-15.

## The three laws

**T1 — Every track scores every image.** A track's output is a dense column: one f32
confidence per image, stored in a per-track sidecar `~/.imgtag/datasets/<slug>/tracks/
<category>.f32` aligned to the index row order (same alignment invariant as shards).
Flags are NOT what is stored — **raw scores are stored; tiers are DERIVED at read time**
from the track's spec (τ values). Consequences: policy/threshold changes are free (no
re-scoring), any past image is queryable at any future threshold, and per-track rollups
are one vectorized pass.

**T2 — The scaling invariant: track cost must not grow with the encoder's.** The
instrument hierarchy, in order of preference:
1. **Embedding-space scoring** (prompt/tag ensembles, linear probes): cost per track ≈
   one D×P matvec over the embedding the index ALREADY computed. 100 tracks ≈ ~1M
   multiply-adds per image ≈ microseconds. This is the default and the only instrument
   that is unconditionally allowed.
2. **Distilled embedding heads** (small MLP over our embeddings, trained offline to mimic
   a stronger teacher): same runtime cost class as (1). **This is the mandated fate of
   every dedicated model**: a dedicated model may serve as an offline TEACHER (labeling
   training data at leisure), but what ships in the indexing hot path is its distilled
   embedding head. (The LAION-safety-head pattern.)
3. **Dedicated per-image models** (extra forward pass per image): EXCEPTION, budgeted —
   the SUM of all dedicated-model FLOPs must stay ≤30% of the encoder's FLOPs (B25).
   Permitted only while a distillation (2) hasn't yet matched its quality, with the
   distillation logged as an open darwin item — and a track EARNS a dedicated model only
   when the shared embedding provably cannot carry its signal (measured, like nudity's
   food-above-lingerie zero-shot failure; never assumed). Current sole occupant:
   nudity's Marqo head (4.5 GFLOPs ≈ 25% of PE-Core-S16) — darwin item D11 distills it.
   **Escape hatch if dedicated heads ever multiply: one shared small backbone + N linear
   classifier heads** (one extra forward total, not N) — recorded here so nobody
   rediscovers it (track-nudity §6b).

**T3 — Per-track auto-improvement.** Each track's spec (prompts, negatives, τ, tier
bands, head weights) is versioned data, not code (`data/moderation.json` + per-model
fitted files). A track improves by: refit on new ground truth → re-score its ONE sidecar
column (never re-embedding images, never touching other tracks) → re-derive tiers.
Re-scoring a track over 10k images is one matvec pass (~ms) + one file swap. The darwin
loop may run per-track autoresearch (better prompts, better teachers, better fits) under
the bench gates; a track upgrade that regresses its acceptance set is reverted.

**T4 — Agents verify, never operate (user law 13:42Z).** The runtime is 100% programmatic:
no agent is ever in the scoring/categorization path, and the system runs identically with
zero agents present. Agents may LOOK at selectively-sampled images only to conduct tests,
diagnose failures, and calibrate — never to perform the track's job, never to bulk-verify.
Token spend rule: every working agent is either improving the system or improving its
track; eyeballing beyond selective test samples is waste and gets stopped.

## The improvement loops (T3 operationalized)

- **Inner loop**: a track agent improves its own track under `.claude/skills/improve-track`
  (the project-local briefing protocol): autoresearch → refit → acceptance gates → sidecar
  re-score → measured delta reported to the ledger.
- **Outer loop**: the conductor observes `.deify/track-improvement-ledger.json` (per
  invocation: track, metric deltas, wall/tokens spent) and darwin-improves the BRIEFING
  PROTOCOL itself — fitness = the change in rate of improvement across invocations, not
  any single track's score. The skill self-improves; the tracks inherit the better briefs.

## Adding track #101

`imgtag track add <category>` (spec entry + optional fitted head) → one scoring pass
appends one sidecar column per dataset (backfill is a background job with the usual
progress/ETA) → searchable + rollup-able immediately. Indexing time changes by ~0%;
search time changes by ~0% (tier filters read sidecars, not models).

## Verification (B25, bench-enforced)

`uv run imgtag bench tracks` measures: (a) marginal index-time per embedding-track
(must be ≤0.5% each, ≤5% at 100 simulated tracks); (b) total dedicated-model FLOPs share
(≤30% of encoder); (c) sidecar alignment integrity (row i of every track column = row i
of the shard); (d) tier-derivation determinism (same scores + same spec = same tiers);
(e) **READER PARITY (2026-07-22, the sports lesson): a track is not done until its
numbers are verified through the SERVING path** — head-vs-reader flag counts on a probe
slice must agree within fp noise. Two scorers, one contract; τ lives in the reader's
score space (margin), never a private one (the prob-space τ gated the reader to 0/1856
while the head fired 137 — invisible until a sibling lane measured the reader).

> 2026-07-22: constituted. Current tracks: nudity (dedicated head — distillation owed,
> D11) · weapons · drugs (refit in progress) · safety · sports — all others
> embedding-space. Score sidecars + tier-derivation-at-read are the active engine change.
