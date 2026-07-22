# ADAPTING — plug imgtag into your platform (social network, marketplace, CMS, anything)

> The design intent, verbatim from the founding vision: *"i want to be able to use this
> one some public sites and to enforce good behavior"*. This cookbook is how. The core
> stance: **imgtag is already operational — patch YOUR system to call it.** Adapt at the
> edges (ingestion, policy numbers, alert routing); leave the engine alone.

## The three integration surfaces (pick one, they're equivalent)

1. **HTTP API** (daemon): `POST /api/index {path, dataset}` · `GET /api/search?q=&dataset=&k=`
   · `GET /api/moderation` · `GET /api/image/<dataset>/<id>/tracks` · SSE `/api/events`
   for live progress. Unix socket by default, TCP opt-in (`--tcp`, sticky).
2. **CLI, JSON-first**: every verb with `--json` emits pure machine-readable stdout;
   exit codes are contractual (0 ok incl. honest no-match · 3 locked · 4 unknown-dataset
   · 5 model-mismatch · 6 corrupt · 7 model-unavailable). Cron/queue-friendly.
3. **The agent skill** (`skill/`): if your platform has AI agents, install it and they
   drive everything with built-in honesty laws (never fabricate, report coverage, respect
   no_match).

## Recipe: user-upload moderation for a social platform

```
upload lands → your pipeline copies/mounts the file locally
  → imgtag index <dir> --dataset uploads-$(date +%Y%m%d) --moderation \
      --meta account_id=<uid> --meta upload_ts=<iso>       # metadata rides every image
  → poll info --job (or subscribe SSE) → job summary carries the batch counts:
      "Found N images with drugs, M with weapons, K with nudity" + per-tier splits
  → your enforcement worker reads per-image flags:
      alert     → page a human NOW (highest severity; deduped by image across tracks)
      violation → auto-action per your policy (only where enforcement_ready + fitted)
      review    → human queue (this tier EXISTS for look-alikes: swimwear, toy weapons,
                  vape — do not auto-action it)
      match     → content label (e.g. sports) — routing/discovery, NEVER moderation
```

Everything stays on your infrastructure. No image, embedding, or score leaves the machine.

## Setting thresholds for YOUR platform (the probe method — do not skip)

Our shipped τ values were fitted on OUR corpora. Your content distribution differs, so
**earn your thresholds the same way we did** (one afternoon, fully scripted pattern):

1. Collect 100–300 true positives per track you care about (your own moderation queue's
   confirmed cases are ideal) → put them in a folder.
2. `imgtag index <folder> --dataset <track>probe --moderation`
3. Score-separation: TP distribution vs your normal-content FP band (see
   `scripts/eval_weapons.py` / `research/bench_scripts/nudity_probe.py` as templates —
   they read sidecar columns, no model re-runs).
4. Pick τ from the FP-band tail at YOUR acceptable FP rate (we report Wilson 95% CIs;
   copy that). Write it into `data/moderation.json` (or a per-model fitted file) — it's
   DATA; the engine re-derives all stored tiers for free (`imgtag track recount`).
5. **Gate check before trusting any threshold:** TP scores must dominate your FP band,
   and the distribution must be SPREAD, not saturated. If a track fails this on your
   data, keep it review-only (recall-first) and say so — a weak auto-flag destroys
   operator trust faster than no flag.

## Adding a platform-specific track (e.g. counterfeit goods, self-harm imagery, spam overlays)

`imgtag track add <category>` after one JSON entry in `data/moderation.json`:
prompt sets per tier + negatives. That's the whole change — the reader derives tiers
from your spec's keys, storage backfills one sidecar column over existing indexes
(a matvec pass, no re-embedding), and indexing/search latency does not move (measured
law: ≤0.5% per track). Escalate to a trained head ONLY after the embedding provably
fails on your probe set (that discipline is `TRACKS.md` T2 — it kept this system fast).

## Scaling notes for real deployments

- One dataset per shard-able unit (per-day uploads, per-community, per-bucket) — search
  spans datasets transparently; global results collapse exact duplicates with `also_in`.
- The daemon is a single polite process; N boxes = N daemons behind your router, datasets
  partitioned. Brute-force scan is intentional — at ~10k rows/dataset it beats ANN on
  latency+simplicity; revisit only past ~1M rows per box (`DARWIN.md` D7 has the design).
- 8GB RAM is a first-class target: default model + int8 text ≈ 183MB resident text-side,
  425MB peak per index worker, polite caps. Old hardware is fine; that was the point.
- HEIC uploads: install the `heic` extra (`imgtag[heic]`).

## What NOT to change (each guarded by a measured lesson)

- Don't enable int8 **vision** — it's refused by a parity gate because every tested int8
  vision tower broke retrieval. The refusal is a feature (`TRICKS` #1).
- Don't auto-action the review tier. It is by construction the look-alike tier.
- Don't re-band a track that has internal arbitration (drugs): consume its head's tiers.
- Don't publish throughput numbers from a busy box — use the bench harness's load-gated
  protocol (`imgtag bench index`), or label ADVISORY like we did.
- Don't "improve" the honest labels away. `calibration: unfitted` on a result is the
  product telling your operators the truth. It's why they'll trust it.

## If you get stuck

`ORACLE.md` §Playbooks is symptom-keyed (exact error text → exact fix). §7 is the
escalation contract. The transcript in this folder shows every problem this system hit
on day one and how it was diagnosed — `TRANSCRIPT-GUIDE.md` tells you where to look.
