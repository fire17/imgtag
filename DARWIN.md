# DARWIN.md — self-improvement rounds (grand-start Phase 7)

> Per-round log: gaps found / closed / open. Loop starts post-delivery (60-min user
> directive, /darwin-skill + autoresearch). Exit: open-gap list empty or 2 consecutive
> no-gap rounds → escalate. Bench gates every round: full budget table, any red = revert.
> Machine-load gate per ORACLE (loadavg > cores×0.6 → round invalid).

## Backlog (seeded pre-loop, from measured findings)

| # | Item | Evidence | Class |
|---|---|---|---|
| D1 | Region/tile embeddings for small-object recall | "person" ranks 800/2177 on a bus scene containing people; global CLIP vector can't represent incidental objects (b-daemon, measured, candidate fix A/B'd + deleted) | architecture |
| D2 | Fitted per-tag τ precision lift | ALL-tier precision 36% @ recall 92% unfitted (cocoval2017, 10 pairs × top-20) — fit should trade to ~70%+ precision | calibration |
| D3 | Mixed tag+free-text multi-term spectrum | v1 spectrum requires all-tag tokens (conservative, correct while free-text match was uncalibrated) | feature |
| D4 | Workers-ship-thumbnails geometry (pixel heads without central-session drop) | moderation pixel head drops job to central session, ≈11.3→8.7 img/s measured | perf |
| D5 | Static/calibrated int8 on the REAL x86 target + OpenVINO EP slot | never measured on target; NEON results non-transferable (ADR-10e) | perf/target |
| D6 | Background-prompt margin as free-text calibration feature | commissioned; beats cos → becomes default (ADR-3) | quality |
| D7 | ANN escalation design past ~300k imgs (build-once + brute-during-jobs) | ADR-2 crossover band unmeasured | scale |
| D8 | CoreML/accel opt-in lane for Macs | rclip CoreML 180 img/s reference; CPU-only law keeps it opt-in | feature |
| D9 | Real-server bench run (locks all 🐧 budgets) | LOCK LAW: no 🐧 locks on proxy numbers | validation |
| D10 | git filter-repo history rewrite (611MB) at full quiescence, pre-publish | weights-in-history debt (field log 11:48Z) | hygiene |
| D-violence-1 | violence-track residual FP + optional distilled teacher | intimate/embracing poses fire the "throat-grab" concept (CLIP pose confusion) — the top violation-tier FPs are couples embracing (track-violence.md §3c "residual"). Fix path when labelled data exists: distil a tiny embedding head from a permissive teacher — jaranohaal/vit-base-violence-detection (Apache, fights-only) or LAION violence_detection_*.npy (MIT, unvalidated); gore has NO permissive published-metric teacher (research/violence-models.md §6) | quality/track |
| D-nudity-review-distill | Distilled review-tier head for lingerie / underwear / bare-chest-male | the shared pecore embedding CANNOT separate these review subcats from the FP band (R@5% ≤ 0.13, measured on nudityprobe — swimwear DOES, R@5%=0.57; research/track-nudity.md §11). Fix path (TRACKS T2 earn-it): distil a small embedding head from a permissive teacher — NudeNet v3 `*_COVERED` classes (Apache, already in nudity_subcat.py) are the ready teacher — then it rides the embedding at ~0 cost like weapons/drugs. Violation-tier recall stays operator-`nsfwprobe`-gated (EVAL DATA LAW). | quality/track |

> 2026-07-22: seeded by conductor pre-loop.

- **D12 — improve-track brief v2 A/B (outer loop, ARMED 2026-07-22).** Ledger holds 5
  round-entries under brief_version 1 (weapons/nudity/drugs strict wins from zero; safety's
  honest withhold). Trigger (3+ entries) met. In the darwin loop: mutate the inner brief
  (gate strictness · autoresearch depth · diagnostic-view budget · subcategory-first vs
  metric-first ordering), run v2 on the next comparable rounds (drugs labels-round, safety
  round-3), fitness = delta-per-round vs the v1 baseline the ledger already records.
  Candidate v2 seeds from today: (a) bake READER-PARITY into the brief's verify step (the
  sports lesson — a round is not done until the serving path agrees); (b) require a
  person/object pre-filter step for pose/context tracks (the safety kitten lesson);
  (c) A/B-gate all prompt promotions (the drugs pattern — rejections are half the value).

- **D13 — close the int8-text 3pt gap (2026-07-22, from bench parity).** Shipping int8
  text costs R@10 77.2→74.2 (28% of query NN-sets shift at cos 0.982 — mean-cos lies,
  nn_agree is the metric). Candidates: static/calibrated int8 text quant · smaller text
  projection head · weight-only variants re-gated on nn_agree not cos. Success = ≥76 R@10
  at ≤200MB resident text.
