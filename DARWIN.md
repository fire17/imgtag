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

> 2026-07-22: seeded by conductor pre-loop.
