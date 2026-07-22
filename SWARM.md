# SWARM.md — the ImgTag agent fleet

> Grand-start Phase 6. Zenith MCP unavailable in this session (predates workspace init) →
> sanctioned downgrade: Agent-tool teammates, ≥12 roster, main agent stays free (F1).
> Model law: every spawn explicit, never Fable (workflow-model-guard). Builders activate
> /ponytail (full) inside their own context — design is locked (ORACLE ADRs), so the
> ladder applies to code, not to the bar.

## Roster (14)

| # | Agent | Lane | Model | Wave | Duty |
|---|---|---|---|---|---|
| 1 | progresslive-imgtag | tracker (resident) | sonnet | 0 ✅ | board + swarm-scan |
| 2 | research-{models,tagging,runtime,priorart,datasets} | research | opus ×5 | 0 ✅ | done — reports in research/ |
| 3 | rev-oracle | revalidation | opus | A | refute ORACLE ADRs/playbooks vs evidence |
| 4 | rev-budgets | revalidation | opus | A | attack BUDGETS thresholds + measurability |
| 5 | rev-architecture | revalidation | opus | A | attack storage/concurrency/decode design |
| 6 | spike-pecore | empirical spike | opus | A | PE-Core→ONNX export + first CPU numbers (top risk) |
| 7 | spike-siglip2 | empirical spike | opus | A | official SigLIP2 ONNX local numbers (anchor) |
| 8 | b-engine | build (ponytail) | opus | B | core lib: decode pipeline→ORT→shards/manifest |
| 9 | b-bench | build (ponytail) | opus | B | candidate bench harness (all `imgtag bench *`) |
| 10 | b-daemon | build (ponytail) | opus | B | resident daemon + HTTP API + progress stream |
| 11 | b-app | build (ponytail+impeccable) | opus | B | PoC app: gallery/search/jobs views |
| 12 | b-skill | build (ponytail) | opus | B | global agent skill (~/.claude/skills/imgtag) |
| 13 | b-showcase | build (impeccable) | opus | C | showcase + dev-showcase sites |
| 14 | l-logistics | logistics | haiku | B | model-file fetches w/ sha256 (escalation clause armed) |
| 15 | editor-reviews | revalidation | opus | A2 ✅sp | apply 50 IMPORTANT/MINOR review fixes under ADJUDICATION law |
| 16 | b-corpus | build | opus | A2 ✅sp | CORPUS-B/B12/D/CAL-SET builder (bench foundation) |

Wave A done (3 reviews in, 22 CRITICALs applied — ADJUDICATION.md). Wave A2 running: editor-reviews + b-corpus + spikes finishing. Wave B after A2.
Wave C after the engine works. Darwin loop agents (Phase 7) spawn post-delivery.

> 2026-07-22: Wave A spawned. Zenith downgrade logged (flag for final report).
