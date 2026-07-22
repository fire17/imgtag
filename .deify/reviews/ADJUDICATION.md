# ADJUDICATION — Wave A review round 1 (2026-07-22, conductor)

22 CRITICAL / 31 IMPORTANT / 19 MINOR across three reviews. Rulings on conflicts + application plan.

## Rulings

1. **Shard format (rev-oracle C-2 vs rev-arch C-1, conflicting fixes):** BOTH superseded by
   the simpler move — **f32 shards on disk, mmap'd, no mirror, f16 dropped entirely**
   (revisit >300k imgs). Disk cost trivial at our scale (20MB @10k); mmap'd f32 pages are
   file-backed/evictable = BETTER for the 8GB box than any heap mirror; BLAS-direct; crash
   recovery simpler. Both reviewers' underlying concerns satisfied with less machinery.
2. **Pipeline geometry (rev-arch C-4/C-5 vs runtime lane's 12×1):** central-ORT-session +
   decode-worker architecture stays the DEFAULT design (memory-superior on 8GB); per-worker-
   session geometry (runtime lane's 181 img/s) becomes a bench-swept axis. Resource policy
   (POLITE/FULL) adopted verbatim from rev-arch C-5; totals govern either geometry.
3. **Corpus registry (rev-budgets C-2):** adopted, sized for disk reality: CORPUS-A coco5k ·
   CORPUS-B photo10k = 10k Unsplash @w=3200 (≈5MP, ≈18GB) · CORPUS-B12 = 300 native
   full-res ≥12MP (decode-bound case) · CORPUS-C mixed10k · CORPUS-D poison (~120 hostile
   files). Plus CAL-SET: ~2k COCO train2017 images (per-image S3 fetch) as the HELD-OUT
   calibration split (rev-budgets C-4). Fetches → l-logistics, Wave B.
4. **B5 replacement (rev-budgets C-1):** adopted exactly (precision@100 mean ≥0.85 over 6
   supercats; per-child recall@R mean ≥0.55, min ≥0.35, no child at 0; all children in
   top-100). The old recall@100 ≥0.8 was mathematically impossible (ceiling 0.086) — my error.
5. **B1 (rev-budgets C-3 + rev-arch C-6):** adopted: polite-mode headline, CORPUS-A ≥150 /
   CORPUS-B ≥60 / ⌂-proxy ≥10; stretch ≥180 (--full-speed, labeled); HEAD-TO-HEAD GATE vs
   rclip same-corpus-same-machine; B15 asserted from the SAME run.
6. **B7 + calibration contract (rev-budgets C-4 + rev-arch C-7):** adopted in full: frozen τ
   in manifest fitted on HELD-OUT CAL-SET; leakage ≤2% AND recall@10 ≥0.70 at the same τ;
   two-layer calibration (model-layer Platt on COCO/LVIS + dataset-layer streaming stats);
   probability-space fusion ONLY (never max over raw scales); near-tag rule θ_syn; calib_sha
   binding with loud refusal; the santa-hat unit test.
7. **Durability (rev-arch C-2/C-3):** adopted verbatim: flock writer exclusion (kernel-owned,
   no pid heuristics), generation-scoped shard names, 7-step fsync flush protocol (incl.
   dir fsync), truncate-on-open-for-write recovery, byte-counts as authority, readers never
   stat, orphans → trash/ never inline-deleted.
8. **Memory (rev-arch C-4):** adopted: uint8-across-boundary with normalization fused into
   the ONNX graph (Ente trick); workers = clamp(ncpu-2, 2, 8); non-JPEG decode semaphore
   (cap 4); B8 restated as process-TREE RSS.
9. **New/fixed budgets (rev-budgets C-5..C-8):** B18 provenance invariant (zero tolerance),
   B20 skill machine-API contract, B24 precision-parity (quant fidelity per rev-oracle C-3:
   gate applies to ALL quantized artifacts INCLUDING official downloads — SigLIP2 official
   int8 must itself pass before serving as anchor), B4/B14 get a real Playwright DEV-dep
   harness (runtime deps unchanged), B21 robustness (CORPUS-D: skip+log, never crash).
10. **Bench noise protocol (rev-oracle C-6):** every bench row = median of ≥3 runs, records
    loadavg; refuses/marks UNRELIABLE above load threshold; darwin loops gate on it.
11. **Escalation contract (rev-oracle C-7):** write surfaces widened: ImgTag tree, ~/.imgtag,
    ~/.claude/skills/imgtag*, scratchpad. (Vision-mandated skill install was blocked.)
12. **WordNet dependency (rev-oracle I-10):** hypernym expansion table is PRECOMPUTED OFFLINE
    (LVIS synsets + OI hierarchy + COCO supercats, all on disk) into a static JSON shipped
    with the engine — no nltk/WordNet at runtime. ADR-7 intact.

## Application

- CRITICAL fixes: applied by conductor now (BUDGETS rewrite + ORACLE edits).
- IMPORTANT + MINOR (50 findings): delegated to one opus editor agent with this adjudication
  as law; conflicts with rulings → flag, don't improvise.
- Re-verify round: rev agents re-run in light mode after application (round 2 of the loop).
