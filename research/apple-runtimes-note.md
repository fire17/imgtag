# Apple-runtimes lane capsule (received 2026-07-22 10:31Z as agent message)

Full report preserved in the session transcript; the load-bearing conclusions for IMGTAG:

## Governance ruling (main session): VISION says "using only the cpu"
CoreML/ANE/MLX are NOT the core path — they are a possible OPT-IN accelerator lane
(post-MVP, macOS-only). Every budget stays CPU-pure. This report informs (a) the bar
(rclip CoreML gets 180 img/s on M1 Max — our CPU-only number will be compared to it by
users), and (b) a future `--accel coreml` flag design.

## Hard evidence adopted
- rclip PR #249: coremltools CompiledMLModel + ComputeUnit.ALL, batch 8 → ~180 img/s
  (M1 Max, ViT-B-32-256); its ORT-CPU is used for text/query only; maintainer: "onnxruntime
  cannot get close to PyTorch+MPS speeds for indexing" (qualitative).
- rclip ORT-CPU search on M1 Max: text ~0.5s/672MB — our warm-daemon design must beat this
  10× (B3 p50 ≤50ms is exactly that claim).
- MLX = GPU-only (no ANE, confirmed by maintainers) — not a CPU candidate.
- ORT CoreML EP = partition thrash + silent fp16 + recompile-per-run → rejected even for
  the accel lane; coremltools direct is the proven path there.
- ANE constraints (Orion arXiv:2603.06728): int8 gives NO speed win on ANE (dequant to
  fp16); ~32MB SRAM cliff; naive HF CLIP conversion loses the ANE-transformer layout wins.
- Queryable ships MobileCLIP-S2 on iPhone at ~33 img/s — an iPhone 12 mini beats every
  self-hosted x86 incumbent; the efficiency ceiling is real.
- immich has NO Apple Silicon acceleration story at all (Linux/WSL2 only) — on Macs the
  field is even weaker than the Linux numbers suggested.

## Gaps confirmed (13 "no measured number found" items)
CPU-EP CLIP img/s on Apple Silicon unpublished; MobileCLIP Mac latency unpublished;
fp16-vs-fp32 recall delta unmeasured — all territory our bench claims first.
