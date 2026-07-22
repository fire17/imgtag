# IMGTAG — measured CPU numbers from shipping tools (lane final-report capsule)

> Provenance: final report text of the priorart research lane (opus), received 2026-07-22
> 10:14Z as an agent message; preserved here because it distills the only MEASURED numbers
> that exist in the wild. Full per-tool detail: `priorart.md`. Rule the lane obeyed: no
> invented numbers; "no measured number found" = literally nothing published.

## The vacuum (our opportunity)
No mainstream photo tool publishes CPU CLIP image-encode ms with (CPU, threads, batch,
precision) stated. immich's famous model table (7800X3D, f32, "2.29 ms" ViT-B/32) is almost
certainly the TEXT tower — physics: f32 ViT-B/32 image encode at 2.29ms needs ~4 TFLOPS,
above a 7800X3D's AVX-512 peak. ImgTag's bench will be the first honest public number set.

## Hard anchors that DO exist
- **DeepSparse int8 ViT-B/32@256: 1230 img/s** image-encode, 64-core AVX512-VNNI, batch 64
  (NeuralMagic card, 2023-12; 2.84× over fp32; IN zero-shot 71.1 vs 72.8 fp32). Int8 flies
  with the right kernels + hardware.
- **clip.cpp counter-anomaly** (issue #85, intel mac, batch 4): f32 272ms < q5_1 322 < q8_0
  334 < q4_0 539 — quantization SLOWER than f32 on that x86; buys only size (85.6MB @ 4-bit).
  Never re-measured on modern ggml. → measure, never assume int8 wins.
- **ORT vs ggml on CPU** (LocalAI face bench, Ryzen 9 9950X3D, T=8): YuNet+SFace 13.8ms ORT
  vs 43.7ms ggml; ArcFace 13.2 vs 44.6; SCRFD 26.0 vs 153.9. "onnxruntime's MLAS kernels
  win." ggml ahead only ~1.09× on rf-detr (transformer-ish).
- **immich real user CPU throughput**: i5-10500, nllb-clip-large-siglip: ~0.4 img/s. 320k
  photos ≈ 20 days (i7-1355U/N150 OpenVINO wiki entry). The bar to demolish.
- **Ente production stack**: mobileclip_s2_image.onnx fp32 143MB + TEXT tower int8 67MB
  ("~70% shrink, no noticeable decline"); intraOpNumThreads:1, arena off, uint8-RGBA
  preprocessing FUSED INTO THE GRAPH; EP ladder XNNPACK→NNAPI→CPU. Search: 50k brute-force
  ~30min → USearch <30s (their brute path was pathological; 50k×512 matmul is ms — but the
  lesson: search-side engineering matters at scale).
- **immich CPU thread config**: intra_op=2, inter_op=1 defaults; NO CLIP batching
  (MaxBatchSize only for faces/OCR); NO int8/fp16 CLIP artifacts shipped. Both = untapped
  headroom nobody has benchmarked. That is exactly where ImgTag wins.

## Runtime verdict across every shipping tool
ONNX Runtime = de-facto standard (immich, Ente, LibrePhotos-tagging, hydrus taggers,
insightface). TF.js only Nextcloud-recognize (no CLIP, no benchmark table — verified absent);
OpenCV-DNN only digiKam (no CLIP); TF-C/Go only PhotoPrism (no CLIP, VLM captioning at
24-36 s/img on Ryzen AI 9). ggml path abandoned by Ente FOR ONNX Runtime ("most mature",
"reliability across many devices"); clip.cpp dormant since 2024-01.

## Design implications adopted into the bench plan
1. ORT is the primary runtime candidate; ggml/candle only as curiosities. OpenVINO EP worth
   one bench slot on x86; XNNPACK relevance for ARM/edge.
2. Bench must sweep threads × batch × precision (fp32/fp16/int8-dynamic/int8-static) per
   model — published numbers cannot substitute (they don't exist).
3. Fused uint8 preprocessing in-graph (Ente trick) — evaluate; decode+resize may dominate.
4. immich's defaults (intra=2) suggest per-image parallelism beats intra-op on many-core —
   bench process/thread-pool geometry (16 cores here: e.g. 8 workers × 2 threads).
5. Search side at 10k is trivial if done right (matmul), pathological if done wrong (Ente's
   30min). Brute-force numpy first; USearch only beyond ~100k.
