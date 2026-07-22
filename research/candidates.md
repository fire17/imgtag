# research/candidates.md — ADR-4 candidate matrix (phase 1)

> Generated 2026-07-22T15:02:35 · b-bench · git aec1c92

> **EVERY NUMBER IS A PROXY.** Bench host is Apple M3 Max (arm64/NEON); the primary target is shared Linux x86_64, 8GB, no GPU. Per ADR-10e int8 speed/accuracy does NOT transfer across ISAs. No 🐧 row may lock on these.

> Machine: macOS-14.4-arm64-arm-64bit · usable_cores=16 (cpu_count) · ORT 1.27.0 · EP=CPUExecutionProvider · numpy 2.5.1 · Pillow 12.3.0

> Corpora: quality/CORPUS-A = coco5k (5,000 val2017 + exhaustive 80-class truth) · B24 fidelity = 200 quick500 · perf = quick500 tiles. Mode: FULL (phase 1 is a model bench, not an engine bench — POLITE/FULL resource policy applies to `bench index`, phase 2).

> Protocol: median of 3 FRESH processes per perf row; `os.getloadavg()` recorded per run; rows measured at 1-min load > usable_cores x 0.6 are marked **ADVISORY** (the swarm was live — advisory rows are honest, not quiet-machine, numbers).


## Ranked table

| # | candidate | ships | B24 | img/s 1-proc | ms/img | per-worker RSS | workers ≤B8 | proj. POLITE img/s | proj. index RSS | B8 | artifacts | B9 | B6 p@k | B5 p@100 | B5 min-child | B17 R@10 | B7 leak |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---:|
| 1 | `siglip-base-224` | fp32 | ✅ (ref) | — | — | —MB | 0 | — | —MB | 🔴 INELIGIBLE-DEFAULT | 831MB | ❌ | 0.893 | 0.917 | 0.353 | 80.5 | 0.200 |
| 2 | `siglip2-base-224` | fp32 | ✅ (ref) | — | — | —MB | 0 | — | —MB | 🔴 INELIGIBLE-DEFAULT | 673MB | ❌ | 0.925 | 0.937 | 0.357 | 77.5 | 0.200 |
| 3 | `pecore-s16-384` | fp32 | ✅ (ref) | — | — | —MB | 0 | — | —MB | 🔴 INELIGIBLE-DEFAULT | 182MB | ❌ | 0.893 | 0.927 | 0.287 | 77.2 | 0.300 |
| 4 | `pecore-t16-384` | fp32 | ✅ (ref) | — | — | —MB | 0 | — | —MB | 🔴 INELIGIBLE-DEFAULT | 302MB | ❌ | 0.841 | 0.908 | 0.317 | 70.9 | 0.467 |
| 5 | `openclip-vitb32` | fp32 | ✅ (ref) | — | — | —MB | 0 | — | —MB | 🔴 INELIGIBLE-DEFAULT | 622MB | ❌ | 0.775 | 0.832 | 0.246 | 65.2 | 0.600 |

`*` = ADVISORY (machine under swarm load during the timed run).
`ships` = default precision. v1 = **fp32 vision** everywhere (no int8 vision artifact clears B24's DEFAULT nn@200≥0.90 bar). B24 col: `✅ (ref)` = a fp32 row IS its own reference; int8 arms classified `✅ default` / `◐ opt-in` (nn 0.60–0.90, printed deltas) / `❌ banned` (below tier-1 cos 0.95 & nn 0.60). int8 opt-in deltas vs fp32:
  - `siglip-base-224` int8 = **optin** (ΔR@10 -0.4, Δp@k +0.004)
  - `siglip2-base-224` int8 = **banned** (ΔR@10 -31.9, Δp@k -0.112)
  - `pecore-s16-384` int8 = **banned** (ΔR@10 -1.7, Δp@k -0.023)
  - `pecore-t16-384` int8 = **banned** (ΔR@10 -15.7, Δp@k -0.075)
  - `openclip-vitb32` int8 = **optin** (ΔR@10 -0.3, Δp@k +0.023)
`artifacts` = B9 shipping sum: fp32 vision + int8 text + tag table (T×dim×4) + ~11MB binary tokenizer.
`workers ≤B8` = how many streaming workers of this size fit under B8's 1.0GB indexing ceiling; `proj. POLITE img/s` = single-process img/s x min(2, that) — 2 being ADR-11's POLITE clamp `clamp(ncpu−2,2,8)` on a 4-core target. Both are **projections** from a single-process bench, not a measured process tree (that is B8/B1's own bench, phase 2), and they assume the near-linear process scaling the runtime lane measured — which is itself an ARM proxy result.

**B17 control** (`openclip-vitb32`, openai weights, same corpus/run): R@10 = 65.2. Gate: default must reach **77.2**.

## Per-candidate detail

### `pecore-s16-384` — Apache-2.0 · res 384 · dim 512
*primary candidate; self-exported (spike-pecore VERDICT exportable)*

Artifacts: `vision_fp32_mb` 102.6MB · `vision_int8_mb` 33.2MB · `text_fp32_mb` 254.0MB · `text_int8_mb` 64.3MB

**B24 fidelity vs own fp32** (n=200 quick500; gate mean cos ≥0.995, min ≥0.97, top-1 NN agreement ≥0.90):

| precision | mean cos | min cos | p1 cos | NN agree | top-3 overlap | verdict |
|---|---:|---:|---:|---:|---:|---|
| int8 | 0.9272 | 0.7831 | 0.8599 | 0.640 | 0.725 | ❌ FAIL |

**Quality / fp32** on CORPUS-A/coco5k:

- **B6** precision@min(10,N_pos): mean **0.893** · min **0.200** · zeros: none → ❌ (gate mean ≥0.90, min ≥0.70, no zeros)
  - worst 8: `knife` 0.20(k=10), `toaster` 0.25(k=8), `handbag` 0.30(k=10), `potted plant` 0.40(k=10), `remote` 0.40(k=10), `spoon` 0.40(k=10), `mouse` 0.50(k=10), `hair drier` 0.56(k=9)
- **B5** hypernym: p@100 mean **0.927** · child recall@R mean **0.764** min **0.287** · children absent from top-100: cat, horse, sheep, toilet, skateboard, skis, snowboard, surfboard → ❌
  - `vehicle` R=1160 p@100=0.99 · weakest children: boat 0.35, bicycle 0.64, car 0.71
  - `animal` R=1016 p@100=1.00 · weakest children: horse 0.41, dog 0.72, bird 0.76
  - `food` R=708 p@100=0.89 · weakest children: cake 0.64, banana 0.70, apple 0.72
  - `furniture` R=1257 p@100=0.92 · weakest children: dining table 0.29, chair 0.50, potted plant 0.69
  - `appliance` R=320 p@100=0.78 · weakest children: sink 0.39, oven 0.63, refrigerator 0.66
  - `sports` R=938 p@100=0.98 · weakest children: surfboard 0.42, kite 0.51, skateboard 0.52
- **B17** text→image on coco-val2017-5k (NOT Karpathy test split), 25014 caption queries: R@1 **41.4** · R@5 **67.7** · R@10 **77.2** · median rank 2
- **B7** negatives (τ fitted in-run to hold recall@10 ≥0.70): τ=0.3101 · recall@10 0.700 · leakage **0.300** over 30 auto-derived absent queries · margin(present−absent) 0.0475 → ❌ (gate ≤0.02)

**Quality / int8** on CORPUS-A/coco5k:

- **B6** precision@min(10,N_pos): mean **0.870** · min **0.100** · zeros: none → ❌ (gate mean ≥0.90, min ≥0.70, no zeros)
  - worst 8: `handbag` 0.10(k=10), `knife` 0.20(k=10), `toaster` 0.25(k=8), `remote` 0.30(k=10), `mouse` 0.40(k=10), `potted plant` 0.40(k=10), `spoon` 0.40(k=10), `sports ball` 0.40(k=10)
- **B5** hypernym: p@100 mean **0.932** · child recall@R mean **0.754** min **0.305** · children absent from top-100: boat, horse, toilet, kite, skis, snowboard, surfboard → ❌
  - `vehicle` R=1160 p@100=1.00 · weakest children: boat 0.32, bicycle 0.60, car 0.70
  - `animal` R=1016 p@100=1.00 · weakest children: horse 0.43, dog 0.71, bird 0.72
  - `food` R=708 p@100=0.90 · weakest children: cake 0.63, banana 0.66, apple 0.67
  - `furniture` R=1257 p@100=0.94 · weakest children: dining table 0.31, chair 0.50, potted plant 0.69
  - `appliance` R=320 p@100=0.77 · weakest children: sink 0.40, refrigerator 0.62, oven 0.64
  - `sports` R=938 p@100=0.98 · weakest children: surfboard 0.32, kite 0.46, snowboard 0.51
- **B17** text→image on coco-val2017-5k (NOT Karpathy test split), 25014 caption queries: R@1 **38.7** · R@5 **65.2** · R@10 **75.6** · median rank 2
- **B7** negatives (τ fitted in-run to hold recall@10 ≥0.70): τ=0.3061 · recall@10 0.701 · leakage **0.300** over 30 auto-derived absent queries · margin(present−absent) 0.0443 → ❌ (gate ≤0.02)

### `pecore-t16-384` — Apache-2.0 · res 384 · dim 512
*edge tier ~10M params*

Artifacts: `vision_fp32_mb` 32.0MB · `vision_int8_mb` 14.7MB · `text_fp32_mb` 254.0MB

**B24 fidelity vs own fp32** (n=200 quick500; gate mean cos ≥0.995, min ≥0.97, top-1 NN agreement ≥0.90):

| precision | mean cos | min cos | p1 cos | NN agree | top-3 overlap | verdict |
|---|---:|---:|---:|---:|---:|---|
| int8 | 0.8767 | 0.7124 | 0.7190 | 0.335 | 0.448 | ❌ FAIL |

**Quality / fp32** on CORPUS-A/coco5k:

- **B6** precision@min(10,N_pos): mean **0.841** · min **0.100** · zeros: none → ❌ (gate mean ≥0.90, min ≥0.70, no zeros)
  - worst 8: `knife` 0.10(k=10), `mouse` 0.10(k=10), `toaster` 0.12(k=8), `hair drier` 0.22(k=9), `handbag` 0.30(k=10), `remote` 0.30(k=10), `chair` 0.40(k=10), `spoon` 0.40(k=10)
- **B5** hypernym: p@100 mean **0.908** · child recall@R mean **0.757** min **0.317** · children absent from top-100: kite, skis, snowboard, surfboard → ❌
  - `vehicle` R=1160 p@100=1.00 · weakest children: boat 0.40, bicycle 0.60, car 0.72
  - `animal` R=1016 p@100=0.95 · weakest children: horse 0.44, bird 0.58, dog 0.75
  - `food` R=708 p@100=0.90 · weakest children: cake 0.62, banana 0.62, apple 0.68
  - `furniture` R=1257 p@100=0.87 · weakest children: dining table 0.32, chair 0.50, toilet 0.66
  - `appliance` R=320 p@100=0.74 · weakest children: sink 0.46, oven 0.59, refrigerator 0.66
  - `sports` R=938 p@100=0.99 · weakest children: surfboard 0.43, frisbee 0.56, snowboard 0.57
- **B17** text→image on coco-val2017-5k (NOT Karpathy test split), 25014 caption queries: R@1 **33.0** · R@5 **59.1** · R@10 **70.9** · median rank 3
- **B7** negatives (τ fitted in-run to hold recall@10 ≥0.70): τ=0.4427 · recall@10 0.700 · leakage **0.467** over 30 auto-derived absent queries · margin(present−absent) 0.0376 → ❌ (gate ≤0.02)

**Quality / int8** on CORPUS-A/coco5k:

- **B6** precision@min(10,N_pos): mean **0.765** · min **0.000** · zeros: mouse → ❌ (gate mean ≥0.90, min ≥0.70, no zeros)
  - worst 8: `mouse` 0.00(k=10), `knife` 0.10(k=10), `hair drier` 0.11(k=9), `toaster` 0.12(k=8), `remote` 0.20(k=10), `toothbrush` 0.20(k=10), `handbag` 0.30(k=10), `spoon` 0.30(k=10)
- **B5** hypernym: p@100 mean **0.878** · child recall@R mean **0.677** min **0.215** · children absent from top-100: horse, skateboard, skis, snowboard, surfboard → ❌
  - `vehicle` R=1160 p@100=1.00 · weakest children: boat 0.26, bicycle 0.42, car 0.64
  - `animal` R=1016 p@100=0.96 · weakest children: horse 0.30, bird 0.46, dog 0.68
  - `food` R=708 p@100=0.89 · weakest children: banana 0.53, apple 0.57, orange 0.64
  - `furniture` R=1257 p@100=0.79 · weakest children: dining table 0.43, chair 0.51, toilet 0.64
  - `appliance` R=320 p@100=0.66 · weakest children: sink 0.39, refrigerator 0.53, oven 0.55
  - `sports` R=938 p@100=0.97 · weakest children: surfboard 0.21, snowboard 0.41, kite 0.44
- **B17** text→image on coco-val2017-5k (NOT Karpathy test split), 25014 caption queries: R@1 **20.8** · R@5 **43.7** · R@10 **55.1** · median rank 8
- **B7** negatives (τ fitted in-run to hold recall@10 ≥0.70): τ=0.4403 · recall@10 0.700 · leakage **0.800** over 30 auto-derived absent queries · margin(present−absent) 0.0228 → ❌ (gate ≤0.02)

### `siglip2-base-224` — Apache-2.0 · res 224 · dim 768
*quality ANCHOR; official int8 must itself pass B24 (official != audited)*

Artifacts: `vision_fp32_mb` 371.8MB · `vision_int8_mb` 94.6MB · `text_int8_mb` 283.4MB

**B24 fidelity vs own fp32** (n=200 quick500; gate mean cos ≥0.995, min ≥0.97, top-1 NN agreement ≥0.90):

| precision | mean cos | min cos | p1 cos | NN agree | top-3 overlap | verdict |
|---|---:|---:|---:|---:|---:|---|
| int8 | 0.7796 | 0.6003 | 0.6136 | 0.280 | 0.408 | ❌ FAIL |

**Quality / fp32** on CORPUS-A/coco5k:

- **B6** precision@min(10,N_pos): mean **0.925** · min **0.250** · zeros: none → ❌ (gate mean ≥0.90, min ≥0.70, no zeros)
  - worst 8: `toaster` 0.25(k=8), `handbag` 0.30(k=10), `knife` 0.50(k=10), `potted plant` 0.50(k=10), `backpack` 0.60(k=10), `spoon` 0.60(k=10), `hair drier` 0.67(k=9), `cup` 0.70(k=10)
- **B5** hypernym: p@100 mean **0.937** · child recall@R mean **0.814** min **0.357** · children absent from top-100: kite, snowboard, surfboard → ❌
  - `vehicle` R=1160 p@100=0.97 · weakest children: boat 0.42, bicycle 0.70, car 0.71
  - `animal` R=1016 p@100=1.00 · weakest children: horse 0.63, bird 0.72, dog 0.85
  - `food` R=708 p@100=0.83 · weakest children: cake 0.61, apple 0.67, banana 0.69
  - `furniture` R=1257 p@100=0.93 · weakest children: dining table 0.36, chair 0.56, bed 0.72
  - `appliance` R=320 p@100=0.92 · weakest children: sink 0.38, oven 0.77, refrigerator 0.85
  - `sports` R=938 p@100=0.97 · weakest children: kite 0.49, snowboard 0.61, skis 0.74
- **B17** text→image on coco-val2017-5k (NOT Karpathy test split), 25014 caption queries: R@1 **44.4** · R@5 **69.0** · R@10 **77.5** · median rank 2
- **B7** negatives (τ fitted in-run to hold recall@10 ≥0.70): τ=0.1195 · recall@10 0.701 · leakage **0.200** over 30 auto-derived absent queries · margin(present−absent) 0.0327 → ❌ (gate ≤0.02)

**Quality / int8** on CORPUS-A/coco5k:

- **B6** precision@min(10,N_pos): mean **0.813** · min **0.000** · zeros: handbag → ❌ (gate mean ≥0.90, min ≥0.70, no zeros)
  - worst 8: `handbag` 0.00(k=10), `knife` 0.10(k=10), `toaster` 0.12(k=8), `parking meter` 0.20(k=10), `potted plant` 0.20(k=10), `remote` 0.20(k=10), `hair drier` 0.22(k=9), `backpack` 0.30(k=10)
- **B5** hypernym: p@100 mean **0.888** · child recall@R mean **0.694** min **0.305** · children absent from top-100: kite, skateboard, skis, snowboard, surfboard → ❌
  - `vehicle` R=1160 p@100=0.95 · weakest children: boat 0.32, bicycle 0.48, car 0.67
  - `animal` R=1016 p@100=1.00 · weakest children: bird 0.62, horse 0.62, dog 0.64
  - `food` R=708 p@100=0.90 · weakest children: banana 0.52, cake 0.56, apple 0.59
  - `furniture` R=1257 p@100=0.84 · weakest children: dining table 0.31, chair 0.45, bed 0.62
  - `appliance` R=320 p@100=0.66 · weakest children: sink 0.35, refrigerator 0.49, oven 0.50
  - `sports` R=938 p@100=0.98 · weakest children: skateboard 0.39, snowboard 0.55, skis 0.58
- **B17** text→image on coco-val2017-5k (NOT Karpathy test split), 25014 caption queries: R@1 **18.3** · R@5 **36.2** · R@10 **45.6** · median rank 14
- **B7** negatives (τ fitted in-run to hold recall@10 ≥0.70): τ=0.1123 · recall@10 0.700 · leakage **0.267** over 30 auto-derived absent queries · margin(present−absent) 0.0279 → ❌ (gate ≤0.02)

### `siglip-base-224` — Apache-2.0 · res 224 · dim 768
*small text tower (ADR-4 target-profile candidate)*

Artifacts: `vision_fp32_mb` 371.7MB · `vision_int8_mb` 95.8MB · `text_fp32_mb` 441.5MB

**B24 fidelity vs own fp32** (n=200 quick500; gate mean cos ≥0.995, min ≥0.97, top-1 NN agreement ≥0.90):

| precision | mean cos | min cos | p1 cos | NN agree | top-3 overlap | verdict |
|---|---:|---:|---:|---:|---:|---|
| int8 | 0.9878 | 0.9715 | 0.9746 | 0.875 | 0.850 | ❌ FAIL |

**Quality / fp32** on CORPUS-A/coco5k:

- **B6** precision@min(10,N_pos): mean **0.893** · min **0.250** · zeros: none → ❌ (gate mean ≥0.90, min ≥0.70, no zeros)
  - worst 8: `toaster` 0.25(k=8), `handbag` 0.30(k=10), `knife` 0.30(k=10), `backpack` 0.40(k=10), `spoon` 0.50(k=10), `hair drier` 0.56(k=9), `bottle` 0.60(k=10), `bowl` 0.60(k=10)
- **B5** hypernym: p@100 mean **0.917** · child recall@R mean **0.787** min **0.353** · children absent from top-100: bicycle, skis, snowboard, surfboard → ❌
  - `vehicle` R=1160 p@100=0.99 · weakest children: boat 0.36, bicycle 0.54, car 0.69
  - `animal` R=1016 p@100=1.00 · weakest children: horse 0.73, bird 0.76, dog 0.79
  - `food` R=708 p@100=0.81 · weakest children: cake 0.56, banana 0.65, apple 0.68
  - `furniture` R=1257 p@100=0.87 · weakest children: dining table 0.35, chair 0.53, bed 0.64
  - `appliance` R=320 p@100=0.86 · weakest children: sink 0.41, oven 0.72, refrigerator 0.81
  - `sports` R=938 p@100=0.97 · weakest children: skis 0.61, snowboard 0.61, surfboard 0.62
- **B17** text→image on coco-val2017-5k (NOT Karpathy test split), 25014 caption queries: R@1 **47.0** · R@5 **71.8** · R@10 **80.5** · median rank 2
- **B7** negatives (τ fitted in-run to hold recall@10 ≥0.70): τ=0.0891 · recall@10 0.701 · leakage **0.200** over 30 auto-derived absent queries · margin(present−absent) 0.0487 → ❌ (gate ≤0.02)

**Quality / int8** on CORPUS-A/coco5k:

- **B6** precision@min(10,N_pos): mean **0.896** · min **0.250** · zeros: none → ❌ (gate mean ≥0.90, min ≥0.70, no zeros)
  - worst 8: `toaster` 0.25(k=8), `handbag` 0.30(k=10), `knife` 0.30(k=10), `backpack` 0.40(k=10), `spoon` 0.50(k=10), `hair drier` 0.56(k=9), `bowl` 0.60(k=10), `fork` 0.60(k=10)
- **B5** hypernym: p@100 mean **0.907** · child recall@R mean **0.791** min **0.347** · children absent from top-100: skis, snowboard, surfboard → ❌
  - `vehicle` R=1160 p@100=0.98 · weakest children: boat 0.35, bicycle 0.54, car 0.68
  - `animal` R=1016 p@100=1.00 · weakest children: horse 0.74, bird 0.75, dog 0.80
  - `food` R=708 p@100=0.81 · weakest children: cake 0.60, banana 0.66, apple 0.70
  - `furniture` R=1257 p@100=0.85 · weakest children: dining table 0.36, chair 0.53, bed 0.65
  - `appliance` R=320 p@100=0.83 · weakest children: sink 0.41, oven 0.72, refrigerator 0.81
  - `sports` R=938 p@100=0.97 · weakest children: surfboard 0.54, snowboard 0.57, skis 0.59
- **B17** text→image on coco-val2017-5k (NOT Karpathy test split), 25014 caption queries: R@1 **46.4** · R@5 **71.2** · R@10 **80.1** · median rank 2
- **B7** negatives (τ fitted in-run to hold recall@10 ≥0.70): τ=0.0890 · recall@10 0.701 · leakage **0.133** over 30 auto-derived absent queries · margin(present−absent) 0.0476 → ❌ (gate ≤0.02)

### `openclip-vitb32` — MIT · res 224 · dim 512
*B17 CONTROL (openai weights) — never the default*

Artifacts: `vision_fp32_mb` 351.8MB · `vision_int8_mb` 96.0MB · `text_fp32_mb` 254.3MB

**B24 fidelity vs own fp32** (n=200 quick500; gate mean cos ≥0.995, min ≥0.97, top-1 NN agreement ≥0.90):

| precision | mean cos | min cos | p1 cos | NN agree | top-3 overlap | verdict |
|---|---:|---:|---:|---:|---:|---|
| int8 | 0.9901 | 0.9707 | 0.9739 | 0.815 | 0.843 | ❌ FAIL |

**Quality / fp32** on CORPUS-A/coco5k:

- **B6** precision@min(10,N_pos): mean **0.775** · min **0.100** · zeros: none → ❌ (gate mean ≥0.90, min ≥0.70, no zeros)
  - worst 8: `fork` 0.10(k=10), `handbag` 0.10(k=10), `knife` 0.10(k=10), `hair drier` 0.11(k=9), `backpack` 0.20(k=10), `bottle` 0.20(k=10), `chair` 0.20(k=10), `toaster` 0.25(k=8)
- **B5** hypernym: p@100 mean **0.832** · child recall@R mean **0.686** min **0.246** · children absent from top-100: kite, skateboard, skis, snowboard, surfboard → ❌
  - `vehicle` R=1160 p@100=0.87 · weakest children: boat 0.36, bicycle 0.44, car 0.56
  - `animal` R=1016 p@100=0.99 · weakest children: horse 0.39, dog 0.69, bird 0.71
  - `food` R=708 p@100=0.95 · weakest children: cake 0.53, apple 0.63, banana 0.71
  - `furniture` R=1257 p@100=0.72 · weakest children: dining table 0.25, chair 0.44, potted plant 0.64
  - `appliance` R=320 p@100=0.48 · weakest children: sink 0.36, oven 0.37, refrigerator 0.47
  - `sports` R=938 p@100=0.98 · weakest children: skis 0.34, surfboard 0.48, snowboard 0.49
- **B17** text→image on coco-val2017-5k (NOT Karpathy test split), 25014 caption queries: R@1 **29.8** · R@5 **54.1** · R@10 **65.2** · median rank 4
- **B7** negatives (τ fitted in-run to hold recall@10 ≥0.70): τ=0.2726 · recall@10 0.701 · leakage **0.600** over 30 auto-derived absent queries · margin(present−absent) 0.0296 → ❌ (gate ≤0.02)

**Quality / int8** on CORPUS-A/coco5k:

- **B6** precision@min(10,N_pos): mean **0.797** · min **0.100** · zeros: none → ❌ (gate mean ≥0.90, min ≥0.70, no zeros)
  - worst 8: `fork` 0.10(k=10), `handbag` 0.10(k=10), `knife` 0.10(k=10), `hair drier` 0.22(k=9), `toaster` 0.25(k=8), `backpack` 0.30(k=10), `bottle` 0.30(k=10), `apple` 0.40(k=10)
- **B5** hypernym: p@100 mean **0.827** · child recall@R mean **0.686** min **0.259** · children absent from top-100: kite, skateboard, skis, snowboard, surfboard → ❌
  - `vehicle` R=1160 p@100=0.91 · weakest children: boat 0.38, bicycle 0.43, car 0.57
  - `animal` R=1016 p@100=0.99 · weakest children: horse 0.41, dog 0.67, bird 0.73
  - `food` R=708 p@100=0.92 · weakest children: cake 0.52, apple 0.63, banana 0.71
  - `furniture` R=1257 p@100=0.68 · weakest children: dining table 0.26, chair 0.45, potted plant 0.65
  - `appliance` R=320 p@100=0.49 · weakest children: sink 0.35, toaster 0.38, oven 0.39
  - `sports` R=938 p@100=0.97 · weakest children: skis 0.37, kite 0.48, snowboard 0.49
- **B17** text→image on coco-val2017-5k (NOT Karpathy test split), 25014 caption queries: R@1 **29.5** · R@5 **53.5** · R@10 **64.9** · median rank 5
- **B7** negatives (τ fitted in-run to hold recall@10 ≥0.70): τ=0.2757 · recall@10 0.701 · leakage **0.567** over 30 auto-derived absent queries · margin(present−absent) 0.0315 → ❌ (gate ≤0.02)


<!-- HANDWRITTEN -->