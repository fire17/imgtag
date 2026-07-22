# research/candidates.md — ADR-4 candidate matrix (phase 1)

> Generated 2026-07-22T17:46:10 · b-bench · git aec848c

> **EVERY NUMBER IS A PROXY.** Bench host is Apple M3 Max (arm64/NEON); the primary target is shared Linux x86_64, 8GB, no GPU. Per ADR-10e int8 speed/accuracy does NOT transfer across ISAs. No 🐧 row may lock on these.

> Machine: macOS-14.4-arm64-arm-64bit · usable_cores=16 (cpu_count) · ORT 1.27.0 · EP=CPUExecutionProvider · numpy 2.5.1 · Pillow 12.3.0

> Corpora: quality/CORPUS-A = coco5k (5,000 val2017 + exhaustive 80-class truth) · B24 fidelity = 200 quick500 · perf = quick500 tiles. Mode: FULL (phase 1 is a model bench, not an engine bench — POLITE/FULL resource policy applies to `bench index`, phase 2).

> Protocol: median of 3 FRESH processes per perf row; `os.getloadavg()` recorded per run; rows measured at 1-min load > usable_cores x 0.6 are marked **ADVISORY** (the swarm was live — advisory rows are honest, not quiet-machine, numbers).


## Ranked table

| # | candidate | ships | B24 | img/s 1-proc | ms/img | per-worker RSS | workers ≤B8 | proj. POLITE img/s | proj. index RSS | B8 | artifacts | B9 | B6 p@k | B5 p@100 | B5 min-child | B17 R@10 | B7 leak |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---:|
| 1 | `pecore-s16-384` | fp32 | ✅ (ref) | 8.6* | 116.2 | 425MB | 2 | 17 | 850MB | ✅ | 182MB | ❌ | 0.893 | 0.927 | 0.287 | 77.2 | 0.300 |
| 2 | `pecore-t16-384` | fp32 | ✅ (ref) | 23.2* | 43.1 | 230MB | 4 | 46 | 459MB | ✅ | 112MB | ✅ | 0.841 | 0.908 | 0.317 | 70.9 | 0.467 |
| 3 | `mobileclip2-s2` | fp32 | ✅ (ref) | — | — | —MB | 0 | — | —MB | 🔴 INELIGIBLE-DEFAULT | 413MB | ❌ | 0.914 | 0.913 | 0.337 | 82.0 | 0.133 |
| 4 | `siglip-base-224` | fp32 | ✅ (ref) | 9.8* | 101.5 | 688MB | 1 | 10 | 1376MB | 🔴 INELIGIBLE-DEFAULT | 831MB | ❌ | 0.893 | 0.917 | 0.353 | 80.5 | 0.200 |
| 5 | `mobileclip2-s0` | fp32 | ✅ (ref) | — | — | —MB | 0 | — | —MB | 🔴 INELIGIBLE-DEFAULT | 315MB | ❌ | 0.895 | 0.907 | 0.339 | 78.6 | 0.200 |
| 6 | `siglip2-base-224` | fp32 | ✅ (ref) | 10.1* | 99.5 | 645MB | 1 | 10 | 1291MB | 🔴 INELIGIBLE-DEFAULT | 673MB | ❌ | 0.925 | 0.937 | 0.357 | 77.5 | 0.200 |
| 7 | `openclip-vitb32` | fp32 | ✅ (ref) | 38.4* | 26.1 | 626MB | 1 | 38 | 1252MB | 🔴 INELIGIBLE-DEFAULT | 622MB | ❌ | 0.775 | 0.832 | 0.246 | 65.2 | 0.600 |

`*` = ADVISORY (machine under swarm load during the timed run).
`ships` = default precision. v1 = **fp32 vision** everywhere (no int8 vision artifact clears B24's DEFAULT nn@200≥0.90 bar). B24 col: `✅ (ref)` = a fp32 row IS its own reference; int8 arms classified `✅ default` / `◐ opt-in` (nn 0.60–0.90, printed deltas) / `❌ banned` (below tier-1 cos 0.95 & nn 0.60). int8 opt-in deltas vs fp32:
  - `pecore-s16-384` int8 = **banned** (ΔR@10 -1.7, Δp@k -0.023)
  - `pecore-t16-384` int8 = **banned** (ΔR@10 -15.7, Δp@k -0.075)
  - `siglip-base-224` int8 = **optin** (ΔR@10 -0.4, Δp@k +0.004)
  - `siglip2-base-224` int8 = **banned** (ΔR@10 -31.9, Δp@k -0.112)
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

**Perf matrix** (median of fresh processes; ms/img, img/s, peak RSS):

| precision | intra | batch | img/s | ms/img | peak RSS | spread | load | status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| fp32 | 1 | 1 | 2.59 | 385.71 | 362MB | 0.0% | None | ADVISORY |
| fp32 | 1 | 2 | 2.56 | 390.54 | 440MB | 0.0% | None | ADVISORY |
| fp32 | 1 | 8 | 2.4 | 416.51 | 763MB | 0.0% | None | ADVISORY |
| fp32 | 2 | 1 | 4.84 | 206.56 | 370MB | 0.0% | None | ADVISORY |
| fp32 | 2 | 2 | 4.75 | 210.71 | 467MB | 0.0% | None | ADVISORY |
| fp32 | 2 | 8 | 4.54 | 220.26 | 748MB | 0.0% | None | ADVISORY |
| fp32 | 4 | 1 | 8.35 | 119.71 | 415MB | 0.0% | None | ADVISORY |
| fp32 | 4 | 2 | 8.61 | 116.18 | 425MB | 0.0% | None | ADVISORY |
| fp32 | 4 | 8 | 8.57 | 116.63 | 737MB | 0.0% | None | ADVISORY |
| int8 | 1 | 1 | 4.78 | 209.0 | 208MB | 0.0% | None | ADVISORY |
| int8 | 1 | 2 | 4.71 | 212.46 | 247MB | 0.0% | None | ADVISORY |
| int8 | 1 | 8 | 4.65 | 215.17 | 460MB | 0.0% | None | ADVISORY |
| int8 | 2 | 1 | 8.73 | 114.52 | 220MB | 0.0% | None | ADVISORY |
| int8 | 2 | 2 | 8.75 | 114.33 | 268MB | 0.0% | None | ADVISORY |
| int8 | 2 | 8 | 8.58 | 116.52 | 448MB | 0.0% | None | ADVISORY |
| int8 | 4 | 1 | 13.93 | 71.81 | 188MB | 0.0% | None | ADVISORY |
| int8 | 4 | 2 | 14.72 | 67.91 | 254MB | 0.0% | None | ADVISORY |
| int8 | 4 | 8 | 14.69 | 68.09 | 475MB | 0.0% | None | ADVISORY |

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

Artifacts: `vision_fp32_mb` 32.0MB · `vision_int8_mb` 14.7MB · `text_fp32_mb` 254.0MB · `text_int8_mb` 64.3MB

**B24 fidelity vs own fp32** (n=200 quick500; gate mean cos ≥0.995, min ≥0.97, top-1 NN agreement ≥0.90):

| precision | mean cos | min cos | p1 cos | NN agree | top-3 overlap | verdict |
|---|---:|---:|---:|---:|---:|---|
| int8 | 0.8767 | 0.7124 | 0.7190 | 0.335 | 0.448 | ❌ FAIL |

**Perf matrix** (median of fresh processes; ms/img, img/s, peak RSS):

| precision | intra | batch | img/s | ms/img | peak RSS | spread | load | status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| fp32 | 1 | 1 | 7.63 | 131.02 | 204MB | 0.0% | None | ADVISORY |
| fp32 | 1 | 2 | 7.61 | 131.45 | 228MB | 0.0% | None | ADVISORY |
| fp32 | 1 | 8 | 7.35 | 136.05 | 411MB | 0.0% | None | ADVISORY |
| fp32 | 2 | 1 | 12.97 | 77.11 | 204MB | 0.0% | None | ADVISORY |
| fp32 | 2 | 2 | 14.07 | 71.07 | 240MB | 0.0% | None | ADVISORY |
| fp32 | 2 | 8 | 13.73 | 72.85 | 408MB | 0.0% | None | ADVISORY |
| fp32 | 4 | 1 | 22.25 | 44.94 | 206MB | 0.0% | None | ADVISORY |
| fp32 | 4 | 2 | 23.21 | 43.09 | 230MB | 0.0% | None | ADVISORY |
| fp32 | 4 | 8 | 24.23 | 41.26 | 394MB | 0.0% | None | ADVISORY |
| int8 | 1 | 1 | 11.32 | 88.33 | 147MB | 0.0% | None | ADVISORY |
| int8 | 1 | 2 | 11.2 | 89.28 | 174MB | 0.0% | None | ADVISORY |
| int8 | 1 | 8 | 11.05 | 90.5 | 292MB | 0.0% | None | ADVISORY |
| int8 | 2 | 1 | 18.09 | 55.28 | 145MB | 0.0% | None | ADVISORY |
| int8 | 2 | 2 | 20.16 | 49.6 | 174MB | 0.0% | None | ADVISORY |
| int8 | 2 | 8 | 20.05 | 49.89 | 290MB | 0.0% | None | ADVISORY |
| int8 | 4 | 1 | 29.38 | 34.04 | 150MB | 0.0% | None | ADVISORY |
| int8 | 4 | 2 | 30.99 | 32.26 | 158MB | 0.0% | None | ADVISORY |
| int8 | 4 | 8 | 33.87 | 29.53 | 296MB | 0.0% | None | ADVISORY |

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

**Perf matrix** (median of fresh processes; ms/img, img/s, peak RSS):

| precision | intra | batch | img/s | ms/img | peak RSS | spread | load | status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| fp32 | 1 | 1 | 2.69 | 371.78 | 627MB | 0.0% | None | ADVISORY |
| fp32 | 1 | 2 | 2.68 | 373.21 | 675MB | 0.0% | None | ADVISORY |
| fp32 | 1 | 8 | 2.51 | 398.36 | 884MB | 0.0% | None | ADVISORY |
| fp32 | 2 | 1 | 5.25 | 190.32 | 631MB | 0.0% | None | ADVISORY |
| fp32 | 2 | 2 | 5.23 | 191.1 | 670MB | 0.0% | None | ADVISORY |
| fp32 | 2 | 8 | 4.93 | 203.04 | 822MB | 0.0% | None | ADVISORY |
| fp32 | 4 | 1 | 10.05 | 99.46 | 645MB | 0.0% | None | ADVISORY |
| fp32 | 4 | 2 | 10.04 | 99.56 | 695MB | 0.0% | None | ADVISORY |
| fp32 | 4 | 8 | 9.5 | 105.31 | 835MB | 0.0% | None | ADVISORY |
| int8 | 1 | 1 | 7.17 | 139.46 | 347MB | 0.0% | None | ADVISORY |
| int8 | 1 | 2 | 7.21 | 138.76 | 406MB | 0.0% | None | ADVISORY |
| int8 | 1 | 8 | 6.96 | 143.58 | 574MB | 0.0% | None | ADVISORY |
| int8 | 2 | 1 | 13.72 | 72.86 | 346MB | 0.0% | None | ADVISORY |
| int8 | 2 | 2 | 13.83 | 72.31 | 372MB | 0.0% | None | ADVISORY |
| int8 | 2 | 8 | 13.55 | 73.8 | 563MB | 0.0% | None | ADVISORY |
| int8 | 4 | 1 | 25.4 | 39.36 | 367MB | 0.0% | None | ADVISORY |
| int8 | 4 | 2 | 26.1 | 38.31 | 394MB | 0.0% | None | ADVISORY |
| int8 | 4 | 8 | 23.95 | 41.75 | 581MB | 0.0% | None | ADVISORY |

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

**Perf matrix** (median of fresh processes; ms/img, img/s, peak RSS):

| precision | intra | batch | img/s | ms/img | peak RSS | spread | load | status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| fp32 | 1 | 1 | 2.68 | 373.15 | 620MB | 0.0% | None | ADVISORY |
| fp32 | 1 | 2 | 2.66 | 375.97 | 660MB | 0.0% | None | ADVISORY |
| fp32 | 1 | 8 | 2.52 | 397.06 | 760MB | 0.0% | None | ADVISORY |
| fp32 | 2 | 1 | 5.24 | 191.01 | 634MB | 0.0% | None | ADVISORY |
| fp32 | 2 | 2 | 5.25 | 190.36 | 637MB | 0.0% | None | ADVISORY |
| fp32 | 2 | 8 | 5.0 | 199.8 | 772MB | 0.0% | None | ADVISORY |
| fp32 | 4 | 1 | 9.72 | 102.91 | 629MB | 0.0% | None | ADVISORY |
| fp32 | 4 | 2 | 9.85 | 101.55 | 688MB | 0.0% | None | ADVISORY |
| fp32 | 4 | 8 | 9.32 | 107.31 | 784MB | 0.0% | None | ADVISORY |
| int8 | 1 | 1 | 7.37 | 135.76 | 362MB | 0.0% | None | ADVISORY |
| int8 | 1 | 2 | 7.36 | 135.8 | 364MB | 0.0% | None | ADVISORY |
| int8 | 1 | 8 | 7.16 | 139.74 | 451MB | 0.0% | None | ADVISORY |
| int8 | 2 | 1 | 14.08 | 71.01 | 350MB | 0.0% | None | ADVISORY |
| int8 | 2 | 2 | 14.41 | 69.42 | 344MB | 0.0% | None | ADVISORY |
| int8 | 2 | 8 | 14.14 | 70.72 | 443MB | 0.0% | None | ADVISORY |
| int8 | 4 | 1 | 26.79 | 37.32 | 331MB | 0.0% | None | ADVISORY |
| int8 | 4 | 2 | 27.34 | 36.58 | 376MB | 0.0% | None | ADVISORY |
| int8 | 4 | 8 | 26.96 | 37.1 | 458MB | 0.0% | None | ADVISORY |

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

### `mobileclip2-s0` — apple-amlr · res 256 · dim 512
*CEILING-REFERENCE only*

Artifacts: `vision_fp32_mb` 45.6MB · `text_fp32_mb` 254.1MB

**Quality / fp32** on CORPUS-A/coco5k:

- **B6** precision@min(10,N_pos): mean **0.895** · min **0.200** · zeros: none → ❌ (gate mean ≥0.90, min ≥0.70, no zeros)
  - worst 8: `handbag` 0.20(k=10), `toaster` 0.25(k=8), `knife` 0.30(k=10), `potted plant` 0.40(k=10), `backpack` 0.50(k=10), `bottle` 0.50(k=10), `hair drier` 0.56(k=9), `mouse` 0.60(k=10)
- **B5** hypernym: p@100 mean **0.907** · child recall@R mean **0.778** min **0.339** · children absent from top-100: toilet, kite, skis, snowboard, surfboard → ❌
  - `vehicle` R=1160 p@100=0.97 · weakest children: boat 0.34, bicycle 0.49, car 0.69
  - `animal` R=1016 p@100=1.00 · weakest children: horse 0.68, bird 0.74, dog 0.80
  - `food` R=708 p@100=0.84 · weakest children: banana 0.64, apple 0.64, orange 0.72
  - `furniture` R=1257 p@100=0.95 · weakest children: dining table 0.35, chair 0.53, potted plant 0.67
  - `appliance` R=320 p@100=0.69 · weakest children: sink 0.36, oven 0.60, refrigerator 0.61
  - `sports` R=938 p@100=0.99 · weakest children: surfboard 0.52, kite 0.56, snowboard 0.61
- **B17** text→image on coco-val2017-5k (NOT Karpathy test split), 25014 caption queries: R@1 **42.7** · R@5 **68.9** · R@10 **78.6** · median rank 2
- **B7** negatives (τ fitted in-run to hold recall@10 ≥0.70): τ=0.2596 · recall@10 0.701 · leakage **0.200** over 30 auto-derived absent queries · margin(present−absent) 0.0580 → ❌ (gate ≤0.02)

### `mobileclip2-s2` — apple-amlr · res 256 · dim 512
*CEILING-REFERENCE only*

Artifacts: `vision_fp32_mb` 143.0MB · `text_fp32_mb` 254.1MB

**Quality / fp32** on CORPUS-A/coco5k:

- **B6** precision@min(10,N_pos): mean **0.914** · min **0.250** · zeros: none → ❌ (gate mean ≥0.90, min ≥0.70, no zeros)
  - worst 8: `toaster` 0.25(k=8), `handbag` 0.40(k=10), `potted plant` 0.40(k=10), `knife` 0.50(k=10), `hair drier` 0.56(k=9), `backpack` 0.60(k=10), `spoon` 0.60(k=10), `sports ball` 0.60(k=10)
- **B5** hypernym: p@100 mean **0.913** · child recall@R mean **0.786** min **0.337** · children absent from top-100: kite, snowboard, surfboard → ❌
  - `vehicle` R=1160 p@100=0.98 · weakest children: boat 0.35, bicycle 0.49, car 0.71
  - `animal` R=1016 p@100=0.99 · weakest children: horse 0.63, bird 0.74, dog 0.83
  - `food` R=708 p@100=0.84 · weakest children: apple 0.59, banana 0.59, cake 0.67
  - `furniture` R=1257 p@100=0.96 · weakest children: dining table 0.36, chair 0.54, potted plant 0.74
  - `appliance` R=320 p@100=0.71 · weakest children: sink 0.34, oven 0.63, microwave 0.65
  - `sports` R=938 p@100=1.00 · weakest children: snowboard 0.51, surfboard 0.53, kite 0.64
- **B17** text→image on coco-val2017-5k (NOT Karpathy test split), 25014 caption queries: R@1 **48.5** · R@5 **73.5** · R@10 **82.0** · median rank 2
- **B7** negatives (τ fitted in-run to hold recall@10 ≥0.70): τ=0.2865 · recall@10 0.701 · leakage **0.133** over 30 auto-derived absent queries · margin(present−absent) 0.0697 → ❌ (gate ≤0.02)

### `openclip-vitb32` — MIT · res 224 · dim 512
*B17 CONTROL (openai weights) — never the default*

Artifacts: `vision_fp32_mb` 351.8MB · `vision_int8_mb` 96.0MB · `text_fp32_mb` 254.3MB

**B24 fidelity vs own fp32** (n=200 quick500; gate mean cos ≥0.995, min ≥0.97, top-1 NN agreement ≥0.90):

| precision | mean cos | min cos | p1 cos | NN agree | top-3 overlap | verdict |
|---|---:|---:|---:|---:|---:|---|
| int8 | 0.9901 | 0.9707 | 0.9739 | 0.815 | 0.843 | ❌ FAIL |

**Perf matrix** (median of fresh processes; ms/img, img/s, peak RSS):

| precision | intra | batch | img/s | ms/img | peak RSS | spread | load | status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| fp32 | 1 | 1 | 10.28 | 97.23 | 589MB | 0.0% | None | ADVISORY |
| fp32 | 1 | 2 | 10.64 | 93.96 | 613MB | 0.0% | None | ADVISORY |
| fp32 | 1 | 8 | 10.38 | 96.29 | 687MB | 0.0% | None | ADVISORY |
| fp32 | 2 | 1 | 19.45 | 51.41 | 624MB | 0.0% | None | ADVISORY |
| fp32 | 2 | 2 | 20.48 | 48.83 | 632MB | 0.0% | None | ADVISORY |
| fp32 | 2 | 8 | 20.51 | 48.75 | 680MB | 0.0% | None | ADVISORY |
| fp32 | 4 | 1 | 35.63 | 28.06 | 648MB | 0.0% | None | ADVISORY |
| fp32 | 4 | 2 | 38.37 | 26.06 | 626MB | 0.0% | None | ADVISORY |
| fp32 | 4 | 8 | 39.17 | 25.53 | 642MB | 0.0% | None | ADVISORY |
| int8 | 1 | 1 | 27.58 | 36.26 | 313MB | 0.0% | None | ADVISORY |
| int8 | 1 | 2 | 28.84 | 34.68 | 312MB | 0.0% | None | ADVISORY |
| int8 | 1 | 8 | 28.91 | 34.59 | 348MB | 0.0% | None | ADVISORY |
| int8 | 2 | 1 | 50.54 | 19.79 | 299MB | 0.0% | None | ADVISORY |
| int8 | 2 | 2 | 53.42 | 18.72 | 350MB | 0.0% | None | ADVISORY |
| int8 | 2 | 8 | 55.15 | 18.13 | 378MB | 0.0% | None | ADVISORY |
| int8 | 4 | 1 | 86.02 | 11.63 | 324MB | 0.0% | None | ADVISORY |
| int8 | 4 | 2 | 93.77 | 10.66 | 362MB | 0.0% | None | ADVISORY |
| int8 | 4 | 8 | 101.99 | 9.81 | 380MB | 0.0% | None | ADVISORY |

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

> **CHECKPOINT 2026-07-22 ~17:55 (b-bench).** Phase 1 contention-immune work COMPLETE +
> committed: candidate matrix (7 arms incl. MC2 ceiling), B24 two-tier, CAL-SET Platt fit
> INSTALLED + verified live end-to-end by b-daemon (model_sha 8c080c43… matched first try,
> honest no_match confirmed). BUDGETS B6 (N_pos≥25 floor) + B9 (≤200MB fp32-vision floor)
> edited per conductor rulings. **PERF STILL ADVISORY** — the authoritative quiet-window
> throughput pass has NOT been run (see the honesty note below the table). The WINNER verdict
> is unaffected: it rests on per-worker RSS (contention-immune) + quality (contention-immune),
> not on img/s. Karpathy-split B17 + fp16-weights RSS/speed bench = next pass.

## WINNER — the default backend recommendation (b-bench, phase 1)

> Everything above this marker is auto-generated each run; everything below is the human
> read. Numbers are M3 Max PROXY (perf ADVISORY under swarm load until a quiet-window
> pass); quality/fidelity/RSS are contention-immune and final.

**Default backend: `pecore-s16-384`, fp32 vision + int8 text.**

It is the *only* candidate that is **both** B8-eligible on the 8GB target **and** near the
B17 +12pt-vs-control gate. ⚠️ **The R@10 column below is measured with fp32 TEXT (the
reference-quality ceiling). The shipped 8GB config uses int8 TEXT** (fp32 text is ~850MB
resident, blows B8) — `bench parity` measured int8 text at **−3.1 pts R@10** (nn_agree 0.72,
a 28% query-rank shift that mean-cos 0.98 hides). So the winner's SHIPPED number is R@10
**74.2 = +9.0pt** over control, not 77.2/+12. Escalated: the +12 gate was set on a text
tower that doesn't fit the target (ruling pending — likely relax to +9).

| candidate | B8 (per-worker RSS) | R@10 fp32-text (ceiling) | R@10 int8-text (shipped) | verdict |
|---|---|---|---|---|
| **pecore-s16-384** | ✅ 425MB | **77.2** | **74.2** (+9.0) | **DEFAULT** |
| pecore-t16-384 | ✅ 230MB | 70.9 | ~67.8 | edge/speed fallback |
| siglip-base-224 | 🔴 688MB | 80.5 (best) | — | B8-INELIGIBLE (fat vision + 785MB text) |
| siglip2-base-224 | 🔴 645MB | 77.5 | — | B8-INELIGIBLE (787MB int8 text tower) |
| openclip-vitb32 | 🔴 626MB | 65.2 | — | B17 CONTROL only |

### The decisive findings (all measured, not projected)

1. **No int8 VISION artifact clears B24's default bar** (nn@200 ≥0.90). The best is
   `siglip-base` int8 at nn 0.875; PE-Core families sit at 0.33–0.64. The ADR-4 weight-only
   recipe did **not** rescue PE-Core vision (swept per-tensor/per-channel/QInt8/+Gemm — all
   fail). **v1 ships fp32 vision everywhere.** int8 vision remains an opt-in speed lane only
   where it clears tier-1 (cos ≥0.95 & nn ≥0.60), with printed deltas.

2. **SigLIP2's OFFICIAL onnx-community int8 is the worst artifact on the roster** — cos
   0.78, nn 0.28, **−31.9 pts R@10** (77.5→45.6). "Official ≠ audited" confirmed; belongs
   on the ORACLE blacklist beside `Xenova/mobileclip_s0`.

3. **The SigLIP text towers are B8-fatal**: int8-resident 787MB (siglip2), fp32 785MB
   (siglip-base) — vs PE-Core int8 text at 183–190MB. Only PE-Core's text path fits an 8GB
   box resident. This, not the vision tower, is what makes the quality leaders ineligible.

4. **fp16-weights is a disk win, not a B8 win.** Bit-equivalent (cos 0.999999) at half the
   disk, but CPU-EP Cast-to-fp32 makes runtime peak RSS *higher* than native fp32 (938 vs
   622MB @batch1 on siglip2). It cannot rescue any candidate into B8; use it only to shrink
   B9 disk, and only if the RSS regression is acceptable.

### Open decisions handed up to the conductor

- **B9 vs the winner**: `pecore-s16-384` shipping sum = 182MB > B9's 150MB (fp32 vision is
  102MB). B9 assumed a quantizable vision tower that B24 removed. Either relax B9 to the
  honest fp32-vision floor, or accept `pecore-t16-384` (~112MB, B9 ✅) at −6.3 pts R@10.
- **B6 as a hard gate fails for every model** (min drags on toaster n=8 / hair-drier n=9) —
  a dataset-sparsity artifact, not a model difference; suggest an N_pos≥25 floor on the min.
- **B17 canonical**: the true Karpathy 5k test split (`data/karpathy/dataset_coco.json`) is
  val2014 and overlaps our val2017 corpus by only 593/5000 — the "within 2pts of model card"
  clause needs a val2014 image fetch. The +12pt-vs-control gate is valid as reported (one
  corpus, one run).

### Calibration (ADR-3 layer-1) — installed, honest ceiling documented
`~/.imgtag/models/<sha256(pecore-s16 fp32 vision)>/tags.{f32,json}`: 2,177-tag two-tier
table, 80 COCO tags Platt+τ-fitted on the held-out CAL-SET, the rest τ=null (honest
uncalibrated). The amended acceptance (ALL-tier AND-precision 36%→70% @ recall≥85%) is
**architecturally unreachable** — an ORACLE upper bound (τ fit directly on val, cheating)
tops out at 0.281; a single global embedding conflates car/bus/street-scene. Honest-no-match,
rank-boost and rare/mid-tag FP-gating all work today; the AND-precision target waits on
region/tile embeddings (darwin D1).