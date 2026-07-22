# UNKNOWNS.md — blind-spot oracle for IMGTAG

> Future models: **check here before solving a problem — it may already be answered.**
> Produced 2026-07-22 by the /unknowns gate run over the full 5-lane research corpus
> (research/*.md), after data download, before candidate selection. Companion: ORACLE.md
> (wargame — decisions, playbooks, escalation). Entries dated; retire on later runs.

## 1. CRITICAL — address before building

**C1. The license fork decides what IMGTAG can ever become.**
Finding: the technically best small models (MobileCLIP2-S0/S2) are `apple-amlr` research-only
— verified from the license text by two independent lanes (re-hosting doesn't launder it).
PE-Core is Apache-2.0 (live-verified 10:09Z after a cross-lane conflict; the FAIR-NC license
is PLM's, not PE's). fire17's registry pattern says projects graduate to published tools.
Resolution (DECIDED): **pluggable model backends**. Default backend must be Apache-2.0/MIT
(PE-Core-S16/T16, SigLIP2-base, SigLIP-v1, UForm, OpenCLIP-B/32 — bench decides ranking).
MobileCLIP2 ships as an opt-in plugin users enable themselves for private use, benched
as the reference ceiling. No `apple-amlr` weights in any published artifact or default path.

**C2. Quantization can regress — never assume int8 is faster.**
Finding: clip.cpp measured q8_0 SLOWER than f32 on x86 (issue #85); an open ORT issue
reports static-uint8 regressions; DeepSparse shows int8 can also be 2.84× faster with the
right kernels; naive numpy int8 scan measured 22× slower than f32 BLAS locally.
Resolution: the bench sweeps precision (fp32 / int8-dynamic / int8-static) × threads
(1/2/4/8/16) × batch (1/8/32) per candidate ON THIS MACHINE, and the engine records the
chosen recipe with its measured numbers in the index manifest (provenance, C5-principle).

**C3. Preprocessing IS the engine — decode dominates by 1–2 orders of magnitude.**
Finding (measured locally, priorart §1.2): worst-case 12MP JPEG decode+resize = 287ms
single-thread vs ~2–6ms model inference (immich's own table). Every incumbent treats decode
as glue; that's why they index at 0.2–2 img/s while their models allow >170 img/s.
Resolution: the architecture puts decode first-class: DCT-scaled `draft()` decode (measured
1.75× win, larger on real photos) → decode-at-target-scale always; parallel decode worker
pool feeding batched inference; EXIF-thumbnail fast path evaluated in bench; libvips as a
bench alternative to Pillow. Indexing pipeline = (N decode workers) → (batch queue) →
(ORT session, all cores) with backpressure.

**C4. Fast preprocessing that silently degrades quality is the classic trap in this field.**
Finding: clip.cpp ships 31.4% ImageNet top-1 vs 66.6% reference for IDENTICAL weights —
caused by resize-interpolation mismatch. Our draft()/thumbnail fast paths risk the same.
Resolution: **preprocessing parity gate in CI**: embeddings of the fast path vs reference
pipeline on quick500 must keep cosine ≥0.99 mean AND quality-bench deltas (precision@10,
hypernym recall) within noise. A fast path that fails parity is disabled, not shipped.

**C5. This machine's framework Python cannot do HTTPS (field-proven today).**
Finding: dataset fetch died on `CERTIFICATE_VERIFY_FAILED` (python.org 3.12 without
certificates wired). Resolution: everything runs in a uv-managed project venv with
`certifi` pinned; model downloads go through curl or certifi-backed httpx; never system
python. (Playbook entry in ORACLE.md.)

## 2. IMPORTANT — will bite within the project's lifetime

**I1. "Semantic flexibility" needs three graded ground truths, not one.**
COCO supercats (vehicle=8 children) test the headline case; LVIS synsets + WordNet closure
test arbitrary-depth hypernyms on the same images; Open Images' independent 600-class tree
cross-checks that recall isn't a taxonomy artifact. All three are downloaded and verified.
Bench scores all three; report per-child recall breakdown (does "vehicle" find trucks as
readily as cars?).

**I2. LVIS federated-annotation FP trap.** Scoring false positives without honoring
`neg_category_ids` / `not_exhaustive_category_ids` over-counts FPs. Resolution: the quality
bench's LVIS scorer implements the federated protocol; unit-tested against a hand-checked
sample. COCO's 80 classes are exhaustive → COCO is the FP source of truth; LVIS adds depth.

**I3. Search latency = text-encoder latency (scan is 0.47ms at 10k).**
Resolution: resident daemon keeps the text tower warm (anti-pattern proven by immich's
model_ttl=300 → 60–70s cold searches); LRU query-embedding cache; precomputed tag-vocab
embedding table (~4–8k tags × D) so tag-path search needs NO text encoder; free-form
queries lazy-load it. Cold start ≤2s is a budget (B13), politeness a new budget (B15).

**I4. Search-while-indexing must be structurally lock-free.**
Resolution (design, to be soak-verified): append-only embedding shards (mmap, contiguous
f32/f16, L2-normalized at write) + atomic manifest rename per flush (~every N images or
T seconds). Readers open the latest manifest snapshot; writers never mutate published
bytes. Coverage % comes free (manifest count vs job total). Bench `concurrent` proves
zero blocking + ≤2s visibility (B11).

**I5. B1 was set below the field's best claim — budgets re-derived before locking.**
rclip claims 119 img/s (M1 Max, but that's CoreML batch-8, not CPU; its pure-CPU number is
unpublished). photofield-ai: 20 img/s on a 2014 6-core. UForm: ~2 img/s/core.
Resolution: bench includes **rclip head-to-head on the same corpus, this machine** as a
system-level baseline; B1 target raised to ≥60 img/s CPU-only (stretch: beat rclip's CoreML
number on pure CPU); edge floor ⌂ raised ≥10 img/s projected. Locked only after the bench.

**I6. Edge/old-computer claims cannot be live-verified on this M3 Max.**
Resolution: honesty protocol — ⌂ budgets reported as "projected via 4-thread throttled run
(documented proxy), NOT yet live-verified on real old hardware". No claimed edge numbers
without a real machine. (Fable credo #2.)

**I7. First-run model download is a product surface.** ~100–400MB from HF; must resume,
sha256-verify, cache under `~/.imgtag/models/`, fail with a clear offline error. The index
manifest records model id+hash — searching with a mismatched model is refused loudly
(silent cross-model search = garbage results, a classic footgun).

**I8. Dependency weight vs "lightweight" law.** Accepted heavy dep: onnxruntime (~50–80MB)
— justified (B5-reuse; MLAS beats alternatives; writing inference = wasted innovation
token). Everything else stays thin: numpy, Pillow, certifi/httpx, stdlib http server or
starlette-class micro. NO torch, NO transformers at runtime (export-time only), NO vector
DB, NO docker requirement. Target install ≤150MB wheel-tree excluding models.

**I9. AGPL contamination hygiene.** immich/Ente/photonix are AGPL; Lap is GPL. We adopt
*ideas* (attributed), never code, from those; code reuse only from MIT/Apache sources
(rclip MIT, LibrePhotos MIT, faiss/UForm/ORT Apache). Note kept in ORACLE decision records.

## 3. WORTH KNOWING — accepted risks & monitored scenarios

- **PE-Core ONNX export is undocumented** — the tagging lane's #1 risk. Mitigation: export
  spike is the FIRST bench task; SigLIP2 (official ONNX) is the guaranteed fallback. If
  export fails in 1 day, PE-Core drops to "revisit-later" without blocking.
- **Two research lanes hit the 200-search session cap** — coverage documented honestly in
  their reports; GitHub API + WebFetch filled gaps. Accepted.
- **Apple accel (CoreML ~180 img/s) exists and users will compare us to it.** CPU-only law
  stands (VISION verbatim); a future opt-in `--accel` lane is designed-for (pluggable
  backend) but not built. Users on Macs get our CPU number honestly labeled.
- **Unsplash images must never be redistributed** (dataset terms) — demo set is fetch-by-
  script only; nothing from data/ is ever committed (gitignored) or published.
- **Runtime-lane naming gap**: two research reports arrived under a generic agent id;
  content was cross-verified against disk and other lanes before adoption. Field-logged.
- **10k-scale honesty**: at 10k images almost everything is fast; the differentiators that
  survive scale scrutiny are decode throughput, cold start, and quality/FP calibration.
  The bench also runs a 100k synthetic-scale scan test so claims degrade gracefully.

## 4. FEATURE IDEAS (opt-in, ranked by value-for-effort)

1. **"Why this matched"** — per-hit nearest-tags explanation (from the tag table, free) —
   turns FP debugging into a feature. (High value, near-zero cost.)
2. **Negative + arithmetic queries** — "car -red", "beach + sunset" (embedding arithmetic;
   rclip prior art, MIT). Cheap on the same scan path.
3. **Live public benchmark page** — the bench emits a hardware-labeled results table into
  the showcase site; the field's #1 unmet demand (HN). Practically free — bench exists.
4. **Duplicate/near-duplicate detection** — same embeddings, cosine >0.98 clustering; a
   dataset-hygiene tool agents can call. (Medium value, small cost.)
5. **Watch-folder incremental indexing** — mtime/hash-gated re-index of changed files only
   (doctrine A4); natural extension of the manifest design.

> All four quadrants were populated by real analysis of this project's evidence; nothing
> generic. Empty categories: none.
