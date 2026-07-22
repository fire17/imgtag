# b-engine — lane checkpoint

> Core-engine lane (store · models · indexer · doctor · progress · cli). Dated notes;
> HEAD is truth, this is the human-readable trail.

## 2026-07-22 14:2x — checkpoint (user save order, during b-bench quiet window)

**State: all owned files committed, clean tree. Last engine commit `45c10f6`.**

Done & verified-by-running (commit trail, newest first):
- `45c10f6` per-category sidecar header (`tracks/<cat>.json`: rows/cols/col_roles/bytes/
  dtype/scorer/model_sha/spec_sha) — b-daemon's read contract; `store.spec_sha` shared.
- `fd584f5` ADR-15 score sidecars + tier-derivation-at-read + `track add` backfill (no
  re-embed) + `alert` tier + free nudity view (bit-identical, gated on backend size 384).
  Category-agnostic tests. Stash@{0} reconciled (store.py + indexer.py + uv.lock).
- `b534d0b` flock = liveness authority (delete-race fixed, repro cleaned) · content dedup
  across jobs · test-home isolation guard (conftest, E1).
- `4c314f0`/`46fa190`/`a2c5c66` B24 precision map · one-owner search (deleted 2nd impl) ·
  ADR-13 daemon client (search 532→80ms) · lazy imports (info 170→70ms).
- `1e138e4` served_by names transport · deprecation lines on legacy adapters.
- `0c0bb39` generic metadata (`--meta`/`--meta-csv`) + moderation hook + `manage delete
  --force`.
- Earlier: `beb531f` store (ADR-6, kill-9 survival) · `cf71028` models+doctor · indexer
  (both geometries) · progress · cli (B20 --json law).

Test posture: 81+ green across engine+contract suites, ruff clean on all engine files.
Measured (M3 Max PROXY, loaded): quick500 fp32-vision 7.6 img/s POLITE; warm search
p50 6.7ms; agent-door search 80ms via daemon.

Open (not mine): `openclip`/`siglip-base` fp32 artifacts are INVALID_PROTOBUF, sizes
disagree with l-logistics' SHA256SUMS → re-fetch owed by l-logistics (quantized load fine).
Reserved extension: multi-column raw margins (C>1) once track heads expose margins —
today heads return one calibrated `p`, so `col_roles=["p"]`.
