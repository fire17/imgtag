#!/usr/bin/env bash
# CORPUS-B "photo10k" — 10,000 Unsplash images at w=3200 (~5MP) → data/unsplash-b/
#
# Why curl: framework Python's urllib is SSL-dead on this machine (ORACLE §4);
# curl uses the system trust store. Single-awk source + one `while read` filter +
# one xargs sink = no SIGPIPE (proven pattern from fetch_unsplash_demo.sh).
# Idempotent + resumable: existing non-empty files are skipped; partial downloads
# land in *.part and are removed on failure, so a re-run repairs them.
# LOCAL RESEARCH ONLY — Unsplash images are NEVER redistributed.
set -euo pipefail
DATA="${IMGTAG_DATA:-$HOME/Creations/ImgTag/data}"
N="${N_B:-10000}"                   # TSV data rows 1..N  (NR 2..N+1)
OUT="${OUT_DIR:-$DATA/unsplash-b}"
LOG="${LOG_FILE:-$(dirname "$DATA")/data-fetch-corpus.log}"
P="${PAR:-12}"                      # bounded parallelism: <=12 connections
mkdir -p "$OUT"

echo "[$(date +%FT%T)] CORPUS-B start: N=$N out=$OUT par=$P" | tee -a "$LOG"
awk -F'\t' -v out="$OUT" -v n="$N" '
  NR>1 && NR<=n+1 { print out"/"$1".jpg\t"$3"?w=3200&q=85" }
  NR>n+1 { exit }' "$DATA/unsplash/photos.tsv000" |
while IFS=$'\t' read -r dst url; do
  [ -s "$dst" ] || printf '%s\t%s\n' "$dst" "$url"
done |
xargs -P "$P" -n 2 sh -c '
  curl -fsSL --retry 3 --retry-delay 2 --max-time 300 -A "imgtag-research/1.0" \
       -o "$1.part" "$2" && mv "$1.part" "$1" || { rm -f "$1.part"; exit 0; }' fetch

HAVE=$(find "$OUT" -name '*.jpg' | wc -l | tr -d ' ')
echo "[$(date +%FT%T)] CORPUS-B done: $HAVE/$N images, $(du -sh "$OUT" | cut -f1)" | tee -a "$LOG"
