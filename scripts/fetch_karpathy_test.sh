#!/usr/bin/env bash
# CORPUS-E "karpathy-test" fetcher — 5,000 val2014 images for the Karpathy TEST split (B17).
# Run build_karpathy_testE.py first (writes data/karpathy-test/.fetch.tsv).
# Idempotent + resumable; POLITE bounded parallelism; curl per ORACLE §4 (certifi law).
set -euo pipefail
DATA="${IMGTAG_DATA:-$HOME/Creations/ImgTag/data}"
OUT="${OUT_DIR:-$DATA/karpathy-test}"
LOG="${LOG_FILE:-$(dirname "$DATA")/data-fetch-corpus.log}"
P="${PAR:-8}"                     # polite mode: <=8 connections
N="${N_E:-0}"                     # 0 = all; else first N (sample runs)
LIST="$OUT/.fetch.tsv"
[ -s "$LIST" ] || { echo "missing $LIST — run scripts/build_karpathy_testE.py first" >&2; exit 1; }

TOTAL=$(wc -l < "$LIST" | tr -d ' ')
echo "[$(date +%FT%T)] CORPUS-E start: $TOTAL urls (N=$N) out=$OUT par=$P" | tee -a "$LOG"
awk -v n="$N" 'n==0 || NR<=n' "$LIST" |
while IFS=$'\t' read -r dst url; do
  [ -s "$dst" ] || printf '%s\t%s\n' "$dst" "$url"
done |
xargs -P "$P" -n 2 sh -c '
  curl -fsSL --retry 3 --retry-delay 2 --max-time 180 -A "imgtag-research/1.0" \
       -o "$1.part" "$2" && mv "$1.part" "$1" || { rm -f "$1.part"; exit 0; }' fetch

HAVE=$(find "$OUT" -name '*.jpg' | wc -l | tr -d ' ')
echo "[$(date +%FT%T)] CORPUS-E done: $HAVE/$TOTAL images, $(du -sh "$OUT" | cut -f1)" | tee -a "$LOG"
