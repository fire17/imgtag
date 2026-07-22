#!/usr/bin/env bash
# CAL-SET "cocotrain2k" fetcher — per-image pulls from images.cocodataset.org.
# Run build_coco_train2k.py first (writes data/coco-train2k/.fetch.tsv + instances_cal.json).
# Idempotent + resumable; bounded parallelism; curl per ORACLE §4.
set -euo pipefail
DATA="${IMGTAG_DATA:-$HOME/Creations/ImgTag/data}"
OUT="${OUT_DIR:-$DATA/coco-train2k}"
LOG="${LOG_FILE:-$(dirname "$DATA")/data-fetch-corpus.log}"
P="${PAR:-12}"
N="${N_CAL:-0}"                    # 0 = all; otherwise first N urls (sample runs)
LIST="$OUT/.fetch.tsv"
[ -s "$LIST" ] || { echo "missing $LIST — run scripts/build_coco_train2k.py first" >&2; exit 1; }

TOTAL=$(wc -l < "$LIST" | tr -d ' ')
echo "[$(date +%FT%T)] CAL-SET start: $TOTAL urls (N=$N) out=$OUT" | tee -a "$LOG"
awk -v n="$N" 'n==0 || NR<=n' "$LIST" |
while IFS=$'\t' read -r dst url; do
  [ -s "$dst" ] || printf '%s\t%s\n' "$dst" "$url"
done |
xargs -P "$P" -n 2 sh -c '
  curl -fsSL --retry 3 --retry-delay 2 --max-time 180 -A "imgtag-research/1.0" \
       -o "$1.part" "$2" && mv "$1.part" "$1" || { rm -f "$1.part"; exit 0; }' fetch

HAVE=$(find "$OUT" -name '*.jpg' | wc -l | tr -d ' ')
echo "[$(date +%FT%T)] CAL-SET done: $HAVE/$TOTAL images, $(du -sh "$OUT" | cut -f1)" | tee -a "$LOG"
