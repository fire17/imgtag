#!/usr/bin/env bash
# CORPUS-B12 "fullres300" — 300 NATIVE full-resolution Unsplash originals (>=12MP target)
# → data/unsplash-fullres/. The decode-bound case for the bench.
#
# URL: raw photo_image_url with NO width param => server returns the native original.
# Candidate window: TSV data rows 10001..11000 (NO overlap with photo10k = rows 1..10000),
# metadata-prefiltered to photo_width>=3000 AND width*height>=12MP, then each downloaded
# file is dimension-verified (>=3000px wide) by verify_corpus.py; the first 300 that
# verify are kept, surplus candidates are deleted.
# curl per ORACLE §4 (framework python SSL-dead). Idempotent + resumable.
# LOCAL RESEARCH ONLY — never redistributed.
set -euo pipefail
DATA="${IMGTAG_DATA:-$HOME/Creations/ImgTag/data}"
OUT="${OUT_DIR:-$DATA/unsplash-fullres}"
LOG="${LOG_FILE:-$(dirname "$DATA")/data-fetch-corpus.log}"
WANT="${WANT:-300}"
CAND="${CAND:-$((WANT + 60))}"      # buffer for rows that fail live verification
P="${PAR:-8}"                       # big files: keep <=8 connections
LIST="$OUT/.candidates.tsv"
mkdir -p "$OUT"

echo "[$(date +%FT%T)] CORPUS-B12 start: want=$WANT cand=$CAND out=$OUT" | tee -a "$LOG"

# deterministic candidate list (TSV order), written once, reused on re-runs
awk -F'\t' -v out="$OUT" -v c="$CAND" '
  NR>=10002 && NR<=11001 && $6>=3000 && ($6*$7)>=12000000 {
    print out"/"$1".jpg\t"$3; k++ }
  k>=c || NR>11001 { exit }' "$DATA/unsplash/photos.tsv000" > "$LIST"
echo "candidates: $(wc -l < "$LIST" | tr -d ' ')" | tee -a "$LOG"

while IFS=$'\t' read -r dst url; do
  [ -s "$dst" ] || printf '%s\t%s\n' "$dst" "$url"
done < "$LIST" |
xargs -P "$P" -n 2 sh -c '
  curl -fsSL --retry 3 --retry-delay 2 --max-time 600 -A "imgtag-research/1.0" \
       -o "$1.part" "$2" && mv "$1.part" "$1" || { rm -f "$1.part"; exit 0; }' fetch

# verify + prune: drop anything <3000px wide, then trim to WANT in candidate order
python3 - "$OUT" "$LIST" "$WANT" <<'PY' | tee -a "$LOG"
import sys, os
from PIL import Image
out, lst, want = sys.argv[1], sys.argv[2], int(sys.argv[3])
kept = 0
for line in open(lst):
    dst = line.split('\t')[0]
    if not os.path.exists(dst):
        continue
    try:
        with Image.open(dst) as im:
            w, h = im.size
    except Exception:
        os.remove(dst); continue
    if w < 3000 or kept >= want:
        os.remove(dst)
    else:
        kept += 1
print(f"CORPUS-B12 verified+kept: {kept}/{want}")
PY

HAVE=$(find "$OUT" -name '*.jpg' | wc -l | tr -d ' ')
echo "[$(date +%FT%T)] CORPUS-B12 done: $HAVE images, $(du -sh "$OUT" | cut -f1)" | tee -a "$LOG"
