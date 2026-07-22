#!/usr/bin/env bash
# Unsplash demo-tier fetch — curl-based (framework Python's urllib lacks SSL certs on this
# machine; curl uses the system trust store). Idempotent: skips existing files.
set -euo pipefail
DATA="${IMGTAG_DATA:-$HOME/Creations/ImgTag/data}"
N="${N_DEMO:-2000}"
OUT="$DATA/unsplash/images"
mkdir -p "$OUT"
awk -F'\t' -v out="$OUT" -v n="$N" '
  NR>1 && NR<=n+1 {print out"/"$1".jpg\t"$3"?w=1080&q=80"}
  NR>n+1 {exit}' "$DATA/unsplash/photos.tsv000" |
while IFS=$'\t' read -r dst url; do
  [ -s "$dst" ] || printf '%s\t%s\n' "$dst" "$url"
done |
xargs -P 12 -n 2 sh -c 'curl -fsSL --retry 2 -A "imgtag-research/1.0" -o "$1.part" "$2" && mv "$1.part" "$1"' fetch
echo "fetched: $(find "$OUT" -name '*.jpg' | wc -l | tr -d ' ') images in $OUT"
