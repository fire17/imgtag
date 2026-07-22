#!/usr/bin/env python3
"""Corpus sample verifier — dims/bytes/megapixels over a directory of images.

Usage: verify_corpus.py <dir> [sample_n] [--min-width N]
Prints per-corpus counts, dim distribution and mean bytes (used for size projection).
Local-only, no network. Pillow is the only dep (framework python has it).
"""
import sys, os, glob, statistics
from PIL import Image

d = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("-") else 50
min_w = 0
if "--min-width" in sys.argv:
    min_w = int(sys.argv[sys.argv.index("--min-width") + 1])

files = sorted(glob.glob(os.path.join(d, "*")))
files = [f for f in files if os.path.isfile(f) and not f.endswith(".part")]
total = len(files)
sample = files[:n] if n > 0 else files

ok, bad, widths, mps, sizes = 0, [], [], [], []
for f in sample:
    try:
        with Image.open(f) as im:
            w, h = im.size
        b = os.path.getsize(f)
        if b == 0 or (min_w and w < min_w):
            bad.append((os.path.basename(f), w, h, b)); continue
        ok += 1; widths.append(w); mps.append(w * h / 1e6); sizes.append(b)
    except Exception as e:
        bad.append((os.path.basename(f), "ERR", str(e)[:60], os.path.getsize(f)))

print(f"dir={d}\nfiles_on_disk={total} sampled={len(sample)} ok={ok} bad={len(bad)}")
if ok:
    print(f"width  min/median/max = {min(widths)}/{int(statistics.median(widths))}/{max(widths)}")
    print(f"MP     min/median/max = {min(mps):.1f}/{statistics.median(mps):.1f}/{max(mps):.1f}")
    mean_b = statistics.mean(sizes)
    print(f"bytes  mean={mean_b/1e6:.2f}MB  median={statistics.median(sizes)/1e6:.2f}MB")
    print(f"projected_size_for_10000 = {mean_b*10000/1e9:.1f} GB")
for b in bad[:15]:
    print("BAD:", b)
sys.exit(1 if bad else 0)
