#!/usr/bin/env python3
"""CORPUS-D "poison" — ~120 hostile image files + manifest.json (B21 robustness corpus).

Built OFFLINE from copies of data/coco/val2017 (already on disk) + Pillow + macOS `sips`.
Idempotent: skips any file already present; rewrites the manifest every run.
Every generator is documented inline; each manifest entry carries its hostility class.

Classes (>=10 files each unless noted):
  truncated_jpeg   valid SOI/JFIF header, byte stream cut at 40% -> decoders EOF mid-scan
  corrupt_jpeg     intact header, 512 random bytes smashed into the middle of the scan
  zero_byte        0-byte files with image extensions
  wrong_extension  real PNG bytes named *.jpg (and JPEG bytes named *.png)
  cmyk_jpeg        4-channel CMYK JPEG (Pillow convert("CMYK")) — RGB assumptions die here
  png16            16-bit-per-channel PNG (I;16 / RGB;16 via numpy-free Pillow mode I)
  tiny_1px         1x1 pixel images (JPEG + PNG) — resize/crop math edge case
  huge_dims        20000x20000 solid-color PNG — ~700KB on disk, 400MP decoded (RAM bomb)
  exif_rotated     real photos with EXIF Orientation 3/6/8 written into the APP1 block
  heic             HEIC via `sips -s format heic` (macOS built-in)
  progressive_eof  progressive JPEG truncated mid-scan (partial-scan decode path)
  filename_hostile valid JPEGs whose NAMES are the attack: spaces, unicode, emoji,
                   leading dash, newline-ish chars, 300-char stem, no extension
  palette_alpha    mode-P PNG carrying a transparency index (palette + alpha)
  animated_gif     4-frame animated GIF (frame-0-only decoders vs iterators)
  decompression_bomb 17500x17500 (306MP) PNG — above Pillow's 178MP bomb guard
  symlink_valid    symlink to a real JPEG outside the corpus dir
  symlink_loop     a->b->a symlink cycle (ELOOP; walkers must not hang)
  perm_denied      valid JPEG chmod 000 (PermissionError on open)
(classes 13-17 added 2026-07-22 to track the widened BUDGETS.md CORPUS-D spec)
"""
import io, json, os, random, shutil, subprocess, sys
from PIL import Image

random.seed(1717)  # deterministic corpus

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "coco", "val2017")
OUT = os.path.join(ROOT, "data", "poison")
os.makedirs(OUT, exist_ok=True)

srcs = sorted(f for f in os.listdir(SRC) if f.endswith(".jpg"))[:200]
manifest = []


def rec(name, cls, note):
    p = os.path.join(OUT, name)
    try:                                   # lstat: symlink loops must not explode here
        size = os.lstat(p).st_size
    except OSError:
        size = 0
    manifest.append({"file": name, "class": cls, "bytes": size, "note": note})


def skip(name):
    p = os.path.join(OUT, name)
    return os.path.exists(p) and os.path.getsize(p) > 0


# 1. truncated_jpeg — header intact, stream cut at 40%
for i in range(12):
    n = f"truncated_{i:02d}.jpg"
    if not skip(n):
        b = open(os.path.join(SRC, srcs[i]), "rb").read()
        open(os.path.join(OUT, n), "wb").write(b[: int(len(b) * 0.4)])
    rec(n, "truncated_jpeg", "valid header, stream cut at 40% of file length")

# 2. corrupt_jpeg — 512 random bytes overwritten mid-scan
for i in range(12):
    n = f"corrupt_{i:02d}.jpg"
    if not skip(n):
        b = bytearray(open(os.path.join(SRC, srcs[20 + i]), "rb").read())
        mid = len(b) // 2
        b[mid: mid + 512] = bytes(random.getrandbits(8) for _ in range(512))
        open(os.path.join(OUT, n), "wb").write(bytes(b))
    rec(n, "corrupt_jpeg", "512 random bytes written over the middle of the entropy scan")

# 3. zero_byte
for i, ext in enumerate(["jpg", "png", "jpeg", "webp", "gif", "tif",
                         "jpg", "png", "jpeg", "bmp", "heic", "jpg"]):
    n = f"zerobyte_{i:02d}.{ext}"
    if not os.path.exists(os.path.join(OUT, n)):
        open(os.path.join(OUT, n), "wb").close()
    rec(n, "zero_byte", "0-byte file with an image extension")

# 4. wrong_extension — PNG bytes named .jpg (10), JPEG bytes named .png (2)
for i in range(10):
    n = f"pngbytes_as_jpeg_{i:02d}.jpg"
    if not skip(n):
        with Image.open(os.path.join(SRC, srcs[40 + i])) as im:
            im.convert("RGB").resize((320, 240)).save(os.path.join(OUT, n), "PNG")
    rec(n, "wrong_extension", "real PNG bytes with a .jpg filename")
for i in range(2):
    n = f"jpegbytes_as_png_{i:02d}.png"
    if not skip(n):
        with Image.open(os.path.join(SRC, srcs[50 + i])) as im:
            im.convert("RGB").save(os.path.join(OUT, n), "JPEG")
    rec(n, "wrong_extension", "real JPEG bytes with a .png filename")

# 5. cmyk_jpeg — 4-channel colorspace
for i in range(10):
    n = f"cmyk_{i:02d}.jpg"
    if not skip(n):
        with Image.open(os.path.join(SRC, srcs[60 + i])) as im:
            im.convert("CMYK").save(os.path.join(OUT, n), "JPEG")
    rec(n, "cmyk_jpeg", "4-channel CMYK JPEG (no RGB assumption survives)")

# 6. png16 — 16-bit per channel
for i in range(10):
    n = f"png16_{i:02d}.png"
    if not skip(n):
        with Image.open(os.path.join(SRC, srcs[70 + i])) as im:
            g = im.convert("L").resize((640, 480))
        im16 = g.point(lambda v: v * 257, "I")          # 8-bit -> 16-bit range
        im16.convert("I;16").save(os.path.join(OUT, n), "PNG")
    rec(n, "png16", "16-bit grayscale PNG (I;16) — bit-depth conversion path")

# 7. tiny_1px
for i in range(10):
    n = f"tiny1px_{i:02d}." + ("jpg" if i % 2 else "png")
    if not skip(n):
        Image.new("RGB", (1, 1), (i * 20 % 255, 30, 60)).save(os.path.join(OUT, n))
    rec(n, "tiny_1px", "1x1 pixel image — resize/aspect math edge case")

# 8. huge_dims — 20000x20000 solid color PNG, written as a STREAM (never allocating the
#    400MP raster ourselves): manual IHDR/IDAT/IEND with a row-by-row zlib compressor.
#    ~10s and <10MB RSS per file here, but ~1.2GB RSS for anything that decodes it.
def write_huge_png(path, w, h, rgb):
    import struct, zlib

    def chunk(fh, typ, data):
        fh.write(struct.pack(">I", len(data)) + typ + data +
                 struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF))

    row = b"\x00" + bytes(rgb) * w        # filter byte 0 + solid-color scanline
    with open(path, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n")
        chunk(fh, b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        co = zlib.compressobj(6)
        buf = b"".join(co.compress(row) for _ in range(h)) + co.flush()
        chunk(fh, b"IDAT", buf)
        chunk(fh, b"IEND", b"")


for i in range(10):
    n = f"hugedims_{i:02d}.png"
    if not skip(n):
        write_huge_png(os.path.join(OUT, n), 20000, 20000, (i * 25 % 255, 10, 200))
    rec(n, "huge_dims", "20000x20000 solid-color PNG — small on disk, 400MP decoded (RAM bomb)")

# 9. exif_rotated — real photos carrying EXIF Orientation 1..8 (BUDGETS: ALL eight)
for i in range(16):
    orient = (i % 8) + 1
    n = f"exifrot{orient}_{i:02d}.jpg"
    if not skip(n):
        with Image.open(os.path.join(SRC, srcs[80 + i])) as im:
            im = im.convert("RGB")
            exif = im.getexif()
            exif[274] = orient                      # 274 = Orientation
            im.save(os.path.join(OUT, n), "JPEG", exif=exif)
    rec(n, "exif_rotated", f"EXIF Orientation={orient} — display vs stored geometry mismatch")

# 10. heic — macOS sips conversion (no extra deps)
heic_made = 0
for i in range(10):
    n = f"heic_{i:02d}.heic"
    if not skip(n):
        r = subprocess.run(["sips", "-s", "format", "heic",
                            os.path.join(SRC, srcs[100 + i]),
                            "--out", os.path.join(OUT, n)],
                           capture_output=True)
        if r.returncode != 0:
            continue
    heic_made += 1
    rec(n, "heic", "HEIC container via macOS `sips -s format heic` (Pillow can't decode without pillow-heif)")

# 11. progressive_eof — progressive JPEG truncated mid-scan
for i in range(10):
    n = f"prog_eof_{i:02d}.jpg"
    if not skip(n):
        buf = io.BytesIO()
        with Image.open(os.path.join(SRC, srcs[120 + i])) as im:
            im.convert("RGB").save(buf, "JPEG", progressive=True, quality=90)
        b = buf.getvalue()
        open(os.path.join(OUT, n), "wb").write(b[: int(len(b) * 0.55)])
    rec(n, "progressive_eof", "progressive JPEG cut at 55% — EOF inside a later scan")

# 12. filename_hostile — the NAME is the attack; bytes are a valid small JPEG
hostile_names = [
    "file with spaces.jpg",
    "  leading and trailing  .jpg",
    "ünïcødé-ñämé-日本語-🔥.jpg",
    "-leading-dash.jpg",
    "double..dots..name.jpg",
    "semi;colon&amp$dollar.jpg",
    "quote'single\"double.jpg",
    "tab\tchar.jpg",
    "no_extension_at_all",
    "UPPER.JPG",
    # 300-char stem was the spec; APFS NAME_MAX is 255 BYTES (open() gives ENAMETOOLONG),
    # so the longest name this filesystem can physically hold is used instead.
    "a" * 251 + ".jpg",
    "percent%20encoded%2Fslash.jpg",
]
for i, hn in enumerate(hostile_names):
    if not skip(hn):
        with Image.open(os.path.join(SRC, srcs[140 + i])) as im:
            im.convert("RGB").resize((320, 240)).save(os.path.join(OUT, hn), "JPEG")
    rec(hn, "filename_hostile", "valid JPEG bytes; the FILENAME is hostile (spaces/unicode/300-char/no-ext)")

# 13. palette_alpha — mode "P" PNG with a transparency index (palette + alpha combo)
for i in range(6):
    n = f"palette_alpha_{i:02d}.png"
    if not skip(n):
        with Image.open(os.path.join(SRC, srcs[160 + i])) as im:
            p = im.convert("RGB").resize((400, 300)).convert(
                "P", palette=Image.ADAPTIVE, colors=64)
        p.info["transparency"] = 0
        p.save(os.path.join(OUT, n), "PNG", transparency=0)
    rec(n, "palette_alpha", "palette (mode P) PNG with a transparency index — palette+alpha path")

# 14. animated_gif — multi-frame GIF (frame-0-only decoders vs iterators)
for i in range(6):
    n = f"animated_{i:02d}.gif"
    if not skip(n):
        frames = []
        for k in range(4):
            with Image.open(os.path.join(SRC, srcs[166 + i])) as im:
                frames.append(im.convert("RGB").resize((240, 180)).rotate(k * 5)
                              .convert("P", palette=Image.ADAPTIVE, colors=64))
        frames[0].save(os.path.join(OUT, n), save_all=True,
                       append_images=frames[1:], duration=100, loop=0)
    rec(n, "animated_gif", "4-frame animated GIF — multi-frame container")

# 15. decompression_bomb — 306MP PNG, above Pillow's 178MP MAX_IMAGE_PIXELS guard
for i in range(6):
    n = f"bomb306mp_{i:02d}.png"
    if not skip(n):
        write_huge_png(os.path.join(OUT, n), 17500, 17500, (200, i * 30 % 255, 40))
    rec(n, "decompression_bomb", "17500x17500 (306MP) PNG — trips Pillow DecompressionBombError (>178MP)")

# 16. symlink_valid / symlink_loop — path-resolution hostility (not file content)
for i in range(4):
    n = f"symlink_valid_{i:02d}.jpg"
    p = os.path.join(OUT, n)
    if not os.path.lexists(p):
        os.symlink(os.path.join(SRC, srcs[172 + i]), p)
    rec(n, "symlink_valid", "symlink pointing at a real JPEG outside the corpus dir")
for i in range(4):
    a = os.path.join(OUT, f"symlink_loop_a{i:02d}.jpg")
    b = os.path.join(OUT, f"symlink_loop_b{i:02d}.jpg")
    if not os.path.lexists(a) and not os.path.lexists(b):
        os.symlink(b, a)          # a -> b
        os.symlink(a, b)          # b -> a  => ELOOP on open()
    rec(os.path.basename(a), "symlink_loop", "symlink cycle a->b->a — open() raises ELOOP, walkers must not hang")
    rec(os.path.basename(b), "symlink_loop", "symlink cycle b->a->b — open() raises ELOOP, walkers must not hang")

# 17. perm_denied — valid JPEG with mode 000 (unreadable by the indexing user)
for i in range(4):
    n = f"perm_denied_{i:02d}.jpg"
    p = os.path.join(OUT, n)
    if not os.path.exists(p):
        shutil.copyfile(os.path.join(SRC, srcs[180 + i]), p)
        os.chmod(p, 0o000)
    rec(n, "perm_denied", "valid JPEG, mode 000 — PermissionError on open, must be skipped+logged")

mpath = os.path.join(OUT, "manifest.json")
classes = {}
for m in manifest:
    classes[m["class"]] = classes.get(m["class"], 0) + 1
json.dump({
    "corpus": "CORPUS-D",
    "name": "poison",
    "purpose": "B21 robustness gate — 0 crashes, 0 hangs, every failure skipped+logged",
    "source": "copies/derivatives of data/coco/val2017 (Apache-friendly COCO images), local only",
    "count": len(manifest),
    "classes": classes,
    "files": manifest,
}, open(mpath, "w"), indent=1)

print(f"CORPUS-D built: {len(manifest)} files in {OUT}")
for c, k in sorted(classes.items()):
    print(f"  {c:18s} {k}")
if heic_made == 0:
    print("WARN: no HEIC produced (sips failed)", file=sys.stderr)
