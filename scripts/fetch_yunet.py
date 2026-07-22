#!/usr/bin/env python3
"""Fetch + integrity-verify the YuNet face detector for track-people.

The .onnx is gitignored (models/**/*.onnx), so this is how a fresh checkout gets it. The
sha256 is the committed integrity anchor — a mismatch REFUSES rather than ships an
unaudited model (ORACLE §7c / B24 fidelity discipline).

    .venv/bin/python scripts/fetch_yunet.py
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
       "face_detection_yunet/face_detection_yunet_2023mar.onnx")
SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
DEST = Path(__file__).resolve().parent.parent / "models/moderation/face-yunet-640.onnx"


def main() -> int:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.is_file() and hashlib.sha256(DEST.read_bytes()).hexdigest() == SHA256:
        print(f"already present and verified: {DEST}")
        return 0
    print(f"downloading {URL}")
    blob = urllib.request.urlopen(URL, timeout=60).read()  # noqa: S310 (trusted GitHub URL)
    got = hashlib.sha256(blob).hexdigest()
    if got != SHA256:
        print(f"REFUSED: sha256 {got} != expected {SHA256}", file=sys.stderr)
        return 1
    DEST.write_bytes(blob)
    (DEST.parent / f"{DEST.name}.sha256").write_text(f"{SHA256}  {DEST.name}\n")
    print(f"verified and wrote {DEST} ({len(blob)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
