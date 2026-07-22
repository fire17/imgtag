#!/usr/bin/env python3
"""Fetch a SMALL lawful, NON-EXPLICIT nudity boundary proxy from the Met Open Access API.

Everything here is **CC0 public-domain museum art** — classical paintings and marble
sculpture. It is NOT photographic pornography and does NOT go near the explicit-adult-
content the EVAL DATA LAW forbids on this machine. Purpose: a first *positive-side* signal
on the nudity boundary (depicted human nudity) and the ADR-14 statue no-flag negative,
without downloading anything unlawful or unsafe. `data/` is gitignored — this makes the set
reproducible instead of a mystery folder.

    python3 scripts/fetch_art_proxy.py

Honest limits (recorded in research/track-nudity.md §10): painted/sculpted nudity is a
DOMAIN SHIFT from photographs, so recall measured here UNDER-estimates real-photo recall —
it is a proxy, labelled as one, never a substitute for the escalated photographic set.
"""

import json
import os
import ssl
import time
import urllib.parse
import urllib.request
from pathlib import Path

import certifi

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://collectionapi.metmuseum.org/public/collection/v1"
CTX = ssl.create_default_context(cafile=certifi.where())
# (query, output subdir, cap). Adult classical nudes + nude statuary only.
JOBS = [("female nude", "nude", 60), ("male nude", "nude", 40), ("nude marble statue", "statue", 40)]


def _get(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "imgtag-research/1.0"})
    return json.load(urllib.request.urlopen(req, timeout=25, context=CTX))


def fetch(query: str, sub: str, cap: int) -> int:
    out = ROOT / "data" / "artproxy" / sub
    out.mkdir(parents=True, exist_ok=True)
    ids = _get(f"{BASE}/search?q={urllib.parse.quote(query)}&hasImages=true").get("objectIDs") or []
    man, got = [], 0
    for oid in ids:
        if got >= cap:
            break
        try:
            o = _get(f"{BASE}/objects/{oid}")
        except Exception:
            continue
        if not o.get("isPublicDomain"):          # CC0 only
            continue
        url = o.get("primaryImageSmall") or o.get("primaryImage")
        if not url:
            continue
        dst = out / f"{oid}.jpg"
        if not dst.exists():
            try:
                data = urllib.request.urlopen(
                    urllib.request.Request(url, headers={"User-Agent": "imgtag-research/1.0"}),
                    timeout=30, context=CTX).read()
                dst.write_bytes(data)
            except Exception:
                continue
        man.append({"id": oid, "title": o.get("title"), "classification": o.get("classification"),
                    "medium": o.get("medium"), "objectDate": o.get("objectDate")})
        got += 1
        time.sleep(0.2)                          # polite
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    print(f"{query!r} -> {got} images in {out}")
    return got


def main() -> int:
    for q, sub, cap in JOBS:
        fetch(q, sub, cap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
