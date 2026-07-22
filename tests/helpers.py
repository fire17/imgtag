"""Tiny index builder for b-daemon's tests + live verification.

NOT the product indexer (that is b-engine's core/indexer.py + cli.py). This exists so the
search/daemon lane can be verified against a REAL on-disk dataset built through the real
store.Writer + models.ModelBackend contracts, with the real xxhash64 image ids (IA.md).
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import xxhash
from PIL import Image, ImageOps

from imgtag.core import models as _models
from imgtag.core.store import Writer


def build_index(dataset: str, files, home: Path | None = None, backend=None, batch: int = 4, log=None):
    """Index ``files`` into ``dataset`` under ``home``. Returns (backend, n_rows)."""
    be = backend or _models.load_backend(_models.DEFAULT_BACKEND)
    files = [Path(f) for f in files]
    n = 0
    with Writer(dataset, be, home) as w:
        for i in range(0, len(files), batch):
            chunk = files[i : i + batch]
            arrs, recs = [], []
            for p in chunk:
                buf = p.read_bytes()  # read ONCE: hash and decode from the same buffer
                im = Image.open(io.BytesIO(buf))
                im = ImageOps.exif_transpose(im).convert("RGB")
                arrs.append(be.preprocess(im))
                recs.append(
                    {
                        "image_id": xxhash.xxh64(buf).hexdigest(),
                        "path": str(p.resolve()),
                        "dataset": dataset,
                        "w": im.width,
                        "h": im.height,
                    }
                )
            w.append(be.embed_images(np.stack(arrs)), recs)
            n += len(recs)
            if log:
                log(f"{n}/{len(files)}")
    return be, n
