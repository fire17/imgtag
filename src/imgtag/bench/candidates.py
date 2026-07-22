"""Candidate registry + preprocessing + ONNX embedding — PHASE 1, engine-independent.

OWNER: b-bench. Deliberately standalone: it owns its own ORT loaders so the candidate
matrix can run before `core/models.py` exists (wave-b-briefs launch order #2).

Preprocessing per candidate comes from the model's OWN config (ORACLE §4 parity playbook:
"the config file wins over folklore, always") — never a shared default.
"""
from __future__ import annotations

import os
import resource
import sys
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

ROOT = os.environ.get("IMGTAG_ROOT", os.path.expanduser("~/Creations/ImgTag"))
MODELS = os.path.join(ROOT, "models")
DATA = os.path.join(ROOT, "data")

BILINEAR, BICUBIC = Image.BILINEAR, Image.BICUBIC
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
HALF = (0.5, 0.5, 0.5)


@dataclass
class Candidate:
    """One ADR-4 roster entry. `vision[prec]` -> onnx path; missing key = not built yet."""

    id: str
    res: int
    dim: int
    vision: dict[str, str]
    text: dict[str, str] = field(default_factory=dict)
    mean: tuple = HALF
    std: tuple = HALF
    resample: int = BILINEAR
    squash: bool = True  # squash-resize (no shortest-side + center-crop)
    out_idx: int = 0  # which ONNX output is the embedding
    ctx: int = 32
    tok: str = "clip"  # tokenizer family -> which cached token npz to use
    license: str = "Apache-2.0"
    note: str = ""

    def path(self, prec: str) -> str | None:
        p = self.vision.get(prec)
        return p if p and os.path.exists(p) else None

    def available(self) -> list[str]:
        return [p for p in ("fp32", "int8") if self.path(p)]


def _m(name: str) -> str:
    return os.path.join(MODELS, name)


# ── ADR-4 bench roster ────────────────────────────────────────────────────────
# PE-Core: squash resize, BILINEAR, mean=std=0.5 (open_clip_config.json — spike-pecore §7.5).
# SigLIP2: 224, resample=2 = BILINEAR, mean=std=0.5 (preprocessor_config.json — spike-siglip2 §1).
# OpenCLIP ViT-B/32 openai: BICUBIC, CLIP constants, shortest-side + center crop.
CANDIDATES: dict[str, Candidate] = {
    c.id: c
    for c in [
        Candidate(
            id="pecore-s16-384", res=384, dim=512,
            vision={"fp32": _m("pecore-s16-384-vision.onnx"),
                    "int8": _m("pecore-s16-384-vision-int8-wo.onnx")},
            text={"fp32": _m("pecore-s16-384-text.onnx"),
                  "int8": _m("pecore-s16-384-text-int8.onnx")},
            note="primary candidate; self-exported (spike-pecore VERDICT exportable)",
        ),
        Candidate(
            id="pecore-b16-224", res=224, dim=1024,
            vision={"fp32": _m("pecore-b16-224-vision.onnx"),
                    "int8": _m("pecore-b16-224-vision-int8-wo.onnx")},
            text={"fp32": _m("pecore-b16-224-text.onnx")},
            note="resolution-dominates-params probe (spike-pecore §cross-check)",
        ),
        Candidate(
            id="pecore-t16-384", res=384, dim=512,
            vision={"fp32": _m("pecore-t16-384-vision.onnx"),
                    "int8": _m("pecore-t16-384-vision-int8-wo.onnx")},
            # text int8 = FULL-graph quant: 64MB vs the weight-only recipe's 140MB, and
            # the text tower is the SAFE one to quantize (spike-pecore: cos 0.988 vs torch,
            # where the vision tower managed only 0.94).
            text={"fp32": _m("pecore-t16-384-text.onnx"),
                  "int8": _m("pecore-t16-384-text-int8-full.onnx")},
            note="edge tier ~10M params",
        ),
        Candidate(
            id="siglip2-base-224", res=224, dim=768,
            vision={"fp32": _m("siglip2-base/vision_model.onnx"),
                    "int8": _m("siglip2-base/vision_model_int8.onnx"),
                    "int8wo": _m("siglip2-base/vision_model_int8_wo.onnx")},
            text={"int8": _m("siglip2-base/text_model_int8.onnx")},
            out_idx=1, ctx=64, tok="gemma",
            note="quality ANCHOR; official int8 must itself pass B24 (official != audited)",
        ),
        Candidate(
            id="siglip-base-224", res=224, dim=768,
            vision={"fp32": _m("siglip-base-224-vision.onnx"),
                    "int8": _m("siglip-base-224-vision-int8-wo.onnx")},
            text={"fp32": _m("siglip-base-224-text.onnx")},
            ctx=64, tok="siglip1",
            note="small text tower (ADR-4 target-profile candidate)",
        ),
        Candidate(
            id="openclip-vitb32", res=224, dim=512,
            vision={"fp32": _m("openclip-vitb32-vision.onnx"),
                    "int8": _m("openclip-vitb32-vision-int8-wo.onnx")},
            text={"fp32": _m("openclip-vitb32-text.onnx")},
            mean=CLIP_MEAN, std=CLIP_STD, resample=BICUBIC, squash=False, ctx=77,
            license="MIT", note="B17 CONTROL (openai weights) — never the default",
        ),
    ]
}


# ── preprocessing ─────────────────────────────────────────────────────────────
def preprocess(path: str, c: Candidate) -> np.ndarray:
    """Model-reference decode path (full decode). The fast draft() path is B16's job."""
    im = Image.open(path)
    im = im.convert("RGB")
    if c.squash:
        im = im.resize((c.res, c.res), c.resample)
    else:  # shortest-side resize + center crop (openai CLIP convention)
        w, h = im.size
        s = c.res / min(w, h)
        im = im.resize((max(c.res, round(w * s)), max(c.res, round(h * s))), c.resample)
        w, h = im.size
        l, t = (w - c.res) // 2, (h - c.res) // 2
        im = im.crop((l, t, l + c.res, t + c.res))
    a = np.asarray(im, np.float32) / 255.0
    a = (a - np.asarray(c.mean, np.float32)) / np.asarray(c.std, np.float32)
    return a.transpose(2, 0, 1)


def session(path: str, intra: int):
    import onnxruntime as ort

    ort.set_default_logger_severity(3)
    o = ort.SessionOptions()
    o.intra_op_num_threads = intra
    o.inter_op_num_threads = 1
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, o, providers=["CPUExecutionProvider"])


def l2(x: np.ndarray) -> np.ndarray:
    """ALWAYS normalize: PE-Core/SigLIP/MobileCLIP exports are UNNORMALIZED (spike §1)."""
    x = np.asarray(x, np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def embed_images(sess, paths: list[str], c: Candidate, batch: int = 8,
                 progress=None) -> np.ndarray:
    out = []
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        X = np.stack([preprocess(p, c) for p in chunk])
        out.append(sess.run(None, {"pixel_values": X})[c.out_idx].astype(np.float32))
        if progress and (i // batch) % 20 == 0:
            progress(i + len(chunk), len(paths))
    return l2(np.concatenate(out))


def embed_texts(sess, ids: np.ndarray, out_idx: int, batch: int = 64) -> np.ndarray:
    out = []
    for i in range(0, len(ids), batch):
        out.append(sess.run(None, {"input_ids": ids[i:i + batch]})[out_idx].astype(np.float32))
    return l2(np.concatenate(out))


# ── machine / noise protocol (BUDGETS bench-protocol header) ──────────────────
def peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e6 if sys.platform == "darwin" else r / 1e3  # macOS bytes, Linux KB


def usable_cores() -> tuple[int, str]:
    """ADR-11 effective cores: affinity -> cgroup v2 -> cgroup v1 -> cpu_count."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0)), "sched_getaffinity"
    for f, src in ((("/sys/fs/cgroup/cpu.max"), "cgroup-v2"),):
        try:
            q, p = open(f).read().split()
            if q != "max":
                return max(1, int(int(q) / int(p))), src
        except OSError:
            pass
    return os.cpu_count() or 1, "cpu_count"


def machine_header() -> dict:
    import platform

    import onnxruntime as ort

    cores, src = usable_cores()
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "usable_cores": cores,
        "usable_cores_source": src,
        "cpu_count": os.cpu_count(),
        "ort_version": ort.__version__,
        "ort_providers": ort.get_available_providers(),
        "numpy": np.__version__,
        "pillow": Image.__version__,
        "PROXY": platform.machine() != "x86_64",
    }


def load_ok(cores: int | None = None) -> tuple[bool, float]:
    """BUDGETS noise protocol: 1-min load must be <= usable_cores * 0.6."""
    cores = cores or usable_cores()[0]
    l1 = os.getloadavg()[0]
    return l1 <= cores * 0.6, l1


def wait_for_quiet(max_wait_s: int = 600, poll_s: int = 30) -> tuple[bool, float]:
    """Poll-wait up to max_wait_s for a quiet machine. Returns (quiet, load).

    Never fabricates a quiet-machine number: caller marks rows ADVISORY when False.
    """
    import time

    cores = usable_cores()[0]
    deadline = time.time() + max_wait_s
    ok, l1 = load_ok(cores)
    while not ok and time.time() < deadline:
        time.sleep(poll_s)
        ok, l1 = load_ok(cores)
    return ok, l1
