"""Model backend registry — ADR-4/ADR-7; contracts in .deify/wave-b-briefs.md.

OWNER: b-engine. Embeddings returned by any backend are ALWAYS L2-normalized f32
(PE-Core/MobileCLIP exports are UNNORMALIZED — norms 5–21, spike-pecore.md §finding 1).

Preprocess config is DATA (``data/backends.json``, copied from each model's own
config file) — never folklore. Text towers load lazily and are releasable: the
PE-Core fp32 text tower alone is an 850MB RSS bomb, the int8 one 154MB at cos 0.988.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

DATA = Path(__file__).resolve().parent.parent / "data"


class ModelUnavailableError(RuntimeError):
    """Model artifact missing and not downloadable (CLI exit 7)."""


@lru_cache(maxsize=1)
def registry() -> dict:
    return {k: v for k, v in json.loads((DATA / "backends.json").read_bytes()).items() if not k.startswith("_")}


def model_roots() -> list[Path]:
    """Where model artifacts may live, most specific first."""
    roots = []
    if os.environ.get("IMGTAG_MODELS_DIR"):
        roots.append(Path(os.environ["IMGTAG_MODELS_DIR"]))
    from .store import imgtag_home

    roots.append(imgtag_home() / "models")
    roots.append(Path(__file__).resolve().parents[3] / "models")  # dev checkout
    return roots


def find_artifact(spec: dict, filename: str) -> Path | None:
    sub = spec.get("subdir")
    for root in model_roots():
        for p in ([root / sub / filename] if sub else []) + [root / filename]:
            if p.is_file():
                return p
    return None


def file_sha256(p: Path) -> str:
    """sha256 of an artifact, memoised in a sidecar (hashing 100MB per load is waste)."""
    side = p.with_suffix(p.suffix + ".sha256")
    st = p.stat()
    stamp = f"{st.st_size}:{st.st_mtime_ns}"
    try:
        sha, got = side.read_text().split()
        if got == stamp:
            return sha
    except (OSError, ValueError):
        pass
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    sha = h.hexdigest()
    try:
        side.write_text(f"{sha} {stamp}")
    except OSError:
        pass
    return sha


# ---------------------------------------------------------------- tokenizer


@lru_cache(maxsize=4)
def _load_bpe(path: str) -> dict:
    """Compact tokenizer binary -> numpy arrays ONLY.

    The measured cost of the dict-of-str shape is 0.61s and 551MB resident for a 34MB
    tokenizer.json (16× the file), so the whole merge loop runs on INTEGERS: pair lookup
    is a searchsorted over packed (left<<32|right) int64 keys, and each merge row carries
    the id of its result. The only Python objects built are the ≤~600 short "seed" tokens
    (single characters and char+"</w>") needed to turn a word into its starting ids.
    """
    z = np.load(path)
    blob, off = z["vocab_blob"], z["vocab_off"]
    lens = np.diff(off)
    seed: dict[str, int] = {}
    for i in np.flatnonzero(lens <= 8):  # short tokens only; never the whole vocabulary
        try:
            seed.setdefault(blob[off[i]:off[i + 1]].tobytes().decode(), int(i))
        except UnicodeDecodeError:
            pass
    return {"key": z["merge_key"], "rank": z["merges"][:, 0], "new": z["merges"][:, 1],
            "seed": seed, "meta": json.loads(z["meta"].tobytes().decode()), "n": len(off) - 1}


# CLIP's word pattern, expressed in the stdlib `re` dialect (no `regex` dep):
# \p{L}+ -> [^\W\d_]+ , \p{N} -> \d , punctuation runs, and bare underscores.
_WORD_RE = re.compile(r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[^\W\d_]+|\d|[^\s\w]+|_+", re.I)


class ClipBPE:
    """CLIP byte-pair tokenizer over the compact bundled binary (PE-Core, OpenCLIP)."""

    def __init__(self, ctx: int, path: Path | None = None):
        t = _load_bpe(str(path or DATA / "clip-bpe.npz"))
        self.key, self.rank, self.new, self.seed = t["key"], t["rank"], t["new"], t["seed"]
        self.ctx = ctx
        self.sot, self.eot = t["n"] - 2, t["n"] - 1  # <|startoftext|>, <|endoftext|>
        bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
        cs, n = bs[:], 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        self.byte_enc = {b: chr(c) for b, c in zip(bs, cs)}
        self._cache: dict[str, list[int]] = {}

    def _pair_rank(self, a: int, b: int) -> tuple[int, int]:
        """(rank, merged_id) for an adjacent id pair — one searchsorted, no dicts."""
        i = int(np.searchsorted(self.key, (a << 32) | b))
        if i < len(self.key) and self.key[i] == (a << 32) | b:
            return int(self.rank[i]), int(self.new[i])
        return 1 << 30, -1

    def _bpe(self, word: str) -> list[int]:
        if word in self._cache:
            return self._cache[word]
        ids = [self.seed.get(c, -1) for c in word[:-1]] + [self.seed.get(word[-1] + "</w>", -1)]
        ids = [i for i in ids if i >= 0]
        while len(ids) > 1:
            best, pos, merged = 1 << 30, -1, -1
            for i in range(len(ids) - 1):
                r, nid = self._pair_rank(ids[i], ids[i + 1])
                if r < best:
                    best, pos, merged = r, i, nid
            if pos < 0:
                break
            ids[pos : pos + 2] = [merged]
        self._cache[word] = ids
        return ids

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.ctx), np.int64)
        for r, text in enumerate(texts):
            toks = [self.sot]
            clean = " ".join(text.lower().strip().split())
            for word in _WORD_RE.findall(clean):
                toks += self._bpe("".join(self.byte_enc[b] for b in word.encode()))
            toks = toks[: self.ctx - 1] + [self.eot]
            out[r, : len(toks)] = toks
        return out


class HFBPE(ClipBPE):
    """HF-BPE tokenizer (SigLIP2's Gemma vocab) over a per-model compact binary.

    Differences from CLIP's: the normalizer replaces " " with "▁" and there is no
    end-of-word suffix, so a word's seed ids are its bare characters; unknown characters
    fall back to their <0xXX> byte tokens. Same integer-only merge loop.
    """

    def __init__(self, ctx: int, path: Path):
        t = _load_bpe(str(path))
        self.key, self.rank, self.new, self.seed = t["key"], t["rank"], t["new"], t["seed"]
        self.ctx, self.meta, self._cache = ctx, t["meta"], {}
        self.replace = t["meta"].get("replace") or [" ", "▁"]
        self.eot = self.seed.get("<eos>", 1)
        self.byte_fallback = bool(t["meta"].get("byte_fallback"))

    def _seed_ids(self, word: str) -> list[int]:
        out = []
        for ch in word:
            i = self.seed.get(ch, -1)
            if i < 0 and self.byte_fallback:
                out += [self.seed[f"<0x{b:02X}>"] for b in ch.encode() if f"<0x{b:02X}>" in self.seed]
            elif i >= 0:
                out.append(i)
        return out

    def _bpe(self, word: str) -> list[int]:
        if word in self._cache:
            return self._cache[word]
        ids = self._seed_ids(word)
        while len(ids) > 1:
            best, pos, merged = 1 << 30, -1, -1
            for i in range(len(ids) - 1):
                r, nid = self._pair_rank(ids[i], ids[i + 1])
                if r < best:
                    best, pos, merged = r, i, nid
            if pos < 0:
                break
            ids[pos : pos + 2] = [merged]
        self._cache[word] = ids
        return ids

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.ctx), np.int64)  # pad id 0
        for r, text in enumerate(texts):
            body = self._bpe(text.replace(*self.replace))[: self.ctx - 1]
            toks = body + [self.eot]  # add_eos_token: true, add_bos_token: false
            out[r, : len(toks)] = toks
        return out


def _tokenizer(spec: dict):
    kind = spec["tokenizer"]
    if kind == "clip-bpe":
        return ClipBPE(spec["ctx"])
    path = find_artifact(spec, spec.get("tokenizer_file", "tokenizer.npz"))
    if path is None:
        raise ModelUnavailableError(
            f"tokenizer binary for {kind!r} not found — build it offline: "
            f"`uv run python scripts/build_tokenizer.py hf <tokenizer.json> <model-dir>/tokenizer.npz` "
            f"(runtime never parses tokenizer.json: measured 0.61s and 551MB resident)"
        )
    return HFBPE(spec["ctx"], path)


# ---------------------------------------------------------------- backend


def _session(path: Path, intra: int, graph_opt: str = "ENABLE_ALL") -> ort.InferenceSession:
    o = ort.SessionOptions()
    o.intra_op_num_threads = int(intra)
    o.inter_op_num_threads = 1
    o.add_session_config_entry("session.use_env_allocators", "0")
    o.enable_cpu_mem_arena = True
    # per-model, per-FILE optimisation level: ENABLE_ALL crashes SigLIP2's fp16 export
    # (ORT 1.27 fusion bug), so that file is pinned to ENABLE_EXTENDED in backends.json
    o.graph_optimization_level = getattr(ort.GraphOptimizationLevel, f"ORT_{graph_opt}")
    return ort.InferenceSession(str(path), o, providers=["CPUExecutionProvider"])


def preprocess_image(im: Image.Image, size: int, squash: bool = True, resample=Image.Resampling.BILINEAR) -> np.ndarray:
    """PIL image -> uint8 [size,size,3]. ``draft()`` decodes JPEGs at 1/2..1/8 scale in
    the DCT domain (measured 1.7–2.1×, runtime.md §4.2); EXIF orientation is applied
    BEFORE the resize. Module-level so decode workers need no ORT session."""
    try:
        im.draft("RGB", (size, size))  # no-op for non-JPEG
    except (AttributeError, ValueError):
        pass
    im = ImageOps.exif_transpose(im)
    if im.mode != "RGB":
        im = im.convert("RGB")
    if squash:
        return np.asarray(im.resize((size, size), resample), np.uint8)
    w, h = im.size
    s = size / min(w, h)
    im = im.resize((max(size, round(w * s)), max(size, round(h * s))), resample)
    w, h = im.size
    l, t = (w - size) // 2, (h - size) // 2
    return np.asarray(im.crop((l, t, l + size, t + size)), np.uint8)


def _embed_output(sess) -> str:
    """The pooled embedding, not the token grid: HF exports put ``last_hidden_state``
    first, so taking output[0] silently hands back [n, tokens, D] (SigLIP2: [n,196,768])."""
    outs = sess.get_outputs()
    for want in ("image_embeds", "text_embeds", "pooler_output", "sentence_embedding", "embeddings"):
        for o in outs:
            if o.name == want:
                return o.name
    for o in outs:  # otherwise the first 2-D output
        if len(o.shape) == 2:
            return o.name
    return outs[0].name


def _l2(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, np.float32)
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    return a / np.maximum(n, 1e-12)


class ModelBackend:
    """One model, two towers. Vision session is eager, text session lazy + releasable."""

    def __init__(self, name: str, spec: dict, profile: dict, vision: bool = True):
        self.name = name
        self.spec = spec
        self.profile = profile
        self.dim = int(spec["dim"])
        self.size = int(spec["size"])
        self.mean = np.asarray(spec["mean"], np.float32) * 255.0
        self.std = np.asarray(spec["std"], np.float32) * 255.0
        self.resample = getattr(Image.Resampling, spec["resample"])
        self.squash = spec["resize_mode"] == "squash"
        self.precision = profile.get("precision", "fp32")
        gated = spec.get("vision_gated") or {}
        if self.precision in gated and not profile.get("allow_gated"):
            # RULING 2026-07-22: a downloaded/naive int8 VISION file is never a usable
            # variant — two official int8 exports were measurably broken (SigLIP2 cos
            # 0.78, Xenova cos 0.008). Only b-bench, via B24 tier-2, may promote one.
            raise ModelUnavailableError(
                f"{name}: {self.precision} vision is GATED — {gated.get('reason')}. "
                f"Available now: {', '.join(spec['vision']) or 'none'}; b-bench may enable it "
                f"with profile allow_gated after B24 passes on this arch."
            )
        vision_files = {**spec["vision"], **({k: v for k, v in gated.items() if k != "reason"}
                                             if profile.get("allow_gated") else {})}
        if not vision_files:
            raise ModelUnavailableError(f"{name}: no ungated vision artifact (gated: {list(gated)})")
        # a model may pin its own default weight file (SigLIP2: the fp16 export, which is
        # bit-equivalent to fp32 at half the bytes); an EXPLICIT --precision still wins
        if not profile.get("precision_explicit") and spec.get("default_precision") in vision_files:
            self.precision = spec["default_precision"]
        if self.precision not in vision_files:
            self.precision = next(iter(vision_files))
        vfile = vision_files[self.precision]
        path = find_artifact(spec, vfile)
        if path is None:
            raise ModelUnavailableError(
                f"{name}: vision artifact {vfile!r} not found in {[str(r) for r in model_roots()]} "
                f"— run `imgtag doctor --fetch {name}` (needs network) or place the file manually"
            )
        self.vision_path = path
        self.model_sha = file_sha256(path)
        self.model_id = f"{name}-{self.precision}"
        self._vs = self._vin = self._fixed_batch = None
        if vision:  # geometry=worker: the coordinator wants the identity, not the session
            self._vs = _session(path, profile.get("intra_op", 2), self._graph_opt(vfile))
            i = self._vs.get_inputs()[0]
            self._vin = i.name
            self._vout = _embed_output(self._vs)
            self._fixed_batch = i.shape[0] if isinstance(i.shape[0], int) else None
            out = next(o for o in self._vs.get_outputs() if o.name == self._vout)
            if len(out.shape) != 2:  # never index a token grid as if it were an embedding
                raise ModelUnavailableError(
                    f"{name}: {path.name} exposes no pooled embedding output "
                    f"(best was {out.name} with shape {out.shape}) — export or fetch a graph "
                    f"with image_embeds/pooler_output"
                )
            if isinstance(out.shape[-1], int) and out.shape[-1] != self.dim:  # graph beats config
                self.dim = out.shape[-1]
        self._ts = None
        self._tok = None

    def _graph_opt(self, filename: str) -> str:
        return (self.spec.get("graph_opt") or {}).get(filename, "ENABLE_ALL")

    # -- preprocess ------------------------------------------------
    def preprocess(self, im: Image.Image) -> np.ndarray:
        """PIL image -> uint8 [H,W,3] per this model's own config."""
        return preprocess_image(im, self.size, self.squash, self.resample)

    # -- towers ----------------------------------------------------
    def embed_images(self, batch: np.ndarray) -> np.ndarray:
        """uint8 [n,H,W,3] -> f32 [n,D], L2-NORMALIZED."""
        x = np.asarray(batch, np.uint8).astype(np.float32)
        x = (x - self.mean) / self.std
        x = np.ascontiguousarray(x.transpose(0, 3, 1, 2))
        n = x.shape[0]
        if self._fixed_batch and n != self._fixed_batch:  # onnx-community exports pin batch
            B = self._fixed_batch
            pad = (-n) % B
            x = np.concatenate([x, np.repeat(x[-1:], pad, 0)]) if pad else x
            outs = [self._vs.run([self._vout], {self._vin: x[i : i + B]})[0] for i in range(0, len(x), B)]
            return _l2(np.concatenate(outs)[:n])
        return _l2(self._vs.run([self._vout], {self._vin: x})[0])

    def _text_session(self):
        if self._ts is None:
            prec = self.spec.get("text_precision", "int8")
            files = self.spec.get("text") or {}
            fname = files.get(prec) or (next(iter(files.values())) if files else None)
            if fname is None:
                raise ModelUnavailableError(f"{self.name}: no text tower artifact configured")
            path = find_artifact(self.spec, fname)
            if path is None:
                raise ModelUnavailableError(f"{self.name}: text artifact {fname!r} not found")
            self._ts = _session(path, self.profile.get("text_intra_op", 2), self._graph_opt(fname))
            self._tin = self._ts.get_inputs()[0].name
            self._tout = _embed_output(self._ts)
            self._tok = _tokenizer(self.spec)
        return self._ts

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """list[str] -> f32 [n,D], L2-NORMALIZED. Lazy-loads the text tower."""
        s = self._text_session()
        ids = self._tok.encode(list(texts))
        return _l2(s.run([self._tout], {self._tin: ids})[0])

    def release_text(self) -> None:
        """Drop the text tower (ADR-5 --text-ttl on the 8GB profile)."""
        self._ts = None
        self._tok = None

    def __repr__(self) -> str:
        return f"<ModelBackend {self.model_id} dim={self.dim} sha={self.model_sha[:12]}>"


def load_backend(name: str, profile: dict | None = None, vision: bool = True) -> ModelBackend:
    reg = registry()
    if name not in reg:
        raise ModelUnavailableError(f"unknown backend {name!r}; known: {', '.join(reg)}")
    if profile is None:
        from .doctor import load_profile

        profile = load_profile()
    return ModelBackend(name, reg[name], profile, vision)


DEFAULT_BACKEND = "pecore-s16-384"


# ---------------------------------------------------------------- downloader


def fetch(name: str, log=print) -> list[Path]:
    """Ranged-resume download of a backend's artifacts into ~/.imgtag/models/<id>/.

    The ONLY permitted egress (B22): announced, logged with exact URLs, and never
    repeated once cached. sha256 is verified against the recorded manifest; the first
    download records it (trust-on-first-use) so later fetches are hard-checked.
    """
    import httpx

    from .store import imgtag_home

    spec = registry()[name]
    urls = spec.get("urls") or {}
    if not urls:
        raise ModelUnavailableError(f"{name}: no download URLs configured (artifacts are local-only)")
    out = imgtag_home() / "models" / (spec.get("subdir") or name)
    out.mkdir(parents=True, exist_ok=True)
    shas_p = out / "sha256.json"
    shas = json.loads(shas_p.read_bytes()) if shas_p.is_file() else {}
    got = []
    with httpx.Client(follow_redirects=True, timeout=120) as c:
        for fname, url in urls.items():
            dest = out / fname
            if dest.is_file() and shas.get(fname) == file_sha256(dest):
                log(f"cached {fname}")
                got.append(dest)
                continue
            part = dest.with_suffix(dest.suffix + ".part")
            have = part.stat().st_size if part.is_file() else 0
            log(f"downloading {url} (resume @{have})")
            headers = {"Range": f"bytes={have}-"} if have else {}
            with c.stream("GET", url, headers=headers) as r:
                if r.status_code == 416:
                    r.close()
                elif r.status_code in (200, 206):
                    mode = "ab" if r.status_code == 206 and have else "wb"
                    with open(part, mode) as f:
                        for chunk in r.iter_bytes(1 << 20):
                            f.write(chunk)
                else:
                    raise ModelUnavailableError(f"{url}: HTTP {r.status_code}")
            os.replace(part, dest)
            sha = file_sha256(dest)
            if fname in shas and shas[fname] != sha:
                dest.unlink()
                raise ModelUnavailableError(f"{fname}: sha256 mismatch (expected {shas[fname]}, got {sha})")
            shas[fname] = sha
            got.append(dest)
    shas_p.write_text(json.dumps(shas, indent=1))
    return got
