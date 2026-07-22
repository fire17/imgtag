"""Backend contract tests: L2-normalization ALWAYS, tokenizer parity, preprocess law."""

import io

import numpy as np
import pytest
from PIL import Image

from imgtag.core import models

HAS_PECORE = models.find_artifact(models.registry()["pecore-s16-384"], "pecore-s16-384-vision.onnx") is not None
needs_model = pytest.mark.skipif(not HAS_PECORE, reason="pecore-s16-384 artifacts not on this host")


def test_tokenizer_matches_reference_clip_ids():
    """Known-good CLIP BPE ids — the compact bundled binary must reproduce them exactly."""
    tok = models.ClipBPE(32)
    ids = tok.encode(["a photo of a cat", "a photo of a dog"])
    assert ids.shape == (2, 32) and ids.dtype == np.int64
    assert list(ids[0, :7]) == [49406, 320, 1125, 539, 320, 2368, 49407]
    assert list(ids[1, :7]) == [49406, 320, 1125, 539, 320, 1929, 49407]
    assert (ids[0, 7:] == 0).all()


def test_tokenizer_truncates_to_context_and_keeps_eot():
    tok = models.ClipBPE(8)
    ids = tok.encode(["a very long sentence with far too many words to fit in eight slots"])
    assert ids.shape == (1, 8) and ids[0, -1] == tok.eot and ids[0, 0] == tok.sot


def test_unknown_backend_is_loud():
    with pytest.raises(models.ModelUnavailableError):
        models.load_backend("no-such-model")


@needs_model
def test_embeddings_are_l2_normalized():
    be = models.load_backend("pecore-s16-384", {"precision": "fp32", "intra_op": 2})
    rng = np.random.default_rng(0)
    batch = rng.integers(0, 255, (3, be.size, be.size, 3), dtype=np.uint8)
    v = be.embed_images(batch)
    assert v.shape == (3, be.dim) and v.dtype == np.float32
    assert 0.999 <= float(np.linalg.norm(v, axis=1).mean()) <= 1.001
    t = be.embed_texts(["a photo of a bear", "a red car"])
    assert t.shape == (2, be.dim)
    assert 0.999 <= float(np.linalg.norm(t, axis=1).mean()) <= 1.001
    be.release_text()
    assert be._ts is None


@needs_model
def test_release_text_then_reload():
    be = models.load_backend("pecore-s16-384", {"precision": "fp32", "intra_op": 1})
    a = be.embed_texts(["a dog"])
    be.release_text()
    b = be.embed_texts(["a dog"])
    np.testing.assert_allclose(a, b, atol=1e-6)


def _jpeg(size, orientation=None):
    im = Image.new("RGB", size)
    im.paste((255, 0, 0), (0, 0, size[0] // 2, size[1]))  # left half red
    buf = io.BytesIO()
    if orientation:
        exif = Image.Exif()
        exif[274] = orientation
        im.save(buf, "JPEG", exif=exif)
    else:
        im.save(buf, "JPEG")
    buf.seek(0)
    return buf


def test_preprocess_shape_and_dtype():
    spec = models.registry()["pecore-s16-384"]
    be = models.ModelBackend.__new__(models.ModelBackend)  # preprocess needs no session
    be.size, be.squash, be.resample = spec["size"], True, Image.Resampling.BILINEAR
    a = be.preprocess(Image.open(_jpeg((800, 600))))
    assert a.shape == (384, 384, 3) and a.dtype == np.uint8


def test_preprocess_applies_exif_orientation():
    spec = models.registry()["pecore-s16-384"]
    be = models.ModelBackend.__new__(models.ModelBackend)
    be.size, be.squash, be.resample = spec["size"], True, Image.Resampling.BILINEAR
    plain = be.preprocess(Image.open(_jpeg((800, 600))))
    rot = be.preprocess(Image.open(_jpeg((800, 600), orientation=6)))  # 90° CW
    # left-half-red becomes top-half-red once the orientation tag is honoured
    assert plain[:, :100].mean() > plain[:, -100:].mean()
    assert rot[:100].mean() > rot[-100:].mean()
