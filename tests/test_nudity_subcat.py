"""Sub-category head (NudeNet) — contract, taxonomy and artifact tests.

The 12MB nudenet-320n.onnx is gitignored, so artifact tests skip when it is absent
(fetch: curl -sSL -o models/moderation/nudenet-320n.onnx \\
    https://huggingface.co/deepghs/nudenet_onnx/resolve/main/320n.onnx).
The taxonomy/tier-mapping tests need no artifact and always run.
"""

import numpy as np
import pytest
from PIL import Image

from imgtag.moderation import nudity_subcat as sc

HAS = sc.find_artifact(sc.SPEC, sc.ARTIFACT) is not None
needs = pytest.mark.skipif(not HAS, reason="nudenet-320n.onnx not on this host")


# ---------------------------------------------------------------- taxonomy (no artifact)


def test_eighteen_classes_in_the_models_own_order():
    assert len(sc.CLASSES) == 18
    assert sc.CLASSES[0] == "FEMALE_GENITALIA_COVERED"   # index 0, from the README table
    assert sc.CLASSES[3] == "FEMALE_BREAST_EXPOSED"
    assert sc.CLASSES[14] == "MALE_GENITALIA_EXPOSED"


def test_tier_mapping_is_adr14():
    for c in ("FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED", "FEMALE_BREAST_EXPOSED",
              "BUTTOCKS_EXPOSED", "ANUS_EXPOSED"):
        assert sc.CLASS_TIER[c] == "violation"
    for c in ("FEMALE_BREAST_COVERED", "BUTTOCKS_COVERED", "FEMALE_GENITALIA_COVERED"):
        assert sc.CLASS_TIER[c] == "review"
    # a face or a foot is not a moderation event
    for c in ("FACE_FEMALE", "FACE_MALE", "FEET_EXPOSED", "FEET_COVERED"):
        assert sc.CLASS_TIER.get(c, "none") == "none"
    assert set(sc.CLASS_TIER) <= set(sc.CLASSES)          # no phantom classes


def test_nms_keeps_the_best_and_drops_the_overlap():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], np.float32)
    scores = np.array([0.9, 0.8, 0.7], np.float32)
    keep = sc._nms(boxes, scores, 0.45)
    assert keep[0] == 0 and 2 in keep and 1 not in keep   # box1 suppressed by box0


def test_letterbox_is_square_aspect_preserving_and_normalised():
    a = sc._letterbox(Image.new("RGB", (640, 320), (200, 100, 50)))
    assert a.shape == (3, 320, 320) and a.dtype == np.float32
    assert 0.0 <= a.min() and a.max() <= 1.0
    assert a[0, 200, 100] == pytest.approx(114 / 255, abs=1e-3)  # padded region is grey 114


# ---------------------------------------------------------------- the real detector


@needs
def test_content_free_images_produce_no_detections():
    """NudeNet's own answer to the colour-prior that fools the binary head: a detector
    finds no body parts in a solid colour, so it needs no structure guard of its own."""
    h = sc.load_subcategory_head({"intra_op": 1})
    for c in ((0, 0, 0), (222, 180, 150), (255, 255, 255)):
        assert h.detect(Image.new("RGB", (320, 320), c)) == []


@needs
def test_label_payload_shape_and_tier_ordering():
    h = sc.load_subcategory_head({"intra_op": 1})
    lab = h.label(Image.new("RGB", (256, 256), (10, 20, 30)))
    assert lab["category"] == "nudity" and lab["model_id"] == "nudenet-320n"
    assert lab["tier"] in ("violation", "review", "none")
    assert isinstance(lab["subcategories"], list) and isinstance(lab["detections"], list)


@needs
def test_every_detection_carries_class_tier_p_box():
    h = sc.load_subcategory_head({"intra_op": 1})
    rng = np.random.default_rng(3)
    im = Image.fromarray(rng.integers(0, 256, (240, 320, 3), dtype=np.uint8))
    for d in h.detect(im):
        assert d["class"] in sc.CLASSES
        assert d["tier"] in ("violation", "review", "none")
        assert 0.0 <= d["p"] <= 1.0 and len(d["box"]) == 4
