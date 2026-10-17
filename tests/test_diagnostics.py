"""
Regression tests for the ML diagnostic layer: preprocessing, output
checks, crop selection and quadrant handling. Fake interpreters only; no
model files or network. Gemini tests live in test_gemini.py.
"""

import numpy as np
import pytest
from PIL import Image

import agri_core as core


# ---- fake TFLite interpreter ----

class FakeInterpreter:
    def __init__(self, outputs, input_dtype=np.float32, input_quant=(0.0, 0),
                 output_dtype=np.float32, output_quant=(0.0, 0)):
        self.outputs = np.asarray([outputs], dtype=output_dtype)
        self.input_dtype = input_dtype
        self.input_quant = input_quant
        self.output_dtype = output_dtype
        self.output_quant = output_quant
        self.inputs = []

    def get_input_details(self):
        return [{"index": 0, "shape": np.array([1, 224, 224, 3]),
                 "dtype": self.input_dtype, "quantization": self.input_quant}]

    def get_output_details(self):
        return [{"index": 1, "shape": np.array([1, self.outputs.shape[1]]),
                 "dtype": self.output_dtype, "quantization": self.output_quant}]

    def set_tensor(self, index, value):
        self.inputs.append(value)

    def invoke(self):
        pass

    def get_tensor(self, index):
        return self.outputs


def leaf(size=(300, 200), color=(40, 160, 60)):
    return Image.new("RGB", size, color)


def test_input_is_resized_raw_pixels_in_model_shape():
    # The models rescale internally; feeding 0-1 pixels was a past cause of
    # the router collapsing to Maize.
    fake = FakeInterpreter([0.1, 0.8, 0.1])
    core.predict_image(leaf(), fake)
    sent = fake.inputs[0]
    assert sent.shape == (1, 224, 224, 3)
    assert sent.dtype == np.float32
    assert sent[0, 0, 0].tolist() == [40.0, 160.0, 60.0]


def test_class_index_maps_to_label_and_confidence():
    scores = [0.02, 0.05, 0.03, 0.85, 0.04, 0.01]
    result = core.classify_image(leaf(), FakeInterpreter(scores), core.CORN_CLASSES, "Maize")
    assert result["label"] == "Maize_Healthy"
    assert result["conf"] == pytest.approx(85.0)
    assert [label for label, _ in result["top"]] == [
        "Maize_Healthy", "Maize_Common_Rust", "Weed_Broadleaf"]


def test_quantized_output_is_dequantized_before_confidence():
    raw = [0, 0, 0, 255, 0, 0]  # uint8 softmax, scale 1/256
    fake = FakeInterpreter(raw, output_dtype=np.uint8, output_quant=(1 / 256, 0))
    result = core.classify_image(leaf(), fake, core.CORN_CLASSES, "Maize")
    assert result["label"] == "Maize_Healthy"
    assert result["conf"] == pytest.approx(99.6, abs=0.1)


@pytest.mark.parametrize("scores", [
    [2.1, 9.7, -1.3, 0.4, 0.0, 1.0],          # logits: would show 970%
    [0.9, 0.9, 0.9, 0.9, 0.9, 0.9],           # sigmoid scores
    [np.nan, 0.2, 0.2, 0.2, 0.2, 0.2],
])
def test_non_probability_outputs_are_rejected_not_shown_as_confidence(scores):
    with pytest.raises(core.ModelFileError):
        core.classify_image(leaf(), FakeInterpreter(scores), core.CORN_CLASSES, "Maize")


def test_router_rejects_non_probability_outputs():
    with pytest.raises(core.ModelFileError):
        core.route_crop(np.array([1.2, 8.5, 3.3]))


# ---- crop selection: rice routed to Maize at ~99% ----

def test_confident_maize_routing_of_rice_leaf_is_not_preselected():
    route = core.route_crop(np.array([0.004, 0.99, 0.006]))
    assert route["crop"] == "Maize" and route["confidence"] > 98
    assert core.default_crop_choice(route) is None
    assert core.crop_route_status(route, None) == "NEEDS REVIEW"


def test_person_choosing_rice_overrides_router():
    route = core.route_crop(np.array([0.004, 0.99, 0.006]))
    assert core.crop_route_status(route, "Rice") == "SET MANUALLY"
    assert core.crop_route_status(route, "Maize") == "CONFIRMED"


def test_confident_cotton_is_preselected_and_accepted():
    route = core.route_crop(np.array([0.97, 0.02, 0.01]))
    assert core.default_crop_choice(route) == "Cotton"
    assert core.crop_route_status(route, "Cotton") == "ACCEPTED"


def test_failed_router_leaves_crop_unselected():
    route = {"crop": None, "confidence": 0.0, "needs_review": True,
             "reason": "router failed", "scores": {}}
    assert core.default_crop_choice(route) is None
    assert core.crop_route_status(route, "Rice") == "SET MANUALLY"


# ---- quadrants all reading Common Rust ----

def test_quadrants_unanimous_rust_cannot_override_healthy_whole_leaf():
    whole = {"label": "Maize_Healthy", "conf": 98.0}
    quads = {
        name: {"label": "Maize_Common_Rust", "conf": 100.0, "top": []}
        for name in core.split_quadrants(leaf())
    }
    summary = core.summarize_diagnosis(whole, quads)
    assert summary["verdict_kind"] == "healthy"
    assert summary["disease"] is None
    assert summary["needs_review"] is True
    assert summary["quadrant_only"] == ["Maize_Common_Rust"]


def test_quadrants_cover_the_whole_image_without_overlap():
    image = leaf((301, 199))
    parts = core.split_quadrants(image)
    assert sum(p.width * p.height for p in parts.values()) == 301 * 199
