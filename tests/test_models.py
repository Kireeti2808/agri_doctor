"""
Tests against the real TFLite models and known crop photos in tests/data.
Skipped when a model file has not been downloaded. No network calls.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import agri_core as core

DATA = Path(__file__).resolve().parent / "data"
ROOT = Path(__file__).resolve().parents[1]


def images(prefix):
    found = sorted(DATA.glob(f"{prefix}_*.jpg"))
    assert found, f"no {prefix} test images"
    return found


@pytest.mark.parametrize(
    "fixture, filename, classes",
    [
        ("router", "crop_router_v2.tflite", core.ROUTER_CLASSES),
        ("corn", "corn_model.tflite", core.CORN_CLASSES),
        ("cotton", "cotton_doctor_latest.tflite", core.COTTON_CLASSES),
        ("rice", "rice_model.tflite", core.RICE_CLASSES),
    ],
)
def test_model_file_shape_and_labels(request, fixture, filename, classes):
    interpreter = request.getfixturevalue(fixture)
    core.validate_tflite_file(ROOT / filename)
    assert core.check_output_classes(interpreter, classes, fixture) == len(classes)

    details = interpreter.get_input_details()[0]
    assert list(details["shape"]) == [1, 224, 224, 3]
    assert details["dtype"] == np.float32


def test_outputs_are_probabilities(router, corn):
    image = Image.open(images("maize")[0])
    for interpreter in (router, corn):
        predictions = core.predict_image(image, interpreter)
        assert predictions.min() >= 0
        assert predictions.sum() == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("path", images("cotton"), ids=lambda p: p.name)
def test_router_accepts_cotton(router, path):
    route = core.route_crop(core.predict_image(Image.open(path), router))
    assert route["crop"] == "Cotton"
    assert route["needs_review"] is False


@pytest.mark.parametrize("path", images("maize"), ids=lambda p: p.name)
def test_router_suggests_maize_but_asks_for_confirmation(router, path):
    route = core.route_crop(core.predict_image(Image.open(path), router))
    assert route["crop"] == "Maize"
    assert route["needs_review"] is True


@pytest.mark.parametrize("path", images("rice"), ids=lambda p: p.name)
def test_rice_is_never_routed_without_review(router, path):
    # The router misroutes several of these rice photos to Maize at >90%.
    # Whatever it suggests, the app must ask a person to confirm.
    route = core.route_crop(core.predict_image(Image.open(path), router))
    assert route["needs_review"] is True


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("maize_common_rust.jpg", "Maize_Common_Rust"),
        ("maize_nclb.jpg", "Maize_Blight"),
        ("maize_fresh_leaf.jpg", "Maize_Healthy"),
    ],
)
def test_whole_leaf_reading_on_known_maize(corn, filename, expected):
    result = core.classify_image(Image.open(DATA / filename), corn, core.CORN_CLASSES, "Maize")
    assert result["label"] == expected


def test_quadrants_corroborate_a_clear_case(corn):
    image = Image.open(DATA / "maize_common_rust.jpg")
    whole = core.classify_image(image, corn, core.CORN_CLASSES, "Maize")
    quads = core.analyze_quadrants(image, corn, core.CORN_CLASSES, "Maize")
    summary = core.summarize_diagnosis(whole, quads)
    assert summary["agreement"] == 4
    assert summary["needs_review"] is False


def test_quadrant_bias_is_flagged_not_reported(corn):
    # Whole leaf: Northern Leaf Blight. Quadrants drift to Common Rust.
    image = Image.open(DATA / "maize_nclb.jpg")
    whole = core.classify_image(image, corn, core.CORN_CLASSES, "Maize")
    quads = core.analyze_quadrants(image, corn, core.CORN_CLASSES, "Maize")
    summary = core.summarize_diagnosis(whole, quads)
    assert summary["disease"] == "Maize_Blight"
    assert summary["needs_review"] is True


# Labelled rice-disease photos (Wikimedia Commons; see data/ATTRIBUTION.md).
# Measured with rice_weed_robust.tflite: only the sheath-blight photo is
# classified correctly. These tests pin that evidence; they are not an
# accuracy benchmark.
RICE_LABELLED = {
    "rice_sheath_blight.jpg": "Rice_Sheath_Blight",
    "rice_brown_spot.jpg": "Rice_Brown_Spot",
    "rice_blast.jpg": "Rice_Leaf_Blast",
    "rice_blast_2.jpg": "Rice_Leaf_Blast",
}


def rice_predictions(rice):
    out = {}
    for filename in RICE_LABELLED:
        image, _ = core.load_leaf_image((DATA / filename).read_bytes())
        out[filename] = core.classify_image(image, rice, core.RICE_CLASSES, "Rice")
    return out


def test_rice_recognises_sheath_blight(rice):
    assert rice_predictions(rice)["rice_sheath_blight.jpg"]["label"] == "Rice_Sheath_Blight"


def test_rice_measured_agreement_matches_documented_evidence(rice):
    predictions = rice_predictions(rice)
    correct = sum(predictions[f]["label"] == t for f, t in RICE_LABELLED.items())
    # If this changes, update UNVALIDATED_MODELS["Rice"] with the new count.
    assert correct == 1
