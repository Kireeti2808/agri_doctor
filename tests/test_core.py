"""Pure logic tests: no model files, no network, no OpenAI calls."""

import json

import numpy as np
import pytest

import agri_core as core


# labels.txt exported alongside the Rice model (Drive file
# 1wd160b_B4pEO-gDzz-QnXG81vYRQVWYA), in training order.
EXPORTED_RICE_LABELS = [
    "Rice_Bacterial_Leaf_Blight",
    "Rice_Brown_Spot",
    "Rice_Healthy_Rice_Leaf",
    "Rice_Leaf_Blast",
    "Rice_Leaf_scald",
    "Rice_Sheath_Blight",
    "Weed_Broadleaf",
    "Weed_Grass",
]


def test_class_list_sizes_and_uniqueness():
    expected = {"Router": 3, "Maize": 6, "Rice": 8, "Cotton": 10}
    lists = {
        "Router": core.ROUTER_CLASSES,
        "Maize": core.CORN_CLASSES,
        "Rice": core.RICE_CLASSES,
        "Cotton": core.COTTON_CLASSES,
    }
    for name, labels in lists.items():
        assert len(labels) == expected[name], name
        assert len(set(labels)) == len(labels), name


def test_router_labels_are_alphabetical_like_keras_directories():
    assert core.ROUTER_CLASSES == sorted(core.ROUTER_CLASSES)


def test_rice_labels_match_exported_training_order():
    normalised = [
        "Rice_Healthy" if label == "Rice_Healthy_Rice_Leaf" else label
        for label in EXPORTED_RICE_LABELS
    ]
    assert core.RICE_CLASSES == normalised


@pytest.mark.parametrize(
    "content, reason",
    [
        ("\n".join(EXPORTED_RICE_LABELS).encode(), "labels.txt"),
        (json.dumps({"cells": []}).encode(), "Colab notebook"),
        (b"<!DOCTYPE html><html>Google Drive - Virus scan</html>", "HTML page"),
    ],
)
def test_validate_rejects_files_that_are_not_models(tmp_path, content, reason):
    path = tmp_path / "rice_model.tflite"
    path.write_bytes(content)
    with pytest.raises(core.ModelFileError, match="not a TensorFlow Lite model"):
        core.validate_tflite_file(path)


def test_validate_rejects_truncated_model(tmp_path):
    path = tmp_path / "rice_model.tflite"
    path.write_bytes(b"\x1c\x00\x00\x00TFL3" + b"\x00" * 64)
    with pytest.raises(core.ModelFileError, match="too small"):
        core.validate_tflite_file(path)


def test_validate_rejects_missing_file(tmp_path):
    with pytest.raises(core.ModelFileError, match="does not exist"):
        core.validate_tflite_file(tmp_path / "absent.tflite")


def test_route_crop_accepts_confident_cotton():
    route = core.route_crop(np.array([0.99, 0.005, 0.005]))
    assert route["crop"] == "Cotton"
    assert route["needs_review"] is False


@pytest.mark.parametrize("scores", [[0.01, 0.99, 0.0], [0.0, 0.01, 0.99]])
def test_route_crop_flags_maize_and_rice_even_when_confident(scores):
    route = core.route_crop(np.array(scores))
    assert route["needs_review"] is True
    assert "rice" in route["reason"].lower()


def test_route_crop_flags_low_confidence():
    route = core.route_crop(np.array([0.1, 0.3, 0.6]))
    assert route["needs_review"] is True
    assert "threshold" in route["reason"]


def test_route_crop_rejects_wrong_output_size():
    with pytest.raises(ValueError):
        core.route_crop(np.array([0.5, 0.5]))


def quadrants_of(*labels_confs):
    names = ["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"]
    return {
        name: {"label": label, "conf": conf, "top": [(label, conf)]}
        for name, (label, conf) in zip(names, labels_confs)
    }


def test_summary_uses_whole_leaf_and_counts_agreement():
    whole = {"label": "Maize_Common_Rust", "conf": 100.0}
    quads = quadrants_of(*[("Maize_Common_Rust", 99.0)] * 4)
    summary = core.summarize_diagnosis(whole, quads)
    assert summary["verdict_kind"] == "disease"
    assert summary["disease"] == "Maize_Common_Rust"
    assert summary["agreement"] == 4
    assert summary["needs_review"] is False


def test_summary_flags_quadrant_disagreement():
    whole = {"label": "Maize_Blight", "conf": 97.6}
    quads = quadrants_of(
        ("Maize_Common_Rust", 100.0),
        ("Maize_Common_Rust", 88.0),
        ("Maize_Blight", 95.0),
        ("Maize_Common_Rust", 94.0),
    )
    summary = core.summarize_diagnosis(whole, quads)
    assert summary["verdict_kind"] == "disease"
    assert summary["disease"] == "Maize_Blight"
    assert summary["needs_review"] is True
    assert summary["quadrant_only"] == ["Maize_Common_Rust"]


def test_summary_reports_uncertain_when_whole_leaf_is_weak():
    whole = {"label": "Maize_Blight", "conf": 41.0}
    quads = quadrants_of(*[("Maize_Healthy", 60.0)] * 4)
    summary = core.summarize_diagnosis(whole, quads)
    assert summary["verdict_kind"] == "uncertain"
    assert summary["disease"] is None and summary["weed"] is None
    assert summary["needs_review"] is True


def test_summary_healthy_leaf_has_no_issue():
    whole = {"label": "Maize_Healthy", "conf": 100.0}
    quads = quadrants_of(*[("Maize_Healthy", 99.0)] * 4)
    summary = core.summarize_diagnosis(whole, quads)
    assert summary["verdict_kind"] == "healthy"
    assert summary["disease"] is None and summary["weed"] is None


def test_pretty_labels():
    assert core.pretty_label("Maize_Common_Rust") == "Common Rust"
    assert core.pretty_label("Weed_Palmer_Amaranth") == "Palmer Amaranth weed"
    assert core.pretty_label("Disease_Healthy") == "Healthy"


def test_soil_moisture_bands():
    assert core.classify_soil_moisture(None) is None
    assert core.classify_soil_moisture(0.08) == "Dry"
    assert core.classify_soil_moisture(0.2) == "Moist"
    assert core.classify_soil_moisture(0.35) == "Wet"
    assert core.classify_soil_moisture(0.45) == "Waterlogged"


# ---- model download (fake downloader, no network) ----

TFLITE_BYTES = b"\x1c\x00\x00\x00TFL3" + b"\x00" * (core.MIN_MODEL_BYTES + 8)


def fake_downloader(payload=None, error=None):
    def download(url, output, quiet):
        if error:
            raise error
        if payload is None:
            return None
        with open(output, "wb") as handle:
            handle.write(payload)
        return output
    return download


class FileURLRetrievalError(Exception):
    """Same name as gdown's exception for an inaccessible file."""


def test_download_error_becomes_model_file_error(tmp_path):
    with pytest.raises(core.ModelFileError, match="Could not download"):
        core.download_model("bad-id", tmp_path / "rice_model.tflite",
                            downloader=fake_downloader(error=FileURLRetrievalError("no access")))
    assert not any(tmp_path.iterdir())


def test_download_of_labels_file_is_rejected_and_removed(tmp_path):
    labels = "\n".join(EXPORTED_RICE_LABELS).encode()
    with pytest.raises(core.ModelFileError, match="not a TensorFlow Lite model"):
        core.download_model("labels-id", tmp_path / "rice_model.tflite",
                            downloader=fake_downloader(payload=labels))
    assert not any(tmp_path.iterdir())


def test_failed_refresh_keeps_working_model(tmp_path):
    model = tmp_path / "cotton.tflite"
    model.write_bytes(TFLITE_BYTES)
    with pytest.raises(core.ModelFileError):
        core.download_model("x", model, force_download=True,
                            downloader=fake_downloader(payload=b'{"cells": []}'))
    assert model.read_bytes() == TFLITE_BYTES


def test_valid_download_is_installed(tmp_path):
    model = tmp_path / "rice_model.tflite"
    core.download_model("ok", model, downloader=fake_downloader(payload=TFLITE_BYTES))
    assert model.read_bytes() == TFLITE_BYTES


def test_existing_valid_model_is_reused_without_download(tmp_path):
    model = tmp_path / "corn.tflite"
    model.write_bytes(TFLITE_BYTES)
    core.download_model("x", model, downloader=fake_downloader(error=AssertionError("should not download")))


def test_existing_invalid_file_is_replaced(tmp_path):
    model = tmp_path / "rice_model.tflite"
    model.write_text("Rice_Brown_Spot\n")
    core.download_model("ok", model, downloader=fake_downloader(payload=TFLITE_BYTES))
    assert model.read_bytes() == TFLITE_BYTES


def test_rice_is_marked_unvalidated_with_evidence():
    note = core.UNVALIDATED_MODELS["Rice"]
    assert "not been validated" in note and "1 of 4" in note
