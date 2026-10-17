"""
Smoke test for training/train_rice.py on a tiny synthetic dataset.

Proves the pipeline mechanics only (audit, leakage-safe split, training,
TFLite export, export verification). It says nothing about accuracy.
Runs only where TensorFlow is installed (training/.venv); skipped in the
app environment.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("tensorflow")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

import train_rice  # noqa: E402
from agri_core import RICE_CLASSES, check_output_classes, validate_tflite_file  # noqa: E402


def make_dataset(root, per_class=10, seed=0):
    rng = np.random.default_rng(seed)
    for index, label in enumerate(RICE_CLASSES):
        folder = root / ("Rice_Healthy_Rice_Leaf" if label == "Rice_Healthy" else label)
        folder.mkdir(parents=True)
        base = np.zeros((64, 64, 3), dtype=np.uint8)
        base[..., index % 3] = 60 + 20 * index
        for n in range(per_class):
            noise = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
            stripe = np.zeros_like(base)
            stripe[:, (n * 6) % 64:(n * 6) % 64 + 8] = 255
            image = np.clip(base.astype(int) + noise // 2 + stripe // 2, 0, 255).astype(np.uint8)
            Image.fromarray(image).save(folder / f"{n}.png")
    # exact duplicate copied into a second file of the same class
    dup = root / RICE_CLASSES[1]
    (dup / "copy_of_0.png").write_bytes((dup / "0.png").read_bytes())


def test_pipeline_end_to_end(tmp_path):
    data = tmp_path / "data"
    make_dataset(data)
    out = tmp_path / "run"

    results = train_rice.main([
        "--data", str(data), "--out", str(out), "--weights", "none",
        "--epochs-head", "1", "--epochs-finetune", "1", "--unfreeze", "5",
        "--batch", "8", "--val", "0.2", "--test", "0.2", "--dup-threshold", "0",
    ])

    report = json.loads((out / "data_report.json").read_text())
    assert report["images"] == 8 * 10 + 1
    assert report["duplicate_groups"] >= 1
    assert set(report["class_counts"]) == set(RICE_CLASSES)

    # no image (or duplicate group) appears in two splits
    split = json.loads((out / "split.json").read_text())
    seen = {}
    for name, rows in split.items():
        for path, _ in rows:
            assert path not in seen, f"{path} in {seen.get(path)} and {name}"
            seen[path] = name
    copy = str(data / RICE_CLASSES[1] / "copy_of_0.png")
    original = str(data / RICE_CLASSES[1] / "0.png")
    assert seen[copy] == seen[original], "duplicate leaked across splits"

    model_path = out / "rice_model.tflite"
    validate_tflite_file(model_path)
    assert (out / "labels.txt").read_text().split() == RICE_CLASSES

    from ai_edge_litert.interpreter import Interpreter
    interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    assert check_output_classes(interpreter, RICE_CLASSES, "Rice") == 8
    details = interpreter.get_input_details()[0]
    assert list(details["shape"]) == [1, 224, 224, 3]

    export = results["export"]
    assert export["keras_vs_tflite_max_abs_difference"] < 1e-3
    assert set(results["final"]["per_class"]) >= set(RICE_CLASSES)
    assert (out / "metrics.md").exists()
