import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).resolve().parent / "data"

sys.path.insert(0, str(ROOT))


def load_interpreter(filename):
    path = ROOT / filename

    if not path.exists():
        pytest.skip(f"{filename} not downloaded")

    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    return interpreter


@pytest.fixture(scope="session")
def router():
    return load_interpreter("crop_router_v2.tflite")


@pytest.fixture(scope="session")
def corn():
    return load_interpreter("corn_model.tflite")


@pytest.fixture(scope="session")
def cotton():
    return load_interpreter("cotton_doctor_latest.tflite")


@pytest.fixture(scope="session")
def rice():
    return load_interpreter("rice_model.tflite")
