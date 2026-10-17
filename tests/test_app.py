"""Streamlit smoke test: the page renders without an upload or secrets."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_app_renders_empty_state():
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    assert not at.exception
    assert any("Read a leaf" in m.value for m in at.markdown)
    labels = [w.label for w in at.selectbox]
    assert "Soil type" in labels and "Soil moisture" in labels
