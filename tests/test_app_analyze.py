"""
Analyze All through the real Streamlit script: real router and Maize
TFLite models, a fake Gemini client patched in. Skipped when the model
files have not been downloaded. No key, network or paid API call.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from google.genai import errors, types
from streamlit.testing.v1 import AppTest

import agri_core as core

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).resolve().parent / "data"
APP = ROOT / "app.py"
KEY = "test-gemini-key-not-real"

pytestmark = pytest.mark.skipif(
    not all((ROOT / f).exists() for f in ("crop_router_v2.tflite", "corn_model.tflite")),
    reason="router or maize model not downloaded",
)

RUST = {
    "crop": "Maize", "closest_label": "Maize_Common_Rust", "condition": "Common rust",
    "symptoms": ["Brick-red pustules"], "supporting_features": ["Scattered pustules"],
    "alternatives": ["Southern rust"], "image_quality_issues": [],
    "multiple_conditions": False, "certainty": "high", "summary": "Common rust.",
}


class FakeGemini:
    def __init__(self, assessment=RUST, error=None):
        self.assessment = assessment
        self.error = error
        self.clients = 0
        self.calls = []
        self.models = SimpleNamespace(generate_content=self._generate)

    def factory(self, api_key):
        assert api_key == KEY
        self.clients += 1
        return self

    def _generate(self, model, contents, config=None):
        self.calls.append(contents)
        if self.error:
            raise self.error
        if isinstance(contents[0], types.Part):
            return SimpleNamespace(text=json.dumps(self.assessment))
        return SimpleNamespace(text="## Plan\nCheck with a local expert.")

    @property
    def image_calls(self):
        return [c for c in self.calls if isinstance(c[0], types.Part)]

    @property
    def prompt_calls(self):
        return [c for c in self.calls if isinstance(c[0], str)]


@pytest.fixture
def app(monkeypatch):
    def start(fake, key=KEY):
        monkeypatch.chdir(ROOT)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setattr(core, "make_gemini_client", fake.factory)
        at = AppTest.from_file(str(APP), default_timeout=120)
        # Always give test secrets: with none, AppTest falls back to the
        # real .streamlit/secrets.toml, which tests must never read.
        at.secrets["gemini_model"] = "gemini-test"
        if key:
            at.secrets["GEMINI_API_KEY"] = key
        at.run()
        [leaf_uploader] = [
            u for u in at.get("file_uploader")
            if any("jpg" in t for t in (u.allowed_type or []))
        ]
        leaf_uploader.upload(
            "maize_common_rust.jpg", (DATA / "maize_common_rust.jpg").read_bytes(),
            "image/jpeg").run()
        assert not at.exception
        return at
    return start


def click(at, label):
    [button] = [b for b in at.button if b.label == label]
    button.click().run()
    assert not at.exception
    return at


def page_text(at):
    return "\n".join(m.value for m in at.markdown) + "\n".join(
        i.value for i in at.info)


def test_analyze_all_shows_ml_result_gemini_view_and_agreement(app):
    fake = FakeGemini()
    at = click(app(fake), "Analyze All")
    text = page_text(at)

    # The router suggested Maize (needs review); Analyze All ran on it and
    # kept the warning instead of hiding it.
    assert "Disease: Common Rust" in text
    assert "ROUTER SUGGESTS MAIZE" in text
    assert "Independent AI assessment" in text
    assert "Combined interpretation: Agreement" in text
    assert "NEVER AVERAGED" in text
    assert "Check with a local expert." in text

    assert len(fake.image_calls) == 1 and len(fake.prompt_calls) == 1
    image_part = fake.image_calls[0][0]
    assert image_part.inline_data.mime_type == "image/jpeg"
    assert image_part.inline_data.data[:2] == b"\xff\xd8"
    # The image prompt carries no ML result; the prescription prompt
    # reuses the assessment as text and does not resend the image.
    assert fake.image_calls[0][1] == core.ASSESSMENT_PROMPT
    assert "Independent visual assessment" in fake.prompt_calls[0][0]
    assert KEY not in text


def test_clicking_again_makes_no_new_gemini_requests(app):
    fake = FakeGemini()
    at = click(app(fake), "Analyze All")
    click(at, "Analyze All")
    assert len(fake.image_calls) == 1 and len(fake.prompt_calls) == 1
    assert "Combined interpretation: Agreement" in page_text(at)


def test_gemini_quota_failure_keeps_ml_diagnosis_and_pauses_ai(app):
    quota = errors.ClientError(429, {"error": {
        "code": 429, "status": "RESOURCE_EXHAUSTED", "message": "quota",
        "details": [{"violations": [{"quotaId": "GenerateRequestsPerDayPerProject"}]}]}})
    fake = FakeGemini(error=quota)
    at = click(app(fake), "Analyze All")
    text = page_text(at)

    assert "Disease: Common Rust" in text
    assert "Independent AI assessment unavailable" in text
    assert "has not been checked by Gemini" in text
    assert "Combined interpretation" not in text

    click(at, "Analyze All")
    assert len(fake.calls) == 1


def test_disagreement_is_shown_and_no_prescription_requested(app):
    fake = FakeGemini(assessment={**RUST, "crop": "Rice", "closest_label": "Rice_Brown_Spot",
                                  "condition": "Brown spot"})
    at = click(app(fake), "Analyze All")
    text = page_text(at)
    assert "Combined interpretation: Disagreement" in text
    assert "Disease: Common Rust" in text
    assert fake.prompt_calls == []


def test_missing_key_leaves_ml_diagnosis_without_any_client(app):
    fake = FakeGemini()
    at = click(app(fake, key=None), "Analyze All")
    text = page_text(at)
    assert "Disease: Common Rust" in text
    assert "no Gemini key is configured" in text
    assert fake.clients == 0


def test_run_diagnosis_alone_makes_no_gemini_assessment(app):
    fake = FakeGemini()
    at = app(fake)
    [radio] = [r for r in at.radio if r.label == "Crop"]
    radio.set_value("Maize").run()
    click(at, "Run diagnosis")
    text = page_text(at)
    assert "Disease: Common Rust" in text
    assert "Independent AI assessment" not in text
    assert fake.image_calls == []
