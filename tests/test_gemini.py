"""
Gemini integration and Analyze All: key handling, error classification,
the independent image assessment, the ML/Gemini comparison and request
reuse. A fake client stands in for Gemini; real google-genai error types
and httpx exceptions are used. No key, network or paid API call.
"""

import copy
import io
import json
from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors, types
from PIL import Image

import agri_core as core


KEY = "test-gemini-key-not-real"


# ---- fake Gemini client ----

class FakeGemini:
    """Records every generate_content call; replies or raises in order."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []
        self.keys = []
        self.models = SimpleNamespace(generate_content=self._generate)

    def factory(self, api_key):
        self.keys.append(api_key)
        return self

    def _generate(self, model, contents, config=None):
        self.calls.append({"model": model, "contents": contents, "config": config})
        reply = self.replies.pop(0) if self.replies else ""
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(text=reply)


def api_error(cls, code, status, message, details=None):
    body = {"error": {"code": code, "status": status, "message": message}}
    if details is not None:
        body["error"]["details"] = details
    return cls(code, body)


def leaf_jpeg(color=(40, 160, 60), size=(320, 240)):
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="JPEG")
    return buffer.getvalue()


ASSESSMENT = {
    "crop": "Maize",
    "closest_label": "Maize_Common_Rust",
    "condition": "Common rust",
    "symptoms": ["Small brick-red pustules on both leaf surfaces"],
    "supporting_features": ["Pustules scattered across the blade"],
    "alternatives": ["Southern rust"],
    "image_quality_issues": [],
    "multiple_conditions": False,
    "certainty": "high",
    "summary": "Looks like common rust.",
}


def assessment(**changes):
    return core.normalize_assessment({**ASSESSMENT, **changes})


def ml_summary(label="Maize_Common_Rust", conf=97.0, kind=None):
    whole = {"label": label, "conf": conf}
    quads = {
        name: {"label": label, "conf": conf, "top": [(label, conf)]}
        for name in ("Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right")
    }
    summary = core.summarize_diagnosis(whole, quads)
    if kind:
        assert summary["verdict_kind"] == kind
    return summary


# ---- GEMINI_API_KEY and model ----

class RaisingSecrets:
    def get(self, name, default=None):
        raise FileNotFoundError("No secrets found")


def test_key_is_read_from_streamlit_secret_GEMINI_API_KEY():
    assert core.resolve_api_key({"GEMINI_API_KEY": f" {KEY} "}, {}) == KEY


def test_key_falls_back_to_environment_and_ignores_openai_names():
    assert core.resolve_api_key(RaisingSecrets(), {"GEMINI_API_KEY": KEY}) == KEY
    assert core.resolve_api_key(
        {"OPENAI_API_KEY": "sk-old", "openai_key": "sk-old"},
        {"OPENAI_API_KEY": "sk-old"}
    ) is None


def test_blank_key_is_missing():
    assert core.resolve_api_key({"GEMINI_API_KEY": "  "}, {"GEMINI_API_KEY": ""}) is None


def test_model_name_from_secret_environment_or_default():
    assert core.resolve_gemini_model({"gemini_model": "m1"}, {"GEMINI_MODEL": "m2"}) == "m1"
    assert core.resolve_gemini_model(RaisingSecrets(), {"GEMINI_MODEL": "m2"}) == "m2"
    assert core.resolve_gemini_model(RaisingSecrets(), {}) == core.DEFAULT_GEMINI_MODEL


def test_real_client_is_built_with_bounded_timeout_without_a_request():
    client = core.make_gemini_client(KEY)
    assert client._api_client._http_options.timeout == core.GEMINI_TIMEOUT_MS


# ---- error classification ----

PER_MINUTE = api_error(
    errors.ClientError, 429, "RESOURCE_EXHAUSTED",
    "You exceeded your current quota, please check your plan and billing "
    "details. Quota exceeded for metric: generate_content_free_tier_requests, "
    "limit: 15",
    [{"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [
        {"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}]}],
)
PER_DAY = api_error(
    errors.ClientError, 429, "RESOURCE_EXHAUSTED",
    "You exceeded your current quota.",
    [{"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [
        {"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}],
)
NO_FREE_TIER = api_error(
    errors.ClientError, 429, "RESOURCE_EXHAUSTED",
    "Quota exceeded for metric: generate_content_free_tier_requests, limit: 0",
)
INVALID_KEY = api_error(
    errors.ClientError, 400, "INVALID_ARGUMENT",
    f"API key not valid. Please pass a valid API key. ({KEY})",
    [{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": "API_KEY_INVALID"}],
)


@pytest.mark.parametrize("error, kind, blocks", [
    (core.MissingKey(), "missing_key", True),
    (INVALID_KEY, "auth", True),
    (api_error(errors.ClientError, 403, "PERMISSION_DENIED", "denied"), "auth", True),
    (PER_DAY, "quota", True),
    (NO_FREE_TIER, "quota", True),
    (PER_MINUTE, "rate_limit", False),
    (api_error(errors.ClientError, 404, "NOT_FOUND", "models/x is not found"), "model", True),
    (api_error(errors.ServerError, 504, "DEADLINE_EXCEEDED", "deadline"), "timeout", False),
    (httpx.ReadTimeout("timed out"), "timeout", False),
    (TimeoutError(), "timeout", False),
    (httpx.ConnectError("dns failure"), "unreachable", False),
    (ConnectionError(), "unreachable", False),
    (api_error(errors.ServerError, 500, "INTERNAL", "internal"), "other", False),
    (api_error(errors.ServerError, 503, "UNAVAILABLE", "overloaded"), "other", False),
    (core.EmptyResponse(), "other", False),
])
def test_gemini_failures_are_classified(error, kind, blocks):
    info = core.describe_advice_error(error)
    assert info["kind"] == kind
    assert info["blocks_session"] is blocks
    assert "image diagnosis above is unaffected" in info["message"]


@pytest.mark.parametrize("error", [INVALID_KEY, PER_DAY, PER_MINUTE])
def test_provider_text_and_key_never_reach_the_user(error):
    info = core.describe_advice_error(error)
    shown = info["message"] + info["detail"]
    assert KEY not in shown
    assert "quotaId" not in shown and "Please pass" not in shown
    assert info["detail"].startswith("ClientError · HTTP ")


def test_quota_message_and_sidebar_first_sentence():
    message = core.describe_advice_error(PER_DAY)["message"]
    assert message.split(". ")[0].startswith("AI prescriptions are unavailable")
    assert "without an AI-generated prescription" in message


# ---- prescription request ----

def test_prescription_success_sends_prompt_to_configured_model():
    fake = FakeGemini("## Plan")
    result = core.request_advice("PROMPT", KEY, "gemini-test", fake.factory)
    assert result == {"ok": True, "text": "## Plan"}
    assert fake.keys == [KEY]
    assert fake.calls[0]["model"] == "gemini-test"
    assert fake.calls[0]["contents"] == ["PROMPT"]


def test_missing_key_never_creates_a_client():
    fake = FakeGemini("unused")
    result = core.request_advice("PROMPT", None, "m", fake.factory)
    assert result["kind"] == "missing_key"
    assert fake.keys == [] and fake.calls == []


@pytest.mark.parametrize("reply", ["", "   ", None])
def test_empty_prescription_is_a_failure(reply):
    result = core.request_advice("PROMPT", KEY, "m", FakeGemini(reply).factory)
    assert result["ok"] is False and result["kind"] == "other"


@pytest.mark.parametrize("error, kind", [
    (PER_DAY, "quota"), (PER_MINUTE, "rate_limit"), (INVALID_KEY, "auth"),
    (httpx.ConnectError("x"), "unreachable"), (httpx.ReadTimeout("x"), "timeout"),
])
def test_prescription_errors_become_error_dicts(error, kind):
    result = core.request_advice("PROMPT", KEY, "m", FakeGemini(error).factory)
    assert result["ok"] is False and result["kind"] == kind


# ---- independent image assessment ----

def test_uploaded_image_reaches_gemini_with_only_the_independent_prompt():
    image = leaf_jpeg()
    fake = FakeGemini(json.dumps(ASSESSMENT))
    result = core.assess_leaf_image(image, KEY, "gemini-test", fake.factory)

    assert result["ok"] is True
    assert result["assessment"]["closest_label"] == "Maize_Common_Rust"

    contents = fake.calls[0]["contents"]
    assert isinstance(contents[0], types.Part)
    assert contents[0].inline_data.data == image
    assert contents[0].inline_data.mime_type == "image/jpeg"
    assert contents[1:] == [core.ASSESSMENT_PROMPT]

    config = fake.calls[0]["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_json_schema == core.ASSESSMENT_SCHEMA


def test_assessment_prompt_is_independent_of_the_ml_result():
    prompt = core.ASSESSMENT_PROMPT
    assert "independently" in prompt
    assert "No other\nsystem's diagnosis is given to you" in prompt
    # Every label is offered as an option; none is singled out as a result.
    assert all(label in prompt for label in core.ASSESSMENT_LABELS)
    assert "%" not in prompt


@pytest.mark.parametrize("reply", ["not json", "[1, 2]", ""])
def test_unreadable_assessment_is_a_failure(reply):
    result = core.assess_leaf_image(leaf_jpeg(), KEY, "m", FakeGemini(reply).factory)
    assert result["ok"] is False and result["kind"] == "other"


def test_assessment_values_outside_the_schema_are_not_trusted():
    normalized = core.normalize_assessment({
        "crop": "Wheat", "closest_label": "Maize_Made_Up", "certainty": "certain",
        "symptoms": "not a list", "multiple_conditions": "yes",
    })
    assert normalized["crop"] == "Unknown"
    assert normalized["closest_label"] == "Unknown"
    assert normalized["certainty"] == "low"
    assert normalized["symptoms"] == []
    assert normalized["multiple_conditions"] is False


def test_assessment_image_is_bounded_jpeg_and_fingerprinted():
    image = Image.new("RGB", (3000, 1500), (90, 140, 60))
    data = core.image_for_assessment(image)
    decoded = Image.open(io.BytesIO(data))
    assert decoded.format == "JPEG" and max(decoded.size) == 1024
    other = core.image_for_assessment(Image.new("RGB", (3000, 1500), (10, 10, 10)))
    assert core.image_fingerprint(data) == core.image_fingerprint(data)
    assert core.image_fingerprint(data) != core.image_fingerprint(other)


# ---- comparison ----

def test_agreement_does_not_claim_proof():
    result = core.compare_assessments("Maize", ml_summary(), assessment())
    assert result["category"] == core.AGREEMENT
    assert "does not prove" in result["next_step"]
    assert result["supports_prescription"] is True


def test_agreement_with_suspected_second_condition_is_partial():
    result = core.compare_assessments(
        "Maize", ml_summary(), assessment(multiple_conditions=True))
    assert result["category"] == core.PARTIAL


def test_same_kind_different_disease_is_partial():
    result = core.compare_assessments(
        "Maize", ml_summary(),
        assessment(closest_label="Maize_Gray_Leaf_Spot", condition="Gray leaf spot",
                   alternatives=["Common rust"]))
    assert result["category"] == core.PARTIAL
    assert any("alternatives" in reason for reason in result["reasons"])


def test_condition_outside_the_label_list_is_partial_when_both_see_a_problem():
    result = core.compare_assessments(
        "Maize", ml_summary(),
        assessment(closest_label="Other", condition="Nitrogen deficiency"))
    assert result["category"] == core.PARTIAL


def test_rice_leaf_routed_to_maize_is_a_disagreement():
    result = core.compare_assessments(
        "Maize", ml_summary(),
        assessment(crop="Rice", closest_label="Rice_Brown_Spot"))
    assert result["category"] == core.DISAGREEMENT
    assert "Confirm the crop" in result["reasons"][0]
    assert result["supports_prescription"] is False


@pytest.mark.parametrize("ml_label, gemini_label", [
    ("Maize_Healthy", "Maize_Common_Rust"),
    ("Maize_Common_Rust", "Maize_Healthy"),
    ("Maize_Common_Rust", "Weed_Grass"),
])
def test_conflicting_findings_are_a_disagreement(ml_label, gemini_label):
    result = core.compare_assessments(
        "Maize", ml_summary(ml_label), assessment(closest_label=gemini_label))
    assert result["category"] == core.DISAGREEMENT
    assert "expert" in result["next_step"]


def test_non_crop_plant_is_a_disagreement():
    result = core.compare_assessments("Maize", ml_summary(), assessment(crop="Other"))
    assert result["category"] == core.DISAGREEMENT


@pytest.mark.parametrize("summary, gemini", [
    (ml_summary(conf=41.0, kind="uncertain"), assessment()),
    (ml_summary(), assessment(certainty="low")),
    (ml_summary(), assessment(closest_label="Unknown")),
])
def test_weak_evidence_is_uncertain(summary, gemini):
    result = core.compare_assessments("Maize", summary, gemini)
    assert result["category"] == core.UNCERTAIN
    assert result["supports_prescription"] is False
    assert "closer, well-lit photo" in result["next_step"]


def test_healthy_agreement_needs_no_prescription():
    result = core.compare_assessments(
        "Maize", ml_summary("Maize_Healthy"), assessment(closest_label="Maize_Healthy"))
    assert result["category"] == core.AGREEMENT
    assert result["supports_prescription"] is False


# ---- Analyze All orchestration ----

class Recorder:
    def __init__(self, assess_results, advice_results=({"ok": True, "text": "## Plan"},)):
        self.assess_results = list(assess_results)
        self.advice_results = list(advice_results)
        self.assess_calls = 0
        self.prescribe_args = []

    def assess(self):
        self.assess_calls += 1
        return self.assess_results.pop(0)

    def prescribe(self, assessment, comparison):
        self.prescribe_args.append((assessment, comparison))
        return self.advice_results.pop(0)


def ok_assessment(**changes):
    return {"ok": True, "assessment": assessment(**changes)}


def run(recorder, cache, summary=None, image_key="img-1", **kwargs):
    return core.run_ai_analysis(
        ml_crop="Maize", ml_summary=summary or ml_summary(), image_key=image_key,
        model="gemini-test", assess=recorder.assess, prescribe=recorder.prescribe,
        cache=cache, **kwargs)


def test_analyze_all_keeps_ml_result_and_reuses_the_assessment_for_advice():
    summary = ml_summary()
    before = copy.deepcopy(summary)
    recorder = Recorder([ok_assessment()])

    result = run(recorder, {}, summary=summary)

    assert summary == before
    assert result["comparison"]["category"] == core.AGREEMENT
    assert result["advice"] == {"ok": True, "text": "## Plan"}
    assert recorder.assess_calls == 1
    # The prescription is written from the assessment already received,
    # not by sending the image again.
    assert recorder.prescribe_args[0][0] is result["assessment"]


def test_repeated_clicks_reuse_answers_instead_of_new_requests():
    recorder = Recorder([ok_assessment()])
    cache = {}
    first = run(recorder, cache)
    second = run(recorder, cache)
    assert recorder.assess_calls == 1 and len(recorder.prescribe_args) == 1
    assert second["assessment"] == first["assessment"]
    assert second["advice"] == first["advice"]


def test_a_different_image_is_never_served_a_cached_answer():
    recorder = Recorder([ok_assessment(), ok_assessment(closest_label="Maize_Healthy")],
                        [{"ok": True, "text": "A"}])
    cache = {}
    run(recorder, cache, image_key="img-1")
    second = run(recorder, cache, image_key="img-2")
    assert recorder.assess_calls == 2
    assert second["assessment"]["closest_label"] == "Maize_Healthy"


def test_changed_field_conditions_get_a_new_prescription_only():
    recorder = Recorder([ok_assessment()],
                        [{"ok": True, "text": "dry"}, {"ok": True, "text": "wet"}])
    cache = {}
    run(recorder, cache, advice_context=["Loam", "Dry"])
    second = run(recorder, cache, advice_context=["Loam", "Wet"])
    assert recorder.assess_calls == 1
    assert second["advice"]["text"] == "wet"


def test_gemini_failure_leaves_ml_result_and_is_retried_on_next_click():
    failure = {"ok": False, **core.describe_advice_error(httpx.ConnectError("x"))}
    recorder = Recorder([failure, ok_assessment()])
    cache = {}
    summary = ml_summary()
    before = copy.deepcopy(summary)

    first = run(recorder, cache, summary=summary)
    assert first["assessment"] is None and first["comparison"] is None
    assert first["error"]["kind"] == "unreachable"
    assert first["blocked"] is None
    assert summary == before

    second = run(recorder, cache, summary=summary)
    assert recorder.assess_calls == 2
    assert second["comparison"]["category"] == core.AGREEMENT


def test_exhausted_quota_pauses_further_requests():
    quota = {"ok": False, **core.describe_advice_error(PER_DAY)}
    recorder = Recorder([quota])
    first = run(recorder, {})
    assert first["blocked"]["kind"] == "quota"

    second = run(recorder, {}, blocked=first["blocked"])
    assert recorder.assess_calls == 1
    assert second["error"]["kind"] == "quota"


def test_disagreement_writes_no_prescription():
    recorder = Recorder([ok_assessment(crop="Rice", closest_label="Rice_Brown_Spot")])
    result = run(recorder, {})
    assert result["comparison"]["category"] == core.DISAGREEMENT
    assert result["advice"] is None and recorder.prescribe_args == []


def test_failed_prescription_keeps_the_assessment():
    failure = {"ok": False, **core.describe_advice_error(PER_MINUTE)}
    recorder = Recorder([ok_assessment()], [failure])
    result = run(recorder, {})
    assert result["comparison"]["category"] == core.AGREEMENT
    assert result["advice"]["kind"] == "rate_limit"
    assert result["blocked"] is None


def test_ai_turned_off_makes_no_requests():
    recorder = Recorder([])
    result = run(recorder, {}, ai_enabled=False)
    assert recorder.assess_calls == 0
    assert result["assessment"] is None and result["error"] is None


# ---- ML thresholds are unchanged by this feature ----

def test_ml_thresholds_are_unchanged():
    assert core.ROUTER_THRESHOLD == 75.0
    assert core.ROUTER_CONFUSABLE == {"Maize", "Rice"}
    assert core.DETECTION_THRESHOLD == 50.0
    assert core.MIN_QUADRANT_AGREEMENT == 2


# ---- a session pause follows the configured key ----

def test_missing_key_pause_lifts_when_a_key_is_added():
    pause = core.block_for_key({"kind": "missing_key"}, None)
    assert core.block_applies(pause, None) is True
    assert core.block_applies(pause, KEY) is False


def test_invalid_key_pause_holds_for_same_key_and_lifts_for_a_new_one():
    pause = core.block_for_key({"kind": "auth"}, "old-key-not-real")
    assert core.block_applies(pause, "old-key-not-real") is True
    assert core.block_applies(pause, KEY) is False
    assert core.block_applies(None, KEY) is False


def test_pause_keeps_only_a_one_way_fingerprint_of_the_key():
    pause = core.block_for_key({"kind": "quota", "message": "m"}, KEY)
    assert KEY not in json.dumps(pause)
    assert pause["key_fingerprint"] == core.key_fingerprint(KEY)
