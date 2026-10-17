import base64
import html
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

try:
    import folium
    from folium.plugins import Draw
    from streamlit_folium import st_folium
except ImportError:
    folium = None

try:
    from streamlit_js_eval import streamlit_js_eval
except ImportError:
    streamlit_js_eval = None

from agri_core import (
    CORN_CLASSES,
    COTTON_CLASSES,
    CROP_CLASSES,
    RICE_CLASSES,
    ROUTER_CLASSES,
    ModelFileError,
    analyze_quadrants,
    check_output_classes,
    classify_image,
    classify_soil_moisture,
    assess_leaf_image,
    block_applies,
    block_for_key,
    describe_advice_error,
    image_fingerprint,
    image_for_assessment,
    run_ai_analysis,
    download_model,
    UNVALIDATED_MODELS,
    crop_route_status,
    default_crop_choice,
    request_advice,
    resolve_gemini_model,
    field_record,
    resolve_api_key,
    ImageInputError,
    load_leaf_image,
    load_field_records,
    parse_geolocation,
    parse_weather,
    polygon_area_hectares,
    label_kind,
    predict_image,
    pretty_label,
    route_crop,
    summarize_diagnosis,
    validate_tflite_file
)

# The lightweight LiteRT runtime is preferred (fast installs on Streamlit
# Cloud); full TensorFlow is used when it is the only runtime available.
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    import tensorflow as tf

    Interpreter = tf.lite.Interpreter


# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Agri-Doctor Pro",
    layout="wide",
    initial_sidebar_state="auto"
)


# ============================================================
# UI STYLING
# Munsell soil-chart world: soil-brown rail, neutral grey mat,
# borderless mounted chips, notation set in tabular mono.
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Hanken+Grotesk:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

    :root {
        --soil: #2e241b;
        --soil-2: #3d3126;
        --bone: #ece5d8;
        --bone-dim: #c4b8a3;
        --mat: #c9c4ba;
        --mat-deep: #bdb7ac;
        --mat-edge: #a9a398;
        --ink: #231b14;
        --ink-2: #4a3f33;
        --rust: #9a3a2a;
        --ochre: #8a6420;
        --ochre-chip: #c3933f;
        --leaf: #46692a;
        --signal: #e0b12e;
        --mono: 'JetBrains Mono', ui-monospace, monospace;
        --sans: 'Hanken Grotesk', system-ui, sans-serif;
    }

    html, body, [data-testid="stAppViewContainer"], .stMarkdown, p, label,
    input, textarea, button, [data-testid="stWidgetLabel"] {
        font-family: var(--sans);
    }

    ::selection { background: var(--soil); color: var(--bone); }

    [data-testid="stAppViewContainer"] { background: var(--mat); }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stAppDeployButton"], #MainMenu, footer { display: none; }

    .block-container {
        padding-top: 2.75rem;
        padding-bottom: 5rem;
        max-width: 1280px;
    }

    /* ---------- Sidebar rail ---------- */
    [data-testid="stSidebar"] { background: var(--soil); }
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        font-family: var(--sans);
        font-size: 0.86rem;
        font-weight: 600;
        color: var(--bone);
    }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        color: var(--bone-dim) !important;
        opacity: 1 !important;
        font-family: var(--sans);
        font-size: 0.78rem;
        line-height: 1.5;
    }
    [data-testid="stSidebar"] input { caret-color: var(--bone); }
    input, textarea { caret-color: var(--soil); }

    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: var(--mat-deep); }
    ::-webkit-scrollbar-thumb { background: var(--soil); border-radius: 5px; border: 2px solid var(--mat-deep); }
    * { scrollbar-color: var(--soil) var(--mat-deep); }
    [data-testid="stSidebar"] * { scrollbar-color: #55473a var(--soil); }

    .wordmark {
        font-weight: 800;
        font-size: 1.9rem;
        letter-spacing: -0.03em;
        line-height: 1;
        color: var(--bone);
        margin: 0.25rem 0 0.6rem;
    }
    .wordmark-note {
        font-family: var(--mono);
        font-size: 0.7rem;
        letter-spacing: 0.05em;
        color: var(--bone-dim);
        line-height: 1.6;
        margin-bottom: 1.75rem;
    }
    .rail-heading {
        font-weight: 700;
        font-size: 0.95rem;
        color: var(--bone);
        margin: 1.75rem 0 0.75rem;
        padding-top: 1.25rem;
        border-top: 1px solid #55473a;
    }
    .rail-reading {
        font-family: var(--sans);
        font-size: 0.8rem;
        color: var(--bone-dim);
        font-variant-numeric: tabular-nums;
        margin: -0.35rem 0 0.9rem;
        line-height: 1.5;
    }
    .rail-reading b { color: var(--bone); font-weight: 500; }

    .pipeline { list-style: none; padding: 0; margin: 0; }
    .pipeline li {
        display: grid;
        grid-template-columns: 14px 1fr;
        gap: 12px;
        align-items: start;
        color: var(--bone);
        font-size: 0.88rem;
        line-height: 1.35;
        margin-bottom: 0.85rem;
    }
    .pipeline li i {
        width: 14px; height: 14px; border-radius: 2px; margin-top: 3px;
    }
    .pipeline li small {
        display: block;
        color: var(--bone-dim);
        font-size: 0.78rem;
    }

    .model-note {
        font-family: var(--mono);
        font-size: 0.7rem;
        color: var(--bone-dim);
        line-height: 1.7;
        font-variant-numeric: tabular-nums;
        background: var(--soil-2);
        padding: 0.75rem 0.85rem;
        border-radius: 4px;
        margin-top: 0.5rem;
    }

    /* ---------- Page heading ---------- */
    .page-title {
        font-weight: 800;
        font-size: clamp(2.1rem, 4.2vw, 3.6rem);
        letter-spacing: -0.035em;
        line-height: 1.02;
        color: var(--ink);
        margin: 0 0 0.9rem;
        text-wrap: balance;
        max-width: 22ch;
    }
    .page-lead {
        font-size: 1.1rem;
        line-height: 1.55;
        color: var(--ink-2);
        max-width: 60ch;
        margin-bottom: 2rem;
    }

    .section-title {
        font-weight: 800;
        font-size: 1.55rem;
        letter-spacing: -0.02em;
        color: var(--ink);
        margin: 3rem 0 1.1rem;
    }

    /* ---------- Uploader as an empty chip page ---------- */
    [data-testid="stFileUploaderDropzone"] {
        background: var(--mat-deep);
        border: 1.5px dashed var(--mat-edge);
        border-radius: 4px;
        padding: 2.25rem 1.75rem;
        transition: background 160ms ease-out, border-color 160ms ease-out;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        background: #b4aea2;
        border-color: var(--soil);
    }
    [data-testid="stFileUploaderDropzone"] button {
        background: var(--soil);
        color: var(--bone);
        border: none;
        font-weight: 600;
    }

    .ghost-chips {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 1.25rem;
        margin-top: 1.75rem;
    }
    .ghost-chip .sw {
        aspect-ratio: 1;
        background: var(--mat-deep);
        border-radius: 2px;
        display: grid;
        place-items: center;
        font-family: var(--mono);
        font-size: 0.8rem;
        color: var(--ink-2);
    }
    .ghost-chip .sw-strip {
        background: var(--mat-deep);
        border-radius: 2px;
    }
    .ghost-chip p {
        font-family: var(--mono);
        font-size: 0.72rem;
        letter-spacing: 0.04em;
        color: var(--ink-2);
        margin: 0.75rem 0 0.5rem;
    }
    .ghost-bar {
        height: 14px;
        background: var(--mat-deep);
        border-radius: 2px;
        margin-bottom: 0.5rem;
    }
    .ghost-bar.thin { height: 6px; }

    /* ---------- Crop page header ---------- */
    .crop-page { margin-bottom: 1.5rem; }
    .router-scores { margin-top: 0.5rem; }

    [data-testid="stRadio"] label p { font-weight: 600; }

    .review-note {
        background: var(--mat-deep);
        border-radius: 4px;
        padding: 1rem 1.15rem;
        margin: 0 0 1.25rem;
    }
    .review-note b {
        display: block;
        font-weight: 800;
        color: var(--ink);
        margin-bottom: 0.3rem;
    }
    .review-note p, .review-note li {
        font-size: 0.92rem;
        line-height: 1.55;
        color: var(--ink);
        margin: 0;
    }
    .review-note ul { margin: 0.3rem 0 0; padding-left: 1.1rem; }
    .verdict + div .review-note, .review-note.after-verdict { margin-top: 1rem; }
    /* Gemini assessment and Why / Next step boxes only */
    .review-note.ai-panel b:not(:first-child) { margin-top: 1rem; }
    .review-note.ai-panel p + p { margin-top: 0.5rem; }

    .review-pill {
        display: inline-block;
        vertical-align: middle;
        font-family: var(--mono);
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--ink);
        background: var(--bone);
        border-radius: 2px;
        padding: 3px 8px;
        margin-left: 0.75rem;
    }

    .chip .chip-agree {
        font-family: var(--sans);
        font-size: 0.78rem;
        font-weight: 600;
        color: var(--ink-2);
        margin: 0.5rem 0 0;
    }
    .chip .chip-agree.differs { color: var(--rust); }

    .crop-name {
        font-weight: 800;
        font-size: 2.6rem;
        letter-spacing: -0.03em;
        line-height: 1;
        color: var(--ink);
    }
    .notation {
        font-family: var(--mono);
        font-size: 0.78rem;
        letter-spacing: 0.04em;
        color: var(--ink-2);
        font-variant-numeric: tabular-nums;
    }
    .meter {
        height: 6px;
        background: var(--mat-edge);
        border-radius: 1px;
        margin-top: 0.5rem;
        overflow: hidden;
    }
    .meter b { display: block; height: 100%; }

    /* ---------- Field conditions strip ---------- */
    .conditions {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 1px;
        background: var(--mat-edge);
        border-radius: 4px;
        overflow: hidden;
        margin: 0.25rem 0 1.5rem;
    }
    .conditions div {
        background: var(--mat-deep);
        padding: 0.8rem 0.9rem;
    }
    .conditions span {
        display: block;
        font-family: var(--mono);
        font-size: 0.66rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--ink-2);
        margin-bottom: 0.3rem;
    }
    .conditions b {
        font-weight: 700;
        font-size: 1.02rem;
        color: var(--ink);
        font-variant-numeric: tabular-nums;
    }
    .conditions.two { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .conditions-source, .map-note {
        font-size: 0.8rem;
        line-height: 1.5;
        color: var(--ink-2);
        margin: -0.9rem 0 1.25rem;
        max-width: 70ch;
    }
    .map-note { margin: 0.5rem 0 0.25rem; }
    .map-subhead {
        font-weight: 800;
        font-size: 1.05rem;
        color: var(--ink);
        margin: 1.5rem 0 0.25rem;
    }
    .saved-field {
        font-size: 0.9rem;
        color: var(--ink);
        margin: 0.4rem 0;
        font-variant-numeric: tabular-nums;
    }
    [data-testid="stExpander"] details {
        border-color: var(--mat-edge);
        background: var(--mat);
    }
    [data-testid="stExpander"] summary p { font-weight: 700; }

    .conditions-empty {
        font-size: 0.92rem;
        color: var(--ink-2);
        margin: 0.25rem 0 1.5rem;
    }

    /* ---------- Buttons ---------- */
    .stButton > button, [data-testid="stBaseButton-primary"] {
        border-radius: 4px;
        font-weight: 700;
        font-size: 1rem;
        padding: 0.8rem 1.4rem;
        transition: transform 120ms ease-out, background 160ms ease-out;
    }
    [data-testid="stBaseButton-primary"] {
        background: var(--soil);
        border: none;
        color: var(--bone);
        box-shadow: 0 2px 6px rgba(46, 36, 27, 0.25);
    }
    [data-testid="stBaseButton-primary"]:hover {
        background: #43352a;
        color: var(--bone);
    }
    [data-testid="stBaseButton-primary"]:active { transform: translateY(1px); }
    button:focus-visible {
        outline: 2px solid var(--soil) !important;
        outline-offset: 2px;
    }

    /* ---------- Images ---------- */
    [data-testid="stImage"] img { border-radius: 3px; }
    .stMarkdown p.leaf-caption, p.leaf-caption {
        font-family: var(--mono);
        font-size: 0.72rem !important;
        letter-spacing: 0.03em;
        color: var(--ink-2);
        font-variant-numeric: tabular-nums;
        margin: 0.6rem 0 0;
    }

    /* ---------- Quadrant chips ---------- */
    .chip-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 1.75rem 1.5rem;
        margin-bottom: 1.5rem;
    }
    .chip-mount {
        display: grid;
        grid-template-columns: 1fr 22%;
        gap: 6px;
    }
    .chip-mount img {
        width: 100%;
        aspect-ratio: 1;
        object-fit: cover;
        border-radius: 2px;
        display: block;
    }
    .chip-tone { border-radius: 2px; }
    .chip-id {
        font-family: var(--mono);
        font-size: 0.7rem;
        letter-spacing: 0.06em;
        color: var(--ink-2);
        margin: 0.75rem 0 0.2rem;
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        gap: 0 0.75rem;
    }
    .chip-score-label { font-family: var(--sans); font-weight: 600; }
    .chip {
        animation: chip-mount 520ms cubic-bezier(0.16, 1, 0.3, 1) both;
    }
    .chip:nth-child(2) { animation-delay: 90ms; }
    .chip:nth-child(3) { animation-delay: 180ms; }
    .chip:nth-child(4) { animation-delay: 270ms; }
    @keyframes chip-mount {
        from { opacity: 0; transform: translateY(10px); clip-path: inset(0 0 18% 0); }
        to { opacity: 1; transform: none; clip-path: inset(0 0 0 0); }
    }
    @media (prefers-reduced-motion: reduce) {
        .chip { animation: none; }
    }
    .chip-class {
        font-weight: 800;
        font-size: 1.15rem;
        letter-spacing: -0.01em;
        line-height: 1.2;
        color: var(--ink);
    }
    .chip-kind {
        display: inline-block;
        font-family: var(--mono);
        font-size: 0.66rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #fff;
        padding: 2px 6px;
        border-radius: 2px;
        margin: 0.4rem 0 0.1rem;
    }
    .chip-score {
        display: flex;
        justify-content: space-between;
        font-family: var(--mono);
        font-size: 0.8rem;
        color: var(--ink);
        font-variant-numeric: tabular-nums;
        margin-top: 0.5rem;
    }
    .runners {
        list-style: none;
        padding: 0;
        margin: 0.6rem 0 0;
        font-family: var(--mono);
        font-size: 0.7rem;
        color: var(--ink-2);
        font-variant-numeric: tabular-nums;
        line-height: 1.6;
    }
    .chip .runners li {
        display: flex;
        justify-content: space-between;
        gap: 8px;
        margin: 0;
        padding: 0;
        font-family: var(--mono);
        font-size: 0.7rem;
        line-height: 1.6;
        color: var(--ink-2);
    }
    .sheet-status {
        font-size: 1.02rem;
        line-height: 1.6;
        color: var(--ink);
        max-width: 64ch;
        margin: 0 0 1rem;
    }
    .st-key-prescription [data-testid="stExpander"] details {
        border-color: var(--mat-edge);
        background: transparent;
    }

    /* ---------- Verdict band ---------- */
    .verdict {
        color: #fff;
        border-radius: 4px;
        padding: 1.6rem 1.75rem;
        margin: 0.5rem 0 0;
    }
    .verdict h3 {
        font-family: var(--sans);
        font-weight: 800;
        font-size: clamp(1.5rem, 2.6vw, 2.1rem);
        letter-spacing: -0.025em;
        line-height: 1.1;
        margin: 0 0 0.5rem;
        padding: 0;
        color: #fff;
    }
    .verdict p.run-details {
        margin-top: 0.75rem;
        padding-top: 0.75rem;
        border-top: 1px solid rgba(255, 255, 255, 0.28);
        color: rgba(255, 255, 255, 0.92);
    }
    .verdict p {
        font-family: var(--mono);
        font-size: 0.75rem;
        letter-spacing: 0.05em;
        margin: 0;
        color: rgba(255, 255, 255, 0.88);
        font-variant-numeric: tabular-nums;
    }

    /* ---------- Prescription sheet ---------- */
    .st-key-prescription {
        background: #f1ece3;
        border-radius: 4px;
        padding: 2rem 2.25rem 1.5rem;
        margin-top: 1.5rem;
        box-shadow: 0 6px 18px rgba(35, 27, 20, 0.14);
    }
    .st-key-prescription p, .st-key-prescription li {
        font-size: 1.02rem;
        line-height: 1.65;
        color: var(--ink);
        max-width: 72ch;
    }
    .st-key-prescription h1, .st-key-prescription h2,
    .st-key-prescription h3, .st-key-prescription h4 {
        font-family: var(--sans);
        font-weight: 800;
        letter-spacing: -0.015em;
        color: var(--ink);
    }
    .sheet-head {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        gap: 1rem;
        flex-wrap: wrap;
        border-bottom: 1px solid var(--mat-edge);
        padding-bottom: 0.9rem;
        margin-bottom: 1.1rem;
    }
    .sheet-head b {
        font-weight: 800;
        font-size: 1.45rem;
        letter-spacing: -0.02em;
        color: var(--ink);
    }

    /* ---------- Alerts ---------- */
    [data-testid="stAlert"] { border-radius: 4px; }

    @media (max-width: 640px) {
        /* The header strip holds the sidebar toggle: make it opaque so
           scrolled content passes under it, and start content below it. */
        [data-testid="stHeader"] { background: var(--mat); }
        .block-container { padding-top: 4.5rem; }
        .ghost-chips, .chip-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1.25rem 1rem; }
        .chip-class { font-size: 1rem; }
        .chip-region { display: none; }
        .conditions { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .crop-name { font-size: 2.1rem; }
        .st-key-prescription { padding: 1.4rem 1.25rem 1rem; }
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# GOOGLE DRIVE MODEL IDS
# ============================================================

ROUTER_MODEL_ID = "10LuRi-n6wFu4um9deAWBu0sr0-8e1MLV"
CORN_MODEL_ID = "1_1PcQqUFFiK9tgpXwivM6J7OJShL18jk"
# rice_weed_robust.tflite: 8 outputs, EfficientNetV2-B0 with built-in
# rescaling; accuracy not validated (see UNVALIDATED_MODELS).
RICE_MODEL_ID = "1p2vZgq_FBigVnlhQPLQD4w2yjDn4zus3"

# Make sure this is the Google Drive file ID of the NEW Cotton model.
COTTON_MODEL_ID = "1tX19vWEBa9pyZCd_7hIGC8j2fwrHQEie"


# ============================================================
# LOCAL MODEL FILENAMES
# ============================================================

ROUTER_MODEL_PATH = Path("crop_router_v2.tflite")
CORN_MODEL_PATH = Path("corn_model.tflite")
RICE_MODEL_PATH = Path("rice_model.tflite")
COTTON_MODEL_PATH = Path("cotton_doctor_latest.tflite")

MODEL_FILES = {
    "Maize": CORN_MODEL_PATH,
    "Rice": RICE_MODEL_PATH,
    "Cotton": COTTON_MODEL_PATH
}


# ============================================================
# FIELD CONDITION OPTIONS
# ============================================================

SOIL_TYPES = [
    "Not specified",
    "Alluvial",
    "Black (Regur)",
    "Red",
    "Laterite",
    "Clay",
    "Loam",
    "Sandy",
    "Silt",
    "Saline / Alkaline"
]

MOISTURE_AUTO = "Auto (from forecast)"

MOISTURE_LEVELS = [
    MOISTURE_AUTO,
    "Dry",
    "Moist",
    "Wet",
    "Waterlogged"
]

CROP_SWATCHES = {
    "Maize": "#c3933f",
    "Rice": "#8a9a5a",
    "Cotton": "#d9cfbf"
}

STATE_COLORS = {
    "disease": "#9a3a2a",
    "weed": "#8a6420",
    "healthy": "#46692a",
    "uncertain": "#5c4d3d"
}

OVERLAY_COLOR = (224, 177, 46)


# ============================================================
# MODEL DOWNLOAD HELPERS
# ============================================================

def open_interpreter(model_path, classes, model_name):
    interpreter = Interpreter(
        model_path=str(model_path)
    )
    interpreter.allocate_tensors()
    check_output_classes(interpreter, classes, model_name)
    return interpreter


# ============================================================
# CACHED MODEL LOADER
# Router, Maize and Rice remain cached.
# ============================================================

@st.cache_resource(show_spinner=False)
def cached_load_model(model_key):
    """
    Load Router, Maize or Rice model.

    These models are cached by Streamlit and their downloaded files are reused.
    """

    model_config = {
        "Router": {
            "file_id": ROUTER_MODEL_ID,
            "path": ROUTER_MODEL_PATH,
            "classes": ROUTER_CLASSES
        },
        "Maize": {
            "file_id": CORN_MODEL_ID,
            "path": CORN_MODEL_PATH,
            "classes": CORN_CLASSES
        },
        "Rice": {
            "file_id": RICE_MODEL_ID,
            "path": RICE_MODEL_PATH,
            "classes": RICE_CLASSES
        }
    }

    if model_key not in model_config:
        raise ValueError(f"Unsupported cached model key: {model_key}")

    config = model_config[model_key]

    with st.spinner(f"Loading {model_key} model..."):
        model_path = download_model(
            file_id=config["file_id"],
            destination=config["path"],
            force_download=False
        )

        return open_interpreter(model_path, config["classes"], model_key)


# ============================================================
# COTTON MODEL LOADER
# Cotton is deliberately NOT cached.
# ============================================================

def load_fresh_cotton_model():
    """
    Always download and load the latest Cotton model.

    This function is not decorated with st.cache_resource.
    The previous local Cotton model is replaced on every diagnosis, but
    only by a download that passes validation; if the refresh fails, the
    last working copy is used and the sidebar says so.
    """

    refresh_error = None

    with st.spinner("Downloading the latest Cotton model..."):
        try:
            model_path = download_model(
                file_id=COTTON_MODEL_ID,
                destination=COTTON_MODEL_PATH,
                force_download=True
            )
        except ModelFileError as error:
            if not COTTON_MODEL_PATH.exists():
                raise
            refresh_error = error
            model_path = validate_tflite_file(COTTON_MODEL_PATH)

    interpreter = open_interpreter(model_path, COTTON_CLASSES, "Cotton")

    input_details = interpreter.get_input_details()
    output_shape = interpreter.get_output_details()[0]["shape"]

    st.session_state["cotton_model_note"] = (
        f"{'COTTON MODEL · REFRESH FAILED, USING LAST COPY' if refresh_error else 'FRESH COTTON MODEL'}<br>"
        f"file  {model_path.name}<br>"
        f"size  {model_path.stat().st_size / (1024 * 1024):.2f} MB<br>"
        f"in    {list(input_details[0]['shape'])}<br>"
        f"out   {list(output_shape)}"
    )

    return interpreter


def load_model(model_key):
    """
    Main model loader.

    Cotton:
        Always freshly downloaded and not cached.

    Router, Maize and Rice:
        Loaded through Streamlit resource cache.
    """
    if model_key == "Cotton":
        return load_fresh_cotton_model()

    return cached_load_model(model_key)


# ============================================================
# WEATHER AND SOIL MOISTURE
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def geocode(query):
    """
    Place search via the Open-Meteo geocoding API. Returns None when
    nothing matches or the service is unreachable.
    """
    try:
        response = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": query, "count": 1},
            timeout=15
        )
        response.raise_for_status()
        results = response.json().get("results") or []
    except (requests.RequestException, ValueError):
        return None

    if not results:
        return None

    place = results[0]

    return {
        "lat": float(place["latitude"]),
        "lon": float(place["longitude"]),
        "label": ", ".join(
            part for part in [
                place.get("name"),
                place.get("admin1"),
                place.get("country_code")
            ]
            if part
        )
    }


@st.cache_data(ttl=600, show_spinner=False)
def get_conditions(lat, lon):
    """
    Current weather and modelled soil moisture for the field point from
    the Open-Meteo forecast API. Returns None when unavailable.
    """
    try:
        response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": round(lat, 4),
                "longitude": round(lon, 4),
                "current": (
                    "temperature_2m,"
                    "relative_humidity_2m,"
                    "precipitation,"
                    "weather_code"
                ),
                "hourly": "soil_moisture_3_to_9cm",
                "forecast_days": 1,
                "timezone": "auto"
            },
            timeout=15
        )
        response.raise_for_status()
        return parse_weather(response.json())
    except (requests.RequestException, ValueError, KeyError, TypeError):
        return None


# Browser Geolocation API, with a timeout so an unavailable fix cannot
# hang the page. Runs only after the person presses the button.
GEOLOCATION_JS = """
new Promise((resolve) => {
  if (!navigator.geolocation) { resolve({error: {code: 0}}); return; }
  navigator.geolocation.getCurrentPosition(
    (p) => resolve({coords: {latitude: p.coords.latitude,
                             longitude: p.coords.longitude,
                             accuracy: p.coords.accuracy},
                    timestamp: p.timestamp}),
    (e) => resolve({error: {code: e.code}}),
    {enableHighAccuracy: true, timeout: 10000, maximumAge: 60000}
  );
})
"""

FIELD_SOURCES = {
    "gps": "from this device's location",
    "search": "from place search",
    "map": "set on the map"
}


# ============================================================
# PRESENTATION HELPERS
# ============================================================

def image_to_data_uri(image, max_side=360):
    preview = image.convert("RGB")
    preview.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    preview.save(buffer, format="JPEG", quality=85)
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/jpeg;base64,{encoded}"


def mean_tone(image):
    """
    The quadrant's mean colour, mounted beside it like a soil chip.
    """
    sample = image.convert("RGB").resize((24, 24))
    red, green, blue = np.asarray(sample).reshape(-1, 3).mean(axis=0)
    return f"#{int(red):02x}{int(green):02x}{int(blue):02x}"


def draw_quadrant_overlay(image, results=None):
    """
    Draw the quadrant grid and Q1-Q4 tags in the reserved overlay colour.
    """
    canvas = image.convert("RGB").copy()
    width, height = canvas.size
    draw = ImageDraw.Draw(canvas)

    line_width = max(2, min(width, height) // 160)
    draw.line(
        [(width // 2, 0), (width // 2, height)],
        fill=OVERLAY_COLOR,
        width=line_width
    )
    draw.line(
        [(0, height // 2), (width, height // 2)],
        fill=OVERLAY_COLOR,
        width=line_width
    )

    font_size = max(14, min(width, height) // 26)

    try:
        font = ImageFont.load_default(size=font_size)
    except TypeError:
        font = ImageFont.load_default()

    padding = font_size // 2
    anchors = [
        (padding, padding),
        (width // 2 + padding, padding),
        (padding, height // 2 + padding),
        (width // 2 + padding, height // 2 + padding)
    ]

    for index, (x, y) in enumerate(anchors):
        tag = f"Q{index + 1}"
        box = draw.textbbox((x, y), tag, font=font)
        draw.rectangle(
            [
                box[0] - padding // 2,
                box[1] - padding // 2,
                box[2] + padding // 2,
                box[3] + padding // 2
            ],
            fill=OVERLAY_COLOR
        )
        draw.text((x, y), tag, fill=(35, 27, 20), font=font)

    return canvas


def render_conditions(weather, soil_type, moisture_label):
    cells = []

    if weather:
        cells += [
            ("Air temp", weather["temp"] or "Unavailable"),
            ("Humidity", weather["humidity"] or "Unavailable"),
            ("Rainfall", weather["precip"] or "Unavailable"),
            ("Sky", weather["condition"] or "Unavailable")
        ]

    cells += [
        ("Soil moisture", moisture_label or "Unknown"),
        ("Soil type", soil_type)
    ]

    body = "".join(
        f"<div><span>{html.escape(name)}</span>"
        f"<b>{html.escape(str(value))}</b></div>"
        for name, value in cells
    )

    columns_class = " two" if len(cells) == 2 else ""

    source = ""

    if weather:
        observed = (weather.get("observed") or "").replace("T", " ")
        source = (
            f'<p class="conditions-source">{html.escape(weather["source"])} · '
            f'current values at {html.escape(observed)} '
            f'{html.escape(weather.get("timezone") or "")}. Soil moisture is a '
            'modelled estimate at 3–9 cm depth, not a field measurement. Soil '
            'type is what you entered.</p>'
        )

    st.markdown(
        f'<div class="conditions{columns_class}">{body}</div>{source}',
        unsafe_allow_html=True
    )


def chip_html(index, quadrant_name, result, whole_label=None):
    label = result["label"]
    kind = label_kind(label)
    color = STATE_COLORS[kind]
    confidence = min(result["conf"], 100.0)

    if whole_label is None:
        agreement = ""
    elif label == whole_label:
        agreement = '<p class="chip-agree">Matches whole leaf</p>'
    else:
        agreement = (
            '<p class="chip-agree differs">Differs from whole leaf</p>'
        )

    runners = "".join(
        f"<li><span>{html.escape(pretty_label(name))}</span>"
        f"<span>{score:5.1f}%</span></li>"
        for name, score in result["top"][1:]
    )

    return f"""
        <div class="chip">
            <div class="chip-mount">
                <img src="{image_to_data_uri(result['img'])}"
                     alt="{html.escape(quadrant_name)} quadrant">
                <div class="chip-tone"
                     style="background:{mean_tone(result['img'])}"></div>
            </div>
            <div class="chip-id">
                <span>Q{index + 1}<span class="chip-region"> · {html.escape(quadrant_name.upper())}</span></span>
                <span>{mean_tone(result['img'])}</span>
            </div>
            <div class="chip-class">{html.escape(pretty_label(label))}</div>
            <span class="chip-kind" style="background:{color}">{kind}</span>
            <div class="chip-score">
                <span class="chip-score-label">Confidence</span><span>{confidence:.1f}%</span>
            </div>
            <div class="meter">
                <b style="width:{confidence:.1f}%;background:{color}"></b>
            </div>
            <ul class="runners">{runners}</ul>
            {agreement}
        </div>
        """


# ============================================================
# AI AGRONOMIST (GEMINI)
# ============================================================

def ai_settings():
    """GEMINI_API_KEY and model name; the key is never displayed."""
    try:
        secrets = st.secrets
    except Exception:
        secrets = {}

    return (
        resolve_api_key(secrets, os.environ),
        resolve_gemini_model(secrets, os.environ)
    )


def get_smart_advice(
    disease_detected,
    weed_detected,
    weather,
    location,
    crop,
    soil_type,
    soil_moisture,
    quadrant_results,
    whole_leaf=None,
    review_reasons=(),
    gemini_assessment=None,
    comparison=None
):
    """
    Returns {"ok": True, "text": ...} or {"ok": False, kind, message,
    detail, blocks_session}. Errors are described, never echoed raw.
    """
    try:
        api_key, model = ai_settings()

        humidity = "Not available"
        precipitation = "Not available"
        weather_text = "Weather information is unavailable"

        if weather:
            weather_text = (
                f"Temperature: {weather.get('temp', 'N/A')}, "
                f"Humidity: {weather.get('humidity', 'N/A')}, "
                f"Rain: {weather.get('precip', 'N/A')}, "
                f"Sky: {weather.get('condition', 'N/A')}"
            )
            humidity = weather.get("humidity", "Not available")
            precipitation = weather.get("precip", "Not available")

        issues = []

        if disease_detected:
            issues.append(
                f"Disease ({pretty_label(disease_detected)})"
            )

        if weed_detected:
            issues.append(
                f"Weed ({pretty_label(weed_detected)})"
            )

        diagnosis = (
            " AND ".join(issues)
            if issues
            else "Healthy Crop"
        )

        candidate_lines = "\n".join(
            f"- {name}: "
            + ", ".join(
                f"{pretty_label(label)} {score:.1f}%"
                for label, score in result["top"]
            )
            for name, result in quadrant_results.items()
        )

        whole_line = (
            f"{pretty_label(whole_leaf['label'])} {whole_leaf['conf']:.1f}%"
            if whole_leaf
            else "Not available"
        )

        review_line = (
            "The app flagged this result for review: "
            + " ".join(review_reasons)
            + " Say clearly that the diagnosis is uncertain."
            if review_reasons
            else ""
        )

        # Analyze All passes Gemini's earlier image assessment as text, so
        # the image is not sent a second time.
        if gemini_assessment and comparison:
            second_opinion = (
                "Independent visual assessment of the same photo (made "
                "without seeing the image-model result): "
                f"crop {gemini_assessment['crop']}, condition "
                f"{gemini_assessment['condition'] or 'not named'}, "
                f"certainty {gemini_assessment['certainty']}. Symptoms: "
                f"{'; '.join(gemini_assessment['symptoms']) or 'none listed'}. "
                f"Alternatives: "
                f"{'; '.join(gemini_assessment['alternatives']) or 'none listed'}.\n"
                f"Relationship between the two: {comparison['category']}. "
                + " ".join(comparison["reasons"])
            )
        else:
            second_opinion = ""

        prompt = f"""
You are an expert agronomist AI.

Crop: {crop}
Detected issues: {diagnosis} in {location or "Unknown"}
Weather: {weather_text}
Soil type: {soil_type}
Soil moisture: {soil_moisture or "Unknown"}

Whole-leaf reading: {whole_line}
Top three image-model candidates for each leaf quadrant:
{candidate_lines}
{review_line}
{second_opinion}

Provide a concise and safe management plan with these sections:

1. DIAGNOSIS SUMMARY
2. FIELD CONDITIONS CHECK
Use the soil type, soil moisture, humidity and rainfall to judge which
candidate issue is most consistent with field conditions. If conditions
argue against the top prediction, or favour a runner-up, say so plainly
and name the most likely specific issue.
3. WEATHER CHECK
Explain whether spraying is appropriate with humidity of
{humidity} and precipitation of {precipitation}.
4. ACTION PLAN
Suggest suitable management options. Clearly state that local
agricultural guidance and product labels must be followed.
Explain whether fungicides and herbicides should be mixed.
5. ORGANIC ALTERNATIVE

Do not present the AI image classification as a confirmed
laboratory diagnosis. Do not give pesticide doses, mixing rates or
product application rates, do not guarantee recovery, and recommend
only treatment types that are generally supported for this crop and
condition. Recommend that a local agricultural extension officer or
expert verifies the diagnosis before any chemical treatment.
Keep the answer structured and concise.
Use markdown headings for the sections.
"""

        return request_advice(prompt, api_key, model)

    except Exception as error:
        return {"ok": False, **describe_advice_error(error)}


# ============================================================
# ANALYZE ALL: GEMINI ASSESSMENT AND COMBINED INTERPRETATION
# ============================================================

COMPARISON_COLORS = {
    "Agreement": STATE_COLORS["healthy"],
    "Partial agreement": STATE_COLORS["weed"],
    "Disagreement": STATE_COLORS["disease"],
    "Uncertain": STATE_COLORS["uncertain"]
}

NOT_AI_CHECKED = (
    "The ML result above is shown on its own; it has not been checked by "
    "Gemini."
)


def html_list(items, empty="None reported."):
    if not items:
        return f"<p>{html.escape(empty)}</p>"
    return "<ul>" + "".join(
        f"<li>{html.escape(item)}</li>" for item in items
    ) + "</ul>"


def render_ai_analysis(ai, summary):
    """
    Gemini's independent assessment, then the combined interpretation.
    Gemini text is untrusted and always escaped. The ML result is
    rendered elsewhere and never changed here.
    """
    if ai["assessment"] is None:
        error = ai["error"]

        if error is None:
            st.info(
                "The AI assessment is turned off in the sidebar. "
                + NOT_AI_CHECKED
            )
            return

        st.markdown(
            f"""
            <div class="review-note">
                <b>Independent AI assessment unavailable</b>
                <p>{html.escape(error["message"])}</p>
                <p>{html.escape(NOT_AI_CHECKED)}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.expander("AI error detail"):
            st.code(error["detail"], language=None, wrap_lines=True)

        return

    assessment = ai["assessment"]
    comparison = ai["comparison"]

    label = assessment["closest_label"]
    label_text = (
        pretty_label(label) if label not in ("Other", "Unknown") else label
    )
    condition = assessment["condition"] or label_text

    st.markdown(
        '<div class="section-title">Independent AI assessment · Gemini, '
        'from the photo alone</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="review-note ai-panel">
            <b>{html.escape(condition)}</b>
            <p>Crop seen: {html.escape(assessment["crop"])} · Closest app
            label: {html.escape(label_text)} · Gemini's own certainty:
            {html.escape(assessment["certainty"])}
            {" · may be more than one condition"
             if assessment["multiple_conditions"] else ""}</p>
            <p>{html.escape(assessment["summary"])}</p>
            <b>Visible symptoms</b>{html_list(assessment["symptoms"])}
            <b>Supporting features</b>{html_list(assessment["supporting_features"])}
            <b>Could also be</b>{html_list(assessment["alternatives"])}
            <b>Image quality</b>{html_list(
                assessment["image_quality_issues"], "No problems reported."
            )}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="verdict" style="background:{COMPARISON_COLORS[comparison["category"]]}">
            <h3>Combined interpretation: {html.escape(comparison["category"])}</h3>
            <p>ML MODEL {html.escape(pretty_label(summary["primary"]).upper())}
            {summary["primary_conf"]:.1f}% (MODEL SCORE) · GEMINI
            {html.escape(condition.upper())}, {html.escape(assessment["certainty"].upper())}
            CERTAINTY (ITS OWN RATING) · REPORTED SEPARATELY, NEVER AVERAGED</p>
        </div>
        <div class="review-note ai-panel after-verdict">
            <b>Why</b>
            {html_list(comparison["reasons"])}
            <b>Next step</b>
            <p>{html.escape(comparison["next_step"])}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <div class="wordmark">Agri-Doctor Pro</div>
    <div class="wordmark-note">
        LEAF PATHOLOGY · QUADRANT SCAN<br>MAIZE / RICE / COTTON
    </div>
    """,
    unsafe_allow_html=True
)

user_location = st.sidebar.text_input(
    "Farm location",
    placeholder="Search a village, town or city"
)

if user_location and user_location != st.session_state.get("last_query"):
    st.session_state["last_query"] = user_location
    place = geocode(user_location)

    if place:
        st.session_state["field_point"] = {
            **place,
            "source": "search",
            "accuracy": None
        }
        st.session_state.pop("geo_message", None)
    else:
        st.session_state["geo_message"] = (
            "No place found for that search, or the search service is "
            "unavailable."
        )

if streamlit_js_eval is not None:
    if st.sidebar.button("Detect my location", width="stretch"):
        st.session_state["locate_token"] = int(time.time() * 1000)
        st.session_state.pop("geo_message", None)

    locate_token = st.session_state.get("locate_token")

    if locate_token:
        with st.sidebar:
            geo_result = streamlit_js_eval(
                js_expressions=GEOLOCATION_JS,
                key=f"geolocate_{locate_token}"
            )

        if geo_result is None:
            st.sidebar.markdown(
                '<div class="rail-reading">Waiting for your browser to share '
                'its location. Allow the permission prompt if one appears.'
                '</div>',
                unsafe_allow_html=True
            )

            if st.sidebar.button("Cancel"):
                st.session_state.pop("locate_token", None)
                st.rerun()
        else:
            st.session_state.pop("locate_token", None)
            located = parse_geolocation(geo_result)

            if located["ok"]:
                accuracy = located["accuracy"]
                st.session_state["field_point"] = {
                    "lat": located["lat"],
                    "lon": located["lon"],
                    "label": "Device location",
                    "source": "gps",
                    "accuracy": accuracy
                }
            else:
                st.session_state["geo_message"] = located["message"]

            st.rerun()

field_point = st.session_state.get("field_point")

if st.session_state.get("geo_message"):
    st.sidebar.markdown(
        f'<div class="rail-reading">{html.escape(st.session_state["geo_message"])}</div>',
        unsafe_allow_html=True
    )

if field_point:
    accuracy_note = (
        f" · ±{field_point['accuracy']:.0f} m"
        if field_point.get("accuracy")
        else ""
    )
    st.sidebar.markdown(
        f'<div class="rail-reading"><b>{html.escape(field_point["label"])}</b><br>'
        f'{field_point["lat"]:.5f}, {field_point["lon"]:.5f} '
        f'{FIELD_SOURCES[field_point["source"]]}{accuracy_note}</div>',
        unsafe_allow_html=True
    )

weather_data = (
    get_conditions(field_point["lat"], field_point["lon"])
    if field_point
    else None
)

if field_point and weather_data is None:
    st.sidebar.markdown(
        '<div class="rail-reading">Weather service unavailable. Weather and '
        'soil moisture are not shown.</div>',
        unsafe_allow_html=True
    )

soil_type = st.sidebar.selectbox(
    "Soil type",
    SOIL_TYPES
)

moisture_choice = st.sidebar.selectbox(
    "Soil moisture",
    MOISTURE_LEVELS
)

auto_band = weather_data["soil_moisture_band"] if weather_data else None
auto_reading = weather_data["soil_moisture"] if weather_data else None

if moisture_choice == MOISTURE_AUTO:
    if auto_band:
        soil_moisture = (
            f"{auto_band} ({auto_reading:.2f} m³/m³ at 3–9 cm, forecast)"
        )
        st.sidebar.markdown(
            f'<div class="rail-reading">Forecast reading '
            f'<b>{auto_reading:.2f} m³/m³</b> · {auto_band}</div>',
            unsafe_allow_html=True
        )
    else:
        soil_moisture = None
        st.sidebar.markdown(
            '<div class="rail-reading">Enter a farm location to read soil '
            'moisture, or choose a level.</div>',
            unsafe_allow_html=True
        )
else:
    soil_moisture = f"{moisture_choice} (reported by farmer)"

moisture_display = (
    soil_moisture.split(" (")[0] if soil_moisture else None
)

if moisture_display and moisture_choice == MOISTURE_AUTO and auto_reading:
    moisture_display = f"{auto_band} · {auto_reading:.2f}"

enable_ai = st.sidebar.toggle(
    "Enable AI agronomist",
    value=True
)

# A session pause belongs to the key it was recorded with. Without this, a
# "no key" pause outlived the key being added, until the tab was reloaded.
if not block_applies(st.session_state.get("advice_blocked"), ai_settings()[0]):
    st.session_state.pop("advice_blocked", None)

# Filled at the end of the run, after any diagnosis in this run has had
# the chance to pause prescriptions.
advice_status_slot = st.sidebar.empty()

st.sidebar.markdown(
    f"""
    <div class="rail-heading">How a leaf is read</div>
    <ul class="pipeline">
        <li><i style="background:#d8c49a"></i>
            <span>Crop router<small>Suggests Maize, Rice or Cotton. You
            confirm it; maize and rice are often confused.</small></span></li>
        <li><i style="background:#8c7a62"></i>
            <span>Crop model<small>A dedicated model per crop for diseases
            and weeds.</small></span></li>
        <li><i style="background:#{''.join(f'{c:02x}' for c in OVERLAY_COLOR)}"></i>
            <span>Whole leaf, then quadrants<small>The whole leaf gives the
            verdict; each quarter is read to check it.</small></span></li>
        <li><i style="background:var(--bone)"></i>
            <span>AI agronomist (Gemini)<small>Analyze All adds an
            independent look at the photo, compares it with the model, and
            writes a cautious plan using weather, soil and
            moisture.</small></span></li>
    </ul>
    """,
    unsafe_allow_html=True
)

st.sidebar.caption(
    "Router, Maize and Rice are cached. "
    "Cotton is freshly downloaded whenever Cotton diagnosis runs."
)

if st.session_state.get("cotton_model_note"):
    st.sidebar.markdown(
        f'<div class="model-note">{st.session_state["cotton_model_note"]}</div>',
        unsafe_allow_html=True
    )


# ============================================================
# MAIN APPLICATION
# ============================================================

title_slot = st.empty()

page_title = "Read a leaf, quadrant by quadrant."
page_lead = (
    "Upload a photo of a maize, rice or cotton leaf. Agri-Doctor Pro "
    "identifies the crop, scans each quarter of the leaf for disease and "
    "weeds, and checks the result against your field's weather, soil type "
    "and moisture."
)

# ============================================================
# FIELD MAP
# ============================================================

def render_field_map():
    point = st.session_state.get("field_point")
    boundary = st.session_state.get("field_boundary")

    if folium is None:
        st.info("The map needs the folium and streamlit-folium packages.")
        return

    if point:
        center, zoom = [point["lat"], point["lon"]], 15
    else:
        center, zoom = [20.59, 78.96], 4

    field_map = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles="OpenStreetMap",
        control_scale=True
    )

    if point:
        folium.Marker(
            [point["lat"], point["lon"]],
            tooltip="Field point"
        ).add_to(field_map)

        if point.get("accuracy"):
            folium.Circle(
                [point["lat"], point["lon"]],
                radius=point["accuracy"],
                color="#2e241b",
                weight=1,
                fill=False
            ).add_to(field_map)

    if boundary:
        folium.Polygon(
            [[lat, lon] for lon, lat in boundary],
            color="#2e241b",
            weight=2,
            fill=True,
            fill_opacity=0.12,
            tooltip="Field boundary (drawn by you)"
        ).add_to(field_map)

    Draw(
        export=False,
        draw_options={
            "polyline": False,
            "rectangle": False,
            "circle": False,
            "circlemarker": False,
            "marker": False,
            "polygon": {"allowIntersection": False, "showArea": True}
        },
        edit_options={"edit": False, "remove": False}
    ).add_to(field_map)

    move_on_click = st.toggle(
        "Clicking the map moves the field point",
        value=True,
        help="Turn off while outlining the field with the polygon tool."
    )

    output = st_folium(
        field_map,
        height=420,
        use_container_width=True,
        returned_objects=["last_clicked", "all_drawings"],
        key="field_map"
    ) or {}

    clicked = output.get("last_clicked")

    if (
        move_on_click
        and clicked
        and clicked != st.session_state.get("last_map_click")
    ):
        st.session_state["last_map_click"] = clicked
        st.session_state["field_point"] = {
            "lat": float(clicked["lat"]),
            "lon": float(clicked["lng"]),
            "label": "Field point",
            "source": "map",
            "accuracy": None
        }
        st.rerun()

    polygons = [
        drawing for drawing in (output.get("all_drawings") or [])
        if (drawing.get("geometry") or {}).get("type") == "Polygon"
    ]

    if polygons:
        ring = polygons[-1]["geometry"]["coordinates"][0]

        if ring != boundary:
            st.session_state["field_boundary"] = ring
            st.rerun()

    notes = [
        "Click the map to set or correct the field point. Outline the field "
        "with the polygon tool (top left). Boundaries are drawn by you, not "
        "detected from imagery. Map data © OpenStreetMap contributors."
    ]

    if boundary:
        notes.append(
            f"Outlined area ≈ {polygon_area_hectares(boundary):.2f} ha "
            "(drawn by you, approximate)."
        )

    st.markdown(
        "".join(f'<p class="map-note">{html.escape(note)}</p>' for note in notes),
        unsafe_allow_html=True
    )

    if boundary and st.button("Clear boundary"):
        st.session_state.pop("field_boundary", None)
        st.rerun()

    render_saved_fields(point, boundary)


def render_saved_fields(point, boundary):
    fields = st.session_state.setdefault("saved_fields", [])

    st.markdown('<div class="map-subhead">Save this field</div>', unsafe_allow_html=True)

    with st.form("save_field", clear_on_submit=True, border=False):
        name_column, crop_column, date_column = st.columns([1.4, 1, 1])
        name = name_column.text_input("Field name", placeholder="e.g. North paddy")
        crop = crop_column.selectbox("Crop", ["Not set", "Maize", "Rice", "Cotton"])
        planted = date_column.date_input("Planting date", value=None)
        notes = st.text_input("Notes (optional)")
        submitted = st.form_submit_button("Save field", disabled=point is None)

    if point is None:
        st.markdown(
            '<p class="map-note">Set a field point first: search, detect '
            'your location, or click the map.</p>',
            unsafe_allow_html=True
        )

    if submitted and point:
        try:
            fields.append(
                field_record(
                    name, point["lat"], point["lon"],
                    None if crop == "Not set" else crop,
                    planted.isoformat() if planted else None,
                    boundary, notes
                )
            )
        except ValueError as error:
            st.warning(str(error))

    loaded = st.file_uploader(
        "Load saved fields (.json)",
        type=["json"],
        key="fields_json"
    )

    if loaded is not None and st.session_state.get("fields_json_id") != loaded.file_id:
        st.session_state["fields_json_id"] = loaded.file_id
        try:
            fields.extend(load_field_records(json.loads(loaded.getvalue())))
        except (ValueError, UnicodeDecodeError):
            st.warning("That file is not a saved-fields JSON file.")

    for index, record in enumerate(fields):
        label_column, use_column = st.columns([4, 1])
        label_column.markdown(
            f'<p class="saved-field"><b>{html.escape(record["name"])}</b> · '
            f'{record["lat"]:.5f}, {record["lon"]:.5f}'
            f'{" · " + html.escape(record["crop"]) if record["crop"] else ""}'
            f'{" · planted " + record["planting_date"] if record["planting_date"] else ""}'
            f'{" · boundary saved" if record["boundary"] else ""}</p>',
            unsafe_allow_html=True
        )

        if use_column.button("Use", key=f"use_field_{index}"):
            st.session_state["field_point"] = {
                "lat": record["lat"],
                "lon": record["lon"],
                "label": record["name"],
                "source": "map",
                "accuracy": None
            }
            if record["boundary"]:
                st.session_state["field_boundary"] = record["boundary"]
            else:
                st.session_state.pop("field_boundary", None)
            st.rerun()

    if fields:
        st.download_button(
            "Download saved fields (JSON)",
            data=json.dumps({"fields": fields}, indent=2),
            file_name="agri-doctor-fields.json",
            mime="application/json"
        )

    st.markdown(
        '<p class="map-note">Saved fields stay in this browser session only. '
        'Download the file to keep them; nothing is stored on the server.</p>',
        unsafe_allow_html=True
    )


with st.expander(
    "Field location and map",
    expanded=False
):
    render_field_map()

uploaded_file = st.file_uploader(
    "Upload leaf image",
    type=["jpg", "png", "jpeg"]
)

if not uploaded_file:
    st.markdown(
        """
        <div class="ghost-chips">
        """
        + "".join(
            f'''<div class="ghost-chip"><div class="chip-mount">
            <div class="sw"></div><div class="sw-strip"></div></div>
            <p>Q{index + 1}<span class="chip-region"> · {region}</span></p>
            <div class="ghost-bar" style="width:72%"></div>
            <div class="ghost-bar thin"></div></div>'''
            for index, region in enumerate(
                ["TOP-LEFT", "TOP-RIGHT", "BOTTOM-LEFT", "BOTTOM-RIGHT"]
            )
        )
        + "</div>",
        unsafe_allow_html=True
    )

if uploaded_file:
    upload_id = getattr(uploaded_file, "file_id", uploaded_file.name)

    if st.session_state.get("upload_id") != upload_id:
        st.session_state["upload_id"] = upload_id
        st.session_state.pop("diagnosis", None)

    try:
        image, image_notes = load_leaf_image(uploaded_file.getvalue())
    except ImageInputError as image_error:
        st.error(str(image_error))
        st.stop()

    for note in image_notes:
        st.caption(note)

    diagnosis = st.session_state.get("diagnosis")

    first_column, second_column = st.columns([1.05, 1], gap="large")

    with first_column:
        st.image(
            draw_quadrant_overlay(image),
            width="stretch"
        )
        st.markdown(
            f'<p class="leaf-caption">{html.escape(uploaded_file.name)} · '
            f'{image.width}×{image.height}px · four quadrants of '
            f'{image.width // 2}×{image.height // 2}px</p>',
            unsafe_allow_html=True
        )

    with second_column:
        try:
            router_interpreter = load_model("Router")

            route = route_crop(
                predict_image(
                    image=image,
                    interpreter=router_interpreter,
                    model_key="Router"
                )
            )

        except Exception as router_error:
            route = {
                "crop": None,
                "confidence": 0.0,
                "needs_review": True,
                "reason": f"The crop router could not run: {router_error}",
                "scores": {}
            }

        crops = list(CROP_SWATCHES)
        suggested_crop = route["crop"]

        default_crop = default_crop_choice(route)

        final_crop_choice = st.radio(
            "Crop",
            crops,
            index=(
                crops.index(default_crop)
                if default_crop in crops
                else None
            ),
            horizontal=True,
            key=f"crop_choice_{upload_id}"
        )

        route_status = crop_route_status(route, final_crop_choice)

        # A person chose the crop, so the router's doubt is settled.
        crop_set_manually = route_status in ("SET MANUALLY", "CONFIRMED")

        if suggested_crop:
            route_notation = (
                f"ROUTER SUGGESTS {suggested_crop.upper()} · "
                f"{route['confidence']:.1f}%"
            )
        else:
            route_notation = "ROUTER UNAVAILABLE"

        router_scores = " · ".join(
            f"{name.upper()} {score:.1f}%"
            for name, score in route["scores"].items()
        )

        # Filled after this run's analysis (see render_crop_header), so a
        # result produced by Analyze All in this same run is reflected.
        crop_header_slot = st.empty()
        crop_note_slot = st.empty()

        def render_crop_header(showing_suggestion):
            if final_crop_choice:
                crop_title = final_crop_choice
            elif showing_suggestion:
                crop_title = f"{suggested_crop} (router suggestion)"
            else:
                crop_title = "Choose the crop"

            crop_header_slot.markdown(
                f"""
                <div class="crop-page">
                    <div class="crop-name">{html.escape(crop_title)}</div>
                    <div class="notation" style="margin-top:.55rem">
                        {route_notation} · {route_status}
                    </div>
                    <div class="meter">
                        <b style="width:{route['confidence']:.1f}%;
                                  background:var(--soil)"></b>
                    </div>
                    <div class="notation router-scores">{router_scores}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Once results exist for the suggestion, the router's doubt is
            # listed under "Why this needs a second look" instead.
            if (
                route["needs_review"]
                and not crop_set_manually
                and not showing_suggestion
            ):
                crop_note_slot.markdown(
                    f"""
                    <div class="review-note">
                        <b>Confirm the crop</b>
                        <p>{html.escape(route["reason"])} Choose the crop above
                        to continue; the diagnosis uses that crop's model.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                crop_note_slot.empty()

        # Analyze All runs on the crop a person chose, or else on the
        # router's suggestion; a suggestion that needs review keeps its
        # warning in the result. Run diagnosis still needs a chosen crop.
        model_key = final_crop_choice or suggested_crop
        current_classes = CROP_CLASSES.get(model_key)

        model_errors = st.session_state.setdefault("model_errors", {})
        model_error = model_errors.get(model_key)

        if model_error:
            st.markdown(
                f"""
                <div class="review-note">
                    <b>{html.escape(model_key)} model unavailable</b>
                    <p>{html.escape(model_error)} Diagnosis for
                    {html.escape(model_key)} is disabled until a valid model
                    file is configured.</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            if st.button("Retry model download"):
                model_errors.pop(model_key, None)
                st.rerun()

        render_conditions(weather_data, soil_type, moisture_display)

        if not weather_data:
            st.markdown(
                '<p class="conditions-empty">Add a farm location in the '
                'sidebar to include live weather and soil moisture.</p>',
                unsafe_allow_html=True
            )

        analyze_clicked = st.button(
            "Analyze All",
            type="primary",
            width="stretch",
            disabled=model_key is None or bool(model_error),
            help=(
                "Runs the crop model on the leaf, then asks Gemini for an "
                "independent look at the same photo and compares the two."
            )
        )

        run_clicked = st.button(
            "Run diagnosis",
            width="stretch",
            disabled=final_crop_choice is None or bool(model_error),
            help="Crop model only, without the independent Gemini check."
        )

    if diagnosis and diagnosis["crop"] != model_key:
        st.session_state.pop("diagnosis", None)
        diagnosis = None

    if run_clicked or analyze_clicked:
        with st.spinner(
            f"Reading the {model_key} leaf..."
        ):
            try:
                interpreter = load_model(model_key)
                model_input = interpreter.get_input_details()[0]["shape"]

                whole_leaf = classify_image(
                    image=image,
                    interpreter=interpreter,
                    classes=current_classes,
                    model_key=model_key
                )

                quadrant_results = analyze_quadrants(
                    image=image,
                    interpreter=interpreter,
                    classes=current_classes,
                    model_key=model_key
                )

                summary = summarize_diagnosis(whole_leaf, quadrant_results)

                review_reasons = list(summary["review_reasons"])

                if model_key in UNVALIDATED_MODELS:
                    review_reasons.append(UNVALIDATED_MODELS[model_key])

                if route["needs_review"] and not crop_set_manually:
                    review_reasons.insert(0, route["reason"])

                advice = None
                ai_analysis = None

                advice_kwargs = dict(
                    disease_detected=summary["disease"],
                    weed_detected=summary["weed"],
                    weather=weather_data,
                    location=(
                        (field_point or {}).get("label")
                        or user_location
                    ),
                    crop=model_key,
                    soil_type=soil_type,
                    soil_moisture=soil_moisture,
                    quadrant_results=quadrant_results,
                    whole_leaf=whole_leaf,
                    review_reasons=review_reasons
                )

                if analyze_clicked:
                    api_key, ai_model = ai_settings()
                    leaf_jpeg = image_for_assessment(image)

                    with st.spinner("Asking Gemini for an independent look..."):
                        ai_analysis = run_ai_analysis(
                            ml_crop=model_key,
                            ml_summary=summary,
                            image_key=image_fingerprint(leaf_jpeg),
                            model=ai_model,
                            assess=lambda: assess_leaf_image(
                                leaf_jpeg, api_key, ai_model
                            ),
                            prescribe=lambda assessment, comparison: get_smart_advice(
                                **advice_kwargs,
                                gemini_assessment=assessment,
                                comparison=comparison
                            ),
                            cache=st.session_state.setdefault("ai_cache", {}),
                            blocked=st.session_state.get("advice_blocked"),
                            ai_enabled=enable_ai,
                            advice_context=[
                                soil_type, soil_moisture, weather_data,
                                advice_kwargs["location"]
                            ]
                        )

                    if ai_analysis["blocked"]:
                        st.session_state["advice_blocked"] = block_for_key(
                            ai_analysis["blocked"], api_key
                        )

                    advice = ai_analysis["advice"]

                elif enable_ai and (summary["disease"] or summary["weed"]):
                    blocked = st.session_state.get("advice_blocked")

                    if blocked:
                        advice = blocked
                    else:
                        with st.spinner("Consulting AI Agronomist..."):
                            advice = get_smart_advice(**advice_kwargs)

                        if not advice["ok"] and advice["blocks_session"]:
                            st.session_state["advice_blocked"] = block_for_key(
                                advice, ai_settings()[0]
                            )

                diagnosis = {
                    "crop": model_key,
                    "whole": whole_leaf,
                    "results": quadrant_results,
                    "summary": summary,
                    "review_reasons": review_reasons,
                    "advice": advice,
                    "ai_analysis": ai_analysis,
                    "ai_enabled": enable_ai,
                    "run_details": (
                        f"ROUTER {ROUTER_MODEL_PATH.stem} · "
                        f"{model_key.upper()} MODEL "
                        f"{MODEL_FILES[model_key].stem} · "
                        f"INPUT {model_input[1]}×{model_input[2]} · "
                        f"{len(current_classes)} CLASSES"
                    )
                }
                st.session_state["diagnosis"] = diagnosis

            except ModelFileError as model_file_error:
                model_errors[model_key] = str(model_file_error)
                st.rerun()

            except Exception as analysis_error:
                st.error(
                    f"Analysis error: {analysis_error}"
                )

    render_crop_header(
        showing_suggestion=(
            final_crop_choice is None
            and diagnosis is not None
            and diagnosis["crop"] == suggested_crop
        )
    )

    if diagnosis:
        summary = diagnosis["summary"]
        whole_leaf = diagnosis["whole"]

        st.markdown(
            f'<div class="section-title">{diagnosis["crop"]} · whole leaf, '
            'then four quadrant chips</div>',
            unsafe_allow_html=True
        )

        chips = "".join(
            chip_html(index, quadrant_name, result, whole_leaf["label"])
            for index, (quadrant_name, result) in enumerate(
                diagnosis["results"].items()
            )
        )

        st.markdown(
            "<div class=\"chip-grid\">"
            + "".join(line.strip() for line in chips.splitlines())
            + "</div>",
            unsafe_allow_html=True
        )

        verdict_kind = summary["verdict_kind"]

        if verdict_kind == "disease":
            verdict_title = f"Disease: {pretty_label(summary['disease'])}"
        elif verdict_kind == "weed":
            verdict_title = f"Weed: {pretty_label(summary['weed'])}"
        elif verdict_kind == "healthy":
            verdict_title = "Leaf reads healthy"
        else:
            verdict_title = "Uncertain result"

        review_reasons = diagnosis["review_reasons"]

        verdict_counts = (
            f"WHOLE LEAF {pretty_label(summary['primary']).upper()} "
            f"{summary['primary_conf']:.1f}% · "
            f"{summary['agreement']} OF 4 QUADRANTS AGREE · "
            "AI CLASSIFICATION, NOT A LABORATORY DIAGNOSIS"
        )

        review_pill = (
            '<span class="review-pill">Needs review</span>'
            if review_reasons
            else ""
        )

        st.markdown(
            f"""
            <div class="verdict" style="background:{STATE_COLORS[verdict_kind]}">
                <h3>{html.escape(verdict_title)}{review_pill}</h3>
                <p>{verdict_counts}</p>
                <p class="run-details">{html.escape(diagnosis.get("run_details", ""))}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        if review_reasons or summary["quadrant_only"]:
            items = "".join(
                f"<li>{html.escape(reason)}</li>"
                for reason in review_reasons
            )

            if summary["quadrant_only"]:
                items += (
                    "<li>Seen only in individual quadrants: "
                    + html.escape(
                        ", ".join(pretty_label(label) for label in summary["quadrant_only"])
                    )
                    + ". Quadrant crops lose leaf context and are less "
                    "reliable than the whole-leaf reading.</li>"
                )

            st.markdown(
                f"""
                <div class="review-note">
                    <b>Why this needs a second look</b>
                    <ul>{items}</ul>
                </div>
                """,
                unsafe_allow_html=True
            )

        if diagnosis.get("ai_analysis") is not None:
            render_ai_analysis(diagnosis["ai_analysis"], summary)

        advice = diagnosis["advice"]

        if advice:
            advice_failed = not advice["ok"]

            sheet_notation = " · ".join([
                diagnosis["crop"].upper(),
                soil_type.upper(),
                (moisture_display or "MOISTURE UNKNOWN").upper(),
                "NOT WRITTEN" if advice_failed else "WRITTEN BY AI"
            ])

            with st.container(key="prescription"):
                st.markdown(
                    f"""
                    <div class="sheet-head">
                        <b>AI agronomist prescription</b>
                        <span class="notation">{html.escape(sheet_notation)}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if advice_failed:
                    st.markdown(
                        f'<p class="sheet-status">{html.escape(advice["message"])}</p>',
                        unsafe_allow_html=True
                    )

                    with st.expander("Error detail"):
                        st.code(advice["detail"], language=None, wrap_lines=True)

                else:
                    st.markdown(advice["text"])

        elif diagnosis["ai_enabled"] and verdict_kind == "healthy":
            st.info(
                "The leaf reads healthy. No prescription is needed. "
                "Continue monitoring the crop."
            )

        elif diagnosis["ai_enabled"] and verdict_kind == "uncertain":
            st.info(
                "No prescription is written for an uncertain result. Confirm "
                "the crop, or try a closer, well-lit photo of a single leaf."
            )

if uploaded_file and st.session_state.get("diagnosis"):
    finished = st.session_state["diagnosis"]
    finished_summary = finished["summary"]
    issue = finished_summary["disease"] or finished_summary["weed"]
    review_suffix = " Needs review." if finished["review_reasons"] else ""

    if finished_summary["verdict_kind"] == "uncertain":
        page_title = f"{finished['crop']} leaf: result uncertain."
    elif issue:
        page_title = f"{finished['crop']} leaf: {pretty_label(issue)}."
    else:
        page_title = f"{finished['crop']} leaf reads healthy."

    page_lead = (
        "The whole leaf is read first; each quadrant is then read on its own "
        "to check it. Scores, runner-up candidates and field conditions are "
        "shown so the result can be checked, not just trusted." + review_suffix
    )

title_slot.markdown(
    f'<h1 class="page-title">{html.escape(page_title)}</h1>'
    f'<p class="page-lead">{html.escape(page_lead)}</p>',
    unsafe_allow_html=True
)

if st.session_state.get("advice_blocked"):
    advice_status_slot.markdown(
        '<div class="rail-reading">Prescriptions paused for this session: '
        f'{html.escape(st.session_state["advice_blocked"]["message"].split(". ")[0])}.'
        '</div>',
        unsafe_allow_html=True
    )
