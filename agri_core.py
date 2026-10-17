"""
Model-side logic for Agri-Doctor Pro.

Kept free of Streamlit so it can be tested directly (see tests/).
"""

import hashlib
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np


# ============================================================
# CLASS MAPPINGS
# Order must match each model's training order (Keras sorts class
# directories alphabetically).
# ============================================================

ROUTER_CLASSES = [
    "Cotton",
    "Maize",
    "Rice"
]

CORN_CLASSES = [
    "Maize_Blight",
    "Maize_Common_Rust",
    "Maize_Gray_Leaf_Spot",
    "Maize_Healthy",
    "Weed_Broadleaf",
    "Weed_Grass"
]

# Matches labels.txt exported with the Rice model (where class 3 is named
# Rice_Healthy_Rice_Leaf).
RICE_CLASSES = [
    "Rice_Bacterial_Leaf_Blight",
    "Rice_Brown_Spot",
    "Rice_Healthy",
    "Rice_Leaf_Blast",
    "Rice_Leaf_scald",
    "Rice_Sheath_Blight",
    "Weed_Broadleaf",
    "Weed_Grass"
]

COTTON_CLASSES = [
    "Disease_Bacterial_Blight",
    "Disease_Curl_Virus",
    "Disease_Healthy",
    "Disease_Herbicide_Damage",
    "Disease_Leaf_Reddening",
    "Weed_Carpetweed",
    "Weed_Morningglory",
    "Weed_Palmer_Amaranth",
    "Weed_Purslane",
    "Weed_Waterhemp"
]

CROP_CLASSES = {
    "Maize": CORN_CLASSES,
    "Rice": RICE_CLASSES,
    "Cotton": COTTON_CLASSES
}


# ============================================================
# DECISION THRESHOLDS
# ============================================================

ROUTER_THRESHOLD = 75.0

# The crop router confuses these two grasses. On known photos in
# tests/data, 3 of 5 rice leaves were routed to Maize at 92-97%
# confidence, so a Maize or Rice routing is never accepted unreviewed,
# whatever its score.
ROUTER_CONFUSABLE = {"Maize", "Rice"}

DETECTION_THRESHOLD = 50.0

# Quadrant crops lose leaf context and drift toward Common Rust (a healthy
# maize leaf read Healthy whole, but Common Rust in 2 of 4 quadrants), so
# the whole-leaf reading leads and quadrants must corroborate it.
MIN_QUADRANT_AGREEMENT = 2


# ============================================================
# MODEL FILE VALIDATION
# ============================================================

TFLITE_IDENTIFIER = b"TFL3"
MIN_MODEL_BYTES = 100_000


class ModelFileError(RuntimeError):
    """A model file is missing, truncated, or not a TFLite model."""


def validate_tflite_file(path):
    """
    Reject files that cannot be a TFLite model before the interpreter
    sees them: a notebook, a labels.txt or an HTML error page saved under
    a .tflite name all fail here with a readable reason.
    """
    path = Path(path)

    if not path.exists():
        raise ModelFileError(f"{path.name} does not exist.")

    size = path.stat().st_size

    with open(path, "rb") as handle:
        header = handle.read(8)

    if header[4:8] != TFLITE_IDENTIFIER:
        preview = header[:8].decode("utf-8", errors="replace")
        raise ModelFileError(
            f"{path.name} is not a TensorFlow Lite model "
            f"(starts with {preview!r}, {size} bytes)."
        )

    if size < MIN_MODEL_BYTES:
        raise ModelFileError(
            f"{path.name} is only {size} bytes, too small to be a model."
        )

    return path


def check_output_classes(interpreter, classes, model_name):
    output_count = int(interpreter.get_output_details()[0]["shape"][-1])

    if output_count != len(classes):
        raise ModelFileError(
            f"{model_name} model returns {output_count} outputs, but "
            f"{len(classes)} class labels are configured."
        )

    return output_count


# Dequantised uint8 softmax outputs carry rounding error of up to 1/512
# per class, so a 10-class sum can drift by ~0.02.
PROBABILITY_TOLERANCE = 0.03


def check_probabilities(predictions, model_name):
    """
    Confidence is shown as output x 100, which only means something for a
    softmax output. Logits, NaN or independent sigmoid scores would show
    as confident-looking percentages, so they are rejected instead.
    """
    values = np.asarray(predictions, dtype=np.float64)

    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ModelFileError(
            f"{model_name} model returned non-finite scores."
        )

    total = float(values.sum())

    if (
        values.min() < -PROBABILITY_TOLERANCE
        or values.max() > 1 + PROBABILITY_TOLERANCE
        or abs(total - 1) > PROBABILITY_TOLERANCE
    ):
        raise ModelFileError(
            f"{model_name} model outputs are not probabilities (they sum "
            f"to {total:.3f}), so they cannot be shown as confidence."
        )

    return predictions


def get_drive_download_url(file_id):
    """
    Creates a direct Google Drive download URL.
    """
    return f"https://drive.google.com/uc?id={file_id}"


def fetch_from_drive(file_id, destination, downloader=None):
    """
    Download one Drive file and prove it is a TFLite model. Every failure
    (network, permissions, wrong ID, wrong file type) becomes a
    ModelFileError, and an invalid download is deleted.
    """
    destination = Path(destination)

    if downloader is None:
        import gdown

        downloader = gdown.download

    try:
        downloaded_file = downloader(
            url=get_drive_download_url(file_id),
            output=str(destination),
            quiet=False
        )
    except Exception as error:
        destination.unlink(missing_ok=True)
        raise ModelFileError(
            f"Could not download {destination.name} from Google Drive "
            f"({type(error).__name__}). Check the file ID and that the file "
            "is shared as 'Anyone with the link'."
        ) from error

    if downloaded_file is None or not destination.exists():
        raise ModelFileError(
            f"Could not download {destination.name} from Google Drive."
        )

    # A wrong ID often downloads a notebook, labels file or HTML page.
    # Reject it here so it never replaces a working model.
    try:
        validate_tflite_file(destination)
    except ModelFileError:
        destination.unlink(missing_ok=True)
        raise

    return destination


def download_model(file_id, destination, force_download=False, downloader=None):
    """
    Return a validated local model, downloading when absent (or always,
    with force_download). A refresh only replaces the current file after
    the new download validates.
    """
    import os

    destination = Path(destination)

    if destination.exists() and not force_download:
        try:
            return validate_tflite_file(destination)
        except ModelFileError:
            destination.unlink(missing_ok=True)

    temporary_path = destination.with_name(
        f"{destination.stem}_downloading{destination.suffix}"
    )
    temporary_path.unlink(missing_ok=True)

    fetch_from_drive(file_id, temporary_path, downloader=downloader)
    os.replace(temporary_path, destination)

    return destination


# ============================================================
# INPUT PREPROCESSING
# ============================================================

def prepare_model_input(image, interpreter, model_key=None):
    """
    Resize and preprocess an image according to the selected model.

    Also supports float32, uint8 and int8 TensorFlow Lite inputs.
    """

    input_details = interpreter.get_input_details()
    input_info = input_details[0]

    input_shape = input_info["shape"]
    input_dtype = input_info["dtype"]

    if len(input_shape) != 4:
        raise ValueError(
            f"Unexpected model input shape: {input_shape}"
        )

    input_height = int(input_shape[1])
    input_width = int(input_shape[2])

    img = image.convert("RGB").resize(
        (input_width, input_height)
    )

    image_array = np.asarray(img, dtype=np.float32)

    # All models take raw 0-255 pixels: each rescales internally (the
    # router maps to [-1, 1]; Maize and Cotton apply 1/255 plus ImageNet
    # normalisation). Feeding 0-1 instead collapses the router to Maize.

    image_array = np.expand_dims(image_array, axis=0)

    if input_dtype in [np.uint8, np.int8]:
        scale, zero_point = input_info["quantization"]

        if scale and scale > 0:
            image_array = image_array / scale + zero_point

        if input_dtype == np.uint8:
            image_array = np.clip(
                image_array,
                0,
                255
            ).astype(np.uint8)

        else:
            image_array = np.clip(
                image_array,
                -128,
                127
            ).astype(np.int8)

    else:
        image_array = image_array.astype(input_dtype)

    return image_array


# ============================================================
# PREDICTION
# ============================================================

def predict_image(image, interpreter, model_key=None):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    image_array = prepare_model_input(
        image=image,
        interpreter=interpreter,
        model_key=model_key
    )

    interpreter.set_tensor(
        input_details[0]["index"],
        image_array
    )

    interpreter.invoke()

    predictions = interpreter.get_tensor(
        output_details[0]["index"]
    )[0]

    output_dtype = output_details[0]["dtype"]
    output_scale, output_zero_point = output_details[0]["quantization"]

    if output_dtype in [np.uint8, np.int8] and output_scale > 0:
        predictions = (
            predictions.astype(np.float32) - output_zero_point
        ) * output_scale

    return predictions


def classify_image(image, interpreter, classes, model_key=None):
    predictions = predict_image(
        image=image,
        interpreter=interpreter,
        model_key=model_key
    )

    if len(predictions) != len(classes):
        raise ValueError(
            f"{model_key} model returned {len(predictions)} outputs, "
            f"but {len(classes)} class labels were provided."
        )

    check_probabilities(predictions, model_key)

    ranked = np.argsort(predictions)[::-1][:3]

    return {
        "label": classes[int(ranked[0])],
        "conf": float(predictions[int(ranked[0])] * 100),
        "top": [
            (classes[int(index)], float(predictions[int(index)] * 100))
            for index in ranked
        ]
    }


# ============================================================
# CROP ROUTING
# ============================================================

def route_crop(router_predictions):
    """
    Turn router output into a crop suggestion plus whether a person must
    confirm it.
    """
    if len(router_predictions) != len(ROUTER_CLASSES):
        raise ValueError(
            f"Router returned {len(router_predictions)} outputs, "
            f"but {len(ROUTER_CLASSES)} labels were provided."
        )

    check_probabilities(router_predictions, "Router")

    crop_index = int(np.argmax(router_predictions))
    crop = ROUTER_CLASSES[crop_index]
    confidence = float(router_predictions[crop_index] * 100)

    if confidence < ROUTER_THRESHOLD:
        needs_review = True
        reason = (
            f"Router confidence is {confidence:.1f}%, below the "
            f"{ROUTER_THRESHOLD:.0f}% threshold."
        )

    elif crop in ROUTER_CONFUSABLE:
        needs_review = True
        reason = (
            "The crop router often mistakes rice leaves for maize, even at "
            "high confidence. Confirm the crop before diagnosing."
        )

    else:
        needs_review = False
        reason = None

    return {
        "crop": crop,
        "confidence": confidence,
        "needs_review": needs_review,
        "reason": reason,
        "scores": {
            name: float(score * 100)
            for name, score in zip(ROUTER_CLASSES, router_predictions)
        }
    }


def default_crop_choice(route):
    """
    The crop to preselect. A routing that needs review is not preselected:
    otherwise one click on Run diagnoses a rice leaf the router called
    Maize with the Maize model, which is how rice leaves ended up
    reported as Common Rust.
    """
    return None if route["needs_review"] else route["crop"]


def crop_route_status(route, choice):
    """How the chosen crop relates to the router's suggestion."""
    if choice is None:
        return "NEEDS REVIEW"

    if choice != route["crop"]:
        return "SET MANUALLY"

    # Not preselected (see default_crop_choice), so a person picked it.
    if route["needs_review"]:
        return "CONFIRMED"

    return "ACCEPTED"


# ============================================================
# QUADRANT ANALYSIS
# ============================================================

def split_quadrants(image):
    width, height = image.size

    middle_width = width // 2
    middle_height = height // 2

    return {
        "Top-Left": image.crop(
            (0, 0, middle_width, middle_height)
        ),
        "Top-Right": image.crop(
            (middle_width, 0, width, middle_height)
        ),
        "Bottom-Left": image.crop(
            (0, middle_height, middle_width, height)
        ),
        "Bottom-Right": image.crop(
            (middle_width, middle_height, width, height)
        )
    }


def analyze_quadrants(
    image,
    interpreter,
    classes,
    model_key
):
    results = {}

    for quadrant_name, cropped_image in split_quadrants(image).items():
        result = classify_image(
            image=cropped_image,
            interpreter=interpreter,
            classes=classes,
            model_key=model_key
        )
        result["img"] = cropped_image
        results[quadrant_name] = result

    return results


def label_kind(label):
    if "Healthy" in label:
        return "healthy"

    if "Weed" in label:
        return "weed"

    return "disease"


def summarize_diagnosis(whole, quadrants):
    """
    The whole-leaf reading is the verdict; quadrants corroborate it.
    Disagreement or low confidence marks the result for review instead of
    reporting a confident answer the evidence does not support.
    """
    whole_kind = label_kind(whole["label"])
    agreement = sum(
        1 for result in quadrants.values()
        if result["label"] == whole["label"]
    )

    detected_diseases = []
    detected_weeds = []

    for result in quadrants.values():
        kind = label_kind(result["label"])

        if result["conf"] > DETECTION_THRESHOLD and kind != "healthy":
            if kind == "weed":
                detected_weeds.append(result["label"])
            else:
                detected_diseases.append(result["label"])

    review_reasons = []

    if whole["conf"] < DETECTION_THRESHOLD:
        verdict_kind = "uncertain"
        review_reasons.append(
            f"Whole-leaf confidence is only {whole['conf']:.1f}%."
        )
    else:
        verdict_kind = whole_kind

    if agreement < MIN_QUADRANT_AGREEMENT:
        review_reasons.append(
            f"Only {agreement} of 4 quadrants agree with the whole-leaf "
            "reading."
        )

    quadrant_only = sorted(
        {
            label for label in detected_diseases + detected_weeds
            if label != whole["label"]
        }
    )

    return {
        "verdict_kind": verdict_kind,
        "primary": whole["label"],
        "primary_conf": whole["conf"],
        "disease": whole["label"] if verdict_kind == "disease" else None,
        "weed": whole["label"] if verdict_kind == "weed" else None,
        "agreement": agreement,
        "disease_count": len(detected_diseases),
        "weed_count": len(detected_weeds),
        "quadrant_only": quadrant_only,
        "needs_review": bool(review_reasons),
        "review_reasons": review_reasons,
        "quadrant_majority": (
            Counter(r["label"] for r in quadrants.values()).most_common(1)[0][0]
        )
    }


# ============================================================
# LABELS AND FIELD CONDITIONS
# ============================================================

def pretty_label(label):
    """
    Maize_Common_Rust -> Common Rust, Weed_Palmer_Amaranth -> Palmer
    Amaranth weed, Disease_Healthy -> Healthy.
    """
    parts = label.split("_")

    if parts[0] == "Weed":
        return " ".join(parts[1:]).title() + " weed"

    if parts[0] in ["Maize", "Rice", "Disease"]:
        parts = parts[1:]

    return " ".join(parts).title()


def classify_soil_moisture(volumetric):
    """
    Map volumetric water content (m³/m³, 3-9 cm depth) to a field
    moisture band. Bands are approximate and soil-texture dependent.
    """
    if volumetric is None:
        return None

    if volumetric < 0.15:
        return "Dry"

    if volumetric < 0.30:
        return "Moist"

    if volumetric < 0.40:
        return "Wet"

    return "Waterlogged"


# ============================================================
# AI (GEMINI) ERRORS
# ============================================================

def _type_names(error):
    return {cls.__name__ for cls in type(error).__mro__}


def describe_advice_error(error):
    """
    Map a Gemini SDK or transport exception to a plain-language reason.
    The provider's message and details are read only to classify the
    error and are never returned: they can echo request data.
    """
    names = _type_names(error)
    name = type(error).__name__
    code = getattr(error, "code", None)
    http = code if isinstance(code, int) else None
    status = getattr(error, "status", None)
    status = status if isinstance(status, str) else None

    try:
        evidence = json.dumps(
            [getattr(error, "message", None), getattr(error, "details", None)],
            default=str
        ).lower()
    except Exception:
        evidence = ""

    unaffected = (
        "The image diagnosis above is unaffected, and you can keep using "
        "the app to diagnose leaves without an AI-generated prescription."
    )

    if "MissingKey" in names:
        kind = "missing_key"
        message = (
            "AI features are off because no Gemini key is configured. Add "
            "GEMINI_API_KEY to the app's Streamlit secrets to enable them. "
            + unaffected
        )

    elif http == 429 or status == "RESOURCE_EXHAUSTED":
        # Gemini reports per-minute limits and exhausted daily or billing
        # quota with the same status; the quota id in the details tells
        # them apart.
        if any(marker in evidence for marker in (
            "perday", "per_day", "limit: 0", "credit", "prepay"
        )):
            kind = "quota"
            message = (
                "AI prescriptions are unavailable because the Gemini API "
                "quota for this app is used up. " + unaffected + " To "
                "restore them, check the key's plan and billing in Google "
                "AI Studio."
            )
        else:
            kind = "rate_limit"
            message = (
                "Gemini is receiving too many requests. Wait a minute and "
                "run the analysis again. " + unaffected
            )

    elif (
        http in (401, 403)
        or status in ("UNAUTHENTICATED", "PERMISSION_DENIED")
        or "api_key_invalid" in evidence
        or "api key not valid" in evidence
    ):
        kind = "auth"
        message = (
            "The Gemini key is invalid or lacks access. Update "
            "GEMINI_API_KEY in the app's Streamlit secrets. " + unaffected
        )

    elif http == 404 or status == "NOT_FOUND":
        kind = "model"
        message = (
            "The configured Gemini model is not available to this key. "
            "Check gemini_model in the app's Streamlit secrets. "
            + unaffected
        )

    elif (
        http in (408, 504)
        or status == "DEADLINE_EXCEEDED"
        or names & {"TimeoutException", "TimeoutError"}
    ):
        kind = "timeout"
        message = (
            "Gemini took too long to answer. Try again in a moment. "
            + unaffected
        )

    elif names & {"ConnectError", "NetworkError", "TransportError",
                  "ConnectionError"}:
        kind = "unreachable"
        message = (
            "Gemini could not be reached. Check the network and try "
            "again. " + unaffected
        )

    else:
        kind = "other"
        message = "The AI service is temporarily unavailable. " + unaffected

    detail = name
    if http:
        detail += f" · HTTP {http}"
    if status:
        detail += f" · {status}"

    return {
        "kind": kind,
        "message": message,
        "detail": detail,
        # Billing and key problems will not fix themselves within a session,
        # so further calls are skipped instead of repeated.
        "blocks_session": kind in ("quota", "auth", "missing_key", "model")
    }


# ============================================================
# FIELD LOCATION
# ============================================================

GEOLOCATION_ERRORS = {
    0: "This browser does not support location access.",
    1: "Location permission was denied. Allow location for this site, or "
       "search for the place or click the field on the map instead.",
    2: "Your device could not determine its location. Search for the place "
       "or click the field on the map instead.",
    3: "Finding your location timed out. Try again outdoors or with Wi-Fi "
       "on, or search for the place instead."
}


def parse_geolocation(result):
    """
    Interpret the browser Geolocation API result. Returns
    {"ok": True, "lat", "lon", "accuracy"} or {"ok": False, "message"}.
    """
    if not isinstance(result, dict):
        return {"ok": False, "message": GEOLOCATION_ERRORS[2]}

    if "error" in result:
        code = (result.get("error") or {}).get("code")
        return {
            "ok": False,
            "message": GEOLOCATION_ERRORS.get(code, GEOLOCATION_ERRORS[2])
        }

    coords = result.get("coords") or {}

    try:
        lat = float(coords["latitude"])
        lon = float(coords["longitude"])
    except (KeyError, TypeError, ValueError):
        return {"ok": False, "message": GEOLOCATION_ERRORS[2]}

    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return {"ok": False, "message": GEOLOCATION_ERRORS[2]}

    accuracy = coords.get("accuracy")

    return {
        "ok": True,
        "lat": lat,
        "lon": lon,
        "accuracy": float(accuracy) if accuracy is not None else None
    }


def polygon_area_hectares(ring):
    """
    Approximate area of a small lon/lat polygon (GeoJSON ring) using a
    local equirectangular projection. Good to about 1% for field-sized
    shapes; not for large or polar areas.
    """
    if not ring or len(ring) < 3:
        return 0.0

    earth_radius = 6_371_008.8
    mean_lat = np.radians(np.mean([point[1] for point in ring]))

    xs = [np.radians(point[0]) * earth_radius * np.cos(mean_lat) for point in ring]
    ys = [np.radians(point[1]) * earth_radius for point in ring]

    area = 0.0
    for index in range(len(ring)):
        next_index = (index + 1) % len(ring)
        area += xs[index] * ys[next_index] - xs[next_index] * ys[index]

    return abs(area) / 2 / 10_000


# ============================================================
# WEATHER
# ============================================================

WEATHER_SOURCE = "Open-Meteo Forecast API (open-meteo.com)"


def wmo_condition(code):
    if code == 0:
        return "Clear sky"
    if code in [1, 2, 3]:
        return "Partly cloudy"
    if code in [45, 48]:
        return "Fog"
    if code in [51, 53, 55, 56, 57, 61, 63, 65, 66, 67]:
        return "Rain"
    if code in [71, 73, 75, 77]:
        return "Snow"
    if code in [80, 81, 82]:
        return "Rain showers"
    if code in [95, 96, 99]:
        return "Thunderstorm"
    return "Unknown"


def parse_weather(payload):
    """
    Turn an Open-Meteo forecast response into display values. Missing
    fields stay None (shown as unavailable), never filled in.
    """
    current = payload.get("current") or {}
    units = payload.get("current_units") or {}

    def reading(key):
        value = current.get(key)
        if value is None:
            return None
        unit = units.get(key, "")
        spacer = "" if unit in ("%", "°C") else " "
        return f"{value}{spacer}{unit}".strip()

    soil_moisture = None
    hourly = payload.get("hourly") or {}
    current_hour = (current.get("time") or "")[:13]

    for timestamp, value in zip(
        hourly.get("time", []),
        hourly.get("soil_moisture_3_to_9cm", [])
    ):
        if timestamp[:13] == current_hour and value is not None:
            soil_moisture = float(value)
            break

    code = current.get("weather_code")

    return {
        "temp": reading("temperature_2m"),
        "humidity": reading("relative_humidity_2m"),
        "precip": reading("precipitation"),
        "condition": wmo_condition(code) if code is not None else None,
        "soil_moisture": soil_moisture,
        "soil_moisture_band": classify_soil_moisture(soil_moisture),
        "observed": current.get("time"),
        "timezone": payload.get("timezone_abbreviation") or payload.get("timezone"),
        "source": WEATHER_SOURCE
    }


# ============================================================
# SAVED FIELDS
# ============================================================

FIELD_KEYS = ("name", "lat", "lon", "crop", "planting_date", "boundary", "notes")


def field_record(name, lat, lon, crop=None, planting_date=None, boundary=None, notes=""):
    """
    Build the saved-field record. Coordinates are rounded to 5 decimals
    (about 1 m), enough to find the field without storing more precision
    than the feature needs.
    """
    if not name or not name.strip():
        raise ValueError("A field needs a name.")

    return {
        "name": name.strip()[:80],
        "lat": round(float(lat), 5),
        "lon": round(float(lon), 5),
        "crop": crop,
        "planting_date": planting_date,
        "boundary": [
            [round(float(x), 6), round(float(y), 6)] for x, y in boundary
        ] if boundary else None,
        "notes": (notes or "").strip()[:500]
    }


def load_field_records(data):
    """
    Validate fields loaded from a user's JSON file; unknown keys are
    dropped and invalid entries skipped.
    """
    if isinstance(data, dict):
        data = data.get("fields", [])

    records = []

    for item in data if isinstance(data, list) else []:
        try:
            records.append(
                field_record(
                    item["name"], item["lat"], item["lon"],
                    item.get("crop"), item.get("planting_date"),
                    item.get("boundary"), item.get("notes", "")
                )
            )
        except (KeyError, TypeError, ValueError):
            continue

    return records


# ============================================================
# UPLOADED IMAGES
# ============================================================

MIN_IMAGE_SIDE = 64
MAX_IMAGE_SIDE = 4096


class ImageInputError(ValueError):
    """The upload cannot be used as a leaf photo."""


def load_leaf_image(data):
    """
    Decode an uploaded photo for diagnosis: apply the camera's EXIF
    rotation (phone photos are often stored sideways), convert to RGB, and
    reject files that are not images or are too small to classify. Very
    large photos are scaled down; the models only see 224x224.
    Returns (image, notes).
    """
    from PIL import Image, ImageOps, UnidentifiedImageError
    import io

    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()
        image = Image.open(io.BytesIO(data))
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as error:
        raise ImageInputError(
            "This file could not be read as an image. Upload a JPG or PNG "
            "photo of a leaf."
        ) from error

    notes = []

    if min(image.size) < MIN_IMAGE_SIDE:
        raise ImageInputError(
            f"This image is {image.width}×{image.height} px. Use a photo at "
            f"least {MIN_IMAGE_SIDE} px on each side."
        )

    if max(image.size) > MAX_IMAGE_SIDE:
        image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
        notes.append(
            f"Large photo scaled to {image.width}×{image.height} px."
        )

    return image, notes


# ============================================================
# MODEL VALIDATION STATUS
# ============================================================

# Models whose accuracy has not been established on held-out labelled data.
# Every result from them is marked for review, with the measured evidence.
UNVALIDATED_MODELS = {
    "Rice": (
        "The Rice model (rice_weed_robust.tflite) has not been validated: "
        "it identified 1 of 4 labelled rice-disease photos correctly in "
        "testing. Treat this result as a preliminary screen."
    )
}



# ============================================================
# GEMINI KEY, MODEL AND CLIENT
# ============================================================

API_KEY_NAME = "GEMINI_API_KEY"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_TIMEOUT_MS = 60_000


def _setting(secrets, environ, secret_name, env_name):
    try:
        value = secrets.get(secret_name)
    except Exception:
        # st.secrets raises when no secrets.toml exists.
        value = None

    if not (isinstance(value, str) and value.strip()):
        value = environ.get(env_name)

    return value.strip() if isinstance(value, str) and value.strip() else None


def resolve_api_key(secrets, environ):
    """
    GEMINI_API_KEY from Streamlit secrets, then the environment. Blank
    values count as missing. The key is never logged or returned in
    messages.
    """
    return _setting(secrets, environ, API_KEY_NAME, API_KEY_NAME)


def key_fingerprint(api_key):
    """One-way tag for the configured key, so state can follow key changes."""
    if not api_key:
        return None
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]


def block_for_key(error, api_key):
    """A session pause, remembered with the key it was caused by."""
    return {**error, "key_fingerprint": key_fingerprint(api_key)}


def block_applies(blocked, api_key):
    """
    A pause holds only while the same key (or the same absence of a key) is
    configured. Adding a key, or replacing an invalid or out-of-quota one,
    lifts it; the same key stays paused.
    """
    return bool(blocked) and blocked.get("key_fingerprint") == key_fingerprint(api_key)


def resolve_gemini_model(secrets, environ):
    return (
        _setting(secrets, environ, "gemini_model", "GEMINI_MODEL")
        or DEFAULT_GEMINI_MODEL
    )


class MissingKey(Exception):
    """No Gemini key in Streamlit secrets or the environment."""


class EmptyResponse(Exception):
    """Gemini answered without usable text."""


def make_gemini_client(api_key):
    """
    Official google-genai client with a bounded timeout. The SDK does not
    retry unless asked, so a failing call returns promptly.
    """
    from google import genai
    from google.genai import types

    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT_MS)
    )


def _generate_text(client, model, contents, config=None):
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config
    )

    text = getattr(response, "text", None)

    if not isinstance(text, str) or not text.strip():
        raise EmptyResponse()

    return text


def _gemini_call(api_key, client_factory, run):
    """Shared wrapper: no key → no client; every failure → error dict."""
    try:
        if not api_key:
            raise MissingKey()

        client = (client_factory or make_gemini_client)(api_key)
        return {"ok": True, **run(client)}

    except Exception as error:
        return {"ok": False, **describe_advice_error(error)}


# ============================================================
# AI AGRONOMIST PRESCRIPTION
# ============================================================

def request_advice(prompt, api_key, model, client_factory=None):
    """
    Ask Gemini for a prescription. Returns {"ok": True, "text"} or
    {"ok": False, kind, message, detail, blocks_session}.
    """
    return _gemini_call(
        api_key,
        client_factory,
        lambda client: {"text": _generate_text(client, model, [prompt])}
    )


# ============================================================
# INDEPENDENT GEMINI IMAGE ASSESSMENT
# ============================================================

GEMINI_CROPS = ["Maize", "Rice", "Cotton"]
ASSESSMENT_LABELS = sorted(set(CORN_CLASSES + RICE_CLASSES + COTTON_CLASSES))
CERTAINTY_LEVELS = ["low", "medium", "high"]
MAX_ASSESSMENT_IMAGE_SIDE = 1024

ASSESSMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "crop": {"type": "string", "enum": GEMINI_CROPS + ["Other", "Unknown"]},
        "closest_label": {
            "type": "string",
            "enum": ASSESSMENT_LABELS + ["Other", "Unknown"]
        },
        "condition": {"type": "string"},
        "symptoms": {"type": "array", "items": {"type": "string"}},
        "supporting_features": {"type": "array", "items": {"type": "string"}},
        "alternatives": {"type": "array", "items": {"type": "string"}},
        "image_quality_issues": {"type": "array", "items": {"type": "string"}},
        "multiple_conditions": {"type": "boolean"},
        "certainty": {"type": "string", "enum": CERTAINTY_LEVELS},
        "summary": {"type": "string"}
    },
    "required": [
        "crop", "closest_label", "condition", "symptoms",
        "supporting_features", "alternatives", "image_quality_issues",
        "multiple_conditions", "certainty", "summary"
    ]
}


def _label_glossary():
    lines = []
    for label in ASSESSMENT_LABELS:
        crops = [
            crop for crop, classes in CROP_CLASSES.items() if label in classes
        ]
        lines.append(f"- {label}: {pretty_label(label)} ({', '.join(crops)})")
    return "\n".join(lines)


# The ML result is deliberately not part of this prompt: the assessment
# must be made from the image alone so the comparison means something.
ASSESSMENT_PROMPT = f"""
You are a plant pathologist examining one uploaded photo of a crop leaf.
Assess it independently, only from what is visible in the image. No other
system's diagnosis is given to you.

Report:
- crop: Maize, Rice or Cotton if visually identifiable; Other if it is a
  different plant; Unknown if you cannot tell.
- condition: the most likely condition in your own words (or "Healthy").
- closest_label: the label below that best matches the main visible
  condition; Other if the condition is not on the list; Unknown if you
  cannot tell.
- symptoms: visible symptoms and patterns (lesion shape, colour, spread).
- supporting_features: the visual features that support your assessment.
- alternatives: other conditions or explanations that could look similar
  (including nutrient deficiency, mechanical or herbicide damage).
- image_quality_issues: blur, lighting, framing, multiple leaves, or none.
- multiple_conditions: true if more than one condition may be present.
- certainty: low, medium or high. Use low when the image or the evidence
  is weak. Do not guess to appear confident.
- summary: two or three plain sentences for a farmer.

Labels:
{_label_glossary()}
""".strip()


def image_for_assessment(image):
    """JPEG bytes of the (already EXIF-corrected) upload, at most 1024 px."""
    copy = image.convert("RGB")
    copy.thumbnail((MAX_ASSESSMENT_IMAGE_SIDE, MAX_ASSESSMENT_IMAGE_SIDE))
    buffer = io.BytesIO()
    copy.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def image_fingerprint(image_bytes):
    return hashlib.sha256(image_bytes).hexdigest()


def _text_list(value, limit=8):
    if not isinstance(value, list):
        return []
    return [str(item).strip()[:200] for item in value if str(item).strip()][:limit]


def normalize_assessment(data):
    """
    Keep only the expected fields, in their allowed values. Anything the
    model returns outside the schema becomes Unknown/low rather than
    being trusted.
    """
    if not isinstance(data, dict):
        raise EmptyResponse()

    crop = data.get("crop")
    label = data.get("closest_label")
    certainty = str(data.get("certainty", "")).lower()

    return {
        "crop": crop if crop in GEMINI_CROPS + ["Other"] else "Unknown",
        "closest_label": (
            label if label in ASSESSMENT_LABELS + ["Other"] else "Unknown"
        ),
        "condition": str(data.get("condition") or "").strip()[:200],
        "symptoms": _text_list(data.get("symptoms")),
        "supporting_features": _text_list(data.get("supporting_features")),
        "alternatives": _text_list(data.get("alternatives")),
        "image_quality_issues": _text_list(data.get("image_quality_issues")),
        "multiple_conditions": data.get("multiple_conditions") is True,
        "certainty": certainty if certainty in CERTAINTY_LEVELS else "low",
        "summary": str(data.get("summary") or "").strip()[:800]
    }


def assess_leaf_image(image_bytes, api_key, model, client_factory=None):
    """
    Send only the image and the independent-assessment prompt to Gemini.
    Returns {"ok": True, "assessment"} or an error dict.
    """
    def run(client):
        from google.genai import types

        text = _generate_text(
            client,
            model,
            [
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                ASSESSMENT_PROMPT
            ],
            types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=ASSESSMENT_SCHEMA,
                temperature=0.2
            )
        )

        try:
            data = json.loads(text)
        except ValueError as error:
            raise EmptyResponse() from error

        return {"assessment": normalize_assessment(data)}

    return _gemini_call(api_key, client_factory, run)


# ============================================================
# COMPARING THE ML DIAGNOSIS WITH THE GEMINI ASSESSMENT
# ============================================================

AGREEMENT = "Agreement"
PARTIAL = "Partial agreement"
DISAGREEMENT = "Disagreement"
UNCERTAIN = "Uncertain"


def compare_assessments(ml_crop, ml_summary, assessment):
    """
    Categorise how the ML diagnosis and Gemini's independent assessment
    relate. Both are kept as they are: confidences are never averaged,
    and agreement is not treated as proof.
    """
    reasons = []
    ml_label = ml_summary["primary"]
    ml_kind = ml_summary["verdict_kind"]
    g_label = assessment["closest_label"]
    g_crop = assessment["crop"]

    if ml_kind == "uncertain":
        reasons.append("The ML whole-leaf reading is below its confidence threshold.")
    if assessment["certainty"] == "low":
        reasons.append("Gemini rated its own assessment as low certainty.")
    if g_label == "Unknown":
        reasons.append("Gemini could not name a likely condition from the image.")

    if reasons:
        category = UNCERTAIN

    elif g_crop == "Other" or (g_crop in GEMINI_CROPS and g_crop != ml_crop):
        category = DISAGREEMENT
        seen = "a different plant" if g_crop == "Other" else g_crop.lower()
        reasons.append(
            f"The ML diagnosis used the {ml_crop} model, but Gemini sees "
            f"{seen}. Confirm the crop before acting on either result."
        )

    else:
        g_kind = "problem" if g_label == "Other" else label_kind(g_label)
        ml_issue = ml_kind in ("disease", "weed")
        g_issue = g_kind in ("disease", "weed", "problem")

        if g_label == ml_label or (ml_kind == "healthy" and g_kind == "healthy"):
            category = AGREEMENT
            reasons.append(
                f"Both read the leaf as {pretty_label(ml_label)}."
            )
            if assessment["multiple_conditions"]:
                category = PARTIAL
                reasons.append(
                    "Gemini also suspects more than one condition may be present."
                )

        elif ml_issue != g_issue:
            category = DISAGREEMENT
            healthy_side = "The ML model" if not ml_issue else "Gemini"
            reasons.append(
                f"{healthy_side} reads the leaf as healthy while the other "
                "finds a problem."
            )

        elif g_kind != "problem" and g_kind != ml_kind:
            category = DISAGREEMENT
            reasons.append(
                f"The ML model finds a {ml_kind} ({pretty_label(ml_label)}), "
                f"Gemini a {g_kind} ({pretty_label(g_label)})."
            )

        else:
            category = PARTIAL
            g_name = assessment["condition"] or pretty_label(g_label)
            reasons.append(
                f"Both find a problem, but the ML model names "
                f"{pretty_label(ml_label)} and Gemini names {g_name}."
            )
            alternatives = " ".join(assessment["alternatives"]).lower()
            if pretty_label(ml_label).lower() in alternatives:
                reasons.append(
                    "Gemini lists the ML diagnosis among its alternatives."
                )

    if category == AGREEMENT:
        next_step = (
            "The two independent readings agree, which makes this result "
            "more plausible but does not prove it. Check the leaf in the "
            "field before treating."
        )
    elif category == PARTIAL:
        next_step = (
            "The readings overlap but differ in detail. Compare the "
            "symptoms below with the leaf, and ask a local agricultural "
            "expert before choosing a treatment."
        )
    else:
        next_step = (
            "Do not act on either result alone. Take a closer, well-lit "
            "photo of a single leaf and run the analysis again, or have a "
            "local agricultural expert examine the crop."
        )

    return {
        "category": category,
        "reasons": reasons,
        "next_step": next_step,
        # A prescription is written only when both readings support the
        # ML finding of a problem; otherwise both results are shown as is.
        "supports_prescription": (
            category in (AGREEMENT, PARTIAL)
            and ml_kind in ("disease", "weed")
        )
    }


# ============================================================
# ANALYZE ALL: AI PART (THE ML PART RUNS UNCHANGED BEFORE THIS)
# ============================================================

def cached_ai_call(cache, key, call):
    """
    Reuse a successful answer for the same key (same image, model and
    inputs) instead of paying for it again. Failures are not cached, so
    clicking again retries a transient error.
    """
    hit = cache.get(key)
    if hit is not None:
        return hit
    result = call()
    if result.get("ok"):
        cache[key] = result
    return result


def run_ai_analysis(ml_crop, ml_summary, image_key, model, assess, prescribe,
                    cache, blocked=None, ai_enabled=True, advice_context=None):
    """
    Gemini's half of Analyze All. `assess()` returns the independent image
    assessment; `prescribe(assessment, comparison)` the prescription. At
    most two Gemini calls are made: the assessment must not see the ML
    result (independence), while the prescription needs both. The ML
    result is never modified here. `advice_context` (soil, moisture,
    weather, location) is part of the prescription's cache key, so changed
    field conditions get a fresh prescription.
    """
    result = {
        "assessment": None,
        "comparison": None,
        "advice": None,
        "error": None,
        "blocked": None
    }

    if not ai_enabled:
        return result

    if blocked:
        result["error"] = blocked
        return result

    assessed = cached_ai_call(cache, ("assess", image_key, model), assess)

    if not assessed["ok"]:
        result["error"] = assessed
        if assessed["blocks_session"]:
            result["blocked"] = assessed
        return result

    assessment = assessed["assessment"]
    comparison = compare_assessments(ml_crop, ml_summary, assessment)
    result["assessment"] = assessment
    result["comparison"] = comparison

    if comparison["supports_prescription"]:
        advice_key = (
            "advice", image_key, model, ml_crop, ml_summary["primary"],
            comparison["category"],
            json.dumps(advice_context, sort_keys=True, default=str)
        )
        advice = cached_ai_call(
            cache, advice_key, lambda: prescribe(assessment, comparison)
        )
        result["advice"] = advice
        if not advice["ok"] and advice["blocks_session"]:
            result["blocked"] = advice

    return result
