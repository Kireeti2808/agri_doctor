import os
from pathlib import Path

import gdown
import numpy as np
import openai
import requests
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications import efficientnet, mobilenet


# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Agri-Doctor Pro",
    layout="wide"
)


# ============================================================
# UI STYLING
# ============================================================

st.markdown(
    """
    <style>
    .stButton > button {
        border-radius: 20px;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border: none;
        padding: 10px 25px;
    }

    img {
        border-radius: 15px;
    }

    .weather-card {
        background: linear-gradient(135deg, #e0f7fa 0%, #ffffff 100%);
        border-radius: 25px;
        padding: 20px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
        border: 1px solid #b2ebf2;
        color: #333;
        display: flex;
        align-items: center;
        justify-content: space-around;
        margin-bottom: 20px;
    }

    .weather-icon {
        font-size: 45px;
        margin-right: 15px;
    }

    .weather-temp {
        font-size: 32px;
        font-weight: 800;
        margin: 0;
        color: #00796b;
    }

    .weather-detail-item {
        font-size: 14px;
        margin: 2px 0;
        color: #555;
    }

    .result-box {
        background-color: #ffffff;
        border: 1px solid #f0f0f0;
        border-radius: 15px;
        padding: 15px;
        margin-bottom: 10px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.02);
    }

    .advice-box {
        background-color: #e3f2fd;
        border-left: 6px solid #1565c0;
        padding: 20px;
        border-radius: 15px;
        margin-top: 20px;
        line-height: 1.6;
        font-size: 16px;
        color: #0d47a1;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# GOOGLE DRIVE MODEL IDS
# ============================================================

ROUTER_MODEL_ID = "10LuRi-n6wFw4um9deAWBu0sr0-8e1MLV"
CORN_MODEL_ID = "1_1PcQQUFF1K9tgpXwivM6J7OJShl18jk"
RICE_MODEL_ID = "1p2vZqa_FBigVn1hQPLQD4w2yDn4zus3"

# Make sure this is the Google Drive file ID of the NEW Cotton model.
COTTON_MODEL_ID = "1tX19wVEBa9pyZCd_7hIGC8j2fwrHQEie"


# ============================================================
# LOCAL MODEL FILENAMES
# ============================================================

ROUTER_MODEL_PATH = Path("crop_router_v2.tflite")
CORN_MODEL_PATH = Path("corn_model.tflite")
RICE_MODEL_PATH = Path("rice_model.tflite")
COTTON_MODEL_PATH = Path("cotton_doctor_latest.tflite")


# ============================================================
# CLASS MAPPINGS
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
    "Weed_Carpweed",
    "Weed_Morningglory",
    "Weed_Palmer_Amaranth",
    "Weed_Purslane",
    "Weed_Waterhemp"
]
# ============================================================
# MODEL DOWNLOAD HELPERS
# ============================================================

def get_drive_download_url(file_id):
    """
    Creates a direct Google Drive download URL.
    """
    return f"https://drive.google.com/uc?id={file_id}"


def download_model(file_id, destination, force_download=False):
    destination = Path(destination)

    if destination.exists() and not force_download:
        return destination

    if force_download:
        temporary_path = destination.with_name(
            f"{destination.stem}_downloading{destination.suffix}"
        )

        if temporary_path.exists():
            temporary_path.unlink()

        # Removed fuzzy=False here
        downloaded_file = gdown.download(
            url=get_drive_download_url(file_id),
            output=str(temporary_path),
            quiet=False
        )

        if downloaded_file is None or not temporary_path.exists():
            raise RuntimeError(
                f"Failed to download model from Google Drive."
            )

        if temporary_path.stat().st_size == 0:
            temporary_path.unlink(missing_ok=True)
            raise RuntimeError("The downloaded model file is empty.")

        os.replace(temporary_path, destination)

    else:
        # Removed fuzzy=False here as well
        downloaded_file = gdown.download(
            url=get_drive_download_url(file_id),
            output=str(destination),
            quiet=False
        )

        if downloaded_file is None or not destination.exists():
            raise RuntimeError(
                f"Failed to download model from Google Drive."
            )

        if destination.stat().st_size == 0:
            destination.unlink(missing_ok=True)
            raise RuntimeError(f"Downloaded model is empty: {destination}")

    return destination

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
            "path": ROUTER_MODEL_PATH
        },
        "Maize": {
            "file_id": CORN_MODEL_ID,
            "path": CORN_MODEL_PATH
        },
        "Rice": {
            "file_id": RICE_MODEL_ID,
            "path": RICE_MODEL_PATH
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

        interpreter = tf.lite.Interpreter(
            model_path=str(model_path)
        )
        interpreter.allocate_tensors()

    return interpreter


# ============================================================
# COTTON MODEL LOADER
# Cotton is deliberately NOT cached.
# ============================================================

def load_fresh_cotton_model():
    """
    Always download and load the latest Cotton model.

    This function is not decorated with st.cache_resource.
    The previous local Cotton model is replaced on every diagnosis.
    """

    with st.spinner("Downloading the latest Cotton model..."):
        model_path = download_model(
            file_id=COTTON_MODEL_ID,
            destination=COTTON_MODEL_PATH,
            force_download=True
        )

    interpreter = tf.lite.Interpreter(
        model_path=str(model_path)
    )
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    output_shape = output_details[0]["shape"]
    output_class_count = int(output_shape[-1])

    st.sidebar.success("Fresh Cotton model downloaded")

    st.sidebar.info(
        f"Cotton model file: {model_path.name}\n\n"
        f"File size: {model_path.stat().st_size / (1024 * 1024):.2f} MB\n\n"
        f"Input shape: {input_details[0]['shape']}\n\n"
        f"Output shape: {output_shape}"
    )

    if output_class_count != len(COTTON_CLASSES):
        raise ValueError(
            "Cotton model output does not match COTTON_CLASSES. "
            f"The model returns {output_class_count} outputs, but "
            f"COTTON_CLASSES contains {len(COTTON_CLASSES)} labels."
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
# INPUT PREPROCESSING
# ============================================================

def prepare_model_input(image, interpreter, model_key):
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

    if model_key == "Cotton":
        image_array = mobilenet.preprocess_input(image_array)

    elif model_key in ["Maize", "Rice"]:
        image_array = efficientnet.preprocess_input(image_array)

    elif model_key == "Router":
        # The original Router preprocessing is preserved.
        # If the Router was trained using normalization, change this
        # to exactly match its training preprocessing.
        pass

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

def predict_image(image, interpreter, model_key):
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

    # Dequantize quantized model output if necessary.
    output_dtype = output_details[0]["dtype"]
    output_scale, output_zero_point = output_details[0]["quantization"]

    if output_dtype in [np.uint8, np.int8] and output_scale > 0:
        predictions = (
            predictions.astype(np.float32) - output_zero_point
        ) * output_scale

    return predictions
    # ============================================================
# WEATHER FUNCTIONS
# ============================================================

def get_wmo_weather_info(code):
    if code == 0:
        return "Clear sky", "☀️"

    if code in [1, 2, 3]:
        return "Partly cloudy", "🌤️"

    if code in [45, 48]:
        return "Fog", "🌫️"

    if code in [
        51, 53, 55, 56, 57,
        61, 63, 65, 66, 67
    ]:
        return "Rain", "🌧️"

    if code in [71, 73, 75, 77]:
        return "Snow", "❄️"

    if code in [80, 81, 82]:
        return "Rain showers", "🌦️"

    if code in [95, 96, 99]:
        return "Thunderstorm", "⛈️"

    return "Unknown", "🌤️"


def get_weather(city_name):
    if not city_name:
        return None

    try:
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"

        geo_response = requests.get(
            geo_url,
            params={
                "name": city_name,
                "count": 1
            },
            timeout=15
        )

        geo_response.raise_for_status()
        geo_data = geo_response.json()

        if not geo_data.get("results"):
            return None

        latitude = geo_data["results"][0]["latitude"]
        longitude = geo_data["results"][0]["longitude"]

        weather_url = "https://api.open-meteo.com/v1/forecast"

        weather_response = requests.get(
            weather_url,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": (
                    "temperature_2m,"
                    "relative_humidity_2m,"
                    "precipitation,"
                    "weather_code"
                ),
            },
            timeout=15
        )

        weather_response.raise_for_status()
        weather_data = weather_response.json()["current"]

        condition_text, icon = get_wmo_weather_info(
            weather_data["weather_code"]
        )

        return {
            "temp": f"{weather_data['temperature_2m']}°C",
            "humidity": f"{weather_data['relative_humidity_2m']}%",
            "precip": f"{weather_data['precipitation']} mm",
            "condition": condition_text,
            "icon": icon
        }

    except (
        requests.RequestException,
        KeyError,
        TypeError,
        ValueError
    ):
        return None


# ============================================================
# QUADRANT ANALYSIS
# ============================================================

def analyze_quadrants(
    image,
    interpreter,
    classes,
    model_key
):
    width, height = image.size

    middle_width = width // 2
    middle_height = height // 2

    quadrants = {
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

    results = {}

    for quadrant_name, cropped_image in quadrants.items():
        predictions = predict_image(
            image=cropped_image,
            interpreter=interpreter,
            model_key=model_key
        )

        if len(predictions) != len(classes):
            raise ValueError(
                f"{model_key} model returned {len(predictions)} outputs, "
                f"but {len(classes)} class labels were provided."
            )

        prediction_index = int(np.argmax(predictions))
        confidence = float(np.max(predictions) * 100)

        results[quadrant_name] = {
            "label": classes[prediction_index],
            "conf": confidence,
            "img": cropped_image
        }

    return results


# ============================================================
# AI AGRONOMIST
# ============================================================

def get_smart_advice(
    disease_detected,
    weed_detected,
    weather,
    location,
    crop
):
    try:
        client = openai.OpenAI(
            api_key=st.secrets["openai_key"]
        )

        if weather:
            weather_text = (
                f"Temperature: {weather['temp']}, "
                f"Humidity: {weather['humidity']}, "
                f"Rain: {weather['precip']}, "
                f"Sky: {weather['condition']}"
            )

            humidity = weather.get(
                "humidity",
                "Not available"
            )

            # The remaining original lines in this section
            # are not fully visible in the photographs.

        else:
            weather_text = "Weather information is unavailable"
            humidity = "Not available"
            precipitation = "Not available"

        issues = []

        if disease_detected:
            issues.append(
                f"Disease ({disease_detected})"
            )

        if weed_detected:
            issues.append(
                f"Weed ({weed_detected})"
            )

        diagnosis = (
            " AND ".join(issues)
            if issues
            else "Healthy Crop"
        )

        prompt = f"""
You are an expert agronomist AI.

Crop: {crop}
Detected issues: {diagnosis} in {location or "Unknown"}
Weather: {weather_text}

Provide a concise and safe management plan with these sections:

1. DIAGNOSIS SUMMARY
2. WEATHER CHECK
Explain whether spraying is appropriate with humidity of
{humidity} and precipitation of {precipitation}.
3. ACTION PLAN
Suggest suitable management options. Clearly state that local
agricultural guidance and product labels must be followed.
Explain whether fungicides and herbicides should be mixed.
4. ORGANIC ALTERNATIVE

Do not present the AI image classification as a confirmed
laboratory diagnosis. Keep the answer structured and concise.
"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.choices[0].message.content

    except Exception as error:
        return (
            "AI Agronomist is currently unavailable. "
            f"Error: {error}"
        )
       # ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Agri-Doctor Pro")
st.sidebar.markdown("---")

user_location = st.sidebar.text_input(
    "Farm Location",
    placeholder="e.g. Hyderabad"
)

enable_ai = st.sidebar.checkbox(
    "Enable AI Agronomist",
    value=True
)

st.sidebar.caption(
    "Router, Maize and Rice are cached. "
    "Cotton is freshly downloaded whenever Cotton diagnosis runs."
)


# ============================================================
# MAIN APPLICATION
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Leaf Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    first_column, second_column = st.columns([1, 1])

    with first_column:
        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    with second_column:
        try:
            router_interpreter = load_model("Router")

            router_predictions = predict_image(
                image=image,
                interpreter=router_interpreter,
                model_key="Router"
            )

            if len(router_predictions) != len(ROUTER_CLASSES):
                raise ValueError(
                    "Router model output does not match ROUTER_CLASSES. "
                    f"Router returned {len(router_predictions)} outputs, "
                    f"but {len(ROUTER_CLASSES)} labels were provided."
                )

            crop_index = int(
                np.argmax(router_predictions)
            )

            confidence_score = float(
                router_predictions[crop_index] * 100
            )

            detected_subject = ROUTER_CLASSES[crop_index]

            if confidence_score < 75.0:
                detected_subject = "Weed_Only"

        except Exception as router_error:
            st.warning(
                f"Router model could not be used: {router_error}"
            )

            detected_subject = "Weed_Only"
            confidence_score = 0.0

        if detected_subject == "Weed_Only":
            st.warning(
                "Unrecognized crop or isolated weed detected."
            )

            st.info(
                "The AI could not confidently identify Maize, Rice, "
                "or Cotton. Please manually confirm the field type."
            )

            final_crop_choice = st.radio(
                "Select field context:",
                ["Maize", "Rice", "Cotton"]
            )

        else:
            st.success(
                f"Auto-Detected Crop: {detected_subject} "
                f"({confidence_score:.1f}% confidence)"
            )

            final_crop_choice = detected_subject

        if final_crop_choice == "Cotton":
            model_key = "Cotton"
            current_classes = COTTON_CLASSES

        elif final_crop_choice == "Maize":
            model_key = "Maize"
            current_classes = CORN_CLASSES

        else:
            model_key = "Rice"
            current_classes = RICE_CLASSES

        weather_data = get_weather(user_location)

        if weather_data:
            st.markdown(
                f"""
                <div class="weather-card">
                    <div style="display:flex; align-items:center;">
                        <span class="weather-icon">
                            {weather_data['icon']}
                        </span>
                        <div>
                            <p class="weather-temp">
                                {weather_data['temp']}
                            </p>
                            <span style="color:#666; font-size:14px;">
                                {weather_data['condition']}
                            </span>
                        </div>
                    </div>
                    <div style="
                        font-size:13px;
                        border-left:1px solid #ddd;
                        padding-left:20px;
                    ">
                        <div class="weather-detail-item">
                            Humidity: {weather_data['humidity']}
                        </div>
                        <div class="weather-detail-item">
                            Rainfall: {weather_data['precip']}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        if st.button(
            "Run Deep Diagnosis",
            type="primary"
        ):
            with st.spinner(
                f"Running {final_crop_choice} Quadrant Analysis..."
            ):
                try:
                    interpreter = load_model(model_key)

                    quadrant_results = analyze_quadrants(
                        image=image,
                        interpreter=interpreter,
                        classes=current_classes,
                        model_key=model_key
                    )

                    st.write("### Quadrant Analysis Results")

                    quadrant_column_1, quadrant_column_2 = (
                        st.columns(2)
                    )

                    detected_diseases = []
                    detected_weeds = []

                    for index, (
                        quadrant_name,
                        result
                    ) in enumerate(quadrant_results.items()):

                        target_column = (
                            quadrant_column_1
                            if index % 2 == 0
                            else quadrant_column_2
                        )

                        with target_column:
                            st.write(f"**{quadrant_name}**")

                            st.image(
                                result["img"],
                                width=150
                            )

                            label = result["label"]
                            confidence = result["conf"]

                            is_weed = "Weed" in label
                            is_healthy = "Healthy" in label

                            if is_healthy:
                                color = "#28a745"

                            elif is_weed:
                                color = "#fd7e14"

                            else:
                                color = "#dc3545"

                            st.markdown(
                                f"""
                                <div class="result-box">
                                    <div style="
                                        display:flex;
                                        justify-content:space-between;
                                        margin-bottom:5px;
                                    ">
                                        <span style="
                                            font-weight:bold;
                                            color:{color};
                                        ">
                                            {label}
                                        </span>
                                        <span style="
                                            color:#666;
                                            font-size:12px;
                                        ">
                                            {confidence:.1f}%
                                        </span>
                                    </div>
                                    <div style="
                                        width:100%;
                                        background:#e0e0e0;
                                        border-radius:5px;
                                        height:8px;
                                    ">
                                        <div style="
                                            width:{min(confidence, 100)}%;
                                            background:{color};
                                            height:8px;
                                            border-radius:5px;
                                        "></div>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                            if confidence > 50 and not is_healthy:
                                if is_weed:
                                    detected_weeds.append(label)
                                else:
                                    detected_diseases.append(label)

                    final_disease = (
                        max(
                            set(detected_diseases),
                            key=detected_diseases.count
                        )
                        if detected_diseases
                        else None
                    )

                    final_weed = (
                        max(
                            set(detected_weeds),
                            key=detected_weeds.count
                        )
                        if detected_weeds
                        else None
                    )

                    if final_disease and final_weed:
                        st.error(
                            "Multiple Issues Detected: "
                            f"{final_disease} AND {final_weed}"
                        )

                    elif final_disease:
                        st.error(
                            f"Disease Detected: {final_disease}"
                        )

                    elif final_weed:
                        st.warning(
                            f"Weed Detected: {final_weed}"
                        )

                    else:
                        st.success("Crop appears Healthy")

                    if enable_ai and (
                        final_disease or final_weed
                    ):
                        with st.spinner(
                            "Consulting AI Agronomist..."
                        ):
                            advice = get_smart_advice(
                                disease_detected=final_disease,
                                weed_detected=final_weed,
                                weather=weather_data,
                                location=user_location,
                                crop=final_crop_choice
                            )

                        formatted_advice = advice.replace(
                            "\n",
                            "<br>"
                        )

                        st.markdown(
                            f"""
                            <div class="advice-box">
                                <h4 style="
                                    margin-top:0;
                                    color:#1565c0;
                                ">
                                    AI Agronomist Prescription
                                </h4>
                                {formatted_advice}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    elif enable_ai:
                        st.info(
                            "Crop is healthy. No prescription is needed. "
                            "Continue monitoring the crop."
                        )

                except Exception as analysis_error:
                    st.error(
                        f"Analysis Error: {analysis_error}"
                    ) 
