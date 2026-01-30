import streamlit as st
import tensorflow as tf
import numpy as np
import requests
from PIL import Image
import os
import gdown
import openai

# --- CONFIGURATION ---
# I have inserted your NEW TFLite ID here:
MODEL_FILE_ID = '1_1PcQqUFFiK9tgpXwivM6J7OJShL18jk'
MODEL_FILENAME = 'corn_model.tflite'

CLASS_NAMES = {
    0: 'Maize_Blight',
    1: 'Maize_Common_Rust',
    2: 'Maize_Gray_Leaf_Spot',
    3: 'Maize_Healthy',
    4: 'Weed_Broadleaf',
    5: 'Weed_Grass'
}

st.set_page_config(page_title="Agri-Smart Advisor", layout="wide")

# --- 1. TFLITE MODEL LOADER (The Fix) ---
@st.cache_resource
def load_model_from_drive():
    # Download if missing
    if not os.path.exists(MODEL_FILENAME):
        url = f'https://drive.google.com/uc?id={MODEL_FILE_ID}'
        gdown.download(url, MODEL_FILENAME, quiet=False, fuzzy=True)

    # Load TFLite (Bypasses all Keras version errors)
    interpreter = tf.lite.Interpreter(model_path=MODEL_FILENAME)
    interpreter.allocate_tensors()
    return interpreter

def tflite_predict(interpreter, image):
    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Resize and Format Image
    img = image.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    # Standardize if needed (EfficientNet usually likes raw 0-255 or preprocessed)
    # We use basic preprocessing here to be safe
    img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)

    # Run Prediction
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# --- 2. WEATHER & UTILS ---
def get_real_weather(city_name):
    try:
        url = f"https://wttr.in/{city_name}?format=j1"
        response = requests.get(url)
        data = response.json()
        current = data['current_condition'][0]
        return {
            "temperature": current['temp_C'],
            "humidity": current['humidity'],
            "condition": current['weatherDesc'][0]['value'],
            "city": city_name
        }
    except:
        return {"temperature": "30", "humidity": "45", "condition": "Sunny", "city": city_name}

# --- 3. OPENAI ADVICE ---
def get_openai_advice(vision_results, weather):
    if "OPENAI_API_KEY" not in st.secrets:
        return "⚠️ OpenAI API Key missing in Secrets."

    client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    problems = ", ".join(list(vision_results))
    
    prompt = f"""
    You are an expert agronomist.
    Situation:
    - Crops: {problems}
    - Weather: {weather['condition']}, {weather['temperature']}C
    
    Provide a 3-step treatment plan (Chemical & Organic). Keep it under 150 words.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"OpenAI Error: {e}"

# --- MAIN APP UI ---
st.title("Agri-Smart Advisor 🌽")

st.sidebar.header("Settings")
city = st.sidebar.text_input("Farm Location (City)", value="Vellore")

# Load Model
try:
    interpreter = load_model_from_drive()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

uploaded_file = st.file_uploader("Upload Crop Image", type=["jpg", "png", "jpeg"])

if uploaded_file and city:
    try:
        original_img = Image.open(uploaded_file).convert("RGB")
        st.image(original_img, caption="Uploaded Image", use_column_width=True)
        
        st.info("Analyzing...")
        
        # Crop Logic
        width, height = original_img.size
        crops = {
            "Top-Left": original_img.crop((0, 0, width//2, height//2)),
            "Top-Right": original_img.crop((width//2, 0, width, height//2)),
            "Bottom-Left": original_img.crop((0, height//2, width//2, height)),
            "Bottom-Right": original_img.crop((width//2, height//2, width, height))
        }
        
        found_problems = set()
        cols = st.columns(2)
        
        for i, (pos, crop_img) in enumerate(crops.items()):
            # Use TFLite Predictor
            preds = tflite_predict(interpreter, crop_img)
            
            pred_index = np.argmax(preds)
            confidence = np.max(preds) * 100
            label = CLASS_NAMES[pred_index]
            
            if confidence > 60:
                found_problems.add(label)
                with cols[i % 2]:
                    st.image(crop_img, caption=f"{pos}: {label} ({confidence:.1f}%)")
        
        st.divider()
        
        weather = get_real_weather(city)
        st.subheader(f"Weather in {weather['city']}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Temp", f"{weather['temperature']}°C")
        c2.metric("Sky", weather['condition'])
        c3.metric("Humidity", f"{weather['humidity']}%")
        
        st.subheader("Diagnosis & Treatment")
        if not found_problems:
            st.success("✅ Crop looks healthy!")
        else:
            with st.spinner("Consulting AI Expert..."):
                advice = get_openai_advice(found_problems, weather)
                st.write(advice)

    except Exception as e:
        st.error(f"An error occurred: {e}")
