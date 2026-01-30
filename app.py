import streamlit as st
import tensorflow as tf
import numpy as np
import requests
from PIL import Image
import os
import gdown
import cv2
import openai
import json
import matplotlib.cm as cm

# --- CONFIGURATION ---
# Your Maize/Weed Model ID
MODEL_FILE_ID = '1NzmXgv3nDe0xorHoxhWF06cYLd21UMM4'
MODEL_FILENAME = 'corn_model.h5'

# Your Class Names (Manually defined since we can't load the JSON easily from Drive)
CLASS_NAMES = {
    0: 'Maize_Blight',
    1: 'Maize_Common_Rust',
    2: 'Maize_Gray_Leaf_Spot',
    3: 'Maize_Healthy',
    4: 'Weed_Broadleaf',
    5: 'Weed_Grass'
}

st.set_page_config(page_title="Agri-Smart Advisor", layout="wide")

# --- 1. ROBUST MODEL LOADER ---
@st.cache_resource
def load_model_from_drive():
    # Delete if corrupt/tiny
    if os.path.exists(MODEL_FILENAME):
        if os.path.getsize(MODEL_FILENAME) < 1000000:
            os.remove(MODEL_FILENAME)

    # Download if missing
    if not os.path.exists(MODEL_FILENAME):
        url = f'https://drive.google.com/uc?id={MODEL_FILE_ID}'
        gdown.download(url, MODEL_FILENAME, quiet=False, fuzzy=True)

    # Load with fixes for the "tuple" error
    model = tf.keras.models.load_model(MODEL_FILENAME, compile=False, safe_mode=False)
    return model

# --- 2. WEATHER FUNCTION ---
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
        return {"temperature": "30", "humidity": "45", "condition": "Sunny (Default)", "city": city_name}

# --- 3. GRAD-CAM (Heatmap) ---
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="top_activation", pred_index=None):
    try:
        # Create a sub-model that maps input -> conv layer output -> final prediction
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )
        
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    except:
        return None # Fail silently if layer name is wrong

def overlay_heatmap(img, heatmap, alpha=0.4):
    if heatmap is None: return img
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    return superimposed_img

# --- 4. OPENAI ADVICE ---
def get_openai_advice(api_key, vision_results, weather):
    if not api_key:
        return "⚠️ OpenAI API Key missing. Please enter it in the sidebar."
    
    client = openai.OpenAI(api_key=api_key)
    problems = ", ".join(list(vision_results))
    
    prompt = f"""
    You are an expert agronomist.
    Current Situation:
    - Detected Crop Issues: {problems}
    - Local Weather: {weather['condition']}, {weather['temperature']}C, Humidity {weather['humidity']}%
    
    Task:
    1. Analyze the detected issues.
    2. Check if the current weather allows for spraying chemicals.
    3. Recommend a specific chemical treatment and an organic alternative.
    4. Keep it concise (under 150 words).
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"OpenAI Error: {e}"

# --- MAIN APP LAYOUT ---
st.title("Agri-Smart Advisor 🌽")

# Sidebar
st.sidebar.header("Settings")
city = st.sidebar.text_input("Farm Location (City)", value="Vellore")
api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# Load Model
try:
    model = load_model_from_drive()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

uploaded_file = st.file_uploader("Upload Crop Image", type=["jpg", "png", "jpeg"])

if uploaded_file and city:
    try:
        original_img = Image.open(uploaded_file).convert("RGB")
        st.image(original_img, caption="Uploaded Image", use_column_width=True)
        
        st.info("Step 1: Analyzing Image Structure...")
        
        # Split image into 4 quadrants (Top-Left, Top-Right, etc.)
        width, height = original_img.size
        crops = {
            "Top-Left": original_img.crop((0, 0, width//2, height//2)),
            "Top-Right": original_img.crop((width//2, 0, width, height//2)),
            "Bottom-Left": original_img.crop((0, height//2, width//2, height)),
            "Bottom-Right": original_img.crop((width//2, height//2, width, height))
        }
        
        found_problems = set()
        cols = st.columns(2)
        
        st.info("Step 2: Scanning for Diseases...")
        
        for i, (pos, crop_img) in enumerate(crops.items()):
            # Preprocess
            img_array = crop_img.resize((224, 224))
            img_array = np.array(img_array)
            img_batch = np.expand_dims(img_array, axis=0)
            preprocessed_img = tf.keras.applications.efficientnet_v2.preprocess_input(img_batch.copy())
            
            # Predict
            preds = model.predict(preprocessed_img, verbose=0)
            pred_index = np.argmax(preds)
            confidence = np.max(preds) * 100
            label = CLASS_NAMES[pred_index]
            
            # If confident, show heatmap
            if confidence > 60:
                found_problems.add(label)
                
                # Attempt GradCAM (might fail if layer name differs, so we wrap in try)
                heatmap = make_gradcam_heatmap(preprocessed_img, model, "top_activation", pred_index)
                if heatmap is not None:
                    final_img = overlay_heatmap(img_array, heatmap)
                else:
                    final_img = crop_img
                
                with cols[i % 2]:
                    st.image(final_img, caption=f"{pos}: {label} ({confidence:.1f}%)")
        
        st.divider()
        
        # Weather & Advice
        st.info("Step 3: Fetching Weather & Expert Advice...")
        weather = get_real_weather(city)
        
        st.subheader(f"Weather in {weather['city']}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Temperature", f"{weather['temperature']}°C")
        c2.metric("Condition", weather['condition'])
        c3.metric("Humidity", f"{weather['humidity']}%")
        
        st.subheader("AI Agronomist Diagnosis")
        if not found_problems:
            st.success("✅ Crop looks healthy! No issues detected.")
        else:
            if api_key:
                with st.spinner("Consulting AI Agronomist..."):
                    advice = get_openai_advice(api_key, found_problems, weather)
                    st.write(advice)
            else:
                st.warning("⚠️ Enter OpenAI API Key in sidebar to get treatment advice.")
                st.write(f"Detected Issues: {', '.join(found_problems)}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
