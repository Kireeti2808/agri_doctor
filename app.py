import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageOps
import gdown
import os
import openai
import requests
from collections import Counter

st.set_page_config(page_title="Agri-Doctor Pro", layout="wide")

CORN_MODEL_ID = '1_1PcQqUFFiK9tgpXwivM6J7OJShL18jk'
RICE_MODEL_ID = '1p2vZgq_FBigVnlhQPLQD4w2yjDn4zus3'

CORN_CLASSES = [
    'Maize_Blight', 'Maize_Common_Rust', 'Maize_Gray_Leaf_Spot', 
    'Maize_Healthy', 'Weed_Broadleaf', 'Weed_Grass'
]

RICE_CLASSES = [
    'Rice_Bacterial_Leaf_Blight', 'Rice_Brown_Spot', 'Rice_Healthy_Rice_Leaf',
    'Rice_Leaf_Blast', 'Rice_Leaf_scald', 'Rice_Sheath_Blight',
    'Weed_Broadleaf', 'Weed_Grass'
]

@st.cache_resource
def load_model(crop_type):
    if crop_type == 'Maize':
        file_id = CORN_MODEL_ID
        filename = 'corn_model.tflite'
    else:
        file_id = RICE_MODEL_ID
        filename = 'rice_model.tflite'
    
    if not os.path.exists(filename):
        with st.spinner(f"Downloading {crop_type} Model..."):
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, filename, quiet=False)
    
    interpreter = tf.lite.Interpreter(model_path=filename)
    interpreter.allocate_tensors()
    return interpreter

def predict_image(image, interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    img = image.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0]

def get_weather(location):
    if not location:
        return None
    try:
        url = f"https://wttr.in/{location}?format=%t|%h|%p|%C"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.text.split("|")
            return {
                "temp": data[0].strip(),
                "humidity": data[1].strip(),
                "precip": data[2].strip(),
                "condition": data[3].strip()
            }
    except:
        return None
    return None

def analyze_quadrants(image, interpreter, classes):
    w, h = image.size
    mid_w, mid_h = w // 2, h // 2
    
    quadrants = {
        "Top-Left": image.crop((0, 0, mid_w, mid_h)),
        "Top-Right": image.crop((mid_w, 0, w, mid_h)),
        "Bottom-Left": image.crop((0, mid_h, mid_w, h)),
        "Bottom-Right": image.crop((mid_w, mid_h, w, h))
    }
    
    results = {}
    for name, img_crop in quadrants.items():
        preds = predict_image(img_crop, interpreter)
        idx = np.argmax(preds)
        conf = np.max(preds) * 100
        label = classes[idx]
        results[name] = {"label": label, "conf": conf, "img": img_crop}
        
    return results

def get_smart_advice(diseases, weather, location):
    try:
        api_key = st.secrets["openai_key"]
    except:
        return "OpenAI API Key missing in Secrets."

    client = openai.OpenAI(api_key=api_key)
    
    disease_str = ", ".join(diseases)
    
    weather_context = "Unknown weather"
    if weather:
        weather_context = f"Current weather in {location}: {weather['temp']}, Humidity {weather['humidity']}, Rain {weather['precip']}."
    
    prompt = f"""
    You are an expert Agronomist. 
    Analysis Report:
    - Primary Issues Detected: {disease_str}
    - {weather_context}

    Task:
    1. Analyze how the current weather might affect these specific issues.
    2. Provide a treatment plan. If a weed and a disease are both present, suggest a compatible approach.
    3. Give immediate actionable advice for the farmer.
    
    Keep response concise and structured.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error connecting to AI Doctor: {e}"

st.sidebar.title("Agri-Doctor")
crop_choice = st.sidebar.radio("Select Crop", ["Maize (Corn)", "Rice (Paddy)"])
user_location = st.sidebar.text_input("Enter Your Location", placeholder="e.g. Hyderabad")
enable_ai = st.sidebar.checkbox("Enable AI Advice", value=True)

if crop_choice == "Maize (Corn)":
    st.header("Maize Disease Detection")
    current_classes = CORN_CLASSES
    model_key = 'Maize'
else:
    st.header("Rice Disease Detection")
    current_classes = RICE_CLASSES
    model_key = 'Rice'

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Leaf', use_column_width=True)
    
    with col2:
        if user_location:
            weather_data = get_weather(user_location)
            if weather_data:
                st.subheader(f"Weather in {user_location}")
                w_col1, w_col2, w_col3 = st.columns(3)
                w_col1.metric("Temperature", weather_data['temp'])
                w_col2.metric("Humidity", weather_data['humidity'])
                w_col3.metric("Rainfall", weather_data['precip'])
            else:
                st.write("Weather data unavailable.")
        else:
            weather_data = None

        st.markdown("---")
        
        if st.button('Run Smart Analysis', use_container_width=True):
            with st.spinner('Scanning leaf quadrants...'):
                try:
                    interpreter = load_model(model_key)
                    
                    quad_results = analyze_quadrants(image, interpreter, current_classes)
                    
                    st.write("Quadrant Analysis")
                    q_col1, q_col2 = st.columns(2)
                    
                    all_detections = []
                    
                    def process_and_display(column, title, data):
                        lbl = data["label"]
                        conf = data["conf"]
                        
                        color_hex = "#777"
                        status = "Uncertain"
                        
                        if conf > 50:
                            if "Healthy" in lbl:
                                color_hex = "#28a745"
                                status = "Healthy"
                            elif "Weed" in lbl:
                                color_hex = "#fd7e14"
                                status = "Weed"
                                all_detections.append((lbl, conf, "Weed"))
                            else:
                                color_hex = "#dc3545"
                                status = "Disease"
                                all_detections.append((lbl, conf, "Disease"))
                        
                        with column:
                            st.image(data["img"], use_column_width=True)
                            st.markdown(f"""
                                <div style="text-align: center; font-size: 14px; margin-bottom: 10px;">
                                    <b>{title}</b><br>
                                    <span style="color: {color_hex}; font-weight: bold;">
                                        {lbl} ({conf:.1f}%)
                                    </span>
                                </div>
                            """, unsafe_allow_html=True)

                    process_and_display(q_col1, "Top-Left", quad_results["Top-Left"])
                    process_and_display(q_col2, "Top-Right", quad_results["Top-Right"])
                    process_and_display(q_col1, "Bottom-Left", quad_results["Bottom-Left"])
                    process_and_display(q_col2, "Bottom-Right", quad_results["Bottom-Right"])

                    if enable_ai:
                        st.markdown("---")
                        st.subheader("AI Doctor Prescription")
                        
                        with st.spinner("Synthesizing treatment plan..."):
                            final_diagnosis = []
                            
                            diseases = [d for d in all_detections if d[2] == "Disease"]
                            weeds = [d for d in all_detections if d[2] == "Weed"]
                            
                            if diseases:
                                top_disease = max(diseases, key=lambda x: x[1])[0]
                                final_diagnosis.append(top_disease)
                            
                            if weeds:
                                top_weed = max(weeds, key=lambda x: x[1])[0]
                                final_diagnosis.append(top_weed)
                                
                            if not final_diagnosis:
                                final_diagnosis = ["Healthy Crop" if not all_detections else "Uncertain Issue"]
                            
                            advice = get_smart_advice(final_diagnosis, weather_data, user_location)
                            
                            st.success(f"Targeting: {', '.join(final_diagnosis)}")
                            st.info(advice)

                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
