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
        url = f"https://wttr.in/{location}?format=%t|%h|%p"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.text.split("|")
            return {
                "temp": data[0].strip(),
                "humidity": data[1].strip(),
                "precip": data[2].strip()
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
    
    weather_context = ""
    if weather:
        weather_context = f"Current weather in {location}: Temperature {weather['temp']}, Humidity {weather['humidity']}, Precipitation {weather['precip']}."
    
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

st.set_page_config(page_title="Agri-Doctor Pro", layout="wide")

st.sidebar.title("Agri-Doctor")
st.sidebar.subheader("Settings")
crop_choice = st.sidebar.radio("Select Crop", ["Maize (Corn)", "Rice (Paddy)"])
user_location = st.sidebar.text_input("Enter Your Location", placeholder="e.g., Hyderabad")
enable_ai = st.sidebar.checkbox("Enable AI Advice", value=True)

if crop_choice == "Maize (Corn)":
    st.title("Maize Disease Detection")
    current_classes = CORN_CLASSES
    model_key = 'Maize'
else:
    st.title("Rice Disease Detection")
    current_classes = RICE_CLASSES
    model_key = 'Rice'

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption='Original Image', use_column_width=True)
    
    with col2:
        if st.button('Run Diagnosis'):
            with st.spinner('Analyzing crop health...'):
                try:
                    interpreter = load_model(model_key)
                    
                    weather = None
                    if user_location:
                        weather = get_weather(user_location)
                    
                    if weather:
                        st.info(f"📍 Weather in {user_location}: {weather['temp']} | 💧 Humidity: {weather['humidity']} | 🌧️ Precip: {weather['precip']}")
                    
                    quad_results = analyze_quadrants(image, interpreter, current_classes)
                    
                    st.write("### Quadrant Analysis")
                    row1 = st.columns(2)
                    row2 = st.columns(2)
                    
                    # Store all detections to filter later
                    all_detections = []
                    
                    def display_quad(col, title, data):
                        with col:
                            st.image(data["img"], use_column_width=True)
                            lbl = data["label"]
                            conf = data["conf"]
                            
                            color = "gray"
                            if conf > 50:
                                if "Healthy" in lbl:
                                    color = "green"
                                elif "Weed" in lbl:
                                    color = "orange"
                                    all_detections.append((lbl, conf, "Weed"))
                                else:
                                    color = "red"
                                    all_detections.append((lbl, conf, "Disease"))
                            else:
                                lbl = "Uncertain"
                            
                            st.markdown(f"**{title}**: :{color}[{lbl} ({conf:.1f}%)]")

                    display_quad(row1[0], "Top-Left", quad_results["Top-Left"])
                    display_quad(row1[1], "Top-Right", quad_results["Top-Right"])
                    display_quad(row2[0], "Bottom-Left", quad_results["Bottom-Left"])
                    display_quad(row2[1], "Bottom-Right", quad_results["Bottom-Right"])
                    
                    if enable_ai:
                        st.write("### AI Doctor Prescription")
                        with st.spinner("Generating targeted recommendation..."):
                            
                            # LOGIC: Filter Top Disease + Top Weed
                            final_diagnosis = []
                            
                            # Separate into categories
                            diseases = [d for d in all_detections if d[2] == "Disease"]
                            weeds = [d for d in all_detections if d[2] == "Weed"]
                            
                            # Pick Top 1 Disease (Highest Confidence)
                            if diseases:
                                top_disease = max(diseases, key=lambda x: x[1])
                                final_diagnosis.append(top_disease[0])
                                
                            # Pick Top 1 Weed (Highest Confidence)
                            if weeds:
                                top_weed = max(weeds, key=lambda x: x[1])
                                final_diagnosis.append(top_weed[0])
                            
                            if not final_diagnosis:
                                if not all_detections:
                                    final_diagnosis = ["Healthy Crop"]
                                else:
                                    final_diagnosis = ["Uncertain Issue"]
                            
                            advice = get_smart_advice(final_diagnosis, weather, user_location)
                            
                            st.success(f"**Targeting:** {', '.join(final_diagnosis)}")
                            st.info(advice)
                            
                except Exception as e:
                    st.error(f"Error: {str(e)}")
