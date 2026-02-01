import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os
import openai
import requests
from tensorflow.keras.applications import efficientnet, mobilenet

st.set_page_config(page_title="Agri-Doctor Pro", layout="wide")

# ==========================================
# CONSTANTS & CONFIGURATION
# ==========================================
CORN_MODEL_ID = '1_1PcQqUFFiK9tgpXwivM6J7OJShL18jk'
RICE_MODEL_ID = '1p2vZgq_FBigVnlhQPLQD4w2yjDn4zus3'
COTTON_DISEASE_ID = '14d3ZHEA8GnOliO164BA811tWnZ-EhPm0'
COTTON_WEED_ID = '1Sk2h23GtVLPHBJ700Ld4AEOU8ra9dRmH'

CORN_CLASSES = [
    'Maize_Blight', 'Maize_Common_Rust', 'Maize_Gray_Leaf_Spot', 
    'Maize_Healthy', 'Weed_Broadleaf', 'Weed_Grass'
]

RICE_CLASSES = [
    'Rice_Bacterial_Leaf_Blight', 'Rice_Brown_Spot', 'Rice_Healthy', 
    'Rice_Leaf_Blast', 'Rice_Leaf_scald', 'Rice_Sheath_Blight', 
    'Weed_Broadleaf', 'Weed_Grass'
]

COTTON_DISEASE_CLASSES = [
    'Bacterial Blight', 'Curl Virus', 'Healthy Leaf', 
    'Herbicide Growth Damage', 'Leaf Redding', 'Leaf Variegation'
]

COTTON_WEED_CLASSES = [
    'Carpetweeds', 'Morningglory', 'PalmerAmaranth', 'Purslane', 'Waterhemp'
]

# ==========================================
# FUNCTIONS
# ==========================================
@st.cache_resource
def load_model(model_key):
    if model_key == 'Maize':
        file_id, filename = CORN_MODEL_ID, 'corn_model.tflite'
    elif model_key == 'Rice':
        file_id, filename = RICE_MODEL_ID, 'rice_model.tflite'
    elif model_key == 'Cotton_Disease':
        file_id, filename = COTTON_DISEASE_ID, 'cotton_disease.tflite'
    elif model_key == 'Cotton_Weed':
        file_id, filename = COTTON_WEED_ID, 'cotton_weed.tflite'
    
    if not os.path.exists(filename):
        with st.spinner(f"Downloading {model_key} Model..."):
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, filename, quiet=False)
    
    interpreter = tf.lite.Interpreter(model_path=filename)
    interpreter.allocate_tensors()
    return interpreter

def predict_image(image, interpreter, model_key):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    img = image.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    
    if 'Cotton' in model_key:
        img_array = mobilenet.preprocess_input(img_array)
    else:
        img_array = efficientnet.preprocess_input(img_array)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0]

# --- PURE CODE VISUALS (CSS SHAPES) ---
def get_weather_visual(condition):
    condition = condition.lower()
    
    # CSS Shapes created by code
    sun_html = """
    <div style="width:40px; height:40px; background:#FFD700; border-radius:50%; box-shadow: 0 0 10px orange; margin:auto;"></div>
    """
    
    cloud_html = """
    <div style="width:50px; height:25px; background:#B0C4DE; border-radius:20px; margin:auto; position:relative; top:10px;"></div>
    """
    
    rain_html = """
    <div style="width:50px; height:25px; background:#778899; border-radius:20px; margin:auto;"></div>
    <div style="text-align:center; color:#1E90FF; font-weight:bold; line-height:10px;">| | |</div>
    """
    
    if "sun" in condition or "clear" in condition:
        return sun_html
    elif "rain" in condition or "shower" in condition:
        return rain_html
    else: # Default to Cloud
        return cloud_html

def get_weather(location):
    if not location: return None
    try:
        # Request Metric (Celsius) by default from wttr.in
        url = f"https://wttr.in/{location}?format=%t|%h|%p|%C"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.text.split("|")
            condition_text = data[3].strip()
            
            # Ensure Celsius
            temp = data[0].strip()
            if "F" in temp:
                 # Basic conversion fallback if API fails to give C
                 val = float(''.join(filter(str.isdigit, temp)))
                 temp = f"{int((val - 32) * 5/9)}°C"
            
            return {
                "temp": temp, 
                "humidity": data[1].strip(), 
                "precip": data[2].strip(), 
                "condition": condition_text,
                "visual": get_weather_visual(condition_text)
            }
    except: return None
    return None

def analyze_quadrants(image, interpreter, classes, model_key):
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
        preds = predict_image(img_crop, interpreter, model_key)
        idx = np.argmax(preds)
        conf = np.max(preds) * 100
        results[name] = {"label": classes[idx], "conf": conf, "img": img_crop}
    return results

def get_smart_advice(diagnosis, weather, location, mode):
    try:
        api_key = st.secrets["openai_key"]
        client = openai.OpenAI(api_key=api_key)
        
        weather_txt = f"{weather['temp']}, {weather['condition']}" if weather else "Unknown"
        prompt = f"""
        Act as an Expert Agronomist. 
        Context: User detected '{diagnosis}' in {mode} Mode.
        Weather: {weather_txt} in {location}.
        
        Task: Provide a concise 3-step solution (Chemical & Organic).
        """
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", 
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except: return "AI Advice Unavailable"

# ==========================================
# MAIN UI
# ==========================================
st.sidebar.title("Agri-Doctor")
st.sidebar.markdown("---")

crop_choice = st.sidebar.radio("Select Crop", ["Maize (Corn)", "Rice (Paddy)", "Cotton"])

if crop_choice == "Cotton":
    st.header("Cotton Intelligence System")
    analysis_type = st.sidebar.radio("Analysis Focus", ["Leaf Disease", "Weed Type"])
    
    if analysis_type == "Leaf Disease":
        model_key = "Cotton_Disease"
        current_classes = COTTON_DISEASE_CLASSES
    else:
        model_key = "Cotton_Weed"
        current_classes = COTTON_WEED_CLASSES
else:
    if crop_choice == "Maize (Corn)":
        st.header("Maize Disease Detection")
        model_key = "Maize"
        current_classes = CORN_CLASSES
    else:
        st.header("Rice Disease Detection")
        model_key = "Rice"
        current_classes = RICE_CLASSES

user_location = st.sidebar.text_input("Location", placeholder="e.g. Hyderabad")
enable_ai = st.sidebar.checkbox("Enable AI Advice", value=True)

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
    
    with col2:
        weather_data = get_weather(user_location)
        if weather_data:
            # PURE CODE WEATHER WIDGET
            st.markdown(f"""
            <div style="background-color:#ffffff; border:1px solid #e0e0e0; padding:15px; border-radius:10px; display:flex; align-items:center; justify-content:space-around; margin-bottom:20px;">
                <div style="text-align:center;">
                    {weather_data['visual']}
                    <div style="font-size:12px; color:#666; margin-top:5px;">{weather_data['condition']}</div>
                </div>
                <div style="text-align:center;">
                    <h2 style="margin:0; color:#333;">{weather_data['temp']}</h2>
                    <div style="font-size:12px; color:#666;">Humidity: {weather_data['humidity']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        if st.button('Run Analysis'):
            with st.spinner('Analyzing...'):
                try:
                    interpreter = load_model(model_key)
                    quad_results = analyze_quadrants(image, interpreter, current_classes, model_key)
                    
                    st.write("### Quadrant Analysis")
                    q1, q2 = st.columns(2)
                    detections = []
                    
                    for i, (name, res) in enumerate(quad_results.items()):
                        target_col = q1 if i % 2 == 0 else q2
                        with target_col:
                            st.image(res['img'], width=150)
                            
                            label = res['label']
                            conf = res['conf']
                            
                            # Determine Color Code
                            color = "#dc3545" # Red
                            if "Healthy" in label: color = "#28a745" # Green
                            if "Weed" in label or label in COTTON_WEED_CLASSES: color = "#fd7e14" # Orange
                            
                            # PURE CODE CONFIDENCE BAR (CSS)
                            st.markdown(f"""
                                <div style="margin-bottom: 15px; background:#f9f9f9; padding:5px; border-radius:5px;">
                                    <div style="font-weight:bold; color:{color}; font-size:14px; margin-bottom:2px;">
                                        {label}
                                    </div>
                                    <div style="width: 100%; background-color: #e0e0e0; border-radius: 4px; height: 10px;">
                                        <div style="width: {conf}%; background-color: {color}; height: 10px; border-radius: 4px;"></div>
                                    </div>
                                    <div style="font-size:11px; color:#666; text-align:right;">
                                        Confidence: {conf:.1f}%
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            if conf > 50: detections.append(label)
                    
                    if enable_ai:
                        st.markdown("---")
                        final_issue = max(set(detections), key=detections.count) if detections else "Healthy/Unknown"
                        
                        st.success(f"Primary Diagnosis: {final_issue}")
                        
                        advice = get_smart_advice(final_issue, weather_data, user_location, crop_choice)
                        st.info(advice)
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
